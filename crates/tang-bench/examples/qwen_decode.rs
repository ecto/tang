//! Qwen3 decode through tang's tensor IR on CUDA: one graph per step
//! (`tang_bench::ir_qwen`), lowered by `to_cuda_decode`, run through
//! `tang_gpu::program::Program` with the KV caches updated in place and
//! the step captured as a CUDA graph.
//!
//! ```text
//! qwen_decode tiny                      random small model: GPU vs the interpreter
//! qwen_decode model <dir> <ids> [opts]  MLX 4-bit checkpoint
//!     --gen N        greedy tokens after the prompt (default 64)
//!     --ctx C        KV capacity (default 4352)
//!     --logits FILE  write every step's logits (f32) for a reference check
//!     --out FILE     write the generated ids
//!     --profile      per-kernel times
//!     --bench L      time the step at context lengths L (comma list)
//! ```
//!
//! `<ids>` is a file of whitespace-separated token ids (the prompt).
#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("qwen_decode needs --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("tiny") => cuda::tiny(),
        Some("model") => cuda::model(&args[1..]),
        _ => eprintln!("usage: qwen_decode tiny | model <dir> <ids> [opts]"),
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use std::collections::HashMap;
    use std::time::Instant;

    use tang_bench::ir_qwen::*;
    use tang_compile::tensor::cuda_gemv::Packing;
    use tang_compile::tensor::wgsl::WgslProgram;
    use tang_compile::tensor::TensorId;
    use tang_gpu::program::{BufferPlan, Program, Runtime};

    /// f32 → bf16 → f32, round to nearest even.
    fn bf16_round(x: f32) -> f32 {
        let b = x.to_bits();
        let r = b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000;
        f32::from_bits(r)
    }

    /// bf16 values (already representable) packed two per word.
    fn pack_bf16(v: &[f32]) -> Vec<f32> {
        v.chunks(2)
            .map(|c| {
                let lo = c[0].to_bits() >> 16;
                let hi = c.get(1).map_or(0, |x| x.to_bits() >> 16);
                f32::from_bits(lo | (hi << 16))
            })
            .collect()
    }

    /// Nibbles 0..=15 packed eight per word, low first.
    fn pack_u4(v: &[f32]) -> Vec<f32> {
        v.chunks(8)
            .map(|c| {
                let w = c
                    .iter()
                    .enumerate()
                    .fold(0u32, |w, (i, &x)| w | ((x as u32) << (4 * i)));
                f32::from_bits(w)
            })
            .collect()
    }

    /// One slot per buffer, except in-place outputs, which share their
    /// donor input's slot.
    fn alias_plan(program: &WgslProgram, inplace: &[(TensorId, usize)]) -> BufferPlan {
        let n = program.buffers.len();
        let buf_of = |t: TensorId| {
            let t = program.storage(t);
            program.buffers.iter().position(|b| b.node == t).unwrap()
        };
        let input_buf = |k: usize| {
            program
                .buffers
                .iter()
                .position(|b| b.input == Some(k))
                .unwrap()
        };
        let mut owner: Vec<usize> = (0..n).collect();
        for &(out, k) in inplace {
            owner[buf_of(out)] = input_buf(k);
        }
        let mut slot_of_owner = HashMap::new();
        let mut slot = vec![0; n];
        let mut slot_len: Vec<usize> = Vec::new();
        for i in 0..n {
            let o = owner[i];
            let s = *slot_of_owner.entry(o).or_insert_with(|| {
                slot_len.push(0);
                slot_len.len() - 1
            });
            slot[i] = s;
            let words = program.buffers[i].len.max(1);
            slot_len[s] = slot_len[s].max(words);
        }
        BufferPlan {
            slot,
            slot_len,
            order: (0..program.kernels.len()).collect(),
        }
    }

    struct Session {
        rt: Runtime,
        prog: Program,
        dims: QwenDims,
        step: TensorId,
        logits: TensorId,
        native: Vec<bool>,
    }

    impl Session {
        fn new(dims: &QwenDims, inputs: &[Vec<f32>], dg: &DecodeGraph) -> Self {
            let t = Instant::now();
            let (g, outs) = compiled_decode(dg);
            let opts = decode_opts(dg, &outs, dims.layers);
            let dp = g.to_cuda_decode(&outs, &opts);
            let native = dp.native.clone();
            let program = dp.program;
            eprintln!(
                "lowered: {} kernels ({} native) in {:.2}s",
                program.kernels.len(),
                native.iter().filter(|n| **n).count(),
                t.elapsed().as_secs_f64()
            );
            let plan = alias_plan(&program, &opts.inplace);
            let rt = Runtime::cuda_capturable().expect("cuda");
            let refs: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();
            let t = Instant::now();
            let mut prog = Program::with_plan(&rt, program, &refs, &plan);
            eprintln!("compiled + uploaded in {:.2}s", t.elapsed().as_secs_f64());
            prog.run(&rt);
            rt.sync();
            prog.capture(&rt);
            let step = prog.input(0);
            Self {
                rt,
                prog,
                dims: dims.clone(),
                step,
                logits: outs[0],
                native,
            }
        }

        /// One decode step: `x` the token's embedding at position `pos`.
        fn step(&self, x: &[f32], pos: usize) {
            let (cos, sin) = rope_angles(&self.dims, pos);
            let mut s = Vec::with_capacity(self.dims.step_len());
            s.extend_from_slice(x);
            s.extend_from_slice(&cos);
            s.extend_from_slice(&sin);
            s.push((pos + 1) as f32);
            self.prog.write(&self.rt, self.step, &s);
            self.prog.run(&self.rt);
        }

        fn logits(&self) -> Vec<f32> {
            self.prog.read(&self.rt, self.logits)
        }
    }

    fn argmax(v: &[f32]) -> usize {
        let mut best = 0;
        for (i, &x) in v.iter().enumerate() {
            if x > v[best] {
                best = i;
            }
        }
        best
    }

    /// Random small model: every step's logits and caches against the
    /// interpreter (f64) on the same logical inputs.
    pub fn tiny() {
        let dims = QwenDims {
            d_model: 256,
            heads: 4,
            kv_heads: 2,
            head_dim: 64,
            ff: 384,
            vocab: 320,
            layers: 2,
            ctx: 24,
            eps: 1e-6,
            theta: 1e4,
        };
        let dg = decode_graph(&dims);
        let mut rng = tang_bench::Rng::new(11);
        let mut logical: Vec<Vec<f32>> = Vec::new();
        for s in &dg.inputs {
            let n: usize = s.shape.iter().product();
            let v: Vec<f32> = match s.packing {
                Some(Packing::U4) => (0..n)
                    .map(|_| ((rng.f32() + 1.0) * 7.999).floor())
                    .collect(),
                Some(Packing::Bf16) if s.name.ends_with(".s") => (0..n)
                    .map(|_| bf16_round(0.01 + 0.01 * rng.f32()))
                    .collect(),
                Some(Packing::Bf16) => (0..n)
                    .map(|_| bf16_round(-0.08 * (1.0 + 0.1 * rng.f32())))
                    .collect(),
                None if s.name.ends_with("cache") || s.name == "step" => vec![0.0; n],
                None => (0..n).map(|_| 1.0 + 0.2 * rng.f32()).collect(),
            };
            logical.push(v);
        }
        let device: Vec<Vec<f32>> = dg
            .inputs
            .iter()
            .zip(&logical)
            .map(|(s, v)| match s.packing {
                Some(Packing::U4) => pack_u4(v),
                Some(Packing::Bf16) => pack_bf16(v),
                None => v.clone(),
            })
            .collect();
        let sess = Session::new(&dims, &device, &dg);
        let mut cpu: Vec<Vec<f64>> = logical
            .iter()
            .map(|v| v.iter().map(|&x| x as f64).collect())
            .collect();
        let mut worst: f64 = 0.0;
        for pos in 0..dims.ctx {
            let x: Vec<f32> = (0..dims.d_model).map(|_| rng.f32()).collect();
            sess.step(&x, pos);
            let got = sess.logits();
            let (cos, sin) = rope_angles(&dims, pos);
            let mut s: Vec<f64> = x.iter().map(|&v| v as f64).collect();
            s.extend(cos.iter().chain(&sin).map(|&v| v as f64));
            s.push((pos + 1) as f64);
            cpu[0] = s;
            let res = dg.g.eval(&cpu, &dg.outputs);
            for l in 0..dims.layers {
                let base = 1 + l * LAYER_INPUTS.len();
                cpu[base + 19] = res[1 + 2 * l].clone();
                cpu[base + 20] = res[2 + 2 * l].clone();
            }
            let want = &res[0];
            let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let err = got
                .iter()
                .zip(want)
                .fold(0.0f64, |m, (&a, &b)| m.max((a as f64 - b).abs()));
            worst = worst.max(err / scale);
            if pos % 6 == 0 || pos + 1 == dims.ctx {
                println!(
                    "pos {pos:2}: max |logit err| {err:.2e} (max |logit| {scale:.2}), argmax gpu {} cpu {}",
                    argmax(&got),
                    argmax(&want.iter().map(|&v| v as f32).collect::<Vec<_>>())
                );
            }
        }
        println!("tiny: worst relative logit error {worst:.2e}");
        assert!(worst < 1e-4, "GPU decode disagrees with the interpreter");
    }

    // ------------------------------------------------------------------
    // MLX checkpoint

    struct St {
        bytes: Vec<u8>,
        base: usize,
        index: HashMap<String, (String, Vec<usize>, usize, usize)>,
    }

    impl St {
        fn open(path: &str) -> Self {
            let bytes = std::fs::read(path).expect("read safetensors");
            let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
            let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
            let mut index = HashMap::new();
            for (k, v) in header.as_object().unwrap() {
                if k == "__metadata__" {
                    continue;
                }
                let dtype = v["dtype"].as_str().unwrap().to_string();
                let shape = v["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_u64().unwrap() as usize)
                    .collect();
                let o = v["data_offsets"].as_array().unwrap();
                let (a, b) = (
                    o[0].as_u64().unwrap() as usize,
                    o[1].as_u64().unwrap() as usize,
                );
                index.insert(k.clone(), (dtype, shape, a, b));
            }
            Self {
                bytes,
                base: 8 + n,
                index,
            }
        }

        fn raw(&self, name: &str) -> &[u8] {
            let (_, _, a, b) = self
                .index
                .get(name)
                .unwrap_or_else(|| panic!("missing tensor {name}"));
            &self.bytes[self.base + a..self.base + b]
        }

        /// 32-bit words of a tensor's bytes (u32 or packed bf16 pairs).
        fn words(&self, name: &str) -> Vec<f32> {
            self.raw(name)
                .chunks(4)
                .map(|c| f32::from_bits(u32::from_le_bytes(c.try_into().unwrap())))
                .collect()
        }

        fn bf16(&self, name: &str) -> Vec<f32> {
            self.raw(name)
                .chunks(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect()
        }
    }

    /// Device inputs for the decode graph from an MLX checkpoint.
    fn load_inputs(st: &St, dg: &DecodeGraph) -> Vec<Vec<f32>> {
        let proj = |l: usize, p: &str| -> &'static [&'static str] {
            let _ = l;
            match p {
                "qkv" => &["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                "o" => &["self_attn.o_proj"],
                "gate" => &["mlp.gate_proj"],
                "up" => &["mlp.up_proj"],
                "down" => &["mlp.down_proj"],
                _ => unreachable!(),
            }
        };
        let suffix = |s: &str| match s {
            "q" => "weight",
            "s" => "scales",
            "b" => "biases",
            _ => unreachable!(),
        };
        dg.inputs
            .iter()
            .map(|s| {
                let n: usize = s.shape.iter().product();
                let name = &s.name;
                if name == "step" || name.ends_with("cache") {
                    return vec![0.0; n];
                }
                if name == "model.norm" {
                    return st.bf16("model.norm.weight");
                }
                if let Some(rest) = name.strip_prefix("lm_head.") {
                    return st.words(&format!("lm_head.{}", suffix(rest)));
                }
                // model.layers.{l}.{what}
                let rest = name.strip_prefix("model.layers.").unwrap();
                let (l, what) = rest.split_once('.').unwrap();
                let l: usize = l.parse().unwrap();
                let pre = format!("model.layers.{l}.");
                match what {
                    "ln1" => st.bf16(&format!("{pre}input_layernorm.weight")),
                    "ln2" => st.bf16(&format!("{pre}post_attention_layernorm.weight")),
                    "q_norm" => st.bf16(&format!("{pre}self_attn.q_norm.weight")),
                    "k_norm" => st.bf16(&format!("{pre}self_attn.k_norm.weight")),
                    _ => {
                        let (p, part) = what.split_once('.').unwrap();
                        // row-major [rows, …]: concatenating rows is
                        // concatenating the word arrays
                        let mut v = Vec::new();
                        for t in proj(l, p) {
                            v.extend(st.words(&format!("{pre}{t}.{}", suffix(part))));
                        }
                        let want = s.packing.unwrap().words(n);
                        assert_eq!(v.len(), want, "{name}");
                        v
                    }
                }
            })
            .collect()
    }

    /// The embedding row of `tok`, dequantized.
    fn embed(st: &St, tok: usize, d: usize) -> Vec<f32> {
        let w = st.raw("model.embed_tokens.weight");
        let s = st.raw("model.embed_tokens.scales");
        let b = st.raw("model.embed_tokens.biases");
        let groups = d / GROUP;
        let bf = |raw: &[u8], i: usize| {
            f32::from_bits((u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]) as u32) << 16)
        };
        (0..d)
            .map(|k| {
                let wi = tok * (d / 8) + k / 8;
                let word = u32::from_le_bytes(w[4 * wi..4 * wi + 4].try_into().unwrap());
                let q = ((word >> (4 * (k % 8))) & 15) as f32;
                let g = tok * groups + k / GROUP;
                q * bf(s, g) + bf(b, g)
            })
            .collect()
    }

    pub fn model(args: &[String]) {
        let dir = &args[0];
        let ids: Vec<usize> = std::fs::read_to_string(&args[1])
            .unwrap()
            .split_whitespace()
            .map(|t| t.parse().unwrap())
            .collect();
        let opt = |k: &str| {
            args.iter()
                .position(|a| a == k)
                .map(|i| args[i + 1].clone())
        };
        let gen: usize = opt("--gen").map_or(64, |v| v.parse().unwrap());
        let ctx: usize = opt("--ctx").map_or(4352, |v| v.parse().unwrap());
        let dims = QwenDims::qwen3_8b(ctx);
        let t = Instant::now();
        let st = St::open(&format!("{dir}/model.safetensors"));
        let dg = decode_graph(&dims);
        let inputs = load_inputs(&st, &dg);
        eprintln!("weights read in {:.1}s", t.elapsed().as_secs_f64());
        let sess = Session::new(&dims, &inputs, &dg);
        drop(inputs);

        // prompt: one step per token (decode-shaped prefill)
        let mut logits_out: Vec<f32> = Vec::new();
        let keep = opt("--logits").is_some();
        let t = Instant::now();
        for (pos, &tok) in ids.iter().enumerate() {
            sess.step(&embed(&st, tok, dims.d_model), pos);
            if keep && pos + 1 == ids.len() {
                logits_out.extend(sess.logits());
            }
        }
        sess.rt.sync();
        let prefill_s = t.elapsed().as_secs_f64();
        let mut out = Vec::new();
        let mut tok = argmax(&sess.logits());
        let t = Instant::now();
        for i in 0..gen {
            out.push(tok);
            let pos = ids.len() + i;
            if pos >= ctx {
                break;
            }
            sess.step(&embed(&st, tok, dims.d_model), pos);
            let l = sess.logits();
            if keep {
                logits_out.extend_from_slice(&l);
            }
            tok = argmax(&l);
        }
        let dec_s = t.elapsed().as_secs_f64();
        println!(
            "prompt {} tokens in {:.3}s ({:.1} tok/s, one step per token)",
            ids.len(),
            prefill_s,
            ids.len() as f64 / prefill_s
        );
        println!(
            "decode {} tokens in {:.3}s: {:.1} tok/s (host loop: embed, write, graph, read logits)",
            out.len(),
            dec_s,
            out.len() as f64 / dec_s
        );
        println!(
            "ids: {}",
            out.iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        );
        if let Some(p) = opt("--out") {
            let s: Vec<String> = out.iter().map(|t| t.to_string()).collect();
            std::fs::write(p, s.join(" ")).unwrap();
        }
        if let Some(p) = opt("--logits") {
            let bytes: Vec<u8> = logits_out.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(p, bytes).unwrap();
        }
        // device time of one step at a few context lengths
        if let Some(ls) = opt("--bench") {
            let x = embed(&st, ids[0], dims.d_model);
            for l in ls.split(',').map(|v| v.parse::<usize>().unwrap()) {
                let pos = l.min(ctx) - 1;
                sess.step(&x, pos);
                sess.rt.sync();
                let ms = sess.prog.time(&sess.rt, 20);
                println!("step at len {l}: {ms:.3} ms device ({:.1} tok/s)", 1e3 / ms);
            }
        }
        if args.iter().any(|a| a == "--profile") {
            profile(&sess);
        }
    }

    /// Per-kernel device time of one step, grouped by kernel kind.
    fn profile(sess: &Session) {
        let p = &sess.prog;
        let mut groups: Vec<(String, f64, usize)> = Vec::new();
        let mut total = 0.0;
        for (i, k) in p.program.kernels.iter().enumerate() {
            let ms = p.time_kernel(&sess.rt, i, 3);
            total += ms;
            let head = k
                .source
                .lines()
                .find(|l| l.starts_with("// ") && !l.contains("Auto-generated"))
                .unwrap_or("")
                .to_string();
            let kind = if !sess.native[i] {
                format!("transpiled map, {} points", k.points)
            } else if k.source.contains("in-place") {
                "cache row update".to_string()
            } else {
                head.trim_start_matches("// ").to_string()
            };
            match groups.iter_mut().find(|g| g.0 == kind) {
                Some(g) => {
                    g.1 += ms;
                    g.2 += 1
                }
                None => groups.push((kind, ms, 1)),
            }
        }
        groups.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        println!("per-kernel (sum of isolated launches {total:.3} ms):");
        for (k, ms, n) in groups {
            println!("  {ms:8.3} ms  {n:4}x  {k}");
        }
    }
}
