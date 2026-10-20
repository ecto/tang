//! Qwen3 decode on a GPU through the tensor IR, as a library: the step
//! programs (`Step`: decode, or a prefill of `seq` tokens), the MLX 4-bit
//! checkpoint reader (`St`) and the device-input conversion, moved from
//! `examples/qwen_decode.rs`; and [`Session`], a chat-turn API over them
//! that keeps the KV cache across turns (prefix reuse: the system prompt is
//! prefilled once).
use std::collections::HashMap;
use std::time::Instant;

use crate::ir_qwen::*;
use tang_compile::tensor::cuda_gemv::DecodeProgram;
use tang_compile::tensor::wgsl::WgslProgram;
use tang_compile::tensor::{TensorGraph, TensorId};
use tang_gpu::program::{BackendKind, BufferPlan, Program, Runtime, Shared};

/// CUDA when built with it and a device is present, else wgpu (Metal:
/// the MSL decode kernels).
pub fn runtime() -> Runtime {
    // TANG_BACKEND=opencl: the OpenCL decode kernels (the K1's Adreno)
    #[cfg(feature = "opencl")]
    if std::env::var("TANG_BACKEND").as_deref() == Ok("opencl") {
        return Runtime::opencl().expect("no OpenCL GPU");
    }
    #[cfg(feature = "cuda")]
    if let Some(rt) = Runtime::cuda_capturable() {
        return rt;
    }
    Runtime::wgpu().expect("no GPU")
}

pub fn lower(
    rt: &Runtime,
    g: &TensorGraph,
    outs: &[TensorId],
    opts: &tang_compile::tensor::cuda_gemv::DecodeOpts,
) -> DecodeProgram {
    match rt {
        // Vulkan (the K1's Adreno, ...): the decode kernels as GLSL,
        // compiled to SPIR-V by tang-gpu (feature `glsl`)
        Runtime::Wgpu { device, .. } if !device.msl_passthrough && device.spirv_passthrough => {
            g.to_glsl_decode(outs, opts)
        }
        _ => match rt.kind() {
            BackendKind::Cuda => g.to_cuda_decode(outs, opts),
            BackendKind::Wgpu => g.to_msl_decode(outs, opts),
            BackendKind::OpenCl => g.to_opencl_decode(outs, opts),
        },
    }
}

pub fn env(k: &str) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

/// The program's buffers on slots (`ir_qwen::plan_slots`): in-place
/// outputs share their donor's, intermediates are reused by liveness.
/// The program's buffers on slots (`ir_qwen::plan_slots`): in-place
/// outputs share their donor's, intermediates are reused by liveness.
pub fn slot_plan(
    program: &WgslProgram,
    outputs: &[TensorId],
    inplace: &[(TensorId, usize)],
) -> BufferPlan {
    let (slot, slot_len) = plan_slots(program, outputs, inplace);
    BufferPlan {
        slot,
        slot_len,
        order: (0..program.kernels.len()).collect(),
    }
}

/// One step program: decode (`seq == 1`) or a prefill of `seq` tokens.
/// One step program: decode (`seq == 1`) or a prefill of `seq` tokens.
pub struct Step {
    pub dg: DecodeGraph,
    pub prog: Program,
    /// `dg.outputs` in the compiled graph, then `dg.keep`; with
    /// `QWEN_KEEP_ALL`, every kernel's result after those (from `named`).
    pub outs: Vec<TensorId>,
    /// How many of `outs` are the step's own outputs.
    pub named: usize,
    /// The compiled graph `outs` index (for checking kernels one by one).
    pub g: TensorGraph,
    pub native: Vec<bool>,
    /// Bytes this program allocated (shared buffers count for their owner).
    pub bytes: usize,
}

impl Step {
    /// Lower and allocate the step; inputs from `inputs` (device words,
    /// per graph input), or — every input named like one of `base`'s
    /// — bound to `base`'s buffers.
    pub fn new(
        rt: &Runtime,
        dims: &QwenDims,
        seq: usize,
        kctx: usize,
        inputs: &dyn Fn(&DecodeGraph) -> Vec<Vec<f32>>,
        base: Option<&Step>,
    ) -> Self {
        let t = Instant::now();
        let dg = step_graph(dims, seq, kctx);
        let (g, mut outs) = compiled_decode(&dg, rt.kind() == BackendKind::Cuda);
        let named = outs.len();
        let options = |outs: &[TensorId]| {
            let mut opts = decode_opts(&dg, outs);
            opts.rows = env("TANG_DECODE_ROWS");
            opts.warps = env("TANG_DECODE_WARPS");
            opts.unroll = env("TANG_DECODE_UNROLL");
            opts.mma_shape = env("TANG_MMA_SHAPE");
            opts
        };
        let mut opts = options(&outs);
        let mut dp = lower(rt, &g, &outs, &opts);
        // QWEN_KEEP_ALL=POS: every kernel's result is an output too, so a
        // test can compare each against the interpreter (debugging aid;
        // it changes fusion where a result had been inlined)
        if std::env::var_os("QWEN_KEEP_ALL").is_some() {
            for k in &dp.program.kernels {
                if !outs.contains(&k.node) {
                    outs.push(k.node);
                }
            }
            opts = options(&outs);
            dp = lower(rt, &g, &outs, &opts);
        }
        let native = dp.native.clone();
        let mut program = dp.program;
        if seq > 1 {
            lazy_order(&mut program);
        }
        if let Ok(dir) = std::env::var("QWEN_DUMP") {
            for (i, k) in program.kernels.iter().enumerate() {
                let ext = if k.source.starts_with("#include") {
                    "metal"
                } else if rt.kind() == BackendKind::Cuda {
                    "cu"
                } else if rt.kind() == BackendKind::OpenCl {
                    "cl"
                } else {
                    "wgsl"
                };
                std::fs::write(format!("{dir}/s{seq}_k{i:03}.{ext}"), &k.source).unwrap();
            }
        }
        let plan = slot_plan(&program, &outs, &opts.inplace);
        let lowered = t.elapsed().as_secs_f64();
        let shared: Vec<Shared> = match base {
            Some(b) => dg
                .inputs
                .iter()
                .enumerate()
                .filter_map(|(k, s)| {
                    let j = b.dg.inputs.iter().position(|bs| bs.name == s.name)?;
                    Some(Shared {
                        input: k,
                        from: &b.prog,
                        tensor: b.prog.input(j),
                    })
                })
                .collect(),
            None => Vec::new(),
        };
        // own inputs left empty (a prefill's chunk) start zeroed
        let words = inputs(&dg);
        let refs: Vec<&[f32]> = words.iter().map(|v| v.as_slice()).collect();
        let owned: usize = {
            let mut borrowed = vec![false; plan.slot_len.len()];
            for sh in &shared {
                let i = program
                    .buffers
                    .iter()
                    .position(|b| b.input == Some(sh.input))
                    .unwrap();
                borrowed[plan.slot[i]] = true;
            }
            plan.slot_len
                .iter()
                .zip(&borrowed)
                .filter(|(_, b)| !**b)
                .map(|(n, _)| 4 * n)
                .sum()
        };
        let t = Instant::now();
        let mut prog = Program::with_shared(rt, program, &refs, &plan, &shared);
        drop(words);
        prog.capture(rt);
        eprintln!(
            "step seq {seq} (window {kctx}): {} kernels ({} native), lowered in {lowered:.2}s, \
             compiled + bound in {:.2}s, {:.0} MB allocated ({} inputs shared)",
            prog.program.kernels.len(),
            native.iter().filter(|n| **n).count(),
            t.elapsed().as_secs_f64(),
            owned as f64 / 1e6,
            shared.len()
        );
        Self {
            dg,
            prog,
            outs,
            named,
            g,
            native,
            bytes: owned,
        }
    }

    pub fn tensor(&self, name: &str) -> TensorId {
        self.prog.input(self.dg.input(name))
    }

    /// `(next token, its position)`.
    pub fn state(&self, rt: &Runtime) -> (usize, usize) {
        let s = self.prog.read(rt, self.outs[1]);
        (s[0] as usize, s[1] as usize)
    }

    pub fn set_state(&self, rt: &Runtime, tok: usize, pos: usize) {
        self.prog
            .write(rt, self.tensor("state"), &[tok as f32, pos as f32]);
    }

    /// The full logits: block 0, then the other vocabulary blocks.
    pub fn logits(&self, rt: &Runtime) -> Vec<f32> {
        let mut l = self.prog.read(rt, self.outs[0]);
        let first = self.dg.outputs.len() + self.dg.keep.len();
        for i in 0..self.dg.logit_chunks.len() {
            l.extend(self.prog.read(rt, self.outs[first + i]));
        }
        l
    }

    /// Prefill: `tokens` at positions `p0..`, then `state` holds the next
    /// token and its position.
    pub fn prefill(&self, rt: &Runtime, tokens: &[usize], p0: usize) {
        assert_eq!(tokens.len(), self.dg.seq);
        self.prog
            .write(rt, self.tensor("chunk"), &chunk_input(tokens, p0));
        self.prog.run(rt);
    }
}

pub fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

/// `QWEN_LOAD_DUTY` (0..1, unset = 1): the fraction of wall time the
/// checkpoint conversion may keep a core busy. A robot without
/// cooling trips its thermal limit while loading at full speed.
pub fn load_duty() -> f64 {
    std::env::var("QWEN_LOAD_DUTY")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .map_or(1.0, |d| d.clamp(0.01, 1.0))
}

/// Sleep so the work since `start` was `load_duty()` of the time.
/// Sleep so the work since `start` was `load_duty()` of the time.
pub fn pace(start: Instant) {
    let d = load_duty();
    if d < 1.0 {
        std::thread::sleep(start.elapsed().mul_f64(1.0 / d - 1.0));
    }
}

/// Worst relative error of `got` against `want`.

pub struct St {
    // mapped, not read: on the 10 GB robot a read copy plus the converted
    // words overflow RAM into zram, whose compression heats every core
    pub bytes: memmap2::Mmap,
    pub base: usize,
    pub index: HashMap<String, (String, Vec<usize>, usize, usize)>,
}

impl St {
    pub fn open(path: &str) -> Self {
        let f = std::fs::File::open(path).expect("open safetensors");
        let bytes = unsafe { memmap2::Mmap::map(&f) }.expect("map safetensors");
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

    pub fn raw(&self, name: &str) -> &[u8] {
        let (_, _, a, b) = self
            .index
            .get(name)
            .or_else(|| self.index.get(&tied(name)))
            .unwrap_or_else(|| panic!("missing tensor {name}"));
        &self.bytes[self.base + a..self.base + b]
    }

    /// 32-bit words of a tensor's bytes (u32 or packed bf16 pairs).
    pub fn words(&self, name: &str) -> Vec<f32> {
        self.raw(name)
            .chunks(4)
            .map(|c| f32::from_bits(u32::from_le_bytes(c.try_into().unwrap())))
            .collect()
    }

    pub fn bf16(&self, name: &str) -> Vec<f32> {
        self.raw(name)
            .chunks(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect()
    }

    /// A 4-bit matrix `[rows, cols]` dequantized exactly (f32).
    pub fn dequant(&self, stem: &str) -> (Vec<f32>, usize, usize) {
        let q = self.raw(&format!("{stem}.weight"));
        let s = self.bf16(&format!("{stem}.scales"));
        let b = self.bf16(&format!("{stem}.biases"));
        let w = format!("{stem}.weight");
        let (_, shape, _, _) = self.index.get(&w).unwrap_or_else(|| &self.index[&tied(&w)]);
        let (rows, cols) = (shape[0], shape[1] * 8);
        let groups = cols / GROUP;
        let mut w = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for k in 0..cols {
                let wi = r * (cols / 8) + k / 8;
                let word = u32::from_le_bytes(q[4 * wi..4 * wi + 4].try_into().unwrap());
                let n = ((word >> (4 * (k % 8))) & 15) as f32;
                let g = r * groups + k / GROUP;
                w[r * cols + k] = n * s[g] + b[g];
            }
        }
        (w, rows, cols)
    }
}

/// Tied embeddings (`tie_word_embeddings`, e.g. Qwen3-0.6B/1.7B/4B): no
/// `lm_head.*` tensors, the LM head is the embedding.
/// Tied embeddings (`tie_word_embeddings`, e.g. Qwen3-0.6B/1.7B/4B): no
/// `lm_head.*` tensors, the LM head is the embedding.
pub fn tied(name: &str) -> String {
    match name.strip_prefix("lm_head.") {
        Some(rest) => format!("model.embed_tokens.{rest}"),
        None => name.to_string(),
    }
}

/// The model's dims from `DIR/config.json` (any Qwen3 size), with a cache
/// of `ctx` positions; Qwen3-8B's when there is no config.
/// The model's dims from `DIR/config.json` (any Qwen3 size), with a cache
/// of `ctx` positions; Qwen3-8B's when there is no config.
pub fn dims_for(dir: &str, ctx: usize) -> QwenDims {
    let mut d = QwenDims::qwen3_8b(ctx);
    let Ok(text) = std::fs::read_to_string(format!("{dir}/config.json")) else {
        return d;
    };
    let c: serde_json::Value = serde_json::from_str(&text).expect("config.json");
    let u = |k: &str, v: &mut usize| {
        if let Some(x) = c[k].as_u64() {
            *v = x as usize;
        }
    };
    u("hidden_size", &mut d.d_model);
    u("num_attention_heads", &mut d.heads);
    u("num_key_value_heads", &mut d.kv_heads);
    u("head_dim", &mut d.head_dim);
    u("intermediate_size", &mut d.ff);
    u("vocab_size", &mut d.vocab);
    u("num_hidden_layers", &mut d.layers);
    if let Some(x) = c["rms_norm_eps"].as_f64() {
        d.eps = x;
    }
    if let Some(x) = c["rope_theta"].as_f64() {
        d.theta = x;
    }
    d
}

/// Device inputs for a step graph from an MLX checkpoint, converted as the
/// iterator is drawn (empty for state, which starts zeroed).
/// Device inputs for a step graph from an MLX checkpoint, converted as the
/// iterator is drawn (empty for state, which starts zeroed).
pub fn load_inputs<'a>(
    st: &'a St,
    dims: &'a QwenDims,
    dg: &'a DecodeGraph,
) -> impl Iterator<Item = Vec<f32>> + 'a {
    let proj = |p: &str| -> &'static [&'static str] {
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
    let one = move |s: &InputSpec| -> Vec<f32> {
        let n: usize = s.shape.iter().product();
        let name = &s.name;
        if name == "state" || name == "chunk" || name.ends_with("cache") {
            return Vec::new();
        }
        if name == "rope" {
            return rope_table(dims);
        }
        if name == "model.norm" {
            return st.bf16("model.norm.weight");
        }
        // the embedding and LM head, whole (`embed.q`) or as row block
        // `c` of `dims.vocab_chunks` (`embed.2.q`): rows are contiguous
        // in the packed words
        let block = |base: &str, rest: &str| -> Vec<f32> {
            let (c, part) = match rest.split_once('.') {
                Some((c, part)) => (c.parse::<usize>().ok(), part),
                None => (None, rest),
            };
            let w = st.words(&format!("{base}.{}", suffix(part)));
            match c {
                None => w,
                Some(c) => {
                    let per = w.len() / dims.vocab_chunks;
                    w[c * per..(c + 1) * per].to_vec()
                }
            }
        };
        if let Some(rest) = name.strip_prefix("lm_head.") {
            return block("lm_head", rest);
        }
        if let Some(rest) = name.strip_prefix("embed.") {
            return block("model.embed_tokens", rest);
        }
        // model.layers.{l}.{what}
        let rest = name.strip_prefix("model.layers.").unwrap();
        let (l, what) = rest.split_once('.').unwrap();
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
                for t in proj(p) {
                    v.extend(st.words(&format!("{pre}{t}.{}", suffix(part))));
                }
                let want = s.packing.unwrap().words(n);
                assert_eq!(v.len(), want, "{name}");
                v
            }
        }
    };
    // QWEN_LOAD_DUTY: hold the conversion to a CPU duty cycle (a hot robot)
    dg.inputs.iter().map(move |s| {
        let t = Instant::now();
        let v = one(s);
        pace(t);
        v
    })
}

/// Prefill chunk sizes a [`Session`] builds programs for (largest first);
/// a remainder below the smallest goes through decode steps.
/// `QWEN_PREFILL_CHUNKS=64,8` (the default) overrides.
fn chunk_sizes() -> Vec<usize> {
    std::env::var("QWEN_PREFILL_CHUNKS")
        .ok()
        .map(|v| {
            v.split(',')
                .filter_map(|x| x.parse().ok())
                .collect::<Vec<usize>>()
        })
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| vec![64, 8])
}

/// The attention window of a prefill reaching position `end`: a power of
/// two from 512 (so a handful of programs serve every turn), at most `ctx`.
fn window(end: usize, ctx: usize) -> usize {
    end.next_power_of_two().max(512).min(ctx)
}

/// How a [`Session`] picks the next token.
#[derive(Clone, Copy, Debug)]
pub struct Sampling {
    /// 0: greedy (the device argmax, no logits readback).
    pub temperature: f32,
    pub top_p: f32,
    /// > 1 discourages tokens already in the context (HF convention).
    pub repetition_penalty: f32,
    /// OpenAI's: subtracted from a token's logit once if it is in the reply
    /// so far, and per occurrence.
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub seed: u64,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            seed: 0,
        }
    }
}

/// A loaded model and its KV cache, for chat turns. The cache holds the
/// tokens of the last prompt and reply ([`Session::cached`]); a new prompt
/// sharing a prefix with them (the system prompt, the conversation so far)
/// prefills only the rest.
pub struct Session {
    pub rt: Runtime,
    pub dims: QwenDims,
    pub dec: Step,
    prefill: HashMap<(usize, usize), Step>,
    pub tokenizer: tokenizers::Tokenizer,
    /// Tokens whose keys and values are in the cache, position order.
    cached: Vec<u32>,
    /// The token the next decode step consumes, and its position.
    next: (u32, usize),
    /// Where the current reply starts in `cached` (presence/frequency penalties).
    reply_start: usize,
    /// The prefill program that ran last (`None`: the decode step).
    last: Option<(usize, usize)>,
    rng: u64,
}

impl Session {
    /// Load `dir` (an MLX 4-bit Qwen3 checkpoint with `config.json` and
    /// `tokenizer.json`) on the default runtime ([`runtime`]: honours
    /// `TANG_BACKEND`), with a KV cache of `ctx` positions.
    pub fn open(dir: &str, ctx: usize) -> Result<Self, String> {
        let tokenizer = tokenizers::Tokenizer::from_file(format!("{dir}/tokenizer.json"))
            .map_err(|e| format!("tokenizer: {e}"))?;
        let mut dims = dims_for(dir, ctx);
        dims.kv_f16 = std::env::var("TANG_KV_F32").is_err();
        let rt = runtime();
        for (k, v) in apply_tuning(&rt, &dims) {
            eprintln!("tang: tuned {k}={v}");
        }
        dims.vocab_chunks = match env("QWEN_VOCAB_CHUNKS") {
            0 => match &rt {
                Runtime::Wgpu { device, .. } if !device.msl_passthrough => 4,
                _ if rt.kind() == BackendKind::OpenCl => {
                    (dims.vocab * dims.d_model / 2).div_ceil(255 << 20).max(1)
                }
                _ => 1,
            },
            n => n,
        };
        let st = St::open(&format!("{dir}/model.safetensors"));
        let dec = Step::new(
            &rt,
            &dims,
            1,
            ctx,
            &|dg| vec![Vec::new(); dg.inputs.len()],
            None,
        );
        for (k, v) in load_inputs(&st, &dims, &dec.dg).enumerate() {
            if !v.is_empty() {
                dec.prog.write(&rt, dec.prog.input(k), &v);
            }
        }
        Ok(Self {
            rt,
            dims,
            dec,
            prefill: HashMap::new(),
            tokenizer,
            cached: Vec::new(),
            next: (0, 0),
            last: None,
            reply_start: 0,
            rng: 0x9e37_79b9_7f4a_7c15,
        })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.tokenizer
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .unwrap_or_default()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        self.tokenizer.decode(ids, false).unwrap_or_default()
    }

    /// The single id of a special token such as `<|im_end|>`.
    pub fn token_id(&self, s: &str) -> Option<u32> {
        self.tokenizer.token_to_id(s)
    }

    /// Forget the cache (the next prompt prefills from position 0).
    pub fn reset(&mut self) {
        self.cached.clear();
    }

    pub fn cached(&self) -> &[u32] {
        &self.cached
    }

    /// Reply tokens that fit after a prompt of `n` tokens.
    pub fn room_for(&self, n: usize) -> usize {
        self.dims.ctx.saturating_sub(n + 1)
    }

    /// Positions still free in the cache.
    pub fn room(&self) -> usize {
        self.dims.ctx.saturating_sub(self.next.1)
    }

    /// Build the prefill programs a prompt of up to `end` positions uses
    /// (compile time off the first turn).
    pub fn warm(&mut self, end: usize) {
        for seq in chunk_sizes() {
            let mut w = 512;
            while w <= window(end, self.dims.ctx) {
                self.step_for(seq, w);
                w *= 2;
            }
        }
    }

    fn step_for(&mut self, seq: usize, kctx: usize) -> &Step {
        if !self.prefill.contains_key(&(seq, kctx)) {
            let s = Step::new(
                &self.rt,
                &self.dims,
                seq,
                kctx,
                &|dg| vec![Vec::new(); dg.inputs.len()],
                Some(&self.dec),
            );
            self.prefill.insert((seq, kctx), s);
        }
        &self.prefill[&(seq, kctx)]
    }

    /// Make the cache hold `tokens` (a whole prompt, from position 0) and
    /// pick the first reply token. Reuses the longest common prefix with
    /// what is cached; returns `(first token, positions reused)`.
    pub fn prompt(&mut self, tokens: &[u32], sampling: &Sampling) -> (u32, usize) {
        assert!(!tokens.is_empty(), "empty prompt");
        assert!(tokens.len() < self.dims.ctx, "prompt longer than the cache");
        let mut lcp = self
            .cached
            .iter()
            .zip(tokens)
            .take_while(|(a, b)| a == b)
            .count();
        // the last prompt token is always run: its logits pick the reply
        lcp = lcp.min(tokens.len() - 1);
        let mut p = lcp;
        let sizes = chunk_sizes();
        while p < tokens.len() {
            let left = tokens.len() - p;
            match sizes.iter().copied().find(|&s| s <= left) {
                Some(seq) => {
                    let kctx = window(p + seq, self.dims.ctx);
                    let ids: Vec<usize> = tokens[p..p + seq].iter().map(|&t| t as usize).collect();
                    let rt = &self.rt as *const Runtime;
                    let step = self.step_for(seq, kctx);
                    // SAFETY: `rt` outlives this call; step_for only borrows `prefill`
                    step.prefill(unsafe { &*rt }, &ids, p);
                    self.last = Some((seq, kctx));
                    p += seq;
                }
                None => {
                    self.dec.set_state(&self.rt, tokens[p] as usize, p);
                    self.dec.prog.run(&self.rt);
                    self.last = None;
                    p += 1;
                }
            }
        }
        self.cached = tokens.to_vec();
        self.reply_start = tokens.len();
        let first = self.choose(sampling);
        (first, lcp)
    }

    /// The reply token after `tok` (the one [`prompt`](Self::prompt) or the
    /// last `step` returned): runs one decode step.
    pub fn step(&mut self, sampling: &Sampling) -> u32 {
        let (tok, pos) = self.next;
        self.dec.set_state(&self.rt, tok as usize, pos);
        self.dec.prog.run(&self.rt);
        self.last = None;
        self.cached.push(tok);
        self.choose(sampling)
    }

    /// The next token from the state the last run left (greedy: the device
    /// argmax) or by sampling its logits; records it as `next`.
    fn choose(&mut self, s: &Sampling) -> u32 {
        let (argmax_tok, pos) = self.dec.state(&self.rt);
        let plain =
            s.repetition_penalty == 1.0 && s.presence_penalty == 0.0 && s.frequency_penalty == 0.0;
        let tok = if s.temperature <= 0.0 && plain {
            argmax_tok as u32
        } else {
            let mut l = self.last_logits();
            self.sample(&mut l, s)
        };
        self.next = (tok, pos);
        tok
    }

    /// Logits of the last position run, from the program that ran it (the
    /// state is shared by every step program, the logits are each one's own).
    pub fn last_logits(&self) -> Vec<f32> {
        match self.last {
            Some(k) => self.prefill[&k].logits(&self.rt),
            None => self.dec.logits(&self.rt),
        }
    }

    fn sample(&mut self, l: &mut [f32], s: &Sampling) -> u32 {
        if s.repetition_penalty != 1.0 {
            let mut seen = std::collections::HashSet::new();
            for &t in &self.cached {
                if seen.insert(t) {
                    let x = &mut l[t as usize];
                    *x = if *x > 0.0 {
                        *x / s.repetition_penalty
                    } else {
                        *x * s.repetition_penalty
                    };
                }
            }
        }
        if s.presence_penalty != 0.0 || s.frequency_penalty != 0.0 {
            let mut count: HashMap<u32, f32> = HashMap::new();
            for &t in &self.cached[self.reply_start.min(self.cached.len())..] {
                *count.entry(t).or_default() += 1.0;
            }
            for (t, n) in count {
                l[t as usize] -= s.presence_penalty + s.frequency_penalty * n;
            }
        }
        if s.temperature <= 0.0 {
            return argmax(l) as u32;
        }
        if self.rng == 0x9e37_79b9_7f4a_7c15 && s.seed != 0 {
            self.rng ^= s.seed;
        }
        let m = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut idx: Vec<(f32, u32)> = l
            .iter()
            .enumerate()
            .map(|(i, &x)| (((x - m) / s.temperature).exp(), i as u32))
            .collect();
        idx.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let total: f32 = idx.iter().map(|x| x.0).sum();
        let mut keep = 0.0;
        let mut n = 0;
        for x in &idx {
            keep += x.0;
            n += 1;
            if keep >= s.top_p * total {
                break;
            }
        }
        // xorshift64*
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        let u = (self.rng.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32 / (1u64 << 24) as f32;
        let mut acc = 0.0;
        for x in &idx[..n] {
            acc += x.0 / keep;
            if u < acc {
                return x.1;
            }
        }
        idx[n - 1].1
    }
}

// ---------------------------------------------------------------------------
// tuning: per device and model shape, measured on the device (`tune`), read at
// startup (`apply_tuning`)

/// The tuning file: `$TANG_TUNE`, else `~/.config/tang/tune.json`. A JSON
/// object keyed by [`tune_key`], each value the env knobs that won:
/// `{"QUALCOMM Adreno(TM) 740|qwen3 d2048 ff6144 l28": {"TANG_DECODE_WARPS": "4", …}}`.
pub fn tune_path() -> std::path::PathBuf {
    if let Some(p) = std::env::var_os("TANG_TUNE") {
        return p.into();
    }
    let home = std::env::var_os("HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_default();
    home.join(".config/tang/tune.json")
}

/// Device name and model shape: what a tuning result is valid for.
pub fn tune_key(rt: &Runtime, dims: &QwenDims) -> String {
    format!(
        "{}|qwen3 d{} ff{} l{} h{} kv{}",
        rt.name(),
        dims.d_model,
        dims.ff,
        dims.layers,
        dims.heads,
        dims.kv_heads
    )
}

/// The knobs the tuner sets (decode GEMV shape, OpenCL prefill register block).
pub const TUNE_KNOBS: [&str; 3] = ["TANG_DECODE_WARPS", "TANG_DECODE_ROWS", "TANG_CL_MM"];

/// Set this device and model's tuned knobs from [`tune_path`], except any
/// already set in the environment (an explicit env var wins). Returns what
/// it applied.
pub fn apply_tuning(rt: &Runtime, dims: &QwenDims) -> Vec<(String, String)> {
    let Ok(text) = std::fs::read_to_string(tune_path()) else {
        return Vec::new();
    };
    let Ok(all) = serde_json::from_str::<serde_json::Value>(&text) else {
        return Vec::new();
    };
    let mut applied = Vec::new();
    if let Some(m) = all[tune_key(rt, dims)].as_object() {
        for (k, v) in m {
            let Some(v) = v.as_str() else { continue };
            if TUNE_KNOBS.contains(&k.as_str()) && std::env::var_os(k).is_none() {
                std::env::set_var(k, v);
                applied.push((k.clone(), v.to_string()));
            }
        }
    }
    applied
}

/// Hottest GPU temperature in °C, if the board exposes one (kgsl on the K1).
fn gpu_temp() -> Option<f64> {
    let t: f64 = std::fs::read_to_string("/sys/class/kgsl/kgsl-3d0/temp")
        .ok()?
        .trim()
        .parse()
        .ok()?;
    Some(if t > 1000.0 { t / 1000.0 } else { t })
}

fn load1() -> Option<f64> {
    std::fs::read_to_string("/proc/loadavg")
        .ok()?
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

/// Wait (bounded) for the GPU to cool below `cool` °C and the 1-minute load
/// to fall below `load`; false when `deadline` passed first.
fn settle(hot: f64, cool: f64, load: f64, deadline: Instant) -> bool {
    let mut waited = false;
    loop {
        let t = gpu_temp().unwrap_or(0.0);
        let l = load1().unwrap_or(0.0);
        let too_hot = t >= hot || (waited && t > cool);
        if !too_hot && l < load {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
        if !waited {
            eprintln!("tune: waiting (gpu {t:.1} C, load {l:.2})");
        }
        waited = true;
        std::thread::sleep(std::time::Duration::from_secs(2));
    }
}

/// Measure, on this device, the decode GEMV shape (`TANG_DECODE_WARPS` ×
/// `TANG_DECODE_ROWS`) and, on OpenCL, the prefill register block
/// (`TANG_CL_MM`) for the model in `dir`, within `budget` seconds; write the
/// winners into [`tune_path`] under [`tune_key`]. Candidates share the first
/// program's weights and caches (one upload). Pauses at `TANG_TUNE_HOT` °C
/// (default 82) until 75, and while the load is above `TANG_TUNE_LOAD`
/// (default 12). Returns the winners.
pub fn tune(dir: &str, budget: f64) -> Vec<(String, String)> {
    let t0 = Instant::now();
    let deadline = t0 + std::time::Duration::from_secs_f64(budget);
    let fenv = |k: &str, d: f64| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let (hot, load) = (fenv("TANG_TUNE_HOT", 82.0), fenv("TANG_TUNE_LOAD", 12.0));
    for k in TUNE_KNOBS {
        std::env::remove_var(k);
    }
    let ctx = 1024;
    let mut dims = dims_for(dir, ctx);
    dims.kv_f16 = true;
    let rt = runtime();
    if rt.kind() == BackendKind::OpenCl {
        dims.vocab_chunks = (dims.vocab * dims.d_model / 2).div_ceil(255 << 20).max(1);
    }
    let key = tune_key(&rt, &dims);
    eprintln!("tune: {key}, budget {budget:.0} s");
    let st = St::open(&format!("{dir}/model.safetensors"));
    let base = Step::new(
        &rt,
        &dims,
        1,
        ctx,
        &|dg| vec![Vec::new(); dg.inputs.len()],
        None,
    );
    for (k, v) in load_inputs(&st, &dims, &base.dg).enumerate() {
        if !v.is_empty() {
            base.prog.write(&rt, base.prog.input(k), &v);
        }
    }
    let time_step = |s: &Step, pos: usize| -> f64 {
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            s.set_state(&rt, 1, pos);
            best = best.min(s.prog.time(&rt, 10));
        }
        best
    };
    let mut won = Vec::new();
    // decode: warps x rows
    let mut best: Option<(f64, usize, usize)> = None;
    for (w, r) in [(4, 2), (4, 4), (8, 4), (4, 8), (2, 4), (8, 2), (8, 8)] {
        if !settle(hot, 75.0, load, deadline) {
            eprintln!("tune: budget spent");
            break;
        }
        std::env::set_var("TANG_DECODE_WARPS", w.to_string());
        std::env::set_var("TANG_DECODE_ROWS", r.to_string());
        let s = Step::new(
            &rt,
            &dims,
            1,
            ctx,
            &|dg| vec![Vec::new(); dg.inputs.len()],
            Some(&base),
        );
        let ms = time_step(&s, 64);
        eprintln!(
            "tune: decode warps {w} rows {r}: {ms:.2} ms (gpu {:.1} C)",
            gpu_temp().unwrap_or(0.0)
        );
        if best.map_or(true, |b| ms < b.0) {
            best = Some((ms, w, r));
        }
    }
    std::env::remove_var("TANG_DECODE_WARPS");
    std::env::remove_var("TANG_DECODE_ROWS");
    if let Some((_, w, r)) = best {
        won.push(("TANG_DECODE_WARPS".to_string(), w.to_string()));
        won.push(("TANG_DECODE_ROWS".to_string(), r.to_string()));
    }
    // prefill register block (OpenCL)
    if rt.kind() == BackendKind::OpenCl {
        let seq = 64;
        let toks: Vec<usize> = (0..seq).map(|i| 1000 + i).collect();
        let mut best: Option<(f64, String)> = None;
        for mm in [
            "4x4x64", "4x8x64", "8x4x64", "2x8x64", "4x4x128", "4x4x32", "8x8x64", "2x4x64",
        ] {
            if !settle(hot, 75.0, load, deadline) {
                eprintln!("tune: budget spent");
                break;
            }
            std::env::set_var("TANG_CL_MM", mm);
            let s = Step::new(
                &rt,
                &dims,
                seq,
                512,
                &|dg| vec![Vec::new(); dg.inputs.len()],
                Some(&base),
            );
            s.prefill(&rt, &toks, 0);
            let ms = s.prog.time(&rt, 3);
            eprintln!(
                "tune: prefill {seq} tokens, block {mm}: {ms:.1} ms ({:.0} tok/s, gpu {:.1} C)",
                seq as f64 * 1e3 / ms,
                gpu_temp().unwrap_or(0.0)
            );
            if best.as_ref().map_or(true, |b| ms < b.0) {
                best = Some((ms, mm.to_string()));
            }
        }
        std::env::remove_var("TANG_CL_MM");
        if let Some((_, mm)) = best {
            won.push(("TANG_CL_MM".to_string(), mm));
        }
    }
    // merge into the tuning file
    let path = tune_path();
    let mut all: serde_json::Value = std::fs::read_to_string(&path)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_else(|| serde_json::json!({}));
    let entry: serde_json::Map<String, serde_json::Value> = won
        .iter()
        .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
        .collect();
    all[key.as_str()] = serde_json::Value::Object(entry);
    if let Some(d) = path.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    std::fs::write(&path, serde_json::to_string_pretty(&all).unwrap()).expect("write tuning file");
    eprintln!(
        "tune: wrote {} in {:.0} s: {won:?}",
        path.display(),
        t0.elapsed().as_secs_f64()
    );
    won
}
