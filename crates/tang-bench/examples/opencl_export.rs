//! Export tang programs as OpenCL C for `tools/opencl-runner/tang_cl.c`
//! (a device with OpenCL and no Rust toolchain, e.g. the Booster K1's
//! Adreno 740), and check what the device wrote back against the f64
//! evaluation.
//!
//!   opencl_export export ROOT [CASE...]   # ROOT/CASE/{manifest.txt,k*.cl,in*.bin,ref*.bin}
//!   opencl_export check  ROOT [CASE...]   # reads ROOT/CASE/out*.bin (+ times.txt)
//!
//! Cases: `mm1024_<sched>`, `mm512_<sched>` (512×576×1536) for every
//! matmul candidate, `block128` (SmolLM-shaped block forward), `train128`
//! (its loss and gradients), `gru` (a GRU world-model step). Without
//! CASE, every case.
use std::fmt::Write as _;
use std::path::Path;
use tang_bench::ir_block::{compiled_block, train_loss_graph, BlockDims};
use tang_compile::tensor::{Format, TensorGraph, TensorId};

struct Case {
    name: String,
    g: TensorGraph,
    outs: Vec<TensorId>,
    /// FLOPs of one run (0 if not meaningful).
    flops: f64,
    /// Inputs in graph input order.
    inputs: Vec<Vec<f32>>,
    /// f64 reference per output; `None` evaluates the graph.
    reference: Option<Vec<Vec<f64>>>,
}

fn dims(seq: usize) -> BlockDims {
    BlockDims {
        seq,
        kv_len: seq,
        rope: true,
        bias: true,
        d_model: 576,
        n_heads: 9,
        n_kv_heads: 3,
        head_dim: 64,
        ff_dim: 1536,
    }
}

fn rand(rng: &mut tang_bench::Rng, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| (rng.f32() * 2.0 - 1.0) * scale).collect()
}

/// Graph inputs' lengths, from the lowered program.
fn input_lens(g: &TensorGraph, outs: &[TensorId]) -> Vec<usize> {
    let p = g.to_opencl(outs);
    let n = p.buffers.iter().filter(|b| b.input.is_some()).count();
    (0..n)
        .map(|k| p.buffers.iter().find(|b| b.input == Some(k)).unwrap().len)
        .collect()
}

fn matmul_ref(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        for p in 0..k {
            let x = a[i * k + p] as f64;
            let row = &b[p * n..p * n + n];
            let out = &mut c[i * n..i * n + n];
            for j in 0..n {
                out[j] += x * row[j] as f64;
            }
        }
    }
    c
}

fn cases(filter: &[String]) -> Vec<Case> {
    let want =
        |name: &str| filter.is_empty() || filter.iter().any(|f| name.starts_with(f.as_str()));
    let mut out = Vec::new();
    for (tag, m, k, n) in [("mm1024", 1024, 1024, 1024), ("mm512", 512, 576, 1536)] {
        let cands = tang_bench::ir_gpu::matmul_candidates(m, k, n);
        if !cands.iter().any(|(s, _, _)| want(&format!("{tag}_{s}"))) {
            continue;
        }
        let mut rng = tang_bench::Rng::new(7);
        let a = rand(&mut rng, m * k, 1.0);
        let b = rand(&mut rng, k * n, 1.0);
        let reference = matmul_ref(&a, &b, m, k, n);
        for (s, g, c) in cands {
            let name = format!("{tag}_{s}");
            if !want(&name) {
                continue;
            }
            out.push(Case {
                name,
                g,
                outs: vec![c],
                flops: 2.0 * (m * k * n) as f64,
                inputs: vec![a.clone(), b.clone()],
                reference: Some(vec![reference.clone()]),
            });
        }
    }
    if want("block128") {
        let d = dims(128);
        let (g, y) = compiled_block(&d);
        let mut rng = tang_bench::Rng::new(7);
        let inputs = input_lens(&g, &[y])
            .into_iter()
            .map(|n| rand(&mut rng, n, 0.1))
            .collect();
        out.push(Case {
            name: "block128".into(),
            flops: block_flops(&d),
            g,
            outs: vec![y],
            inputs,
            reference: None,
        });
    }
    if want("train128") {
        let d = dims(128);
        let (tg, outs, target) = train_loss_graph(&d);
        let (tg, outs) = tg.stabilize_softmax(&outs);
        let (tg, outs) = tg.substitute_maps(&outs);
        let (tg, outs) = tg.eliminate_common(&outs);
        let (tg, outs) = tg.merge_siblings(&outs);
        let (tg, outs) = tg.relayout(&outs);
        let (tg, outs) = tg.merge_reduce_dims(&outs);
        let (mut tg, outs) = tg.split_reduce_dims(&outs);
        tg.schedule_auto(&outs);
        let mut rng = tang_bench::Rng::new(7);
        let inputs = input_lens(&tg, &outs)
            .into_iter()
            .enumerate()
            .map(|(i, n)| {
                if i == target {
                    vec![0.0; n]
                } else {
                    rand(&mut rng, n, 0.1)
                }
            })
            .collect();
        out.push(Case {
            name: "train128".into(),
            flops: 3.0 * block_flops(&d),
            g: tg,
            outs,
            inputs,
            reference: None,
        });
    }
    if want("gru") {
        for (batch, obs, hidden) in [(1usize, 64usize, 256usize), (64, 64, 256)] {
            let (g, outs) = gru_step(batch, obs, hidden);
            let mut rng = tang_bench::Rng::new(7);
            let inputs = input_lens(&g, &outs)
                .into_iter()
                .map(|n| rand(&mut rng, n, 0.2))
                .collect();
            out.push(Case {
                name: format!("gru_b{batch}"),
                flops: 2.0 * (batch * 3 * hidden * (obs + hidden)) as f64,
                g,
                outs,
                inputs,
                reference: None,
            });
        }
    }
    out
}

/// Matmul FLOPs of the block forward (projections, attention, MLP).
fn block_flops(d: &BlockDims) -> f64 {
    let (s, dm, h, kvh, hd, ff) = (
        d.seq as f64,
        d.d_model as f64,
        d.n_heads as f64,
        d.n_kv_heads as f64,
        d.head_dim as f64,
        d.ff_dim as f64,
    );
    let proj = 2.0 * s * dm * (h * hd + 2.0 * kvh * hd) + 2.0 * s * h * hd * dm;
    let attn = 2.0 * 2.0 * s * d.kv_len as f64 * h * hd;
    let mlp = 2.0 * s * dm * ff * 3.0;
    proj + attn + mlp
}

/// One step of a GRU cell (PyTorch gate convention, as ipse-world's
/// `GruCell`): inputs `x [batch, obs]`, `h [batch, hidden]`, then per gate
/// r, z, n: `W_i [obs, hidden]`, `W_h [hidden, hidden]`, `b_i`, `b_h
/// [hidden]`; output `h' [batch, hidden]`.
fn gru_step(batch: usize, obs: usize, hidden: usize) -> (TensorGraph, Vec<TensorId>) {
    use tang::Scalar;
    use tang_expr::ExprId;
    let mut g = TensorGraph::new();
    let x = g.input(&[batch, obs]);
    let h = g.input(&[batch, hidden]);
    let mut gate = |g: &mut TensorGraph| {
        let w_i = g.input(&[obs, hidden]);
        let w_h = g.input(&[hidden, hidden]);
        let b_i = g.input(&[1, hidden]);
        let b_h = g.input(&[1, hidden]);
        let gi = g.matmul(x, w_i).unwrap();
        let gh = g.matmul(h, w_h).unwrap();
        let b_i = g.broadcast(b_i, &[batch, hidden]).unwrap();
        let b_h = g.broadcast(b_h, &[batch, hidden]).unwrap();
        let gi = g.map(&[gi, b_i], |v| v[0] + v[1]).unwrap();
        let gh = g.map(&[gh, b_h], |v| v[0] + v[1]).unwrap();
        (gi, gh)
    };
    let (ir, hr) = gate(&mut g);
    let (iz, hz) = gate(&mut g);
    let (inn, hn) = gate(&mut g);
    let one = || ExprId::from_f64(1.0);
    let sigmoid = |e: ExprId| one() / (one() + (-e).exp());
    let h_next = g
        .map(&[ir, hr, iz, hz, inn, hn, h], |v| {
            let r = sigmoid(v[0] + v[1]);
            let z = sigmoid(v[2] + v[3]);
            let n = (v[4] + r * v[5]).tanh();
            (one() - z) * n + z * v[6]
        })
        .unwrap();
    let (g, outs) = g.substitute_maps(&[h_next]);
    let (mut g, outs) = g.eliminate_common(&outs);
    g.schedule_auto(&outs);
    (g, outs)
}

fn bytes_of(format: Format, len: usize) -> usize {
    match format {
        Format::F32 => 4 * len,
        _ => 2 * len,
    }
}

fn export(root: &Path, c: &Case) {
    let dir = root.join(&c.name);
    std::fs::create_dir_all(&dir).unwrap();
    let p = c.g.to_opencl(&c.outs);
    let buf_index = |t: TensorId| {
        let t = p.storage(t);
        p.buffers.iter().position(|b| b.node == t).expect("buffer")
    };
    let mut m = String::from("tang-opencl 1\n");
    writeln!(m, "buffers {}", p.buffers.len()).unwrap();
    for b in &p.buffers {
        let input = b.input.map_or(-1, |k| k as i64);
        writeln!(m, "b {} {input}", bytes_of(b.format, b.len)).unwrap();
        if let Some(k) = b.input {
            assert_eq!(b.format, Format::F32, "f32 inputs only");
            let data: Vec<u8> = c.inputs[k].iter().flat_map(|x| x.to_le_bytes()).collect();
            assert_eq!(data.len(), 4 * b.len);
            std::fs::write(dir.join(format!("in{k}.bin")), data).unwrap();
        }
    }
    writeln!(m, "kernels {}", p.kernels.len()).unwrap();
    for (i, k) in p.kernels.iter().enumerate() {
        let file = format!("k{i:02}.cl");
        std::fs::write(dir.join(&file), &k.source).unwrap();
        let args: Vec<String> = k
            .reads
            .iter()
            .chain(std::iter::once(&k.node))
            .map(|&t| buf_index(t).to_string())
            .collect();
        writeln!(
            m,
            "k {file} {} {} {} {} {}",
            k.workgroup_size,
            k.count,
            k.points,
            args.len(),
            args.join(" ")
        )
        .unwrap();
    }
    writeln!(m, "outputs {}", c.outs.len()).unwrap();
    for (o, &t) in c.outs.iter().enumerate() {
        writeln!(m, "o {} out{o}.bin", buf_index(t)).unwrap();
    }
    std::fs::write(dir.join("manifest.txt"), m).unwrap();
    // the f64 reference is kept across re-exports (same seeded inputs)
    let reference = match &c.reference {
        _ if dir.join("ref0.bin").exists() => Vec::new(),
        Some(r) => r.clone(),
        None => {
            let f64s: Vec<Vec<f64>> = c
                .inputs
                .iter()
                .map(|v| v.iter().map(|&x| x as f64).collect())
                .collect();
            c.g.eval(&f64s, &c.outs)
        }
    };
    for (o, r) in reference.iter().enumerate() {
        let data: Vec<u8> = r.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(dir.join(format!("ref{o}.bin")), data).unwrap();
    }
    std::fs::write(dir.join("flops.txt"), format!("{}\n", c.flops)).unwrap();
    println!(
        "{}: {} kernels, {} buffers",
        c.name,
        p.kernels.len(),
        p.buffers.len()
    );
}

fn read_f64(path: &Path) -> Vec<f64> {
    std::fs::read(path)
        .unwrap()
        .chunks(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn check(root: &Path, name: &str) {
    let dir = root.join(name);
    let flops: f64 = std::fs::read_to_string(dir.join("flops.txt"))
        .map(|s| s.trim().parse().unwrap())
        .unwrap_or(0.0);
    let times = std::fs::read_to_string(dir.join("times.txt")).unwrap_or_default();
    let prog_ms: f64 = times
        .lines()
        .find_map(|l| l.strip_prefix("program "))
        .map_or(f64::NAN, |v| v.parse().unwrap());
    let (mut err, mut scale) = (0.0f64, 0.0f64);
    let mut o = 0;
    while dir.join(format!("ref{o}.bin")).exists() {
        let want = read_f64(&dir.join(format!("ref{o}.bin")));
        let got: Vec<f32> = match std::fs::read(dir.join(format!("out{o}.bin"))) {
            Ok(b) => b
                .chunks(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            Err(_) => {
                println!("{name}: no out{o}.bin");
                return;
            }
        };
        assert_eq!(got.len(), want.len(), "{name} output {o} length");
        for (&g, &w) in got.iter().zip(&want) {
            let e = (g as f64 - w).abs();
            err = if e.is_nan() {
                f64::INFINITY
            } else {
                err.max(e)
            };
        }
        scale = want.iter().fold(scale, |s, w| s.max(w.abs()));
        o += 1;
    }
    let gflops = flops / (prog_ms * 1e-3) / 1e9;
    println!(
        "{name:16} {prog_ms:9.3} ms  {gflops:7.1} GFLOP/s  max err {err:.2e}  (max |ref| {scale:.2}, rel {:.1e})",
        err / scale.max(1e-30)
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (cmd, root, filter) = match args.as_slice() {
        [cmd, root, rest @ ..] => (cmd.as_str(), Path::new(root), rest.to_vec()),
        _ => panic!("usage: opencl_export export|check ROOT [CASE...]"),
    };
    match cmd {
        "export" => {
            for c in cases(&filter) {
                export(root, &c);
            }
        }
        "check" => {
            let mut names: Vec<String> = std::fs::read_dir(root)
                .unwrap()
                .filter_map(|e| e.ok())
                .filter(|e| e.path().join("manifest.txt").exists())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|n| filter.is_empty() || filter.iter().any(|f| n.starts_with(f.as_str())))
                .collect();
            names.sort();
            for n in names {
                check(root, &n);
            }
        }
        _ => panic!("unknown command {cmd}"),
    }
}
