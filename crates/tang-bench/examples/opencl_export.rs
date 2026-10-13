//! Export tang programs as OpenCL C for `tools/opencl-runner/tang_cl.c`
//! (a device with OpenCL and no Rust toolchain, e.g. the Booster K1's
//! Adreno 740), and check what the device wrote back against the f64
//! evaluation.
//!
//!   opencl_export export ROOT [CASE...]   # ROOT/CASE/{manifest.txt,k*.cl,in*.bin,ref*.bin}
//!   opencl_export check  ROOT [CASE...]   # reads ROOT/CASE/out*.bin (+ times.txt)
//!   opencl_export tune-export ROOT CASE... # + every kernel's schedule candidates
//!   opencl_export tune-pick ROOT CACHE_DIR CASE... # fastest correct one per kernel
//!
//! With `TANG_CL_TUNING=CACHE_DIR`, `export` applies `CACHE_DIR/CASE.txt`
//! (a `TuningCache` keyed by the device name in `CACHE_DIR/device.txt` and
//! the untuned OpenCL sources) when its key matches.
//!
//! Cases: `mm1024_<sched>`, `mm512_<sched>` (512×576×1536) for every
//! matmul candidate, `block128` (SmolLM-shaped block forward), `train128`
//! (its loss and gradients), `gru` (a GRU world-model step). Without
//! CASE, every case.
use std::fmt::Write as _;
use std::path::Path;
use tang_bench::ir_block::{compiled_block, train_loss_graph, BlockDims};
use tang_bench::ir_gpu::TuningCache;
use tang_compile::tensor::{Format, ScheduleCandidate, TensorGraph, TensorId};

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
    if want("gru") || want("grufused") {
        for (batch, obs, hidden) in [(1usize, 64usize, 256usize), (64, 64, 256)] {
            let (g, outs) = gru_step(batch, obs, hidden);
            let mut rng = tang_bench::Rng::new(7);
            let inputs: Vec<Vec<f32>> = input_lens(&g, &outs)
                .into_iter()
                .map(|n| rand(&mut rng, n, 0.2))
                .collect();
            out.push(Case {
                name: format!("gru_b{batch}"),
                flops: 2.0 * (batch * 3 * hidden * (obs + hidden)) as f64,
                g,
                outs,
                inputs: inputs.clone(),
                reference: None,
            });
            let (g, outs) = gru_fused(batch, obs, hidden);
            out.push(Case {
                name: format!("grufused_b{batch}"),
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

/// The same GRU step with the input and hidden contractions concatenated
/// along one reduce axis of `obs + hidden` (guarded views read zero past
/// each operand's end), so all four gate sums `r, z, i_n, h_n` share a
/// reduce space; `merge_siblings` makes them one tuple reduction whose
/// epilogue is the gate math: one kernel per step. Same inputs as
/// [`gru_step`].
fn gru_fused(batch: usize, obs: usize, hidden: usize) -> (TensorGraph, Vec<TensorId>) {
    use tang::Scalar;
    use tang_compile::tensor::{Arg, Form, Monoid, View};
    use tang_expr::ExprId;
    let mut g = TensorGraph::new();
    let x = g.input(&[batch, obs]);
    let h = g.input(&[batch, hidden]);
    let mut w = Vec::new();
    for _ in 0..3 {
        let w_i = g.input(&[obs, hidden]);
        let w_h = g.input(&[hidden, hidden]);
        let b_i = g.input(&[1, hidden]);
        let b_h = g.input(&[1, hidden]);
        w.push((w_i, w_h, b_i, b_h));
    }
    let k = obs + hidden;
    let shifted = |d: usize| Form {
        offset: -(obs as i64),
        ..Form::unit(3, d)
    };
    let xv = || View {
        forms: vec![Form::unit(3, 0), Form::unit(3, 2)],
        guard: true,
    };
    let hv = || View {
        forms: vec![Form::unit(3, 0), shifted(2)],
        guard: true,
    };
    let wiv = || View {
        forms: vec![Form::unit(3, 2), Form::unit(3, 1)],
        guard: true,
    };
    let whv = || View {
        forms: vec![shifted(2), Form::unit(3, 1)],
        guard: true,
    };
    let mut sum = |g: &mut TensorGraph, args: Vec<Arg>| {
        let two = args.len() == 4;
        g.comp(&[batch, hidden], &[k], args, Some(Monoid::sum()), |v| {
            if two {
                vec![v[0] * v[1] + v[2] * v[3]]
            } else {
                vec![v[0] * v[1]]
            }
        })
        .unwrap()
    };
    let a = |src, view| Arg { src, view };
    let r = sum(
        &mut g,
        vec![a(x, xv()), a(w[0].0, wiv()), a(h, hv()), a(w[0].1, whv())],
    );
    let z = sum(
        &mut g,
        vec![a(x, xv()), a(w[1].0, wiv()), a(h, hv()), a(w[1].1, whv())],
    );
    let i_n = sum(&mut g, vec![a(x, xv()), a(w[2].0, wiv())]);
    let h_n = sum(&mut g, vec![a(h, hv()), a(w[2].1, whv())]);
    let bias: Vec<TensorId> = w
        .iter()
        .flat_map(|&(_, _, b_i, b_h)| [b_i, b_h])
        .map(|b| g.broadcast(b, &[batch, hidden]).unwrap())
        .collect();
    let one = || ExprId::from_f64(1.0);
    let sigmoid = |e: ExprId| one() / (one() + (-e).exp());
    let mut ins = vec![r, z, i_n, h_n, h];
    ins.extend(&bias);
    let h_next = g
        .map(&ins, |v| {
            let r = sigmoid(v[0] + v[5] + v[6]);
            let z = sigmoid(v[1] + v[7] + v[8]);
            let n = (v[2] + v[9] + r * (v[3] + v[10])).tanh();
            (one() - z) * n + z * v[4]
        })
        .unwrap();
    let (g, outs) = g.substitute_maps(&[h_next]);
    let (g, outs) = g.eliminate_common(&outs);
    let (mut g, outs) = g.merge_siblings(&outs);
    g.schedule_auto(&outs);
    (g, outs)
}

fn bytes_of(format: Format, len: usize) -> usize {
    match format {
        Format::F32 => 4 * len,
        _ => 2 * len,
    }
}

/// Schedules worth timing on a small-CU GPU for `t`: tang-compile's
/// candidates plus, for matmul-shaped tiles, register tiles up to 8×8,
/// shorter k chunks and 8×8-thread groups (more, smaller work-groups), and
/// for workgroup splits, 64 and 128 threads.
fn adreno_candidates(g: &TensorGraph, t: TensorId) -> Vec<ScheduleCandidate> {
    let mut out = g.schedule_candidates(t);
    if out.contains(&(0, (16, 16, 32), (1, 1))) {
        for tile in [
            (16, 16, 32),
            (16, 16, 16),
            (16, 16, 8),
            (8, 8, 32),
            (8, 8, 16),
        ] {
            for reg in [(2, 2), (4, 4), (4, 8), (8, 4), (8, 8)] {
                out.push((0, tile, reg));
            }
        }
    }
    if out.iter().any(|c| c.0 == 256) {
        out.push((64, (0, 0, 0), (1, 1)));
        out.push((128, (0, 0, 0), (1, 1)));
    }
    let mut seen = Vec::new();
    out.retain(|c| {
        let new = !seen.contains(c);
        seen.push(*c);
        new
    });
    out
}

fn schedule_of(g: &TensorGraph, t: TensorId) -> ScheduleCandidate {
    let n = &g.nodes()[t.index()];
    (n.split, n.tile, n.reg)
}

/// f16-storage variants of a case (the precision schedule of #36/#55,
/// enumerated rather than searched: the device is offline): `_hops`
/// stores every register-tile operand (matmul inputs, weights included)
/// in f16, `_hall` every buffer but the outputs. Arithmetic and
/// accumulation stay f32. Error is still measured against the f64 `eval`.
fn f16_variants(c: &Case) -> Vec<Case> {
    use tang_compile::tensor::Precision;
    let ops: Vec<TensorId> = tang_bench::ir_precision::moves(&c.g, &c.outs, &c.outs)
        .into_iter()
        .next()
        .map(|m| m.buffers)
        .unwrap_or_default();
    let all: Vec<TensorId> =
        c.g.to_opencl(&c.outs)
            .buffers
            .iter()
            .map(|b| b.node)
            .filter(|n| !c.outs.iter().any(|o| c.g.storage(*o) == *n))
            .collect();
    [("hops", ops), ("hall", all)]
        .into_iter()
        .filter(|(_, b)| !b.is_empty())
        .map(|(tag, bufs)| {
            let mut g = c.g.clone();
            for b in bufs {
                g.set_precision(b, Precision::storage(Format::F16));
            }
            Case {
                name: format!("{}_{tag}", c.name),
                g,
                outs: c.outs.clone(),
                flops: c.flops,
                inputs: c.inputs.clone(),
                reference: c.reference.clone(),
            }
        })
        .collect()
}

/// Apply `CACHE_DIR/CASE.txt` if `TANG_CL_TUNING` is set and the key matches.
fn apply_tuning(c: &mut Case) {
    let Ok(dir) = std::env::var("TANG_CL_TUNING") else {
        return;
    };
    let dir = Path::new(&dir);
    let Ok(device) = std::fs::read_to_string(dir.join("device.txt")) else {
        return;
    };
    let cache = TuningCache::new(dir.join(format!("{}.txt", c.name)));
    let key = TuningCache::key_with(device.trim(), &c.g, &c.outs, |g, o| g.to_opencl(o));
    if let Some(entries) = cache.load(key) {
        for (node, split, tile, reg) in &entries {
            c.g.set_schedule(TensorId::from_index(*node), *split, *tile, *reg);
        }
        println!(
            "{}: {} tuned kernels from {}",
            c.name,
            entries.len(),
            cache.path.display()
        );
    }
}

fn export(root: &Path, c: &Case, tune: bool) {
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
            let data: Vec<u8> = match b.format {
                Format::F32 => c.inputs[k].iter().flat_map(|x| x.to_le_bytes()).collect(),
                Format::F16 => c.inputs[k]
                    .iter()
                    .flat_map(|&x| {
                        tang_compile::tensor::precision::f16_bits(x as f64).to_le_bytes()
                    })
                    .collect(),
                f => panic!("{f:?} inputs"),
            };
            assert_eq!(data.len(), bytes_of(b.format, b.len));
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
    if tune {
        // every base kernel's alternatives, writing the same buffer
        let mut lines = Vec::new();
        let mut meta = String::new();
        let mut files: Vec<String> = Vec::new();
        for (i, k) in p.kernels.iter().enumerate() {
            let cur = schedule_of(&c.g, k.node);
            for s in adreno_candidates(&c.g, k.node) {
                if s == cur {
                    continue;
                }
                let mut g = c.g.clone();
                g.set_schedule(k.node, s.0, s.1, s.2);
                let p2 = g.to_opencl(&c.outs);
                let Some(k2) = p2.kernels.iter().find(|k2| k2.node == k.node) else {
                    continue;
                };
                if k2.source == k.source && k2.workgroup_size == k.workgroup_size {
                    continue;
                }
                let file = match files.iter().position(|f| {
                    std::fs::read_to_string(dir.join(f)).is_ok_and(|x| x == k2.source)
                }) {
                    Some(j) => files[j].clone(),
                    None => {
                        let f = format!("c{:03}.cl", files.len());
                        std::fs::write(dir.join(&f), &k2.source).unwrap();
                        files.push(f.clone());
                        f
                    }
                };
                let args: Vec<String> = k2
                    .reads
                    .iter()
                    .chain(std::iter::once(&k2.node))
                    .map(|&t| buf_index(t).to_string())
                    .collect();
                writeln!(
                    meta,
                    "{} {i} {} {} {} {} {} {} {}",
                    lines.len(),
                    k.node.index(),
                    s.0,
                    s.1 .0,
                    s.1 .1,
                    s.1 .2,
                    s.2 .0,
                    s.2 .1
                )
                .unwrap();
                lines.push(format!(
                    "c {i} {file} {} {} {} {} {}",
                    k2.workgroup_size,
                    k2.count,
                    k2.points,
                    args.len(),
                    args.join(" ")
                ));
            }
        }
        writeln!(m, "candidates {}", lines.len()).unwrap();
        for l in &lines {
            writeln!(m, "{l}").unwrap();
        }
        std::fs::write(dir.join("cands.txt"), meta).unwrap();
        println!("{}: {} candidates", c.name, lines.len());
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
    // worst output relative to its own max |ref| (a small gradient next
    // to a large one is invisible in the pooled figure)
    let mut worst_own = 0.0f64;
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
        let mut own = 0.0f64;
        for (&g, &w) in got.iter().zip(&want) {
            let e = (g as f64 - w).abs();
            let e = if e.is_nan() { f64::INFINITY } else { e };
            err = err.max(e);
            own = own.max(e);
        }
        let own_scale = want.iter().fold(0.0f64, |s, w| s.max(w.abs()));
        worst_own = worst_own.max(own / own_scale.max(1e-30));
        scale = scale.max(own_scale);
        o += 1;
    }
    let gflops = flops / (prog_ms * 1e-3) / 1e9;
    println!(
        "{name:20} {prog_ms:9.3} ms  {gflops:7.1} GFLOP/s  max err {err:.2e}  (max |ref| {scale:.2}, rel {:.1e}, worst output rel {worst_own:.1e})",
        err / scale.max(1e-30)
    );
}

/// Pick each kernel's fastest candidate that matches the base output
/// (relative error ≤ 1e-4 — f32 reassociation, not a different answer) and
/// beats the base kernel; write them as a `TuningCache`.
fn tune_pick(root: &Path, cache_dir: &Path, c: &Case) {
    let dir = root.join(&c.name);
    let tune = std::fs::read_to_string(dir.join("tune.txt")).expect("tune.txt");
    let times = std::fs::read_to_string(dir.join("times.txt")).expect("times.txt");
    let meta = std::fs::read_to_string(dir.join("cands.txt")).expect("cands.txt");
    let device = tune
        .lines()
        .find_map(|l| l.strip_prefix("device "))
        .expect("device line")
        .trim()
        .to_string();
    let base: Vec<f64> = times
        .lines()
        .filter_map(|l| l.strip_prefix("kernel "))
        .map(|l| l.split_whitespace().nth(1).unwrap().parse().unwrap())
        .collect();
    let mut timed = std::collections::HashMap::new();
    for l in tune.lines().filter_map(|l| l.strip_prefix("c ")) {
        let v: Vec<&str> = l.split_whitespace().collect();
        if let (Some(ms), Some(err)) = (v.get(1), v.get(2)) {
            if let (Ok(ms), Ok(err)) = (ms.parse::<f64>(), err.parse::<f64>()) {
                timed.insert(v[0].parse::<usize>().unwrap(), (ms, err));
            }
        }
    }
    let p = c.g.to_opencl(&c.outs);
    let mut best: Vec<(f64, ScheduleCandidate)> = p
        .kernels
        .iter()
        .enumerate()
        .map(|(i, k)| (base[i], schedule_of(&c.g, k.node)))
        .collect();
    let (mut wrong, mut n) = (0, 0);
    for l in meta.lines() {
        let v: Vec<u32> = l.split_whitespace().map(|x| x.parse().unwrap()).collect();
        let (ci, ki) = (v[0] as usize, v[1] as usize);
        let s = (v[3], (v[4], v[5], v[6]), (v[7], v[8]));
        let Some(&(ms, err)) = timed.get(&ci) else {
            continue;
        };
        n += 1;
        if err > 1e-4 {
            wrong += 1;
            continue;
        }
        if ms < best[ki].0 {
            best[ki] = (ms, s);
        }
    }
    let before: f64 = base.iter().sum();
    let after: f64 = best.iter().map(|b| b.0).sum();
    println!(
        "{}: {n} candidates timed ({wrong} wrong), sum of kernels {before:.3} -> {after:.3} ms",
        c.name
    );
    for (i, k) in p.kernels.iter().enumerate() {
        let cur = schedule_of(&c.g, k.node);
        if best[i].1 != cur {
            println!(
                "  k{i:<3} {:8.3} -> {:8.3} ms  {cur:?} -> {:?}",
                base[i], best[i].0, best[i].1
            );
        }
    }
    let entries: Vec<_> = p
        .kernels
        .iter()
        .zip(&best)
        .map(|(k, b)| (k.node.index(), b.1 .0, b.1 .1, b.1 .2))
        .collect();
    std::fs::create_dir_all(cache_dir).unwrap();
    std::fs::write(cache_dir.join("device.txt"), format!("{device}\n")).unwrap();
    let key = TuningCache::key_with(&device, &c.g, &c.outs, |g, o| g.to_opencl(o));
    TuningCache::new(cache_dir.join(format!("{}.txt", c.name))).store_entries(key, &entries);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (cmd, root, filter) = match args.as_slice() {
        [cmd, root, rest @ ..] => (cmd.as_str(), Path::new(root), rest.to_vec()),
        _ => panic!("usage: opencl_export export|check ROOT [CASE...]"),
    };
    match cmd {
        "export" | "tune-export" => {
            let f16 = std::env::var("TANG_CL_F16").is_ok();
            for mut c in cases(&filter) {
                apply_tuning(&mut c);
                export(root, &c, cmd == "tune-export");
                if f16 {
                    for v in f16_variants(&c) {
                        export(root, &v, false);
                    }
                }
            }
        }
        "tune-pick" => {
            let (cache_dir, filter) = filter
                .split_first()
                .expect("tune-pick ROOT CACHE_DIR CASE...");
            for c in cases(filter) {
                tune_pick(root, Path::new(cache_dir), &c);
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
