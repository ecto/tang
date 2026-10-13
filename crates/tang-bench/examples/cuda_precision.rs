//! Precision schedules on CUDA: the transformer block (forward, and
//! forward + backward) in f32, all-f16 storage, all-bf16 storage, and the
//! schedule `ir_precision::tune_precision` picks under an error budget
//! against the f64 evaluation.
//!
//! Phases keep CPU work outside the GPU lock:
//! - `ref DIR [CASES]`: f64 references and inputs, and nvrtc for the
//!   search's candidates (CPU only);
//! - `tune DIR [CASES]`: the search on the GPU, written to `DIR/CASE.prec`;
//! - `compile DIR [CASES]`: nvrtc for every tile shape of every setting (CPU);
//! - `measure DIR [CASES]`: each setting through `Program::tuned`, with error
//!   against f64, time and output digest.
//!
//! CASES is a comma list of `fwd128,fwd512,train128,train512` (default all).
//! `TANG_BUDGET` (relative, default 1e-2) is the error budget.
use tang_bench::ir_block::{compiled_block, train_loss_graph, BlockDims};
use tang_bench::ir_precision;
use tang_compile::tensor::{ErrorBudget, ErrorStats, Format, Precision, TensorGraph, TensorId};

struct Case {
    name: String,
    g: TensorGraph,
    outs: Vec<TensorId>,
    inputs: Vec<Vec<f32>>,
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

fn case(name: &str) -> Case {
    let train = name.starts_with("train");
    let seq: usize = name
        .trim_start_matches(|c: char| c.is_alphabetic())
        .parse()
        .unwrap();
    let d = dims(seq);
    let (g, outs) = if train {
        let (tg, outs, _) = train_loss_graph(&d);
        let (tg, outs) = tg.stabilize_softmax(&outs);
        let (tg, outs) = tg.substitute_maps(&outs);
        let (tg, outs) = tg.eliminate_common(&outs);
        let (tg, outs) = tg.merge_siblings(&outs);
        let (tg, outs) = tg.relayout(&outs);
        let (tg, outs) = tg.merge_reduce_dims(&outs);
        let (mut tg, outs) = tg.split_reduce_dims(&outs);
        tg.schedule_auto(&outs);
        (tg, outs)
    } else {
        let (g, y) = compiled_block(&d);
        (g, vec![y])
    };
    // the inputs cuda_block uses: uniform ±0.1 (seed 7, per buffer in
    // input order), a zero target for the loss
    let program = g.to_cuda(&outs);
    let mut lens: Vec<(usize, usize)> = program
        .buffers
        .iter()
        .filter_map(|b| b.input.map(|k| (k, b.len)))
        .collect();
    lens.sort_unstable();
    let mut rng = tang_bench::Rng::new(7);
    let scale: f32 = std::env::var("TANG_SCALE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.1);
    let n = lens.len();
    let inputs = lens
        .iter()
        .map(|&(k, len)| {
            if train && k == n - 1 {
                vec![0.0; len]
            } else {
                (0..len).map(|_| (rng.f32() * 2.0 - 1.0) * scale).collect()
            }
        })
        .collect();
    Case {
        name: name.into(),
        g,
        outs,
        inputs,
    }
}

/// Every buffer in format `f`.
fn all(g: &TensorGraph, outs: &[TensorId], f: Format) -> TensorGraph {
    let mut v = g.clone();
    for b in g.to_cuda(outs).buffers {
        v.set_precision(b.node, Precision::storage(f));
    }
    v
}

fn settings(c: &Case, dir: &str) -> Vec<(String, TensorGraph)> {
    let mut out = vec![
        ("f32".to_string(), c.g.clone()),
        ("f16".to_string(), all(&c.g, &c.outs, Format::F16)),
        ("bf16".to_string(), all(&c.g, &c.outs, Format::BF16)),
    ];
    if let Ok(text) = std::fs::read_to_string(format!("{dir}/{}.prec", c.name)) {
        let mut t = c.g.clone();
        ir_precision::read_assignment(&mut t, &text);
        out.push(("tuned".to_string(), t));
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let phase = args.get(1).map_or("measure", |s| s.as_str()).to_string();
    let dir = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "/tmp/tang_prec".into());
    let names: Vec<String> = args
        .get(3)
        .map_or("fwd128,fwd512,train128,train512", |s| s.as_str())
        .split(',')
        .map(String::from)
        .collect();
    std::fs::create_dir_all(&dir).unwrap();
    let budget = ErrorBudget::rel(
        std::env::var("TANG_BUDGET")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1e-2),
    );
    let formats = [Format::F16, Format::BF16];
    for name in &names {
        let c = case(name);
        let ref_path = format!("{dir}/{name}.ref");
        match phase.as_str() {
            "ref" => {
                let t = std::time::Instant::now();
                let f64s: Vec<Vec<f64>> = c
                    .inputs
                    .iter()
                    .map(|v| v.iter().map(|&x| x as f64).collect())
                    .collect();
                ir_precision::write_f64s(&ref_path, &c.g.eval(&f64s, &c.outs));
                println!("{name}: reference in {:.1?}", t.elapsed());
                #[cfg(feature = "cuda")]
                {
                    let t = std::time::Instant::now();
                    let srcs = ir_precision::candidate_sources(&c.g, &c.outs, &[], &formats);
                    tang_gpu::cuda::precompile(srcs.iter().map(|s| s.as_str()), "sm_86");
                    println!(
                        "{name}: {} candidate sources, nvrtc {:.1?}",
                        srcs.len(),
                        t.elapsed()
                    );
                }
            }
            #[cfg(feature = "cuda")]
            "compile" => {
                let t = std::time::Instant::now();
                let mut srcs = Vec::new();
                for (_, g) in settings(&c, &dir) {
                    for v in std::iter::once(None).chain((0..7).map(Some)) {
                        srcs.extend(
                            g.to_cuda_variant(&c.outs, v)
                                .kernels
                                .into_iter()
                                .map(|k| k.source),
                        );
                    }
                }
                tang_gpu::cuda::precompile(srcs.iter().map(|s| s.as_str()), "sm_86");
                println!("{name}: nvrtc {:.1?}", t.elapsed());
            }
            "tune" => {
                let rt = tang_gpu::program::Runtime::select().expect("gpu");
                let want = ir_precision::read_f64s(&ref_path);
                let refs: Vec<&[f32]> = c.inputs.iter().map(|v| v.as_slice()).collect();
                let r = ir_precision::tune_precision(
                    &rt,
                    c.g.clone(),
                    &c.outs,
                    &refs,
                    &want,
                    budget,
                    &[],
                    &formats,
                    20,
                );
                for t in &r.trials {
                    println!(
                        "{name}: {:<44} rel {:.2e} {} {}",
                        t.what,
                        t.error.max_rel,
                        t.ms.zip(t.vs)
                            .map_or("over budget".to_string(), |(m, r)| format!(
                                "{m:.3} ms ({r:.3}x current)"
                            )),
                        if t.kept { "kept" } else { "" }
                    );
                }
                println!(
                    "{name}: f32 {:.3} ms (rel {:.2e}) -> {:.3} ms (rel {:.2e}), {} buffers changed",
                    r.base_ms,
                    r.base_error.max_rel,
                    r.ms,
                    r.error.max_rel,
                    r.assignment.len()
                );
                for (t, p) in &r.assignment {
                    let n = &r.graph.nodes()[t.index()];
                    println!(
                        "{name}:   node {} {:?} shape {:?}",
                        t.index(),
                        p.storage,
                        n.shape.iter().map(|d| d.n()).collect::<Vec<_>>()
                    );
                }
                std::fs::write(
                    format!("{dir}/{name}.prec"),
                    ir_precision::write_assignment(&r.assignment),
                )
                .unwrap();
            }
            "measure" => {
                use tang_gpu::program::{Program, Runtime};
                let rt = Runtime::select().expect("gpu");
                let want = ir_precision::read_f64s(&ref_path);
                let refs: Vec<&[f32]> = c.inputs.iter().map(|v| v.as_slice()).collect();
                let rounds: usize = std::env::var("TANG_ROUNDS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(3);
                let mut progs = Vec::new();
                for (s, g) in settings(&c, &dir) {
                    let path = format!(
                        "{}/target/tang-tuning-cuda-prec-{name}-{s}.txt",
                        env!("CARGO_MANIFEST_DIR")
                    );
                    let (p, rep) =
                        Program::tuned(&rt, &g, &c.outs, &refs, Some(std::path::Path::new(&path)));
                    let err: ErrorStats = ir_precision::program_error(&rt, &p, &c.outs, &want);
                    let mut h = 0xcbf2_9ce4_8422_2325u64;
                    for &o in &c.outs {
                        for x in p.read(&rt, o) {
                            for b in x.to_bits().to_le_bytes() {
                                h = (h ^ b as u64).wrapping_mul(0x0100_0000_01b3);
                            }
                        }
                    }
                    let half_mma = p
                        .program
                        .kernels
                        .iter()
                        .filter(|k| k.source.contains("m16n8k16"))
                        .count();
                    println!(
                        "{name} {s}: rel err {:.2e} (mean {:.2e}), {} of {} native, {half_mma} on half mma, digest {h:016x}",
                        err.max_rel, err.mean_rel, rep.native_kept, rep.native_total
                    );
                    progs.push((s, p, f64::INFINITY, f64::INFINITY));
                }
                // interleaved rounds, best of each
                for _ in 0..rounds {
                    for (_, p, best, ks) in progs.iter_mut() {
                        *best = best.min(p.time(&rt, 50));
                        let sum: f64 = (0..p.program.kernels.len())
                            .map(|i| p.time_kernel(&rt, i, 20))
                            .sum();
                        *ks = ks.min(sum);
                    }
                }
                for (s, _, best, ks) in &progs {
                    println!("{name} {s}: {best:.3} ms whole program, {ks:.3} ms sum of kernels");
                }
            }
            _ => panic!("phase {phase}"),
        }
    }
}
