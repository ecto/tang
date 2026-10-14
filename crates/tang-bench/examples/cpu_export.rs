//! Export tang programs as C (`TensorGraph::to_c`) for
//! `tools/cpu-runner/bench.c` (a target with gcc and no Rust toolchain,
//! e.g. the Booster K1's Cortex-X3/A715 cores), and check what it wrote
//! back against the f64 evaluation.
//!
//!   cpu_export export ROOT [CASE...]   # ROOT/CASE/{model.c,in*.bin,info.txt}
//!   cpu_export check  ROOT [CASE...]   # reads ROOT/CASE/out*.bin
//!                                      # (CHECK_FILE=out_naive.bin: naive_gru's)
//!   cpu_export local  ROOT [CASE...]   # export, build with cc, run, check (host)
//!
//! Cases: `gru_b1`, `grufused_b1`, `gru_b64`, `grufused_b64`, `mlp_b1`,
//! `mlp_b64`, `block128` (CPU pass order: no flash rewrite), `block128flash`
//! (the GPU pipeline, `compiled_block`), each also as `_h16` (every buffer but the
//! outputs stored in f16; arithmetic f32). Error is always against the f64
//! `eval` of the f32 graph.
use std::fmt::Write as _;
use std::path::Path;
use tang_bench::ir_block::{compiled_block, transformer_block, BlockDims};
use tang_bench::ir_control::{gru_fused, gru_step, mlp_policy};
use tang_compile::tensor::{Format, Precision, TensorGraph, TensorId};

struct Case {
    name: String,
    g: TensorGraph,
    outs: Vec<TensorId>,
    flops: f64,
    inputs: Vec<Vec<f32>>,
    /// The graph the reference is evaluated on (f32 storage).
    reference: TensorGraph,
}

fn rand(rng: &mut tang_bench::Rng, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| (rng.f32() * 2.0 - 1.0) * scale).collect()
}

fn input_lens(g: &TensorGraph, outs: &[TensorId]) -> Vec<usize> {
    let p = g.to_c(outs);
    let n = p
        .program
        .buffers
        .iter()
        .filter(|b| b.input.is_some())
        .count();
    (0..n)
        .map(|k| {
            p.program
                .buffers
                .iter()
                .find(|b| b.input == Some(k))
                .unwrap()
                .len
        })
        .collect()
}

fn case(name: String, g: TensorGraph, outs: Vec<TensorId>, flops: f64, scale: f32) -> Case {
    let mut rng = tang_bench::Rng::new(7);
    let inputs = input_lens(&g, &outs)
        .into_iter()
        .map(|n| rand(&mut rng, n, scale))
        .collect();
    Case {
        name,
        reference: g.clone(),
        g,
        outs,
        flops,
        inputs,
    }
}

fn half(c: &Case) -> Case {
    let mut g = c.g.clone();
    let p = g.to_c(&c.outs);
    for b in &p.program.buffers {
        if !c.outs.iter().any(|o| g.storage(*o) == b.node) {
            g.set_precision(b.node, Precision::storage(Format::F16));
        }
    }
    Case {
        name: format!("{}_h16", c.name),
        g,
        outs: c.outs.clone(),
        flops: c.flops,
        inputs: c.inputs.clone(),
        reference: c.reference.clone(),
    }
}

fn cases(filter: &[String]) -> Vec<Case> {
    let want = |name: &str| filter.is_empty() || filter.iter().any(|f| f == name);
    let mut base = Vec::new();
    let (obs, hidden) = (64, 256);
    for batch in [1usize, 64] {
        let flops = 2.0 * (batch * 3 * hidden * (obs + hidden)) as f64;
        let (g, o) = gru_step(batch, obs, hidden);
        base.push(case(format!("gru_b{batch}"), g, o, flops, 0.2));
        let (g, o) = gru_fused(batch, obs, hidden);
        base.push(case(format!("grufused_b{batch}"), g, o, flops, 0.2));
        let (g, o) = mlp_policy(batch, obs, hidden, 2, 12);
        let flops = 2.0 * (batch * (obs * hidden + hidden * hidden + hidden * 12)) as f64;
        base.push(case(format!("mlp_b{batch}"), g, o, flops, 0.2));
    }
    if filter.is_empty() || filter.iter().any(|f| f.starts_with("block128")) {
        let d = BlockDims {
            seq: 128,
            kv_len: 128,
            rope: true,
            bias: true,
            d_model: 576,
            n_heads: 9,
            n_kv_heads: 3,
            head_dim: 64,
            ff_dim: 1536,
        };
        let (s, dm, ff) = (128.0, 576.0, 1536.0);
        let flops = 2.0 * s * dm * (576.0 + 2.0 * 192.0)
            + 2.0 * s * 576.0 * dm
            + 4.0 * s * s * 576.0
            + 6.0 * s * dm * ff;
        // the GPU pipeline (flash attention: the online-softmax state per
        // value column, which a CPU loop recomputes per column) ...
        let (g, y) = compiled_block(&d);
        base.push(case("block128flash".into(), g, vec![y], flops, 0.1));
        // ... and the same passes without `stabilize_softmax`: scores,
        // max and sum materialized, P·V a plain contraction
        let (g, y) = transformer_block(&d);
        let (g, o) = g.substitute_maps(&[y]);
        let (g, o) = g.eliminate_common(&o);
        let (g, o) = g.merge_siblings(&o);
        let (g, o) = g.relayout(&o);
        let (g, o) = g.merge_reduce_dims(&o);
        let (mut g, o) = g.split_reduce_dims(&o);
        g.schedule_auto(&o);
        base.push(case("block128".into(), g, o, flops, 0.1));
    }
    let mut out = Vec::new();
    for c in base {
        let h = half(&c);
        for c in [c, h] {
            if want(&c.name) {
                out.push(c);
            }
        }
    }
    out
}

fn export(root: &Path, c: &Case) {
    let dir = root.join(&c.name);
    std::fs::create_dir_all(&dir).unwrap();
    let p = c.g.to_c(&c.outs);
    std::fs::write(dir.join("model.c"), &p.source).unwrap();
    for (k, x) in c.inputs.iter().enumerate() {
        let data: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(dir.join(format!("in{k}.bin")), data).unwrap();
    }
    let mut info = String::new();
    writeln!(info, "case {}", c.name).unwrap();
    writeln!(info, "flops {}", c.flops).unwrap();
    writeln!(
        info,
        "kernels {} (unique C functions {})",
        p.program.kernels.len(),
        p.unique_kernels
    )
    .unwrap();
    writeln!(info, "bytes {}", p.plan.total_bytes()).unwrap();
    std::fs::write(dir.join("info.txt"), &info).unwrap();
    print!("{info}");
}

/// Worst relative error (to max |ref|) over the outputs, or `None` if a
/// file is missing.
fn check(root: &Path, c: &Case) -> Option<f64> {
    let dir = root.join(&c.name);
    let f64_in: Vec<Vec<f64>> = c
        .inputs
        .iter()
        .map(|v| v.iter().map(|x| *x as f64).collect())
        .collect();
    let want = c.reference.eval(&f64_in, &c.outs);
    let mut worst = 0.0f64;
    for (i, w) in want.iter().enumerate() {
        let file = match std::env::var("CHECK_FILE") {
            Ok(f) => f,
            Err(_) => format!("out{i}.bin"),
        };
        let raw = std::fs::read(dir.join(file)).ok()?;
        let got: Vec<f32> = raw
            .chunks(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(got.len(), w.len(), "{}: output {i} length", c.name);
        let scale = w.iter().fold(1e-30f64, |m, x| m.max(x.abs()));
        let err = got
            .iter()
            .zip(w)
            .map(|(a, b)| {
                let d = (*a as f64 - b).abs();
                if d.is_nan() {
                    f64::INFINITY
                } else {
                    d
                }
            })
            .fold(0.0, f64::max);
        worst = worst.max(err / scale);
    }
    Some(worst)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(cmd), Some(root)) = (args.first(), args.get(1)) else {
        eprintln!("usage: cpu_export export|check|local ROOT [CASE...]");
        std::process::exit(2);
    };
    let root = Path::new(root);
    let cs = cases(&args[2..]);
    match cmd.as_str() {
        "export" => cs.iter().for_each(|c| export(root, c)),
        "check" => {
            for c in &cs {
                match check(root, c) {
                    Some(e) => println!("{}: rel err {e:.2e}", c.name),
                    None => println!("{}: no output", c.name),
                }
            }
        }
        "local" => {
            let bench =
                Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/cpu-runner/bench.c");
            let cflags = std::env::var("TANG_CFLAGS")
                .unwrap_or_else(|_| tang_gpu::cpu::default_cflags().join(" "));
            for c in &cs {
                export(root, c);
                let dir = root.join(&c.name).canonicalize().unwrap();
                let exe = dir.join("bench");
                let st = std::process::Command::new("cc")
                    .args(cflags.split_whitespace())
                    .arg(format!("-DMODEL=\"{}\"", dir.join("model.c").display()))
                    .arg("-o")
                    .arg(&exe)
                    .arg(&bench)
                    .arg("-lm")
                    .status()
                    .unwrap();
                assert!(st.success(), "cc failed");
                let st = std::process::Command::new(&exe)
                    .arg(&dir)
                    .arg(std::env::var("STEPS").unwrap_or_else(|_| "20000".into()))
                    .status()
                    .unwrap();
                assert!(st.success(), "bench failed");
                match check(root, c) {
                    Some(e) => println!("{}: rel err {e:.2e}", c.name),
                    None => println!("{}: no output", c.name),
                }
            }
        }
        _ => {
            eprintln!("unknown command {cmd}");
            std::process::exit(2);
        }
    }
}
