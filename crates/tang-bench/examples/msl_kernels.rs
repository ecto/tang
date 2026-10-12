//! Kernel-only timings on the native Metal backend (MSL lowering) for
//! the `kernel_only` shapes: matmul and causal attention, MSL f32 (every
//! WGSL schedule, lowered to MSL) and MSL f16 operands on simdgroup
//! fragments (f32 accumulate). Same method as `kernel_only`: warm-up,
//! then 50 launches in one submission, one sync, divide. Every result is
//! checked against an f64 reference over the full output.
//!
//!   cargo run --release -p tang-bench --features metal --example msl_kernels
use tang_bench::ir_gpu::{attention_candidates, matmul_candidates};
use tang_bench::ir_msl::{Executable, MslDevice};
use tang_compile::tensor::{Format, Precision, TensorGraph, TensorId};

const ITERS: usize = 50;

fn rand(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = tang_bench::Rng::new(seed);
    (0..n).map(|_| rng.f32() * 2.0 - 1.0).collect()
}

fn rel_err(got: &[f32], want: &[f64]) -> f64 {
    let scale = want.iter().fold(1e-30f64, |m, v| m.max(v.abs()));
    got.iter()
        .zip(want)
        .fold(0.0f64, |m, (&g, &w)| m.max((g as f64 - w).abs()))
        / scale
}

fn matmul_ref(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut y = vec![0.0; m * n];
    for i in 0..m {
        for kk in 0..k {
            let av = a[i * k + kk] as f64;
            for (yv, bv) in y[i * n..(i + 1) * n]
                .iter_mut()
                .zip(&b[kk * n..(kk + 1) * n])
            {
                *yv += av * *bv as f64;
            }
        }
    }
    y
}

/// Best of three timings.
fn time(dev: &MslDevice, exe: &Executable) -> f64 {
    (0..3)
        .map(|_| exe.time(dev, ITERS))
        .fold(f64::INFINITY, f64::min)
}

fn run(
    dev: &MslDevice,
    g: &TensorGraph,
    y: TensorId,
    inputs: &[&[f32]],
    want: &[f64],
) -> (f64, f64) {
    let exe = Executable::new(dev, g.to_msl(&[y]), inputs);
    exe.run(dev);
    let err = rel_err(&exe.read(y), want);
    (time(dev, &exe), err)
}

fn main() {
    let dev = MslDevice::new().expect("metal");
    println!(
        "{} — MSL backend, kernel-only, {ITERS} launches per sample\n",
        dev.name
    );
    for (m, k, n) in [
        (1024usize, 1024usize, 1024usize),
        (128, 576, 1536),
        (512, 576, 1536),
    ] {
        let (a, b) = (rand(m * k, 1), rand(k * n, 2));
        let want = matmul_ref(&a, &b, m, k, n);
        let flop = 2.0 * (m * k * n) as f64;
        let tf = |ms: f64| flop / ms / 1e9;
        let mut line = format!("matmul {m}x{k}x{n}:");
        let mut best = (f64::INFINITY, "");
        for (name, g, c) in matmul_candidates(m, k, n) {
            if name == "split" {
                continue;
            }
            let (ms, err) = run(&dev, &g, c, &[&a, &b], &want);
            line.push_str(&format!("  {name} {ms:.3}"));
            assert!(err < 1e-5, "{name} err {err:.1e}");
            if ms < best.0 {
                best = (ms, name);
            }
        }
        println!("{line}");
        println!(
            "  MSL f32 best: {} {:.3} ms  {:.2} TFLOP/s",
            best.1,
            best.0,
            tf(best.0)
        );
        // f16 operands, f32 accumulate, on fragments; f32 and f16 output
        for out in [Format::F32, Format::F16] {
            let mut g = TensorGraph::new();
            let ga = g.input(&[m, k]);
            let gb = g.input(&[k, n]);
            let c = g.matmul(ga, gb).unwrap();
            g.set_precision(ga, Precision::storage(Format::F16));
            g.set_precision(gb, Precision::storage(Format::F16));
            g.set_precision(c, Precision::storage(out));
            g.set_fragment(c, true);
            let (ms, err) = run(&dev, &g, c, &[&a, &b], &want);
            println!(
                "  MSL f16 fragments ({out:?} out): {ms:.3} ms  {:.2} TFLOP/s  err vs f64 {err:.1e}",
                tf(ms)
            );
            // the same f16 operands without fragments: the scalar 4x4 tile
            g.set_fragment(c, false);
            g.set_schedule(c, 0, (16, 16, 32), (4, 4));
            let (ms, err) = run(&dev, &g, c, &[&a, &b], &want);
            println!(
                "  MSL f16 operands, scalar reg4 ({out:?} out): {ms:.3} ms  {:.2} TFLOP/s  err {err:.1e}",
                tf(ms)
            );
        }
    }
    let d = 64;
    for n in [512usize, 1024] {
        let (q, k, v) = (rand(n * d, 1), rand(n * d, 2), rand(n * d, 3));
        let (g0, y0) = tang_bench::ir_gpu::naive_attention(n, d, d);
        let f64s = |x: &[f32]| x.iter().map(|&v| v as f64).collect::<Vec<_>>();
        let want = g0.eval(&[f64s(&q), f64s(&k), f64s(&v)], &[y0]).remove(0);
        let mut line = format!("attention n={n}:");
        for (name, g, y) in attention_candidates(n, d, d) {
            if name == "flash_split" {
                continue;
            }
            let (ms, err) = run(&dev, &g, y, &[&q, &k, &v], &want);
            assert!(err < 1e-4, "{name} err {err:.1e}");
            line.push_str(&format!("  {name} {ms:.3}"));
        }
        println!("{line}");
    }
}
