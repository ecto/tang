//! Microbenchmarks for the ML-side reduction loops touched by the `algebraic`
//! feature (tang-tensor, tang-train, tang-infer).
//!
//! Run both arms with the same toolchain and profile:
//!
//!   CARGO_PROFILE_RELEASE_LTO=true CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 \
//!     cargo +nightly run --release -p tang-bench --example algebraic_reductions
//!   ... same, plus --features algebraic
//!
//! Best-of-7 after a warmup, `black_box` on inputs and results. GB/s counts the
//! bytes each reduction must stream from memory.

use std::hint::black_box;
use std::time::Instant;

use tang_infer::{Sampler, SamplingConfig};
use tang_tensor::{Shape, Tensor};
use tang_train::{cross_entropy_loss, mse_loss, LayerNorm, Module, RMSNorm};

const REPS: usize = 7;

/// Best-of-REPS wall time for one call, after a warmup pass.
fn best_of<T>(iters: usize, mut f: impl FnMut() -> T) -> f64 {
    for _ in 0..iters.min(32) {
        black_box(f());
    }
    let mut best = f64::INFINITY;
    for _ in 0..REPS {
        let t0 = Instant::now();
        for _ in 0..iters {
            black_box(f());
        }
        let secs = t0.elapsed().as_secs_f64() / iters as f64;
        if secs < best {
            best = secs;
        }
    }
    best
}

fn report(name: &str, n: usize, bytes: usize, secs: f64) {
    let gbs = bytes as f64 / secs / 1e9;
    println!(
        "{name:<34} n={n:<9} {:>10.3} us  {gbs:>8.1} GB/s",
        secs * 1e6
    );
}

/// Deterministic pseudo-random fill in [-1, 1); no rand dependency so the two
/// arms see byte-identical inputs.
fn fill<S: tang::Scalar>(n: usize, seed: u64) -> Vec<S> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (s >> 11) as f64 / (1u64 << 53) as f64;
            S::from_f64(u * 2.0 - 1.0)
        })
        .collect()
}

fn iters_for(n: usize) -> usize {
    (1 << 24) / n.max(1) + 1
}

// --- Tensor::sum ------------------------------------------------------------

fn bench_sum<S: tang::Scalar>(label: &str) {
    for &n in &[1 << 10, 1 << 14, 1 << 18, 1 << 22] {
        let t = Tensor::<S>::new(fill(n, 7), Shape::from_slice(&[n]));
        let t = black_box(t);
        let secs = best_of(iters_for(n), || t.sum());
        report(
            &format!("Tensor::sum<{label}>"),
            n,
            n * size_of::<S>(),
            secs,
        );
    }
}

// --- MSE loss (sum of squared differences) ----------------------------------

fn bench_mse<S: tang::Scalar>(label: &str) {
    for &n in &[1 << 10, 1 << 14, 1 << 18, 1 << 22] {
        let a = Tensor::<S>::new(fill(n, 11), Shape::from_slice(&[n]));
        let b = Tensor::<S>::new(fill(n, 29), Shape::from_slice(&[n]));
        let (a, b) = (black_box(a), black_box(b));
        let secs = best_of(iters_for(n), || mse_loss(&a, &b));
        report(
            &format!("mse_loss<{label}>"),
            n,
            2 * n * size_of::<S>(),
            secs,
        );
    }
}

// --- Softmax denominator ----------------------------------------------------

fn bench_softmax<S: tang::Scalar>(label: &str) {
    for &(rows, cols) in &[(1usize, 4096usize), (64, 4096), (8, 32768)] {
        let n = rows * cols;
        let t = Tensor::<S>::new(fill(n, 13), Shape::from_slice(&[rows, cols]));
        let t = black_box(t);
        let secs = best_of(iters_for(n).min(2000), || t.softmax(1));
        report(
            &format!("Tensor::softmax<{label}> {rows}x{cols}"),
            n,
            n * size_of::<S>(),
            secs,
        );
    }
}

// --- Norm layers ------------------------------------------------------------

fn bench_rmsnorm<S: tang::Scalar>(label: &str) {
    for &(batch, feat) in &[(8usize, 576usize), (64, 2048), (16, 8192)] {
        let n = batch * feat;
        let x = Tensor::<S>::new(fill(n, 17), Shape::from_slice(&[batch, feat]));
        let x = black_box(x);
        let mut l = RMSNorm::<S>::new(feat);
        let secs = best_of(iters_for(n).min(2000), || l.forward(&x));
        report(
            &format!("RMSNorm::forward<{label}> {batch}x{feat}"),
            n,
            n * size_of::<S>(),
            secs,
        );
    }
}

fn bench_layernorm<S: tang::Scalar>(label: &str) {
    for &(batch, feat) in &[(8usize, 576usize), (64, 2048), (16, 8192)] {
        let n = batch * feat;
        let x = Tensor::<S>::new(fill(n, 19), Shape::from_slice(&[batch, feat]));
        let x = black_box(x);
        let mut l = LayerNorm::<S>::new(feat);
        let secs = best_of(iters_for(n).min(2000), || l.forward(&x));
        report(
            &format!("LayerNorm::forward<{label}> {batch}x{feat}"),
            n,
            n * size_of::<S>(),
            secs,
        );
    }
}

// --- Cross-entropy ----------------------------------------------------------

fn bench_cross_entropy<S: tang::Scalar>(label: &str) {
    for &(batch, vocab) in &[(8usize, 4096usize), (32, 32000)] {
        let n = batch * vocab;
        let logits = Tensor::<S>::new(fill(n, 23), Shape::from_slice(&[batch, vocab]));
        let targets = Tensor::<S>::from_fn(Shape::from_slice(&[batch]), |i| {
            S::from_f64((i[0] * 7 % vocab) as f64)
        });
        let (logits, targets) = (black_box(logits), black_box(targets));
        let secs = best_of(iters_for(n).min(2000), || {
            cross_entropy_loss(&logits, &targets)
        });
        report(
            &format!("cross_entropy<{label}> {batch}x{vocab}"),
            n,
            n * size_of::<S>(),
            secs,
        );
    }
}

// --- Sampling ---------------------------------------------------------------

fn bench_sampling() {
    for &vocab in &[4096usize, 32000] {
        let logits = Tensor::<f32>::new(fill(vocab, 31), Shape::from_slice(&[vocab]));
        let logits = black_box(logits);
        let mut s = Sampler::with_seed(
            SamplingConfig {
                temperature: 0.7,
                top_p: 0.9,
                ..Default::default()
            },
            42,
        );
        let secs = best_of(2000, || s.sample(&logits, &[]));
        // The sampler works in f64 over a vocab-sized scratch buffer.
        report(
            &format!("Sampler::sample vocab={vocab}"),
            vocab,
            vocab * size_of::<f64>(),
            secs,
        );
    }
}

fn main() {
    let arm = if cfg!(feature = "algebraic") {
        "algebraic ON"
    } else {
        "algebraic OFF (baseline)"
    };
    println!("=== {arm} ===\n");

    bench_sum::<f32>("f32");
    bench_sum::<f64>("f64");
    println!();
    bench_mse::<f32>("f32");
    bench_mse::<f64>("f64");
    println!();
    bench_softmax::<f32>("f32");
    println!();
    bench_rmsnorm::<f32>("f32");
    bench_layernorm::<f32>("f32");
    println!();
    bench_cross_entropy::<f32>("f32");
    println!();
    bench_sampling();
}
