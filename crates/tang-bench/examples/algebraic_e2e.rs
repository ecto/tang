//! End-to-end training benchmarks — the honest check on whether the per-op
//! reduction speedups survive into epoch time.
//!
//! Two real training loops:
//!   1. MLP regression with MSE loss (reduction-heavy relative to the model).
//!   2. Classifier with RMSNorm + cross-entropy (the transformer-shaped path).
//!
//! Run both arms with the same toolchain and profile:
//!
//!   CARGO_PROFILE_RELEASE_LTO=true CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 \
//!     cargo +nightly run --release -p tang-bench --example algebraic_e2e
//!   ... same, plus --features algebraic

use std::hint::black_box;
use std::time::Instant;

use tang_tensor::{Shape, Tensor};
use tang_train::*;

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (s >> 11) as f64 / (1u64 << 53) as f64;
            (u * 2.0 - 1.0) as f32
        })
        .collect()
}

/// Median of the per-epoch times, plus the total.
fn summarize(name: &str, mut epochs: Vec<f64>, final_loss: f32) {
    let total: f64 = epochs.iter().sum();
    epochs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = epochs[epochs.len() / 2];
    println!(
        "{name:<38} {:>9.2} ms/epoch (median)  {:>9.2} ms total   loss={final_loss:.6}",
        median * 1e3,
        total * 1e3
    );
}

/// MLP regression trained with MSE — exercises mse_loss + Tensor::sum.
fn e2e_mlp_mse(batch: usize, d_in: usize, hidden: usize, epochs: usize) {
    let x = Tensor::new(fill(batch * d_in, 3), Shape::from_slice(&[batch, d_in]));
    let y = Tensor::new(fill(batch, 5), Shape::from_slice(&[batch, 1]));
    let (x, y) = (black_box(x), black_box(y));

    let mut model = Sequential::<f32>::new(vec![
        Box::new(Linear::new(d_in, hidden, 42)),
        Box::new(Tanh::new()),
        Box::new(Linear::new(hidden, hidden, 43)),
        Box::new(Tanh::new()),
        Box::new(Linear::new(hidden, 1, 44)),
    ]);
    let mut opt = ModuleAdam::new(0.001);

    let mut times = Vec::with_capacity(epochs);
    let mut loss = 0.0f32;
    for _ in 0..epochs {
        let t0 = Instant::now();
        let pred = model.forward(&x);
        loss = mse_loss(&pred, &y);
        let grad = mse_loss_grad(&pred, &y);
        model.backward(&grad);
        opt.step(&mut model.parameters_mut());
        times.push(t0.elapsed().as_secs_f64());
        black_box(loss);
    }
    summarize(
        &format!("MLP+MSE {batch}x{d_in}->{hidden} e={epochs}"),
        times,
        loss,
    );
}

/// Classifier with RMSNorm + cross-entropy — the transformer-shaped path.
fn e2e_rmsnorm_ce(batch: usize, d_model: usize, classes: usize, epochs: usize) {
    let x = Tensor::new(
        fill(batch * d_model, 7),
        Shape::from_slice(&[batch, d_model]),
    );
    let targets = Tensor::from_fn(Shape::from_slice(&[batch]), |i| (i[0] % classes) as f32);
    let (x, targets) = (black_box(x), black_box(targets));

    let mut norm = RMSNorm::<f32>::new(d_model);
    let mut head = Linear::<f32>::new(d_model, classes, 42);
    let mut opt = ModuleAdam::new(0.001);

    let mut times = Vec::with_capacity(epochs);
    let mut loss = 0.0f32;
    for _ in 0..epochs {
        let t0 = Instant::now();
        let h = norm.forward(&x);
        let logits = head.forward(&h);
        loss = cross_entropy_loss(&logits, &targets);
        let g = cross_entropy_loss_grad(&logits, &targets);
        let gh = head.backward(&g);
        norm.backward(&gh);
        {
            let mut p = head.parameters_mut();
            p.extend(norm.parameters_mut());
            opt.step(&mut p);
        }
        times.push(t0.elapsed().as_secs_f64());
        black_box(loss);
    }
    summarize(
        &format!("RMSNorm+CE {batch}x{d_model}->{classes} e={epochs}"),
        times,
        loss,
    );
}

fn main() {
    let arm = if cfg!(feature = "algebraic") {
        "algebraic ON"
    } else {
        "algebraic OFF (baseline)"
    };
    println!("=== end-to-end training: {arm} ===\n");

    e2e_mlp_mse(256, 128, 256, 60);
    e2e_mlp_mse(1024, 256, 512, 30);
    println!();
    // Kept small: cross_entropy_loss_grad recomputes the softmax denominator
    // per element (O(batch * classes^2)), so larger vocabs are dominated by
    // that, not by anything the `algebraic` feature touches.
    e2e_rmsnorm_ce(64, 576, 256, 20);
    e2e_rmsnorm_ce(32, 2048, 512, 10);
}
