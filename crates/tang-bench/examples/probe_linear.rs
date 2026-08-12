//! Reproduces the "Dense GEMM" table in BENCHMARKS.md.
//!
//! ```text
//! cargo run -p tang-bench --example probe_linear --release
//! cargo run -p tang-bench --example probe_linear --release --features threads
//! ```
//!
//! Criterion covers the layer-level benchmarks; this exists to attribute their
//! time to the individual pieces — the transpose, the GEMM, and the layer
//! overhead around them — which is what makes a regression in any one of them
//! legible.

use std::time::Instant;
use tang_la::DMat;
use tang_tensor::{Shape, Tensor};
use tang_train::{Linear, Module};

fn time(reps: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..3 {
        f();
    }
    let start = Instant::now();
    for _ in 0..reps {
        f();
    }
    start.elapsed().as_secs_f64() / reps as f64
}

fn row(label: &str, secs: f64, flops: Option<f64>) {
    match flops {
        Some(fl) => println!(
            "{label:<26} {:>8.3} ms   {:>7.1} GFLOP/s",
            secs * 1e3,
            fl / secs / 1e9
        ),
        None => println!("{label:<26} {:>8.3} ms", secs * 1e3),
    }
}

fn main() {
    // SmolLM2-135M projection shape at a training batch of 16.
    let (batch, dim) = (16usize, 576usize);

    let mut linear: Linear<f32> = Linear::new(dim, dim, 42);
    let x = Tensor::<f32>::from_fn(Shape::from_slice(&[batch, dim]), |i| {
        (i[0] + i[1]) as f32 * 1e-3
    });
    let w = Tensor::<f32>::from_fn(Shape::from_slice(&[dim, dim]), |i| {
        (i[0] * i[1]) as f32 * 1e-4
    });
    let grad = Tensor::<f32>::from_fn(Shape::from_slice(&[batch, dim]), |i| {
        (i[0] + i[1]) as f32 * 1e-3
    });

    let layer_flops = 2.0 * (batch * dim * dim) as f64;

    let fwd = time(50, || {
        std::hint::black_box(linear.forward(&x));
    });
    let bwd = time(50, || {
        std::hint::black_box(linear.backward(&grad));
    });
    let init = time(50, || {
        std::hint::black_box(Linear::<f32>::new(dim, dim, 42));
    });
    let tr = time(50, || {
        std::hint::black_box(w.transpose());
    });
    let mm = time(50, || {
        std::hint::black_box(x.matmul(&w));
    });
    let nt = time(50, || {
        std::hint::black_box(x.matmul_nt(&w));
    });

    println!("--- layer ({batch}x{dim} -> {dim}, f32) ---");
    row("Linear::forward", fwd, None);
    row("Linear::backward", bwd, None);
    row("Linear::new (randn init)", init, None);
    row("transpose 576x576", tr, None);
    row("Tensor::matmul", mm, Some(layer_flops));
    row("Tensor::matmul_nt", nt, Some(layer_flops));

    let a = DMat::<f32>::from_fn(batch, dim, |i, j| (i + j) as f32 * 1e-3);
    let b = DMat::<f32>::from_fn(dim, dim, |i, j| (i + j) as f32 * 1e-3);
    let skinny = time(50, || {
        std::hint::black_box(a.mul_mat(&b));
    });

    let n = 512usize;
    let a5 = DMat::<f32>::from_fn(n, n, |i, j| (i + j) as f32 * 1e-3);
    let b5 = DMat::<f32>::from_fn(n, n, |i, j| (i + j) as f32 * 1e-3);
    let square = time(10, || {
        std::hint::black_box(a5.mul_mat(&b5));
    });

    println!("\n--- DMat::mul_mat ---");
    row("16x576x576", skinny, Some(layer_flops));
    row("512x512x512", square, Some(2.0 * (n * n * n) as f64));
}
