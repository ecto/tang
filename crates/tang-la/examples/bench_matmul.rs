use std::time::Instant;
use tang_la::DMat;

fn bench_matmul<S: tang::Scalar + From<f32>>(label: &str, n: usize, iters: usize) {
    let a = DMat::from_fn(n, n, |i, j| S::from(((i * 7 + j * 13) % 100) as f32));
    let b = DMat::from_fn(n, n, |i, j| S::from(((i * 11 + j * 3) % 100) as f32));

    // Warmup
    let _ = a.mul_mat(&b);

    let start = Instant::now();
    for _ in 0..iters {
        let _ = a.mul_mat(&b);
    }
    let elapsed = start.elapsed();
    let per_iter = elapsed / iters as u32;

    let flops = 2.0 * (n as f64).powi(3) * iters as f64;
    let gflops = flops / elapsed.as_secs_f64() / 1e9;

    println!(
        "{label:>12}  {n}x{n}  {iters} iters  {elapsed:>10.3?} total  {per_iter:>10.3?}/iter  {gflops:.2} GFLOP/s"
    );
}

fn main() {
    let n = 512;
    let iters = 20;

    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    println!("=== BLAS (Accelerate) ===");
    #[cfg(not(all(feature = "accelerate", target_os = "macos")))]
    println!("=== Generic (no BLAS) ===");

    bench_matmul::<f32>("f32", n, iters);
    bench_matmul::<f64>("f64", n, iters);
}
