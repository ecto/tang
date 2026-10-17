//! Q4 `linear` at a few rows (speculative-decoding verify) vs one: time and agreement.
//! `cargo run --release --example q4_rows --features metal [K N]...`
use std::time::Instant;
use tang_compute::{ComputeDevice, MetalDevice};

fn main() {
    let dev = MetalDevice::new().expect("metal");
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    // Qwen3-32B: q/o, gate+up fused, down.
    let shapes: Vec<(usize, usize)> = if args.len() >= 2 {
        args.chunks(2).map(|c| (c[0], c[1])).collect()
    } else {
        vec![(5120, 10240), (5120, 51200), (25600, 5120)]
    };
    let mut seed = 1u32;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    };
    for (k, n) in shapes {
        let group = 64;
        let packed: Vec<u32> = (0..n * k / 8).map(|_| rnd()).collect();
        let bf = |x: f32| (x.to_bits() >> 16) as u16;
        let scales: Vec<u16> = (0..n * k / group)
            .map(|_| bf(0.01 + (rnd() % 100) as f32 * 1e-4))
            .collect();
        let biases: Vec<u16> = (0..n * k / group).map(|_| bf(-0.08)).collect();
        let w = dev.upload_q4(&packed, &scales, &biases, group);
        let x: Vec<f32> = (0..8 * k)
            .map(|_| (rnd() % 2000) as f32 / 1000.0 - 1.0)
            .collect();
        // Reference: each row alone (m = 1).
        let refs: Vec<Vec<f32>> = (0..8)
            .map(|r| dev.download(&dev.linear(&dev.upload(&x[r * k..(r + 1) * k]), &w, 1, k, n)))
            .collect();
        let mut base = 0.0;
        for m in 1..=8 {
            let xb = dev.upload(&x[..m * k]);
            let y = dev.download(&dev.linear(&xb, &w, m, k, n));
            let mut err = 0f32;
            for r in 0..m {
                for c in 0..n {
                    err = err.max((y[r * n + c] - refs[r][c]).abs());
                }
            }
            let reps = 30;
            dev.sync();
            let t = Instant::now();
            for _ in 0..reps {
                let _ = dev.linear(&xb, &w, m, k, n);
            }
            dev.sync();
            let us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
            if m == 1 {
                base = us;
            }
            let gbs = (n * k / 2) as f64 / us / 1e3;
            println!(
                "K={k:6} N={n:6} m={m}: {us:8.1} us  {:.2}x  {gbs:6.0} GB/s  max|d| {err:.2e}",
                us / base
            );
        }
    }
}
