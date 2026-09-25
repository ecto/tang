//! Prefill-shaped kernel timings on Metal (Qwen3-4B, 512-token chunk).
use std::time::Instant;
use tang_compute::{ComputeDevice, MetalDevice};

fn time(dev: &MetalDevice, name: &str, flops: f64, f: impl Fn()) {
    f();
    dev.sync();
    let reps = 5;
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    dev.sync();
    let per = t.elapsed().as_secs_f64() / reps as f64;
    println!(
        "{name:<34} {:>9.2} ms  {:>6.2} TFLOP/s",
        per * 1e3,
        flops / per / 1e12
    );
}

fn main() {
    let dev = MetalDevice::new().unwrap();
    let m = 512;
    for &(k, n) in &[(2560, 9728 * 2), (9728, 2560), (2560, 6144)] {
        let x = dev.upload(&vec![0.01f32; m * k]);
        let packed = vec![0x12345678u32; k * n / 8];
        let sb = vec![0x3c00u16; k * n / 64];
        let wq = dev.upload_q4(&packed, &sb, &sb, 64);
        let wb = dev.upload_bf16(&vec![0x3f80u16; k * n]);
        let fl = 2.0 * (m * k * n) as f64;
        time(&dev, &format!("linear q4   {m}x{k}x{n}"), fl, || {
            dev.linear(&x, &wq, m, k, n);
        });
        time(&dev, &format!("linear bf16 {m}x{k}x{n}"), fl, || {
            dev.linear(&x, &wb, m, k, n);
        });
    }
    let (nh, nkv, d) = (32, 8, 128);
    for &start in &[0usize, 1536] {
        let q = dev.upload(&vec![0.01f32; m * nh * d]);
        let kv = dev.upload(&vec![0.01f32; (start + m) * nkv * d]);
        let fl = 4.0 * (m * nh * d) as f64 * (start as f64 + m as f64 / 2.0);
        time(&dev, &format!("attention q={m} after {start}"), fl, || {
            dev.kv_attention(&q, &kv, &kv, start, m, nh, nkv, d);
        });
    }
}
