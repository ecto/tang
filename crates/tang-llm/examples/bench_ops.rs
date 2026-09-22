//! Per-kernel decode timings on Metal (Qwen3-0.6B shapes).
use std::time::Instant;
use tang_compute::{ComputeDevice, MetalDevice};

fn time(dev: &MetalDevice, name: &str, bytes: usize, reps: usize, f: impl Fn()) {
    f();
    dev.sync();
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    dev.sync();
    let per = t.elapsed().as_secs_f64() / reps as f64;
    println!(
        "{name:<28} {:>8.1} µs  {:>6.0} GB/s",
        per * 1e6,
        bytes as f64 / per / 1e9
    );
}

fn main() {
    let dev = MetalDevice::new().unwrap();
    let reps = 200;
    for &(k, n) in &[(1024, 3072), (3072, 1024), (1024, 151936), (2560, 9728)] {
        let x = dev.upload(&vec![0.1f32; k]);
        let wb = dev.upload_bf16(&vec![0x3f80u16; k * n]);
        let wf = dev.upload(&vec![1.0f32; k * n]);
        time(&dev, &format!("gemv bf16 {k}x{n}"), k * n * 2, reps, || {
            dev.linear(&x, &wb, 1, k, n);
        });
        time(&dev, &format!("gemv f32  {k}x{n}"), k * n * 4, reps, || {
            dev.linear(&x, &wf, 1, k, n);
        });
    }
    let h = dev.upload(&vec![0.1f32; 1024]);
    let w = dev.upload(&vec![1.0f32; 1024]);
    time(&dev, "rms_norm 1024", 8192, reps, || {
        dev.rms_norm(&h, &w, 1, 1024, 1e-6);
    });
    time(&dev, "add 1024", 12288, reps, || {
        dev.add_tensors_buf(&h, &h, 1024);
    });
    let q = dev.upload(&vec![0.1f32; 2048]);
    for &len in &[64usize, 512, 2048] {
        let kc = dev.upload(&vec![0.1f32; len * 1024]);
        time(
            &dev,
            &format!("kv_attention len {len}"),
            len * 1024 * 8,
            reps,
            || {
                dev.kv_attention(&q, &kc, &kc, len - 1, 1, 16, 8, 128);
            },
        );
    }
    let t = dev.upload(&vec![0.0f32; 4096 * 64]);
    time(&dev, "rope q", 0, reps, || {
        dev.rope_half_cached(&q, &t, &t, 1, 16, 128, 5);
    });
}
