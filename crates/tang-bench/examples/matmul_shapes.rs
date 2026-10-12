//! Hand matmul vs derived matmul at the transformer block's shapes.
use std::time::Instant;
use tang_bench::ir_gpu::{matmul_candidates, Executable};
use tang_gpu::{GpuDevice, GpuTensor, KernelCache};

fn main() {
    let device = GpuDevice::new_sync().expect("gpu");
    let mut cache = KernelCache::new();
    for &(m, k, n) in &[(128usize, 576usize, 1536usize), (128, 1536, 576), (512, 576, 1536), (512, 576, 576)] {
        let a: Vec<f32> = (0..m * k).map(|i| ((i * 7919) % 97) as f32 * 0.01 - 0.5).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i * 104729) % 89) as f32 * 0.01 - 0.4).collect();
        let at = GpuTensor::from_slice(&device, &a, &[m, k]);
        let bt = GpuTensor::from_slice(&device, &b, &[k, n]);
        let _ = tang_gpu::matmul::matmul(&device, &mut cache, &at, &bt).to_vec_flushed(&device, &mut cache);
        let start = Instant::now();
        for _ in 0..20 {
            let _ = tang_gpu::matmul::matmul(&device, &mut cache, &at, &bt).to_vec_flushed(&device, &mut cache);
        }
        let hand = start.elapsed().as_secs_f64() * 1e3 / 20.0;
        let mut line = format!("{m}x{k}x{n}: hand {hand:.3} ms");
        for (name, g, c) in matmul_candidates(m, k, n) {
            let exe = Executable::new(&device, g.to_wgsl(&[c]), &[&a, &b]);
            exe.run(&device, &mut cache);
            cache.flush(&device);
            let _ = exe.read(&device, c);
            let start = Instant::now();
            for _ in 0..20 {
                exe.run(&device, &mut cache);
                cache.flush(&device);
                let _ = exe.read(&device, c);
            }
            line.push_str(&format!("  ir_{name} {:.3} ms", start.elapsed().as_secs_f64() * 1e3 / 20.0));
        }
        println!("{line}");
    }
}
