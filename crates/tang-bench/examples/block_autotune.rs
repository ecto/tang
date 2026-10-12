//! Per-kernel autotune of the derived transformer block, then compare
//! the tuned schedule against `schedule_auto` and the hand path.
use std::time::Instant;
use tang_bench::ir_block::{compiled_block, BlockDims};
use tang_bench::ir_block_gpu::{hand_forward, params};
use tang_bench::ir_gpu::{autotune_kernels, Executable};
use tang_gpu::{GpuDevice, GpuTensor, KernelCache};

fn main() {
    let seq: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(128);
    let dims = BlockDims {
        seq,
        kv_len: seq,
        d_model: 576,
        n_heads: 9,
        n_kv_heads: 3,
        head_dim: 64,
        ff_dim: 1536,
    };
    let device = GpuDevice::new_sync().expect("gpu");
    let mut cache = KernelCache::new();
    let x: Vec<f32> = (0..seq * 576)
        .map(|i| ((i * 7919) % 97) as f32 * 0.01 - 0.5)
        .collect();
    let input = GpuTensor::from_slice(&device, &x, &[seq, 576]);
    let p = params(&device, &mut cache, dims, x);
    let inputs: Vec<&[f32]> = p.cpu.iter().map(|v| v.as_slice()).collect();

    let (g, y) = compiled_block(&p.dims);
    let (tuned, log) = autotune_kernels(&device, &mut cache, g.clone(), &[y], &inputs, 10);
    for k in &log {
        let auto = &g.nodes()[k.node.index()];
        println!(
            "node {:>2}: tuned split {:>3} tile {:?} {:>8.3} ms   (auto: split {} tile {:?})",
            k.node.index(),
            k.split,
            k.tile,
            k.time.as_secs_f64() * 1e3,
            auto.split,
            auto.tile
        );
    }

    let time = |exe: &Executable, cache: &mut KernelCache| {
        exe.run(&device, cache);
        cache.flush(&device);
        let _ = exe.read(&device, y);
        let start = Instant::now();
        for _ in 0..20 {
            exe.run(&device, cache);
            cache.flush(&device);
            let _ = exe.read(&device, y);
        }
        start.elapsed().as_secs_f64() * 1e3 / 20.0
    };
    let auto_exe = Executable::new(&device, g.to_wgsl(&[y]), &inputs);
    let tuned_exe = Executable::new(&device, tuned.to_wgsl(&[y]), &inputs);
    let auto_ms = time(&auto_exe, &mut cache);
    let tuned_ms = time(&tuned_exe, &mut cache);
    let _ = hand_forward(&device, &mut cache, &p, &input).to_vec_flushed(&device, &mut cache);
    let start = Instant::now();
    for _ in 0..20 {
        let _ = hand_forward(&device, &mut cache, &p, &input).to_vec_flushed(&device, &mut cache);
    }
    let hand_ms = start.elapsed().as_secs_f64() * 1e3 / 20.0;
    println!("seq {seq}: hand {hand_ms:.2} ms   ir schedule_auto {auto_ms:.2} ms   ir autotuned {tuned_ms:.2} ms");
}
