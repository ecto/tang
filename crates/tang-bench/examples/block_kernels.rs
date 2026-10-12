//! List and time the kernels the compiled transformer block lowers to.
use std::time::Instant;
use tang_bench::ir_block::{compiled_block, BlockDims};
use tang_bench::ir_block_gpu::params;
use tang_bench::ir_gpu::Executable;
use tang_gpu::{GpuDevice, KernelCache};

fn main() {
    let seq = 128;
    let dims = BlockDims {
        seq,
        kv_len: seq,
        rope: true,
        bias: true,
        d_model: 576,
        n_heads: 9,
        n_kv_heads: 3,
        head_dim: 64,
        ff_dim: 1536,
    };
    let (g, y) = compiled_block(&dims);
    let prog = g.to_wgsl(&[y]);
    let device = GpuDevice::new_sync().expect("gpu");
    let mut cache = KernelCache::new();
    let x: Vec<f32> = (0..seq * 576)
        .map(|i| ((i * 7919) % 97) as f32 * 0.01 - 0.5)
        .collect();
    let p = params(&device, &mut cache, dims, x);
    let inputs: Vec<&[f32]> = p.cpu.iter().map(|v| v.as_slice()).collect();
    let exe = Executable::new(&device, prog, &inputs);
    exe.run(&device, &mut cache);
    cache.flush(&device);
    let _ = exe.read(&device, y);
    let iters = 20;
    let mut total = 0.0;
    for (i, k) in exe.program.kernels.iter().enumerate() {
        let node = &g.nodes()[k.node.index()];
        let t = Instant::now();
        for _ in 0..iters {
            exe.run_kernel(&device, &mut cache, i);
        }
        cache.flush(&device);
        let _ = exe.read(&device, k.node);
        let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
        total += ms;
        println!(
            "{ms:7.3} ms  node {:>2} shape {:?} split {} tile {:?} reads {}",
            k.node.index(),
            node.shape.iter().map(|d| d.n()).collect::<Vec<_>>(),
            node.split,
            node.tile,
            k.reads.len()
        );
    }
    println!(
        "{total:7.3} ms  total over {} kernels (each incl. one readback)",
        exe.program.kernels.len()
    );
}
