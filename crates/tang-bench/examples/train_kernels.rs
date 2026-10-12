//! The block's forward and training-step executables, stage by stage:
//! first-run (shader compile) times, the kernel list, then every kernel
//! of the training step timed on its own. `block_kernels.rs` is the
//! forward-only version. Usage: `train_kernels [seq]`.
use std::time::Instant;
use tang_bench::ir_block::{train_loss_graph, BlockDims};
use tang_bench::ir_block_gpu::{hand_forward, ir_executable, params};
use tang_bench::ir_gpu::Executable;
use tang_gpu::{GpuDevice, GpuTensor, KernelCache};

fn rand(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = tang_bench::Rng::new(seed);
    (0..n).map(|_| rng.f32() * 2.0 - 1.0).collect()
}

fn main() {
    let seq: usize = std::env::args().nth(1).map_or(128, |s| s.parse().unwrap());
    let device = GpuDevice::new_sync().expect("gpu");
    let mut cache = KernelCache::new();
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
    let x = rand(seq * 576, 7);
    let input = GpuTensor::from_slice(&device, &x, &[seq, 576]);
    let p = params(&device, &mut cache, dims, x);
    let t = Instant::now();
    let h = hand_forward(&device, &mut cache, &p, &input).to_vec_flushed(&device, &mut cache);
    eprintln!("hand fwd ok {:?} ({} vals)", t.elapsed(), h.len());
    let (exe, y) = ir_executable(&device, &p, true);
    eprintln!("ir fwd: {} kernels", exe.program.kernels.len());
    let t = Instant::now();
    exe.run(&device, &mut cache);
    cache.flush(&device);
    let v = exe.read(&device, y);
    eprintln!("ir fwd ok {:?} ({} vals)", t.elapsed(), v.len());
    let (g, outs, _) = train_loss_graph(&p.dims);
    let (g, outs) = g.stabilize_softmax(&outs);
    let (g, outs) = g.substitute_maps(&outs);
    let (g, outs) = g.eliminate_common(&outs);
    let (mut g, outs) = g.merge_siblings(&outs);
    g.schedule_auto(&outs);
    let target = vec![0.0f32; seq * 576];
    let mut inputs: Vec<&[f32]> = p.cpu.iter().map(|v| v.as_slice()).collect();
    inputs.push(&target);
    let prog = g.to_wgsl(&outs);
    eprintln!("train: {} kernels", prog.kernels.len());
    for (i, k) in prog.kernels.iter().enumerate() {
        eprintln!(
            "  k{i} node {:?} points {} count {} wg {}",
            k.node, k.points, k.count, k.workgroup_size
        );
    }
    let train = Executable::new(&device, prog, &inputs);
    let t = Instant::now();
    train.run(&device, &mut cache);
    cache.flush(&device);
    let v = train.read(&device, outs[0]);
    eprintln!("train ok {:?} ({} vals)", t.elapsed(), v.len());
    // per kernel: run it 3x, sync, report
    for i in 0..train.program.kernels.len() {
        let k = &train.program.kernels[i];
        let t = Instant::now();
        for _ in 0..3 {
            train.run_kernel(&device, &mut cache, i);
        }
        cache.flush(&device);
        let _ = train.read(&device, k.node);
        let ms = t.elapsed().as_secs_f64() * 1e3 / 3.0;
        if ms > 0.5 {
            let node = &g.nodes()[k.node.index()];
            eprintln!(
                "  SLOW k{i} {ms:.2} ms node {:?} shape {:?} split {} tile {:?} reg {:?}",
                k.node, node.shape, node.split, node.tile, node.reg
            );
        }
    }
}
