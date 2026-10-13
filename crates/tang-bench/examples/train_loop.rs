//! A training loop that is entirely derived: block forward + MSE loss +
//! backward for every weight as one program, Adam as another, weights
//! fed back device-side. Prints the loss going down. Runs on tang-gpu's
//! selected backend: CUDA with `--features cuda` on NVIDIA, else wgpu.
use tang_bench::ir_block::{train_loss_graph, BlockDims};
use tang_bench::ir_block_gpu::params;
use tang_bench::ir_gpu::adam_step;
use tang_compile::tensor::TensorId;
use tang_gpu::program::{Program, Runtime};
use tang_gpu::{GpuDevice, KernelCache};

fn main() {
    let steps: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(20);
    let dims = BlockDims {
        seq: 32,
        kv_len: 32,
        rope: true,
        bias: true,
        d_model: 128,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 32,
        ff_dim: 256,
    };
    let device = GpuDevice::new_sync().expect("gpu");
    let mut cache = KernelCache::new();
    let n = dims.seq * dims.d_model;
    let x: Vec<f32> = (0..n)
        .map(|i| ((i * 7919) % 97) as f32 * 0.02 - 1.0)
        .collect();
    let target: Vec<f32> = (0..n)
        .map(|i| ((i * 104729) % 89) as f32 * 0.02 - 0.9)
        .collect();
    // the weights are initialized by tang-gpu's hand layers (wgpu) and
    // read back; the training itself runs on the selected backend
    let p = params(&device, &mut cache, dims, x);
    drop((device, cache));
    let rt = Runtime::select().expect("gpu");
    println!("backend: {:?} — {}", rt.kind(), rt.name());

    let (g, outs, target_idx) = train_loss_graph(&p.dims);
    let (g, outs) = g.substitute_maps(&outs);
    let (g, outs) = g.eliminate_common(&outs);
    let (g, outs) = g.merge_siblings(&outs);
    let (g, outs) = g.relayout(&outs);
    let (g, outs) = g.merge_reduce_dims(&outs);
    let (mut g, outs) = g.split_reduce_dims(&outs);
    g.schedule_auto(&outs);
    let program = rt.lower(&g, &outs);
    println!("train program: {} kernels", program.kernels.len());
    let mut inputs: Vec<&[f32]> = p.cpu.iter().map(|v| v.as_slice()).collect();
    inputs.push(&target);
    assert_eq!(inputs.len(), target_idx + 1);
    let train = Program::new(&rt, program, &inputs);

    // one Adam program per weight, with its own m/v state
    let weight_inputs = [1usize, 3, 5, 7, 9, 11, 13];
    let (lr, b1, b2, eps) = (1e-3f64, 0.9f64, 0.999f64, 1e-8f64);
    let adams: Vec<(usize, Program, [TensorId; 3])> = weight_inputs
        .iter()
        .map(|&k| {
            let len = p.cpu[k].len();
            let (ag, aouts) = adam_step(len, lr, b1, b2, eps);
            let zeros = vec![0.0f32; len];
            let bc = [1.0f32, 1.0];
            let exe = Program::compile(&rt, &ag, &aouts, &[&p.cpu[k], &zeros, &zeros, &zeros, &bc]);
            (k, exe, aouts)
        })
        .collect();

    for step in 1..=steps {
        train.run(&rt);
        let loss = train.read(&rt, outs[0])[0];
        println!(
            "step {step:>3}: loss {loss:.6} (bits {:08x})",
            loss.to_bits()
        );
        let bc = [
            (1.0 - b1.powi(step as i32)) as f32,
            (1.0 - b2.powi(step as i32)) as f32,
        ];
        for (j, (k, adam, aouts)) in adams.iter().enumerate() {
            // adam inputs: p, g, m, v, bc ; outputs m', v', p'
            adam.write(&rt, adam.input(4), &bc);
            adam.copy(&rt, adam.input(0), &train, train.input(*k));
            adam.copy(&rt, adam.input(1), &train, outs[1 + j]);
            adam.run(&rt);
            adam.copy(&rt, adam.input(2), adam, aouts[0]);
            adam.copy(&rt, adam.input(3), adam, aouts[1]);
            train.copy(&rt, train.input(*k), adam, aouts[2]);
        }
    }
}
