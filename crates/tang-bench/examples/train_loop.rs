//! A training loop that is entirely derived: block forward + MSE loss +
//! backward for every weight as one program, Adam as another, weights
//! fed back GPU-side. Prints the loss going down.
use tang_bench::ir_block::{train_loss_graph, BlockDims};
use tang_bench::ir_block_gpu::params;
use tang_bench::ir_gpu::{adam_step, Executable};
use tang_compile::tensor::TensorId;
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
    let p = params(&device, &mut cache, dims, x);

    let (g, outs, target_idx) = train_loss_graph(&p.dims);
    let (g, outs) = g.substitute_maps(&outs);
    let (g, outs) = g.eliminate_common(&outs);
    let (mut g, outs) = g.merge_siblings(&outs);
    g.schedule_auto(&outs);
    let program = g.to_wgsl(&outs);
    println!("train program: {} kernels", program.kernels.len());
    let mut inputs: Vec<&[f32]> = p.cpu.iter().map(|v| v.as_slice()).collect();
    inputs.push(&target);
    assert_eq!(inputs.len(), target_idx + 1);
    let train = Executable::new(&device, program, &inputs);

    // one Adam program per weight, with its own m/v state
    let weight_inputs = [1usize, 3, 5, 7, 9, 11, 13];
    let (lr, b1, b2, eps) = (1e-3f64, 0.9f64, 0.999f64, 1e-8f64);
    let adams: Vec<(usize, Executable, [TensorId; 3])> = weight_inputs
        .iter()
        .map(|&k| {
            let len = p.cpu[k].len();
            let (ag, aouts) = adam_step(len, lr, b1, b2, eps);
            let zeros = vec![0.0f32; len];
            let bc = [1.0f32, 1.0];
            let exe = Executable::new(
                &device,
                ag.to_wgsl(&aouts),
                &[&p.cpu[k], &zeros, &zeros, &zeros, &bc],
            );
            (k, exe, aouts)
        })
        .collect();

    for step in 1..=steps {
        train.run(&device, &mut cache);
        cache.flush(&device);
        let loss = train.read(&device, outs[0])[0];
        println!("step {step:>3}: loss {loss:.6}");
        let bc = [
            (1.0 - b1.powi(step as i32)) as f32,
            (1.0 - b2.powi(step as i32)) as f32,
        ];
        for (j, (k, adam, aouts)) in adams.iter().enumerate() {
            // adam inputs: p, g, m, v, bc ; outputs m', v', p'
            adam.input_buffer(4).write(&device, &bc);
            adam.input_buffer(0)
                .copy_from(&device, &mut cache, train.input_buffer(*k));
            adam.input_buffer(1)
                .copy_from(&device, &mut cache, train.buffer(outs[1 + j]));
            adam.run(&device, &mut cache);
            adam.input_buffer(2)
                .copy_from(&device, &mut cache, adam.buffer(aouts[0]));
            adam.input_buffer(3)
                .copy_from(&device, &mut cache, adam.buffer(aouts[1]));
            train
                .input_buffer(*k)
                .copy_from(&device, &mut cache, adam.buffer(aouts[2]));
        }
        cache.flush(&device);
    }
}
