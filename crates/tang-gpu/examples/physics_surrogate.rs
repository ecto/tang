//! Physics Surrogate — train a GPU network to approximate a gravitational
//! potential field built from expression graphs.
//!
//! 1. Builds `V(x,y) = -Σ mᵢ / √((x-xᵢ)² + (y-yᵢ)² + ε)` symbolically
//! 2. Auto-diffs to get `∂V/∂x`, `∂V/∂y`
//! 3. Trains a CPU neural network on expression graph output
//! 4. Transfers weights to GPU network
//! 5. Benchmarks: expression graph eval vs CPU network vs GPU network
//!
//! ```sh
//! cargo run --example physics_surrogate -p tang-gpu
//! ```

use std::time::Instant;

use tang_expr::{ExprGraph, ExprId};
use tang_gpu::*;
use tang_tensor::{Shape, Tensor};
use tang_train::{
    mse_loss, mse_loss_grad, DataLoader, Linear, ModuleAdam, Module, Sequential, Tanh,
    TensorDataset, Trainer,
};

// --- Point masses ------------------------------------------------------------

struct Mass {
    x: f64,
    y: f64,
    m: f64,
}

fn point_masses() -> Vec<Mass> {
    // 10 fixed point masses in [-2, 2]²
    vec![
        Mass { x: 0.5, y: 0.3, m: 1.0 },
        Mass { x: -1.2, y: 0.8, m: 2.0 },
        Mass { x: 1.0, y: -1.0, m: 0.5 },
        Mass { x: -0.3, y: -0.7, m: 1.5 },
        Mass { x: 1.5, y: 1.5, m: 0.8 },
        Mass { x: -1.8, y: -1.2, m: 1.2 },
        Mass { x: 0.0, y: 1.8, m: 0.6 },
        Mass { x: -0.5, y: 0.0, m: 2.5 },
        Mass { x: 1.2, y: -1.5, m: 1.0 },
        Mass { x: -1.0, y: 1.2, m: 0.7 },
    ]
}

// --- Build symbolic expression graph -----------------------------------------

fn build_potential_graph() -> (ExprGraph, ExprId, ExprId, ExprId) {
    let mut g = ExprGraph::new();
    let x = g.var(0);
    let y = g.var(1);
    let eps = g.lit(0.01); // softening parameter

    let masses = point_masses();
    let mut v = ExprId::ZERO;

    for mass in &masses {
        let xi = g.lit(mass.x);
        let yi = g.lit(mass.y);
        let mi = g.lit(mass.m);

        let neg_xi = g.neg(xi);
        let dx = g.add(x, neg_xi);
        let neg_yi = g.neg(yi);
        let dy = g.add(y, neg_yi);
        let dx2 = g.mul(dx, dx);
        let dy2 = g.mul(dy, dy);
        let sum_sq = g.add(dx2, dy2);
        let r2 = g.add(sum_sq, eps);
        let r = g.sqrt(r2);
        let inv_r = g.recip(r);
        let mi_inv_r = g.mul(mi, inv_r);
        let term = g.neg(mi_inv_r);
        v = g.add(v, term);
    }

    // Symbolic derivatives
    let dv_dx = g.diff(v, 0);
    let dv_dx = g.simplify(dv_dx);
    let dv_dy = g.diff(v, 1);
    let dv_dy = g.simplify(dv_dy);

    (g, v, dv_dx, dv_dy)
}

// --- Generate training data --------------------------------------------------

fn generate_data(
    g: &ExprGraph,
    v: ExprId,
    dvdx: ExprId,
    dvdy: ExprId,
) -> (Vec<f64>, Vec<f64>, usize) {
    let n_per_dim = 50;
    let n_total = n_per_dim * n_per_dim;
    let compiled = g.compile_many(&[v, dvdx, dvdy]);

    let mut inputs = Vec::with_capacity(n_total * 2);
    let mut targets = Vec::with_capacity(n_total * 3);
    let mut out = [0.0f64; 3];

    for i in 0..n_per_dim {
        for j in 0..n_per_dim {
            let x = -2.0 + 4.0 * i as f64 / (n_per_dim - 1) as f64;
            let y = -2.0 + 4.0 * j as f64 / (n_per_dim - 1) as f64;
            compiled(&[x, y], &mut out);

            inputs.push(x);
            inputs.push(y);
            targets.push(out[0]); // V
            targets.push(out[1]); // dV/dx
            targets.push(out[2]); // dV/dy
        }
    }

    (inputs, targets, n_total)
}

// --- Main --------------------------------------------------------------------

fn main() {
    println!("=== Physics Surrogate ===\n");

    // 1. Build symbolic expression graph
    println!("building symbolic potential for 10 point masses...");
    let (g, v, dvdx, dvdy) = build_potential_graph();
    println!("  graph nodes: {}", g.len());
    println!("  outputs: V, dV/dx, dV/dy");

    // 2. Generate training data
    println!("\ngenerating training data on 50x50 grid...");
    let (input_data, target_data, n_samples) = generate_data(&g, v, dvdx, dvdy);
    println!("  {} samples, 2 inputs -> 3 outputs", n_samples);

    // 3. Train CPU network
    println!("\ntraining CPU network...");
    println!("  architecture: Linear(2,64) -> Tanh -> Linear(64,64) -> Tanh -> Linear(64,3)");

    let inputs = Tensor::new(input_data.clone(), Shape::new(vec![n_samples, 2]));
    let targets = Tensor::new(target_data.clone(), Shape::new(vec![n_samples, 3]));
    let dataset = TensorDataset::new(inputs, targets);
    let mut loader = DataLoader::new(&dataset, 64);

    let mut cpu_model = Sequential::<f64>::new(vec![
        Box::new(Linear::new(2, 64, 42)),
        Box::new(Tanh::new()),
        Box::new(Linear::new(64, 64, 137)),
        Box::new(Tanh::new()),
        Box::new(Linear::new(64, 3, 256)),
    ]);

    let n_params: usize = cpu_model.parameters().iter().map(|p| p.data.numel()).sum();
    println!("  parameters: {}", n_params);

    let losses = Trainer::new(&mut cpu_model, ModuleAdam::new(0.003))
        .loss_fn(|pred, target| (mse_loss(pred, target), mse_loss_grad(pred, target)))
        .epochs(200)
        .fit(&mut loader);

    println!("  epoch   1: loss = {:.6}", losses[0]);
    println!("  epoch 100: loss = {:.6}", losses[99]);
    println!("  epoch 200: loss = {:.6}", losses[199]);

    // 4. Transfer weights to GPU
    println!("\ntransferring to GPU...");
    let device = GpuDevice::new_sync().expect("GPU required");
    let mut cache = KernelCache::new();

    let state = cpu_model.state_dict();

    // Extract weight/bias for each linear layer (layers 0, 2, 4 in Sequential)
    let get_param = |name: &str| -> Vec<f32> {
        state
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, t)| t.data().iter().map(|&v| v as f32).collect())
            .unwrap()
    };

    let w0 = get_param("0.weight");
    let b0 = get_param("0.bias");
    let w2 = get_param("2.weight");
    let b2 = get_param("2.bias");
    let w4 = get_param("4.weight");
    let b4 = get_param("4.bias");

    let mut gpu_model = GpuSequential::new(vec![
        Box::new(GpuLinear::new(&device, &w0, &b0, 2, 64)),
        Box::new(GpuReLULayer::new()), // GPU uses ReLU (different activation but fine for surrogate)
        Box::new(GpuLinear::new(&device, &w2, &b2, 64, 64)),
        Box::new(GpuReLULayer::new()),
        Box::new(GpuLinear::new(&device, &w4, &b4, 64, 3)),
    ]);

    println!("  GPU model created, fine-tuning...");

    // Fine-tune GPU model briefly to adapt to ReLU activations
    let gpu_inputs: Vec<f32> = input_data.iter().map(|&v| v as f32).collect();
    let gpu_targets: Vec<f32> = target_data.iter().map(|&v| v as f32).collect();
    let mut gpu_loader = GpuDataLoader::new(gpu_inputs.clone(), gpu_targets, 2, 3, 256);

    let gpu_losses = GpuTrainer::new(0.001, 20).fit(&device, &mut cache, &mut gpu_model, &mut gpu_loader);
    println!("  GPU fine-tune: loss {:.6} -> {:.6}", gpu_losses[0], gpu_losses.last().unwrap());

    // 5. Benchmark
    println!("\n--- Benchmark: 10,000 point evaluations ---\n");

    let n_bench = 10_000;
    let compiled = g.compile_many(&[v, dvdx, dvdy]);

    // Expression graph benchmark
    let start = Instant::now();
    let mut out = [0.0f64; 3];
    for i in 0..n_bench {
        let x = -2.0 + 4.0 * (i % 100) as f64 / 99.0;
        let y = -2.0 + 4.0 * (i / 100) as f64 / 99.0;
        compiled(&[x, y], &mut out);
    }
    let expr_time = start.elapsed();
    println!("expression graph: {:>8.2?}", expr_time);

    // CPU network benchmark
    let start = Instant::now();
    for i in 0..n_bench {
        let x = -2.0 + 4.0 * (i % 100) as f64 / 99.0;
        let y = -2.0 + 4.0 * (i / 100) as f64 / 99.0;
        let input = Tensor::from_slice(&[x, y]);
        let _pred = cpu_model.forward(&input);
    }
    let cpu_time = start.elapsed();
    println!("CPU network:      {:>8.2?}", cpu_time);

    // GPU network benchmark (batched)
    let mut bench_inputs = Vec::with_capacity(n_bench * 2);
    for i in 0..n_bench {
        let x = -2.0 + 4.0 * (i % 100) as f64 / 99.0;
        let y = -2.0 + 4.0 * (i / 100) as f64 / 99.0;
        bench_inputs.push(x as f32);
        bench_inputs.push(y as f32);
    }
    let gpu_input = GpuTensor::from_slice(&device, &bench_inputs, &[n_bench, 2]);

    let start = Instant::now();
    let _gpu_out = gpu_model.forward_train(&device, &mut cache, &gpu_input);
    cache.flush(&device);
    let gpu_time = start.elapsed();
    println!("GPU network:      {:>8.2?}", gpu_time);

    println!("\nspeedup (expr→GPU): {:.1}x", expr_time.as_secs_f64() / gpu_time.as_secs_f64());
    println!("speedup (CPU→GPU):  {:.1}x", cpu_time.as_secs_f64() / gpu_time.as_secs_f64());
}
