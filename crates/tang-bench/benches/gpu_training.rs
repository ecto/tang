//! Benchmarks for tang-gpu training pipeline.
//!
//! Measures: matmul, fused kernels, linear forward/backward, sequential
//! forward/backward, training step, and full XOR training.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tang_gpu::*;

fn get_device() -> GpuDevice {
    GpuDevice::new_sync().expect("GPU device required for benchmarks")
}

fn bench_matmul(c: &mut Criterion) {
    let device = get_device();
    let mut cache = KernelCache::new();
    let mut group = c.benchmark_group("gpu_matmul");

    for &n in &[16, 32, 64, 128] {
        let data_a: Vec<f32> = (0..n * n).map(|i| (i as f32) * 0.001).collect();
        let data_b: Vec<f32> = (0..n * n).map(|i| (i as f32) * 0.002).collect();
        let a = GpuTensor::from_slice(&device, &data_a, &[n, n]);
        let b = GpuTensor::from_slice(&device, &data_b, &[n, n]);

        group.bench_function(format!("{n}x{n}"), |bench| {
            bench.iter(|| {
                let c = tang_gpu::matmul::matmul(&device, &mut cache, &a, &b);
                black_box(c);
            });
        });
    }
    group.finish();
}

fn bench_fused_elementwise(c: &mut Criterion) {
    let device = get_device();
    let mut cache = KernelCache::new();
    let mut group = c.benchmark_group("gpu_fused_elementwise");

    for &n in &[256, 1024, 4096] {
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let a = GpuTensor::from_slice(&device, &data, &[n]);
        let b = GpuTensor::from_slice(&device, &data, &[n]);

        group.bench_function(format!("add_sq_{n}"), |bench| {
            bench.iter(|| {
                let c = map_elementwise(&device, &mut cache, &[&a, &b], |args| {
                    let sum = args[0] + args[1];
                    sum * sum
                });
                black_box(c);
            });
        });
    }
    group.finish();
}

fn bench_linear_forward(c: &mut Criterion) {
    let device = get_device();
    let mut cache = KernelCache::new();

    c.bench_function("gpu_linear_forward_128x64", |bench| {
        let mut linear = GpuLinear::kaiming(&device, 128, 64, 42);
        let input = GpuTensor::from_slice(
            &device,
            &vec![0.1f32; 4 * 128],
            &[4, 128],
        );
        bench.iter(|| {
            let out = linear.forward_train(&device, &mut cache, &input);
            black_box(out);
        });
    });
}

fn bench_linear_backward(c: &mut Criterion) {
    let device = get_device();
    let mut cache = KernelCache::new();

    c.bench_function("gpu_linear_backward_128x64", |bench| {
        let mut linear = GpuLinear::kaiming(&device, 128, 64, 42);
        let input = GpuTensor::from_slice(
            &device,
            &vec![0.1f32; 4 * 128],
            &[4, 128],
        );
        // Forward first to cache input
        let _ = linear.forward_train(&device, &mut cache, &input);
        let grad_out = GpuTensor::from_slice(
            &device,
            &vec![0.01f32; 4 * 64],
            &[4, 64],
        );
        bench.iter(|| {
            let gi = linear.backward(&device, &mut cache, &grad_out);
            black_box(gi);
        });
    });
}

fn bench_sequential_forward_backward(c: &mut Criterion) {
    let device = get_device();
    let mut cache = KernelCache::new();

    c.bench_function("gpu_sequential_2_8_1_fwd_bwd", |bench| {
        let mut model = GpuSequential::new(vec![
            Box::new(GpuLinear::kaiming(&device, 2, 8, 42)),
            Box::new(GpuReLULayer::new()),
            Box::new(GpuLinear::kaiming(&device, 8, 1, 43)),
        ]);
        let input = GpuTensor::from_slice(&device, &[1.0, 0.5, 0.3, 0.7], &[2, 2]);
        let grad = GpuTensor::from_slice(&device, &[1.0, 1.0], &[2, 1]);

        bench.iter(|| {
            model.zero_grad();
            let out = model.forward_train(&device, &mut cache, &input);
            let gi = model.backward(&device, &mut cache, &grad);
            black_box((out, gi));
        });
    });
}

fn bench_training_step(c: &mut Criterion) {
    let device = get_device();
    let mut cache = KernelCache::new();

    c.bench_function("gpu_training_step_xor", |bench| {
        let mut model = GpuSequential::new(vec![
            Box::new(GpuLinear::kaiming(&device, 2, 8, 42)),
            Box::new(GpuReLULayer::new()),
            Box::new(GpuLinear::kaiming(&device, 8, 1, 43)),
        ]);
        let mut loader = GpuDataLoader::new(
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            vec![0.0, 1.0, 1.0, 0.0],
            2, 1, 4,
        );
        let mut trainer = GpuTrainer::new(0.01, 1);

        bench.iter(|| {
            let losses = trainer.fit(&device, &mut cache, &mut model, &mut loader);
            black_box(losses);
        });
    });
}

fn bench_mse_loss(c: &mut Criterion) {
    let device = get_device();
    let mut cache = KernelCache::new();

    c.bench_function("gpu_mse_loss_64", |bench| {
        let pred = GpuTensor::from_slice(&device, &vec![0.5f32; 64], &[64]);
        let target = GpuTensor::from_slice(&device, &vec![1.0f32; 64], &[64]);
        bench.iter(|| {
            let (loss, grad) = gpu_mse_loss(&device, &mut cache, &pred, &target);
            black_box((loss, grad));
        });
    });
}

criterion_group!(
    benches,
    bench_matmul,
    bench_fused_elementwise,
    bench_linear_forward,
    bench_linear_backward,
    bench_sequential_forward_backward,
    bench_training_step,
    bench_mse_loss,
);
criterion_main!(benches);
