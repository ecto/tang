//! Benchmarks for tang-train CPU training primitives.
//!
//! Measures: Linear, SwiGLU, GQA, RMSNorm, RoPE, sequence cross-entropy,
//! optimizer step, and full training step.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use tang_tensor::{Shape, Tensor};
use tang_train::*;

// ---------------------------------------------------------------------------
// Linear
// ---------------------------------------------------------------------------

fn bench_linear_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_linear_forward");
    for &(batch, d_in, d_out) in &[(4, 128, 64), (16, 576, 576), (1, 576, 1536)] {
        let mut linear = Linear::<f64>::new(d_in, d_out, 42);
        let input = Tensor::from_fn(Shape::from_slice(&[batch, d_in]), |idx| {
            ((idx[0] * d_in + idx[1]) as f64) * 0.001
        });
        group.bench_with_input(
            BenchmarkId::new("fwd", format!("{batch}x{d_in}->{d_out}")),
            &(),
            |b, _| {
                b.iter(|| black_box(linear.forward(&input)));
            },
        );
    }
    group.finish();
}

fn bench_linear_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_linear_backward");
    let d_in = 576;
    let d_out = 576;
    let batch = 16;
    let mut linear = Linear::<f64>::new(d_in, d_out, 42);
    let input = Tensor::from_fn(Shape::from_slice(&[batch, d_in]), |idx| {
        ((idx[0] * d_in + idx[1]) as f64) * 0.001
    });
    let _ = linear.forward(&input);
    let grad_out = Tensor::from_fn(Shape::from_slice(&[batch, d_out]), |_| 0.01);
    group.bench_function(BenchmarkId::new("bwd", format!("{batch}x{d_in}->{d_out}")), |b| {
        b.iter(|| black_box(linear.backward(&grad_out)));
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// SwiGLU
// ---------------------------------------------------------------------------

fn bench_swiglu(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_swiglu");
    for &(batch, d_model, ff_dim) in &[(4, 64, 128), (16, 576, 1536)] {
        let mut swiglu = SwiGLU::<f64>::new(d_model, ff_dim, 42);
        let input = Tensor::from_fn(Shape::from_slice(&[batch, d_model]), |idx| {
            ((idx[0] * d_model + idx[1]) as f64) * 0.001
        });
        group.bench_with_input(
            BenchmarkId::new("fwd", format!("{batch}x{d_model}")),
            &(),
            |b, _| {
                b.iter(|| black_box(swiglu.forward(&input)));
            },
        );

        let output = swiglu.forward(&input);
        let grad_out = Tensor::from_fn(output.shape().clone(), |_| 0.01);
        group.bench_with_input(
            BenchmarkId::new("bwd", format!("{batch}x{d_model}")),
            &(),
            |b, _| {
                let _ = swiglu.forward(&input);
                b.iter(|| black_box(swiglu.backward(&grad_out)));
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Grouped-Query Attention
// ---------------------------------------------------------------------------

fn bench_gqa(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_gqa");
    for &(seq, d_model, n_heads, n_kv) in &[(8, 64, 4, 2), (32, 64, 4, 2)] {
        let mut gqa = GroupedQueryAttention::<f64>::new(d_model, n_heads, n_kv, 42)
            .with_causal(true);
        let input = Tensor::from_fn(Shape::from_slice(&[seq, d_model]), |idx| {
            ((idx[0] * d_model + idx[1]) as f64) * 0.001
        });
        group.bench_with_input(
            BenchmarkId::new("fwd", format!("seq{seq}_d{d_model}_h{n_heads}")),
            &(),
            |b, _| {
                b.iter(|| black_box(gqa.forward(&input)));
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// RMSNorm + RoPE
// ---------------------------------------------------------------------------

fn bench_rmsnorm(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_rmsnorm");
    for &(batch, dim) in &[(16, 64), (16, 576)] {
        let mut norm = RMSNorm::<f64>::new(dim);
        let input = Tensor::from_fn(Shape::from_slice(&[batch, dim]), |idx| {
            ((idx[0] * dim + idx[1]) as f64) * 0.01
        });
        group.bench_function(BenchmarkId::new("fwd", format!("{batch}x{dim}")), |b| {
            b.iter(|| black_box(norm.forward(&input)));
        });
    }
    group.finish();
}

fn bench_rope(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_rope");
    for &(seq, dim) in &[(32, 64), (128, 64)] {
        let rope = RotaryEmbedding::<f64>::with_base(dim, 512, 10000.0);
        let input = Tensor::from_fn(Shape::from_slice(&[seq, dim]), |idx| {
            ((idx[0] * dim + idx[1]) as f64) * 0.001
        });
        group.bench_function(
            BenchmarkId::new("apply", format!("seq{seq}_d{dim}")),
            |b| {
                b.iter(|| black_box(rope.apply(&input, 0)));
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Loss functions
// ---------------------------------------------------------------------------

fn bench_cross_entropy(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_loss");
    for &(seq, vocab) in &[(32, 256), (128, 256)] {
        let logits = Tensor::from_fn(Shape::from_slice(&[seq, vocab]), |idx| {
            ((idx[0] * vocab + idx[1]) as f64) * 0.001
        });
        let targets = Tensor::from_fn(Shape::from_slice(&[seq]), |idx| {
            (idx[0] % vocab) as f64
        });
        group.bench_function(
            BenchmarkId::new("seq_ce", format!("seq{seq}_v{vocab}")),
            |b| {
                b.iter(|| black_box(sequence_cross_entropy(&logits, &targets, 0)));
            },
        );
        group.bench_function(
            BenchmarkId::new("seq_ce_grad", format!("seq{seq}_v{vocab}")),
            |b| {
                b.iter(|| black_box(sequence_cross_entropy_grad(&logits, &targets, 0)));
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// KV Cache
// ---------------------------------------------------------------------------

fn bench_kv_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_kv_cache");
    let num_layers = 4;
    let max_seq = 512;
    let n_kv_heads = 2;
    let head_dim = 64;
    let kv_dim = n_kv_heads * head_dim;

    // Prefill: append 128 tokens at once
    group.bench_function("prefill_128tok", |b| {
        b.iter(|| {
            let mut cache = tang_infer::KVCache::<f64>::new(num_layers, max_seq, n_kv_heads, head_dim);
            let k = Tensor::new(vec![0.1; 128 * kv_dim], Shape::from_slice(&[128, kv_dim]));
            let v = Tensor::new(vec![0.2; 128 * kv_dim], Shape::from_slice(&[128, kv_dim]));
            for layer in 0..num_layers {
                cache.append(layer, &k, &v);
            }
            black_box(&cache);
        });
    });

    // Decode: append 1 token, 128 times
    group.bench_function("decode_128steps", |b| {
        b.iter(|| {
            let mut cache = tang_infer::KVCache::<f64>::new(num_layers, max_seq, n_kv_heads, head_dim);
            for step in 0..128 {
                let k = Tensor::new(vec![step as f64 * 0.01; kv_dim], Shape::from_slice(&[1, kv_dim]));
                let v = Tensor::new(vec![step as f64 * 0.02; kv_dim], Shape::from_slice(&[1, kv_dim]));
                for layer in 0..num_layers {
                    cache.append(layer, &k, &v);
                }
            }
            black_box(&cache);
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

fn bench_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_sampling");
    let vocab = 256;
    let logits = Tensor::from_fn(Shape::from_slice(&[vocab]), |idx| {
        (idx[0] as f64 - 128.0) * 0.05
    });

    group.bench_function("greedy_v256", |b| {
        let mut sampler = tang_infer::Sampler::new(tang_infer::SamplingConfig::greedy());
        b.iter(|| black_box(sampler.sample(&logits, &[])));
    });
    group.bench_function("topk40_v256", |b| {
        let mut sampler = tang_infer::Sampler::with_seed(tang_infer::SamplingConfig::standard(), 42);
        b.iter(|| black_box(sampler.sample(&logits, &[])));
    });
    group.bench_function("topk40_rep_pen_v256", |b| {
        let config = tang_infer::SamplingConfig {
            repetition_penalty: 1.2,
            ..tang_infer::SamplingConfig::standard()
        };
        let mut sampler = tang_infer::Sampler::with_seed(config, 42);
        let past: Vec<usize> = (0..50).collect();
        b.iter(|| black_box(sampler.sample(&logits, &past)));
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// Quantization
// ---------------------------------------------------------------------------

fn bench_quantize(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_quantize");
    let dim = 576;
    let data: Vec<f32> = (0..dim).map(|i| (i as f32 - 288.0) * 0.01).collect();

    group.bench_function("q8_roundtrip_576", |b| {
        b.iter(|| {
            let blocks = tang_infer::quantize_q8(&data);
            let out = tang_infer::dequantize_q8(&blocks);
            black_box(out);
        });
    });
    group.bench_function("q4_roundtrip_576", |b| {
        b.iter(|| {
            let blocks = tang_infer::quantize_q4(&data);
            let out = tang_infer::dequantize_q4(&blocks);
            black_box(out);
        });
    });

    // Q8 matrix-vector product
    let rows = 576;
    let cols = 576;
    let mat: Vec<f32> = (0..rows * cols).map(|i| ((i % 1000) as f32 - 500.0) * 0.001).collect();
    let vec_data: Vec<f64> = (0..cols).map(|i| (i as f64) * 0.01).collect();
    let q8_mat = tang_infer::quantize_matrix_q8(&mat, rows, cols);
    group.bench_function("q8_matvec_576x576", |b| {
        b.iter(|| black_box(tang_infer::q8_matvec(&q8_mat, &vec_data)));
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_linear_forward,
    bench_linear_backward,
    bench_swiglu,
    bench_gqa,
    bench_rmsnorm,
    bench_rope,
    bench_cross_entropy,
    bench_kv_cache,
    bench_sampling,
    bench_quantize,
);
criterion_main!(benches);
