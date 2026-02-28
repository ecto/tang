//! Benchmarks for tang-gpu LLM inference primitives.
//!
//! Measures: embedding lookup, RMSNorm, RoPE, matmul at model-relevant sizes,
//! SwiGLU fused, KV-cached attention, and full forward pass.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use tang_gpu::*;
use tang_gpu::matmul::matmul;
use tang_gpu::nn::bias_add;

fn get_device() -> GpuDevice {
    GpuDevice::new_sync().expect("GPU device required for benchmarks")
}

fn random_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n).map(|_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as f32 / (1u32 << 31) as f32 - 0.5
    }).collect()
}

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

fn bench_embedding(c: &mut Criterion) {
    let device = get_device();
    let mut cache = KernelCache::new();
    let mut group = c.benchmark_group("gpu_embedding");

    for &(vocab, dim, seq) in &[(256, 64, 32), (49152, 576, 128), (49152, 576, 1)] {
        let w = random_f32(vocab * dim, 42);
        let embed = GpuEmbedding::new(&device, &w, vocab, dim);
        let tokens: Vec<u32> = (0..seq).map(|i| (i % vocab) as u32).collect();
        let tok_buf = GpuBuffer::from_u32_slice(&device, &tokens);

        group.bench_function(
            BenchmarkId::new("lookup", format!("v{vocab}_d{dim}_s{seq}")),
            |b| {
                b.iter(|| {
                    let out = embed.forward(&device, &mut cache, &tok_buf, seq);
                    black_box(out);
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------

fn bench_gpu_rmsnorm(c: &mut Criterion) {
    let device = get_device();
    let mut cache = KernelCache::new();
    let mut group = c.benchmark_group("gpu_rmsnorm");

    for &(batch, dim) in &[(16, 64), (128, 576), (1, 576)] {
        let norm = GpuRMSNorm::new(&device, dim, 1e-6);
        let data = random_f32(batch * dim, 42);
        let input = GpuTensor::from_slice(&device, &data, &[batch, dim]);

        group.bench_function(
            BenchmarkId::new("fwd", format!("{batch}x{dim}")),
            |b| {
                b.iter(|| {
                    let out = norm.forward(&device, &mut cache, &input);
                    black_box(out);
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// RoPE (interleaved)
// ---------------------------------------------------------------------------

fn bench_gpu_rope(c: &mut Criterion) {
    let device = get_device();
    let mut cache = KernelCache::new();
    let mut group = c.benchmark_group("gpu_rope_interleaved");

    for &(seq, n_heads, head_dim) in &[(128, 9, 64), (1, 9, 64), (512, 9, 64)] {
        let rope = GpuInterleavedRoPE::new(&device, head_dim, 2048, 10000.0);
        let data = random_f32(seq * n_heads * head_dim, 42);
        let input = GpuTensor::from_slice(&device, &data, &[seq, n_heads, head_dim]);

        group.bench_function(
            BenchmarkId::new("fwd", format!("s{seq}_h{n_heads}_d{head_dim}")),
            |b| {
                b.iter(|| {
                    let out = rope.forward(&device, &mut cache, &input, 0);
                    black_box(out);
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Matmul (model-relevant sizes)
// ---------------------------------------------------------------------------

fn bench_gpu_matmul_llm(c: &mut Criterion) {
    let device = get_device();
    let mut cache = KernelCache::new();
    let mut group = c.benchmark_group("gpu_matmul_llm");

    // (batch, K, N) — typical projections
    for &(m, k, n, label) in &[
        (128, 576, 576, "qkv_proj_s128"),
        (1, 576, 576, "qkv_proj_s1"),
        (128, 576, 1536, "ffn_up_s128"),
        (1, 576, 1536, "ffn_up_s1"),
        (128, 1536, 576, "ffn_down_s128"),
        (1, 1536, 576, "ffn_down_s1"),
        (128, 576, 49152, "lm_head_s128"),
        (1, 576, 49152, "lm_head_s1"),
    ] {
        let a_data = random_f32(m * k, 42);
        let b_data = random_f32(k * n, 43);
        let a = GpuTensor::from_slice(&device, &a_data, &[m, k]);
        let b = GpuTensor::from_slice(&device, &b_data, &[k, n]);

        group.bench_function(BenchmarkId::new("mm", label), |bench| {
            bench.iter(|| {
                let c = matmul(&device, &mut cache, &a, &b);
                black_box(c);
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// SwiGLU fused
// ---------------------------------------------------------------------------

fn bench_gpu_swiglu(c: &mut Criterion) {
    let device = get_device();
    let mut cache = KernelCache::new();
    let mut group = c.benchmark_group("gpu_swiglu_fused");

    for &(batch, ff_dim) in &[(128, 1536), (1, 1536)] {
        let gate_data = random_f32(batch * ff_dim, 42);
        let up_data = random_f32(batch * ff_dim, 43);
        let gate = GpuTensor::from_slice(&device, &gate_data, &[batch, ff_dim]);
        let up = GpuTensor::from_slice(&device, &up_data, &[batch, ff_dim]);

        group.bench_function(
            BenchmarkId::new("fused", format!("{batch}x{ff_dim}")),
            |b| {
                b.iter(|| {
                    let out = swiglu_fused_pub(&device, &mut cache, &gate, &up);
                    black_box(out);
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// KV-cached attention
// ---------------------------------------------------------------------------

fn bench_gpu_kv_attention(c: &mut Criterion) {
    let device = get_device();
    let mut cache = KernelCache::new();
    let mut group = c.benchmark_group("gpu_kv_attention");

    let n_heads = 9;
    let n_kv_heads = 3;
    let head_dim = 64;

    // Prefill: q_len=128, kv_len=128
    {
        let q_len = 128;
        let kv_len = 128;
        let q_data = random_f32(q_len * n_heads * head_dim, 42);
        let k_data = random_f32(kv_len * n_kv_heads * head_dim, 43);
        let v_data = random_f32(kv_len * n_kv_heads * head_dim, 44);
        let q = GpuTensor::from_slice(&device, &q_data, &[q_len, n_heads * head_dim]);
        let k = GpuTensor::from_slice(&device, &k_data, &[kv_len, n_kv_heads * head_dim]);
        let v = GpuTensor::from_slice(&device, &v_data, &[kv_len, n_kv_heads * head_dim]);

        group.bench_function("prefill_q128_kv128", |b| {
            b.iter(|| {
                let out = kv_attention_fused(
                    &device, &mut cache,
                    &q, &k, &v,
                    q_len, kv_len,
                    n_heads, n_kv_heads, head_dim,
                    0,
                );
                black_box(out);
            });
        });
    }

    // Decode: q_len=1, kv_len=256
    {
        let q_len = 1;
        let kv_len = 256;
        let q_data = random_f32(q_len * n_heads * head_dim, 42);
        let k_data = random_f32(kv_len * n_kv_heads * head_dim, 43);
        let v_data = random_f32(kv_len * n_kv_heads * head_dim, 44);
        let q = GpuTensor::from_slice(&device, &q_data, &[q_len, n_heads * head_dim]);
        let k = GpuTensor::from_slice(&device, &k_data, &[kv_len, n_kv_heads * head_dim]);
        let v = GpuTensor::from_slice(&device, &v_data, &[kv_len, n_kv_heads * head_dim]);

        group.bench_function("decode_q1_kv256", |b| {
            b.iter(|| {
                let out = kv_attention_fused(
                    &device, &mut cache,
                    &q, &k, &v,
                    q_len, kv_len,
                    n_heads, n_kv_heads, head_dim,
                    255,
                );
                black_box(out);
            });
        });
    }

    // Decode: q_len=1, kv_len=1024
    {
        let q_len = 1;
        let kv_len = 1024;
        let q_data = random_f32(q_len * n_heads * head_dim, 42);
        let k_data = random_f32(kv_len * n_kv_heads * head_dim, 43);
        let v_data = random_f32(kv_len * n_kv_heads * head_dim, 44);
        let q = GpuTensor::from_slice(&device, &q_data, &[q_len, n_heads * head_dim]);
        let k = GpuTensor::from_slice(&device, &k_data, &[kv_len, n_kv_heads * head_dim]);
        let v = GpuTensor::from_slice(&device, &v_data, &[kv_len, n_kv_heads * head_dim]);

        group.bench_function("decode_q1_kv1024", |b| {
            b.iter(|| {
                let out = kv_attention_fused(
                    &device, &mut cache,
                    &q, &k, &v,
                    q_len, kv_len,
                    n_heads, n_kv_heads, head_dim,
                    1023,
                );
                black_box(out);
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// GPU KV Cache ops
// ---------------------------------------------------------------------------

fn bench_gpu_kv_cache(c: &mut Criterion) {
    let device = get_device();
    let mut cache = KernelCache::new();
    let mut group = c.benchmark_group("gpu_kv_cache");

    let n_kv_heads = 3;
    let head_dim = 64;
    let max_seq = 2048;

    // Append (prefill batch)
    group.bench_function("append_128tok", |b| {
        let k_data = random_f32(128 * n_kv_heads * head_dim, 42);
        let v_data = random_f32(128 * n_kv_heads * head_dim, 43);
        let k = GpuTensor::from_slice(&device, &k_data, &[128, n_kv_heads, head_dim]);
        let v = GpuTensor::from_slice(&device, &v_data, &[128, n_kv_heads, head_dim]);
        b.iter(|| {
            let mut kv = GpuKVCache::new(&device, n_kv_heads, head_dim, max_seq);
            kv.append(&device, &mut cache, &k, &v);
            black_box(&kv);
        });
    });

    // Get keys/values GPU-side (no CPU roundtrip)
    group.bench_function("get_keys_gpu_256tok", |b| {
        let mut kv = GpuKVCache::new(&device, n_kv_heads, head_dim, max_seq);
        let k_data = random_f32(256 * n_kv_heads * head_dim, 42);
        let v_data = random_f32(256 * n_kv_heads * head_dim, 43);
        let k = GpuTensor::from_slice(&device, &k_data, &[256, n_kv_heads, head_dim]);
        let v = GpuTensor::from_slice(&device, &v_data, &[256, n_kv_heads, head_dim]);
        kv.append(&device, &mut cache, &k, &v);
        b.iter(|| {
            let keys = kv.get_keys_gpu(&device, &mut cache);
            black_box(keys);
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Full transformer layer forward (GPU)
// ---------------------------------------------------------------------------

fn bench_gpu_transformer_layer(c: &mut Criterion) {
    let device = get_device();
    let mut kcache = KernelCache::new();
    let mut group = c.benchmark_group("gpu_transformer_layer");

    let d_model = 576;
    let n_heads = 9;
    let n_kv_heads = 3;
    let head_dim = 64;
    let ff_dim = 1536;
    let max_seq = 2048;

    // Build a single transformer layer manually:
    // ln1 -> Q/K/V -> RoPE -> KV attention -> wo -> residual -> ln2 -> SwiGLU -> residual
    let ln1 = GpuRMSNorm::new(&device, d_model, 1e-6);
    let ln2 = GpuRMSNorm::new(&device, d_model, 1e-6);
    let wq = GpuLinear::kaiming(&device, d_model, d_model, 42);
    let wk = GpuLinear::kaiming(&device, d_model, n_kv_heads * head_dim, 43);
    let wv = GpuLinear::kaiming(&device, d_model, n_kv_heads * head_dim, 44);
    let wo = GpuLinear::kaiming(&device, d_model, d_model, 45);
    let ffn_gate = GpuLinear::kaiming(&device, d_model, ff_dim, 46);
    let ffn_up = GpuLinear::kaiming(&device, d_model, ff_dim, 47);
    let ffn_down = GpuLinear::kaiming(&device, ff_dim, d_model, 48);
    let rope = GpuInterleavedRoPE::new(&device, head_dim, max_seq, 10000.0);

    // Helper: batched linear on GPU
    let linear_2d = |device: &GpuDevice, cache: &mut KernelCache, linear: &GpuLinear, input: &GpuTensor| -> GpuTensor {
        let wt = linear.weight.transpose_gpu(device, cache);
        let out = matmul(device, cache, input, &wt);
        bias_add(device, cache, &out, &linear.bias)
    };

    // Prefill: seq_len=128
    group.bench_function("prefill_s128_d576", |b| {
        let data = random_f32(128 * d_model, 42);
        let input = GpuTensor::from_slice(&device, &data, &[128, d_model]);
        b.iter(|| {
            let mut kv = GpuKVCache::new(&device, n_kv_heads, head_dim, max_seq);
            let normed = ln1.forward(&device, &mut kcache, &input);
            let q = linear_2d(&device, &mut kcache, &wq, &normed);
            let k = linear_2d(&device, &mut kcache, &wk, &normed);
            let v = linear_2d(&device, &mut kcache, &wv, &normed);
            let q3 = q.reshape(&[128, n_heads, head_dim]);
            let k3 = k.reshape(&[128, n_kv_heads, head_dim]);
            let v3 = v.reshape(&[128, n_kv_heads, head_dim]);
            let qr = rope.forward(&device, &mut kcache, &q3, 0);
            let kr = rope.forward(&device, &mut kcache, &k3, 0);
            kv.append(&device, &mut kcache, &kr, &v3);
            let kf = kv.get_keys_gpu(&device, &mut kcache);
            let vf = kv.get_values_gpu(&device, &mut kcache);
            let q_flat = qr.reshape(&[128, n_heads * head_dim]);
            let k_flat = kf.reshape(&[128, n_kv_heads * head_dim]);
            let v_flat = vf.reshape(&[128, n_kv_heads * head_dim]);
            let attn = kv_attention_fused(
                &device, &mut kcache, &q_flat, &k_flat, &v_flat,
                128, 128, n_heads, n_kv_heads, head_dim, 0,
            );
            let proj = linear_2d(&device, &mut kcache, &wo, &attn);
            let res1 = add_tensors(&device, &mut kcache, &input, &proj);
            let normed2 = ln2.forward(&device, &mut kcache, &res1);
            let gate = linear_2d(&device, &mut kcache, &ffn_gate, &normed2);
            let up = linear_2d(&device, &mut kcache, &ffn_up, &normed2);
            let act = swiglu_fused_pub(&device, &mut kcache, &gate, &up);
            let ffn_out = linear_2d(&device, &mut kcache, &ffn_down, &act);
            let out = add_tensors(&device, &mut kcache, &res1, &ffn_out);
            black_box(out);
        });
    });

    // Decode: seq_len=1, kv_len=256
    group.bench_function("decode_s1_kv256_d576", |b| {
        // Pre-fill a KV cache with 255 tokens
        let mut kv = GpuKVCache::new(&device, n_kv_heads, head_dim, max_seq);
        let prefill_k = random_f32(255 * n_kv_heads * head_dim, 50);
        let prefill_v = random_f32(255 * n_kv_heads * head_dim, 51);
        let pk = GpuTensor::from_slice(&device, &prefill_k, &[255, n_kv_heads, head_dim]);
        let pv = GpuTensor::from_slice(&device, &prefill_v, &[255, n_kv_heads, head_dim]);
        kv.append(&device, &mut kcache, &pk, &pv);

        let data = random_f32(d_model, 52);
        let input = GpuTensor::from_slice(&device, &data, &[1, d_model]);
        let pos = 255;

        b.iter(|| {
            let normed = ln1.forward(&device, &mut kcache, &input);
            let q = linear_2d(&device, &mut kcache, &wq, &normed);
            let k = linear_2d(&device, &mut kcache, &wk, &normed);
            let v = linear_2d(&device, &mut kcache, &wv, &normed);
            let q3 = q.reshape(&[1, n_heads, head_dim]);
            let k3 = k.reshape(&[1, n_kv_heads, head_dim]);
            let _v3 = v.reshape(&[1, n_kv_heads, head_dim]);
            let qr = rope.forward(&device, &mut kcache, &q3, pos);
            let _kr = rope.forward(&device, &mut kcache, &k3, pos);
            // Note: in real inference we'd append to kv, but for benchmarking
            // the compute cost, we just use the pre-filled cache
            let kf = kv.get_keys_gpu(&device, &mut kcache);
            let vf = kv.get_values_gpu(&device, &mut kcache);
            let total = kv.len;
            let q_flat = qr.reshape(&[1, n_heads * head_dim]);
            let k_flat = kf.reshape(&[total, n_kv_heads * head_dim]);
            let v_flat = vf.reshape(&[total, n_kv_heads * head_dim]);
            let attn = kv_attention_fused(
                &device, &mut kcache, &q_flat, &k_flat, &v_flat,
                1, total, n_heads, n_kv_heads, head_dim, pos - 1,
            );
            let proj = linear_2d(&device, &mut kcache, &wo, &attn);
            let res1 = add_tensors(&device, &mut kcache, &input, &proj);
            let normed2 = ln2.forward(&device, &mut kcache, &res1);
            let gate = linear_2d(&device, &mut kcache, &ffn_gate, &normed2);
            let up = linear_2d(&device, &mut kcache, &ffn_up, &normed2);
            let act = swiglu_fused_pub(&device, &mut kcache, &gate, &up);
            let ffn_out = linear_2d(&device, &mut kcache, &ffn_down, &act);
            let out = add_tensors(&device, &mut kcache, &res1, &ffn_out);
            black_box(out);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_embedding,
    bench_gpu_rmsnorm,
    bench_gpu_rope,
    bench_gpu_matmul_llm,
    bench_gpu_swiglu,
    bench_gpu_kv_attention,
    bench_gpu_kv_cache,
    bench_gpu_transformer_layer,
);
criterion_main!(benches);
