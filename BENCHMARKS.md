# Tang Benchmarks

**Date:** 2026-02-28
**Hardware:** Apple M4 Max (16 cores: 12P + 4E), 48 GB RAM, integrated GPU
**Toolchain:** `cargo bench -p tang-bench --release` (criterion 0.5)

## CPU Training (`cpu_training`)

### Linear Layer

| Benchmark | ns/iter | ~Time |
|-----------|---------|-------|
| `linear_forward/fwd/4x128->64` | 146,086 | 146 µs |
| `linear_forward/fwd/16x576->576` | 6,831,088 | 6.83 ms |
| `linear_forward/fwd/1x576->1536` | 14,582,729 | 14.6 ms |
| `linear_backward/bwd/16x576->576` | 3,580,891 | 3.58 ms |

### SwiGLU FFN

| Benchmark | ns/iter | ~Time |
|-----------|---------|-------|
| `swiglu/fwd/4x64` | 443,184 | 443 µs |
| `swiglu/bwd/4x64` | 363,886 | 364 µs |
| `swiglu/fwd/16x576` | 54,809,729 | 54.8 ms |
| `swiglu/bwd/16x576` | 46,500,166 | 46.5 ms |

### Grouped-Query Attention (causal)

| Benchmark | ns/iter | ~Time |
|-----------|---------|-------|
| `gqa/fwd/seq8_d64_h4` | 269,116 | 269 µs |
| `gqa/fwd/seq32_d64_h4` | 408,700 | 409 µs |

### RMSNorm / RoPE

| Benchmark | ns/iter | ~Time |
|-----------|---------|-------|
| `rmsnorm/fwd/16x64` | 2,068 | 2.1 µs |
| `rmsnorm/fwd/16x576` | 18,755 | 18.8 µs |
| `rope/apply/seq32_d64` | 6,207 | 6.2 µs |
| `rope/apply/seq128_d64` | 26,166 | 26.2 µs |

### Loss Functions

| Benchmark | ns/iter | ~Time |
|-----------|---------|-------|
| `loss/seq_ce/seq32_v256` | 20,722 | 20.7 µs |
| `loss/seq_ce/seq128_v256` | 84,986 | 85.0 µs |
| `loss/seq_ce_grad/seq32_v256` | 5,433,120 | 5.43 ms |
| `loss/seq_ce_grad/seq128_v256` | 22,287,882 | 22.3 ms |

### KV Cache (CPU, tang-infer)

| Benchmark | ns/iter | ~Time |
|-----------|---------|-------|
| `kv_cache/prefill_128tok` (4 layers, kv_dim=128) | 21,713 | 21.7 µs |
| `kv_cache/decode_128steps` (4 layers, 1 tok/step) | 36,337 | 36.3 µs |

### Sampling

| Benchmark | ns/iter | ~Time |
|-----------|---------|-------|
| `sampling/greedy_v256` | 405 | 405 ns |
| `sampling/topk40_v256` | 2,314 | 2.3 µs |
| `sampling/topk40_rep_pen_v256` | 2,336 | 2.3 µs |

### Quantization

| Benchmark | ns/iter | ~Time |
|-----------|---------|-------|
| `quantize/q8_roundtrip_576` | 655 | 655 ns |
| `quantize/q4_roundtrip_576` | 865 | 865 ns |
| `quantize/q8_matvec_576x576` | 207,992 | 208 µs |

### BLAS Acceleration (Apple Accelerate)

`DMat::mul_mat` with `--features accelerate` dispatches to `cblas_sgemm`/`cblas_dgemm` via Apple's Accelerate framework (AMX/NEON). Zero-copy — `DMat` is already column-major matching CBLAS layout.

**512x512 matmul** (`cargo run -p tang-la --example bench_matmul --release`):

| Type | Generic | Accelerate | Speedup | GFLOP/s |
|------|---------|------------|---------|---------|
| f32 | 8.19 ms | 0.18 ms | **45x** | 1,478 |
| f64 | 15.5 ms | 0.67 ms | **23x** | 402 |

Enable with `cargo build --features accelerate` (macOS only). Falls through to the generic loop for non-f32/f64 types.

---

## GPU Inference (`gpu_inference`)

All GPU benchmarks use wgpu compute shaders. Sizes match SmolLM2-135M
(d_model=576, n_heads=9, n_kv_heads=3, head_dim=64, ff_dim=1536, vocab=49152).

### Embedding Lookup

| Benchmark | ns/iter | ~Time |
|-----------|---------|-------|
| `embedding/lookup/v256_d64_s32` | 31,476 | 31.5 µs |
| `embedding/lookup/v49152_d576_s128` | 41,736 | 41.7 µs |
| `embedding/lookup/v49152_d576_s1` | 31,560 | 31.6 µs |

### RMSNorm

| Benchmark | ns/iter | ~Time |
|-----------|---------|-------|
| `rmsnorm/fwd/16x64` | 31,366 | 31.4 µs |
| `rmsnorm/fwd/128x576` | 40,985 | 41.0 µs |
| `rmsnorm/fwd/1x576` | 32,182 | 32.2 µs |

### Interleaved RoPE

| Benchmark | ns/iter | ~Time |
|-----------|---------|-------|
| `rope_interleaved/fwd/s128_h9_d64` | 40,506 | 40.5 µs |
| `rope_interleaved/fwd/s1_h9_d64` | 31,051 | 31.1 µs |
| `rope_interleaved/fwd/s512_h9_d64` | 58,349 | 58.3 µs |

### Matmul (LLM projection sizes)

| Benchmark | ns/iter | ~Time | Description |
|-----------|---------|-------|-------------|
| `matmul_llm/mm/qkv_proj_s128` | 146,336 | 146 µs | [128,576]@[576,576] |
| `matmul_llm/mm/qkv_proj_s1` | 40,003 | 40.0 µs | [1,576]@[576,576] |
| `matmul_llm/mm/ffn_up_s128` | 378,013 | 378 µs | [128,576]@[576,1536] |
| `matmul_llm/mm/ffn_up_s1` | 61,410 | 61.4 µs | [1,576]@[576,1536] |
| `matmul_llm/mm/ffn_down_s128` | 399,599 | 400 µs | [128,1536]@[1536,576] |
| `matmul_llm/mm/ffn_down_s1` | 107,770 | 108 µs | [1,1536]@[1536,576] |
| `matmul_llm/mm/lm_head_s128` | 11,400,571 | 11.4 ms | [128,576]@[576,49152] |
| `matmul_llm/mm/lm_head_s1` | 1,302,962 | 1.30 ms | [1,576]@[576,49152] |

### SwiGLU Fused

| Benchmark | ns/iter | ~Time |
|-----------|---------|-------|
| `swiglu_fused/fused/128x1536` | 51,051 | 51.1 µs |
| `swiglu_fused/fused/1x1536` | 34,313 | 34.3 µs |

### KV-Cached Attention

| Benchmark | ns/iter | ~Time |
|-----------|---------|-------|
| `kv_attention/prefill_q128_kv128` | 989,550 | 990 µs |
| `kv_attention/decode_q1_kv256` | 761,508 | 762 µs |
| `kv_attention/decode_q1_kv1024` | 2,968,394 | 2.97 ms |

### GPU KV Cache Ops

| Benchmark | ns/iter | ~Time |
|-----------|---------|-------|
| `kv_cache/append_128tok` | 73,342 | 73.3 µs |
| `kv_cache/get_keys_gpu_256tok` | 17,275 | 17.3 µs |

### Full Transformer Layer (d=576, h=9, kv=3)

| Benchmark | ns/iter | ~Time |
|-----------|---------|-------|
| `transformer_layer/prefill_s128_d576` | 3,417,569 | **3.42 ms** |
| `transformer_layer/decode_s1_kv256_d576` | 2,677,459 | **2.68 ms** |

---

## GPU Training (`gpu_training`)

### Matmul (square)

| Benchmark | ns/iter | ~Time |
|-----------|---------|-------|
| `matmul/16x16` | 29,353 | 29.4 µs |
| `matmul/32x32` | 30,582 | 30.6 µs |
| `matmul/64x64` | 33,885 | 33.9 µs |
| `matmul/128x128` | 35,710 | 35.7 µs |

### Fused Elementwise

| Benchmark | ns/iter | ~Time |
|-----------|---------|-------|
| `fused_elementwise/add_sq_256` | 66,910 | 66.9 µs |
| `fused_elementwise/add_sq_1024` | 65,463 | 65.5 µs |
| `fused_elementwise/add_sq_4096` | 72,052 | 72.1 µs |

### Linear / Sequential / Training

| Benchmark | ns/iter | ~Time |
|-----------|---------|-------|
| `linear_forward_128x64` | 111,377 | 111 µs |
| `linear_backward_128x64` | 131,256 | 131 µs |
| `sequential_2_8_1_fwd_bwd` | 596,925 | 597 µs |
| `training_step_xor` | 2,180,537 | 2.18 ms |
| `mse_loss_64` | 290,317 | 290 µs |

---

## Throughput Estimates (SmolLM2-135M, 30 layers)

Based on single-layer GPU benchmarks extrapolated to 30 layers:

| Scenario | Per-layer | 30 layers + LM head | Tokens/sec |
|----------|-----------|---------------------|------------|
| **Prefill** (128 tokens) | 3.42 ms | ~114 ms | ~1,123 tok/s |
| **Decode** (1 token, kv=256) | 2.68 ms | ~82 ms | ~12 tok/s |

Decode is memory-bandwidth-bound: attention scales linearly with KV cache length.
LM head matmul (576x49152) adds ~1.3 ms per decode step.
