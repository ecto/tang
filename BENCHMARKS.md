# Tang Benchmarks

**Date:** 2026-08-11
**Hardware:** Apple M4 Max (16 cores: 12P + 4E), 48 GB RAM, integrated GPU
**Toolchain:** `cargo bench -p tang-bench --release` (criterion 0.5)

The "before" column throughout is the 2026-02-28 run on the same hardware,
prior to the row-major GEMM kernels in `tang-la::gemm`. See
[What changed](#what-changed) for why the numbers moved.

## CPU Training (`cpu_training`)

### Linear Layer

| Benchmark | Before | Now | Speedup |
|-----------|--------|-----|---------|
| `linear_forward/fwd/4x128->64` | 146 µs | **2.26 µs** | 65x |
| `linear_forward/fwd/16x576->576` | 6.83 ms | **213 µs** | 32x |
| `linear_forward/fwd/1x576->1536` | 14.6 ms | **106 µs** | 138x |
| `linear_backward/bwd/16x576->576` | 3.58 ms | **543 µs** | 6.6x |

### SwiGLU FFN

| Benchmark | Before | Now | Speedup |
|-----------|--------|-----|---------|
| `swiglu/fwd/4x64` | 443 µs | **10.0 µs** | 44x |
| `swiglu/bwd/4x64` | 364 µs | **31.7 µs** | 11x |
| `swiglu/fwd/16x576` | 54.8 ms | **2.22 ms** | 25x |
| `swiglu/bwd/16x576` | 46.5 ms | **6.26 ms** | 7.4x |

### Grouped-Query Attention (causal)

| Benchmark | Before | Now | Speedup |
|-----------|--------|-----|---------|
| `gqa/fwd/seq8_d64_h4` | 269 µs | **17.0 µs** | 16x |
| `gqa/fwd/seq32_d64_h4` | 409 µs | **73.6 µs** | 5.6x |

### RMSNorm / RoPE

Neither is matmul-bound, so both are roughly unchanged.

| Benchmark | Before | Now |
|-----------|--------|-----|
| `rmsnorm/fwd/16x64` | 2.1 µs | 1.78 µs |
| `rmsnorm/fwd/16x576` | 18.8 µs | 14.6 µs |
| `rope/apply/seq32_d64` | 6.2 µs | 6.18 µs |
| `rope/apply/seq128_d64` | 26.2 µs | 24.9 µs |

### Loss Functions

| Benchmark | Before | Now |
|-----------|--------|-----|
| `loss/seq_ce/seq32_v256` | 20.7 µs | 19.3 µs |
| `loss/seq_ce/seq128_v256` | 85.0 µs | 79.4 µs |
| `loss/seq_ce_grad/seq32_v256` | 5.43 ms | 4.97 ms |
| `loss/seq_ce_grad/seq128_v256` | 22.3 ms | 20.3 ms |

`seq_ce_grad` is still the slowest CPU path by a wide margin and does not go
through GEMM — it is the next thing worth looking at.

### KV Cache (CPU, tang-infer)

| Benchmark | Before | Now |
|-----------|--------|-----|
| `kv_cache/prefill_128tok` (4 layers, kv_dim=128) | 21.7 µs | 17.7 µs |
| `kv_cache/decode_128steps` (4 layers, 1 tok/step) | 36.3 µs | 35.8 µs |

### Sampling

Unchanged — no matmul involved.

| Benchmark | Before | Now |
|-----------|--------|-----|
| `sampling/greedy_v256` | 405 ns | 415 ns |
| `sampling/topk40_v256` | 2.3 µs | 2.24 µs |
| `sampling/topk40_rep_pen_v256` | 2.3 µs | 2.36 µs |

### Quantization

| Benchmark | Before | Now | Speedup |
|-----------|--------|-----|---------|
| `quantize/q8_roundtrip_576` | 655 ns | 612 ns | 1.1x |
| `quantize/q4_roundtrip_576` | 865 ns | 754 ns | 1.1x |
| `quantize/q8_matvec_576x576` | 208 µs | **72.8 µs** | 2.9x |

The matvec win is from hoisting the per-block scale out of the elementwise
loop and moving the `in_dim` bounds test out of it into a separate tail block,
which is what had been blocking vectorization. Both kernels still accumulate
in `f64`; see [What's still missing](#whats-still-missing).

### Dense GEMM

`DMat::mul_mat` and `Tensor::matmul` both route through `tang-la::gemm`, which
is register-tiled and NEON-vectorized for `f32`/`f64`. `--features threads`
splits the output rows across a rayon pool.

**512x512x512 f32** (`cargo run -p tang-bench --example probe_linear --release`):

| Path | Time | GFLOP/s | vs. before |
|------|------|---------|------------|
| Before (naive triple loop) | 8.19 ms | 33 | — |
| `gemm`, 1 core | 2.74 ms | 98 | 3.0x |
| `gemm`, `--features threads` | **0.487 ms** | **551** | **16.8x** |
| Accelerate (`--features accelerate`) | 0.18 ms | 1,478 | 45x |

Accelerate still wins by ~2.7x because it reaches Apple's AMX coprocessor,
which NEON code cannot. The portable path is now within striking distance
rather than 45x behind, and it is the only path available off macOS.

### BLAS Acceleration (Apple Accelerate)

`DMat::mul_mat` with `--features accelerate` dispatches to `cblas_sgemm`/`cblas_dgemm` via Apple's Accelerate framework (AMX/NEON). Zero-copy — `DMat` is already column-major matching CBLAS layout.

| Type | Generic (before) | `gemm` | `gemm` + threads | Accelerate |
|------|------------------|--------|------------------|------------|
| f32 | 8.19 ms | 2.74 ms | 0.487 ms | 0.18 ms |
| f64 | 15.5 ms | — | — | 0.67 ms |

Enable with `cargo build --features accelerate` (macOS only). Falls through to the `gemm` kernels for non-f32/f64 types.

## What changed

Three things, in descending order of measured impact:

1. **`Tensor::transpose` allocated a `Vec` per element.** It was built on
   `from_fn` with an `idx.to_vec()` inside the closure, so a 576x576 transpose
   did 331,776 heap allocations. `Linear::forward` called it on the weight
   matrix on every forward pass, which was 70% of that benchmark's time. The
   2-D contiguous case is now a cache-blocked copy.
2. **No GEMM.** `DMat::mul_mat` was a naive triple loop and `Tensor::matmul`
   round-tripped both operands through `DMat` element-by-element. Both now call
   `tang-la::gemm`, which blocks over `k`, holds a 4-row tile of `C` in vector
   registers across the whole panel, and dispatches `f32`/`f64` to NEON.
3. **Linear layers transposed to multiply.** Weights are stored `[out, in]`, so
   `gemm_nt` contracts along contiguous rows of both operands directly. The
   weight gradient uses `gemm_tn` the same way. No transposed copy is
   materialized in either direction.

## What's still missing

- **No x86 SIMD.** The microkernels are NEON-only; on x86 `f32`/`f64` fall
  through to the generic path. AVX2/FMA kernels are two more `kernels!`
  expansions in `crates/tang-la/src/gemm.rs`.
- **Threading only splits rows.** `par_rows` splits `m`, which is the batch
  dimension for `Tensor::matmul_nt` — a batch of 16 gets 4 tasks, not 12.
  Splitting `n` as well would need the kernels to take raw pointers.
- **Quantized matvec accumulates in `f64`.** Q8 exists to make weights `i8`;
  the kernel should do `i8`x`i8`->`i32` dot products (NEON `sdot`) with
  on-the-fly activation quantization. Changing the public signature off
  `&[f64]` is a breaking change, so it is not done here.
- **`Parameter::randn` costs 3.6 ms per 576x576 matrix.** `Rng::normal` uses
  Box-Muller and discards the sine half, so every sample pays two RNG draws
  plus `ln`/`sqrt`/`cos`. Caching the second variate is a 2x win on model
  construction, but it changes the seeded RNG stream.
- **The `faer` feature is dead.** `crates/tang-la/Cargo.toml` declares it and
  pulls the optional dependency, but `faer` appears nowhere in
  `crates/tang-la/src/` outside two doc comments. The README claim that it
  "enables world-class f64 performance" has been removed; the feature itself is
  left in place pending a decision to wire it up or drop it.
- **`la`/`tensor`/`train` are vendored twice.** `crates/tang/src/{la,tensor,train}`
  are copies of the standalone crates that differ only in import paths, and
  they have already drifted (different panic messages, 161 differing lines in
  `layers.rs`). Every change above had to be applied in both places.

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
