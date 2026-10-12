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

## Kernels vs tensor IR (`kernels_vs_ir`)

**Date:** 2026-10-11. Same hardware. See `crates/tang-compile/src/tensor.rs`
for the IR (views + comprehensions + monoids) and `tensor_wgsl.rs` /
`tensor_tile.rs` for the schedules.

### CPU: hand kernels vs the *scalar* `tang-expr` IR (f64)

The scalar IR interprets a closure per element and cannot express a tensor
op as one node, so it loses everywhere and the gap grows with size. This is
the motivation for the tensor-level IR, not a result about it.

| block | size | hand | ir | ir_opt | trace+compile |
|---|---|---|---|---|---|
| swiglu act | 1M | 2.6 ms | 42.8 ms | 42.4 ms | – |
| rmsnorm | 16×576 | 15 µs | 90 µs | 78 µs | 77 ms |
| rmsnorm | 16×2048 | 52 µs | 330 µs | 281 µs | 1.06 s |
| matvec | 255 | 14 µs | 414 µs | 534 µs | 92 ms |

### GPU: hand-written causal attention vs attention *derived* from tensor IR

Single head, d = 64, f32. `hand` is `kv_attention_fused`. The `ir_*`
columns are the same naive `softmax(q·kᵀ/√d + mask)·v` graph under
different compiler schedules: `naive` materializes the n×n scores;
`flash` is after `substitute_maps` + `stabilize_softmax` (one
online-softmax reduction over `[rows, value-columns]` points with q·k
recomputed inside, nothing n×n in memory); `split` adds a 256-thread tree
reduction per point; `tiled` is `schedule_tile_2d` with 64 value-column
lanes × 4 rows per workgroup, the k/v tiles *and the q·k scores* staged
cooperatively in workgroup memory per 32-key chunk. Times include ~1.3 ms
of submit + readback; `autotune`'s kernel-only timing of `tiled` at
n = 1024 is ~0.5 ms.

| n | hand | ir_naive | ir_flash | ir_flash_split | ir_flash_tiled |
|---|---|---|---|---|---|
| 128 | 1.33 ms | 1.68 ms | 1.33 ms | 3.6 ms | 1.40 ms |
| 512 | 4.26 ms | 2.30 ms | 4.15 ms | 5.1 ms | **2.99 ms** |
| 1024 | 10.3 ms | 3.9 ms | 7.7 ms | 16.5 ms | **1.35 ms** |

Takeaways:

- Every derived schedule matches the hand kernel's output (checked at
  n = 100, 512, 1024); none contains attention-specific code.
- The tiled schedule is flash attention's real structure — rows × lanes,
  cooperative score tiles, online softmax — and it came out of three
  general rules: substitution, the online-softmax monoid rewrite, and
  "stage anything the lane axis does not depend on". At n = 1024 it is
  7.6× faster than the hand kernel and ~3× faster than the naive graph.
- The lane layout is only good *with* tiling: `split` recomputes the
  scores per lane and regresses. Schedules are not independent of the
  algebra, which is why `schedule_auto` looks for a lane axis first.

### GPU: hand-written matmul vs matmul derived from tensor IR

`hand` is `tang_gpu::matmul::matmul`. The IR version is one comprehension,
`Σ_k a[i,k]·b[k,j]`, under the same three schedules; `tiled` is a 16×16
point tile per workgroup with the `a` and `b` tiles staged in workgroup
memory (`schedule_tile_2d`), the classic shared-memory matmul, derived.

| n | hand | ir_naive | ir_split | ir_tiled |
|---|---|---|---|---|
| 256 | 1.50 ms | 1.47 ms | 2.38 ms | 1.45 ms |
| 1024 | 4.14 ms | 4.2–5.6 ms | 12.3 ms | 5.73 ms |

The derived matmul is at parity with the hand kernel. Shared-memory
staging does not pay on this chip either: the operands already live in
cache, so the extra barriers cost more than the global reads they save.

### GPU: rmsnorm and swiglu, hand vs derived

`hand` is `GpuRMSNorm::forward` (one workgroup per row, shared-memory
reduction) and `swiglu_fused_pub`. The IR rmsnorm is `x · rsqrt(mean(x²)+ε)`
after `substitute_maps`: one Sum reduction and one map.

| rmsnorm | hand | ir_naive | ir_split | ir_tiled |
|---|---|---|---|---|
| 128×576 | 1.46 ms | 1.49 ms | 1.48 ms | 1.50 ms |
| 2048×2048 | 2.16 ms | 3.90 ms | **2.31 ms** | 6.32 ms |

| swiglu | hand | ir |
|---|---|---|
| 65 536 | 1.57 ms | 1.55 ms |
| 4 194 304 | 2.09 ms | 2.25 ms |

| adam step | hand (`dispatch_adam`, 1 in-place kernel) | ir (3 maps) |
|---|---|---|
| 65 536 | 1.59 ms | 1.63 ms |
| 4 194 304 | 2.34 ms | 2.38 ms |

Note the winner flips: attention wanted the naive schedule, rmsnorm wants
the split one (its reduce axis is 2048 wide and there are only 2048 rows,
so one thread per row starves the GPU). Same compiler, same cost of
being wrong, different answer per op — which is the case for autotuning
rather than a fixed heuristic.

### `schedule_auto`

A fixed rule distilled from the tables above: a materialized reduction
whose points end in a lane axis (16–128 wide, with rows above it) *and*
that has a lane-invariant single-use producer to share is tiled rows ×
lanes with that producer inlined and staged cooperatively; otherwise one
with fewer than 2¹⁴ points and a reduce axis ≥ 64 is split across a
256-thread workgroup; everything else stays one thread per point. It lands within noise of the
best hand-picked schedule for attention (tiled), rmsnorm (split) and
matmul (naive-ish; the tile does not pay here). It does not choose between
algebras — materialized scores vs online softmax — which is `autotune`'s
job.

### GPU: a whole transformer block, hand-written vs derived

The `gpu_transformer_layer` prefill path (rmsnorm → q/k/v → interleaved
RoPE → GQA causal attention (9 heads, 3 kv heads, d=64) → wo → residual →
rmsnorm → SwiGLU → residual, d_model 576, ff 1536) rebuilt as one tensor
IR graph (`tang_bench::ir_block`) and run under `substitute_maps` +
`stabilize_softmax` + `schedule_auto`. Identical weights; outputs agree
with the hand path and with the f64 interpreter to 5e-7.

| seq | hand | ir_auto | ir_tuned |
|---|---|---|---|
| 128 | 5.07 ms | 4.63 ms | **4.45 ms** |
| 512 | 24.1 ms | 14.3 ms | **13.9 ms** |

`ir_tuned` is the per-kernel autotuner's schedule, measured once per
(program, device) and kept in a `TuningCache` file (`crates/tang-bench/
target/tang-tuning-block-<seq>-<kv>.txt`); later runs load it.

Decode (one query token against a rotated k/v cache, the launch-bound
regime where hand kernels were expected to look best):

| q / kv | hand | ir_auto | ir_tuned |
|---|---|---|---|
| 1 / 256 | 4.75 ms | **2.96 ms** | 2.98 ms |
| 1 / 1024 | 7.78 ms | 2.79 ms | **2.73 ms** |

The derived block issues 10 kernels to the hand path's ~16 and its
attention kernel reads the cache once; both matter when every dispatch
costs more than the math.

(10 kernels; `examples/block_kernels.rs` times each one. An earlier
version of `schedule_auto` lane-tiled the plain k/v projections and lost
1.5 ms to it — the tile only pays when there is a lane-invariant
sub-computation to share, which is now the rule.)

GQA's `head / heads_per_group` is not affine, so heads are expressed as
`[groups, per_group]` dims and the views do the rest; the causal mask is an
input. Nothing in the compiler knows what a transformer is.

### Per-kernel autotuning

`ir_gpu::autotune_kernels` times every thread mapping in
`schedule_candidates` for each materialized comp on the device, with the
graph's real inputs in place, and keeps the fastest
(`examples/block_autotune.rs`). Candidates: one thread per point, a
256-thread split, rows × lanes, and the classic 16×16 shared-memory tile.

| seq | hand | `schedule_auto` | autotuned |
|---|---|---|---|
| 128 | 4.93 ms | 4.07 ms | 4.03 ms |
| 512 | 24.3 ms | 15.3 ms | **13.5 ms** |

The tuner agrees with `schedule_auto` on every reduction and attention
kernel and overrides it on three projection matmuls, where the 16×16 tile
wins at these shapes (128×576×1536: tiled 1.39 ms vs one-thread-per-point
2.66 ms vs hand 2.97 ms) although it lost at 1024². That is the whole
argument in one table: the right schedule depends on the shape and the
chip, and measuring beats remembering.

### Derived backward pass

`ir_block_gpu::tests::derived_block_backward_on_gpu` differentiates the
whole block (`backward` on the algorithm graph), runs the gradient graph
through `substitute_maps` + `schedule_auto` + `to_wgsl`, and compares the
GPU gradients w.r.t. the input and `wq` against the f64 interpreter:
max error 5e-7, 42 kernels. There is no hand-written backward kernel for
any of it; rope and GQA gradients come out of the `iota`-gated scatter.

### GPU: a training step, hand-written vs derived

`GpuTrainTransformerBlock` is tang-gpu's hand-written trainable block (no
rope, no biases; its attention backward runs on the CPU over copied
activations — nobody wrote the GPU version). The derived step is the same
block built in IR with `rope: false, bias: false`, `backward` for the
input and all seven weights, then `substitute_maps` + `schedule_auto` +
`to_wgsl`, one program. All nine outputs match to ~1e-6.

| seq | hand fwd+bwd | derived fwd+bwd |
|---|---|---|
| 32 | 60.9 ms | **4.87 ms** |
| 128 | 174 ms | **10.1 ms** |

The asymmetry is the point: the hand path has a fast forward and a slow
backward because a GPU backward for attention was never written. The
derived path has neither problem, because nothing about it was written.

### A derived training loop

`examples/train_loop.rs`: block forward + MSE loss + backward for every
weight as one 47-kernel program, Adam as one program per weight (bias
corrections are a 2-element input, updated each step), new weights and
moments fed back GPU-side with `GpuBuffer::copy_from`. Loss on a fixed
target: 0.606 → 0.038 in 30 steps. Nothing in the loop is hand-written.

## Kernels vs tensor IR on an RTX 3090 (Vulkan)

**Date:** 2026-10-12. Ubuntu 24.04, driver 580, wgpu Vulkan backend. Same
branch, same code, no per-chip knowledge anywhere in the compiler; the
tuning cache is keyed by device name. Readback floor is ~1.25 ms here.

All 11 GPU tests pass with the same ~1e-6 errors as on the M4.

| op | hand | best derived | schedule |
|---|---|---|---|
| causal attention 1024, d=64 | 3.85 ms | **1.40 ms** | tiled rows×lanes |
| causal attention 512 | 1.88 ms | 1.42 ms | tiled |
| matmul 1024² | 2.31 ms | 2.35 ms | 16×16 tile |
| rmsnorm 2048×2048 | 5.96 ms | 6.05 ms | naive (16 MB readback) |
| swiglu 4M | 6.62 ms | 6.22 ms | map |
| adam step 4M | 5.88 ms | 5.97 ms | map |
| transformer block seq 128 | 3.13 ms | **2.50 ms** | per-kernel autotune |
| transformer block seq 512 | 10.4 ms | **6.20 ms** | per-kernel autotune |
| decode block q1 / kv 256 | 2.49 ms | 1.71 ms | per-kernel autotune |
| decode block q1 / kv 1024 | 3.85 ms | 1.68 ms | auto |
| training step seq 32 | 53.6 ms | **3.71 ms** | auto |
| training step seq 128 | 175 ms | **9.64 ms** | auto |

What the second chip taught the compiler (all of it landed in this
branch):

- The block initially *lost* here (5.5 ms vs 3.0) while winning on the
  M4. Per-kernel profiling showed `substitute_maps` folding the
  normalization, residual add and SwiGLU activation (with its `exp`) into
  the projection matmuls' bodies — recomputed once per output column.
  The M4 shrugged; the 3090 did not. Substitution is now cost-aware.
- Folding a matmul into its bias-add map (one thread per point) hid the
  reduction from every tiled schedule. Reductions now stay materialized
  under maps; that is what let the autotuner pick the 16×16 tile for the
  projections and take the block from 5.1 ms to 2.5 ms.
- The staged tiles were laid out point-major, so consecutive lanes read
  words 32 apart: 32-way bank conflicts. Point-minor layout took the
  16×16 matmul tile from 6.25 ms to 2.35 ms (hand parity) on the 3090
  and changed nothing on the M4.
- `schedule_auto`'s rules of thumb were distilled on the M4 and are
  wrong for matmul here (naive 2.77 vs tiled 2.35); `split` on a 66-wide
  tuple is catastrophic on NVIDIA (59 ms). The per-kernel autotuner is
  the answer on both chips, with the rules as its starting point.

M4 figures earlier in this file were measured on a different day;
absolute times on that machine drift by ±2× with its power state (the
same commit measured 1.35 ms and 3.8 ms for tiled attention an hour
apart). Compare columns within one table, not across days.
