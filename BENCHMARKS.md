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

## M4 Max, one consistent run (2026-10-12, final branch state)

The sections above were measured on different days as the branch grew
and the machine's power state wandered. This table is one run of
`cargo bench -p tang-bench --bench kernels_vs_ir -- kvi_gpu` on the final
code; the readback floor during this run was ~1.8 ms (the machine was in
its slow state, so absolute times are ~1.5× the earlier sections).

| op | hand | best derived | schedule |
|---|---|---|---|
| causal attention 128 | 3.70 ms | **1.82 ms** | tiled |
| causal attention 512 | 4.91 ms | **2.02 ms** | auto (tiled) |
| causal attention 1024 | 10.9 ms | **3.31 ms** | naive (n² threads) |
| matmul 1024² | 6.68 ms | 5.49 ms | auto |
| rmsnorm 2048×2048 | 2.52 ms | 2.70 ms | auto (split) |
| swiglu 4M | 2.50 ms | 2.71 ms | map |
| adam step 4M | 2.67 ms | 2.71 ms | map |
| transformer block seq 128 | 5.02 ms | **3.74 ms** | auto |
| transformer block seq 512 | 23.9 ms | **10.1 ms** | per-kernel autotune |
| decode block q1 / kv 256 | 5.29 ms | **2.13 ms** | auto |
| decode block q1 / kv 1024 | 8.80 ms | **3.28 ms** | per-kernel autotune |
| training step seq 32 | 65.8 ms | **4.73 ms** | auto |
| training step seq 128 | 179 ms | **7.96 ms** | auto |

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

## Register tiles (both chips)

`TNode.reg = (rx, ry)`: each thread owns an rx×ry block of points of a
16×16-thread tile, with its own accumulators and hoisted reads; the
staged tiles grow to 64×64 points. Same emitter, same staging rules —
the only new idea is "more than one point per thread". Matmul 1024²,
`cargo bench ... kvi_gpu_matmul`:

| | hand | one point/thread tile | 2×2 | 4×4 |
|---|---|---|---|---|
| M4 Max (floor ~1.8 ms) | 6.82 ms | 5.74 ms | 5.51 ms | **5.23 ms** |
| RTX 3090 (floor ~1.25 ms) | 2.27 ms | 2.41 ms | 1.98 ms | **1.92 ms** |

Kernel-only that is roughly 1.5× the hand matmul on both chips. With the
register tiles in the autotuner's candidate list the transformer block
moves again (per-kernel autotune, fresh cache):

| block | hand | derived, autotuned |
|---|---|---|
| M4 seq 128 | 5.02 ms | **3.65 ms** |
| M4 seq 512 | 23.9 ms | **5.61 ms** |
| 3090 seq 128 | 4.94 ms | **3.89 ms** |
| 3090 seq 512 | 13.1 ms | **6.71 ms** |

(3090 hand numbers in this run are ~50% above the earlier run on the same
code; that box was not idle. The derived column moved by the same factor.)

### `schedule_auto` after the second chip

With the register-tiled matmul in its rules, `schedule_auto` alone — no
measurement — now beats the hand path on both chips:

| block, rules only | hand | derived (`schedule_auto`) |
|---|---|---|
| M4 seq 128 | 5.87 ms | **3.44 ms** |
| M4 seq 512 | 23.9 ms | **5.55 ms** |
| 3090 seq 128 | 3.05 ms | **2.61 ms** |
| 3090 seq 512 | 10.8 ms | **5.58 ms** |
| matmul 1024², M4 | 6.81 ms | **4.61 ms** |
| matmul 1024², 3090 | 2.27 ms | **1.90 ms** |

The tuner still exists for the next chip; the rules are what it found.

### Training step at seq 512

| | hand fwd+bwd | derived fwd+bwd |
|---|---|---|
| M4 Max | 1.01 s | **17.5 ms** |
| RTX 3090 | 1.15 s | **11.3 ms** |

The hand path's CPU-side attention backward is quadratic in sequence
length; the derived one is a graph like any other.

### Placement

`shard_reduction` splits a reduce axis into partial reductions (one per
device) and `fold_partials` writes the all-reduce as a comprehension over
the monoid's combine. `examples/shard_attention.rs` runs the flash-fused
attention's 256 keys as two partial (m, l, acc) states on two wgpu
devices and merges them with the online-softmax combine: max error
1.7e-6 (M4) / 1.4e-6 (3090) against the single-device result. Transport
is host-mediated; the IR side of sequence parallelism is done.

## Against PyTorch (RTX 3090, kernel-only)

Everything above compares derived kernels with tang's own hand-written
ones, and every criterion number includes ~1.3 ms of submit + readback.
This section is the comparison that matters: `scripts/torch_bench.py`
(PyTorch 2.10 + cu128, CUDA events, fp32 with TF32 off) against
`examples/kernel_only.rs` (50 launches, one sync, divide) on the same
3090, same shapes.

| op | tang hand | tang derived (best) | PyTorch fp32 | derived ÷ torch |
|---|---|---|---|---|
| matmul 1024² | 1.38 ms | 0.54 ms (4×4 register tile) | **0.112 ms** (TF32: 0.079) | 4.8× |
| matmul 128×576×1536 | 0.38 | 0.20 | **0.030** | 7× |
| matmul 512×576×1536 | 0.65 | 0.28 | **0.053** | 5× |
| causal attention n=512, d=64 | 1.31 | 0.27 | **0.090** (sdpa) / 0.040 (naive) | 3–7× |
| causal attention n=1024 | 3.55 | 0.77 | **0.180** / 0.078 | 4–10× |
| transformer block fwd, seq 128 | 2.25 | 1.97 | **0.295** | 6.7× |
| transformer block fwd, seq 512 | 10.1 | 4.41 | **0.610** | 7× |
| block fwd+bwd, seq 128 | (CPU bwd) | 5.0 | **1.32** | 3.8× |
| block fwd+bwd, seq 512 | (CPU bwd) | 10.1 | **1.73** | 5.9× |
| adam step 4M | 0.85 | 0.54 | **0.395** | 1.4× |

So the derived kernels are 2–4× faster than tang's hand-written ones and
4–7× slower than PyTorch. Both statements are true; only the second one
says how fast this is.

Where the gap is, and how much of it this path can close:

- **Matmul instruction quality.** 0.54 ms at 1024² is ~4 TFLOPS; cuBLAS
  fp32 reaches ~19 TFLOPS on this part. The derived tile uses scalar
  loads, a 4×4 register tile, 32-wide chunks with a barrier each, and no
  double buffering. An 8×8 register tile with `vec4` loads and a 128×128
  block is the known recipe for ~10–15 TFLOPS in a portable shader —
  roughly 3× from here, within ~2× of cuBLAS. That is the ceiling for
  fp32 through WGSL.
- **Tensor cores are off the table.** TF32/FP16 MMA is not reachable from
  WGSL/wgpu today (no cooperative-matrix extension). That is the other
  2–4× PyTorch has, and it is a backend question (MSL simdgroup matrices,
  CUDA via `tang-expr`'s CUDA dialect), not a schedule question.
- **Launch overhead.** The block issued 20 kernels (72 with backward)
  when this table was taken, each as its own command buffer through
  wgpu; seq 128 forward was 1.97 ms for ~0.6 ms of math. Since then:
  one submission per program, and three fusion passes — epilogue
  fusion (bias, activation and the softmax normalizer run on the thread
  holding the final accumulator), views through affine reshapes, and
  sibling-reduction merging (gate and up projections as one
  tuple-valued matmul) — bring the forward block to 16 kernels (62
  with backward). PyTorch's 0.295 ms is ~10 cuBLAS/flash launches.
- **Attention.** Flash-attention proper keeps the (m, l, acc) state in
  registers across a 64-key tile computed as two small tensor-core
  matmuls; ours recomputes q·k cooperatively into shared memory with
  scalar FMAs. Same structure, 4–10× slower execution, for the two
  reasons above.

### After the fusion passes and vec4 stage reads

Same script, both chips, one run each. The 3090 run had the GPU to
itself (1% util); its CPU was loaded (load avg 26), which is why the
PyTorch block numbers — launch-bound on the CPU side — came out higher
than the first table's. Both PyTorch columns are shown.

| op | 3090 before | 3090 after | PyTorch (1st / this run) | M4 after |
|---|---|---|---|---|
| matmul 1024² | 0.54 ms | **0.28** (7.6 TFLOPS) | 0.112 / 0.114 | 0.52 |
| matmul 128×576×1536 | 0.20 | **0.10** | 0.030 / 0.035 | 0.13 |
| matmul 512×576×1536 | 0.28 | **0.16** | 0.053 / 0.054 | 0.23 |
| attention n=512 | 0.27 | 0.25 | 0.090 | 0.26 |
| attention n=1024 | 0.77 | 0.74 | 0.180 | 0.76 |
| block fwd, seq 128 | 1.97 | **1.06** (16 kernels) | 0.295 / 0.566 | 1.14 |
| block fwd, seq 512 | 4.41 | **3.34** | 0.610 / 1.208 | 3.37 |
| block fwd+bwd, seq 128 | 5.0 | **3.15** (62 kernels) | 1.32 / 2.11 | 3.53 |
| block fwd+bwd, seq 512 | 10.1 | **8.19** | 1.73 / 1.99 | 13.2 |
| adam 4M | 0.54 | 0.47 | 0.395 / 0.451 | — |

What moved it:

- **Fusion** (epilogues for tuple-state reductions, views through
  affine reshapes, sibling merging) plus one submission per program:
  the launch-bound seq-128 block, −46% forward, −37% training step.
- **`vec4` reads of the point-minor stage.** A thread's register-tile
  points are contiguous there, so a 4-wide tile reads each operand as
  one vector per r0 instead of four scalars: matmul 1024² 0.54 → 0.28 ms
  on the 3090 and 0.78 → 0.52 on the M4. This is the shared-memory
  load-issue bound the previous section predicted; the next rungs on the
  same ladder are 8×8 tiles with vec4 (now that loads are cheap enough
  to feed them), `vec4` global loads into the stage, and double
  buffering.
- Coalesced staging loads alone changed nothing measurable; the tile
  was never global-load bound at these sizes.
- With vec4 reads in place, a sweep of 8×8 register tiles (chunk 8, 16,
  32), an 8×8-thread block and a 32×8 block at 1024² still lands on
  16×16 threads × 4×4 × chunk 32 on both chips (3090: 0.29 ms vs 0.35
  for 8×8; M4: 0.53 vs 0.50 for 8×8 chunk 16 — within noise), so
  `schedule_auto`'s rule stands. 8×8 is register-pressure bound: more
  FMAs per load but fewer workgroups in flight.
- Double-buffered staging (two stage sets, next chunk prefetched into
  registers during the reduce loop, one barrier per chunk) was tried and
  reverted: on the M4 the dynamically indexed prefetch arrays spill and
  the 4×4 chunk-32 tile went from 0.53 to 3.2 ms; on the 3090 it was
  neutral (0.28–0.33). Without async copies, the extra registers cost
  more than the hidden latency is worth on either chip.

The gap to PyTorch is now 2.5× on matmul, 3–4× on attention and the
block forward, 4× on the training step (against its quieter numbers).
What remains is tensor cores (unreachable from WGSL; a backend
question) and the rest of the instruction-quality ladder above.

### A native dialect: Metal simdgroup matrices (first cut)

`tensor_msl.rs` lowers a matmul-shaped tile — one reduce axis, body
`a·b`, each operand read along one tile axis and the reduce axis, any
affine layout — to MSL with `simdgroup_float8x8` fragments, through
wgpu 25's MSL passthrough. The comp's epilogue (bias, activation) runs
on the stored block. Kernels that don't fit stay WGSL, so a program is a
mix. Correct on ragged shapes (staged tiles) and full tiles (fragments
loaded straight from device memory); 12/12 GPU tests.

M4 Max, kernel-only (`mma` column of `kernel_only`):

| shape | WGSL 4×4 register tile | simdgroup matrices |
|---|---|---|
| 1024² | 0.56 ms | 0.93 |
| 128×576×1536 | 0.13 | 0.27 |
| 512×576×1536 | 0.25 | 0.48 |

So on this chip the first-cut fp32 simdgroup kernel is 1.7–2× *slower*
than the FMA tile. Ablations: the staged version spent two thirds of
its time in scalar staging; fragments straight from memory halved
that but the MMA loop itself runs at ~2.3 TFLOPS, where the FMA tile
runs at 3.7. MLX's fp32 GEMM on the same hardware shows what the
instruction is worth with vectorized staging, swizzled tiles and
deeper software pipelining; that is engineering on top of a lowering
that now exists, and `program_for` keeps WGSL the default until it
wins. The structural claim stands: the same comprehension lowered to a
second dialect with no op-specific code, epilogue included. The CUDA
dialect (tensor cores proper, where the gap to PyTorch lives) is the
same shape of work with nvcc in the loop.

### Tensor cores: the CUDA dialect (RTX 3090, nvcc harness)

The same bilinear tile lowered to CUDA C++ with `wmma` 16×16×8 TF32
fragments (`tensor_cuda.rs`; 64×64 block, four warps of 32×32, operands
staged through shared memory from any affine layout, epilogue on the
stored block). tang-gpu runs on wgpu and cannot dispatch CUDA, so this
is measured by `scripts/cuda_bench.cu`: the derived kernel and
`cublasSgemm` (fp32, then `CUBLAS_TF32_TENSOR_OP_MATH`) on the same
device, CUDA events, 50 launches. Error is against fp32 cuBLAS; TF32
inputs give ~1e-3 relative, as expected.

| shape | tang wmma (TF32) | tang WGSL (fp32) | cuBLAS fp32 | cuBLAS TF32 |
|---|---|---|---|---|
| 1024² | **0.150 ms** (14 TFLOPS) | 0.28 | 0.113 | 0.079 |
| 1024² + bias + silu epilogue | 0.159 | — | 0.107 (+ separate kernels) | 0.079 |
| 128×576×1536 | 0.052 | 0.10 | 0.031 | 0.015 |
| 512×576×1536 | 0.091 (10 TFLOPS) | 0.16 | 0.053 | 0.048 |
| 4096³ | 6.58 (21 TFLOPS) | — | 5.67 | 3.98 |

First cut, no tuning: 1.4–1.9× cuBLAS TF32 at the LLM shapes, 1.65× at
4096³, with the bias/activation epilogue fused for free (cuBLAS needs
extra launches for that). The remaining gap is the standard GEMM
engineering — 128×128 blocks with 8 warps, `cp.async` double buffering,
`ldmatrix`-friendly swizzles — on top of a lowering that now exists.
Together with the WGSL numbers above this is the whole picture: the
derived kernels are 2–4× faster than tang's hand kernels through the
portable path, and within 2× of cuBLAS through the native one, from one
comprehension and no op-specific code.

### The whole block on CUDA (`to_cuda` + cudarc, RTX 3090)

`TensorGraph::to_cuda` emits a program for the CUDA driver: matmul-shaped
tiles as the wmma kernel above, every other kernel the WGSL transpiled
to CUDA C++ (generated WGSL is regular enough that a line-level
translation with a 20-line prelude covers the naive, split and tile
emitters — no second emitter). `tang_bench::ir_cuda::CudaExecutable`
compiles with nvrtc and launches in program order. Two layout passes
landed on the way: `relayout` (a producer's point dims reordered to
its consumer's reads) and `merge_reduce_dims` (adjacent reduce dims
read contiguously fold into one), which turn the attention-output
projection Σ_{g,r,d} into a 576-wide matmul — so 5 of the block's 16
kernels run on tensor cores.

| | CUDA (5 kernels on tensor cores) | Vulkan/WGSL | PyTorch fp32 |
|---|---|---|---|
| block fwd, seq 128 | **0.62 ms** | 1.06 | 0.30 / 0.57 |
| block fwd, seq 512 | **1.86 ms** | 3.34 | 0.61 / 1.21 |

Per kernel at seq 512: five wmma matmuls 0.05–0.18 ms each, the
elementwise and split kernels 2–14 µs each (the Vulkan path's ~0.1 ms
per-kernel floor is gone), and the attention tile 1.25 ms — two thirds
of the total. Attention on tensor cores (q·k and p·v as wmma fragments
inside the online-softmax tile) is the one remaining piece between this
and PyTorch's number; it is the same lowering applied inside a tile
whose reduction is a monoid rather than a sum. Correctness: 5e-5
relative against the f64 evaluation at seq 128 (TF32 inputs).

What this does and does not show: the compiler derives the *structure* of
the fast kernels (flash attention, register-tiled matmul, sequence
parallelism) from the algebra, with no op-specific code, and picks per
chip. It does not yet generate the instruction-level quality of a vendor
library, and through this backend it cannot reach tensor cores at all.
