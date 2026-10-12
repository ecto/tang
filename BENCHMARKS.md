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
of the total. Correctness: 5e-5 relative against the f64 evaluation at
seq 128 (TF32 inputs).

### Attention on tensor cores

`cuda_flash_kernel` recognizes the fused attention comp — an
online-softmax reduction over keys whose score is an inlined bilinear
comp (here two terms, the rope even/odd halves) plus a mask over
(i, j), whose value is a read along (j, lane), and whose normalizer is
the epilogue map — and emits a flash kernel: 64 queries per 128-thread
block, keys in 64-chunks, `S = Σ_t Q_t·K_tᵀ` and `O += P·V` as wmma
TF32 fragments, the online softmax on the stored score block with the
comp's own mask/value/normalizer expressions evaluated where they
belong. Nothing attention-specific is hard-coded beyond that shape.

| | before | after |
|---|---|---|
| attention kernel, seq 128 | 0.098 ms | **0.042** |
| attention kernel, seq 512 | 1.25 | **0.150** |
| block fwd, seq 128 | 0.62 | **0.59** |
| block fwd, seq 512 | 1.86 | **0.76** |

Six of the block's sixteen kernels now run on tensor cores. Back to
back in one session (GPU shared with another tenant at ~27%), the
like-for-like comparison — tang's CUDA path uses TF32 tensor cores, so
PyTorch gets `allow_tf32` too:

| block forward | PyTorch fp32 | PyTorch TF32 | tang (CUDA, TF32) |
|---|---|---|---|
| seq 128 | 0.32 ms | 0.34 | 0.59 |
| seq 512 | 0.98 | 0.48 | **0.78** |

So at seq 512 the derived block is 1.6× PyTorch-TF32 and faster than
PyTorch-fp32; at seq 128 it is launch-bound (16 launches to PyTorch's
~10 fused cuBLAS/sdpa calls) at 1.8×. The remaining per-kernel time is
the matmul tiles at 1.4–1.9× cuBLAS TF32 and the ten small kernels at
a few µs each; the next rungs are the standard GEMM ladder (128×128
blocks, `cp.async` pipelining) and fusing the small kernels into their
neighbours' prologues. Same error as before (5e-5 relative at seq
128).

What this does and does not show: the compiler derives the *structure* of
the fast kernels (flash attention, register-tiled matmul, sequence
parallelism) from the algebra, with no op-specific code, and picks per
chip. It does not yet generate the instruction-level quality of a vendor
library, and through this backend it cannot reach tensor cores at all.

## Kokoro-shaped ops (M4 Max, Metal, kernel-only)

TTS vocoder ops (Kokoro's iSTFTNet decoder, Vocos-style ConvNeXt) derived
from the IR, against PyTorch 2.14.1 on MPS. Both sides: fp32, 5 warmups,
50 launches, one sync (`examples/kokoro_ops.rs`,
`scripts/torch_kokoro_ops.py`). Every IR output is checked against a
straightforward f64 CPU reference (max error / max |ref| ≤ 1e-5, all ops
pass at 1e-6–1e-5); forward+backward is checked on a small instance
against the f64 interpreter (≤ 2e-7).

These ops need zero padding, so this branch adds **guarded views**: a
`View` with `guard: true` may read outside its source and gets 0 there.
Same-padded and dilated convs, the polyphase upsampler and the iSTFT
overlap-add are all guarded views; nothing op-specific was added. The
backward of a guarded read eliminates an out coordinate (`t = τ − 3k + 9`)
instead of scattering, so `dx` of a conv is a gather with a guarded `dy`
read, not a reduction over all of `t`.

Before = the first cut of this section (single-reduce tiler, im2col for
convs); after = multi-dim tiled reductions + epilogue fusion (below). MPS
rerun alongside the "after" column; it moved between runs (a 0.85→0.71,
f 0.41→0.24 ms), so both are listed. The M4 was noisy on later runs (a
direct/auto 2.43 vs 3.10 ms back to back); the table takes the best of
three full runs.

| op | before (ms) | after (ms) | windowed (ms) | windowed: algebra / schedule | MPS (ms) | windowed / MPS |
|---|---:|---:|---:|---|---:|---:|
| a conv1d 512→512 k7 d3, T 2000 | 2.926 im2col | 2.408 | **2.199** | direct / autotune (1 kernel), 3.34 TFLOP/s | 0.855 | 2.6× slower |
| b upsample 512→256 k20 s10, T 2000 → 20000 | 3.544 | 3.381 | 3.321 | unfold + bmm / auto (3); direct polyphase 3.98 (was 3.75) | 1.072 conv only · 1.274 | 3.1× slower |
| c iSTFT n_fft 20 hop 5, 4000 frames | 0.085 | 0.052 | 0.074 | DFT inlined into OLA / autotune (1) | 0.110 | 1.5× faster |
| d conv (a) + bias + snake | 3.021 im2col | 2.421 | **2.197** | direct + snake epilogue / auto (1 kernel) | 0.847 | 2.6× slower |
| e AdaIN C 512 T 2000, style 128 | 0.214 | 0.213 | 0.197 | auto (6) | 0.245 | 1.2× faster |
| f ConvNeXt block C 512 T 200 F 1536 | 0.571 | 0.544 | 0.550 | auto (8) | 0.246 | 2.2× slower |
| a fwd+bwd (dx, dw) | 6.137 | 6.173 im2col · 7.003 direct | 6.263 im2col · 6.627 direct (was 6.936) | im2col / auto (5) | 2.244 | 2.8× slower |
| f fwd+bwd (all 9 params) | 1.606 | 1.591 | 1.619 | auto (30) | 1.031 | 1.6× slower |

**Windowed staging** (this column): a staged read affine in the lane
axis, `t` and the unrolled inner reduce dims folds the overlapping vars
into one window per chunk. The conv's `x[c, t + 3k − 9]` stages
`64 + 3·6 = 82` floats per channel instead of `7 × 64`, so under the
16 KiB cap the chunk grows from 4 to 7 (the weight tile, 448 floats per
channel, is now the limit). Guarded windows rebuild the guarded coord
from the window position, so zero padding stays at staging. Windowed and
before columns are best of 3 serial runs on an otherwise idle M4 Max
(the machine was still noisy: one run of three was ~1.3× slow across the
board); "before" re-measured alongside: a 2.407, b 3.415, d 2.398,
a fwd+bwd 5.967 im2col. MPS best of 3 from the same session. The
polyphase upsample's direct kernel did not improve (its x window is
already small next to the `w[o, c, r + s·j]` tile).

Full per-schedule run, after (naive / `schedule_auto` / per-kernel autotune):

| op | algebra | naive | auto | autotune | before: auto |
|---|---|---:|---:|---:|---:|
| a | direct conv (1 kernel) | 12.39 | **2.43** | **2.41** | 12.69 |
| a | im2col | 34.34 | 2.96 | 2.95 | 2.93 |
| b | direct polyphase (1 kernel) | 15.87 | 3.64 | 3.68 | 15.73 |
| b | unfold + bmm + transpose | 20.91 | 3.38 | 3.40 | 3.55 |
| c | DFT + OLA | 0.112 | 0.052 | 0.054 | 0.086 |
| d | direct + snake (epilogue-fused: 1 kernel) | 12.57 | **2.46** | **2.42** | 12.88 (2 kernels) |
| d | direct, conv inlined under snake (1 kernel, untiled) | 12.32 | | | 12.36 |
| d | im2col + snake map | 34.33 | 2.95 | 2.94 | 3.02 |
| e | | 0.437 | 0.213 | 0.442 | 0.214 |
| f | | 1.20 | 0.544 | 0.608 | 0.571 |
| a fwd+bwd | direct | 36.71 | 7.00 | 7.03 | 44.03 |
| a fwd+bwd | im2col | 73.71 | 6.17 | 6.20 | 6.14 |
| f fwd+bwd | | 4.20 | 1.59 | 1.67 | 1.61 |

**Merged with #34** (generalized epilogues, `merge_siblings`, `relayout`,
`merge_reduce_dims` in `kokoro_ops`' pass list, coalesced + vec4 staging).
Same session, M4 Max, best schedule per op; the #35 binary rerun
alongside (the M4 throttled later in the session, so only the first
back-to-back pair is listed):

| op | #35 windowed (ms) | merged (ms) | merged: algebra / schedule |
|---|---:|---:|---|
| a conv1d | 2.216 | **2.036** | im2col / auto (direct/auto 2.30) |
| b upsample | 3.276 | **2.474** | bmm / auto (2 kernels, was 3) |
| c iSTFT | 0.075 | 0.072 | dft+ola / auto |
| d conv + bias + snake | 2.183 | **2.039** | im2col / autotune (direct/auto 2.31, 1 kernel) |
| e AdaIN | 0.194 | **0.157** | auto (4 kernels, was 6) |
| f ConvNeXt block | 0.579 | **0.429** | autotune (7 kernels, was 8) |
| a fwd+bwd | 6.383 | **4.816** | im2col / auto |
| f fwd+bwd | 1.643 | **1.246** | auto (30) |

Direct conv alone (one kernel) is unchanged within noise: interleaved
#35/merged runs on the throttled machine gave 2.81/3.98, 3.93/3.89,
3.81/3.83 ms. The exp/log epilogue rule survives as a schedule
heuristic (`unfuse_heavy_epilogue`): under a register tile, an epilogue
with exp/log runs as its own map.

What changed, and where the IR still loses:

- **Multi-dim reductions tile.** `tile_kernel` now chunks the largest
  reduce dim as `r0` and runs the others (`u{i}`: a conv's taps, the
  polyphase `j`) as inner loops inside each chunk step — unrolled when
  their product is ≤ 16, outside the register-tile copies so all 16
  copies share one staged read per step. Staged regions just gain the
  inner vars (they already supported them for nested comps).
  `tileable_red` accepts any reduction whose non-chunked dims multiply to
  ≤ 32. No conv-specific code: dx of the conv (`red = (o, k)`) and the
  polyphase upsample (`red = (j, c)`) take the same path.
- **Guards are applied when staging.** A guarded read used to stage the
  (possibly wrapped) value and `select` at every use — 16 copies × 7 taps
  of compares per step; 16.6 → 5.5 ms on its own. The region key now
  carries the guard's source coords, so the zero padding happens once per
  staged element.
- **Occupancy, not staging, was the last 2×.** Inner dims multiply every
  staged region (x: chunk·7·64, w: chunk·7·64). At chunk 8 that was 28 KiB
  of workgroup memory — one workgroup per M4 core — and 5.0 ms; capping
  staged memory at 16 KiB (chunk 4, also the WebGPU default limit) gives
  2.4 ms. The cap applies to every tiled kernel, halving the chunk until
  it fits.
- **Narrow trailing point dims are uniform, not lanes.** `tile_axes` picks
  the last two point dims ≥ 16 as the tile's x / y, so `dw[o, c, k]`
  tiles over `(o, c)` with `k` per-workgroup (36 → 7 ms fwd+bwd) and the
  polyphase `[o, q, r]` tiles over `(o, q)` with `r` uniform (15.6 →
  3.6 ms).
- **Epilogue fusion.** When a tiled reduction's only consumer is a map
  over the same points (identity view, unguarded), `schedule_auto` moves
  the tile schedule onto the map and inlines the reduction; the tiled
  kernel evaluates the map at write-out. (d) is now one kernel at (a)'s
  cost (2.42 vs 2.41 ms), and ConvNeXt's residual add fuses into the
  second pointwise matmul (9 → 8 kernels). Autotune sees an epilogue map's
  candidates as its reduction's tile candidates. A GELU epilogue measured
  *slower* fused (block 0.60 vs 0.56 ms), so maps containing exp/log stay
  separate — a measured rule, not a principled one.
- **Still 3× behind MPS on convs.** 2.4 ms for 7.3 GFLOP is ~3 TFLOP/s;
  MPS's 0.71 ms is ~10. The tile is the generic 16×16 threads × 4×4
  registers with scalar f32 shared-memory reads; MPS uses simdgroup
  matrix ops. The x region also stages each tap separately (`chunk·7·64`
  floats where the sliding window needs `chunk·(64 + 18)`), which is what
  forces chunk 4. A windowed region (merge an inner var into the offset
  span when its coefficient is a multiple of a lane coefficient) and
  `simdgroup_matrix`-style lowering are the next two steps.
- **fwd+bwd of the conv: im2col still wins (6.2 vs 7.0 ms).** Both
  gradients tile now, but `dw` re-stages `dy` for each of the 7 taps
  (`k` is per-workgroup) and the guarded `x` read inside it is staged per
  tap as above.

- **The upsample.** `conv_transpose1d` on MPS takes a 24 ms slow path; the
  same math as a polyphase conv1d (2 taps, 2560 output channels) is
  1.17 ms there. The IR's direct polyphase comp (reduce over `(j, c)`) now
  tiles (3.6 ms, was 15.6); unfold + one matmul per phase is 3.4 ms. The IR computes `conv_transpose1d(p = 0)` cropped to 20000
  samples — identical FLOPs to `p = 5`, offset by 5 samples (a `p = 5`
  form needs a third tap or a guarded weight read).
- **iSTFT wins** because `schedule_auto` inlines the 11-bin inverse DFT
  into the overlap-add (each output sample recomputes its 4 frames), so
  the whole head is one launch with no frame buffer. `torch.istft` is
  several kernels (complex polar, irfft, window, fold, envelope divide).
- **AdaIN wins** on launch structure: the stats are reductions and
  everything elementwise splices into one map.
- **ConvNeXt at T = 200 is launch-bound** (9 kernels forward, 30 for
  forward+backward; the two pointwise matmuls are only 0.3 GFLOP each).
  LayerNorm's mean and variance are separate reductions over C, each
  re-reading the depthwise conv output.
- **Autotune ≥ auto here, sometimes worse** (AdaIN 0.45 vs 0.21 ms,
  ConvNeXt 0.88 vs 0.57). It times each kernel in isolation for 10
  launches with the rest of the program cold; on these sub-0.1 ms kernels
  that measurement is noise-dominated. The tuner should time in program
  context and with more launches.
- Guarded reads of a map keep that map materialized (`f(0) ≠ 0` in
  general), so a conv over `snake(adain(x))` would read a buffer, not
  splice. Splicing is valid whenever the map is zero on zero input
  (snake is); not done yet.

## Precision schedules (M4 Max, Metal, kernel-only)

Precision is a per-node schedule field (`set_precision`, see
`docs/tensor-ir.md`): storage format of each buffer, accumulate format of
each reduction. `tune_precision` starts from the per-kernel autotuned f32
schedule, lowers one buffer at a time to f16 (largest first), runs it on
the device against the f64 CPU reference, and keeps it only if the whole
program stays within the error budget and is ≥ 3% faster; the final
choice is re-timed best-of-3 against f32 and dropped if the gain does not
hold. Error = max |got − f64| / max |f64|. "Predicted" is the f64
interpreter's emulation (`precision_error`) of the same assignment on a
small instance of the same builder.
`cargo run --release -p tang-bench --example kokoro_ops -- precision`.

| op | budget (rel) | f32 ms | chosen ms | speedup | chosen assignment | measured err | predicted err (small) |
|---|---|---:|---:|---:|---|---:|---:|
| a conv1d 512→512 k7 d3 T2000 | 1e-2 | 2.201 | 2.201 | 1.00× | all f32 | 4.0e-6 | 6.8e-7 |
| a conv1d 512→512 k7 d3 T2000 | 1e-3 | 2.204 | 2.204 | 1.00× | all f32 | 4.0e-6 | 6.8e-7 |
| b upsample 512→256 k20 s10 | 1e-2 | 3.919 | 3.662 | **1.07×** | w[256×512×20] f16 | 6.2e-5 | 1.4e-4 |
| b upsample 512→256 k20 s10 | 1e-3 | 3.931 | 3.825 | 1.03× | w[256×512×20] f16 | 6.2e-5 | 1.4e-4 |
| c iSTFT n_fft20 hop5 T4000 | 1e-2 | 0.073 | 0.073 | 1.00× | all f32 | 1.3e-6 | 9.3e-8 |
| c iSTFT n_fft20 hop5 T4000 | 1e-3 | 0.074 | 0.072 | 1.02× | OLA output f16 | 3.0e-4 | 2.4e-4 |
| e AdaIN C512 T2000 style128 | 1e-2 | 0.250 | 0.250 | 1.00× | all f32 | 7.3e-7 | 6.0e-7 |
| e AdaIN C512 T2000 style128 | 1e-3 | 0.249 | 0.249 | 1.00× | all f32 | 7.3e-7 | 6.0e-7 |
| f ConvNeXt C512 T200 F1536 | 1e-2 | 0.697 | 0.697 | 1.00× | all f32 | 9.7e-6 | 4.5e-8 |
| f ConvNeXt C512 T200 F1536 | 1e-3 | 0.695 | 0.695 | 1.00× | all f32 | 9.7e-6 | 4.5e-8 |
| matmul 1024³ | 1e-2 | 0.761 | 0.761 | 1.00× | all f32 | 1.8e-6 | 1.5e-6 |
| matmul 1024³ | 1e-3 | 0.758 | 0.758 | 1.00× | all f32 | 1.8e-6 | 1.5e-6 |

Every f16 *storage* trial fit both budgets (errors 1e-5–5e-4; an f16
output costs ~3e-4, the half-ulp of the result). The search said no
because almost nothing got faster:

- **Without matrix units, f16 storage buys bandwidth only, and these
  ops are not bandwidth-bound.** The convs and the 1024³ matmul run at
  ~3.3 TFLOP/s from staged f32 tiles: compute-bound, halving the bytes
  read changes nothing (a: x f16 2.18, w f16 2.23 vs 2.20 ms; matmul:
  a/b f16 0.76 vs 0.76). The upsample is the one op whose weight read shows: its
  10.5 MB `w` in f16 is 3–7% faster across runs. AdaIN, iSTFT and the
  ConvNeXt block are a few hundred µs of 6–8 small kernels — launch- and
  latency-bound at 80–100 GB/s against the M4 Max's ~400.
- **f16 outputs never pay** (a: 2.21 vs 2.20 ms; matmul 0.80 vs 0.76):
  two threads own the halves of one word, so the store is
  two atomics. An emitter that gives each thread a pair of points (the
  register tile already does) can write whole words.
- Run-to-run noise on the M4 is ±5% for sub-ms programs, which is why a
  kept choice is re-timed; earlier runs with a 2% bar "kept" AdaIN and
  ConvNeXt assignments that re-timed at 0.97–1.00×.

**Where the budget rejects f16.** Accumulators, not storage. f16
accumulation is lowered as rounding after every combine (the numerics of
an f16 accumulator; speed needs native half math), so these trials are
judged on error only. At rel 1e-3:

| op | reduction | measured (GPU) | predicted (interp., sequential) |
|---|---|---:|---:|
| a conv1d | out, Σ over (c, k) = 3584 terms | 3.1e-2 | 5.6e-3 (c = 32) |
| b upsample | Σ over (j, c) = 1024 | 1.5e-2 | 2.6e-3 (c = 32) |
| e AdaIN | variance Σ_t (x − μ)², T = 2000, one thread per row | 1.3e-2 | 7.9e-3 |
| e AdaIN | same, 256-thread split schedule | 2.8e-4 (admitted) | |
| e AdaIN | style projection W·s (128) | 2.9e-3 | 5.0e-4 (style 16) |
| f ConvNeXt | LayerNorm statistic Σ over C = 512 (n10) | 6.9e-2 | 1.5e-4 (C = 32) |
| f ConvNeXt | pointwise F→C matmul (1536) | 1.7e-2 | 1.4e-4 (F = 96) |
| matmul 1024³ | Σ_k, 1024 terms | 1.5e-2 | 1.2e-2 |
| snake `dα = Σ_t` (backward, T = 4096) | unit test | | > 1e-3, rejected |

The predicted column is for the small instances, whose shorter sums
round less; with the same reduction length (matmul, AdaIN) prediction
and measurement agree within ~1.5×. The AdaIN rows are the point: the
same f16 variance is rejected with one thread walking T and admitted
under a split, because the tree of 256 partial sums rounds at small
magnitudes. Precision and thread mapping are one search space.

## Native Metal backend: MSL lowering and simdgroup fragments (M4 Max, kernel-only)

`TensorGraph::to_msl` lowers the same kernels as the WGSL path to Metal
Shading Language and `tang_bench::ir_msl` runs them on buffers it owns
(feature `metal`; see `docs/tensor-ir.md`). Two things change against
WGSL: the kernels compile through Apple's compiler directly (no
wgpu/naga, native `half`), and an f16 × f16 → f32 contraction can be
scheduled onto `simdgroup_matrix` 8×8 fragments (`set_fragment`).

Method: every timing is 50 launches encoded into one command buffer
after a warm-up, one wait, divided; best of 3. WGSL is the existing
`kernel_only` / `kokoro_ops` method (one wgpu submission per run, one
sync); MPS is `scripts/torch_kokoro_ops.py` (5 warmups, 50 calls, one
`torch.mps.synchronize`), f32 and with `f16` (weights and activations
in half). All four were run serially in one session on an otherwise
idle M4 Max. Every IR result is checked over the full output against an
f64 reference (`err` = max |got − f64| / max |f64|).
`cargo run --release -p tang-bench --features metal --example msl_kernels`,
`... --example kokoro_ops -- msl`.

### Matmul and attention

| shape | WGSL best | MSL f32 best | MSL f16 operands, fragments | MPS f32 | MPS f16 |
|---|---:|---:|---:|---:|---:|
| matmul 1024³ | 0.587 (reg8), 3.66 TFLOP/s | 0.492 (auto), 4.37 | **0.191, 11.23** (err 8.3e-5) | 0.194, 11.08 | 0.176, 12.17 |
| matmul 128×576×1536 | 0.176 (auto), 1.29 | 0.090 (reg2), 2.52 | **0.040, 5.72** (9.8e-5) | 0.045, 5.00 | 0.046, 4.98 |
| matmul 512×576×1536 | 0.329 (auto), 2.75 | 0.214 (auto), 4.24 | **0.084, 10.79** (1.1e-4) | 0.097, 9.33 | 0.097, 9.37 |
| attention n=512, d=64, causal | 0.240 (flash_auto) | 0.078 (naive) | — | 0.043 (sdpa) | 0.054 |
| attention n=1024 | 0.746 (flash_auto) | 0.216 (naive) | — | 0.078 | 0.105 |

(ms, TFLOP/s.) The same f16 operands *without* fragments (scalar 4×4
register tile) run 0.498 / 0.099 / 0.213 ms: f16 storage alone buys
nothing on these compute-bound shapes; the matrix unit is the whole win.
An f16 output costs nothing either way now (native `half` stores: 0.190
vs 0.191 ms; the WGSL path's packed-word atomics made it slower).

### Kokoro ops

| op | WGSL best | MSL f32 best | MSL tuned @ 1e-3 (choice) | err | **MSL windowed fragments** | MPS f32 | MPS f16 |
|---|---:|---:|---:|---|---:|---:|---:|
| a conv1d 512→512 k7 d3 T2000 | 2.196 (3.34 TFLOP/s) | 2.296 (3.20) | 1.057 (6.94) x, w f16, conv on fragments | 5.3e-5 | **0.693 (10.6)** | 0.920 (7.98) | 0.672 (10.9) |
| b upsample 512→256 k20 s10 | 3.274 | 2.713 | 2.366: w, polyphase output f16; fragments rejected (3.77) | 4.6e-4 | **1.317** (8.0), err 8.4e-5 | 1.041 conv / 1.179 | 0.892 / 1.108 |
| c iSTFT n_fft 20 hop 5 T4000 | 0.073 | **0.018** | 0.023: all f32 | 1.4e-6 | — | 0.119 manual / 0.325 | 0.110 / 0.310 |
| d conv (a) + bias + snake | 2.201 | 2.363 | 1.066 x, w f16, fragments + snake epilogue (1 kernel) | 8.2e-5 | **0.698** (tuned; 1 kernel) | 0.820 | 0.786 |
| e AdaIN C512 T2000 style128 | 0.205 | **0.030** | 0.028: x, W, out f16 | 5.1e-4 | — | 0.250 | 0.272 |
| f ConvNeXt C512 T200 F1536 | 0.553 | 0.345 | 0.318: w1, normed act f16, pw1 on fragments | 2.0e-4 | 0.312 (pw2 eligible now, not kept: 0.420 forced) | 0.231 | 0.413 |
| a fwd+bwd (dx, dw) | 6.064 | 4.125 (im2col) | f32 only | | see next row | 2.194 | 2.064 |
| a fwd+bwd, dy given (y, dx, dw) | | 6.4 (autotune, loaded machine) | y, dx, dw on fragments (err 3.4e-4) | 3.4e-4 | **2.069** (y 0.69 · dx 0.69 · dw 0.67) | 2.194 | 2.064 |
| f fwd+bwd (all params) | 1.570 | 0.927 | f32 only | | — | 0.962 | 1.077 |

**Windowed fragments column** (PR on top of #37, `examples/conv_frag.rs`
and `kokoro_ops -- msl`). Measured on a *loaded* machine (a browser
process held the GPU; the f32 rows above re-measured 1.0–2× slower than
listed), so every number in this column is the min of two interleaved
rounds with MPS in the same window. MPS in that window: a 0.810 (f16) /
0.864 (f32), b conv-only 0.897 / 1.059 and with interleave 1.063 / 1.241,
d 0.763 / 0.825, fwd+bwd 2.071 / 2.290. Against those, a and d are ahead
of MPS f16 (0.69 vs 0.81, 0.70 vs 0.76) and fwd+bwd ties it; against the
idle-machine MPS figures in the table, a is 3% behind MPS f16 (0.693 vs
0.672), d ahead (0.698 vs 0.786), fwd+bwd level (2.07 vs 2.06), and the
upsample still behind MPS f32 (1.32 vs 1.04). Every output is checked
against f64; the forward ops sit at 5–8e-5, fwd+bwd at 3.4e-4 under the
1e-3 budget (`tune_precision` puts y, dx and dw all on fragments).

What changed:

- **Windowed staging.** A conv-shaped operand (one reduce dim read
  through its own forms, the free axis plus all other reduce dims through
  one form `st·n + Σ coef·tap`) is staged once per (channel chunk, time
  tile) as one zero-padded window — 16 × 83 halves for op a instead of a
  per-K-step im2col gather — de-interleaved by stride phase, and each tap
  loads its fragments from it at a shifted offset. The weight is staged
  once per chunk for all taps as tap planes `[k][o][c]`, one thread per
  `(o, c)` copying its taps with constant offsets. A/B in one process:
  conv 0.693 vs 1.048 ms gather, upsample 1.317 vs 3.630.
- **Where the time went** (op a): the MMA loop alone (staging stubbed)
  is 0.59 ms = 12.4 TFLOP/s; staging the weight through the generic view
  read cost +0.6 ms, the tap-fastest store order bank-conflicted (stride
  bm·cb) for another +0.3; one thread per (o, c) walking c fixed both.
  Measured and dropped: double-buffered staging (one barrier per chunk,
  0.74 vs 0.69), 128-wide n tiles (2×4 groups 0.82; 2×2 groups spill),
  per-lane `thread_elements` assembly of A from a natural-layout tile
  (1.57), 8 vs 16 channel chunks (equal).
- **Upsample** is bound by its weight reads `w[o, c, r + s·j]`: for one
  phase they are 40 B apart (0.93 ms with the loads stubbed vs 1.32).
  Ordering tiles batch(phase)-fastest so the ten phases of one tile share
  lines in L2 took it from 1.91 to 1.32. Folding the phase into the tile
  (one window, ten weight slices) is the remaining step.
- **Backward.** `dy = 1` (the fwd+bwd row) collapses dx and dw to plain
  sums, so the fragment row uses `L = Σ y ⊙ gy` (MPS times
  `backward(ones)`, same cost). dx is a transposed conv over gy and takes
  the window path (its weight view is guarded but provably in range, so
  it is staged as affine). dw is a GEMM over t per tap; its
  `x[c, t + 3k − 9]` is guarded, so it now loads fragments straight from
  device memory on every K chunk whose reads are all in range and
  gathers only the edge chunks: 2.24 → 0.67 ms. `fragment_plan` accepts
  `v_i · v_j` bodies with unread extra args (gradients keep the product
  rule's dropped factor).
- **ConvNeXt's second pointwise** (operand = inlined GELU map) is now
  eligible: the map is evaluated at staging and rounded to f16. It does
  not pay here (forced 0.420 vs 0.312 kept): GELU is recomputed per
  output-row tile, and at 0.3 GFLOP the block is launch-bound.

"Tuned" is `ir_msl::tune_precision` from the MSL-autotuned f32 schedule:
first, per contraction, the move "operands f16 + fragments"; then f16
storage per buffer, largest first; then f16 accumulators (native
rounding through `half`, timed like any move); a move is kept if the
program stays within budget vs f64 and gets ≥ 3% faster, and the final
choice is re-timed against f32. The 1e-2 budget made the same choices.

What this shows, and where fragments do not help:

- **Matmul reaches MPS.** Both operands are affine with a unit stride,
  so interior tiles load fragments straight from device memory (the
  spike's kernel shape: 64×64 tile, 2×2 SIMD groups of 4×4 fragments);
  ragged edge tiles gather into threadgroup memory with zero padding.
  11.2 TFLOP/s at 1024³ vs MPS f32 11.1 and f16 12.2; ahead of MPS at
  the two transformer shapes.
- **Conv as implicit GEMM: 2.2× over the f32 tile, still 1.15× (f32) /
  1.6× (f16) behind MPS.** The weight `w[o, (c,k)]` flattens to one
  unit-stride K and loads directly; `x[c, t + 3k − 9]` is guarded and
  dilated, so it is gathered per K-chunk through its view (zero padding
  at staging) — a 32×64 half tile per step with a div/mod per element.
  Staging *both* operands cost 2.4 ms, no better than f32; loading the
  weight directly is what got to 1.06. A windowed `x` stage (the f32
  tiler's trick) and a larger K chunk are the next steps.
- **The upsample's fragments lose (3.77 vs 2.71 ms f32).** Its weight
  read `w[o, c, r + s·j]` does not flatten `(j, c)` onto one stride and
  the phase `r` is a uniform batch dim, so both operands are gathered,
  with small K (1024) per tile. The search rejected it and kept f16
  storage instead (1.15×).
- **ConvNeXt: fragments apply to one of the two pointwise matmuls.** The
  first reads the materialized LayerNorm output (made f16), the second
  reads GELU(h + b) through a substituted map, which `fragment_plan`
  refuses (operands must be f16 buffers). At 0.3 GFLOP per matmul the
  block is launch-bound anyway (8 kernels).
- **Much of the f32 gain is the backend, not the kernel.** The MSL f32
  kernels are the WGSL emitter's kernels, yet AdaIN (6 small kernels)
  runs 0.030 vs 0.205 ms and the iSTFT 0.018 vs 0.073. With one command
  buffer per run (wgpu's pattern) MSL is still 0.038 / 0.023 ms, so the
  gap is wgpu/naga per-dispatch cost and code generation, not batching.
  The same holds for the block-shaped work: attention's naive form
  0.216 vs 0.746 ms and conv fwd+bwd 4.1 vs 6.1 ms. Big single kernels
  (the conv) are a wash between the two.
- **Attention has no fragment lowering yet** (QKᵀ and PV under the
  online-softmax monoid). MSL's best is the materialized naive form;
  MPS's sdpa is 2–3× ahead.
- **Not measured as speedups:** f16 accumulators. The search tried every
  remaining reduction with an f16 accumulator; none was kept (over budget
  or no gain).
- A pre-existing WGSL issue showed up here: the tiler can emit staging
  larger than any device allows (iSTFT's OLA tile, 84016 floats = 328 KiB,
  because the chunk-halving loop stops at chunk 1). wgpu rejects it at
  pipeline creation (autotune catches the panic); Apple's compiler service
  crashes on it (`XPC_ERROR_CONNECTION_INTERRUPTED`). MSL kernels now
  carry `threadgroup_bytes` and the Metal autotuner skips candidates over
  the device limit.
- `kernel_only` hangs in its transformer-block section on this machine
  (also on the base commit `fd193b4`); its matmul/attention rows above are
  from the part that runs.

### Flash attention on simdgroup fragments (M4 Max, kernel-only)

`cargo run --release -p tang-bench --features metal --example attention_fragments`
vs `scripts/torch_attention.py`. Both ran interleaved in one
`gpu-lock` window (two rounds; round 2 shown, round 1 within ~5%).
d = dv = 64; q/k/v random in [−1, 1]. `f32-auto` is the derived graph
under `schedule_auto` lowered to MSL (the best f32 schedule we have for
the multi-head graph); `fragment` is the same graph with q/k/v stored
f16 and the fused kernel flagged `fragment`: S = Q·Kᵀ on half
fragments (f32 acc), online softmax in registers (MLX lane layout), P
cast to half, O += P·V on fragments, normalizer fused. Causal runs skip
key tiles the mask covers — derived from the graph (the mask
`select(j − i, −1e30, 0)` is affine in the iota args; evaluated at the
tile box it gives a uniform skip test). TFLOP/s counts 4·h·n²·d (×½ causal).

| heads | causal | n | f32-auto | fragment | vs f32 | TFLOP/s | SDPA f32 | SDPA f16 |
|---|---|---|---|---|---|---|---|---|
| 1 | no | 512 | 0.080 | 0.154 | 0.52× | 0.43 | 0.173 | 0.079 |
| 1 | no | 1024 | 0.250 | 0.314 | 0.80× | 0.85 | 0.077 | 0.100 |
| 1 | no | 2048 | 0.860 | 0.678 | 1.27× | 1.58 | 0.179 | 0.189 |
| 1 | no | 4096 | 3.304 | 1.506 | 2.19× | 2.85 | 0.554 | 0.516 |
| 1 | yes | 512 | 0.088 | 0.148 | 0.60× | 0.23 | 0.046 | 0.054 |
| 1 | yes | 1024 | 0.281 | 0.290 | 0.97× | 0.46 | 0.081 | 0.099 |
| 1 | yes | 2048 | 0.977 | 0.590 | 1.65× | 0.91 | 0.165 | 0.191 |
| 1 | yes | 4096 | 3.743 | 1.238 | 3.02× | 1.73 | 0.409 | 0.463 |
| 8 | no | 512 | 0.422 | 0.191 | 2.21× | 2.81 | 0.081 | 0.073 |
| 8 | no | 1024 | 1.659 | 0.780 | 2.13× | 2.75 | 0.251 | 0.233 |
| 8 | no | 2048 | 6.449 | 2.951 | 2.18× | 2.91 | 0.882 | 0.757 |
| 8 | no | 4096 | 26.50 | 12.59 | 2.11× | 2.73 | 3.654 | 2.992 |
| 8 | yes | 512 | 0.508 | 0.175 | 2.91× | 1.54 | 0.065 | 0.069 |
| 8 | yes | 1024 | 2.068 | 0.440 | 4.70× | 2.44 | 0.167 | 0.150 |
| 8 | yes | 2048 | 8.224 | 1.637 | 5.02× | 2.62 | 0.534 | 0.461 |
| 8 | yes | 4096 | 33.23 | 5.788 | 5.74× | 2.97 | 1.951 | 1.626 |

(ms.) Honest reading:

- Against our own f32 path the fragment kernel wins 2–5.7× once there is
  enough work (8 heads, or n ≥ 2048 single-head); causal gains more
  because skipped tiles are free. Max |fragment − f32| is 0.6–1.9e-3 (P
  and q/k/v in half); `tune_precision` keeps it under a 1% budget vs f64.
- Against PyTorch MPS SDPA it is **3–4× slower** at 8 heads × 4096 and
  worse on small single-head problems (grid of n/32 threadgroups is too
  small to fill the GPU: 16 groups at n = 512). ~3 TFLOP/s against
  SDPA's ~10. Known gaps, in likely order of payoff: K/V staged by a
  generic `arg_read` gather each tile (scalar loads, two barriers, no
  double buffering, no direct device fragment loads); BK = 32 keys and
  8 rows per SIMD group (MLX uses larger query tiles per SIMD group and
  `exp2` with log2e folded into the scale); single-head small-n needs
  key-split (flash-decoding) parallelism.
- The CUDA wmma flash kernel from #34 (`flash_tile`) matches a narrower
  pattern (exact `(score, 1, v)` slots, unscaled score, d_lanes ∈ 16..64);
  the Metal `attention_plan` is a superset (literal scale, ragged d/n/dv,
  guarded views, any per-row slots, dv blocked as a grid axis, GQA via
  `[hk, g]` heads). Folding `flash_tile` onto `attention_plan` is a
  follow-up.

#### Speed pass (supersedes the fragment column above)

Same harness (`ATT_HEADS`, `ATT_CAUSAL`, `ATT_SAMPLES` select shapes;
best of 7), SDPA from the same `gpu-lock` window. The 4-head rows at
n = 128–512, non-causal, are the phoneme-encoder shape of the voice
model.

| heads | causal | n | f32-auto | fragment | vs f32 | TFLOP/s | SDPA f32 | SDPA f16 | vs SDPA f16 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | no | 512 | 0.080 | **0.029** | 2.75× | 2.32 | 0.169 | 0.084 | 2.9× faster |
| 1 | no | 1024 | 0.250 | **0.052** | 4.82× | 5.17 | 0.076 | 0.099 | 1.9× faster |
| 1 | no | 4096 | 3.311 | **0.504** | 6.57× | 8.52 | 0.555 | 0.517 | 1.03× faster |
| 1 | yes | 512 | 0.086 | **0.030** | 2.88× | 1.12 | 0.046 | 0.054 | 1.8× faster |
| 1 | yes | 1024 | 0.280 | **0.054** | 5.21× | 2.50 | 0.080 | 0.099 | 1.8× faster |
| 1 | yes | 4096 | 3.768 | **0.298** | 12.6× | 7.20 | 0.408 | 0.462 | 1.6× faster |
| 4 | no | 128 | 0.024 | **0.012** | 1.96× | 1.35 | 0.058 | 0.069 | 5.8× faster |
| 4 | no | 256 | 0.067 | **0.018** | 3.68× | 3.71 | 0.095 | 0.049 | 2.7× faster |
| 4 | no | 512 | 0.222 | **0.041** | 5.44× | 6.57 | 0.051 | 0.055 | 1.3× faster |
| 4 | no | 4096 | 13.01 | 1.967 | 6.61× | 8.73 | 1.722 | 1.496 | 1.31× slower |
| 4 | yes | 4096 | 15.03 | 0.835 | 18.0× | 10.3 | 0.993 | 0.871 | 1.04× faster |
| 8 | no | 512 | 0.422 | 0.069 | 6.09× | 7.74 | 0.080 | 0.071 | 1.03× faster |
| 8 | no | 1024 | 1.660 | **0.221** | 7.53× | 9.74 | 0.250 | 0.235 | 1.06× faster |
| 8 | no | 4096 | 26.15 | 3.411 | 7.67× | 10.1 | 3.459 | 2.696 | 1.27× slower |
| 8 | yes | 512 | 0.482 | **0.047** | 10.3× | 5.72 | 0.062 | 0.066 | 1.4× faster |
| 8 | yes | 1024 | 1.925 | **0.124** | 15.5× | 8.65 | 0.166 | 0.153 | 1.23× faster |
| 8 | yes | 4096 | 29.85 | 1.669 | 17.9× | 10.3 | 1.874 | 1.506 | 1.11× slower |

(ms.) ~3 → ~10 TFLOP/s. Each change measured on its own (8 heads, d 64,
ms before → after; the GPU is shared, so ±10% run to run):

| change | 8h n1024 | 8h n4096 | 8h n4096 causal | 1h n512 | kept |
|---|---|---|---|---|---|
| baseline (PR #40) | 0.785 | 16.46 | 7.77 | 0.182 | |
| Q/K/V staged with `half4` copies when the read is a direct unit-stride f16 view (`direct_f16`, `stage_direct`) | 0.533 | 11.48 | 6.31 | 0.121 | yes |
| `exp2` with log2 e folded into the score | 0.546 | 11.88 | 6.55 | 0.111 | no (no change: fast-math `exp` already is `exp2`) |
| K/V fragments `simdgroup_load`ed straight from device memory when every key tile and value block is full — no staging, no barriers in the key loop | 0.495 | 10.69 | 4.75 | 0.100 | yes |
| epilogue from the accumulator registers (lane owns row `fm`, cols `fn`, `fn+1`): no result tile, no row-stat tile, threadgroup memory 9.6 → 4.6 KB | 0.452 | 7.85 | 4.39 | 0.104 | yes |
| 16 rows per SIMD group (RF 2) via references into the fragment arrays | 0.648 | 11.76 | 8.22 | 0.231 | no (spills) |
| **every constant-trip loop unrolled in the generated text** (`unroll_const_loops`: `{ const uint jf = 3u; … }` scopes) | **0.230** | **3.85** | **1.98** | **0.031** | yes |
| then RF 2 again (fully unrolled) | 0.279 | 4.13 | 2.11 | 0.107 | no |
| then BK 64 | 0.266 | 4.37 | 2.55 | 0.035 | no |
| then `exp2` again | 0.255 | 3.85 | 2.04 | 0.030 | no (neutral) |

What this shows:

- **The big one was the compiler, not the algorithm.** The kernel indexed
  `simdgroup_matrix` arrays and `thread_elements()` with loop variables;
  Metal left some of those loops rolled and the fragments went through
  the stack. Ablations before the fix said as much: dropping the K/V
  loads or the `exp`s changed nothing, so the MMA loop itself was
  stalled. Unrolling in the emitter (constant indices everywhere)
  doubled throughput on every shape. Any fragment kernel that indexes
  fragment arrays dynamically should get the same pass.
- Bigger register tiles (16 rows per SIMD group, BK 64) lose on M4 at
  d = 64: register pressure costs more occupancy than the extra
  fragment reuse buys. MLX's 8 rows × BK 32 is the right shape here.
- Key-split (flash-decoding) was not needed for the targets: 1 head
  n = 512 runs 0.029 ms, 2.75× under f32-auto and 2.9× under SDPA f16,
  despite only 16 threadgroups. It stays the lever for single-query /
  very short-query decode.
- Still behind SDPA f16 by 1.1–1.3× at n = 4096 (8 heads; 4 heads
  non-causal). Ragged shapes (n not a multiple of 32, dv not a multiple
  of the 64-column block) take the staged `half4`/gather path with
  barriers.
