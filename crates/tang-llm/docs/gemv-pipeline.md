# A pipelined 4-bit decode GEMV (design, not built)

Status: design only, not built or measured. Dense 8B/32B speed work is paused (2026-10-21).
This page is what the next attempt should build, and why it should pay.

## Where the time is

Qwen3-8B-4bit, one stream, H200, main at 701af6c: 4.82 ms a step (207 tok/s). The step reads
~4.26 GB of 4-bit weights (36 layers × 108.5 MB plus a 350 MB head), so the GEMV is the step.

| | ms a step | basis |
|---|---|---|
| 4-bit GEMV (`gemv_q4_xv4_m1`), 145 launches | ~2.6 | profile's 4.76 ms, less the 2.2 ms #80 removed |
| attention, norms, the rest | ~1.0 | profile after #82/#83 |
| launch gaps, the logits sync, sampling | ~1.2 | profile (0.6 + 0.54) plus host time |

`bench_q4_gemv_cuda` (H200, m=1): 0.93 TB/s at 4096×4096, 1.39 at 12288×4096, 1.56 at
4096×24576, 1.89 at 4096×151936. Weighted by the step's bytes, ~1.6 TB/s, a third of the
4.8 TB/s peak.

## Roofline

| GEMV bandwidth | GEMV ms | step ms | 8B tok/s | vs 207 |
|---|---|---|---|---|
| today, ~1.6 TB/s | 2.6 | 4.82 | 207 | |
| 50% of peak, 2.4 TB/s | 1.78 | 4.0 | 250 | +21% |
| 60%, 2.9 TB/s | 1.47 | 3.7 | 270 | +30% |
| 70%, 3.4 TB/s | 1.27 | 3.5 | 286 | +38% |

32B is more GEMV-bound (18.4 GB a step, 14.2 ms today), so it gains a bit more: 70% → ~8.0 ms,
~120 tok/s vs 70.7 (+70%), minus whatever its non-GEMV time is.

The AA-style 8B number (200.5 tok/s today) moves with the solo step. 70% of peak would put it
near 280, still short of Eigen's 303. Closing the rest takes the host-side items (CUDA graphs,
GPU sampling for the lone stream), which now belong to tang-pro.

## What failed, and what that says

| kernel | why it falls short |
|---|---|
| `xs4` (before #80) | a u32 a lane a column: 4 B a load, too few bytes in flight |
| `xv4` (#80, today) | uint4 loads, but one stage of weights in flight a warp, two barriers a 128-word stage, and the x stage is refilled from L2 every stage |
| `xv4` + register double-buffer | registers for the next stage cut occupancy (slower) |
| `xr` (x resident, all loads up front) | one 8-warp block an SM, and each block loads all, then computes: no overlap across blocks (37-49% slower) |

So the requirement is **several stages of weights in flight per SM without spending registers
on them**. That is what `cp.async` (sm_80+) into shared memory is for.

## The kernel

`gemv_q4_xp_m{M}`, for K % 32 == 0 and group % 32 == 0, M = 1..8:

- **Block:** 4 warps, 16 columns (4 a warp), K split as today (`gemv_q4_splits`, shape only,
  ~4 blocks an SM) so a row's sum order never depends on M.
- **x:** this block's K slice staged once into shared memory as rows of float4 (swizzled as in
  `xv`), M × slice × 4 B; for 8B shapes with the split that's ≤ 16 KB a row.
- **Weights:** a ring of 3 stages in shared memory. A stage is 16 columns × 128 words (8 KB),
  and the scales and biases ride along (16 columns × 4 groups × 2 × 2 B). Each lane issues its
  stage's `cp.async.cg.shared.global` 16-byte copies, then `cp.async.commit_group`. Before
  computing stage s it does `cp.async.wait_group 1`, so stage s+1 is still landing while s
  computes. No registers hold weights in flight.
- **Compute:** as `xv` (a lane takes 4 consecutive words, one uint4 from shared memory, one
  scale group). A `__syncwarp()` per stage is enough: a warp only reads its own 4 columns'
  stage, so stage reuse needs no block barrier.
- **Shared memory:** 24 KB of ring + x. That's about 5 blocks (20 warps) an SM with 3 stages
  each, roughly 15 × 8 KB = 120 KB of weights in flight per SM, against the ~35 KB an SM that
  Little's law needs at 4.8 TB/s and ~1 µs.
- **Invariance:** a lane's words in stage order, then the warp tree: the same order for every M,
  as today. The batch-invariance test from the `xr` attempt
  (`q4_gemv_rows_are_batch_invariant`, 1..7 rows vs 8) carries over.

The sum order is the same family as `xv`, a lane's vectors in order and then the warp tree.
The bits will still differ from `xv` because the split and stage boundaries differ, so make it
the default only once the batch and speculate tests pass on GPU.

## Measuring it (one run, ~30 min on an H200)

1. `bench_q4_gemv_cuda` with `TANG_GEMV=xv` / `xp`: the five 8B shapes at m=1 and m=8. Go on only
   if the weighted bandwidth is ≥ 2.4 TB/s (+21% expected on the step).
2. `bench_batch --q4` 8B and 32B at 1 and 8 streams, both kernels from one binary.
3. `speed_bench.py` on `tang-llm serve` for 8B and 32B (kill servers by the binary's full path).
4. The `cuda_q4*` tests, plus batch and speculate.

Expected: a clear ≥10% on 8B solo if the kernel reaches 2.2 TB/s or more. Below 2.0 TB/s,
log it in the speed log's dead ends.
