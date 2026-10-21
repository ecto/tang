# Speed log

Decode speed of `main` after each performance change, measured on rented GPUs. One row per
(change, GPU, model, streams). `bench_batch --q4` (greedy, no speculation, 256 tokens for 8B
and 128 for 32B, `mlx-community/Qwen3-*-4bit`) unless the row says zest-harness. tok/s is
total across streams. Speed of light is the weight bytes a step over HBM bandwidth (8B ~4.26 GB,
32B ~18.4 GB; H200 4.8 TB/s, H100 SXM 3.35 TB/s): 8B 1120 tok/s on an H200 and 790 on an
H100, 32B 260 and 182. % is of that, for one stream. Rows compare against the previous row
for the same GPU, model and streams; GPUs differ run to run (±5%), so treat changes under
~5% as noise unless the same pod measured both.

| date | commit | change | GPU | model | streams | tok/s | % SoL | vs prev |
|---|---|---|---|---|---|---|---|---|
| 2026-10-20 | bfcb90f | baseline | H200 | 8B | 1 | 121 | 10.8% | |
| 2026-10-20 | bfcb90f | baseline | H200 | 8B | 8 | 603 | | |
| 2026-10-20 | bfcb90f | baseline | H200 | 32B | 1 | 42.9 | 16.5% | |
| 2026-10-20 | bfcb90f | baseline | H200 | 32B | 8 | 172 | | |
| 2026-10-20 | c623ce1 | #80 GEMV fill, #81 pool, #82 attention splits, #83 rms_norm | H200 | 8B | 1 | 208 | 18.5% | +72% |
| 2026-10-20 | c623ce1 | (same) | H200 | 8B | 8 | 709 | | +18% |
| 2026-10-20 | c623ce1 | (same) | H200 | 32B | 1 | 69.5 | 26.7% | +62% |
| 2026-10-20 | c623ce1 | (same) | H200 | 32B | 8 | 215 | | +25% |
| 2026-10-20 | c623ce1 | (same, new pod) | H100 | 8B | 1 | 197 | 24.9% | |
| 2026-10-20 | c623ce1 | (same) | H100 | 8B | 8 | 700 | | |
| 2026-10-20 | c623ce1 | (same) | H100 | 8B | 16 | 846 | | |
| 2026-10-20 | c623ce1 | (same) | H100 | 32B | 1 | 67.6 | 37.1% | |
| 2026-10-20 | c623ce1 | (same) | H100 | 32B | 8 | 213 | | |
| 2026-10-20 | c623ce1 | (same) | H100 | 32B | 16 | 232 | | |

Measured on the same H100 pod, waiting to merge (each on top of c623ce1):

| change | model | streams | tok/s | vs c623ce1 |
|---|---|---|---|---|
| draft only at ≤2 streams (zest-harness, speculation on) | 8B | 4 / 8 / 16 | 408 / 542 / 600 | +38% / +61% / +91% |
| (same) | 32B | 8 / 16 | 149 / 160 (≤4) | +58% / +86% |
| device top-64 sampling | 8B | 8 / 16 | 785 / 1002 | +12% / +18% |
| (same) | 32B | 8 / 16 | 221 / 243 | +4% / +5% |
| tensor-core 4-bit GEMM, 9-32 rows | 8B | 16 | 895 | +6% |
| (same) | 32B | 16 | 282 | +21% |
| all three | 8B / 32B | 16 | 1051 / 298 | +24% / +28% |

## Dead ends

Tried and measured at under 3% (or worse):

- **4-bit GEMV, a warp a column, x read through L1 (no shared-memory stage):** 30% slower at one
  row, worse with more rows (L1 traffic is 8x the weight bytes per extra row).
- **GEMV with the next stage's weights double-buffered in registers:** 5-15% slower; register
  pressure cut occupancy.
- **GEMV, 2 columns a warp instead of 4:** 2-10% slower.
- **GEMV K-split sizing, 1 / 4 / 8 / 16 blocks an SM:** within 3% of each other once the
  128-bit loads are in; 4 kept.
- **Device top-k for a lone stream:** 1.5% slower than sampling its one row on the host;
  only batches cut on the device.
- **The VRAM-pressure check on pool misses only (#81)** measured +1.5% alone. It's kept because
  it takes ~2.6 ms of CPU off each step, which matters once kernels stop hiding host time.

## Qwen3-Coder-Next (qwen3next Flash engine)

`Qwen3-Coder-Next-UD-Q4_K_XL.gguf` (unsloth, 49.6 GB), the Flash engine's qwen3next path
(`src/flash/coder.rs`), one stream, greedy. A step reads 3.42 GB at a 1.3k context (3.52 GB at
4k): dense weights, 10 routed + 1 shared expert a layer, head, K/V. Speed of light is that over
HBM bandwidth: 1,400 tok/s on an H200, 970 on an H100 SXM.

AA rows are `flash-aa`, Artificial Analysis-style: a ~1.24k-token chat prompt (one of
`explain` / `refactor` / `write`, the same 140-line Rust file with a different task), 512
output tokens, tok/s after the first token. The fastest public API for this model is Novita at
174 tok/s.

| date | commit | change | GPU | prompt | drafts | tok/s | % SoL | vs prev |
|---|---|---|---|---|---|---|---|---|
| 2026-10-20 | c33c4df | engine (#84), flash-bench 4k ctx | H100 | code ×4k | none | 156 | 16.1% | |
| 2026-10-20 | b9fbcad | fusions, K-split GEMV, 1024-thread norms (#90) | H100 | code ×4k | none | 198 | 20.4% | +27% |
| 2026-10-21 | d9f1d29 | (same kernels) AA | H200 | explain | none | 205.5 | 14.7% | |
| 2026-10-21 | d9f1d29 | | H200 | refactor | none | 205.3 | 14.7% | |
| 2026-10-21 | d9f1d29 | | H200 | write | none | 205.3 | 14.7% | |
| 2026-10-21 | d9f1d29 | | H200 | explain | suffix | 205.1 | | 1.15 tok/window |
| 2026-10-21 | d9f1d29 | | H200 | refactor | suffix | 340.5 | | 2.71 tok/window |
| 2026-10-21 | d9f1d29 | | H200 | write | suffix | 218.7 | | 1.28 tok/window |
| 2026-10-21 | d9f1d29 | separate SiLU launch (now the default) | H200 | explain | none | 217.4 | 15.5% | +2% (noise) |

Stage sweep on that H200 pod (`TANG_CODER_STAGES=1`: each stage alone in a CUDA graph, replayed;
µs a step at a 1.3k context, window 4,856 µs): routed+shared gate|up 876 (771 GB/s), down 801
(501 GB/s), GDN input GEMV 618 (1.6 TB/s), GDN output 301, top-k 286, GDN conv+step 238,
moe combine+norm 214, attention split-K 207 (154 GB/s), router 176, head 174, post-mixer norm
166, attention in/out 260. The stages sum to 4,461 µs, so launch gaps are ~8% of the window.

### Dead ends (Coder-Next)

- **22 → 12 launches a layer with no faster kernels:** +2% (156 → 159 on H100). The step is
  kernel-latency-bound, not launch-bound.
- **Expert GEMVs, 4 rows a warp** (all staged before decoding): −4% vs 2.
- **K-split GEMV with 8 warps a row** instead of 4: −7%. Both together: −10%.
- **SiLU + int8 of h fused into the down kernel:** −2% vs a separate launch (each down block
  recomputes its pair's 512 values; the extra latency outweighs the saved launch).
- **The eager event profiler** (`TANG_CODER_PROF`) put about 2× the graph's time on small
  kernels. Use the stage sweep for decisions.
- **nsys / ncu on the RunPod image:** ncu's injection lacks `cuTensorMapEncodeIm2colWide`,
  which cudarc resolves at load, and nsys isn't installed (apt fails).
