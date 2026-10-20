# Where a decode step goes on an H200

Nsight Systems profile of `tang-llm serve` on one H200 SXM (RunPod secure, US-NC-1, driver
595.91.07, CUDA 12.8.1 devel image), tang `main` at 324a487, `mlx-community/Qwen3-8B-4bit`,
`--ctx 8192 --kv-slots 16`, `TANG_MAX_BATCH=16`. Each run: one warm-up request, then N
concurrent chat requests of 512 completion tokens (`temperature 0.7`, distinct prompts, ~24
prompt tokens). Trace: `nsys profile -t cuda,nvtx,osrt --sample=cpu`.

Summaries (per-step tables from the sqlite exports) are in `docs/h200-profile/*.summary.txt`.
The raw `.nsys-rep` files (16–73 MB each) are not committed; they're in `bench/nsys/` on the
Mac that ran this. The analysis window is the decode part of the final burst (first 15% cut to
drop prefill). A "step" is one logits download (one `cuMemcpyDtoH`).

| run | streams | spec | tok/s total (under nsys) | ms/step | kernels/step | GPU busy | idle/step |
|---|---|---|---|---|---|---|---|
| s1_nospec | 1 | off | 120 | 8.31 | 435 | 86.3% | 1.14 ms |
| s1_spec | 1 | on | 119 | 8.67 | 435 | 85.9% | 1.22 ms |
| s8_nospec | 8 | off | **480** | 13.39 | 375 | 72.7% | 3.65 ms |
| s8 | 8 | on | 350 | 14.89 | 553 | 74.4% | 3.82 ms |
| s16 | 16 | on | 323 | 12.75 | 336 | 64.2% | 4.56 ms |

nsys adds ~12% (120 vs the 136 tok/s measured without it), almost all on the CPU side; GPU
kernel times are as they'd be untraced.

## Headline

**The single-stream step is not overhead-bound. The GPU is busy 86% of it; the kernels are
slow.** The brief's guess (launch gaps and syncs) explains ~1.1 of the 8.3 ms. The other
7.2 ms is kernel time, and two-thirds of that is one kernel, the 4-bit GEMV, running at
~1 TB/s on a 4.8 TB/s part because its grids are smaller than the GPU.

## One decode step, 1 stream, speculation off (8.31 ms)

| | ms/step | share |
|---|---|---|
| `gemv_q4_xs4_m1` ×145 | 4.76 | 57% |
| `attn_partial` ×36 | 1.39 | 17% |
| `rms_norm` ×73 | 0.73 | 9% |
| everything else (prep, add, combine, swiglu, embed) ×181 | 0.29 | 3% |
| **GPU busy** | **7.17** | **86%** |
| launch gaps between kernels (434 × median 1.1 µs) | 0.60 | 7% |
| one big gap per step: sync on logits → CPU sample → next launch | 0.54 | 6% |
| **idle** | **1.14** | **14%** |

Host↔device traffic per step: one 608 KB DtoH (full 151,936-vocab f32 logits, 14 µs of
copy), two tiny HtoDs (token id, position), two `cuMemAllocAsync`/`FreeAsync`. No other syncs.

### The GEMV (4.76 ms, should be ~1.0)

Weights read per step: ~4.26 GB (36 layers × 108.5 MB plus a 350 MB lm_head, at 4.5 bits a
weight). At 4.8 TB/s that is 0.89 ms; at a realistic 80% of peak, ~1.1 ms.

| matmul | grid (blocks × 256 threads) | µs | effective BW |
|---|---|---|---|
| gate+up (24576×4096) | 768 | 44.7 | 1.27 TB/s |
| o_proj / down_proj (4096 out) | **128** | 30.9 avg | ~0.6 TB/s |
| qkv (6144×4096) | **192** | 19.1 | 0.74 TB/s |
| lm_head (151936×4096) | 4748 | 237 | 1.48 TB/s |

`gemv_q4_xs4_m*` launches `n / 32` blocks of 8 warps (4 columns a warp,
`crates/tang-compute/src/cuda/llm.rs:254`). For the 4096-wide outputs that is 128 blocks on
132 SMs: one block, 8 warps, per SM, and 32 KB of shared memory each. That's far too few
bytes in flight to cover HBM latency. Even the big launches top out at ~30% of peak, so the
inner loop also under-issues loads (one 4-bit word a lane at a time, judging by the 64-reg,
smem-staged shape).

### Attention (1.39 ms, should be ~0.1)

`attn_partial` runs on 32×2 = 64 blocks and takes 38–51 µs a layer at a context of ~600
tokens. That's ~2.4 MB of bf16 K/V a layer, so ~60 GB/s. It splits the sequence only 2 ways;
flash-decoding wants enough splits to fill 132 SMs several times over.

### rms_norm (0.73 ms, should be ~0.1 or 0)

One block (grid 1×1×1, 256 threads) for 4096 floats, 10 µs, 73 times a step. The kernel is a
latency chain on one SM.

### Host side

`cuMemGetInfo` is called once per kernel launch: 435 calls, 2.6 ms of CPU a step. It comes
from `CudaPool::maybe_evict_under_pressure` (`crates/tang-compute/src/pool.rs:119`), which runs
on every pool allocation once the pool holds over 64 MB. With `cuLaunchKernel` (1.7 ms), the
CPU spends ~4.3 ms a step issuing work. Today that hides under the 7.2 ms of GPU time.
**Once the kernels are fixed it becomes the bottleneck**, so it has to go in the same round.

## Speculation

- **1 stream:** no change (119 vs 120 tok/s). The essay prompts rarely draft. When it does,
  `gemv_q4_xs4_m2` costs 51.5 µs against m1's 32.8, so a verify of 2 rows costs 1.57× a
  decode. Acceptance has to be high to pay.
- **8 streams: speculation costs 27% throughput** (350 vs 480 tok/s). Verify rows push the
  batch past 8, onto `gemm_cols16_q4` (80 µs, versus 62.7 for `gemv_q4_xs4_m8`) and
  `gemm_cols32_q4` (153 µs). It also adds ~170 DtoD copies a step (0.7 ms of API time) and
  ~180 extra launches. The zest bench ran with speculation on, which is part of why it
  flattened at 8.

## 8 and 16 streams

- **8, spec off (13.4 ms/step for 8 rows):** `gemv_q4_xs4_m8` is 75% of busy at 62.7 µs a
  launch, 1.9× m1. For a memory-bound kernel, 8 rows should cost about the same as 1, so the
  m8 GEMV is compute- or issue-bound as written. Idle rises to 3.65 ms a step, almost all in
  the once-a-step gap (2.9 ms). The 4 MB logits download is only 0.21 ms of that. The rest is
  host work between the sync and the next launch, which scales with rows: about 0.3 ms a row,
  consistent with CPU sampling over 152k logits a row plus scheduling. (Inferred from the gap
  scaling; the CPU samples weren't broken down further.)
- **16, spec on:** the batch lands on `gemm_cols32_q4` (153 µs a launch) and `dequant_q4` +
  cuBLAS TF32 GEMM (1.06 ms + 0.16 ms a step). Idle 4.6 ms a step, 1 ms of it the DtoH.
  Throughput (323) is below 8 streams with spec off (480).

## Ranked fixes

Gains are estimated from the trace for the 1-stream step (8.3 ms traced, ~7.35 ms untraced),
applied in order.

| # | fix | evidence | expected step | 1-stream tok/s |
|---|---|---|---|---|
| 0 | baseline | | 7.35 ms | 136 |
| 1 | **Fill the GPU with the 4-bit GEMV.** Split-K or 1–2 columns a warp, so the 4096-wide outputs launch ≥4×132 blocks; vectorised 128-bit weight loads; no 32 KB smem stage for x (read it through L1). Target ≥3 TB/s. | 4.76 of 7.17 ms busy; 0.6–1.5 TB/s achieved | −3.4 ms → ~3.9 ms | ~255 |
| 2 | **Delete the per-launch `cuMemGetInfo`.** Check pressure every N allocations or on a pool miss only. | 435 calls, 2.6 ms CPU a step | 0 GPU now. Required for #1 to show: CPU issue time drops from 4.3 to ~1.7 ms, under the new GPU time | — |
| 3 | **Split attention decode across the sequence** (flash-decoding: splits = f(ctx, SMs), then the existing `attn_combine`). | 1.39 ms at ~60 GB/s on 64 blocks | −1.2 ms → ~2.7 ms | ~370 |
| 4 | **Fuse rms_norm into its neighbours** (residual add + norm in one kernel, or fold the norm scale into the next GEMV's x load); at minimum, a multi-block norm. | 73 × 10 µs single-block launches | −0.6 ms, −145 launches → ~2.0 ms | ~490 |
| 5 | **Sample on the GPU.** Temperature/top-p/argmax kernel, download one u32 a row instead of 608 KB of logits. | 0.54 ms gap at 1 row, ~2.9 ms at 8 (≈0.3 ms a row) | −0.4 ms at 1 stream, −2.5 ms a step at 8 | ~560 (1), large at 8 |
| 6 | **CUDA graphs for the decode step** (per batch size, KV length as a device scalar). | ~290 launches left after #4 × ~1.1 µs gaps + ~4 µs CPU each | −0.3 ms GPU gaps; makes the CPU free to run the next step's scheduling | ~620 |
| 7 | **Speculation off above ~4 streams** (or price verify rows with the real m-row costs in `step_cost.rs`). Then rework the ≥9-row path: a tensor-core (`mma.sync` / wgmma on sm_90) 4-bit GEMM to replace `gemm_cols16/32_q4` and dequant+cuBLAS. | 8 streams: 480 off vs 350 on; cols32 at 153 µs | 8 streams +37% today, for free; the tensor-core GEMM is what lifts the 16–64-stream plateau | — |
| 8 | **Fix the m>1 GEMV scaling.** m8 should cost ≈m1 when memory-bound; today 1.9×. Same rewrite as #1, with the rows sharing each weight load. | m8 62.7 µs vs m1 32.8 | 8 streams: GEMV 7.3 → ~1.5 ms a step | 8 streams ≈ 3–4× today's 480 with #5 |
| 9 | Persistent megakernel (`docs/fastest.md`) | after 1–8, the residual is ~0.3 ms of gaps + tail effects | last ~10–15% | toward the ~1000 bound |

Items 1–5 are ordinary kernel work and get the solo stream from 136 to ~500 tok/s (≈50% of
the ~1000 bound, where the 3090 already is). Items 5, 7 and 8 matter most for zest's $/M,
because they set the 8–16-stream throughput. Graphs (6) and the megakernel (9) come after:
on this trace, launch overhead is ~7% of the step, not the bulk of it.

## Reproducing

```bash
TANG_SPECULATE=0 TANG_MAX_BATCH=16 nsys profile -t cuda,nvtx,osrt --sample=cpu \
  -o s1_nospec --wait=primary \
  target/release/tang-llm serve <Qwen3-8B-4bit dir> --device cuda --ctx 8192 --kv-slots 16
# drive N concurrent /v1/chat/completions with max_tokens 512, then SIGINT the server
nsys export -t sqlite -o s1_nospec.sqlite s1_nospec.nsys-rep
```

The driver and analysis scripts are `docs/h200-profile/prof.sh` and `an.py` (`python3 an.py run.sqlite`).
