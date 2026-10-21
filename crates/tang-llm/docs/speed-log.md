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
| 2026-10-21 | 701af6c | #87 speculation ≤2 streams, #89 tensor-core GEMM (no change at 1/8 rows in bench_batch, as expected) | H200 | 8B | 1 | 207 | 18.5% | −0.5% |
| 2026-10-21 | 701af6c | (same) | H200 | 8B | 8 | 711 | | +0.3% |
| 2026-10-21 | 701af6c | (same) | H200 | 32B | 1 | 70.7 | 27.2% | +1.7% |
| 2026-10-21 | 701af6c | (same) | H200 | 32B | 8 | 219 | | +1.9% |

AA-style single stream (`tang-llm serve`, speculation on as deployed, ~850-token prompt, 500
output tokens, temperature 0.7, streamed; output tok/s from first token to last, median of 3;
the probe is `docs/bench/aa.py`):

| date | commit | GPU | model | AA output tok/s | bar (fastest public API) |
|---|---|---|---|---|---|
| 2026-10-21 | 701af6c | H200 | 8B | 200.5 | Eigen 303 |

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

## gpt-oss-20b

Single stream, AA-style (a ~1k-token chat prompt, output tok/s; `scripts/gpt_oss_speed.sh`).
The bar is Groq's claimed ~1,200 tok/s.

**Bytes per decoded token** (what a step must read; gpt-oss-20b, 24 layers):

| part | format | per token |
|---|---|---:|
| experts: 4 of 32 per layer, gate/up 5760×2880 + down 2880×2880 | MXFP4 (4.25 bits) | 1.27 GB |
| attention Q/K/V/O, 26.5M a layer | bf16 | 1.27 GB |
| output head 201088×2880 | bf16 | 1.16 GB |
| KV at ~1.3k context (12 full layers ~1.3k keys, 12 sliding 128), f32 | | ~0.05 GB |
| router, norms, biases | f32 | ~0.01 GB |
| **total** | | **~3.75 GB** |

Speed of light at that size: 1280 tok/s on an H200 (4.8 TB/s), 890 on an H100, 270 on a 4090.
Groq's 1,200 is ~94% of the H200's. **To pass it, the weights have to shrink:** attention and
the head at 4 bits (Q4, as the Qwen rows run) would take a token to ~1.95 GB, a 2460 tok/s ceiling
on the H200. The experts are already 4-bit.

**Per kernel** (4090 nsys, 256 decoded tokens, before the 4-rows-a-warp GEMV): the expert GEMV
was 48% of GPU time at ~350 GB/s (a third of peak) and the bf16 GEMVs 36% at ~0.94 TB/s (at
peak). Sink attention was 6%, rms_norm 4%. Host side: ~460 launches a token, with a
host-to-device upload (ids, block tables) and a logits download every step.

| kernel | bytes/token | at 4.8 TB/s | measured | what's left |
|---|---:|---:|---|---|
| `moe_gemv_mxfp4` (48 calls) | 1.27 GB | 0.26 ms | 4090: 3.6 ms → ~2.2 ms after 4 rows a warp | input staged in smem (new, A/B below); more columns a block |
| `gemv_bf16` attention (48 calls) | 1.27 GB | 0.26 ms | at bandwidth | 4-bit attention weights |
| `gemv_bf16` head (1 call) | 1.16 GB | 0.24 ms | at bandwidth | 4-bit head, or device top-k so only the argmax comes back |
| `attn_sink_partial` + combine | ~0.05 GB | 0.01 ms | 6% (launch-bound) | fold into fewer launches |
| everything else (~360 launches) | | | ~1.5 ms on H100 | CUDA graphs |

Measured so far (before the rebase onto #88/#89):

| date | commit | change | GPU | model | streams | tok/s | % SoL | vs prev |
|---|---|---|---|---|---|---|---|---|
| 2026-10-20 | 29221a7 | #95, 13-token prompt | RTX 4090 | gpt-oss-20b | 1 | 157 | 58% | |
| 2026-10-20 | 1b8474c | #95, 13-token prompt | H100 | gpt-oss-20b | 1 | 182 | 20% | |
| 2026-10-20 | 1b8474c | #95, 13-token prompt | H100 | gpt-oss-120b | 1 | 119 | | |

**Plan for when gpt-oss is back in focus** (on hold since 2026-10-21; 30% of zest goes to
speed, all of it on Coder-Next and tang-pro for now). In order, each with what's ready:

1. **Q4 attention and head**, the path past Groq: 3.75 → ~1.95 GB a token, H200 ceiling
   1280 → ~2460 tok/s. `--q4` already quantizes attention, the embedding and the head at load
   (affine 4-bit, group 64) and the 4-bit GEMV serves them. Correctness checked offline:
   `scripts/gpt_oss_q4_check.py` applies the same rounding to transformers' weights, and tang
   `--q4` matches it at max KL 1.1e-4 (tiny model, CPU). Still unmeasured: what Q4 costs the
   real 20b. `gpt_oss_speed.sh` measures it (`--q4` vs bf16 KL, 256 positions) beside the speed.
   If the head costs too much, keep the head bf16 and quantize attention only (1.27 GB saved of
   the 1.8), or try group 32.
2. **The MXFP4 GEMV A/B** (input via L1, the default, vs staged in shared memory,
   `TANG_MOE_GEMV=smem`), in the same script. The smem kernel has not run on a GPU yet.
3. **CUDA graphs over the decode step**: the H100 step is launch-bound (~460 launches a token).
4. **Speculation**: already exact for gpt-oss (`tests/speculate.rs` passes on CUDA).

One H200 hour covers 1 and 2 (`scripts/gpt_oss_speed.sh <gpt-oss-20b dir>`; it prints these
rows and an nsys summary).

## Dead ends

Tried and measured at under 3% (or worse):

- **4-bit GEMV, a warp a column, x read through L1 (no shared-memory stage):** 30% slower at one
  row, worse with more rows (L1 traffic is 8x the weight bytes per extra row).
- **GEMV with the next stage's weights double-buffered in registers:** 5-15% slower; register
  pressure cut occupancy.
- **GEMV, 2 columns a warp instead of 4:** 2-10% slower.
- **GEMV K-split sizing, 1 / 4 / 8 / 16 blocks an SM:** within 3% of each other once the
  128-bit loads are in; 4 kept.
- **GEMV with the whole K slice of x resident in shared memory and every weight load issued up
  front (`gemv_q4_xr`, 2026-10-21, H200):** 8B solo 207 → 130 tok/s, 32B 71 → 36. Registers
  (16 uint4 of weights a lane) leave one 8-warp block an SM, and a block loads everything,
  then computes: no overlap between one block's math and the next one's loads. The fix has to
  keep several blocks (or a load pipeline) per SM.
- **Device top-k for a lone stream:** 1.5% slower than sampling its one row on the host;
  only batches cut on the device.
- **The VRAM-pressure check on pool misses only (#81)** measured +1.5% alone. It's kept because
  it takes ~2.6 ms of CPU off each step, which matters once kernels stop hiding host time.
