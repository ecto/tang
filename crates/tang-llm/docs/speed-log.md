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

The order to try on the H200: (1) the smem GEMV A/B in the script; (2) CUDA graphs over the decode
step (launch-bound on Hopper); (3) a Q4 head and Q4 attention, which takes the ceiling from 1280
to ~2460; (4) speculation, already exact for gpt-oss (`tests/speculate.rs`).

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
| 2026-10-21 | 4ac3171 | K-split GEMV, 2 warps a row (was 4) | H200 | explain | none | 227.2 | 16.2% | +4.4% |
| 2026-10-21 | 4ac3171 | (same pod, 2 warps for 4k+-row matrices only) | H200 | explain | none | 226.9 | 16.2% | +4.3% |
| 2026-10-21 | 4ac3171 | (same build, AA) | H200 | refactor | suffix | 370.2 | | 2.71 tok/window |
| 2026-10-21 | 4ac3171 | | H200 | write | suffix | 229.0 | | 1.25 tok/window |
| 2026-10-21 | 4ac3171 | | H200 | explain | suffix | 222.3 | | 1.16 tok/window |
| 2026-10-21 | 2a723ae | (round-3 pod, defaults of 4ac3171) | H200 | explain | none | 222.3 | 15.9% | pod baseline |
| 2026-10-21 | 2a723ae | fast Q4_K/Q5_K expert dot (32 weights a lane) | H200 | explain | none | 230.9 | 16.5% | +3.9% |
| 2026-10-21 | 2a723ae | Q8X dense repack (aligned codes, separate scales) | H200 | explain | none | 231.8 | 16.6% | +4.3% |
| 2026-10-21 | 8fdfd13 | Q8X + fast expert dot (the defaults now) | H200 | explain | none | 240.8 | 17.2% | +8.3% |
| 2026-10-21 | 2a723ae | + bf16 router (opt-in) | H200 | explain | none | 243.1 | 17.4% | +1% |
| 2026-10-21 | 2a723ae | (Q8X + bf16 router, AA) | H200 | refactor | suffix | 403.5 | | 2.71 tok/window |
| 2026-10-21 | 2a723ae | | H200 | write | suffix | 252.0 | | 1.25 tok/window |

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
- **Top-k in the router GEMV's last block** (atomic counter, one launch fewer): −2% (217 → 213).
  The last block's serial top-k costs more than the launch it saves.
- **K-split routed experts** (Q warps a row, unstaged, like `cx_gemvk`): gate|up −4% / −6% /
  −16% at Q = 1 / 2 / 4, down +0.5% / −6% / −18%. Experts want the staged rows; the dense
  matrices want the split.
- **K-split GEMV, 1 warp a row** (no split): −1.5% vs 4, −6% vs 2.
- **A Qwen3 drafter** (estimate, not built): teacher-forced top-1 agreement with the engine's
  greedy output is 58 / 88 / 71% for Qwen3-0.6B and 62 / 90 / 77% for 1.7B
  (explain / refactor / write). A verify window of 8 costs ~3× a T=1 step here (each token
  routes to its own experts), so a drafter only pays on refactor-like text, which the free
  suffix drafter already covers (2.7 tokens a window). Expected gain on fresh text ≤10%, for a
  second model engine. Not worth building yet.
- **First rows of a `flash-aa` run** read ~4% low (205 vs 213–217 for the same kernels later in
  the run). Compare variants inside the A/B block, which warms up first.
- **bf16 router weights** (`TANG_CODER_ROUTER16=1`): +1% (240.8 → 243.1), under the bar. The
  router is 4.2 MB a layer of F32 but reads at ~1.1 TB/s either way. Kept opt-in.
- **nsys / ncu on the RunPod image:** ncu's injection lacks `cuTensorMapEncodeIm2colWide`,
  which cudarc resolves at load, and nsys isn't installed (apt fails).
