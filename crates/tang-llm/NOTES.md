# tang-llm notes

Checkpoint log for GPU-path findings. Newest first. Each entry: what was wrong, the
evidence that pinned it, the fix, and how it was verified. Only things read back from
logs, not inferred.

## 2026-10-28 — MoE round 3 on B300: roofline microbench; shared-expert tail fix; +1.0 tok/s, ids unchanged (branch claude/glm5-moe3)

Roofline (new `bench/glm5-moe-roofline/roofline.cu`, nvcc sm_100, one routed layer's gate|up at T=1:
8 experts x 2 x 2048 rows x 2304 B Q4_K = 75.5 MB, 16 disjoint expert sets so every rep is L2-cold):
- pure streaming of those bytes: grid-stride flat read 4.6–5.3 TB/s; warp-per-row-group with all
  16-byte loads issued up front 5.8–5.9 TB/s (12.7–13.1 us) for EVERY shape tried (1/2/4 rows a warp,
  1/2/4 warps a row, 4/8 warps a block, min-blocks 4/8/16). Shape / bytes in flight is not the limiter.
- g5f_gun's own access pattern without the math (`gunlite`): weight loads only 14.4 us (5.2 TB/s);
  + the per-chunk activation loads (xq, xq2, xd: 2 KB per warp per iteration) 18.3 us, warm-L2 9.9 -> 17.7 us.
- production g5f_gun, routed slots only: 24.7 us cold / 23.2 warm (3.0 TB/s) -> L2-warm is no faster, so
  it is issue/LSU bound, not DRAM: ~4 us activation re-reads + ~6 us decode math over the 14.4 us floor.
  No residual codes (q8 only): 22.5 us. UNROLL 2/4 (any MINB) slower (26.9–37.7), MINB 16 spills (75 us).
- + the shared expert slot (Q8_0, 8-weight fma chain of g5f_lane_dot): 35.7 us, i.e. +11 us for 17.8 MB,
  because it is the last grid row and runs as a tail.
- Splitting a row's chunk loop over warps (or keeping activations in registers across rows, `g5f_gux`
  v1: routed 23.0 us) is NOT bit-identical: nvcc contracts `acc += dx * (...)` into an fma, so the
  per-lane chunk sum cannot be re-associated (2616/18432 h values differed). Dropped.

Kept:
1. `g5f_gux` (TANG_GLM5_MOE_GUX=4 rows a block, 0 = g5f_gun): g5f_gun's per-row math with the shared
   slot as block row 0 (scheduled first) and its gate / up rows on separate warps. Microbench gate|up
   + shared 35.7 -> 31.0 us, h bit-identical to g5f_gun (0/18432 differ). Same slot remap in g5f_down_splitn.
2. Top-k select with the choice values in registers and the id/weight stores done once by lane j
   (TANG_GLM5_MOE_SEL2=0: old). Same float ops.
3. The K-quant min term (sum of activation codes) once per chunk for all rows (same ints/float ops).
4. g5f_lane_dot U template: 0 = compiler's unrolling (as before). An intermediate `#pragma unroll 1`
   slowed g5f_gemv 2.19 -> 2.77 ms/token in nsys — fixed before the numbers below. LDU 0/4/8 on the
   Q8_0 shared slot: no measurable difference (31.0 / 30.8 / 30.4 us).

nsys (B flags, -n 32, ms/token; base = /root/tang-moe2 binary = moe2 kernels, same pod):
g5f_gun 2.384 -> g5f_gux 2.028 | down_splitn 1.428 -> 1.482 | route 0.561 -> 0.518 | down (dense) 0.155 -> 0.151 |
q8 0.118 -> 0.119 | sum 0.069 -> 0.068 | MoE total 4.72 -> 4.38. g5f_gemv 2.188 -> 2.184 (unchanged).
Bench (scripts/glm5-bench.sh, e5083544, haiku gate OK): -n 128 59.60 tok/s md5 7174cdd6 (was 58.64);
GLM5_N=512 60.22 tok/s md5 1453cf0e (was 59.64). mew (3090): flash::glm5 83 passed / 0 failed.
Target (MoE <= 3 ms/token) NOT reached. route is 12.3 us/call after the select rewrite, so the select was
not its cost (unattributed). Next for gate|up: cut the activation re-reads (they cost as much L1 traffic
as the weights) without re-associating the per-lane sum, e.g. 2 rows a warp sharing one x load
with x in registers only for the chunk being processed (GUR=2 was slower in round 2 — registers),
or stage x in shared memory per block.

## 2026-10-28 — MoE fast path on B300: word-wide headers, hoisted activations, down-row L2 prefetch; +0.4–1.2 tok/s, ids unchanged (branch claude/glm5-moe2)

What was wrong: per layer (T=1, `real_layers_fast_matches_cpu`, `TANG_GLM5_MOE_SPLIT=1`, traced µs) g5f_gu
44–48 (2.1–2.3 TB/s), g5f_down_split 32 (Q5_K) / 52 (Q6_K, 1.2 TB/s), g5f_route 27, q8 7, sum 7; total
118 (l4) / 142 (l11). ncu is unusable on this pod (ERR_NVGPUCTRPERM), so attribution is by A/B only.

What did not help (B300, traced µs gu / down on l4, l11):
- UNROLL=4: gu 44 -> 50/55. UNROLL=4 + MINB=4: did not run (binary rebuilt under the queued run).
- More rows a warp (new g5f_gun / g5f_down_splitn): GUR=2 DNR=4 gu 50/54; GUR=4 DNR=8 gu 61/65, down 38/55;
  GUR=2 DNR=4 MINB=4 gu 54/63, down 39/86. Fewer, fatter warps are consistently slower.
- L2 prefetch of the gate|up rows (`prefetch.global.L2` over the row's lines): gu unchanged/slightly worse.
So the gu kernel is not bound by per-warp load latency or row-level MLP; the true limiter is still open.

Kept (cac0e91c, defaults GUR=1 DNR=2 PF=2):
1. `g5f_c32v`: Q4_K/Q5_K block header (d, dmin, 12 scale bytes) as one 16-byte load instead of ~7 byte
   loads; Q8_0r/Q6_Kr d and the scale pair as 16-bit loads; activations loaded once per chunk for all rows.
   Same values, same float ops and per-row sum order -> h / y bit-identical. l4 119 -> 115 us/call,
   l11 143 -> 139 (untraced, device x/y).
2. `TANG_GLM5_MOE_PF` mask (1 gate|up, 2 down; default 2): L2 prefetch of a warp's down row pair. l11
   (Q6_K down) down 52 -> 44 traced, call 139 -> 134.
`TANG_GLM5_MOE_GUR=0 DNR=0 PF=0` restores the previous kernels.

Bench (scripts/glm5-bench.sh, B flags, bin 6e93f80f, haiku gate OK): -n 128 58.64 tok/s ids md5 7174cdd6
(baseline 58.28, same md5); GLM5_N=512 59.64 tok/s md5 1453cf0e (baseline 58.48, same md5).
Target (MoE <= 2.5 ms/token) NOT reached: this is ~5% of the MoE time. Next: g5f_route (27 us traced,
serial top-8 in one warp) and a different gu mapping (lane-contiguous 16 B reads); needs counters.

## 2026-10-28 — KDA resident upload 462 s -> 17 s: NVRTC ran 3x per layer (branch claude/glm5-kda-load)

What was wrong: `kda_dev::DevLayer::new_gguf` builds three CUDA modules per KDA layer
(`kda_cuda::cuda::load`, `Kernels5::new`, `kda_decode::Fast::new`), and `tang_moe::gpu::Gpu::module`
NVRTC-compiled the source every time — on B300 twice per module, since the CUDA 12.8 NVRTC
rejects sm_103 and retries at sm_90. 34 layers x 3 compiles of identical sources = the ~13.5 s/layer.
The weight uploads themselves were never the cost.

Fix: `Gpu::module` memoises the PTX text per (arch, source) in a process-global map; every
later load is just `cuModuleLoadData` of the same bytes. No kernel or numeric change.

Evidence/verification (B300, B flag set, `scripts/glm5-bench.sh`, GLM5_ROOT=/root/tang-kda-load):
before "8 KDA layers resident (104 s) … 34 … (462 s)", open 504 s; after "8 KDA layers resident
(13 s)", "34 KDA layers resident on the GPU(s) (17 s)", "open 50.3 s". Gate ids OK, decode
58.25 tok/s (baseline 58.28), ids md5 7174cdd6 (unchanged). mew (3090): flash::glm5 83 passed / 0 failed.
## 2026-10-28 (gputime) — first per-kernel GPU profile of the resident token on the B300: mHC finish 3.3 ms, MoE 7.4 ms of 16.5 (branch frog/glm5-forward)

The ncu shipped in the runpod image (Nsight Compute 2025.1.1) cannot profile the B300 (compute
10.3): `==WARNING== No kernels were profiled` after a 400 s run. Installed nsight-systems-2026.3.2
and nsight-compute-2026.3.1 from the CUDA apt repo; `nsys profile -t cuda --cuda-graph-trace=node`
sees every graph node. Binary 83b3ab92, B flags, segment graph, haiku prompt -n 32 (8 prompt
tokens, so per-token counts are ≈ instances/32 and the max values are prefill instances).
Full csv: `bench/glm5-prof/b300-83b3ab92-n32-kern_sum.csv`. Total 526.9 ms GPU = **16.47 ms/token**,
consistent with 58 tok/s being all kernel time.

| kernel | inst/32 tok | avg µs | med µs | ~ms/token |
|---|---|---|---|---|
| mhcf_finish | 2880 | 36.9 | 36.9 | 3.32 |
| g5f_gu | 1440 | 58.2 | 40.3 | 2.62 |
| g5f_gemv | 2316 | 30.4 | 10.8 | 2.20 |
| g5f_down_split | 1344 | 39.8 | 25.5 | 1.67 |
| k5_qgemv9 | 1054 | 30.5 | 30.5 | 1.01 |
| k5d_tail | 1054 | 26.5 | 26.4 | 0.87 |
| mla_fold_q2 | 352 | 74.3 | 73.1 | 0.82 |
| mhcf_part | 2912 | 8.2 | 8.3 | 0.75 |
| mla_attn_abs2 | 352 | 57.1 | 56.4 | 0.63 |
| g5f_route | 1344 | 13.3 | 12.9 | 0.56 |
| k5_qgemv17 | 1054 | 16.2 | 16.3 | 0.53 |
| k5_gemv8 | 51 | 254.6 | 254.5 | 0.41 |
| k5_gemv | 68 | 94.4 | 93.6 | 0.20 |
| g5f_down | 96 | 51.4 | 42.3 | 0.15 |
| g5f_q8 | 2592 | 1.6 | 1.5 | 0.13 |
| kda_pf_state | 34 | 92.7 | 92.6 | 0.10 |
| k5_gated_norm | 34 | 91.2 | 91.2 | 0.10 |

Reading: `mhcf_finish` (one block, 20 serial Sinkhorn iterations) is the single largest kernel
at 37 µs × 90/token; `mhcf_part` + finish = 4.1 ms. The MoE `g5f_*` family is ~7.4 ms/token
against a ~0.45 ms bytes floor (top-8 × 45 layers ≈ 3–4 GB at 8 TB/s) — underfilled grids
for t = 1. KDA (`k5_qgemv9` + `k5d_tail` + `k5_qgemv17`) ≈ 2.5 ms, DSA (`mla_fold_q2` +
`mla_attn_abs2`) ≈ 1.5 ms. Also from this run's log: model open is 504 s, of which the 34 KDA
layers' resident upload is 462 s (13.5 s/layer) while all 45 FFNs upload in 30 s — an
iteration-speed problem, not a decode one.

Work farmed out from this profile: claude/glm5-mhc2 (finish kernel), claude/glm5-moe2 (g5f
grids), claude/glm5-kda-load (open time).

## 2026-10-28 (integration round 4) — merged on B300: whole-step graph default, 58.5 tok/s; MTP host draft is 6 s/pass (branch frog/glm5-forward)

Pod: fresh B300 SXM6 (US-WA-2, 275 GB, CUDA 12.8 image, container disk). `scripts/glm5-bench.sh`
on claude/glm5-integrate4 83b3ab92 with the B flag set (MOE=fast GEMV8 KDA_PF2 MHC_FUSED
MLA_RESIDENT GPU_HEAD KDA_G5F=2 KDA_FUSED KDA_GRAPH), then frog/glm5-forward 75249bda as the
reference. ids md5 = md5 of the `ids:` line's token list.

| run | STEP_GRAPH | -n | decode tok/s | ids md5 |
|---|---|---|---|---|
| integrate4 | 1 (segments) | 128 | 55.92 | 7174cdd6 |
| integrate4 | 2 (whole) | 128 | 58.28 | 7174cdd6 |
| integrate4 | 1 | 512 | 53.94 | 1453cf0e |
| integrate4 | 2 | 512 | **58.48** | 1453cf0e |
| forward 75249bda | 1 | 512 | 54.29 | 1453cf0e |

Haiku oracle on every run. The whole-step graph is bit-identical to the segments and to
forward at 512 (across the dense→sparse indexer switch), and 2.4–4.5 tok/s faster. It is the
default now (d68e2b65: unset / `2` = whole, `1` = segments, `0` = eager; the resident-step
test runs all three). mew tiny-model suite 83/0/2 on f4005f33.

MTP on the same pod (`--mtp K`, WIN_STEP default K+1), bench prompt -n 128:

| K | accept | tok/pass | draft s | trunk s | ids md5 |
|---|---|---|---|---|---|
| 1 | 61/67 = 91% | 1.74 | 445.7 | 7.2 | baced18e |
| 2 | 33/44 = 75% | 1.92 | 535.2 | 6.7 | 7174cdd6 |
| 3 | 18/29 = 62% | 1.87 | 615.2 | 7.3 | 7174cdd6 |

Haiku gates (-n 8) match the oracle at every depth. The whole run is 470–640 s for 128 tokens:
the *draft* (the MTP layer blk.45 forward) costs ~6.6 s per pass on the host path while the
trunk verify is ~0.1 s/pass. MTP is unusable until the draft layer runs on the device inside the
resident step; the ×1.8 tok/pass is real but buried under a 60× draft cost. Also noted, not
chased: K=1's 128-token ids differ from greedy (baced18e vs 7174cdd6) while K=2/3 match — a
depth-1 acceptance edge to bisect before MTP goes on the device.

Timings in the bench summary's stage columns are meaningless on the resident path (the trace
span wraps the one sync per token); the ncu graph-node profile is the per-kernel source.

## 2026-10-28 (integration round 4 prep) — mtp-ready + stepgraph2 + kda-q4 merged; one test env lock; suite green in parallel (branch claude/glm5-integrate4)

No real weights this round (no pods). Base frog/glm5-forward 75249bda, then `--no-ff` merges in order:
1. claude/glm5-mtp-ready 6383697b. Textual conflict: this file only (both kept). engine.rs / kda_dev.rs
   auto-merged; builds as is.
2. claude/glm5-stepgraph2 b236d771. Conflicts: this file; `resident.rs` `step_resident`'s mode parse
   (forward made segments the default with `=0` eager; stepgraph2 had `=1` opt-in and added `=2`).
   Now: unset / `1` = segments (default), `2` = whole step, `0` = eager.
3. claude/glm5-kda-q4 32d84547. Conflict: this file. The Q4_K `k5_qgemv_q4k5/9` tables are picked by
   tensor type at load (every KDA matrix Q4_K), so the default Q8_0 GGUF path is unchanged.

Env lock: there were two Mutexes (stepgraph2's `glm5::ENV_LOCK`, mtp-ready's engine-tests one) and
only the setters held them, so readers still raced. Now `glm5::ENV_LOCK` is an `RwLock`; every test in
engine.rs, kda_dev.rs, kda_decode.rs, mla.rs, mhc_fused.rs, moe_tests.rs, mtp_tests.rs holds exactly one
guard for its body: `env_write()` if it sets / removes `TANG_GLM5_*` or toggles the trace
(real_weights_gpu_matches_cpu, slots_batch_matches_single, win_step_window_matches_steps,
resident_step_matches_host, trace_is_transparent, glm5_kda_layer_bench_real, kda_g5f_real_bench,
kda_quant_real_bench), else `env_read()`. Not re-entrant: `gpu_slots_case` no longer locks.

MTP vs resident (decision): MTP stays on the host `window()` path entirely: prefill, the t>1 verify
window, and the rerun of the accepted prefix (including t=1 reruns) all call `window()`, which drops the
step graph and refreshes stale DSA mirrors. The resident step can't serve MTP anyway: it doesn't produce
`last_h`, which the drafter needs. `restore` clears `kda_dev_fresh` (so `resident_ready` is false until a
window re-seeds), now also calls `drop_step_graph()` (segments, whole graphs, `[tok,pos]`, cap) and
clamps `DsaCache::stale_from` to the rewind point. A plain decode after an MTP rewind takes one eager
resident token, then recaptures.

Tests:
- local CPU `flash::glm5`: 42 passed.
- mew 3090 CUDA `flash::glm5` (TANG_REQUIRE_CUDA=1), PARALLEL: 83 passed / 0 failed / 2 ignored (the two
  `#[ignore]` real-weight benches, not flakes). Same with `TANG_GLM5_STEP_GRAPH=2`: 83/0/2, and `=0`: 83/0/2.
  Caveat: tests that set STEP_GRAPH themselves remove it after, so later tests in those runs see the default.
- mew tiny f32, prompt `3 17 42 7 99 5`, `-n 40`, flags MOE=fast KDA_PF2 KDA_GRAPH GEMV8 MHC_FUSED
  MLA_RESIDENT KDA_FUSED: plain greedy ids md5 bc55ff4b under graph default (2 segments captured),
  `=2` (dense + sparse whole graphs captured) and `=0`. `--mtp 1/2/3` ids = plain under all three.
  The tiny drafter accepts 0% (random weights), so every pass rewinds + reruns: restore is exercised
  39 times per run; accept rate / tok/s mean nothing here.

Still needs a B300:
- graph=2 real-weights gate: haiku `-n 8` ids = oracle, `-n 512` md5 vs graph=1, tok/s and launches.
- `--mtp 1/2/3`: ids vs plain, tok/s, accept rate per depth.
- WIN_STEP cost at depth 3 (default K+1 vs `TANG_GLM5_KDA_WIN_STEP=0`).
- kda-q4: Q4_K-requantized KDA GGUF vs Q8_0, tok/s A/B and the ids/perplexity gate.

## 2026-10-28 (integration round 3) — resident step merged; the step graph is now the default: B300 56.7 tok/s (branch frog/glm5-forward)

Fast-forwarded frog/glm5-forward 9c49294d -> claude/glm5-resident 5c024107 (clean ff). Then:
- 4825819d `glm5-bench.sh`: `GLM5_ROOT` (a private remote dir, default /root/tang-bench); the
  launcher also holds /root/gpu.lock around the remote half.
- 339ae535 `TANG_GLM5_STEP_GRAPH` default ON; `=0` runs the resident step eagerly.
  `engine_gpu_resident_step_matches_host` now sets `=0` for its eager leg.

Tests:
- local CPU `flash::glm5`: 40 passed.
- mew CUDA `flash::glm5`, parallel: 76 passed / 3 failed / 2 ignored, both with and without
  `STEP_GRAPH=1`. The 3 (`engine_gpu_trace_is_transparent`, `engine_gpu_slots_batch_matches_single`,
  `engine_gpu_resident_step_matches_host`) each pass alone. They race on process-global env
  (`set_var` of `TANG_GLM5_STEP_GRAPH` / `TANG_GLM5_MOE` / `TANG_GLM5_RESIDENT` in sibling tests):
  e.g. "step graph segments left 0 right 2", "naive: MoE kernel path left true right false".
- mew CUDA `flash::glm5 -- --test-threads=1`: 79 passed / 0 failed / 2 ignored, before and after
  the default flip.

B300, tree 4825819d, `scripts/glm5-bench.sh` (`GLM5_ROOT=/root/tang-integrate3`), flags B
(`MOE=fast GEMV8=1 KDA_PF2=1 MHC_FUSED=1 MLA_RESIDENT=1 GPU_HEAD=1 KDA_G5F=2 KDA_FUSED=1
KDA_GRAPH=1`), bench prompt (238 tok), `-n 128 --temp 0`, `TANG_GLM5_TRACE=1`:

| run | decode tok/s | wall/token | launches | H2D# | D2H# | syncs |
|---|---|---|---|---|---|---|
| (a) B (resident, eager) | 47.39 | 19.1 ms | 563 | 1 | 1 | 1 |
| (b) B + `STEP_GRAPH=1` | **56.72** | 17.1 ms | 164 | 1 | 1 | 1 |
| (c) B + `RESIDENT=0` | 35.05 | 26.4 ms | 561 | 182 | 226 | 216 |
| (d) B + `STEP_GRAPH=1` (repeat) | 56.67 | 17.1 ms | 164 | 1 | 1 | 1 |
| (e) B on 339ae535 (graph default) | 56.66 | 17.1 ms | 164 | 1 | 1 | 1 |

Stages, wall ms/token (top-level trace rows; mhc_post = both `mhc_post.*` rows):

| run | kda | dsa | dsa/o_proj | mhc_pre.attn | mhc_pre.ffn | mhc_post | ffn | head |
|---|---|---|---|---|---|---|---|---|
| (a) | 0.12 | 0.69 | 0.09 | 0.40 | 0.40 | 0.00 | 0.94 | 16.14 |
| (b) | 0.03 | 0.61 | 0.08 | 0.00 | 0.00 | 0.00 | 0.01 | 15.91 |
| (c) | 4.15 | 6.82 | 5.22 | 3.84 | 3.88 | 0.58 | 6.53 | 0.51 |
| (d) | 0.03 | 1.01 | 0.14 | 0.01 | 0.01 | 0.00 | 0.01 | 15.49 |

On the resident paths the stage rows are host enqueue time; the GPU time lands in `head`'s
single sync (16.1 ms in (a), 15.9 in (b)). Only (c) is a per-stage cost breakdown. The trace's GPU-ms
column is not usable here (as in the resident entry).

Gates:
- haiku `-n 8 --temp 0`: `785 1196 6801 264 6386 38038 911 279` for a, b, c, d, e.
- `-n 128` id streams: md5 04e96df4 for a, b, c, d, e.
- `-n 512` on the bench prompt, untraced: (a) 56.14 tok/s, (b) 57.43 tok/s, 512 ids each, md5
  2ddf0aab for both. The step graph holds across the dense->sparse DSA switch on real weights.
- MoE shadow, B + `STEP_GRAPH=1 MOE_DEBUG=1 -n 8`: 360 calls checked, 0 failed (0 routing), worst
  rel 9.7e-5 (tol 2e-2), ids = the oracle. BUT `resident_ready` returns false whenever
  `moe_dbg` is set, so this run took the host-staged path. It checks the MoE kernels, not
  the step graph. The step graph's evidence is the id equality above.

Decision: (b)/(d) agree within 0.1%, ids match (a) at -n 8 / 128 / 512, and the shadow passes.
The step graph is now ON by default; `TANG_GLM5_STEP_GRAPH=0` opts out. Run (e) confirms the
default takes the graph (164 launches).

Not verified:
- A shadow of the step graph itself (the MoE shadow forces the host path).
- Multi-slot. Resident (and so the graph) is still gated to n_slots == 1.
- The parallel env races in the test suite are not fixed.

## 2026-10-28 (later) — `TANG_GLM5_STEP_GRAPH=2`: the whole decode token as one graph replay (branch claude/glm5-stepgraph2)

On top of claude/glm5-resident 5c024107. Code: `resident.rs` (`step_whole`, `rs_embed`), `mla.rs`
(`*_p` kernels, `MlaCuda::{row_cap, reserve_pool, stage_ape, *_pos}`), `mla_dev.rs` (`window_dev_pos`).
- Position counter: `Resident::tok` is now `[token, pos]` u32. `rs_argmax(..., inc)` writes the
  token and, in the whole graph, advances `pos`. The DSA kernels read `pos` (`kv_len = pos + 1`):
  `mla_lat_norm_bf16_p` / `mla_idx_ln_p` write cache row `pos` from the cache BASE (the wk / gate
  GEMVs now go to scratch `yk`/`yg`, the LN kernel copies them in), `idx_pool_keys_p` computes the
  pool this token completes (grid 1, early exit), `idx_select_p` (grid 1, smem at the topk/kp cap),
  `mla_attn_abs2_p` (width = pos + 1 dense or `select_width`; smem sized to max(topk, select_width),
  which bounds the dense width because dense only holds while kv_len <= topk). The eager kernels
  call the same `__device__` bodies.
- Dense/sparse: two captured graphs `[dense, sparse]`, picked on the host from the cached `pos0`
  (one compare, no transfer). Before a sparse replay the eager incremental `idx_pool` catches up
  pools no replay computed (the dense graph skips the indexer); a no-op once caught up.
- Capacity: the graphs bake the DSA cache bases, so they are valid while `pos0 + 1 <= min row_cap`.
  Past it both drop, the token runs eagerly (regrows), the next token recaptures. Pools are
  reserved for `cap / kpool + 1` before capture; a pool realloc drops the other graph.
- Embed: `rs_embed` gathers row `tok[0]` of the GGUF `token_embd` (uploaded once, own quant type:
  F32/F16/BF16/Q8_0/Q4_K/Q5_K/Q6_K) with `_rn` ops in the host's evaluation order (no FMA
  contraction), broadcast to the hc streams in `MhcFused::r`. Other types: host gather + upload.
- Host per token: (8-byte upload only when `[tok, pos]` is out of step: after an eager token or a
  host window), graph launch, the DSA watermarks / mirror sizes, sync, 4-byte read.
- `TANG_GLM5_STEP_GRAPH=1` is unchanged; default unchanged.

Verified:
- mew tiny: `engine_gpu_resident_step_matches_host` now also runs graph=2. Host = resident =
  graph=1 = graph=2 greedy ids over 40 decode tokens across the dense->sparse switch (graph=2
  replays: 11 dense, 28 sparse). Passes alone.
- mew tiny, flash::glm5 suite: 78 passed / 0 failed / 2 ignored, with the trace test skipped (it
  passes alone). The parallel failure of the resident test was an env-var race. Fix:
  `glm5::ENV_LOCK`, held by the resident test and `gpu_slots_case` (which sets / removes
  `TANG_GLM5_MOE` / `KDA_PF2`).
- H200 gate with the flags as given (`TANG_GLM5_GPU_FFN=0`): ids = oracle for graph=1 and graph=2.
  But 0 of 45 FFNs were on the GPU, so the resident step never ran, and neither graph did.
- H200 with `GPU_FFN` unset (all FFNs on the GPU): the load OOMs at layer 31's FFN
  (`CUDA_ERROR_OUT_OF_MEMORY`, 24 of 45 uploaded), so every gate and bench failed. The resident
  step needs every FFN on one GPU, and this single H200 (141 GB) cannot hold them with this tree's
  footprint. The real-weights check needs the B300 or a multi-GPU box.

Not verified:
- Real weights with the FFNs on the GPU: the gate, the -n 512 md5 vs graph=1, the bench table.
- The capacity-regrow / recapture path (the tiny run stays inside its 72-row cap).
- Embed bit-exactness on the real `token_embd` type.

## 2026-10-28 — device-resident decode step: B300 35.8 -> 47.7 tok/s, step graph 56.2 (branch claude/glm5-resident)

The residual stream stays on the device for the whole decode token. Code: `glm5/resident.rs`, a
child module of engine.rs, called from `step_slot`.
- Embed: the host gathers one row and `MhcFused::begin_async` uploads it. That is the step's only H2D.
- mHC pre: `pre_dev` writes the normed x straight into the consumer's input buffer (KDA `xb`, the
  MLA scratch `x`, or the FFN's `Resident::x`).
- Consumers, all async on the fused-mHC stream:
  - KDA: `decode_dev`, its per-layer graph launched on the step stream.
  - DSA: `MlaProj::window_dev`. The selection stays on the device (`idx_run_dev_resident` +
    `attend_dev_sel`). `pos0`/`kv_len` are passed per token as plain kernel args; no device
    position counter. The dense->sparse switch is still a host branch.
  - FFN: `ffn_async`.
- mHC post: `post_dev` takes the producer's y pointer.
- Head: `end_dev`, then `rs_norm` (f64 sum, like the host `inv_rms`), the Q8_0 LM head (or a
  `k5_gemv` head for the f32 fixture), then `rs_argmax` (first max). One sync, a 4-byte read.
- DSA host mirrors are only resized. `DsaCache::stale_from` marks them, and `refresh_stale_dsa`
  pulls the rows before any host DSA use, or before a device cache regrow.

Default ON when every stage is resident (single slot, one GPU, MOE=fast, KDA fused/graph, mHC
fused, MLA resident). `TANG_GLM5_RESIDENT=0` forces the old path.

`TANG_GLM5_STEP_GRAPH=1` (default off) splits the step into 12 graph segments (n_dsa + 1) at the
DSA cores, which run eagerly between segment launches.
- Captured after one eager warm-up token. Any host window or slot resize drops the segments.
- First version diverged from token 3 on (deterministic; a re-capture every token gave the
  same wrong ids). Cause: the capture pass skipped the DSA items entirely, so their mHC post never
  ran and the next segment's pre was captured with `do_post=0`, dropping the attention output.
  Fix: `Pass::HostOnly` at each boundary.
- Also changed: kda `gemv8_launch_on(s)`; the fused decode's non-G5F arm had launched on the layer's own stream.

B300, tree = this branch rebased on 9c49294d. Flags: B (`MOE=fast GEMV8=1 KDA_PF2=1 MHC_FUSED=1
MLA_RESIDENT=1 GPU_HEAD=1 KDA_G5F=2 KDA_FUSED=1 KDA_GRAPH=1`). Bench prompt, `-n 128 --temp 0`,
`TANG_GLM5_TRACE=1`. All three -n 128 id streams are identical (md5 ab6f6605). The haiku gate =
`785 1196 6801 264 6386 38038 911 279` for resident and for step graph.

| path | decode tok/s | trace wall/token | launches | H2D# | D2H# | syncs |
|---|---|---|---|---|---|---|
| host (`RESIDENT=0`) | 35.83 | 25.8 ms | 561 | 182 | 226 | 216 |
| resident (default) | **47.67** | 19.1 ms | 563 | 1 | 1 | 1 |
| resident + `STEP_GRAPH=1` | **56.23** | 17.2 ms | 164 | 1 | 1 | 1 |

On 38bb2bed, before the rebase: host 30.72, resident 40.26 / 40.29, step graph 48.09. The rebase
gained +7.4 tok/s on the resident path vs +5.1 on the host path.
- Resident is GPU-bound: the single sync waits 14.7 of 19.1 ms. Host enqueue is ~4.4 ms and overlaps.
- The exact-timed trace (`TANG_TRACE_EXACT=1`) gives negative per-stage GPU ms here and does not
  time graph nodes. Use wall.

Still blocking a single whole-step graph:
1. DSA. `kv_len`/`pos0`, the append row pointers, and the attend width and smem
   (`(2*dl + width)*4`) change every token. Options: a device position counter read by
   `mla_attn_abs2` / `lat_norm` / `idx_ln` / the kv_a gemv's output offset, with attend sized to a
   max width; or `cuGraphExecKernelNodeSetParams` on ~14 nodes per DSA layer per token.
2. Sparse (kv > top_k): `idx_pool`'s `n_pool` and the select grid move too. It runs eager and
   copy-free, but has only been exercised on the tiny fixture (top_k 8), not real weights.
3. Embed: a pageable async H2D before the first segment. A device gather of the Q-quant
   `token_embd` row fed by `rs_argmax`'s output would close the token loop on the device.

Verified:
- mew tiny: `engine_gpu_resident_step_matches_host`. Host vs resident vs step-graph greedy ids are
  equal over 40 decode tokens across the dense->sparse switch. A host window afterwards reads the
  refreshed mirrors with logits max abs 0.
- mew tiny: flash::glm5 suite 76 passed / 0 failed / 2 ignored (pre-rebase tree, trace-flake test
  skipped).

Not verified:
- The trace-flake test run alone.
- The suite on the rebased tree.
- The mirror-refresh-on-regrow path (rows_fit false) on real weights.
- Multi-slot: the resident path is gated to n_slots == 1.

## 2026-10-28 (mtp-ready) — MTP + WIN_STEP bisect fix merged onto 9c49294d; WIN_STEP = K+1 by default under --mtp (branch claude/glm5-mtp-ready)
- Merge of claude/glm5-mtp-bisect (contains claude/glm5-mtp) onto frog/glm5-forward 9c49294d.
  Conflicts: this file (all entries kept), engine.rs CLI (`--slots` and `--mtp` both kept,
  now mutually exclusive), kda_dev.rs `core_pf2` (the WIN_STEP serial loop now passes the
  batcher's `slot`). Semantic conflict: `Snap`/`snapshot`/`restore` ported to the per-slot
  `kda[slot][layer]` / `dsa[slot][layer]` layout — MTP rewinds slot 0 only.
- `TANG_GLM5_KDA_WIN_STEP` unset now falls back to `mtp::WIN_STEP_DEFAULT`, which
  `mtp::cli_run` sets to K+1 (the verify window), so `--mtp K` is exact by default; env
  still overrides (`=0` restores the chunked core). Plain decode: default 0, unchanged.
- Test `engine_gpu_win_step_window_matches_steps` (tiny, PF2): one t-token window with
  win_step >= t vs t single decode steps, t = 2,3,4 — every logit row max abs 0 (bitwise),
  and the step after the window also 0. Its PF2 env set/remove raced
  `engine_gpu_slots_batch_matches_single` in the parallel run (slot 0 max abs 9.8); fixed
  with a test `ENV_LOCK` around both loads.
- Verified: local CPU `flash::glm5` 42/0. mew 3090 (TANG_REQUIRE_CUDA=1) 80 passed / 1
  failed — the known `engine_gpu_trace_is_transparent` parallel flake (passes alone).
- H200 #1 (/root/w-mtp, B flags): haiku -n 8 --temp 0 ids gate plain = oracle
  `785 1196 6801 264 6386 38038 911 279`; --mtp 1 = oracle (accept 4/4, win_step=2 logged).
  --mtp 2, --mtp 3 and --mtp 3 with WIN_STEP=0 exited rc=0 but their ids were NOT read
  back: the box was stopped externally before the logs could be fetched. The tok/s bench
  (plain, --mtp 1/2/3, and the WIN_STEP=0 vs default cost at depth 3) never ran — the GPU
  lock was contended by other agents for ~3 h. OPEN: rerun all of it on a live H200.


## 2026-10-28 — KDA projections requantized Q8_0 → Q4_K / Q5_K: quality, sizes, kernel µs (H200, branch claude/glm5-kda-q4)

The 34 KDA layers' q/k/v/o/f_a/g_a (204 tensors) are Q8_0 in UD-Q4_K_XL: 4.886 GB read per
token, more than the MoE's active experts. `tools/glm5-kda-requant.py` rewrites ONLY those tensors
(gguf-py dequant → ggml's `ggml_quantize_chunk`, no imatrix); every other tensor and every KV is
copied verbatim, shards without a target (1, 6) are symlinked. The MLA layers' attn_output
(16384 wide) is untouched. No loader change was needed: `moe::gpu::upload` and every KDA GEMV path
(k5_gemv / g5_dot, g5f_gemv) already take any G5W type; gemv8 is per-matrix Q8_0-only and falls back.

| KDA proj type | KDA bytes/token | whole GGUF |
|---|---|---|
| Q8_0 (as shipped) | 4.886 GB | 199.71 GB |
| Q5_K | 3.162 GB | 197.98 GB |
| Q4_K | 2.587 GB | 197.41 GB |

Quality: llama.cpp `llama-perplexity` (8ca4060, CUDA, experts on the CPU) on
`bench/glm5-ppl/eval.txt` (prose + chat + Rust/Python, 3072 tokens scored: `-c 512 --chunks 6`),
KL against the original's logits on the same chunks. The CPU reference (`glm5-ref`) was tried
first: >10 min for 36 tokens, single-threaded — not usable for this.

| KDA proj | PPL | PPL ratio | mean KLD | same top-1 | RMS Δp |
|---|---|---|---|---|---|
| Q8_0 | 3.4315 ± 0.188 | — | — | — | — |
| Q5_K | 3.4807 ± 0.194 | 1.0143 ± 0.0082 | 0.0329 | 92.8 % | 5.7 % |
| Q4_K | 3.4850 ± 0.195 | 1.0156 ± 0.0095 | 0.0431 | 92.5 % | 6.5 % |

Q5_K and Q4_K are indistinguishable on PPL here (the gap is inside the error); Q4_K costs ~30 %
more KLD than Q5_K. Both cost ~1.5 % PPL, which is the price of dropping the imatrix-free
Q8_0 → 4/5-bit step on 1/4 of the per-token weight bytes; an imatrix requant was not tried.

tang engine, Q4_K GGUF (`G5F=2 GEMV8 KDA_PF2 MOE=fast GPU_FFN_LAYERS=0-29`):
- ids gate `-n 8 --temp 0` = `785 1196 6801 264 6386 38038 911 279` = the oracle, also with
  GPU_FFN=0. The requant does not move the first 8 greedy tokens.
- 3 greedy `-n 128` generations vs the original (scripts/glm5-kda-q4-eval.sh; logs in
  /root/w-q4/out on the H200). Common prefix: code (Rust fibonacci) 72/128, prose (harbor town)
  2/128, chat (sourdough tips) 49/128. Every divergence is a paraphrase at the same plan step
  ("u64 or u128, or discuss overflow" vs "u64 or u128 for larger values, though u64 overflows at
  F(93)"; "is asking me to describe" vs "wants a description of"); the content, structure and
  code that follow are the same in kind. Q5_K generations were not run (lock time).

Kernel: `kda_quant_real_bench` (new, kda_decode.rs): layer 0, CUDA events, T=1, median of 50; both
kernels checked against the CPU f32 dequant · x (rel 3.4e-3 … 3.8e-3 everywhere = the int8
activation band, same as Q8_0). `k5_qgemv_q4k5/9` (new, kda_k5.cuh) is k5_qgemv with a Q4_K dot
(lane = 32-wide sub-block, d·sc·Σq·x − dmin·m·Σx), so G5F=2 keeps its one launch for
q|k|v|f_a|g_a. Q5_K has no qgemv variant and runs on g5f_gemv.

| µs (TB/s) | Q8_0 g5f | Q8_0 qgemv | Q5_K g5f | Q4_K g5f | Q4_K qgemv |
|---|---|---|---|---|---|
| q / k / v | 19.8 (1.80) | 16.7 (2.13) | 16.0 (1.44) | 15.1 (1.25) | 13.4 (1.41) |
| f_a / g_a | 9.6 | 7.0 | 8.4 | 8.5 | 7.3 |
| o | 16.8 (2.12) | 18.8 (1.89) | 15.8 (1.46) | 14.3 (1.32) | 13.6 (1.38) |
| merged q\|k\|v\|f_a\|g_a | — | 39.4 (2.74) | — | — | 26.8 (2.13) |
| per layer (G5F=2 = merged + o) | 95.3 | **58.2** | 80.7 | 76.6 | **40.4** |
| × 34 layers, ms/token | | 1.98 | 2.74 | | **1.37** |

G5F=2 vs G5F=1, 32 naive decode steps on Q4_K and Q5_K: 0 difference (Q8_0: 1.7e-5). Not
understood yet; the per-matrix timings differ, so the two kernels did run.

Reading:
- Q4_K + qgemv saves 0.61 ms/token of KDA projection time on the H200 (−31 %), not the −47 % the
  bytes promise: the 4-bit dot is ALU-heavier and the kernels sit at 1.4–2.1 TB/s instead of
  2.1–2.7. Still the best KDA config measured.
- Q5_K is SLOWER than today's Q8_0 G5F=2 (80.7 vs 58.2 µs/layer) until it gets a qgemv variant;
  even then its byte saving is 35 %, not 47 %.

Recommendation: ship Q4_K KDA projections with G5F=2 (k5_qgemv_q4k) as an opt-in GGUF, not Q5_K:
same PPL within error, 0.6 ms/token less on the H200 (expect similar on the B300), the ids gate
holds. Before making it default: an imatrix requant (UD's own recipe for the rest of the file),
a longer PPL text (3072 tokens gives ±0.9 % on the ratio), and the B300 A/B tok/s.
NOT verified: B300; fused/graph path bit-equality on Q4_K (decode_fast uses the same
k5_qgemv via g5f_x/g5f_o, so it should hold); Q5_K generations; multi-slot batched decode (T>1
takes g5f_gemv, which handles Q4_K).

Ops: the network volume is quota-limited (~290 GB; a second 200 GB copy failed at 100 GB), and
/dev/shm is noexec — the requant shards were split over /workspace, /root and /dev/shm via
symlinks, one variant at a time, and deleted after. llama-perplexity with experts on the CPU took
~7 min to load and ~400 s per 512-token chunk under the other agents' CPU load.
## 2026-10-28 (integration round 2) — k5d_tail + FFN repack merged: B300 35.02 tok/s decode with repack off, now the default (branch claude/glm5-integrate2)

Merged onto frog/glm5-forward 38bb2bed: claude/glm5-kda-kernels 1073e571 (k5d_tail, `TANG_GLM5_KDA_TAIL`
default on) and claude/glm5-ffn-repack 3b96bac7 (Q8_0/Q6_K repack at upload, g5f_route,
`TANG_GLM5_MOE_REPACK` default on). Only conflict: this file (both entries kept). CPU flash::glm5 40/40;
mew 3090 CUDA flash::glm5 78/78 (engine_gpu_trace_is_transparent flaked in the parallel run, passed alone).

B300 (all 45 FFN layers resident), scripts/glm5-bench.sh, 128-token decode on the 238-token prompt,
ids = oracle `785 1196 6801 264 6386 38038 911 279` on every run. Base flags B = MOE=fast GEMV8=1
KDA_PF2=1 MHC_FUSED=1 MLA_RESIDENT=1 GPU_HEAD=1 KDA_G5F=2 KDA_FUSED=1 KDA_GRAPH=1. Stage ms per token (wall):

| run              | decode tok/s | kda  | dsa  | mhc  | ffn  | head |
|------------------|-------------:|-----:|-----:|-----:|-----:|-----:|
| B (round 1)      | 30.85        |      |      |      |      |      |
| B                | 33.51        | 4.28 | 6.85 | 9.34 | 6.41 | 0.61 |
| B + KDA_TAIL=0   | 33.01        | 6.80 | 6.60 | 8.25 | 5.92 | 0.51 |
| B + MOE_REPACK=0 | 35.02        | 3.95 | 6.67 | 8.43 | 6.65 | 0.52 |
| B (repeat)       | 33.51        | 4.29 | 6.84 | 9.40 | 6.40 | 0.57 |

Tail: kda stage 6.80 → 4.28 ms/token, as the H200 bench predicted. Repack: the ffn stage itself is
0.24 ms faster with it on, but the repack-on runs are slower overall (mhc_pre and dsa wall up ~1 ms,
sync-copy 7.3 vs 5.6 ms). The B300 was shared with another agent's runs; the two B runs agree to
0.01 tok/s, the A/Bs were single runs. Repack-off is the faster whole run (35.02 vs 33.51 x2), so `TANG_GLM5_MOE_REPACK`
now defaults OFF; `=1` opts in (H200, where it measured 112 → 92 µs/routed layer). Why it loses on the
B300 is unexplained. MoE shadow on the B300 (B, repack on, TANG_GLM5_MOE_DEBUG=1, -n 8): 360 calls checked, 0 failed, 0 routing, worst rel 9.7e-5, ids = oracle.

## 2026-10-27 (night) — k5d_tail: beta + gate2 + step + o_proj q8 in one launch, fused G5F step 175 → 87 µs/layer (H200, branch claude/glm5-kda-kernels)

The fused G5F=2 step had seven launches. Four of them were never tuned: k5_beta, k5d_gate2, k5d_step
and k5_q8 over yb. Together they took ~108 µs/layer, against 62 µs for the two projection GEMVs.
`k5d_tail` (kda_decode.cuh) replaces all four with ONE launch: grid nh=64, block hd=128.

Why one block per head works: everything a head needs is head-local.
- its 128 G/G2 rows (f_b/g_b over the shared, hd-wide g1/g2a);
- beta[h] (b_proj row h);
- the 3 conv channels each thread owns;
- the state S[h].

What the kernel does differently (the arithmetic is the same expressions in the same order):
- The head's f_b/g_b rows (2×17 KB) and its b_proj row (4.3 KB) are staged into smem with
  coalesced 16 B loads. k5d_gate2 had each thread stream its own 136 B row byte by byte.
- The f16 scale is decoded once per 32 values, and expf(G) once per row.
- The state column is read once into registers and written once: 8 MB/layer of traffic instead
  of k5d_step's 16 MB, for a state of 64×128×128 fp32 = 4 MiB/layer.
- The conv history is updated in place.
- o_proj's Q8 codes are written straight into xq8, so o_proj runs k5_qgemv with no q8 pre-pass.

Launch count per step: 7 → 4. `TANG_GLM5_KDA_TAIL=0` restores the old launches.
Preconditions, checked on the host (if they fail, the old path runs): f_b/g_b/b_proj Q8_0 and
hidden ≤ 4096, a multiple of 256.

Bug on the first run: the tail read g1/g2a at `h*D + x`. Those vectors are hd wide and shared by
every head, so only head 0 was right (8064/8192 g differed). The step-0 tap diff in the bench
located it.

H200, real layer 0, `kda_tail_real_bench`. CUDA events around each launch, back to back on one
stream, median of 56 steps:

| kernel | old µs | tail µs | MB moved (old) | TB/s (old) | launches |
|---|---|---|---|---|---|
| k5_q8 x | 5.5 | 5.5 | 0.02 | — | 1 / 1 |
| k5_qgemv q\|k\|v\|f_a\|g_a | 39.4 | 39.2 | 108.1 | 2.74 | 1 / 1 |
| k5_beta | 33.1 | — | 0.29 | 0.009 | 1 / — |
| k5d_gate2 | 51.3 | — | 2.2 | 0.04 | 1 / — |
| k5d_step | 17.8 | — | 16.8 (2× r+w state) + 1.1 conv | 1.0 | 1 / — |
| k5_q8 yb | 5.4 | — | 0.04 | — | 1 / — |
| **k5d_tail** | — | **19.0** | 2.5 weights + 8.4 state + 1.1 conv = 12.0 | 0.63 | — / 1 |
| k5_qgemv o | 22.8 | 22.7 | 35.7 | 1.57 | 1 / 1 |
| **step total** | **174.5** | **86.5** | | | 7 / 4 |

decode_fast host wall: 104 µs/step fused, 103 µs/step graph (64 steps, including the capture).

Parity, 64 steps against the old launches: 0 differing bits in beta/g/g2/o/y at step 0, and in
output, state and conv over all 64 steps, for tail eager, decode_fast fused and decode_fast graph.
The graph capture works with the tail.

ids gate on H200 #2 (bin md5 b6a088bb). Flags: GPU_FFN=0 GEMV8=1 KDA_PF2=1 MHC_FUSED=1 MOE=fast
MLA_RESIDENT=1 GPU_HEAD=1 KDA_G5F=2 KDA_FUSED=1 KDA_GRAPH=1. `-n 8 --temp 0` gives
`785 1196 6801 264 6386 38038 911 279` = oracle at both KDA_TAIL=1 and 0. The TRACE `-n 32` ids are
identical between the two.

| KDA_TAIL | kda ms/token (wall, 34 layers) | launches |
|---|---|---|
| 0 | 8.60 | 15/34/34/69 |
| 1 | **5.17** | 9/34/34/69 |

The decode tok/s of these runs is dominated by the CPU FFN (GPU_FFN=0), so it is not a speed number.
The B300 has not been run.

What is left: k5d_tail is at 0.63 TB/s. Its floor is ~3 µs, but it only has 64 blocks and runs
serial 128-long dot and fma chains. The next 10 µs are in the projections (62 µs), plus the two
k5_q8 launches and the launch gaps.

## 2026-10-27 — MoE FFN: Q8_0 / Q6_K repacked at upload, fused gate|up rows, one routing launch (H200, branch claude/glm5-ffn-repack)

Second H200 (pod nxn0swopws2xi8), real layers, `real_layers_fast_matches_cpu`. New knob
`TANG_GLM5_MOE_SPLIT=1` (+ `TANG_TRACE_EXACT=1`) prints per-kernel GPU µs / declared bytes /
TB/s over 50 calls (trace on, so totals include per-launch event gaps; the first kernel of a call
also absorbs the gap since the previous call — the 12 µs `g5f_q8(x)` vs 6 µs `g5f_q8(h)`).

Kernel split, T=1, µs (TB/s) — before = this branch with `REPACK=0 STAGE=1 MINB=6 FUSE2=0 ROUTE1=0`
(= frog/glm5-forward 38bb2bed behaviour; the first H200 gave the same numbers within 1-2 µs):

| layer | q8 x | router | select | gu | q8 h | down | sum | total |
|---|---|---|---|---|---|---|---|---|
| l4 before | 12.2 | 8.3 | 10.4 | 47.6 (1.96) | (in q8) | 37.8 (1.46) | 6.0 | 122.3 |
| l4 after | — | route 17.1 | — | 41.9 (2.23) | 5.8 | 30.1 (1.83) | 6.0 | 100.8 |
| l12 before (Q6_K down, Q8_0 shared) | 11.9 | 8.1 | 10.3 | 47.8 (1.95) | | 53.4 (1.20) | 6.1 | 137.5 |
| l12 after | — | route 17.3 | — | 42.0 (2.22) | 5.8 | 32.2 (1.99) | 6.0 | 103.2 |
| l0 dense Q8_0 before | 12.5 | | | 50.4 (2.12) | | 72.1 (0.74) | | 135.0 |
| l0 after | 12.6 | | | 34.1 (3.13) | | 26.1 (2.05) | | 72.8 |

Untraced bench, device x/y, µs/call T1 (before -> after): l0 129 -> 66, l4 112 -> 92 (1.36 -> 1.67
TB/s of 153 MB), l11 132 -> 97, l12 127 -> 94. T8: l4 574 -> 545, l12 728 -> 491, l0 822 -> 327.
Parity: fast-q8x2 rel l2 worst 5.1e-5 (unchanged), routing sets = CPU; ids with GPU_FFN_LAYERS=0-29
+ the baseline flags: `785 1196 6801 264 6386 38038 911 279`.

Steps:
1. Repack at upload (`upload_ffn`, DevMoe only; GGUF untouched; `TANG_GLM5_MOE_REPACK=0` off):
   Q8_0 -> ty 108 `qs[k] | f16 d[k/32]`, Q6_K -> ty 114 `{ql,qh}[192] x nb | sc[16] x nb | d x nb`,
   rows padded to 16 B. `g5f_lane_dot32r` reads both as 16 B loads per 32-chunk; the naive
   reference decodes them through `g5_deq_r` (needs k). l12 down 54.8 -> 39.2, l0 135 -> 101.
2. Defaults `MINB` 6 -> 8 and `STAGE` on -> off: with every routed/shared type on the 32-chunk dot,
   the smem staging round trip costs more than it saves (l12 122 -> 111). Unroll 2/4: worse.
3. `G5F_FUSE2` (default on): gate and up rows (and the down row pair) dotted in one loop, both rows'
   loads in flight per iteration — gu 46 -> 42.
4. `g5f_route` (`TANG_GLM5_MOE_ROUTE1`, default on for n_group <= 1, F32 router, ne >= hid/32):
   q8(x) + router + select in one launch; the last block per token (atomic counter, reset on exit)
   runs the select. 3 launches -> 1, ~30 -> 17 µs traced.

Still open (target 60 µs not reached): gu 2.2 TB/s and down 1.8-2.0 TB/s on 93 / 55 MB; route 17 µs
(256 router rows = 256 blocks, then a serial top-8 in one warp); sum 6 µs + q8(h) 6 µs launches.
Next: router at more rows per block / split-K, fold q8(h) into gu (per-32 amax needs 32 rows of one
slot in one block: a block-per-32-rows gu), fold sum into the next consumer.

## 2026-10-27 (integration) — batcher + kda-gemv + moe + bench + step-graph merged: B300 30.8 tok/s decode (branch claude/glm5-integrate)

Merged onto frog/glm5-forward 66aa4b5e: frog/glm5-batcher, claude/glm5-kda-gemv, claude/glm5-moe,
claude/glm5-bench, claude/glm5-step-graph (NOT claude/glm5-mtp). Conflicts: NOTES.md (every entry
kept); kda_dev.rs (whitespace around `Ok(Self {` vs gemv's `Self { .. }.with_seg_tables()`: gemv's);
kda_decode.rs (gemv's G5F if/else around the proj launches; `mark(1)` after the proj stage as at
66aa4b5e — the batcher's own merge had moved it before). moe_cuda.cuh / moe_gpu.rs auto-merged (gemv
never touched moe_cuda.cuh; its qgemv9/17 handles are in). Fixup: `kda_g5f_real_bench` passes the
batcher's new `slots` arg to `new_gguf` (cuda test build failed without it).

Verified: local `cargo test -p tang-llm --lib flash::glm5` 40/0. mew (TANG_REQUIRE_CUDA=1, tiny):
76 passed, 1 failed, 2 ignored — the failure is `engine_gpu_trace_is_transparent` ("a trace report"
None), which fails the same way in the parallel suite at 66aa4b5e and at base+batcher, and passes
run alone on the merged tree: pre-existing parallel-test flake, not a merge regression.

B300 (scripts/glm5-bench.sh, src md5 matched, bin 31e14c85, all 45 FFNs on GPU, -n 128 on the
238-token prompt). Base = `MOE=fast GEMV8=1 KDA_PF2=1 MHC_FUSED=1 MLA_RESIDENT=1 GPU_HEAD=1`.
Every gate printed `785 1196 6801 264 6386 38038 911 279`. Stage = wall ms per decode token.

| config | decode tok/s | trace wall/token | launches | kda | ffn | dsa | mhc_pre attn+ffn | head |
|---|---|---|---|---|---|---|---|---|
| base (was 22.87 at 66aa4b5e) | 25.19 | 35.5 ms | 1153 | 12.93 | 6.99 | 6.72 | 3.83+3.89 | 0.52 |
| + KDA_G5F=2 (was 25.60) | 27.39 | 32.8 ms | 1017 | 10.31 | 7.00 | 6.72 | 3.82+3.81 | 0.53 |
| + G5F=2 FUSED=1 GRAPH=1 | 30.85 | 29.9 ms | 647 | 6.88 | 7.10 | 6.80 | 3.96+3.90 | 0.59 |
| same, repeat | 30.77 | 30.0 ms | 647 | 6.92 | 7.11 | 6.81 | 4.02+3.90 | 0.54 |

Repeat spread 0.08 tok/s (0.3%). The KDA stage is now ~equal to ffn and dsa; dsa's o_proj
(5.3 ms, sync+D2H per layer) and kda's sync+read are the remaining host waits.

## 2026-10-27 (late) — batched KDA decode: conv-slot offset fix, real-weights gate (H200, branch frog/glm5-batcher)

Bug: a slot's conv history is `[chan][d_conv−1]` = conv_dim·(d_conv−1) floats (kda.rs
`ConvState`, `DevKda::read_conv_at/seed_at`, `kda_step`'s `bi*KDA_CONV*(KDA_KC−1)`), but
`kda_conv_commit_rows` offset by `SLOT[bi]*KDA_CONV` — slot s>0 committed into slot s/3's
region. Same missing factor in the b>1 offsets of the prefill `kda_conv`/`kda_conv_commit`
(single-row callers never saw it). All four kernels now agree.

Verified: mew glm5 suite (TANG_REQUIRE_CUDA=1) 76 passed / 0 failed. H200 (bin md5 835cb84e,
src md5 matched both sides), GEMV8 KDA_PF2 GPU_FFN=0:
- `-n 8 --temp 0`: `785 1196 6801 264 6386 38038 911 279` = oracle.
- TRACE `-n 32`: decode kda 12.76 ms/token wall (34 layers; forward baseline 13.05).
  Single-slot decode does not take the batched path, so this is a no-regression number.
- `--slots 8 --diverse -n 8` (KDA_MAXT=64): slot 0 = oracle 8/8, 7/7 other slots differ;
  trace shows `kda_step` and `kda_conv_commit_rows` once per window (27 calls each),
  i.e. the batched path ran. Per-step wall 21.9 s median (FFN on CPU dominates; was 20.5 s
  with per-row KDA, within noise).

## 2026-10-27 — TANG_GLM5_KDA_G5F: fast row kernels on the KDA projections (branch claude/glm5-kda-gemv)

Flag, read at load, default 0: `=1` puts q/k/v, f_a, g_a, o_proj on `g5f_gemv` (with k5_q8's codes);
`=2` uses `k5_qgemv` (kda_k5.cuh). That kernel does q|k|v|f_a|g_a in ONE T=1 Q8_0 launch: per lane,
all 16 B nc loads are issued first, then staged in smem, then one dp4a chain per 32 elements.
o_proj gets its own launch. Wired into the naive PF2 path (the one that ships) and into the fused
path. When G5F is on, the fused path runs beta on k5_beta instead of k5d_proj. k5d_proj's beta is one
THREAD per 4096-wide row, ~1.34 ms/layer on the H200. That is the likely cause of the B300's
fused 13 → 51 ms/token regression (34 × 1.34 ≈ 45 ms). The regression cause is not measured yet.

H200, real layer 0, CUDA events, kernel only, T=1, median of 50 (`kda_g5f_real_bench`):

| matrix | MB | k5_gemv8 µs | g5f_gemv µs | k5_qgemv µs | qgemv TB/s |
|---|---|---|---|---|---|
| q / k / v (8192×4096) | 35.7 | 23.3 | 21.2 | 16.3 | 2.19 |
| f_a / g_a (128×4096) | 0.6 | 10.0 | 10.4 | 6.9 | — |
| o (4096×8192) | 35.7 | 30.5 | 19.5 | 18.7 | 1.91 |
| merged q\|k\|v\|f_a\|g_a | 108.1 | — | — | 39.4 | 2.74 |
| per layer, 6 matrices | 143.8 | 120.2 | 103.8 | 81.4 (merged + o: 58.1) | 2.47 |

Parity: G5F=1 vs gemv8 naive is BIT-IDENTICAL (rel 0). This is expected: g5f's Q8_0 lane dot walks
the same lane*8 + 256i steps with the same fmaf and reduce. The test now asserts nonzero only for
G5F=2. NOT yet measured: G5F=2 parity, fused bit-equality, ids gate, trace kda ms/token.
>3 TB/s not reached (best 2.74 merged).

### B300, 2026-10-27: verified (src md5-matched; Compiling tang-llm in the build log; all 45 FFN layers on the GPU)
Base flags: `MOE=fast GEMV8=1 KDA_PF2=1 MHC_FUSED=1 MLA_RESIDENT=1 GPU_HEAD=1`.

The real-layer-0 bench: T=1, CUDA events, kernel only, median of 50.

| matrix | MB | k5_gemv8 µs | g5f_gemv µs | k5_qgemv µs (TB/s) |
|---|---|---|---|---|
| q / k / v (8192×4096) | 35.7 | 22.5 | 19.2 | 15.3 (2.33) |
| f_a / g_a (128×4096) | 0.6 | 11.1 | 11.4 | 8.0 |
| o (4096×8192) | 35.7 | 30.4 | 19.6 | 17.5 (2.03) |
| merged q\|k\|v\|f_a\|g_a | 108.1 | — | — | 35.6 (**3.04**) |
| per layer, 6 matrices | 143.8 | 120.0 | 99.9 | 79.5; merged + o **53.1** |

- Parity, 32 decode steps.
  - G5F=1 vs gemv8: bit-identical. This is by construction: the same lane dot and the same order.
  - G5F=2 vs gemv8: out rel 1.7e-5, state rel 2.5e-7.
  - Fused and graph vs naive: 0 differing bits in output and state, at both G5F=1 and G5F=2.
- Fused per-stage GPU µs/step (proj / gate2 / step / o_proj / total):

| G5F | proj | gate2 | step | o_proj | total |
|---|---|---|---|---|---|
| 0 | 1347 | 52 | 36 | 38 | 1473 |
| 1 | 109 | 52 | 36 | 40 | 237 |
| 2 | 68 | 52 | 26 | 23 | 169 |

  This CONFIRMS the fused regression. With gemv8, `k5d_proj` computes only beta, one thread per
  4096-wide row, and that is 1.28 ms of the 1.35 ms proj stage. Under G5F the fused path runs
  beta on k5_beta.
- ids gate (n 8, temp 0): all = `785 1196 6801 264 6386 38038 911 279` at G5F=0, 1 and 2. The
  -n 32 trace runs produce the same 32 ids at all three settings.
- TRACE -n 32, decode per token (kda = host wall; the trace's GPU column is unreliable on the B300):

| G5F | kda ms/token | launches/token | decode tok/s |
|---|---|---|---|
| 0 | 12.91 | 1111 | 23.75 |
| 1 | 12.73 | 1111 | 23.85 |
| 2 | **10.34** | 975 | **25.60** |

Reading:
- The KDA stage on the naive path is ~300 µs/layer of host wall against ~170 µs of GPU work.
  The rest is launch and sync overhead.
- The fused+graph path at G5F=2 is now 187 µs/step host in the bench, vs 224 for naive.
- It has not been A/B'd end-to-end with G5F=2 yet. That is the next step: graph + G5F=2.

Ops: my sync script's `rsync --delete` wiped the run scripts and logs I had written inside the
synced dir (/root/w-gemv, /root/w-gemv2). Nothing on the pod deletes files. Keep run artifacts
outside the rsync target.

## 2026-10-27 — fast MoE: grouped launches at full fan-out, 32-chunk K-quant dot (H200, branch claude/glm5-moe)
Real model, `TANG_GLM5_MOE=fast TANG_GLM5_GPU_FFN_LAYERS=0-29 TANG_GLM5_GEMV8=1 TANG_GLM5_KDA_PF2=1`,
-n 32 (31 decode tokens). Active weight bytes per routed layer per token: 153 MB (l4: 8 experts
Q4_K/Q4_K/Q5_K + Q8_0 shared + F32 router; `DevMoe::active_bytes`), so the H200 floor is
153 MB / 4.8 TB/s = 32 µs and the 1.5x target 48 µs.

| | base (frog/glm5-forward 66aa4b5e) | after (this branch) |
|---|---|---|
| engine FFN call, T=1 mean over 30 resident layers (`TANG_GLM5_MOE_PROF`, host x/y, trace on) | 244 µs | 157 µs |
| resident FFN per token (30 layers) | 7.3 ms | 4.7 ms |
| bench l4 T1, device x/y (`real_layers_fast_matches_cpu`) | 210 µs (0.73 TB/s) | 107 µs (1.44 TB/s) |
| bench l11 / l12 T1 | 227 / 211 µs | 127 / 122 µs |
| bench l0 (dense Q8_0) T1 | 132 µs | 130 µs |
| trace per token-layer: gu / down / router / select | 101 / 77 / 13.1 / 12.0 µs | 48.8 / 38.8+6.0 (sum) / 8.8 / 11.7 µs |
| trace kernel BW: gu / down | 1.21 / 0.85 TB/s (after step 1) | 1.95 / 1.44 TB/s |
| launches per routed layer | 6 (q8, router, select, gu, q8, down) | 7 (+ g5f_sum) |
| host copies / syncs per layer | 1 H2D x + 1 D2H y (16 KB each) / 1 | same in the engine (`ffn_host`); `ffn_async` now has none |

Steps (each verified: ids `785 1196 6801 264 6386 38038 911 279` = oracle, MoE shadow 240/240):
1. `g5f_gu` grid (rows/4, S, T): one warp per (slot, row) instead of a warp walking 9 slots;
   `g5f_down` splits slots over a block's warps (`sw`, 1 on dense — the first cut regressed l0
   132 -> 263 µs by idling 3 of 4 warps). l4 210 -> 169.
2. Q4_K / Q5_K `g5f_lane_dot32`: a lane owns whole 32-weight sub-blocks (scale/min decoded once
   per 32, nibbles / high bits / int8 x as 16-byte loads) — the 8-weight decode was the bound,
   not bytes (q8x2 vs q8 differed by ~10%). Router 4 warps a row (288 rows were 72 blocks),
   select caches scores in smem. `__launch_bounds__(128, G5F_MINB)`, default 6 (<= 80 regs;
   down had 139-154): l4 169 -> 126 (MINB 1) -> 110 (MINB 6; 4: 115, 8: ~109).
3. Routed down split: `g5f_down_split` one warp per (slot, row pair) -> yp, `g5f_sum` in slot
   order. 110 -> 107 bench; trace down 61 -> 39 + 6.
Tried and dropped: double-buffered down staging (cp.async groups) — +regs, 169 -> 188.
`TANG_GLM5_MOE_STAGE=0` (no smem staging) is slower (149 vs 126). ncu does not run here
(nsight-compute 2025.1 lacks a cudarc symbol: `cuTensorMapEncodeIm2colWide`).
Still open: 107 µs is 2.2x the 48 µs target. The remaining gap: Q8_0 shared expert and Q6_K
down still use the 8-weight decode (Q8_0's 34-byte and Q6_K's 210-byte blocks are not 16-byte
aligned — a repack at upload would fix both); 7 launches + the engine's per-layer host round
trip (x up, y down, sync: ~25 µs of the 157) — `ffn_async` is the device-pointer entry for
whoever owns engine.rs; T=8 (565 µs l4) re-decodes each row per token.

## 2026-10-27 — the all-GPU B300 baseline: 22.9 tok/s decode (scripts/glm5-bench.sh)

New harness `scripts/glm5-bench.sh` (+ `glm5-bench-remote.sh`, `glm5-bench-prompt.txt`;
how-to in `docs/glm5-bench.md`): rsync + md5 guard + cuda build, then the ids gate
(haiku, -n 8 = the CPU oracle) and a timed -n 128 decode on a 238-token prompt with
`TANG_GLM5_TRACE=1`, all 45 FFNs on the GPU (`TANG_GLM5_GPU_FFN*` unset). B300
tree-verdict-b300 (q9ce4c6w5bjmpv), tree 66aa4b5e (frog/glm5-forward), src md5
e6102a1a… both sides, binary 61ba4e0d…, weights already present (no refetch).
Every row's ids gate passed (8/8 = the oracle). Stage columns are wall ms per decode token.

base = `MOE=fast GEMV8=1 KDA_PF2=1 MHC_FUSED=1` (all `TANG_GLM5_`)

| config | prefill tok/s | decode tok/s | kda | dsa | mhc | ffn | head |
|---|---|---|---|---|---|---|---|
| (a) naive (no flags) | 1.3 | 0.34 | 359.05 | 1654.01 | 294.82 | 589.84 | 12.06 |
| (b) base | 3.0 | 0.62 | 13.26 | 1565.90 | 8.73 | 11.48 | 12.86 |
| (c) base + KDA_FUSED=1 | 2.9 | 0.60 | 52.70 | 1564.09 | 8.81 | 11.39 | 13.23 |
| (d) base + KDA_GRAPH=1 | 2.9 | 0.60 | 50.85 | 1572.86 | 8.58 | 11.35 | 11.93 |
| **(e) base + MLA_RESIDENT=1 GPU_HEAD=1** | **12.8** | **22.87** | 12.91 | 6.86 | 8.23 | 11.19 | 0.52 |
| (e) + KDA_FUSED=1 | 12.7 | 12.06 | 52.36 | 6.88 | 8.34 | 11.16 | 0.52 |
| (e) + KDA_GRAPH=1 | 12.8 | 12.55 | 50.45 | 6.89 | 8.25 | 11.13 | 0.57 |

- **(b) is not the speed config**: without `TANG_GLM5_MLA_RESIDENT=1` DSA stays on the
  host-staged MLA path (1.57 s/token, 97% of wall). The real all-GPU baseline is (e),
  **22.9 tok/s** (traced run: 25.1 tok/s, 39.8 ms/token). Quote (e) as base from now on.
- **KDA_FUSED and KDA_GRAPH regress on the B300**: kda 12.9 → ~51 ms/token, decode
  22.9 → ~12 tok/s (GEMV8 held fixed across the pair). Correct ids, slower. Unexplained.
- Budget at (e): kda 12.9, ffn 11.2, mhc 8.2 (pre.attn 3.8 + pre.ffn 3.8, both ~0.08 ms
  ×45 with a 0.74 MB D2H each), dsa 6.9, head 0.5. 1111 launches/token, 226 D2H calls.
- The trace's per-stage `gpu ms` column is not trustworthy on this box (kda −15.7 ms,
  sum 164 ms > 39.8 ms wall) — use wall.
- The "52 tok/s" in the program brief is the verdict's 0.52 tok/s.

## 2026-10-27 — decode-step audit: every host round trip per token, and what blocks a whole-step graph (branch claude/glm5-step-graph)

Trace now counts host waits (`syncs`: Stream/Event sync, per stage) and `TANG_TRACE_EXACT=1`
times each launch from its own start event (the default start = previous launch's end
counts host gaps as GPU: tiny q8_0 on the 3090 read 6.5 ms "GPU" per 2.9 ms token;
exact 2.1; nsys kernel sum 1.68).

H200 gate (GEMV8 PF2 MHC_FUSED MOE=fast GPU_FFN_LAYERS=0-29 MLA_RESIDENT GPU_HEAD):
ids `785 1196 6801 264 6386 38038 911 279` = oracle.

Per decode token the structure is fixed: every mixer/FFN takes a host `x` and returns a
host `y`, so each stage is H2D + launches + sync + D2H. The B300 count (226 D2H/token)
decomposes exactly (34 KDA, 11 DSA dense, 45 FFN, 45×2 mHC pre):

| site (file:line) | stage | D2H/token | why |
|---|---|---|---|
| kda_dev.rs:717 sync, :721 `outb.read` (finish) | kda/sync+read | 34 | y to host for mhc post |
| mhc_fused.rs:501 sync, :502 `x.read` (pre) | mhc_pre.attn/.ffn | 90 | normed x to host for the mixer/FFN |
| moe_gpu.rs:643 sync (ffn_fast), :666 `P_Y.read` (ffn_host) | ffn | 45 | y to host; routing top-k is already on device (g5f_select) |
| mla_dev.rs:251 sync, :252 `y.read` | dsa/o_proj | 11 | y to host |
| mla.rs:2468 refresh_mirrors → debug_rows :2622-2625 (4 reads) | dsa/mirrors | 44 | host cache mirrors, only for re-upload/CPU fallback |
| mla.rs:2225-2226 `sd_out.read` (sparse only, kv_len > top_k) | dsa/idx_select | 0 (11 when sparse) | top-k selection to host, re-fed to attend |
| mhc_fused.rs:561-563 `h.read` (end) | head.mean | 1 | h to host for output_norm (CPU) |
| engine.rs:178-180 head sync + logits read (620 KB) | lm_head | 1 | argmax on host |
| **total** | | **226** | |

H2D per token: embed (host gather, 1), mhc post y (90), kda xb (34), ffn x (45), mla x (11),
head x (1). Tiny (5 layers) measured: 126 launches, 23 H2D, 27 D2H, 23 syncs; wall 2.66 ms vs
kernel busy 1.68 ms (nsys) — 37% host overhead even on 5 layers.

**Whole-step graph: not prototyped — every stage blocks capture**, and the fixes are in
files other workstreams own. What each needs (a `*_dev(s, x_ptr, y_ptr)` entry, no sync, no
host copy, on the ONE capturing stream or event-joined):
1. KDA (kda_dev.rs / kda_decode.rs, KDA owner): decode from a device x into a device y, no
   `s.sync`/`outb.read`; own stream `self.s` + `core.s` must join the step stream. Do NOT
   reuse the KDA_FUSED/KDA_GRAPH mixer graph: 4× slower on the B300 (kda 13 → 51 ms/token),
   cause unknown.
2. MoE (moe_gpu.rs, MoE owner): `ffn` without the trailing `s.sync()` (:643); pooled stream
   is per call (`pool.stream.take()`), needs the caller's stream.
3. mHC fused (mhc_fused.rs, unowned): `pre` leaving x on device (drop :501-502), `post`
   taking a device y ptr (:541 write), `end` leaving h on device.
4. DSA (mla_dev.rs / mla.rs, MLA owner): drop the y read + mirrors (make mirrors lazy, on
   fallback only); `kv_len`/`pos0` are kernel args → per-token graph update
   (cuGraphExecKernelNodeSetParams) or a device position counter; dense→sparse switch is a
   host branch (needs two graphs, switch at kv_len = top_k); sparse `sel` must stay on device.
5. Engine (mine): output_norm on device, device argmax + token feedback, embed gather on
   device, graph per (kv_len bucket) — waits on 1-4.
Unverified: H200 exact-timed trace (queued behind other agents' runs for >1 h at
/root/w-graph/trace1.log); ms/token with a step graph (not buildable yet).

## 2026-10-27 — GLM-5.3 server batcher: N sequence slots, one batched decode window (branch frog/glm5-batcher)

What was built:
- **Per-slot engine state.** `Glm5Engine` holds `n_slots` sequences: KDA conv + recurrent
  state `[slots][...]` on the device (`DevKda` h0/st, one allocation, slot = pointer
  offset; `decode_dev`/`prefill_dev`/`seed_at`/`read_back_at` take the slot) and per-slot
  host copies; the DSA caches per slot (host mirrors `dsa[slot][layer]`, the device
  caches as `MlaCuda` LRU entries keyed by slot, `set_slots` pre-creates them, `reset_seq`
  drops one); per-slot tokens (slot 0 = `tokens`, the single-sequence path, unchanged);
  `kda_dev_fresh[slot][layer]`. The residual streams and logits are per window (no state).
  `use_gpu_slots(N)` sizes it (4 MiB + 288 KiB per KDA layer per slot: 32 slots = 4.5 GiB).
- **`window_batch(rows)`**: ONE window over N decode rows, row i = one token of slot i.
  Embedding, mHC (fused), norms, the KDA projections / gates / o_proj (`window_rows`), the
  FFN (CPU MoE or the GPU `ffn_host` with T = N, naive or `TANG_GLM5_MOE=fast`) and the head
  run once for all rows; only the KDA core (`core_rows`: the decode step on each row's slot
  state, two launches per row) and the DSA attention (one row at a time on its slot's
  caches) split. Needs `TANG_GLM5_KDA_PF2=1`. `TANG_GLM5_KDA_GRAPH` captures slot 0 only.
- **Serve glue**: `max_slots()` = `TANG_GLM5_SLOTS` (default 8, or `--batch`), `decode_batch`
  = `window_batch` (`TANG_GLM5_BATCH_RR=1`: one window per row), `save/restore_running_at`
  carry the real KDA state (68 parts, read off the device) so a restore no longer replays
  the prefix; the DSA caches (positional) rebuild on the suffix prefill. Fixed on the way:
  the old glue's `verify`/`window` appended each token twice (the engine appends itself)
  — the DSA `pos0` check would have failed on the first real request.
- **CLI bench**: `flash-generate ... --slots N [--diverse] [--rr]`; `--diverse` changes one
  prompt token in slots 1.. (distinct states and expert routing; slot 0 keeps the oracle
  prompt, so its ids are the bleed check).

The cost that hid the batcher: with `TANG_GLM5_GPU_FFN=0` the CPU MoE's `matvec` was the
single-threaded scalar f64 reference — ~10 GMAC per token on one core, paid PER ROW, so
per-step wall grew linearly with N whatever the GPU side did. Now parallel over output rows
and over experts, with the weighted sums in the serial order: bit-exact (oracle unchanged).
Same for the DSA host GEMVs (`gemv` was parallel over tokens = 1 thread per decode row),
the host o_proj, and the attend wk/wv (2 × 25 MB copied per DSA row; now once per layer).
N=1 decode: H200 22.0 → 11.3 s/token, mew 18.1 → 5.8.

Correctness (prompt "Write a haiku about the sea.", greedy):
- single slot, GEMV8 PF2 MHC_FUSED (+ GPU_HEAD MLA_RESIDENT on the final tree):
  `785 1196 6801 264 6386 38038 911 279` = the CPU oracle, on the H200 (bins b7002a28,
  c93795d3) and mew (3641b874; final tree 56a774fc with every flag).
- batched, diverse: slot 0's ids = the oracle prefix in every run (8 and 32 slots; 2 of 7
  and 16 of 31 other slots decode differently, so the states really differ).
- tests: `engine_slots_batch_matches_separate` (CPU, two slots with different prompts:
  batched = round-robin = lone engines ≤ 1e-4), `engine_gpu_slots_batch_matches_single`
  (GPU, every FFN resident, naive and fast MoE). glm5 GPU suite on mew (TANG_REQUIRE_CUDA=1):
  **76 passed, 0 failed** (56a774fc).

Decode scaling, FINAL tree (every fast flag: GEMV8 KDA_PF2 MHC_FUSED GPU_HEAD
MLA_RESIDENT; GPU_FFN=0 KDA_MAXT=64; diverse prompts; H200 bin 0862c49c, 3090 bin
56a774fc; both build logs show `Compiling tang-llm`). Per-step wall = one token for every
slot; N=1 is the plain decode (`decode: 7 tokens in …`):

| box | N | per-step wall | ms/token/slot | aggregate tok/s | slot 0 ids |
|---|---|---|---|---|---|
| H200 | 1 | 11.4 s | 11 440 | 0.09 | oracle, 8/8 |
| H200 | 8 | 20.5 s | 2 563 | 0.41 | oracle prefix |
| H200 | 32 | 39.9 s | 1 248 | 0.74 | oracle prefix |
| 3090 | 1 | 5.5 s | 5 520 | 0.18 | oracle, 8/8 |
| 3090 | 8 | 14.3 s | 1 787 | 0.63 | oracle prefix |
| 3090 | 32 | 27.4 s | 855 | 1.04 | oracle prefix |

Window split at 32 rows (TANG_GLM5_TRACE, wall): H200 ffn (CPU MoE) 93.1%, DSA 4.7%, KDA
1.4%, head 0.6%; 3090 ffn 89.4%, DSA 7.6%, KDA 2.1%, head 0.8%. MLA_RESIDENT took the DSA
share from 9–15% (host q/o GEMVs) to 3–8%.

Earlier points, same engine path without GPU_HEAD / MLA_RESIDENT (H200 b7002a28, 3090
3641b874): H200 N=8 41.0 s, N=32 41.9 s (0.70 tok/s); 3090 N=8 15.1 s, N=32 30.1 s (0.96
tok/s), N=8 round-robin (one window per slot) 43.7 s — the batched window is 2.9× it.
Serial-MoE baseline: H200 N=8 same-prompt 86.0 s/step, 3090 N=8 diverse 75.5 s/step.

The GPU-batched stages barely grow with N; in this config the step is the CPU MoE (the
experts don't fit the H200's 141 GB). The number that matters needs every FFN on the GPU
(B300 / 2×H200, `TANG_GLM5_MOE=fast`): `window_batch` already hands it T = N rows. The
trace's GPU-ms column is unreliable here (negative / inflated event spans): wall only.

What breaks at high N:
- `TANG_GLM5_KDA_MAXT` (64 here) caps the rows of one window (KDA scratch, fused mHC):
  N > max_t falls back to per-row KDA / no fused mHC. Raise it for 64+ slots.
- The KDA core is two launches per row (`core_rows`): ~2·N·34 launches per step; a
  slot-indexed single-launch step is the next step. DSA is fully per row (host q_a/q_b/
  o_proj + one attend per row) — the largest non-FFN cost at 32 rows; a batched attend over
  rows with different caches is the fix.
- Memory: 145 MiB of KDA state per slot (4.5 GiB at 32) + the DSA caches (27.5 KiB/token
  per slot with the indexer); the CPU MoE holds one dequantised expert (~100 MB) per
  worker thread.
- `TANG_GLM5_KDA_GRAPH=1` (captured decode step) serves slot 0 only — other slots refuse.

Bugs found in frog/glm5-forward on the way (fixed here):
- `k5_beta`'s warp kernel (368714ff) passed a u64 to `g5_deq(const unsigned char*)`:
  NVRTC rejected the K5 source, so NO resident KDA layer could load (39d4100c).
- its launch used `grid(t*nh/4)` — `grid` divides by 128 again — so 1/128 of beta was
  computed and the rest of `bb` stayed stale: 158 top-1 flips GPU vs CPU (5a0baa26).
- the gemv8 A/B test asserted the max abs against the rel bound it printed (6c254855).
- non-cuda build: `GpuFfn` / `HeadDev` had lost their cuda cfg (merge fd0404c2).

Ops: two copies of a run script once interleaved on one status file and target dir (the
second launch while the first was mid-queue) — run scripts now take an instance lock
(`flock -n` on a run lock). `pkill -f <pattern>` over ssh matches the ssh's own shell and
other workers' jobs: kill by cwd/pid only.
## 2026-10-28 — MTP near-tie bisected: the KDA chunked core (t>1) vs the recurrent step (branch claude/glm5-mtp-bisect)
Tool: `TANG_GLM5_WINDIFF=S,T` (engine.rs `windiff`): step-decode to gen[S], snapshot, then
(A) T single steps, (A2) again, (B) one T-token window over the same tokens; per-layer taps
(attn in/out, ffn in/out, l2 rel per row) + logits top-2. H200 /root/w-bisect, fast env
(GEMV8 PF2 MHC_FUSED MOE=fast GPU_FFN_LAYERS=0-29 MLA_RESIDENT GPU_HEAD), haiku prompt.
- A vs A2: 0 everywhere (steps deterministic; snapshot/restore exact).
- Near-tie (S=59, feeding `42612 11 30957 11`, row 1 = the token after `…42612 11`): step
  30957 19.2940 vs 342 19.2044, **margin 0.0897**; window flips it: 342 19.4025 vs 30957
  19.2110. Row logits window-vs-step rel 1.3e-2, max abs 0.24.
- First difference: **l00 attn_out (KDA), row 0, rel 2.7e-7** with bit-identical input
  (attn_in 0). Grows through the stack: l01 attn_out ~1e-3, l02 2.5e-3, l06 ffn_out 3e-2 (one
  row jumps — MoE routing amplifies), l07–l10 up to 1e-1 on that row, settling ~5e-3–1e-2.
  (Table at S=56: l00 attn_out 2.6e-7, ffn_out 1.1e-5, l01 attn_out 6.3e-4, l04 3.8e-3, l06
  ffn_out 3.0e-2, l07 ffn_out 1.0e-1, l09 ffn_out 1.0e-1, l12 1.9e-2, l19 ~5e-3.)
- Cause: `kda_dev::core_pf2` runs t>1 through `prefill_dev` (conv_ip + PF2 chunked WY form),
  t=1 through `decode_dev` (fused conv + recurrent delta step). Same math, different fp
  order. Everything else (gemv8/xq8 projections, gates, MoE fast tile, fused mHC, resident
  MLA, GPU head) is already per-token bit-identical between t=4 and 4×t=1 — proven by the fix:
- Fix (flag): `TANG_GLM5_KDA_WIN_STEP=N` — windows with t ≤ N run the core as t
  `decode_dev` launches (token offsets into qkvb/gb_out/bb/ob). With N=4 at S=59: **every tap
  of all 45 layers and all 4 logit rows bit-identical (0.0)**.
- Verified: `--mtp 3 -n 64` with WIN_STEP=4 ids == plain decode, all 64 (ends `42612 11 30957
  11 342`); 1.75 tok/pass (23 verify + 13 rerun). Gate -n 8: plain and --mtp 3 both `785 1196
  6801 264 6386 38038 911 279`.
- Cost: t serial step launches per KDA layer per window instead of one chunk (tiny at t≤4;
  not timed). Default off. Unverified: prefill/long windows untouched (still chunked, so a
  prefill-vs-decode difference of the same ~1e-2 logit band remains by design); bound
  otherwise is logits ~1e-2 rel / ~0.3 abs, so near-ties with margin < ~0.3 can flip.

## 2026-10-27 (late) — MTP: the NextN layer drafts at 85–93% (branch claude/glm5-mtp)
GGUF facts (unsloth UD-Q4_K_XL, dumped on the H200): `nextn_predict_layers = 1`,
`block_count = 46`; `blk.45` is the NextN block — `nextn.eh_proj` [8192→4096] Q8_0,
`nextn.{enorm,hnorm,shared_head_norm}` F32, a full DSA attention (MLA + indexer tensors) and
the 288-expert MoE + shared expert. No `nextn.embed_tokens` / `shared_head.head` (uses
`token_embd` / `output`), no mHC tensors. `index_share_mtp = true`. Forward per llama.cpp
`graph_mtp` (/private/tmp/llamacpp-ref/glm5-next.cpp): `eh_proj @ [enorm(embed(x_{i+1})) ;
hnorm(h_i)]`, plain-residual DSA + MoE, `shared_head_norm`, `output`. `h_i` is the trunk's
**post-output_norm** hidden; a chain step's `h` is the previous MTP cell's post-shared_head_norm.
Code: `mtp.rs` (CPU drafter + greedy draft-and-verify, `--mtp K` / `TANG_GLM5_MTP=K`),
`mtp_tests.rs`; engine.rs hooks `last_h`, `snapshot`/`restore` (KDA host copies, read back
from resident layers first; DSA caches truncate), and the GPU head now runs every window row.
A partial accept rewinds and re-runs the accepted prefix (no per-token KDA checkpoints).
Results (H200, fast env: GEMV8 PF2 MHC_FUSED MOE=fast GPU_FFN_LAYERS=0-29 MLA_RESIDENT
GPU_HEAD, k=3, -n 128; depth d rate is conditional on depths < d accepted):
- gate "Write a haiku about the sea." -n 8 --mtp 3: ids `785 1196 6801 264 6386 38038 911 279`
  (= oracle), 2 verify passes, 100% accept.
- code: d1 88.4% d2 78.4% d3 62.1%; 1.91 tok/pass actual, 2.98 verify-only.
- prose: d1 84.8% d2 64.1% d3 75.0%; 1.75 actual, 2.78 verify-only.
- chat: d1 93.0% d2 70.0% d3 57.1%; 1.81 actual, 2.95 verify-only.
- haiku -n 64: 1.85 actual / 2.86 verify-only.
NOT identical at -n 64: MTP on/off agree for 60 tokens, then plain gives `30957 11 342`, MTP
`342 69898 11` (inside an accepted verify window). Not the head (same with the per-row GPU
head, logs2). It's a t=4-window vs t=1-step numerics difference in the trunk (KDA fused
decode / MoE fast / mHC t=1 paths vs window paths) at a near-tie; unbisected.
Tiny fixture (CPU): output == plain greedy for MTP / junk / oracle drafters, k=1..3; the
oracle is never rewound. Drafting is CPU (~2 s/cell at 4096: CPU MoE for blk.45).
GPU version needs: blk.45 resident (MLA via mla_dev + its MoE in GpuFfn + eh_proj gemv +
device head/argmax, no host round trip per cell); per-token KDA state checkpoints in the
window kernel (drop the re-run: 1.8 → 2.9 tok/pass); trunk window rows bitwise equal to t=1
decode (or bisect the near-tie flip first).

## 2026-10-27 — the absorbed + device-append MLA resident path verified (H200, r16)
The worker session died twice mid-iteration; the manager committed its dirty tree as
clearly-marked WIP (709f70a7) and a resumed worker verified it on the H200 (r16, tree
md5-matched to 68dc73a8, 1127 files):
- build rc 0; flash::glm5::mla 11/11; engine_gpu 5/5; gate + dbg + trace ids all = the
  oracle `785 1196 6801 264 6386 38038 911 279`.
- MLA shadow: 88 windows checked, 0 failed; worst rel 5.7e-4 (lat, tol 1e-3).
- Trace (TANG_GLM5_TRACE=1, -n 32, GPU_FFN=0): decode per token — dsa **6.00 ms** (was
  123 at 337d454c, 1,552 before MLA went resident), kda 13.05, mhc ~10.4, head 0.76,
  embed 0.32. The GPU-side decode budget is ~30 ms/token.
What the WIP added (now verified): the cache appends are device-side (kv_a / indexer
wk / pool gate as g5f_gemv straight into the resident rows, `mla_lat_norm_bf16` +
`mla_idx_ln` finishing in place; no host staging, no sync), pool keys stay on the
device, and the attend is the coalesced absorbed form (`mla_fold_q2` +
`mla_attn_abs2`); `o_proj` reads the attention output in place. Pre-abs2 the o_proj
sync waited on `mla_attn_naive` (r14 trace: dsa 68.35 ms/token).
Forward fast-forwarded to 68dc73a8.

## 2026-10-26 — MLA: both divergence signals were real bugs (3 of them); device-resident projections (branch frog/glm5-mla-divergence)

Method: a new MLA shadow (`mla_debug.rs`, `TANG_GLM5_MLA_DEBUG=1`, on tang-verify) runs
the CPU reference on the same inputs and the device's own rows for every DSA window. It
checks latents, indexer rows, mirrors, valid, pool and attention (using the GPU's own
selection), plus the shadow's own caches. It compares the selection as sets, with an f64
tie classifier. Every finding below was A/B'd against a planted copy (bug restored).

1. **`cuda_resident_flow_matches_upload` 1.148e-1 = a use-after-free, not the test.**
   On their upload branch, `attend`/`attend_absorbed` wrote through `DevBuf { ptr, len }`,
   which aliased the sequence's latent buffer. DevBuf frees on drop, so the next `from_slice`
   (q, sel) reused the memory, and the kernel read those bytes as latents. Only a fresh
   MlaCuda takes that branch: the test's "upload" reference. Planted: `resident vs CPU
   7.451e-9, upload vs CPU 1.148e-1`. Fixed: both 7.451e-9; all 4 flow checks 0.000e0.
2. **`engine_gpu_mla_matches_cpu` 167/170, 16.47 = the incremental pool path.**
   `idx_pool_keys` used `p = blockIdx.x` while the launcher offset the output by
   `pool_upto`. So every sparse window after the first stored pools `[0, n-from)` at
   slots `[from, n)`. Shadow: `pool` BAD w3 rel 1.6e-1, w4 3.5e-1; selection 7–11
   queries differ by f64 gaps 1.6e-2..7.7e-1 of max (real, not ties); every other tap
   ≤ 7.6e-7. The flow test's pool check shared the code path and couldn't see it.
   Regression `resident_pool_content_matches_a_fresh_pool`: planted 1.221e-1, fixed 0.
3. **Mirror read-back offsets**: `kv_append`/`idx_append` read the decode rows' host
   mirrors at element offsets (`DevBuf::read` takes bytes), so the mirrors held row pos0/2
   or pos0/4. This was latent until a re-upload (cache realloc, LRU re-entry). The flow
   test now asserts the decode-row mirrors (planted: false ×3).

Test bug: the flow test used `Stream::new()` (NON_BLOCKING; the stream-blocking.md
class). It is blocking now, and every MlaCuda launcher asserts `Stream::is_blocking()`.

After the fixes (H200, tiny f32): shadow 5/5 windows clean, 0/158 selections differ. The
remaining gap to the f32 CPU engine is the bf16 latent format, so the test now measures
against `Glm5Engine::bf16_latents`. GPU vs CPU-bf16: 0 flips / 170, logits max abs
1.982e-3. GPU vs CPU-f32: 1 flip, 15.22, which equals the CPU's own bf16-vs-f32 gap
(1 flip, 15.22; logit scale 45). The B300 8-token match fits: that run never hit an
incremental sparse pool (prompt < top-k = 2048, so the indexer is dense).

**Device-resident projections** (`TANG_GLM5_MLA_RESIDENT=1`, default off, `mla_dev.rs`):
q_a + `mla_rms_norm`, q_b, o_proj and the indexer's wq_b / weights_proj run on `g5f_gemv`
(GGUF types, q8x2 activations). Activations stay on the device from x to y.
- Tiny: 0 flips and logits 2.0e-5 vs the naive GPU path, shadow clean.
- Real model (H200, `TANG_GLM5_GPU_FFN=0 TANG_GLM5_KDA_MAXT=64`): ids
  `785 1196 6801 264 6386 38038 911 279` = the oracle, with and without the shadow.
  Shadow: `88 windows checked, 0 failed`; q_resid rel ≤ 1.6e-5, q ≤ 1.3e-5, out ≤ 2.1e-5.
  Selection was not exercised: the prompt is dense.
- mla mixer per step (sum of 11 DSA layers, `TANG_GLM5_PROF=1`, same binary
  ba50fdfb, both runs ids = oracle):

  | | naive (host GEMVs) | resident | |
  |---|---|---|---|
  | prefill (20 tok) | 6466.9 ms | 200.6 ms | 32× |
  | decode / step | 1440.6–1583.8 ms | 84.0–98.9 ms | ~16× |

  Resident decode grows ~2.5 ms/step with kv_len. Next: the remaining ~8 ms/layer
  (cache appends staged from the host, mirror read-backs, the pool host copy, syncs).

`test result` lines (H200): `flash::glm5::mla` 11 passed; `engine_gpu*` 4 passed;
`engine_reset_gpu` ok.

## 2026-10-26 (night) — fused KDA decode + CUDA graph: real-model ids gate passed

Branch `frog/glm5-cuda-graphs`, merged with the integration (`frog/glm5-forward`
c274b66a: PF2, fused mHC, tang-verify, gemv8 on o_proj + f_a/g_a). H200 (`/root/w-graphs`),
source md5-checked both sides, `Compiling tang-llm` in every build log.

### The gate (real GLM-5.3, "Write a haiku about the sea." -n 8 --temp 0)

Env for all four: `TANG_GLM5_GEMV8=1 TANG_GLM5_KDA_PF2=1 TANG_GLM5_GPU_FFN=0` (KDA, DSA,
mHC resident; FFN on the CPU — the full model OOMs the 141 GB H200 at FFN layer 32, and
this engine has no per-layer FFN knob; the all-GPU verdict was a 188 GB B300).

| run | ids |
|---|---|
| naive (control) | 785 1196 6801 264 6386 38038 911 279 |
| `TANG_GLM5_KDA_FUSED=1` | 785 1196 6801 264 6386 38038 911 279 |
| `TANG_GLM5_KDA_GRAPH=1` | 785 1196 6801 264 6386 38038 911 279 |
| fused + `KDA_DEBUG=1 WINDOWS=2 TOL=2e-2` | same; **68/68 windows clean**, worst rel 1.105e-2 (w1 l44 `out`) |

w1 is the first decode token, i.e. the fused step under the probe. The 1e-2 band is the
int8 gemv8 activations vs the CPU's f32, the same band as the naive gemv8 path. Tok/s
of these runs are CPU-FFN numbers (prefill 267 s), not speed.

### Two bugs the merge exposed (both found by `kda_decode_real_bench`, bit-equality assert)

1. **Mixed kernels per matrix.** The integration moved f_a/g_a onto `k5_gemv8` in the
   naive path; the fused `k5d_proj` still ran them on `g5_dot` (f32 dequant). The two round
   differently and the forget gate amplifies it through the state: 262143/262144 outputs
   and 1048481/1048576 state values differed, conv bit-equal. Fix: `k5d_proj` takes a SKIP
   mask; f_a/g_a go through `k5_gemv8` on both paths (the per-matrix Q8_0 rule).
2. **Stale int8 codes (my fix for 1, first version).** The f_a/g_a gemv8 launches went
   BEFORE the `k5_q8` pre-pass, so they read xq8 still holding the previous step's o_proj
   codes (yb, d_inner wide): 262144/262144 outputs differed. Fix: launch them after it.
   The xq8 question from the brief, answered: `k5_q8` is INSIDE the captured region, the
   graph re-quantises x every replay; the graph is keyed on the gemv8 flag.

Also: `decode_fast` moved onto the `tang_verify::Probe` API (poison list, taps, host conv
shift only under a probe); fused o_proj on gemv8 with its own yb quantise (finish()'s rule).
After both fixes the bench is **bit-identical**: fused and graph vs naive, 0 differing bits
in output, state and conv (64 steps, real layer 0).

### Per-layer bench (real layer 0, T=1, GEMV8=1, PF2=1, 56 steps timed)

| path | µs/step |
|---|---|
| naive | 1550 |
| fused | 1543 |
| graph | 1537 |

Fused stage GPU time: proj 1410, gate2 54, step 24, o_proj 38 µs. The old 6908 → 4035 µs
win (pre-gemv8, pre-PF2) is gone: PF2's device-resident decode already removed the host
round trips the fused step targeted, so the graph now buys ~1%. 92% of the layer is the
projection GEMVs — that, not launch overhead, is the next speed step.
FUSED/GRAPH stay default-off.

Ops: `cargo build` of the whole workspace fails on CUDA 12.8 (crates/tang pins cudarc 0.12);
build `-p tang-llm` only. Don't hold the GPU flock around builds — other sessions'
90-minute runs queue on it.

## 2026-10-26 — g5f_gemv: the fast MoE row kernel as a plain GEMV (branch frog/glm5-grouped-moe)

- `g5f_gemv` (moe_cuda.cuh) + `Kernels5::fast` (`FastGemv::q8` / `::gemv`, `Act`): any G5W type,
  int8 dp4a, two rows a warp, 8-token tile. Not wired into kda_dev (the manager does that);
  no env flag. Staging only when it fits 48 KB (o_proj's 70 KB staged was 35.8 µs vs 22.8).
- Parity: `gpu_gemv_fast_matches_cpu` 144 cases (tiny F32 + Q8_0, T 1/3/9, staged/not,
  q8x2/q8), NaN-filled outputs; q8x2 rel ≤ 4.2e-5, q8 ≤ 1.05e-2. The tiny q4_k_m GGUF is
  absent on the H200, so K-quant rows are covered only by the MoE tests (same dot code).
- H200, real l0 KDA projections (Q8_0, 35.7 MB each), CUDA events, kernel only, T=1:
  q/k/v 8192×4096: g5f-q8 18.2 µs (1.95 TB/s), g5f-q8x2 23.6, k5_gemv8 20.5;
  o 4096×8192: g5f-q8 15.5 (2.29 TB/s), q8x2 22.8, k5_gemv8 28.1.
  q+k+v+o: g5f-q8 70.4 µs, q8x2 93.6, k5_gemv8 89.4. T=8: 430 / 657 / 534 µs. At T=8 the cost
  is compute (each token re-decodes the row), not bytes.
- Gate (fast MoE, `TANG_GLM5_GPU_FFN_LAYERS=0-29`; 186 GB doesn't fit 141 GB with all 45):
  ids 785 1196 6801 264 6386 38038 911 279 = oracle; MoE shadow 240/240 within tol, worst rel 9.4e-5.
- Full `flash::glm5` suite, default and `TANG_GLM5_MOE=fast`: 62 passed, 2 failed, the same
  two in both: `engine_gpu_mla_matches_cpu` (tiny-fixture gate, top-1 167/170, logits 16.47;
  see "not a valid gate" below) and `cuda_resident_flow_matches_upload`. After three test
  shape fixes the latter now gets to its first numeric check and fails it: resident-flow vs
  upload attend maxerr 1.148e-1 (bar 1e-5). Open: an MLA resident-path issue, or the test.

## 2026-10-26 — tang-verify: the four ad-hoc tools are now a crate; the KDA shadow check ported onto it (branch frog/kernel-verify-harness)

[`tang-verify`](../../docs/kernel-verify.md) = shadow + NaN poison + first-divergence
attribution + skip_or_fail, as a reusable API (the same `Stat`/row format/env semantics as
the old `kda_debug`, so old logs diff cleanly). Core is dependency-free; feature `cuda`
adds the device side on tang-moe (`Probe::poison`/`read`, `kernel_test!` with
`TANG_REQUIRE_CUDA`). The attribution now ranks provenance ("these values are `k`'s,
written at offset 2048 in `qkv_pre` where 8192 was expected = an ELEMENT offset added to a
BYTE address") over pipeline order — it names the *k* writer for cc578187, where the old
rule blamed attn_q.

Evidence:
- Ported check, real model (mew 3090, KDA+DSA+mHC GPU / FFN CPU, `TANG_GLM5_KDA_MAXT=64`,
  prompt "Write a haiku about the sea.", `-n 8 --temp 0`): ids
  `785 1196 6801 264 6386 38038 911 279` = the oracle, and
  `kdadbg SUMMARY: 272 windows checked, 0 failed`, worst rel 2.8e-6 (34 KDA layers × 8).
- `kda_dev` kernel tests (`TANG_REQUIRE_CUDA=1`, mew 3090, 2/2): the q/k/v GEMVs vs the
  host matmul at three shapes, plus `k5_gemv_element_offset_bug_is_named`, which plants
  the bug in the test itself and asserts the verdict names `k`'s writer with the ×4 note.
- The planted-bug demo on the real model (H200 pod): `scripts/kernel-verify-demo.sh`
  reverts cc578187's byte scaling in a scratch copy, runs the model once planted and once
  fixed, and prints the verdict. Planted run: `qkv.k` nonfinite 81920 = 20×8192/2 and
  `qkv.v` 163840 = 20×8192 (exactly the offset arithmetic), verdict
  "`qkv.k` wrote into `qkv.q` … → suspect: k5_gemv attn_k (device)"; fixed run 34/34
  clean, `ids: 785 …`. The transcript is in docs/kernel-verify.md.

## 2026-10-26 (speed step 4) — fused, device-resident mHC: 170.8 → 10.5 ms per decode step (3090)

`TANG_GLM5_MHC_FUSED=1` (default off), `mhc_fused.rs`. The 4 residual streams stay on
the device for a whole window. Per half block: one kernel applies the previous half's
post in place and accumulates this half's 24 mix dots + flat sum of squares over 128-feature
chunks (`mhcf_part`, grid `(n/128, t)`), one kernel per token reduces them, runs
pre/post/comb/Sinkhorn, collapses and applies the block's weighted RMSNorm
(`mhcf_finish`). Host traffic per half: `y` up, `x` down (`t·n` f32 each); the window
uploads `r` once and reads back only the head mean. Was: 4 transfers each way incl. two
`t·4n`, a host repack and the host norm, and a one-warp-per-token mixes GEMV.

Why it was slow: the old `mhc_pre` kernel is one 32-thread block per token streaming
24 × 16384 weights — `mhc_cuda_real_shape` bench: 842 µs per pre at t=1 (891 µs at
t=8), post 102 µs. The fused unit bench: post+pre+norm 59 µs at t=1 vs `MhcGpu::pre`
843 µs alone.

mew 3090, real model, `TANG_GLM5_GPU_FFN=0 TANG_GLM5_KDA_MAXT=64`, same binary (md5
0bfa2f62…), `TANG_GLM5_PROF=1`, per-stage sums (decode = mean of the 7 steps):

| stage | off: prefill 20 tok | off: decode/step | fused: prefill | fused: decode/step |
|---|---|---|---|---|
| mhc_pre+norm(attn) | 154.3 ms | 73.2 ms | 41.7 ms | 4.6 ms |
| mhc_pre+norm(ffn) | 224.9 ms | 73.6 ms | 30.5 ms | 5.8 ms |
| mhc_post(attn) | 41.9 ms | 10.3 ms | 0.0 ms | 0.0 ms |
| mhc_post(ffn) | 55.3 ms | 13.7 ms | 4.5 ms | 0.1 ms |
| **mHC total** | 476.4 ms | 170.8 ms | 76.7 ms | 10.5 ms |

ids both runs = the oracle (785 1196 6801 264 6386 38038 911 279). End-to-end decode on
mew is unchanged (0.05 tok/s both: 7 tok in 129.36 vs 128.75 s) because the CPU FFN is
17.0 s of the 18.4 s step there; on the all-GPU B300 (1.9 s/step) the same ~160 ms
saving is ~8% — unmeasured.

Verification: fused + `TANG_GLM5_MHC_DEBUG=1` (new: every pre/post shadowed on the CPU
from the device's own `r`) + `TANG_GLM5_KDA_DEBUG=1`: `mhcdbg SUMMARY: 2888 checks, 0
failed; worst rel 2.984e-5`, `kdadbg SUMMARY: 272 windows checked, 0 failed`, ids = oracle.
Tiny: `engine_gpu_fused_mhc_matches_cpu` 0 flips, logits max abs 7.9e-5, 4 fused
windows; `fused_chain_matches_cpu` (2 layers × both blocks, distinct hc + norm weights
per block, asserts the wrong block is visibly wrong). Pre-existing, not from this:
`engine_gpu_mla_matches_cpu` (167/170, 16.47) and `mla::cuda_resident_flow_matches_upload`
fail identically in the gemv8 worker's logs.

Limits: single device only (`TANG_GPUS=2` keeps the per-call path); windows past
`max_t` and `TANG_DUMP_DIR` runs fall back per window.
## 2026-10-26 (speed step 1) — k5_gemv8: correct on the real model, 1.61× per KDA decode layer

k5_gemv8 (the dequant-reuse Q8_0 GEMV for the KDA q/k/v projections, c4925878) on the
H200 (pod uj2gjryncj9dfh), real model, KDA+DSA+mHC on the GPU, FFN on the CPU:

- Gate: `TANG_GLM5_GEMV8=1` → ids 785 1196 6801 264 6386 38038 911 279 = the CPU oracle;
  shadow check 68/68 windows clean at the int8 tolerance (TANG_GLM5_KDA_DEBUG_TOL=1e-2;
  worst rel 5.4e-3 — int8-activation noise, the fe_gemv8 band).
- Timing (`TANG_GLM5_PROF=1`, `prof kda mixer` per KDA layer):

| | naive | gemv8 | |
|---|---|---|---|
| decode step (t=1) | 12.15 ms | 7.56 ms | 1.61× |
| prefill window (t=20) | 298.9 ms | 274.6 ms | 1.09× (core on the CPU tail) |

Reading: q/k/v went from ~4.6 ms/layer to ~0 (107 MB of Q8_0 is ~20 µs at H200 SOL). The
remaining ~7.5 ms/layer of a decode window is everything else — host round trips (qkv/g/
beta read back every window), the naive o_proj / gate GEMVs / gated norm, and launch
overhead: the CUDA-graphs + device-resident workstream's target. End-to-end tok/s on this
box (0.03 both) is CPU-FFN-bound and not a kernel number.

Bring-up bugs, all caught before the real-model run by the harness or the in-process A/B
test: no dynamic smem (ILLEGAL_ADDRESS), misaligned u64 on 34-byte Q8_0 blocks, missing
__syncwarp after staging + overlapping per-warp regions, k5_q8 MAXT given t, and f32 rows
read as Q8_0 (needs the Q8_0-only guard).

## 2026-10-26 (housekeeping) — GPU tests can no longer silently skip; regressions for both fixture-blind bug classes

`skip_or_fail()` in engine.rs tests: with `TANG_REQUIRE_CUDA=1` a skip panics; without
it the skip prints a loud banner. `engine_gpu_matches_cpu` and `engine_gpu_mla_matches_cpu`
use it — the old silent `return` is how the GLM-5.3 GPU bugs survived a review cycle.

New regressions:
- `mhc_gpu_pre_block1_matches_cpu_block1` — the mHC block-index bug (324e7f6b). Checks
  the resident pre against the CPU for BOTH blocks and that the blocks differ.
- `mla_resident_is_per_layer` — the DSA geometry invariant behind 74192590 (11 DSA
  layers, first at 3), paired with the load-log residency line.

Tiny-test "flap" explained (the 118-flip episode): the failing runs at 13:19–13:24 used
a binary built before the mHC fix — deterministic failure, identical flip set — and the
passing run at 13:21 was a pre-relink artifact. With the current binary: 16/16 pass
(all-off 8/8, default 8/8), logits max abs 9.6e-5 stable. The test is trustworthy again;
the skip-guard is what keeps it honest.

## 2026-10-26 (final) — THE TREE VERDICT: all-GPU = the CPU oracle, 8/8 ids

Pod tree-verdict-b300-4 (0izjldh4haq9c1, B300 188 GB, EUR-IS-1), branch head 43f12709
content, source tar md5-verified pod-side (0a4f803d…; kda_dev.rs 0e414ba0… = local),
BUILD_RC=0, all-GPU, shadow off:

```
glm5: 11 DSA layers resident (MLA, one per layer), 45 mHC layers resident
glm5: 45 of 45 FFNs on the GPU
prompt: 20 tokens, prefill 22.91 s (0.9 tok/s)
ids: 785 1196 6801 264 6386 38038 911 279   ← the CPU oracle exactly, 8/8
decode: 7 tokens in 13.45 s (0.52 tok/s)
text: The user wants a haiku about the
```

With all four fixes in (KDA byte offsets, DSA per-layer MLA, mHC block index, teardown),
the wired GPU path matches the CPU engine on the real model. Every GPU-resident path —
KDA, DSA/MLA, mHC, FFN/MoE — has now been exercised at real weights in a matching run.

These tok/s are the NAIVE kernels (one thread per output GEMV, scalar core), i.e. a
correctness number, not the engine's speed. The speed work is the pre-existing kernel
plan; the verdict removes "is the GPU path even correct" from it.

Ops note: this pod's RunPod ssh proxy gives a PTY shell only from macOS OpenSSH 10.3
(command execution rejected: "Your SSH client doesn't support PTY"; pasted data garbles).
What worked: a `cloudflared` quick tunnel from the Mac to a local HTTP server, the pod
curls the 11 MB source tarball from the trycloudflare URL (md5-checked both sides).

## 2026-10-26 (later) — B300 all-GPU verdict: GPU mHC ran the wrong block's weights

B300 (pod 0izjldh4haq9c1, branch 43f12709 = main 89ac0bc9), load log: 34 KDA, "11 DSA
layers resident (MLA, one per layer), 45 mHC layers resident", 45/45 FFNs on the GPU.

| config | ids |
|---|---|
| all GPU | 4615 6540 315 279 … (wrong from token 1) |
| mHC on CPU, rest GPU | 785 1196 6801 264 6386 38038 911 279 = oracle |
| FFN on CPU, mHC GPU | 4615 6540 304 … (wrong) |

So GPU FFN/MoE is correct at real weights; GPU mHC was not. Speed of the correct config
(naive kernels, correctness verdict only): prefill 20 tok / 23.1 s (0.9 tok/s), decode
7 tok / 11.9 s (0.59 tok/s).

Cause: `Glm5Engine::mhc_pre` called `MhcGpu::pre(0, ..)` for BOTH halves of every layer
— block 0 is `hc_attn`, so every FFN-half mix ran with the attention half's hc weights.
The CPU path used the `w` it was handed, hence mHC-on-CPU matching. Every 3090 verify ran
mHC on the CPU, so none caught it. Fix: `mhc_pre(l, block, ..)` with 0 / 1 at the two
call sites. Verified on the 3090 (KDA + DSA + mHC on the GPU, FFN CPU): ids = oracle,
KDA shadow 34/34 clean. Remaining: one all-GPU B300 run (FFN on the GPU too).

## 2026-10-26 — GLM-5.3 staging: three GPU bugs, one bad test

Branch `frog/glm5-forward`. CPU oracle (mew, `cpu-dump7/8.log`): prompt "Write a haiku
about the sea.", `-n 8 --temp 0` → ids `785 1196 6801 264 6386 38038 911 279`.

### 1. KDA q/k/v offsets in elements, not bytes (cc578187) — THE staging bug

`kda_dev::window` launched the k/v projection GEMVs at `qkvb.ptr + off` with `off` =
`di`/`2·di` *floats*; a `CUdeviceptr` is a byte address. k landed `di/4`, v `di/2` floats
into the q block. Per token (di = 8192): q[0,2048) right, q[2048,8192) overwritten, half
of k and all of v never written. On a fresh (zeroed) buffer v read 0.0 → the KDA core
output was exactly 0 in every layer → garbage ids on the B300.

- Why the old dumps lied: they printed q[0..16] only — the one region nothing overwrote.
- Found by `TANG_GLM5_KDA_DEBUG=1` (shadow check + NaN poisoning, `kda_debug.rs`):
  first window, layer 0 — `qkv.k` nonfinite 81920 = 20 tok × 4096 (exactly half),
  `qkv.v` 163840 = 20 × 8192 (all), conv history 36864 = (4096 + 8192) × 3.
- Verified: real model on mew's 3090, KDA resident (FFN/MLA/mHC CPU): ids = the oracle
  exactly; shadow check 68/68 windows clean, worst rel 3.2e-6.
- Same commit: `DevKda`'s stream is now `Stream::blocking()` (it staged x/g/beta with the
  synchronous `DevBuf::write`, which a NON_BLOCKING stream doesn't order against — the
  hazard `tang_moe::gpu::Stream::blocking` documents). Correct, but not a cause here.

### 2. DSA/MLA: one `MlaCuda` per device served all 11 DSA layers

`GpuFfn::load` built `MlaCuda` from the first DSA layer's weights, once per device.
`MlaCuda` stages per-layer constants lazily and never replaces them (`kv_a`/`kv_a_norm`,
indexer `i_wk`/`i_gate`/`k_norm`, the resident `wk`/`wv` decompress matrices) and keeps
the latent + indexer caches **per sequence id, not per layer**, with watermarks
(`lat_rows`, `idx_rows`, `pool_upto`). So every DSA layer ran layer 3's K/V + indexer
weights, and at the next window the first DSA layer saw the *last* layer's watermark
(= its own pos0), skipped the re-upload, and attended over the last layer's latents.

- Invisible to the tiny fixture: it has one DSA layer.
- Reported by the interp research worker (docs/interp, engine.rs:138-144 / :455-462);
  confirmed in code (mla.rs SeqLru / `contiguous` checks).
- Fix: `mla_k` is per LAYER (`None` on KDA layers), each `MlaCuda::new_at` built from
  that layer's `mla_weights(l)`. Memory ~70–150 MB × 11.
- Verify (3090, real model, KDA + DSA on the GPU by default, FFN/mHC CPU, 74192590):
  ids = the oracle exactly; KDA shadow 34/34 windows clean. That run's log had no MLA
  residency line (MLA on is inferred from code: default-on, per-layer `new_at` errors
  would have failed the run). 4ac43280 adds `N DSA layers resident` to the load log so
  the B300 run proves it.

### 3. Teardown SIGSEGV at the end of every GPU run

`GpuFfn`'s fields drop in declaration order: `devs` (the CUDA contexts) first, then every
DevLayer stream, MLA buffer and module — compute-sanitizer: `cuModuleUnload`
INVALID_HANDLE, `cuStreamDestroy` CONTEXT_IS_DESTROYED. Fix: `impl Drop for GpuFfn`
clears device-owned members first, then drops each device's kernels before the device.
Verified: the GPU test exits 0 (was signal 11).

### The tiny-fixture GPU test is not a valid gate (open)

`engine_gpu_matches_cpu` on mew: fails with top-1 flips of margin up to ~20 logits even
with **every** GPU component off (`TANG_GLM5_GPU_KDA=0 MHC=0 FFN=0`, one rayon thread),
and flips pass ↔ fail over minutes with the same binary and inputs. Not GPU. Cause
unknown. Also: it `return`s (passes) when no GPU is found — the earlier "tiny fixture
passes" was almost certainly that silent skip. Don't use it as a gate until explained.
Note mew's `/private/tmp/glm5-tiny/glm5-tiny-f32.gguf` (md5 87ba…, Oct 24) differs from
the Mac's / mew's `~/tang-scratch/glm5-tiny` copy (73a8…).

### Process lessons

- md5 the source is not enough: `rsync -a` keeps old mtimes, and cargo skipped a changed
  file once (stale-binary incident #5). Sync with `rsync -rlp --checksum` (no `-t`) and
  refuse to run unless the build log shows `Compiling tang-llm`.
- Reproduce on mew's 3090 first (KDA + DSA fit at 24 GB with FFN/mHC on the CPU); every
  find above cost $0 there. The B300 is for the all-GPU verdict only.
- Poison scratch with NaN in any GPU debug run: unwritten memory reading 0.0 looks
  plausible; NaN can't hide.
