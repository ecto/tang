# tang-llm notes

Checkpoint log for GPU-path findings. Newest first. Each entry: what was wrong, the
evidence that pinned it, the fix, and how it was verified. Only things read back from
logs, not inferred.

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
