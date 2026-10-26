# tang-llm notes

Checkpoint log for GPU-path findings. Newest first. Each entry: what was wrong, the
evidence that pinned it, the fix, and how it was verified. Only things read back from
logs, not inferred.

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
