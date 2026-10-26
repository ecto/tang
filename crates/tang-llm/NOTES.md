# tang-llm notes

Checkpoint log for GPU-path findings. Newest first. Each entry: what was wrong, the
evidence that pinned it, the fix, and how it was verified. Only things read back from
logs, not inferred.

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
- Verify: real model, KDA + DSA resident on the 3090 — see the next entry when it lands.

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
