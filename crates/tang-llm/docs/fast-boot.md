# Fast boot

Boot of the flash engines (Qwen3-Coder-Next / Flash-Next) is two phases: **weights**
(read the GGUF, requantize, upload to VRAM) and **kernels** (NVRTC-compile the CUDA C).
On the frog cloud node (H200, Qwen3-Coder-Next UD-Q4_K_XL) a cold boot measured
**weights 190 s, kernels 97 s, total 287 s**; some `ffn_*_shexp` weights are also
requantized to Q8_0 on every boot. Both phases are now cached on disk, and the boot
log reports each phase.

## What the caches are

| Cache | Env var | Contents | Key |
|---|---|---|---|
| kernels | `TANG_KERNEL_CACHE=<dir>` | each compiled module: its CUBIN (the arch-specific binary nvrtc emits for `sm_XX`), plus its PTX | source text hash + arch; the sources only change with the binary |
| weights | `TANG_WEIGHT_CACHE=<dir>` | one blob per model: every device-side weight buffer, in load order — post-requantization, post-permutation, the exact VRAM bytes | sha256 of the GGUF shards + tang version + arch + `max_ctx` + layout env vars + MTP sidecar |

Both default **off**: nothing is written unless the dir is named. `0` also disables
(`TANG_KERNEL_CACHE=0`). A cache error never fails a boot — it costs a recompile or a
reload, and a stale weight blob is rewritten by the cold load that superseded it.

## Kernels (`TANG_KERNEL_CACHE`)

`Gpu::module` (crates/tang-moe/src/gpu.rs) compiles each module once with
`--gpu-architecture=sm_XX` (a real arch, so nvrtc emits a **CUBIN**, not just PTX) and
loads the CUBIN directly — no driver PTX JIT even on a cold boot. With the cache on,
the CUBIN + PTX go to `<dir>/<arch>/k-<hash>.bin`, and a warm boot loads the CUBIN
with `cuModuleLoadData`, skipping NVRTC entirely. Every `gpu.module(...)` call site
(the Coder's three modules, the Flash-Next engine's, the sampler's, MTP's) shares it.

AOT alternative (not implemented): build-time cubins for sm_90 embedded in the binary.
NVRTC at runtime + this cache gets the same warm-boot cost with less build plumbing;
the cache dir on the pod's volume survives reboots, so only the first boot pays.

## Weights (`TANG_WEIGHT_CACHE`)

On a **cold** load (no blob for the key), the engine builds each device buffer exactly
as before (GGUF read, dequantize, requantize to Q8_0 for `ffn_*_shexp` and any other
type the expert kernels don't decode, the GDN `ssm_ba` row permutation, the Q8X repack)
and records the resulting bytes into `<dir>/weights-<key>.bin` (`WeightWriter`,
flash/bootcache.rs).

On a **warm** load, the blob is mmapped and each buffer is streamed straight into VRAM
with one large sequential read per buffer — no GGUF parsing, no requantization. The
per-buffer metadata (segment words, expert strides) is rebuilt from the GGUF header,
which is cheap.

The blob is keyed by the **content** of the model file (all shards hashed), so moving
or re-copying a model keeps its cache, and any changed file invalidates it. The
tang version, GPU arch, `max_ctx`, the layout env vars (`TANG_CODER_ROUTER16`,
`TANG_CODER_Q8X`) and the MTP sidecar's presence are hashed into the key too, so any
of those changes means a cold load, never a wrong-bytes boot. A blob whose buffer
lengths don't match the load anyway (a layout change the key missed, say a dev build
with the same version string) is deleted by the failing read and the load retried
once, cold, rewriting it.

### Bit-identity

`flash/bootcache.rs` holds the pure byte-layout functions the CUDA loaders use
(`dense_layout`/`dense_bytes`, `experts_layout`/`experts_bytes`, `f32_bytes`,
`quantize_q8_0`), so the cold path's bytes are testable without a GPU:

- `warm_blob_is_bit_identical_to_cold_load` builds a tiny GGUF (F32 and Q8_0 dense
  weights, a native and an F32 expert stack, a norm), records a cold load into a
  blob, then reads the blob back and compares every buffer byte for byte.
- `q8x_repack_is_lossless` and `permuted_rows` pin the two byte transforms.
- The blob tests (`blob_round_trip_and_invalidations`, `key_tracks_content_not_name`)
  cover the keying and the stale-format/truncation rejections.

On a GPU, the same proof end to end: boot twice with the dir set, and diff the load
reports — the second must say `warm`, and `tang-llm flash-ref`'s logits for the same
prompt must match across the two boots.

## Per-phase timings

`Coder::load`'s report line now reads:

```
coder: 48 layers (36 GDN, 12 attention), 512 experts top-10, hidden 2048, vocab 262k;
 78.2 GB VRAM (74.1 GB experts), weights 38 s (2212 buffers, warm boot cache),
 kernels 0.4 s, total 39 s
```

(`weights ... s (N buffers, warm|cold|off boot cache)` — the buffer count is the
blob's, cold and warm alike.) The Flash-Next engine's report (`load: warm (...)`)
already breaks out pack/dense/experts phases.

## Measuring on a pod

```sh
# once: make the dirs (a volume that survives pod restarts, not /tmp)
mkdir -p /workspace/tang-cache/{weights,kernels}

# cold boot: writes both caches (weights ~190 s + kernels ~97 s + blob write)
TANG_WEIGHT_CACHE=/workspace/tang-cache/weights \
TANG_KERNEL_CACHE=/workspace/tang-cache/kernels \
  target/release/tang-llm flash-serve <gguf> 2>&1 | grep -E "boot cache|coder:"

# warm boot: expect weights to drop to read+PCIe time, kernels to ~0
TANG_WEIGHT_CACHE=/workspace/tang-cache/weights \
TANG_KERNEL_CACHE=/workspace/tang-cache/kernels \
  target/release/tang-llm flash-serve <gguf> 2>&1 | grep -E "boot cache|coder:"

# force a cold weights load without losing the blob (A/B the phases)
TANG_WEIGHT_CACHE=0 TANG_KERNEL_CACHE=/workspace/tang-cache/kernels ...

# bits-identity on the GPU: same prompt, same logits, cold vs warm
tang-llm flash-ref <gguf> --logits <ids> > cold.jsonl
# (boot warm, then)
tang-llm flash-ref <gguf> --logits <ids> > warm.jsonl
diff cold.jsonl warm.jsonl
```

Per-phase detail on the cold path is on stderr (`coder: N of M layers in VRAM (t s)`
every 8 layers). To time the warm path's reads alone, watch
`boot cache: weights warm (<key>)` to the `coder:` report line.

## Sizes

The weight blob is the model's device bytes, ~1.0× the GGUF (the Q8_0 requantized
tensors grow to 8.5 bits/weight, the Q8X repack is lossless in size); the kernel
cache is a few MB. One H200 model (UD-Q4_K_XL, ~65 GB) needs ~65 GB of cache space;
plan the volume accordingly.
