# Qwen3.8-Flash-Next on tang: handoff

Takeover update: the merged-head GPU exactness checks have been rerun, and an offline recursive
MTP trainer is available. See [flash-mtp-results.md](flash-mtp-results.md) for measured results
and [flash-mtp-train.md](flash-mtp-train.md) for commands and checkpoint format.

Status as of 2026-10-05. Read with [fastest.md](fastest.md) (the plan), [strata.md](strata.md)
(block math), [flash-next-tensors.md](flash-next-tensors.md) (tensor inventory, parity studies,
corrections) and [beyond-strata.md](beyond-strata.md) (research ranking).

## Where it stands

Decode, mew (RTX 3090 24 GB, i9-12900K, 64 GB), 4K context, 512 tokens, greedy, exact:

| prompt | thinking | best tok/s | config | tokens/window |
|---|---|---|---|---|
| code | off | 232.4 | hybrid5 | 4.88 |
| chat | off | 198.8 | mtp3 | 3.07 |
| code | on | 173.0 | mtp3 | 2.76 |
| chat | on | 162.5 | mtp3 | 2.57 |

- No drafts: ~110 tok/s. llama.cpp on the same box: 20–22.
- **Exact**: `flash-spec-test` (drafts on == off, incl. forced-wrong drafts, greedy and sampled) and
  `flash-tcheck` (a token's bits are the same at any window width) pass.
- **Parity** vs the pure-Rust reference `flash-ref`: KL 0.0017–0.0067, inside the int8-activation
  band. The reference matches llama.cpp at KL 0.0015–0.007.
- **Prefill**: 64-token windows are the served default (`TANG_FLASH_WIDE=0` for 8-token windows).
  frog find-answer turn 1 (10,373 tokens): cold TTFT 35.9 s (8-token: 37.9 s), tools block
  cached 8.2 s (10.7 s). Cold prompts are bound by CPU misses (expert weights read at ~30 GB/s
  of DRAM) until the cache adapts; warm, `flash-serve-replay` runs 393 tok/s. The first wide
  windows after load take ~33 s once (absorbed by the startup capture).
- **Serving**: `tang-llm serve <gguf> --mtp <gguf>` speaks frog's OpenAI dialect: Qwen XML tool
  calls, thinking as `reasoning_content`, bitwise-exact prefix reuse (running + positional state
  snapshots, tools/system blocks pinned to disk), top-k/top-p/presence in the window graph.
  frog-bench find-answer, add-function and fix-rust-add pass.

## Architecture in one paragraph

Dense weights (native GGUF types, 3.9 GB) and the hottest ~12.9k of 24,576 routed experts live in
VRAM. The rest of the experts sit in a pinned, device-mapped host arena (`tang-moe`), and the CPU
computes those misses on the P-cores (AVX-VNNI Q2_0) through a mapped-memory doorbell, bitwise
equal to the GPU kernel. Every decode step is a verify window of T ≤ 8 tokens in one captured
CUDA graph. GDN state is never written during verify; a commit graph replays the kept tokens.
Drafts come from the model's MTP layer on a side stream, optionally mixed with a suffix drafter.
N-gram (PLE) rows are read from NVMe with O_DIRECT while layer 0 runs.

## Next steps, in order

1. **Distill the MTP drafter on thinking text.** The two thinking-on cells are limited by draft
   acceptance (chat/on d1/d2/d3 = 77/49/31%), not GPU time.
   Trainer: `flash-mtp-train` (`--features cuda,mtp-train`), documented above. Data:
   `mew:~/mtp-train/` (README there; final 4-stream residual fp16 + ids per generated position,
   200 thinking-on prompts, 40 GB cap; `gen.sh` is resumable). The first 50-update dense-cell
   pilot is saved at `~/flash-codex/pilot50/`; routed experts and the main embedding/head stayed
   frozen. Use held-out engine acceptance and throughput to select weights, then expand training
   beyond the initial short-window pilot as needed. Neither pilot checkpoint improves the
   deployed table overall; keep the original drafter (see the measured results above).
2. **Served wide prefill** (done): spans were never the problem; the wide CPU-miss path re-read
   every missed expert per 8-token slice. Fixed (one executor pass per window) and the cache now
   adapts every wide window. `flash-serve-replay` reproduces the server's prefill offline.
3. **Tensor-core native GEMV for prefill** (`fl_natmma`, `TANG_FLASH_NATW=mma`) is exact but
   slower than dp4a (187 vs ~100 µs at T=32). Prefill ≥ 1,000 tok/s needs it plus a tiled bf16 path.
4. **GPU reads of host experts** (measured, not adopted): staging the per-layer PCIe share into
   VRAM with a wide copy kernel (`TANG_FLASH_PCIE_STAGE=1`) is no faster than direct reads, and
   any share (8/16/32 per layer) loses to CPU-only misses both at T=8 and in wide windows.
5. MTP: fuse chain steps; eh_proj and HC GEMVs are ~0.3 ms each.

## Working on mew

- `ssh cam@192.168.2.12`. Build there, not on the Mac:
  `CARGO_TARGET_DIR=target4 cargo build --release -p tang-llm --no-default-features --features cuda`
  with `LD_LIBRARY_PATH=/usr/local/cuda/lib64`.
- Only one model fits on the GPU. Wrap every GPU run in `flock ~/tang-gpu.lock <cmd>`.
- Wait on done-files, never `pgrep -f name` loops over ssh (they match themselves).
- Weights: `~/models/qwen3.8-flash-next-ista/Q2_0/` (target), MTP
  `~/models/qwen3.8-flash-next/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf`. Pack caches in
  `~/.cache/tang/flash/` (`b091fed7…` is current `flash-pack-v2`; `39ddbfdb…` is the old v1).
- Reference dumps for per-layer diffs: `~/flash-truth/`. Engine logs and scripts:
  `~/flash-engine/`. Serving wrapper: `~/flash-serve/server.sh` (stops on `server.stop`).
- The old dense server `tang-llm.service` (user unit, port 8420, for kiln/frog) is stopped on
  purpose; frog uses the Flash server on the same port and key.

## Tools

`flash-ref`, `flash-parity`, `flash-tcheck`, `flash-spec-test`, `flash-resume-test`,
`flash-sampler-test`, `flash-bench`, `flash-prefill-bench`, `flash-generate`
(`--dump-mtp-train`), `flash-gemv-check`; `flash-kernel-bench` in tang-compute;
`tang-moe-bench` in tang-moe.
