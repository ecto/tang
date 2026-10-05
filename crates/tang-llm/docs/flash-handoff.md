# Qwen3.8-Flash-Next on tang: handoff

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
- **Prefill**: ~350–370 tok/s with 8-token windows (default). 64-token windows reach 580 / 421
  tok/s at 2K / 10K in `flash-prefill-bench`, but a served frog turn got slower, so they're off
  (`TANG_FLASH_WIDE=64` to enable).
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
   acceptance (chat/on d1/d2/d3 = 77/49/31%), not GPU time. Data:
   `mew:~/mtp-train/` (README there; final 4-stream residual fp16 + ids per generated position,
   200 thinking-on prompts, 40 GB cap; `gen.sh` is resumable). Build a trainer on tang-ad/tang-train
   that fine-tunes the MTP layer recursively three steps deep (FastMTP, arXiv 2509.18362), write
   the weights back as a GGUF the engine loads, re-measure acceptance and the table.
2. **Why served wide prefill is slower than the bench.** Likely the chunk splitting at saved-state
   points and mixed window widths. Fix, then turn `TANG_FLASH_WIDE=64` back on.
3. **Tensor-core native GEMV for prefill** (`fl_natmma`, `TANG_FLASH_NATW=mma`) is exact but
   slower than dp4a (187 vs ~100 µs at T=32). Prefill ≥ 1,000 tok/s needs it plus a tiled bf16 path.
4. **More loads in flight per host expert group** when the GPU reads missed experts over PCIe
   (today ~4–5 GB/s effective, latency-bound; the link does 25 GB/s).
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
