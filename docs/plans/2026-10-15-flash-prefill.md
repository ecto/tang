# Flash prefill on mew

Goal: cut Frog's time-to-first-token on Qwen3.8-Flash-Next (RTX 3090). Today a 10,373-token
find-answer turn 1 is 35.9 s cold, 8.2 s with the tools block cached; warm replay runs 393 tok/s
(~163 ms per 64-token window). Targets: cold TTFT ≤ 20 s, warm prefill ≥ 1,000 tok/s. Exactness
(`flash-spec-test`, `flash-tcheck`, reference KL band) must hold at every step.

Bench for every phase: `flash-serve-replay` on the recorded 10K frog prompt (cold = fresh server,
warm = second run) plus frog-bench find-answer turn-1 TTFT. Three runs each, under
`~/tang-gpu.lock`. Free the GPU first (other jobs held ~12 GB on 2026-10-15).

## Phase 0: seed the expert cache from real routing (1 day)

Cold prefill is bound by CPU expert misses streaming from DRAM (~30 GB/s). `Engine` already
seeds the VRAM cache from a routing-count profile (`engine.rs` `ranking`, flash CLI `--profile`),
but `serve` exposes no `--profile` and never saves counts, so every server start seeds
layer-uniform.

- `serve --profile <file>`: pass through to `opts.profile`.
- Save routing counts on shutdown and every N requests (`save_routing`), to
  `TANG_FLASH_STATE_DIR`; load it by default when present.
- Optionally add `--warm-request` with a recorded Frog system+tools prompt in `server.sh`, so
  the tools block is cached before the first request (it is already pinned to disk).
- Measure: cold TTFT and miss rate per window, uniform vs profile seed.
- Decision: if cold TTFT ≤ 20 s, phase 3 drops in priority.

## Phase 1: profile a warm wide window (1 day)

There's no breakdown of the ~163 ms per 64-token window yet. nsys one warm replay, then bucket:
dense GEMVs (native GGUF dp4a), GDN prefill, QSA + indexer, routed experts (GPU hits), CPU miss
wait, n-gram PLE reads, HC, graph/launch gaps. Write the table into `flash-handoff.md`.
Everything below gets reordered by this table.

## Phase 2: dense GEMMs on tensor cores (3–5 days, if dense is a top bucket)

At T=64 the dense projections are GEMMs, not GEMVs. `fl_natmma` (`TANG_FLASH_NATW=mma`) is
exact but slower than dp4a (187 vs ~100 µs at T=32).

- Find out why first: dequant layout, tile shape, or occupancy. Fix `fl_natmma` if it's local.
- Otherwise add a tiled path: dequant a weight tile to bf16 in shared memory, mma with bf16
  activations, f32 accumulate. Prefill only; decode stays dp4a.
- Exactness: prefill and decode must commit the same state. If the bf16 path changes bits,
  either keep it bitwise-equal to dp4a (int8 activations in the mma) or accept it under the
  KL band and re-run `flash-tcheck` with a prefill/decode split check.

## Phase 3: cold misses beyond seeding (only if phase 0 misses the target)

- Wider windows (128/256) so each missed expert blob is read once per more tokens.
- The full-sweep expert streaming ring from `hybrid-moe.md` M7: at ≥ 1K-token chunks, stream
  every expert of a layer through borrowed cache slots and run them on the GPU.
- Already measured and not worth repeating: per-layer PCIe share (`TANG_FLASH_PCIE_STAGE`).

## Phase 4: GDN and QSA prefill (if phase 1 says so)

Chunked-WY GDN prefill instead of the sequential in-kernel walk; batch the QSA indexer.

## Not in scope

MTP drafter work, images, decode-path changes.
