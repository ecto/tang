# tang: the fastest way to run a model you own

Status: master plan (2026-10-04). Nothing here has landed.
Design detail lives in [hybrid-moe.md](hybrid-moe.md), the baseline we studied in
[strata.md](strata.md), and the research ranking in [beyond-strata.md](beyond-strata.md).

## The claim

**tang will be the fastest lossless single-user inference engine for the Qwen hybrid family on
one consumer GPU**, and it will prove it with numbers anyone can reproduce.

"Fastest" is scoped on purpose:

- **Single user, batch 1.** A person or an agent waiting on tokens. Not datacenter throughput.
- **One consumer GPU plus the PC around it.** The reference box is mew: RTX 3090 24 GB,
  i9-12900K, 64 GB DDR5-4800, NVMe.
- **Lossless by default.** Same output as the model at the same seed. Lossy speedups exist, but
  they are opt-in and never count on the scoreboard.
- **The family:** Qwen3.8-27B, Qwen3.6-35B-A3B, Qwen3-Next-80B-A3B, Qwen3.8-Flash-Next, and
  whatever Qwen4 ships on the same architecture.

## The scoreboard

Decode tok/s at 4K context on mew. Only the llama.cpp row is measured so far.

| Model | Engine | Decode | Status |
|---|---|---|---|
| Flash-Next (2-bit) | llama.cpp, `--n-cpu-moe 35` | 20–22 tok/s | **measured 2026-10-04** |
| Flash-Next (Q2_0) | Strata | 100–140 tok/s | Strata's own estimate for a 3090; measure in M0 |
| Flash-Next (Q2_0) | **tang, first light** | > 22 tok/s | target (M6) |
| Flash-Next (Q2_0) | **tang, parity** | = Strata on mew | target (M7) |
| Flash-Next (Q2_0) | **tang, lead** | ≥ 1.5× Strata on mew | target (M8) |
| Flash-Next (Q2_0) | **tang, stretch** | 300+ tok/s | target (M9) |
| Qwen3.8-27B (Q4) | ollama | unmeasured | measure in M0 |
| Qwen3.8-27B (Q4) | ninfer-4090 | ~149 tok/s with MTP, on a 4090 | their report, not reproduced |
| Qwen3.8-27B (Q4) | **tang** | ≥ 110 tok/s | target (M2) |

### Speed of light

Every tang benchmark reports **% of speed of light (SOL)**: bytes the model must read per step,
divided by the memory bandwidth of the device that reads them. It is the one number that says how
much is left.

- **Flash-Next on mew.** A verify window reads ~3.5 GB of dense weights on a 936 GB/s card:
  3.7 ms. At ~3.2 tokens per window, SOL is roughly **860 tok/s**.
- **Where engines sit today.** Strata's GPU half runs at ~35% of bandwidth. vLLM-class engines
  sit at 39–50%. Megakernel engines report 62–78% (Cohere, Hazy Research).
- **What the stretch target means.** 300 tok/s is ~35% of SOL end to end, including the CPU half
  and drafting. It is ambitious and physically reasonable.

These are estimates until M0's profile replaces them.

## Why tang can win

1. **mew is GPU-bound, and nobody has optimised for that.** Strata was tuned on a 12 GB card
   where CPU and GPU halves are equal. With 24 GB, ~93% of expert lookups hit VRAM and the dense
   GPU pass is ~10 of ~13 ms. That pass runs at a third of what the card can do.
2. **Runtime-specialised kernels.** tang compiles CUDA through NVRTC at load. Every shape (2560,
   640, top-10, 48 layers) can be a compile-time constant, and the whole decode step can be one
   persistent megakernel generated for this exact model.
3. **tang can train.** tang-ad and tang-train exist. The MTP drafter can be distilled every
   night on the household's own traffic, and quantisation can be learned (GSQ-style) in-house.
   No other local engine can improve its own drafts and its own quants.
4. **Conversation state as a first-class system.** Paged, content-addressed KV with a disk tier
   is already landing. Add recurrent-state anchors and a follow-up, a fork or a subagent starts
   in milliseconds. Strata keeps 6 checkpoints in RAM and nothing on disk.
5. **Small enough to change.** tang-llm is ~5k lines. Strata is ~95k. A new architecture is a
   week, not a quarter.

## Rules

- **Every milestone runs a real model end to end,** and the house moves onto it. Stopping after
  any milestone leaves something useful.
- **Lossless is the default.** Speculation is exact-match with `Philox(seed, position)`.
  `tests/speculate.rs` ("spec output == plain output") must pass on every architecture.
- **Parity before speed.** Each block lands with a CPU reference and a `*_vs_cpu` test, then
  logits against HF fp32 or llama.cpp at the existing KL bar.
- **No number without a method.** Commit bodies carry the measured result and how it was
  measured, as tang does today.
- **Profile, then build.** Anything estimated above gets measured before code is written
  against it.

## The ladder

Durations are rough and sequential; about 9–12 weeks to parity, then the lead.

### Act I — run the family (M0–M3)

**M0. Ground truth (days)**
- Strata on mew: decode at 4K and 32K, prefill at 32K, hit rate, nsys profile of the GPU half.
- ollama numbers for 27B and 35B-A3B on the same box.
- Golden logits from llama.cpp; tiny random-init checkpoints for per-block parity.
- `tang-llm bench`: one command that prints tok/s, tokens per window, hit rate, % SOL.
- **Exit:** the scoreboard above has measured rows instead of estimates.

**M1. Qwen3.8-27B (1–2 weeks)**
- Layer dispatch; Gated DeltaNet decode, prefill and commit kernels; gated attention.
- `RunningState` and anchors on the paged KV pool and the disk tier.
- "Verify reads, commit replays" rollback.
- **Exit:** logits match HF; decode ≥ ollama; open-webui's house model runs on tang.

**M2. MTP and verify windows (1 week)**
- MTP chain drafts, Philox-coupled sampling, a draft policy over MTP and suffix drafts.
- Window length priced by measured cost.
- **Exit:** ≥ 1.5× over M1; 27B at ≥ 110 tok/s.

**M3. Qwen3.6-35B-A3B on the GPU (1 week)**
- Router, grouped expert kernel, shared expert.
- **Exit:** logits parity; faster than ollama's `qwen3.6:35b-a3b`.

### Act II — off the card (M4–M6)

**M4. Graphs and fusion (days)**
- One captured CUDA graph per window size; no allocations on the token path.
- **Exit:** measured gain on 27B and 35B-A3B.

**M5. `tang-moe`: the offload runtime (2–3 weeks)**
- Pinned hugepage expert arena, VRAM expert cache seeded from our own routing profile,
  AVX-VNNI CPU experts, the mapped-memory doorbell.
- Proved on 35B-A3B with a forced-small cache, then Qwen3-Next-80B-A3B.
- **Exit:** 80B-A3B faster than llama.cpp `--n-cpu-moe` on mew.

**M6. Flash-Next, first light (2–3 weeks)**
- GGUF reader, Q2_0 experts, hyper-connections, sparse attention, n-gram table.
- **Exit:** KL ≤ 0.03 against llama.cpp; decode above the 22 tok/s floor on day one.

### Act III — take the lead (M7–M9)

**M7. Parity with Strata**
- Adaptive expert cache, in-graph PCIe share, prefill expert streaming, int8 KV.
- Fused hyper-connection kernels; indexer run once per draft loop.
- **Exit:** tang = Strata on mew, decode and prefill.

**M8. The megakernel**
- One persistent kernel per verify window, generated for the model's shapes.
- Hyper-connection weights in 8-bit or less, behind a KL check.
- One-layer-ahead expert prefetch so the GPU computes predicted misses itself.
- **Exit:** ≥ 1.5× Strata on mew; GPU half at ≥ 60% of bandwidth.

**M9. Everything else that is free**
- N-gram rows packed by frequency and prefetched from draft tokens.
- fp16 running state; anchors every 2–4K tokens on disk.
- Forecast-driven cache eviction; chunked-WY prefill if the profile says so.
- **Exit:** 300+ tok/s on Flash-Next; follow-up turns start in under 100 ms.

### Act IV — the edge nobody else has (continuous)

- **The nightly drafter.** Log (hidden state, token) pairs from household and frog traffic.
  Distil the MTP layer on them while mew is idle. Acceptance goes up every week on *our* text.
- **Our own quants.** Learned 4-level grids at the same 2.25 bits; colder experts pushed lower.
  Better quality per byte, or fewer bytes per miss.
- **Opt-in lossy mode,** behind a household eval set: top-k 10 → 5–8 with decoupled
  renormalisation first. Reported separately, never on the scoreboard.

## Proof

- **`BENCHMARKS.md` gets an LLM section,** regenerated by `tang-llm bench` on mew: every model,
  every engine, the method, the commit.
- **Each act ends with a write-up** on campedersen.com with the numbers and the profile.
- **Reproducible by others:** a pinned Strata and llama.cpp commit, the same files, the same
  prompts.

## Risks and kill criteria

- **Strata on mew is already near SOL.** Then the lead is small. M0 answers this in days; if its
  GPU half is above ~60% of bandwidth, Act III shrinks to the tang-only edges in Act IV.
- **Parity traps** (GDN head order, router precision, indexer pooling). Tiny-model tests against
  HF catch them before any big download.
- **Device spin waits on a stalled host thread hang the GPU.** Pinned and prefaulted arena,
  P-core pinning, a watchdog.
- **mew is shared** with frigate, sunshine and home assistant. ~40 GB RAM and ~23 GB VRAM are
  the real budgets, and `LimitMEMLOCK=infinity` is required.
- **Scope.** This is an inference engine. The ladder is the answer: each rung ships.

## First week

1. M0 in full: Strata and ollama measured on mew, nsys profile, `tang-llm bench`.
2. Tiny `qwen3_5` checkpoint and the GDN step kernel with its CPU reference.
3. 27B loading and producing HF-matching logits for one token.
