# Hybrid and offloaded MoE models: the plan

Status: design for the plan in [fastest.md](fastest.md). Nothing here has landed.

## Goal

Run Qwen3.8-Flash-Next well on mew, on code we own. The model is 125B MoE plus 51B n-gram
embedding, with 6B active. mew is an RTX 3090 24 GB, an i9-12900K and 64 GB DDR5-4800, of which
~40 GB is free for this.

Do it by building, one model at a time, the pieces every model in this family needs. Each
milestone runs a real model end to end.

[strata.md](strata.md) records how Strata does it. We copy the ideas that matter, not the code.
[beyond-strata.md](beyond-strata.md) ranks what research says we can do better, and where it
slots into these milestones.
Strata is C++/CUDA and about 95k lines without its SYCL port; tang-llm is about 5k.

## The family

Every model below shares the 248k Qwen vocab, partial rotary with interleaved mrope (θ 1e7), and
an MTP layer.

| Model | Mixers | FFN | Extra | Weights at Q4 | On mew |
|---|---|---|---|---|---|
| Qwen3.8-27B (house model today, via ollama) | 48 GDN + 16 gated attn (4 kv × 256) | dense 17408 | — | ~15 GB | all in VRAM |
| Qwen3.6-35B-A3B | 30 GDN + 10 gated attn (2 kv × 256) | 256 experts, top-8, + shared | — | ~20 GB | all in VRAM (tight), or offload test bed |
| Qwen3-Next-80B-A3B | 36 GDN + 12 gated attn | 512 experts, top-10, + shared | — | ~45 GB | experts offloaded |
| Qwen3.8-Flash-Next | 36 GDN + 12 QSA (sparse, indexer) | 512 experts, top-10, + shared | 4-stream hyper-connections, n-gram PLE | Q2_0: 34 GB experts + 3.5 GB dense + 29 GB table | experts offloaded, table on NVMe |

So the order is set by what each model adds, not by size:

1. **27B:** the hybrid mixer (GDN plus gated attention) and the recurrent state that breaks
   tang's "state is KV rows you truncate" assumption.
2. **MTP:** turns verify windows into the decode unit.
3. **35B-A3B:** MoE on the GPU.
4. **Offload** (35B-A3B with a forced-small cache, then 80B-A3B): the CPU+GPU expert runtime.
5. **Flash-Next:** adds QSA, hyper-connections, PLE and a GGUF loader.

## Targets on mew

- **Floor (measured 2026-10-04):** llama.cpp with UD-Q2_K_XL and `--n-cpu-moe 35` gives
  20–22 tok/s decode.
- **Bar:** Strata, once measured on mew (M0). Its own estimate for a 3090 is 100–140 tok/s.

Back-of-envelope for mew, Flash-Next Q2_0, 4K context:

- **VRAM.** 24 GB − 3.5 dense − 0.8 MTP − ~1 KV/state − 0.7 reserve ≈ 17 GB of slots. That is
  ~12k of the 24,576 experts. Strata's own curve gives a hit rate of ~0.93 at 24 GB.
- **CPU side.** Per verify window (T ≈ 3.2): ~3 × 480 routed × 0.07 miss ≈ 100 blobs × 1.38 MB
  ≈ 140 MB. At ~45 GB/s that is ~3 ms.
- **GPU side.** 3.5 GB of dense weights at 936 GB/s is 3.7 ms of pure bandwidth. Strata reaches
  about 35% of bandwidth on its GPU half (14.6 ms on a 672 GB/s card), so call it ~10 ms on a
  3090.
- **Window:** ~13–14 ms for ~3.2 tokens, a ceiling near 230 tok/s.

**The lesson for mew:** with 24 GB the GPU half dominates, unlike Strata's 12 GB reference
machine. Kernel efficiency, CUDA graphs and the per-layer host round trip matter more here than
CPU expert kernels. The CPU kernels still have to exist, but a plain AVX-VNNI kernel at RAM speed
is enough.

## What changes in tang

### Layers become a dispatch, and state gets a taxonomy

`model.rs` today has one dense `Layer` and an `is_gemma()` switch. It becomes:

```rust
enum Mixer { Attention(Attn), GatedAttention(Attn), DeltaNet(Gdn), SparseAttention(Qsa) }
enum Ffn { Dense(Mlp), Moe(Moe) }        // Moe: router, experts (handle into an ExpertStore), shared + gate
enum Residual { Plain, Hyper(Hc) }       // Hc: hc streams, per read/write weights
```

plus an optional `Ple` block before a given layer. Config parsing reads `layer_types`, `num_experts*`,
`hc_*`, `ple_*`, `indexer_*`, `output_gate_type` (`swish` on 27B, `sigmoid` on Flash-Next) and
`attn_output_gate`.

State splits the way Strata found it must (strata.md, "State, KV, conversations"):

- **Positional:** attention K/V, pooled indexer keys, MTP K/V. This is the paged block pool from
  [paged-kv.md](paged-kv.md), unchanged. Rewinding still only moves `len`.
- **Running:** GDN state + conv history, indexer tail, PLE history. This is a new
  `RunningState`, fixed-size per sequence. Sizes: 27B is 48 × 3 MiB ≈ 151 MB; 35B-A3B ≈ 63 MB;
  Flash-Next ≈ 118 MB.

`Cache` becomes `Seq { kv: block table, run: RunningState, anchors }`.

Running state can't be truncated, so everything that truncates today needs a new answer:

- **Speculative rollback: "verify reads, commit replays."**
  - The verify forward never writes `RunningState`. GDN layers stash their per-token inputs
    (conv input, q|k|v, gate, beta) in scratch, and the step kernel runs from the old state,
    emitting outputs only.
  - After acceptance, `commit(n_keep)` replays just the recurrence for the kept tokens, and sets
    conv history to the last 3 of `[history | kept inputs]`.
  - No snapshots. Replay is bitwise equal to what verify computed, so
    `tests/speculate.rs` "spec output == plain output" keeps holding.
- **Prefix reuse and paged blocks: anchors.**
  - A sealed block can't carry a 151 MB state; that is 600 KB/token against the 64 KB of K/V.
  - Instead, a sequence keeps **anchors**: `RunningState` snapshots at chosen positions. These
    are the last turn boundary, a long system prompt (pinned), and every 16K tokens, as Strata
    does.
  - An anchor is keyed by the block hash chain up to its position, so it rides the same
    content addressing.
  - `attach` takes the longest block-chain match that has an anchor at or before it, restores
    that anchor, and prefills from there.
  - Anchors live in RAM with an LRU budget, and can spill to the disk tier as `<hash>.run`. The
    disk tier is something Strata doesn't have, and it matters for frog's forked conversations.
- **Slots:** swap `Seq` as today; the running state comes along.

### New device ops

The plan is to add these to `ComputeDevice`, each with a CPU reference and a `*_vs_cpu` parity
test in the existing style. They are grouped by when they are needed.

**Recurrent mixer (M1):**
- `gdn_conv(x, hist, w) -> h` — conv plus SiLU plus q/k L2 norm. `hist` is read-only.
- `gdn_step(state, h, gate, beta, mode) -> o`. `mode` is either `ReadOnly { tokens }` or
  `Commit { n_keep: &Buffer }`, with `n_keep` on the device so it can be graph-captured.
- `gdn_prefill(state, ...)` — the sequential-in-kernel walk over a chunk, as Strata does.
  Chunked WY is a later optimisation.
- `gated_rmsnorm` (with the sigmoid or swish gate).
- Partial rope with mrope sections. For text it reduces to plain rope on the first 64 dims.

**MoE (M3):**
- `router_topk(logits, k) -> (ids, w)` — softmax over all experts, stable top-k, renorm with
  the 2^-14 clamp.
- `moe_grouped(x_q, plan, slots) -> y`. `plan` is device data: per-expert slot pointer, token
  list, destination rows. That way one captured graph serves any routing. Each weight row is
  read once per expert and applied to all its tokens.
- `moe_combine(parts, w, shared, gate)`.

**Flash-Next-specific (M6):**
- `hc_read`, `hc_write` (fused write-into-next-read later).
- `qsa_index`, `qsa_select`, `qsa_attend`.
- `ple_block`.

### Weights and formats

- **27B and 35B-A3B:** the HF bf16 safetensors through the existing `--q4` path. Nothing new.
- **Flash-Next:** a GGUF v3 reader (mmap, header-only, about 400 lines, like Strata's
  `gguf_reader.hpp`), and ISTA-DASLab's GSQ-RCO **Q2_0** file.
  - Q2_0 is the simplest format there is: 64 weights in 18 B, `(code − 1) · d`.
  - **Dense tensors** come in mixed types (IQ4_XS, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, bf16). Dequant
    them on the CPU at load and requant to tang Q4 or keep bf16. A dequant is a small loop per
    type, so no GPU kernels per type.
  - **Experts stay raw Q2_0**, repacked once into per-(layer, expert) blobs of
    `[gate+up | down | scales]`.
  - **MTP** isn't in the GGUF. It comes from the HF BF16 checkpoint's `mtp.*` tensors (~5 GB, one
    shard), or from unsloth's `MTP/mtp-*-Q8_0.gguf`.
- **I-quants** (IQ2_XS, IQ3_*) wait until after Flash-Next works. They are compute-bound on CPU
  and need codebook kernels on both sides. Q2_0 is RAM-bound with a VNNI kernel, and mew has
  AVX-VNNI.

### The offload runtime (new crate: `tang-moe`)

`ComputeDevice` is single-device and `!Sync`, so the hybrid executor sits beside it. It talks to
cudarc directly for mapped memory and graphs.

- **`ExpertStore`.**
  - Host arena: anonymous memory with `madvise(MADV_HUGEPAGE)`, registered with
    `cuMemHostRegister(PORTABLE | DEVICEMAP)` before first touch. If one range is refused, fall
    back to per-layer ranges.
  - In resident mode the arena holds only experts *not* in VRAM, ~17 GB for Flash-Next on mew.
    Never serve from a file mmap.
  - mew's memlock limit is 8 GB, so the service needs `LimitMEMLOCK=infinity`.
- **`ExpertCache`.**
  - VRAM slots, one per (layer, expert) blob, plus a dense residency table on host and device.
  - Sized last from free memory, then touched, re-measured and shrunk.
  - Seeded from a frequency profile. We record our own with a `--dump-routing` run over frog
    and open-webui traffic, which is a chance to beat a generic profile.
  - Adapts with decayed LFU: every 4 windows, ≤ 96 swaps, threshold 2, margin 1.5, decay 0.7.
    Copies go on a side stream, admitted when they land.
- **`CpuExperts`.**
  - One pinned worker per P-core; the E-cores are opt-in, measure first. The host thread also
    works. CAS job claim, spin then park.
  - Row-split two-phase execution across all missed experts.
  - Kernels in `std::arch`: AVX-VNNI (`vpdpbusd`, 256-bit) and AVX2 for Q2_0 and tang Q4,
    multi-token: decode once, dot against up to 8 tokens. AVX-512 behind detection.
- **`Doorbell`.** The per-layer protocol from strata.md: the GPU publishes ids and activations
  into mapped memory and bumps `seq`. The host publishes the GPU's plan first, then runs CPU
  experts. The GPU spin-waits on a mapped flag. CPU rows come back through mapped memory.
  - **Watchdog:** a device spin waiting on a stalled host thread hangs the GPU. The host polls
    `cuStreamQuery` every 2 ms, and the CPU side must never take a lock the CUDA driver can hold
    (Strata's issue #31).

### Decode becomes a verify window in a graph

- **Every decode step is a window** of T = 1..8 tokens. The suffix drafter and the MTP both
  propose, and a `DraftPolicy` (online ms-per-T cost and acceptance per match length) picks.
  tang's `draft.rs` already has the cost model half of this.
- **Acceptance stays exact match.** Sampling moves from sequential xorshift to
  `Philox(seed, position)`, so output depends only on the seed and position, whatever was
  drafted.
- **One captured graph per T** with no allocations on the token path. `begin_capture` and
  `end_capture` already exist in tang-compute (cuda.rs:397) and nothing uses them yet. This pays
  off for every model, dense ones included.

## Milestones

Each one is a branch or PR series with a measured result in the commit body, as tang does now.

**M0 — measure and reference (days)**
- Run Strata on mew in Docker (ISTA Q2_0, 66 GB) for the real bar: decode at 4K and 32K, prefill
  at 32K, hit rate, `STRATA_VERIFY_PROFILE` splits.
- Golden logits from llama.cpp for a fixed prompt set. These are the reference wherever HF fp32
  is too big to run.
  - Check that upstream llama.cpp reads type-42 Q2_0. Strata vendors a ggml that does, and
    upstream may not.
- Tiny random-init `qwen3_5`, `qwen3_5_moe` and `qwen4_exp` checkpoints, built with transformers
  5.8 dev (needs checking for `qwen4_exp`). These give per-block parity with `scripts/check_logits.py`.

**M1 — Qwen3.8-27B (1–2 weeks)**
- Layer dispatch and `RunningState`.
- GDN decode, prefill and commit kernels; gated attention with partial mrope; zero-centred norms.
- Anchors in slots, paged blocks and the disk tier.
- Rollback with the suffix drafter.
- Exit:
  - logits match HF fp32 at the existing KL bar;
  - `speculate.rs` passes on a hybrid;
  - decode at or above ollama's on the same 3090;
  - open-webui's house model moves from ollama to tang.

**M2 — MTP (about a week)**
- Load the MTP layer and draft a chain from the final hidden state.
- `DraftPolicy` over MTP and suffix drafts.
- Philox sampling.
- Exit: ≥ 1.5× decode on 27B; output identical with drafts forced wrong.

**M3 — Qwen3.6-35B-A3B on the GPU (about a week)**
- Router, grouped experts, shared expert gate, prefill grouping by expert.
- Exit: logits parity and tok/s against ollama's `qwen3.6:35b-a3b`.

**M4 — graphs (days, can move earlier)**
- One graph per T plus a commit graph; no allocations on the token path.
- Exit: measured decode gain on 27B and 35B-A3B.

**M5 — offload (2–3 weeks)**
- `tang-moe`: ExpertStore, ExpertCache, CpuExperts, Doorbell.
- Proving ground: 35B-A3B with the cache forced to N slots, so every number has an all-VRAM
  twin to compare against. Then Qwen3-Next-80B-A3B (Q4, ~45 GB).
- Exit:
  - cache on/off top-1 parity (Strata sees 2–5% near-tie flips with equal perplexity; match
    that);
  - hit rate against cache size curve;
  - 80B-A3B decode faster than llama.cpp `--n-cpu-moe` on mew.

**M6 — Flash-Next (2–3 weeks)**
- GGUF reader; Q2_0 GPU grouped kernel (two dp4a chains: `Σc·x − Σx`) and AVX-VNNI CPU kernel.
- Hyper-connections, QSA, PLE with `O_DIRECT` reads and a row cache.
- MTP from HF shard.
- Exit:
  - KL against llama.cpp on the same file ≤ 0.03 (Strata: 0.022);
  - decode beats the 21 tok/s floor on day one, then closes on the M0 bar.

**M7 — prefill and long context (ongoing)**
- Full-sweep expert streaming ring at ≥ 1K-token chunks, borrowing cache slots.
- int8 KV above 8K; KV streaming above 64K.
- I-quants if the quality per byte is worth it.

## Not doing (for now)

- Windows, WDDM workarounds, HIP, SYCL.
- Multi-GPU.
- Batched slots: one user at a time is mew's load, and Strata measured slots as a throughput
  loss on small cards.
- Q4 KV.
- Strata's "experimental speed projection". It is a refusal-direction vector, not an
  optimisation.

## Risks

- **GDN head order.** GGUF pairs v head h with k head h % 16; HF probably h / 3. Get it wrong
  and everything is subtly off. Test both orders against HF on the tiny model first.
- **Router sensitivity.** Small activation error flips top-k membership. Strata needed
  bf16-rounded router input and a double-precision softmax. Pin the router's precision contract
  before tuning anything.
- **Device spin plus host thread.** Any host hiccup (page fault in the arena, a lock, a
  preempted worker) stalls the GPU. The arena must be pinned and prefaulted, and workers pinned
  to P-cores.
- **RAM on mew.** ~19 GB is taken by frigate, home assistant, sunshine and the rest. Resident
  mode at ~17 GB fits; the full 34 GB arena doesn't, comfortably.
- **One 3090, shared.** sunshine and frigate's ffmpeg hold ~1 GB of VRAM and some of the GPU.
  Budget the reserve from measured free memory, never a constant.
- **Scope.** This is most of an inference engine. The ladder exists so that stopping after any
  milestone still leaves something useful running: the house model on tang (M1–M2), a fast MoE
  (M3), an offload runtime (M5).
