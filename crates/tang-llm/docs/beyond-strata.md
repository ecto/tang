# Past Strata: ideas and research, ranked for mew

Status: research notes (2026-10-04). Companion to [hybrid-moe.md](hybrid-moe.md) (the plan) and
[strata.md](strata.md) (the baseline).

## Caveats

- The citations come from a literature sweep. Many 2026 items were read from the abstract or
  HTML only.
- Headline multipliers in this field are usually measured against weak baselines: naive
  offload, HF transformers, vLLM prefetch.
- Where a paper measured against a strong baseline (KTransformers, llama.cpp, EAGLE-3), its
  margin is quoted below. Treat every gain as a hypothesis until it is measured on mew.

## Where mew's time goes

The estimate is for Flash-Next Q2_0, 4K context, a verify window of ~3.2 tokens. The split is
estimated, not profiled: **the first job is an nsys profile of Strata on mew (M0).**

| Part | Time | Bound by |
|---|---|---|
| GPU dense half (3.5 GB of mixers, hyper-connections, routers, shared experts, head) | ~10 ms | launches, syncs and small GEMVs at ~35% of 936 GB/s |
| CPU experts (~100–180 MB of misses at ~93% hit) | ~3 ms | DDR5 bandwidth |
| Draft + commit + other | ~2 ms | host round trips |

Two things follow from that split:

- **The GPU half is the big lever on mew.** Strata's paper shows a ~50/50 split, but that was on
  a 12 GB card.
- **Ideas aimed at the CPU-miss budget overlap each other.** Expert deferral, prefetch and
  dropping misses all spend the same ~3 ms, so their gains don't add.

## Lossless: same output as the model, just faster

These are on by default in tang.

### 1. Persistent decode megakernel

The biggest single lever. Expected: GPU half ~10 → ~5–6 ms, so +30–60% decode.

- **Evidence:**
  - Hazy Research's "No Bubbles" megakernel reaches 78% of H100 bandwidth, against ~50% for
    vLLM/SGLang.
  - Cohere's decode megakernel (2026-09) on a 30B-A3B MoE at batch 1 runs at 62% of
    speed-of-light against vLLM's 39%, a 1.58× gain.
  - MPK/Mirage (2512.22219) is a compiler for the same thing, but shows no Ampere or
    quantized-weight support.
- **Design:**
  - One persistent kernel per verify window.
  - An instruction list: per-layer ops for every layer.
  - Weights prefetched into shared-memory pages one op ahead.
  - Global-memory counters instead of grid barriers.
  - The doorbell waits become in-kernel polls of the mapped flag that already exists.
  - Fused hyper-connection read and write. SGLang measured +7.6% and +5.5% end to end for those
    two fusions alone.
- **Rust angle:** tang already compiles CUDA C through NVRTC at runtime, so the kernel can be
  specialized to the exact shapes (2560, 640, 10, 48) with constants baked in.
- **Path:**
  - Get CUDA graphs and fusion first (plan M4), which banks most of the launch-gap win.
  - Build the megakernel once M6 works.

### 2. Cheaper dense bytes

- **Hyper-connection weights:** 1.3 GB of the 3.5 GB dense read is bf16 hyper-connection
  weights. Qwen's own design paper (2608.30320) stores the residual branches in fp8.
  - Try int8 or 4-bit for `hc_*` weights with a KL check. The 3090 has no fp8 tensor cores, but
    a GEMV only needs a dequant.
  - Paired with #1, this can drop ~0.6–1 GB per window.
- **IndexShare-MTP** (LMSYS SGLang blog, 2026-08-26): run the QSA indexer once per draft loop
  instead of per draft step. The selection from the last accepted row is reused, so the
  indexer stops being the long-context bottleneck under speculation.

### 3. Window length that prices expert cost

Expected +5–15%, low effort.

- **Rule:** extend the draft from k to k+1 tokens only if the marginal expected accepted tokens
  beats the marginal window time.
- **Window-time model:** `t(T) = f + (1 − f)·U(T)`, where U is the measured expert-union curve,
  with *non-resident* experts priced above resident ones.
- **Sources:**
  - "Limits of speculation for MoE" (2609.22156) shows that a single threshold is optimal.
  - EVICT (2605.00342) is 1.21× over EAGLE-3.
  - Cascade (2506.20675) and S2-MoE's admission rule (2608.15018) are the same idea.
- **On mew** the fixed part f is large, because the run is GPU-bound. Extra draft tokens
  therefore cost *less* than on Strata's reference box. Expect the controller to choose *longer*
  windows here, possibly 5–6 tokens.
- **Calibrate MTP probabilities first.** Raw softmax is overconfident.

### 4. Better drafts

1. **Philox-coupled MTP sampling.** At temperature > 0, the drafter picks its token with the
   target's own `Philox(seed, pos)` Gumbel noise. Output is unchanged, and acceptance tracks
   distribution closeness instead of greedy-vs-sampled.
   - Sources: 2408.07978; Strata's `STRATA_SPEC_COUPLED` is the same idea, off by default.
   - Free, and worth maybe +0.3–0.6 tokens/window for open-webui, which samples.
2. **Self-distil the MTP layer on our own traffic** (FastMTP 2509.18362, MTP-D 2603.23911).
   - Fine-tune the one MTP layer recursively, three steps deep, as it is run.
   - FastMTP on Qwen3-Next-80B: position-2 acceptance 0.48 → 0.62.
   - **This is where tang is uniquely placed.** tang-train and tang-ad exist, and mew is idle
     most of the night.
   - Log (hidden state, token) pairs from household and frog traffic, and distil the drafter to
     *our* distribution: code, the family's languages, frog's tool-call shapes. An optional
     router-agreement term (DraftExpert 2607.24434) makes the MTP's routing predict the
     verifier's, for prefetch (#6).
3. **Retrieval drafting with more corpora** (SuffixDecoding 2411.04975, AgSpec 2610.01108).
   - tang's global suffix store already exists. Add frog's open files and session trajectory as
     indexed corpora, plus per-context length caps.
   - Expected +20–50% on agent and code-edit loops, ~0 on chat.

### 5. Hide the miss path, losslessly

- **One-layer-ahead expert prefetch** (Speculating Experts 2603.19289).
  - Apply layer l+1's router to layer l's normalised residual (plus a default MoE vector) to
    predict the next layer's experts. ~90% per-layer accuracy on Qwen3, training-free.
  - Push the ~3–4 predicted misses per layer over PCIe during the ~0.2 ms of GPU layer time,
    so the GPU computes them instead of the CPU.
  - Pair with **Least-Stale eviction** (SpecMD 2602.03921) so prefetched experts aren't evicted
    by the same pass.
  - Expected +5–12%, medium effort.
- **Forecast-driven eviction** (SeqMoE 2609.12978).
  - A small sequence model forecasts routing across the stack, then evicts with "probabilistic
    Belady". 97% hit against 88.5% for the best baseline, at 45% of experts resident; the
    baselines included KTransformers and llama.cpp.
  - At our ~93% that might halve misses again. Do this later, after decayed-LFU plus prefetch
    are measured.
- **Non-uniform per-layer cache capacity** (MoE-CORE 2610.01950): give VRAM to layers by miss
  cost and routing entropy. Cheap.
- **Our own routing profile.** Seed the cache from `--dump-routing` over household traffic
  rather than a generic profile. Optionally keep per-conversation-key hot sets.

### 6. N-gram table off the SSD critical path

- **Timing:** the block-1 window at batch 1 is ~50–100 µs, about one NVMe 4K random read. Cold
  misses stall.
- **Read amplification:** a 4 KiB `O_DIRECT` read per ~90–160 B row is ~13× (CXL-Engram
  2603.10087).
- **Fixes, cheapest first:**
  - Pack rows by frequency, so one page carries several hot rows.
  - Grow the row cache from 0.1–0.3 GB to several GB.
  - Prefetch the n-gram rows of MTP *draft* tokens as soon as they're drafted. TF-Engram
    (2607.07388) does this on an RTX 5090 / i9 / NVMe box close to ours: 439 → 460 tok/s.
  - In prefill, issue all of a prompt's reads up front through `io_uring`.
- RAM is too tight on mew to hold the whole 27 GB table. A frequency-packed partial table is
  the realistic version.

### 7. Running state: fp16, denser checkpoints, on disk

- **Running state to fp16, never bf16 or int8.** DAMP (2608.27513) on Qwen3.6-35B AIME26: fp32
  85.5, fp16 84.6, bf16 79.7, uniform int8 18.5. LeapQuant (2609.38166) shows per-step int8
  collapse (87.9 → 7.1).
  - fp16 halves the 118 MB state traffic, and anchors become ~59 MB.
  - Strictly this is not bit-lossless. Gate it on the KL bar, like the KV format.
- **Anchors every 2–4K tokens instead of 16K,** spilled to disk.
  - ninfer-4090 restores 6.9K-token sessions in ~0.1 s from disk.
  - The zolotukhin.ai write-up and vLLM's hybrid cache manager recommend a separate state plane
    with ~4K checkpoints.
  - This fits the anchor design in hybrid-moe.md as-is.
- **Optional, approximate: Tail-Replay** (2608.30310). Reuse K/V at any token and rebuild GDN
  state by replaying only the last 5–10% of the matched prefix, since old inputs decay. 9–14×
  TTFT at 32K, 93–99.9% quality.
  - Opt-in for edits and branches that land between anchors.

### 8. Prefill

- **Chunked-WY GDN** (flash-linear-attention `chunk_gated_delta_rule`, sm86 OK; FlashQLA is
  sm90+ but a design reference).
  - 2–3× on GDN ops. But a small engine on GB10 saw only 4% end to end, because MoE dominated
    (dgpp PR #48).
  - **Profile GDN's share before building it.**
- **Long-context KV residency, if 64K+ matters:** the HiSparse (2608.07009) in-graph hit/LRU/H2D
  kernel, AVSG (2609.37538) lifetime aging instead of plain CLOCK, and cross-layer selection
  prefetch hints.

### 9. Same bytes, better quality: GSQ-style quantisation in tang

- ISTA's GSQ (2604.18556) learns grid assignment and scales jointly with Gumbel-softmax. It
  closes most of the gap to QTIP at 2–3 bits while staying a scalar format our kernels already
  run.
- A **learned 4-level grid** per tensor or per expert, in place of the fixed {−1,0,1,2}, costs
  one `pshufb` on CPU and one byte-permute on GPU ahead of the same dot. It stays
  bandwidth-bound.
- Better quality at 2.25 bpw buys room to push *cold, CPU-side* experts to ~1.75–2 bpw (DynaExq
  2511.15015, SliceMoE 2512.12990), which means fewer RAM bytes per miss.
- tang can do this calibration itself, layer by layer, on the 3090.

## Lossy knobs: opt-in, behind an eval

These change the model. Each one needs a household eval set first: frog task replays, a family
chat sample, a few reasoning sets, plus KL against the lossless engine.

| Knob | Source | Effect | Expected on mew |
|---|---|---|---|
| Top-k 10 → 5–8 with decoupled renorm (k2 = 10) | 2609.04575 (validated on Qwen3.5-397B, 512 experts, top-10: −0.55 MMLU at k=5); LDA 2609.09241 for k < 5 | halves expert bytes and the window union | +15–30%, trivial to try |
| Expert deferral: CPU-computed misses join the residual at layer k+2 | KTransformers SOSP'25 (+33–45%, ≤0.5% acc on DeepSeek-V3) | CPU miss time hides under the next layer's GPU work | +15–25%; overlaps lossless #5 |
| Drop or substitute *low-score* misses | AcceptMoE 2608.02989, BuddyMoE 2511.10054 | fewer CPU bytes | +8–15%; overlaps the above |
| REAP pruning of 20–25% of experts | 2510.13999 | hot set fits better | +3–8%, domain risk |
| Down-proj activation sparsity on CPU experts | TEAL / WINA 2505.19427 | ~15% fewer CPU bytes | +10% of CPU time |

Deferral, dropping misses and lossless prefetch all spend the same CPU budget. On mew, pick one
after profiling. Top-k reduction is the only lossy knob that also cuts the GPU half.

## Skipped, and why

- **Wide or dynamic trees** (EAGLE-2, Sequoia, OPT-Tree, SpecExec) and long-block drafters
  (DFlash 2602.06036): they fight the MoE union cost at batch 1.
  - If we ever want trees, TreeWY (2608.20961) and SpecLA (2607.16673) are the GDN-safe
    primitives. Our "verify reads, commit replays" already extends to trees, since every path
    walks from the same old state.
- **LUT CPU kernels** (T-MAC, bitnet TL2, Vec-LUT): Q2_0 with VNNI is already bandwidth-bound.
  Intel's AVX2/AVX-VNNI study (2508.06753) chose up-convert + VNNI over LUT. Borrow their
  interleaved layout to free cores.
- **Trellis or lattice formats for CPU experts** (IQ2_KT, QTIP, AQLM, QuIP#, Leech/Tetra):
  compute-bound on a 12900K. GPU-only if ever.
- **Self-speculation from the linear-attention subgraph:** acceptance 0.038 on sequential hybrids
  like Qwen3.5 (2605.01106).
- **Delta-compressed experts, AMX/AVX-512 kernels, and throughput-batching systems.**

## Order of work

1. **M0:** nsys profile of Strata on mew, to confirm the time split above.
2. **During M4 (graphs):** fused hyper-connection read/write, IndexShare-MTP, cost-priced window
   length.
3. **During M2/M5:** Philox-coupled drafting, one-layer-ahead prefetch + Least-Stale, own
   routing profile.
4. **After M6:**
   - megakernel;
   - hyper-connection weight quantisation;
   - n-gram prefetch from drafts plus frequency packing;
   - fp16 state + dense anchors.
5. **Background, nightly:** MTP self-distillation on household traffic. This is the
   tang-only edge.
6. **With an eval harness in place:** try top-k reduction first, as the one lossy knob that
   touches the GPU half.
