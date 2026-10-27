# DeepSeek-V4.1-Flash on tang: the port, milestone by milestone

Status: plan (2026-10-27). Nothing here has landed. The goal is frog cloud's "bull" model:
`cloud/bull` in frog, served by tang.

Shapes below are from the real `config.json` (`deepseek_v41_text`), not the estimates in
[models.md](models.md), which predates the GLM-5.3 work and is stale on two counts: tang now has
a Coder-Next GPU engine (`flash/coder.rs`) and most of GLM-5.3-Flash on the CPU reference
(`flash/glm5/`). Both cut the DeepSeek bill.

## The model, in tang's terms

| Piece | config | tang has | Gap |
|---|---|---|---|
| Residual | 4-stream HC, Sinkhorn, 20 iters, eps 1e-6 | `glm5/mhc.rs` (same shape) | none on CPU; GPU kernel is shared work with GLM |
| Attention | 40 layers, 64 q heads × 512, **1 kv head**, rope 64 of 512, q_lora 1280, o_lora 1024 in 8 groups, sliding window 128, YaRN ×16 over 64K | QSA indexer (Flash-Next), MLA latents (`glm5/mla.rs`) | the 512-wide single-head kernel, the window, o_lora groups, YaRN |
| Compressed attention | `compress_ratios` per layer: 0 / 2 / 1 (layers 0–1 none, 2–19 ratio 2, 20–39 ratio 1), own rope θ 160000 | nothing | **new**: a compressed KV per layer class |
| Cross-layer KV | KV written only at layers 2, 8, 14, 20 (`kv_source_layer_ids`); the rest read a source layer's cache | nothing | **new**: cache owned by a source layer, read by its clients |
| Indexer | 32 heads × 128, top 512 (`index_topk`), index written at layers 2, 8, 14, 20, 24, 28, 32, 36 | QSA's indexer and top-k | sharing across layers |
| Candidate blocks | from layer 20, block 8, top 2048 blocks (`candidate_*`) | glm5 pooled indexer (kpool 4, top 2048) is the closest | **new**: a block-level pre-filter in front of the token indexer |
| Router | 384 experts top-6, `sqrtsoftplus`, `noaux_tc`, scale 1.5, `norm_topk_prob` | `glm5/moe.rs` (sigmoid + noaux_tc, scale 2.5) | the scoring function; one more variant |
| Experts | FF 2304, SwiGLU limit 10, 1 shared, **FP4** weights | Q2_0, Q4X, MXFP4 GEMVs; SwiGLU limit (glm5) | an FP4 expert blob + grouped GEMV; confirm the FP4 format (block scales?) from `quantization_config` |
| Dense weights | FP8, 32×32 block scales, dynamic activations | FP8 safetensors loading (glm5) | block-scaled FP8 GEMV |
| Engram | 2 layers (1, 14), n-grams ≤ 4, 16M hashed vocab, 8 heads × 256, tables of 384,006,168 and 384,016,682 rows | PLE n-gram table with `O_DIRECT` reads and a row cache (Flash-Next) | the hashing, the two-table layout, and **sizing**: the rows alone are ~100 GB a table at bf16, so they live on host/NVMe like PLE does |
| MTP | 3 next-token layers | `flash/mtp.rs` chain (one layer, Qwen) | a 3-layer chain |
| DSpark | drafter: 128 experts top-3, markov rank 256, block 5, noise token 128799, reads target layers 37–39 | verify windows in graphs, `DraftPolicy` | **new**: a second small MoE model as the drafter |
| Vocab | 129,280 | — | tokenizer only |
| Vision | 32-layer encoder, ≤ 1024 image tokens | `qwen3vl.rs` pattern | not in scope for bull |

No `first_k_dense_replace` in the config, so assume all 40 layers route to experts until the
modelling code says otherwise. `compress_ratios` has 43 entries: 40 layers plus the 3 MTP layers.

### Bytes

- Experts: 40 × 384 × 3 × (5120 × 2304) ≈ 544B params. At FP4 ≈ **272 GB**; at 2-bit ≈ 136 GB.
- Active per token: 6 + 1 experts × 40 layers ≈ 9.9B expert params + dense ≈ **~13B** (so the
  "16B" in press copy counts something else; measure it in M0).
- Engram: two tables of ~384M rows. Row width is 8 × 256 at whatever dtype the shards use; read
  the shard index in M0 before believing any number.

The expert bytes decide the hardware. 272 GB does not fit one H200 (141 GB) or one B200
(192 GB) with KV and Engram cache. Either experts stream from host (tang-moe) or they shard
across GPUs. Streaming bounds decode near 6 × 3 × 11.8M × 0.5 B ≈ 35 MB a layer, ~1.4 GB a
token, so ≈ 25–40 tok/s on PCIe 5 before cache hits. The 800 tok/s class needs every expert in
VRAM: **4× B200/B300 or 8× H200, with expert parallelism tang does not have.** That is the
largest item in this plan and the reason the old 6–10 week figure was low.

## Milestones

Each one is a branch with a measured result in the commit body, as tang does now. Weeks are for
one engineer who knows the Flash and glm5 code; M1 and the kernels in M2 can run in parallel.

**M0 — truth (days)**
- Pull `config.json`, the modelling file, `quantization_config` and the safetensors index.
  Write down the FP4 format, the Engram row dtype, and whether any layer is dense.
- Tiny random-init `deepseek_v41` checkpoint via transformers, like `scripts/glm5next_tiny.py`.
  HF fp32 golden logits for a fixed prompt set, per-block dumps for `check_logits.py`.
- Pick the hardware target and price it: 4× B200 vs 8× H200 on RunPod.
- Exit: a sizing table with measured bytes; a tiny fixture under `tests/fixtures/deepseek/`.

**M1 — CPU f32 reference forward (2 weeks)**
- `flash/deepseek/` on the `glm5/` layout: `gguf`/`safetensors` geometry, `forward.rs`, one
  module per owner.
- Reuse: `mhc.rs` as is; `moe.rs` with a `sqrtsoftplus` scorer; glm5's FP8 loader; SwiGLU limit.
- New: the single-kv-head attention with window and o_lora groups; compressed KV with its own
  rope; cross-layer KV and index sharing; candidate blocks; Engram hashing and lookup against
  the real tables (mmap, no cache yet); YaRN.
- MTP layers and DSpark off. Vision off.
- Exit: tiny-model logits match HF fp32 at the existing KL bar, block by block; on the real
  model, the first two layers (no compression, no sharing) match HF on a 512-token prompt in RAM.

**M2 — one-GPU decode with experts offloaded (3 weeks)**
- Engine on the `coder.rs`/`glm5/engine.rs` pattern, CUDA first.
- Kernels: FP4 expert grouped GEMV (sibling of the Q2_0 and MXFP4 ones), block-scaled FP8
  GEMV for dense, the 512-wide attention with window + indexed sparse + compressed reads,
  Sinkhorn mHC on GPU (shared with GLM's speed plan), `sqrtsoftplus` + top-k.
- Engram through the PLE path: `O_DIRECT` row reads, row cache, prefetch off the n-gram of the
  tokens already known.
- Experts in a tang-moe pinned arena with the VRAM cache; proving ground is a forced small
  cache so every number has an all-VRAM twin, as hybrid-moe M5 did.
- Exit: parity with M1 (KL ≤ 0.03 on the prompt set, top-1 flips ≤ 2%); a first tok/s on one
  H200, expected 10–40; `tang-verify` shadow checks on every new kernel.

**M3 — expert parallelism (3–4 weeks; the new infrastructure)**
- Fill the `AllReduce` slot in tang-compute (`docs/extensions.md`) with NCCL through cudarc.
- EP: experts sharded by id across ranks; dispatch and combine as all-to-all; dense weights,
  attention and Engram replicated (they are small). One process, one thread per GPU, one
  CUDA graph per rank per T.
- Rank-0 owns sampling and the KV; the others mirror block ids (the design `ext.rs` already
  assumes).
- Exit: every expert in VRAM on the M0 box; T=1 decode ≥ 100 tok/s batch 1; parity unchanged.

**M4 — drafting (2 weeks)**
- 3-layer MTP chain in `mtp.rs`; `DraftPolicy` picks MTP, suffix or DSpark.
- DSpark: load its 128-expert MoE and Markov table, feed it target layers 37–39, block of 5,
  verify 6 a window. It is a second small engine; reuse the EP dispatch so its experts shard too.
- Exit: `speculate.rs` passes with drafts forced wrong (identical output); ≥ 3.5 tokens a
  window on the code prompt; decode ≥ 2.5× T=1.

**M5 — low-latency serve and bull (2 weeks)**
- CPU between steps: one graph launch per step, DP sync off the hot path, KV admission capped,
  as vLLM's DeepSeek post lists. Measure % of speed of light per step as tang does.
- `serve_glue.rs` for the OpenAI endpoint; `bull` in frog's `STATIC_CLOUD`
  (`crates/frog-engine/src/catalog.rs`), fallback as today.
- Exit: single-stream P90 tok/s published in `speed-log.md` with the SOL fraction;
  `frog -m cloud/bull` runs the frog-bench suite end to end.

**M6 — prefill and long context (ongoing)**
- Prefill kernels for compressed attention and the candidate pre-filter; bounded replay for
  cached prefixes; YaRN past 64K to the 1M the config declares; int8 KV above 8K.
- Exit: 128K prefill tok/s and decode at 128K within 10% of decode at 4K.

Sum: **~13–15 weeks** for one engineer, ~10 with two (one on the reference and formats, one on
kernels and EP from M0). Expert parallelism is a third of it and is the part that also unlocks
GLM-5.3-Flash at speed, so it is not DeepSeek-only work.

## Risks

- **The FP4 format.** If it is a DeepSeek-specific block layout rather than NVFP4/MXFP4, the
  GEMV is new and the requant path (`flash/requant.rs`) needs a target. Settle it in M0.
- **Compressed attention semantics** are only in the modelling code. Ratio 2 vs 1 vs 0 and how
  compressed and raw KV combine decide the kernel; read before designing.
- **Engram size.** Two ~100 GB tables on host means the row cache's hit rate sets prefill speed.
  PLE's numbers are for a far smaller table.
- **EP is new.** NCCL through cudarc, all-to-all in a graph, and multi-rank determinism are all
  first-time code in tang. Budget the slip here, not in the model work.
- **Hopper vs Blackwell.** FP4 tensor cores exist only on Blackwell; batch-1 GEMV dequant is
  fine on H200, but prefill and any batching want B200.
- **Press numbers.** 1500 tok/s is a vendor demo on a Tetris prompt with thinking off. The
  reproducible bar is vLLM's 826 P90 on 4× B300 TP4 with DSpark. Target that.
