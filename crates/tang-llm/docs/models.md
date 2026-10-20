# Which open-weight models tang-llm can serve

Status: 2026-10-20. Configs were read from each model's Hugging Face `config.json`. GGUF layouts
come from the published files' headers (unsloth). Effort assumes one engineer who knows the
codebase.

## What tang-llm has today

| Path | Code | Runs | Formats |
|---|---|---|---|
| **Dense** | `model.rs`, `engine.rs`, `batch.rs` | Qwen3 dense (8B, 32B), Gemma; GQA, full RoPE, SwiGLU, one residual; CUDA, Metal, CPU | MLX 4-bit safetensors (`--q4`), bf16 |
| **Flash** | `src/flash/` (CUDA), tang-compute `flash*`, tang-moe | Qwen3.8-Flash-Next (`qwen4exp`) only: GDN + QSA (indexer-sparse gated attention) + 4-stream hyper-connections + PLE n-gram table + 512×top-10 MoE with shared expert, MTP drafts, verify windows in CUDA graphs, experts offloaded to a pinned host arena with CPU misses | GGUF: dense tensors in native types (Q2_0…Q8_0, K-quants, IQ4_XS/NL, bf16), experts **Q2_0 only** |
| **Flash reference** | `src/flash/reference.rs` (`tang-llm flash-ref`) | slow f32 CPU forward of any supported GGUF; the parity truth | every type `gguf.rs` dequantises |
| **tang-moe** | `crates/tang-moe` | expert arena, VRAM cache policy, AVX-VNNI/AVX2 Q2_0 CPU experts, GPU↔CPU doorbell | Q2_0 blobs |
| **Vision** | `vision.rs` | image encoder for the dense path | MLX safetensors |

Two facts shape everything below:

- **The Flash kernels are compiled for one geometry.** `tang_compute::flash::shape` and the
  `#define`s at the top of `kernels/flash_cuda.rs` fix HIDDEN 2560, GDN 16 k/48 v heads, QSA 24
  heads, 512 experts × FF 640 and HC 4. The engine refuses any other shape
  (`engine.rs`, `Engine::load`).
- **The dense path has no recurrent state, MoE or gated attention.** `Cache` is KV rows that you
  truncate.

## Support matrix

"Active" is parameters per token. A "†" marks an estimate from config shapes, not a published
number.

| Model | HF arch | Total / active | Mixer | FFN | Extras | Formats published | tang today | Missing | Effort |
|---|---|---|---|---|---|---|---|---|---|
| **Qwen3-Coder-Next** | `qwen3_next` (GGUF `qwen3next`) | 80B / 3B | 36 GDN (16 k, 32 v heads, grouped) + 12 gated attention (16 q × 256, 2 kv, rope 64 of 256, θ 5e6) | 512 experts top-10 (FF 512) + shared (512, sigmoid gate) | none (no MTP in the GGUFs) | MLX 4-bit (45 GB), GGUF Q2_K…Q8_0, UD-*, MXFP4_MOE, BF16; FP8 | **CPU reference: yes, as of this change** (HF parity below). GPU: no | Flash kernels for its geometry; plain residual + RMSNorm instead of HC; attention without an indexer; experts in Q4-class formats instead of Q2_0 | **1–2 weeks** to a GPU engine |
| **Qwen3-Next-80B-A3B** (Instruct / Thinking) | `qwen3_next` | 81B / 3B | same as Coder-Next (θ 1e7) | same | HF has MTP weights; GGUFs mostly drop them | same set + Qwen's own GGUF | same as Coder-Next: same code, different weights | same | **free** once Coder-Next is done |
| **Qwen3.8-27B** | `qwen3_5` (GGUF `qwen35`) | 28B dense | 48 GDN (16 k, 48 v, tiled order) + 16 gated attention (24 q, 4 kv), interleaved mrope | dense 17408 | MTP, 248k vocab, vision tower | MLX 4-bit, GGUF, FP8 | no | everything Coder-Next needs, plus a dense FFN in the Flash engine and the third geometry. The tiled v-head order and the swish gate already exist (Flash-Next) | ~1 week after Coder-Next |
| **GLM-5.3-Flash** | `glm5_next` | 321B / ~18B | 34 KDA layers (Kimi delta attention: per-channel decay, 64 heads × 128) + 11 DSA layers (MLA, q_lora 1536, kv_lora 512, no rope, pooled-key indexer: 32 heads, kpool 4, top 2048) | 3 dense, then 288 experts top-8, sigmoid + `noaux_tc` bias, routed scale 2.5, shared, SwiGLU limit 10 | mHC hyper-connections (4 streams, 20 Sinkhorn iterations), MTP | FP8 (HF), MLX 4-bit, GGUF (community) | no | KDA (GDN with a vector gate), MLA absorbed into the QSA-style kernel, Sinkhorn HC (our HC is the sigmoid kind), sigmoid router with bias, ~160 GB at Q4 → expert offload even on H200 | 4–6 weeks |
| **Mistral Small 4** | `mistral4` (in `Mistral3ForConditionalGeneration`) | 119B / 6.5B | 36 MLA layers (32 heads, q_lora 1024, kv_lora 256, rope 64 + nope 64, YaRN ×128, llama-4 attention scaling) | 128 experts top-4 (FF 2048) + shared, softmax | vision encoder | FP8 (HF), MLX 4-bit, GGUF (unsloth, bartowski) | no | MLA (latent KV cache + absorbed attention), YaRN, a MoE engine for this geometry, plus the vision tower for multimodal | 3–4 weeks text-only |
| **DeepSeek-V4.1-Flash** | `deepseek_v41` | ~760B stored (int8-packed experts) / ~13B† | 40 layers: 1 kv head × 512, rope 64, sliding window 128, per-layer compressed attention (`compress_ratios`), KV and indexer shared across layers (`kv_source_layer_ids`), o_lora groups | 384 experts top-6, `sqrtsoftplus` scoring, `noaux_tc`, scale 1.5, shared, SwiGLU limit | 4-stream HC (Sinkhorn), Engram n-gram tables (two layers, ~384M rows each), 3 MTP layers + DSpark drafter | FP8 + int8 (HF), GGUF (antirez, unsloth); no MLX 4-bit | no | almost all of it: cross-layer KV sharing, compressed attention, Engram (our PLE is close), new router; >300 GB even at 2–3 bit | 6–10 weeks |
| **gpt-oss-120b** | `gpt_oss` | 117B / 5.1B | 36 layers alternating sliding-window 128 / full GQA (64 q × 64, 8 kv), attention sinks, qkv/o biases, YaRN ×32 | 128 experts top-4 (FF 2880), softmax after top-k, clamped SwiGLU with α 1.702 and +1 on up | o200k vocab, harmony chat format | MXFP4 experts (native), GGUF (MXFP4), MLX 4-bit | no | sliding window + sinks in the dense attention kernel, biases, a MoE path (none outside Flash), an MXFP4 GEMV | 2–3 weeks; MoE on the dense path is shared work with Mistral 4 |
| **MiniMax M3** | `minimax_m3` | 427B / ~23B† | 60 GQA layers (64 q × 128, 4 kv, rope 64 of 128, per-head qk-norm, Gemma-style norms); layers 3+ use block-sparse attention (4 index heads, top 16 blocks of 128 + local block) | 3 dense layers, then 128 experts top-4 (FF 3072) + shared, sigmoid + bias, scale 2, `swigluoai` | 7 MTP modules | BF16 (HF), MXFP4/NVFP4 (AMD, NVIDIA), MLX 4-bit, GGUF | no | block-sparse indexer (close to QSA's), sigmoid router, ~230 GB at Q4 → offload | 3–5 weeks |

### What the gaps are made of

Most of the missing pieces fall into a few kinds of work, and each kind serves more than one
model:

- **Flash engine geometry as data** (Coder-Next, Qwen3-Next, Qwen3.8-27B; later Qwen4). The
  kernels are NVRTC source with `#define`s. Generating that prelude from a runtime shape, instead
  of Rust consts, opens the whole Qwen hybrid family. This is the one investment that pays off
  three times.
- **Plain residual and attention without an indexer** (all the Qwen ones). `hc_read` becomes
  RMSNorm, `hc_write` becomes an add, and QSA with selection off becomes ordinary gated GQA.
  The reference now does both (below).
- **Expert formats beyond Q2_0** (every MoE here). The Q2_0 expert blob, the grouped kernel and
  the CPU miss kernels assume 2-bit. The good small Coder-Next files are Q4_K/Q5_K
  (UD-Q4_K_XL: 49.6 GB) or MLX 4-bit (45 GB). Q4X already exists for dense weights. A Q4X
  expert blob plus a 4-bit grouped kernel covers Coder-Next, gpt-oss (requantised from MXFP4)
  and Mistral 4.
- **MLA** (GLM-5.3-Flash, Mistral 4, DeepSeek V4.1). A latent KV cache and absorbed
  projections. This is new attention work, and the biggest single gap after GDN.
- **Sinkhorn hyper-connections and sigmoid/bias routers** (GLM, DeepSeek, MiniMax). These are
  small next to MLA, but each is a parity hazard.

## Priority

1. **Qwen3-Coder-Next** (then Qwen3-Next for free). It is the product: 3B active is the
   smallest per-token read of any model here, so it gives the highest single-stream tok/s. At
   Q4 the whole model fits in one 80 GB H100 with ~30 GB left for KV and graphs, so it needs no
   offload. A 4-bit decode step reads about 2.1 GB: 0.9 GB of experts (10 routed + 1 shared × 48
   layers), 0.9 GB of GDN and attention projections, and 0.3 GB of router and head. At H100
   bandwidth (3.35 TB/s) that puts the speed-of-light bound near 1,600 tok/s before drafting. No other candidate is within 2× of that.
2. **gpt-oss-120b**: 5.1B active, native MXFP4, very popular. It needs MoE on a non-hybrid
   path.
3. **Qwen3.8-27B**: falls out of step 1's geometry work, but it is dense 27B, so it is slow per
   token.
4. **Mistral Small 4**, then **MiniMax M3** and **GLM-5.3-Flash**: these need MLA or offload.
   DeepSeek-V4.1-Flash comes last.

Side note: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF` is a coder-tuned Flash-Next.
It is the same `qwen4exp` architecture the Flash engine runs, but it is published only as IQ1_M,
which the engine can't read (experts are Q2_0 only). A Q2_0 requant of it, or IQ1_M expert
kernels, would give a coder model on the existing engine.

## Qwen3-Coder-Next: what landed and what's next

**Landed (CPU, no GPU needed):** `src/flash/reference.rs` reads GGUF architecture `qwen3next`:

- `Arch::Qwen3Next`: one residual stream, `attn_norm` and `post_attention_norm` pre-norms.
  The converter folds the zero-centred +1 into these, and into `output_norm`.
- GDN v head `h` reads k head `h / (hv / hk)`. This is grouped order; qwen4exp and qwen35 use
  tiled order, `h % hk`. Getting it wrong is silent and fatal: the tiny-model check below gives
  KL 31 with the tiled order.
- Fused `ssm_ba`: per k head, `[b × r | a × r]`.
- The GDN output norm is gated with SiLU (`ssm_norm` has no +1).
- Full-attention layers are QSA with no indexer: per-head `[q | gate]`, q/k RMSNorm, NeoX rope
  on 64 of 256 dims, sigmoid output gate.
- MoE: unchanged (softmax, top-k, renormalise, sigmoid-gated shared expert).
- `flash-ref --logits FILE` writes every output position's full logits.

**Parity:** `scripts/qwen3next_tiny.py` builds a random-init `Qwen3NextForCausalLM` (4 layers: 3
GDN + 1 attention; 2 k / 4 v heads; 8 experts top-2 + shared; non-zero zero-centred norms;
Qwen3-Coder-Next's tokenizer). It converts the model with llama.cpp's `convert_hf_to_gguf.py`
(f32), then compares HF fp32 with `flash-ref` on a 52-token code prompt.

| seed | HF entropy | max \|Δlogit\| | top-1 agreement | max KL | mean KL |
|---|---|---|---|---|---|
| 0 | 1.45 nats | 1.8e-4 | 100% | 2.7e-10 | 5.0e-11 |
| 1 | 1.44 nats | 1.2e-4 | 100% | 1.6e-10 | 3.0e-11 |

The bar is top-1 100% and max KL < 1e-3. With the wrong (tiled) v-head order the same check
fails: top-1 agreement 0%, max KL 31.

**Next: the GPU engine.** The order matters, because each step keeps Flash-Next bitwise
unchanged:

1. **Geometry as data.** A `Geometry` struct (from `Hparams`) replaces `flash::shape` consts.
   `flash_cuda.rs`, `native_cuda.rs`, the doorbell and the MoE plan get their `#define`
   prelude generated from it. For Flash-Next the generated source must match today's byte for
   byte, so the existing `*_vs_cpu` tests and `flash-tcheck` cover the refactor. Run it first, on
   its own PR, coordinated with the H200 kernel work.
2. **Residual mode.** `hc_read` / `hc_write` become an RMSNorm and an add (HC = 1). Add the head
   norm.
3. **Attention without an indexer.** Bypass QSA's selection (attend to every cell). The width
   cap `QSA_WIDTH` = 2051 goes away for this mode; use the split-K attention the MTP layer
   already has (`m_attn_part` / `m_attn_merge`).
4. **Experts.** Add a Q4X expert blob (gate|up|down + scales) and a grouped Q4X kernel. On an
   80 GB card all 24,576 experts are VRAM-resident, so there are no CPU misses: run with
   `ExpertMode` "all VRAM" and skip the doorbell. Requantise from the GGUF's Q4_K/Q5_K/Q8_0 at
   load. Bitwise CPU twin: the existing `cpu::flash` pattern.
5. **Exit:**
   - `flash-ref` vs engine KL inside the int8-activation band, as Flash-Next has today
     (0.002–0.007);
   - `flash-spec-test` and `flash-tcheck` pass, with the n-gram suffix drafter (no MTP in the
     GGUF);
   - decode tok/s on an H100 at 4K context, with % of speed of light.
6. **Real-weights check on the GPU pod** (it didn't fit on the dev Mac's free disk):
   `flash-ref` vs llama.cpp on UD-Q4_K_XL for a few prompts, bar KL ≤ 0.03.

The rented-GPU budget for steps 5–6: one H100 80 GB for ~3 hours.
