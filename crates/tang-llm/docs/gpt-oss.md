# gpt-oss on the dense path

gpt-oss-20b (21B total, 3.6B active) and gpt-oss-120b (117B / 5.1B) run on tang-llm's ordinary
decoder (`model.rs`), not the Flash engine: the same paged KV cache, batching, chunked prefill,
speculation and server as Qwen3 and Gemma. The checkpoints load as published
(`openai/gpt-oss-20b`, `openai/gpt-oss-120b`: MXFP4 experts, bf16 everything else), and so do
transformers' dense saves.

## What the architecture needs, and where it lives

| Piece | gpt-oss | Where |
| --- | --- | --- |
| Layer pattern | `layer_types`: sliding-window 128 and full attention alternating | `Config::window` (the Gemma window path) |
| Attention | GQA 64 q × 64, 8 kv (20b and 120b alike), Q/K/V/O biases | `Layer::bqkv`, `Layer::bo` |
| Sinks | one learned logit per head, a key with value 0 | `ComputeDevice::kv_attention_paged_sinks` |
| RoPE | YaRN: θ 150000, ×32, β 32/1, untruncated, from 4096 | `model::yarn` (any `rope_type: "yarn"` config) |
| Router | linear + bias, top-4, softmax over the 4 | `moe::Router::TopKSoftmax` |
| Experts | 32 (20b) / 128 (120b), FF 2880, gate/up interleaved, biases | `Mlp::Moe`, `ComputeDevice::moe_linear` |
| Activation | `(clamp(up, ±7) + 1) · g · σ(1.702 g)`, `g = min(gate, 7)` | `ComputeDevice::swiglu_clamped` |
| Expert weights | MXFP4: E2M1 nibbles, E8M0 scale per 32 | `moe::Mxfp4`, `moe_gemv_mxfp4` |
| Vocabulary | o200k_harmony (201088), untied head | the tokenizer.json as shipped |
| Chat | harmony: `<|start|>role<|channel|>c<|message|>…<|end|>` | `chat::Parser::harmony`, the checkpoint's own template |

## MoE on the dense path

`Mlp` is now `Dense { w_gate_up, w_down }` or `Moe(Moe)`. A MoE layer is four device calls,
each with a host reference in `tang_compute::moe` that the `ComputeDevice` defaults run (so
every backend is correct before it is fast) and the CUDA kernels are checked against:

1. `moe_route(logits, rows, experts, k, router)` → ids and weights `[rows, k]`.
2. `moe_linear(x, w, fmt, experts, bias, ids, k_top, k, n, per_pick)` for gate/up (every pick
   of a row reads the row) and for down (`per_pick`: each pick its own row).
3. `swiglu_clamped` (or the plain `swiglu_split`).
4. `moe_combine(parts, w, rows, k, h)`.

Nothing is fixed to gpt-oss's geometry (the Flash engine's MoE is `const`-shaped for its one
model). The parts that vary between MoE models are arguments:

- **Router** (`moe::Router`): `TopKSoftmax` (gpt-oss, Mixtral), `SoftmaxTopK { renorm }`
  (Qwen3-MoE and Qwen3-Next's `norm_topk_prob`). A sigmoid router with a bias for selection
  (DeepSeek V3-style `noaux_tc`, which Mistral Small 4 and GLM use) is one more variant.
- **Expert format** (`moe::ExpertFmt`): `F32`, `Bf16`, `Mxfp4`. A stack is `[experts, n, k]`,
  rows are outputs. MLX's affine 4-bit (group 64, the Qwen3-Coder-Next MLX checkpoints) is
  one more variant: the same kernel shape as `moe_gemv_mxfp4` with tang-Q4's
  `scale · q + bias` per group in place of the E2M1 table and E8M0 scale.
- **Activation**: `clamp: Option<(alpha, limit)>`; `None` is plain SwiGLU.
- **Shared expert**: not needed for gpt-oss. For Qwen3-Next / Mistral 4 it is a dense MLP
  added to `moe_combine`'s output (with Qwen's sigmoid gate), so it goes beside, not inside,
  these ops.

So Mistral Small 4 needs a router variant and (for its MLA attention, separate work) nothing
here; Qwen3-Coder-Next's 4-bit experts on the dense path need the Q4 format variant and the
shared expert. Loading is per model (`load_moe`), since checkpoints name and lay out experts
differently.

## The CUDA kernels (`kernels/moe_cuda.rs`)

Written beside the existing decode kernels rather than into them.

- `moe_route`: one warp per row, top-k by repeated warp argmax (ties to the lower index, as
  the reference), then the softmax.
- `moe_sort` groups a launch's picks by expert, and `moe_gemv_mxfp4` (grid `(N/8, experts)`)
  gives each warp one output row of one expert and applies it to every pick of that expert,
  8 at a time, so a weight row is read once per 8 picks. A lane takes 32-weight blocks: one
  16-byte load of nibbles, one scale byte. Each pick's dot is summed in the same order however
  many picks share its expert, so a row's numbers don't depend on its batch.
- `attn_sink_partial` / `attn_sink_combine`: split-KV attention grouped by KV head (one block
  holds a 32-key tile of K and V for its 8 query heads), then a combine that adds the sink to
  the softmax denominator. Each row splits its keys by its own span alone, so verify windows
  and batched rows match one-token decodes.

`TANG_MOE_HOST=route,linear,swiglu,combine,sinks` (or `all`) sends those ops to the host
reference, for bisecting a mismatch.

## Parity

`scripts/gpt_oss_tiny.py` builds a tiny random gpt-oss with the real layer pattern (window 8
so a 52-token prompt crosses it), real head dims, sinks, biases, YaRN as shipped, 8 experts
top-4, with experts drawn as MXFP4 codes and dequantized exactly into the dense weights HF
runs. It writes a dense and an MXFP4 checkpoint and compares tang (prefill and one token at a
time) to HF fp32 eager. Removing any one of the window, YaRN, the clamp or the sinks moves max
KL to ~2.

| checkpoint | device | path | top-1 | max KL |
| --- | --- | --- | ---: | ---: |
| tiny, dense and MXFP4 | CPU | prefill / step | 100% | 6.9e-7 / 5.9e-7 |
| tiny, MXFP4 | RTX 4090, f32 KV | prefill / step | 100% | 1.8e-5 / 7.3e-7 |

The real gpt-oss-20b against transformers fp32 on its own dequantized weights (every MXFP4
value is exact in fp32), on a harmony chat prompt plus 48 greedy tokens (138 positions),
RTX 4090:

| KV | TF32 (cuBLAS prefill) | path | top-1 | max KL | mean KL |
| --- | --- | --- | ---: | ---: | ---: |
| f32 | off | prefill | 100% | 3.5e-7 | ~0 |
| f32 | off | step | 100% | 4.4e-7 | ~0 |
| f32 | on (default) | prefill | 100% | 8.8e-2 | 1.0e-3 |
| f32 | on (default) | step | 100% | 4.4e-7 | ~0 |
| bf16 | off | step | 99.3% | 1.8e-1 | 3.2e-3 |

Two things are not gpt-oss's: prefill chunks go through the existing cuBLAS GEMMs, which run
TF32 (`NVIDIA_TF32_OVERRIDE=0` turns that off and prefill matches decode); and bf16 keys move
the sink attention enough to flip a near-tie, so gpt-oss keeps an f32 cache by default
(`TANG_KV_F32=0` for bf16). Its 8 KV heads make that cheap: 2% of decode speed, nothing in
prefill.

`tests/speculate.rs` on gpt-oss-20b (CUDA, RTX 4090): all three pass, every request byte-identical
with drafting off and on, alone and batched, staggered or not (the edit task accepts 158 of 193
short drafts, 236 of 283 default ones).

## Speed

Batch 1, greedy, 13-token prompt (`tang-llm generate`), and cold prefill (`bench-prefill`,
512-token chunks):

| model | GPU | decode | prefill 512 | 2048 | 8192 |
| --- | --- | ---: | ---: | ---: | ---: |
| gpt-oss-20b | RTX 4090 | 157 tok/s | 3132 tok/s | 2760 | 1873 |
| gpt-oss-20b | H100 SXM | 182 tok/s | 5399 tok/s | 4629 | 2864 |
| gpt-oss-120b | H100 SXM | 119 tok/s | 1418 tok/s | 1305 | |

A k-row forward (verify windows, batched decode) at 1024 context costs 1.05× one row at k = 4
and 1.2× at k = 8 on the 4090; 1.8× and 3.0× on the H100, where 8 tokens wake most of the 32
experts.

Where decode time goes (20b, 4090, nsys): expert GEMV ~45%, bf16 GEMVs (attention and the
201k-row head) ~35%, sink attention ~6%. On the H100 the step is launch-bound (~460 launches a
token without CUDA graphs).

## Harmony

The tokenizer has single tokens for `<|start|> <|end|> <|message|> <|channel|> <|constrain|>
<|call|> <|return|>`; `Tags` picks them up and `Parser::for_tags` switches to the harmony
state machine. A message's header decides its body: `to=functions.NAME` (before or after the
channel) is a tool call whose body is the JSON arguments; `analysis` is reasoning; `final`,
`commentary` (preambles) or no channel is text. `<|return|>` and `<|call|>` are stop tokens,
so a call's arguments are emitted when generation stops. Prompts come from the checkpoint's
own template (`chat_template.jinja`), which needed `strftime_now`; the render is byte-identical
to transformers' `apply_chat_template` with tools and a tool round trip.

## Not done yet

- Batched decode with sinks runs each sequence's attention separately (the fused
  `attention_paged_rows` has no sinks). `attn_sink_partial` already takes per-row positions and
  tables, so the fused path is a prep call plus this kernel.
- Sliding layers keep a full-length cache; a ring of 128 would cut KV memory nearly in half.
- Prefill MoE widens each expert to f32 once per chunk and runs cuBLAS; a grouped GEMM straight
  off MXFP4 would skip the scratch. Prefill attention with sinks isn't tiled over queries yet
  (it shows at 8k).
- Decode: CUDA graphs for the dense path, and a 4-bit head.
- `reasoning_tokens` in usage counts think-tag tokens only, so it reads 0 for harmony.
- 120b: same code. It needs ~65 GB, so an 80 GB GPU.
