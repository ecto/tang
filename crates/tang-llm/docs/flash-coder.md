# Flash-Next Coder on the Flash engine (2026-10-20)

ISTA-DASLab's [Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)
is Flash-Next (`qwen4exp`) with half its routed experts pruned for code (RCO, 256 of 512 per
layer, the same count in every layer) and published only as "IQ1_M". This note covers how it runs
on the Flash engine and what that costs.

## What the file is

- License apache-2.0 (inherits the base's), so a requantized copy may be redistributed.
- Same architecture and geometry as base Flash-Next except `expert_count = 256`: 48 layers,
  hidden 2560, 4 HC streams, top-10, same PLE / n-gram block, same tokenizer.
- "IQ1_M" is just the label. Experts are mixed per tensor: IQ3_XXS, IQ4_NL, IQ2_S, IQ3_S, IQ4_XS
  and Q2_0. Dense tensors are all types the engine already reads (Q4_K, Q5_K, Q6_K, IQ4_NL,
  IQ4_XS, Q8_0, BF16), and mostly at higher precision than base's.
- Shard 2 (the n-gram table) is byte-identical to base's (same sha256), so one download serves both.
- There's no MTP layer (base doesn't have one either). Drafts use unsloth's
  `mtp-Qwen3.8-Flash-Next-Q8_0.gguf`, trained against base.
- ISTA publishes no higher-precision Coder weights and no imatrix.

## Route

Requantize, not IQ kernels. The engine's experts are Q2_0 only. Serving IQ1_M's mix as-is would
take five new expert formats on GPU and CPU, plus the 256-expert geometry regardless.
[`scripts/flash_coder_requant.py`](../scripts/flash_coder_requant.py) rewrites shard 1 with every
`ffn_*_exps` tensor as Q2_0 and copies everything else byte for byte. Each 64-weight block's scale
comes from a grid search plus a least-squares refit. There's no imatrix, so the error is unweighted
MSE. Mean relative MSE is 0.149 over 135 tensors; the 9 tensors already in Q2_0 are copied. It
takes 41 minutes on 9 vCPUs, and shard 1 shrinks from 29.6 GB to 21.4 GB.

```bash
pip install numpy gguf
python3 -I scripts/flash_coder_requant.py \
  IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf \
  Qwen3.8-Flash-Next-Coder-Q2_0-00001-of-00002.gguf --jobs 9
ln -s IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf \
  Qwen3.8-Flash-Next-Coder-Q2_0-00002-of-00002.gguf
```

## Engine changes

**TEMPORARY: the 256 → 512 mask.** The engine's geometry is fixed at 512 experts. Until
ecto/tang#79 makes it a runtime value, a pruned model is padded:

- **Router:** `pack` adds zero rows up to 512, so the shared gate stays at row 512.
  `moe_route_into` top-k's only the first `n_expert` logits.
- **Expert file:** `experts.v2.bin` keeps keys `layer · 512 + expert` and leaves the padding as a
  hole, which takes no disk.
- **Memory:** VRAM slots are capped at the live keys, and the ranking skips padding.
  `ResidentCache::new_fill_masked` (tang-moe) maps every padding key to one shared zero host blob.
  On the A40 this loads as 12,288 VRAM slots (16.99 GB) with a 0 GB host arena. On a 24 GB card
  the live experts that don't fit go to the host arena as usual, and the padding costs one blob.

Once #79 lands, switch to a real 256-expert geometry and delete the mask (search `TEMPORARY`).

**Type generalizations (permanent):**

- `ple_key` may be bf16 (the Coder's) as well as Q2_0, uploaded as native bf16, which is
  width-invariant like the Q2_0 path.
- The MTP layer embeds from an IQ4_XS main embedding as well as Q3_K (`fe_embed_iq4xs`, the same
  `decode8` path).
- MTP `head2`, the pruned-vocab fast path, is used only with a Q5_K head. The Coder's Q6_K head
  takes the generic path.

## Results (A40 48 GB, RunPod)

The engine was tuned on an RTX 3090. The A40 has less memory bandwidth, and all experts are
VRAM-resident on it, so absolute tok/s here is not comparable to the 3090 numbers in
`flash-mtp-results.md`. Base and Coder were measured on the same pod.

Commands are run with the prompt "Write a Python function that parses an ISO 8601 duration…",
thinking off, 512 tokens, greedy.

| | base Flash-Next Q2_0 | Coder Q2_0 (this) |
|---|---:|---:|
| decode, no drafts | 57.6 tok/s | 54.6 tok/s |
| decode, MTP | 131.7 tok/s (3.55 tok/window) | 116.4 tok/s (3.41 tok/window) |
| MTP accept d1/d2/d3 | 93 / 89 / 89 % | 95 / 87 / 80 % |

- **Spec exactness:** `flash-spec-test` on the Coder passes. Drafts on match drafts off, greedy
  and at temp 0.8, for mtp, hybrid, suffix, forced-wrong and oracle drafts.
- **Parity:** `flash-ref` (f32 reference) vs the GPU engine on the Coder, over 170 code tokens:
  top-1 agrees 166/170, KL mean 0.0031, median 0.00053, p99 0.034, max 0.043. That's inside the
  repo's 0.0017–0.0067 band.

**HumanEval, all 164 problems:** greedy, thinking off, chat prompt, scored with
`scripts/humaneval_chat.py`. Both tang models run on `tang-llm serve`; the IQ1_M original runs on
llama-server (`-ngl 99`).

| model | pass@1 |
|---|---:|
| base Flash-Next GSQ-RCO Q2_0 (tang) | 152/164 = 92.7 % |
| Coder IQ1_M original (llama.cpp) | 155/164 = 94.5 % |
| Coder Q2_0 requant (tang) | 148/164 = 90.2 % |

The requant loses 9 problems the original solves (9, 59, 64, 91, 93, 109, 130, 139, 147) and
gains 2 (38, 97). The direct measure agrees: on the same 170 tokens, llama.cpp's IQ1_M original vs
`flash-ref` on the requant gives top-1 147/170, KL mean 0.19, median 0.024, p99 1.96
(`flash_llama_ref.py compare`, a lower bound on full-vocab KL). The llama.cpp↔`flash-ref`
difference on the same weights is 0.0015–0.007, so nearly all of the 0.19 is the requant.

llama.cpp runs the IQ1_M original at 57 tok/s on the A40 without drafts, about the same as tang
without drafts. tang's edge here is MTP: 116 tok/s, exact.

## Verdict

**Don't ship the Q2_0 requant.** It's exact on the engine and fast, but re-rounding experts from
~3.5 bpw IQ to plain Q2_0 without calibration loses 4.3 HumanEval points against the original,
and that puts it below base Flash-Next. The produced GGUF was kept local and not uploaded. Rebuild
it with the commands above.

What could close the gap, roughly cheapest first:

1. **Calibrated requant.** Run `llama-imatrix` on the IQ1_M original with code text, and weight
   the Q2_0 search by it (`flash_coder_requant.py` takes unweighted MSE today). This is likely to
   help but unlikely to reach the original: ISTA's base Q2_0 was GSQ-optimized from bf16, and we
   start from already-quantized weights.
2. **Serve the original's expert types (route b).** IQ3_XXS / IQ4_NL / IQ2_S / IQ3_S / IQ4_XS
   expert kernels on GPU (and CPU for misses), with per-layer expert formats. That gives exactly
   the original's quality at ~3.5 bpw. The expert blobs grow from 1.38 MB to ~2.2 MB, which on a
   24 GB card means fewer VRAM-resident experts.
3. Ask ISTA for the Coder's expert mask (it's in their allocation file) applied to their GSQ
   Q2_0 base experts. That is a Q2_0 Coder made the way base was.

Raw logs: the session notes folder `coder-flash-results/`.

