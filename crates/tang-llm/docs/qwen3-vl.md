# Qwen3-VL

Qwen3-VL-8B (dense) runs on the dense path with image input. Text and tools behave as they do on
Qwen3-8B; an image costs one ViT pass and then sits in the KV cache like text.

## What's implemented

- **Tower** (`src/qwen3vl.rs`, independent of the decoder): 16×16 patches (the temporal conv
  taps summed, since a still image is its frame twice), learned positions bilinearly resampled
  (align corners) from the 48×48 table to the image's grid, 2D axial RoPE (rows on the first
  quarter of the head, columns on the second), bidirectional attention, LayerNorm blocks with
  tanh GELU. Patches stay in 2×2 merge-block order throughout, so the merger is a reshape.
- **Merger**: LayerNorm per patch, then 4h → 4h → out with exact (erf) GELU.
  `ComputeDevice::gelu_erf` is new; its default runs on the host.
- **DeepStack**: blocks 8, 16, 24 (8B) each go through their own merger (LayerNorm over the
  merged 4h). The decoder adds feature `i` to the image tokens' hidden states after layer `i`.
- **M-RoPE** (interleaved, sections `[24, 20, 20]`): image token at merged (r, c) gets
  positions (t, t + r, t + c); text after the image continues at `t + max(rows, cols)`. A
  cache that holds an image owns RoPE rows (`Cache::mrope`, `[max_ctx, head_dim / 2]` cos and
  sin, written as positions are fed: image rows computed on the host, text rows copied from the
  model's table at `index + delta`). Attention kernels are unchanged: they take the cache's
  rows instead of the model's. Text-only caches don't allocate them. Decode steps that batch
  several one-token rows into one attention call skip that fusion for caches with their own
  rows.
- **Preprocessing**: `smart_resize` (multiples of 32, area within `preprocessor_config.json`'s
  `shortest_edge`/`longest_edge`; defaults 65,536 and 16,777,216 pixels), antialiased bicubic
  resize as torchvision does it (PIL's filter, a = −0.5), normalize to [−1, 1], patchify.
- **Prompt**: each image becomes `<|vision_start|>` + `<|image_pad|>` × tokens +
  `<|vision_end|>`, as the processor writes it. Content parts are joined without a separator
  next to an image (as the HF templates render image parts) and with `\n` between text parts.
  The template comes from `chat_template.json` when `tokenizer_config.json` has none.
- **Server**: `image_url` parts take data URLs or `https://` URLs. Fetching resolves the host
  once, refuses non-public addresses (loopback, private, link-local, CGNAT, ULA, metadata),
  pins the address, refuses redirects, and caps at 16 images, 20 MiB each, 15 s.
  `TANG_IMAGE_FETCH=0` accepts data URLs only.

The tower takes a weight prefix (`model.visual` in HF and mlx-community checkpoints) and its
own config, so the Qwen3.8-Flash-Next mmproj (`qwen3vl_merger`, 27 layers, 1152 wide,
DeepStack) can load the same `QwenVision` from GGUF tensors later. Only the loader would differ.

## Parity

`scripts/qwen3vl_tiny.py` builds a random-init `Qwen3VLForConditionalGeneration`: a 4-block
ViT with head dim 72 (as the 8B: 1152 / 16), an 8×8 position table (so every image resamples
it), DeepStack after blocks 0 and 2, and a 4-layer decoder with head dim 64 and sections
`[12, 10, 10]`. It uses Qwen3-VL-8B-Instruct's tokenizer, template and processor. The prompt
holds two non-square synthetic viewport renders (14×22 and 18×16 patches) and 175 tokens. HF
runs fp32 with eager attention. tang runs the CPU device in f32.

| check | seed 0 | seed 1 |
|---|---|---|
| tower, per block (embedding, 4 blocks, merger, 2 DeepStack), rel. max \|Δ\| | ≤ 1.6e-6 | ≤ 1.6e-6 |
| logits, one prefill (HF's patches), max KL / top-1 | 4.2e-10 / 100% | 5.2e-10 / 100% |
| prefill through image 1, then decode one token at a time to image 2 | 4.2e-10 / 100% | 5.2e-10 / 100% |
| prefill through both images, then decode the rest | 4.2e-10 / 100% | 5.2e-10 / 100% |
| for scale: HF itself without M-RoPE (1-D positions) | KL 5.4, top-1 53% | KL 4.2, top-1 66% |
| for scale: HF itself without DeepStack | KL 33, top-1 10% | KL 32, top-1 15% |
| tang's own preprocessing of the PNGs (not gated) | 6.6e-4 / 100% | 2.7e-3 / 99.4% |

HF entropy is ~1.1 nats (the head is scaled so the bar means something). Preprocessing gives
the same grids. Pixels differ by at most one u8 level (mean 2e-4 after normalizing), from
rounding ties in the resize. On a model this sharp, that moves the logits a little. The real
checkpoint's check below measures it on real weights.

Metal (`--device metal`, f32 weights, bf16 KV): tower ≤ 1.4e-6; logits max KL 5.9e-4, top-1
100%.

`tests/speculate.rs::images_drafts_and_batching_match` (`TANG_LLM_TEST_VL_MODEL=<dir>`) checks
that speculative decoding and batching give image prompts the exact bytes plain decoding gives
them alone. This covers one image, two images, and text-only on the VL model, with drafts
accepted and rejected, so the cache's M-RoPE rows are truncated and rewritten. It passes on
Metal with the tiny model, and the text-only `speculate.rs` tests still pass (Qwen3-0.6B).

```sh
python crates/tang-llm/scripts/qwen3vl_tiny.py --tang target/release/tang-llm --out /tmp/vl-tiny [--seed 1] [--device metal]
TANG_LLM_TEST_VL_MODEL=/tmp/vl-tiny/hf cargo test --release -p tang-llm --features metal --test speculate images_
```

Debug commands: `tang-llm vl-preprocess`, `vl-trace` (per-block tower states as JSON),
`vl-logits` (`--image` / `--patches`, `--step K` to decode after a prefix).

## GPU

The tower uses the dense path's existing kernels: GEMM, full attention (head dim 72),
LayerNorm, half-split RoPE and tanh GELU. The only new kernel is `gelu_erf` for the mergers
(CUDA `erff`, Metal via Abramowitz & Stegun 7.1.26). The image's M-RoPE rows go through the
same paged attention prep and attention kernels as text.

On the RunPod L40S (2026-10-21), `scripts/qwen3vl_pod.sh` runs every check below from a fresh
clone:

- **Tiny model, CUDA:** `tang-llm` turns TF32 on for cuBLAS by default on CUDA. With it on, the
  tower sits at ~1e-3 relative and the logits at max KL 8e-4 / 1.1e-3. The bar runs with
  `GAIA_TF32=0`.
- **`tests/speculate.rs` on CUDA:** all 4 pass (Qwen3-0.6B, plus image prompts on the tiny VL
  model). Drafting and batching are byte-exact.

## Real checkpoint (Qwen3-VL-8B-Instruct)

`scripts/qwen3vl_real.py` uses a 1280×720 synthetic viewport render (44×80 patches, 880 image
tokens) with a 917-token prompt. It compares the last 48 positions (the image's end and all
the text after it) against HF fp32 with eager attention. tang runs on CUDA (L40S).

| run | top-1 | KL mean | p99 | max |
|---|---|---|---|---|
| HF's own bf16 (for scale) | 81.2% | 0.86 | 13 | 14 |
| tang bf16 weights, bf16 KV, TF32 (production) | 91.7% | 0.022 | 0.37 | 0.51 |
| same, tang's own preprocessing of the PNG | 91.7% | 0.022 | 0.37 | 0.51 |
| same, prefill through the image, then decode one token at a time | 91.7% | 0.022 | 0.37 | 0.51 |
| tang f32 weights, TF32 off | 97.9% | 3.1e-3 | 0.051 | 0.084 |
| tang f32 weights and f32 KV, TF32 off | 100% | 1.5e-3 | 0.019 | 0.021 |
| text-only prompt, tang bf16 production | 100% | 1.0e-5 | 7.5e-5 | 9.1e-5 |
| text-only prompt, HF bf16 (for scale) | 100% | 1.1e-3 | 7.6e-3 | 9.2e-3 |

This image prompt is unusually sensitive to precision: HF in bf16 moves it 40× further from fp32
than tang in bf16 does. At f32 tang agrees on every position, and its production KL mean
(0.022) is inside the repo's real-checkpoint bar (≤ 0.03). The tower's per-block error against
HF fp32 (tang bf16) is ~2e-3 relative through block 25. It reaches 6e-2 at the last block and
7e-2 at the merger, where the activations carry large outliers; the DeepStack features are at
1.7e-3, 1.1e-2 and 1.7e-2. The 880-token image encodes in 1.2 s on the L40S, with the ViT
running once per image.
