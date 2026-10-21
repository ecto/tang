#!/usr/bin/env python3
"""Parity of tang's Qwen3-VL path (vision tower, DeepStack, M-RoPE) against Hugging Face fp32,
on a tiny random-init Qwen3VLForConditionalGeneration.

    pip install torch torchvision transformers safetensors numpy pillow
    qwen3vl_tiny.py --tang target/release/tang-llm --out /tmp/qwen3vl-tiny [--seed 0]

1. Build the model with the real structure at toy width: a 4-block ViT (head dim 72 like the
   8B's 1152/16, patch 16, merge 2, an 8x8 learned position grid so every image resamples it),
   DeepStack after blocks 0 and 2, a 4-layer Qwen3 decoder (head dim 64, interleaved M-RoPE
   sections [12, 10, 10]). Every weight random at a scale that keeps logits spread out.
   Tokenizer, chat template and processor: Qwen3-VL-8B-Instruct's.
2. Two synthetic viewport renders (non-square, different sizes) in one chat prompt, through the
   HF processor. HF runs fp32, eager attention, one prefill.
3. tang, CPU, f32:
   - preprocessing: `vl-preprocess` on the PNGs vs HF's pixel_values;
   - per block: `vl-trace` on HF's own patches, the hidden state after the patch embedding and
     after each block, the merged embeddings and the DeepStack features;
   - logits: `vl-logits` on HF's patches as one prefill, and again prefilling only through the
     first image and decoding the rest one token at a time (the cache's M-RoPE rows), and
     from the PNGs with tang's own preprocessing.

The bar is the repo's for fp32 references: top-1 agreement 100% and max KL < 1e-3.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration

REPO = "Qwen/Qwen3-VL-8B-Instruct"
FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
         "preprocessor_config.json", "chat_template.json", "generation_config.json"]


def render(w, h, seed):
    """A flat-shaded 'CAD viewport': grid floor, a box, a cylinder, an axis triad."""
    rng = np.random.default_rng(seed)
    im = Image.new("RGB", (w, h), (38, 40, 46))
    d = ImageDraw.Draw(im)
    for x in range(0, w, 16):
        d.line([(x, h // 2), (x - h // 3, h)], fill=(60, 63, 70))
    for y in range(h // 2, h, 12):
        d.line([(0, y), (w, y)], fill=(60, 63, 70))
    x0, y0 = int(rng.integers(10, w // 3)), int(rng.integers(10, h // 3))
    d.polygon([(x0, y0 + 40), (x0 + 60, y0 + 25), (x0 + 110, y0 + 45), (x0 + 50, y0 + 62)], fill=(170, 175, 190))
    d.polygon([(x0, y0 + 40), (x0 + 50, y0 + 62), (x0 + 50, y0 + 120), (x0, y0 + 95)], fill=(120, 125, 140))
    d.polygon([(x0 + 50, y0 + 62), (x0 + 110, y0 + 45), (x0 + 110, y0 + 100), (x0 + 50, y0 + 120)], fill=(95, 100, 115))
    cx = int(rng.integers(w // 2, w - 50))
    d.rectangle([cx - 25, h // 3, cx + 25, h // 3 + 70], fill=(200, 120, 60))
    d.ellipse([cx - 25, h // 3 - 10, cx + 25, h // 3 + 10], fill=(230, 150, 80))
    for (dx, dy), c in zip([(30, 0), (0, -30), (-18, 14)], [(220, 60, 60), (60, 200, 80), (70, 110, 230)]):
        d.line([(20, h - 20), (20 + dx, h - 20 + dy)], fill=c, width=3)
    return im


def build(out, seed):
    torch.manual_seed(seed)
    cfg = json.load(open(hf_hub_download(REPO, "config.json")))
    t, v = cfg["text_config"], cfg["vision_config"]
    t.update(hidden_size=128, intermediate_size=256, num_hidden_layers=4, num_attention_heads=4,
             num_key_value_heads=2, head_dim=64, max_position_embeddings=4096)
    t["rope_parameters"] = dict(t.get("rope_parameters") or t.get("rope_scaling") or {},
                                mrope_section=[12, 10, 10], rope_type="default",
                                rope_theta=t.get("rope_theta", 5e6), mrope_interleaved=True)
    t.pop("rope_scaling", None)
    v.update(depth=4, hidden_size=144, intermediate_size=320, num_heads=2, out_hidden_size=128,
             num_position_embeddings=64, deepstack_visual_indexes=[0, 2])
    cfg["torch_dtype"] = "float32"
    config = Qwen3VLConfig(**{k: x for k, x in cfg.items() if k not in ("architectures", "transformers_version")})
    config._attn_implementation = "eager"
    model = Qwen3VLForConditionalGeneration(config).eval().float()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name and name.endswith("weight"):
                p.copy_(1 + 0.2 * torch.randn(p.shape, generator=g))
            elif name.endswith("bias") or "norm" in name:
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
            elif "embed_tokens" in name:
                p.copy_(torch.randn(p.shape, generator=g))
            elif "pos_embed" in name:
                p.copy_(0.5 * torch.randn(p.shape, generator=g))
            else:
                fan_in = p[0].numel()
                # A sharp head (HF entropy ~1-3 nats), so the KL bar means something.
                scale = 12.0 if name == "lm_head.weight" else 1.5
                p.copy_(torch.randn(p.shape, generator=g) * (scale / fan_in ** 0.5))
    hf = os.path.join(out, "hf")
    model.save_pretrained(hf, safe_serialization=True)
    for f in FILES:
        try:
            shutil.copy(hf_hub_download(REPO, f), hf)
        except Exception:
            pass
    return model, hf


DEVICE = "cpu"


def tang(binary, *args):
    r = subprocess.run([binary, *map(str, args), "--device", DEVICE], capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"tang failed: {' '.join(map(str, args))}\n{r.stderr}")
    return r.stdout


def write_f32(path, a):
    np.ascontiguousarray(a, dtype="<f4").tofile(path)


def compare_logits(name, ref, got):
    ref, got = ref.astype(np.float64), np.asarray(got, dtype=np.float64)
    assert ref.shape == got.shape, (ref.shape, got.shape)
    lp = ref - np.logaddexp.reduce(ref, axis=-1, keepdims=True)
    lq = got - np.logaddexp.reduce(got, axis=-1, keepdims=True)
    kl = (np.exp(lp) * (lp - lq)).sum(-1)
    top1 = (ref.argmax(-1) == got.argmax(-1)).mean()
    ent = -(np.exp(lp) * lp).sum(-1).mean()
    ok = top1 == 1.0 and kl.max() < 1e-3
    print(f"{name:34s} pos {len(ref):4d}  HF entropy {ent:.2f}  max|Δ| {np.abs(ref - got).max():.1e}  "
          f"top-1 {top1 * 100:.1f}%  max KL {kl.max():.1e}  mean KL {kl.mean():.1e}  {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tang", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu (the parity bar), metal or cuda")
    a = ap.parse_args()
    global DEVICE
    DEVICE = a.device
    os.makedirs(a.out, exist_ok=True)
    model, hf = build(a.out, a.seed)
    proc = AutoProcessor.from_pretrained(hf)
    imgs = [render(330, 200, a.seed), render(260, 300, a.seed + 1)]
    paths = []
    for i, im in enumerate(imgs):
        p = os.path.join(a.out, f"img{i}.png")
        im.save(p)
        paths.append(p)
    msgs = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": "Which part is the cylinder?"},
        {"type": "image"}, {"type": "text", "text": "And what changed between the two views?"}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inputs = proc(text=[text], images=imgs, return_tensors="pt")
    ids = inputs["input_ids"][0].tolist()
    grids = inputs["image_grid_thw"].tolist()
    px = inputs["pixel_values"].float()
    ps = 16
    first = px.reshape(-1, 3, 2, ps, ps)[:, :, 0].reshape(px.shape[0], -1).numpy()

    # Reference: per-block vision states, embeddings, DeepStack, and logits.
    blocks = []
    hooks = [model.model.visual.blocks[0].register_forward_pre_hook(lambda m, i: blocks.append(i[0].detach().clone()))]
    for b in model.model.visual.blocks:
        hooks.append(b.register_forward_hook(lambda m, i, o: blocks.append(o.detach().clone())))
    with torch.no_grad():
        vis = model.model.visual(px, grid_thw=inputs["image_grid_thw"], return_dict=True)
        blocks_ref = list(blocks)
        for h in hooks:
            h.remove()
        logits = model(**inputs).logits[0].float().numpy()
    counts = [t * h * w for t, h, w in grids]
    offs = np.cumsum([0] + counts)
    tok_offs = np.cumsum([0] + [c // 4 for c in counts])

    ok = True
    print(f"images {[(h, w) for _, h, w in grids]} (patch grid), prompt {len(ids)} tokens")
    # Preprocessing.
    for i, p in enumerate(paths):
        out = os.path.join(a.out, f"tang_px{i}.f32")
        gh, gw = map(int, tang(a.tang, "vl-preprocess", hf, p, out).split())
        mine = np.fromfile(out, dtype="<f4").reshape(-1, 3 * ps * ps)
        assert (gh, gw) == tuple(grids[i][1:]), ((gh, gw), grids[i])
        d = np.abs(mine - first[offs[i]:offs[i + 1]])
        print(f"preprocess image {i}: grid {gh}x{gw} matches; max |Δpixel| {d.max():.4f} "
              f"(1/255 = {2 / 255:.4f} after normalising), mean {d.mean():.2e}")
    # Per-block tower trace, on HF's own patches.
    for i in range(len(imgs)):
        f = os.path.join(a.out, f"hf_px{i}.f32")
        write_f32(f, first[offs[i]:offs[i + 1]])
        tr = json.loads(tang(a.tang, "vl-trace", hf, f, grids[i][1], grids[i][2], "--f32"))
        worst = 0
        for k, (r, g) in enumerate(zip(blocks_ref, tr["blocks"])):
            r = r[offs[i]:offs[i + 1]].numpy().ravel()
            rel = np.abs(r - np.array(g)).max() / np.abs(r).max()
            worst = max(worst, rel)
            print(f"  image {i} {'embed' if k == 0 else f'block {k - 1}':8s} rel max|Δ| {rel:.1e}")
        r = vis.pooler_output[tok_offs[i]:tok_offs[i + 1]].numpy().ravel()
        rel = np.abs(r - np.array(tr["embeds"])).max() / np.abs(r).max()
        print(f"  image {i} merger   rel max|Δ| {rel:.1e}")
        worst = max(worst, rel)
        for j, ds in enumerate(tr["deepstack"]):
            r = vis.deepstack_features[j][tok_offs[i]:tok_offs[i + 1]].numpy().ravel()
            rel = np.abs(r - np.array(ds)).max() / np.abs(r).max()
            print(f"  image {i} deepstack {j} rel max|Δ| {rel:.1e}")
            worst = max(worst, rel)
        ok &= worst < 1e-4
    # Logits.
    patch_args = []
    for i in range(len(imgs)):
        patch_args += ["--patches", os.path.join(a.out, f"hf_px{i}.f32"), grids[i][1], grids[i][2]]
    got = json.loads(tang(a.tang, "vl-logits", hf, *patch_args, *ids, "--f32"))
    ok &= compare_logits("prefill (HF patches)", logits, got)
    pad = model.config.image_token_id
    runs = [i for i in range(len(ids)) if ids[i] == pad and (i == 0 or ids[i - 1] != pad)]
    ends = [r + c // 4 for r, c in zip(runs, counts)]
    k = ends[0] + 3
    got = json.loads(tang(a.tang, "vl-logits", hf, *patch_args[:4], *ids[:k], "--f32"))
    ok &= compare_logits(f"prefill through image 1 ({k})", logits[:k], got)
    # Decode after the first image (the second needs a prefill, so stop before it), then
    # after both.
    got = json.loads(tang(a.tang, "vl-logits", hf, *patch_args[:4], *ids[:runs[1]], "--step", k, "--f32"))
    ok &= compare_logits(f"decode {k}..{runs[1]} one at a time", logits[:runs[1]], got)
    k2 = ends[1] + 2
    got = json.loads(tang(a.tang, "vl-logits", hf, *patch_args, *ids, "--step", k2, "--f32"))
    ok &= compare_logits(f"decode {k2}..{len(ids)} one at a time", logits, got)
    # Sensitivity: what the bar would see if M-RoPE or DeepStack were missing.
    with torch.no_grad():
        flat = torch.arange(len(ids))[None].expand(3, 1, -1)
        no_mrope = model(**inputs, position_ids=flat).logits[0].float().numpy()
        ds = model.model.language_model._deepstack_process
        model.model.language_model._deepstack_process = lambda h, m, e: h
        no_ds = model(**inputs).logits[0].float().numpy()
        model.model.language_model._deepstack_process = ds
    compare_logits("(HF without M-RoPE, for scale)", logits, no_mrope)
    compare_logits("(HF without DeepStack, for scale)", logits, no_ds)
    img_args = sum([["--image", p] for p in paths], [])
    got = json.loads(tang(a.tang, "vl-logits", hf, *img_args, *ids, "--f32"))
    compare_logits("prefill (tang preprocessing)", logits, got)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
