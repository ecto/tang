#!/usr/bin/env python3
"""Real Qwen3-VL-8B-Instruct: tang (CUDA or Metal, bf16 weights) against Hugging Face on the same
image prompt.

    qwen3vl_real.py --tang target/release/tang-llm --model <Qwen3-VL-8B-Instruct dir> --out DIR
        [--device cuda] [--hf-dtype float32]

HF runs on the GPU (fp32 by default, eager attention). tang runs three ways: one prefill on
HF's own pixels (`--patches`, isolates the model), one prefill on the PNG with tang's
preprocessing (`--image`, what the server does), and a prefill through the image followed by
one-token decode steps. Logits are compared over the last `--last` positions (the image's
end and all the text after it): top-1 agreement, KL mean / p99 / max.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

sys.path.insert(0, os.path.dirname(__file__))
from qwen3vl_tiny import render  # noqa: E402

QUESTIONS = [
    "This is a screenshot of a CAD viewport. Describe the parts you can see, their shapes and "
    "colors, and where each one is.",
]


def kl_stats(name, ref, got):
    ref, got = ref.astype(np.float64), np.asarray(got, dtype=np.float64)
    lp = ref - np.logaddexp.reduce(ref, axis=-1, keepdims=True)
    lq = got - np.logaddexp.reduce(got, axis=-1, keepdims=True)
    kl = (np.exp(lp) * (lp - lq)).sum(-1)
    top1 = (ref.argmax(-1) == got.argmax(-1)).mean()
    print(f"{name:36s} pos {len(ref):3d}  top-1 {top1 * 100:5.1f}%  KL mean {kl.mean():.1e}  "
          f"p99 {np.percentile(kl, 99):.1e}  max {kl.max():.1e}", flush=True)
    return top1, kl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tang", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hf-dtype", default="float32")
    ap.add_argument("--last", type=int, default=48)
    ap.add_argument("--image", help="a PNG instead of the synthetic render")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    def tang(*args):
        r = subprocess.run([a.tang, *map(str, args), "--device", a.device], capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"tang failed: {r.stderr[-3000:]}")
        print("  " + r.stderr.strip().replace("\n", "\n  "), flush=True)
        return json.loads(r.stdout)

    proc = AutoProcessor.from_pretrained(a.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        a.model, dtype=getattr(torch, a.hf_dtype), attn_implementation="eager").to("cuda").eval()
    from PIL import Image
    img = Image.open(a.image).convert("RGB") if a.image else render(1280, 720, 7)
    png = os.path.join(a.out, "view.png")
    img.save(png)
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTIONS[0]}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inputs = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
    ids = inputs["input_ids"][0].tolist()
    _, gh, gw = inputs["image_grid_thw"][0].tolist()
    # Per-block tower states (HF), to place any difference.
    blocks = []
    vis = model.model.visual
    hooks = [vis.blocks[0].register_forward_pre_hook(lambda m, i: blocks.append(i[0].float().cpu().numpy()))]
    for b in vis.blocks:
        hooks.append(b.register_forward_hook(lambda m, i, o: blocks.append(o.float().cpu().numpy())))
    with torch.no_grad():
        vout = vis(inputs["pixel_values"].to(vis.dtype), grid_thw=inputs["image_grid_thw"], return_dict=True)
    for h in hooks:
        h.remove()
    hf_embeds = vout.pooler_output.float().cpu().numpy()
    hf_ds = [d.float().cpu().numpy() for d in vout.deepstack_features]
    text_msgs = [{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number, with a docstring."}]
    text_ids = proc.tokenizer.apply_chat_template(text_msgs, add_generation_prompt=True, tokenize=True)
    if isinstance(text_ids, dict):
        text_ids = text_ids["input_ids"]
    with torch.no_grad():
        text_logits = model(input_ids=torch.tensor([text_ids], device="cuda")).logits[0].float().cpu().numpy()
        logits = model(**inputs).logits[0, -a.last:].float().cpu().numpy()
        gen = model.generate(**inputs, max_new_tokens=40, do_sample=False)
    print(f"prompt {len(ids)} tokens, image {gh}x{gw} patches ({gh * gw // 4} tokens), HF {a.hf_dtype}")
    print("HF greedy:", repr(proc.decode(gen[0, len(ids):], skip_special_tokens=True)), flush=True)
    px = inputs["pixel_values"].float().cpu()
    first = px.reshape(-1, 3, 2, 16, 16)[:, :, 0].reshape(px.shape[0], -1).numpy()
    pf = os.path.join(a.out, "hf_px.f32")
    np.ascontiguousarray(first, dtype="<f4").tofile(pf)
    pad = model.config.image_token_id
    # tang needs the GPU's memory: drop HF first. For scale: HF itself in bf16.
    del model, vis
    torch.cuda.empty_cache()
    m16 = Qwen3VLForConditionalGeneration.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()
    with torch.no_grad():
        l16 = m16(**inputs).logits[0, -a.last:].float().cpu().numpy()
        t16 = m16(input_ids=torch.tensor([text_ids], device="cuda")).logits[0].float().cpu().numpy()
    del m16, inputs
    torch.cuda.empty_cache()
    kl_stats("(HF bf16 vs HF fp32, image prompt)", logits, l16)
    kl_stats("(HF bf16 vs HF fp32, text prompt)", text_logits, t16)

    # Tower, per block (tang bf16 weights vs HF fp32).
    td = os.path.join(a.out, "trace")
    os.makedirs(td, exist_ok=True)
    r = subprocess.run([a.tang, "vl-trace", a.model, pf, str(gh), str(gw), td, "--device", a.device],
                       capture_output=True, text=True)
    if r.returncode:
        sys.exit(r.stderr[-3000:])
    rel = lambda ref, f: np.abs(ref.ravel() - np.fromfile(os.path.join(td, f), "<f4")).max() / np.abs(ref).max()
    errs = [rel(b, f"block_{i}.f32") for i, b in enumerate(blocks)]
    print("tower rel max|Δ| by block: " + " ".join(f"{e:.1e}" for e in errs))
    print(f"merger {rel(hf_embeds, 'embeds.f32'):.1e}, deepstack " +
          " ".join(f"{rel(d, f'deepstack_{i}.f32'):.1e}" for i, d in enumerate(hf_ds)), flush=True)

    got = tang("vl-logits", a.model, *text_ids, "--last", len(text_ids))
    kl_stats("tang text-only prompt", text_logits, got)

    got = tang("vl-logits", a.model, "--patches", pf, gh, gw, *ids, "--last", a.last)
    kl_stats("tang prefill, HF pixels", logits, got)
    got = tang("vl-logits", a.model, "--image", png, *ids, "--last", a.last)
    kl_stats("tang prefill, tang preprocessing", logits, got)
    end = max(i for i, t in enumerate(ids) if t == pad) + 1
    os.environ["GAIA_TF32"] = "0"
    got = tang("vl-logits", a.model, "--patches", pf, gh, gw, *ids, "--last", a.last, "--f32")
    kl_stats("tang f32 weights, no TF32", logits, got)
    os.environ.pop("GAIA_TF32")
    got = tang("vl-logits", a.model, "--patches", pf, gh, gw, *ids, "--step", end + 1, "--last", a.last)
    kl_stats(f"tang prefill to {end + 1}, then decode", logits, got)


if __name__ == "__main__":
    main()
