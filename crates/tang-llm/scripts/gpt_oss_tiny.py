#!/usr/bin/env python3
"""Parity of tang's gpt-oss path (`tang-llm logits`) against Hugging Face fp32, on a tiny
random-init gpt-oss (the architecture of gpt-oss-20b and gpt-oss-120b).

    pip install torch transformers safetensors numpy
    gpt_oss_tiny.py --tang target/release/tang-llm --out /tmp/gpt-oss-tiny

1. Build a GptOssForCausalLM with the real layer pattern at toy width: 4 layers alternating
   sliding-window (window 8, so a 40-token prompt crosses it) and full attention; 4 q heads
   and 2 kv heads of the real 64 dims; attention sinks; Q/K/V/O biases; YaRN as gpt-oss ships
   it (θ 150000, ×32, β 32/1, untruncated); 8 experts, top-4, with router and expert biases.
   Every weight random at a scale that keeps logits spread out and pushes some expert gates
   past the SwiGLU clamp. Tokenizer: gpt-oss's (o200k_harmony).
2. The experts are drawn as MXFP4 codes and scales, then dequantized exactly (every value is
   a power of two times an E2M1 value) into the dense weights HF runs. Two checkpoints come
   out: `dense/` (transformers' own save, experts `[E, in, out]` f32) and `mxfp4/` (the same
   with experts as `*_blocks`/`*_scales` u8, the layout of the published checkpoints).
3. Run the same token ids through HF (fp32, eager attention, one prefill) and tang on both
   checkpoints, as one prefill and one token at a time (`logits-step`, the decode path), and
   compare every position's full distribution: top-1 agreement and KL(HF || tang).

The bar is the repo's for fp32 references: top-1 agreement 100% and max KL < 1e-3.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, GptOssConfig, GptOssForCausalLM

TOKENIZER_REPO = "openai/gpt-oss-20b"
PROMPT = (
    "def quicksort(xs):\n"
    "    \"\"\"Sort a list with quicksort.\"\"\"\n"
    "    if len(xs) <= 1:\n"
    "        return xs\n"
    "    pivot = xs[len(xs) // 2]\n"
    "    left = [x for x in xs if x < pivot]\n"
)
FP4 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def mxfp4(experts, rows, cols, scale_lo, scale_hi, gen):
    """Random MXFP4 `(blocks [E, rows, cols/32, 16] u8, scales [E, rows, cols/32] u8)` and
    their exact dequantization `[E, rows, cols]` (HF's convert_moe_packed_tensors)."""
    blocks = torch.randint(0, 256, (experts, rows, cols // 32, 16), generator=gen, dtype=torch.uint8)
    scales = torch.randint(scale_lo, scale_hi + 1, (experts, rows, cols // 32), generator=gen, dtype=torch.uint8)
    lo, hi = FP4[(blocks & 15).long()], FP4[(blocks >> 4).long()]
    vals = torch.stack([lo, hi], dim=-1).reshape(experts, rows, cols // 32, 32)
    vals = torch.ldexp(vals, (scales.to(torch.int32) - 127)[..., None])
    return blocks, scales, vals.reshape(experts, rows, cols)


def build(out, seed):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    layers = 4
    cfg = GptOssConfig(
        vocab_size=201088,
        hidden_size=128,
        intermediate_size=64,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        num_local_experts=8,
        num_experts_per_tok=4,
        sliding_window=8,
        layer_types=["sliding_attention", "full_attention"] * (layers // 2),
        rope_parameters={
            "rope_type": "yarn",
            "rope_theta": 150000.0,
            "factor": 32.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "truncate": False,
            "original_max_position_embeddings": 4096,
        },
        max_position_embeddings=131072,
        rms_norm_eps=1e-5,
        attention_bias=True,
        swiglu_limit=7.0,
        tie_word_embeddings=False,
        eos_token_id=200002,
        pad_token_id=199999,
    )
    cfg._attn_implementation = "eager"
    model = GptOssForCausalLM(cfg).eval()
    blocks = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.copy_(1.0 + 0.2 * torch.randn_like(p))
            elif "sinks" in name:
                p.copy_(torch.randn_like(p))
            elif "embed" in name or "lm_head" in name:
                p.normal_(0, 0.5 if "embed" in name else 0.15)
            elif name.endswith("bias"):
                p.normal_(0, 0.3)
            elif "experts" in name:
                continue
            else:
                p.normal_(0, 0.6 / p.shape[-1] ** 0.5 * 2)
        for l, layer in enumerate(model.model.layers):
            ex = layer.mlp.experts
            E, h, ff = cfg.num_local_experts, cfg.hidden_size, cfg.intermediate_size
            # Gate/up: rows are the 2ff outputs (gate/up interleaved), cols the hidden input.
            gb, gs, gv = mxfp4(E, 2 * ff, h, 120, 125, gen)
            db, ds, dv = mxfp4(E, h, ff, 118, 122, gen)
            ex.gate_up_proj.copy_(gv.transpose(1, 2))
            ex.down_proj.copy_(dv.transpose(1, 2))
            ex.gate_up_proj_bias.normal_(0, 0.5)
            ex.down_proj_bias.normal_(0, 0.1)
            pre = f"model.layers.{l}.mlp.experts"
            blocks[f"{pre}.gate_up_proj_blocks"] = gb
            blocks[f"{pre}.gate_up_proj_scales"] = gs
            blocks[f"{pre}.down_proj_blocks"] = db
            blocks[f"{pre}.down_proj_scales"] = ds
    dense = os.path.join(out, "dense")
    model.save_pretrained(dense, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(TOKENIZER_REPO)
    tok.save_pretrained(dense)

    # The MXFP4 checkpoint: dense experts swapped for blocks/scales, as published.
    mx = os.path.join(out, "mxfp4")
    shutil.rmtree(mx, ignore_errors=True)
    shutil.copytree(dense, mx)
    tensors = {}
    for f in sorted(os.listdir(mx)):
        if f.endswith(".safetensors"):
            tensors.update(load_file(os.path.join(mx, f)))
            os.remove(os.path.join(mx, f))
    for f in os.listdir(mx):
        if f.endswith(".index.json"):
            os.remove(os.path.join(mx, f))
    for k in [k for k in tensors if k.endswith("gate_up_proj") or k.endswith("down_proj")]:
        del tensors[k]
    tensors.update(blocks)
    save_file({k: v.contiguous() for k, v in tensors.items()}, os.path.join(mx, "model.safetensors"))
    with open(os.path.join(mx, "config.json")) as f:
        c = json.load(f)
    c["quantization_config"] = {"quant_method": "mxfp4", "modules_to_not_convert": ["model.layers.*.self_attn", "model.layers.*.mlp.router", "model.embed_tokens", "lm_head"]}
    with open(os.path.join(mx, "config.json"), "w") as f:
        json.dump(c, f, indent=2)
    return model, tok, dense, mx


def compare(name, ref, got):
    agree = (got.argmax(-1) == ref.argmax(-1)).float().mean().item()
    lp_ref, lp_got = ref.log_softmax(-1), got.log_softmax(-1)
    kl = (lp_ref.exp() * (lp_ref - lp_got)).sum(-1)
    ok = agree == 1.0 and kl.max().item() < 1e-3
    print(f"{name:24} {ref.shape[0]} positions · max |Δlogit| {(got - ref).abs().max():.2e} · "
          f"top-1 {agree:.0%} · max KL {kl.max():.2e} · {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tang", required=True)
    ap.add_argument("--out", default="/tmp/gpt-oss-tiny")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tang-flags", default="--f32",
                    help="weight dtype flag for tang (--f32, or --q4 for 4-bit attention and head)")
    a = ap.parse_args()
    model, tok, dense, mx = build(a.out, a.seed)
    ids = tok(PROMPT)["input_ids"]
    with torch.no_grad():
        ref = model(torch.tensor([ids])).logits[0].float()
    spread = ref.std(-1).mean().item()
    print(f"{len(ids)} tokens, HF logit std {spread:.2f}")
    ok = True
    for ckpt in (dense, mx):
        for cmd in ("logits", "logits-step"):
            out = subprocess.run([a.tang, cmd, ckpt, *map(str, ids), *a.tang_flags.split()],
                                 capture_output=True, text=True)
            if out.returncode != 0:
                print(out.stderr)
                sys.exit(1)
            got = torch.tensor(json.loads(out.stdout))
            ok &= compare(f"{os.path.basename(ckpt)} {cmd} {a.tang_flags}", ref, got)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
