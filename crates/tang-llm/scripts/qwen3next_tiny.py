#!/usr/bin/env python3
"""Parity of tang's qwen3next CPU reference (`tang-llm flash-ref`) against Hugging Face fp32, on a
tiny random-init Qwen3-Next (the architecture of Qwen3-Next-80B-A3B and Qwen3-Coder-Next).

    pip install torch transformers safetensors gguf numpy
    qwen3next_tiny.py --tang target/release/tang-llm --llama-cpp ~/llama.cpp --out /tmp/q3n

1. Build a Qwen3NextForCausalLM with the real layer pattern at toy width (4 layers: 3 GDN + 1
   gated attention; 2 k heads, 4 v heads so the grouped v→k mapping is exercised; 8 experts,
   top-2, shared expert), every weight random at a scale that keeps logits spread out, the
   zero-centred norms non-zero so the converter's +1 matters. Tokenizer: Qwen3-Coder-Next's.
2. Convert with llama.cpp's convert_hf_to_gguf.py (f32), the same converter the published
   GGUFs come from.
3. Run the same token ids through HF (fp32, one prefill) and `flash-ref --logits`, and compare
   every position's full distribution: top-1 agreement and KL(HF || tang).

The bar is the repo's for fp32 references: top-1 agreement 100% and max KL < 1e-3.

`--real-shapes` builds the same 4 layers at Qwen3-Coder-Next's widths instead (hidden 2048; GDN 16
k / 32 v heads of 128; attention 16 q / 2 kv heads of 256; experts FF 512, top-10 of 16; shared
FF 512), which the CUDA engine's kernels are compiled for, and `--outtype q8_0` quantises it the
way the engine reads real files. Then also run the engine against flash-ref:

    tang-llm flash-ref GGUF IDS... --top 40 > ref.jsonl
    tang-llm flash-parity GGUF --prompt-ids "IDS" --ref ref.jsonl
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3NextConfig, Qwen3NextForCausalLM

TOKENIZER_REPO = "Qwen/Qwen3-Coder-Next"
PROMPT = (
    "def quicksort(xs):\n"
    "    \"\"\"Sort a list with quicksort.\"\"\"\n"
    "    if len(xs) <= 1:\n"
    "        return xs\n"
    "    pivot = xs[len(xs) // 2]\n"
    "    left = [x for x in xs if x < pivot]\n"
)


TINY = dict(
    hidden_size=128,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    linear_key_head_dim=32,
    linear_value_head_dim=32,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    num_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=64,
    shared_expert_intermediate_size=64,
)
REAL = dict(
    hidden_size=2048,
    num_attention_heads=16,
    num_key_value_heads=2,
    head_dim=256,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_num_key_heads=16,
    linear_num_value_heads=32,
    num_experts=16,
    num_experts_per_tok=10,
    moe_intermediate_size=512,
    shared_expert_intermediate_size=512,
)


def build(out, seed, shapes):
    torch.manual_seed(seed)
    cfg = Qwen3NextConfig(
        vocab_size=151936,
        intermediate_size=256,
        num_hidden_layers=4,
        partial_rotary_factor=0.25,
        rope_theta=5_000_000.0,
        rms_norm_eps=1e-6,
        linear_conv_kernel_dim=4,
        **shapes,
        norm_topk_prob=True,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        full_attention_interval=4,
        tie_word_embeddings=False,
        bos_token_id=151643,
        eos_token_id=151645,
    )
    model = Qwen3NextForCausalLM(cfg).float().eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("A_log"):
                p.uniform_(-1.0, 1.5)
            elif name.endswith("dt_bias"):
                p.normal_(0.0, 0.5)
            elif "norm" in name:
                # zero-centred norms (weight + 1) except the GDN output norm (plain weight)
                p.normal_(1.0 if name.endswith("linear_attn.norm.weight") else 0.0, 0.2)
            elif "conv1d" in name:
                p.normal_(0.0, 0.3)
            elif "embed_tokens" in name:
                p.normal_(0.0, 1.0)
            elif name == "lm_head.weight":
                # sharp enough that top-1 agreement means something
                p.normal_(0.0, 8.0 / p.shape[-1] ** 0.5)
            else:
                p.normal_(0.0, 1.0 / p.shape[-1] ** 0.5)
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(TOKENIZER_REPO).save_pretrained(out)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tang", required=True, help="tang-llm binary")
    ap.add_argument("--llama-cpp", required=True, help="llama.cpp checkout (for the converter)")
    ap.add_argument("--out", required=True, help="work dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--real-shapes", action="store_true", help="Qwen3-Coder-Next's widths")
    ap.add_argument("--outtype", default="f32", help="converter --outtype (f32, q8_0, ...)")
    a = ap.parse_args()

    hf_dir = os.path.join(a.out, "hf")
    gguf = os.path.join(a.out, f"qwen3next-{'real' if a.real_shapes else 'tiny'}-{a.outtype}.gguf")
    model = build(hf_dir, a.seed, REAL if a.real_shapes else TINY)
    subprocess.run(
        [sys.executable, os.path.join(a.llama_cpp, "convert_hf_to_gguf.py"), hf_dir,
         "--outtype", a.outtype, "--no-mtp", "--outfile", gguf],
        check=True, stdout=subprocess.DEVNULL,
    )

    ids = AutoTokenizer.from_pretrained(hf_dir)(PROMPT)["input_ids"]
    print("ids:", " ".join(map(str, ids)))
    with torch.no_grad():
        ref = model(torch.tensor([ids])).logits[0].double()

    lp = os.path.join(a.out, "tang.logits")
    subprocess.run([a.tang, "flash-ref", gguf, *map(str, ids), "--top", "1", "--logits", lp],
                   check=True, stdout=subprocess.DEVNULL)
    got = torch.from_numpy(np.fromfile(lp, dtype="<f4").astype(np.float64)).view(len(ids), -1)
    assert got.shape == ref.shape, (got.shape, ref.shape)

    lp_ref, lp_got = ref.log_softmax(-1), got.log_softmax(-1)
    kl = (lp_ref.exp() * (lp_ref - lp_got)).sum(-1)
    agree = (got.argmax(-1) == ref.argmax(-1)).double().mean().item()
    diff = (got - ref).abs()
    ent = -(lp_ref.exp() * lp_ref).sum(-1).mean().item()
    print(f"{len(ids)} tokens · HF entropy {ent:.2f} nats · max |Δlogit| {diff.max():.2e} · "
          f"top-1 agreement {agree:.0%} · max KL {kl.max():.2e} · mean KL {kl.mean():.2e}")
    ok = agree == 1.0 and kl.max() < 1e-3
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
