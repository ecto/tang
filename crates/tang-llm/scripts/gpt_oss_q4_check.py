#!/usr/bin/env python3
"""tang's Q4 attention + head (`--q4`) against transformers with the same quantization.

    gpt_oss_q4_check.py <checkpoint> <tang-llm binary>

Applies tang's `quantize_q4` (groups of 64 along rows, bf16 scale and bias, round to nearest)
to HF's attention, embedding and head matrices, then compares HF fp32 on those weights to
`tang-llm logits --q4`. That checks the Q4 path is right independent of how much Q4 itself
costs, which it also prints (HF-q4 vs HF). On the tiny random model from gpt_oss_tiny.py: tang-q4
vs HF-q4 max KL 1.1e-4, top-1 97% (near-ties); HF-q4 vs HF max KL 1.55, top-1 10% (random
weights are far more sensitive to 4 bits than trained ones; measure that cost on the real
checkpoint).
"""
import json, subprocess, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
d, tang = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(d)
ids = tok("def quicksort(xs):\n    if len(xs) <= 1:\n        return xs\n    pivot = xs[len(xs) // 2]\n")["input_ids"]
def bf(x): return x.to(torch.bfloat16).float()
def q4(w):  # tang's quantize_q4: groups of 64 along rows, bf16 scale/bias, round to nearest
    g = w.reshape(-1, 64)
    lo, hi = g.min(1, keepdim=True).values, g.max(1, keepdim=True).values
    s = bf(((hi - lo) / 15).clamp(min=1e-8)); b = bf(lo)
    return (((g - b) / s).round().clamp(0, 15) * s + b).reshape(w.shape)
m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32, attn_implementation="eager")
with torch.no_grad(): ref = m(torch.tensor([ids])).logits[0]
with torch.no_grad():
    for n, p in m.named_parameters():
        if p.dim() == 2 and ("self_attn" in n or "embed" in n or "lm_head" in n) and "bias" not in n:
            p.copy_(q4(p))
    refq = m(torch.tensor([ids])).logits[0]
got = torch.tensor(json.loads(subprocess.run([tang, "logits", d, *map(str, ids), "--q4"], capture_output=True, text=True, check=True).stdout))
def kl(a, b):
    la, lb = a.log_softmax(-1), b.log_softmax(-1); return (la.exp() * (la - lb)).sum(-1).max().item()
print(f"HF-q4 vs HF: KL {kl(ref, refq):.2e} top1 {(ref.argmax(-1)==refq.argmax(-1)).float().mean():.2f}")
print(f"tang-q4 vs HF-q4: KL {kl(refq, got):.2e} top1 {(refq.argmax(-1)==got.argmax(-1)).float().mean():.2f}")
