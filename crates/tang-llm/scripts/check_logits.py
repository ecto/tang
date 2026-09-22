"""Compare tang-llm logits against Hugging Face transformers (fp32) for the same token ids.

usage: python check_logits.py <model-dir> <tang-llm binary> [prompt]
"""
import json, subprocess, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir, binary = sys.argv[1], sys.argv[2]
prompt = sys.argv[3] if len(sys.argv) > 3 else "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n"
tok = AutoTokenizer.from_pretrained(model_dir)
ids = tok(prompt)["input_ids"]
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
with torch.no_grad():
    ref = model(torch.tensor([ids])).logits[0]

out = subprocess.run([binary, "logits", model_dir, *map(str, ids)], capture_output=True, text=True, check=True)
got = torch.tensor(json.loads(out.stdout))
assert got.shape == ref.shape, (got.shape, ref.shape)

diff = (got - ref).abs()
agree = (got.argmax(-1) == ref.argmax(-1)).float().mean().item()
lp_ref, lp_got = ref.log_softmax(-1), got.log_softmax(-1)
kl = (lp_ref.exp() * (lp_ref - lp_got)).sum(-1)
print(f"{len(ids)} tokens · max |Δlogit| {diff.max():.4f} · mean {diff.mean():.5f} · "
      f"top-1 agreement {agree:.0%} · max KL {kl.max():.2e}")
ok = agree == 1.0 and kl.max() < 1e-3
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
