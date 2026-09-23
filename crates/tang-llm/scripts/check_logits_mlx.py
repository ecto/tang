"""Compare tang-llm logits on an MLX-quantized checkpoint against mlx-lm (computing in fp32).

usage: python check_logits_mlx.py <model-dir> <tang-llm binary> [prompt]
"""
import json, subprocess, sys
import numpy as np
import mlx.core as mx
from mlx.utils import tree_map
from mlx_lm import load

model_dir, binary = sys.argv[1], sys.argv[2]
prompt = sys.argv[3] if len(sys.argv) > 3 else "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n"
model, tok = load(model_dir)
# Same arithmetic as tang: fp32 activations (quantized weights stay packed).
model.update(tree_map(lambda p: p.astype(mx.float32) if mx.issubdtype(p.dtype, mx.floating) else p, model.parameters()))
ids = tok.encode(prompt)
ref = np.array(model(mx.array([ids]))[0].astype(mx.float32))

out = subprocess.run([binary, "logits", model_dir, *map(str, ids)], capture_output=True, text=True, check=True)
got = np.array(json.loads(out.stdout), dtype=np.float32)
assert got.shape == ref.shape, (got.shape, ref.shape)

def log_softmax(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))

lr, lg = log_softmax(ref), log_softmax(got)
kl = (np.exp(lr) * (lr - lg)).sum(-1)
agree = (got.argmax(-1) == ref.argmax(-1)).mean()
print(f"{len(ids)} tokens · max |Δlogit| {np.abs(got - ref).max():.4f} · top-1 agreement {agree:.0%} · max KL {kl.max():.2e}")
ok = agree == 1.0 and kl.max() < 1e-3
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
