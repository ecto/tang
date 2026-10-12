"""PyTorch MPS scaled_dot_product_attention, the comparison point for
`cargo run --release -p tang-bench --features metal --example attention_fragments`.
Same shapes: d = 64, heads 1, 4, 8 (ATT_HEADS), causal and not. 20 launches per
sample, one sync, best of three; f32 and f16.

    ~/.cache/kv/bin/python scripts/torch_attention.py [n ...]
"""
import sys
import time

import torch
import torch.nn.functional as F

ns = [int(a) for a in sys.argv[1:]] or [512, 1024, 2048, 4096]
d, iters = 64, 20
dev = torch.device("mps")
print(f"torch {torch.__version__} MPS SDPA, d = {d}, ms")
print("heads causal     n    sdpa-f32   sdpa-f16")
import os
heads = [int(h) for h in os.environ.get("ATT_HEADS", "1,4,8").split(",")]
causals = {"0": [False], "1": [True]}.get(os.environ.get("ATT_CAUSAL", ""), [False, True])
for h in heads:
    for causal in causals:
        for n in ns:
            row = []
            for dt in (torch.float32, torch.float16):
                q, k, v = (torch.rand(1, h, n, d, device=dev, dtype=dt) * 2 - 1 for _ in range(3))
                for _ in range(3):
                    F.scaled_dot_product_attention(q, k, v, is_causal=causal)
                torch.mps.synchronize()
                best = float("inf")
                for _ in range(3):
                    t = time.perf_counter()
                    for _ in range(iters):
                        F.scaled_dot_product_attention(q, k, v, is_causal=causal)
                    torch.mps.synchronize()
                    best = min(best, (time.perf_counter() - t) / iters * 1e3)
                row.append(best)
            print(f"{h:5} {str(causal):6} {n:5} {row[0]:10.3f} {row[1]:10.3f}")
