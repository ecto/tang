#!/usr/bin/env python3
"""KV-cache codec sweep on the tensors_ppl.py KV eval (SmolLM2-135M,
WikiText-2 tokens [0, 4096), 1024-token windows, every layer's K/V replaced
by its reconstruction). CPU torch; the clean KV of each window is captured
once and reused across codec configs.

    $UVW --with torch python bench/kv_ppl.py [config-group ...]

Writes bench/out/kv_ppl.csv (one row per codec config: bpv, relL2, ppl).
"""
from __future__ import annotations

import csv
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kv_codecs as K  # noqa: E402
from tensors_dump import Llama, smol_weights  # noqa: E402
from tensors_io import T  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
WIN, NWIN = 1024, 4


def capture(m, wiki):
    caps = []
    for w in range(NWIN):
        x = torch.from_numpy(wiki[w * WIN : w * WIN + WIN + 1].astype(np.int64))[None]
        cap = {}
        with torch.no_grad():
            m(x[:, :-1], kv_hook=lambda l, k, v: cap.update({(l, "K"): k[0].float().numpy().copy(), (l, "V"): v[0].float().numpy().copy()}))
        caps.append((x, cap))
    return caps


def _job(a):
    name, x, is_key = a
    t = time.perf_counter()
    bpv, y = K.CONFIGS[name](x, is_key)
    return bpv, y.astype(np.float32), time.perf_counter() - t


def evaluate(m, caps, name, ex):
    nll = bits = nv = err = nrm = 0.0
    secs = 0.0
    for x, cap in caps:
        keys = list(cap)
        rec = {}
        for key, (bpv, y, dt) in zip(keys, ex.map(_job, [(name, cap[k], k[1] == "K") for k in keys])):
            rec[key] = torch.from_numpy(y)[None]
            bits += bpv * y.size
            nv += y.size
            err += float(((cap[key].astype(np.float64) - y) ** 2).sum())
            nrm += float((cap[key].astype(np.float64) ** 2).sum())
            secs += dt
        with torch.no_grad():
            lo = m(x[:, :-1], kv_override=lambda l, k, v: (rec[(l, "K")], rec[(l, "V")]))
            nll += torch.nn.functional.cross_entropy(lo[0].float(), x[0, 1:], reduction="sum").item()
    return bits / nv, math.sqrt(err / nrm), math.exp(nll / (NWIN * WIN)), secs


def main():
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "6")))
    wiki = np.load(T / "tokens.npz")["wiki"]
    m = Llama(smol_weights(), "cpu")
    caps = capture(m, wiki)
    nll = 0.0
    for x, _ in caps:
        with torch.no_grad():
            nll += torch.nn.functional.cross_entropy(m(x[:, :-1])[0].float(), x[0, 1:], reduction="sum").item()
    ref = math.exp(nll / (NWIN * WIN))
    print(f"ref ppl {ref:.4f}", flush=True)
    groups = sys.argv[1:] or ["all"]
    names = [n for n in K.CONFIGS if any(g == "all" or n.startswith(g) for g in groups)]
    OUT.mkdir(exist_ok=True)
    path = OUT / "kv_ppl.csv"
    new = not path.exists()
    with open(path, "a", newline="") as f, ProcessPoolExecutor(int(os.environ.get("KV_PROCS", "6"))) as ex:
        w = csv.writer(f)
        if new:
            w.writerow(["config", "bpv", "relL2", "ppl", "dppl", "codec_secs"])
        for n in names:
            bpv, rel, p, secs = evaluate(m, caps, n, ex)
            w.writerow([n, f"{bpv:.4f}", f"{rel:.5f}", f"{p:.4f}", f"{p - ref:.4f}", f"{secs:.2f}"])
            f.flush()
            print(f"{n:40s} bpv {bpv:6.3f} rel {rel:.4f} ppl {p:.4f} d {p - ref:+.4f}", flush=True)


if __name__ == "__main__":
    main()
