#!/usr/bin/env python3
"""Per-tensor bits/value vs relative-L2 curves on the 32 KV dumps
(bench/tensors/kv), decode cost, and the streaming (append) path.

    KVC_BIN=... TC_BIN=... $UVW python bench/kv_bench.py

Writes bench/out/kv_curves.csv and prints a summary (median over tensors of
bpv at relL2 = 2/5/10%, log-linear on each codec's Pareto front) plus
decode ns/value and append-vs-one-shot rate.
"""
from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kv_codecs as K  # noqa: E402
import tensors_codecs as C  # noqa: E402
from tensors_bench import at_rel  # noqa: E402
from tensors_io import T  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
RELS = (0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2)


def rel(x, y):
    return float(np.linalg.norm((x - y).ravel()) / np.linalg.norm(x.ravel()))


def curves(name, x, is_key):
    rows = []
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    for e in RELS:
        for tag, extra in (("tkv-pre", ()), ("tkv-post", ("--post",))):
            b, y = K.tkv(x, is_key, e, extra)
            rows.append((name, tag, e, b, rel(x, y)))
        b, y = K.sz3(x, is_key, e)
        rows.append((name, "sz3-abs", e, b, rel(x, y)))
        b, y = C.tang(x, "wavelet", e * rms, f"kvb_{name}_{e}")
        rows.append((name, "tang-wavelet(P1)", e, b, rel(x, y)))
    for bits in (2, 3, 4, 5, 6, 8):
        b, y = K.kivi(x, is_key, bits, 32)
        rows.append((name, "kivi-g32", bits, b, rel(x, y)))
        b, y = K.kivi(x, is_key, bits, 128)
        rows.append((name, "kivi-g128", bits, b, rel(x, y)))
    return rows


def decode_cost(x, is_key, e=0.05, reps=20):
    h, t, d = x.shape
    with tempfile.TemporaryDirectory(dir=C.SCRATCH) as td:
        fi, fe, fo = (Path(td) / s for s in ("x.f32", "x.kvc", "y.f32"))
        np.ascontiguousarray(x, "<f4").tofile(fi)
        rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
        base = [K.KVC, "encode", fi, fe, "--shape", f"{h}x{t}x{d}", "--kind", "k" if is_key else "v", "--rmse", repr(e * rms)]
        one = json.loads(subprocess.run(base, capture_output=True, text=True, check=True).stdout.splitlines()[-1])
        full = json.loads(subprocess.run([K.KVC, "decode", fe, fo, "--reps", str(reps)], capture_output=True, text=True, check=True).stdout.splitlines()[-1])
        last = json.loads(subprocess.run([K.KVC, "decode", fe, fo, "--range", f"{t - 1}:{t}", "--reps", str(reps * 10)], capture_output=True, text=True, check=True).stdout.splitlines()[-1])
        app = json.loads(subprocess.run(base + ["--prefill", str(t // 2)], capture_output=True, text=True, check=True).stdout.splitlines()[-1])
    # SZ3 decode time at the same nominal error
    from pysz import sz, szConfig, szErrorBoundMode

    c = szConfig()
    c.errorBoundMode = szErrorBoundMode.ABS
    c.absErrorBound = e * rms
    xc = np.ascontiguousarray(x, np.float32)
    cmp, _ = sz.compress(xc, c)
    t0 = time.perf_counter()
    for _ in range(reps):
        sz.decompress(cmp, np.float32, xc.shape)
    sz_ns = 1e9 * (time.perf_counter() - t0) / reps / x.size
    return dict(bpv=one["bpv"], enc_ns=1e9 * one["encode_s"] / x.size, dec_ns=full["ns_per_value"],
                last_token_us=1e6 * last["decode_s"], append_bpv=app["bpv"],
                append_us_per_token=1e6 * app["append_s"] / max(app["appended"], 1), sz3_dec_ns=sz_ns)


def main():
    OUT.mkdir(exist_ok=True)
    C.SCRATCH.mkdir(parents=True, exist_ok=True)
    files = sorted((T / "kv").glob("*.npy"))
    rows = []
    for f in files:
        x = np.load(f).astype(np.float32)
        rows += curves(f.stem, x, f.stem.endswith("_K"))
        print("curves", f.stem, flush=True)
    with open(OUT / "kv_curves.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tensor", "codec", "param", "bpv", "rel"])
        w.writerows(rows)
    print("\n| codec | bpv @2% | bpv @5% | bpv @10% |\n|---|---|---|---|")
    for codec in dict.fromkeys(r[1] for r in rows):
        per = {}
        for r in rows:
            if r[1] == codec:
                per.setdefault(r[0], []).append(dict(bpv=r[3], rel=r[4]))
        cells = []
        for tgt in (0.02, 0.05, 0.10):
            a = [at_rel(p, tgt) for p in per.values()]
            cells.append(f"{np.nanmedian(a):.2f}" if any(map(math.isfinite, a)) else "—")
        print(f"| {codec} | " + " | ".join(cells) + " |")
    print("\n| tensor | bpv@5% | append bpv | enc ns/v | dec ns/v | last-token decode µs | append µs/token | SZ3 dec ns/v |\n|---|---|---|---|---|---|---|---|")
    for f in files:
        if f.stem.startswith("wiki0"):
            r = decode_cost(np.load(f).astype(np.float32), f.stem.endswith("_K"))
            print(f"| {f.stem} | {r['bpv']:.2f} | {r['append_bpv']:.2f} | {r['enc_ns']:.0f} | {r['dec_ns']:.1f} | {r['last_token_us']:.0f} | {r['append_us_per_token']:.1f} | {r['sz3_dec_ns']:.1f} |", flush=True)


if __name__ == "__main__":
    main()
