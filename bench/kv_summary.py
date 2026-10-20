#!/usr/bin/env python3
"""Cheapest bpv per codec family at Δppl bounds from a kvhf_ppl_*.csv.

    python bench/kv_summary.py bench/out/kvhf_ppl_smol135_w1024.csv [bound ...]
"""
import csv
import math
import re
import sys

FAM = [("tkv", r"^tkv-"), ("tkv (K 0.7x)", r"^tkvk0\.7-"), ("tkv (K 1.0x)", r"^tkvk1\.0-"), ("tkv (K 1.4x)", r"^tkvk1\.4-"), ("tkv GPU layout", r"^tkvgpu-"), ("tkv GPU layout (K 0.7x)", r"^tkvgpuk0\.7-"), ("tang kvcodec, any K/V split", r"^tkv"), ("KIVI, fp16 window R=128", r"^kivi-R128-"),
       ("KIVI, no window", r"^kivi-R0-"), ("KVQuant-style", r"^kvquant-"), ("Hadamard (QuaRot-style), g=hd", r"^hadamard-"),
       ("Hadamard, clip 0.9", r"^hadamardc90-"), ("Hadamard, g=32", r"^hadamardg32-"), ("TurboQuant-mse", r"^turbo-")]


def front(rows):
    pts = sorted((float(r["bpv"]), float(r["dppl"]), r["config"]) for r in rows)
    out, best = [], math.inf
    for b, d, n in pts:
        if d < best:
            out.append((b, d, n))
            best = d
    return out


def at(fr, bound):
    ok = [p for p in fr if p[1] <= bound]
    if not ok:
        return None, None, None
    b, d, n = ok[0]
    i = fr.index(ok[0])
    interp = b
    if i > 0:  # log-linear interpolation of the bound crossing
        b0, d0, _ = fr[i - 1]
        if d0 > bound and d > 0 and d0 > 0:
            f = (math.log(d0) - math.log(bound)) / (math.log(d0) - math.log(d))
            interp = b0 + f * (b - b0)
    return (b, d, n), interp, ok


def main():
    path = sys.argv[1]
    bounds = [float(x) for x in sys.argv[2:]] or [0.1, 0.5]
    rows = list(csv.DictReader(open(path)))
    print(f"ref ppl {rows[0]['ref']}")
    print("| codec | " + " | ".join(f"Δppl ≤ {b}: measured (interp.)" for b in bounds) + " |")
    print("|---|" + "---|" * len(bounds))
    res = {}
    for name, rx in FAM:
        fr = front([r for r in rows if re.match(rx, r["config"])])
        if not fr:
            continue
        cells = []
        for bd in bounds:
            p, ib, _ = at(fr, bd)
            res[(name, bd)] = ib
            cells.append("not reached (best Δ %+.3f at %.2f)" % (min(fr, key=lambda q: q[1])[1], min(fr, key=lambda q: q[1])[0]) if p is None
                         else f"{p[0]:.2f} ({ib:.2f}) `{p[2]}` Δ{p[1]:+.3f}")
        print(f"| {name} | " + " | ".join(cells) + " |")
    for bd in bounds:
        t = res.get(("tang kvcodec, any K/V split", bd))
        others = {k[0]: v for k, v in res.items() if k[1] == bd and not k[0].startswith("tkv") and v}
        if t and others:
            bn, bv = min(others.items(), key=lambda kv: kv[1])
            print(f"Δ≤{bd}: tkv {t:.2f} vs best baseline {bn} {bv:.2f}: {100 * (1 - t / bv):.1f}% fewer bits (interp.)")


if __name__ == "__main__":
    main()
