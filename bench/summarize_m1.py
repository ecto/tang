#!/usr/bin/env python3
"""M1 frontier tables: tang-compress wavelet/Tucker vs SPERR/TTHRESH/SZ3/QTT.

Reads the M0 results (SPERR, TTHRESH, SZ3, ZFP, QTT; same crops, same
harness error definitions) and the tang M1 runs (results_tang*.csv), builds
each codec's measured Pareto front and prints matched-PSNR and max-abs tables.
"""
import csv, sys, math
from collections import defaultdict
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from run_bench import pareto, bpv_at

files = sys.argv[1:] or ["out/tang-compress-qtt-m0-results.csv", "out/results_tang.csv"]
rows = []
for f in files:
    for r in csv.DictReader(open(f)):
        for k in ("bpv", "psnr", "max_abs", "enc_s", "dec_s"):
            r[k] = float(r[k]) if r.get(k) not in (None, "") else float("nan")
        rows.append(r)
skip = {"wavelet-H0", "sperr-mt"}
by = defaultdict(list)
for r in rows:
    if r["kind"] == "psnr" and r["codec"] not in skip:
        c = "qtt" if r["codec"].startswith("qtt") else r["codec"]
        by[(r["dataset"], c)].append(r)
datasets = sorted({d for d, _ in by})
codecs = sorted({c for _, c in by})
targets = [40, 60, 80, 100]
print("## matched PSNR (bpv, measured Pareto fronts, log-linear interpolation)\n")
print("| dataset | codec | " + " | ".join(f"{t} dB" for t in targets) + " |")
print("|---|---|" + "---|" * len(targets))
table = {}
for d in datasets:
    for c in codecs:
        if (d, c) not in by:
            continue
        fr = pareto(by[(d, c)])
        vals = [bpv_at(fr, t) for t in targets]
        table[(d, c)] = vals
        print(f"| {d} | {c} | " + " | ".join("—" if v is None else f"{v:.4f}" for v in vals) + " |")
print("\n## max-abs (bpv; rows verified ≤ b)\n")
ma = defaultdict(dict)
for r in rows:
    if r["kind"] == "maxabs":
        rel = float(r["bound_rel"])
        rng_ok = True
        c = "qtt" if r["codec"].startswith("qtt") else r["codec"]
        prev = ma[(r["dataset"], rel)].get(c)
        if prev is None or r["bpv"] < prev:
            ma[(r["dataset"], rel)][c] = r["bpv"]
mcod = sorted({c for v in ma.values() for c in v})
print("| dataset | rel | " + " | ".join(mcod) + " |")
print("|---|---|" + "---|" * len(mcod))
for (d, rel) in sorted(ma):
    print(f"| {d} | {rel:g} | " + " | ".join(f"{ma[(d, rel)].get(c, float('nan')):.4f}" for c in mcod) + " |")
