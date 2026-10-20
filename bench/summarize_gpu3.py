#!/usr/bin/env python3
"""v3 (code-block) summary: geomean bpv ratios vs serial tc-wavelet, vs
SPERR, and vs whole-field TCG1 (tc-gpu, 128^3 / 1024^2 class)."""
import csv, glob, json, math
from collections import defaultdict
from run_bench import pareto, bpv_at

old = [json.loads(l) for l in open("out/gpu_ratio.jsonl") if l.strip()]
import sys
new = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "out/gpu3_ratio.jsonl") if l.strip()]
ser = {(r["field"], r["target"]): r for r in old if r["codec"] == "tc-wavelet"}
v1 = {(r["field"], r["target"]): r for r in old if r["codec"] == "tc-gpu" and r["brick"] in (128, 1024)}
sp = defaultdict(list); spm = {}
for f in ["out/results_frontier.csv"] + glob.glob("out/heldout_*.csv"):
    for r in csv.DictReader(open(f)):
        if r["codec"] != "sperr":
            continue
        d = r["dataset"].replace("-64x128x128", "")
        if r["kind"] == "psnr":
            sp[d].append({"bpv": float(r["bpv"]), "psnr": float(r["psnr"])})
        elif r["kind"] == "maxabs":
            k = (d, float(r["bound_rel"])); spm[k] = min(spm.get(k, 1e9), float(r["bpv"]))
cls = {16: "S", 64: "S", 32: "M", 128: "M", 4096: "W"}
lab = {"S": "16³ / 64²", "M": "32³ / 128²", "W": "one block per band"}
R = defaultdict(lambda: defaultdict(list)); per = defaultdict(dict)
for r in new:
    f = r["field"].replace("-64x128x128", ""); t = r["target"]; c = cls[r["brick"]]; k = (r["field"], t)
    per[(f, t)][c] = r["bpv"]
    if k in ser:
        R["ser"][(c, t)].append(r["bpv"] / ser[k]["bpv"]); per[(f, t)]["serial"] = ser[k]["bpv"]
    if k in v1:
        R["v1"][(c, t)].append(r["bpv"] / v1[k]["bpv"]); per[(f, t)]["v1"] = v1[k]["bpv"]
    b = bpv_at(pareto(sp[f]), r["psnr"]) if t.startswith("p") and f in sp else spm.get((f, float(t[1:]))) if not t.startswith("p") else None
    if b:
        R["sp"][(c, t)].append(r["bpv"] / b); per[(f, t)]["sperr"] = b
gm = lambda v: math.exp(sum(map(math.log, v)) / len(v)) if v else float("nan")
T = ["p60", "p80", "p100", "m1e-3", "m1e-4"]
print("| ratio | blocks | " + " | ".join(T) + " |\n|---|---|" + "---|" * len(T))
for key, name in [("ser", "tc-gpu3 / tc-wavelet (serial)"), ("v1", "tc-gpu3 / TCG1 whole-field"), ("sp", "tc-gpu3 / SPERR")]:
    for c in "SMW":
        print(f"| {name} | {lab[c]} | " + " | ".join(f"{gm(R[key][(c,t)]):.3f} (n={len(R[key][(c,t)])})" for t in T) + " |")
print("\n| field | target | serial | TCG1 whole | S | M | W | SPERR |\n|---|---|---|---|---|---|---|---|")
for (f, t), v in sorted(per.items()):
    g = lambda k: f"{v[k]:.4g}" if k in v else "—"
    print(f"| {f} | {t} | {g('serial')} | {g('v1')} | {g('S')} | {g('M')} | {g('W')} | {g('sperr')} |")
