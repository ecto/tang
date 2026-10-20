#!/usr/bin/env python3
"""Summarize out/gpu_ratio.jsonl: tc-gpu vs serial tc-wavelet (same target,
direct ratio) and vs SPERR (SPERR bpv interpolated on its measured front at
tc-gpu's measured PSNR; max-abs: SPERR's bpv at the same bound). Geometric
means over fields, per brick-size class (A: 32^3 / 256^2, B: 64^3 / 512^2,
C: 128^3 / 1024^2)."""
import csv, glob, json, math
from collections import defaultdict
from run_bench import pareto, bpv_at

rows = [json.loads(l) for l in open("out/gpu_ratio.jsonl") if l.strip()]
sp = defaultdict(list); spm = {}
files = ["out/results_frontier.csv"] + glob.glob("out/heldout_*.csv")
for f in files:
    for r in csv.DictReader(open(f)):
        if r["codec"] != "sperr":
            continue
        d = r["dataset"].replace("-64x128x128", "")
        if r["kind"] == "psnr":
            sp[d].append({"bpv": float(r["bpv"]), "psnr": float(r["psnr"])})
        elif r["kind"] == "maxabs":
            k = (d, float(r["bound_rel"]))
            spm[k] = min(spm.get(k, 1e9), float(r["bpv"]))
cls = {32: "A", 256: "A", 64: "B", 512: "B", 128: "C", 1024: "C"}
ser = {(r["field"], r["target"]): r for r in rows if r["codec"] == "tc-wavelet"}
gpu = [r for r in rows if r["codec"] == "tc-gpu"]
vs_ser = defaultdict(list); vs_sp = defaultdict(list); ser_sp = defaultdict(list)
per = defaultdict(dict)
for r in gpu:
    f = r["field"].replace("-64x128x128", ""); t = r["target"]; c = cls[r["brick"]]
    s = ser.get((r["field"], t))
    if s:
        vs_ser[(c, t)].append(r["bpv"] / s["bpv"])
        per[(f, t)]["serial"] = s["bpv"]
    per[(f, t)][c] = r["bpv"]
    if t.startswith("p"):
        b = bpv_at(pareto(sp[f]), r["psnr"]) if f in sp else None
        sb = bpv_at(pareto(sp[f]), s["psnr"]) if (s and f in sp) else None
    else:
        b = spm.get((f, float(t[1:])))
        sb = b
    if b:
        vs_sp[(c, t)].append(r["bpv"] / b)
        per[(f, t)]["sperr"] = b
    if sb and c == "A" and s:
        ser_sp[t].append(s["bpv"] / sb)
gm = lambda v: math.exp(sum(map(math.log, v)) / len(v)) if v else float("nan")
T = ["p60", "p80", "p100", "m1e-3", "m1e-4"]
print("### Geometric-mean bpv ratio\n")
print("| ratio | bricks | " + " | ".join(T) + " |\n|---|---|" + "---|" * len(T))
for c, lab in [("A", "32³ / 256²"), ("B", "64³ / 512²"), ("C", "128³ / 1024²")]:
    print(f"| tc-gpu / tc-wavelet (serial) | {lab} | " + " | ".join(f"{gm(vs_ser[(c,t)]):.3f} (n={len(vs_ser[(c,t)])})" for t in T) + " |")
for c, lab in [("A", "32³ / 256²"), ("B", "64³ / 512²"), ("C", "128³ / 1024²")]:
    print(f"| tc-gpu / SPERR | {lab} | " + " | ".join(f"{gm(vs_sp[(c,t)]):.3f} (n={len(vs_sp[(c,t)])})" for t in T) + " |")
print(f"| tc-wavelet (serial) / SPERR | — | " + " | ".join(f"{gm(ser_sp[t]):.3f} (n={len(ser_sp[t])})" for t in T) + " |")
print("\n### Per-field bpv\n\n| field | target | serial | A | B | C | SPERR |\n|---|---|---|---|---|---|---|")
for (f, t), v in sorted(per.items()):
    g = lambda k: f"{v[k]:.4g}" if k in v else "—"
    print(f"| {f} | {t} | {g('serial')} | {g('A')} | {g('B')} | {g('C')} | {g('sperr')} |")
