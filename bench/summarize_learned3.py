#!/usr/bin/env python3
"""TCG3 learned-probability summary: geomean bpv ratios vs serial
tc-wavelet, vs TCG3 static (same blocks/lanes), vs SPERR; all / original
5 / held-out 14."""
import csv, glob, json, math
from collections import defaultdict
from run_bench import pareto, bpv_at

ORIG = {"miranda-density", "miranda-velocityx", "hurricane-TC", "hurricane-U", "nyx-temperature"}
old = [json.loads(l) for l in open("out/gpu_ratio.jsonl") if l.strip()]
ser = {(r["field"].replace("-64x128x128", ""), r["target"]): r["bpv"] for r in old if r["codec"] == "tc-wavelet"}
st = {(r["field"].replace("-64x128x128", ""), r["target"]): r["bpv"] for r in map(json.loads, open("out/gpu3_l8_ratio.jsonl"))}
rows = [json.loads(l) for l in open("out/gpu3_learned_ratio.jsonl") if l.strip()]
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
T = ["p60", "p80", "p100", "m1e-3", "m1e-4"]
gm = lambda v: math.exp(sum(map(math.log, v)) / len(v)) if v else float("nan")
fails = [r for r in rows if "FAIL" in r]
print(f"failures: {len(fails)}")
R = defaultdict(list)
for r in rows:
    if "FAIL" in r:
        continue
    f = r["field"].replace("-64x128x128", ""); t = r["target"]; m = r["model"]
    grp = "orig" if f in ORIG else "held"
    refs = {"serial": ser.get((f, t)), "static": st.get((f, t))}
    refs["sperr"] = (bpv_at(pareto(sp[f]), r["psnr"]) if t.startswith("p") else spm.get((f, float(t[1:])))) if f in sp or not t.startswith("p") else None
    for k, b in refs.items():
        if b:
            for g in ("all", grp):
                R[(m, k, g, t)].append(r["bpv"] / b)
models = sorted({r["model"] for r in rows if "model" in r})
print("| model | vs | fields | " + " | ".join(T) + " |\n|---|---|---|" + "---|" * len(T))
for m in models:
    for k in ("serial", "static", "sperr"):
        for g in ("all", "orig", "held"):
            print(f"| {m} | {k} | {g} | " + " | ".join(f"{gm(R[(m,k,g,t)]):.3f}" for t in T) + " |")
