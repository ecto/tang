#!/usr/bin/env python3
"""Phase-2 gates from run_fresh.py output.

  summarize_fresh.py fresh   quality gate (tc-portfolio vs best external of
                             QTT/SZ3/ZFP/SPERR/TTHRESH from out/fresh_ext_*.csv at
                             40/60/80/100 dB and max-abs 1e-2/1e-3/1e-4: no point
                             > 1.05x) + fast gate -> out/summary_fresh.md
  summarize_fresh.py seen    fast gate only, on the seen fields (tuning check)

Fast gate: geo-mean encode speed-up tc-portfolio / tc-fast >= 20x and geo-mean
bpv tc-fast / tc-portfolio <= 1.10 over the same points.
"""
import csv, glob, math, statistics, sys
from collections import defaultdict
from run_bench import pareto, bpv_at

which = sys.argv[1] if len(sys.argv) > 1 else "fresh"
rows = []
files = sorted(glob.glob(f"out/{which}_*.csv"))
files = [f for f in files if "_ext_" not in f]
if which == "fresh":
    files += sorted(glob.glob("out/fresh_ext_*.csv"))
for f in files:
    for r in csv.DictReader(open(f)):
        for k in ("bpv", "psnr", "max_abs", "enc_s"):
            r[k] = float(r[k]) if r.get(k) else math.nan
        rows.append(r)
EXT = ["qtt", "sz3", "zfp", "sperr", "tthresh"]
OURS = ["tc-portfolio", "tc-fast"]
norm = lambda c: "qtt" if c.startswith("qtt") else c
targets, rels = [40, 60, 80, 100], [1e-2, 1e-3, 1e-4]
ps, mx, pick, tm = defaultdict(list), defaultdict(dict), {}, defaultdict(dict)
for r in rows:
    c = norm(r["codec"])
    if c not in EXT + OURS:
        continue
    if r["kind"] == "psnr":
        ps[(r["dataset"], c)].append(r)
    elif r["kind"] == "maxabs":
        k = (r["dataset"], float(r["bound_rel"]))
        if r["bpv"] < mx[k].get(c, math.inf):
            mx[k][c] = r["bpv"]
            if c in OURS:
                pick[(k, c)] = r["extra"].split("chose=")[1].split()[0]
                tm[(r["dataset"], f"max{float(r['bound_rel']):g}")][c] = r["enc_s"]
fields = sorted({r["dataset"] for r in rows if r["codec"] == "tc-portfolio"})
points = [f"{t}dB" for t in targets] + [f"max{r:g}" for r in rels]
val = {}
for d in fields:
    for t in targets:
        val[(d, f"{t}dB")] = {c: v for c in EXT + OURS if (d, c) in ps and (v := bpv_at(pareto(ps[(d, c)]), t)) is not None}
        # encode time at the run target t (both modes run the same targets)
        for c in OURS:
            m = [r["enc_s"] for r in ps.get((d, c), []) if abs(r["psnr"] - t) < 3]
            if m:
                tm[(d, f"{t}dB")][c] = min(m)
    for rel in rels:
        val[(d, f"max{rel:g}")] = dict(mx[(d, rel)])
gm = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs)) if xs else math.nan
out = []
P = out.append
fails, ratios = [], defaultdict(list)
if which == "fresh":
    P("| field | point | best external | bpv | tc-portfolio | ratio | pick |")
    P("|---|---|---|---|---|---|---|")
    for d in fields:
        for pt in points:
            v = val[(d, pt)]
            ext = {c: v[c] for c in EXT if c in v}
            if not ext or "tc-portfolio" not in v:
                P(f"| {d} | {pt} | – | – | {v.get('tc-portfolio', math.nan):.4g} | n/a | |")
                continue
            be = min(ext, key=ext.get)
            rp = v["tc-portfolio"] / ext[be]
            ratios[pt].append(rp)
            pk = pick.get(((d, float(pt[3:])), "tc-portfolio"), "") if pt.startswith("max") else ""
            flag = " **FAIL**" if rp > 1.05 else ""
            if rp > 1.05:
                fails.append((d, pt, be, ext[be], v["tc-portfolio"], rp, pk))
            P(f"| {d} | {pt} | {be} | {ext[be]:.4g} | {v['tc-portfolio']:.4g} | {rp:.3f}{flag} | {pk} |")
    P("\n**Geo-mean tc-portfolio / best external:** " + ", ".join(f"{pt} {gm(ratios[pt]):.3f} (n={len(ratios[pt])})" for pt in points))
    wl = defaultdict(lambda: [0, 0, 0])
    for pt, xs in ratios.items():
        k = "PSNR" if pt.endswith("dB") else "max-abs"
        for x in xs:
            wl[k][0 if x < 0.99 else 1 if x > 1.01 else 2] += 1
    P("\n**Win/loss/tie (±1%) vs best external:** " + "; ".join(f"{k} {a}–{b}–{c}" for k, (a, b, c) in wl.items()))
    n = sum(len(x) for x in ratios.values())
    P(f"\n**Quality gate (no point > 1.05× best external, {n} points):** {'PASS' if not fails else 'FAIL at ' + str(len(fails)) + ' points'}")
    for f in fails:
        P(f"- {f[0]} {f[1]}: tc-portfolio {f[4]:.4g} vs {f[2]} {f[3]:.4g} ({f[5]:.3f}×), pick={f[6]}")
# fast gate
br, sp, per = [], [], []
for d in fields:
    for pt in points:
        v, t = val[(d, pt)], tm.get((d, pt), {})
        if "tc-portfolio" in v and "tc-fast" in v and len(t) == 2:
            br.append(v["tc-fast"] / v["tc-portfolio"])
            sp.append(t["tc-portfolio"] / t["tc-fast"])
            per.append((d, pt, br[-1], sp[-1]))
P(f"\n**Fast gate ({len(br)} points):** bits tc-fast / tc-portfolio geo-mean {gm(br):.3f} (max {max(br):.2f}); "
  f"encode speed-up geo-mean {gm(sp):.1f}× (min {min(sp):.1f}×) → {'PASS' if gm(br) <= 1.10 and gm(sp) >= 20 else 'FAIL'}")
by = defaultdict(lambda: ([], []))
for d, pt, b, s in per:
    by[d][0].append(b)
    by[d][1].append(s)
P("\n| field | fast bits ratio (geo) | speed-up (geo) |\n|---|---|---|")
for d, (b, s) in by.items():
    P(f"| {d} | {gm(b):.3f} | {gm(s):.1f}× |")
et = defaultdict(list)
for r in rows:
    if r["codec"] in OURS and r["kind"] in ("psnr", "maxabs"):
        et[(r["codec"], r["kind"])].append(r["enc_s"])
P("\n**Encode time per encode (s), median / mean:** " + "; ".join(f"{c} {k}: {statistics.median(v):.3f} / {statistics.mean(v):.3f} (n={len(v)})" for (c, k), v in sorted(et.items())))
picks = defaultdict(int)
for r in rows:
    if r["codec"] in OURS:
        picks[(r["codec"], r["kind"], r["extra"].split("chose=")[1].split()[0])] += 1
P("\n**Picks:** " + ", ".join(f"{k[0]} {k[1]}/{k[2]}={n}" for k, n in sorted(picks.items())))
qtt = [r for r in rows if r["codec"] == "tc-portfolio" and "'qtt:" in r["extra"]]
P(f"\n**Full QTT tried (probe passed):** {len(qtt)} of {sum(n for k, n in picks.items() if k[0] == 'tc-portfolio')} encodes")
open(f"out/summary_{which}.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
