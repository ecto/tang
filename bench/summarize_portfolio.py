#!/usr/bin/env python3
"""Held-out gate for the portfolio codec: tc-portfolio (frozen) vs the best
external codec (QTT row, SZ3, ZFP, SPERR, TTHRESH; rows from the M1 held-out
run in out/ext/) at 40/60/80/100 dB and max-abs 1e-2/1e-3/1e-4. Gate: no
point worse than the best external by more than 5%."""
import csv, glob, json, math, statistics
from collections import defaultdict
from run_bench import pareto, bpv_at, DATA

man = {m["name"]: m for m in json.load(open(DATA / "heldout" / "manifest.json"))}
rows = []
for f in sorted(glob.glob("out/ext/heldout_*.csv")) + sorted(glob.glob("out/portfolio_heldout_*.csv")):
    for r in csv.DictReader(open(f)):
        for k in ("bpv", "psnr", "max_abs", "enc_s"):
            r[k] = float(r[k]) if r.get(k) else math.nan
        rows.append(r)
EXT = ["qtt", "sz3", "zfp", "sperr", "tthresh"]
OURS = ["tc-auto", "tc-portfolio"]
norm = lambda c: "qtt" if c.startswith("qtt") else c
targets, rels = [40, 60, 80, 100], [1e-2, 1e-3, 1e-4]
ps = defaultdict(list)
mx = defaultdict(dict)
pick = {}
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
            if c == "tc-portfolio":
                pick[k] = r["extra"].split("chose=")[1].split()[0]
fields = sorted({r["dataset"] for r in rows if r["codec"] == "tc-portfolio"}, key=lambda d: (man[d]["group"], d))
points = [f"{t}dB" for t in targets] + [f"max{r:g}" for r in rels]
val = {}
for d in fields:
    for t in targets:
        val[(d, f"{t}dB")] = {c: v for c in EXT + OURS if (d, c) in ps and (v := bpv_at(pareto(ps[(d, c)]), t)) is not None}
    for rel in rels:
        val[(d, f"max{rel:g}")] = dict(mx[(d, rel)])

gm = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs)) if xs else math.nan
out = []
P = out.append
P("| field | point | best external | bpv | tc-auto (M1) | ratio | tc-portfolio | ratio | pick |")
P("|---|---|---|---|---|---|---|---|---|")
fails, ratios = [], defaultdict(list)
for d in fields:
    for pt in points:
        v = val[(d, pt)]
        ext = {c: v[c] for c in EXT if c in v}
        if not ext or "tc-portfolio" not in v:
            continue
        be = min(ext, key=ext.get)
        ra = v["tc-auto"] / ext[be] if "tc-auto" in v else math.nan
        rp = v["tc-portfolio"] / ext[be]
        ratios[pt].append(rp)
        pk = pick.get((d, float(pt[3:])), "") if pt.startswith("max") else ""
        flag = " **FAIL**" if rp > 1.05 else ""
        if rp > 1.05:
            fails.append((d, pt, be, ext[be], v["tc-portfolio"], rp, pk))
        P(f"| {d} | {pt} | {be} | {ext[be]:.4g} | {v.get('tc-auto', math.nan):.4g} | {ra:.3f} | {v['tc-portfolio']:.4g} | {rp:.3f}{flag} | {pk} |")
P("\n**Geo-mean tc-portfolio / best external:** " + ", ".join(f"{pt} {gm(ratios[pt]):.3f} (n={len(ratios[pt])})" for pt in points))
wl = defaultdict(lambda: [0, 0, 0])
for pt, xs in ratios.items():
    k = "PSNR" if pt.endswith("dB") else "max-abs"
    for x in xs:
        wl[k][0 if x < 0.99 else 1 if x > 1.01 else 2] += 1
P("\n**Win/loss/tie (±1%) vs best external:** " + "; ".join(f"{k} {a}–{b}–{c}" for k, (a, b, c) in wl.items()))
P(f"\n**Gate (no point > 1.05× best external):** {'PASS' if not fails else 'FAIL at ' + str(len(fails)) + ' points'}")
for f in fails:
    P(f"- {f[0]} {f[1]}: tc-portfolio {f[4]:.4g} vs {f[2]} {f[3]:.4g} ({f[5]:.3f}×), pick={f[6]}")
# encode time
et = defaultdict(list)
for r in rows:
    if r["codec"] in OURS and r["dataset"] in fields and r["kind"] in ("psnr", "maxabs"):
        if r["kind"] == "psnr" and float(r["psnr"]) < 35:
            pass
        et[(r["codec"], r["kind"])].append(r["enc_s"])
P("\n**Encode time per encode (s), median / mean:** " + "; ".join(f"{c} {k}: {statistics.median(v):.2f} / {statistics.mean(v):.2f} (n={len(v)})" for (c, k), v in sorted(et.items())))
picks = defaultdict(int)
for r in rows:
    if r["codec"] == "tc-portfolio":
        picks[(r["kind"], r["extra"].split("chose=")[1].split()[0])] += 1
P("\n**Picks:** " + ", ".join(f"{k[0]}/{k[1]}={n}" for k, n in sorted(picks.items())))
qtt_tried = sum(1 for r in rows if r["codec"] == "tc-portfolio" and "qtt:" in r["extra"])
P(f"\n**QTT tried (pre-screen passed):** {qtt_tried} of {sum(picks.values())} encodes; peakiness per field: " +
  ", ".join(f"{d} {float(next(r['extra'] for r in rows if r['dataset'] == d and r['codec'] == 'tc-portfolio').split('peak=')[1].split()[0]):.1f}" for d in fields))
open("out/summary_portfolio.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
