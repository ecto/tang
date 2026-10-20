#!/usr/bin/env python3
"""Held-out validation summary (frozen codec): per-field tables, geometric-
mean bpv ratios vs SPERR and vs the best external codec, win/loss counts,
chooser vs oracle, new-app vs same-app split."""
import csv, glob, json, math
from collections import defaultdict
from run_bench import pareto, bpv_at, DATA

man = {m["name"]: m for m in json.load(open(DATA / "heldout" / "manifest.json"))}
rows = []
for f in sorted(glob.glob("out/heldout_*.csv")):
    for r in csv.DictReader(open(f)):
        for k in ("bpv", "psnr", "max_abs"):
            r[k] = float(r[k])
        rows.append(r)
EXT = ["qtt", "sz3", "zfp", "sperr", "tthresh"]
TANG = ["tc-wavelet", "tc-tucker", "tc-wavelet-overfit", "tc-wavelet-pretrained", "tc-auto"]
CODECS = EXT + TANG
norm = lambda c: "qtt" if c.startswith("qtt") else c
targets = [40, 60, 80, 100]
rels = [1e-2, 1e-3, 1e-4]
ps = defaultdict(list)
mx = defaultdict(lambda: defaultdict(lambda: math.inf))
for r in rows:
    c = norm(r["codec"])
    if c not in CODECS:
        continue
    if r["kind"] == "psnr":
        ps[(r["dataset"], c)].append(r)
    elif r["kind"] == "maxabs":
        mx[(r["dataset"], float(r["bound_rel"]))][c] = min(mx[(r["dataset"], float(r["bound_rel"]))][c], r["bpv"])
fields = sorted({r["dataset"] for r in rows}, key=lambda d: (man[d]["group"], d))
val = {}  # (field, point) -> {codec: bpv}
for d in fields:
    for t in targets:
        val[(d, f"{t}dB")] = {c: v for c in CODECS if (d, c) in ps and (v := bpv_at(pareto(ps[(d, c)]), t)) is not None}
    for rel in rels:
        val[(d, f"max{rel:g}")] = {c: v for c, v in mx[(d, rel)].items() if math.isfinite(v)}
points = [f"{t}dB" for t in targets] + [f"max{r:g}" for r in rels]

def gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")

out = []
P = out.append
P("### Per-field bpv (matched measured PSNR / verified max-abs; — = not reached on the codec's front)\n")
P("| field | group | point | " + " | ".join(CODECS) + " | best external | tc-auto vs best ext |")
P("|---|---|---|" + "---|" * (len(CODECS) + 2))
for d in fields:
    for pt in points:
        v = val[(d, pt)]
        ext = {c: v[c] for c in EXT if c in v}
        be = min(ext, key=ext.get) if ext else None
        rat = f"{v['tc-auto'] / ext[be]:.3f}" if be and "tc-auto" in v else "—"
        P(f"| {d} | {man[d]['group']} | {pt} | " + " | ".join(f"{v[c]:.4g}" if c in v else "—" for c in CODECS) + f" | {be or '—'} | {rat} |")

P("\n### Aggregate: geometric-mean bpv ratio (codec / reference) over fields where both are defined\n")
for group in ["all", "new-app", "same-app"]:
    fs = [d for d in fields if group == "all" or man[d]["group"] == group]
    P(f"\n**{group}** ({len(fs)} fields)\n")
    P("| codec | ref | " + " | ".join(points) + " |")
    P("|---|---|" + "---|" * len(points))
    for c in TANG:
        for ref in ["sperr", "best-ext"]:
            cells = []
            for pt in points:
                xs = []
                for d in fs:
                    v = val[(d, pt)]
                    if c not in v:
                        continue
                    r = v.get("sperr") if ref == "sperr" else min([v[e] for e in EXT if e in v], default=None)
                    if r:
                        xs.append(v[c] / r)
                cells.append(f"{gm(xs):.3f} (n={len(xs)})" if xs else "—")
            P(f"| {c} | {ref} | " + " | ".join(cells) + " |")

P("\n### Win/loss: tc-auto vs best external codec per (field, point)\n")
P("| group | points | wins | losses | ties (±1%) |")
P("|---|---|---|---|---|")
for group in ["all", "new-app", "same-app"]:
    for kind, pts in [("PSNR", points[:4]), ("max-abs", points[4:])]:
        w = l = t = 0
        for d in fields:
            if group != "all" and man[d]["group"] != group:
                continue
            for pt in pts:
                v = val[(d, pt)]
                ext = [v[e] for e in EXT if e in v]
                if "tc-auto" not in v or not ext:
                    continue
                r = v["tc-auto"] / min(ext)
                if r < 0.99:
                    w += 1
                elif r > 1.01:
                    l += 1
                else:
                    t += 1
        P(f"| {group} | {kind} | {w} | {l} | {t} |")

P("\n### Losses of tc-auto (ratio > 1.01 vs best external)\n")
for d in fields:
    for pt in points:
        v = val[(d, pt)]
        ext = {e: v[e] for e in EXT if e in v}
        if "tc-auto" in v and ext:
            be = min(ext, key=ext.get)
            if v["tc-auto"] / ext[be] > 1.01:
                P(f"- {d} {pt}: tc-auto {v['tc-auto']:.4g} vs {be} {ext[be]:.4g} ({v['tc-auto'] / ext[be]:.2f}×)")

P("\n### Chooser vs oracle\n")
picks = defaultdict(int)
for r in rows:
    if r["codec"] == "tc-auto":
        picks["tucker" if "chose=tucker" in r.get("extra", "") else "wavelet"] += 1
P(f"Per-encode picks: {dict(picks)}.\n")
for oracle_set, label in [(["tc-wavelet-overfit", "tc-tucker"], "its two candidates (wavelet-overfit, Tucker)"),
                          (TANG[:-1], "all four tang codecs")]:
    xs = []
    hit = 0
    for d in fields:
        for pt in points:
            v = val[(d, pt)]
            o = [v[c] for c in oracle_set if c in v]
            if "tc-auto" in v and o:
                xs.append(v["tc-auto"] / min(o))
                hit += v["tc-auto"] <= min(o) * 1.01
    P(f"- vs oracle over {label}: geo-mean ratio {gm(xs):.4f}, within 1% of oracle at {hit}/{len(xs)} points, worst {max(xs):.3f}×")
open("out/summary_heldout.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
