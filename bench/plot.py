#!/usr/bin/env python3
"""Rate–distortion plots + matched-error tables from bench/out/results_*.csv."""
from __future__ import annotations

import csv
import glob
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_bench import bpv_at, pareto  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
STYLE = {
    "qtt-morton": ("#c0392b", "-", "o"),
    "qtt-row": ("#e67e22", "--", "o"),
    "sz3": ("#2471a3", "-", "s"),
    "zfp": ("#27ae60", "-", "^"),
    "wavelet-H0": ("#7d3c98", ":", "d"),
    "wavelet-zstd": ("#a569bd", "-.", "d"),
    "sperr": ("#17202a", "-", "P"),
    "tthresh": ("#7f8c8d", "-", "X"),
}


def load():
    rows = []
    for f in sorted(glob.glob(str(OUT / "results_*.csv"))):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                for k in ("bpv", "psnr", "max_abs", "rmse", "enc_s", "dec_s", "n"):
                    r[k] = float(r[k])
                rows.append(r)
    # later files override earlier ones for the same (dataset, codec, kind, extra)
    dedup = {}
    for r in rows:
        dedup[(r["dataset"], r["codec"], r["kind"], r.get("extra", ""), r.get("bound_rel", ""))] = r
    return list(dedup.values())


def main():
    rows = load()
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["kind"] == "psnr":
            by[r["dataset"]][r["codec"]].append(r)
    md = []
    targets = [40, 60, 80, 100]
    md.append("| dataset | codec | " + " | ".join(f"bpv@{t}dB" for t in targets) + " |")
    md.append("|---|---|" + "---|" * len(targets))
    for ds in sorted(by):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for codec in sorted(by[ds]):
            front = pareto(by[ds][codec])
            c, ls, mk = STYLE.get(codec, ("k", "-", "x"))
            ax.plot([r["bpv"] for r in front], [r["psnr"] for r in front], ls, marker=mk, color=c, label=codec, ms=4)
            vals = []
            for t in targets:
                b = bpv_at(front, t)
                vals.append(f"{b:.4g}" if b is not None else "–")
            md.append(f"| {ds} | {codec} | " + " | ".join(vals) + " |")
        ax.set_xscale("log")
        ax.set_xlabel("bits per value (log)")
        ax.set_ylabel("PSNR (dB, range-normalised)")
        ax.set_title(ds, fontsize=10)
        ax.set_ylim(20, 125)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / f"rd_{ds.replace('^', '')}.png", dpi=120)
        plt.close(fig)
    md.append("")
    md.append("| dataset | bound (×range) | codec | bpv | max err (×range) | PSNR | enc s | dec s | notes |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    mx = [r for r in rows if r["kind"] == "maxabs"]
    for r in sorted(mx, key=lambda r: (r["dataset"], -float(r["bound_rel"]), r["codec"])):
        rel = float(r["bound_rel"])
        md.append(
            f"| {r['dataset']} | {rel:g} | {r['codec']} | {r['bpv']:.4g} | {r['max_abs']:.3g} abs | {r['psnr']:.1f} | {r['enc_s']:.2f} | {r['dec_s']:.3f} | {r.get('extra','')} |"
        )
    md.append("")
    # speeds at ~60 dB
    md.append("| dataset | codec | PSNR | bpv | enc MB/s | dec MB/s |")
    md.append("|---|---|---|---|---|---|")
    for ds in sorted(by):
        for codec in sorted(by[ds]):
            pts = sorted(by[ds][codec], key=lambda r: abs(r["psnr"] - 60))
            if pts:
                r = pts[0]
                mb = 4 * r["n"] / 1e6
                md.append(f"| {ds} | {codec} | {r['psnr']:.1f} | {r['bpv']:.4g} | {mb / r['enc_s']:.2f} | {mb / r['dec_s']:.1f} |")
    md.append("")
    md.append("| dataset | QTT order | PSNR target | ranks |")
    md.append("|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["dataset"], r["codec"], r["psnr"])):
        if r["codec"].startswith("qtt") and r["kind"] == "psnr" and r.get("ranks"):
            md.append(f"| {r['dataset']} | {r['codec']} | {r['psnr']:.1f} | {r['ranks']} |")
    # frontier: best codec per dataset/target, excluding the idealized wavelet-H0
    md.append("")
    md.append("| dataset | PSNR | frontier codec | bpv | QTT-row bpv | QTT / frontier | runner-up |")
    md.append("|---|---|---|---|---|---|---|")
    for ds in sorted(by):
        for t in [40, 60, 80, 100]:
            vals = {}
            for codec in by[ds]:
                if codec in ("wavelet-H0", "qtt-morton"):
                    continue
                b = bpv_at(pareto(by[ds][codec]), t)
                if b is not None:
                    vals[codec] = b
            if not vals:
                continue
            order = sorted(vals, key=vals.get)
            q = vals.get("qtt-row")
            ru = f"{order[1]} {vals[order[1]]:.4g}" if len(order) > 1 else "–"
            md.append(f"| {ds} | {t} | {order[0]} | {vals[order[0]]:.4g} | {q:.4g} | {q / vals[order[0]]:.2f}× |  {ru} |" if q else f"| {ds} | {t} | {order[0]} | {vals[order[0]]:.4g} | – | – | {ru} |")
    md.append("")
    md.append("| dataset | SPERR omp=1 enc/dec s | SPERR all-threads enc/dec s (60 dB) |")
    md.append("|---|---|---|")
    sp = [r for r in rows if r["codec"] == "sperr-mt"]
    for r in sp:
        one = min((x for x in rows if x["codec"] == "sperr" and x["dataset"] == r["dataset"] and x.get("extra", "").startswith("psnr=60")), key=lambda x: 0, default=None)
        if one:
            md.append(f"| {r['dataset']} | {one['enc_s']:.3f} / {one['dec_s']:.3f} | {r['enc_s']:.3f} / {r['dec_s']:.3f} |")
    (OUT / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:60]))


if __name__ == "__main__":
    main()
