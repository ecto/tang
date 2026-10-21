"""Kodak RD summary + BD-rates (Bjontegaard, cubic in log-rate, PSNR domain).

Usage: python rd.py [--ref LABEL] COOLCHIC_PUBLISHED.tsv BASELINES.tsv [OURS.tsv ...]

- published Cool-chic tsv: lmbda seq_name n_pixels loss psnr_db rate_bpp
- baselines tsv (baselines.py): codec q image bpp psnr
- ours tsv: label lmbda image bpp psnr  (one or more labels per file)

Curves are dataset averages (mean bpp, mean PSNR over the images present at
every point of that curve). BD-rates are computed against the published
Cool-chic 5.0 curve restricted to the same images.
"""

import sys
from collections import defaultdict

import numpy as np


def bd_rate(r1, p1, r2, p2):
    """Average rate difference (%) of curve 2 vs curve 1 over the PSNR overlap."""
    l1, l2 = np.log(r1), np.log(r2)
    f1, f2 = np.polyfit(p1, l1, 3), np.polyfit(p2, l2, 3)
    lo, hi = max(min(p1), min(p2)), min(max(p1), max(p2))
    if hi <= lo:
        return float("nan")
    i1, i2 = np.polyint(f1), np.polyint(f2)
    a1 = np.polyval(i1, hi) - np.polyval(i1, lo)
    a2 = np.polyval(i2, hi) - np.polyval(i2, lo)
    return (np.exp((a2 - a1) / (hi - lo)) - 1) * 100


def curve(points, images):
    """points: {q: {image: (bpp, psnr)}} -> sorted (bpp[], psnr[]) means."""
    out = []
    for q, d in points.items():
        if all(i in d for i in images):
            out.append((np.mean([d[i][0] for i in images]), np.mean([d[i][1] for i in images])))
    out.sort()
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


REF = None


def main():
    global REF
    if "--ref" in sys.argv:
        i = sys.argv.index("--ref")
        REF = sys.argv[i + 1]
        del sys.argv[i : i + 2]
    pub = defaultdict(dict)
    for line in open(sys.argv[1]).read().splitlines()[1:]:
        l, name, _, _, ps, bpp = line.split("\t")
        pub[l][name] = (float(bpp), float(ps))
    curves = defaultdict(lambda: defaultdict(dict))
    for line in open(sys.argv[2]).read().splitlines()[1:]:
        c, q, name, bpp, ps = line.split("\t")
        curves[c][q][name.replace(".png", "")] = (float(bpp), float(ps))
    for f in sys.argv[3:]:
        for line in open(f).read().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            lab, l, name, bpp, ps = line.split()
            curves[lab][l][name] = (float(bpp), float(ps))
    ref = curves.get(REF) if REF else None
    for lab, pts in curves.items():
        imgs = sorted(set.intersection(*[set(d) for d in pts.values()]))
        if not imgs:
            continue
        r, p = curve(pts, imgs)
        rc, pc = curve(pub, imgs)
        bd = bd_rate(rc, pc, r, p)
        extra = ""
        if ref is not None and lab != REF:
            rr, rp = curve(ref, imgs)
            extra = f"   vs {REF}: {bd_rate(rr, rp, r, p):+6.1f} %"
        print(f"{lab:>14}  n={len(imgs):2d}  BD-rate vs Cool-chic 5.0 (published): {bd:+6.1f} %{extra}")
        for a, b in zip(r, p):
            print(f"{'':>16}{a:7.4f} bpp  {b:6.2f} dB")
    rc, pc = curve(pub, sorted(next(iter(pub.values())).keys()))
    print("Cool-chic 5.0 published (24 images):")
    for a, b in zip(rc, pc):
        print(f"{'':>16}{a:7.4f} bpp  {b:6.2f} dB")


if __name__ == "__main__":
    main()
