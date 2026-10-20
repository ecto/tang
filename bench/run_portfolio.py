#!/usr/bin/env python3
"""Portfolio codec (tc --codec auto) runs.

  run_portfolio.py --dev        original 5 fields: every candidate, no pre-screen
                                (portfolio-all) vs screened auto, plus SZ3 at
                                the same max-abs bounds -> out/portfolio_dev.csv
  run_portfolio.py --heldout    14 held-out fields, frozen codec, screened auto
                                at PSNR 30..100 and max-rel 1e-2/1e-3/1e-4
                                -> out/portfolio_heldout_<field>.csv

Error is measured here on the decoder output; rate is the stream file size.
"""
import csv, json, sys
from run_bench import OUT, real_fields, heldout_fields, run_tc, run_sz3

MODEL = "overfit"


def add(rows, name, x, codec, r, kind, **kw):
    rng = float(x.max() - x.min())
    r = {"dataset": name, "n": x.size, "codec": codec, "kind": kind, **kw, **r}
    rows.append(r)
    print(f"  {codec:16s} {kind:6s} {kw} bpv={r['bpv']:.4f} psnr={r['psnr']:.2f} max={r['max_abs']/rng:.2e}R enc={r['enc_s']:.2f}s {r.get('extra','')}", flush=True)


def save(rows, path):
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def dev():
    rows = []
    for name, x in real_fields().items():
        print(f"== {name} {x.shape}", flush=True)
        rng = float(x.max() - x.min())
        for rel in [1e-2, 1e-3, 1e-4]:
            b = ["--max-rel", repr(rel)]
            add(rows, name, x, "tc-portfolio-all", run_tc(x, "portfolio-all", b, MODEL), "maxabs", bound_rel=rel)
            add(rows, name, x, "tc-portfolio", run_tc(x, "auto", b, MODEL), "maxabs", bound_rel=rel)
            add(rows, name, x, "sz3", run_sz3(x, "abs", rel * rng), "maxabs", bound_rel=rel)
        for p in [40, 60, 80, 100]:
            add(rows, name, x, "tc-portfolio-all", run_tc(x, "portfolio-all", ["--psnr", str(p)], MODEL), "psnr")
    save(rows, OUT / "portfolio_dev.csv")


def heldout():
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    for name, x in heldout_fields().items():
        if only and not any(o in name for o in only):
            continue
        path = OUT / f"portfolio_heldout_{name}.csv"
        if path.exists():
            continue
        rows = []
        print(f"== {name} {x.shape}", flush=True)
        for p in [30, 40, 50, 60, 70, 80, 90, 100]:
            add(rows, name, x, "tc-portfolio", run_tc(x, "auto", ["--psnr", str(p)], MODEL), "psnr")
        for rel in [1e-2, 1e-3, 1e-4]:
            add(rows, name, x, "tc-portfolio", run_tc(x, "auto", ["--max-rel", repr(rel)], MODEL), "maxabs", bound_rel=rel)
        save(rows, path)


if __name__ == "__main__":
    dev() if "--dev" in sys.argv else heldout() if "--heldout" in sys.argv else sys.exit(__doc__)
