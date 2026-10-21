#!/usr/bin/env python3
"""Phase-2 portfolio runs (frozen screen v2 + `--fast`).

  run_fresh.py --seen    the 19 seen fields (5 originals + 14 phase-1 held-out):
                         tc auto and tc --fast -> out/seen_<field>.csv (tuning check)
  run_fresh.py --fresh   the fresh held-out set (fetch_fresh.py): same runs
                         -> out/fresh_<field>.csv (Phase-2 gate)
  run_fresh.py --fresh3  the Phase-4 held-out set (fetch_fresh3.py)
                         -> out/fresh3_<field>.csv (Phase-4 gate)

Targets: PSNR 30..100 dB and max-rel 1e-2/1e-3/1e-4. auto uses the overfit
wavelet model, --fast the static (hand) model. Error is measured on the
decoder output; rate is the stream file size. Set TC_BIN to the frozen binary.
"""
import sys
from run_bench import OUT, real_fields, heldout_fields, fresh_fields, run_tc
from run_portfolio import add, save


def run(fields, prefix):
    for name, x in fields.items():
        path = OUT / f"{prefix}_{name}.csv"
        if path.exists():
            continue
        rows = []
        print(f"== {name} {x.shape}", flush=True)
        tg = [(["--psnr", str(p)], "psnr", {}) for p in [30, 40, 50, 60, 70, 80, 90, 100]]
        tg += [(["--max-rel", repr(r)], "maxabs", {"bound_rel": r}) for r in [1e-2, 1e-3, 1e-4]]
        for b, kind, kw in tg:
            add(rows, name, x, "tc-portfolio", run_tc(x, "auto", b, "overfit"), kind, **kw)
            add(rows, name, x, "tc-fast", run_tc(x, "fast", b, "hand"), kind, **kw)
        save(rows, path)


if __name__ == "__main__":
    if "--seen" in sys.argv:
        run({**real_fields(), **heldout_fields()}, "seen")
    elif "--fresh3" in sys.argv:
        run(fresh_fields("fresh3"), "fresh3")
    elif "--fresh" in sys.argv:
        run(fresh_fields(), "fresh")
    else:
        sys.exit(__doc__)
