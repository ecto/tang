#!/usr/bin/env python3
"""Quick look: Adam state (exp_avg m, exp_avg_sq v) as the next tensor type.

Metric that matters is the Adam *update* u = m / (sqrt(v) + eps): we report
relL2 of u when m or v alone is compressed. Codecs (ideal order-0 entropy of
the quantized symbols + side info, no real coder):
  - ecsq-lin:   uniform step on the raw tensor
  - ecsq-log:   uniform step on log v (relative error in v), m: on m / sqrt(v_hat)
  - r1log:      log v = a_i + b_j + r (Adafactor-style rank-1 in the log
                domain, a/b at f16) + uniform step on r with per-row context
  - lowrank+res (m): rank-k SVD of m (f16 factors) + uniform residual
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tensors_codecs import h0  # noqa: E402
from tensors_io import T  # noqa: E402

EPS = 1e-8


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(a))


def ctx_h(q, axis_rows):
    """Per-row-context ideal entropy (bits/value)."""
    return float(np.mean([h0(r) for r in q])) if axis_rows else h0(q)


def main():
    rows = []
    for fm in sorted((T / "grad").glob("adam_m_*.npy")):
        m = np.load(fm).astype(np.float64)
        v = np.load(str(fm).replace("adam_m_", "adam_v_")).astype(np.float64)
        m, v = m[:512], v[:512]  # crop rows for speed
        u = m / (np.sqrt(v) + EPS)
        name = fm.stem.replace("adam_m_model.layers.", "L").replace(".weight", "")
        lv = np.log(v + 1e-30)
        for s in np.geomspace(0.005, 1.0, 10):
            # v, linear domain: step s^2 * rms(v) (s sweeps both)
            st = s * s * np.sqrt(np.mean(v**2))
            qv = np.round(v / st)
            vh = np.maximum(qv * st, 0)
            rows.append((name, "v ecsq-lin", s, h0(qv), rel(u, m / (np.sqrt(vh) + EPS))))
            ql = np.round(lv / s)
            rows.append((name, "v ecsq-log", s, h0(ql), rel(u, m / (np.sqrt(np.exp(ql * s)) + EPS))))
            a = lv.mean(1, keepdims=True)
            b = (lv - a).mean(0, keepdims=True)
            a16, b16 = a.astype(np.float16).astype(np.float64), b.astype(np.float16).astype(np.float64)
            r = lv - a16 - b16
            qr = np.round(r / s)
            side = 16 * (lv.shape[0] + lv.shape[1]) / lv.size
            rows.append((name, "v r1log", s, h0(qr) + side, rel(u, m / (np.sqrt(np.exp(qr * s + a16 + b16)) + EPS))))
            rows.append((name, "v r1log+rowctx", s, ctx_h(qr, True) + side, rel(u, m / (np.sqrt(np.exp(qr * s + a16 + b16)) + EPS))))
        sv = np.sqrt(v) + EPS
        for s in np.geomspace(0.003, 1.0, 10):
            st = s * np.sqrt(np.mean(m**2))
            qm = np.round(m / st)
            rows.append((name, "m ecsq-lin", s, h0(qm), rel(u, qm * st / sv)))
            # quantize in the update domain: m/sqrt(v) with the step scaled by sqrt(v) (needs v at decode)
            uu = m / sv
            su = s * np.sqrt(np.mean(uu**2))
            qu = np.round(uu / su)
            rows.append((name, "m ecsq-in-update-domain", s, h0(qu), rel(u, qu * su)))
            for k in (4, 16):
                U, S, Vt = np.linalg.svd(m, full_matrices=False)
                lr = (U[:, :k] * S[:k]) @ Vt[:k]
                lr = lr.astype(np.float16).astype(np.float64)
                res = m - lr
                q = np.round(res / st)
                side = 16 * k * (m.shape[0] + m.shape[1]) / m.size
                rows.append((name, f"m rank{k}+res", s, h0(q) + side, rel(u, (lr + q * st) / sv)))
    import math

    print("| codec | bpv @ update relL2 1% | @ 5% | @ 10% | n |\n|---|---|---|---|---|")
    from tensors_bench import at_rel

    for c in dict.fromkeys(r[1] for r in rows):
        per = {}
        for r in rows:
            if r[1] == c:
                per.setdefault(r[0], []).append(dict(bpv=r[3], rel=r[4]))
        cells = []
        for t in (0.01, 0.05, 0.10):
            a = [at_rel(p, t) for p in per.values()]
            ok = [x for x in a if math.isfinite(x)]
            cells.append(f"{np.median(ok):.2f} ({len(ok)}/{len(a)})" if ok else "—")
        print(f"| {c} | " + " | ".join(cells) + f" | {len(per)} |")


if __name__ == "__main__":
    main()
