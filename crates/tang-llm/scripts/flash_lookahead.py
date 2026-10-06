#!/usr/bin/env python3
"""Analyse `tang-llm flash-ref --lookahead FILE` records: how well layer l-1's state predicts
layer l's routed experts, and whether predicted experts cover the cache misses.

    flash_lookahead.py a.bin [b.bin ...] [--hot 0.5] [--profile p.bin,q.bin] [--per-layer]

Each file: one JSON header line, then u16 records [layer][token][k true + 4 methods x P predicted]
(methods: x_l, r_mid, r_mid+mean_moe, r_post; layer 0 has no prediction).

Reports, per method and predicted-set size m in (10, 16, 24, 32):
  recall    = |true top-k ∩ predicted top-m| / k, over layers 1..47 and all tokens;
  miss recall = the same restricted to true experts that are NOT resident, where the resident
              set is the hottest `--hot` fraction of all (layer, expert) pairs by routing count
              over the `--profile` files (default: the evaluated files themselves, which is
              optimistic for short sequences);
  false prefetches = predicted experts per (token, layer) that are non-resident and not used.
"""
import json
import sys

import numpy as np


def load(path):
    with open(path, "rb") as f:
        hdr = json.loads(f.readline())
        data = np.frombuffer(f.read(), dtype="<u2")
    L, T, k, P = hdr["layers"], hdr["tokens"], hdr["k"], hdr["pred"]
    rec = data.reshape(L, T, k + 4 * P).astype(np.int32)
    true = rec[:, :, :k]
    pred = rec[:, :, k:].reshape(L, T, 4, P)
    return hdr, true, pred


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    hot = 0.5
    if "--hot" in sys.argv:
        hot = float(sys.argv[sys.argv.index("--hot") + 1])
        args.remove(sys.argv[sys.argv.index("--hot") + 1])
    per_layer = "--per-layer" in sys.argv
    profile = None
    if "--profile" in sys.argv:
        profile = sys.argv[sys.argv.index("--profile") + 1].split(",")
        args.remove(sys.argv[sys.argv.index("--profile") + 1])
    runs = [load(p) for p in args]
    prof_runs = [load(p) for p in profile] if profile else runs
    hdr = runs[0][0]
    L, ne = hdr["layers"], 512
    counts = np.zeros((L, ne), dtype=np.int64)
    for _, true, _ in prof_runs:
        for l in range(L):
            np.add.at(counts[l], true[l].ravel(), 1)
    # the evaluated sequences' own routes, for the hit rate of that resident set
    own = np.zeros((L, ne), dtype=np.int64)
    for _, true, _ in runs:
        for l in range(L):
            np.add.at(own[l], true[l].ravel(), 1)
    flat = np.argsort(-counts.ravel(), kind="stable")
    resident = np.zeros(L * ne, dtype=bool)
    resident[flat[: int(hot * L * ne)]] = True
    resident = resident.reshape(L, ne)
    hit = sum(own[l][resident[l]].sum() for l in range(L)) / own.sum()
    print(f"files {args}; resident = hottest {hot:.0%} of (layer, expert) pairs by {profile or 'the same files'}; hit rate {hit:.3f}")
    names = hdr["methods"]
    print("method | m | recall | miss recall | false prefetches / token-layer | misses / token-layer")
    per = {}
    for mi, name in enumerate(names):
        for m in (10, 16, 24, 32):
            tp = tot = mtp = mtot = fp = cells = 0
            lay = []
            for _, true, pred in runs:
                T = true.shape[1]
                for l in range(1, L):
                    t_ = true[l]
                    p_ = pred[l, :, mi, :m]
                    inter = (t_[:, :, None] == p_[:, None, :]).any(axis=2)  # [T, k]
                    res_t = resident[l][t_]
                    tp_l = inter.sum()
                    miss = ~res_t
                    mtp_l = (inter & miss).sum()
                    mtot_l = miss.sum()
                    used = (p_[:, :, None] == t_[:, None, :]).any(axis=2)  # [T, m]
                    fp_l = ((~resident[l][p_]) & ~used).sum()
                    tp += tp_l
                    tot += t_.size
                    mtp += mtp_l
                    mtot += mtot_l
                    fp += fp_l
                    cells += T
                    lay.append((l, tp_l / t_.size, mtp_l / max(mtot_l, 1)))
            per[(name, m)] = lay
            print(f"{name} | {m} | {tp / tot:.3f} | {mtp / max(mtot, 1):.3f} | {fp / cells:.2f} | {mtot / cells:.2f}")
    if per_layer:
        for name in names:
            row = per[(name, 16)]
            print(name, "recall@16 by layer:", " ".join(f"{l}:{r:.2f}" for l, r, _ in row))


if __name__ == "__main__":
    main()
