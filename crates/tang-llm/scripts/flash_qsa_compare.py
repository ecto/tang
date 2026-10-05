#!/usr/bin/env python3
"""Compare QSA block selection: llama.cpp (flash_qsa_topk.cpp's topk-/score-<layer>.txt) against
`tang-llm flash-ref --dump` (LNN.qsa_selected.u32), for the last token of the same prompt.

    flash_qsa_compare.py <llama dir> <tang dump dir>

Per QSA layer: complete blocks tang selected, blocks llama.cpp selected (live ones: score > -inf),
the overlap, and for any disagreement where the swapped blocks sit in llama.cpp's own score
ranking relative to the cut (a near-tie at the 512th place is expected noise; a far miss is a bug).
"""
import glob
import json
import os
import struct
import sys


def read_all(path):
    vals = []
    for line in open(path):
        if line.startswith("#"):
            continue
        vals.append(float(line))
    return vals


K_TOP = 512  # top_k / kpool


def main():
    ldir, tdir = sys.argv[1], sys.argv[2]
    idx = json.load(open(os.path.join(tdir, "index.json")))
    n_kv = idx["position"] + 1
    kp = 4
    n_blocks = n_kv // kp
    out = []
    for f in sorted(glob.glob(os.path.join(ldir, "topk-*.txt")), key=lambda p: int(p.rsplit("-", 1)[1][:-4])):
        layer = int(f.rsplit("-", 1)[1][:-4])
        # the last ubatch, every token: take the last token's slice
        top_all = read_all(f)
        n_tok = len(top_all) // K_TOP
        top = [int(v) for v in top_all[-K_TOP:]]
        score_all = read_all(os.path.join(ldir, f"score-{layer}.txt"))
        per = len(score_all) // n_tok
        score = score_all[-per:]
        live = {b for b in top if b < n_blocks and score[b] != float("-inf")}
        raw = open(os.path.join(tdir, f"L{layer:02d}.qsa_selected.u32"), "rb").read()
        cells = struct.unpack(f"<{len(raw) // 4}I", raw)
        ours = {c // kp for c in cells if c // kp < n_blocks}
        rank = sorted(range(n_blocks), key=lambda b: -score[b])
        pos = {b: i for i, b in enumerate(rank)}
        only_l = sorted(live - ours)
        only_t = sorted(ours - live)
        cut = score[rank[len(live) - 1]] if live else 0.0
        out.append({
            "layer": layer,
            "tang_blocks": len(ours),
            "llama_blocks": len(live),
            "overlap": len(ours & live),
            "only_llama": [(b, pos[b], round(score[b], 4)) for b in only_l[:6]],
            "only_tang": [(b, pos[b], round(score[b], 4)) for b in only_t[:6]],
            "llama_cut_score": round(cut, 4),
        })
    for o in out:
        print(json.dumps(o))


if __name__ == "__main__":
    main()
