#!/usr/bin/env python3
"""Check `tang-llm gguf-rows` dequantization against gguf-py (llama.cpp's own Python reader).

For one tensor of each quantization type in the given GGUF shards, dequantize the first rows
with gguf-py and with tang, and report the max abs difference. gguf-py can't represent ggml type
42 (ISTA's Q2_0), so Q2_0 is covered by tang's unit test and by end-to-end parity instead.

    pip install gguf numpy
    flash_gguf_check.py --tang ./target/release/tang-llm --rows 4 shard1.gguf [shard2.gguf ...]

(The tang binary must be able to open the first shard; it finds the others itself.)
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
from gguf import GGUFReader, GGMLQuantizationType as Q
from gguf.quants import dequantize
import gguf.constants as C

# Teach gguf-py the size of type 42 so it can *open* ISTA's files (it still can't dequantize it).
if 42 not in C.GGMLQuantizationType._value2member_map_:
    _q2_0 = int.__new__(C.GGMLQuantizationType, 42)
    _q2_0._name_, _q2_0._value_ = "Q2_0", 42
    C.GGMLQuantizationType._value2member_map_[42] = _q2_0
    C.GGML_QUANT_SIZES[_q2_0] = (64, 18)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tang", required=True)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("shards", nargs="+")
    a = ap.parse_args()
    seen = {}
    for path in a.shards:
        try:
            r = GGUFReader(path)
        except Exception as e:  # gguf-py refuses files with types it doesn't know (Q2_0)
            print(f"{path}: gguf-py can't open it ({e})", file=sys.stderr)
            continue
        for t in r.tensors:
            name = t.tensor_type.name
            if name in seen or name == "Q2_0":
                continue
            seen[name] = (path, t)
    worst = 0.0
    with tempfile.TemporaryDirectory() as d:
        for qname, (path, t) in sorted(seen.items()):
            row_len = int(t.shape[0])
            n_rows = int(np.prod([int(x) for x in t.shape[1:]])) if len(t.shape) > 1 else 1
            raw = np.asarray(t.data).reshape(-1).view(np.uint8)
            rb = raw.size // n_rows
            k = min(a.rows, n_rows)
            rows = np.ascontiguousarray(raw[: k * rb]).reshape(k, rb)
            if t.tensor_type == Q.F32:
                ref = rows.view(np.float32).reshape(-1)
            elif t.tensor_type == Q.F16:
                ref = rows.view(np.float16).astype(np.float32).reshape(-1)
            else:
                ref = dequantize(rows, t.tensor_type).astype(np.float32).reshape(-1)
            n = ref.size // row_len
            out = os.path.join(d, "rows.f32")
            subprocess.run([a.tang, "gguf-rows", path, t.name, "0", str(n), out],
                           check=True, stderr=subprocess.DEVNULL)
            got = np.fromfile(out, dtype=np.float32)
            if got.size != ref.size:
                print(f"{qname:8s} {t.name}: size {got.size} vs {ref.size}")
                worst = float("inf")
                continue
            diff = float(np.max(np.abs(got - ref)))
            scale = float(np.max(np.abs(ref))) or 1.0
            worst = max(worst, diff / scale)
            print(f"{qname:8s} {t.name:40s} {n} rows x {row_len}: max |tang - gguf-py| = {diff:.3g} (max |x| {scale:.3g})")
    print(f"worst relative difference: {worst:.3g}")
    sys.exit(0 if worst < 1e-6 else 1)


if __name__ == "__main__":
    main()
