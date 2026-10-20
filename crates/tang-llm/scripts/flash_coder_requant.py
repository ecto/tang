#!/usr/bin/env python3
"""Requantize ISTA's Flash-Next GSQ-RCO *Coder* GGUF so the Flash engine can serve it.

The Coder release (ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF, "IQ1_M") is the Flash-Next
architecture with half the routed experts pruned (256 of 512 per layer) and its experts stored in
mixed per-tensor types (IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, Q2_0). The engine's expert path
reads Q2_0 only, so this rewrites the first shard with every `ffn_{gate,up,down}_exps` tensor as
Q2_0 and copies everything else byte for byte (the dense types are ones the engine already reads).
The second shard (the n-gram table) is not touched: link it next to the output under the matching
split name.

Q2_0 (ggml type 42): 64 weights in 18 bytes, an f16 scale `d` then 2-bit codes, weight `j` at
byte `2 + j/4`, bits `2·(j%4)`, value `(q - 1)·d`, so a block holds {-d, 0, d, 2d}. Each block's
`d` comes from a grid search over candidate scales (MSE of the block after rounding), then one
least-squares refit of `d` for the chosen codes. No imatrix: ISTA did not publish one.

    pip install numpy gguf
    flash_coder_requant.py IN-00001-of-00002.gguf OUT-00001-of-00002.gguf [--jobs N]

Header parsing and writing are done here (gguf-py's reader rejects type 42); gguf-py is used only
for its IQ dequantizers.
"""
import argparse
import mmap
import os
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

Q2_0 = 42
# ggml type -> (block elements, block bytes) for every type the Coder file holds.
BLOCK = {0: (1, 4), 1: (1, 2), 8: (32, 34), 12: (256, 144), 13: (256, 176), 14: (256, 210),
         18: (256, 98), 20: (32, 18), 21: (256, 110), 22: (256, 82), 23: (256, 136),
         30: (1, 2), 42: (64, 18)}
SCALAR = {4: "I", 5: "i", 6: "f", 7: "?", 0: "B", 1: "b", 2: "H", 3: "h", 10: "Q", 11: "q", 12: "d"}


class Reader:
    def __init__(self, buf):
        self.b, self.o = buf, 0

    def u(self, fmt):
        v = struct.unpack_from("<" + fmt, self.b, self.o)[0]
        self.o += struct.calcsize("<" + fmt)
        return v

    def s(self):
        n = self.u("Q")
        v = bytes(self.b[self.o:self.o + n]).decode()
        self.o += n
        return v

    def skip_val(self, t):
        if t == 8:
            self.s()
        elif t == 9:
            et, n = self.u("I"), self.u("Q")
            for _ in range(n):
                self.skip_val(et)
        else:
            self.u(SCALAR[t])


def parse(buf):
    r = Reader(buf)
    assert bytes(buf[:4]) == b"GGUF"
    r.o = 4
    version, n_t, n_kv = r.u("I"), r.u("Q"), r.u("Q")
    kv0 = r.o
    align = 32
    for _ in range(n_kv):
        k = r.s()
        t = r.u("I")
        if k == "general.alignment":
            align = r.u("I")
        else:
            r.skip_val(t)
    kv = bytes(buf[kv0:r.o])
    tensors = []
    for _ in range(n_t):
        name = r.s()
        dims = [r.u("Q") for _ in range(r.u("I"))]
        ty, off = r.u("I"), r.u("Q")
        tensors.append([name, dims, ty, off])
    data0 = (r.o + align - 1) // align * align
    return version, n_kv, kv, tensors, data0, align


def nbytes(dims, ty):
    n = int(np.prod(dims))
    be, bb = BLOCK[ty]
    assert n % be == 0, (dims, ty)
    return n // be * bb


def quantize_q2_0(w):
    """f32 [n] (n % 64 == 0) -> Q2_0 bytes."""
    x = w.reshape(-1, 64).astype(np.float32)
    mx, mn = x.max(1, keepdims=True), x.min(1, keepdims=True)
    base = np.maximum(mx / 2, -mn)
    base = np.where(base > 0, base, 1.0)
    best_err = np.full((x.shape[0], 1), np.inf, np.float32)
    best_d = base.copy()
    for f in np.linspace(0.45, 1.25, 33, dtype=np.float32):
        d = base * f
        q = np.clip(np.rint(x / d), -1, 2)
        e = ((x - q * d) ** 2).sum(1, keepdims=True)
        m = e < best_err
        best_err = np.where(m, e, best_err)
        best_d = np.where(m, d, best_d)
    q = np.clip(np.rint(x / best_d), -1, 2)
    qq = (q * q).sum(1, keepdims=True)
    d = np.where(qq > 0, (q * x).sum(1, keepdims=True) / np.maximum(qq, 1e-30), best_d)
    d = np.where(d > 0, d, best_d)
    d16 = d.astype(np.float16)
    df = d16.astype(np.float32)
    df = np.where(df > 0, df, 1.0)
    q = np.clip(np.rint(x / df), -1, 2).astype(np.int32) + 1
    codes = q.reshape(-1, 16, 4)
    packed = (codes[:, :, 0] | codes[:, :, 1] << 2 | codes[:, :, 2] << 4 | codes[:, :, 3] << 6).astype(np.uint8)
    out = np.empty((x.shape[0], 18), np.uint8)
    out[:, :2] = d16.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = packed
    return out.tobytes()


def dequant_q2_0(b):
    a = np.frombuffer(b, np.uint8).reshape(-1, 18)
    d = a[:, :2].copy().view(np.float16).astype(np.float32)
    c = a[:, 2:]
    q = np.stack([(c >> s) & 3 for s in (0, 2, 4, 6)], -1).reshape(-1, 64).astype(np.float32) - 1
    return (q * d).reshape(-1)


def dequant(raw, ty):
    if ty == Q2_0:
        return dequant_q2_0(raw)
    import gguf
    return gguf.quants.dequantize(np.frombuffer(raw, np.uint8), gguf.GGMLQuantizationType(ty)).reshape(-1)


def work(args):
    path, off, nb, ty = args
    with open(path, "rb") as f:
        f.seek(off)
        raw = f.read(nb)
    w = dequant(raw, ty)
    out = quantize_q2_0(w)
    back = dequant_q2_0(out)
    err = float(((w - back) ** 2).sum())
    return out, err, float((w * w).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args()

    f = open(a.src, "rb")
    buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    version, n_kv, kv, tensors, data0, align = parse(buf)

    plan = []  # (name, dims, src_type, src_off, src_bytes, dst_type, dst_bytes)
    for name, dims, ty, off in tensors:
        nb = nbytes(dims, ty)
        to = Q2_0 if "_exps." in name else ty  # Q2_0 experts copy as they are
        plan.append((name, dims, ty, data0 + off, nb, to, nbytes(dims, to)))

    hdr = bytearray(b"GGUF")
    hdr += struct.pack("<IQQ", version, len(plan), n_kv) + kv
    cur = 0
    for name, dims, _, _, _, to, db in plan:
        nb = name.encode()
        hdr += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(dims))
        hdr += b"".join(struct.pack("<Q", d) for d in dims)
        hdr += struct.pack("<IQ", to, cur)
        cur = (cur + db + align - 1) // align * align
    hdr += b"\0" * ((-len(hdr)) % align)

    t0 = time.time()
    tmp = a.dst + ".tmp"
    with open(tmp, "wb") as out, ProcessPoolExecutor(a.jobs) as pool:
        out.write(hdr)
        for i, (name, dims, ty, off, nb, to, db) in enumerate(plan):
            if to == ty:
                out.write(buf[off:off + nb])
            else:
                # One job per expert: [k, rows, n_expert] in ggml order, experts outermost.
                ne = dims[-1]
                per = nb // ne
                jobs = [(a.src, off + e * per, per, ty) for e in range(ne)]
                err = pw = 0.0
                for b, e2, p2 in pool.map(work, jobs):
                    out.write(b)
                    err += e2
                    pw += p2
                print(f"{name}: type {ty} -> Q2_0, rel MSE {err / max(pw, 1e-30):.4f} "
                      f"({time.time() - t0:.0f} s)", file=sys.stderr, flush=True)
            out.write(b"\0" * ((-db) % align))
    os.rename(tmp, a.dst)
    print(f"wrote {a.dst} in {time.time() - t0:.0f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
