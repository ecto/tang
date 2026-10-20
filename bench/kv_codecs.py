"""KV-cache codecs for kv_ppl.py / kv_bench.py: (x [heads, T, hd] f32,
is_key) -> (bits per value, x_hat).

Baselines reuse tensors_codecs (KIVI-style asym groups, SZ3, int per-row).
`proto-*` is the numpy prototype of the tang KV path (ideal per-context
entropy, used to pick the design); `tkv-*` runs the real Rust codec
(examples/kvc) and counts its actual bytes.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from functools import partial
from pathlib import Path

import numpy as np

import tensors_codecs as C

KVC = os.environ.get("KVC_BIN", str(Path(__file__).resolve().parent.parent / "target/release/examples/kvc"))
THETA, HD = 1e5, 64
INV = 1.0 / THETA ** (np.arange(0, HD, 2) / HD)


def rope(x: np.ndarray, sign: float = 1.0, t0: int = 0) -> np.ndarray:
    """HF-Llama half-split RoPE on [..., T, hd]; sign=-1 inverts it."""
    t = (t0 + np.arange(x.shape[-2]))[:, None] * INV[None]
    c = np.cos(np.concatenate([t, t], -1))
    s = sign * np.sin(np.concatenate([t, t], -1))
    h = x.shape[-1] // 2
    return x * c + np.concatenate([-x[..., h:], x[..., :h]], -1) * s


def _rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


# ------------------------------------------------------------- baselines
def kivi(x, is_key, bits, g=32):
    h, t, d = x.shape
    if is_key:  # per-channel groups of g tokens (KIVI keys)
        a = x.transpose(0, 2, 1).reshape(-1, g)
        _, y = C._asym(a, bits)
        y = y.reshape(h, d, t).transpose(0, 2, 1)
    else:  # per-token groups of g channels (KIVI values)
        _, y = C._asym(x.reshape(-1, g), bits)
        y = y.reshape(h, t, d)
    return bits + 32 / g, y


def sz3(x, is_key, rel):
    return C.sz3(x, rel * _rms(x))


def introw(x, is_key, bits):
    b, _, y = C.quant_rows(x, "kv", bits)
    return b, y


# ------------------------------------------------------------- prototype
def h0_cols(q: np.ndarray) -> float:
    """sum over columns (contexts) of n * order-0 entropy; q [n, ncols]."""
    tot = 0.0
    for j in range(q.shape[1]):
        _, c = np.unique(q[:, j], return_counts=True)
        p = c / c.sum()
        tot += float(-(c * np.log2(p)).sum())
    return tot


def proto(x, is_key, rel, alpha=0.0, pre=True, rho=True, ctx="chan", block=10**9, sink=0, sinkdiv=1.0, beta=0.0):
    """Closed-loop DPCM along tokens per (head, channel) around the channel
    mean, uniform step d_c = d * w_c, w_c = (pair sd / mean sd)^alpha.
    Ideal static per-context entropy + 28-bit/(head,channel) header."""
    h, t, d = x.shape
    y = rope(x.astype(np.float64), -1.0) if (is_key and pre) else x.astype(np.float64)
    mu = y.mean(1)  # [h, d]
    sd = y.std(1)
    if is_key:  # RoPE pairs share a step
        ps = np.sqrt((sd[:, : d // 2] ** 2 + sd[:, d // 2 :] ** 2) / 2)
        sd = np.concatenate([ps, ps], -1)
    w = (sd / sd.mean()) ** alpha
    step = rel * _rms(x) * np.sqrt(12) * w  # [h, d]
    tw = np.ones((h, t))
    side = 0
    if beta and not is_key:  # per-token step ~ (token rms / median)^beta, 8-bit log code
        nr = np.sqrt((x.astype(np.float64) ** 2).mean(-1))
        tw = np.minimum((nr / np.median(nr, 1, keepdims=True)) ** beta, 1.0)
        tw = 2.0 ** (np.round(np.log2(tw) * 8) / 8)
        side += 4 * h * t
    tw[:, :sink] /= sinkdiv
    r = y - mu[:, None]
    if rho:
        a = (r[:, 1:] * r[:, :-1]).sum(1) / np.maximum((r**2).sum(1), 1e-30)
        a = np.clip(np.round(a * 16) / 16, 0, 15 / 16)
    else:
        a = np.zeros((h, d))
    q = np.empty((h, t, d), np.int64)
    rec = np.empty_like(r)
    prev = np.zeros((h, d))
    for i in range(t):
        p = 0.0 if i % block == 0 else a * prev
        st = step * tw[:, i : i + 1]
        qi = np.round((r[:, i] - p) / st)
        q[:, i] = qi
        prev = p + qi * st
        rec[:, i] = prev
    yh = rec + mu[:, None]
    xh = rope(yh, 1.0) if (is_key and pre) else yh
    if ctx == "chan":
        bits = h0_cols(q.transpose(1, 0, 2).reshape(t, h * d))
    elif ctx == "head":
        bits = h0_cols(q.transpose(1, 2, 0).reshape(t * d, h))
    else:
        bits = h0_cols(q.reshape(-1, 1))
    bits += 28 * h * d + side
    return bits / x.size, xh.astype(np.float32)


# ------------------------------------------------------------- rust codec
def tkv(x, is_key, rel, extra=()):
    h, t, d = x.shape
    with tempfile.TemporaryDirectory(dir=C.SCRATCH) as td:
        fi, fe, fo = (Path(td) / s for s in ("x.f32", "x.kvc", "y.f32"))
        np.ascontiguousarray(x, "<f4").tofile(fi)
        kind = "k" if is_key else "v"
        r = subprocess.run([KVC, "encode", fi, fe, "--shape", f"{h}x{t}x{d}", "--kind", kind,
                            "--rmse", repr(rel * _rms(x)), *extra], capture_output=True, text=True, check=True)
        subprocess.run([KVC, "decode", fe, fo], capture_output=True, check=True)
        j = json.loads(r.stdout.strip().splitlines()[-1])
        return 8 * j["bytes"] / x.size, np.fromfile(fo, "<f4").reshape(x.shape)


CONFIGS = {}
for b in (2, 3, 4, 5, 6, 8):
    for g in (32, 64, 128):
        CONFIGS[f"kivi-b{b}-g{g}"] = partial(kivi, bits=b, g=g)
for e in (0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3):
    CONFIGS[f"sz3-{e}"] = partial(sz3, rel=e)
for b in (4, 5, 6, 8):
    CONFIGS[f"introw-b{b}"] = partial(introw, bits=b)
for e in (0.03, 0.05, 0.07, 0.1, 0.15):
    for pre in (0, 1):
        n = "pre" if pre else "post"
        CONFIGS[f"proto-{n}-{e}"] = partial(proto, rel=e, pre=bool(pre))
        CONFIGS[f"psink-{n}-s4d16-{e}"] = partial(proto, rel=e, pre=bool(pre), sink=4, sinkdiv=16)
        CONFIGS[f"psink-{n}-s1d16-{e}"] = partial(proto, rel=e, pre=bool(pre), sink=1, sinkdiv=16)
        CONFIGS[f"pvtok-{n}-b1-s4-{e}"] = partial(proto, rel=e, pre=bool(pre), sink=4, sinkdiv=16, beta=1.0)
        CONFIGS[f"pvtok-{n}-b05-s4-{e}"] = partial(proto, rel=e, pre=bool(pre), sink=4, sinkdiv=16, beta=0.5)


def tkv_split(x, is_key, rel, kx=1.0, extra=()):
    """K at rel*kx, V at rel/kx (same nominal mean error budget in log)."""
    return tkv(x, is_key, rel * (kx if is_key else 1 / kx), extra)


for e in (0.03, 0.04, 0.05, 0.06, 0.07, 0.085, 0.1, 0.12, 0.15):
    CONFIGS[f"tkv-pre-{e}"] = partial(tkv, rel=e)
    CONFIGS[f"tkv-post-{e}"] = partial(tkv, rel=e, extra=("--post",))
for e in (0.05, 0.07, 0.1):
    for kx in (0.7, 1.4, 2.0):
        CONFIGS[f"tkvsplit-k{kx}-{e}"] = partial(tkv_split, rel=e, kx=kx)
    for b in (32, 64, 256):
        CONFIGS[f"tkvblk-b{b}-{e}"] = partial(tkv, rel=e, extra=("--block", str(b)))
for kx in (0.5, 0.35):
    for e in (0.05, 0.07, 0.085, 0.1, 0.12, 0.15, 0.2):
        CONFIGS[f"tkvk{kx}-{e}"] = partial(tkv_split, rel=e, kx=kx)
