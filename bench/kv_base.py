"""KV quantizer baselines for kv_hf.py (Phase 3 fair-baseline run).

Every codec is f(t, cfg) -> (bpv, y, domain) where `t` is one layer's
cache for one window: dict(pre=[kvh,T,hd] pre-RoPE K or V, post=post-RoPE
K or V, key=bool, calib=calibration tensor in the codec's domain or None)
and `domain` says whether y is pre-RoPE ('pre', the harness re-applies the
model's RoPE) or post-RoPE ('post').

All baselines are numpy re-implementations from the papers (no upstream
code is run); bit counts include every side value at its stated width.

- KIVI (Liu et al. 2024, arXiv 2402.02750, github.com/jy-yuan/KIVI): K
  per-channel asym groups of g tokens, V per-token asym groups of g
  channels, fp16 scale+zero, and an fp16 window of the R most recent
  tokens (paper default g=32, R=128). The window is applied per query in
  the attention (kv_hf.py); its 16-bit cost is charged at the eval length.
- KVQuant-style (Hooper et al. 2024, arXiv 2401.18079): pre-RoPE K with
  per-channel ranges calibrated offline, V with per-token ranges online,
  non-uniform datatype per layer (1-D k-means on calibration data; the
  paper's Fisher weighting is omitted), dense-and-sparse: values outside
  the ranges kept in fp16 (+16-bit index), first token (sink) in fp16.
- QuaRot-style rotation (Ashkboos et al. 2024, arXiv 2404.00456): post-RoPE
  K/V rotated by a normalized Walsh-Hadamard matrix over head_dim, then
  per-(head, token) asym quantization with clip ratio c, fp16 scale+zero.
- TurboQuant-mse (Zandieh et al. 2025, arXiv 2504.19874): per-(head, token)
  vector: fp16 norm, random orthogonal rotation, per-coordinate Lloyd-Max
  quantizer for N(0, 1/hd). (No QJL residual, no outlier-channel split.)
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from functools import lru_cache, partial
from pathlib import Path

import numpy as np

KVC = os.environ.get("KVC_BIN", str(Path(__file__).resolve().parent.parent / "target/release/examples/kvc"))
SCRATCH = Path(os.environ.get("TC_SCRATCH", tempfile.gettempdir()))


def _rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def _f16(a):
    return a.astype(np.float16).astype(np.float32)


def _asym(a, bits, clip=1.0):
    lo, hi = clip * a.min(-1, keepdims=True), clip * a.max(-1, keepdims=True)
    s = _f16((hi - lo) / (2**bits - 1))
    lo = _f16(lo)
    s[s == 0] = 1
    q = np.clip(np.round((a - lo) / s), 0, 2**bits - 1)
    return q * s + lo


# --------------------------------------------------------------- tang kvc
def tkv(t, cfg, rel, kx=0.5, extra=()):
    """The tang codec on the pre-RoPE tensor (--post: no internal RoPE).
    The step is rel x the *centered* rms (per-(head, channel) token mean
    removed): the codec codes the mean exactly-ish anyway, and a K channel
    constant is invisible to softmax, so a huge constant (Qwen2.5 K bias,
    rms 69 vs centered rms 1.0 in layer 0) must not set the step."""
    x = t["pre"]
    h, n, d = x.shape
    r = rel * (kx if t["key"] else 1 / kx)
    with tempfile.TemporaryDirectory(dir=SCRATCH) as td:
        fi, fe, fo = (Path(td) / s for s in ("x.f32", "x.kvc", "y.f32"))
        np.ascontiguousarray(x, "<f4").tofile(fi)
        o = subprocess.run([KVC, "encode", fi, fe, "--shape", f"{h}x{n}x{d}", "--kind", "k" if t["key"] else "v",
                            "--post", "--rmse", repr(r * _rms(x - x.mean(1, keepdims=True))), *extra], capture_output=True, text=True, check=True)
        subprocess.run([KVC, "decode", fe, fo], capture_output=True, check=True)
        j = json.loads(o.stdout.strip().splitlines()[-1])
        return 8 * j["bytes"] / x.size, np.fromfile(fo, "<f4").reshape(x.shape), "pre"


# --------------------------------------------------------------- KIVI
def kivi(t, cfg, bits, g=32, R=128):
    x = t["post"]
    h, n, d = x.shape
    if t["key"]:
        y = _asym(x.transpose(0, 2, 1).reshape(-1, g), bits).reshape(h, d, n).transpose(0, 2, 1)
    else:
        y = _asym(x.reshape(-1, g), bits).reshape(h, n, d)
    r = min(R, n)
    bpv = ((n - r) * (bits + 32 / g) + r * 16) / n
    return bpv, y, "post"


# --------------------------------------------------------------- KVQuant-style
def _kmeans1d(v, k, iters=25):
    v = np.sort(v.astype(np.float64).ravel())
    if v.size > 400_000:
        v = v[:: v.size // 400_000]
    c = np.quantile(v, (np.arange(k) + 0.5) / k)
    for _ in range(iters):
        e = (c[1:] + c[:-1]) / 2
        idx = np.searchsorted(e, v)
        s = np.bincount(idx, v, k)
        m = np.bincount(idx, None, k)
        c = np.where(m > 0, s / np.maximum(m, 1), c)
    return np.sort(c)


def _nearest(z, c):
    e = (c[1:] + c[:-1]) / 2
    return c[np.searchsorted(e, z)]


def _kvq_norm_k(x, lo, hi):
    mid, half = (hi + lo) / 2, np.maximum((hi - lo) / 2, 1e-12)
    return (x - mid) / half, mid, half


def _kvq_norm_v(x, outl):
    # per token across all kv heads: [h,n,d] -> thresholds per token
    h, n, d = x.shape
    a = x.transpose(1, 0, 2).reshape(n, h * d)
    k = max(1, int(round(outl * h * d / 2)))
    srt = np.sort(a, 1)
    lo, hi = srt[:, k : k + 1], srt[:, -k - 1 : -k]
    lo, hi = _f16(lo), _f16(hi)
    mid, half = (hi + lo) / 2, np.maximum((hi - lo) / 2, 1e-12)
    z = (a - mid) / half
    return z, mid, half


def kvquant(t, cfg, bits, outl=0.01):
    x = t["pre"].astype(np.float32)
    h, n, d = x.shape
    cal = t["calib"]
    k = 2**bits
    if t["key"]:
        # per-channel ranges from calibration (offline), [h,1,d]
        lo = np.quantile(cal, outl / 2, axis=1, keepdims=True)
        hi = np.quantile(cal, 1 - outl / 2, axis=1, keepdims=True)
        zc, _, _ = _kvq_norm_k(cal, lo, hi)
        cb = _kmeans1d(zc[np.abs(zc) <= 1], k)
        z, mid, half = _kvq_norm_k(x, lo, hi)
        out = np.abs(z) > 1
        y = _nearest(np.clip(z, -1, 1), cb) * half + mid
        y = np.where(out, _f16(x), y)
        side = 0.0
    else:
        zc, _, _ = _kvq_norm_v(cal, outl)
        cb = _kmeans1d(zc[np.abs(zc) <= 1], k)
        z, mid, half = _kvq_norm_v(x, outl)
        a = x.transpose(1, 0, 2).reshape(n, h * d)
        out = np.abs(z) > 1
        y = np.where(out, _f16(a), _nearest(np.clip(z, -1, 1), cb) * half + mid)
        y = y.reshape(n, h, d).transpose(1, 0, 2)
        out = out.reshape(n, h, d).transpose(1, 0, 2)
        side = 32 / (h * d)
    y = y.astype(np.float32)
    y[:, 0] = _f16(x[:, 0])  # attention sink in fp16
    nout = int(out[:, 1:].sum())
    bits_tot = (n - 1) * h * d * (bits + side) + nout * 32 + h * d * 16
    return bits_tot / x.size, y, "pre"


# --------------------------------------------------------------- rotations
@lru_cache(None)
def _hadamard(d):
    m = np.array([[1.0]])
    while m.shape[0] < d:
        m = np.block([[m, m], [m, -m]])
    assert m.shape[0] == d, "head_dim must be a power of two"
    return (m / np.sqrt(d)).astype(np.float32)


def hadamard(t, cfg, bits, clip=1.0, g=None):
    x = t["post"]
    d = x.shape[-1]
    g = g or d
    H = _hadamard(d)
    r = x @ H
    y = _asym(r.reshape(*r.shape[:-1], d // g, g), bits, clip).reshape(r.shape) @ H.T
    return bits + 32 / g, y.astype(np.float32), "post"


@lru_cache(None)
def _rot(d, seed=0):
    q, r = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, d)))
    return (q * np.sign(np.diag(r))).astype(np.float32)


@lru_cache(None)
def _lloyd_gauss(bits, d):
    v = np.random.default_rng(1).standard_normal(400_000) / np.sqrt(d)
    return _kmeans1d(v, 2**bits, 60)


def turboquant(t, cfg, bits):
    x = t["post"].astype(np.float32)
    d = x.shape[-1]
    nrm = _f16(np.linalg.norm(x, axis=-1, keepdims=True))
    P = _rot(d)
    u = (x / np.maximum(nrm, 1e-12)) @ P.T
    y = (_nearest(u, _lloyd_gauss(bits, d)) @ P) * nrm
    return bits + 16 / d, y.astype(np.float32), "post"


# --------------------------------------------------------------- registry
CONFIGS = {"identity": lambda t, cfg: (16.0, t["pre"], "pre")}
for e in (0.03, 0.04, 0.05, 0.06, 0.07, 0.085, 0.1, 0.12, 0.15, 0.2):
    CONFIGS[f"tkv-{e}"] = partial(tkv, rel=e)
for e in (0.05, 0.07, 0.085, 0.1):
    CONFIGS[f"tkvk0.7-{e}"] = partial(tkv, rel=e, kx=0.7)
for b in (2, 3, 4, 5, 6, 8):
    for g in (32, 128):
        CONFIGS[f"kivi-R128-g{g}-b{b}"] = partial(kivi, bits=b, g=g, R=128)
        CONFIGS[f"kivi-R0-g{g}-b{b}"] = partial(kivi, bits=b, g=g, R=0)
for b in (2, 3, 4, 5):
    CONFIGS[f"kvquant-b{b}"] = partial(kvquant, bits=b)
for b in (3, 4, 5, 6):
    CONFIGS[f"hadamard-b{b}"] = partial(hadamard, bits=b)
    CONFIGS[f"hadamardc90-b{b}"] = partial(hadamard, bits=b, clip=0.9)
    CONFIGS[f"hadamardg32-b{b}"] = partial(hadamard, bits=b, g=32)
for b in (2, 3, 4, 5, 6):
    CONFIGS[f"turbo-b{b}"] = partial(turboquant, bits=b)
for e in (0.05, 0.06, 0.07, 0.085, 0.1, 0.12):
    CONFIGS[f"tkvgpu-{e}"] = partial(tkv, rel=e, extra=("--gpu",))
for e in (0.07, 0.085, 0.1, 0.12, 0.15):
    for kx in (1.0, 1.4):
        CONFIGS[f"tkvk{kx}-{e}"] = partial(tkv, rel=e, kx=kx)
    CONFIGS[f"tkvgpuk0.7-{e}"] = partial(tkv, rel=e, kx=0.7, extra=("--gpu",))
