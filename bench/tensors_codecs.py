"""Codecs for the ML-tensor bench. Every lossy codec maps (x f32, param) ->
(bits, x_hat); lossless ones map native-dtype bytes -> bits.

Rates count everything a decoder needs except tiny fixed headers we can't
see (zstd/zipnn frames are included as returned). `ideal` rows are order-0
entropy bounds (no table cost) — upper bounds on what an ANS coder could do.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from tensors_io import f32_to_bf16_bits

SCRATCH = Path(os.environ.get("TC_SCRATCH", tempfile.gettempdir()))
TC = os.environ.get("TC_BIN", str(Path(__file__).resolve().parent.parent / "target/release/examples/tc"))
QTT = os.environ.get("QTT_BIN", str(Path(__file__).resolve().parent.parent / "target/release/examples/qtt"))


# ------------------------------------------------------------------ helpers
def native_bytes(x: np.ndarray, dtype: str) -> tuple[np.ndarray, int]:
    """(uint view of the native representation, bits per value)."""
    if dtype == "BF16":
        return f32_to_bf16_bits(x).ravel(), 16
    return np.ascontiguousarray(x, np.float32).view(np.uint32).ravel(), 32


def h0(sym: np.ndarray) -> float:
    """order-0 entropy in bits/symbol."""
    _, c = np.unique(sym, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def rows_of(x: np.ndarray, kind: str) -> np.ndarray:
    """Quantization rows: output channel for weights/grads, last axis
    (per token / per head-token) for KV and activations."""
    if kind in ("weight", "grad", "adam"):
        return x.reshape(x.shape[0], -1) if x.ndim > 1 else x.reshape(1, -1)
    return x.reshape(-1, x.shape[-1])


# ----------------------------------------------------------------- lossless
def lossless(x: np.ndarray, dtype: str) -> dict[str, float]:
    import zstandard

    u, nb = native_bytes(x, dtype)
    n = u.size
    raw = u.tobytes()
    z = zstandard.ZstdCompressor(level=19)
    out = {"zstd19": 8 * len(z.compress(raw)) / n}
    planes = u.view(np.uint8).reshape(n, nb // 8)
    out["bytegroup+zstd19"] = 8 * sum(len(z.compress(planes[:, i].tobytes())) for i in range(nb // 8)) / n
    try:
        import zipnn

        zz = zipnn.ZipNN(input_format="byte", bytearray_dtype="bfloat16" if nb == 16 else "float32")
        out["zipnn"] = 8 * len(zz.compress(raw)) / n
    except Exception as e:  # noqa: BLE001
        out["zipnn"] = float("nan")
        print("zipnn failed:", e)
    # DietGPU float mode: rANS on the 8-bit exponent, sign+mantissa raw.
    exp = (u >> (7 if nb == 16 else 23)) & 0xFF
    out["ans-exp(ideal)"] = h0(exp) + (nb - 8)
    return out


# -------------------------------------------------------------------- lossy
def quant_rows(x: np.ndarray, kind: str, bits: int, group: int | None = None):
    """Symmetric absmax RTN, per row (group=None) or per `group` contiguous
    values along each row. fp16 scales."""
    r = rows_of(x, kind)
    if group:
        pad = (-r.shape[1]) % group
        rr = np.pad(r, ((0, 0), (0, pad))).reshape(-1, group)
    else:
        rr = r
    qmax = 2 ** (bits - 1) - 1
    s = (np.abs(rr).max(1, keepdims=True) / qmax).astype(np.float16).astype(np.float32)
    s[s == 0] = 1
    q = np.clip(np.round(rr / s), -qmax - 1, qmax)
    y = (q * s).reshape(r.shape[0], -1)[:, : r.shape[1]]
    side = 16 / rr.shape[1]
    return bits + side, h0(q) + side, y.reshape(x.shape)


def _asym(a: np.ndarray, bits: int):
    lo, hi = a.min(1, keepdims=True), a.max(1, keepdims=True)
    s = ((hi - lo) / (2**bits - 1)).astype(np.float16).astype(np.float32)
    lo = lo.astype(np.float16).astype(np.float32)
    s[s == 0] = 1
    q = np.clip(np.round((a - lo) / s), 0, 2**bits - 1)
    return q, q * s + lo


def ecsq(x: np.ndarray, step: float):
    """Entropy-coded scalar quantization: one global uniform step, ideal
    order-0 code (no transform, no context modelling). The reference point
    for 'how much does a transform/context model buy'."""
    q = np.round(x / step)
    return h0(q), (q * step).astype(np.float32)


def kivi_kv(x: np.ndarray, is_key: bool, bits: int, g: int = 32):
    h, T, d = x.shape
    if is_key:  # per-channel: groups of g tokens for each (head, channel)
        a = x.transpose(0, 2, 1).reshape(-1, g)
        q, y = _asym(a, bits)
        y = y.reshape(h, d, T).transpose(0, 2, 1)
    else:  # per-token: groups of g channels
        a = x.reshape(-1, g)
        q, y = _asym(a, bits)
        y = y.reshape(h, T, d)
    side = 32 / g
    return bits + side, h0(q) + side, y


def sz3(x: np.ndarray, eb: float):
    from pysz import sz, szConfig, szErrorBoundMode

    c = szConfig()
    c.errorBoundMode = szErrorBoundMode.ABS
    c.absErrorBound = float(eb)
    xc = np.ascontiguousarray(x, np.float32)
    cmp, _ = sz.compress(xc, c)
    y, _ = sz.decompress(cmp, np.float32, xc.shape)
    return 8 * cmp.nbytes / x.size, y


def zfp(x: np.ndarray, tol: float):
    import zfpy

    xc = np.ascontiguousarray(x, np.float32)
    if xc.ndim > 4:
        xc = xc.reshape(xc.shape[0], -1)
    b = zfpy.compress_numpy(xc, tolerance=float(tol))
    return 8 * len(b) / x.size, zfpy.decompress_numpy(b).reshape(x.shape)


def tang(x: np.ndarray, codec: str, rmse: float, tag: str):
    """Run current tang-compress (tc wavelet/tucker or qtt) on x as-is."""
    xc = np.ascontiguousarray(x, np.float32)
    dims = list(xc.shape) if xc.ndim <= 3 else [xc.shape[0], xc.shape[1], -1]
    if -1 in dims:
        dims[2] = xc.size // (dims[0] * dims[1])
    if len(dims) == 1:
        dims = [1, dims[0]]
    d = "x".join(map(str, dims))
    fin, fenc, fout = (SCRATCH / f"{tag}.{s}" for s in ("f32", "enc", "dec.f32"))
    xc.astype("<f4").tofile(fin)
    if codec == "qtt":
        cmd = [QTT, "encode", fin, fenc, "--dims", d, "--rmse", repr(rmse)]
        dcmd = [QTT, "decode", fenc, fout]
    else:
        cmd = [TC, "encode", fin, fenc, "--dims", d, "--codec", codec, "--rmse", repr(rmse)]
        dcmd = [TC, "decode", fenc, fout]
    try:
        r = subprocess.run(list(map(str, cmd)), capture_output=True, text=True, timeout=600, check=True)
        subprocess.run(list(map(str, dcmd)), capture_output=True, timeout=600, check=True)
        j = json.loads(r.stdout.strip().splitlines()[-1])
        y = np.fromfile(fout, "<f4").reshape(x.shape)
        return 8 * j["bytes"] / x.size, y
    except Exception as e:  # noqa: BLE001
        print(f"tang {codec} failed on {tag}: {e}")
        return float("nan"), None
    finally:
        for f in (fin, fenc, fout):
            f.unlink(missing_ok=True)
