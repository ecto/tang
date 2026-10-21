#!/usr/bin/env python3
"""QTT vs SZ3 / ZFP / wavelet benchmark (tang-compress M0).

Every codec sees the same float32 array and the same error definitions:

* PSNR = 20 log10(range / RMSE), range = max - min of the original;
* max-abs error bound b = rel * range.

Rate is the full compressed size in bits per value (bpv). For QTT this is the
whole self-describing stream; for SZ3 / ZFP it is the library's output buffer;
for the wavelet baseline it is the *ideal* zeroth-order entropy of the
quantized coefficients per subband (no table/header cost: optimistic) and,
separately, the zstd-19 size of the int32 coefficient array (a real coder).

Each codec is swept over its own knobs and reduced to its Pareto front
(bpv vs PSNR); matched-PSNR rates are log-linear interpolations on that front.
Timings are single-threaded wall clock (SZ3 openmp off; zfpy is serial).
"""
from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pywt
import zfpy
import zstandard
from pysz import sz, szConfig, szErrorBoundMode

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
DATA = HERE / "data"
QTT = Path(os.environ.get("QTT_BIN", HERE.parent / "target" / "release" / "examples" / "qtt"))
TC = Path(os.environ.get("TC_BIN", HERE.parent / "target" / "release" / "examples" / "tc"))
OUT.mkdir(exist_ok=True)


# ----------------------------------------------------------------- datasets
def synthetic(n: int = 1 << 20) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    t = np.arange(n) / n
    out = {}
    k = 5
    s = np.zeros(n)
    for _ in range(k):
        a, f, ph, g = rng.uniform(0.2, 1), rng.uniform(5, 400), rng.uniform(0, 6.28), rng.uniform(0, 8)
        s += a * np.exp(-g * t) * np.sin(2 * np.pi * f * t + ph)
    out["1d-damped-sines-K5"] = s
    out["1d-poly-deg5"] = np.polyval(rng.uniform(-1, 1, 6), 2 * t - 1)
    ps = np.sin(9 * t) + 0.3 * np.cos(31 * t) ** 2
    for j in rng.uniform(0, 1, 7):
        ps += rng.uniform(-0.5, 0.5) * (t > j)
    out["1d-piecewise-jumps"] = ps
    out["1d-white-noise"] = rng.standard_normal(n)
    return {k: v.astype(np.float32) for k, v in out.items()}


def real_fields() -> dict[str, np.ndarray]:
    out = {}
    def load(name, dtype, shape):
        p = DATA / name
        if not p.exists() or p.stat().st_size != np.dtype(dtype).itemsize * math.prod(shape):
            print(f"missing/incomplete {p}", file=sys.stderr)
            return None
        return np.fromfile(p, dtype=dtype).reshape(shape).astype(np.float32)
    m = load("miranda-density.d64", "<f8", (256, 384, 384))
    if m is not None:
        out["miranda-density-64x128x128"] = np.ascontiguousarray(m[96:160, 128:256, 128:256])
    m = load("miranda-velocityx.d64", "<f8", (256, 384, 384))
    if m is not None:
        out["miranda-velocityx-64x128x128"] = np.ascontiguousarray(m[96:160, 128:256, 128:256])
    h = load("hurricane-isabel-Uf48.bin.f32", "<f4", (100, 500, 500))
    if h is not None:
        out["hurricane-U-64x128x128"] = np.ascontiguousarray(h[18:82, 186:314, 186:314])
    h = load("hurricane-isabel-TCf48.bin.f32", "<f4", (100, 500, 500))
    if h is not None:
        out["hurricane-TC-64x128x128"] = np.ascontiguousarray(h[18:82, 186:314, 186:314])
    n = load("nyx-temperature.f32", "<f4", (512, 512, 512))
    if n is not None:
        out["nyx-temperature-64x128x128"] = np.ascontiguousarray(n[:64, :128, :128])
    return out


# ----------------------------------------------------------------- metrics
def metrics(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    x64, y64 = x.astype(np.float64), y.astype(np.float64)
    rng = float(x64.max() - x64.min())
    err = x64 - y64
    rmse = float(np.sqrt(np.mean(err * err)))
    psnr = 20 * math.log10(rng / rmse) if rmse > 0 and rng > 0 else 999.0
    return rmse, psnr, float(np.abs(err).max())


# ----------------------------------------------------------------- codecs
def run_qtt(x: np.ndarray, bound_args: list[str], order: str = "morton", pad: str = "mixed") -> dict:
    tmp = OUT / "tmp_in.f32"
    x.astype("<f4").tofile(tmp)
    dims = "x".join(str(d) for d in x.shape)
    cmd = [str(QTT), "bench", str(tmp), "--dims", dims, "--order", order, "--pad", pad, *bound_args]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    j = json.loads(res.stdout.strip().splitlines()[-1])
    return {
        "bpv": j["bpv"], "psnr": j["psnr"], "max_abs": j["max_abs"], "rmse": j["rmse"],
        "enc_s": j["encode_s"], "dec_s": j["decode_s"], "ranks": j["ranks"],
        "extra": f"res={j['n_residual']} hdr={j['header_bytes']} core={j['core_bytes']} resb={j['residual_bytes']} passes={j['passes']}",
    }


def run_sz3(x: np.ndarray, mode: str, val: float) -> dict:
    c = szConfig()
    try:
        c.openmp = False
    except Exception:
        pass
    if mode == "abs":
        c.errorBoundMode = szErrorBoundMode.ABS
        c.absErrorBound = val
    else:
        c.errorBoundMode = szErrorBoundMode.PSNR
        c.psnrErrorBound = val
    t = time.perf_counter()
    cmp, _ = sz.compress(x, c)
    enc = time.perf_counter() - t
    t = time.perf_counter()
    y, _ = sz.decompress(cmp, np.float32, x.shape)
    dec = time.perf_counter() - t
    rmse, psnr, mx = metrics(x, y)
    return {"bpv": 8 * cmp.nbytes / x.size, "psnr": psnr, "max_abs": mx, "rmse": rmse, "enc_s": enc, "dec_s": dec, "extra": f"{mode}={val:.3g}"}


def run_zfp(x: np.ndarray, mode: str, val: float) -> dict:
    t = time.perf_counter()
    # fixed-accuracy or fixed-precision; zfpy's fixed-rate path aborts on 1D
    # input and gains nothing over these two for an RMSE target.
    b = zfpy.compress_numpy(x, tolerance=val) if mode == "acc" else zfpy.compress_numpy(x, precision=int(val))
    enc = time.perf_counter() - t
    t = time.perf_counter()
    y = zfpy.decompress_numpy(b)
    dec = time.perf_counter() - t
    rmse, psnr, mx = metrics(x, y)
    return {"bpv": 8 * len(b) / x.size, "psnr": psnr, "max_abs": mx, "rmse": rmse, "enc_s": enc, "dec_s": dec, "extra": f"{mode}={val:.3g}"}


TP = HERE / "third_party"
SPERR3 = TP / "SPERR" / "build" / "bin" / "sperr3d"
SPERR2 = TP / "SPERR" / "build" / "bin" / "sperr2d"
TTHRESH = TP / "tthresh" / "build" / "tthresh"


def _as3d(x):
    """1D inputs are reshaped (C order) for the 2D/3D-only tools; noted in the report."""
    if x.ndim == 1:
        return x.reshape(64, 128, 128) if x.size == 1 << 20 else x.reshape(1, 1, -1)
    return x


def _timed(cmd):
    t = time.perf_counter()
    subprocess.run(cmd, check=True, capture_output=True)
    return time.perf_counter() - t


def run_sperr(x: np.ndarray, mode: str, val: float, omp: int = 1) -> dict:
    """SPERR 3D (sperr3d); 1D inputs go through sperr2d as 1024x1024."""
    src, bs, out = OUT / "sp_in.f32", OUT / "sp.bs", OUT / "sp_out.f32"
    if x.ndim == 1:
        exe, dims = SPERR2, ["1024", "1024"]
    elif x.ndim == 2:
        exe, dims = SPERR2, [str(d) for d in x.shape[::-1]]
    else:
        exe, dims = SPERR3, [str(d) for d in x.shape[::-1]]
    x.astype("<f4").tofile(src)
    knob = ["--pwe", repr(val)] if mode == "pwe" else ["--psnr", repr(val)]
    enc = _timed([str(exe), "-c", "--ftype", "32", "--dims", *dims, *knob, "--bitstream", str(bs), *(["--omp", str(omp)] if x.ndim > 2 else []), str(src)])
    dec = _timed([str(exe), "-d", "--decomp_f", str(out), *(["--omp", str(omp)] if x.ndim > 2 else []), str(bs)])
    y = np.fromfile(out, dtype="<f4").reshape(x.shape)
    rmse, psnr, mx = metrics(x, y)
    return {"bpv": 8 * bs.stat().st_size / x.size, "psnr": psnr, "max_abs": mx, "rmse": rmse, "enc_s": enc, "dec_s": dec, "extra": f"{mode}={val:.4g} omp={omp}"}


def run_tthresh(x: np.ndarray, psnr_target: float) -> dict:
    src, bs, out = OUT / "tt_in.f32", OUT / "tt.bs", OUT / "tt_out.f32"
    x3 = _as3d(x)
    x3.astype("<f4").tofile(src)
    dims = [str(d) for d in x3.shape[::-1]]
    enc = _timed([str(TTHRESH), "-i", str(src), "-t", "float", "-s", *dims, "-p", repr(psnr_target), "-c", str(bs)])
    dec = _timed([str(TTHRESH), "-c", str(bs), "-o", str(out)])
    y = np.fromfile(out, dtype="<f4").reshape(x.shape)
    rmse, psnr, mx = metrics(x, y)
    return {"bpv": 8 * bs.stat().st_size / x.size, "psnr": psnr, "max_abs": mx, "rmse": rmse, "enc_s": enc, "dec_s": dec, "extra": f"p={psnr_target:g}"}


def run_tc(x: np.ndarray, codec: str, bound_args: list[str], model: str = "hand") -> dict:
    """tang-compress wavelet/Tucker via the `tc` example; decoded by the
    binary's decoder and measured here, rate = stream file size."""
    src, bs, out = OUT / "tc_in.f32", OUT / "tc.bs", OUT / "tc_out.f32"
    x.astype("<f4").tofile(src)
    dims = "x".join(str(d) for d in x.shape)
    t = time.perf_counter()
    res = subprocess.run([str(TC), "encode", str(src), str(bs), "--dims", dims, "--codec", codec, "--model", model, *bound_args],
                         capture_output=True, text=True, check=True)
    enc = time.perf_counter() - t
    j = json.loads(res.stdout.strip().splitlines()[-1])
    t = time.perf_counter()
    ex = ["--model", model] if model.startswith("pretrained:") else []
    d = json.loads(subprocess.run([str(TC), "decode", str(bs), str(out), *ex], capture_output=True, text=True, check=True).stdout)
    dec = time.perf_counter() - t
    y = np.fromfile(out, dtype="<f4").reshape(x.shape)
    rmse, psnr, mx = metrics(x, y)
    extra = " ".join(f"{k}={j[k]}" for k in ("coef_bytes", "model_bytes", "outlier_bytes", "n_outliers", "core_bytes", "factor_bytes", "ranks", "chose", "peak", "tried") if k in j)
    return {"bpv": 8 * bs.stat().st_size / x.size, "psnr": psnr, "max_abs": mx, "rmse": rmse,
            "enc_s": j["encode_s"], "dec_s": d["decode_s"], "enc_wall_s": enc, "dec_wall_s": dec, "extra": extra}


def tang_extra(name, x, rows, models=("hand",), auto_model=None):
    """tang-compress M1 codecs: wavelet (PSNR + max-abs) and Tucker (PSNR)."""
    rng = float(x.max() - x.min())
    def add(codec, r, kind, **kw):
        r = {"dataset": name, "n": x.size, "codec": codec, "kind": kind, **kw, **r}
        rows.append(r)
        print(f"  {codec:16s} {kind:6s} bpv={r['bpv']:.4f} psnr={r['psnr']:.2f} max={r['max_abs']/rng:.2e}R enc={r['enc_s']:.2f}s dec={r['dec_s']:.3f}s {r['extra']}", flush=True)
    for model in models:
        tag = "" if model == "hand" else "-" + model.split(":")[0]
        for p in [30, 40, 50, 60, 70, 80, 90, 100]:
            add("tc-wavelet" + tag, run_tc(x, "wavelet", ["--psnr", str(p)], model), "psnr")
        for rel in [1e-2, 1e-3, 1e-4]:
            add("tc-wavelet" + tag, run_tc(x, "wavelet", ["--max-rel", repr(rel)], model), "maxabs", bound_rel=rel)
    if x.ndim > 1 and "hand" in models:
        for p in [30, 40, 50, 60, 70, 80, 90, 100]:
            add("tc-tucker", run_tc(x, "tucker", ["--psnr", str(p)]), "psnr")
    if auto_model:
        for p in [30, 40, 50, 60, 70, 80, 90, 100]:
            add("tc-auto", run_tc(x, "auto", ["--psnr", str(p)], auto_model), "psnr")
        for rel in [1e-2, 1e-3, 1e-4]:
            add("tc-auto", run_tc(x, "auto", ["--max-rel", repr(rel)], auto_model), "maxabs", bound_rel=rel)


def frontier_extra(name, x, rows):
    """SPERR (PSNR + PWE, 1 thread; default threads at ~60 dB) and TTHRESH (PSNR)."""
    rng = float(x.max() - x.min())
    def add(codec, r, kind, **kw):
        r = {"dataset": name, "n": x.size, "codec": codec, "kind": kind, **kw, **r}
        rows.append(r)
        print(f"  {codec:14s} {kind:6s} bpv={r['bpv']:.4f} psnr={r['psnr']:.2f} max={r['max_abs']/rng:.2e}R enc={r['enc_s']:.2f}s {r['extra']}", flush=True)
    for p in [30, 40, 50, 60, 70, 80, 90, 100]:
        add("sperr", run_sperr(x, "psnr", p), "psnr")
        # PWE sweep also traces the PSNR front (SPERR's PWE mode is its native one)
        add("sperr", run_sperr(x, "pwe", rng * 10 ** (-p / 20) * math.sqrt(3)), "psnr")
        try:
            add("tthresh", run_tthresh(x, p), "psnr")
        except subprocess.CalledProcessError as e:
            print("  tthresh failed", p, e.stderr[-200:] if e.stderr else "", flush=True)
    if x.ndim > 1:
        add("sperr-mt", run_sperr(x, "psnr", 60, omp=0), "speed")
    for rel in [1e-2, 1e-3, 1e-4]:
        add("sperr", run_sperr(x, "pwe", rel * rng), "maxabs", bound_rel=rel)


def entropy_bits(q: np.ndarray) -> float:
    _, counts = np.unique(q, return_counts=True)
    p = counts / q.size
    return float(-(counts * np.log2(p)).sum())


def run_wavelet(x: np.ndarray, step: float) -> tuple[dict, dict]:
    """bior4.4 (CDF 9/7), periodization, uniform mid-tread quantizer, one step."""
    x64 = x.astype(np.float64)
    # periodization: any level with every axis divisible by 2^L works; pywt's
    # dwtn_max_level (filter-length based) is far more conservative (L=2 at 64).
    level = 1
    while level < (10 if x.ndim == 1 else 5) and all(s % (1 << (level + 1)) == 0 and s >> (level + 1) >= 4 for s in x.shape):
        level += 1
    t = time.perf_counter()
    coeffs = pywt.wavedecn(x64, "bior4.4", mode="periodization", level=level)
    arr, slices = pywt.coeffs_to_array(coeffs)
    q = np.round(arr / step).astype(np.int64)
    # per-subband entropy (ideal static coder, no table cost)
    bits = 0.0
    for sl in [slices[0]] + [s for d in slices[1:] for s in d.values()]:
        bits += entropy_bits(q[sl].ravel())
    zbytes = len(zstandard.ZstdCompressor(level=19).compress(q.astype(np.int32).tobytes()))
    enc = time.perf_counter() - t
    t = time.perf_counter()
    rec = pywt.waverecn(pywt.array_to_coeffs(q * step, slices, output_format="wavedecn"), "bior4.4", mode="periodization")
    rec = rec[tuple(slice(0, s) for s in x.shape)].astype(np.float32)
    dec = time.perf_counter() - t
    rmse, psnr, mx = metrics(x, rec)
    base = {"psnr": psnr, "max_abs": mx, "rmse": rmse, "enc_s": enc, "dec_s": dec, "extra": f"step={step:.3g} L={level}"}
    return {**base, "bpv": bits / x.size}, {**base, "bpv": 8 * zbytes / x.size}


# ----------------------------------------------------------------- driver
def pareto(rows):
    rows = sorted(rows, key=lambda r: r["bpv"])
    front, best = [], -1e9
    for r in rows:
        if r["psnr"] > best + 1e-9:
            front.append(r)
            best = r["psnr"]
    return front


def bpv_at(front, target):
    pts = [(r["psnr"], r["bpv"]) for r in front if r["bpv"] > 0]
    for (p0, b0), (p1, b1) in zip(pts, pts[1:]):
        if p0 <= target <= p1:
            if p1 == p0:
                return b0
            t = (target - p0) / (p1 - p0)
            return math.exp(math.log(b0) + t * (math.log(b1) - math.log(b0)))
    return None


def heldout_fields() -> dict[str, np.ndarray]:
    """Held-out SDRBench crops written by fetch_heldout.py (never used for tuning)."""
    man = json.load(open(DATA / "heldout" / "manifest.json"))
    return {m["name"]: np.fromfile(DATA / "heldout" / f"{m['name']}.f32", dtype="<f4").reshape(m["shape"]) for m in man}


def fresh_fields(sub: str = "fresh") -> dict[str, np.ndarray]:
    """Fresh held-out crops: Phase 2 (fetch_fresh.py, `fresh`) or Phase 4
    (fetch_fresh3.py, `fresh3`)."""
    man = json.load(open(HERE / sub / "manifest.json"))
    return {m["name"]: np.fromfile(HERE / sub / f"{m['name']}.f32", dtype="<f4").reshape(m["shape"]) for m in man}


def sweep(name: str, x: np.ndarray, rows: list, quick: bool, orders=None):
    rng = float(x.max() - x.min())
    psnrs = [30, 40, 50, 60, 70, 80, 90, 100] if not quick else [40, 60, 80]
    def add(codec, r, kind):
        r = {"dataset": name, "n": x.size, "codec": codec, "kind": kind, **r}
        r.pop("ranks", None) if codec.startswith("sz") or codec.startswith("zfp") else None
        if not all(math.isfinite(r[k]) for k in ("bpv", "psnr", "max_abs")):
            return r
        rows.append(r)
        print(f"  {codec:14s} {kind:6s} bpv={r['bpv']:.4f} psnr={r['psnr']:.2f} max={r['max_abs']/rng:.2e}·R enc={r['enc_s']:.2f}s {r.get('extra','')}", flush=True)
        return r
    for order in orders or (["morton", "row"] if x.ndim > 1 else ["row"]):
        for p in psnrs:
            r = run_qtt(x, ["--psnr", str(p)], order=order)
            r["ranks"] = "-".join(map(str, r["ranks"]))
            add(f"qtt-{order}", r, "psnr")
    for p in psnrs:
        add("sz3", run_sz3(x, "psnr", p), "psnr")
        add("sz3", run_sz3(x, "abs", rng * 10 ** (-p / 20) * math.sqrt(3)), "psnr")
    for e in range(-1, -8, -1) if not quick else [-2, -4, -6]:
        add("zfp", run_zfp(x, "acc", rng * 10.0 ** e), "psnr")
    for prec in ([6, 10, 14, 18, 22] if quick else [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]):
        add("zfp", run_zfp(x, "prec", prec), "psnr")
    for p in psnrs:
        step = rng * 10 ** (-p / 20) * math.sqrt(12)
        ent, zst = run_wavelet(x, step)
        add("wavelet-H0", ent, "psnr")
        add("wavelet-zstd", zst, "psnr")
    # fixed max-abs bounds
    for rel in [1e-2, 1e-3, 1e-4]:
        b = rel * rng
        for order in orders or (["morton", "row"] if x.ndim > 1 else ["row"]):
            r = run_qtt(x, ["--max-abs", repr(b)], order=order)
            r["ranks"] = "-".join(map(str, r["ranks"]))
            add(f"qtt-{order}", {**r, "bound_rel": rel}, "maxabs")
        add("sz3", {**run_sz3(x, "abs", b), "bound_rel": rel}, "maxabs")
        # ZFP: largest tolerance whose *actual* max error stays <= b (fair best mode)
        lo, hi, best = b, b * 64, None
        r = run_zfp(x, "acc", lo)
        if r["max_abs"] <= b:
            best = r
            for _ in range(12):
                mid = math.sqrt(lo * hi)
                r = run_zfp(x, "acc", mid)
                if r["max_abs"] <= b:
                    lo, best = mid, r
                else:
                    hi = mid
        if best:
            add("zfp", {**best, "bound_rel": rel}, "maxabs")


def wavelet_only(name, x, rows):
    rng = float(x.max() - x.min())
    for p in [30, 40, 50, 60, 70, 80, 90, 100]:
        ent, zst = run_wavelet(x, rng * 10 ** (-p / 20) * math.sqrt(12))
        for codec, r in (("wavelet-H0", ent), ("wavelet-zstd", zst)):
            rows.append({"dataset": name, "n": x.size, "codec": codec, "kind": "psnr", **r})


def main():
    quick = "--quick" in sys.argv
    if "--frontier" in sys.argv:
        rows = []
        only = [a for a in sys.argv[1:] if not a.startswith("--")]
        for name, x in {**synthetic(), **real_fields()}.items():
            if only and not any(o in name for o in only):
                continue
            print(f"== {name} {x.shape}", flush=True)
            frontier_extra(name, x, rows)
        keys = sorted({k for r in rows for k in r})
        with open(OUT / "results_frontier.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        return
    if "--fresh-ext" in sys.argv or "--fresh3-ext" in sys.argv:
        sub = "fresh3" if "--fresh3-ext" in sys.argv else "fresh"
        for name, x in fresh_fields(sub).items():
            path = OUT / f"{sub}_ext_{name}.csv"
            if path.exists():
                continue
            rows = []
            print(f"== {name} {x.shape}", flush=True)
            for stage in (lambda: sweep(name, x, rows, False, orders=["row"]),
                          lambda: frontier_extra(name, x, rows)):
                try:
                    stage()
                except subprocess.CalledProcessError as e:
                    print("  STAGE FAILED", e.cmd[:3], (e.stderr or "")[-300:], flush=True)
            keys = sorted({k for r in rows for k in r})
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)
        return
    if "--heldout" in sys.argv:
        only = [a for a in sys.argv[1:] if not a.startswith("--")]
        pre = str(HERE / "models" / "pre_ALL.tcm")
        for name, x in heldout_fields().items():
            if only and not any(o in name for o in only):
                continue
            path = OUT / f"heldout_{name}.csv"
            if path.exists():
                continue
            rows = []
            print(f"== {name} {x.shape}", flush=True)
            for stage in (lambda: sweep(name, x, rows, False, orders=["row"]),
                          lambda: frontier_extra(name, x, rows),
                          lambda: tang_extra(name, x, rows, ["hand", "overfit", "pretrained:" + pre], "overfit")):
                try:
                    stage()
                except subprocess.CalledProcessError as e:
                    print("  STAGE FAILED", e.cmd[:3], (e.stderr or "")[-300:], flush=True)
            keys = sorted({k for r in rows for k in r})
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)
        return
    if "--tang" in sys.argv:
        rows = []
        only = [a for a in sys.argv[1:] if not a.startswith("--")]
        models = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--models=")), ["hand"])
        tag = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--tag=")), "tang")
        auto = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--auto=")), None)
        if "--models=" in " ".join(sys.argv) and models == [""]:
            models = []
        for name, x in {**synthetic(), **real_fields()}.items():
            if only and not any(o in name for o in only):
                continue
            print(f"== {name} {x.shape}", flush=True)
            tang_extra(name, x, rows, models, auto)
        keys = sorted({k for r in rows for k in r})
        with open(OUT / f"results_{tag}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        return
    if "--wavelet-only" in sys.argv:
        rows = []
        for name, x in {**synthetic(), **real_fields()}.items():
            wavelet_only(name, x, rows)
        keys = sorted({k for r in rows for k in r})
        with open(OUT / "results_wavelet.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        return
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    sets = {**synthetic(), **real_fields()}
    rows = []
    for name, x in sets.items():
        if only and not any(o in name for o in only):
            continue
        print(f"== {name} {x.shape}", flush=True)
        sweep(name, x, rows, quick)
    tag = "_".join(only) if only else "all"
    keys = sorted({k for r in rows for k in r})
    with open(OUT / f"results_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("wrote", OUT / f"results_{tag}.csv")


if __name__ == "__main__":
    main()
