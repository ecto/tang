"""Build measured cram.sh example assets. Runs on mew. Every number comes from
an encode + decode performed by this script; PSNR/max-abs measured on decoded output
(PSNR = 20 log10(range/RMSE), range = max-min of original, as in bench/run_bench.py)."""
import json, math, subprocess, sys, time, os, platform
from pathlib import Path
import numpy as np
import zfpy
from pysz import sz, szConfig, szErrorBoundMode
import matplotlib
from matplotlib import cm
from PIL import Image

HOME = Path.home()
TC = HOME / "tang-pub/target/release/examples/tc"
SP3 = HOME / "cram-dist/work/SPERR/build/bin/sperr3d"
INP = HOME / "cram-site-in"
OUT = HOME / "cram-dist/examples"
W = HOME / "cram-dist/work/tmp"
W.mkdir(parents=True, exist_ok=True)
DIMS = (64, 128, 128)
PRETRAINED = INP / "pre3_32x16.tcm"
TARGET = 60.0
CMDS = []


def metrics(x, y):
    x64, y64 = x.astype(np.float64), y.astype(np.float64)
    rng = float(x64.max() - x64.min())
    e = x64 - y64
    rmse = float(np.sqrt(np.mean(e * e)))
    return (20 * math.log10(rng / rmse) if rmse > 0 else 999.0), float(np.abs(e).max())


def run(cmd):
    CMDS.append(" ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, check=True).stdout


def cram(x, psnr, model="hand"):
    src, bs, o = W / "c_in.f32", W / "c.tc", W / "c_out.f32"
    x.astype("<f4").tofile(src)
    j = json.loads(run([TC, "encode", src, bs, "--dims", "x".join(map(str, x.shape)), "--codec", "auto", "--model", model, "--psnr", repr(psnr)]).strip().splitlines()[-1])
    run([TC, "decode", bs, o, *(["--model", model] if model != "hand" else [])])
    y = np.fromfile(o, "<f4").reshape(x.shape)
    p, m = metrics(x, y)
    return dict(bytes=bs.stat().st_size, psnr=p, max_abs=m, knob=f"auto --model {model.split('/')[-1]} --psnr {psnr:.2f}", chose=j.get("chose")), y


def sperr(x, psnr):
    src, bs, o = W / "s_in.f32", W / "s.bs", W / "s_out.f32"
    x.astype("<f4").tofile(src)
    run([SP3, "-c", "--ftype", "32", "--dims", *map(str, x.shape[::-1]), "--psnr", repr(psnr), "--bitstream", bs, "--omp", "1", src])
    run([SP3, "-d", "--decomp_f", o, "--omp", "1", bs])
    y = np.fromfile(o, "<f4").reshape(x.shape)
    p, m = metrics(x, y)
    return dict(bytes=bs.stat().st_size, psnr=p, max_abs=m, knob=f"--psnr {psnr:.2f}"), y


def sz3(x, psnr):
    c = szConfig()
    try:
        c.openmp = False
    except Exception:
        pass
    c.errorBoundMode = szErrorBoundMode.PSNR
    c.psnrErrorBound = psnr
    b, _ = sz.compress(x, c)
    y, _ = sz.decompress(b, np.float32, x.shape)
    CMDS.append(f"pysz sz.compress(PSNR={psnr:.2f})") if not CMDS or "pysz" not in CMDS[-1] else None
    p, m = metrics(x, y)
    return dict(bytes=int(b.nbytes), psnr=p, max_abs=m, knob=f"PSNR mode {psnr:.2f}"), y


def zfp(x, logtol):
    tol = float(10 ** logtol)
    b = zfpy.compress_numpy(x, tolerance=tol)
    y = zfpy.decompress_numpy(b)
    p, m = metrics(x, y)
    return dict(bytes=len(b), psnr=p, max_abs=m, knob=f"tolerance {tol:.4g}"), y


def bisect(fn, x, target, lo, hi, iters=18):
    """fn(x, knob) with measured PSNR increasing in knob; smallest knob in [lo,hi] whose
    measured PSNR >= target (i.e. the cheapest setting that still meets it)."""
    best = fn(x, hi)
    if best[0]["psnr"] < target:
        raise RuntimeError(f"hi={hi} misses target")
    for _ in range(iters):
        mid = (lo + hi) / 2
        r = fn(x, mid)
        if r[0]["psnr"] >= target:
            hi, best = mid, r
        else:
            lo = mid
    return best


def hit_psnr(fn, x, target, lo=None, hi=None, iters=18):
    return bisect(fn, x, target, target - 15 if lo is None else lo, target + 15 if hi is None else hi, iters)


def sperr_pwe(x, k):
    rng = float(x.max() - x.min()); b = rng * 10 ** (-k)
    src, bs, o = W / "s_in.f32", W / "s.bs", W / "s_out.f32"
    x.astype("<f4").tofile(src)
    run([SP3, "-c", "--ftype", "32", "--dims", *map(str, x.shape[::-1]), "--pwe", repr(b), "--bitstream", bs, "--omp", "1", src])
    run([SP3, "-d", "--decomp_f", o, "--omp", "1", bs])
    y = np.fromfile(o, "<f4").reshape(x.shape)
    p, m = metrics(x, y)
    return dict(bytes=bs.stat().st_size, psnr=p, max_abs=m, knob=f"--pwe {b:.4g}"), y


def sz3_abs(x, k):
    rng = float(x.max() - x.min()); b = rng * 10 ** (-k)
    c = szConfig()
    try:
        c.openmp = False
    except Exception:
        pass
    c.errorBoundMode = szErrorBoundMode.ABS
    c.absErrorBound = b
    cb, _ = sz.compress(x, c)
    y, _ = sz.decompress(cb, np.float32, x.shape)
    p, m = metrics(x, y)
    return dict(bytes=int(cb.nbytes), psnr=p, max_abs=m, knob=f"ABS mode {b:.4g}"), y


def zfp_prec(x, prec):
    b = zfpy.compress_numpy(x, precision=int(prec))
    y = zfpy.decompress_numpy(b)
    p, m = metrics(x, y)
    return dict(bytes=len(b), psnr=p, max_abs=m, knob=f"precision {int(prec)}"), y


def cheapest(*rs):
    return min(rs, key=lambda t: t[0]["bytes"])


def best_sperr(x, t):
    return cheapest(hit_psnr(sperr, x, t), bisect(sperr_pwe, x, t, 0.5, 8))


def best_sz3(x, t):
    return cheapest(hit_psnr(sz3, x, t), bisect(sz3_abs, x, t, 0.5, 8))


def hit_zfp(x, target):
    rng = float(x.max() - x.min())
    a = bisect(lambda x, k: zfp(x, -k), x, target, -math.log10(rng), 12 - math.log10(rng), 30)
    ps = [zfp_prec(x, p) for p in range(4, 33)]
    ok = [r for r in ps if r[0]["psnr"] >= target]
    return cheapest(a, *ok) if ok else a


def best_cram(x, target):
    a = hit_psnr(lambda x, t: cram(x, t, "hand"), x, target, target - 3, target + 3, 10)
    b = hit_psnr(lambda x, t: cram(x, t, f"pretrained:{PRETRAINED}"), x, target, target - 3, target + 3, 10)
    return cheapest(a, b), (b if cheapest(a, b) is a else a)


def img(arr, cmap, vmin, vmax, path, width=480):
    a = np.clip((arr - vmin) / (vmax - vmin if vmax > vmin else 1), 0, 1)
    rgb = (matplotlib.colormaps[cmap](a)[..., :3] * 255).astype(np.uint8)
    im = Image.fromarray(rgb[::-1])  # row 0 at the bottom
    h = round(im.height * width / im.width)
    im = im.resize((width, h), Image.NEAREST if im.width < width else Image.LANCZOS)
    for q in (82, 75, 65, 55, 45):
        im.save(path, "WEBP", quality=q, method=6)
        if path.stat().st_size <= 60_000:
            return q
    return q


def main():
    fields = sys.argv[1:] or ["miranda-density", "hurricane-TC", "nyx-temperature"]
    ex = []
    for f in fields:
        x = np.fromfile(INP / f"{f}-64x128x128.f32", "<f4").reshape(DIMS)
        (cr, cy), (cram_alt, _) = best_cram(x, TARGET)
        sr, sy = best_sperr(x, TARGET)
        zr, zy = best_sz3(x, TARGET)
        fr, fy = hit_zfp(x, TARGET)
        z = DIMS[0] // 2
        sl = x[z]
        lo, hi = float(np.percentile(sl, 0.5)), float(np.percentile(sl, 99.5))
        names = {k: f"{f}-{k}.webp" for k in ("original", "cram", "cram-err", "sz3-err", "zfp-err")}
        img(sl, "cividis", lo, hi, OUT / names["original"])
        img(cy[z], "cividis", lo, hi, OUT / names["cram"])
        errs = {"cram": np.abs(cy[z].astype(np.float64) - sl), "sz3": np.abs(zy[z].astype(np.float64) - sl), "zfp": np.abs(fy[z].astype(np.float64) - sl)}
        emax = float(max(np.percentile(e, 99.5) for e in errs.values()))
        for k in ("cram", "sz3", "zfp"):
            img(errs[k], "magma", 0.0, emax, OUT / names[f"{k}-err"])
        ob = x.nbytes
        codecs = {"cram": cr, "sperr": sr, "sz3": zr, "zfp": fr}
        for k, r in codecs.items():
            r["ratio"] = ob / r["bytes"]
            r["bpv"] = 8 * r["bytes"] / x.size
            r["vs_sperr"] = sr["bytes"] / r["bytes"]
        ex.append(dict(name=f, dims=list(DIMS), original_bytes=ob, slice_z=z, codecs=codecs, cram_other_model=cram_alt,
                       images=names, image_scale=dict(field_cmap="cividis", field_range=[lo, hi], field_range_note="0.5/99.5 percentile of original slice",
                                                     error_cmap="magma", error_range=[0.0, emax], error_range_note="shared across cram/sz3/zfp error maps: 0..max of their 99.5th percentiles")))
        print(f, {k: (r["bytes"], round(r["psnr"], 2), r["max_abs"]) for k, r in codecs.items()}, flush=True)

    # RD dataset on the first field
    f = fields[0]
    x = np.fromfile(INP / f"{f}-64x128x128.f32", "<f4").reshape(DIMS)
    rd = {"field": f, "dims": list(DIMS), "metric": "bits per value vs measured PSNR (dB)", "points": {k: [] for k in ("cram", "sperr", "sz3", "zfp")}}
    for t in (40, 55, 70, 85, 100):
        for k, fn in (("cram", lambda x, t: best_cram(x, t)[0]), ("sperr", best_sperr), ("sz3", best_sz3), ("zfp", hit_zfp)):
            r, _ = fn(x, t)
            rd["points"][k].append(dict(target=t, bpv=8 * r["bytes"] / x.size, psnr=r["psnr"], max_abs=r["max_abs"], bytes=r["bytes"], knob=r["knob"]))
        print("rd", t, {k: round(v[-1]["psnr"], 2) for k, v in rd["points"].items()}, flush=True)
    (OUT / "rd.json").write_text(json.dumps(rd, indent=1))

    def sh(c):
        return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
    prov = dict(
        generated=time.strftime("%Y-%m-%dT%H:%M:%S%z"), host=platform.node(), gpu=sh("nvidia-smi -L"),
        tang_commit=sh("git -C ~/tang-pub rev-parse HEAD"), cram_codec_version=sh("grep -m1 '^version' ~/tang-pub/crates/cram-codec/Cargo.toml"),
        tc_build="CARGO_TARGET_DIR=$HOME/tang-pub/target cargo build --release -p cram-codec --example tc",
        rustc=sh("rustc --version"),
        sperr=dict(version=sh(f"{SP3.parent}/show_version | head -1"), commit=sh(f"git -C {SP3.parents[2]} rev-parse HEAD"), source="https://github.com/NCAR/SPERR (built Release, USE_OMP=ON, run --omp 1)"),
        sz3=dict(pysz=sh("~/cram-dist/venv/bin/pip show pysz | grep Version")), zfp=dict(zfpy=sh("~/cram-dist/venv/bin/pip show zfpy | grep Version")),
        psnr_def="20*log10((max-min)/RMSE) over the full 64x128x128 volume, float64",
        matching="every codec: bisection on its knob for the cheapest setting whose MEASURED PSNR >= target; SPERR = min(--psnr, --pwe), SZ3 = min(PSNR mode, ABS mode), ZFP = min(fixed-accuracy tolerance, fixed-precision 4..32); cram = min over --psnr knob in target+-3",
        cram_pick="smaller of tc auto --model hand and --model pretrained:pre3_32x16.tcm (pretrained decode needs the same model file)",
        script="mk_examples.py (this directory)", example_commands=sorted(set(c.replace(str(HOME), "~") for c in CMDS))[:40])
    (OUT / "examples.json").write_text(json.dumps(dict(target_psnr_db=TARGET, fields=ex, provenance=prov), indent=1))


main()
