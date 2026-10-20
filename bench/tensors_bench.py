#!/usr/bin/env python3
"""ML-tensor compression bench: lossless + lossy baselines + tang-compress.

    UVW="uv run --no-project --python 3.12 --with zipnn --with zfpy --with pysz --with zstandard --with numpy"
    $UVW python bench/tensors_prep.py     # (also --with tokenizers --with pyarrow)
    $UVW python bench/tensors_dump.py     # KV / activations / gradients
    TC_BIN=... QTT_BIN=... $UVW python bench/tensors_bench.py [--quick]
    $UVW python bench/tensors_ppl.py      # downstream perplexity

Distortion is relative L2 error  e = ||x - x_hat|| / ||x||  (scale-free, the
usual ML-tensor metric). Each lossy codec is swept; bpv at e = 1% and 10% is
log-linear interpolation on its Pareto front (blank = not reached).
Writes bench/out/tensors_points.csv and bench/out/tensors_summary.md.
"""
from __future__ import annotations

import csv
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tensors_codecs as C  # noqa: E402
from tensors_io import RESNET, SMOL, VIT, T, load_safetensors  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
MAXN = 1 << 20  # crop rows so each tensor is <= 1M values


def manifest() -> list[tuple[str, str, str, np.ndarray]]:
    """(name, kind, native dtype, f32 array)."""
    out = []
    smol = load_safetensors(SMOL)
    for l in (0, 15, 29):
        for n in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                  "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            k = f"model.layers.{l}.{n}.weight"
            out.append((f"smol.L{l}.{n.split('.')[-1]}", "weight", "BF16", smol[k][0]))
    out.append(("smol.embed[:1792]", "weight", "BF16", smol["model.embed_tokens.weight"][0][:1792]))
    rn = load_safetensors(RESNET)
    for k in ("conv1.weight", "layer2.0.conv1.weight", "layer3.1.conv2.weight", "layer4.1.conv2.weight", "fc.weight"):
        out.append((f"resnet18.{k[:-7]}", "weight", "F32", rn[k][0]))
    vit = load_safetensors(VIT)
    for k in ("patch_embed.proj.weight", "blocks.5.attn.qkv.weight", "blocks.5.mlp.fc1.weight", "blocks.11.mlp.fc2.weight"):
        out.append((f"vit_tiny.{k[:-7]}", "weight", "F32", vit[k][0]))
    for f in sorted((T / "kv").glob("*.npy")):
        if f.stem.startswith("wiki0") or f.stem.startswith("gen0"):
            out.append((f"kv.{f.stem}", "kv", "BF16", np.load(f)))
    for f in sorted((T / "act").glob("*.npy")):
        if "_L0_" not in f.stem:
            out.append((f"act.{f.stem}", "act", "BF16", np.load(f)))
    for f in sorted((T / "grad").glob("*.npy")):
        s = f.stem.replace("model.layers.", "L").replace(".weight", "")
        if s.startswith("adam"):
            out.append((s, "adam", "F32", np.load(f)))
        elif "q_proj" in s or "down_proj" in s or "k_proj" in s:
            out.append((f"grad.{s}", "grad", "F32", np.load(f)))
    res = []
    for name, kind, dt, x in out:
        if x.size > MAXN:
            x = x[: max(1, MAXN // (x.size // x.shape[0]))]
        res.append((name, kind, dt, np.ascontiguousarray(x, np.float32)))
    return res


def rel(x, y):
    return float(np.linalg.norm((x - y).ravel()) / max(np.linalg.norm(x.ravel()), 1e-30))


def run_one(args):
    name, kind, dt, x, quick = args
    rows = []
    ll = C.lossless(x, dt)
    for k, v in ll.items():
        rows.append(dict(tensor=name, kind=kind, codec=k, param="lossless", bpv=v, rel=0.0, maxrel=0.0))
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    rng = float(x.max() - x.min()) or 1.0

    def add(codec, param, bpv, y, ent=None):
        if y is None or not math.isfinite(bpv):
            return
        r = dict(tensor=name, kind=kind, codec=codec, param=param, bpv=bpv, rel=rel(x, y),
                 maxrel=float(np.abs(x - y).max() / rng))
        rows.append(r)
        if ent is not None:
            rows.append({**r, "codec": codec + "+H0(ideal)", "bpv": ent})

    for b in (8, 6, 4, 3, 2):
        bpv, ent, y = C.quant_rows(x, kind, b)
        add("int-perrow", f"b={b}", bpv, y, ent)
        bpv, ent, y = C.quant_rows(x, kind, b, group=128)
        add("int-g128", f"b={b}", bpv, y, ent)
    if kind == "kv":
        for b in (8, 4, 3, 2):
            bpv, ent, y = C.kivi_kv(x, name.endswith("_K"), b)
            add("kivi-g32", f"b={b}", bpv, y, ent)
    for c in (3.0, 1.0, 0.3, 0.1, 0.03, 0.01, 0.003):
        add("ecsq(ideal)", f"step={c}rms", *C.ecsq(x, c * rms))
        add("sz3-abs", f"eb={c}rms", *C.sz3(x, c * rms))
        add("zfp-acc", f"tol={c}rms", *C.zfp(x, c * rms))
    tags = (0.1, 0.01) if quick else (0.2, 0.1, 0.03, 0.01, 0.003)
    for e in tags:
        add("tang-wavelet", f"rmse={e}rms", *C.tang(x, "wavelet", e * rms, f"{name}.w{e}"))
    if not quick and x.size <= 600_000:
        for e in (0.1, 0.01):
            add("tang-tucker", f"rmse={e}rms", *C.tang(x, "tucker", e * rms, f"{name}.t{e}"))
            add("tang-qtt", f"rmse={e}rms", *C.tang(x, "qtt", e * rms, f"{name}.q{e}"))
    print(f"done {name} {x.shape}", flush=True)
    return rows


def at_rel(pts, target):
    """bpv at relative error `target` by log-linear interpolation on the
    Pareto front (min bpv for error <= e)."""
    pts = sorted((p["rel"], p["bpv"]) for p in pts if p["rel"] > 0 and math.isfinite(p["bpv"]))
    front, best = [], math.inf
    for e, b in pts:  # ascending error; keep strictly decreasing bpv
        if b < best:
            front.append((e, b))
            best = b
    for (e0, b0), (e1, b1) in zip(front, front[1:]):
        if e0 <= target <= e1:
            t = (math.log(target) - math.log(e0)) / (math.log(e1) - math.log(e0))
            return b0 + t * (b1 - b0)
    if front and front[0][0] <= target and len(front) == 1:
        return front[0][1]
    return math.nan


def summarize(rows):
    kinds = {}
    for r in rows:
        kinds.setdefault(r["kind"], {}).setdefault(r["codec"], {}).setdefault(r["tensor"], []).append(r)
    lines = ["| tensor type | codec | lossless bpv | bpv @1% relL2 | bpv @10% relL2 | n |", "|---|---|---|---|---|---|"]
    for kind, codecs in kinds.items():
        for codec, per in codecs.items():
            if per[next(iter(per))][0]["param"] == "lossless":
                v = np.mean([p[0]["bpv"] for p in per.values()])
                lines.append(f"| {kind} | {codec} | {v:.2f} | | | {len(per)} |")
                continue
            a1 = [at_rel(p, 0.01) for p in per.values()]
            a10 = [at_rel(p, 0.10) for p in per.values()]
            f = lambda a: f"{np.nanmedian(a):.2f}" + ("" if all(map(math.isfinite, a)) else f" ({sum(map(math.isfinite, a))}/{len(a)})") if any(map(math.isfinite, a)) else "—"
            lines.append(f"| {kind} | {codec} | | {f(a1)} | {f(a10)} | {len(per)} |")
    return "\n".join(lines)


def main():
    quick = "--quick" in sys.argv
    OUT.mkdir(exist_ok=True)
    C.SCRATCH.mkdir(parents=True, exist_ok=True)
    jobs = [(*m, quick) for m in manifest()]
    rows = []
    with ProcessPoolExecutor(4) as ex:
        for r in ex.map(run_one, jobs):
            rows += r
    with open(OUT / "tensors_points.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    s = summarize(rows)
    (OUT / "tensors_summary.md").write_text(s + "\n")
    print(s)


if __name__ == "__main__":
    main()
