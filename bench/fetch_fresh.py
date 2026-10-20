#!/usr/bin/env python3
"""Fetch the Phase-2 fresh held-out set (SDRBench apps never used before:
S3D, Basilisk, XGC, EXAALT, HACC, NWChem) into fresh/<name>.f32 by HTTP
range requests (only each crop's slab is downloaded), plus fresh/manifest.json.
Every crop is float32 C-order; f64 sources are cast."""
import json, math, subprocess
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent / "fresh"
OUT.mkdir(parents=True, exist_ok=True)
BASE = "https://huggingface.co/datasets/sdrbench/{repo}/resolve/main/{path}"


def get(repo, path, start, length):
    b = subprocess.run(["curl", "-sfL", "-r", f"{start}-{start + length - 1}", BASE.format(repo=repo, path=path)],
                       capture_output=True, check=True).stdout
    assert len(b) == length, (path, len(b), length)
    return b


def slab(repo, path, dtype, shape, z0, nz, off=0):
    """Rows z0:z0+nz of the slowest axis (contiguous), `off` bytes into the file."""
    it = np.dtype(dtype).itemsize
    row = math.prod(shape[1:]) * it
    return np.frombuffer(get(repo, path, off + z0 * row, nz * row), dtype=dtype).reshape((nz, *shape[1:]))


M = []
def save(name, app, a, note=""):
    a = np.ascontiguousarray(a.astype(np.float32))
    a.tofile(OUT / f"{name}.f32")
    M.append({"name": name, "app": app, "group": "fresh", "shape": list(a.shape), "note": note,
              "min": float(a.min()), "max": float(a.max())})
    print(name, a.shape, float(a.min()), float(a.max()), flush=True)


# S3D stat_planar: 11 variables x 500^3 f64 (variable slowest); 64x128x128 from the middle
S3D = "s3d/stat_planar.2.9950E-03.field.d64"
for v, label in [(1, "O2"), (6, "T"), (8, "U")]:
    s = slab("s3d", S3D, "<f8", (500, 500, 500), 218, 64, off=v * 500 ** 3 * 8)
    save(f"s3d-{label}", "S3D", s[:, 186:314, 186:314], "f64 source cast to f32")
# Basilisk 2D turbulence: 1501 x 1024 x 1024 f32 (time slowest), timestep 750
for v in ["u", "v"]:
    s = slab("basilisk-turbulence", f"basilisk/{v}_1024x1024x1501Re10k.f32", "<f4", (1501, 1024, 1024), 750, 1)
    save(f"basilisk-{v}", "Basilisk", s[0])
# XGC dpot 20694 x 512 f64: nodes 8000:10048
s = slab("xgc", "xgc/xgc.3d.08100.bp_dpot_20694x512_double", "<f8", (20694, 512), 8000, 2048)
save("xgc-dpot", "XGC", s, "f64 source cast to f32")
# EXAALT copper dataset1 5423 x 3137 f32: 1024x1024 crop
s = slab("exaalt", "copper/dataset1-5423x3137.x.f32.dat", "<f4", (5423, 3137), 2000, 1024)
save("exaalt-cu-x", "EXAALT", s[:, 1000:2024])
# HACC medium vx (1D particles): 2^20 values from the middle
n = 1 << 20
b = get("hacc", "medium/vx.f32", (280953867 // 2) * 4, n * 4)
save("hacc-vx", "HACC", np.frombuffer(b, "<f4"))
# NWChem 631-tst f32 (1D two-electron integrals): 2^20 values from the middle
b = get("nwchem", "nwchem/631-tst.bin.f32", (411812992 // 8) * 4 // 4 * 4, n * 4)
save("nwchem-631", "NWChem", np.frombuffer(b, "<f4"))
json.dump(M, open(OUT / "manifest.json", "w"), indent=1)
