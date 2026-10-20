#!/usr/bin/env python3
"""Fetch the Phase-4 held-out set (the third fresh set) into fresh3/<name>.f32
plus fresh3/manifest.json, by HTTP range requests.

Every SDRBench app on the mirror has been used before, so this set is new
*fields, runs and crops*: no field, timestep or dataset below was used for
any earlier tuning or gate. It is weighted toward the classes that failed
the Phase-2 gate (EXAALT/HACC particles, Basilisk, XGC). Chosen before any
Phase-4 result; nothing is tuned on it. Every crop is float32 C-order; f64
sources are cast."""
import json, math, subprocess
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent / "fresh3"
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


def flat(repo, path, n_total, start, n, dtype="<f4"):
    it = np.dtype(dtype).itemsize
    assert start + n <= n_total
    return np.frombuffer(get(repo, path, start * it, n * it), dtype=dtype)


M = []
def save(name, app, a, note=""):
    a = np.ascontiguousarray(a.astype(np.float32))
    a.tofile(OUT / f"{name}.f32")
    M.append({"name": name, "app": app, "group": "fresh3", "shape": list(a.shape), "note": note,
              "min": float(a.min()), "max": float(a.max())})
    print(name, a.shape, float(a.min()), float(a.max()), flush=True)


N = 1 << 20
# EXAALT: helium dataset1 (7852 x 1037), y; another copper dataset (83 x 1077290), z; 2869440 run, vx
s = slab("exaalt", "helium/dataset1-7852x1037.y.f32.dat", "<f4", (7852, 1037), 3000, 1024)
save("exaalt-he-y", "EXAALT", s[:, :1024])
s = slab("exaalt", "copper/dataset2-83x1077290.z.f32.dat", "<f4", (83, 1077290), 0, 64)
save("exaalt-cu2-z", "EXAALT", s[:, 500000:516384])
save("exaalt-2869440-vx", "EXAALT", flat("exaalt", "2869440/vx.f32", 2869440, 2869440 // 2 - N // 2, N))
# HACC medium: positions yy (middle) and velocity vz (first quarter)
nh = 1123815468 // 4
save("hacc-yy", "HACC", flat("hacc", "medium/yy.f32", nh, nh // 2, N))
save("hacc-vz", "HACC", flat("hacc", "medium/vz.f32", nh, nh // 4, N))
# Basilisk 2048^2 run (never used), u at timestep 500, centre 1024^2
s = slab("basilisk-turbulence", "basilisk/u_2048x2048x1001Re10k.f32", "<f4", (1001, 2048, 2048), 500, 1)
save("basilisk2k-u", "Basilisk", s[0, 512:1536, 512:1536])
# XGC: iden (never used) at step 08500, nodes 12000:14048
s = slab("xgc", "xgc/xgc.3d.08500.bp_iden_20694x512_double", "<f8", (20694, 512), 12000, 2048)
save("xgc-iden", "XGC", s, "f64 source cast to f32")
# S3D: another time file, variable 3, an off-centre crop
s = slab("s3d", "s3d/stat_planar.1.7000E-03.field.d64", "<f8", (500, 500, 500), 100, 64, off=3 * 500 ** 3 * 8)
save("s3d-v3", "S3D", s[:, 100:228, 100:228], "f64 source cast to f32")
# NWChem acd-tst (never used), middle 2^20
na = 3204395564 // 4
save("nwchem-acd", "NWChem", flat("nwchem", "nwchem/acd-tst.bin.f32", na, na // 2, N))
# NYX baryon density (heavy-tailed), Hurricane QCLOUD and W, Miranda viscocity, SCALE W, CESM FLNT
s = slab("nyx", "512x512x512/baryon_density.f32", "<f4", (512, 512, 512), 200, 64)
save("nyx-baryon", "NYX", s[:, 192:320, 192:320])
for v in ["QCLOUD", "W"]:
    s = slab("hurricane-isabel", f"100x500x500/{v}f48.bin.f32", "<f4", (100, 500, 500), 18, 64)
    save(f"hurricane-{v}", "Hurricane", s[:, 186:314, 186:314])
s = slab("miranda", "256x384x384/viscocity.d64", "<f8", (256, 384, 384), 96, 64)
save("miranda-viscocity", "Miranda", s[:, 128:256, 128:256], "f64 source cast to f32")
s = slab("scale-letkf", "98x1200x1200/W-98x1200x1200.f32", "<f4", (98, 1200, 1200), 17, 64)
save("scale-W", "SCALE-LETKF", s[:, 536:664, 536:664])
s = slab("cesm-atm", "1800x3600/FLNT_1_1800_3600.f32", "<f4", (1800, 3600), 388, 1024)
save("cesm-FLNT", "CESM-ATM", s[:, 1288:2312])
json.dump(M, open(OUT / "manifest.json", "w"), indent=1)
