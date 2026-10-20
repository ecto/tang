#!/usr/bin/env python3
"""Fetch held-out SDRBench crops (HF mirror, HTTP range requests; only the
bytes of each crop's slab are downloaded) into data/heldout/<name>.f32.

Every crop is float32 C-order; f64 sources are cast (noted in the name list).
The manifest (name, app, group, shape) goes to data/heldout/manifest.json.
"""
import json, math, subprocess, sys
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent / "data" / "heldout"
OUT.mkdir(parents=True, exist_ok=True)
BASE = "https://huggingface.co/datasets/sdrbench/{repo}/resolve/main/{path}"

def get(repo, path, start, length):
    b = subprocess.run(["curl", "-sfL", "-r", f"{start}-{start + length - 1}", BASE.format(repo=repo, path=path)],
                       capture_output=True, check=True).stdout
    assert len(b) == length, (path, len(b), length)
    return b

def slab(repo, path, dtype, shape, z0, nz):
    """Rows z0:z0+nz of the slowest axis (contiguous)."""
    it = np.dtype(dtype).itemsize
    row = math.prod(shape[1:]) * it
    b = get(repo, path, z0 * row, nz * row)
    return np.frombuffer(b, dtype=dtype).reshape((nz, *shape[1:]))

M = []
def save(name, app, group, a, note=""):
    a = np.ascontiguousarray(a.astype(np.float32))
    a.tofile(OUT / f"{name}.f32")
    M.append({"name": name, "app": app, "group": group, "shape": list(a.shape), "note": note,
              "min": float(a.min()), "max": float(a.max())})
    print(name, a.shape, float(a.min()), float(a.max()), flush=True)

# SCALE-LETKF (new app): 98x1200x1200 f32, crop 64x128x128 from the middle
for v in ["T", "U", "QV", "PRES"]:
    s = slab("scale-letkf", f"98x1200x1200/{v}-98x1200x1200.f32", "<f4", (98, 1200, 1200), 17, 64)
    save(f"scale-{v}", "SCALE-LETKF", "new-app", s[:, 536:664, 536:664])
# CESM-ATM 2D (new app): 1800x3600 f32, crop 1024x1024
for v in ["CLDTOT", "FLDS", "LHFLX", "TS"]:
    try:
        s = slab("cesm-atm", f"1800x3600/{v}_1_1800_3600.f32", "<f4", (1800, 3600), 388, 1024)
    except Exception as e:
        print("skip", v, e)
        continue
    save(f"cesm-{v}", "CESM-ATM", "new-app", s[:, 1288:2312])
# EXAFEL (new app): event 0, 1024x1024 detector crop
s = slab("exafel", "130x1480x1552/SDRBENCH-EXAFEL-data-130x1480x1552.f32", "<f4", (130 * 1480, 1552), 228, 1024)
save("exafel-ev0", "EXAFEL", "new-app", s[:, 264:1288])
# QMCPACK (new app): (115,69,69,288) with orbital fastest -> whole file needed
if "--qmcpack" in sys.argv:
    tmp = OUT / "qmc.tmp"
    if not tmp.exists():
        subprocess.run(["curl", "-sfL", "-o", str(tmp), BASE.format(repo="qmcpack", path="115x69x69x288/einspline_115_69_69_288.f32")], check=True)
    q = np.fromfile(tmp, dtype="<f4").reshape(115, 69, 69, 288)
    for k in [0, 100]:
        save(f"qmcpack-orb{k}", "QMCPACK", "new-app", q[:, :, :, k])
    tmp.unlink()
# same-app controls (fields not used in M0/M1)
s = slab("miranda", "256x384x384/pressure.d64", "<f8", (256, 384, 384), 96, 64)
save("miranda-pressure", "Miranda", "same-app", s[:, 128:256, 128:256], "f64 source cast to f32")
s = slab("hurricane-isabel", "100x500x500/Vf48.bin.f32", "<f4", (100, 500, 500), 18, 64)
save("hurricane-V", "Hurricane", "same-app", s[:, 186:314, 186:314])
s = slab("nyx", "512x512x512/velocity_x.f32", "<f4", (512, 512, 512), 0, 64)
save("nyx-velocity_x", "NYX", "same-app", s[:, :128, :128])
json.dump(M, open(OUT / "manifest.json", "w"), indent=1)
