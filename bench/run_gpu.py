#!/usr/bin/env python3
"""tc-gpu ratio sweep (M1d): serial tc-wavelet (hand) vs tc-gpu per brick size
on the 5 original crops + the 14 held-out fields, at 60/80/100 dB and
max-abs 1e-3/1e-4. Every tc-gpu stream is decoded and checked by gpubench.
Appends JSON lines to out/gpu_ratio.jsonl."""
import json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
GB = HERE.parent / "target" / "release" / "examples" / "gpubench"
OUT = HERE / "out" / "gpu_ratio.jsonl"

ORIG = ["miranda-density", "miranda-velocityx", "hurricane-TC", "hurricane-U", "nyx-temperature"]


def fields():
    f = [(HERE / "out" / f"{n}-64x128x128.f32", [64, 128, 128]) for n in ORIG]
    for m in json.load(open(HERE / "data" / "heldout" / "manifest.json")):
        f.append((HERE / "data" / "heldout" / f"{m['name']}.f32", m["shape"]))
    return f


def run(item):
    path, shape = item
    bricks = "256,512,1024" if len(shape) == 2 else "32,64,128"
    cmd = [str(GB), "ratio", str(path), "x".join(map(str, shape)), "--bricks", bricks, "--threads", "4", "--lanes", sys.argv[1] if len(sys.argv) > 1 else "32"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(path.name, "FAILED", r.stderr[-2000:], flush=True)
    print(path.name, "done", flush=True)
    return r.stdout


if __name__ == "__main__":
    with ThreadPoolExecutor(4) as ex:
        outs = list(ex.map(run, fields()))
    with open(OUT, "a") as f:
        for o in outs:
            f.write(o)
