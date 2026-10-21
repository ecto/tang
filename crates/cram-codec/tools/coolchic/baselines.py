"""JPEG-XL / AVIF rate-distortion on Kodak (RGB PSNR as Cool-chic: MSE over
all RGB samples per image). Usage: python baselines.py KODAK_DIR OUT.tsv [CODECS]"""

import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

kd, out = sys.argv[1], sys.argv[2]
imgs = sorted(f for f in os.listdir(kd) if f.endswith(".png"))
JXL_D = [0.25, 0.4, 0.6, 0.9, 1.3, 1.9, 2.8, 4.0, 6.0]
AVIF_Q = [95, 90, 85, 78, 70, 60, 50, 40, 30]


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255.0**2 / mse)


def run(job):
    codec, q, name = job
    src = os.path.join(kd, name)
    ref = np.asarray(Image.open(src).convert("RGB"))
    with tempfile.TemporaryDirectory() as td:
        if codec == "jxl":
            enc, dec = os.path.join(td, "a.jxl"), os.path.join(td, "a.png")
            subprocess.run(["cjxl", src, enc, "-d", str(q), "-e", "7", "--quiet"], check=True, capture_output=True)
            subprocess.run(["djxl", enc, dec, "--quiet"], check=True, capture_output=True)
        else:
            enc, dec = os.path.join(td, "a.avif"), os.path.join(td, "a.png")
            tune = ["-a", "tune=psnr"] if codec == "avif-psnr" else []
            subprocess.run(["avifenc", "-y", "444", "-s", "4", "-q", str(q), "-j", "1", *tune, src, enc], check=True, capture_output=True)
            subprocess.run(["avifdec", enc, dec], check=True, capture_output=True)
        rec = np.asarray(Image.open(dec).convert("RGB"))
        nb = os.path.getsize(enc)
    return codec, q, name, nb * 8 / (ref.shape[0] * ref.shape[1]), psnr(ref, rec)


CODECS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["jxl", "avif", "avif-psnr"]
jobs = [(c, d, n) for c in CODECS if c == "jxl" for d in JXL_D for n in imgs]
jobs += [(c, q, n) for c in CODECS if c.startswith("avif") for q in AVIF_Q for n in imgs]
with ThreadPoolExecutor(4) as ex, open(out, "w") as f:
    f.write("codec\tq\timage\tbpp\tpsnr\n")
    for r in ex.map(run, jobs):
        f.write("\t".join(str(x) for x in r) + "\n")
        f.flush()
