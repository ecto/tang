#!/usr/bin/env python3
"""SPERR (1 and 16 OpenMP threads) and SZ3 (serial) decode throughput on a
field, at PSNR targets. Usage: speed_ref.py FIELD.f32 AxBxC [psnr...]"""
import sys, numpy as np
from run_bench import run_sperr, run_sz3
x = np.fromfile(sys.argv[1], dtype="<f4").reshape([int(d) for d in sys.argv[2].split("x")])
gb = x.nbytes / 1e9
for p in [float(v) for v in sys.argv[3:]] or [80.0]:
    for omp in (1, 16):
        r = run_sperr(x, "psnr", p, omp)
        print(f"sperr omp={omp} target={p} bpv={r['bpv']:.4f} psnr={r['psnr']:.2f} dec_s={r['dec_s']:.3f} dec_gbs={gb / r['dec_s']:.3f} (incl. file I/O)", flush=True)
    r = run_sz3(x, "psnr", p)
    print(f"sz3 serial target={p} bpv={r['bpv']:.4f} psnr={r['psnr']:.2f} dec_s={r['dec_s']:.3f} dec_gbs={gb / r['dec_s']:.3f}", flush=True)
