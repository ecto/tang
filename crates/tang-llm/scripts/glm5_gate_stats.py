#!/usr/bin/env python3
"""How long a poked KDA state row survives real traffic, from `TANG_GLM5_GATE_DUMP` gates.

    glm5_gate_stats.py DIR --heads NH

DIR holds `l{L}_g.f32` (pre-exp log decay, [t][nh*hd]) and `l{L}_beta.f32` ([t][nh]). Per
layer: decay g = exp(pre) mean/min/max, beta mean/max, share of (head, channel) whose mean g
exceeds .99/.999, and the gate-only remaining fraction of a row after n tokens,
exp(n * mean log g) (geometric-mean decay; ignores the delta rule's extra erosion by beta),
for the median and the best channel.
"""
import argparse, glob, os, re
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--heads", type=int, required=True)
a = ap.parse_args()
ns = (1, 10, 100, 1000)
print("layer tokens  g mean   g min    g max    b mean b max  >.99   >.999  exact1 chmin  remain@" + "/".join(map(str, ns)) + " median | best (h,ch)")
best_all = None
for f in sorted(glob.glob(os.path.join(a.dir, "l*_g.f32")), key=lambda p: int(re.findall(r"l(\d+)_", p)[-1])):
    l = int(re.findall(r"l(\d+)_", f)[-1])
    beta = np.fromfile(f.replace("_g.f32", "_beta.f32"), dtype="<f4").reshape(-1, a.heads)
    t = beta.shape[0]
    lg = np.fromfile(f, dtype="<f4").reshape(t, a.heads, -1).astype(np.float64)
    g = np.exp(lg)
    mg = g.mean(0)
    g32 = np.exp(lg.astype(np.float32))  # what an f32 scan multiplies by
    exact = int((g32 == 1.0).all(0).sum())  # channels bitwise 1.0 on every token
    cmin = g32.min(0).ravel()  # per-channel floor over the run
    ml = lg.mean(0).ravel()
    med, best = np.median(ml), ml.max()
    h, ch = divmod(int(ml.argmax()), lg.shape[2])
    rem = lambda m: "/".join(f"{np.exp(n * m):.1e}" for n in ns)
    print(f"{l:5} {t:6}  {g.mean():.4f}  {g.min():.2e} {g.max():.6f} {beta.mean():.3f} {beta.max():.3f}"
          f"  {(mg > .99).mean():.4f} {(mg > .999).mean():.4f}  exact1 {exact}/{cmin.size} chmin med {np.median(cmin):.4f} max {cmin.max():.6f}  {rem(med)} | {rem(best)} ({h},{ch})")
    if best_all is None or best > best_all[0]:
        best_all = (best, l, h, ch, np.exp(best))
print(f"best channel overall: layer {best_all[1]} head {best_all[2]} ch {best_all[3]}, geometric-mean g {best_all[4]:.6f}, "
      f"half-life {np.log(.5) / best_all[0]:.0f} tokens")
