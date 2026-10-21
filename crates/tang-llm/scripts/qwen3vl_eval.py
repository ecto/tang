#!/usr/bin/env python3
"""A small vcad-style viewport eval for Qwen3-VL-8B: 20 rendered CAD scenes (shaded boxes,
cylinders, spheres and cones on a grid floor, with an axis triad, as a CAD viewport shows
them), one short-answer question each, with the answer known from the scene.

    qwen3vl_eval.py --model <Qwen3-VL-8B-Instruct dir> --server http://127.0.0.1:8090 --out DIR
        [--hf-dtype bfloat16]

tang answers through `tang-llm serve` (`/v1/chat/completions`, the image as a data URL), so
the whole serving path runs: template, placeholders, preprocessing, tower, prefill, decode.
HF answers the same chat greedily. Reported: tang's agreement with HF (exact answer match),
and both models' accuracy against the scene's ground truth.
"""
import argparse
import base64
import io
import json
import os
import random
import re
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

COLORS = {"red": "#d04040", "blue": "#3a6fd8", "green": "#3fa34d", "yellow": "#e3c23a",
          "orange": "#e07b2a", "gray": "#9aa0a8"}
SHAPES = ["cube", "cylinder", "sphere", "cone"]


def box(ax, c, s, color):
    x, y, z = c
    v = np.array([[x - s, y - s, 0], [x + s, y - s, 0], [x + s, y + s, 0], [x - s, y + s, 0],
                  [x - s, y - s, 2 * s], [x + s, y - s, 2 * s], [x + s, y + s, 2 * s], [x - s, y + s, 2 * s]])
    f = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]]
    ax.add_collection3d(Poly3DCollection([v[i] for i in f], facecolors=color, edgecolors="#222", linewidths=0.6, shade=True))
    return 2 * s


def revolve(ax, c, s, color, kind):
    x, y, _ = c
    t = np.linspace(0, 2 * np.pi, 40)
    if kind == "sphere":
        u, w = np.meshgrid(t, np.linspace(0, np.pi, 20))
        ax.plot_surface(x + s * np.cos(u) * np.sin(w), y + s * np.sin(u) * np.sin(w), s + s * np.cos(w),
                        color=color, shade=True, linewidth=0)
        return 2 * s
    h = 3 * s
    zz = np.linspace(0, h, 2)
    tt, z = np.meshgrid(t, zz)
    r = s if kind == "cylinder" else s * (1 - z / h)
    ax.plot_surface(x + r * np.cos(tt), y + r * np.sin(tt), z, color=color, shade=True, linewidth=0)
    if kind == "cylinder":
        ax.add_collection3d(Poly3DCollection([list(zip(x + s * np.cos(t), y + s * np.sin(t), np.full_like(t, h)))],
                                             facecolor=color, edgecolor="#222", linewidths=0.4))
    return h


def scene(seed):
    rng = random.Random(seed)
    n = rng.randint(1, 4)
    names = rng.sample(list(COLORS), n)
    shapes = [rng.choice(SHAPES) for _ in range(n)]
    xs = sorted(rng.sample([-6, -3, 0, 3, 6], n))
    sizes = [rng.choice([0.8, 1.1, 1.5, 1.9]) for _ in range(n)]
    fig = plt.figure(figsize=(9.6, 5.4), dpi=100, facecolor="#2b2d33")
    ax = fig.add_subplot(projection="3d", facecolor="#2b2d33")
    g = np.arange(-8, 9, 2)
    for v in g:
        ax.plot([v, v], [-8, 8], [0, 0], color="#4a4d55", lw=0.6)
        ax.plot([-8, 8], [v, v], [0, 0], color="#4a4d55", lw=0.6)
    heights = []
    for x, nm, sh, s in zip(xs, names, shapes, sizes):
        c = (x, 0, 0)
        heights.append(box(ax, c, s, COLORS[nm]) if sh == "cube" else revolve(ax, c, s, COLORS[nm], sh))
    for d, col in zip(np.eye(3) * 2, ["#e04848", "#48c060", "#4878e0"]):
        ax.plot([-8, -8 + d[0]], [-8, -8 + d[1]], [0, d[2]], color=col, lw=2.5)
    ax.set_xlim(-8, 8), ax.set_ylim(-8, 8), ax.set_zlim(0, 8)
    ax.set_box_aspect((1, 1, 0.5), zoom=1.35)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.view_init(elev=22, azim=-90 + rng.uniform(-12, 12))
    ax.set_axis_off()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    # One question per scene, cycling through kinds.
    kind = seed % 5
    if kind == 0:
        q, ans = "How many solid parts are in this CAD viewport? Answer with a single number.", str(n)
    elif kind == 1:
        i = rng.randrange(n)
        q, ans = f"What color is the {shapes[i]}? Answer with one word.", names[i]
        if shapes.count(shapes[i]) > 1:
            q, ans = "What color is the leftmost part? Answer with one word.", names[0]
    elif kind == 2:
        q, ans = "What shape is the leftmost part: cube, cylinder, sphere or cone? Answer with one word.", shapes[0]
    elif kind == 3:
        q, ans = "What color is the rightmost part? Answer with one word.", names[-1]
    elif sorted(heights)[-1] == sorted(heights)[-2:][0] and n > 1:
        q, ans = "How many solid parts are in this CAD viewport? Answer with a single number.", str(n)
    else:
        i = int(np.argmax(heights))
        q, ans = "Which part is the tallest? Answer with its color only, one word.", names[i]
    return buf.getvalue(), q, ans


def norm(s):
    s = s.strip().lower()
    words = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "grey": "gray", "box": "cube"}
    w = re.findall(r"[a-z0-9]+", s)
    return words.get(w[0], w[0]) if w else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--server", default="http://127.0.0.1:8090")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hf-dtype", default="bfloat16")
    ap.add_argument("-n", type=int, default=20)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    items = [scene(i) for i in range(a.n)]
    for i, (png, _, _) in enumerate(items):
        open(os.path.join(a.out, f"scene{i:02d}.png"), "wb").write(png)

    def chat(png, q):
        url = "data:image/png;base64," + base64.b64encode(png).decode()
        return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                             {"type": "text", "text": q}]}]

    tang = []
    for png, q, _ in items:
        body = {"model": "qwen3-vl", "messages": chat(png, q), "max_tokens": 8, "temperature": 0}
        req = urllib.request.Request(a.server + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        tang.append(json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]["content"])

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained(a.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        a.model, dtype=getattr(torch, a.hf_dtype)).to("cuda").eval()
    hf = []
    for png, q, _ in items:
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
        text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        inp = proc(text=[text], images=[Image.open(io.BytesIO(png)).convert("RGB")], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=8, do_sample=False)
        hf.append(proc.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True))

    rows, same, t_ok, h_ok = [], 0, 0, 0
    for i, ((_, q, ans), t, h) in enumerate(zip(items, tang, hf)):
        nt, nh = norm(t), norm(h)
        same += nt == nh
        t_ok += nt == ans
        h_ok += nh == ans
        rows.append({"scene": i, "question": q, "truth": ans, "tang": t, "hf": h})
        print(f"{i:2d} truth {ans:8s} tang {t!r:14s} hf {h!r:14s} {'=' if nt == nh else '≠'}", flush=True)
    n = len(items)
    print(f"tang = HF on {same}/{n}; accuracy vs ground truth: tang {t_ok}/{n}, HF ({a.hf_dtype}) {h_ok}/{n}")
    json.dump(rows, open(os.path.join(a.out, "results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
