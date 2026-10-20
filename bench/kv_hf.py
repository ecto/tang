#!/usr/bin/env python3
"""Model-generic KV-cache compression eval on HF transformers (Phase 3).

Two passes per window: a clean pass captures every layer's pre-RoPE K,
post-RoPE K and V (the model's own rotary is patched to record its input);
then, per codec config, every layer's cached K/V for the compressed span is
replaced by the codec's reconstruction (re-rotated with the model's own
cos/sin when the codec works pre-RoPE) and the forward is re-run.

Attention semantics in the override pass: query t attends to the
reconstruction of key s, except (a) the KIVI fp16 window: keys with
t - s < R stay exact, and (b) keys at s >= Tc (the uncompressed suffix in
the long-context task) stay exact. Rows before q0 are not computed (their
K/V in every layer come from the capture, so nothing reads them).

    uv run --no-project --with torch --with transformers --with numpy --with pyarrow \
        python bench/kv_hf.py ppl  --model smol135 [config-prefix ...]
    ... python bench/kv_hf.py niah --model qwen1.5 [config ...]
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kv_base as B  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
WIKI = HERE / "tensors/hf/wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet"
MODELS = {
    "smol135": "HuggingFaceTB/SmolLM2-135M",
    "qwen1.5": "Qwen/Qwen2.5-1.5B",
    "qwen7": "Qwen/Qwen2.5-7B",
    "llama3b": "meta-llama/Llama-3.2-3B",
}

STATE = dict(mode="clean", cap=None, rec=None, R=0, q0=0, last=None)


def _rot_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], -1)


def _patch_rope():
    import importlib

    for name in ("llama", "qwen2", "qwen3", "mistral"):
        try:
            m = importlib.import_module(f"transformers.models.{name}.modeling_{name}")
        except ImportError:
            continue
        orig = m.apply_rotary_pos_emb

        def f(q, k, cos, sin, *a, _o=orig, **kw):
            STATE["last"] = (k, cos, sin)
            return _o(q, k, cos, sin, *a, **kw)

        m.apply_rotary_pos_emb = f


def _rep(x, g):
    return x if g == 1 else x.repeat_interleave(g, dim=1)


def kvc_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    g = module.num_key_value_groups
    l = module.layer_idx
    mode = STATE["mode"]
    if mode == "capture":
        kpre = STATE["last"][0]
        STATE["cap"][l] = tuple(a[0].float().cpu().numpy() for a in (kpre, key, value))
    if mode != "override":
        o = torch.nn.functional.scaled_dot_product_attention(query.float(), _rep(key, g).float(), _rep(value, g).float(), is_causal=True, scale=scaling)
        return o.to(query.dtype).transpose(1, 2).contiguous(), None
    _, cos, sin = STATE["last"]
    rec = STATE["rec"][l]
    T = key.shape[2]
    dt = query.dtype
    query, key, value = query.float(), key.float(), value.float()
    kh = rec["K"].to(key.device, key.dtype)
    tc = kh.shape[1]
    if rec["Kdom"] == "pre":
        kh = kh * cos[0, :tc] + _rot_half(kh) * sin[0, :tc]
    vh = rec["V"].to(value.device, value.dtype)
    Kh = torch.cat([kh[None], key[:, :, tc:]], 2)
    Vh = torch.cat([vh[None], value[:, :, tc:]], 2)
    q0, R = STATE["q0"], STATE["R"]
    q = query[:, :, q0:]
    t = torch.arange(q0, T, device=q.device)[:, None]
    s = torch.arange(T, device=q.device)[None]
    causal = s <= t
    S = (q @ _rep(Kh, g).transpose(2, 3)) * scaling
    if R > 0:
        near = causal & (t - s < R)
        S = torch.where(near, (q @ _rep(key, g).transpose(2, 3)) * scaling, S)
    S = S.masked_fill(~causal, float("-inf"))
    P = torch.softmax(S, -1)
    if R > 0:
        nf = near.to(P.dtype)
        o = (P * nf) @ _rep(value, g) + (P * (1 - nf)) @ _rep(Vh, g)
    else:
        o = P @ _rep(Vh, g)
    full = torch.zeros_like(query)
    full[:, :, q0:] = o
    return full.to(dt).transpose(1, 2).contiguous(), None


def load(model, dev, dtype):
    from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

    _patch_rope()
    AttentionInterface.register("kvc", kvc_attention)
    tok = AutoTokenizer.from_pretrained(MODELS[model])
    m = AutoModelForCausalLM.from_pretrained(MODELS[model], dtype=dtype, attn_implementation="kvc").eval()
    if os.environ.get("KV_STREAM") and dev.startswith("cuda"):
        stream_layers(m, dev)
    else:
        m = m.to(dev)
    return tok, m


def stream_layers(m, dev):
    """Weights stay in (pinned) host memory; each decoder layer and the LM
    head is copied to the GPU just for its own forward. Peak GPU use is one
    layer plus activations (~2-3 GB for a 7B at 2K tokens)."""
    def swap(mod, out_cpu=False):
        host = {n: p.data.pin_memory() for n, p in mod.named_parameters()}
        mod._host = host

        def pre(module, args):
            for n, p in module.named_parameters():
                p.data = module._host[n].to(dev, non_blocking=True)

        def post(module, args, out):
            for n, p in module.named_parameters():
                p.data = module._host[n]
            return out.to("cpu") if out_cpu else out

        mod.register_forward_pre_hook(pre)
        mod.register_forward_hook(post)

    for layer in m.model.layers:
        swap(layer)
    swap(m.lm_head, out_cpu=True)  # 2K x 152K logits stay off the GPU
    m.model.norm.to(dev)
    m.model.rotary_emb.to(dev)
    m.model.embed_tokens.register_forward_hook(lambda mod, a, out: out.to(dev))


def wiki_ids(tok, model):
    if model == "smol135":  # identical token stream to kv_ppl.py / tensors_ppl.py
        return np.load(HERE / "tensors/tokens.npz")["wiki"].astype(np.int64)
    import pyarrow.parquet as pq

    text = "".join(pq.read_table(WIKI).column("text").to_pylist())
    return np.array(tok(text)["input_ids"], np.int64)


def forward(m, ids, dev, mode, rec=None, R=0, q0=0, keep=0):
    STATE.update(mode=mode, rec=rec, R=R, q0=q0, cap={} if mode == "capture" else None)
    with torch.no_grad():
        lo = m(torch.from_numpy(ids)[None].to(m.model.embed_tokens.weight.device), logits_to_keep=keep).logits[0].float()
    return lo, STATE["cap"]


def _job(a):
    name, t = a
    t0 = time.perf_counter()
    bpv, y, dom = B.CONFIGS[name](t, None)
    return bpv, y.astype(np.float32), dom, time.perf_counter() - t0


def window_of(name):
    r = re.search(r"-R(\d+)-", name)
    return int(r.group(1)) if r else 0


def compress(ex, name, cap, calib, tc=None):
    """cap: {layer: (kpre, kpost, v)} -> rec dict, bits, values, err, nrm (post-RoPE domain)."""
    jobs, keys = [], []
    for l, (kpre, kpost, v) in cap.items():
        sl = slice(0, tc)
        for is_key, pre, post in ((True, kpre[:, sl], kpost[:, sl]), (False, v[:, sl], v[:, sl])):
            dom_cal = None
            if calib is not None:
                c = calib[l]
                dom_cal = c[0] if is_key else c[2]
            jobs.append((name, dict(pre=np.ascontiguousarray(pre), post=np.ascontiguousarray(post), key=is_key, calib=dom_cal)))
            keys.append((l, is_key))
    rec = {l: {} for l in cap}
    bits = nv = 0.0
    for (l, is_key), (job, (bpv, y, dom, _)) in zip(keys, zip(jobs, ex.map(_job, jobs))):
        bits += bpv * y.size
        nv += y.size
        if is_key:
            rec[l]["K"], rec[l]["Kdom"] = torch.from_numpy(y), dom
        else:
            rec[l]["V"] = torch.from_numpy(y)
    return rec, bits / nv


def cmd_ppl(a, tok, m, dev, ex):
    ids = wiki_ids(tok, a.model)
    W, N = a.win, a.nwin
    assert len(ids) > (N + 1) * W + 1, len(ids)
    wins = [ids[w * W : w * W + W + 1] for w in range(N)]
    # calibration window (for KVQuant-style offline ranges / codebooks): disjoint, after the eval windows
    _, calib = forward(m, ids[N * W : N * W + W], dev, "capture")
    nll, caps = 0.0, []
    for x in wins:
        lo, cap = forward(m, x[:-1], dev, "capture")
        caps.append(cap)
        nll += torch.nn.functional.cross_entropy(lo, torch.from_numpy(x[1:]).to(lo.device), reduction="sum").item()
    ref = math.exp(nll / (N * W))
    print(f"{a.model} ref ppl {ref:.4f} (win {W} x {N})", flush=True)
    names = [n for n in B.CONFIGS if any(n == g or (g.endswith("*") and n.startswith(g[:-1])) for g in a.configs)] if a.configs else list(B.CONFIGS)
    OUT.mkdir(exist_ok=True)
    path = OUT / f"kvhf_ppl_{a.model}_w{W}.csv"
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["config", "bpv", "relL2", "ppl", "dppl", "ref"])
        for n in names:
            R = window_of(n)
            nll = bits = err = nrm = 0.0
            for x, cap in zip(wins, caps):
                rec, bpv = compress(ex, n, cap, calib)
                bits += bpv
                for l, (kpre, kpost, v) in cap.items():
                    # relL2 measured post-RoPE (attention's view), window ignored
                    kh = rec[l]["K"].numpy()
                    if rec[l]["Kdom"] == "pre":
                        kh = B_rope(kh, kpre, kpost)
                    err += float(((kh - kpost) ** 2).sum() + ((rec[l]["V"].numpy() - v) ** 2).sum())
                    nrm += float((kpost.astype(np.float64) ** 2).sum() + (v.astype(np.float64) ** 2).sum())
                lo, _ = forward(m, x[:-1], dev, "override", rec, R)
                nll += torch.nn.functional.cross_entropy(lo, torch.from_numpy(x[1:]).to(lo.device), reduction="sum").item()
            p = math.exp(nll / (N * W))
            row = [n, f"{bits / N:.4f}", f"{math.sqrt(err / nrm):.5f}", f"{p:.4f}", f"{p - ref:.4f}", f"{ref:.4f}"]
            w.writerow(row)
            f.flush()
            print(f"{n:28s} bpv {bits / N:6.3f} rel {math.sqrt(err / nrm):.4f} ppl {p:.4f} d {p - ref:+.4f}", flush=True)


def B_rope(kh_pre, kpre, kpost):
    """Re-rotate a pre-RoPE reconstruction using the exact per-(t, pair) rotation
    implied by the clean pre/post pair (numpy, for the relL2 stat only)."""
    h = kpre.shape[-1] // 2
    a, b = kpre[..., :h].astype(np.float64), kpre[..., h:].astype(np.float64)
    c_, d_ = kpost[..., :h].astype(np.float64), kpost[..., h:].astype(np.float64)
    n2 = np.maximum(a * a + b * b, 1e-30)
    cos = (a * c_ + b * d_) / n2
    sin = (a * d_ - b * c_) / n2
    nrm = np.sqrt(cos**2 + sin**2)
    cos, sin = cos / np.maximum(nrm, 1e-30), sin / np.maximum(nrm, 1e-30)
    x, y = kh_pre[..., :h], kh_pre[..., h:]
    return np.concatenate([x * cos - y * sin, y * cos + x * sin], -1)


# ------------------------------------------------------------------ NIAH
NEEDLE = "The special magic number for {city} is {num}."
QUESTION = "\nWhat is the special magic number for {city}? The special magic number for {city} is"
CITIES = ["Lisbon", "Nairobi", "Osaka", "Quito", "Tbilisi", "Perth", "Bergen", "Cusco", "Hanoi", "Dakar"]


def niah_cases(tok, ids_hay, lengths, depths, seed=0):
    rng = np.random.default_rng(seed)
    cases = []
    for L in lengths:
        for di, dep in enumerate(depths):
            city = CITIES[(len(cases)) % len(CITIES)]
            num = str(rng.integers(1_000_000, 9_999_999))
            nd = tok(" " + NEEDLE.format(city=city, num=num) + " ", add_special_tokens=False)["input_ids"]
            q = tok(QUESTION.format(city=city), add_special_tokens=False)["input_ids"]
            ans = tok(" " + num, add_special_tokens=False)["input_ids"]
            off = int(rng.integers(0, len(ids_hay) - L))
            hay = list(ids_hay[off : off + L - len(nd)])
            pos = int(dep * len(hay))
            ctx = hay[:pos] + nd + hay[pos:]
            cases.append(dict(L=L, depth=dep, ctx=np.array(ctx, np.int64), q=np.array(q + ans, np.int64), nans=len(ans)))
    return cases


def cmd_niah(a, tok, m, dev, ex):
    ids = wiki_ids(tok, a.model)
    cases = niah_cases(tok, ids, a.lengths, a.depths)
    _, calib = forward(m, ids[: a.win], dev, "capture")
    names = a.configs or ["exact"]
    OUT.mkdir(exist_ok=True)
    path = OUT / f"kvhf_niah_{a.model}.csv"
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["config", "L", "depth", "bpv", "correct", "ans_nll"])
        for n in names:
            acc, nlls, bl = [], [], []
            for c in cases:
                full = np.concatenate([c["ctx"], c["q"]])
                tc = len(c["ctx"])
                if n == "exact":
                    lo, _ = forward(m, full, dev, "clean", keep=len(c["q"]) + 1)
                    bpv = 16.0
                else:
                    _, cap = forward(m, full, dev, "capture", keep=1)
                    rec, bpv = compress(ex, n, cap, calib, tc)
                    lo, _ = forward(m, full, dev, "override", rec, window_of(n), q0=tc - 1, keep=len(c["q"]) + 1)
                na = c["nans"]
                tgt = torch.from_numpy(full[-na:]).to(lo.device)
                pl = lo[-na - 1 : -1]
                ok = bool((pl.argmax(-1) == tgt).all())
                nl = torch.nn.functional.cross_entropy(pl, tgt, reduction="sum").item()
                acc.append(ok)
                nlls.append(nl)
                bl.append(bpv)
                w.writerow([n, c["L"], c["depth"], f"{bpv:.4f}", int(ok), f"{nl:.4f}"])
                f.flush()
            print(f"{n:28s} bpv {np.mean(bl):6.3f} acc {np.mean(acc):.3f} ans_nll {np.mean(nlls):.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ppl", "niah"])
    ap.add_argument("--model", default="smol135")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--win", type=int, default=1024)
    ap.add_argument("--nwin", type=int, default=4)
    ap.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192])
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument("-c", "--configs", nargs="*", default=[], help="names; trailing * = prefix")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "6")))
    if a.device.startswith("cuda") and os.environ.get("KV_GPU_GB"):  # hard cap on a shared GPU
        tot = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(float(os.environ["KV_GPU_GB"]) * 2**30 / tot)
    tok, m = load(a.model, a.device, getattr(torch, a.dtype))
    with ProcessPoolExecutor(int(os.environ.get("KV_PROCS", "6"))) as ex:
        (cmd_ppl if a.cmd == "ppl" else cmd_niah)(a, tok, m, a.device, ex)


if __name__ == "__main__":
    main()
