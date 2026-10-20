#!/usr/bin/env python3
"""Dump KV caches, activations and gradients from SmolLM2-135M (needs torch).

A ~100-line Llama forward written against the raw safetensors (no
transformers). Writes bench/tensors/{kv,act,grad}/*.npy (f32) and
bench/tensors/dump_meta.json. Also exposes `Llama` for tensors_ppl.py.

Token budget (WikiText-2 test, 45,960 SmolLM2 tokens): [0, 8192) is the
perplexity eval set; KV/activations come from [16384, 18432) (two 1024-token
windows) plus greedy continuation of 4 fixed prompts to 512 tokens; gradients
come from 3 AdamW steps (lr 1e-5, batch 4x256) on [24576, 28672).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tensors_io import SMOL, T, load_safetensors  # noqa: E402

CFG = dict(d=576, ff=1536, L=30, H=9, KV=3, hd=64, theta=1e5, eps=1e-5)


class Llama(torch.nn.Module):
    def __init__(self, w: dict[str, np.ndarray], dev="cpu"):
        super().__init__()
        self.p = torch.nn.ParameterDict(
            {k.replace(".", "__"): torch.nn.Parameter(torch.from_numpy(v.copy()).to(dev)) for k, v in w.items()}
        )
        hd = CFG["hd"]
        self.inv = 1.0 / CFG["theta"] ** (torch.arange(0, hd, 2, device=dev).float() / hd)
        self.dev = dev

    def W(self, k):
        return self.p[k.replace(".", "__")]

    def rms(self, x, k):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + CFG["eps"]) * self.W(k)

    def rope(self, x, pos):
        f = pos[:, None].float() * self.inv[None]
        c, s = torch.cat([f.cos()] * 2, -1), torch.cat([f.sin()] * 2, -1)
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return x * c + torch.cat([-x2, x1], -1) * s

    def forward(self, ids, kv_hook=None, act_hook=None, kv_override=None):
        """ids [B,T]. kv_hook(l, k, v) sees post-RoPE K and V [B,KV,T,hd];
        kv_override(l, k, v) -> (k, v) replaces them (KV-cache compression)."""
        B, Tn = ids.shape
        H, KV, hd = CFG["H"], CFG["KV"], CFG["hd"]
        pos = torch.arange(Tn, device=self.dev)
        x = self.W("model.embed_tokens.weight")[ids]
        for l in range(CFG["L"]):
            p = f"model.layers.{l}."
            h = self.rms(x, p + "input_layernorm.weight")
            q = (h @ self.W(p + "self_attn.q_proj.weight").T).view(B, Tn, H, hd).transpose(1, 2)
            k = (h @ self.W(p + "self_attn.k_proj.weight").T).view(B, Tn, KV, hd).transpose(1, 2)
            v = (h @ self.W(p + "self_attn.v_proj.weight").T).view(B, Tn, KV, hd).transpose(1, 2)
            q, k = self.rope(q, pos), self.rope(k, pos)
            if kv_hook:
                kv_hook(l, k, v)
            if kv_override:
                k, v = kv_override(l, k, v)
            k, v = k.repeat_interleave(H // KV, 1), v.repeat_interleave(H // KV, 1)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, Tn, -1)
            x = x + a @ self.W(p + "self_attn.o_proj.weight").T
            h = self.rms(x, p + "post_attention_layernorm.weight")
            g = F.silu(h @ self.W(p + "mlp.gate_proj.weight").T) * (h @ self.W(p + "mlp.up_proj.weight").T)
            if act_hook:
                act_hook(l, "mlp_act", g)
            x = x + g @ self.W(p + "mlp.down_proj.weight").T
            if act_hook:
                act_hook(l, "resid", x)
        x = self.rms(x, "model.norm.weight")
        return x @ self.W("model.embed_tokens.weight").T


def smol_weights():
    return {k: v for k, (v, _) in load_safetensors(SMOL).items()}


def ppl(model, ids, win=1024, **kw):
    nll, n = 0.0, 0
    with torch.no_grad():
        for s in range(0, len(ids) - 1, win):
            x = torch.from_numpy(ids[s : s + win + 1].astype(np.int64)).to(model.dev)[None]
            lo = model(x[:, :-1], **kw)
            nll += F.cross_entropy(lo[0].float(), x[0, 1:], reduction="sum").item()
            n += x.shape[1] - 1
    return float(np.exp(nll / n))


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = np.load(T / "tokens.npz")
    wiki = tok["wiki"]
    m = Llama(smol_weights(), dev)
    for d in ("kv", "act", "grad"):
        (T / d).mkdir(exist_ok=True)
    meta = {"model": "HuggingFaceTB/SmolLM2-135M", "device": dev}
    meta["ppl_fp32"] = ppl(m, wiki[:8192])
    print("ppl", meta["ppl_fp32"])

    # --- KV caches + activations: two 1024-token wiki windows
    LAY = (0, 7, 15, 22, 29)
    for w, s in enumerate((16384, 17408)):
        x = torch.from_numpy(wiki[s : s + 1024].astype(np.int64)).to(dev)[None]

        def kvh(l, k, v, w=w):
            if l in LAY:
                np.save(T / f"kv/wiki{w}_L{l}_K.npy", k[0].float().cpu().numpy())
                np.save(T / f"kv/wiki{w}_L{l}_V.npy", v[0].float().cpu().numpy())

        def acth(l, name, a, w=w):
            if l in LAY and w == 0:
                np.save(T / f"act/wiki0_L{l}_{name}.npy", a[0].float().cpu().numpy())

        with torch.no_grad():
            m(x, kv_hook=kvh, act_hook=acth)

    # --- KV from greedy generation on the prompts (512 tokens, no cache: re-run)
    for i in range(2):
        ids = list(tok[f"prompt{i}"])
        with torch.no_grad():
            while len(ids) < 512:
                lo = m(torch.tensor(ids, device=dev)[None])
                ids.append(int(lo[0, -1].argmax()))
            caps = {}
            m(torch.tensor(ids, device=dev)[None], kv_hook=lambda l, k, v: caps.update({l: (k, v)}) if l in (0, 15, 29) else None)
        for l, (k, v) in caps.items():
            np.save(T / f"kv/gen{i}_L{l}_K.npy", k[0].float().cpu().numpy())
            np.save(T / f"kv/gen{i}_L{l}_V.npy", v[0].float().cpu().numpy())

    # --- gradients + Adam state: 3 AdamW steps on batch 4x256
    keep = [f"model.layers.{l}.{n}.weight" for l in (0, 15, 29)
            for n in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                      "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")]
    opt = torch.optim.AdamW(m.parameters(), lr=1e-5)
    for step in range(3):
        s = 24576 + step * 1024
        x = torch.from_numpy(wiki[s : s + 1028].astype(np.int64)).to(dev).view(4, 257)
        lo = m(x[:, :-1])
        loss = F.cross_entropy(lo.reshape(-1, lo.shape[-1]).float(), x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        print("step", step, "loss", loss.item())
        if step in (0, 2):
            for k in keep:
                np.save(T / f"grad/step{step}_{k}.npy", m.W(k).grad.float().cpu().numpy())
        opt.step()
    for k in keep[::3]:
        st = opt.state[m.W(k)]
        np.save(T / f"grad/adam_m_{k}.npy", st["exp_avg"].float().cpu().numpy())
        np.save(T / f"grad/adam_v_{k}.npy", st["exp_avg_sq"].float().cpu().numpy())
    (T / "dump_meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
