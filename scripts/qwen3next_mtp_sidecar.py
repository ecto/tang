#!/usr/bin/env python3
"""Qwen3-Next's multi-token-prediction layer as a sidecar GGUF for the Coder engine.

Qwen3-Coder-Next ships without MTP weights; Qwen3-Next-80B-A3B-Instruct has a one-layer MTP head
of the same shape (hidden 2048, 512 experts, gated attention), and drafts for Coder-Next at ~78%
k=1 acceptance as is. This writes that head as `mtp.*` tensors the engine loads with `--mtp`
(or `TANG_CODER_MTP`):

    mtp.fc.weight                 [2n -> n]   input projection of [rmsnorm(embed) | rmsnorm(hidden)]
    mtp.enorm / mtp.hnorm         the two input norms
    mtp.attn_norm, mtp.attn_q/k/v/output, mtp.attn_q_norm/k_norm, mtp.post_attention_norm,
    mtp.ffn_gate_inp(_shexp), mtp.ffn_{gate,up,down}_exps, mtp.ffn_{gate,up,down}_shexp
    mtp.norm                      the output norm before the shared head

Tensors follow the trunk GGUF's conventions, checked rather than assumed: for the trunk's first
attention layer, each kind of tensor in the GGUF is compared with the HF checkpoint it came from
(norm weights stored as 1 + w or w; matrices in the HF layout), and the same transform is applied
to the MTP tensors. Matrices are Q8_0, norms F32.

    python3 scripts/qwen3next_mtp_sidecar.py --trunk Qwen3-Coder-Next-UD-Q4_K_XL.gguf --out mtp.gguf

Needs `gguf`, `safetensors`, `huggingface_hub`, `numpy`, `torch` (bf16 reads).
"""
import argparse
import json
import os

import numpy as np
import torch
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType
from gguf.quants import dequantize, quantize
from huggingface_hub import hf_hub_download
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--trunk", required=True, help="the Coder-Next GGUF the sidecar goes with")
ap.add_argument("--out", required=True)
ap.add_argument("--head", default="Qwen/Qwen3-Next-80B-A3B-Instruct", help="where the MTP layer comes from")
ap.add_argument("--ref", default="Qwen/Qwen3-Coder-Next", help="the trunk's HF checkpoint (for the convention checks)")
a = ap.parse_args()

trunk = GGUFReader(a.trunk)
tt = {t.name: t for t in trunk.tensors}
L = next(l for l in range(64) if f"blk.{l}.attn_q.weight" in tt)
print(f"trunk: first attention layer blk.{L}")


def gguf_f32(name):
    t = tt[name]
    shape = [int(x) for x in reversed(t.shape)]
    if t.tensor_type in (GGMLQuantizationType.F32, GGMLQuantizationType.F16, GGMLQuantizationType.BF16):
        return np.asarray(t.data, dtype=np.float32).reshape(shape)
    return dequantize(t.data, t.tensor_type).reshape(shape).astype(np.float32)


class Hf:
    def __init__(self, repo):
        self.repo = repo
        self.idx = json.load(open(hf_hub_download(repo, "model.safetensors.index.json")))["weight_map"]
        self.files = {}

    def get(self, name):
        shard = self.idx[name]
        if shard not in self.files:
            self.files[shard] = safe_open(hf_hub_download(self.repo, shard), "pt")
        return self.files[shard].get_tensor(name).float().numpy()


ref, head = Hf(a.ref), Hf(a.head)


def corr(x, y):
    x, y = x.ravel().astype(np.float64), y.ravel().astype(np.float64)
    return float(np.corrcoef(x, y)[0, 1])


# ---- conventions, from the trunk's attention layer L vs its HF weights
hl = f"model.layers.{L}."
NORMS = {  # gguf suffix -> hf suffix
    "attn_norm": "input_layernorm",
    "post_attention_norm": "post_attention_layernorm",
    "attn_q_norm": "self_attn.q_norm",
    "attn_k_norm": "self_attn.k_norm",
}
offset = {}
for g, h in NORMS.items():
    gv, hv = gguf_f32(f"blk.{L}.{g}.weight"), ref.get(hl + h + ".weight")
    if np.allclose(gv, hv + 1, atol=1e-2):
        offset[g] = 1.0
    elif np.allclose(gv, hv, atol=1e-2):
        offset[g] = 0.0
    else:
        raise SystemExit(f"{g}: GGUF is neither w nor 1 + w (max diff {np.abs(gv - hv).max():.3g})")
on = gguf_f32("output_norm.weight")
hn = ref.get("model.norm.weight")
offset["output_norm"] = 1.0 if np.allclose(on, hn + 1, atol=1e-2) else 0.0
assert np.allclose(on, hn + offset["output_norm"], atol=1e-2), "output_norm convention"
print("norm offsets:", offset)

MATS = {  # gguf suffix -> hf suffix (identity layout expected)
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_gate_inp": "mlp.gate",
    "ffn_gate_inp_shexp": "mlp.shared_expert_gate",
    "ffn_gate_shexp": "mlp.shared_expert.gate_proj",
    "ffn_up_shexp": "mlp.shared_expert.up_proj",
    "ffn_down_shexp": "mlp.shared_expert.down_proj",
}
for g, h in MATS.items():
    gv, hv = gguf_f32(f"blk.{L}.{g}.weight"), ref.get(hl + h + ".weight")
    assert gv.size == hv.size, f"{g}: {gv.shape} vs {hv.shape}"
    c = corr(gv.reshape(hv.shape), hv)
    print(f"  {g}: layout corr {c:.4f}")
    assert c > 0.98, f"{g}: GGUF layout differs from HF (corr {c:.3f})"
for g, h in [("ffn_gate_exps", "gate_proj"), ("ffn_up_exps", "up_proj"), ("ffn_down_exps", "down_proj")]:
    gv = gguf_f32(f"blk.{L}.{g}.weight")
    for e in (0, 311):
        c = corr(gv[e], ref.get(hl + f"mlp.experts.{e}.{h}.weight"))
        assert c > 0.98, f"{g}[{e}]: corr {c:.3f}"
    print(f"  {g}: per-expert layout ok")

# ---- the MTP layer
p = "mtp.layers.0."
out = {}
for g, h in NORMS.items():
    out[f"mtp.{g}.weight"] = head.get(p + h + ".weight") + offset[g]
for g, h in MATS.items():
    out[f"mtp.{g}.weight"] = head.get(p + h + ".weight")
E = int(tt[f"blk.{L}.ffn_gate_exps.weight"].shape[-1])
for g, h in [("ffn_gate_exps", "gate_proj"), ("ffn_up_exps", "up_proj"), ("ffn_down_exps", "down_proj")]:
    out[f"mtp.{g}.weight"] = np.stack([head.get(p + f"mlp.experts.{e}.{h}.weight") for e in range(E)])
out["mtp.fc.weight"] = head.get("mtp.fc.weight")
out["mtp.enorm.weight"] = head.get("mtp.pre_fc_norm_embedding.weight") + offset["output_norm"]
out["mtp.hnorm.weight"] = head.get("mtp.pre_fc_norm_hidden.weight") + offset["output_norm"]
out["mtp.norm.weight"] = head.get("mtp.norm.weight") + offset["output_norm"]

w = GGUFWriter(a.out, "qwen3next-mtp")
w.add_name("Qwen3-Next-80B-A3B-Instruct MTP layer, for Qwen3-Coder-Next")
w.add_string("general.license", "apache-2.0")
w.add_string("general.source.repo", a.head)
w.add_string("mtp.trunk", os.path.basename(a.trunk))
for name, v in out.items():
    v = np.ascontiguousarray(v, dtype=np.float32)
    suffix = name[len("mtp."):-len(".weight")]
    ref_name = f"blk.{L}.{suffix}.weight"
    if ref_name in tt:
        want = [int(x) for x in reversed(tt[ref_name].shape)]
        assert list(v.shape) == want or v.size == int(np.prod(want)), f"{name}: {v.shape} vs trunk {want}"
        v = v.reshape(want)
    if v.ndim == 1 or v.shape[-1] % 32:
        w.add_tensor(name, v)
    else:
        w.add_tensor(name, quantize(v, GGMLQuantizationType.Q8_0), raw_dtype=GGMLQuantizationType.Q8_0)
    print(f"{name}: {list(v.shape)}")
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
with open(a.out + ".NOTICE", "w") as f:
    f.write(
        "This file contains the multi-token-prediction (MTP) layer of Qwen3-Next-80B-A3B-Instruct\n"
        "(https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct), Copyright Alibaba Cloud / the\n"
        "Qwen team, licensed under the Apache License, Version 2.0.\n"
        "Changes: converted to GGUF (Q8_0 matrices, F32 norms stored in the trunk GGUF's convention)\n"
        "for use as a draft layer with Qwen3-Coder-Next by tang-llm.\n"
    )
print("wrote", a.out)
