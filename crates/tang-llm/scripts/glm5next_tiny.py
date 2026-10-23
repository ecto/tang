#!/usr/bin/env python3
"""GLM-5.3-Flash (Glm5Next) tiny-model harness: HF fp32 reference + GGUF conversion + dumps.

W0 harness for the tang port. Builds a TINY Glm5Next text model that exercises every
component of the real architecture, saves it to /private/tmp/glm5-tiny/hf, dumps fp32
reference outputs (embeddings, per-layer mHC stream outputs after every block, final
hidden, logits) for fixed token ids, and converts to GGUF with llama.cpp's
convert_hf_to_gguf.py (f32), including a synthesized NextN/MTP block so the GGUF has 46
blocks like the real UD-Q4_K_XL file.

Tiny shape choices (real -> tiny):
  hidden 4096 -> 256, layers 45 (34 KDA + 11 DSA, every 4th) -> 5 (3 KDA + 1 DSA at 3,
  like the real 0..3 pattern scaled), dense lead 3 -> 2, dense FFN 12288 -> 768,
  experts 288 top-8 + 1 shared, FFN 2048 -> 8 top-2 + shared, FFN 128, mHC 4 streams
  (same), KDA heads 64 x 128 -> 4 x 64, conv k=4 (same), gate lower bound -5 (same),
  DSA: q_lora 1536 -> 64, kv_lora 512 -> 64, qk_nope 256 -> 64, v_head 256 -> 32,
  NoPE (same), indexer 32 heads x 128, kpool 4, top-2048 -> 4 heads x 32, kpool 4,
  topk 16, vocab 154880 (kept so the real tokenizer works unchanged), swiglu_limit 10
  (same), routed scale 2.5, sigmoid router + noaux_tc bias (same).

Usage:
  /private/tmp/glm5-venv/bin/python glm5next_tiny.py --out /private/tmp/glm5-tiny \
      --llama-cpp /private/tmp/llama.cpp

Outputs under --out:
  hf/                       HF checkpoint (config, safetensors incl. synthesized MTP block)
  ref/                      fp32 dumps (see DUMP_FORMAT below) + meta.json
  glm5-tiny-f32.gguf        converted GGUF (f32, 46 blocks incl. MTP)

DUMP_FORMAT (little-endian f32 .bin + sidecar .json {"shape": [...]}), row-major:
  embed.bin      [T, H]        embedding output (single stream; runtime replicates to N=4)
  layerB{i}.bin  [T, 4, H]     mHC stream output after block i (attn + ffn combined)
  final.bin      [T, H]        after final norm + stream mean (hc_head)
  logits.bin     [T, V]        lm_head output (seq1 only)
  meta.json      lengths + token ids
seq1 = 64-token prompt; seq2 = 2052 tokens (>2048 so the indexer top-16/kpool-4 actually
drops keys; dumps layerB{i}_seq2.bin for the second sequence).
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

from transformers import AutoTokenizer
from transformers.models.glm5_next.configuration_glm5_next import (
    Glm5NextConfig,
    Glm5NextTextConfig,
)
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextForConditionalGeneration

TOKENIZER_REPO = "zai-org/GLM-5.3-Flash"  # tokenizer files only, no weights
PROMPT = (
    "def quicksort(xs):\n"
    "    \"\"\"Sort a list with quicksort.\"\"\"\n"
    "    if len(xs) <= 1:\n"
    "        return xs\n"
    "    pivot = xs[len(xs) // 2]\n"
    "    left = [x for x in xs if x < pivot]\n"
)

# 5 trunk layers: every 4th (index 3) is DSA like the real model's 0..11 pattern; the MTP
# block (layer 5, DSA) is synthesized in the safetensors, giving 46-block GGUFs like real.
TINY = dict(
    vocab_size=154880,             # real vocab, so the real tokenizer ids work as-is
    hidden_size=256,
    intermediate_size=768,         # dense FFN (first 2 layers)
    moe_intermediate_size=128,
    num_hidden_layers=5,
    num_attention_heads=4,
    num_key_value_heads=4,
    q_lora_rank=64,
    kv_lora_rank=64,
    qk_nope_head_dim=64,
    qk_rope_head_dim=0,           # NoPE, asserted by the converter
    v_head_dim=32,
    n_routed_experts=8,
    n_shared_experts=1,
    num_experts_per_tok=2,
    norm_topk_prob=True,
    routed_scaling_factor=2.5,
    layer_types=["linear_attention", "linear_attention", "linear_attention",
                 "deepseek_sparse_attention", "linear_attention"],
    mlp_layer_types=["dense", "dense", "sparse", "sparse", "sparse"],
    indexer_types=["full"] * 5,
    index_topk=16,
    index_kpool=4,
    index_n_heads=4,
    index_head_dim=32,
    max_position_embeddings=4096,
    swiglu_limit=10.0,
    linear_head_dim=64,
    linear_num_heads=4,
    linear_conv_kernel_dim=4,
    linear_lower_bound=-5.0,
    tie_word_embeddings=False,
    pad_token_id=154820,          # the real model's pad id
    rms_norm_eps=1e-5,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    linear_num_key_heads=4,       # unused by glm5_next, kept out
)
TINY.pop("linear_num_key_heads")

# vision tower shrunk to nothing-ish: the port is text-only; a 1-block 64-dim tower is
# cheap and keeps the checkpoint loadable by stock transformers.
VISION = dict(hidden_size=64, depth=1, num_heads=4, out_hidden_size=64,
              projection_intermediate_size=64, intermediate_size=64,
              image_size=56, patch_size=28, spatial_merge_size=2)


def _init_tiny(model):
    """Random init at scales that keep fp32 logits spread out; fixed seed handled by caller."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "visual" in name:
                continue
            if name.endswith("A_log"):
                # the published checkpoints store -exp(A_log_init); the safe-gate path zeros
                # it in HF init. zero keeps the HF reference and the checkpoint identical.
                p.zero_()
            elif name.endswith("dt_bias"):
                g = torch.empty_like(p)
                torch.nn.init.uniform_(g, -6.9, -2.3)  # log(1e-3)..log(1e-1) like HF init
                dt = g.exp().clamp_min(1e-4)
                p.copy_(dt + torch.log(-torch.expm1(-dt)))
            elif name.endswith(("hc_attn_scale", "hc_ffn_scale")):
                p.fill_(1.0)
            elif name.endswith("e_score_correction_bias"):
                p.normal_(0.0, 0.3)  # noaux_tc bias: nonzero so selection != weighting
            elif name.endswith("index_kpool_compress_ape"):
                p.normal_(0.0, 0.2)
            elif name.endswith(("hc_attn_base", "hc_ffn_base")):
                p.normal_(0.0, 0.2)
            elif "conv1d" in name:
                p.normal_(0.0, 0.3)
            elif "norm" in name or name.endswith("layernorm"):
                p.normal_(1.0, 0.2)
            elif "embed_tokens" in name:
                p.normal_(0.0, 1.0)
            elif name == "lm_head.weight":
                p.normal_(0.0, 8.0 / p.shape[-1] ** 0.5)
            else:
                p.normal_(0.0, 1.0 / max(1.0, p.shape[-1]) ** 0.5)


def build(hf_dir, seed=0):
    torch.manual_seed(seed)
    cfg = Glm5NextConfig(text_config=Glm5NextTextConfig(**TINY), **VISION)
    # Glm5NextConfig takes vision_config via kwargs of __post_init__; pass properly:
    # (sub-configs go through the SubConfigSpec) -> rebuild to be explicit:
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig
    cfg = Glm5NextConfig(text_config=Glm5NextTextConfig(**TINY),
                         vision_config=Glm5NextVisionConfig(**VISION))
    model = Glm5NextForConditionalGeneration(cfg).float().eval()
    _init_tiny(model)
    os.makedirs(hf_dir, exist_ok=True)
    model.save_pretrained(hf_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(TOKENIZER_REPO).save_pretrained(hf_dir)
    return model


def synthesize_mtp(hf_dir, seed=0):
    """Append a NextN/MTP block as model.layers.<n> in the safetensors + fix config.json.

    transformers 5.2x has no MTP support for glm5_next (no num_nextn_predict_layers field,
    no first_k_dense_replace / linear_attn_config / mla_use_nope / mhc keys, which the real
    zai config.json carries and llama.cpp's converter requires). We synthesize:

      1. config.json text_config += the keys the converter needs (mirroring the real
         zai-org/GLM-5.3-Flash config layout).
      2. a 6th block (DSA layer copied from layer 3 + fresh eh_proj/enorm/hnorm/
         shared_head.norm) so the converter emits the 46-block layout the real UD-Q4_K_XL
         GGUF has (block 45 = MTP).
    """
    from safetensors.torch import load_file, save_file
    n_main = TINY["num_hidden_layers"]
    torch.manual_seed(seed + 1)
    tensors = load_file(os.path.join(hf_dir, "model.safetensors"))
    out = dict(tensors)
    # copy layer 3 (DSA) weights to layer 5 = n_main; skip the mHC tensors
    # (saved as flat 'hc_attn_fn'/'hc_ffn_*' keys, like the real zai checkpoint) —
    # llama.cpp never loads hc_* for the NextN block and the real UD GGUF's blk.45
    # carries no hc_* either
    src = "model.language_model.layers.3."
    dst = f"model.language_model.layers.{n_main}."
    for k, v in tensors.items():
        if k.startswith(src) and ".hc_attn_" not in k and ".hc_ffn_" not in k:
            out[dst + k[len(src):]] = v.clone()
    # fresh nextn projections
    h = TINY["hidden_size"]
    out[dst + "eh_proj.weight"] = torch.randn(h, 2 * h) / (2 * h) ** 0.5  # Linear(2*n_embd -> n_embd), GGUF ne [2h, h]
    out[dst + "enorm.weight"] = torch.ones(h) + 0.1 * torch.randn(h)
    out[dst + "hnorm.weight"] = torch.ones(h) + 0.1 * torch.randn(h)
    out[dst + "shared_head.norm.weight"] = torch.ones(h) + 0.1 * torch.randn(h)
    save_file(out, os.path.join(hf_dir, "model.safetensors"))

    # patch config.json: keys the real checkpoint carries at the text_config level
    cpath = os.path.join(hf_dir, "config.json")
    with open(cpath) as f:
        cfg = json.load(f)
    tc = cfg["text_config"]
    tc["num_nextn_predict_layers"] = 1
    tc["first_k_dense_replace"] = 2  # our mlp_layer_types has 2 dense layers
    tc["linear_attn_config"] = {
        "num_heads": TINY["linear_num_heads"],
        "head_dim": TINY["linear_head_dim"],
        "short_conv_kernel_size": TINY["linear_conv_kernel_dim"],
        "gate_lower_bound": TINY["linear_lower_bound"],
        "safe_gate": True,
    }
    tc["mla_use_nope"] = True
    tc["mhc"] = True
    tc["scoring_func"] = "sigmoid"
    tc["topk_method"] = "noaux_tc"
    tc["moe_router_dtype"] = "float32"
    with open(cpath, "w") as f:
        json.dump(cfg, f, indent=2)
    return n_main


def dump(path_stem, t):
    t = t.detach().float().contiguous().cpu()
    t.numpy().tofile(path_stem + ".bin")
    with open(path_stem + ".json", "w") as f:
        json.dump({"shape": list(t.shape)}, f)


def run_and_dump(model, ids, ref_dir, suffix=""):
    """One fp32 forward; returns (logits, per-block mHC stream outputs, final hidden)."""
    n_layers = model.config.text_config.num_hidden_layers
    taps = {}
    hooks = []
    for i, layer in enumerate(model.model.language_model.layers):
        def mk(i):
            def hook(mod, args, output):
                taps[i] = output[0].detach().float().cpu()
            return hook
        hooks.append(layer.register_forward_hook(mk(i)))
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]))
        # final hidden: rerun text model to get last_hidden_state (post norm + hc mean)
        text_out = model.model.language_model(input_ids=torch.tensor([ids]))
    for h_ in hooks:
        h_.remove()
    for i in range(n_layers):
        assert taps[i].shape[2] == TINY["hc_mult"], (i, taps[i].shape)
        dump(os.path.join(ref_dir, f"layerB{i}{suffix}"), taps[i][0])  # [T, 4, H]
    dump(os.path.join(ref_dir, f"final{suffix}"), text_out.last_hidden_state[0])
    return out.logits[0], text_out.last_hidden_state[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/private/tmp/glm5-tiny")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--llama-cpp", default="/private/tmp/llama.cpp")
    ap.add_argument("--skip-build", action="store_true", help="reuse existing hf dir")
    ap.add_argument("--no-convert", action="store_true")
    a = ap.parse_args()

    hf_dir = os.path.join(a.out, "hf")
    ref_dir = os.path.join(a.out, "ref")
    os.makedirs(ref_dir, exist_ok=True)

    if a.skip_build:
        model = Glm5NextForConditionalGeneration.from_pretrained(hf_dir).float().eval()
    else:
        model = build(hf_dir, a.seed)
    if not a.skip_build:
        synthesize_mtp(hf_dir, a.seed)
        # reload so state_dict matches what's on disk (MTP block is ignored by HF)
        model = Glm5NextForConditionalGeneration.from_pretrained(hf_dir).float().eval()

    tok = AutoTokenizer.from_pretrained(hf_dir)
    ids1 = tok(PROMPT)["input_ids"]
    ids2 = [(i * 7 + 3) % 150000 + 20 for i in range(2052)]
    print(f"seq1 {len(ids1)} tokens; seq2 {len(ids2)} tokens")

    with torch.no_grad():
        emb = model.model.language_model.embed_tokens(torch.tensor([ids1]))
    dump(os.path.join(ref_dir, "embed"), emb[0])

    logits1, _ = run_and_dump(model, ids1, ref_dir)
    dump(os.path.join(ref_dir, "logits"), logits1)
    _, _ = run_and_dump(model, ids2, ref_dir, suffix="_seq2")

    meta = {
        "seed": a.seed,
        "seq1_len": len(ids1), "seq2_len": len(ids2),
        "seq1_ids": ids1,
        "seq2_ids": ids2,
        "n_layers": TINY["num_hidden_layers"],
        "hc_mult": TINY["hc_mult"],
        "hidden": TINY["hidden_size"],
        "vocab": TINY["vocab_size"],
        "layer_types": TINY["layer_types"],
        "mlp_layer_types": TINY["mlp_layer_types"],
        "mtp_block_index": TINY["num_hidden_layers"],
        "note": "all dumps little-endian f32; layerB{i} = [T, 4, H] mHC streams after block i",
    }
    with open(os.path.join(ref_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"logits: max|.| {logits1.abs().max():.3f} mean|.| {logits1.abs().mean():.4f} "
          f"finite {bool(torch.isfinite(logits1).all())}")

    if a.no_convert:
        return

    gguf = os.path.join(a.out, "glm5-tiny-f32.gguf")
    subprocess.run(
        [sys.executable, os.path.join(a.llama_cpp, "convert_hf_to_gguf.py"), hf_dir,
         "--outtype", "f32", "--outfile", gguf],
        check=True, stdout=subprocess.DEVNULL,
    )
    print("gguf:", gguf)


if __name__ == "__main__":
    main()
