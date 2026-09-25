"""Check tang-llm's Gemma 3 vision path against mlx-vlm (fp32): the projector's image features,
then logits for a prompt with the image, using the HF attention pattern (causal, plus
bidirectional within the image's tokens; mlx-vlm's own prefill mask doesn't do this).

usage: python check_vision_mlx.py <model-dir> <tang-llm binary> <image> [question]
"""
import json, os, subprocess, sys, tempfile
import numpy as np
import mlx.core as mx
from mlx.utils import tree_map
from mlx_vlm import load
from PIL import Image

model_dir, binary, image = sys.argv[1:4]
question = sys.argv[4] if len(sys.argv) > 4 else "What is written in this image?"
model, processor = load(model_dir)
model.update(tree_map(lambda p: p.astype(mx.float32) if mx.issubdtype(p.dtype, mx.floating) else p, model.parameters()))
cfg = model.config

# Same pixels for both: 896x896 bilinear, scaled to [-1, 1].
img = Image.open(image).convert("RGB").resize((896, 896), Image.BILINEAR)
px = (np.asarray(img, dtype=np.float32) / 255.0 - 0.5) / 0.5
tmp = tempfile.NamedTemporaryFile(suffix=".f32", delete=False)
px.astype("<f4").tofile(tmp.name)

# 1. Image features.
hidden, _, _ = model.vision_tower(mx.array(px[None]), output_hidden_states=True)
ref_feat = np.array(model.multi_modal_projector(hidden)[0])
out = subprocess.run([binary, "image-features", model_dir, tmp.name], capture_output=True, text=True, check=True)
got_feat = np.array(json.loads(out.stdout), dtype=np.float32)
rel = np.abs(got_feat - ref_feat).max() / np.abs(ref_feat).max()
print(f"image features {got_feat.shape} · max |Δ| {np.abs(got_feat - ref_feat).max():.4f} (relative {rel:.1e})")

# 2. Logits with the image in a chat prompt.
tok = processor.tokenizer
n_img = cfg.text_config.mm_tokens_per_image if hasattr(cfg.text_config, "mm_tokens_per_image") else 256
text = ("<bos><start_of_turn>user\n\n\n<start_of_image>" + "<image_soft_token>" * 256 + "<end_of_image>\n\n"
        + question + "<end_of_turn>\n<start_of_turn>model\n")
ids = tok.encode(text, add_special_tokens=False)
img_id = cfg.image_token_index
pos = [i for i, t in enumerate(ids) if t == img_id]
assert len(pos) == 256, len(pos)
lm = model.language_model
emb = lm.model.embed_tokens(mx.array([ids]))
emb = np.array(emb)
emb[0, pos] = ref_feat / (cfg.text_config.hidden_size ** 0.5)
L = len(ids)
allow = np.tril(np.ones((L, L), dtype=bool))
allow[np.ix_(pos, pos)] = True
mask = mx.array(np.where(allow, 0.0, -np.inf).astype(np.float32))[None, None]
ref = np.array(lm(mx.array([ids]), inputs_embeds=mx.array(emb), mask=mask).logits[0]) if hasattr(lm(mx.array([ids]), inputs_embeds=mx.array(emb), mask=mask), "logits") else np.array(lm(mx.array([ids]), inputs_embeds=mx.array(emb), mask=mask)[0])
last = 8
out = subprocess.run([binary, "logits-image", model_dir, tmp.name, *map(str, ids), "--last", str(last)], capture_output=True, text=True, check=True)
got = np.array(json.loads(out.stdout), dtype=np.float32)
ref = ref[-last:]

def log_softmax(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))

lr, lg = log_softmax(ref), log_softmax(got)
kl = (np.exp(lr) * (lr - lg)).sum(-1)
agree = (got.argmax(-1) == ref.argmax(-1)).mean()
print(f"{L} tokens · last {last}: max |Δlogit| {np.abs(got - ref).max():.4f} · top-1 agreement {agree:.0%} · max KL {kl.max():.2e}")
print("next token:", tok.decode([int(got[-1].argmax())]), "| ref:", tok.decode([int(ref[-1].argmax())]))
print("PASS" if kl.max() < 1e-3 and agree == 1.0 else "FAIL")
