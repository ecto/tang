#!/usr/bin/env python3
"""Real Qwen3-VL-8B-Instruct: tang against llama.cpp's multimodal path (mtmd) on one image prompt.

    qwen3vl_llamacpp.py --tang tang-llm --model <HF dir> --server http://127.0.0.1:8080 \
        --image view.png [--device cuda] [-n 32]

The server is `llama-server -m qwen3vl-bf16.gguf --mmproj mmproj-bf16.gguf` (from
`convert_hf_to_gguf.py` with and without `--mmproj`). llama.cpp greedily generates `n` tokens
with its top-20 log-probabilities at each step. tang runs the same prompt plus those tokens
(its own preprocessing, as the server would) and is compared at each generated position:
top-1 agreement with llama.cpp's choice, and KL over llama.cpp's top-20 (both renormalized
to those 20 tokens; the tail beyond them isn't reported by the server).
"""
import argparse
import base64
import re
import json
import subprocess
import sys
import urllib.request

import numpy as np
from transformers import AutoProcessor
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tang", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--image", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("-n", type=int, default=32)
    ap.add_argument("--question", default="This is a screenshot of a CAD viewport. Describe the parts you can see, their shapes and colors, and where each one is.")
    a = ap.parse_args()

    proc = AutoProcessor.from_pretrained(a.model)
    tok = proc.tokenizer
    img = Image.open(a.image).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": a.question}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = proc(text=[text], images=[img], return_tensors="np")["input_ids"][0].tolist()

    # llama.cpp: the same chat, its own template and preprocessing.
    data = base64.b64encode(open(a.image, "rb").read()).decode()
    # The image as llama.cpp's media marker (mtmd adds the vision start/end tokens itself).
    prompt, n = re.subn(r"<\|vision_start\|>(<\|image_pad\|>)*<\|vision_end\|>", "<__media__>", text)
    assert n == 1, text
    body = {"prompt": {"prompt_string": prompt, "multimodal_data": [data]}, "n_predict": a.n,
            "temperature": 0, "n_probs": 20, "cache_prompt": False, "return_tokens": True}
    req = urllib.request.Request(a.server + "/completion", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=600))
    probs = out["completion_probabilities"]
    gen = [p["id"] for p in probs]
    print("llama.cpp greedy:", repr(tok.decode(gen)), flush=True)

    r = subprocess.run([a.tang, "vl-logits", a.model, "--image", a.image, *map(str, ids + gen),
                        "--last", str(len(gen) + 1), "--device", a.device], capture_output=True, text=True)
    if r.returncode:
        sys.exit(r.stderr[-3000:])
    rows = np.array(json.loads(r.stdout), dtype=np.float64)[:-1]  # row i predicts gen[i]
    tang_greedy = rows.argmax(-1).tolist()
    print("tang greedy (teacher-forced):", repr(tok.decode(tang_greedy)), flush=True)
    kls, agree = [], 0
    for i, p in enumerate(probs):
        top = p["top_probs"] if "top_probs" in p else p["top_logprobs"]
        tid = [t["id"] for t in top]
        lp = np.array([t["logprob"] for t in top], dtype=np.float64)
        lp -= np.logaddexp.reduce(lp)
        lq = rows[i, tid] - np.logaddexp.reduce(rows[i, tid])
        kls.append(float((np.exp(lp) * (lp - lq)).sum()))
        agree += int(rows[i].argmax() == gen[i])
    kls = np.array(kls)
    print(f"tang vs llama.cpp mtmd: {len(gen)} generated positions, top-1 {agree / len(gen) * 100:.1f}%, "
          f"top-20 KL mean {kls.mean():.1e} p99 {np.percentile(kls, 99):.1e} max {kls.max():.1e}", flush=True)


if __name__ == "__main__":
    main()
