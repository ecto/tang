#!/usr/bin/env python3
"""Token ids for the Qwen3-Coder-Next AA-style prompts (`tang-llm flash-aa --ids-files`).

    pip install transformers
    make_prompts.py [--out DIR]     # writes explain.ids, refactor.ids, write.ids

Each prompt is one user turn in Qwen3-Coder-Next's chat template (add_generation_prompt), with
the same 140-line Rust file (`context.rs`, the head of tang's `crates/tang-llm/src/flash/ngram.rs`)
and a different task. About 1.24k input tokens each.
"""
import argparse, os
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=".")
a = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__))
code = open(os.path.join(here, "context.rs")).read()
t = AutoTokenizer.from_pretrained("Qwen/Qwen3-Coder-Next")
tasks = {
    "explain": "Explain what this Rust module does, how its data structures fit together, and any performance pitfalls you see.\n\n```rust\n" + code + "\n```",
    "refactor": "Refactor this Rust module for readability: keep behaviour identical, add doc comments where missing, and return the complete updated file.\n\n```rust\n" + code + "\n```",
    "write": "Here is a Rust module for context.\n\n```rust\n" + code + "\n```\n\nNow write a new, self-contained Rust module (not a modification of this one) implementing an LRU cache with O(1) get and put, generic over key and value, with unit tests.",
}
os.makedirs(a.out, exist_ok=True)
for name, msg in tasks.items():
    s = t.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True, tokenize=False)
    ids = t(s, add_special_tokens=False)["input_ids"]
    open(os.path.join(a.out, f"{name}.ids"), "w").write(" ".join(map(str, ids)))
    print(name, len(ids))
