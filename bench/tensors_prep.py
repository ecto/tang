#!/usr/bin/env python3
"""Tokenize the ML-tensor bench text (run once; needs `tokenizers`).

    uv run --no-project --with tokenizers --with numpy --with pyarrow python bench/tensors_prep.py

Eval text: WikiText-2 raw test split (Salesforce/wikitext,
wikitext-2-raw-v1/test-00000-of-00001.parquet), first 600 rows. Prompts for the
KV/activation dumps are fixed below. Writes bench/tensors/tokens.npz.
"""
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

T = Path(__file__).resolve().parent / "tensors"
PROMPTS = [
    "The theory of general relativity predicts that massive objects bend spacetime, and",
    "def quicksort(xs):\n    if len(xs) <= 1:\n        return xs\n",
    "Once upon a time, in a small village by the sea, there lived an old fisherman who",
    "The mitochondria is the powerhouse of the cell. In eukaryotic organisms, it",
]


def wikitext(n_rows=600):
    import pyarrow.parquet as pq
    t = pq.read_table(T / "hf/wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet")
    return "".join(t.column("text").to_pylist()[:n_rows])


tok = Tokenizer.from_file(str(T / "hf/SmolLM2-135M/tokenizer.json"))
text = wikitext()
ids = np.array(tok.encode(text).ids, dtype=np.int32)
prompts = [np.array(tok.encode(p).ids, dtype=np.int32) for p in PROMPTS]
np.savez(T / "tokens.npz", wiki=ids, **{f"prompt{i}": p for i, p in enumerate(prompts)})
print("wikitext tokens", len(ids), "prompt lens", [len(p) for p in prompts])
