#!/usr/bin/env python3
"""Downstream effect: SmolLM2-135M WikiText-2 perplexity with (a) every
linear weight in the 30 blocks compressed, (b) the KV cache compressed
(simulated: K/V of every layer replaced by their reconstruction during
prefill, as in KIVI-style evals). Embeddings/norms stay bf16.

Eval: tokens [0, 8192) for weights, [0, 4096) for KV, 1024-token windows.
Writes bench/out/tensors_ppl.md.
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tensors_codecs as C  # noqa: E402
from tensors_dump import Llama, ppl, smol_weights  # noqa: E402
from tensors_io import T  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"

WEIGHT = [("int-perrow", 8), ("int-perrow", 4), ("int-g128", 4), ("int-g128", 3),
          ("sz3-abs", 0.3), ("sz3-abs", 0.1), ("zfp-acc", 0.3), ("zfp-acc", 0.1),
          ("tang-wavelet", 0.1), ("tang-wavelet", 0.03)]
KV = [("int-perrow", 8), ("int-perrow", 4), ("int-perrow", 2), ("kivi-g32", 4), ("kivi-g32", 2),
      ("sz3-abs", 0.3), ("sz3-abs", 1.0), ("zfp-acc", 0.3), ("zfp-acc", 1.0),
      ("tang-wavelet", 0.1), ("tang-wavelet", 0.3)]


def apply(job):
    """(codec, p, name, x, kind) -> (bpv, sq err, x_hat)."""
    codec, p, name, x, kind = job
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    if codec == "int-perrow":
        bpv, _, y = C.quant_rows(x, kind, p)
    elif codec == "int-g128":
        bpv, _, y = C.quant_rows(x, kind, p, group=128)
    elif codec == "kivi-g32":
        bpv, _, y = C.kivi_kv(x, name.endswith("K"), p)
    elif codec == "sz3-abs":
        bpv, y = C.sz3(x, p * rms)
    elif codec == "zfp-acc":
        bpv, y = C.zfp(x, p * rms)
    else:
        bpv, y = C.tang(x, "wavelet", p * rms, name.replace("/", "_"))
    return bpv, float(((x - y) ** 2).sum()), y.astype(np.float32)


def run_weights(ex, base, dev, wiki, lines):
    names = [k for k in base if ".layers." in k and k.endswith("proj.weight")]
    for codec, p in WEIGHT:
        jobs = [(codec, p, n, base[n], "weight") for n in names]
        w = dict(base)
        bits = err = nrm = 0.0
        for n, (bpv, se, y) in zip(names, ex.map(apply, jobs)):
            w[n] = y
            bits += bpv * y.size
            err += se
            nrm += float((base[n].astype(np.float64) ** 2).sum())
        n_tot = sum(base[n].size for n in names)
        m = Llama(w, dev)
        lines.append(f"| weights | {codec} | {p} | {bits / n_tot:.2f} | {np.sqrt(err / nrm):.4f} | {ppl(m, wiki[:8192]):.3f} |")
        print(lines[-1], flush=True)
        del m


def run_kv(ex, m, wiki, lines):
    for codec, p in KV:
        nll = bits = n_vals = err = nrm = 0.0
        nt = 0
        for s in range(0, 4096, 1024):
            x = torch.from_numpy(wiki[s : s + 1025].astype(np.int64)).to(m.dev)[None]
            cap = {}
            with torch.no_grad():
                m(x[:, :-1], kv_hook=lambda l, k, v: cap.update({(l, "K"): k[0].float().cpu().numpy(), (l, "V"): v[0].float().cpu().numpy()}))
            keys = list(cap)
            jobs = [(codec, p, f"kv{s}_L{l}_{kv}", cap[(l, kv)], "kv") for l, kv in keys]
            rec = {}
            for key, (bpv, se, y) in zip(keys, ex.map(apply, jobs)):
                rec[key] = torch.from_numpy(y).to(m.dev)[None]
                bits += bpv * y.size
                n_vals += y.size
                err += se
                nrm += float((cap[key].astype(np.float64) ** 2).sum())
            with torch.no_grad():
                lo = m(x[:, :-1], kv_override=lambda l, k, v: (rec[(l, "K")], rec[(l, "V")]))
                nll += torch.nn.functional.cross_entropy(lo[0].float(), x[0, 1:], reduction="sum").item()
            nt += 1024
        lines.append(f"| kv | {codec} | {p} | {bits / n_vals:.2f} | {np.sqrt(err / nrm):.4f} | {np.exp(nll / nt):.3f} |")
        print(lines[-1], flush=True)


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    wiki = np.load(T / "tokens.npz")["wiki"]
    base = smol_weights()
    C.SCRATCH.mkdir(parents=True, exist_ok=True)
    m = Llama(base, dev)
    lines = ["| target | codec | param | bpv | relL2 | ppl |", "|---|---|---|---|---|---|",
             f"| weights | bf16 (ref) | | 16 | 0 | {ppl(m, wiki[:8192]):.3f} |",
             f"| kv | bf16-weights fp32-KV (ref) | | 32 | 0 | {ppl(m, wiki[:4096]):.3f} |"]
    print("\n".join(lines), flush=True)
    with ProcessPoolExecutor(4) as ex:
        if "--weights-only" not in sys.argv:
            run_kv(ex, m, wiki, lines)
        del m
        if "--kv-only" not in sys.argv:
            run_weights(ex, base, dev, wiki, lines)
    OUT.mkdir(exist_ok=True)
    (OUT / "tensors_ppl.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
