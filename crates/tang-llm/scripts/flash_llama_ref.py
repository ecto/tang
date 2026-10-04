#!/usr/bin/env python3
"""Reference next-token distributions from llama.cpp's server, for checking `tang-llm flash-ref`.

Stdlib only. Point it at a running llama-server (CPU-only is the cleanest reference:
`docker run ... ghcr.io/ggml-org/llama.cpp:server-cuda -m <gguf> -ngl 0 -c 4096 --parallel 1`
with no `--gpus`, so the CUDA backend can't load and every op runs on ggml-cpu).

    flash_llama_ref.py tokenize --url URL (--text S | --file F | --chat S) > ids.txt
    flash_llama_ref.py probs    --url URL --ids ids.txt [--from I] [--to J] [--n-probs 10] > llama.jsonl
    flash_llama_ref.py compare  llama.jsonl tang.jsonl

`probs` asks for every prefix `ids[:i+1]` in order with `n_predict: 1`, `cache_prompt: true`:
each request extends the cached prompt by one token, so llama.cpp decodes one token per position
(no rollback, which a recurrent model can't do) and `top_logprobs` is the softmax of the raw
logits over the whole vocabulary (`post_sampling_probs` false). Output lines match
`tang-llm flash-ref`: `{"pos": i, "top": [[id, logprob], ...]}`.

`compare` reports, over the positions both files have:
  - top-1 agreement;
  - KL(llama || tang) on the coarse distribution {llama's top-k tokens, everything else}: p_i from
    llama, q_i from tang's list (tang should print more, e.g. `--top 40`, so llama's top-k are
    covered), and one "rest" bucket 1 - sum. That is a lower bound on the full-vocabulary KL and
    is exact in the limit where both lists cover the mass.
"""
import argparse
import json
import math
import sys
import time
import urllib.request


def post(url, path, body):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)


def read_ids(path):
    s = open(path).read().replace(",", " ").replace("[", " ").replace("]", " ")
    return [int(w) for w in s.split()]


def cmd_tokenize(a):
    if a.chat is not None:
        text = post(a.url, "/apply-template", {"messages": [{"role": "user", "content": a.chat}]})["prompt"]
    elif a.file:
        text = open(a.file).read()
    else:
        text = a.text
    toks = post(a.url, "/tokenize", {"content": text, "add_special": False, "parse_special": True})["tokens"]
    print(" ".join(str(t) for t in toks))


def cmd_probs(a):
    ids = read_ids(a.ids)
    lo = a.from_ or 0
    hi = len(ids) if a.to is None else min(a.to, len(ids))
    t0 = time.time()
    for i in range(lo, hi):
        d = post(a.url, "/completion", {
            "prompt": ids[: i + 1],
            "n_predict": 1,
            "n_probs": a.n_probs,
            "cache_prompt": True,
            "temperature": 0,
        })
        cp = d["completion_probabilities"][0]
        tops = cp.get("top_logprobs")
        if tops is None:  # older servers: probabilities, not logprobs
            tops = [{"id": p["id"], "logprob": math.log(max(p["prob"], 1e-300))} for p in cp["probs"]]
        print(json.dumps({"pos": i, "top": [[t["id"], t["logprob"]] for t in tops],
                          "prompt_n": d.get("timings", {}).get("prompt_n")}), flush=True)
        if a.verbose:
            print(f"pos {i}: {time.time() - t0:.1f}s", file=sys.stderr)


def load_jsonl(path):
    out = {}
    for line in open(path):
        line = line.strip()
        if line.startswith("{"):
            d = json.loads(line)
            out[d["pos"]] = d["top"]
    return out


def cmd_compare(a):
    ref = load_jsonl(a.llama)
    got = load_jsonl(a.tang)
    pos = sorted(set(ref) & set(got))
    if not pos:
        sys.exit("no common positions")
    agree = 0
    kls = []
    missing = 0
    worst = []
    for p in pos:
        r, g = ref[p][: a.k], got[p]
        if r[0][0] == g[0][0]:
            agree += 1
        gq = {t: lp for t, lp in g}
        floor = min(lp for _, lp in g)
        pr = [math.exp(lp) for _, lp in r]
        qs = []
        for t, _ in r:
            if t in gq:
                qs.append(math.exp(gq[t]))
            else:
                missing += 1
                qs.append(math.exp(floor))
        prest = max(1.0 - sum(pr), 0.0)
        qrest = max(1.0 - sum(qs), 1e-12)
        kl = sum(pi * (math.log(pi) - math.log(max(qi, 1e-30))) for pi, qi in zip(pr, qs) if pi > 0)
        if prest > 0:
            kl += prest * (math.log(prest) - math.log(qrest))
        kls.append(kl)
        worst.append((kl, p, r[0][0], g[0][0]))
    worst.sort(reverse=True)
    n = len(pos)
    kls_sorted = sorted(kls)
    print(json.dumps({
        "positions": n,
        "top1_agree": agree / n,
        "top1_agree_count": agree,
        "kl_mean": sum(kls) / n,
        "kl_median": kls_sorted[n // 2],
        "kl_p99": kls_sorted[min(n - 1, int(n * 0.99))],
        "kl_max": kls_sorted[-1],
        "llama_topk_missing_from_tang": missing,
        "worst": [{"pos": p, "kl": round(k, 5), "llama_top1": rt, "tang_top1": gt} for k, p, rt, gt in worst[:5]],
    }, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tokenize")
    t.add_argument("--url", default="http://127.0.0.1:18080")
    t.add_argument("--text")
    t.add_argument("--file")
    t.add_argument("--chat", help="a user message, through the model's chat template")
    p = sub.add_parser("probs")
    p.add_argument("--url", default="http://127.0.0.1:18080")
    p.add_argument("--ids", required=True)
    p.add_argument("--from", dest="from_", type=int)
    p.add_argument("--to", type=int)
    p.add_argument("--n-probs", type=int, default=10)
    p.add_argument("-v", "--verbose", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("llama")
    c.add_argument("tang")
    c.add_argument("-k", type=int, default=10)
    a = ap.parse_args()
    {"tokenize": cmd_tokenize, "probs": cmd_probs, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
