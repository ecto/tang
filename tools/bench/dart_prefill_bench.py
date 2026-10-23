#!/usr/bin/env python3
"""dart fair-share prefill bench: correctness (concurrent == solo, no starvation) then TTFT.

Run on the pod against a local `tang-llm serve` (no auth, 127.0.0.1:8912).
  prefill_bench.py prompts <repo> <outdir>     build distinct cold prompts from repo text
  prefill_bench.py solo <outdir> <tag>         4×58k one at a time (cold) → outputs + TTFT
  prefill_bench.py conc <outdir> <tag>         4×58k at once + a short prompt at +5 s + a running decode stream
  prefill_bench.py long <outdir> <tag>         one ~117k prompt alone
Stdlib only.
"""
import json, os, sys, time, threading, urllib.request, glob, random

URL = "http://127.0.0.1:8912/v1/chat/completions"
CHARS_PER_TOK = 3.4  # rough for code; real counts come from usage.prompt_tokens


def build_prompts(repo, out):
    files = sorted(glob.glob(f"{repo}/crates/**/*.rs", recursive=True))
    text = []
    for f in files:
        try:
            text.append(f"// ==== {os.path.relpath(f, repo)}\n" + open(f, errors="ignore").read())
        except OSError:
            pass
    os.makedirs(out, exist_ok=True)
    def make(i, toks):
        rnd = random.Random(1000 + i)
        order = text[:]
        rnd.shuffle(order)
        body, n = [], 0
        for t in order:
            body.append(t)
            n += len(t)
            if n >= toks * CHARS_PER_TOK:
                break
        # unique nonce FIRST so no two prompts share a cached prefix
        head = f"[session {i}-{rnd.getrandbits(64):016x}] Read this Rust source and answer at the end.\n\n"
        q = "\n\nIn two sentences, what does the first file above do?"
        return head + "".join(body)[: int(toks * CHARS_PER_TOK)] + q
    p = {f"p{i}": make(i, 58000) for i in range(4)}
    p["long"] = make(9, 117000)
    p["short"] = make(7, 1000)
    p["decoder"] = "Write a long, detailed essay about the history of compilers. Keep going."
    json.dump(p, open(f"{out}/prompts.json", "w"))
    print({k: len(v) for k, v in p.items()})


def req(prompt, max_tokens):
    """Stream one request. Returns dict with ttft, token arrival times, text, usage."""
    body = json.dumps({
        "model": "dart", "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0, "max_tokens": max_tokens, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    r = urllib.request.Request(URL, body, {"content-type": "application/json"})
    t0 = time.time(); times, text, usage, err = [], [], None, None
    try:
        with urllib.request.urlopen(r, timeout=900) as resp:
            for line in resp:
                line = line.decode().strip()
                if not line.startswith("data:"):
                    continue
                d = line[5:].strip()
                if d == "[DONE]":
                    break
                j = json.loads(d)
                if j.get("usage"):
                    usage = j["usage"]
                for c in j.get("choices", []):
                    s = (c.get("delta") or {}).get("content") or (c.get("delta") or {}).get("reasoning_content")
                    if s:
                        times.append(time.time() - t0); text.append(s)
    except Exception as e:  # record, never crash the bench
        err = repr(e)
    return {"ttft": times[0] if times else None, "times": times, "text": "".join(text),
            "usage": usage, "err": err, "wall": time.time() - t0}


def run_parallel(jobs):
    """jobs: list of (name, prompt, max_tokens, delay_s). Returns {name: result}."""
    res, th = {}, []
    def go(n, p, m, d):
        time.sleep(d); r = req(p, m); r["start_delay"] = d; res[n] = r
    for j in jobs:
        t = threading.Thread(target=go, args=j); t.start(); th.append(t)
    for t in th:
        t.join()
    return res


def main():
    mode, out = sys.argv[1], sys.argv[-2] if sys.argv[1] != "prompts" else sys.argv[3]
    if mode == "prompts":
        return build_prompts(sys.argv[2], sys.argv[3])
    tag = sys.argv[-1]
    P = json.load(open(f"{out}/prompts.json"))
    if mode == "solo":
        res = {f"p{i}": req(P[f"p{i}"], 64) for i in range(4)}
    elif mode == "conc":
        # decoder starts first and keeps generating through the burst (starvation probe)
        jobs = [("decoder", P["decoder"], 1500, 0.0)]
        jobs += [(f"p{i}", P[f"p{i}"], 64, 3.0) for i in range(4)]
        jobs += [("short", P["short"], 64, 8.0)]
        res = run_parallel(jobs)
        d = res["decoder"]["times"]
        res["decoder_max_gap_s"] = max((b - a for a, b in zip(d, d[1:])), default=None)
    elif mode == "long":
        res = {"long": req(P["long"], 16)}
    json.dump(res, open(f"{out}/{mode}-{tag}.json", "w"))
    summ = {k: (round(v["ttft"], 2) if isinstance(v, dict) and v.get("ttft") else v if not isinstance(v, dict) else v.get("err"))
            for k, v in res.items()}
    print(mode, tag, json.dumps(summ))


if __name__ == "__main__":
    main()
