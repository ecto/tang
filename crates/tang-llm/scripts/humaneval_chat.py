#!/usr/bin/env python3
"""HumanEval pass@1 (greedy, thinking off) against an OpenAI-compatible chat endpoint.

Stdlib only, plus the HumanEval parquet from `hf download openai/openai_humaneval`, read with
pyarrow or pandas if present (otherwise pass `--jsonl`). Used to compare the requantized
Flash-Next Coder against Flash-Next base (both on `tang-llm serve`) and the original IQ1_M Coder
(on llama-server), so the three see the same prompts and the same grader.

    humaneval_chat.py --url http://127.0.0.1:8911 --data test.parquet --n 50 --out run.jsonl

Each problem's completion runs with its tests in a fresh `python3 -I` subprocess with a timeout.
That is not a sandbox: run it only on a throwaway machine.
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.request


def load(path, jsonl):
    if jsonl:
        return [json.loads(l) for l in open(path)]
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def chat(url, prompt, max_tokens, key):
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "think": False,
    }
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), h)
    t = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        j = json.load(r)
    return j["choices"][0]["message"].get("content") or "", j.get("usage", {}), time.time() - t


def extract(text, p):
    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", text, re.S)
    code = max(blocks, key=len) if blocks else text
    # A bare body (no def) gets the prompt's signature in front of it.
    if f"def {p['entry_point']}" not in code:
        code = p["prompt"] + code
    return code


def run(code, p, timeout):
    prog = code + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(prog)
    try:
        r = subprocess.run([sys.executable, "-I", f.name], capture_output=True, timeout=timeout)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--jsonl", action="store_true")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--key", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    probs = load(a.data, a.jsonl)[: a.n]
    passed, toks, secs = 0, 0, 0.0
    with open(a.out, "w") as out:
        for i, p in enumerate(probs):
            prompt = ("Complete this Python function. Reply with the whole function in one "
                      "```python block, no explanation.\n\n```python\n" + p["prompt"] + "```")
            text, usage, dt = chat(a.url, prompt, a.max_tokens, a.key)
            ok = run(extract(text, p), p, 20)
            passed += ok
            toks += usage.get("completion_tokens", 0)
            secs += dt
            out.write(json.dumps({"task": p["task_id"], "pass": ok, "text": text,
                                  "tokens": usage.get("completion_tokens"), "s": dt}) + "\n")
            print(f"{i + 1}/{len(probs)} {p['task_id']} {'PASS' if ok else 'fail'} "
                  f"({passed}/{i + 1})", file=sys.stderr, flush=True)
    print(json.dumps({"pass@1": passed / len(probs), "passed": passed, "n": len(probs),
                      "completion_tokens": toks, "seconds": round(secs, 1)}))


if __name__ == "__main__":
    main()
