#!/usr/bin/env python3
"""Head-to-head speed and quality for any OpenAI-compatible chat endpoint (stdlib only).

    h2h.py --base-url https://api.example.com/v1 --model MODEL [--key-env API_KEY] [--n 3]
           [--label NAME] [--out results.json] [--skip-speed] [--skip-quality]
    h2h.py --selftest                      # check the quality tests against reference solutions

Speed. Each prompt runs n times, streaming, temperature 0, one request at a time:
  - `explain`, `refactor`, `write`: the AA-style prompts (`../qwen3next-aa/`, ~1.2k tokens in,
    512 out);
  - `agent`: an agent-length prompt (~10k tokens in: a system prompt, five tool schemas, a
    34 KB Rust file, a review task), 512 out.
For each request it records the time to first token (the first content, reasoning or tool-call
delta) and the output speed after the first token: (completion tokens − 1) / (last delta −
first delta), with completion tokens from the final `usage` chunk (`stream_options.include_usage`)
or, when the endpoint sends none, the count of content deltas (marked "approx").

Quality. Twenty small Python tasks (`tasks.py`): the model writes the function, the harness
runs the task's asserts in a subprocess (isolated mode, a temp dir, a 10 s timeout). pass@1,
temperature 0. Model output is run on this machine, so run the harness somewhere you're happy
to execute it (a container or a throwaway VM for unknown endpoints).

Same-day comparison: run it against each endpoint on the same day, from the same client host,
and compare the JSON outputs.
"""
import argparse, json, os, re, statistics, subprocess, sys, tempfile, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tasks import TASKS  # noqa: E402


def aa_prompts():
    code = open(os.path.join(HERE, "..", "qwen3next-aa", "context.rs")).read()
    return {
        "explain": "Explain what this Rust module does, how its data structures fit together, and any performance pitfalls you see.\n\n```rust\n" + code + "\n```",
        "refactor": "Refactor this Rust module for readability: keep behaviour identical, add doc comments where missing, and return the complete updated file.\n\n```rust\n" + code + "\n```",
        "write": "Here is a Rust module for context.\n\n```rust\n" + code + "\n```\n\nNow write a new, self-contained Rust module (not a modification of this one) implementing an LRU cache with O(1) get and put, generic over key and value, with unit tests.",
    }


AGENT_TOOLS = [
    {"type": "function", "function": {"name": n, "description": d, "parameters": {
        "type": "object", "properties": p, "required": list(p)}}}
    for n, d, p in [
        ("read_file", "Read a file from the repository.", {"path": {"type": "string", "description": "Repository-relative path"}}),
        ("write_file", "Replace a file's contents.", {"path": {"type": "string"}, "content": {"type": "string"}}),
        ("grep", "Search the repository with a regular expression.", {"pattern": {"type": "string"}, "glob": {"type": "string"}}),
        ("run", "Run a shell command in the repository and return its output.", {"command": {"type": "string"}}),
        ("finish", "End the task with a summary for the user.", {"summary": {"type": "string"}}),
    ]
]


def agent_messages():
    code = open(os.path.join(HERE, "agent_context.rs")).read()
    return [
        {"role": "system", "content": "You are a careful senior engineer working in a Rust repository. "
         "Use the tools when you need them. Be concise and concrete."},
        {"role": "user", "content": "Here is crates/tang-llm/src/draft.rs:\n\n```rust\n" + code + "\n```\n\n"
         "Review this module for correctness bugs and performance problems. For each finding give the "
         "function, what goes wrong, and a fix. Don't call any tools; answer directly."},
    ]


class Endpoint:
    def __init__(self, base, model, key):
        self.url = base.rstrip("/") + "/chat/completions"
        self.model, self.key = model, key

    def _req(self, body):
        h = {"Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = "Bearer " + self.key
        return urllib.request.Request(self.url, json.dumps(body).encode(), h)

    def stream(self, messages, max_tokens, tools=None):
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
        if tools:
            body["tools"] = tools
        t0 = time.perf_counter()
        first = last = None
        deltas, usage, text = 0, None, []
        with urllib.request.urlopen(self._req(body), timeout=600) as r:
            for line in r:
                if not line.startswith(b"data: "):
                    continue
                data = line[6:].strip()
                if data == b"[DONE]":
                    break
                ev = json.loads(data)
                if ev.get("usage"):
                    usage = ev["usage"]
                for c in ev.get("choices") or []:
                    d = c.get("delta") or {}
                    piece = (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
                    if piece or d.get("tool_calls"):
                        now = time.perf_counter()
                        first = first or now
                        last = now
                        deltas += 1
                        text.append(d.get("content") or "")
        out = {"ttft_ms": (first - t0) * 1e3 if first else None, "deltas": deltas}
        n = (usage or {}).get("completion_tokens")
        out["completion_tokens"] = n if n is not None else deltas
        out["approx"] = n is None
        out["tok_s"] = ((out["completion_tokens"] - 1) / (last - first)
                        if first and last and last > first and out["completion_tokens"] > 1 else None)
        out["text"] = "".join(text)
        return out

    def complete(self, messages, max_tokens):
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
        with urllib.request.urlopen(self._req(body), timeout=600) as r:
            res = json.load(r)
        return res["choices"][0]["message"].get("content") or ""


def summarize(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"min": min(xs), "median": statistics.median(xs), "max": max(xs), "n": len(xs)}


def run_speed(ep, n):
    prompts = {k: [{"role": "user", "content": v}] for k, v in aa_prompts().items()}
    prompts["agent"] = agent_messages()
    out = {}
    for name, msgs in prompts.items():
        runs = []
        for i in range(n):
            r = ep.stream(msgs, 512, tools=AGENT_TOOLS if name == "agent" else None)
            r.pop("text")
            runs.append(r)
            print(f"  {name:9} run {i + 1}: {r['tok_s'] or 0:7.1f} tok/s after first token, "
                  f"TTFT {r['ttft_ms'] or 0:7.0f} ms, {r['completion_tokens']} tokens"
                  f"{' (approx)' if r['approx'] else ''}", flush=True)
        out[name] = {"runs": runs, "tok_s": summarize([r["tok_s"] for r in runs]),
                     "ttft_ms": summarize([r["ttft_ms"] for r in runs])}
    return out


def extract_code(text, fname):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    for b in blocks:
        if f"def {fname}" in b:
            return b
    return blocks[0] if blocks else text


def check(code, tests):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.py")
        open(p, "w").write(code + "\n\n" + tests + "\n")
        try:
            r = subprocess.run([sys.executable, "-I", p], cwd=d, capture_output=True, timeout=10)
            return r.returncode == 0, r.stderr.decode()[-400:]
        except subprocess.TimeoutExpired:
            return False, "timeout"


def run_quality(ep):
    results, passed = [], 0
    for fname, prompt, tests in TASKS:
        msg = ("Complete this Python function. Reply with the full function in one ```python code "
               "block, standard library only, no tests or examples.\n\n```python\n" + prompt + "\n```")
        text = ep.complete([{"role": "user", "content": msg}], 1024)
        ok, err = check(extract_code(text, fname), tests)
        passed += ok
        results.append({"task": fname, "pass": ok, "error": None if ok else err})
        print(f"  {fname:20} {'PASS' if ok else 'FAIL'}", flush=True)
    return {"pass_at_1": passed / len(TASKS), "passed": passed, "total": len(TASKS), "tasks": results}


def selftest():
    from tasks_ref import REF
    bad = []
    for fname, _, tests in TASKS:
        ok, err = check(REF[fname], tests)
        if not ok:
            bad.append((fname, err))
    for f, e in bad:
        print("REF FAILS", f, e)
    print(f"selftest: {len(TASKS) - len(bad)}/{len(TASKS)} reference solutions pass their tests")
    return not bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url")
    ap.add_argument("--model", default="default")
    ap.add_argument("--key-env", default="API_KEY", help="environment variable holding the API key")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out")
    ap.add_argument("--skip-speed", action="store_true")
    ap.add_argument("--skip-quality", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not a.base_url:
        ap.error("--base-url is required")
    ep = Endpoint(a.base_url, a.model, os.environ.get(a.key_env))
    res = {"label": a.label or a.base_url, "model": a.model, "date": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
           "client_host": os.uname().nodename}
    if not a.skip_speed:
        print(f"speed: {a.n} runs a prompt, streaming, temperature 0", flush=True)
        res["speed"] = run_speed(ep, a.n)
    if not a.skip_quality:
        print("quality: 20 tasks, pass@1, temperature 0", flush=True)
        res["quality"] = run_quality(ep)
    print("\nsummary:", res["label"])
    for name, s in (res.get("speed") or {}).items():
        t, f = s["tok_s"], s["ttft_ms"]
        if t and f:
            print(f"  {name:9} tok/s median {t['median']:7.1f} (min {t['min']:.1f}, max {t['max']:.1f}, n {t['n']})"
                  f"  TTFT median {f['median']:6.0f} ms")
    if "quality" in res:
        q = res["quality"]
        print(f"  pass@1 {q['passed']}/{q['total']} = {q['pass_at_1']:.2f}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
