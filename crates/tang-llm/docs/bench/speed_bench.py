# AA-style single-stream output speed: ~1k-token prompt, stream, tok/s from first token to last.
# usage: python3 -I speed_bench.py <runs> [port]
import json, sys, time, urllib.request, random
runs = int(sys.argv[1]); port = sys.argv[2] if len(sys.argv) > 2 else "8912"
random.seed(1)
words = ("river stone market ledger harbor winter signal copper orchard lantern bridge engine "
         "letter garden mirror thunder canvas meadow quarry silver timber valley").split()
para = lambda: " ".join(random.choice(words) for _ in range(60)) + "."
res = []
for i in range(runs):
    ctx = "\n\n".join(para() for _ in range(12))  # ~800-1000 tokens
    body = {"max_tokens": 500, "temperature": 0.7, "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": f"Notes {i}:\n{ctx}\n\nWrite a detailed essay that weaves these notes into a story. /no_think"}]}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    t0 = time.time(); first = last = None; chunks = 0; usage = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]": continue
            d = json.loads(line[5:])
            if d.get("usage"): usage = d["usage"]
            for c in d.get("choices", []):
                if (c.get("delta") or {}).get("content"):
                    now = time.time(); first = first or now; last = now; chunks += 1
    n = (usage or {}).get("completion_tokens", chunks)
    p = (usage or {}).get("prompt_tokens", 0)
    res.append((n - 1) / (last - first)); print(f"run {i}: prompt {p} out {n} ttft {first - t0:.3f}s {res[-1]:.1f} tok/s", flush=True)
res.sort(); print(f"AA median {res[len(res)//2]:.1f} tok/s")
