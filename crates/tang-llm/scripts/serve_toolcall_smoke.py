#!/usr/bin/env python3
"""OpenAI tool-call smoke test against a running `tang-llm serve` (stdlib only).

    serve_toolcall_smoke.py --base-url http://127.0.0.1:8911/v1 [--model NAME] [--key-file F]

1. A request with one tool and a user turn that needs it: expects `finish_reason`
   `tool_calls`, one call to `get_weather` whose `arguments` parse as JSON with a `city`.
2. The same conversation with the tool's result appended: expects a plain answer that uses it.
3. Step 1 streamed: expects the call to arrive as `delta.tool_calls` chunks that reassemble to
   the same name and JSON arguments.

Exits non-zero on the first failure. Greedy (`temperature` 0) so runs are repeatable.
"""
import argparse, json, sys, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", required=True)
ap.add_argument("--model", default="tang")
ap.add_argument("--key-file")
a = ap.parse_args()
url = a.base_url.rstrip("/") + "/chat/completions"
headers = {"Content-Type": "application/json"}
if a.key_file:
    headers["Authorization"] = "Bearer " + open(a.key_file).read().strip()

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}]
messages = [
    {"role": "system", "content": "You are a helpful assistant. Use tools when they help."},
    {"role": "user", "content": "What's the weather in Paris right now? Use the tool."},
]
base = {"model": a.model, "tools": tools, "temperature": 0, "max_tokens": 512}


def post(body, stream=False):
    req = urllib.request.Request(url, json.dumps(body).encode(), headers)
    return urllib.request.urlopen(req, timeout=300)


def fail(msg, obj=None):
    print("FAIL", msg, json.dumps(obj, indent=2)[:2000] if obj is not None else "")
    sys.exit(1)


# 1. a tool call
with post({**base, "messages": messages}) as r:
    res = json.load(r)
ch = res["choices"][0]
if ch["finish_reason"] != "tool_calls":
    fail("finish_reason is not tool_calls", res)
calls = ch["message"].get("tool_calls") or []
if len(calls) != 1 or calls[0]["function"]["name"] != "get_weather":
    fail("expected one get_weather call", res)
args = json.loads(calls[0]["function"]["arguments"])
if "paris" not in str(args.get("city", "")).lower():
    fail("city argument", res)
print(f"PASS tool call: get_weather({args}) "
      f"[{res['usage'].get('completion_tokens')} tokens, drafts {res['usage'].get('accepted_tokens', 0)}"
      f"/{res['usage'].get('draft_tokens', 0)} accepted]", flush=True)

# 2. the tool's result goes back in; expect an answer that uses it
follow = messages + [
    {"role": "assistant", "content": ch["message"].get("content") or "", "tool_calls": calls},
    {"role": "tool", "tool_call_id": calls[0].get("id", "call_0"),
     "content": json.dumps({"city": "Paris", "temp_c": 17, "sky": "light rain"})},
]
with post({**base, "messages": follow}) as r:
    res2 = json.load(r)
ch2 = res2["choices"][0]
text = ch2["message"].get("content") or ""
if ch2["finish_reason"] != "stop" or "17" not in text:
    fail("follow-up should answer with the tool result (17 °C)", res2)
print(f"PASS tool result used: {text.strip()[:120]!r}", flush=True)

# 3. streamed: the call arrives as delta.tool_calls
name, argbuf, finish = "", "", None
with post({**base, "messages": messages, "stream": True}) as r:
    for line in r:
        if not line.startswith(b"data: "):
            continue
        data = line[6:].strip()
        if data == b"[DONE]":
            break
        ev = json.loads(data)
        for c in ev.get("choices", []):
            for tc in (c.get("delta") or {}).get("tool_calls") or []:
                f = tc.get("function") or {}
                name += f.get("name") or ""
                argbuf += f.get("arguments") or ""
            finish = c.get("finish_reason") or finish
if name != "get_weather" or finish != "tool_calls" or json.loads(argbuf) != args:
    fail("streamed call differs", {"name": name, "args": argbuf, "finish": finish})
print("PASS streamed tool call matches", flush=True)
print("tool-call smoke: PASS")
