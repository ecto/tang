#!/usr/bin/env python3
"""Request sets (chat completion bodies, one JSON per line) for `tang-llm bench-spec`.

    spec_workloads.py edit  <out.jsonl> [--root DIR]      # rewrite code from a file in context
    spec_workloads.py chat  <out.jsonl>                   # plain chat, few copies to draft
    spec_workloads.py kiln  <out.jsonl> SESSIONS_DIR... [--max N | --all-calls LAST_N_SESSIONS]

`kiln` rebuilds each model call's conversation from kiln session logs (user text, assistant
thinking/text/tool calls, tool results). The logs lack kiln's system prompt and tool schemas, so
a short system prompt and schemas inferred from the calls stand in.
"""

import glob
import json
import os
import sys

EDIT_FILES = [
    "crates/tang-llm/src/sample.rs",
    "crates/tang-llm/src/config.rs",
    "crates/tang-llm/src/weights.rs",
    "crates/tang-llm/src/lib.rs",
]


def body(messages, max_tokens, think, budget=None, tools=None):
    b = {"messages": messages, "max_tokens": max_tokens, "temperature": 0,
         "chat_template_kwargs": {"enable_thinking": think}}
    if budget is not None:
        b["thinking_budget"] = budget
    if tools:
        b["tools"] = tools
    return b


def edit(root):
    out = []
    tasks = [
        ("Add a one-line `///` doc comment to every function that lacks one. Output the whole "
         "updated file in one code block, nothing else.", 900),
        ("Rename every local variable called `v` or `x` to a more descriptive name. Output the "
         "whole updated file in one code block, nothing else.", 900),
        ("Rewrite just the longest function in this file so it has early returns instead of "
         "nested ifs where possible. Output only that function in a code block.", 500),
    ]
    for f in EDIT_FILES:
        src = open(os.path.join(root, f)).read()
        if len(src) > 9000:
            src = src[:9000]
        for task, mx in tasks:
            msg = f"Here is `{f}`:\n\n```rust\n{src}\n```\n\n{task}"
            out.append(body([{"role": "user", "content": msg}], mx, False))
    return out


def chat():
    prompts = [
        "Write a short poem about a lighthouse keeper.",
        "Explain the difference between TCP and UDP to a new programmer.",
        "Give me three ideas for a weekend project with a Raspberry Pi, with one paragraph each.",
        "What are the main causes of inflation? Keep it under 200 words.",
        "Summarize the plot of Moby-Dick in one paragraph.",
        "How do I make a good cup of pour-over coffee?",
        "Write a limerick about a cat who learns to code.",
        "What's a good way to learn a new language as an adult?",
    ]
    return [body([{"role": "user", "content": p}], 300, False) for p in prompts]


SYSTEM = ("You are kiln, a coding agent working in the user's project directory. Use the tools "
          "to read, search and edit files and run commands, then answer briefly.")


def kiln(dirs, per_session=4, max_tokens=400, budget=256, last_sessions=0):
    out, schemas = [], {}
    files = sorted(f for d in dirs for f in glob.glob(os.path.join(d, "*.jsonl")))
    sessions = []
    for f in files:
        msgs, calls = [{"role": "system", "content": SYSTEM}], []
        for line in open(f):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            t = r.get("type")
            if t == "push":
                m = r["message"]
                blocks = m.get("content") or []
                if m.get("role") == "user":
                    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
                    if text:
                        msgs.append({"role": "user", "content": text})
                elif m.get("role") == "assistant":
                    calls.append(list(msgs))
                    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
                    tcs = []
                    for b in blocks:
                        if b.get("type") == "tool_use":
                            inp = b.get("input") or {}
                            tcs.append({"id": b.get("id"), "type": "function",
                                        "function": {"name": b["name"], "arguments": json.dumps(inp)}})
                            props = schemas.setdefault(b["name"], {})
                            for k, v in inp.items():
                                props.setdefault(k, {"type": {int: "integer", bool: "boolean"}.get(type(v), "string")})
                    a = {"role": "assistant", "content": text}
                    if tcs:
                        a["tool_calls"] = tcs
                    msgs.append(a)
            elif t == "extend":
                for b in r.get("blocks") or []:
                    if b.get("type") == "tool_result":
                        c = b.get("content")
                        if not isinstance(c, str):
                            c = json.dumps(c)
                        msgs.append({"role": "tool", "tool_call_id": b.get("id"), "content": c[:6000]})
                    elif b.get("type") == "text":
                        msgs.append({"role": "user", "content": b["text"]})
        if calls:
            sessions.append(calls)
    if last_sessions:
        sessions = sessions[-last_sessions:]
    tools = [{"type": "function", "function": {"name": n, "description": f"The {n} tool.",
              "parameters": {"type": "object", "properties": p}}} for n, p in sorted(schemas.items())]
    for calls in sessions:
        # Spread the picks over the session: early calls and later ones with more history.
        step = max(1, len(calls) // min(per_session, len(calls)))
        for msgs in calls[::step][:per_session]:
            out.append(body(msgs, max_tokens, True, budget, tools))
    return out


def main():
    kind, dst, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
    if kind == "edit":
        root = rest[rest.index("--root") + 1] if "--root" in rest else os.path.join(os.path.dirname(__file__), "../../..")
        reqs = edit(root)
    elif kind == "chat":
        reqs = chat()
    elif kind == "kiln":
        opt = lambda k: int(rest[rest.index(k) + 1]) if k in rest else 0
        cap, every = opt("--max"), opt("--all-calls")
        dirs = [d for i, d in enumerate(rest) if not d.startswith("--") and (i == 0 or not rest[i - 1].startswith("--"))]
        # --all-calls N: every call of the last N sessions (consecutive calls, as served).
        reqs = kiln(dirs, per_session=10**9 if every else 2, last_sessions=every)
        if cap and len(reqs) > cap:
            # Evenly spaced, keeping chronological order.
            reqs = [reqs[i * len(reqs) // cap] for i in range(cap)]
    else:
        sys.exit(__doc__)
    with open(dst, "w") as f:
        for r in reqs:
            f.write(json.dumps(r) + "\n")
    print(f"{len(reqs)} requests -> {dst}")


if __name__ == "__main__":
    main()
