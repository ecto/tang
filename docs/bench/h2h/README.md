# Head-to-head: any OpenAI-compatible endpoint

Speed and a small quality-parity check, the same for every endpoint (ours, Novita, anyone's), so
a comparison is like for like. Stdlib Python only.

```bash
API_KEY=... python3 docs/bench/h2h/h2h.py --base-url https://api.example.com/v1 --model MODEL --label example --out example.json
```

## Speed

Four prompts, each run `--n` times (default 3), one request at a time, streaming, temperature 0,
512 output tokens:

| prompt | input | what it is |
|---|---|---|
| explain, refactor, write | ~1.2k tokens | the AA-style prompts in `../qwen3next-aa/` (the same Rust file, three tasks) |
| agent | ~10k tokens | a system prompt, five tool schemas, a 34 KB Rust file (`agent_context.rs`), a review task |

Each run reports:
- **TTFT:** time to the first content, reasoning or tool-call delta.
- **Speed after the first token:** (completion tokens − 1) / (last delta − first delta).
  Completion tokens come from the stream's final `usage` chunk. An endpoint that sends no usage
  gets the delta count, marked `approx`.

## Quality

Twenty small Python tasks (`tasks.py`, written for this harness), each with asserts:
- The model writes the function at temperature 0.
- The harness runs the asserts in a subprocess (Python isolated mode, a temp dir, 10 s timeout).
- The score is pass@1. `--selftest` checks the asserts against reference solutions
  (`tasks_ref.py`).

**This executes model-written code on the client.** For endpoints you don't control, run it in
a container or a throwaway VM.

## Same-day protocol

Run every endpoint on the same day, from the same client host, with the same `--n`. Compare the
`--out` JSON files. They record each run's numbers, the date and the client host. `--skip-speed`
or `--skip-quality` runs one half.
