# Qwen3-Coder-Next decode speed, measured like Artificial Analysis

This is the method behind tang's Qwen3-Coder-Next speed claim: output tokens per second for a
single stream, measured after the first token, on a ~1k-token prompt. Artificial Analysis
reports API speed the same way.

## Setup

- **Model:** `unsloth/Qwen3-Coder-Next-GGUF`, file `Qwen3-Coder-Next-UD-Q4_K_XL.gguf`
  (49.6 GB). Weights are used as stored: Q8_0 dense, Q4_K/Q5_K experts, F32 router. The Q8_0
  rows are repacked losslessly at load (Q8X).
- **Engine:** `tang-llm`'s Flash engine, qwen3next path (`crates/tang-llm/src/flash/coder.rs`),
  greedy, with the window as one CUDA graph.
- **GPU:** one H200 SXM 141 GB (RunPod secure cloud, CUDA 12.8.1 image). Record the driver
  with `nvidia-smi` for each run.

## Prompts

`make_prompts.py` writes `explain.ids`, `refactor.ids` and `write.ids`. Each is one user turn in
the model's chat template, holding the same 140-line Rust file (`context.rs`) and a different
task:

| prompt | input tokens | task |
|---|---|---|
| explain | 1,235 | explain the module and its performance pitfalls (fresh prose) |
| refactor | 1,238 | return the whole file refactored (output largely copies the input) |
| write | 1,263 | write a new LRU cache module with tests (fresh code) |

## Measurement

`tang-llm flash-aa` loads the model once, then for each prompt and each drafts setting
(`none`, `suffix`) runs the following:

- **Prefill:** the prompt.
- **Output:** 512 tokens, greedy, decoded.
- **Score:** tokens 3..512 divided by their wall time. The first two tokens are excluded: one
  is the prefill's output, the other the first decode step, which captures its graph. This is
  the rate after the first token.

Drafts:
- `none`: one token a window.
- `suffix`: the n-gram suffix drafter (`crate::draft`) proposes up to 7 tokens from the
  sequence so far. Each window verifies them exactly, so the output is identical to `none`
  (checked by `flash-spec-test`).

The first prompt of a fresh process reads about 4% low (warm-up). Discard it, or use the
kernel-variant block, which warms up first.

## Reproduce

On a rented H200 with the CUDA toolkit:

```bash
cargo build --release -p tang-llm --no-default-features --features cuda --bin tang-llm
hf download unsloth/Qwen3-Coder-Next-GGUF --include "*UD-Q4_K_XL*" --local-dir m
python3 docs/bench/qwen3next-aa/make_prompts.py --out ids
for i in 1 2 3; do
  target/release/tang-llm flash-aa m/Qwen3-Coder-Next-UD-Q4_K_XL.gguf \
    --ids-files ids/explain.ids,ids/refactor.ids,ids/write.ids -n 512 --bw 4800
done
```

Correctness on the same build:

```bash
target/release/tang-llm flash-ref m/Qwen3-Coder-Next-UD-Q4_K_XL.gguf $(cat ids/explain.ids) --top 40 > ref.jsonl
target/release/tang-llm flash-parity m/Qwen3-Coder-Next-UD-Q4_K_XL.gguf --ids-file ids/explain.ids --ref ref.jsonl
target/release/tang-llm flash-spec-test m/Qwen3-Coder-Next-UD-Q4_K_XL.gguf --ids-file ids/explain.ids -n 96
```

Run `flash-ref` before or after a benchmark, never during one: it uses the GPU too.

## Results

Release run: H200 SXM (RunPod secure), driver 595.91.07, CUDA 12.8.1 image,
`Qwen3-Coder-Next-UD-Q4_K_XL.gguf`, tang commit 79461fc (#110). n = 3 fresh processes, each
running every prompt × drafts setting once. 512 output tokens, greedy. Tok/s is measured after
the first token.

| prompt | drafts | tok/s min / median / max (n = 3) | tokens a window |
|---|---|---|---|
| explain | none | 234.4 / 234.4 / 234.4 | 1.00 |
| refactor | none | 234.3 / 234.4 / 234.4 | 1.00 |
| write | none | 234.1 / 234.2 / 234.4 | 1.00 |
| explain | suffix | 237.3 / 238.1 / 238.5 | 1.13 |
| refactor | suffix | 411.3 / 411.4 / 411.4 | 2.71 |
| write | suffix | 256.6 / 256.6 / 256.9 | 1.25 |

**Headline (drafts off): 234 tok/s**, single stream, on all three prompts. Drafts change the
speed but never the output. They add about +1% on prose, +10% on fresh code, and +75% on
edit-style output that copies its input.

Earlier rows (n = 1 per pod, other pods) are in `crates/tang-llm/docs/speed-log.md`: 241–247
on kernel-variant runs, after a warm-up. Pods differ by a few percent.

### Serving load (same pod, `zest-harness` straight at `tang-llm serve`)

`zest-harness --levels 1,2,4,8 --rounds 3 --max-tokens 256` (its built-in prompts, streaming),
with suffix drafts off:

| streams | per-stream tok/s p10 / p50 / p90 | total tok/s | TTFT p50 / p90 ms |
|---|---|---|---|
| 1 | 205.0 / 205.5 / 205.5 | 186 | 107 / 126 |
| 2 | 205.3 / 205.5 / 205.7 | 195 | 1,307 / 1,413 |
| 4 | 205.5 / 205.6 / 205.7 | 200 | 3,070 / 3,854 |
| 8 | 205.6 / 205.7 / 205.8 | 200 | 7,374 / 8,157 |

`flash serve` decodes one sequence at a time, so requests queue. Per-stream speed holds,
total throughput stays at one stream's, and time to first token grows with the queue.
Batching is designed in `crates/tang-llm/docs/qwen3next-batching.md`. With suffix drafts on,
the harness reads 400–590 tok/s total, but its prompts repeat, so the drafter replays
memorized greedy completions (7.5 tokens a window). That's not representative.

### Higher precision: Q8_0 (2026-10-21, H200 secure, same method, n = 3)

FP8 doesn't load: the Flash engine reads GGUF, and FP8 Coder-Next is published only as
safetensors. The closest is unsloth's `Q8_0` GGUF (84.8 GB in 3 shards, 84.8 GB of VRAM). Its
8-bit experts take the generic Q8_0 dot, not the fast K-quant path.

| prompt | drafts | Q8_0 tok/s min / median / max | UD-Q4_K_XL median |
|---|---|---|---|
| explain | none | 221.9 / 222.1 / 222.3 | 234.4 |
| refactor | none | 222.0 / 222.2 / 222.2 | 234.4 |
| write | none | 222.0 / 222.1 / 222.2 | 234.2 |
| explain | suffix | 220.5 / 220.6 / 220.7 | 238.1 |
| refactor | suffix | 374.8 / 374.9 / 374.9 | 411.4 |
| write | suffix | 232.1 / 232.4 / 232.5 | 256.6 |

Q8_0 is about 5% slower than the Q4 headline.

KL of UD-Q4_K_XL against Q8_0 (`flash-parity`, Q8_0's top-40 per position as the reference,
teacher-forced over the prompt):

| prompt | positions | KL mean / median / p99 | top-1 agreement |
|---|---|---|---|
| explain | 1,235 | 0.0127 / 0.0005 / 0.171 | 96.8% |
| write | 1,263 | 0.0180 / 0.0005 / 0.321 | 97.0% |

### Through `tang-llm serve` (the head-to-head harness, `../h2h/`)

Same pod, UD-Q4_K_XL, drafts off, n = 3, streaming over HTTP on localhost. Each run here used
the same prompt; later runs vary the context file per run.

| prompt | tok/s after first token, median | TTFT cold (first run) |
|---|---|---|
| explain / refactor / write | 211.5 / 211.6 / 211.5 | 1.85 / 0.11 / 0.13 s (the latter two had the shared file cached) |
| agent (~10k tokens) | 174.5 | 16.4 s |

pass@1 on the 20 tasks: 20/20. The served rate is ~10% below `flash-aa`'s (211.5 vs 234),
because `serve` runs the top-k/top-p sampler even on greedy requests. A fix is planned.
