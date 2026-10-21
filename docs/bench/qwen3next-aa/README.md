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

So far each number is one run (n = 1) per prompt per setting. The release table needs n ≥ 3 on
one pod. The rows are in `crates/tang-llm/docs/speed-log.md` (Qwen3-Coder-Next section).

| commit | GPU | prompt | drafts | tok/s (n = 1) |
|---|---|---|---|---|
| 4a76565 | H200 | explain | none | 246.7 |
| 4a76565 | H200 | explain | suffix | ~250 |
| 4a76565 | H200 | write | suffix | 247.2 |
| 4a76565 (Q8X + bf16 router build) | H200 | refactor | suffix | 403.5 (2.71 tokens a window) |

Drafts-off is the honest headline. Drafts only pay when the output repeats text the drafter
has seen: about +70% on refactor-style edits, +1–9% on fresh code and prose.
