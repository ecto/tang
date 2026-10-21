# tang-llm

LLM inference on [tang-compute](https://crates.io/crates/tang-compute): Metal, CUDA and CPU
backends, with kernels compiled for the exact model shape at load.

- **Models:** Qwen3 dense (8B, 32B, …), Gemma 3 (text and vision), Qwen3.8-Flash-Next and
  Qwen3-Coder-Next / Qwen3-Next on the Flash engine (CUDA), and gpt-oss. MLX 4-bit and bf16
  safetensors, GGUF on the Flash engine. See [docs/models.md](docs/models.md).
- **Serving:** `tang-llm serve` speaks the OpenAI chat API (streaming, tool calls, structured
  output with llguidance), with batched decode across requests, time-budgeted chunked prefill,
  a content-addressed paged KV cache with a disk tier and optional retention limits
  (`--kv-ttl`), and `/node` for schedulers.
- **Speed:** lossless speculative decoding (suffix and MTP drafts, output identical to plain
  decoding at the same seed) and batch-invariant kernels. Numbers and method are in
  [docs/speed-log.md](docs/speed-log.md).

```sh
cargo install tang-llm --features metal        # Apple silicon
cargo install tang-llm --no-default-features --features cuda   # NVIDIA, CUDA 12.x
tang-llm serve mlx-community/Qwen3-8B-4bit
```

CUDA setup and what fits in 24 GB: [CUDA.md](CUDA.md). Optional features: `metal`, `cuda`,
`otel` (OpenTelemetry traces and step metrics), `mtp-train` (train MTP drafters with
tang-train).

License: MIT.
