# tang-llm on NVIDIA GPUs (CUDA)

Written for an RTX 3090 (24 GB, Ampere, sm_86) on Linux. Anything from Turing up should work.

## What runs on the GPU

Everything in the forward pass, with the same semantics as Metal: MLX 4-bit and bf16 weights
stay packed on the GPU (decode GEMVs read them directly; prefill widens a chunk of rows at a
time into cuBLAS SGEMM), and attention (split-KV decode, tiled prefill) handles GQA, Gemma's
sliding windows, bidirectional image blocks and head dims up to 256. Activations and the KV
cache are f32, as on Metal. Qwen3 and Gemma 3 (text and vision) are covered.

## Requirements

- NVIDIA driver 525 or newer (`nvidia-smi` should list the card).
- CUDA toolkit 12.0 to 12.9, for `nvcc` (the build reads the version from it) and the runtime
  libraries tang loads at startup: `libcuda.so` (driver), `libnvrtc.so` (kernels are compiled
  on first use) and `libcublas.so`/`libcublasLt.so`. If they aren't on the loader path:
  `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH`.
  Without `nvcc` on `PATH`, name the version instead: `export CUDARC_CUDA_VERSION=12080`
  (for 12.8; 12090, 12060, ... 11040 are accepted). CUDA 13 isn't supported by the pinned
  cudarc yet.
- Rust 1.94 or newer (`rustup update`).

## Build

```sh
cargo build --release -p tang-llm --no-default-features --features cuda
```

`--no-default-features` drops the Metal backend (on by default, macOS-only). The binary is
`target/release/tang-llm`. On a Mac, `--features cuda` type-checks the CUDA backend (set
`CUDARC_CUDA_VERSION=12080`, there's no `nvcc`), but it can only run on an NVIDIA machine.

## Run the server

Models come from a directory or the local Hugging Face cache (download first, e.g.
`huggingface-cli download mlx-community/Qwen3-8B-4bit`).

```sh
./target/release/tang-llm serve mlx-community/Qwen3-8B-4bit --device cuda --host 0.0.0.0 --port 8911
```

`--host 0.0.0.0` makes it reachable from the network (the default is 127.0.0.1). There's no
authentication, so keep it on a trusted LAN. It speaks the OpenAI chat API:

```sh
curl http://<box>:8911/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 64}'
```

`--device auto` (the default) picks Metal, then CUDA, then the CPU, so `--device cuda` is only
there to fail loudly if the GPU isn't found.

### What fits in 24 GB

Memory is the weights plus an f32 KV cache sized for `--ctx` (default 32768) up front:

| model | weights | KV cache per token | at `--ctx 32768` |
|---|---|---|---|
| mlx-community/Qwen3-4B-4bit | 2.1 GB | 288 KiB | 9.0 GiB |
| mlx-community/Qwen3-8B-4bit | ~4.6 GB | 288 KiB | 9.0 GiB |
| mlx-community/Qwen3-14B-4bit | ~8.3 GB | 320 KiB | 10 GiB (tight: use `--ctx 16384`) |
| mlx-community/gemma-3-4b-it-4bit (with vision) | 3.2 GB | 272 KiB | 8.5 GiB |
| mlx-community/gemma-3-12b-it-4bit | ~8 GB | 768 KiB | too big: use `--ctx 8192` (6 GiB) |

Qwen3-8B-4bit at the default context is the comfortable choice. Leave ~2 GB for prefill
scratch (a dequantized weight chunk is up to 128 MB) and the CUDA context.

## Validate on the 3090

1. Kernel parity: every CUDA LLM op against the CPU reference, on small shapes (Qwen and Gemma
   head dims, GQA, windows, bidirectional blocks, 4-bit and bf16 weights). Without a GPU these
   tests skip; `TANG_REQUIRE_CUDA=1` makes a missing GPU a failure instead.

   ```sh
   TANG_REQUIRE_CUDA=1 cargo test --release -p tang-compute --features cuda --lib cuda::llm
   ```

2. End to end: logits from the CUDA backend against tang's CPU backend for a real checkpoint,
   through one prefill and through token-by-token decode (same top-1 everywhere, KL < 1e-3).
   The CPU side widens the weights to f32 and is slow, so use small models:

   ```sh
   huggingface-cli download mlx-community/Qwen3-0.6B-4bit
   huggingface-cli download mlx-community/gemma-3-1b-it-4bit
   TANG_LLM_TEST_MODEL=mlx-community/Qwen3-0.6B-4bit \
     cargo test --release -p tang-llm --no-default-features --features cuda --test gpu_logits -- --nocapture
   TANG_LLM_TEST_MODEL=mlx-community/gemma-3-1b-it-4bit \
     cargo test --release -p tang-llm --no-default-features --features cuda --test gpu_logits -- --nocapture
   ```

   `TANG_LLM_TEST_TOKENS=600` runs past Gemma 3 1B's 512-token sliding window (several minutes
   on the CPU side). Gemma 3 was brought up on the 4B checkpoint; 1B is the same architecture,
   small enough for the CPU reference.

3. Against Hugging Face transformers (fp32), with a bf16 checkpoint (needs `torch` and
   `transformers`; `STEP=1` checks the decode path instead of prefill):

   ```sh
   huggingface-cli download Qwen/Qwen3-4B
   python crates/tang-llm/scripts/check_logits.py \
     ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/<rev> target/release/tang-llm
   STEP=1 python crates/tang-llm/scripts/check_logits.py <same dir> target/release/tang-llm
   ```

   `check_logits_mlx.py` and `check_vision_mlx.py` compare MLX checkpoints against mlx-lm /
   mlx-vlm; with `pip install "mlx[cuda]" mlx-lm mlx-vlm` they may run on Linux too.

4. Vision tower (Gemma 3): the same pixels through Metal and CUDA should give the same
   features. Make a test image once, run it on both machines, compare:

   ```sh
   python -c "import numpy as np; np.random.default_rng(0).uniform(-1, 1, (896, 896, 3)).astype('<f4').tofile('px.f32')"
   ./target/release/tang-llm image-features <gemma-3-4b-it-4bit dir> px.f32 > cuda.json
   # on the Mac: tang-llm image-features <same model> px.f32 > metal.json
   python -c "import json, numpy as np; a, b = (np.array(json.load(open(f))) for f in ('cuda.json', 'metal.json')); print(np.abs(a - b).max(), np.abs(b).max())"
   ```

5. Speed: `./target/release/tang-llm generate <model> 785 3974 13876 -n 128` prints prefill and
   decode tok/s.

## Known limits

- f32 activations and KV cache, like Metal: exact, but twice the cache memory of bf16.
- The `tang-llm` binary runs prefill GEMMs on TF32 tensor cores (10-bit mantissas in the
  products, f32 accumulate); `GAIA_TF32=0` switches back to FP32. On an RTX 3090 with
  Qwen3-8B-4bit, cold prefill goes 928 → 1217 tok/s at 1k tokens, 644 → 758 at 4k and
  416 → 461 at 9k; against the CPU reference (Qwen3-0.6B-4bit) max KL goes from ~1e-9 to ~3e-5,
  same top-1. Decode is unaffected (its GEMVs don't use cuBLAS). The library default
  (`CudaComputeDevice::new`, so the tests) stays FP32 unless `GAIA_TF32=1`.
- Attention is plain CUDA-core code (no tensor cores). With more than one query (prefill, and a
  speculative-decoding verify of k tokens) it runs one block per 32 queries per head, each
  walking every cached key, so few queries against a long cache use a fraction of the GPU: at
  9k tokens attention dominates prefill, and verifying 2 tokens on a 16k cache takes ~280 ms
  against ~18 ms for one (`tang-llm bench-verify`). Single-token decode splits the keys across
  blocks and is fine.
- Kernels are compiled with NVRTC the first time each is used, so the first request is a little
  slower.
