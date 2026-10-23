# GLM-5.3-Flash: real-model load path (UD-Q4_K_XL, 6 shards, ~244 GB)

How the first token run loads `unsloth/GLM-5.3-Flash-GGUF` UD-Q4_K_XL on a 2×H200 pod:
mmap only, never a whole-file read; the split picks per-layer byte budgets from the shard
headers; first tokens come from `glm5-ref` (CPU reference, exact) and then the CUDA engine
once the glm5 kernels are wired in.

## What already exists (no work needed)

- **`Gguf::open` is mmap-only** (`crates/tang-llm/src/gguf.rs`): `Mmap::map` per shard,
  the header parsed from the mapping, tensor data faulted in page-by-page only when a
  dequantiser walks it. The whole 244 GB is *never* read into RAM; RSS stays at header +
  whatever the current layer touched. `discover_shards` handles unsloth's
  `...-00001-of-00006.gguf` naming (test `gguf::tests::shard_names`); shard 1 of the real
  file is 9.4 MB of metadata + 0 tensors and `Gguf::open` takes metadata from the first
  shard and tensors from every shard, exactly matching that layout.
- **`moe::MoeParams::from_gguf` keeps experts as GGUF handles** (`W::Gguf(g, info)`), and
  `W::mat` dequantises one expert at a time (`g.dequantize`), so the 190 GB of routed
  experts is also streamed per layer, not resident. The fp32 reference forward never holds
  more than one layer's weights (~4.4 GB worst, the MoE layers).
- **Per-layer byte budget** for the split is already in `docs/models/glm5-gguf-map.md`
  (measured from the shard headers): routed experts 4.379 GB/sparse layer, everything else
  per family listed there; totals ~186 GiB of payload.
- **Split point** (`coder::split_point` / `balanced`) sums per-layer bytes from the tensor
  table and picks the balanced cut — for glm5 it needs `hp.n_layer` and the expert tensor
  names, which match; at the real geometry it lands at k=23 (92.3/93.6 GiB), same as the
  plan in `docs/models/glm5-split-plan.md`.

## The one gap: RAM ceiling of the fp32 reference on the real model

`glm5-ref` dequantises per layer and frees between layers, but the **MoE layers are the
risk**: W5's `MoeParams` keeps routed experts as GGUF handles and dequantises per expert
inside `route_and_run` — good — but `KdaLayerW`/`MlaWeights` load all their layer's
attention matrices up front (a few hundred MB, fine). The dense `Mat::apply` path materialises
`[t_len][rows]` outputs — at prefill t=52 that's noise; at serve-scale prefill (4096-token
chunks) it's 4096 × 8192 × 4 B = 134 MB per matmul, fine.

So `glm5-ref` runs the real model in well under 8 GB RSS. It is slow (f32 matmuls on CPU,
~0.5 tok/s single-threaded at hidden 4096) — that is what it is for: exactness, not speed.

## Pod launch commands (RunPod 2×H200)

```bash
# pod: 2×H200, 1.9 TB NVMe, container with CUDA 12.x + rust
git clone https://github.com/muni-works/tang tang && cd tang
cargo build --release -p tang-llm --no-default-features --features cuda
# (~6 min from a warm target/ on the persistent volume /root/tang)

# 1. fetch the shards (only headers needed for the split plan; full files for the run)
pip install -U huggingface_hub
huggingface-cli download unsloth/GLM-5.3-Flash-GGUF \
    --include "*UD-Q4_K_XL*" --local-dir /root/glm53

# 2. CPU reference first tokens (exactness gate; runs on CPU, no GPU needed)
target/release/tang-llm glm5-ref /root/glm53/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf \
    --ids-file /root/prompt.ids --top 8 --logits /root/first.logits -v
# shard discovery is automatic; RSS stays < 8 GB; ~0.5 tok/s at hidden 4096 on the pod CPU

# 3. GPU parity on the tiny GGUF (W0's fixture copied to /private/tmp/glm5-tiny on the pod,
#    the path the tests read): W2–W5 kernel tests + the engine's GPU-FFN vs CPU check
rsync -a <mac>:/private/tmp/glm5-tiny/ /private/tmp/glm5-tiny/
TANG_REQUIRE_CUDA=1 cargo test --release -p tang-llm --no-default-features --features cuda \
    --lib glm5 -- --nocapture
TANG_REQUIRE_CUDA=1 TANG_GPUS=2 TANG_SPLIT_AT=2 cargo test --release -p tang-llm \
    --no-default-features --features cuda --lib glm5::engine -- --nocapture

# 4. first tokens on the real model: one GPU can't hold the experts (190 GB), so start split.
#    TANG_GLM5_CACHE=1 keeps the f32 attention weights in RAM (~26 GB) instead of
#    re-dequantising them every token.
G=/root/glm53/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf
TANG_GPUS=2 TANG_GLM5_CACHE=1 target/release/tang-llm flash-generate $G \
    --prompt "Write a haiku about the sea." -n 32 --temp 0 -v --out /root/g2.ids
# (TANG_SPLIT_AT=23 is the balanced default it picks itself; set it to override)

# 5. split-vs-reference check: the same greedy ids from the CPU reference engine
TANG_GLM5_GPU=0 TANG_GLM5_CACHE=1 target/release/tang-llm flash-generate $G \
    --prompt "Write a haiku about the sea." -n 8 --temp 0 --out /root/cpu.ids
head -c 60 /root/cpu.ids; head -c 60 /root/g2.ids   # first 8 ids must match

# 6. throughput (naive; the number to beat, not a result)
TANG_GPUS=2 TANG_GLM5_CACHE=1 target/release/tang-llm flash-bench $G \
    --prompt-ids "1000 1001 1002" -n 64
```

Notes:

- `glm5-ref` takes the *first shard* and finds the rest. Metadata comes from shard 1;
  the 15 tensors of shard 6 (MTP) are found by the same discovery.
- `flash-generate` / `flash-bench` route a `glm5-next` GGUF to `glm5::engine::Glm5Engine`
  (bin dispatch, any build), not to the qwen3next `Coder` / Flash-Next `Engine`. The engine
  is resumable: KDA state + conv history per layer, DSA latent + indexer caches per layer.
  It is checked against the full reference forward on the tiny model (chunked prefill +
  decode: logits ≤3e-5, greedy identical).
- What runs where (naive-but-correct, GATES.md):
  - **GPU** (cuda build): every FFN, i.e. the dense lead-in and all routed + shared experts
    (W5's `DevMoe`, GGUF bytes resident, split across GPUs at `TANG_SPLIT_AT`).
  - **CPU f32**: embedding, mHC, KDA, MLA + indexer, norms, head. These are the small parts
    (~6 GB of the 196 GB). W2/W3/W4's CUDA kernels are written and type-check, but they
    aren't in the decode path yet. They're the next wiring step, and this engine is their oracle.
  - Activations cross host↔GPU once each way per FFN. That's slow but correct.
- `TANG_GLM5_GPU=0` forces the all-CPU engine; `TANG_GLM5_CACHE=1` caches mixer weights.
- Type-checking the cuda build without a toolkit: `CUDARC_CUDA_VERSION=12080 cargo check
  -p tang-llm --no-default-features --features cuda --lib --tests --bins`.
- Tokenizer: `tokenizer.ggml.pre = "glm4"`, supported since aff94dd6 (`tokenize.rs`); the
  chat template ships in the GGUF (`encode_chat`).
- For the engine the real quant mix is Q8_0 (attention/projections), Q4_K/Q5_K (gate/up
  experts), Q5_K/Q6_K (down experts) — all in `GgmlType::can_dequantize`; the CUDA engine
  uploads native blocks for its kernel types (Q8_0 direct; Q4_K/Q5_K/Q6_K through requant,
  same as Coder-Next's pipeline).
