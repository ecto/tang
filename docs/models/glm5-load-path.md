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

# 3. split parity gate on the tiny GGUF (W0's fixture, uploads to the pod volume)
TANG_LLM_TEST_GGUF=/root/glm5-tiny/glm5-tiny-f32.gguf TANG_REQUIRE_CUDA=1 \
    cargo test --release -p tang-llm --no-default-features --features cuda \
    --test flash_split -- --nocapture

# 4. the engine, one GPU first (eager), then the split
TANG_GPUS=1 target/release/tang-llm flash-generate /root/glm53/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf \
    --prompt-ids 1000 1001 1002 --n 16 --temp 0 --eager --out /root/g1.ids
TANG_GPUS=2 TANG_SPLIT_AT=23 target/release/tang-llm flash-generate \
    /root/glm53/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf \
    --prompt-ids 1000 1001 1002 --n 16 --temp 0 --eager --out /root/g2.ids
cmp /root/g1.ids /root/g2.ids   # must be byte-identical

# 5. throughput
TANG_GPUS=2 TANG_SPLIT_AT=23 target/release/tang-llm flash-bench \
    /root/glm53/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf \
    --prompt-ids 1000 1001 1002 --n 128
```

Notes:

- `glm5-ref` takes the *first shard* and finds the rest. Metadata comes from shard 1;
  the 15 tensors of shard 6 (MTP) are found by the same discovery.
- The CUDA engine (`flash-generate`) still runs the **qwen3next** kernels; pointing it at a
  glm5-next GGUF fails the geometry check in `Engine::load` / `Coder::load` until the glm5
  CUDA path lands. `glm5-ref` is the first-tokens step while W2–W5's kernels are wired in.
- Tokenizer: `tokenizer.ggml.pre = "glm4"`, supported since aff94dd6 (`tokenize.rs`); the
  chat template ships in the GGUF (`encode_chat`).
- For the engine the real quant mix is Q8_0 (attention/projections), Q4_K/Q5_K (gate/up
  experts), Q5_K/Q6_K (down experts) — all in `GgmlType::can_dequantize`; the CUDA engine
  uploads native blocks for its kernel types (Q8_0 direct; Q4_K/Q5_K/Q6_K through requant,
  same as Coder-Next's pipeline).
