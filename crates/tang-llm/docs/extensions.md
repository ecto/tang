# Extensions: faster kernels from another crate

tang ships one set of kernels that runs everywhere. Some deployments want faster ones for a
particular GPU and model (a persistent megakernel for sm_90, a CUDA-graph step, a fused
all-reduce, a learned low-bit format) that live in another crate, possibly a private one. This
is the seam they plug into. The built-in kernels stay the fallback for everything an extension
doesn't cover, or declines.

Code: `tang_compute::ext` (registry, op slots), `tang_llm::ext` (step slot, parity harness),
`tests/extensions.rs` (end-to-end example).

## How ops dispatch today

`Model<D: ComputeDevice>` holds its weights as `D::Buffer`s and calls the device for every op:
`dev.linear` (dense, bf16, Q4, through `Model::linear_rows`), `dev.rms_norm`,
`dev.attention_prep_paged` + `dev.kv_attention_paged` (or `dev.attention_paged_rows` for
batched one-token decode), `dev.swiglu_split`, and so on. Each backend (`cpu.rs`, `metal.rs`,
`cuda.rs` + `cuda/llm.rs`) implements the trait; CUDA kernels are CUDA C strings in
`kernels/*_cuda.rs`, compiled by NVRTC on first use and cached per device by source hash
(`CudaComputeDevice::module_cache`). The engine only calls `Model::forward` /
`Model::forward_batch`.

So there are two natural places to swap in an implementation: per op call, and per forward.

## The registry

A **slot** is a newtype around an `Arc<dyn Fn>` with the op's signature, one per thing that can
be replaced:

| slot | replaces | in |
| --- | --- | --- |
| `Linear<D>` | `dev.linear` in the decoder layers and the batched head (GEMV/GEMM, any weight format) | tang-compute |
| `RmsNorm<D>` | the decoder layers' `dev.rms_norm`s | tang-compute |
| `KvAttentionPaged<D>` | `dev.kv_attention_paged` (prefill, verify windows, single decode) | tang-compute |
| `AllReduce<D>` | tensor-parallel reduction across GPUs | tang-compute |
| `Step<D>` | all of `Model::forward_batch` (megakernel, CUDA graph) | tang-llm |

An implementation is registered as an `Entry { name, matches, priority, imp }`. `matches` is a
`Match` over the **target**:

- `arch`: `ComputeDevice::arch()`: `sm_86`, `sm_90`, `sm_100` (from the compute capability),
  `metal`, `cpu`;
- `model`: `model_type` from `config.json` (`qwen3`, `gemma3_text`, ...);
- `shape`: hidden, layers, heads, KV heads, head dim, FFN width, vocab (exact, or any);
- `quant`: how the matrices are stored (`f32`, `bf16`, `q4`, or a format an extension owns).

Empty fields match anything. Among the entries that match, the highest `priority` wins (ties:
the later registration). Slots are keyed by Rust type (`TypeId`), so `Linear<CudaComputeDevice>`
and `Linear<MetalDevice>` are separate registries and an implementation is always typed against
the concrete device it runs on: it can reach `CudaComputeDevice`'s stream, its NVRTC cache, and
the raw pointers of its buffers.

`Model::load` builds the target and resolves every slot once (`tang_llm::ext::Ext`); the hot
path then checks an `Option` per call. Each implementation may still **decline** a call by
returning `None` (a GEMV kernel asked for 64 rows; a decode megakernel given a prefill chunk),
and the built-in kernel runs that call. A step override that declines falls back to
`Model::forward_batch_builtin`, which it can also call itself.

### Granularity

- **Single ops** (`Linear`, `RmsNorm`, `KvAttentionPaged`): drop-in replacements with the
  device method's exact contract. The cheapest thing to write and to verify.
- **Whole step** (`Step`): gets `&Model<D>` and the batch's `Segment`s and must do everything
  `forward_batch` does: write each segment's K/V into its paged cache, append its tokens, and
  return the same logits. `Model::layer_weights(l)` and `Model::io_weights()` hand it the
  weights (read-only, in the load format), `Model::pool()` the KV blocks. A persistent
  megakernel or captured graph builds its launch state once, keyed by the model, and reuses it.
- **Multi-GPU** (`AllReduce`): tang has no tensor-parallel path yet. The slot is defined so the
  path that grows one resolves its reduction the same way (built-in: copy through host or NCCL;
  extension: a fused one-shot all-reduce).

Batched one-token decode attends through `dev.attention_paged_rows`, which has no op slot yet:
a fast decode attention covers it today through `Step`, and gets its own slot when one needs it.

## Registering

No dynamic loading. The extension crate exposes a function that calls
`tang_compute::ext::register` (or the re-export `tang_llm::ext::register`) for each entry, and
the binary calls it at startup, before loading a model:

```rust
fn main() -> anyhow::Result<()> {
    #[cfg(feature = "pro")]
    tang_pro::register();
    // ... Engine::load / Model::load as usual
}
```

The downstream binary owns the cargo feature (`pro = ["dep:tang-pro"]`); tang itself has no
feature, no dependency and no `cfg` for it. A model loaded before `register` keeps what it
resolved; `Model::resolve_ext` resolves again. `TANG_EXT=0` disables every extension at
resolve time, for A/B runs against the built-in kernels on the same binary.

## Correctness

A registered implementation is held to the bar the built-in kernels are:

- `tang_llm::ext::parity::linear` / `rms_norm`: the implementation against the device's own
  built-in op over a list of shapes, within a tolerance (the built-ins are themselves held to
  the CPU by the `*_vs_cpu` tests). Cases the implementation declines are skipped.
- `tang_llm::ext::parity::model`: a model end to end (prefill of several prompts, then batched
  greedy decode), with the model's extensions and with the built-in kernels: logits within a
  tolerance at every step and **identical greedy tokens**, which is the property
  `tests/speculate.rs` holds speculative decoding to. `parity::tiny_checkpoint` writes a small
  random Qwen3 checkpoint so this runs without a download; point it at a real checkpoint too.
- `tests/speculate.rs` and the other `TANG_LLM_TEST_MODEL` tests run unchanged with an
  extension registered (they go through `Engine`, which loads through `Model::load`), so an
  extension crate runs them by registering first. Kernels that change summation order across
  batch sizes break `bitwise_exact_with_short_drafts`; that is a real regression, not noise.

`tests/extensions.rs` is the template: renamed copies of the built-in ops and step, registered
for the device under test, resolved, observed on the hot path (call counters), and passed
through every check with zero tolerance. It runs on the CPU, Metal and CUDA.

## Cost when nothing is registered

`Model::load` takes the registry lock once per slot and finds nothing. Per op call the model
checks one `Option` (`None`) and calls the same device method as before; `forward_batch` checks
one more. No allocation, no locking, no dynamic dispatch on the hot path, and no behaviour
change: outputs are bit-identical (the same kernels run in the same order).

The one API change: `ComputeDevice` now requires `'static` (every device already is), and gains
`arch()` with a default.
