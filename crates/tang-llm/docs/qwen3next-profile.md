# Where a Qwen3-Coder-Next decode step goes on an H100

Profile of the qwen3next Flash engine (`src/flash/coder.rs`) on one H100 SXM 80GB (RunPod
secure, EUR-IS-3, CUDA 12.8.1 image), unsloth `Qwen3-Coder-Next-UD-Q4_K_XL.gguf` (49.6 GB),
2026-10-20. Single stream, one sequence, a whole window (48 layers, embedding to sampled
token) as one CUDA graph.

**Method.** `docs/h200-profile.md` used Nsight Systems. Here neither profiler would run:

- `ncu` fails at load: its injection library lacks `cuTensorMapEncodeIm2colWide`, which
  cudarc resolves eagerly.
- `nsys` is not in the image, and the apt install failed.

So the numbers below come from the engine's own profiler instead. `TANG_CODER_PROF=1` (eager
windows only) records a CUDA event after every launch and charges the time since the previous
event to that launch's stage. That adds a few µs a launch, so stage times are upper bounds and
small kernels are overstated. Window wall times are from graph replay
(`flash-bench`, no profiler).

There is no host work in the decode loop worth measuring. A window is:

- staging (the control words, plus `t` embedding rows dequantised on the CPU, ~10 µs);
- one graph launch;
- one stream sync;
- reading `t` sampled ids from pinned memory.

The graph has no syncs or host round trips inside it.

## Results

`flash-bench --ctx 4096 -n 256` (the code prompt cycled to 4,096 tokens), speed of light at
3.35 TB/s. A step reads 3.52 GB: dense weights, 10 routed experts and the shared expert per
layer, the head, and K/V.

| build | drafts | decode tok/s | ms / token | % of speed of light (951 tok/s) |
|---|---|---|---|---|
| #84 (before) | none | 156.2 | 6.39 | 16.5% |
| #84 (before) | suffix* | 409.8 | | |
| fused, first cut | none | 172.5 | 5.79 | 18.1% |
| fused, first cut | suffix* | 222.1 | | |
| fused, first cut, `TANG_CODER_GEMV=v1` (no K-split) | none | 159.3 | | 16.8% |
| fused + 1024-thread norms + top-k fix (this PR) | none | **198.1** | 5.05 | **20.8%** |
| fused, chat prompt (33 tokens, no cycling) | none | 208.5 | | 21.4% (979 tok/s at ctx 161) |
| fused, chat prompt | suffix | 203.6 | | |

\* The cycled code prompt repeats every 702 tokens, so the suffix drafter copies whole
stretches and the acceptance rate depends on where the greedy output falls into the cycle. The
gap between the two suffix rows is that, not the kernels. The chat prompt (no repetition) is
the honest drafting number: drafts barely fire there, so it is no faster than none.

**8 streams:** not measured. The qwen3next engine decodes one sequence; batching several needs
per-sequence GDN state, K/V and windows, the same work `serve` does for the dense path.

Correctness of the fused build, on the same file:
- `flash-parity` vs `flash-ref`: chat KL mean 0.0027 (top-1 100%); code 0.0043 (97.9%);
- `flash-tcheck`: t1, t8 and t8-verify bitwise equal at every layer;
- `flash-spec-test`: PASS.

## One decode step (fused build, eager, ctx 4096)

| stage | launches / step | share | what it is |
|---|---|---|---|
| `moe_gu` | 48 | 15.6% | routed + shared gate\|up rows, 11 × 1024 rows × 2048 |
| `moe_down` | 48 | 12.1% | routed + shared down rows, 11 × 2048 rows × 512 |
| `gdn_in` | 36 | 11.9% | stacked qkv\|z\|a\|b GEMV, 12,352 × 2048 |
| `gdn_out` | 36 | 6.9% | 2048 × 4096 GEMV (+ its int8 quantisation) |
| `route` | 48 | 6.5% | top-10 of 512, one warp a token |
| `moe_add_norm` | 48 | 6.2% | combine + residual + next norm + int8 |
| `norm_q8` | 49 | 5.4% | post-mixer residual + norm + int8 |
| GDN conv+step, commit | 72 | ~7% | |
| attention (prep, part, merge, in, out) | 60 | ~14% | 12 layers |
| router GEMV, head, argmax | | ~7% | |

Before the fusions, a layer was about 22 launches. The top stages were the same GEMVs, but with
separate int8 quantisations, a separate shared expert (gate\|up GEMV, SiLU, down GEMV) and
single-block norms. Per launch: add+norm 9.4 µs, `route` 11.7 µs, `moe_gu` 19.6 µs.

## What the numbers say

**The step is latency-bound, not bandwidth-bound.** A layer reads about 73 MB. At 3.35 TB/s
that is 22 µs; it takes about 105 µs. No single kernel dominates:

- the biggest, `moe_gu`, reads ~12 MB in ~20 µs (~0.6 TB/s);
- the small ones (`route`, the norms) are ~5–10 µs of pure latency each, at one block or
  one warp per token.

**What helped, and by how much:**

1. **1024-thread norms and a top-k without the global softmax** (the denominator cancels in
   the top-k renormalisation): the two norm stages fell from 18% to 12% of the step,
   172 → 198 tok/s (+15%).
2. **K-split dense GEMV**: 159 → 172 (+8%), A/B against `TANG_CODER_GEMV=v1` on the same
   build. As in #80, four warps a row fill the SMs for the 2048-row outputs, which had ~8 warps
   an SM before.
3. **Fusions**: 22 → 12 launches a layer, only 156 → 159 (+2%). Fewer graph nodes barely
   matter while each remaining kernel is slow, which confirms the step is kernel-latency-bound
   rather than launch-bound.

## Next, in order of expected gain

1. **Expert GEMVs at full bandwidth** (28%): one kernel for gate|up and down per layer with a
   grid barrier between them (all 11 experts' rows), and more rows a warp with the next rows'
   loads in flight (`cp.async` double-buffering, as `fe_gemv`'s wide path does). Target
   ~2 TB/s: about −1 ms a step.
2. **`gdn_in` / `gdn_out`** (19%): the same K-split already applies. What's left is the int8
   quantisation launch before `gdn_out` (fold it into the GDN step's output, which already
   holds each head's values) and `cp.async` staging.
3. **Small serial kernels** (`route`, norms, commit: ~20%): move top-k into the router GEMV's
   last block (a grid-done counter), and fold the GDN conv-history commit into the step
   (each v-head block owns its channels).
4. **A persistent per-layer kernel.** With every stage at bandwidth, the floor is the ~12
   launch ramps a layer (~2–3 µs each in a graph, 1–1.5 ms a step). Past ~400 tok/s, the step
   needs a few persistent kernels with device-side barriers instead of ~580 graph nodes.

For the 500 tok/s target: (1) + (2) + (3) should reach ~350 tok/s single-stream without drafts,
and (4) is what crosses 500. Drafting adds more only when the drafts are accepted. That is the
case for code edits and repeated text, but not for fresh prose (chat prompt above).

Cost of this run: one H100 SXM for ~1 hour, $3.99/hr.
