# Tensor IR

`tang_compile::tensor` is a tensor-level IR for tang. This note is the
design argument; the code is the spec.

## The claim

Hand-written kernels are the fastest way to run one op on one chip today.
They are also forks: one per op, per backend, per fusion opportunity, each
re-deriving the same math. The claim is that if the *whole* model goes
through an IR, the compiler can run optimization passes over it and then
emit the best code for the network and chip at hand — and that past a
certain scale this beats hand kernels, because the decisions that matter
(what to fuse, where data lives, how to spread work across threads and
devices) are decisions a hand kernel has already frozen.

`BENCHMARKS.md` has the measurements. In short: every hand kernel we
compared against is matched or beaten by code derived from the IR, the
derived flash attention is 7.6× faster than the hand one at n = 1024, and
a whole transformer block derived from the IR runs 18% faster than the
hand-written path at seq 512. No op-specific code exists in the compiler.

## Two primitives

Everything is built from two ideas.

**Views.** A `View` is an affine map from an iteration space to a source
tensor's coordinates: one `Form` (`offset + Σ coef·idx`) per source dim.
Broadcast is a zero form, transpose a permutation, slice an offset, stride a
coefficient, "head `h` reads columns `64h+2i`" a form with two terms. Views
compose by substitution and never touch memory. Because they are the only
way a comprehension addresses its inputs, every layout decision is a view
and every view is analyzable.

A view can be *guarded*: a read outside the source returns 0 instead of
being a shape error. That is zero padding as a view — same-padded and
dilated convolutions, transposed (polyphase) convolutions and overlap-add
are guarded reads. The guard follows the leaf read through composition;
a map read through a guarded view stays materialized, since `f(0)` need
not be 0. The gradient of a guarded read eliminates a coordinate rather
than scattering: for `x[c, t + 3k − 9]`, `dx[c, τ]` reads `dy[o, τ − 3k + 9]`
through a guard, so padding contributions never reach `x`.

**Comprehensions.** The only compute op:

```
out[i…] = ⊕_{k…} body(x₀[view₀(i,k)], x₁[view₁(i,k)], …)
```

`body` is a scalar `tang-expr` graph (nine RISC ops), `⊕` is a `Monoid`
whose combine is also a scalar graph. A map is a comprehension with no
reduce dims; sum and max are monoids; matmul is `Σ_k a[i,k]·b[k,j]`;
attention scores, convolution and gather are the same op with different
views. A comprehension may produce several values per point — its trailing
`tuple` dims are filled by the body — so monoids can carry tuple state.

Dims are `Dim::Fixed(n)` or `Dim::Sym("seq")`. Construction, rewriting and
differentiation work with symbols; `bind` produces the concrete graph that
schedules, the interpreter and lowering need.

## What falls out

**Fusion is one rule.** A map is a pure function of its inputs, so
`substitute_maps` splices its body into every consumer and composes the
views. Map→map, map→reduce, transpose→anything: same rule. Reductions are
not substituted — they are *inlined* by a schedule decision, which is the
difference between algebra and scheduling.

**Reshape is a view, not a copy.** A row-major reshape of a map is read
through the map's own coordinates whenever the reshape only merges dims
seen from the reader (`View::through_reshape`); the reshape node
dissolves and the map inlines like any other. Only a reshape that splits
a source dim — which needs `/` and `%` — keeps its source materialized.

**Epilogues ride the reduction.** A map that reads a reduction
point-for-point — a bias, an activation, the `acc / l` that finishes an
online softmax — never gets its own kernel: the tiled and split
schedules run it on the thread that holds the final accumulator. Any
affine read of the reduction's points qualifies, and each arg picks a
constant tuple slot. A slot that cannot vary along a point dim (the
softmax row sum along its lanes, found through the monoid's combine) may
be read at any coordinate of that dim, since every thread's accumulator
agrees there.

**Siblings share a pass.** Two reductions over the same space with the
same monoid, read by the same consumers, are one reduction under the
product monoid (`merge_siblings`): the gate and up projections of a
SwiGLU MLP become one tuple-valued comp, their activation becomes its
epilogue, and the normed input is read once.

**Flash attention is a rewrite.** Online softmax is a monoid over
`(max, sum, acc…)`. `stabilize_softmax` finds `m = max_j s` and every
`Σ_j g·exp(s − m)` over the same `s` (compared as canonical
sub-expressions, so masks and scales are fine) and replaces them with one
reduction under that monoid. The extra out dims of the members — the value
column of `Σ_j p·v` — become point dims of the fused comp, which is what
lets a lane per column share the reduction later. Nothing n×n is ever
materialized. Logsumexp, Welford variance and top-k are the same move.

**Gradients are comprehensions.** For `y = ⊕_k body(x[v(i,k)])`, the
gradient w.r.t. `x` through a pick view is found by *renaming
coordinates*: the coords `v` picks become `x`'s out dims, the rest become
reduce dims, every other view is pushed through the same renaming, and the
body is differentiated by `tang-expr`'s symbolic `diff`. A dim read through
any other form — an offset, a stride, a diagonal, a broadcast — stays an
out dim and the body is gated by `[coord == form]` using `iota`, the
coordinates-as-values node. `Max` gates the term to the argmax with a
select. The online-softmax monoid has its own rule: with the forward's
`m` and accumulators read back, `∂accᵢ/∂s_j = gᵢ·exp(s_j−m) − accᵢ·[s_j = m]`,
so the flash-fused graph differentiates too and its gradients equal the
naive graph's. The backward graph is an ordinary graph; the same passes optimize
it. A whole transformer block (`tang-bench::ir_block`) differentiates this
way and matches finite differences.

**Schedules are separate from algebra.** A comprehension says *what*. The
schedule fields on a node say *how*, and can't change the answer:

- `inline` — recompute inside the consumer instead of writing a buffer.
- `split` — one workgroup per point; threads stride the reduce space and
  tree-combine through workgroup memory. Valid for any monoid because
  combine is associative — including online softmax.
- `tile` / `tile_2d` — a `bx × by` block of threads per workgroup, each
  owning an `rx × ry` register tile of points (`reg`). The emitter inlines the closure into one loop nest,
  tracks every address as an affine form over the loop variables, and
  stages anything that does not depend on the lane axis in workgroup
  memory once per reduce chunk: buffer reads (k/v tiles, the matmul `b`
  tile) and *inlined sub-computations* (the q·k scores) alike.
  Reduce-invariant per-thread reads (the q row) are hoisted to registers.
- `precision` — number formats (`set_precision(t, Precision { storage,
  accum })`): the format a materialized buffer or input is *stored* in
  (`F32 | F16 | BF16`) and the format a reduction *accumulates* in
  (`F32 | F16`). See [Precision is a schedule](#precision-is-a-schedule).
- `schedule_auto` — the rules of thumb distilled from measurement;
  `autotune` tries candidate graphs on the device and keeps the fastest.

The derived flash kernel — rows × lanes per workgroup, cooperative score
tiles, online softmax in registers — is what those rules produce from the
naive graph. No one wrote it.

**Hand kernels become lowering targets.** The remaining role for a
hand-written kernel is as an intrinsic: "this comprehension shape has a
fast implementation on this chip". The compiler picks it when it wins and
derives one when it doesn't. The existing kernels keep their speed and stop
being forks.

## Precision is a schedule

f16 is not a different program; it is a different way to run the same
one. So precision is a per-node schedule field, like `inline` and `tile`,
and the graph stays math. Everything defaults to f32, which is exactly
the old behaviour.

- **Storage** applies where a value is written to memory: graph inputs
  (weights) and materialized comps. An inlined comp has no buffer, so its
  storage is ignored. Loads widen to f32; arithmetic is f32.
- **Accum** applies to a reduction's running state: after every combine
  step the state is rounded to the format.
- **The interpreter predicts the error.** `eval` stays pure f64 and
  ignores precision. `eval_emulated` rounds every stored value and every
  accumulator step to its scheduled format (round to nearest even,
  subnormals, overflow to ∞) while body arithmetic stays f64.
  `precision_error` compares the two (`ErrorStats`: max / mean, abs /
  relative to the output's largest magnitude); `ErrorBudget { max_rel,
  max_abs }` admits or rejects a schedule. Emulation matches the GPU's
  f16 kernels to 1 f16 ulp when the reduction order is the same.
- **The reduction order is part of the error.** The interpreter folds a
  reduction sequentially. A `split` schedule tree-combines 256 partial
  sums, so the same f16 accumulator is far more accurate under it
  (AdaIN's variance over T = 2000, measured on the GPU: 2.8e-4 under a
  256-thread split, 1.3e-2 with one thread per row; the interpreter's
  sequential prediction is 7.9e-3). Emulation models the sequential
  order, the worst case.
- **WGSL lowering.** naga 24 rejects `enable f16`, so f16 storage is
  packed two per `u32` with core-WGSL `unpack2x16float` on load; a store
  merges its half into the word with two atomics on disjoint bits
  (neighbouring halves belong to different threads). No device feature
  needed. f16 accumulation is lowered as a round-trip through
  `pack2x16float` after each combine — the numerics of an f16
  accumulator, not its speed. bf16 is interpreter-only (WGSL has no bf16
  type; the lowering panics on it).
- **Search.** `tang_bench::ir_gpu::tune_precision` takes a scheduled
  graph and a budget, lowers buffers to f16 one at a time (largest
  first), runs each on the device against the f64 reference, and keeps a
  change only if the program stays within budget and gets faster (≥ 3%,
  re-timed at the end against f32). Accumulators are tried against the
  budget only.

Set precision after the rewriting passes (they rebuild the graph); the
schedule passes (`schedule_auto`, autotune) leave it alone. `backward`
appends to the same graph, so a forward input stored in f16 is read as
f16 by the gradient kernels too, and backward reductions (a snake's
`dα = Σ_t …`) take their own precision.

## What the measurements say about the claim

The winning schedule is a property of (graph, chip, size), not of the op:

- Attention at n = 1024 on an M4 Max wants the tiled flash schedule.
  At n = 512 it wants the *naive* graph with n² threads — the 1 MB scores
  buffer is free and parallelism is the bottleneck, not bandwidth.
- Rmsnorm wants a split reduction (few rows, wide axis).
- Matmul wants one thread per point; shared-memory tiling does not pay
  because the operands already sit in cache.

Then the same branch ran on an RTX 3090. The attention and matmul results
held; the transformer block *lost* until three chip-agnostic fixes landed
(cost-aware substitution, keeping reductions schedulable under their
epilogue maps, bank-conflict-free tile layout), after which the autotuned
block beat the hand path there too (2.5 vs 3.1 ms, 6.2 vs 10.4 ms). The
rules of thumb distilled on the M4 pick the wrong matmul schedule on
NVIDIA; the per-kernel tuner picks the right one on both.

That is the strong form of the claim: decomposing into IR and choosing the
schedule per target is not just as good as hand kernels, it is the only
way to be right on every chip at once. A hand kernel is one point in the
schedule space; the compiler searches it.

## Not yet

- Native codegen is a first cut. Matmul-shaped tiles lower to Metal
  simdgroup matrices (`tensor_msl.rs`, via wgpu's MSL passthrough) and
  to CUDA `wmma` TF32 tensor cores (`tensor_cuda.rs`), epilogues
  included; `to_cuda` emits whole programs (wmma tiles plus the other
  kernels transpiled from their WGSL) that `tang_bench::ir_cuda` runs
  through cudarc. The block forward runs in 0.62 ms at seq 128 on a 3090
  with six kernels on tensor cores — the five matmul-shaped tiles and
  the fused attention, which `cuda_flash_kernel` lowers to a wmma flash
  kernel from the comp's structure (bilinear score, mask, value,
  normalizer); the Metal fp32 kernel does not yet beat the WGSL
  register tile.
- The CUDA path is a separate runtime (tang-gpu runs on wgpu, which
  cannot dispatch CUDA); buffers do not interoperate between the two.
- Native half arithmetic. Storage is f16, math is f32; matrix units
  (Apple `simdgroup_matrix`, 11 TFLOP/s with f16 inputs vs ~3.3 for the
  IR's f32 tiles) need an MSL emitter that reads the precision schedule.
- Scatter gradients reduce over the whole source coordinate they gate on;
  a stride-2 read pays 2× in the backward pass. Fine for rope, not for a
  large gather.
- Non-affine indexing — KV-cache paging, MoE routing — needs a `Gather`
  view that reads an index tensor, which blocks some fusion.
- Placement is a transform now (`shard_reduction` + `fold_partials`:
  partial reductions over slices, the all-reduce as a comprehension over
  the monoid's combine; `examples/shard_attention.rs` runs flash
  attention's key axis across two devices), but the transport is
  host-mediated. A device-to-device runtime is the missing piece.
- A real cost model. `schedule_auto` is three measured rules; `autotune`
  is the honest answer until there is one.
