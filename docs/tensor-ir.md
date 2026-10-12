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
- `tile` / `tile_2d` — a `bx × by` block of points per workgroup, one
  point per thread. The emitter inlines the closure into one loop nest,
  tracks every address as an affine form over the loop variables, and
  stages anything that does not depend on the lane axis in workgroup
  memory once per reduce chunk: buffer reads (k/v tiles, the matmul `b`
  tile) and *inlined sub-computations* (the q·k scores) alike.
  Reduce-invariant per-thread reads (the q row) are hoisted to registers.
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

- Codegen beyond WGSL (MSL/CUDA emitters exist in `tang-expr` for scalar
  bodies; the loop-nest emitters are WGSL-only).
- Scatter gradients reduce over the whole source coordinate they gate on;
  a stride-2 read pays 2× in the backward pass. Fine for rope, not for a
  large gather.
- Non-affine indexing — KV-cache paging, MoE routing — needs a `Gather`
  view that reads an index tensor, which blocks some fusion.
- Placement: splitting an axis across devices is the same transform as
  tiling it across a workgroup, with an all-reduce as the combine. The
  machinery is here; the runtime is not.
- A real cost model. `schedule_auto` is three measured rules; `autotune`
  is the honest answer until there is one.
