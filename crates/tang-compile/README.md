# tang-compile

Graph optimization and compilation passes for tang

Part of [tang](https://github.com/ecto/tang).

## Tensor IR

`tang_compile::tensor` is a tensor-level IR with two primitives:

- **Views** — affine index maps. Broadcast, transpose, slice and stride
  compose and never touch memory.
- **Comprehensions** — `out[i…] = ⊕_{k…} body(x₀[view₀(i,k)], …)`, where
  `body` and the monoid's combine are scalar `tang-expr` graphs. Map,
  reduce and matmul are sugar.

What falls out:

- `substitute_maps` is the single fusion rule (cost-aware: a map is not
  recomputed inside a reduction unless it is cheap); `eliminate_common`
  merges identical comps.
- `stabilize_softmax` rewrites `max_j s` + `Σ_j g·exp(s−max)` into one
  online-softmax reduction: flash attention is derived, not written.
- `backward` differentiates a comprehension into comprehensions.
- `schedule_inline` / `schedule_split` / `schedule_tile(_2d)` are
  schedule decisions; `to_wgsl` lowers them to kernels (`tang-gpu` runs
  them). See `BENCHMARKS.md` for derived attention and matmul against the
  hand-written kernels.
- Dims are `Dim::Fixed(n)` or `Dim::Sym("seq")`. Build, rewrite and
  differentiate once with symbols; `bind` per size before scheduling,
  evaluating or lowering.
- `iota` gives a body its coordinates (masks, positional tables);
  `reshape` is a zero-cost alias so strided reads are pick views.
- `backward` inverts pick views by renaming coordinates and everything
  else (offset, stride, diagonal, broadcast) by an indicator scatter, so a
  whole transformer block — rmsnorm, rope, GQA attention, SwiGLU — is
  differentiable (see `tang-bench::ir_block`).
