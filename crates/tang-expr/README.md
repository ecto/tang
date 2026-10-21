# tang-expr

Symbolic expression graphs. Trace Rust math into a DAG, differentiate it, simplify it, compile it to native closures or WGSL compute shaders.

## The pipeline

```
  Rust code           trace           symbolic
  with ExprId   ─────────────►   expression graph
                                       │
                          ┌────────────┼────────────┐
                          │            │            │
                        diff       simplify      deps
                          │            │            │
                     derivatives   fewer nodes   sparsity
                          │            │         bitmask
                          └────────────┼────────────┘
                                       │
                          ┌────────────┼────────────┐
                          │            │            │
                       compile      eval        to_wgsl
                          │            │            │
                     Box<dyn Fn>    f64 value    WGSL kernel
                      (CPU)                      (GPU)
```

## Quickstart

```rust
use tang_expr::{trace, ExprId};

// trace a function into a graph
let (graph, y) = trace(|g| {
    let x = ExprId::var(g, 0);
    let y = ExprId::var(g, 1);
    x * x + y * y
});

// evaluate
let result = graph.eval(y, &[3.0, 4.0]); // 25.0

// differentiate
let mut graph = graph;
let dy_dx = graph.diff(y, ExprId::var(&graph, 0));
let gradient = graph.eval(dy_dx, &[3.0, 4.0]); // 6.0

// simplify
let simplified = graph.simplify(dy_dx);

// compile to native Rust
let f = graph.compile(y);
assert_eq!(f(&[3.0, 4.0]), 25.0);

// compile to WGSL for GPU
let kernel = graph.to_wgsl(&[y], 2);
```

## 9 RISC primitives

All of mathematics reduces to these operations:

```
Add   Mul   Neg   Recip   Sqrt   Sin   Atan2   Exp2   Log2
```

Plus `Var` (input), `Lit` (constant), and `Select` (branchless ternary).

`ExprId` implements the full `Scalar` trait, so `sin`, `cos`, `tan`, `exp`, `ln`, `min`, `max`, `pow`, `atan2` etc. all work through decomposition into the 9 primitives. You trace normal tang math and get a graph for free.

## Sparsity analysis

Know which outputs depend on which inputs without evaluating:

```rust
let sparsity = graph.jacobian_sparsity(&outputs, n_vars);
// sparsity[i] is a u64 bitmask: bit j set means output i depends on var j
```

Supports up to 64 input variables.

## Integer dialect (`tang_expr::int`)

A separate graph (`IntGraph`) for exact 32-bit integer kernels, such as
entropy decoders that must match a CPU reference bit for bit. The f64 RISC
graph above is untouched.

- **Types:** `i32`, `u32` and `bool`. `i8`/`u8` exist only as buffer
  storage (`Elem`), widened on load and packed 4 per `u32` word.
- **Ops:** wrapping `+ − ×`, `MulHi` (so `mul_wide` gives the exact 64-bit
  product as `(lo, hi)`), `& | ^ ! <<`, and `>>` (arithmetic on `i32`).
  Shift amounts are taken mod 32. Also `min`/`max`/`clamp`, `div`/`rem`
  (WGSL semantics), `clz`, `== != < <=`, `select`, `bitcast`, and
  read-only `load`.
- **Scratch buffers:** `buffer_rw(name, elem, len, base)` declares a small
  fixed-length read-write array (in WGSL, `name[base + i]` with `base` a host
  expression, e.g. a workgroup array slice). `load_rw` and `store` thread an
  effect token (`token()`, and each `store` returns the next one), so data
  dependencies order the effects. Scratch loads and stores are never inlined
  or moved: every backend runs them in the one shared schedule order, so
  their results agree across backends for any graph. Out-of-range loads read
  0 and out-of-range stores are dropped. tang-compress uses this for the
  per-lane adaptive probability maps of its learned rANS coder.
- **Loops:** `while_loop(init, |g, s| (cond, next))` is structured SSA, like
  MLIR `scf.while`, with `for_range` as sugar. A loop's states are
  `State(l, k)` nodes and its results are `LoopOut(l, k)` nodes. Loops nest,
  and each node belongs to the innermost loop whose state it reads, so loop
  invariants are hoisted by construction. Mixing the states of two sibling
  loops is rejected.
- **Contract:** nothing is reassociated or contracted. The only rewrite is
  folding all-constant nodes, which uses the interpreter's own op functions.
- **Lowerings,** all from one schedule:
  - `Program`: register bytecode, the CPU reference.
  - `to_wgsl_body` / `to_wgsl_fn`: WGSL. `MulHi` uses 16-bit-limb helpers in
    `WGSL_PRELUDE`, because WGSL has no 64-bit integers.
  - `to_rust`: Rust source.
- **Emission:** a single-use node is inlined into its user when both are in
  the same scope. Every other node becomes a `let`, and loop states become
  `var`s that are snapshotted at the top of each iteration.
- **Tests:** every op on both types against Rust's own semantics, and loops
  (nested, data-dependent trip count, parallel state update, loads) against
  plain Rust. The emitted WGSL is validated with naga 25, and the emitted
  Rust is compiled with `rustc` and its output compared with the interpreter.

**First user.** tang-compress builds its learned TCG3 MLP as an `IntGraph`
(`lmodel3::Net32::graph`), with the stream's weights as constants. The GPU
kernel's `mlp_gen` is `to_wgsl_body` of that graph, and `TC_CPU_IR=1` makes
the CPU decoder evaluate the same graph through `Program`.

**Not yet covered**, but needed to generate the whole rANS lane-decode
kernel:
- Stores and scatter into storage buffers (only lane-local scratch
  buffers exist).
- Barriers and `workgroupUniformLoad`.
- Side effects ordered with loops, for example a data-dependent word fetch
  that advances a stream pointer. This can be expressed today as loop state
  plus `load`, but there is no write-back.
- Kernel entry points and bindings.
- A C lowering, which would need unsigned casts to avoid signed-overflow UB.

## Integer dialect (0.2)

`tang_expr::int::IntGraph` is a separate, bit-exact 32-bit integer graph:
`i32`/`u32`/`bool`, `i8`/`u8` storage, wrapping arithmetic, exact
`mul_wide`, and structured `while` loops with invariant hoisting. The CPU
interpreter, the generated WGSL and the generated Rust compute the same bits.

```rust
use tang_expr::int::{IType, IntGraph};

// sum of i*i for i in 0..n
let mut g = IntGraph::new();
let n = g.param(0, IType::U32);
let zero = g.u32(0);
let s = g.for_range(n, &[zero], |g, i, s| {
    let sq = g.mul(i, i);
    vec![g.add(s[0], sq)]
});
assert_eq!(g.eval(&s, &[10], &[]), vec![285]);
let wgsl = g.to_wgsl_fn("sum_sq", &s); // or g.to_rust("sum_sq", &s)
```

See `examples/int_dialect.rs`.

## Design

- **Structural interning** — identical subexpressions always share the same `ExprId` (automatic CSE)
- **Thread-local graph** — `trace()` isolates graphs cleanly; `with_graph()` for manual control
- **Memoized differentiation** — `diff()` caches results to avoid redundant work
- **Fixpoint simplification** — rewrites until convergence: constant folding, identity/annihilation rules, cancellation (`x + (-x) → 0`, `x * (1/x) → 1`)
- **Dead code elimination** — `compile()` only emits reachable operations

## License

[MIT](../../LICENSE)
