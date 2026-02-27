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

## Design

- **Structural interning** — identical subexpressions always share the same `ExprId` (automatic CSE)
- **Thread-local graph** — `trace()` isolates graphs cleanly; `with_graph()` for manual control
- **Memoized differentiation** — `diff()` caches results to avoid redundant work
- **Fixpoint simplification** — rewrites until convergence: constant folding, identity/annihilation rules, cancellation (`x + (-x) → 0`, `x * (1/x) → 1`)
- **Dead code elimination** — `compile()` only emits reachable operations

## License

[MIT](../../LICENSE)
