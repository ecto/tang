# tang-ad

Reverse-mode automatic differentiation. Record a tape, sweep backward, get gradients of scalar functions with many inputs — the shape that matters for optimization.

## How it works

```
Forward pass:              Backward pass:

  x₀ ──┐                    ∂L/∂x₀ ←──┐
        ├── f ── y               │      │
  x₁ ──┘       │            ∂L/∂x₁ ←──┤
                │                       │
                └── g ── L       ∂L/∂L = 1
                                    │
                    tape records →   sweeps ←
```

## Quickstart

```rust
use tang_ad::grad;

// gradient of a scalar function
let g = grad(
    |x| x[0]*x[0] + x[1]*x[1] + x[2]*x[2],
    &[1.0, 2.0, 3.0],
);
// g = [2.0, 4.0, 6.0]
```

## Full API

Five functions cover the common cases:

```rust
use tang_ad::{grad, jacobian_fwd, hessian, vjp, jvp};

// ∇f: gradient (reverse-mode, scalar output)
let g = grad(|x| x[0].sin() * x[1].cos(), &[1.0, 2.0]);

// J: Jacobian (forward-mode via Dual numbers)
let j = jacobian_fwd(|x| vec![x[0] + x[1], x[0] * x[1]], &[3.0, 4.0]);

// H: Hessian (forward-over-forward, Dual<Dual<f64>>)
let h = hessian(|x| x[0]*x[0]*x[1], &[1.0, 2.0]);

// vᵀJ: vector-Jacobian product (reverse-mode)
let vj = vjp(|x| vec![x[0] + x[1], x[0] * x[1]], &[3.0, 4.0], &[1.0, 0.0]);

// Jv: Jacobian-vector product (forward-mode)
let jv = jvp(|x| vec![x[0] + x[1], x[0] * x[1]], &[3.0, 4.0], &[1.0, 0.0]);
```

## Building the tape manually

For more control, build a `Tape` and use `Var` directly:

```rust
use tang_ad::{Tape, Var};

let tape = Tape::new();
let x = Var::new(&tape, 3.0);
let y = Var::new(&tape, 4.0);

let z = (x * x + y * y).sqrt();
let grads = z.backward();
// grads[0] = 3/5, grads[1] = 4/5
```

`Var` supports `sin`, `cos`, `exp`, `ln`, `sqrt`, `abs`, `tanh`, `powf`, and all arithmetic operators with mixed `Var`/`f64` operands.

## Mixed-mode AD

| Function | Mode | Best for |
|----------|------|----------|
| `grad` | Reverse | f: ℝⁿ → ℝ (loss functions) |
| `jacobian_fwd` | Forward (Dual) | f: ℝⁿ → ℝᵐ where m ≈ n |
| `hessian` | Forward-over-forward | Second derivatives |
| `vjp` | Reverse | Backpropagation through vector outputs |
| `jvp` | Forward | Directional derivatives |

## Design

- **`#![no_std]`** — uses `alloc` for heap
- **Tape is `Arc`-wrapped** — thread-safe reference counting
- **Index-based graph** — each `Var` is a position on the tape, no pointer chasing
- **Sparse backward** — skips zero adjoints during the sweep

## License

[MIT](../../LICENSE)
