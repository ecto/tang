# tang-optim

Gradient descent and beyond. Five optimizers, from simple SGD to second-order Newton, all operating on `DVec<f64>`.

## Optimizers

```
                    ┌──────────────────┐
                    │    Your loss     │
                    │   landscape      │
                    └────────┬─────────┘
                             │ ∇f
              ┌──────────────┼──────────────┐
              │              │              │
         first-order    quasi-Newton    second-order
              │              │              │
     ┌────┴────┐          L-BFGS      ┌────┴────┐
    SGD      Adam                  Newton     LM
```

| Optimizer | Best for | State |
|-----------|----------|-------|
| **SGD** | Simple problems, fine-tuning | Optional momentum buffer |
| **Adam** / **AdamW** | General-purpose, noisy gradients | First & second moment estimates |
| **L-BFGS** | Smooth, deterministic objectives | History of param/grad differences |
| **Newton** | Small systems, quadratic convergence | None (computes Hessian each step) |
| **Levenberg-Marquardt** | Nonlinear least-squares | Damping parameter λ |

## Usage

```rust
use tang_optim::Adam;
use tang_la::DVec;
use tang_ad::grad;

let mut opt = Adam::new(0.001);
let mut params = DVec::from_vec(vec![1.0, 2.0, 3.0]);

for _ in 0..1000 {
    let g = grad(|x| x[0]*x[0] + x[1]*x[1] + x[2]*x[2], params.as_slice());
    opt.step(&mut params, &g);
}
// params ≈ [0, 0, 0]
```

## Second-order methods

Newton and Levenberg-Marquardt take closures for the objective and gradient/Hessian:

```rust
use tang_optim::LevenbergMarquardt;
use tang_la::DVec;

let mut lm = LevenbergMarquardt::new();
let x0 = DVec::from_vec(vec![5.0, 5.0]);

let x = lm.minimize(
    x0,
    |x| { /* residuals */ },
    |x| { /* Jacobian */ },
);
```

## Line search

Armijo backtracking is available as a standalone function:

```rust
use tang_optim::armijo;

let alpha = armijo(|a| f(x + a * d), f0, slope, 1.0, 0.5, 1e-4);
```

## Design

- **`#![no_std]`** with `alloc`
- Lazy initialization — optimizer state allocated on first `.step()`
- Direct `DVec` mutation — params updated in-place
- No trait abstraction — concrete types, simple API

## License

[MIT](../../LICENSE)
