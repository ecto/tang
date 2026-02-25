# tang

Math for physical reality. The shared foundation beneath geometry kernels, physics engines, and the full design→simulate→manufacture→deploy loop.

```rust
use tang::*;
use tang_la::{DVec, DMat, Svd};
use tang_ad::{Tape, grad};
use tang_optim::Adam;

// Dual numbers give you exact derivatives for free
let x = Dual::new(2.0, 1.0);
let y = (x * x).sin(); // y.dual = cos(x²) · 2x

// Reverse-mode AD for large parameter spaces
let loss = |x: &[f64]| x[0]*x[0] + x[1]*x[1] + x[2]*x[2];
let g = grad(loss, &[1.0, 2.0, 3.0]); // [2, 4, 6]

// Dense linear algebra — LU, SVD, Cholesky, QR, Eigen
let a = DMat::from_fn(3, 3, |i, j| if i == j { 2.0 } else { -1.0 });
let svd = Svd::new(&a);

// The same Scalar trait flows through everything
let q = Quat::axis_angle(Dir3::Z, core::f64::consts::FRAC_PI_2);
let v = q.rotate(Vec3::new(1.0, 0.0, 0.0)); // ≈ (0, 1, 0)
```

## Crates

| Crate | Description |
|-------|-------------|
| **[`tang`](crates/tang)** | Core types — `Vec2/3/4`, `Mat3/4`, `Quat`, `Transform`, `Dual<S>`, spatial algebra |
| **[`tang-la`](crates/tang-la)** | Dynamic linear algebra — `DVec`, `DMat`, LU, SVD, Cholesky, QR, eigendecomposition |
| **[`tang-ad`](crates/tang-ad)** | Reverse-mode autodiff — tape, `grad`, `jacobian`, `hessian`, VJP, JVP |
| **[`tang-sparse`](crates/tang-sparse)** | Sparse matrices — CSR, CSC, COO, SpMV |
| **[`tang-gpu`](crates/tang-gpu)** | GPU compute via wgpu — SpMV, batch ops, tensor kernels |
| **[`tang-optim`](crates/tang-optim)** | Optimizers — SGD, Adam/AdamW, L-BFGS, Newton, Levenberg-Marquardt |
| **[`tang-tensor`](crates/tang-tensor)** | N-d arrays — broadcasting, slicing, matmul, reductions |
| **[`tang-train`](crates/tang-train)** | Training — `Module` trait, layers, loss functions |

## Architecture

```
tang/
├── crates/
│   ├── tang/            # Scalar, Vec, Mat, Quat, Transform, Dual, spatial
│   ├── tang-la/         # DVec, DMat, LU, SVD, Cholesky, QR, Eigen
│   ├── tang-ad/         # Reverse-mode AD tape
│   ├── tang-sparse/     # CSR, CSC, COO
│   ├── tang-gpu/        # wgpu compute backend
│   ├── tang-optim/      # Gradient and second-order optimizers
│   ├── tang-tensor/     # N-d arrays with broadcasting
│   └── tang-train/      # Modules, layers, loss functions
└── benches/
```

### Dependency Graph

```
tang ← tang-la ← tang-sparse
          ↑           ↑
       tang-ad     tang-gpu
          ↑           ↑
       tang-tensor ───┘
          ↑
       tang-optim
          ↑
       tang-train
```

## Design

**One `Scalar` trait.** `f32`, `f64`, and `Dual<S>` all implement `Scalar`. Write your physics once, get exact derivatives by swapping the type parameter. Forward-mode for small systems, reverse-mode for large ones.

**`no_std` throughout.** Every crate is `#![no_std]` with `alloc`. Use tang on embedded, in WASM, wherever.

**No heavyweight dependencies.** Core types are hand-rolled `#[repr(C)]` with optional `bytemuck` and `serde` support. Dense LA is native Rust, generic over `Scalar`. An optional `faer` feature enables world-class f64 performance.

**Physics-native ML.** The same types that run your constraint solver and physics engine also train your neural nets. `tang-tensor` → `tang-ad` → `tang-optim` → `tang-train` is a complete differentiable programming stack.

## Quick Start

```bash
cargo add tang tang-la tang-ad
```

```rust
use tang::{Vec3, Quat, Dual, Scalar};
use tang_la::{DVec, DMat, Lu};
use tang_ad::grad;
```

## Development

```bash
cargo test --workspace        # 141 tests
cargo test --workspace --all-features
```

## License

[MIT](LICENSE)
