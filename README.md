<p align="center">
  <img src="site/tang-mascot.svg" alt="tang mascot" width="120">
</p>

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
| **[`tang-gpu`](crates/tang-gpu)** | GPU compute & training via wgpu — fused kernels, matmul, backward passes, trainer |
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

## nalgebra Compatibility

tang provides drop-in compatibility aliases so you can migrate from nalgebra with minimal call-site changes.

### Type mapping

| nalgebra | tang / tang-la |
|----------|---------------|
| `Vector2<f64>` | `Vec2<f64>` |
| `Vector3<f64>` | `Vec3<f64>` |
| `Vector4<f64>` | `Vec4<f64>` |
| `Point3<f64>` | `Point3<f64>` |
| `Unit<Vector3<f64>>` | `Dir3<f64>` |
| `Matrix3<f64>` | `Mat3<f64>` |
| `Matrix4<f64>` | `Mat4<f64>` |
| `UnitQuaternion<f64>` | `Quat<f64>` |
| `DVector<f64>` | `DVec<f64>` |
| `DMatrix<f64>` | `DMat<f64>` |

### API compatibility

Both by-value and by-reference calling conventions work:

```rust
// nalgebra style (by-ref) — works
v.dot(&w);
v.cross(&w);

// tang style (by-value) — also works
v.dot(w);
v.cross(w);
```

Aliases provided for common nalgebra names:

| nalgebra | tang equivalent |
|----------|----------------|
| `Vec3::zeros()` | `Vec3::zero()` (+ `zeros()` alias) |
| `v.norm_squared()` | `v.norm_sq()` (+ `norm_squared()` alias) |
| `Unit::new_normalize(v)` | `Dir3::new(v)` (+ `new_normalize()` alias) |
| `dir.as_ref()` | `dir.as_ref()` (AsRef + Deref to Vec3) |
| `Mat3::from_diagonal(&v)` | `Mat3::diagonal(v)` (+ `from_diagonal(&v)` alias) |
| `m[(i,j)]` | Index for Mat3, Mat4, DMat (+ IndexMut for DMat) |
| `DMatrix::identity(n, n)` | `DMat::identity(n)` |
| `DVector::from_column_slice(s)` | `DVec::from_slice(s)` (+ `from_column_slice()` alias) |
| `DVector::from_iterator(n, it)` | `DVec::from_iterator(n, it)` |
| `DMatrix::from_iterator(r, c, it)` | `DMat::from_iterator(r, c, it)` |
| `DMatrix::from_row_slice(r, c, s)` | `DMat::from_row_slice(r, c, s)` |
| `m.symmetric_eigen()` | `m.symmetric_eigen()` (method on DMat) |
| `m.svd(true, true)` | `m.svd(true, true)` (method on DMat) |
| `svd.singular_values` | `svd.s` (+ `.singular_values()` accessor) |
| `svd.v_t` | `svd.vt` (+ `.v_t()` accessor) |
| `a.clone().lu().solve(&b)` | `a.clone().lu().solve(&b)` (DMatLu wrapper) |
| `m.try_inverse()` | `m.try_inverse()` (DMat, Mat3, Mat4) |

## Benchmarks

All benchmarks on Apple M-series, single-threaded. Run with `cargo bench -p tang-bench`.

### Geometry & physics primitives (f64 unless noted)

| Operation | tang | nalgebra | glam (f32) |
|-----------|------|----------|------------|
| vec3 dot | 2.2ns | 2.2ns | 1.6ns |
| vec3 cross | 1.8ns | 2.2ns | 1.9ns |
| vec3 normalize | 3.7ns | 3.5ns | 2.5ns |
| mat3 mul | 5.9ns | 6.9ns | — |
| mat4 mul | 11.0ns | 12.1ns | 5.5ns |
| mat4 inverse | 12.0ns | 16.0ns | 8.8ns |
| **quat rotate** | **2.5ns** | 2.6ns | 2.3ns |
| quat mul | 3.3ns | 3.3ns | 2.0ns |
| quat slerp | 8.1ns | 11.0ns | 6.3ns |

### Differentiable physics

tang's key advantage: the same code that runs your physics also gives you exact derivatives. Write your simulation once with generic `S: Scalar`, then swap in `Dual<f64>` to get gradients, Jacobians, and Hessians — no finite differences, no truncation error.

| Benchmark | tang AD | finite diff | notes |
|-----------|---------|-------------|-------|
| rigid body gradient (6 params) | 19ns | 18ns | exact vs ε-approximate |
| FK Jacobian (3-link arm, 3×3) | 278ns | 199ns | exact Jacobian, no tuning h |
| **LU solve derivative** | **81ns** | 167ns | **2x faster** — Dual flows through LU |
| Hessian Rosenbrock (2×2) | 77ns | 27ns | exact 2nd derivatives |

The speed comparison is secondary — the real win is that tang's derivatives are *exact* to machine precision. Finite differences require careful step-size tuning (too large → truncation error, too small → cancellation error) and break down for stiff systems. With tang, you just change the type parameter.

The LU solve benchmark highlights a capability nalgebra cannot match: because tang's decompositions are generic over `Scalar`, `Lu::<Dual<f64>>` gives you derivatives of the solution *through* the linear solve for free.

### Dense linear algebra (f64, tang vs nalgebra)

| Operation | n | tang | nalgebra | ratio |
|-----------|---|------|----------|-------|
| GEMM | 32 | 6.6µs | 1.5µs | 4.4x |
| | 128 | 250µs | 84µs | 3.0x |
| | 512 | 20ms | 4.9ms | 4.1x |
| **LU solve** | 32 | 3.6µs | 3.2µs | 1.1x |
| | 128 | 118µs | 107µs | 1.1x |
| | 512 | **9.8ms** | 8.6ms | 1.1x |
| Cholesky solve | 32 | 3.0µs | 2.6µs | 1.2x |
| | 128 | 170µs | 63µs | 2.7x |
| | 512 | 28ms | 3.7ms | 7.6x |
| QR | 32 | 5.7µs | 5.0µs | 1.1x |
| | 128 | 360µs | 211µs | 1.7x |
| | 512 | 31ms | 13ms | 2.4x |
| **Sym. Eigen** | 32 | 66µs | 28µs | 2.4x |
| | 128 | **4.8ms** | 942µs | 5.1x |

tang's dense LA is pure generic Rust (works with `Dual<f64>`, any `Scalar` impl). nalgebra dispatches to optimized BLAS/LAPACK-style routines for `f64`. The gap is expected and acceptable for tang's use case — when you need peak f64 throughput, enable the `faer` feature.

### Autodiff overhead

| Operation | plain f64 | Dual f64 | overhead |
|-----------|-----------|----------|----------|
| trig chain | 4.5ns | 8.2ns | 1.8x |
| vec3 chain | 3.7ns | 15.4ns | 4.2x |
| quat rotate | 15.0ns | 30.9ns | 2.1x |

### GPU training pipeline (tang-gpu, Apple M-series Metal)

| Operation | time |
|-----------|------|
| matmul 16x16 | 124µs |
| matmul 32x32 | 124µs |
| matmul 64x64 | 127µs |
| matmul 128x128 | 127µs |
| fused elementwise (a+b)² 4096 | 2.9ms |
| Linear forward [4,128]→[4,64] | 6.2ms |
| Linear backward [4,128]→[4,64] | 3.4ms |
| Sequential(2→8→1) fwd+bwd | 28ms |
| MSE loss (64 elements) | 8.4ms |
| XOR training step (4 samples) | 60ms |

Matmul is dispatch-bound at small sizes (constant ~124µs for 16-128). The training step includes forward, loss, backward, and Adam update for a 3-layer network.

## Examples

### The Quantum Poet

A character-level text generator trained on physics haikus. Demonstrates the full `tang-train` pipeline — dataset construction, sequential model, cross-entropy loss, Adam optimizer, and text generation.

```bash
cargo run --example quantum_poet -p tang-train
```

```
=== The Quantum Poet ===

corpus: 1491 chars, 1483 training samples, 8 parameters window
model: 21534 parameters

training...
  epoch   1: loss = 2.9453
  epoch  20: loss = 0.0916
  epoch 200: loss = 0.0207

--- seed: "quantum " ---
quantum fields vibrate below,
dimensions curl up and hide,
theory seeks the truth...
```

~21K parameters, trains in seconds on CPU. See [`crates/tang-train/examples/quantum_poet.rs`](crates/tang-train/examples/quantum_poet.rs).

## Development

```bash
cargo test --workspace
cargo test --workspace --all-features
```

## License

[MIT](LICENSE)
