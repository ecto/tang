# tang-la

Dense linear algebra on the heap. Dynamic vectors, dynamic matrices, and every decomposition you need — all generic over `Scalar`, so your LU solve can propagate dual-number gradients.

## Decompositions

```
        A
       / \
      /   \
    LU    QR ──── least squares
    │      │
  solve  solve
  det
  inv

        A (symmetric)
       / \
      /   \
  Cholesky  Eigen ── eigenvalues λ
    │               eigenvectors V
  solve             A = V diag(λ) Vᵀ

        A (any shape)
        │
       SVD ── singular values σ
               pseudoinverse A⁺
               rank
               A = U Σ Vᵀ
```

## Usage

```rust
use tang_la::{DVec, DMat, Svd, Lu};

// build a system
let a = DMat::from_fn(3, 3, |i, j| if i == j { 4.0 } else { 1.0 });
let b = DVec::from_vec(vec![1.0, 2.0, 3.0]);

// solve via LU
let x = Lu::new(a.clone()).solve(&b);

// or decompose via SVD
let svd = Svd::new(&a);
let rank = svd.rank(1e-10);
let pinv = svd.pseudoinverse(1e-10);
```

## The differentiable advantage

Because everything is generic over `S: Scalar`, you can solve a linear system *and get derivatives of the solution through the solve*:

```rust
use tang::Dual;
use tang_la::{DMat, DVec, Lu};

// Dual<f64> flows through LU factorization
let a = DMat::<Dual<f64>>::from_fn(2, 2, |i, j| /* ... */);
let b = DVec::<Dual<f64>>::from_vec(vec![/* ... */]);
let x = Lu::new(a).solve(&b);
// x[i].dual contains ∂x[i]/∂(whatever you seeded)
```

nalgebra can't do this — its decompositions are specialized for `f64`.

## Algorithms

| Decomposition | Method | Complexity |
|---------------|--------|------------|
| LU | Partial pivoting | O(n³) |
| Cholesky | L·Lᵀ | O(n³/3) |
| QR | Householder reflections | O(2n³/3) |
| SVD | One-sided Jacobi | O(n³) |
| Symmetric Eigen | Householder → implicit QR + Wilkinson shifts | O(n³) |

A branchless Jacobi eigendecomposition is also available for symbolic expression tracing (no data-dependent branches).

## Performance

Pure generic Rust — works with any `Scalar`. When you need peak `f64` throughput, enable the `faer` feature for world-class BLAS-level performance.

## Features

| Feature | Default | Purpose |
|---------|---------|---------|
| `std` | yes | Standard library |
| `faer` | no | High-performance f64/f32 backend |

## License

[MIT](../../LICENSE)
