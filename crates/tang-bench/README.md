# tang-bench

Benchmarks for the tang ecosystem. Measures everything from `Vec3::dot` to GPU training steps, with nalgebra and glam as baselines.

## Run

```bash
cargo bench -p tang-bench
cargo bench -p tang-bench -- "vec3"       # filter
cargo bench -p tang-bench -- "gpu"        # GPU only
```

## Benchmark suites

### `fixed_size` — geometry primitives

Vec3, Mat3, Mat4, Quat operations compared against nalgebra (f64) and glam (f32):

| Operation | tang | nalgebra | glam (f32) |
|-----------|------|----------|------------|
| vec3 dot | 2.2ns | 2.2ns | 1.6ns |
| vec3 cross | 1.8ns | 2.2ns | 1.9ns |
| mat4 mul | 11.0ns | 12.1ns | 5.5ns |
| quat slerp | 8.1ns | 11.0ns | 6.3ns |

### `dense_la` — linear algebra

GEMM, LU, Cholesky, QR, SVD, eigendecomposition at various matrix sizes.

### `autodiff` — differentiation overhead

Forward-mode (Dual), reverse-mode (tape), and finite differences compared.

### `differentiable` — physics applications

Rigid body gradients, FK Jacobians, differentiable LU solve, Rosenbrock Hessian.

### `expr_graph` — symbolic pipeline

Expression graph construction, symbolic differentiation, simplification, compilation, WGSL generation.

### `gpu_training` — Metal/Vulkan/DX12

Matmul, fused kernels, linear layers, sequential forward/backward, XOR training, MSE loss.

## Helpers

The crate also provides random data generators for writing your own benchmarks:

```rust
use tang_bench::*;

let vecs = random_tang_vec3f64(1000);
let mats = random_dmat(128, 128);
let spd = random_spd_dmat(64); // symmetric positive definite
```

## License

[MIT](../../LICENSE)
