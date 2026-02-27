# tang-tensor

N-dimensional arrays with broadcasting, reductions, and all the operations you expect. The bridge between tang's linear algebra and its training pipeline.

## Basics

```rust
use tang_tensor::{Tensor, Shape};
use tang::Scalar;

let a = Tensor::from_fn(Shape::new(vec![2, 3]), |idx| {
    (idx[0] * 3 + idx[1]) as f64
});
// [[0, 1, 2],
//  [3, 4, 5]]

let b = Tensor::ones(Shape::new(vec![2, 3]));
let c = a.add(&b); // element-wise
```

## Broadcasting

NumPy-style shape alignment — dimensions are matched right-to-left, and size-1 dimensions stretch:

```
[2, 3] + [3]    → [2, 3]     ✓ row broadcast
[2, 3] + [2, 1] → [2, 3]     ✓ column broadcast
[4, 1, 3] + [1, 5, 3] → [4, 5, 3]  ✓ multi-axis
[2, 3] + [2, 4] → error      ✗ incompatible
```

```rust
let matrix = Tensor::from_fn(Shape::new(vec![2, 3]), |_| 1.0);
let row = Tensor::from_fn(Shape::new(vec![3]), |idx| idx[0] as f64);
let result = matrix.add(&row); // row added to each row of matrix
```

## Operations

| Category | Functions |
|----------|-----------|
| **Construct** | `zeros`, `ones`, `from_fn`, `scalar`, `from_slice` |
| **Arithmetic** | `add`, `sub`, `mul`, `div`, `neg`, `scale` |
| **Math** | `exp`, `ln`, `tanh`, `abs`, `sqrt`, `relu` |
| **Reduce** | `sum`, `mean`, `max`, `sum_axis`, `mean_axis` |
| **Shape** | `reshape`, `transpose`, `stack` |
| **LA** | `matmul` (delegates to tang-la) |
| **Custom** | `map(f)`, `zip_with(other, f)` |

## Conversions

```rust
use tang_la::{DVec, DMat};

let v: DVec<f64> = tensor_1d.to_dvec();
let m: DMat<f64> = tensor_2d.to_dmat();

let t = Tensor::from_dvec(&v);
let t = Tensor::from_dmat(&m);
```

## Design

- **`#![no_std]`** with `alloc`
- Generic over `S: Scalar`
- Row-major (C-order) contiguous layout with stride tracking
- `zip_with` handles all broadcasting logic
- Optional `gpu` feature for tang-gpu backend

## License

[MIT](../../LICENSE)
