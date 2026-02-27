# tang-sparse

Sparse matrices. Three formats, convert between them, multiply against dense vectors.

## Formats

```
COO (coordinate)          CSR (compressed row)       CSC (compressed column)

  (0,0) 1.0                row_ptrs: [0, 2, 3, 5]    col_ptrs: [0, 2, 3, 4]
  (0,2) 2.0                col_idx:  [0, 2, 1, 0, 2]  row_idx:  [0, 2, 1, 0, 2]
  (1,1) 3.0                values:   [1, 2, 3, 4, 5]  values:   [1, 4, 3, 2, 5]
  (2,0) 4.0
  (2,2) 5.0              ┌ 1  ·  2 ┐
                          │ ·  3  · │
  assembly format →       └ 4  ·  5 ┘  → computation format
```

## Usage

Build with COO, convert to CSR for computation:

```rust
use tang_sparse::{CooMatrix, CsrMatrix};
use tang_la::DVec;

let mut coo = CooMatrix::new(3, 3);
coo.push(0, 0, 1.0);
coo.push(0, 2, 2.0);
coo.push(1, 1, 3.0);
coo.push(2, 0, 4.0);
coo.push(2, 2, 5.0);

let csr = coo.to_csr();

// sparse matrix-vector product
let x = DVec::from_vec(vec![1.0, 1.0, 1.0]);
let y = csr.spmv(&x); // [3.0, 3.0, 9.0]
```

## API

| Type | Key methods |
|------|-------------|
| `CooMatrix<S>` | `new`, `with_capacity`, `push`, `to_csr` |
| `CsrMatrix<S>` | `spmv`, `spmv_add`, `get`, `transpose`, `to_dense` |
| `CscMatrix<S>` | `spmv`, `get`, `from_csr` |

Duplicate entries at the same position are automatically summed during COO → CSR conversion.

## Design

- **`#![no_std]`** with `alloc`
- Generic over `S: Scalar` — works with `f32`, `f64`, `Dual<f64>`
- COO for assembly, CSR/CSC for computation
- Direct field access on all structs for advanced usage

## License

[MIT](../../LICENSE)
