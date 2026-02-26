use crate::{DMat, DVec};
use alloc::vec::Vec;
use tang::Scalar;

/// LU decomposition with partial pivoting: PA = LU
pub struct Lu<S> {
    /// Combined L (lower, unit diagonal) and U (upper) in one matrix.
    lu: DMat<S>,
    /// Pivot indices: row i was swapped with row piv[i].
    piv: Vec<usize>,
    /// Number of row swaps (for determinant sign).
    swaps: usize,
}

impl<S: Scalar> Lu<S> {
    /// Compute LU decomposition of a square matrix.
    pub fn new(a: &DMat<S>) -> Option<Self> {
        assert!(a.is_square(), "LU: matrix must be square");
        let n = a.nrows();
        let mut lu = a.clone();
        let mut piv: Vec<usize> = (0..n).collect();
        let mut swaps = 0;

        for k in 0..n {
            // Find pivot: largest |a[i][k]| for i >= k
            let col_k = lu.col(k);
            let mut max_val = S::ZERO;
            let mut max_row = k;
            for i in k..n {
                let v = col_k[i].abs();
                if v > max_val {
                    max_val = v;
                    max_row = i;
                }
            }

            if max_val < S::EPSILON {
                return None; // Singular
            }

            if max_row != k {
                lu.swap_rows(k, max_row);
                piv.swap(k, max_row);
                swaps += 1;
            }

            let pivot_inv = lu.get(k, k).recip();

            // Compute factors for column k
            let col_k_mut = lu.col_mut(k);
            for i in (k + 1)..n {
                col_k_mut[i] = col_k_mut[i] * pivot_inv;
            }

            // Update submatrix: for each column j > k, col_j[i] -= factor[i] * col_j[k]
            let data = lu.data_mut();
            for j in (k + 1)..n {
                let row_k_val = data[j * n + k]; // lu[k, j]
                for i in (k + 1)..n {
                    let factor = data[k * n + i]; // lu[i, k] (the factor)
                    data[j * n + i] = data[j * n + i] - factor * row_k_val;
                }
            }
        }

        Some(Self { lu, piv, swaps })
    }

    /// Solve Ax = b.
    pub fn solve(&self, b: &DVec<S>) -> DVec<S> {
        let n = self.lu.nrows();
        assert_eq!(b.len(), n);

        // Apply pivot permutation
        let mut x = DVec::from_fn(n, |i| b[self.piv[i]]);

        // Forward substitution (L * y = Pb)
        for i in 1..n {
            let mut sum = x[i];
            for j in 0..i {
                sum = sum - self.lu.get(i, j) * x[j];
            }
            x[i] = sum;
        }

        // Back substitution (U * x = y)
        for i in (0..n).rev() {
            let mut sum = x[i];
            for j in (i + 1)..n {
                sum = sum - self.lu.get(i, j) * x[j];
            }
            x[i] = sum * self.lu.get(i, i).recip();
        }

        x
    }

    /// Solve AX = B (multiple right-hand sides).
    pub fn solve_mat(&self, b: &DMat<S>) -> DMat<S> {
        let ncols = b.ncols();
        let mut result = DMat::zeros(b.nrows(), ncols);
        for j in 0..ncols {
            let col = b.col_vec(j);
            let x = self.solve(&col);
            for i in 0..x.len() {
                result.set(i, j, x[i]);
            }
        }
        result
    }

    /// Determinant.
    pub fn det(&self) -> S {
        let n = self.lu.nrows();
        let mut d = if self.swaps % 2 == 0 { S::ONE } else { -S::ONE };
        for i in 0..n {
            d = d * self.lu.get(i, i);
        }
        d
    }

    /// Inverse (via solving A * A^-1 = I).
    pub fn inverse(&self) -> DMat<S> {
        let n = self.lu.nrows();
        self.solve_mat(&DMat::identity(n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_simple() {
        // [2 1] [x]   [5]    x=2, y=1
        // [1 3] [y] = [5]
        let a = DMat::from_fn(2, 2, |i, j| [[2.0, 1.0], [1.0, 3.0]][i][j]);
        let b = DVec::from_slice(&[5.0, 5.0]);
        let lu = Lu::new(&a).unwrap();
        let x = lu.solve(&b);
        assert!((x[0] - 2.0).abs() < 1e-10);
        assert!((x[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn determinant() {
        let a = DMat::from_fn(2, 2, |i, j| [[3.0, 7.0], [1.0, -4.0]][i][j]);
        let lu = Lu::new(&a).unwrap();
        assert!((lu.det() - (-19.0)).abs() < 1e-10);
    }

    #[test]
    fn inverse_roundtrip() {
        let a = DMat::from_fn(3, 3, |i, j| {
            [[2.0, 1.0, 1.0], [1.0, 3.0, 2.0], [1.0, 0.0, 0.0]][i][j]
        });
        let lu = Lu::new(&a).unwrap();
        let ainv = lu.inverse();
        let prod = a.mul_mat(&ainv);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (prod.get(i, j) - expected).abs() < 1e-10,
                    "mismatch at ({}, {}): {}",
                    i,
                    j,
                    prod.get(i, j)
                );
            }
        }
    }

    #[test]
    fn singular_returns_none() {
        let a = DMat::from_fn(2, 2, |i, j| [[1.0, 2.0], [2.0, 4.0]][i][j]);
        assert!(Lu::new(&a).is_none());
    }
}
