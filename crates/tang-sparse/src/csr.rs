use alloc::vec::Vec;
use tang::Scalar;
use tang_la::DVec;

/// Compressed Sparse Row matrix.
pub struct CsrMatrix<S> {
    pub nrows: usize,
    pub ncols: usize,
    pub row_ptrs: Vec<usize>,    // length nrows + 1
    pub col_indices: Vec<usize>, // length nnz
    pub values: Vec<S>,          // length nnz
}

impl<S: Scalar> CsrMatrix<S> {
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Sparse matrix-vector product: y += A * x
    pub fn spmv_add(&self, x: &DVec<S>, y: &mut DVec<S>) {
        assert_eq!(x.len(), self.ncols);
        assert_eq!(y.len(), self.nrows);
        for i in 0..self.nrows {
            let start = self.row_ptrs[i];
            let end = self.row_ptrs[i + 1];
            let mut sum = S::ZERO;
            for k in start..end {
                sum += self.values[k] * x[self.col_indices[k]];
            }
            y[i] = y[i] + sum;
        }
    }

    /// Sparse matrix-vector product: y = A * x
    pub fn spmv(&self, x: &DVec<S>) -> DVec<S> {
        let mut y = DVec::zeros(self.nrows);
        self.spmv_add(x, &mut y);
        y
    }

    /// Get element (i, j). O(nnz_row) lookup.
    pub fn get(&self, row: usize, col: usize) -> S {
        let start = self.row_ptrs[row];
        let end = self.row_ptrs[row + 1];
        for k in start..end {
            if self.col_indices[k] == col {
                return self.values[k];
            }
        }
        S::ZERO
    }

    /// Transpose to CSR (builds a new CSR of A^T).
    pub fn transpose(&self) -> Self {
        let mut row_counts = alloc::vec![0usize; self.ncols + 1];
        for &c in &self.col_indices {
            row_counts[c + 1] += 1;
        }
        for i in 1..=self.ncols {
            row_counts[i] += row_counts[i - 1];
        }
        let nnz = self.nnz();
        let mut col_indices = alloc::vec![0usize; nnz];
        let mut values = alloc::vec![S::ZERO; nnz];
        let mut offsets = row_counts.clone();
        for i in 0..self.nrows {
            for k in self.row_ptrs[i]..self.row_ptrs[i + 1] {
                let c = self.col_indices[k];
                let pos = offsets[c];
                col_indices[pos] = i;
                values[pos] = self.values[k];
                offsets[c] += 1;
            }
        }
        CsrMatrix {
            nrows: self.ncols,
            ncols: self.nrows,
            row_ptrs: row_counts,
            col_indices,
            values,
        }
    }

    /// Convert to dense matrix.
    pub fn to_dense(&self) -> tang_la::DMat<S> {
        tang_la::DMat::from_fn(self.nrows, self.ncols, |i, j| self.get(i, j))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CooMatrix;

    #[test]
    fn spmv() {
        // [[1 0 2]  * [1]   [5]
        //  [0 3 0]]   [2] = [6]
        //              [2]
        let mut coo = CooMatrix::new(2, 3);
        coo.push(0, 0, 1.0);
        coo.push(0, 2, 2.0);
        coo.push(1, 1, 3.0);
        let csr = coo.to_csr();

        let x = DVec::from_slice(&[1.0, 2.0, 2.0]);
        let y = csr.spmv(&x);
        assert!((y[0] - 5.0).abs() < 1e-10);
        assert!((y[1] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn transpose() {
        let mut coo = CooMatrix::new(2, 3);
        coo.push(0, 0, 1.0);
        coo.push(0, 2, 2.0);
        coo.push(1, 1, 3.0);
        let csr = coo.to_csr();
        let csrt = csr.transpose();
        assert_eq!(csrt.nrows, 3);
        assert_eq!(csrt.ncols, 2);
        assert!((csrt.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((csrt.get(2, 0) - 2.0).abs() < 1e-10);
        assert!((csrt.get(1, 1) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn duplicate_entries_summed() {
        let mut coo = CooMatrix::new(2, 2);
        coo.push(0, 0, 1.0);
        coo.push(0, 0, 2.0); // duplicate
        coo.push(1, 1, 3.0);
        let csr = coo.to_csr();
        assert!((csr.get(0, 0) - 3.0).abs() < 1e-10);
        assert_eq!(csr.nnz(), 2); // merged
    }
}
