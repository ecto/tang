use alloc::vec::Vec;
use tang::Scalar;
use tang_la::DVec;

/// Compressed Sparse Column matrix.
pub struct CscMatrix<S> {
    pub nrows: usize,
    pub ncols: usize,
    pub col_ptrs: Vec<usize>,    // length ncols + 1
    pub row_indices: Vec<usize>, // length nnz
    pub values: Vec<S>,          // length nnz
}

impl<S: Scalar> CscMatrix<S> {
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Sparse matrix-vector product: y = A * x
    pub fn spmv(&self, x: &DVec<S>) -> DVec<S> {
        assert_eq!(x.len(), self.ncols);
        let mut y = DVec::zeros(self.nrows);
        for j in 0..self.ncols {
            let xj = x[j];
            for k in self.col_ptrs[j]..self.col_ptrs[j + 1] {
                y[self.row_indices[k]] = y[self.row_indices[k]] + self.values[k] * xj;
            }
        }
        y
    }

    /// Get element (i, j).
    pub fn get(&self, row: usize, col: usize) -> S {
        let start = self.col_ptrs[col];
        let end = self.col_ptrs[col + 1];
        for k in start..end {
            if self.row_indices[k] == row {
                return self.values[k];
            }
        }
        S::ZERO
    }

    /// Convert from CSR.
    pub fn from_csr(csr: &super::CsrMatrix<S>) -> Self {
        let nrows = csr.nrows;
        let ncols = csr.ncols;
        let nnz = csr.nnz();

        let mut col_counts = alloc::vec![0usize; ncols + 1];
        for &c in &csr.col_indices {
            col_counts[c + 1] += 1;
        }
        for j in 1..=ncols {
            col_counts[j] += col_counts[j - 1];
        }

        let mut row_indices = alloc::vec![0usize; nnz];
        let mut values = alloc::vec![S::ZERO; nnz];
        let mut offsets = col_counts.clone();

        for i in 0..nrows {
            for k in csr.row_ptrs[i]..csr.row_ptrs[i + 1] {
                let c = csr.col_indices[k];
                let pos = offsets[c];
                row_indices[pos] = i;
                values[pos] = csr.values[k];
                offsets[c] += 1;
            }
        }

        Self {
            nrows,
            ncols,
            col_ptrs: col_counts,
            row_indices,
            values,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CooMatrix;

    #[test]
    fn csc_spmv() {
        let mut coo = CooMatrix::new(2, 3);
        coo.push(0, 0, 1.0);
        coo.push(0, 2, 2.0);
        coo.push(1, 1, 3.0);
        let csr = coo.to_csr();
        let csc = CscMatrix::from_csr(&csr);

        let x = DVec::from_slice(&[1.0, 2.0, 2.0]);
        let y = csc.spmv(&x);
        assert!((y[0] - 5.0).abs() < 1e-10);
        assert!((y[1] - 6.0).abs() < 1e-10);
    }
}
