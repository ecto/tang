use crate::CsrMatrix;
use alloc::vec::Vec;
use tang::Scalar;

/// Coordinate (triplet) format for assembling sparse matrices.
///
/// Good for incremental construction, then convert to CSR for computation.
pub struct CooMatrix<S> {
    pub nrows: usize,
    pub ncols: usize,
    pub rows: Vec<usize>,
    pub cols: Vec<usize>,
    pub vals: Vec<S>,
}

impl<S: Scalar> CooMatrix<S> {
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            rows: Vec::new(),
            cols: Vec::new(),
            vals: Vec::new(),
        }
    }

    pub fn with_capacity(nrows: usize, ncols: usize, nnz: usize) -> Self {
        Self {
            nrows,
            ncols,
            rows: Vec::with_capacity(nnz),
            cols: Vec::with_capacity(nnz),
            vals: Vec::with_capacity(nnz),
        }
    }

    /// Add a triplet (row, col, value). Duplicates are summed during conversion.
    pub fn push(&mut self, row: usize, col: usize, val: S) {
        assert!(row < self.nrows && col < self.ncols);
        self.rows.push(row);
        self.cols.push(col);
        self.vals.push(val);
    }

    pub fn nnz(&self) -> usize {
        self.rows.len()
    }

    /// Convert to CSR format (summing duplicate entries).
    pub fn to_csr(&self) -> CsrMatrix<S> {
        let mut row_counts = alloc::vec![0usize; self.nrows + 1];
        for &r in &self.rows {
            row_counts[r + 1] += 1;
        }
        // Prefix sum
        for i in 1..=self.nrows {
            row_counts[i] += row_counts[i - 1];
        }
        let nnz = row_counts[self.nrows];
        let mut col_indices = alloc::vec![0usize; nnz];
        let mut values = alloc::vec![S::ZERO; nnz];
        let mut offsets = row_counts.clone();

        for k in 0..self.rows.len() {
            let r = self.rows[k];
            let pos = offsets[r];
            col_indices[pos] = self.cols[k];
            values[pos] = self.vals[k];
            offsets[r] += 1;
        }

        // Sort each row by column and merge duplicates
        let row_ptrs = row_counts;
        for i in 0..self.nrows {
            let start = row_ptrs[i];
            let end = row_ptrs[i + 1];
            // Insertion sort (rows are typically short)
            for j in (start + 1)..end {
                let mut k = j;
                while k > start && col_indices[k] < col_indices[k - 1] {
                    col_indices.swap(k, k - 1);
                    values.swap(k, k - 1);
                    k -= 1;
                }
            }
        }

        // Merge duplicates
        let mut new_col = Vec::with_capacity(nnz);
        let mut new_val = Vec::with_capacity(nnz);
        let mut new_ptrs = alloc::vec![0usize; self.nrows + 1];

        for i in 0..self.nrows {
            let start = row_ptrs[i];
            let end = row_ptrs[i + 1];
            let mut j = start;
            while j < end {
                let c = col_indices[j];
                let mut v = values[j];
                j += 1;
                while j < end && col_indices[j] == c {
                    v += values[j];
                    j += 1;
                }
                new_col.push(c);
                new_val.push(v);
            }
            new_ptrs[i + 1] = new_col.len();
        }

        CsrMatrix {
            nrows: self.nrows,
            ncols: self.ncols,
            row_ptrs: new_ptrs,
            col_indices: new_col,
            values: new_val,
        }
    }
}
