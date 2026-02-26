use tang::Scalar;
use crate::DVec;
use core::ops::{Add, Sub, Mul, Neg, Index, IndexMut};
use alloc::vec::Vec;

/// Heap-allocated column-major matrix.
///
/// Element (row, col) is stored at `data[col * nrows + row]`.
#[derive(Clone, Debug, PartialEq)]
pub struct DMat<S> {
    data: Vec<S>,
    nrows: usize,
    ncols: usize,
}

impl<S: Scalar> DMat<S> {
    /// Create from raw column-major data.
    pub fn from_raw(nrows: usize, ncols: usize, data: Vec<S>) -> Self {
        assert_eq!(data.len(), nrows * ncols, "DMat: data length mismatch");
        Self { data, nrows, ncols }
    }

    /// Create from a function.
    pub fn from_fn(nrows: usize, ncols: usize, f: impl Fn(usize, usize) -> S) -> Self {
        let mut data = Vec::with_capacity(nrows * ncols);
        for j in 0..ncols {
            for i in 0..nrows {
                data.push(f(i, j));
            }
        }
        Self { data, nrows, ncols }
    }

    /// Zero matrix.
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        Self { data: alloc::vec![S::ZERO; nrows * ncols], nrows, ncols }
    }

    /// Identity matrix.
    pub fn identity(n: usize) -> Self {
        Self::from_fn(n, n, |i, j| if i == j { S::ONE } else { S::ZERO })
    }

    /// Create from an iterator in column-major order (nalgebra compatibility).
    ///
    /// nalgebra's `DMatrix::from_iterator(nrows, ncols, iter)` fills column-major.
    pub fn from_iterator(nrows: usize, ncols: usize, iter: impl IntoIterator<Item = S>) -> Self {
        let data: Vec<S> = iter.into_iter().take(nrows * ncols).collect();
        assert_eq!(data.len(), nrows * ncols, "DMat::from_iterator: iterator yielded fewer than expected");
        Self::from_raw(nrows, ncols, data)
    }

    /// Create from row-major data slice.
    pub fn from_row_slice(nrows: usize, ncols: usize, data: &[S]) -> Self {
        assert_eq!(data.len(), nrows * ncols, "DMat: data length mismatch");
        Self::from_fn(nrows, ncols, |i, j| data[i * ncols + j])
    }

    /// Diagonal matrix from a vector.
    pub fn from_diagonal(diag: &DVec<S>) -> Self {
        let n = diag.len();
        Self::from_fn(n, n, |i, j| if i == j { diag[i] } else { S::ZERO })
    }

    #[inline]
    pub fn nrows(&self) -> usize { self.nrows }

    #[inline]
    pub fn ncols(&self) -> usize { self.ncols }

    /// Element access (row, col).
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> S {
        self.data[col * self.nrows + row]
    }

    /// Mutable element access.
    #[inline]
    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut S {
        &mut self.data[col * self.nrows + row]
    }

    /// Set element.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, val: S) {
        self.data[col * self.nrows + row] = val;
    }

    /// Raw column-major data.
    #[inline]
    pub fn as_slice(&self) -> &[S] { &self.data }

    /// Column slice.
    pub fn col(&self, j: usize) -> &[S] {
        let start = j * self.nrows;
        &self.data[start..start + self.nrows]
    }

    /// Extract column as DVec.
    pub fn col_vec(&self, j: usize) -> DVec<S> {
        DVec::from_slice(self.col(j))
    }

    /// Extract row as DVec.
    pub fn row_vec(&self, i: usize) -> DVec<S> {
        DVec::from_fn(self.ncols, |j| self.get(i, j))
    }

    /// Extract diagonal.
    pub fn diagonal(&self) -> DVec<S> {
        let n = self.nrows.min(self.ncols);
        DVec::from_fn(n, |i| self.get(i, i))
    }

    /// Transpose.
    pub fn transpose(&self) -> Self {
        Self::from_fn(self.ncols, self.nrows, |i, j| self.get(j, i))
    }

    /// Matrix-vector product: y = A * x.
    pub fn mul_vec(&self, x: &DVec<S>) -> DVec<S> {
        assert_eq!(self.ncols, x.len(), "DMat mul_vec: dimension mismatch");
        let mut y = DVec::zeros(self.nrows);
        for j in 0..self.ncols {
            let xj = x[j];
            for i in 0..self.nrows {
                y[i] += self.get(i, j) * xj;
            }
        }
        y
    }

    /// Matrix-matrix product: C = A * B.
    pub fn mul_mat(&self, rhs: &DMat<S>) -> DMat<S> {
        assert_eq!(self.ncols, rhs.nrows, "DMat mul_mat: dimension mismatch");
        let mut c = DMat::zeros(self.nrows, rhs.ncols);
        for j in 0..rhs.ncols {
            for k in 0..self.ncols {
                let b_kj = rhs.get(k, j);
                for i in 0..self.nrows {
                    *c.get_mut(i, j) = c.get(i, j) + self.get(i, k) * b_kj;
                }
            }
        }
        c
    }

    /// Frobenius norm squared.
    pub fn norm_sq(&self) -> S {
        let mut s = S::ZERO;
        for &x in &self.data {
            s += x * x;
        }
        s
    }

    /// Frobenius norm.
    pub fn norm(&self) -> S {
        self.norm_sq().sqrt()
    }

    /// Trace (sum of diagonal).
    pub fn trace(&self) -> S {
        let n = self.nrows.min(self.ncols);
        let mut s = S::ZERO;
        for i in 0..n {
            s += self.get(i, i);
        }
        s
    }

    /// Scale all elements.
    pub fn scale(&self, s: S) -> Self {
        Self::from_fn(self.nrows, self.ncols, |i, j| self.get(i, j) * s)
    }

    /// Is this matrix square?
    #[inline]
    pub fn is_square(&self) -> bool { self.nrows == self.ncols }

    /// Swap two rows.
    pub fn swap_rows(&mut self, a: usize, b: usize) {
        if a == b { return; }
        for j in 0..self.ncols {
            let va = self.get(a, j);
            let vb = self.get(b, j);
            self.set(a, j, vb);
            self.set(b, j, va);
        }
    }

    /// Extract a submatrix.
    pub fn submatrix(&self, row_start: usize, col_start: usize, nrows: usize, ncols: usize) -> Self {
        Self::from_fn(nrows, ncols, |i, j| self.get(row_start + i, col_start + j))
    }

    // -- nalgebra compatibility --

    /// Alias for [`col_vec()`](Self::col_vec) (nalgebra compatibility).
    ///
    /// nalgebra uses `.column(j).into_owned()`.
    pub fn column(&self, j: usize) -> DVec<S> { self.col_vec(j) }

    /// Compute eigendecomposition (nalgebra compatibility).
    ///
    /// nalgebra uses `h.symmetric_eigen()`. tang-la equivalent: `SymmetricEigen::new(&h)`.
    pub fn symmetric_eigen(&self) -> crate::SymmetricEigen<S> {
        crate::SymmetricEigen::new(self)
    }

    /// Compute SVD (nalgebra compatibility).
    ///
    /// nalgebra uses `m.svd(true, true)`. The bool args are ignored since
    /// tang-la always computes both U and V.
    pub fn svd(&self, _compute_u: bool, _compute_v: bool) -> crate::Svd<S> {
        crate::Svd::new(self)
    }

    /// Compute the matrix inverse via LU decomposition (nalgebra compatibility).
    ///
    /// nalgebra uses `m.try_inverse()`.
    pub fn try_inverse(&self) -> Option<Self> {
        assert!(self.is_square(), "DMat::try_inverse: not square");
        let n = self.nrows;
        let lu = crate::Lu::new(self)?;
        let mut inv = DMat::zeros(n, n);
        for j in 0..n {
            let mut e = DVec::zeros(n);
            e[j] = S::ONE;
            let col = lu.solve(&e);
            for i in 0..n {
                inv.set(i, j, col[i]);
            }
        }
        Some(inv)
    }

    /// Compute LU decomposition and solve (nalgebra compatibility).
    ///
    /// nalgebra uses `a.clone().lu().solve(&b)` returning `Option<DVec>`.
    pub fn lu(self) -> DMatLu<S> { DMatLu(self) }
}

/// Wrapper for nalgebra-compatible LU chaining: `a.clone().lu().solve(&b)`.
pub struct DMatLu<S>(DMat<S>);

impl<S: Scalar> DMatLu<S> {
    /// Solve Ax = b, returning None if singular.
    pub fn solve(&self, b: &DVec<S>) -> Option<DVec<S>> {
        crate::Lu::new(&self.0).map(|lu| lu.solve(b))
    }
}

impl<S: Scalar> Index<(usize, usize)> for DMat<S> {
    type Output = S;
    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &S {
        &self.data[col * self.nrows + row]
    }
}

impl<S: Scalar> IndexMut<(usize, usize)> for DMat<S> {
    #[inline]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut S {
        &mut self.data[col * self.nrows + row]
    }
}

impl<S: Scalar> Add for &DMat<S> {
    type Output = DMat<S>;
    fn add(self, rhs: &DMat<S>) -> DMat<S> {
        assert_eq!(self.nrows, rhs.nrows);
        assert_eq!(self.ncols, rhs.ncols);
        DMat::from_fn(self.nrows, self.ncols, |i, j| self.get(i, j) + rhs.get(i, j))
    }
}

impl<S: Scalar> Sub for &DMat<S> {
    type Output = DMat<S>;
    fn sub(self, rhs: &DMat<S>) -> DMat<S> {
        assert_eq!(self.nrows, rhs.nrows);
        assert_eq!(self.ncols, rhs.ncols);
        DMat::from_fn(self.nrows, self.ncols, |i, j| self.get(i, j) - rhs.get(i, j))
    }
}

impl<S: Scalar> Neg for &DMat<S> {
    type Output = DMat<S>;
    fn neg(self) -> DMat<S> {
        DMat::from_fn(self.nrows, self.ncols, |i, j| -self.get(i, j))
    }
}

impl<S: Scalar> Mul<&DVec<S>> for &DMat<S> {
    type Output = DVec<S>;
    fn mul(self, rhs: &DVec<S>) -> DVec<S> { self.mul_vec(rhs) }
}

impl<S: Scalar> Mul for &DMat<S> {
    type Output = DMat<S>;
    fn mul(self, rhs: &DMat<S>) -> DMat<S> { self.mul_mat(rhs) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_mul() {
        let i = DMat::<f64>::identity(3);
        let x = DVec::from_slice(&[1.0, 2.0, 3.0]);
        let y = i.mul_vec(&x);
        assert_eq!(y[0], 1.0);
        assert_eq!(y[1], 2.0);
        assert_eq!(y[2], 3.0);
    }

    #[test]
    fn mat_mul() {
        let a = DMat::from_fn(2, 3, |i, j| (i * 3 + j + 1) as f64);
        let b = DMat::from_fn(3, 2, |i, j| (i * 2 + j + 1) as f64);
        let c = a.mul_mat(&b);
        assert_eq!(c.nrows(), 2);
        assert_eq!(c.ncols(), 2);
        // [1 2 3] * [1 2]   = [22 28]
        // [4 5 6]   [3 4]     [49 64]
        //           [5 6]
        assert_eq!(c.get(0, 0), 22.0);
        assert_eq!(c.get(0, 1), 28.0);
        assert_eq!(c.get(1, 0), 49.0);
        assert_eq!(c.get(1, 1), 64.0);
    }

    #[test]
    fn transpose() {
        let m = DMat::from_fn(2, 3, |i, j| (i * 3 + j) as f64);
        let mt = m.transpose();
        assert_eq!(mt.nrows(), 3);
        assert_eq!(mt.ncols(), 2);
        assert_eq!(mt.get(0, 0), 0.0);
        assert_eq!(mt.get(0, 1), 3.0);
        assert_eq!(mt.get(1, 0), 1.0);
    }

    #[test]
    fn trace() {
        let m = DMat::from_fn(3, 3, |i, j| if i == j { (i + 1) as f64 } else { 0.0 });
        assert_eq!(m.trace(), 6.0);
    }

    #[test]
    fn diagonal() {
        let m = DMat::from_fn(3, 3, |i, j| (i * 3 + j) as f64);
        let d = m.diagonal();
        assert_eq!(d[0], 0.0);
        assert_eq!(d[1], 4.0);
        assert_eq!(d[2], 8.0);
    }
}
