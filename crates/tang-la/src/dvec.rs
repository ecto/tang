use tang::Scalar;
use core::ops::{Add, Sub, Mul, Neg, Index, IndexMut, AddAssign, SubAssign, MulAssign};
use alloc::vec::Vec;

/// Heap-allocated vector of scalars.
#[derive(Clone, Debug, PartialEq)]
pub struct DVec<S> {
    data: Vec<S>,
}

impl<S: Scalar> DVec<S> {
    /// Create from raw data.
    #[inline]
    pub fn from_vec(data: Vec<S>) -> Self {
        Self { data }
    }

    /// Create a zero vector of given length.
    pub fn zeros(n: usize) -> Self {
        Self { data: alloc::vec![S::ZERO; n] }
    }

    /// Create from a function.
    pub fn from_fn(n: usize, f: impl Fn(usize) -> S) -> Self {
        Self { data: (0..n).map(f).collect() }
    }

    /// Create from a slice.
    pub fn from_slice(s: &[S]) -> Self {
        Self { data: s.to_vec() }
    }

    /// Create from an iterator, taking exactly `n` elements.
    pub fn from_iterator(n: usize, iter: impl IntoIterator<Item = S>) -> Self {
        let data: Vec<S> = iter.into_iter().take(n).collect();
        assert_eq!(data.len(), n, "DVec::from_iterator: iterator yielded fewer than {n} elements");
        Self { data }
    }

    /// Alias for [`from_slice()`](Self::from_slice) (nalgebra compatibility).
    ///
    /// nalgebra's `DVector::from_column_slice(s)` is equivalent to `DVec::from_slice(s)`.
    #[inline]
    pub fn from_column_slice(data: &[S]) -> Self { Self::from_slice(data) }

    #[inline]
    pub fn len(&self) -> usize { self.data.len() }

    #[inline]
    pub fn is_empty(&self) -> bool { self.data.is_empty() }

    #[inline]
    pub fn as_slice(&self) -> &[S] { &self.data }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [S] { &mut self.data }

    #[inline]
    pub fn into_vec(self) -> Vec<S> { self.data }

    /// Dot product.
    pub fn dot(&self, other: &DVec<S>) -> S {
        assert_eq!(self.len(), other.len(), "DVec dot: length mismatch");
        let mut sum = S::ZERO;
        for i in 0..self.len() {
            sum += self.data[i] * other.data[i];
        }
        sum
    }

    /// Euclidean norm.
    pub fn norm(&self) -> S {
        self.dot(self).sqrt()
    }

    /// Squared norm.
    pub fn norm_sq(&self) -> S {
        self.dot(self)
    }

    /// Normalize in-place, return the original norm.
    pub fn normalize(&mut self) -> S {
        let n = self.norm();
        if n > S::EPSILON {
            let inv = n.recip();
            for x in &mut self.data {
                *x = *x * inv;
            }
        }
        n
    }

    /// Return a normalized copy.
    pub fn normalized(&self) -> Self {
        let mut v = self.clone();
        v.normalize();
        v
    }

    /// Scale all elements.
    pub fn scale(&mut self, s: S) {
        for x in &mut self.data {
            *x = *x * s;
        }
    }

    /// Axpy: self += a * x
    pub fn axpy(&mut self, a: S, x: &DVec<S>) {
        assert_eq!(self.len(), x.len());
        for i in 0..self.len() {
            self.data[i] += a * x.data[i];
        }
    }

    /// Sum of all elements.
    pub fn sum(&self) -> S {
        let mut s = S::ZERO;
        for &x in &self.data {
            s += x;
        }
        s
    }

    /// Max absolute value.
    pub fn amax(&self) -> S {
        let mut m = S::ZERO;
        for &x in &self.data {
            m = m.max(x.abs());
        }
        m
    }

    /// Iterator over elements.
    pub fn iter(&self) -> core::slice::Iter<'_, S> {
        self.data.iter()
    }
}

impl<S: Scalar> Index<usize> for DVec<S> {
    type Output = S;
    #[inline]
    fn index(&self, i: usize) -> &S { &self.data[i] }
}

impl<S: Scalar> IndexMut<usize> for DVec<S> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut S { &mut self.data[i] }
}

impl<S: Scalar> Add for &DVec<S> {
    type Output = DVec<S>;
    fn add(self, rhs: &DVec<S>) -> DVec<S> {
        assert_eq!(self.len(), rhs.len());
        DVec::from_fn(self.len(), |i| self[i] + rhs[i])
    }
}

impl<S: Scalar> Sub for &DVec<S> {
    type Output = DVec<S>;
    fn sub(self, rhs: &DVec<S>) -> DVec<S> {
        assert_eq!(self.len(), rhs.len());
        DVec::from_fn(self.len(), |i| self[i] - rhs[i])
    }
}

impl<S: Scalar> Neg for &DVec<S> {
    type Output = DVec<S>;
    fn neg(self) -> DVec<S> {
        DVec::from_fn(self.len(), |i| -self[i])
    }
}

impl<S: Scalar> Mul<S> for &DVec<S> {
    type Output = DVec<S>;
    fn mul(self, rhs: S) -> DVec<S> {
        DVec::from_fn(self.len(), |i| self[i] * rhs)
    }
}

impl<S: Scalar> AddAssign<&DVec<S>> for DVec<S> {
    fn add_assign(&mut self, rhs: &DVec<S>) {
        assert_eq!(self.len(), rhs.len());
        for i in 0..self.len() {
            self.data[i] += rhs.data[i];
        }
    }
}

impl<S: Scalar> SubAssign<&DVec<S>> for DVec<S> {
    fn sub_assign(&mut self, rhs: &DVec<S>) {
        assert_eq!(self.len(), rhs.len());
        for i in 0..self.len() {
            self.data[i] -= rhs.data[i];
        }
    }
}

impl<S: Scalar> MulAssign<S> for DVec<S> {
    fn mul_assign(&mut self, rhs: S) {
        self.scale(rhs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_product() {
        let a = DVec::from_slice(&[1.0, 2.0, 3.0]);
        let b = DVec::from_slice(&[4.0, 5.0, 6.0]);
        assert_eq!(a.dot(&b), 32.0);
    }

    #[test]
    fn norm() {
        let v = DVec::from_slice(&[3.0, 4.0]);
        assert!((v.norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn normalize() {
        let mut v = DVec::from_slice(&[3.0, 4.0]);
        v.normalize();
        assert!((v.norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn arithmetic() {
        let a = DVec::from_slice(&[1.0, 2.0]);
        let b = DVec::from_slice(&[3.0, 4.0]);
        let sum = &a + &b;
        assert_eq!(sum[0], 4.0);
        assert_eq!(sum[1], 6.0);
        let diff = &a - &b;
        assert_eq!(diff[0], -2.0);
    }

    #[test]
    fn axpy() {
        let mut y = DVec::from_slice(&[1.0, 2.0]);
        let x = DVec::from_slice(&[3.0, 4.0]);
        y.axpy(2.0, &x);
        assert_eq!(y[0], 7.0);
        assert_eq!(y[1], 10.0);
    }
}
