use crate::{Scalar, Vec3};
use core::ops::{Add, Index, Mul, Neg, Sub};

/// 3x3 matrix, column-major storage.
///
/// Used for rotations, spatial inertia, and skew-symmetric (cross-product) matrices.
/// Stored as three column vectors for natural column access.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Mat3<S> {
    /// Column 0
    pub c0: Vec3<S>,
    /// Column 1
    pub c1: Vec3<S>,
    /// Column 2
    pub c2: Vec3<S>,
}

impl<S: Scalar> Mat3<S> {
    /// Construct from individual elements (row-major argument order for readability).
    /// ```text
    /// | m00 m01 m02 |
    /// | m10 m11 m12 |
    /// | m20 m21 m22 |
    /// ```
    #[inline]
    pub fn new(m00: S, m01: S, m02: S, m10: S, m11: S, m12: S, m20: S, m21: S, m22: S) -> Self {
        Self {
            c0: Vec3::new(m00, m10, m20),
            c1: Vec3::new(m01, m11, m21),
            c2: Vec3::new(m02, m12, m22),
        }
    }

    /// Construct from column vectors
    #[inline]
    pub fn from_cols(c0: Vec3<S>, c1: Vec3<S>, c2: Vec3<S>) -> Self {
        Self { c0, c1, c2 }
    }

    #[inline]
    pub fn zero() -> Self {
        Self::from_cols(Vec3::zero(), Vec3::zero(), Vec3::zero())
    }

    #[inline]
    pub fn identity() -> Self {
        Self::new(
            S::ONE,
            S::ZERO,
            S::ZERO,
            S::ZERO,
            S::ONE,
            S::ZERO,
            S::ZERO,
            S::ZERO,
            S::ONE,
        )
    }

    #[inline]
    pub fn diagonal(d: Vec3<S>) -> Self {
        Self::new(
            d.x,
            S::ZERO,
            S::ZERO,
            S::ZERO,
            d.y,
            S::ZERO,
            S::ZERO,
            S::ZERO,
            d.z,
        )
    }

    /// Alias for [`diagonal()`](Self::diagonal) accepting a reference (nalgebra compatibility).
    ///
    /// nalgebra's `Matrix3::from_diagonal(&v)` is equivalent to `Mat3::diagonal(v)`.
    #[inline]
    pub fn from_diagonal(d: &Vec3<S>) -> Self {
        Self::diagonal(*d)
    }

    /// Element access (row, col)
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> S {
        match col {
            0 => match row {
                0 => self.c0.x,
                1 => self.c0.y,
                _ => self.c0.z,
            },
            1 => match row {
                0 => self.c1.x,
                1 => self.c1.y,
                _ => self.c1.z,
            },
            _ => match row {
                0 => self.c2.x,
                1 => self.c2.y,
                _ => self.c2.z,
            },
        }
    }

    /// Column access
    #[inline]
    pub fn col(&self, i: usize) -> Vec3<S> {
        match i {
            0 => self.c0,
            1 => self.c1,
            _ => self.c2,
        }
    }

    /// Row access
    #[inline]
    pub fn row(&self, i: usize) -> Vec3<S> {
        Vec3::new(self.get(i, 0), self.get(i, 1), self.get(i, 2))
    }

    #[inline]
    pub fn transpose(&self) -> Self {
        Self::new(
            self.c0.x, self.c0.y, self.c0.z, self.c1.x, self.c1.y, self.c1.z, self.c2.x, self.c2.y,
            self.c2.z,
        )
    }

    #[inline]
    pub fn determinant(&self) -> S {
        self.c0.x * (self.c1.y * self.c2.z - self.c2.y * self.c1.z)
            - self.c1.x * (self.c0.y * self.c2.z - self.c2.y * self.c0.z)
            + self.c2.x * (self.c0.y * self.c1.z - self.c1.y * self.c0.z)
    }

    pub fn try_inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < S::EPSILON {
            return None;
        }
        let inv_det = det.recip();
        Some(Self::new(
            (self.c1.y * self.c2.z - self.c2.y * self.c1.z) * inv_det,
            (self.c2.x * self.c1.z - self.c1.x * self.c2.z) * inv_det,
            (self.c1.x * self.c2.y - self.c2.x * self.c1.y) * inv_det,
            (self.c2.y * self.c0.z - self.c0.y * self.c2.z) * inv_det,
            (self.c0.x * self.c2.z - self.c2.x * self.c0.z) * inv_det,
            (self.c2.x * self.c0.y - self.c0.x * self.c2.y) * inv_det,
            (self.c0.y * self.c1.z - self.c1.y * self.c0.z) * inv_det,
            (self.c1.x * self.c0.z - self.c0.x * self.c1.z) * inv_det,
            (self.c0.x * self.c1.y - self.c1.x * self.c0.y) * inv_det,
        ))
    }

    /// Matrix-vector product
    #[inline]
    pub fn mul_vec(&self, v: Vec3<S>) -> Vec3<S> {
        self.c0 * v.x + self.c1 * v.y + self.c2 * v.z
    }

    /// Matrix-matrix product
    #[inline]
    pub fn mul_mat(&self, rhs: &Mat3<S>) -> Mat3<S> {
        Mat3::from_cols(
            self.mul_vec(rhs.c0),
            self.mul_vec(rhs.c1),
            self.mul_vec(rhs.c2),
        )
    }

    /// Frobenius norm squared
    #[inline]
    pub fn norm_sq(&self) -> S {
        self.c0.norm_sq() + self.c1.norm_sq() + self.c2.norm_sq()
    }

    /// Trace
    #[inline]
    pub fn trace(&self) -> S {
        self.c0.x + self.c1.y + self.c2.z
    }

    /// Rotation matrix about X axis
    pub fn rotation_x(angle: S) -> Self {
        let (s, c) = angle.sin_cos();
        Self::new(S::ONE, S::ZERO, S::ZERO, S::ZERO, c, -s, S::ZERO, s, c)
    }

    /// Rotation matrix about Y axis
    pub fn rotation_y(angle: S) -> Self {
        let (s, c) = angle.sin_cos();
        Self::new(c, S::ZERO, s, S::ZERO, S::ONE, S::ZERO, -s, S::ZERO, c)
    }

    /// Rotation matrix about Z axis
    pub fn rotation_z(angle: S) -> Self {
        let (s, c) = angle.sin_cos();
        Self::new(c, -s, S::ZERO, s, c, S::ZERO, S::ZERO, S::ZERO, S::ONE)
    }

    /// Rotation matrix about an arbitrary axis (Rodrigues' formula)
    pub fn rotation_axis(axis: Vec3<S>, angle: S) -> Self {
        let (s, c) = angle.sin_cos();
        let t = S::ONE - c;
        let Vec3 { x, y, z } = axis;
        Self::new(
            t * x * x + c,
            t * x * y - s * z,
            t * x * z + s * y,
            t * x * y + s * z,
            t * y * y + c,
            t * y * z - s * x,
            t * x * z - s * y,
            t * y * z + s * x,
            t * z * z + c,
        )
    }
}

// nalgebra compatibility: index by (row, col) tuple.
impl<S: Scalar> Index<(usize, usize)> for Mat3<S> {
    type Output = S;
    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &S {
        match col {
            0 => match row {
                0 => &self.c0.x,
                1 => &self.c0.y,
                _ => &self.c0.z,
            },
            1 => match row {
                0 => &self.c1.x,
                1 => &self.c1.y,
                _ => &self.c1.z,
            },
            _ => match row {
                0 => &self.c2.x,
                1 => &self.c2.y,
                _ => &self.c2.z,
            },
        }
    }
}

impl<S: Scalar> Default for Mat3<S> {
    fn default() -> Self {
        Self::identity()
    }
}

impl<S: Scalar> Add for Mat3<S> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::from_cols(self.c0 + rhs.c0, self.c1 + rhs.c1, self.c2 + rhs.c2)
    }
}

impl<S: Scalar> Sub for Mat3<S> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::from_cols(self.c0 - rhs.c0, self.c1 - rhs.c1, self.c2 - rhs.c2)
    }
}

impl<S: Scalar> Neg for Mat3<S> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::from_cols(-self.c0, -self.c1, -self.c2)
    }
}

impl<S: Scalar> Mul<S> for Mat3<S> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: S) -> Self {
        Self::from_cols(self.c0 * rhs, self.c1 * rhs, self.c2 * rhs)
    }
}

// Mat3 * Vec3
impl<S: Scalar> Mul<Vec3<S>> for Mat3<S> {
    type Output = Vec3<S>;
    #[inline]
    fn mul(self, rhs: Vec3<S>) -> Vec3<S> {
        self.mul_vec(rhs)
    }
}

// Mat3 * Mat3
impl<S: Scalar> Mul for Mat3<S> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.mul_mat(&rhs)
    }
}

// nalgebra compatibility: Mat3 * &Vec3
impl<S: Scalar> Mul<&Vec3<S>> for Mat3<S> {
    type Output = Vec3<S>;
    #[inline]
    fn mul(self, rhs: &Vec3<S>) -> Vec3<S> {
        self.mul_vec(*rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity() {
        let m = Mat3::<f64>::identity();
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(m * v, v);
    }

    #[test]
    fn transpose() {
        let m = Mat3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
        let mt = m.transpose();
        assert_eq!(mt.get(0, 1), 4.0);
        assert_eq!(mt.get(1, 0), 2.0);
    }

    #[test]
    fn inverse() {
        let m = Mat3::new(1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0);
        let mi = m.try_inverse().unwrap();
        let prod = m * mi;
        let id = Mat3::<f64>::identity();
        for r in 0..3 {
            for c in 0..3 {
                assert!((prod.get(r, c) - id.get(r, c)).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn rotation_roundtrip() {
        let r = Mat3::rotation_z(std::f64::consts::FRAC_PI_2);
        let v = Vec3::new(1.0, 0.0, 0.0);
        let rotated = r * v;
        assert!((rotated.x).abs() < 1e-10);
        assert!((rotated.y - 1.0).abs() < 1e-10);
    }

    #[test]
    fn determinant() {
        let id = Mat3::<f64>::identity();
        assert!((id.determinant() - 1.0).abs() < 1e-10);
    }
}
