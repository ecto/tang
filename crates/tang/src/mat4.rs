use crate::{Scalar, Vec3, Vec4, Point3, Mat3};
use core::ops::{Add, Sub, Mul, Neg};

/// 4x4 matrix, column-major storage.
///
/// Used for homogeneous transforms (affine: rotation + translation + scale).
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Mat4<S> {
    pub c0: Vec4<S>,
    pub c1: Vec4<S>,
    pub c2: Vec4<S>,
    pub c3: Vec4<S>,
}

impl<S: Scalar> Mat4<S> {
    /// Construct from elements in row-major argument order.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        m00: S, m01: S, m02: S, m03: S,
        m10: S, m11: S, m12: S, m13: S,
        m20: S, m21: S, m22: S, m23: S,
        m30: S, m31: S, m32: S, m33: S,
    ) -> Self {
        Self {
            c0: Vec4::new(m00, m10, m20, m30),
            c1: Vec4::new(m01, m11, m21, m31),
            c2: Vec4::new(m02, m12, m22, m32),
            c3: Vec4::new(m03, m13, m23, m33),
        }
    }

    #[inline]
    pub fn from_cols(c0: Vec4<S>, c1: Vec4<S>, c2: Vec4<S>, c3: Vec4<S>) -> Self {
        Self { c0, c1, c2, c3 }
    }

    #[inline]
    pub fn zero() -> Self {
        Self::from_cols(Vec4::zero(), Vec4::zero(), Vec4::zero(), Vec4::zero())
    }

    #[inline]
    pub fn identity() -> Self {
        Self::new(
            S::ONE,  S::ZERO, S::ZERO, S::ZERO,
            S::ZERO, S::ONE,  S::ZERO, S::ZERO,
            S::ZERO, S::ZERO, S::ONE,  S::ZERO,
            S::ZERO, S::ZERO, S::ZERO, S::ONE,
        )
    }

    /// Build from rotation (3x3) and translation
    pub fn from_rotation_translation(rot: Mat3<S>, trans: Vec3<S>) -> Self {
        Self::new(
            rot.c0.x, rot.c1.x, rot.c2.x, trans.x,
            rot.c0.y, rot.c1.y, rot.c2.y, trans.y,
            rot.c0.z, rot.c1.z, rot.c2.z, trans.z,
            S::ZERO,  S::ZERO,  S::ZERO,  S::ONE,
        )
    }

    /// Translation matrix
    pub fn translation(dx: S, dy: S, dz: S) -> Self {
        Self::new(
            S::ONE,  S::ZERO, S::ZERO, dx,
            S::ZERO, S::ONE,  S::ZERO, dy,
            S::ZERO, S::ZERO, S::ONE,  dz,
            S::ZERO, S::ZERO, S::ZERO, S::ONE,
        )
    }

    /// Non-uniform scale matrix
    pub fn scale(sx: S, sy: S, sz: S) -> Self {
        Self::new(
            sx,      S::ZERO, S::ZERO, S::ZERO,
            S::ZERO, sy,      S::ZERO, S::ZERO,
            S::ZERO, S::ZERO, sz,      S::ZERO,
            S::ZERO, S::ZERO, S::ZERO, S::ONE,
        )
    }

    /// Rotation about X axis
    pub fn rotation_x(angle: S) -> Self {
        Self::from_rotation_translation(Mat3::rotation_x(angle), Vec3::zero())
    }

    /// Rotation about Y axis
    pub fn rotation_y(angle: S) -> Self {
        Self::from_rotation_translation(Mat3::rotation_y(angle), Vec3::zero())
    }

    /// Rotation about Z axis
    pub fn rotation_z(angle: S) -> Self {
        Self::from_rotation_translation(Mat3::rotation_z(angle), Vec3::zero())
    }

    /// Rotation about arbitrary axis (Rodrigues' formula)
    pub fn rotation_axis(axis: Vec3<S>, angle: S) -> Self {
        Self::from_rotation_translation(Mat3::rotation_axis(axis, angle), Vec3::zero())
    }

    /// Element access (row, col)
    pub fn get(&self, row: usize, col: usize) -> S {
        let c = match col { 0 => &self.c0, 1 => &self.c1, 2 => &self.c2, _ => &self.c3 };
        match row { 0 => c.x, 1 => c.y, 2 => c.z, _ => c.w }
    }

    /// Extract the upper-left 3x3 submatrix
    #[inline]
    pub fn upper_left_3x3(&self) -> Mat3<S> {
        Mat3::from_cols(
            self.c0.truncate(),
            self.c1.truncate(),
            self.c2.truncate(),
        )
    }

    /// Extract the translation column
    #[inline]
    pub fn translation_vec(&self) -> Vec3<S> {
        self.c3.truncate()
    }

    #[inline]
    pub fn transpose(&self) -> Self {
        Self::new(
            self.c0.x, self.c0.y, self.c0.z, self.c0.w,
            self.c1.x, self.c1.y, self.c1.z, self.c1.w,
            self.c2.x, self.c2.y, self.c2.z, self.c2.w,
            self.c3.x, self.c3.y, self.c3.z, self.c3.w,
        )
    }

    /// Matrix-Vec4 product
    #[inline]
    pub fn mul_vec4(&self, v: Vec4<S>) -> Vec4<S> {
        self.c0 * v.x + self.c1 * v.y + self.c2 * v.z + self.c3 * v.w
    }

    /// Transform a point (w=1, includes translation)
    #[inline]
    pub fn transform_point(&self, p: Point3<S>) -> Point3<S> {
        let v = self.mul_vec4(p.to_homogeneous());
        Point3::new(v.x, v.y, v.z)
    }

    /// Transform a vector (w=0, ignores translation)
    #[inline]
    pub fn transform_vec(&self, v: Vec3<S>) -> Vec3<S> {
        let r = self.mul_vec4(v.extend(S::ZERO));
        Vec3::new(r.x, r.y, r.z)
    }

    /// Transform a normal (uses inverse transpose of 3x3)
    pub fn transform_normal(&self, n: Vec3<S>) -> Vec3<S> {
        let m3 = self.upper_left_3x3();
        match m3.try_inverse() {
            Some(inv) => inv.transpose().mul_vec(n),
            None => n,
        }
    }

    /// Matrix-matrix product
    pub fn mul_mat(&self, rhs: &Mat4<S>) -> Mat4<S> {
        Mat4::from_cols(
            self.mul_vec4(rhs.c0),
            self.mul_vec4(rhs.c1),
            self.mul_vec4(rhs.c2),
            self.mul_vec4(rhs.c3),
        )
    }

    /// 4x4 matrix inverse via cofactor expansion
    pub fn try_inverse(&self) -> Option<Self> {
        let m = |r, c| self.get(r, c);

        let s0 = m(0,0) * m(1,1) - m(1,0) * m(0,1);
        let s1 = m(0,0) * m(1,2) - m(1,0) * m(0,2);
        let s2 = m(0,0) * m(1,3) - m(1,0) * m(0,3);
        let s3 = m(0,1) * m(1,2) - m(1,1) * m(0,2);
        let s4 = m(0,1) * m(1,3) - m(1,1) * m(0,3);
        let s5 = m(0,2) * m(1,3) - m(1,2) * m(0,3);

        let c5 = m(2,2) * m(3,3) - m(3,2) * m(2,3);
        let c4 = m(2,1) * m(3,3) - m(3,1) * m(2,3);
        let c3 = m(2,1) * m(3,2) - m(3,1) * m(2,2);
        let c2 = m(2,0) * m(3,3) - m(3,0) * m(2,3);
        let c1 = m(2,0) * m(3,2) - m(3,0) * m(2,2);
        let c0 = m(2,0) * m(3,1) - m(3,0) * m(2,1);

        let det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
        if det.abs() < S::EPSILON { return None; }

        let inv_det = det.recip();
        Some(Self::new(
            ( m(1,1) * c5 - m(1,2) * c4 + m(1,3) * c3) * inv_det,
            (-m(0,1) * c5 + m(0,2) * c4 - m(0,3) * c3) * inv_det,
            ( m(3,1) * s5 - m(3,2) * s4 + m(3,3) * s3) * inv_det,
            (-m(2,1) * s5 + m(2,2) * s4 - m(2,3) * s3) * inv_det,
            (-m(1,0) * c5 + m(1,2) * c2 - m(1,3) * c1) * inv_det,
            ( m(0,0) * c5 - m(0,2) * c2 + m(0,3) * c1) * inv_det,
            (-m(3,0) * s5 + m(3,2) * s2 - m(3,3) * s1) * inv_det,
            ( m(2,0) * s5 - m(2,2) * s2 + m(2,3) * s1) * inv_det,
            ( m(1,0) * c4 - m(1,1) * c2 + m(1,3) * c0) * inv_det,
            (-m(0,0) * c4 + m(0,1) * c2 - m(0,3) * c0) * inv_det,
            ( m(3,0) * s4 - m(3,1) * s2 + m(3,3) * s0) * inv_det,
            (-m(2,0) * s4 + m(2,1) * s2 - m(2,3) * s0) * inv_det,
            (-m(1,0) * c3 + m(1,1) * c1 - m(1,2) * c0) * inv_det,
            ( m(0,0) * c3 - m(0,1) * c1 + m(0,2) * c0) * inv_det,
            (-m(3,0) * s3 + m(3,1) * s1 - m(3,2) * s0) * inv_det,
            ( m(2,0) * s3 - m(2,1) * s1 + m(2,2) * s0) * inv_det,
        ))
    }
}

impl<S: Scalar> Default for Mat4<S> {
    fn default() -> Self { Self::identity() }
}

impl<S: Scalar> Add for Mat4<S> {
    type Output = Self;
    #[inline] fn add(self, rhs: Self) -> Self {
        Self::from_cols(self.c0 + rhs.c0, self.c1 + rhs.c1, self.c2 + rhs.c2, self.c3 + rhs.c3)
    }
}

impl<S: Scalar> Sub for Mat4<S> {
    type Output = Self;
    #[inline] fn sub(self, rhs: Self) -> Self {
        Self::from_cols(self.c0 - rhs.c0, self.c1 - rhs.c1, self.c2 - rhs.c2, self.c3 - rhs.c3)
    }
}

impl<S: Scalar> Neg for Mat4<S> {
    type Output = Self;
    #[inline] fn neg(self) -> Self {
        Self::from_cols(-self.c0, -self.c1, -self.c2, -self.c3)
    }
}

impl<S: Scalar> Mul<S> for Mat4<S> {
    type Output = Self;
    #[inline] fn mul(self, rhs: S) -> Self {
        Self::from_cols(self.c0 * rhs, self.c1 * rhs, self.c2 * rhs, self.c3 * rhs)
    }
}

// Mat4 * Vec4
impl<S: Scalar> Mul<Vec4<S>> for Mat4<S> {
    type Output = Vec4<S>;
    #[inline] fn mul(self, rhs: Vec4<S>) -> Vec4<S> { self.mul_vec4(rhs) }
}

// Mat4 * Mat4
impl<S: Scalar> Mul for Mat4<S> {
    type Output = Self;
    #[inline] fn mul(self, rhs: Self) -> Self { self.mul_mat(&rhs) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_transform() {
        let m = Mat4::<f64>::identity();
        let p = Point3::new(1.0, 2.0, 3.0);
        assert_eq!(m.transform_point(p), p);
    }

    #[test]
    fn translation() {
        let m = Mat4::translation(10.0, 20.0, 30.0);
        let p = Point3::new(1.0, 2.0, 3.0);
        let result = m.transform_point(p);
        assert_eq!(result, Point3::new(11.0, 22.0, 33.0));
    }

    #[test]
    fn translation_ignores_vectors() {
        let m = Mat4::translation(10.0, 20.0, 30.0);
        let v = Vec3::new(1.0, 0.0, 0.0);
        assert_eq!(m.transform_vec(v), v);
    }

    #[test]
    fn inverse_roundtrip() {
        let m = Mat4::translation(1.0, 2.0, 3.0) * Mat4::rotation_z(0.5);
        let mi = m.try_inverse().unwrap();
        let prod = m * mi;
        let id = Mat4::<f64>::identity();
        for r in 0..4 {
            for c in 0..4 {
                assert!((prod.get(r, c) - id.get(r, c)).abs() < 1e-10,
                    "mismatch at ({}, {}): {} vs {}", r, c, prod.get(r, c), id.get(r, c));
            }
        }
    }

    #[test]
    fn compose() {
        let t = Mat4::translation(1.0, 0.0, 0.0);
        let r = Mat4::rotation_z(std::f64::consts::FRAC_PI_2);
        // Rotate then translate
        let m = t * r;
        let p = Point3::new(1.0, 0.0, 0.0);
        let result = m.transform_point(p);
        // Rotating (1,0,0) by 90° gives (0,1,0), then translating by (1,0,0) gives (1,1,0)
        assert!((result.x - 1.0).abs() < 1e-10);
        assert!((result.y - 1.0).abs() < 1e-10);
    }
}
