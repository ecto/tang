use crate::Scalar;
use core::ops::{Add, Sub, Mul, Div, Neg, AddAssign, SubAssign, MulAssign};

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Vec4<S> {
    pub x: S,
    pub y: S,
    pub z: S,
    pub w: S,
}

impl<S: Scalar> Vec4<S> {
    #[inline]
    pub fn new(x: S, y: S, z: S, w: S) -> Self { Self { x, y, z, w } }

    #[inline]
    pub fn zero() -> Self { Self::new(S::ZERO, S::ZERO, S::ZERO, S::ZERO) }

    #[inline]
    pub fn splat(v: S) -> Self { Self::new(v, v, v, v) }

    #[inline]
    pub fn dot(self, rhs: Self) -> S {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
    }

    #[inline]
    pub fn norm_sq(self) -> S { self.dot(self) }

    #[inline]
    pub fn norm(self) -> S { self.norm_sq().sqrt() }

    /// Truncate to Vec3 (drop w)
    #[inline]
    pub fn truncate(self) -> crate::Vec3<S> {
        crate::Vec3::new(self.x, self.y, self.z)
    }

    /// Perspective divide: xyz / w
    #[inline]
    pub fn to_point3(self) -> crate::Point3<S> {
        let inv_w = self.w.recip();
        crate::Point3::new(self.x * inv_w, self.y * inv_w, self.z * inv_w)
    }
}

// nalgebra compatibility: `v.dot(&w)` works via Into.
impl<S: Scalar> From<&Vec4<S>> for Vec4<S> {
    #[inline]
    fn from(v: &Vec4<S>) -> Self { *v }
}

impl<S: Scalar> Default for Vec4<S> {
    fn default() -> Self { Self::zero() }
}

impl<S: Scalar> Add for Vec4<S> {
    type Output = Self;
    #[inline] fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z, self.w + rhs.w)
    }
}

impl<S: Scalar> Sub for Vec4<S> {
    type Output = Self;
    #[inline] fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z, self.w - rhs.w)
    }
}

impl<S: Scalar> Neg for Vec4<S> {
    type Output = Self;
    #[inline] fn neg(self) -> Self { Self::new(-self.x, -self.y, -self.z, -self.w) }
}

impl<S: Scalar> Mul<S> for Vec4<S> {
    type Output = Self;
    #[inline] fn mul(self, rhs: S) -> Self {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs, self.w * rhs)
    }
}

impl<S: Scalar> Div<S> for Vec4<S> {
    type Output = Self;
    #[inline] fn div(self, rhs: S) -> Self {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs, self.w / rhs)
    }
}

impl<S: Scalar> AddAssign for Vec4<S> {
    #[inline] fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x; self.y += rhs.y; self.z += rhs.z; self.w += rhs.w;
    }
}

impl<S: Scalar> SubAssign for Vec4<S> {
    #[inline] fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x; self.y -= rhs.y; self.z -= rhs.z; self.w -= rhs.w;
    }
}

impl<S: Scalar> MulAssign<S> for Vec4<S> {
    #[inline] fn mul_assign(&mut self, rhs: S) {
        self.x *= rhs; self.y *= rhs; self.z *= rhs; self.w *= rhs;
    }
}

impl<S: Scalar> Vec4<S> {
    #[inline]
    pub fn normalize(self) -> Self { self / self.norm() }

    #[inline]
    pub fn lerp(self, other: Self, t: S) -> Self {
        self * (S::ONE - t) + other * t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_product() {
        let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let b = Vec4::new(5.0, 6.0, 7.0, 8.0);
        assert_eq!(a.dot(b), 70.0); // 5+12+21+32
    }

    #[test]
    fn normalize() {
        let v = Vec4::new(1.0, 2.0, 2.0, 0.0);
        let n = v.normalize();
        assert!((n.norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn add_assign() {
        let mut a = Vec4::new(1.0, 2.0, 3.0, 4.0);
        a += Vec4::new(10.0, 20.0, 30.0, 40.0);
        assert_eq!(a, Vec4::new(11.0, 22.0, 33.0, 44.0));
    }

    #[test]
    fn truncate_and_extend() {
        let v4 = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let v3 = v4.truncate();
        assert_eq!(v3, crate::Vec3::new(1.0, 2.0, 3.0));
        let v4b = v3.extend(4.0);
        assert_eq!(v4, v4b);
    }

    #[test]
    fn perspective_divide() {
        let v = Vec4::new(2.0, 4.0, 6.0, 2.0);
        let p = v.to_point3();
        assert_eq!(p, crate::Point3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn neg() {
        let v = Vec4::new(1.0, -2.0, 3.0, -4.0);
        assert_eq!(-v, Vec4::new(-1.0, 2.0, -3.0, 4.0));
    }

    #[test]
    fn lerp() {
        let a = Vec4::new(0.0, 0.0, 0.0, 0.0);
        let b = Vec4::new(10.0, 20.0, 30.0, 40.0);
        let mid = a.lerp(b, 0.5);
        assert_eq!(mid, Vec4::new(5.0, 10.0, 15.0, 20.0));
    }
}
