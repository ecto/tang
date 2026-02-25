use crate::Scalar;
use core::ops::{Add, Sub, Mul, Div, Neg};

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
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
