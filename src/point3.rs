use crate::{Scalar, Vec3};
use core::ops::{Add, Sub};

/// A point in 3D space (distinct from Vec3 — points have position, vectors have direction).
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct Point3<S> {
    pub x: S,
    pub y: S,
    pub z: S,
}

impl<S: Scalar> Point3<S> {
    #[inline]
    pub fn new(x: S, y: S, z: S) -> Self { Self { x, y, z } }

    #[inline]
    pub fn origin() -> Self { Self::new(S::ZERO, S::ZERO, S::ZERO) }

    #[inline]
    pub fn to_vec(self) -> Vec3<S> { Vec3::new(self.x, self.y, self.z) }

    #[inline]
    pub fn from_vec(v: Vec3<S>) -> Self { Self::new(v.x, v.y, v.z) }

    #[inline]
    pub fn distance(self, other: Self) -> S { (other - self).norm() }

    #[inline]
    pub fn distance_sq(self, other: Self) -> S { (other - self).norm_sq() }

    #[inline]
    pub fn lerp(self, other: Self, t: S) -> Self {
        Self::from_vec(self.to_vec().lerp(other.to_vec(), t))
    }

    #[inline]
    pub fn midpoint(self, other: Self) -> Self {
        self.lerp(other, S::HALF)
    }

    /// Extend to homogeneous coordinates (w=1)
    #[inline]
    pub fn to_homogeneous(self) -> crate::Vec4<S> {
        crate::Vec4::new(self.x, self.y, self.z, S::ONE)
    }
}

impl<S: Scalar> Default for Point3<S> {
    fn default() -> Self { Self::origin() }
}

// Point - Point = Vec
impl<S: Scalar> Sub for Point3<S> {
    type Output = Vec3<S>;
    #[inline] fn sub(self, rhs: Self) -> Vec3<S> {
        Vec3::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

// Point + Vec = Point
impl<S: Scalar> Add<Vec3<S>> for Point3<S> {
    type Output = Self;
    #[inline] fn add(self, rhs: Vec3<S>) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

// Point - Vec = Point
impl<S: Scalar> Sub<Vec3<S>> for Point3<S> {
    type Output = Self;
    #[inline] fn sub(self, rhs: Vec3<S>) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl<S: Scalar> core::fmt::Display for Point3<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}
