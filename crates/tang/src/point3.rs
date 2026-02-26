use crate::{Scalar, Vec3};
use core::ops::{Add, Sub, AddAssign, SubAssign};

/// A point in 3D space (distinct from Vec3 — points have position, vectors have direction).
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

    /// Alias for [`to_vec()`](Self::to_vec) as a field-like accessor (nalgebra compatibility).
    ///
    /// nalgebra's `Point3.coords` returns the underlying `Vector3`.
    #[inline]
    pub fn coords(self) -> Vec3<S> { self.to_vec() }

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

impl<S: Scalar> From<Vec3<S>> for Point3<S> {
    #[inline]
    fn from(v: Vec3<S>) -> Self { Self::from_vec(v) }
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

impl<S: Scalar> AddAssign<Vec3<S>> for Point3<S> {
    #[inline] fn add_assign(&mut self, rhs: Vec3<S>) {
        self.x += rhs.x; self.y += rhs.y; self.z += rhs.z;
    }
}

impl<S: Scalar> SubAssign<Vec3<S>> for Point3<S> {
    #[inline] fn sub_assign(&mut self, rhs: Vec3<S>) {
        self.x -= rhs.x; self.y -= rhs.y; self.z -= rhs.z;
    }
}

// nalgebra compatibility: reference-based operators.
impl<S: Scalar> Sub for &Point3<S> {
    type Output = Vec3<S>;
    #[inline] fn sub(self, rhs: &Point3<S>) -> Vec3<S> { *self - *rhs }
}

impl<S: Scalar> Add<&Vec3<S>> for &Point3<S> {
    type Output = Point3<S>;
    #[inline] fn add(self, rhs: &Vec3<S>) -> Point3<S> { *self + *rhs }
}

impl<S: Scalar> Sub<&Vec3<S>> for &Point3<S> {
    type Output = Point3<S>;
    #[inline] fn sub(self, rhs: &Vec3<S>) -> Point3<S> { *self - *rhs }
}

// Mixed ref/value: &Point3 - Point3
impl<S: Scalar> Sub<Point3<S>> for &Point3<S> {
    type Output = Vec3<S>;
    #[inline] fn sub(self, rhs: Point3<S>) -> Vec3<S> { *self - rhs }
}

// Mixed ref/value: Point3 - &Point3
impl<S: Scalar> Sub<&Point3<S>> for Point3<S> {
    type Output = Vec3<S>;
    #[inline] fn sub(self, rhs: &Point3<S>) -> Vec3<S> { self - *rhs }
}

// Mixed ref/value: &Point3 + Vec3
impl<S: Scalar> Add<Vec3<S>> for &Point3<S> {
    type Output = Point3<S>;
    #[inline] fn add(self, rhs: Vec3<S>) -> Point3<S> { *self + rhs }
}

// Mixed ref/value: &Point3 - Vec3
impl<S: Scalar> Sub<Vec3<S>> for &Point3<S> {
    type Output = Point3<S>;
    #[inline] fn sub(self, rhs: Vec3<S>) -> Point3<S> { *self - rhs }
}

impl<S: Scalar> core::fmt::Display for Point3<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance() {
        let a = Point3::new(0.0, 0.0, 0.0);
        let b = Point3::new(1.0, 2.0, 2.0);
        assert!((a.distance(b) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn midpoint() {
        let a = Point3::new(0.0, 0.0, 0.0);
        let b = Point3::new(10.0, 20.0, 30.0);
        assert_eq!(a.midpoint(b), Point3::new(5.0, 10.0, 15.0));
    }

    #[test]
    fn to_homogeneous() {
        let p = Point3::new(1.0, 2.0, 3.0);
        let h = p.to_homogeneous();
        assert_eq!(h, crate::Vec4::new(1.0, 2.0, 3.0, 1.0));
    }

    #[test]
    fn add_sub_assign() {
        let mut p = Point3::new(1.0, 2.0, 3.0);
        p += Vec3::new(10.0, 20.0, 30.0);
        assert_eq!(p, Point3::new(11.0, 22.0, 33.0));
        p -= Vec3::new(10.0, 20.0, 30.0);
        assert_eq!(p, Point3::new(1.0, 2.0, 3.0));
    }
}
