use crate::{Scalar, Vec2};
use core::ops::{Add, Sub, AddAssign, SubAssign};

/// A point in 2D space (distinct from Vec2 — points have position, vectors have direction).
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Point2<S> {
    pub x: S,
    pub y: S,
}

impl<S: Scalar> Point2<S> {
    #[inline]
    pub fn new(x: S, y: S) -> Self { Self { x, y } }

    #[inline]
    pub fn origin() -> Self { Self::new(S::ZERO, S::ZERO) }

    #[inline]
    pub fn to_vec(self) -> Vec2<S> { Vec2::new(self.x, self.y) }

    #[inline]
    pub fn from_vec(v: Vec2<S>) -> Self { Self::new(v.x, v.y) }

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
}

impl<S: Scalar> Default for Point2<S> {
    fn default() -> Self { Self::origin() }
}

// Point - Point = Vec
impl<S: Scalar> Sub for Point2<S> {
    type Output = Vec2<S>;
    #[inline] fn sub(self, rhs: Self) -> Vec2<S> {
        Vec2::new(self.x - rhs.x, self.y - rhs.y)
    }
}

// Point + Vec = Point
impl<S: Scalar> Add<Vec2<S>> for Point2<S> {
    type Output = Self;
    #[inline] fn add(self, rhs: Vec2<S>) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

// Point - Vec = Point
impl<S: Scalar> Sub<Vec2<S>> for Point2<S> {
    type Output = Self;
    #[inline] fn sub(self, rhs: Vec2<S>) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl<S: Scalar> AddAssign<Vec2<S>> for Point2<S> {
    #[inline] fn add_assign(&mut self, rhs: Vec2<S>) { self.x += rhs.x; self.y += rhs.y; }
}

impl<S: Scalar> SubAssign<Vec2<S>> for Point2<S> {
    #[inline] fn sub_assign(&mut self, rhs: Vec2<S>) { self.x -= rhs.x; self.y -= rhs.y; }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance() {
        let a = Point2::new(0.0, 0.0);
        let b = Point2::new(3.0, 4.0);
        assert!((a.distance(b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn midpoint() {
        let a = Point2::new(0.0, 0.0);
        let b = Point2::new(10.0, 20.0);
        assert_eq!(a.midpoint(b), Point2::new(5.0, 10.0));
    }

    #[test]
    fn point_vec_arithmetic() {
        let p = Point2::new(1.0, 2.0);
        let v = Vec2::new(10.0, 20.0);
        assert_eq!(p + v, Point2::new(11.0, 22.0));
        assert_eq!(p - v, Point2::new(-9.0, -18.0));
    }

    #[test]
    fn point_diff() {
        let a = Point2::new(1.0, 2.0);
        let b = Point2::new(4.0, 6.0);
        assert_eq!(b - a, Vec2::new(3.0, 4.0));
    }

    #[test]
    fn add_assign() {
        let mut p = Point2::new(1.0, 2.0);
        p += Vec2::new(10.0, 20.0);
        assert_eq!(p, Point2::new(11.0, 22.0));
    }
}
