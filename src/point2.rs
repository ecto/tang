use crate::{Scalar, Vec2};
use core::ops::{Add, Sub};

/// A point in 2D space (distinct from Vec2 — points have position, vectors have direction).
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
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
