use crate::Scalar;
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Vec2<S> {
    pub x: S,
    pub y: S,
}

impl<S: Scalar> Vec2<S> {
    #[inline]
    pub fn new(x: S, y: S) -> Self {
        Self { x, y }
    }

    #[inline]
    pub fn zero() -> Self {
        Self::new(S::ZERO, S::ZERO)
    }

    #[inline]
    pub fn splat(v: S) -> Self {
        Self::new(v, v)
    }

    #[inline]
    pub fn x() -> Self {
        Self::new(S::ONE, S::ZERO)
    }

    #[inline]
    pub fn y() -> Self {
        Self::new(S::ZERO, S::ONE)
    }

    /// Alias for [`zero()`](Self::zero) (nalgebra compatibility).
    #[inline]
    pub fn zeros() -> Self {
        Self::zero()
    }

    /// Dot product. Accepts both owned and borrowed args (nalgebra compatibility).
    #[inline]
    pub fn dot(self, rhs: impl Into<Self>) -> S {
        let rhs = rhs.into();
        self.x * rhs.x + self.y * rhs.y
    }

    /// 2D cross product (returns scalar = signed area of parallelogram).
    /// Accepts both owned and borrowed args (nalgebra compatibility).
    #[inline]
    pub fn cross(self, rhs: impl Into<Self>) -> S {
        let rhs = rhs.into();
        self.x * rhs.y - self.y * rhs.x
    }

    /// Alias for [`norm_sq()`](Self::norm_sq) (nalgebra compatibility).
    #[inline]
    pub fn norm_squared(self) -> S {
        self.norm_sq()
    }

    #[inline]
    pub fn norm_sq(self) -> S {
        self.dot(self)
    }

    #[inline]
    pub fn norm(self) -> S {
        self.norm_sq().sqrt()
    }

    #[inline]
    pub fn normalize(self) -> Self {
        self / self.norm()
    }

    #[inline]
    pub fn lerp(self, other: Self, t: S) -> Self {
        self * (S::ONE - t) + other * t
    }

    #[inline]
    pub fn perp(self) -> Self {
        Self::new(-self.y, self.x)
    }

    #[inline]
    pub fn component_min(self, other: Self) -> Self {
        Self::new(self.x.min(other.x), self.y.min(other.y))
    }

    #[inline]
    pub fn component_max(self, other: Self) -> Self {
        Self::new(self.x.max(other.x), self.y.max(other.y))
    }
}

impl<S: Scalar> Default for Vec2<S> {
    fn default() -> Self {
        Self::zero()
    }
}

// nalgebra compatibility: `v.dot(&w)` works via Into.
impl<S: Scalar> From<&Vec2<S>> for Vec2<S> {
    #[inline]
    fn from(v: &Vec2<S>) -> Self {
        *v
    }
}

impl<S: Scalar> Add for Vec2<S> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl<S: Scalar> Sub for Vec2<S> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl<S: Scalar> Neg for Vec2<S> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y)
    }
}

impl<S: Scalar> Mul<S> for Vec2<S> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: S) -> Self {
        Self::new(self.x * rhs, self.y * rhs)
    }
}

impl<S: Scalar> Div<S> for Vec2<S> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: S) -> Self {
        Self::new(self.x / rhs, self.y / rhs)
    }
}

impl<S: Scalar> AddAssign for Vec2<S> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl<S: Scalar> SubAssign for Vec2<S> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl<S: Scalar> MulAssign<S> for Vec2<S> {
    #[inline]
    fn mul_assign(&mut self, rhs: S) {
        self.x *= rhs;
        self.y *= rhs;
    }
}

// Scalar * Vec2 (commutative)
impl Mul<Vec2<f64>> for f64 {
    type Output = Vec2<f64>;
    #[inline]
    fn mul(self, rhs: Vec2<f64>) -> Vec2<f64> {
        rhs * self
    }
}

impl Mul<Vec2<f32>> for f32 {
    type Output = Vec2<f32>;
    #[inline]
    fn mul(self, rhs: Vec2<f32>) -> Vec2<f32> {
        rhs * self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_product() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(3.0, 4.0);
        assert_eq!(a.dot(b), 11.0);
    }

    #[test]
    fn cross_product_2d() {
        let a = Vec2::new(1.0, 0.0);
        let b = Vec2::new(0.0, 1.0);
        assert_eq!(a.cross(b), 1.0);
        assert_eq!(b.cross(a), -1.0);
    }

    #[test]
    fn normalize() {
        let v = Vec2::new(3.0, 4.0);
        let n = v.normalize();
        assert!((n.norm() - 1.0).abs() < 1e-10);
    }
}
