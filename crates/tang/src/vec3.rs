use crate::Scalar;
use core::ops::{Add, Sub, Mul, Div, Neg, AddAssign, SubAssign, MulAssign};

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Vec3<S> {
    pub x: S,
    pub y: S,
    pub z: S,
}

impl<S: Scalar> Vec3<S> {
    #[inline]
    pub fn new(x: S, y: S, z: S) -> Self { Self { x, y, z } }

    #[inline]
    pub fn zero() -> Self { Self::new(S::ZERO, S::ZERO, S::ZERO) }

    #[inline]
    pub fn splat(v: S) -> Self { Self::new(v, v, v) }

    #[inline]
    pub fn x() -> Self { Self::new(S::ONE, S::ZERO, S::ZERO) }

    #[inline]
    pub fn y() -> Self { Self::new(S::ZERO, S::ONE, S::ZERO) }

    #[inline]
    pub fn z() -> Self { Self::new(S::ZERO, S::ZERO, S::ONE) }

    #[inline]
    pub fn dot(self, rhs: Self) -> S {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    #[inline]
    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    #[inline]
    pub fn norm_sq(self) -> S { self.dot(self) }

    #[inline]
    pub fn norm(self) -> S { self.norm_sq().sqrt() }

    #[inline]
    pub fn normalize(self) -> Self {
        let n = self.norm();
        self / n
    }

    #[inline]
    pub fn try_normalize(self) -> Option<Self> {
        let n = self.norm();
        if n > S::EPSILON { Some(self / n) } else { None }
    }

    #[inline]
    pub fn lerp(self, other: Self, t: S) -> Self {
        self * (S::ONE - t) + other * t
    }

    #[inline]
    pub fn component_min(self, other: Self) -> Self {
        Self::new(self.x.min(other.x), self.y.min(other.y), self.z.min(other.z))
    }

    #[inline]
    pub fn component_max(self, other: Self) -> Self {
        Self::new(self.x.max(other.x), self.y.max(other.y), self.z.max(other.z))
    }

    #[inline]
    pub fn abs(self) -> Self {
        Self::new(self.x.abs(), self.y.abs(), self.z.abs())
    }

    /// Returns the element-wise product (Hadamard product)
    #[inline]
    pub fn hadamard(self, other: Self) -> Self {
        Self::new(self.x * other.x, self.y * other.y, self.z * other.z)
    }

    #[inline]
    pub fn min_element(self) -> S {
        self.x.min(self.y.min(self.z))
    }

    #[inline]
    pub fn max_element(self) -> S {
        self.x.max(self.y.max(self.z))
    }

    /// Extend to Vec4 with a given w component
    #[inline]
    pub fn extend(self, w: S) -> crate::Vec4<S> {
        crate::Vec4::new(self.x, self.y, self.z, w)
    }

    /// Construct from a slice (panics if len < 3)
    #[inline]
    pub fn from_slice(s: &[S]) -> Self {
        Self::new(s[0], s[1], s[2])
    }

    #[inline]
    pub fn as_array(&self) -> [S; 3] {
        [self.x, self.y, self.z]
    }
}

impl<S: Scalar> Default for Vec3<S> {
    fn default() -> Self { Self::zero() }
}

impl<S: Scalar> From<[S; 3]> for Vec3<S> {
    fn from(a: [S; 3]) -> Self { Self::new(a[0], a[1], a[2]) }
}

impl<S: Scalar> From<Vec3<S>> for [S; 3] {
    fn from(v: Vec3<S>) -> Self { [v.x, v.y, v.z] }
}

impl<S: Scalar> Add for Vec3<S> {
    type Output = Self;
    #[inline] fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl<S: Scalar> Sub for Vec3<S> {
    type Output = Self;
    #[inline] fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl<S: Scalar> Neg for Vec3<S> {
    type Output = Self;
    #[inline] fn neg(self) -> Self { Self::new(-self.x, -self.y, -self.z) }
}

impl<S: Scalar> Mul<S> for Vec3<S> {
    type Output = Self;
    #[inline] fn mul(self, rhs: S) -> Self {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl<S: Scalar> Div<S> for Vec3<S> {
    type Output = Self;
    #[inline] fn div(self, rhs: S) -> Self {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

impl<S: Scalar> AddAssign for Vec3<S> {
    #[inline] fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x; self.y += rhs.y; self.z += rhs.z;
    }
}

impl<S: Scalar> SubAssign for Vec3<S> {
    #[inline] fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x; self.y -= rhs.y; self.z -= rhs.z;
    }
}

impl<S: Scalar> MulAssign<S> for Vec3<S> {
    #[inline] fn mul_assign(&mut self, rhs: S) {
        self.x *= rhs; self.y *= rhs; self.z *= rhs;
    }
}

// Scalar * Vec3 (commutative)
impl Mul<Vec3<f64>> for f64 {
    type Output = Vec3<f64>;
    #[inline] fn mul(self, rhs: Vec3<f64>) -> Vec3<f64> { rhs * self }
}

impl Mul<Vec3<f32>> for f32 {
    type Output = Vec3<f32>;
    #[inline] fn mul(self, rhs: Vec3<f32>) -> Vec3<f32> { rhs * self }
}

impl<S: Scalar> core::fmt::Display for Vec3<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_product() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(a.dot(b), 32.0);
    }

    #[test]
    fn cross_product() {
        let x = Vec3::<f64>::x();
        let y = Vec3::<f64>::y();
        let z = x.cross(y);
        assert_eq!(z, Vec3::z());
        // Anti-commutative
        assert_eq!(y.cross(x), -z);
    }

    #[test]
    fn normalize() {
        let v = Vec3::new(1.0, 2.0, 2.0);
        let n = v.normalize();
        assert!((n.norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn scalar_mul_commutative() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(v * 2.0, 2.0 * v);
    }

    #[test]
    fn lerp() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(10.0, 10.0, 10.0);
        let mid = a.lerp(b, 0.5);
        assert_eq!(mid, Vec3::new(5.0, 5.0, 5.0));
    }

    #[test]
    fn f32_vec3() {
        let v = Vec3::<f32>::new(1.0, 0.0, 0.0);
        assert_eq!(v.norm(), 1.0f32);
    }
}
