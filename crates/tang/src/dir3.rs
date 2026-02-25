use crate::{Scalar, Vec3};
use core::ops::Deref;

/// A normalized direction vector in 3D space.
///
/// Invariant: inner vector always has unit length.
/// Derefs to Vec3 — no more `.as_ref()` pain.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Dir3<S> {
    inner: Vec3<S>,
}

impl<S: Scalar> Dir3<S> {
    /// Create from a vector, normalizing it. Panics if zero-length.
    #[inline]
    pub fn new(v: Vec3<S>) -> Self {
        Self { inner: v.normalize() }
    }

    /// Create from a vector, normalizing it. Returns None if zero-length.
    #[inline]
    pub fn try_new(v: Vec3<S>) -> Option<Self> {
        v.try_normalize().map(|inner| Self { inner })
    }

    /// Create from a vector assumed to already be normalized.
    ///
    /// # Safety
    /// The caller must ensure the vector has unit length.
    #[inline]
    pub fn new_unchecked(v: Vec3<S>) -> Self {
        Self { inner: v }
    }

    /// The underlying vector (always unit length)
    #[inline]
    pub fn as_vec(&self) -> &Vec3<S> { &self.inner }

    /// Consume and return the inner vector
    #[inline]
    pub fn into_inner(self) -> Vec3<S> { self.inner }

    #[inline]
    pub fn x() -> Self { Self { inner: Vec3::x() } }

    #[inline]
    pub fn y() -> Self { Self { inner: Vec3::y() } }

    #[inline]
    pub fn z() -> Self { Self { inner: Vec3::z() } }

    /// Negate the direction
    #[inline]
    pub fn neg(self) -> Self { Self { inner: -self.inner } }

    /// Dot product with another direction
    #[inline]
    pub fn dot(self, other: Dir3<S>) -> S { self.inner.dot(other.inner) }

    /// Cross product with another direction (result may not be unit length)
    #[inline]
    pub fn cross(self, other: Dir3<S>) -> Vec3<S> { self.inner.cross(other.inner) }
}

// The key improvement: Dir3 derefs to Vec3
impl<S: Scalar> Deref for Dir3<S> {
    type Target = Vec3<S>;
    #[inline]
    fn deref(&self) -> &Vec3<S> { &self.inner }
}

impl<S: Scalar> PartialEq for Dir3<S> {
    fn eq(&self, other: &Self) -> bool { self.inner == other.inner }
}

impl<S: Scalar> core::fmt::Display for Dir3<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "Dir3{}", self.inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deref_to_vec3() {
        let d = Dir3::new(Vec3::new(3.0, 0.0, 0.0));
        // Can use Vec3 methods directly via Deref
        assert!((d.x - 1.0).abs() < 1e-10);
        assert!((d.y).abs() < 1e-10);
        assert!((d.norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn normalize_on_construct() {
        let d = Dir3::new(Vec3::new(1.0, 1.0, 1.0));
        assert!((d.norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn axis_constructors() {
        assert_eq!(Dir3::<f64>::x().into_inner(), Vec3::x());
        assert_eq!(Dir3::<f64>::y().into_inner(), Vec3::y());
        assert_eq!(Dir3::<f64>::z().into_inner(), Vec3::z());
    }
}
