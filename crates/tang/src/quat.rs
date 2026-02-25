use crate::{Scalar, Vec3, Mat3};

/// Quaternion: w + xi + yj + zk
///
/// Stored as scalar part `w` and vector part `v = (x, y, z)`.
/// Represents rotations when unit-length. Supports exp/log maps
/// for Lie group operations (differentiation on SO(3)).
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Quat<S> {
    pub w: S,
    pub v: Vec3<S>,
}

impl<S: Scalar> Quat<S> {
    #[inline]
    pub fn new(w: S, x: S, y: S, z: S) -> Self {
        Self { w, v: Vec3::new(x, y, z) }
    }

    #[inline]
    pub fn identity() -> Self {
        Self { w: S::ONE, v: Vec3::zero() }
    }

    /// Quaternion from axis-angle representation
    pub fn from_axis_angle(axis: Vec3<S>, angle: S) -> Self {
        let half = angle * S::HALF;
        let (s, c) = half.sin_cos();
        Self { w: c, v: axis * s }
    }

    #[inline]
    pub fn norm_sq(&self) -> S {
        self.w * self.w + self.v.norm_sq()
    }

    #[inline]
    pub fn norm(&self) -> S { self.norm_sq().sqrt() }

    pub fn normalize(&self) -> Self {
        let n = self.norm();
        Self { w: self.w / n, v: self.v / n }
    }

    /// Quaternion multiplication (Hamilton product)
    pub fn mul(&self, other: &Quat<S>) -> Quat<S> {
        Quat {
            w: self.w * other.w - self.v.dot(other.v),
            v: other.v * self.w + self.v * other.w + self.v.cross(other.v),
        }
    }

    /// Conjugate (inverse for unit quaternions)
    #[inline]
    pub fn conjugate(&self) -> Self {
        Self { w: self.w, v: -self.v }
    }

    /// Convert to 3x3 rotation matrix
    pub fn to_matrix(&self) -> Mat3<S> {
        let two = S::TWO;
        let x = self.v.x;
        let y = self.v.y;
        let z = self.v.z;
        let w = self.w;

        Mat3::new(
            S::ONE - two * (y * y + z * z), two * (x * y - w * z),         two * (x * z + w * y),
            two * (x * y + w * z),         S::ONE - two * (x * x + z * z), two * (y * z - w * x),
            two * (x * z - w * y),         two * (y * z + w * x),         S::ONE - two * (x * x + y * y),
        )
    }

    /// Convert from rotation matrix (Shepperd's method for numerical stability)
    pub fn from_matrix(m: &Mat3<S>) -> Self {
        let trace = m.trace();
        let half = S::HALF;

        if trace > S::ZERO {
            let s = (trace + S::ONE).sqrt() * S::TWO;
            let inv_s = s.recip();
            Quat::new(
                s * half * half, // s / 4
                (m.get(2, 1) - m.get(1, 2)) * inv_s,
                (m.get(0, 2) - m.get(2, 0)) * inv_s,
                (m.get(1, 0) - m.get(0, 1)) * inv_s,
            )
        } else if m.get(0, 0) > m.get(1, 1) && m.get(0, 0) > m.get(2, 2) {
            let s = (S::ONE + m.get(0, 0) - m.get(1, 1) - m.get(2, 2)).sqrt() * S::TWO;
            let inv_s = s.recip();
            Quat::new(
                (m.get(2, 1) - m.get(1, 2)) * inv_s,
                s * half * half,
                (m.get(0, 1) + m.get(1, 0)) * inv_s,
                (m.get(0, 2) + m.get(2, 0)) * inv_s,
            )
        } else if m.get(1, 1) > m.get(2, 2) {
            let s = (S::ONE + m.get(1, 1) - m.get(0, 0) - m.get(2, 2)).sqrt() * S::TWO;
            let inv_s = s.recip();
            Quat::new(
                (m.get(0, 2) - m.get(2, 0)) * inv_s,
                (m.get(0, 1) + m.get(1, 0)) * inv_s,
                s * half * half,
                (m.get(1, 2) + m.get(2, 1)) * inv_s,
            )
        } else {
            let s = (S::ONE + m.get(2, 2) - m.get(0, 0) - m.get(1, 1)).sqrt() * S::TWO;
            let inv_s = s.recip();
            Quat::new(
                (m.get(1, 0) - m.get(0, 1)) * inv_s,
                (m.get(0, 2) + m.get(2, 0)) * inv_s,
                (m.get(1, 2) + m.get(2, 1)) * inv_s,
                s * half * half,
            )
        }
    }

    /// Exponential map: so(3) → SO(3)
    ///
    /// Takes a rotation vector (axis * angle) and returns the corresponding
    /// unit quaternion. This is the key operation for Lie group-aware autodiff.
    pub fn exp(omega: &Vec3<S>) -> Quat<S> {
        let angle = omega.norm();
        if angle < S::EPSILON {
            return Quat { w: S::ONE, v: *omega * S::HALF };
        }
        let half_angle = angle * S::HALF;
        let (s, c) = half_angle.sin_cos();
        Quat { w: c, v: *omega * (s / angle) }
    }

    /// Logarithmic map: SO(3) → so(3)
    ///
    /// Returns the rotation vector (axis * angle). Inverse of exp.
    pub fn log(&self) -> Vec3<S> {
        let norm_v = self.v.norm();
        if norm_v < S::EPSILON {
            return self.v * S::TWO;
        }
        let angle = S::TWO * norm_v.atan2(self.w);
        self.v * (angle / norm_v)
    }

    /// Rotate a vector by this quaternion: q * v * q^-1
    pub fn rotate(&self, v: Vec3<S>) -> Vec3<S> {
        let qv = Quat { w: S::ZERO, v };
        let result = self.mul(&qv).mul(&self.conjugate());
        result.v
    }

    /// Spherical linear interpolation
    pub fn slerp(&self, other: &Quat<S>, t: S) -> Quat<S> {
        let mut dot = self.w * other.w + self.v.dot(other.v);
        let mut other = *other;

        // Ensure shortest path
        if dot < S::ZERO {
            other = Quat { w: -other.w, v: -other.v };
            dot = -dot;
        }

        // Fall back to lerp for nearly-parallel quaternions
        if dot > S::ONE - S::EPSILON {
            return Quat {
                w: self.w + (other.w - self.w) * t,
                v: self.v + (other.v - self.v) * t,
            }.normalize();
        }

        let theta = dot.acos();
        let sin_theta = theta.sin();
        let a = ((S::ONE - t) * theta).sin() / sin_theta;
        let b = (t * theta).sin() / sin_theta;

        Quat {
            w: self.w * a + other.w * b,
            v: self.v * a + other.v * b,
        }
    }
}

impl<S: Scalar> Default for Quat<S> {
    fn default() -> Self { Self::identity() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_rotation() {
        let q = Quat::<f64>::identity();
        let v = Vec3::new(1.0, 2.0, 3.0);
        let rotated = q.rotate(v);
        assert!((rotated.x - v.x).abs() < 1e-10);
        assert!((rotated.y - v.y).abs() < 1e-10);
        assert!((rotated.z - v.z).abs() < 1e-10);
    }

    #[test]
    fn axis_angle_90_degrees() {
        let q = Quat::from_axis_angle(Vec3::z(), std::f64::consts::FRAC_PI_2);
        let v = Vec3::new(1.0, 0.0, 0.0);
        let rotated = q.rotate(v);
        assert!(rotated.x.abs() < 1e-10);
        assert!((rotated.y - 1.0).abs() < 1e-10);
    }

    #[test]
    fn matrix_roundtrip() {
        let q = Quat::from_axis_angle(
            Vec3::new(1.0, 1.0, 1.0).normalize(),
            1.2,
        );
        let m = q.to_matrix();
        let q2 = Quat::from_matrix(&m);
        // Quaternions are equivalent up to sign
        let dot = q.w * q2.w + q.v.dot(q2.v);
        assert!((dot.abs() - 1.0).abs() < 1e-8);
    }

    #[test]
    fn exp_log_roundtrip() {
        let omega = Vec3::new(0.3, -0.5, 0.7);
        let q = Quat::exp(&omega);
        let recovered = q.log();
        assert!((recovered.x - omega.x).abs() < 1e-10);
        assert!((recovered.y - omega.y).abs() < 1e-10);
        assert!((recovered.z - omega.z).abs() < 1e-10);
    }

    #[test]
    fn slerp_endpoints() {
        let q1 = Quat::<f64>::identity();
        let q2 = Quat::from_axis_angle(Vec3::z(), 1.0);
        let s0 = q1.slerp(&q2, 0.0);
        let s1 = q1.slerp(&q2, 1.0);
        assert!((s0.w - q1.w).abs() < 1e-10);
        assert!((s1.w - q2.w).abs() < 1e-10);
    }
}
