use crate::{Scalar, Vec3, Point3, Mat3, Mat4, Dir3, Quat};

/// Rigid body transform (isometry): rotation + translation.
///
/// Unlike Mat4, this is always a valid rigid transform (no scale/shear).
/// Composition and inversion are exact (no matrix inversion needed).
///
/// This replaces both vcad's `Transform` (for rigid cases) and
/// phyz's `SpatialTransform` as the core rigid body transform.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Transform<S> {
    pub rotation: Mat3<S>,
    pub translation: Vec3<S>,
}

impl<S: Scalar> Transform<S> {
    #[inline]
    pub fn new(rotation: Mat3<S>, translation: Vec3<S>) -> Self {
        Self { rotation, translation }
    }

    #[inline]
    pub fn identity() -> Self {
        Self::new(Mat3::identity(), Vec3::zero())
    }

    #[inline]
    pub fn from_translation(t: Vec3<S>) -> Self {
        Self::new(Mat3::identity(), t)
    }

    #[inline]
    pub fn from_rotation(r: Mat3<S>) -> Self {
        Self::new(r, Vec3::zero())
    }

    pub fn from_quat(q: &Quat<S>) -> Self {
        Self::new(q.to_matrix(), Vec3::zero())
    }

    pub fn from_quat_translation(q: &Quat<S>, t: Vec3<S>) -> Self {
        Self::new(q.to_matrix(), t)
    }

    pub fn rot_x(angle: S) -> Self { Self::from_rotation(Mat3::rotation_x(angle)) }
    pub fn rot_y(angle: S) -> Self { Self::from_rotation(Mat3::rotation_y(angle)) }
    pub fn rot_z(angle: S) -> Self { Self::from_rotation(Mat3::rotation_z(angle)) }

    pub fn rot_axis(axis: Vec3<S>, angle: S) -> Self {
        Self::from_rotation(Mat3::rotation_axis(axis, angle))
    }

    /// Transform a point: R * p + t
    #[inline]
    pub fn apply_point(&self, p: Point3<S>) -> Point3<S> {
        Point3::from_vec(self.rotation.mul_vec(p.to_vec()) + self.translation)
    }

    /// Transform a vector: R * v (no translation)
    #[inline]
    pub fn apply_vec(&self, v: Vec3<S>) -> Vec3<S> {
        self.rotation.mul_vec(v)
    }

    /// Transform a direction (same as apply_vec for rigid transforms)
    #[inline]
    pub fn apply_dir(&self, d: &Dir3<S>) -> Dir3<S> {
        Dir3::new_unchecked(self.rotation.mul_vec(d.into_inner()))
    }

    /// Inverse transform a point: R^T * (p - t)
    #[inline]
    pub fn inv_apply_point(&self, p: Point3<S>) -> Point3<S> {
        let rt = self.rotation.transpose();
        Point3::from_vec(rt.mul_vec(p.to_vec() - self.translation))
    }

    /// Inverse transform a vector: R^T * v
    #[inline]
    pub fn inv_apply_vec(&self, v: Vec3<S>) -> Vec3<S> {
        self.rotation.transpose().mul_vec(v)
    }

    /// Compose: self then other (other applied first, then self)
    /// Result: (R1 * R2, R1 * t2 + t1)
    #[inline]
    pub fn compose(&self, other: &Transform<S>) -> Transform<S> {
        Transform {
            rotation: self.rotation.mul_mat(&other.rotation),
            translation: self.rotation.mul_vec(other.translation) + self.translation,
        }
    }

    /// Exact inverse: (R^T, -R^T * t)
    /// No matrix inversion needed — rotation matrices are orthogonal.
    #[inline]
    pub fn inverse(&self) -> Transform<S> {
        let rt = self.rotation.transpose();
        Transform {
            rotation: rt,
            translation: -rt.mul_vec(self.translation),
        }
    }

    /// Convert to a 4x4 homogeneous matrix
    #[inline]
    pub fn to_mat4(&self) -> Mat4<S> {
        Mat4::from_rotation_translation(self.rotation, self.translation)
    }

    /// Convert to quaternion + translation
    pub fn to_quat_translation(&self) -> (Quat<S>, Vec3<S>) {
        (Quat::from_matrix(&self.rotation), self.translation)
    }
}

impl<S: Scalar> Default for Transform<S> {
    fn default() -> Self { Self::identity() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_roundtrip() {
        let t = Transform::new(
            Mat3::rotation_z(1.0),
            Vec3::new(1.0, 2.0, 3.0),
        );
        let ti = t.inverse();
        let id = t.compose(&ti);
        let p = Point3::new(5.0, 6.0, 7.0);
        let result = id.apply_point(p);
        assert!((result.x - p.x).abs() < 1e-10);
        assert!((result.y - p.y).abs() < 1e-10);
        assert!((result.z - p.z).abs() < 1e-10);
    }

    #[test]
    fn compose_matches_sequential_apply() {
        let t1 = Transform::new(Mat3::rotation_x(0.5), Vec3::new(1.0, 0.0, 0.0));
        let t2 = Transform::new(Mat3::rotation_y(0.3), Vec3::new(0.0, 2.0, 0.0));
        let composed = t1.compose(&t2);
        let p = Point3::new(1.0, 1.0, 1.0);
        let sequential = t1.apply_point(t2.apply_point(p));
        let direct = composed.apply_point(p);
        assert!((sequential.x - direct.x).abs() < 1e-10);
        assert!((sequential.y - direct.y).abs() < 1e-10);
        assert!((sequential.z - direct.z).abs() < 1e-10);
    }

    #[test]
    fn translation_only() {
        let t = Transform::from_translation(Vec3::new(10.0, 0.0, 0.0));
        let p = Point3::new(1.0, 2.0, 3.0);
        assert_eq!(t.apply_point(p), Point3::new(11.0, 2.0, 3.0));
    }
}
