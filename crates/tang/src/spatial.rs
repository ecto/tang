use crate::{skew, Mat3, Scalar, Transform, Vec3};
use core::ops::{Add, Mul, Neg, Sub};

/// Spatial (6D) vector — represents either a twist (motion) or wrench (force).
///
/// Featherstone convention: angular component first, linear second.
/// Stored as two Vec3s instead of a Vec6 — cleaner API, same layout.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SpatialVec<S> {
    pub angular: Vec3<S>,
    pub linear: Vec3<S>,
}

impl<S: Scalar> SpatialVec<S> {
    #[inline]
    pub fn new(angular: Vec3<S>, linear: Vec3<S>) -> Self {
        Self { angular, linear }
    }

    #[inline]
    pub fn zero() -> Self {
        Self::new(Vec3::zero(), Vec3::zero())
    }

    #[inline]
    pub fn from_linear(v: Vec3<S>) -> Self {
        Self::new(Vec3::zero(), v)
    }

    #[inline]
    pub fn from_angular(w: Vec3<S>) -> Self {
        Self::new(w, Vec3::zero())
    }

    /// Spatial cross product for motion vectors: [v] × m
    /// Used in velocity propagation through kinematic chains.
    pub fn cross_motion(&self, other: &SpatialVec<S>) -> SpatialVec<S> {
        SpatialVec::new(
            self.angular.cross(other.angular),
            self.angular.cross(other.linear) + self.linear.cross(other.angular),
        )
    }

    /// Spatial cross product for force vectors: [v] ×* f
    /// Used in bias force computation.
    pub fn cross_force(&self, other: &SpatialVec<S>) -> SpatialVec<S> {
        SpatialVec::new(
            self.angular.cross(other.angular) + self.linear.cross(other.linear),
            self.angular.cross(other.linear),
        )
    }

    /// Spatial dot product: v^T * f (motion dotted with force = power)
    #[inline]
    pub fn dot(&self, other: &SpatialVec<S>) -> S {
        self.angular.dot(other.angular) + self.linear.dot(other.linear)
    }

    /// Convert to a flat [S; 6] array
    #[inline]
    pub fn as_array(&self) -> [S; 6] {
        [
            self.angular.x,
            self.angular.y,
            self.angular.z,
            self.linear.x,
            self.linear.y,
            self.linear.z,
        ]
    }
}

impl<S: Scalar> Default for SpatialVec<S> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<S: Scalar> Add for SpatialVec<S> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.angular + rhs.angular, self.linear + rhs.linear)
    }
}

impl<S: Scalar> Sub for SpatialVec<S> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.angular - rhs.angular, self.linear - rhs.linear)
    }
}

impl<S: Scalar> Neg for SpatialVec<S> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.angular, -self.linear)
    }
}

impl<S: Scalar> Mul<S> for SpatialVec<S> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: S) -> Self {
        Self::new(self.angular * rhs, self.linear * rhs)
    }
}

// ---------------------------------------------------------------------------

/// 6x6 spatial matrix, stored as four 3x3 blocks.
///
/// ```text
/// | upper_left  upper_right |
/// | lower_left  lower_right |
/// ```
///
/// This eliminates the `fixed_view_mut::<3,3>` pain entirely.
/// Plücker transforms and spatial inertia matrices are naturally block-structured.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SpatialMat<S> {
    pub upper_left: Mat3<S>,
    pub upper_right: Mat3<S>,
    pub lower_left: Mat3<S>,
    pub lower_right: Mat3<S>,
}

impl<S: Scalar> SpatialMat<S> {
    #[inline]
    pub fn new(ul: Mat3<S>, ur: Mat3<S>, ll: Mat3<S>, lr: Mat3<S>) -> Self {
        Self {
            upper_left: ul,
            upper_right: ur,
            lower_left: ll,
            lower_right: lr,
        }
    }

    #[inline]
    pub fn zero() -> Self {
        let z = Mat3::zero();
        Self::new(z, z, z, z)
    }

    #[inline]
    pub fn identity() -> Self {
        Self::new(
            Mat3::identity(),
            Mat3::zero(),
            Mat3::zero(),
            Mat3::identity(),
        )
    }

    /// Multiply by a spatial vector (block matrix-vector product)
    #[inline]
    pub fn mul_vec(&self, v: &SpatialVec<S>) -> SpatialVec<S> {
        SpatialVec::new(
            self.upper_left.mul_vec(v.angular) + self.upper_right.mul_vec(v.linear),
            self.lower_left.mul_vec(v.angular) + self.lower_right.mul_vec(v.linear),
        )
    }

    /// Block matrix-matrix product
    pub fn mul_mat(&self, rhs: &SpatialMat<S>) -> SpatialMat<S> {
        SpatialMat::new(
            self.upper_left.mul_mat(&rhs.upper_left) + self.upper_right.mul_mat(&rhs.lower_left),
            self.upper_left.mul_mat(&rhs.upper_right) + self.upper_right.mul_mat(&rhs.lower_right),
            self.lower_left.mul_mat(&rhs.upper_left) + self.lower_right.mul_mat(&rhs.lower_left),
            self.lower_left.mul_mat(&rhs.upper_right) + self.lower_right.mul_mat(&rhs.lower_right),
        )
    }

    #[inline]
    pub fn transpose(&self) -> SpatialMat<S> {
        SpatialMat::new(
            self.upper_left.transpose(),
            self.lower_left.transpose(),
            self.upper_right.transpose(),
            self.lower_right.transpose(),
        )
    }
}

impl<S: Scalar> Add for SpatialMat<S> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(
            self.upper_left + rhs.upper_left,
            self.upper_right + rhs.upper_right,
            self.lower_left + rhs.lower_left,
            self.lower_right + rhs.lower_right,
        )
    }
}

impl<S: Scalar> Sub for SpatialMat<S> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(
            self.upper_left - rhs.upper_left,
            self.upper_right - rhs.upper_right,
            self.lower_left - rhs.lower_left,
            self.lower_right - rhs.lower_right,
        )
    }
}

impl<S: Scalar> Neg for SpatialMat<S> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(
            -self.upper_left,
            -self.upper_right,
            -self.lower_left,
            -self.lower_right,
        )
    }
}

impl<S: Scalar> Mul<S> for SpatialMat<S> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: S) -> Self {
        Self::new(
            self.upper_left * rhs,
            self.upper_right * rhs,
            self.lower_left * rhs,
            self.lower_right * rhs,
        )
    }
}

impl<S: Scalar> Mul<SpatialVec<S>> for SpatialMat<S> {
    type Output = SpatialVec<S>;
    #[inline]
    fn mul(self, rhs: SpatialVec<S>) -> SpatialVec<S> {
        self.mul_vec(&rhs)
    }
}

impl<S: Scalar> Mul for SpatialMat<S> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.mul_mat(&rhs)
    }
}

// ---------------------------------------------------------------------------

/// Spatial transform: Plücker coordinate transform.
///
/// Stores rotation (Mat3) + translation (Vec3), same as `Transform`,
/// but provides spatial-algebra-specific operations (apply to motion/force vectors,
/// convert to 6x6 Plücker matrices).
///
/// Key improvement: `apply_motion` and `apply_force` operate directly
/// on the rotation + translation without constructing a 6x6 matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SpatialTransform<S> {
    pub rot: Mat3<S>,
    pub pos: Vec3<S>,
}

impl<S: Scalar> SpatialTransform<S> {
    #[inline]
    pub fn new(rot: Mat3<S>, pos: Vec3<S>) -> Self {
        Self { rot, pos }
    }

    #[inline]
    pub fn identity() -> Self {
        Self::new(Mat3::identity(), Vec3::zero())
    }

    #[inline]
    pub fn from_translation(pos: Vec3<S>) -> Self {
        Self::new(Mat3::identity(), pos)
    }

    pub fn rot_x(angle: S) -> Self {
        Self::new(Mat3::rotation_x(angle), Vec3::zero())
    }
    pub fn rot_y(angle: S) -> Self {
        Self::new(Mat3::rotation_y(angle), Vec3::zero())
    }
    pub fn rot_z(angle: S) -> Self {
        Self::new(Mat3::rotation_z(angle), Vec3::zero())
    }

    pub fn rot_axis(axis: Vec3<S>, angle: S) -> Self {
        Self::new(Mat3::rotation_axis(axis, angle), Vec3::zero())
    }

    /// Convert from a Transform
    #[inline]
    pub fn from_transform(t: &Transform<S>) -> Self {
        Self::new(t.rotation, t.translation)
    }

    /// Convert to a Transform
    #[inline]
    pub fn to_transform(&self) -> Transform<S> {
        Transform::new(self.rot, self.pos)
    }

    /// 6x6 motion transform matrix (for velocity/acceleration)
    /// ```text
    /// | R        0     |
    /// | -R[p]×   R     |
    /// ```
    pub fn to_motion_matrix(&self) -> SpatialMat<S> {
        let px = skew(&self.pos);
        SpatialMat::new(self.rot, Mat3::zero(), -self.rot.mul_mat(&px), self.rot)
    }

    /// 6x6 force transform matrix (for forces/torques)
    /// ```text
    /// | R       -R[p]× |
    /// | 0        R     |
    /// ```
    pub fn to_force_matrix(&self) -> SpatialMat<S> {
        let px = skew(&self.pos);
        SpatialMat::new(self.rot, -self.rot.mul_mat(&px), Mat3::zero(), self.rot)
    }

    /// Apply to a motion vector (twist) without building the 6x6 matrix
    pub fn apply_motion(&self, v: &SpatialVec<S>) -> SpatialVec<S> {
        let rot_w = self.rot.mul_vec(v.angular);
        SpatialVec::new(
            rot_w,
            self.rot.mul_vec(v.linear) + (-self.rot.mul_vec(self.pos.cross(v.angular))),
        )
    }

    /// Apply to a force vector (wrench) without building the 6x6 matrix
    pub fn apply_force(&self, f: &SpatialVec<S>) -> SpatialVec<S> {
        SpatialVec::new(
            self.rot.mul_vec(f.angular) + (-self.rot.mul_vec(self.pos.cross(f.linear))),
            self.rot.mul_vec(f.linear),
        )
    }

    /// Inverse-apply to a motion vector: X^{-1} * v
    pub fn inv_apply_motion(&self, v: &SpatialVec<S>) -> SpatialVec<S> {
        let rt = self.rot.transpose();
        let w = rt.mul_vec(v.angular);
        SpatialVec::new(w, rt.mul_vec(v.linear) + self.pos.cross(w))
    }

    /// Inverse-apply to a force vector: X^{-T} * f
    pub fn inv_apply_force(&self, f: &SpatialVec<S>) -> SpatialVec<S> {
        let rt = self.rot.transpose();
        SpatialVec::new(
            rt.mul_vec(f.angular) + self.pos.cross(rt.mul_vec(f.linear)),
            rt.mul_vec(f.linear),
        )
    }

    /// Compose two spatial transforms
    #[inline]
    pub fn compose(&self, other: &SpatialTransform<S>) -> SpatialTransform<S> {
        SpatialTransform {
            rot: self.rot.mul_mat(&other.rot),
            pos: other.pos + other.rot.transpose().mul_vec(self.pos),
        }
    }

    /// Inverse spatial transform
    #[inline]
    pub fn inverse(&self) -> SpatialTransform<S> {
        let rt = self.rot.transpose();
        SpatialTransform {
            rot: rt,
            pos: -self.rot.mul_vec(self.pos),
        }
    }
}

impl<S: Scalar> Default for SpatialTransform<S> {
    fn default() -> Self {
        Self::identity()
    }
}

// ---------------------------------------------------------------------------

/// Spatial inertia — compact representation.
///
/// Stored as mass + center of mass + rotational inertia (3x3).
/// Converts to 6x6 spatial inertia matrix on demand.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SpatialInertia<S> {
    pub mass: S,
    pub com: Vec3<S>,
    pub inertia: Mat3<S>,
}

impl<S: Scalar> SpatialInertia<S> {
    #[inline]
    pub fn new(mass: S, com: Vec3<S>, inertia: Mat3<S>) -> Self {
        Self { mass, com, inertia }
    }

    /// Point mass at a given position
    pub fn point_mass(mass: S, pos: Vec3<S>) -> Self {
        Self::new(mass, pos, Mat3::zero())
    }

    /// Uniform rod along Y axis
    pub fn rod(mass: S, length: S) -> Self {
        let i = mass * length * length / S::from_i32(12);
        Self::new(mass, Vec3::zero(), Mat3::diagonal(Vec3::new(i, S::ZERO, i)))
    }

    /// Uniform solid sphere
    pub fn sphere(mass: S, radius: S) -> Self {
        let i = S::TWO * mass * radius * radius / S::from_i32(5);
        Self::new(mass, Vec3::zero(), Mat3::diagonal(Vec3::new(i, i, i)))
    }

    /// Convert to 6x6 spatial inertia matrix (block-structured)
    ///
    /// ```text
    /// | I + m[c]×[c]×^T    m[c]× |
    /// | m[c]×^T             mE    |
    /// ```
    pub fn to_matrix(&self) -> SpatialMat<S> {
        let cx = skew(&self.com);
        let m_cx = cx * self.mass;
        let m_cx_cxt = m_cx.mul_mat(&cx.transpose());

        SpatialMat::new(
            self.inertia + m_cx_cxt,
            m_cx,
            m_cx.transpose(),
            Mat3::diagonal(Vec3::splat(self.mass)),
        )
    }

    /// Transform spatial inertia to a new reference frame
    pub fn transform(&self, xform: &SpatialTransform<S>) -> SpatialInertia<S> {
        let x_force = xform.to_force_matrix();
        let i_mat = self.to_matrix();
        let x_motion = xform.to_motion_matrix();
        let transformed = x_force.mul_mat(&i_mat).mul_mat(&x_motion.transpose());

        SpatialInertia::new(
            self.mass,
            xform.rot.mul_vec(self.com) + xform.pos,
            transformed.upper_left,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spatial_vec_cross_motion() {
        let v = SpatialVec::new(Vec3::new(1.0, 0.0, 0.0), Vec3::zero());
        let m = SpatialVec::new(Vec3::new(0.0, 1.0, 0.0), Vec3::zero());
        let result = v.cross_motion(&m);
        // [1,0,0] × [0,1,0] = [0,0,1]
        assert!((result.angular.z - 1.0).abs() < 1e-10);
    }

    #[test]
    fn spatial_transform_identity() {
        let x = SpatialTransform::<f64>::identity();
        let v = SpatialVec::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0));
        let result = x.apply_motion(&v);
        assert_eq!(result.angular, v.angular);
        assert_eq!(result.linear, v.linear);
    }

    #[test]
    fn spatial_transform_inverse_roundtrip() {
        let x = SpatialTransform::new(Mat3::rotation_z(1.0), Vec3::new(1.0, 2.0, 3.0));
        let v = SpatialVec::new(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let forward = x.apply_motion(&v);
        let back = x.inv_apply_motion(&forward);
        assert!((back.angular.x - v.angular.x).abs() < 1e-10);
        assert!((back.linear.y - v.linear.y).abs() < 1e-10);
    }

    #[test]
    fn spatial_mat_block_structure() {
        let x = SpatialTransform::new(Mat3::rotation_z(0.5), Vec3::new(1.0, 0.0, 0.0));
        let motion_mat = x.to_motion_matrix();
        // Upper-right block should be zero for motion transforms
        assert_eq!(motion_mat.upper_right, Mat3::zero());
    }

    #[test]
    fn sphere_inertia_symmetry() {
        let si = SpatialInertia::sphere(1.0, 1.0);
        let mat = si.to_matrix();
        // Should be symmetric
        let diff = mat.upper_right + (-mat.lower_left.transpose());
        assert!((diff.c0.norm_sq() + diff.c1.norm_sq() + diff.c2.norm_sq()) < 1e-10);
    }
}
