//! tang — Math library for physical reality
//!
//! Unified math foundation for geometry kernels, physics engines, and
//! differentiable simulation. Generic over scalar type to support f32, f64,
//! dual numbers (autodiff), and interval arithmetic from the same code.
//!
//! # Design principles
//! - Generic over `Scalar` type (f32, f64, Dual<S>, Interval<S>)
//! - `#[repr(C)]` everywhere for GPU interop
//! - No nalgebra dependency — full control of the stack
//! - Block-structured spatial matrices (no 6x6 index gymnastics)
//! - `Dir3` derefs to `Vec3` (no `.as_ref()` pain)

#![no_std]

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

mod scalar;
mod vec2;
mod vec3;
mod vec4;
mod point2;
mod point3;
mod dir3;
mod mat3;
mod mat4;
mod quat;
mod transform;
mod spatial;
mod dual;

#[cfg(feature = "exact")]
pub mod predicates;

pub use scalar::Scalar;
pub use vec2::Vec2;
pub use vec3::Vec3;
pub use vec4::Vec4;
pub use point2::Point2;
pub use point3::Point3;
pub use dir3::Dir3;
pub use mat3::Mat3;
pub use mat4::Mat4;
pub use quat::Quat;
pub use transform::Transform;
pub use spatial::{SpatialVec, SpatialMat, SpatialTransform, SpatialInertia};
pub use dual::Dual;

/// Cross-product matrix [v]× such that [v]× w = v × w
pub fn skew<S: Scalar>(v: &Vec3<S>) -> Mat3<S> {
    Mat3::new(
        S::ZERO, -v.z,    v.y,
        v.z,     S::ZERO, -v.x,
        -v.y,    v.x,     S::ZERO,
    )
}

pub const GRAVITY: f64 = 9.81;

// Bytemuck impls for concrete f32/f64 types (generic structs can't derive Pod)
#[cfg(feature = "bytemuck")]
mod bytemuck_impls {
    use super::*;

    macro_rules! impl_pod {
        ($t:ty) => {
            // SAFETY: All fields are the same float type, #[repr(C)], no padding
            unsafe impl bytemuck::Zeroable for $t {}
            unsafe impl bytemuck::Pod for $t {}
        };
    }

    impl_pod!(Vec2<f32>);
    impl_pod!(Vec2<f64>);
    impl_pod!(Vec3<f32>);
    impl_pod!(Vec3<f64>);
    impl_pod!(Vec4<f32>);
    impl_pod!(Vec4<f64>);
    impl_pod!(Point2<f32>);
    impl_pod!(Point2<f64>);
    impl_pod!(Point3<f32>);
    impl_pod!(Point3<f64>);
    impl_pod!(Mat3<f32>);
    impl_pod!(Mat3<f64>);
    impl_pod!(Mat4<f32>);
    impl_pod!(Mat4<f64>);
    impl_pod!(Quat<f32>);
    impl_pod!(Quat<f64>);
    impl_pod!(Transform<f32>);
    impl_pod!(Transform<f64>);
    impl_pod!(SpatialVec<f32>);
    impl_pod!(SpatialVec<f64>);
    impl_pod!(SpatialMat<f32>);
    impl_pod!(SpatialMat<f64>);
    impl_pod!(SpatialTransform<f32>);
    impl_pod!(SpatialTransform<f64>);
    impl_pod!(SpatialInertia<f32>);
    impl_pod!(SpatialInertia<f64>);

    // Dir3 wraps Vec3 transparently
    unsafe impl bytemuck::TransparentWrapper<Vec3<f32>> for Dir3<f32> {}
    unsafe impl bytemuck::TransparentWrapper<Vec3<f64>> for Dir3<f64> {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skew_cross_product() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let w = Vec3::new(4.0, 5.0, 6.0);
        let skew_v = skew(&v);
        let result = skew_v * w;
        let expected = v.cross(w);
        assert!((result.x - expected.x).abs() < 1e-10);
        assert!((result.y - expected.y).abs() < 1e-10);
        assert!((result.z - expected.z).abs() < 1e-10);
    }

    #[test]
    fn skew_antisymmetric() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let s = skew(&v);
        let st = s.transpose();
        let sum = s + st;
        assert!((sum.c0.norm_sq() + sum.c1.norm_sq() + sum.c2.norm_sq()) < 1e-20);
    }

    #[test]
    fn mat4_add_sub_neg_scale() {
        let a = Mat4::<f64>::identity();
        let b = Mat4::<f64>::identity();
        let sum = a + b;
        assert!((sum.get(0, 0) - 2.0).abs() < 1e-10);

        let diff = a - b;
        assert!(diff.get(0, 0).abs() < 1e-10);

        let neg = -a;
        assert!((neg.get(0, 0) - (-1.0)).abs() < 1e-10);

        let scaled = a * 3.0;
        assert!((scaled.get(0, 0) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn spatial_mat_neg_scale_mul_vec() {
        let m = SpatialMat::<f64>::identity();
        let v = SpatialVec::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0));

        // Identity * v = v
        let result = m * v;
        assert_eq!(result.angular, v.angular);
        assert_eq!(result.linear, v.linear);

        // Neg
        let neg = -m;
        let result2 = neg * v;
        assert_eq!(result2.angular, -v.angular);
        assert_eq!(result2.linear, -v.linear);

        // Scale
        let scaled = m * 2.0;
        let result3 = scaled * v;
        assert!((result3.angular.x - 2.0).abs() < 1e-10);
    }
}
