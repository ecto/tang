//! kern-math — Math library for physical reality
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
