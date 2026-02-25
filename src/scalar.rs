use core::fmt;
use core::ops::{Add, Sub, Mul, Div, Neg, AddAssign, SubAssign, MulAssign, DivAssign};

/// Trait for scalar types that can be used throughout kern-math.
///
/// Implemented for f32, f64. Will be implemented for Dual<S> (autodiff)
/// and Interval<S> (verified computation).
pub trait Scalar:
    Copy
    + Clone
    + fmt::Debug
    + fmt::Display
    + PartialEq
    + PartialOrd
    + Default
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + Send
    + Sync
    + 'static
{
    const ZERO: Self;
    const ONE: Self;
    const TWO: Self;
    const HALF: Self;
    const PI: Self;
    const TAU: Self;
    const FRAC_PI_2: Self;
    const EPSILON: Self;
    const INFINITY: Self;
    const NEG_INFINITY: Self;

    fn sqrt(self) -> Self;
    fn abs(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan2(self, other: Self) -> Self;
    fn sin_cos(self) -> (Self, Self);
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
    fn clamp(self, lo: Self, hi: Self) -> Self;
    fn recip(self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn copysign(self, sign: Self) -> Self;
    fn signum(self) -> Self;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn round(self) -> Self;

    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
    fn from_i32(v: i32) -> Self;
}

macro_rules! impl_scalar_float {
    ($t:ty, $zero:expr, $one:expr, $two:expr, $half:expr,
     $pi:expr, $tau:expr, $frac_pi_2:expr, $eps:expr, $inf:expr, $neg_inf:expr) => {
        impl Scalar for $t {
            const ZERO: Self = $zero;
            const ONE: Self = $one;
            const TWO: Self = $two;
            const HALF: Self = $half;
            const PI: Self = $pi;
            const TAU: Self = $tau;
            const FRAC_PI_2: Self = $frac_pi_2;
            const EPSILON: Self = $eps;
            const INFINITY: Self = $inf;
            const NEG_INFINITY: Self = $neg_inf;

            #[inline] fn sqrt(self) -> Self { <$t>::sqrt(self) }
            #[inline] fn abs(self) -> Self { <$t>::abs(self) }
            #[inline] fn sin(self) -> Self { <$t>::sin(self) }
            #[inline] fn cos(self) -> Self { <$t>::cos(self) }
            #[inline] fn tan(self) -> Self { <$t>::tan(self) }
            #[inline] fn asin(self) -> Self { <$t>::asin(self) }
            #[inline] fn acos(self) -> Self { <$t>::acos(self) }
            #[inline] fn atan2(self, other: Self) -> Self { <$t>::atan2(self, other) }
            #[inline] fn sin_cos(self) -> (Self, Self) { <$t>::sin_cos(self) }
            #[inline] fn min(self, other: Self) -> Self { <$t>::min(self, other) }
            #[inline] fn max(self, other: Self) -> Self { <$t>::max(self, other) }
            #[inline] fn clamp(self, lo: Self, hi: Self) -> Self { <$t>::clamp(self, lo, hi) }
            #[inline] fn recip(self) -> Self { <$t>::recip(self) }
            #[inline] fn powi(self, n: i32) -> Self { <$t>::powi(self, n) }
            #[inline] fn copysign(self, sign: Self) -> Self { <$t>::copysign(self, sign) }
            #[inline] fn signum(self) -> Self { <$t>::signum(self) }
            #[inline] fn floor(self) -> Self { <$t>::floor(self) }
            #[inline] fn ceil(self) -> Self { <$t>::ceil(self) }
            #[inline] fn round(self) -> Self { <$t>::round(self) }

            #[inline] fn from_f64(v: f64) -> Self { v as $t }
            #[inline] fn to_f64(self) -> f64 { self as f64 }
            #[inline] fn from_i32(v: i32) -> Self { v as $t }
        }
    };
}

impl_scalar_float!(f32, 0.0, 1.0, 2.0, 0.5,
    core::f32::consts::PI, core::f32::consts::TAU, core::f32::consts::FRAC_PI_2,
    f32::EPSILON, f32::INFINITY, f32::NEG_INFINITY);
impl_scalar_float!(f64, 0.0, 1.0, 2.0, 0.5,
    core::f64::consts::PI, core::f64::consts::TAU, core::f64::consts::FRAC_PI_2,
    f64::EPSILON, f64::INFINITY, f64::NEG_INFINITY);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f64_basics() {
        assert_eq!(f64::ZERO, 0.0);
        assert_eq!(f64::ONE, 1.0);
        assert!((f64::PI - std::f64::consts::PI).abs() < f64::EPSILON);
        assert_eq!(4.0_f64.sqrt(), 2.0);
        assert_eq!((-3.0_f64).abs(), 3.0);
    }

    #[test]
    fn f32_basics() {
        assert_eq!(f32::ZERO, 0.0);
        assert!((f32::PI - std::f32::consts::PI).abs() < f32::EPSILON);
    }
}
