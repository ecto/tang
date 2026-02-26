use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

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

    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn powf(self, p: Self) -> Self;
    fn tanh(self) -> Self;
    fn sinh(self) -> Self;
    fn cosh(self) -> Self;
    fn acosh(self) -> Self;
    fn asinh(self) -> Self;
    fn atanh(self) -> Self;

    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
    fn from_i32(v: i32) -> Self;
}

// In std mode, use inherent float methods. In no_std, use libm.
// Dispatch via a helper trait to keep the macro clean.
#[cfg(feature = "std")]
mod float_ops {
    #[inline(always)]
    pub fn sqrt_f32(x: f32) -> f32 {
        x.sqrt()
    }
    #[inline(always)]
    pub fn sqrt_f64(x: f64) -> f64 {
        x.sqrt()
    }
    #[inline(always)]
    pub fn abs_f32(x: f32) -> f32 {
        x.abs()
    }
    #[inline(always)]
    pub fn abs_f64(x: f64) -> f64 {
        x.abs()
    }
    #[inline(always)]
    pub fn sin_f32(x: f32) -> f32 {
        x.sin()
    }
    #[inline(always)]
    pub fn sin_f64(x: f64) -> f64 {
        x.sin()
    }
    #[inline(always)]
    pub fn cos_f32(x: f32) -> f32 {
        x.cos()
    }
    #[inline(always)]
    pub fn cos_f64(x: f64) -> f64 {
        x.cos()
    }
    #[inline(always)]
    pub fn tan_f32(x: f32) -> f32 {
        x.tan()
    }
    #[inline(always)]
    pub fn tan_f64(x: f64) -> f64 {
        x.tan()
    }
    #[inline(always)]
    pub fn asin_f32(x: f32) -> f32 {
        x.asin()
    }
    #[inline(always)]
    pub fn asin_f64(x: f64) -> f64 {
        x.asin()
    }
    #[inline(always)]
    pub fn acos_f32(x: f32) -> f32 {
        x.acos()
    }
    #[inline(always)]
    pub fn acos_f64(x: f64) -> f64 {
        x.acos()
    }
    #[inline(always)]
    pub fn atan2_f32(y: f32, x: f32) -> f32 {
        y.atan2(x)
    }
    #[inline(always)]
    pub fn atan2_f64(y: f64, x: f64) -> f64 {
        y.atan2(x)
    }
    #[inline(always)]
    pub fn sin_cos_f32(x: f32) -> (f32, f32) {
        x.sin_cos()
    }
    #[inline(always)]
    pub fn sin_cos_f64(x: f64) -> (f64, f64) {
        x.sin_cos()
    }
    #[inline(always)]
    pub fn floor_f32(x: f32) -> f32 {
        x.floor()
    }
    #[inline(always)]
    pub fn floor_f64(x: f64) -> f64 {
        x.floor()
    }
    #[inline(always)]
    pub fn ceil_f32(x: f32) -> f32 {
        x.ceil()
    }
    #[inline(always)]
    pub fn ceil_f64(x: f64) -> f64 {
        x.ceil()
    }
    #[inline(always)]
    pub fn round_f32(x: f32) -> f32 {
        x.round()
    }
    #[inline(always)]
    pub fn round_f64(x: f64) -> f64 {
        x.round()
    }
    #[inline(always)]
    pub fn exp_f32(x: f32) -> f32 {
        x.exp()
    }
    #[inline(always)]
    pub fn exp_f64(x: f64) -> f64 {
        x.exp()
    }
    #[inline(always)]
    pub fn ln_f32(x: f32) -> f32 {
        x.ln()
    }
    #[inline(always)]
    pub fn ln_f64(x: f64) -> f64 {
        x.ln()
    }
    #[inline(always)]
    pub fn powf_f32(x: f32, p: f32) -> f32 {
        x.powf(p)
    }
    #[inline(always)]
    pub fn powf_f64(x: f64, p: f64) -> f64 {
        x.powf(p)
    }
    #[inline(always)]
    pub fn tanh_f32(x: f32) -> f32 {
        x.tanh()
    }
    #[inline(always)]
    pub fn tanh_f64(x: f64) -> f64 {
        x.tanh()
    }
    #[inline(always)]
    pub fn sinh_f32(x: f32) -> f32 {
        x.sinh()
    }
    #[inline(always)]
    pub fn sinh_f64(x: f64) -> f64 {
        x.sinh()
    }
    #[inline(always)]
    pub fn cosh_f32(x: f32) -> f32 {
        x.cosh()
    }
    #[inline(always)]
    pub fn cosh_f64(x: f64) -> f64 {
        x.cosh()
    }
    #[inline(always)]
    pub fn acosh_f32(x: f32) -> f32 {
        x.acosh()
    }
    #[inline(always)]
    pub fn acosh_f64(x: f64) -> f64 {
        x.acosh()
    }
    #[inline(always)]
    pub fn asinh_f32(x: f32) -> f32 {
        x.asinh()
    }
    #[inline(always)]
    pub fn asinh_f64(x: f64) -> f64 {
        x.asinh()
    }
    #[inline(always)]
    pub fn atanh_f32(x: f32) -> f32 {
        x.atanh()
    }
    #[inline(always)]
    pub fn atanh_f64(x: f64) -> f64 {
        x.atanh()
    }
    #[inline(always)]
    pub fn copysign_f32(x: f32, s: f32) -> f32 {
        x.copysign(s)
    }
    #[inline(always)]
    pub fn copysign_f64(x: f64, s: f64) -> f64 {
        x.copysign(s)
    }
    #[inline(always)]
    pub fn powi_f32(x: f32, n: i32) -> f32 {
        x.powi(n)
    }
    #[inline(always)]
    pub fn powi_f64(x: f64, n: i32) -> f64 {
        x.powi(n)
    }
}

#[cfg(all(not(feature = "std"), feature = "libm"))]
mod float_ops {
    #[inline(always)]
    pub fn sqrt_f32(x: f32) -> f32 {
        libm::sqrtf(x)
    }
    #[inline(always)]
    pub fn sqrt_f64(x: f64) -> f64 {
        libm::sqrt(x)
    }
    #[inline(always)]
    pub fn abs_f32(x: f32) -> f32 {
        libm::fabsf(x)
    }
    #[inline(always)]
    pub fn abs_f64(x: f64) -> f64 {
        libm::fabs(x)
    }
    #[inline(always)]
    pub fn sin_f32(x: f32) -> f32 {
        libm::sinf(x)
    }
    #[inline(always)]
    pub fn sin_f64(x: f64) -> f64 {
        libm::sin(x)
    }
    #[inline(always)]
    pub fn cos_f32(x: f32) -> f32 {
        libm::cosf(x)
    }
    #[inline(always)]
    pub fn cos_f64(x: f64) -> f64 {
        libm::cos(x)
    }
    #[inline(always)]
    pub fn tan_f32(x: f32) -> f32 {
        libm::tanf(x)
    }
    #[inline(always)]
    pub fn tan_f64(x: f64) -> f64 {
        libm::tan(x)
    }
    #[inline(always)]
    pub fn asin_f32(x: f32) -> f32 {
        libm::asinf(x)
    }
    #[inline(always)]
    pub fn asin_f64(x: f64) -> f64 {
        libm::asin(x)
    }
    #[inline(always)]
    pub fn acos_f32(x: f32) -> f32 {
        libm::acosf(x)
    }
    #[inline(always)]
    pub fn acos_f64(x: f64) -> f64 {
        libm::acos(x)
    }
    #[inline(always)]
    pub fn atan2_f32(y: f32, x: f32) -> f32 {
        libm::atan2f(y, x)
    }
    #[inline(always)]
    pub fn atan2_f64(y: f64, x: f64) -> f64 {
        libm::atan2(y, x)
    }
    #[inline(always)]
    pub fn sin_cos_f32(x: f32) -> (f32, f32) {
        libm::sincosf(x)
    }
    #[inline(always)]
    pub fn sin_cos_f64(x: f64) -> (f64, f64) {
        libm::sincos(x)
    }
    #[inline(always)]
    pub fn floor_f32(x: f32) -> f32 {
        libm::floorf(x)
    }
    #[inline(always)]
    pub fn floor_f64(x: f64) -> f64 {
        libm::floor(x)
    }
    #[inline(always)]
    pub fn ceil_f32(x: f32) -> f32 {
        libm::ceilf(x)
    }
    #[inline(always)]
    pub fn ceil_f64(x: f64) -> f64 {
        libm::ceil(x)
    }
    #[inline(always)]
    pub fn round_f32(x: f32) -> f32 {
        libm::roundf(x)
    }
    #[inline(always)]
    pub fn round_f64(x: f64) -> f64 {
        libm::round(x)
    }
    #[inline(always)]
    pub fn exp_f32(x: f32) -> f32 {
        libm::expf(x)
    }
    #[inline(always)]
    pub fn exp_f64(x: f64) -> f64 {
        libm::exp(x)
    }
    #[inline(always)]
    pub fn ln_f32(x: f32) -> f32 {
        libm::logf(x)
    }
    #[inline(always)]
    pub fn ln_f64(x: f64) -> f64 {
        libm::log(x)
    }
    #[inline(always)]
    pub fn powf_f32(x: f32, p: f32) -> f32 {
        libm::powf(x, p)
    }
    #[inline(always)]
    pub fn powf_f64(x: f64, p: f64) -> f64 {
        libm::pow(x, p)
    }
    #[inline(always)]
    pub fn tanh_f32(x: f32) -> f32 {
        libm::tanhf(x)
    }
    #[inline(always)]
    pub fn tanh_f64(x: f64) -> f64 {
        libm::tanh(x)
    }
    #[inline(always)]
    pub fn sinh_f32(x: f32) -> f32 {
        libm::sinhf(x)
    }
    #[inline(always)]
    pub fn sinh_f64(x: f64) -> f64 {
        libm::sinh(x)
    }
    #[inline(always)]
    pub fn cosh_f32(x: f32) -> f32 {
        libm::coshf(x)
    }
    #[inline(always)]
    pub fn cosh_f64(x: f64) -> f64 {
        libm::cosh(x)
    }
    #[inline(always)]
    pub fn acosh_f32(x: f32) -> f32 {
        libm::acoshf(x)
    }
    #[inline(always)]
    pub fn acosh_f64(x: f64) -> f64 {
        libm::acosh(x)
    }
    #[inline(always)]
    pub fn asinh_f32(x: f32) -> f32 {
        libm::asinhf(x)
    }
    #[inline(always)]
    pub fn asinh_f64(x: f64) -> f64 {
        libm::asinh(x)
    }
    #[inline(always)]
    pub fn atanh_f32(x: f32) -> f32 {
        libm::atanhf(x)
    }
    #[inline(always)]
    pub fn atanh_f64(x: f64) -> f64 {
        libm::atanh(x)
    }
    #[inline(always)]
    pub fn copysign_f32(x: f32, s: f32) -> f32 {
        libm::copysignf(x, s)
    }
    #[inline(always)]
    pub fn copysign_f64(x: f64, s: f64) -> f64 {
        libm::copysign(x, s)
    }
    #[inline(always)]
    pub fn powi_f32(x: f32, n: i32) -> f32 {
        libm::powf(x, n as f32)
    }
    #[inline(always)]
    pub fn powi_f64(x: f64, n: i32) -> f64 {
        libm::pow(x, n as f64)
    }
}

macro_rules! impl_scalar_float {
    ($t:ty, $suffix:ident, $zero:expr, $one:expr, $two:expr, $half:expr,
     $pi:expr, $tau:expr, $frac_pi_2:expr, $eps:expr, $inf:expr, $neg_inf:expr) => {
        ::paste::paste! {
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

            #[inline] fn sqrt(self) -> Self { float_ops::[<sqrt_ $suffix>](self) }
            #[inline] fn abs(self) -> Self { float_ops::[<abs_ $suffix>](self) }
            #[inline] fn sin(self) -> Self { float_ops::[<sin_ $suffix>](self) }
            #[inline] fn cos(self) -> Self { float_ops::[<cos_ $suffix>](self) }
            #[inline] fn tan(self) -> Self { float_ops::[<tan_ $suffix>](self) }
            #[inline] fn asin(self) -> Self { float_ops::[<asin_ $suffix>](self) }
            #[inline] fn acos(self) -> Self { float_ops::[<acos_ $suffix>](self) }
            #[inline] fn atan2(self, other: Self) -> Self { float_ops::[<atan2_ $suffix>](self, other) }
            #[inline] fn sin_cos(self) -> (Self, Self) { float_ops::[<sin_cos_ $suffix>](self) }
            #[inline] fn floor(self) -> Self { float_ops::[<floor_ $suffix>](self) }
            #[inline] fn ceil(self) -> Self { float_ops::[<ceil_ $suffix>](self) }
            #[inline] fn round(self) -> Self { float_ops::[<round_ $suffix>](self) }
            #[inline] fn exp(self) -> Self { float_ops::[<exp_ $suffix>](self) }
            #[inline] fn ln(self) -> Self { float_ops::[<ln_ $suffix>](self) }
            #[inline] fn powf(self, p: Self) -> Self { float_ops::[<powf_ $suffix>](self, p) }
            #[inline] fn tanh(self) -> Self { float_ops::[<tanh_ $suffix>](self) }
            #[inline] fn sinh(self) -> Self { float_ops::[<sinh_ $suffix>](self) }
            #[inline] fn cosh(self) -> Self { float_ops::[<cosh_ $suffix>](self) }
            #[inline] fn acosh(self) -> Self { float_ops::[<acosh_ $suffix>](self) }
            #[inline] fn asinh(self) -> Self { float_ops::[<asinh_ $suffix>](self) }
            #[inline] fn atanh(self) -> Self { float_ops::[<atanh_ $suffix>](self) }
            #[inline] fn copysign(self, sign: Self) -> Self { float_ops::[<copysign_ $suffix>](self, sign) }
            #[inline] fn powi(self, n: i32) -> Self { float_ops::[<powi_ $suffix>](self, n) }

            #[inline] fn min(self, other: Self) -> Self { if self < other { self } else { other } }
            #[inline] fn max(self, other: Self) -> Self { if self > other { self } else { other } }
            #[inline] fn clamp(self, lo: Self, hi: Self) -> Self {
                if self < lo { lo } else if self > hi { hi } else { self }
            }
            #[inline] fn recip(self) -> Self { 1.0 as $t / self }
            #[inline] fn signum(self) -> Self {
                if self > 0.0 as $t { 1.0 as $t } else if self < 0.0 as $t { -(1.0 as $t) } else { 0.0 as $t }
            }

            #[inline] fn from_f64(v: f64) -> Self { v as $t }
            #[inline] fn to_f64(self) -> f64 { self as f64 }
            #[inline] fn from_i32(v: i32) -> Self { v as $t }
        }
        }
    };
}

impl_scalar_float!(
    f32,
    f32,
    0.0,
    1.0,
    2.0,
    0.5,
    core::f32::consts::PI,
    core::f32::consts::TAU,
    core::f32::consts::FRAC_PI_2,
    f32::EPSILON,
    f32::INFINITY,
    f32::NEG_INFINITY
);
impl_scalar_float!(
    f64,
    f64,
    0.0,
    1.0,
    2.0,
    0.5,
    core::f64::consts::PI,
    core::f64::consts::TAU,
    core::f64::consts::FRAC_PI_2,
    f64::EPSILON,
    f64::INFINITY,
    f64::NEG_INFINITY
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f64_basics() {
        assert_eq!(f64::ZERO, 0.0);
        assert_eq!(f64::ONE, 1.0);
        assert!((f64::PI - core::f64::consts::PI).abs() < f64::EPSILON);
        assert_eq!(Scalar::sqrt(4.0_f64), 2.0);
        assert_eq!(Scalar::abs(-3.0_f64), 3.0);
    }

    #[test]
    fn f32_basics() {
        assert_eq!(f32::ZERO, 0.0);
        assert!((f32::PI - core::f32::consts::PI).abs() < f32::EPSILON);
    }
}
