use crate::Scalar;
use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Forward-mode automatic differentiation via dual numbers.
///
/// `Dual<S>` represents a value `a + bε` where ε² = 0.
/// The `real` part carries the function value, the `dual` part carries the derivative.
///
/// This is the key to differentiable physics: every operation through kern-math
/// automatically propagates gradients when `S = f64` and the outer type is `Dual<f64>`.
///
/// # Example
/// ```
/// use tang::{Dual, Vec3, Scalar};
///
/// // f(x) = x² at x = 3
/// let x = Dual::var(3.0_f64);
/// let y = x * x;
/// assert_eq!(y.real, 9.0);  // f(3) = 9
/// assert_eq!(y.dual, 6.0);  // f'(3) = 2*3 = 6
/// ```
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Dual<S> {
    pub real: S,
    pub dual: S,
}

impl<S: Scalar> Dual<S> {
    /// Constant (derivative = 0)
    #[inline]
    pub fn constant(real: S) -> Self {
        Self {
            real,
            dual: S::ZERO,
        }
    }

    /// Variable (derivative = 1)
    #[inline]
    pub fn var(real: S) -> Self {
        Self { real, dual: S::ONE }
    }

    /// Construct with explicit derivative
    #[inline]
    pub fn new(real: S, dual: S) -> Self {
        Self { real, dual }
    }
}

impl<S: Scalar> PartialEq for Dual<S> {
    fn eq(&self, other: &Self) -> bool {
        self.real == other.real
    }
}

impl<S: Scalar> PartialOrd for Dual<S> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.real.partial_cmp(&other.real)
    }
}

impl<S: Scalar> Default for Dual<S> {
    fn default() -> Self {
        Self::constant(S::ZERO)
    }
}

impl<S: Scalar> fmt::Display for Dual<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}+{}ε", self.real, self.dual)
    }
}

// Arithmetic: dual number rules
// (a + bε) + (c + dε) = (a+c) + (b+d)ε
// (a + bε) * (c + dε) = ac + (ad + bc)ε
// (a + bε) / (c + dε) = a/c + (bc - ad)/c²ε

impl<S: Scalar> Add for Dual<S> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            real: self.real + rhs.real,
            dual: self.dual + rhs.dual,
        }
    }
}

impl<S: Scalar> Sub for Dual<S> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            real: self.real - rhs.real,
            dual: self.dual - rhs.dual,
        }
    }
}

impl<S: Scalar> Mul for Dual<S> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            real: self.real * rhs.real,
            dual: self.real * rhs.dual + self.dual * rhs.real,
        }
    }
}

impl<S: Scalar> Div for Dual<S> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let inv = rhs.real.recip();
        Self {
            real: self.real * inv,
            dual: (self.dual * rhs.real - self.real * rhs.dual) * inv * inv,
        }
    }
}

impl<S: Scalar> Neg for Dual<S> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            real: -self.real,
            dual: -self.dual,
        }
    }
}

impl<S: Scalar> AddAssign for Dual<S> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.real += rhs.real;
        self.dual += rhs.dual;
    }
}

impl<S: Scalar> SubAssign for Dual<S> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.real -= rhs.real;
        self.dual -= rhs.dual;
    }
}

impl<S: Scalar> MulAssign for Dual<S> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        let new_dual = self.real * rhs.dual + self.dual * rhs.real;
        self.real *= rhs.real;
        self.dual = new_dual;
    }
}

impl<S: Scalar> DivAssign for Dual<S> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

/// Implement Scalar for Dual<S> — this is what makes all kern-math types
/// automatically differentiable.
impl<S: Scalar> Scalar for Dual<S> {
    const ZERO: Self = Dual {
        real: S::ZERO,
        dual: S::ZERO,
    };
    const ONE: Self = Dual {
        real: S::ONE,
        dual: S::ZERO,
    };
    const TWO: Self = Dual {
        real: S::TWO,
        dual: S::ZERO,
    };
    const HALF: Self = Dual {
        real: S::HALF,
        dual: S::ZERO,
    };
    const PI: Self = Dual {
        real: S::PI,
        dual: S::ZERO,
    };
    const TAU: Self = Dual {
        real: S::TAU,
        dual: S::ZERO,
    };
    const FRAC_PI_2: Self = Dual {
        real: S::FRAC_PI_2,
        dual: S::ZERO,
    };
    const EPSILON: Self = Dual {
        real: S::EPSILON,
        dual: S::ZERO,
    };
    const INFINITY: Self = Dual {
        real: S::INFINITY,
        dual: S::ZERO,
    };
    const NEG_INFINITY: Self = Dual {
        real: S::NEG_INFINITY,
        dual: S::ZERO,
    };

    // d/dx sqrt(x) = 1/(2*sqrt(x))
    #[inline]
    fn sqrt(self) -> Self {
        let r = self.real.sqrt();
        Dual {
            real: r,
            dual: self.dual / (S::TWO * r),
        }
    }

    // d/dx |x| = sign(x)
    #[inline]
    fn abs(self) -> Self {
        Dual {
            real: self.real.abs(),
            dual: self.dual * self.real.signum(),
        }
    }

    // d/dx sin(x) = cos(x)
    #[inline]
    fn sin(self) -> Self {
        Dual {
            real: self.real.sin(),
            dual: self.dual * self.real.cos(),
        }
    }

    // d/dx cos(x) = -sin(x)
    #[inline]
    fn cos(self) -> Self {
        Dual {
            real: self.real.cos(),
            dual: -self.dual * self.real.sin(),
        }
    }

    // d/dx tan(x) = 1/cos²(x)
    #[inline]
    fn tan(self) -> Self {
        let c = self.real.cos();
        Dual {
            real: self.real.tan(),
            dual: self.dual / (c * c),
        }
    }

    // d/dx asin(x) = 1/sqrt(1-x²)
    #[inline]
    fn asin(self) -> Self {
        Dual {
            real: self.real.asin(),
            dual: self.dual / (S::ONE - self.real * self.real).sqrt(),
        }
    }

    // d/dx acos(x) = -1/sqrt(1-x²)
    #[inline]
    fn acos(self) -> Self {
        Dual {
            real: self.real.acos(),
            dual: -self.dual / (S::ONE - self.real * self.real).sqrt(),
        }
    }

    // d/dx atan2(y,x) requires both partials
    #[inline]
    fn atan2(self, other: Self) -> Self {
        let denom = self.real * self.real + other.real * other.real;
        Dual {
            real: self.real.atan2(other.real),
            dual: (self.dual * other.real - self.real * other.dual) / denom,
        }
    }

    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        let (s, c) = self.real.sin_cos();
        (
            Dual {
                real: s,
                dual: self.dual * c,
            },
            Dual {
                real: c,
                dual: -self.dual * s,
            },
        )
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        if self.real < other.real {
            self
        } else {
            other
        }
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        if self.real > other.real {
            self
        } else {
            other
        }
    }

    #[inline]
    fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    #[inline]
    fn recip(self) -> Self {
        let inv = self.real.recip();
        Dual {
            real: inv,
            dual: -self.dual * inv * inv,
        }
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        let r = self.real.powi(n);
        Dual {
            real: r,
            dual: self.dual * S::from_i32(n) * self.real.powi(n - 1),
        }
    }

    #[inline]
    fn copysign(self, sign: Self) -> Self {
        let flipped = self.real.signum() != sign.real.signum();
        Dual {
            real: self.real.copysign(sign.real),
            dual: if flipped { -self.dual } else { self.dual },
        }
    }

    #[inline]
    fn signum(self) -> Self {
        Dual::constant(self.real.signum())
    }

    #[inline]
    fn floor(self) -> Self {
        Dual::constant(self.real.floor())
    }
    #[inline]
    fn ceil(self) -> Self {
        Dual::constant(self.real.ceil())
    }
    #[inline]
    fn round(self) -> Self {
        Dual::constant(self.real.round())
    }

    // d/dx exp(x) = exp(x)
    #[inline]
    fn exp(self) -> Self {
        let e = self.real.exp();
        Dual {
            real: e,
            dual: self.dual * e,
        }
    }

    // d/dx ln(x) = 1/x
    #[inline]
    fn ln(self) -> Self {
        Dual {
            real: self.real.ln(),
            dual: self.dual / self.real,
        }
    }

    // d/dx x^p = p * x^(p-1) * dx + x^p * ln(x) * dp
    #[inline]
    fn powf(self, p: Self) -> Self {
        let val = self.real.powf(p.real);
        Dual {
            real: val,
            dual: val * (p.dual * self.real.ln() + p.real * self.dual / self.real),
        }
    }

    // d/dx tanh(x) = 1 - tanh²(x)
    #[inline]
    fn tanh(self) -> Self {
        let t = self.real.tanh();
        Dual {
            real: t,
            dual: self.dual * (S::ONE - t * t),
        }
    }

    // d/dx sinh(x) = cosh(x)
    #[inline]
    fn sinh(self) -> Self {
        Dual {
            real: self.real.sinh(),
            dual: self.dual * self.real.cosh(),
        }
    }

    // d/dx cosh(x) = sinh(x)
    #[inline]
    fn cosh(self) -> Self {
        Dual {
            real: self.real.cosh(),
            dual: self.dual * self.real.sinh(),
        }
    }

    // d/dx acosh(x) = 1/sqrt(x²-1)
    #[inline]
    fn acosh(self) -> Self {
        Dual {
            real: self.real.acosh(),
            dual: self.dual / (self.real * self.real - S::ONE).sqrt(),
        }
    }

    // d/dx asinh(x) = 1/sqrt(x²+1)
    #[inline]
    fn asinh(self) -> Self {
        Dual {
            real: self.real.asinh(),
            dual: self.dual / (self.real * self.real + S::ONE).sqrt(),
        }
    }

    // d/dx atanh(x) = 1/(1-x²)
    #[inline]
    fn atanh(self) -> Self {
        Dual {
            real: self.real.atanh(),
            dual: self.dual / (S::ONE - self.real * self.real),
        }
    }

    #[inline]
    fn from_f64(v: f64) -> Self {
        Dual::constant(S::from_f64(v))
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self.real.to_f64()
    }
    #[inline]
    fn from_i32(v: i32) -> Self {
        Dual::constant(S::from_i32(v))
    }

    // Straight-through estimator: condition doesn't contribute gradient
    #[inline]
    fn select(cond: Self, a: Self, b: Self) -> Self {
        if cond.real > S::ZERO {
            a
        } else {
            b
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derivative_of_square() {
        let x = Dual::var(3.0_f64);
        let y = x * x;
        assert_eq!(y.real, 9.0);
        assert_eq!(y.dual, 6.0); // d/dx x² = 2x = 6
    }

    #[test]
    fn derivative_of_reciprocal() {
        let x = Dual::var(2.0_f64);
        let y = x.recip();
        assert!((y.real - 0.5).abs() < 1e-10);
        assert!((y.dual - (-0.25)).abs() < 1e-10); // d/dx 1/x = -1/x²
    }

    #[test]
    fn derivative_of_sqrt() {
        let x = Dual::var(4.0_f64);
        let y = x.sqrt();
        assert!((y.real - 2.0).abs() < 1e-10);
        assert!((y.dual - 0.25).abs() < 1e-10); // d/dx sqrt(x) = 1/(2*sqrt(x))
    }

    #[test]
    fn derivative_of_sin() {
        let x = Dual::var(0.0_f64);
        let y = x.sin();
        assert!(y.real.abs() < 1e-10); // sin(0) = 0
        assert!((y.dual - 1.0).abs() < 1e-10); // cos(0) = 1
    }

    #[test]
    fn chain_rule() {
        // d/dx sin(x²) = 2x * cos(x²)
        let x = Dual::var(1.0_f64);
        let x_sq = x * x;
        let y = x_sq.sin();
        let expected = 2.0 * 1.0_f64.cos(); // 2*1 * cos(1)
        assert!((y.dual - expected).abs() < 1e-10);
    }

    #[test]
    fn vec3_with_dual() {
        use crate::Vec3;
        // Derivative of norm([x, 0, 0]) = |x| → d/dx = sign(x) = 1
        let v = Vec3::new(Dual::var(3.0_f64), Dual::constant(0.0), Dual::constant(0.0));
        let n = v.norm();
        assert!((n.real - 3.0).abs() < 1e-10);
        assert!((n.dual - 1.0).abs() < 1e-10);
    }

    #[test]
    fn derivative_of_exp() {
        // d/dx exp(x) = exp(x)
        let x = Dual::var(1.0_f64);
        let y = x.exp();
        let e = 1.0_f64.exp();
        assert!((y.real - e).abs() < 1e-10);
        assert!((y.dual - e).abs() < 1e-10);
    }

    #[test]
    fn derivative_of_ln() {
        // d/dx ln(x) = 1/x
        let x = Dual::var(2.0_f64);
        let y = x.ln();
        assert!((y.real - 2.0_f64.ln()).abs() < 1e-10);
        assert!((y.dual - 0.5).abs() < 1e-10);
    }

    #[test]
    fn derivative_of_tanh() {
        // d/dx tanh(x) = 1 - tanh²(x) = sech²(x)
        let x = Dual::var(0.5_f64);
        let y = x.tanh();
        let t = 0.5_f64.tanh();
        assert!((y.real - t).abs() < 1e-10);
        assert!((y.dual - (1.0 - t * t)).abs() < 1e-10);
    }

    #[test]
    fn derivative_of_sinh_cosh() {
        // d/dx sinh(x) = cosh(x), d/dx cosh(x) = sinh(x)
        let x = Dual::var(1.0_f64);
        let s = x.sinh();
        let c = x.cosh();
        assert!((s.dual - 1.0_f64.cosh()).abs() < 1e-10);
        assert!((c.dual - 1.0_f64.sinh()).abs() < 1e-10);
    }

    #[test]
    fn derivative_of_acosh() {
        // d/dx acosh(x) = 1/sqrt(x²-1)
        let x = Dual::var(2.0_f64);
        let y = x.acosh();
        assert!((y.real - 2.0_f64.acosh()).abs() < 1e-10);
        let expected = 1.0 / (4.0_f64 - 1.0).sqrt();
        assert!((y.dual - expected).abs() < 1e-10);
    }

    #[test]
    fn derivative_of_asinh() {
        // d/dx asinh(x) = 1/sqrt(x²+1)
        let x = Dual::var(1.0_f64);
        let y = x.asinh();
        assert!((y.real - 1.0_f64.asinh()).abs() < 1e-10);
        let expected = 1.0 / (2.0_f64).sqrt();
        assert!((y.dual - expected).abs() < 1e-10);
    }

    #[test]
    fn copysign_flips_dual() {
        let x = Dual::var(3.0_f64);
        let s = Dual::constant(-1.0_f64);
        let y = x.copysign(s);
        assert_eq!(y.real, -3.0); // sign flipped
        assert_eq!(y.dual, -1.0); // dual negated because sign changed
    }

    #[test]
    fn copysign_no_flip() {
        let x = Dual::var(3.0_f64);
        let s = Dual::constant(1.0_f64);
        let y = x.copysign(s);
        assert_eq!(y.real, 3.0);
        assert_eq!(y.dual, 1.0); // dual unchanged
    }

    #[test]
    fn derivative_of_powf() {
        // d/dx x^3 = 3x^2 at x=2 -> 12
        let x = Dual::var(2.0_f64);
        let p = Dual::constant(3.0_f64);
        let y = x.powf(p);
        assert!((y.real - 8.0).abs() < 1e-10);
        assert!((y.dual - 12.0).abs() < 1e-8);
    }
}
