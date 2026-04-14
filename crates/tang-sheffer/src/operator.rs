//! Binary operators over complex values.
//!
//! The search for a "better than EML" operator proceeds by systematic
//! replacement of EML's structural components:
//!   - growth primitive (exp) → alternatives (sinh, tan, x², pow, Bessel-ish)
//!   - inverse primitive (ln) → their corresponding inverses (arsinh, arctan,
//!     sqrt, log)
//!   - combiner (−) → alternatives (/, +, pow, etc.)
//!
//! Each concrete candidate here implements `Operator` so the verifier can
//! plug it into bootstrap without further ceremony.

use crate::C;

/// A binary operator on complex values.
pub trait Operator: Send + Sync {
    fn name(&self) -> &str;
    fn eval(&self, x: C, y: C) -> C;
}

// -- EML family (paper-standard) ---------------------------------------------

/// EML: `exp(x) − ln(y)`. Paper-proven universal with constant 1. Baseline.
#[derive(Debug, Clone, Copy)]
pub struct Eml;
impl Operator for Eml {
    fn name(&self) -> &str {
        "eml"
    }
    fn eval(&self, x: C, y: C) -> C {
        x.exp() - y.ln()
    }
}

/// EDL: `exp(x) / ln(y)`. Paper's variant, universal with constant e.
#[derive(Debug, Clone, Copy)]
pub struct Edl;
impl Operator for Edl {
    fn name(&self) -> &str {
        "edl"
    }
    fn eval(&self, x: C, y: C) -> C {
        x.exp() / y.ln()
    }
}

// -- Hyperbolic analogs ------------------------------------------------------

/// `sinh(x) − arsinh(y)`. sinh grows like exp/2 for large x but is odd and
/// defined on all of C without branch cut issues that plague ln.
#[derive(Debug, Clone, Copy)]
pub struct SinhAsinh;
impl Operator for SinhAsinh {
    fn name(&self) -> &str {
        "sinh-asinh"
    }
    fn eval(&self, x: C, y: C) -> C {
        x.sinh() - y.asinh()
    }
}

/// `cosh(x) − acosh(y)`. cosh is even, so this has a different symmetry
/// profile. acosh has a branch cut at (-∞, 1).
#[derive(Debug, Clone, Copy)]
pub struct CoshAcosh;
impl Operator for CoshAcosh {
    fn name(&self) -> &str {
        "cosh-acosh"
    }
    fn eval(&self, x: C, y: C) -> C {
        x.cosh() - y.acosh()
    }
}

/// `tanh(x) − artanh(y)`. Both bounded in the unit interval on the real line;
/// very benign growth but small reachable set.
#[derive(Debug, Clone, Copy)]
pub struct TanhAtanh;
impl Operator for TanhAtanh {
    fn name(&self) -> &str {
        "tanh-atanh"
    }
    fn eval(&self, x: C, y: C) -> C {
        x.tanh() - y.atanh()
    }
}

// -- Trig analogs ------------------------------------------------------------

/// `sin(x) − arcsin(y)`. Periodic + inverse; arcsin has branch cuts outside
/// [-1, 1]. Bounded for real inputs but unbounded on complex plane.
#[derive(Debug, Clone, Copy)]
pub struct SinAsin;
impl Operator for SinAsin {
    fn name(&self) -> &str {
        "sin-asin"
    }
    fn eval(&self, x: C, y: C) -> C {
        x.sin() - y.asin()
    }
}

/// `tan(x) − arctan(y)`. tan has period π and poles; arctan is smooth and
/// bounded on the real line (useful for containing growth).
#[derive(Debug, Clone, Copy)]
pub struct TanAtan;
impl Operator for TanAtan {
    fn name(&self) -> &str {
        "tan-atan"
    }
    fn eval(&self, x: C, y: C) -> C {
        x.tan() - y.atan()
    }
}

// -- Algebraic (polynomial-growth candidates — Strategy D) -------------------

/// `x² − sqrt(y)`. Pure algebraic: polynomial growth (squaring doubles
/// magnitude per level, not exponentiates). Paper-noted as likely incomplete,
/// but a baseline for polynomial-only operators.
#[derive(Debug, Clone, Copy)]
pub struct SqrSqrt;
impl Operator for SqrSqrt {
    fn name(&self) -> &str {
        "sqr-sqrt"
    }
    fn eval(&self, x: C, y: C) -> C {
        x * x - y.sqrt()
    }
}

/// `x² / sqrt(y)`. Division variant of SqrSqrt.
#[derive(Debug, Clone, Copy)]
pub struct SqrDivSqrt;
impl Operator for SqrDivSqrt {
    fn name(&self) -> &str {
        "sqr/sqrt"
    }
    fn eval(&self, x: C, y: C) -> C {
        (x * x) / y.sqrt()
    }
}

/// `x^y − y^x`. Pure `pow` operator — both growth and inverse implicit in
/// pow. Antisymmetric (zero on the diagonal x=y), which makes f(x,x)=0
/// automatically. Strategy D polynomial-growth candidate.
#[derive(Debug, Clone, Copy)]
pub struct PowSkew;
impl Operator for PowSkew {
    fn name(&self) -> &str {
        "pow-skew"
    }
    fn eval(&self, x: C, y: C) -> C {
        x.powc(y) - y.powc(x)
    }
}

/// `x^y / y^x`. Multiplicative sibling of PowSkew. Identity on x=y is 1.
#[derive(Debug, Clone, Copy)]
pub struct PowRatio;
impl Operator for PowRatio {
    fn name(&self) -> &str {
        "pow-ratio"
    }
    fn eval(&self, x: C, y: C) -> C {
        x.powc(y) / y.powc(x)
    }
}

// -- Mixed-family variants ---------------------------------------------------

/// `exp(x) − sqrt(y)`. Exp for growth, sqrt for inversion (instead of ln).
/// Keeps exp's transcendental reach but swaps the inverse component.
#[derive(Debug, Clone, Copy)]
pub struct ExpMinusSqrt;
impl Operator for ExpMinusSqrt {
    fn name(&self) -> &str {
        "exp-sqrt"
    }
    fn eval(&self, x: C, y: C) -> C {
        x.exp() - y.sqrt()
    }
}

/// `sinh(x) − ln(y)`. Same combiner as EML, swap exp→sinh. sinh is odd so
/// f(0, y) = -ln(y), giving us `ln` almost directly.
#[derive(Debug, Clone, Copy)]
pub struct SinhLn;
impl Operator for SinhLn {
    fn name(&self) -> &str {
        "sinh-ln"
    }
    fn eval(&self, x: C, y: C) -> C {
        x.sinh() - y.ln()
    }
}

// -- Constant-free candidates (antisymmetric under swap) --------------------
//
// If f(x, x) = constant for all x, then the operator generates its own
// constants without a distinguished leaf. Antisymmetric operators f(x, y) =
// g(x, y) - g(y, x) automatically have f(x, x) = 0, giving us 0 for free.

/// `exp(x − y) − 1`. Antisymmetric-like: f(x, x) = 0 always. Transcendental
/// reach via exp (f(x, 0) = exp(x) - 1) while remaining constant-free.
#[derive(Debug, Clone, Copy)]
pub struct ExpDiff;
impl Operator for ExpDiff {
    fn name(&self) -> &str {
        "exp-diff"
    }
    fn eval(&self, x: C, y: C) -> C {
        (x - y).exp() - C::new(1.0, 0.0)
    }
}

/// `sinh(x − y)`. Odd function of (x-y), so f(x, x) = 0. Transcendental
/// via sinh; less explosive growth than exp for moderate inputs.
#[derive(Debug, Clone, Copy)]
pub struct SinhDiff;
impl Operator for SinhDiff {
    fn name(&self) -> &str {
        "sinh-diff"
    }
    fn eval(&self, x: C, y: C) -> C {
        (x - y).sinh()
    }
}

/// `tan(x − y)`. f(x, x) = 0. Tan has periodicity and poles — introduces
/// singular structure that may reach π naturally.
#[derive(Debug, Clone, Copy)]
pub struct TanDiff;
impl Operator for TanDiff {
    fn name(&self) -> &str {
        "tan-diff"
    }
    fn eval(&self, x: C, y: C) -> C {
        (x - y).tan()
    }
}

/// `ln(1 + x − y)`. f(x, x) = ln(1) = 0. Has a singularity at x = y - 1
/// but is otherwise well-behaved on moderate inputs.
#[derive(Debug, Clone, Copy)]
pub struct LnDiff;
impl Operator for LnDiff {
    fn name(&self) -> &str {
        "ln-diff"
    }
    fn eval(&self, x: C, y: C) -> C {
        (C::new(1.0, 0.0) + x - y).ln()
    }
}

/// `x^y − y` (not antisymmetric, but f(x, 1) = x - 1 naturally). Uses pow
/// for growth with plain subtraction as the combiner.
#[derive(Debug, Clone, Copy)]
pub struct PowMinus;
impl Operator for PowMinus {
    fn name(&self) -> &str {
        "pow-minus"
    }
    fn eval(&self, x: C, y: C) -> C {
        x.powc(y) - y
    }
}

/// `(x^y − y^x) + (exp(x) − exp(y))`. Combines the constant-free algebraic
/// reach of PowSkew with transcendental reach via exp. Fully antisymmetric
/// so f(x, x) = 0 for any x, making it constant-free. Critically,
/// f(x, 0) = 1 + exp(x) − 1 = exp(x) — a one-op shortcut to exp(x) once 0
/// is generated.
#[derive(Debug, Clone, Copy)]
pub struct PowExpSkew;
impl Operator for PowExpSkew {
    fn name(&self) -> &str {
        "pow-exp-skew"
    }
    fn eval(&self, x: C, y: C) -> C {
        x.powc(y) - y.powc(x) + x.exp() - y.exp()
    }
}

/// `(x^y − y^x) + (ln(x) − ln(y))`. Log variant: f(x, 0) has ln(0) = -∞ so
/// hits the IEEE infinity edge. Stays antisymmetric and constant-free.
#[derive(Debug, Clone, Copy)]
pub struct PowLnSkew;
impl Operator for PowLnSkew {
    fn name(&self) -> &str {
        "pow-ln-skew"
    }
    fn eval(&self, x: C, y: C) -> C {
        x.powc(y) - y.powc(x) + x.ln() - y.ln()
    }
}
