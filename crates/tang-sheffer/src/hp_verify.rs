//! Structural floating-point artifact detection.
//!
//! The f64 multi-point verification at `{γ, A, G}` cannot distinguish
//! genuine Sheffer identities from `(1+ε)^(1/ε)`-class floating-point
//! limit artifacts (see `NOTES.md` §4d and §MAX_SIZE=9 epilogue).
//! Higher-precision arithmetic doesn't help either: the limit
//! `(1+ε)^(1/ε) → e` is scale-invariant, so shrinking ε from 1e-16 to
//! 1e-32 still produces e.
//!
//! This module catches the artifacts *structurally*. During an `Expr`
//! evaluation, we watch every `powc(base, exponent)` call. If
//!
//!   |base − 1| < ε_threshold   AND   |exponent| > 1 / ε_threshold
//!
//! at ANY pow step, the result is a limit-derived value that depends
//! on the exact machine-precision magnitude of `base − 1`. Real
//! algebraic identities don't sit at this knife-edge: either the base
//! is well away from 1, or the exponent is modest, or both.
//!
//! The threshold is a tunable knob. 1e-8 is aggressive (catches
//! everything at f64 precision); 1e-6 is conservative (only catches
//! blatant artifacts). We use 1e-8 by default.

use crate::expr::{Expr, Leaf, LeafSource};
use crate::op_enum::{BinaryOp, OpExpr, UnaryOp};
use crate::C;

/// Threshold for the `(1+ε)^(1/ε)` limit detector. Any pow call with
/// `|base − 1| < LIMIT_THRESHOLD` AND `|exponent| > 1 / LIMIT_THRESHOLD`
/// is flagged as an artifact.
pub const LIMIT_THRESHOLD: f64 = 1e-8;

/// Evaluator state: pristine result plus a flag indicating whether any
/// pow call fell inside the `(1+ε)^(1/ε)` limit regime.
#[derive(Debug, Clone)]
pub struct LimitAwareResult {
    pub value: C,
    /// True iff at least one `pow(base, exp)` call during this
    /// evaluation had `|base − 1| < threshold` AND `|exp| > 1/threshold`.
    pub limit_artifact: bool,
    /// The worst-offending (base, exponent) pair we saw, for reporting.
    pub worst_base: Option<C>,
    pub worst_exp: Option<C>,
    pub worst_product: f64,
}

impl LimitAwareResult {
    fn new(value: C) -> Self {
        Self {
            value,
            limit_artifact: false,
            worst_base: None,
            worst_exp: None,
            worst_product: 0.0,
        }
    }
}

fn combine_flags(a: &LimitAwareResult, b: &LimitAwareResult) -> (bool, Option<C>, Option<C>, f64) {
    let flag = a.limit_artifact || b.limit_artifact;
    if a.worst_product >= b.worst_product {
        (flag, a.worst_base, a.worst_exp, a.worst_product)
    } else {
        (flag, b.worst_base, b.worst_exp, b.worst_product)
    }
}

/// Evaluate an `OpExpr` at `(x, y)` while watching for `(1+ε)^(1/ε)`
/// limit-regime pow calls. Returns the final value plus the worst-seen
/// (base, exponent) product.
pub fn eval_opexpr_limit_aware(
    expr: &OpExpr,
    x: C,
    y: C,
    threshold: f64,
) -> LimitAwareResult {
    match expr {
        OpExpr::Atom(a) => LimitAwareResult::new(a.eval(x, y)),
        OpExpr::Unary(op, inner) => {
            let inner_r = eval_opexpr_limit_aware(inner, x, y, threshold);
            let v = apply_unary(*op, inner_r.value);
            LimitAwareResult {
                value: v,
                limit_artifact: inner_r.limit_artifact,
                worst_base: inner_r.worst_base,
                worst_exp: inner_r.worst_exp,
                worst_product: inner_r.worst_product,
            }
        }
        OpExpr::Binary(BinaryOp::Pow, l, r) => {
            let lr = eval_opexpr_limit_aware(l, x, y, threshold);
            let rr = eval_opexpr_limit_aware(r, x, y, threshold);

            // Check the limit condition.
            let base = lr.value;
            let exponent = rr.value;
            let dist_from_one = (base - C::new(1.0, 0.0)).norm();
            let exp_mag = exponent.norm();
            let product = dist_from_one * exp_mag;

            let (flag, wb, we, wp) = combine_flags(&lr, &rr);
            let mut result = LimitAwareResult {
                value: base.powc(exponent),
                limit_artifact: flag,
                worst_base: wb,
                worst_exp: we,
                worst_product: wp,
            };

            // Flag if |base - 1| < threshold AND |exp| > 1/threshold.
            // Equivalently: dist_from_one < threshold && exp_mag > 1/threshold.
            if dist_from_one < threshold
                && exp_mag > 1.0 / threshold
                && dist_from_one.is_finite()
                && exp_mag.is_finite()
            {
                if product > result.worst_product {
                    result.worst_product = product;
                    result.worst_base = Some(base);
                    result.worst_exp = Some(exponent);
                }
                result.limit_artifact = true;
            }

            result
        }
        OpExpr::Binary(op, l, r) => {
            let lr = eval_opexpr_limit_aware(l, x, y, threshold);
            let rr = eval_opexpr_limit_aware(r, x, y, threshold);
            let v = apply_binary(*op, lr.value, rr.value);
            let (flag, wb, we, wp) = combine_flags(&lr, &rr);
            LimitAwareResult {
                value: v,
                limit_artifact: flag,
                worst_base: wb,
                worst_exp: we,
                worst_product: wp,
            }
        }
    }
}

fn apply_unary(op: UnaryOp, v: C) -> C {
    match op {
        UnaryOp::Neg => -v,
        UnaryOp::Inv => C::new(1.0, 0.0) / v,
        UnaryOp::Sqr => v * v,
        UnaryOp::Sqrt => v.sqrt(),
        UnaryOp::Exp => v.exp(),
        UnaryOp::Ln => v.ln(),
        UnaryOp::Sin => v.sin(),
        UnaryOp::Cos => v.cos(),
        UnaryOp::Sinh => v.sinh(),
        UnaryOp::Tanh => v.tanh(),
    }
}

fn apply_binary(op: BinaryOp, a: C, b: C) -> C {
    match op {
        BinaryOp::Add => a + b,
        BinaryOp::Sub => a - b,
        BinaryOp::Mul => a * b,
        BinaryOp::Div => a / b,
        BinaryOp::Pow => a.powc(b),
    }
}

/// Evaluate a bootstrap-discovered `Expr` where each internal `Op`
/// node applies an `OpExpr` operator body. Propagates the limit-
/// artifact flag up the evaluation so if ANY pow call anywhere in the
/// computation hit the limit regime, the final result is flagged.
pub fn eval_bootstrap_limit_aware(
    expr: &Expr,
    op_body: &OpExpr,
    leaves: &[Leaf],
    threshold: f64,
) -> LimitAwareResult {
    match expr {
        Expr::Leaf(i) => match &leaves[*i].source {
            LeafSource::Derived { original } => {
                eval_bootstrap_limit_aware(original, op_body, leaves, threshold)
            }
            _ => LimitAwareResult::new(leaves[*i].value),
        },
        Expr::Op(a, b) => {
            let ar = eval_bootstrap_limit_aware(a, op_body, leaves, threshold);
            let br = eval_bootstrap_limit_aware(b, op_body, leaves, threshold);
            // Evaluate the operator body with x=ar.value, y=br.value,
            // watching for limit-regime pow calls inside the body.
            let body_r = eval_opexpr_limit_aware(op_body, ar.value, br.value, threshold);
            let (flag, wb, we, wp) = combine_flags(&ar, &br);
            let (flag, wb, we, wp) = {
                let b2 = LimitAwareResult {
                    value: body_r.value,
                    limit_artifact: flag,
                    worst_base: wb,
                    worst_exp: we,
                    worst_product: wp,
                };
                combine_flags(&b2, &body_r)
            };
            LimitAwareResult {
                value: body_r.value,
                limit_artifact: flag,
                worst_base: wb,
                worst_exp: we,
                worst_product: wp,
            }
        }
    }
}

/// Verdict on whether a (discovery, target) pair represents a genuine
/// algebraic identity or a floating-point limit artifact.
#[derive(Debug, Clone)]
pub struct ArtifactReport {
    pub target_name: String,
    pub target_value: C,
    pub computed_value: C,
    pub residual: f64,
    pub is_artifact: bool,
    pub worst_base: Option<C>,
    pub worst_exp: Option<C>,
}

/// Run the limit-aware evaluator over a bootstrap discovery and
/// return a verdict. `is_artifact = true` means at least one pow call
/// in the full computation was in the `(1+ε)^(1/ε)` limit regime.
pub fn check_discovery(
    expr: &Expr,
    op_body: &OpExpr,
    leaves: &[Leaf],
    target_name: &str,
    target_value: C,
    threshold: f64,
) -> ArtifactReport {
    let r = eval_bootstrap_limit_aware(expr, op_body, leaves, threshold);
    let residual = (r.value - target_value).norm();
    ArtifactReport {
        target_name: target_name.to_string(),
        target_value,
        computed_value: r.value,
        residual,
        is_artifact: r.limit_artifact,
        worst_base: r.worst_base,
        worst_exp: r.worst_exp,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::E;
    use std::sync::Arc;

    fn x() -> Arc<OpExpr> {
        Arc::new(OpExpr::Atom(Atom::X))
    }
    fn y() -> Arc<OpExpr> {
        Arc::new(OpExpr::Atom(Atom::Y))
    }
    fn neg(a: Arc<OpExpr>) -> Arc<OpExpr> {
        Arc::new(OpExpr::Unary(UnaryOp::Neg, a))
    }
    fn sub(a: Arc<OpExpr>, b: Arc<OpExpr>) -> Arc<OpExpr> {
        Arc::new(OpExpr::Binary(BinaryOp::Sub, a, b))
    }
    fn powc(a: Arc<OpExpr>, b: Arc<OpExpr>) -> Arc<OpExpr> {
        Arc::new(OpExpr::Binary(BinaryOp::Pow, a, b))
    }

    #[test]
    fn detector_catches_neg_nested_pow_e_chain() {
        // Manually chain NegNestedPow to reach e:
        //   f(f(f(f(1, 2), 1), -1), 0) with f = -((x-y)^(x^y))
        // The f(C2, -1) step is where the (1+ε)^(1/ε) limit fires.
        let body = neg(powc(sub(x(), y()), powc(x(), y())));
        let body = (*body).clone();

        let one = C::new(1.0, 0.0);
        let two = C::new(2.0, 0.0);
        let zero = C::new(0.0, 0.0);
        let neg_one = C::new(-1.0, 0.0);

        // C1 = f(1, 2)
        let c1 = eval_opexpr_limit_aware(&body, one, two, LIMIT_THRESHOLD);
        // C2 = f(1, C1)
        let c2 = eval_opexpr_limit_aware(&body, one, c1.value, LIMIT_THRESHOLD);
        // C3 = f(C2, -1)  ← limit fires here
        let c3 = eval_opexpr_limit_aware(&body, c2.value, neg_one, LIMIT_THRESHOLD);
        // e = f(C3, 0)
        let result = eval_opexpr_limit_aware(&body, c3.value, zero, LIMIT_THRESHOLD);

        // Pristine value is close to e (reproducing the artifact).
        assert!(
            (result.value - C::new(E, 0.0)).norm() < 1e-10,
            "pristine chain should reach e numerically, got {:?}",
            result.value
        );

        // But the detector should have fired on at least one of the
        // downstream steps (C3 or the final). We check C3 directly —
        // that's where the 1^(huge) limit happens.
        assert!(
            c3.limit_artifact,
            "C3 step should have triggered the limit detector. \
             worst_base = {:?}, worst_exp = {:?}",
            c3.worst_base,
            c3.worst_exp
        );
    }

    #[test]
    fn detector_does_not_flag_genuine_subpow_diagonal() {
        // Subpow's diagonal reaches 0 via (x-x)^x = 0^x = 0. The pow
        // call has base=0, exponent=x — base is FAR from 1 (|0−1|=1),
        // so the limit detector should NOT fire.
        let body = powc(sub(x(), y()), y());
        let body = (*body).clone();

        let x_val = C::new(0.5772, 0.0);
        let y_val = x_val; // diagonal
        let r = eval_opexpr_limit_aware(&body, x_val, y_val, LIMIT_THRESHOLD);
        assert!(
            !r.limit_artifact,
            "SubPow diagonal should not trigger the limit detector"
        );
        assert!(r.value.norm() < 1e-10, "expected 0, got {:?}", r.value);
    }

    #[test]
    fn detector_does_not_flag_eml_at_typical_input() {
        // EML = exp(x) - ln(y). No pow calls at all. Should not flag.
        let body = Arc::new(OpExpr::Binary(
            BinaryOp::Sub,
            Arc::new(OpExpr::Unary(UnaryOp::Exp, x())),
            Arc::new(OpExpr::Unary(UnaryOp::Ln, y())),
        ));
        let body = (*body).clone();

        let r = eval_opexpr_limit_aware(
            &body,
            C::new(1.5, 0.0),
            C::new(2.0, 0.0),
            LIMIT_THRESHOLD,
        );
        assert!(!r.limit_artifact);
    }
}
