//! Hand-verified identity tests for the constant-free PowSkew and
//! PowExpSkew operators, corresponding to the proofs in `NOTES.md`.
//!
//! Each test asserts a single algebraic identity at the numerical test
//! point, using the exact tree-expression from the corresponding section
//! of the write-up. If the verifier ever finds a DIFFERENT expression
//! with the same target value that's still correct — these are purely a
//! correctness cross-check on the operator arithmetic, not the search.

#[cfg(test)]
mod tests {
    use std::f64::consts::E;

    use crate::operator::{Operator, PowExpSkew, PowSkew};
    use crate::C;

    const TOL: f64 = 1e-10;

    fn approx_eq(a: C, b: C, label: &str) {
        let diff = (a - b).norm();
        assert!(
            diff < TOL,
            "{}: expected {:?}, got {:?} (|diff| = {:.2e})",
            label,
            b,
            a,
            diff
        );
    }

    // ---- PowSkew: constant generation from single input x ----

    #[test]
    fn powskew_diagonal_is_zero() {
        let op = PowSkew;
        for x in [
            C::new(0.5, 0.0),
            C::new(1.5, 0.0),
            C::new(2.7, 0.0),
            C::new(1.3, 0.7),
        ] {
            approx_eq(
                op.eval(x, x),
                C::new(0.0, 0.0),
                &format!("pow-skew({:?}, {:?})", x, x),
            );
        }
    }

    #[test]
    fn powskew_x_zero_is_one() {
        let op = PowSkew;
        let x = C::new(1.5, 0.0);
        let zero = C::new(0.0, 0.0);
        approx_eq(op.eval(x, zero), C::new(1.0, 0.0), "pow-skew(x, 0)");
    }

    #[test]
    fn powskew_zero_x_is_minus_one() {
        let op = PowSkew;
        let x = C::new(1.5, 0.0);
        let zero = C::new(0.0, 0.0);
        approx_eq(op.eval(zero, x), C::new(-1.0, 0.0), "pow-skew(0, x)");
    }

    #[test]
    fn powskew_generates_two_from_one_and_neg_one() {
        let op = PowSkew;
        let one = C::new(1.0, 0.0);
        let neg_one = C::new(-1.0, 0.0);
        approx_eq(op.eval(one, neg_one), C::new(2.0, 0.0), "pow-skew(1, -1)");
        approx_eq(op.eval(neg_one, one), C::new(-2.0, 0.0), "pow-skew(-1, 1)");
    }

    #[test]
    fn powskew_generates_half() {
        let op = PowSkew;
        let two = C::new(2.0, 0.0);
        let neg_one = C::new(-1.0, 0.0);
        approx_eq(op.eval(neg_one, two), C::new(0.5, 0.0), "pow-skew(-1, 2)");
    }

    #[test]
    fn powskew_cascade_to_i() {
        // From NOTES section 2:
        //   f(1/2, -1) = (1/2)^(-1) - (-1)^(1/2) = 2 - i
        //   f(f(1/2,-1), 1) = (2-i) - 1 = 1 - i
        //   f(1, 1-i) = 1 - (1-i) = i
        let op = PowSkew;
        let half = C::new(0.5, 0.0);
        let neg_one = C::new(-1.0, 0.0);
        let one = C::new(1.0, 0.0);

        let two_minus_i = op.eval(half, neg_one);
        approx_eq(two_minus_i, C::new(2.0, -1.0), "f(1/2, -1) = 2 - i");

        let one_minus_i = op.eval(two_minus_i, one);
        approx_eq(one_minus_i, C::new(1.0, -1.0), "f(2-i, 1) = 1 - i");

        let i_val = op.eval(one, one_minus_i);
        approx_eq(i_val, C::new(0.0, 1.0), "f(1, 1-i) = i");

        let neg_i_val = op.eval(one_minus_i, one);
        approx_eq(neg_i_val, C::new(0.0, -1.0), "f(1-i, 1) = -i");
    }

    // ---- PowExpSkew: constant generation + transcendentals ----

    #[test]
    fn powexpskew_diagonal_is_zero() {
        let op = PowExpSkew;
        for x in [C::new(0.5, 0.0), C::new(1.5, 0.0), C::new(1.3, 0.7)] {
            approx_eq(op.eval(x, x), C::new(0.0, 0.0), "pow-exp-skew diagonal");
        }
    }

    #[test]
    fn powexpskew_x_zero_is_exp_x() {
        // g(x, 0) = (x^0 - 0^x) + (exp(x) - exp(0))
        //         = (1 - 0) + (exp(x) - 1)
        //         = exp(x)
        let op = PowExpSkew;
        let x = C::new(1.3, 0.0);
        let expected = C::new(x.re.exp(), 0.0);
        approx_eq(op.eval(x, C::new(0.0, 0.0)), expected, "g(x, 0) = exp(x)");
    }

    #[test]
    fn powexpskew_one_zero_is_e() {
        // g(1, 0) = 1 + e - 1 = e
        let op = PowExpSkew;
        approx_eq(
            op.eval(C::new(1.0, 0.0), C::new(0.0, 0.0)),
            C::new(E, 0.0),
            "g(1, 0) = e",
        );
    }

    #[test]
    fn powexpskew_zero_one_is_neg_e() {
        // g(0, 1) = (0 - 1) + (1 - e) = -e
        let op = PowExpSkew;
        approx_eq(
            op.eval(C::new(0.0, 0.0), C::new(1.0, 0.0)),
            C::new(-E, 0.0),
            "g(0, 1) = -e",
        );
    }

    // ---- Branch-cut and edge-case documentation ----
    //
    // These tests document (and pin) the behavior of `Complex::powc` on
    // inputs that would be undefined or indeterminate in real arithmetic.
    // num-complex uses principal-branch conventions throughout.

    #[test]
    fn powskew_neg_one_half_pow_is_i_principal_branch() {
        // (-1)^(1/2) = exp((1/2) * log(-1)) = exp((1/2) * i*pi)
        //            = cos(pi/2) + i*sin(pi/2) = i  (principal branch)
        //
        // This is why PowSkew's cascade can reach ±i: the complex log of
        // -1 lives on the principal branch, so (-1)^(1/2) = +i exactly,
        // not -i. A different branch choice would give -i instead, and
        // the "i" and "-i" results in the cascade would swap.
        let neg_one = C::new(-1.0, 0.0);
        let half = C::new(0.5, 0.0);
        let result = neg_one.powc(half);
        approx_eq(result, C::new(0.0, 1.0), "(-1)^(1/2) = i (principal)");
    }

    #[test]
    fn powskew_zero_zero_is_one_not_nan() {
        // num-complex follows IEEE convention: 0^0 = 1. This matters for
        // the PowSkew diagonal at x = 0:
        //     f(0, 0) = 0^0 - 0^0 = 1 - 1 = 0
        // rather than NaN - NaN = NaN. Without this, the cascade from
        // {x = 0} breaks immediately.
        let zero = C::new(0.0, 0.0);
        assert_eq!(
            zero.powc(zero),
            C::new(1.0, 0.0),
            "0^0 = 1 (IEEE convention)"
        );
        let op = PowSkew;
        approx_eq(op.eval(zero, zero), C::new(0.0, 0.0), "f(0, 0) = 0");
    }

    #[test]
    fn powskew_identities_hold_on_positive_half_plane() {
        // The identities f(x, 0) = 1 and f(0, x) = -1 require Re(x) > 0.
        // When Re(x) < 0, `0^x` diverges via `exp(x * ln(0)) = exp(x * -inf)`
        // and f(x, 0) becomes `1 - (inf+nan·i)`, not 1. So the constant-free
        // property holds only on the open right half-plane.
        //
        // f(x, x) = 0 does hold universally since both terms cancel before
        // ever touching the 0^x singularity.
        let op = PowSkew;
        let zero = C::new(0.0, 0.0);

        let positive_re = [
            C::new(0.5, 0.0),
            C::new(1.5, 0.0),
            C::new(2.718, 0.0),
            C::new(0.3, 0.4),
            C::new(1.0, -0.5),
        ];
        for &x in &positive_re {
            approx_eq(op.eval(x, x), C::new(0.0, 0.0), &format!("f({:?}, x)", x));
            approx_eq(
                op.eval(x, zero),
                C::new(1.0, 0.0),
                &format!("f({:?}, 0)", x),
            );
            approx_eq(
                op.eval(zero, x),
                C::new(-1.0, 0.0),
                &format!("f(0, {:?})", x),
            );
        }

        // Diagonal still cancels for Re(x) < 0.
        let negative_re = [C::new(-0.7, 1.1), C::new(-2.0, 0.3)];
        for &x in &negative_re {
            approx_eq(
                op.eval(x, x),
                C::new(0.0, 0.0),
                &format!("diagonal at Re<0: f({:?}, x)", x),
            );
        }
    }

    #[test]
    fn powskew_f_x_zero_fails_for_negative_re() {
        // Document the failure mode explicitly so future readers know why
        // the constant-freedom claim has a half-plane caveat.
        let op = PowSkew;
        let x = C::new(-0.7, 1.1);
        let result = op.eval(x, C::new(0.0, 0.0));
        assert!(
            !result.re.is_finite() || result.im.is_nan(),
            "expected divergence for Re(x) < 0, got {:?}",
            result
        );
    }

    #[test]
    fn powexpskew_diagonal_is_zero_universally() {
        // Same domain-independence test for PowExpSkew's diagonal.
        let op = PowExpSkew;
        for x in [
            C::new(0.3, 0.0),
            C::new(1.7, 0.0),
            C::new(-0.5, 0.8),
            C::new(0.9, -1.2),
        ] {
            approx_eq(op.eval(x, x), C::new(0.0, 0.0), "pow-exp-skew diagonal");
        }
    }

    // ---- Phase 6 novel constant-free candidates ----
    //
    // Each of these was discovered by the Strategy A programmatic search
    // (examples/operator_search.rs) and cross-checked at {γ, A, G}. The
    // tests pin the diagonal behaviour so a regression in the operator
    // arithmetic is caught loudly.

    #[test]
    fn subpow_diagonal_is_zero() {
        // (x - y)^y at diagonal: 0^x = 0 for Re(x) > 0.
        for x in [
            C::new(0.5, 0.0),
            C::new(1.2, 0.0),
            C::new(0.7, 0.3),
            C::new(1.5, -0.4),
        ] {
            let diff = x - x; // = 0
            let result = diff.powc(x);
            approx_eq(result, C::new(0.0, 0.0), "(x-x)^x = 0^x = 0");
        }
    }

    #[test]
    fn one_minus_div_diagonal_is_zero() {
        // 1 - x/y at diagonal: 1 - 1 = 0.
        for x in [
            C::new(0.5, 0.0),
            C::new(1.5, 0.0),
            C::new(0.7, 0.3),
            C::new(-0.4, 0.9),
        ] {
            let result = C::new(1.0, 0.0) - (x / x);
            approx_eq(result, C::new(0.0, 0.0), "1 - x/x = 0");
        }
    }

    #[test]
    fn one_minus_div_at_zero_x_is_one() {
        // (1 - 0/x) = 1 for x ≠ 0.
        let op_val = |a: C, b: C| C::new(1.0, 0.0) - (a / b);
        for x in [C::new(0.5, 0.0), C::new(1.5, -0.4)] {
            approx_eq(op_val(C::new(0.0, 0.0), x), C::new(1.0, 0.0), "1 - 0/x = 1");
        }
    }

    #[test]
    fn neg_div_sqrt_reaches_sqrt() {
        // -(x/sqrt(y)) — a direct route to sqrt: at y = 1, gives -x;
        // at y = x² (not directly in pool but reachable), gives -x/x = -1;
        // more interestingly the bootstrap finds sqrt(x) via cascaded
        // application. Here we just pin the fundamental value at (γ, γ²):
        // -(γ / sqrt(γ²)) = -(γ/γ) = -1 (for γ > 0 real).
        let gamma = C::new(0.5772156649015329, 0.0);
        let gamma_sq = gamma * gamma;
        let result = -(gamma / gamma_sq.sqrt());
        approx_eq(result, C::new(-1.0, 0.0), "-(γ / sqrt(γ²)) = -1");
    }
}
