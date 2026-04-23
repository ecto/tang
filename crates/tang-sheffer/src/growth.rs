//! Measure how fast an operator's output magnitude grows when iterated.
//!
//! Apply `f(x, x)` once, then `f(prev, prev)` repeatedly, and track
//! |prev|. Classify by fitting log-log slope: bounded, polynomial, or
//! exponential growth.
//!
//! A *polynomial*-growth operator is the goal from Strategy D of the research
//! prompt. The idea is that iterated |f| ≈ c · d^k where d is the level —
//! any such operator is a dramatic upgrade over EML's double-exponential.

use crate::operator::Operator;
use crate::C;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrowthClass {
    /// Sequence converges / stays bounded.
    Bounded,
    /// |f^k(x)| ~ O(k^p) for some finite p.
    Polynomial,
    /// |f^k(x)| ~ O(a^k) for some a > 1.
    Exponential,
    /// |f^k(x)| ~ O(a^(a^k)) — iterated exponential, the EML case.
    DoubleExponential,
    /// Overflowed to infinity within the test budget.
    Overflow,
    /// NaN'd out.
    Nan,
}

#[derive(Debug, Clone)]
pub struct GrowthProfile {
    pub magnitudes: Vec<f64>,
    pub classification: GrowthClass,
}

/// Apply the operator iteratively starting from (seed, seed). Report
/// magnitudes at each level and a classification based on the growth rate
/// of the *finite* portion of the sequence.
pub fn profile(op: &dyn Operator, seed: C, depth: usize) -> GrowthProfile {
    let mut magnitudes = Vec::with_capacity(depth + 1);
    magnitudes.push(seed.norm());

    let mut cur = seed;
    let mut overflowed = false;
    for _ in 0..depth {
        let next = op.eval(cur, cur);
        if next.re.is_nan() || next.im.is_nan() {
            magnitudes.push(f64::NAN);
            return GrowthProfile {
                classification: classify(&magnitudes),
                magnitudes,
            };
        }
        let m = next.norm();
        magnitudes.push(m);
        if !m.is_finite() || m > 1e300 {
            overflowed = true;
            break;
        }
        cur = next;
    }

    let mut class = classify(&magnitudes);
    if overflowed && matches!(class, GrowthClass::Polynomial | GrowthClass::Bounded) {
        // Overflowed but the finite prefix didn't confirm a higher class —
        // still, the overflow itself means at least exponential.
        class = GrowthClass::Exponential;
    }
    GrowthProfile {
        classification: class,
        magnitudes,
    }
}

/// Classify a magnitude sequence. Uses the longest finite prefix, looks at
/// the growth rate over the last 3 finite samples, and ranks by severity:
/// bounded < polynomial < exponential < double-exponential.
fn classify(m: &[f64]) -> GrowthClass {
    // Use only the finite, positive prefix — NaN / inf / 0 terminate the run.
    let finite: Vec<f64> = m
        .iter()
        .take_while(|v| v.is_finite() && **v > 1e-300)
        .copied()
        .collect();
    let n = finite.len();

    if n == 0 {
        return GrowthClass::Nan;
    }
    // Overflow if the original sequence had a non-finite step.
    if n < m.len() && !m[n].is_finite() {
        // At least one term exploded — classify by what we have below, then
        // upgrade at the caller.
    }
    if n < 3 {
        return GrowthClass::Polynomial; // too short to be sure; caller fixes up
    }

    let last = finite[n - 1];
    let prev = finite[n - 2];
    let prev2 = finite[n - 3];

    // Bounded: no meaningful change over the window.
    if (last / prev).abs() < 1.5 && (prev / prev2).abs() < 1.5 && last.max(prev).max(prev2) < 10.0 {
        return GrowthClass::Bounded;
    }

    // Double-exponential: log log grows super-linearly.
    if last > 10.0 && prev > 10.0 && prev2 > 10.0 {
        let ll = |v: f64| v.ln().ln();
        if ll(last).is_finite() && ll(prev).is_finite() && ll(prev2).is_finite() {
            let d1 = ll(last) - ll(prev);
            let d0 = ll(prev) - ll(prev2);
            if d1 > 0.3 && d0 > 0.3 && d1 > d0 * 0.7 {
                return GrowthClass::DoubleExponential;
            }
        }
    }

    // Exponential: log grows linearly.
    if last > 2.0 && prev > 2.0 {
        let d1 = last.ln() - prev.ln();
        let d0 = prev.ln() - prev2.ln();
        if d1 > 0.3 && d0 > 0.3 && d1 > d0 * 0.7 {
            return GrowthClass::Exponential;
        }
    }

    // Monotone growing but slower than exponential.
    if last > prev && prev > prev2 {
        return GrowthClass::Polynomial;
    }

    GrowthClass::Bounded
}
