//! Multi-test-point cross-check.
//!
//! A single numerical match at one transcendental test point could in
//! principle be a coincidence. Schanuel's conjecture says two (or three)
//! algebraically independent transcendentals, matched simultaneously,
//! effectively rule out coincidences — if a Rust f64 expression over a
//! fixed operator matches the target at γ AND at Glaisher-Kinkelin A AND
//! at Catalan's constant G, all within tolerance 1e-10, the identity is
//! almost certainly real.
//!
//! This module takes a set of `Discovery` results from a single-point run
//! and verifies each one by re-evaluating the discovered expression with
//! new leaf values (corresponding to the new test point) and comparing to
//! the target's value at the same new point.

use crate::expr::{Leaf, LeafSource};
use crate::operator::Operator;
use crate::verify::Discovery;
use crate::C;

/// Three algebraically-independent transcendental test points. These are
/// believed (by Schanuel's conjecture) to be mutually transcendental, so a
/// triple numerical match at all of them implies a true identity modulo the
/// conjecture.
pub const TEST_POINTS: [f64; 3] = [
    0.5772156649015329,  // Euler-Mascheroni γ
    1.2824271291006226,  // Glaisher-Kinkelin A
    0.9159655941772190,  // Catalan's constant G
];

/// Swap the numeric value of each Variable leaf in `leaves` to the new x
/// value. Constants stay fixed. Derived leaves are marked stale by zeroing
/// their `value` field so any accidental direct `eval` access fails loudly;
/// correct clients must use `eval_recursive`.
pub fn rebind_leaves(leaves: &[Leaf], new_x: f64) -> Vec<Leaf> {
    leaves
        .iter()
        .map(|leaf| match leaf.source {
            LeafSource::Variable => Leaf {
                name: leaf.name.clone(),
                value: C::new(new_x, 0.0),
                source: LeafSource::Variable,
            },
            LeafSource::Constant => leaf.clone(),
            LeafSource::Derived { .. } => Leaf {
                name: leaf.name.clone(),
                // Zero is a deliberately-wrong sentinel — callers must use
                // eval_recursive which recomputes from `original`.
                value: C::new(0.0, 0.0),
                source: leaf.source.clone(),
            },
        })
        .collect()
}

/// Result of cross-checking a single discovery across multiple test points.
#[derive(Debug, Clone)]
pub struct CrossCheckReport {
    pub target_name: String,
    pub size: usize,
    /// For each test point: (computed value, expected value, residual)
    pub points: Vec<(C, C, f64)>,
    pub passed: bool,
}

/// For each discovery in `discoveries`, re-evaluate at each test point and
/// compare to the target value computed at that point by `target_fn`.
/// Returns one `CrossCheckReport` per discovery.
pub fn cross_check<TargetFn>(
    discoveries: &[Discovery],
    op: &dyn Operator,
    base_leaves: &[Leaf],
    test_points: &[f64],
    mut target_fn: TargetFn,
    tol: f64,
) -> Vec<CrossCheckReport>
where
    TargetFn: FnMut(&str, f64) -> Option<C>,
{
    let mut out = Vec::with_capacity(discoveries.len());
    for d in discoveries {
        let mut report = CrossCheckReport {
            target_name: d.target_name.clone(),
            size: d.expanded_size,
            points: Vec::with_capacity(test_points.len()),
            passed: true,
        };

        for &x in test_points {
            let leaves = rebind_leaves(base_leaves, x);
            let computed = d.expression.eval_recursive(op, &leaves);
            let expected = match target_fn(&d.target_name, x) {
                Some(v) => v,
                None => {
                    report.passed = false;
                    report.points.push((computed, C::new(f64::NAN, 0.0), f64::NAN));
                    continue;
                }
            };
            let residual = (computed - expected).norm();
            if !(residual < tol) {
                report.passed = false;
            }
            report.points.push((computed, expected, residual));
        }

        out.push(report);
    }
    out
}
