//! Score an enumerated operator against the standard target set.
//!
//! `score` runs the existing `Verifier::bootstrap` twice for each
//! candidate — once with leaves `{1, x}` (the normal Sheffer setup) and
//! once with leaves `{x}` only (the constant-free version) — and rolls
//! up:
//!
//!   - coverage on each pool
//!   - growth classification on a generic complex seed
//!   - smallest expanded-size expression reaching `ln(x)` (a
//!     representative hard target)
//!   - targets uniquely reached relative to a baseline (EML / PowSkew)
//!
//! One `score` call completes in milliseconds for small budgets, so
//! scoring ~5 k candidates serially is minutes of CPU.

use std::collections::HashSet;

use crate::expr::Leaf;
use crate::growth::{profile, GrowthClass};
use crate::op_enum::OpExpr;
use crate::operator::Operator;
use crate::targets::{standard_constants, standard_functions, Target, TEST_POINT};
use crate::verify::{Discovery, Verifier};
use crate::C;

/// Scoring budget — kept small so 5k+ candidates finish in minutes. Can
/// be bumped at the top of the example if the first run is fast.
pub const SCORE_BUDGET: usize = 3;
pub const SCORE_ITERS: usize = 3;

/// A single operator's evaluation on the standard benchmark. Size is
/// the `OpExpr` node count; growth is measured at a generic complex seed;
/// coverages are bootstrap target counts against the 31 standard targets.
#[derive(Debug, Clone)]
pub struct Scorecard {
    pub op_name: String,
    pub op_size: usize,
    pub growth: GrowthClass,

    pub with_const_coverage: usize,
    pub const_free_coverage: usize,

    pub ln_x_expanded_size: Option<usize>,
    pub unique_targets: Vec<String>,

    /// Target name set reached with `{1, x}` pool.
    pub with_const_targets: HashSet<String>,
    /// Target name set reached with `{x}` pool alone.
    pub const_free_targets: HashSet<String>,
}

/// Compute the combined target set (constants + functions) used for all
/// scoring. Kept in a helper so every `score` call shares the same list.
pub fn standard_all_targets() -> Vec<Target> {
    let mut v = standard_constants();
    v.extend(standard_functions());
    v
}

fn reached_names(disc: &[Discovery]) -> HashSet<String> {
    disc.iter().map(|d| d.target_name.clone()).collect()
}

fn min_ln_x(disc: &[Discovery]) -> Option<usize> {
    disc.iter()
        .filter(|d| d.target_name == "ln(x)")
        .map(|d| d.expanded_size)
        .min()
}

/// Score a candidate operator. `baseline` is the union of target names
/// reached by the hand-curated operators at the same budget (EML ∪
/// PowSkew ∪ EDL, typically) — `unique_targets` is filled with the
/// candidate's hits that don't appear in the baseline.
pub fn score(op: &dyn Operator, expr: &OpExpr, baseline: &HashSet<String>) -> Scorecard {
    score_generic(op, expr.size(), baseline)
}

/// Lower-level scoring entry point: takes the operator and its node-
/// count size directly. Works for any `Operator` implementation —
/// used by both `OpExpr`-based and arena-based candidates.
pub fn score_generic(op: &dyn Operator, op_size: usize, baseline: &HashSet<String>) -> Scorecard {
    let targets = standard_all_targets();

    // {1, x} bootstrap
    let leaves = vec![
        Leaf::constant("1", C::new(1.0, 0.0)),
        Leaf::variable("x", C::new(TEST_POINT, 0.0)),
    ];
    let mut v = Verifier::new(leaves);
    let disc = v.bootstrap(op, &targets, SCORE_BUDGET, SCORE_ITERS);
    let wc_names = reached_names(&disc);
    let ln_depth = min_ln_x(&disc);

    // {x} only bootstrap
    let leaves = vec![Leaf::variable("x", C::new(TEST_POINT, 0.0))];
    let mut v = Verifier::new(leaves);
    let disc_cf = v.bootstrap(op, &targets, SCORE_BUDGET, SCORE_ITERS);
    let cf_names = reached_names(&disc_cf);

    // Growth: generic complex seed to avoid accidental fixed points of
    // the diagonal.
    let seed = C::new(1.3, 0.7);
    let growth = profile(op, seed, 6).classification;

    let combined: HashSet<String> = wc_names.union(&cf_names).cloned().collect();
    let unique: Vec<String> = combined.difference(baseline).cloned().collect();

    Scorecard {
        op_name: op.name().to_string(),
        op_size,
        growth,
        with_const_coverage: wc_names.len(),
        const_free_coverage: cf_names.len(),
        ln_x_expanded_size: ln_depth,
        unique_targets: unique,
        with_const_targets: wc_names,
        const_free_targets: cf_names,
    }
}

/// Ranking key: primary = with_const_coverage + 2 * const_free_coverage,
/// tiebreaks are smaller size, then non-double-exp growth preferred.
pub fn ranking_key(s: &Scorecard) -> (i64, i64, i64) {
    let primary = -(s.with_const_coverage as i64 + 2 * s.const_free_coverage as i64);
    let size = s.op_size as i64;
    let growth_penalty = match s.growth {
        GrowthClass::DoubleExponential => 3,
        GrowthClass::Exponential => 2,
        GrowthClass::Polynomial => 1,
        GrowthClass::Bounded => 0,
        GrowthClass::Overflow => 3,
        GrowthClass::Nan => 4,
    };
    (primary, size, growth_penalty)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op_enum::{Atom, BinaryOp, EnumOp, OpExpr, UnaryOp};

    #[test]
    fn score_eml_matches_phase_5_baseline() {
        use std::sync::Arc;
        // Build exp(x) - ln(y) as an OpExpr, wrap in EnumOp, score it.
        // Should reproduce approximately the Phase-4 EML coverage of
        // ~21 targets at budget 3 × 3 iters.
        let eml = OpExpr::Binary(
            BinaryOp::Sub,
            Arc::new(OpExpr::Unary(UnaryOp::Exp, Arc::new(OpExpr::Atom(Atom::X)))),
            Arc::new(OpExpr::Unary(UnaryOp::Ln, Arc::new(OpExpr::Atom(Atom::Y)))),
        );
        let op = EnumOp::new(eml.clone());
        let baseline: HashSet<String> = HashSet::new();
        let card = score(&op, &eml, &baseline);

        // We don't assert exact numbers because budget 3 × 3 iters is
        // smaller than Phase 5's budget 4 × 6 iters; just check we reach
        // a reasonable number with the {1,x} pool.
        assert!(
            card.with_const_coverage >= 5,
            "EML {{1,x}} coverage at budget 3×3: {}",
            card.with_const_coverage
        );
    }
}
