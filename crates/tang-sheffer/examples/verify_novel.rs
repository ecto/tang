//! Multi-point verification of Phase 6 novel candidate operators.
//!
//! The Strategy-A search (`operator_search`) surfaces a handful of
//! operators that rival or match the hand-curated zoo on constant-free
//! reach or coverage. Those candidates were scored at budget 3×3 with
//! a single test point; any claim of "reaches target X at small depth"
//! could still be a Phase-4-style x-dependent numerical coincidence.
//!
//! This example re-runs the bootstrap for each novel candidate and
//! cross-checks every discovery at three independent transcendentals
//! {γ, A, G}. Survivors are the real deal.
//!
//! Run: `cargo run --release --example verify_novel -p tang-sheffer`

use std::collections::HashSet;
use std::f64::consts::{E, PI};
use std::sync::Arc;

use tang_sheffer::hp_verify::{check_discovery, LIMIT_THRESHOLD};
use tang_sheffer::op_enum::{Atom, BinaryOp, EnumOp, OpExpr, UnaryOp};
use tang_sheffer::{
    cross_check, standard_constants, standard_functions, Leaf, Operator, Verifier, C, TEST_POINT,
    TEST_POINTS,
};

const BUDGET: usize = 4;
const ITERS: usize = 5;

/// Novel candidates surfaced by `operator_search` at MAX_SIZE=5 that
/// rival PowSkew on constant-free coverage. Each is constructed by
/// hand here so we can name it and run multi-point checks.
fn candidates() -> Vec<(&'static str, OpExpr)> {
    let x = || Arc::new(OpExpr::Atom(Atom::X));
    let y = || Arc::new(OpExpr::Atom(Atom::Y));
    let one = || Arc::new(OpExpr::Atom(Atom::One));
    let neg_one = || Arc::new(OpExpr::Atom(Atom::NegOne));

    let sub = |a, b| Arc::new(OpExpr::Binary(BinaryOp::Sub, a, b));
    let div = |a, b| Arc::new(OpExpr::Binary(BinaryOp::Div, a, b));
    let add = |a, b| Arc::new(OpExpr::Binary(BinaryOp::Add, a, b));
    let powc = |a, b| Arc::new(OpExpr::Binary(BinaryOp::Pow, a, b));
    let neg = |a| Arc::new(OpExpr::Unary(UnaryOp::Neg, a));
    let inv = |a| Arc::new(OpExpr::Unary(UnaryOp::Inv, a));
    let sqrt = |a| Arc::new(OpExpr::Unary(UnaryOp::Sqrt, a));
    let sqr = |a| Arc::new(OpExpr::Unary(UnaryOp::Sqr, a));
    let exp = |a| Arc::new(OpExpr::Unary(UnaryOp::Exp, a));
    let ln_u = |a| Arc::new(OpExpr::Unary(UnaryOp::Ln, a));
    let sinh = |a| Arc::new(OpExpr::Unary(UnaryOp::Sinh, a));
    let tanh = |a| Arc::new(OpExpr::Unary(UnaryOp::Tanh, a));

    // Each entry clones the Arc root to own a plain OpExpr for EnumOp.
    let rc_to_owned = |rc: Arc<OpExpr>| (*rc).clone();

    vec![
        // Size-5 champions from the earlier run
        ("SubPow", rc_to_owned(powc(sub(x(), y()), y()))),
        ("OneMinusDiv", rc_to_owned(sub(one(), div(x(), y())))),
        ("NegDivSqrt", rc_to_owned(neg(div(x(), sqrt(y()))))),
        // Size-6 candidate
        ("SubPowInv", rc_to_owned(powc(sub(x(), y()), inv(y())))),
        // Size-7 candidates from MAX_SIZE=7 sweep
        //
        // RHPRep: "right-half-plane-representative"
        //   (1 - sqrt(sqr(y)) / x) — uses sqrt(sqr(y)) as branch-adjusted |y|
        (
            "RHPRep",
            rc_to_owned(sub(one(), div(sqrt(sqr(y())), x()))),
        ),
        //   ((x - y) / sqrt(sqr(y))) — asymmetric version
        (
            "DiffDivRHP",
            rc_to_owned(div(sub(x(), y()), sqrt(sqr(y())))),
        ),
        // Mobius: (x - 1) / (y - 1) — simple rational map, polynomial growth
        (
            "Mobius",
            rc_to_owned(div(sub(x(), one()), sub(y(), one()))),
        ),
        // Nested power: (x - y)^(y^x)
        (
            "NestedPow",
            rc_to_owned(powc(sub(x(), y()), powc(y(), x()))),
        ),
        // Suspicious pi-claimer from the unique-reach column
        (
            "SinhLnNeg1Pow",
            rc_to_owned(div(sinh(ln_u(powc(neg_one(), x()))), y())),
        ),
        // ---- MAX_SIZE=8 candidates ----
        // Top of ranked list: -((x - y)^(x^y))  scored 14/{1,x}, 13/{x}
        // If verified, this is the first operator to break 13 with {1,x}.
        (
            "NegNestedPow",
            rc_to_owned(neg(powc(sub(x(), y()), powc(x(), y())))),
        ),
        // ---- MAX_SIZE=9 candidates ----
        // Top candidate: e / (y - x) — a simple Möbius-ish operator
        // with e baked into the body. Scored 17/{1,x}, 16/{x}.
        (
            "EDivDiff",
            rc_to_owned(div(
                Arc::new(OpExpr::Atom(Atom::E)),
                sub(y(), x()),
            )),
        ),
        // `tanh(exp(e))^x - y/x` — rank 2, scored 19/15. Almost certainly
        // an FP artifact because tanh(exp(e)) = 1 - 1.65e-14 in f64, so
        // this is "approximately (1 - y/x) with a tiny exponent twist".
        {
            let tanh_u = |a| Arc::new(OpExpr::Unary(UnaryOp::Tanh, a));
            let exp_u = |a| Arc::new(OpExpr::Unary(UnaryOp::Exp, a));
            let e = Arc::new(OpExpr::Atom(Atom::E));
            (
                "TanhExpEPow",
                rc_to_owned(sub(
                    powc(tanh_u(exp_u(e)), x()),
                    div(y(), x()),
                )),
            )
        },
        // Sibling of DivMinusOne via reciprocal: 1/(x/y - 1)
        // Scored 13/13 at size 8, possibly a Group B rewrite.
        (
            "InvDivMinusOne",
            rc_to_owned(inv(sub(div(x(), y()), one()))),
        ),
        // sqrt(-1) * (ln(-x)^y) — claimed {pi, pi/2} in unique-reach.
        // Almost certainly a coincidence but check.
        (
            "SqrtNeg1LnNegPow",
            rc_to_owned(Arc::new(OpExpr::Binary(
                BinaryOp::Mul,
                sqrt(neg_one()),
                powc(ln_u(neg(x())), y()),
            ))),
        ),
        // Reference
        ("PowSkew", rc_to_owned(sub(powc(x(), y()), powc(y(), x())))),
    ]
}

fn target_at(name: &str, x: f64) -> Option<C> {
    let cx = C::new(x, 0.0);
    let v = match name {
        "0" => C::new(0.0, 0.0),
        "1" => C::new(1.0, 0.0),
        "-1" => C::new(-1.0, 0.0),
        "2" => C::new(2.0, 0.0),
        "-2" => C::new(-2.0, 0.0),
        "1/2" => C::new(0.5, 0.0),
        "e" => C::new(E, 0.0),
        "-e" => C::new(-E, 0.0),
        "1/e" => C::new(1.0 / E, 0.0),
        "e^2" => C::new(E * E, 0.0),
        "pi" => C::new(PI, 0.0),
        "pi/2" => C::new(PI / 2.0, 0.0),
        "2pi" => C::new(2.0 * PI, 0.0),
        "i" => C::new(0.0, 1.0),
        "-i" => C::new(0.0, -1.0),
        "i*pi" => C::new(0.0, PI),
        "x" => cx,
        "exp(x)" => cx.exp(),
        "ln(x)" => cx.ln(),
        "-x" => -cx,
        "1/x" => C::new(1.0, 0.0) / cx,
        "x^2" => cx * cx,
        "sqrt(x)" => cx.sqrt(),
        "x+1" => cx + C::new(1.0, 0.0),
        "x-1" => cx - C::new(1.0, 0.0),
        "2x" => cx * C::new(2.0, 0.0),
        "e*x" => cx * C::new(E, 0.0),
        "exp(exp(x))" => cx.exp().exp(),
        "ln(ln(x))" => cx.ln().ln(),
        "sin(x)" => cx.sin(),
        "cos(x)" => cx.cos(),
        _ => return None,
    };
    Some(v)
}

fn run_candidate(label: &str, expr: &OpExpr) {
    let mut targets = standard_constants();
    targets.extend(standard_functions());

    let op = EnumOp::new(expr.clone());

    println!("\n=== {} : {} ===", label, expr.pretty());

    // ---- {1, x} run ----
    let leaves = vec![
        Leaf::constant("1", C::new(1.0, 0.0)),
        Leaf::variable("x", C::new(TEST_POINT, 0.0)),
    ];
    let mut v = Verifier::new(leaves);
    let disc = v.bootstrap(&op, &targets, BUDGET, ITERS);

    let reports = cross_check(
        &disc,
        &op,
        &v.leaves,
        &TEST_POINTS,
        |name, x| target_at(name, x),
        1e-10,
    );
    // Standard multi-point verified set.
    let cp_verified: HashSet<String> = reports
        .iter()
        .filter(|r| r.passed)
        .map(|r| r.target_name.clone())
        .collect();

    // Now apply the structural FP-limit detector. For each discovery
    // that passed multi-point verification, re-evaluate with the
    // limit-aware evaluator; if any pow call in the full expansion
    // hit the `(1+ε)^(1/ε)` regime, flag as an artifact and drop
    // from the "genuine" set.
    let mut genuine_verified: HashSet<String> = HashSet::new();
    let mut artifact_hits: Vec<(String, C)> = Vec::new();
    for d in &disc {
        if !cp_verified.contains(&d.target_name) {
            continue;
        }
        let Some(target_val) = target_at(&d.target_name, TEST_POINT) else {
            continue;
        };
        let report = check_discovery(
            &d.expression,
            expr,
            &v.leaves,
            &d.target_name,
            target_val,
            LIMIT_THRESHOLD,
        );
        if report.is_artifact {
            artifact_hits.push((d.target_name.clone(), report.worst_base.unwrap_or_default()));
        } else {
            genuine_verified.insert(d.target_name.clone());
        }
    }

    let failed: Vec<&str> = reports
        .iter()
        .filter(|r| !r.passed)
        .map(|r| r.target_name.as_str())
        .collect();
    println!(
        "  {{1, x}}: {} hits → {} cross-verified → {} genuine (rejected MP: {})",
        disc.len(),
        cp_verified.len(),
        genuine_verified.len(),
        if failed.is_empty() {
            "none".into()
        } else {
            failed.join(", ")
        }
    );
    let mut sorted: Vec<&String> = genuine_verified.iter().collect();
    sorted.sort();
    println!(
        "    genuine: {}",
        sorted.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
    );
    if !artifact_hits.is_empty() {
        let names: Vec<&str> = artifact_hits.iter().map(|(n, _)| n.as_str()).collect();
        println!("    FP-limit artifacts: {}", names.join(", "));
    }

    // ---- {x} only (constant-free) ----
    let leaves = vec![Leaf::variable("x", C::new(TEST_POINT, 0.0))];
    let mut v = Verifier::new(leaves);
    let disc = v.bootstrap(&op, &targets, BUDGET, ITERS);

    let reports = cross_check(
        &disc,
        &op,
        &v.leaves,
        &TEST_POINTS,
        |name, x| target_at(name, x),
        1e-10,
    );
    let cp_verified_cf: HashSet<String> = reports
        .iter()
        .filter(|r| r.passed)
        .map(|r| r.target_name.clone())
        .collect();

    let mut genuine_cf: HashSet<String> = HashSet::new();
    let mut artifact_cf: Vec<String> = Vec::new();
    for d in &disc {
        if !cp_verified_cf.contains(&d.target_name) {
            continue;
        }
        let Some(target_val) = target_at(&d.target_name, TEST_POINT) else {
            continue;
        };
        let report = check_discovery(
            &d.expression,
            expr,
            &v.leaves,
            &d.target_name,
            target_val,
            LIMIT_THRESHOLD,
        );
        if report.is_artifact {
            artifact_cf.push(d.target_name.clone());
        } else {
            genuine_cf.insert(d.target_name.clone());
        }
    }

    let failed_cf: Vec<&str> = reports
        .iter()
        .filter(|r| !r.passed)
        .map(|r| r.target_name.as_str())
        .collect();
    println!(
        "  {{x}} only: {} hits → {} cross-verified → {} genuine constant-free (rejected MP: {})",
        disc.len(),
        cp_verified_cf.len(),
        genuine_cf.len(),
        if failed_cf.is_empty() {
            "none".into()
        } else {
            failed_cf.join(", ")
        }
    );
    let mut sorted: Vec<&String> = genuine_cf.iter().collect();
    sorted.sort();
    println!(
        "    genuine: {}",
        sorted.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
    );
    if !artifact_cf.is_empty() {
        println!("    FP-limit artifacts: {}", artifact_cf.join(", "));
    }
}

fn main() {
    println!("=== Phase 6: multi-point verification of novel candidates ===");
    println!("budget: {} ops × {} iters per run, tolerance 1e-10 at {{γ, A, G}}", BUDGET, ITERS);

    for (label, expr) in candidates() {
        run_candidate(label, &expr);
    }
}
