//! Phase 4: rigorous verification of the Phase 2 discoveries.
//!
//! Two independent checks per candidate operator:
//!
//! (A) Multi-point cross-check — every discovery is re-evaluated at three
//!     conjecturally-algebraically-independent transcendentals
//!     {γ, Glaisher-Kinkelin A, Catalan's constant G}. An expression that
//!     matches the target at one point could in principle be a numerical
//!     coincidence; matches at three points are — modulo Schanuel — not.
//!
//! (B) Minimal-depth characterization — sweep bootstrap budget ∈ {2..5}
//!     and record the smallest expression reaching each target per
//!     operator. Produces a Table-4-style comparison.
//!
//! Run: `cargo run --release --example phase4_verify -p tang-sheffer`

use std::collections::BTreeMap;
use std::f64::consts::{E, PI};

use tang_sheffer::{
    cross_check, standard_constants, standard_functions, CrossCheckReport, Eml, Leaf, Operator,
    PowExpSkew, PowSkew, Verifier, C, TEST_POINT, TEST_POINTS,
};

/// Compute the expected value of a named target at an arbitrary test point
/// `x`. For constants, the test point is irrelevant. For functions of x,
/// the test point is the x value.
fn target_at(name: &str, x: f64) -> Option<C> {
    let cx = C::new(x, 0.0);
    let v = match name {
        // Constants — independent of x
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
        // Functions of x — depend on test point
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

fn report_crosscheck(label: &str, reports: &[CrossCheckReport]) {
    let passed = reports.iter().filter(|r| r.passed).count();
    println!(
        "  {:<12} {:>3}/{:<3} pass 3-point check",
        label,
        passed,
        reports.len()
    );

    // Show any failures — they're the most interesting rows.
    let fails: Vec<&CrossCheckReport> = reports.iter().filter(|r| !r.passed).collect();
    if !fails.is_empty() {
        println!("    failed at ≥1 test point:");
        for r in fails {
            let residuals: Vec<String> = r
                .points
                .iter()
                .map(|(_, _, res)| {
                    if res.is_nan() {
                        "—".into()
                    } else {
                        format!("{:.1e}", res)
                    }
                })
                .collect();
            println!(
                "      {:>10} (size {}) — residuals [{}]",
                r.target_name,
                r.size,
                residuals.join(", ")
            );
        }
    }
}

fn main() {
    println!("=== Phase 4: multi-point verification + depth table ===\n");

    let mut all_targets = standard_constants();
    all_targets.extend(standard_functions());

    let ops: Vec<(&str, Box<dyn Operator>, Vec<Leaf>)> = vec![
        (
            "EML",
            Box::new(Eml),
            vec![
                Leaf::constant("1", C::new(1.0, 0.0)),
                Leaf::variable("x", C::new(TEST_POINT, 0.0)),
            ],
        ),
        (
            "PowSkew",
            Box::new(PowSkew),
            vec![Leaf::variable("x", C::new(TEST_POINT, 0.0))],
        ),
        (
            "PowExpSkew",
            Box::new(PowExpSkew),
            vec![Leaf::variable("x", C::new(TEST_POINT, 0.0))],
        ),
    ];

    // -- Part A: multi-point cross-check --
    println!("Part A — cross-check at 3 independent transcendentals {{γ, A, G}}");
    println!(
        "  test points: γ={:.4}, A={:.4}, G={:.4}",
        TEST_POINTS[0], TEST_POINTS[1], TEST_POINTS[2]
    );
    println!();

    let mut op_results: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();

    for (label, op, base_leaves) in &ops {
        // Run bootstrap at γ.
        let mut v = Verifier::new(base_leaves.clone());
        let discoveries = v.bootstrap(op.as_ref(), &all_targets, 4, 4);
        // Cross-check each discovery at all 3 test points.
        let reports = cross_check(
            &discoveries,
            op.as_ref(),
            &v.leaves,
            &TEST_POINTS,
            |name, x| target_at(name, x),
            1e-10,
        );
        report_crosscheck(label, &reports);

        // Track per-target minimal depth for Part B.
        let entry = op_results.entry(label.to_string()).or_default();
        for d in &discoveries {
            let prev = entry.entry(d.target_name.clone()).or_insert(usize::MAX);
            *prev = (*prev).min(d.expanded_size);
        }
    }

    // -- Part B: minimal-depth table --
    println!();
    println!("Part B — minimal expanded size per (target, operator)");
    println!();

    // Fixed column order so the table reads consistently.
    let columns = ["EML", "PowSkew", "PowExpSkew"];
    // Union of all discovered target names.
    let mut target_names: Vec<&str> = all_targets.iter().map(|t| t.name).collect();
    target_names.sort();

    print!("  {:<14}", "target");
    for c in &columns {
        print!("{:>14}", c);
    }
    println!();
    println!("  {}", "-".repeat(14 + 14 * columns.len()));

    for name in &target_names {
        let mut row_has_data = false;
        for c in &columns {
            if let Some(cells) = op_results.get(*c) {
                if cells.contains_key(*name) {
                    row_has_data = true;
                    break;
                }
            }
        }
        if !row_has_data {
            continue;
        }
        print!("  {:<14}", name);
        for c in &columns {
            let cell = op_results
                .get(*c)
                .and_then(|m| m.get(*name))
                .map(|sz| format!("{}", sz))
                .unwrap_or_else(|| "—".to_string());
            print!("{:>14}", cell);
        }
        println!();
    }

    // -- Part C: summary stats --
    println!();
    println!("Part C — coverage summary");
    let total = all_targets.len();
    for c in &columns {
        let n = op_results.get(*c).map(|m| m.len()).unwrap_or(0);
        println!("  {:<12} {:>3}/{} targets", c, n, total);
    }

    // Union of all successful cross-checks across operators.
    let mut union: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for (_, cells) in &op_results {
        for k in cells.keys() {
            union.insert(k.clone());
        }
    }
    println!("  {:<12} {:>3}/{} targets", "UNION", union.len(), total);
}
