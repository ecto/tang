//! Reproduce EML discoveries from Odrzywołek 2026.
//!
//! Runs two enumerative searches over `eml(x, y) = exp(x) − ln(y)`:
//!   1. Constant derivation from the pool {1}: what constants are reachable?
//!   2. Function derivation from {1, x = γ}: what functions of x appear?
//!
//! For each, prints the per-level distinct-value count and the smallest
//! expression matching each standard target.
//!
//! Run with: `cargo run --example eml_reproduction -p tang-sheffer`.

use tang_sheffer::Eml;
use tang_sheffer::{
    standard_constants, standard_functions, Found, Leaf, Operator, Verifier, C, TEST_POINT,
};

const MAX_OPS: usize = 5;

fn main() {
    let op = Eml;
    println!("=== EML reproduction (search up to {} ops) ===", MAX_OPS);
    println!();

    // ---- Constant search ----
    println!("-- Constant search: leaves = {{1}} --");
    let v = Verifier::new(vec![Leaf::constant("1", C::new(1.0, 0.0))]);
    let cat = v.enumerate(&op, MAX_OPS);
    print_levels(&cat);

    let flat: Vec<Found> = cat.into_iter().flatten().collect();
    let discoveries = v.match_targets(&flat, &standard_constants());
    println!();
    println!("  discovered constants:");
    if discoveries.is_empty() {
        println!("    (none within tolerance)");
    }
    for d in &discoveries {
        println!(
            "    {:>4} = {}  [size {}, residual {:.2e}]",
            d.target_name,
            d.expression.format(op.name(), &v.leaves),
            d.size,
            d.residual,
        );
    }

    // ---- Function search ----
    println!();
    println!("-- Function search: leaves = {{1, x = γ}} --");
    let v = Verifier::new(vec![
        Leaf::constant("1", C::new(1.0, 0.0)),
        Leaf::variable("x", C::new(TEST_POINT, 0.0)),
    ]);
    let cat = v.enumerate(&op, MAX_OPS);
    print_levels(&cat);

    let flat: Vec<Found> = cat.into_iter().flatten().collect();
    let discoveries = v.match_targets(&flat, &standard_functions());
    println!();
    println!("  discovered functions:");
    if discoveries.is_empty() {
        println!("    (none within tolerance)");
    }
    for d in &discoveries {
        println!(
            "    {:>12} = {}  [size {}, residual {:.2e}]",
            d.target_name,
            d.expression.format(op.name(), &v.leaves),
            d.size,
            d.residual,
        );
    }
}

fn print_levels(cat: &[Vec<Found>]) {
    let total: usize = cat.iter().map(|l| l.len()).sum();
    println!("  {} distinct values across all levels", total);
    for (n, level) in cat.iter().enumerate() {
        println!("    ops={}: {} values", n, level.len());
    }
}
