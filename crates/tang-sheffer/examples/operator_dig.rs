//! Dig into specific operators: run bootstrap with a high budget/iteration
//! count, print every discovery in order with expanded sizes, and compare
//! the target sets found by each.
//!
//! This is the "deep dive" complement to operator_survey — use when the
//! survey surfaces a surprising candidate and you want to know WHY.

use std::collections::HashSet;
use std::f64::consts::E;

use tang_sheffer::{
    standard_constants, standard_functions, Edl, Eml, Leaf, Operator, PowExpSkew, PowSkew,
    Verifier, C, TEST_POINT,
};

const BUDGET: usize = 4;
const ITERATIONS: usize = 6;

fn main() {
    let mut targets = standard_constants();
    targets.extend(standard_functions());

    let runs: Vec<(&str, Box<dyn Operator>, &str, C)> = vec![
        ("EML", Box::new(Eml), "1", C::new(1.0, 0.0)),
        ("EDL", Box::new(Edl), "e", C::new(E, 0.0)),
        ("PowSkew", Box::new(PowSkew), "2", C::new(2.0, 0.0)),
        ("PowExpSkew+1", Box::new(PowExpSkew), "1", C::new(1.0, 0.0)),
    ];

    let mut per_op: Vec<(String, HashSet<String>)> = Vec::new();

    for (label, op, const_name, const_value) in &runs {
        println!("=== {} ===", label);
        let leaves = vec![
            Leaf::constant(*const_name, *const_value),
            Leaf::variable("x", C::new(TEST_POINT, 0.0)),
        ];
        let mut v = Verifier::new(leaves);
        let discoveries = v.bootstrap(op.as_ref(), &targets, BUDGET, ITERATIONS);

        let mut names = HashSet::new();
        for d in &discoveries {
            names.insert(d.target_name.clone());
            println!(
                "  iter {} | {:>10} | size {:>3} (expanded {:>3}) = {}",
                d.iteration,
                d.target_name,
                d.size,
                d.expanded_size,
                d.expression.format(op.name(), &v.leaves),
            );
        }
        println!("  total: {}/{}\n", discoveries.len(), targets.len());
        per_op.push((label.to_string(), names));
    }

    // Pairwise set diffs: what does each pair uniquely find vs share?
    println!("=== Pairwise differences ===");
    for i in 0..per_op.len() {
        for j in 0..per_op.len() {
            if i == j {
                continue;
            }
            let (name_a, set_a) = &per_op[i];
            let (name_b, set_b) = &per_op[j];
            let only_a: Vec<_> = set_a.difference(set_b).cloned().collect();
            if !only_a.is_empty() {
                let mut sorted = only_a;
                sorted.sort();
                println!("  {} \\ {}: {}", name_a, name_b, sorted.join(", "));
            }
        }
    }

    // Global coverage: what's reached by at least one?
    let mut union: HashSet<String> = HashSet::new();
    for (_, set) in &per_op {
        union.extend(set.iter().cloned());
    }
    let mut still_missing: Vec<&str> = targets
        .iter()
        .map(|t| t.name)
        .filter(|n| !union.contains(*n))
        .collect();
    still_missing.sort();
    println!("\n  union across all 3: {}/{}", union.len(), targets.len());
    if !still_missing.is_empty() {
        println!("  missing from all: {}", still_missing.join(", "));
    }
}
