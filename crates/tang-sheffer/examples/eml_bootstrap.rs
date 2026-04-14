//! Bootstrap-style EML reproduction.
//!
//! Single-pass enumeration at depth ≥5 blows up quickly, so the paper uses
//! iterative promotion: search a shallow horizon, add every discovered
//! formula to the leaf pool, search again. Each iteration, new compositions
//! become reachable at the same depth budget because the "atoms" now include
//! bigger formulas.
//!
//! This example runs the bootstrap loop over EML with a target list of the
//! core constants and single-variable functions, printing the cascade.
//!
//! Run with: `cargo run --release --example eml_bootstrap -p tang-sheffer`.

use tang_sheffer::{
    standard_constants, standard_functions, Eml, Leaf, Operator, Verifier, C, TEST_POINT,
};

const MAX_OPS_PER_ITER: usize = 4;
const MAX_ITERATIONS: usize = 10;

fn main() {
    let op = Eml;
    println!("=== EML bootstrap ===");
    println!(
        "budget: {} ops/iteration × {} iterations\n",
        MAX_OPS_PER_ITER, MAX_ITERATIONS
    );

    // Combine all targets: constants + functions of x.
    let mut targets = standard_constants();
    targets.extend(standard_functions());

    let mut v = Verifier::new(vec![
        Leaf::constant("1", C::new(1.0, 0.0)),
        Leaf::variable("x", C::new(TEST_POINT, 0.0)),
    ]);

    let discoveries = v.bootstrap_with_progress(
        &op,
        &targets,
        MAX_OPS_PER_ITER,
        MAX_ITERATIONS,
        |p| {
            println!(
                "iter {:>2} | leaves={:>3} | catalogue={:>7} | +{} | found={:>2}/{} | {:.2}s (total {:.2}s)",
                p.iteration,
                p.leaf_count,
                p.catalogue_size,
                if p.new_targets.is_empty() {
                    "—".to_string()
                } else {
                    p.new_targets.join(",")
                },
                p.total_found,
                targets.len(),
                p.iter_elapsed.as_secs_f64(),
                p.total_elapsed.as_secs_f64(),
            );
        },
    );

    println!(
        "found {}/{} targets; final leaf pool = {} leaves\n",
        discoveries.len(),
        targets.len(),
        v.leaves.len()
    );

    println!("discoveries (in bootstrap order):");
    for d in &discoveries {
        println!(
            "  iter {} | {:>6} | local size {:>2} | expanded {:>3}  =  {}",
            d.iteration,
            d.target_name,
            d.size,
            d.expanded_size,
            d.expression.format(op.name(), &v.leaves),
        );
    }

    // Print the base-alphabet expansion for a few marquee targets so the
    // reader can audit them by hand.
    println!("\nbase-alphabet audit:");
    for target_name in ["e", "0", "-1", "ln(x)", "exp(x)", "pi", "i"] {
        if let Some(d) = discoveries.iter().find(|d| d.target_name == target_name) {
            let expanded = d.expression.format_expanded(op.name(), &v.leaves);
            let truncated = if expanded.len() > 180 {
                format!("{}... [{}]", &expanded[..180], expanded.len())
            } else {
                expanded
            };
            println!("  {:>6} = {}", target_name, truncated);
        }
    }

    // List the targets we failed to discover, for diagnosis.
    let found_names: std::collections::HashSet<_> =
        discoveries.iter().map(|d| d.target_name.clone()).collect();
    let missing: Vec<_> = targets
        .iter()
        .filter(|t| !found_names.contains(t.name))
        .map(|t| t.name)
        .collect();
    if !missing.is_empty() {
        println!("\nstill missing after {} iterations:", MAX_ITERATIONS);
        for name in missing {
            println!("  - {}", name);
        }
    }
}
