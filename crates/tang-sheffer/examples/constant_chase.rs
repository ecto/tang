//! Phase 5a: constant chase from `{1}` only.
//!
//! Bootstrap EML and EDL with NO variable in the leaf pool — every
//! discovered value is automatically a true constant identity because
//! there's no x to depend on. Uses the extended target list with
//! stepping-stone values (±∞, iπ/2, intermediate exponentials) that the
//! bootstrap loop can promote to leaves to cascade deeper.
//!
//! Goal: produce a verified constant pool including {π, i, iπ, i/2, etc.}
//! that Phase 5b will then use as seed leaves for function search.
//!
//! Run: `cargo run --release --example constant_chase -p tang-sheffer`

use std::collections::HashSet;
use std::f64::consts::E;

use tang_sheffer::{stepping_stone_constants, Edl, Eml, Leaf, Operator, Verifier, C};

const BUDGET: usize = 4;
const ITERATIONS: usize = 10;

fn main() {
    let targets = stepping_stone_constants();
    println!(
        "=== constant chase from {{1}} only ({} targets, budget {}, {} iters) ===\n",
        targets.len(),
        BUDGET,
        ITERATIONS
    );

    let runs: Vec<(&str, Box<dyn Operator>, Vec<Leaf>)> = vec![
        (
            "EML",
            Box::new(Eml),
            vec![Leaf::constant("1", C::new(1.0, 0.0))],
        ),
        (
            "EDL",
            Box::new(Edl),
            vec![Leaf::constant("e", C::new(E, 0.0))],
        ),
    ];

    for (label, op, leaves) in runs {
        println!("-- {} --", label);
        let mut v = Verifier::new(leaves);
        let discoveries =
            v.bootstrap_with_progress(op.as_ref(), &targets, BUDGET, ITERATIONS, |p| {
                let new = if p.new_targets.is_empty() {
                    "—".to_string()
                } else {
                    p.new_targets.join(", ")
                };
                println!(
                    "  iter {} | leaves={:>3} | cat={:>8} | found={:>3}/{} | +{}  ({:.2}s)",
                    p.iteration,
                    p.leaf_count,
                    p.catalogue_size,
                    p.total_found,
                    targets.len(),
                    new,
                    p.iter_elapsed.as_secs_f64(),
                );
            });

        println!();
        println!("  final discoveries ({}):", discoveries.len());
        let mut names: HashSet<&str> = HashSet::new();
        for d in &discoveries {
            names.insert(&d.target_name);
            println!(
                "    {:>8}  size {:>2} (expanded {:>3})  iter {}  = {}",
                d.target_name,
                d.size,
                d.expanded_size,
                d.iteration,
                d.expression.format(op.name(), &v.leaves),
            );
        }
        let missing: Vec<&str> = targets
            .iter()
            .map(|t| t.name)
            .filter(|n| !names.contains(n))
            .collect();
        println!();
        println!("  missed ({}): {}", missing.len(), missing.join(", "));
        println!();
    }
}
