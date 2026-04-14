//! Test the "constant-free" property: can an operator generate its own
//! constants from a single arbitrary input variable, no distinguished
//! leaf required?
//!
//! The paper (Table 2, conjectures) explicitly calls this out as an open
//! problem: "A binary operator needing NO distinguished constant (generates
//! constants from arbitrary input)". PowSkew(x, y) = x^y - y^x has the
//! trivial property pow-skew(x, x) = 0, and thus pow-skew(x, 0) = 1 and
//! pow-skew(0, x) = -1 — so starting from just {x}, we immediately have
//! {x, 0, 1, -1} at depth 2.
//!
//! This example runs the bootstrap verifier starting from a single variable
//! leaf (no constants) to see how far each candidate operator can reach.

use tang_sheffer::{
    standard_constants, standard_functions, Eml, ExpDiff, Leaf, LnDiff, Operator, PowExpSkew,
    PowLnSkew, PowMinus, PowRatio, PowSkew, SinhDiff, SqrSqrt, TanDiff, Verifier, C, TEST_POINT,
};

const BUDGET: usize = 4;
const ITERATIONS: usize = 5;

fn main() {
    let mut targets = standard_constants();
    targets.extend(standard_functions());

    let runs: Vec<(&str, Box<dyn Operator>)> = vec![
        ("PowSkew", Box::new(PowSkew)),
        ("PowExpSkew", Box::new(PowExpSkew)),
        ("PowLnSkew", Box::new(PowLnSkew)),
        ("PowRatio", Box::new(PowRatio)),
        ("PowMinus", Box::new(PowMinus)),
        ("ExpDiff", Box::new(ExpDiff)),
        ("SinhDiff", Box::new(SinhDiff)),
        ("TanDiff", Box::new(TanDiff)),
        ("LnDiff", Box::new(LnDiff)),
        ("SqrSqrt", Box::new(SqrSqrt)),
        ("Eml (control)", Box::new(Eml)),
    ];

    println!("=== Constant-free bootstrap ===");
    println!("leaves = {{x = γ}} only (no distinguished constant)\n");

    for (label, op) in &runs {
        let leaves = vec![Leaf::variable("x", C::new(TEST_POINT, 0.0))];
        let mut v = Verifier::new(leaves);
        let discoveries = v.bootstrap(op.as_ref(), &targets, BUDGET, ITERATIONS);

        println!("--- {} ---", label);
        for d in &discoveries {
            println!(
                "  iter {} | {:>10} size {:>2} (expanded {:>3}) = {}",
                d.iteration,
                d.target_name,
                d.size,
                d.expanded_size,
                d.expression.format(op.name(), &v.leaves),
            );
        }
        println!(
            "  total: {}/{} from {{x}} alone\n",
            discoveries.len(),
            targets.len()
        );
    }
}
