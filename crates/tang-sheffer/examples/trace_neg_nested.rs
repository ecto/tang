//! Trace the `e`-discovery mechanism of NegNestedPow = −((x − y)^(x^y)).
//!
//! MAX_SIZE=8 found that this operator reaches 14 targets constant-free
//! from `{x}` alone, including the transcendental constants `e` and `−e`.
//! This example runs the bootstrap verbosely and prints every discovery
//! in iteration order with (a) its local expression, (b) its full
//! expansion back to the base alphabet, (c) the numerical value at each
//! test point to confirm it's not a coincidence, and (d) a manual trace
//! of the key intermediate values so we can understand WHY the chain
//! reaches `e` rather than getting stuck at a fixed point.

use std::f64::consts::E;
use std::sync::Arc;

use tang_sheffer::op_enum::{Atom, BinaryOp, EnumOp, OpExpr, UnaryOp};
use tang_sheffer::{
    standard_constants, standard_functions, Leaf, Operator, Verifier, C, TEST_POINT, TEST_POINTS,
};

const BUDGET: usize = 4;
const ITERS: usize = 6;

fn build_neg_nested_pow() -> OpExpr {
    // -((x - y) ^ (x ^ y))
    let x = || Arc::new(OpExpr::Atom(Atom::X));
    let y = || Arc::new(OpExpr::Atom(Atom::Y));
    let sub = |a, b| Arc::new(OpExpr::Binary(BinaryOp::Sub, a, b));
    let powc = |a, b| Arc::new(OpExpr::Binary(BinaryOp::Pow, a, b));
    let neg = |a| Arc::new(OpExpr::Unary(UnaryOp::Neg, a));

    let root = neg(powc(sub(x(), y()), powc(x(), y())));
    (*root).clone()
}

fn main() {
    println!("=== Tracing NegNestedPow = -((x - y) ^ (x ^ y)) ===\n");
    let expr = build_neg_nested_pow();
    let op = EnumOp::new(expr.clone());

    // --- Step 1: hand-verify the key base-case inputs ---
    println!("--- Key input identities (hand-computed) ---");
    let x = C::new(TEST_POINT, 0.0);
    let zero = C::new(0.0, 0.0);
    let one = C::new(1.0, 0.0);
    let neg_one = C::new(-1.0, 0.0);

    for (label, a, b) in [
        ("f(x, x)", x, x),
        ("f(x, 0)", x, zero),
        ("f(0, x)", zero, x),
        ("f(x, 1)", x, one),
        ("f(1, x)", one, x),
        ("f(0, 0)", zero, zero),
        ("f(-1, x)", neg_one, x),
        ("f(x, -1)", x, neg_one),
        ("f(-1, 0)", neg_one, zero),
        ("f(0, -1)", zero, neg_one),
        ("f(1, -1)", one, neg_one),
        ("f(-1, 1)", neg_one, one),
        ("f(1, 1)", one, one),
    ] {
        let v = op.eval(a, b);
        let re = v.re;
        let im = v.im;
        let annot = annotate(v);
        println!("  {:<12} = {:+.6}{:+.6}i     {}", label, re, im, annot);
    }

    // --- Step 2: run bootstrap from {x} and print the cascade ---
    println!();
    println!("--- Bootstrap cascade from {{x = γ ≈ 0.5772...}} ---");
    let leaves = vec![Leaf::variable("x", x)];
    let mut v = Verifier::new(leaves);

    let mut targets = standard_constants();
    targets.extend(standard_functions());

    let discoveries =
        v.bootstrap_with_progress(&op, &targets, BUDGET, ITERS, |p| {
            let new = if p.new_targets.is_empty() {
                "—".to_string()
            } else {
                p.new_targets.join(", ")
            };
            println!(
                "  iter {} | leaves={:>3} | cat={:>8} | found={:>2}/{} | +{}  ({:.2}s)",
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
    println!("--- Full discovery list (in bootstrap order) ---");
    for d in &discoveries {
        let pretty = d.expression.format("neg-nested-pow", &v.leaves);
        let short = truncate(&pretty, 80);
        println!(
            "  iter {} | {:>8} | local {:>2} (expanded {:>4}) = {}",
            d.iteration, d.target_name, d.size, d.expanded_size, short
        );
    }

    // --- Step 2.5: manually evaluate the specific e-chain from the
    // verifier's discovered expression, step by step, so we can see
    // exactly how `e` appears. The expression from the output:
    //   e = f(f(f(1, f(1, 2)), -1), 0)
    // where all constant leaves are themselves derived from {x}.
    println!();
    println!("--- Step-by-step trace of f(f(f(1, f(1, 2)), -1), 0) ---");
    let one = C::new(1.0, 0.0);
    let neg_one_c = C::new(-1.0, 0.0);
    let zero_c = C::new(0.0, 0.0);
    let two = C::new(2.0, 0.0);

    let c1 = op.eval(one, two);
    println!("  C1 = f(1, 2)   = -((1-2)^(1^2))                    = {:?}", c1);
    let c2 = op.eval(one, c1);
    println!("  C2 = f(1, C1)  = -((1-C1)^(1^C1))                  = {:?}", c2);
    let c3 = op.eval(c2, neg_one_c);
    println!("  C3 = f(C2, -1) = -((C2-(-1))^(C2^(-1)))            = {:?}", c3);
    let cfinal = op.eval(c3, zero_c);
    println!("  e  = f(C3, 0)  = -((C3-0)^(C3^0))                  = {:?}", cfinal);
    println!("  target e = {:?}", C::new(E, 0.0));
    println!(
        "  residual = {:.2e}",
        (cfinal - C::new(E, 0.0)).norm()
    );

    // --- Step 3: deep-dive the e, -e, ±i, -x chain ---
    //
    // For each of the targets that are genuinely new to the Phase-6 zoo,
    // print its expanded expression and evaluate it at all three multi-
    // point test values. If it's a true identity, all three must match.
    println!();
    println!("--- e, -e, ±i, -x verification at three test points ---");
    let of_interest = ["-x", "-1", "e", "-e", "i", "-i", "1", "0", "x+1", "x-1"];
    for name in of_interest {
        let Some(d) = discoveries.iter().find(|d| d.target_name == name) else {
            println!("  {:<6}: not found", name);
            continue;
        };
        let expanded = d.expression.format_expanded("neg-nested-pow", &v.leaves);
        let short = truncate(&expanded, 140);
        println!();
        println!(
            "  {:<4} (expanded size {}):",
            name, d.expanded_size
        );
        println!("    {}", short);

        // Evaluate at {γ, A, G} by rebinding the variable leaf. Because
        // the Stage-B leaves are all Variable (there's only `x`), rebind
        // is straightforward via Expr::eval with a synthetic leaf pool.
        use tang_sheffer::crosscheck::rebind_leaves;
        for (&tp, label) in TEST_POINTS.iter().zip(["γ", "A", "G"].iter()) {
            let rebound = rebind_leaves(&v.leaves, tp);
            let val = d.expression.eval_recursive(&op, &rebound);
            println!(
                "    at x = {:<4} ≈ {:.4}  →  {:+.12e}{:+.12e}i",
                label, tp, val.re, val.im
            );
        }
    }
}

/// Best-effort annotation: try to recognize the numeric value as a small
/// named constant so the hand-computed identities are self-documenting.
fn annotate(v: C) -> String {
    let tests = [
        ("0", C::new(0.0, 0.0)),
        ("1", C::new(1.0, 0.0)),
        ("-1", C::new(-1.0, 0.0)),
        ("e", C::new(E, 0.0)),
        ("-e", C::new(-E, 0.0)),
        ("x", C::new(TEST_POINT, 0.0)),
        ("-x", C::new(-TEST_POINT, 0.0)),
        ("x+1", C::new(TEST_POINT + 1.0, 0.0)),
        ("x-1", C::new(TEST_POINT - 1.0, 0.0)),
    ];
    for (name, probe) in tests {
        if (v - probe).norm() < 1e-10 {
            return format!("(= {})", name);
        }
    }
    String::new()
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max.saturating_sub(1)])
    }
}
