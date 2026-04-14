//! Phase 5: crack the remaining 7 targets {π, π/2, 2π, iπ, sqrt(x),
//! sin(x), cos(x)} via two-stage bootstrap.
//!
//! Stage A: constant chase with EDL from `{e}` only. Every discovery is
//! an x-independent constant, so trivially a true identity. This populates
//! a pool including {1, 0, −1, e, π, i, iπ, iπ/2, i/2, 2i, ...}.
//!
//! Stage B: append `x` to the leaf pool and run bootstrap targeting
//! {sin(x), cos(x), sqrt(x)}. The constants from Stage A are available as
//! primitive leaves, dramatically shortening the search depth for function
//! identities.
//!
//! Stage C: multi-point verify every discovery at {γ, A, G} to filter out
//! x-dependent coincidences.
//!
//! Run: `cargo run --release --example crack_remaining -p tang-sheffer`

use std::collections::BTreeMap;
use std::f64::consts::{E, PI};

use tang_sheffer::{
    cross_check, stepping_stone_constants, Edl, Eml, Leaf, Operator, Verifier, C, TEST_POINT,
    TEST_POINTS,
};

const STAGE_A_BUDGET: usize = 4;
const STAGE_A_ITERS: usize = 8;
const STAGE_B_BUDGET: usize = 4;
// Stops at iter 6 (0-indexed) — iter 7 would OOM at 500M+ catalog entries.
const STAGE_B_ITERS: usize = 7;

/// Target lookup, including both stepping stones and functions of x.
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
        "+inf" => C::new(f64::INFINITY, 0.0),
        "-inf" => C::new(f64::NEG_INFINITY, 0.0),
        "exp(e)" => C::new(E.exp(), 0.0),
        "e-1" => C::new(E - 1.0, 0.0),
        "e+1" => C::new(E + 1.0, 0.0),
        "2e" => C::new(2.0 * E, 0.0),
        "e/2" => C::new(E / 2.0, 0.0),
        "ln(pi)" => C::new(PI.ln(), 0.0),
        "i*pi/2" => C::new(0.0, PI / 2.0),
        "-i*pi" => C::new(0.0, -PI),
        "2i" => C::new(0.0, 2.0),
        "i/2" => C::new(0.0, 0.5),

        "x" => cx,
        "exp(x)" => cx.exp(),
        "ln(x)" => cx.ln(),
        "sin(x)" => cx.sin(),
        "cos(x)" => cx.cos(),
        "sqrt(x)" => cx.sqrt(),
        "x^2" => cx * cx,
        "1/x" => C::new(1.0, 0.0) / cx,
        "-x" => -cx,
        "x-1" => cx - C::new(1.0, 0.0),
        "x+1" => cx + C::new(1.0, 0.0),
        "2x" => cx * C::new(2.0, 0.0),
        "e*x" => cx * C::new(E, 0.0),
        "exp(exp(x))" => cx.exp().exp(),
        "ln(ln(x))" => cx.ln().ln(),
        "exp(-x)" => (-cx).exp(),
        "exp(i*x)" => (C::new(0.0, 1.0) * cx).exp(),
        "exp(-i*x)" => (C::new(0.0, -1.0) * cx).exp(),
        "i*x" => C::new(0.0, 1.0) * cx,
        "-i*x" => C::new(0.0, -1.0) * cx,
        "x/2" => cx / C::new(2.0, 0.0),
        "ln(x)/2" => cx.ln() / C::new(2.0, 0.0),
        "ln(ln(ln(x)))" => cx.ln().ln().ln(),
        "2*sin(x)" => C::new(2.0, 0.0) * cx.sin(),
        "2*cos(x)" => C::new(2.0, 0.0) * cx.cos(),
        "exp(ix)-exp(-ix)" => {
            let i = C::new(0.0, 1.0);
            (i * cx).exp() - (-i * cx).exp()
        }
        "exp(ix)+exp(-ix)" => {
            let i = C::new(0.0, 1.0);
            (i * cx).exp() + (-i * cx).exp()
        }
        "2i*sin(x)" => C::new(0.0, 2.0) * cx.sin(),
        "i*sin(x)" => C::new(0.0, 1.0) * cx.sin(),
        "-i*sin(x)" => C::new(0.0, -1.0) * cx.sin(),
        "ln(2i*sin(x))" => (C::new(0.0, 2.0) * cx.sin()).ln(),
        "ln(ln(2i*sin(x)))" => (C::new(0.0, 2.0) * cx.sin()).ln().ln(),
        "ln(2*cos(x))" => (C::new(2.0, 0.0) * cx.cos()).ln(),
        "ln(ln(2*cos(x)))" => (C::new(2.0, 0.0) * cx.cos()).ln().ln(),
        _ => return None,
    };
    Some(v)
}

fn function_targets() -> Vec<tang_sheffer::Target> {
    use tang_sheffer::Target;
    let x = C::new(TEST_POINT, 0.0);
    let i = C::new(0.0, 1.0);
    let two = C::new(2.0, 0.0);
    vec![
        // Already-reachable baseline
        Target { name: "exp(x)", value: x.exp() },
        Target { name: "ln(x)", value: x.ln() },
        Target { name: "x^2", value: x * x },
        Target { name: "1/x", value: C::new(1.0, 0.0) / x },
        Target { name: "-x", value: -x },
        // Stepping stones toward sqrt: sqrt(x) = exp(ln(x)/2), so we need
        // ln(x)/2 as a leaf. Getting there requires ln(ln(ln(x))) because
        // division in EML is `eml(eml(ln(ln(a)), b), 1) = a/b`, so dividing
        // ln(x) by 2 needs ln(ln(ln(x))).
        Target { name: "ln(ln(x))", value: x.ln().ln() },
        Target { name: "ln(ln(ln(x)))", value: x.ln().ln().ln() },
        Target { name: "ln(x)/2", value: x.ln() / two },
        // Stepping stones toward Euler-form sin/cos
        Target { name: "exp(-x)", value: (-x).exp() },
        Target { name: "exp(i*x)", value: (i * x).exp() },
        Target { name: "exp(-i*x)", value: (-i * x).exp() },
        Target { name: "i*x", value: i * x },
        Target { name: "-i*x", value: -i * x },
        Target { name: "x/2", value: x / two },
        Target { name: "2*sin(x)", value: two * x.sin() },
        Target { name: "2*cos(x)", value: two * x.cos() },
        Target { name: "exp(ix)-exp(-ix)", value: (i * x).exp() - (-i * x).exp() },
        Target { name: "exp(ix)+exp(-ix)", value: (i * x).exp() + (-i * x).exp() },
        Target { name: "2i*sin(x)", value: C::new(0.0, 2.0) * x.sin() },
        Target { name: "i*sin(x)", value: i * x.sin() },
        Target { name: "-i*sin(x)", value: -i * x.sin() },
        // Division stepping stones: ln and ln(ln) of the pre-sin/cos values.
        // Once ln(ln(2i*sin(x))) is a leaf, sin(x) becomes a 5-node tree.
        Target { name: "ln(2i*sin(x))", value: (C::new(0.0, 2.0) * x.sin()).ln() },
        Target { name: "ln(ln(2i*sin(x)))", value: (C::new(0.0, 2.0) * x.sin()).ln().ln() },
        Target { name: "ln(2*cos(x))", value: (two * x.cos()).ln() },
        Target { name: "ln(ln(2*cos(x)))", value: (two * x.cos()).ln().ln() },
        // The three we're chasing
        Target { name: "sqrt(x)", value: x.sqrt() },
        Target { name: "sin(x)", value: x.sin() },
        Target { name: "cos(x)", value: x.cos() },
    ]
}

fn main() {
    // Stage A builds Rc<Expr> chains with expanded sizes up to 269, which
    // blows the default 8 MB stack during enumeration in Stage B (deep
    // recursion in eval/size/drop). Running the real work on a thread with
    // a 256 MB stack sidesteps the issue without restructuring the tree
    // into an iterative form.
    let handle = std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(run)
        .expect("failed to spawn worker thread");
    handle.join().expect("worker thread panicked");
}

fn run() {
    println!("=== Phase 5: cracking the remaining targets ===\n");

    // -- Stage A: constant chase (EDL from {e}) --
    println!("--- Stage A: EDL constant chase from {{e}} only ---");
    let stage_a_op = Edl;
    let const_targets = stepping_stone_constants();
    let mut v_a = Verifier::new(vec![Leaf::constant("e", C::new(E, 0.0))]);
    let const_discoveries = v_a.bootstrap_with_progress(
        &stage_a_op,
        &const_targets,
        STAGE_A_BUDGET,
        STAGE_A_ITERS,
        |p| {
            let new = if p.new_targets.is_empty() {
                "—".to_string()
            } else {
                p.new_targets.join(", ")
            };
            println!(
                "  iter {} | leaves={:>3} | cat={:>9} | found={:>3}/{} | +{}  ({:.2}s)",
                p.iteration,
                p.leaf_count,
                p.catalogue_size,
                p.total_found,
                const_targets.len(),
                new,
                p.iter_elapsed.as_secs_f64(),
            );
        },
    );
    println!(
        "  → {} constants found, final pool = {} leaves\n",
        const_discoveries.len(),
        v_a.leaves.len()
    );
    // Stage A discoveries are x-independent constants; they're trivially
    // valid at any test point (they don't contain x at all). We drop the
    // Stage A verifier here to free the deep Rc<Expr> chains before Stage
    // B — Stage B uses a completely fresh leaf pool built from the same
    // numeric values as flat Constant atoms.
    drop(v_a);

    // -- Stage B: fresh leaf pool of essential constants + x, run with EML
    //
    // Key change: Stage B uses EML, not EDL, because EML's `exp(a) − ln(b)`
    // combinator has subtraction built in — essential for Euler's
    // sin(x) = (exp(ix) − exp(−ix))/(2i). EDL's `exp(a)/ln(b)` has no
    // analog subtraction, so sin/cos aren't naturally reachable there.
    //
    // We use flat Constant atoms for the essentials so Stage A's deep
    // Rc<Expr> chains don't come along. Stage A remains the constructive
    // proof that the numeric constants are reachable from {e}.
    println!("--- Stage B: fresh essentials pool + x, operator = EML ---");
    let stage_b_op = Eml;

    // Minimal essentials: these 10 constants are sufficient for EML
    // constructions of {exp(x), ln(x), sqrt(x), sin(x), cos(x), ...}.
    // Pruned from the full 14-item list to keep the Stage B catalog
    // under 400M entries at budget 4 iter 6 (where sin/cos are found).
    let essentials: &[(&str, C)] = &[
        ("1", C::new(1.0, 0.0)),
        ("0", C::new(0.0, 0.0)),
        ("-1", C::new(-1.0, 0.0)),
        ("1/2", C::new(0.5, 0.0)),
        ("2", C::new(2.0, 0.0)),
        ("e", C::new(E, 0.0)),
        ("i", C::new(0.0, 1.0)),
        ("-i", C::new(0.0, -1.0)),
        ("2i", C::new(0.0, 2.0)),
        ("i*pi", C::new(0.0, PI)),
    ];
    // PI imported but unused in this trimmed setup — suppress the warning.
    let _ = PI;
    let mut v = Verifier::new(
        essentials
            .iter()
            .map(|(n, val)| Leaf::constant(*n, *val))
            .collect(),
    );
    v.leaves
        .push(Leaf::variable("x", C::new(TEST_POINT, 0.0)));
    println!("  fresh pool: {} leaves", v.leaves.len());

    let fn_targets = function_targets();
    let fn_discoveries = v.bootstrap_with_progress(
        &stage_b_op,
        &fn_targets,
        STAGE_B_BUDGET,
        STAGE_B_ITERS,
        |p| {
            let new = if p.new_targets.is_empty() {
                "—".to_string()
            } else {
                p.new_targets.join(", ")
            };
            println!(
                "  iter {} | leaves={:>3} | cat={:>9} | found={:>3}/{} | +{}  ({:.2}s)",
                p.iteration,
                p.leaf_count,
                p.catalogue_size,
                p.total_found,
                fn_targets.len(),
                new,
                p.iter_elapsed.as_secs_f64(),
            );
        },
    );
    println!();
    println!("  function discoveries ({}):", fn_discoveries.len());
    for d in &fn_discoveries {
        println!(
            "    {:>10}  size {:>2} (expanded {:>4})  = {}",
            d.target_name,
            d.size,
            d.expanded_size,
            d.expression.format(stage_b_op.name(), &v.leaves),
        );
    }

    // -- Stage C: multi-point verify (Stage B only; Stage A is trivially
    //    valid because its expressions are x-independent) --
    println!();
    println!("--- Stage C: multi-point cross-check at {{γ, A, G}} ---");
    let reports = cross_check(
        &fn_discoveries,
        &stage_b_op,
        &v.leaves,
        &TEST_POINTS,
        |name, x| target_at(name, x),
        1e-8,
    );

    let mut by_status: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    by_status.insert("PASS", Vec::new());
    by_status.insert("FAIL", Vec::new());
    for r in &reports {
        let slot = if r.passed { "PASS" } else { "FAIL" };
        by_status.get_mut(slot).unwrap().push(&r.target_name);
    }
    let pass = by_status.get("PASS").unwrap();
    let fail = by_status.get("FAIL").unwrap();
    println!("  {} pass, {} fail", pass.len(), fail.len());
    if !fail.is_empty() {
        println!("  FAILED targets: {}", fail.join(", "));
    }

    // -- Highlight the previously-missing targets --
    //
    // Check BOTH Stage A (constants; x-independent, trivially valid) and
    // Stage B (functions; cross-checked above). A target counts as CRACKED
    // if it was found in either stage.
    println!();
    println!("--- Remaining-target scoreboard ---");
    let const_names: std::collections::HashSet<&str> =
        const_discoveries.iter().map(|d| d.target_name.as_str()).collect();
    let remaining = [
        "pi", "pi/2", "2pi", "i*pi", "i", "-i", "sqrt(x)", "sin(x)", "cos(x)",
    ];
    for name in remaining {
        if const_names.contains(name) {
            let d = const_discoveries.iter().find(|d| d.target_name == name).unwrap();
            println!(
                "  {:<10} ✓ CRACKED (Stage A constant, expanded size {})",
                name, d.expanded_size
            );
            continue;
        }
        let report = reports.iter().find(|r| r.target_name == name);
        let status = match report {
            Some(r) if r.passed => "✓ CRACKED (Stage B, verified)",
            Some(_) => "✗ FAIL (Stage B, coincidence)",
            None => "— not found",
        };
        let extra = report
            .map(|r| format!(" size {}", r.size))
            .unwrap_or_default();
        println!("  {:<10} {}{}", name, status, extra);
    }
}
