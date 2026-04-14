//! Browser-facing wrapper for tang-sheffer.
//!
//! Exposes a handful of named binary operators (the verified Phase 6
//! constant-free Sheffer candidates plus EML/EDL from the Odrzywołek
//! paper) and a tiny bootstrap driver, all callable from JavaScript.
//!
//! The blog post embeds this so readers can:
//!   1. evaluate any operator at any complex (x, y),
//!   2. watch the bootstrap "grow" the reachable value set from a single
//!      seed leaf by repeatedly applying the operator and promoting hits.

use num_complex::Complex64;
use std::collections::HashSet;
use std::sync::Arc;
use wasm_bindgen::prelude::*;

use tang_sheffer::op_enum::{Atom, BinaryOp, OpExpr, UnaryOp};

type C = Complex64;

fn x() -> Arc<OpExpr> { Arc::new(OpExpr::Atom(Atom::X)) }
fn y() -> Arc<OpExpr> { Arc::new(OpExpr::Atom(Atom::Y)) }
fn one() -> Arc<OpExpr> { Arc::new(OpExpr::Atom(Atom::One)) }

fn unary(op: UnaryOp, e: Arc<OpExpr>) -> Arc<OpExpr> {
    Arc::new(OpExpr::Unary(op, e))
}
fn binary(op: BinaryOp, a: Arc<OpExpr>, b: Arc<OpExpr>) -> Arc<OpExpr> {
    Arc::new(OpExpr::Binary(op, a, b))
}

/// The named operators the blog post can call by string. Each one
/// returns the same OpExpr the CPU search would build internally.
fn build(name: &str) -> Option<Arc<OpExpr>> {
    Some(match name {
        // EML = exp(x) - ln(y)
        "eml" => binary(
            BinaryOp::Sub,
            unary(UnaryOp::Exp, x()),
            unary(UnaryOp::Ln, y()),
        ),
        // EDL = exp(x) / ln(y)
        "edl" => binary(
            BinaryOp::Div,
            unary(UnaryOp::Exp, x()),
            unary(UnaryOp::Ln, y()),
        ),
        // PowSkew = x^y - y^x
        "powskew" => binary(
            BinaryOp::Sub,
            binary(BinaryOp::Pow, x(), y()),
            binary(BinaryOp::Pow, y(), x()),
        ),
        // SubPow = (x - y)^y
        "subpow" => binary(
            BinaryOp::Pow,
            binary(BinaryOp::Sub, x(), y()),
            y(),
        ),
        // OneMinusDiv = 1 - x/y
        "oneminusdiv" => binary(
            BinaryOp::Sub,
            one(),
            binary(BinaryOp::Div, x(), y()),
        ),
        _ => return None,
    })
}

/// Evaluate `op_name(x, y)` once. Returns `[re, im]`. NaN/infinity are
/// returned verbatim; the caller decides how to display them.
#[wasm_bindgen(js_name = evalOp)]
pub fn eval_op(op_name: &str, x_re: f64, x_im: f64, y_re: f64, y_im: f64) -> Box<[f64]> {
    let Some(expr) = build(op_name) else {
        return Box::new([f64::NAN, f64::NAN]);
    };
    let v = expr.eval(C::new(x_re, x_im), C::new(y_re, y_im));
    Box::new([v.re, v.im])
}

/// Pretty-print the named operator (e.g. "((exp(x)) - (ln(y)))").
#[wasm_bindgen]
pub fn pretty(op_name: &str) -> String {
    build(op_name).map(|e| e.pretty()).unwrap_or_default()
}

// ── Bootstrap demo ───────────────────────────────────────────────────
//
// A miniature version of `Verifier::bootstrap` that operates on raw
// Complex64 values with a fixed-tolerance dedup. Optimized for
// "watching the values appear" rather than full verification — no
// expression tracking, just the sequence of distinct values reached
// and at which iteration they first showed up.

fn quantize(v: C) -> (i64, i64) {
    // 1e-9 grid is plenty for the demo and matches the visual
    // tolerance the user can read.
    let scale = 1e9;
    let qr = if v.re.is_finite() { (v.re * scale).round() as i64 } else { i64::MAX };
    let qi = if v.im.is_finite() { (v.im * scale).round() as i64 } else { i64::MAX };
    (qr, qi)
}

#[derive(Clone)]
struct Found {
    value: C,
    iter: u32,
}

/// Run a bounded bootstrap. `seed_re/im` is the single starting variable
/// `x`; the operator is applied symmetrically (left and right) to every
/// pair of currently-known values. After each iteration, every newly-
/// discovered distinct value is promoted to a leaf for the next round.
///
/// Returns a JSON string: `[{re, im, iter, label?}, ...]` ordered by
/// iteration first found. The first entry is always the seed itself.
#[wasm_bindgen]
pub fn bootstrap(op_name: &str, seed_re: f64, seed_im: f64, max_iters: u32, max_per_iter: u32) -> String {
    let Some(expr) = build(op_name) else {
        return "[]".to_string();
    };

    let mut found: Vec<Found> = vec![Found { value: C::new(seed_re, seed_im), iter: 0 }];
    let mut seen: HashSet<(i64, i64)> = HashSet::new();
    seen.insert(quantize(found[0].value));

    for it in 1..=max_iters {
        let snapshot = found.clone();
        let mut added = 0u32;
        'pairs: for a in &snapshot {
            for b in &snapshot {
                let v = expr.eval(a.value, b.value);
                if !v.re.is_finite() || !v.im.is_finite() { continue; }
                // Filter wildly-large intermediates so the visualization
                // doesn't get steamrolled by overflow chains.
                if v.norm() > 1e6 { continue; }
                let key = quantize(v);
                if seen.insert(key) {
                    found.push(Found { value: v, iter: it });
                    added += 1;
                    if added >= max_per_iter { break 'pairs; }
                }
            }
        }
        if added == 0 { break; }
    }

    // Hand-roll JSON (no serde dep just for this).
    let mut out = String::from("[");
    for (i, f) in found.iter().enumerate() {
        if i > 0 { out.push(','); }
        out.push_str(&format!(
            "{{\"re\":{:.6},\"im\":{:.6},\"iter\":{}}}",
            f.value.re, f.value.im, f.iter
        ));
    }
    out.push(']');
    out
}

/// Quick comma-separated label list of well-known target constants the
/// bootstrap might land on (so the visualization can annotate them).
#[wasm_bindgen]
pub fn label_for(re: f64, im: f64) -> String {
    let candidates: &[(f64, f64, &str)] = &[
        (0.0, 0.0, "0"),
        (1.0, 0.0, "1"),
        (-1.0, 0.0, "−1"),
        (2.0, 0.0, "2"),
        (-2.0, 0.0, "−2"),
        (0.5, 0.0, "1/2"),
        (0.0, 1.0, "i"),
        (0.0, -1.0, "−i"),
        (std::f64::consts::E, 0.0, "e"),
        (std::f64::consts::PI, 0.0, "π"),
    ];
    for (cr, ci, name) in candidates {
        if (re - cr).abs() < 1e-6 && (im - ci).abs() < 1e-6 {
            return name.to_string();
        }
    }
    String::new()
}
