//! Programmatic operator enumeration.
//!
//! Generates candidate binary operators `f(x, y)` as small expression
//! trees over a fixed alphabet of atoms, unary ops, and binary ops, then
//! deduplicates by evaluating at a handful of transcendental test points.
//! Each surviving `OpExpr` is wrapped as an `Operator` impl (via `EnumOp`)
//! so the existing `Verifier` can run a bootstrap over it.
//!
//! This is the Strategy-A search from the original research prompt:
//! rather than hand-writing candidate operators, enumerate them and score
//! each with the existing infrastructure. The hope is to surface
//! operators we haven't hand-thought-of that beat EML/EDL/PowSkew on some
//! axis (coverage, constant-freedom, growth, depth).

use std::collections::HashSet;
use std::f64::consts::E;
use std::sync::Arc;

use crate::operator::Operator;
use crate::verify::{quantize, ValueKey};
use crate::C;

/// Atomic inputs to an enumerated operator. Includes the two formal
/// variables `x, y` and a few common small constants so we can build
/// things like `x + 1` or `exp(x) - e` inside the operator definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Atom {
    X,
    Y,
    Zero,
    One,
    NegOne,
    E,
}

impl Atom {
    pub const ALL: [Atom; 6] = [
        Atom::X,
        Atom::Y,
        Atom::Zero,
        Atom::One,
        Atom::NegOne,
        Atom::E,
    ];

    pub fn eval(&self, x: C, y: C) -> C {
        match self {
            Atom::X => x,
            Atom::Y => y,
            Atom::Zero => C::new(0.0, 0.0),
            Atom::One => C::new(1.0, 0.0),
            Atom::NegOne => C::new(-1.0, 0.0),
            Atom::E => C::new(E, 0.0),
        }
    }

    pub fn pretty(&self) -> &'static str {
        match self {
            Atom::X => "x",
            Atom::Y => "y",
            Atom::Zero => "0",
            Atom::One => "1",
            Atom::NegOne => "-1",
            Atom::E => "e",
        }
    }
}

/// One-argument operations applied elementwise to an `OpExpr`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Neg,
    Inv,
    Sqr,
    Sqrt,
    Exp,
    Ln,
    Sin,
    Cos,
    Sinh,
    Tanh,
}

impl UnaryOp {
    pub const ALL: [UnaryOp; 10] = [
        UnaryOp::Neg,
        UnaryOp::Inv,
        UnaryOp::Sqr,
        UnaryOp::Sqrt,
        UnaryOp::Exp,
        UnaryOp::Ln,
        UnaryOp::Sin,
        UnaryOp::Cos,
        UnaryOp::Sinh,
        UnaryOp::Tanh,
    ];

    pub fn eval(&self, v: C) -> C {
        match self {
            UnaryOp::Neg => -v,
            UnaryOp::Inv => C::new(1.0, 0.0) / v,
            UnaryOp::Sqr => v * v,
            UnaryOp::Sqrt => v.sqrt(),
            UnaryOp::Exp => v.exp(),
            UnaryOp::Ln => v.ln(),
            UnaryOp::Sin => v.sin(),
            UnaryOp::Cos => v.cos(),
            UnaryOp::Sinh => v.sinh(),
            UnaryOp::Tanh => v.tanh(),
        }
    }

    pub fn pretty(&self) -> &'static str {
        match self {
            UnaryOp::Neg => "-",
            UnaryOp::Inv => "1/",
            UnaryOp::Sqr => "sqr",
            UnaryOp::Sqrt => "sqrt",
            UnaryOp::Exp => "exp",
            UnaryOp::Ln => "ln",
            UnaryOp::Sin => "sin",
            UnaryOp::Cos => "cos",
            UnaryOp::Sinh => "sinh",
            UnaryOp::Tanh => "tanh",
        }
    }
}

/// Two-argument operations over the complex numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

impl BinaryOp {
    pub const ALL: [BinaryOp; 5] = [
        BinaryOp::Add,
        BinaryOp::Sub,
        BinaryOp::Mul,
        BinaryOp::Div,
        BinaryOp::Pow,
    ];

    pub fn eval(&self, a: C, b: C) -> C {
        match self {
            BinaryOp::Add => a + b,
            BinaryOp::Sub => a - b,
            BinaryOp::Mul => a * b,
            BinaryOp::Div => a / b,
            BinaryOp::Pow => a.powc(b),
        }
    }

    pub fn pretty(&self) -> &'static str {
        match self {
            BinaryOp::Add => "+",
            BinaryOp::Sub => "-",
            BinaryOp::Mul => "*",
            BinaryOp::Div => "/",
            BinaryOp::Pow => "^",
        }
    }
}

/// Expression tree for an enumerated operator. Each tree evaluates to a
/// `Complex<f64>` given the input pair `(x, y)`.
///
/// Children are `Arc<OpExpr>` so subtrees can be shared cheaply during
/// bottom-up enumeration — at MAX_SIZE=6 that's the difference between
/// ~150 MB (shared) and several gigabytes (cloned `Box<OpExpr>` copies).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OpExpr {
    Atom(Atom),
    Unary(UnaryOp, Arc<OpExpr>),
    Binary(BinaryOp, Arc<OpExpr>, Arc<OpExpr>),
}

impl OpExpr {
    /// Number of nodes (atoms + unary + binary).
    pub fn size(&self) -> usize {
        match self {
            OpExpr::Atom(_) => 1,
            OpExpr::Unary(_, e) => 1 + e.size(),
            OpExpr::Binary(_, a, b) => 1 + a.size() + b.size(),
        }
    }

    /// Evaluate the tree with given `x` and `y`. Overflows and domain
    /// errors propagate as NaN/inf; the scorer filters those out.
    pub fn eval(&self, x: C, y: C) -> C {
        match self {
            OpExpr::Atom(a) => a.eval(x, y),
            OpExpr::Unary(op, e) => op.eval(e.eval(x, y)),
            OpExpr::Binary(op, l, r) => op.eval(l.eval(x, y), r.eval(x, y)),
        }
    }

    /// Pretty-print using standard infix conventions for binary ops.
    /// Parenthesizes aggressively so the output round-trips visually.
    pub fn pretty(&self) -> String {
        match self {
            OpExpr::Atom(a) => a.pretty().to_string(),
            OpExpr::Unary(UnaryOp::Neg, e) => format!("-({})", e.pretty()),
            OpExpr::Unary(UnaryOp::Inv, e) => format!("1/({})", e.pretty()),
            OpExpr::Unary(op, e) => format!("{}({})", op.pretty(), e.pretty()),
            OpExpr::Binary(op, l, r) => format!("({} {} {})", l.pretty(), op.pretty(), r.pretty()),
        }
    }

    /// True if the tree contains at least one `Atom::X` and at least one
    /// `Atom::Y`. Trivial filter: an "operator" that ignores one input
    /// is not a binary operator at all.
    pub fn uses_both(&self) -> bool {
        fn walk(e: &OpExpr, seen_x: &mut bool, seen_y: &mut bool) {
            match e {
                OpExpr::Atom(Atom::X) => *seen_x = true,
                OpExpr::Atom(Atom::Y) => *seen_y = true,
                OpExpr::Atom(_) => {}
                OpExpr::Unary(_, inner) => walk(inner, seen_x, seen_y),
                OpExpr::Binary(_, l, r) => {
                    walk(l, seen_x, seen_y);
                    walk(r, seen_x, seen_y);
                }
            }
        }
        let (mut x, mut y) = (false, false);
        walk(self, &mut x, &mut y);
        x && y
    }
}

/// `Operator` impl for an enumerated `OpExpr`. The name is the
/// pretty-printed form so output is self-documenting.
#[derive(Debug, Clone)]
pub struct EnumOp {
    name: String,
    expr: OpExpr,
}

impl EnumOp {
    pub fn new(expr: OpExpr) -> Self {
        Self {
            name: expr.pretty(),
            expr,
        }
    }

    pub fn expr(&self) -> &OpExpr {
        &self.expr
    }
}

impl Operator for EnumOp {
    fn name(&self) -> &str {
        &self.name
    }
    fn eval(&self, x: C, y: C) -> C {
        self.expr.eval(x, y)
    }
}

// ---------------------------------------------------------------------------
// Enumeration
//
// Iterative bottom-up: build `by_size[k]` for k = 1..=max once, then
// larger sizes reference smaller ones via `Arc<OpExpr>` instead of
// cloning. No recomputation, no subtree duplication. For MAX_SIZE=6
// this is the difference between ~10 s + gigabytes of RAM and ~1 s +
// ~150 MB.

fn build_by_size(max_size: usize) -> Vec<Vec<Arc<OpExpr>>> {
    let mut by_size: Vec<Vec<Arc<OpExpr>>> = vec![Vec::new(); max_size + 1];
    if max_size == 0 {
        return by_size;
    }

    // Size 1: atoms.
    for a in Atom::ALL {
        by_size[1].push(Arc::new(OpExpr::Atom(a)));
    }

    // Sizes 2..=max: unary wrap + binary split.
    for n in 2..=max_size {
        // Unary over size n-1.
        let prev_len = by_size[n - 1].len();
        let mut level: Vec<Arc<OpExpr>> = Vec::with_capacity(prev_len * UnaryOp::ALL.len());
        for i in 0..prev_len {
            let inner = by_size[n - 1][i].clone();
            for op in UnaryOp::ALL {
                level.push(Arc::new(OpExpr::Unary(op, inner.clone())));
            }
        }

        // Binary over (k, n-1-k).
        if n >= 3 {
            for k in 1..=(n - 2) {
                let rk = n - 1 - k;
                let l_len = by_size[k].len();
                let r_len = by_size[rk].len();
                level.reserve(l_len * r_len * BinaryOp::ALL.len());
                for i in 0..l_len {
                    for j in 0..r_len {
                        let l = by_size[k][i].clone();
                        let r = by_size[rk][j].clone();
                        for op in BinaryOp::ALL {
                            level.push(Arc::new(OpExpr::Binary(op, l.clone(), r.clone())));
                        }
                    }
                }
            }
        }

        by_size[n] = level;
    }

    by_size
}

/// Generate all distinct `OpExpr` trees of *exactly* `size` nodes via
/// bottom-up iterative construction. Result shares subtrees via `Rc`.
pub fn trees_of_size(size: usize) -> Vec<Arc<OpExpr>> {
    if size == 0 {
        return Vec::new();
    }
    let by = build_by_size(size);
    by.into_iter().nth(size).unwrap_or_default()
}

/// Generate all `OpExpr` trees with size 1..=max_size as a single flat
/// vector. Produces the smallest sizes first. Uses an internal bottom-
/// up cache so each tree is built exactly once.
pub fn trees_up_to(max_size: usize) -> Vec<Arc<OpExpr>> {
    let by = build_by_size(max_size);
    let total: usize = by.iter().map(|v| v.len()).sum();
    let mut out = Vec::with_capacity(total);
    for level in by.into_iter() {
        out.extend(level);
    }
    out
}

/// Stream every tree with size 1..=max_size through a callback without
/// materializing a secondary flat Vec in the caller. The internal
/// bottom-up cache holds sizes 1..max_size (so later sizes can reference
/// earlier ones), but the TOP level is streamed without being stored —
/// once a size-`max_size` tree is handed to the callback, it's dropped
/// unless the callback kept a reference. This is the difference between
/// ~200 GB and ~15 GB at MAX_SIZE=8.
pub fn for_each_tree(max_size: usize, mut f: impl FnMut(&Arc<OpExpr>)) {
    if max_size == 0 {
        return;
    }
    let mut by_size: Vec<Vec<Arc<OpExpr>>> = vec![Vec::new(); max_size + 1];

    // Size 1: atoms, streamed.
    for a in Atom::ALL {
        let t = Arc::new(OpExpr::Atom(a));
        f(&t);
        by_size[1].push(t);
    }

    for n in 2..=max_size {
        let is_top = n == max_size;
        // Only accumulate into `level` if some future size will reference
        // it. When `n == max_size`, nothing downstream needs these trees,
        // so we stream them through and let the caller decide whether to
        // retain.
        let mut level: Vec<Arc<OpExpr>> = Vec::new();

        // Unary over size n-1.
        for i in 0..by_size[n - 1].len() {
            let inner = by_size[n - 1][i].clone();
            for op in UnaryOp::ALL {
                let t = Arc::new(OpExpr::Unary(op, inner.clone()));
                f(&t);
                if !is_top {
                    level.push(t);
                }
            }
        }

        // Binary over (k, n-1-k).
        if n >= 3 {
            for k in 1..=(n - 2) {
                let rk = n - 1 - k;
                for i in 0..by_size[k].len() {
                    for j in 0..by_size[rk].len() {
                        let l = by_size[k][i].clone();
                        let r = by_size[rk][j].clone();
                        for op in BinaryOp::ALL {
                            let t = Arc::new(OpExpr::Binary(op, l.clone(), r.clone()));
                            f(&t);
                            if !is_top {
                                level.push(t);
                            }
                        }
                    }
                }
            }
        }

        // Only store the level if it's not the streamed top.
        if !is_top {
            by_size[n] = level;
        }
    }
}

// ---------------------------------------------------------------------------
// Semantic dedup

/// Five conjecturally-algebraically-independent complex test pairs. An
/// `OpExpr` that evaluates to the same values at ALL of these is almost
/// certainly semantically equivalent to another with the same profile,
/// modulo Schanuel. More test points = fewer false dedup collapses.
const TEST_PAIRS: [(C, C); 5] = [
    (C { re: 0.5772156649015329, im: 0.0 }, C { re: 1.2824271291006226, im: 0.0 }),
    (C { re: 1.2824271291006226, im: 0.0 }, C { re: 0.9159655941772190, im: 0.0 }),
    (C { re: 0.9159655941772190, im: 0.0 }, C { re: 0.5772156649015329, im: 0.0 }),
    (C { re: 1.5, im: 0.3 }, C { re: 0.7, im: -0.4 }),
    (C { re: 0.4, im: 1.1 }, C { re: -0.6, im: 0.8 }),
];

/// Collapses semantically-equivalent `OpExpr` by their evaluation
/// fingerprint at the test pairs. Also rejects NaN/infinity candidates
/// (meaningless for our search).
pub struct DedupSet {
    seen: HashSet<[ValueKey; 5]>,
}

impl Default for DedupSet {
    fn default() -> Self {
        Self::new()
    }
}

impl DedupSet {
    pub fn new() -> Self {
        Self {
            seen: HashSet::new(),
        }
    }

    /// Try to insert `op`. Returns true if the operator is new (insert
    /// happened) and valid, false if duplicate or NaN/infinity.
    pub fn insert(&mut self, op: &OpExpr) -> bool {
        let zero = quantize(C::new(0.0, 0.0));
        let mut key = [zero; 5];
        for (i, (x, y)) in TEST_PAIRS.iter().enumerate() {
            let v = op.eval(*x, *y);
            if v.re.is_nan() || v.im.is_nan() {
                return false;
            }
            key[i] = quantize(v);
        }
        self.seen.insert(key)
    }

    pub fn len(&self) -> usize {
        self.seen.len()
    }

    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn atom(a: Atom) -> Arc<OpExpr> {
        Arc::new(OpExpr::Atom(a))
    }
    fn unary(op: UnaryOp, e: Arc<OpExpr>) -> Arc<OpExpr> {
        Arc::new(OpExpr::Unary(op, e))
    }
    fn binary(op: BinaryOp, l: Arc<OpExpr>, r: Arc<OpExpr>) -> Arc<OpExpr> {
        Arc::new(OpExpr::Binary(op, l, r))
    }

    #[test]
    fn size_1_is_six_atoms() {
        let trees = trees_of_size(1);
        assert_eq!(trees.len(), Atom::ALL.len());
    }

    #[test]
    fn size_2_is_sixty() {
        // 10 unary ops applied to each of 6 atoms = 60.
        let trees = trees_of_size(2);
        assert_eq!(trees.len(), UnaryOp::ALL.len() * Atom::ALL.len());
    }

    #[test]
    fn size_3_count() {
        // Size 3 = Unary(size 2) + Binary(size 1, size 1)
        //        = 10 * 60      + 5 * 6 * 6
        //        = 600 + 180 = 780.
        let trees = trees_of_size(3);
        assert_eq!(trees.len(), 780);
    }

    #[test]
    fn for_each_tree_matches_trees_up_to() {
        // Streaming callback and flat Vec should produce identical counts.
        let flat = trees_up_to(3);
        let mut streamed: usize = 0;
        for_each_tree(3, |_| streamed += 1);
        assert_eq!(flat.len(), streamed);
    }

    #[test]
    fn eval_simple_operators() {
        let x = C::new(2.0, 0.0);
        let y = C::new(3.0, 0.0);

        // x + y = 5
        let add = binary(BinaryOp::Add, atom(Atom::X), atom(Atom::Y));
        assert!((add.eval(x, y) - C::new(5.0, 0.0)).norm() < 1e-12);

        // exp(x) - ln(y) = EML
        let eml = binary(
            BinaryOp::Sub,
            unary(UnaryOp::Exp, atom(Atom::X)),
            unary(UnaryOp::Ln, atom(Atom::Y)),
        );
        let expected = x.exp() - y.ln();
        assert!((eml.eval(x, y) - expected).norm() < 1e-12);
    }

    #[test]
    fn uses_both_filter() {
        let just_x = unary(UnaryOp::Exp, atom(Atom::X));
        assert!(!just_x.uses_both());

        let both = binary(
            BinaryOp::Sub,
            unary(UnaryOp::Exp, atom(Atom::X)),
            unary(UnaryOp::Ln, atom(Atom::Y)),
        );
        assert!(both.uses_both());
    }

    #[test]
    fn dedup_collapses_commutative_add() {
        let mut d = DedupSet::new();
        let xy = binary(BinaryOp::Add, atom(Atom::X), atom(Atom::Y));
        let yx = binary(BinaryOp::Add, atom(Atom::Y), atom(Atom::X));
        assert!(d.insert(&xy));
        assert!(!d.insert(&yx), "x+y and y+x should collapse under dedup");
    }

    #[test]
    fn dedup_keeps_distinct_operators() {
        let mut d = DedupSet::new();
        // Build EML and EDL as OpExprs; they should be distinct under dedup.
        let eml = binary(
            BinaryOp::Sub,
            unary(UnaryOp::Exp, atom(Atom::X)),
            unary(UnaryOp::Ln, atom(Atom::Y)),
        );
        let edl = binary(
            BinaryOp::Div,
            unary(UnaryOp::Exp, atom(Atom::X)),
            unary(UnaryOp::Ln, atom(Atom::Y)),
        );
        assert!(d.insert(&eml));
        assert!(d.insert(&edl));
    }
}
