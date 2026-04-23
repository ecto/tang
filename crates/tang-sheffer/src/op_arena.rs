//! Arena-backed variant of `OpExpr` enumeration for large-scale searches.
//!
//! For MAX_SIZE=7 and below, the `Arc<OpExpr>` representation in
//! `op_enum` is fine — ~150 MB at MAX_SIZE=5, ~12 GB at MAX_SIZE=7.
//! But at MAX_SIZE=8 the same representation uses ~25 GB, and at
//! MAX_SIZE=9 it would need ~200+ GB — out of reach on any dev box.
//!
//! This module replaces `Arc<OpExpr>` with a flat `Vec<Node>` arena
//! where each tree is a `NodeId = u32` index. Subtree sharing is
//! automatic: when tree A contains subtree S and tree B also contains
//! S, both store the same child index pointing to S's root node in the
//! arena. A single `Node` is ~12 bytes (vs ~240 bytes for one heap-
//! allocated `Arc<OpExpr>`), giving a ~20× memory reduction at scale.
//!
//! The arena is immutable after construction (each `push` appends
//! only), so it's trivially `Send + Sync` for rayon-parallel scoring.

use std::sync::Arc;

use crate::op_enum::{Atom, BinaryOp, UnaryOp};
use crate::operator::Operator;
use crate::verify::quantize;
use crate::C;

/// Handle into an `OpArena`. 32 bits is enough for any enumeration we'd
/// reasonably do on a single machine (~4B distinct nodes).
pub type NodeId = u32;

/// One node of an expression tree in the arena. Sized to ~12 bytes on
/// typical platforms after enum discriminant packing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Node {
    Atom(Atom),
    Unary(UnaryOp, NodeId),
    Binary(BinaryOp, NodeId, NodeId),
}

/// Flat arena of expression-tree nodes. Append-only — nodes are never
/// removed or rewritten after insertion, so indices are stable for the
/// lifetime of the arena.
#[derive(Debug, Default)]
pub struct OpArena {
    nodes: Vec<Node>,
}

impl OpArena {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(cap),
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Append a node and return its `NodeId`.
    #[inline]
    pub fn push(&mut self, node: Node) -> NodeId {
        let id = self.nodes.len() as u32;
        self.nodes.push(node);
        id
    }

    #[inline]
    pub fn get(&self, id: NodeId) -> Node {
        self.nodes[id as usize]
    }

    /// Recursively count nodes in the subtree rooted at `id`.
    pub fn size(&self, id: NodeId) -> usize {
        match self.get(id) {
            Node::Atom(_) => 1,
            Node::Unary(_, c) => 1 + self.size(c),
            Node::Binary(_, l, r) => 1 + self.size(l) + self.size(r),
        }
    }

    /// Depth of the longest root-to-leaf path, in internal-op counts.
    pub fn depth(&self, id: NodeId) -> usize {
        match self.get(id) {
            Node::Atom(_) => 0,
            Node::Unary(_, c) => 1 + self.depth(c),
            Node::Binary(_, l, r) => 1 + self.depth(l).max(self.depth(r)),
        }
    }

    /// Evaluate the subtree at `id` given `x, y`. Propagates NaN/inf.
    pub fn eval(&self, id: NodeId, x: C, y: C) -> C {
        match self.get(id) {
            Node::Atom(a) => a.eval(x, y),
            Node::Unary(op, c) => op.eval(self.eval(c, x, y)),
            Node::Binary(op, l, r) => op.eval(self.eval(l, x, y), self.eval(r, x, y)),
        }
    }

    /// True iff the subtree contains at least one `Atom::X` and one
    /// `Atom::Y`. Trivial filter for "operator ignores one input".
    pub fn uses_both(&self, id: NodeId) -> bool {
        fn walk(arena: &OpArena, id: NodeId, sx: &mut bool, sy: &mut bool) {
            match arena.get(id) {
                Node::Atom(Atom::X) => *sx = true,
                Node::Atom(Atom::Y) => *sy = true,
                Node::Atom(_) => {}
                Node::Unary(_, c) => walk(arena, c, sx, sy),
                Node::Binary(_, l, r) => {
                    walk(arena, l, sx, sy);
                    walk(arena, r, sx, sy);
                }
            }
        }
        let (mut x, mut y) = (false, false);
        walk(self, id, &mut x, &mut y);
        x && y
    }

    /// Pretty-print the subtree using infix notation for binary ops.
    pub fn pretty(&self, id: NodeId) -> String {
        match self.get(id) {
            Node::Atom(a) => a.pretty().to_string(),
            Node::Unary(UnaryOp::Neg, c) => format!("-({})", self.pretty(c)),
            Node::Unary(UnaryOp::Inv, c) => format!("1/({})", self.pretty(c)),
            Node::Unary(op, c) => format!("{}({})", op.pretty(), self.pretty(c)),
            Node::Binary(op, l, r) => format!(
                "({} {} {})",
                self.pretty(l),
                op.pretty(),
                self.pretty(r)
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Enumeration in the arena

/// Build an arena containing every tree with size 1..=max_size, and
/// return a per-level index `by_size[k]` listing the root `NodeId`s of
/// each tree with size exactly `k`.
///
/// Memory: arena has ~sum(trees(k)) for k=1..=max_size nodes, each ~12
/// bytes. At max_size=8 that's ~925M nodes ≈ 11 GB (vs ~25 GB for the
/// Arc-based version).
pub fn build_arena(max_size: usize) -> (OpArena, Vec<Vec<NodeId>>) {
    let mut arena = OpArena::new();
    let mut by_size: Vec<Vec<NodeId>> = vec![Vec::new(); max_size + 1];
    if max_size == 0 {
        return (arena, by_size);
    }

    // Size 1: atoms.
    for a in Atom::ALL {
        let id = arena.push(Node::Atom(a));
        by_size[1].push(id);
    }

    for n in 2..=max_size {
        // Unary wraps of size n-1.
        let prev_len = by_size[n - 1].len();
        for i in 0..prev_len {
            let inner = by_size[n - 1][i];
            for op in UnaryOp::ALL {
                let id = arena.push(Node::Unary(op, inner));
                by_size[n].push(id);
            }
        }
        // Binary splits.
        if n >= 3 {
            for k in 1..=(n - 2) {
                let rk = n - 1 - k;
                let l_len = by_size[k].len();
                let r_len = by_size[rk].len();
                for i in 0..l_len {
                    for j in 0..r_len {
                        let l = by_size[k][i];
                        let r = by_size[rk][j];
                        for op in BinaryOp::ALL {
                            let id = arena.push(Node::Binary(op, l, r));
                            by_size[n].push(id);
                        }
                    }
                }
            }
        }
    }

    (arena, by_size)
}

/// Stream every tree with size 1..=max_size through a callback. Like
/// `build_arena` but doesn't store size-`max_size` nodes in the arena —
/// it streams them via the callback instead. The internal arena is
/// smaller (sizes 1..max_size-1 only) and the caller handles its own
/// dedup/filtering via the callback.
///
/// Still constructs every tree node (can't avoid that), but the top-
/// level allocation is avoided: the callback gets a synthetic `Node`
/// with child references into the arena, and unless the callback keeps
/// the node elsewhere, it's dropped immediately.
pub fn for_each_tree_in_arena(
    max_size: usize,
    mut f: impl FnMut(&OpArena, Node),
) {
    if max_size == 0 {
        return;
    }
    // Build the cache up to max_size - 1, then stream max_size.
    let cache_up_to = max_size.saturating_sub(1).max(1);
    let (arena, by_size) = build_arena(cache_up_to);

    // Stream sizes 1..=cache_up_to first: they're already in the arena.
    for n in 1..=cache_up_to {
        for id in &by_size[n] {
            f(&arena, arena.get(*id));
        }
    }

    if max_size <= cache_up_to {
        return;
    }
    let n = max_size;

    // Unary wraps of size n-1. These are new nodes, streamed.
    for id in &by_size[n - 1] {
        for op in UnaryOp::ALL {
            let node = Node::Unary(op, *id);
            f(&arena, node);
        }
    }

    // Binary splits. Each produces a new Node that references existing
    // arena IDs. The node is created on the stack, passed to the
    // callback, and dropped — no arena growth.
    if n >= 3 {
        for k in 1..=(n - 2) {
            let rk = n - 1 - k;
            for i in 0..by_size[k].len() {
                for j in 0..by_size[rk].len() {
                    let l = by_size[k][i];
                    let r = by_size[rk][j];
                    for op in BinaryOp::ALL {
                        let node = Node::Binary(op, l, r);
                        f(&arena, node);
                    }
                }
            }
        }
    }

    // Keep the arena alive for the duration of the callback stream.
    drop(arena);
}

// ---------------------------------------------------------------------------
// Arena-aware DedupSet

/// Five conjecturally-algebraically-independent complex test pairs.
/// (Same as `op_enum::DedupSet`.)
const TEST_PAIRS: [(C, C); 5] = [
    (C { re: 0.5772156649015329, im: 0.0 }, C { re: 1.2824271291006226, im: 0.0 }),
    (C { re: 1.2824271291006226, im: 0.0 }, C { re: 0.9159655941772190, im: 0.0 }),
    (C { re: 0.9159655941772190, im: 0.0 }, C { re: 0.5772156649015329, im: 0.0 }),
    (C { re: 1.5, im: 0.3 }, C { re: 0.7, im: -0.4 }),
    (C { re: 0.4, im: 1.1 }, C { re: -0.6, im: 0.8 }),
];

/// DedupSet for arena nodes. Keys are a single `u64` hash of the
/// 5-point evaluation fingerprint rather than the full 5 `ValueKey`s.
/// This shrinks per-entry memory from ~120 bytes (HashSet<[ValueKey; 5]>)
/// to ~20 bytes (HashSet<u64>) — essential at MAX_SIZE=9 where the
/// dedup set grows to hundreds of millions of entries.
///
/// Collision risk: at 300M entries the birthday probability of any
/// accidental collision is ~5e-3 across the whole set — acceptable for
/// exploratory search. For the final verification pass we re-run each
/// surviving candidate at full `[ValueKey; 5]` precision in
/// `op_enum::DedupSet` or (better) via multi-point cross-check with
/// `crosscheck::cross_check`.
#[derive(Default)]
pub struct ArenaDedupSet {
    seen: std::collections::HashSet<u64>,
}

impl ArenaDedupSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.seen.len()
    }

    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
    }

    /// Insert the fingerprint of `arena.eval(id, ...)` at all test
    /// pairs. Returns true if new (and valid), false if duplicate or
    /// NaN-producing.
    ///
    /// The fingerprint is a single `u64` hash of the five quantized
    /// complex evaluations, computed with the default `DefaultHasher`.
    /// This is a lossy fingerprint — two truly distinct operators have
    /// a ~1/2^64 chance of colliding per pair, and the dedup HashSet
    /// has a compound birthday-paradox collision at ~5e-3 for 300M
    /// entries. In exchange, memory drops by ~85%.
    pub fn insert_node(&mut self, arena: &OpArena, node: Node) -> bool {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        for (x, y) in &TEST_PAIRS {
            let v = eval_node_view(arena, node, *x, *y);
            if v.re.is_nan() || v.im.is_nan() {
                return false;
            }
            let k = quantize(v);
            k.hash(&mut h);
        }
        let fingerprint = h.finish();
        self.seen.insert(fingerprint)
    }

    pub fn insert_id(&mut self, arena: &OpArena, id: NodeId) -> bool {
        self.insert_node(arena, arena.get(id))
    }
}

/// Evaluate a top-level `Node` view (whose children point into the
/// arena) at `(x, y)` without storing the node in the arena first.
pub fn eval_node_view(arena: &OpArena, node: Node, x: C, y: C) -> C {
    match node {
        Node::Atom(a) => a.eval(x, y),
        Node::Unary(op, c) => op.eval(arena.eval(c, x, y)),
        Node::Binary(op, l, r) => op.eval(arena.eval(l, x, y), arena.eval(r, x, y)),
    }
}

/// Check if a streamed top-level node uses both `x` and `y`.
pub fn uses_both_view(arena: &OpArena, node: Node) -> bool {
    fn walk(arena: &OpArena, id: NodeId, sx: &mut bool, sy: &mut bool) {
        match arena.get(id) {
            Node::Atom(Atom::X) => *sx = true,
            Node::Atom(Atom::Y) => *sy = true,
            Node::Atom(_) => {}
            Node::Unary(_, c) => walk(arena, c, sx, sy),
            Node::Binary(_, l, r) => {
                walk(arena, l, sx, sy);
                walk(arena, r, sx, sy);
            }
        }
    }
    let (mut x, mut y) = (false, false);
    match node {
        Node::Atom(Atom::X) => x = true,
        Node::Atom(Atom::Y) => y = true,
        Node::Atom(_) => {}
        Node::Unary(_, c) => walk(arena, c, &mut x, &mut y),
        Node::Binary(_, l, r) => {
            walk(arena, l, &mut x, &mut y);
            walk(arena, r, &mut x, &mut y);
        }
    }
    x && y
}

// ---------------------------------------------------------------------------
// Operator adapter

/// Arena-backed implementation of the `Operator` trait. Holds an
/// `Arc<OpArena>` (so clones are cheap for rayon-parallel scoring) and
/// a root `NodeId`. The displayed name is the arena's pretty-print of
/// that root.
#[derive(Clone)]
pub struct ArenaOp {
    name: String,
    arena: Arc<OpArena>,
    root: NodeId,
}

impl ArenaOp {
    pub fn new(arena: Arc<OpArena>, root: NodeId) -> Self {
        let name = arena.pretty(root);
        Self { name, arena, root }
    }

    pub fn root(&self) -> NodeId {
        self.root
    }

    pub fn arena(&self) -> &OpArena {
        &self.arena
    }

    pub fn size(&self) -> usize {
        self.arena.size(self.root)
    }
}

impl std::fmt::Debug for ArenaOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArenaOp").field("name", &self.name).finish()
    }
}

impl Operator for ArenaOp {
    fn name(&self) -> &str {
        &self.name
    }
    fn eval(&self, x: C, y: C) -> C {
        self.arena.eval(self.root, x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_1_is_six_atoms() {
        let (arena, by_size) = build_arena(1);
        assert_eq!(by_size[1].len(), Atom::ALL.len());
        assert_eq!(arena.len(), Atom::ALL.len());
    }

    #[test]
    fn size_counts_match_op_enum() {
        let (_, by_size) = build_arena(4);
        // Same counts as op_enum::trees_of_size (verified in op_enum
        // unit tests): 6, 60, 780, 11400 at sizes 1..4.
        assert_eq!(by_size[1].len(), 6);
        assert_eq!(by_size[2].len(), 60);
        assert_eq!(by_size[3].len(), 780);
        assert_eq!(by_size[4].len(), 11400);
    }

    #[test]
    fn eval_matches_op_enum_eml() {
        // Build eml = exp(x) - ln(y) in the arena and compare its
        // evaluation to a direct complex computation.
        let mut arena = OpArena::new();
        let x = arena.push(Node::Atom(Atom::X));
        let y = arena.push(Node::Atom(Atom::Y));
        let exp_x = arena.push(Node::Unary(UnaryOp::Exp, x));
        let ln_y = arena.push(Node::Unary(UnaryOp::Ln, y));
        let eml = arena.push(Node::Binary(BinaryOp::Sub, exp_x, ln_y));

        let xv = C::new(1.5, 0.0);
        let yv = C::new(2.0, 0.0);
        let got = arena.eval(eml, xv, yv);
        let expected = xv.exp() - yv.ln();
        assert!((got - expected).norm() < 1e-12);
    }

    #[test]
    fn uses_both_arena() {
        let mut arena = OpArena::new();
        let x = arena.push(Node::Atom(Atom::X));
        let y = arena.push(Node::Atom(Atom::Y));
        let exp_x = arena.push(Node::Unary(UnaryOp::Exp, x));
        let just_x = arena.push(Node::Unary(UnaryOp::Exp, exp_x));
        assert!(!arena.uses_both(just_x));

        let ln_y = arena.push(Node::Unary(UnaryOp::Ln, y));
        let both = arena.push(Node::Binary(BinaryOp::Sub, exp_x, ln_y));
        assert!(arena.uses_both(both));
    }

    #[test]
    fn dedup_collapses_commutative_add() {
        let mut arena = OpArena::new();
        let x = arena.push(Node::Atom(Atom::X));
        let y = arena.push(Node::Atom(Atom::Y));
        let xy = arena.push(Node::Binary(BinaryOp::Add, x, y));
        let yx = arena.push(Node::Binary(BinaryOp::Add, y, x));
        let mut d = ArenaDedupSet::new();
        assert!(d.insert_id(&arena, xy));
        assert!(!d.insert_id(&arena, yx));
    }
}
