//! Expression trees and leaves.
//!
//! An Expr is a binary tree over a fixed pool of Leaves, with every internal
//! node applying the same (unspecified) binary operator. Leaves carry their
//! numeric value at the test point; Expr::eval walks the tree applying the
//! operator, re-reading leaf values from the pool each time.

use std::rc::Rc;

use crate::operator::Operator;
use crate::C;

/// How a leaf came to be in the pool.
#[derive(Debug, Clone)]
pub enum LeafSource {
    /// A hard-coded constant (e.g. 1, e).
    Constant,
    /// A free variable bound to a transcendental test value (e.g. x = γ).
    Variable,
    /// A formula previously discovered during bootstrapping and reused as a
    /// primitive on the next iteration.
    Derived { original: Rc<Expr> },
}

#[derive(Debug, Clone)]
pub struct Leaf {
    pub name: String,
    pub value: C,
    pub source: LeafSource,
}

impl Leaf {
    pub fn constant(name: impl Into<String>, value: C) -> Self {
        Self {
            name: name.into(),
            value,
            source: LeafSource::Constant,
        }
    }

    pub fn variable(name: impl Into<String>, value: C) -> Self {
        Self {
            name: name.into(),
            value,
            source: LeafSource::Variable,
        }
    }
}

/// Expression tree over a single binary operator. Leaves reference the pool
/// by index so every Expr is interpreted against a specific leaf set.
#[derive(Debug, Clone)]
pub enum Expr {
    Leaf(usize),
    Op(Rc<Expr>, Rc<Expr>),
}

impl Expr {
    /// Total node count (leaves + internal ops). Always odd.
    pub fn size(&self) -> usize {
        match self {
            Expr::Leaf(_) => 1,
            Expr::Op(a, b) => 1 + a.size() + b.size(),
        }
    }

    /// Number of internal operator applications.
    pub fn ops(&self) -> usize {
        (self.size() - 1) / 2
    }

    /// Depth of the longest root-to-leaf path (in op nodes).
    pub fn depth(&self) -> usize {
        match self {
            Expr::Leaf(_) => 0,
            Expr::Op(a, b) => 1 + a.depth().max(b.depth()),
        }
    }

    /// Node count when every Derived leaf is recursively replaced by its
    /// original expression. This is the "true" size in the base alphabet —
    /// size() counts a bootstrapped leaf as 1 node even though it may expand
    /// to dozens of base-operator applications.
    pub fn expanded_size(&self, leaves: &[Leaf]) -> usize {
        match self {
            Expr::Leaf(i) => match &leaves[*i].source {
                LeafSource::Derived { original } => original.expanded_size(leaves),
                _ => 1,
            },
            Expr::Op(a, b) => 1 + a.expanded_size(leaves) + b.expanded_size(leaves),
        }
    }

    /// Same as `format` but expands Derived leaves recursively so the output
    /// is in the base alphabet (constants + variables only). Useful for
    /// auditing bootstrapped discoveries.
    pub fn format_expanded(&self, op_name: &str, leaves: &[Leaf]) -> String {
        match self {
            Expr::Leaf(i) => match &leaves[*i].source {
                LeafSource::Derived { original } => original.format_expanded(op_name, leaves),
                _ => leaves[*i].name.clone(),
            },
            Expr::Op(a, b) => format!(
                "{}({}, {})",
                op_name,
                a.format_expanded(op_name, leaves),
                b.format_expanded(op_name, leaves),
            ),
        }
    }

    /// Evaluate at the stored test point: leaves look up by index, ops
    /// apply recursively. Uses the leaf's pre-computed `value` field for
    /// Derived leaves, so the result is only valid at the test point the
    /// leaf was discovered at.
    pub fn eval(&self, op: &dyn Operator, leaves: &[Leaf]) -> C {
        match self {
            Expr::Leaf(i) => leaves[*i].value,
            Expr::Op(a, b) => {
                let av = a.eval(op, leaves);
                let bv = b.eval(op, leaves);
                op.eval(av, bv)
            }
        }
    }

    /// Evaluate at the current leaf values, recursively re-computing Derived
    /// leaves from their original expression. This is the correct routine
    /// when the leaf pool has had its numerical values replaced (e.g. for
    /// multi-test-point cross-checking): Derived leaves have stale cached
    /// values and must be re-evaluated against the new base-leaf values.
    pub fn eval_recursive(&self, op: &dyn Operator, leaves: &[Leaf]) -> C {
        match self {
            Expr::Leaf(i) => match &leaves[*i].source {
                LeafSource::Derived { original } => original.eval_recursive(op, leaves),
                _ => leaves[*i].value,
            },
            Expr::Op(a, b) => {
                let av = a.eval_recursive(op, leaves);
                let bv = b.eval_recursive(op, leaves);
                op.eval(av, bv)
            }
        }
    }

    /// Pretty-print using the operator's name and leaf names.
    pub fn format(&self, op_name: &str, leaves: &[Leaf]) -> String {
        match self {
            Expr::Leaf(i) => leaves[*i].name.clone(),
            Expr::Op(a, b) => format!(
                "{}({}, {})",
                op_name,
                a.format(op_name, leaves),
                b.format(op_name, leaves),
            ),
        }
    }
}
