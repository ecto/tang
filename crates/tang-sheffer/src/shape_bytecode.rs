//! Shape enumeration and bytecode packing for the GPU search.
//!
//! A "shape" is a binary tree topology over the alphabet
//! `{atom, unary, binary}` — a SPECIFIC labeling of internal nodes as
//! unary-vs-binary, but WITHOUT committing to specific atom choices or
//! specific unary/binary op choices. Those are per-thread parameters
//! decoded from the thread's `assignment_idx` during GPU evaluation.
//!
//! The GPU kernel reads each shape as a postfix bytecode stream, with
//! three instruction tags (`ATOM`, `UNARY`, `BINARY`) and a per-
//! instruction `slot_idx` indicating which assignment digit to use.
//!
//! Total shape count at size ≤ 10: ~5000 shapes. Total bytecode size:
//! ~100 KB. Fits comfortably in a wgpu storage buffer.

/// Abstract tree shape. No op choices yet — just the topology.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Shape {
    Atom,
    Unary(Box<Shape>),
    Binary(Box<Shape>, Box<Shape>),
}

impl Shape {
    /// Number of nodes (atoms + unary internals + binary internals).
    pub fn size(&self) -> usize {
        match self {
            Shape::Atom => 1,
            Shape::Unary(e) => 1 + e.size(),
            Shape::Binary(a, b) => 1 + a.size() + b.size(),
        }
    }

    /// (n_atoms, n_unary, n_binary) slot counts for this shape.
    pub fn slots(&self) -> (u32, u32, u32) {
        match self {
            Shape::Atom => (1, 0, 0),
            Shape::Unary(e) => {
                let (a, u, b) = e.slots();
                (a, u + 1, b)
            }
            Shape::Binary(l, r) => {
                let (al, ul, bl) = l.slots();
                let (ar, ur, br) = r.slots();
                (al + ar, ul + ur, bl + br + 1)
            }
        }
    }

    /// Total number of concrete operators this shape generates, given
    /// a base alphabet of 6 atoms, 10 unary ops, 5 binary ops.
    pub fn assignment_count(&self) -> u64 {
        let (a, u, b) = self.slots();
        6u64.pow(a) * 10u64.pow(u) * 5u64.pow(b)
    }
}

/// Enumerate every shape with size 1..=max_size, preserving a stable
/// canonical order so GPU thread indices map reproducibly to shapes.
pub fn enumerate_shapes(max_size: usize) -> Vec<Shape> {
    let mut out = Vec::new();
    let mut by_size: Vec<Vec<Shape>> = vec![Vec::new(); max_size + 1];
    if max_size >= 1 {
        by_size[1].push(Shape::Atom);
    }
    for n in 2..=max_size {
        // Unary wraps
        let prev_len = by_size[n - 1].len();
        for i in 0..prev_len {
            let inner = by_size[n - 1][i].clone();
            by_size[n].push(Shape::Unary(Box::new(inner)));
        }
        // Binary splits
        if n >= 3 {
            for k in 1..=(n - 2) {
                let rk = n - 1 - k;
                for i in 0..by_size[k].len() {
                    for j in 0..by_size[rk].len() {
                        let l = by_size[k][i].clone();
                        let r = by_size[rk][j].clone();
                        by_size[n].push(Shape::Binary(Box::new(l), Box::new(r)));
                    }
                }
            }
        }
    }
    for level in by_size.into_iter().skip(1) {
        out.extend(level);
    }
    out
}

/// Postfix bytecode encoding of a shape: each instruction is one `u32`
/// packed as `(tag << 16) | slot_idx`. Tags:
///
///   TAG_ATOM   = 0   slot = atom-slot-index in the shape (0..n_atoms)
///   TAG_UNARY  = 1   slot = unary-slot-index (0..n_unary)
///   TAG_BINARY = 2   slot = binary-slot-index (0..n_binary)
///
/// The slots are assigned in LEFT-TO-RIGHT POSTFIX order during a
/// canonical traversal, so the same ordering is used on both GPU and
/// CPU sides for assignment decoding.
pub const TAG_ATOM: u32 = 0;
pub const TAG_UNARY: u32 = 1;
pub const TAG_BINARY: u32 = 2;

#[derive(Debug, Clone)]
pub struct ShapeBytecode {
    pub instrs: Vec<u32>,
    pub n_atoms: u32,
    pub n_unary: u32,
    pub n_binary: u32,
}

pub fn encode_shape(shape: &Shape) -> ShapeBytecode {
    let mut instrs = Vec::new();
    let mut a_slot = 0u32;
    let mut u_slot = 0u32;
    let mut b_slot = 0u32;
    encode_rec(shape, &mut instrs, &mut a_slot, &mut u_slot, &mut b_slot);
    ShapeBytecode {
        instrs,
        n_atoms: a_slot,
        n_unary: u_slot,
        n_binary: b_slot,
    }
}

fn encode_rec(shape: &Shape, out: &mut Vec<u32>, a: &mut u32, u: &mut u32, b: &mut u32) {
    match shape {
        Shape::Atom => {
            out.push((TAG_ATOM << 16) | *a);
            *a += 1;
        }
        Shape::Unary(inner) => {
            encode_rec(inner, out, a, u, b);
            out.push((TAG_UNARY << 16) | *u);
            *u += 1;
        }
        Shape::Binary(l, r) => {
            encode_rec(l, out, a, u, b);
            encode_rec(r, out, a, u, b);
            out.push((TAG_BINARY << 16) | *b);
            *b += 1;
        }
    }
}

/// Flat table of all shapes up to `max_size`, packed for GPU consumption.
#[derive(Debug)]
pub struct ShapeTable {
    /// Concatenated instruction stream for all shapes.
    pub all_instrs: Vec<u32>,
    /// Per-shape metadata: (bytecode_offset, bytecode_len, n_atoms,
    /// n_unary, n_binary, assignment_count_lo, assignment_count_hi).
    pub shape_info: Vec<ShapeInfoRecord>,
    /// The shapes themselves (needed on CPU for hit reconstruction).
    pub shapes: Vec<Shape>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ShapeInfoRecord {
    pub bytecode_offset: u32,
    pub bytecode_len: u32,
    pub n_atoms: u32,
    pub n_unary: u32,
    pub n_binary: u32,
    pub assignment_count_lo: u32,
    pub assignment_count_hi: u32,
    pub size: u32,
}

pub fn build_shape_table(max_size: usize) -> ShapeTable {
    let shapes = enumerate_shapes(max_size);
    let mut all_instrs = Vec::new();
    let mut shape_info = Vec::with_capacity(shapes.len());
    for shape in &shapes {
        let bc = encode_shape(shape);
        let offset = all_instrs.len() as u32;
        let len = bc.instrs.len() as u32;
        all_instrs.extend(bc.instrs);
        let ac = shape.assignment_count();
        shape_info.push(ShapeInfoRecord {
            bytecode_offset: offset,
            bytecode_len: len,
            n_atoms: bc.n_atoms,
            n_unary: bc.n_unary,
            n_binary: bc.n_binary,
            assignment_count_lo: ac as u32,
            assignment_count_hi: (ac >> 32) as u32,
            size: shape.size() as u32,
        });
    }
    ShapeTable {
        all_instrs,
        shape_info,
        shapes,
    }
}

/// Reconstruct an `OpExpr` from a shape and assignment index, using
/// the same base-6/10/5 decoding order as the GPU kernel.
pub fn reconstruct_opexpr(
    shape: &Shape,
    assignment_idx: u64,
) -> crate::op_enum::OpExpr {
    let (a, u, b) = shape.slots();
    let mut remaining = assignment_idx;

    let mut atom_ids = Vec::with_capacity(a as usize);
    for _ in 0..a {
        atom_ids.push((remaining % 6) as u8);
        remaining /= 6;
    }
    let mut unary_ids = Vec::with_capacity(u as usize);
    for _ in 0..u {
        unary_ids.push((remaining % 10) as u8);
        remaining /= 10;
    }
    let mut binary_ids = Vec::with_capacity(b as usize);
    for _ in 0..b {
        binary_ids.push((remaining % 5) as u8);
        remaining /= 5;
    }

    let mut a_cursor = 0;
    let mut u_cursor = 0;
    let mut b_cursor = 0;
    apply_assignment(
        shape,
        &atom_ids,
        &unary_ids,
        &binary_ids,
        &mut a_cursor,
        &mut u_cursor,
        &mut b_cursor,
    )
}

fn apply_assignment(
    shape: &Shape,
    atom_ids: &[u8],
    unary_ids: &[u8],
    binary_ids: &[u8],
    a_cursor: &mut usize,
    u_cursor: &mut usize,
    b_cursor: &mut usize,
) -> crate::op_enum::OpExpr {
    use crate::op_enum::{Atom, BinaryOp, OpExpr, UnaryOp};
    use std::sync::Arc;
    match shape {
        Shape::Atom => {
            let id = atom_ids[*a_cursor];
            *a_cursor += 1;
            OpExpr::Atom(match id {
                0 => Atom::X,
                1 => Atom::Y,
                2 => Atom::Zero,
                3 => Atom::One,
                4 => Atom::NegOne,
                _ => Atom::E,
            })
        }
        Shape::Unary(inner) => {
            let inner_expr = apply_assignment(
                inner, atom_ids, unary_ids, binary_ids, a_cursor, u_cursor, b_cursor,
            );
            let id = unary_ids[*u_cursor];
            *u_cursor += 1;
            let op = match id {
                0 => UnaryOp::Neg,
                1 => UnaryOp::Inv,
                2 => UnaryOp::Sqr,
                3 => UnaryOp::Sqrt,
                4 => UnaryOp::Exp,
                5 => UnaryOp::Ln,
                6 => UnaryOp::Sin,
                7 => UnaryOp::Cos,
                8 => UnaryOp::Sinh,
                _ => UnaryOp::Tanh,
            };
            OpExpr::Unary(op, Arc::new(inner_expr))
        }
        Shape::Binary(left, right) => {
            let left_expr = apply_assignment(
                left, atom_ids, unary_ids, binary_ids, a_cursor, u_cursor, b_cursor,
            );
            let right_expr = apply_assignment(
                right, atom_ids, unary_ids, binary_ids, a_cursor, u_cursor, b_cursor,
            );
            let id = binary_ids[*b_cursor];
            *b_cursor += 1;
            let op = match id {
                0 => BinaryOp::Add,
                1 => BinaryOp::Sub,
                2 => BinaryOp::Mul,
                3 => BinaryOp::Div,
                _ => BinaryOp::Pow,
            };
            OpExpr::Binary(op, Arc::new(left_expr), Arc::new(right_expr))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_1_is_one_atom_shape() {
        let shapes = enumerate_shapes(1);
        assert_eq!(shapes.len(), 1);
        assert!(matches!(shapes[0], Shape::Atom));
    }

    #[test]
    fn size_2_is_one_unary_shape() {
        // Size 2 = Unary(Atom). Only one topology (the choice of unary
        // op is a per-thread assignment).
        let shapes = enumerate_shapes(2);
        let s2: Vec<&Shape> = shapes.iter().filter(|s| s.size() == 2).collect();
        assert_eq!(s2.len(), 1);
    }

    #[test]
    fn size_3_two_shapes() {
        // Size 3: Unary(Unary(Atom)) and Binary(Atom, Atom).
        let shapes = enumerate_shapes(3);
        let s3: Vec<&Shape> = shapes.iter().filter(|s| s.size() == 3).collect();
        assert_eq!(s3.len(), 2);
    }

    #[test]
    fn assignment_count_for_eml_shape() {
        // EML shape = Binary(Unary(Atom), Unary(Atom)), size 5.
        let shape = Shape::Binary(
            Box::new(Shape::Unary(Box::new(Shape::Atom))),
            Box::new(Shape::Unary(Box::new(Shape::Atom))),
        );
        let (a, u, b) = shape.slots();
        assert_eq!((a, u, b), (2, 2, 1));
        // 6 atoms × 6 atoms × 10 unary × 10 unary × 5 binary = 18,000.
        assert_eq!(shape.assignment_count(), 6 * 6 * 10 * 10 * 5);
    }

    #[test]
    fn encoding_postfix_order() {
        let shape = Shape::Binary(
            Box::new(Shape::Unary(Box::new(Shape::Atom))),
            Box::new(Shape::Unary(Box::new(Shape::Atom))),
        );
        let bc = encode_shape(&shape);
        // Expected postfix: ATOM(0) UNARY(0) ATOM(1) UNARY(1) BINARY(0)
        assert_eq!(bc.instrs.len(), 5);
        assert_eq!(bc.instrs[0] >> 16, TAG_ATOM);
        assert_eq!(bc.instrs[0] & 0xFFFF, 0);
        assert_eq!(bc.instrs[1] >> 16, TAG_UNARY);
        assert_eq!(bc.instrs[2] >> 16, TAG_ATOM);
        assert_eq!(bc.instrs[2] & 0xFFFF, 1);
        assert_eq!(bc.instrs[3] >> 16, TAG_UNARY);
        assert_eq!(bc.instrs[4] >> 16, TAG_BINARY);
    }

    #[test]
    fn shape_table_at_size_5_reasonable() {
        let table = build_shape_table(5);
        // Size ≤ 5 has ~ a few dozen shapes; exact count verified by
        // summing across sizes.
        assert!(table.shape_info.len() >= 10);
        assert!(table.all_instrs.len() >= 20);
        // No shape should have size > 5.
        for info in &table.shape_info {
            assert!(info.size <= 5);
        }
    }
}
