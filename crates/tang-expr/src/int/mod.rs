//! Integer dialect: exact 32-bit integer graphs with loop-carried state.
//!
//! A separate graph from the f64 RISC [`ExprGraph`](crate::ExprGraph): the
//! float ops are untouched, and nothing here is ever reassociated, contracted
//! or simplified beyond folding constant-only nodes with the interpreter's own
//! semantics. The contract is bit-exactness: [`IntGraph::eval`] (the CPU
//! reference), the generated WGSL ([`IntGraph::to_wgsl`]) and the generated
//! Rust ([`IntGraph::to_rust`]) compute the same bits.
//!
//! - Types: `i32`, `u32`, `bool`; `i8`/`u8` exist only as storage
//!   ([`Elem`]) and widen on load.
//! - Arithmetic wraps (`+ − ×`, negation); shift amounts are taken mod 32;
//!   `>>` is arithmetic on `i32`, logical on `u32`.
//! - [`IntGraph::mul_wide`]: exact 32×32→64 as `(lo, hi)`; WGSL (no 64-bit
//!   integers) gets 16-bit-limb helpers.
//! - Loops: [`IntGraph::while_loop`] is structured SSA (`scf.while`): state
//!   values, a condition and a next-state over [`INode::State`] nodes.
//!   Results are [`INode::LoopOut`] nodes. Loops nest; every node belongs to
//!   the innermost loop whose state it depends on, so invariant code is
//!   hoisted by construction.
//! - Buffers are read-only ([`IntGraph::load`]), except scratch buffers
//!   ([`IntGraph::buffer_rw`]): small fixed-length arrays read and written
//!   through ordered effects. An effect token ([`IntGraph::token`]) is
//!   threaded from store to store ([`IntGraph::store`] returns the next
//!   token); a scratch load ([`IntGraph::load_rw`]) depends on a token. The
//!   tokens only order the effects through data dependencies: every backend
//!   executes scratch loads and stores in the one shared schedule order
//!   (never inlined, never moved), so effects are deterministic and
//!   identical across backends whatever the graph. A builder that wants
//!   "load sees the state as of its token" makes every load of a token feed
//!   the store that consumes it. Everything else is pure. Results leave
//!   through the caller's bindings of the outputs.

mod emit;
mod eval;
mod sched;
mod wire;

pub use emit::WGSL_PRELUDE;
pub use eval::Program;
pub use wire::{Limits, WireError};

use std::collections::HashMap;

/// Value type of a node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IType {
    /// Signed 32-bit, two's complement, wrapping.
    I32,
    /// Unsigned 32-bit, wrapping.
    U32,
    /// Boolean (comparison results, select conditions).
    Bool,
}

/// Storage element type of a read-only buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Elem {
    /// Sign-extended to `i32` on load.
    I8,
    /// Zero-extended to `u32` on load.
    U8,
    /// `i32`.
    I32,
    /// `u32`.
    U32,
}

impl Elem {
    /// The value type a load produces.
    pub fn ty(self) -> IType {
        match self {
            Elem::I8 | Elem::I32 => IType::I32,
            Elem::U8 | Elem::U32 => IType::U32,
        }
    }
}

/// Handle to a node of an [`IntGraph`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct IId(pub(crate) u32);

impl IId {
    /// Raw index.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Binary integer ops (operands of one type, except shift amounts).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinOp {
    /// Wrapping add.
    Add,
    /// Wrapping subtract.
    Sub,
    /// Wrapping multiply (low 32 bits).
    Mul,
    /// High 32 bits of the exact 64-bit product (signed for `i32`).
    MulHi,
    /// Bitwise / logical and.
    And,
    /// Bitwise / logical or.
    Or,
    /// Bitwise / logical xor.
    Xor,
    /// Left shift by `b mod 32`.
    Shl,
    /// Right shift by `b mod 32` (arithmetic for `i32`).
    Shr,
    /// Minimum.
    Min,
    /// Maximum.
    Max,
    /// Division, WGSL semantics: `x / 0 = x`, `i32::MIN / −1 = i32::MIN`;
    /// signed division truncates.
    Div,
    /// Remainder, WGSL semantics: `x % 0 = 0`, `i32::MIN % −1 = 0`; the sign
    /// follows the dividend.
    Rem,
}

/// Comparisons (produce `bool`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CmpOp {
    /// `==`
    Eq,
    /// `!=`
    Ne,
    /// `<`
    Lt,
    /// `<=`
    Le,
}

/// Unary ops.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnOp {
    /// Wrapping negation.
    Neg,
    /// Bitwise not (logical not on `bool`).
    Not,
    /// Count of leading zero bits (32 for 0).
    Clz,
}

/// A node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum INode {
    /// Constant (bits; `bool` is 0/1).
    Const(IType, u32),
    /// Function input.
    Param(u16, IType),
    /// Binary op.
    Bin(BinOp, IId, IId),
    /// Unary op.
    Un(UnOp, IId),
    /// Comparison.
    Cmp(CmpOp, IId, IId),
    /// `if c { a } else { b }` (both sides evaluated).
    Select(IId, IId, IId),
    /// Reinterpret bits as another 32-bit type.
    Bitcast(IType, IId),
    /// `buf[index]` from a read-only buffer (index is `u32`).
    Load(u16, IId),
    /// State `k` of loop `l`, inside its body.
    State(u32, u16),
    /// Final state `k` of loop `l`, after it.
    LoopOut(u32, u16),
    /// The initial effect token (`u32`, value 0).
    Tok,
    /// `buf[index]` from a scratch buffer, as of effect token `tok`.
    LoadRw(u16, IId, IId),
    /// `buf[index] = value` on a scratch buffer after effect token `tok`;
    /// yields the next token.
    Store(u16, IId, IId, IId),
}

/// A structured while loop: `s = init; while cond(s) { s = next(s) }`.
#[derive(Clone, Debug)]
pub struct Loop {
    /// Initial states (in the enclosing scope).
    pub init: Vec<IId>,
    /// Condition, checked before each iteration.
    pub cond: IId,
    /// Next states.
    pub next: Vec<IId>,
    /// Enclosing loop, if nested.
    pub parent: Option<u32>,
    /// Trip cap: when set, the loop also exits after this many iterations
    /// (every backend counts the same way). Untrusted graphs always carry one.
    pub cap: Option<u32>,
}

/// A buffer declaration.
#[derive(Clone, Debug)]
pub struct Buffer {
    /// Name used by the lowerings.
    pub name: String,
    /// Element type.
    pub elem: Elem,
    /// Scratch buffer: `(length, base)`. Element `i < length` lives at
    /// `name[base + i]` in WGSL (`base` is a host expression); out-of-range
    /// loads read 0 and out-of-range stores are dropped.
    pub rw: Option<(u32, String)>,
}

/// Integer expression graph with structural interning.
#[derive(Clone, Default)]
pub struct IntGraph {
    nodes: Vec<INode>,
    types: Vec<IType>,
    /// Innermost loop each node depends on.
    scope: Vec<Option<u32>>,
    intern: HashMap<INode, IId>,
    loops: Vec<Loop>,
    bufs: Vec<Buffer>,
    building: Vec<u32>,
}

impl IntGraph {
    /// Empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the graph has no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Node of `id`.
    pub fn node(&self, id: IId) -> INode {
        self.nodes[id.index()]
    }

    /// Type of `id`.
    pub fn ty(&self, id: IId) -> IType {
        self.types[id.index()]
    }

    /// Loop `l`.
    pub fn loop_(&self, l: u32) -> &Loop {
        &self.loops[l as usize]
    }

    /// Declared buffers.
    pub fn buffers(&self) -> &[Buffer] {
        &self.bufs
    }

    fn depth(&self, mut l: Option<u32>) -> usize {
        let mut d = 0;
        while let Some(x) = l {
            d += 1;
            l = self.loops[x as usize].parent;
        }
        d
    }

    /// Innermost of two scopes, which must lie on one chain of nested loops.
    fn inner(&self, a: Option<u32>, b: Option<u32>) -> Option<u32> {
        self.try_inner(a, b)
            .expect("int graph: node mixes states of unrelated loops")
    }

    fn try_inner(&self, a: Option<u32>, b: Option<u32>) -> Result<Option<u32>, &'static str> {
        let (da, db) = (self.depth(a), self.depth(b));
        let (deep, shallow, mut d, ds) = if da >= db {
            (a, b, da, db)
        } else {
            (b, a, db, da)
        };
        let mut x = deep;
        while d > ds {
            x = self.loops[x.unwrap() as usize].parent;
            d -= 1;
        }
        if x != shallow {
            return Err("node mixes states of unrelated loops");
        }
        Ok(deep)
    }

    fn insert(&mut self, n: INode) -> IId {
        if let Some(&id) = self.intern.get(&n) {
            return id;
        }
        let (ty, scope) = self.infer(n);
        if let Some(folded) = self.fold(n, ty) {
            return folded;
        }
        let id = IId(self.nodes.len() as u32);
        self.nodes.push(n);
        self.types.push(ty);
        self.scope.push(scope);
        self.intern.insert(n, id);
        id
    }

    fn infer(&self, n: INode) -> (IType, Option<u32>) {
        self.try_infer(n)
            .unwrap_or_else(|e| panic!("int graph: {e}"))
    }

    fn try_infer(&self, n: INode) -> Result<(IType, Option<u32>), &'static str> {
        let sc = |a: IId| self.scope[a.index()];
        let ok = |c: bool, e: &'static str| if c { Ok(()) } else { Err(e) };
        Ok(match n {
            INode::Const(t, _) => (t, None),
            INode::Param(_, t) => (t, None),
            INode::Bin(op, a, b) => {
                let (ta, tb) = (self.ty(a), self.ty(b));
                match op {
                    BinOp::Shl | BinOp::Shr => {
                        ok(ta != IType::Bool && tb != IType::Bool, "shift of bool")?
                    }
                    BinOp::And | BinOp::Or | BinOp::Xor => {
                        ok(ta == tb, "bitwise op type mismatch")?
                    }
                    _ => {
                        ok(ta == tb, "arithmetic type mismatch")?;
                        ok(ta != IType::Bool, "arithmetic on bool")?;
                    }
                }
                (ta, self.try_inner(sc(a), sc(b))?)
            }
            INode::Un(op, a) => {
                ok(
                    op == UnOp::Not || self.ty(a) != IType::Bool,
                    "integer op on bool",
                )?;
                (self.ty(a), sc(a))
            }
            INode::Cmp(_, a, b) => {
                ok(self.ty(a) == self.ty(b), "compare type mismatch")?;
                (IType::Bool, self.try_inner(sc(a), sc(b))?)
            }
            INode::Select(c, a, b) => {
                ok(self.ty(c) == IType::Bool, "select condition must be bool")?;
                ok(self.ty(a) == self.ty(b), "select type mismatch")?;
                let s = self.try_inner(sc(a), sc(b))?;
                (self.ty(a), self.try_inner(s, sc(c))?)
            }
            INode::Bitcast(t, a) => {
                ok(
                    t != IType::Bool && self.ty(a) != IType::Bool,
                    "bitcast of bool",
                )?;
                (t, sc(a))
            }
            INode::Load(b, i) => {
                ok(self.ty(i) == IType::U32, "load index must be u32")?;
                let buf = self.bufs.get(b as usize).ok_or("buffer")?;
                ok(buf.rw.is_none(), "plain load from a scratch buffer")?;
                (buf.elem.ty(), sc(i))
            }
            INode::Tok => (IType::U32, None),
            INode::LoadRw(b, i, t) => {
                let buf = self.bufs.get(b as usize).ok_or("buffer")?;
                ok(buf.rw.is_some(), "scratch load from a read-only buffer")?;
                ok(
                    self.ty(i) == IType::U32 && self.ty(t) == IType::U32,
                    "scratch load index and token must be u32",
                )?;
                let s = self.try_inner(sc(i), sc(t))?;
                ok(s == sc(t), "scratch load outside its token's scope")?;
                (buf.elem.ty(), s)
            }
            INode::Store(b, i, v, t) => {
                let buf = self.bufs.get(b as usize).ok_or("buffer")?;
                ok(buf.rw.is_some(), "store to a read-only buffer")?;
                ok(
                    self.ty(i) == IType::U32 && self.ty(t) == IType::U32,
                    "store index and token must be u32",
                )?;
                ok(self.ty(v) == buf.elem.ty(), "store value type")?;
                let s = self.try_inner(self.try_inner(sc(i), sc(v))?, sc(t))?;
                ok(s == sc(t), "store outside its token's scope")?;
                (IType::U32, s)
            }
            INode::State(l, k) => (self.ty(self.loops[l as usize].init[k as usize]), Some(l)),
            INode::LoopOut(l, k) => (
                self.ty(self.loops[l as usize].init[k as usize]),
                self.loops[l as usize].parent,
            ),
        })
    }

    /// Fold nodes whose operands are all constants (same semantics as eval).
    fn fold(&mut self, n: INode, ty: IType) -> Option<IId> {
        let c = |a: IId| match self.nodes[a.index()] {
            INode::Const(_, v) => Some(v),
            _ => None,
        };
        let v = match n {
            INode::Bin(op, a, b) => eval::bin(op, self.ty(a), c(a)?, c(b)?),
            INode::Un(op, a) => eval::un(op, self.ty(a), c(a)?),
            INode::Cmp(op, a, b) => eval::cmp(op, self.ty(a), c(a)?, c(b)?),
            INode::Select(s, a, b) => {
                if c(s)? != 0 {
                    return Some(a);
                } else {
                    return Some(b);
                }
            }
            INode::Bitcast(_, a) => c(a)?,
            _ => return None,
        };
        Some(self.insert(INode::Const(ty, v)))
    }

    // ---- atoms ----

    /// `i32` constant.
    pub fn i32(&mut self, v: i32) -> IId {
        self.insert(INode::Const(IType::I32, v as u32))
    }

    /// `u32` constant.
    pub fn u32(&mut self, v: u32) -> IId {
        self.insert(INode::Const(IType::U32, v))
    }

    /// `bool` constant.
    pub fn bool(&mut self, v: bool) -> IId {
        self.insert(INode::Const(IType::Bool, v as u32))
    }

    /// Function input `i` of type `t`.
    pub fn param(&mut self, i: u16, t: IType) -> IId {
        self.insert(INode::Param(i, t))
    }

    /// Declare a read-only buffer; returns its index for [`load`](Self::load).
    pub fn buffer(&mut self, name: &str, elem: Elem) -> u16 {
        self.bufs.push(Buffer {
            name: name.into(),
            elem,
            rw: None,
        });
        (self.bufs.len() - 1) as u16
    }

    /// Declare a scratch buffer of `len` 32-bit elements (`elem` is `I32` or
    /// `U32`), at `name[base + i]` in WGSL.
    pub fn buffer_rw(&mut self, name: &str, elem: Elem, len: u32, base: &str) -> u16 {
        assert!(
            matches!(elem, Elem::I32 | Elem::U32),
            "scratch buffers hold 32-bit elements"
        );
        self.bufs.push(Buffer {
            name: name.into(),
            elem,
            rw: Some((len, base.into())),
        });
        (self.bufs.len() - 1) as u16
    }

    /// The initial effect token.
    pub fn token(&mut self) -> IId {
        self.insert(INode::Tok)
    }

    /// `buf[index]` of a scratch buffer as of effect token `tok`.
    pub fn load_rw(&mut self, buf: u16, index: IId, tok: IId) -> IId {
        self.insert(INode::LoadRw(buf, index, tok))
    }

    /// `buf[index] = value` after `tok`; returns the next token.
    pub fn store(&mut self, buf: u16, index: IId, value: IId, tok: IId) -> IId {
        self.insert(INode::Store(buf, index, value, tok))
    }

    /// `buf[index]`, widened per [`Elem`].
    pub fn load(&mut self, buf: u16, index: IId) -> IId {
        self.insert(INode::Load(buf, index))
    }

    // ---- ops ----

    /// Binary op.
    pub fn bin(&mut self, op: BinOp, a: IId, b: IId) -> IId {
        self.insert(INode::Bin(op, a, b))
    }

    /// Wrapping `a + b`.
    pub fn add(&mut self, a: IId, b: IId) -> IId {
        self.bin(BinOp::Add, a, b)
    }

    /// Wrapping `a − b`.
    pub fn sub(&mut self, a: IId, b: IId) -> IId {
        self.bin(BinOp::Sub, a, b)
    }

    /// Wrapping `a × b`.
    pub fn mul(&mut self, a: IId, b: IId) -> IId {
        self.bin(BinOp::Mul, a, b)
    }

    /// Exact 64-bit product as `(lo, hi)`.
    pub fn mul_wide(&mut self, a: IId, b: IId) -> (IId, IId) {
        (self.bin(BinOp::Mul, a, b), self.bin(BinOp::MulHi, a, b))
    }

    /// `a & b`.
    pub fn and(&mut self, a: IId, b: IId) -> IId {
        self.bin(BinOp::And, a, b)
    }

    /// `a | b`.
    pub fn or(&mut self, a: IId, b: IId) -> IId {
        self.bin(BinOp::Or, a, b)
    }

    /// `a ^ b`.
    pub fn xor(&mut self, a: IId, b: IId) -> IId {
        self.bin(BinOp::Xor, a, b)
    }

    /// `a << (b mod 32)`.
    pub fn shl(&mut self, a: IId, b: IId) -> IId {
        self.bin(BinOp::Shl, a, b)
    }

    /// `a >> (b mod 32)`, arithmetic for `i32`.
    pub fn shr(&mut self, a: IId, b: IId) -> IId {
        self.bin(BinOp::Shr, a, b)
    }

    /// `min(a, b)`.
    pub fn min(&mut self, a: IId, b: IId) -> IId {
        self.bin(BinOp::Min, a, b)
    }

    /// `max(a, b)`.
    pub fn max(&mut self, a: IId, b: IId) -> IId {
        self.bin(BinOp::Max, a, b)
    }

    /// `min(max(x, lo), hi)`.
    pub fn clamp(&mut self, x: IId, lo: IId, hi: IId) -> IId {
        let m = self.max(x, lo);
        self.min(m, hi)
    }

    /// Wrapping `−a`.
    pub fn neg(&mut self, a: IId) -> IId {
        self.insert(INode::Un(UnOp::Neg, a))
    }

    /// `!a` (bitwise on integers).
    pub fn not(&mut self, a: IId) -> IId {
        self.insert(INode::Un(UnOp::Not, a))
    }

    /// Leading zero bits of `a` (`firstLeadingBit(a) = 31 − clz(a)` for `a > 0`).
    pub fn clz(&mut self, a: IId) -> IId {
        self.insert(INode::Un(UnOp::Clz, a))
    }

    /// Comparison.
    pub fn cmp(&mut self, op: CmpOp, a: IId, b: IId) -> IId {
        self.insert(INode::Cmp(op, a, b))
    }

    /// `a == b`.
    pub fn eq(&mut self, a: IId, b: IId) -> IId {
        self.cmp(CmpOp::Eq, a, b)
    }

    /// `a < b`.
    pub fn lt(&mut self, a: IId, b: IId) -> IId {
        self.cmp(CmpOp::Lt, a, b)
    }

    /// `a <= b`.
    pub fn le(&mut self, a: IId, b: IId) -> IId {
        self.cmp(CmpOp::Le, a, b)
    }

    /// `if c { a } else { b }`.
    pub fn select(&mut self, c: IId, a: IId, b: IId) -> IId {
        self.insert(INode::Select(c, a, b))
    }

    /// Reinterpret as `t`.
    pub fn bitcast(&mut self, t: IType, a: IId) -> IId {
        if self.ty(a) == t {
            return a;
        }
        self.insert(INode::Bitcast(t, a))
    }

    // ---- loops ----

    /// `s = init; while cond(s) { s = next(s) }`; returns the final states.
    ///
    /// `body` gets the state nodes and returns `(cond, next)`.
    pub fn while_loop(
        &mut self,
        init: &[IId],
        body: impl FnOnce(&mut Self, &[IId]) -> (IId, Vec<IId>),
    ) -> Vec<IId> {
        let l = self.loops.len() as u32;
        let parent = self.building.last().copied();
        for &x in init {
            // inits live in the enclosing scope
            self.inner(self.scope[x.index()], parent);
        }
        self.loops.push(Loop {
            init: init.to_vec(),
            cond: IId(0),
            next: Vec::new(),
            parent,
            cap: None,
        });
        self.building.push(l);
        let st: Vec<IId> = (0..init.len())
            .map(|k| self.insert(INode::State(l, k as u16)))
            .collect();
        let (cond, next) = body(self, &st);
        self.building.pop();
        assert_eq!(self.ty(cond), IType::Bool, "loop condition must be bool");
        assert_eq!(next.len(), init.len(), "loop next-state arity");
        for (k, &n) in next.iter().enumerate() {
            assert_eq!(self.ty(n), self.ty(init[k]), "loop state {k} changes type");
        }
        self.loops[l as usize].cond = cond;
        self.loops[l as usize].next = next;
        (0..init.len())
            .map(|k| self.insert(INode::LoopOut(l, k as u16)))
            .collect()
    }

    /// Cap the trip count of the loop that produced `out` (a loop result).
    pub fn cap_loop(&mut self, out: IId, cap: u32) {
        match self.node(out) {
            INode::LoopOut(l, _) => self.loops[l as usize].cap = Some(cap),
            _ => panic!("cap_loop: not a loop result"),
        }
    }

    /// Unsigned / signed division (WGSL semantics, see [`BinOp::Div`]).
    pub fn div(&mut self, a: IId, b: IId) -> IId {
        self.bin(BinOp::Div, a, b)
    }

    /// Remainder (WGSL semantics, see [`BinOp::Rem`]).
    pub fn rem(&mut self, a: IId, b: IId) -> IId {
        self.bin(BinOp::Rem, a, b)
    }

    /// `for i in 0..n` (`n: u32`) over states; `body(g, i, s)` returns next states.
    pub fn for_range(
        &mut self,
        n: IId,
        init: &[IId],
        body: impl FnOnce(&mut Self, IId, &[IId]) -> Vec<IId>,
    ) -> Vec<IId> {
        let zero = self.u32(0);
        let mut s0 = vec![zero];
        s0.extend_from_slice(init);
        let out = self.while_loop(&s0, |g, s| {
            let c = g.lt(s[0], n);
            let one = g.u32(1);
            let mut next = vec![g.add(s[0], one)];
            next.extend(body(g, s[0], &s[1..]));
            (c, next)
        });
        out[1..].to_vec()
    }
}

#[cfg(test)]
mod tests;
