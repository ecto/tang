//! Wire format and verifier for untrusted integer graphs.
//!
//! A graph that arrives from outside (e.g. a decoder shipped inside a
//! compressed stream) is read with [`IntGraph::from_bytes`], which rebuilds
//! it node by node and rejects anything the trusted builder could not have
//! produced, plus anything over the caller's [`Limits`]:
//!
//! - every operand refers to an earlier node (the graph is a DAG in id order);
//! - types and loop scopes check exactly as in the builder;
//! - parameters and buffers match the caller's ABI (count and type); buffer
//!   names come from the caller, never from the bytes;
//! - every loop carries a trip cap (`≤ max_trip`), nests at most `max_depth`
//!   deep, and has exactly one state node per state;
//! - the worst-case work, Σ over nodes of the product of the caps of the
//!   enclosing loops, is at most `max_cost`;
//! - outputs are top-level and have the caller's types.
//!
//! The op set has no stores, no I/O and no unbounded control flow; loads are
//! read-only and bounds-checked by the interpreter and by
//! [`IntGraph::to_wgsl_body_guarded`]. So a verified graph is a total
//! function of its parameters and buffers.
//!
//! Layout (all integers LEB128 varints unless noted):
//! `"TIR1"`, `nloops`, per loop `parent+1 | 0`, `cap`, `n`, `init[n]`,
//! `cond`, `next[n]`; `nnodes`, per node a tag byte and its fields (operands
//! as `self − operand ≥ 1`); `nout`, `out[nout]`. Loop fields are node ids.

use std::collections::HashMap;

use super::{BinOp, CmpOp, Elem, IId, INode, IType, IntGraph, Loop, UnOp};

const MAGIC: &[u8; 4] = b"TIR1";

/// Resource limits for [`IntGraph::from_bytes`].
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// Maximum node count.
    pub max_nodes: usize,
    /// Maximum loop count.
    pub max_loops: usize,
    /// Maximum loop nesting depth.
    pub max_depth: usize,
    /// Maximum trip cap of one loop.
    pub max_trip: u32,
    /// Maximum Σ nodes × enclosing trip caps (work per evaluation).
    pub max_cost: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self { max_nodes: 1 << 14, max_loops: 64, max_depth: 4, max_trip: 1 << 12, max_cost: 1 << 20 }
    }
}

/// Why a graph was rejected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WireError(pub String);

impl std::fmt::Display for WireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "int graph rejected: {}", self.0)
    }
}

impl std::error::Error for WireError {}

fn err<T>(m: impl Into<String>) -> Result<T, WireError> {
    Err(WireError(m.into()))
}

fn put(o: &mut Vec<u8>, mut v: u64) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            o.push(b);
            return;
        }
        o.push(b | 0x80);
    }
}

struct Rd<'a> {
    b: &'a [u8],
    p: usize,
}

impl Rd<'_> {
    fn u8(&mut self) -> Result<u8, WireError> {
        let v = *self.b.get(self.p).ok_or(WireError("truncated".into()))?;
        self.p += 1;
        Ok(v)
    }
    fn var(&mut self) -> Result<u64, WireError> {
        let mut v = 0u64;
        for s in 0..10 {
            let b = self.u8()?;
            v |= ((b & 0x7f) as u64) << (7 * s);
            if b & 0x80 == 0 {
                return Ok(v);
            }
        }
        err("varint too long")
    }
    fn u32(&mut self) -> Result<u32, WireError> {
        u32::try_from(self.var()?).or(err("value out of range"))
    }
    fn n(&mut self, max: usize, what: &str) -> Result<usize, WireError> {
        let v = self.var()?;
        if v > max as u64 {
            return err(format!("too many {what}"));
        }
        Ok(v as usize)
    }
}

fn ty_code(t: IType) -> u8 {
    match t {
        IType::I32 => 0,
        IType::U32 => 1,
        IType::Bool => 2,
    }
}

fn ty_of(c: u8) -> Result<IType, WireError> {
    Ok(match c {
        0 => IType::I32,
        1 => IType::U32,
        2 => IType::Bool,
        _ => return err("type"),
    })
}

const BINS: [BinOp; 13] = [
    BinOp::Add,
    BinOp::Sub,
    BinOp::Mul,
    BinOp::MulHi,
    BinOp::And,
    BinOp::Or,
    BinOp::Xor,
    BinOp::Shl,
    BinOp::Shr,
    BinOp::Min,
    BinOp::Max,
    BinOp::Div,
    BinOp::Rem,
];
const CMPS: [CmpOp; 4] = [CmpOp::Eq, CmpOp::Ne, CmpOp::Lt, CmpOp::Le];
const UNS: [UnOp; 2] = [UnOp::Neg, UnOp::Not];

impl IntGraph {
    /// Serialize the whole graph with `outputs` (see the module docs).
    /// Buffers are not named on the wire; the reader supplies the ABI.
    pub fn to_bytes(&self, outputs: &[IId]) -> Vec<u8> {
        let mut o = MAGIC.to_vec();
        put(&mut o, self.loops.len() as u64);
        for lp in &self.loops {
            put(&mut o, lp.parent.map_or(0, |p| p as u64 + 1));
            put(&mut o, lp.cap.expect("to_bytes: every loop needs a trip cap") as u64);
            put(&mut o, lp.init.len() as u64);
            for &x in &lp.init {
                put(&mut o, x.0 as u64);
            }
            put(&mut o, lp.cond.0 as u64);
            for &x in &lp.next {
                put(&mut o, x.0 as u64);
            }
        }
        put(&mut o, self.nodes.len() as u64);
        for (i, n) in self.nodes.iter().enumerate() {
            let d = |a: IId| (i - a.index()) as u64;
            match *n {
                INode::Const(t, v) => {
                    o.push(0);
                    o.push(ty_code(t));
                    // small negative i32s as zigzag
                    let z = if t == IType::I32 { ((v << 1) ^ (((v as i32) >> 31) as u32)) as u64 } else { v as u64 };
                    put(&mut o, z);
                }
                INode::Param(p, t) => {
                    o.push(1);
                    put(&mut o, p as u64);
                    o.push(ty_code(t));
                }
                INode::Bin(op, a, b) => {
                    o.push(2);
                    o.push(BINS.iter().position(|&x| x == op).unwrap() as u8);
                    put(&mut o, d(a));
                    put(&mut o, d(b));
                }
                INode::Un(op, a) => {
                    o.push(3);
                    o.push(UNS.iter().position(|&x| x == op).unwrap() as u8);
                    put(&mut o, d(a));
                }
                INode::Cmp(op, a, b) => {
                    o.push(4);
                    o.push(CMPS.iter().position(|&x| x == op).unwrap() as u8);
                    put(&mut o, d(a));
                    put(&mut o, d(b));
                }
                INode::Select(c, a, b) => {
                    o.push(5);
                    put(&mut o, d(c));
                    put(&mut o, d(a));
                    put(&mut o, d(b));
                }
                INode::Bitcast(t, a) => {
                    o.push(6);
                    o.push(ty_code(t));
                    put(&mut o, d(a));
                }
                INode::Load(b, a) => {
                    o.push(7);
                    put(&mut o, b as u64);
                    put(&mut o, d(a));
                }
                INode::State(l, k) => {
                    o.push(8);
                    put(&mut o, l as u64);
                    put(&mut o, k as u64);
                }
                INode::LoopOut(l, k) => {
                    o.push(9);
                    put(&mut o, l as u64);
                    put(&mut o, k as u64);
                }
            }
        }
        put(&mut o, outputs.len() as u64);
        for x in outputs {
            put(&mut o, x.0 as u64);
        }
        o
    }

    /// Read and verify an untrusted graph (see the module docs). `params`
    /// gives the type of each parameter, `bufs` the name and element type of
    /// each buffer, `outs` the output types. Returns the graph and its outputs.
    pub fn from_bytes(bytes: &[u8], params: &[IType], bufs: &[(&str, Elem)], outs: &[IType], lim: &Limits) -> Result<(IntGraph, Vec<IId>), WireError> {
        let mut r = Rd { b: bytes, p: 0 };
        if bytes.len() < 4 || &bytes[..4] != MAGIC {
            return err("magic");
        }
        r.p = 4;
        let mut g = IntGraph::new();
        for &(name, elem) in bufs {
            g.buffer(name, elem);
        }
        let nl = r.n(lim.max_loops, "loops")?;
        let mut caps = Vec::with_capacity(nl);
        for l in 0..nl {
            let parent = match r.var()? {
                0 => None,
                p if p - 1 < l as u64 => Some((p - 1) as u32),
                _ => return err("loop parent must be an earlier loop"),
            };
            let cap = r.u32()?;
            if cap > lim.max_trip {
                return err("loop trip cap over limit");
            }
            caps.push(cap);
            let k = r.n(256, "loop states")?;
            let ids = |r: &mut Rd, k: usize| (0..k).map(|_| r.u32().map(IId)).collect::<Result<Vec<_>, _>>();
            let init = ids(&mut r, k)?;
            let cond = IId(r.u32()?);
            let next = ids(&mut r, k)?;
            let lp = Loop { init, cond, next, parent, cap: Some(cap) };
            if g.depth(parent) + 1 > lim.max_depth {
                return err("loops nest too deep");
            }
            g.loops.push(lp);
        }
        let nn = r.n(lim.max_nodes, "nodes")?;
        let mut state_seen: Vec<Vec<bool>> = g.loops.iter().map(|l| vec![false; l.init.len()]).collect();
        let mut intern: HashMap<INode, IId> = HashMap::new();
        for i in 0..nn {
            let tag = r.u8()?;
            let opnd = |r: &mut Rd| -> Result<IId, WireError> {
                let d = r.var()?;
                if d == 0 || d > i as u64 {
                    return err("operand must be an earlier node");
                }
                Ok(IId((i as u64 - d) as u32))
            };
            let lk = |r: &mut Rd, g: &IntGraph| -> Result<(u32, u16), WireError> {
                let l = r.var()?;
                let k = r.var()?;
                if l >= g.loops.len() as u64 || k >= g.loops[l as usize].init.len() as u64 {
                    return err("loop reference");
                }
                Ok((l as u32, k as u16))
            };
            let n = match tag {
                0 => {
                    let t = ty_of(r.u8()?)?;
                    let z = r.u32()?;
                    let v = if t == IType::I32 { (z >> 1) ^ (z & 1).wrapping_neg() } else { z };
                    if t == IType::Bool && v > 1 {
                        return err("bool constant");
                    }
                    INode::Const(t, v)
                }
                1 => {
                    let p = r.var()?;
                    let t = ty_of(r.u8()?)?;
                    if p >= params.len() as u64 || params[p as usize] != t {
                        return err("parameter not in the ABI");
                    }
                    INode::Param(p as u16, t)
                }
                2 => {
                    let op = *BINS.get(r.u8()? as usize).ok_or(WireError("op".into()))?;
                    INode::Bin(op, opnd(&mut r)?, opnd(&mut r)?)
                }
                3 => {
                    let op = *UNS.get(r.u8()? as usize).ok_or(WireError("op".into()))?;
                    INode::Un(op, opnd(&mut r)?)
                }
                4 => {
                    let op = *CMPS.get(r.u8()? as usize).ok_or(WireError("op".into()))?;
                    INode::Cmp(op, opnd(&mut r)?, opnd(&mut r)?)
                }
                5 => INode::Select(opnd(&mut r)?, opnd(&mut r)?, opnd(&mut r)?),
                6 => {
                    let t = ty_of(r.u8()?)?;
                    INode::Bitcast(t, opnd(&mut r)?)
                }
                7 => {
                    let b = r.var()?;
                    if b >= bufs.len() as u64 {
                        return err("buffer not in the ABI");
                    }
                    INode::Load(b as u16, opnd(&mut r)?)
                }
                8 => {
                    let (l, k) = lk(&mut r, &g)?;
                    if g.loops[l as usize].init[k as usize].index() >= i {
                        return err("loop init must precede its state");
                    }
                    if std::mem::replace(&mut state_seen[l as usize][k as usize], true) {
                        return err("duplicate loop state");
                    }
                    INode::State(l, k)
                }
                9 => {
                    let (l, k) = lk(&mut r, &g)?;
                    let lp = &g.loops[l as usize];
                    if lp.cond.index() >= i || lp.next.iter().any(|x| x.index() >= i) {
                        return err("loop body must precede its result");
                    }
                    if !state_seen[l as usize].iter().all(|&s| s) {
                        return err("loop result before all its states");
                    }
                    INode::LoopOut(l, k)
                }
                _ => return err("node tag"),
            };
            let (ty, scope) = g.try_infer(n).map_err(|e| WireError(e.into()))?;
            g.nodes.push(n);
            g.types.push(ty);
            g.scope.push(scope);
            intern.entry(n).or_insert(IId(i as u32));
        }
        g.intern = intern;
        let n = g.nodes.len();
        // loops: closed, typed, scoped
        for l in 0..g.loops.len() {
            let lp = g.loops[l].clone();
            if !state_seen[l].iter().all(|&s| s) {
                return err("loop without all its states");
            }
            if lp.cond.index() >= n || lp.next.iter().chain(&lp.init).any(|x| x.index() >= n) {
                return err("loop reference out of range");
            }
            if g.ty(lp.cond) != IType::Bool {
                return err("loop condition must be bool");
            }
            let me = Some(l as u32);
            let within = |g: &IntGraph, x: IId, s: Option<u32>| g.try_inner(g.scope[x.index()], s).ok() == Some(s);
            if !within(&g, lp.cond, me) || !lp.next.iter().all(|&x| within(&g, x, me)) {
                return err("loop body uses an unrelated scope");
            }
            if !lp.init.iter().all(|&x| within(&g, x, lp.parent)) {
                return err("loop init outside the enclosing scope");
            }
            for (k, &x) in lp.next.iter().enumerate() {
                if g.ty(x) != g.ty(lp.init[k]) {
                    return err("loop state changes type");
                }
            }
        }
        // worst-case work
        let mut cost = 0u64;
        for i in 0..n {
            let mut c = 1u64;
            let mut s = g.scope[i];
            while let Some(l) = s {
                c = c.saturating_mul(caps[l as usize].max(1) as u64);
                s = g.loops[l as usize].parent;
            }
            cost = cost.saturating_add(c);
        }
        if cost > lim.max_cost {
            return err(format!("worst-case cost {cost} over limit"));
        }
        let no = r.n(outs.len(), "outputs")?;
        if no != outs.len() {
            return err("output count");
        }
        let mut o = Vec::with_capacity(no);
        for &t in outs {
            let x = r.u32()? as usize;
            if x >= n || g.scope[x].is_some() || g.types[x] != t {
                return err("output not a top-level value of the ABI type");
            }
            o.push(IId(x as u32));
        }
        if r.p != bytes.len() {
            return err("trailing bytes");
        }
        Ok((g, o))
    }
}
