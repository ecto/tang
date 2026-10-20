//! CPU reference: the definition of every op's bits, and a register
//! bytecode interpreter for whole graphs.

use super::sched::Item;
use super::{BinOp, CmpOp, Elem, IId, INode, IType, IntGraph, UnOp};

pub(crate) fn bin(op: BinOp, t: IType, a: u32, b: u32) -> u32 {
    let s = t == IType::I32;
    match op {
        BinOp::Add => a.wrapping_add(b),
        BinOp::Sub => a.wrapping_sub(b),
        BinOp::Mul => a.wrapping_mul(b),
        BinOp::MulHi if s => ((a as i32 as i64 * b as i32 as i64) >> 32) as u32,
        BinOp::MulHi => ((a as u64 * b as u64) >> 32) as u32,
        BinOp::And => a & b,
        BinOp::Or => a | b,
        BinOp::Xor => a ^ b,
        BinOp::Shl => a.wrapping_shl(b),
        BinOp::Shr if s => (a as i32).wrapping_shr(b) as u32,
        BinOp::Shr => a.wrapping_shr(b),
        BinOp::Min if s => (a as i32).min(b as i32) as u32,
        BinOp::Min => a.min(b),
        BinOp::Max if s => (a as i32).max(b as i32) as u32,
        BinOp::Max => a.max(b),
        BinOp::Div if b == 0 => a,
        BinOp::Div if s => (a as i32).wrapping_div(b as i32) as u32,
        BinOp::Div => a / b,
        BinOp::Rem if b == 0 => 0,
        BinOp::Rem if s => (a as i32).wrapping_rem(b as i32) as u32,
        BinOp::Rem => a % b,
    }
}

pub(crate) fn un(op: UnOp, t: IType, a: u32) -> u32 {
    match (op, t) {
        (UnOp::Neg, _) => a.wrapping_neg(),
        (UnOp::Not, IType::Bool) => a ^ 1,
        (UnOp::Not, _) => !a,
        (UnOp::Clz, _) => a.leading_zeros(),
    }
}

pub(crate) fn cmp(op: CmpOp, t: IType, a: u32, b: u32) -> u32 {
    let o = if t == IType::I32 { (a as i32).cmp(&(b as i32)) } else { a.cmp(&b) };
    (match op {
        CmpOp::Eq => o.is_eq(),
        CmpOp::Ne => o.is_ne(),
        CmpOp::Lt => o.is_lt(),
        CmpOp::Le => o.is_le(),
    }) as u32
}

pub(crate) fn load(elem: Elem, buf: &[u32], i: u32) -> u32 {
    // out of bounds reads 0 (the guarded WGSL lowering does the same)
    let i = i as usize;
    let w = |j: usize| buf.get(j).copied().unwrap_or(0);
    match elem {
        Elem::I32 | Elem::U32 => w(i),
        Elem::U8 => (w(i >> 2) >> ((i & 3) * 8)) & 0xff,
        Elem::I8 => ((w(i >> 2) >> ((i & 3) * 8)) as u8 as i8) as i32 as u32,
    }
}

#[derive(Clone, Copy)]
enum Op {
    Bin(BinOp, IType, u32, u32, u32),
    Un(UnOp, IType, u32, u32),
    Cmp(CmpOp, IType, u32, u32, u32),
    Sel(u32, u32, u32, u32),
    Mov(u32, u32),
    Load(Elem, u16, u32, u32),
    /// Scratch load: `(rw buffer, len, dst, index)`.
    LoadRw(u16, u32, u32, u32),
    /// Scratch store: `(rw buffer, len, index, value, token dst)`.
    Store(u16, u32, u32, u32, u32),
    JmpIfZero(u32, u32),
    Jmp(u32),
}

/// A graph compiled to register bytecode: the CPU reference evaluator.
///
/// Buffers are `u32` words; `i8`/`u8` buffers are packed 4 per word, little
/// endian (the WGSL layout).
#[derive(Clone)]
pub struct Program {
    ops: Vec<Op>,
    regs: Vec<u32>,
    params: Vec<(u16, u32)>,
    outs: Vec<u32>,
}

impl Program {
    /// Compile `outputs` of `g`.
    pub fn new(g: &IntGraph, outputs: &[IId]) -> Self {
        let sched = g.schedule(outputs);
        let n = g.nodes.len();
        let mut regs = vec![0u32; n];
        let mut params = Vec::new();
        for (i, node) in g.nodes.iter().enumerate() {
            match *node {
                INode::Const(_, v) => regs[i] = v,
                INode::Param(p, _) => params.push((p, i as u32)),
                _ => {}
            }
        }
        // state register of (loop, k)
        let mut state = vec![Vec::new(); g.loops.len()];
        for (i, node) in g.nodes.iter().enumerate() {
            if let INode::State(l, k) = *node {
                let s = &mut state[l as usize];
                if s.len() <= k as usize {
                    s.resize(k as usize + 1, u32::MAX);
                }
                s[k as usize] = i as u32;
            }
        }
        // read-only and scratch buffers are numbered separately
        let (mut nro, mut nrw) = (0u16, 0u16);
        let bidx = g
            .bufs
            .iter()
            .map(|b| {
                let c = if b.rw.is_some() { &mut nrw } else { &mut nro };
                *c += 1;
                *c - 1
            })
            .collect();
        let mut c = Compiler { g, ops: Vec::new(), state, nregs: n as u32 + 1, consts: vec![(n as u32, 1)], one: n as u32, bidx };
        c.items(&sched.items);
        regs.resize(c.nregs as usize, 0);
        for &(r, v) in &c.consts {
            regs[r as usize] = v;
        }
        Self { ops: c.ops, regs, params, outs: outputs.iter().map(|o| o.0).collect() }
    }

    /// Run with `params` (bits, by param index) and buffers; writes the
    /// outputs' bits into `out`.
    pub fn run(&mut self, params: &[u32], bufs: &[&[u32]], out: &mut [u32]) {
        self.run_rw(params, bufs, &mut [], out)
    }

    /// [`Self::run`] for a graph with scratch buffers: `bufs` are the
    /// read-only buffers and `rw` the scratch ones, each in declaration order.
    pub fn run_rw(&mut self, params: &[u32], bufs: &[&[u32]], rw: &mut [&mut [u32]], out: &mut [u32]) {
        let r = &mut self.regs;
        for &(p, i) in &self.params {
            r[i as usize] = params[p as usize];
        }
        let mut pc = 0usize;
        while pc < self.ops.len() {
            match self.ops[pc] {
                Op::Bin(op, t, d, a, b) => r[d as usize] = bin(op, t, r[a as usize], r[b as usize]),
                Op::Un(op, t, d, a) => r[d as usize] = un(op, t, r[a as usize]),
                Op::Cmp(op, t, d, a, b) => r[d as usize] = cmp(op, t, r[a as usize], r[b as usize]),
                Op::Sel(d, c, a, b) => r[d as usize] = if r[c as usize] != 0 { r[a as usize] } else { r[b as usize] },
                Op::Mov(d, a) => r[d as usize] = r[a as usize],
                Op::Load(e, b, d, i) => r[d as usize] = load(e, bufs[b as usize], r[i as usize]),
                Op::LoadRw(b, n, d, i) => {
                    let i = r[i as usize];
                    r[d as usize] = if i < n { rw[b as usize].get(i as usize).copied().unwrap_or(0) } else { 0 };
                }
                Op::Store(b, n, i, v, d) => {
                    let i = r[i as usize];
                    if i < n {
                        if let Some(w) = rw[b as usize].get_mut(i as usize) {
                            *w = r[v as usize];
                        }
                    }
                    r[d as usize] = 0;
                }
                Op::JmpIfZero(c, t) => {
                    if r[c as usize] == 0 {
                        pc = t as usize;
                        continue;
                    }
                }
                Op::Jmp(t) => {
                    pc = t as usize;
                    continue;
                }
            }
            pc += 1;
        }
        for (o, &i) in out.iter_mut().zip(&self.outs) {
            *o = r[i as usize];
        }
    }
}

struct Compiler<'a> {
    g: &'a IntGraph,
    ops: Vec<Op>,
    state: Vec<Vec<u32>>,
    nregs: u32,
    /// Registers preset to constants (loop caps).
    consts: Vec<(u32, u32)>,
    /// Register holding 1u.
    one: u32,
    /// Index of each buffer among the read-only or the scratch buffers.
    bidx: Vec<u16>,
}

impl Compiler<'_> {
    fn items(&mut self, items: &[Item]) {
        for it in items {
            match it {
                Item::Node(id) => self.node(*id),
                Item::Loop { l, pre, post } => self.lp(*l, pre, post),
            }
        }
    }

    fn node(&mut self, id: IId) {
        let g = self.g;
        let d = id.0;
        let op = match g.node(id) {
            INode::Const(..) | INode::Param(..) | INode::State(..) | INode::Tok => return,
            INode::Bin(op, a, b) => Op::Bin(op, g.ty(a), d, a.0, b.0),
            INode::Un(op, a) => Op::Un(op, g.ty(a), d, a.0),
            INode::Cmp(op, a, b) => Op::Cmp(op, g.ty(a), d, a.0, b.0),
            INode::Select(c, a, b) => Op::Sel(d, c.0, a.0, b.0),
            INode::Bitcast(_, a) => Op::Mov(d, a.0),
            INode::Load(b, i) => Op::Load(g.bufs[b as usize].elem, self.bidx[b as usize], d, i.0),
            INode::LoadRw(b, i, _) => Op::LoadRw(self.bidx[b as usize], g.bufs[b as usize].rw.as_ref().unwrap().0, d, i.0),
            INode::Store(b, i, v, _) => Op::Store(self.bidx[b as usize], g.bufs[b as usize].rw.as_ref().unwrap().0, i.0, v.0, d),
            INode::LoopOut(l, k) => Op::Mov(d, self.state[l as usize][k as usize]),
        };
        self.ops.push(op);
    }

    fn lp(&mut self, l: u32, pre: &[Item], post: &[Item]) {
        let lp = &self.g.loops[l as usize];
        let st = self.state[l as usize].clone();
        for (k, &i) in lp.init.iter().enumerate() {
            self.ops.push(Op::Mov(st[k], i.0));
        }
        // trip cap: a counter register, a constant register and a flag
        let capr = lp.cap.map(|c| {
            let (ctr, lim, ok) = (self.nregs, self.nregs + 1, self.nregs + 2);
            self.nregs += 3;
            self.consts.push((lim, c));
            self.ops.push(Op::Bin(BinOp::Xor, IType::U32, ctr, ctr, ctr));
            (ctr, lim, ok)
        });
        let top = self.ops.len() as u32;
        let jcap = capr.map(|(ctr, lim, ok)| {
            self.ops.push(Op::Cmp(CmpOp::Lt, IType::U32, ok, ctr, lim));
            let j = self.ops.len();
            self.ops.push(Op::JmpIfZero(ok, 0));
            self.ops.push(Op::Bin(BinOp::Add, IType::U32, ctr, ctr, self.one));
            j
        });
        self.items(pre);
        let jz = self.ops.len();
        self.ops.push(Op::JmpIfZero(lp.cond.0, 0));
        self.items(post);
        // parallel state update through temporaries
        let tmp = self.nregs;
        self.nregs += lp.next.len() as u32;
        for (k, &x) in lp.next.iter().enumerate() {
            self.ops.push(Op::Mov(tmp + k as u32, x.0));
        }
        for (k, &r) in st.iter().enumerate().take(lp.next.len()) {
            self.ops.push(Op::Mov(r, tmp + k as u32));
        }
        self.ops.push(Op::Jmp(top));
        let end = self.ops.len() as u32;
        self.ops[jz] = Op::JmpIfZero(lp.cond.0, end);
        if let (Some(j), Some((_, _, ok))) = (jcap, capr) {
            self.ops[j] = Op::JmpIfZero(ok, end);
        }
    }
}

impl IntGraph {
    /// Evaluate `outputs` once (compiles a [`Program`]; reuse one for speed).
    pub fn eval(&self, outputs: &[IId], params: &[u32], bufs: &[&[u32]]) -> Vec<u32> {
        let mut p = Program::new(self, outputs);
        let mut out = vec![0; outputs.len()];
        p.run(params, bufs, &mut out);
        out
    }
}
