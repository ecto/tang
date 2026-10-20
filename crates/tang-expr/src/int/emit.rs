//! Source lowerings: WGSL and Rust, from one schedule.
//!
//! Single-use nodes are inlined into their user (same scope only, so loop
//! invariants stay hoisted); everything else is a `let`. Loop states are
//! `var`s, snapshotted at the top of each iteration so the state update is
//! parallel.

use std::fmt::Write;

use super::sched::{Item, Sched};
use super::{BinOp, CmpOp, Elem, IId, INode, IType, IntGraph, UnOp};

/// Helpers the WGSL lowering calls (exact high words of 32×32 products in
/// 16-bit limbs). Include once per module.
pub const WGSL_PRELUDE: &str = "\
fn tx_mulhi_u32(a: u32, b: u32) -> u32 {
    let al = a & 0xffffu; let ah = a >> 16u;
    let bl = b & 0xffffu; let bh = b >> 16u;
    let lh = al * bh; let hl = ah * bl;
    let mid = ((al * bl) >> 16u) + (lh & 0xffffu) + (hl & 0xffffu);
    return ah * bh + (lh >> 16u) + (hl >> 16u) + (mid >> 16u);
}
fn tx_mulhi_i32(a: i32, b: i32) -> i32 {
    let ua = bitcast<u32>(a); let ub = bitcast<u32>(b);
    return bitcast<i32>(tx_mulhi_u32(ua, ub) - select(0u, ub, a < 0i) - select(0u, ua, b < 0i));
}
";

#[derive(Clone, Copy, PartialEq)]
enum Lang {
    Wgsl,
    Rust,
}

struct Em<'a> {
    g: &'a IntGraph,
    lang: Lang,
    sched: Sched,
    params: Vec<String>,
    out: String,
}

fn ty_name(lang: Lang, t: IType) -> &'static str {
    match (lang, t) {
        (_, IType::I32) => "i32",
        (_, IType::U32) => "u32",
        (_, IType::Bool) => "bool",
    }
}

impl Em<'_> {
    fn lit(&self, t: IType, v: u32) -> String {
        match (self.lang, t) {
            (_, IType::Bool) => (v != 0).to_string(),
            (Lang::Wgsl, IType::I32) if v as i32 == i32::MIN => "i32(-2147483648)".into(),
            (Lang::Rust, IType::I32) if v as i32 == i32::MIN => "i32::MIN".into(),
            (Lang::Wgsl, IType::I32) if (v as i32) < 0 => format!("({}i)", v as i32),
            (Lang::Wgsl, IType::I32) => format!("{}i", v as i32),
            (Lang::Rust, IType::I32) if (v as i32) < 0 => format!("({}i32)", v as i32),
            (Lang::Rust, IType::I32) => format!("{}i32", v as i32),
            (Lang::Wgsl, IType::U32) => format!("{v}u"),
            (Lang::Rust, IType::U32) => format!("{v}u32"),
        }
    }

    /// Reference to a node's value (atom, name, or inlined expression).
    fn r(&self, id: IId) -> String {
        match self.g.node(id) {
            INode::Const(t, v) => self.lit(t, v),
            INode::Param(p, _) => format!("({})", self.params[p as usize]),
            INode::State(l, k) => format!("st{l}_{k}"),
            INode::LoopOut(l, k) => format!("s{l}_{k}"),
            _ if self.sched.inline[id.index()] => self.def(id),
            _ => format!("t{}", id.0),
        }
    }

    fn def(&self, id: IId) -> String {
        let g = self.g;
        let w = self.lang == Lang::Wgsl;
        match g.node(id) {
            INode::Bin(op, a, b) => {
                let t = g.ty(a);
                let (x, y) = (self.r(a), self.r(b));
                let tn = ty_name(self.lang, t);
                if w {
                    let amt = |y: String| if g.ty(b) == IType::U32 { y } else { format!("bitcast<u32>({y})") };
                    match op {
                        BinOp::Add => format!("({x} + {y})"),
                        BinOp::Sub => format!("({x} - {y})"),
                        BinOp::Mul => format!("({x} * {y})"),
                        BinOp::MulHi => format!("tx_mulhi_{tn}({x}, {y})"),
                        BinOp::And => format!("({x} & {y})"),
                        BinOp::Or => format!("({x} | {y})"),
                        BinOp::Xor if t == IType::Bool => format!("({x} != {y})"),
                        BinOp::Xor => format!("({x} ^ {y})"),
                        BinOp::Shl => format!("({x} << {})", amt(y)),
                        BinOp::Shr => format!("({x} >> {})", amt(y)),
                        BinOp::Min => format!("min({x}, {y})"),
                        BinOp::Max => format!("max({x}, {y})"),
                    }
                } else {
                    match op {
                        BinOp::Add => format!("{x}.wrapping_add({y})"),
                        BinOp::Sub => format!("{x}.wrapping_sub({y})"),
                        BinOp::Mul => format!("{x}.wrapping_mul({y})"),
                        BinOp::MulHi if t == IType::I32 => format!("(((({x} as i64) * ({y} as i64)) >> 32) as i32)"),
                        BinOp::MulHi => format!("(((({x} as u64) * ({y} as u64)) >> 32) as u32)"),
                        BinOp::And => format!("({x} & {y})"),
                        BinOp::Or => format!("({x} | {y})"),
                        BinOp::Xor => format!("({x} ^ {y})"),
                        BinOp::Shl => format!("{x}.wrapping_shl({y} as u32)"),
                        BinOp::Shr => format!("{x}.wrapping_shr({y} as u32)"),
                        BinOp::Min => format!("{x}.min({y})"),
                        BinOp::Max => format!("{x}.max({y})"),
                    }
                }
            }
            INode::Un(op, a) => {
                let x = self.r(a);
                match (op, g.ty(a), self.lang) {
                    (UnOp::Neg, IType::U32, Lang::Wgsl) => format!("(0u - {x})"),
                    (UnOp::Neg, _, Lang::Wgsl) => format!("(-{x})"),
                    (UnOp::Neg, _, Lang::Rust) => format!("{x}.wrapping_neg()"),
                    (UnOp::Not, IType::Bool, _) | (UnOp::Not, _, Lang::Rust) => format!("(!{x})"),
                    (UnOp::Not, _, Lang::Wgsl) => format!("(~{x})"),
                }
            }
            INode::Cmp(op, a, b) => {
                let s = match op {
                    CmpOp::Eq => "==",
                    CmpOp::Ne => "!=",
                    CmpOp::Lt => "<",
                    CmpOp::Le => "<=",
                };
                format!("({} {s} {})", self.r(a), self.r(b))
            }
            INode::Select(c, a, b) => {
                if w {
                    format!("select({}, {}, {})", self.r(b), self.r(a), self.r(c))
                } else {
                    format!("(if {} {{ {} }} else {{ {} }})", self.r(c), self.r(a), self.r(b))
                }
            }
            INode::Bitcast(t, a) => {
                if w {
                    format!("bitcast<{}>({})", ty_name(self.lang, t), self.r(a))
                } else {
                    format!("({} as {})", self.r(a), ty_name(self.lang, t))
                }
            }
            INode::Load(b, i) => {
                let buf = &g.bufs[b as usize];
                let (n, i) = (&buf.name, self.r(i));
                match (buf.elem, self.lang) {
                    (Elem::I32 | Elem::U32, Lang::Wgsl) => format!("{n}[{i}]"),
                    (Elem::U8, Lang::Wgsl) => format!("extractBits({n}[{i} >> 2u], ({i} & 3u) * 8u, 8u)"),
                    (Elem::I8, Lang::Wgsl) => format!("extractBits(bitcast<i32>({n}[{i} >> 2u]), ({i} & 3u) * 8u, 8u)"),
                    (Elem::U32, Lang::Rust) => format!("{n}[{i} as usize]"),
                    (Elem::I32, Lang::Rust) => format!("({n}[{i} as usize] as i32)"),
                    (Elem::U8, Lang::Rust) => format!("(({n}[({i} >> 2) as usize] >> (({i} & 3) * 8)) & 255)"),
                    (Elem::I8, Lang::Rust) => format!("(({n}[({i} >> 2) as usize] >> (({i} & 3) * 8)) as u8 as i8 as i32)"),
                }
            }
            INode::Const(..) | INode::Param(..) | INode::State(..) | INode::LoopOut(..) => self.r(id),
        }
    }

    fn line(&mut self, depth: usize, s: &str) {
        for _ in 0..depth {
            self.out.push_str("    ");
        }
        self.out.push_str(s);
        self.out.push('\n');
    }

    fn items(&mut self, items: &[Item], depth: usize) {
        for it in items {
            match it {
                Item::Node(id) => {
                    let id = *id;
                    let skip = matches!(self.g.node(id), INode::Const(..) | INode::Param(..) | INode::State(..) | INode::LoopOut(..));
                    if skip || self.sched.inline[id.index()] {
                        continue;
                    }
                    let s = format!("let t{}: {} = {};", id.0, ty_name(self.lang, self.g.ty(id)), self.def(id));
                    self.line(depth, &s);
                }
                Item::Loop { l, pre, post } => self.lp(*l, pre, post, depth),
            }
        }
    }

    fn lp(&mut self, l: u32, pre: &[Item], post: &[Item], depth: usize) {
        let g = self.g;
        let lp = &g.loops[l as usize];
        for (k, &i) in lp.init.iter().enumerate() {
            let t = ty_name(self.lang, g.ty(i));
            let s = match self.lang {
                Lang::Wgsl => format!("var s{l}_{k}: {t} = {};", self.r(i)),
                Lang::Rust => format!("let mut s{l}_{k}: {t} = {};", self.r(i)),
            };
            self.line(depth, &s);
        }
        self.line(depth, "loop {");
        for k in 0..lp.init.len() {
            self.line(depth + 1, &format!("let st{l}_{k} = s{l}_{k};"));
        }
        self.items(pre, depth + 1);
        let c = self.r(lp.cond);
        self.line(depth + 1, &format!("if !{} {{ break; }}", paren(&c)));
        self.items(post, depth + 1);
        for (k, &x) in lp.next.iter().enumerate() {
            let s = format!("s{l}_{k} = {};", self.r(x));
            self.line(depth + 1, &s);
        }
        self.line(depth, "}");
    }
}

fn paren(s: &str) -> String {
    if s.starts_with('(') && s.ends_with(')') {
        s.to_string()
    } else {
        format!("({s})")
    }
}

impl IntGraph {
    fn emitter(&self, lang: Lang, outputs: &[IId], params: &[&str]) -> Em<'_> {
        Em { g: self, lang, sched: self.schedule(outputs), params: params.iter().map(|s| s.to_string()).collect(), out: String::new() }
    }

    fn nparams(&self) -> Vec<IType> {
        let mut t: Vec<IType> = Vec::new();
        for n in &self.nodes {
            if let INode::Param(p, ty) = *n {
                if t.len() <= p as usize {
                    t.resize(p as usize + 1, IType::U32);
                }
                t[p as usize] = ty;
            }
        }
        t
    }

    /// WGSL statements computing `outputs`: param `i` reads the expression
    /// `params[i]`, output `k` is assigned to `dests[k]`. Needs
    /// [`WGSL_PRELUDE`] in the module when the graph uses `MulHi`; buffers are
    /// referenced by their declared names.
    pub fn to_wgsl_body(&self, outputs: &[IId], params: &[&str], dests: &[&str], depth: usize) -> String {
        let mut e = self.emitter(Lang::Wgsl, outputs, params);
        let items = std::mem::take(&mut e.sched.items);
        e.items(&items, depth);
        // all outputs are computed before any destination is written (a
        // destination may be a param's source)
        let rs: Vec<String> = outputs.iter().map(|o| e.r(*o)).collect();
        for (k, r) in rs.iter().enumerate() {
            e.line(depth, &format!("let r{k} = {r};"));
        }
        for (k, d) in dests.iter().enumerate().take(outputs.len()) {
            e.line(depth, &format!("{d} = r{k};"));
        }
        e.out
    }

    /// A WGSL function `fn name(p0: T0, …) -> name_out` returning a struct of
    /// the outputs (`o0`, `o1`, …).
    pub fn to_wgsl_fn(&self, name: &str, outputs: &[IId]) -> String {
        let pt = self.nparams();
        let pn: Vec<String> = (0..pt.len()).map(|i| format!("p{i}")).collect();
        let pr: Vec<&str> = pn.iter().map(|s| s.as_str()).collect();
        let mut s = format!("struct {name}_out {{\n");
        for (k, o) in outputs.iter().enumerate() {
            let _ = writeln!(s, "    o{k}: {},", ty_name(Lang::Wgsl, self.ty(*o)));
        }
        s.push_str("}\n");
        let args: Vec<String> = pt.iter().enumerate().map(|(i, t)| format!("p{i}: {}", ty_name(Lang::Wgsl, *t))).collect();
        let _ = writeln!(s, "fn {name}({}) -> {name}_out {{", args.join(", "));
        let dn: Vec<String> = (0..outputs.len()).map(|k| format!("r.o{k}")).collect();
        let dr: Vec<&str> = dn.iter().map(|s| s.as_str()).collect();
        s.push_str(&format!("    var r: {name}_out;\n"));
        s.push_str(&self.to_wgsl_body(outputs, &pr, &dr, 1));
        s.push_str("    return r;\n}\n");
        s
    }

    /// A Rust function `fn name(p0: T0, …, buf: &[u32], …) -> [T; N]` (or a
    /// tuple when the output types differ); buffers
    /// take `u32` words (8-bit elements packed little endian).
    pub fn to_rust(&self, name: &str, outputs: &[IId]) -> String {
        let pt = self.nparams();
        let pn: Vec<String> = (0..pt.len()).map(|i| format!("p{i}")).collect();
        let pr: Vec<&str> = pn.iter().map(|s| s.as_str()).collect();
        let mut args: Vec<String> = pt.iter().enumerate().map(|(i, t)| format!("p{i}: {}", ty_name(Lang::Rust, *t))).collect();
        args.extend(self.bufs.iter().map(|b| format!("{}: &[u32]", b.name)));
        let rt: Vec<&str> = outputs.iter().map(|o| ty_name(Lang::Rust, self.ty(*o))).collect();
        let mut e = self.emitter(Lang::Rust, outputs, &pr);
        // one type: an array; mixed: a tuple
        let same = rt.windows(2).all(|w| w[0] == w[1]) && !rt.is_empty();
        let ret = if same { format!("[{}; {}]", rt[0], rt.len()) } else { format!("({},)", rt.join(", ")) };
        e.out = format!("#[allow(unused, clippy::all)]\npub fn {name}({}) -> {ret} {{\n", args.join(", "));
        let items = std::mem::take(&mut e.sched.items);
        e.items(&items, 1);
        let rs: Vec<String> = outputs.iter().map(|o| e.r(*o)).collect();
        let s = if same { format!("[{}]", rs.join(", ")) } else { format!("({},)", rs.join(", ")) };
        e.line(1, &s);
        e.out.push_str("}\n");
        e.out
    }
}
