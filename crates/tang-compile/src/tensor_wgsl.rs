//! Lower a [`TensorGraph`] to WGSL compute kernels.
//!
//! One kernel per materialized comp. Inlined comps become WGSL functions
//! called from their consumers, so a fused graph (e.g. online-softmax
//! attention with `q·k` recomputed inside) lowers to a single dispatch.
//! Views become index arithmetic.
//!
//! Schedules honoured: `inline` (recompute vs buffer) and `split` (one
//! workgroup per point, threads stride the reduce space and tree-combine
//! through workgroup memory — valid for any monoid, since combine is
//! associative). No tiling or shared-memory staging yet.

use std::collections::HashSet;
use std::fmt::Write;

use tang_expr::{ExprGraph, ExprId, Node};

use super::{fixed, numel, Comp, Format, Monoid, Op, TensorGraph, TensorId};

/// One dispatch.
pub struct WgslKernel {
    pub source: String,
    /// Node this kernel materializes.
    pub node: TensorId,
    /// Nodes bound as read-only storage, in binding order. The output is
    /// bound after these, then a uniform `{points, 0, 0, 0}`.
    pub reads: Vec<TensorId>,
    /// Output points; goes in the uniform as `params.count`.
    pub points: u32,
    /// Threads to launch (`points`, padded or multiplied for workgroup schedules).
    pub count: u32,
    pub workgroup_size: u32,
}

/// A buffer the program needs.
pub struct BufferSpec {
    pub node: TensorId,
    /// Length in elements.
    pub len: usize,
    /// Element format of the buffer (the node's storage precision):
    /// `F32` is `array<f32>`, `F16` is `array<f16>` (two per 32-bit word,
    /// needs `wgpu::Features::SHADER_F16`).
    pub format: Format,
    /// `Some(k)` if this is graph input `k`.
    pub input: Option<usize>,
}

pub struct WgslProgram {
    /// Kernels in dependency order.
    pub kernels: Vec<WgslKernel>,
    /// Every node that needs a buffer: inputs and materialized comps.
    pub buffers: Vec<BufferSpec>,
    /// `(reshape node, node whose buffer holds its bytes)`.
    pub aliases: Vec<(TensorId, TensorId)>,
}

impl WgslProgram {
    /// The buffer-owning node for `t` (itself unless `t` is a reshape).
    pub fn storage(&self, t: TensorId) -> TensorId {
        self.aliases
            .iter()
            .find(|(r, _)| *r == t)
            .map_or(t, |(_, s)| *s)
    }
}

impl TensorGraph {
    pub fn to_wgsl(&self, outputs: &[TensorId]) -> WgslProgram {
        let live = self.live(outputs);
        let n = self.nodes.len();
        let needs_buffer = |i: usize| {
            live[i]
                && !self.nodes[i].inline
                && !matches!(self.nodes[i].op, Op::Iota(_) | Op::Reshape(_))
        };
        // reshapes alias their storage node's buffer
        let aliases: Vec<(TensorId, TensorId)> = (0..n)
            .filter(|&i| live[i] && matches!(self.nodes[i].op, Op::Reshape(_)))
            .map(|i| (TensorId(i as u32), self.storage(TensorId(i as u32))))
            .collect();
        let buffers = (0..n)
            .filter(|&i| needs_buffer(i))
            .map(|i| {
                let (len, input) = match &self.nodes[i].op {
                    Op::Input(k) => (numel(&self.nodes[i].shape), Some(*k)),
                    Op::Comp(c) => (numel(c.points()) * c.width(), None),
                    Op::Iota(_) | Op::Reshape(_) => unreachable!(),
                };
                let format = self.nodes[i].precision.storage;
                assert!(
                    format != Format::BF16,
                    "bf16 storage is interpreter-only: WGSL has no bf16 type"
                );
                BufferSpec {
                    node: TensorId(i as u32),
                    len,
                    input,
                    format,
                }
            })
            .collect();

        let kernels = (0..n)
            .filter(|&i| needs_buffer(i) && self.comp_at(i).is_some())
            .map(|i| self.kernel_for(i))
            .collect();

        WgslProgram {
            kernels,
            buffers,
            aliases,
        }
    }

    /// Nodes reachable from `i` through inlined comps, including `i`,
    /// in dependency order.
    fn inline_closure(&self, i: usize) -> Vec<usize> {
        let mut order = Vec::new();
        let mut seen = HashSet::new();
        let mut stack = vec![(i, false)];
        while let Some((j, done)) = stack.pop() {
            if done {
                order.push(j);
                continue;
            }
            if !seen.insert(j) {
                continue;
            }
            stack.push((j, true));
            let descend = match &self.nodes[j].op {
                Op::Comp(_) => j == i || self.nodes[j].inline,
                Op::Reshape(_) => true,
                Op::Input(_) | Op::Iota(_) => false,
            };
            if descend {
                stack.extend(self.deps(j).into_iter().map(|d| (d, false)));
            }
        }
        order
    }

    fn kernel_for(&self, root: usize) -> WgslKernel {
        let c = self.comp_at(root).unwrap();
        let order = self.inline_closure(root);
        let reads: Vec<usize> = order
            .iter()
            .copied()
            .filter(|&j| {
                j != root
                    && !self.nodes[j].inline
                    && !matches!(self.nodes[j].op, Op::Iota(_) | Op::Reshape(_))
            })
            .collect();

        let mut s = String::new();
        writeln!(s, "// Auto-generated by tang-compile from a TensorGraph").unwrap();
        writeln!(
            s,
            "struct Params {{ count: u32, _p1: u32, _p2: u32, _p3: u32 }}"
        )
        .unwrap();
        for (b, &j) in reads.iter().enumerate() {
            writeln!(
                s,
                "@group(0) @binding({b}) var<storage, read> b{j}: array<{}>;",
                elem_ty(self.nodes[j].precision.storage)
            )
            .unwrap();
        }
        writeln!(
            s,
            "@group(0) @binding({}) var<storage, read_write> out: array<{}>;",
            reads.len(),
            out_ty(self.nodes[root].precision.storage)
        )
        .unwrap();
        writeln!(
            s,
            "@group(0) @binding({}) var<uniform> params: Params;\n",
            reads.len() + 1
        )
        .unwrap();

        for &j in &order {
            self.emit_node(&mut s, j, &reads);
        }

        let w = c.width();
        let k = c.points().len();
        let points = numel(c.points()) as u32;
        let split = self.nodes[root].split;
        let out_f = self.nodes[root].precision.storage;
        let accum = self.nodes[root].precision.accum;
        if self.nodes[root].tile.0 > 0 {
            return self.tile_kernel(root, s, reads);
        }
        let reads = reads.into_iter().map(|j| TensorId(j as u32)).collect();

        if split == 0 {
            let args = (0..k)
                .map(|d| format!("p{d}"))
                .collect::<Vec<_>>()
                .join(", ");
            writeln!(s, "@compute @workgroup_size(256)").unwrap();
            writeln!(
                s,
                "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{"
            )
            .unwrap();
            writeln!(s, "    let g = gid.x + gid.y * {}u;", 65535 * 256).unwrap();
            writeln!(s, "    if (g >= params.count) {{ return; }}").unwrap();
            emit_decompose(&mut s, "g", &fixed(c.points()), "p");
            writeln!(s, "    let v = point_{root}({args});").unwrap();
            for slot in 0..w {
                let st = store(out_f, &format!("g * {w}u + {slot}u"), &format!("v[{slot}]"));
                writeln!(s, "    {st}").unwrap();
            }
            writeln!(s, "}}").unwrap();
            return WgslKernel {
                source: s,
                node: TensorId(root as u32),
                reads,
                points,
                count: points,
                workgroup_size: 256,
            };
        }

        // One workgroup per point; threads stride the flattened reduce
        // space, then tree-combine through workgroup memory.
        let m = c.monoid.as_ref().expect("split needs a reduction");
        let r_total = numel(&c.red);
        writeln!(s, "var<workgroup> wg_acc: array<array<f32, {w}>, {split}>;").unwrap();
        writeln!(s, "@compute @workgroup_size({split})").unwrap();
        writeln!(
            s,
            "fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{"
        )
        .unwrap();
        writeln!(s, "    let g = wid.x + wid.y * 65535u;").unwrap();
        writeln!(s, "    let tid = lid.x;").unwrap();
        writeln!(s, "    if (g >= params.count) {{ return; }}").unwrap();
        emit_decompose(&mut s, "g", &fixed(c.points()), "p");
        emit_scalar_acc(&mut s, "    ", m);
        writeln!(
            s,
            "    for (var ri: u32 = tid; ri < {r_total}u; ri = ri + {split}u) {{"
        )
        .unwrap();
        emit_decompose(&mut s, "ri", &fixed(&c.red), "r");
        emit_step(
            &mut s,
            "        ",
            &format!("body_{root}({})", arg_reads(self, c)),
            m,
            accum,
        );
        writeln!(s, "    }}").unwrap();
        writeln!(s, "    wg_acc[tid] = {};", acc_array(w)).unwrap();
        writeln!(s, "    workgroupBarrier();").unwrap();
        writeln!(
            s,
            "    for (var stride: u32 = {split}u / 2u; stride > 0u; stride = stride / 2u) {{"
        )
        .unwrap();
        writeln!(s, "        if (tid < stride) {{").unwrap();
        writeln!(
            s,
            "            wg_acc[tid] = combine_{root}(wg_acc[tid], wg_acc[tid + stride]);"
        )
        .unwrap();
        writeln!(s, "        }}").unwrap();
        writeln!(s, "        workgroupBarrier();").unwrap();
        writeln!(s, "    }}").unwrap();
        writeln!(s, "    if (tid == 0u) {{").unwrap();
        for slot in 0..w {
            let st = store(
                out_f,
                &format!("g * {w}u + {slot}u"),
                &format!("wg_acc[0][{slot}]"),
            );
            writeln!(s, "        {st}").unwrap();
        }
        writeln!(s, "    }}").unwrap();
        writeln!(s, "}}").unwrap();

        WgslKernel {
            source: s,
            node: TensorId(root as u32),
            reads,
            points,
            count: points * split,
            workgroup_size: split,
        }
    }

    /// `read_j` for every node; `body_j`/`combine_j`/`point_j` for comps
    /// that are computed inside this kernel.
    fn emit_node(&self, s: &mut String, j: usize, reads: &[usize]) {
        let node = &self.nodes[j];
        let r = node.shape.len();
        let coords = (0..r)
            .map(|d| format!("c{d}: u32"))
            .collect::<Vec<_>>()
            .join(", ");
        match &node.op {
            Op::Input(_) => {
                writeln!(s, "fn read_{j}({coords}) -> f32 {{").unwrap();
                let v = load(
                    node.precision.storage,
                    &format!("b{j}"),
                    &flat("c", 0, &fixed(&node.shape)),
                );
                writeln!(s, "    return {v};").unwrap();
                writeln!(s, "}}\n").unwrap();
            }
            Op::Iota(d) => {
                writeln!(s, "fn read_{j}({coords}) -> f32 {{").unwrap();
                writeln!(s, "    return f32(c{d});").unwrap();
                writeln!(s, "}}\n").unwrap();
            }
            Op::Reshape(src) => {
                // same bytes: flat index here is the flat index there
                let src_shape = fixed(&self.nodes[src.n()].shape);
                writeln!(s, "fn read_{j}({coords}) -> f32 {{").unwrap();
                writeln!(s, "    var p = {};", flat("c", 0, &fixed(&node.shape))).unwrap();
                let mut args = Vec::new();
                for d in (0..src_shape.len()).rev() {
                    writeln!(s, "    let s{d} = p % {}u;", src_shape[d]).unwrap();
                    writeln!(s, "    p = p / {}u;", src_shape[d]).unwrap();
                    args.push(format!("s{d}"));
                }
                args.reverse();
                writeln!(s, "    return read_{}({});", src.n(), args.join(", ")).unwrap();
                writeln!(s, "}}\n").unwrap();
            }
            Op::Comp(c) => {
                let w = c.width();
                let k = c.points().len();
                let materialized = reads.contains(&j);
                if !materialized {
                    emit_scalar_fn(s, &format!("body_{j}"), &c.body, &c.body_outs, c.args.len());
                    if let Some(m) = &c.monoid {
                        emit_combine(s, j, m, node.precision.accum);
                    }
                    self.emit_point(s, j, c);
                }

                writeln!(s, "fn read_{j}({coords}) -> f32 {{").unwrap();
                let slot = flat("c", k, &fixed(&c.out[k..]));
                if materialized {
                    // buffer layout: [points][width]
                    let pt = flat("c", 0, &fixed(c.points()));
                    let v = load(
                        node.precision.storage,
                        &format!("b{j}"),
                        &format!("({pt}) * {w}u + ({slot})"),
                    );
                    writeln!(s, "    return {v};").unwrap();
                } else {
                    let args = (0..k)
                        .map(|d| format!("c{d}"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    writeln!(s, "    let v = point_{j}({args});").unwrap();
                    writeln!(s, "    return v[{slot}];").unwrap();
                }
                writeln!(s, "}}\n").unwrap();
            }
        }
    }

    /// `point_j(p…) -> array<f32, W>`: the comp at one output point.
    fn emit_point(&self, s: &mut String, j: usize, c: &Comp) {
        let w = c.width();
        let k = c.points().len();
        let params = (0..k)
            .map(|d| format!("p{d}: u32"))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(s, "fn point_{j}({params}) -> array<f32, {w}> {{").unwrap();
        let reads = arg_reads(self, c);
        let Some(m) = &c.monoid else {
            writeln!(s, "    return body_{j}({reads});").unwrap();
            writeln!(s, "}}\n").unwrap();
            return;
        };
        emit_scalar_acc(s, "    ", m);
        let mut indent = String::from("    ");
        for (i, n) in fixed(&c.red).into_iter().enumerate() {
            writeln!(
                s,
                "{indent}for (var r{i}: u32 = 0u; r{i} < {n}u; r{i} = r{i} + 1u) {{"
            )
            .unwrap();
            indent.push_str("    ");
        }
        emit_step(s, &indent, &format!("body_{j}({reads})"), m, self.nodes[j].precision.accum);
        for _ in &c.red {
            indent.truncate(indent.len() - 4);
            writeln!(s, "{indent}}}").unwrap();
        }
        writeln!(s, "    return {};", acc_array(w)).unwrap();
        writeln!(s, "}}\n").unwrap();
    }
}

/// `var acc0: f32 = id0; …` — accumulator slots as scalars, so the
/// combine runs in registers instead of copying `array<f32, W>` values.
fn emit_scalar_acc(s: &mut String, indent: &str, m: &Monoid) {
    for (i, v) in m.identity.iter().enumerate() {
        writeln!(s, "{indent}var acc{i}: f32 = {};", lit(*v)).unwrap();
    }
}

fn acc_array(w: usize) -> String {
    let slots = (0..w)
        .map(|i| format!("acc{i}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("array<f32, {w}>({slots})")
}

/// One reduction step: evaluate `elem` (an `array<f32, W>` expression)
/// and fold it into the scalar accumulators with the combine inlined.
fn emit_step(s: &mut String, indent: &str, elem: &str, m: &Monoid, accum: Format) {
    let w = m.width;
    writeln!(s, "{indent}let el = {elem};").unwrap();
    let outs = emit_scalar_mapped(s, indent, &m.combine, &m.outs, "k", &|v| {
        if (v as usize) < w {
            format!("acc{v}")
        } else {
            format!("el[{}]", v as usize - w)
        }
    });
    for (i, o) in outs.iter().enumerate() {
        writeln!(s, "{indent}let nx{i} = {o};").unwrap();
    }
    for i in 0..w {
        writeln!(s, "{indent}acc{i} = {};", quant(accum, &format!("nx{i}"))).unwrap();
    }
}

/// Emit `let {prefix}{i} = …;` for the live nodes of `g` and return the
/// expressions for `outs`; `var_expr` names input var `i`.
fn emit_scalar_mapped(
    s: &mut String,
    indent: &str,
    g: &ExprGraph,
    outs: &[ExprId],
    prefix: &str,
    var_expr: &dyn Fn(u16) -> String,
) -> Vec<String> {
    let mut live: Vec<usize> = g.live_set(outs).into_iter().collect();
    live.sort_unstable();
    let nodes = g.nodes_slice();
    let e = |id: &ExprId| match nodes[id.index() as usize] {
        Node::Var(v) => var_expr(v),
        _ => format!("{prefix}{}", id.index()),
    };
    for i in live {
        let rhs = match &nodes[i] {
            Node::Var(_) => continue,
            Node::Lit(b) => lit(f64::from_bits(*b)),
            Node::Add(a, b) => format!("{} + {}", e(a), e(b)),
            Node::Mul(a, b) => format!("{} * {}", e(a), e(b)),
            Node::Neg(a) => format!("-{}", e(a)),
            Node::Recip(a) => format!("1.0 / {}", e(a)),
            Node::Sqrt(a) => format!("sqrt({})", e(a)),
            Node::Sin(a) => format!("sin({})", e(a)),
            Node::Atan2(a, b) => format!("atan2({}, {})", e(a), e(b)),
            Node::Exp2(a) => format!("exp2({})", e(a)),
            Node::Log2(a) => format!("log2({})", e(a)),
            Node::Select(c, a, b) => format!("select({}, {}, {} > 0.0)", e(b), e(a), e(c)),
        };
        writeln!(s, "{indent}let {prefix}{i} = {rhs};").unwrap();
    }
    outs.iter().map(e).collect()
}

pub(super) fn identity(m: &Monoid) -> String {
    m.identity
        .iter()
        .map(|v| lit(*v))
        .collect::<Vec<_>>()
        .join(", ")
}

/// `read_src(view(p…, r…))` for every arg, as call arguments.
fn arg_reads(g: &TensorGraph, c: &Comp) -> String {
    let k = c.points().len();
    let coord_name = |d: usize| {
        if d < k {
            format!("p{d}")
        } else {
            format!("r{}", d - k)
        }
    };
    c.args
        .iter()
        .map(|a| {
            let signed: Vec<String> = a
                .view
                .forms
                .iter()
                .map(|f| {
                    let mut terms = vec![format!("{}", f.offset)];
                    for (d, &coef) in f.coefs.iter().enumerate() {
                        if coef != 0 {
                            terms.push(format!("{coef} * i32({})", coord_name(d)));
                        }
                    }
                    format!("({})", terms.join(" + "))
                })
                .collect();
            let idx: Vec<String> = signed.iter().map(|e| format!("u32({e})")).collect();
            let read = format!("read_{}({})", a.src.n(), idx.join(", "));
            if !a.view.guard {
                return read;
            }
            // zero padding: the read itself may wrap (robust buffer
            // access keeps it in bounds); the select discards it
            let shape = fixed(&g.nodes[a.src.n()].shape);
            let cond = signed
                .iter()
                .zip(&shape)
                .map(|(e, n)| format!("(u32({e}) < {n}u)"))
                .collect::<Vec<_>>()
                .join(" && ");
            format!("select(0.0, {read}, {cond})")
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Row-major flat index of coords `prefix{start}..` over `shape`.
fn flat(prefix: &str, start: usize, shape: &[usize]) -> String {
    if shape.is_empty() {
        return "0u".into();
    }
    let mut acc = format!("{prefix}{start}");
    for (i, n) in shape.iter().enumerate().skip(1) {
        acc = format!("({acc}) * {n}u + {prefix}{}", start + i);
    }
    acc
}

/// `g` → `p0..pk` over `shape`, row-major.
pub(super) fn emit_decompose(s: &mut String, g: &str, shape: &[usize], prefix: &str) {
    let mut rem = g.to_string();
    for d in (0..shape.len()).rev() {
        let n = shape[d];
        writeln!(s, "    let {prefix}{d} = ({rem}) % {n}u;").unwrap();
        rem = format!("({rem}) / {n}u");
    }
}

pub(super) fn lit(v: f64) -> String {
    if v == f64::NEG_INFINITY {
        return "-3.4028235e38".into();
    }
    if v == f64::INFINITY {
        return "3.4028235e38".into();
    }
    let s = format!("{:?}", v as f32);
    if s.contains('.') || s.contains('e') {
        s
    } else {
        format!("{s}.0")
    }
}

/// `fn name(v0: f32, …) -> array<f32, W>` evaluating `outs` of `g`.
fn emit_scalar_fn(s: &mut String, name: &str, g: &ExprGraph, outs: &[ExprId], n_vars: usize) {
    let params = (0..n_vars)
        .map(|i| format!("v{i}: f32"))
        .collect::<Vec<_>>()
        .join(", ");
    writeln!(s, "fn {name}({params}) -> array<f32, {}> {{", outs.len()).unwrap();
    emit_scalar_body(s, g, outs, Format::F32);
    writeln!(s, "}}\n").unwrap();
}

fn emit_combine(s: &mut String, j: usize, m: &Monoid, accum: Format) {
    let w = m.width;
    writeln!(
        s,
        "fn combine_{j}(a: array<f32, {w}>, e: array<f32, {w}>) -> array<f32, {w}> {{"
    )
    .unwrap();
    for i in 0..w {
        writeln!(s, "    let v{i} = a[{i}];").unwrap();
        writeln!(s, "    let v{} = e[{i}];", w + i).unwrap();
    }
    emit_scalar_body(s, &m.combine, &m.outs, accum);
    writeln!(s, "}}\n").unwrap();
}

fn emit_scalar_body(s: &mut String, g: &ExprGraph, outs: &[ExprId], round: Format) {
    let mut live: Vec<usize> = g.live_set(outs).into_iter().collect();
    live.sort_unstable();
    let nodes = g.nodes_slice();
    let e = |id: &ExprId| match nodes[id.index() as usize] {
        Node::Var(v) => format!("v{v}"),
        _ => format!("e{}", id.index()),
    };
    for i in live {
        let rhs = match &nodes[i] {
            Node::Var(_) => continue,
            Node::Lit(b) => lit(f64::from_bits(*b)),
            Node::Add(a, b) => format!("{} + {}", e(a), e(b)),
            Node::Mul(a, b) => format!("{} * {}", e(a), e(b)),
            Node::Neg(a) => format!("-{}", e(a)),
            Node::Recip(a) => format!("1.0 / {}", e(a)),
            Node::Sqrt(a) => format!("sqrt({})", e(a)),
            Node::Sin(a) => format!("sin({})", e(a)),
            Node::Atan2(a, b) => format!("atan2({}, {})", e(a), e(b)),
            Node::Exp2(a) => format!("exp2({})", e(a)),
            Node::Log2(a) => format!("log2({})", e(a)),
            Node::Select(c, a, b) => format!("select({}, {}, {} > 0.0)", e(b), e(a), e(c)),
        };
        writeln!(s, "    let e{i} = {rhs};").unwrap();
    }
    let vals = outs
        .iter()
        .map(|o| quant(round, &e(o)))
        .collect::<Vec<_>>()
        .join(", ");
    writeln!(s, "    return array<f32, {}>({vals});", outs.len()).unwrap();
}

/// WGSL element type of a read-only buffer in format `f`. f16 is packed
/// two per `u32` (naga 24 has no `enable f16`; `unpack2x16float` is core
/// WGSL), so it needs no device feature.
pub(super) fn elem_ty(f: Format) -> &'static str {
    match f {
        Format::F32 => "f32",
        Format::F16 => "u32",
        Format::BF16 => panic!("bf16 storage is interpreter-only: WGSL has no bf16 type"),
    }
}

/// Element type of the output buffer: packed f16 halves of one word are
/// written by different threads, so the word is atomic.
fn out_ty(f: Format) -> &'static str {
    match f {
        Format::F16 => "atomic<u32>",
        f => elem_ty(f),
    }
}

/// Element `idx` of buffer `buf` (format `f`), as the f32 the kernel
/// computes in.
pub(super) fn load(f: Format, buf: &str, idx: &str) -> String {
    match f {
        Format::F32 => format!("{buf}[{idx}]"),
        Format::F16 => format!("unpack2x16float({buf}[({idx}) >> 1u])[({idx}) & 1u]"),
        Format::BF16 => panic!("bf16 storage is interpreter-only: WGSL has no bf16 type"),
    }
}

/// Statement writing f32 value `v` to `out[idx]` in format `f`. An f16
/// half is merged into its word with two atomics on disjoint bits, so
/// the neighbour's half is never clobbered whatever the interleaving.
pub(super) fn store(f: Format, idx: &str, v: &str) -> String {
    match f {
        Format::F32 => format!("out[{idx}] = {v};"),
        Format::F16 => format!(
            "{{ let i_ = {idx}; let sh_ = (i_ & 1u) * 16u; \
             atomicAnd(&out[i_ >> 1u], ~(0xffffu << sh_)); \
             atomicOr(&out[i_ >> 1u], (pack2x16float(vec2<f32>({v}, 0.0)) & 0xffffu) << sh_); }}"
        ),
        Format::BF16 => panic!("bf16 storage is interpreter-only: WGSL has no bf16 type"),
    }
}

/// An f32 accumulator value `e` rounded to format `f` (kept in an f32
/// register): f16 accumulation as rounding after every combine, which
/// is what the interpreter emulates. It reproduces the numerics of an
/// f16 accumulator, not its speed.
pub(super) fn quant(f: Format, e: &str) -> String {
    match f {
        Format::F32 => e.to_string(),
        Format::F16 => format!("unpack2x16float(pack2x16float(vec2<f32>({e}, 0.0))).x"),
        Format::BF16 => panic!("bf16 accumulation is interpreter-only"),
    }
}
