//! Tiled WGSL schedule: one point per thread, workgroup-invariant reads
//! staged chunk by chunk in workgroup memory.
//!
//! Unlike the function-per-comp emitter in `tensor_wgsl`, this one fully
//! inlines the closure into a single loop nest and tracks every buffer
//! address as an affine form over the loop variables. A read whose
//! address does not depend on the point coordinates is the same for every
//! thread in the workgroup, so it is loaded cooperatively once per chunk
//! of the reduce axis. With attention this stages the k and v tiles; with
//! matmul it stages the b tile. Nothing here knows about either.

use std::collections::BTreeMap;
use std::fmt::Write;

use tang_expr::{ExprGraph, ExprId, Node};

use super::wgsl::{emit_decompose, identity, lit, WgslKernel};
use super::{numel, Comp, Op, TensorGraph, TensorId};

/// Affine expression over named loop variables.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
struct Sym {
    terms: BTreeMap<String, i64>,
    offset: i64,
}

impl Sym {
    fn var(name: &str) -> Self {
        let mut s = Sym::default();
        s.terms.insert(name.to_string(), 1);
        s
    }

    fn scaled(&self, c: i64) -> Self {
        Sym {
            terms: self.terms.iter().map(|(k, v)| (k.clone(), v * c)).collect(),
            offset: self.offset * c,
        }
    }

    fn add(&mut self, other: &Sym) {
        for (k, v) in &other.terms {
            *self.terms.entry(k.clone()).or_insert(0) += v;
        }
        self.terms.retain(|_, v| *v != 0);
        self.offset += other.offset;
    }

    fn depends_on_prefix(&self, prefix: &str) -> bool {
        self.terms.keys().any(|k| k.starts_with(prefix))
    }

    fn wgsl(&self) -> String {
        let mut parts = vec![self.offset.to_string()];
        for (k, v) in &self.terms {
            parts.push(format!("{v} * i32({k})"));
        }
        format!("u32({})", parts.join(" + "))
    }
}

/// A staged region: all reads of `buffer` with these coefficients over
/// the reduce variable and inner loop variables, differing only in offset.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct RegionKey {
    buffer: usize,
    /// coefficient on the root reduce var
    red: i64,
    /// (inner var name, coefficient)
    inner: Vec<(String, i64)>,
}

struct Region {
    key: RegionKey,
    lo: i64,
    hi: i64,
    /// extents of the inner vars, in `key.inner` order
    extents: Vec<usize>,
}

enum Val {
    Expr(String),
    Arr(String),
}

impl Val {
    fn slot(&self, slot: &Sym) -> String {
        match self {
            Val::Expr(e) => e.clone(),
            Val::Arr(a) => format!("{a}[{}]", slot.wgsl()),
        }
    }
}

struct Ctx<'a> {
    g: &'a TensorGraph,
    reads: &'a [usize],
    /// extents of inner loop vars
    extents: BTreeMap<String, usize>,
    regions: Vec<Region>,
    /// first pass collects regions, second emits against them
    collect: bool,
    counter: usize,
}

impl Ctx<'_> {
    fn fresh(&mut self) -> usize {
        self.counter += 1;
        self.counter
    }

    /// Read of buffer-backed node `j` at an address.
    fn buffer_read(&mut self, j: usize, addr: Sym) -> String {
        if !addr.depends_on_prefix("p") && addr.terms.contains_key("r0") {
            let red = addr.terms["r0"];
            let inner: Vec<(String, i64)> = addr
                .terms
                .iter()
                .filter(|(k, _)| k.as_str() != "r0")
                .map(|(k, v)| (k.clone(), *v))
                .collect();
            let key = RegionKey {
                buffer: j,
                red,
                inner,
            };
            let idx = match self.regions.iter().position(|r| r.key == key) {
                Some(i) => {
                    let r = &mut self.regions[i];
                    r.lo = r.lo.min(addr.offset);
                    r.hi = r.hi.max(addr.offset);
                    i
                }
                None => {
                    let extents = key.inner.iter().map(|(k, _)| self.extents[k]).collect();
                    self.regions.push(Region {
                        key,
                        lo: addr.offset,
                        hi: addr.offset,
                        extents,
                    });
                    self.regions.len() - 1
                }
            };
            if self.collect {
                return String::new();
            }
            let r = &self.regions[idx];
            let span = r.hi - r.lo + 1;
            // local index: (t * innerN + innerFlat) * span + (off - lo)
            let mut inner_flat = String::from("0u");
            for ((name, _), &ext) in r.key.inner.iter().zip(&r.extents) {
                inner_flat = format!("({inner_flat}) * {ext}u + {name}");
            }
            let inner_n: usize = r.extents.iter().product();
            return format!(
                "stage_{idx}[((r0 - chunk) * {inner_n}u + ({inner_flat})) * {span}u + {}u]",
                addr.offset - r.lo
            );
        }
        format!("b{j}[{}]", addr.wgsl())
    }

    /// Value of node `j` at full-shape coordinates `coords`.
    fn expand(&mut self, out: &mut String, j: usize, coords: &[Sym]) -> Val {
        let node = &self.g.nodes[j];
        match &node.op {
            Op::Input(_) => {
                let addr = flat(coords, &node.shape);
                Val::Expr(self.buffer_read(j, addr))
            }
            Op::Comp(c) if self.reads.contains(&j) => {
                let k = c.points().len();
                let mut addr = flat(&coords[..k], c.points()).scaled(c.width() as i64);
                addr.add(&flat(&coords[k..], &c.out[k..]));
                Val::Expr(self.buffer_read(j, addr))
            }
            Op::Comp(c) => {
                let c: &Comp = c;
                let k = c.points().len();
                let n = self.fresh();
                let w = c.width();
                let red_vars: Vec<String> = (0..c.red.len()).map(|i| format!("q{n}_{i}")).collect();
                for (v, &ext) in red_vars.iter().zip(&c.red) {
                    self.extents.insert(v.clone(), ext);
                }
                let mut space: Vec<Sym> = coords[..k].to_vec();
                space.extend(red_vars.iter().map(|v| Sym::var(v)));

                let mut inner = String::new();
                let arg_vals: Vec<String> = c
                    .args
                    .iter()
                    .map(|a| {
                        let src_shape = self.g.nodes[a.src.n()].shape.clone();
                        let cs: Vec<Sym> = a
                            .view
                            .forms
                            .iter()
                            .map(|f| {
                                let mut s = Sym {
                                    offset: f.offset,
                                    ..Default::default()
                                };
                                for (d, &coef) in f.coefs.iter().enumerate() {
                                    if coef != 0 {
                                        s.add(&space[d].scaled(coef));
                                    }
                                }
                                s
                            })
                            .collect();
                        let pk = src_shape.len() - self.tuple_rank(a.src.n());
                        let slot = flat(&cs[pk..], &src_shape[pk..]);
                        let v = self.expand(&mut inner, a.src.n(), &cs);
                        v.slot(&slot)
                    })
                    .collect();
                let body =
                    emit_scalar(&mut inner, &c.body, &c.body_outs, &format!("t{n}_"), &|v| {
                        arg_vals[v as usize].clone()
                    });
                let Some(m) = &c.monoid else {
                    out.push_str(&inner);
                    if w == 1 {
                        return Val::Expr(body[0].clone());
                    }
                    writeln!(out, "var v{n} = array<f32, {w}>({});", body.join(", ")).unwrap();
                    return Val::Arr(format!("v{n}"));
                };
                writeln!(out, "var acc{n} = array<f32, {w}>({});", identity(m)).unwrap();
                for (v, &ext) in red_vars.iter().zip(&c.red) {
                    writeln!(
                        out,
                        "for (var {v}: u32 = 0u; {v} < {ext}u; {v} = {v} + 1u) {{"
                    )
                    .unwrap();
                }
                out.push_str(&inner);
                let combined = emit_scalar(out, &m.combine, &m.outs, &format!("c{n}_"), &|v| {
                    if (v as usize) < w {
                        format!("acc{n}[{v}]")
                    } else {
                        body[v as usize - w].clone()
                    }
                });
                writeln!(out, "acc{n} = array<f32, {w}>({});", combined.join(", ")).unwrap();
                for _ in &c.red {
                    writeln!(out, "}}").unwrap();
                }
                Val::Arr(format!("acc{n}"))
            }
        }
    }

    fn tuple_rank(&self, j: usize) -> usize {
        match &self.g.nodes[j].op {
            Op::Comp(c) => c.tuple,
            Op::Input(_) => 0,
        }
    }
}

/// Row-major flat index of symbolic coords over `shape`.
fn flat(coords: &[Sym], shape: &[usize]) -> Sym {
    let mut acc = Sym::default();
    for (c, &n) in coords.iter().zip(shape) {
        acc = acc.scaled(n as i64);
        acc.add(c);
    }
    acc
}

/// Emit `let {prefix}{i} = …;` for the live nodes of `g` and return the
/// expressions for `outs`. `var_expr` names input var `i`.
fn emit_scalar(
    out: &mut String,
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
        writeln!(out, "let {prefix}{i} = {rhs};").unwrap();
    }
    outs.iter().map(e).collect()
}

impl TensorGraph {
    /// Emit the tiled kernel for `root`, continuing `s` (which already
    /// holds the bindings and per-node functions).
    pub(super) fn tile_kernel(&self, root: usize, mut s: String, reads: Vec<usize>) -> WgslKernel {
        let c = self.comp_at(root).unwrap();
        let (b, chunk) = self.nodes[root].tile;
        let w = c.width();
        let k = c.points().len();
        let points = numel(c.points()) as u32;
        let r_total = c.red[0];
        let m = c.monoid.as_ref().expect("tile needs a reduction");

        // Expand twice: first to discover staged regions, then for real.
        let mut body = String::new();
        let mut ctx = Ctx {
            g: self,
            reads: &reads,
            extents: BTreeMap::new(),
            regions: Vec::new(),
            collect: true,
            counter: 0,
        };
        ctx.extents.insert("r0".into(), r_total);
        let coords: Vec<Sym> = (0..k).map(|d| Sym::var(&format!("p{d}"))).collect();
        let (body_vals, combined) = {
            let run = |ctx: &mut Ctx, body: &mut String| {
                let space: Vec<Sym> = coords.iter().cloned().chain([Sym::var("r0")]).collect();
                let arg_vals: Vec<String> = c
                    .args
                    .iter()
                    .map(|a| {
                        let src_shape = self.nodes[a.src.n()].shape.clone();
                        let cs: Vec<Sym> = a
                            .view
                            .forms
                            .iter()
                            .map(|f| {
                                let mut sy = Sym {
                                    offset: f.offset,
                                    ..Default::default()
                                };
                                for (d, &coef) in f.coefs.iter().enumerate() {
                                    if coef != 0 {
                                        sy.add(&space[d].scaled(coef));
                                    }
                                }
                                sy
                            })
                            .collect();
                        let pk = src_shape.len() - ctx.tuple_rank(a.src.n());
                        let slot = flat(&cs[pk..], &src_shape[pk..]);
                        let v = ctx.expand(body, a.src.n(), &cs);
                        v.slot(&slot)
                    })
                    .collect();
                let vals = emit_scalar(body, &c.body, &c.body_outs, "t0_", &|v| {
                    arg_vals[v as usize].clone()
                });
                let combined = emit_scalar(body, &m.combine, &m.outs, "c0_", &|v| {
                    if (v as usize) < w {
                        format!("acc[{v}]")
                    } else {
                        vals[v as usize - w].clone()
                    }
                });
                (vals, combined)
            };
            run(&mut ctx, &mut body);
            ctx.collect = false;
            ctx.counter = 0;
            body.clear();
            run(&mut ctx, &mut body)
        };
        let _ = body_vals;

        // workgroup storage
        for (i, r) in ctx.regions.iter().enumerate() {
            let inner_n: usize = r.extents.iter().product();
            let span = (r.hi - r.lo + 1) as usize;
            writeln!(
                s,
                "var<workgroup> stage_{i}: array<f32, {}>;",
                chunk as usize * inner_n * span
            )
            .unwrap();
        }

        writeln!(s, "@compute @workgroup_size({b})").unwrap();
        writeln!(
            s,
            "fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{"
        )
        .unwrap();
        writeln!(s, "    let tid = lid.x;").unwrap();
        writeln!(s, "    let g_raw = (wid.x + wid.y * 65535u) * {b}u + tid;").unwrap();
        writeln!(s, "    let is_live = g_raw < params.count;").unwrap();
        writeln!(s, "    let g = min(g_raw, params.count - 1u);").unwrap();
        emit_decompose(&mut s, "g", c.points(), "p");
        writeln!(s, "    var acc = array<f32, {w}>({});", identity(m)).unwrap();
        writeln!(
            s,
            "    for (var chunk: u32 = 0u; chunk < {r_total}u; chunk = chunk + {chunk}u) {{"
        )
        .unwrap();
        for (i, r) in ctx.regions.iter().enumerate() {
            let inner_n: usize = r.extents.iter().product();
            let span = (r.hi - r.lo + 1) as usize;
            let total = chunk as usize * inner_n * span;
            writeln!(
                s,
                "        for (var idx: u32 = tid; idx < {total}u; idx = idx + {b}u) {{"
            )
            .unwrap();
            writeln!(s, "            let t = idx / {}u;", inner_n * span).unwrap();
            writeln!(s, "            let rem = idx % {}u;", inner_n * span).unwrap();
            writeln!(s, "            let o = rem % {span}u;").unwrap();
            writeln!(s, "            let inner = rem / {span}u;").unwrap();
            let mut addr = format!("{} + {} * i32(chunk + t) + i32(o)", r.lo, r.key.red);
            let mut rem = "inner".to_string();
            for ((name, coef), &ext) in r.key.inner.iter().zip(&r.extents).rev() {
                writeln!(s, "            let {name} = ({rem}) % {ext}u;").unwrap();
                rem = format!("({rem}) / {ext}u");
                addr.push_str(&format!(" + {coef} * i32({name})"));
            }
            writeln!(s, "            if (chunk + t < {r_total}u) {{").unwrap();
            writeln!(
                s,
                "                stage_{i}[idx] = b{}[u32({addr})];",
                r.key.buffer
            )
            .unwrap();
            writeln!(s, "            }}").unwrap();
            writeln!(s, "        }}").unwrap();
        }
        writeln!(s, "        workgroupBarrier();").unwrap();
        writeln!(
            s,
            "        for (var r0: u32 = chunk; r0 < min(chunk + {chunk}u, {r_total}u); r0 = r0 + 1u) {{"
        )
        .unwrap();
        for line in body.lines() {
            writeln!(s, "            {line}").unwrap();
        }
        writeln!(
            s,
            "            acc = array<f32, {w}>({});",
            combined.join(", ")
        )
        .unwrap();
        writeln!(s, "        }}").unwrap();
        writeln!(s, "        workgroupBarrier();").unwrap();
        writeln!(s, "    }}").unwrap();
        writeln!(s, "    if (is_live) {{").unwrap();
        for slot in 0..w {
            writeln!(s, "        out[g * {w}u + {slot}u] = acc[{slot}];").unwrap();
        }
        writeln!(s, "    }}").unwrap();
        writeln!(s, "}}").unwrap();

        WgslKernel {
            source: s,
            node: TensorId(root as u32),
            reads: reads.into_iter().map(|j| TensorId(j as u32)).collect(),
            points,
            count: points.div_ceil(b) * b,
            workgroup_size: b,
        }
    }
}
