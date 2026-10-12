//! Tiled WGSL schedule: a `bx × by` block of points per workgroup, one
//! point per thread, with shared reads staged chunk by chunk in workgroup
//! memory.
//!
//! Unlike the function-per-comp emitter in `tensor_wgsl`, this one fully
//! inlines the closure into a single loop nest and tracks every buffer
//! address as an affine form over the loop variables. A read that depends
//! on the reduce variable but on at most one of the tiled point
//! coordinates is shared along the other tile axis, so it is loaded
//! cooperatively once per chunk. Attention stages its k/v tiles, matmul
//! stages both its `a` and `b` tiles. Nothing here knows about either.
//!
//! Point coordinates `p{k-1}` (x) and `p{k-2}` (y) are tiled; any leading
//! point dims are uniform across the workgroup.

use std::collections::BTreeMap;
use std::fmt::Write;

use tang_expr::{ExprGraph, ExprId, Node};

use super::wgsl::{identity, lit, WgslKernel};
use super::{fixed, Comp, Op, TensorGraph, TensorId};

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

    fn wgsl(&self) -> String {
        let mut parts = vec![self.offset.to_string()];
        for (k, v) in &self.terms {
            parts.push(format!("{v} * i32({k})"));
        }
        format!("u32({})", parts.join(" + "))
    }
}

/// Reads of one buffer sharing every coefficient and differing only in
/// offset are one staged region.
#[derive(Clone, Debug, PartialEq, Eq)]
struct RegionKey {
    buffer: usize,
    /// coefficient on the x tile coord (`p{k-1}`), if any
    px: i64,
    /// coefficient on the y tile coord (`p{k-2}`), if any
    py: i64,
    /// coefficient on the reduce var `r0`
    red: i64,
    /// (inner loop var, coefficient)
    inner: Vec<(String, i64)>,
    /// (uniform var, coefficient): leading point dims
    uniform: Vec<(String, i64)>,
}

struct Region {
    key: RegionKey,
    lo: i64,
    hi: i64,
    /// extents of the inner vars, in `key.inner` order
    extents: Vec<usize>,
}

impl Region {
    fn span(&self) -> usize {
        (self.hi - self.lo + 1) as usize
    }
    fn inner_n(&self) -> usize {
        self.extents.iter().product()
    }
}

/// A per-thread, reduce-invariant read: loaded once before the reduce
/// loop into a private array indexed by the inner loop vars.
struct Hoist {
    buffer: usize,
    /// (var, coefficient) for every var in the address (point coords and
    /// inner vars; no `r0`), sorted
    coefs: Vec<(String, i64)>,
    lo: i64,
    hi: i64,
    /// (inner var, extent) in `coefs` order, inner vars only
    inner: Vec<(String, usize)>,
}

impl Hoist {
    fn span(&self) -> usize {
        (self.hi - self.lo + 1) as usize
    }
    fn len(&self) -> usize {
        self.inner.iter().map(|(_, e)| *e).product::<usize>() * self.span()
    }
}

/// An inlined comp whose point coordinates depend on the reduce var but
/// not on the lane (x) axis: computed cooperatively once per chunk into
/// workgroup memory, indexed by `(ly, t)`.
struct CompRegion {
    node: usize,
    /// point coords with `r0` and `py` as symbols
    coords: Vec<Sym>,
    width: usize,
    uses_py: bool,
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
    px: String,
    py: Option<String>,
    bx: usize,
    by: usize,
    chunk: usize,
    /// extents of inner loop vars
    extents: BTreeMap<String, usize>,
    regions: Vec<Region>,
    hoists: Vec<Hoist>,
    comp_regions: Vec<CompRegion>,
    /// inside a cooperative load loop: every read goes straight to its buffer
    in_load: bool,
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
        if self.in_load {
            return format!("b{j}[{}]", addr.wgsl());
        }
        let px = addr.terms.get(&self.px).copied().unwrap_or(0);
        let py = self
            .py
            .as_ref()
            .and_then(|n| addr.terms.get(n))
            .copied()
            .unwrap_or(0);
        let shared = addr.terms.contains_key("r0") && (px == 0 || py == 0);
        if !shared {
            let has_inner = addr.terms.keys().any(|k| !k.starts_with('p') && k != "r0");
            if !addr.terms.contains_key("r0") && has_inner {
                return self.hoisted_read(j, addr);
            }
            return format!("b{j}[{}]", addr.wgsl());
        }
        let is_tiled = |k: &str| k == self.px || self.py.as_deref() == Some(k);
        let inner: Vec<(String, i64)> = addr
            .terms
            .iter()
            .filter(|(k, _)| k.as_str() != "r0" && !is_tiled(k) && !k.starts_with('p'))
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        let uniform: Vec<(String, i64)> = addr
            .terms
            .iter()
            .filter(|(k, _)| k.starts_with('p') && !is_tiled(k))
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        let key = RegionKey {
            buffer: j,
            px,
            py,
            red: addr.terms["r0"],
            inner,
            uniform,
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
        // local index: (((ly * bx + lx) * chunk + t) * innerN + inner) * span + (off - lo)
        let mut local = String::from("0u");
        if r.key.py != 0 {
            local = format!("({local}) * {}u + ty", self.by);
        }
        if r.key.px != 0 {
            local = format!("({local}) * {}u + tx", self.bx);
        }
        local = format!("({local}) * {}u + (r0 - chunk)", self.chunk);
        for ((name, _), &ext) in r.key.inner.iter().zip(&r.extents) {
            local = format!("({local}) * {ext}u + {name}");
        }
        format!(
            "stage_{idx}[({local}) * {}u + {}u]",
            r.span(),
            addr.offset - r.lo
        )
    }

    /// A read that does not depend on the reduce var: served from a
    /// private array filled once per thread.
    fn hoisted_read(&mut self, j: usize, addr: Sym) -> String {
        let coefs: Vec<(String, i64)> = addr.terms.iter().map(|(k, v)| (k.clone(), *v)).collect();
        let idx = match self
            .hoists
            .iter()
            .position(|h| h.buffer == j && h.coefs == coefs)
        {
            Some(i) => {
                let h = &mut self.hoists[i];
                h.lo = h.lo.min(addr.offset);
                h.hi = h.hi.max(addr.offset);
                i
            }
            None => {
                let inner = coefs
                    .iter()
                    .filter(|(k, _)| !k.starts_with('p'))
                    .map(|(k, _)| (k.clone(), self.extents[k]))
                    .collect();
                self.hoists.push(Hoist {
                    buffer: j,
                    coefs,
                    lo: addr.offset,
                    hi: addr.offset,
                    inner,
                });
                self.hoists.len() - 1
            }
        };
        if self.collect {
            return String::new();
        }
        let h = &self.hoists[idx];
        let mut flat = String::from("0u");
        for (name, ext) in &h.inner {
            flat = format!("({flat}) * {ext}u + {name}");
        }
        format!(
            "hoist_{idx}[({flat}) * {}u + {}u]",
            h.span(),
            addr.offset - h.lo
        )
    }

    /// Value of node `j` at full-shape coordinates `coords`.
    fn expand(&mut self, out: &mut String, j: usize, coords: &[Sym]) -> Val {
        let node = &self.g.nodes[j];
        match &node.op {
            Op::Input(_) => {
                let addr = flat(coords, &fixed(&node.shape));
                Val::Expr(self.buffer_read(j, addr))
            }
            Op::Iota(d) => Val::Expr(format!("f32({})", coords[*d].wgsl())),
            Op::Comp(c) if self.reads.contains(&j) => {
                let k = c.points().len();
                let mut addr = flat(&coords[..k], &fixed(c.points())).scaled(c.width() as i64);
                addr.add(&flat(&coords[k..], &fixed(&c.out[k..])));
                Val::Expr(self.buffer_read(j, addr))
            }
            Op::Comp(c)
                if !self.in_load && self.is_workgroup_shared(&coords[..c.points().len()]) =>
            {
                let k = c.points().len();
                let w = c.width();
                let pts: Vec<Sym> = coords[..k].to_vec();
                let uses_py = self
                    .py
                    .as_ref()
                    .is_some_and(|py| pts.iter().any(|c| c.terms.contains_key(py)));
                let idx = match self
                    .comp_regions
                    .iter()
                    .position(|r| r.node == j && r.coords == pts)
                {
                    Some(i) => i,
                    None => {
                        self.comp_regions.push(CompRegion {
                            node: j,
                            coords: pts,
                            width: w,
                            uses_py,
                        });
                        self.comp_regions.len() - 1
                    }
                };
                if self.collect {
                    return Val::Expr(String::new());
                }
                let slot = flat(&coords[k..], &fixed(&c.out[k..]));
                let row = if uses_py { "ty" } else { "0u" };
                Val::Expr(format!(
                    "cstage_{idx}[(({row}) * {}u + (r0 - chunk)) * {w}u + {}]",
                    self.chunk,
                    slot.wgsl()
                ))
            }
            Op::Comp(c) => {
                let k = c.points().len();
                let n = self.fresh();
                let w = c.width();
                let red_vars: Vec<String> = (0..c.red.len()).map(|i| format!("q{n}_{i}")).collect();
                for (v, ext) in red_vars.iter().zip(fixed(&c.red)) {
                    self.extents.insert(v.clone(), ext);
                }
                let mut space: Vec<Sym> = coords[..k].to_vec();
                space.extend(red_vars.iter().map(|v| Sym::var(v)));

                let mut inner = String::new();
                let arg_vals = self.arg_values(&mut inner, c, &space);
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
                for (v, ext) in red_vars.iter().zip(fixed(&c.red)) {
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

    /// Expressions for each arg of `c` evaluated at iteration coords `space`.
    fn arg_values(&mut self, out: &mut String, c: &Comp, space: &[Sym]) -> Vec<String> {
        c.args
            .iter()
            .map(|a| {
                let src_shape = fixed(&self.g.nodes[a.src.n()].shape);
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
                let v = self.expand(out, a.src.n(), &cs);
                v.slot(&slot)
            })
            .collect()
    }

    /// Point coords that vary with the reduce var but not the lane axis.
    fn is_workgroup_shared(&self, pts: &[Sym]) -> bool {
        let on_r0 = pts.iter().any(|c| c.terms.contains_key("r0"));
        let on_px = pts.iter().any(|c| c.terms.contains_key(&self.px));
        let inner = pts
            .iter()
            .any(|c| c.terms.keys().any(|k| k != "r0" && !k.starts_with('p')));
        on_r0 && !on_px && !inner && self.bx > 1
    }

    fn tuple_rank(&self, j: usize) -> usize {
        match &self.g.nodes[j].op {
            Op::Comp(c) => c.tuple,
            Op::Input(_) | Op::Iota(_) => 0,
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
        let (bx, by, chunk) = self.nodes[root].tile;
        let (bx, by, chunk) = (bx as usize, by as usize, chunk as usize);
        let w = c.width();
        let pts_v = fixed(c.points());
        let pts = pts_v.as_slice();
        let k = pts.len();
        let r_total = c.red[0].n();
        let m = c.monoid.as_ref().expect("tile needs a reduction");
        let threads = bx * by;

        // Tile the last two non-unit point dims; everything else is
        // uniform across the workgroup.
        let mut non_unit = (0..k).rev().filter(|&d| pts[d] > 1);
        let dx = non_unit.next().unwrap_or(k - 1);
        let dy = if by > 1 { non_unit.next() } else { None };
        let others: Vec<usize> = (0..k).filter(|&d| d != dx && Some(d) != dy).collect();
        let nx = pts[dx];
        let ny = dy.map_or(1, |d| pts[d]);
        let tiles_x = nx.div_ceil(bx);
        let tiles_y = ny.div_ceil(by);
        let n_tiles = tiles_x * tiles_y * others.iter().map(|&d| pts[d]).product::<usize>();

        let mut ctx = Ctx {
            g: self,
            reads: &reads,
            px: format!("p{dx}"),
            py: dy.map(|d| format!("p{d}")),
            bx,
            by,
            chunk,
            extents: BTreeMap::new(),
            regions: Vec::new(),
            hoists: Vec::new(),
            comp_regions: Vec::new(),
            in_load: false,
            collect: true,
            counter: 0,
        };
        ctx.extents.insert("r0".into(), r_total);
        let space: Vec<Sym> = (0..k)
            .map(|d| Sym::var(&format!("p{d}")))
            .chain([Sym::var("r0")])
            .collect();
        let run = |ctx: &mut Ctx, body: &mut String| {
            let arg_vals = ctx.arg_values(body, c, &space);
            let vals = emit_scalar(body, &c.body, &c.body_outs, "t0_", &|v| {
                arg_vals[v as usize].clone()
            });
            emit_scalar(body, &m.combine, &m.outs, "c0_", &|v| {
                if (v as usize) < w {
                    format!("acc[{v}]")
                } else {
                    vals[v as usize - w].clone()
                }
            })
        };
        // Expand twice: first to discover staged regions, then for real.
        let mut body = String::new();
        run(&mut ctx, &mut body);
        ctx.collect = false;
        ctx.counter = 0;
        body.clear();
        let combined = run(&mut ctx, &mut body);

        let region_total = |r: &Region| {
            (if r.key.py != 0 { by } else { 1 })
                * (if r.key.px != 0 { bx } else { 1 })
                * chunk
                * r.inner_n()
                * r.span()
        };
        for (i, r) in ctx.regions.iter().enumerate() {
            writeln!(
                s,
                "var<workgroup> stage_{i}: array<f32, {}>;",
                region_total(r)
            )
            .unwrap();
        }
        // Cooperative loads for shared comps are expanded in the load
        // phase with r0 := chunk + t and py := tile_y*by + ly.
        let mut comp_loads: Vec<(usize, String)> = Vec::new();
        for i in 0..ctx.comp_regions.len() {
            let r = &ctx.comp_regions[i];
            let rows = if r.uses_py { by } else { 1 };
            writeln!(
                s,
                "var<workgroup> cstage_{i}: array<f32, {}>;",
                rows * chunk * r.width
            )
            .unwrap();
            let mut sub = |c: &Sym| {
                let mut out = Sym {
                    offset: c.offset,
                    ..Default::default()
                };
                for (k, v) in &c.terms {
                    if k == "r0" {
                        out.add(&Sym::var("chunk").scaled(*v));
                        out.add(&Sym::var("t").scaled(*v));
                    } else if Some(k) == ctx.py.as_ref() {
                        out.add(&Sym::var("gy").scaled(*v));
                    } else {
                        out.add(&Sym::var(k).scaled(*v));
                    }
                }
                out
            };
            let coords: Vec<Sym> = r.coords.iter().map(&mut sub).collect();
            let node = r.node;
            let width = r.width;
            let mut code = String::new();
            ctx.in_load = true;
            let saved = ctx.counter;
            ctx.counter = 1000 * (i + 1);
            let v = ctx.expand(&mut code, node, &coords);
            ctx.counter = saved;
            ctx.in_load = false;
            for slot in 0..width {
                let sl = Sym {
                    offset: slot as i64,
                    ..Default::default()
                };
                writeln!(
                    code,
                    "cstage_{i}[((ly) * {chunk}u + t) * {width}u + {slot}u] = {};",
                    v.slot(&sl)
                )
                .unwrap();
            }
            comp_loads.push((rows, code));
        }

        writeln!(s, "@compute @workgroup_size({threads})").unwrap();
        writeln!(
            s,
            "fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{"
        )
        .unwrap();
        writeln!(s, "    let tid = lid.x;").unwrap();
        writeln!(s, "    let tx = tid % {bx}u;").unwrap();
        writeln!(s, "    let ty = tid / {bx}u;").unwrap();
        writeln!(s, "    let tile = wid.x + wid.y * 65535u;").unwrap();
        writeln!(s, "    if (tile >= {n_tiles}u) {{ return; }}").unwrap();
        writeln!(s, "    let tile_x = tile % {tiles_x}u;").unwrap();
        writeln!(s, "    let tile_y = (tile / {tiles_x}u) % {tiles_y}u;").unwrap();
        writeln!(s, "    var lead = tile / {}u;", tiles_x * tiles_y).unwrap();
        for &d in others.iter().rev() {
            writeln!(s, "    let p{d} = lead % {}u;", pts[d]).unwrap();
            writeln!(s, "    lead = lead / {}u;", pts[d]).unwrap();
        }
        writeln!(s, "    let x_raw = tile_x * {bx}u + tx;").unwrap();
        writeln!(s, "    let y_raw = tile_y * {by}u + ty;").unwrap();
        writeln!(s, "    let is_live = x_raw < {nx}u && y_raw < {ny}u;").unwrap();
        writeln!(s, "    let p{dx} = min(x_raw, {}u);", nx - 1).unwrap();
        if let Some(dy) = dy {
            writeln!(s, "    let p{dy} = min(y_raw, {}u);", ny - 1).unwrap();
        }
        for (i, h) in ctx.hoists.iter().enumerate() {
            writeln!(s, "    var hoist_{i}: array<f32, {}>;", h.len()).unwrap();
            writeln!(s, "    {{").unwrap();
            let mut flat = String::from("0u");
            for (name, ext) in &h.inner {
                writeln!(
                    s,
                    "    for (var {name}: u32 = 0u; {name} < {ext}u; {name} = {name} + 1u) {{"
                )
                .unwrap();
                flat = format!("({flat}) * {ext}u + {name}");
            }
            let span = h.span();
            let addr = h
                .coefs
                .iter()
                .map(|(k, c)| format!(" + {c} * i32({k})"))
                .collect::<String>();
            writeln!(s, "    for (var o: u32 = 0u; o < {span}u; o = o + 1u) {{").unwrap();
            writeln!(
                s,
                "        hoist_{i}[({flat}) * {span}u + o] = b{}[u32({}{addr} + i32(o))];",
                h.buffer, h.lo
            )
            .unwrap();
            writeln!(s, "    }}").unwrap();
            for _ in &h.inner {
                writeln!(s, "    }}").unwrap();
            }
            writeln!(s, "    }}").unwrap();
        }
        writeln!(s, "    var acc = array<f32, {w}>({});", identity(m)).unwrap();
        writeln!(
            s,
            "    for (var chunk: u32 = 0u; chunk < {r_total}u; chunk = chunk + {chunk}u) {{"
        )
        .unwrap();
        for (i, r) in ctx.regions.iter().enumerate() {
            let total = region_total(r);
            let span = r.span();
            writeln!(
                s,
                "        for (var idx: u32 = tid; idx < {total}u; idx = idx + {threads}u) {{"
            )
            .unwrap();
            writeln!(s, "            let o = idx % {span}u;").unwrap();
            writeln!(s, "            var rem = idx / {span}u;").unwrap();
            let mut addr = format!("{}", r.lo);
            for ((name, coef), &ext) in r.key.inner.iter().zip(&r.extents).rev() {
                writeln!(s, "            let {name} = rem % {ext}u;").unwrap();
                writeln!(s, "            rem = rem / {ext}u;").unwrap();
                addr.push_str(&format!(" + {coef} * i32({name})"));
            }
            writeln!(s, "            let t = rem % {chunk}u;").unwrap();
            writeln!(s, "            rem = rem / {chunk}u;").unwrap();
            addr.push_str(&format!(" + {} * i32(chunk + t) + i32(o)", r.key.red));
            let mut live = format!("chunk + t < {r_total}u");
            if r.key.px != 0 {
                writeln!(s, "            let lx = rem % {bx}u;").unwrap();
                writeln!(s, "            rem = rem / {bx}u;").unwrap();
                writeln!(s, "            let gx = tile_x * {bx}u + lx;").unwrap();
                addr.push_str(&format!(" + {} * i32(gx)", r.key.px));
                live.push_str(&format!(" && gx < {nx}u"));
            }
            if r.key.py != 0 {
                writeln!(s, "            let ly = rem % {by}u;").unwrap();
                writeln!(s, "            let gy = tile_y * {by}u + ly;").unwrap();
                addr.push_str(&format!(" + {} * i32(gy)", r.key.py));
                live.push_str(&format!(" && gy < {ny}u"));
            }
            for (name, coef) in &r.key.uniform {
                addr.push_str(&format!(" + {coef} * i32({name})"));
            }
            writeln!(s, "            if ({live}) {{").unwrap();
            writeln!(
                s,
                "                stage_{i}[idx] = b{}[u32({addr})];",
                r.key.buffer
            )
            .unwrap();
            writeln!(s, "            }}").unwrap();
            writeln!(s, "        }}").unwrap();
        }
        for (i, (rows, code)) in comp_loads.iter().enumerate() {
            let total = rows * chunk;
            writeln!(
                s,
                "        for (var idx: u32 = tid; idx < {total}u; idx = idx + {threads}u) {{"
            )
            .unwrap();
            writeln!(s, "            let t = idx % {chunk}u;").unwrap();
            writeln!(s, "            let ly = idx / {chunk}u;").unwrap();
            writeln!(
                s,
                "            let gy = min(tile_y * {by}u + ly, {}u);",
                ny - 1
            )
            .unwrap();
            writeln!(s, "            if (chunk + t < {r_total}u) {{").unwrap();
            for line in code.lines() {
                writeln!(s, "                {line}").unwrap();
            }
            writeln!(s, "            }}").unwrap();
            writeln!(s, "        }}").unwrap();
            let _ = i;
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
        let point_flat = {
            let mut acc = String::from("0u");
            for (d, n) in pts.iter().enumerate() {
                acc = format!("({acc}) * {n}u + p{d}");
            }
            acc
        };
        writeln!(s, "        let g = {point_flat};").unwrap();
        for slot in 0..w {
            writeln!(s, "        out[g * {w}u + {slot}u] = acc[{slot}];").unwrap();
        }
        writeln!(s, "    }}").unwrap();
        writeln!(s, "}}").unwrap();

        WgslKernel {
            source: s,
            node: TensorId(root as u32),
            reads: reads.into_iter().map(|j| TensorId(j as u32)).collect(),
            points: pts.iter().product::<usize>() as u32,
            count: (n_tiles * threads) as u32,
            workgroup_size: threads as u32,
        }
    }
}
