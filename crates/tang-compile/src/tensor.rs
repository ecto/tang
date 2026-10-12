//! Tensor-level IR: views + one compute op.
//!
//! Everything above the scalar [`tang_expr`] graph is built from two ideas:
//!
//! - A [`View`] is an affine map from a comprehension's iteration space to a
//!   source tensor's coordinates. Broadcast, transpose, slice and stride are
//!   all views. They compose and never touch memory.
//! - A [`Comp`] (comprehension) is the only compute op:
//!
//!   ```text
//!   out[i…] = ⊕_{k…} body(x₀[view₀(i,k)], x₁[view₁(i,k)], …)
//!   ```
//!
//!   `body` is a scalar `tang_expr` graph and `⊕` is a [`Monoid`] whose
//!   combine is also a scalar graph. A map is a comp with no reduce dims;
//!   sum/max are comps with a monoid; matmul is `Σ_k a[i,k]·b[k,j]`.
//!
//! A comp may produce several values per point: its trailing `tuple` out
//! dims are filled by the body rather than iterated. That is how monoids
//! carry tuple state — online softmax is `(max, sum, acc…)` — while
//! consumers still address the result with ordinary views.
//!
//! Passes:
//! - [`TensorGraph::substitute_maps`] — the single fusion rule. A map is a
//!   pure function of its inputs, so it is inlined into every consumer's
//!   body and its views are composed with the consumer's.
//! - [`TensorGraph::stabilize_softmax`] — rewrites `max_j s` + `Σ_j g·exp(s−max)`
//!   into one reduction with the online-softmax monoid.
//! - [`TensorGraph::schedule_inline`] — a schedule decision: a single-use
//!   reduction is recomputed inside its consumer when that costs no more
//!   than materializing it.
//!
//! Scope of this sketch: fixed shapes, f64, a reference interpreter that
//! respects materialization decisions, no autodiff/codegen/sharding yet.

use std::collections::HashMap;

use tang::Scalar;
use tang_expr::{trace, ExprGraph, ExprId, Node};

#[path = "tensor_grad.rs"]
mod grad;
#[path = "tensor_tile.rs"]
mod tile;
#[path = "tensor_wgsl.rs"]
pub mod wgsl;

// ---------------------------------------------------------------------------
// Views
// ---------------------------------------------------------------------------

/// `offset + Σ coefs[d] · idx[d]` over an iteration space.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Form {
    pub coefs: Vec<i64>,
    pub offset: i64,
}

impl Form {
    fn zero(rank: usize) -> Self {
        Self {
            coefs: vec![0; rank],
            offset: 0,
        }
    }

    fn unit(rank: usize, d: usize) -> Self {
        let mut f = Self::zero(rank);
        f.coefs[d] = 1;
        f
    }

    fn constant(rank: usize, c: i64) -> Self {
        let mut f = Self::zero(rank);
        f.offset = c;
        f
    }

    fn eval(&self, idx: &[usize]) -> i64 {
        self.offset
            + self
                .coefs
                .iter()
                .zip(idx)
                .map(|(c, i)| c * *i as i64)
                .sum::<i64>()
    }

    /// Substitute each coordinate `d` of this form by `inner[d]`.
    fn compose(&self, inner: &[Form], rank: usize) -> Form {
        let mut out = Form::zero(rank);
        out.offset = self.offset;
        for (c, f) in self.coefs.iter().zip(inner) {
            out.offset += c * f.offset;
            for (o, i) in out.coefs.iter_mut().zip(&f.coefs) {
                *o += c * i;
            }
        }
        out
    }

    fn range(&self, space: &[usize]) -> (i64, i64) {
        let (mut lo, mut hi) = (self.offset, self.offset);
        for (c, n) in self.coefs.iter().zip(space) {
            let extent = c * (*n as i64 - 1);
            if extent < 0 {
                lo += extent
            } else {
                hi += extent
            }
        }
        (lo, hi)
    }

    /// `Some(d)` if this form is exactly coordinate `d`.
    fn as_unit(&self) -> Option<usize> {
        if self.offset != 0 || self.coefs.iter().any(|c| *c != 0 && *c != 1) {
            return None;
        }
        let mut it = self.coefs.iter().enumerate().filter(|(_, c)| **c == 1);
        match (it.next(), it.next()) {
            (Some((d, _)), None) => Some(d),
            _ => None,
        }
    }
}

/// One form per source dimension, over the consumer's iteration space.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct View {
    pub forms: Vec<Form>,
}

impl View {
    /// Source dim `d` reads iteration coordinate `picks[d]`, or is
    /// broadcast (constant 0) when `None`.
    pub fn pick(rank: usize, picks: &[Option<usize>]) -> Self {
        let forms = picks
            .iter()
            .map(|p| p.map_or_else(|| Form::zero(rank), |d| Form::unit(rank, d)))
            .collect();
        Self { forms }
    }

    /// Identity over the first `src_rank` coordinates of a `rank`-dim space.
    pub fn identity(rank: usize, src_rank: usize) -> Self {
        Self::pick(rank, &(0..src_rank).map(Some).collect::<Vec<_>>())
    }

    fn rank(&self) -> usize {
        self.forms.first().map_or(0, |f| f.coefs.len())
    }

    fn coords(&self, idx: &[usize]) -> Vec<usize> {
        self.forms.iter().map(|f| f.eval(idx) as usize).collect()
    }

    /// `self ∘ inner`: this view is over a space whose coordinates are
    /// given by `inner` over a `rank`-dim space.
    fn compose(&self, inner: &View, rank: usize) -> View {
        View {
            forms: self
                .forms
                .iter()
                .map(|f| f.compose(&inner.forms, rank))
                .collect(),
        }
    }

    /// Drop dependence on size-1 coordinates so equal views compare equal.
    fn normalized(&self, space: &[Dim]) -> View {
        let mut v = self.clone();
        for f in &mut v.forms {
            for (c, &n) in f.coefs.iter_mut().zip(space) {
                if n == Dim::Fixed(1) {
                    *c = 0;
                }
            }
        }
        v
    }

    fn depends_on(&self, d: usize) -> bool {
        self.forms.iter().any(|f| f.coefs[d] != 0)
    }
}

// ---------------------------------------------------------------------------
// Monoids
// ---------------------------------------------------------------------------

/// Reduction state of `width` values. `combine` has vars `0..w` for the
/// accumulator and `w..2w` for the incoming element, and `w` outputs.
#[derive(Clone)]
pub struct Monoid {
    pub width: usize,
    pub identity: Vec<f64>,
    pub combine: ExprGraph,
    pub outs: Vec<ExprId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MonoidKind {
    Sum,
    Max,
    OnlineSoftmax,
    Other,
}

impl Monoid {
    fn build(
        width: usize,
        identity: Vec<f64>,
        f: impl FnOnce(&[ExprId], &[ExprId]) -> Vec<ExprId>,
    ) -> Self {
        let (combine, outs) = trace(|| {
            let acc: Vec<ExprId> = (0..width).map(|i| ExprId::var(i as u16)).collect();
            let el: Vec<ExprId> = (0..width)
                .map(|i| ExprId::var((width + i) as u16))
                .collect();
            f(&acc, &el)
        });
        Self {
            width,
            identity,
            combine,
            outs,
        }
    }

    pub fn sum() -> Self {
        Self::build(1, vec![0.0], |a, e| vec![a[0] + e[0]])
    }

    pub fn max() -> Self {
        Self::build(1, vec![f64::NEG_INFINITY], |a, e| vec![max(a[0], e[0])])
    }

    /// State `(m, acc₁…accₖ)`, element `(s, g₁…gₖ)`:
    /// `m' = max(m, s)`, `accᵢ' = accᵢ·exp(m−m') + gᵢ·exp(s−m')`.
    pub fn online_softmax(k: usize) -> Self {
        let mut id = vec![0.0; k + 1];
        id[0] = f64::NEG_INFINITY;
        Self::build(k + 1, id, |a, e| {
            let m = max(a[0], e[0]);
            let sa = (a[0] - m).exp();
            let se = (e[0] - m).exp();
            let mut out = vec![m];
            out.extend((1..=k).map(|i| a[i] * sa + e[i] * se));
            out
        })
    }

    pub fn kind(&self) -> MonoidKind {
        let n = self.combine.nodes_slice();
        if self.width == 1 {
            return match n[self.outs[0].index() as usize] {
                Node::Add(a, b) if is_var(n, a, 0) && is_var(n, b, 1) => MonoidKind::Sum,
                Node::Select(c, a, b)
                    if is_var(n, a, 0)
                        && is_var(n, b, 1)
                        && matches!(n[c.index() as usize], Node::Add(..)) =>
                {
                    MonoidKind::Max
                }
                _ => MonoidKind::Other,
            };
        }
        if self.identity[0] == f64::NEG_INFINITY {
            MonoidKind::OnlineSoftmax
        } else {
            MonoidKind::Other
        }
    }
}

/// `max(a, b)` via select, so it is exact at `-inf` (the `Scalar` impl
/// uses `sqrt((a-b)²)`, which is NaN there).
fn max(a: ExprId, b: ExprId) -> ExprId {
    let d = a - b;
    tang_expr::with_graph(|g| g.select(d, a, b))
}

fn is_var(nodes: &[Node], id: ExprId, v: u16) -> bool {
    nodes[id.index() as usize] == Node::Var(v)
}

// ---------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TensorId(u32);

impl TensorId {
    fn n(self) -> usize {
        self.0 as usize
    }

    /// Index into [`TensorGraph::nodes`].
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Arg {
    pub src: TensorId,
    pub view: View,
}

#[derive(Clone)]
pub struct Comp {
    /// Output shape. The last `tuple` dims are produced per point.
    pub out: Vec<Dim>,
    /// Trailing out dims filled by the body (0 for ordinary comps).
    pub tuple: usize,
    pub red: Vec<Dim>,
    pub args: Vec<Arg>,
    pub body: ExprGraph,
    pub body_outs: Vec<ExprId>,
    pub monoid: Option<Monoid>,
}

impl Comp {
    fn width(&self) -> usize {
        self.body_outs.len()
    }

    /// Iterated out dims (excluding tuple dims).
    fn points(&self) -> &[Dim] {
        &self.out[..self.out.len() - self.tuple]
    }

    fn rank(&self) -> usize {
        self.points().len() + self.red.len()
    }

    fn space(&self) -> Vec<Dim> {
        self.points().iter().chain(&self.red).copied().collect()
    }
}

#[derive(Clone)]
#[allow(clippy::large_enum_variant)]
pub enum Op {
    Input(usize),
    Comp(Comp),
}

#[derive(Clone)]
pub struct TNode {
    pub op: Op,
    pub shape: Vec<Dim>,
    /// Schedule: recompute inside consumers instead of writing a buffer.
    pub inline: bool,
    /// Schedule: split the reduction across a workgroup of this many
    /// threads, tree-combined with the monoid (0 = one thread per point).
    pub split: u32,
    /// Schedule: `(tile over last point dim, tile over second-to-last,
    /// reduce chunk)`. Each thread owns one point; reads shared along a
    /// tile axis are staged chunk by chunk in workgroup memory.
    pub tile: (u32, u32, u32),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShapeError(pub String);

impl std::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "shape error: {}", self.0)
    }
}

impl std::error::Error for ShapeError {}

pub(super) fn err<T>(msg: impl Into<String>) -> Result<T, ShapeError> {
    Err(ShapeError(msg.into()))
}

/// A tensor dimension: a fixed size or a named symbol (`"seq"`, `"batch"`).
///
/// Symbolic dims flow through construction, rewriting and autodiff.
/// Anything that needs numbers — the interpreter, schedules, lowering —
/// works on a graph after [`TensorGraph::bind`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dim {
    Fixed(usize),
    Sym(&'static str),
}

impl Dim {
    pub fn fixed(self) -> Option<usize> {
        match self {
            Dim::Fixed(n) => Some(n),
            Dim::Sym(_) => None,
        }
    }

    /// The size, panicking on an unbound symbol.
    pub fn n(self) -> usize {
        match self {
            Dim::Fixed(n) => n,
            Dim::Sym(name) => panic!("dim `{name}` is unbound; call TensorGraph::bind first"),
        }
    }
}

impl From<usize> for Dim {
    fn from(n: usize) -> Self {
        Dim::Fixed(n)
    }
}

impl PartialEq<usize> for Dim {
    fn eq(&self, other: &usize) -> bool {
        *self == Dim::Fixed(*other)
    }
}

impl std::fmt::Display for Dim {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Dim::Fixed(n) => write!(f, "{n}"),
            Dim::Sym(s) => write!(f, "{s}"),
        }
    }
}

/// Concrete sizes of a bound shape.
pub fn fixed(shape: &[Dim]) -> Vec<usize> {
    shape.iter().map(|d| d.n()).collect()
}

/// Number of elements of a bound shape.
pub fn numel(shape: &[Dim]) -> usize {
    shape.iter().map(|d| d.n()).product()
}

fn dims<D: Into<Dim> + Copy>(d: &[D]) -> Vec<Dim> {
    d.iter().map(|&x| x.into()).collect()
}

#[derive(Clone, Default)]
pub struct TensorGraph {
    nodes: Vec<TNode>,
    n_inputs: usize,
}

impl TensorGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn nodes(&self) -> &[TNode] {
        &self.nodes
    }

    /// Set a node's thread-mapping schedule (`split`, `tile`), e.g. from
    /// an autotuner. Materialization (`inline`) is left alone: it changes
    /// which kernels exist, not how one runs.
    pub fn set_schedule(&mut self, t: TensorId, split: u32, tile: (u32, u32, u32)) {
        self.nodes[t.n()].split = split;
        self.nodes[t.n()].tile = tile;
    }

    /// Thread mappings worth trying for a materialized comp on a GPU:
    /// one thread per point, a workgroup split (reductions), and rows ×
    /// lanes tiles (single-axis reductions with a lane dim).
    pub fn schedule_candidates(&self, t: TensorId) -> Vec<(u32, (u32, u32, u32))> {
        let mut out = vec![(0, (0, 0, 0))];
        let Some(c) = self.comp_at(t.n()) else {
            return out;
        };
        if c.monoid.is_none() {
            return out;
        }
        out.push((256, (0, 0, 0)));
        if c.red.len() == 1 {
            let pts = fixed(c.points());
            let lanes = pts.iter().rev().find(|&&n| n > 1).copied().unwrap_or(1);
            let rows = pts.iter().rev().filter(|&&n| n > 1).nth(1).is_some();
            if (16..=128).contains(&lanes) && rows {
                out.push((0, (lanes as u32, (256 / lanes) as u32, 32)));
            }
            if lanes >= 64 && !rows {
                out.push((0, (lanes.min(256) as u32, 1, 32)));
            }
            if lanes >= 16 && rows {
                // the classic shared-memory 2-D tile (matmul-shaped comps)
                out.push((0, (16, 16, 32)));
            }
        }
        out
    }

    pub fn shape(&self, t: TensorId) -> &[Dim] {
        &self.nodes[t.n()].shape
    }

    pub fn input<D: Into<Dim> + Copy>(&mut self, shape: &[D]) -> TensorId {
        let i = self.n_inputs;
        self.n_inputs += 1;
        self.nodes.push(TNode {
            op: Op::Input(i),
            shape: dims(shape),
            inline: false,
            split: 0,
            tile: (0, 0, 0),
        });
        TensorId(self.nodes.len() as u32 - 1)
    }

    /// The general op. `f` receives one scalar per arg and returns the
    /// per-point values (`monoid.width` of them when reducing).
    ///
    /// `f` runs inside a fresh `tang_expr` trace: every literal
    /// (`ExprId::from_f64`) and operation must be created *inside* the
    /// closure. An `ExprId` made outside refers to some other graph and
    /// silently means the wrong node.
    pub fn comp<D: Into<Dim> + Copy>(
        &mut self,
        out: &[D],
        red: &[D],
        args: Vec<Arg>,
        monoid: Option<Monoid>,
        f: impl FnOnce(&[ExprId]) -> Vec<ExprId>,
    ) -> Result<TensorId, ShapeError> {
        let (body, body_outs) = trace(|| {
            let vars: Vec<ExprId> = (0..args.len()).map(|i| ExprId::var(i as u16)).collect();
            f(&vars)
        });
        self.push_comp(Comp {
            out: dims(out),
            tuple: 0,
            red: dims(red),
            args,
            body,
            body_outs,
            monoid,
        })
    }

    fn push_comp(&mut self, comp: Comp) -> Result<TensorId, ShapeError> {
        let space = comp.space();
        for (i, a) in comp.args.iter().enumerate() {
            let src = self.shape(a.src);
            if a.view.forms.len() != src.len() || a.view.rank() != space.len() {
                return err(format!(
                    "arg {i}: view has {} forms of rank {}, source {src:?} in space {space:?}",
                    a.view.forms.len(),
                    a.view.rank()
                ));
            }
            let bound: Option<Vec<usize>> = space.iter().map(|d| d.fixed()).collect();
            if let Some(space) = bound {
                for (f, &n) in a.view.forms.iter().zip(src) {
                    let (lo, hi) = f.range(&space);
                    if lo < 0 || n.fixed().is_some_and(|n| hi >= n as i64) {
                        return err(format!("arg {i}: view reaches {lo}..={hi}, dim is {n}"));
                    }
                }
            }
        }
        let width = numel(&comp.out[comp.out.len() - comp.tuple..]); // tuple dims are always fixed
        if comp.width() != width {
            return err(format!(
                "body produces {} values, tuple dims need {width}",
                comp.width()
            ));
        }
        match &comp.monoid {
            Some(m) if m.width != width => {
                return err(format!(
                    "monoid width {} but body produces {width}",
                    m.width
                ))
            }
            None if !comp.red.is_empty() => return err("reduce dims without a monoid"),
            _ => {}
        }
        self.nodes.push(TNode {
            shape: comp.out.clone(),
            inline: false,
            split: 0,
            tile: (0, 0, 0),
            op: Op::Comp(comp),
        });
        Ok(TensorId(self.nodes.len() as u32 - 1))
    }

    // -- conveniences, all sugar over `comp` ---------------------------------

    /// Elementwise over equally-shaped args.
    pub fn map(
        &mut self,
        args: &[TensorId],
        f: impl FnOnce(&[ExprId]) -> ExprId,
    ) -> Result<TensorId, ShapeError> {
        let Some(&first) = args.first() else {
            return err("map needs at least one arg");
        };
        let shape = self.shape(first).to_vec();
        if let Some(a) = args.iter().find(|a| self.shape(**a) != shape) {
            return err(format!(
                "map args differ: {shape:?} vs {:?}",
                self.shape(*a)
            ));
        }
        let r = shape.len();
        let args = args
            .iter()
            .map(|&src| Arg {
                src,
                view: View::identity(r, r),
            })
            .collect();
        self.comp(&shape, &[], args, None, |v| vec![f(v)])
    }

    /// Reduce `axis` of `x` with `m`; the axis is kept as size 1 so the
    /// result broadcasts back against `x`.
    pub fn reduce(&mut self, x: TensorId, axis: usize, m: Monoid) -> Result<TensorId, ShapeError> {
        let shape = self.shape(x).to_vec();
        if axis >= shape.len() {
            return err(format!("axis {axis} out of range for {shape:?}"));
        }
        let mut out = shape.clone();
        out[axis] = Dim::Fixed(1);
        let red = [shape[axis]];
        let r = shape.len();
        // iteration space = out dims ++ [axis]; source dim `axis` reads the red coord
        let picks: Vec<Option<usize>> = (0..r)
            .map(|d| Some(if d == axis { r } else { d }))
            .collect();
        let args = vec![Arg {
            src: x,
            view: View::pick(r + 1, &picks),
        }];
        self.comp(&out, &red, args, Some(m), |v| vec![v[0]])
    }

    /// Broadcast `x` (with size-1 dims) up to `shape`, as a view-only map.
    pub fn broadcast<D: Into<Dim> + Copy>(
        &mut self,
        x: TensorId,
        shape: &[D],
    ) -> Result<TensorId, ShapeError> {
        let shape = &dims(shape);
        let src = self.shape(x).to_vec();
        if src.len() != shape.len() {
            return err(format!("broadcast {src:?} -> {shape:?}: rank differs"));
        }
        let r = shape.len();
        let picks: Vec<Option<usize>> = src
            .iter()
            .zip(shape)
            .enumerate()
            .map(|(d, (&s, &t))| match (s == t, s == Dim::Fixed(1)) {
                (true, _) => Ok(Some(d)),
                (false, true) => Ok(None),
                _ => err(format!("broadcast {src:?} -> {shape:?}")),
            })
            .collect::<Result<_, _>>()?;
        let args = vec![Arg {
            src: x,
            view: View::pick(r, &picks),
        }];
        self.comp(shape, &[], args, None, |v| vec![v[0]])
    }

    pub fn transpose(&mut self, x: TensorId, perm: &[usize]) -> Result<TensorId, ShapeError> {
        let src = self.shape(x).to_vec();
        let r = src.len();
        let mut inv = vec![None; r];
        for (o, &p) in perm.iter().enumerate() {
            if p >= r || inv[p].replace(o).is_some() {
                return err(format!("transpose {src:?} by {perm:?}"));
            }
        }
        if perm.len() != r {
            return err(format!("transpose {src:?} by {perm:?}"));
        }
        let shape: Vec<Dim> = perm.iter().map(|&p| src[p]).collect();
        let args = vec![Arg {
            src: x,
            view: View::pick(r, &inv),
        }];
        self.comp(&shape, &[], args, None, |v| vec![v[0]])
    }

    /// `[m,k] x [k,n] -> [m,n]`.
    pub fn matmul(&mut self, a: TensorId, b: TensorId) -> Result<TensorId, ShapeError> {
        let (sa, sb) = (self.shape(a).to_vec(), self.shape(b).to_vec());
        if sa.len() != 2 || sb.len() != 2 || sa[1] != sb[0] {
            return err(format!("matmul {sa:?} x {sb:?}"));
        }
        let args = vec![
            Arg {
                src: a,
                view: View::pick(3, &[Some(0), Some(2)]),
            },
            Arg {
                src: b,
                view: View::pick(3, &[Some(2), Some(1)]),
            },
        ];
        self.comp(&[sa[0], sb[1]], &[sa[1]], args, Some(Monoid::sum()), |v| {
            vec![v[0] * v[1]]
        })
    }

    // -- analysis -----------------------------------------------------------

    fn comp_at(&self, i: usize) -> Option<&Comp> {
        match &self.nodes[i].op {
            Op::Comp(c) => Some(c),
            Op::Input(_) => None,
        }
    }

    fn live(&self, outputs: &[TensorId]) -> Vec<bool> {
        let mut live = vec![false; self.nodes.len()];
        let mut stack: Vec<usize> = outputs.iter().map(|t| t.n()).collect();
        while let Some(i) = stack.pop() {
            if std::mem::replace(&mut live[i], true) {
                continue;
            }
            if let Some(c) = self.comp_at(i) {
                stack.extend(c.args.iter().map(|a| a.src.n()));
            }
        }
        live
    }

    /// Live consumers per node (an output counts once).
    fn use_counts(&self, outputs: &[TensorId], live: &[bool]) -> Vec<usize> {
        let mut uses = vec![0; self.nodes.len()];
        for i in (0..self.nodes.len()).filter(|&i| live[i]) {
            if let Some(c) = self.comp_at(i) {
                for a in &c.args {
                    uses[a.src.n()] += 1;
                }
            }
        }
        for o in outputs {
            uses[o.n()] += 1;
        }
        uses
    }

    /// Live, non-inlined comps: the buffers a backend would allocate.
    pub fn materialized(&self, outputs: &[TensorId]) -> Vec<(TensorId, Vec<Dim>)> {
        let live = self.live(outputs);
        (0..self.nodes.len())
            .filter(|&i| live[i] && !self.nodes[i].inline && self.comp_at(i).is_some())
            .map(|i| (TensorId(i as u32), self.nodes[i].shape.clone()))
            .collect()
    }

    pub fn count_comps(&self, outputs: &[TensorId]) -> usize {
        let live = self.live(outputs);
        (0..self.nodes.len())
            .filter(|&i| live[i] && self.comp_at(i).is_some())
            .count()
    }

    /// Monoid kinds of all live reductions.
    pub fn reduction_kinds(&self, outputs: &[TensorId]) -> Vec<MonoidKind> {
        let live = self.live(outputs);
        (0..self.nodes.len())
            .filter(|&i| live[i])
            .filter_map(|i| self.comp_at(i)?.monoid.as_ref().map(Monoid::kind))
            .collect()
    }

    // -- passes ---------------------------------------------------------------

    /// Fuse every map (comp without reduction) into its consumers by
    /// splicing its body and composing views. Dead nodes are dropped.
    pub fn substitute_maps(&self, outputs: &[TensorId]) -> (TensorGraph, Vec<TensorId>) {
        let n = self.nodes.len();
        let live = self.live(outputs);
        let is_output = |i: usize| outputs.iter().any(|o| o.n() == i);
        let inline: Vec<bool> = (0..n)
            .map(|i| {
                self.comp_at(i)
                    .is_some_and(|c| c.red.is_empty() && c.tuple == 0)
                    && !is_output(i)
            })
            .collect();

        let mut ng = TensorGraph {
            nodes: Vec::new(),
            n_inputs: self.n_inputs,
        };
        let mut remap: Vec<Option<usize>> = vec![None; n];
        for i in (0..n).filter(|&i| live[i] && !inline[i]) {
            let node = &self.nodes[i];
            let op = match &node.op {
                Op::Input(k) => Op::Input(*k),
                Op::Comp(c) => {
                    let rank = c.rank();
                    let mut body = ExprGraph::new();
                    let mut args: Vec<Arg> = Vec::new();
                    let vars: Vec<ExprId> = c
                        .args
                        .iter()
                        .map(|a| self.splice(a, rank, &inline, &remap, &mut body, &mut args))
                        .collect();
                    let body_outs = replay(&c.body, &c.body_outs, &vars, &mut body);
                    Op::Comp(Comp {
                        out: c.out.clone(),
                        tuple: c.tuple,
                        red: c.red.clone(),
                        args,
                        body,
                        body_outs,
                        monoid: c.monoid.clone(),
                    })
                }
            };
            ng.nodes.push(TNode {
                op,
                shape: node.shape.clone(),
                inline: node.inline,
                split: node.split,
                tile: node.tile,
            });
            remap[i] = Some(ng.nodes.len() - 1);
        }
        let outs = outputs
            .iter()
            .map(|o| TensorId(remap[o.n()].unwrap() as u32))
            .collect();
        (ng, outs)
    }

    /// Expression for `arg` inside a consumer of iteration rank `rank`:
    /// either a leaf var or the producer map's body, recursively.
    fn splice(
        &self,
        arg: &Arg,
        rank: usize,
        inline: &[bool],
        remap: &[Option<usize>],
        body: &mut ExprGraph,
        args: &mut Vec<Arg>,
    ) -> ExprId {
        let p = arg.src.n();
        if !inline[p] {
            let leaf = Arg {
                src: TensorId(remap[p].unwrap() as u32),
                view: arg.view.clone(),
            };
            return body.var(intern_arg(args, leaf) as u16);
        }
        let c = self.comp_at(p).unwrap();
        let vars: Vec<ExprId> = c
            .args
            .iter()
            .map(|a| {
                let composed = Arg {
                    src: a.src,
                    view: a.view.compose(&arg.view, rank),
                };
                self.splice(&composed, rank, inline, remap, body, args)
            })
            .collect();
        replay(&c.body, &c.body_outs, &vars, body)[0]
    }

    /// Canonical form of the expression `out` of `body` over `args`, with
    /// views over `space`: leaves renumbered in order of first use and
    /// size-1 coordinates dropped, so equal computations compare equal.
    fn canon(body: &ExprGraph, out: ExprId, args: &[Arg], space: &[Dim]) -> Canon {
        let mut leaves: Vec<Arg> = Vec::new();
        let mut g = ExprGraph::new();
        let vars: Vec<ExprId> = args
            .iter()
            .enumerate()
            .map(|(k, a)| {
                if !vars_reachable(body, out).contains(&(k as u16)) {
                    return ExprId::ZERO;
                }
                let leaf = Arg {
                    src: a.src,
                    view: a.view.normalized(space),
                };
                g.var(intern_arg(&mut leaves, leaf) as u16)
            })
            .collect();
        let out = replay(body, &[out], &vars, &mut g)[0];
        Canon {
            leaves,
            nodes: g.nodes_slice()[..=out.index() as usize].to_vec(),
        }
    }

    /// Does `m_arg` (as read by `c`) hold `max` over `c`'s reduce coord of
    /// exactly the expression `s` of `c`'s body? Returns `s`'s canonical form.
    fn is_row_max_of(&self, c: &Comp, s: ExprId, m_arg: &Arg) -> Option<Canon> {
        let mc = self.comp_at(m_arg.src.n())?;
        let n_out = c.points().len();
        let structural = mc.monoid.as_ref().map(Monoid::kind) == Some(MonoidKind::Max)
            && mc.tuple == 0
            && mc.red.len() == 1
            && c.red.len() == 1
            && !m_arg.view.depends_on(n_out);
        if !structural {
            return None;
        }
        // Extend c's view of m with "m's reduce coord = c's reduce coord",
        // push m's args through it, and compare m's body with `s`.
        let mut ext = m_arg.view.clone();
        ext.forms.push(Form::unit(c.rank(), n_out));
        let space = c.space();
        let m_args: Vec<Arg> = mc
            .args
            .iter()
            .map(|a| Arg {
                src: a.src,
                view: a.view.compose(&ext, c.rank()),
            })
            .collect();
        let via_m = Self::canon(&mc.body, mc.body_outs[0], &m_args, &space);
        let direct = Self::canon(&c.body, s, &c.args, &space);
        (via_m == direct).then_some(direct)
    }

    /// Rewrite `m = max_j s` and every `Σ_j g·exp(s − m)` over the same
    /// `s` into a single online-softmax reduction. Run after
    /// [`substitute_maps`](Self::substitute_maps) so `exp(s − m)` is visible
    /// in reduction bodies.
    ///
    /// Members may have extra out dims beyond `m`'s (e.g. `Σ_j p[i,j]·v[j,c]`
    /// has a `c`); those are unrolled into the fused comp's tuple dim.
    pub fn stabilize_softmax(&self, outputs: &[TensorId]) -> (TensorGraph, Vec<TensorId>) {
        struct Member {
            node: usize,
            m_view: View,
            s: ExprId,
            g: Option<ExprId>,
        }
        let is_output = |i: usize| outputs.iter().any(|o| o.n() == i);
        // (max node, canonical s) -> members
        let mut groups: HashMap<(usize, Canon), Vec<Member>> = HashMap::new();
        for i in 0..self.nodes.len() {
            let Some(c) = self.comp_at(i) else { continue };
            let sum = c.monoid.as_ref().map(Monoid::kind) == Some(MonoidKind::Sum);
            if !sum || c.red.len() != 1 || c.tuple != 0 || is_output(i) {
                continue;
            }
            let Some((s, m_var, g)) = match_exp_diff(&c.body, c.body_outs[0]) else {
                continue;
            };
            let m_arg = &c.args[m_var as usize];
            if let Some(canon) = self.is_row_max_of(c, s, m_arg) {
                groups
                    .entry((m_arg.src.n(), canon))
                    .or_default()
                    .push(Member {
                        node: i,
                        m_view: m_arg.view.clone(),
                        s,
                        g,
                    });
            }
        }
        if groups.is_empty() {
            return (self.clone(), outputs.to_vec());
        }

        let mut g2 = self.clone();
        // member node -> (fused node, projection from member out coords to fused coords)
        let mut redirect: HashMap<usize, (usize, View)> = HashMap::new();
        for ((m_node, _), members) in groups {
            let mc = self.comp_at(m_node).unwrap();
            let m_out = mc.out.len();

            // Extra out dims of members (e.g. the value column of Σ_j p·v)
            // become point dims of the fused comp, so a lane per column
            // can share the reduction. All members must agree on them.
            let member_extra: Vec<(Vec<Option<usize>>, Vec<usize>)> = members
                .iter()
                .map(|mem| {
                    let c = self.comp_at(mem.node).unwrap();
                    let n_out = c.points().len();
                    let reads: Vec<Option<usize>> = (0..n_out)
                        .map(|d| mem.m_view.forms.iter().position(|f| f.as_unit() == Some(d)))
                        .collect();
                    // size-1 dims carry no lanes; they are neither read nor extra
                    let extra: Vec<usize> = (0..n_out)
                        .filter(|&d| reads[d].is_none() && c.out[d] != Dim::Fixed(1))
                        .collect();
                    (reads, extra)
                })
                .collect();
            let extra_shape: Vec<Dim> = member_extra
                .iter()
                .zip(&members)
                .map(|((_, extra), mem)| {
                    let c = self.comp_at(mem.node).unwrap();
                    extra.iter().map(|&d| c.out[d]).collect::<Vec<Dim>>()
                })
                .max_by_key(|v| v.len())
                .unwrap_or_default();
            let agree = member_extra.iter().zip(&members).all(|((_, extra), mem)| {
                let c = self.comp_at(mem.node).unwrap();
                let mine: Vec<Dim> = extra.iter().map(|&d| c.out[d]).collect();
                mine.is_empty() || mine == extra_shape
            });
            if !agree {
                continue;
            }
            let e = extra_shape.len();
            let fused_rank = m_out + e + 1;

            let mut body = ExprGraph::new();
            let mut args: Vec<Arg> = Vec::new();
            // slot 0: s, from the max's own body, re-indexed into the fused space
            let m_ext = View {
                forms: (0..=m_out)
                    .map(|d| Form::unit(fused_rank, if d == m_out { m_out + e } else { d }))
                    .collect(),
            };
            let m_vars: Vec<ExprId> = mc
                .args
                .iter()
                .map(|a| {
                    let leaf = Arg {
                        src: a.src,
                        view: a.view.compose(&m_ext, fused_rank),
                    };
                    body.var(intern_arg(&mut args, leaf) as u16)
                })
                .collect();
            let mut outs = replay(&mc.body, &mc.body_outs, &m_vars, &mut body);
            let mut projections = Vec::new();

            for (mem, (reads, extra)) in members.iter().zip(&member_extra) {
                let c = self.comp_at(mem.node).unwrap();
                let n_out = c.points().len();
                let slot = outs.len() as i64;
                let inv = View {
                    forms: (0..c.rank())
                        .map(|d| match reads.get(d) {
                            Some(Some(j)) => Form::unit(fused_rank, *j),
                            // lane dim, or a size-1 dim that is always 0
                            Some(None) => match extra.iter().position(|x| *x == d) {
                                Some(k) => Form::unit(fused_rank, m_out + k),
                                None => Form::zero(fused_rank),
                            },
                            None => Form::unit(fused_rank, m_out + e),
                        })
                        .collect(),
                };
                let _ = mem.s;
                let used = mem.g.map_or_else(Vec::new, |g| vars_reachable(&c.body, g));
                let vars: Vec<ExprId> = c
                    .args
                    .iter()
                    .enumerate()
                    .map(|(k, a)| {
                        if !used.contains(&(k as u16)) {
                            return ExprId::ZERO;
                        }
                        let leaf = Arg {
                            src: a.src,
                            view: a.view.compose(&inv, fused_rank),
                        };
                        body.var(intern_arg(&mut args, leaf) as u16)
                    })
                    .collect();
                outs.push(match mem.g {
                    Some(g) => replay(&c.body, &[g], &vars, &mut body)[0],
                    None => body.lit(1.0),
                });

                // member[o] == fused[m_view(o), extra coords (0 if absent), slot]
                let mut proj: Vec<Form> = mem
                    .m_view
                    .forms
                    .iter()
                    .map(|f| Form {
                        coefs: f.coefs[..n_out].to_vec(),
                        offset: f.offset,
                    })
                    .collect();
                for k in 0..e {
                    proj.push(match extra.get(k) {
                        Some(&d) => Form::unit(n_out, d),
                        None => Form::zero(n_out),
                    });
                }
                proj.push(Form::constant(n_out, slot));
                projections.push((mem.node, View { forms: proj }));
            }

            let width = outs.len();
            let mut out = mc.out.clone();
            out.extend(extra_shape.iter().copied());
            out.push(Dim::Fixed(width));
            let fused = g2
                .push_comp(Comp {
                    out,
                    tuple: 1,
                    red: mc.red.clone(),
                    args,
                    body,
                    body_outs: outs,
                    monoid: Some(Monoid::online_softmax(width - 1)),
                })
                .expect("fused softmax comp is well-formed");
            for (node, proj) in projections {
                redirect.insert(node, (fused.n(), proj));
            }
        }
        if redirect.is_empty() {
            return (self.clone(), outputs.to_vec());
        }

        for node in &mut g2.nodes {
            let Op::Comp(c) = &mut node.op else { continue };
            let rank = c.rank();
            for a in &mut c.args {
                if let Some((fused, proj)) = redirect.get(&a.src.n()) {
                    a.view = proj.compose(&a.view, rank);
                    a.src = TensorId(*fused as u32);
                }
            }
        }
        g2.eliminate_dead(outputs)
    }

    /// Live nodes in dependency order (post-order DFS from the outputs).
    fn topo(&self, outputs: &[TensorId]) -> Vec<usize> {
        let mut order = Vec::new();
        let mut seen = vec![false; self.nodes.len()];
        let mut stack: Vec<(usize, bool)> = outputs.iter().map(|o| (o.n(), false)).collect();
        while let Some((i, done)) = stack.pop() {
            if done {
                order.push(i);
                continue;
            }
            if std::mem::replace(&mut seen[i], true) {
                continue;
            }
            stack.push((i, true));
            if let Some(c) = self.comp_at(i) {
                stack.extend(c.args.iter().map(|a| (a.src.n(), false)));
            }
        }
        order
    }

    /// Drop dead nodes and renumber in dependency order.
    pub fn eliminate_dead(&self, outputs: &[TensorId]) -> (TensorGraph, Vec<TensorId>) {
        let mut remap = vec![None; self.nodes.len()];
        let mut ng = TensorGraph {
            nodes: Vec::new(),
            n_inputs: self.n_inputs,
        };
        for i in self.topo(outputs) {
            let mut node = self.nodes[i].clone();
            if let Op::Comp(c) = &mut node.op {
                for a in &mut c.args {
                    a.src = TensorId(remap[a.src.n()].unwrap() as u32);
                }
            }
            ng.nodes.push(node);
            remap[i] = Some(ng.nodes.len() - 1);
        }
        let outs = outputs
            .iter()
            .map(|o| TensorId(remap[o.n()].unwrap() as u32))
            .collect();
        (ng, outs)
    }

    /// Schedule: recompute a single-use reduction inside its consumer when
    /// the consumer's iteration space is no larger than the producer's
    /// point count, i.e. inlining evaluates each point at most once.
    pub fn schedule_inline(&mut self, outputs: &[TensorId]) {
        self.schedule_inline_up_to(outputs, 1);
    }

    /// Like [`schedule_inline`](Self::schedule_inline) but allows the
    /// consumer to evaluate the producer up to `factor` times per point.
    /// Worth it when the consumer's extra points are lanes that a tiled
    /// schedule computes the shared part for cooperatively.
    pub fn schedule_inline_up_to(&mut self, outputs: &[TensorId], factor: usize) {
        let live = self.live(outputs);
        let uses = self.use_counts(outputs, &live);
        let is_output = |i: usize| outputs.iter().any(|o| o.n() == i);
        let n = self.nodes.len();
        let mut consumer_space = vec![0usize; n];
        for i in (0..n).filter(|&i| live[i]) {
            if let Some(c) = self.comp_at(i) {
                for a in &c.args {
                    consumer_space[a.src.n()] = numel(&c.space());
                }
            }
        }
        for i in 0..n {
            let Some(c) = self.comp_at(i) else { continue };
            if live[i]
                && uses[i] == 1
                && !is_output(i)
                && consumer_space[i] <= factor * numel(c.points())
            {
                self.nodes[i].inline = true;
            }
        }
    }

    /// Schedule: every materialized reduction with at least `min_red`
    /// elements per point is split across a workgroup of `threads`.
    /// The monoid's combine is associative, so this is always valid —
    /// including for tuple-state monoids like online softmax.
    pub fn schedule_split(&mut self, outputs: &[TensorId], threads: u32, min_red: usize) {
        let live = self.live(outputs);
        let targets: Vec<usize> = (0..self.nodes.len())
            .filter(|&i| live[i] && !self.nodes[i].inline)
            .filter(|&i| {
                self.comp_at(i)
                    .is_some_and(|c| c.monoid.is_some() && numel(&c.red) >= min_red)
            })
            .collect();
        for i in targets {
            self.nodes[i].split = threads;
        }
    }

    /// Schedule: materialized single-axis reductions with at least
    /// `min_red` elements per point get `points` threads per workgroup
    /// (one point each) and stage workgroup-invariant reads in chunks of
    /// `chunk` reduce indices. This is the structure of flash attention
    /// and of a shared-memory matmul, derived from the views alone.
    pub fn schedule_tile(&mut self, outputs: &[TensorId], points: u32, chunk: u32, min_red: usize) {
        self.schedule_tile_2d(outputs, points, 1, chunk, min_red);
    }

    /// Like [`schedule_tile`](Self::schedule_tile) with a `bx × by` block
    /// of points per workgroup over the last two point dims. Reads that
    /// vary along only one tile axis (matmul's `a[i,k]`, `b[k,j]`) are
    /// staged once per workgroup instead of once per thread.
    pub fn schedule_tile_2d(
        &mut self,
        outputs: &[TensorId],
        bx: u32,
        by: u32,
        chunk: u32,
        min_red: usize,
    ) {
        let live = self.live(outputs);
        let targets: Vec<usize> = (0..self.nodes.len())
            .filter(|&i| live[i] && !self.nodes[i].inline && self.nodes[i].split == 0)
            .filter(|&i| {
                self.comp_at(i).is_some_and(|c| {
                    c.monoid.is_some()
                        && c.red.len() == 1
                        && c.red[0].n() >= min_red
                        && (by == 1 || c.points().len() >= 2)
                })
            })
            .collect();
        for i in targets {
            self.nodes[i].tile = (bx, by, chunk);
        }
    }

    /// Replace symbolic dims by sizes. Every `Sym` in the graph must be
    /// named; the result is a concrete graph that can be scheduled,
    /// evaluated and lowered.
    pub fn bind(&self, bindings: &[(&str, usize)]) -> Result<TensorGraph, ShapeError> {
        let sub = |d: &Dim| match d {
            Dim::Fixed(n) => Ok(Dim::Fixed(*n)),
            Dim::Sym(name) => bindings
                .iter()
                .find(|(k, _)| k == name)
                .map(|(_, v)| Dim::Fixed(*v))
                .ok_or_else(|| ShapeError(format!("bind: no value for dim `{name}`"))),
        };
        let mut g = self.clone();
        for node in &mut g.nodes {
            node.shape = node.shape.iter().map(sub).collect::<Result<_, _>>()?;
            if let Op::Comp(c) = &mut node.op {
                c.out = c.out.iter().map(sub).collect::<Result<_, _>>()?;
                c.red = c.red.iter().map(sub).collect::<Result<_, _>>()?;
            }
        }
        Ok(g)
    }

    /// `true` if no dim is symbolic.
    pub fn is_bound(&self) -> bool {
        self.nodes.iter().all(|n| {
            let shape_ok = n.shape.iter().all(|d| d.fixed().is_some());
            let comp_ok = match &n.op {
                Op::Comp(c) => c.red.iter().all(|d| d.fixed().is_some()),
                Op::Input(_) => true,
            };
            shape_ok && comp_ok
        })
    }

    /// Default thread mapping for a bound graph, from the measured rules
    /// of thumb in `BENCHMARKS.md`: a reduction whose points end in a lane
    /// axis (attention's value columns) is tiled rows × lanes with shared
    /// work staged cooperatively; a reduction with few output points is
    /// split across a workgroup; one with many points is already parallel
    /// enough one-thread-per-point.
    /// `autotune` over candidate graphs still decides which *algebra* to
    /// run (e.g. materialized scores vs online softmax).
    pub fn schedule_auto(&mut self, outputs: &[TensorId]) {
        const MANY_POINTS: usize = 1 << 14;
        const LANES: usize = 16;
        self.schedule_inline(outputs);
        let live = self.live(outputs);
        let reductions: Vec<usize> = (0..self.nodes.len())
            .filter(|&i| live[i] && !self.nodes[i].inline)
            .filter(|&i| {
                self.comp_at(i)
                    .is_some_and(|c| c.monoid.is_some() && numel(&c.red) >= 64)
            })
            .collect();
        let uses = self.use_counts(outputs, &live);
        let is_output = |i: usize| outputs.iter().any(|o| o.n() == i);
        for i in reductions {
            let c = self.comp_at(i).unwrap();
            let pts = fixed(c.points());
            let lane_dim = (0..pts.len()).rev().find(|&d| pts[d] > 1);
            let lanes = lane_dim.map_or(1, |d| pts[d]);
            let rows = pts.iter().rev().filter(|&&n| n > 1).nth(1).is_some();
            // Producers the lane axis does not depend on: a tile computes
            // them once per workgroup instead of once per lane.
            let shared: Vec<usize> = lane_dim
                .map(|lane| {
                    c.args
                        .iter()
                        .filter(|a| !a.view.depends_on(lane) && uses[a.src.n()] == 1)
                        .map(|a| a.src.n())
                        .filter(|&p| !is_output(p) && self.comp_at(p).is_some())
                        .collect()
                })
                .unwrap_or_default();
            let by = (256 / lanes).max(1) as u32;
            if c.red.len() == 1 && (LANES..=128).contains(&lanes) && rows && !shared.is_empty() {
                // Rows × lanes per workgroup with the shared sub-computation
                // staged cooperatively. Without one (a plain matmul) the
                // tile has nothing to share and one thread per point wins.
                self.nodes[i].tile = (lanes as u32, by, 32);
                for p in shared {
                    self.nodes[p].inline = true;
                }
            } else if pts.iter().product::<usize>() < MANY_POINTS {
                self.nodes[i].split = 256;
            }
        }
    }

    // -- reference interpreter -------------------------------------------------

    /// Evaluate `outputs`. Materialized nodes are computed once into
    /// buffers; inlined nodes are recomputed at every point they are read.
    pub fn eval(&self, inputs: &[Vec<f64>], outputs: &[TensorId]) -> Vec<Vec<f64>> {
        let live = self.live(outputs);
        let mut ev = Evaluator {
            g: self,
            inputs,
            bufs: vec![Vec::new(); self.nodes.len()],
            compiled: self
                .nodes
                .iter()
                .map(|n| match &n.op {
                    Op::Comp(c) => Some(Compiled {
                        body: c.body.compile_many(&c.body_outs),
                        combine: c.monoid.as_ref().map(|m| m.combine.compile_many(&m.outs)),
                    }),
                    Op::Input(_) => None,
                })
                .collect(),
        };
        for i in (0..self.nodes.len()).filter(|&i| live[i] && !self.nodes[i].inline) {
            if let Some(c) = self.comp_at(i) {
                let w = c.width();
                let mut buf = vec![0.0; numel(c.points()) * w];
                for (p, idx) in indices(&fixed(c.points())).enumerate() {
                    ev.point(i, &idx, &mut buf[p * w..(p + 1) * w]);
                }
                ev.bufs[i] = buf;
            }
        }
        outputs
            .iter()
            .map(|o| {
                indices(&fixed(&self.nodes[o.n()].shape))
                    .map(|idx| ev.read(*o, &idx))
                    .collect()
            })
            .collect()
    }
}

fn intern_arg(args: &mut Vec<Arg>, a: Arg) -> usize {
    args.iter().position(|x| *x == a).unwrap_or_else(|| {
        args.push(a);
        args.len() - 1
    })
}

struct Compiled {
    body: tang_expr::compile::CompiledMany,
    combine: Option<tang_expr::compile::CompiledMany>,
}

struct Evaluator<'a> {
    g: &'a TensorGraph,
    inputs: &'a [Vec<f64>],
    bufs: Vec<Vec<f64>>,
    compiled: Vec<Option<Compiled>>,
}

impl Evaluator<'_> {
    /// Value of `t` at full-shape coordinates `idx`.
    fn read(&self, t: TensorId, idx: &[usize]) -> f64 {
        let node = &self.g.nodes[t.n()];
        match &node.op {
            Op::Input(k) => self.inputs[*k][flat_index(&fixed(&node.shape), idx)],
            Op::Comp(c) => {
                let (pt, tup) = idx.split_at(c.points().len());
                let slot = flat_index(&fixed(&c.out[c.points().len()..]), tup);
                if node.inline {
                    let mut vals = vec![0.0; c.width()];
                    self.point(t.n(), pt, &mut vals);
                    return vals[slot];
                }
                let w = c.width();
                self.bufs[t.n()][flat_index(&fixed(c.points()), pt) * w + slot]
            }
        }
    }

    fn point(&self, i: usize, out_idx: &[usize], vals: &mut [f64]) {
        let c = self.g.comp_at(i).unwrap();
        let k = self.compiled[i].as_ref().unwrap();
        let w = c.width();
        let mut arg_vals = vec![0.0; c.args.len()];
        let mut idx = out_idx.to_vec();
        let base = idx.len();
        idx.extend(std::iter::repeat_n(0, c.red.len()));
        let mut body_at = |idx: &[usize], elem: &mut [f64]| {
            for (j, a) in c.args.iter().enumerate() {
                arg_vals[j] = self.read(a.src, &a.view.coords(idx));
            }
            (k.body)(&arg_vals, elem);
        };
        let Some(m) = &c.monoid else {
            body_at(&idx, vals);
            return;
        };
        vals.copy_from_slice(&m.identity);
        let combine = k.combine.as_ref().unwrap();
        let mut elem = vec![0.0; w];
        let mut pair = vec![0.0; 2 * w];
        for red_idx in indices(&fixed(&c.red)) {
            idx[base..].copy_from_slice(&red_idx);
            body_at(&idx, &mut elem);
            pair[..w].copy_from_slice(vals);
            pair[w..].copy_from_slice(&elem);
            combine(&pair, vals);
        }
    }
}

fn flat_index(shape: &[usize], idx: &[usize]) -> usize {
    idx.iter().zip(shape).fold(0, |acc, (i, n)| acc * n + i)
}

fn indices(shape: &[usize]) -> impl Iterator<Item = Vec<usize>> + '_ {
    (0..shape.iter().product::<usize>()).map(move |mut p| {
        let mut idx = vec![0; shape.len()];
        for d in (0..shape.len()).rev() {
            idx[d] = p % shape[d];
            p /= shape[d];
        }
        idx
    })
}

/// Copy the subgraph under `outs` into `dst`, substituting `Var(i)` with `vars[i]`.
pub(super) fn replay(
    src: &ExprGraph,
    outs: &[ExprId],
    vars: &[ExprId],
    dst: &mut ExprGraph,
) -> Vec<ExprId> {
    let nodes = src.nodes_slice();
    let last = outs.iter().map(|o| o.index()).max().unwrap_or(0) as usize;
    let mut map: Vec<ExprId> = Vec::with_capacity(last + 1);
    for node in &nodes[..=last] {
        let m = |e: &ExprId| map[e.index() as usize];
        let id = match node {
            Node::Var(v) => vars[*v as usize],
            Node::Lit(bits) => dst.lit(f64::from_bits(*bits)),
            Node::Add(a, b) => dst.add(m(a), m(b)),
            Node::Mul(a, b) => dst.mul(m(a), m(b)),
            Node::Neg(a) => dst.neg(m(a)),
            Node::Recip(a) => dst.recip(m(a)),
            Node::Sqrt(a) => dst.sqrt(m(a)),
            Node::Sin(a) => dst.sin(m(a)),
            Node::Atan2(a, b) => dst.atan2(m(a), m(b)),
            Node::Exp2(a) => dst.exp2(m(a)),
            Node::Log2(a) => dst.log2(m(a)),
            Node::Select(c, a, b) => dst.select(m(c), m(a), m(b)),
        };
        map.push(id);
    }
    outs.iter().map(|o| map[o.index() as usize]).collect()
}

/// Input vars reachable from `out`.
fn vars_reachable(g: &ExprGraph, out: ExprId) -> Vec<u16> {
    let live = g.live_set(&[out]);
    let nodes = g.nodes_slice();
    live.iter()
        .filter_map(|&i| match nodes[i] {
            Node::Var(v) => Some(v),
            _ => None,
        })
        .collect()
}

/// Canonical expression for structural comparison (see [`TensorGraph::canon`]).
#[derive(Clone, PartialEq, Eq, Hash)]
struct Canon {
    leaves: Vec<Arg>,
    nodes: Vec<Node>,
}

/// Match `g · exp(s − m)` (or just `exp(s − m)`) where `m` is a var and
/// `s` is any expression. `exp(x)` lowers to `exp2(x · log2e)` and
/// `a − b` to `add(a, neg(b))`.
fn match_exp_diff(g: &ExprGraph, out: ExprId) -> Option<(ExprId, u16, Option<ExprId>)> {
    let n = g.nodes_slice();
    let at = |e: ExprId| n[e.index() as usize];
    let exp_diff = |e: ExprId| -> Option<(ExprId, u16)> {
        let Node::Exp2(x) = at(e) else { return None };
        let Node::Mul(d, l) = at(x) else { return None };
        if at(l) != Node::lit(std::f64::consts::LOG2_E) {
            return None;
        }
        let Node::Add(s, nb) = at(d) else { return None };
        let Node::Neg(b) = at(nb) else { return None };
        match at(b) {
            Node::Var(m) => Some((s, m)),
            _ => None,
        }
    };
    if let Some((s, m)) = exp_diff(out) {
        return Some((s, m, None));
    }
    let Node::Mul(a, b) = at(out) else {
        return None;
    };
    exp_diff(a)
        .map(|(s, m)| (s, m, Some(b)))
        .or_else(|| exp_diff(b).map(|(s, m)| (s, m, Some(a))))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() <= 1e-9 * (1.0 + y.abs()), "{x} vs {y}");
        }
    }

    fn ramp(n: usize, scale: f64, off: f64) -> Vec<f64> {
        (0..n)
            .map(|i| ((i * 7919) % 97) as f64 * scale + off)
            .collect()
    }

    fn rmsnorm(g: &mut TensorGraph, x: TensorId) -> TensorId {
        let shape = g.shape(x).to_vec();
        let dim = shape[1].n();
        let sq = g.map(&[x], |v| v[0] * v[0]).unwrap();
        let sum = g.reduce(sq, 1, Monoid::sum()).unwrap();
        let inv = g
            .map(&[sum], |v| {
                ExprId::from_f64(1.0)
                    / (v[0] / ExprId::from_f64(dim as f64) + ExprId::from_f64(1e-6)).sqrt()
            })
            .unwrap();
        let inv = g.broadcast(inv, &shape).unwrap();
        g.map(&[x, inv], |v| v[0] * v[1]).unwrap()
    }

    #[test]
    fn rmsnorm_matches_reference_and_fuses_to_two_comps() {
        let (rows, dim) = (3, 8);
        let mut g = TensorGraph::new();
        let x = g.input(&[rows, dim]);
        let y = rmsnorm(&mut g, x);
        let data = ramp(rows * dim, 0.1, -1.0);

        let mut want = vec![0.0; rows * dim];
        for r in 0..rows {
            let row = &data[r * dim..(r + 1) * dim];
            let rms = (row.iter().map(|v| v * v).sum::<f64>() / dim as f64 + 1e-6).sqrt();
            for c in 0..dim {
                want[r * dim + c] = row[c] / rms;
            }
        }
        close(&g.eval(std::slice::from_ref(&data), &[y])[0], &want);

        // sq folds into the reduction; inv + broadcast fold into the output map
        let (fg, outs) = g.substitute_maps(&[y]);
        assert_eq!(fg.count_comps(&outs), 2);
        close(&fg.eval(&[data], &outs)[0], &want);
    }

    #[test]
    fn matmul_and_transpose() {
        let mut g = TensorGraph::new();
        let a = g.input(&[2, 3]);
        let b = g.input(&[2, 3]);
        let bt = g.transpose(b, &[1, 0]).unwrap();
        let c = g.matmul(a, bt).unwrap();
        assert_eq!(g.shape(c), &[2, 2]);
        let av = vec![1., 2., 3., 4., 5., 6.];
        let bv = vec![1., 0., 1., 0., 1., 0.];
        close(
            &g.eval(&[av.clone(), bv.clone()], &[c])[0],
            &[4., 2., 10., 5.],
        );

        // the transpose is a view: after substitution it is gone
        let (fg, outs) = g.substitute_maps(&[c]);
        assert_eq!(fg.count_comps(&outs), 1);
        close(&fg.eval(&[av, bv], &outs)[0], &[4., 2., 10., 5.]);
    }

    #[test]
    fn shape_errors() {
        let mut g = TensorGraph::new();
        let a = g.input(&[2, 3]);
        let b = g.input(&[4, 3]);
        assert!(g.matmul(a, b).is_err());
        assert!(g.map(&[a, b], |v| v[0] + v[1]).is_err());
        assert!(g.broadcast(a, &[2, 4]).is_err());
        assert!(g.transpose(a, &[0, 0]).is_err());
        assert!(g.reduce(a, 2, Monoid::sum()).is_err());
        let out_of_bounds = Arg {
            src: a,
            view: View::pick(2, &[Some(1), Some(0)]),
        };
        assert!(g
            .comp(&[2, 3], &[], vec![out_of_bounds], None, |v| vec![v[0]])
            .is_err());
    }

    #[test]
    fn monoid_kinds() {
        assert_eq!(Monoid::sum().kind(), MonoidKind::Sum);
        assert_eq!(Monoid::max().kind(), MonoidKind::Max);
        assert_eq!(Monoid::online_softmax(2).kind(), MonoidKind::OnlineSoftmax);
    }

    /// softmax(q·kᵀ [+ mask])·v with every intermediate written out.
    fn naive_attention(
        g: &mut TensorGraph,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        mask: Option<TensorId>,
    ) -> TensorId {
        let n = g.shape(q)[0];
        let dv = g.shape(v)[1];
        let kt = g.transpose(k, &[1, 0]).unwrap();
        let mut s = g.matmul(q, kt).unwrap(); // [n, n]
        if let Some(mask) = mask {
            s = g.map(&[s, mask], |v| v[0] + v[1]).unwrap();
        }
        let m = g.reduce(s, 1, Monoid::max()).unwrap(); // [n, 1]
        let mb = g.broadcast(m, &[n, n]).unwrap();
        let p = g.map(&[s, mb], |v| (v[0] - v[1]).exp()).unwrap(); // [n, n]
        let l = g.reduce(p, 1, Monoid::sum()).unwrap(); // [n, 1]
        let o = g.matmul(p, v).unwrap(); // [n, dv]
        let lb = g.broadcast(l, &[n, dv]).unwrap();
        g.map(&[o, lb], |v| v[0] / v[1]).unwrap()
    }

    /// One symbolic graph serves every sequence length: rewrite and
    /// differentiate with `n` unbound, bind per size, match fixed builds.
    #[test]
    fn symbolic_seq_len_binds_to_any_size() {
        let (d, dv) = (4, 4);
        let n = Dim::Sym("n");
        let mut g = TensorGraph::new();
        let q = g.input(&[n, Dim::Fixed(d)]);
        let k = g.input(&[n, Dim::Fixed(d)]);
        let v = g.input(&[n, Dim::Fixed(dv)]);
        let y = naive_attention(&mut g, q, k, v, None);
        assert!(!g.is_bound());
        let grads = g.backward(y, &[q]).unwrap();
        let (sg, outs) = g.substitute_maps(&[y, grads[0]]);
        let (sg, outs) = sg.stabilize_softmax(&outs);
        assert!(sg
            .reduction_kinds(&outs)
            .contains(&MonoidKind::OnlineSoftmax));
        assert!(sg.bind(&[("m", 3)]).is_err(), "wrong symbol must fail");

        for size in [5usize, 16] {
            let mut bound = sg.bind(&[("n", size)]).unwrap();
            assert!(bound.is_bound());
            bound.schedule_inline_up_to(&outs, dv);
            let inputs = [
                ramp(size * d, 0.05, -1.0),
                ramp(size * d, 0.07, 0.3),
                ramp(size * dv, 0.1, -2.0),
            ];
            let got = bound.eval(&inputs, &outs);

            let mut fg = TensorGraph::new();
            let fq = fg.input(&[size, d]);
            let fk = fg.input(&[size, d]);
            let fv = fg.input(&[size, dv]);
            let fy = naive_attention(&mut fg, fq, fk, fv, None);
            let fgr = fg.backward(fy, &[fq]).unwrap();
            let want = fg.eval(&inputs, &[fy, fgr[0]]);
            close(&got[0], &want[0]);
            close(&got[1], &want[1]);
        }
    }

    #[test]
    fn schedule_auto_splits_narrow_reductions_only() {
        // 64 rows x 2048: few points, wide reduce -> split
        let mut g = TensorGraph::new();
        let x = g.input(&[64, 2048]);
        let y = g.reduce(x, 1, Monoid::sum()).unwrap();
        g.schedule_auto(&[y]);
        assert_eq!(g.nodes()[y.n()].split, 256);

        // 1M points x 8: plenty of threads -> one thread per point
        let mut g = TensorGraph::new();
        let x = g.input(&[1 << 20, 8]);
        let y = g.reduce(x, 1, Monoid::sum()).unwrap();
        g.schedule_auto(&[y]);
        assert_eq!(g.nodes()[y.n()].split, 0);
        assert_eq!(g.nodes()[y.n()].tile, (0, 0, 0));

        // a plain matmul has nothing to share across lanes -> one thread per point
        let mut g = TensorGraph::new();
        let a = g.input(&[256, 128]);
        let b = g.input(&[128, 64]);
        let c = g.matmul(a, b).unwrap();
        g.schedule_auto(&[c]);
        assert_eq!(g.nodes()[c.n()].tile, (0, 0, 0));
        assert_eq!(g.nodes()[c.n()].split, 0);

        // attention after the softmax rewrite: scores are lane-invariant -> tiled, scores inlined
        let (n, d, dv) = (64, 16, 32);
        let mut g = TensorGraph::new();
        let q = g.input(&[n, d]);
        let k = g.input(&[n, d]);
        let v = g.input(&[n, dv]);
        let y = naive_attention(&mut g, q, k, v, None);
        let (g, outs) = g.substitute_maps(&[y]);
        let (mut g, outs) = g.stabilize_softmax(&outs);
        g.schedule_auto(&outs);
        let fused = g
            .nodes()
            .iter()
            .position(|nd| matches!(&nd.op, Op::Comp(c) if c.tuple == 1))
            .unwrap();
        assert_eq!(g.nodes()[fused].tile, (dv as u32, 8, 32));
        assert_eq!(g.materialized(&outs).len(), 2);
    }

    #[test]
    fn unbound_graph_cannot_be_scheduled_or_evaluated() {
        let mut g = TensorGraph::new();
        let x = g.input(&[Dim::Sym("n"), Dim::Fixed(2)]);
        let y = g.reduce(x, 1, Monoid::sum()).unwrap();
        let r = std::panic::catch_unwind(|| g.eval(&[vec![0.0; 4]], &[y]));
        assert!(r.is_err());
    }

    #[test]
    fn attention_is_derived_not_written() {
        check_attention_derivation(false);
    }

    /// With a mask, `s` in `exp(s − max_j s)` is `scores + mask`, not a
    /// bare arg: the matcher must compare expressions, not vars.
    #[test]
    fn causal_attention_is_derived_too() {
        check_attention_derivation(true);
    }

    fn check_attention_derivation(causal: bool) {
        let (n, d, dv) = (16, 4, 4);
        let mut g = TensorGraph::new();
        let q = g.input(&[n, d]);
        let k = g.input(&[n, d]);
        let v = g.input(&[n, dv]);
        let mask = causal.then(|| g.input(&[n, n]));
        let y = naive_attention(&mut g, q, k, v, mask);

        let mut inputs = vec![
            ramp(n * d, 0.05, -1.0),
            ramp(n * d, 0.07, 0.3),
            ramp(n * dv, 0.1, -2.0),
        ];
        if causal {
            inputs.push(
                (0..n * n)
                    .map(|p| if p % n <= p / n { 0.0 } else { -1e30 })
                    .collect(),
            );
        }
        let want = g.eval(&inputs, &[y]).remove(0);

        // naive: scores and p are materialized n×n
        assert!(g.materialized(&[y]).iter().any(|(_, s)| numel(s) == n * n));

        let (g, outs) = g.substitute_maps(&[y]);
        let (mut g, outs) = g.stabilize_softmax(&outs);
        g.schedule_inline_up_to(&outs, dv);

        // flash: one online-softmax reduction, q·k recomputed inside it,
        // and nothing n×n ever hits memory.
        let kinds = g.reduction_kinds(&outs);
        assert_eq!(
            kinds
                .iter()
                .filter(|k| **k == MonoidKind::OnlineSoftmax)
                .count(),
            1,
            "{kinds:?}"
        );
        assert!(!kinds.contains(&MonoidKind::Max), "max_j s should be gone");
        let mats = g.materialized(&outs);
        assert!(
            mats.iter().all(|(_, s)| numel(s) < n * n),
            "materialized: {mats:?}"
        );
        // scores (inlined), online reduce [n, 1, dv, 3], final divide
        assert_eq!(g.count_comps(&outs), 3);
        assert_eq!(mats.len(), 2);

        close(&g.eval(&inputs, &outs)[0], &want);
    }
}
