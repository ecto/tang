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
    fn normalized(&self, space: &[usize]) -> View {
        let mut v = self.clone();
        for f in &mut v.forms {
            for (c, &n) in f.coefs.iter_mut().zip(space) {
                if n == 1 {
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
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Arg {
    pub src: TensorId,
    pub view: View,
}

#[derive(Clone)]
pub struct Comp {
    /// Output shape. The last `tuple` dims are produced per point.
    pub out: Vec<usize>,
    /// Trailing out dims filled by the body (0 for ordinary comps).
    pub tuple: usize,
    pub red: Vec<usize>,
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
    fn points(&self) -> &[usize] {
        &self.out[..self.out.len() - self.tuple]
    }

    fn rank(&self) -> usize {
        self.points().len() + self.red.len()
    }

    fn space(&self) -> Vec<usize> {
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
    pub shape: Vec<usize>,
    /// Schedule: recompute inside consumers instead of writing a buffer.
    pub inline: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShapeError(pub String);

impl std::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "shape error: {}", self.0)
    }
}

impl std::error::Error for ShapeError {}

fn err<T>(msg: impl Into<String>) -> Result<T, ShapeError> {
    Err(ShapeError(msg.into()))
}

pub fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
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

    pub fn shape(&self, t: TensorId) -> &[usize] {
        &self.nodes[t.n()].shape
    }

    pub fn input(&mut self, dims: &[usize]) -> TensorId {
        let i = self.n_inputs;
        self.n_inputs += 1;
        self.nodes.push(TNode {
            op: Op::Input(i),
            shape: dims.to_vec(),
            inline: false,
        });
        TensorId(self.nodes.len() as u32 - 1)
    }

    /// The general op. `f` receives one scalar per arg and returns the
    /// per-point values (`monoid.width` of them when reducing).
    pub fn comp(
        &mut self,
        out: &[usize],
        red: &[usize],
        args: Vec<Arg>,
        monoid: Option<Monoid>,
        f: impl FnOnce(&[ExprId]) -> Vec<ExprId>,
    ) -> Result<TensorId, ShapeError> {
        let (body, body_outs) = trace(|| {
            let vars: Vec<ExprId> = (0..args.len()).map(|i| ExprId::var(i as u16)).collect();
            f(&vars)
        });
        self.push_comp(Comp {
            out: out.to_vec(),
            tuple: 0,
            red: red.to_vec(),
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
            for (f, &n) in a.view.forms.iter().zip(src) {
                let (lo, hi) = f.range(&space);
                if lo < 0 || hi >= n as i64 {
                    return err(format!("arg {i}: view reaches {lo}..={hi}, dim is {n}"));
                }
            }
        }
        let width = numel(&comp.out[comp.out.len() - comp.tuple..]);
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
        out[axis] = 1;
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
    pub fn broadcast(&mut self, x: TensorId, shape: &[usize]) -> Result<TensorId, ShapeError> {
        let src = self.shape(x).to_vec();
        if src.len() != shape.len() {
            return err(format!("broadcast {src:?} -> {shape:?}: rank differs"));
        }
        let r = shape.len();
        let picks: Vec<Option<usize>> = src
            .iter()
            .zip(shape)
            .enumerate()
            .map(|(d, (&s, &t))| match (s == t, s == 1) {
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
        let shape: Vec<usize> = perm.iter().map(|&p| src[p]).collect();
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
    pub fn materialized(&self, outputs: &[TensorId]) -> Vec<(TensorId, Vec<usize>)> {
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

    /// Does `m_arg` (as read by `c`) hold `max` over `c`'s reduce coord of
    /// exactly what `c` reads through `s_arg`?
    fn is_row_max_of(&self, c: &Comp, s_arg: &Arg, m_arg: &Arg) -> bool {
        let Some(mc) = self.comp_at(m_arg.src.n()) else {
            return false;
        };
        let n_out = c.points().len();
        let structural = mc.monoid.as_ref().map(Monoid::kind) == Some(MonoidKind::Max)
            && mc.tuple == 0
            && mc.args.len() == 1
            && mc.red.len() == 1
            && c.red.len() == 1
            && mc.body.node(mc.body_outs[0]) == Node::Var(0)
            && !m_arg.view.depends_on(n_out);
        if !structural {
            return false;
        }
        // Extend c's view of m with "m's reduce coord = c's reduce coord",
        // then push m's own view of s through it: must land on c's view of s.
        let mut ext = m_arg.view.clone();
        ext.forms.push(Form::unit(c.rank(), n_out));
        let space = c.space();
        let via_m = mc.args[0].view.compose(&ext, c.rank()).normalized(&space);
        via_m == s_arg.view.normalized(&space)
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
            g: Option<ExprId>,
        }
        let is_output = |i: usize| outputs.iter().any(|o| o.n() == i);
        // max node -> members
        let mut groups: HashMap<usize, Vec<Member>> = HashMap::new();
        for i in 0..self.nodes.len() {
            let Some(c) = self.comp_at(i) else { continue };
            let sum = c.monoid.as_ref().map(Monoid::kind) == Some(MonoidKind::Sum);
            if !sum || c.red.len() != 1 || c.tuple != 0 || is_output(i) {
                continue;
            }
            let Some((s_var, m_var, g)) = match_exp_diff(&c.body, c.body_outs[0]) else {
                continue;
            };
            let (s_arg, m_arg) = (&c.args[s_var as usize], &c.args[m_var as usize]);
            if self.is_row_max_of(c, s_arg, m_arg) {
                groups.entry(m_arg.src.n()).or_default().push(Member {
                    node: i,
                    m_view: m_arg.view.clone(),
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
        for (m_node, members) in groups {
            let mc = self.comp_at(m_node).unwrap();
            let m_out = mc.out.len();
            let fused_rank = m_out + 1;
            let mut body = ExprGraph::new();
            let mut args = vec![mc.args[0].clone()];
            let mut outs = vec![body.var(0)];
            let mut projections = Vec::new();

            for mem in &members {
                let c = self.comp_at(mem.node).unwrap();
                let n_out = c.points().len();
                // member out dim -> which m out dim reads it (None: extra dim)
                let reads: Vec<Option<usize>> = (0..n_out)
                    .map(|d| mem.m_view.forms.iter().position(|f| f.as_unit() == Some(d)))
                    .collect();
                let extra: Vec<usize> = (0..n_out).filter(|&d| reads[d].is_none()).collect();
                let extra_shape: Vec<usize> = extra.iter().map(|&d| c.out[d]).collect();
                let base = outs.len() as i64;

                // unroll the extra dims into tuple slots
                for e in indices(&extra_shape) {
                    let forms = (0..c.rank())
                        .map(|d| match reads.get(d) {
                            Some(Some(j)) => Form::unit(fused_rank, *j),
                            Some(None) => {
                                let k = extra.iter().position(|x| *x == d).unwrap();
                                Form::constant(fused_rank, e[k] as i64)
                            }
                            None => Form::unit(fused_rank, m_out),
                        })
                        .collect();
                    let inv = View { forms };
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
                }

                // member[o] == fused[m_view(o), base + flat(extra coords)]
                let strides = {
                    let mut s = vec![1i64; extra.len()];
                    for k in (0..extra.len().saturating_sub(1)).rev() {
                        s[k] = s[k + 1] * extra_shape[k + 1] as i64;
                    }
                    s
                };
                let mut slot = Form::constant(n_out, base);
                for (k, &d) in extra.iter().enumerate() {
                    slot.coefs[d] = strides[k];
                }
                let mut proj: Vec<Form> = mem
                    .m_view
                    .forms
                    .iter()
                    .map(|f| Form {
                        coefs: f.coefs[..n_out].to_vec(),
                        offset: f.offset,
                    })
                    .collect();
                proj.push(slot);
                projections.push((mem.node, View { forms: proj }));
            }

            let width = outs.len();
            let mut out = mc.out.clone();
            out.push(width);
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
            if live[i] && uses[i] == 1 && !is_output(i) && consumer_space[i] <= numel(c.points()) {
                self.nodes[i].inline = true;
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
                for (p, idx) in indices(c.points()).enumerate() {
                    ev.point(i, &idx, &mut buf[p * w..(p + 1) * w]);
                }
                ev.bufs[i] = buf;
            }
        }
        outputs
            .iter()
            .map(|o| {
                indices(&self.nodes[o.n()].shape)
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
            Op::Input(k) => self.inputs[*k][flat_index(&node.shape, idx)],
            Op::Comp(c) => {
                let (pt, tup) = idx.split_at(c.points().len());
                let slot = flat_index(&c.out[c.points().len()..], tup);
                if node.inline {
                    let mut vals = vec![0.0; c.width()];
                    self.point(t.n(), pt, &mut vals);
                    return vals[slot];
                }
                let w = c.width();
                self.bufs[t.n()][flat_index(c.points(), pt) * w + slot]
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
        for red_idx in indices(&c.red) {
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
    (0..numel(shape)).map(move |mut p| {
        let mut idx = vec![0; shape.len()];
        for d in (0..shape.len()).rev() {
            idx[d] = p % shape[d];
            p /= shape[d];
        }
        idx
    })
}

/// Copy the subgraph under `outs` into `dst`, substituting `Var(i)` with `vars[i]`.
fn replay(src: &ExprGraph, outs: &[ExprId], vars: &[ExprId], dst: &mut ExprGraph) -> Vec<ExprId> {
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

/// Match `g · exp(s − m)` (or just `exp(s − m)`) where `s`, `m` are vars.
/// `exp(x)` lowers to `exp2(x · log2e)` and `a − b` to `add(a, neg(b))`.
fn match_exp_diff(g: &ExprGraph, out: ExprId) -> Option<(u16, u16, Option<ExprId>)> {
    let n = g.nodes_slice();
    let at = |e: ExprId| n[e.index() as usize];
    let exp_diff = |e: ExprId| -> Option<(u16, u16)> {
        let Node::Exp2(x) = at(e) else { return None };
        let Node::Mul(d, l) = at(x) else { return None };
        if at(l) != Node::lit(std::f64::consts::LOG2_E) {
            return None;
        }
        let Node::Add(a, nb) = at(d) else { return None };
        let Node::Neg(b) = at(nb) else { return None };
        match (at(a), at(b)) {
            (Node::Var(s), Node::Var(m)) => Some((s, m)),
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
        let dim = shape[1];
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

    /// softmax(q·kᵀ)·v with every intermediate written out.
    fn naive_attention(g: &mut TensorGraph, q: TensorId, k: TensorId, v: TensorId) -> TensorId {
        let n = g.shape(q)[0];
        let dv = g.shape(v)[1];
        let kt = g.transpose(k, &[1, 0]).unwrap();
        let s = g.matmul(q, kt).unwrap(); // [n, n]
        let m = g.reduce(s, 1, Monoid::max()).unwrap(); // [n, 1]
        let mb = g.broadcast(m, &[n, n]).unwrap();
        let p = g.map(&[s, mb], |v| (v[0] - v[1]).exp()).unwrap(); // [n, n]
        let l = g.reduce(p, 1, Monoid::sum()).unwrap(); // [n, 1]
        let o = g.matmul(p, v).unwrap(); // [n, dv]
        let lb = g.broadcast(l, &[n, dv]).unwrap();
        g.map(&[o, lb], |v| v[0] / v[1]).unwrap()
    }

    #[test]
    fn attention_is_derived_not_written() {
        let (n, d, dv) = (16, 4, 4);
        let mut g = TensorGraph::new();
        let q = g.input(&[n, d]);
        let k = g.input(&[n, d]);
        let v = g.input(&[n, dv]);
        let y = naive_attention(&mut g, q, k, v);

        let inputs = [
            ramp(n * d, 0.05, -1.0),
            ramp(n * d, 0.07, 0.3),
            ramp(n * dv, 0.1, -2.0),
        ];
        let want = g.eval(&inputs, &[y]).remove(0);

        // naive: scores and p are materialized n×n
        assert!(g.materialized(&[y]).iter().any(|(_, s)| numel(s) == n * n));

        let (g, outs) = g.substitute_maps(&[y]);
        let (mut g, outs) = g.stabilize_softmax(&outs);
        g.schedule_inline(&outs);

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
        // scores (inlined), online reduce [n, 1, 2+dv], final divide
        assert_eq!(g.count_comps(&outs), 3);
        assert_eq!(mats.len(), 2);

        close(&g.eval(&inputs, &outs)[0], &want);
    }
}
