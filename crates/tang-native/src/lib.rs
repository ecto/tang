//! FHE-native language models and their standard twin, built as one tang
//! [`TensorGraph`]: the forward pass, a cross-entropy loss and the backward
//! pass (and, in [`train`], an AdamW step) for one batch of sequences.
//!
//! Four switches turn the standard twin into the FHE-native model, one
//! constraint at a time, each chosen because it is cheap under encryption
//! (see ryng's `docs/fhe-native-model.md`):
//! - **ternary**: weights quantized to {−1, 0, 1}·s by absmean, trained
//!   through a straight-through estimator ([`TensorGraph::stop_grad`]).
//!   One activation plane, half the encrypted matmul cost.
//! - **dyt**: DyT, γ·tanh(αx) + β, in place of RMSNorm. One lookup per value,
//!   no data-dependent division.
//! - **relu2**: up, squared ReLU, down in place of SwiGLU. One lookup, no
//!   ciphertext × ciphertext gate.
//! - **retention**: (φ(Q)φ(K)ᵀ ⊙ D)V with φ = squared ReLU and a fixed
//!   per-head decay D[h,t,s] = γ_h^(t−s), then DyT, in place of softmax
//!   attention with RoPE. No exp, no division, constant cost per token.
//!
//! Attention is full multi-head (grouped KV indexing isn't an affine view),
//! the same in both. Activations are not fake-quantized while training: the
//! FHE-faithful eval quantizes them, and on BitNet ±112 cost +0.34%.

#[cfg(feature = "gpu")]
pub mod gpu;
pub mod train;

use tang::Scalar;
use tang_compile::tensor::{Arg, Monoid, TensorGraph, TensorId, View};
use tang_expr::{with_graph, ExprId};

/// Which constraints are on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Variant {
    pub ternary: bool,
    pub dyt: bool,
    pub relu2: bool,
    pub retention: bool,
}

impl Variant {
    pub const TWIN: Variant = Variant {
        ternary: false,
        dyt: false,
        relu2: false,
        retention: false,
    };
    pub const NATIVE: Variant = Variant {
        ternary: true,
        dyt: true,
        relu2: true,
        retention: true,
    };

    /// `twin`, `ternary`, `dyt`, `relu2` or `native`: the ablation ladder,
    /// each adding one constraint to the one before.
    pub fn parse(s: &str) -> Option<Variant> {
        let t = Variant::TWIN;
        Some(match s {
            "twin" => t,
            "ternary" => Variant { ternary: true, ..t },
            "dyt" => Variant {
                ternary: true,
                dyt: true,
                ..t
            },
            "relu2" => Variant {
                retention: false,
                ..Variant::NATIVE
            },
            "native" => Variant::NATIVE,
            _ => return None,
        })
    }
}

/// Model shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    pub vocab: usize,
    pub d: usize,
    pub layers: usize,
    pub heads: usize,
    /// MLP width for squared ReLU; SwiGLU uses 2/3 of it so the two match
    /// in parameters
    pub ff: usize,
    /// tokens per sequence
    pub seq: usize,
    /// sequences per step
    pub batch: usize,
}

impl Config {
    /// Token rows per step: batch × seq.
    pub fn rows(&self) -> usize {
        self.batch * self.seq
    }

    pub fn head_dim(&self) -> usize {
        self.d / self.heads
    }

    fn ff_for(&self, v: Variant) -> usize {
        if v.relu2 {
            self.ff
        } else {
            (2 * self.ff / 3).max(1)
        }
    }

    /// The spec's pilot: 125M-class.
    pub fn pilot(vocab: usize) -> Config {
        Config {
            vocab,
            d: 768,
            layers: 12,
            heads: 12,
            ff: 3072,
            seq: 512,
            batch: 8,
        }
    }

    /// The spec's main run: ~0.48B with a 128k vocabulary.
    pub fn main(vocab: usize) -> Config {
        Config {
            vocab,
            d: 1024,
            layers: 28,
            heads: 8,
            ff: 4096,
            seq: 1024,
            batch: 4,
        }
    }
}

/// How a parameter starts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Init {
    Normal(f64),
    Const(f64),
}

#[derive(Clone, Debug)]
pub struct Param {
    pub name: String,
    pub id: TensorId,
    pub shape: Vec<usize>,
    pub init: Init,
    /// matrices get weight decay; norms and DyT scalars don't
    pub decay: bool,
}

/// The built graph: inputs, parameters, loss, gradients.
pub struct Model {
    pub graph: TensorGraph,
    pub config: Config,
    pub variant: Variant,
    /// token ids, as f32/f64 values, `[batch·seq]`, sequence-major
    pub tokens: TensorId,
    /// next-token ids, `[batch·seq]`
    pub targets: TensorId,
    /// RoPE tables `[seq, head_dim/2]` (twin attention only)
    pub rope: Option<(TensorId, TensorId)>,
    pub params: Vec<Param>,
    pub logits: TensorId,
    /// mean cross-entropy, `[1]`
    pub loss: TensorId,
    /// d loss / d param, in `params` order
    pub grads: Vec<TensorId>,
    /// the residual stream after each block, then the loss gradient wrt
    /// each: compile barriers, so fusion never splices one block's sum (or
    /// its gradient's) into every later reader
    pub boundaries: Vec<TensorId>,
}

fn lit(x: f64) -> ExprId {
    ExprId::from_f64(x)
}

fn select(c: ExprId, a: ExprId, b: ExprId) -> ExprId {
    with_graph(|g| g.select(c, a, b))
}

fn arg(src: TensorId, rank: usize, picks: &[Option<usize>]) -> Arg {
    Arg {
        src,
        view: View::pick(rank, picks),
    }
}

struct B<'a> {
    g: &'a mut TensorGraph,
    params: Vec<Param>,
    v: Variant,
}

impl B<'_> {
    fn param(&mut self, name: String, shape: &[usize], init: Init, decay: bool) -> TensorId {
        let id = self.g.input(shape);
        self.params.push(Param {
            name,
            id,
            shape: shape.to_vec(),
            init,
            decay,
        });
        id
    }

    /// Absmean ternary quantization with a straight-through estimator:
    /// forward round(w / s) clamped to {−1, 0, 1}, times s = mean|w|;
    /// gradient the identity.
    fn ternary(&mut self, w: TensorId) -> TensorId {
        let g = &mut *self.g;
        let shape = g.shape(w).to_vec();
        let (o, i) = (
            tang_compile::tensor::numel(&shape[..1]),
            tang_compile::tensor::numel(&shape[1..]),
        );
        let n = (o * i) as f64;
        let s = g
            .comp(
                &[1usize],
                &[o, i],
                vec![arg(w, 3, &[Some(1), Some(2)])],
                Some(Monoid::sum()),
                |v| vec![v[0].abs() / lit(n)],
            )
            .unwrap();
        let q = g
            .comp(
                &[o, i],
                &[],
                vec![arg(w, 2, &[Some(0), Some(1)]), arg(s, 2, &[None])],
                None,
                |v| {
                    let sc = v[1] + lit(1e-8);
                    let u = v[0] / sc;
                    let t = select(
                        u - lit(0.5),
                        lit(1.0),
                        select(lit(-0.5) - u, lit(-1.0), lit(0.0)),
                    );
                    vec![t * sc]
                },
            )
            .unwrap();
        let diff = g.map(&[q, w], |v| v[0] - v[1]).unwrap();
        let stopped = g.stop_grad(diff).unwrap();
        g.map(&[w, stopped], |v| v[0] + v[1]).unwrap()
    }

    /// x `[T, i]` · wᵀ, w `[o, i]`.
    fn linear(&mut self, x: TensorId, name: String, o: usize) -> TensorId {
        let (t, i) = {
            let s = self.g.shape(x);
            (
                tang_compile::tensor::numel(&s[..1]),
                tang_compile::tensor::numel(&s[1..]),
            )
        };
        let w = self.param(name, &[o, i], Init::Normal(1.0 / (i as f64).sqrt()), true);
        let w = if self.v.ternary { self.ternary(w) } else { w };
        self.g
            .comp(
                &[t, o],
                &[i],
                vec![
                    arg(x, 3, &[Some(0), Some(2)]),
                    arg(w, 3, &[Some(1), Some(2)]),
                ],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[1]],
            )
            .unwrap()
    }

    fn rmsnorm(&mut self, x: TensorId, name: String) -> TensorId {
        let s = self.g.shape(x).to_vec();
        let (t, d) = (
            tang_compile::tensor::numel(&s[..1]),
            tang_compile::tensor::numel(&s[1..]),
        );
        let w = self.param(name, &[d], Init::Const(1.0), false);
        let g = &mut *self.g;
        let ms = g
            .comp(
                &[t],
                &[d],
                vec![arg(x, 2, &[Some(0), Some(1)])],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[0] / lit(d as f64)],
            )
            .unwrap();
        g.comp(
            &[t, d],
            &[],
            vec![
                arg(x, 2, &[Some(0), Some(1)]),
                arg(ms, 2, &[Some(0)]),
                arg(w, 2, &[Some(1)]),
            ],
            None,
            |v| vec![v[0] / (v[1] + lit(1e-6)).sqrt() * v[2]],
        )
        .unwrap()
    }

    /// DyT: γ·tanh(αx) + β, with α a learned scalar.
    fn dyt(&mut self, x: TensorId, name: String, alpha0: f64) -> TensorId {
        let s = self.g.shape(x).to_vec();
        let (t, d) = (
            tang_compile::tensor::numel(&s[..1]),
            tang_compile::tensor::numel(&s[1..]),
        );
        let a = self.param(format!("{name}.alpha"), &[1], Init::Const(alpha0), false);
        let gm = self.param(format!("{name}.gamma"), &[d], Init::Const(1.0), false);
        let bt = self.param(format!("{name}.beta"), &[d], Init::Const(0.0), false);
        self.g
            .comp(
                &[t, d],
                &[],
                vec![
                    arg(x, 2, &[Some(0), Some(1)]),
                    arg(a, 2, &[None]),
                    arg(gm, 2, &[Some(1)]),
                    arg(bt, 2, &[Some(1)]),
                ],
                None,
                |v| vec![v[2] * (v[1] * v[0]).tanh() + v[3]],
            )
            .unwrap()
    }

    fn norm(&mut self, x: TensorId, name: String, alpha0: f64) -> TensorId {
        if self.v.dyt {
            self.dyt(x, name, alpha0)
        } else {
            self.rmsnorm(x, name)
        }
    }
}

fn relu2(x: ExprId) -> ExprId {
    select(x, x * x, lit(0.0))
}

/// Rotate (even, odd) pairs of `x` `[B, T, H, hd]` by RoPE tables `[T, hd/2]`.
fn rope(g: &mut TensorGraph, x: TensorId, cos: TensorId, sin: TensorId, c: &Config) -> TensorId {
    let (bt, t, h, hd) = (c.batch, c.seq, c.heads, c.head_dim());
    let half = hd / 2;
    let x5 = g.reshape(x, &[bt, t, h, half, 2]).unwrap();
    let at = |parity: i64| {
        let mut v = View::pick(5, &[Some(0), Some(1), Some(2), Some(3), None]);
        v.forms[4].offset = parity;
        Arg { src: x5, view: v }
    };
    let par = g.iota(&[bt, t, h, half, 2], 4);
    let out = g
        .comp(
            &[bt, t, h, half, 2],
            &[],
            vec![
                at(0),
                at(1),
                arg(cos, 5, &[Some(1), Some(3)]),
                arg(sin, 5, &[Some(1), Some(3)]),
                arg(par, 5, &[Some(0), Some(1), Some(2), Some(3), Some(4)]),
            ],
            None,
            |v| {
                let (x0, x1, co, si) = (v[0], v[1], v[2], v[3]);
                vec![select(v[4], x1 * co + x0 * si, x0 * co - x1 * si)]
            },
        )
        .unwrap();
    g.reshape(out, &[bt, t, h, hd]).unwrap()
}

/// `[B, H, T, T]` scores Σ_e a[b,t,h,e]·c[b,s,h,e]·scale, a and c `[B, T, H, hd]`.
fn scores(g: &mut TensorGraph, a: TensorId, c: TensorId, cf: &Config, scale: f64) -> TensorId {
    let (bt, t, h, hd) = (cf.batch, cf.seq, cf.heads, cf.head_dim());
    g.comp(
        &[bt, h, t, t],
        &[hd],
        vec![
            arg(a, 5, &[Some(0), Some(2), Some(1), Some(4)]),
            arg(c, 5, &[Some(0), Some(3), Some(1), Some(4)]),
        ],
        Some(Monoid::sum()),
        |x| vec![x[0] * x[1] * lit(scale)],
    )
    .unwrap()
}

/// Reduce the last axis of a `[B, H, T, T]` tensor to `[B, H, T]`.
fn last_axis(g: &mut TensorGraph, x: TensorId, cf: &Config, m: Monoid) -> TensorId {
    let (bt, t, h) = (cf.batch, cf.seq, cf.heads);
    g.comp(
        &[bt, h, t],
        &[t],
        vec![arg(x, 4, &[Some(0), Some(1), Some(2), Some(3)])],
        Some(m),
        |z| vec![z[0]],
    )
    .unwrap()
}

/// `[B, H, T, T]` with a `[B, H, T]` value per row, combined elementwise.
fn per_row(
    g: &mut TensorGraph,
    x: TensorId,
    r: TensorId,
    cf: &Config,
    f: impl FnOnce(ExprId, ExprId) -> ExprId,
) -> TensorId {
    let (bt, t, h) = (cf.batch, cf.seq, cf.heads);
    g.comp(
        &[bt, h, t, t],
        &[],
        vec![
            arg(x, 4, &[Some(0), Some(1), Some(2), Some(3)]),
            arg(r, 4, &[Some(0), Some(1), Some(2)]),
        ],
        None,
        |z| vec![f(z[0], z[1])],
    )
    .unwrap()
}

/// Build the model's forward, loss and backward for one step of
/// `batch` sequences of `seq` tokens.
pub fn build(c: Config, v: Variant) -> Model {
    assert!(
        c.d.is_multiple_of(c.heads) && c.head_dim().is_multiple_of(2),
        "head_dim even"
    );
    let mut graph = TensorGraph::new();
    let (t, d, h, hd, vocab) = (c.seq, c.d, c.heads, c.head_dim(), c.vocab);
    let (bt, rows) = (c.batch, c.rows());
    let tokens = graph.input(&[rows]);
    let targets = graph.input(&[rows]);
    let rope_tables =
        (!v.retention).then(|| (graph.input(&[t, hd / 2]), graph.input(&[t, hd / 2])));
    let mut b = B {
        g: &mut graph,
        params: Vec::new(),
        v,
    };
    let emb = b.param("embed".into(), &[vocab, d], Init::Normal(0.02), true);
    // embedding as a one-hot sum: x[r, e] = Σ_v [v = tok_r]·E[v, e]
    let viota = b.g.iota(&[vocab], 0);
    let mut x =
        b.g.comp(
            &[rows, d],
            &[vocab],
            vec![
                arg(tokens, 3, &[Some(0)]),
                arg(viota, 3, &[Some(2)]),
                arg(emb, 3, &[Some(2), Some(1)]),
            ],
            Some(Monoid::sum()),
            |v| vec![select(lit(0.5) - (v[0] - v[1]).abs(), v[2], lit(0.0))],
        )
        .unwrap();
    let ih = b.g.iota(&[bt, h, t, t], 1);
    let it = b.g.iota(&[bt, h, t, t], 2);
    let is = b.g.iota(&[bt, h, t, t], 3);
    let mut boundaries = Vec::new();
    for l in 0..c.layers {
        // attention
        let n = b.norm(x, format!("l{l}.attn_norm"), 0.5);
        let q = b.linear(n, format!("l{l}.wq"), d);
        let k = b.linear(n, format!("l{l}.wk"), d);
        let vv = b.linear(n, format!("l{l}.wv"), d);
        let g = &mut *b.g;
        let (q4, k4, v4) = (
            g.reshape(q, &[bt, t, h, hd]).unwrap(),
            g.reshape(k, &[bt, t, h, hd]).unwrap(),
            g.reshape(vv, &[bt, t, h, hd]).unwrap(),
        );
        let p = if v.retention {
            let fq = g.map(&[q4], |x| relu2(x[0])).unwrap();
            let fk = g.map(&[k4], |x| relu2(x[0])).unwrap();
            let s = scores(g, fq, fk, &c, 1.0 / hd as f64);
            // D[b,h,t,s] = γ_h^(t−s) for s ≤ t, γ_h = 1 − 2^(−5 − 7h/(H−1))
            let span = if h > 1 { 7.0 / (h - 1) as f64 } else { 0.0 };
            g.map(&[s, ih, it, is], |x| {
                let lg = (lit(1.0) - (lit(-5.0) - x[1] * lit(span)).exp2_()).ln()
                    / lit(std::f64::consts::LN_2);
                let dt = x[2] - x[3];
                select(dt + lit(0.5), x[0] * (dt * lg).exp2_(), lit(0.0))
            })
            .unwrap()
        } else {
            let (cos, sin) = rope_tables.expect("twin attention has RoPE tables");
            let (rq, rk) = (rope(g, q4, cos, sin, &c), rope(g, k4, cos, sin, &c));
            let s = scores(g, rq, rk, &c, 1.0 / (hd as f64).sqrt());
            let masked = g
                .map(&[s, it, is], |x| select(x[2] - x[1], lit(-1e30), x[0]))
                .unwrap();
            let mx = last_axis(g, masked, &c, Monoid::max());
            let e = per_row(g, masked, mx, &c, |a, m| (a - m).exp());
            let den = last_axis(g, e, &c, Monoid::sum());
            per_row(g, e, den, &c, |a, s| a / s)
        };
        // o[b,t,h,e] = Σ_s P[b,h,t,s]·V[b,s,h,e]
        let o = g
            .comp(
                &[bt, t, h, hd],
                &[t],
                vec![
                    arg(p, 5, &[Some(0), Some(2), Some(1), Some(4)]),
                    arg(v4, 5, &[Some(0), Some(4), Some(2), Some(3)]),
                ],
                Some(Monoid::sum()),
                |x| vec![x[0] * x[1]],
            )
            .unwrap();
        let o = g.reshape(o, &[rows, d]).unwrap();
        let o = if v.retention {
            b.dyt(o, format!("l{l}.ret_norm"), 0.5)
        } else {
            o
        };
        let a = b.linear(o, format!("l{l}.wo"), d);
        x = b.g.map(&[x, a], |z| z[0] + z[1]).unwrap();
        // MLP
        let n = b.norm(x, format!("l{l}.mlp_norm"), 0.2);
        let ff = c.ff_for(v);
        let m = if v.relu2 {
            let u = b.linear(n, format!("l{l}.up"), ff);
            b.g.map(&[u], |z| relu2(z[0])).unwrap()
        } else {
            let gt = b.linear(n, format!("l{l}.gate"), ff);
            let u = b.linear(n, format!("l{l}.up"), ff);
            b.g.map(&[gt, u], |z| z[0] / (lit(1.0) + (-z[0]).exp()) * z[1])
                .unwrap()
        };
        let dn = b.linear(m, format!("l{l}.down"), d);
        x = b.g.map(&[x, dn], |z| z[0] + z[1]).unwrap();
        boundaries.push(x);
    }
    let x = b.norm(x, "final_norm".into(), 0.5);
    let params = std::mem::take(&mut b.params);
    let g = &mut graph;
    // tied head: logits = x · Eᵀ
    let logits = g
        .comp(
            &[rows, vocab],
            &[d],
            vec![
                arg(x, 3, &[Some(0), Some(2)]),
                arg(emb, 3, &[Some(1), Some(2)]),
            ],
            Some(Monoid::sum()),
            |z| vec![z[0] * z[1]],
        )
        .unwrap();
    let mx = g
        .comp(
            &[rows],
            &[vocab],
            vec![arg(logits, 2, &[Some(0), Some(1)])],
            Some(Monoid::max()),
            |z| vec![z[0]],
        )
        .unwrap();
    let mxs = g.stop_grad(mx).unwrap(); // the max shift is a constant to the loss
    let sum = g
        .comp(
            &[rows],
            &[vocab],
            vec![arg(logits, 2, &[Some(0), Some(1)]), arg(mxs, 2, &[Some(0)])],
            Some(Monoid::sum()),
            |z| vec![(z[0] - z[1]).exp()],
        )
        .unwrap();
    let tgt = g
        .comp(
            &[rows],
            &[vocab],
            vec![
                arg(logits, 2, &[Some(0), Some(1)]),
                arg(targets, 2, &[Some(0)]),
                arg(viota, 2, &[Some(1)]),
            ],
            Some(Monoid::sum()),
            |z| vec![select(lit(0.5) - (z[1] - z[2]).abs(), z[0], lit(0.0))],
        )
        .unwrap();
    let loss = g
        .comp(
            &[1usize],
            &[rows],
            vec![
                arg(sum, 2, &[Some(1)]),
                arg(mxs, 2, &[Some(1)]),
                arg(tgt, 2, &[Some(1)]),
            ],
            Some(Monoid::sum()),
            |z| vec![(z[0].ln() + z[1] - z[2]) / lit(rows as f64)],
        )
        .unwrap();
    // differentiate wrt the residual stream too: its gradients are the
    // backward pass's own chain of adds, so they become barriers as well
    let ids: Vec<TensorId> = params
        .iter()
        .map(|p| p.id)
        .chain(boundaries.iter().copied())
        .collect();
    let mut grads = g.backward(loss, &ids).unwrap();
    let dboundaries = grads.split_off(params.len());
    boundaries.extend(dboundaries);
    Model {
        graph,
        config: c,
        variant: v,
        tokens,
        targets,
        rope: rope_tables,
        params,
        logits,
        loss,
        grads,
        boundaries,
    }
}

/// RoPE tables for `seq` positions: `(cos, sin)`, each `[seq, head_dim/2]`.
pub fn rope_tables(c: &Config) -> (Vec<f64>, Vec<f64>) {
    let (t, half) = (c.seq, c.head_dim() / 2);
    let mut cos = Vec::with_capacity(t * half);
    let mut sin = Vec::with_capacity(t * half);
    for p in 0..t {
        for i in 0..half {
            let f = p as f64 / 10000f64.powf(2.0 * i as f64 / c.head_dim() as f64);
            cos.push(f.cos());
            sin.push(f.sin());
        }
    }
    (cos, sin)
}

trait Exp2 {
    fn exp2_(self) -> Self;
}

impl Exp2 for ExprId {
    fn exp2_(self) -> Self {
        with_graph(|g| g.exp2(self))
    }
}

#[cfg(test)]
mod tests;
