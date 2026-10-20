//! One training step as one graph: the model's loss and gradients, global
//! gradient-norm clipping, and an AdamW update of every parameter. The
//! schedule lives in a small input, `hyper = [lr, 1 − β1ᵗ, 1 − β2ᵗ]`, so one
//! compiled step serves warmup, cosine and BitNet's two-stage decay.
//!
//! [`Trainer`] runs it on the f64 interpreter: the reference, and the smoke
//! test. The GPU runner (tang-gpu, CUDA on mew) runs the same graph.

use crate::{build, rope_tables, Config, Init, Model, Variant};
use tang::Scalar;
use tang_compile::tensor::{Arg, Monoid, Op, TensorId, View};
use tang_expr::{with_graph, ExprId};

/// AdamW, BitNet-style defaults.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hyper {
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    /// global gradient-norm clip
    pub clip: f64,
}

impl Default for Hyper {
    fn default() -> Self {
        Hyper {
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay: 0.1,
            clip: 1.0,
        }
    }
}

/// A parameter's optimizer state and its values after the step.
#[derive(Clone, Copy, Debug)]
pub struct Slot {
    pub m: TensorId,
    pub v: TensorId,
    pub p2: TensorId,
    pub m2: TensorId,
    pub v2: TensorId,
}

pub struct Step {
    pub model: Model,
    pub hyper_in: TensorId,
    pub slots: Vec<Slot>,
    /// the global gradient norm before clipping, `[1]`
    pub grad_norm: TensorId,
    /// the running sums of squared gradients: compile barriers, so fusion
    /// doesn't read every gradient in one kernel (wgpu allows 16 bindings)
    pub norm_partials: Vec<TensorId>,
    pub hyper: Hyper,
}

fn lit(x: f64) -> ExprId {
    ExprId::from_f64(x)
}

fn all(rank: usize) -> Vec<Option<usize>> {
    (0..rank).map(Some).collect()
}

impl Step {
    /// The model for `(c, v)` plus clipping and AdamW. `weight_decay` applies
    /// only to parameters marked `decay` (matrices, not norms or DyT scalars);
    /// pass `stage2 = true` to build BitNet's second stage, which drops it.
    pub fn new(c: Config, v: Variant, h: Hyper, stage2: bool) -> Step {
        let mut model = build(c, v);
        let g = &mut model.graph;
        let hyper_in = g.input(&[3usize]);
        // global norm: Σ over every gradient of Σ g²
        let mut sq: Option<TensorId> = None;
        let mut norm_partials = Vec::new();
        for &gr in &model.grads {
            let shape = g.shape(gr).to_vec();
            let n = tang_compile::tensor::numel(&shape);
            let flat = g.reshape(gr, &[n]).unwrap();
            let s = g
                .comp(
                    &[1usize],
                    &[n],
                    vec![Arg {
                        src: flat,
                        view: View::pick(2, &[Some(1)]),
                    }],
                    Some(Monoid::sum()),
                    |x| vec![x[0] * x[0]],
                )
                .unwrap();
            sq = Some(match sq {
                Some(acc) => g.map(&[acc, s], |x| x[0] + x[1]).unwrap(),
                None => s,
            });
            norm_partials.extend(sq);
        }
        let sq = sq.expect("a model has parameters");
        let grad_norm = g.map(&[sq], |x| x[0].sqrt()).unwrap();
        let clip = h.clip;
        let scale = g
            .map(&[grad_norm], |x| {
                // min(1, clip / norm), branch-free
                let r = lit(clip) / (x[0] + lit(1e-12));
                let (cond, one) = (lit(1.0) - r, lit(1.0));
                with_graph(|eg| eg.select(cond, r, one))
            })
            .unwrap();
        let mut slots = Vec::new();
        let (b1, b2, eps) = (h.beta1, h.beta2, h.eps);
        for (p, &gr) in model.params.iter().zip(&model.grads) {
            let shape = g.shape(p.id).to_vec();
            let rank = shape.len();
            let m = g.input(&shape);
            let vv = g.input(&shape);
            let id = |t: TensorId| Arg {
                src: t,
                view: View::pick(rank, &all(rank)),
            };
            let at = |t: TensorId, k: i64| {
                let mut view = View::pick(rank, &[None; 1]);
                view.forms[0].offset = k;
                Arg { src: t, view }
            };
            let m2 = g
                .comp(&shape, &[], vec![id(m), id(gr), at(scale, 0)], None, |x| {
                    vec![x[0] * lit(b1) + x[1] * x[2] * lit(1.0 - b1)]
                })
                .unwrap();
            let v2 = g
                .comp(&shape, &[], vec![id(vv), id(gr), at(scale, 0)], None, |x| {
                    let gc = x[1] * x[2];
                    vec![x[0] * lit(b2) + gc * gc * lit(1.0 - b2)]
                })
                .unwrap();
            let wd = if p.decay && !stage2 {
                h.weight_decay
            } else {
                0.0
            };
            let p2 = g
                .comp(
                    &shape,
                    &[],
                    vec![
                        id(p.id),
                        id(m2),
                        id(v2),
                        at(hyper_in, 0),
                        at(hyper_in, 1),
                        at(hyper_in, 2),
                    ],
                    None,
                    |x| {
                        let u = (x[1] / x[4]) / ((x[2] / x[5]).sqrt() + lit(eps));
                        vec![x[0] - x[3] * (u + lit(wd) * x[0])]
                    },
                )
                .unwrap();
            slots.push(Slot {
                m,
                v: vv,
                p2,
                m2,
                v2,
            });
        }
        Step {
            model,
            hyper_in,
            slots,
            grad_norm,
            norm_partials,
            hyper: h,
        }
    }

    /// `[loss, grad_norm, p2…, m2…, v2…]`.
    pub fn outputs(&self) -> Vec<TensorId> {
        let mut o = vec![self.model.loss, self.grad_norm];
        o.extend(self.slots.iter().map(|s| s.p2));
        o.extend(self.slots.iter().map(|s| s.m2));
        o.extend(self.slots.iter().map(|s| s.v2));
        o
    }

    pub fn input_index(&self, t: TensorId) -> usize {
        match self.model.graph.nodes()[t.index()].op {
            Op::Input(k) => k,
            _ => panic!("not an input"),
        }
    }
}

/// Deterministic initial values.
pub struct Rng(pub u64);

impl Rng {
    /// Uniform in [−1, 1).
    pub fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    /// Standard normal (Box–Muller).
    pub fn normal(&mut self) -> f64 {
        let (u1, u2) = ((self.uniform() + 1.0) / 2.0, (self.uniform() + 1.0) / 2.0);
        (-2.0 * (u1.max(1e-300)).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// The step on the f64 interpreter, holding every input's current value.
pub struct Trainer {
    pub step: Step,
    pub values: Vec<Vec<f64>>,
    pub t: u64,
}

impl Trainer {
    pub fn new(step: Step, seed: u64) -> Trainer {
        let g = &step.model.graph;
        let n = g
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Input(_)))
            .count();
        let mut values = vec![Vec::new(); n];
        let mut rng = Rng(seed);
        let c = step.model.config;
        values[step.input_index(step.model.tokens)] = vec![0.0; c.rows()];
        values[step.input_index(step.model.targets)] = vec![0.0; c.rows()];
        if let Some((cos, sin)) = step.model.rope {
            let (cv, sv) = rope_tables(&c);
            values[step.input_index(cos)] = cv;
            values[step.input_index(sin)] = sv;
        }
        for (p, s) in step.model.params.iter().zip(&step.slots) {
            let len: usize = p.shape.iter().product();
            values[step.input_index(p.id)] = match p.init {
                Init::Normal(sd) => (0..len).map(|_| rng.normal() * sd).collect(),
                Init::Const(v) => vec![v; len],
            };
            values[step.input_index(s.m)] = vec![0.0; len];
            values[step.input_index(s.v)] = vec![0.0; len];
        }
        values[step.input_index(step.hyper_in)] = vec![0.0; 3];
        Trainer { step, values, t: 0 }
    }

    /// One optimizer step on `(tokens, targets)` at learning rate `lr`.
    /// Returns `(loss, grad_norm)`.
    pub fn step(&mut self, tokens: &[u32], targets: &[u32], lr: f64) -> (f64, f64) {
        self.t += 1;
        let s = &self.step;
        let h = s.hyper;
        let tk = s.input_index(s.model.tokens);
        let tg = s.input_index(s.model.targets);
        self.values[tk] = tokens.iter().map(|&x| x as f64).collect();
        self.values[tg] = targets.iter().map(|&x| x as f64).collect();
        let hi = s.input_index(s.hyper_in);
        self.values[hi] = vec![
            lr,
            1.0 - h.beta1.powi(self.t as i32),
            1.0 - h.beta2.powi(self.t as i32),
        ];
        let out = s.model.graph.eval(&self.values, &s.outputs());
        let np = s.slots.len();
        for (i, (p, sl)) in s.model.params.iter().zip(&s.slots).enumerate() {
            self.values[s.input_index(p.id)] = out[2 + i].clone();
            self.values[s.input_index(sl.m)] = out[2 + np + i].clone();
            self.values[s.input_index(sl.v)] = out[2 + 2 * np + i].clone();
        }
        (out[0][0], out[1][0])
    }
}
