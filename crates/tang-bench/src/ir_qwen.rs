//! Qwen3 as one tensor-IR graph per step: decode (one token) and
//! prefill (`seq` tokens from an empty cache).
//!
//! A step reads its tokens' embeddings and rope angles from a `step`
//! input, runs every layer against KV caches of fixed capacity `ctx`, and
//! writes the last token's logits and the caches' new rows. Weights are
//! MLX 4-bit: per projection a `[rows, cols/64, 64]` tensor of nibbles
//! (logical values 0..=15) and bf16 `[rows, cols/64]` scales and biases.
//! The dequantization `q·s + b` is a map the matmul reads: in decode it
//! is inlined and evaluated where the packed words are loaded
//! (`to_cuda_decode`, `Packing::U4`); in prefill, where each weight is
//! read by `seq` tokens, it stays materialized and the matmul is a
//! tensor-core tile over it.
//!
//! Attention masks key positions past each query's position with
//! `select(pos − j + ½, …, identity)` in every reduction over the cache
//! axis; `to_cuda_decode` turns those into loop bounds, so a step reads
//! `len` rows of cache, not `ctx`. The cache updates are maps
//! `select(len − seq ≤ j < len, new, old)` that overwrite the cache in
//! place.
use tang::Scalar;
use tang_compile::tensor::cuda_gemv::{DecodeOpts, Packing};
use tang_compile::tensor::{Arg, Form, Monoid, TensorGraph, TensorId, View};
use tang_expr::ExprId;

/// Quantization group: one scale and bias per 64 weights of a row.
pub const GROUP: usize = 64;

#[derive(Clone, Debug)]
pub struct QwenDims {
    pub d_model: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub ff: usize,
    pub vocab: usize,
    pub layers: usize,
    /// KV cache capacity (positions).
    pub ctx: usize,
    pub eps: f64,
    pub theta: f64,
}

impl QwenDims {
    /// Qwen3-8B (`config.json`), with a cache of `ctx` positions.
    pub fn qwen3_8b(ctx: usize) -> Self {
        Self {
            d_model: 4096,
            heads: 32,
            kv_heads: 8,
            head_dim: 128,
            ff: 12288,
            vocab: 151936,
            layers: 36,
            ctx,
            eps: 1e-6,
            theta: 1e6,
        }
    }

    /// Rows of the fused q|k|v projection.
    pub fn qkv_rows(&self) -> usize {
        (self.heads + 2 * self.kv_heads) * self.head_dim
    }

    /// The step input for `seq` tokens: `x [seq, d]`, `cos [seq, hd/2]`,
    /// `sin [seq, hd/2]`, then `len` (positions in the cache after the
    /// step).
    pub fn step_len(&self, seq: usize) -> usize {
        seq * (self.d_model + self.head_dim) + 1
    }
}

/// One graph input: name (MLX tensor name stem), logical shape, packing.
#[derive(Clone, Debug)]
pub struct InputSpec {
    pub name: String,
    pub shape: Vec<usize>,
    pub packing: Option<Packing>,
}

/// Inputs per layer, in order.
pub const LAYER_INPUTS: [&str; 21] = [
    "qkv.q", "qkv.s", "qkv.b", "o.q", "o.s", "o.b", "gate.q", "gate.s", "gate.b", "up.q", "up.s",
    "up.b", "down.q", "down.s", "down.b", "ln1", "ln2", "q_norm", "k_norm", "k_cache", "v_cache",
];

pub struct DecodeGraph {
    pub g: TensorGraph,
    /// Tokens per step.
    pub seq: usize,
    /// `[logits, k_cache'₀, v_cache'₀, k_cache'₁, …]`.
    pub outputs: Vec<TensorId>,
    /// The residual stream after each attention and MLP block, kept
    /// materialized: a residual read once per element would otherwise be
    /// substituted into its readers, and the stream would become a sum
    /// over every earlier layer.
    pub keep: Vec<TensorId>,
    pub inputs: Vec<InputSpec>,
}

fn unit(rank: usize, d: usize) -> Form {
    Form::unit(rank, d)
}

fn form(rank: usize, terms: &[(usize, i64)], offset: i64) -> Form {
    let mut f = Form::constant(rank, offset);
    for &(d, c) in terms {
        f.coefs[d] += c;
    }
    f
}

fn ident(rank: usize) -> Vec<Form> {
    (0..rank).map(|d| unit(rank, d)).collect()
}

fn arg(src: TensorId, forms: Vec<Form>) -> Arg {
    Arg {
        src,
        view: View {
            guard: false,
            forms,
        },
    }
}

fn select(c: ExprId, t: ExprId, f: ExprId) -> ExprId {
    tang_expr::with_graph(|g| g.select(c, t, f))
}

/// Rows of a per-token tensor: a `[seq, n]` comp, or the flat `step`
/// input at `off` with row stride `stride`.
#[derive(Clone, Copy)]
struct Rows {
    src: TensorId,
    off: i64,
    stride: i64,
}

struct Builder {
    g: TensorGraph,
    inputs: Vec<InputSpec>,
}

impl Builder {
    fn input(&mut self, name: String, shape: &[usize], packing: Option<Packing>) -> TensorId {
        self.inputs.push(InputSpec {
            name,
            shape: shape.to_vec(),
            packing,
        });
        self.g.input(shape)
    }

    /// Read row `t` element `k` of `x` (forms over the comp's space).
    fn row(&self, x: Rows, t: Form, k: Form) -> Arg {
        if self.g.shape(x.src).len() == 1 {
            let mut f = k;
            f.offset += x.off;
            for (c, tc) in f.coefs.iter_mut().zip(&t.coefs) {
                *c += x.stride * tc;
            }
            f.offset += x.stride * t.offset;
            arg(x.src, vec![f])
        } else {
            let mut k = k;
            k.offset += x.off;
            arg(x.src, vec![t, k])
        }
    }

    /// A 4-bit projection `[rows, cols]`: nibbles, scales, biases.
    fn q4(&mut self, name: &str, rows: usize, cols: usize) -> [TensorId; 3] {
        let groups = cols / GROUP;
        [
            self.input(
                format!("{name}.q"),
                &[rows, groups, GROUP],
                Some(Packing::U4),
            ),
            self.input(format!("{name}.s"), &[rows, groups], Some(Packing::Bf16)),
            self.input(format!("{name}.b"), &[rows, groups], Some(Packing::Bf16)),
        ]
    }

    /// `y[t, n] = Σ_k x[t, k]·(q[n,k]·s[n,k/64] + b[n,k/64])` over `seq`
    /// rows of `x` starting at row `t0`.
    fn linear(&mut self, x: Rows, seq: usize, t0: i64, w: [TensorId; 3]) -> TensorId {
        let shape = self.g.shape(w[0]).to_vec();
        let (rows, groups) = (shape[0].n(), shape[1].n());
        // dequantization: a map over the nibbles the matmul reads
        let r3 = 3;
        let dq = self
            .g
            .comp(
                &[rows, groups, GROUP],
                &[],
                vec![
                    arg(w[0], ident(3)),
                    arg(w[1], vec![unit(r3, 0), unit(r3, 1)]),
                    arg(w[2], vec![unit(r3, 0), unit(r3, 1)]),
                ],
                None,
                |v| vec![v[0] * v[1] + v[2]],
            )
            .unwrap();
        if seq > 1 {
            // many rows: the weight is read by every one, so the
            // dequantized weight is materialized (seq-fold recompute
            // otherwise) and the matmul runs over one flat k — a
            // tensor-core tile
            let k = groups * GROUP;
            let dq = self.g.reshape(dq, &[rows, k]).unwrap();
            let xa = self.row(x, form(3, &[(0, 1)], t0), unit(3, 2));
            return self
                .g
                .comp(
                    &[seq, rows],
                    &[k],
                    vec![xa, arg(dq, vec![unit(3, 1), unit(3, 2)])],
                    Some(Monoid::sum()),
                    |v| vec![v[0] * v[1]],
                )
                .unwrap();
        }
        // space (t, n, g, j)
        let r4 = 4;
        let xa = self.row(
            x,
            form(r4, &[(0, 1)], t0),
            form(r4, &[(2, GROUP as i64), (3, 1)], 0),
        );
        self.g
            .comp(
                &[seq, rows],
                &[groups, GROUP],
                vec![xa, arg(dq, vec![unit(r4, 1), unit(r4, 2), unit(r4, 3)])],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[1]],
            )
            .unwrap()
    }

    /// RMSNorm of `seq` rows of `x` (from row `t0`, `n` wide) with weight `w`.
    fn rms_norm(
        &mut self,
        x: Rows,
        seq: usize,
        t0: i64,
        n: usize,
        w: TensorId,
        eps: f64,
    ) -> TensorId {
        let xa = self.row(x, form(2, &[(0, 1)], t0), unit(2, 1));
        let ss = self
            .g
            .comp(&[seq], &[n], vec![xa], Some(Monoid::sum()), |v| {
                vec![v[0] * v[0]]
            })
            .unwrap();
        let xa = self.row(x, form(2, &[(0, 1)], t0), unit(2, 1));
        self.g
            .comp(
                &[seq, n],
                &[],
                vec![xa, arg(ss, vec![unit(2, 0)]), arg(w, vec![unit(2, 1)])],
                None,
                |v| {
                    let inv = ExprId::from_f64(1.0)
                        / (v[1] / ExprId::from_f64(n as f64) + ExprId::from_f64(eps)).sqrt();
                    vec![v[0] * inv * v[2]]
                },
            )
            .unwrap()
    }

    /// `a[t, i] + b[t, i]` over `[seq, n]`.
    fn add(&mut self, a: Rows, b: TensorId, seq: usize, n: usize) -> TensorId {
        let aa = self.row(a, unit(2, 0), unit(2, 1));
        self.g
            .comp(&[seq, n], &[], vec![aa, arg(b, ident(2))], None, |v| {
                vec![v[0] + v[1]]
            })
            .unwrap()
    }
}

/// The decode step (one token at position `len − 1`).
pub fn decode_graph(dims: &QwenDims) -> DecodeGraph {
    step_graph(dims, 1)
}

/// A step over `seq` tokens. `seq == 1`: decode at any position.
/// `seq > 1`: prefill of positions `0..seq` (the step's `len` must be
/// `seq`; the cache rows are the tokens').
pub fn step_graph(dims: &QwenDims, seq: usize) -> DecodeGraph {
    let QwenDims {
        d_model: d,
        heads,
        kv_heads: kvh,
        head_dim: hd,
        ff,
        vocab,
        layers,
        ctx,
        eps,
        ..
    } = *dims;
    // prefill reads token j for cache row j: a plain (unguarded) view
    // needs as many tokens as rows
    assert!(seq == 1 || seq == ctx, "prefill needs ctx == seq");
    let half = hd / 2;
    let per = heads / kvh;
    let mut b = Builder {
        g: TensorGraph::new(),
        inputs: Vec::new(),
    };
    let step = b.input("step".into(), &[dims.step_len(seq)], None);
    let x_rows = Rows {
        src: step,
        off: 0,
        stride: d as i64,
    };
    let cos_rows = Rows {
        src: step,
        off: (seq * d) as i64,
        stride: half as i64,
    };
    let sin_rows = Rows {
        src: step,
        off: (seq * d + seq * half) as i64,
        stride: half as i64,
    };
    let len_off = (seq * (d + hd)) as i64;
    let len_at = |rank: usize| arg(step, vec![Form::constant(rank, len_off)]);
    let q_rows = heads * hd;
    let k_off = q_rows as i64;
    let v_off = (q_rows + kvh * hd) as i64;
    let scale = 1.0 / (hd as f64).sqrt();
    let seq_f = seq as f64;
    let mut outputs = Vec::new();
    let mut keep = Vec::new();
    let mut x = x_rows;
    for l in 0..layers {
        let p = |s: &str| format!("model.layers.{l}.{s}");
        let qkv_w = b.q4(&p("qkv"), dims.qkv_rows(), d);
        let o_w = b.q4(&p("o"), d, q_rows);
        let gate_w = b.q4(&p("gate"), ff, d);
        let up_w = b.q4(&p("up"), ff, d);
        let down_w = b.q4(&p("down"), d, ff);
        let ln1 = b.input(p("ln1"), &[d], None);
        let ln2 = b.input(p("ln2"), &[d], None);
        let qn = b.input(p("q_norm"), &[hd], None);
        let kn = b.input(p("k_norm"), &[hd], None);
        let k_cache = b.input(p("k_cache"), &[kvh, ctx, 2, half], None);
        let v_cache = b.input(p("v_cache"), &[kvh, hd, ctx], None);

        let normed = b.rms_norm(x, seq, 0, d, ln1, eps);
        let normed = Rows {
            src: normed,
            off: 0,
            stride: d as i64,
        };
        let qkv = b.linear(normed, seq, 0, qkv_w);
        // per-head Σ x² of q and k heads: [seq, heads + kvh]
        let nh = heads + kvh;
        let hss =
            b.g.comp(
                &[seq, nh],
                &[hd],
                vec![arg(
                    qkv,
                    vec![unit(3, 0), form(3, &[(1, hd as i64), (2, 1)], 0)],
                )],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[0]],
            )
            .unwrap();
        // QK-norm then rotate-half rope over (t, head…, half h, i):
        // element 64h+i pairs with 64(1−h)+i, the sin term's sign is 2h−1
        #[allow(clippy::too_many_arguments)]
        let roped = |b: &mut Builder,
                     base: i64,
                     out: &[usize],
                     head: Form,
                     h: usize,
                     i: usize,
                     ss_head: Form,
                     w: TensorId,
                     mul: f64| {
            let rank = out.len();
            let mut self_f = head.clone();
            self_f.coefs[h] += half as i64;
            self_f.coefs[i] += 1;
            self_f.offset += base;
            let mut other = head.clone();
            other.coefs[h] -= half as i64;
            other.coefs[i] += 1;
            other.offset += base + half as i64;
            let w_self = form(rank, &[(h, half as i64), (i, 1)], 0);
            let w_other = form(rank, &[(h, -(half as i64)), (i, 1)], half as i64);
            let iota = b.g.iota(out, h);
            let cos = b.row(cos_rows, unit(rank, 0), unit(rank, i));
            let sin = b.row(sin_rows, unit(rank, 0), unit(rank, i));
            b.g.comp(
                out,
                &[],
                vec![
                    arg(qkv, vec![unit(rank, 0), self_f]),
                    arg(qkv, vec![unit(rank, 0), other]),
                    arg(w, vec![w_self]),
                    arg(w, vec![w_other]),
                    arg(hss, vec![unit(rank, 0), ss_head]),
                    cos,
                    sin,
                    arg(iota, ident(rank)),
                ],
                None,
                |v| {
                    let inv = ExprId::from_f64(1.0)
                        / (v[4] / ExprId::from_f64(hd as f64) + ExprId::from_f64(eps)).sqrt();
                    let sign = v[7] * ExprId::from_f64(2.0) - ExprId::from_f64(1.0);
                    let rot = v[0] * v[2] * v[5] + sign * v[1] * v[3] * v[6];
                    vec![rot * inv * ExprId::from_f64(mul)]
                },
            )
            .unwrap()
        };
        // q: [seq, kvh, per, 2, half], head kvh·per + r, scaled by 1/√hd
        let q_rot = roped(
            &mut b,
            0,
            &[seq, kvh, per, 2, half],
            form(5, &[(1, (per * hd) as i64), (2, hd as i64)], 0),
            3,
            4,
            form(5, &[(1, per as i64), (2, 1)], 0),
            qn,
            scale,
        );
        // k: [seq, kvh, 2, half]
        let k_rot = roped(
            &mut b,
            k_off,
            &[seq, kvh, 2, half],
            form(4, &[(1, hd as i64)], 0),
            2,
            3,
            form(4, &[(1, 1)], heads as i64),
            kn,
            1.0,
        );
        // cache rows len−seq ≤ j < len, in place; token t = j − (len − seq),
        // which is affine when seq == 1 (t = 0) or len == seq (t = j)
        let in_step = |len: ExprId, j: ExprId, new: ExprId, old: ExprId| {
            let le = len - ExprId::from_f64(1.0) - j + ExprId::from_f64(0.5);
            let ge = j - len + ExprId::from_f64(seq_f) + ExprId::from_f64(0.5);
            let inner = select(ge, new, old);
            select(le, inner, old)
        };
        let tok = |rank: usize, j: usize| {
            if seq == 1 {
                Form::zero(rank)
            } else {
                unit(rank, j)
            }
        };
        let kc_shape = [kvh, ctx, 2, half];
        let j_iota = b.g.iota(&kc_shape, 1);
        let k_new =
            b.g.comp(
                &kc_shape,
                &[],
                vec![
                    arg(k_rot, vec![tok(4, 1), unit(4, 0), unit(4, 2), unit(4, 3)]),
                    arg(k_cache, ident(4)),
                    arg(j_iota, ident(4)),
                    len_at(4),
                ],
                None,
                |v| vec![in_step(v[3], v[2], v[0], v[1])],
            )
            .unwrap();
        let vc_shape = [kvh, hd, ctx];
        let jv_iota = b.g.iota(&vc_shape, 2);
        let v_new =
            b.g.comp(
                &vc_shape,
                &[],
                vec![
                    arg(
                        qkv,
                        vec![tok(3, 2), form(3, &[(0, hd as i64), (1, 1)], v_off)],
                    ),
                    arg(v_cache, ident(3)),
                    arg(jv_iota, ident(3)),
                    len_at(3),
                ],
                None,
                |v| vec![in_step(v[3], v[2], v[0], v[1])],
            )
            .unwrap();
        outputs.push(k_new);
        outputs.push(v_new);

        // causal: key j is visible to token t at position len − seq + t
        let visible = |len: ExprId, t: ExprId, j: ExprId| {
            len - ExprId::from_f64(seq_f) + t - j + ExprId::from_f64(0.5)
        };
        // scores[t, kvh, r, j] over (h, i): space (t, kvh, r, j, h, i)
        let sp = [seq, kvh, per, ctx, 2, half];
        let (ts, js) = (b.g.iota(&sp, 0), b.g.iota(&sp, 3));
        let scores =
            b.g.comp(
                &[seq, kvh, per, ctx],
                &[2, half],
                vec![
                    arg(
                        q_rot,
                        vec![unit(6, 0), unit(6, 1), unit(6, 2), unit(6, 4), unit(6, 5)],
                    ),
                    arg(k_new, vec![unit(6, 1), unit(6, 3), unit(6, 4), unit(6, 5)]),
                    arg(ts, ident(6)),
                    arg(js, ident(6)),
                    len_at(6),
                ],
                Some(Monoid::sum()),
                |v| {
                    let c = visible(v[4], v[2], v[3]);
                    let z = ExprId::from_f64(0.0);
                    vec![select(c, v[0] * v[1], z)]
                },
            )
            .unwrap();
        // softmax statistics over j: space (t, kvh, r, j)
        let sp = [seq, kvh, per, ctx];
        let (tr, jr) = (b.g.iota(&sp, 0), b.g.iota(&sp, 3));
        let s_args = |m: Option<TensorId>| {
            let mut a = vec![
                arg(scores, ident(4)),
                arg(tr, ident(4)),
                arg(jr, ident(4)),
                len_at(4),
            ];
            if let Some(m) = m {
                a.push(arg(m, vec![unit(4, 0), unit(4, 1), unit(4, 2)]));
            }
            a
        };
        let m =
            b.g.comp(
                &[seq, kvh, per],
                &[ctx],
                s_args(None),
                Some(Monoid::max()),
                |v| {
                    let c = visible(v[3], v[1], v[2]);
                    let lo = ExprId::from_f64(-1e30);
                    vec![select(c, v[0], lo)]
                },
            )
            .unwrap();
        let lsum =
            b.g.comp(
                &[seq, kvh, per],
                &[ctx],
                s_args(Some(m)),
                Some(Monoid::sum()),
                |v| {
                    let c = visible(v[3], v[1], v[2]);
                    let z = ExprId::from_f64(0.0);
                    vec![select(c, (v[0] - v[4]).exp(), z)]
                },
            )
            .unwrap();
        // attn[t, kvh, r, c] = Σ_j p·v / l: space (t, kvh, r, c, j)
        let sp = [seq, kvh, per, hd, ctx];
        let (to, jo) = (b.g.iota(&sp, 0), b.g.iota(&sp, 4));
        let mlv = vec![unit(5, 0), unit(5, 1), unit(5, 2)];
        let attn =
            b.g.comp(
                &[seq, kvh, per, hd],
                &[ctx],
                vec![
                    arg(scores, vec![unit(5, 0), unit(5, 1), unit(5, 2), unit(5, 4)]),
                    arg(m, mlv.clone()),
                    arg(lsum, mlv),
                    arg(v_new, vec![unit(5, 1), unit(5, 3), unit(5, 4)]),
                    arg(to, ident(5)),
                    arg(jo, ident(5)),
                    len_at(5),
                ],
                Some(Monoid::sum()),
                |v| {
                    let c = visible(v[6], v[4], v[5]);
                    let pv = (v[0] - v[1]).exp() * v[3] / v[2];
                    let z = ExprId::from_f64(0.0);
                    vec![select(c, pv, z)]
                },
            )
            .unwrap();
        let attn = b.g.reshape(attn, &[seq, q_rows]).unwrap();
        let attn = Rows {
            src: attn,
            off: 0,
            stride: q_rows as i64,
        };
        let proj = b.linear(attn, seq, 0, o_w);
        let res1 = b.add(x, proj, seq, d);
        let res1_rows = Rows {
            src: res1,
            off: 0,
            stride: d as i64,
        };
        let normed2 = b.rms_norm(res1_rows, seq, 0, d, ln2, eps);
        let normed2 = Rows {
            src: normed2,
            off: 0,
            stride: d as i64,
        };
        let gate = b.linear(normed2, seq, 0, gate_w);
        let up = b.linear(normed2, seq, 0, up_w);
        let act =
            b.g.map(&[gate, up], |a| {
                a[0] / (ExprId::from_f64(1.0) + (-a[0]).exp()) * a[1]
            })
            .unwrap();
        let act = Rows {
            src: act,
            off: 0,
            stride: ff as i64,
        };
        let down = b.linear(act, seq, 0, down_w);
        let res2 = b.add(res1_rows, down, seq, d);
        keep.extend([res1, res2]);
        x = Rows {
            src: res2,
            off: 0,
            stride: d as i64,
        };
    }
    // logits of the last token only
    let norm = b.input("model.norm".into(), &[d], None);
    let lm = b.q4("lm_head", vocab, d);
    let normed = b.rms_norm(x, 1, seq as i64 - 1, d, norm, eps);
    let normed = Rows {
        src: normed,
        off: 0,
        stride: d as i64,
    };
    let logits = b.linear(normed, 1, 0, lm);
    outputs.insert(0, logits);
    DecodeGraph {
        g: b.g,
        seq,
        outputs,
        keep,
        inputs: b.inputs,
    }
}

/// The passes: maps fused, siblings (gate/up) merged, epilogues
/// inlined; for prefill (`seq > 1`) the reduce dims of the matmuls
/// merged, so they are tensor-core tiles over the dequantized weights.
/// Returns the graph and its outputs: `dg.outputs` (same order), then
/// `dg.keep`.
pub fn compiled_decode(dg: &DecodeGraph) -> (TensorGraph, Vec<TensorId>) {
    let all: Vec<TensorId> = dg.outputs.iter().chain(&dg.keep).copied().collect();
    let (g, outs) = dg.g.substitute_maps(&all);
    let (g, outs) = g.eliminate_common(&outs);
    let (g, outs) = g.merge_siblings(&outs);
    let (mut g, outs) = if dg.seq > 1 {
        g.merge_reduce_dims(&outs)
    } else {
        (g, outs)
    };
    g.schedule_inline(&outs);
    g.schedule_epilogues(&outs);
    (g, outs)
}

/// Packed inputs and the in-place cache updates for `to_cuda_decode`.
pub fn decode_opts(dg: &DecodeGraph, outputs: &[TensorId], layers: usize) -> DecodeOpts {
    let packed = dg
        .inputs
        .iter()
        .enumerate()
        .filter_map(|(k, s)| s.packing.map(|p| (k, p)))
        .collect();
    let mut inplace = Vec::new();
    for l in 0..layers {
        let base = 1 + l * LAYER_INPUTS.len();
        inplace.push((outputs[1 + 2 * l], base + 19));
        inplace.push((outputs[2 + 2 * l], base + 20));
    }
    DecodeOpts {
        packed,
        inplace,
        mma: dg.seq > 1,
        ..Default::default()
    }
}

/// `cos`/`sin` of rope at `pos` (`head_dim/2` each).
pub fn rope_angles(dims: &QwenDims, pos: usize) -> (Vec<f32>, Vec<f32>) {
    let half = dims.head_dim / 2;
    (0..half)
        .map(|i| {
            let inv = dims.theta.powf(-(2.0 * i as f64) / dims.head_dim as f64);
            let a = pos as f64 * inv;
            (a.cos() as f32, a.sin() as f32)
        })
        .unzip()
}

/// The step input for tokens `xs` (embeddings) at positions
/// `pos0..pos0 + xs.len()`.
pub fn step_input(dims: &QwenDims, xs: &[Vec<f32>], pos0: usize) -> Vec<f32> {
    let seq = xs.len();
    let mut s = Vec::with_capacity(dims.step_len(seq));
    for x in xs {
        s.extend_from_slice(x);
    }
    let angles: Vec<_> = (0..seq).map(|t| rope_angles(dims, pos0 + t)).collect();
    for (c, _) in &angles {
        s.extend_from_slice(c);
    }
    for (_, sn) in &angles {
        s.extend_from_slice(sn);
    }
    s.push((pos0 + seq) as f32);
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> QwenDims {
        QwenDims {
            d_model: 128,
            heads: 4,
            kv_heads: 2,
            head_dim: 32,
            ff: 192,
            vocab: 256,
            layers: 2,
            ctx: 16,
            eps: 1e-6,
            theta: 1e4,
        }
    }

    #[test]
    fn decode_graph_lowers() {
        let dims = tiny();
        let dg = decode_graph(&dims);
        let (g, outs) = compiled_decode(&dg);
        let opts = decode_opts(&dg, &outs, dims.layers);
        let p = g.to_cuda_decode(&outs, &opts);
        let transpiled = p.native.iter().filter(|n| !**n).count();
        assert!(p.native.iter().filter(|n| **n).count() >= 10 * dims.layers);
        assert!(transpiled <= 4 * dims.layers + 2, "{transpiled} transpiled");
        // packed weights are sized in words
        let q = p
            .program
            .buffers
            .iter()
            .find(|b| b.input == Some(1))
            .unwrap();
        assert_eq!(q.len, dims.qkv_rows() * dims.d_model / 8);
    }

    #[test]
    fn prefill_graph_lowers() {
        let dims = tiny();
        let dg = step_graph(&dims, dims.ctx);
        let (g, outs) = compiled_decode(&dg);
        let opts = decode_opts(&dg, &outs, dims.layers);
        let p = g.to_cuda_decode(&outs, &opts);
        let mma = p
            .program
            .kernels
            .iter()
            .filter(|k| k.source.contains("mma"))
            .count();
        // the seven projections per layer are tensor-core tiles (gate/up
        // merged: six)
        assert!(mma >= 5 * dims.layers, "{mma} mma tiles");
    }
}
