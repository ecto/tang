//! Qwen3 decode as one tensor-IR graph per step.
//!
//! The step reads one token's embedding and its rope angles from a
//! `step` input, runs every layer against KV caches of fixed capacity
//! `ctx`, and writes the logits and the caches' new rows. Weights are
//! MLX 4-bit: per projection a `[rows, cols/64, 64]` tensor of nibbles
//! (logical values 0..=15) and bf16 `[rows, cols/64]` scales and biases;
//! the dequantization `q·s + b` is a map the matmul reads, so it is
//! inlined and evaluated where the packed words are loaded
//! (`to_cuda_decode`, `Packing::U4`).
//!
//! Attention masks positions past the current length `len` with
//! `select(len − j − ½, …, identity)` in every reduction over the cache
//! axis; `to_cuda_decode` turns those into loop bounds, so a step reads
//! `len` rows of cache, not `ctx`. The cache updates are maps
//! `select(j == len − 1, new, old)` that overwrite the cache in place.
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

    /// `x`, `cos`, `sin` (head_dim/2 each), `len`.
    pub fn step_len(&self) -> usize {
        self.d_model + self.head_dim + 1
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
    /// `[logits, k_cache'₀, v_cache'₀, k_cache'₁, …]`.
    pub outputs: Vec<TensorId>,
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

fn arg(src: TensorId, forms: Vec<Form>) -> Arg {
    Arg {
        src,
        view: View {
            guard: false,
            forms,
        },
    }
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

    /// `y[n] = Σ_k x[k]·(q[n,k]·s[n,k/64] + b[n,k/64])`, `x` a vector
    /// read at `x_off + k`.
    fn linear(&mut self, x: TensorId, x_off: i64, w: [TensorId; 3]) -> TensorId {
        let shape = self.g.shape(w[0]).to_vec();
        let (rows, groups) = (shape[0].n(), shape[1].n());
        // dequantization: a map over the nibbles, inlined into the matmul
        let r3 = 3;
        let dq = self
            .g
            .comp(
                &[rows, groups, GROUP],
                &[],
                vec![
                    arg(w[0], vec![unit(r3, 0), unit(r3, 1), unit(r3, 2)]),
                    arg(w[1], vec![unit(r3, 0), unit(r3, 1)]),
                    arg(w[2], vec![unit(r3, 0), unit(r3, 1)]),
                ],
                None,
                |v| vec![v[0] * v[1] + v[2]],
            )
            .unwrap();
        // space (n, g, j)
        self.g
            .comp(
                &[rows],
                &[groups, GROUP],
                vec![
                    arg(x, vec![form(r3, &[(1, GROUP as i64), (2, 1)], x_off)]),
                    arg(dq, vec![unit(r3, 0), unit(r3, 1), unit(r3, 2)]),
                ],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[1]],
            )
            .unwrap()
    }

    /// `Σ_k x[off + k]²` over `n` elements, as a `[1]` tensor.
    fn sum_sq(&mut self, x: TensorId, off: i64, n: usize) -> TensorId {
        self.g
            .comp(
                &[1usize],
                &[n],
                vec![arg(x, vec![form(2, &[(1, 1)], off)])],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[0]],
            )
            .unwrap()
    }

    /// RMSNorm of `x[off..off+n]` with weight `w`: a `[n]` map.
    fn rms_norm(&mut self, x: TensorId, off: i64, n: usize, w: TensorId, eps: f64) -> TensorId {
        let ss = self.sum_sq(x, off, n);
        self.g
            .comp(
                &[n],
                &[],
                vec![
                    arg(x, vec![form(1, &[(0, 1)], off)]),
                    arg(ss, vec![Form::zero(1)]),
                    arg(w, vec![unit(1, 0)]),
                ],
                None,
                |v| {
                    let inv = ExprId::from_f64(1.0)
                        / (v[1] / ExprId::from_f64(n as f64) + ExprId::from_f64(eps)).sqrt();
                    vec![v[0] * inv * v[2]]
                },
            )
            .unwrap()
    }

    /// `a[i] + b[i]` over `[n]`.
    fn add(&mut self, a: TensorId, a_off: i64, b: TensorId, n: usize) -> TensorId {
        self.g
            .comp(
                &[n],
                &[],
                vec![
                    arg(a, vec![form(1, &[(0, 1)], a_off)]),
                    arg(b, vec![unit(1, 0)]),
                ],
                None,
                |v| vec![v[0] + v[1]],
            )
            .unwrap()
    }
}

/// The decode step for `dims`.
pub fn decode_graph(dims: &QwenDims) -> DecodeGraph {
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
    let half = hd / 2;
    let per = heads / kvh;
    let mut b = Builder {
        g: TensorGraph::new(),
        inputs: Vec::new(),
    };
    let step = b.input("step".into(), &[dims.step_len()], None);
    let (cos_off, sin_off, len_off) = (d as i64, (d + half) as i64, (d + hd) as i64);
    let q_rows = heads * hd;
    let k_off = q_rows as i64;
    let v_off = (q_rows + kvh * hd) as i64;
    let scale = 1.0 / (hd as f64).sqrt();
    let mut outputs = Vec::new();
    // the residual stream: (tensor, offset)
    let mut x = (step, 0i64);
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

        let normed = b.rms_norm(x.0, x.1, d, ln1, eps);
        let qkv = b.linear(normed, 0, qkv_w);
        // per-head Σ x² of q and k heads: [heads + kvh]
        let nh = heads + kvh;
        let hss =
            b.g.comp(
                &[nh],
                &[hd],
                vec![arg(qkv, vec![form(2, &[(0, hd as i64), (1, 1)], 0)])],
                Some(Monoid::sum()),
                |v| vec![v[0] * v[0]],
            )
            .unwrap();
        // QK-norm then rotate-half rope over (head, half h, i): element
        // 64h+i pairs with 64(1−h)+i, the sign of the sin term is 2h−1
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
            b.g.comp(
                out,
                &[],
                vec![
                    arg(qkv, vec![self_f]),
                    arg(qkv, vec![other]),
                    arg(w, vec![w_self]),
                    arg(w, vec![w_other]),
                    arg(hss, vec![ss_head]),
                    arg(step, vec![form(rank, &[(i, 1)], cos_off)]),
                    arg(step, vec![form(rank, &[(i, 1)], sin_off)]),
                    arg(iota, (0..rank).map(|d| unit(rank, d)).collect()),
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
        // q: [kvh, per, 2, half], head kvh·per + r, scaled by 1/√hd
        let q_rot = roped(
            &mut b,
            0,
            &[kvh, per, 2, half],
            form(4, &[(0, (per * hd) as i64), (1, hd as i64)], 0),
            2,
            3,
            form(4, &[(0, per as i64), (1, 1)], 0),
            qn,
            scale,
        );
        // k: [kvh, 2, half]
        let k_rot = roped(
            &mut b,
            k_off,
            &[kvh, 2, half],
            form(3, &[(0, hd as i64)], 0),
            1,
            2,
            form(3, &[(0, 1)], heads as i64),
            kn,
            1.0,
        );
        // cache rows at pos = len − 1, in place
        let len_at = |rank: usize| arg(step, vec![Form::constant(rank, len_off)]);
        let at_pos = |len: ExprId, j: ExprId, new: ExprId, old: ExprId| {
            let pos = len - ExprId::from_f64(1.0);
            let half_ = ExprId::from_f64(0.5);
            let le = pos - j + half_;
            let ge = j - pos + ExprId::from_f64(0.5);
            tang_expr::with_graph(|g| {
                let inner = g.select(ge, new, old);
                g.select(le, inner, old)
            })
        };
        let kc_shape = [kvh, ctx, 2, half];
        let j_iota = b.g.iota(&kc_shape, 1);
        let k_new =
            b.g.comp(
                &kc_shape,
                &[],
                vec![
                    arg(k_rot, vec![unit(4, 0), unit(4, 2), unit(4, 3)]),
                    arg(k_cache, (0..4).map(|d| unit(4, d)).collect()),
                    arg(j_iota, (0..4).map(|d| unit(4, d)).collect()),
                    len_at(4),
                ],
                None,
                |v| vec![at_pos(v[3], v[2], v[0], v[1])],
            )
            .unwrap();
        let vc_shape = [kvh, hd, ctx];
        let jv_iota = b.g.iota(&vc_shape, 2);
        let v_new =
            b.g.comp(
                &vc_shape,
                &[],
                vec![
                    arg(qkv, vec![form(3, &[(0, hd as i64), (1, 1)], v_off)]),
                    arg(v_cache, (0..3).map(|d| unit(3, d)).collect()),
                    arg(jv_iota, (0..3).map(|d| unit(3, d)).collect()),
                    len_at(3),
                ],
                None,
                |v| vec![at_pos(v[3], v[2], v[0], v[1])],
            )
            .unwrap();
        outputs.push(k_new);
        outputs.push(v_new);

        // valid position: j < len
        let valid = |len: ExprId, j: ExprId| len - j - ExprId::from_f64(0.5);
        // scores[kvh, r, j] over (h, i): space (kvh, r, j, h, i)
        let s_shape = [kvh, per, ctx];
        let js = b.g.iota(&[kvh, per, ctx, 2, half], 2);
        let scores =
            b.g.comp(
                &s_shape,
                &[2, half],
                vec![
                    arg(q_rot, vec![unit(5, 0), unit(5, 1), unit(5, 3), unit(5, 4)]),
                    arg(k_new, vec![unit(5, 0), unit(5, 2), unit(5, 3), unit(5, 4)]),
                    arg(js, (0..5).map(|d| unit(5, d)).collect()),
                    len_at(5),
                ],
                Some(Monoid::sum()),
                |v| {
                    let c = valid(v[3], v[2]);
                    let qk = v[0] * v[1];
                    let z = ExprId::from_f64(0.0);
                    vec![tang_expr::with_graph(|g| g.select(c, qk, z))]
                },
            )
            .unwrap();
        // softmax statistics over j: space (kvh, r, j)
        let jr = b.g.iota(&[kvh, per, ctx], 2);
        let s_args = |m: Option<TensorId>| {
            let mut a = vec![
                arg(scores, vec![unit(3, 0), unit(3, 1), unit(3, 2)]),
                arg(jr, vec![unit(3, 0), unit(3, 1), unit(3, 2)]),
                len_at(3),
            ];
            if let Some(m) = m {
                a.push(arg(m, vec![unit(3, 0), unit(3, 1)]));
            }
            a
        };
        let m =
            b.g.comp(
                &[kvh, per],
                &[ctx],
                s_args(None),
                Some(Monoid::max()),
                |v| {
                    let c = valid(v[2], v[1]);
                    let lo = ExprId::from_f64(-1e30);
                    vec![tang_expr::with_graph(|g| g.select(c, v[0], lo))]
                },
            )
            .unwrap();
        let lsum =
            b.g.comp(
                &[kvh, per],
                &[ctx],
                s_args(Some(m)),
                Some(Monoid::sum()),
                |v| {
                    let c = valid(v[2], v[1]);
                    let p = (v[0] - v[3]).exp();
                    let z = ExprId::from_f64(0.0);
                    vec![tang_expr::with_graph(|g| g.select(c, p, z))]
                },
            )
            .unwrap();
        // attn[kvh, r, c] = Σ_j p·v / l: space (kvh, r, c, j)
        let jo = b.g.iota(&[kvh, per, hd, ctx], 3);
        let attn =
            b.g.comp(
                &[kvh, per, hd],
                &[ctx],
                vec![
                    arg(scores, vec![unit(4, 0), unit(4, 1), unit(4, 3)]),
                    arg(m, vec![unit(4, 0), unit(4, 1)]),
                    arg(lsum, vec![unit(4, 0), unit(4, 1)]),
                    arg(v_new, vec![unit(4, 0), unit(4, 2), unit(4, 3)]),
                    arg(jo, (0..4).map(|d| unit(4, d)).collect()),
                    len_at(4),
                ],
                Some(Monoid::sum()),
                |v| {
                    let c = valid(v[5], v[4]);
                    let pv = (v[0] - v[1]).exp() * v[3] / v[2];
                    let z = ExprId::from_f64(0.0);
                    vec![tang_expr::with_graph(|g| g.select(c, pv, z))]
                },
            )
            .unwrap();
        let attn = b.g.reshape(attn, &[q_rows]).unwrap();
        let proj = b.linear(attn, 0, o_w);
        let res1 = b.add(x.0, x.1, proj, d);
        let normed2 = b.rms_norm(res1, 0, d, ln2, eps);
        let gate = b.linear(normed2, 0, gate_w);
        let up = b.linear(normed2, 0, up_w);
        let act =
            b.g.map(&[gate, up], |a| {
                a[0] / (ExprId::from_f64(1.0) + (-a[0]).exp()) * a[1]
            })
            .unwrap();
        let down = b.linear(act, 0, down_w);
        let res2 = b.add(res1, 0, down, d);
        x = (res2, 0);
    }
    let norm = b.input("model.norm".into(), &[d], None);
    let lm = b.q4("lm_head", vocab, d);
    let normed = b.rms_norm(x.0, x.1, d, norm, eps);
    let logits = b.linear(normed, 0, lm);
    outputs.insert(0, logits);
    DecodeGraph {
        g: b.g,
        outputs,
        inputs: b.inputs,
    }
}

/// The passes for decode: maps fused, siblings (gate/up) merged,
/// epilogues inlined. Returns the graph and its outputs (same order).
pub fn compiled_decode(dg: &DecodeGraph) -> (TensorGraph, Vec<TensorId>) {
    let (g, outs) = dg.g.substitute_maps(&dg.outputs);
    let (g, outs) = g.eliminate_common(&outs);
    let (mut g, outs) = g.merge_siblings(&outs);
    g.schedule_inline(&outs);
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
        for (k, n) in p.program.kernels.iter().zip(&p.native) {
            if !n {
                eprintln!("transpiled: {} points", k.points);
            }
        }
        assert!(p.native.iter().filter(|n| **n).count() >= 10 * dims.layers);
        assert!(transpiled <= 4 * dims.layers + 2, "{transpiled} transpiled");
    }
}
