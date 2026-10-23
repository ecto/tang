//! KDA: Kimi delta attention (W4, branch frog/glm5-kda).
//!
//! The KDA layers (`is_kda[l] == true`): per-head delta-rule state with a *per-channel*
//! (vector) decay gate — unlike tang's Qwen3-Next GDN whose gate is a scalar per head.
//! Head dim `kda.head_dim` (128; 8 in the tiny fixture) × `attention.head_count` = `d_inner`;
//! short conv `ssm.conv_kernel` (4) with **separate** q/k/v conv weights, silu-activated;
//! `ssm_a` = `-exp(A_log)` per head; `ssm_dt.bias` per channel; `ssm_f_a`/`ssm_f_b` produce
//! the decay gate `g = lower_bound * sigmoid(exp(A_log) · g1)` with `lower_bound` = −5
//! (`kda.gate_lower_bound`), so the per-channel decay `exp(g)` ∈ (exp(-5), 1); without a
//! lower bound, `g = -exp(A_log) · softplus(g1)`. beta = `sigmoid(ssm_beta · x)` per head;
//! q/k get an FLA-style L2 norm (eps added *inside* the sqrt, 1e-6) before the recurrence;
//! output `RMSNorm(o) * sigmoid(ssm_g_b(ssm_g_a(x)))` (per channel), then `attn_output`.
//!
//! HF reference: `Glm5NextTextLinearAttention`, `Glm5NextTextForgetGate`,
//! `recurrent_kimi_delta_attention`. GGUF names in [`super::gguf`].
//!
//! This is the reference (full-prefill, per-token recurrent) form: exact, O(t·d²) — fine for
//! parity runs; the chunked + incremental-state form is W4's to land (this module yields when
//! `frog/glm5-kda` merges).

use super::Hparams;
use crate::flash::reference::{rms_norm_mul, silu, Mat};
use crate::gguf::Gguf;
use anyhow::{ensure, Result};
use rayon::prelude::*;

/// One KDA layer's weights, dequantised to f32.
pub struct KdaWeights {
    pub n_head: usize,
    pub head_dim: usize,
    pub d_inner: usize,
    pub d_conv: usize,
    /// `kda.gate_lower_bound` (−5), or `None` for the softplus gate.
    pub lower_bound: Option<f32>,
    pub q: Mat,
    pub k: Mat,
    pub v: Mat,
    /// `[d_conv][d_inner]` (channel fastest; tap 0 = oldest).
    pub conv_q: Vec<f32>,
    pub conv_k: Vec<f32>,
    pub conv_v: Vec<f32>,
    /// `ssm_f_a.weight` [n_embd, head_dim].
    pub f_a: Mat,
    /// `ssm_f_b.weight` [head_dim, d_inner].
    pub f_b: Mat,
    /// `ssm_dt.bias` [d_inner].
    pub dt_bias: Vec<f32>,
    /// `ssm_a`, one per head: `-exp(A_log)` (the converter's convention).
    pub a: Vec<f32>,
    /// `ssm_beta.weight` [n_embd, n_head].
    pub beta: Mat,
    /// `ssm_g_a.weight` [n_embd, head_dim].
    pub g_a: Mat,
    /// `ssm_g_b.weight` [head_dim, d_inner].
    pub g_b: Mat,
    /// `ssm_norm.weight` [head_dim].
    pub o_norm: Vec<f32>,
    /// `attn_output.weight` [d_inner, n_embd].
    pub out: Mat,
}

impl KdaWeights {
    pub fn from_gguf(g: &Gguf, hp: &Hparams, l: usize) -> Result<Self> {
        let t = |s: &str| super::t(l, s);
        let mat = |name: &str| -> Result<Mat> {
            let info = g.info(&t(name))?;
            ensure!(
                info.dims.len() <= 2,
                "blk.{l}.{name}: not a matrix ({:?})",
                info.dims
            );
            let cols = info.dims[0] as usize;
            let rows = if info.dims.len() == 2 {
                info.dims[1] as usize
            } else {
                1
            };
            Ok(Mat {
                rows,
                cols,
                data: g.dequantize(info)?,
                act: crate::flash::reference::Act::F32,
            })
        };
        let vec = |name: &str| -> Result<Vec<f32>> { g.tensor_f32(&t(name)) };
        // conv weights are [d_conv, 1, d_inner, 1] → flat [d_inner][d_conv], tap fastest
        let conv = |name: &str| -> Result<Vec<f32>> {
            let info = g.info(&t(name))?;
            ensure!(
                info.dims.len() == 4
                    && info.dims[0] == hp.ssm_d_conv as u64
                    && info.dims[1] == 1
                    && info.dims[2] == (hp.kda_head_dim * hp.n_head) as u64
                    && info.dims[3] == 1,
                "blk.{l}.{name}: dims {:?}",
                info.dims
            );
            Ok(g.tensor_f32(&t(name))?)
        };
        Ok(Self {
            n_head: hp.n_head,
            head_dim: hp.kda_head_dim,
            d_inner: hp.kda_head_dim * hp.n_head,
            d_conv: hp.ssm_d_conv,
            lower_bound: hp.kda_gate_lower_bound,
            q: mat("attn_q.weight")?,
            k: mat("attn_k.weight")?,
            v: mat("attn_v.weight")?,
            conv_q: conv("ssm_conv1d_q.weight")?,
            conv_k: conv("ssm_conv1d_k.weight")?,
            conv_v: conv("ssm_conv1d_v.weight")?,
            f_a: mat("ssm_f_a.weight")?,
            f_b: mat("ssm_f_b.weight")?,
            dt_bias: vec("ssm_dt.bias")?,
            a: vec("ssm_a")?,
            beta: mat("ssm_beta.weight")?,
            g_a: mat("ssm_g_a.weight")?,
            g_b: mat("ssm_g_b.weight")?,
            o_norm: vec("ssm_norm.weight")?,
            out: mat("attn_output.weight")?,
        })
    }
}

/// FLA's l2norm: `x / sqrt(sum(x²) + eps)` (eps inside the sqrt, plain `/`).
fn l2norm(x: &mut [f32]) {
    let s: f32 = x.iter().map(|v| v * v).sum();
    let inv = 1.0 / (s + 1e-6).sqrt();
    for v in x.iter_mut() {
        *v *= inv;
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Run the layer over a full prefill `x` `[t_len][n_embd]` starting at position 0 (the conv
/// pads with zeros). Returns `[t_len][n_embd]`.
pub fn kda_forward(w: &KdaWeights, x: &[f32], t_len: usize) -> Result<Vec<f32>> {
    let (n_head, hd, d_inner, d_conv) = (w.n_head, w.head_dim, w.d_inner, w.d_conv);
    let n_embd = w.q.cols;
    let win = d_conv - 1;
    let scale = (hd as f32).sqrt().recip();

    // ---- batch the matmuls and the conv (per q/k/v), then the recurrence runs per head
    let mut convd: [Vec<f32>; 3] = [
        vec![0f32; t_len * d_inner],
        vec![0f32; t_len * d_inner],
        vec![0f32; t_len * d_inner],
    ];
    let proj: [Vec<f32>; 3] = [
        w.q.apply(x, t_len, false),
        w.k.apply(x, t_len, false),
        w.v.apply(x, t_len, false),
    ];
    let conv_w = [&w.conv_q, &w.conv_k, &w.conv_v];
    for ((p, c), cw) in proj.iter().zip(convd.iter_mut()).zip(conv_w) {
        c.par_chunks_mut(d_inner)
            .enumerate()
            .for_each(|(t, row)| {
                for ch in 0..d_inner {
                    let mut acc = 0f32;
                    for k in 0..d_conv {
                        let idx = t as i64 - win as i64 + k as i64;
                        let v = if idx >= 0 {
                            p[idx as usize * d_inner + ch]
                        } else {
                            0.0 // prefill from 0: nothing before the first token
                        };
                        acc += cw[ch * d_conv + k] * v;
                    }
                    row[ch] = silu(acc);
                }
            });
    }
    // FLA l2 norm on q and k, per head
    for c in convd[..2].iter_mut() {
        c.par_chunks_mut(hd).for_each(l2norm);
    }

    // ---- the decay gate g1 = f_b(f_a(x)) + dt_bias, per token
    let fa = w.f_a.apply(x, t_len, false);
    let mut g1 = w.f_b.apply(&fa, t_len, false);
    for (row, dtr) in g1.chunks_mut(d_inner).zip(std::iter::repeat(&w.dt_bias)) {
        for (g, dt) in row.iter_mut().zip(dtr) {
            *g += dt;
        }
    }
    // ---- beta per token per head, sigmoid
    let mut beta = w.beta.apply(x, t_len, false);
    for b in beta.iter_mut() {
        *b = sigmoid(*b);
    }
    // ---- the output gate g2 = g_b(g_a(x))
    let ga = w.g_a.apply(x, t_len, false);
    let g2 = w.g_b.apply(&ga, t_len, false);

    // ---- the delta-rule recurrence: sequential in t, parallel over heads
    let mut o = vec![0f32; t_len * d_inner];
    let mut states = vec![vec![0f32; hd * hd]; n_head];
    let decay = |h: usize, c: usize, g1: &[f32]| -> f32 {
        // ssm_a holds -exp(A_log); decay = exp(g)
        match w.lower_bound {
            Some(lb) => (lb * sigmoid(-w.a[h] * g1[c])).exp(),
            None => (w.a[h] * softplus(g1[c])).exp(),
        }
    };
    for t in 0..t_len {
        let b_t = &beta[t * n_head..(t + 1) * n_head];
        let g1_t = &g1[t * d_inner..(t + 1) * d_inner];
        let o_t: &mut [f32] = &mut o[t * d_inner..(t + 1) * d_inner];
        // each head writes its own chunk of o_t, so the split borrows don't alias
        let o_splits: Vec<&mut [f32]> = o_t.chunks_mut(hd).collect();
        states
            .par_iter_mut()
            .zip(o_splits)
            .enumerate()
                .for_each(|(h, (s, o_h))| {
                    let q = &convd[0][t * d_inner + h * hd..t * d_inner + (h + 1) * hd];
                    let k = &convd[1][t * d_inner + h * hd..t * d_inner + (h + 1) * hd];
                    let v = &convd[2][t * d_inner + h * hd..t * d_inner + (h + 1) * hd];
                    // per-channel decay (rows = key channels)
                    for kk in 0..hd {
                        let g = decay(h, h * hd + kk, g1_t);
                        for vv in 0..hd {
                            s[kk * hd + vv] *= g;
                        }
                    }
                    // delta = (v − Sᵀk)·beta, S[k][v]
                    let mut delta = vec![0f32; hd];
                    for vv in 0..hd {
                        let mut mem = 0f32;
                        for kk in 0..hd {
                            mem += s[kk * hd + vv] * k[kk];
                        }
                        delta[vv] = (v[vv] - mem) * b_t[h];
                    }
                    for kk in 0..hd {
                        for vv in 0..hd {
                            s[kk * hd + vv] += k[kk] * delta[vv];
                        }
                    }
                    // o = Sᵀ q
                    for vv in 0..hd {
                        let mut acc = 0f32;
                        for kk in 0..hd {
                            acc += s[kk * hd + vv] * q[kk];
                        }
                        o_h[vv] = acc * scale;
                    }
                });
    }

    // ---- gated norm: RMSNorm(o per head) * sigmoid(g2), then attn_output
    let mut gated = o;
    gated
        .par_chunks_mut(hd)
        .zip(g2.par_chunks(hd))
        .for_each(|(row, g)| {
            rms_norm_mul(row, &w.o_norm, 1e-5);
            for (v, g) in row.iter_mut().zip(g) {
                *v *= sigmoid(*g);
            }
        });
    Ok(w.out.apply(&gated, t_len, false))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Debug parity vs the HF torch fixture in /tmp (written by /tmp/kda_hf_ref.py):
    /// writes my q/k/v conv outputs and the gate for comparison.
    #[test]
    fn debug_intermediates() {
        let (x, core) = match (
            std::fs::read("/tmp/kda_x.bin"),
            std::fs::read("/tmp/kda_core.bin"),
        ) {
            (Ok(x), Ok(_)) => (x, ()),
            _ => return,
        };
        let _ = core;
        let x: Vec<f32> = x
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let g = crate::gguf::Gguf::open(std::path::Path::new(
            "/private/tmp/glm5-tiny/glm5-tiny-f32.gguf",
        ))
        .unwrap();
        let hp = super::super::Hparams::from_gguf(&g).unwrap();
        let w = KdaWeights::from_gguf(&g, &hp, 0).unwrap();
        let t_len = 52;
        let (hd, d_inner, d_conv, n_head) = (w.head_dim, w.d_inner, w.d_conv, w.n_head);
        let n_embd = w.q.cols;
        let win = d_conv - 1;

        // conv outputs (same code as kda_forward)
        let proj: [Vec<f32>; 3] = [
            w.q.apply(&x, t_len, false),
            w.k.apply(&x, t_len, false),
            w.v.apply(&x, t_len, false),
        ];
        let mut convd: [Vec<f32>; 3] = [
            vec![0f32; t_len * d_inner],
            vec![0f32; t_len * d_inner],
            vec![0f32; t_len * d_inner],
        ];
        for ((p, c), cw) in proj.iter().zip(convd.iter_mut()).zip([&w.conv_q, &w.conv_k, &w.conv_v])
        {
            for (t, row) in c.chunks_mut(d_inner).enumerate() {
                for ch in 0..d_inner {
                    let mut acc = 0f32;
                    for k in 0..d_conv {
                        let idx = t as i64 - win as i64 + k as i64;
                        let v = if idx >= 0 {
                            p[idx as usize * d_inner + ch]
                        } else {
                            0.0
                        };
                        acc += cw[ch * d_conv + k] * v;
                    }
                    row[ch] = silu(acc);
                }
            }
        }
        // HF's saved q/k/v are pre-l2 (the norm happens inside the recurrence)
        let pre_l2: [Vec<f32>; 3] = [convd[0].clone(), convd[1].clone(), convd[2].clone()];
        for c in convd[..2].iter_mut() {
            c.chunks_mut(hd).for_each(l2norm);
        }
        let fa = w.f_a.apply(&x, t_len, false);
        let mut g1 = w.f_b.apply(&fa, t_len, false);
        {
            let b: Vec<u8> = fa.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write("/tmp/my_fa.bin", b).unwrap();
        }
        {
            let m = &w.f_b;
            // manual row-dot for token 1, rows 0..4
            let fa1 = &fa[64..128];
            let manual: Vec<f32> = (0..4)
                .map(|r| {
                    let row = &m.data[r * m.cols..(r + 1) * m.cols];
                    crate::flash::reference::dot(row, fa1)
                })
                .collect();
            println!("manual f_b[1,:4] {manual:?}");
            println!("f_b rows {} cols {} dt[:4] {:?} fbrow0[:6] {:?}", m.rows, m.cols, &w.dt_bias[..4], &m.data[..6]);
        }
        for (row, dtr) in g1.chunks_mut(d_inner).zip(std::iter::repeat(&w.dt_bias)) {
            for (g, dt) in row.iter_mut().zip(dtr) {
                *g += dt;
            }
        }
        // write: q conv (post l2), g1, beta raw
        for (name, data) in [
            ("/tmp/my_qconv.bin", &convd[0]),
            ("/tmp/my_kconv.bin", &convd[1]),
            ("/tmp/my_vconv.bin", &convd[2]),
            ("/tmp/my_g1.bin", &g1),
        ] {
            let b: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(name, b).unwrap();
        }
        let mut beta = w.beta.apply(&x, t_len, false);
        for b in beta.iter_mut() {
            *b = sigmoid(*b);
        }
        let b: Vec<u8> = beta.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write("/tmp/my_beta.bin", b).unwrap();

        // HF's saved q/k/v are post-conv pre-l2; g is the final gate
        let d = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };
        let rd = |p: &str| -> Option<Vec<f32>> {
            std::fs::read(p)
                .ok()?
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect::<Vec<_>>()
                .into()
        };
        for name in ["q", "k", "v", "g", "beta"] {
            let hf = rd(&format!("/tmp/hf_{name}.bin")).unwrap_or_default();
            if hf.is_empty() {
                continue;
            }
            let mine = match name {
                "q" => &pre_l2[0],
                "k" => &pre_l2[1],
                "v" => &pre_l2[2],
                "g" => &g1,
                _ => &beta,
            };
            // HF g is the final gate value (the log-decay); my g1 needs the lb·sigmoid transform
            let diff = if name == "g" {
                let hg: Vec<f32> = (0..t_len * d_inner)
                    .map(|i| {
                        let h = (i % d_inner) / hd;
                        match w.lower_bound {
                            // ssm_a = -exp(A_log); g = lb · sigmoid(exp(A_log)·g1)
                            Some(lb) => lb * sigmoid(-w.a[h] * g1[i]),
                            None => w.a[h] * softplus(g1[i]),
                        }
                    })
                    .collect();
                let hf = rd("/tmp/hf_g.bin").unwrap();
                let mut wi = 0;
                for (j, (a, b)) in hg.iter().zip(&hf).enumerate() {
                    if (a - b).abs() > (hg[wi] - hf[wi]).abs() {
                        wi = j;
                    }
                }
                println!(
                    "g worst at {wi}: mine {} hf {} (g1 {})",
                    hg[wi],
                    hf[wi],
                    g1[wi]
                );
                d(&hg, &hf)
            } else {
                d(mine, &hf)
            };
            println!("{name}: maxabs {diff}");
        }
    }

    #[test]
    fn placeholder() {}
}


