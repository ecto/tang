//! Mixture-of-experts on the dense decoder path, and the other pieces gpt-oss needs there.
//!
//! These are the host references the [`ComputeDevice`](crate::ComputeDevice) defaults run and
//! the GPU kernels are checked against. Nothing here is fixed to one model's geometry (unlike
//! [`crate::flash`]): expert count, top-k, widths and the weight format are arguments.
//!
//! - [`Router`]: how router logits become `(expert ids, weights)` per row.
//! - [`ExpertFmt`]: how a stack of expert matrices `[experts, n, k]` is stored on the device.
//! - [`moe_linear`]: each row's picked experts applied to it (gate/up or down), with biases.
//! - [`swiglu_clamped`]: gpt-oss's activation.
//! - [`moe_combine`]: the weighted sum of each row's expert outputs.
//! - [`attention_sinks`]: causal attention with a learned per-head sink logit.

/// How router logits `[rows, experts]` pick and weight experts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Router {
    /// Top-k by logit, then a softmax over those k (gpt-oss, Mixtral).
    TopKSoftmax,
    /// Softmax over every expert, then the top k by probability; `renorm` rescales the k to
    /// sum to 1 (Qwen3 MoE's `norm_topk_prob`, which makes it equal to `TopKSoftmax`).
    SoftmaxTopK { renorm: bool },
}

/// How a stack of expert matrices `[experts, n, k]` (rows are outputs, as checkpoints store a
/// linear layer) is kept on the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertFmt {
    /// f32 (or bf16 widened by `upload_bf16` on devices without bf16 storage).
    F32,
    /// bfloat16 bits (`upload_bf16`).
    Bf16,
    /// OCP MXFP4 ([`Mxfp4`]), uploaded with `upload_bytes`.
    Mxfp4,
}

/// OCP microscaling FP4: blocks of 32 weights along `k` share an E8M0 scale (`2^(s - 127)`),
/// each weight an E2M1 nibble, two to a byte, the even element in the low nibble. This is
/// gpt-oss's checkpoint format (`*_blocks` `[experts, n, k/32, 16]` u8, `*_scales`
/// `[experts, n, k/32]` u8), and the device layout is the same two arrays back to back:
/// every expert's blocks, then every expert's scales.
pub struct Mxfp4;

/// E2M1 code → value.
pub const FP4: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

impl Mxfp4 {
    /// Bytes of one expert's blocks (`n * k / 2`) and scales (`n * k / 32`).
    pub fn sizes(n: usize, k: usize) -> (usize, usize) {
        (n * k / 2, n * k / 32)
    }

    /// The device layout of `experts` matrices from checkpoint `blocks` and `scales`.
    pub fn pack(blocks: &[u8], scales: &[u8], experts: usize, n: usize, k: usize) -> Vec<u8> {
        let (bs, ss) = Self::sizes(n, k);
        assert!(k.is_multiple_of(32), "MXFP4 rows must be a multiple of 32");
        assert_eq!(blocks.len(), experts * bs, "MXFP4 blocks");
        assert_eq!(scales.len(), experts * ss, "MXFP4 scales");
        let mut out = Vec::with_capacity(blocks.len() + scales.len());
        out.extend_from_slice(blocks);
        out.extend_from_slice(scales);
        out
    }

    /// Row `r` of expert `e` as f32.
    pub fn row(packed: &[u8], experts: usize, n: usize, k: usize, e: usize, r: usize) -> Vec<f32> {
        let (bs, ss) = Self::sizes(n, k);
        let blocks = &packed[e * bs + r * k / 2..][..k / 2];
        let scales = &packed[experts * bs + e * ss + r * k / 32..][..k / 32];
        let mut out = Vec::with_capacity(k);
        for (b, &s) in scales.iter().enumerate() {
            let m = e8m0(s);
            for &byte in &blocks[b * 16..b * 16 + 16] {
                out.push(FP4[(byte & 15) as usize] * m);
                out.push(FP4[(byte >> 4) as usize] * m);
            }
        }
        out
    }

    /// Reorder each expert's rows: new row `i` is old row `perm[i]` (blocks and scales alike).
    pub fn permute_rows(
        packed: &[u8],
        experts: usize,
        n: usize,
        k: usize,
        perm: &[usize],
    ) -> Vec<u8> {
        let (bs, ss) = Self::sizes(n, k);
        let (rb, rs) = (k / 2, k / 32);
        let mut out = vec![0u8; packed.len()];
        for e in 0..experts {
            for (i, &p) in perm.iter().enumerate() {
                out[e * bs + i * rb..][..rb].copy_from_slice(&packed[e * bs + p * rb..][..rb]);
                let (o, s) = (experts * bs + e * ss, experts * bs + e * ss);
                out[o + i * rs..][..rs].copy_from_slice(&packed[s + p * rs..][..rs]);
            }
        }
        out
    }
}

/// E8M0 scale byte → `2^(s - 127)` (255 is NaN in the spec; checkpoints don't use it).
pub fn e8m0(s: u8) -> f32 {
    if s == 0 {
        // 2^-127 is subnormal in f32: exact.
        return f32::from_bits(1 << 22);
    }
    f32::from_bits((s as u32) << 23)
}

/// Pick experts for `rows` rows of router `logits` (`[rows, experts]`): `(ids, weights)`, each
/// `[rows, k]`, in rank order (logit descending; ties to the lower index, as `torch.topk` on
/// the CPU does).
pub fn route(
    logits: &[f32],
    rows: usize,
    experts: usize,
    k: usize,
    router: Router,
) -> (Vec<u32>, Vec<f32>) {
    let (mut ids, mut ws) = (Vec::with_capacity(rows * k), Vec::with_capacity(rows * k));
    for r in 0..rows {
        let l = &logits[r * experts..][..experts];
        let mut order: Vec<usize> = (0..experts).collect();
        order.sort_by(|&a, &b| l[b].total_cmp(&l[a]).then(a.cmp(&b)));
        let top = &order[..k];
        let m = l[top[0]];
        match router {
            Router::TopKSoftmax => {
                let e: Vec<f32> = top.iter().map(|&i| (l[i] - m).exp()).collect();
                let z: f32 = e.iter().sum();
                ws.extend(e.iter().map(|x| x / z));
            }
            Router::SoftmaxTopK { renorm } => {
                let z: f32 = l.iter().map(|&x| (x - m).exp()).sum();
                let p: Vec<f32> = top.iter().map(|&i| (l[i] - m).exp() / z).collect();
                let s: f32 = if renorm { p.iter().sum() } else { 1.0 };
                ws.extend(p.iter().map(|x| x / s));
            }
        }
        ids.extend(top.iter().map(|&i| i as u32));
    }
    (ids, ws)
}

/// One stack of expert matrices as the host sees it.
pub enum ExpertWeights<'a> {
    /// `[experts, n, k]` f32.
    F32(&'a [f32]),
    /// [`Mxfp4`] device layout.
    Mxfp4(&'a [u8]),
}

/// `out[r * k_top + i] = W[ids[r * k_top + i]] · x_row + bias[that expert]`, `[rows * k_top, n]`.
/// `x` is `[rows, k]` (every pick of a row reads the row: gate/up), or with `per_pick`
/// `[rows * k_top, k]` (each pick its own row: down).
#[allow(clippy::too_many_arguments)]
pub fn moe_linear(
    x: &[f32],
    w: ExpertWeights,
    experts: usize,
    bias: Option<&[f32]>,
    ids: &[u32],
    k_top: usize,
    k: usize,
    n: usize,
    per_pick: bool,
) -> Vec<f32> {
    let picks = ids.len();
    let mut out = vec![0f32; picks * n];
    for (p, &e) in ids.iter().enumerate() {
        let e = e as usize;
        let xr = if per_pick {
            &x[p * k..][..k]
        } else {
            &x[p / k_top * k..][..k]
        };
        for r in 0..n {
            let dot = match w {
                ExpertWeights::F32(w) => dot(&w[(e * n + r) * k..][..k], xr),
                ExpertWeights::Mxfp4(w) => dot(&Mxfp4::row(w, experts, n, k, e, r), xr),
            };
            out[p * n + r] = dot + bias.map_or(0.0, |b| b[e * n + r]);
        }
    }
    out
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// gpt-oss's gated activation over `gu` (`[rows, 2 * ff]`, gate then up): `gate` clamped above
/// at `limit`, `up` to `±limit`, then `(up + 1) · gate · σ(alpha · gate)`.
pub fn swiglu_clamped(gu: &[f32], rows: usize, ff: usize, alpha: f32, limit: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * ff);
    for r in 0..rows {
        let (g, u) = gu[r * 2 * ff..][..2 * ff].split_at(ff);
        for (&g, &u) in g.iter().zip(u) {
            let g = g.min(limit);
            let u = u.clamp(-limit, limit);
            out.push((u + 1.0) * (g / (1.0 + (-alpha * g).exp())));
        }
    }
    out
}

/// `y[r] = Σ_i w[r * k_top + i] · parts[r * k_top + i]`, `[rows, h]`.
pub fn moe_combine(parts: &[f32], w: &[f32], rows: usize, k_top: usize, h: usize) -> Vec<f32> {
    let mut y = vec![0f32; rows * h];
    for r in 0..rows {
        for i in 0..k_top {
            let p = r * k_top + i;
            for (o, v) in y[r * h..][..h].iter_mut().zip(&parts[p * h..][..h]) {
                *o += w[p] * v;
            }
        }
    }
    y
}

/// Attention over contiguous `k`/`v` (`[keys, nkv * hd]`, the first `cache_start` keys already
/// cached) for `q_len` queries `[q_len, nh * hd]` at positions `cache_start..`, scale
/// `hd^-0.5`. Each query sees the keys up to its own (all of them without `causal`), only the
/// last `window` of those when `window > 0`, and, with `sinks`, one more logit per head
/// (`sinks[h]`) whose value is zero: it takes probability mass without adding to the output.
#[allow(clippy::too_many_arguments)]
pub fn attention_sinks(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    cache_start: usize,
    q_len: usize,
    (nh, nkv, hd): (usize, usize, usize),
    window: usize,
    causal: bool,
    sinks: Option<&[f32]>,
) -> Vec<f32> {
    let (kvd, scale) = (nkv * hd, 1.0 / (hd as f32).sqrt());
    let mut out = vec![0f32; q_len * nh * hd];
    for qi in 0..q_len {
        let qpos = cache_start + qi + 1;
        let end = if causal { qpos } else { cache_start + q_len };
        let lo = if window > 0 {
            qpos.saturating_sub(window)
        } else {
            0
        };
        for h in 0..nh {
            let kh = h / (nh / nkv);
            let qv = &q[(qi * nh + h) * hd..][..hd];
            let s: Vec<f32> = (lo..end)
                .map(|j| dot(qv, &k[j * kvd + kh * hd..][..hd]) * scale)
                .collect();
            let sink = sinks.map(|s| s[h]);
            let m = s
                .iter()
                .copied()
                .chain(sink)
                .fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = s.iter().map(|x| (x - m).exp()).collect();
            let z: f32 = e.iter().sum::<f32>() + sink.map_or(0.0, |x| (x - m).exp());
            let o = &mut out[(qi * nh + h) * hd..][..hd];
            for (t, j) in (lo..end).enumerate() {
                let p = e[t] / z;
                for (d, od) in o.iter_mut().enumerate() {
                    *od += p * v[j * kvd + kh * hd + d];
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mxfp4_rows_decode_low_nibble_first_with_their_scale() {
        // One expert, two rows of 32: row 0 scale 2^0, row 1 scale 2^-1.
        let mut blocks = vec![0u8; 32];
        blocks[0] = 0x21; // elements 0, 1: codes 1 (0.5), 2 (1.0)
        blocks[16] = 0xf7; // row 1 elements 0, 1: codes 7 (6.0), 15 (-6.0)
        let packed = Mxfp4::pack(&blocks, &[127, 126], 1, 2, 32);
        let r0 = Mxfp4::row(&packed, 1, 2, 32, 0, 0);
        let r1 = Mxfp4::row(&packed, 1, 2, 32, 0, 1);
        assert_eq!(&r0[..3], &[0.5, 1.0, 0.0]);
        assert_eq!(&r1[..2], &[3.0, -3.0]);
        let swapped = Mxfp4::permute_rows(&packed, 1, 2, 32, &[1, 0]);
        assert_eq!(Mxfp4::row(&swapped, 1, 2, 32, 0, 0), r1);
        assert_eq!(e8m0(0), 2f32.powi(-127));
    }

    #[test]
    fn routers_agree_when_the_top_k_is_renormalised() {
        let l = [0.3, -1.0, 2.0, 0.3, 1.5];
        let (a, wa) = route(&l, 1, 5, 3, Router::TopKSoftmax);
        let (b, wb) = route(&l, 1, 5, 3, Router::SoftmaxTopK { renorm: true });
        assert_eq!(a, vec![2, 4, 0]);
        assert_eq!(a, b);
        for (x, y) in wa.iter().zip(&wb) {
            assert!((x - y).abs() < 1e-6);
        }
    }

    #[test]
    fn a_sink_only_shrinks_the_output() {
        let (q, k, v) = ([1.0, 0.0], [1.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 4.0]);
        let plain = attention_sinks(&q, &k, &v, 1, 1, (1, 1, 2), 0, true, None);
        let sunk = attention_sinks(&q, &k, &v, 1, 1, (1, 1, 2), 0, true, Some(&[0.0]));
        // With a sink at logit 0: one more exp(0 - m) in the denominator.
        let (s0, s1) = (1.0 / 2f32.sqrt(), 0.0);
        let (e0, e1) = (s0.exp(), f32::exp(s1));
        assert!((plain[0] - 2.0 * e0 / (e0 + e1)).abs() < 1e-6);
        assert!((sunk[0] - 2.0 * e0 / (e0 + e1 + 1.0)).abs() < 1e-6);
        assert!((sunk[1] - 4.0 * e1 / (e0 + e1 + 1.0)).abs() < 1e-6);
    }
}
