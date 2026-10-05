//! Reference math for the Flash-Next decode ops (see [`crate::flash`] for the contracts). These
//! are the spec the GPU kernels are tested against: plain loops over host slices, with the
//! summation orders the contracts pin written out (`mul_add` where a kernel pins an fma).
//! Word buffers are `u32`.

use crate::flash::shape::*;
use crate::flash::{f16_to_f32, ExpertBlob, MoePlan, QAct, MAX_T};

pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub(crate) fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// `v[0]` after the 32-lane xor butterfly `v[l] += v[l ^ o]`, o = 16, 8, 4, 2, 1.
pub(crate) fn butterfly<T: Copy + std::ops::Add<Output = T>>(mut v: [T; 32]) -> T {
    for o in [16, 8, 4, 2, 1] {
        let prev = v;
        for l in 0..32 {
            v[l] = prev[l] + prev[l ^ o];
        }
    }
    v[0]
}

// ---- int8 activations and Q2_0 ----

/// Quantize `m` rows of `k` per the [`QAct`] contract.
pub(crate) fn quantize_act(x: &[f32], m: usize, k: usize) -> Vec<u32> {
    let l = QAct { m, k };
    let mut w = vec![0u32; l.words()];
    quantize_act_rows(x, l, 0, m, &mut w);
    w
}

/// Quantize rows `r0..r0 + rows` of `x` (each `l.k` wide, `x` starting at row `r0`) into `w`.
pub(crate) fn quantize_act_rows(x: &[f32], l: QAct, r0: usize, rows: usize, w: &mut [u32]) {
    let k = l.k;
    for r in r0..r0 + rows {
        let xr = &x[(r - r0) * k..(r - r0 + 1) * k];
        for i in 0..k / 4 {
            w[l.codes(r) + i] = 0;
        }
        for c in 0..k / 32 {
            let xs = &xr[c * 32..c * 32 + 32];
            let amax = xs.iter().fold(0f32, |a, v| a.max(v.abs()));
            let d = amax / 127.0;
            let mut hx = 0i32;
            for (i, &v) in xs.iter().enumerate() {
                let q = if d == 0.0 {
                    0
                } else {
                    (v / d).round().clamp(-127.0, 127.0) as i32
                };
                hx += q;
                let (wi, b) = QAct::slot(c * 32 + i);
                w[l.codes(r) + wi] |= ((q as i8 as u8) as u32) << (8 * b);
            }
            w[l.scales(r) + c] = d.to_bits();
            w[l.sums(r) + c] = hx as u32;
        }
    }
}

/// One row of int8 activations, decoded from [`QAct`] words into element order.
pub(crate) struct QRow {
    pub(crate) q: Vec<i32>,
    pub(crate) d: Vec<f32>,
    pub(crate) hx: Vec<i32>,
}

impl QRow {
    pub(crate) fn decode(xq: &[u32], l: QAct, r: usize) -> Self {
        let q = (0..l.k)
            .map(|e| {
                let (wi, by) = QAct::slot(e);
                (xq[l.codes(r) + wi] >> (8 * by)) as u8 as i8 as i32
            })
            .collect();
        let d = (0..l.k / 32)
            .map(|c| f32::from_bits(xq[l.scales(r) + c]))
            .collect();
        let hx = (0..l.k / 32).map(|c| xq[l.sums(r) + c] as i32).collect();
        QRow { q, d, hx }
    }
}

/// One Q2_0 block (16 code bytes, scale `d`) at 64-wide column `c` against a row of int8
/// activations: its two chunks, `fma(d · d_x, Σ code·q − Σ q, acc)` each, ascending.
pub(crate) fn q2_block_dot(codes: &[u8], d: f32, c: usize, x: &QRow, mut acc: f32) -> f32 {
    for h in 0..2 {
        let ch = 2 * c + h;
        let mut s = 0i32;
        for (j, &byte) in codes[8 * h..8 * h + 8].iter().enumerate() {
            for f in 0..4 {
                s += ((byte >> (2 * f)) & 3) as i32 * x.q[ch * 32 + 4 * j + f];
            }
        }
        acc = (d * x.d[ch]).mul_add((s - x.hx[ch]) as f32, acc);
    }
    acc
}

/// Row `o` of a repacked `[n, k]` Q2_0 matrix `wb` against a row of int8 activations: the
/// row's blocks through [`q2_block_dot`], ascending.
pub(crate) fn q2_row_dot(wb: &[u8], n: usize, k: usize, o: usize, x: &QRow) -> f32 {
    let codes = &wb[o * k / 4..(o + 1) * k / 4];
    let sc = n * k / 4 + o * (k / 64) * 2;
    let mut acc = 0f32;
    for c in 0..k / 64 {
        let d = f16_to_f32(u16::from_le_bytes([wb[sc + 2 * c], wb[sc + 2 * c + 1]]));
        acc = q2_block_dot(&codes[16 * c..16 * c + 16], d, c, x, acc);
    }
    acc
}

/// `out[m, n] = W · x̂` for repacked Q2_0 `W` `[n, k]` and quantized `x` (`m` rows).
pub(crate) fn q2_linear(xq: &[u32], wb: &[u8], m: usize, k: usize, n: usize) -> Vec<f32> {
    let l = QAct { m, k };
    let mut out = vec![0f32; m * n];
    for r in 0..m {
        let x = QRow::decode(xq, l, r);
        for o in 0..n {
            out[r * n + o] = q2_row_dot(wb, n, k, o, &x);
        }
    }
    out
}

/// Row `o` of a Q4X matrix ([`crate::flash::q4x_repack`]) against a row of int8 activations:
/// per chunk `S = Σ q·x̂`, `acc = fma(d_x, fma(scale, S, bias · Σ x̂), acc)`, chunks ascending.
pub(crate) fn q4x_row_dot(wb: &[u8], n: usize, k: usize, o: usize, x: &QRow) -> f32 {
    let codes = &wb[o * k / 2..(o + 1) * k / 2];
    let bf = |off: usize| f32::from_bits((u16::from_le_bytes([wb[off], wb[off + 1]]) as u32) << 16);
    let (sp, bp) = (n * k / 2, n * k / 2 + 2 * (n * k / 64));
    let mut acc = 0f32;
    for c in 0..k / 32 {
        let g = o * (k / 64) + c / 2;
        let (s, b) = (bf(sp + 2 * g), bf(bp + 2 * g));
        let mut sum = 0i32;
        for i in 0..32 {
            let (byte, sh) = crate::flash::q4x_slot(i);
            sum += ((codes[c * 16 + byte] >> sh) & 0xf) as i32 * x.q[c * 32 + i];
        }
        let v = s.mul_add(sum as f32, b * x.hx[c] as f32);
        acc = x.d[c].mul_add(v, acc);
    }
    acc
}

/// `out[m, n] = W · x̂` for a Q4X weight `[n, k]`.
pub(crate) fn q4x_linear(xq: &[u32], wb: &[u8], m: usize, k: usize, n: usize) -> Vec<f32> {
    let l = QAct { m, k };
    let mut out = vec![0f32; m * n];
    for r in 0..m {
        let x = QRow::decode(xq, l, r);
        for o in 0..n {
            out[r * n + o] = q4x_row_dot(wb, n, k, o, &x);
        }
    }
    out
}

/// `out[m, n] = W · x̂` for a Q8X weight `[n, k]` ([`crate::flash::q8x_repack`]): per half
/// chunk `fma(d_w · d_x, Σ w·q, acc)`, ascending.
pub(crate) fn q8x_linear(xq: &[u32], wb: &[u8], m: usize, k: usize, n: usize) -> Vec<f32> {
    let l = QAct { m, k };
    let mut out = vec![0f32; m * n];
    for r in 0..m {
        let x = QRow::decode(xq, l, r);
        for o in 0..n {
            let mut acc = 0f32;
            for c in 0..k / 32 {
                let so = n * k + 2 * (o * k / 32 + c);
                let d = f16_to_f32(u16::from_le_bytes([wb[so], wb[so + 1]]));
                for h in 0..2 {
                    let mut s = 0i32;
                    for b in 0..4 {
                        for f in 0..4 {
                            let w = wb[o * k + c * 32 + 16 * h + 4 * f + b] as i8 as i32;
                            s += w * x.q[c * 32 + 16 * h + 4 * b + f];
                        }
                    }
                    acc = (d * x.d[c]).mul_add(s as f32, acc);
                }
            }
            out[r * n + o] = acc;
        }
    }
    out
}

/// `out[m, n] = W · x̂` for a NatX weight ([`crate::flash_native`]): per chunk the exact
/// integer half sums, the type's scale combination, `fma(d_x, v, acc)`, chunks ascending.
pub(crate) fn native_linear(
    ty: crate::flash_native::NatType,
    xq: &[u32],
    wb: &[u8],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let ir = crate::flash_native::nat_unpack(ty, wb, n, k);
    let l = QAct { m, k };
    let mut out = vec![0f32; m * n];
    for r in 0..m {
        let x = QRow::decode(xq, l, r);
        for o in 0..n {
            let mut acc = 0f32;
            for c in 0..k / 32 {
                let e0 = o * k + c * 32;
                let mut s = [0i32; 2];
                for (hh, sh) in s.iter_mut().enumerate() {
                    for i in 0..16 {
                        *sh += ir.code[e0 + 16 * hh + i] as i32 * x.q[c * 32 + 16 * hh + i];
                    }
                }
                let (sc0, sc1) = (ir.scale[e0 / 16], ir.scale[e0 / 16 + 1]);
                let v = if ty.per16() {
                    sc1.mul_add(s[1] as f32, sc0 * s[0] as f32)
                } else if ty.has_min() {
                    sc0.mul_add((s[0] + s[1]) as f32, -(ir.min[e0 / 32] * x.hx[c] as f32))
                } else {
                    sc0 * (s[0] + s[1]) as f32
                };
                acc = x.d[c].mul_add(v, acc);
            }
            out[r * n + o] = acc;
        }
    }
    out
}

// ---- hyper-connections ----

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `R[t][c] += y[t] · 2σ(inj[t][c] / 4)`, as `fma(y, g, R)`.
pub(crate) fn hc_write(r: &mut [f32], y: &[f32], inj: &[f32], t: usize) {
    for tt in 0..t {
        for c in 0..HC {
            let g = 2.0 / (1.0 + (-(inj[tt * HC + c] * 0.25)).exp());
            let row = &mut r[(tt * HC + c) * HIDDEN..(tt * HC + c + 1) * HIDDEN];
            for (d, v) in row.iter_mut().enumerate() {
                *v = y[tt * HIDDEN + d].mul_add(g, *v);
            }
        }
    }
}

/// Hyper-connection read (strata.md): returns `(x [t][HIDDEN], inj [t][HC])`. With `pending`,
/// first applies that write to `r` in place (the fused write-then-read).
#[allow(clippy::too_many_arguments)]
pub(crate) fn hc_read(
    r: &mut [f32],
    pending: Option<(&[f32], &[f32])>,
    norm: &[f32],
    down: &[f32],
    up: &[f32],
    inject: Option<&[f32]>,
    t: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    if let Some((y, inj)) = pending {
        hc_write(r, y, inj, t);
    }
    let w = HC * HIDDEN;
    let (mut x, mut inj_out) = (vec![0f32; t * HIDDEN], vec![0f32; t * HC]);
    for tt in 0..t {
        let rt = &r[tt * w..(tt + 1) * w];
        let mut xn = vec![0f32; w];
        for c in 0..HC {
            let row = &rt[c * HIDDEN..(c + 1) * HIDDEN];
            let ss: f32 = row.iter().map(|v| v * v).sum();
            let rs = 1.0 / (ss / HIDDEN as f32 + eps).sqrt();
            for d in 0..HIDDEN {
                xn[c * HIDDEN + d] = row[d] * rs * norm[c * HIDDEN + d];
            }
        }
        let lo: Vec<f32> = (0..HC_LR)
            .map(|k| silu(dot(&down[k * w..(k + 1) * w], &xn) * 0.25))
            .collect();
        if let Some(inj) = inject {
            for c in 0..HC {
                inj_out[tt * HC + c] = dot(&inj[c * w..(c + 1) * w], &xn);
            }
        }
        for d in 0..HIDDEN {
            let mut acc = 0f32;
            for c in 0..HC {
                let i = c * HIDDEN + d;
                let row: f32 = (0..HC_LR)
                    .map(|k| up[crate::flash::hc_up_index(i, k)] * lo[k])
                    .sum();
                let g = sigmoid(row);
                acc = xn[i].mul_add(g, acc);
            }
            x[tt * HIDDEN + d] = acc * 0.25;
        }
    }
    (x, inj_out)
}

// ---- GDN ----

/// `h[t][GDN_CONV]`: causal 4-tap conv over `[hist | qkv_0..]` (fma chain, oldest tap first),
/// SiLU, then each q and k head scaled by `1 / sqrt(Σx² + eps)`.
pub(crate) fn gdn_conv(
    proj: &[f32],
    stride: usize,
    hist: &[f32],
    w: &[f32],
    t: usize,
    eps: f32,
) -> Vec<f32> {
    let ext = |e: usize, ch: usize| {
        if e < GDN_TAPS - 1 {
            hist[e * GDN_CONV + ch]
        } else {
            proj[(e - (GDN_TAPS - 1)) * stride + ch]
        }
    };
    let mut h = vec![0f32; t * GDN_CONV];
    for tt in 0..t {
        for ch in 0..GDN_CONV {
            let mut acc = ext(tt, ch) * w[ch * 4];
            for i in 1..GDN_TAPS {
                acc = ext(tt + i, ch).mul_add(w[ch * 4 + i], acc);
            }
            h[tt * GDN_CONV + ch] = silu(acc);
        }
        for head in 0..2 * GDN_HK {
            let s = &mut h[tt * GDN_CONV + head * GDN_D..tt * GDN_CONV + (head + 1) * GDN_D];
            let ss: f32 = s.iter().map(|v| v * v).sum();
            let rs = 1.0 / (ss + eps).sqrt();
            for v in s.iter_mut() {
                *v *= rs;
            }
        }
    }
    h
}

/// Conv history after keeping `n` tokens: the last 3 rows of `[hist | qkv_0 .. qkv_{n−1}]`.
pub(crate) fn gdn_conv_commit(hist: &mut [f32], proj: &[f32], stride: usize, n: usize) {
    let old = hist.to_vec();
    for s in 0..GDN_TAPS - 1 {
        let e = n + s;
        for ch in 0..GDN_CONV {
            hist[s * GDN_CONV + ch] = if e < GDN_TAPS - 1 {
                old[e * GDN_CONV + ch]
            } else {
                proj[(e - (GDN_TAPS - 1)) * stride + ch]
            };
        }
    }
}

/// 1 / sqrt(128) as f32.
pub(crate) const INV_SQRT_D: f32 = f32::from_bits(0x3db5_04f3);

/// The GDN recurrence (see [`crate::flash::GDN_STATE`]) for the first `n_run` tokens; writes the
/// state back when `write`. Returns `y [t][GDN_V]` (rows past `n_run` are zero).
#[allow(clippy::too_many_arguments)]
pub(crate) fn gdn_step(
    state: &mut [f32],
    h: &[f32],
    proj: &[f32],
    stride: usize,
    (dt, ssm_a, norm): (&[f32], &[f32], &[f32]),
    t: usize,
    n_run: usize,
    write: bool,
    eps: f32,
) -> Vec<f32> {
    let d = GDN_D;
    let mut y = vec![0f32; t * GDN_V];
    for hv in 0..GDN_HV {
        let hk = hv % GDN_HK;
        let mut s = vec![0f32; d * d];
        for i in 0..d {
            for j in 0..d {
                s[i * d + j] = state[(i * GDN_HV + hv) * d + j];
            }
        }
        for tt in 0..n_run {
            let p = &proj[tt * stride..];
            let a = p[GDN_A + hv] + dt[hv];
            let sp = if a > 20.0 { a } else { a.exp().ln_1p() };
            let g = (sp * ssm_a[hv]).exp();
            let beta = sigmoid(p[GDN_B + hv]);
            let hh = &h[tt * GDN_CONV..(tt + 1) * GDN_CONV];
            let q = &hh[hk * d..(hk + 1) * d];
            let k = &hh[GDN_HK * d + hk * d..GDN_HK * d + (hk + 1) * d];
            let v = &hh[2 * GDN_HK * d + hv * d..2 * GDN_HK * d + (hv + 1) * d];
            let grouped = |s: &[f32], vec: &[f32], j: usize| {
                let mut parts = [0f32; 4];
                for (rg, part) in parts.iter_mut().enumerate() {
                    let mut acc = 0f32;
                    for r in 0..32 {
                        let i = rg * 32 + r;
                        acc = s[i * d + j].mul_add(vec[i], acc);
                    }
                    *part = acc;
                }
                ((parts[0] + parts[1]) + parts[2]) + parts[3]
            };
            let mut o = vec![0f32; d];
            for v_ in s.iter_mut() {
                *v_ *= g;
            }
            for j in 0..d {
                let sk = grouped(&s, k, j);
                let dj = (v[j] - sk) * beta;
                for i in 0..d {
                    s[i * d + j] = k[i].mul_add(dj, s[i * d + j]);
                }
                o[j] = grouped(&s, q, j) * INV_SQRT_D;
            }
            let ss: f32 = o.iter().map(|x| x * x).sum();
            let rs = 1.0 / (ss / d as f32 + eps).sqrt();
            for j in 0..d {
                let z = p[GDN_Z + hv * d + j];
                y[tt * GDN_V + hv * d + j] = o[j] * rs * norm[j] * sigmoid(z);
            }
        }
        if write {
            for i in 0..d {
                for j in 0..d {
                    state[(i * GDN_HV + hv) * d + j] = s[i * d + j];
                }
            }
        }
    }
    y
}

// ---- MoE ----

/// Router: per token, the top `TOPK` experts by (logit desc, index asc), weighted
/// `e_k / Σ_top e_j` with `e_k = exp(l_k − l_max)` in f64 (the sum in rank order), as f32.
///
/// That is the softmax-over-all-experts weight renormalised over the top k,
/// `p_k / max(Σ_top p, 2^-14)`: `Z` cancels, and the clamp cannot bind because the top k of
/// `n` probabilities sum to at least `k / n` (10 / 512 here). Returns `(ids, w)`, `[t][TOPK]`.
pub(crate) fn router_topk(
    logits: &[f32],
    stride: usize,
    t: usize,
    n_expert: usize,
) -> (Vec<u32>, Vec<f32>) {
    assert!(TOPK as f64 / n_expert as f64 >= 1.0 / 16384.0);
    let (mut ids, mut ws) = (Vec::with_capacity(t * TOPK), Vec::with_capacity(t * TOPK));
    for tt in 0..t {
        let l = &logits[tt * stride..tt * stride + n_expert];
        let mut idx: Vec<usize> = (0..n_expert).collect();
        idx.sort_by(|&a, &b| l[b].partial_cmp(&l[a]).unwrap().then(a.cmp(&b)));
        let m = l[idx[0]] as f64;
        let e: Vec<f64> = idx[..TOPK]
            .iter()
            .map(|&i| (l[i] as f64 - m).exp())
            .collect();
        let sum = e.iter().fold(0f64, |a, v| a + v);
        for (k, &i) in idx[..TOPK].iter().enumerate() {
            ids.push(i as u32);
            ws.push((e[k] / sum) as f32);
        }
    }
    (ids, ws)
}

/// The [`MoePlan`] for router ids `[t][TOPK]` against a residency table of `EXPERTS` addresses
/// (0 = not resident), plus the shared expert when `shared != 0`.
pub(crate) fn moe_plan(ids: &[u32], table: &[u64], shared: u64, t: usize) -> Vec<u32> {
    let mut plan = vec![0u32; MoePlan::WORDS];
    let n = t * TOPK;
    let (mut groups, mut missing): (Vec<(u64, Vec<usize>)>, Vec<u32>) = (vec![], vec![]);
    let mut seen: Vec<u32> = vec![];
    for i in 0..n {
        let e = ids[i];
        if seen.contains(&e) {
            continue;
        }
        seen.push(e);
        let ptr = table[e as usize];
        if ptr == 0 {
            missing.push(e);
            continue;
        }
        groups.push((ptr, (i..n).filter(|&j| ids[j] == e).collect()));
    }
    let mut ents: Vec<(u32, u32)> = vec![];
    let mut starts = vec![];
    for (_, es) in &groups {
        starts.push(ents.len() as u32);
        for &j in es {
            ents.push(((j / TOPK) as u32, j as u32));
        }
    }
    let mut ptrs: Vec<u64> = groups.iter().map(|g| g.0).collect();
    if shared != 0 {
        starts.push(ents.len() as u32);
        ptrs.push(shared);
        for tt in 0..t {
            ents.push((tt as u32, (MoePlan::SHARED_ROW + tt) as u32));
        }
    }
    let ng = ptrs.len();
    plan[0] = ng as u32;
    plan[1] = ents.len() as u32;
    plan[2] = missing.len() as u32;
    for (g, &p) in ptrs.iter().enumerate() {
        plan[MoePlan::GROUP_PTR + 2 * g] = p as u32;
        plan[MoePlan::GROUP_PTR + 2 * g + 1] = (p >> 32) as u32;
        plan[MoePlan::GROUP_START + g] = starts[g];
    }
    plan[MoePlan::GROUP_START + ng] = ents.len() as u32;
    for (e, &(tok, dst)) in ents.iter().enumerate() {
        plan[MoePlan::ENT_TOK + e] = tok;
        plan[MoePlan::ENT_DST + e] = dst;
    }
    for (i, &e) in missing.iter().enumerate() {
        plan[MoePlan::MISSING + i] = e;
    }
    plan
}

/// One expert row against a row of int8 activations, in the pinned order of
/// [`ExpertBlob::ORDER`]: eight lane accumulators over the row's 128-weight groups, then the
/// fixed tree. `group(g)` gives the byte offsets of the row's group `g` in `blob`.
pub(crate) fn expert_row_dot(
    blob: &[u8],
    groups: usize,
    group: impl Fn(usize) -> (usize, usize),
    x: &QRow,
) -> f32 {
    let mut acc = [0f32; 8];
    for g in 0..groups {
        let (co, so) = group(g);
        let v = &blob[co..co + 32];
        let dw = [
            f16_to_f32(u16::from_le_bytes([blob[so], blob[so + 1]])),
            f16_to_f32(u16::from_le_bytes([blob[so + 2], blob[so + 3]])),
        ];
        for (i, a) in acc.iter_mut().enumerate() {
            let (c, h) = (4 * g + i / 2, i % 2);
            let mut s = 0i32;
            for b in 0..4 {
                let byte = v[4 * i + b];
                for f in 0..4 {
                    s += ((byte >> (2 * f)) & 3) as i32 * x.q[c * 32 + 16 * h + 4 * b + f];
                }
            }
            if h == 0 {
                s -= x.hx[c];
            }
            *a = (s as f32).mul_add(dw[i / 4] * x.d[c], *a);
        }
    }
    ((acc[0] + acc[4]) + (acc[2] + acc[6])) + ((acc[1] + acc[5]) + (acc[3] + acc[7]))
}

/// SiLU with the pinned exponential: `x / (1 + pexp(−x))`.
pub(crate) fn psilu(x: f32) -> f32 {
    x / (1.0 + crate::flash::pexp(-x))
}

/// Evaluate every planned (expert, token) pair into `parts` rows: gate/up rows against the
/// token's int8 activations (`xq`, `QAct { m: t, k: HIDDEN }`), `h = psilu(g) · u`, `h`
/// quantized per the [`QAct`] contract, then down rows against it, every row in the pinned
/// order of [`ExpertBlob::ORDER`]. This is the spec both the GPU kernels and the CPU expert
/// kernels match bit for bit.
///
/// # Safety
/// Every group address in `plan` must point at a live [`ExpertBlob`] readable from the host.
pub(crate) unsafe fn moe_grouped(xq: &[u32], plan: &[u32], parts: &mut [f32], t: usize) {
    let lx = QAct { m: t, k: HIDDEN };
    let lh = QAct { m: 1, k: FF };
    let ng = plan[0] as usize;
    for g in 0..ng {
        let ptr = plan[MoePlan::GROUP_PTR + 2 * g] as u64
            | (plan[MoePlan::GROUP_PTR + 2 * g + 1] as u64) << 32;
        // SAFETY: the caller guarantees the address is a live blob.
        let blob = unsafe { std::slice::from_raw_parts(ptr as *const u8, ExpertBlob::BYTES) };
        for e in
            plan[MoePlan::GROUP_START + g] as usize..plan[MoePlan::GROUP_START + g + 1] as usize
        {
            let (tok, dst) = (
                plan[MoePlan::ENT_TOK + e] as usize,
                plan[MoePlan::ENT_DST + e] as usize,
            );
            let x = QRow::decode(xq, lx, tok);
            let hrow = expert_h(blob, &x);
            let hq = QRow::decode(&quantize_act(&hrow, 1, FF), lh, 0);
            for r in 0..HIDDEN {
                parts[dst * HIDDEN + r] =
                    expert_row_dot(blob, FF / 128, |g| ExpertBlob::down_group(r, g), &hq);
            }
        }
    }
}

/// An expert's SwiGLU activations `h [FF]` for one token, in the pinned order.
pub(crate) fn expert_h(blob: &[u8], x: &QRow) -> Vec<f32> {
    (0..FF)
        .map(|r| {
            let gv = expert_row_dot(blob, HIDDEN / 128, |g| ExpertBlob::gu_group(2 * r, g), x);
            let uv = expert_row_dot(
                blob,
                HIDDEN / 128,
                |g| ExpertBlob::gu_group(2 * r + 1, g),
                x,
            );
            psilu(gv) * uv
        })
        .collect()
}

/// `y[t] = Σ_i w[t][i] · parts[t·TOPK + i]` (fma chain, i ascending), then
/// `+ σ(logits[t][sg]) · parts[SHARED_ROW + t]` when `sg` names the shared-gate column.
pub(crate) fn moe_combine(
    parts: &[f32],
    w: &[f32],
    logits: &[f32],
    stride: usize,
    sg: Option<usize>,
    t: usize,
) -> Vec<f32> {
    let mut y = vec![0f32; t * HIDDEN];
    for tt in 0..t {
        let gs = sg.map(|c| sigmoid(logits[tt * stride + c]));
        for d in 0..HIDDEN {
            let mut acc = 0f32;
            for i in 0..TOPK {
                acc = w[tt * TOPK + i].mul_add(parts[(tt * TOPK + i) * HIDDEN + d], acc);
            }
            if let Some(g) = gs {
                acc = g.mul_add(parts[(MoePlan::SHARED_ROW + tt) * HIDDEN + d], acc);
            }
            y[tt * HIDDEN + d] = acc;
        }
    }
    y
}

const _: () = assert!(MAX_T * TOPK + MAX_T == MoePlan::CAP);

// ---- QSA ----

/// RMSNorm (`x · (1 / sqrt(mean x² + eps)) · w`) then NeoX rope on the first `QSA_ROT` dims.
fn norm_rope(x: &[f32], w: &[f32], cos: &[f32], sin: &[f32], pos: usize, eps: f32) -> Vec<f32> {
    let n = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum();
    let rs = 1.0 / (ss / n as f32 + eps).sqrt();
    let mut y: Vec<f32> = x.iter().zip(w).map(|(v, w)| v * rs * w).collect();
    let half = QSA_ROT / 2;
    for i in 0..half {
        let (c, s) = (cos[pos * half + i], sin[pos * half + i]);
        let (x0, x1) = (y[i], y[i + half]);
        y[i] = x0.mul_add(c, -(x1 * s));
        y[i + half] = x1.mul_add(c, x0 * s);
    }
    y
}

/// QSA prologue for a window: queries into `q` ([`crate::flash::qsa_q_words`]), K/V into the
/// caches (bf16-rounded), raw indexer keys into the ring, and every block the window completes
/// pooled.
#[allow(clippy::too_many_arguments)]
pub(crate) fn qsa_prep(
    proj: &[f32],
    stride: usize,
    pos0: usize,
    (qn, kn, iqn, ikn): (&[f32], &[f32], &[f32], &[f32]),
    (cos, sin): (&[f32], &[f32]),
    t: usize,
    eps: f32,
    q: &mut [f32],
    (kc, vc, ring, pooled): (&mut [f32], &mut [f32], &mut [f32], &mut [f32]),
) {
    use crate::flash::bf16_round;
    let qd = QSA_HEADS * QSA_D;
    for tt in 0..t {
        let pos = pos0 + tt;
        let p = &proj[tt * stride..];
        for h in 0..QSA_HEADS {
            let y = norm_rope(
                &p[h * 2 * QSA_D..h * 2 * QSA_D + QSA_D],
                qn,
                cos,
                sin,
                pos,
                eps,
            );
            q[tt * qd + h * QSA_D..tt * qd + (h + 1) * QSA_D].copy_from_slice(&y);
        }
        for g in 0..QSA_KV {
            let k = norm_rope(
                &p[QSA_K + g * QSA_D..QSA_K + (g + 1) * QSA_D],
                kn,
                cos,
                sin,
                pos,
                eps,
            );
            for j in 0..QSA_D {
                kc[(pos * QSA_KV + g) * QSA_D + j] = bf16_round(k[j]);
                vc[(pos * QSA_KV + g) * QSA_D + j] = bf16_round(p[QSA_V + g * QSA_D + j]);
            }
        }
        for h in 0..IDX_HEADS {
            let y = norm_rope(
                &p[QSA_IQ + h * IDX_D..QSA_IQ + (h + 1) * IDX_D],
                iqn,
                cos,
                sin,
                pos,
                eps,
            );
            let o = t * qd + (tt * IDX_HEADS + h) * IDX_D;
            q[o..o + IDX_D].copy_from_slice(&y);
        }
    }
    // Pool from the inputs as the kernel does: cells inside the window from `proj`, older ones
    // from the ring (which this window's writes never reach).
    for tt in 0..t {
        let pos = pos0 + tt;
        if pos % IDX_BLOCK != IDX_BLOCK - 1 {
            continue;
        }
        let b = pos / IDX_BLOCK;
        let raw = |c: usize| -> &[f32] {
            if c >= pos0 {
                &proj[(c - pos0) * stride + QSA_IK..(c - pos0) * stride + QSA_IK + IDX_D]
            } else {
                &ring[(c % QSA_RING) * IDX_D..(c % QSA_RING + 1) * IDX_D]
            }
        };
        let c0 = b * IDX_BLOCK;
        let mean: Vec<f32> = (0..IDX_D)
            .map(|j| (((raw(c0)[j] + raw(c0 + 1)[j]) + raw(c0 + 2)[j]) + raw(c0 + 3)[j]) * 0.25)
            .collect();
        let y = norm_rope(&mean, ikn, cos, sin, c0, eps);
        pooled[b * IDX_D..(b + 1) * IDX_D].copy_from_slice(&y);
    }
    for tt in 0..t {
        let pos = pos0 + tt;
        ring[(pos % QSA_RING) * IDX_D..(pos % QSA_RING + 1) * IDX_D]
            .copy_from_slice(&proj[tt * stride + QSA_IK..tt * stride + QSA_IK + IDX_D]);
    }
}

/// Block scores `[t][max_blocks]` for blocks `b < n_kv / 4` (others untouched), with the lane
/// order the contract pins.
pub(crate) fn qsa_scores(
    pooled: &[f32],
    iq: &[f32],
    pos0: usize,
    t: usize,
    max_blocks: usize,
    scores: &mut [f32],
) {
    for tt in 0..t {
        let n_bid = (pos0 + tt + 1) / IDX_BLOCK;
        for b in 0..n_bid.min(max_blocks) {
            let pk = &pooled[b * IDX_D..(b + 1) * IDX_D];
            let mut total = 0f32;
            for h in 0..IDX_HEADS {
                let qv = &iq[(tt * IDX_HEADS + h) * IDX_D..(tt * IDX_HEADS + h + 1) * IDX_D];
                let mut lanes = [0f32; 32];
                for (l, lane) in lanes.iter_mut().enumerate() {
                    let i = 4 * l;
                    let mut a = qv[i] * pk[i];
                    a = qv[i + 1].mul_add(pk[i + 1], a);
                    a = qv[i + 2].mul_add(pk[i + 2], a);
                    *lane = qv[i + 3].mul_add(pk[i + 3], a);
                }
                let s = butterfly(lanes);
                total += if s > 0.0 { s } else { 0.0 };
            }
            scores[tt * max_blocks + b] = total * INV_SQRT_D;
        }
    }
}

/// Order-preserving u32 key of an f32.
pub(crate) fn ordered(x: f32) -> u32 {
    let b = x.to_bits();
    if b & 0x8000_0000 != 0 {
        !b
    } else {
        b | 0x8000_0000
    }
}

/// Cells a token selects: every cell while it sees at most 512 complete blocks, else the 512
/// selected blocks' 2048 cells plus the incomplete tail's.
pub(crate) fn qsa_n_sel(n_kv: usize) -> usize {
    if n_kv / IDX_BLOCK <= crate::flash::QSA_BLOCKS {
        n_kv
    } else {
        IDX_BLOCK * crate::flash::QSA_BLOCKS + n_kv % IDX_BLOCK
    }
}

/// Selected cells per the contract on [`crate::flash::qsa_score_blocks`]: ids `[t][QSA_WIDTH]`.
pub(crate) fn qsa_select(scores: &[f32], pos0: usize, t: usize, max_blocks: usize) -> Vec<u32> {
    let mut ids = vec![0u32; t * QSA_WIDTH];
    for tt in 0..t {
        let n_kv = pos0 + tt + 1;
        let out = &mut ids[tt * QSA_WIDTH..(tt + 1) * QSA_WIDTH];
        let n_bid = n_kv / IDX_BLOCK;
        if n_bid <= crate::flash::QSA_BLOCKS {
            for (i, o) in out.iter_mut().take(n_kv).enumerate() {
                *o = i as u32;
            }
            continue;
        }
        let s = &scores[tt * max_blocks..tt * max_blocks + n_bid];
        let mut order: Vec<usize> = (0..n_bid).collect();
        order.sort_by(|&a, &b| ordered(s[b]).cmp(&ordered(s[a])).then(a.cmp(&b)));
        let mut keep: Vec<usize> = order[..crate::flash::QSA_BLOCKS].to_vec();
        keep.sort_unstable();
        let mut i = 0;
        for b in keep {
            for j in 0..IDX_BLOCK {
                out[i] = (b * IDX_BLOCK + j) as u32;
                i += 1;
            }
        }
        for c in n_bid * IDX_BLOCK..n_kv {
            out[i] = c as u32;
            i += 1;
        }
        debug_assert_eq!(i, qsa_n_sel(n_kv));
    }
    ids
}

/// Attention over the selected cells, gated: `out [t][QSA_OUT]`,
/// `o_h = softmax(q_h · k / 16) · v`, times `σ(gate_h)` from the projection.
#[allow(clippy::too_many_arguments)]
pub(crate) fn qsa_attend(
    q: &[f32],
    kc: &[f32],
    vc: &[f32],
    ids: &[u32],
    proj: &[f32],
    stride: usize,
    pos0: usize,
    t: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; t * QSA_OUT];
    for tt in 0..t {
        let n_sel = qsa_n_sel(pos0 + tt + 1);
        let cells = &ids[tt * QSA_WIDTH..tt * QSA_WIDTH + n_sel];
        for h in 0..QSA_HEADS {
            let g = h / (QSA_HEADS / QSA_KV);
            let qh = &q[(tt * QSA_HEADS + h) * QSA_D..(tt * QSA_HEADS + h + 1) * QSA_D];
            let s: Vec<f32> = cells
                .iter()
                .map(|&c| dot(qh, &kc[(c as usize * QSA_KV + g) * QSA_D..][..QSA_D]) * 0.0625)
                .collect();
            let m = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = s.iter().map(|v| (v - m).exp()).collect();
            let z: f32 = e.iter().sum();
            for j in 0..QSA_D {
                let mut o = 0f32;
                for (i, &c) in cells.iter().enumerate() {
                    o += e[i] * vc[(c as usize * QSA_KV + g) * QSA_D + j];
                }
                let gate = proj[tt * stride + h * 2 * QSA_D + QSA_D + j];
                out[tt * QSA_OUT + h * QSA_D + j] = o / z * sigmoid(gate);
            }
        }
    }
    out
}

/// Properties of the reference itself (the GPU kernels are checked against it in
/// `cuda::flash::tests`).
#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash::{f32_to_f16, q2_bytes, q2_repack};

    fn vals(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
            })
            .collect()
    }

    #[test]
    fn quantize_round_trips_within_half_a_step() {
        let (m, k) = (3, 256);
        let x = vals(m * k, 1, 5.0);
        let w = quantize_act(&x, m, k);
        let l = QAct { m, k };
        for r in 0..m {
            let row = QRow::decode(&w, l, r);
            for c in 0..k / 32 {
                assert_eq!(row.hx[c], row.q[c * 32..c * 32 + 32].iter().sum::<i32>());
                for i in 0..32 {
                    let e = c * 32 + i;
                    let back = row.q[e] as f32 * row.d[c];
                    assert!((back - x[r * k + e]).abs() <= row.d[c] * 0.5 + 1e-6);
                }
            }
        }
    }

    #[test]
    fn q2_linear_is_the_dequantized_dot() {
        let (m, n, k) = (2, 5, 192);
        let mut raw = Vec::with_capacity(q2_bytes(n, k));
        let codes = vals(n * k, 2, 1.0);
        for b in 0..n * k / 64 {
            raw.extend(f32_to_f16(0.01 + 0.002 * b as f32).to_le_bytes());
            for j in 0..16 {
                let mut byte = 0u8;
                for f in 0..4 {
                    let c = ((codes[b * 64 + 4 * j + f] + 1.0) * 1.999) as u8;
                    byte |= c << (2 * f);
                }
                raw.push(byte);
            }
        }
        let wb = q2_repack(&raw, n, k);
        let x = vals(m * k, 3, 1.0);
        let xq = quantize_act(&x, m, k);
        let y = q2_linear(&xq, &wb, m, k, n);
        let l = QAct { m, k };
        for r in 0..m {
            let row = QRow::decode(&xq, l, r);
            for o in 0..n {
                let mut want = 0f64;
                for e in 0..k {
                    let blk = &raw[(o * k + e) / 64 * 18..];
                    let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])) as f64;
                    let i = e % 64;
                    let code = ((blk[2 + i / 4] >> (2 * (i % 4))) & 3) as f64;
                    want += (code - 1.0) * d * row.q[e] as f64 * row.d[e / 32] as f64;
                }
                assert!(
                    (y[r * n + o] as f64 - want).abs() < 1e-5,
                    "{} vs {want}",
                    y[r * n + o]
                );
            }
        }
    }

    #[test]
    fn router_picks_distinct_top_logits_and_renormalizes() {
        let l = vals(2 * EXPERTS, 4, 3.0);
        let (ids, w) = router_topk(&l, EXPERTS, 2, EXPERTS);
        for t in 0..2 {
            let row = &ids[t * TOPK..(t + 1) * TOPK];
            let lt = &l[t * EXPERTS..(t + 1) * EXPERTS];
            for k in 1..TOPK {
                assert!(lt[row[k - 1] as usize] >= lt[row[k] as usize]);
            }
            let min_sel = lt[row[TOPK - 1] as usize];
            assert_eq!(lt.iter().filter(|&&v| v > min_sel).count(), TOPK - 1);
            let s: f32 = w[t * TOPK..(t + 1) * TOPK].iter().sum();
            assert!((s - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn plan_groups_dedupe_in_routing_order() {
        let t = 2;
        let mut ids: Vec<u32> = (0..TOPK as u32).collect();
        ids.extend((5..5 + TOPK as u32).rev());
        let mut table = vec![0u64; EXPERTS];
        for (e, p) in table.iter_mut().enumerate() {
            if e != 7 {
                *p = 0x1000 + e as u64;
            }
        }
        let plan = moe_plan(&ids, &table, 0xabc, t);
        // 15 distinct experts, expert 7 missing, plus the shared group.
        assert_eq!(&plan[..3], &[15, 20 - 2 + 2, 1]);
        assert_eq!(plan[MoePlan::MISSING], 7);
        // Expert 5 (group 5) has entries 5 (token 0) and 19 (token 1, rank 9).
        let g5 = 5;
        assert_eq!(plan[MoePlan::GROUP_PTR + 2 * g5], 0x1005);
        let s = plan[MoePlan::GROUP_START + g5] as usize;
        assert_eq!(plan[MoePlan::GROUP_START + g5 + 1] as usize - s, 2);
        assert_eq!(
            &plan[MoePlan::ENT_DST + s..MoePlan::ENT_DST + s + 2],
            &[5, 19]
        );
        assert_eq!(
            &plan[MoePlan::ENT_TOK + s..MoePlan::ENT_TOK + s + 2],
            &[0, 1]
        );
        let last = plan[0] as usize - 1;
        assert_eq!(plan[MoePlan::GROUP_PTR + 2 * last], 0xabc);
        let s = plan[MoePlan::GROUP_START + last] as usize;
        assert_eq!(
            plan[MoePlan::ENT_DST + s + 1] as usize,
            MoePlan::SHARED_ROW + 1
        );
    }

    #[test]
    fn selection_is_identity_then_width_cells_with_the_tail() {
        let max_blocks = 1024;
        let scores = vals(max_blocks, 5, 1.0)
            .iter()
            .map(|v| v.abs())
            .collect::<Vec<_>>();
        let ids = qsa_select(&scores, 99, 1, max_blocks);
        assert!(ids[..100].iter().enumerate().all(|(i, &c)| c == i as u32));
        let pos0 = 3001; // n_kv 3002: tail of 2 cells
        let ids = qsa_select(&scores, pos0, 1, max_blocks);
        let ids = &ids[..qsa_n_sel(3002)];
        assert_eq!(ids.len(), 2050);
        assert!(ids.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(&ids[2048..], &[3000, 3001]);
        // n_kv = 2052: 513 complete blocks, no tail: 512 whole blocks, no partial one.
        let ids2 = qsa_select(&scores, 2051, 1, max_blocks);
        assert_eq!(qsa_n_sel(2052), 2048);
        assert!(ids2[..2048]
            .chunks(4)
            .all(|c| c[0] % 4 == 0 && c[3] == c[0] + 3));
        // The best block is in, the worst is out.
        let n_bid = 3002 / 4;
        let best = (0..n_bid)
            .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
            .unwrap();
        let worst = (0..n_bid)
            .min_by(|&a, &b| scores[a].total_cmp(&scores[b]))
            .unwrap();
        assert!(ids.contains(&(best as u32 * 4)) && !ids.contains(&(worst as u32 * 4)));
    }

    #[test]
    fn gdn_first_step_from_zero_state() {
        // From S = 0: S = k ⊗ (β v), so o = (q · k) β v / sqrt(128).
        let t = 1;
        let mut proj = vals(GDN_PROJ, 6, 1.0);
        let h = vals(GDN_CONV, 7, 1.0);
        let (dt, a, norm) = (vec![0.0; GDN_HV], vec![-1.0; GDN_HV], vec![1.0; GDN_D]);
        proj[GDN_Z..GDN_Z + GDN_V]
            .iter_mut()
            .for_each(|z| *z = 40.0); // sigmoid(z) = 1
        let mut state = vec![0f32; crate::flash::GDN_STATE];
        let y = gdn_step(
            &mut state,
            &h,
            &proj,
            GDN_PROJ,
            (&dt, &a, &norm),
            t,
            1,
            true,
            0.0,
        );
        let hv = 17;
        let hk = hv % GDN_HK;
        let q = &h[hk * GDN_D..(hk + 1) * GDN_D];
        let k = &h[GDN_HK * GDN_D + hk * GDN_D..][..GDN_D];
        let v = &h[2 * GDN_HK * GDN_D + hv * GDN_D..][..GDN_D];
        let qk: f32 = q.iter().zip(k).map(|(a, b)| a * b).sum();
        let beta = sigmoid(proj[GDN_B + hv]);
        let o: Vec<f32> = v.iter().map(|v| qk * beta * v * INV_SQRT_D).collect();
        let rms = (o.iter().map(|x| x * x).sum::<f32>() / GDN_D as f32).sqrt();
        for j in 0..GDN_D {
            let want = o[j] / rms;
            assert!((y[hv * GDN_D + j] - want).abs() < 1e-4 * (1.0 + want.abs()));
        }
    }

    #[test]
    fn pexp_is_within_two_ulp_of_exp() {
        let mut x = -87.0f32;
        while x < 88.0 {
            let (got, want) = (crate::flash::pexp(x), (x as f64).exp());
            let ulp = (want as f32).to_bits().abs_diff(got.to_bits());
            assert!(ulp <= 2, "pexp({x}) = {got}, exp = {want}");
            x += 0.0137;
        }
    }

    #[test]
    fn expert_row_dot_is_the_dequantized_dot() {
        use crate::flash::{ExpertBlob, Q2_BLOCK_BYTES};
        let mut s = 7u64;
        let mut byte = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s as u8
        };
        let mat = |n: usize, k: usize, byte: &mut dyn FnMut() -> u8| -> Vec<u8> {
            let mut m = vec![];
            for b in 0..n * k / 64 {
                m.extend(f32_to_f16(0.01 + 0.001 * (b % 7) as f32).to_le_bytes());
                m.extend((0..16).map(|_| byte()));
            }
            m
        };
        let (gate, up, down) = (
            mat(FF, HIDDEN, &mut byte),
            mat(FF, HIDDEN, &mut byte),
            mat(HIDDEN, FF, &mut byte),
        );
        let blob = ExpertBlob::from_gguf(&gate, &up, &down);
        let x = vals(HIDDEN, 9, 1.0);
        let xq = quantize_act(&x, 1, HIDDEN);
        let row = QRow::decode(&xq, QAct { m: 1, k: HIDDEN }, 0);
        for r in [0, 17, 639] {
            let got = expert_row_dot(
                &blob,
                HIDDEN / 128,
                |g| ExpertBlob::gu_group(2 * r + 1, g),
                &row,
            );
            let mut want = 0f64;
            for e in 0..HIDDEN {
                let blk = &up[(r * HIDDEN + e) / 64 * Q2_BLOCK_BYTES..];
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])) as f64;
                let i = e % 64;
                let code = ((blk[2 + i / 4] >> (2 * (i % 4))) & 3) as f64;
                want += (code - 1.0) * d * row.q[e] as f64 * row.d[e / 32] as f64;
            }
            assert!(
                (got as f64 - want).abs() < 1e-4 * (1.0 + want.abs()),
                "{got} vs {want}"
            );
        }
    }

    #[test]
    fn hc_write_with_zero_injection_is_a_plain_add() {
        let mut r = vals(HC * HIDDEN, 8, 1.0);
        let r0 = r.clone();
        let y = vals(HIDDEN, 9, 1.0);
        hc_write(&mut r, &y, &[0.0; HC], 1);
        for c in 0..HC {
            for d in 0..HIDDEN {
                assert_eq!(r[c * HIDDEN + d], r0[c * HIDDEN + d] + y[d]);
            }
        }
    }
}
