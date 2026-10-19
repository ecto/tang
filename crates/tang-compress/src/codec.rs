//! The bit-accounted QTT codec.
//!
//! # Encoder
//!
//! One *pass* at absolute quantization step `Δ` runs TT-SVD left to right
//! with **error feedback**: core `k` is computed from the remainder that is
//! least-squares optimal *given the already-quantized prefix*, quantized, and
//! only then is the remainder refit against it.
//!
//! Let `P_{k−1}` be the quantized prefix (rows = prefix indices, cols =
//! `r_{k−1}`), `H = P_{k−1}ᵀP_{k−1}` its Gram, and `C` the LS remainder
//! (`X = (P_{k−1}⊗I)·C + E`, `E ⊥ range`). In the metric `H̃ = H ⊗ I_{n_k}`:
//!
//! 1. eigendecompose the whitened Gram of `C` → singular values `σ_j` and an
//!    `H̃`-orthonormal core `G` (the canonical left-orthogonal gauge, now
//!    w.r.t. the true prefix);
//! 2. quantize column `j` of `G` with step `Δ / σ̂_j`, `σ̂_j = 2^{e_j/4}`,
//!    `e_j = round(4 log₂ σ_j)` — the equal-slope rule: perturbing column `j`
//!    costs `σ_j²·step²/12` per entry, so every entry ends up with the same
//!    distortion `Δ²/12` as an entry of the last core. Columns that round to
//!    all zeros are dropped, so **ranks are chosen by rate–distortion, not by
//!    a separate tolerance**;
//! 3. refit `C ← (QᵀH̃Q)⁻¹ QᵀH̃ C`, `H ← QᵀH̃Q`, and add the exact discarded
//!    energy `‖C‖²_H̃ − ‖proj‖²` to the running SSE.
//!
//! The last core (the LS remainder, carrying the norm) is quantized with step
//! `Δ`; its error `eᵀH̃e` is added too. The SSE is therefore known **exactly at
//! encode time** without reconstructing the tensor (the error pieces are
//! mutually orthogonal). `Δ` is then found by a bracketed log–log secant search
//! on that SSE.
//!
//! # Error modes
//!
//! - [`ErrorBound::Rmse`] / [`ErrorBound::Psnr`] (range-normalised): the
//!   searched SSE is followed by an actual decode; if the true RMSE on the
//!   original (unpadded) cells exceeds the target the search is redone.
//! - [`ErrorBound::MaxAbs`] (SZ-style pointwise bound `b`): the TT is coded at
//!   an RMSE target `f·b` for a few `f`, the encoder decodes its own stream,
//!   and every cell with `|x − x̂| > b` gets a correction `k = round((x−x̂)/2b)`
//!   (then `|x − (x̂+2bk)| ≤ b`), stored as entropy-coded position gaps + values
//!   (a raw-value escape covers float-cast edge cases). The `f` giving the
//!   smallest total stream wins.
//!
//! # Bit account
//!
//! The stream is self-describing: magic, dtype, digit order, padding policy,
//! shape, `Δ`, all ranks, all per-column step exponents, every frequency
//! table, all payloads, the residual stream. `bytes.len()` is the size.

use crate::bitio::{ByteReader, ByteWriter, DecodeError};
use crate::linalg::{
    cholesky, gram_rows, kron_eye_apply, matmul, matmul_tn, solve_lower, solve_lower_t,
    sym_eig_desc,
};
use crate::rans::{decode_blocks, encode_blocks};
use crate::tensorize::{DigitOrder, Layout, PadPolicy};
use crate::tt::{contract, gd_rank};

const MAGIC: &[u8; 4] = b"QTT1";

/// Error target.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ErrorBound {
    /// Root-mean-square error over the original cells, absolute units.
    Rmse(f64),
    /// Range-normalised PSNR in dB: `20 log₁₀(range / RMSE)`.
    Psnr(f64),
    /// Pointwise `|x − x̂| ≤ b`, absolute units.
    MaxAbs(f64),
}

/// Encoder configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    /// Error target.
    pub bound: ErrorBound,
    /// Digit order of the tensorization.
    pub order: DigitOrder,
    /// Padding policy for non-power-of-two axes.
    pub pad: PadPolicy,
    /// Cap ranks at the Gavish–Donoho threshold of each unfolding.
    pub gavish_donoho: bool,
    /// Hard rank cap.
    pub max_rank: usize,
}

impl Config {
    /// Defaults (Morton, mixed radix, no GD, rank cap 1024) for a bound.
    pub fn new(bound: ErrorBound) -> Self {
        Self {
            bound,
            order: DigitOrder::Morton,
            pad: PadPolicy::MixedRadix,
            gavish_donoho: false,
            max_rank: 1024,
        }
    }
}

/// Sample type of the original array (decoded output has the same type).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    /// 32-bit float.
    F32,
    /// 64-bit float.
    F64,
}

/// What the encoder did, for benchmarking.
#[derive(Clone, Debug, PartialEq)]
pub struct EncodeStats {
    /// Total stream size in bytes (everything).
    pub bytes: usize,
    /// `8·bytes / cells`.
    pub bits_per_value: f64,
    /// Bytes before the core payload (header, ranks, step exponents).
    pub header_bytes: usize,
    /// Core payload bytes (tables + rANS + raw bits).
    pub core_bytes: usize,
    /// Residual-correction stream bytes (max-abs mode).
    pub residual_bytes: usize,
    /// Number of corrected cells.
    pub n_residual: usize,
    /// TT bond ranks.
    pub ranks: Vec<usize>,
    /// Tensor mode sizes.
    pub modes: Vec<usize>,
    /// Final absolute step `Δ`.
    pub delta: f64,
    /// Encoder passes run.
    pub passes: usize,
    /// Achieved RMSE over the original cells.
    pub rmse: f64,
    /// Achieved max-abs error over the original cells.
    pub max_abs: f64,
    /// Value range of the input.
    pub range: f64,
}

/// A decoded array.
#[derive(Clone, Debug, PartialEq)]
pub enum Decoded {
    /// f32 samples.
    F32 {
        /// Shape.
        dims: Vec<usize>,
        /// Row-major samples.
        data: Vec<f32>,
    },
    /// f64 samples.
    F64 {
        /// Shape.
        dims: Vec<usize>,
        /// Row-major samples.
        data: Vec<f64>,
    },
}

/// One quantized pass.
#[derive(Clone, Debug)]
struct QPass {
    ranks: Vec<usize>,
    exps: Vec<Vec<i32>>,
    q: Vec<Vec<i64>>,
    sse: f64,
}

/// `Δ·2^{−e/4}`, exact in binary floating point.
fn col_step(delta: f64, e: i32) -> f64 {
    const FRAC: [f64; 4] = [
        1.0,
        0.840_896_415_253_714_5, // 2^(-1/4)
        core::f64::consts::FRAC_1_SQRT_2,
        0.594_603_557_501_360_5, // 2^(-3/4)
    ];
    let a = e.div_euclid(4);
    let b = e.rem_euclid(4) as usize;
    delta * FRAC[b] * 2f64.powi(-a)
}

fn kron_eye(h: &[f64], r: usize, n: usize) -> Vec<f64> {
    let m = r * n;
    let mut out = vec![0.0; m * m];
    for a in 0..r {
        for b in 0..r {
            let v = h[a * r + b];
            if v != 0.0 {
                for i in 0..n {
                    out[(a * n + i) * m + b * n + i] = v;
                }
            }
        }
    }
    out
}

fn quantized_pass(x: &[f64], modes: &[usize], delta: f64, gd: bool, max_rank: usize) -> QPass {
    let d = modes.len();
    let mut ranks = vec![1usize];
    let mut exps = Vec::new();
    let mut qs = Vec::new();
    let mut sse = 0.0;
    let mut c = x.to_vec();
    let mut h = vec![1.0];
    let mut r_prev = 1usize;
    let mut cols: usize = x.len();
    for &nk in modes.iter().take(d.saturating_sub(1)) {
        let m = r_prev * nk;
        cols /= nk;
        if r_prev == 0 {
            ranks.push(0);
            exps.push(Vec::new());
            qs.push(Vec::new());
            continue;
        }
        let ht = kron_eye(&h, r_prev, nk);
        // Whitened spectrum and H̃-orthonormal candidate core G (m × kc).
        let (sig, g, kc, c_energy) = if m <= cols {
            let cc = gram_rows(&c, m, cols);
            let l = cholesky(&ht, m).expect("prefix Gram is SPD");
            let t = matmul(&cc, &l, m, m, m);
            let gw = matmul_tn(&l, &t, m, m, m);
            let (lam, vw) = sym_eig_desc(&gw, m);
            let sig: Vec<f64> = lam.iter().map(|&v| v.max(0.0).sqrt()).collect();
            let energy: f64 = lam.iter().map(|v| v.max(0.0)).sum();
            let kc = candidate_count(&sig, delta, gd, m, cols, max_rank);
            let mut g = vec![0.0; m * kc];
            for i in 0..m {
                g[i * kc..i * kc + kc].copy_from_slice(&vw[i * m..i * m + kc]);
            }
            solve_lower_t(&l, &mut g, m, kc);
            (sig, g, kc, energy)
        } else {
            let htc = kron_eye_apply(&h, r_prev, nk, &c, cols);
            let gw = matmul_tn(&c, &htc, m, cols, cols);
            let (lam, v) = sym_eig_desc(&gw, cols);
            let sig: Vec<f64> = lam.iter().map(|&v| v.max(0.0).sqrt()).collect();
            let energy: f64 = lam.iter().map(|v| v.max(0.0)).sum();
            let smax = sig.first().copied().unwrap_or(0.0);
            let kc = candidate_count(&sig, delta, gd, m, cols, max_rank).min(
                sig.iter()
                    .take_while(|&&s| s > 1e-13 * smax && s > 0.0)
                    .count(),
            );
            let mut vk = vec![0.0; cols * kc];
            for i in 0..cols {
                vk[i * kc..i * kc + kc].copy_from_slice(&v[i * cols..i * cols + kc]);
            }
            let mut g = matmul(&c, &vk, m, cols, kc);
            for j in 0..kc {
                for i in 0..m {
                    g[i * kc + j] /= sig[j];
                }
            }
            (sig, g, kc, energy)
        };
        // Quantize columns with equal-slope steps; drop all-zero columns.
        let mut keep: Vec<(i32, Vec<i64>)> = Vec::new();
        for j in 0..kc {
            let e = (4.0 * sig[j].log2()).round() as i32;
            let step = col_step(delta, e);
            let col: Vec<i64> = (0..m)
                .map(|i| (g[i * kc + j] / step).round() as i64)
                .collect();
            if col.iter().any(|&v| v != 0) {
                keep.push((e, col));
            }
        }
        // Refit; drop trailing columns if the quantized Gram is singular.
        loop {
            let r = keep.len();
            if r == 0 {
                sse += c_energy;
                break;
            }
            let mut qm = vec![0.0; m * r];
            for (j, (e, col)) in keep.iter().enumerate() {
                let step = col_step(delta, *e);
                for i in 0..m {
                    qm[i * r + j] = col[i] as f64 * step;
                }
            }
            let htq = kron_eye_apply(&h, r_prev, nk, &qm, r);
            let hk = matmul_tn(&qm, &htq, m, r, r);
            // Drop trailing (smallest-σ) columns while the quantized Gram is
            // singular or ill-conditioned: the metric bookkeeping (and the
            // refit) degrade like cond(Hk), and a near-dependent column buys
            // no rate anyway.
            let lk = match cholesky(&hk, r) {
                Some(lk) if chol_cond_ok(&lk, r) => lk,
                _ => {
                    keep.pop();
                    continue;
                }
            };
            let b = matmul_tn(&htq, &c, m, r, cols);
            // W = Hk⁻¹ B through an explicit inverse (Hk ≈ I, well conditioned)
            // so the r×cols product is one GEMM instead of two triangular
            // sweeps; projected energy = ⟨B, W⟩.
            let mut hinv = vec![0.0; r * r];
            for i in 0..r {
                hinv[i * r + i] = 1.0;
            }
            solve_lower(&lk, &mut hinv, r, r);
            solve_lower_t(&lk, &mut hinv, r, r);
            let w = matmul(&hinv, &b, r, r, cols);
            let proj: f64 = b.iter().zip(&w).map(|(x, y)| x * y).sum();
            sse += (c_energy - proj).max(0.0);
            c = w;
            h = hk;
            break;
        }
        let r = keep.len();
        let mut qcore = vec![0i64; m * r];
        let mut ecore = Vec::with_capacity(r);
        for (j, (e, col)) in keep.iter().enumerate() {
            ecore.push(*e);
            for i in 0..m {
                qcore[i * r + j] = col[i];
            }
        }
        ranks.push(r);
        exps.push(ecore);
        qs.push(qcore);
        r_prev = r;
        if r == 0 {
            c.clear();
        }
    }
    // Last core: (r_{d−1}·n_d) × 1, step Δ, error in the H̃ metric.
    let nd = *modes.last().unwrap();
    if r_prev == 0 {
        qs.push(Vec::new());
    } else {
        let m = r_prev * nd;
        let ht = kron_eye(&h, r_prev, nd);
        let q: Vec<i64> = c.iter().map(|&v| (v / delta).round() as i64).collect();
        let e: Vec<f64> = c
            .iter()
            .zip(&q)
            .map(|(&v, &qi)| v - qi as f64 * delta)
            .collect();
        let he = matmul(&ht, &e, m, m, 1);
        sse += e.iter().zip(&he).map(|(a, b)| a * b).sum::<f64>().max(0.0);
        qs.push(q);
    }
    ranks.push(1);
    QPass {
        ranks,
        exps,
        q: qs,
        sse,
    }
}

/// `min/max` of the Cholesky diagonal ≥ 1e-3, i.e. cond(Hk) ≲ 1e6.
fn chol_cond_ok(l: &[f64], r: usize) -> bool {
    let d: Vec<f64> = (0..r).map(|i| l[i * r + i]).collect();
    let (mn, mx) = d
        .iter()
        .fold((f64::INFINITY, 0.0f64), |(a, b), &v| (a.min(v), b.max(v)));
    mn >= 1e-3 * mx
}

fn candidate_count(
    sig: &[f64],
    delta: f64,
    gd: bool,
    m: usize,
    n: usize,
    max_rank: usize,
) -> usize {
    // A column with σ < Δ/2 quantizes to all zeros (|G_ij| ≤ 1 in the gauge).
    let mut k = sig.iter().take_while(|&&s| s >= 0.35 * delta).count();
    if gd {
        k = k.min(gd_rank(sig, m, n));
    }
    k.min(max_rank)
}

/// Rebuild float cores from quantized integers — shared by encoder and decoder
/// so both contract bit-identical numbers.
fn rebuild_cores(
    modes: &[usize],
    ranks: &[usize],
    exps: &[Vec<i32>],
    q: &[Vec<i64>],
    delta: f64,
) -> Vec<Vec<f64>> {
    let d = modes.len();
    let mut cores = Vec::with_capacity(d);
    for k in 0..d {
        let r = ranks[k + 1];
        let qk = &q[k];
        if k + 1 < d {
            let steps: Vec<f64> = exps[k].iter().map(|&e| col_step(delta, e)).collect();
            cores.push(
                qk.iter()
                    .enumerate()
                    .map(|(i, &v)| v as f64 * steps[i % r])
                    .collect(),
            );
        } else {
            cores.push(qk.iter().map(|&v| v as f64 * delta).collect());
        }
    }
    cores
}

fn reconstruct(layout: &Layout, pass: &QPass, delta: f64) -> Vec<f64> {
    if pass.ranks.contains(&0) {
        return vec![0.0; layout.len()];
    }
    let cores = rebuild_cores(&layout.modes, &pass.ranks, &pass.exps, &pass.q, delta);
    layout.from_tensor(&contract(&layout.modes, &pass.ranks, &cores))
}

/// Round to the output dtype (what the decoder hands back).
fn cast(v: f64, dt: Dtype) -> f64 {
    match dt {
        Dtype::F32 => v as f32 as f64,
        Dtype::F64 => v,
    }
}

/// Find the largest `Δ` with pass SSE ≤ `target_sse`.
fn search_delta(
    tensor: &[f64],
    modes: &[usize],
    cfg: &Config,
    target_sse: f64,
    guess: f64,
    passes: &mut usize,
) -> (f64, QPass) {
    let mut run = |delta: f64| {
        *passes += 1;
        quantized_pass(tensor, modes, delta, cfg.gavish_donoho, cfg.max_rank)
    };
    let mut lo: Option<(f64, f64, QPass)> = None; // feasible (delta, sse, pass)
    let mut hi: Option<(f64, f64)> = None; // infeasible
                                           // Lowest-SSE pass seen, returned if the target is unreachable (rank cap).
    let mut best_infeasible: Option<(f64, f64, QPass)> = None;
    let mut delta = guess;
    for _ in 0..14 {
        let p = run(delta);
        let s = p.sse;
        if s <= target_sse {
            if lo.as_ref().is_none_or(|l| delta > l.0) {
                lo = Some((delta, s, p));
            }
        } else {
            if hi.is_none_or(|h| delta < h.0) {
                hi = Some((delta, s));
            }
            if best_infeasible.as_ref().is_none_or(|b| s < b.1) {
                best_infeasible = Some((delta, s, p));
            } else if lo.is_none() {
                // Shrinking Δ stopped helping: the rank cap binds.
                break;
            }
        }
        delta = match (&lo, &hi) {
            (Some((dl, sl, _)), Some((dh, sh))) => {
                if dh / dl < 1.03 {
                    break;
                }
                // log–log secant inside the bracket, kept away from the ends.
                let (a, b) = (dl.ln(), dh.ln());
                let (fa, fb) = ((sl.max(1e-300)).ln(), sh.ln());
                let t = if fb > fa {
                    ((target_sse.ln() - fa) / (fb - fa)).clamp(0.15, 0.85)
                } else {
                    0.5
                };
                (a + t * (b - a)).exp()
            }
            (Some((dl, sl, _)), None) => {
                if *sl <= 0.0 {
                    dl * 4.0
                } else {
                    dl * (target_sse / sl).sqrt().clamp(1.05, 8.0)
                }
            }
            (None, Some((dh, sh))) => dh * (target_sse / sh).sqrt().clamp(0.1, 0.95),
            (None, None) => unreachable!(),
        };
    }
    match lo {
        Some((d, _, p)) => (d, p),
        None => {
            let (d, _, p) = best_infeasible.expect("at least one pass ran");
            (d, p)
        }
    }
}

/// Sparse max-abs corrections: positions, integer multiples of `2·bound`
/// (0 = raw escape), and the escaped raw values.
struct Residual {
    bound: f64,
    pos: Vec<usize>,
    vals: Vec<i64>,
    raw: Vec<f64>,
}

struct Built {
    layout: Layout,
    delta: f64,
    pass: QPass,
}

/// Serialize; returns (bytes, header bytes, core bytes, residual bytes).
fn write_stream(
    dt: Dtype,
    b: &Built,
    residual: Option<&Residual>,
) -> (Vec<u8>, usize, usize, usize) {
    let mut w = ByteWriter::new();
    w.bytes(MAGIC);
    w.u8(match dt {
        Dtype::F32 => 0,
        Dtype::F64 => 1,
    });
    w.u8(match b.layout.order {
        DigitOrder::RowMajor => 0,
        DigitOrder::Morton => 1,
    });
    w.u8(match b.layout.pad {
        PadPolicy::MixedRadix => 0,
        PadPolicy::Pow2 => 1,
    });
    w.varint(b.layout.dims.len() as u64);
    for &n in &b.layout.dims {
        w.varint(n as u64);
    }
    w.f64(b.delta);
    let d = b.layout.modes.len();
    for k in 1..d {
        w.varint(b.pass.ranks[k] as u64);
    }
    for k in 0..d.saturating_sub(1) {
        let mut prev = 0i64;
        for &e in &b.pass.exps[k] {
            w.svarint(e as i64 - prev);
            prev = e as i64;
        }
    }
    let header = w.buf.len();
    // A zero rank anywhere means an all-zero tensor: no core payload (the
    // decoder expects none).
    let blocks: Vec<&[i64]> = if b.pass.ranks.contains(&0) {
        Vec::new()
    } else {
        b.pass.q.iter().map(|v| v.as_slice()).collect()
    };
    encode_blocks(&mut w, &blocks);
    let core = w.buf.len() - header;
    match residual {
        None => w.u8(0),
        Some(Residual {
            bound,
            pos,
            vals,
            raw,
        }) => {
            w.u8(1);
            w.f64(*bound);
            w.varint(pos.len() as u64);
            let mut gaps = Vec::with_capacity(pos.len());
            let mut prev = 0usize;
            for (i, &p) in pos.iter().enumerate() {
                gaps.push(if i == 0 { p } else { p - prev - 1 } as i64);
                prev = p;
            }
            encode_blocks(&mut w, &[&gaps, vals.as_slice()]);
            for &v in raw {
                match dt {
                    Dtype::F32 => w.bytes(&(v as f32).to_le_bytes()),
                    Dtype::F64 => w.bytes(&v.to_le_bytes()),
                }
            }
        }
    }
    let residual_bytes = w.buf.len() - header - core;
    (w.buf, header, core, residual_bytes)
}

fn errors(orig: &[f64], rec: &[f64]) -> (f64, f64) {
    let mut s = 0.0;
    let mut mx: f64 = 0.0;
    for (a, b) in orig.iter().zip(rec) {
        let e = a - b;
        s += e * e;
        mx = mx.max(e.abs());
    }
    ((s / orig.len() as f64).sqrt(), mx)
}

/// Encode an f32 array of shape `dims` (row-major).
pub fn encode_f32(data: &[f32], dims: &[usize], cfg: &Config) -> (Vec<u8>, EncodeStats) {
    let x: Vec<f64> = data.iter().map(|&v| v as f64).collect();
    encode_impl(&x, dims, cfg, Dtype::F32)
}

/// Encode an f64 array of shape `dims` (row-major).
pub fn encode_f64(data: &[f64], dims: &[usize], cfg: &Config) -> (Vec<u8>, EncodeStats) {
    encode_impl(data, dims, cfg, Dtype::F64)
}

fn encode_impl(x: &[f64], dims: &[usize], cfg: &Config, dt: Dtype) -> (Vec<u8>, EncodeStats) {
    let n: usize = dims.iter().product();
    assert_eq!(x.len(), n, "encode: data length != product of dims");
    assert!(x.iter().all(|v| v.is_finite()), "encode: non-finite input");
    let layout = Layout::new(dims, cfg.order, cfg.pad);
    let tensor = layout.to_tensor(&layout.pad_grid(x));
    let (mn, mx) = x
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
            (a.min(v), b.max(v))
        });
    let range = (mx - mn).max(f64::MIN_POSITIVE);
    let ntens = layout.tensor_len() as f64;
    let mut passes = 0usize;

    let l2_build = |rmse: f64, guess: f64, passes: &mut usize| -> Built {
        let mut target = rmse * rmse * ntens;
        let mut g = guess;
        let mut out = None;
        for _ in 0..4 {
            let (delta, pass) = search_delta(&tensor, &layout.modes, cfg, target, g, passes);
            let rec: Vec<f64> = reconstruct(&layout, &pass, delta)
                .iter()
                .map(|&v| cast(v, dt))
                .collect();
            let (true_rmse, _) = errors(x, &rec);
            // Stop if met, or if the search could not even meet its own
            // (predicted) target — tightening further cannot help.
            let ok = true_rmse <= rmse || pass.sse > target;
            out = Some(Built {
                layout: layout.clone(),
                delta,
                pass,
            });
            if ok {
                break;
            }
            target *= (rmse / true_rmse).powi(2) * 0.98;
            g = delta * 0.9;
        }
        out.unwrap()
    };

    let finish = |b: Built, residual: Option<Residual>, passes: usize| {
        let (bytes, header, core, res) = write_stream(dt, &b, residual.as_ref());
        let rec = match decode(&bytes).expect("self-decode") {
            Decoded::F32 { data, .. } => data.iter().map(|&v| v as f64).collect::<Vec<_>>(),
            Decoded::F64 { data, .. } => data,
        };
        let (rmse, max_abs) = errors(x, &rec);
        let stats = EncodeStats {
            bytes: bytes.len(),
            bits_per_value: 8.0 * bytes.len() as f64 / n as f64,
            header_bytes: header,
            core_bytes: core,
            residual_bytes: res,
            n_residual: residual.as_ref().map_or(0, |r| r.pos.len()),
            ranks: b.pass.ranks.clone(),
            modes: layout.modes.clone(),
            delta: b.delta,
            passes,
            rmse,
            max_abs,
            range,
        };
        (bytes, stats)
    };

    match cfg.bound {
        ErrorBound::Rmse(r) | ErrorBound::Psnr(r) => {
            let rmse = match cfg.bound {
                ErrorBound::Psnr(p) => range / 10f64.powf(p / 20.0),
                _ => r,
            };
            let b = l2_build(rmse, rmse * 12f64.sqrt(), &mut passes);
            finish(b, None, passes)
        }
        ErrorBound::MaxAbs(bound) => {
            let mut best: Option<(usize, Built, Residual)> = None;
            let mut guess = bound;
            let mut prev_f = 0.0;
            // f = 0 is the residual-only candidate (TT quantized to zero): it
            // wins when the field has no low-rank structure (noise).
            for f in [0.0, 0.25, 0.4, 0.6] {
                let b = if f == 0.0 {
                    passes += 1;
                    let delta = 1e300;
                    let pass = quantized_pass(&tensor, &layout.modes, delta, false, 0);
                    Built {
                        layout: layout.clone(),
                        delta,
                        pass,
                    }
                } else {
                    if prev_f > 0.0 {
                        guess *= f / prev_f;
                    }
                    prev_f = f;
                    let b = l2_build(f * bound, guess, &mut passes);
                    guess = b.delta;
                    b
                };
                let rec = reconstruct(&layout, &b.pass, b.delta);
                let mut pos = Vec::new();
                let mut vals = Vec::new();
                let mut raw = Vec::new();
                for (i, (&xi, &ri)) in x.iter().zip(&rec).enumerate() {
                    if (xi - cast(ri, dt)).abs() > bound {
                        let k = ((xi - ri) / (2.0 * bound)).round() as i64;
                        let cand = cast(ri + 2.0 * bound * k as f64, dt);
                        pos.push(i);
                        if k != 0 && (xi - cand).abs() <= bound {
                            vals.push(k);
                        } else {
                            vals.push(0);
                            raw.push(xi);
                        }
                    }
                }
                let res = Residual {
                    bound,
                    pos,
                    vals,
                    raw,
                };
                let (bytes, _, _, _) = write_stream(dt, &b, Some(&res));
                if best.as_ref().is_none_or(|(s, _, _)| bytes.len() < *s) {
                    best = Some((bytes.len(), b, res));
                }
            }
            let (_, b, res) = best.unwrap();
            finish(b, Some(res), passes)
        }
    }
}

/// Decode a stream produced by [`encode_f32`] / [`encode_f64`].
pub fn decode(bytes: &[u8]) -> Result<Decoded, DecodeError> {
    let mut r = ByteReader::new(bytes);
    if r.take(4)? != MAGIC {
        return Err(DecodeError::Invalid("magic"));
    }
    let dt = match r.u8()? {
        0 => Dtype::F32,
        1 => Dtype::F64,
        _ => return Err(DecodeError::Invalid("dtype")),
    };
    let order = match r.u8()? {
        0 => DigitOrder::RowMajor,
        1 => DigitOrder::Morton,
        _ => return Err(DecodeError::Invalid("order")),
    };
    let pad = match r.u8()? {
        0 => PadPolicy::MixedRadix,
        1 => PadPolicy::Pow2,
        _ => return Err(DecodeError::Invalid("pad")),
    };
    let nd = r.varint()? as usize;
    if nd == 0 || nd > 16 {
        return Err(DecodeError::Invalid("ndims"));
    }
    let dims: Vec<usize> = (0..nd)
        .map(|_| Ok(r.varint()? as usize))
        .collect::<Result<_, DecodeError>>()?;
    if dims.contains(&0)
        || dims
            .iter()
            .try_fold(1usize, |a, &n| a.checked_mul(n))
            .is_none()
    {
        return Err(DecodeError::Invalid("dims"));
    }
    let layout = Layout::new(&dims, order, pad);
    let delta = r.f64()?;
    let d = layout.modes.len();
    let mut ranks = vec![1usize];
    for _ in 1..d {
        let rk = r.varint()? as usize;
        if rk > 1 << 16 {
            return Err(DecodeError::Invalid("rank"));
        }
        ranks.push(rk);
    }
    ranks.push(1);
    let mut exps = Vec::new();
    for k in 0..d.saturating_sub(1) {
        let mut prev = 0i64;
        let mut e = Vec::with_capacity(ranks[k + 1]);
        for _ in 0..ranks[k + 1] {
            prev += r.svarint()?;
            e.push(i32::try_from(prev).map_err(|_| DecodeError::Invalid("exponent"))?);
        }
        exps.push(e);
    }
    let any_zero = ranks.contains(&0);
    let lens: Vec<usize> = (0..d)
        .map(|k| {
            if any_zero {
                0
            } else {
                ranks[k] * layout.modes[k] * ranks[k + 1]
            }
        })
        .collect();
    let q = decode_blocks(&mut r, &lens)?;
    let pass = QPass {
        ranks,
        exps,
        q,
        sse: 0.0,
    };
    let rec = reconstruct(&layout, &pass, delta);
    let mut out: Vec<f64> = rec.iter().map(|&v| cast(v, dt)).collect();
    if r.u8()? == 1 {
        let bound = r.f64()?;
        let nres = r.varint()? as usize;
        if nres > out.len() {
            return Err(DecodeError::Invalid("residual count"));
        }
        let bl = decode_blocks(&mut r, &[nres, nres])?;
        let (gaps, vals) = (&bl[0], &bl[1]);
        let mut p = 0usize;
        for i in 0..nres {
            p = if i == 0 {
                gaps[0] as usize
            } else {
                p + 1 + gaps[i] as usize
            };
            if p >= out.len() {
                return Err(DecodeError::Invalid("residual position"));
            }
            out[p] = if vals[i] == 0 {
                match dt {
                    Dtype::F32 => f32::from_le_bytes(r.take(4)?.try_into().unwrap()) as f64,
                    Dtype::F64 => f64::from_le_bytes(r.take(8)?.try_into().unwrap()),
                }
            } else {
                cast(rec[p] + 2.0 * bound * vals[i] as f64, dt)
            };
        }
    }
    Ok(match dt {
        Dtype::F32 => Decoded::F32 {
            dims,
            data: out.iter().map(|&v| v as f32).collect(),
        },
        Dtype::F64 => Decoded::F64 { dims, data: out },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (30.0 * t).sin() * (-2.0 * t).exp()
                    + 0.3 * (7.0 * t).cos()
                    + if t > 0.6 { 0.5 } else { 0.0 }
            })
            .collect()
    }

    fn as_f64(d: Decoded) -> Vec<f64> {
        match d {
            Decoded::F32 { data, .. } => data.iter().map(|&v| v as f64).collect(),
            Decoded::F64 { data, .. } => data,
        }
    }

    #[test]
    fn predicted_sse_matches_reconstruction() {
        let x = field(1 << 12);
        let layout = Layout::new(&[1 << 12], DigitOrder::RowMajor, PadPolicy::Pow2);
        for delta in [1e-2, 1e-3, 1e-4] {
            let p = quantized_pass(&x, &layout.modes, delta, false, 512);
            let rec = reconstruct(&layout, &p, delta);
            let sse: f64 = x.iter().zip(&rec).map(|(a, b)| (a - b) * (a - b)).sum();
            // Agreement is limited by the Gram-route eigensolver, not the bookkeeping.
            assert!((sse - p.sse).abs() <= 1e-3 * sse, "{sse} vs {}", p.sse);
        }
    }

    #[test]
    fn psnr_and_rmse_bounds_hold() {
        let x = field(3000); // non power of two: 3000 = 3·5³·2³
        for psnr in [40.0, 60.0, 80.0] {
            let cfg = Config::new(ErrorBound::Psnr(psnr));
            let (bytes, st) = encode_f64(&x, &[3000], &cfg);
            let rec = as_f64(decode(&bytes).unwrap());
            let (rmse, _) = errors(&x, &rec);
            let achieved = 20.0 * (st.range / rmse).log10();
            assert!(achieved >= psnr - 1e-9, "psnr {achieved} < {psnr}");
            assert_eq!(st.bytes, bytes.len());
        }
    }

    #[test]
    fn max_abs_bound_holds_f32_2d() {
        let (h, w) = (48usize, 80usize);
        let x: Vec<f32> = (0..h * w)
            .map(|i| {
                let (a, b) = ((i / w) as f32 / h as f32, (i % w) as f32 / w as f32);
                (6.0 * a).sin() * (4.0 * b).cos() + if a + b > 1.1 { 0.7 } else { 0.0 }
            })
            .collect();
        for rel in [1e-2, 1e-3, 1e-4] {
            let bound = rel * 2.4;
            for order in [DigitOrder::Morton, DigitOrder::RowMajor] {
                let mut cfg = Config::new(ErrorBound::MaxAbs(bound));
                cfg.order = order;
                let (bytes, st) = encode_f32(&x, &[h, w], &cfg);
                let Decoded::F32 { data, dims } = decode(&bytes).unwrap() else {
                    panic!()
                };
                assert_eq!(dims, vec![h, w]);
                let mx = x
                    .iter()
                    .zip(&data)
                    .map(|(a, b)| (a - b).abs() as f64)
                    .fold(0.0, f64::max);
                assert!(mx <= bound, "max {mx} > {bound}");
                assert_eq!(st.max_abs, mx);
            }
        }
    }

    #[test]
    fn deterministic_stream_and_decode() {
        let x = field(1 << 11);
        let cfg = Config::new(ErrorBound::MaxAbs(1e-3));
        let (a, _) = encode_f64(&x, &[1 << 11], &cfg);
        let (b, _) = encode_f64(&x, &[1 << 11], &cfg);
        assert_eq!(a, b);
        assert_eq!(decode(&a).unwrap(), decode(&a).unwrap());
    }

    #[test]
    fn low_rank_signal_is_tiny() {
        let n = 1 << 16;
        let x: Vec<f64> = (0..n)
            .map(|i| (0.001 * i as f64).sin() + 0.5 * (0.0003 * i as f64).cos())
            .collect();
        let (bytes, st) = encode_f64(&x, &[n], &Config::new(ErrorBound::Psnr(80.0)));
        assert!(st.ranks.iter().all(|&r| r <= 6), "{:?}", st.ranks);
        assert!(bytes.len() < 2000, "{} bytes", bytes.len());
    }

    #[test]
    fn rank_cap_and_zero_rank_still_meet_max_abs() {
        // White noise at rank cap 4: the L2 target is unreachable, so the
        // residual stream must carry the bound. A huge Δ quantizes everything
        // to zero (rank 0); the stream must still parse.
        let mut seed = 9u64;
        let x: Vec<f64> = (0..4096)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            })
            .collect();
        let mut cfg = Config::new(ErrorBound::MaxAbs(1e-3));
        cfg.max_rank = 4;
        let (bytes, st) = encode_f64(&x, &[4096], &cfg);
        let rec = as_f64(decode(&bytes).unwrap());
        assert!(errors(&x, &rec).1 <= 1e-3, "{}", st.max_abs);
        let (bytes, _) = encode_f64(&x, &[4096], &Config::new(ErrorBound::MaxAbs(0.45)));
        let rec = as_f64(decode(&bytes).unwrap());
        assert!(errors(&x, &rec).1 <= 0.45);
    }

    #[test]
    fn rejects_garbage() {
        assert!(decode(b"nope").is_err());
        let x = field(256);
        let (bytes, _) = encode_f64(&x, &[256], &Config::new(ErrorBound::Psnr(50.0)));
        assert!(decode(&bytes[..bytes.len() - 3]).is_err());
    }
}
