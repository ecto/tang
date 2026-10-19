//! Tucker (HOSVD) codec, TTHRESH-style, on the shared quantizer / context
//! coder / range coder stack.
//!
//! 1. **HOSVD.** For each mode `k` the eigenvectors of the mode Gram
//!    `X₍ₖ₎X₍ₖ₎ᵀ` give an orthonormal factor `Uₖ`; the core
//!    `C = X ×₁U₁ᵀ ×₂U₂ᵀ ×₃U₃ᵀ` concentrates the energy at low indices.
//! 2. **Ranks by rate–distortion.** The core is quantized with the deadzone
//!    quantizer at step `Δ`; `Rₖ` = 1 + the last mode-`k` slice with a nonzero
//!    quantized coefficient.
//! 3. **Factor quantization, equal slope.** Perturbing column `j` of `Uₖ` by
//!    `e` costs `‖e‖²·‖C₍ₖ₎[j,:]‖²`, so the column gets step
//!    `sⱼ = φΔ/‖C₍ₖ₎[j,:]‖` (stored as a quarter-octave exponent). Columns are
//!    coded as DPCM residuals (order 0/1/2, chosen per column) with a Rice
//!    split: adaptive Elias-gamma high part + raw low bits.
//! 4. **Error feedback.** The core is recomputed by least squares against the
//!    *quantized* factors, `C = X ×ₖ (ÛₖᵀÛₖ)⁻¹Ûₖᵀ`, then quantized and coded
//!    with the wavelet coefficient coder over a dyadic pseudo-pyramid in core
//!    index space (energy decays with index the way it does across wavelet
//!    levels, so the same band / parent / neighbour contexts apply).
//! 5. **Rate control.** `Δ` is bisected on the measured RMSE of the decoder's
//!    own reconstruction.
//!
//! Reconstruction uses faer GEMMs (sequential): deterministic on a given CPU;
//! across ISAs the float output may differ in the last ulp. The bitstream
//! itself is integer-coded and identical everywhere. No max-abs mode
//! (TTHRESH has none either).

use crate::abac::{Decoder, Encoder, UintModel, ONE};
use crate::bitio::{ByteReader, ByteWriter, DecodeError};
use crate::codec::{Decoded, ErrorBound};
use crate::coefcode::{code_bands, fit_predictors, read_preds, write_preds, Bits, HandModel};
use crate::linalg::{cholesky, gram_cols, gram_rows, matmul, matmul_tn, solve_lower, solve_lower_t, sym_eig_desc, transpose};
use crate::wavelet::{as3, Pyramid, MAX_LEVELS};

/// Stream magic.
pub const MAGIC: &[u8; 4] = b"TCT1";

const RHO: f64 = 0.45;
/// Factor-step multiplier `φ`.
const PHI: f64 = 1.0;

/// What the encoder did.
#[derive(Clone, Debug, Default)]
pub struct TuckerStats {
    /// Core payload bytes.
    pub core_bytes: usize,
    /// Factor bytes.
    pub factor_bytes: usize,
    /// Kept ranks.
    pub ranks: Vec<usize>,
    /// Core step.
    pub delta: f64,
    /// Measured RMSE.
    pub rmse: f64,
    /// Measured max-abs error.
    pub max_abs: f64,
}

/// `X ×ₖ M` for `M` (n_k × r): mode `k` becomes size `r`,
/// `Y[.., j, ..] = Σᵢ M[i, j] X[.., i, ..]`.
fn mode_mul(x: &[f64], d: [usize; 3], k: usize, m: &[f64], r: usize) -> (Vec<f64>, [usize; 3]) {
    let n = d[k];
    let mut od = d;
    od[k] = r;
    let y = match k {
        0 => matmul_tn(m, x, n, r, d[1] * d[2]),
        2 => matmul(x, m, d[0] * d[1], n, r),
        _ => {
            let mut y = Vec::with_capacity(d[0] * r * d[2]);
            for i in 0..d[0] {
                let s = &x[i * d[1] * d[2]..(i + 1) * d[1] * d[2]];
                y.extend(matmul_tn(m, s, n, r, d[2]));
            }
            y
        }
    };
    (y, od)
}

/// Inverse direction: `M` is n × r, mode `k` (size r) grows back to `n`,
/// `Y[.., i, ..] = Σⱼ M[i, j] X[.., j, ..]`.
fn mode_mul_t(x: &[f64], d: [usize; 3], k: usize, m: &[f64], n: usize) -> (Vec<f64>, [usize; 3]) {
    let r = d[k];
    let mt = transpose(m, n, r);
    mode_mul(x, d, k, &mt, n)
}

fn mode_gram(x: &[f64], d: [usize; 3], k: usize) -> Vec<f64> {
    match k {
        0 => gram_rows(x, d[0], d[1] * d[2]),
        2 => gram_cols(x, d[0] * d[1], d[2]),
        _ => {
            let mut g = vec![0.0; d[1] * d[1]];
            for i in 0..d[0] {
                let s = &x[i * d[1] * d[2]..(i + 1) * d[1] * d[2]];
                for (a, b) in g.iter_mut().zip(gram_rows(s, d[1], d[2])) {
                    *a += b;
                }
            }
            g
        }
    }
}

/// Mode-`k` slice norms of a core.
fn slice_norms(c: &[f64], d: [usize; 3], k: usize) -> Vec<f64> {
    let mut s = vec![0.0; d[k]];
    for i in 0..d[0] {
        for j in 0..d[1] {
            for l in 0..d[2] {
                let v = c[(i * d[1] + j) * d[2] + l];
                s[[i, j, l][k]] += v * v;
            }
        }
    }
    s.iter().map(|v| v.sqrt()).collect()
}

fn quantize_core(c: &[f64], delta: f64) -> (Vec<i32>, [u8; 2]) {
    let inv = 1.0 / delta;
    let (mut s, mut n) = ([0.0f64; 2], [0u64; 2]);
    let q = c
        .iter()
        .map(|&v| {
            let a = v.abs() * inv;
            let m = (a + RHO).floor().min(1e9);
            if m < 1.0 {
                return 0;
            }
            let j = (m >= 2.0) as usize;
            s[j] += m - a;
            n[j] += 1;
            if v < 0.0 {
                -(m as i32)
            } else {
                m as i32
            }
        })
        .collect();
    let d = [0, 1].map(|j| if n[j] > 0 { ((s[j] / n[j] as f64).clamp(0.0, 0.498) * 512.0).round() as u8 } else { 0 });
    (q, d)
}

fn dequantize_core(q: &[i32], delta: f64, d: [u8; 2]) -> Vec<f64> {
    q.iter()
        .map(|&v| {
            if v == 0 {
                return 0.0;
            }
            let m = v.unsigned_abs() as f64 - d[(v.unsigned_abs() >= 2) as usize] as f64 / 512.0;
            if v < 0 {
                -m * delta
            } else {
                m * delta
            }
        })
        .collect()
}

/// One quantized factor matrix (n × r) with its column exponents.
#[derive(Clone, Debug)]
struct QFactor {
    n: usize,
    r: usize,
    exps: Vec<i32>,
    q: Vec<i64>, // column-major ints
}

impl QFactor {
    fn quantize(u: &[f64], n: usize, full_r: usize, r: usize, norms: &[f64], delta: f64) -> Self {
        let mut exps = Vec::with_capacity(r);
        let mut q = Vec::with_capacity(n * r);
        for j in 0..r {
            let s = PHI * delta / norms[j].max(1e-300);
            let e = (4.0 * s.log2()).round().clamp(-4000.0, 4000.0) as i32;
            let s = 2f64.powf(e as f64 / 4.0);
            exps.push(e);
            for i in 0..n {
                q.push((u[i * full_r + j] / s).round().clamp(-4e15, 4e15) as i64);
            }
        }
        Self { n, r, exps, q }
    }
    fn dequantize(&self) -> Vec<f64> {
        let mut u = vec![0.0; self.n * self.r];
        for j in 0..self.r {
            let s = 2f64.powf(self.exps[j] as f64 / 4.0);
            for i in 0..self.n {
                u[i * self.r + j] = self.q[j * self.n + i] as f64 * s;
            }
        }
        // a column quantized to all zeros would make the Gram singular
        for j in 0..self.r {
            if (0..self.n).all(|i| u[i * self.r + j] == 0.0) {
                u[(j % self.n) * self.r + j] = 2f64.powf(self.exps[j] as f64 / 4.0);
            }
        }
        u
    }
}

#[inline]
fn dpcm_pred(c: &[i64], i: usize, order: usize) -> i64 {
    match (order, i) {
        (0, _) | (_, 0) => 0,
        (1, _) | (2, 1) => c[i - 1],
        _ => 2 * c[i - 1] - c[i - 2],
    }
}

fn zz(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

fn unzz(u: u64) -> i64 {
    ((u >> 1) as i64) ^ -((u & 1) as i64)
}

fn code_factor<B: Bits>(b: &mut B, f: &mut QFactor, km: &mut UintModel, hm: &mut UintModel) {
    for j in 0..f.r {
        let col = f.q[j * f.n..(j + 1) * f.n].to_vec();
        // encoder picks the DPCM order and Rice parameter; the decoder reads them
        let res_of = |o: usize| -> Vec<u64> { (0..col.len()).map(|i| zz(col[i] - dpcm_pred(&col, i, o))).collect() };
        let bo = (0..3).min_by_key(|&o| res_of(o).iter().sum::<u64>()).unwrap();
        let res = res_of(bo);
        let mean = res.iter().sum::<u64>() / f.n.max(1) as u64;
        let k = if mean > 2 { (mean / 2).ilog2() } else { 0 };
        let o = b.bitp(bo & 1 == 1, ONE / 2) as usize | (b.bitp(bo & 2 == 2, ONE / 2) as usize) << 1;
        let k = b.uint(k as u64, km) as u32;
        if k > 62 || o > 2 {
            return; // corrupt stream; decoder output is garbage but bounded
        }
        let mut out: Vec<i64> = Vec::with_capacity(f.n);
        for i in 0..f.n {
            let u = res[i];
            let hi = b.uint(u >> k, hm);
            let mut lo = 0u64;
            for t in (0..k).rev() {
                lo = (lo << 1) | b.bitp((u >> t) & 1 == 1, ONE / 2) as u64;
            }
            let v = unzz((hi << k) | lo) + dpcm_pred(&out, i, o);
            out.push(v);
        }
        f.q[j * f.n..(j + 1) * f.n].copy_from_slice(&out);
    }
}

/// Least-squares core against (non-orthonormal) factors: `X ×ₖ (UᵀU)⁻¹Uᵀ`.
fn ls_core(x: &[f64], d: [usize; 3], us: &[Vec<f64>], rs: [usize; 3]) -> Vec<f64> {
    let mut c = x.to_vec();
    let mut cd = d;
    for k in 0..3 {
        let (n, r) = (d[k], rs[k]);
        let g = matmul_tn(&us[k], &us[k], n, r, r);
        // quantized columns can be (numerically) dependent on tiny-range
        // fields: retry with a growing ridge (encoder-side only; the decoder
        // never solves this system, so streams that factor cleanly are unchanged)
        let tr = (0..r).map(|i| g[i * r + i]).sum::<f64>() / r as f64;
        let l = cholesky(&g, r)
            .or_else(|| {
                (1..=12).find_map(|e| {
                    let mut gg = g.clone();
                    for i in 0..r {
                        gg[i * r + i] += tr * 10f64.powi(e - 16);
                    }
                    cholesky(&gg, r)
                })
            })
            .expect("factor Gram not SPD even with ridge");
        // Z = (UᵀU)⁻¹Uᵀ (r × n); mode_mul wants M = Zᵀ (n × r)
        let mut z = transpose(&us[k], n, r);
        solve_lower(&l, &mut z, r, n);
        solve_lower_t(&l, &mut z, r, n);
        let m = transpose(&z, r, n);
        let (y, yd) = mode_mul(&c, cd, k, &m, r);
        c = y;
        cd = yd;
    }
    c
}

fn reconstruct(core: &[f64], rs: [usize; 3], us: &[Vec<f64>], d: [usize; 3]) -> Vec<f64> {
    let mut y = core.to_vec();
    let mut yd = rs;
    for k in 0..3 {
        let (z, zd) = mode_mul_t(&y, yd, k, &us[k], d[k]);
        y = z;
        yd = zd;
    }
    y
}

struct Hosvd {
    d: [usize; 3],
    us: Vec<Vec<f64>>, // full orthonormal factors n×n
    core: Vec<f64>,
}

fn hosvd(x: &[f64], d: [usize; 3]) -> Hosvd {
    let us: Vec<Vec<f64>> = (0..3).map(|k| sym_eig_desc(&mode_gram(x, d, k), d[k]).1).collect();
    let mut c = x.to_vec();
    let mut cd = d;
    for k in 0..3 {
        let (y, yd) = mode_mul(&c, cd, k, &us[k], d[k]);
        c = y;
        cd = yd;
    }
    Hosvd { d, us, core: c }
}

/// Everything one trial at step `Δ` produces.
struct Trial {
    rs: [usize; 3],
    facs: Vec<QFactor>,
    q: Vec<i32>,
    dl: [u8; 2],
    y: Vec<f64>,
}

fn trial(x: &[f64], h: &Hosvd, delta: f64) -> Trial {
    let d = h.d;
    let (q0, _) = quantize_core(&h.core, delta);
    let mut rs = [1usize; 3];
    for i in 0..d[0] {
        for j in 0..d[1] {
            for l in 0..d[2] {
                if q0[(i * d[1] + j) * d[2] + l] != 0 {
                    rs = [rs[0].max(i + 1), rs[1].max(j + 1), rs[2].max(l + 1)];
                }
            }
        }
    }
    let facs: Vec<QFactor> = (0..3)
        .map(|k| QFactor::quantize(&h.us[k], d[k], d[k], rs[k], &slice_norms(&h.core, d, k), delta))
        .collect();
    let ud: Vec<Vec<f64>> = facs.iter().map(|f| f.dequantize()).collect();
    let core = ls_core(x, d, &ud, rs);
    let (q, dl) = quantize_core(&core, delta);
    let y = reconstruct(&dequantize_core(&q, delta, dl), rs, &ud, d);
    Trial { rs, facs, q, dl, y }
}

fn err_of(x: &[f64], y: &[f64]) -> (f64, f64) {
    let mut se = 0.0;
    let mut mx = 0.0f64;
    for (a, b) in x.iter().zip(y) {
        let e = a - *b as f32 as f64;
        se += e * e;
        mx = mx.max(e.abs());
    }
    ((se / x.len() as f64).sqrt(), mx)
}

/// Encode f32 samples (1D–3D, row-major) at a PSNR or RMSE target.
pub fn encode_f32(x: &[f32], dims: &[usize], bound: ErrorBound) -> (Vec<u8>, TuckerStats) {
    let x: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let d = as3(dims);
    assert_eq!(d.iter().product::<usize>(), x.len());
    let (lo_v, hi_v) = x.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
    let target = match bound {
        ErrorBound::Rmse(r) => r,
        ErrorBound::Psnr(db) => (hi_v - lo_v) * 10f64.powf(-db / 20.0),
        ErrorBound::MaxAbs(_) => panic!("the Tucker codec has no max-abs mode; use the wavelet codec"),
    }
    .max(f64::MIN_POSITIVE);
    let h = hosvd(&x, d);
    let ok = |delta: f64| err_of(&x, &trial(&x, &h, delta).y).0 <= target;
    let (mut lo, mut hi) = (target * 0.5, target * 6.0);
    let mut guard = 0;
    while !ok(lo) && guard < 40 {
        hi = lo;
        lo *= 0.5;
        guard += 1;
    }
    while ok(hi) && guard < 80 {
        lo = hi;
        hi *= 2.0;
        guard += 1;
    }
    for _ in 0..12 {
        let mid = (lo * hi).sqrt();
        if ok(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let delta = lo;
    let t = trial(&x, &h, delta);
    let mut w = ByteWriter::new();
    w.bytes(MAGIC);
    w.u8(0); // f32
    w.varint(dims.len() as u64);
    for &v in dims {
        w.varint(v as u64);
    }
    w.f64(delta);
    for k in 0..3 {
        w.varint(t.rs[k] as u64);
    }
    w.bytes(&t.dl);
    for f in &t.facs {
        let mut prev = 0;
        for &e in &f.exps {
            w.svarint((e - prev) as i64);
            prev = e;
        }
    }
    let fb0 = w.buf.len();
    let mut e = Encoder::new();
    let (mut km, mut hm) = (UintModel::default(), UintModel::default());
    for f in &t.facs {
        let mut f = f.clone();
        code_factor(&mut e, &mut f, &mut km, &mut hm);
    }
    w.block(&e.finish());
    let factor_bytes = w.buf.len() - fb0;
    let p = Pyramid::new(t.rs, MAX_LEVELS);
    let preds = fit_predictors(&p, &t.q);
    let cb0 = w.buf.len();
    write_preds(&mut w, &preds);
    w.u8(0);
    let mut e = Encoder::new();
    let mut qq = t.q.clone();
    code_bands(&p, &mut qq, &preds, &mut e, &mut HandModel::default());
    w.block(&e.finish());
    let core_bytes = w.buf.len() - cb0;
    let (rmse, max_abs) = err_of(&x, &t.y);
    let st = TuckerStats { core_bytes, factor_bytes, ranks: t.rs.to_vec(), delta, rmse, max_abs };
    (w.buf, st)
}

/// Decode a Tucker stream.
pub fn decode(bytes: &[u8]) -> Result<Decoded, DecodeError> {
    let mut r = ByteReader::new(bytes);
    if r.take(4)? != MAGIC {
        return Err(DecodeError::Invalid("magic"));
    }
    if r.u8()? != 0 {
        return Err(DecodeError::Invalid("dtype"));
    }
    let nd = r.varint()? as usize;
    if nd == 0 || nd > 3 {
        return Err(DecodeError::Invalid("ndims"));
    }
    let mut dims = Vec::new();
    for _ in 0..nd {
        dims.push(r.varint()? as usize);
    }
    let d = as3(&dims);
    if d.iter().try_fold(1usize, |a, &v| a.checked_mul(v)).is_none_or(|n| n == 0 || n > 1 << 34) {
        return Err(DecodeError::Invalid("dims"));
    }
    let delta = r.f64()?;
    let mut rs = [0usize; 3];
    for k in 0..3 {
        rs[k] = r.varint()? as usize;
        if rs[k] == 0 || rs[k] > d[k] {
            return Err(DecodeError::Invalid("rank"));
        }
    }
    let dl = [r.u8()?, r.u8()?];
    let mut facs = Vec::new();
    for k in 0..3 {
        let mut exps = Vec::new();
        let mut prev = 0i64;
        for _ in 0..rs[k] {
            prev += r.svarint()?;
            exps.push(prev.clamp(-4000, 4000) as i32);
        }
        facs.push(QFactor { n: d[k], r: rs[k], exps, q: vec![0; d[k] * rs[k]] });
    }
    let buf = r.block()?;
    let mut dd = Decoder::new(buf);
    let (mut km, mut hm) = (UintModel::default(), UintModel::default());
    for f in facs.iter_mut() {
        code_factor(&mut dd, f, &mut km, &mut hm);
    }
    let ud: Vec<Vec<f64>> = facs.iter().map(|f| f.dequantize()).collect();
    let p = Pyramid::new(rs, MAX_LEVELS);
    let preds = read_preds(&mut r, p.bands.len())?;
    if r.u8()? != 0 {
        return Err(DecodeError::Invalid("core model"));
    }
    let buf = r.block()?;
    let mut dd = Decoder::new(buf);
    let mut q = vec![0i32; p.len()];
    code_bands(&p, &mut q, &preds, &mut dd, &mut HandModel::default());
    let y = reconstruct(&dequantize_core(&q, delta, dl), rs, &ud, d);
    Ok(Decoded::F32 { dims, data: y.iter().map(|&v| v as f32).collect() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tucker_round_trip_meets_psnr() {
        let d = [12, 20, 16];
        let mut x = Vec::new();
        for i in 0..d[0] {
            for j in 0..d[1] {
                for k in 0..d[2] {
                    let (a, b, c) = (i as f32 / 12.0, j as f32 / 20.0, k as f32 / 16.0);
                    x.push((3.0 * a).sin() * (2.0 * b).cos() + c * c + 0.1 * (9.0 * a * c).sin());
                }
            }
        }
        for db in [50.0, 90.0] {
            let (bytes, st) = encode_f32(&x, &d, ErrorBound::Psnr(db));
            let Decoded::F32 { data, .. } = decode(&bytes).unwrap() else { panic!() };
            let xd: Vec<f64> = x.iter().map(|&v| v as f64).collect();
            let (rmse, _) = err_of(&xd, &data.iter().map(|&v| v as f64).collect::<Vec<_>>());
            assert_eq!(rmse, st.rmse);
            let range = xd.iter().cloned().fold(f64::MIN, f64::max) - xd.iter().cloned().fold(f64::MAX, f64::min);
            assert!(20.0 * (range / rmse).log10() >= db - 1e-9);
            assert_eq!(encode_f32(&x, &d, ErrorBound::Psnr(db)).0, bytes);
        }
    }
}
