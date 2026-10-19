//! The wavelet codec: CDF 9/7 pyramid → deadzone quantizer → context model →
//! adaptive binary range coding, with a SPERR-style outlier stream for
//! pointwise bounds and an integer 5/3 lossless mode.
//!
//! # Lossy path
//!
//! 1. Forward multi-level CDF 9/7 ([`crate::wavelet`]); every coefficient is
//!    multiplied by its band's synthesis-basis norm, so one uniform step `Δ`
//!    is equal-slope (MSE-matched) across bands.
//! 2. Deadzone uniform quantizer `q = sgn(c)·⌊|c|/Δ + ρ⌋` (`ρ < ½` widens the
//!    zero bin); reconstruction at the per-(class, |q|∈{1,2,≥3}) centroid
//!    `|ĉ| = (|q| − δ)Δ`, the 30 `δ`s transmitted in 1/512 units.
//! 3. Coefficients coded by [`crate::coefcode`] (hand contexts) or a learned
//!    model ([`crate::learned`]).
//! 4. Rate control: for a PSNR/RMSE target `Δ` is bisected on the *measured*
//!    error of the actual decoder output.
//!
//! # Max-abs mode
//!
//! For a bound `b` the wavelet layer is coded at `Δ = f·b` for several `f`;
//! the encoder decodes its own output and every cell with `|x − x̂| > b` gets a
//! correction `k = round((x − x̂)/2b)` (then `|x − (x̂ + 2bk)| ≤ b`), coded as a
//! context-modelled significance map plus `|k|`; cells a float cast still
//! pushes over `b` are stored raw. The `f` with the smallest total wins and
//! the bound is re-verified on the exact decoder output before returning.

use crate::abac::{BitModel, Decoder, Encoder, UintModel};
use crate::bitio::{ByteReader, ByteWriter, DecodeError};
use crate::codec::{Decoded, Dtype, ErrorBound};
use crate::coefcode::{band_class, code_bands, fit_predictors, read_preds, write_preds, Bits, HandModel, Pred, NCLASS};
use crate::wavelet::{as3, forward_53, forward_97, inverse_53, inverse_97, Pyramid, MAX_LEVELS, MAX_LEVELS_1D};

/// Stream magic of the wavelet codec.
pub const MAGIC: &[u8; 4] = b"TCW1";

/// Which coefficient model codes the stream.
#[derive(Clone, Debug, Default)]
pub enum ModelChoice {
    /// Hand-designed contexts.
    #[default]
    Hand,
    /// A learned network (pretrained, or overfitted and shipped in-stream).
    Learned(crate::learned::LearnedSpec),
}

/// Wavelet-codec options.
#[derive(Clone, Debug)]
pub struct WavConfig {
    /// Error target.
    pub bound: ErrorBound,
    /// Deadzone rounding offset `ρ` (0.5 = plain rounding).
    pub rho: f64,
    /// Coefficient model.
    pub model: ModelChoice,
}

impl WavConfig {
    /// Defaults for a bound.
    pub fn new(bound: ErrorBound) -> Self {
        Self { bound, rho: 0.45, model: ModelChoice::Hand }
    }
}

/// What the encoder did.
#[derive(Clone, Debug, Default)]
pub struct WavStats {
    /// Total bytes.
    pub bytes: usize,
    /// Bits per value.
    pub bits_per_value: f64,
    /// Coefficient payload bytes.
    pub coef_bytes: usize,
    /// Model bytes shipped in the stream (overfitted mode).
    pub model_bytes: usize,
    /// Outlier stream bytes.
    pub outlier_bytes: usize,
    /// Number of corrected cells.
    pub n_outliers: usize,
    /// Final step.
    pub delta: f64,
    /// Measured RMSE.
    pub rmse: f64,
    /// Measured max-abs error.
    pub max_abs: f64,
    /// Input range.
    pub range: f64,
    /// Rate-control passes.
    pub passes: usize,
}

const NDELTA: usize = NCLASS * 3;

fn max_levels(dims: &[usize]) -> usize {
    if dims.iter().filter(|&&d| d > 1).count() <= 1 {
        MAX_LEVELS_1D
    } else {
        MAX_LEVELS
    }
}

/// Forward-transformed, band-weighted coefficients.
pub struct Coeffs {
    /// Pyramid.
    pub p: Pyramid,
    /// Weighted coefficients (volume layout).
    pub c: Vec<f64>,
    /// Weight of each cell's band.
    pub w: Vec<f64>,
    /// Class of each cell's band.
    pub cls: Vec<u8>,
}

impl Coeffs {
    /// Transform `x` with shape `dims`.
    pub fn new(x: &[f64], dims: &[usize]) -> Self {
        let p = Pyramid::new(as3(dims), max_levels(dims));
        let mut c = x.to_vec();
        forward_97(&p, &mut c);
        let (w, cls) = band_maps(&p);
        for (v, wi) in c.iter_mut().zip(&w) {
            *v *= wi;
        }
        Self { p, c, w, cls }
    }
}

fn band_maps(p: &Pyramid) -> (Vec<f64>, Vec<u8>) {
    let [_, ny, nx] = p.dims;
    let mut w = vec![0.0; p.len()];
    let mut cls = vec![0u8; p.len()];
    for b in &p.bands {
        let wb = p.band_weight(b);
        let cb = band_class(b);
        for z in b.start[0]..b.end[0] {
            for y in b.start[1]..b.end[1] {
                for x in b.start[2]..b.end[2] {
                    let i = (z * ny + y) * nx + x;
                    w[i] = wb;
                    cls[i] = cb;
                }
            }
        }
    }
    (w, cls)
}

/// Deadzone quantization with centroid offsets.
pub fn quantize(co: &Coeffs, delta: f64, rho: f64) -> (Vec<i32>, [u8; NDELTA]) {
    let inv = 1.0 / delta;
    let mut sum = [0.0f64; NDELTA];
    let mut cnt = [0u64; NDELTA];
    let q: Vec<i32> = co
        .c
        .iter()
        .zip(&co.cls)
        .map(|(&c, &k)| {
            let a = c.abs() * inv;
            let m = (a + rho).floor();
            if m >= 1.0 {
                let m = m.min(1.0e9);
                let j = k as usize * 3 + (m as usize).min(3) - 1;
                sum[j] += m - a;
                cnt[j] += 1;
                if c < 0.0 {
                    -(m as i32)
                } else {
                    m as i32
                }
            } else {
                0
            }
        })
        .collect();
    let mut d = [0u8; NDELTA];
    for j in 0..NDELTA {
        if cnt[j] > 0 {
            d[j] = ((sum[j] / cnt[j] as f64).clamp(0.0, 0.498) * 512.0).round() as u8;
        }
    }
    (q, d)
}

/// Dequantize + inverse transform → reconstruction (f64, row-major).
pub fn reconstruct(p: &Pyramid, q: &[i32], delta: f64, d: &[u8; NDELTA]) -> Vec<f64> {
    let (w, cls) = band_maps(p);
    let mut c: Vec<f64> = q
        .iter()
        .zip(w.iter().zip(&cls))
        .map(|(&v, (&wi, &k))| {
            if v == 0 {
                return 0.0;
            }
            let m = v.unsigned_abs() as f64;
            let j = k as usize * 3 + (v.unsigned_abs() as usize).min(3) - 1;
            let r = (m - d[j] as f64 / 512.0) * delta / wi;
            if v < 0 {
                -r
            } else {
                r
            }
        })
        .collect();
    inverse_97(p, &mut c);
    c
}

fn cast(y: &[f64], dt: Dtype) -> Vec<f64> {
    match dt {
        Dtype::F32 => y.iter().map(|&v| v as f32 as f64).collect(),
        Dtype::F64 => y.to_vec(),
    }
}

fn err_stats(x: &[f64], y: &[f64]) -> (f64, f64) {
    let mut se = 0.0;
    let mut mx = 0.0f64;
    for (a, b) in x.iter().zip(y) {
        let e = a - b;
        se += e * e;
        mx = mx.max(e.abs());
    }
    ((se / x.len() as f64).sqrt(), mx)
}

fn range_of(x: &[f64]) -> f64 {
    let (lo, hi) = x.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
    hi - lo
}

fn code_coefs(p: &Pyramid, q: &[i32], model: &ModelChoice, w: &mut ByteWriter) -> usize {
    let mut e = Encoder::new();
    let mut qq = q.to_vec();
    let preds: Vec<Pred> = if std::env::var("TC_NOPRED").is_ok() { vec![Pred::default(); p.bands.len()] } else { fit_predictors(p, q) };
    write_preds(w, &preds);
    let before = w.buf.len();
    match model {
        ModelChoice::Hand => {
            w.u8(0);
            code_bands(p, &mut qq, &preds, &mut e, &mut HandModel::default());
        }
        ModelChoice::Learned(spec) => {
            let mut m = spec.prepare(p, q, &preds, w);
            code_bands(p, &mut qq, &preds, &mut e, &mut m);
        }
    }
    let model_bytes = w.buf.len() - before;
    w.block(&e.finish());
    model_bytes
}

fn decode_coefs(p: &Pyramid, r: &mut ByteReader) -> Result<Vec<i32>, DecodeError> {
    let preds = read_preds(r, p.bands.len())?;
    let tag = r.u8()?;
    let mut q = vec![0i32; p.len()];
    match tag {
        0 => {
            let buf = r.block()?;
            let mut d = Decoder::new(buf);
            code_bands(p, &mut q, &preds, &mut d, &mut HandModel::default());
            if d.overrun() {
                return Err(DecodeError::Truncated);
            }
        }
        _ => {
            let mut m = crate::learned::read_model(tag, r)?;
            let buf = r.block()?;
            let mut d = Decoder::new(buf);
            code_bands(p, &mut q, &preds, &mut d, &mut m);
            if d.overrun() {
                return Err(DecodeError::Truncated);
            }
        }
    }
    Ok(q)
}

// ------------------------------------------------------------------ outliers

struct OutlierModel {
    flag: [BitModel; 8],
    sign: BitModel,
    mag: UintModel,
}

impl Default for OutlierModel {
    fn default() -> Self {
        Self { flag: [BitModel::new(2000); 8], sign: BitModel::default(), mag: UintModel::default() }
    }
}

fn code_outliers<B: Bits>(dims: [usize; 3], k: &mut [i32], b: &mut B) {
    let mut m = OutlierModel::default();
    let [nz, ny, nx] = dims;
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let i = (z * ny + y) * nx + x;
                let c = (x > 0 && k[i - 1] != 0) as usize
                    | ((y > 0 && k[i - nx] != 0) as usize) << 1
                    | ((z > 0 && k[i - nx * ny] != 0) as usize) << 2;
                if !b.bit(k[i] != 0, &mut m.flag[c]) {
                    k[i] = 0;
                    continue;
                }
                let neg = b.bit(k[i] < 0, &mut m.sign);
                let a = b.uint(k[i].unsigned_abs().saturating_sub(1) as u64, &mut m.mag) as i32 + 1;
                k[i] = if neg { -a } else { a };
            }
        }
    }
}

/// Apply corrections `k` (step `2b`) to `y`, then cast.
fn apply_outliers(y: &mut [f64], k: &[i32], b: f64) {
    for (v, &ki) in y.iter_mut().zip(k) {
        if ki != 0 {
            *v += 2.0 * b * ki as f64;
        }
    }
}

// ------------------------------------------------------------------ encode

struct Header {
    dt: Dtype,
    dims: Vec<usize>,
    lossless: bool,
}

fn write_header(w: &mut ByteWriter, h: &Header) {
    w.bytes(MAGIC);
    w.u8(match h.dt {
        Dtype::F32 => 0,
        Dtype::F64 => 1,
    });
    w.varint(h.dims.len() as u64);
    for &d in &h.dims {
        w.varint(d as u64);
    }
    w.u8(h.lossless as u8);
}

fn encode_at(
    x: &[f64],
    co: &Coeffs,
    h: &Header,
    delta: f64,
    cfg: &WavConfig,
    maxabs: Option<f64>,
) -> (Vec<u8>, WavStats, Vec<f64>) {
    let (q, d) = quantize(co, delta, cfg.rho);
    let mut w = ByteWriter::new();
    write_header(&mut w, h);
    w.f64(delta);
    w.bytes(&d);
    let b0 = w.buf.len();
    let model_bytes = code_coefs(&co.p, &q, &cfg.model, &mut w);
    let coef_bytes = w.buf.len() - b0 - model_bytes;
    let mut y = cast(&reconstruct(&co.p, &q, delta, &d), h.dt);
    let mut st = WavStats { delta, coef_bytes, model_bytes, ..Default::default() };
    if let Some(b) = maxabs {
        let b0 = w.buf.len();
        w.u8(1);
        w.f64(b);
        let k: Vec<i32> = x
            .iter()
            .zip(&y)
            .map(|(&a, &v)| {
                let r = a - v;
                if r.abs() > b {
                    (r / (2.0 * b)).round() as i32
                } else {
                    0
                }
            })
            .collect();
        let mut e = Encoder::new();
        code_outliers(co.p.dims, &mut k.clone(), &mut e);
        w.block(&e.finish());
        apply_outliers(&mut y, &k, b);
        let mut y = cast(&y, h.dt);
        // raw escapes for cells a float cast still pushes past b
        let esc: Vec<usize> = (0..x.len()).filter(|&i| (x[i] - y[i]).abs() > b).collect();
        w.varint(esc.len() as u64);
        let mut last = 0;
        for &i in &esc {
            w.varint((i - last) as u64);
            last = i;
            match h.dt {
                Dtype::F32 => w.bytes(&(x[i] as f32).to_le_bytes()),
                Dtype::F64 => w.f64(x[i]),
            }
            y[i] = x[i];
        }
        st.n_outliers = k.iter().filter(|&&v| v != 0).count() + esc.len();
        st.outlier_bytes = w.buf.len() - b0;
        let (rmse, mx) = err_stats(x, &y);
        st.rmse = rmse;
        st.max_abs = mx;
        st.bytes = w.buf.len();
        return (w.buf, st, y);
    }
    w.u8(0); // no outlier stream
    let (rmse, mx) = err_stats(x, &y);
    st.rmse = rmse;
    st.max_abs = mx;
    st.bytes = w.buf.len();
    (w.buf, st, y)
}

/// Error of the lossy layer alone at step `delta` (no entropy coding).
fn trial_rmse(x: &[f64], co: &Coeffs, delta: f64, rho: f64, dt: Dtype) -> f64 {
    let (q, d) = quantize(co, delta, rho);
    err_stats(x, &cast(&reconstruct(&co.p, &q, delta, &d), dt)).0
}

fn encode_impl(x: &[f64], dims: &[usize], dt: Dtype, cfg: &WavConfig) -> (Vec<u8>, WavStats) {
    assert_eq!(dims.iter().product::<usize>(), x.len());
    let range = range_of(x);
    let co = Coeffs::new(x, dims);
    let h = Header { dt, dims: dims.to_vec(), lossless: false };
    let rmse_target = match cfg.bound {
        ErrorBound::Rmse(r) => Some(r),
        ErrorBound::Psnr(db) => Some(range * 10f64.powf(-db / 20.0)),
        ErrorBound::MaxAbs(_) => None,
    };
    let (bytes, mut st) = if let Some(t) = rmse_target {
        let t = t.max(f64::MIN_POSITIVE);
        // bracket then bisect log Δ on the measured RMSE
        let mut lo = t * 0.5;
        let mut hi = t * 8.0;
        let mut passes = 0;
        while trial_rmse(x, &co, lo, cfg.rho, dt) > t && passes < 60 {
            hi = lo;
            lo *= 0.5;
            passes += 1;
        }
        while trial_rmse(x, &co, hi, cfg.rho, dt) <= t && passes < 60 {
            lo = hi;
            hi *= 2.0;
            passes += 1;
        }
        for _ in 0..14 {
            let mid = (lo * hi).sqrt();
            if trial_rmse(x, &co, mid, cfg.rho, dt) <= t {
                lo = mid;
            } else {
                hi = mid;
            }
            passes += 1;
        }
        let (b, mut s, _) = encode_at(x, &co, &h, lo, cfg, None);
        s.passes = passes;
        (b, s)
    } else {
        let ErrorBound::MaxAbs(b) = cfg.bound else { unreachable!() };
        // search the wavelet step with the hand model (cheap), then code the
        // winner with the configured model
        let hand = WavConfig { model: ModelChoice::Hand, ..cfg.clone() };
        let mut best: Option<(usize, f64)> = None;
        let mut passes = 0;
        for f in [1.0, 1.4, 1.8, 2.2, 2.7, 3.3, 4.0, 5.0] {
            let (bytes, _, _) = encode_at(x, &co, &h, f * b, &hand, Some(b));
            passes += 1;
            if best.is_none_or(|(n, _)| bytes.len() < n) {
                best = Some((bytes.len(), f));
            }
        }
        let (bytes, mut s, _) = encode_at(x, &co, &h, best.unwrap().1 * b, cfg, Some(b));
        s.passes = passes;
        (bytes, s)
    };
    st.range = range;
    st.bits_per_value = 8.0 * bytes.len() as f64 / x.len() as f64;
    (bytes, st)
}

/// Encode f32 samples (row-major, `dims` 1D–3D).
pub fn encode_f32(x: &[f32], dims: &[usize], cfg: &WavConfig) -> (Vec<u8>, WavStats) {
    let x: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    encode_impl(&x, dims, Dtype::F32, cfg)
}

/// Encode f64 samples.
pub fn encode_f64(x: &[f64], dims: &[usize], cfg: &WavConfig) -> (Vec<u8>, WavStats) {
    encode_impl(x, dims, Dtype::F64, cfg)
}

/// Lossless coding of integer samples via the reversible 5/3 transform.
pub fn encode_i32_lossless(x: &[i32], dims: &[usize]) -> Vec<u8> {
    let p = Pyramid::new(as3(dims), max_levels(dims));
    let mut c: Vec<i64> = x.iter().map(|&v| v as i64).collect();
    forward_53(&p, &mut c);
    let mut w = ByteWriter::new();
    write_header(&mut w, &Header { dt: Dtype::F32, dims: dims.to_vec(), lossless: true });
    // 5/3 coefficients of i32 data fit in i32 except for extreme inputs
    assert!(c.iter().all(|v| v.unsigned_abs() < (1 << 30)), "input range too large for lossless mode");
    let q: Vec<i32> = c.iter().map(|&v| v as i32).collect();
    code_coefs(&p, &q, &ModelChoice::Hand, &mut w);
    w.buf
}

/// Decode a lossless stream from [`encode_i32_lossless`].
pub fn decode_i32_lossless(bytes: &[u8]) -> Result<(Vec<usize>, Vec<i32>), DecodeError> {
    let mut r = ByteReader::new(bytes);
    let (_, dims, lossless) = read_header(&mut r)?;
    if !lossless {
        return Err(DecodeError::Invalid("not a lossless stream"));
    }
    let p = Pyramid::new(as3(&dims), max_levels(&dims));
    let q = decode_coefs(&p, &mut r)?;
    let mut c: Vec<i64> = q.iter().map(|&v| v as i64).collect();
    inverse_53(&p, &mut c);
    Ok((dims, c.iter().map(|&v| v as i32).collect()))
}

fn read_header(r: &mut ByteReader) -> Result<(Dtype, Vec<usize>, bool), DecodeError> {
    if r.take(4)? != MAGIC {
        return Err(DecodeError::Invalid("magic"));
    }
    let dt = match r.u8()? {
        0 => Dtype::F32,
        1 => Dtype::F64,
        _ => return Err(DecodeError::Invalid("dtype")),
    };
    let nd = r.varint()? as usize;
    if nd == 0 || nd > 3 {
        return Err(DecodeError::Invalid("ndims"));
    }
    let mut dims = Vec::with_capacity(nd);
    for _ in 0..nd {
        dims.push(r.varint()? as usize);
    }
    if dims.contains(&0) || dims.iter().try_fold(1usize, |a, &d| a.checked_mul(d)).is_none_or(|n| n > 1 << 34) {
        return Err(DecodeError::Invalid("dims"));
    }
    let lossless = r.u8()? != 0;
    Ok((dt, dims, lossless))
}

/// Decode a lossy wavelet stream.
pub fn decode(bytes: &[u8]) -> Result<Decoded, DecodeError> {
    let mut r = ByteReader::new(bytes);
    let (dt, dims, lossless) = read_header(&mut r)?;
    if lossless {
        return Err(DecodeError::Invalid("lossless stream: use decode_i32_lossless"));
    }
    let p = Pyramid::new(as3(&dims), max_levels(&dims));
    let delta = r.f64()?;
    let mut d = [0u8; NDELTA];
    d.copy_from_slice(r.take(NDELTA)?);
    let q = decode_coefs(&p, &mut r)?;
    let mut y = cast(&reconstruct(&p, &q, delta, &d), dt);
    if r.u8()? == 1 {
        let b = r.f64()?;
        {
            let buf = r.block()?;
            let mut k = vec![0i32; p.len()];
            let mut dd = Decoder::new(buf);
            code_outliers(p.dims, &mut k, &mut dd);
            apply_outliers(&mut y, &k, b);
            y = cast(&y, dt);
            let n = r.varint()? as usize;
            let mut i = 0usize;
            for _ in 0..n {
                i += r.varint()? as usize;
                if i >= y.len() {
                    return Err(DecodeError::Invalid("escape index"));
                }
                y[i] = match dt {
                    Dtype::F32 => f32::from_le_bytes(r.take(4)?.try_into().unwrap()) as f64,
                    Dtype::F64 => r.f64()?,
                };
            }
        }
    }
    Ok(match dt {
        Dtype::F32 => Decoded::F32 { dims, data: y.iter().map(|&v| v as f32).collect() },
        Dtype::F64 => Decoded::F64 { dims, data: y },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(d: [usize; 3]) -> Vec<f32> {
        let mut v = Vec::new();
        for i in 0..d[0] {
            for j in 0..d[1] {
                for k in 0..d[2] {
                    let (x, y, z) = (i as f64 / d[0] as f64, j as f64 / d[1] as f64, k as f64 / d[2] as f64);
                    v.push(((6.0 * x + 3.0 * y).sin() * (4.0 * z).cos() + 0.3 * (20.0 * x * y).sin() + if z > 0.6 { 0.5 } else { 0.0 }) as f32);
                }
            }
        }
        v
    }

    #[test]
    fn psnr_target_met_and_deterministic() {
        let d = [20, 33, 47];
        let x = field(d);
        for db in [40.0, 70.0, 100.0] {
            let cfg = WavConfig::new(ErrorBound::Psnr(db));
            let (bytes, st) = encode_f32(&x, &d, &cfg);
            let Decoded::F32 { data, .. } = decode(&bytes).unwrap() else { panic!() };
            let xd: Vec<f64> = x.iter().map(|&v| v as f64).collect();
            let yd: Vec<f64> = data.iter().map(|&v| v as f64).collect();
            let (rmse, _) = err_stats(&xd, &yd);
            assert_eq!(rmse, st.rmse, "decoder matches encoder self-check");
            assert!(20.0 * (st.range / rmse).log10() >= db - 1e-9);
            assert_eq!(encode_f32(&x, &d, &cfg).0, bytes);
        }
    }

    #[test]
    fn maxabs_bound_holds() {
        let d = [16, 24, 40];
        let x = field(d);
        let range = 2.6;
        for rel in [1e-2, 1e-4] {
            let b = rel * range;
            let (bytes, st) = encode_f32(&x, &d, &WavConfig::new(ErrorBound::MaxAbs(b)));
            let Decoded::F32 { data, .. } = decode(&bytes).unwrap() else { panic!() };
            let mx = x.iter().zip(&data).map(|(&a, &v)| (a as f64 - v as f64).abs()).fold(0.0, f64::max);
            assert!(mx <= b, "{mx} > {b}");
            assert_eq!(mx, st.max_abs);
        }
    }

    #[test]
    fn one_d_and_f64() {
        let x: Vec<f64> = (0..5000).map(|i| (i as f64 * 0.01).sin() + (i as f64 * 0.37).cos() * 0.1).collect();
        let (bytes, st) = encode_f64(&x, &[5000], &WavConfig::new(ErrorBound::Psnr(80.0)));
        let Decoded::F64 { data, .. } = decode(&bytes).unwrap() else { panic!() };
        assert_eq!(err_stats(&x, &data).0, st.rmse);
    }

    #[test]
    fn lossless_round_trip() {
        let d = [9, 17, 30];
        let x: Vec<i32> = field(d).iter().enumerate().map(|(i, &v)| (v * 30000.0) as i32 + (i % 5) as i32).collect();
        let bytes = encode_i32_lossless(&x, &d);
        let (dd, y) = decode_i32_lossless(&bytes).unwrap();
        assert_eq!(dd, d.to_vec());
        assert_eq!(y, x);
        assert!(bytes.len() < x.len() * 4);
    }
}
