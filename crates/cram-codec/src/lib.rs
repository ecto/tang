//! Error-bounded compression of scientific arrays with a **bit-accounted
//! quantized tensor train (QTT)** codec.
//!
//! A grid of `N` samples is tensorized into a `d`-way tensor with mode sizes
//! of mostly 2 ([`tensorize`]), decomposed by TT-SVD in the canonical
//! left-orthogonal gauge ([`tt`]), and coded by [`codec`]: per-column
//! equal-slope quantization *with error feedback* (each core is refit against
//! the already-quantized prefix), ranks picked by rate–distortion rather than
//! a separate tolerance, and static rANS over transmitted frequency tables
//! ([`rans`]). The stream is self-describing and every byte counts toward
//! the reported rate — header, shape, ranks, step exponents, tables, payload,
//! and (in max-abs mode) the sparse residual-correction stream.
//!
//! ```
//! use cram_codec::{decode, encode_f32, Config, Decoded, ErrorBound};
//! let x: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.01).sin()).collect();
//! let (bytes, stats) = encode_f32(&x, &[64, 64], &Config::new(ErrorBound::MaxAbs(1e-3)));
//! let Decoded::F32 { data, .. } = decode(&bytes).unwrap() else { unreachable!() };
//! assert!(x.iter().zip(&data).all(|(a, b)| (a - b).abs() <= 1e-3));
//! assert_eq!(stats.bytes, bytes.len());
//! ```
//!
//! Other codecs — wavelet ([`wavcodec`]), Tucker ([`tucker`]),
//! interpolation ([`interp`]) and the brick-parallel GPU layouts
//! ([`gpuwav`], [`gpuwav3`]) — share the [`ErrorBound`] vocabulary, and
//! [`decode_any`] decodes any of them by magic.
//!
//! # Features and targets
//!
//! - Default features are empty: CPU encode and decode only.
//! - `gpu` adds `gpudec::GpuDecoder`, a wgpu decoder for tc-gpu / TCG3
//!   streams whose output is bit-identical to the CPU decoder.
//! - The crate builds for `wasm32-unknown-unknown`; the wasm-bindgen browser
//!   front end is the (unpublished) `cram-codec-wasm` crate in the tang
//!   repository.
//!
//! # Numerical notes
//!
//! Truncated SVDs use the Gram matrix of the smaller unfolding side with
//! `tang_la::SymmetricEigen` and `tang_la` GEMMs (row-major, NEON on
//! aarch64), rather than `tang_la::Svd` (one-sided Jacobi), which is far too
//! slow on `2 × 2^23` unfoldings. Squaring the spectrum resolves singular
//! values down to roughly `1e-7·σ₁` in f64 — finer than any step the codec
//! uses at PSNR ≤ ~120 dB. Encoding and decoding are single threaded and
//! deterministic: the same input yields the same bytes, and decoding the same
//! bytes yields the same array, run to run.

#![warn(missing_docs)]
// Dense index loops read more clearly than iterator chains here; tang CI
// allows this lint workspace-wide too.
#![allow(clippy::needless_range_loop)]

pub mod abac;
pub mod bitio;
pub mod codec;
pub mod coefcode;
#[cfg(all(feature = "container", not(target_arch = "wasm32")))]
pub mod container;
#[cfg(feature = "gpu")]
pub mod gpudec;
pub mod gpuwav;
pub mod gpuwav3;
pub mod img;
#[cfg(feature = "gpu")]
pub mod imggpu;
pub mod interp;
pub mod irdec;
pub mod kvcodec;
pub mod kvgpu;
pub mod learned;
pub mod linalg;
pub mod lmodel3;
pub mod lorenzo;
pub mod rans;
pub mod shipdec;
pub mod tensorize;
pub mod tt;
pub mod tucker;
pub mod wavcodec;
pub mod wavelet;

pub use codec::{decode, encode_f32, encode_f64, Config, Decoded, Dtype, EncodeStats, ErrorBound};
pub use tensorize::{DigitOrder, Layout, PadPolicy};
pub use tt::{tt_svd, TensorTrain, Truncation};

/// Decode any cram-codec stream (QTT, wavelet, Tucker, interpolation or Lorenzo),
/// by magic; a leading portfolio tag byte is skipped.
pub fn decode_any(bytes: &[u8]) -> Result<Decoded, bitio::DecodeError> {
    if let Some(&t) = bytes.first() {
        if (TAG_WAVELET..=TAG_LORENZO).contains(&t) {
            return decode_any(&bytes[1..]);
        }
    }
    match bytes.get(..4) {
        Some(m) if m == wavcodec::MAGIC => wavcodec::decode(bytes),
        Some(m) if m == tucker::MAGIC => tucker::decode(bytes),
        Some(m) if m == interp::MAGIC => interp::decode(bytes),
        Some(m) if m == lorenzo::MAGIC => lorenzo::decode(bytes),
        _ => codec::decode(bytes),
    }
}

/// Portfolio tag: wavelet stream follows.
pub const TAG_WAVELET: u8 = 0xA0;
/// Portfolio tag: Tucker stream follows.
pub const TAG_TUCKER: u8 = 0xA1;
/// Portfolio tag: QTT stream follows.
pub const TAG_QTT: u8 = 0xA2;
/// Portfolio tag: interpolation stream follows.
pub const TAG_INTERP: u8 = 0xA3;
/// Portfolio tag: Lorenzo stream follows.
pub const TAG_LORENZO: u8 = 0xA4;

/// A portfolio candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Candidate {
    /// [`wavcodec`] with the configured coefficient model.
    Wavelet,
    /// [`tucker`] (PSNR/RMSE targets, 2D/3D).
    Tucker,
    /// The QTT [`codec`] in row-major digit order.
    Qtt,
    /// [`interp`].
    Interp,
    /// The QTT codec's residual-only stream (max-abs; QTT tag and format).
    QttSparse,
    /// [`lorenzo`].
    Lorenzo,
}

impl Candidate {
    /// Stream tag.
    pub fn tag(self) -> u8 {
        match self {
            Candidate::Wavelet => TAG_WAVELET,
            Candidate::Tucker => TAG_TUCKER,
            Candidate::Qtt => TAG_QTT,
            Candidate::Interp => TAG_INTERP,
            Candidate::QttSparse => TAG_QTT,
            Candidate::Lorenzo => TAG_LORENZO,
        }
    }
    /// Short name.
    pub fn name(self) -> &'static str {
        match self {
            Candidate::Wavelet => "wavelet",
            Candidate::Tucker => "tucker",
            Candidate::Qtt => "qtt",
            Candidate::Interp => "interp",
            Candidate::QttSparse => "qtt-sparse",
            Candidate::Lorenzo => "lorenzo",
        }
    }
}

/// What the portfolio encoder did.
#[derive(Clone, Debug)]
pub struct AutoReport {
    /// The candidate whose stream was kept.
    pub pick: Candidate,
    /// Every candidate encoded: `(candidate, stream bytes, encode seconds)`.
    pub tried: Vec<(Candidate, usize, f64)>,
    /// Candidates the pre-screen skipped.
    pub skipped: Vec<Candidate>,
    /// Pre-screen QTT probe max rank (see [`prescreen`]).
    pub probe_rank: usize,
}

/// Every-`stride`-th sample along each axis longer than `4·stride`
/// (row-major), with the new shape.
pub fn decimate(x: &[f32], dims: &[usize], stride: usize) -> (Vec<f32>, Vec<usize>) {
    let st: Vec<usize> = dims
        .iter()
        .map(|&d| if d >= 4 * stride { stride } else { 1 })
        .collect();
    let nd: Vec<usize> = dims.iter().zip(&st).map(|(&d, &s)| d.div_ceil(s)).collect();
    let n: usize = nd.iter().product();
    let mut out = Vec::with_capacity(n);
    let mut idx = vec![0usize; dims.len()];
    for _ in 0..n {
        let mut off = 0;
        for (k, &i) in idx.iter().enumerate() {
            off = off * dims[k] + i * st[k];
        }
        out.push(x[off]);
        for k in (0..dims.len()).rev() {
            idx[k] += 1;
            if idx[k] < nd[k] {
                break;
            }
            idx[k] = 0;
        }
    }
    (out, nd)
}

/// QTT rate probe: row-major QTT on a `stride`-decimated copy at the same
/// absolute bound (PSNR is turned into the full field's RMSE), ranks capped
/// at `cap`. Returns `(probe bytes, max rank, seconds)`.
pub fn qtt_probe(
    x: &[f32],
    dims: &[usize],
    bound: codec::ErrorBound,
    stride: usize,
    cap: usize,
) -> (usize, usize, f64) {
    let t = std::time::Instant::now();
    let b = match bound {
        codec::ErrorBound::Psnr(p) => {
            let (lo, hi) = x
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| {
                    (a.min(v), b.max(v))
                });
            codec::ErrorBound::Rmse((hi - lo) as f64 / 10f64.powf(p / 20.0))
        }
        b => b,
    };
    let (y, nd) = decimate(x, dims, stride);
    let mut q = codec::Config::new(b);
    q.order = DigitOrder::RowMajor;
    q.max_rank = cap;
    let (s, st) = codec::encode_f32(&y, &nd, &q);
    let r = st.ranks.iter().copied().max().unwrap_or(0);
    (s.len(), r, t.elapsed().as_secs_f64())
}

/// QTT probe rank cap ([`qtt_probe`] with stride 2): full QTT runs only
/// when the probe's ranks stay below it.
pub const QTT_PROBE_CAP: usize = 64;

/// Which candidates the pre-screen keeps for a bound, before any full encode.
///
/// - wavelet: always (coded with `compact`: the configured model, the hand
///   model or the tiled hand model, whichever is smallest);
/// - Tucker: PSNR/RMSE targets on ≥2D data;
/// - interpolation: always (max-abs directly; RMSE/PSNR through its
///   rate-controlled bound, about half a wavelet encode);
/// - Lorenzo: always (one encode per prediction mode at max-abs, then rate
///   control of one mode at RMSE/PSNR; cheaper than interpolation);
/// - QTT residual-only: max-abs targets (a few % of a wavelet encode);
/// - full QTT: only when [`qtt_probe`] (stride 2, ranks capped at
///   [`QTT_PROBE_CAP`]) keeps every rank below the cap — the field is
///   genuinely low-TT-rank, so QTT is both cheap and possibly smallest. At
///   max-abs a probe rank ≤ 1 means the TT part was dropped (the residual
///   stream alone won), which the residual-only candidate already covers.
///
/// Returns `(kept, skipped, probe max rank)`.
pub fn prescreen(
    x: &[f32],
    dims: &[usize],
    bound: codec::ErrorBound,
) -> (Vec<Candidate>, Vec<Candidate>, usize) {
    let maxabs = matches!(bound, codec::ErrorBound::MaxAbs(_));
    let multi = dims.iter().filter(|&&d| d > 1).count() >= 2;
    let (_, rank, _) = qtt_probe(x, dims, bound, 2, QTT_PROBE_CAP);
    let mut keep = vec![Candidate::Wavelet];
    let mut skip = vec![];
    if !maxabs && multi {
        keep.push(Candidate::Tucker);
    }
    keep.push(Candidate::Interp);
    keep.push(Candidate::Lorenzo);
    if maxabs {
        keep.push(Candidate::QttSparse);
    }
    if rank < QTT_PROBE_CAP && (!maxabs || rank > 1) {
        keep.push(Candidate::Qtt);
    } else {
        skip.push(Candidate::Qtt);
    }
    (keep, skip, rank)
}

/// Encode with one candidate (no tag byte).
pub fn encode_candidate_f32(
    c: Candidate,
    x: &[f32],
    dims: &[usize],
    cfg: &wavcodec::WavConfig,
) -> Vec<u8> {
    match c {
        Candidate::Wavelet => {
            let cfg = wavcodec::WavConfig {
                compact: true,
                ..cfg.clone()
            };
            wavcodec::encode_f32(x, dims, &cfg).0
        }
        Candidate::Tucker => tucker::encode_f32(x, dims, cfg.bound).0,
        Candidate::Qtt => {
            let mut q = codec::Config::new(cfg.bound);
            q.order = DigitOrder::RowMajor;
            codec::encode_f32(x, dims, &q).0
        }
        Candidate::Interp => match cfg.bound {
            codec::ErrorBound::MaxAbs(b) => interp::encode_f32(x, dims, b),
            codec::ErrorBound::Rmse(r) => interp::encode_rmse_f32(x, dims, r),
            codec::ErrorBound::Psnr(p) => interp::encode_psnr_f32(x, dims, p),
        },
        Candidate::QttSparse => {
            let mut q = codec::Config::new(cfg.bound);
            q.order = DigitOrder::RowMajor;
            q.residual_only = true;
            codec::encode_f32(x, dims, &q).0
        }
        Candidate::Lorenzo => match cfg.bound {
            codec::ErrorBound::MaxAbs(b) => lorenzo::encode_f32(x, dims, b),
            codec::ErrorBound::Rmse(r) => lorenzo::encode_rmse_f32(x, dims, r),
            codec::ErrorBound::Psnr(p) => lorenzo::encode_psnr_f32(x, dims, p),
        },
    }
}

/// Portfolio encode: encode with every candidate the pre-screen keeps (all
/// applicable ones when `screen` is false) and keep the smallest stream. The
/// output is one tag byte (counted in the rate) followed by the stream.
pub fn encode_portfolio_f32(
    x: &[f32],
    dims: &[usize],
    cfg: &wavcodec::WavConfig,
    screen: bool,
) -> (Vec<u8>, AutoReport) {
    let (mut keep, mut skip, probe_rank) = prescreen(x, dims, cfg.bound);
    if !screen {
        keep.append(&mut skip);
    }
    let mut best: Option<(Candidate, Vec<u8>)> = None;
    let mut tried = vec![];
    for c in keep {
        let t = std::time::Instant::now();
        // a candidate that fails on this input (e.g. a numerically singular
        // Tucker factor) is skipped rather than aborting the encode; wavelet
        // runs first and always codes
        let Ok(s) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            encode_candidate_f32(c, x, dims, cfg)
        })) else {
            skip.push(c);
            continue;
        };
        tried.push((c, s.len(), t.elapsed().as_secs_f64()));
        if best.as_ref().is_none_or(|(_, b)| s.len() < b.len()) {
            best = Some((c, s));
        }
    }
    let (pick, body) = best.unwrap();
    let mut out = Vec::with_capacity(body.len() + 1);
    out.push(pick.tag());
    out.extend(body);
    (
        out,
        AutoReport {
            pick,
            tried,
            skipped: skip,
            probe_rank,
        },
    )
}

/// Centered crop with every axis of length ≥ 16 halved (1D: the middle
/// eighth), and its shape.
pub fn center_crop(x: &[f32], dims: &[usize]) -> (Vec<f32>, Vec<usize>) {
    let f = if dims.iter().filter(|&&d| d > 1).count() == 1 {
        8
    } else {
        2
    };
    let nd: Vec<usize> = dims
        .iter()
        .map(|&d| if d >= 16 { d / f } else { d })
        .collect();
    let lo: Vec<usize> = dims.iter().zip(&nd).map(|(&d, &n)| (d - n) / 2).collect();
    let n: usize = nd.iter().product();
    let mut out = Vec::with_capacity(n);
    let mut idx = vec![0usize; dims.len()];
    for _ in 0..n {
        let mut off = 0;
        for k in 0..dims.len() {
            off = off * dims[k] + lo[k] + idx[k];
        }
        out.push(x[off]);
        for k in (0..dims.len()).rev() {
            idx[k] += 1;
            if idx[k] < nd[k] {
                break;
            }
            idx[k] = 0;
        }
    }
    (out, nd)
}

/// Fast mode's single candidate: each fast variant that applies (max-abs:
/// wavelet, interpolation, Lorenzo, QTT residual-only; PSNR/RMSE: wavelet, plus
/// Tucker on 3D) is trial-encoded on [`center_crop`] (an eighth of the
/// field) at the same absolute bound, and the smallest stream names the pick.
pub fn fast_pick(x: &[f32], dims: &[usize], cfg: &wavcodec::WavConfig) -> Candidate {
    let bound = match cfg.bound {
        codec::ErrorBound::Psnr(p) => {
            let (lo, hi) = x
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| {
                    (a.min(v), b.max(v))
                });
            codec::ErrorBound::Rmse((hi - lo) as f64 / 10f64.powf(p / 20.0))
        }
        b => b,
    };
    let three = dims.iter().filter(|&&d| d > 1).count() == 3;
    let cands: Vec<Candidate> = match bound {
        codec::ErrorBound::MaxAbs(_) => {
            vec![
                Candidate::Wavelet,
                Candidate::Interp,
                Candidate::Lorenzo,
                Candidate::QttSparse,
            ]
        }
        _ if three => vec![Candidate::Wavelet, Candidate::Tucker],
        _ => return Candidate::Wavelet,
    };
    let (y, nd) = center_crop(x, dims);
    let c = wavcodec::WavConfig {
        bound,
        ..cfg.clone()
    };
    cands
        .into_iter()
        .min_by_key(|&k| try_fast(k, &y, &nd, &c).map_or(usize::MAX, |s| s.len()))
        .unwrap()
}

/// [`encode_candidate_fast_f32`], or `None` when the candidate fails on this
/// input (e.g. a numerically singular Tucker factor).
fn try_fast(c: Candidate, x: &[f32], dims: &[usize], cfg: &wavcodec::WavConfig) -> Option<Vec<u8>> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        encode_candidate_fast_f32(c, x, dims, cfg)
    }))
    .ok()
}

/// Encode with one candidate's fast variant (no tag byte): step from a
/// formula, one interpolation preset, the configured (pretrained or hand)
/// model and never an in-stream overfit. QTT has no fast variant.
pub fn encode_candidate_fast_f32(
    c: Candidate,
    x: &[f32],
    dims: &[usize],
    cfg: &wavcodec::WavConfig,
) -> Vec<u8> {
    match c {
        Candidate::Wavelet => {
            let cfg = wavcodec::WavConfig {
                fast: true,
                compact: true,
                ..cfg.clone()
            };
            wavcodec::encode_f32(x, dims, &cfg).0
        }
        Candidate::Tucker => tucker::encode_fast_f32(x, dims, cfg.bound).0,
        Candidate::Interp => match cfg.bound {
            codec::ErrorBound::MaxAbs(b) => interp::encode_fast_f32(x, dims, b),
            _ => panic!("interp codes max-abs bounds only"),
        },
        Candidate::Lorenzo => match cfg.bound {
            codec::ErrorBound::MaxAbs(b) => lorenzo::encode_f32(x, dims, b),
            _ => panic!("fast Lorenzo codes max-abs bounds only"),
        },
        Candidate::Qtt | Candidate::QttSparse => encode_candidate_f32(c, x, dims, cfg),
    }
}

/// Fast encode (`tc --fast`): the single candidate [`fast_pick`] names, coded
/// with [`encode_candidate_fast_f32`]. Same tagged stream format as
/// [`encode_auto_f32`].
pub fn encode_fast_f32(
    x: &[f32],
    dims: &[usize],
    cfg: &wavcodec::WavConfig,
) -> (Vec<u8>, Candidate) {
    let pick = match std::env::var("TC_FAST_PICK").ok().as_deref() {
        Some("wavelet") => Candidate::Wavelet,
        Some("tucker") => Candidate::Tucker,
        Some("interp") => Candidate::Interp,
        Some("lorenzo") => Candidate::Lorenzo,
        _ => fast_pick(x, dims, cfg),
    };
    // the wavelet always codes, so it backs up a pick that fails at full size
    let (pick, body) = match try_fast(pick, x, dims, cfg) {
        Some(b) => (pick, b),
        None => (
            Candidate::Wavelet,
            encode_candidate_fast_f32(Candidate::Wavelet, x, dims, cfg),
        ),
    };
    let mut out = Vec::with_capacity(body.len() + 1);
    out.push(pick.tag());
    out.extend(body);
    (out, pick)
}

/// Automatic codec choice: [`encode_portfolio_f32`] with the pre-screen on.
/// Returns the tagged stream and the chosen candidate.
pub fn encode_auto_f32(
    x: &[f32],
    dims: &[usize],
    cfg: &wavcodec::WavConfig,
) -> (Vec<u8>, Candidate) {
    let (b, r) = encode_portfolio_f32(x, dims, cfg, true);
    (b, r.pick)
}

#[cfg(test)]
mod auto_tests {
    use super::*;

    #[test]
    fn auto_round_trips_and_keeps_the_smaller() {
        let d = [16, 24, 20];
        let x: Vec<f32> = (0..d.iter().product::<usize>())
            .map(|i| ((i % 20) as f32 * 0.3).sin() + ((i / 480) as f32 * 0.2).cos())
            .collect();
        for b in [
            codec::ErrorBound::Psnr(70.0),
            codec::ErrorBound::MaxAbs(1e-3),
        ] {
            let cfg = wavcodec::WavConfig::new(b);
            let (bytes, _) = encode_auto_f32(&x, &d, &cfg);
            let w = wavcodec::encode_f32(&x, &d, &cfg).0.len();
            assert!(bytes.len() <= w + 1);
            let Decoded::F32 { data, .. } = decode_any(&bytes).unwrap() else {
                panic!()
            };
            assert_eq!(data.len(), x.len());
            if let codec::ErrorBound::MaxAbs(e) = b {
                assert!(x
                    .iter()
                    .zip(&data)
                    .all(|(a, c)| ((a - c) as f64).abs() <= e));
            }
        }
    }
}
