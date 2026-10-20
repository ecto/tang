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
//! use tang_compress::{decode, encode_f32, Config, Decoded, ErrorBound};
//! let x: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.01).sin()).collect();
//! let (bytes, stats) = encode_f32(&x, &[64, 64], &Config::new(ErrorBound::MaxAbs(1e-3)));
//! let Decoded::F32 { data, .. } = decode(&bytes).unwrap() else { unreachable!() };
//! assert!(x.iter().zip(&data).all(|(a, b)| (a - b).abs() <= 1e-3));
//! assert_eq!(stats.bytes, bytes.len());
//! ```
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
#[cfg(feature = "gpu")]
pub mod gpudec;
pub mod gpuwav;
pub mod gpuwav3;
pub mod interp;
pub mod irdec;
pub mod kvcodec;
pub mod learned;
pub mod linalg;
pub mod lmodel3;
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

/// Decode any tang-compress stream (QTT, wavelet, Tucker or interpolation),
/// by magic; a leading portfolio tag byte is skipped.
pub fn decode_any(bytes: &[u8]) -> Result<Decoded, bitio::DecodeError> {
    if let Some(&t) = bytes.first() {
        if (TAG_WAVELET..=TAG_INTERP).contains(&t) {
            return decode_any(&bytes[1..]);
        }
    }
    match bytes.get(..4) {
        Some(m) if m == wavcodec::MAGIC => wavcodec::decode(bytes),
        Some(m) if m == tucker::MAGIC => tucker::decode(bytes),
        Some(m) if m == interp::MAGIC => interp::decode(bytes),
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

/// A portfolio candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Candidate {
    /// [`wavcodec`] with the configured coefficient model.
    Wavelet,
    /// [`tucker`] (PSNR/RMSE targets, 2D/3D).
    Tucker,
    /// The QTT [`codec`] in row-major digit order.
    Qtt,
    /// [`interp`] (max-abs targets).
    Interp,
}

impl Candidate {
    /// Stream tag.
    pub fn tag(self) -> u8 {
        match self {
            Candidate::Wavelet => TAG_WAVELET,
            Candidate::Tucker => TAG_TUCKER,
            Candidate::Qtt => TAG_QTT,
            Candidate::Interp => TAG_INTERP,
        }
    }
    /// Short name.
    pub fn name(self) -> &'static str {
        match self {
            Candidate::Wavelet => "wavelet",
            Candidate::Tucker => "tucker",
            Candidate::Qtt => "qtt",
            Candidate::Interp => "interp",
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
    /// Pre-screen peakiness statistic (see [`peakiness`]).
    pub peak: f64,
}

/// Pre-screen threshold on [`peakiness`] above which QTT is tried.
pub const QTT_PEAK: f64 = 8.0;

/// Peakiness of a field: `range / (2·IQR-ish spread)` measured as
/// `range / (q99 − q01)` over a strided sample of at most 2¹⁶ cells. Smooth
/// and turbulent fields sit near 1–2; fields whose range is set by a few
/// sharp, sparse features (detector peaks, orbital cusps) are far larger.
pub fn peakiness(x: &[f32]) -> f64 {
    let step = (x.len() / (1 << 16)).max(1);
    let mut s: Vec<f32> = x
        .iter()
        .step_by(step)
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    if s.len() < 2 {
        return 1.0;
    }
    let (lo, hi) = x
        .iter()
        .filter(|v| v.is_finite())
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| {
            (a.min(v), b.max(v))
        });
    s.sort_by(f32::total_cmp);
    let q = |f: f64| s[((s.len() - 1) as f64 * f) as usize] as f64;
    let spread = q(0.99) - q(0.01);
    let range = (hi - lo) as f64;
    if range <= 0.0 {
        1.0
    } else if spread <= 0.0 {
        f64::INFINITY
    } else {
        range / spread
    }
}

/// Which candidates the pre-screen keeps for a bound, before any encode.
///
/// - wavelet: always;
/// - Tucker: PSNR/RMSE targets on ≥2D data (as before);
/// - interpolation: max-abs targets (it costs a few % of a wavelet encode);
/// - QTT: only when [`peakiness`] ≥ [`QTT_PEAK`] (its encode costs several
///   wavelet encodes, and it only wins on peaky / sparse-feature data).
pub fn prescreen(
    x: &[f32],
    dims: &[usize],
    bound: codec::ErrorBound,
) -> (Vec<Candidate>, Vec<Candidate>, f64) {
    let maxabs = matches!(bound, codec::ErrorBound::MaxAbs(_));
    let multi = dims.iter().filter(|&&d| d > 1).count() >= 2;
    let peak = peakiness(x);
    let mut keep = vec![Candidate::Wavelet];
    let mut skip = vec![];
    if !maxabs && multi {
        keep.push(Candidate::Tucker);
    }
    if maxabs {
        keep.push(Candidate::Interp);
    }
    if peak >= QTT_PEAK {
        keep.push(Candidate::Qtt);
    } else {
        skip.push(Candidate::Qtt);
    }
    (keep, skip, peak)
}

/// Encode with one candidate (no tag byte).
pub fn encode_candidate_f32(
    c: Candidate,
    x: &[f32],
    dims: &[usize],
    cfg: &wavcodec::WavConfig,
) -> Vec<u8> {
    match c {
        Candidate::Wavelet => wavcodec::encode_f32(x, dims, cfg).0,
        Candidate::Tucker => tucker::encode_f32(x, dims, cfg.bound).0,
        Candidate::Qtt => {
            let mut q = codec::Config::new(cfg.bound);
            q.order = DigitOrder::RowMajor;
            codec::encode_f32(x, dims, &q).0
        }
        Candidate::Interp => match cfg.bound {
            codec::ErrorBound::MaxAbs(b) => interp::encode_f32(x, dims, b),
            _ => panic!("interp codes max-abs bounds only"),
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
    let (mut keep, mut skip, peak) = prescreen(x, dims, cfg.bound);
    if !screen {
        keep.append(&mut skip);
    }
    let mut best: Option<(Candidate, Vec<u8>)> = None;
    let mut tried = vec![];
    for c in keep {
        let t = std::time::Instant::now();
        let s = encode_candidate_f32(c, x, dims, cfg);
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
            peak,
        },
    )
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
