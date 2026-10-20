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
pub mod coefcode;
pub mod codec;
#[cfg(feature = "gpu")]
pub mod gpudec;
pub mod gpuwav;
pub mod gpuwav3;
pub mod learned;
pub mod linalg;
pub mod rans;
pub mod tensorize;
pub mod tt;
pub mod tucker;
pub mod wavcodec;
pub mod wavelet;

pub use codec::{decode, encode_f32, encode_f64, Config, Decoded, Dtype, EncodeStats, ErrorBound};
pub use tensorize::{DigitOrder, Layout, PadPolicy};
pub use tt::{tt_svd, TensorTrain, Truncation};

/// Decode any tang-compress stream (QTT, wavelet or Tucker), by magic.
pub fn decode_any(bytes: &[u8]) -> Result<Decoded, bitio::DecodeError> {
    if let Some(&t) = bytes.first() {
        if t == 0xA0 || t == 0xA1 {
            return decode_any(&bytes[1..]);
        }
    }
    match bytes.get(..4) {
        Some(m) if m == wavcodec::MAGIC => wavcodec::decode(bytes),
        Some(m) if m == tucker::MAGIC => tucker::decode(bytes),
        _ => codec::decode(bytes),
    }
}

/// Automatic transform choice: encode with the wavelet codec (with `cfg`'s
/// coefficient model) and, for PSNR/RMSE targets on 2D/3D data, with the
/// Tucker codec; keep the smaller stream. The output is one tag byte
/// (`0xA0` wavelet, `0xA1` Tucker — counted in the rate) followed by the
/// chosen stream. Returns `(bytes, chose_tucker)`.
pub fn encode_auto_f32(x: &[f32], dims: &[usize], cfg: &wavcodec::WavConfig) -> (Vec<u8>, bool) {
    let (w, _) = wavcodec::encode_f32(x, dims, cfg);
    let tucker_ok = !matches!(cfg.bound, codec::ErrorBound::MaxAbs(_)) && dims.iter().filter(|&&d| d > 1).count() >= 2;
    let t = tucker_ok.then(|| tucker::encode_f32(x, dims, cfg.bound).0);
    let (tag, body) = match t {
        Some(t) if t.len() < w.len() => (0xA1u8, t),
        _ => (0xA0u8, w),
    };
    let mut out = Vec::with_capacity(body.len() + 1);
    out.push(tag);
    out.extend(body);
    (out, tag == 0xA1)
}

#[cfg(test)]
mod auto_tests {
    use super::*;

    #[test]
    fn auto_round_trips_and_keeps_the_smaller() {
        let d = [16, 24, 20];
        let x: Vec<f32> = (0..d.iter().product::<usize>()).map(|i| ((i % 20) as f32 * 0.3).sin() + ((i / 480) as f32 * 0.2).cos()).collect();
        for b in [codec::ErrorBound::Psnr(70.0), codec::ErrorBound::MaxAbs(1e-3)] {
            let cfg = wavcodec::WavConfig::new(b);
            let (bytes, _) = encode_auto_f32(&x, &d, &cfg);
            let w = wavcodec::encode_f32(&x, &d, &cfg).0.len();
            assert!(bytes.len() <= w + 1);
            let Decoded::F32 { data, .. } = decode_any(&bytes).unwrap() else { panic!() };
            assert_eq!(data.len(), x.len());
            if let codec::ErrorBound::MaxAbs(e) = b {
                assert!(x.iter().zip(&data).all(|(a, c)| ((a - c) as f64).abs() <= e));
            }
        }
    }
}
