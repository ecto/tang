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

pub mod bitio;
pub mod codec;
pub mod linalg;
pub mod rans;
pub mod tensorize;
pub mod tt;

pub use codec::{decode, encode_f32, encode_f64, Config, Decoded, Dtype, EncodeStats, ErrorBound};
pub use tensorize::{DigitOrder, Layout, PadPolicy};
pub use tt::{tt_svd, TensorTrain, Truncation};
