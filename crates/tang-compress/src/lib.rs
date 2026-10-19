//! Error-bounded compression of scientific arrays with a **bit-accounted
//! quantized tensor train (QTT)** codec.
//!
//! Building blocks in this commit: tensorization, TT-SVD, rANS.
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
pub mod linalg;
pub mod rans;
pub mod tensorize;
pub mod tt;

pub use tensorize::{DigitOrder, Layout, PadPolicy};
pub use tt::{tt_svd, TensorTrain, Truncation};
