//! Dynamic linear algebra — DVec, DMat, decompositions.
//!
//! Generic over `tang::Scalar`, with optional `faer` bridge for
//! high-performance f64/f32 decompositions.

#![no_std]

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

#[cfg(all(feature = "accelerate", target_os = "macos"))]
mod blas;

mod cholesky;
mod dmat;
mod dvec;
mod eigen;
pub mod gemm;
mod lu;
mod qr;
pub mod stats;
mod svd;

pub use cholesky::Cholesky;
pub use dmat::DMat;
pub use dvec::DVec;
pub use eigen::{branchless_jacobi_eigen, SymmetricEigen};
pub use gemm::{gemm_nn, gemm_nt, gemm_tn};
pub use lu::Lu;
pub use qr::Qr;
pub use stats::{
    central_moment, kurtosis, kurtosis_raw, mean, moments, skewness, stddev, stddev_sample,
    variance, variance_sample, Moments,
};
pub use svd::Svd;
