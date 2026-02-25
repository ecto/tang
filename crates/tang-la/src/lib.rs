//! Dynamic linear algebra — DVec, DMat, decompositions.
//!
//! Generic over `tang::Scalar`, with optional `faer` bridge for
//! high-performance f64/f32 decompositions.

#![no_std]

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

mod dvec;
mod dmat;
mod lu;
mod svd;
mod cholesky;
mod qr;
mod eigen;

pub use dvec::DVec;
pub use dmat::DMat;
pub use lu::Lu;
pub use svd::Svd;
pub use cholesky::Cholesky;
pub use qr::Qr;
pub use eigen::SymmetricEigen;
