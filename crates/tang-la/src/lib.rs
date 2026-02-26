//! Dynamic linear algebra — DVec, DMat, decompositions.
//!
//! Generic over `tang::Scalar`, with optional `faer` bridge for
//! high-performance f64/f32 decompositions.

#![no_std]

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

mod cholesky;
mod dmat;
mod dvec;
mod eigen;
mod lu;
mod qr;
mod svd;

pub use cholesky::Cholesky;
pub use dmat::DMat;
pub use dvec::DVec;
pub use eigen::SymmetricEigen;
pub use lu::Lu;
pub use qr::Qr;
pub use svd::Svd;
