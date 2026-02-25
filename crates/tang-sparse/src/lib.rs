//! CSR/CSC sparse matrices and sparse solvers.

#![no_std]

extern crate alloc;

mod csr;
mod csc;
mod coo;

pub use csr::CsrMatrix;
pub use csc::CscMatrix;
pub use coo::CooMatrix;
