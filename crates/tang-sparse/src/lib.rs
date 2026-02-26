//! CSR/CSC sparse matrices and sparse solvers.

#![no_std]

extern crate alloc;

mod coo;
mod csc;
mod csr;

pub use coo::CooMatrix;
pub use csc::CscMatrix;
pub use csr::CsrMatrix;
