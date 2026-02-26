//! Optimization algorithms — Adam, L-BFGS, Newton, Levenberg-Marquardt.

#![no_std]

extern crate alloc;

mod adam;
mod lbfgs;
mod line_search;
mod lm;
mod newton;
mod sgd;

pub use adam::{Adam, AdamW};
pub use lbfgs::Lbfgs;
pub use line_search::armijo;
pub use lm::LevenbergMarquardt;
pub use newton::Newton;
pub use sgd::Sgd;
