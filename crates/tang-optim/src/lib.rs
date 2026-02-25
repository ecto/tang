//! Optimization algorithms — Adam, L-BFGS, Newton, Levenberg-Marquardt.

#![no_std]

extern crate alloc;

mod adam;
mod sgd;
mod lbfgs;
mod newton;
mod lm;
mod line_search;

pub use adam::{Adam, AdamW};
pub use sgd::Sgd;
pub use lbfgs::Lbfgs;
pub use newton::Newton;
pub use lm::LevenbergMarquardt;
pub use line_search::armijo;
