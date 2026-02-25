//! Computational graph, parameter management, training loop.

#![no_std]

extern crate alloc;

mod parameter;
mod module;
mod layers;
mod loss;

pub use parameter::Parameter;
pub use module::Module;
pub use layers::{Linear, ReLU, Sequential};
pub use loss::{mse_loss, huber_loss, softmax, cross_entropy_loss};
