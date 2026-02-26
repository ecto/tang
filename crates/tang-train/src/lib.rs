//! Computational graph, parameter management, training loop.

#![no_std]

extern crate alloc;

mod layers;
mod loss;
mod module;
mod parameter;

pub use layers::{Linear, ReLU, Sequential};
pub use loss::{cross_entropy_loss, huber_loss, mse_loss, softmax};
pub use module::Module;
pub use parameter::Parameter;
