//! N-dimensional arrays with CPU and GPU backends.

#![no_std]

extern crate alloc;

mod shape;
mod tensor;

pub use shape::Shape;
pub use tensor::Tensor;
