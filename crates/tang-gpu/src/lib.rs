//! GPU compute via wgpu — SpMV, batch ops, tensor kernels.
//!
//! Enable the `wgpu` feature to get actual GPU device support.
//! Without it, this crate provides type stubs for compilation.

#![no_std]

extern crate alloc;

mod device;

pub use device::GpuDevice;
