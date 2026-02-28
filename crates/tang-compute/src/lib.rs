//! tang-compute — Native GPU compute backends via tang-expr multi-target codegen.
//!
//! Provides a unified `ComputeDevice` trait with backends for:
//! - CPU (always available, with Accelerate BLAS on macOS)
//! - Metal (macOS/iOS, with simdgroup_matrix acceleration)
//! - CUDA (NVIDIA GPUs, with wmma/tensor core acceleration)

pub mod cpu;
pub mod device;
pub mod kernels;
pub mod modules;
pub mod ops;
pub mod tensor;

#[cfg(feature = "metal")]
pub mod metal;

#[cfg(feature = "cuda")]
pub mod cuda;

pub use cpu::CpuDevice;
pub use device::{ComputeBuffer, ComputeDevice};
pub use tang_expr::codegen::Dialect;

pub use modules::{Embedding, InterleavedRoPE, KVCache, Linear, RMSNorm};
pub use ops::{add_tensors, bias_add, swiglu_fused};
pub use tensor::ComputeTensor;

#[cfg(feature = "metal")]
pub use metal::MetalDevice;

#[cfg(feature = "cuda")]
pub use cuda::CudaComputeDevice;
