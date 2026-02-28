//! Hand-optimized compute kernels for performance-critical operations.
//!
//! Each submodule contains kernel source code as `const &str` for a specific
//! dialect (MSL or CUDA). These are compiled at runtime and cached by the
//! respective backend.

#[cfg(feature = "metal")]
pub mod matmul_msl;

#[cfg(feature = "metal")]
pub mod reduce_msl;

#[cfg(feature = "metal")]
pub mod attention_msl;

#[cfg(feature = "cuda")]
pub mod matmul_cuda;

#[cfg(feature = "cuda")]
pub mod reduce_cuda;

#[cfg(feature = "cuda")]
pub mod attention_cuda;
