//! LLM inference on tang-compute.
//!
//! Loads Qwen3-family safetensors checkpoints straight from a Hugging Face snapshot directory
//! and runs them on any [`ComputeDevice`](tang_compute::ComputeDevice) (Metal on Apple Silicon),
//! with a device-resident KV cache.

pub mod config;
pub mod model;
pub mod weights;

pub use config::Config;
pub use model::{Cache, Model};
