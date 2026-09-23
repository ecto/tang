//! LLM inference on tang-compute.
//!
//! Loads Qwen3-family safetensors checkpoints straight from a Hugging Face snapshot directory
//! and runs them on any [`ComputeDevice`](tang_compute::ComputeDevice) (Metal on Apple Silicon),
//! with a device-resident KV cache, the checkpoint's own chat template, and an
//! OpenAI-compatible server.

pub mod chat;
pub mod config;
pub mod engine;
pub mod model;
pub mod sample;
pub mod server;
pub mod weights;

pub use config::Config;
pub use engine::{Engine, Request};
pub use model::{Cache, Dtype, Model};

use std::path::PathBuf;

/// A model directory, or a Hugging Face repo id (`Qwen/Qwen3-4B`) already in the local HF cache.
pub fn resolve_model(spec: &str) -> anyhow::Result<PathBuf> {
    let p = PathBuf::from(spec);
    if p.join("config.json").exists() {
        return Ok(p);
    }
    let hub = std::env::var_os("HF_HUB_CACHE")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HF_HOME").map(|h| PathBuf::from(h).join("hub")))
        .or_else(|| {
            std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/huggingface/hub"))
        })
        .unwrap_or_default();
    let snaps = hub
        .join(format!("models--{}", spec.replace('/', "--")))
        .join("snapshots");
    let mut dirs: Vec<_> = std::fs::read_dir(&snaps)
        .map(|r| r.filter_map(|e| e.ok().map(|e| e.path())).collect())
        .unwrap_or_default();
    dirs.retain(|d: &PathBuf| d.join("config.json").exists());
    dirs.sort_by_key(|d| std::fs::metadata(d).and_then(|m| m.modified()).ok());
    dirs.pop().ok_or_else(|| {
        anyhow::anyhow!(
            "{spec}: not a model directory, and not in the Hugging Face cache ({})",
            snaps.display()
        )
    })
}
