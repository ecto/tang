//! Inference runtime for tang models.
//!
//! Provides the components needed for efficient autoregressive generation:
//!
//! - [`KVCache`] — per-layer key/value cache for O(n) incremental attention
//! - [`Sampler`] — configurable token sampling (greedy, top-k, top-p, temperature)
//! - [`generate`] — autoregressive generation loop
//!
//! # Example
//!
//! ```ignore
//! use tang_infer::{KVCache, Sampler, SamplingConfig};
//!
//! let mut cache = KVCache::new(num_layers, max_seq_len, num_kv_heads, head_dim);
//! let sampler = Sampler::new(SamplingConfig::default());
//!
//! // During generation, append new K/V and sample next token
//! cache.append(layer, new_k, new_v);
//! let (full_k, full_v) = cache.get(layer);
//! let next_token = sampler.sample(&logits);
//! ```

#![no_std]

extern crate alloc;

mod kv_cache;
mod sampling;

pub use kv_cache::KVCache;
pub use sampling::{generate, Sampler, SamplingConfig};
