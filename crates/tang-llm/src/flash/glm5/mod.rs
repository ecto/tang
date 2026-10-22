//! GLM-5.3-Flash component modules. One file per worker (see
//! `docs/models/port-audit-glm53-deepseekv41.md`); W1 owns `gguf.rs` / `forward.rs`
//! and merges.
pub mod mla; // W3: DSA layers' MLA-NoPE attention + k-pool indexer
