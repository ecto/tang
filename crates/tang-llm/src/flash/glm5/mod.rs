//! GLM-5.3-Flash ("frog-tree", GGUF arch `glm5_next`): 45 layers = 34 KDA (Kimi delta
//! attention, linear) + 11 DSA (MLA + pooled indexer), mHC 4-stream hyper-connections,
//! 3 dense + MoE (288 experts, top-8 + shared), 1 MTP layer. One file per owner of the
//! parallel port; W1 (`mod.rs`, `gguf.rs`, `forward.rs`) integrates.
//!
//! - [`kda`]: W4 — the KDA layers (this branch).
//! - `mhc`, `mla`, `moe`: W2/W3/W5, their own branches.

pub mod kda;
