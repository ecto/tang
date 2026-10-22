//! GLM-5.3-Flash (`glm5-next`): per-component modules, one file per worker.
//! W1 owns this mod.rs and gguf.rs/forward.rs; mhc.rs = W2, mla.rs = W3, kda.rs = W4, moe.rs = W5.
pub mod moe;
