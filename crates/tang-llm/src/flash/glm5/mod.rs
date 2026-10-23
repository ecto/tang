//! GLM-5.3-Flash (GGUF architecture `glm5-next`): the W0–W5 port. One file per owner, the
//! loaders and the layer loop in `forward` (W1), each component in its own module with a
//! CPU f32 reference checked against Hugging Face fixtures, CUDA kernel sources and
//! cuda-gated launchers.
//!
//! Owner map (see the port brief): `mod`/`gguf`/`forward` = W1, `mhc` = W2, `mla` = W3,
//! `kda` = W4, `moe` = W5.

pub mod mhc;

/// GLM-5.3-Flash's mHC defaults from the config (`Glm5NextTextConfig`): 4 streams, Sinkhorn
/// 20 iterations, eps 1e-6 (both the comb eps added after the softmax and the Sinkhorn
/// denominator eps), RMSNorm eps 1e-5.
pub const HC_STREAMS: usize = 4;
pub const SINKHORN_ITERS: u32 = 20;
pub const HC_EPS: f32 = 1e-6;
