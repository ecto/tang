//! mHC: manifold-constrained hyper-connections (W2, branch frog/glm5-mhc).
//!
//! GLM-5.3-Flash carries every token on `hc` = 4 residual streams. Around each mixer/FFN block:
//!
//! - *pre*: a learned projection of the flattened streams gives per-stream weights
//!   `sigmoid(w·scale + base) + eps`; the block input is their weighted sum (collapse).
//! - *post*: per-stream expansion weights `2 · sigmoid(w·scale + base)` on the block output.
//! - *comb*: an `hc × hc` mixing matrix (softmax + eps, then Sinkhorn normalisation to
//!   doubly-stochastic, `sinkhorn_iterations` = 20, `eps` = 1e-6); the new streams are
//!   `post ⊙ block_out + comb^T @ residual`.
//! - After the last layer the streams collapse by unweighted mean (`Glm5NextTextHyperHead`).
//!
//! HF reference: `Glm5NextTextHyperConnection` in
//! `transformers/models/glm5_next/modeling_glm5_next.py`. GGUF tensors per layer:
//! `hc_attn_fn/hc_attn_base/hc_attn_scale` and `hc_ffn_*` (see [`super::gguf`]).
//!
//! # CPU reference (W2's contract)
//!
//! ```ignore
//! /// The weights one mHC block needs, dequantised. `fn` is [mix][hc*n], `base` [mix],
//! /// `scale` [3]; mix = (2 + hc) * hc.
//! pub struct HcWeights { pub f: Mat, pub base: Vec<f32>, pub scale: Vec<f32> }
//!
//! /// Compute (post, comb, collapsed) for every position.
//! /// `streams` is [t_len][hc][n]; returns post [t_len][hc], comb [t_len][hc][hc],
//! /// collapsed [t_len][n].
//! pub fn hc_pre(w: &HcWeights, streams: &[f32], t_len: usize, hp: &Hparams) -> Result<...>;
//!
//! /// New streams: `post[c] * block_out + Σ_src comb[src][dst] * residual[src]`.
//! pub fn hc_post(streams_new: &mut [f32], block_out: &[f32], residual: &[f32],
//!                post: &[f32], comb: &[f32], t_len: usize, hp: &Hparams);
//! ```
//!
//! Until W2 lands, [`super::forward`] calls these through the stubs below (todo!()).

/// Placeholder until W2's implementation lands; see the module docs for the contract.
pub fn hc_pre_stub() {}

#[cfg(test)]
mod tests {
    #[test]
    fn placeholder() {}
}
