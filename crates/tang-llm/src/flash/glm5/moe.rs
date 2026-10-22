//! MoE: sigmoid router + shared expert (W5, branch frog/glm5-moe).
//!
//! Router (`Glm5NextTextTopkRouter`, `topk_method` = noaux_tc): scores = `sigmoid(router(x))`
//! in f32; selection by `scores + e_score_correction_bias` (`exp_probs_b.bias`; bias only
//! affects selection, weights come from the unbiased scores); group-limited top-k if
//! `n_group > 1` (GLM-5.3-Flash: `n_group` = 1, so plain top-8 of 288); weights renormalised
//! (`norm_topk_prob`, denominator `+ 1e-20`) and scaled by `routed_scaling_factor` 2.5.
//! Every MoE layer *always* runs the `n_expert_shared` = 1 shared expert (no gate) and adds it.
//! swiglu with clamp ±`swiglu_limit` 10 (`gate.clamp(max=10)`, `up.clamp(-10, 10)`) — the
//! same clamp in the dense lead-in FFN (W1 implements that one).
//!
//! HF reference: `Glm5NextTextMoE`, `Glm5NextTextTopkRouter`, `Glm5NextTextExperts`.
//! GGUF names in [`super::gguf`].

/// Placeholder until W5's implementation lands.
pub fn moe_stub() {}

#[cfg(test)]
mod tests {
    #[test]
    fn placeholder() {}
}
