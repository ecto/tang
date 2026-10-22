//! KDA: Kimi delta attention (W4, branch frog/glm5-kda).
//!
//! The 34 KDA layers (`is_kda[l] == true`): per-head delta-rule state with a *per-channel*
//! (vector) decay gate — unlike tang's Qwen3-Next GDN whose gate is a scalar per head.
//! Head dim `kda.head_dim` 128 × 64 heads = `d_inner`; short conv `ssm.conv_kernel` = 4 with
//! **separate** q/k/v conv weights; `ssm_a` = `-exp(A_log)` per head; `ssm_dt.bias` per channel;
//! `ssm_f_a`/`ssm_f_b` produce the decay gate g = `lower_bound * sigmoid(exp(A_log) · g)` with
//! `lower_bound` = −5 (`kda.gate_lower_bound`), so the per-channel decay is
//! `exp(lower_bound · sigmoid(A · g))` ∈ (exp(-5), 1); beta = `sigmoid(ssm_beta · x)` per head;
//! output `RMSNorm(o) * sigmoid(ssm_g_b(ssm_g_a(x)))` (per channel), then `attn_output`.
//! Q/K get an FLA-style L2 norm (eps on the squared norm, 1e-6) before the recurrence.
//!
//! HF reference: `Glm5NextTextLinearAttention`, `Glm5NextTextForgetGate`,
//! `recurrent_kimi_delta_attention` / `chunk_kimi_delta_attention`. GGUF names in [`super::gguf`].

/// Placeholder until W4's implementation lands.
pub fn kda_stub() {}

#[cfg(test)]
mod tests {
    #[test]
    fn placeholder() {}
}
