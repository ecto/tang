//! MLA-NoPE / DSA attention with the pooled indexer (W3, branch frog/glm5-mla). Critical path.
//!
//! The 11 DSA layers (`is_kda[l] == false`) are DeepSeek-style MLA with **NoPE** (no rope
//! anywhere, `qk_rope_head_dim` = 0): `q_lora_rank` 1536 → q heads (64 × qk 256),
//! `kv_lora_rank` 512 latent per token (the cache stores only the latent), per-head
//! decompression through `attn_k_b` [256, 512, 64] and `attn_v_b` [512, 256, 64], v head dim 256.
//!
//! The DSA indexer (`attention.indexer.*`: 32 heads, head dim 128, top-k 2048, kpool 4)
//! pools every 4 cached keys into one candidate (a learned softmax over the pool, gated by
//! `indexer_compressor_gate` + `indexer_compressor_ape`), scores pools per query
//! (relu · per-head weights from `indexer_proj`), and the attention runs over the top
//! `top_k / kpool` = 512 pools plus the visible tail. `indexer.k_norm` is a **biased LayerNorm**
//! (eps 1e-6), unlike everything else in the model.
//!
//! W3 also answers whether ≤2048-token sequences make the indexer a no-op (dense causal
//! attention) — until then [`super::forward`] runs it dense when `t_len ≤ idx_top_k`.
//!
//! HF reference: `Glm5NextTextAttention`, `Glm5NextTextIndexer`. GGUF names in [`super::gguf`].

/// Placeholder until W3's implementation lands.
pub fn mla_stub() {}

#[cfg(test)]
mod tests {
    #[test]
    fn placeholder() {}
}
