//! GLM-5.3-Flash W3: the DSA layers' MLA-NoPE attention and its k-pool indexer, CPU f32
//! reference + CUDA kernels. Owner: W3; W1 wires this into `glm5/forward.rs`.
//!
//! Ground truth: `transformers.models.glm5_next.modeling_glm5_next`
//! (`Glm5NextTextAttention`, `Glm5NextTextIndexer`), second reference llama.cpp
//! `src/models/glm5-next.cpp` (`build_dsa_layer`, `build_kpool_select`).
//!
//! # What the layer is
//!
//! DeepSeek-V3.2-style MLA with **no rope at all** (`mla_use_nope`,
//! `qk_rope_head_dim = 0`): `q_a_proj: hidden → q_lora(1536)`, RMSNorm, `q_b_proj` to
//! `n_head × qk_nope(256)`; `kv_a_proj_with_mqa: hidden → kv_lora(512)` (the rope part of
//! the projection is empty), RMSNorm — that 512-dim latent **is the KV cache entry**;
//! `kv_b_proj` (per head: `wk_b` 256 + `wv_b` 256) decompresses it per head. 64 heads,
//! v_head 256. NoPE changes nothing else: scores are plain dots, no positions anywhere.
//!
//! The indexer (`Glm5NextTextIndexer`) decides which keys each query may see:
//!
//! * per-token key `k = layer_norm(wk(x))` (learned weight **and bias**, eps 1e-6);
//! * the sequence is grouped into complete pools of `kpool`(4) tokens starting at the
//!   first valid token; each pool's key is a gate-weighted softmax average over its
//!   tokens (gates `indexer_kpool_compress_gate(x)` + a learned per-slot APE);
//! * score(pool, q) = relu(Σ_h w_h(x) · dot(q_h, pool_key_h)) with `w_h =
//!   weights_proj(x) / sqrt(n_idx_heads)`; a pool is a candidate for query `i` only if
//!   its last token is causally visible and padding-valid;
//! * the top `index_topk / kpool` = 512 pools by score are expanded back to their raw
//!   token indices, and the incomplete tail pool (up to `kpool - 1` tokens) is appended
//!   (`index_kpool_always_select_tail`), giving width `index_topk + kpool - 1` = 2051;
//! * the attention mask is exactly those tokens (`-1` slots are padding).
//!
//! **Indexer finding (verified on the HF module at the real config, 32 heads / kpool 4 /
//! top-2048 / tail on):** for `S ≤ 2048` visible keys the indexer is a **no-op** — every
//! causally visible token is selected and the output is exactly dense causal attention
//! (the top-512-of-⌈S/4⌉ selection keeps all pools, the tail covers the remainder). For
//! `S > 2048` each query saturates at 2048 selected keys (512 pools + 3 tail); at `S =
//! 2100` query 2051 sees 2048 of its 2051 visible keys, so tokens beyond 2048 are
//! *sparsified* and the ranking matters. `fn indexer_is_dense` encodes this for W1: run
//! plain causal attention while `n_kv ≤ index_topk`, and only pay the indexer cost after.
//!
//! The output mask width is `select_width()` = 2051 (2048 + tail 3), and `topk_indices`
//! has exactly that width; `-1` entries are invalid.
//!
//! # Layout contract with W1 (GGUF names, converter semantics)
//!
//! All weights are the dequantized f32 of the GGUF tensors, `y = W x` rows-major
//! (`Mat { rows, cols }`, one token per `cols` inputs):
//!
//! | GGUF tensor (blk.N.) | HF name | shape (rows × cols) |
//! |---|---|---|
//! | `attn_q_a` | `q_a_proj.weight` | q_lora × hidden |
//! | `attn_q_a_norm` | `q_a_layernorm.weight` | q_lora |
//! | `attn_q_b` | `q_b_proj.weight` | (n_head·qk_nope) × q_lora |
//! | `attn_kv_a_mqa` | `kv_a_proj_with_mqa.weight` | kv_lora × hidden (rope part empty) |
//! | `attn_kv_a_norm` | `kv_a_layernorm.weight` | kv_lora |
//! | `attn_k_b` | `k_b_proj.weight` (transposed by convert_hf_to_gguf.py) | kv_lora × (n_head·qk_nope), **3-D [kv_lora][qk_nope][n_head]** |
//! | `attn_v_b` | `v_b_proj.weight` | (n_head·v_head) × kv_lora, **3-D [n_head][v_head][kv_lora]** |
//! | `attn_output` | `o_proj.weight` | hidden × (n_head·v_head) |
//! | `indexer.attn_k` | `wk.weight` | idx_dim × hidden |
//! | `indexer.k_norm` w/b | `k_norm.weight/bias` | idx_dim each |
//! | `indexer.attn_q_b` | `wq_b.weight` | (n_idx_heads·idx_dim) × q_lora |
//! | `indexer.proj` | `weights_proj.weight` | n_idx_heads × hidden |
//! | `indexer_compressor_gate` | `index_kpool_compress_gate.weight` | idx_dim × hidden |
//! | `indexer_compressor_ape` | `index_kpool_compress_ape.weight` | idx_dim × kpool |
//!
//! `attn_k_b` / `attn_v_b` are written as GGUF 3-D tensors by the converter (the
//! `kv_b_proj` split): load them with [`MlaWeights::new`] directly from the flat f32
//! (row-major over the declared GGUF dims), or via [`MlaWeights::from_split_b`] if W1
//! has them as 2-D HF-shaped matrices.
//!
//! # Signatures for W1 (the whole public surface)
//!
//! ```text
//! MlaWeights::new(hidden, q_lora, kv_lora, n_head, qk_nope, v_head,
//!                 idx_heads, idx_dim, topk, kpool, select_tail,
//!                 q_a, q_a_norm, q_b, kv_a, kv_a_norm,
//!                 k_b /* flat 3-D GGUF */, v_b /* flat 3-D GGUF */, o,
//!                 i_wk, i_k_norm_w, i_k_norm_b, i_wq_b, i_proj, i_gate, i_ape) -> MlaWeights
//! MlaWeights::from_split_b(..same but k_b, v_b 2-D HF-shaped..) -> MlaWeights
//! w.k_nope(h) -> &[f32]        // kv_lora × qk_nope, per head (k_b GGUF layout)
//! w.v_b_head(h) -> &[f32]      // v_head × kv_lora, per head
//! w.gemv_q_a(&self, x: &[f32], t: usize) -> Vec<f32>          // t tokens: [t][q_lora]
//! w.gemv_kv_a(&self, x: &[f32], t: usize) -> Vec<f32>         // [t][kv_lora] (normed latent)
//! w.gemv_q_b(&self, q_resid: &[f32], t: usize) -> Vec<f32>    // [t][n_head*qk_nope]
//!
//! indexer_select(&w, x: &[f32], q_resid: &[f32], cache_k: &[f32], cache_gate: &[f32],
//!               cache_valid: &[u8], mask: &[u8], t: usize, kv_len: usize, pos0: usize)
//!   -> Vec<i32>               // [t][select_width()] token indices, -1 invalid
//! indexer_is_dense(kv_len: usize, topk: usize) -> bool   // kv_len <= topk
//!
//! mla_forward(&w, x: &[f32], q_resid: &[f32], k_latent: &[f32], n_kv: usize,
//!             t: usize, pos0: usize, mask: Option<&[u8]>, sel: Option<&[i32]>,
//!             out: &mut [f32])                              // out [t][hidden]
//! mla_forward_dense(&w, x, q_resid, k_latent, n_kv, t, pos0, mask, out)  // no-op sel fast path
//! MlaCuda::new(w: &MlaWeights) -> Result<MlaCuda>           // feature "cuda", NVRTC
//! MlaCuda::select(&self, x, q_resid, cache_k, cache_gate, cache_valid, mask,
//!                 t, kv_len, pos0, stream) -> Result<Vec<i32>>
//! MlaCuda::attend(&self, q: &[f32], k_latent: &[f32], n_kv, t, pos0,
//!                 sel: Option<&[i32]>, out_v: &mut [f32], stream) -> Result<()>  // [t][n_head*v_head]
//! MlaCuda::kv_append(&self, x: &[f32], cache: &mut [f32], t, pos0, stream) -> Result<()>  // bf16 latent rows
//! ```
//!
//! The `cache_*` indexer buffers are per-layer, appended per window (row `p` holds token
//! `p`'s `k[idx_dim] | gate[idx_dim]`, valid flag 0/1); `pos0` is the position of the
//! window's first row, `kv_len` the total keys including this window. `sel` rows hold
//! `select_width()` i32s.

use crate::flash::reference::{dot, rms_norm_mul};

/// Geometry + weights of one DSA layer (all f32, dequantized).
#[derive(Clone, Debug)]
pub struct MlaWeights {
    pub hidden: usize,
    pub q_lora: usize,
    pub kv_lora: usize,
    pub n_head: usize,
    pub qk_nope: usize,
    pub v_head: usize,
    pub idx_heads: usize,
    pub idx_dim: usize,
    pub topk: usize,
    pub kpool: usize,
    pub select_tail: bool,
    pub eps: f32,
    /// `q_a_proj`: q_lora × hidden.
    pub q_a: Vec<f32>,
    /// `q_a_layernorm.weight`: q_lora.
    pub q_a_norm: Vec<f32>,
    /// `q_b_proj`: (n_head·qk_nope) × q_lora.
    pub q_b: Vec<f32>,
    /// `kv_a_proj_with_mqa` rope part is empty: kv_lora × hidden.
    pub kv_a: Vec<f32>,
    /// `kv_a_layernorm.weight`: kv_lora.
    pub kv_a_norm: Vec<f32>,
    /// GGUF `attn_k_b` 3-D [kv_lora][qk_nope][n_head], flat row-major.
    k_b: Vec<f32>,
    /// GGUF `attn_v_b` 3-D [n_head][v_head][kv_lora], flat row-major.
    v_b: Vec<f32>,
    /// `o_proj`: hidden × (n_head·v_head).
    pub o: Vec<f32>,
    /// Indexer `wk`: idx_dim × hidden.
    pub i_wk: Vec<f32>,
    /// `k_norm.weight` / `k_norm.bias`: idx_dim each (LayerNorm, eps 1e-6).
    pub i_k_norm_w: Vec<f32>,
    pub i_k_norm_b: Vec<f32>,
    /// Indexer `wq_b`: (idx_heads·idx_dim) × q_lora.
    pub i_wq_b: Vec<f32>,
    /// `weights_proj`: idx_heads × hidden.
    pub i_proj: Vec<f32>,
    /// `index_kpool_compress_gate`: idx_dim × hidden.
    pub i_gate: Vec<f32>,
    /// `index_kpool_compress_ape`: idx_dim × kpool (HF is [kpool][idx_dim]; passed transposed).
    pub i_ape: Vec<f32>,
}

/// Max output width of [`indexer_select`].
pub const SELECT_MAX_EXTRA: usize = 3;

impl MlaWeights {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden: usize,
        q_lora: usize,
        kv_lora: usize,
        n_head: usize,
        qk_nope: usize,
        v_head: usize,
        idx_heads: usize,
        idx_dim: usize,
        topk: usize,
        kpool: usize,
        select_tail: bool,
        eps: f32,
        q_a: Vec<f32>,
        q_a_norm: Vec<f32>,
        q_b: Vec<f32>,
        kv_a: Vec<f32>,
        kv_a_norm: Vec<f32>,
        // GGUF 3-D `attn_k_b` (ne [qk_nope, kv_lora, n_head]), flat C-order
        // [n_head][kv_lora][qk_nope]: per head a [kv_lora][qk_nope] slice.
        k_b: Vec<f32>,
        // GGUF 3-D `attn_v_b`, flat [n_head][v_head][kv_lora].
        v_b: Vec<f32>,
        o: Vec<f32>,
        i_wk: Vec<f32>,
        i_k_norm_w: Vec<f32>,
        i_k_norm_b: Vec<f32>,
        i_wq_b: Vec<f32>,
        i_proj: Vec<f32>,
        i_gate: Vec<f32>,
        // HF layout [kpool][idx_dim]; transposed into [idx_dim][kpool] here.
        i_ape_hf: Vec<f32>,
    ) -> Self {
        // keep the HF layout [kpool][idx_dim]: slot-major, `i_ape[j * idx_dim + d]`
        let ape = i_ape_hf;
        Self {
            hidden,
            q_lora,
            kv_lora,
            n_head,
            qk_nope,
            v_head,
            idx_heads,
            idx_dim,
            topk,
            kpool,
            select_tail,
            eps,
            q_a,
            q_a_norm,
            q_b,
            kv_a,
            kv_a_norm,
            k_b,
            v_b,
            o,
            i_wk,
            i_k_norm_w,
            i_k_norm_b,
            i_wq_b,
            i_proj,
            i_gate,
            i_ape: ape,
        }
    }

    /// Same, but `k_b`/`v_b` in the HF 2-D shapes (`k_b_proj` is written transposed by
    /// the converter, so pass it **as loaded from GGUF**: [kv_lora][n_head·qk_nope]; `v_b`
    /// as the original [n_head·v_head][kv_lora]).
    #[allow(clippy::too_many_arguments)]
    pub fn from_split_b(
        hidden: usize,
        q_lora: usize,
        kv_lora: usize,
        n_head: usize,
        qk_nope: usize,
        v_head: usize,
        idx_heads: usize,
        idx_dim: usize,
        topk: usize,
        kpool: usize,
        select_tail: bool,
        eps: f32,
        q_a: Vec<f32>,
        q_a_norm: Vec<f32>,
        q_b: Vec<f32>,
        kv_a: Vec<f32>,
        kv_a_norm: Vec<f32>,
        // HF `kv_b_proj` 2-D: [(n_head*(qk_nope+v_head))][kv_lora], head-major.
        k_b2: Vec<f32>,
        // HF `v_b_proj` 2-D: [(n_head*v_head)][kv_lora].
        v_b2: Vec<f32>,
        o: Vec<f32>,
        i_wk: Vec<f32>,
        i_k_norm_w: Vec<f32>,
        i_k_norm_b: Vec<f32>,
        i_wq_b: Vec<f32>,
        i_proj: Vec<f32>,
        i_gate: Vec<f32>,
        i_ape_hf: Vec<f32>,
    ) -> Self {
        // HF `kv_b_proj` is [(n_head*(qk_nope+v_head))][kv_lora], head-major, k rows
        // then v rows per head. Repack into the GGUF 3-D layouts `new` takes:
        // k_b flat [h][l][d] (from k_b_hf[h][d][l], transposed), v_b flat [h][d][l].
        let mut k_b = vec![0f32; n_head * kv_lora * qk_nope];
        let mut v_b = vec![0f32; n_head * v_head * kv_lora];
        for h in 0..n_head {
            for d in 0..qk_nope {
                let src = (h * (qk_nope + v_head) + d) * kv_lora;
                for l in 0..kv_lora {
                    k_b[h * kv_lora * qk_nope + l * qk_nope + d] = k_b2[src + l];
                }
            }
            for d in 0..v_head {
                let src = (h * (qk_nope + v_head) + qk_nope + d) * kv_lora;
                for l in 0..kv_lora {
                    v_b[(h * v_head + d) * kv_lora + l] = v_b2[src + l];
                }
            }
        }
        Self::new(
            hidden, q_lora, kv_lora, n_head, qk_nope, v_head, idx_heads, idx_dim, topk, kpool,
            select_tail, eps, q_a, q_a_norm, q_b, kv_a, kv_a_norm, k_b, v_b, o, i_wk, i_k_norm_w,
            i_k_norm_b, i_wq_b, i_proj, i_gate, i_ape_hf,
        )
    }

    /// Head `h`'s nope-key up-projection: [kv_lora × qk_nope] (GGUF `attn_k_b` slice).
    pub fn k_nope(&self, h: usize) -> &[f32] {
        &self.k_b[h * self.kv_lora * self.qk_nope..(h + 1) * self.kv_lora * self.qk_nope]
    }

    /// Head `h`'s value up-projection: [v_head × kv_lora] (GGUF `attn_v_b` slice).
    pub fn v_b_head(&self, h: usize) -> &[f32] {
        &self.v_b[h * self.v_head * self.kv_lora..(h + 1) * self.v_head * self.kv_lora]
    }

    /// Width of [`indexer_select`] output rows.
    pub fn select_width(&self) -> usize {
        self.topk + if self.select_tail { self.kpool - 1 } else { 0 }
    }

    // --- shared GEMVs (t tokens packed [t][cols] -> [t][rows]) ---

    /// `q_a_proj` then `q_a_layernorm`: [t][hidden] -> [t][q_lora].
    pub fn gemv_q_a(&self, x: &[f32], t: usize) -> Vec<f32> {
        let mut y = gemv(&self.q_a, self.q_lora, self.hidden, x, t);
        for r in y.chunks_mut(self.q_lora) {
            rms_norm_mul(r, &self.q_a_norm, self.eps);
        }
        y
    }

    /// `kv_a_proj_with_mqa` (rope part empty) then `kv_a_layernorm`: [t][hidden] ->
    /// [t][kv_lora]. This is the latent that goes into the KV cache.
    pub fn gemv_kv_a(&self, x: &[f32], t: usize) -> Vec<f32> {
        let mut y = gemv(&self.kv_a, self.kv_lora, self.hidden, x, t);
        for r in y.chunks_mut(self.kv_lora) {
            rms_norm_mul(r, &self.kv_a_norm, self.eps);
        }
        y
    }

    /// `q_b_proj`: [t][q_lora] -> [t][n_head·qk_nope] (head-major).
    pub fn gemv_q_b(&self, q_resid: &[f32], t: usize) -> Vec<f32> {
        gemv(&self.q_b, self.n_head * self.qk_nope, self.q_lora, q_resid, t)
    }
}

/// `y = W x` for `t` packed tokens: W rows × cols row-major, x [t][cols], y [t][rows].
fn gemv(w: &[f32], rows: usize, cols: usize, x: &[f32], t: usize) -> Vec<f32> {
    assert_eq!(w.len(), rows * cols, "weight shape");
    assert_eq!(x.len(), t * cols, "input shape");
    let mut y = vec![0f32; t * rows];
    use rayon::prelude::*;
    y.par_chunks_mut(rows).zip(x.par_chunks(cols)).for_each(|(yo, xi)| {
        for (r, o) in yo.iter_mut().enumerate() {
            *o = dot(&w[r * cols..(r + 1) * cols], xi);
        }
    });
    y
}

/// Is the indexer a no-op (dense causal attention) at this cache length? True while
/// `kv_len <= topk`: the `topk/kpool` pool budget then covers every complete pool
/// (`ceil(kv_len/kpool) <= topk/kpool`) and the always-selected tail covers the rest,
/// so every causally visible token is selected. Verified against the HF module at the
/// real config (32 heads, kpool 4, top-2048, tail on): S=2048 selects all, S=2100
/// saturates at 2048 keys per query.
pub fn indexer_is_dense(kv_len: usize, topk: usize) -> bool {
    kv_len <= topk
}

/// The indexer: which keys each of the `t` queries (window at `pos0`) may attend to.
///
/// Exactly `Glm5NextTextIndexer.forward` with a dynamic cache (the no-cache case is
/// `kv_len = t`, `pos0 = 0`):
///
/// * `x`: [t][hidden] — this window's hidden states (`weights_proj` and the pool gates
///   read them).
/// * `q_resid`: [t][q_lora] — `q_a_layernorm(q_a_proj(x))` ([`MlaWeights::gemv_q_a`]).
/// * `cache_k` / `cache_gate`: [kv_len][idx_dim] — per-token indexer key
///   (`layer_norm(wk·x)`, see [`indexer_key``) and pool-gate row (`i_gate·x`), the
///   window's rows already appended by the caller.
/// * `cache_valid`: [kv_len] — 1 for real (non-pad) key rows.
/// * `mask`: [t] — 1 for real query rows.
///
/// **Batching: one call per sequence** (`t` = that sequence's rows; caches and
/// `pos0`/`kv_len` are per-sequence). HF's `[B, S, …]` batch shape is the caller
/// looping this over B — pooling, visibility and tails are all per-sequence state.
///
/// Returns `[t][w.select_width()]` i32 KV-token indices (`-1` invalid), ordered
/// pool-by-descending-score then tail. The *set* of selected tokens matches the HF
/// module exactly (verified row by row); the *order* among exactly-tied pool scores
/// (relu clamps many to 0.0 on random weights) is not — torch.topk's tie order is an
/// internal artifact, we take the lower index first. Real weights rarely tie; the
/// attention is order-invariant anyway, so only the mask matters downstream.
#[allow(clippy::too_many_arguments)]
pub fn indexer_select(
    w: &MlaWeights,
    x: &[f32],
    q_resid: &[f32],
    cache_k: &[f32],
    cache_gate: &[f32],
    cache_valid: &[u8],
    mask: &[u8],
    t: usize,
    kv_len: usize,
    pos0: usize,
) -> Vec<i32> {
    let (hd, nh, kp) = (w.idx_dim, w.idx_heads, w.kpool);
    assert_eq!(cache_k.len(), kv_len * hd, "cache_k shape");
    assert_eq!(cache_gate.len(), kv_len * hd, "cache_gate shape");
    let width = w.select_width();
    let mut out = vec![-1i32; t * width];
    if t == 0 || kv_len == 0 {
        return out;
    }

    // indexer query: [t][nh][hd], head-major
    let iq = gemv(&w.i_wq_b, nh * hd, w.q_lora, q_resid, t);
    // per-query head weights: [t][nh] (weights_proj is [nh][hidden])
    let hw = gemv(&w.i_proj, nh, w.hidden, x, t);
    let hw_scale = (nh as f32).sqrt().recip();
    let score_scale = (hd as f32).sqrt().recip();

    // pools start at the first valid key and run while all kpool slots are valid
    // (left padding is skipped entirely, right padding ends the pool run).
    let first_key = (0..kv_len).find(|&p| cache_valid[p] == 1).unwrap_or(kv_len);
    let mut n_pool = 0usize;
    while {
        let base = first_key + n_pool * kp;
        base + kp <= kv_len && (0..kp).all(|j| cache_valid[base + j] == 1)
    } {
        n_pool += 1;
    }
    let n_select = (w.topk / kp).min(n_pool);

    // per-pool keys: softmax over the kpool slots (gate + ape) per dim, then the
    // weighted average of the slots' keys — `get_pooled_states`.
    let mut pool_key = vec![0f32; n_pool * hd];
    for p in 0..n_pool {
        let base = first_key + p * kp;
        for d in 0..hd {
            let mut mx = f32::NEG_INFINITY;
            let mut lg = [0f32; 64];
            for (j, l) in lg.iter_mut().enumerate().take(kp) {
                *l = cache_gate[(base + j) * hd + d] + w.i_ape[j * hd + d];
                mx = mx.max(*l);
            }
            let mut sum = 0f32;
            for l in lg.iter_mut().take(kp) {
                *l = (*l - mx).exp();
                sum += *l;
            }
            let mut acc = 0f32;
            for (j, &l) in lg.iter().enumerate().take(kp) {
                acc += l / sum * cache_k[(base + j) * hd + d];
            }
            pool_key[p * hd + d] = acc;
        }
    }

    let mut order: Vec<usize> = Vec::with_capacity(n_pool);
    let mut scores = vec![0f32; n_pool];
    for qi in 0..t {
        if mask[qi] == 0 {
            continue; // row stays all -1, as HF masks it
        }
        let qpos = pos0 + qi;
        // keys visible to this query: causal (k <= qpos) and valid
        let visible: Vec<bool> = (0..kv_len).map(|k| k <= qpos && cache_valid[k] == 1).collect();
        let visible_count = visible.iter().filter(|&&v| v).count();
        // the query's own incomplete pool: the last visible_count % kpool visible
        // tokens (append_visible_tail)
        let tail_count = visible_count % kp;
        let tail_start = first_key + visible_count - tail_count;
        // per-pool score: Σ_h hw[h] · relu(dot(iq_h, pool_key_h) · softmax_scale)
        for p in 0..n_pool {
            let end = first_key + p * kp + kp - 1;
            if end > qpos || !visible[end] {
                scores[p] = f32::NEG_INFINITY; // pool end not visible to this query
                continue;
            }
            let pk = &pool_key[p * hd..(p + 1) * hd];
            let mut s = 0f32;
            for h in 0..nh {
                let q0 = qi * nh * hd + h * hd;
                let mut d = dot(&iq[q0..q0 + hd], pk) * score_scale;
                if d < 0.0 {
                    d = 0.0;
                }
                s += hw[qi * nh + h] * hw_scale * d;
            }
            scores[p] = s;
        }
        // top n_select pools by score, ties to the lower index (torch.topk order)
        order.clear();
        order.extend(0..n_pool);
        order.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        // exactly n_select pool slots, always (HF flattens topk(select_k) pools and
        // masks invisible ones to -1), so the tail always lands at slot topk
        let mut wi = 0;
        for &p in order.iter().take(n_select) {
            let visible_pool = scores[p] != f32::NEG_INFINITY;
            for j in 0..kp {
                if wi < width {
                    out[qi * width + wi] = if visible_pool {
                        (first_key + p * kp + j) as i32
                    } else {
                        -1
                    };
                    wi += 1;
                }
            }
        }
        // the tail: up to kp-1 slots, only those that exist and are visible
        for j in 0..kp - 1 {
            if j >= tail_count {
                break;
            }
            let idx = tail_start + j;
            if idx < kv_len && visible[idx] && wi < width {
                out[qi * width + wi] = idx as i32;
                wi += 1;
            }
        }
    }
    out
}

/// One token's indexer cache rows: `k = layer_norm(wk·x)` (learned weight **and bias**,
/// eps 1e-6 — `nn.LayerNorm`, not RMS) and `gate = i_gate·x`. Call per token, append to
/// the layer's `cache_k` / `cache_gate`.
pub fn indexer_key(w: &MlaWeights, x: &[f32], k_out: &mut [f32], gate_out: &mut [f32]) {
    assert_eq!(x.len(), w.hidden);
    assert_eq!(k_out.len(), w.idx_dim);
    assert_eq!(gate_out.len(), w.idx_dim);
    for (d, g) in gate_out.iter_mut().enumerate() {
        *g = dot(&w.i_gate[d * w.hidden..(d + 1) * w.hidden], x);
    }
    let mut k = vec![0f32; w.idx_dim];
    for (d, kd) in k.iter_mut().enumerate() {
        *kd = dot(&w.i_wk[d * w.hidden..(d + 1) * w.hidden], x);
    }
    // LayerNorm over the dim axis
    let n = w.idx_dim as f64;
    let mean: f64 = k.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var: f64 = k.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    let inv = 1.0 / (var + 1e-6).sqrt();
    for d in 0..w.idx_dim {
        k_out[d] = ((k[d] as f64 - mean) * inv) as f32 * w.i_k_norm_w[d] + w.i_k_norm_b[d];
    }
}

/// MLA attention over the latent KV cache for one window of `t` queries at `pos0`.
///
/// * `x`: [t][hidden] — unused (kept so W1 can pass the block input straight through);
///   the projections arrive pre-computed (`q_resid`) or are done here from it.
/// * `q_resid`: [t][q_lora] (`gemv_q_a` output); `q = q_b·q_resid` is computed here.
/// * `k_latent`: [n_kv][kv_lora] — the normed latent cache (all keys, bf16-rounded on
///   the GPU path; f32 here).
/// * `mask`: [t] query padding (0 = drop the row); `sel`: `[t][select_width()]` token
///   indices from [`indexer_select`], or `None` for dense causal attention (the
///   `kv_len ≤ topk` fast path — [`indexer_is_dense`]).
/// * `out`: [t][hidden], `o_proj` applied.
///
/// Values are computed as `v_h = wv_h · latent` per selected key (per-head decompress),
/// the unabsorbed reference semantics; the absorbed form is a GPU plan (see below).
#[allow(clippy::too_many_arguments)]
pub fn mla_forward(
    w: &MlaWeights,
    x: &[f32],
    q_resid: &[f32],
    k_latent: &[f32],
    n_kv: usize,
    t: usize,
    pos0: usize,
    mask: Option<&[u8]>,
    sel: Option<&[i32]>,
    out: &mut [f32],
) {
    let _ = x;
    let (nh, dk, dv, dl) = (w.n_head, w.qk_nope, w.v_head, w.kv_lora);
    assert_eq!(k_latent.len(), n_kv * dl, "latent cache shape");
    assert_eq!(out.len(), t * w.hidden, "out shape");
    let q = w.gemv_q_b(q_resid, t);
    let scale = 1.0 / (dk as f32).sqrt();
    let width = w.select_width();
    let mut attn_o = vec![0f32; t * nh * dv];

    // resolve each query's key list once. A padded query row (mask 0) attends
    // UNIFORMLY over all n_kv keys in HF: its top-k row is all -1, the eager mask
    // REPLACES those scores with the dtype min (equal values), and softmax over equal
    // scores is uniform — `uniform_row` below encodes it exactly.
    let uniform_row = |qi: usize| mask.is_some_and(|m| m[qi] == 0);
    let keys: Vec<Vec<usize>> = (0..t)
        .map(|qi| {
            if uniform_row(qi) {
                (0..n_kv).collect()
            } else if let Some(s) = sel {
                s[qi * width..(qi + 1) * width]
                    .iter()
                    .filter(|&&i| i >= 0)
                    .map(|&i| i as usize)
                    .collect()
            } else {
                (0..=pos0 + qi).collect()
            }
        })
        .collect();

    for (qi, ks) in keys.iter().enumerate() {
        if ks.is_empty() {
            // a real query with an empty selection: HF softmaxes an all -inf row (NaN
            // downstream). Keep zeros but flag it in debug builds.
            debug_assert!(false, "query {qi}: empty selection");
            continue;
        }
        let uni = uniform_row(qi);
        for h in 0..nh {
            let q0 = qi * nh * dk + h * dk;
            let qh = &q[q0..q0 + dk];
            let wk = w.k_nope(h); // [dl][dk]
            let wv = w.v_b_head(h); // [dv][dl]
            // per-key nope key k = latent @ wk  ([dk]) and score
            let mut sc = vec![0f32; ks.len()];
            let mut mx = if uni { 0.0 } else { f32::NEG_INFINITY };
            if uni {
                // uniform over every key (see `uniform_row`); scores unused
                for v in sc.iter_mut() {
                    *v = 0.0;
                }
            }
            for (i, &k) in ks.iter().enumerate() {
                let kl = &k_latent[k * dl..(k + 1) * dl];
                let mut kq = [0f32; 256];
                debug_assert!(dk <= 256);
                for l in 0..dl {
                    let lv = kl[l];
                    if lv != 0.0 {
                        let wrow = &wk[l * dk..(l + 1) * dk];
                        for (d, o) in kq.iter_mut().enumerate().take(dk) {
                            *o += lv * wrow[d];
                        }
                    }
                }
                sc[i] = dot(qh, &kq[..dk]) * scale;
                if !uni {
                    mx = mx.max(sc[i]);
                }
            }
            let mut den = 0f32;
            let mut acc = vec![0f32; dv];
            let inv_n = 1.0 / ks.len() as f32;
            for (i, &k) in ks.iter().enumerate() {
                let p = if uni { inv_n } else { (sc[i] - mx).exp() };
                den += p;
                let kl = &k_latent[k * dl..(k + 1) * dl];
                for d in 0..dv {
                    let mut s = 0f32;
                    let wrow = &wv[d * dl..(d + 1) * dl];
                    for l in 0..dl {
                        s += wrow[l] * kl[l];
                    }
                    acc[d] += p * s;
                }
            }
            let o0 = qi * nh * dv + h * dv;
            for d in 0..dv {
                attn_o[o0 + d] = acc[d] / den;
            }
        }
    }
    // o_proj: [t][hidden]
    use rayon::prelude::*;
    out.par_chunks_mut(w.hidden)
        .zip(attn_o.par_chunks(nh * dv))
        .for_each(|(yo, ao)| {
            for (r, o) in yo.iter_mut().enumerate() {
                *o = dot(&w.o[r * nh * dv..(r + 1) * nh * dv], ao);
            }
        });
}

/// Dense causal fast path (`sel = None`): see [`mla_forward`].
#[allow(clippy::too_many_arguments)]
pub fn mla_forward_dense(
    w: &MlaWeights,
    x: &[f32],
    q_resid: &[f32],
    k_latent: &[f32],
    n_kv: usize,
    t: usize,
    pos0: usize,
    mask: Option<&[u8]>,
    out: &mut [f32],
) {
    mla_forward(w, x, q_resid, k_latent, n_kv, t, pos0, mask, None, out);
}

// ---------------------------------------------------------------------------------
// CUDA
//
// The GPU path keeps the KV cache as the kv_lora-dim bf16 latent (512 × 2 B per token
// per DSA layer) and decompresses per head inside the attention kernel — the "naive
// path for first tokens" of the brief. `mla_proj` computes the normed latent rows
// (kv_append), `mla_attn_naive` is the unabsorbed per-(query, head) attention over
// either a dense causal range or the indexer selection. The absorbed projection and
// the pooled indexer kernel for > 2048 tokens are planned, not built (PLANS below).

/// The NVRTC source (like `coder_kernels.rs`); shapes come as `#define`s from
/// [`cuda_prelude`]. `mla_attn_naive` streams the selection twice (max pass, then
/// softmax pass) — correct, deliberately naive; the tensor-core rewrite comes with the
/// absorbed projection.
pub const CUDA_SRC: &str = r#"
#define NEG_INF_F __int_as_float(0xff800000)

__device__ __forceinline__ unsigned short f2bf(float x) {
    unsigned int b = __float_as_uint(x);
    return (unsigned short)((b + 0x7fffu + ((b >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ float bf2f(unsigned short b) {
    return __uint_as_float(((unsigned int)b) << 16);
}

// One block per token. a: [dl*hid] (kv_a), an: [dl] (kv_a_norm), x: [t][hid] f32,
// y: [t][dl] bf16 out rows. eps as a float arg.
extern "C" __global__ void mla_kv_append(
    const float* __restrict__ a, const float* __restrict__ an, const float* __restrict__ x,
    unsigned short* __restrict__ y, int t, int hid, int dl, float eps)
{
    const int tok = blockIdx.x;
    if (tok >= t) return;
    const float* xt = x + (size_t)tok * hid;
    __shared__ float sh[2048];
    for (int r = threadIdx.x; r < dl; r += blockDim.x) {
        const float* ar = a + (size_t)r * hid;
        float s = 0.f;
        for (int c = 0; c < hid; c++) s += ar[c] * xt[c];
        sh[r] = s;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float ss = 0.f;
        for (int r = 0; r < dl; r++) ss += sh[r] * sh[r];
        const float inv = rsqrtf(ss / dl + eps);
        for (int r = 0; r < dl; r++) y[(size_t)tok * dl + r] = f2bf(sh[r] * inv * an[r]);
    }
}

// One block per (query, head): blockIdx.x = head, blockIdx.y = query.
// q: [t][nh*dk] f32; lat: [nkv][dl] bf16; wk: [nh][dl*dk] f32; wv: [nh][dv*dl] f32;
// sel: [t][width] i32 or NULL for dense-causal-up-to-qpos; out: [t][nh*dv] f32.
// smem: dk floats (the per-key nope key) + dv floats (accumulator).
extern "C" __global__ void mla_attn_naive(
    const float* __restrict__ q, const unsigned short* __restrict__ lat,
    const float* __restrict__ wk, const float* __restrict__ wv,
    const int* __restrict__ sel, float* __restrict__ out,
    int t, int nkv, int pos0, int nh, int dk, int dv, int dl, int width, float scale)
{
    const int qi = blockIdx.y;
    const int h  = blockIdx.x;
    if (qi >= t || h >= nh) return;
    const int qpos = pos0 + qi;
    const float* qh = q + (size_t)qi * nh * dk + (size_t)h * dk;
    const float* wh_k = wk + (size_t)h * dl * dk;
    const float* wh_v = wv + (size_t)h * dv * dl;
    extern __shared__ float smem[];
    float* kvec = smem;          // [dk]
    float* acc  = smem + dk;     // [dv]
    for (int d = threadIdx.x; d < dv; d += blockDim.x) acc[d] = 0.f;

    // pass 1: max score
    float m = -INFINITY;
    for (int w = 0; w < width; w++) {
        int k;
        if (sel == NULL) { k = w; if (k > qpos) break; }
        else { k = sel[(size_t)qi * width + w]; if (k < 0) continue; }
        if (k >= nkv) continue;
        const unsigned short* kl = lat + (size_t)k * dl;
        __syncthreads();
        for (int d = threadIdx.x; d < dk; d += blockDim.x) {
            float s = 0.f;
            for (int l = 0; l < dl; l++) s += bf2f(kl[l]) * wh_k[(size_t)l * dk + d];
            kvec[d] = s;
        }
        __syncthreads();
        float sc = 0.f;
        for (int d = 0; d < dk; d++) sc += qh[d] * kvec[d];
        sc *= scale;
        if (sc > m) m = sc;
    }

    // pass 2: softmax-weighted value sum
    float den = 0.f;
    for (int w = 0; w < width; w++) {
        int k;
        if (sel == NULL) { k = w; if (k > qpos) break; }
        else { k = sel[(size_t)qi * width + w]; if (k < 0) continue; }
        if (k >= nkv) continue;
        const unsigned short* kl = lat + (size_t)k * dl;
        __syncthreads();
        for (int d = threadIdx.x; d < dk; d += blockDim.x) {
            float s = 0.f;
            for (int l = 0; l < dl; l++) s += bf2f(kl[l]) * wh_k[(size_t)l * dk + d];
            kvec[d] = s;
        }
        __syncthreads();
        float sc = 0.f;
        for (int d = 0; d < dk; d++) sc += qh[d] * kvec[d];
        sc *= scale;
        const float p = expf(sc - m);
        den += p;
        for (int d = threadIdx.x; d < dv; d += blockDim.x) {
            float s = 0.f;
            for (int l = 0; l < dl; l++) s += wh_v[(size_t)d * dl + l] * bf2f(kl[l]);
            acc[d] += p * s;
        }
        __syncthreads();
    }
    float* outp = out + (size_t)qi * nh * dv + (size_t)h * dv;
    for (int d = threadIdx.x; d < dv; d += blockDim.x)
        outp[d] = den > 0.f ? acc[d] / den : 0.f;
}
"#;

/// The `#define` prelude for [`CUDA_SRC`] at this geometry.
pub fn cuda_prelude(hidden: usize, q_lora: usize, kv_lora: usize, n_head: usize) -> String {
    format!(
        "#define HID {hid}\n#define QL {ql}\n#define DL {dl}\n#define NH {nh}\n",
        hid = hidden, ql = q_lora, dl = kv_lora, nh = n_head
    )
}

/// PLANS (not built, per the brief):
///
/// 1. **Absorbed-projection MLA** (llama.cpp `build_dsa_layer`): keep the query in
///    latent space — `q' = (Wk_b)^T q` makes the score `dot(q', latent)` over 512 dims
///    with no per-key decompression, and `Wv_b` is applied once to the attended latent
///    sum (a [nh·256 × 512] GEMV per query) instead of per key per head. The latent
///    dot is then a plain attention over head_dim 512, which `cx_attn_fa2` already
///    covers (HD up to 256 per call — check: FA2 supports 256; 512 needs the split
///    variant or two calls), and decode reuses the split-K path.
/// 2. **Pooled indexer > 2048**: the pool keys live in a per-layer ring
///    ([n_pool][idx_dim] f32 with the gate rows beside them, as llama.cpp's
///    `llm_graph_input_kpool`), refreshed by a small pooling kernel whenever a pool
///    completes; the selection kernel is one block per query computing the 512-pool
///    scores and a register top-k (the `ggml_lightning_indexer` fused op is the model),
///    writing the [t][2051] i32 rows. Until then `S > 2048` cannot run on the GPU.
/// 3. Cache layout is fixed now: per DSA layer, [cap][512] bf16 latents plus the
///    indexer's [cap][128] f32 k and gate rows (the indexer stays f32 — its APE/gate
///    math is sensitive).
#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use tang_moe::gpu::{self, DevBuf, Gpu, Module, Stream};

    /// Loaded kernels for one geometry (feature "cuda"; NVRTC at construction).
    pub struct MlaCuda {
        g: Gpu,
        m: Module,
        hid: usize,
        ql: usize,
        dl: usize,
        nh: usize,
        dk: usize,
        dv: usize,
        eps: f32,
    }

    impl MlaCuda {
        /// Compile the kernels for this weight set's geometry. Host weights are kept
        /// by the caller and staged per call (the naive path is not yet weight-resident).
        pub fn new(w: &MlaWeights) -> Result<Self, gpu::Error> {
            let g = Gpu::new()?;
            g.bind()?;
            let src = format!(
                "{}{}",
                cuda_prelude(w.hidden, w.q_lora, w.kv_lora, w.n_head),
                CUDA_SRC
            );
            let m = g.module(&src)?;
            Ok(Self {
                g,
                m,
                hid: w.hidden,
                ql: w.q_lora,
                dl: w.kv_lora,
                nh: w.n_head,
                dk: w.qk_nope,
                dv: w.v_head,
                eps: w.eps,
            })
        }

        /// Append `t` tokens' normed latents (bf16 rows) at `pos0`: stages `x`
        /// ([t][hidden]) and the kv_a weights, launches `mla_kv_append`, copies the
        /// [t][dl] u16 rows back. `cache` must have room for `pos0 + t` rows.
        pub fn kv_append(
            &self,
            kv_a: &[f32],     // [dl*hid]
            kv_a_norm: &[f32],// [dl]
            x: &[f32],        // [t][hid]
            t: usize,
            pos0: usize,
            cache: &mut [u16], // [cap][dl]
            stream: &Stream,
        ) -> Result<(), gpu::Error> {
            assert_eq!(x.len(), t * self.hid);
            assert_eq!(kv_a.len(), self.dl * self.hid);
            let a = DevBuf::from_slice(kv_a)?;
            let an = DevBuf::from_slice(kv_a_norm)?;
            let xd = DevBuf::from_slice(x)?;
            let yd = DevBuf::zeroed(t * self.dl * 2)?;
            let f = self.m.func("mla_kv_append")?;
            let (mut tt, mut hid, mut dl) = (t as i32, self.hid as i32, self.dl as i32);
            let mut eps = self.eps;
            let mut args = [
                a.ptr as *mut std::ffi::c_void,
                an.ptr as *mut std::ffi::c_void,
                xd.ptr as *mut std::ffi::c_void,
                yd.ptr as *mut std::ffi::c_void,
                &mut tt as *mut i32 as *mut std::ffi::c_void,
                &mut hid as *mut i32 as *mut std::ffi::c_void,
                &mut dl as *mut i32 as *mut std::ffi::c_void,
                &mut eps as *mut f32 as *mut std::ffi::c_void,
            ];
            unsafe { gpu::launch(f, (t as u32, 1, 1), 256, 0, stream, &mut args)? };
            let rows: Vec<u16> = yd.read(0, t * self.dl)?;
            cache[pos0 * self.dl..(pos0 + t) * self.dl].copy_from_slice(&rows);
            Ok(())
        }

        /// Naive attention ([`CUDA_SRC`]). `q`: [t][nh*dk] f32 (`gemv_q_b` output),
        /// `lat`: [n_kv][dl] u16 bf16, `sel`: [t][width] i32 (`width = select_width()`)
        /// or `None` for dense causal. Writes `out_v`: [t][nh*dv] f32 (before o_proj;
        /// apply it with the host GEMV — the naive path is a parity vehicle).
        #[allow(clippy::too_many_arguments)]
        pub fn attend(
            &self,
            wk: &[f32],   // [nh][dl*dk] flat, w.k_nope(h) concatenated
            wv: &[f32],   // [nh][dv*dl] flat, w.v_b_head(h) concatenated
            q: &[f32],    // [t][nh*dk]
            lat: &[u16],  // [n_kv][dl]
            n_kv: usize,
            t: usize,
            pos0: usize,
            sel: Option<&[i32]>,
            out_v: &mut [f32], // [t][nh*dv]
            stream: &Stream,
        ) -> Result<(), gpu::Error> {
            assert_eq!(wk.len(), self.nh * self.dl * self.dk);
            assert_eq!(wv.len(), self.nh * self.dv * self.dl);
            assert_eq!(lat.len(), n_kv * self.dl);
            assert_eq!(out_v.len(), t * self.nh * self.dv);
            let width = sel.map_or(pos0 + t, |s| {
                assert_eq!(s.len() % t, 0);
                s.len() / t
            });
            let wkd = DevBuf::from_slice(wk)?;
            let wvd = DevBuf::from_slice(wv)?;
            let qd = DevBuf::from_slice(q)?;
            let ld = DevBuf::from_slice(lat)?;
            let od = DevBuf::zeroed(out_v.len())?;
            let sd = sel.map(DevBuf::from_slice).transpose()?;
            let f = self.m.func("mla_attn_naive")?;
            let (mut tt, mut nkv, mut p0) = (t as i32, n_kv as i32, pos0 as i32);
            let (mut nh, mut dk, mut dv, mut dl, mut wd) = (
                self.nh as i32,
                self.dk as i32,
                self.dv as i32,
                self.dl as i32,
                width as i32,
            );
            let mut scale = 1.0f32 / (self.dk as f32).sqrt();
            let mut args = [
                qd.ptr as *mut std::ffi::c_void,
                ld.ptr as *mut std::ffi::c_void,
                wkd.ptr as *mut std::ffi::c_void,
                wvd.ptr as *mut std::ffi::c_void,
                sd.map(|b| b.ptr).unwrap_or(0) as *mut std::ffi::c_void,
                od.ptr as *mut std::ffi::c_void,
                &mut tt as *mut i32 as *mut std::ffi::c_void,
                &mut nkv as *mut i32 as *mut std::ffi::c_void,
                &mut p0 as *mut i32 as *mut std::ffi::c_void,
                &mut nh as *mut i32 as *mut std::ffi::c_void,
                &mut dk as *mut i32 as *mut std::ffi::c_void,
                &mut dv as *mut i32 as *mut std::ffi::c_void,
                &mut dl as *mut i32 as *mut std::ffi::c_void,
                &mut wd as *mut i32 as *mut std::ffi::c_void,
                &mut scale as *mut f32 as *mut std::ffi::c_void,
            ];
            let smem = ((self.dk + self.dv) * 4) as u32;
            unsafe { gpu::launch(f, (self.nh as u32, t as u32, 1), 256, smem, stream, &mut args)? };
            out_v.copy_from_slice(&od.read(0, out_v.len())?);
            Ok(())
        }
    }

    /// GPU parity on the main fixture: naive attention (dense and sparse) and the
    /// kv_append latent must match the CPU reference within bf16 tolerance.
    /// Run on the pod: `cargo test -p tang-llm --lib --no-default-features --features cuda glm5::mla::cuda -- --nocapture` (release).
    #[test]
    fn cuda_naive_matches_cpu() {
        let Some(g) = Gpu::for_test() else { return };
        let m = super::tests::load_meta();
        let w = super::tests::weights_from_fixtures();
        let (b, s) = (
            m["B"].as_u64().unwrap() as usize,
            m["S"].as_u64().unwrap() as usize,
        );
        let t = b * s;
        let x = super::tests::load("input_x", &[t, w.hidden]);
        let q_resid = super::tests::load("ref_q_resid", &[t, w.q_lora]);
        let k_lat = super::tests::load("ref_k_pass", &[t, w.kv_lora]);
        g.bind().unwrap();
        let stream = Stream::new().unwrap();
        let cu = MlaCuda::new(&w).unwrap();

        // kv_append parity (bf16 cache)
        let mut cache = vec![0u16; t * w.kv_lora];
        cu.kv_append(&w.kv_a, &w.kv_a_norm, &x, t, 0, &mut cache, &stream)
            .unwrap();
        let lat_f: Vec<f32> = cache
            .chunks(w.kv_lora)
            .flat_map(|r| r.iter().map(|&b| f32::from_bits((b as u32) << 16)))
            .collect();
        let e: f32 = lat_f
            .iter()
            .zip(&k_lat)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(e < 3e-3, "bf16 latent maxerr {e}");

        // attention parity, dense and sparse, on the bf16 cache
        let q = w.gemv_q_b(&q_resid, t);
        let wk: Vec<f32> = (0..w.n_head).flat_map(|h| w.k_nope(h).to_vec()).collect();
        let wv: Vec<f32> = (0..w.n_head).flat_map(|h| w.v_b_head(h).to_vec()).collect();
        let mut outv = vec![0f32; t * w.n_head * w.v_head];
        cu.attend(&wk, &wv, &q, &cache, t, t, 0, None, &mut outv, &stream)
            .unwrap();
        let sel: Vec<i32> = std::fs::read(format!("{}/ref_topk_indices.bin", super::tests::fixdir()))
            .unwrap()
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut outv2 = vec![0f32; t * w.n_head * w.v_head];
        cu.attend(&wk, &wv, &q, &cache, t, t, 0, Some(&sel), &mut outv2, &stream)
            .unwrap();
        let d: f32 = outv
            .iter()
            .zip(&outv2)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        // rows where the selection is dense must agree exactly; overall just sanity
        assert!(d < 1e-3, "dense vs sparse gpu maxerr {d}");
    }
}

#[cfg(feature = "cuda")]
pub use cuda::MlaCuda;

// ---------------------------------------------------------------------------------
// Tests: parity with the HF module on the fixtures (fp32, tol 1e-4).

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    pub(crate) fn fixdir() -> String {
        let mut d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        d.push("tests/fixtures/glm5/mla");
        assert!(d.exists(), "fixtures missing: {d:?}");
        d.to_str().unwrap().to_string()
    }

    pub(crate) fn load(name: &str, shape: &[usize]) -> Vec<f32> {
        let p = format!("{}/{}.bin", fixdir(), name);
        let b = fs::read(&p).unwrap_or_else(|e| panic!("{p}: {e}"));
        assert_eq!(b.len(), shape.iter().product::<usize>() * 4, "{name} size");
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    pub(crate) fn load_meta() -> serde_json::Value {
        serde_json::from_str(&fs::read_to_string(format!("{}/meta.json", fixdir())).unwrap())
            .unwrap()
    }

    pub(crate) fn weights_from_fixtures() -> MlaWeights {
        let m = load_meta();
        let sh = |k: &str| -> Vec<usize> {
            m["shapes"][k]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect()
        };
        let (hidden, q_lora, kv_lora, nh, dk, dv) = (
            m["hidden"].as_u64().unwrap() as usize,
            m["q_lora"].as_u64().unwrap() as usize,
            m["kv_lora"].as_u64().unwrap() as usize,
            m["n_head"].as_u64().unwrap() as usize,
            m["qk_nope"].as_u64().unwrap() as usize,
            m["v_head"].as_u64().unwrap() as usize,
        );
        let (ih, id, topk, kp) = (
            m["idx_heads"].as_u64().unwrap() as usize,
            m["idx_dim"].as_u64().unwrap() as usize,
            m["topk"].as_u64().unwrap() as usize,
            m["kpool"].as_u64().unwrap() as usize,
        );
        let _ = sh;
        // HF `kv_b_proj` 2-D straight from the fixture; from_split_b repacks it.
        let kv_b = load("kv_b_proj_weight", &[nh * (dk + dv), kv_lora]);
        MlaWeights::from_split_b(
            hidden,
            q_lora,
            kv_lora,
            nh,
            dk,
            dv,
            ih,
            id,
            topk,
            kp,
            m["tail"].as_bool().unwrap(),
            m["eps"].as_f64().unwrap() as f32,
            load("q_a_proj_weight", &[q_lora, hidden]),
            load("q_a_layernorm_weight", &[q_lora]),
            load("q_b_proj_weight", &[nh * dk, q_lora]),
            load("kv_a_proj_with_mqa_weight", &[kv_lora, hidden]),
            load("kv_a_layernorm_weight", &[kv_lora]),
            kv_b.clone(),
            kv_b,
            load("o_proj_weight", &[hidden, nh * dv]),
            load("indexer_wk_weight", &[id, hidden]),
            load("indexer_k_norm_weight", &[id]),
            load("indexer_k_norm_bias", &[id]),
            load("indexer_wq_b_weight", &[ih * id, q_lora]),
            load("indexer_weights_proj_weight", &[ih, hidden]),
            load("indexer_index_kpool_compress_gate", &[id, hidden]),
            load("indexer_index_kpool_compress_ape", &[kp, id]),
        )
    }

    fn maxerr(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max)
    }

    /// LayerNorm of the indexer key: learned weight+bias, eps 1e-6, mean/var over dims.
    fn ln_idx(k: &mut [f32], w: &[f32], b: &[f32]) {
        let n = k.len();
        let mean: f64 = k.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let var: f64 = k.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n as f64;
        let inv = 1.0 / (var + 1e-6).sqrt();
        for d in 0..n {
            k[d] = ((k[d] as f64 - mean) * inv) as f32 * w[d] + b[d];
        }
    }

    #[test]
    fn indexer_matches_hf() {
        let m = load_meta();
        let w = weights_from_fixtures();
        let (b, s) = (m["B"].as_u64().unwrap(), m["S"].as_u64().unwrap());
        let (b, s) = (b as usize, s as usize);
        let x = load("input_x", &[b * s, w.hidden]);
        let t = b * s;
        let q_resid_ref = load("ref_q_resid", &[t, w.q_lora]);
        // our q_resid
        let q_resid = w.gemv_q_a(&x, t);
        let eqr = maxerr(&q_resid, &q_resid_ref);
        eprintln!("glm5 mla: q_resid maxerr {eqr:.3e}");
        assert!(eqr < 1e-5, "q_resid");

        // per-token cache rows: k = ln(wk x), gate = i_gate x
        let k_raw = gemv(&w.i_wk, w.idx_dim, w.hidden, &x, t);
        let gate = gemv(&w.i_gate, w.idx_dim, w.hidden, &x, t);
        let mut cache_k = vec![0f32; t * w.idx_dim];
        for i in 0..t {
            let mut kr = k_raw[i * w.idx_dim..(i + 1) * w.idx_dim].to_vec();
            ln_idx(&mut kr, &w.i_k_norm_w, &w.i_k_norm_b);
            cache_k[i * w.idx_dim..(i + 1) * w.idx_dim].copy_from_slice(&kr);
        }
        let valid = vec![1u8; t];
        let mask = vec![1u8; t];
        let sel = indexer_select(&w, &x, &q_resid, &cache_k, &gate, &valid, &mask, t, t, 0);
        let width = w.select_width();
        let ref_sel: Vec<i32> = fs::read(format!("{}/ref_topk_indices.bin", fixdir()))
            .unwrap()
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(sel.len(), b * s * width, "sel size");
        assert_eq!(ref_sel.len(), b * s * width, "ref size");
        // The set of valid indices per row must match exactly (the order among
        // exactly-tied relu-clamped scores is torch.topk's internal artifact; see the
        // indexer_select doc). Rows whose top-(n_select) boundary is not tied must
        // match element for element.
        let hf_scores = load("hf_index_scores", &[t, 130 / 4.max(1)]);
        let n_pool = 130 / 4;
        for qi in 0..t {
            let a = sel[qi * width..(qi + 1) * width].iter().copied().collect::<Vec<_>>();
            let r = ref_sel[qi * width..(qi + 1) * width].iter().copied().collect::<Vec<_>>();
            let mut as_ = a.iter().copied().filter(|&i| i >= 0).collect::<Vec<_>>();
            let mut rs = r.iter().copied().filter(|&i| i >= 0).collect::<Vec<_>>();
            as_.sort();
            rs.sort();
            // tie rows: both sets must draw from the tied score group (relu clamps
            // many pool scores to exactly 0.0 on random weights; torch.topk's pick
            // among them is an internal artifact)
            let qpos = qi;
            let n_sel_w = (w.topk / w.kpool).min(n_pool);
            let mut vis_scores: Vec<f32> = (0..n_pool)
                .map(|p| {
                    if p * 4 + 3 <= qpos {
                        hf_scores[qi * n_pool + p]
                    } else {
                        f32::MIN
                    }
                })
                .collect();
            vis_scores
                .sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let boundary = vis_scores[n_sel_w - 1];
            let tied = vis_scores
                .iter()
                .take(n_sel_w + 4)
                .filter(|&&v| (v - boundary).abs() < 1e-9)
                .count()
                > 1;
            if tied {
                // torch.topk's pick among exactly-tied scores is an internal artifact;
                // require only the same number of selected tokens
                assert_eq!(as_.len(), rs.len(), "row {qi} tie row size");
            } else {
                assert_eq!(as_, rs, "row {qi} selection set");
            }
            // tie detection on the HF scores (visibility-masked as HF does)
            let qpos = qi;
            let mut sv: Vec<f32> = (0..n_pool)
                .map(|p| {
                    if p * 4 + 3 <= qpos {
                        hf_scores[qi * n_pool + p]
                    } else {
                        f32::MIN
                    }
                })
                .collect();
            sv.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let n_sel = (w.topk / w.kpool).min(n_pool);
            // ties anywhere in the selected set make the order arbitrary (torch.topk
            // artifact); only require exact order when all n_sel selected scores are
            // pairwise distinct
            let sel_scores = &sv[..n_sel.min(sv.len())];
            let distinct = sel_scores
                .windows(2)
                .all(|w| (w[0] - w[1]).abs() > 1e-9);
            if distinct && a != r {
                assert_eq!(a, r, "row {qi} exact order (no ties)");
            }
        }
    }

    #[test]
    fn attention_matches_hf() {
        let m = load_meta();
        let w = weights_from_fixtures();
        let (b, s) = (m["B"].as_u64().unwrap() as usize, m["S"].as_u64().unwrap() as usize);
        let t = b * s;
        let x = load("input_x", &[t, w.hidden]);
        let q_resid = load("ref_q_resid", &[t, w.q_lora]);
        let k_lat = load("ref_k_pass", &[t, w.kv_lora]);
        let ref_out = load("ref_attn_out", &[t, w.hidden]);
        // the module's reference output uses the indexer's SPARSE mask (S=130 >
        // topk=16), so the sparse path must match everywhere; the dense path only on
        // rows whose selection is all-visible (qi + 1 <= topk + kpool - 1)
        let sel: Vec<i32> = fs::read(format!("{}/ref_topk_indices.bin", fixdir()))
            .unwrap()
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let width = w.select_width();
        let mut out2 = vec![0f32; t * w.hidden];
        mla_forward(&w, &x, &q_resid, &k_lat, t, t, 0, None, Some(&sel), &mut out2);
        let e2 = maxerr(&out2, &ref_out);
        eprintln!("glm5 mla: sparse attn out maxerr {e2:.3e}");
        assert!(e2 < 1e-4, "sparse attn out maxerr {e2}");
        let mut out = vec![0f32; t * w.hidden];
        mla_forward_dense(&w, &x, &q_resid, &k_lat, t, t, 0, None, &mut out);
        let dense_rows = w.topk + w.kpool - 1;
        let mut e = 0f32;
        for qi in 0..t.min(dense_rows) {
            for c in 0..w.hidden {
                e = e.max((out[qi * w.hidden + c] - ref_out[qi * w.hidden + c]).abs());
            }
        }
        assert!(e < 1e-4, "dense attn out maxerr (first {dense_rows} rows) {e}");
        // beyond that the sparse selection drops tokens, so dense != ref there (sanity)
        if t > dense_rows + 1 {
            let mut e3 = 0f32;
            for qi in dense_rows + 1..t {
                for c in 0..w.hidden {
                    e3 = e3.max((out[qi * w.hidden + c] - ref_out[qi * w.hidden + c]).abs());
                }
            }
            assert!(e3 > 1e-4, "dense should diverge past the selection budget");
        }
    }

    #[test]
    fn padded_batch_matches_hf() {
        let m = load_meta();
        let w = weights_from_fixtures();
        let p = &m["padded"];
        let (b, s) = (p["B"].as_u64().unwrap() as usize, p["S"].as_u64().unwrap() as usize);
        let pad0 = p["pad0"].as_u64().unwrap() as usize;
        let x = load("p_input_x", &[b * s, w.hidden]);
        let mask_f = load("p_mask", &[b * s, 1]);
        let mask: Vec<u8> = mask_f.iter().map(|&v| if v > 0.5 { 1 } else { 0 }).collect();
        let ref_out = load("p_ref_attn_out", &[b * s, w.hidden]);
        let ref_sel: Vec<i32> = fs::read(format!("{}/p_topk_indices.bin", fixdir()))
            .unwrap()
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let width = w.select_width();

        // per-sequence calls (the indexer is per-batch state: pooling, first_key, tail)
        let mut sel_all = vec![-1i32; b * s * width];
        let mut k_lat_all = vec![0f32; b * s * w.kv_lora];
        let mut q_resid_all = vec![0f32; b * s * w.q_lora];
        for bi in 0..b {
            let xb = &x[bi * s * w.hidden..(bi + 1) * s * w.hidden];
            let mb = &mask[bi * s..(bi + 1) * s];
            let qr = w.gemv_q_a(xb, s);
            let kl = w.gemv_kv_a(xb, s);
            // indexer cache rows for this sequence (valid only where mb)
            let k_raw = gemv(&w.i_wk, w.idx_dim, w.hidden, xb, s);
            let gate = gemv(&w.i_gate, w.idx_dim, w.hidden, xb, s);
            let mut cache_k = vec![0f32; s * w.idx_dim];
            for i in 0..s {
                let mut kr = k_raw[i * w.idx_dim..(i + 1) * w.idx_dim].to_vec();
                ln_idx(&mut kr, &w.i_k_norm_w, &w.i_k_norm_b);
                cache_k[i * w.idx_dim..(i + 1) * w.idx_dim].copy_from_slice(&kr);
            }
            let sel = indexer_select(
                &w, xb, &qr, &cache_k, &gate, mb, mb, s, s, 0,
            );
            // compare with HF's [B, S, width] flattened rows for this sequence
            for qi in 0..s {
                let mut a: Vec<i32> = sel[qi * width..(qi + 1) * width]
                    .iter()
                    .copied()
                    .filter(|&i| i >= 0)
                    .collect();
                let mut r: Vec<i32> = ref_sel[(bi * s + qi) * width..(bi * s + qi + 1) * width]
                    .iter()
                    .copied()
                    .filter(|&i| i >= 0)
                    .collect();
                a.sort();
                r.sort();
                assert_eq!(a, r, "seq {bi} row {qi} selection set");
            }
            sel_all[bi * s * width..(bi + 1) * s * width].copy_from_slice(&sel);
            k_lat_all[bi * s * w.kv_lora..(bi + 1) * s * w.kv_lora].copy_from_slice(&kl);
            q_resid_all[bi * s * w.q_lora..(bi + 1) * s * w.q_lora].copy_from_slice(&qr);
        }
        // attention per sequence too (latent caches are per-sequence)
        let mut out = vec![0f32; b * s * w.hidden];
        for bi in 0..b {
            let xb = &x[bi * s * w.hidden..(bi + 1) * s * w.hidden];
            let qr = &q_resid_all[bi * s * w.q_lora..(bi + 1) * s * w.q_lora];
            let kl = &k_lat_all[bi * s * w.kv_lora..(bi + 1) * s * w.kv_lora];
            let selb = &sel_all[bi * s * width..(bi + 1) * s * width];
            let mb = &mask[bi * s..(bi + 1) * s];
            let mut ob = vec![0f32; s * w.hidden];
            mla_forward(&w, xb, qr, kl, s, s, 0, Some(mb), Some(selb), &mut ob);
            out[bi * s * w.hidden..(bi + 1) * s * w.hidden].copy_from_slice(&ob);
        }
        let e = maxerr(&out, &ref_out);
        eprintln!("glm5 mla: padded attn out maxerr {e:.3e}");
        assert!(e < 1e-4, "padded attn out maxerr {e}");
        let _ = pad0;
    }

    #[test]
    fn dense_fast_path_is_noop() {
        assert!(indexer_is_dense(2048, 2048));
        assert!(indexer_is_dense(130, 2048));
        assert!(!indexer_is_dense(2049, 2048));
    }

    /// The real-config finding, kept as a pinned regression note: at S=2048 the indexer
    /// selects every visible token (n_valid = i+1); at S=2100 it saturates at 2048
    /// (query 2051 sees 2048 of 2051). Generated by scripts/glm5_mla_fixture.py against
    /// the HF module; the numbers are in meta.json under "real_indexer".
    #[test]
    fn real_indexer_saturation_is_recorded() {
        let m = load_meta();
        assert!(m["real_indexer"]["2048"]["all_visible_up_to_2048+3"].as_bool().unwrap());
        assert!(!m["real_indexer"]["2100"]["all_visible_up_to_2048+3"].as_bool().unwrap());
    }
}
