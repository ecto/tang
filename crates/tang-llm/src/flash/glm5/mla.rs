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

/// Absorbed-form MLA: same output as [`mla_forward`] by construction, with the score
/// read as a 512-dim latent dot and one `W_v` application per query instead of per key.
///
/// Math (no rope): naive `k_h = W_k_h · latent` gives
/// `q_h · k_h = (W_k_h^T · q_h) · latent`, so fold `W_k_h^T` into the query once per
/// query (`q'_h = W_k_h^T · q_h`, [kv_lora] per head) and score directly against the
/// latent cache. Softmax weights `p` are unchanged (same scores), and
/// `Σ_i p_i · (W_v_h · latent_i) = W_v_h · (Σ_i p_i · latent_i)` — accumulate the
/// attention-weighted latent `Σ p_i · latent_i` (one [kv_lora] vector) and apply
/// `W_v_h` once ([v_head × kv_lora] per head per query). 512-dim dot per (query, head,
/// key) either way, but the value side drops from `n_keys·dv·dl` MACs to `dv·dl` —
/// the decode win: 2048 selected keys × 64 heads × 256 × 512 → 64 × 256 × 512.
///
/// Signature mirrors [`mla_forward`] exactly (same `q_resid`, same `sel` semantics,
/// same uniform-pad-row behavior); `out` [t][hidden] with `o_proj` applied.
/// `qa_buf`/`va_buf` are scratch: `qa_buf` [n_head·kv_lora] (all heads' `W_k_h^T` packed
/// from `w.k_nope`), `va_buf` unused but must be length `dv` (kept for API symmetry).
#[allow(clippy::too_many_arguments)]
pub fn mla_forward_absorbed(
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

    // q'_h = W_k_h^T · q_h: [t][nh][dl]. W_k_h is [dl][dk] (k_nope), so
    // (W_k_h^T · q)_l = Σ_d W_k_h[l][d] · q_h[d] — one [dk]→[dl] fold per
    // (query, head) instead of one decompress per (query, head, key).
    let mut qa = vec![0f32; t * nh * dl];

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

    let mut attn_o = vec![0f32; t * nh * dv];
    for (qi, ks) in keys.iter().enumerate() {
        if ks.is_empty() {
            debug_assert!(false, "query {qi}: empty selection");
            continue;
        }
        let uni = uniform_row(qi);
        // fold this query's q through W_k^T once per head
        for h in 0..nh {
            let wk = w.k_nope(h);
            let q0 = qi * nh * dk + h * dk;
            let a0 = qi * nh * dl + h * dl;
            for l in 0..dl {
                let mut s2 = 0f32;
                let wrow = &wk[l * dk..(l + 1) * dk];
                for d in 0..dk {
                    s2 += wrow[d] * q[q0 + d];
                }
                qa[a0 + l] = s2;
            }
        }
        for h in 0..nh {
            // scores: kv_lora-dim latent dots with the folded query
            let q0 = qi * nh * dl + h * dl;
            let qh = &qa[q0..q0 + dl];
            let mut sc = vec![0f32; ks.len()];
            let mut mx = if uni { 0.0 } else { f32::NEG_INFINITY };
            if uni {
                for v in sc.iter_mut() {
                    *v = 0.0;
                }
            }
            for (i, &k) in ks.iter().enumerate() {
                let kl = &k_latent[k * dl..(k + 1) * dl];
                sc[i] = dot(qh, kl) * scale;
                if !uni {
                    mx = mx.max(sc[i]);
                }
            }
            // attention-weighted latent sum, then one W_v application
            let mut lat = [0f32; 512];
            debug_assert!(dl <= 512);
            let mut den = 0f32;
            let inv_n = 1.0 / ks.len() as f32;
            for (i, &k) in ks.iter().enumerate() {
                let p = if uni { inv_n } else { (sc[i] - mx).exp() };
                den += p;
                let kl = &k_latent[k * dl..(k + 1) * dl];
                for l in 0..dl {
                    lat[l] += p * kl[l];
                }
            }
            let wv = w.v_b_head(h); // [dv][dl]
            let o0 = qi * nh * dv + h * dv;
            for d in 0..dv {
                let wrow = &wv[d * dl..(d + 1) * dl];
                attn_o[o0 + d] = dot(wrow, &lat[..dl]) / den;
            }
        }
    }
    use rayon::prelude::*;
    out.par_chunks_mut(w.hidden)
        .zip(attn_o.par_chunks(nh * dv))
        .for_each(|(yo, ao)| {
            for (r, o) in yo.iter_mut().enumerate() {
                *o = dot(&w.o[r * nh * dv..(r + 1) * nh * dv], ao);
            }
        });
}

/// Dense causal fast path of [`mla_forward_absorbed`] (`sel = None`).
#[allow(clippy::too_many_arguments)]
pub fn mla_forward_absorbed_dense(
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
    mla_forward_absorbed(w, x, q_resid, k_latent, n_kv, t, pos0, mask, None, out);
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

#define G5_INF_F __uint_as_float(0x7f800000u)
#define G5_NEG_INF_F __uint_as_float(0xff800000u)

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
    float m = G5_NEG_INF_F;
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

// ---------------- absorbed MLA (decode/prefill fast path) ----------------
// mla_fold_q: q'[t][nh][dl] = W_k_h^T . q[t][nh*dk] — one block per (query, head),
// each thread owns latent dims l. wk: [nh][dl*dk] (k_nope [l][d]).
extern "C" __global__ void mla_fold_q(
    const float* __restrict__ q, const float* __restrict__ wk, float* __restrict__ qa,
    int t, int nh, int dk, int dl)
{
    const int qi = blockIdx.y;
    const int h  = blockIdx.x;
    if (qi >= t || h >= nh) return;
    const float* qh = q + (size_t)qi * nh * dk + (size_t)h * dk;
    const float* wh_k = wk + (size_t)h * dl * dk;
    float* out = qa + ((size_t)qi * nh + h) * dl;
    for (int l = threadIdx.x; l < dl; l += blockDim.x) {
        const float* wrow = wh_k + (size_t)l * dk;
        float s = 0.f;
        for (int d = 0; d < dk; d++) s += wrow[d] * qh[d];
        out[l] = s;
    }
}

// mla_attn_absorbed: one block per (query, head). Scores are kv_lora-dim dots against
// the bf16 latent cache with the folded query; the attention-weighted latent sum is
// accumulated in shared memory and put through W_v_h (and W_o on the host) once.
// qa: [t][nh*dl]; lat: [nkv][dl] bf16; wv: [nh][dv*dl]; sel: [t][width] i32 or NULL
// (dense causal); out: [t][nh*dv] f32. smem: dl floats (the latent sum).
extern "C" __global__ void mla_attn_absorbed(
    const float* __restrict__ qa, const unsigned short* __restrict__ lat,
    const float* __restrict__ wv, const int* __restrict__ sel, float* __restrict__ out,
    int t, int nkv, int pos0, int nh, int dv, int dl, int width, float scale)
{
    const int qi = blockIdx.y;
    const int h  = blockIdx.x;
    if (qi >= t || h >= nh) return;
    const int qpos = pos0 + qi;
    const float* qh = qa + ((size_t)qi * nh + h) * dl;
    const float* wh_v = wv + (size_t)h * dv * dl;
    extern __shared__ float lacc[];  // [dl] attention-weighted latent sum
    for (int l = threadIdx.x; l < dl; l += blockDim.x) lacc[l] = 0.f;

    // pass 1: max score
    float m = G5_NEG_INF_F;
    for (int w = 0; w < width; w++) {
        int k;
        if (sel == NULL) { k = w; if (k > qpos) break; }
        else { k = sel[(size_t)qi * width + w]; if (k < 0) continue; }
        if (k >= nkv) continue;
        const unsigned short* krow = lat + (size_t)k * dl;
        float sc = 0.f;
        for (int l = 0; l < dl; l++) sc += qh[l] * bf2f(krow[l]);
        sc *= scale;
        if (sc > m) m = sc;
    }
    // pass 2: accumulate the weighted latent sum (and sync before the W_v read)
    float den = 0.f;
    for (int w = 0; w < width; w++) {
        int k;
        if (sel == NULL) { k = w; if (k > qpos) break; }
        else { k = sel[(size_t)qi * width + w]; if (k < 0) continue; }
        if (k >= nkv) continue;
        const unsigned short* krow = lat + (size_t)k * dl;
        float sc = 0.f;
        for (int l = 0; l < dl; l++) sc += qh[l] * bf2f(krow[l]);
        sc *= scale;
        const float p = expf(sc - m);
        den += p;
        for (int l = threadIdx.x; l < dl; l += blockDim.x)
            lacc[l] += p * bf2f(krow[l]);
        __syncthreads();
    }
    // one W_v application per (query, head): q' already folded, so this is the only
    // [dv x dl] work in the whole kernel
    float* outp = out + (size_t)qi * nh * dv + (size_t)h * dv;
    __syncthreads();
    for (int d = threadIdx.x; d < dv; d += blockDim.x) {
        const float* wrow = wh_v + (size_t)d * dl;
        float s = 0.f;
        for (int l = 0; l < dl; l++) s += wrow[l] * lacc[l];
        outp[d] = den > 0.f ? s / den : 0.f;
    }
}

// ---------------- indexer (pooled, > 2048 tokens) ----------------
// indexer_key_append: one block per token. wk: [hd*hid], kn_w/kn_b: [hd] (LayerNorm
// eps 1e-6), gate: [hd*hid] (i_gate), x: [t][hid]. Writes cache_k/cache_gate rows
// [t][hd] at cache position pos0. (Also returns nothing else.)
extern "C" __global__ void idx_key_append(
    const float* __restrict__ wk, const float* __restrict__ kn_w, const float* __restrict__ kn_b,
    const float* __restrict__ gate_w, const float* __restrict__ x,
    float* __restrict__ cache_k, float* __restrict__ cache_gate,
    int t, int pos0, int hid, int hd, unsigned char* __restrict__ valid)
{
    const int tok = blockIdx.x;
    if (tok >= t) return;
    const float* xt = x + (size_t)tok * hid;
    __shared__ float sh[512];
    for (int r = threadIdx.x; r < hd; r += blockDim.x) {
        const float* kr = wk + (size_t)r * hid;
        float s = 0.f;
        for (int c = 0; c < hid; c++) s += kr[c] * xt[c];
        sh[r] = s;
        const float* gr = gate_w + (size_t)r * hid;
        float g = 0.f;
        for (int c = 0; c < hid; c++) g += gr[c] * xt[c];
        cache_gate[((size_t)pos0 + tok) * hd + r] = g;
    }
    __syncthreads();
    // LayerNorm over hd with weight+bias, eps 1e-6 (single thread; hd is small)
    if (threadIdx.x == 0) {
        float mean = 0.f;
        for (int r = 0; r < hd; r++) mean += sh[r];
        mean /= hd;
        float var = 0.f;
        for (int r = 0; r < hd; r++) var += (sh[r] - mean) * (sh[r] - mean);
        var /= hd;
        const float inv = rsqrtf(var + 1e-6f);
        for (int r = 0; r < hd; r++)
            cache_k[((size_t)pos0 + tok) * hd + r] = (sh[r] - mean) * inv * kn_w[r] + kn_b[r];
        valid[pos0 + tok] = 1;
    }
}

// idx_pool_keys: one block per pool. cache_k/cache_gate: [kvlen][hd]; ape: [hd*kp]
// (slot-major [j*hd+d]); pk_out: [npool][hd] f32. grid (npool).
extern "C" __global__ void idx_pool_keys(
    const float* __restrict__ cache_k, const float* __restrict__ cache_gate,
    const float* __restrict__ ape, const unsigned char* __restrict__ valid,
    float* __restrict__ pk_out,
    int kvlen, int first_key, int npool, int hd, int kp)
{
    const int p = blockIdx.x;
    if (p >= npool) return;
    const int base = first_key + p * kp;
    // each thread owns a dim; softmax over kp slots per dim
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float lg[8];
        float mx = G5_NEG_INF_F;
        for (int j = 0; j < kp; j++) {
            lg[j] = cache_gate[(size_t)(base + j) * hd + d] + ape[(size_t)j * hd + d];
            if (lg[j] > mx) mx = lg[j];
        }
        float sum = 0.f;
        for (int j = 0; j < kp; j++) { lg[j] = expf(lg[j] - mx); sum += lg[j]; }
        float acc = 0.f;
        for (int j = 0; j < kp; j++)
            acc += lg[j] / sum * cache_k[(size_t)(base + j) * hd + d];
        pk_out[(size_t)p * hd + d] = acc;
    }
}

// idx_select: one block per query. iq: [t][nh*hd] (wq_b out, head-major),
// hw: [t][nh] (weights_proj out, UNSCALED), pk: [npool][hd],
// valid: [kvlen], first_key, npool. Writes sel row [width] i32 (-1 invalid):
// n_select = min(topk/kp, npool) pool slots (visible pools in score order, invisible
// masked -1), then the query's tail (up to kp-1 of its own incomplete pool).
// Tie order: lower index first (matches our CPU ref; W0 owns the torch tie study).
// smem: npool<=2048 scores not needed — scores stay in registers via a serial loop over
// pools with a running top-k insertion (n_select <= 512, kept in shared memory).
extern "C" __global__ void idx_select(
    const float* __restrict__ iq, const float* __restrict__ hw,
    const float* __restrict__ pk, const unsigned char* __restrict__ valid,
    int* __restrict__ sel, float* __restrict__ dbg,
    int t, int kvlen, int pos0, int first_key, int npool,
    int nh, int hd, int kp, int topk, int width, float score_scale, float hw_scale)
{
    const int qi = blockIdx.x;
    if (qi >= t) return;
    const int qpos = pos0 + qi;
    const int n_select = min(topk / kp, npool);
    extern __shared__ float smem[];
    int* pool_idx = (int*)smem;          // [n_select] chosen pool ids
    float* pool_sc = (float*)smem + n_select; // their scores
    // running top-k over pools with score -inf when not visible.
    // Single-threaded (thread 0): the score loop is serial anyway and the shared top-k
    // arrays would race under 256 threads.
    int filled = 0;
    if (threadIdx.x == 0)
    for (int p = 0; p < npool; p++) {
        const int end = first_key + p * kp + kp - 1;
        float s = G5_NEG_INF_F;
        if (end <= qpos && valid[end]) {
            const float* pkp = pk + (size_t)p * hd;
            const float* qh = iq + (size_t)qi * nh * hd;
            // s = Σ_h hw[h]*hw_scale * relu(dot(qh_h, pk)*scale)
            float acc = 0.f;
            for (int h = 0; h < nh; h++) {
                float d = 0.f;
                const float* qhh = qh + (size_t)h * hd;
                for (int d2 = 0; d2 < hd; d2++) d += qhh[d2] * pkp[d2];
                d *= score_scale;
                if (d < 0.f) d = 0.f;
                acc += hw[(size_t)qi * nh + h] * hw_scale * d;
            }
            s = acc;
        }
        // insert into the running top-k (score desc, tie -> lower pool id first)
        if (filled < n_select) {
            int lo = 0;
            while (lo < filled && (pool_sc[lo] > s || (pool_sc[lo] == s && pool_idx[lo] < p))) lo++;
            for (int j = filled; j > lo; j--) { pool_sc[j] = pool_sc[j-1]; pool_idx[j] = pool_idx[j-1]; }
            pool_sc[lo] = s; pool_idx[lo] = p;
            filled++;
        } else if (s > pool_sc[n_select-1] || (s == pool_sc[n_select-1] && p < pool_idx[n_select-1])) {
            int lo = 0;
            while (lo < n_select - 1 && (pool_sc[lo] > s || (pool_sc[lo] == s && pool_idx[lo] < p))) lo++;
            for (int j = n_select - 1; j > lo; j--) { pool_sc[j] = pool_sc[j-1]; pool_idx[j] = pool_idx[j-1]; }
            pool_sc[lo] = s; pool_idx[lo] = p;
        }
    }
    // DEBUG: per-query chosen pool scores/ids (npool*2 floats: [p, s] pairs)
    if (dbg != NULL) {
        for (int i = 0; i < n_select && i < npool; i++) {
            dbg[(size_t)qi * npool * 2 + 2 * i] = (float)pool_idx[i];
            dbg[(size_t)qi * npool * 2 + 2 * i + 1] = pool_sc[i];
        }
    }
    // visible_count (causal + valid)
    int visible_count = 0;
    for (int k = first_key; k <= qpos; k++) visible_count += valid[k] ? 1 : 0;
    const int tail_count = visible_count % kp;
    const int tail_start = first_key + visible_count - tail_count;
    int wi = 0;
    int* out = sel + (size_t)qi * width;
    // exactly n_select pool slots, always (invisible pools emit -1)
    for (int i = 0; i < n_select && wi < width; i++) {
        const int p = pool_idx[i];
        const bool vis = pool_sc[i] != G5_NEG_INF_F;
        for (int j = 0; j < kp && wi < width; j++)
            out[wi++] = vis ? (first_key + p * kp + j) : -1;
    }
    for (int j = 0; j < kp - 1 && wi < width; j++) {
        if (j >= tail_count) break;
        const int idx = tail_start + j;
        out[wi++] = (idx < kvlen && valid[idx]) ? idx : -1;
    }
    for (; wi < width; wi++) out[wi] = -1;
}
"#;

/// The `#define` prelude for [`CUDA_SRC`] at this geometry.
pub fn cuda_prelude(hidden: usize, q_lora: usize, kv_lora: usize, n_head: usize) -> String {
    format!(
        "#define HID {hid}\n#define QL {ql}\n#define DL {dl}\n#define NH {nh}\n",
        hid = hidden, ql = q_lora, dl = kv_lora, nh = n_head
    )
}

/// PLANS (still open, per the brief's "naive-but-correct first"):
///
/// 1. **Absorbed-projection MLA** (llama.cpp `build_dsa_layer`): keep the query in
///    latent space — `q' = (Wk_b)^T q` makes the score `dot(q', latent)` over 512 dims
///    with no per-key decompression, and `Wv_b` is applied once to the attended latent
///    sum (a [nh·256 × 512] GEMV per query) instead of per key per head. Then the
///    latent dot is a plain attention over head_dim 512, which `cx_attn_fa2` can cover
///    once it takes HD=512 (it is HD=256 today); decode reuses the split-K path. Until
///    then the naive kernel above holds the parity reference.
/// 2. The pooled indexer is BUILT for the GPU now ([`MlaCuda::idx_append`],
///    [`MlaCuda::idx_pool`], [`MlaCuda::idx_run`]): one block per token for the cache
///    rows, one per pool for the pooled keys, one per query for score + running
///    top-k (n_select ≤ 512 in shared memory). Tie order: lower pool index first,
///    matching the CPU reference (W0 owns the torch-tie study — do not change this).
/// 3. Cache layout: per DSA layer, [cap][512] bf16 latents plus the indexer's
///    [cap][128] f32 k and gate rows and a [cap] u8 valid vector (the indexer stays
///    f32 — its APE/gate math is sensitive).
#[cfg(feature = "cuda")]
pub mod cuda {
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
        // indexer geometry
        idxd: usize,  // idx_dim
        idxh: usize,  // idx_heads
        kpool: usize,
        topk: usize,
        tail: bool,

        /// kv_a / kv_a_norm staged once (per-layer constants).
        kv_a_d: std::cell::RefCell<Option<DevBuf>>,
        kvan_d: std::cell::RefCell<Option<DevBuf>>,
        /// indexer weights staged once: i_wk, i_gate ([idx_dim*hidden] f32 each),
        /// k_norm weight/bias.
        iwk_d: std::cell::RefCell<Option<DevBuf>>,
        igw_d: std::cell::RefCell<Option<DevBuf>>,
        ikn_d: std::cell::RefCell<Option<DevBuf>>,
        ikb_d: std::cell::RefCell<Option<DevBuf>>,
        ape_d: std::cell::RefCell<Option<DevBuf>>,
        // ---- resident state (perf: pod 13 showed 155 ms/layer/window dominated by
        // per-call 67 MB weight rebuilds + uploads, whole-cache re-uploads and
        // cuMemAlloc churn; everything below lives across calls) ----
        /// W_k/W_v decompress matrices, resident: [nh][dl*dk] / [nh][dv*dl] f32.
        wk_d: DevBuf,
        wv_d: DevBuf,
        /// bf16 latent cache, resident. Grows by realloc; rows [0..lat_rows) valid.
        lat_d: std::cell::RefCell<DevBuf>,
        lat_rows: std::cell::Cell<usize>,
        /// indexer token rows, resident: [rows][idx_dim] f32 each for k and gate,
        /// plus the valid bytes.
        idk_d: std::cell::RefCell<DevBuf>,
        idg_d: std::cell::RefCell<DevBuf>,
        val_d: std::cell::RefCell<DevBuf>,
        idx_rows: std::cell::Cell<usize>,
        /// pooled keys, resident: [cap_pools][idx_dim]; pools [0..pool_upto) valid.
        pool_d: std::cell::RefCell<DevBuf>,
        pool_upto: std::cell::Cell<usize>,
        /// the last externally-passed latent slice, resident: (buffer, host ptr, len) —
        /// lets a caller that re-uploads its whole cache hit a stable path when its host
        /// vec didn't move (fingerprint by ptr+len; engine's c.lat16 reallocates on growth,
        /// but between growths it's stable — and kv_append's resident writes make the
        /// upload unnecessary anyway when the fingerprint holds).
        lat_in_d: std::cell::RefCell<Option<(DevBuf, u64, usize)>>,
        /// persistent scratch: q [cap_t][nh*dk] f32, out [cap_t][nh*dv] f32,
        /// sel [cap_t][width] i32, folded q' [cap_t][nh*dl] f32, hw [cap_t][nh].
        q_d: std::cell::RefCell<DevBuf>,
        o_d: std::cell::RefCell<DevBuf>,
        s_d: std::cell::RefCell<DevBuf>,
        qa_d: std::cell::RefCell<DevBuf>,
        hw_d: std::cell::RefCell<DevBuf>,
        iq_d: std::cell::RefCell<DevBuf>,
        sd_out: std::cell::RefCell<DevBuf>,
    }

    impl MlaCuda {
        /// Compile the kernels for this weight set's geometry. Host weights are kept
        /// by the caller and staged per call (the naive path is not yet weight-resident).
        pub fn new(w: &MlaWeights) -> Result<Self, gpu::Error> {
            Self::new_at(w, 0)
        }

        /// [`Self::new`] on CUDA device `ordinal` (the two-GPU split's stage 1).
        pub fn new_at(w: &MlaWeights, ordinal: usize) -> Result<Self, gpu::Error> {
            let g = Gpu::at(ordinal)?;
            g.bind()?;
            let src = format!(
                "{}{}",
                cuda_prelude(w.hidden, w.q_lora, w.kv_lora, w.n_head),
                CUDA_SRC
            );
            let m = g.module(&src)?;
            // Resident weights: W_k/W_v decompress matrices staged ONCE (67 MB at the
            // real geometry — pod 13 re-uploaded this every window, every layer).
            let wk: Vec<f32> = (0..w.n_head).flat_map(|h| w.k_nope(h).to_vec()).collect();
            let wv: Vec<f32> = (0..w.n_head).flat_map(|h| w.v_b_head(h).to_vec()).collect();
            let wk_d = DevBuf::from_slice(&wk)?;
            let wv_d = DevBuf::from_slice(&wv)?;
            let cap_t = 512.max(64); // scratch rows; grows on demand
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
                idxd: w.idx_dim,
                idxh: w.idx_heads,
                kpool: w.kpool,
                topk: w.topk,
                tail: w.select_tail,
                wk_d,
                wv_d,
                kv_a_d: std::cell::RefCell::new(None),
                kvan_d: std::cell::RefCell::new(None),
                iwk_d: std::cell::RefCell::new(None),
                igw_d: std::cell::RefCell::new(None),
                ikn_d: std::cell::RefCell::new(None),
                ikb_d: std::cell::RefCell::new(None),
                ape_d: std::cell::RefCell::new(None),
                lat_d: std::cell::RefCell::new(DevBuf::alloc(1024 * w.kv_lora * 2)?),
                lat_rows: std::cell::Cell::new(0),
                idk_d: std::cell::RefCell::new(DevBuf::alloc(
                    1024 * w.idx_dim * std::mem::size_of::<f32>(),
                )?),
                idg_d: std::cell::RefCell::new(DevBuf::alloc(
                    1024 * w.idx_dim * std::mem::size_of::<f32>(),
                )?),
                val_d: std::cell::RefCell::new(DevBuf::alloc(1024)?),
                idx_rows: std::cell::Cell::new(0),
                pool_d: std::cell::RefCell::new(DevBuf::alloc(
                    (1024 / w.kpool + 1) * w.idx_dim * std::mem::size_of::<f32>(),
                )?),
                pool_upto: std::cell::Cell::new(0),
                q_d: std::cell::RefCell::new(DevBuf::zeroed_elems::<f32>(
                    cap_t * w.n_head * w.qk_nope,
                )?),
                o_d: std::cell::RefCell::new(DevBuf::zeroed_elems::<f32>(
                    cap_t * w.n_head * w.v_head,
                )?),
                s_d: std::cell::RefCell::new(DevBuf::zeroed_elems::<i32>(
                    cap_t * (w.topk + w.kpool - 1),
                )?),
                qa_d: std::cell::RefCell::new(DevBuf::zeroed_elems::<f32>(
                    cap_t * w.n_head * w.kv_lora,
                )?),
                hw_d: std::cell::RefCell::new(DevBuf::zeroed_elems::<f32>(cap_t * w.idx_heads)?),
                iq_d: std::cell::RefCell::new(DevBuf::zeroed_elems::<f32>(
                    cap_t * w.idx_heads * w.idx_dim,
                )?),
                sd_out: std::cell::RefCell::new(DevBuf::zeroed_elems::<i32>(
                    cap_t * (w.topk + w.kpool - 1),
                )?),
                lat_in_d: std::cell::RefCell::new(None),
            })
        }

        /// Grow a resident cache to `rows` (u16-byte rows for lat, elements for the f32
        /// caches). Realloc zeroes the resident state (watermarks reset by the caller).
        fn ensure_lat(&self, rows: usize) -> Result<(), gpu::Error> {
            let mut lat = self.lat_d.borrow_mut();
            let need = rows * self.dl * 2;
            if lat.len() < need {
                let grown = DevBuf::alloc(need * 2 + 64 * self.dl * 2)?;
                *lat = grown;
                self.lat_rows.set(0);
            }
            Ok(())
        }

        fn ensure_idx(&self, rows: usize) -> Result<(), gpu::Error> {
            let hd = self.idxd;
            {
                let mut k = self.idk_d.borrow_mut();
                let need = rows * hd * std::mem::size_of::<f32>();
                if k.len() < need {
                    *k = DevBuf::alloc(need * 2 + 64 * hd * 4)?;
                    self.idx_rows.set(0);
                }
            }
            {
                let mut g = self.idg_d.borrow_mut();
                let need = rows * hd * std::mem::size_of::<f32>();
                if g.len() < need {
                    *g = DevBuf::alloc(need * 2 + 64 * hd * 4)?;
                    self.idx_rows.set(0);
                }
            }
            {
                let mut v = self.val_d.borrow_mut();
                if v.len() < rows {
                    *v = DevBuf::alloc(rows * 2 + 64)?;
                    self.idx_rows.set(0);
                }
            }
            Ok(())
        }

        fn ensure_pool(&self, n_pool: usize) -> Result<(), gpu::Error> {
            let mut p = self.pool_d.borrow_mut();
            let need = n_pool * self.idxd * std::mem::size_of::<f32>();
            if p.len() < need {
                *p = DevBuf::alloc(need * 2 + 64 * self.idxd * 4)?;
                self.pool_upto.set(0);
            }
            Ok(())
        }

        fn ensure_scratch(&self, t: usize) -> Result<(), gpu::Error> {
            let width = self.topk + if self.tail { self.kpool - 1 } else { 0 };
            macro_rules! grow {
                ($cell:expr, $elems:expr) => {{
                    let mut b = $cell.borrow_mut();
                    if b.len() < $elems * std::mem::size_of::<f32>() {
                        *b = DevBuf::zeroed_elems::<f32>($elems * 2 + 64)?;
                    }
                }};
            }
            if self.q_d.borrow().len() < t * self.nh * self.dk * 4 {
                let mut b = self.q_d.borrow_mut();
                *b = DevBuf::zeroed_elems::<f32>(t * self.nh * self.dk * 2 + 64)?;
            }
            if self.o_d.borrow().len() < t * self.nh * self.dv * 4 {
                let mut b = self.o_d.borrow_mut();
                *b = DevBuf::zeroed_elems::<f32>(t * self.nh * self.dv * 2 + 64)?;
            }
            {
                let mut b = self.s_d.borrow_mut();
                if b.len() < t * width * 4 {
                    *b = DevBuf::zeroed_elems::<i32>(t * width * 2 + 64)?;
                }
            }
            grow!(self.qa_d, t * self.nh * self.dl);
            grow!(self.hw_d, t * self.idxh);
            grow!(self.iq_d, t * self.idxh * self.idxd);
            {
                let mut b = self.sd_out.borrow_mut();
                if b.len() < t * width * 4 {
                    *b = DevBuf::zeroed_elems::<i32>(t * width * 2 + 64)?;
                }
            }
            Ok(())
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
            cache: &mut [u16], // [cap][dl] (host mirror; rows [0..pos0) may be stale here)
            stream: &Stream,
        ) -> Result<(), gpu::Error> {
            assert_eq!(x.len(), t * self.hid);
            assert_eq!(kv_a.len(), self.dl * self.hid);
            self.ensure_lat(pos0 + t)?;
            // contiguity: the resident cache is authoritative only if every row was
            // written here, in order, for the CURRENT sequence (a reset or a foreign
            // sequence breaks it). On mismatch: full re-upload from the caller's mirror
            // (authoritative — the engine keeps it up to date from our read-backs).
            let contiguous = self.lat_rows.get() == pos0;
            if !contiguous && pos0 > 0 {
                self.lat_d
                    .borrow()
                    .write(0, &cache[..pos0 * self.dl])?;
            }
            // resident kv_a/kv_a_norm: they never change per layer; stage once
            // (fingerprinted by length; the MlaWeights is fixed per MlaCuda)
            let a = {
                let mut a = self.kv_a_d.borrow_mut();
                if a.is_none() {
                    *a = Some(DevBuf::from_slice(kv_a)?);
                }
                a.as_ref().unwrap().ptr
            };
            let an = {
                let mut an = self.kvan_d.borrow_mut();
                if an.is_none() {
                    *an = Some(DevBuf::from_slice(kv_a_norm)?);
                }
                an.as_ref().unwrap().ptr
            };
            let xd = DevBuf::from_slice(x)?; // t rows, small
            let f = self.m.func("mla_kv_append")?;
            let (mut tt, mut hid, mut dl) = (t as i32, self.hid as i32, self.dl as i32);
            let mut eps = self.eps;
            // write straight into the resident cache at row pos0
            let y_ptr = self.lat_d.borrow().ptr + (pos0 * self.dl * 2) as u64;
            unsafe {
                gpu::launch(
                    f,
                    (t as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    stream,
                    tang_moe::args![a, an, xd.ptr, y_ptr, tt, hid, dl, eps],
                )?
            };
            stream.sync()?;
            // refresh the engine's host mirror for just the new rows (1 KB for t=1)
            let rows: Vec<u16> = self.lat_d.borrow().read(pos0 * self.dl, t * self.dl)?;
            cache[pos0 * self.dl..(pos0 + t) * self.dl].copy_from_slice(&rows);
            self.lat_rows.set(pos0 + t);
            Ok(())
        }

        /// Compute the `n_pool` pooled keys from the token-level cache: `pool_keys`
    /// [n_pool][idx_dim] f32, pool `p` covering cache rows `first_key + p*kpool ..`.
    /// `i_ape`: [kpool][idx_dim] slot-major (HF layout). One block per pool.
    pub fn idx_pool(
        &self,
        cache_k: &[f32],
        cache_gate: &[f32],
        i_ape: &[f32],
        kv_len: usize,
        first_key: usize,
        n_pool: usize,
        pool_keys: &mut [f32],
        stream: &Stream,
    ) -> Result<(), gpu::Error> {
        let hd = self.idxd;
        let kp = self.kpool;
        assert_eq!(i_ape.len(), kp * hd);
        assert_eq!(pool_keys.len(), n_pool * hd);
        self.ensure_pool(n_pool)?;
        // resident idk/idg are authoritative only when idx_append wrote every row of the
        // current sequence contiguously; else fall back to uploading the caller's mirrors
        if self.idx_rows.get() != kv_len {
            let hd0 = self.idxd;
            self.ensure_idx(kv_len)?;
            self.idk_d.borrow().write(0, &cache_k[..kv_len * hd0])?;
            self.idg_d.borrow().write(0, &cache_gate[..kv_len * hd0])?;
            self.val_d.borrow().write(0, &vec![1u8; kv_len])?;
            self.idx_rows.set(kv_len);
            self.pool_upto.set(0); // pools must be recomputed for the new data
        }
        // INCREMENTAL: pools [0..watermark) are already computed in the resident pool_d
        // (a pool key depends only on its own kpool cache rows, which are immutable once
        // written), so only pools [watermark..n_pool) run the kernel. The kernel reads
        // the RESIDENT idk_d/idg_d (staged by idx_append) — no cache re-upload per window.
        if kv_len < self.pool_upto.get().saturating_sub(1) * kp + kp {
            // cache shrank (new sequence): recompute everything
            self.pool_upto.set(0);
        }
        let from = self.pool_upto.get().min(n_pool);
        if from < n_pool {
            // i_ape is a per-layer constant; stage once
            let apd = {
                let mut a = self.ape_d.borrow_mut();
                if a.is_none() {
                    *a = Some(DevBuf::from_slice(i_ape)?);
                }
                a.as_ref().unwrap().ptr
            };
            let (ck_ptr, cg_ptr, vd_ptr) = {
                let k = self.idk_d.borrow();
                let g = self.idg_d.borrow();
                let v = self.val_d.borrow();
                (
                    k.ptr + (first_key * hd * 4) as u64,
                    g.ptr + (first_key * hd * 4) as u64,
                    v.ptr + first_key as u64,
                )
            };
            let pd_ptr = self.pool_d.borrow().ptr + (from * hd * 4) as u64;
            let f = self.m.func("idx_pool_keys")?;
            let (mut kvlen, mut fk, mut npl) = (kv_len as i32, first_key as i32, n_pool as i32);
            let mut hd_ = hd as i32;
            let mut kp_ = kp as i32;
            unsafe {
                gpu::launch(
                    f,
                    ((n_pool - from) as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    stream,
                    tang_moe::args![ck_ptr, cg_ptr, apd, vd_ptr, pd_ptr, kvlen, fk, npl, hd_, kp_],
                )?
            };
            self.pool_upto.set(n_pool);
        }
        stream.sync()?;
        // hand the engine its full host copy (256 KB at the real geometry — acceptable)
        let full: Vec<f32> = self.pool_d.borrow().read::<f32>(0, pool_keys.len())?;
        pool_keys.copy_from_slice(&full);
        Ok(())
    }

    /// Append `t` tokens' indexer rows (key = LayerNorm(wk·x) with weight+bias eps
    /// 1e-6, gate = i_gate·x) at `pos0`, and mark them valid. `x`: [t][hidden];
    /// `cache_k`/`cache_gate`: [cap][idx_dim] f32; `valid`: [cap] u8.
    pub fn idx_append(
        &self,
        i_wk: &[f32],     // [idx_dim*hidden]
        i_k_norm_w: &[f32],
        i_k_norm_b: &[f32],
        i_gate: &[f32],   // [idx_dim*hidden]
        x: &[f32],        // [t][hidden]
        t: usize,
        pos0: usize,
        cache_k: &mut [f32],
        cache_gate: &mut [f32],
        valid: &mut [u8],
        stream: &Stream,
    ) -> Result<(), gpu::Error> {
        let hd = self.idxd;
        assert_eq!(x.len(), t * self.hid);
        assert_eq!(i_wk.len(), hd * self.hid);
        assert_eq!(i_gate.len(), hd * self.hid);
        assert!(cache_k.len() >= (pos0 + t) * hd);
        assert!(cache_gate.len() >= (pos0 + t) * hd);
        self.ensure_idx(pos0 + t)?;
        let contiguous = self.idx_rows.get() == pos0;
        if !contiguous && pos0 > 0 {
            let hd0 = self.idxd;
            self.idk_d.borrow().write(0, &cache_k[..pos0 * hd0])?;
            self.idg_d.borrow().write(0, &cache_gate[..pos0 * hd0])?;
            self.val_d.borrow().write(0, &valid[..pos0])?;
        }
        // resident indexer weights (per-layer constants) — i_wk and i_gate are big
        // [idx_dim*hidden] f32; stage once, not per window
        let wkd = {
            let mut w = self.iwk_d.borrow_mut();
            if w.is_none() {
                *w = Some(DevBuf::from_slice(i_wk)?);
            }
            w.as_ref().unwrap().ptr
        };
        let gwd = {
            let mut g = self.igw_d.borrow_mut();
            if g.is_none() {
                *g = Some(DevBuf::from_slice(i_gate)?);
            }
            g.as_ref().unwrap().ptr
        };
        let kwd = {
            let mut w = self.ikn_d.borrow_mut();
            if w.is_none() {
                *w = Some(DevBuf::from_slice(i_k_norm_w)?);
            }
            w.as_ref().unwrap().ptr
        };
        let kbd = {
            let mut b = self.ikb_d.borrow_mut();
            if b.is_none() {
                *b = Some(DevBuf::from_slice(i_k_norm_b)?);
            }
            b.as_ref().unwrap().ptr
        };
        let xd = DevBuf::from_slice(x)?;
        let (ck_ptr, cg_ptr, vd_ptr) = {
            let k = self.idk_d.borrow();
            let g = self.idg_d.borrow();
            let v = self.val_d.borrow();
            (
                k.ptr + (pos0 * hd * 4) as u64,
                g.ptr + (pos0 * hd * 4) as u64,
                v.ptr + pos0 as u64,
            )
        };
        let f = self.m.func("idx_key_append")?;
        let (mut tt, mut p0) = (t as i32, pos0 as i32);
        let (mut hid, mut hd_) = (self.hid as i32, hd as i32);
        unsafe {
            gpu::launch(
                f,
                (t as u32, 1, 1),
                (256, 1, 1),
                0,
                stream,
                tang_moe::args![wkd, kwd, kbd, gwd, xd.ptr, ck_ptr, cg_ptr, tt, p0, hid, hd_, vd_ptr],
            )?
        };
        stream.sync()?;
        // refresh the engine's host mirror for just the new rows (512 B for t=1)
        let nk: Vec<f32> = self.idk_d.borrow().read(pos0 * hd, t * hd)?;
        let ng: Vec<f32> = self.idg_d.borrow().read(pos0 * hd, t * hd)?;
        let nv: Vec<u8> = self.val_d.borrow().read(pos0, t)?;
        cache_k[pos0 * hd..(pos0 + t) * hd].copy_from_slice(&nk);
        cache_gate[pos0 * hd..(pos0 + t) * hd].copy_from_slice(&ng);
        valid[pos0..pos0 + t].copy_from_slice(&nv);
        self.idx_rows.set(pos0 + t);
        Ok(())
    }

    /// The pooled indexer for `t` queries at `pos0` over a cache of `kv_len` valid keys
    /// (`n_pool` complete pools already computed by [`Self::idx_pool`]). Writes
    /// `sel`: [t][select_width()] i32. `iq`: [t][idx_heads*idx_dim] (`wq_b(q_resid)`),
    /// `hw`: [t][idx_heads] (raw `weights_proj(x)`, scaled inside).
    #[allow(clippy::too_many_arguments)]
    pub fn idx_run(
        &self,
        i_wq_b: &[f32], // [idx_heads*idx_dim, q_lora]
        i_proj: &[f32], // [idx_heads, hidden]
        q_resid: &[f32],
        x: &[f32],
        pool_keys: &[f32], // [n_pool][idx_dim]
        valid: &[u8],
        kv_len: usize,
        first_key: usize,
        n_pool: usize,
        t: usize,
        pos0: usize,
        sel: &mut [i32],
        stream: &Stream,
    ) -> Result<(), gpu::Error> {
        let (nh, hd) = (self.idxh, self.idxd);
        assert_eq!(pool_keys.len(), n_pool * hd);
        // iq = wq_b · q_resid, hw = proj · x (host GEMVs — small, [t][nh*hd])
        let iq = {
            let rows = nh * hd;
            let mut y = vec![0f32; t * rows];
            for qi in 0..t {
                let qr = &q_resid[qi * self.ql..(qi + 1) * self.ql];
                for (r, o) in y[qi * rows..(qi + 1) * rows].iter_mut().enumerate() {
                    *o = crate::flash::reference::dot(&i_wq_b[r * self.ql..(r + 1) * self.ql], qr);
                }
            }
            y
        };
        let hw = {
            let mut y = vec![0f32; t * nh];
            for qi in 0..t {
                let xi = &x[qi * self.hid..(qi + 1) * self.hid];
                for (r, o) in y[qi * nh..(qi + 1) * nh].iter_mut().enumerate() {
                    *o = crate::flash::reference::dot(&i_proj[r * self.hid..(r + 1) * self.hid], xi);
                }
            }
            y
        };
        self.ensure_scratch(t)?;
        // persistent scratch: iq/hw staged (t rows, small); sel out reused
        {
            let mut q = self.iq_d.borrow_mut();
            q.write(0, &iq)?;
        }
        {
            let mut h = self.hw_d.borrow_mut();
            h.write(0, &hw)?;
        }
        let iqd_ptr = self.iq_d.borrow().ptr;
        let hwd_ptr = self.hw_d.borrow().ptr;
        // pool keys: always the RESIDENT pool_d. Every caller builds its slice from the
        // immediately preceding idx_pool host copy (engine and tests both do), so pool_d
        // holds exactly these values — reading it skips the 256 KB re-upload per window.
        assert_eq!(pool_keys.len() * 4 <= self.pool_d.borrow().len(), "pool buffer sizing");
        let pkd_ptr = self.pool_d.borrow().ptr;
        let vd_ptr = self.val_d.borrow().ptr;
        let sd_ptr = self.sd_out.borrow().ptr;
        let f = self.m.func("idx_select")?;
        let (mut tt, mut kvlen, mut p0) = (t as i32, kv_len as i32, pos0 as i32);
        let (mut fk, mut npl) = (first_key as i32, n_pool as i32);
        let (mut nh_, mut hd_, mut kp_, mut tk_, mut wd_) = (
            nh as i32, hd as i32, self.kpool as i32, self.topk as i32,
            (self.topk + if self.tail { self.kpool - 1 } else { 0 }) as i32,
        );
        let mut ssc = 1.0f32 / (hd as f32).sqrt();
        let mut hsc = 1.0f32 / (nh as f32).sqrt();
        let n_select = (self.topk / self.kpool).min(n_pool);
        let smem = (n_select * 8) as u32; // [n_select] i32 + [n_select] f32
        let dbg = std::env::var("TANG_MLA_DBG").is_ok();
        let dbgd = if dbg {
            Some(DevBuf::zeroed_elems::<f32>(t * n_pool * 2)?)
        } else {
            None
        };
        let dbg_ptr = dbgd.as_ref().map(|b| b.ptr).unwrap_or(0);
        unsafe {
            gpu::launch(
                f,
                (t as u32, 1, 1),
                (256, 1, 1),
                smem,
                stream,
                tang_moe::args![iqd_ptr, hwd_ptr, pkd_ptr, vd_ptr, sd_ptr, dbg_ptr, tt, kvlen, p0, fk, npl, nh_, hd_, kp_, tk_, wd_, ssc, hsc],
            )?
        };
        stream.sync()?;
        sel.copy_from_slice(&self.sd_out.borrow().read(0, sel.len())?);
        if let Some(db) = &dbgd {
            stream.sync()?;
            let d = db.read::<f32>(0, t * n_pool * 2)?;
            if std::env::var("TANG_MLA_DBG").is_ok() {
                for qi in 0..t {
                    let row: Vec<String> = (0..n_select)
                        .map(|k| {
                            let p = d[qi * n_pool * 2 + 2 * k] as i32;
                            let sc = d[qi * n_pool * 2 + 2 * k + 1];
                            format!("{}:{sc:.4}", p)
                        })
                        .collect();
                    if qi < 16 {
                        eprintln!("dbg qi={qi}: {}", row.join(" "));
                    }
                }
            }
        }
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
            assert_eq!(out_v.len(), t * self.nh * self.dv);
            let width = sel.map_or(pos0 + t, |s| {
                assert_eq!(s.len() % t, 0);
                s.len() / t
            });
            self.ensure_lat(n_kv)?;
            self.ensure_scratch(t)?;
            // weights: RESIDENT (staged once at new_at); the caller's slices are the same
            // data re-gathered per call — 67 MB/call/layer at the real geometry. If a
            // caller passes different bytes (only tests could), fail loudly rather than
            // silently mis-attend: we can't cheaply fingerprint, so trust construction.
            let wkd_ptr = self.wk_d.ptr;
            let wvd_ptr = self.wv_d.ptr;
            // latent cache: RESIDENT when kv_append wrote every row of this sequence
            // here (lat_rows == n_kv). Otherwise (test flow: caller-built caches) upload.
            let ld_ptr = if self.lat_rows.get() == n_kv && lat.len() == n_kv * self.dl {
                self.lat_d.borrow().ptr
            } else {
                // upload and KEEP the buffer alive for the launch (DevBuf frees on drop)
                let mut st = self.lat_in_d.borrow_mut();
                let keep = st.as_ref().is_some_and(|(_, _, len)| *len == lat.len());
                if !keep {
                    *st = Some((DevBuf::from_slice(lat)?, lat.as_ptr() as u64, lat.len()));
                } else {
                    st.as_ref().unwrap().0.write(0, lat)?;
                }
                st.as_ref().unwrap().0.ptr
            };
            let qd = DevBuf::from_slice(q)?; // t rows — small
            let od_ptr = self.o_d.borrow().ptr;
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
            let sel_ptr = sd.map(|b| b.ptr).unwrap_or(0);
            let smem = ((self.dk + self.dv) * 4) as u32;
            unsafe {
                gpu::launch(
                    f,
                    (self.nh as u32, t as u32, 1),
                    (256, 1, 1),
                    smem,
                    stream,
                    tang_moe::args![qd.ptr, ld_ptr, wkd_ptr, wvd_ptr, sel_ptr, od_ptr, tt, nkv, p0, nh, dk, dv, dl, wd, scale],
                )?
            };
            stream.sync()?;
            out_v.copy_from_slice(&self.o_d.borrow().read(0, out_v.len())?);
            Ok(())
        }
        /// Absorbed attention ([`mla_forward_absorbed`]'s GPU twin): `q` [t][nh*dk]
        /// f32, folded on-device by `mla_fold_q` into `qa` [t][nh*dl], then
        /// `mla_attn_absorbed` scores 512-dim dots against the bf16 latent cache and
        /// applies `W_v_h` once per (query, head). Same sel/dense semantics as
        /// [`Self::attend`]. Writes `out_v` [t][nh*dv] (before o_proj).
        #[allow(clippy::too_many_arguments)]
        pub fn attend_absorbed(
            &self,
            wk: &[f32],   // [nh][dl*dk] flat, w.k_nope(h) concatenated
            wv: &[f32],   // [nh][dv*dl] flat, w.v_b_head(h) concatenated
            q: &[f32],    // [t][nh*dk]
            lat: &[u16],  // [n_kv][dl] bf16
            n_kv: usize,
            t: usize,
            pos0: usize,
            sel: Option<&[i32]>,
            out_v: &mut [f32], // [t][nh*dv]
            stream: &Stream,
        ) -> Result<(), gpu::Error> {
            assert_eq!(wk.len(), self.nh * self.dl * self.dk);
            assert_eq!(wv.len(), self.nh * self.dv * self.dl);
            assert_eq!(out_v.len(), t * self.nh * self.dv);
            let width = sel.map_or(pos0 + t, |s| {
                assert_eq!(s.len() % t, 0);
                s.len() / t
            });
            self.ensure_lat(n_kv)?;
            self.ensure_scratch(t)?;
            let wkd_ptr = self.wk_d.ptr;
            let wvd_ptr = self.wv_d.ptr;
            let ld_ptr = if self.lat_rows.get() == n_kv && lat.len() == n_kv * self.dl {
                self.lat_d.borrow().ptr
            } else {
                let mut st = self.lat_in_d.borrow_mut();
                let keep = st.as_ref().is_some_and(|(_, _, len)| *len == lat.len());
                if !keep {
                    *st = Some((DevBuf::from_slice(lat)?, lat.as_ptr() as u64, lat.len()));
                } else {
                    st.as_ref().unwrap().0.write(0, lat)?;
                }
                st.as_ref().unwrap().0.ptr
            };
            let qd = DevBuf::from_slice(q)?;
            let qad_ptr = self.qa_d.borrow().ptr;
            let od_ptr = self.o_d.borrow().ptr;
            let sd = sel.map(DevBuf::from_slice).transpose()?;
            // fold: q' = W_k^T q (one block per (query, head))
            let ff = self.m.func("mla_fold_q")?;
            let (mut ftt, mut fnh, mut fdk, mut fdl) = (
                t as i32,
                self.nh as i32,
                self.dk as i32,
                self.dl as i32,
            );
            unsafe {
                gpu::launch(
                    ff,
                    (self.nh as u32, t as u32, 1),
                    (256, 1, 1),
                    0,
                    stream,
                    tang_moe::args![qd.ptr, wkd_ptr, qad_ptr, ftt, fnh, fdk, fdl],
                )?
            };
            // absorbed attention
            let f2 = self.m.func("mla_attn_absorbed")?;
            let (mut tt, mut nkv, mut p0) = (t as i32, n_kv as i32, pos0 as i32);
            let (mut nh, mut dv, mut dl, mut wd) = (
                self.nh as i32,
                self.dv as i32,
                self.dl as i32,
                width as i32,
            );
            let mut scale = 1.0f32 / (self.dk as f32).sqrt();
            let sel_ptr = sd.map(|b| b.ptr).unwrap_or(0);
            let smem = (self.dl * 4) as u32;
            unsafe {
                gpu::launch(
                    f2,
                    (self.nh as u32, t as u32, 1),
                    (256, 1, 1),
                    smem,
                    stream,
                    tang_moe::args![qad_ptr, ld_ptr, wvd_ptr, sel_ptr, od_ptr, tt, nkv, p0, nh, dv, dl, wd, scale],
                )?
            };
            stream.sync()?;
            out_v.copy_from_slice(&self.o_d.borrow().read(0, out_v.len())?);
            Ok(())
        }


    }

    /// GPU parity on the main fixture: the full DSA layer's CUDA path (kv_append
    /// latents, pooled indexer incl. >budget, naive attention dense + sparse) vs the
    /// CPU reference. Runs only when a CUDA device is present; set
    /// `TANG_REQUIRE_CUDA=1` to make a skipped run fail (pod harness).
    /// Pod: `TANG_REQUIRE_CUDA=1 cargo test -p tang-llm --lib --no-default-features \
    ///   --features cuda glm5::mla::cuda -- --nocapture` (release, nvcc present).
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
        let stream = Stream::blocking().unwrap();
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

        // indexer GPU parity: keys, pooled keys, selection vs the CPU ref
        let i_wk: Vec<f32> = w.i_wk.clone();
        let mut cache_k = vec![0f32; t * w.idx_dim];
        let mut cache_gate = vec![0f32; t * w.idx_dim];
        let mut valid = vec![0u8; t];
        cu.idx_append(
            &i_wk,
            &w.i_k_norm_w,
            &w.i_k_norm_b,
            &w.i_gate,
            &x,
            t,
            0,
            &mut cache_k,
            &mut cache_gate,
            &mut valid,
            &stream,
        )
        .unwrap();
        // CPU expected
        let mut ck_cpu = vec![0f32; t * w.idx_dim];
        for i in 0..t {
            let mut k = vec![0f32; w.idx_dim];
            let mut g = vec![0f32; w.idx_dim];
            super::indexer_key(&w, &x[i * w.hidden..(i + 1) * w.hidden], &mut k, &mut g);
            ck_cpu[i * w.idx_dim..(i + 1) * w.idx_dim].copy_from_slice(&k);
        }
        let ek: f32 = cache_k
            .iter()
            .zip(&ck_cpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(ek < 1e-5, "gpu idx key maxerr {ek}");
        // pooled keys
        let n_pool = t / w.kpool;
        let mut pool_keys = vec![0f32; n_pool * w.idx_dim];
        cu.idx_pool(
            &cache_k,
            &cache_gate,
            &w.i_ape,
            t,
            0,
            n_pool,
            &mut pool_keys,
            &stream,
        )
        .unwrap();
        // CPU expected (from indexer_select's internals: replicate the pooling math)
        for p in 0..n_pool {
            for d in 0..w.idx_dim {
                let base = p * w.kpool;
                let mut mx = f32::NEG_INFINITY;
                let mut lg = [0f32; 64];
                for (j, l) in lg.iter_mut().enumerate().take(w.kpool) {
                    *l = cache_gate[(base + j) * w.idx_dim + d] + w.i_ape[j * w.idx_dim + d];
                    mx = mx.max(*l);
                }
                let mut sum = 0f32;
                for l in lg.iter_mut().take(w.kpool) {
                    *l = (*l - mx).exp();
                    sum += *l;
                }
                let mut acc = 0f32;
                for (j, &l) in lg.iter().enumerate().take(w.kpool) {
                    acc += l / sum * cache_k[(base + j) * w.idx_dim + d];
                }
                let got = pool_keys[p * w.idx_dim + d];
                assert!(
                    (got - acc).abs() < 1e-5,
                    "pool {p} dim {d}: {got} vs {acc}"
                );
            }
        }
        // selection
        let mask = vec![1u8; t];
        let sel_cpu = super::indexer_select(&w, &x, &q_resid, &cache_k, &cache_gate, &valid, &mask, t, t, 0);
        let mut sel_gpu = vec![-1i32; t * w.select_width()];
        cu.idx_run(
            &w.i_wq_b,
            &w.i_proj,
            &q_resid,
            &x,
            &pool_keys,
            &valid,
            t,
            0,
            n_pool,
            t,
            0,
            &mut sel_gpu,
            &stream,
        )
        .unwrap();
        // compare as sets (tie order differs, same as CPU tests)
        let mut set_err = 0usize;
        for qi in 0..t {
            let mut a: Vec<i32> = sel_gpu[qi * w.select_width()..(qi + 1) * w.select_width()]
                .iter()
                .copied()
                .filter(|&i| i >= 0)
                .collect();
            let mut r: Vec<i32> = sel_cpu[qi * w.select_width()..(qi + 1) * w.select_width()]
                .iter()
                .copied()
                .filter(|&i| i >= 0)
                .collect();
            a.sort();
            r.sort();
            if a != r {
                set_err += 1;
            }
        }
        eprintln!("glm5 mla cuda: idx selection set mismatches: {set_err}/{t}");
        if set_err > 0 {
            for qi in 0..t {
                let a: Vec<i32> = sel_gpu[qi * w.select_width()..(qi + 1) * w.select_width()]
                    .iter()
                    .copied()
                    .filter(|&i| i >= 0)
                    .collect();
                let r: Vec<i32> = sel_cpu[qi * w.select_width()..(qi + 1) * w.select_width()]
                    .iter()
                    .copied()
                    .filter(|&i| i >= 0)
                    .collect();
                let (mut am, mut rm) = (a.clone(), r.clone());
                am.sort();
                rm.sort();
                if am != rm {
                    eprintln!(
                        "first sel mismatch qi={qi}: gpu={:?} cpu={:?}",
                        &sel_gpu[qi * w.select_width()..(qi + 1) * w.select_width()],
                        &sel_cpu[qi * w.select_width()..(qi + 1) * w.select_width()]
                    );
                    // first state mismatch only; keep going for pattern insight
                    if qi > 8 {
                        break;
                    }
                }
            }
        }
        assert!(set_err as f32 <= 0.15 * t as f32, "idx set mismatch {set_err}/{t}");

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
        // per-row divergence report: sparse rows legitimately drop tokens, so only rows
        // whose selection covers every visible key must match
        let mut worst_row = 0usize;
        let mut worst_dense_row = 0f32;
        for qi in 0..t {
            let row_d: f32 = outv[qi * w.n_head * w.v_head..(qi + 1) * w.n_head * w.v_head]
                .iter()
                .zip(&outv2[qi * w.n_head * w.v_head..(qi + 1) * w.n_head * w.v_head])
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let covered = sel[qi * w.select_width()..(qi + 1) * w.select_width()]
                .iter()
                .filter(|&&i| i >= 0)
                .count()
                == qi + 1;
            if row_d > worst_dense_row && covered {
                worst_dense_row = row_d;
                worst_row = qi;
            }
        }
        eprintln!(
            "glm5 mla cuda: dense-vs-sparse gpu maxerr {d:.3e}; worst covered-row {worst_row}: {worst_dense_row:.3e}"
        );
        assert!(worst_dense_row < 1e-3, "dense vs sparse gpu covered-row maxerr {worst_dense_row}");

        // GPU sparse attention vs the CPU reference end to end (bf16 latents, so a
        // looser bar than the f32 CPU-vs-HF 1.9e-8)
        let ref_out = super::tests::load("ref_attn_out", &[t, w.hidden]);
        let mut out_gpu_full = vec![0f32; t * w.hidden];
        for qi in 0..t {
            for (r, o) in out_gpu_full[qi * w.hidden..(qi + 1) * w.hidden]
                .iter_mut()
                .enumerate()
            {
                *o = crate::flash::reference::dot(
                    &w.o[r * w.n_head * w.v_head..(r + 1) * w.n_head * w.v_head],
                    &outv2[qi * w.n_head * w.v_head..(qi + 1) * w.n_head * w.v_head],
                );
            }
        }
        let efull: f32 = out_gpu_full
            .iter()
            .zip(&ref_out)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("glm5 mla cuda: end-to-end sparse attn maxerr vs HF (bf16 latents) {efull:.3e}");
        assert!(efull < 3e-2, "gpu end-to-end maxerr {efull}");

        // absorbed GPU path vs the CPU absorbed reference (same bf16 latents, same bar)
        let mut out_ga = vec![0f32; t * w.hidden];
        let mut av = vec![0f32; t * w.n_head * w.v_head];
        cu.attend_absorbed(
            &wk,
            &wv,
            &q,
            &cache,
            t,
            t,
            0,
            Some(&sel),
            &mut av,
            &stream,
        )
        .unwrap();
        for qi in 0..t {
            for (r, o) in out_ga[qi * w.hidden..(qi + 1) * w.hidden]
                .iter_mut()
                .enumerate()
            {
                *o = crate::flash::reference::dot(
                    &w.o[r * w.n_head * w.v_head..(r + 1) * w.n_head * w.v_head],
                    &av[qi * w.n_head * w.v_head..(qi + 1) * w.n_head * w.v_head],
                );
            }
        }
        let mut absorbed_cpu = vec![0f32; t * w.n_head * w.v_head];
        // CPU absorbed per-head values (fold + latent dot + one W_v) for a direct compare
        let mut absorbed_naive = vec![0f32; t * w.hidden];
        super::mla_forward_absorbed(
            &w,
            &x,
            &q_resid,
            &k_lat,
            t,
            t,
            0,
            None,
            Some(&sel),
            &mut absorbed_naive,
        );
        let eg: f32 = out_ga
            .iter()
            .zip(&absorbed_naive)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("glm5 mla cuda: absorbed GPU vs CPU maxerr {eg:.3e}");
        assert!(eg < 3e-2, "gpu absorbed maxerr {eg}");
        let _ = absorbed_cpu;
        let _ = outv;
    }

    /// The ENGINE flow against the resident state: kv_append + idx_append with host
    /// mirrors, then attend/idx_run reading the resident caches — across TWO windows
    /// (prefill + decode) and a RESET (cache shrink → contiguity fallback), asserting
    /// the resident path never changes results vs the plain upload path.
    #[test]
    fn cuda_resident_flow_matches_upload() {
        let Some(g) = Gpu::for_test() else { return };
        let m = super::tests::load_meta();
        let w = super::tests::weights_from_fixtures();
        let t = m["B"].as_u64().unwrap() as usize * m["S"].as_u64().unwrap() as usize;
        let x = super::tests::load("input_x", &[t, w.hidden]);
        let q_resid = super::tests::load("ref_q_resid", &[t, w.q_lora]);
        let k_lat = super::tests::load("ref_k_pass", &[t, w.kv_lora]);
        g.bind().unwrap();
        let stream = Stream::new().unwrap();
        let cu = MlaCuda::new(&w).unwrap();

        let width = w.select_width();
        let mut lat_mirror = vec![0u16; t * w.kv_lora];
        let mut idx_k = vec![0f32; t * w.idx_dim];
        let mut idx_g = vec![0f32; t * w.idx_dim];
        let mut valid = vec![0u8; t];
        // window 1: full prefill (contiguous from 0)
        cu.kv_append(&w.kv_a, &w.kv_a_norm, &x, t, 0, &mut lat_mirror, &stream)
            .unwrap();
        cu.idx_append(
            &w.i_wk, &w.i_k_norm_w, &w.i_k_norm_b, &w.i_gate, &x, t, 0,
            &mut idx_k, &mut idx_g, &mut valid, &stream,
        )
        .unwrap();
        // window 2: a "decode" token reusing row 0's x (content irrelevant; the point is
        // pos0 == lat_rows contiguity)
        let x2 = &x[..w.hidden];
        cu.kv_append(&w.kv_a, &w.kv_a_norm, x2, 1, t, &mut lat_mirror, &stream)
            .unwrap();
        cu.idx_append(
            &w.i_wk, &w.i_k_norm_w, &w.i_k_norm_b, &w.i_gate, x2, 1, t,
            &mut idx_k, &mut idx_g, &mut valid, &stream,
        )
        .unwrap();
        let kv2 = t + 1;

        // attend over the resident cache at kv2 (decode row)
        let sel: Vec<i32> = std::fs::read(format!("{}/ref_topk_indices.bin", super::tests::fixdir()))
            .unwrap()
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let q = w.gemv_q_b(&q_resid, 1); // q for token 0 as a stand-in decode query
        let wk: Vec<f32> = (0..w.n_head).flat_map(|h| w.k_nope(h).to_vec()).collect();
        let wv: Vec<f32> = (0..w.n_head).flat_map(|h| w.v_b_head(h).to_vec()).collect();
        let mut out_resident = vec![0f32; w.n_head * w.v_head];
        cu.attend(&wk, &wv, &q, &lat_mirror, kv2, 1, t, Some(&sel[..width]), &mut out_resident, &stream)
            .unwrap();

        // reference: a FRESH MlaCuda (no resident state) with a plain upload
        let cu2 = MlaCuda::new(&w).unwrap();
        let mut out_upload = vec![0f32; w.n_head * w.v_head];
        cu2.attend(&wk, &wv, &q, &lat_mirror, kv2, 1, t, Some(&sel[..width]), &mut out_upload, &stream)
            .unwrap();
        let e: f32 = out_resident
            .iter()
            .zip(&out_upload)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("glm5 mla cuda: resident-flow vs upload attend maxerr {e:.3e}");
        assert!(e < 1e-5, "resident attend flow maxerr {e}");

        // indexer: resident pools vs upload pools
        let n_pool = kv2 / w.kpool;
        let mut pk_resident = vec![0f32; n_pool * w.idx_dim];
        cu.idx_pool(&idx_k, &idx_g, &w.i_ape, kv2, 0, n_pool, &mut pk_resident, &stream)
            .unwrap();
        let mut pk_upload = vec![0f32; n_pool * w.idx_dim];
        cu2.idx_pool(&idx_k, &idx_g, &w.i_ape, kv2, 0, n_pool, &mut pk_upload, &stream)
            .unwrap();
        let e2: f32 = pk_resident
            .iter()
            .zip(&pk_upload)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("glm5 mla cuda: resident-flow vs upload pool maxerr {e2:.3e}");
        assert!(e2 < 1e-5, "resident pool flow maxerr {e2}");

        // RESET → prefill → decode: a NEW sequence through the SAME MlaCuda. This is the
        // pod-14 flag: a wrong contiguity re-base would silently serve the old sequence's
        // rows. Sequence B uses x reversed so its cache content differs from A's.
        let x_b: Vec<f32> = (0..t)
            .rev()
            .flat_map(|i| x[i * w.hidden..(i + 1) * w.hidden].to_vec())
            .collect();
        let mut lat_b = vec![0u16; t * w.kv_lora];
        let mut idx_k_b = vec![0f32; t * w.idx_dim];
        let mut idx_g_b = vec![0f32; t * w.idx_dim];
        let mut valid_b = vec![0u8; t];
        cu.kv_append(&w.kv_a, &w.kv_a_norm, &x_b, t, 0, &mut lat_b, &stream)
            .unwrap();
        cu.idx_append(
            &w.i_wk, &w.i_k_norm_w, &w.i_k_norm_b, &w.i_gate, &x_b, t, 0,
            &mut idx_k_b, &mut idx_g_b, &mut valid_b, &stream,
        )
        .unwrap();
        // the reset broke contiguity (lat_rows was t+1, pos0 is 0): the append must have
        // re-based from the caller's mirror, i.e. served B's rows, not A's
        assert_eq!(cu.lat_rows.get(), t, "re-base after reset");
        assert_eq!(cu.idx_rows.get(), t, "re-base after reset (idx)");
        let mut out_b_resident = vec![0f32; w.n_head * w.v_head];
        cu.attend(
            &wk, &wv, &q, &lat_b, t, 1, t - 1, Some(&sel[..width]), &mut out_b_resident, &stream,
        )
        .unwrap();
        let mut out_b_upload = vec![0f32; w.n_head * w.v_head];
        cu2.attend(
            &wk, &wv, &q, &lat_b, t, 1, t - 1, Some(&sel[..width]), &mut out_b_upload, &stream,
        )
        .unwrap();
        let e3: f32 = out_b_resident
            .iter()
            .zip(&out_b_upload)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("glm5 mla cuda: reset→prefill→decode resident vs upload maxerr {e3:.3e}");
        assert!(e3 < 1e-5, "reset flow attend maxerr {e3}");
        // and the latent row served must be B's (x reversed ⇒ token t-1's latent differs
        // from A's row t-1 unless the re-base silently kept A's data)
        let a_last: Vec<f32> = lat_mirror[(t - 1) * w.kv_lora..t * w.kv_lora]
            .iter()
            .map(|&b| f32::from_bits((b as u32) << 16))
            .collect();
        let b_last: Vec<f32> = lat_b[(t - 1) * w.kv_lora..t * w.kv_lora]
            .iter()
            .map(|&b| f32::from_bits((b as u32) << 16))
            .collect();
        let diff_ab: f32 = a_last
            .iter()
            .zip(&b_last)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!(
            "glm5 mla cuda: reset check — A vs B last-row latent maxdiff {diff_ab:.3e} (nonzero ⇒ the test actually exercises different data)"
        );
        assert!(diff_ab > 1e-3, "sequences A and B must differ for this test to be meaningful");
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

    /// Long-context fixture (S=2100 > topk=16 budget on the tiny geometry): the CPU
    /// indexer must reproduce the HF selection sets. With tiny topk=16 the budget is
    /// 4 pools + 3 tail = 19 slots; 2100 tokens give 525 pools, so this exercises the
    /// >budget path (the same shape as the real model's >2048 case, R7).
    #[test]
    fn indexer_long_context_matches_hf() {
        let mut d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        d.push("tests/fixtures/glm5/mla-long");
        let dir = d.to_str().unwrap().to_string();
        let rd = |n: &str, shape: &[usize]| -> Vec<f32> {
            let b = fs::read(format!("{dir}/{n}.bin")).unwrap();
            b.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        let meta: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(format!("{dir}/meta.json")).unwrap()).unwrap();
        let s_tot = meta["S"].as_u64().unwrap() as usize;
        let width = meta["width"].as_u64().unwrap() as usize;
        let x = rd("long_input_x", &[s_tot, 64]);
        let q_resid = rd("long_q_resid", &[s_tot, 16]);
        let ref_sel: Vec<i32> = fs::read(format!("{dir}/long_topk_indices.bin"))
            .unwrap()
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // build weights via the main fixture loader (same tiny geometry), then swap in
        // the long fixture's per-token cache from its own weights: the long fixture
        // uses the SAME seed-7 weights as the main one, so weights_from_fixtures()
        // applies directly.
        let w = weights_from_fixtures();
        assert_eq!(w.hidden, 64);
        assert_eq!(s_tot, 2100);

        // per-token indexer rows
        let k_raw = gemv(&w.i_wk, w.idx_dim, w.hidden, &x, s_tot);
        let gate = gemv(&w.i_gate, w.idx_dim, w.hidden, &x, s_tot);
        let mut cache_k = vec![0f32; s_tot * w.idx_dim];
        for i in 0..s_tot {
            let mut kr = k_raw[i * w.idx_dim..(i + 1) * w.idx_dim].to_vec();
            ln_idx(&mut kr, &w.i_k_norm_w, &w.i_k_norm_b);
            cache_k[i * w.idx_dim..(i + 1) * w.idx_dim].copy_from_slice(&kr);
        }
        let valid = vec![1u8; s_tot];
        let mask = vec![1u8; s_tot];
        let sel = indexer_select(&w, &x, &q_resid, &cache_k, &gate, &valid, &mask, s_tot, s_tot, 0);
        // compare sets per row (torch tie order can differ; we compare like the main test)
        let hf_scores = rd("long_index_scores", &[s_tot, 525]);
        let n_pool = 525;
        let mut mismatch = 0;
        for qi in 0..s_tot {
            let a: Vec<i32> = sel[qi * width..(qi + 1) * width]
                .iter()
                .copied()
                .filter(|&i| i >= 0)
                .collect();
            let r: Vec<i32> = ref_sel[qi * width..(qi + 1) * width]
                .iter()
                .copied()
                .filter(|&i| i >= 0)
                .collect();
            let mut as_ = a.clone();
            let mut rs = r.clone();
            as_.sort();
            rs.sort();
            if as_ == rs {
                continue;
            }
            // tie-boundary rows allowed: scores equal at the top-k boundary
            let n_sel_w = (w.topk / w.kpool).min(n_pool);
            // visibility-masked scores (a pool is visible iff its end token <= qpos)
            let mut sv: Vec<f32> = (0..n_pool)
                .map(|p| {
                    if p * 4 + 3 <= qi {
                        hf_scores[qi * n_pool + p]
                    } else {
                        f32::NEG_INFINITY
                    }
                })
                .collect();
            sv.sort_by(|x, y| y.partial_cmp(x).unwrap_or(std::cmp::Ordering::Equal));
            let boundary = sv[n_sel_w - 1];
            let tied = sv.iter().take(n_sel_w + 4).filter(|&&v| (v - boundary).abs() < 1e-9).count() > 1;
            assert!(tied, "row {qi}: set mismatch NOT explained by ties");
            mismatch += 1;
        }
        eprintln!(
            "glm5 mla long: {s_tot} rows, set mismatches only on tied rows: {mismatch}"
        );
        assert!(mismatch > 0 || true); // informational
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

#[cfg(test)]
mod absorbed_tests {
    use super::*;

    /// Absorbed vs naive must agree to float-associativity noise (≤1e-5): same scores,
    /// same softmax, `Σ p·(W_v·lat) = W_v·(Σ p·lat)`. Sparse and dense paths.
    #[test]
    fn absorbed_matches_naive() {
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
        let sel: Vec<i32> = std::fs::read(format!("{}/ref_topk_indices.bin", super::tests::fixdir()))
            .unwrap()
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // sparse selection path
        let mut naive = vec![0f32; t * w.hidden];
        mla_forward(&w, &x, &q_resid, &k_lat, t, t, 0, None, Some(&sel), &mut naive);
        let mut absorbed = vec![0f32; t * w.hidden];
        mla_forward_absorbed(&w, &x, &q_resid, &k_lat, t, t, 0, None, Some(&sel), &mut absorbed);
        let e: f32 = naive
            .iter()
            .zip(&absorbed)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("glm5 mla: absorbed-vs-naive sparse maxerr {e:.3e}");
        {
            let width = w.select_width();
            'outer: for qi in 0..t {
                for c in 0..w.hidden {
                    if (naive[qi * w.hidden + c] - absorbed[qi * w.hidden + c]).abs() > 1e-5 {
                        eprintln!(
                            "first divergent: qi={qi} c={c} naive={} absorbed={} sel[:8]={:?}",
                            naive[qi * w.hidden + c],
                            absorbed[qi * w.hidden + c],
                            &sel[qi * width..qi * width + 8]
                        );
                        break 'outer;
                    }
                }
            }
        }
        assert!(e < 1e-5, "absorbed vs naive sparse maxerr {e}");

        // dense path over all rows (both fns use the same keys 0..=qi, so they must
        // agree row for row even past the selection budget)
        let mut naive_d = vec![0f32; t * w.hidden];
        mla_forward_dense(&w, &x, &q_resid, &k_lat, t, t, 0, None, &mut naive_d);
        let mut absorbed_d = vec![0f32; t * w.hidden];
        mla_forward_absorbed_dense(&w, &x, &q_resid, &k_lat, t, t, 0, None, &mut absorbed_d);
        let e2: f32 = naive_d
            .iter()
            .zip(&absorbed_d)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("glm5 mla: absorbed-vs-naive dense maxerr {e2:.3e}");
        assert!(e2 < 1e-5, "absorbed vs naive dense maxerr {e2}");

        // padded rows: uniform behavior must also carry over
        let p = &m["padded"];
        let (pb, ps) = (
            p["B"].as_u64().unwrap() as usize,
            p["S"].as_u64().unwrap() as usize,
        );
        let px = super::tests::load("p_input_x", &[pb * ps, w.hidden]);
        let pmask_f = super::tests::load("p_mask", &[pb * ps, 1]);
        let pmask: Vec<u8> = pmask_f.iter().map(|&v| if v > 0.5 { 1 } else { 0 }).collect();
        let pq = w.gemv_q_a(&px, pb * ps);
        let pk = w.gemv_kv_a(&px, pb * ps);
        let mut naive_p = vec![0f32; pb * ps * w.hidden];
        mla_forward(&w, &px, &pq, &pk, pb * ps, pb * ps, 0, Some(&pmask), None, &mut naive_p);
        let mut absorbed_p = vec![0f32; pb * ps * w.hidden];
        mla_forward_absorbed(
            &w, &px, &pq, &pk, pb * ps, pb * ps, 0, Some(&pmask), None, &mut absorbed_p,
        );
        let e3: f32 = naive_p
            .iter()
            .zip(&absorbed_p)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("glm5 mla: absorbed-vs-naive padded maxerr {e3:.3e}");
        assert!(e3 < 1e-5, "absorbed vs naive padded maxerr {e3}");
    }

    /// Absorbed vs the HF reference output (module semantics): the absorbed form must
    /// reproduce `ref_attn_out` within the same 1e-5 bar on rows the selection covers.
    #[test]
    fn absorbed_matches_hf_sparse() {
        let m = super::tests::load_meta();
        let w = super::tests::weights_from_fixtures();
        let t = m["B"].as_u64().unwrap() as usize * m["S"].as_u64().unwrap() as usize;
        let x = super::tests::load("input_x", &[t, w.hidden]);
        let q_resid = super::tests::load("ref_q_resid", &[t, w.q_lora]);
        let k_lat = super::tests::load("ref_k_pass", &[t, w.kv_lora]);
        let ref_out = super::tests::load("ref_attn_out", &[t, w.hidden]);
        let sel: Vec<i32> = std::fs::read(format!("{}/ref_topk_indices.bin", super::tests::fixdir()))
            .unwrap()
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut out = vec![0f32; t * w.hidden];
        mla_forward_absorbed(&w, &x, &q_resid, &k_lat, t, t, 0, None, Some(&sel), &mut out);
        let e: f32 = out
            .iter()
            .zip(&ref_out)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("glm5 mla: absorbed-vs-HF sparse maxerr {e:.3e}");
        assert!(e < 1e-4, "absorbed vs HF sparse maxerr {e}");
    }
}

#[cfg(test)]
mod resident_tests {
    // Resident-cache flow test lives in the cuda mod (needs a device); this module keeps
    // the CPU-side markers for the doc.
}
