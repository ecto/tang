//! Qwen3.8-Flash-Next decode kernels: shapes, buffer layouts and numerics contracts.
//!
//! The ops live on [`ComputeDevice`](crate::ComputeDevice) (`quantize_act_into`, `q2_linear_into`,
//! `hc_read_into`, `gdn_step`, `router_topk_into`, `moe_grouped_into`, `qsa_select_into`, ...). The
//! reference math is `cpu::flash` (what `CpuDevice` runs, and what the trait's portable defaults
//! run on any device by downloading); the CUDA kernels are tested against it. Model math is
//! `crates/tang-llm/docs/strata.md`, "Block math".
//!
//! # Rules every op here follows
//!
//! - **A decode step is a verify window** of `t` = 1..=[`MAX_T`] tokens; `t` is a host argument
//!   (one captured graph per `t`).
//! - **Graph-capturable:** every op writes into caller-allocated buffers (`*_into`, `&mut`
//!   outputs), allocates nothing, and reads every per-window value from device memory: the
//!   window record `win` ([`Win`]) holds the first token's position and the commit count. No
//!   host scalar that changes between windows is ever a kernel argument.
//! - **Words.** Buffers are `f32`-element buffers. Integer data (ids, counts, plans, int8
//!   activations, byte blobs) is stored as raw 32-bit words (`f32::from_bits`), as
//!   `upload_u32` already does; byte data is packed little-endian, 4 bytes per word
//!   ([`ComputeDevice::upload_bytes`](crate::ComputeDevice::upload_bytes)). 64-bit device
//!   addresses are two words, low first.
//! - **Weights** are `[out, in]` row-major, the GGUF order (`ne0` = in is fastest). bf16 weights
//!   always multiply fp32 activations (the "fp32 contract"): no op rounds an activation to bf16
//!   except the QSA K/V cache store.

/// Largest verify window.
pub const MAX_T: usize = 8;

/// Model shapes (Qwen3.8-Flash-Next, GGUF architecture `qwen4exp`).
pub mod shape {
    /// Width of one residual stream.
    pub const HIDDEN: usize = 2560;
    /// Hyper-connection streams.
    pub const HC: usize = 4;
    /// Hyper-connection bottleneck rank.
    pub const HC_LR: usize = 320;

    /// GDN key/query heads.
    pub const GDN_HK: usize = 16;
    /// GDN value heads (v head `h` pairs with k head `h % GDN_HK`: GGUF order, modulo).
    pub const GDN_HV: usize = 48;
    /// GDN head dim (k, q and v).
    pub const GDN_D: usize = 128;
    /// Conv channels: `[q 2048 | k 2048 | v 6144]`.
    pub const GDN_CONV: usize = 2 * GDN_HK * GDN_D + GDN_HV * GDN_D;
    /// Value width (`z`, `o`, `y`).
    pub const GDN_V: usize = GDN_HV * GDN_D;
    /// Conv taps.
    pub const GDN_TAPS: usize = 4;
    /// Offsets in a GDN layer's stacked input projection row ([`GDN_PROJ`] wide):
    /// `[qkv | z | a | b]`, i.e. the GGUF tensors `attn_qkv`, `attn_gate`, `ssm_alpha`, `ssm_beta`
    /// stacked by rows into one weight so one GEMV makes all of them.
    pub const GDN_Z: usize = GDN_CONV;
    /// Offset of `a` (`ssm_alpha · x`, one per v head).
    pub const GDN_A: usize = GDN_Z + GDN_V;
    /// Offset of `b` (`ssm_beta · x`, one per v head).
    pub const GDN_B: usize = GDN_A + GDN_HV;
    /// Width of the stacked GDN input projection.
    pub const GDN_PROJ: usize = GDN_B + GDN_HV;

    /// QSA query heads.
    pub const QSA_HEADS: usize = 24;
    /// QSA K/V heads (q head `h` uses kv head `h / 12`).
    pub const QSA_KV: usize = 2;
    /// QSA head dim.
    pub const QSA_D: usize = 256;
    /// Rotated dims (NeoX pairs `(i, i + 32)` for `i < 32`), for attention and indexer alike.
    pub const QSA_ROT: usize = 64;
    /// Indexer query heads.
    pub const IDX_HEADS: usize = 4;
    /// Indexer key/query width.
    pub const IDX_D: usize = 128;
    /// Cells pooled per indexer block.
    pub const IDX_BLOCK: usize = 4;
    /// Cells selected per query: 2048 + the incomplete tail's 3.
    pub const QSA_WIDTH: usize = 2051;
    /// Offsets in a QSA layer's stacked input projection row ([`QSA_PROJ`] wide):
    /// `[q|gate per head (24 × 512, q first) | k 512 | v 512 | idx q 512 | idx k 128]`.
    pub const QSA_K: usize = QSA_HEADS * 2 * QSA_D;
    /// Offset of `v`.
    pub const QSA_V: usize = QSA_K + QSA_KV * QSA_D;
    /// Offset of the indexer query.
    pub const QSA_IQ: usize = QSA_V + QSA_KV * QSA_D;
    /// Offset of the raw indexer key.
    pub const QSA_IK: usize = QSA_IQ + IDX_HEADS * IDX_D;
    /// Width of the stacked QSA input projection.
    pub const QSA_PROJ: usize = QSA_IK + IDX_D;
    /// Attention output width (input of `attn_output`).
    pub const QSA_OUT: usize = QSA_HEADS * QSA_D;

    /// Routed experts per MoE layer.
    pub const EXPERTS: usize = 512;
    /// Experts per token.
    pub const TOPK: usize = 10;
    /// Expert FFN width.
    pub const FF: usize = 640;
}

use shape::*;

/// The per-window record in device memory (u32 words), written by the host before each window
/// (one 8-byte H2D copy, which a graph replays from its source).
pub struct Win;

impl Win {
    /// Position of the window's first token; token `i` is at `pos0 + i`.
    pub const POS0: usize = 0;
    /// Tokens to keep (commit): `min(n_keep, t)` are replayed into the running state.
    pub const N_KEEP: usize = 1;
    /// Words in the record.
    pub const WORDS: usize = 2;
}

// ---- int8 activations ----

/// Layout of int8-quantized activations, `m` rows of `k` (`k % 32 == 0`), as u32 words:
///
/// ```text
/// [m][k/4]   int8 codes, 4 per word, in chunk-permuted order (below)
/// [m][k/32]  f32 scale d of each 32-wide chunk
/// [m][k/32]  i32 Σ q over the chunk (the "hx" correction of Q2_0's (code − 1))
/// ```
///
/// **Quantization contract** (shared with the CPU expert kernels, so a VRAM hit and a CPU miss
/// see the same integers): per 32-element chunk, `amax = max |x|`, `d = amax / 127` (f32),
/// `q = d == 0 ? 0 : clamp(round_half_away_from_zero(x / d), -127, 127)` with `x / d` an IEEE
/// division.
///
/// **Chunk permutation.** Word `4h + f` (`h` < 2, `f` < 4) of a chunk holds, in bytes 0..4, the
/// codes of elements `16h + f`, `16h + 4 + f`, `16h + 8 + f`, `16h + 12 + f`. That is the order in
/// which `(w >> 2f) & 0x03030303` extracts four 2-bit codes from code word `h` of a Q2_0 chunk, so
/// the GPU dots a chunk with eight `dp4a` and no shuffles.
#[derive(Clone, Copy, Debug)]
pub struct QAct {
    /// Rows.
    pub m: usize,
    /// Row width.
    pub k: usize,
}

impl QAct {
    /// Words for `m` rows of `k`.
    pub fn words(self) -> usize {
        self.m * self.k / 4 + 2 * self.m * self.k / 32
    }
    /// Word offset of row `r`'s codes.
    pub fn codes(self, r: usize) -> usize {
        r * self.k / 4
    }
    /// Word offset of row `r`'s scales.
    pub fn scales(self, r: usize) -> usize {
        self.m * self.k / 4 + r * self.k / 32
    }
    /// Word offset of row `r`'s code sums.
    pub fn sums(self, r: usize) -> usize {
        self.m * self.k / 4 + self.m * self.k / 32 + r * self.k / 32
    }
    /// Position (word, byte) of element `e` of a row inside that row's codes.
    pub fn slot(e: usize) -> (usize, usize) {
        let (c, i) = (e / 32, e % 32);
        let (h, b, f) = (i / 16, (i % 16) / 4, i % 4);
        (c * 8 + 4 * h + f, b)
    }
}

// ---- Q2_0 ----

/// Bytes of one GGUF Q2_0 block: fp16 `d`, then 16 bytes of 2-bit codes, 64 weights.
pub const Q2_BLOCK_BYTES: usize = 18;

/// Bytes of an `[n, k]` Q2_0 matrix (GGUF and repacked alike).
pub fn q2_bytes(n: usize, k: usize) -> usize {
    n * k / 64 * Q2_BLOCK_BYTES
}

/// Repack GGUF Q2_0 (`[n, k]`, row-major blocks of 18 bytes: fp16 `d`, then codes, element
/// `4j + i` of the block in bits `2i..2i+2` of code byte `j`; `w = (code − 1) · d`) into the
/// device layout: `[n][k/4]` code bytes (the same bytes, rows contiguous), then `[n][k/64]`
/// fp16 scales. Same size, 16-byte aligned code rows, scales in their own plane.
pub fn q2_repack(raw: &[u8], n: usize, k: usize) -> Vec<u8> {
    assert!(
        k.is_multiple_of(64),
        "Q2_0 rows must be whole 64-weight blocks"
    );
    assert_eq!(raw.len(), q2_bytes(n, k), "Q2_0 matrix size");
    let (nb, codes) = (n * k / 64, n * k / 4);
    let mut out = vec![0u8; raw.len()];
    for b in 0..nb {
        let blk = &raw[b * 18..b * 18 + 18];
        out[codes + 2 * b..codes + 2 * b + 2].copy_from_slice(&blk[..2]);
        out[16 * b..16 * b + 16].copy_from_slice(&blk[2..]);
    }
    out
}

/// Position of chunk element `i` (< 32) in a repacked Q4X chunk (16 bytes): (byte, shift).
/// Code word `w` (< 4) of the chunk holds, at nibble `2b + s`, the element whose int8
/// activation sits in byte `b` of [`QAct`] word `2w + s`, so `(w >> 4s) & 0x0f0f0f0f` pairs
/// with the activation words in order, as Q2_0's fields do.
pub fn q4x_slot(i: usize) -> (usize, u32) {
    let (h, b, f) = (i / 16, (i % 16) / 4, i % 4);
    let j = 4 * h + f;
    let nib = 2 * b + j % 2;
    ((j / 2) * 4 + nib / 2, 4 * (nib % 2) as u32)
}

/// Repack tang-Q4 (MLX affine, group 64: `packed [n][k/8]` u32 low nibble first, `scales`,
/// `biases` `[n][k/64]` bf16; `w = scale · q + bias`) into Q4X for the int8-activation GEMV:
/// `[n][k/2]` code bytes in [`q4x_slot`] order, then the scales plane, then the biases plane
/// (bf16 bits, little-endian). 4.5 bits per weight, as tang-Q4.
pub fn q4x_repack(packed: &[u32], scales: &[u16], biases: &[u16], n: usize, k: usize) -> Vec<u8> {
    assert!(k.is_multiple_of(64) && packed.len() == n * k / 8);
    assert!(scales.len() == n * k / 64 && biases.len() == n * k / 64);
    let mut out = vec![0u8; n * k / 2 + 4 * n * k / 64];
    for r in 0..n {
        for e in 0..k {
            let q = ((packed[(r * k + e) / 8] >> (4 * (e % 8))) & 0xf) as u8;
            let (byte, sh) = q4x_slot(e % 32);
            out[r * k / 2 + (e / 32) * 16 + byte] |= q << sh;
        }
    }
    let sp = n * k / 2;
    for (i, (&s, &b)) in scales.iter().zip(biases).enumerate() {
        out[sp + 2 * i..sp + 2 * i + 2].copy_from_slice(&s.to_le_bytes());
        let bp = sp + 2 * (n * k / 64) + 2 * i;
        out[bp..bp + 2].copy_from_slice(&b.to_le_bytes());
    }
    out
}

/// One routed expert's blob, 1,382,400 bytes (the three GGUF Q2_0 matrices' bytes,
/// rearranged), laid out so a warp's 16-byte-per-lane loads are coalesced and every lane stays
/// on one row:
///
/// ```text
/// GU_CODES    320 tiles × [40 cols][4 rows] × 16 B   gate/up rows interleaved: gu row 2r = gate r,
/// GU_SCALES   320 tiles × [40 cols][4 rows] × fp16   2r + 1 = up r; tile i = gu rows 4i .. 4i + 4
/// DOWN_CODES  160 tiles × [10 cols][16 rows] × 16 B  down rows, tile i = rows 16i .. 16i + 16
/// DOWN_SCALES 160 tiles × [10 cols][16 rows] × fp16
/// ```
///
/// A "col" is one GGUF Q2_0 block (64 weights: its 16 code bytes, its fp16 `d`). A blob may live
/// in VRAM or in device-mapped host memory; the kernels only see its address.
pub struct ExpertBlob;

impl ExpertBlob {
    /// Gate/up code tiles.
    pub const GU_CODES: usize = 0;
    /// Gate/up scale tiles.
    pub const GU_SCALES: usize = 2 * FF * HIDDEN / 4;
    /// Down code tiles.
    pub const DOWN_CODES: usize = Self::GU_SCALES + 2 * FF * HIDDEN / 64 * 2;
    /// Down scale tiles.
    pub const DOWN_SCALES: usize = Self::DOWN_CODES + HIDDEN * FF / 4;
    /// Blob size.
    pub const BYTES: usize = Self::DOWN_SCALES + HIDDEN * FF / 64 * 2;

    /// Byte offsets of (codes, scale) of gu row `row` (2r gate r, 2r + 1 up r), block `c`.
    pub fn gu_block(row: usize, c: usize) -> (usize, usize) {
        let (tile, r) = (row / 4, row % 4);
        let i = tile * 160 + c * 4 + r;
        (Self::GU_CODES + 16 * i, Self::GU_SCALES + 2 * i)
    }

    /// Byte offsets of (codes, scale) of down row `row`, block `c`.
    pub fn down_block(row: usize, c: usize) -> (usize, usize) {
        let (tile, r) = (row / 16, row % 16);
        let i = tile * 160 + c * 16 + r;
        (Self::DOWN_CODES + 16 * i, Self::DOWN_SCALES + 2 * i)
    }

    /// Build a blob from the three GGUF Q2_0 tensors of one expert (`gate`, `up` `[FF, HIDDEN]`,
    /// `down` `[HIDDEN, FF]`).
    pub fn from_gguf(gate: &[u8], up: &[u8], down: &[u8]) -> Vec<u8> {
        assert!(gate.len() == q2_bytes(FF, HIDDEN) && up.len() == gate.len());
        assert_eq!(down.len(), q2_bytes(HIDDEN, FF));
        let mut b = vec![0u8; Self::BYTES];
        let mut put = |(co, so): (usize, usize), blk: &[u8]| {
            b[so..so + 2].copy_from_slice(&blk[..2]);
            b[co..co + 16].copy_from_slice(&blk[2..18]);
        };
        let nb = HIDDEN / 64;
        for r in 0..FF {
            for c in 0..nb {
                let o = (r * nb + c) * Q2_BLOCK_BYTES;
                put(Self::gu_block(2 * r, c), &gate[o..o + 18]);
                put(Self::gu_block(2 * r + 1, c), &up[o..o + 18]);
            }
        }
        let nb = FF / 64;
        for r in 0..HIDDEN {
            for c in 0..nb {
                let o = (r * nb + c) * Q2_BLOCK_BYTES;
                put(Self::down_block(r, c), &down[o..o + 18]);
            }
        }
        b
    }
}

// ---- MoE plan ----

/// The device-side plan for [`moe_grouped_into`](crate::ComputeDevice::moe_grouped_into), u32
/// words. Built on the device by `moe_plan_into` from router ids, or by the host (offload path)
/// in the same layout. Capacities are fixed so one graph serves any routing.
///
/// ```text
/// [0] n_groups   [1] n_entries   [2] n_missing   [3] 0
/// [GROUP_PTR + 2g, +1]   group g's blob address (lo, hi)              g < CAP
/// [GROUP_START + g]      first entry of group g; [GROUP_START + n_groups] = n_entries
/// [ENT_TOK + e]          token whose activation row entry e reads
/// [ENT_DST + e]          row of `parts` entry e writes: t·TOPK + slot (routed), SHARED_ROW + t
/// [MISSING + i]          expert id of the i-th distinct non-resident expert (address 0)
/// ```
///
/// Order: groups are distinct experts in order of first appearance in routing order (token
/// major, rank minor), then the shared expert; a group's entries ascend in routing order.
/// Non-resident experts get no group: their `parts` rows are left for the host to fill.
pub struct MoePlan;

impl MoePlan {
    /// Group and entry capacity: every routed slot of a full window, plus the shared expert.
    pub const CAP: usize = MAX_T * TOPK + MAX_T;
    /// Word offset of group addresses.
    pub const GROUP_PTR: usize = 4;
    /// Word offset of group starts (CAP + 1 words).
    pub const GROUP_START: usize = Self::GROUP_PTR + 2 * Self::CAP;
    /// Word offset of entry tokens.
    pub const ENT_TOK: usize = Self::GROUP_START + Self::CAP + 1;
    /// Word offset of entry destination rows.
    pub const ENT_DST: usize = Self::ENT_TOK + Self::CAP;
    /// Word offset of the missing-expert list.
    pub const MISSING: usize = Self::ENT_DST + Self::CAP;
    /// Words in a plan.
    pub const WORDS: usize = Self::MISSING + Self::CAP;
    /// First `parts` row of the shared expert (row `SHARED_ROW + t` for token t).
    pub const SHARED_ROW: usize = MAX_T * TOPK;
    /// Rows of `parts` ([`HIDDEN`] floats each).
    pub const PARTS_ROWS: usize = Self::CAP;
    /// Words of `moe_grouped_into` scratch: the SwiGLU activations of every entry, f32.
    pub fn scratch_words() -> usize {
        Self::CAP * FF
    }
}

// ---- GDN ----

/// GDN per-layer parameters. `dt_bias`, `ssm_a` (stored as GGUF stores it, `−exp(A_log)`),
/// `[GDN_HV]`; `norm` (`ssm_norm`) `[GDN_D]`, shared by heads; `conv` (`ssm_conv1d`)
/// `[GDN_CONV][GDN_TAPS]`, tap 0 the oldest.
pub struct GdnParams<'a, B> {
    /// `ssm_conv1d.weight`, `[channel][tap]`.
    pub conv: &'a B,
    /// `ssm_dt.bias`.
    pub dt_bias: &'a B,
    /// `ssm_a` = `−exp(A_log)`.
    pub ssm_a: &'a B,
    /// `ssm_norm.weight`.
    pub norm: &'a B,
}

/// What [`gdn_step`](crate::ComputeDevice::gdn_step) does with the state.
pub enum GdnMode<'a, B> {
    /// Verify: walk all `t` tokens from the stored state, emit outputs, leave the state as it
    /// was.
    ReadOnly,
    /// Commit: replay the first `min(win[N_KEEP], t)` tokens of the same inputs and write the
    /// state. Every output it emits is bitwise what `ReadOnly` emitted for that token.
    Commit {
        /// The window record ([`Win`]).
        win: &'a B,
    },
}

/// The GDN recurrence, per v head `h` (k head `h % 16`) and token, as `gdn_step` pins it:
///
/// ```text
/// g    = exp(softplus(a[h] + dt_bias[h]) * ssm_a[h])   softplus(x) = x > 20 ? x : log1p(exp(x))
/// β    = sigmoid(b[h])
/// S    = g · S                                          S is [i (k dim)][j (v dim)], fp32
/// sk_j = Σ_i S[i][j] k_i ;  d_j = (v_j − sk_j) · β ;  S[i][j] += k_i d_j
/// o_j  = (Σ_i S[i][j] q_i) · 128^-½
/// y_j  = o_j · rsqrt(mean_j o² + eps) · norm_j · sigmoid(z_j)
/// ```
///
/// Each `Σ_i` is four 32-row partial sums (fma chains, i ascending) added in group order.
/// State layout: `[i][h][j]` = `[128][48][128]` f32 per layer.
pub const GDN_STATE: usize = GDN_D * GDN_HV * GDN_D;

/// Conv history: `[GDN_TAPS − 1][GDN_CONV]` f32, oldest row first.
pub const GDN_HIST: usize = (GDN_TAPS - 1) * GDN_CONV;

// ---- hyper-connections ----

/// One hyper-connection read's weights (bf16 except `norm`).
///
/// - `norm` `[HC · HIDDEN]` f32 (`hc_*_norm`, stored as `1 + w`)
/// - `down` `[HC_LR][HC · HIDDEN]` (GGUF order)
/// - `up`: GGUF `[HC · HIDDEN][HC_LR]` repacked by [`hc_up_repack`], so the four rows that gate
///   one output column interleave at 8-value granularity
/// - `inject` `[HC][HC · HIDDEN]`, or `None` for the final read before the head.
pub struct HcWeights<'a, B> {
    /// Per-stream RMSNorm weight.
    pub norm: &'a B,
    /// Down projection.
    pub down: &'a B,
    /// Up projection.
    pub up: &'a B,
    /// Injection rows.
    pub inject: Option<&'a B>,
}

/// A write still owed to the residual, applied by `hc_read_into` before it reads (the fused
/// write-then-read). Both forms are bitwise the separate ops: `Write` is `hc_write(r, y, inj)`,
/// `Moe` is `moe_combine_into(parts, w, logits, stride, sg)` then that write.
pub enum HcPending<'a, B> {
    /// `r[t][c] += y[t] · 2σ(inj[t][c] / HC)`.
    Write {
        /// The block output `[t][HIDDEN]`.
        y: &'a B,
        /// Its injection `[t][HC]`.
        inj: &'a B,
    },
    /// The same with `y` the MoE combine of `parts`.
    Moe {
        /// Expert outputs ([`MoePlan::PARTS_ROWS`] rows).
        parts: &'a B,
        /// Router weights `[t][TOPK]`.
        w: &'a B,
        /// Router logits `[t][stride]` (for the shared-expert gate).
        logits: &'a B,
        /// Row stride of `logits`.
        stride: usize,
        /// Shared-gate column, or `None` without a shared expert.
        sg: Option<usize>,
        /// Injection `[t][HC]`.
        inj: &'a B,
    },
}

/// Repack a hyper-connection up projection (bf16 bits, GGUF `[HC · HIDDEN][HC_LR]`, row
/// `c · HIDDEN + d`) for `hc_read_into`: for each output column `d`, its four rows' 8-value
/// groups interleaved, `out[((d · 40 + p) · 4 + c) · 8 + e] = up[(c · HIDDEN + d) · 320 + 8p + e]`.
pub fn hc_up_repack(up: &[u16]) -> Vec<u16> {
    assert_eq!(up.len(), HC * HIDDEN * HC_LR);
    let mut out = vec![0u16; up.len()];
    for d in 0..HIDDEN {
        for p in 0..HC_LR / 8 {
            for c in 0..HC {
                let src = (c * HIDDEN + d) * HC_LR + 8 * p;
                let dst = ((d * (HC_LR / 8) + p) * HC + c) * 8;
                out[dst..dst + 8].copy_from_slice(&up[src..src + 8]);
            }
        }
    }
    out
}

/// Index of GGUF up element (`row` = `c · HIDDEN + d`, `k`) in the [`hc_up_repack`] layout.
pub fn hc_up_index(row: usize, k: usize) -> usize {
    let (c, d) = (row / HIDDEN, row % HIDDEN);
    ((d * (HC_LR / 8) + k / 8) * HC + c) * 8 + k % 8
}

/// Words of `hc_read_into` scratch for a window of `t`: normalized streams `[t][HC·HIDDEN]`
/// and the bottleneck `[t][HC_LR]`.
pub fn hc_scratch_words(t: usize) -> usize {
    t * (HC * HIDDEN + HC_LR)
}

// ---- QSA ----

/// RoPE tables for the rotated dims: `cos`/`sin` `[max_pos][QSA_ROT / 2]`,
/// `angle = pos · theta^(−2i / QSA_ROT)`, computed in f64.
pub fn rope_table(max_pos: usize, theta: f64) -> (Vec<f32>, Vec<f32>) {
    let half = QSA_ROT / 2;
    let (mut c, mut s) = (
        Vec::with_capacity(max_pos * half),
        Vec::with_capacity(max_pos * half),
    );
    for p in 0..max_pos {
        for i in 0..half {
            let a = p as f64 * theta.powf(-2.0 * i as f64 / QSA_ROT as f64);
            c.push(a.cos() as f32);
            s.push(a.sin() as f32);
        }
    }
    (c, s)
}

/// A QSA layer's norm weights (`[QSA_D]` for q and k, `[IDX_D]` for the indexer's).
pub struct QsaNorms<'a, B> {
    /// `attn_q_norm`.
    pub q: &'a B,
    /// `attn_k_norm`.
    pub k: &'a B,
    /// Indexer query norm.
    pub iq: &'a B,
    /// Indexer key norm.
    pub ik: &'a B,
}

/// A QSA layer's positional state, for a context of up to `max_ctx` cells.
///
/// - `k_cache`, `v_cache`: bf16 (`alloc_bf16`), `[max_ctx][QSA_KV][QSA_D]`, cell-major. Values
///   are the fp32 K (normed, rotated) and V rounded to bf16, nearest-even.
/// - `ring`: `[16][IDX_D]` f32, raw indexer keys by `pos % 16`. Sixteen slots is what makes
///   verify windows safe without a snapshot: a window writes positions `pos0..pos0+8`, and a
///   block still pooling needs at most the 3 cells before `pos0`.
/// - `pooled`: `[max_ctx / 4][IDX_D]` f32: block `b` = rope(rmsnorm(mean of its 4 raw keys) ·
///   ik_norm, position 4b), written when the block's last cell is appended. The mean is
///   `((r0 + r1) + r2) + r3) · 0.25`.
///
/// Positions start at 0 for the sequence (cell index = position).
pub struct QsaCache<'a, B> {
    /// K cache.
    pub k: &'a mut B,
    /// V cache.
    pub v: &'a mut B,
    /// Raw indexer-key ring.
    pub ring: &'a mut B,
    /// Pooled block keys.
    pub pooled: &'a mut B,
}

/// Words of QSA prep output `q`: `[t][QSA_HEADS][QSA_D]` normed, rotated queries, then
/// `[t][IDX_HEADS][IDX_D]` normed, rotated indexer queries.
pub fn qsa_q_words(t: usize) -> usize {
    t * (QSA_HEADS * QSA_D + IDX_HEADS * IDX_D)
}

/// Selection: ids `[t][QSA_WIDTH]` (u32 cells, ascending); token `i` selects
/// `min(n_kv, QSA_WIDTH)` of them, `n_kv = pos0 + i + 1`.
///
/// When `n_kv <= QSA_WIDTH` it is every cell. Otherwise: block `b < n_kv / 4` scores
/// `Σ_h relu(iq_h · pooled_b)` (heads summed in order; each dot is 32 lanes of 4-dim fma chains
/// reduced by an xor butterfly, which the reference reproduces), each cell takes its block's
/// score, the incomplete tail's cells score +∞, and the top `QSA_WIDTH` cells by (score desc,
/// cell asc) are returned ascending.
pub fn qsa_score_blocks(max_ctx: usize) -> usize {
    max_ctx / IDX_BLOCK
}

/// Selected cells per attention chunk (the split-K unit of `qsa_attend_into`).
pub const QSA_CHUNK: usize = 64;

/// Words of `qsa_attend_into` scratch: per token, head and chunk, `[m, l, acc[QSA_D]]`.
pub fn qsa_attend_scratch_words(t: usize) -> usize {
    t * QSA_HEADS * QSA_WIDTH.div_ceil(QSA_CHUNK) * (QSA_D + 2)
}

// ---- helpers for word buffers ----

/// Bits of `u32` words as `f32` buffer elements.
pub fn words_of(w: &[u32]) -> Vec<f32> {
    w.iter().map(|&x| f32::from_bits(x)).collect()
}

/// `u32` words from `f32` buffer elements.
pub fn u32s(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Bytes packed little-endian into words (zero padded).
pub fn bytes_to_words(b: &[u8]) -> Vec<u32> {
    b.chunks(4)
        .map(|c| {
            let mut w = [0u8; 4];
            w[..c.len()].copy_from_slice(c);
            u32::from_le_bytes(w)
        })
        .collect()
}

/// Round to bf16 (nearest even) and back.
pub fn bf16_round(x: f32) -> f32 {
    let b = x.to_bits();
    let r = b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000;
    f32::from_bits(r)
}

/// IEEE half bits to f32.
pub fn f16_to_f32(h: u16) -> f32 {
    let (s, e, m) = (
        (h >> 15) as u32,
        ((h >> 10) & 0x1f) as u32,
        (h & 0x3ff) as u32,
    );
    let bits = match (e, m) {
        (0, 0) => s << 31,
        (0, _) => {
            // Subnormal: normalize.
            let mut e = 127 - 15 + 1;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (s << 31) | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
        (31, 0) => (s << 31) | 0x7f80_0000,
        (31, _) => (s << 31) | 0x7fc0_0000 | (m << 13),
        _ => (s << 31) | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// f32 to IEEE half bits, round to nearest even (for building test weights).
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let m = b & 0x7f_ffff;
    if e >= 31 {
        return s | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return s;
        }
        let m = m | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let r = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let r = if rem > half || (rem == half && r & 1 == 1) {
            r + 1
        } else {
            r
        };
        return s | r as u16;
    }
    let r = m >> 13;
    let rem = m & 0x1fff;
    let mut v = ((e as u32) << 10) | r;
    if rem > 0x1000 || (rem == 0x1000 && r & 1 == 1) {
        v += 1;
    }
    s | v as u16
}
