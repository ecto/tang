//! The data contract shared with the GPU kernels, and the scalar reference that is the spec for
//! the CPU expert kernels.
//!
//! This mirrors `tang-compute/src/flash.rs` and `cpu/flash.rs` on the kernel track
//! (`~/Developer/tang-kernels`, branch `claude/flash-kernels`, read 2026-10-04, uncommitted
//! there): [`QAct`] (int8 activations), the repacked Q2_0 [`ExpertBlob`], and [`MoePlan`]. It
//! is copied rather than imported because that code has not landed; when it does, this module
//! should re-export it instead. The reference functions follow theirs line for line
//! (`quantize_act_rows`, `q2_row_dot`, `moe_grouped`).

/// Model width.
pub const HIDDEN: usize = 2560;
/// Expert FFN width.
pub const FF: usize = 640;
/// Routed experts per token.
pub const TOPK: usize = 10;
/// Largest verify window.
pub const MAX_T: usize = 8;

/// int8-quantized activations, `m` rows of `k` (`k % 32 == 0`), as u32 words:
///
/// ```text
/// [m][k/4]   int8 codes, 4 per word, chunk-permuted
/// [m][k/32]  f32 scale d of each 32-wide chunk
/// [m][k/32]  i32 Σ q over the chunk
/// ```
///
/// Per 32-element chunk: `amax = max |x|`, `d = amax / 127` (f32),
/// `q = d == 0 ? 0 : clamp(round_half_away_from_zero(x / d), -127, 127)`.
/// Word `4h + f` (`h` < 2, `f` < 4) of a chunk holds the codes of elements `16h + f`,
/// `16h + 4 + f`, `16h + 8 + f`, `16h + 12 + f` in bytes 0..4: the order in which
/// `(w >> 2f) & 0x03030303` extracts codes from word `h` of a Q2_0 chunk.
#[derive(Clone, Copy, Debug)]
pub struct QAct {
    pub m: usize,
    pub k: usize,
}

impl QAct {
    pub const fn words(self) -> usize {
        self.m * self.k / 4 + 2 * self.m * self.k / 32
    }
    pub const fn codes(self, r: usize) -> usize {
        r * self.k / 4
    }
    pub const fn scales(self, r: usize) -> usize {
        self.m * self.k / 4 + r * self.k / 32
    }
    pub const fn sums(self, r: usize) -> usize {
        self.m * self.k / 4 + self.m * self.k / 32 + r * self.k / 32
    }
    /// (word, byte) of element `e` inside its row's codes.
    pub fn slot(e: usize) -> (usize, usize) {
        let (c, i) = (e / 32, e % 32);
        let (h, b, f) = (i / 16, (i % 16) / 4, i % 4);
        (c * 8 + 4 * h + f, b)
    }
}

/// Bytes of one GGUF Q2_0 block: fp16 `d`, then 16 bytes of 2-bit codes, 64 weights.
pub const Q2_BLOCK_BYTES: usize = 18;

/// Bytes of an `[n, k]` Q2_0 matrix.
pub const fn q2_bytes(n: usize, k: usize) -> usize {
    n * k / 64 * Q2_BLOCK_BYTES
}

/// One routed expert: three repacked Q2_0 matrices, `gate [FF, HIDDEN] | up [FF, HIDDEN] |
/// down [HIDDEN, FF]`. Repacked = `[n][k/4]` code bytes (element `4j + i` of a 64-weight
/// block in bits `2i..2i+2` of code byte `j`, LSB first), then `[n][k/64]` fp16 scales;
/// `w = (code − 1) · d`.
pub struct ExpertBlob;

impl ExpertBlob {
    pub const GATE: usize = 0;
    pub const UP: usize = q2_bytes(FF, HIDDEN);
    pub const DOWN: usize = 2 * Self::UP;
    pub const BYTES: usize = 3 * Self::UP;
}

/// Repack a GGUF Q2_0 `[n, k]` matrix (blocks of fp16 `d` + 16 code bytes) into the device
/// layout.
pub fn q2_repack(raw: &[u8], n: usize, k: usize) -> Vec<u8> {
    assert!(k.is_multiple_of(64));
    assert_eq!(raw.len(), q2_bytes(n, k));
    let (nb, codes) = (n * k / 64, n * k / 4);
    let mut out = vec![0u8; raw.len()];
    for b in 0..nb {
        let blk = &raw[b * 18..b * 18 + 18];
        out[codes + 2 * b..codes + 2 * b + 2].copy_from_slice(&blk[..2]);
        out[16 * b..16 * b + 16].copy_from_slice(&blk[2..]);
    }
    out
}

/// The MoE plan (u32 words), built by the host on the offload path in the same layout the
/// device-side `moe_plan_into` uses.
///
/// ```text
/// [0] n_groups   [1] n_entries   [2] n_missing   [3] 0
/// [GROUP_PTR + 2g, +1]   group g's blob address (lo, hi)
/// [GROUP_START + g]      first entry of group g; [GROUP_START + n_groups] = n_entries
/// [ENT_TOK + e]          token whose activation row entry e reads
/// [ENT_DST + e]          `parts` row entry e writes: t·TOPK + rank, or SHARED_ROW + t
/// [MISSING + i]          expert id of the i-th distinct non-resident expert
/// ```
///
/// Groups are distinct resident experts in order of first appearance in routing order (token
/// major, rank minor), then the shared expert; a group's entries ascend in routing order.
/// Non-resident experts get no group; their `parts` rows are filled from the host.
pub struct MoePlan;

impl MoePlan {
    pub const CAP: usize = MAX_T * TOPK + MAX_T;
    pub const GROUP_PTR: usize = 4;
    pub const GROUP_START: usize = Self::GROUP_PTR + 2 * Self::CAP;
    pub const ENT_TOK: usize = Self::GROUP_START + Self::CAP + 1;
    pub const ENT_DST: usize = Self::ENT_TOK + Self::CAP;
    pub const MISSING: usize = Self::ENT_DST + Self::CAP;
    pub const WORDS: usize = Self::MISSING + Self::CAP;
    pub const SHARED_ROW: usize = MAX_T * TOPK;
    pub const PARTS_ROWS: usize = Self::CAP;
}

pub fn f16_to_f32(h: u16) -> f32 {
    let (s, e, m) = (
        (h >> 15) as u32,
        ((h >> 10) & 0x1f) as u32,
        (h & 0x3ff) as u32,
    );
    let bits = match (e, m) {
        (0, 0) => s << 31,
        (0, _) => {
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

/// f32 to IEEE half, round to nearest even (normal range; for building test weights).
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    if e <= 0 {
        return s;
    }
    if e >= 31 {
        return s | 0x7c00;
    }
    let m = b & 0x7f_ffff;
    let mut h = ((e as u32) << 10) | (m >> 13);
    let rest = m & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && h & 1 == 1) {
        h += 1;
    }
    s | h as u16
}

/// SiLU with tang-compute's pinned exponential (`flash::pexp`), so a CPU-computed expert row is
/// bitwise the GPU's.
pub fn silu(x: f32) -> f32 {
    x / (1.0 + pexp(-x))
}

/// `exp` as a pinned sequence of IEEE f32 operations (tang-compute `flash::pexp`, copied).
pub fn pexp(x: f32) -> f32 {
    const LN2_HI: f32 = f32::from_bits(0x3f31_7200);
    const LN2_LO: f32 = f32::from_bits(0x35bf_be8e);
    const C: [f32; 8] = [
        f32::from_bits(0x3950_0d01),
        f32::from_bits(0x3ab6_0b61),
        f32::from_bits(0x3c08_8889),
        f32::from_bits(0x3d2a_aaab),
        f32::from_bits(0x3e2a_aaab),
        0.5,
        1.0,
        1.0,
    ];
    let x = x.clamp(-87.0, 88.0);
    let n = (x * std::f32::consts::LOG2_E).round_ties_even();
    let r = (-n).mul_add(LN2_LO, (-n).mul_add(LN2_HI, x));
    let mut p = C[0];
    for &c in &C[1..] {
        p = p.mul_add(r, c);
    }
    p * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

/// Quantize rows `r0..r0 + rows` of `x` into `w` per the [`QAct`] contract.
pub fn quantize_act_rows(x: &[f32], l: QAct, r0: usize, rows: usize, w: &mut [u32]) {
    let k = l.k;
    for r in r0..r0 + rows {
        let xr = &x[(r - r0) * k..(r - r0 + 1) * k];
        for i in 0..k / 4 {
            w[l.codes(r) + i] = 0;
        }
        for c in 0..k / 32 {
            let xs = &xr[c * 32..c * 32 + 32];
            let amax = xs.iter().fold(0f32, |a, v| a.max(v.abs()));
            let d = amax / 127.0;
            let mut hx = 0i32;
            for (i, &v) in xs.iter().enumerate() {
                let q = if d == 0.0 {
                    0
                } else {
                    (v / d).round().clamp(-127.0, 127.0) as i32
                };
                hx += q;
                let (wi, b) = QAct::slot(c * 32 + i);
                w[l.codes(r) + wi] |= ((q as i8 as u8) as u32) << (8 * b);
            }
            w[l.scales(r) + c] = d.to_bits();
            w[l.sums(r) + c] = hx as u32;
        }
    }
}

pub fn quantize_act(x: &[f32], m: usize, k: usize) -> Vec<u32> {
    let l = QAct { m, k };
    let mut w = vec![0u32; l.words()];
    quantize_act_rows(x, l, 0, m, &mut w);
    w
}

/// One row of int8 activations in element order.
pub struct QRow {
    pub q: Vec<i32>,
    pub d: Vec<f32>,
    pub hx: Vec<i32>,
}

impl QRow {
    pub fn decode(xq: &[u32], l: QAct, r: usize) -> Self {
        let q = (0..l.k)
            .map(|e| {
                let (wi, by) = QAct::slot(e);
                (xq[l.codes(r) + wi] >> (8 * by)) as u8 as i8 as i32
            })
            .collect();
        let d = (0..l.k / 32)
            .map(|c| f32::from_bits(xq[l.scales(r) + c]))
            .collect();
        let hx = (0..l.k / 32).map(|c| xq[l.sums(r) + c] as i32).collect();
        QRow { q, d, hx }
    }
}

/// Integer part of chunk `c` of row `o`: `Σ code·q − Σ q`.
pub fn q2_chunk_int(wb: &[u8], k: usize, o: usize, c: usize, x: &QRow) -> i32 {
    let codes = &wb[o * k / 4..(o + 1) * k / 4];
    let mut s = 0i32;
    for (j, &byte) in codes[c * 8..c * 8 + 8].iter().enumerate() {
        for f in 0..4 {
            s += ((byte >> (2 * f)) & 3) as i32 * x.q[c * 32 + 4 * j + f];
        }
    }
    s - x.hx[c]
}

/// Row `o` of repacked `[n, k]` Q2_0 `wb` against `x`:
/// `Σ_chunks fma(d_block · d_x, (Σ code·q − Σ q), acc)`, chunks ascending.
pub fn q2_row_dot(wb: &[u8], n: usize, k: usize, o: usize, x: &QRow) -> f32 {
    let sc = n * k / 4 + o * (k / 64) * 2;
    let mut acc = 0f32;
    for c in 0..k / 32 {
        let b = sc + 2 * (c / 2);
        let d = f16_to_f32(u16::from_le_bytes([wb[b], wb[b + 1]]));
        acc = (d * x.d[c]).mul_add(q2_chunk_int(wb, k, o, c, x) as f32, acc);
    }
    acc
}

/// The reference expert: `down(quantize(silu(gate·x̂) · (up·x̂)))` for one token row of `xq`
/// (`QAct { m, k: HIDDEN }`, row `r`). Returns `HIDDEN` floats, not router-weighted.
pub fn expert_ref(blob: &[u8], xq: &[u32], m: usize, r: usize) -> Vec<f32> {
    let x = QRow::decode(xq, QAct { m, k: HIDDEN }, r);
    let (gate, up, down) = (
        &blob[ExpertBlob::GATE..ExpertBlob::UP],
        &blob[ExpertBlob::UP..ExpertBlob::DOWN],
        &blob[ExpertBlob::DOWN..ExpertBlob::BYTES],
    );
    let h: Vec<f32> = (0..FF)
        .map(|o| silu(q2_row_dot(gate, FF, HIDDEN, o, &x)) * q2_row_dot(up, FF, HIDDEN, o, &x))
        .collect();
    let hq = QRow::decode(&quantize_act(&h, 1, FF), QAct { m: 1, k: FF }, 0);
    (0..HIDDEN)
        .map(|o| q2_row_dot(down, HIDDEN, FF, o, &hq))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blob_is_flash_next_size() {
        assert_eq!(ExpertBlob::BYTES, 1_382_400);
    }

    #[test]
    fn quantize_rounds_half_away_and_sums() {
        // amax 127 → d = 1; 2.5 → 3, -2.5 → -3.
        let mut x = vec![0f32; 32];
        x[0] = 127.0;
        x[1] = 2.5;
        x[2] = -2.5;
        let w = quantize_act(&x, 1, 32);
        let r = QRow::decode(&w, QAct { m: 1, k: 32 }, 0);
        assert_eq!(&r.q[..3], &[127, 3, -3]);
        assert_eq!(r.d[0], 1.0);
        assert_eq!(r.hx[0], 127);
    }

    #[test]
    fn f16_roundtrip() {
        for v in [0.0f32, 1.0, -0.5, 0.0099, 65504.0, 2.71] {
            let h = f32_to_f16(v);
            assert!((f16_to_f32(h) - v).abs() <= v.abs() * 1e-3);
        }
    }
}
