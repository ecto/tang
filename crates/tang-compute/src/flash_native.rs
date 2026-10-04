//! Native GGUF weight types for the int8-activation GEMV ([`ComputeDevice::native_linear_into`]
//! (crate::ComputeDevice::native_linear_into)): the file's own quantization, repacked once at
//! load into planes the GPU reads with aligned 16-byte loads, no requantization (the engine
//! track measured requantizing dense tensors to Q4X at KL 0.042 against 0.0013 native).
//!
//! # The repacked layout ("NatX")
//!
//! Every type is described per 32-weight chunk by an exact integer code per weight and a scale
//! per 16 weights (and, for Q4_K / Q5_K, a min per 32): `w = sc · code − mn`. Planes, each
//! starting 16-byte aligned, rows contiguous within a plane:
//!
//! ```text
//! L  [n][k/32][lb]  low code bits in activation order: 4-bit types 16 B a chunk, nibbles in
//!                   flash::q4x_slot order; 2/3-bit types 8 B, 2-bit fields in Q2_0 order
//! H  [n][k/32][hb]  extra bits, u32 planes with element (word j, byte b) at bit 8b + j:
//!                   Q5_* 1 plane (bit 4), Q3_K 1 plane (bit 2), Q6_K 2 planes (bits 4, 5)
//! S  scales:        Q4_0 / Q5_0 / IQ4_NL: [n][k/32] f16 d;  Q2_0: [n][k/64] f16 d;
//!                   IQ4_XS: [n][k/256] × 12 B (d f16, 2 B pad, 8 × i8 ls − 32);
//!                   Q3_K / Q6_K: [n][k/256] × 20 B (d f16, 2 B pad, 16 × i8 per-16 scale);
//!                   Q4_K / Q5_K: [n][k/256] × 20 B (d f16, dmin f16, 8 × u8 s, 8 × u8 m)
//! ```
//!
//! # The dot (spec: `cpu::flash::native_linear`)
//!
//! Per chunk `c` and column: `S_h = Σ code · q` over half `h` (exact integers), then
//! `v = fma(sc_1, S_1, sc_0 · S_0)` (per-16 types), `v = sc · (S_0 + S_1)` (one scale a chunk), or
//! `v = fma(sc, S_0 + S_1, −(mn · Σq))` (Q4_K, Q5_K); `acc = fma(d_x, v, acc)`. Scales are
//! formed in f32 exactly (`d · s` of an f16 and a small int). The order chunks are summed in is
//! the backend's.

use crate::flash::{f16_to_f32, q4x_slot, QAct};

/// A GGUF weight type the native GEMV reads (ggml type ids).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NatType {
    /// 32 × 4-bit, f16 d, `(n − 8) d`.
    Q4_0,
    /// 32 × 5-bit, f16 d, `(n − 16) d`.
    Q5_0,
    /// 64 × 2-bit, f16 d, `(n − 1) d` (type 42).
    Q2_0,
    /// 32 × 4-bit grid index, f16 d.
    Iq4Nl,
    /// 256: f16 d, 8 × 6-bit sub-scales, 4-bit grid indices.
    Iq4Xs,
    /// 256: 3-bit codes, 16 × 6-bit sub-scales.
    Q3K,
    /// 256: 4-bit codes, 8 × (6-bit scale, 6-bit min).
    Q4K,
    /// 256: 5-bit codes, 8 × (6-bit scale, 6-bit min).
    Q5K,
    /// 256: 6-bit codes, 16 × i8 sub-scales.
    Q6K,
}

/// llama.cpp's `kvalues_iq4nl`.
pub const IQ4NL_VALUES: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

impl NatType {
    /// From a ggml type id.
    pub fn from_ggml(id: u32) -> Option<Self> {
        Some(match id {
            2 => Self::Q4_0,
            6 => Self::Q5_0,
            42 => Self::Q2_0,
            20 => Self::Iq4Nl,
            23 => Self::Iq4Xs,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            _ => return None,
        })
    }

    /// The ggml type id (also the kernel's template argument).
    pub fn ggml(self) -> u32 {
        match self {
            Self::Q4_0 => 2,
            Self::Q5_0 => 6,
            Self::Q2_0 => 42,
            Self::Iq4Nl => 20,
            Self::Iq4Xs => 23,
            Self::Q3K => 11,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
        }
    }

    /// (weights, bytes) of one GGUF block.
    pub fn block(self) -> (usize, usize) {
        match self {
            Self::Q4_0 | Self::Iq4Nl => (32, 18),
            Self::Q5_0 => (32, 22),
            Self::Q2_0 => (64, 18),
            Self::Iq4Xs => (256, 136),
            Self::Q3K => (256, 110),
            Self::Q4K => (256, 144),
            Self::Q5K => (256, 176),
            Self::Q6K => (256, 210),
        }
    }

    /// Bytes of an `[n, k]` matrix in the GGUF.
    pub fn gguf_bytes(self, n: usize, k: usize) -> usize {
        let (w, b) = self.block();
        n * k / w * b
    }

    /// (L bytes, H bytes) per 32-weight chunk.
    pub fn planes(self) -> (usize, usize) {
        match self {
            Self::Q4_0 | Self::Iq4Nl | Self::Iq4Xs | Self::Q4K => (16, 0),
            Self::Q5_0 | Self::Q5K => (16, 4),
            Self::Q6K => (16, 8),
            Self::Q3K => (8, 4),
            Self::Q2_0 => (8, 0),
        }
    }

    /// Bytes of a row's scales.
    pub fn scale_bytes(self, k: usize) -> usize {
        match self {
            Self::Q4_0 | Self::Q5_0 | Self::Iq4Nl => k / 32 * 2,
            Self::Q2_0 => k / 64 * 2,
            Self::Iq4Xs => k / 256 * 12,
            Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K => k / 256 * 20,
        }
    }

    /// Whether codes are unsigned with a per-32 min (`w = sc · code − mn`).
    pub fn has_min(self) -> bool {
        matches!(self, Self::Q4K | Self::Q5K)
    }

    /// Whether the scale is per 16 weights.
    pub fn per16(self) -> bool {
        matches!(self, Self::Q3K | Self::Q6K)
    }
}

/// Byte offsets of the (L, H, S) planes of an `[n, k]` NatX matrix, and its total size.
pub fn nat_layout(ty: NatType, n: usize, k: usize) -> (usize, usize, usize, usize) {
    let al = |x: usize| x.next_multiple_of(16);
    let (lb, hb) = ty.planes();
    let l = 0;
    let h = al(n * k / 32 * lb);
    let s = h + al(n * k / 32 * hb);
    (l, h, s, s + al(n * ty.scale_bytes(k)))
}

/// The exact decomposition of a matrix: per-weight integer codes, per-16 scales, per-32 mins
/// (`w = scale[e / 16] · code[e] − min[e / 32]`).
pub struct NatIr {
    /// `[n][k]` codes.
    pub code: Vec<i8>,
    /// `[n][k/16]` scales.
    pub scale: Vec<f32>,
    /// `[n][k/32]` mins (zero for types without).
    pub min: Vec<f32>,
}

fn h(b: &[u8], at: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[at], b[at + 1]]))
}

fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xf) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

fn q3k_scales(s: &[u8]) -> [u8; 16] {
    let rd = |i: usize| u32::from_le_bytes([s[i], s[i + 1], s[i + 2], s[i + 3]]);
    let (a0, a1, tmp) = (rd(0), rd(4), rd(8));
    let (m1, m2) = (0x0303_0303u32, 0x0f0f_0f0fu32);
    let aux = [
        (a0 & m2) | ((tmp & m1) << 4),
        (a1 & m2) | (((tmp >> 2) & m1) << 4),
        ((a0 >> 4) & m2) | (((tmp >> 4) & m1) << 4),
        ((a1 >> 4) & m2) | (((tmp >> 6) & m1) << 4),
    ];
    let mut out = [0u8; 16];
    for (i, w) in aux.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
    out
}

/// Decode raw GGUF blocks into the exact integer form. Also yields the K-quant headers' raw
/// fields for the repack (`hdr`: per 256, (d bits, dmin bits, 16 sub-scale bytes)).
fn decode(ty: NatType, raw: &[u8], n: usize, k: usize) -> (NatIr, Vec<(u16, u16, [u8; 16])>) {
    assert!(k.is_multiple_of(256) || (k.is_multiple_of(64) && ty.block().0 <= 64));
    assert_eq!(raw.len(), ty.gguf_bytes(n, k), "{ty:?}: GGUF size");
    let mut code = vec![0i8; n * k];
    let mut scale = vec![0f32; n * k / 16];
    let mut min = vec![0f32; n * k / 32];
    let mut hdr = vec![];
    let (bw, bb) = ty.block();
    let rd16 = |b: &[u8], at: usize| u16::from_le_bytes([b[at], b[at + 1]]);
    for (bi, b) in raw.chunks(bb).enumerate() {
        let e0 = bi * bw;
        let (c, s, mn) = (
            &mut code[e0..e0 + bw],
            &mut scale[e0 / 16..(e0 + bw) / 16],
            &mut min[e0 / 32..(e0 + bw) / 32],
        );
        match ty {
            NatType::Q4_0 | NatType::Iq4Nl => {
                for j in 0..16 {
                    let (lo, hi) = (b[2 + j] & 0xf, b[2 + j] >> 4);
                    if ty == NatType::Q4_0 {
                        c[j] = lo as i8 - 8;
                        c[j + 16] = hi as i8 - 8;
                    } else {
                        c[j] = IQ4NL_VALUES[lo as usize];
                        c[j + 16] = IQ4NL_VALUES[hi as usize];
                    }
                }
                s.fill(h(b, 0));
            }
            NatType::Q5_0 => {
                let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
                for j in 0..16 {
                    let h0 = (((qh >> j) << 4) & 0x10) as u8;
                    let h1 = ((qh >> (j + 12)) & 0x10) as u8;
                    c[j] = ((b[6 + j] & 0xf) | h0) as i8 - 16;
                    c[j + 16] = ((b[6 + j] >> 4) | h1) as i8 - 16;
                }
                s.fill(h(b, 0));
            }
            NatType::Q2_0 => {
                for j in 0..64 {
                    c[j] = ((b[2 + j / 4] >> ((j % 4) * 2)) & 3) as i8 - 1;
                }
                s.fill(h(b, 0));
            }
            NatType::Iq4Xs => {
                let d = h(b, 0);
                let sh = rd16(b, 2) as u32;
                let mut sub = [0u8; 16];
                for ib in 0..8 {
                    let ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) as i32
                        | ((((sh >> (2 * ib)) & 3) << 4) as i32);
                    sub[ib] = (ls - 32) as i8 as u8;
                    s[2 * ib] = d * (ls - 32) as f32;
                    s[2 * ib + 1] = s[2 * ib];
                    let q = &b[8 + ib * 16..8 + ib * 16 + 16];
                    for j in 0..16 {
                        c[ib * 32 + j] = IQ4NL_VALUES[(q[j] & 0xf) as usize];
                        c[ib * 32 + j + 16] = IQ4NL_VALUES[(q[j] >> 4) as usize];
                    }
                }
                hdr.push((rd16(b, 0), 0, sub));
            }
            NatType::Q3K => {
                let hm = &b[0..32];
                let d = h(b, 108);
                let sc = q3k_scales(&b[96..108]);
                let mut sub = [0u8; 16];
                let (mut yi, mut is, mut m) = (0, 0, 1u8);
                for nn in 0..2 {
                    let q = &b[32 + nn * 32..64 + nn * 32];
                    for shift in [0u32, 2, 4, 6] {
                        for half in 0..2 {
                            sub[is] = (sc[is] as i32 - 32) as i8 as u8;
                            s[is] = d * (sc[is] as i32 - 32) as f32;
                            is += 1;
                            for l in 0..16 {
                                let li = l + 16 * half;
                                c[yi] = ((q[li] >> shift) & 3) as i8
                                    - if hm[li] & m != 0 { 0 } else { 4 };
                                yi += 1;
                            }
                        }
                        m <<= 1;
                    }
                }
                hdr.push((rd16(b, 108), 0, sub));
            }
            NatType::Q4K | NatType::Q5K => {
                let (d, dmin) = (h(b, 0), h(b, 2));
                let scb = &b[4..16];
                let mut sub = [0u8; 16];
                let (qs, qh) = if ty == NatType::Q4K {
                    (16, 0)
                } else {
                    (48, 16)
                };
                for j in 0..4 {
                    let (s1, m1) = scale_min_k4(2 * j, scb);
                    let (s2, m2) = scale_min_k4(2 * j + 1, scb);
                    sub[2 * j] = s1;
                    sub[2 * j + 1] = s2;
                    sub[8 + 2 * j] = m1;
                    sub[8 + 2 * j + 1] = m2;
                    for l in 0..32 {
                        let q = b[qs + j * 32 + l];
                        let (mut lo, mut hi) = (q & 0xf, q >> 4);
                        if ty == NatType::Q5K {
                            let hb = b[qh + l];
                            lo |= ((hb >> (2 * j)) & 1) << 4;
                            hi |= ((hb >> (2 * j + 1)) & 1) << 4;
                        }
                        c[j * 64 + l] = lo as i8;
                        c[j * 64 + 32 + l] = hi as i8;
                    }
                    s[4 * j] = d * s1 as f32;
                    s[4 * j + 1] = s[4 * j];
                    s[4 * j + 2] = d * s2 as f32;
                    s[4 * j + 3] = s[4 * j + 2];
                    mn[2 * j] = dmin * m1 as f32;
                    mn[2 * j + 1] = dmin * m2 as f32;
                }
                hdr.push((rd16(b, 0), rd16(b, 2), sub));
            }
            NatType::Q6K => {
                let d = h(b, 208);
                let mut sub = [0u8; 16];
                sub.copy_from_slice(&b[192..208]);
                for nn in 0..2 {
                    let (ql, qh) = (
                        &b[nn * 64..nn * 64 + 64],
                        &b[128 + nn * 32..128 + nn * 32 + 32],
                    );
                    for l in 0..32 {
                        let y = &mut c[nn * 128..nn * 128 + 128];
                        y[l] = ((ql[l] & 0xf) | ((qh[l] & 3) << 4)) as i8 - 32;
                        y[l + 32] = ((ql[l + 32] & 0xf) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
                        y[l + 64] = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
                        y[l + 96] = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
                    }
                }
                for (i, v) in s.iter_mut().enumerate() {
                    *v = d * (sub[i] as i8) as f32;
                }
                hdr.push((rd16(b, 208), 0, sub));
            }
        }
    }
    (NatIr { code, scale, min }, hdr)
}

/// The exact integer form of a GGUF matrix (for references and tests).
pub fn nat_decode(ty: NatType, raw: &[u8], n: usize, k: usize) -> NatIr {
    decode(ty, raw, n, k).0
}

/// Repack a GGUF matrix `[n, k]` into NatX (module docs).
pub fn nat_repack(ty: NatType, raw: &[u8], n: usize, k: usize) -> Vec<u8> {
    let (ir, hdr) = decode(ty, raw, n, k);
    let (lo, ho, so, total) = nat_layout(ty, n, k);
    let (lb, hb) = ty.planes();
    let mut out = vec![0u8; total];
    for row in 0..n {
        for c in 0..k / 32 {
            let ch = row * (k / 32) + c;
            let codes = &ir.code[row * k + c * 32..row * k + c * 32 + 32];
            // Unsigned stored value per element (the kernel adds the offset back).
            let stored = |i: usize| -> u32 {
                let v = codes[i] as i32;
                (match ty {
                    NatType::Q4_0 => v + 8,
                    NatType::Q5_0 => v + 16,
                    NatType::Q2_0 => v + 1,
                    NatType::Q3K => v + 4,
                    NatType::Q6K => v + 32,
                    NatType::Iq4Nl | NatType::Iq4Xs => {
                        IQ4NL_VALUES.iter().position(|&g| g as i32 == v).unwrap() as i32
                    }
                    NatType::Q4K | NatType::Q5K => v,
                }) as u32
            };
            let l = &mut out[lo + ch * lb..lo + ch * lb + lb];
            for i in 0..32 {
                let s = stored(i);
                if lb == 16 {
                    let (byte, sh) = q4x_slot(i);
                    l[byte] |= ((s & 0xf) as u8) << sh;
                } else {
                    // Q2_0 field order: element 16h + 4b + f in field f of byte b of word h.
                    let (hh, b, f) = (i / 16, (i % 16) / 4, i % 4);
                    l[4 * hh + b] |= ((s & 3) as u8) << (2 * f);
                }
            }
            if hb > 0 {
                let hp = &mut out[ho + ch * hb..ho + ch * hb + hb];
                for i in 0..32 {
                    let s = stored(i);
                    let (j, b) = QAct::slot(i);
                    let bit = 8 * b + j;
                    let (b0, b1) = match ty {
                        NatType::Q3K => ((s >> 2) & 1, 0),
                        NatType::Q6K => ((s >> 4) & 1, (s >> 5) & 1),
                        _ => ((s >> 4) & 1, 0),
                    };
                    let mut put = |plane: usize, v: u32| {
                        let w =
                            u32::from_le_bytes(hp[4 * plane..4 * plane + 4].try_into().unwrap())
                                | (v << bit);
                        hp[4 * plane..4 * plane + 4].copy_from_slice(&w.to_le_bytes());
                    };
                    put(0, b0);
                    if hb == 8 {
                        put(1, b1);
                    }
                }
            }
        }
        // Scales.
        let srow = &mut out[so + row * ty.scale_bytes(k)..so + (row + 1) * ty.scale_bytes(k)];
        let (bw, bb) = ty.block();
        match ty {
            NatType::Q4_0 | NatType::Q5_0 | NatType::Iq4Nl | NatType::Q2_0 => {
                for blk in 0..k / bw {
                    let src = &raw[(row * (k / bw) + blk) * bb..];
                    srow[2 * blk..2 * blk + 2].copy_from_slice(&src[..2]);
                }
            }
            _ => {
                let hsz = if ty == NatType::Iq4Xs { 12 } else { 20 };
                for sb in 0..k / 256 {
                    let (d, dmin, sub) = hdr[row * (k / 256) + sb];
                    let o = &mut srow[sb * hsz..(sb + 1) * hsz];
                    o[0..2].copy_from_slice(&d.to_le_bytes());
                    o[2..4].copy_from_slice(&dmin.to_le_bytes());
                    o[4..hsz].copy_from_slice(&sub[..hsz - 4]);
                }
            }
        }
    }
    out
}

/// Decode a NatX matrix back to its exact integer form (what the kernels see): codes, per-16
/// scales, per-32 mins.
pub fn nat_unpack(ty: NatType, wb: &[u8], n: usize, k: usize) -> NatIr {
    let (lo, ho, so, _) = nat_layout(ty, n, k);
    let (lb, hb) = ty.planes();
    let mut ir = NatIr {
        code: vec![0; n * k],
        scale: vec![0.0; n * k / 16],
        min: vec![0.0; n * k / 32],
    };
    let rd16 = |at: usize| u16::from_le_bytes([wb[at], wb[at + 1]]);
    for row in 0..n {
        for c in 0..k / 32 {
            let ch = row * (k / 32) + c;
            for i in 0..32 {
                let mut s = if lb == 16 {
                    let (byte, sh) = q4x_slot(i);
                    ((wb[lo + ch * lb + byte] >> sh) & 0xf) as u32
                } else {
                    let (hh, b, f) = (i / 16, (i % 16) / 4, i % 4);
                    ((wb[lo + ch * lb + 4 * hh + b] >> (2 * f)) & 3) as u32
                };
                if hb > 0 {
                    let (j, b) = QAct::slot(i);
                    let rdw = |p: usize| {
                        u32::from_le_bytes(
                            wb[ho + ch * hb + 4 * p..ho + ch * hb + 4 * p + 4]
                                .try_into()
                                .unwrap(),
                        )
                    };
                    let b0 = (rdw(0) >> (8 * b + j)) & 1;
                    s |= match ty {
                        NatType::Q3K => b0 << 2,
                        NatType::Q6K => (b0 << 4) | (((rdw(1) >> (8 * b + j)) & 1) << 5),
                        _ => b0 << 4,
                    };
                }
                ir.code[row * k + c * 32 + i] = match ty {
                    NatType::Q4_0 => s as i32 - 8,
                    NatType::Q5_0 => s as i32 - 16,
                    NatType::Q2_0 => s as i32 - 1,
                    NatType::Q3K => s as i32 - 4,
                    NatType::Q6K => s as i32 - 32,
                    NatType::Iq4Nl | NatType::Iq4Xs => IQ4NL_VALUES[s as usize] as i32,
                    NatType::Q4K | NatType::Q5K => s as i32,
                } as i8;
            }
        }
        let srow = so + row * ty.scale_bytes(k);
        for e16 in 0..k / 16 {
            let e = e16 * 16;
            let (sc, mn) = match ty {
                NatType::Q4_0 | NatType::Q5_0 | NatType::Iq4Nl => {
                    (f16_to_f32(rd16(srow + 2 * (e / 32))), 0.0)
                }
                NatType::Q2_0 => (f16_to_f32(rd16(srow + 2 * (e / 64))), 0.0),
                NatType::Iq4Xs => {
                    let hd = srow + (e / 256) * 12;
                    (
                        f16_to_f32(rd16(hd)) * (wb[hd + 4 + (e % 256) / 32] as i8) as f32,
                        0.0,
                    )
                }
                NatType::Q3K | NatType::Q6K => {
                    let hd = srow + (e / 256) * 20;
                    (
                        f16_to_f32(rd16(hd)) * (wb[hd + 4 + (e % 256) / 16] as i8) as f32,
                        0.0,
                    )
                }
                NatType::Q4K | NatType::Q5K => {
                    let hd = srow + (e / 256) * 20;
                    let j = (e % 256) / 32;
                    (
                        f16_to_f32(rd16(hd)) * wb[hd + 4 + j] as f32,
                        f16_to_f32(rd16(hd + 2)) * wb[hd + 12 + j] as f32,
                    )
                }
            };
            ir.scale[row * k / 16 + e16] = sc;
            ir.min[row * k / 32 + e16 / 2] = mn;
        }
    }
    ir
}

/// Test helpers (random GGUF blocks of each type), public for the GPU parity tests and the
/// benchmark.
pub mod tests {
    use super::*;

    /// Random GGUF blocks of `ty` with sane scales (f16 fields in [0.002, 0.03]).
    pub fn random_gguf(ty: NatType, n: usize, k: usize, seed: u64) -> Vec<u8> {
        let mut s = seed | 1;
        let mut byte = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s as u8
        };
        let (_, bb) = ty.block();
        let mut raw = vec![0u8; ty.gguf_bytes(n, k)];
        for blk in raw.chunks_mut(bb) {
            for v in blk.iter_mut() {
                *v = byte();
            }
            let f16 = |b: &mut dyn FnMut() -> u8| {
                crate::flash::f32_to_f16(0.002 + 0.028 * b() as f32 / 255.0).to_le_bytes()
            };
            let at = match ty {
                NatType::Q3K => vec![108],
                NatType::Q6K => vec![208],
                NatType::Q4K | NatType::Q5K => vec![0, 2],
                _ => vec![0],
            };
            for a in at {
                let v = f16(&mut byte);
                blk[a..a + 2].copy_from_slice(&v);
            }
        }
        raw
    }

    pub const ALL: [NatType; 9] = [
        NatType::Q4_0,
        NatType::Q5_0,
        NatType::Q2_0,
        NatType::Iq4Nl,
        NatType::Iq4Xs,
        NatType::Q3K,
        NatType::Q4K,
        NatType::Q5K,
        NatType::Q6K,
    ];

    #[test]
    fn repack_round_trips_the_exact_form() {
        for ty in ALL {
            let (n, k) = (3, 512);
            let raw = random_gguf(ty, n, k, ty.ggml() as u64);
            let ir = nat_decode(ty, &raw, n, k);
            let back = nat_unpack(ty, &nat_repack(ty, &raw, n, k), n, k);
            assert_eq!(ir.code, back.code, "{ty:?} codes");
            assert_eq!(ir.scale, back.scale, "{ty:?} scales");
            assert_eq!(ir.min, back.min, "{ty:?} mins");
        }
    }

    #[test]
    fn decode_matches_llama_dequant_on_hand_built_blocks() {
        // Q4_0: d = 0.5, bytes 0x21 -> elements (1 - 8) * 0.5 and (2 - 8) * 0.5.
        let mut b = vec![0u8; 18];
        b[..2].copy_from_slice(&crate::flash::f32_to_f16(0.5).to_le_bytes());
        b[2] = 0x21;
        let ir = nat_decode(NatType::Q4_0, &[b.clone(), b.clone()].concat(), 1, 64);
        assert_eq!((ir.code[0], ir.code[16], ir.scale[0]), (-7, -6, 0.5));
        // Q2_0: code byte 0b11_10_01_00 -> -1, 0, 1, 2.
        let mut q = vec![0u8; 18];
        q[2] = 0b1110_0100;
        let ir = nat_decode(NatType::Q2_0, &q, 1, 64);
        assert_eq!(&ir.code[..4], &[-1, 0, 1, 2]);
    }
}
