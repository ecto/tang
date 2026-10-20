//! Adaptive binary arithmetic coding — integer only, bit-exact everywhere.
//!
//! A carry-propagating 32-bit range coder (the LZMA construction) driven by
//! 16-bit probabilities. Everything from a probability to an output byte is
//! integer arithmetic, so any backend that feeds the same probabilities
//! produces and consumes the same bytes. Probabilities come either from
//! [`BitModel`] (two-rate adaptive counter) or straight from a model (e.g. the
//! fixed-point network in [`crate::learned`]) via [`Encoder::encode_p`].

/// Probability scale: `p1 ∈ [P_MIN, ONE − P_MIN]` means `P(bit = 1) = p1 / ONE`.
pub const ONE: u32 = 1 << 16;
/// Smallest probability either symbol may get.
pub const P_MIN: u32 = 32;
const TOP: u32 = 1 << 24;

/// Clamp a probability of a one into the codable range.
#[inline]
pub fn clamp_p(p1: u32) -> u32 {
    p1.clamp(P_MIN, ONE - P_MIN)
}

/// Two-rate adaptive bit model (fast + slow exponential counters, averaged).
#[derive(Clone, Copy, Debug)]
pub struct BitModel {
    fast: u16,
    slow: u16,
}

impl Default for BitModel {
    fn default() -> Self {
        Self::new(ONE / 2)
    }
}

impl BitModel {
    /// Model starting at `P(1) = p1 / 2^16`.
    pub fn new(p1: u32) -> Self {
        let p = clamp_p(p1) as u16;
        Self { fast: p, slow: p }
    }
    /// `fast | slow << 16` (the GPU decoder's model word).
    pub fn packed(&self) -> u32 {
        self.fast as u32 | (self.slow as u32) << 16
    }
    /// Current `P(1)` in 16-bit fixed point.
    #[inline]
    pub fn p1(&self) -> u32 {
        clamp_p((self.fast as u32 + self.slow as u32) >> 1)
    }
    /// Adapt towards `bit`.
    #[inline]
    pub fn update(&mut self, bit: bool) {
        if bit {
            self.fast += (65535 - self.fast) >> 4;
            self.slow += (65535 - self.slow) >> 7;
        } else {
            self.fast -= self.fast >> 4;
            self.slow -= self.slow >> 7;
        }
    }
}

/// Range encoder.
#[derive(Debug)]
pub struct Encoder {
    low: u64,
    range: u32,
    cache: u8,
    cache_size: u64,
    out: Vec<u8>,
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Encoder {
    /// Fresh encoder.
    pub fn new() -> Self {
        Self { low: 0, range: u32::MAX, cache: 0, cache_size: 1, out: Vec::new() }
    }

    fn shift_low(&mut self) {
        if self.low < 0xFF00_0000 || self.low > 0xFFFF_FFFF {
            let carry = (self.low >> 32) as u8;
            let mut temp = self.cache;
            loop {
                self.out.push(temp.wrapping_add(carry));
                temp = 0xFF;
                self.cache_size -= 1;
                if self.cache_size == 0 {
                    break;
                }
            }
            self.cache = ((self.low >> 24) & 0xFF) as u8;
        }
        self.cache_size += 1;
        self.low = (self.low & 0x00FF_FFFF) << 8;
    }

    /// Code `bit` with `P(1) = p1 / 2^16` (clamped).
    #[inline]
    pub fn encode_p(&mut self, bit: bool, p1: u32) {
        let p0 = ONE - clamp_p(p1);
        let bound = (self.range >> 16) * p0;
        if bit {
            self.low += bound as u64;
            self.range -= bound;
        } else {
            self.range = bound;
        }
        while self.range < TOP {
            self.range <<= 8;
            self.shift_low();
        }
    }

    /// Code `bit` with an adaptive model, then adapt it.
    #[inline]
    pub fn encode(&mut self, bit: bool, m: &mut BitModel) {
        self.encode_p(bit, m.p1());
        m.update(bit);
    }

    /// `n` raw (equiprobable) bits of `v`, MSB first.
    pub fn encode_raw(&mut self, v: u64, n: u32) {
        for i in (0..n).rev() {
            self.encode_p((v >> i) & 1 == 1, ONE / 2);
        }
    }

    /// Flush and return the bytes.
    pub fn finish(mut self) -> Vec<u8> {
        for _ in 0..5 {
            self.shift_low();
        }
        // The first byte is always the initial zero cache; drop it.
        self.out.remove(0);
        self.out
    }
}

/// Range decoder over a byte slice (reads past the end as zeros).
#[derive(Debug)]
pub struct Decoder<'a> {
    code: u32,
    range: u32,
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Decoder<'a> {
    /// Start decoding `buf` (as produced by [`Encoder::finish`]).
    pub fn new(buf: &'a [u8]) -> Self {
        let mut d = Self { code: 0, range: u32::MAX, buf, pos: 0 };
        for _ in 0..4 {
            d.code = (d.code << 8) | d.next() as u32;
        }
        d
    }

    #[inline]
    fn next(&mut self) -> u8 {
        let b = self.buf.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        b
    }

    /// Decode a bit coded with `P(1) = p1 / 2^16`.
    #[inline]
    pub fn decode_p(&mut self, p1: u32) -> bool {
        let p0 = ONE - clamp_p(p1);
        let bound = (self.range >> 16) * p0;
        let bit = if self.code < bound {
            self.range = bound;
            false
        } else {
            self.code -= bound;
            self.range -= bound;
            true
        };
        while self.range < TOP {
            self.range <<= 8;
            self.code = (self.code << 8) | self.next() as u32;
        }
        bit
    }

    /// Decode with an adaptive model, then adapt it.
    #[inline]
    pub fn decode(&mut self, m: &mut BitModel) -> bool {
        let b = self.decode_p(m.p1());
        m.update(b);
        b
    }

    /// `n` raw bits, MSB first.
    pub fn decode_raw(&mut self, n: u32) -> u64 {
        let mut v = 0;
        for _ in 0..n {
            v = (v << 1) | self.decode_p(ONE / 2) as u64;
        }
        v
    }

    /// Whether the decoder consumed more than the buffer (corrupt stream).
    pub fn overrun(&self) -> bool {
        self.pos > self.buf.len() + 4
    }
}

/// Adaptive Elias-gamma style coder for unbounded non-negative integers:
/// unary bucket `k = ⌊log₂(v+1)⌋` with one model per bucket position, then
/// the `k` mantissa bits (top `MODELED` with models, the rest raw).
#[derive(Clone, Debug)]
pub struct UintModel {
    unary: Vec<BitModel>,
    mant: Vec<BitModel>,
}

const UBITS: usize = 40;
const MODELED: usize = 2;

impl Default for UintModel {
    fn default() -> Self {
        Self { unary: vec![BitModel::default(); UBITS], mant: vec![BitModel::default(); UBITS * MODELED] }
    }
}

impl UintModel {
    /// Encode `v`.
    pub fn encode(&mut self, e: &mut Encoder, v: u64) {
        let w = v + 1;
        let k = 63 - w.leading_zeros() as usize;
        for i in 0..k {
            e.encode(true, &mut self.unary[i]);
        }
        if k < UBITS - 1 {
            e.encode(false, &mut self.unary[k]);
        }
        for (j, i) in (0..k).rev().enumerate() {
            let bit = (w >> i) & 1 == 1;
            if j < MODELED {
                e.encode(bit, &mut self.mant[k * MODELED + j]);
            } else {
                e.encode_p(bit, ONE / 2);
            }
        }
    }
    /// Decode one value.
    pub fn decode(&mut self, d: &mut Decoder) -> u64 {
        let mut k = 0;
        while k < UBITS - 1 && d.decode(&mut self.unary[k]) {
            k += 1;
        }
        let mut w = 1u64;
        for j in 0..k {
            let bit = if j < MODELED { d.decode(&mut self.mant[k * MODELED + j]) } else { d.decode_p(ONE / 2) };
            w = (w << 1) | bit as u64;
        }
        w - 1
    }
}

/// Ideal code length in bits of coding `bit` at `P(1) = p1 / 2^16`.
#[inline]
pub fn cost_bits(bit: bool, p1: u32) -> f64 {
    let p = clamp_p(p1) as f64 / ONE as f64;
    -(if bit { p } else { 1.0 - p }).log2()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(s: &mut u64) -> u64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *s >> 33
    }

    #[test]
    fn round_trip_models_and_raw() {
        let mut s = 7;
        let bits: Vec<bool> = (0..200_000).map(|i| lcg(&mut s) % 100 < if i % 3 == 0 { 3 } else { 60 }).collect();
        let vals: Vec<u64> = (0..5000).map(|_| lcg(&mut s) % 3000).collect();
        let mut e = Encoder::new();
        let mut ms = [BitModel::default(); 3];
        for (i, &b) in bits.iter().enumerate() {
            e.encode(b, &mut ms[i % 3]);
        }
        let mut um = UintModel::default();
        for &v in &vals {
            um.encode(&mut e, v);
        }
        e.encode_raw(0xDEAD_BEEF, 32);
        e.encode_p(true, 1);
        e.encode_p(false, 70000);
        let buf = e.finish();
        let mut d = Decoder::new(&buf);
        let mut ms = [BitModel::default(); 3];
        for (i, &b) in bits.iter().enumerate() {
            assert_eq!(d.decode(&mut ms[i % 3]), b, "bit {i}");
        }
        let mut um = UintModel::default();
        for &v in &vals {
            assert_eq!(um.decode(&mut d), v);
        }
        assert_eq!(d.decode_raw(32), 0xDEAD_BEEF);
        assert!(d.decode_p(1));
        assert!(!d.decode_p(70000));
        // adaptive coding lands near the empirical entropy
        let mut e = Encoder::new();
        let mut m = BitModel::default();
        let skew: Vec<bool> = (0..100_000).map(|_| lcg(&mut s) % 100 < 5).collect();
        for &b in &skew {
            e.encode(b, &mut m);
        }
        let h = -(0.05f64 * 0.05f64.log2() + 0.95 * 0.95f64.log2()) * 100_000.0;
        let got = e.finish().len() as f64 * 8.0;
        assert!(got < h * 1.05, "{got} vs {h}");
    }

    #[test]
    fn carry_heavy_stream() {
        // long runs of highly probable ones push `low` into carries
        let mut e = Encoder::new();
        for i in 0..100_000 {
            e.encode_p(i % 997 != 0, ONE - P_MIN);
        }
        let buf = e.finish();
        let mut d = Decoder::new(&buf);
        for i in 0..100_000 {
            assert_eq!(d.decode_p(ONE - P_MIN), i % 997 != 0);
        }
    }
}
