//! Deterministic static rANS over a hybrid-integer alphabet.
//!
//! Signed integers are zigzag-mapped to `u`, then split into a *token* that is
//! entropy coded and *extra bits* that are stored raw:
//!
//! - `u < 16` → token `u`, no extra bits;
//! - otherwise with `nb = ⌊log2 u⌋`, the token carries `nb` and the two bits
//!   below the leading one (`16 + 4(nb−4) + m`), and the remaining `nb − 2`
//!   low bits go to the raw stream.
//!
//! This is the JPEG-XL "hybrid uint" idea: the coded token captures the shape
//! of the distribution, and the raw bits are (nearly) uniform, so storing them
//! verbatim costs little. Each block's token frequencies are normalised to a
//! `2^14` total and **transmitted** (varint per symbol), and the rANS state is
//! the 32-bit / byte-renormalised variant (Giesen's `rans_byte`). Integer
//! arithmetic only: encode and decode are bit-exact on every platform.

use crate::bitio::{unzigzag, zigzag, BitReader, BitWriter, ByteReader, ByteWriter, DecodeError};

/// Probability resolution of the frequency tables, in bits.
pub const SCALE_BITS: u32 = 14;
const M: u32 = 1 << SCALE_BITS;
const RANS_L: u32 = 1 << 23;

/// Map `u` to `(token, extra_bits_value, extra_bit_count)`.
pub fn split_uint(u: u64) -> (u32, u64, u32) {
    if u < 16 {
        return (u as u32, 0, 0);
    }
    let nb = 63 - u.leading_zeros();
    let m = ((u >> (nb - 2)) & 3) as u32;
    let token = 16 + 4 * (nb - 4) + m;
    let extra = nb - 2;
    (token, u & ((1u64 << extra) - 1), extra)
}

/// Extra-bit count and base value for `token`.
fn token_base(token: u32) -> (u64, u32) {
    if token < 16 {
        return (token as u64, 0);
    }
    let t = token - 16;
    let nb = t / 4 + 4;
    let m = (t % 4) as u64;
    (((4 | m) << (nb - 2)), nb - 2)
}

/// Normalise raw counts to a `2^SCALE_BITS` total, every present symbol ≥ 1.
fn normalize(counts: &[u64]) -> Vec<u32> {
    let total: u64 = counts.iter().sum();
    let mut f: Vec<u32> = counts
        .iter()
        .map(|&c| {
            if c == 0 {
                0
            } else {
                (((c as u128) * (M as u128) / (total as u128)) as u32).max(1)
            }
        })
        .collect();
    let mut sum: i64 = f.iter().map(|&x| x as i64).sum();
    // Fix the residual deterministically: take from / give to the largest bins.
    while sum != M as i64 {
        let mut order: Vec<usize> = (0..f.len()).filter(|&i| f[i] > 0).collect();
        order.sort_by(|&a, &b| f[b].cmp(&f[a]).then(a.cmp(&b)));
        if sum < M as i64 {
            f[order[0]] += (M as i64 - sum) as u32;
            sum = M as i64;
        } else {
            for &i in &order {
                if sum == M as i64 {
                    break;
                }
                let can = (f[i] - 1) as i64;
                let take = can.min(sum - M as i64);
                f[i] -= take as u32;
                sum -= take;
            }
        }
    }
    f
}

/// One static model: frequencies and cumulative starts.
struct Model {
    freq: Vec<u32>,
    start: Vec<u32>,
    /// Slot → symbol lookup for decoding.
    lut: Vec<u16>,
}

impl Model {
    fn new(freq: Vec<u32>) -> Result<Self, DecodeError> {
        let mut start = Vec::with_capacity(freq.len());
        let mut acc = 0u32;
        for &f in &freq {
            start.push(acc);
            acc = acc.checked_add(f).ok_or(DecodeError::Invalid("freq sum"))?;
        }
        if acc != M {
            return Err(DecodeError::Invalid("freq table does not sum to 2^14"));
        }
        let mut lut = vec![0u16; M as usize];
        for (s, (&st, &f)) in start.iter().zip(&freq).enumerate() {
            for slot in lut.iter_mut().skip(st as usize).take(f as usize) {
                *slot = s as u16;
            }
        }
        Ok(Self { freq, start, lut })
    }
}

fn write_table(w: &mut ByteWriter, freq: &[u32]) {
    w.varint(freq.len() as u64);
    for &f in freq {
        w.varint(f as u64);
    }
}

fn read_table(r: &mut ByteReader) -> Result<Vec<u32>, DecodeError> {
    let n = r.varint()? as usize;
    if n == 0 || n > 300 {
        return Err(DecodeError::Invalid("alphabet size"));
    }
    (0..n).map(|_| Ok(r.varint()? as u32)).collect()
}

fn token_counts<'a>(vals: impl Iterator<Item = &'a i64>) -> Vec<u64> {
    let mut counts = vec![0u64; 16];
    for &v in vals {
        let (t, _, _) = split_uint(zigzag(v));
        if t as usize >= counts.len() {
            counts.resize(t as usize + 1, 0);
        }
        counts[t as usize] += 1;
    }
    while counts.len() > 1 && *counts.last().unwrap() == 0 {
        counts.pop();
    }
    counts
}

/// rANS-encode `vals` under `model`; returns (rANS bytes, raw extra bits).
fn encode_with(vals: &[i64], model: &Model) -> (Vec<u8>, Vec<u8>) {
    let mut bits = BitWriter::new();
    let mut toks = Vec::with_capacity(vals.len());
    for &v in vals {
        let (t, e, n) = split_uint(zigzag(v));
        bits.put(e, n);
        toks.push(t);
    }
    let mut out = Vec::new();
    let mut x: u32 = RANS_L;
    for &t in toks.iter().rev() {
        let f = model.freq[t as usize];
        let x_max = ((RANS_L >> SCALE_BITS) << 8) * f;
        while x >= x_max {
            out.push(x as u8);
            x >>= 8;
        }
        x = ((x / f) << SCALE_BITS) + (x % f) + model.start[t as usize];
    }
    out.extend_from_slice(&x.to_le_bytes());
    out.reverse();
    (out, bits.finish())
}

fn decode_with(n: usize, model: &Model, rans: &[u8], raw: &[u8]) -> Result<Vec<i64>, DecodeError> {
    if rans.len() < 4 {
        return Err(DecodeError::Truncated);
    }
    let mut x = u32::from_be_bytes(rans[..4].try_into().unwrap());
    let mut pos = 4;
    let mut bits = BitReader::new(raw);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let slot = x & (M - 1);
        let t = model.lut[slot as usize] as u32;
        x = model.freq[t as usize] * (x >> SCALE_BITS) + slot - model.start[t as usize];
        while x < RANS_L {
            let b = *rans.get(pos).ok_or(DecodeError::Truncated)?;
            pos += 1;
            x = (x << 8) | b as u32;
        }
        let (base, nextra) = token_base(t);
        let u = base + bits.get(nextra)?;
        out.push(unzigzag(u));
    }
    Ok(out)
}

/// Encode one block (table + rANS payload + raw bits) into `w`.
fn encode_single(w: &mut ByteWriter, vals: &[i64]) {
    let freq = normalize(&token_counts(vals.iter()));
    write_table(w, &freq);
    let model = Model::new(freq).expect("normalised table");
    let (rans, raw) = encode_with(vals, &model);
    w.block(&rans);
    w.block(&raw);
}

/// Entropy-code a list of integer blocks whose lengths the decoder already
/// knows. The encoder tries one shared table vs one table per block and
/// keeps the smaller (1 flag byte). Empty blocks cost nothing.
pub fn encode_blocks(w: &mut ByteWriter, blocks: &[&[i64]]) {
    let total: usize = blocks.iter().map(|b| b.len()).sum();
    if total == 0 {
        return;
    }
    // Shared table.
    let mut shared = ByteWriter::new();
    {
        let all: Vec<i64> = blocks.iter().flat_map(|b| b.iter().copied()).collect();
        encode_single(&mut shared, &all);
    }
    // Per-block tables.
    let mut per = ByteWriter::new();
    for b in blocks {
        if !b.is_empty() {
            encode_single(&mut per, b);
        }
    }
    if shared.buf.len() <= per.buf.len() {
        w.u8(0);
        w.bytes(&shared.buf);
    } else {
        w.u8(1);
        w.bytes(&per.buf);
    }
}

fn decode_single(r: &mut ByteReader, n: usize) -> Result<Vec<i64>, DecodeError> {
    let model = Model::new(read_table(r)?)?;
    let rans = r.block()?;
    let raw = r.block()?;
    decode_with(n, &model, rans, raw)
}

/// Inverse of [`encode_blocks`].
pub fn decode_blocks(r: &mut ByteReader, lens: &[usize]) -> Result<Vec<Vec<i64>>, DecodeError> {
    let total: usize = lens.iter().sum();
    if total == 0 {
        return Ok(lens.iter().map(|_| Vec::new()).collect());
    }
    match r.u8()? {
        0 => {
            let all = decode_single(r, total)?;
            let mut out = Vec::with_capacity(lens.len());
            let mut off = 0;
            for &l in lens {
                out.push(all[off..off + l].to_vec());
                off += l;
            }
            Ok(out)
        }
        1 => lens
            .iter()
            .map(|&l| {
                if l == 0 {
                    Ok(Vec::new())
                } else {
                    decode_single(r, l)
                }
            })
            .collect(),
        _ => Err(DecodeError::Invalid("table mode")),
    }
}

/// Zeroth-order empirical entropy of the token stream plus raw bits, in
/// bits — the floor a static per-block coder can approach (ignores tables).
pub fn empirical_bits(vals: &[i64]) -> f64 {
    let counts = token_counts(vals.iter());
    let n = vals.len() as f64;
    let mut h = 0.0;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 / n;
            h -= c as f64 * p.log2();
        }
    }
    let raw: u64 = vals.iter().map(|&v| split_uint(zigzag(v)).2 as u64).sum();
    h + raw as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *seed >> 33
    }

    #[test]
    fn token_split_is_invertible() {
        for u in (0..5000u64).chain([1 << 20, (1 << 40) + 12345, u32::MAX as u64]) {
            let (t, e, n) = split_uint(u);
            let (base, nb) = token_base(t);
            assert_eq!(nb, n);
            assert_eq!(base + e, u, "u={u}");
        }
    }

    #[test]
    fn rans_roundtrip_and_near_entropy() {
        let mut seed = 7u64;
        // Laplacian-ish integers.
        let a: Vec<i64> = (0..20000)
            .map(|_| {
                let r = lcg(&mut seed) % 1000;
                let mag = (-(1.0 - r as f64 / 1000.0).ln() * 3.0) as i64;
                if lcg(&mut seed) & 1 == 0 {
                    mag
                } else {
                    -mag
                }
            })
            .collect();
        let b: Vec<i64> = vec![5; 300];
        let c: Vec<i64> = (0..500).map(|i| (i * i) as i64 - 100_000).collect();
        let mut w = ByteWriter::new();
        encode_blocks(&mut w, &[&a, &b, &[], &c]);
        let mut r = ByteReader::new(&w.buf);
        let out = decode_blocks(&mut r, &[a.len(), b.len(), 0, c.len()]).unwrap();
        assert_eq!(out[0], a);
        assert_eq!(out[1], b);
        assert!(out[2].is_empty());
        assert_eq!(out[3], c);
        assert_eq!(r.position(), w.buf.len());
        // Coding overhead vs zeroth-order entropy stays small for a big block.
        let mut wa = ByteWriter::new();
        encode_blocks(&mut wa, &[&a]);
        let ideal = empirical_bits(&a) / 8.0;
        let actual = wa.buf.len() as f64;
        assert!(
            actual < ideal * 1.02 + 64.0,
            "actual {actual} ideal {ideal}"
        );
    }

    #[test]
    fn rans_is_deterministic() {
        let v: Vec<i64> = (0..1000).map(|i| (i % 17) - 8).collect();
        let mut w1 = ByteWriter::new();
        let mut w2 = ByteWriter::new();
        encode_blocks(&mut w1, &[&v]);
        encode_blocks(&mut w2, &[&v]);
        assert_eq!(w1.buf, w2.buf);
    }
}
