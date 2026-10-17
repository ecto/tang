//! Graph building blocks the signature shares with the KEM: absorbing a
//! message whose secret parts are graph values, bit-field extraction, an
//! oblivious read at a secret index, and tree reductions. All branch-free;
//! every address is affine or a mask select.

use crate::keccak::{keccak_f, Tables, SHAKE256_RATE};
use tang_compile::tensor::modq::{BVal, Body, ModGraph, ModId};

/// `vals[sel]` for a public-or-secret selector, by a tree of mask selects
/// (`vals.len()` a power of two, or padded with the last value).
pub fn mux(b: &mut Body, sel: BVal, vals: &[BVal]) -> BVal {
    let mut level: Vec<BVal> = vals.to_vec();
    let mut bit = 0;
    let one = b.lit(1);
    while level.len() > 1 {
        let shift = b.lit(bit);
        let s = b.shr(sel, shift);
        let s = b.and(s, one);
        level = level
            .chunks(2)
            .map(|p| {
                if p.len() == 2 {
                    b.sel(s, p[1], p[0])
                } else {
                    p[0]
                }
            })
            .collect();
        bit += 1;
    }
    level[0]
}

/// The padded message stream: the public `tail` (separator, public fields,
/// padding — zeros where secrets go) XOR each secret part at its word offset.
pub fn stream(g: &mut ModGraph, tail: ModId, parts: &[(ModId, usize)]) -> ModId {
    let len = g.shape(tail)[0];
    let mut args = vec![(tail, ModGraph::id(1))];
    for &(p, off) in parts {
        args.push((p, ModGraph::view(&[&[1]], &[-(off as i64)]).guarded()));
    }
    let k = args.len();
    g.map(&[len], &args, |b| {
        let mut acc = b.arg(0);
        for i in 1..k {
            let v = b.arg(i);
            acc = b.xor(acc, v);
        }
        acc
    })
}

/// Absorb a padded stream of `blocks` SHAKE-256 blocks. Returns the state
/// with the last block XORed in but not yet permuted, as
/// [`squeeze`](crate::keccak::squeeze) expects.
pub fn absorb(g: &mut ModGraph, t: &Tables, stream: ModId, blocks: usize) -> ModId {
    let rate = SHAKE256_RATE;
    let mask: Vec<u32> = (0..50)
        .map(|w| if w < rate { u32::MAX } else { 0 })
        .collect();
    let mask = g.constant_words(&mask);
    let zero = g.constant_words(&[0; 50]);
    let mut state = zero;
    for blk in 0..blocks {
        if blk > 0 {
            state = keccak_f(g, t, state);
        }
        let words = ModGraph::view(&[&[1]], &[(blk * rate) as i64]).guarded();
        state = g.map(
            &[50],
            &[
                (state, ModGraph::id(1)),
                (stream, words),
                (mask, ModGraph::id(1)),
            ],
            |b| {
                let (s, w, m) = (b.arg(0), b.arg(1), b.arg(2));
                let w = b.and(w, m);
                b.xor(s, w)
            },
        );
    }
    state
}

/// `count` little-endian fields of `bits` bits from a word stream, as raw
/// values. Field j of a group starts at bit `bits·j`; a field may straddle
/// two words.
pub fn extract_bits(g: &mut ModGraph, words: ModId, bits: u32, count: usize) -> ModId {
    let lcm = (1..=32)
        .map(|m| bits * m)
        .find(|x| x % 32 == 0)
        .expect("lcm");
    let (per, wpg) = ((lcm / bits) as usize, (lcm / 32) as i64);
    assert!(per.is_power_of_two() && count.is_multiple_of(per));
    let groups = count / per;
    let mask = if bits == 32 {
        u32::MAX
    } else {
        (1 << bits) - 1
    };
    let fields: Vec<ModId> = (0..per)
        .map(|j| {
            let off = bits as usize * j;
            let (wi, sh) = ((off / 32) as i64, (off % 32) as u32);
            g.map(
                &[groups],
                &[
                    (words, ModGraph::view(&[&[wpg]], &[wi]).guarded()),
                    (words, ModGraph::view(&[&[wpg]], &[wi + 1]).guarded()),
                ],
                |b| {
                    let (lo, hi) = (b.arg(0), b.arg(1));
                    let (s, inv, one, m) = (b.lit(sh), b.lit(sh ^ 31), b.lit(1), b.lit(mask));
                    let a = b.shr(lo, s);
                    let h1 = b.shl(hi, one);
                    let h2 = b.shl(h1, inv);
                    let x = b.or(a, h2);
                    b.and(x, m)
                },
            )
        })
        .collect();
    let args: Vec<_> = fields
        .iter()
        .map(|&f| (f, ModGraph::view(&[&[1, 0]], &[0])))
        .collect();
    let out = g.map(&[groups, per], &args, |b| {
        let vals: Vec<BVal> = (0..per).map(|i| b.arg(i)).collect();
        let j = b.coord(1);
        mux(b, j, &vals)
    });
    g.reshape(out, &[count])
}

/// `x[j]` for a `[1]` index `j` that may be secret: a mux tree over all of
/// `x`, so the read touches every element. `len(x)` a power of two.
pub fn mux_read(g: &mut ModGraph, x: ModId, j: ModId) -> ModId {
    let mut cur = x;
    let mut len = g.shape(x)[0];
    let mut bit = 0u32;
    while len > 1 {
        len /= 2;
        cur = g.map(
            &[len],
            &[
                (cur, ModGraph::view(&[&[2]], &[0])),
                (cur, ModGraph::view(&[&[2]], &[1])),
                (j, ModGraph::view(&[&[0]], &[0])),
            ],
            |b| {
                let (even, odd, j) = (b.arg(0), b.arg(1), b.arg(2));
                let (s, one) = (b.lit(bit), b.lit(1));
                let jb = b.shr(j, s);
                let jb = b.and(jb, one);
                b.sel(jb, odd, even)
            },
        );
        bit += 1;
    }
    cur
}

#[derive(Clone, Copy)]
pub enum Fold {
    And,
    Add,
}

/// Reduce a `[2^m]` tensor to `[1]` by a balanced tree.
pub fn fold(g: &mut ModGraph, x: ModId, op: Fold) -> ModId {
    let mut cur = x;
    let mut len = g.shape(x)[0];
    assert!(len.is_power_of_two());
    while len > 1 {
        len /= 2;
        cur = g.map(
            &[len],
            &[
                (cur, ModGraph::view(&[&[2]], &[0])),
                (cur, ModGraph::view(&[&[2]], &[1])),
            ],
            |b| {
                let (a, c) = (b.arg(0), b.arg(1));
                match op {
                    Fold::And => b.and(a, c),
                    Fold::Add => b.add(a, c),
                }
            },
        );
    }
    cur
}

/// Combine `[1]` tensors pairwise in order.
pub fn combine(g: &mut ModGraph, xs: &[ModId], op: Fold) -> ModId {
    let mut acc = xs[0];
    for &x in &xs[1..] {
        acc = crate::zip(g, acc, x, |b, a, c| match op {
            Fold::And => b.and(a, c),
            Fold::Add => b.add(a, c),
        });
    }
    acc
}

/// |x| for a residue read as centred in (−q/2, q/2].
pub fn abs_centred(b: &mut Body, x: BVal, q: u32) -> BVal {
    let (half, zero) = (b.lit((q - 1) / 2), b.lit(0));
    let neg = b.sub(zero, x);
    let big = b.lt(half, x);
    b.sel(big, neg, x)
}

/// `width`-bit fields packed `32 / width` to a word, low field first (FIPS 204
/// w₁Encode at width 4; message bits at width 1).
pub fn pack_fields(g: &mut ModGraph, poly: ModId, width: u32) -> ModId {
    let n = g.shape(poly)[0];
    let per = (32 / width) as usize;
    let args: Vec<_> = (0..per)
        .map(|m| (poly, ModGraph::view(&[&[per as i64]], &[m as i64])))
        .collect();
    g.map(&[n / per], &args, |b| {
        let mut acc = b.arg(0);
        for m in 1..per {
            let v = b.arg(m);
            let s = b.lit(width * m as u32);
            let v = b.shl(v, s);
            acc = b.or(acc, v);
        }
        acc
    })
}

/// Host side: the public tail of a message of `len` bytes — `public` fields
/// at their byte offsets, zeros elsewhere — with SHAKE padding, as words.
pub fn tail(len: usize, public: &[(usize, &[u8])]) -> (Vec<u32>, usize) {
    let rate = SHAKE256_RATE * 4;
    let blocks = len / rate + 1;
    let mut m = vec![0u8; blocks * rate];
    for (off, bytes) in public {
        m[*off..off + bytes.len()].copy_from_slice(bytes);
    }
    m[len] ^= 0x1f;
    m[blocks * rate - 1] ^= 0x80;
    let words = m
        .chunks(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (words, blocks)
}
