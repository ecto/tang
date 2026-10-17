//! Keccak-f[1600] and SHAKE squeezing as a tang [`ModGraph`], so hashing a
//! secret seed runs inside the secrecy check.
//!
//! The state is 50 raw u32 words: lane `L = x + 5y` is words `2L` (low) and
//! `2L + 1` (high), little-endian as in FIPS 202. Every step is a fixed map;
//! the ρ∘π lane shuffle is a [`ModGraph::gather`] through a public index table.

use tang_compile::tensor::modq::{BVal, Body, ModGraph, ModId};

const RC: [u64; 24] = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808A,
    0x8000000080008000,
    0x000000000000808B,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008A,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000A,
    0x000000008000808B,
    0x800000000000008B,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800A,
    0x800000008000000A,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
];

/// ρ offsets, `ROT[x][y]`.
const ROT: [[u32; 5]; 5] = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
];

/// One half of a 64-bit rotate-left of `(hi:lo)` by a public amount `r`:
/// the low half where `h == 0`, the high half where `h == 1`.
fn rot64(b: &mut Body, lo: BVal, hi: BVal, r: BVal, h: BVal) -> BVal {
    let (c5, c1, c31) = (b.lit(5), b.lit(1), b.lit(31));
    let big = b.shr(r, c5);
    let big = b.and(big, c1);
    let a = b.sel(big, hi, lo);
    let c = b.sel(big, lo, hi);
    let s = b.and(r, c31);
    let inv = b.xor(s, c31);
    let al = b.shl(a, s);
    let c1s = b.shr(c, c1);
    let cr = b.shr(c1s, inv);
    let new_lo = b.or(al, cr);
    let cl = b.shl(c, s);
    let a1s = b.shr(a, c1);
    let ar = b.shr(a1s, inv);
    let new_hi = b.or(cl, ar);
    b.sel(h, new_hi, new_lo)
}

/// Public tables every round shares.
pub struct Tables {
    pi_index: ModId,
    rot: ModId,
    iota: ModId,
}

impl Tables {
    pub fn new(g: &mut ModGraph) -> Self {
        let mut index = vec![0u32; 50];
        let mut rot = vec![0u32; 25];
        for (x, offsets) in ROT.iter().enumerate() {
            for (y, &r) in offsets.iter().enumerate() {
                let dest = y + 5 * ((2 * x + 3 * y) % 5);
                for h in 0..2 {
                    index[dest * 2 + h] = ((x + 5 * y) * 2 + h) as u32;
                }
                rot[dest] = r;
            }
        }
        let mut iota = vec![0u32; 24 * 50];
        for (r, rc) in RC.iter().enumerate() {
            iota[r * 50] = *rc as u32;
            iota[r * 50 + 1] = (rc >> 32) as u32;
        }
        Self {
            pi_index: g.constant_words(&index),
            rot: g.constant_words(&rot),
            iota: g.constant_words(&iota),
        }
    }
}

/// Keccak-f[1600] on a `[50]` word state.
pub fn keccak_f(g: &mut ModGraph, t: &Tables, state: ModId) -> ModId {
    let mut a = state;
    for round in 0..24 {
        // θ: column parities C[x], then D[x] = C[x−1] ⊕ rot(C[x+1], 1)
        let cols: Vec<_> = (0..5)
            .map(|y| (a, ModGraph::view(&[&[2, 1]], &[10 * y])))
            .collect();
        let c = g.map(&[5, 2], &cols, |b| {
            let mut acc = b.arg(0);
            for i in 1..5 {
                let v = b.arg(i);
                acc = b.xor(acc, v);
            }
            acc
        });
        // C twice over, so x ± 1 mod 5 is an affine read
        let cc = g.map(
            &[2, 5, 2],
            &[(c, ModGraph::view(&[&[0, 1, 0], &[0, 0, 1]], &[0, 0]))],
            |b| b.arg(0),
        );
        let cc = g.reshape(cc, &[10, 2]);
        let d = g.map(
            &[5, 2],
            &[
                (cc, ModGraph::view(&[&[1, 0], &[0, 1]], &[4, 0])),
                (cc, ModGraph::view(&[&[1, 0], &[0, 0]], &[1, 0])),
                (cc, ModGraph::view(&[&[1, 0], &[0, 0]], &[1, 1])),
            ],
            |b| {
                let (left, lo, hi) = (b.arg(0), b.arg(1), b.arg(2));
                let (one, h) = (b.lit(1), b.coord(1));
                let r = rot64(b, lo, hi, one, h);
                b.xor(left, r)
            },
        );
        let th = g.map(
            &[5, 5, 2],
            &[
                (a, ModGraph::view(&[&[10, 2, 1]], &[0])),
                (d, ModGraph::view(&[&[0, 1, 0], &[0, 0, 1]], &[0, 0])),
            ],
            |b| {
                let (x, y) = (b.arg(0), b.arg(1));
                b.xor(x, y)
            },
        );
        let th = g.reshape(th, &[50]);

        // ρ∘π: move every lane to its destination, then rotate it
        let moved = g.gather(th, t.pi_index);
        let rotated = g.map(
            &[25, 2],
            &[
                (moved, ModGraph::view(&[&[2, 0]], &[0])),
                (moved, ModGraph::view(&[&[2, 0]], &[1])),
                (t.rot, ModGraph::view(&[&[1, 0]], &[0])),
            ],
            |b| {
                let (lo, hi, r) = (b.arg(0), b.arg(1), b.arg(2));
                let h = b.coord(1);
                rot64(b, lo, hi, r, h)
            },
        );
        let bst = g.reshape(rotated, &[5, 5, 2]);

        // χ: B[x] ⊕ (¬B[x+1] ∧ B[x+2]) along each row
        let bb = g.map(
            &[5, 2, 5, 2],
            &[(
                bst,
                ModGraph::view(&[&[1, 0, 0, 0], &[0, 0, 1, 0], &[0, 0, 0, 1]], &[0, 0, 0]),
            )],
            |b| b.arg(0),
        );
        let bb = g.reshape(bb, &[5, 10, 2]);
        let row = |off: i64| ModGraph::view(&[&[1, 0, 0], &[0, 1, 0], &[0, 0, 1]], &[0, off, 0]);
        let chi = g.map(
            &[5, 5, 2],
            &[(bst, ModGraph::id(3)), (bb, row(1)), (bb, row(2))],
            |b| {
                let (x, x1, x2) = (b.arg(0), b.arg(1), b.arg(2));
                let n1 = b.not(x1);
                let m = b.and(n1, x2);
                b.xor(x, m)
            },
        );
        let chi = g.reshape(chi, &[50]);

        // ι: the round constant into lane 0
        a = g.map(
            &[50],
            &[
                (chi, ModGraph::id(1)),
                (t.iota, ModGraph::view(&[&[1]], &[round as i64 * 50])),
            ],
            |b| {
                let (x, c) = (b.arg(0), b.arg(1));
                b.xor(x, c)
            },
        );
    }
    a
}

/// SHAKE rate in 32-bit words: 136 bytes for SHAKE-256.
pub const SHAKE256_RATE: usize = 34;

/// Squeeze `blocks × rate` words from an absorbed `[50]` state (the host has
/// already XORed the padded message into a zero state). Returns one tensor
/// `[blocks · rate]`.
pub fn squeeze(g: &mut ModGraph, t: &Tables, absorbed: ModId, rate: usize, blocks: usize) -> ModId {
    assert!(
        (1..=8).contains(&blocks),
        "squeeze joins at most eight blocks"
    );
    let mut states = vec![keccak_f(g, t, absorbed)];
    while states.len() < blocks {
        let next = keccak_f(g, t, *states.last().expect("one state"));
        states.push(next);
    }
    let args: Vec<_> = states
        .iter()
        .map(|&s| (s, ModGraph::view(&[&[0, 1]], &[0])))
        .collect();
    let joined = g.map(&[blocks, rate], &args, |b| {
        let vals: Vec<BVal> = (0..blocks).map(|i| b.arg(i)).collect();
        let blk = b.coord(0);
        crate::graph::mux(b, blk, &vals)
    });
    g.reshape(joined, &[blocks * rate])
}

/// Bytes → the `[50]` absorbed state for a message shorter than one block,
/// with SHAKE padding (domain bits 0x1F, final 0x80).
pub fn absorb_one_block(msg: &[u8], rate_bytes: usize) -> Vec<u32> {
    assert!(msg.len() < rate_bytes, "one block only");
    let mut block = [0u8; 200];
    block[..msg.len()].copy_from_slice(msg);
    block[msg.len()] ^= 0x1f;
    block[rate_bytes - 1] ^= 0x80;
    block
        .chunks(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

pub fn words_to_bytes(w: &[u32]) -> Vec<u8> {
    w.iter().flat_map(|x| x.to_le_bytes()).collect()
}
