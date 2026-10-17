//! The secret samplers as a graph: PRF (SHAKE-256 of seed ‖ nonce), then
//! centred binomial η = 2 or uniform 12-bit noise, all inside the secrecy
//! check. Bit extraction uses shifts by public amounts only.

use crate::keccak::{squeeze, Tables, SHAKE256_RATE};
use tang_compile::tensor::modq::{Label, ModGraph, ModId};

/// k CBD(2) polys from one squeeze each.
fn cbd2(g: &mut ModGraph, words: ModId, n: usize) -> ModId {
    let wpp = n / 8; // 8 nibbles per word
    let out = g.map(
        &[wpp, 8],
        &[(words, ModGraph::view(&[&[1, 0]], &[0]))],
        |b| {
            let w = b.arg(0);
            let (two, c1, c2, c3, c15) = (b.lit(2), b.lit(1), b.lit(2), b.lit(3), b.lit(15));
            let j = b.coord(1);
            let sh = b.shl(j, two);
            let nib = b.shr(w, sh);
            let nib = b.and(nib, c15);
            let b0 = b.and(nib, c1);
            let n1 = b.shr(nib, c1);
            let b1 = b.and(n1, c1);
            let n2 = b.shr(nib, c2);
            let b2 = b.and(n2, c1);
            let n3 = b.shr(nib, c3);
            let b3 = b.and(n3, c1);
            let pos = b.add(b0, b1);
            let neg = b.add(b2, b3);
            b.sub(pos, neg)
        },
    );
    g.reshape(out, &[n])
}

/// One poly uniform on [−2¹¹, 2¹¹): 12 bits per coefficient, 8 per 3 words.
fn uniform12(g: &mut ModGraph, words: ModId, n: usize) -> ModId {
    let (mut f1, mut f2, mut sh) = ([0u32; 8], [0u32; 8], [0u32; 8]);
    for j in 0..8 {
        let (w, s) = (12 * j / 32, 12 * j % 32);
        f1[j] = (w >= 1) as u32;
        f2[j] = (w == 2) as u32;
        sh[j] = s as u32;
    }
    let (f1, f2, sh) = (
        g.constant_words(&f1),
        g.constant_words(&f2),
        g.constant_words(&sh),
    );
    let word = |k: i64| ModGraph::view(&[&[3, 0]], &[k]);
    let by_j = || ModGraph::view(&[&[0, 1]], &[0]);
    let out = g.map(
        &[n / 8, 8],
        &[
            (words, word(0)),
            (words, word(1)),
            (words, word(2)),
            (f1, by_j()),
            (f2, by_j()),
            (sh, by_j()),
        ],
        |b| {
            let (w0, w1, w2, f1, f2, s) =
                (b.arg(0), b.arg(1), b.arg(2), b.arg(3), b.arg(4), b.arg(5));
            let lo = b.sel(f1, w1, w0);
            let lo = b.sel(f2, w2, lo);
            let hi = b.sel(f1, w2, w1);
            let hi = b.sel(f2, w2, hi);
            let (one, c31, mask, half) = (b.lit(1), b.lit(31), b.lit(0xfff), b.lit(2048));
            let a = b.shr(lo, s);
            let inv = b.xor(s, c31);
            let h1 = b.shl(hi, one);
            let h2 = b.shl(h1, inv);
            let x = b.or(a, h2);
            let x = b.and(x, mask);
            b.sub(x, half)
        },
    );
    g.reshape(out, &[n])
}

/// A graph from absorbed PRF states to secret noise polys: `cbd` binomial
/// polys followed by `unif` uniform ones.
pub struct Sampler {
    g: ModGraph,
    outs: Vec<ModId>,
    pub cbd: usize,
    pub unif: usize,
}

impl Sampler {
    pub fn new(q: u32, n: usize, cbd: usize, unif: usize) -> Self {
        assert_eq!(n, 256, "the sampler layouts assume n = 256");
        let mut g = ModGraph::new(q);
        let t = Tables::new(&mut g);
        let states = g.input_words(&[(cbd + unif) * 50], Label::Secret);
        let mut outs = Vec::new();
        for i in 0..cbd + unif {
            let st = g.map(
                &[50],
                &[(states, ModGraph::view(&[&[1]], &[50 * i as i64]))],
                |b| b.arg(0),
            );
            // CBD needs 128 bytes (one block); uniform needs 384 (three)
            let blocks = if i < cbd { 1 } else { 3 };
            let words = squeeze(&mut g, &t, st, SHAKE256_RATE, blocks);
            outs.push(if i < cbd {
                cbd2(&mut g, words, n)
            } else {
                uniform12(&mut g, words, n)
            });
        }
        Self { g, outs, cbd, unif }
    }

    /// The proofs it relies on (none) — or the taint error.
    pub fn check(&self) -> Result<Vec<&'static str>, tang_compile::tensor::modq::TaintError> {
        let outs: Vec<_> = self.outs.iter().map(|&o| (o, Label::Secret)).collect();
        self.g.check(&outs)
    }

    /// Absorbed states, `(cbd + unif) × 50` words → the polys, concatenated
    /// as (all cbd) then (all uniform).
    pub fn run(&self, states: Vec<u32>) -> (Vec<u32>, Vec<u32>) {
        let polys = self.g.eval(&[states], &self.outs);
        let (c, u) = polys.split_at(self.cbd);
        (c.concat(), u.concat())
    }
}
