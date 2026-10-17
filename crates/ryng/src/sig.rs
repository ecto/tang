//! ryng's signature: Fiat–Shamir with aborts over the KEM's ring, shaped
//! like ML-DSA-65 (k = 6, ℓ = 5, τ = 49, γ₁ = 2¹⁹, γ₂ = (q − 1)/32, ω = 55,
//! d = 13). **Unaudited research code — do not use to protect anything.**
//!
//! Where it departs from FIPS 204, it does so to keep every secret out of
//! control flow and addresses, and the graph checker holds it to that:
//!
//! - s₁, s₂ are uniform on [−4, 3] by masking 3 bits, not rejection-sampled
//!   on [−4, 4]. Best known key-recovery attack (lattice-estimator):
//!   2^200.4, against 2^203.8 for FIPS 204's distribution.
//! - SampleInBall draws each index by a 32×32 multiply-high (statistical
//!   distance ≤ 2⁻²⁴ per draw from uniform) instead of rejection, and swaps
//!   at the secret index through a mux tree.
//! - Each signing attempt is one graph. Its outputs are masked to zero
//!   unless the attempt was accepted and only then declassified, so a
//!   rejected attempt publishes nothing but its own rejection — the attempt
//!   count, which rejection sampling makes independent of the key.
//! - Separators are ryng's, so signatures are not FIPS 204 signatures.

use crate::graph::{
    abs_centred, absorb, combine, extract_bits, fold, mux_read, pack_fields, stream, tail, Fold,
};
use crate::keccak::{squeeze, Tables, SHAKE256_RATE};
use crate::{each, pack, row, sep, unpack, zip};
use sha3::digest::{ExtendableOutput, Update, XofReader};
use sha3::Shake256;
use tang_compile::tensor::modq::{intt, ntt, Label, ModGraph, ModId, TaintError};

pub const Q: u32 = 8380417;
const ZETA: u32 = 1753;
const N: usize = 256;
pub const K: usize = 6;
pub const L: usize = 5;
const TAU: usize = 49;
const BETA: u32 = 196;
const GAMMA1: u32 = 1 << 19;
const GAMMA2: u32 = (Q - 1) / 32;
const OMEGA: usize = 55;
const D: u32 = 13;
const CTILDE_WORDS: usize = 12;

pub const PK_BYTES: usize = 32 + K * N * 10 / 8;
pub const SIG_BYTES: usize = CTILDE_WORDS * 4 + L * N * 20 / 8 + OMEGA + K;

pub const RHO_PROOF: &str = "xof: ρ is independent of ρ′ and K";
pub const PK_PROOF: &str = "mlwe: t = A·s₁ + s₂ is pseudorandom";
pub const ATTEMPT_PROOF: &str = "fswa: the attempt count is independent of the key";
pub const SIG_PROOF: &str = "fswa: an accepted (c̃, z, h) is simulatable without the key";

/// A public matrix from ρ, sampled in the NTT domain by rejection from
/// SHAKE-128 (public bytes only).
fn sample_mat(rho: &[u8], rows: usize, cols: usize) -> Vec<u32> {
    use sha3::Shake128;
    let mut out = Vec::with_capacity(rows * cols * N);
    for i in 0..rows {
        for j in 0..cols {
            let mut x = Shake128::default();
            x.update(&sep(b"sig/A"));
            x.update(rho);
            x.update(&[i as u8, j as u8]);
            let mut rd = x.finalize_xof();
            let mut got = 0;
            while got < N {
                let mut b = [0u8; 3];
                rd.read(&mut b);
                let c = u32::from_le_bytes([b[0], b[1], b[2], 0]) & 0x7f_ffff;
                if c < Q {
                    out.push(c);
                    got += 1;
                }
            }
        }
    }
    out
}

fn shake(name: &[u8], parts: &[&[u8]], len: usize) -> Vec<u8> {
    let mut x = Shake256::default();
    x.update(&sep(name));
    for p in parts {
        x.update(p);
    }
    let mut out = vec![0; len];
    x.finalize_xof().read(&mut out);
    out
}

fn words(b: &[u8]) -> Vec<u32> {
    b.chunks(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn bytes(w: &[u32]) -> Vec<u8> {
    w.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Σ_j A[i·cols + j] ∘ x̂_j for each row i, back out of the NTT domain.
fn matvec(g: &mut ModGraph, a: ModId, rows: usize, xh: &[ModId]) -> Vec<ModId> {
    (0..rows)
        .map(|i| {
            let mut acc: Option<ModId> = None;
            for (j, &x) in xh.iter().enumerate() {
                let aij = row(g, a, i * xh.len() + j, N);
                let p = zip(g, aij, x, |b, u, v| b.mul(u, v));
                acc = Some(match acc {
                    None => p,
                    Some(s) => zip(g, s, p, |b, u, v| b.add(u, v)),
                });
            }
            intt(g, acc.expect("cols ≥ 1"), ZETA)
        })
        .collect()
}

fn polys(g: &mut ModGraph, x: ModId, count: usize) -> Vec<ModId> {
    (0..count).map(|i| row(g, x, i, N)).collect()
}

fn ntts(g: &mut ModGraph, xs: &[ModId]) -> Vec<ModId> {
    xs.iter().map(|&x| ntt(g, x, ZETA)).collect()
}

/// SampleInBall inside the graph: c with τ coefficients ±1, from c̃.
pub fn sample_in_ball(g: &mut ModGraph, t: &Tables, ctilde: ModId) -> ModId {
    let (tw, blocks) = tail(32 + CTILDE_WORDS * 4, &[(0, &sep(b"sig/ball"))]);
    let tw = g.constant_words(&tw);
    let st = stream(g, tw, &[(ctilde, 8)]);
    let st = absorb(g, t, st, blocks);
    let w = squeeze(g, t, st, SHAKE256_RATE, 2);
    let mut c = g.constant_words(&[0; N]);
    for step in 0..TAU {
        let i = (N - TAU + step) as u32;
        let j = g.map(
            &[1],
            &[(w, ModGraph::view(&[&[0]], &[2 + step as i64]))],
            |b| {
                let (x, m) = (b.arg(0), b.lit(i + 1));
                b.mulhi(x, m)
            },
        );
        let sign_word = (step / 32) as i64;
        let val = g.map(&[1], &[(w, ModGraph::view(&[&[0]], &[sign_word]))], |b| {
            let x = b.arg(0);
            let (s, one, neg) = (b.lit((step % 32) as u32), b.lit(1), b.lit(Q - 1));
            let bit = b.shr(x, s);
            let bit = b.and(bit, one);
            b.sel(bit, neg, one)
        });
        let old = mux_read(g, c, j);
        let bc = || ModGraph::view(&[&[0]], &[0]);
        c = g.map(
            &[N],
            &[(c, ModGraph::id(1)), (j, bc()), (old, bc()), (val, bc())],
            |b| {
                let (cur, j, old, val) = (b.arg(0), b.arg(1), b.arg(2), b.arg(3));
                let (k, li) = (b.coord(0), b.lit(i));
                let is_i = b.eq(k, li);
                let is_j = b.eq(k, j);
                let r = b.sel(is_i, old, cur); // c_i ← c_j
                b.sel(is_j, val, r) // c_j ← ±1
            },
        );
    }
    c
}

/// Host twin of [`sample_in_ball`] for verification (public inputs).
pub fn sample_in_ball_host(ctilde: &[u8]) -> Vec<u32> {
    let w = words(&shake(b"sig/ball", &[ctilde], 2 * SHAKE256_RATE * 4));
    let mut c = vec![0u32; N];
    for step in 0..TAU {
        let i = N - TAU + step;
        let j = ((w[2 + step] as u64 * (i as u64 + 1)) >> 32) as usize;
        let bit = (w[step / 32] >> (step % 32)) & 1;
        c[i] = c[j];
        c[j] = if bit == 1 { Q - 1 } else { 1 };
    }
    c
}

struct KeySeeds {
    g: ModGraph,
    rho: ModId,
    key: ModId,
    s: Vec<ModId>,
}

struct KeyT {
    g: ModGraph,
    t1: Vec<ModId>,
    t0: Vec<ModId>,
}

struct Attempt {
    g: ModGraph,
    accept: ModId,
    ctilde: ModId,
    z: Vec<ModId>,
    h: Vec<ModId>,
    /// Ungated, for diagnostics only: (z ok, r₀ ok, c·t₀ ok, hints ok, hint count).
    checks: [ModId; 5],
}

struct Verify {
    g: ModGraph,
    w: Vec<ModId>,
}

/// The signature's graphs, built once.
pub struct Sig {
    seeds: KeySeeds,
    keyt: KeyT,
    attempt: Attempt,
    verify: Verify,
}

/// A signing key. Fields are residues mod q (or raw words for `key`).
pub struct SigningKey {
    rho: Vec<u8>,
    key: Vec<u32>,
    tr: Vec<u8>,
    s1: Vec<u32>,
    s2: Vec<u32>,
    t0: Vec<u32>,
}

impl Default for Sig {
    fn default() -> Self {
        Self::new()
    }
}

impl Sig {
    pub fn new() -> Self {
        // ξ → ρ, ρ′, K, then s₁, s₂ from ρ′
        let mut g = ModGraph::new(Q);
        let t = Tables::new(&mut g);
        let xi_tail = g.input_words(&[SHAKE256_RATE], Label::Public);
        let xi = g.input_words(&[8], Label::Secret);
        let s_tails = g.input_words(&[(K + L) * SHAKE256_RATE], Label::Public);
        let st = stream(&mut g, xi_tail, &[(xi, 8)]);
        let st = absorb(&mut g, &t, st, 1);
        let out = squeeze(&mut g, &t, st, SHAKE256_RATE, 1);
        let rho = g.map(&[8], &[(out, ModGraph::view(&[&[1]], &[0]))], |b| b.arg(0));
        let rho = g.declassify(rho, RHO_PROOF);
        let rho2 = g.map(&[16], &[(out, ModGraph::view(&[&[1]], &[8]))], |b| b.arg(0));
        let key = g.map(&[8], &[(out, ModGraph::view(&[&[1]], &[24]))], |b| b.arg(0));
        let s = (0..K + L)
            .map(|i| {
                let ti = g.map(
                    &[SHAKE256_RATE],
                    &[(
                        s_tails,
                        ModGraph::view(&[&[1]], &[(i * SHAKE256_RATE) as i64]),
                    )],
                    |b| b.arg(0),
                );
                let st = stream(&mut g, ti, &[(rho2, 8)]);
                let st = absorb(&mut g, &t, st, 1);
                let w = squeeze(&mut g, &t, st, SHAKE256_RATE, 1);
                let x = extract_bits(&mut g, w, 3, N);
                each(&mut g, x, |b, v| {
                    let four = b.lit(4);
                    b.sub(v, four)
                })
            })
            .collect();
        let seeds = KeySeeds { g, rho, key, s };

        // t = A·s₁ + s₂ → (t₁ published, t₀ kept)
        let mut g = ModGraph::new(Q);
        let a = g.input(&[K * L * N], Label::Public);
        let s1 = g.input(&[L * N], Label::Secret);
        let s2 = g.input(&[K * N], Label::Secret);
        let s1p = polys(&mut g, s1, L);
        let s1h = ntts(&mut g, &s1p);
        let as1 = matvec(&mut g, a, K, &s1h);
        let (mut t1, mut t0) = (Vec::new(), Vec::new());
        for (i, &x) in as1.iter().enumerate() {
            let s2i = row(&mut g, s2, i, N);
            let ti = zip(&mut g, x, s2i, |b, u, v| b.add(u, v));
            let hi = each(&mut g, ti, |b, v| b.power2round(v, D));
            let lo = zip(&mut g, ti, hi, |b, u, v| {
                let s = b.lit(1 << D);
                let up = b.mul(v, s);
                b.sub(u, up)
            });
            t1.push(g.declassify(hi, PK_PROOF));
            t0.push(lo);
        }
        let keyt = KeyT { g, t1, t0 };

        let attempt = Self::attempt_graph();

        // verify: w′ = A·z − c·t₁·2^d, all public
        let mut g = ModGraph::new(Q);
        let a = g.input(&[K * L * N], Label::Public);
        let t1 = g.input(&[K * N], Label::Public);
        let z = g.input(&[L * N], Label::Public);
        let c = g.input(&[N], Label::Public);
        let zp = polys(&mut g, z, L);
        let zh = ntts(&mut g, &zp);
        let az = matvec(&mut g, a, K, &zh);
        let ch = ntt(&mut g, c, ZETA);
        let w = (0..K)
            .map(|i| {
                let ti = row(&mut g, t1, i, N);
                let ti = each(&mut g, ti, |b, v| {
                    let s = b.lit(1 << D);
                    b.mul(v, s)
                });
                let th = ntt(&mut g, ti, ZETA);
                let ct = zip(&mut g, ch, th, |b, u, v| b.mul(u, v));
                let ct = intt(&mut g, ct, ZETA);
                zip(&mut g, az[i], ct, |b, u, v| b.sub(u, v))
            })
            .collect();
        let verify = Verify { g, w };

        Self {
            seeds,
            keyt,
            attempt,
            verify,
        }
    }

    fn attempt_graph() -> Attempt {
        let mut g = ModGraph::new(Q);
        let t = Tables::new(&mut g);
        let a = g.input(&[K * L * N], Label::Public);
        let s1 = g.input(&[L * N], Label::Secret);
        let s2 = g.input(&[K * N], Label::Secret);
        let t0 = g.input(&[K * N], Label::Secret);
        let key = g.input_words(&[8], Label::Secret);
        let rho2_tail = g.input_words(&[2 * SHAKE256_RATE], Label::Public);
        let y_tails = g.input_words(&[L * SHAKE256_RATE], Label::Public);
        let c_tail = g.input_words(&[7 * SHAKE256_RATE], Label::Public);

        // ρ″ = H(K ‖ rnd ‖ μ), rnd = 0 (deterministic signing)
        let st = stream(&mut g, rho2_tail, &[(key, 8)]);
        let st = absorb(&mut g, &t, st, 2);
        let out = squeeze(&mut g, &t, st, SHAKE256_RATE, 1);
        let rho2 = g.map(&[16], &[(out, ModGraph::id(1))], |b| b.arg(0));

        // y from ρ″ and this attempt's nonces, uniform on (−γ₁, γ₁]
        let y: Vec<ModId> = (0..L)
            .map(|i| {
                let ti = g.map(
                    &[SHAKE256_RATE],
                    &[(
                        y_tails,
                        ModGraph::view(&[&[1]], &[(i * SHAKE256_RATE) as i64]),
                    )],
                    |b| b.arg(0),
                );
                let st = stream(&mut g, ti, &[(rho2, 8)]);
                let st = absorb(&mut g, &t, st, 1);
                let w = squeeze(&mut g, &t, st, SHAKE256_RATE, 5);
                let x = extract_bits(&mut g, w, 20, N);
                each(&mut g, x, |b, v| {
                    let g1 = b.lit(GAMMA1);
                    b.sub(g1, v)
                })
            })
            .collect();
        let yh = ntts(&mut g, &y);
        let w = matvec(&mut g, a, K, &yh);

        // c̃ = H(μ ‖ w₁Encode(HighBits(w))), c = SampleInBall(c̃)
        let parts: Vec<(ModId, usize)> = w
            .iter()
            .enumerate()
            .map(|(i, &wi)| {
                let w1 = each(&mut g, wi, |b, v| b.high_bits(v, GAMMA2));
                (pack_fields(&mut g, w1, 4), 24 + 32 * i)
            })
            .collect();
        let st = stream(&mut g, c_tail, &parts);
        let st = absorb(&mut g, &t, st, 7);
        let out = squeeze(&mut g, &t, st, SHAKE256_RATE, 1);
        let ctilde = g.map(&[CTILDE_WORDS], &[(out, ModGraph::id(1))], |b| b.arg(0));
        let c = sample_in_ball(&mut g, &t, ctilde);
        let ch = ntt(&mut g, c, ZETA);
        let times_c = |g: &mut ModGraph, x: ModId| {
            let xh = ntt(g, x, ZETA);
            let p = zip(g, ch, xh, |b, u, v| b.mul(u, v));
            intt(g, p, ZETA)
        };

        // z = y + c·s₁; r₀ = LowBits(w − c·s₂); h = MakeHint(−c·t₀, w − c·s₂ + c·t₀)
        let s1p = polys(&mut g, s1, L);
        let z: Vec<ModId> = (0..L)
            .map(|j| {
                let cs = times_c(&mut g, s1p[j]);
                zip(&mut g, y[j], cs, |b, u, v| b.add(u, v))
            })
            .collect();
        let s2p = polys(&mut g, s2, K);
        let t0p = polys(&mut g, t0, K);
        let mut oks = Vec::new();
        let mut counts = Vec::new();
        let mut hs = Vec::new();
        for &zj in &z {
            let ok = each(&mut g, zj, |b, v| {
                let a = abs_centred(b, v, Q);
                let bound = b.lit(GAMMA1 - BETA);
                b.lt(a, bound)
            });
            oks.push(fold(&mut g, ok, Fold::And));
        }
        for i in 0..K {
            let cs2 = times_c(&mut g, s2p[i]);
            let wm = zip(&mut g, w[i], cs2, |b, u, v| b.sub(u, v));
            let ok_r0 = each(&mut g, wm, |b, v| {
                let r0 = b.low_bits(v, GAMMA2);
                let a = abs_centred(b, r0, Q);
                let bound = b.lit(GAMMA2 - BETA);
                b.lt(a, bound)
            });
            oks.push(fold(&mut g, ok_r0, Fold::And));
            let ct0 = times_c(&mut g, t0p[i]);
            let ok_ct0 = each(&mut g, ct0, |b, v| {
                let a = abs_centred(b, v, Q);
                let bound = b.lit(GAMMA2);
                b.lt(a, bound)
            });
            oks.push(fold(&mut g, ok_ct0, Fold::And));
            let h = zip(&mut g, wm, ct0, |b, u, v| {
                let s = b.add(u, v);
                let hs = b.high_bits(s, GAMMA2);
                let hu = b.high_bits(u, GAMMA2);
                let same = b.eq(hs, hu);
                let one = b.lit(1);
                b.xor(same, one)
            });
            counts.push(fold(&mut g, h, Fold::Add));
            hs.push(h);
        }
        let count = combine(&mut g, &counts, Fold::Add);
        let ok_h = each(&mut g, count, |b, v| {
            let bound = b.lit(OMEGA as u32 + 1);
            b.lt(v, bound)
        });
        oks.push(ok_h);
        let ok_z = combine(&mut g, &oks[..L], Fold::And);
        let r0s: Vec<ModId> = (0..K).map(|i| oks[L + 2 * i]).collect();
        let ct0s: Vec<ModId> = (0..K).map(|i| oks[L + 2 * i + 1]).collect();
        let ok_r0 = combine(&mut g, &r0s, Fold::And);
        let ok_ct0 = combine(&mut g, &ct0s, Fold::And);
        let checks = [ok_z, ok_r0, ok_ct0, ok_h, count];
        let accept = combine(&mut g, &oks, Fold::And);

        // publish only what an accepted attempt would publish
        let gate = |g: &mut ModGraph, x: ModId| {
            let n = g.shape(x)[0];
            let gated = g.map(
                &[n],
                &[
                    (x, ModGraph::id(1)),
                    (accept, ModGraph::view(&[&[0]], &[0])),
                ],
                |b| {
                    let (v, ok, zero) = (b.arg(0), b.arg(1), b.lit(0));
                    b.sel(ok, v, zero)
                },
            );
            g.declassify(gated, SIG_PROOF)
        };
        let z = z.iter().map(|&x| gate(&mut g, x)).collect();
        let h = hs.iter().map(|&x| gate(&mut g, x)).collect();
        let ctilde = gate(&mut g, ctilde);
        let accept = g.declassify(accept, ATTEMPT_PROOF);
        Attempt {
            g,
            accept,
            ctilde,
            z,
            h,
            checks,
        }
    }

    /// Every signature graph with its outputs, for compiling and linting.
    #[doc(hidden)]
    pub fn graphs(&self) -> Vec<(&ModGraph, Vec<ModId>)> {
        let s = &self.seeds;
        let mut so = vec![s.rho, s.key];
        so.extend(&s.s);
        let mut ko = self.keyt.t1.clone();
        ko.extend(&self.keyt.t0);
        let a = &self.attempt;
        let mut ao = vec![a.accept, a.ctilde];
        ao.extend(a.z.iter().chain(&a.h));
        vec![
            (&s.g, so),
            (&self.keyt.g, ko),
            (&a.g, ao),
            (&self.verify.g, self.verify.w.clone()),
        ]
    }

    /// The attempt graph's inputs for one attempt, for differential testing.
    #[doc(hidden)]
    pub fn attempt_example(&self, sk: &SigningKey, msg: &[u8]) -> [Vec<u32>; 8] {
        let mu = shake(b"sig/mu", &[&sk.tr, msg], 64);
        let a = sample_mat(&sk.rho, K, L);
        let (rho2_tail, _) = tail(160, &[(0, &sep(b"sig/rho2")), (96, &mu)]);
        let (c_tail, _) = tail(32 + 64 + K * N / 2, &[(0, &sep(b"sig/c")), (32, &mu)]);
        attempt_inputs(sk, &a, &rho2_tail, &c_tail, 0)
    }

    /// The secrecy check on every signature graph, and the proofs each relies on.
    pub fn check(&self) -> Result<Vec<Vec<&'static str>>, TaintError> {
        let s = &self.seeds;
        let mut outs = vec![(s.rho, Label::Public), (s.key, Label::Secret)];
        outs.extend(s.s.iter().map(|&x| (x, Label::Secret)));
        let a = &self.attempt;
        let mut aouts = vec![(a.accept, Label::Public), (a.ctilde, Label::Public)];
        aouts.extend(a.z.iter().chain(&a.h).map(|&x| (x, Label::Public)));
        let mut kouts: Vec<_> = self.keyt.t1.iter().map(|&x| (x, Label::Public)).collect();
        kouts.extend(self.keyt.t0.iter().map(|&x| (x, Label::Secret)));
        let vouts: Vec<_> = self.verify.w.iter().map(|&x| (x, Label::Public)).collect();
        Ok(vec![
            s.g.check(&outs)?,
            self.keyt.g.check(&kouts)?,
            a.g.check(&aouts)?,
            self.verify.g.check(&vouts)?,
        ])
    }

    /// Deterministic keygen from a 32-byte seed ξ. Returns (pk, sk).
    pub fn keygen(&self, xi: &[u8; 32]) -> (Vec<u8>, SigningKey) {
        let s = &self.seeds;
        let (xi_tail, _) = tail(64, &[(0, &sep(b"sig/xi"))]);
        let mut s_tails = Vec::new();
        for i in 0..(K + L) as u16 {
            let (ti, _) = tail(32 + 64 + 2, &[(0, &sep(b"sig/s")), (96, &i.to_le_bytes())]);
            s_tails.extend(ti);
        }
        let mut outs = vec![s.rho, s.key];
        outs.extend(&s.s);
        let res = s.g.eval(&[xi_tail, words(xi), s_tails], &outs);
        let rho = bytes(&res[0]);
        let key = res[1].clone();
        let s1 = res[2..2 + L].concat();
        let s2 = res[2 + L..].concat();

        let a = sample_mat(&rho, K, L);
        let mut outs = self.keyt.t1.clone();
        outs.extend(&self.keyt.t0);
        let res = self.keyt.g.eval(&[a, s1.clone(), s2.clone()], &outs);
        let t1 = res[..K].concat();
        let t0 = res[K..].concat();
        let mut pk = rho.clone();
        pk.extend(pack(&t1, 10));
        let tr = shake(b"sig/tr", &[&pk], 64);
        (
            pk,
            SigningKey {
                rho,
                key,
                tr,
                s1,
                s2,
                t0,
            },
        )
    }

    /// Sign. Returns the signature and how many attempts it took.
    pub fn sign(&self, sk: &SigningKey, msg: &[u8]) -> (Vec<u8>, usize) {
        let mu = shake(b"sig/mu", &[&sk.tr, msg], 64);
        let a = sample_mat(&sk.rho, K, L);
        let (rho2_tail, _) = tail(160, &[(0, &sep(b"sig/rho2")), (96, &mu)]);
        let (c_tail, _) = tail(32 + 64 + K * N / 2, &[(0, &sep(b"sig/c")), (32, &mu)]);
        let at = &self.attempt;
        let mut outs = vec![at.accept, at.ctilde];
        outs.extend(&at.z);
        outs.extend(&at.h);
        for attempt in 0..1000usize {
            let ins = attempt_inputs(sk, &a, &rho2_tail, &c_tail, attempt);
            let res = at.g.eval(&ins, &outs);
            if res[0][0] != 1 {
                continue;
            }
            let mut sig = bytes(&res[1]);
            let z = res[2..2 + L].concat();
            let zenc: Vec<u32> = z.iter().map(|&v| (GAMMA1 + Q - v) % Q).collect();
            sig.extend(pack(&zenc, 20));
            sig.extend(encode_hint(&res[2 + L..]));
            return (sig, attempt + 1);
        }
        panic!("no signature in 1000 attempts (expected ~5)");
    }

    /// Diagnostics: every rejection condition, attempt by attempt, ignoring
    /// acceptance — `[z ok, r₀ ok, c·t₀ ok, hints ok, hint count]`. Reads
    /// secret-derived flags, so it is for tests, never for a real key.
    #[doc(hidden)]
    pub fn diagnose(&self, sk: &SigningKey, msg: &[u8], attempts: usize) -> Vec<[u32; 5]> {
        let mu = shake(b"sig/mu", &[&sk.tr, msg], 64);
        let a = sample_mat(&sk.rho, K, L);
        let (rho2_tail, _) = tail(160, &[(0, &sep(b"sig/rho2")), (96, &mu)]);
        let (c_tail, _) = tail(32 + 64 + K * N / 2, &[(0, &sep(b"sig/c")), (32, &mu)]);
        let at = &self.attempt;
        (0..attempts)
            .map(|attempt| {
                let ins = attempt_inputs(sk, &a, &rho2_tail, &c_tail, attempt);
                let r = at.g.eval(&ins, &at.checks);
                [r[0][0], r[1][0], r[2][0], r[3][0], r[4][0]]
            })
            .collect()
    }

    pub fn verify(&self, pk: &[u8], msg: &[u8], sig: &[u8]) -> bool {
        if pk.len() != PK_BYTES || sig.len() != SIG_BYTES {
            return false;
        }
        let (rho, t1b) = pk.split_at(32);
        let t1 = unpack(t1b, 10, K * N);
        let ctilde = &sig[..CTILDE_WORDS * 4];
        let zb = &sig[CTILDE_WORDS * 4..CTILDE_WORDS * 4 + L * N * 20 / 8];
        let z: Vec<u32> = unpack(zb, 20, L * N)
            .iter()
            .map(|&e| (GAMMA1 + Q - e) % Q)
            .collect();
        let Some(h) = decode_hint(&sig[sig.len() - OMEGA - K..]) else {
            return false;
        };
        if z.iter()
            .any(|&v| centred(v).unsigned_abs() >= (GAMMA1 - BETA) as u64)
        {
            return false;
        }
        let c = sample_in_ball_host(ctilde);
        let a = sample_mat(rho, K, L);
        let w = self.verify.g.eval(&[a, t1, z, c], &self.verify.w);
        let mut w1 = Vec::with_capacity(K * N);
        for (wi, hi) in w.iter().zip(&h) {
            w1.extend(wi.iter().zip(hi).map(|(&r, &hb)| use_hint(hb, r)));
        }
        let tr = shake(b"sig/tr", &[pk], 64);
        let mu = shake(b"sig/mu", &[&tr, msg], 64);
        let w1b = pack(&w1, 4);
        let want = shake(b"sig/c", &[&mu, &w1b], 48);
        // ctilde is public; an ordinary compare is fine here
        want == ctilde
    }
}

/// The attempt graph's inputs for attempt number `attempt`.
fn attempt_inputs(
    sk: &SigningKey,
    a: &[u32],
    rho2_tail: &[u32],
    c_tail: &[u32],
    attempt: usize,
) -> [Vec<u32>; 8] {
    let mut y_tails = Vec::new();
    for i in 0..L {
        let nonce = ((attempt * L + i) as u16).to_le_bytes();
        let (ti, _) = tail(32 + 64 + 2, &[(0, &sep(b"sig/y")), (96, &nonce)]);
        y_tails.extend(ti);
    }
    [
        a.to_vec(),
        sk.s1.clone(),
        sk.s2.clone(),
        sk.t0.clone(),
        sk.key.clone(),
        rho2_tail.to_vec(),
        y_tails,
        c_tail.to_vec(),
    ]
}

fn centred(v: u32) -> i64 {
    if v > (Q - 1) / 2 {
        v as i64 - Q as i64
    } else {
        v as i64
    }
}

/// Host Decompose on public data.
fn decompose(r: u32) -> (u32, i64) {
    let alpha = 2 * GAMMA2 as i64;
    let mut r0 = r as i64 % alpha;
    if r0 > alpha / 2 {
        r0 -= alpha;
    }
    if r as i64 - r0 == Q as i64 - 1 {
        (0, r0 - 1)
    } else {
        (((r as i64 - r0) / alpha) as u32, r0)
    }
}

fn use_hint(h: u32, r: u32) -> u32 {
    let m = (Q - 1) / (2 * GAMMA2);
    let (r1, r0) = decompose(r);
    match (h, r0 > 0) {
        (1, true) => (r1 + 1) % m,
        (1, false) => (r1 + m - 1) % m,
        _ => r1,
    }
}

fn encode_hint(h: &[Vec<u32>]) -> Vec<u8> {
    let mut y = vec![0u8; OMEGA + K];
    let mut idx = 0;
    for (i, hi) in h.iter().enumerate() {
        for (j, &b) in hi.iter().enumerate() {
            if b == 1 {
                y[idx] = j as u8;
                idx += 1;
            }
        }
        y[OMEGA + i] = idx as u8;
    }
    y
}

fn decode_hint(y: &[u8]) -> Option<Vec<Vec<u32>>> {
    let mut h = vec![vec![0u32; N]; K];
    let mut idx = 0usize;
    for (i, hi) in h.iter_mut().enumerate() {
        let end = y[OMEGA + i] as usize;
        if end < idx || end > OMEGA {
            return None;
        }
        for k in idx..end {
            if k > idx && y[k] <= y[k - 1] {
                return None; // strictly increasing: one encoding per hint
            }
            hi[y[k] as usize] = 1;
        }
        idx = end;
    }
    y[idx..OMEGA].iter().all(|&b| b == 0).then_some(h)
}
