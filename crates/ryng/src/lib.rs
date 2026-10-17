//! ryng's KEM: module-LWE over the one ring ℤ_q[x]/(x²⁵⁶ + 1), q = 8380417,
//! the modulus ML-DSA already uses. **Unaudited research code — do not use to
//! protect anything.**
//!
//! Every operation that touches a secret is a tang [`ModGraph`] checked by
//! the secrecy pass: keygen (d → ρ, σ → s, t₁), seal (G(m ‖ H(pk)), the
//! noise, the encryption) and open (decrypt, G, re-encrypt, a mask compare,
//! J(z ‖ ct), a mask select). Secrets never steer an address or a count; the
//! only declassifications are ρ, the public key and the ciphertext, each
//! naming its proof. Hashing is Keccak inside the graph ([`keccak`]); G and
//! J are SHAKE-256. Only public work is host Rust: H(pk), sampling A, and
//! packing.
//!
//! ```
//! let ryng = ryng::Ryng::new(ryng::Params::RYNG);
//! let (pk, sk) = ryng.keygen(&[7; 64]);
//! let (ct, k1) = ryng.seal(&pk, &[9; 32]);
//! assert_eq!(ryng.open(&sk, &ct), k1);
//! assert_eq!((pk.len(), ct.len()), (1312, 1440));
//! ```

pub mod graph;
pub mod keccak;
pub mod sample;
pub mod sig;
pub mod sig_gpu;

use sha3::digest::{ExtendableOutput, Update, XofReader};
use sha3::{Digest, Sha3_256, Shake128, Shake256};
use tang_compile::tensor::modq::{intt, ntt, BVal, Body, Label, ModGraph, ModId};

/// Every constant of one parameter set. `0` for a compression width means
/// "no compression" (the toy set).
#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub q: u32,
    pub n: usize,
    /// A primitive 2n-th root of unity mod q.
    pub zeta: u32,
    pub k: usize,
    pub du: u32,
    pub dv: u32,
    /// Low bits of t dropped from the public key.
    pub dt: u32,
}

impl Params {
    /// The set `docs/plans/2026-10-17-ryng-design.md` derived: core-SVP 2^233,
    /// best lattice-estimator attack 2^242.7, failure ≤ 2^−185.6.
    pub const RYNG: Params = Params {
        q: 8380417,
        n: 256,
        zeta: 1753,
        k: 4,
        du: 10,
        dv: 5,
        dt: 13,
    };

    /// The pencil-and-paper example: ℤ₁₇[x]/(x⁴ + 1), no compression.
    pub const TOY: Params = Params {
        q: 17,
        n: 4,
        zeta: 2,
        k: 1,
        du: 0,
        dv: 0,
        dt: 0,
    };

    pub fn pk_bytes(&self) -> usize {
        32 + self.k * self.n * (bits(self.q) - self.dt) as usize / 8
    }

    pub fn ct_bytes(&self) -> usize {
        (self.k * self.n * self.du as usize + self.n * self.dv as usize) / 8
    }
}

pub(crate) fn bits(q: u32) -> u32 {
    32 - (q - 1).leading_zeros()
}

/// The proofs the KEM's declassifications rest on.
pub const PK_PROOF: &str = "mlwe: t = A·s + e is pseudorandom";
pub const CT_PROOF: &str = "mlwe: (Aᵀr + e₁, tᵀr + e₂) is pseudorandom";

struct Keygen {
    g: ModGraph,
    t1: Vec<ModId>,
}
struct Encrypt {
    g: ModGraph,
    u: Vec<ModId>,
    v: ModId,
}
struct Decrypt {
    g: ModGraph,
    m: ModId,
    w: ModId,
}

/// d → (ρ, σ): ρ is published, σ seeds the noise.
struct Seeds {
    g: ModGraph,
    rho: ModId,
    sigma: ModId,
}
/// σ, A → s (kept) and t₁ (published).
struct KeyT {
    g: ModGraph,
    s: Vec<ModId>,
    t1: Vec<ModId>,
}
/// m, H(pk), A, t₁ → ciphertext (published) and K (kept).
struct Seal {
    g: ModGraph,
    u: Vec<ModId>,
    v: ModId,
    key: ModId,
}
/// s, z, H(pk), ciphertext → K, or J(z ‖ ct) if re-encryption disagrees.
struct Open {
    g: ModGraph,
    key: ModId,
    /// A, t₁, s, z and G's tail: fixed by the key, shared by a GPU batch
    #[cfg_attr(not(test), allow(dead_code))]
    key_inputs: Vec<ModId>,
}

/// The Fujisaki–Okamoto KEM as graphs: every hash of m, z, σ or d is Keccak
/// inside a checked graph, and so is the ciphertext comparison.
struct Fo {
    seeds: Seeds,
    keyt: KeyT,
    seal: Seal,
    open: Open,
}

/// The graphs for one parameter set, built once.
pub struct Ryng {
    pub p: Params,
    keygen: Keygen,
    enc: Encrypt,
    dec: Decrypt,
    /// `None` for the toy set, which is driven by hand through `pke_*`.
    fo: Option<Fo>,
}

pub(crate) fn row(g: &mut ModGraph, x: ModId, i: usize, n: usize) -> ModId {
    let v = ModGraph::view(&[&[1]], &[(i * n) as i64]);
    g.map(&[n], &[(x, v)], |b| b.arg(0))
}

pub(crate) fn zip(
    g: &mut ModGraph,
    a: ModId,
    b: ModId,
    f: impl FnOnce(&mut Body, BVal, BVal) -> BVal,
) -> ModId {
    let n = g.shape(a)[0];
    g.map(&[n], &[(a, ModGraph::id(1)), (b, ModGraph::id(1))], |bd| {
        let (x, y) = (bd.arg(0), bd.arg(1));
        f(bd, x, y)
    })
}

pub(crate) fn each(g: &mut ModGraph, a: ModId, f: impl FnOnce(&mut Body, BVal) -> BVal) -> ModId {
    let n = g.shape(a)[0];
    g.map(&[n], &[(a, ModGraph::id(1))], |bd| {
        let x = bd.arg(0);
        f(bd, x)
    })
}

fn polys(g: &mut ModGraph, x: ModId, count: usize, n: usize) -> Vec<ModId> {
    (0..count).map(|i| row(g, x, i, n)).collect()
}

/// Σ_j A[idx(j)] ∘ x̂_j, then back out of the NTT domain.
fn inner(
    g: &mut ModGraph,
    p: &Params,
    a: ModId,
    idx: impl Fn(usize) -> usize,
    xs: &[ModId],
) -> ModId {
    let mut acc: Option<ModId> = None;
    for (j, &x) in xs.iter().enumerate() {
        let aj = row(g, a, idx(j), p.n);
        let prod = zip(g, aj, x, |b, u, v| b.mul(u, v));
        acc = Some(match acc {
            None => prod,
            Some(s) => zip(g, s, prod, |b, u, v| b.add(u, v)),
        });
    }
    intt(g, acc.expect("k ≥ 1"), p.zeta)
}

// -- the PKE, as builders over any graph ---------------------------------------

/// t₁ = HighBits(A·s + e), not yet declassified.
fn build_keygen(g: &mut ModGraph, p: &Params, a: ModId, s: &[ModId], e: &[ModId]) -> Vec<ModId> {
    let k = p.k;
    let sh: Vec<ModId> = s.iter().map(|&x| ntt(g, x, p.zeta)).collect();
    (0..k)
        .map(|i| {
            let as_ = inner(g, p, a, |j| i * k + j, &sh);
            let t = zip(g, as_, e[i], |b, x, y| b.add(x, y));
            if p.dt > 0 {
                each(g, t, |b, x| b.power2round(x, p.dt))
            } else {
                t
            }
        })
        .collect()
}

/// u = Compress(Aᵀr + e₁), v = Compress(tᵀr + e₂ + m·⌊q/2⌋), not yet declassified.
#[allow(clippy::too_many_arguments)]
fn build_encrypt(
    g: &mut ModGraph,
    p: &Params,
    a: ModId,
    t1: ModId,
    r: &[ModId],
    e1: &[ModId],
    e2: ModId,
    m: ModId,
) -> (Vec<ModId>, ModId) {
    let k = p.k;
    let rh: Vec<ModId> = r.iter().map(|&x| ntt(g, x, p.zeta)).collect();
    let th: Vec<ModId> = (0..k)
        .map(|j| {
            let tj = row(g, t1, j, p.n);
            let t = each(g, tj, |b, x| {
                let s = b.lit(1 << p.dt);
                b.mul(x, s)
            });
            ntt(g, t, p.zeta)
        })
        .collect();
    let u = (0..k)
        .map(|i| {
            let ar = inner(g, p, a, |j| j * k + i, &rh);
            let ui = zip(g, ar, e1[i], |b, x, y| b.add(x, y));
            if p.du > 0 {
                each(g, ui, |b, x| b.compress(x, p.du))
            } else {
                ui
            }
        })
        .collect();
    let mut acc: Option<ModId> = None;
    for j in 0..k {
        let prod = zip(g, th[j], rh[j], |b, x, y| b.mul(x, y));
        acc = Some(match acc {
            None => prod,
            Some(s) => zip(g, s, prod, |b, x, y| b.add(x, y)),
        });
    }
    let tr = intt(g, acc.expect("k ≥ 1"), p.zeta);
    let half = p.q / 2;
    let v = zip(g, tr, e2, |b, x, y| b.add(x, y));
    let v = zip(g, v, m, |b, x, bit| {
        let h = b.lit(half);
        let mh = b.mul(bit, h);
        b.add(x, mh)
    });
    let v = if p.dv > 0 {
        each(g, v, |b, x| b.compress(x, p.dv))
    } else {
        v
    };
    (u, v)
}

/// (m, w): the message bits and the noisy w they round from.
fn build_decrypt(g: &mut ModGraph, p: &Params, u: ModId, v: ModId, s: &[ModId]) -> (ModId, ModId) {
    let k = p.k;
    let sh: Vec<ModId> = s.iter().map(|&x| ntt(g, x, p.zeta)).collect();
    let uh: Vec<ModId> = (0..k)
        .map(|j| {
            let uj = row(g, u, j, p.n);
            let uj = if p.du > 0 {
                each(g, uj, |b, x| b.decompress(x, p.du))
            } else {
                uj
            };
            ntt(g, uj, p.zeta)
        })
        .collect();
    let mut acc: Option<ModId> = None;
    for j in 0..k {
        let prod = zip(g, sh[j], uh[j], |b, x, y| b.mul(x, y));
        acc = Some(match acc {
            None => prod,
            Some(s) => zip(g, s, prod, |b, x, y| b.add(x, y)),
        });
    }
    let su = intt(g, acc.expect("k ≥ 1"), p.zeta);
    let v = if p.dv > 0 {
        each(g, v, |b, x| b.decompress(x, p.dv))
    } else {
        v
    };
    let w = zip(g, v, su, |b, x, y| b.sub(x, y));
    (each(g, w, |b, x| b.compress(x, 1)), w)
}

// -- hashing and noise inside the graph -----------------------------------------

/// SHAKE-256(tail ⊕ parts), squeezed to `out_words`.
fn xof(
    g: &mut ModGraph,
    t: &keccak::Tables,
    tail: ModId,
    parts: &[(ModId, usize)],
    out_words: usize,
) -> ModId {
    let blocks = g.shape(tail)[0] / keccak::SHAKE256_RATE;
    let st = graph::stream(g, tail, parts);
    let st = graph::absorb(g, t, st, blocks);
    let sq = keccak::squeeze(
        g,
        t,
        st,
        keccak::SHAKE256_RATE,
        out_words.div_ceil(keccak::SHAKE256_RATE),
    );
    g.map(&[out_words], &[(sq, ModGraph::id(1))], |b| b.arg(0))
}

/// `cbd` binomial then `unif` uniform polys from PRF(seed, nonce0 + i).
fn noise(
    g: &mut ModGraph,
    t: &keccak::Tables,
    seed: ModId,
    nonce0: u8,
    cbd: usize,
    unif: usize,
) -> Vec<ModId> {
    (0..cbd + unif)
        .map(|i| {
            let nonce = nonce0 + i as u8;
            let (tw, _) = graph::tail(65, &[(0, &sep(b"prf")), (64, &[nonce])]);
            let tw = g.constant_words(&tw);
            let words = if i < cbd {
                keccak::SHAKE256_RATE
            } else {
                3 * keccak::SHAKE256_RATE
            };
            let w = xof(g, t, tw, &[(seed, 8)], words);
            if i < cbd {
                sample::cbd2(g, w, 256)
            } else {
                sample::uniform12(g, w, 256)
            }
        })
        .collect()
}

/// A secret `[1]` that is 1 when `a` and `b` agree everywhere.
fn all_equal(g: &mut ModGraph, pairs: &[(ModId, ModId)]) -> ModId {
    let flags: Vec<ModId> = pairs
        .iter()
        .map(|&(a, b)| {
            let e = zip(g, a, b, |bd, x, y| bd.eq(x, y));
            graph::fold(g, e, graph::Fold::And)
        })
        .collect();
    graph::combine(g, &flags, graph::Fold::And)
}

/// The KEM's message coins: (K, coins) = G(sep ‖ m ‖ H(pk)), then r, e₁, e₂
/// and the encryption — shared by seal and by open's re-encryption.
fn build_seal_core(
    g: &mut ModGraph,
    t: &keccak::Tables,
    p: &Params,
    a: ModId,
    t1: ModId,
    m_words: ModId,
    g_tail: ModId,
) -> (Vec<ModId>, ModId, ModId) {
    let kc = xof(g, t, g_tail, &[(m_words, 8)], 16);
    let key = g.map(&[8], &[(kc, ModGraph::id(1))], |b| b.arg(0));
    let coins = g.map(&[8], &[(kc, ModGraph::view(&[&[1]], &[8]))], |b| b.arg(0));
    let nz = noise(g, t, coins, 0, p.k, p.k + 1);
    let mbits = graph::extract_bits(g, m_words, 1, p.n);
    let (r, e) = nz.split_at(p.k);
    let (u, v) = build_encrypt(g, p, a, t1, r, &e[..p.k], e[p.k], mbits);
    (u, v, key)
}

impl Fo {
    fn new(p: &Params) -> Self {
        let (n, k) = (p.n, p.k);

        // d → ρ (published), σ
        let mut g = ModGraph::new(p.q);
        let t = keccak::Tables::new(&mut g);
        let d = g.input_words(&[8], Label::Secret);
        let (tw, _) = graph::tail(64, &[(0, &sep(b"keygen"))]);
        let tw = g.constant_words(&tw);
        let rs = xof(&mut g, &t, tw, &[(d, 8)], 16);
        let rho = g.map(&[8], &[(rs, ModGraph::id(1))], |b| b.arg(0));
        let rho = g.declassify(rho, RHO_PROOF);
        let sigma = g.map(&[8], &[(rs, ModGraph::view(&[&[1]], &[8]))], |b| b.arg(0));
        let seeds = Seeds { g, rho, sigma };

        // σ, A → s, t₁
        let mut g = ModGraph::new(p.q);
        let t = keccak::Tables::new(&mut g);
        let a = g.input(&[k * k * n], Label::Public);
        let sigma = g.input_words(&[8], Label::Secret);
        let nz = noise(&mut g, &t, sigma, 0, k, k);
        let (s, e) = nz.split_at(k);
        let t1 = build_keygen(&mut g, p, a, s, e)
            .into_iter()
            .map(|x| g.declassify(x, PK_PROOF))
            .collect();
        let keyt = KeyT {
            g,
            s: s.to_vec(),
            t1,
        };

        // seal
        let mut g = ModGraph::new(p.q);
        let t = keccak::Tables::new(&mut g);
        let a = g.input(&[k * k * n], Label::Public);
        let t1 = g.input(&[k * n], Label::Public);
        let m = g.input_words(&[8], Label::Secret);
        let g_tail = g.input_words(&[keccak::SHAKE256_RATE], Label::Public);
        let (u, v, key) = build_seal_core(&mut g, &t, p, a, t1, m, g_tail);
        let u = u.into_iter().map(|x| g.declassify(x, CT_PROOF)).collect();
        let v = g.declassify(v, CT_PROOF);
        let seal = Seal { g, u, v, key };

        // open: decrypt, re-encrypt, compare, choose — nothing declassified
        let mut g = ModGraph::new(p.q);
        let t = keccak::Tables::new(&mut g);
        let a = g.input(&[k * k * n], Label::Public);
        let t1 = g.input(&[k * n], Label::Public);
        let s = g.input(&[k * n], Label::Secret);
        let z = g.input_words(&[8], Label::Secret);
        let u = g.input(&[k * n], Label::Public);
        let v = g.input(&[n], Label::Public);
        let g_tail = g.input_words(&[keccak::SHAKE256_RATE], Label::Public);
        let j_tail = g.input_words(&[12 * keccak::SHAKE256_RATE], Label::Public);
        let sp = polys(&mut g, s, k, n);
        let (mbits, _) = build_decrypt(&mut g, p, u, v, &sp);
        let m = graph::pack_fields(&mut g, mbits, 1);
        let (u2, v2, key) = build_seal_core(&mut g, &t, p, a, t1, m, g_tail);
        let up = polys(&mut g, u, k, n);
        let mut pairs: Vec<(ModId, ModId)> = up.into_iter().zip(u2).collect();
        pairs.push((v, v2));
        let same = all_equal(&mut g, &pairs);
        let reject = xof(&mut g, &t, j_tail, &[(z, 8)], 8);
        let bc = ModGraph::view(&[&[0]], &[0]);
        let key = g.map(
            &[8],
            &[
                (key, ModGraph::id(1)),
                (reject, ModGraph::id(1)),
                (same, bc),
            ],
            |b| {
                let (kk, rj, ok) = (b.arg(0), b.arg(1), b.arg(2));
                b.sel(ok, kk, rj)
            },
        );
        let open = Open {
            g,
            key,
            key_inputs: vec![a, t1, s, z, g_tail],
        };

        Self {
            seeds,
            keyt,
            seal,
            open,
        }
    }

    fn check(&self) -> Result<Vec<Vec<&'static str>>, tang_compile::tensor::modq::TaintError> {
        let mut kt: Vec<_> = self.keyt.t1.iter().map(|&x| (x, Label::Public)).collect();
        kt.extend(self.keyt.s.iter().map(|&x| (x, Label::Secret)));
        let mut se: Vec<_> = self.seal.u.iter().map(|&x| (x, Label::Public)).collect();
        se.push((self.seal.v, Label::Public));
        se.push((self.seal.key, Label::Secret));
        Ok(vec![
            self.seeds.g.check(&[
                (self.seeds.rho, Label::Public),
                (self.seeds.sigma, Label::Secret),
            ])?,
            self.keyt.g.check(&kt)?,
            self.seal.g.check(&se)?,
            self.open.g.check(&[(self.open.key, Label::Secret)])?,
        ])
    }
}

/// ρ is the XOF's first 32 bytes; σ, the next 32, stays secret.
pub const RHO_PROOF: &str = "xof: ρ is independent of σ";

impl Ryng {
    pub fn new(p: Params) -> Self {
        let (n, k) = (p.n, p.k);

        let mut g = ModGraph::new(p.q);
        let a = g.input(&[k * k * n], Label::Public);
        let s = g.input(&[k * n], Label::Secret);
        let e = g.input(&[k * n], Label::Secret);
        let (sp, ep) = (polys(&mut g, s, k, n), polys(&mut g, e, k, n));
        let t1 = build_keygen(&mut g, &p, a, &sp, &ep)
            .into_iter()
            .map(|x| g.declassify(x, PK_PROOF))
            .collect();
        let keygen = Keygen { g, t1 };

        let mut g = ModGraph::new(p.q);
        let a = g.input(&[k * k * n], Label::Public);
        let t1 = g.input(&[k * n], Label::Public);
        let r = g.input(&[k * n], Label::Secret);
        let e1 = g.input(&[k * n], Label::Secret);
        let e2 = g.input(&[n], Label::Secret);
        let m = g.input(&[n], Label::Secret);
        let (rp, e1p) = (polys(&mut g, r, k, n), polys(&mut g, e1, k, n));
        let (u, v) = build_encrypt(&mut g, &p, a, t1, &rp, &e1p, e2, m);
        let u = u.into_iter().map(|x| g.declassify(x, CT_PROOF)).collect();
        let v = g.declassify(v, CT_PROOF);
        let enc = Encrypt { g, u, v };

        let mut g = ModGraph::new(p.q);
        let u = g.input(&[k * n], Label::Public);
        let v = g.input(&[n], Label::Public);
        let s = g.input(&[k * n], Label::Secret);
        let sp = polys(&mut g, s, k, n);
        let (m, w) = build_decrypt(&mut g, &p, u, v, &sp);
        let dec = Decrypt { g, m, w };

        let fo = (p.n == 256).then(|| Fo::new(&p));
        Self {
            p,
            keygen,
            enc,
            dec,
            fo,
        }
    }

    /// The secrecy check on the PKE graphs: the proofs each one relies on.
    pub fn check(&self) -> Result<[Vec<&'static str>; 3], tang_compile::tensor::modq::TaintError> {
        let kg: Vec<_> = self.keygen.t1.iter().map(|&t| (t, Label::Public)).collect();
        let mut en: Vec<_> = self.enc.u.iter().map(|&t| (t, Label::Public)).collect();
        en.push((self.enc.v, Label::Public));
        Ok([
            self.keygen.g.check(&kg)?,
            self.enc.g.check(&en)?,
            self.dec.g.check(&[(self.dec.m, Label::Secret)])?,
        ])
    }

    /// The secrecy check on the KEM graphs (seeds, keygen, seal, open).
    pub fn check_kem(
        &self,
    ) -> Result<Vec<Vec<&'static str>>, tang_compile::tensor::modq::TaintError> {
        self.fo.as_ref().expect("n = 256").check()
    }

    /// Does the decrypt graph's message output pass if declared public? (It must not.)
    pub fn dec_output_is_secret(&self) -> bool {
        self.dec.g.check(&[(self.dec.m, Label::Public)]).is_err()
    }

    // -- the PKE on residues --------------------------------------------------

    pub fn pke_keygen(&self, a: &[u32], s: &[u32], e: &[u32]) -> Vec<u32> {
        let out = self
            .keygen
            .g
            .eval(&[a.to_vec(), s.to_vec(), e.to_vec()], &self.keygen.t1);
        out.concat()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn pke_encrypt(
        &self,
        a: &[u32],
        t1: &[u32],
        r: &[u32],
        e1: &[u32],
        e2: &[u32],
        m: &[u32],
    ) -> (Vec<u32>, Vec<u32>) {
        let mut outs = self.enc.u.clone();
        outs.push(self.enc.v);
        let mut res = self
            .enc
            .g
            .eval(&[a, t1, r, e1, e2, m].map(<[u32]>::to_vec), &outs);
        let v = res.pop().expect("v");
        (res.concat(), v)
    }

    /// Message bits, and the noisy w they were rounded from.
    pub fn pke_decrypt(&self, u: &[u32], v: &[u32], s: &[u32]) -> (Vec<u32>, Vec<u32>) {
        let mut res = self.dec.g.eval(
            &[u.to_vec(), v.to_vec(), s.to_vec()],
            &[self.dec.m, self.dec.w],
        );
        let w = res.pop().expect("w");
        (res.pop().expect("m"), w)
    }

    // -- the KEM --------------------------------------------------------------

    fn fo(&self) -> &Fo {
        self.fo.as_ref().expect("the KEM needs n = 256")
    }

    /// Deterministic keygen from 64 bytes of seed (d ‖ z). Returns (pk, sk).
    pub fn keygen(&self, seed: &[u8; 64]) -> (Vec<u8>, Vec<u8>) {
        let (p, fo) = (&self.p, self.fo());
        let (d, z) = seed.split_at(32);
        let rs = fo
            .seeds
            .g
            .eval(&[words(d)], &[fo.seeds.rho, fo.seeds.sigma]);
        let rho = bytes(&rs[0]);
        let a = sample_a(p, &rho);
        let mut outs = fo.keyt.t1.clone();
        outs.extend(&fo.keyt.s);
        let res = fo.keyt.g.eval(&[a, rs[1].clone()], &outs);
        let t1 = res[..p.k].concat();
        let s = res[p.k..].concat();
        let mut pk = rho;
        pk.extend(pack(&t1, bits(p.q) - p.dt));
        let mut sk = pack(&s, bits(p.q));
        sk.extend(&pk);
        sk.extend(hash256(b"pk", &pk));
        sk.extend(z);
        (pk, sk)
    }

    /// Encapsulate under `pk` with 32 bytes of fresh randomness. Returns (ct, key).
    pub fn seal(&self, pk: &[u8], m: &[u8; 32]) -> (Vec<u8>, [u8; 32]) {
        let (p, fo) = (&self.p, self.fo());
        let h = hash256(b"pk", pk);
        let (rho, t1b) = pk.split_at(32);
        let a = sample_a(p, rho);
        let t1 = unpack(t1b, bits(p.q) - p.dt, p.k * p.n);
        let mut outs = fo.seal.u.clone();
        outs.extend([fo.seal.v, fo.seal.key]);
        let res = fo.seal.g.eval(&[a, t1, words(m), g_tail(&h)], &outs);
        let mut ct = pack(&res[..p.k].concat(), p.du);
        ct.extend(pack(&res[p.k], p.dv));
        (ct, bytes(&res[p.k + 1]).try_into().expect("32"))
    }

    /// Decapsulate. Never fails: a tampered ciphertext yields J(z ‖ ct)
    /// (implicit rejection), chosen by mask inside the graph.
    pub fn open(&self, sk: &[u8], ct: &[u8]) -> [u8; 32] {
        let (p, fo) = (&self.p, self.fo());
        let s_len = p.k * p.n * bits(p.q) as usize / 8;
        let pk_len = p.pk_bytes();
        let s = unpack(&sk[..s_len], bits(p.q), p.k * p.n);
        let pk = &sk[s_len..s_len + pk_len];
        let h = &sk[s_len + pk_len..s_len + pk_len + 32];
        let z = &sk[s_len + pk_len + 32..];
        let (rho, t1b) = pk.split_at(32);
        let a = sample_a(p, rho);
        let t1 = unpack(t1b, bits(p.q) - p.dt, p.k * p.n);
        let (u, v) = self.unpack_ct(ct);
        let (jt, _) = graph::tail(32 + 32 + ct.len(), &[(0, &sep(b"reject")), (64, ct)]);
        let ins = [a, t1, s, words(z), u, v, g_tail(h), jt];
        let key = fo.open.g.eval(&ins, &[fo.open.key]).remove(0);
        bytes(&key).try_into().expect("32")
    }

    fn unpack_ct(&self, ct: &[u8]) -> (Vec<u32>, Vec<u32>) {
        let p = &self.p;
        let ub = p.k * p.n * p.du as usize / 8;
        (
            unpack(&ct[..ub], p.du, p.k * p.n),
            unpack(&ct[ub..], p.dv, p.n),
        )
    }
}

/// G's public tail: sep("seal") ‖ (m, filled in the graph) ‖ H(pk).
fn g_tail(h: &[u8]) -> Vec<u32> {
    graph::tail(96, &[(0, &sep(b"seal")), (64, h)]).0
}

fn words(b: &[u8]) -> Vec<u32> {
    b.chunks(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn bytes(w: &[u32]) -> Vec<u8> {
    w.iter().flat_map(|x| x.to_le_bytes()).collect()
}

// -- symmetric primitives: SHA-3 with ryng's domain separators ----------------

/// `SHAKE-256("ryng/" ‖ name)`, the only byte strings the scheme invents.
pub(crate) fn sep(name: &[u8]) -> [u8; 32] {
    let mut x = Shake256::default();
    x.update(b"ryng/");
    x.update(name);
    let mut out = [0; 32];
    x.finalize_xof().read(&mut out);
    out
}

fn hash256(name: &[u8], data: &[u8]) -> [u8; 32] {
    let mut h = Sha3_256::new();
    Digest::update(&mut h, sep(name));
    Digest::update(&mut h, data);
    h.finalize().into()
}

#[cfg(test)]
fn shake256_32(name: &[u8], data: &[u8]) -> [u8; 32] {
    let mut x = Shake256::default();
    x.update(&sep(name));
    x.update(data);
    let mut out = [0; 32];
    x.finalize_xof().read(&mut out);
    out
}

/// The absorbed SHAKE-256 states of PRF(seed, 0), …, PRF(seed, count − 1),
/// for the standalone sampler graph to squeeze.
#[cfg(test)]
fn prf_states(seed: &[u8], count: usize) -> Vec<u32> {
    (0..count as u8)
        .flat_map(|nonce| {
            keccak::absorb_one_block(&[&sep(b"prf")[..], seed, &[nonce]].concat(), 136)
        })
        .collect()
}

#[cfg(test)]
fn prf(seed: &[u8], nonce: u8, len: usize) -> Vec<u8> {
    let mut x = Shake256::default();
    x.update(&sep(b"prf"));
    x.update(seed);
    x.update(&[nonce]);
    let mut out = vec![0; len];
    x.finalize_xof().read(&mut out);
    out
}

/// The public matrix, sampled directly in the NTT domain by rejection from
/// SHAKE-128. Variable-time, but every byte it touches is public.
fn sample_a(p: &Params, rho: &[u8]) -> Vec<u32> {
    let mask = (1u32 << bits(p.q)) - 1;
    let mut out = Vec::with_capacity(p.k * p.k * p.n);
    for i in 0..p.k {
        for j in 0..p.k {
            let mut x = Shake128::default();
            x.update(&sep(b"A"));
            x.update(rho);
            x.update(&[i as u8, j as u8]);
            let mut rd = x.finalize_xof();
            let mut got = 0;
            while got < p.n {
                let mut b = [0u8; 3];
                rd.read(&mut b);
                let c = u32::from_le_bytes([b[0], b[1], b[2], 0]) & mask;
                if c < p.q {
                    out.push(c);
                    got += 1;
                }
            }
        }
    }
    out
}

/// Host reference for the sampler graph: k polynomials, centred binomial
/// η = 2, (b₀ + b₁) − (b₂ + b₃) per coefficient.
#[cfg(test)]
fn cbd2(p: &Params, seed: &[u8], nonce0: u8) -> Vec<u32> {
    let mut out = Vec::with_capacity(p.k * p.n);
    for i in 0..p.k {
        let buf = prf(seed, nonce0 + i as u8, p.n / 2);
        for byte in buf {
            for half in [byte & 15, byte >> 4] {
                let v = (half & 1) + (half >> 1 & 1);
                let w = (half >> 2 & 1) + (half >> 3 & 1);
                out.push((p.q + v as u32 - w as u32) % p.q);
            }
        }
    }
    out
}

/// Host reference: uniform on [−2¹¹, 2¹¹), 12 masked bits, no rejection. k
/// polys from nonce0 on, or one poly when nonce0 is `2k` (e₂).
#[cfg(test)]
fn uniform12(p: &Params, seed: &[u8], nonce0: u8) -> Vec<u32> {
    let polys = if nonce0 as usize == 2 * p.k { 1 } else { p.k };
    let mut out = Vec::with_capacity(polys * p.n);
    for i in 0..polys {
        let buf = prf(seed, nonce0 + i as u8, p.n * 3 / 2);
        for c in buf.chunks(3) {
            let x = u32::from_le_bytes([c[0], c[1], c[2], 0]);
            for v in [x & 0xfff, x >> 12] {
                out.push((p.q + v - 2048) % p.q);
            }
        }
    }
    out
}

// -- packing -------------------------------------------------------------------

pub(crate) fn pack(v: &[u32], w: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * w as usize / 8);
    let (mut acc, mut n) = (0u64, 0u32);
    for &x in v {
        acc |= (x as u64) << n;
        n += w;
        while n >= 8 {
            out.push(acc as u8);
            acc >>= 8;
            n -= 8;
        }
    }
    out
}

pub(crate) fn unpack(b: &[u8], w: u32, count: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(count);
    let (mut acc, mut n) = (0u64, 0u32);
    let mut it = b.iter();
    while out.len() < count {
        while n < w {
            acc |= (*it.next().expect("short buffer") as u64) << n;
            n += 8;
        }
        out.push((acc & ((1 << w) - 1)) as u32);
        acc >>= w;
        n -= w;
    }
    out
}

#[cfg(test)]
pub(crate) fn bytes_to_bits(m: &[u8], n: usize) -> Vec<u32> {
    (0..n).map(|i| (m[i / 8] >> (i % 8) & 1) as u32).collect()
}

#[cfg(test)]
pub(crate) fn bits_to_bytes(b: &[u32]) -> Vec<u8> {
    b.chunks(8)
        .map(|c| {
            c.iter()
                .enumerate()
                .fold(0u8, |acc, (i, &x)| acc | ((x as u8 & 1) << i))
        })
        .collect()
}

/// 0xff if equal, 0x00 if not, without an early exit.
#[cfg(test)]
fn ct_eq(a: &[u8], b: &[u8]) -> u8 {
    let diff = a
        .iter()
        .zip(b)
        .fold((a.len() ^ b.len()) as u8, |acc, (x, y)| acc | (x ^ y));
    // diff == 0 → 0xff
    ((diff as u16).wrapping_sub(1) >> 8) as u8
}

#[cfg(test)]
mod tests;
