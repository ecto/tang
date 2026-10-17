//! ryng's KEM: module-LWE over the one ring ℤ_q[x]/(x²⁵⁶ + 1), q = 8380417,
//! the modulus ML-DSA already uses. **Unaudited research code — do not use to
//! protect anything.**
//!
//! The lattice arithmetic is three tang [`ModGraph`]s (keygen, encrypt,
//! decrypt), each checked by the secrecy pass: secrets never steer an
//! address or a count, and the only places a secret-derived value becomes
//! public are named `declassify` nodes whose proof is the MLWE assumption.
//! The secret noise — SHAKE-256 of the seed, then binomial or uniform
//! sampling — is a fourth graph ([`sample`], over [`keccak`]), so no secret
//! byte is touched outside a checked graph except by copying. The hashes
//! of the Fujisaki–Okamoto transform (G, H, J on m and z), sampling the
//! public matrix, and packing are still host Rust.
//!
//! ```
//! let ryng = ryng::Ryng::new(ryng::Params::RYNG);
//! let (pk, sk) = ryng.keygen(&[7; 64]);
//! let (ct, k1) = ryng.seal(&pk, &[9; 32]);
//! assert_eq!(ryng.open(&sk, &ct), k1);
//! assert_eq!((pk.len(), ct.len()), (1312, 1440));
//! ```

pub mod keccak;
pub mod sample;

use sha3::digest::{ExtendableOutput, Update, XofReader};
use sha3::{Digest, Sha3_256, Sha3_512, Shake128, Shake256};
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

fn bits(q: u32) -> u32 {
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

/// The three graphs for one parameter set, built once.
pub struct Ryng {
    pub p: Params,
    keygen: Keygen,
    enc: Encrypt,
    dec: Decrypt,
    /// s, e from σ: k binomial + k uniform polys. `None` for the toy set,
    /// which is driven by hand through the `pke_*` calls.
    keygen_noise: Option<sample::Sampler>,
    /// r, e₁, e₂ from the coins: k binomial + (k + 1) uniform polys.
    enc_noise: Option<sample::Sampler>,
}

fn row(g: &mut ModGraph, x: ModId, i: usize, n: usize) -> ModId {
    let v = ModGraph::view(&[&[1]], &[(i * n) as i64]);
    g.map(&[n], &[(x, v)], |b| b.arg(0))
}

fn zip(
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

fn each(g: &mut ModGraph, a: ModId, f: impl FnOnce(&mut Body, BVal) -> BVal) -> ModId {
    let n = g.shape(a)[0];
    g.map(&[n], &[(a, ModGraph::id(1))], |bd| {
        let x = bd.arg(0);
        f(bd, x)
    })
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

impl Ryng {
    pub fn new(p: Params) -> Self {
        let (n, k) = (p.n, p.k);

        // keygen: t = A·s + e, published as its high bits
        let mut g = ModGraph::new(p.q);
        let a = g.input(&[k * k * n], Label::Public);
        let s = g.input(&[k * n], Label::Secret);
        let e = g.input(&[k * n], Label::Secret);
        let sh: Vec<ModId> = (0..k)
            .map(|j| {
                let sj = row(&mut g, s, j, n);
                ntt(&mut g, sj, p.zeta)
            })
            .collect();
        let t1 = (0..k)
            .map(|i| {
                let as_ = inner(&mut g, &p, a, |j| i * k + j, &sh);
                let ei = row(&mut g, e, i, n);
                let t = zip(&mut g, as_, ei, |b, x, y| b.add(x, y));
                let t1 = if p.dt > 0 {
                    each(&mut g, t, |b, x| b.power2round(x, p.dt))
                } else {
                    t
                };
                g.declassify(t1, PK_PROOF)
            })
            .collect();
        let keygen = Keygen { g, t1 };

        // encrypt: u = Aᵀr + e₁, v = tᵀr + e₂ + m·⌊q/2⌋, both compressed
        let mut g = ModGraph::new(p.q);
        let a = g.input(&[k * k * n], Label::Public);
        let t1 = g.input(&[k * n], Label::Public);
        let r = g.input(&[k * n], Label::Secret);
        let e1 = g.input(&[k * n], Label::Secret);
        let e2 = g.input(&[n], Label::Secret);
        let m = g.input(&[n], Label::Secret);
        let rh: Vec<ModId> = (0..k)
            .map(|j| {
                let rj = row(&mut g, r, j, n);
                ntt(&mut g, rj, p.zeta)
            })
            .collect();
        let th: Vec<ModId> = (0..k)
            .map(|j| {
                let tj = row(&mut g, t1, j, n);
                let t = each(&mut g, tj, |b, x| {
                    let s = b.lit(1 << p.dt);
                    b.mul(x, s)
                });
                ntt(&mut g, t, p.zeta)
            })
            .collect();
        let u = (0..k)
            .map(|i| {
                let ar = inner(&mut g, &p, a, |j| j * k + i, &rh);
                let ei = row(&mut g, e1, i, n);
                let ui = zip(&mut g, ar, ei, |b, x, y| b.add(x, y));
                let ui = if p.du > 0 {
                    each(&mut g, ui, |b, x| b.compress(x, p.du))
                } else {
                    ui
                };
                g.declassify(ui, CT_PROOF)
            })
            .collect();
        let mut acc: Option<ModId> = None;
        for j in 0..k {
            let prod = zip(&mut g, th[j], rh[j], |b, x, y| b.mul(x, y));
            acc = Some(match acc {
                None => prod,
                Some(s) => zip(&mut g, s, prod, |b, x, y| b.add(x, y)),
            });
        }
        let tr = intt(&mut g, acc.expect("k ≥ 1"), p.zeta);
        let half = p.q / 2;
        let v = zip(&mut g, tr, e2, |b, x, y| b.add(x, y));
        let v = zip(&mut g, v, m, |b, x, bit| {
            let h = b.lit(half);
            let mh = b.mul(bit, h);
            b.add(x, mh)
        });
        let v = if p.dv > 0 {
            each(&mut g, v, |b, x| b.compress(x, p.dv))
        } else {
            v
        };
        let v = g.declassify(v, CT_PROOF);
        let enc = Encrypt { g, u, v };

        // decrypt: m = round(2·(v − sᵀu) / q)
        let mut g = ModGraph::new(p.q);
        let u = g.input(&[k * n], Label::Public);
        let v = g.input(&[n], Label::Public);
        let s = g.input(&[k * n], Label::Secret);
        let sh: Vec<ModId> = (0..k)
            .map(|j| {
                let sj = row(&mut g, s, j, n);
                ntt(&mut g, sj, p.zeta)
            })
            .collect();
        let uh: Vec<ModId> = (0..k)
            .map(|j| {
                let uj = row(&mut g, u, j, n);
                let uj = if p.du > 0 {
                    each(&mut g, uj, |b, x| b.decompress(x, p.du))
                } else {
                    uj
                };
                ntt(&mut g, uj, p.zeta)
            })
            .collect();
        let mut acc: Option<ModId> = None;
        for j in 0..k {
            let prod = zip(&mut g, sh[j], uh[j], |b, x, y| b.mul(x, y));
            acc = Some(match acc {
                None => prod,
                Some(s) => zip(&mut g, s, prod, |b, x, y| b.add(x, y)),
            });
        }
        let su = intt(&mut g, acc.expect("k ≥ 1"), p.zeta);
        let v = if p.dv > 0 {
            each(&mut g, v, |b, x| b.decompress(x, p.dv))
        } else {
            v
        };
        let w = zip(&mut g, v, su, |b, x, y| b.sub(x, y));
        let m = each(&mut g, w, |b, x| b.compress(x, 1));
        let dec = Decrypt { g, m, w };

        let full = p.n == 256;
        let keygen_noise = full.then(|| sample::Sampler::new(p.q, p.n, k, k));
        let enc_noise = full.then(|| sample::Sampler::new(p.q, p.n, k, k + 1));
        Self {
            p,
            keygen,
            enc,
            dec,
            keygen_noise,
            enc_noise,
        }
    }

    /// The secrecy check on all three graphs: the proofs each one relies on.
    pub fn check(&self) -> Result<[Vec<&'static str>; 3], tang_compile::tensor::modq::TaintError> {
        for s in self.keygen_noise.iter().chain(&self.enc_noise) {
            assert!(s.check()?.is_empty(), "sampling declassifies nothing");
        }
        let kg: Vec<_> = self.keygen.t1.iter().map(|&t| (t, Label::Public)).collect();
        let mut en: Vec<_> = self.enc.u.iter().map(|&t| (t, Label::Public)).collect();
        en.push((self.enc.v, Label::Public));
        Ok([
            self.keygen.g.check(&kg)?,
            self.enc.g.check(&en)?,
            self.dec.g.check(&[(self.dec.m, Label::Secret)])?,
        ])
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

    /// Deterministic keygen from 64 bytes of seed (d ‖ z). Returns (pk, sk).
    pub fn keygen(&self, seed: &[u8; 64]) -> (Vec<u8>, Vec<u8>) {
        let p = &self.p;
        let (d, z) = seed.split_at(32);
        let rs = hash512(b"keygen", d);
        let (rho, sigma) = rs.split_at(32);
        let a = sample_a(p, rho);
        let (s, e) = self
            .keygen_noise
            .as_ref()
            .expect("n = 256")
            .run(prf_states(sigma, 2 * p.k));
        let t1 = self.pke_keygen(&a, &s, &e);
        let mut pk = rho.to_vec();
        pk.extend(pack(&t1, bits(p.q) - p.dt));
        let mut sk = pack(&s, bits(p.q));
        sk.extend(&pk);
        sk.extend(hash256(b"pk", &pk));
        sk.extend(z);
        (pk, sk)
    }

    /// Encapsulate under `pk` with 32 bytes of fresh randomness. Returns (ct, key).
    pub fn seal(&self, pk: &[u8], m: &[u8; 32]) -> (Vec<u8>, [u8; 32]) {
        let h = hash256(b"pk", pk);
        let (key, ct) = self.encrypt_fo(pk, m, &h);
        (ct, key)
    }

    /// Decapsulate. Never fails: a tampered ciphertext yields a pseudorandom
    /// key (implicit rejection), chosen by mask.
    pub fn open(&self, sk: &[u8], ct: &[u8]) -> [u8; 32] {
        let p = &self.p;
        let s_len = p.k * p.n * bits(p.q) as usize / 8;
        let pk_len = p.pk_bytes();
        let s = unpack(&sk[..s_len], bits(p.q), p.k * p.n);
        let pk = &sk[s_len..s_len + pk_len];
        let h: [u8; 32] = sk[s_len + pk_len..s_len + pk_len + 32]
            .try_into()
            .expect("h");
        let z = &sk[s_len + pk_len + 32..];
        let (u, v) = self.unpack_ct(ct);
        let (mbits, _) = self.pke_decrypt(&u, &v, &s);
        let m: [u8; 32] = bits_to_bytes(&mbits).try_into().expect("n = 256");
        let (key, ct2) = self.encrypt_fo(pk, &m, &h);
        let reject = shake256_32(b"reject", &[z, ct].concat());
        let same = ct_eq(ct, &ct2);
        let mut out = [0u8; 32];
        for i in 0..32 {
            out[i] = (key[i] & same) | (reject[i] & !same);
        }
        out
    }

    fn encrypt_fo(&self, pk: &[u8], m: &[u8; 32], h: &[u8; 32]) -> ([u8; 32], Vec<u8>) {
        let p = &self.p;
        let kc = hash512(b"seal", &[&m[..], h].concat());
        let (key, coins) = kc.split_at(32);
        let (rho, t1b) = pk.split_at(32);
        let a = sample_a(p, rho);
        let t1 = unpack(t1b, bits(p.q) - p.dt, p.k * p.n);
        let (r, e12) = self
            .enc_noise
            .as_ref()
            .expect("n = 256")
            .run(prf_states(coins, 2 * p.k + 1));
        let (e1, e2) = e12.split_at(p.k * p.n);
        let mbits = bytes_to_bits(m, p.n);
        let (u, v) = self.pke_encrypt(&a, &t1, &r, e1, e2, &mbits);
        let mut ct = pack(&u, p.du);
        ct.extend(pack(&v, p.dv));
        (key.try_into().expect("32"), ct)
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

// -- symmetric primitives: SHA-3 with ryng's domain separators ----------------

/// `SHAKE-256("ryng/" ‖ name)`, the only byte strings the scheme invents.
fn sep(name: &[u8]) -> [u8; 32] {
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

fn hash512(name: &[u8], data: &[u8]) -> Vec<u8> {
    let mut h = Sha3_512::new();
    Digest::update(&mut h, sep(name));
    Digest::update(&mut h, data);
    h.finalize().to_vec()
}

fn shake256_32(name: &[u8], data: &[u8]) -> [u8; 32] {
    let mut x = Shake256::default();
    x.update(&sep(name));
    x.update(data);
    let mut out = [0; 32];
    x.finalize_xof().read(&mut out);
    out
}

/// The absorbed SHAKE-256 states of PRF(seed, 0), …, PRF(seed, count − 1),
/// for the sampler graph to squeeze.
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

fn pack(v: &[u32], w: u32) -> Vec<u8> {
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

fn unpack(b: &[u8], w: u32, count: usize) -> Vec<u32> {
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

fn bytes_to_bits(m: &[u8], n: usize) -> Vec<u32> {
    (0..n).map(|i| (m[i / 8] >> (i % 8) & 1) as u32).collect()
}

fn bits_to_bytes(b: &[u32]) -> Vec<u8> {
    b.chunks(8)
        .map(|c| {
            c.iter()
                .enumerate()
                .fold(0u8, |acc, (i, &x)| acc | ((x as u8 & 1) << i))
        })
        .collect()
}

/// 0xff if equal, 0x00 if not, without an early exit.
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
