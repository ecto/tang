//! Integer tensors mod q, with a secrecy label on every tensor.
//!
//! A sibling of [`TensorGraph`](super::TensorGraph) for lattice
//! cryptography: same affine [`View`]s, but values are `u32` residues mod a
//! prime `q < 2^31` and every node is `Public` or `Secret`.
//!
//! The shape of the IR is the constant-time argument. Views are fixed affine
//! maps and shapes are static, so a map's control flow and addresses cannot
//! depend on any value; its scalar ops ([`Body`]) are branch-free (Montgomery
//! multiplication, mask-based conditional subtraction and select). Only three
//! ops look at data, and [`ModGraph::check`] polices them:
//!
//! - [`ModGraph::gather`] reads a table at a runtime index — the index must be
//!   `Public`.
//! - [`ModGraph::filter`] keeps the values below a bound (rejection sampling)
//!   — its input must be `Public`.
//! - [`ModGraph::declassify`] turns a `Secret` into a `Public`, naming the
//!   proof that makes it safe.
//!
//! and a tensor the caller declares `Public` must not carry a secret.
//!
//! ```
//! use tang_compile::tensor::modq::{Label, ModGraph};
//!
//! let mut g = ModGraph::new(17);
//! let s = g.input(&[4], Label::Secret);
//! let t = g.map(&[4], &[(s, ModGraph::id(1))], |b| {
//!     let x = b.arg(0);
//!     b.add(x, x)
//! });
//! assert_eq!(g.eval(&[vec![1, 9, 16, 0]], &[t]), vec![vec![2, 1, 15, 0]]);
//! assert!(g.check(&[(t, Label::Public)]).is_err());   // a secret, published
//! assert!(g.check(&[(t, Label::Secret)]).is_ok());
//! ```

use super::{Form, View};

/// Who may learn a value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Label {
    Public,
    Secret,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModId(u32);

/// One scalar op inside a map body. Every op runs in constant time.
#[derive(Clone, Copy, Debug)]
enum BOp {
    Arg(usize),
    Lit(u32),
    /// Coordinate along a point dim (public by construction).
    Coord(usize),
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    /// `a` where `c == 1`, `b` where `c == 0`, by mask.
    Sel(usize, usize, usize),
    /// round(x·2^d / q) mod 2^d: keep the top d bits of a residue.
    Compress(usize, u32),
    /// round(y·q / 2^d): back to a residue.
    Decompress(usize, u32),
    /// round(x / 2^d), ties down: the high part when the low d bits are dropped.
    Power2Round(usize, u32),
}

/// A branch-free scalar program over residues mod q.
pub struct Body {
    ops: Vec<BOp>,
}

/// A value inside a [`Body`].
#[derive(Clone, Copy, Debug)]
pub struct BVal(usize);

impl Body {
    fn push(&mut self, op: BOp) -> BVal {
        self.ops.push(op);
        BVal(self.ops.len() - 1)
    }
    pub fn arg(&mut self, i: usize) -> BVal {
        self.push(BOp::Arg(i))
    }
    pub fn lit(&mut self, v: u32) -> BVal {
        self.push(BOp::Lit(v))
    }
    pub fn coord(&mut self, d: usize) -> BVal {
        self.push(BOp::Coord(d))
    }
    pub fn add(&mut self, a: BVal, b: BVal) -> BVal {
        self.push(BOp::Add(a.0, b.0))
    }
    pub fn sub(&mut self, a: BVal, b: BVal) -> BVal {
        self.push(BOp::Sub(a.0, b.0))
    }
    pub fn mul(&mut self, a: BVal, b: BVal) -> BVal {
        self.push(BOp::Mul(a.0, b.0))
    }
    pub fn sel(&mut self, c: BVal, a: BVal, b: BVal) -> BVal {
        self.push(BOp::Sel(c.0, a.0, b.0))
    }
    pub fn compress(&mut self, x: BVal, d: u32) -> BVal {
        self.push(BOp::Compress(x.0, d))
    }
    pub fn decompress(&mut self, y: BVal, d: u32) -> BVal {
        self.push(BOp::Decompress(y.0, d))
    }
    pub fn power2round(&mut self, x: BVal, d: u32) -> BVal {
        self.push(BOp::Power2Round(x.0, d))
    }
}

enum MOp {
    Input(usize),
    Const(Vec<u32>),
    Reshape(ModId),
    Map {
        args: Vec<(ModId, View)>,
        body: Body,
    },
    Gather {
        table: ModId,
        index: ModId,
    },
    Filter {
        src: ModId,
        bound: u32,
    },
    Declassify {
        src: ModId,
        proof: &'static str,
    },
}

struct MNode {
    op: MOp,
    shape: Vec<usize>,
    /// Declared label of an input; derived for everything else.
    label: Label,
}

/// Why [`ModGraph::check`] refused a graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TaintError {
    /// A secret decides which table entry is read: a cache channel.
    SecretIndex(ModId),
    /// A secret decides how many values survive: a timing channel.
    SecretFilter(ModId),
    /// A tensor declared public carries a secret.
    SecretOutput(ModId),
}

/// Constant-time arithmetic mod q, Montgomery with R = 2^32.
#[derive(Clone, Copy)]
struct Field {
    q: u32,
    /// −q⁻¹ mod 2^32
    qneg_inv: u32,
    /// R² mod q, to leave the Montgomery domain after one product.
    r2: u32,
    /// floor(2^64 / q), for division by q without a divide instruction.
    barrett: u64,
}

impl Field {
    fn new(q: u32) -> Self {
        assert!(q % 2 == 1 && q < 1 << 31, "q must be odd and below 2^31");
        let mut inv: u32 = 1;
        for _ in 0..5 {
            inv = inv.wrapping_mul(2u32.wrapping_sub(q.wrapping_mul(inv)));
        }
        let r2 = ((1u128 << 64) % q as u128) as u32;
        Self {
            q,
            qneg_inv: inv.wrapping_neg(),
            r2,
            barrett: ((1u128 << 64) / q as u128) as u64,
        }
    }
    /// x − q if x ≥ q, for x < 2q, by mask.
    fn csub(&self, x: u32) -> u32 {
        let y = x.wrapping_sub(self.q);
        let mask = 0u32.wrapping_sub(y >> 31);
        y.wrapping_add(self.q & mask)
    }
    fn add(&self, a: u32, b: u32) -> u32 {
        self.csub(a + b)
    }
    fn sub(&self, a: u32, b: u32) -> u32 {
        self.csub(a + self.q - b)
    }
    /// a·b·R⁻¹ mod q.
    fn mont(&self, a: u32, b: u32) -> u32 {
        let t = a as u64 * b as u64;
        let m = (t as u32).wrapping_mul(self.qneg_inv);
        self.csub(((t + m as u64 * self.q as u64) >> 32) as u32)
    }
    fn mul(&self, a: u32, b: u32) -> u32 {
        self.mont(self.mont(a, b), self.r2)
    }
    /// floor(v / q) for v < 2^56: the Barrett estimate is low by at most one,
    /// fixed by mask.
    fn div_q(&self, v: u64) -> u64 {
        let est = ((v as u128 * self.barrett as u128) >> 64) as u64;
        let r = v - est * self.q as u64;
        let ge = (((self.q as u64).wrapping_sub(1).wrapping_sub(r)) >> 63) & 1;
        est + ge
    }
    fn compress(&self, x: u32, d: u32) -> u32 {
        let v = ((x as u64) << d) + (self.q as u64 >> 1);
        (self.div_q(v) as u32) & ((1u32 << d) - 1)
    }
    fn decompress(&self, y: u32, d: u32) -> u32 {
        ((y as u64 * self.q as u64 + (1u64 << d >> 1)) >> d) as u32
    }
    fn power2round(&self, x: u32, d: u32) -> u32 {
        (x + (1 << d >> 1) - 1) >> d
    }
    fn sel(&self, c: u32, a: u32, b: u32) -> u32 {
        let mask = 0u32.wrapping_sub(c & 1);
        (a & mask) | (b & !mask)
    }
}

/// A graph of integer tensors mod q.
pub struct ModGraph {
    f: Field,
    nodes: Vec<MNode>,
    inputs: usize,
}

fn row_major(i: usize, shape: &[usize]) -> Vec<usize> {
    let mut idx = vec![0; shape.len()];
    let mut r = i;
    for d in (0..shape.len()).rev() {
        idx[d] = r % shape[d];
        r /= shape[d];
    }
    idx
}

fn flat(idx: &[usize], shape: &[usize]) -> usize {
    idx.iter().zip(shape).fold(0, |acc, (i, n)| acc * n + i)
}

impl ModGraph {
    pub fn new(q: u32) -> Self {
        Self {
            f: Field::new(q),
            nodes: Vec::new(),
            inputs: 0,
        }
    }

    pub fn q(&self) -> u32 {
        self.f.q
    }

    pub fn shape(&self, t: ModId) -> &[usize] {
        &self.nodes[t.0 as usize].shape
    }

    fn push(&mut self, op: MOp, shape: Vec<usize>, label: Label) -> ModId {
        self.nodes.push(MNode { op, shape, label });
        ModId(self.nodes.len() as u32 - 1)
    }

    /// The identity view of a rank-`r` source.
    pub fn id(r: usize) -> View {
        View::identity(r, r)
    }

    /// A view whose source coordinate `s` is `offsets[s] + Σ coefs[s][d]·i_d`.
    pub fn view(coefs: &[&[i64]], offsets: &[i64]) -> View {
        View {
            forms: coefs
                .iter()
                .zip(offsets)
                .map(|(c, &o)| Form {
                    coefs: c.to_vec(),
                    offset: o,
                })
                .collect(),
            guard: false,
        }
    }

    pub fn input(&mut self, shape: &[usize], label: Label) -> ModId {
        self.inputs += 1;
        self.push(MOp::Input(self.inputs - 1), shape.to_vec(), label)
    }

    /// A public table, reduced mod q.
    pub fn constant(&mut self, values: &[u32]) -> ModId {
        let q = self.f.q;
        let v = values.iter().map(|x| x % q).collect();
        self.push(MOp::Const(v), vec![values.len()], Label::Public)
    }

    /// The same values row-major under a new shape. Free.
    pub fn reshape(&mut self, src: ModId, shape: &[usize]) -> ModId {
        let n: usize = shape.iter().product();
        assert_eq!(
            n,
            self.shape(src).iter().product::<usize>(),
            "reshape changes size"
        );
        self.push(MOp::Reshape(src), shape.to_vec(), Label::Public)
    }

    /// `out[i] = body(args[0][view_0(i)], …)` over every point of `shape`.
    pub fn map(
        &mut self,
        shape: &[usize],
        args: &[(ModId, View)],
        f: impl FnOnce(&mut Body) -> BVal,
    ) -> ModId {
        for (src, v) in args {
            assert_eq!(
                v.forms.len(),
                self.shape(*src).len(),
                "view rank ≠ source rank"
            );
            assert!(
                v.forms.iter().all(|f| f.coefs.len() == shape.len()),
                "view arity ≠ map rank"
            );
        }
        let mut body = Body { ops: Vec::new() };
        let out = f(&mut body);
        assert_eq!(
            out.0,
            body.ops.len() - 1,
            "the body's last op is its result"
        );
        let op = MOp::Map {
            args: args.to_vec(),
            body,
        };
        self.push(op, shape.to_vec(), Label::Public)
    }

    /// `out[i] = table[index[i]]`. Data-dependent address: index must be public.
    pub fn gather(&mut self, table: ModId, index: ModId) -> ModId {
        let shape = self.shape(index).to_vec();
        self.push(MOp::Gather { table, index }, shape, Label::Public)
    }

    /// The values of `src` below `bound`, in order, padded with zeros to the
    /// same length. How many survive depends on data: src must be public.
    pub fn filter(&mut self, src: ModId, bound: u32) -> ModId {
        let shape = self.shape(src).to_vec();
        self.push(MOp::Filter { src, bound }, shape, Label::Public)
    }

    /// Publish a secret, naming the proof that it reveals nothing.
    pub fn declassify(&mut self, src: ModId, proof: &'static str) -> ModId {
        let shape = self.shape(src).to_vec();
        self.push(MOp::Declassify { src, proof }, shape, Label::Public)
    }

    /// Every node's label, inputs as declared, the rest joined from their
    /// sources.
    pub fn labels(&self) -> Vec<Label> {
        let mut l: Vec<Label> = Vec::with_capacity(self.nodes.len());
        for n in &self.nodes {
            let label = match &n.op {
                MOp::Input(_) => n.label,
                MOp::Const(_) | MOp::Declassify { .. } => Label::Public,
                MOp::Reshape(s) | MOp::Filter { src: s, .. } => l[s.0 as usize],
                MOp::Map { args, .. } => args
                    .iter()
                    .map(|(s, _)| l[s.0 as usize])
                    .max()
                    .unwrap_or(Label::Public),
                MOp::Gather { table, index } => l[table.0 as usize].max(l[index.0 as usize]),
            };
            l.push(label);
        }
        l
    }

    /// Refuse the graph if a secret steers an address or a count, or reaches
    /// an output declared public. Returns the proofs every declassify relies on.
    pub fn check(&self, outputs: &[(ModId, Label)]) -> Result<Vec<&'static str>, TaintError> {
        let l = self.labels();
        let secret = |t: &ModId| l[t.0 as usize] == Label::Secret;
        let mut proofs = Vec::new();
        for (i, n) in self.nodes.iter().enumerate() {
            let id = ModId(i as u32);
            match &n.op {
                MOp::Gather { index, .. } if secret(index) => {
                    return Err(TaintError::SecretIndex(id))
                }
                MOp::Filter { src, .. } if secret(src) => return Err(TaintError::SecretFilter(id)),
                MOp::Declassify { proof, .. } => proofs.push(*proof),
                _ => {}
            }
        }
        for (t, want) in outputs {
            if *want == Label::Public && secret(t) {
                return Err(TaintError::SecretOutput(*t));
            }
        }
        Ok(proofs)
    }

    /// Run the graph. Inputs are reduced mod q.
    pub fn eval(&self, inputs: &[Vec<u32>], outputs: &[ModId]) -> Vec<Vec<u32>> {
        let f = self.f;
        let mut vals: Vec<Vec<u32>> = Vec::with_capacity(self.nodes.len());
        for n in &self.nodes {
            let size: usize = n.shape.iter().product();
            let v = match &n.op {
                MOp::Input(i) => {
                    assert_eq!(inputs[*i].len(), size, "input {i} has the wrong length");
                    inputs[*i].iter().map(|x| x % f.q).collect()
                }
                MOp::Const(c) => c.clone(),
                MOp::Reshape(s) | MOp::Declassify { src: s, .. } => vals[s.0 as usize].clone(),
                MOp::Gather { table, index } => {
                    let t = &vals[table.0 as usize];
                    vals[index.0 as usize]
                        .iter()
                        .map(|&i| t[i as usize % t.len()])
                        .collect()
                }
                MOp::Filter { src, bound } => {
                    let mut kept: Vec<u32> = vals[src.0 as usize]
                        .iter()
                        .copied()
                        .filter(|x| x < bound)
                        .collect();
                    kept.resize(size, 0);
                    kept
                }
                MOp::Map { args, body } => {
                    let mut out = Vec::with_capacity(size);
                    let mut reg = vec![0u32; body.ops.len()];
                    for p in 0..size {
                        let idx = row_major(p, &n.shape);
                        for (k, op) in body.ops.iter().enumerate() {
                            reg[k] = match *op {
                                BOp::Arg(a) => {
                                    let (src, view) = &args[a];
                                    let at = flat(&view.coords(&idx), self.shape(*src));
                                    vals[src.0 as usize][at]
                                }
                                BOp::Lit(x) => x % f.q,
                                BOp::Coord(d) => idx[d] as u32,
                                BOp::Add(a, b) => f.add(reg[a], reg[b]),
                                BOp::Sub(a, b) => f.sub(reg[a], reg[b]),
                                BOp::Mul(a, b) => f.mul(reg[a], reg[b]),
                                BOp::Sel(c, a, b) => f.sel(reg[c], reg[a], reg[b]),
                                BOp::Compress(a, d) => f.compress(reg[a], d),
                                BOp::Decompress(a, d) => f.decompress(reg[a], d),
                                BOp::Power2Round(a, d) => f.power2round(reg[a], d),
                            };
                        }
                        out.push(reg[body.ops.len() - 1]);
                    }
                    out
                }
            };
            vals.push(v);
        }
        outputs.iter().map(|t| vals[t.0 as usize].clone()).collect()
    }
}

/// The negacyclic NTT over Z_q[x]/(x^n + 1) as a graph of log2(n)
/// Cooley–Tukey stages, output in bit-reversed order (the Dilithium order).
/// `zeta` is a primitive 2n-th root of unity. Pointwise products of two
/// transforms are the transform of the negacyclic product.
pub fn ntt(g: &mut ModGraph, x: ModId, zeta: u32) -> ModId {
    let n = g.shape(x)[0];
    let zetas = g.constant(&twiddles(g.q(), zeta, n));
    let mut cur = x;
    let mut len = n / 2;
    while len >= 1 {
        let blocks = n / (2 * len);
        let src = g.reshape(cur, &[blocks, 2, len]);
        let lo = ModGraph::view(&[&[1, 0, 0], &[0, 0, 0], &[0, 0, 1]], &[0, 0, 0]);
        let hi = ModGraph::view(&[&[1, 0, 0], &[0, 0, 0], &[0, 0, 1]], &[0, 1, 0]);
        let z = ModGraph::view(&[&[1, 0, 0]], &[blocks as i64]);
        let out = g.map(
            &[blocks, 2, len],
            &[(src, lo), (src, hi), (zetas, z)],
            |b| {
                let (x0, x1, z) = (b.arg(0), b.arg(1), b.arg(2));
                let t = b.mul(z, x1);
                let (top, bot) = (b.add(x0, t), b.sub(x0, t));
                let h = b.coord(1);
                b.sel(h, bot, top)
            },
        );
        cur = g.reshape(out, &[n]);
        len /= 2;
    }
    cur
}

/// The inverse of [`ntt`]: Gentleman–Sande stages, then a scale by n⁻¹.
pub fn intt(g: &mut ModGraph, x: ModId, zeta: u32) -> ModId {
    let n = g.shape(x)[0];
    let q = g.q();
    let neg: Vec<u32> = twiddles(q, zeta, n).iter().map(|z| (q - z) % q).collect();
    let zetas = g.constant(&neg);
    let mut cur = x;
    let mut len = 1;
    while len < n {
        let blocks = n / (2 * len);
        let src = g.reshape(cur, &[blocks, 2, len]);
        let lo = ModGraph::view(&[&[1, 0, 0], &[0, 0, 0], &[0, 0, 1]], &[0, 0, 0]);
        let hi = ModGraph::view(&[&[1, 0, 0], &[0, 0, 0], &[0, 0, 1]], &[0, 1, 0]);
        let z = ModGraph::view(&[&[-1, 0, 0]], &[2 * blocks as i64 - 1]);
        let out = g.map(
            &[blocks, 2, len],
            &[(src, lo), (src, hi), (zetas, z)],
            |b| {
                let (x0, x1, z) = (b.arg(0), b.arg(1), b.arg(2));
                let top = b.add(x0, x1);
                let d = b.sub(x0, x1);
                let bot = b.mul(z, d);
                let h = b.coord(1);
                b.sel(h, bot, top)
            },
        );
        cur = g.reshape(out, &[n]);
        len *= 2;
    }
    let ninv = pow(n as u32, q - 2, q);
    g.map(&[n], &[(cur, ModGraph::id(1))], |b| {
        let (x, s) = (b.arg(0), b.lit(ninv));
        b.mul(x, s)
    })
}

fn pow(b: u32, mut e: u32, q: u32) -> u32 {
    let mut r = 1u64;
    let mut b64 = b as u64 % q as u64;
    while e > 0 {
        if e & 1 == 1 {
            r = r * b64 % q as u64;
        }
        b64 = b64 * b64 % q as u64;
        e >>= 1;
    }
    r as u32
}

/// `zetas[k] = zeta^brv(k)` for k < n, brv over log2(n) bits. Public constants.
fn twiddles(q: u32, zeta: u32, n: usize) -> Vec<u32> {
    let bits = n.trailing_zeros();
    (0..n as u32)
        .map(|k| pow(zeta, k.reverse_bits() >> (32 - bits), q))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const Q: u32 = 8380417;
    const ZETA: u32 = 1753;
    const N: usize = 256;

    fn lcg(seed: &mut u64) -> u32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as u32) % Q
    }

    fn schoolbook(a: &[u32], b: &[u32]) -> Vec<u32> {
        let mut c = vec![0i128; N];
        for i in 0..N {
            for j in 0..N {
                let p = a[i] as i128 * b[j] as i128;
                if i + j < N {
                    c[i + j] += p
                } else {
                    c[i + j - N] -= p
                }
            }
        }
        c.iter().map(|x| x.rem_euclid(Q as i128) as u32).collect()
    }

    #[test]
    fn field_matches_naive() {
        let f = Field::new(Q);
        let mut s = 7;
        for _ in 0..10_000 {
            let (a, b) = (lcg(&mut s), lcg(&mut s));
            assert_eq!(f.add(a, b) as u64, (a as u64 + b as u64) % Q as u64);
            assert_eq!(
                f.sub(a, b) as u64,
                (a as u64 + Q as u64 - b as u64) % Q as u64
            );
            assert_eq!(f.mul(a, b) as u64, a as u64 * b as u64 % Q as u64);
        }
        assert_eq!(f.mul(Q - 1, Q - 1), 1);
    }

    #[test]
    fn compress_matches_rounding() {
        let f = Field::new(Q);
        let mut s = 3;
        for d in [1, 5, 10, 11] {
            for _ in 0..5_000 {
                let x = lcg(&mut s);
                let want =
                    ((((x as u128) << d) + Q as u128 / 2) / Q as u128) as u32 & ((1 << d) - 1);
                assert_eq!(f.compress(x, d), want);
                let y = x & ((1 << d) - 1);
                let back = ((y as u128 * Q as u128 + (1 << d >> 1)) >> d) as u32;
                assert_eq!(f.decompress(y, d), back);
            }
            for x in [0, 1, Q / 2, Q / 2 + 1, Q - 1] {
                let want =
                    ((((x as u128) << d) + Q as u128 / 2) / Q as u128) as u32 & ((1 << d) - 1);
                assert_eq!(f.compress(x, d), want, "x={x} d={d}");
            }
        }
        // dropping 13 bits leaves an error in (−2^12, 2^12]
        for x in [0, 4095, 4096, 4097, Q - 1] {
            let e = x as i64 - ((f.power2round(x, 13) as i64) << 13);
            assert!(-(1 << 12) < e && e <= 1 << 12, "x={x} e={e}");
        }
    }

    #[test]
    fn zeta_is_a_primitive_512th_root() {
        assert_eq!(pow(ZETA, 256, Q), Q - 1);
        assert_eq!(pow(ZETA, 512, Q), 1);
    }

    #[test]
    fn ntt_round_trips() {
        let mut g = ModGraph::new(Q);
        let a = g.input(&[N], Label::Secret);
        let fa = ntt(&mut g, a, ZETA);
        let back = intt(&mut g, fa, ZETA);
        let mut s = 1;
        let x: Vec<u32> = (0..N).map(|_| lcg(&mut s)).collect();
        assert_eq!(g.eval(std::slice::from_ref(&x), &[back])[0], x);
    }

    #[test]
    fn ntt_multiplies_negacyclically() {
        let mut g = ModGraph::new(Q);
        let (a, b) = (g.input(&[N], Label::Secret), g.input(&[N], Label::Public));
        let (fa, fb) = (ntt(&mut g, a, ZETA), ntt(&mut g, b, ZETA));
        let prod = g.map(&[N], &[(fa, ModGraph::id(1)), (fb, ModGraph::id(1))], |b| {
            let (x, y) = (b.arg(0), b.arg(1));
            b.mul(x, y)
        });
        let c = intt(&mut g, prod, ZETA);
        let mut s = 42;
        let x: Vec<u32> = (0..N).map(|_| lcg(&mut s)).collect();
        let y: Vec<u32> = (0..N).map(|_| lcg(&mut s)).collect();
        assert_eq!(g.eval(&[x.clone(), y.clone()], &[c])[0], schoolbook(&x, &y));
        // the product of a secret is secret, and the NTT itself needed no declassify
        assert_eq!(g.check(&[(c, Label::Secret)]), Ok(vec![]));
        assert_eq!(
            g.check(&[(c, Label::Public)]),
            Err(TaintError::SecretOutput(c))
        );
    }

    #[test]
    fn x_to_the_n_is_minus_one() {
        // x^255 · x = x^256 = −1 in the ring
        let mut a = vec![0; N];
        let mut b = vec![0; N];
        a[255] = 1;
        b[1] = 1;
        let mut want = vec![0; N];
        want[0] = Q - 1;
        assert_eq!(schoolbook(&a, &b), want);
        let mut g = ModGraph::new(Q);
        let (x, y) = (g.input(&[N], Label::Public), g.input(&[N], Label::Public));
        let (fx, fy) = (ntt(&mut g, x, ZETA), ntt(&mut g, y, ZETA));
        let p = g.map(&[N], &[(fx, ModGraph::id(1)), (fy, ModGraph::id(1))], |b| {
            let (u, v) = (b.arg(0), b.arg(1));
            b.mul(u, v)
        });
        let c = intt(&mut g, p, ZETA);
        assert_eq!(g.eval(&[a, b], &[c])[0], want);
    }

    #[test]
    fn secret_index_is_refused() {
        let mut g = ModGraph::new(Q);
        let table = g.constant(&[10, 20, 30, 40]);
        let s = g.input(&[2], Label::Secret);
        let r = g.gather(table, s);
        assert_eq!(
            g.check(&[(r, Label::Secret)]),
            Err(TaintError::SecretIndex(r))
        );

        let mut g = ModGraph::new(Q);
        let table = g.constant(&[10, 20, 30, 40]);
        let p = g.input(&[2], Label::Public);
        let r = g.gather(table, p);
        assert_eq!(g.check(&[(r, Label::Public)]), Ok(vec![]));
        assert_eq!(g.eval(&[vec![3, 1]], &[r])[0], vec![40, 20]);
    }

    #[test]
    fn rejection_sampling_is_public_only() {
        // expanding the public matrix A: reject candidates ≥ q (here from a seed)
        let mut g = ModGraph::new(Q);
        let seed = g.input(&[4], Label::Public);
        let a = g.filter(seed, 100);
        assert_eq!(g.check(&[(a, Label::Public)]), Ok(vec![]));
        assert_eq!(g.eval(&[vec![5, 200, 7, 99]], &[a])[0], vec![5, 7, 99, 0]);

        let mut g = ModGraph::new(Q);
        let s = g.input(&[4], Label::Secret);
        let bad = g.filter(s, 100);
        assert_eq!(
            g.check(&[(bad, Label::Secret)]),
            Err(TaintError::SecretFilter(bad))
        );
    }

    #[test]
    fn declassify_names_its_proof() {
        // the signature's attempt count: secret-derived, provably independent of the key
        let mut g = ModGraph::new(Q);
        let s = g.input(&[1], Label::Secret);
        let attempts = g.declassify(s, "fswa-abort-count-independent");
        let next = g.filter(attempts, 5);
        assert_eq!(
            g.check(&[(next, Label::Public)]),
            Ok(vec!["fswa-abort-count-independent"])
        );
    }
}
