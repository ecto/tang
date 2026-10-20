//! Learned probabilities for TCG3 code-blocks, in **32-bit integer** arithmetic
//! end to end, so a WGSL kernel (no `i64`) reproduces every probability.
//!
//! - Features: the TCG3-visible context (`Feats::raw`: block-local causal
//!   neighbours, coarser-level parent, residual magnitudes) plus the class
//!   one-hot, as signed `Q8` log-magnitudes (integer `lg_q8`).
//! - Network: M1b's MLP (`learned::Net`, ReLU, Q8 weights, Q16 biases),
//!   evaluated with `i32` accumulators: weights clamped to `±2047`
//!   (±8.0), activations to `[0, 2^14]` (hidden) / `±2^14` (outputs), biases
//!   to `±2^24`, so no partial sum can overflow (`28·2^11·2^14 + 2^24 < 2^31`).
//! - Output: a 2-component discretized logistic mixture (π logit, μ₁, μ₂,
//!   log₂ s₁, log₂ s₂, all Q8). CDF in Q16 from two integer tables (σ in
//!   Q16 over `|z| ≤ 16`, `2^{f/256}` in Q16), built on the CPU with basic IEEE
//!   operations only and uploaded to the GPU as data.
//! - Binarization (as M1b, without the APM): zero flag, sign, exp-Golomb
//!   bucket by unary tail ratios, mantissa by integer bisection; each
//!   decision's `P(1)` is an integer ratio of Q16 CDF masses.

use crate::gpuwav::{Feats, Sink};
use crate::learned::{lg_q8, Net, NF, NO};
use std::sync::OnceLock;
use tang_expr::int::{IId, IType, IntGraph, Program};

/// Features actually used (the rest of the `NF` inputs are zero).
pub const NFL: usize = 28;
/// `|z|` range of the σ table (Q8).
pub const ZM: i32 = 16 * 256;

/// σ table (Q16, `k/256`, `k = 0..=ZM`) and `2^{f/256}` table (Q16).
pub fn tables() -> &'static (Vec<u32>, Vec<u32>) {
    static T: OnceLock<(Vec<u32>, Vec<u32>)> = OnceLock::new();
    T.get_or_init(|| {
        let x = 1.0f64 / 256.0;
        let (mut term, mut em) = (1.0f64, 1.0f64);
        for k in 1..20 {
            term = term * (-x) / k as f64;
            em += term;
        }
        let mut sig = Vec::with_capacity(ZM as usize + 1);
        let mut p = 1.0f64;
        for _ in 0..=ZM {
            sig.push((65536.0 / (1.0 + p)).round() as u32);
            p *= em;
        }
        let mut r = 2.0f64;
        for _ in 0..8 {
            r = r.sqrt();
        }
        let mut pw = Vec::with_capacity(257);
        let mut v = 1.0f64;
        for _ in 0..=256 {
            pw.push((v * 65536.0).round() as u32);
            v *= r;
        }
        (sig, pw)
    })
}

#[inline]
fn slg(v: i32) -> i32 {
    if v < 0 {
        -lg_q8(v.unsigned_abs())
    } else {
        lg_q8(v as u32)
    }
}

/// Integer feature vector of a coefficient.
pub fn feat_vec3(f: &Feats, class: usize) -> [i32; NF] {
    let r = &f.raw;
    let mut x = [0i32; NF];
    for i in 0..7 {
        x[i] = slg(r[i]);
    }
    for i in 7..14 {
        x[i] = lg_q8(r[i] as u32);
    }
    x[14] = r[14] * 64;
    x[15] = r[15] * 32;
    x[16] = r[16] * 32;
    x[17] = r[17] * 256;
    x[18 + class.min(9)] = 256;
    x
}

/// A network prepared for 32-bit inference.
#[derive(Clone, Debug)]
pub struct Net32 {
    /// Layer sizes (input = [`NFL`]).
    pub sizes: Vec<usize>,
    /// Clamped Q8 weights per layer (out × in).
    pub w: Vec<Vec<i32>>,
    /// Clamped Q16 biases.
    pub b: Vec<Vec<i32>>,
}

impl Net32 {
    /// From an M1b network (its first `NFL` inputs).
    pub fn new(net: &Net) -> Self {
        assert_eq!(net.sizes[0], NF);
        assert_eq!(*net.sizes.last().unwrap(), NO);
        let mut sizes = net.sizes.clone();
        sizes[0] = NFL;
        let w = (0..sizes.len() - 1)
            .map(|l| {
                let (ni_full, ni, no) = (net.sizes[l], sizes[l], sizes[l + 1]);
                (0..no).flat_map(|o| (0..ni).map(move |i| (o, i))).map(|(o, i)| net.wq[l][o * ni_full + i].clamp(-2047, 2047)).collect()
            })
            .collect();
        let b = net.bq.iter().map(|v| v.iter().map(|&x| x.clamp(-(1 << 24), 1 << 24)).collect()).collect();
        Self { sizes, w, b }
    }

    /// Integer forward pass.
    pub fn infer(&self, x: &[i32; NF]) -> [i32; NO] {
        let mut a = [0i32; 64];
        let mut nb = [0i32; 64];
        for i in 0..NFL {
            a[i] = x[i].clamp(-(1 << 14), 1 << 14);
        }
        let nl = self.sizes.len() - 1;
        for l in 0..nl {
            let (ni, no) = (self.sizes[l], self.sizes[l + 1]);
            for o in 0..no {
                let mut acc = self.b[l][o];
                for i in 0..ni {
                    acc = acc.wrapping_add(self.w[l][o * ni + i].wrapping_mul(a[i]));
                }
                let v = acc.wrapping_add(128) >> 8;
                nb[o] = if l + 1 < nl { v.clamp(0, 1 << 14) } else { v.clamp(-(1 << 14), 1 << 14) };
            }
            std::mem::swap(&mut a, &mut nb);
        }
        [a[0], a[1], a[2], a[3], a[4]]
    }

    /// The forward pass as a `tang-expr` integer graph: params are the
    /// [`NFL`] clamped inputs, outputs the [`NO`] mixture parameters. Zero
    /// weights are dropped (integer addition is associative, so no bit
    /// changes). This one graph is lowered to the GPU kernel's MLP and
    /// evaluated as the CPU reference ([`Net32::program`]).
    pub fn graph(&self) -> (IntGraph, Vec<IId>) {
        let mut g = IntGraph::new();
        let mut a: Vec<IId> = (0..NFL).map(|i| g.param(i as u16, IType::I32)).collect();
        let nl = self.sizes.len() - 1;
        let (r, sh) = (g.i32(128), g.u32(8));
        for l in 0..nl {
            let (ni, no) = (self.sizes[l], self.sizes[l + 1]);
            let (lo, hi) = if l + 1 < nl { (g.i32(0), g.i32(1 << 14)) } else { (g.i32(-(1 << 14)), g.i32(1 << 14)) };
            a = (0..no)
                .map(|o| {
                    let mut acc = g.i32(self.b[l][o]);
                    for (i, &ai) in a.iter().enumerate().take(ni) {
                        let w = self.w[l][o * ni + i];
                        if w != 0 {
                            let w = g.i32(w);
                            let t = g.mul(w, ai);
                            acc = g.add(acc, t);
                        }
                    }
                    let v = g.add(acc, r);
                    let v = g.shr(v, sh);
                    g.clamp(v, lo, hi)
                })
                .collect();
        }
        (g, a)
    }

    /// [`Net32::graph`] compiled for [`Net32::infer_ir`].
    pub fn program(&self) -> Program {
        let (g, o) = self.graph();
        Program::new(&g, &o)
    }

    /// [`Net32::infer`] through the integer-graph interpreter (bit-identical).
    pub fn infer_ir(&self, p: &mut Program, x: &[i32; NF]) -> [i32; NO] {
        let mut inp = [0u32; NFL];
        for i in 0..NFL {
            inp[i] = x[i].clamp(-(1 << 14), 1 << 14) as u32;
        }
        let mut o = [0u32; NO];
        p.run(&inp, &[], &mut o);
        o.map(|v| v as i32)
    }

    /// Flattened `[nl, sizes…, w₀, b₀, w₁, b₁, …]` for the GPU.
    pub fn flat(&self) -> Vec<u32> {
        let mut v = vec![(self.sizes.len() - 1) as u32];
        v.extend(self.sizes.iter().map(|&s| s as u32));
        for l in 0..self.sizes.len() - 1 {
            v.extend(self.w[l].iter().map(|&x| x as u32));
            v.extend(self.b[l].iter().map(|&x| x as u32));
        }
        v
    }
}

/// σ(z), z in Q8 → Q16.
#[inline]
pub fn sig16(z: i32) -> u32 {
    let (s, _) = tables();
    let a = z.unsigned_abs().min(ZM as u32) as usize;
    if z >= 0 {
        s[a]
    } else {
        65536 - s[a]
    }
}

/// `d · 2^{−ls/256}` (both Q8; result Q8) saturated to `±(ZM+1)`, with
/// `ls` clamped to `[−16·256, 24·256]`: `t = ⌊d·m/2^16⌉` with the Q16
/// mantissa `m = 2^{(256−f)/256}`, then an arithmetic or saturating shift.
#[inline]
pub fn zval(d: i32, ls: i32) -> i32 {
    let (_, pw) = tables();
    let ls = ls.clamp(-16 * 256, 24 * 256);
    let i = ls.div_euclid(256);
    let f = ls.rem_euclid(256);
    let m = pw[(256 - f) as usize] as i32;
    let t = crate::gpuwav::mulsh(d.clamp(-(1 << 25), 1 << 25), m, 16);
    let sh = i + 1;
    if sh >= 0 {
        t >> sh.min(31)
    } else {
        let k = -sh;
        let lim = (ZM + 1) >> k;
        if t > lim {
            ZM + 1
        } else if t < -lim {
            -(ZM + 1)
        } else {
            t << k
        }
    }
}

/// Mixture parameters.
#[derive(Clone, Copy, Debug)]
pub struct Mix3 {
    pi15: u32,
    mu: [i32; 2],
    ls: [i32; 2],
}

impl Mix3 {
    /// From network outputs.
    pub fn new(o: &[i32; NO]) -> Self {
        Self { pi15: sig16(o[0]) >> 1, mu: [o[1], o[2]], ls: [o[3], o[4]] }
    }

    /// `P(r ≤ m)` in Q16.
    #[inline]
    pub fn cdf(&self, m: i32) -> u32 {
        let t = m.clamp(-65535, 65535) * 256 + 128;
        let c = |k: usize| sig16(zval(t - self.mu[k], self.ls[k]));
        (self.pi15 * c(0) + (32768 - self.pi15) * c(1)) >> 15
    }
}

/// `P(1)·2^16` from masses (`num ≤ den ≤ 2^16`), clamped for rANS.
#[inline]
pub fn ratio16(num: u32, den: u32) -> u32 {
    let p = (num.min(den) * 65535).checked_div(den).unwrap_or(32768);
    p.clamp(32, 65504)
}

/// Inverse of [`sig16`] on a `p >> 4` grid: the smallest Q8 `z` with
/// `sig16(z) ≥ p` (`p = 16·i + 8`), so `stretch(p) ∈ [−ZM, ZM]`.
pub fn stretch_tab() -> &'static Vec<i32> {
    static T: OnceLock<Vec<i32>> = OnceLock::new();
    T.get_or_init(|| {
        let mut z = -ZM;
        (0..4096u32)
            .map(|i| {
                let p = 16 * i + 8;
                while z < ZM && sig16(z) < p {
                    z += 1;
                }
                z
            })
            .collect()
    })
}

/// Adaptive-layer settings (stream header).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdCfg {
    /// APM weight in sixteenths (0 = off).
    pub wa: u32,
    /// Count limit of the APM bins (rate floor `2^-⌊log₂(lim+2)⌋`).
    pub lim: u32,
    /// Decision mask: bit 0 zero flag, 1 sign, 2 bucket, 3 mantissa.
    pub mask: u32,
}

/// APM bins per context (step 1.0 in the logistic domain, `±8`).
pub const AB: usize = 17;
/// APM contexts.
pub const AC: usize = 8;

/// A lane's fresh adaptive state as the kernels hold it: `(p << 8) | count`
/// per bin (count 0), the layout of the adaptive residual graph's `adt`.
pub fn ad_init() -> [u32; AC * AB] {
    std::array::from_fn(|i| sig16(((i % AB) as i32 - 8) * 256) << 8)
}

/// Per-lane adaptive state over the MLP output: an adaptive probability map
/// (secondary estimation) per decision type, interpolated over [`AB`]
/// stretch-domain bins, count-based power-of-two rate. Integer only; reset per block.
#[derive(Clone, Debug)]
pub struct Adapt {
    /// Settings.
    pub c: AdCfg,
    t: [u32; AC * AB],
    n: [u8; AC * AB],
}

impl Adapt {
    /// Fresh state.
    pub fn new(c: AdCfg) -> Self {
        let t = std::array::from_fn(|i| sig16(((i % AB) as i32 - 8) * 256));
        Self { c, t, n: [0; AC * AB] }
    }

    /// Code `bit` with model `P(1) = p1` on context `ctx`.
    #[inline]
    fn bit<S: Sink>(&mut self, s: &mut S, lane: usize, ctx: usize, bit: bool, p1: u32) -> bool {
        let grp = match ctx {
            0 | 1 => ctx,
            2..=5 => 2,
            _ => 3,
        };
        if self.c.wa == 0 || (self.c.mask >> grp) & 1 == 0 {
            return s.bit(lane, bit, p1);
        }
        let st = stretch_tab()[(p1 >> 4) as usize].clamp(-2048, 2047) + 2048;
        let (j, w) = ((st >> 8) as usize, (st & 255) as u32);
        let b = ctx * AB + j;
        let pr = (self.t[b] * (256 - w) + self.t[b + 1] * w) >> 8;
        let pf = (pr * self.c.wa + p1 * (16 - self.c.wa)) >> 4;
        let got = s.bit(lane, bit, pf.clamp(32, 65504));
        let k = if w < 128 { b } else { b + 1 };
        let tgt = if got { 65535 } else { 0 };
        let d = tgt - self.t[k] as i32;
        // rate 2^-⌊log₂(n+2)⌋ (≈ 1/(n+1.5) within 2×, no divide on the GPU)
        self.t[k] = (self.t[k] as i32 + (d >> (31 - (self.n[k] as u32 + 2).leading_zeros()))) as u32;
        if (self.n[k] as u32) < self.c.lim {
            self.n[k] += 1;
        }
        got
    }
}

/// Code residual `r` under `mx` on `lane`.
pub fn code_learned<S: Sink>(s: &mut S, lane: usize, mx: &Mix3, r: i32) -> i32 {
    let mut ad = Adapt::new(AdCfg { wa: 0, lim: 0, mask: 0 });
    code_learned_ad(s, lane, mx, r, &mut ad)
}

/// [`code_learned`] through the lane's adaptive layer.
pub fn code_learned_ad<S: Sink>(s: &mut S, lane: usize, mx: &Mix3, r: i32, ad: &mut Adapt) -> i32 {
    let c0 = mx.cdf(0);
    let cm1 = mx.cdf(-1);
    let pz = c0.saturating_sub(cm1);
    if !ad.bit(s, lane, 0, r != 0, ratio16(65536 - pz, 65536)) {
        return 0;
    }
    let negm = cm1;
    let posm = 65536 - c0;
    let neg = ad.bit(s, lane, 1, r < 0, ratio16(negm, negm + posm));
    let tail = |m: u32| -> u32 {
        if m > 1 << 30 {
            return 0;
        }
        if neg {
            mx.cdf(-(m as i32))
        } else {
            65536 - mx.cdf(m as i32 - 1)
        }
    };
    let mag = r.unsigned_abs();
    let mut k = 0u32;
    while k < 30 {
        let a = tail(1 << k);
        let bb = tail(1 << (k + 1));
        if !ad.bit(s, lane, 2 + k.min(3) as usize, mag >= 1 << (k + 1), ratio16(bb, a)) {
            break;
        }
        k += 1;
    }
    let (mut lo, mut hi) = (1u32 << k, ((1u64 << (k + 1)) - 1) as u32);
    let mut first = true;
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        let end = if hi == u32::MAX { 0 } else { tail(hi + 1) };
        let up = tail(mid).saturating_sub(end);
        let all = tail(lo).saturating_sub(end);
        let ctx = if first { 6 } else { 7 };
        first = false;
        if ad.bit(s, lane, ctx, mag >= mid, ratio16(up, all)) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    if neg {
        (lo as i32).wrapping_neg()
    } else {
        lo as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpuwav::{rans_encode, DecSink, RansDec, RecSink};

    #[test]
    fn ir_graph_matches_infer() {
        let mut st = 7u64;
        let mut rnd = || {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (st >> 33) as i32
        };
        for sizes in [vec![NFL, 16, NO], vec![NFL, 32, 16, NO]] {
            let w = (0..sizes.len() - 1)
                .map(|l| (0..sizes[l] * sizes[l + 1]).map(|k| if k % 5 == 0 { 0 } else { rnd() % 4095 - 2047 }).collect())
                .collect();
            let b = (1..sizes.len()).map(|l| (0..sizes[l]).map(|_| rnd() % (1 << 25) - (1 << 24)).collect()).collect();
            let net = Net32 { sizes, w, b };
            let mut p = net.program();
            for k in 0..5000 {
                let mut x = [0i32; NF];
                for v in x.iter_mut() {
                    *v = match k % 3 {
                        0 => rnd() % 40000 - 20000,
                        1 => rnd() % 512,
                        _ => [16384, -16384, i32::MAX, i32::MIN][(rnd() & 3) as usize],
                    };
                }
                assert_eq!(net.infer_ir(&mut p, &x), net.infer(&x));
            }
        }
    }

    #[test]
    fn learned_round_trip() {
        let mut st = 99u64;
        let mut rnd = || {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (st >> 33) as i32
        };
        let cases: Vec<(Mix3, i32)> = (0..20000)
            .map(|i| {
                let o = [rnd() % 2000, rnd() % 3000, rnd() % 600, rnd() % 4000 - 600, rnd() % 2500];
                let r = match i % 5 {
                    0 => 0,
                    1 => rnd() % 7 - 3,
                    2 => rnd() % 300 - 150,
                    3 => rnd() % 200000 - 100000,
                    _ => [i32::MAX, i32::MIN + 1, 65536, -65537][(i / 5) % 4],
                };
                (Mix3::new(&o), r)
            })
            .collect();
        let mut s = RecSink { syms: vec![Vec::new()] };
        for (m, r) in &cases {
            assert_eq!(code_learned(&mut s, 0, m, *r), *r);
        }
        let w = rans_encode(&s.syms[0]);
        let mut d = DecSink { lanes: vec![RansDec::new(&w)] };
        for (m, r) in &cases {
            assert_eq!(code_learned(&mut d, 0, m, 0), *r);
        }
    }
}
