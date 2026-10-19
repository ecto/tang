//! Context-modelled coding of quantized wavelet coefficients.
//!
//! Every coefficient `q` is binarized as: significance `|q| > 0`, then
//! `|q| > k` for `k = 1 … KB`, an adaptive Elias-gamma escape for
//! `|q| − KB − 1`, and a sign. Each binary decision is coded by the integer
//! range coder in [`crate::abac`] with a probability supplied by a
//! [`CoefModel`]:
//!
//! - [`HandModel`]: classic hand-designed contexts — subband class (level ×
//!   orientation), causal neighbour activity in the same band, and the parent
//!   coefficient's neighbourhood one level up — each indexing a two-rate
//!   adaptive [`BitModel`];
//! - the learned models in [`crate::learned`], which predict the bins from the
//!   same [`Feat`] vector with a fixed-point network.
//!
//! Encoder and decoder run the *same* traversal ([`code_bands`]) through the
//! [`Bits`] trait, so the context computation cannot drift between them.

use crate::abac::{BitModel, Decoder, Encoder, UintModel};
use crate::wavelet::{Band, Pyramid};

/// Unary magnitude bins after significance (`|q| > 1 … |q| > KB`).
pub const KB: usize = 14;
/// Neighbour magnitudes are capped at this before use.
pub const CAP: u16 = 65535;
/// Number of subband classes: 3 level groups × 3 orientation counts + LL.
pub const NCLASS: usize = 10;

/// Symmetric encode/decode of single decisions.
pub trait Bits {
    /// Code `bit` (ignored when decoding) with an adaptive model.
    fn bit(&mut self, bit: bool, m: &mut BitModel) -> bool;
    /// Code `bit` with a fixed probability `P(1) = p1/2^16`.
    fn bitp(&mut self, bit: bool, p1: u32) -> bool;
    /// Code an unbounded non-negative integer.
    fn uint(&mut self, v: u64, m: &mut UintModel) -> u64;
    /// Accounting tag for the decisions that follow (no-op for real coders).
    #[inline]
    fn tag(&mut self, _t: usize) {}
}

/// Cost meter: models adapt exactly as in the real coder, ideal code
/// lengths are tallied per [`Bits::tag`].
#[derive(Default, Debug)]
pub struct Meter {
    /// Bits per tag.
    pub bits: [f64; 8],
    cur: usize,
}

impl Bits for Meter {
    fn bit(&mut self, bit: bool, m: &mut BitModel) -> bool {
        self.bits[self.cur] += crate::abac::cost_bits(bit, m.p1());
        m.update(bit);
        bit
    }
    fn bitp(&mut self, bit: bool, p1: u32) -> bool {
        self.bits[self.cur] += crate::abac::cost_bits(bit, p1);
        bit
    }
    fn uint(&mut self, v: u64, m: &mut UintModel) -> u64 {
        let mut e = Encoder::new();
        m.encode(&mut e, v);
        self.bits[self.cur] += 8.0 * e.finish().len() as f64 - 32.0;
        v
    }
    fn tag(&mut self, t: usize) {
        self.cur = t;
    }
}

impl Bits for Encoder {
    #[inline]
    fn bit(&mut self, bit: bool, m: &mut BitModel) -> bool {
        self.encode(bit, m);
        bit
    }
    #[inline]
    fn bitp(&mut self, bit: bool, p1: u32) -> bool {
        self.encode_p(bit, p1);
        bit
    }
    fn uint(&mut self, v: u64, m: &mut UintModel) -> u64 {
        m.encode(self, v);
        v
    }
}

impl Bits for Decoder<'_> {
    #[inline]
    fn bit(&mut self, _: bool, m: &mut BitModel) -> bool {
        self.decode(m)
    }
    #[inline]
    fn bitp(&mut self, _: bool, p1: u32) -> bool {
        self.decode_p(p1)
    }
    fn uint(&mut self, _: u64, m: &mut UintModel) -> u64 {
        m.decode(self)
    }
}

/// Context features of one coefficient — all causal (already coded).
#[derive(Clone, Copy, Debug, Default)]
pub struct Feat {
    /// Capped magnitudes of causal neighbours: W, N, U, NW, NE, UW, UN, WW
    /// (axis 2 = W/E, axis 1 = N, axis 0 = U), 0 outside the band.
    pub nb: [u16; 8],
    /// Signs of W, N, U (−1, 0, +1).
    pub sg: [i8; 3],
    /// Capped parent magnitude (0 if no parent band).
    pub par: u16,
    /// Capped sum of the parent's 6 face neighbours.
    pub pnb: u16,
    /// Subband class `0..NCLASS`.
    pub class: u8,
    /// Highpass mask.
    pub mask: u8,
    /// Whether a parent band exists.
    pub has_par: bool,
    /// Capped magnitude of the in-band prediction (0 when prediction is off).
    pub pm: u16,
}

/// Subband class of a band.
pub fn band_class(b: &Band) -> u8 {
    if b.mask == 0 {
        return (NCLASS - 1) as u8;
    }
    let nh = b.mask.count_ones() as usize;
    (b.level.min(2) * 3 + nh - 1) as u8
}

/// A probability model for one quantized coefficient.
pub trait CoefModel {
    /// Code (or decode) the coefficient `v` with context `f`; returns it.
    fn code<B: Bits>(&mut self, b: &mut B, f: &Feat, v: i32) -> i32;
}

fn bucket(a: u32, th: &[u32]) -> usize {
    th.iter().position(|&t| a <= t).unwrap_or(th.len())
}

const A_TH: [u32; 11] = [0, 1, 2, 3, 5, 7, 10, 14, 20, 30, 48];
const NA: usize = A_TH.len() + 1;
const P_TH: [u32; 4] = [0, 2, 6, 16];
const NP: usize = P_TH.len() + 1;
/// Magnitude-scale classes (see [`mag_scale`]).
pub const NE: usize = 18;

/// Expected-magnitude class `⌊log₂(1 + 2μ)⌋` from the weighted mean `μ` of
/// the causal neighbours' and parent's magnitudes (an adaptive-Golomb
/// parameter: the low `e − 2` bits of `|q| − 1` are coded near-raw).
#[inline]
pub fn mag_scale(f: &Feat) -> usize {
    let n = f.nb.map(|v| v as u64);
    let mut s = 2 * (n[0] + n[1] + n[2]) + n[3] + n[4] + n[5] + n[6];
    let mut w = 10;
    s += 2 * f.pm as u64;
    if f.has_par {
        s += 2 * f.par as u64;
        w += 2;
    }
    let v = 1 + 2 * s / w;
    (v.ilog2() as usize).min(NE - 1)
}

/// Hand-designed context model (adaptive bit models per context).
#[derive(Clone, Debug)]
pub struct HandModel {
    sig: Vec<BitModel>,
    mag: Vec<BitModel>,
    low: Vec<BitModel>,
    esc: Vec<UintModel>,
    sign: Vec<BitModel>,
}

impl Default for HandModel {
    fn default() -> Self {
        Self {
            sig: vec![BitModel::default(); NCLASS * NA * NP],
            mag: vec![BitModel::default(); NCLASS * NE * KB],
            low: vec![BitModel::default(); NE * 4],
            esc: vec![UintModel::default(); NCLASS],
            sign: vec![BitModel::default(); NCLASS * 27],
        }
    }
}

/// Neighbour activity used by the hand model.
#[inline]
pub fn activity(f: &Feat) -> u32 {
    let n = f.nb.map(|v| v.min(15) as u32);
    2 * (n[0] + n[1] + n[2]) + n[3] + n[4] + n[5] + n[6] + n[7]
}

impl CoefModel for HandModel {
    fn code<B: Bits>(&mut self, b: &mut B, f: &Feat, v: i32) -> i32 {
        let c = f.class as usize;
        let a = activity(f) + f.pm.min(15) as u32;
        let pa = 4 * f.par.min(15) as u32 + f.pnb.min(60) as u32;
        let ab = bucket(a, &A_TH);
        let pb = if f.has_par { 1 + bucket(pa, &P_TH[..NP - 2]) } else { 0 };
        let mag = v.unsigned_abs();
        b.tag(0);
        if !b.bit(mag > 0, &mut self.sig[(c * NA + ab) * NP + pb]) {
            return 0;
        }
        b.tag(1);
        let e = mag_scale(f);
        let k = e.saturating_sub(2) as u32;
        let r = mag - 1;
        let hi = r >> k;
        let mut h = 0u32;
        while (h as usize) < KB {
            if !b.bit(hi > h, &mut self.mag[(c * NE + e) * KB + h as usize]) {
                break;
            }
            h += 1;
        }
        if h as usize == KB {
            h += b.uint((hi - h) as u64, &mut self.esc[c]) as u32;
        }
        let mut lo = 0u32;
        b.tag(2);
        for j in (0..k).rev() {
            let bit = (r >> j) & 1 == 1;
            let pos = (k - 1 - j) as usize;
            let got = if pos < 2 {
                b.bit(bit, &mut self.low[(e * 2 + pos) * 2 + (pos == 1 && lo & 1 == 1) as usize])
            } else {
                b.bitp(bit, crate::abac::ONE / 2)
            };
            lo = (lo << 1) | got as u32;
        }
        let m = 1 + ((h << k) | lo);
        b.tag(3);
        let sctx = c * 27 + ((f.sg[0] + 1) * 9 + (f.sg[1] + 1) * 3 + (f.sg[2] + 1)) as usize;
        let neg = b.bit(v < 0, &mut self.sign[sctx]);
        if neg {
            -(m as i32)
        } else {
            m as i32
        }
    }
}

/// Gather the features of position `(z, y, x)` in band `bi`.
#[inline]
pub fn features(p: &Pyramid, bi: usize, pbi: Option<usize>, q: &[i32], z: usize, y: usize, x: usize) -> Feat {
    let b = &p.bands[bi];
    let [_, ny, nx] = p.dims;
    let at = |z: usize, y: usize, x: usize| q[(z * ny + y) * nx + x];
    let mag = |v: i32| (v.unsigned_abs().min(CAP as u32)) as u16;
    let sgn = |v: i32| v.signum() as i8;
    let (z0, y0, x0) = (b.start[0], b.start[1], b.start[2]);
    let x1 = b.end[2];
    let mut f = Feat { class: band_class(b), mask: b.mask, ..Default::default() };
    if x > x0 {
        let w = at(z, y, x - 1);
        f.nb[0] = mag(w);
        f.sg[0] = sgn(w);
        if x > x0 + 1 {
            f.nb[7] = mag(at(z, y, x - 2));
        }
    }
    if y > y0 {
        let n = at(z, y - 1, x);
        f.nb[1] = mag(n);
        f.sg[1] = sgn(n);
        if x > x0 {
            f.nb[3] = mag(at(z, y - 1, x - 1));
        }
        if x + 1 < x1 {
            f.nb[4] = mag(at(z, y - 1, x + 1));
        }
    }
    if z > z0 {
        let u = at(z - 1, y, x);
        f.nb[2] = mag(u);
        f.sg[2] = sgn(u);
        if x > x0 {
            f.nb[5] = mag(at(z - 1, y, x - 1));
        }
        if y > y0 {
            f.nb[6] = mag(at(z - 1, y - 1, x));
        }
    }
    if let Some(pi) = pbi {
        let pb = &p.bands[pi];
        let pp = p.parent_pos(b, pb, [z, y, x]);
        f.has_par = true;
        f.par = mag(at(pp[0], pp[1], pp[2]));
        let mut s = 0u32;
        for a in 0..3 {
            let mut lo = pp;
            if lo[a] > pb.start[a] {
                lo[a] -= 1;
                s += at(lo[0], lo[1], lo[2]).unsigned_abs().min(CAP as u32);
            }
            let mut hi = pp;
            if hi[a] + 1 < pb.end[a] {
                hi[a] += 1;
                s += at(hi[0], hi[1], hi[2]).unsigned_abs().min(CAP as u32);
            }
        }
        f.pnb = s.min(CAP as u32) as u16;
    }
    f
}

/// Number of in-band linear-prediction taps: W, N, U, NW, UW, UN, parent.
pub const NTAP: usize = 7;
/// Fixed-point shift of predictor weights (Q12).
pub const PSHIFT: u32 = 12;

/// Per-band causal linear predictor of the quantized value (integer
/// weights, Q12). Smooth fields leave highpass bands smooth along their
/// lowpass axes (neighbour correlation ≈ 0.9) and anticorrelated (≈ −0.5)
/// along the highpass axis — structure significance coders such as SPECK
/// leave on the table. The coder codes `q − pred`; contexts are computed on
/// the residuals.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Pred {
    /// Whether prediction is on for the band.
    pub on: bool,
    /// Tap weights, Q12.
    pub w: [i16; NTAP],
}

#[inline]
fn taps(p: &Pyramid, bi: usize, pbi: Option<usize>, q: &[i32], z: usize, y: usize, x: usize) -> [i32; NTAP] {
    let b = &p.bands[bi];
    let [_, ny, nx] = p.dims;
    let at = |z: usize, y: usize, x: usize| q[(z * ny + y) * nx + x];
    let (zi, yi, xi) = (z > b.start[0], y > b.start[1], x > b.start[2]);
    let mut t = [0i32; NTAP];
    if xi {
        t[0] = at(z, y, x - 1);
    }
    if yi {
        t[1] = at(z, y - 1, x);
        if xi {
            t[3] = at(z, y - 1, x - 1);
        }
    }
    if zi {
        t[2] = at(z - 1, y, x);
        if xi {
            t[4] = at(z - 1, y, x - 1);
        }
        if yi {
            t[5] = at(z - 1, y - 1, x);
        }
    }
    if let Some(pi) = pbi {
        let pb = &p.bands[pi];
        let pp = p.parent_pos(b, pb, [z, y, x]);
        t[6] = at(pp[0], pp[1], pp[2]);
    }
    t
}

#[inline]
fn predict(w: &[i16; NTAP], t: &[i32; NTAP]) -> i32 {
    let s: i64 = w.iter().zip(t).map(|(&a, &b)| a as i64 * b as i64).sum();
    ((s + (1 << (PSHIFT - 1))) >> PSHIFT).clamp(-(1 << 30), 1 << 30) as i32
}

/// Fit per-band predictors by least squares on the quantized values the
/// decoder will see, and keep a band's predictor only when it lowers the
/// `Σ log₂(1+|r|)` cost proxy by > 2%.
pub fn fit_predictors(p: &Pyramid, q: &[i32]) -> Vec<Pred> {
    let [_, ny, nx] = p.dims;
    (0..p.bands.len())
        .map(|bi| {
            let band = &p.bands[bi];
            let pbi = p.parent_band(bi);
            let mut g = [[0.0f64; NTAP]; NTAP];
            let mut h = [0.0f64; NTAP];
            let mut n = 0usize;
            for z in band.start[0]..band.end[0] {
                for y in band.start[1]..band.end[1] {
                    for x in band.start[2]..band.end[2] {
                        let t = taps(p, bi, pbi, q, z, y, x);
                        let v = q[(z * ny + y) * nx + x] as f64;
                        if v == 0.0 && t.iter().all(|&a| a == 0) {
                            continue;
                        }
                        n += 1;
                        for i in 0..NTAP {
                            h[i] += t[i] as f64 * v;
                            for j in 0..NTAP {
                                g[i][j] += t[i] as f64 * t[j] as f64;
                            }
                        }
                    }
                }
            }
            if n < 32 {
                return Pred::default();
            }
            // ridge + Gaussian elimination
            let mut a = g;
            let mut r = h;
            for i in 0..NTAP {
                a[i][i] += 1e-3 * (g[i][i] + 1.0);
            }
            for c in 0..NTAP {
                let piv = a[c][c];
                if piv.abs() < 1e-12 {
                    continue;
                }
                for rr in 0..NTAP {
                    if rr != c {
                        let f = a[rr][c] / piv;
                        for k in 0..NTAP {
                            a[rr][k] -= f * a[c][k];
                        }
                        r[rr] -= f * r[c];
                    }
                }
            }
            let mut w = [0i16; NTAP];
            for i in 0..NTAP {
                let wi = if a[i][i].abs() > 1e-12 { r[i] / a[i][i] } else { 0.0 };
                w[i] = (wi * (1 << PSHIFT) as f64).round().clamp(-32767.0, 32767.0) as i16;
            }
            let (mut c0, mut c1) = (0.0f64, 0.0f64);
            for z in band.start[0]..band.end[0] {
                for y in band.start[1]..band.end[1] {
                    for x in band.start[2]..band.end[2] {
                        let v = q[(z * ny + y) * nx + x];
                        let pr = predict(&w, &taps(p, bi, pbi, q, z, y, x));
                        c0 += (1.0 + v.unsigned_abs() as f64).log2();
                        c1 += (1.0 + (v - pr).unsigned_abs() as f64).log2();
                    }
                }
            }
            if c1 < 0.98 * c0 {
                Pred { on: true, w }
            } else {
                Pred::default()
            }
        })
        .collect()
}

/// Serialize predictors.
pub fn write_preds(w: &mut crate::bitio::ByteWriter, preds: &[Pred]) {
    for pr in preds {
        w.u8(pr.on as u8);
        if pr.on {
            for &v in &pr.w {
                w.svarint(v as i64);
            }
        }
    }
}

/// Parse predictors for `nbands` bands.
pub fn read_preds(r: &mut crate::bitio::ByteReader, nbands: usize) -> Result<Vec<Pred>, crate::bitio::DecodeError> {
    (0..nbands)
        .map(|_| {
            let mut pr = Pred { on: r.u8()? != 0, ..Default::default() };
            if pr.on {
                for v in pr.w.iter_mut() {
                    *v = r.svarint()?.clamp(-32767, 32767) as i16;
                }
            }
            Ok(pr)
        })
        .collect()
}

/// Traverse every coefficient in coding order. `q` holds values (filled in
/// when decoding), `res` the residuals `q − pred` (the context source).
/// `f(features, band, predicted value, residual) -> residual` codes one
/// coefficient.
fn traverse(p: &Pyramid, q: &mut [i32], res: &mut [i32], preds: &[Pred], mut f: impl FnMut(&Feat, i32, i32) -> i32) {
    let [_, ny, nx] = p.dims;
    for bi in 0..p.bands.len() {
        let pbi = p.parent_band(bi);
        let band = p.bands[bi].clone();
        let pr = preds.get(bi).copied().unwrap_or_default();
        for z in band.start[0]..band.end[0] {
            for y in band.start[1]..band.end[1] {
                for x in band.start[2]..band.end[2] {
                    let i = (z * ny + y) * nx + x;
                    let mut ft = features(p, bi, pbi, res, z, y, x);
                    let pv = if pr.on { predict(&pr.w, &taps(p, bi, pbi, q, z, y, x)) } else { 0 };
                    ft.pm = pv.unsigned_abs().min(CAP as u32) as u16;
                    let r = f(&ft, pv, q[i].wrapping_sub(pv));
                    res[i] = r;
                    q[i] = r + pv;
                }
            }
        }
    }
}

/// Code (encode or decode) every band of `q` (volume layout) in pyramid
/// order with per-band predictors. When decoding, `q` must be zeroed and is
/// filled in.
pub fn code_bands<B: Bits, M: CoefModel>(p: &Pyramid, q: &mut [i32], preds: &[Pred], b: &mut B, model: &mut M) {
    let mut res = vec![0i32; q.len()];
    traverse(p, q, &mut res, preds, |f, _, r| model.code(b, f, r));
}

/// Visit every coefficient in coding order: `f(features, residual)` (for
/// training and analysis).
pub fn visit<F: FnMut(&Feat, i32)>(p: &Pyramid, q: &[i32], preds: &[Pred], mut f: F) {
    let mut qq = q.to_vec();
    let mut res = vec![0i32; q.len()];
    traverse(p, &mut qq, &mut res, preds, |ft, _, r| {
        f(ft, r);
        r
    });
}

/// Encode `q` with the hand model; returns the arithmetic-coded bytes.
pub fn encode_hand(p: &Pyramid, q: &[i32], preds: &[Pred]) -> Vec<u8> {
    let mut e = Encoder::new();
    let mut qq = q.to_vec();
    code_bands(p, &mut qq, preds, &mut e, &mut HandModel::default());
    e.finish()
}

/// Decode what [`encode_hand`] produced.
pub fn decode_hand(p: &Pyramid, bytes: &[u8], preds: &[Pred]) -> Vec<i32> {
    let mut d = Decoder::new(bytes);
    let mut q = vec![0i32; p.len()];
    code_bands(p, &mut q, preds, &mut d, &mut HandModel::default());
    q
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wavelet::{forward_97, MAX_LEVELS};

    #[test]
    fn round_trip_hand_model() {
        let d = [12, 20, 33];
        let p = Pyramid::new(d, MAX_LEVELS);
        let mut x: Vec<f64> = (0..p.len()).map(|i| ((i as f64) * 0.013).sin() * 100.0 + (i % 17) as f64).collect();
        forward_97(&p, &mut x);
        let mut q: Vec<i32> = x.iter().map(|v| (v / 0.7).round() as i32).collect();
        q[5] = 1_000_000; // exercise the escape path
        q[77] = -123_456;
        for preds in [vec![], fit_predictors(&p, &q)] {
            let bytes = encode_hand(&p, &q, &preds);
            assert_eq!(decode_hand(&p, &bytes, &preds), q);
            assert_eq!(encode_hand(&p, &q, &preds), bytes, "deterministic");
        }
    }
}
