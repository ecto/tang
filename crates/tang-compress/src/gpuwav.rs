//! `tc-gpu` (stream magic `TCG1`): a GPU-decodable, random-access variant of
//! the wavelet codec whose decode is **integer only**, so a CPU and a GPU
//! decoder produce the same bytes.
//!
//! # Layout
//!
//! - The field is cut into independent **bricks** (3D `B³`, 2D `B²` tiles),
//!   each with its own CDF 9/7 pyramid and its own entropy-coded lanes. A
//!   header index of brick sizes gives random access: a sub-box decodes only
//!   the bricks it touches.
//! - Values live on a fixed-point grid `x ≈ v·2^e` (`v` an `i32`); `e` is
//!   chosen so `|x|/2^e ≤ 2^23` (the final `i32 → f32` conversion is exact)
//!   and `|x − mean|/2^e ≤ 2^20` (transform headroom inside `i32`).
//!
//! # Decode (integer only)
//!
//! 1. **Coefficients.** Each brick is coded in [`lanes`](GpuConfig::lanes)
//!    independent rANS lanes (32-bit state, 16-bit words, 16-bit
//!    probabilities). Bands go coarse → fine; inside a band the lines along
//!    the fastest axis are split into *passes* by the parity of their
//!    `(y, z)` position (4 passes in 3D: `(0,0) (1,1) (1,0) (0,1)`; 3 in 2D:
//!    `y≡0 mod 4`, `y≡2 mod 4`, odd `y`). A line conditions only on lines of
//!    earlier passes, coarser bands, and its own earlier samples, so every
//!    line of a pass decodes in parallel; lines of a pass are dealt
//!    round-robin to the lanes. Each lane owns its adaptive binary models
//!    (the hand model's binarization: significance, adaptive-Golomb unary,
//!    low bits, sign) — adaptation is per lane, which is what keeps lanes
//!    independent.
//! 2. **Prediction.** Per (level, orientation, pass) a 6-tap Q12 predictor
//!    over `W, WW, Σfaces, Σdiagonals, Σ(±2 lines), parent`, restricted to
//!    what the pass can see.
//! 3. **Dequantization** in fixed point: `c = ⌊|q|·m/2^s⌉ − ⌊δ·m/2^(s+9)⌉`
//!    with a per-band 31-bit step mantissa `m`.
//! 4. **Inverse CDF 9/7** lifting in `i32` with Q24 constants and
//!    round-half-up products `⌊(a·k + 2^23)/2^24⌋` (a 32×32→64 multiply).
//! 5. **Max-abs corrections** `v += k·corr` in the integer domain, then
//!    `f32 = (v + mean) as f32 · 2^e` — exact, since `|v| < 2^24` and the
//!    scale is a normal power of two. Cells the grid still pushes past the
//!    bound are stored as raw `f32` bits.

use crate::bitio::{ByteReader, ByteWriter, DecodeError};
use crate::codec::ErrorBound;
use crate::coefcode::{band_class, KB, NCLASS, NE};
use crate::wavelet::{forward_97, Pyramid, MAX_LEVELS, MAX_LEVELS_1D};

/// Stream magic.
pub const MAGIC: &[u8; 4] = b"TCG1";

/// Fixed-point shift of the lifting constants.
pub const QL: u32 = 24;

fn qc(v: f64) -> i32 {
    (v * (1u64 << QL) as f64).round() as i32
}

const ALPHA: f64 = -1.586_134_342_059_924;
const BETA: f64 = -0.052_980_118_572_961;
const GAMMA: f64 = 0.882_911_075_530_934;
const DELTA: f64 = 0.443_506_852_043_971;
const KAPPA: f64 = 1.149_604_398_860_241;

/// Inverse-lifting constants (Q24): `1/κ, κ, −δ, −γ, −β, −α`.
pub fn lift_consts() -> [i32; 6] {
    [qc(1.0 / KAPPA), qc(KAPPA), qc(-DELTA), qc(-GAMMA), qc(-BETA), qc(-ALPHA)]
}

/// `⌊(a·b + 2^(s−1)) / 2^s⌋`, low 32 bits (the GPU emulates the 64-bit product).
#[inline]
pub fn mulsh(a: i32, b: i32, s: u32) -> i32 {
    ((a as i64 * b as i64 + (1i64 << (s - 1))) >> s) as i32
}

// ------------------------------------------------------------------ models

/// Two-rate adaptive bit model packed in a `u32` (fast in the low half).
pub const MINIT: u32 = 0x8000_8000;

#[inline]
pub(crate) fn mp1(m: u32) -> u32 {
    (((m & 0xFFFF) + (m >> 16)) >> 1).clamp(32, 65504)
}

#[inline]
fn mupd(m: u32, bit: bool) -> u32 {
    let (mut f, mut s) = (m & 0xFFFF, m >> 16);
    if bit {
        f += (65535 - f) >> 4;
        s += (65535 - s) >> 7;
    } else {
        f -= f >> 4;
        s -= s >> 7;
    }
    f | s << 16
}

const A_TH: [u32; 11] = [0, 1, 2, 3, 5, 7, 10, 14, 20, 30, 48];
const NA: usize = 12;
const P_TH: [u32; 3] = [0, 2, 6];
const NPB: usize = 5;

/// Model offsets inside one lane's state.
pub const SIG0: usize = 0;
const NSIG: usize = NCLASS * 2 * NA * NPB;
/// Magnitude unary models.
pub const MAG0: usize = SIG0 + NSIG;
/// Low-bit models.
pub const LOW0: usize = MAG0 + NCLASS * NE * KB;
/// Sign models.
pub const SGN0: usize = LOW0 + NE * 4;
/// Escape (Elias-gamma unary) models.
pub const ESC0: usize = SGN0 + NCLASS * 9;
/// Models per coefficient lane.
pub const NMODEL: usize = ESC0 + NCLASS * 32;
/// Models of the outlier lane (count, gap, |k| gamma unaries).
pub const NOMODEL: usize = 96;

#[inline]
fn bucket(a: u32, th: &[u32]) -> usize {
    th.iter().position(|&t| a <= t).unwrap_or(th.len())
}

// ------------------------------------------------------------------ rANS

/// A symbol sink/source for one brick: binary decisions at a probability, and
/// raw bit chunks, per lane.
pub trait Sink {
    /// Code `bit` with `P(1) = p1/2^16` on `lane`; returns the bit.
    fn bit(&mut self, lane: usize, bit: bool, p1: u32) -> bool;
    /// Code the low `n ≤ 16` bits of `v` raw.
    fn raw(&mut self, lane: usize, v: u32, n: u32) -> u32;
    /// Observe a modelled decision on model `idx` (prior counting).
    #[inline]
    fn note(&mut self, _idx: usize, _bit: bool) {}
}

/// Encoder sink: records `(cum, freq)` per lane in decode order.
pub struct RecSink {
    /// Per-lane symbols.
    pub syms: Vec<Vec<(u32, u32)>>,
}

impl Sink for RecSink {
    #[inline]
    fn bit(&mut self, lane: usize, bit: bool, p1: u32) -> bool {
        let f0 = 65536 - p1;
        self.syms[lane].push(if bit { (f0, p1) } else { (0, f0) });
        bit
    }
    #[inline]
    fn raw(&mut self, lane: usize, v: u32, n: u32) -> u32 {
        let v = v & ((1u32 << n) - 1);
        self.syms[lane].push((v << (16 - n), 1 << (16 - n)));
        v
    }
}

/// rANS-encode one lane (symbols in decode order) into 16-bit words.
pub fn rans_encode(syms: &[(u32, u32)]) -> Vec<u16> {
    if syms.is_empty() {
        return Vec::new();
    }
    let mut x: u32 = 1 << 16;
    let mut out: Vec<u16> = Vec::with_capacity(syms.len() / 8 + 4);
    for &(cum, f) in syms.iter().rev() {
        let xmax = f << 16;
        if x >= xmax {
            out.push(x as u16);
            x >>= 16;
        }
        x = ((x / f) << 16) + (x % f) + cum;
    }
    out.push((x >> 16) as u16);
    out.push(x as u16);
    // decode order: state low word, high word, then renorm words
    out.reverse();
    out
}

/// Decoder of one lane.
#[derive(Clone, Debug)]
pub struct RansDec<'a> {
    w: &'a [u16],
    pos: usize,
    x: u32,
}

impl<'a> RansDec<'a> {
    /// Start decoding `w` (empty lanes are allowed and never read).
    pub fn new(w: &'a [u16]) -> Self {
        let x = if w.len() >= 2 { w[0] as u32 | (w[1] as u32) << 16 } else { 1 << 16 };
        Self { w, pos: 2, x }
    }
    #[inline]
    fn step(&mut self, cum: u32, f: u32) {
        self.x = f.wrapping_mul(self.x >> 16).wrapping_add((self.x & 0xFFFF).wrapping_sub(cum));
        if self.x < 1 << 16 {
            let v = self.w.get(self.pos).copied().unwrap_or(0) as u32;
            self.pos += 1;
            self.x = (self.x << 16) | v;
        }
    }
    /// Decode a bit at `P(1) = p1/2^16`.
    #[inline]
    pub fn bit(&mut self, p1: u32) -> bool {
        let f0 = 65536 - p1;
        let b = (self.x & 0xFFFF) >= f0;
        if b {
            self.step(f0, p1)
        } else {
            self.step(0, f0)
        }
        b
    }
    /// Decode `n ≤ 16` raw bits.
    #[inline]
    pub fn raw(&mut self, n: u32) -> u32 {
        let v = (self.x & 0xFFFF) >> (16 - n);
        self.step(v << (16 - n), 1 << (16 - n));
        v
    }
    /// Whether more words were consumed than the lane holds.
    pub fn overrun(&self) -> bool {
        self.w.len() >= 2 && self.pos > self.w.len()
    }
}

/// Decoder sink over a brick's lanes.
pub struct DecSink<'a> {
    /// Lanes.
    pub lanes: Vec<RansDec<'a>>,
}

impl Sink for DecSink<'_> {
    #[inline]
    fn bit(&mut self, lane: usize, _: bool, p1: u32) -> bool {
        self.lanes[lane].bit(p1)
    }
    #[inline]
    fn raw(&mut self, lane: usize, _: u32, n: u32) -> u32 {
        self.lanes[lane].raw(n)
    }
}

#[inline]
pub(crate) fn bitm<S: Sink, const A: bool>(s: &mut S, lane: usize, m: &mut [u32], idx: usize, bit: bool) -> bool {
    let b = s.bit(lane, bit, mp1(m[idx]));
    if A {
        m[idx] = mupd(m[idx], b);
    }
    s.note(idx, b);
    b
}

#[inline]
pub(crate) fn raw_n<S: Sink>(s: &mut S, lane: usize, v: u32, mut n: u32) -> u32 {
    let mut out = 0u32;
    while n > 16 {
        out = (out << 16) | s.raw(lane, v >> (n - 16), 16);
        n -= 16;
    }
    if n > 0 {
        out = (out << n) | s.raw(lane, v, n);
    }
    out
}

/// Adaptive Elias-gamma for `v < 2^31`.
#[inline]
pub(crate) fn gamma<S: Sink, const A: bool>(s: &mut S, lane: usize, m: &mut [u32], base: usize, v: u32) -> u32 {
    let w = v.wrapping_add(1);
    let nbe = 31 - w.max(1).leading_zeros();
    let mut nb = 0u32;
    while nb < 31 && bitm::<S, A>(s, lane, m, base + nb as usize, nb < nbe) {
        nb += 1;
    }
    let lo = raw_n(s, lane, w, nb);
    ((1u32 << nb) | lo).wrapping_sub(1)
}

// ------------------------------------------------------------------ priors

/// `e^x` from IEEE `+ − × /` only (deterministic on every platform).
fn exp_det(x: f64) -> f64 {
    let ln2 = core::f64::consts::LN_2;
    let k = (x / ln2).round();
    let r = x - k * ln2;
    let (mut t, mut s) = (1.0f64, 1.0f64);
    for i in 1..25 {
        t = t * r / i as f64;
        s += t;
    }
    s * f64::from_bits(((k as i64 + 1023) as u64) << 52)
}

/// The 64 prior probabilities `P(1)·2^16` (logistic grid, step 0.28 in logit).
pub fn prior_table() -> &'static [u32; 64] {
    use std::sync::OnceLock;
    static T: OnceLock<[u32; 64]> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0u32; 64];
        for (k, v) in t.iter_mut().enumerate() {
            let l = (k as f64 - 31.5) * 0.28;
            *v = (65536.0 / (1.0 + exp_det(-l))).round().clamp(32.0, 65504.0) as u32;
        }
        t
    })
}

/// Counts decisions per model.
pub(crate) struct CountSink {
    pub(crate) n: Vec<[u32; 2]>,
}

impl Sink for CountSink {
    #[inline]
    fn bit(&mut self, _: usize, bit: bool, _: u32) -> bool {
        bit
    }
    #[inline]
    fn raw(&mut self, _: usize, v: u32, n: u32) -> u32 {
        v & ((1u32 << n) - 1)
    }
    #[inline]
    fn note(&mut self, idx: usize, bit: bool) {
        self.n[idx][bit as usize] += 1;
    }
}

/// Choose priors (index into [`prior_table`]) from pooled counts.
pub(crate) fn choose_priors(n: &[[u32; 2]], adaptive: bool) -> Vec<Option<u8>> {
    let th: u32 = std::env::var("TC_PRIOR_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(if adaptive { 16 } else { 4 });
    let t = prior_table();
    n.iter()
        .map(|&[n0, n1]| {
            if n0 + n1 < th {
                return None;
            }
            let cost = |p: u32| {
                let p = p as f64 / 65536.0;
                -(n1 as f64) * p.log2() - (n0 as f64) * (1.0 - p).log2()
            };
            (0..64u8).min_by(|&a, &b| cost(t[a as usize]).total_cmp(&cost(t[b as usize])))
        })
        .collect()
}

pub(crate) fn write_priors(w: &mut ByteWriter, pr: &[Option<u8>]) {
    use crate::abac::{BitModel, Encoder, UintModel};
    let mut e = Encoder::new();
    let mut pm = [BitModel::default(); 2];
    let mut um = UintModel::default();
    let (mut prev, mut last) = (false, 32i64);
    for p in pr {
        e.encode(p.is_some(), &mut pm[prev as usize]);
        prev = p.is_some();
        if let Some(k) = p {
            um.encode(&mut e, crate::bitio::zigzag(*k as i64 - last));
            last = *k as i64;
        }
    }
    w.block(&e.finish());
}

pub(crate) fn read_priors(r: &mut ByteReader) -> Result<Vec<u32>, DecodeError> {
    use crate::abac::{BitModel, Decoder, UintModel};
    let buf = r.block()?;
    let mut d = Decoder::new(buf);
    let mut pm = [BitModel::default(); 2];
    let mut um = UintModel::default();
    let (mut prev, mut last) = (false, 32i64);
    let t = prior_table();
    let mut init = vec![MINIT; NMODEL];
    for v in init.iter_mut() {
        prev = d.decode(&mut pm[prev as usize]);
        if prev {
            last += crate::bitio::unzigzag(um.decode(&mut d));
            let k = last.clamp(0, 63) as usize;
            *v = t[k] | t[k] << 16;
        }
    }
    if d.overrun() {
        return Err(DecodeError::Truncated);
    }
    Ok(init)
}

pub(crate) fn init_of(pr: &[Option<u8>]) -> Vec<u32> {
    let t = prior_table();
    pr.iter().map(|p| p.map_or(MINIT, |k| t[k as usize] | t[k as usize] << 16)).collect()
}

// ------------------------------------------------------------------ shapes

/// Per-band decode parameters.
#[derive(Clone, Debug)]
pub struct BandInfo {
    /// Region start (brick-local).
    pub start: [usize; 3],
    /// Region end (exclusive).
    pub end: [usize; 3],
    /// Context class.
    pub class: usize,
    /// Parent band index.
    pub parent: Option<usize>,
    /// Parent-position shift per axis (1 if the parent level splits it).
    pub pshift: [u32; 3],
    /// Number of passes.
    pub npass: usize,
    /// Predictor key per pass.
    pub pkey: [usize; 8],
    /// Dequantization step mantissa.
    pub m: i32,
    /// Dequantization step shift.
    pub sh: u32,
    /// Synthesis-basis weight.
    pub w: f64,
    /// Origin of the subband this region belongs to (parent mapping); equal
    /// to `start` for whole bands, the band start for a code-block.
    pub org: [usize; 3],
    /// Raster order inside the region (one lane, every earlier line
    /// visible) instead of parity passes.
    pub raster: bool,
}

/// A brick shape and its pyramid.
#[derive(Clone, Debug)]
pub struct Shape {
    /// Brick extent.
    pub dims: [usize; 3],
    /// Pyramid.
    pub p: Pyramid,
    /// Bands in coding order.
    pub bands: Vec<BandInfo>,
}

pub(crate) fn max_levels(dims: [usize; 3]) -> usize {
    if dims.iter().filter(|&&d| d > 1).count() <= 1 {
        MAX_LEVELS_1D
    } else {
        MAX_LEVELS
    }
}

/// Pass count of a band.
pub fn npass_of(start: [usize; 3], end: [usize; 3]) -> usize {
    let (ez, ey) = (end[0] - start[0], end[1] - start[1]);
    if ez > 1 && ey > 1 {
        7
    } else if ez > 1 || ey > 1 {
        4
    } else {
        1
    }
}

/// Pass of band-relative line `(zr, yr)`.
#[inline]
pub fn pass_of(npass: usize, two_axes: bool, zy_is_y: bool, zr: usize, yr: usize) -> usize {
    let _ = two_axes;
    match npass {
        7 => {
            if (yr | zr) & 1 == 0 {
                [0, 2, 3, 1][(yr >> 1 & 1) | (zr >> 1 & 1) << 1]
            } else {
                [0, 5, 6, 4][(yr & 1) | (zr & 1) << 1]
            }
        }
        4 => {
            let t = if zy_is_y { yr } else { zr };
            if t & 1 == 1 {
                3
            } else if t & 3 == 2 {
                2
            } else if t & 7 == 4 {
                1
            } else {
                0
            }
        }
        _ => 0,
    }
}

#[inline]
fn band_pass(b: &BandInfo, zr: usize, yr: usize) -> usize {
    let ey = b.end[1] - b.start[1];
    pass_of(b.npass, false, ey > 1, zr, yr)
}

/// Predictor key of `(level, mask, pass)`; the LL band is level 15.
fn key_id(level: usize, mask: u8, pass: usize, nlev: usize) -> usize {
    let l = if mask == 0 { 15 } else { level.min(14) };
    let _ = nlev;
    (l * 8 + mask as usize) * 8 + pass
}

/// Number of distinct predictor keys addressable.
pub const NKEY: usize = 16 * 8 * 8;

impl Shape {
    pub(crate) fn new(dims: [usize; 3]) -> Self {
        let p = Pyramid::new(dims, max_levels(dims));
        let bands = (0..p.bands.len())
            .map(|bi| {
                let b = &p.bands[bi];
                let parent = p.parent_band(bi);
                let mut pshift = [0; 3];
                if let Some(pi) = parent {
                    let pl = p.bands[pi].level;
                    for a in 0..3 {
                        pshift[a] = p.split[pl][a] as u32;
                    }
                }
                let npass = npass_of(b.start, b.end);
                let mut pkey = [0; 8];
                for (ps, k) in pkey.iter_mut().enumerate().take(npass) {
                    *k = key_id(b.level, b.mask, ps, p.nlev());
                }
                BandInfo {
                    start: b.start,
                    end: b.end,
                    class: band_class(b) as usize,
                    parent,
                    pshift,
                    npass,
                    pkey,
                    m: 0,
                    sh: 1,
                    w: p.band_weight(b),
                    org: b.start,
                    raster: false,
                }
            })
            .collect();
        Self { dims, p, bands }
    }

    pub(crate) fn set_steps(&mut self, delta_g: f64) {
        for b in self.bands.iter_mut() {
            let (m, sh) = step_repr(delta_g / b.w);
            b.m = m;
            b.sh = sh;
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.dims.iter().product()
    }
}

/// `step ≈ m·2^−sh` with `m ∈ [2^29, 2^30]`.
pub(crate) fn step_repr(step: f64) -> (i32, u32) {
    let lg = step.log2().floor() as i32;
    let sh = (29 - lg).clamp(1, 52);
    let m = (step * 2f64.powi(sh)).round().clamp(1.0, (1u64 << 31) as f64 - 1.0) as i32;
    (m, sh as u32)
}

// ------------------------------------------------------------------ features

/// Row neighbour offsets `(dz, dy, category)`: 0 faces, 1 diagonals, 2 far.
const ROWNB: [(i32, i32, u8); 20] = [
    (0, -1, 0),
    (0, 1, 0),
    (-1, 0, 0),
    (1, 0, 0),
    (-1, -1, 1),
    (-1, 1, 1),
    (1, -1, 1),
    (1, 1, 1),
    (0, -2, 2),
    (0, 2, 2),
    (-2, 0, 2),
    (2, 0, 2),
    (-2, -2, 2),
    (-2, 2, 2),
    (2, -2, 2),
    (2, 2, 2),
    (0, -4, 2),
    (0, 4, 2),
    (-4, 0, 2),
    (4, 0, 2),
];

/// Number of predictor taps.
pub const NTAP: usize = 6;
const TCLAMP: i32 = 1 << 13;

/// Lines (as base indices of `(z, y, start_x)`) visible from a line.
#[derive(Clone, Copy, Default)]
pub(crate) struct RowCtx {
    pub(crate) base: [usize; 20],
    pub(crate) cat: [u8; 20],
    pub(crate) n: usize,
}

pub(crate) fn row_ctx(sh: &Shape, b: &BandInfo, p: usize, z: usize, y: usize) -> RowCtx {
    let [_, ny, nx] = sh.dims;
    let mut rc = RowCtx::default();
    for &(dz, dy, cat) in &ROWNB {
        let (z2, y2) = (z as i32 + dz, y as i32 + dy);
        if z2 < b.start[0] as i32 || z2 >= b.end[0] as i32 || y2 < b.start[1] as i32 || y2 >= b.end[1] as i32 {
            continue;
        }
        let (z2, y2) = (z2 as usize, y2 as usize);
        if b.raster {
            if (z2, y2) >= (z, y) {
                continue;
            }
        } else if band_pass(b, z2 - b.start[0], y2 - b.start[1]) >= p {
            continue;
        }
        rc.base[rc.n] = (z2 * ny + y2) * nx;
        rc.cat[rc.n] = cat;
        rc.n += 1;
    }
    rc
}

/// Everything the coder derives for one coefficient.
#[derive(Clone, Copy, Debug, Default)]
pub struct Feats {
    /// Predictor taps.
    pub taps: [i32; NTAP],
    /// Significance context.
    pub sig: usize,
    /// Magnitude scale.
    pub e: usize,
    /// Sign context.
    pub sctx: usize,
    /// Raw context quantities for a learned model: W, WW, Σfaces, Σdiag,
    /// Σfar, parent, prediction (values); |W|, |WW|, Σ|faces|, Σ|diag|,
    /// Σ|far|, |parent|, Σ|parent nbrs| (context magnitudes); counts of
    /// visible face/diag/far lines; has-parent.
    pub raw: [i32; 18],
}

#[inline]
pub(crate) fn cap(v: i32) -> u32 {
    v.unsigned_abs().min(65535)
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn feats(sh: &Shape, b: &BandInfo, rc: &RowCtx, q: &[i32], rs: Option<&[i32]>, z: usize, y: usize, x: usize, w: &[i32; NTAP]) -> (Feats, i32) {
    // magnitudes/signs for contexts come from `m` (the residuals when given)
    let m = rs.unwrap_or(q);
    let [_, ny, nx] = sh.dims;
    let idx = (z * ny + y) * nx + x;
    let x0 = b.start[2];
    let wv = if x > x0 { q[idx - 1] } else { 0 };
    let wwv = if x > x0 + 1 { q[idx - 2] } else { 0 };
    let wm = if x > x0 { m[idx - 1] } else { 0 };
    let wwm = if x > x0 + 1 { m[idx - 2] } else { 0 };
    let (mut sf, mut sd, mut sr) = (0i32, 0i32, 0i32);
    let (mut af, mut ad, mut ar) = (0u32, 0u32, 0u32);
    let (mut nf, mut nd, mut nr) = (0u32, 0u32, 0u32);
    for j in 0..rc.n {
        let v = q[rc.base[j] + x];
        let vm = m[rc.base[j] + x];
        match rc.cat[j] {
            0 => {
                sf = sf.wrapping_add(v);
                af += cap(vm);
                if b.raster {
                    // code-blocks: the face lines' x±1 neighbours feed the activity
                    if x > x0 {
                        ad += cap(m[rc.base[j] + x - 1]);
                        nd += 1;
                    }
                    if x + 1 < b.end[2] {
                        ad += cap(m[rc.base[j] + x + 1]);
                        nd += 1;
                    }
                }
                nf += 1;
            }
            1 => {
                sd = sd.wrapping_add(v);
                ad += cap(vm);
                nd += 1;
            }
            _ => {
                sr = sr.wrapping_add(v);
                ar += cap(vm);
                nr += 1;
            }
        }
    }
    let (mut pv, mut pm, mut pnb, mut has_p) = (0i32, 0i32, 0u32, false);
    if let Some(pi) = b.parent {
        let pb = &sh.bands[pi];
        let mut pp = [0usize; 3];
        let pos = [z, y, x];
        for a in 0..3 {
            pp[a] = (pb.start[a] + ((pos[a] - b.org[a]) >> b.pshift[a])).min(pb.end[a] - 1);
        }
        let at = |p: [usize; 3]| m[(p[0] * ny + p[1]) * nx + p[2]];
        pv = q[(pp[0] * ny + pp[1]) * nx + pp[2]];
        pm = at(pp);
        has_p = true;
        for a in 0..3 {
            if pp[a] > pb.start[a] {
                let mut t = pp;
                t[a] -= 1;
                pnb += cap(at(t));
            }
            if pp[a] + 1 < pb.end[a] {
                let mut t = pp;
                t[a] += 1;
                pnb += cap(at(t));
            }
        }
    }
    let cl = |v: i32| v.clamp(-TCLAMP, TCLAMP);
    let taps = [cl(wv), cl(wwv), cl(sf), cl(sd), cl(sr), cl(pv)];
    let mut acc = 0i32;
    for i in 0..NTAP {
        acc = acc.wrapping_add(w[i].wrapping_mul(taps[i]));
    }
    let pred = acc.wrapping_add(2048) >> 12;
    let s = 2 * cap(wm) + cap(wwm) + 2 * af + ad + ar + 2 * cap(pm) + 2 * cap(pred);
    let ws = 3 + 2 * nf + nd + nr + 2 * has_p as u32 + 2;
    let a = 12 * s / ws;
    let ab = bucket(a, &A_TH);
    let pb = if has_p { 1 + bucket(4 * cap(pm).min(15) + pnb.min(60), &P_TH) } else { 0 };
    let g = (nf + nd + nr > 0) as usize;
    let c = b.class;
    let e = ((1 + 2 * s / ws).ilog2() as usize).min(NE - 1);
    let sgn = |v: i32| (v.signum() + 1) as usize;
    let f = Feats {
        taps,
        sig: ((c * 2 + g) * NA + ab) * NPB + pb,
        e,
        sctx: c * 9 + sgn(wm) * 3 + sgn(sf.wrapping_add(sd)),
        raw: [
            wv,
            wwv,
            sf,
            sd,
            sr,
            pv,
            pred,
            cap(wm) as i32,
            cap(wwm) as i32,
            af as i32,
            ad as i32,
            ar as i32,
            cap(pm) as i32,
            pnb as i32,
            nf as i32,
            nd as i32,
            nr as i32,
            has_p as i32,
        ],
    };
    (f, pred)
}

/// Code one residual.
#[inline]
pub(crate) fn code_res<S: Sink, const A: bool>(s: &mut S, lane: usize, m: &mut [u32], c: usize, f: &Feats, r: i32) -> i32 {
    let mag = r.unsigned_abs();
    if !bitm::<S, A>(s, lane, m, SIG0 + f.sig, mag > 0) {
        return 0;
    }
    let e = f.e;
    let k = e.saturating_sub(2) as u32;
    let rr = mag.wrapping_sub(1);
    let hi = rr >> k;
    let mut h = 0usize;
    while h < KB {
        if !bitm::<S, A>(s, lane, m, MAG0 + (c * NE + e) * KB + h, hi > h as u32) {
            break;
        }
        h += 1;
    }
    let mut hv = h as u32;
    if h == KB {
        hv = hv.wrapping_add(gamma::<S, A>(s, lane, m, ESC0 + c * 32, hi.wrapping_sub(KB as u32)));
    }
    let mut lo = 0u32;
    let mut j = k;
    while j > 0 {
        let pos = k - j;
        if pos < 2 {
            let idx = LOW0 + (e * 2 + pos as usize) * 2 + (pos == 1 && lo & 1 == 1) as usize;
            let bit = bitm::<S, A>(s, lane, m, idx, (rr >> (j - 1)) & 1 == 1);
            lo = (lo << 1) | bit as u32;
            j -= 1;
        } else {
            let got = raw_n(s, lane, rr & ((1u32 << j) - 1), j);
            lo = (lo << j) | got;
            j = 0;
        }
    }
    let mv = ((hv << k) | lo).wrapping_add(1);
    let neg = bitm::<S, A>(s, lane, m, SGN0 + f.sctx, r < 0);
    if neg {
        (mv as i32).wrapping_neg()
    } else {
        mv as i32
    }
}

/// Lines of pass `p` form a lattice `z ≡ rz (mod sz)`, `y ≡ ry (mod sy)`
/// (band-relative): returns `(rz, sz, ry, sy)`.
pub fn pass_lattice(npass: usize, zy_is_y: bool, p: usize) -> (usize, usize, usize, usize) {
    match npass {
        7 => [(0, 4, 0, 4), (2, 4, 2, 4), (0, 4, 2, 4), (2, 4, 0, 4), (1, 2, 1, 2), (0, 2, 1, 2), (1, 2, 0, 2)][p],
        4 => {
            let (r, st) = [(0, 8), (4, 8), (2, 4), (1, 2)][p];
            if zy_is_y {
                (0, 1, r, st)
            } else {
                (r, st, 0, 1)
            }
        }
        _ => (0, 1, 0, 1),
    }
}

/// Visit the lines of `band` in coding order: `f(pass, lane, z, y)` — per
/// pass, the pass's lines in raster order, dealt round-robin to the lanes.
pub(crate) fn for_lines(b: &BandInfo, lanes: usize, mut f: impl FnMut(usize, usize, usize, usize)) {
    if b.raster {
        for z in b.start[0]..b.end[0] {
            for y in b.start[1]..b.end[1] {
                f(0, 0, z, y);
            }
        }
        return;
    }
    let (ez, ey) = (b.end[0] - b.start[0], b.end[1] - b.start[1]);
    for p in 0..b.npass {
        let (rz, sz, ry, sy) = pass_lattice(b.npass, ey > 1, p);
        if rz >= ez || ry >= ey {
            continue;
        }
        let cy = (ey - ry).div_ceil(sy);
        let n = (ez - rz).div_ceil(sz) * cy;
        for j in 0..n {
            f(p, j % lanes, b.start[0] + rz + (j / cy) * sz, b.start[1] + ry + (j % cy) * sy);
        }
    }
}

/// Code (encode or decode) a brick's coefficients `q`; `adapt` selects
/// per-lane adaptive models (else the priors are static probabilities).
pub fn code_brick<S: Sink>(sh: &Shape, q: &mut [i32], preds: &[[i32; NTAP]], init: &[u32], lanes: usize, adapt: bool, s: &mut S) {
    if adapt {
        code_brick_impl::<S, true>(sh, q, preds, init, lanes, s)
    } else {
        code_brick_impl::<S, false>(sh, q, preds, init, lanes, s)
    }
}

fn code_brick_impl<S: Sink, const A: bool>(sh: &Shape, q: &mut [i32], preds: &[[i32; NTAP]], init: &[u32], lanes: usize, s: &mut S) {
    let [_, ny, nx] = sh.dims;
    let mut models: Vec<u32> = (0..lanes).flat_map(|_| init.iter().copied()).collect();
    for b in &sh.bands {
        for_lines(b, lanes, |p, lane, z, y| {
            let rc = row_ctx(sh, b, p, z, y);
            let w = &preds[b.pkey[p]];
            let m = &mut models[lane * NMODEL..(lane + 1) * NMODEL];
            for x in b.start[2]..b.end[2] {
                let (f, pred) = feats(sh, b, &rc, q, None, z, y, x, w);
                let i = (z * ny + y) * nx + x;
                let r = code_res::<S, A>(s, lane, m, b.class, &f, q[i].wrapping_sub(pred));
                q[i] = r.wrapping_add(pred);
            }
        });
    }
}

// ------------------------------------------------------------------ integer transform

pub(crate) fn inv97_line(x: &mut [i32], tmp: &mut Vec<i32>, k: &[i32; 6]) {
    let n = x.len();
    if n < 2 {
        return;
    }
    let ns = n.div_ceil(2);
    tmp.clear();
    tmp.extend_from_slice(x);
    let (s, d) = tmp.split_at_mut(ns);
    let nd = d.len();
    for v in s.iter_mut() {
        *v = mulsh(*v, k[0], QL);
    }
    for v in d.iter_mut() {
        *v = mulsh(*v, k[1], QL);
    }
    let lift_s = |s: &mut [i32], d: &[i32], c: i32| {
        for i in 0..ns {
            let l = if i == 0 { d[0] } else { d[i - 1] };
            let r = if i < nd { d[i] } else { d[nd - 1] };
            s[i] = s[i].wrapping_add(mulsh(l.wrapping_add(r), c, QL));
        }
    };
    let lift_d = |s: &[i32], d: &mut [i32], c: i32| {
        for i in 0..nd {
            let r = if i + 1 < ns { s[i + 1] } else { s[ns - 1] };
            d[i] = d[i].wrapping_add(mulsh(s[i].wrapping_add(r), c, QL));
        }
    };
    lift_s(s, d, k[2]);
    lift_d(s, d, k[3]);
    lift_s(s, d, k[4]);
    lift_d(s, d, k[5]);
    for i in 0..ns {
        x[2 * i] = s[i];
    }
    for i in 0..nd {
        x[2 * i + 1] = d[i];
    }
}

/// Inverse multi-level integer CDF 9/7 on a brick.
pub fn inverse_int(p: &Pyramid, x: &mut [i32]) {
    let k = lift_consts();
    let mut line = Vec::new();
    let mut tmp = Vec::new();
    let st = [p.dims[1] * p.dims[2], p.dims[2], 1];
    for l in (0..p.nlev()).rev() {
        for a in [2usize, 1, 0] {
            if !p.split[l][a] {
                continue;
            }
            let ext = p.lens[l];
            let o: Vec<usize> = (0..3).filter(|&b| b != a).collect();
            for i in 0..ext[o[0]] {
                for j in 0..ext[o[1]] {
                    let off = i * st[o[0]] + j * st[o[1]];
                    line.clear();
                    line.extend((0..ext[a]).map(|t| x[off + t * st[a]]));
                    inv97_line(&mut line, &mut tmp, &k);
                    for (t, &v) in line.iter().enumerate() {
                        x[off + t * st[a]] = v;
                    }
                }
            }
        }
    }
}

/// Dequantize one value of band `b`.
#[inline]
pub fn dequant(v: i32, b: &BandInfo, dl: &[u8; NDL]) -> i32 {
    if v == 0 {
        return 0;
    }
    let a = v.unsigned_abs().min(i32::MAX as u32) as i32;
    let d = dl[b.class * 3 + (v.unsigned_abs().min(3) as usize) - 1] as i32;
    let c = mulsh(a, b.m, b.sh).wrapping_sub(mulsh(d, b.m, b.sh + 9));
    if v < 0 {
        c.wrapping_neg()
    } else {
        c
    }
}

/// Number of centroid offsets.
pub const NDL: usize = NCLASS * 3;

/// Largest magnitude allowed into the final `i32 → f32` conversion.
pub const VMAX: i32 = (1 << 24) - 1;

/// `f32` bits of fixed-point `v` at scale `2^e` (exact: `|v| < 2^24`).
#[inline]
pub fn to_f32_bits(v: i32, scale: f32) -> u32 {
    ((v.clamp(-VMAX, VMAX) as f32) * scale).to_bits()
}

// ------------------------------------------------------------------ stream

/// tc-gpu options.
#[derive(Clone, Debug)]
pub struct GpuConfig {
    /// Error target.
    pub bound: ErrorBound,
    /// Brick edge (3D `B³`, 2D `B²`, 1D `B²` samples).
    pub brick: usize,
    /// rANS lanes per brick.
    pub lanes: usize,
    /// Deadzone rounding offset.
    pub rho: f64,
    /// Per-lane adaptive context models (CPU decode only); the default is
    /// static per-context probabilities (the transmitted priors), which the
    /// GPU decoder requires.
    pub adaptive: bool,
}

impl GpuConfig {
    /// Defaults (64³ bricks, 32 lanes).
    pub fn new(bound: ErrorBound) -> Self {
        Self { bound, brick: 64, lanes: 32, rho: 0.45, adaptive: false }
    }
}

/// Encoder statistics.
#[derive(Clone, Debug, Default)]
pub struct GpuStats {
    /// Total bytes.
    pub bytes: usize,
    /// Bits per value.
    pub bits_per_value: f64,
    /// Header bytes (incl. brick index and predictors).
    pub header_bytes: usize,
    /// Lane-size index bytes inside bricks.
    pub index_bytes: usize,
    /// Outlier-lane + escape bytes.
    pub outlier_bytes: usize,
    /// Corrected cells.
    pub n_outliers: usize,
    /// Raw escapes.
    pub n_escapes: usize,
    /// Measured RMSE.
    pub rmse: f64,
    /// Measured max-abs error.
    pub max_abs: f64,
    /// Step (absolute units).
    pub delta: f64,
}

/// A brick of a parsed stream.
#[derive(Clone, Debug)]
pub struct BrickRef {
    /// Origin in the field.
    pub origin: [usize; 3],
    /// Shape index.
    pub shape: usize,
    /// Byte offset of the brick payload.
    pub off: usize,
    /// Word offsets (absolute, in 16-bit words of the stream) of the lanes
    /// (coefficient lanes, then the outlier lane), and one past the end.
    pub lane_word: Vec<usize>,
    /// Raw escapes `(local index, f32 bits)`.
    pub esc: Vec<(usize, u32)>,
}

/// A parsed stream header + brick index.
#[derive(Clone, Debug)]
pub struct Parsed {
    /// Original dims.
    pub dims: Vec<usize>,
    /// 3D dims.
    pub d3: [usize; 3],
    /// Brick extent.
    pub bdims: [usize; 3],
    /// Lanes per brick.
    pub lanes: usize,
    /// Adaptive (vs static) context models.
    pub adaptive: bool,
    /// Grid exponent.
    pub e: i32,
    /// Mean on the grid.
    pub mean_fix: i32,
    /// Step in grid units.
    pub delta_g: f64,
    /// Centroid offsets.
    pub dl: [u8; NDL],
    /// Max-abs correction step on the grid (0 = none).
    pub corr: i32,
    /// Predictor weights per key.
    pub preds: Vec<[i32; NTAP]>,
    /// Initial model state of every coefficient lane.
    pub init: Vec<u32>,
    /// Shapes.
    pub shapes: Vec<Shape>,
    /// Bricks.
    pub bricks: Vec<BrickRef>,
}

impl Parsed {
    /// Output scale `2^e`.
    pub fn scale(&self) -> f32 {
        f32::from_bits(((self.e + 127) as u32) << 23)
    }
}

fn brick_grid(d3: [usize; 3], bd: [usize; 3]) -> Vec<([usize; 3], [usize; 3])> {
    let mut v = Vec::new();
    for z in (0..d3[0]).step_by(bd[0]) {
        for y in (0..d3[1]).step_by(bd[1]) {
            for x in (0..d3[2]).step_by(bd[2]) {
                let o = [z, y, x];
                v.push((o, [0, 1, 2].map(|a| bd[a].min(d3[a] - o[a]))));
            }
        }
    }
    v
}

fn brick_dims(d3: [usize; 3], b: usize) -> [usize; 3] {
    let nd = d3.iter().filter(|&&d| d > 1).count();
    match nd {
        3 => [b.min(d3[0]), b.min(d3[1]), b.min(d3[2])],
        2 => [1, b.min(d3[1]), b.min(d3[2])],
        _ => [1, 1, (b * b).min(d3[2])],
    }
}

/// Distinct shapes and per-brick shape index.
fn shapes_for(grid: &[([usize; 3], [usize; 3])]) -> (Vec<Shape>, Vec<usize>) {
    let mut shapes: Vec<Shape> = Vec::new();
    let ids = grid
        .iter()
        .map(|(_, s)| {
            if let Some(i) = shapes.iter().position(|sh| sh.dims == *s) {
                i
            } else {
                shapes.push(Shape::new(*s));
                shapes.len() - 1
            }
        })
        .collect();
    (shapes, ids)
}

/// Keys used by any shape, sorted.
fn used_keys(shapes: &[Shape]) -> Vec<usize> {
    let mut k: Vec<usize> = shapes.iter().flat_map(|s| s.bands.iter().flat_map(|b| b.pkey[..b.npass].to_vec())).collect();
    k.sort_unstable();
    k.dedup();
    k
}

/// Parse a stream's header and brick index.
pub fn parse(bytes: &[u8]) -> Result<Parsed, DecodeError> {
    let mut r = ByteReader::new(bytes);
    if r.take(4)? != MAGIC {
        return Err(DecodeError::Invalid("not a tc-gpu stream"));
    }
    let nd = r.varint()? as usize;
    if nd == 0 || nd > 3 {
        return Err(DecodeError::Invalid("rank"));
    }
    let dims: Vec<usize> = (0..nd).map(|_| r.varint().map(|v| v as usize)).collect::<Result<_, _>>()?;
    let d3 = crate::wavelet::as3(&dims);
    let mut bdims = [0usize; 3];
    for v in bdims.iter_mut() {
        *v = r.varint()? as usize;
        if *v == 0 {
            return Err(DecodeError::Invalid("brick"));
        }
    }
    let lanes = r.varint()? as usize;
    if lanes == 0 || lanes > 1024 {
        return Err(DecodeError::Invalid("lanes"));
    }
    let adaptive = r.u8()? != 0;
    let e = r.svarint()? as i32;
    let mean_fix = r.svarint()? as i32;
    let delta_g = r.f64()?;
    let mut dl = [0u8; NDL];
    dl.copy_from_slice(r.take(NDL)?);
    let corr = r.varint()? as i32;
    let grid = brick_grid(d3, bdims);
    let (mut shapes, ids) = shapes_for(&grid);
    for s in shapes.iter_mut() {
        s.set_steps(delta_g);
    }
    let mut preds = vec![[0i32; NTAP]; NKEY];
    for k in used_keys(&shapes) {
        if r.u8()? != 0 {
            for t in preds[k].iter_mut() {
                *t = r.svarint()? as i32;
            }
        }
    }
    let init = read_priors(&mut r)?;
    let sizes: Vec<usize> = (0..grid.len()).map(|_| r.varint().map(|v| v as usize)).collect::<Result<_, _>>()?;
    let mut off = r.position().next_multiple_of(2);
    let mut bricks = Vec::with_capacity(grid.len());
    for (bi, (o, _)) in grid.iter().enumerate() {
        let end = off + sizes[bi];
        if end > bytes.len() {
            return Err(DecodeError::Truncated);
        }
        let mut br = ByteReader::new(&bytes[off..end]);
        let nw: Vec<usize> = (0..=lanes).map(|_| br.varint().map(|v| v as usize)).collect::<Result<_, _>>()?;
        let ne = br.varint()? as usize;
        let mut esc = Vec::with_capacity(ne);
        let mut last = 0usize;
        for i in 0..ne {
            let g = br.varint()? as usize;
            let idx = if i == 0 { g } else { last + g };
            last = idx;
            let b = br.take(4)?;
            esc.push((idx, u32::from_le_bytes([b[0], b[1], b[2], b[3]])));
        }
        let mut w = (off + br.position().next_multiple_of(2)) / 2;
        let mut lane_word = Vec::with_capacity(lanes + 2);
        for &n in &nw {
            lane_word.push(w);
            w += n;
        }
        lane_word.push(w);
        if w * 2 > end {
            return Err(DecodeError::Truncated);
        }
        bricks.push(BrickRef { origin: *o, shape: ids[bi], off, lane_word, esc });
        off = end;
    }
    Ok(Parsed { dims, d3, bdims, lanes, adaptive, e, mean_fix, delta_g, dl, corr, preds, init, shapes, bricks })
}

fn words(bytes: &[u8], a: usize, b: usize) -> Vec<u16> {
    (a..b).map(|i| u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]])).collect()
}

/// Decode one brick to fixed-point values after corrections (brick-local
/// layout) — the integer reference pipeline.
pub fn decode_brick_fix(ps: &Parsed, bytes: &[u8], bi: usize) -> Result<Vec<i32>, DecodeError> {
    let br = &ps.bricks[bi];
    let sh = &ps.shapes[br.shape];
    let lw: Vec<Vec<u16>> = (0..=ps.lanes).map(|l| words(bytes, br.lane_word[l], br.lane_word[l + 1])).collect();
    let mut sink = DecSink { lanes: lw.iter().map(|w| RansDec::new(w)).collect() };
    let mut q = vec![0i32; sh.len()];
    code_brick(sh, &mut q, &ps.preds, &ps.init, ps.lanes, ps.adaptive, &mut sink);
    let mut c = dequant_brick(sh, &q, &ps.dl);
    inverse_int(&sh.p, &mut c);
    // outlier lane
    let ol = ps.lanes;
    let mut om = vec![MINIT; NOMODEL];
    let n = gamma::<_, true>(&mut sink, ol, &mut om, 0, 0) as usize;
    let mut last: i64 = -1;
    for _ in 0..n {
        let g = gamma::<_, true>(&mut sink, ol, &mut om, 32, 0) as i64;
        let idx = (last + 1 + g) as usize;
        last = idx as i64;
        let neg = sink.raw(ol, 0, 1) == 1;
        let k = gamma::<_, true>(&mut sink, ol, &mut om, 64, 0).wrapping_add(1) as i32;
        let k = if neg { -k } else { k };
        if idx < c.len() {
            c[idx] = c[idx].wrapping_add(k.wrapping_mul(ps.corr));
        }
    }
    if sink.lanes.iter().any(|l| l.overrun()) {
        return Err(DecodeError::Truncated);
    }
    Ok(c)
}

pub(crate) fn dequant_brick(sh: &Shape, q: &[i32], dl: &[u8; NDL]) -> Vec<i32> {
    let [_, ny, nx] = sh.dims;
    let mut c = vec![0i32; q.len()];
    for b in &sh.bands {
        for z in b.start[0]..b.end[0] {
            for y in b.start[1]..b.end[1] {
                for x in b.start[2]..b.end[2] {
                    let i = (z * ny + y) * nx + x;
                    c[i] = dequant(q[i], b, dl);
                }
            }
        }
    }
    c
}

/// Write a decoded brick into the output (f32 bits), cropping to `[lo, hi)`
/// of the field; `out` has the extent of the box.
fn emit_brick(ps: &Parsed, bi: usize, c: &[i32], lo: [usize; 3], hi: [usize; 3], out: &mut [u32]) {
    let br = &ps.bricks[bi];
    let sh = &ps.shapes[br.shape];
    let scale = ps.scale();
    let bx = [0, 1, 2].map(|a| hi[a] - lo[a]);
    let mut esc = br.esc.iter().peekable();
    let [bz, by, bxx] = sh.dims;
    for z in 0..bz {
        for y in 0..by {
            for x in 0..bxx {
                let i = (z * by + y) * bxx + x;
                let mut bits = to_f32_bits(c[i].wrapping_add(ps.mean_fix), scale);
                while let Some(&&(ei, eb)) = esc.peek() {
                    if ei < i {
                        esc.next();
                    } else {
                        if ei == i {
                            bits = eb;
                        }
                        break;
                    }
                }
                let g = [br.origin[0] + z, br.origin[1] + y, br.origin[2] + x];
                if (0..3).all(|a| g[a] >= lo[a] && g[a] < hi[a]) {
                    out[((g[0] - lo[0]) * bx[1] + (g[1] - lo[1])) * bx[2] + (g[2] - lo[2])] = bits;
                }
            }
        }
    }
}

fn intersects(ps: &Parsed, bi: usize, lo: [usize; 3], hi: [usize; 3]) -> bool {
    let br = &ps.bricks[bi];
    let d = ps.shapes[br.shape].dims;
    (0..3).all(|a| br.origin[a] < hi[a] && br.origin[a] + d[a] > lo[a])
}

/// Decode the sub-box `[lo, hi)` (3D coordinates) with `threads` threads;
/// returns `f32` bits in box layout.
pub fn decode_box_bits(bytes: &[u8], lo: [usize; 3], hi: [usize; 3], threads: usize) -> Result<Vec<u32>, DecodeError> {
    let ps = parse(bytes)?;
    decode_box_parsed(&ps, bytes, lo, hi, threads)
}

/// [`decode_box_bits`] on an already parsed stream.
pub fn decode_box_parsed(ps: &Parsed, bytes: &[u8], lo: [usize; 3], hi: [usize; 3], threads: usize) -> Result<Vec<u32>, DecodeError> {
    for a in 0..3 {
        if lo[a] >= hi[a] || hi[a] > ps.d3[a] {
            return Err(DecodeError::Invalid("box"));
        }
    }
    let n: usize = (0..3).map(|a| hi[a] - lo[a]).product();
    let todo: Vec<usize> = (0..ps.bricks.len()).filter(|&b| intersects(ps, b, lo, hi)).collect();
    let mut out = vec![0u32; n];
    if threads <= 1 || todo.len() <= 1 {
        for &bi in &todo {
            let c = decode_brick_fix(ps, bytes, bi)?;
            emit_brick(ps, bi, &c, lo, hi, &mut out);
        }
        return Ok(out);
    }
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    let next = AtomicUsize::new(0);
    let err = Mutex::new(None);
    let outp = SendPtr(out.as_mut_ptr());
    std::thread::scope(|s| {
        for _ in 0..threads.min(todo.len()) {
            let (next, err, todo) = (&next, &err, &todo);
            let outp = &outp;
            s.spawn(move || loop {
                let t = next.fetch_add(1, Ordering::Relaxed);
                if t >= todo.len() {
                    break;
                }
                let bi = todo[t];
                match decode_brick_fix(ps, bytes, bi) {
                    Ok(c) => {
                        // bricks write disjoint cells
                        let o = unsafe { std::slice::from_raw_parts_mut(outp.0, n) };
                        emit_brick(ps, bi, &c, lo, hi, o);
                    }
                    Err(e) => *err.lock().unwrap() = Some(e),
                }
            });
        }
    });
    if let Some(e) = err.into_inner().unwrap() {
        return Err(e);
    }
    Ok(out)
}

struct SendPtr(*mut u32);
unsafe impl Sync for SendPtr {}
unsafe impl Send for SendPtr {}

/// Decode a whole stream to `f32` with `threads` threads.
pub fn decode_f32(bytes: &[u8], threads: usize) -> Result<(Vec<usize>, Vec<f32>), DecodeError> {
    let ps = parse(bytes)?;
    let bits = decode_box_parsed(&ps, bytes, [0; 3], ps.d3, threads)?;
    Ok((ps.dims.clone(), bits.into_iter().map(f32::from_bits).collect()))
}

// ------------------------------------------------------------------ encoder

pub(crate) struct EncBrick {
    pub(crate) shape: usize,
    pub(crate) origin: [usize; 3],
    /// Float coefficients (grid units), brick layout.
    pub(crate) c: Vec<f64>,
}

pub(crate) struct Enc<'a> {
    pub(crate) x: &'a [f32],
    pub(crate) d3: [usize; 3],
    pub(crate) bdims: [usize; 3],
    pub(crate) shapes: Vec<Shape>,
    pub(crate) bricks: Vec<EncBrick>,
    pub(crate) e: i32,
    pub(crate) mean_fix: i32,
    pub(crate) threads: usize,
}

pub(crate) fn par_map<T: Send + Sync, R: Send>(items: &[T], threads: usize, f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    if threads <= 1 || items.len() <= 1 {
        return items.iter().map(f).collect();
    }
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    let next = AtomicUsize::new(0);
    let res: Mutex<Vec<Option<R>>> = Mutex::new((0..items.len()).map(|_| None).collect());
    std::thread::scope(|s| {
        for _ in 0..threads.min(items.len()) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= items.len() {
                    break;
                }
                let r = f(&items[i]);
                res.lock().unwrap()[i] = Some(r);
            });
        }
    });
    res.into_inner().unwrap().into_iter().map(|r| r.unwrap()).collect()
}

impl<'a> Enc<'a> {
    pub(crate) fn new(x: &'a [f32], dims: &[usize], brick: usize, threads: usize) -> Self {
        let d3 = crate::wavelet::as3(dims);
        let bdims = brick_dims(d3, brick);
        let grid = brick_grid(d3, bdims);
        let (shapes, ids) = shapes_for(&grid);
        let n = x.len() as f64;
        let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n;
        let amax = x.iter().fold(0.0f64, |m, &v| m.max((v as f64).abs()));
        let dmax = x.iter().fold(0.0f64, |m, &v| m.max((v as f64 - mean).abs()));
        let need = |v: f64, lim: f64| if v > 0.0 { (v / lim).log2().ceil() as i32 } else { -126 };
        let e = need(amax, (1u32 << 23) as f64).max(need(dmax, (1u32 << 20) as f64)).clamp(-126, 100);
        let sc = 2f64.powi(-e);
        let mean_fix = (mean * sc).round() as i32;
        type Item = (usize, ([usize; 3], [usize; 3]));
        let items: Vec<Item> = ids.iter().copied().zip(grid.iter().copied()).collect();
        let bricks = par_map(&items, threads, |&(sid, (o, bd))| {
            let sh = &shapes[sid];
            let mut c = vec![0.0f64; bd.iter().product()];
            for z in 0..bd[0] {
                for y in 0..bd[1] {
                    for xx in 0..bd[2] {
                        let g = ((o[0] + z) * d3[1] + o[1] + y) * d3[2] + o[2] + xx;
                        c[(z * bd[1] + y) * bd[2] + xx] = x[g] as f64 * sc - mean_fix as f64;
                    }
                }
            }
            forward_97(&sh.p, &mut c);
            EncBrick { shape: sid, origin: o, c }
        });
        Self { x, d3, bdims, shapes, bricks, e, mean_fix, threads }
    }

    pub(crate) fn scale(&self) -> f32 {
        f32::from_bits(((self.e + 127) as u32) << 23)
    }

    /// Quantize all bricks at grid step `dg`; returns q per brick + centroids.
    pub(crate) fn quantize(&self, dg: f64, rho: f64) -> (Vec<Vec<i32>>, [u8; NDL]) {
        let parts = par_map(&self.bricks, self.threads, |b| {
            let sh = &self.shapes[b.shape];
            let [_, ny, nx] = sh.dims;
            let mut q = vec![0i32; b.c.len()];
            let mut sum = [0.0f64; NDL];
            let mut cnt = [0u64; NDL];
            for bd in &sh.bands {
                let inv = bd.w / dg;
                for z in bd.start[0]..bd.end[0] {
                    for y in bd.start[1]..bd.end[1] {
                        for x in bd.start[2]..bd.end[2] {
                            let i = (z * ny + y) * nx + x;
                            let a = b.c[i].abs() * inv;
                            let m = (a + rho).floor();
                            if m >= 1.0 {
                                let m = m.min(1.0e9);
                                let j = bd.class * 3 + (m as usize).min(3) - 1;
                                sum[j] += m - a;
                                cnt[j] += 1;
                                q[i] = if b.c[i] < 0.0 { -(m as i32) } else { m as i32 };
                            }
                        }
                    }
                }
            }
            (q, sum, cnt)
        });
        let mut sum = [0.0f64; NDL];
        let mut cnt = [0u64; NDL];
        for (_, s, c) in &parts {
            for j in 0..NDL {
                sum[j] += s[j];
                cnt[j] += c[j];
            }
        }
        let mut d = [0u8; NDL];
        for j in 0..NDL {
            if cnt[j] > 0 {
                d[j] = ((sum[j] / cnt[j] as f64).clamp(0.0, 0.498) * 512.0).round() as u8;
            }
        }
        (parts.into_iter().map(|p| p.0).collect(), d)
    }

    /// Integer reconstruction per brick (before corrections).
    pub(crate) fn reconstruct(&self, shapes: &[Shape], qs: &[Vec<i32>], dl: &[u8; NDL]) -> Vec<Vec<i32>> {
        let idx: Vec<usize> = (0..self.bricks.len()).collect();
        par_map(&idx, self.threads, |&bi| {
            let sh = &shapes[self.bricks[bi].shape];
            let mut c = dequant_brick(sh, &qs[bi], dl);
            inverse_int(&sh.p, &mut c);
            c
        })
    }

    pub(crate) fn global_index(&self, bi: usize, i: usize) -> usize {
        let b = &self.bricks[bi];
        let d = self.shapes[b.shape].dims;
        let (z, y, x) = (i / (d[1] * d[2]), (i / d[2]) % d[1], i % d[2]);
        ((b.origin[0] + z) * self.d3[1] + b.origin[1] + y) * self.d3[2] + b.origin[2] + x
    }

    /// Squared error and max error of fixed-point reconstruction `cs`.
    pub(crate) fn errors(&self, cs: &[Vec<i32>]) -> (f64, f64) {
        let scale = self.scale();
        let idx: Vec<usize> = (0..cs.len()).collect();
        let parts = par_map(&idx, self.threads, |&bi| {
            let mut se = 0.0;
            let mut mx = 0.0f64;
            for (i, &v) in cs[bi].iter().enumerate() {
                let y = f32::from_bits(to_f32_bits(v.wrapping_add(self.mean_fix), scale)) as f64;
                let e = self.x[self.global_index(bi, i)] as f64 - y;
                se += e * e;
                mx = mx.max(e.abs());
            }
            (se, mx)
        });
        parts.iter().fold((0.0, 0.0), |(a, b), &(s, m)| (a + s, b.max(m)))
    }

    pub(crate) fn shapes_at(&self, dg: f64) -> Vec<Shape> {
        let mut s = self.shapes.clone();
        for sh in s.iter_mut() {
            sh.set_steps(dg);
        }
        s
    }

    pub(crate) fn trial_rmse(&self, dg: f64, rho: f64) -> f64 {
        let shapes = self.shapes_at(dg);
        let (q, d) = self.quantize(dg, rho);
        let cs = self.reconstruct(&shapes, &q, &d);
        (self.errors(&cs).0 / self.x.len() as f64).sqrt()
    }
}

/// Fit predictors per key on the quantized values (pooled over bricks).
fn fit_preds(shapes: &[Shape], bricks: &[EncBrick], qs: &[Vec<i32>], lanes: usize) -> Vec<Option<[i32; NTAP]>> {
    let mut g = vec![[[0.0f64; NTAP]; NTAP]; NKEY];
    let mut h = vec![[0.0f64; NTAP]; NKEY];
    let mut n = vec![0usize; NKEY];
    let zero = [0i32; NTAP];
    let mut obs: Vec<Vec<([i32; NTAP], i32)>> = vec![Vec::new(); NKEY];
    for (bi, b) in bricks.iter().enumerate() {
        let sh = &shapes[b.shape];
        let q = &qs[bi];
        let [_, ny, nx] = sh.dims;
        for bd in &sh.bands {
            for_lines(bd, lanes, |p, _, z, y| {
                let rc = row_ctx(sh, bd, p, z, y);
                let k = bd.pkey[p];
                for x in bd.start[2]..bd.end[2] {
                    let (f, _) = feats(sh, bd, &rc, q, None, z, y, x, &zero);
                    let v = q[(z * ny + y) * nx + x];
                    if v == 0 && f.taps.iter().all(|&t| t == 0) {
                        continue;
                    }
                    n[k] += 1;
                    for i in 0..NTAP {
                        h[k][i] += f.taps[i] as f64 * v as f64;
                        for j in 0..NTAP {
                            g[k][i][j] += f.taps[i] as f64 * f.taps[j] as f64;
                        }
                    }
                    if obs[k].len() < 200_000 {
                        obs[k].push((f.taps, v));
                    }
                }
            });
        }
    }
    (0..NKEY)
        .map(|k| {
            if n[k] < 64 {
                return None;
            }
            let mut a = g[k];
            let mut r = h[k];
            for i in 0..NTAP {
                a[i][i] += 1e-3 * (g[k][i][i] + 1.0);
            }
            for c in 0..NTAP {
                let piv = a[c][c];
                if piv.abs() < 1e-12 {
                    continue;
                }
                for rr in 0..NTAP {
                    if rr != c {
                        let f = a[rr][c] / piv;
                        for kk in 0..NTAP {
                            a[rr][kk] -= f * a[c][kk];
                        }
                        r[rr] -= f * r[c];
                    }
                }
            }
            let mut w = [0i32; NTAP];
            for i in 0..NTAP {
                let wi = if a[i][i].abs() > 1e-12 { r[i] / a[i][i] } else { 0.0 };
                w[i] = (wi * 4096.0).round().clamp(-32767.0, 32767.0) as i32;
            }
            let (mut c0, mut c1) = (0.0f64, 0.0f64);
            for &(t, v) in &obs[k] {
                let mut acc = 0i32;
                for i in 0..NTAP {
                    acc = acc.wrapping_add(w[i].wrapping_mul(t[i]));
                }
                let pr = acc.wrapping_add(2048) >> 12;
                c0 += (1.0 + v.unsigned_abs() as f64).log2();
                c1 += (1.0 + v.wrapping_sub(pr).unsigned_abs() as f64).log2();
            }
            if c1 < 0.98 * c0 && c0 - c1 > 64.0 {
                Some(w)
            } else {
                None
            }
        })
        .collect()
}

/// Encode `x` (row-major `f32`, 1D–3D) as a tc-gpu stream.
pub fn encode_f32(x: &[f32], dims: &[usize], cfg: &GpuConfig) -> (Vec<u8>, GpuStats) {
    encode_f32_mt(x, dims, cfg, 1)
}

/// [`encode_f32`] with `threads` worker threads (output bytes are identical
/// for any thread count).
pub fn encode_f32_mt(x: &[f32], dims: &[usize], cfg: &GpuConfig, threads: usize) -> (Vec<u8>, GpuStats) {
    assert!(cfg.lanes >= 1 && cfg.lanes <= 1024 && cfg.brick >= 1);
    rate_control(x, dims, cfg.brick, cfg.bound, cfg.rho, threads, |enc, dg, mx| encode_at(enc, dims, cfg, dg, mx))
}

/// Shared rate control: bisect the grid step on the measured error of the
/// integer decoder (PSNR/RMSE), or try a ladder of steps (max-abs) and keep
/// the smallest stream.
pub(crate) fn rate_control(
    x: &[f32],
    dims: &[usize],
    brick: usize,
    bound: ErrorBound,
    rho: f64,
    threads: usize,
    encode_at: impl Fn(&Enc, f64, Option<f64>) -> (Vec<u8>, GpuStats),
) -> (Vec<u8>, GpuStats) {
    assert_eq!(dims.iter().product::<usize>(), x.len());
    let enc = Enc::new(x, dims, brick, threads);
    let (lo, hi) = x.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v as f64), b.max(v as f64)));
    let range = hi - lo;
    let g = 2f64.powi(enc.e);
    match bound {
        ErrorBound::MaxAbs(b) => {
            let mut best: Option<(Vec<u8>, GpuStats)> = None;
            for f in [1.0, 1.4, 1.8, 2.2, 2.7, 3.3, 4.0, 5.0] {
                let r = encode_at(&enc, (f * b / g).max(1.0), Some(b));
                if best.as_ref().is_none_or(|(bb, _)| r.0.len() < bb.len()) {
                    best = Some(r);
                }
            }
            best.unwrap()
        }
        ErrorBound::Psnr(_) | ErrorBound::Rmse(_) => {
            let t = match bound {
                ErrorBound::Rmse(r) => r,
                ErrorBound::Psnr(db) => range * 10f64.powf(-db / 20.0),
                _ => unreachable!(),
            }
            .max(f64::MIN_POSITIVE);
            let tg = t / g;
            let (mut lo, mut hi) = (tg * 0.5, tg * 8.0);
            let mut guard = 0;
            while enc.trial_rmse(lo.max(1.0), rho) > t && guard < 40 && lo > 1.0 {
                hi = lo;
                lo *= 0.5;
                guard += 1;
            }
            while enc.trial_rmse(hi, rho) <= t && guard < 80 {
                lo = hi;
                hi *= 2.0;
                guard += 1;
            }
            lo = lo.max(1.0);
            for _ in 0..14 {
                let mid = (lo * hi).sqrt();
                if enc.trial_rmse(mid, rho) <= t {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            encode_at(&enc, lo, None)
        }
    }
}

pub(crate) fn encode_at(enc: &Enc, dims: &[usize], cfg: &GpuConfig, dg: f64, maxabs: Option<f64>) -> (Vec<u8>, GpuStats) {
    let shapes = enc.shapes_at(dg);
    let (qs, dl) = enc.quantize(dg, cfg.rho);
    let fitted = if std::env::var("TC_NOPRED").is_ok() { vec![None; NKEY] } else { fit_preds(&shapes, &enc.bricks, &qs, cfg.lanes) };
    let preds: Vec<[i32; NTAP]> = fitted.iter().map(|p| p.unwrap_or([0; NTAP])).collect();
    let mut cs = enc.reconstruct(&shapes, &qs, &dl);
    let scale = enc.scale();
    let g = 2f64.powi(enc.e);
    // max-abs: corrections k·corr on the grid, then raw escapes
    let corr = maxabs.map(|b| ((2.0 * b / g).floor() as i64).clamp(0, 1 << 30) as i32).unwrap_or(0);
    let nb = enc.bricks.len();
    let mut outl: Vec<Vec<(usize, i32)>> = vec![Vec::new(); nb];
    let mut escs: Vec<Vec<(usize, u32)>> = vec![Vec::new(); nb];
    if let Some(b) = maxabs {
        for bi in 0..nb {
            for i in 0..cs[bi].len() {
                let xv = enc.x[enc.global_index(bi, i)];
                let y = f32::from_bits(to_f32_bits(cs[bi][i].wrapping_add(enc.mean_fix), scale)) as f64;
                let r = xv as f64 - y;
                if r.abs() > b && corr > 0 {
                    let k = (r / g / corr as f64).round().clamp(-(1 << 20) as f64, (1 << 20) as f64) as i32;
                    if k != 0 {
                        cs[bi][i] = cs[bi][i].wrapping_add(k.wrapping_mul(corr));
                        outl[bi].push((i, k));
                    }
                }
                let y = f32::from_bits(to_f32_bits(cs[bi][i].wrapping_add(enc.mean_fix), scale)) as f64;
                if (xv as f64 - y).abs() > b {
                    escs[bi].push((i, xv.to_bits()));
                }
            }
        }
    }
    // priors: pooled decision counts per model
    let idx: Vec<usize> = (0..nb).collect();
    let counts = par_map(&idx, enc.threads, |&bi| {
        let sh = &shapes[enc.bricks[bi].shape];
        let mut cs = CountSink { n: vec![[0; 2]; NMODEL] };
        let mut q = qs[bi].clone();
        code_brick(sh, &mut q, &preds, &vec![MINIT; NMODEL], cfg.lanes, cfg.adaptive, &mut cs);
        cs.n
    });
    let mut n = vec![[0u32; 2]; NMODEL];
    for c in &counts {
        for (a, b) in n.iter_mut().zip(c) {
            a[0] += b[0];
            a[1] += b[1];
        }
    }
    let priors = if std::env::var("TC_NOPRIOR").is_ok() { vec![None; NMODEL] } else { choose_priors(&n, cfg.adaptive) };
    let init = init_of(&priors);
    // entropy-code bricks
    let payloads: Vec<(Vec<u8>, usize, usize)> = par_map(&idx, enc.threads, |&bi| {
        let sh = &shapes[enc.bricks[bi].shape];
        let mut sink = RecSink { syms: vec![Vec::new(); cfg.lanes + 1] };
        let mut q = qs[bi].clone();
        code_brick(sh, &mut q, &preds, &init, cfg.lanes, cfg.adaptive, &mut sink);
        let ol = cfg.lanes;
        let mut om = vec![MINIT; NOMODEL];
        let ov = &outl[bi];
        if maxabs.is_some() {
            gamma::<_, true>(&mut sink, ol, &mut om, 0, ov.len() as u32);
            let mut last: i64 = -1;
            for &(i, k) in ov {
                gamma::<_, true>(&mut sink, ol, &mut om, 32, (i as i64 - last - 1) as u32);
                last = i as i64;
                sink.raw(ol, (k < 0) as u32, 1);
                gamma::<_, true>(&mut sink, ol, &mut om, 64, k.unsigned_abs() - 1);
            }
        }
        let lanes: Vec<Vec<u16>> = sink.syms.iter().map(|s| rans_encode(s)).collect();
        let mut w = ByteWriter::new();
        for l in &lanes {
            w.varint(l.len() as u64);
        }
        let ib = w.buf.len();
        w.varint(escs[bi].len() as u64);
        let mut last = 0usize;
        for (j, &(i, bits)) in escs[bi].iter().enumerate() {
            w.varint(if j == 0 { i } else { i - last } as u64);
            last = i;
            w.bytes(&bits.to_le_bytes());
        }
        let eb = w.buf.len() - ib;
        if w.buf.len() % 2 == 1 {
            w.u8(0);
        }
        let ob = lanes[ol].len() * 2 + eb;
        for l in &lanes {
            for &v in l {
                w.bytes(&v.to_le_bytes());
            }
        }
        (w.buf, ib, ob)
    });
    // header
    let mut w = ByteWriter::new();
    w.bytes(MAGIC);
    w.varint(dims.len() as u64);
    for &d in dims {
        w.varint(d as u64);
    }
    for &d in &enc.bdims {
        w.varint(d as u64);
    }
    w.varint(cfg.lanes as u64);
    w.u8(cfg.adaptive as u8);
    w.svarint(enc.e as i64);
    w.svarint(enc.mean_fix as i64);
    w.f64(dg);
    w.bytes(&dl);
    w.varint(corr as u64);
    for k in used_keys(&shapes) {
        match fitted[k] {
            Some(p) => {
                w.u8(1);
                for &t in &p {
                    w.svarint(t as i64);
                }
            }
            None => w.u8(0),
        }
    }
    write_priors(&mut w, &priors);
    for (p, _, _) in &payloads {
        w.varint(p.len() as u64);
    }
    if w.buf.len() % 2 == 1 {
        w.u8(0);
    }
    let header_bytes = w.buf.len();
    for (p, _, _) in &payloads {
        w.bytes(p);
    }
    let (se, mx) = {
        let (se, _) = enc.errors(&cs);
        // escapes are exact
        let mut se = se;
        let mut mx = 0.0f64;
        for bi in 0..nb {
            let mut e = escs[bi].iter().peekable();
            for i in 0..cs[bi].len() {
                let xv = enc.x[enc.global_index(bi, i)] as f64;
                let y = f32::from_bits(to_f32_bits(cs[bi][i].wrapping_add(enc.mean_fix), scale)) as f64;
                if e.peek().is_some_and(|&&(ei, _)| ei == i) {
                    e.next();
                    se -= (xv - y) * (xv - y);
                    continue;
                }
                mx = mx.max((xv - y).abs());
            }
        }
        (se.max(0.0), mx)
    };
    let n = enc.x.len();
    let st = GpuStats {
        bytes: w.buf.len(),
        bits_per_value: 8.0 * w.buf.len() as f64 / n as f64,
        header_bytes,
        index_bytes: payloads.iter().map(|p| p.1).sum(),
        outlier_bytes: payloads.iter().map(|p| p.2).sum(),
        n_outliers: outl.iter().map(|v| v.len()).sum(),
        n_escapes: escs.iter().map(|v| v.len()).sum(),
        rmse: (se / n as f64).sqrt(),
        max_abs: mx,
        delta: dg * g,
    };
    (w.buf, st)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(d: [usize; 3]) -> Vec<f32> {
        let mut v = Vec::new();
        for i in 0..d[0] {
            for j in 0..d[1] {
                for k in 0..d[2] {
                    let (x, y, z) = (i as f64, j as f64, k as f64);
                    v.push(((0.3 * x).sin() + (0.11 * y * z / 8.0).cos() + 0.01 * x * y - (k % 7) as f64 * 0.01 + 3.0) as f32);
                }
            }
        }
        v
    }

    fn err(x: &[f32], y: &[f32]) -> (f64, f64) {
        let mut se = 0.0;
        let mut mx = 0.0f64;
        for (a, b) in x.iter().zip(y) {
            let e = *a as f64 - *b as f64;
            se += e * e;
            mx = mx.max(e.abs());
        }
        ((se / x.len() as f64).sqrt(), mx)
    }

    #[test]
    fn rans_round_trip() {
        let mut s = RecSink { syms: vec![Vec::new()] };
        let mut st = 12345u32;
        let mut vals = Vec::new();
        for i in 0..20000 {
            st = st.wrapping_mul(1664525).wrapping_add(1013904223);
            if i % 3 == 0 {
                let n = 1 + (st >> 28) % 16;
                vals.push((1, st >> 7, n));
                s.raw(0, st >> 7, n);
            } else {
                let p = 32 + (st >> 16) % 65000;
                let b = (st & 0xFF) < 128;
                vals.push((0, b as u32, p));
                s.bit(0, b, p);
            }
        }
        let w = rans_encode(&s.syms[0]);
        let mut d = RansDec::new(&w);
        for &(k, v, n) in &vals {
            if k == 1 {
                assert_eq!(d.raw(n), v & ((1 << n) - 1));
            } else {
                assert_eq!(d.bit(n) as u32, v);
            }
        }
        assert!(!d.overrun());
    }

    #[test]
    fn psnr_round_trip_and_bricks() {
        let d = [20, 37, 45];
        let x = field(d);
        for brick in [8, 16, 64] {
            for db in [50.0, 90.0] {
                let cfg = GpuConfig { brick, lanes: 4, ..GpuConfig::new(ErrorBound::Psnr(db)) };
                let (bytes, st) = encode_f32(&x, &d, &cfg);
                let (dd, y) = decode_f32(&bytes, 1).unwrap();
                assert_eq!(dd, d.to_vec());
                let (rmse, mx) = err(&x, &y);
                assert!((rmse - st.rmse).abs() <= 1e-12 * rmse, "brick {brick} {db}");
                assert_eq!(mx, st.max_abs);
                let range = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) - x.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                assert!(20.0 * (range as f64 / rmse).log10() >= db - 1e-6);
                let (_, y4) = decode_f32(&bytes, 4).unwrap();
                assert_eq!(y.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), y4.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
                assert_eq!(encode_f32_mt(&x, &d, &cfg, 3).0, bytes, "thread-count independent");
            }
        }
    }

    #[test]
    fn maxabs_and_random_access() {
        let d = [24, 30, 33];
        let x = field(d);
        for rel in [1e-2, 1e-4] {
            let b = rel * 2.5;
            let cfg = GpuConfig { brick: 16, lanes: 8, ..GpuConfig::new(ErrorBound::MaxAbs(b)) };
            let (bytes, st) = encode_f32(&x, &d, &cfg);
            let (_, y) = decode_f32(&bytes, 2).unwrap();
            let (_, mx) = err(&x, &y);
            assert!(mx <= b, "{mx} > {b}");
            assert_eq!(mx, st.max_abs);
            let (lo, hi) = ([3, 5, 7], [21, 29, 20]);
            let sub = decode_box_bits(&bytes, lo, hi, 1).unwrap();
            let mut k = 0;
            for z in lo[0]..hi[0] {
                for yy in lo[1]..hi[1] {
                    for xx in lo[2]..hi[2] {
                        assert_eq!(sub[k], y[(z * d[1] + yy) * d[2] + xx].to_bits());
                        k += 1;
                    }
                }
            }
        }
    }

    #[test]
    fn two_d_and_one_d() {
        let x2: Vec<f32> = (0..70 * 90).map(|i| ((i % 90) as f32 * 0.07).sin() * ((i / 90) as f32 * 0.05).cos()).collect();
        let (b, st) = encode_f32(&x2, &[70, 90], &GpuConfig { brick: 32, lanes: 8, ..GpuConfig::new(ErrorBound::Psnr(70.0)) });
        let (_, y) = decode_f32(&b, 1).unwrap();
        assert!((err(&x2, &y).0 - st.rmse).abs() <= 1e-12 * st.rmse);
        let x1: Vec<f32> = (0..3000).map(|i| (i as f32 * 0.01).sin()).collect();
        let (b, st) = encode_f32(&x1, &[3000], &GpuConfig { brick: 16, lanes: 2, ..GpuConfig::new(ErrorBound::Psnr(70.0)) });
        let (_, y) = decode_f32(&b, 1).unwrap();
        assert!((err(&x1, &y).0 - st.rmse).abs() <= 1e-12 * st.rmse);
    }

    #[test]
    fn lattice_matches_pass_of() {
        for (ez, ey) in [(1, 1), (1, 9), (9, 1), (5, 7), (16, 16), (3, 13)] {
            let b = BandInfo {
                start: [2, 3, 0],
                end: [2 + ez, 3 + ey, 4],
                class: 0,
                parent: None,
                pshift: [0; 3],
                npass: npass_of([2, 3, 0], [2 + ez, 3 + ey, 4]),
                pkey: [0; 8],
                m: 1,
                sh: 1,
                w: 1.0,
                org: [2, 3, 0],
                raster: false,
            };
            let mut got = Vec::new();
            for_lines(&b, 3, |p, l, z, y| got.push((p, l, z, y)));
            let mut want = Vec::new();
            for p in 0..b.npass {
                let mut j = 0;
                for z in b.start[0]..b.end[0] {
                    for y in b.start[1]..b.end[1] {
                        if band_pass(&b, z - b.start[0], y - b.start[1]) == p {
                            want.push((p, j % 3, z, y));
                            j += 1;
                        }
                    }
                }
            }
            assert_eq!(got, want, "{ez}x{ey}");
        }
    }

    #[test]
    fn mulsh_matches_wide() {
        let mut s = 7u64;
        for _ in 0..10000 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let a = (s >> 33) as i32;
            let b = (s as i32) >> 1;
            let sh = 1 + (s >> 58) as u32 % 60;
            let w = (a as i128 * b as i128 + (1i128 << (sh - 1))) >> sh;
            assert_eq!(mulsh(a, b, sh), w as i64 as i32);
        }
    }
}
