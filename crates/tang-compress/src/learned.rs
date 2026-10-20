//! Learned coefficient models: a small MLP predicts each residual's
//! distribution — a two-component **discretized logistic mixture** — from the
//! same causal context features the hand model uses ([`Feat`]: neighbour and
//! parent magnitudes, neighbour signs, prediction magnitude, band class and
//! orientation).
//!
//! # Bit-exact inference
//!
//! Everything from features to coded bits is integer arithmetic:
//!
//! - features are `Q8` fixed-point `log₂(1+v)` computed with integer ops;
//! - the network runs on `i32` weights (`Q8`) and `i64` accumulators, with
//!   `>> 8` rescaling and integer ReLU;
//! - the mixture CDF uses integer tables (logistic in `Q24`, `2^{f/256}` in
//!   `Q16`) built once from IEEE `+ − × / √` only — correctly rounded, so
//!   identical on every conforming backend (no `exp`/`pow` from libm);
//! - the residual is binarized (zero, sign, exp-Golomb bucket, mantissa) and
//!   each binary probability is an integer ratio of CDF masses, clamped and fed
//!   to the range coder (optionally refined by an integer adaptive
//!   probability map).
//!
//! A GPU decoder that reproduces these integer ops reproduces the bits.
//!
//! # Training
//!
//! Float training minimizes the code length (cross-entropy, in bits) of the
//! actual quantized residuals under the mixture, by minibatch Adam
//! ([`tang_optim::Adam`]) with hand-written backprop. `tang-ad`'s scalar tape
//! records one node per scalar op; at ~10⁹ ops per epoch that is far too slow,
//! so the trainer differentiates the fixed MLP + mixture likelihood directly
//! (checked against finite differences in the tests).
//!
//! # Modes
//!
//! - **Pretrained** ([`LearnedSpec::Pretrained`]): a network trained offline on
//!   other fields; the stream carries its 64-bit fingerprint, and the decoder
//!   resolves it from [`register`]ed networks.
//! - **Overfitted** ([`LearnedSpec::Overfit`]): trained at encode time on the
//!   file's own residuals and **shipped in the stream** (entropy-coded `Q8`
//!   weights, counted in the rate) — Cool-chic style.

use crate::abac::{BitModel, Decoder, Encoder, UintModel};
use crate::bitio::{ByteReader, ByteWriter, DecodeError};
use crate::coefcode::{visit, Bits, CoefModel, Feat, Pred, NCLASS};
use crate::wavelet::Pyramid;
use std::sync::{Arc, OnceLock};

/// Input features.
pub const NF: usize = 37;
/// Network outputs: π logit, μ₁, μ₂, log₂ s₁, log₂ s₂.
pub const NO: usize = 5;
const Q: i64 = 256; // Q8
const CDF_BITS: u32 = 24;
const TOT: u64 = 1 << CDF_BITS;
const ZMAX: i64 = 16 * 256; // logistic table covers |z| ≤ 16 (Q8)

/// Integer `Q8` log₂(1 + v).
#[inline]
pub fn lg_q8(v: u32) -> i32 {
    let w = v as u64 + 1;
    let e = w.ilog2();
    let m = if e >= 8 {
        (w >> (e - 8)) & 0xFF
    } else {
        (w << (8 - e)) & 0xFF
    };
    (e as i32) * 256 + m as i32
}

/// Integer feature vector (`Q8`).
pub fn feat_vec(f: &Feat) -> [i32; NF] {
    let slg = |v: i32| {
        if v < 0 {
            -lg_q8(v.unsigned_abs())
        } else {
            lg_q8(v as u32)
        }
    };
    let mut x = [0i32; NF];
    for i in 0..9 {
        x[i] = slg(f.sv[i]);
    }
    for i in 9..crate::coefcode::NSV {
        x[19 + i] = slg(f.sv[i]);
    }
    x[9] = lg_q8(f.pnb as u32);
    x[10] = slg(f.pv);
    // magnitude activity summaries (redundant with the signed inputs, but
    // cheap for a small ReLU net to use)
    let a: u32 = f.nb[..3].iter().map(|&v| v as u32).sum();
    let b: u32 = f.nb[3..].iter().map(|&v| v as u32).sum();
    x[11] = lg_q8(a);
    x[12] = lg_q8(b);
    x[13] = lg_q8(f.par as u32);
    x[14] = f.has_par as i32 * 256;
    x[15 + f.class as usize] = 256;
    for a in 0..3 {
        x[15 + NCLASS + a] = ((f.mask >> a) & 1) as i32 * 256;
    }
    x
}

// ------------------------------------------------------------------ integer tables

fn tables() -> &'static (Vec<u32>, Vec<u32>) {
    static T: OnceLock<(Vec<u32>, Vec<u32>)> = OnceLock::new();
    T.get_or_init(|| {
        // e^{-1/256} by a fixed Taylor series (basic IEEE ops only)
        let x = 1.0f64 / 256.0;
        let mut term = 1.0f64;
        let mut em = 1.0f64;
        for k in 1..20 {
            term = term * (-x) / k as f64;
            em += term;
        }
        // logistic σ(k/256), k = 0..=ZMAX, Q24, by cumulative products
        let mut sig = Vec::with_capacity(ZMAX as usize + 1);
        let mut p = 1.0f64;
        for _ in 0..=ZMAX {
            sig.push(((TOT as f64) / (1.0 + p)).round() as u32);
            p *= em;
        }
        // 2^{f/256}, f = 0..=256, Q16, from 8 square roots of 2
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

/// `σ(z)` with `z` in Q8, result in Q24.
#[inline]
fn sigmoid_q(z: i64) -> u64 {
    let (sig, _) = tables();
    let a = z.unsigned_abs().min(ZMAX as u64) as usize;
    let s = sig[a] as u64;
    if z >= 0 {
        s
    } else {
        TOT - s
    }
}

/// `2^{−ls/256}` in Q16 (saturating), `ls` = log₂ s in Q8.
#[inline]
fn inv_scale_q16(ls: i64) -> u64 {
    let (_, pw) = tables();
    // 2^{-ls/256} = 2^{-(i+1)} · 2^{(256−f)/256}, ls = 256 i + f
    let ls = ls.clamp(-20 * 256, 40 * 256);
    let i = ls.div_euclid(256);
    let f = ls.rem_euclid(256);
    let m = pw[(256 - f) as usize] as u64;
    let sh = i + 1;
    if sh >= 0 {
        m >> sh.min(63)
    } else {
        (m << (-sh).min(20)).min(1 << 40)
    }
}

/// Integer mixture parameters of one coefficient.
#[derive(Clone, Copy, Debug)]
struct Mix {
    pi: u64,
    mu: [i64; 2],
    inv: [u64; 2],
}

impl Mix {
    fn from_out(o: &[i32; NO]) -> Self {
        Self {
            pi: sigmoid_q(o[0] as i64),
            mu: [o[1] as i64, o[2] as i64],
            inv: [inv_scale_q16(o[3] as i64), inv_scale_q16(o[4] as i64)],
        }
    }
    /// `P(r ≤ m)` in Q24.
    #[inline]
    fn cdf(&self, m: i64) -> u64 {
        let t = m.saturating_mul(Q).saturating_add(Q / 2);
        let c = |k: usize| {
            let d = (t as i128 - self.mu[k] as i128) * self.inv[k] as i128;
            let z = (d >> 16).clamp(-(ZMAX as i128) - 1, ZMAX as i128 + 1) as i64;
            sigmoid_q(z)
        };
        (self.pi * c(0) + (TOT - self.pi) * c(1)) >> CDF_BITS
    }
}

#[inline]
fn ratio(num: u64, den: u64) -> u32 {
    if den == 0 {
        return 1 << 15;
    }
    ((num.min(den) as u128 * 65536 / den as u128) as u32).min(65535)
}

/// Code residual `v` under mixture `mx`: zero flag, sign, exp-Golomb bucket
/// (unary), mantissa (bisection) — each an integer ratio of CDF masses.
fn code_mix<B: Bits>(b: &mut B, mx: &Mix, v: i32, apm: &mut Apm, cls: usize) -> i32 {
    let c0 = mx.cdf(0);
    let cm1 = mx.cdf(-1);
    let p_zero = c0.saturating_sub(cm1);
    b.tag(0);
    let nz = apm.code(b, v != 0, ratio(TOT - p_zero.min(TOT), TOT), cls * 4);
    if !nz {
        return 0;
    }
    let neg_mass = cm1;
    let pos_mass = TOT - c0.min(TOT);
    b.tag(3);
    let neg = apm.code(b, v < 0, ratio(neg_mass, neg_mass + pos_mass), cls * 4 + 1);
    let tail = |m: u64| -> u64 {
        if m > (1u64 << 40) {
            return 0;
        }
        if neg {
            mx.cdf(-(m as i64))
        } else {
            TOT - mx.cdf(m as i64 - 1).min(TOT)
        }
    };
    let mag = v.unsigned_abs() as u64;
    let mut k = 0u32;
    b.tag(1);
    while k < 30 {
        let a = tail(1 << k);
        let bb = tail(1 << (k + 1));
        if !apm.code(b, mag >= (1 << (k + 1)), ratio(bb, a), cls * 4 + 2) {
            break;
        }
        k += 1;
    }
    let (mut lo, mut hi) = (1u64 << k, (1u64 << (k + 1)) - 1);
    b.tag(2);
    let mut first = true;
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        let end = tail(hi + 1);
        let up = tail(mid).saturating_sub(end);
        let all = tail(lo).saturating_sub(end);
        let p = ratio(up, all);
        let bit = if first {
            apm.code(b, mag >= mid, p, cls * 4 + 3)
        } else {
            b.bitp(mag >= mid, p)
        };
        first = false;
        if bit {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    if neg {
        -(lo as i32)
    } else {
        lo as i32
    }
}

/// Adaptive probability map (secondary estimation): refines a model
/// probability with an adaptive table over 33 stretch-domain buckets per
/// (class, decision type), linearly interpolated, averaged with the input.
/// Integer only.
#[derive(Clone, Debug)]
struct Apm {
    on: bool,
    t: Vec<BitModel>,
}

const APM_N: usize = 33;

fn stretch_tab() -> &'static Vec<i32> {
    // bucket centres p = σ((i−16)/2), Q16, from the Q24 logistic table
    static S: OnceLock<Vec<i32>> = OnceLock::new();
    S.get_or_init(|| {
        (0..APM_N)
            .map(|i| (sigmoid_q((i as i64 - 16) * 128) >> 8) as i32)
            .collect()
    })
}

impl Apm {
    fn new(on: bool) -> Self {
        let st = stretch_tab();
        let t = (0..(NCLASS + 1) * 4)
            .flat_map(|_| st.iter().map(|&p| BitModel::new(p as u32)))
            .collect();
        Self { on, t }
    }
    #[inline]
    fn code<B: Bits>(&mut self, b: &mut B, bit: bool, p1: u32, ctx: usize) -> bool {
        if !self.on {
            return b.bitp(bit, p1);
        }
        let st = stretch_tab();
        let p = p1 as i32;
        let j = st.partition_point(|&s| s <= p).clamp(1, APM_N - 1) - 1;
        let (a, c) = (st[j], st[j + 1]);
        let w = if c > a {
            ((p - a).clamp(0, c - a) as i64 * 4096 / (c - a) as i64) as u32
        } else {
            0
        };
        let base = ctx * APM_N;
        let pa = self.t[base + j].p1();
        let pc = self.t[base + j + 1].p1();
        let pr = (pa * (4096 - w) + pc * w) >> 12;
        let pf = (pr + p1) / 2;
        let got = b.bitp(bit, pf);
        let jj = if w < 2048 { j } else { j + 1 };
        self.t[base + jj].update(got);
        got
    }
}

// ------------------------------------------------------------------ network

/// An integer network (`Q8` weights, `Q16` biases).
#[derive(Clone, Debug)]
pub struct Net {
    /// Layer sizes, `[NF, h…, NO]`.
    pub sizes: Vec<usize>,
    /// Q8 weights per layer (row-major out × in).
    pub wq: Vec<Vec<i32>>,
    /// Q16 biases per layer.
    pub bq: Vec<Vec<i32>>,
    /// Whether coding uses the adaptive probability map.
    pub apm: bool,
}

impl Net {
    /// Integer forward pass (outputs in Q8).
    pub fn infer(&self, x: &[i32; NF]) -> [i32; NO] {
        let mut a = [0i64; 256];
        let mut b = [0i64; 256];
        for (d, &v) in a.iter_mut().zip(x) {
            *d = v as i64;
        }
        let nl = self.sizes.len() - 1;
        for l in 0..nl {
            let (ni, no) = (self.sizes[l], self.sizes[l + 1]);
            let w = &self.wq[l];
            let src = &a[..ni];
            for o in 0..no {
                let row = &w[o * ni..(o + 1) * ni];
                let mut acc = self.bq[l][o] as i64;
                for (&wi, &ai) in row.iter().zip(src) {
                    acc += wi as i64 * ai;
                }
                let v = (acc + 128) >> 8;
                b[o] = if l + 1 < nl {
                    v.clamp(0, 1 << 22)
                } else {
                    v.clamp(-(1 << 22), 1 << 22)
                };
            }
            std::mem::swap(&mut a, &mut b);
        }
        let mut o = [0i32; NO];
        for k in 0..NO {
            o[k] = a[k] as i32;
        }
        o
    }

    /// 64-bit FNV-1a fingerprint of the integer model.
    pub fn fingerprint(&self) -> u64 {
        self.to_bytes()
            .iter()
            .fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
                (h ^ b as u64).wrapping_mul(0x100_0000_01b3)
            })
    }

    /// Number of parameters.
    pub fn num_params(&self) -> usize {
        self.wq.iter().map(Vec::len).sum::<usize>() + self.bq.iter().map(Vec::len).sum::<usize>()
    }

    fn write_sizes(&self, w: &mut ByteWriter) {
        w.varint(self.sizes.len() as u64);
        for &s in &self.sizes {
            w.varint(s as u64);
        }
        w.u8(self.apm as u8);
    }

    fn read_sizes(r: &mut ByteReader) -> Result<(Vec<usize>, bool), DecodeError> {
        let nl = r.varint()? as usize;
        if !(2..=6).contains(&nl) {
            return Err(DecodeError::Invalid("net layers"));
        }
        let mut sizes = Vec::new();
        for _ in 0..nl {
            let s = r.varint()? as usize;
            if s == 0 || s > 256 {
                return Err(DecodeError::Invalid("net size"));
            }
            sizes.push(s);
        }
        if sizes[0] != NF || sizes[nl - 1] != NO {
            return Err(DecodeError::Invalid("net io"));
        }
        Ok((sizes, r.u8()? != 0))
    }

    /// Serialize (plain varints) — the pretrained-model file format.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut w = ByteWriter::new();
        self.write_sizes(&mut w);
        for l in 0..self.wq.len() {
            for &v in self.wq[l].iter().chain(&self.bq[l]) {
                w.svarint(v as i64);
            }
        }
        w.buf
    }

    /// Parse [`Net::to_bytes`].
    pub fn from_bytes(b: &[u8]) -> Result<Self, DecodeError> {
        let mut r = ByteReader::new(b);
        let (sizes, apm) = Self::read_sizes(&mut r)?;
        let mut wq = Vec::new();
        let mut bq = Vec::new();
        for l in 0..sizes.len() - 1 {
            let mut w = Vec::new();
            for _ in 0..sizes[l] * sizes[l + 1] {
                w.push(r.svarint()?.clamp(-(1 << 30), 1 << 30) as i32);
            }
            let mut b = Vec::new();
            for _ in 0..sizes[l + 1] {
                b.push(r.svarint()?.clamp(-(1 << 30), 1 << 30) as i32);
            }
            wq.push(w);
            bq.push(b);
        }
        Ok(Self { sizes, wq, bq, apm })
    }

    /// Entropy-coded serialization for shipping in a stream.
    fn write_coded(&self, w: &mut ByteWriter) {
        self.write_sizes(w);
        let mut e = Encoder::new();
        let mut mw = UintModel::default();
        let mut mb = UintModel::default();
        for l in 0..self.wq.len() {
            for &v in &self.wq[l] {
                mw.encode(&mut e, zz(v as i64));
            }
            for &v in &self.bq[l] {
                mb.encode(&mut e, zz(v as i64));
            }
        }
        w.block(&e.finish());
    }

    fn read_coded(r: &mut ByteReader) -> Result<Self, DecodeError> {
        let (sizes, apm) = Self::read_sizes(r)?;
        let buf = r.block()?;
        let mut d = Decoder::new(buf);
        let mut mw = UintModel::default();
        let mut mb = UintModel::default();
        let mut wq = Vec::new();
        let mut bq = Vec::new();
        for l in 0..sizes.len() - 1 {
            wq.push(
                (0..sizes[l] * sizes[l + 1])
                    .map(|_| unzz(mw.decode(&mut d)).clamp(-(1 << 30), 1 << 30) as i32)
                    .collect(),
            );
            bq.push(
                (0..sizes[l + 1])
                    .map(|_| unzz(mb.decode(&mut d)).clamp(-(1 << 30), 1 << 30) as i32)
                    .collect(),
            );
        }
        Ok(Self { sizes, wq, bq, apm })
    }
}

fn zz(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

fn unzz(u: u64) -> i64 {
    ((u >> 1) as i64) ^ -((u & 1) as i64)
}

/// The coefficient model driven by a network.
pub struct NetModel {
    net: Arc<Net>,
    cache: Vec<Option<[i32; NO]>>,
    apm: Apm,
}

impl NetModel {
    /// Wrap a network.
    pub fn new(net: Arc<Net>) -> Self {
        let apm = Apm::new(net.apm);
        Self {
            net,
            cache: vec![None; NCLASS * 8 * 2],
            apm,
        }
    }
}

impl CoefModel for NetModel {
    fn code<B: Bits>(&mut self, b: &mut B, f: &Feat, v: i32) -> i32 {
        let quiet = f.sv.iter().all(|&a| a == 0) && f.pm == 0;
        let o = if quiet {
            let k = (f.class as usize * 8 + f.mask as usize) * 2 + f.has_par as usize;
            *self.cache[k].get_or_insert_with(|| self.net.infer(&feat_vec(f)))
        } else {
            self.net.infer(&feat_vec(f))
        };
        code_mix(b, &Mix::from_out(&o), v, &mut self.apm, f.class as usize)
    }
}

// ------------------------------------------------------------------ specs / stream

/// How the learned model is obtained.
#[derive(Clone, Debug)]
pub enum LearnedSpec {
    /// A network trained offline; only its fingerprint goes in the stream.
    Pretrained(Arc<Net>),
    /// Train on this file at encode time and ship the weights.
    Overfit(TrainConfig),
}

const TAG_PRETRAINED: u8 = 1;
const TAG_OVERFIT: u8 = 2;

impl LearnedSpec {
    /// Write the model section (tag + model) and return the model.
    pub fn prepare(&self, p: &Pyramid, q: &[i32], preds: &[Pred], w: &mut ByteWriter) -> NetModel {
        match self {
            LearnedSpec::Pretrained(net) => {
                w.u8(TAG_PRETRAINED);
                w.bytes(&net.fingerprint().to_le_bytes());
                NetModel::new(net.clone())
            }
            LearnedSpec::Overfit(cfg) => {
                let samples = collect(p, q, preds);
                let net = train(&samples, cfg);
                w.u8(TAG_OVERFIT);
                net.write_coded(w);
                NetModel::new(Arc::new(net))
            }
        }
    }
}

/// Registry of pretrained networks the decoder resolves by fingerprint.
fn registry() -> &'static std::sync::Mutex<Vec<Arc<Net>>> {
    static R: OnceLock<std::sync::Mutex<Vec<Arc<Net>>>> = OnceLock::new();
    R.get_or_init(|| std::sync::Mutex::new(Vec::new()))
}

/// Make a pretrained network available to decoders in this process.
pub fn register(net: Arc<Net>) {
    let mut r = registry().lock().unwrap();
    if !r.iter().any(|n| n.fingerprint() == net.fingerprint()) {
        r.push(net);
    }
}

/// A registered pretrained network by fingerprint.
pub fn lookup(fp: u64) -> Option<Arc<Net>> {
    registry()
        .lock()
        .unwrap()
        .iter()
        .find(|n| n.fingerprint() == fp)
        .cloned()
}

/// Read the model section for `tag` (after the tag byte).
pub fn read_model(tag: u8, r: &mut ByteReader) -> Result<NetModel, DecodeError> {
    match tag {
        TAG_PRETRAINED => {
            let fp = u64::from_le_bytes(r.take(8)?.try_into().unwrap());
            let reg = registry().lock().unwrap();
            let net = reg
                .iter()
                .find(|n| n.fingerprint() == fp)
                .ok_or(DecodeError::Invalid("pretrained model not registered"))?;
            Ok(NetModel::new(net.clone()))
        }
        TAG_OVERFIT => Ok(NetModel::new(Arc::new(Net::read_coded(r)?))),
        _ => Err(DecodeError::Invalid("model tag")),
    }
}

/// Parse a CLI model argument: `hand`, `overfit[:h1[,h2]]`, or
/// `pretrained:PATH` (the network is also registered for decoding).
pub fn parse_model_arg(s: &str) -> crate::wavcodec::ModelChoice {
    use crate::wavcodec::ModelChoice;
    if s == "hand" {
        return ModelChoice::Hand;
    }
    if let Some(rest) = s.strip_prefix("overfit") {
        let mut cfg = TrainConfig::overfit();
        if let Some(h) = rest.strip_prefix(':') {
            cfg.hidden = h.split(',').map(|v| v.parse().unwrap()).collect();
        }
        return ModelChoice::Learned(LearnedSpec::Overfit(cfg));
    }
    if let Some(path) = s.strip_prefix("pretrained:") {
        let net = Arc::new(
            Net::from_bytes(&std::fs::read(path).expect("read model")).expect("parse model"),
        );
        register(net.clone());
        return ModelChoice::Learned(LearnedSpec::Pretrained(net));
    }
    panic!("unknown model {s}")
}

// ------------------------------------------------------------------ training

/// One training sample.
#[derive(Clone, Copy, Debug)]
pub struct Sample {
    /// Integer features.
    pub x: [i32; NF],
    /// Residual.
    pub r: i32,
}

/// Collect `(features, residual)` for every coefficient of `q`, in coding
/// order (the features the coder will see).
pub fn collect(p: &Pyramid, q: &[i32], preds: &[Pred]) -> Vec<Sample> {
    let mut out = Vec::with_capacity(q.len());
    visit(p, q, preds, |f, r| out.push(Sample { x: feat_vec(f), r }));
    out
}

/// Trainer settings.
#[derive(Clone, Debug)]
pub struct TrainConfig {
    /// Hidden layer widths.
    pub hidden: Vec<usize>,
    /// Passes.
    pub epochs: usize,
    /// Samples per pass (a seeded random subset when fewer than available).
    pub per_epoch: usize,
    /// Minibatch size.
    pub batch: usize,
    /// Adam learning rate (cosine-decayed to 10%).
    pub lr: f64,
    /// Use the adaptive probability map at coding time.
    pub apm: bool,
    /// Seed.
    pub seed: u64,
}

impl TrainConfig {
    /// Small network for per-file overfitting (its weights are shipped).
    pub fn overfit() -> Self {
        Self {
            hidden: vec![16],
            epochs: 4,
            per_epoch: 200_000,
            batch: 256,
            lr: 4e-3,
            apm: true,
            seed: 1,
        }
    }
    /// Larger network for offline pretraining.
    pub fn pretrain() -> Self {
        Self {
            hidden: vec![32, 16],
            epochs: 10,
            per_epoch: 500_000,
            batch: 256,
            lr: 3e-3,
            apm: true,
            seed: 7,
        }
    }
}

struct FNet {
    sizes: Vec<usize>,
    off: Vec<(usize, usize)>, // per layer: (W offset, b offset)
    n: usize,
}

impl FNet {
    fn new(sizes: Vec<usize>) -> Self {
        let mut off = Vec::new();
        let mut n = 0;
        for l in 0..sizes.len() - 1 {
            off.push((n, n + sizes[l] * sizes[l + 1]));
            n += sizes[l] * sizes[l + 1] + sizes[l + 1];
        }
        Self { sizes, off, n }
    }
    fn forward(&self, th: &[f64], x: &[f64]) -> Vec<Vec<f64>> {
        let mut acts = vec![x.to_vec()];
        let nl = self.sizes.len() - 1;
        for l in 0..nl {
            let (ni, no) = (self.sizes[l], self.sizes[l + 1]);
            let (wo, bo) = self.off[l];
            let a = acts.last().unwrap();
            let mut out = vec![0.0; no];
            for o in 0..no {
                let row = &th[wo + o * ni..wo + (o + 1) * ni];
                let mut s = th[bo + o];
                for i in 0..ni {
                    s += row[i] * a[i];
                }
                out[o] = if l + 1 < nl { s.max(0.0) } else { s };
            }
            acts.push(out);
        }
        acts
    }
    fn backward(&self, th: &[f64], acts: &[Vec<f64>], gout: &[f64], g: &mut [f64]) {
        let nl = self.sizes.len() - 1;
        let mut delta = gout.to_vec();
        for l in (0..nl).rev() {
            let (ni, no) = (self.sizes[l], self.sizes[l + 1]);
            let (wo, bo) = self.off[l];
            let a = &acts[l];
            let mut dprev = vec![0.0; ni];
            for o in 0..no {
                let d = delta[o];
                if d == 0.0 {
                    continue;
                }
                g[bo + o] += d;
                let row = wo + o * ni;
                for i in 0..ni {
                    g[row + i] += d * a[i];
                    dprev[i] += d * th[row + i];
                }
            }
            if l > 0 {
                for i in 0..ni {
                    if acts[l][i] <= 0.0 {
                        dprev[i] = 0.0;
                    }
                }
            }
            delta = dprev;
        }
    }
}

#[inline]
fn sigm(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

/// Code length (nats) of residual `r` under float outputs `o`, and dL/do.
fn nll_grad(o: &[f64], r: i32) -> (f64, [f64; NO]) {
    let pi = sigm(o[0]);
    let mut d = [0.0; 2];
    let mut dmu = [0.0; 2];
    let mut dls = [0.0; 2];
    for c in 0..2 {
        let mu = o[1 + c];
        let ls = o[3 + c].clamp(-20.0, 40.0);
        let s = ls.exp2();
        let (za, zb) = ((r as f64 + 0.5 - mu) / s, (r as f64 - 0.5 - mu) / s);
        let (a, b) = (sigm(za), sigm(zb));
        d[c] = a - b;
        let (ga, gb) = (a * (1.0 - a), b * (1.0 - b));
        dmu[c] = -(ga - gb) / s;
        dls[c] = -(ga * za - gb * zb) * std::f64::consts::LN_2;
    }
    let p = (pi * d[0] + (1.0 - pi) * d[1]).max(1e-12);
    let g = [
        -(d[0] - d[1]) * pi * (1.0 - pi) / p,
        -pi * dmu[0] / p,
        -(1.0 - pi) * dmu[1] / p,
        -pi * dls[0] / p,
        -(1.0 - pi) * dls[1] / p,
    ];
    (-p.ln(), g)
}

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s >> 11
}

/// Train a network on samples (deterministic for a given seed); returns the
/// quantized integer network.
pub fn train(samples: &[Sample], cfg: &TrainConfig) -> Net {
    let mut sizes = vec![NF];
    sizes.extend(&cfg.hidden);
    sizes.push(NO);
    let fnet = FNet::new(sizes.clone());
    let mut seed = cfg.seed;
    let mut th = vec![0.0f64; fnet.n];
    for l in 0..sizes.len() - 1 {
        let (wo, bo) = fnet.off[l];
        let scale = (6.0 / sizes[l] as f64).sqrt() * 0.5;
        for v in &mut th[wo..bo] {
            *v = ((lcg(&mut seed) % 2_000_001) as f64 / 1_000_000.0 - 1.0) * scale;
        }
    }
    let n = samples.len();
    let mean_abs = samples
        .iter()
        .map(|s| s.r.unsigned_abs() as f64)
        .sum::<f64>()
        / n.max(1) as f64;
    let (_, bo) = fnet.off[sizes.len() - 2];
    th[bo + 3] = -1.5;
    th[bo + 4] = (mean_abs + 0.5).log2();
    if n == 0 {
        return quantize_net(&fnet, &th, cfg.apm);
    }
    let mut adam = tang_optim::Adam::new(cfg.lr);
    let mut params = tang_la::DVec::from_vec(th);
    let mut idx: Vec<usize> = (0..n).collect();
    let m = cfg.per_epoch.min(n);
    let steps_total = cfg.epochs * m.div_ceil(cfg.batch);
    let mut step = 0usize;
    for _ in 0..cfg.epochs {
        for i in (1..n).rev() {
            let j = (lcg(&mut seed) % (i as u64 + 1)) as usize;
            idx.swap(i, j);
        }
        for chunk in idx[..m].chunks(cfg.batch) {
            let th = params.as_slice().to_vec();
            let mut g = vec![0.0; fnet.n];
            for &i in chunk {
                let x = samples[i].x.map(|v| v as f64 / 256.0);
                let acts = fnet.forward(&th, &x);
                let (_, go) = nll_grad(acts.last().unwrap(), samples[i].r);
                fnet.backward(&th, &acts, &go, &mut g);
            }
            let inv = 1.0 / chunk.len() as f64;
            for v in g.iter_mut() {
                *v *= inv;
            }
            let t = step as f64 / steps_total.max(1) as f64;
            adam.lr = cfg.lr * (0.1 + 0.45 * (1.0 + (std::f64::consts::PI * t).cos()));
            adam.step(&mut params, &tang_la::DVec::from_vec(g));
            step += 1;
        }
    }
    quantize_net(&fnet, params.as_slice(), cfg.apm)
}

fn quantize_net(f: &FNet, th: &[f64], apm: bool) -> Net {
    let mut wq = Vec::new();
    let mut bq = Vec::new();
    for l in 0..f.sizes.len() - 1 {
        let (wo, bo) = f.off[l];
        let no = f.sizes[l + 1];
        wq.push(
            th[wo..bo]
                .iter()
                .map(|v| (v * 256.0).round().clamp(-1e9, 1e9) as i32)
                .collect(),
        );
        bq.push(
            th[bo..bo + no]
                .iter()
                .map(|v| (v * 65536.0).round().clamp(-1e9, 1e9) as i32)
                .collect(),
        );
    }
    Net {
        sizes: f.sizes.clone(),
        wq,
        bq,
        apm,
    }
}

/// Mean ideal code length (bits/sample) of `samples` under the float view of
/// an integer net (for reporting; the coder's actual bits include binarization
/// and probability rounding).
pub fn mean_bits_float(net: &Net, samples: &[Sample]) -> f64 {
    let f = FNet::new(net.sizes.clone());
    let mut th = vec![0.0; f.n];
    for l in 0..net.sizes.len() - 1 {
        let (wo, bo) = f.off[l];
        for (i, &v) in net.wq[l].iter().enumerate() {
            th[wo + i] = v as f64 / 256.0;
        }
        for (i, &v) in net.bq[l].iter().enumerate() {
            th[bo + i] = v as f64 / 65536.0;
        }
    }
    let s: f64 = samples
        .iter()
        .map(|s| {
            nll_grad(
                f.forward(&th, &s.x.map(|v| v as f64 / 256.0))
                    .last()
                    .unwrap(),
                s.r,
            )
            .0
        })
        .sum();
    s / samples.len().max(1) as f64 / std::f64::consts::LN_2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coefcode::{code_bands, fit_predictors};
    use crate::wavelet::{forward_97, MAX_LEVELS};

    #[test]
    fn lg_is_monotone_and_exact_at_powers() {
        assert_eq!(lg_q8(0), 0);
        assert_eq!(lg_q8(1), 256);
        assert_eq!(lg_q8(3), 512);
        let mut last = -1;
        for v in 0..5000 {
            let l = lg_q8(v);
            assert!(l >= last);
            last = l;
        }
    }

    #[test]
    fn tables_are_sane() {
        assert_eq!(sigmoid_q(0), TOT / 2);
        assert!(sigmoid_q(ZMAX) > TOT - 4);
        assert_eq!(inv_scale_q16(0), 65536);
        assert_eq!(inv_scale_q16(256), 32768);
        assert_eq!(inv_scale_q16(-256), 131072);
    }

    #[test]
    fn cdf_monotone_and_saturates() {
        let mx = Mix::from_out(&[100, -300, 900, -200, 700]);
        let mut last = 0;
        for m in -2000..2000 {
            let c = mx.cdf(m);
            assert!(c >= last && c <= TOT);
            last = c;
        }
        assert!(mx.cdf(-1_000_000) < 1000 && mx.cdf(1_000_000) > TOT - 1000);
    }

    #[test]
    fn gradient_matches_finite_differences() {
        let o = [0.3, -0.7, 2.1, 0.4, 1.9];
        for r in [-5, 0, 1, 7] {
            let (_, g) = nll_grad(&o, r);
            for k in 0..NO {
                let mut a = o;
                let mut b = o;
                a[k] += 1e-6;
                b[k] -= 1e-6;
                let fd = (nll_grad(&a, r).0 - nll_grad(&b, r).0) / 2e-6;
                assert!(
                    (fd - g[k]).abs() < 1e-4 * (1.0 + fd.abs()),
                    "r={r} k={k}: {fd} vs {}",
                    g[k]
                );
            }
        }
    }

    #[test]
    fn overfit_model_round_trips_bit_exact() {
        let d = [10, 24, 30];
        let p = Pyramid::new(d, MAX_LEVELS);
        let mut x: Vec<f64> = (0..p.len())
            .map(|i| ((i as f64) * 0.021).sin() * 50.0 + ((i * 7) % 13) as f64)
            .collect();
        forward_97(&p, &mut x);
        let mut q: Vec<i32> = x.iter().map(|v| (v / 0.9).round() as i32).collect();
        q[3] = 2_000_000; // far tail
        let preds = fit_predictors(&p, &q);
        let cfg = TrainConfig {
            hidden: vec![8],
            epochs: 2,
            per_epoch: 4000,
            batch: 128,
            lr: 5e-3,
            apm: true,
            seed: 3,
        };
        let mut w = ByteWriter::new();
        let mut m = LearnedSpec::Overfit(cfg.clone()).prepare(&p, &q, &preds, &mut w);
        let mut e = Encoder::new();
        let mut qq = q.clone();
        code_bands(&p, &mut qq, &preds, &mut e, &mut m);
        let bytes = e.finish();
        let mut r = ByteReader::new(&w.buf);
        let tag = r.u8().unwrap();
        let mut m2 = read_model(tag, &mut r).unwrap();
        let mut d2 = Decoder::new(&bytes);
        let mut out = vec![0i32; q.len()];
        code_bands(&p, &mut out, &preds, &mut d2, &mut m2);
        assert_eq!(out, q);
        let mut w2 = ByteWriter::new();
        LearnedSpec::Overfit(cfg).prepare(&p, &q, &preds, &mut w2);
        assert_eq!(w.buf, w2.buf, "deterministic training");
    }
}
