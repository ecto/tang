//! KV-cache codec: append-friendly, error-bounded coding of attention K/V.
//!
//! Input is one layer's cache, `[heads][tokens][head_dim]` f32. The design
//! follows the statistics of real caches (SmolLM2 dumps, see
//! `bench/kv_ppl.py`) rather than treating the cache as a generic 3-D array:
//!
//! - **K is coded pre-RoPE.** The rotation is undone with the stream's
//!   `theta` and absolute token position, coded, and re-applied on decode.
//!   Pre-RoPE, most of K's energy is a per-channel constant; post-RoPE that
//!   constant becomes a position-dependent oscillation. RoPE is a per-pair
//!   rotation, so a pre-RoPE pair error of `e` is a post-RoPE pair error of
//!   `e` too, and every per-channel quantity is shared by the two channels of
//!   a RoPE pair (`c`, `c + hd/2`).
//! - **Prediction runs along the token axis only**, per (head, channel):
//!   closed-loop DPCM around the channel mean with a per-channel AR(1)
//!   coefficient, uniform step. Closed loop means every reconstructed value is
//!   within `step/2` of the (pre-RoPE) input — the error bound is exact.
//! - **Attention sinks** (the first `sink` absolute positions) are coded with
//!   a `sink_div`-times finer step: they carry most of the attention mass, so
//!   their error is amplified far beyond their share of the values.
//! - **Context models are per head**: each residual is coded with adaptive
//!   binary models selected by (head, channel spread class, magnitude of the
//!   same channel's previous residual). Models start from a discrete-Laplace
//!   prior of the class's spread, so a fresh block learns nothing from zero.
//! - **Blocks of `block` tokens are independent** (prediction and models
//!   reset), so new tokens are coded without touching the past and any token
//!   range decodes by decoding only the blocks that cover it. The stream
//!   statistics (channel means, AR coefficients, classes) are fixed from the
//!   first chunk the encoder sees (the prefill).
//! - **GPU layout** (`gpu`, header flag bit 3): reconstruction is integer
//!   (values are `i32` multiples of the unit `u = step / sink_div`, stored as
//!   `f32`; the AR prediction is `mu + (rho·(prev − mu)) >> 4`), and every
//!   block is coded as one independent arithmetic-coded substream per head.
//!   The `(block, head)` substreams are the GPU decoder's lanes
//!   ([`crate::kvgpu`]), and its output is bit-identical to
//!   [`KvStream::decode_range`]: the only float operation is the final
//!   `(q as f32) * u`. The layout codes the coded domain directly (the caller
//!   passes pre-RoPE K and applies RoPE itself).

use crate::abac::{BitModel, Decoder, Encoder, ONE};
use crate::bitio::{ByteReader, ByteWriter, DecodeError};

const MAGIC: &[u8; 4] = b"TKV1";
const NCLASS: usize = 48;
const NACT: usize = 3;
const UBITS: usize = 24;

/// Which half of the cache a stream holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Keys (RoPE applied by the model).
    K,
    /// Values.
    V,
}

/// Encoder settings.
#[derive(Clone, Debug)]
pub struct KvParams {
    /// K or V.
    pub kind: Kind,
    /// Number of KV heads.
    pub heads: usize,
    /// Head dimension (even).
    pub hd: usize,
    /// RoPE base (HF-Llama half-split convention).
    pub theta: f64,
    /// Code K before RoPE (undo, code, redo). Ignored for V.
    pub pre_rope: bool,
    /// Quantizer step in data units (max abs error is `step / 2`).
    pub step: f64,
    /// Number of leading absolute positions treated as attention sinks.
    pub sink: usize,
    /// Step divisor for sink positions.
    pub sink_div: u32,
    /// Tokens per independent block.
    pub block: usize,
    /// Fit a per-channel AR(1) predictor (else predict the channel mean).
    pub ar: bool,
    /// GPU layout: integer reconstruction, one substream per (block, head).
    pub gpu: bool,
    /// GPU layout: channels per substream (even, divides `hd`; 0 = `hd`).
    /// Each block then has `heads × hd / lane_ch` substreams; lane `g` of a
    /// head holds the RoPE pairs `g·lane_ch/2 ..` (channels `c` and
    /// `c + hd/2` together) so the decoder can rotate on write.
    pub lane_ch: usize,
}

impl KvParams {
    /// Defaults for a cache with `heads` × `hd` per token (SmolLM2 RoPE base).
    pub fn new(kind: Kind, heads: usize, hd: usize, step: f64) -> Self {
        Self {
            kind,
            heads,
            hd,
            theta: 1e5,
            pre_rope: true,
            step,
            sink: 4,
            sink_div: 16,
            block: 128,
            ar: true,
            gpu: false,
            lane_ch: 0,
        }
    }
    /// Channels per GPU substream.
    pub fn lane_width(&self) -> usize {
        if self.lane_ch == 0 {
            self.hd
        } else {
            self.lane_ch
        }
    }
    /// GPU substreams per (block, head).
    pub fn lane_groups(&self) -> usize {
        self.hd / self.lane_width()
    }
    fn rotated(&self) -> bool {
        self.kind == Kind::K && self.pre_rope && !self.gpu
    }
    /// GPU-layout unit: every reconstruction is an `i32` multiple of it.
    pub fn unit(&self) -> f32 {
        (self.step / self.sink_div as f64) as f32
    }
}

/// Per-stream statistics, fixed from the prefill and stored in the header.
#[derive(Clone, Debug)]
struct Stats {
    /// Channel mean as an integer multiple of the channel step.
    mu: Vec<i64>,
    /// AR(1) coefficient in sixteenths, 0..=15.
    rho: Vec<u8>,
    /// Spread class: round(4 log2(residual sd / step)) + 8, clamped.
    class: Vec<u8>,
}

/// Rotate `[heads][hd]` token vectors at absolute position `pos` by RoPE
/// (`sign = 1`) or its inverse (`sign = -1`).
fn rope_token(x: &mut [f64], heads: usize, hd: usize, theta: f64, pos: usize, sign: f64) {
    let half = hd / 2;
    for i in 0..half {
        let f = pos as f64 / theta.powf(2.0 * i as f64 / hd as f64);
        let (s, c) = f.sin_cos();
        let s = sign * s;
        for h in 0..heads {
            let a = x[h * hd + i];
            let b = x[h * hd + half + i];
            x[h * hd + i] = a * c - b * s;
            x[h * hd + half + i] = b * c + a * s;
        }
    }
}

/// Initial adaptive models for one context, from a discrete Laplace prior
/// with residual sd `sd` (in steps).
#[derive(Clone)]
struct Ctx {
    zero: BitModel,
    unary: [BitModel; UBITS],
    mant: [BitModel; UBITS],
}

impl Ctx {
    fn prior(sd: f64) -> Self {
        let b = (sd / std::f64::consts::SQRT_2).max(0.05);
        let r = (-1.0 / b).exp();
        let p = |x: f64| ((x * ONE as f64) as u32).clamp(64, ONE - 64);
        let pz = 1.0 - (-0.5 / b).exp();
        let mut unary = [BitModel::default(); UBITS];
        let mut mant = [BitModel::default(); UBITS];
        for i in 0..UBITS {
            let ri = r.powf((1u64 << i.min(40)) as f64);
            unary[i] = BitModel::new(p(ri));
            let rh = if i == 0 {
                0.5
            } else {
                r.powf((1u64 << (i - 1)) as f64)
            };
            mant[i] = BitModel::new(p(rh / (1.0 + rh)));
        }
        Self {
            zero: BitModel::new(p(pz)),
            unary,
            mant,
        }
    }
}

fn class_sd(class: u8) -> f64 {
    2f64.powf((class as f64 - 8.0) / 4.0)
}

/// Fresh context set for one block.
fn fresh_models(heads: usize) -> Vec<Ctx> {
    let mut v = Vec::with_capacity(heads * NCLASS * NACT);
    for _ in 0..heads {
        for c in 0..NCLASS {
            for a in 0..NACT {
                // previous residual magnitude 0 / 1 / 2+ shifts the prior spread
                let sd = class_sd(c as u8) * [0.8, 1.0, 1.4][a];
                v.push(Ctx::prior(sd));
            }
        }
    }
    v
}

fn put(e: &mut Encoder, m: &mut Ctx, q: i64) {
    e.encode(q == 0, &mut m.zero);
    if q == 0 {
        return;
    }
    e.encode_p(q < 0, ONE / 2);
    let w = q.unsigned_abs(); // >= 1
    let k = 63 - w.leading_zeros() as usize;
    for i in 0..k {
        e.encode(true, &mut m.unary[i.min(UBITS - 1)]);
    }
    e.encode(false, &mut m.unary[k.min(UBITS - 1)]);
    for i in (0..k).rev() {
        let bit = (w >> i) & 1 == 1;
        if i + 1 == k {
            e.encode(bit, &mut m.mant[k]);
        } else {
            e.encode_p(bit, ONE / 2);
        }
    }
}

fn get(d: &mut Decoder, m: &mut Ctx) -> i64 {
    if d.decode(&mut m.zero) {
        return 0;
    }
    let neg = d.decode_p(ONE / 2);
    let mut k = 0;
    while k < 62 && d.decode(&mut m.unary[k.min(UBITS - 1)]) {
        k += 1;
    }
    let mut w = 1u64;
    for i in (0..k).rev() {
        let bit = if i + 1 == k {
            d.decode(&mut m.mant[k])
        } else {
            d.decode_p(ONE / 2)
        };
        w = (w << 1) | bit as u64;
    }
    if neg {
        -(w as i64)
    } else {
        w as i64
    }
}

/// Shared closed-loop state of one block (encoder and decoder run the same
/// arithmetic, so their reconstructions agree bit for bit).
struct Lane<'a> {
    p: &'a KvParams,
    s: &'a Stats,
    step: &'a [f64],
    prev: Vec<f64>,
    act: Vec<u8>,
    models: Vec<Ctx>,
    first: bool,
}

impl<'a> Lane<'a> {
    fn new(p: &'a KvParams, s: &'a Stats, step: &'a [f64]) -> Self {
        let n = p.heads * p.hd;
        Self {
            p,
            s,
            step,
            prev: vec![0.0; n],
            act: vec![0; n],
            models: fresh_models(p.heads),
            first: true,
        }
    }
    #[inline]
    fn pred(&self, j: usize) -> f64 {
        let mu = self.s.mu[j] as f64 * self.step[j];
        if self.first {
            mu
        } else {
            mu + self.s.rho[j] as f64 / 16.0 * (self.prev[j] - mu)
        }
    }
    #[inline]
    fn ctx(&self, j: usize) -> usize {
        let h = j / self.p.hd;
        (h * NCLASS + self.s.class[j] as usize) * NACT + self.act[j] as usize
    }
    #[inline]
    fn set(&mut self, j: usize, q: i64, st: f64) -> f64 {
        let y = self.pred(j) + q as f64 * st;
        self.act[j] = q.unsigned_abs().min(2) as u8;
        y
    }
}

fn steps(p: &KvParams) -> Vec<f64> {
    vec![p.step; p.heads * p.hd]
}

fn tok_step(p: &KvParams, pos: usize) -> f64 {
    if pos < p.sink {
        1.0 / p.sink_div as f64
    } else {
        1.0
    }
}

/// Channel of lane-local index `i` in channel group `g` (see [`KvParams::lane_ch`]).
#[inline]
pub fn lane_chan(hd: usize, w: usize, g: usize, i: usize) -> usize {
    let base = g * w / 2;
    if i < w / 2 {
        base + i
    } else {
        hd / 2 + base + i - w / 2
    }
}

/// Encode tokens `[pos0, pos0 + n)` (row-major `[n][heads*hd]`, already in the
/// coded domain) as one independent block.
fn encode_block(
    p: &KvParams,
    s: &Stats,
    step: &[f64],
    y: &[f64],
    pos0: usize,
    n: usize,
) -> Vec<u8> {
    let m = p.heads * p.hd;
    let mut lane = Lane::new(p, s, step);
    let mut e = Encoder::new();
    for t in 0..n {
        let ts = tok_step(p, pos0 + t);
        let mut row = vec![0.0; m];
        for j in 0..m {
            let st = step[j] * ts;
            let q = ((y[t * m + j] - lane.pred(j)) / st).round() as i64;
            let ci = lane.ctx(j);
            put(&mut e, &mut lane.models[ci], q);
            row[j] = lane.set(j, q, st);
        }
        lane.prev = row;
        lane.first = false;
    }
    e.finish()
}

fn decode_block(
    p: &KvParams,
    s: &Stats,
    step: &[f64],
    bytes: &[u8],
    pos0: usize,
    n: usize,
    out: &mut [f64],
) {
    let m = p.heads * p.hd;
    let mut lane = Lane::new(p, s, step);
    let mut d = Decoder::new(bytes);
    for t in 0..n {
        let ts = tok_step(p, pos0 + t);
        for j in 0..m {
            let st = step[j] * ts;
            let ci = lane.ctx(j);
            let q = get(&mut d, &mut lane.models[ci]);
            out[t * m + j] = lane.set(j, q, st);
        }
        lane.prev.copy_from_slice(&out[t * m..(t + 1) * m]);
        lane.first = false;
    }
}

/// GPU layout: encode tokens `[pos0, pos0 + n)` as one substream per head.
fn encode_block_gpu(p: &KvParams, s: &Stats, y: &[f64], pos0: usize, n: usize) -> Vec<Vec<u8>> {
    let (h, d) = (p.heads, p.hd);
    let m = h * d;
    let u = p.unit() as f64;
    let (w, ng) = (p.lane_width(), p.lane_groups());
    (0..h * ng)
        .map(|lane| {
            let (hh, g) = (lane / ng, lane % ng);
            let mut models = fresh_models(1);
            let mut e = Encoder::new();
            let mut prev = vec![0i64; w];
            let mut act = vec![0u8; w];
            for t in 0..n {
                let ts = if pos0 + t < p.sink {
                    1
                } else {
                    p.sink_div as i64
                };
                for c in 0..w {
                    let j = hh * d + lane_chan(d, w, g, c);
                    let mu = s.mu[j];
                    let pred = if t == 0 {
                        mu
                    } else {
                        mu + ((s.rho[j] as i64 * (prev[c] - mu)) >> 4)
                    };
                    let q = ((y[t * m + j] / u - pred as f64) / ts as f64).round() as i64;
                    let ci = s.class[j] as usize * NACT + act[c] as usize;
                    put(&mut e, &mut models[ci], q);
                    let r = pred + q * ts;
                    assert!(
                        r.abs() < 1 << 24,
                        "kv gpu layout: value out of i24 range (step too small)"
                    );
                    prev[c] = r;
                    act[c] = q.unsigned_abs().min(2) as u8;
                }
            }
            e.finish()
        })
        .collect()
}

/// GPU layout decode of one (block, head, channel group) substream into
/// `out[t][c]` (`n × hd`, f32; only the group's channels are written). This is
/// the reference the GPU kernel matches.
#[allow(clippy::too_many_arguments)]
fn decode_lane_gpu(
    p: &KvParams,
    s: &Stats,
    hh: usize,
    g: usize,
    bytes: &[u8],
    pos0: usize,
    n: usize,
    out: &mut [f32],
) {
    let d = p.hd;
    let u = p.unit();
    let mut models = fresh_models(1);
    let mut dec = Decoder::new(bytes);
    let w = p.lane_width();
    let mut prev = vec![0i64; w];
    let mut act = vec![0u8; w];
    for t in 0..n {
        let ts = if pos0 + t < p.sink {
            1
        } else {
            p.sink_div as i64
        };
        for c in 0..w {
            let cc = lane_chan(d, w, g, c);
            let j = hh * d + cc;
            let mu = s.mu[j];
            let pred = if t == 0 {
                mu
            } else {
                mu + ((s.rho[j] as i64 * (prev[c] - mu)) >> 4)
            };
            let ci = s.class[j] as usize * NACT + act[c] as usize;
            let q = get(&mut dec, &mut models[ci]);
            let r = pred + q * ts;
            prev[c] = r;
            act[c] = q.unsigned_abs().min(2) as u8;
            out[t * d + cc] = (r as i32 as f32) * u;
        }
    }
}

/// `[heads][n][hd]` f32 → token-major `[n][heads*hd]` f64 in the coded
/// domain (pre-RoPE for K when enabled), positions starting at `pos0`.
fn to_coded(p: &KvParams, x: &[f32], n: usize, pos0: usize) -> Vec<f64> {
    let (h, d) = (p.heads, p.hd);
    assert_eq!(x.len(), h * n * d, "shape mismatch");
    let mut y = vec![0.0; n * h * d];
    for t in 0..n {
        for hh in 0..h {
            for c in 0..d {
                y[t * h * d + hh * d + c] = x[(hh * n + t) * d + c] as f64;
            }
        }
        if p.rotated() {
            rope_token(
                &mut y[t * h * d..(t + 1) * h * d],
                h,
                d,
                p.theta,
                pos0 + t,
                -1.0,
            );
        }
    }
    y
}

fn from_coded(p: &KvParams, y: &mut [f64], n: usize, pos0: usize) -> Vec<f32> {
    let (h, d) = (p.heads, p.hd);
    let mut x = vec![0f32; n * h * d];
    for t in 0..n {
        if p.rotated() {
            rope_token(
                &mut y[t * h * d..(t + 1) * h * d],
                h,
                d,
                p.theta,
                pos0 + t,
                1.0,
            );
        }
        for hh in 0..h {
            for c in 0..d {
                x[(hh * n + t) * d + c] = y[t * h * d + hh * d + c] as f32;
            }
        }
    }
    x
}

fn fit_stats(p: &KvParams, y: &[f64], n: usize, step: &[f64]) -> Stats {
    let m = p.heads * p.hd;
    let half = p.hd / 2;
    let mut mean = vec![0.0; m];
    for t in 0..n {
        for j in 0..m {
            mean[j] += y[t * m + j];
        }
    }
    mean.iter_mut().for_each(|v| *v /= n.max(1) as f64);
    let (mut c0, mut c1) = (vec![0.0; m], vec![0.0; m]);
    for t in 0..n {
        for j in 0..m {
            let r = y[t * m + j] - mean[j];
            c0[j] += r * r;
            if t > 0 {
                c1[j] += r * (y[(t - 1) * m + j] - mean[j]);
            }
        }
    }
    let mut rho = vec![0u8; m];
    let mut var = vec![0.0; m];
    for j in 0..m {
        let a = if p.ar && c0[j] > 0.0 {
            c1[j] / c0[j]
        } else {
            0.0
        };
        let a = (a * 16.0).round().clamp(0.0, 15.0);
        rho[j] = a as u8;
        let a = a / 16.0;
        // residual variance of r_t - a r_{t-1}
        var[j] = (c0[j] / n.max(1) as f64) * (1.0 - a * a).max(0.05);
    }
    let mut class = vec![0u8; m];
    for j in 0..m {
        let pair = if p.kind == Kind::K {
            j - (j % p.hd) + (j % p.hd + half) % p.hd
        } else {
            j
        };
        let v = if p.kind == Kind::K {
            0.5 * (var[j] + var[pair])
        } else {
            var[j]
        };
        let r = v.sqrt() / step[j];
        class[j] = ((4.0 * r.max(1e-9).log2()).round() + 8.0).clamp(0.0, NCLASS as f64 - 1.0) as u8;
    }
    let mu = if p.gpu {
        let u = p.unit() as f64;
        (0..m).map(|j| (mean[j] / u).round() as i64).collect()
    } else {
        (0..m).map(|j| (mean[j] / step[j]).round() as i64).collect()
    };
    Stats { mu, rho, class }
}

/// Append-friendly encoder for one layer's K or V stream.
pub struct KvEncoder {
    /// Settings (fixed for the stream).
    pub params: KvParams,
    stats: Stats,
    step: Vec<f64>,
    blocks: Vec<Vec<u8>>,
    /// GPU layout: per sealed block, one substream per head.
    gblocks: Vec<Vec<Vec<u8>>>,
    /// Pending tokens of the unsealed tail block, coded domain.
    tail: Vec<f64>,
    tokens: usize,
}

impl KvEncoder {
    /// Start a stream from the prefill `[heads][n][hd]`; its statistics fix
    /// the header.
    pub fn new(params: KvParams, prefill: &[f32], n: usize) -> Self {
        assert!(params.hd.is_multiple_of(2) && params.block > 0 && n > 0);
        let step = steps(&params);
        let y = to_coded(&params, prefill, n, 0);
        let stats = fit_stats(&params, &y, n, &step);
        if params.gpu {
            assert!(
                !(params.kind == Kind::K && params.pre_rope),
                "gpu layout codes pre-RoPE K given by the caller (set pre_rope = false)"
            );
        }
        let mut e = Self {
            params,
            stats,
            step,
            blocks: Vec::new(),
            gblocks: Vec::new(),
            tail: Vec::new(),
            tokens: 0,
        };
        e.push_coded(y, n);
        e
    }

    /// Append `n` new tokens `[heads][n][hd]`. Sealed blocks are never touched.
    pub fn append(&mut self, x: &[f32], n: usize) {
        let y = to_coded(&self.params, x, n, self.tokens);
        self.push_coded(y, n);
    }

    fn push_coded(&mut self, y: Vec<f64>, n: usize) {
        let m = self.params.heads * self.params.hd;
        let b = self.params.block;
        self.tail.extend_from_slice(&y);
        self.tokens += n;
        while self.tail.len() >= b * m {
            let pos0 = self.sealed_blocks() * b;
            if self.params.gpu {
                let blk = encode_block_gpu(&self.params, &self.stats, &self.tail[..b * m], pos0, b);
                self.gblocks.push(blk);
            } else {
                let blk = encode_block(
                    &self.params,
                    &self.stats,
                    &self.step,
                    &self.tail[..b * m],
                    pos0,
                    b,
                );
                self.blocks.push(blk);
            }
            self.tail.drain(..b * m);
        }
    }

    /// Tokens seen so far.
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// Number of sealed (final) blocks.
    pub fn sealed_blocks(&self) -> usize {
        self.blocks.len() + self.gblocks.len()
    }

    /// Serialize: header, sealed blocks, and the tail coded as a short block.
    /// Only the tail (< `block` tokens) is coded here.
    pub fn to_bytes(&self) -> Vec<u8> {
        let p = &self.params;
        let mut w = ByteWriter::new();
        w.bytes(MAGIC);
        w.u8(match p.kind {
            Kind::K => 0,
            Kind::V => 1,
        } | (p.pre_rope as u8) << 1
            | (p.ar as u8) << 2
            | (p.gpu as u8) << 3
            | ((p.gpu && p.lane_groups() > 1) as u8) << 4);
        w.varint(p.heads as u64);
        w.varint(p.hd as u64);
        w.f64(p.theta);
        w.f64(p.step);
        w.varint(p.sink as u64);
        w.varint(p.sink_div as u64);
        w.varint(p.block as u64);
        w.varint(self.tokens as u64);
        if p.gpu && p.lane_groups() > 1 {
            w.varint(p.lane_width() as u64);
        }
        for j in 0..p.heads * p.hd {
            w.svarint(self.stats.mu[j]);
            w.u8(self.stats.class[j] | if p.ar { 0 } else { 0x80 });
            if p.ar {
                // two channels' rho per byte would save ~1 B/channel; keep simple
                w.u8(self.stats.rho[j]);
            }
        }
        for blk in &self.blocks {
            w.block(blk);
        }
        for blk in &self.gblocks {
            blk.iter().for_each(|sub| w.block(sub));
        }
        let m = p.heads * p.hd;
        let nt = self.tail.len() / m;
        if nt > 0 {
            let pos0 = self.sealed_blocks() * p.block;
            if p.gpu {
                encode_block_gpu(p, &self.stats, &self.tail, pos0, nt)
                    .iter()
                    .for_each(|sub| w.block(sub));
            } else {
                w.block(&encode_block(
                    p,
                    &self.stats,
                    &self.step,
                    &self.tail,
                    pos0,
                    nt,
                ));
            }
        }
        w.buf
    }
}

/// Parsed stream: header plus block payload slices.
pub struct KvStream<'a> {
    /// Settings recovered from the header.
    pub params: KvParams,
    /// Total tokens.
    pub tokens: usize,
    stats: Stats,
    blocks: Vec<&'a [u8]>,
    /// Byte offset of each entry of `blocks` in the parsed buffer.
    offsets: Vec<usize>,
}

impl<'a> KvStream<'a> {
    /// Parse a stream produced by [`KvEncoder::to_bytes`].
    pub fn parse(bytes: &'a [u8]) -> Result<Self, DecodeError> {
        let mut r = ByteReader::new(bytes);
        if r.take(4)? != MAGIC {
            return Err(DecodeError::Invalid("not a TKV1 stream"));
        }
        let f = r.u8()?;
        let kind = if f & 1 == 0 { Kind::K } else { Kind::V };
        let heads = r.varint()? as usize;
        let hd = r.varint()? as usize;
        let theta = r.f64()?;
        let step = r.f64()?;
        let sink = r.varint()? as usize;
        let sink_div = r.varint()? as u32;
        let block = r.varint()? as usize;
        let tokens = r.varint()? as usize;
        let lane_ch = if f & 16 != 0 { r.varint()? as usize } else { 0 };
        if !hd.is_multiple_of(2) || block == 0 || heads == 0 || sink_div == 0 {
            return Err(DecodeError::Invalid("bad header"));
        }
        if lane_ch != 0 && (!lane_ch.is_multiple_of(2) || !hd.is_multiple_of(lane_ch)) {
            return Err(DecodeError::Invalid("bad lane width"));
        }
        let ar = f & 4 != 0;
        let gpu = f & 8 != 0;
        let params = KvParams {
            kind,
            heads,
            hd,
            theta,
            pre_rope: f & 2 != 0,
            step,
            sink,
            sink_div,
            block,
            ar,
            gpu,
            lane_ch,
        };
        let m = heads * hd;
        let (mut mu, mut rho, mut class) = (
            Vec::with_capacity(m),
            Vec::with_capacity(m),
            Vec::with_capacity(m),
        );
        for _ in 0..m {
            mu.push(r.svarint()?);
            class.push(r.u8()? & 0x7f);
            rho.push(if ar { r.u8()?.min(15) } else { 0 });
        }
        if class.iter().any(|&c| c as usize >= NCLASS) {
            return Err(DecodeError::Invalid("class out of range"));
        }
        let nb = tokens.div_ceil(block) * if gpu { heads * params.lane_groups() } else { 1 };
        let mut blocks = Vec::with_capacity(nb);
        let mut offsets = Vec::with_capacity(nb);
        for _ in 0..nb {
            let b = r.block()?;
            offsets.push(b.as_ptr() as usize - bytes.as_ptr() as usize);
            blocks.push(b);
        }
        if gpu && mu.iter().any(|&v| v.abs() >= 1 << 24) {
            return Err(DecodeError::Invalid("mean out of range"));
        }
        Ok(Self {
            params,
            tokens,
            stats: Stats { mu, rho, class },
            blocks,
            offsets,
        })
    }

    /// Decode tokens `[t0, t1)` as `[heads][t1-t0][hd]`, decoding only the
    /// blocks that cover the range.
    pub fn decode_range(&self, t0: usize, t1: usize) -> Vec<f32> {
        assert!(t0 <= t1 && t1 <= self.tokens);
        let p = &self.params;
        if p.gpu {
            return self.decode_range_gpu(t0, t1);
        }
        let m = p.heads * p.hd;
        let step = steps(p);
        let (b0, b1) = (t0 / p.block, t1.div_ceil(p.block));
        let mut y = Vec::with_capacity((b1 - b0) * p.block * m);
        let mut buf = vec![0.0; p.block * m];
        for b in b0..b1 {
            let pos0 = b * p.block;
            let n = p.block.min(self.tokens - pos0);
            decode_block(
                p,
                &self.stats,
                &step,
                self.blocks[b],
                pos0,
                n,
                &mut buf[..n * m],
            );
            y.extend_from_slice(&buf[..n * m]);
        }
        let off = t0 - b0 * p.block;
        let mut sel = y[off * m..(off + t1 - t0) * m].to_vec();
        from_coded(p, &mut sel, t1 - t0, t0)
    }

    fn decode_range_gpu(&self, t0: usize, t1: usize) -> Vec<f32> {
        let p = &self.params;
        let (h, d, nt) = (p.heads, p.hd, t1 - t0);
        let mut x = vec![0f32; h * nt * d];
        let mut buf = vec![0f32; p.block * d];
        for b in t0 / p.block..t1.div_ceil(p.block) {
            let pos0 = b * p.block;
            let n = p.block.min(self.tokens - pos0);
            let ng = p.lane_groups();
            for hh in 0..h {
                for g in 0..ng {
                    decode_lane_gpu(
                        p,
                        &self.stats,
                        hh,
                        g,
                        self.blocks[(b * h + hh) * ng + g],
                        pos0,
                        n,
                        &mut buf[..n * d],
                    );
                }
                for t in pos0.max(t0)..(pos0 + n).min(t1) {
                    let dst = (hh * nt + t - t0) * d;
                    x[dst..dst + d].copy_from_slice(&buf[(t - pos0) * d..(t - pos0 + 1) * d]);
                }
            }
        }
        x
    }

    /// GPU layout tables for [`crate::kvgpu`]: per channel `mu` (`i32` bits),
    /// `rho`, `class`, and per (block, head) lane `(byte offset, byte length,
    /// first token, tokens)`.
    pub fn gpu_tables(&self) -> Option<GpuTables> {
        let p = &self.params;
        if !p.gpu {
            return None;
        }
        let ng = p.lane_groups();
        let lanes = (0..self.blocks.len())
            .map(|i| {
                let (b, hh, g) = (i / (p.heads * ng), i / ng % p.heads, i % ng);
                let pos0 = b * p.block;
                [
                    self.offsets[i] as u32,
                    self.blocks[i].len() as u32,
                    (pos0) as u32,
                    p.block.min(self.tokens - pos0) as u32,
                    hh as u32,
                    g as u32,
                ]
            })
            .collect();
        Some(GpuTables {
            mu: self.stats.mu.iter().map(|&v| v as i32).collect(),
            rho: self.stats.rho.iter().map(|&v| v as u32).collect(),
            class: self.stats.class.iter().map(|&v| v as u32).collect(),
            lanes,
        })
    }
}

/// Per-stream tables the GPU decoder uploads (see [`KvStream::gpu_tables`]).
pub struct GpuTables {
    /// Channel mean in units of [`KvParams::unit`].
    pub mu: Vec<i32>,
    /// AR(1) coefficient in sixteenths.
    pub rho: Vec<u32>,
    /// Spread class.
    pub class: Vec<u32>,
    /// `[byte offset, byte length, first token, tokens, head, channel group]`
    /// per lane.
    pub lanes: Vec<[u32; 6]>,
}

/// Initial adaptive-model table shared by every GPU-layout stream:
/// `[class][act][49]` with entry 0 the zero flag, 1..=24 the unary models and
/// 25..=48 the mantissa models, each `p | p << 16` (fast, slow).
pub fn gpu_prior_table() -> Vec<u32> {
    let mut v = Vec::with_capacity(NCLASS * NACT * (1 + 2 * UBITS));
    for c in fresh_models(1) {
        for m in std::iter::once(&c.zero)
            .chain(c.unary.iter())
            .chain(c.mant.iter())
        {
            v.push(m.packed());
        }
    }
    v
}

/// Models per context in [`gpu_prior_table`].
pub const GPU_CTX_MODELS: usize = 1 + 2 * UBITS;
/// Contexts per lane (classes × previous-magnitude bins).
pub const GPU_CTXS: usize = NCLASS * NACT;

/// One-shot encode of `[heads][n][hd]` (prefill = everything).
pub fn encode(params: KvParams, x: &[f32], n: usize) -> Vec<u8> {
    KvEncoder::new(params, x, n).to_bytes()
}

/// Decode a whole stream → (params, tokens, `[heads][tokens][hd]`).
pub fn decode(bytes: &[u8]) -> Result<(KvParams, usize, Vec<f32>), DecodeError> {
    let s = KvStream::parse(bytes)?;
    let x = s.decode_range(0, s.tokens);
    Ok((s.params, s.tokens, x))
}

/// Rate-targeted mode: the step whose one-shot stream is closest to (and not
/// above) `bpv` bits per value, by bisection on log(step).
pub fn step_for_bpv(mut params: KvParams, x: &[f32], n: usize, bpv: f64) -> f64 {
    let vals = x.len() as f64;
    let rms = (x.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / vals)
        .sqrt()
        .max(1e-30);
    let (mut lo, mut hi) = ((rms * 1e-5).ln(), (rms * 10.0).ln());
    for _ in 0..30 {
        let mid = 0.5 * (lo + hi);
        params.step = mid.exp();
        let b = 8.0 * encode(params.clone(), x, n).len() as f64 / vals;
        if b > bpv {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    hi.exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth(h: usize, n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        let mut rnd = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        let mut x = vec![0f32; h * n * d];
        for hh in 0..h {
            for c in 0..d {
                let mu = 3.0 * rnd();
                let mut prev = 0.0;
                for t in 0..n {
                    prev = 0.5 * prev + rnd() * (1.0 + c as f64 / 16.0);
                    x[(hh * n + t) * d + c] = (mu + prev) as f32;
                }
            }
        }
        x
    }

    fn pre(p: &KvParams, x: &[f32], n: usize) -> Vec<f64> {
        to_coded(p, x, n, 0)
    }

    #[test]
    fn error_bound_and_append_and_ranges() {
        let (h, n, d) = (3, 300, 16);
        for kind in [Kind::K, Kind::V] {
            let x = synth(h, n, d, 7);
            let mut p = KvParams::new(kind, h, d, 0.05);
            p.block = 32;
            // one-shot
            let one = encode(p.clone(), &x, n);
            let (_, nt, y) = decode(&one).unwrap();
            assert_eq!(nt, n);
            let (a, b) = (pre(&p, &x, n), pre(&p, &y, n));
            for t in 0..n {
                let lim = p.step / 2.0 * tok_step(&p, t) + 1e-4;
                for j in 0..h * d {
                    let e = (a[t * h * d + j] - b[t * h * d + j]).abs();
                    assert!(e <= lim, "{kind:?} t={t} j={j} err {e} > {lim}");
                }
            }
            // prefill 100 then token-at-a-time append: same stats source differs,
            // so compare against its own decode and check sealed blocks are stable
            let slice = |t0: usize, t1: usize| -> Vec<f32> {
                let mut v = Vec::new();
                for hh in 0..h {
                    v.extend_from_slice(&x[(hh * n + t0) * d..(hh * n + t1) * d]);
                }
                v
            };
            let mut e = KvEncoder::new(p.clone(), &slice(0, 100), 100);
            let mut sealed: Vec<Vec<u8>> = Vec::new();
            for t in 100..n {
                e.append(&slice(t, t + 1), 1);
                if e.sealed_blocks() > sealed.len() {
                    sealed = e.blocks.clone();
                }
                assert_eq!(&e.blocks[..sealed.len()], &sealed[..]);
            }
            let bytes = e.to_bytes();
            let s = KvStream::parse(&bytes).unwrap();
            let full = s.decode_range(0, n);
            for (t0, t1) in [(0, 1), (31, 33), (100, 300), (250, 300), (64, 64)] {
                let r = s.decode_range(t0, t1);
                for hh in 0..h {
                    for t in t0..t1 {
                        for c in 0..d {
                            assert_eq!(
                                r[(hh * (t1 - t0) + t - t0) * d + c],
                                full[(hh * n + t) * d + c]
                            );
                        }
                    }
                }
            }
            let (a, b) = (pre(&p, &x, n), pre(&p, &full, n));
            let me = a
                .iter()
                .zip(&b)
                .map(|(u, v)| (u - v).abs())
                .fold(0.0, f64::max);
            assert!(me <= p.step / 2.0 + 1e-4);
        }
    }

    #[test]
    fn gpu_layout_bound_ranges_append() {
        let (h, n, d) = (3, 300, 16);
        for (kind, lc) in [(Kind::K, 0), (Kind::V, 0), (Kind::K, 4), (Kind::V, 2)] {
            let x = synth(h, n, d, 11);
            let mut p = KvParams::new(kind, h, d, 0.05);
            p.block = 32;
            p.pre_rope = false;
            p.gpu = true;
            p.lane_ch = lc;
            let bytes = encode(p.clone(), &x, n);
            let s = KvStream::parse(&bytes).unwrap();
            let y = s.decode_range(0, n);
            for hh in 0..h {
                for t in 0..n {
                    let lim = p.step / 2.0 * tok_step(&p, t) + 1e-5;
                    for c in 0..d {
                        let i = (hh * n + t) * d + c;
                        assert!(((x[i] - y[i]) as f64).abs() <= lim, "{kind:?} t={t} c={c}");
                    }
                }
            }
            for (t0, t1) in [(0, 1), (31, 33), (250, 300)] {
                let r = s.decode_range(t0, t1);
                for hh in 0..h {
                    for t in t0..t1 {
                        assert_eq!(
                            r[(hh * (t1 - t0) + t - t0) * d..][..d],
                            y[(hh * n + t) * d..][..d]
                        );
                    }
                }
            }
            let tb = s.gpu_tables().unwrap();
            assert_eq!(tb.lanes.len(), n.div_ceil(32) * h * p.lane_groups());
        }
    }

    #[test]
    fn rate_target() {
        let (h, n, d) = (2, 256, 32);
        let x = synth(h, n, d, 3);
        let p = KvParams::new(Kind::V, h, d, 1.0);
        let st = step_for_bpv(p.clone(), &x, n, 3.0);
        let b = 8.0 * encode(KvParams { step: st, ..p }, &x, n).len() as f64 / x.len() as f64;
        assert!(b <= 3.0 && b > 2.7, "bpv {b}");
    }
}
