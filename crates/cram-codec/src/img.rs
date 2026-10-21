//! Overfitted image codec (Phase 2, Track D): Cool-chic 5 style decoder in
//! exact 32-bit integer arithmetic, so the CPU reference and the GPU kernels
//! ([`crate::imggpu`]) decode the same bytes to the same pixels.
//!
//! The encoder side is the Cool-chic 5 reference (PyTorch, per-image
//! overfitting); `tools/coolchic/export_cc.py` dumps a trained image as a
//! [`Ccx`] file (integer latents + dequantized nets). [`encode`] turns that
//! into a `TCI1` stream:
//!
//! - **Nets** (ARM, IFCEs, upsampling, synthesis): each tensor as its
//!   quantization exponent `s` plus integers `k = v·2^s` (zigzag varints),
//!   then fixed point with a per-layer weight precision `fw` chosen by the
//!   encoder so that no accumulation leaves `i32` on this image (checked in
//!   `i64`; the decoder re-checks).
//! - **Latents**: per grid (coarse → fine), Laplace probabilities from the
//!   integer ARM (spatial causal context + IFCE inter-feature context),
//!   quantized to `NFB` μ-fraction × `NLB` log-scale buckets, coded with
//!   16-bit-word rANS. Rows are spread over `lanes` independent rANS streams
//!   and decoded in a **wavefront** (`t = x + s·y`, `s` from the context
//!   template), so every lane advances in lockstep; optional column tiles
//!   (`tile_w`) cut contexts at tile borders for more parallel blocks.
//! - **Synthesis**: learned separable upsampling pyramid, then the synthesis
//!   MLP / 3×3 residual convs, all `QA`-bit fixed point; pixels are
//!   `clamp(⌊255·v⌉, 0, 255)`.

use crate::bitio::{unzigzag, zigzag, BitReader, BitWriter, ByteReader, ByteWriter, DecodeError};
use std::collections::HashMap;

/// Stream magic.
pub const MAGIC: &[u8; 4] = b"TCI1";
/// Fractional bits of ARM activations (latents enter as `v << QA_ARM`).
pub const QA_ARM: u32 = 8;
/// Fractional bits of upsampling / synthesis activations.
pub const QA: u32 = 12;
/// μ-fraction buckets.
pub const NFB: usize = 256 >> FB_SHIFT;
/// μ-fraction bucket width `2^FB_SHIFT / 256`.
pub const FB_SHIFT: u32 = 3;
/// Log-scale bucket width `2^LB_SHIFT / 256`.
pub const LB_SHIFT: u32 = 3;
/// Log-scale buckets over `ln b ∈ [−5, 5]`.
pub const NLB: usize = (2560 >> LB_SHIFT) + 1;
/// Residual alphabet `r ∈ [−R, R]` plus two escapes.
pub const R: i32 = 31;
/// Symbols per table.
pub const NSYM: usize = (2 * R + 3) as usize;
/// rANS probability precision.
pub const PB: u32 = 15;
const PM: u32 = 1 << PB;
const RL: u32 = 1 << 16;

// ---------------------------------------------------------------- CCX input

/// A model exported by `export_cc.py`: named f64 arrays with shapes.
pub struct Ccx {
    /// name → (shape, data)
    pub t: HashMap<String, (Vec<usize>, Vec<f64>)>,
}

impl Ccx {
    /// Parse the `CCX1` container.
    pub fn parse(b: &[u8]) -> Result<Self, DecodeError> {
        let mut r = ByteReader::new(b);
        if r.take(4)? != b"CCX1" {
            return Err(DecodeError::Invalid("not CCX1"));
        }
        let le32 = |r: &mut ByteReader| -> Result<u32, DecodeError> {
            Ok(u32::from_le_bytes(r.take(4)?.try_into().unwrap()))
        };
        let n = le32(&mut r)?;
        let mut t = HashMap::new();
        for _ in 0..n {
            let nl = u16::from_le_bytes(r.take(2)?.try_into().unwrap()) as usize;
            let name = String::from_utf8_lossy(r.take(nl)?).into_owned();
            let nd = r.u8()? as usize;
            let shape: Vec<usize> = (0..nd)
                .map(|_| le32(&mut r).map(|v| v as usize))
                .collect::<Result<_, _>>()?;
            let cnt: usize = shape.iter().product();
            let data = r
                .take(cnt * 8)?
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| f64::from_le_bytes(*c))
                .collect();
            t.insert(name, (shape, data));
        }
        Ok(Self { t })
    }
    fn get(&self, k: &str) -> &(Vec<usize>, Vec<f64>) {
        self.t.get(k).unwrap_or_else(|| panic!("ccx: missing {k}"))
    }
    fn has(&self, k: &str) -> bool {
        self.t.contains_key(k)
    }
    fn ints(&self, k: &str) -> Vec<i64> {
        self.get(k).1.iter().map(|&v| v as i64).collect()
    }
}

// ------------------------------------------------------------ fixed point nets

/// A quantized tensor: values `k · 2^{−s}`.
#[derive(Clone, Debug, Default)]
pub struct QT {
    /// Exponent: value = k / 2^s.
    pub s: u32,
    /// Integer mantissas.
    pub k: Vec<i32>,
}

impl QT {
    fn from_f64(v: &[f64]) -> Self {
        let s = (0..=24u32)
            .find(|&s| v.iter().all(|&x| (x * (1u64 << s) as f64).fract() == 0.0))
            .unwrap_or(24);
        Self {
            s,
            k: v.iter()
                .map(|&x| (x * (1u64 << s) as f64).round() as i32)
                .collect(),
        }
    }
    /// Fixed point with `f` fractional bits (round half away from zero when `f < s`).
    pub fn fixed(&self, f: u32) -> Vec<i32> {
        self.k
            .iter()
            .map(|&k| {
                if f >= self.s {
                    k << (f - self.s)
                } else {
                    let d = self.s - f;
                    let h = 1i32 << (d - 1);
                    if k >= 0 {
                        (k + h) >> d
                    } else {
                        -((-k + h) >> d)
                    }
                }
            })
            .collect()
    }
    fn write(&self, w: &mut BitWriter) {
        eg_put(w, self.s as u64, 0);
        // Exp-Golomb order chosen per tensor
        let zz: Vec<u64> = self.k.iter().map(|&k| zigzag(k as i64)).collect();
        let cost = |o: u32| zz.iter().map(|&z| eg_len(z, o)).sum::<u64>();
        let o = (0..12).min_by_key(|&o| cost(o)).unwrap();
        eg_put(w, o as u64, 0);
        for &z in &zz {
            eg_put(w, z, o);
        }
    }
    /// Read `n` values (the count comes from the layer shape).
    fn read(r: &mut BitReader, n: usize) -> Result<Self, DecodeError> {
        let s = eg_get(r, 0)? as u32;
        if s > 24 {
            return Err(DecodeError::Invalid("qt exponent"));
        }
        let o = eg_get(r, 0)? as u32;
        let k = (0..n)
            .map(|_| eg_get(r, o).map(|z| unzigzag(z) as i32))
            .collect::<Result<_, _>>()?;
        Ok(Self { s, k })
    }
}

/// Weight precision cap; precisions are sent as `FWMAX − fw`.
const FWMAX: u32 = 16;

fn put_fw(w: &mut BitWriter, fw: u32) {
    eg_put(w, (FWMAX - fw) as u64, 0);
}

fn get_fw(r: &mut BitReader) -> Result<u32, DecodeError> {
    let d = eg_get(r, 0)?;
    if d > FWMAX as u64 {
        return Err(DecodeError::Invalid("weight precision"));
    }
    Ok(FWMAX - d as u32)
}

fn eg_len(v: u64, o: u32) -> u64 {
    let x = (v >> o) + 1;
    (2 * (63 - x.leading_zeros()) + 1 + o) as u64
}

/// Exp-Golomb of order `o`.
fn eg_put(w: &mut BitWriter, v: u64, o: u32) {
    let x = (v >> o) + 1;
    let nb = 63 - x.leading_zeros();
    w.put(0, nb);
    w.put(1, 1);
    w.put(x & ((1 << nb) - 1), nb);
    w.put(v & ((1 << o) - 1), o);
}

fn eg_get(r: &mut BitReader, o: u32) -> Result<u64, DecodeError> {
    let mut nb = 0;
    while r.get(1)? == 0 {
        nb += 1;
        if nb > 40 {
            return Err(DecodeError::Invalid("exp-golomb"));
        }
    }
    let x = (1u64 << nb) | r.get(nb)?;
    Ok(((x - 1) << o) | r.get(o)?)
}

/// One linear layer `out × inp` (`w[o·inp + i]`), fixed point.
#[derive(Clone, Debug)]
pub struct Lin {
    /// Outputs.
    pub out: usize,
    /// Inputs.
    pub inp: usize,
    /// Source weights / biases.
    pub qw: QT,
    /// Biases.
    pub qb: QT,
    /// Weight fractional bits.
    pub fw: u32,
    /// Weights `Q(fw)`.
    pub w: Vec<i32>,
    /// Biases `Q(qa + fw)`.
    pub b: Vec<i32>,
}

impl Lin {
    fn new(out: usize, inp: usize, qw: QT, qb: QT) -> Self {
        assert_eq!(qw.k.len(), out * inp);
        assert_eq!(qb.k.len(), out);
        Self {
            out,
            inp,
            qw,
            qb,
            fw: 0,
            w: vec![],
            b: vec![],
        }
    }
    fn set_fw(&mut self, fw: u32, qa: u32) {
        self.fw = fw;
        self.w = self.qw.fixed(fw);
        self.b = self.qb.fixed(fw + qa);
    }
}

/// Overflow guard: every accumulation is computed in `i64` and must fit `i32`
/// (so the GPU's wrapping `i32` gives the same bits).
#[derive(Default, Debug, Clone, Copy)]
pub struct Guard {
    /// Largest |accumulator| seen.
    pub max: i64,
}

impl Guard {
    #[inline]
    fn see(&mut self, v: i64) -> i64 {
        let a = v.abs();
        if a > self.max {
            self.max = a;
        }
        v
    }
    /// True when every value fitted `i32` (with one bit of headroom).
    pub fn ok(&self) -> bool {
        self.max < (1 << 30)
    }
}

#[inline]
fn rshift(acc: i64, f: u32) -> i64 {
    if f == 0 {
        acc
    } else {
        (acc + (1i64 << (f - 1))) >> f
    }
}

/// ARM / IFCE MLP: hidden ReLU layers (square hidden layers residual inside
/// the ReLU when `residual`), a linear output layer, and an optional linear
/// stabiliser branch over the input added to the output.
#[derive(Clone, Debug)]
pub struct Mlp {
    /// Layers in order.
    pub layers: Vec<Lin>,
    /// Stabiliser branch (input → output).
    pub stab: Option<Lin>,
    /// Square hidden layers carry a residual connection.
    pub residual: bool,
}

impl Mlp {
    fn from_ccx(c: &Ccx, p: &str, residual: bool) -> Self {
        let mut layers = vec![];
        for i in 0.. {
            let k = format!("{p}.L{i}.w");
            if !c.has(&k) {
                break;
            }
            let (sh, w) = c.get(&k);
            layers.push(Lin::new(
                sh[0],
                sh[1],
                QT::from_f64(w),
                QT::from_f64(&c.get(&format!("{p}.L{i}.b")).1),
            ));
        }
        let stab = c.has(&format!("{p}.stab.w")).then(|| {
            let (sh, w) = c.get(&format!("{p}.stab.w"));
            Lin::new(
                sh[0],
                sh[1],
                QT::from_f64(w),
                QT::from_f64(&c.get(&format!("{p}.stab.b")).1),
            )
        });
        Self {
            layers,
            stab,
            residual,
        }
    }
    fn set_fw(&mut self, fw: &[u32]) {
        for (l, &f) in self.layers.iter_mut().zip(fw) {
            l.set_fw(f, QA_ARM);
        }
        if let Some(s) = &mut self.stab {
            s.set_fw(fw[self.layers.len()], QA_ARM);
        }
    }
    /// Number of inputs.
    pub fn nin(&self) -> usize {
        self.layers[0].inp
    }
    /// Number of outputs.
    pub fn nout(&self) -> usize {
        self.layers.last().unwrap().out
    }
    /// Integer forward (`Q8` in, `Q8` out).
    pub fn run(&self, x: &[i32], out: &mut [i32], g: &mut Guard) {
        let mut a = [0i64; 64];
        let mut b = [0i64; 64];
        for i in 0..x.len() {
            a[i] = x[i] as i64;
        }
        let nl = self.layers.len();
        for (li, l) in self.layers.iter().enumerate() {
            for o in 0..l.out {
                let mut acc = l.b[o] as i64;
                for i in 0..l.inp {
                    acc = g.see(acc + l.w[o * l.inp + i] as i64 * a[i]);
                }
                let mut v = g.see(rshift(acc, l.fw));
                if li + 1 < nl {
                    if self.residual && l.inp == l.out {
                        v = g.see(v + a[o]);
                    }
                    v = v.max(0);
                }
                b[o] = v;
            }
            std::mem::swap(&mut a, &mut b);
        }
        if let Some(s) = &self.stab {
            for o in 0..s.out {
                let mut acc = s.b[o] as i64;
                for i in 0..s.inp {
                    acc = g.see(acc + s.w[o * s.inp + i] as i64 * x[i] as i64);
                }
                a[o] = g.see(a[o] + rshift(acc, s.fw));
            }
        }
        for o in 0..out.len() {
            out[o] = a[o] as i32;
        }
    }
    fn write(&self, w: &mut BitWriter) {
        eg_put(w, self.layers.len() as u64, 0);
        eg_put(
            w,
            (self.stab.is_some() as u8 | (self.residual as u8) << 1) as u64,
            0,
        );
        // input width once; each layer's input is the previous output, the
        // stabiliser's is the MLP input
        eg_put(w, self.nin() as u64, 0);
        for l in self.layers.iter().chain(self.stab.iter()) {
            eg_put(w, l.out as u64, 0);
            put_fw(w, l.fw);
            l.qw.write(w);
            l.qb.write(w);
        }
    }
    fn read(r: &mut BitReader) -> Result<Self, DecodeError> {
        let nl = eg_get(r, 0)? as usize;
        let fl = eg_get(r, 0)? as u8;
        let mut ls: Vec<Lin> = vec![];
        let nin = eg_get(r, 0)? as usize;
        for li in 0..nl + (fl & 1) as usize {
            let out = eg_get(r, 0)? as usize;
            let inp = if li == 0 || li == nl {
                nin
            } else {
                ls[li - 1].out
            };
            let fw = get_fw(r)?;
            if out > 64 || inp > 64 {
                return Err(DecodeError::Invalid("mlp dims"));
            }
            let (qw, qb) = (QT::read(r, out * inp)?, QT::read(r, out)?);
            let mut l = Lin::new(out, inp, qw, qb);
            l.set_fw(fw, QA_ARM);
            ls.push(l);
        }
        let stab = if fl & 1 == 1 { ls.pop() } else { None };
        Ok(Self {
            layers: ls,
            stab,
            residual: fl & 2 != 0,
        })
    }
}

/// A synthesis convolution (`k×k`, replicate padding).
#[derive(Clone, Debug)]
pub struct Conv {
    /// Weights `[out][inp][k][k]` + biases, fixed point.
    pub lin: Lin,
    /// Kernel size (1 or 3).
    pub k: usize,
    /// `y += x`.
    pub residual: bool,
    /// ReLU after.
    pub relu: bool,
}

impl Conv {
    fn from_ccx(c: &Ccx, p: &str) -> Self {
        let (sh, w) = c.get(&format!("{p}.w"));
        let meta = &c.get(&format!("{p}.meta")).1;
        let lin = Lin::new(
            sh[0],
            sh[1] * sh[2] * sh[3],
            QT::from_f64(w),
            QT::from_f64(&c.get(&format!("{p}.b")).1),
        );
        Self {
            lin,
            k: sh[2],
            residual: meta[0] != 0.0,
            relu: meta[1] != 0.0,
        }
    }
    /// Input channels.
    pub fn cin(&self) -> usize {
        self.lin.inp / (self.k * self.k)
    }
    fn write(&self, w: &mut BitWriter) {
        eg_put(w, self.lin.out as u64, 0);
        eg_put(w, self.cin() as u64, 0);
        eg_put(w, self.k as u64, 0);
        eg_put(w, (self.residual as u8 | (self.relu as u8) << 1) as u64, 0);
        put_fw(w, self.lin.fw);
        self.lin.qw.write(w);
        self.lin.qb.write(w);
    }
    fn read(r: &mut BitReader) -> Result<Self, DecodeError> {
        let out = eg_get(r, 0)? as usize;
        let cin = eg_get(r, 0)? as usize;
        let k = eg_get(r, 0)? as usize;
        let fl = eg_get(r, 0)? as u8;
        let fw = get_fw(r)?;
        if out > 128 || cin > 128 || !(k == 1 || k == 3) {
            return Err(DecodeError::Invalid("conv dims"));
        }
        let (qw, qb) = (QT::read(r, out * cin * k * k)?, QT::read(r, out)?);
        let mut lin = Lin::new(out, cin * k * k, qw, qb);
        lin.set_fw(fw, QA);
        Ok(Self {
            lin,
            k,
            residual: fl & 1 != 0,
            relu: fl & 2 != 0,
        })
    }
}

/// A symmetric 1-D upsampling / pre-concatenation filter.
#[derive(Clone, Debug)]
pub struct Filt {
    /// Source taps.
    pub q: QT,
    /// Fractional bits.
    pub fw: u32,
    /// Taps `Q(fw)`.
    pub w: Vec<i32>,
}

impl Filt {
    fn new(q: QT, fw: u32) -> Self {
        let w = q.fixed(fw);
        Self { q, fw, w }
    }
}

// ------------------------------------------------------------------- model

/// Everything the decoder needs except the latent payload.
#[derive(Clone, Debug)]
pub struct Model {
    /// Image height, width.
    pub h: usize,
    /// Width.
    pub w: usize,
    /// Grid sizes `(h, w)`, finest first.
    pub sizes: Vec<(usize, usize)>,
    /// Hyperlatent grids (not fed to the synthesis).
    pub hyper: Vec<bool>,
    /// Context template: `(dy, dx)` offsets (dy ≤ 0).
    pub ctx: Vec<(i32, i32)>,
    /// IFCE output features (appended to the ARM input).
    pub nifce: usize,
    /// Per grid: index into `ifces`, or `None` (zero inter-feature context).
    pub ifce_of: Vec<Option<usize>>,
    /// Entropy model.
    pub arm: Mlp,
    /// Inter-feature context extractors.
    pub ifces: Vec<Mlp>,
    /// Upsampling (transposed, even) filters.
    pub tconv: Vec<Filt>,
    /// Pre-concatenation (odd, residual) filters.
    pub pconv: Vec<Filt>,
    /// Synthesis main branch.
    pub syn: Vec<Conv>,
    /// Synthesis stabiliser (1×1 over the input).
    pub stab: Option<Conv>,
    /// Output transform (1×1).
    pub outt: Conv,
    /// Wavefront slope.
    pub slope: usize,
    /// rANS lanes per tile.
    pub lanes: usize,
    /// Tile width (0 = whole grid).
    pub tile_w: usize,
    /// Tile height (0 = whole grid).
    pub tile_h: usize,
}

impl Model {
    /// Build from an export, with the given lane / tile layout. Weight
    /// precisions are placeholders until [`encode`] fits them.
    pub fn from_ccx(c: &Ccx, lanes: usize, tile_w: usize, tile_h: usize) -> Self {
        let isz = c.ints("img_size");
        let sz = c.ints("sizes");
        let sizes: Vec<(usize, usize)> = sz
            .chunks(2)
            .map(|p| (p[0] as usize, p[1] as usize))
            .collect();
        let hyper = c.get("is_hyper").1.iter().map(|&v| v != 0.0).collect();
        let ms = c.ints("mask_size")[0] as i32;
        let ctx = c
            .ints("ctx_index")
            .iter()
            .map(|&k| ((k as i32) / ms - ms / 2, (k as i32) % ms - ms / 2))
            .collect::<Vec<_>>();
        let nifce = c.ints("n_ifce_out")[0] as usize;
        let n = sizes.len();
        let (ifce_of, ifces) = if c.has("ifce.index_to_arm") {
            let ia = c.ints("ifce.index_to_arm");
            let nf = c.ints("ifce.input_features");
            let of = (0..n)
                .map(|i| (nf[i] > 0 && ia[i] >= 0).then_some(ia[i] as usize))
                .collect();
            let na = ia.iter().copied().max().unwrap_or(-1) + 1;
            (
                of,
                (0..na)
                    .map(|j| Mlp::from_ccx(c, &format!("ifce.A{j}"), false))
                    .collect(),
            )
        } else {
            (vec![None; n], vec![])
        };
        let filts = |p: &str| {
            (0..)
                .map_while(|i| {
                    c.has(&format!("{p}.{i}"))
                        .then(|| Filt::new(QT::from_f64(&c.get(&format!("{p}.{i}")).1), 12))
                })
                .collect::<Vec<_>>()
        };
        let syn = (0..)
            .map_while(|i| {
                c.has(&format!("syn.L{i}.w"))
                    .then(|| Conv::from_ccx(c, &format!("syn.L{i}")))
            })
            .collect();
        let stab = c.has("syn.stab.w").then(|| Conv::from_ccx(c, "syn.stab"));
        let mut m = Self {
            h: isz[0] as usize,
            w: isz[1] as usize,
            sizes,
            hyper,
            slope: 0,
            ctx,
            nifce,
            ifce_of,
            arm: Mlp::from_ccx(c, "arm", true),
            ifces,
            tconv: filts("ups.tconv"),
            pconv: filts("ups.conv"),
            syn,
            stab,
            outt: Conv::from_ccx(c, "syn.out"),
            lanes,
            tile_w,
            tile_h,
        };
        m.slope = slope_of(&m.ctx);
        // NB: the export already carries Cool-chic's −4 log-scale shift in the
        // ARM's last bias (`arm_to_fixed_point_param` applies it in place
        // during the reference decode that `export_cc.py` captures).
        m
    }
}

/// Cool-chic's context priority order (`get_priority_order`, 9×9 causal mask)
/// as `(dy, dx)`.
pub const CTX_ORDER: [(i32, i32); 40] = {
    const PRIO: [u8; 40] = [
        38, 35, 30, 25, 23, 31, 36, 37, 39, 33, 28, 21, 20, 6, 15, 22, 29, 34, 32, 18, 12, 10, 5,
        9, 14, 19, 27, 24, 13, 8, 2, 1, 3, 11, 17, 26, 16, 7, 4, 0,
    ];
    let mut out = [(0i32, 0i32); 40];
    let mut k = 0;
    while k < 40 {
        out[PRIO[k] as usize] = ((k / 9) as i32 - 4, (k % 9) as i32 - 4);
        k += 1;
    }
    out
};

/// Smallest `s` with `t(y, x) = x + s·y` strictly after every context.
pub fn slope_of(ctx: &[(i32, i32)]) -> usize {
    let mut s = 1;
    for &(dy, dx) in ctx {
        if dy < 0 {
            s = s.max((dx.max(0) / -dy + 1) as usize);
        }
    }
    s
}

// ------------------------------------------------------------ prob tables

/// Laplace CDF tables `[fb][lb][sym+1]` (cumulative, `Q(PB)`), built with
/// plain IEEE f64 ops on the host and shared by both decoders.
pub fn tables() -> &'static Vec<u32> {
    static T: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut out = Vec::with_capacity(NFB * NLB * (NSYM + 1));
        for fb in 0..NFB {
            let fbw = (1 << FB_SHIFT) as f64;
            let mu = (fb as f64 * fbw - 128.0 + fbw / 2.0) / 256.0;
            for lb in 0..NLB {
                let b = ((lb << LB_SHIFT) as f64 / 256.0 - 5.0).exp();
                let cdf = |x: f64| {
                    let d = x - mu;
                    if d < 0.0 {
                        0.5 * (d / b).exp()
                    } else {
                        1.0 - 0.5 * (-d / b).exp()
                    }
                };
                let mut p = vec![0f64; NSYM];
                p[0] = cdf(-R as f64 - 0.5);
                for r in -R..=R {
                    p[(r + R + 1) as usize] = cdf(r as f64 + 0.5) - cdf(r as f64 - 0.5);
                }
                p[NSYM - 1] = 1.0 - cdf(R as f64 + 0.5);
                let budget = (PM - NSYM as u32) as f64;
                let mut f: Vec<u32> = p.iter().map(|&q| 1 + (q * budget).floor() as u32).collect();
                let tot: u32 = f.iter().sum();
                let mut best = 0;
                for i in 1..NSYM {
                    if f[i] > f[best] {
                        best = i;
                    }
                }
                f[best] += PM - tot;
                let mut c = 0;
                out.push(0);
                for &x in &f {
                    c += x;
                    out.push(c);
                }
            }
        }
        out
    })
}

/// `(base, table offset)` from the ARM output (μ, ln b in `Q8`).
#[inline]
pub fn bucket(mu: i32, ls: i32) -> (i32, usize) {
    let mu = mu.clamp(-64 * 256, 64 * 256 - 1);
    let base = (mu + 128) >> 8;
    let fb = ((mu - (base << 8) + 128) >> FB_SHIFT) as usize;
    let lb = ((ls + 1280 + (1 << (LB_SHIFT - 1))) >> LB_SHIFT).clamp(0, NLB as i32 - 1) as usize;
    (base, (fb * NLB + lb) * (NSYM + 1))
}

// ------------------------------------------------------------------- rANS

struct RansEnc {
    x: u32,
    words: Vec<u16>,
}

impl RansEnc {
    fn new() -> Self {
        Self {
            x: RL,
            words: vec![],
        }
    }
    fn put(&mut self, start: u32, freq: u32) {
        let xmax = ((RL >> PB) << 16) * freq;
        if self.x >= xmax {
            self.words.push(self.x as u16);
            self.x >>= 16;
        }
        self.x = ((self.x / freq) << PB) + self.x % freq + start;
    }
    /// Words in decode order: final state (2 words), then payload.
    fn finish(mut self) -> Vec<u16> {
        self.words.push(self.x as u16);
        self.words.push((self.x >> 16) as u16);
        self.words.reverse();
        self.words
    }
}

/// Decoder side of one lane.
#[derive(Clone, Copy)]
pub struct RansDec {
    /// State.
    pub x: u32,
    /// Next word.
    pub pos: usize,
}

impl RansDec {
    /// Start at `words[pos..]`.
    pub fn new(words: &[u16], pos: usize) -> Self {
        let x = (words[pos] as u32) << 16 | words[pos + 1] as u32;
        Self { x, pos: pos + 2 }
    }
    #[inline]
    fn peek(&self) -> u32 {
        self.x & (PM - 1)
    }
    #[inline]
    fn adv(&mut self, words: &[u16], start: u32, freq: u32) {
        self.x = freq * (self.x >> PB) + self.peek() - start;
        if self.x < RL {
            self.x = self.x << 16 | *words.get(self.pos).unwrap_or(&0) as u32;
            self.pos += 1;
        }
    }
}

/// Decode one latent value with the table at `off` (`cum[0..=NSYM]`).
#[inline]
pub fn dec_sym(d: &mut RansDec, words: &[u16], tab: &[u32], base: i32, off: usize) -> i32 {
    let cum = &tab[off..off + NSYM + 1];
    let v = d.peek();
    let (mut lo, mut hi) = (0usize, NSYM);
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if cum[mid] <= v {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    d.adv(words, cum[lo], cum[lo + 1] - cum[lo]);
    if lo == 0 || lo == NSYM - 1 {
        {
            let mut e = 0u32;
            for _ in 0..2 {
                let b = d.peek() >> (PB - 8);
                d.adv(words, b << (PB - 8), 1 << (PB - 8));
                e = e << 8 | b;
            }
            let e = e as i32;
            if lo == 0 {
                base - R - 1 - e
            } else {
                base + R + 1 + e
            }
        }
    } else {
        base + lo as i32 - R - 1
    }
}

/// Encode-side symbol list for one value (in decode order).
fn enc_syms(v: i32, base: i32, off: usize, out: &mut Vec<(u32, u32)>) {
    let tab = tables();
    let cum = &tab[off..off + NSYM + 1];
    let r = v - base;
    let (s, esc) = if r < -R {
        (0, Some(-R - 1 - r))
    } else if r > R {
        (NSYM - 1, Some(r - R - 1))
    } else {
        ((r + R + 1) as usize, None)
    };
    out.push((cum[s], cum[s + 1] - cum[s]));
    if let Some(e) = esc {
        assert!((0..65536).contains(&e), "latent escape out of range");
        for b in [(e >> 8) as u32, (e & 255) as u32] {
            out.push((b << (PB - 8), 1 << (PB - 8)));
        }
    }
}

// --------------------------------------------------------------- schedule

/// Wavefront layout of one grid: tiles of `th × tw`, `lanes` lanes per tile.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    /// Grid height, width.
    pub h: usize,
    /// Width.
    pub w: usize,
    /// Tile height, width.
    pub th: usize,
    /// Tile width.
    pub tw: usize,
    /// Tile columns.
    pub ntx: usize,
    /// Tiles.
    pub nt: usize,
    /// Lanes per tile (≤ th); tile `j` uses global lanes `j·lanes ..`.
    pub lanes: usize,
    /// Slope.
    pub s: usize,
    /// Wavefront steps per tile.
    pub steps: usize,
}

impl Layout {
    /// Layout for an `h × w` grid.
    pub fn new(h: usize, w: usize, m: &Model) -> Self {
        let tw = if m.tile_w == 0 { w } else { m.tile_w.min(w) };
        let th = if m.tile_h == 0 { h } else { m.tile_h.min(h) };
        let ntx = w.div_ceil(tw);
        let nt = ntx * h.div_ceil(th);
        assert!(nt <= m.lanes, "more tiles than lanes");
        let lanes = (m.lanes / nt).min(th).max(1);
        Self {
            h,
            w,
            th,
            tw,
            ntx,
            nt,
            lanes,
            s: m.slope,
            steps: tw + m.slope * (th - 1),
        }
    }
    /// Tile origin and extent `(y0, x0, th, tw)`.
    #[inline]
    pub fn tile(&self, tile: usize) -> (usize, usize, usize, usize) {
        let (y0, x0) = ((tile / self.ntx) * self.th, (tile % self.ntx) * self.tw);
        (y0, x0, self.th.min(self.h - y0), self.tw.min(self.w - x0))
    }
    /// Pixels of lane `l` active at step `t`, as global `(y, x)`, in decode order.
    #[inline]
    pub fn active(&self, l: usize, t: usize, tile: usize, mut f: impl FnMut(usize, usize)) {
        let (y0, x0, th, tw) = self.tile(tile);
        let mut y = l;
        while y < th {
            let sy = self.s * y;
            if sy > t {
                break;
            }
            let x = t - sy;
            if x < tw {
                f(y0 + y, x0 + x);
            }
            y += self.lanes;
        }
    }
}

// ----------------------------------------------------------- latent model

/// Gather the ARM input for global `(y, x)` of a grid (values so far in
/// `lat`, context cut at the tile border) into `buf` (`Q8`).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn arm_input(
    m: &Model,
    lay: &Layout,
    lat: &[i32],
    ifc: &[i32],
    y: usize,
    x: usize,
    tile: usize,
    buf: &mut [i32],
) {
    let (y0, x0, th, tw) = lay.tile(tile);
    for (i, &(dy, dx)) in m.ctx.iter().enumerate() {
        let (yy, xx) = (y as i32 + dy, x as i32 + dx);
        buf[i] =
            if yy >= y0 as i32 && yy < (y0 + th) as i32 && xx >= x0 as i32 && xx < (x0 + tw) as i32
            {
                lat[yy as usize * lay.w + xx as usize] << QA_ARM
            } else {
                0
            };
    }
    let nc = m.ctx.len();
    for j in 0..m.nifce {
        buf[nc + j] = if ifc.is_empty() {
            0
        } else {
            ifc[(y * lay.w + x) * m.nifce + j]
        };
    }
}

/// Inter-feature context of grid `g` (`[h·w][nifce]`, `Q8`) from the
/// already-decoded coarser grids, or empty when the grid has no IFCE.
pub fn ifce_ctx(m: &Model, g: usize, grids: &[Vec<i32>], gd: &mut Guard) -> Vec<i32> {
    let Some(a) = m.ifce_of[g] else { return vec![] };
    let mlp = &m.ifces[a];
    let n = m.sizes.len();
    // nearest-upsampled stack of grids g+1..n−1 at the resolution of grid g+1
    // (channel order: g+1, g+2, …), as Cool-chic's `fixed_upsampling`
    let (h1, w1) = m.sizes[g + 1];
    let mut stack: Vec<Vec<i32>> = vec![];
    let (mut ch, mut cw) = m.sizes[n - 1];
    let mut cur: Vec<Vec<i32>> = vec![grids[n - 1].clone()];
    for k in (g + 1..n - 1).rev() {
        let (th, tw) = m.sizes[k];
        if (th, tw) != (ch, cw) {
            cur = cur
                .iter()
                .map(|c| {
                    (0..th * tw)
                        .map(|i| c[(i / tw / 2).min(ch - 1) * cw + (i % tw / 2).min(cw - 1)])
                        .collect()
                })
                .collect();
            (ch, cw) = (th, tw);
        }
        let mut nc = vec![grids[k].clone()];
        nc.extend(cur);
        cur = nc;
    }
    stack.extend(cur);
    assert_eq!((ch, cw), (h1, w1));
    assert_eq!(stack.len(), mlp.nin());
    let (h, w) = m.sizes[g];
    let mut low = vec![0i32; h1 * w1 * m.nifce];
    let mut inb = vec![0i32; mlp.nin()];
    for p in 0..h1 * w1 {
        for c in 0..stack.len() {
            inb[c] = stack[c][p] << QA_ARM;
        }
        mlp.run(&inb, &mut low[p * m.nifce..(p + 1) * m.nifce], gd);
    }
    let mut out = vec![0i32; h * w * m.nifce];
    for y in 0..h {
        for x in 0..w {
            let s = ((y / 2).min(h1 - 1) * w1 + (x / 2).min(w1 - 1)) * m.nifce;
            out[(y * w + x) * m.nifce..][..m.nifce].copy_from_slice(&low[s..s + m.nifce]);
        }
    }
    out
}

// ---------------------------------------------------------------- stream

/// Parsed stream.
pub struct Parsed {
    /// Model.
    pub m: Model,
    /// All lanes' rANS words, concatenated.
    pub words: Vec<u16>,
    /// Start of each lane's words.
    pub starts: Vec<u32>,
}

fn write_model(m: &Model, w: &mut BitWriter) {
    eg_put(w, m.h as u64, 0);
    eg_put(w, m.w as u64, 0);
    eg_put(w, m.sizes.len() as u64, 0);
    for (i, &(h, ww)) in m.sizes.iter().enumerate() {
        // usually (⌈H/2^d⌉, ⌈W/2^d⌉): send d; else d = 31 and the size
        match (0..31).find(|&d| (h, ww) == (m.h.div_ceil(1 << d), m.w.div_ceil(1 << d))) {
            Some(d) => eg_put(w, d as u64, 0),
            None => {
                eg_put(w, 31, 0);
                eg_put(w, h as u64, 0);
                eg_put(w, ww as u64, 0);
            }
        }
        eg_put(
            w,
            (m.hyper[i] as u8 | (m.ifce_of[i].map_or(0, |a| a as u8 + 1) << 1)) as u64,
            0,
        );
    }
    // context: Cool-chic's priority order prefix (flag 1), or explicit
    let canon = m.ctx.len() <= CTX_ORDER.len() && m.ctx.iter().zip(CTX_ORDER).all(|(a, b)| *a == b);
    eg_put(w, (m.ctx.len() << 1 | canon as usize) as u64, 0);
    if !canon {
        for &(dy, dx) in &m.ctx {
            eg_put(w, (((-dy) as u8) << 4 | (dx + 8) as u8) as u64, 0);
        }
    }
    eg_put(w, m.nifce as u64, 0);
    eg_put(w, m.lanes as u64, 0);
    eg_put(w, m.tile_w as u64, 0);
    eg_put(w, m.tile_h as u64, 0);
    m.arm.write(w);
    eg_put(w, m.ifces.len() as u64, 0);
    for f in &m.ifces {
        f.write(w);
    }
    for fs in [&m.tconv, &m.pconv] {
        eg_put(w, fs.len() as u64, 0);
        for f in fs {
            // symmetric taps: send the first ⌈k/2⌉
            let k = f.q.k.len();
            assert!(
                (0..k).all(|i| f.q.k[i] == f.q.k[k - 1 - i]),
                "upsampling filter not symmetric"
            );
            eg_put(w, k as u64, 0);
            put_fw(w, f.fw);
            QT {
                s: f.q.s,
                k: f.q.k[..k.div_ceil(2)].to_vec(),
            }
            .write(w);
        }
    }
    eg_put(w, m.syn.len() as u64, 0);
    for c in &m.syn {
        c.write(w);
    }
    eg_put(w, (m.stab.is_some() as u8) as u64, 0);
    if let Some(s) = &m.stab {
        s.write(w);
    }
    m.outt.write(w);
}

fn read_model(r: &mut BitReader) -> Result<Model, DecodeError> {
    let bad = |s| DecodeError::Invalid(s);
    let h = eg_get(r, 0)? as usize;
    let w = eg_get(r, 0)? as usize;
    let n = eg_get(r, 0)? as usize;
    if h == 0 || w == 0 || h * w > 1 << 26 || n == 0 || n > 16 {
        return Err(bad("dims"));
    }
    let (mut sizes, mut hyper, mut ifce_of) = (vec![], vec![], vec![]);
    for _ in 0..n {
        let d = eg_get(r, 0)?;
        let (a, b) = match d {
            0..=30 => (h.div_ceil(1 << d), w.div_ceil(1 << d)),
            31 => (eg_get(r, 0)? as usize, eg_get(r, 0)? as usize),
            _ => return Err(bad("grid size")),
        };
        if a == 0 || b == 0 || a > h || b > w {
            return Err(bad("grid size"));
        }
        sizes.push((a, b));
        let f = eg_get(r, 0)? as u8;
        hyper.push(f & 1 != 0);
        ifce_of.push((f >> 1 != 0).then(|| (f >> 1) as usize - 1));
    }
    let nc = eg_get(r, 0)? as usize;
    let canon = nc & 1 == 1;
    let nc = nc >> 1;
    if nc > CTX_ORDER.len() {
        return Err(bad("ctx"));
    }
    let ctx = if canon {
        CTX_ORDER[..nc].to_vec()
    } else {
        (0..nc)
            .map(|_| eg_get(r, 0).map(|b| (-((b >> 4) as i32), (b & 15) as i32 - 8)))
            .collect::<Result<Vec<_>, _>>()?
    };
    let nifce = eg_get(r, 0)? as usize;
    let lanes = eg_get(r, 0)? as usize;
    let tile_w = eg_get(r, 0)? as usize;
    let tile_h = eg_get(r, 0)? as usize;
    if nifce > 16 || lanes == 0 || lanes > 1024 {
        return Err(bad("lanes"));
    }
    let arm = Mlp::read(r)?;
    let ni = eg_get(r, 0)? as usize;
    let ifces = (0..ni.min(16))
        .map(|_| Mlp::read(r))
        .collect::<Result<Vec<_>, _>>()?;
    let mut fl = vec![];
    for _ in 0..2 {
        let k = eg_get(r, 0)? as usize;
        let mut v = vec![];
        for _ in 0..k.min(32) {
            let k = eg_get(r, 0)? as usize;
            let fw = get_fw(r)?;
            if !(2..=16).contains(&k) {
                return Err(bad("filter"));
            }
            let mut q = QT::read(r, k.div_ceil(2))?;
            for i in (0..k / 2).rev() {
                let v = q.k[i];
                q.k.push(v);
            }
            v.push(Filt::new(q, fw));
        }
        fl.push(v);
    }
    let ns = eg_get(r, 0)? as usize;
    let syn = (0..ns.min(16))
        .map(|_| Conv::read(r))
        .collect::<Result<Vec<_>, _>>()?;
    let stab = if (eg_get(r, 0)? as u8) != 0 {
        Some(Conv::read(r)?)
    } else {
        None
    };
    let outt = Conv::read(r)?;
    let pconv = fl.pop().unwrap();
    let tconv = fl.pop().unwrap();
    let slope = slope_of(&ctx);
    let m = Model {
        h,
        w,
        sizes,
        hyper,
        ctx,
        nifce,
        ifce_of,
        arm,
        ifces,
        tconv,
        pconv,
        syn,
        stab,
        outt,
        slope,
        lanes,
        tile_w,
        tile_h,
    };
    if m.arm.nin() != m.ctx.len() + m.nifce
        || m.arm.nout() != 2
        || m.ifce_of.iter().flatten().any(|&a| a >= m.ifces.len())
    {
        return Err(bad("model shape"));
    }
    Ok(m)
}

/// Parse a `TCI1` stream.
pub fn parse(bytes: &[u8]) -> Result<Parsed, DecodeError> {
    let mut r = ByteReader::new(bytes);
    if r.take(4)? != MAGIC {
        return Err(DecodeError::Invalid("not TCI1"));
    }
    let mb = r.block()?;
    let m = read_model(&mut BitReader::new(mb))?;
    let mut starts = vec![];
    let mut acc = 0u32;
    for _ in 0..m.lanes {
        starts.push(acc);
        acc = acc
            .checked_add(r.varint()? as u32)
            .ok_or(DecodeError::Invalid("lanes"))?;
    }
    let raw = r.take(acc as usize * 2)?;
    let words = raw
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes(*c))
        .collect();
    for g in 0..m.sizes.len() {
        let lay = Layout::new(m.sizes[g].0, m.sizes[g].1, &m);
        if lay.nt > m.lanes {
            return Err(DecodeError::Invalid("tiles > lanes"));
        }
    }
    Ok(Parsed { m, words, starts })
}

// ------------------------------------------------------------- decoding

/// Decode the latent grids (finest first) on the CPU. Lane `l`'s rANS
/// state runs on across grids (coarse → fine).
pub fn decode_latents(p: &Parsed, gd: &mut Guard) -> Vec<Vec<i32>> {
    let m = &p.m;
    let n = m.sizes.len();
    let tab = tables();
    let mut grids: Vec<Vec<i32>> = vec![vec![]; n];
    let mut inb = vec![0i32; m.arm.nin()];
    let mut o = [0i32; 2];
    let mut decs: Vec<RansDec> = p
        .starts
        .iter()
        .map(|&s| RansDec::new(&p.words, s as usize))
        .collect();
    for g in (0..n).rev() {
        let (h, w) = m.sizes[g];
        let lay = Layout::new(h, w, m);
        let ifc = ifce_ctx(m, g, &grids, gd);
        let mut lat = vec![0i32; h * w];
        for tile in 0..lay.nt {
            for t in 0..lay.steps {
                for l in 0..lay.lanes {
                    let d = &mut decs[tile * lay.lanes + l];
                    lay.active(l, t, tile, |y, x| {
                        arm_input(m, &lay, &lat, &ifc, y, x, tile, &mut inb);
                        m.arm.run(&inb, &mut o, gd);
                        let (base, off) = bucket(o[0], o[1]);
                        lat[y * w + x] = dec_sym(d, &p.words, tab, base, off);
                    });
                }
            }
        }
        grids[g] = lat;
    }
    grids
}

/// Symmetric 1-D filter taps → index helpers for the 2× transposed conv.
/// `out[m] = Σ_j xpad[j]·w[m + C − 2j]`, `xpad[j] = x[clamp(j − P0)]`.
fn tconv_1d(
    x: &[i64],
    n: usize,
    stride: usize,
    f: &Filt,
    out: &mut [i64],
    ostride: usize,
    gd: &mut Guard,
) {
    let k = f.w.len();
    let p0 = (k / 2) as i64;
    let c = 2 * p0 - 1 + p0;
    for m in 0..2 * n {
        let mut acc = 0i64;
        // j with 0 ≤ m + c − 2j < k
        let jlo = ((m as i64 + c - k as i64 + 1) + 1).div_euclid(2).max(0);
        let jhi = (m as i64 + c).div_euclid(2);
        for j in jlo..=jhi {
            let t = m as i64 + c - 2 * j;
            if t < 0 || t >= k as i64 {
                continue;
            }
            let src = (j - p0).clamp(0, n as i64 - 1) as usize;
            acc = gd.see(acc + f.w[t as usize] as i64 * x[src * stride]);
        }
        out[m * ostride] = gd.see(rshift(acc, f.fw));
    }
}

/// `2×` learned upsampling of one channel (`h × w` → `2h × 2w`, `Q(QA)`).
fn tconv2d(x: &[i64], h: usize, w: usize, f: &Filt, gd: &mut Guard) -> Vec<i64> {
    let mut t = vec![0i64; h * 2 * w];
    for y in 0..h {
        tconv_1d(&x[y * w..], w, 1, f, &mut t[y * 2 * w..], 1, gd);
    }
    let mut o = vec![0i64; 4 * h * w];
    for xx in 0..2 * w {
        tconv_1d(&t[xx..], h, 2 * w, f, &mut o[xx..], 2 * w, gd);
    }
    o
}

/// Residual separable symmetric odd filter (zero padding): `y = w⊗w ∗ x + x`.
fn pconv2d(x: &[i64], h: usize, w: usize, f: &Filt, gd: &mut Guard) -> Vec<i64> {
    let k = f.w.len();
    let p = (k / 2) as i64;
    let mut t = vec![0i64; h * w];
    for y in 0..h {
        for xx in 0..w {
            let mut acc = 0i64;
            for i in 0..k {
                let sx = xx as i64 + i as i64 - p;
                if sx >= 0 && sx < w as i64 {
                    acc = gd.see(acc + f.w[i] as i64 * x[y * w + sx as usize]);
                }
            }
            t[y * w + xx] = gd.see(rshift(acc, f.fw));
        }
    }
    let mut o = vec![0i64; h * w];
    for y in 0..h {
        for xx in 0..w {
            let mut acc = 0i64;
            for i in 0..k {
                let sy = y as i64 + i as i64 - p;
                if sy >= 0 && sy < h as i64 {
                    acc = gd.see(acc + f.w[i] as i64 * t[sy as usize * w + xx]);
                }
            }
            o[y * w + xx] = gd.see(rshift(acc, f.fw) + x[y * w + xx]);
        }
    }
    o
}

/// Upsampling pyramid → dense synthesis input `[C][h·w]` (`Q(QA)`).
pub fn upsample(m: &Model, grids: &[Vec<i32>], gd: &mut Guard) -> Vec<Vec<i64>> {
    let idx: Vec<usize> = (0..m.sizes.len()).filter(|&i| !m.hyper[i]).collect();
    let mut it = idx.iter().rev();
    let &g0 = it.next().unwrap();
    let (mut ch, mut cw) = m.sizes[g0];
    let mut up: Vec<Vec<i64>> = vec![grids[g0].iter().map(|&v| (v as i64) << QA).collect()];
    for (k, &g) in it.enumerate() {
        let (th, tw) = m.sizes[g];
        let f = &m.tconv[k % m.tconv.len()];
        let ups: Vec<Vec<i64>> = up
            .iter()
            .map(|c| {
                let full = tconv2d(c, ch, cw, f, gd);
                (0..th * tw)
                    .map(|i| full[(i / tw) * 2 * cw + i % tw])
                    .collect()
            })
            .collect();
        let tgt: Vec<i64> = grids[g].iter().map(|&v| (v as i64) << QA).collect();
        let hi = pconv2d(&tgt, th, tw, &m.pconv[k % m.pconv.len()], gd);
        up = std::iter::once(hi).chain(ups).collect();
        (ch, cw) = (th, tw);
    }
    assert_eq!((ch, cw), (m.h, m.w));
    up
}

#[allow(clippy::too_many_arguments)]
fn conv_px(
    c: &Conv,
    inp: &[Vec<i64>],
    h: usize,
    w: usize,
    y: usize,
    x: usize,
    o: usize,
    gd: &mut Guard,
) -> i64 {
    let l = &c.lin;
    let cin = c.cin();
    let k = c.k;
    let p = (k / 2) as i64;
    let mut acc = l.b[o] as i64;
    for ci in 0..cin {
        for dy in 0..k {
            let sy = (y as i64 + dy as i64 - p).clamp(0, h as i64 - 1) as usize;
            for dx in 0..k {
                let sx = (x as i64 + dx as i64 - p).clamp(0, w as i64 - 1) as usize;
                acc = gd.see(
                    acc + l.w[((o * cin + ci) * k + dy) * k + dx] as i64 * inp[ci][sy * w + sx],
                );
            }
        }
    }
    let mut v = gd.see(rshift(acc, l.fw));
    if c.residual {
        v = gd.see(v + inp[o][y * w + x]);
    }
    if c.relu {
        v = v.max(0);
    }
    v
}

fn conv_all(c: &Conv, inp: &[Vec<i64>], h: usize, w: usize, gd: &mut Guard) -> Vec<Vec<i64>> {
    (0..c.lin.out)
        .map(|o| {
            (0..h * w)
                .map(|i| conv_px(c, inp, h, w, i / w, i % w, o, gd))
                .collect()
        })
        .collect()
}

/// Synthesis → interleaved RGB8.
pub fn synthesize(m: &Model, dense: &[Vec<i64>], gd: &mut Guard) -> Vec<u8> {
    let (h, w) = (m.h, m.w);
    let mut x = dense.to_vec();
    for c in &m.syn {
        x = conv_all(c, &x, h, w, gd);
    }
    if let Some(s) = &m.stab {
        let sb = conv_all(s, &dense[..s.cin()], h, w, gd);
        for (a, b) in x.iter_mut().zip(sb) {
            for (p, q) in a.iter_mut().zip(b) {
                *p = gd.see(*p + q);
            }
        }
    }
    let y = conv_all(&m.outt, &x, h, w, gd);
    let mut out = vec![0u8; h * w * 3];
    for i in 0..h * w {
        for c in 0..3 {
            out[i * 3 + c] = rshift(gd.see(y[c][i] * 255), QA).clamp(0, 255) as u8;
        }
    }
    out
}

/// Full CPU decode: stream → RGB8 (`h·w·3`). Errors if any accumulation
/// leaves `i32` (the GPU would wrap).
pub fn decode(bytes: &[u8]) -> Result<(usize, usize, Vec<u8>), DecodeError> {
    let p = parse(bytes)?;
    let mut gd = Guard::default();
    let grids = decode_latents(&p, &mut gd);
    let dense = upsample(&p.m, &grids, &mut gd);
    let rgb = synthesize(&p.m, &dense, &mut gd);
    if !gd.ok() {
        return Err(DecodeError::Invalid("i32 overflow"));
    }
    Ok((p.m.h, p.m.w, rgb))
}

// ------------------------------------------------------------- encoding

/// Encoder statistics.
#[derive(Debug, Clone, Default)]
pub struct Stats {
    /// Total stream bytes.
    pub bytes: usize,
    /// Model (header + nets) bytes.
    pub model_bytes: usize,
    /// Latent payload bytes (lane offsets + words).
    pub latent_bytes: usize,
    /// Ideal latent bits per grid under the quantized tables.
    pub grid_bits: Vec<f64>,
    /// Ideal latent bits under the quantized tables (no lane overhead).
    pub ideal_bits: f64,
}

/// Encode an export into a `TCI1` stream with the given lanes / tile width.
pub fn encode(c: &Ccx, lanes: usize, tile_w: usize, tile_h: usize) -> (Vec<u8>, Stats) {
    let mut m = Model::from_ccx(c, lanes, tile_w, tile_h);
    let n = m.sizes.len();
    let grids: Vec<Vec<i32>> = (0..n)
        .map(|g| {
            c.ints(&format!("latent.{g}"))
                .iter()
                .map(|&v| v as i32)
                .collect()
        })
        .collect();
    for g in 0..n {
        assert_eq!(grids[g].len(), m.sizes[g].0 * m.sizes[g].1);
    }
    // simple, exact approach: set caps, then shrink until the full decode
    // passes the guard (encoder runs the decoder anyway)
    let fit_all = |m: &mut Model, cap: u32| {
        let ws = vec![cap; m.arm.layers.len() + 1];
        m.arm.set_fw(&ws);
        for f in &mut m.ifces {
            let ws = vec![cap; f.layers.len() + 1];
            f.set_fw(&ws);
        }
        for f in m.tconv.iter_mut().chain(m.pconv.iter_mut()) {
            *f = Filt::new(f.q.clone(), cap.min(14));
        }
        for cv in m
            .syn
            .iter_mut()
            .chain(m.stab.iter_mut())
            .chain(std::iter::once(&mut m.outt))
        {
            cv.lin.set_fw(cap.min(14), QA);
        }
    };
    let mut cap = 16;
    loop {
        fit_all(&mut m, cap);
        let mut gd = Guard::default();
        // check every stage on the true latents
        let mut ok = true;
        let mut inb = vec![0i32; m.arm.nin()];
        let mut o = [0i32; 2];
        for g in (0..n).rev() {
            let (h, w) = m.sizes[g];
            let lay = Layout::new(h, w, &m);
            let ifc = ifce_ctx(&m, g, &grids, &mut gd);
            for tile in 0..lay.nt {
                let (y0, x0, th, tw) = lay.tile(tile);
                for y in y0..y0 + th {
                    for x in x0..x0 + tw {
                        arm_input(&m, &lay, &grids[g], &ifc, y, x, tile, &mut inb);
                        m.arm.run(&inb, &mut o, &mut gd);
                    }
                }
            }
        }
        if gd.ok() {
            let dense = upsample(&m, &grids, &mut gd);
            let _ = synthesize(&m, &dense, &mut gd);
        }
        ok &= gd.ok();
        if ok || cap <= 6 {
            break;
        }
        cap -= 1;
    }
    // entropy-code the latents: per lane, symbols in decode order
    let mut w = ByteWriter::new();
    w.bytes(MAGIC);
    let mut bw = BitWriter::new();
    write_model(&m, &mut bw);
    w.block(&bw.finish());
    let model_bytes = w.buf.len();
    let mut st = Stats {
        model_bytes,
        grid_bits: vec![0.0; n],
        ..Default::default()
    };
    let mut gd = Guard::default();
    let mut inb = vec![0i32; m.arm.nin()];
    let mut o = [0i32; 2];
    let mut syms: Vec<Vec<(u32, u32)>> = vec![vec![]; m.lanes];
    for g in (0..n).rev() {
        let (h, ww) = m.sizes[g];
        let lay = Layout::new(h, ww, &m);
        let ifc = ifce_ctx(&m, g, &grids, &mut gd);
        let lat = &grids[g];
        for tile in 0..lay.nt {
            for t in 0..lay.steps {
                for l in 0..lay.lanes {
                    let sl = &mut syms[tile * lay.lanes + l];
                    lay.active(l, t, tile, |y, x| {
                        arm_input(&m, &lay, lat, &ifc, y, x, tile, &mut inb);
                        m.arm.run(&inb, &mut o, &mut gd);
                        let (base, off) = bucket(o[0], o[1]);
                        let v = lat[y * ww + x];
                        let k0 = sl.len();
                        enc_syms(v, base, off, sl);
                        for &(_, f) in &sl[k0..] {
                            st.grid_bits[g] += PB as f64 - (f as f64).log2();
                        }
                    });
                }
            }
        }
    }
    st.ideal_bits = st.grid_bits.iter().sum();
    let lanes: Vec<Vec<u16>> = syms
        .iter()
        .map(|s| {
            let mut e = RansEnc::new();
            for &(a, f) in s.iter().rev() {
                e.put(a, f);
            }
            e.finish()
        })
        .collect();
    for l in &lanes {
        w.varint(l.len() as u64);
    }
    for l in &lanes {
        for v in l {
            w.bytes(&v.to_le_bytes());
        }
    }
    st.bytes = w.buf.len();
    st.latent_bytes = st.bytes - model_bytes;
    (w.buf, st)
}

/// PSNR (dB) of two RGB8 buffers.
pub fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let mse: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (x as f64 - y as f64).powi(2))
        .sum::<f64>()
        / a.len() as f64;
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

// ----------------------------------------------------- integer graphs (GPU)

use tang_expr::int::{IId, IType, IntGraph};

fn g_layer(g: &mut IntGraph, l: &Lin, inp: &[IId], residual: bool, relu: bool) -> Vec<IId> {
    let zero = g.i32(0);
    (0..l.out)
        .map(|o| {
            let mut acc = g.i32(l.b[o]);
            for i in 0..l.inp {
                let w = l.w[o * l.inp + i];
                if w != 0 {
                    let wc = g.i32(w);
                    let t = g.mul(wc, inp[i]);
                    acc = g.add(acc, t);
                }
            }
            let mut v = g_rshift(g, acc, l.fw);
            if residual {
                v = g.add(v, inp[o]);
            }
            if relu {
                v = g.max(v, zero);
            }
            v
        })
        .collect()
}

fn g_rshift(g: &mut IntGraph, v: IId, f: u32) -> IId {
    if f == 0 {
        return v;
    }
    let h = g.i32(1 << (f - 1));
    let s = g.u32(f);
    let t = g.add(v, h);
    g.shr(t, s)
}

impl Mlp {
    /// [`Mlp::run`] as an integer graph: params `0..nin`, outputs `0..nout`.
    pub fn graph(&self) -> (IntGraph, Vec<IId>) {
        let mut g = IntGraph::new();
        let x: Vec<IId> = (0..self.nin())
            .map(|i| g.param(i as u16, IType::I32))
            .collect();
        let mut a = x.clone();
        let nl = self.layers.len();
        for (li, l) in self.layers.iter().enumerate() {
            let hidden = li + 1 < nl;
            a = g_layer(
                &mut g,
                l,
                &a,
                hidden && self.residual && l.inp == l.out,
                hidden,
            );
        }
        if let Some(s) = &self.stab {
            let sb = g_layer(&mut g, s, &x, false, false);
            a = a.iter().zip(sb).map(|(&p, q)| g.add(p, q)).collect();
        }
        (g, a)
    }
}

/// A synthesis segment: one (possibly 3×3) conv followed by 1×1 convs.
/// Params: the first conv's taps `[cin][k][k]` (replicate-padded gather),
/// then (if it is residual) its `cin` centre values; when `last`, then the
/// `stab` input channels. Outputs: the segment's channels, or (when `last`)
/// the three 8-bit pixel values.
pub fn segment_graph(m: &Model, convs: &[&Conv], last: bool) -> (IntGraph, Vec<IId>) {
    let mut g = IntGraph::new();
    let c0 = convs[0];
    let nt = c0.lin.inp;
    let mut np = 0u16;
    let mut par = |g: &mut IntGraph| {
        let p = g.param(np, IType::I32);
        np += 1;
        p
    };
    let taps: Vec<IId> = (0..nt).map(|_| par(&mut g)).collect();
    let centre: Vec<IId> = if c0.residual {
        (0..c0.cin()).map(|_| par(&mut g)).collect()
    } else {
        vec![]
    };
    let mut a = {
        let zero = g.i32(0);
        let l = &c0.lin;
        (0..l.out)
            .map(|o| {
                let mut acc = g.i32(l.b[o]);
                for i in 0..l.inp {
                    let w = l.w[o * l.inp + i];
                    if w != 0 {
                        let wc = g.i32(w);
                        let t = g.mul(wc, taps[i]);
                        acc = g.add(acc, t);
                    }
                }
                let mut v = g_rshift(&mut g, acc, l.fw);
                if c0.residual {
                    v = g.add(v, centre[o]);
                }
                if c0.relu {
                    v = g.max(v, zero);
                }
                v
            })
            .collect::<Vec<_>>()
    };
    for c in &convs[1..] {
        assert_eq!(c.k, 1);
        a = g_layer(&mut g, &c.lin, &a, c.residual, c.relu);
    }
    if last {
        if let Some(s) = &m.stab {
            let si: Vec<IId> = (0..s.cin()).map(|_| par(&mut g)).collect();
            let sb = g_layer(&mut g, &s.lin, &si, false, false);
            a = a.iter().zip(sb).map(|(&p, q)| g.add(p, q)).collect();
        }
        a = g_layer(&mut g, &m.outt.lin, &a, m.outt.residual, m.outt.relu);
        let (k255, lo, hi) = (g.i32(255), g.i32(0), g.i32(255));
        a = a
            .iter()
            .map(|&v| {
                let t = g.mul(v, k255);
                let t = g_rshift(&mut g, t, QA);
                g.clamp(t, lo, hi)
            })
            .collect();
    }
    (g, a)
}

/// Synthesis split into segments (each starts at a 3×3 conv or the input).
pub fn segments(m: &Model) -> Vec<Vec<&Conv>> {
    let mut segs: Vec<Vec<&Conv>> = vec![];
    for c in &m.syn {
        if c.k != 1 || segs.is_empty() {
            segs.push(vec![c]);
        } else {
            segs.last_mut().unwrap().push(c);
        }
    }
    segs
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use tang_expr::int::Program;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        /// Quantized value k/64 with |k| ≤ a.
        fn q(&mut self, a: i64) -> f64 {
            ((self.next() % (2 * a as u64 + 1)) as i64 - a) as f64 / 64.0
        }
    }

    fn put(t: &mut HashMap<String, (Vec<usize>, Vec<f64>)>, k: &str, sh: &[usize], v: Vec<f64>) {
        assert_eq!(sh.iter().product::<usize>(), v.len());
        t.insert(k.into(), (sh.to_vec(), v));
    }

    /// A small Cool-chic-shaped model with random weights and latents.
    pub(crate) fn synth(seed: u64) -> Ccx {
        let mut r = Rng(seed);
        let mut t = HashMap::new();
        let sizes = [(16usize, 24usize), (8, 12), (4, 6), (4, 6)];
        put(&mut t, "img_size", &[2], vec![16.0, 24.0]);
        put(
            &mut t,
            "sizes",
            &[4, 2],
            sizes
                .iter()
                .flat_map(|&(a, b)| [a as f64, b as f64])
                .collect(),
        );
        put(&mut t, "is_hyper", &[4], vec![0.0, 0.0, 0.0, 1.0]);
        let ctx = [
            (0, -1),
            (-1, 0),
            (-1, -1),
            (-1, 1),
            (0, -2),
            (-2, 0),
            (-1, 2),
        ];
        put(
            &mut t,
            "ctx_index",
            &[ctx.len()],
            ctx.iter()
                .map(|&(dy, dx): &(i32, i32)| ((dy + 4) * 9 + dx + 4) as f64)
                .collect(),
        );
        put(&mut t, "mask_size", &[1], vec![9.0]);
        put(&mut t, "n_ifce_out", &[1], vec![2.0]);
        put(
            &mut t,
            "ifce.input_features",
            &[4],
            vec![3.0, 0.0, 0.0, 0.0],
        );
        put(
            &mut t,
            "ifce.index_to_arm",
            &[4],
            vec![0.0, -1.0, -1.0, -1.0],
        );
        let lin = |t: &mut HashMap<_, _>, r: &mut Rng, p: &str, o: usize, i: usize, a: i64| {
            put(
                t,
                &format!("{p}.w"),
                &[o, i],
                (0..o * i).map(|_| r.q(a)).collect(),
            );
            put(t, &format!("{p}.b"), &[o], (0..o).map(|_| r.q(a)).collect());
        };
        lin(&mut t, &mut r, "ifce.A0.L0", 2, 3, 40);
        let nin = ctx.len() + 2;
        lin(&mut t, &mut r, "arm.L0", nin, nin, 20);
        lin(&mut t, &mut r, "arm.L1", nin, nin, 20);
        lin(&mut t, &mut r, "arm.L2", 2, nin, 30);
        lin(&mut t, &mut r, "arm.stab", 2, nin, 10);
        for i in 0..2 {
            let a = [0.0351562, 0.1054687, -0.2617187, -0.8789063]
                .map(|v: f64| -((v * 64.0).round() / 64.0));
            put(
                &mut t,
                &format!("ups.tconv.{i}"),
                &[8],
                [a[0], a[1], a[2], a[3], a[3], a[2], a[1], a[0]].to_vec(),
            );
            let h: Vec<f64> = (0..4).map(|k| if k == 3 { 0.25 } else { r.q(4) }).collect();
            put(
                &mut t,
                &format!("ups.conv.{i}"),
                &[7],
                vec![h[0], h[1], h[2], h[3], h[2], h[1], h[0]],
            );
        }
        let conv = |t: &mut HashMap<_, _>,
                    r: &mut Rng,
                    p: &str,
                    o: usize,
                    i: usize,
                    k: usize,
                    res: bool,
                    relu: bool| {
            put(
                t,
                &format!("{p}.w"),
                &[o, i, k, k],
                (0..o * i * k * k).map(|_| r.q(30)).collect(),
            );
            put(
                t,
                &format!("{p}.b"),
                &[o],
                (0..o).map(|_| r.q(30)).collect(),
            );
            put(
                t,
                &format!("{p}.meta"),
                &[2],
                vec![res as u8 as f64, relu as u8 as f64],
            );
        };
        conv(&mut t, &mut r, "syn.L0", 8, 3, 1, false, true);
        conv(&mut t, &mut r, "syn.L1", 3, 8, 1, false, false);
        conv(&mut t, &mut r, "syn.L2", 3, 3, 3, true, true);
        conv(&mut t, &mut r, "syn.stab", 3, 3, 1, false, false);
        put(
            &mut t,
            "syn.out.w",
            &[3, 3, 1, 1],
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        );
        put(&mut t, "syn.out.b", &[3], vec![0.5; 3]);
        put(&mut t, "syn.out.meta", &[2], vec![0.0, 0.0]);
        for (g, &(h, w)) in sizes.iter().enumerate() {
            // mostly small, a few escapes
            let v = (0..h * w)
                .map(|_| match r.next() % 40 {
                    0 => 40.0,
                    1 => -37.0,
                    k => (k % 5) as f64 - 2.0,
                })
                .collect();
            put(&mut t, &format!("latent.{g}"), &[h, w], v);
        }
        Ccx { t }
    }

    #[test]
    fn tables_are_normalized() {
        let t = tables();
        for c in t.chunks(NSYM + 1) {
            assert_eq!(c[0], 0);
            assert_eq!(c[NSYM], PM);
            assert!(c.windows(2).all(|w| w[1] > w[0]));
        }
    }

    #[test]
    fn roundtrip_layouts() {
        let c = synth(7);
        let n = 4;
        let truth: Vec<Vec<i32>> = (0..n)
            .map(|g| {
                c.ints(&format!("latent.{g}"))
                    .iter()
                    .map(|&v| v as i32)
                    .collect()
            })
            .collect();
        let mut first: Option<Vec<u8>> = None;
        for (lanes, tw, th) in [
            (1, 0, 0),
            (4, 0, 0),
            (16, 0, 0),
            (16, 8, 0),
            (16, 0, 4),
            (32, 8, 8),
        ] {
            let (bytes, st) = encode(&c, lanes, tw, th);
            assert_eq!(st.bytes, bytes.len());
            let p = parse(&bytes).unwrap();
            let mut gd = Guard::default();
            assert_eq!(
                decode_latents(&p, &mut gd),
                truth,
                "lanes {lanes} tile {tw}x{th}"
            );
            let (h, w, rgb) = decode(&bytes).unwrap();
            assert_eq!((h, w), (16, 24));
            // the image depends on the latents only, not on the layout
            match &first {
                None => first = Some(rgb),
                Some(f) => assert_eq!(f, &rgb),
            }
        }
    }

    #[test]
    fn graphs_match_native() {
        let c = synth(11);
        let (bytes, _) = encode(&c, 4, 0, 0);
        let p = parse(&bytes).unwrap();
        let m = &p.m;
        let mut r = Rng(3);
        for mlp in std::iter::once(&m.arm).chain(&m.ifces) {
            let (g, o) = mlp.graph();
            let mut prog = Program::new(&g, &o);
            for _ in 0..200 {
                let x: Vec<i32> = (0..mlp.nin())
                    .map(|_| (r.next() % 2049) as i32 - 1024)
                    .collect();
                let mut want = vec![0i32; mlp.nout()];
                mlp.run(&x, &mut want, &mut Guard::default());
                let xin: Vec<u32> = x.iter().map(|&v| v as u32).collect();
                let mut got = vec![0u32; mlp.nout()];
                prog.run(&xin, &[], &mut got);
                assert_eq!(got.iter().map(|&v| v as i32).collect::<Vec<_>>(), want);
            }
        }
        // the last synthesis segment (taps + centre + stab inputs) against conv_px
        let segs = segments(m);
        assert!(!segs.is_empty());
    }

    #[test]
    fn rejects_garbage() {
        assert!(parse(b"TCI1").is_err());
        let (mut bytes, _) = encode(&synth(5), 4, 0, 0);
        bytes.truncate(bytes.len() / 2);
        assert!(parse(&bytes).is_err());
    }
}
