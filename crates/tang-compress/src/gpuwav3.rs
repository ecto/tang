//! `tc-gpu` v3 (stream magic `TCG3`): **one global** integer CDF 9/7
//! transform over the whole field, with the coefficients entropy-coded in
//! independently decodable **code-blocks** (JPEG2000-style).
//!
//! # Dependency rule
//!
//! - Subbands are decoded coarse → fine (LL, then levels `nlev−1 … 0`). A
//!   code-block may read **any coefficient of a coarser level** (its
//!   parent band: the parent and the parent's 6 face neighbours) — those are
//!   decoded first — and, inside its own region, every earlier line in
//!   raster order plus its own earlier samples on the line. Nothing outside
//!   the block at its own level is visible, so all blocks of one level decode
//!   in parallel (one rANS lane each: the parallelism is the block count).
//! - Contexts use the *residuals* `q − pred` of the visible cells
//!   (optional, `resctx`), predictions use the values.
//!
//! # Random access
//!
//! A sub-box needs, at every level, the coefficients in the support of the
//! integer synthesis lifting (±3 samples per half-band per level, a
//! conservative bound for the 4-step 9/7 lifting), plus — for decoding — the
//! parent cells their blocks condition on. Only those code-blocks (and the
//! outlier tiles touching the box) are read; the inverse lifting runs only
//! on the lines inside the needed region at each level.

use crate::bitio::{ByteReader, ByteWriter, DecodeError};
use crate::lmodel3::{code_learned_ad, feat_vec3, AdCfg, Adapt, Mix3, Net32};
use crate::codec::ErrorBound;
use crate::gpuwav::{
    code_res, dequant, feats, for_lines, gamma, init_of, lift_consts, prior_table, rate_control, read_priors, row_ctx, to_f32_bits, write_priors,
    BandInfo, CountSink, DecSink, Enc, GpuStats, RansDec, RecSink, Shape, Sink, MINIT, NDL, NMODEL, NOMODEL, NTAP,
};

/// Stream magic.
pub const MAGIC3: &[u8; 4] = b"TCG3";

/// v3 options.
#[derive(Clone, Debug)]
pub struct Gpu3Config {
    /// Error target.
    pub bound: ErrorBound,
    /// Code-block edge for 3D data (in coefficients).
    pub block: usize,
    /// Code-block edge for 2D data.
    pub block2d: usize,
    /// Outlier-tile edge (3D; 2D tiles are `4×` this).
    pub tile: usize,
    /// Residual-based contexts.
    pub resctx: bool,
    /// Deadzone rounding offset.
    pub rho: f64,
    /// rANS lanes per code-block: 1 = raster order (best ratio); more
    /// splits a block's lines into parity passes dealt to the lanes (more
    /// GPU parallelism, TCG1's rule inside the block).
    pub lanes: usize,
    /// Coefficient probability model.
    pub model: Model3,
    /// Compact lanes: a multi-lane block of `v` coefficients uses
    /// `clamp(8·v·lanes / V, 1, lanes)` lanes, `V` the full block volume
    /// (small coarse-level blocks stop paying `lanes` lanes of rANS framing,
    /// while no lane runs longer than 1/8 of a full block's lane), and the
    /// last lane's length is implied by the block size.
    pub compact: bool,
    /// Axis-grouped line taps ([`crate::gpuwav::BandInfo::ax`]).
    pub axis: bool,
    /// Per-lane adaptive probability maps over the learned model
    /// ([`crate::lmodel3::Adapt`]); ignored by static models.
    pub adapt: Option<AdCfg>,
}

/// Probability model of a v3 stream.
#[derive(Clone, Debug, Default)]
pub enum Model3 {
    /// Static per-context priors (hand contexts).
    #[default]
    Static,
    /// A pretrained integer MLP, referenced by fingerprint (register it with
    /// [`crate::learned::register`] before decoding).
    Pretrained(std::sync::Arc<crate::learned::Net>),
    /// An MLP trained at encode time; its weights are shipped in the stream.
    Overfit {
        /// Hidden widths.
        hidden: Vec<usize>,
        /// Samples per epoch.
        per_epoch: usize,
        /// Epochs.
        epochs: usize,
    },
}

impl Gpu3Config {
    /// Defaults: 32³ / 128² blocks, 64³ outlier tiles, residual contexts.
    pub fn new(bound: ErrorBound) -> Self {
        Self { bound, block: 32, block2d: 128, tile: 64, resctx: true, rho: 0.45, lanes: 1, model: Model3::Static, compact: true, axis: true, adapt: Some(ad_default()) }
    }
}

/// Predictor key of a code-block band (pass 0; passes add `p`).
pub fn key3(level: usize, mask: u8) -> usize {
    let l = if mask == 0 { 15 } else { level.min(14) };
    (l * 8 + mask as usize) * 8
}

/// Compact streams store predictor taps in units of `2^TAPQ` (Q12 → Q8).
const TAPQ: u32 = 4;

/// Cells per lane of compact-lane streams (0 = not compact).
fn lane_cells(cfg: &Gpu3Config, cb: [usize; 3]) -> usize {
    if cfg.compact && cfg.lanes > 1 {
        // a coarse block's per-lane chain stays ≤ 1/8 of a full block's, so
        // the coarse levels (decoded one after another) add little latency
        (cb.iter().product::<usize>() / (8 * cfg.lanes)).max(1).next_power_of_two()
    } else {
        0
    }
}

/// Code-blocks of a shape in coding order.
pub fn blocks_of(sh: &Shape, cb: [usize; 3], lanes: usize, lane_cells: usize, axis: bool) -> (Vec<BandInfo>, Vec<usize>) {
    let mut v = Vec::new();
    let mut band = Vec::new();
    for (bi, b) in sh.bands.iter().enumerate() {
        let pb = &sh.p.bands[bi];
        for z in (b.start[0]..b.end[0]).step_by(cb[0]) {
            for y in (b.start[1]..b.end[1]).step_by(cb[1]) {
                for x in (b.start[2]..b.end[2]).step_by(cb[2]) {
                    let mut k = b.clone();
                    k.start = [z, y, x];
                    k.end = [(z + cb[0]).min(b.end[0]), (y + cb[1]).min(b.end[1]), (x + cb[2]).min(b.end[2])];
                    k.org = b.start;
                    k.ax = axis;
                    let base = key3(pb.level, pb.mask);
                    if lanes <= 1 {
                        k.raster = true;
                        k.npass = 1;
                        k.pkey = [base; 8];
                    } else {
                        k.raster = false;
                        k.npass = crate::gpuwav::npass_of(k.start, k.end);
                        k.pkey = std::array::from_fn(|p| base + p);
                        k.nl = k.end.iter().zip(&k.start).map(|(e, s)| e - s).product::<usize>().checked_div(lane_cells).map_or(lanes, |v| v.clamp(1, lanes));
                    }
                    v.push(k);
                    band.push(bi);
                }
            }
        }
    }
    (v, band)
}

/// Code one block (one lane, static models). `rs` receives residuals.
#[allow(clippy::too_many_arguments)]
fn code_block<S: Sink>(
    sh: &Shape,
    b: &BandInfo,
    q: &mut [i32],
    mut rs: Option<&mut [i32]>,
    preds: &[[i32; NTAP]],
    m: &mut [u32],
    lanes: usize,
    net: Option<&Net32>,
    adc: AdCfg,
    s: &mut S,
    mut ir: Option<&mut IrCtx>,
) {
    let [_, ny, nx] = sh.dims;
    // TC_CPU_IR: evaluate the net with the tang-expr integer interpreter
    let mut prog = net.filter(|_| cpu_ir()).map(|n| n.program());
    let mut ad = vec![Adapt::new(adc); lanes.max(1)];
    for_lines(b, lanes.max(1), |p, lane, z, y| {
        let w = &preds[b.pkey[p]];
        let rc = row_ctx(sh, b, p, z, y);
        for x in b.start[2]..b.end[2] {
            let (f, pred) = feats(sh, b, &rc, q, rs.as_deref(), z, y, x, w);
            let i = (z * ny + y) * nx + x;
            let r = match (net, ir.as_deref_mut()) {
                // decoding through tang-expr graphs (shipped or TC_CPU_IRRES)
                (None, Some(c)) => c.res.res_static(&mut c.lanes[lane], c.mt, c.sw, f.sig as u32, f.e as u32, b.class as u32, f.sctx as u32),
                (Some(n), Some(c)) => {
                    let x = feat_vec3(&f, b.class);
                    let o = match c.mlp.as_mut() {
                        Some(p) => n.infer_ir(p, &x),
                        None => n.infer(&x),
                    };
                    c.lres.res_learned(&mut c.lanes[lane], c.mt, c.sw, &o, &mut c.adt[lane])
                }
                (None, None) => code_res::<S, false>(s, lane, m, b.class, &f, q[i].wrapping_sub(pred)),
                (Some(n), None) => {
                    let x = feat_vec3(&f, b.class);
                    let o = match prog.as_mut() {
                        Some(p) => n.infer_ir(p, &x),
                        None => n.infer(&x),
                    };
                    let mx = Mix3::new(&o);
                    code_learned_ad(s, lane, &mx, q[i].wrapping_sub(pred), &mut ad[lane])
                }
            };
            q[i] = r.wrapping_add(pred);
            if let Some(rs) = rs.as_deref_mut() {
                rs[i] = r;
            }
        }
    });
}

/// Default adaptive layer (`TC_AD=wa,lim,mask` overrides, for sweeps).
pub fn ad_default() -> AdCfg {
    static V: std::sync::OnceLock<AdCfg> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let v: Vec<u32> = std::env::var("TC_AD").unwrap_or("10,14,15".into()).split(',').map(|x| x.parse().unwrap()).collect();
        AdCfg { wa: v[0].min(16), lim: v[1].min(255), mask: v[2] & 15 }
    })
}

const AD_OFF: AdCfg = AdCfg { wa: 0, lim: 0, mask: 0 };

fn ad_pack(a: AdCfg) -> u64 {
    (a.wa | a.lim << 5 | a.mask << 13) as u64
}

fn ad_unpack(v: u64) -> AdCfg {
    AdCfg { wa: (v & 31).min(16) as u32, lim: (v >> 5 & 255) as u32, mask: (v >> 13 & 15) as u32 }
}

/// CPU decoding state for tang-expr residual graphs.
pub(crate) struct IrCtx<'a> {
    res: crate::irdec::CpuRes,
    lres: crate::irdec::CpuRes,
    mlp: Option<tang_expr::int::Program>,
    lanes: Vec<crate::irdec::Lane>,
    /// Per-lane adaptive state (fresh per block).
    adt: Vec<[u32; crate::lmodel3::AC * crate::lmodel3::AB]>,
    mt: &'a [u32],
    sw: &'a [u32],
}

/// Per-decode buffers shared by the blocks.
struct IrBase {
    dec: std::sync::Arc<crate::shipdec::Decoder>,
    mt: Vec<u32>,
    off: [u32; 2],
    sw: Vec<u32>,
}

impl IrBase {
    fn new(ps: &Parsed3, bytes: &[u8]) -> Option<Self> {
        let dec = ps.dec.clone().or_else(|| cpu_irres().then(builtin_decoder))?;
        let (mt, off) = crate::irdec::cpu_mt(&ps.init);
        Some(Self { dec, mt, off, sw: crate::irdec::stream_words(bytes) })
    }
}

fn builtin_decoder() -> std::sync::Arc<crate::shipdec::Decoder> {
    static D: std::sync::OnceLock<std::sync::Arc<crate::shipdec::Decoder>> = std::sync::OnceLock::new();
    D.get_or_init(|| std::sync::Arc::new(crate::shipdec::compile(&crate::shipdec::builtin_program(None, false)).unwrap())).clone()
}

fn cpu_irres() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("TC_CPU_IRRES").is_ok())
}

fn cpu_ir() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("TC_CPU_IR").is_ok())
}

fn block_dims(d3: [usize; 3], cfg_block: usize, block2d: usize) -> [usize; 3] {
    match d3.iter().filter(|&&d| d > 1).count() {
        3 => [cfg_block; 3],
        2 => [1, block2d, block2d],
        _ => [1, 1, block2d * block2d],
    }
}

fn tile_dims(d3: [usize; 3], t: usize) -> [usize; 3] {
    match d3.iter().filter(|&&d| d > 1).count() {
        3 => [t; 3],
        2 => [1, 4 * t, 4 * t],
        _ => [1, 1, 64 * t * t],
    }
}

fn grid(d3: [usize; 3], bd: [usize; 3]) -> Vec<([usize; 3], [usize; 3])> {
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

/// Static priors with a cost-based header: a context gets a transmitted
/// probability only when it saves more than ~12 bits over `p = ½`.
fn choose_priors3(n: &[[u32; 2]]) -> Vec<Option<u8>> {
    let t = prior_table();
    n.iter()
        .map(|&[n0, n1]| {
            if n0 + n1 == 0 {
                return None;
            }
            let cost = |p: u32| {
                let p = p as f64 / 65536.0;
                -(n1 as f64) * p.log2() - (n0 as f64) * (1.0 - p).log2()
            };
            let k = (0..64u8).min_by(|&a, &b| cost(t[a as usize]).total_cmp(&cost(t[b as usize])))?;
            if cost(32768) - cost(t[k as usize]) > 12.0 {
                Some(k)
            } else {
                None
            }
        })
        .collect()
}

/// LS predictor fit per key over the blocks.
fn fit3(sh: &Shape, blocks: &[BandInfo], q: &[i32], tq: u32) -> Vec<Option<[i32; NTAP]>> {
    let nkey = crate::gpuwav::NKEY;
    let mut g = vec![[[0.0f64; NTAP]; NTAP]; nkey];
    let mut h = vec![[0.0f64; NTAP]; nkey];
    let mut n = vec![0usize; nkey];
    let mut obs: Vec<Vec<([i32; NTAP], i32)>> = vec![Vec::new(); nkey];
    let zero = [0i32; NTAP];
    let [_, ny, nx] = sh.dims;
    for b in blocks {
        for_lines(b, 1, |p, _, z, y| {
            let k = b.pkey[p];
            let rc = row_ctx(sh, b, p, z, y);
            for x in b.start[2]..b.end[2] {
                let (f, _) = feats(sh, b, &rc, q, None, z, y, x, &zero);
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
    (0..nkey)
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
                w[i] = ((wi * 4096.0 / (1 << tq) as f64).round().clamp(-(32767 >> tq) as f64, (32767 >> tq) as f64) as i32) << tq;
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

fn used_keys3(blocks: &[BandInfo]) -> Vec<usize> {
    let mut k: Vec<usize> = blocks.iter().flat_map(|b| b.pkey[..b.npass].to_vec()).collect();
    k.sort_unstable();
    k.dedup();
    k
}

fn words_le(v: &[u16], w: &mut ByteWriter) {
    for &x in v {
        w.bytes(&x.to_le_bytes());
    }
}

/// `(features, residual)` of every coefficient in coding order (strided down
/// to at most `max` samples).
fn collect3(sh: &Shape, blocks: &[BandInfo], q: &mut [i32], resctx: bool, preds: &[[i32; NTAP]], lanes: usize, max: usize) -> Vec<crate::learned::Sample> {
    let n: usize = q.len();
    let stride = n.div_ceil(max.max(1)).max(1);
    let mut rs = vec![0i32; n];
    let mut out = Vec::new();
    let [_, ny, nx] = sh.dims;
    let mut k = 0usize;
    for b in blocks {
        for_lines(b, lanes.max(1), |p, _, z, y| {
            let rc = row_ctx(sh, b, p, z, y);
            for x in b.start[2]..b.end[2] {
                let (f, pred) = feats(sh, b, &rc, q, resctx.then_some(&rs[..]), z, y, x, &preds[b.pkey[p]]);
                let i = (z * ny + y) * nx + x;
                let r = q[i].wrapping_sub(pred);
                rs[i] = r;
                if k.is_multiple_of(stride) {
                    out.push(crate::learned::Sample { x: feat_vec3(&f, b.class), r });
                }
                k += 1;
            }
        });
    }
    out
}

/// Training samples for a field at a PSNR/RMSE target (the coefficients the
/// v3 coder would see at the rate-controlled step).
pub fn samples3(x: &[f32], dims: &[usize], cfg: &Gpu3Config, threads: usize, max: usize) -> Vec<crate::learned::Sample> {
    let big = dims.iter().copied().max().unwrap_or(1).max(2) * 2;
    let got = std::sync::Mutex::new(Vec::new());
    rate_control(x, dims, big, cfg.bound, cfg.rho, threads, |enc, dg, _| {
        let shapes = enc.shapes_at(dg);
        let sh = &shapes[0];
        let (qs, _) = enc.quantize(dg, cfg.rho);
        let cb = block_dims(sh.dims, cfg.block, cfg.block2d);
        let lc = lane_cells(cfg, cb);
        let (blocks, _) = blocks_of(sh, cb, cfg.lanes, lc, cfg.axis);
        let fitted = fit3(sh, &blocks, &qs[0], if lc > 0 { TAPQ } else { 0 });
        let preds: Vec<[i32; NTAP]> = fitted.iter().map(|p| p.unwrap_or([0; NTAP])).collect();
        let mut q = qs[0].clone();
        *got.lock().unwrap() = collect3(sh, &blocks, &mut q, cfg.resctx, &preds, cfg.lanes, max);
        (Vec::new(), GpuStats::default())
    });
    got.into_inner().unwrap()
}

/// Encode as a v3 stream.
pub fn encode3_f32(x: &[f32], dims: &[usize], cfg: &Gpu3Config, threads: usize) -> (Vec<u8>, GpuStats) {
    // a brick larger than every axis: Enc then holds the global transform
    let big = dims.iter().copied().max().unwrap_or(1).max(2) * 2;
    rate_control(x, dims, big, cfg.bound, cfg.rho, threads, |enc, dg, mx| encode_at3(enc, dims, cfg, dg, mx))
}

fn encode_at3(enc: &Enc, dims: &[usize], cfg: &Gpu3Config, dg: f64, maxabs: Option<f64>) -> (Vec<u8>, GpuStats) {
    let shapes = enc.shapes_at(dg);
    assert_eq!(shapes.len(), 1);
    let sh = &shapes[0];
    let d3 = sh.dims;
    let (qs, dl) = enc.quantize(dg, cfg.rho);
    let cb = block_dims(d3, cfg.block, cfg.block2d);
    let lc = lane_cells(cfg, cb);
    let (blocks, _) = blocks_of(sh, cb, cfg.lanes, lc, cfg.axis);
    let fitted = if std::env::var("TC_NOPRED").is_ok() { vec![None; crate::gpuwav::NKEY] } else { fit3(sh, &blocks, &qs[0], if lc > 0 { TAPQ } else { 0 }) };
    let preds: Vec<[i32; NTAP]> = fitted.iter().map(|p| p.unwrap_or([0; NTAP])).collect();
    let mut cs = enc.reconstruct(&shapes, &qs, &dl);
    let scale = enc.scale();
    let g = 2f64.powi(enc.e);
    let corr = maxabs.map(|b| ((2.0 * b / g).floor() as i64).clamp(0, 1 << 30) as i32).unwrap_or(0);
    let td = tile_dims(d3, cfg.tile);
    let tiles = grid(d3, td);
    let n = enc.x.len();
    // corrections + escapes, per outlier tile (global indices)
    let mut outl: Vec<Vec<(usize, i32)>> = vec![Vec::new(); tiles.len()];
    let mut escs: Vec<Vec<(usize, u32)>> = vec![Vec::new(); tiles.len()];
    if let Some(b) = maxabs {
        let c = &mut cs[0];
        for (ti, (o, td)) in tiles.iter().enumerate() {
            for z in 0..td[0] {
                for y in 0..td[1] {
                    for xx in 0..td[2] {
                        let li = (z * td[1] + y) * td[2] + xx;
                        let gi = ((o[0] + z) * d3[1] + o[1] + y) * d3[2] + o[2] + xx;
                        let xv = enc.x[gi];
                        let yv = f32::from_bits(to_f32_bits(c[gi].wrapping_add(enc.mean_fix), scale)) as f64;
                        let r = xv as f64 - yv;
                        if r.abs() > b && corr > 0 {
                            let k = (r / g / corr as f64).round().clamp(-(1 << 20) as f64, (1 << 20) as f64) as i32;
                            if k != 0 {
                                c[gi] = c[gi].wrapping_add(k.wrapping_mul(corr));
                                outl[ti].push((li, k));
                            }
                        }
                        let yv = f32::from_bits(to_f32_bits(c[gi].wrapping_add(enc.mean_fix), scale)) as f64;
                        if (xv as f64 - yv).abs() > b {
                            escs[ti].push((li, xv.to_bits()));
                        }
                    }
                }
            }
        }
    }
    // priors: counts over every block decision (sequential: contexts need
    // the residuals of coarser blocks)
    let mut q = qs[0].clone();
    let mut rs = vec![0i32; n];
    let mut cnt = CountSink { n: vec![[0; 2]; NMODEL] };
    let mut m0 = vec![MINIT; NMODEL];
    for b in &blocks {
        code_block(sh, b, &mut q, cfg.resctx.then_some(&mut rs[..]), &preds, &mut m0, cfg.lanes, None, AD_OFF, &mut cnt, None);
    }
    // learned model (overfit: trained here on this field's coefficients)
    let mut q2 = qs[0].clone();
    let (net, net_bytes): (Option<Net32>, Vec<u8>) = match &cfg.model {
        Model3::Static => (None, vec![0]),
        Model3::Pretrained(n) => {
            let mut v = vec![1];
            v.extend(n.fingerprint().to_le_bytes());
            (Some(Net32::new(n)), v)
        }
        Model3::Overfit { hidden, per_epoch, epochs } => {
            let smp = collect3(sh, &blocks, &mut q2, cfg.resctx, &preds, cfg.lanes, 2_000_000);
            let tc = crate::learned::TrainConfig { hidden: hidden.clone(), epochs: *epochs, per_epoch: *per_epoch, batch: 256, lr: 4e-3, apm: false, seed: 1 };
            let n = crate::learned::train(&smp, &tc);
            let mut v = vec![2];
            let nb = n.to_bytes();
            v.extend((nb.len() as u32).to_le_bytes());
            v.extend(nb);
            (Some(Net32::new(&n)), v)
        }
    };
    let adc = if net.is_some() { cfg.adapt.unwrap_or(AD_OFF) } else { AD_OFF };
    let priors = if net.is_some() { vec![None; NMODEL] } else { choose_priors3(&cnt.n) };
    let mut init = init_of(&priors);
    let mut bw: Vec<Vec<u8>> = Vec::with_capacity(blocks.len());
    let dbg = std::env::var("TC_STATS").is_ok();
    let (mut ideal, mut nlanes, mut wbytes) = (0f64, 0usize, 0usize);
    for b in &blocks {
        let nl = cfg.lanes.max(1);
        let mut sink = RecSink { syms: vec![Vec::new(); nl] };
        code_block(sh, b, &mut q, cfg.resctx.then_some(&mut rs[..]), &preds, &mut init, nl, net.as_ref(), adc, &mut sink, None);
        let lw: Vec<Vec<u16>> = sink.syms.iter().map(|s| crate::gpuwav::rans_encode(s)).collect();
        if dbg {
            ideal += sink.syms.iter().flatten().map(|&(_, f)| 16.0 - (f as f64).log2()).sum::<f64>();
            nlanes += lw.iter().filter(|l| !l.is_empty()).count();
            wbytes += 2 * lw.iter().map(|l| l.len()).sum::<usize>();
        }
        let mut w = ByteWriter::new();
        if nl > 1 {
            let used = if lc > 0 { b.nl } else { nl };
            debug_assert!(lw[used..].iter().all(|l| l.is_empty()));
            let nv = if lc > 0 { used - 1 } else { used };
            let mut prev = 0i64;
            for l in &lw[..nv] {
                if lc > 0 {
                    // compact: zig-zag delta from the previous lane (lanes
                    // get rows round-robin, so their lengths are close)
                    w.svarint(l.len() as i64 - prev);
                    prev = l.len() as i64;
                } else {
                    w.varint(l.len() as u64);
                }
            }
            if w.buf.len() % 2 == 1 {
                w.u8(0);
            }
        }
        for l in &lw {
            words_le(l, &mut w);
        }
        bw.push(w.buf);
    }
    // outlier tiles
    let mut tp: Vec<Vec<u8>> = Vec::with_capacity(tiles.len());
    for ti in 0..tiles.len() {
        let mut sink = RecSink { syms: vec![Vec::new()] };
        if maxabs.is_some() {
            let mut om = vec![MINIT; NOMODEL];
            gamma::<_, true>(&mut sink, 0, &mut om, 0, outl[ti].len() as u32);
            let mut last: i64 = -1;
            for &(i, k) in &outl[ti] {
                gamma::<_, true>(&mut sink, 0, &mut om, 32, (i as i64 - last - 1) as u32);
                last = i as i64;
                sink.raw(0, (k < 0) as u32, 1);
                gamma::<_, true>(&mut sink, 0, &mut om, 64, k.unsigned_abs() - 1);
            }
        }
        let words = crate::gpuwav::rans_encode(&sink.syms[0]);
        let mut w = ByteWriter::new();
        w.varint(words.len() as u64);
        w.varint(escs[ti].len() as u64);
        let mut last = 0usize;
        for (j, &(i, bits)) in escs[ti].iter().enumerate() {
            w.varint(if j == 0 { i } else { i - last } as u64);
            last = i;
            w.bytes(&bits.to_le_bytes());
        }
        if w.buf.len() % 2 == 1 {
            w.u8(0);
        }
        words_le(&words, &mut w);
        tp.push(w.buf);
    }
    // header
    let mut w = ByteWriter::new();
    w.bytes(MAGIC3);
    w.varint(dims.len() as u64);
    for &d in dims {
        w.varint(d as u64);
    }
    for a in 0..3 {
        w.varint(cb[a] as u64);
    }
    for a in 0..3 {
        w.varint(td[a] as u64);
    }
    w.u8(cfg.resctx as u8);
    let has_ad = adc.wa > 0;
    if lc > 0 || cfg.axis || has_ad {
        w.varint((cfg.lanes.max(1) | 1 << 9 | (cfg.axis as usize) << 10 | (has_ad as usize) << 11) as u64);
        w.u8(if lc > 0 { lc.trailing_zeros() as u8 + 1 } else { 0 });
        if has_ad {
            w.varint(ad_pack(adc));
        }
    } else {
        w.varint(cfg.lanes.max(1) as u64);
    }
    w.bytes(&net_bytes);
    w.svarint(enc.e as i64);
    w.svarint(enc.mean_fix as i64);
    w.f64(dg);
    w.bytes(&dl);
    w.varint(corr as u64);
    let uk = used_keys3(&blocks);
    let pred_at = w.buf.len();
    if lc > 0 {
        // compact layout: presence bitset, then the taps
        for c in uk.chunks(8) {
            w.u8(c.iter().enumerate().map(|(i, &k)| (fitted[k].is_some() as u8) << i).sum());
        }
        for &k in &uk {
            for &t in fitted[k].iter().flatten() {
                w.svarint((t >> TAPQ) as i64);
            }
        }
    } else {
        for &k in &uk {
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
    }
    let pred_bytes = w.buf.len() - pred_at;
    write_priors(&mut w, &priors);
    let hdr_fixed = w.buf.len();
    for b in &bw {
        w.varint(b.len() as u64);
    }
    for t in &tp {
        w.varint(t.len() as u64);
    }
    if w.buf.len() % 2 == 1 {
        w.u8(0);
    }
    let header_bytes = w.buf.len();
    if dbg {
        let bb: usize = bw.iter().map(|b| b.len()).sum();
        eprintln!(
            "STATS hdr_fixed {hdr_fixed} index {} blocks {} block_bytes {bb} ideal_bytes {:.0} pred_bytes {pred_bytes} nkeys {} nfit {} nonempty_lanes {nlanes} word_bytes {wbytes} tiles_bytes {}",
            header_bytes - hdr_fixed,
            blocks.len(),
            ideal / 8.0,
            uk.len(),
            uk.iter().filter(|&&k| fitted[k].is_some()).count(),
            tp.iter().map(|t| t.len()).sum::<usize>()
        );
    }
    for b in &bw {
        w.bytes(b);
    }
    for t in &tp {
        w.bytes(t);
    }
    // stats from the exact output
    let c = &cs[0];
    let mut se = 0.0;
    let mut mx = 0.0f64;
    let mut esc_at: Vec<(usize, u32)> = Vec::new();
    for (ti, (o, td)) in tiles.iter().enumerate() {
        for &(li, bits) in &escs[ti] {
            let (z, y, xx) = (li / (td[1] * td[2]), (li / td[2]) % td[1], li % td[2]);
            esc_at.push((((o[0] + z) * d3[1] + o[1] + y) * d3[2] + o[2] + xx, bits));
        }
    }
    esc_at.sort_unstable();
    let mut ei = esc_at.iter().peekable();
    for gi in 0..n {
        let mut yv = f32::from_bits(to_f32_bits(c[gi].wrapping_add(enc.mean_fix), scale));
        if ei.peek().is_some_and(|e| e.0 == gi) {
            yv = f32::from_bits(ei.next().unwrap().1);
        }
        let e = enc.x[gi] as f64 - yv as f64;
        se += e * e;
        mx = mx.max(e.abs());
    }
    let st = GpuStats {
        bytes: w.buf.len(),
        bits_per_value: 8.0 * w.buf.len() as f64 / n as f64,
        header_bytes,
        index_bytes: header_bytes - hdr_fixed,
        outlier_bytes: tp.iter().map(|t| t.len()).sum(),
        n_outliers: outl.iter().map(|v| v.len()).sum(),
        n_escapes: esc_at.len(),
        rmse: (se / n as f64).sqrt(),
        max_abs: mx,
        delta: dg * g,
    };
    (w.buf, st)
}

// ------------------------------------------------------------------ parse

/// An outlier tile.
#[derive(Clone, Debug)]
pub struct TileRef {
    /// Origin.
    pub origin: [usize; 3],
    /// Extent.
    pub dims: [usize; 3],
    /// Word range `[lo, hi)` of its lane (absolute 16-bit word index).
    pub words: (usize, usize),
    /// Byte range of the whole tile payload.
    pub bytes: (usize, usize),
    /// Raw escapes `(tile-local index, f32 bits)`.
    pub esc: Vec<(usize, u32)>,
}

/// A parsed v3 stream.
#[derive(Clone, Debug)]
pub struct Parsed3 {
    /// Original dims.
    pub dims: Vec<usize>,
    /// 3D dims.
    pub d3: [usize; 3],
    /// Code-block extent.
    pub cb: [usize; 3],
    /// Residual contexts.
    pub resctx: bool,
    /// Grid exponent.
    pub e: i32,
    /// Mean on the grid.
    pub mean_fix: i32,
    /// Centroids.
    pub dl: [u8; NDL],
    /// Correction step.
    pub corr: i32,
    /// Predictors per key.
    pub preds: Vec<[i32; NTAP]>,
    /// Static model probabilities.
    pub init: Vec<u32>,
    /// The global shape (pyramid + bands with steps).
    pub sh: Shape,
    /// Code-blocks in coding order.
    pub blocks: Vec<BandInfo>,
    /// Band index of each block.
    pub block_band: Vec<usize>,
    /// Byte range of each block payload.
    pub block_words: Vec<(usize, usize)>,
    /// Lanes per block.
    pub lanes: usize,
    /// Word ranges of each block's lanes.
    pub block_lanes: Vec<Vec<(usize, usize)>>,
    /// Outlier tiles.
    pub tiles: Vec<TileRef>,
    /// Learned model (None = static priors).
    pub net: Option<Net32>,
    /// Adaptive layer of the learned model.
    pub ad: AdCfg,
    /// Header + index bytes.
    pub header_bytes: usize,
    /// Decoder shipped in the stream (`TCS1`), replacing the built-in coding path.
    pub dec: Option<std::sync::Arc<crate::shipdec::Decoder>>,
}

impl Parsed3 {
    /// Output scale.
    pub fn scale(&self) -> f32 {
        f32::from_bits(((self.e + 127) as u32) << 23)
    }
    /// Block index ranges per level, in decode order.
    pub fn level_groups(&self) -> Vec<(usize, usize)> {
        let mut v: Vec<(usize, usize)> = Vec::new();
        let mut cur = usize::MAX;
        for (i, &bi) in self.block_band.iter().enumerate() {
            let l = self.sh.p.bands[bi].level;
            if l != cur {
                v.push((i, i + 1));
                cur = l;
            } else {
                v.last_mut().unwrap().1 = i + 1;
            }
        }
        v
    }
}

/// Parse a v3 stream, plain (`TCG3`) or carrying its decoder (`TCS1`, see
/// [`crate::shipdec`]). Offsets in the result index `bytes` either way.
pub fn parse3(bytes: &[u8]) -> Result<Parsed3, DecodeError> {
    let (dec, inner) = crate::shipdec::open(bytes)?;
    let Some(dec) = dec else { return parse3_inner(bytes, false) };
    let base = bytes.len() - inner.len();
    let mut ps = parse3_inner(inner, dec.mlp.is_some())?;
    for w in ps.block_words.iter_mut() {
        *w = (w.0 + base, w.1 + base);
    }
    for lv in ps.block_lanes.iter_mut() {
        for w in lv.iter_mut() {
            *w = (w.0 + base / 2, w.1 + base / 2);
        }
    }
    for t in ps.tiles.iter_mut() {
        t.words = (t.words.0 + base / 2, t.words.1 + base / 2);
        t.bytes = (t.bytes.0 + base, t.bytes.1 + base);
    }
    ps.header_bytes += base;
    ps.dec = Some(dec);
    Ok(ps)
}

/// Parse a plain v3 stream; `ext_net`: the model is shipped as a graph, so
/// an unregistered fingerprint is not an error.
fn parse3_inner(bytes: &[u8], ext_net: bool) -> Result<Parsed3, DecodeError> {
    let mut r = ByteReader::new(bytes);
    if r.take(4)? != MAGIC3 {
        return Err(DecodeError::Invalid("not a tc-gpu v3 stream"));
    }
    let nd = r.varint()? as usize;
    if nd == 0 || nd > 3 {
        return Err(DecodeError::Invalid("rank"));
    }
    let dims: Vec<usize> = (0..nd).map(|_| r.varint().map(|v| v as usize)).collect::<Result<_, _>>()?;
    let d3 = crate::wavelet::as3(&dims);
    let mut cb = [0usize; 3];
    let mut td = [0usize; 3];
    for v in cb.iter_mut().chain(td.iter_mut()) {
        *v = r.varint()? as usize;
        if *v == 0 {
            return Err(DecodeError::Invalid("block"));
        }
    }
    let resctx = r.u8()? != 0;
    let lv = r.varint()? as usize;
    let (lanes, lane_cells, axis, ad) = if lv & 1 << 9 != 0 {
        let t = r.u8()?;
        let ad = if lv & 1 << 11 != 0 { ad_unpack(r.varint()?) } else { AD_OFF };
        (lv & 511, if t > 0 { 1usize << (t - 1).min(30) } else { 0 }, lv & 1 << 10 != 0, ad)
    } else {
        (lv, 0, false, AD_OFF)
    };
    if lanes == 0 || lanes > 256 {
        return Err(DecodeError::Invalid("lanes"));
    }
    let net = match r.u8()? {
        0 => None,
        1 => {
            let b = r.take(8)?;
            let fp = u64::from_le_bytes(b.try_into().unwrap());
            match crate::learned::lookup(fp) {
                Some(n) => Some(Net32::new(&n)),
                None if ext_net => Some(crate::shipdec::placeholder_net()),
                None => return Err(DecodeError::Invalid("pretrained model not registered")),
            }
        }
        2 => {
            let b = r.take(4)?;
            let n = u32::from_le_bytes(b.try_into().unwrap()) as usize;
            Some(Net32::new(&crate::learned::Net::from_bytes(r.take(n)?)?))
        }
        _ => return Err(DecodeError::Invalid("model tag")),
    };
    let e = r.svarint()? as i32;
    let mean_fix = r.svarint()? as i32;
    let delta_g = r.f64()?;
    let mut dl = [0u8; NDL];
    dl.copy_from_slice(r.take(NDL)?);
    let corr = r.varint()? as i32;
    let mut sh = Shape::new(d3);
    sh.set_steps(delta_g);
    let (blocks, block_band) = blocks_of(&sh, cb, lanes, lane_cells, axis);
    let mut preds = vec![[0i32; NTAP]; crate::gpuwav::NKEY];
    let uk = used_keys3(&blocks);
    let has: Vec<bool> = if lane_cells > 0 {
        let bits: Vec<u8> = (0..uk.len().div_ceil(8)).map(|_| r.u8()).collect::<Result<_, _>>()?;
        (0..uk.len()).map(|i| bits[i / 8] >> (i % 8) & 1 != 0).collect()
    } else {
        Vec::new()
    };
    for (i, &k) in uk.iter().enumerate() {
        if if lane_cells > 0 { has[i] } else { r.u8()? != 0 } {
            for t in preds[k].iter_mut() {
                let v = r.svarint()?;
                *t = if lane_cells > 0 { (v.clamp(-32767 >> TAPQ, 32767 >> TAPQ) as i32) << TAPQ } else { v as i32 };
            }
        }
    }
    let init = read_priors(&mut r)?;
    let tg = grid(d3, td);
    let bsz: Vec<usize> = (0..blocks.len()).map(|_| r.varint().map(|v| v as usize)).collect::<Result<_, _>>()?;
    let tsz: Vec<usize> = (0..tg.len()).map(|_| r.varint().map(|v| v as usize)).collect::<Result<_, _>>()?;
    let header_bytes = r.position().next_multiple_of(2);
    let mut off = header_bytes;
    let mut block_words = Vec::with_capacity(blocks.len());
    let mut block_lanes = Vec::with_capacity(blocks.len());
    for (bi, &n) in bsz.iter().enumerate() {
        let end = off + n;
        if end > bytes.len() || n % 2 == 1 {
            return Err(DecodeError::Truncated);
        }
        let mut lv = Vec::with_capacity(lanes);
        if lanes == 1 {
            lv.push((off / 2, end / 2));
        } else {
            let mut br = ByteReader::new(&bytes[off..end]);
            let used = if lane_cells > 0 { blocks[bi].nl } else { lanes };
            let nv = if lane_cells > 0 { used - 1 } else { used };
            let mut nw = Vec::with_capacity(nv);
            let mut prev = 0i64;
            for _ in 0..nv {
                let k = if lane_cells > 0 { prev + br.svarint()? } else { br.varint()? as i64 };
                if k < 0 {
                    return Err(DecodeError::Invalid("lane length"));
                }
                nw.push(k as usize);
                prev = k;
            }
            let mut w = (off + br.position().next_multiple_of(2)) / 2;
            for &k in &nw {
                lv.push((w, w + k));
                w += k;
            }
            if w * 2 > end {
                return Err(DecodeError::Truncated);
            }
            if nv < used {
                lv.push((w, end / 2));
                w = end / 2;
            }
            while lv.len() < lanes {
                lv.push((w, w));
            }
        }
        block_words.push((off, end));
        block_lanes.push(lv);
        off = end;
    }
    let mut tiles = Vec::with_capacity(tg.len());
    for (ti, (o, d)) in tg.iter().enumerate() {
        let end = off + tsz[ti];
        if end > bytes.len() {
            return Err(DecodeError::Truncated);
        }
        let mut br = ByteReader::new(&bytes[off..end]);
        let nw = br.varint()? as usize;
        let ne = br.varint()? as usize;
        let mut esc = Vec::with_capacity(ne);
        let mut last = 0;
        for i in 0..ne {
            let g = br.varint()? as usize;
            let idx = if i == 0 { g } else { last + g };
            last = idx;
            let b = br.take(4)?;
            esc.push((idx, u32::from_le_bytes([b[0], b[1], b[2], b[3]])));
        }
        let ws = (off + br.position().next_multiple_of(2)) / 2;
        if (ws + nw) * 2 > end {
            return Err(DecodeError::Truncated);
        }
        tiles.push(TileRef { origin: *o, dims: *d, words: (ws, ws + nw), bytes: (off, end), esc });
        off = end;
    }
    Ok(Parsed3 { dims, d3, cb, resctx, e, mean_fix, dl, corr, preds, init, sh, blocks, block_band, block_words, lanes, block_lanes, tiles, header_bytes, net, ad, dec: None })
}

// ------------------------------------------------------------------ decode

fn words(bytes: &[u8], a: usize, b: usize) -> Vec<u16> {
    (a..b).map(|i| u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]])).collect()
}

#[derive(Clone, Copy)]
struct Ptr(*mut i32);
unsafe impl Send for Ptr {}
unsafe impl Sync for Ptr {}

impl Ptr {
    fn get(&self) -> *mut i32 {
        self.0
    }
}

fn decode_block(ps: &Parsed3, bytes: &[u8], bi: usize, q: Ptr, rs: Ptr, n: usize, irb: Option<&IrBase>) -> Result<(), DecodeError> {
    if let Some(ib) = irb {
        let d = &ib.dec;
        let mut c = IrCtx {
            res: crate::irdec::CpuRes::new(&d.stat.0, &d.stat.1, false, [0, 0]),
            lres: if ps.ad.wa > 0 && ps.net.is_some() {
                crate::irdec::CpuRes::new_ad(&d.learned_ad.0, &d.learned_ad.1, ib.off, ps.ad)
            } else {
                crate::irdec::CpuRes::new(&d.learned.0, &d.learned.1, true, ib.off)
            },
            mlp: d.mlp_program(),
            lanes: ps.block_lanes[bi].iter().map(|&(a, b)| crate::irdec::Lane::new(&ib.sw, a as u32, b as u32)).collect(),
            adt: vec![crate::lmodel3::ad_init(); ps.lanes.max(1)],
            mt: &ib.mt,
            sw: &ib.sw,
        };
        let q = unsafe { std::slice::from_raw_parts_mut(q.get(), n) };
        let rs = unsafe { std::slice::from_raw_parts_mut(rs.get(), n) };
        let mut m = ps.init.clone();
        let mut sink = DecSink { lanes: Vec::new() };
        code_block(&ps.sh, &ps.blocks[bi], q, ps.resctx.then_some(rs), &ps.preds, &mut m, ps.lanes, ps.net.as_ref(), ps.ad, &mut sink, Some(&mut c));
        if c.lanes.iter().any(|l| l.overrun()) {
            return Err(DecodeError::Truncated);
        }
        return Ok(());
    }
    let lw: Vec<Vec<u16>> = ps.block_lanes[bi].iter().map(|&(a, b)| words(bytes, a, b)).collect();
    let mut sink = DecSink { lanes: lw.iter().map(|w| RansDec::new(w)).collect() };
    let mut m = ps.init.clone();
    // blocks write disjoint cells and read only coarser levels (decoded
    // before this level started) and their own cells
    let q = unsafe { std::slice::from_raw_parts_mut(q.get(), n) };
    let rs = unsafe { std::slice::from_raw_parts_mut(rs.get(), n) };
    code_block(&ps.sh, &ps.blocks[bi], q, ps.resctx.then_some(rs), &ps.preds, &mut m, ps.lanes, ps.net.as_ref(), ps.ad, &mut sink, None);
    if sink.lanes.iter().any(|l| l.overrun()) {
        return Err(DecodeError::Truncated);
    }
    Ok(())
}

fn run_par(items: &[usize], threads: usize, f: impl Fn(usize) -> Result<(), DecodeError> + Sync) -> Result<(), DecodeError> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    if threads <= 1 || items.len() <= 1 {
        for &i in items {
            f(i)?;
        }
        return Ok(());
    }
    let next = AtomicUsize::new(0);
    let err = Mutex::new(None);
    std::thread::scope(|s| {
        for _ in 0..threads.min(items.len()) {
            s.spawn(|| loop {
                let t = next.fetch_add(1, Ordering::Relaxed);
                if t >= items.len() {
                    break;
                }
                if let Err(e) = f(items[t]) {
                    *err.lock().unwrap() = Some(e);
                }
            });
        }
    });
    match err.into_inner().unwrap() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// Half-open interval list per axis (a union of at most a few ranges).
type Need = [Vec<(usize, usize)>; 3];

fn inside(v: &[(usize, usize)], i: usize) -> bool {
    v.iter().any(|&(a, b)| i >= a && i < b)
}

fn hits(v: &[(usize, usize)], a: usize, b: usize) -> bool {
    v.iter().any(|&(c, d)| c < b && a < d)
}

/// Per level: sample-domain need (`ext = lens[l]`) and coefficient-domain
/// need, for an output box `[lo, hi)`. Margin 3 per half-band per level.
fn needs(sh: &Shape, lo: [usize; 3], hi: [usize; 3]) -> (Vec<Need>, Vec<Need>) {
    let p = &sh.p;
    let mut samp: Vec<Need> = vec![[vec![(lo[0], hi[0])], vec![(lo[1], hi[1])], vec![(lo[2], hi[2])]]];
    let mut coef: Vec<Need> = Vec::new();
    for l in 0..p.nlev() {
        let cur = samp[l].clone();
        let mut c: Need = [vec![], vec![], vec![]];
        let mut next: Need = [vec![], vec![], vec![]];
        for a in 0..3 {
            let n = p.lens[l][a];
            if !p.split[l][a] {
                c[a] = cur[a].clone();
                next[a] = cur[a].clone();
                continue;
            }
            let ns = n.div_ceil(2);
            let nd = n / 2;
            for &(x0, x1) in &cur[a] {
                let i0 = (x0 / 2).saturating_sub(3);
                let i1 = (x1.saturating_sub(1)) / 2 + 4;
                c[a].push((i0, i1.min(ns)));
                c[a].push((ns + i0.min(nd), ns + i1.min(nd)));
                next[a].push((i0, i1.min(ns)));
            }
        }
        coef.push(c);
        samp.push(next);
    }
    (samp, coef)
}

/// Integer inverse on the whole field, optionally restricted to the lines
/// a box needs; lines are split over `threads`.
fn inverse_window(sh: &Shape, x: &mut [i32], win: Option<(&[Need], &[Need])>, threads: usize) {
    let p = &sh.p;
    let k = lift_consts();
    let st = [p.dims[1] * p.dims[2], p.dims[2], 1];
    let n = x.len();
    let xp = Ptr(x.as_mut_ptr());
    for l in (0..p.nlev()).rev() {
        let mut done = [false; 3];
        for a in [2usize, 1, 0] {
            if !p.split[l][a] {
                continue;
            }
            let ext = p.lens[l];
            let o: Vec<usize> = (0..3).filter(|&b| b != a).collect();
            let sel = |b: usize, i: usize| match win {
                None => true,
                Some((s, c)) => {
                    if done[b] {
                        inside(&s[l][b], i)
                    } else {
                        inside(&c[l][b], i)
                    }
                }
            };
            let (o0, o1) = (o[0], o[1]);
            let mut lines: Vec<usize> = Vec::new();
            for i in (0..ext[o0]).filter(|&i| sel(o0, i)) {
                for j in (0..ext[o1]).filter(|&j| sel(o1, j)) {
                    lines.push(i * st[o0] + j * st[o1]);
                }
            }
            let chunks: Vec<usize> = (0..threads.max(1)).collect();
            let per = lines.len().div_ceil(threads.max(1));
            let _ = run_par(&chunks, threads, |c| {
                let x = unsafe { std::slice::from_raw_parts_mut(xp.get(), n) };
                let mut line = Vec::new();
                let mut tmp = Vec::new();
                for &off in lines.iter().skip(c * per).take(per) {
                    line.clear();
                    line.extend((0..ext[a]).map(|t| x[off + t * st[a]]));
                    crate::gpuwav::inv97_line(&mut line, &mut tmp, &k);
                    for (t, &v) in line.iter().enumerate() {
                        x[off + t * st[a]] = v;
                    }
                }
                Ok(())
            });
            done[a] = true;
        }
    }
}

fn apply_tile(ps: &Parsed3, bytes: &[u8], t: &TileRef, c: &mut [i32]) -> Result<(), DecodeError> {
    if ps.corr == 0 {
        return Ok(());
    }
    let w = words(bytes, t.words.0, t.words.1);
    let mut sink = DecSink { lanes: vec![RansDec::new(&w)] };
    let mut om = vec![MINIT; NOMODEL];
    let n = gamma::<_, true>(&mut sink, 0, &mut om, 0, 0) as usize;
    let mut last: i64 = -1;
    let d3 = ps.d3;
    let ncell: usize = t.dims.iter().product();
    for _ in 0..n {
        let g = gamma::<_, true>(&mut sink, 0, &mut om, 32, 0) as i64;
        let li = (last + 1 + g) as usize;
        last = li as i64;
        let neg = sink.raw(0, 0, 1) == 1;
        let k = gamma::<_, true>(&mut sink, 0, &mut om, 64, 0).wrapping_add(1) as i32;
        let k = if neg { -k } else { k };
        if li < ncell {
            let (z, y, x) = (li / (t.dims[1] * t.dims[2]), (li / t.dims[2]) % t.dims[1], li % t.dims[2]);
            let gi = ((t.origin[0] + z) * d3[1] + t.origin[1] + y) * d3[2] + t.origin[2] + x;
            c[gi] = c[gi].wrapping_add(k.wrapping_mul(ps.corr));
        }
    }
    if sink.lanes[0].overrun() {
        return Err(DecodeError::Truncated);
    }
    Ok(())
}

/// Result of a (sub-box) decode.
pub struct Decoded3 {
    /// `f32` bits in box layout.
    pub bits: Vec<u32>,
    /// Stream bytes the decode had to read (header + blocks + tiles).
    pub bytes_read: usize,
    /// Code-blocks decoded.
    pub blocks: usize,
}

/// Decode the box `[lo, hi)` (whole field: `[0, d3)`).
pub fn decode3_box(bytes: &[u8], lo: [usize; 3], hi: [usize; 3], threads: usize) -> Result<Decoded3, DecodeError> {
    let ps = parse3(bytes)?;
    decode3_box_parsed(&ps, bytes, lo, hi, threads)
}

/// [`decode3_box`] on a parsed stream.
pub fn decode3_box_parsed(ps: &Parsed3, bytes: &[u8], lo: [usize; 3], hi: [usize; 3], threads: usize) -> Result<Decoded3, DecodeError> {
    let d3 = ps.d3;
    for a in 0..3 {
        if lo[a] >= hi[a] || hi[a] > d3[a] {
            return Err(DecodeError::Invalid("box"));
        }
    }
    let n: usize = d3.iter().product();
    let whole = lo == [0; 3] && hi == d3;
    let (samp, coef) = needs(&ps.sh, lo, hi);
    // blocks whose region meets the needed coefficients
    let nb = ps.blocks.len();
    let mut want = vec![whole; nb];
    if !whole {
        let p = &ps.sh.p;
        for (i, b) in ps.blocks.iter().enumerate() {
            let band = &p.bands[ps.block_band[i]];
            let need = if band.mask == 0 { &samp[p.nlev()] } else { &coef[band.level] };
            want[i] = (0..3).all(|a| hits(&need[a], b.start[a], b.end[a]));
        }
        // parent closure: a block needs the parent cells it conditions on
        for i in (0..nb).rev() {
            if !want[i] {
                continue;
            }
            let b = &ps.blocks[i];
            let Some(pi) = b.parent else { continue };
            let pb = &ps.sh.bands[pi];
            let mut plo = [0usize; 3];
            let mut phi = [0usize; 3];
            for a in 0..3 {
                plo[a] = (pb.start[a] + ((b.start[a] - b.org[a]) >> b.pshift[a])).min(pb.end[a] - 1).saturating_sub(1).max(pb.start[a]);
                phi[a] = (pb.start[a] + ((b.end[a] - 1 - b.org[a]) >> b.pshift[a])).min(pb.end[a] - 1) + 2;
            }
            for j in 0..i {
                if ps.block_band[j] == pi && !want[j] {
                    let c = &ps.blocks[j];
                    if (0..3).all(|a| c.start[a] < phi[a] && plo[a] < c.end[a]) {
                        want[j] = true;
                    }
                }
            }
        }
    }
    let mut q = vec![0i32; n];
    let mut rs = vec![0i32; n];
    let (qp, rp) = (Ptr(q.as_mut_ptr()), Ptr(rs.as_mut_ptr()));
    let mut bytes_read = ps.header_bytes;
    let mut nblk = 0;
    let irb = IrBase::new(ps, bytes);
    for (g0, g1) in ps.level_groups() {
        let items: Vec<usize> = (g0..g1).filter(|&i| want[i]).collect();
        for &i in &items {
            bytes_read += ps.block_words[i].1 - ps.block_words[i].0;
        }
        nblk += items.len();
        run_par(&items, threads, |i| decode_block(ps, bytes, i, qp, rp, n, irb.as_ref()))?;
    }
    drop(rs);
    // dequantize the decoded blocks
    let [_, ny, nx] = d3;
    for (i, b) in ps.blocks.iter().enumerate() {
        if !want[i] {
            continue;
        }
        let band = &ps.sh.bands[ps.block_band[i]];
        for z in b.start[0]..b.end[0] {
            for y in b.start[1]..b.end[1] {
                for x in b.start[2]..b.end[2] {
                    let j = (z * ny + y) * nx + x;
                    q[j] = dequant(q[j], band, &ps.dl);
                }
            }
        }
    }
    inverse_window(&ps.sh, &mut q, if whole { None } else { Some((&samp, &coef)) }, threads);
    let scale = ps.scale();
    let bx = [0, 1, 2].map(|a| hi[a] - lo[a]);
    for t in &ps.tiles {
        if (0..3).all(|a| t.origin[a] < hi[a] && t.origin[a] + t.dims[a] > lo[a]) {
            bytes_read += t.bytes.1 - t.bytes.0;
            apply_tile(ps, bytes, t, &mut q)?;
        }
    }
    let mut out = vec![0u32; bx.iter().product()];
    for z in lo[0]..hi[0] {
        for y in lo[1]..hi[1] {
            let src = (z * ny + y) * nx;
            let dst = ((z - lo[0]) * bx[1] + (y - lo[1])) * bx[2];
            for x in lo[2]..hi[2] {
                out[dst + x - lo[2]] = to_f32_bits(q[src + x].wrapping_add(ps.mean_fix), scale);
            }
        }
    }
    for t in &ps.tiles {
        for &(li, bits) in &t.esc {
            let (z, y, x) = (li / (t.dims[1] * t.dims[2]), (li / t.dims[2]) % t.dims[1], li % t.dims[2]);
            let g = [t.origin[0] + z, t.origin[1] + y, t.origin[2] + x];
            if (0..3).all(|a| g[a] >= lo[a] && g[a] < hi[a]) {
                out[((g[0] - lo[0]) * bx[1] + g[1] - lo[1]) * bx[2] + g[2] - lo[2]] = bits;
            }
        }
    }
    Ok(Decoded3 { bits: out, bytes_read, blocks: nblk })
}

/// Decode a whole v3 stream to `f32`.
pub fn decode3_f32(bytes: &[u8], threads: usize) -> Result<(Vec<usize>, Vec<f32>), DecodeError> {
    let ps = parse3(bytes)?;
    let d = decode3_box_parsed(&ps, bytes, [0; 3], ps.d3, threads)?;
    Ok((ps.dims.clone(), d.bits.into_iter().map(f32::from_bits).collect()))
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
                    v.push(((0.3 * x).sin() + (0.11 * y * z / 8.0).cos() + 0.01 * x * y - ((k * 7 + j * 3) % 11) as f64 * 0.003 + 3.0) as f32);
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
    fn round_trip_and_random_access() {
        let d = [24, 37, 45];
        let x = field(d);
        for bound in [ErrorBound::Psnr(60.0), ErrorBound::Psnr(95.0), ErrorBound::MaxAbs(1e-3), ErrorBound::MaxAbs(2e-5)] {
            for (block, resctx, lanes) in [(8, true, 1), (16, false, 1), (64, true, 1), (16, true, 4)] {
                let cfg = Gpu3Config { block, resctx, lanes, tile: 16, ..Gpu3Config::new(bound) };
                let (bytes, st) = encode3_f32(&x, &d, &cfg, 2);
                let (_, y) = decode3_f32(&bytes, 1).unwrap();
                let (rmse, mx) = err(&x, &y);
                assert!((rmse - st.rmse).abs() <= 1e-12 * rmse.max(1e-30), "{bound:?} {block}");
                assert_eq!(mx, st.max_abs);
                if let ErrorBound::MaxAbs(b) = bound {
                    assert!(mx <= b);
                }
                let (_, y4) = decode3_f32(&bytes, 4).unwrap();
                assert!(y.iter().zip(&y4).all(|(a, b)| a.to_bits() == b.to_bits()));
                for (lo, hi) in [([3, 5, 7], [21, 29, 20]), ([0, 0, 0], [1, 1, 1]), ([23, 36, 44], [24, 37, 45]), ([10, 0, 30], [14, 37, 31])] {
                    let sub = decode3_box(&bytes, lo, hi, 1).unwrap();
                    let mut k = 0;
                    for z in lo[0]..hi[0] {
                        for yy in lo[1]..hi[1] {
                            for xx in lo[2]..hi[2] {
                                assert_eq!(sub.bits[k], y[(z * d[1] + yy) * d[2] + xx].to_bits(), "{bound:?} {block} {lo:?}");
                                k += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn two_d() {
        let x2: Vec<f32> = (0..70 * 90).map(|i| ((i % 90) as f32 * 0.07).sin() * ((i / 90) as f32 * 0.05).cos()).collect();
        let (b, st) = encode3_f32(&x2, &[70, 90], &Gpu3Config { block2d: 16, tile: 4, ..Gpu3Config::new(ErrorBound::Psnr(70.0)) }, 1);
        let (_, y) = decode3_f32(&b, 1).unwrap();
        assert!((err(&x2, &y).0 - st.rmse).abs() <= 1e-12 * st.rmse);
        let sub = decode3_box(&b, [0, 30, 40], [1, 50, 70], 1).unwrap();
        assert!(sub.bytes_read < b.len());
        for (k, (yy, xx)) in (30..50).flat_map(|a| (40..70).map(move |c| (a, c))).enumerate() {
            assert_eq!(sub.bits[k], y[yy * 90 + xx].to_bits());
        }
    }
}
