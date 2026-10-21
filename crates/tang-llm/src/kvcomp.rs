//! Compressed KV cache, decoded on the GPU every decode step (benchmark path).
//!
//! [`Model::compress_kv`] takes a cache's first `upto` positions (whole 128-token codec
//! blocks), compresses every layer's K and V, and keeps the compressed form on the device.
//! From then on each decode step, right before a layer's attention, the device decodes that
//! layer's compressed rows back into the paged pool ([`Model::restore_kv`], called from the
//! decoder loop), so attention reads exactly the codec's reconstruction. Positions after
//! `upto` (the tail and newly generated tokens) stay bf16, like KIVI's fp16 residual.
//!
//! Modes:
//! - `tkv`: `cram_codec::kvcodec` GPU layout. K is coded pre-RoPE (the pool's post-RoPE K is
//!   rotated back with the model's own cos/sin tables; the kernel rotates the decoded rows
//!   forward with the same tables), V as is. Step = `rel × centered rms × √12`, K at
//!   `0.5 rel`, V at `2 rel` (the `tkv-<rel>` configs of `bench/kv_base.py`).
//! - `kivi`: KIVI-style asymmetric integer groups on post-RoPE K (per channel, `g` tokens) and V
//!   (per token, `g` channels), half scale and zero.
//!
//! This prototype overwrites the pool rows in place, so it measures the decode cost and the
//! quality, not the memory saving: a deployment decodes into a per-layer scratch instead (one
//! layer of bf16 rows resident, everything else compressed). [`KvComp::bytes`] is the
//! compressed size that would be resident.
use crate::blocks::{self, BLOCK};
use crate::model::Model;
use cram_codec::kvcodec::{encode, Kind, KvParams};
use cram_codec::kvgpu::{Batch, Placement};
use tang_compute::ComputeDevice;

/// Which compressor.
#[derive(Clone, Copy, Debug)]
pub enum KvMode {
    /// tang kvcodec, GPU layout, at relative step `rel`, `block`-token lanes of `lane_ch`
    /// channels (0 = whole head).
    Tkv {
        rel: f64,
        block: usize,
        lane_ch: usize,
    },
    /// KIVI-style `bits`-bit groups of `g`.
    Kivi { bits: u32, g: usize },
}

impl KvMode {
    /// `tkv:0.07`, `tkv:0.07:16`, `tkv:0.07:16:4` (rel, block, lane channels), `kivi:4:32`.
    pub fn parse(s: &str) -> Option<Self> {
        let p: Vec<&str> = s.split(':').collect();
        match p.as_slice() {
            ["tkv", r] => Some(Self::Tkv {
                rel: r.parse().ok()?,
                block: 128,
                lane_ch: 0,
            }),
            ["tkv", r, b] => Some(Self::Tkv {
                rel: r.parse().ok()?,
                block: b.parse().ok()?,
                lane_ch: 0,
            }),
            ["tkv", r, b, w] => Some(Self::Tkv {
                rel: r.parse().ok()?,
                block: b.parse().ok()?,
                lane_ch: w.parse().ok()?,
            }),
            ["kivi", b, g] => Some(Self::Kivi {
                bits: b.parse().ok()?,
                g: g.parse().ok()?,
            }),
            _ => None,
        }
    }
}

/// Decode-speed experiments on top of a [`KvMode`] (`docs/compression-math-survey/
/// tang-compress-kv-decode-speed.md`).
#[derive(Clone, Copy, Debug, Default)]
pub struct KvOpts {
    /// Keep the newest `window` positions raw (bf16): only blocks older than that are
    /// compressed and decoded each step.
    pub window: usize,
    /// Decode the compressed rows once at compression time and keep the decoded tiles
    /// resident (no per-step decode; resident memory is bf16 plus the compressed copy).
    pub cached: bool,
    /// Run the per-step decode without writing its output (attention then reads the rows
    /// decoded once at compression time): the decode cost a fused decode + attention kernel
    /// would still pay.
    pub decode_only: bool,
}

impl KvOpts {
    /// Parse `+cache`, `+win<N>`, `+nostore` suffixes off a mode string.
    pub fn split(s: &str) -> (&str, Self) {
        let mut it = s.split('+');
        let base = it.next().unwrap_or("");
        let mut o = Self::default();
        for f in it {
            match f {
                "cache" => o.cached = true,
                "nostore" => o.decode_only = true,
                w if w.starts_with("win") => o.window = w[3..].parse().expect("+win<N>"),
                _ => panic!("unknown kv option {f}"),
            }
        }
        (base, o)
    }
}

enum Layer<B> {
    Tkv {
        stream: B,
        lanes: B,
        chans: B,
        scratch: B,
        n_lanes: usize,
        rope_k: bool,
    },
    Kivi {
        kc: B,
        ksz: B,
        vc: B,
        vsz: B,
        p: [u32; 6],
    },
}

/// A cache's compressed rows, per layer (see the module docs).
pub struct KvComp<B> {
    layers: Vec<(Layer<B>, Option<Layer<B>>)>,
    prior: Option<B>,
    /// Positions covered.
    pub upto: usize,
    /// Decoded once per layer (first step), then the decoded tiles are reused.
    pub cached: bool,
    decoded: Vec<std::sync::atomic::AtomicBool>,
    /// Compressed bytes (streams, side tables and per-stream headers).
    pub bytes: usize,
    /// Values covered (`upto × layers × 2 × kv_dim`).
    pub values: usize,
}

fn bf(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn f16_bits(x: f32) -> u16 {
    // round-to-nearest-even f32 -> f16 (finite inputs; KV values are far from the limits)
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let m = b & 0x7f_ffff;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = m | 0x80_0000;
        let sh = (14 - e) as u32;
        let half = 1 << (sh - 1);
        let r = (m + half - 1 + ((m >> sh) & 1)) >> sh;
        return sign | r as u16;
    }
    let r = m + 0xfff + ((m >> 13) & 1);
    let (e, r) = if r & 0x80_0000 != 0 {
        (e + 1, 0)
    } else {
        (e, r >> 13)
    };
    if e >= 31 {
        return sign | 0x7c00;
    }
    sign | ((e as u16) << 10) | r as u16
}

fn f16_val(h: u16) -> f32 {
    let s = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f32;
    s * if e == 0 {
        m * 2f32.powi(-24)
    } else {
        (1.0 + m / 1024.0) * 2f32.powi(e - 15)
    }
}

/// KIVI groups on `x` (`[t][kvd]`, kvd = nkv·hd): (packed codes, half2 (scale, zero) words).
fn kivi_quant(
    x: &[f32],
    n: usize,
    kvd: usize,
    bits: u32,
    g: usize,
    key: bool,
) -> (Vec<u32>, Vec<u32>) {
    let qmax = ((1u32 << bits) - 1) as f32;
    let groups: Vec<Vec<usize>> = if key {
        (0..n / g)
            .flat_map(|tg| {
                (0..kvd).map(move |c| (tg * g..tg * g + g).map(|t| t * kvd + c).collect())
            })
            .collect()
    } else {
        (0..n)
            .flat_map(|t| {
                (0..kvd / g).map(move |cg| (cg * g..cg * g + g).map(|c| t * kvd + c).collect())
            })
            .collect()
    };
    let mut codes = vec![0u32; (n * kvd * bits as usize).div_ceil(32)];
    let mut sz = Vec::with_capacity(groups.len());
    for idx in &groups {
        let lo = idx.iter().map(|&i| x[i]).fold(f32::INFINITY, f32::min);
        let hi = idx.iter().map(|&i| x[i]).fold(f32::NEG_INFINITY, f32::max);
        let (sh, zh) = (f16_bits((hi - lo) / qmax), f16_bits(lo));
        let (s, z) = (f16_val(sh), f16_val(zh));
        for &i in idx {
            let q = if s > 0.0 {
                ((x[i] - z) / s).round().clamp(0.0, qmax) as u32
            } else {
                0
            };
            let b = i * bits as usize;
            codes[b >> 5] |= q << (b & 31);
        }
        sz.push(sh as u32 | (zh as u32) << 16);
    }
    (codes, sz)
}

impl<D: ComputeDevice> Model<D> {
    /// Compress `cache` positions `0..upto` (rounded down to 128) in every layer.
    pub fn compress_kv(
        &self,
        cache: &crate::model::Cache<D::Buffer>,
        mode: KvMode,
    ) -> KvComp<D::Buffer> {
        self.compress_kv_with(cache, mode, KvOpts::default())
    }

    /// [`Self::compress_kv`] with the decode-speed options of [`KvOpts`].
    pub fn compress_kv_with(
        &self,
        cache: &crate::model::Cache<D::Buffer>,
        mode: KvMode,
        opts: KvOpts,
    ) -> KvComp<D::Buffer> {
        let dev = &self.dev;
        let c = &self.cfg;
        let (nkv, hd, kvd, nl) = (c.kv_heads(), c.head_dim(), c.kv_dim(), self.n_layers());
        // codec blocks must not straddle a pool block
        let upto = cache.len.saturating_sub(opts.window) / 128 * 128;
        let rows = self.read_rows(cache, 0, upto);
        let re = self.row_elems();
        let table = cache.table.clone();
        let phys = move |t: usize| ((table[t / BLOCK] as usize) * BLOCK + t % BLOCK) * kvd;
        let half = hd / 2;
        let mut bytes = 0;
        let mut layers = Vec::with_capacity(nl);
        for l in 0..nl {
            // [t][kvd] post-RoPE K and V
            let k: Vec<f32> = (0..upto)
                .flat_map(|t| rows[t * re + 2 * l * kvd..][..kvd].iter().map(|&b| bf(b)))
                .collect();
            let v: Vec<f32> = (0..upto)
                .flat_map(|t| {
                    rows[t * re + (2 * l + 1) * kvd..][..kvd]
                        .iter()
                        .map(|&b| bf(b))
                })
                .collect();
            match mode {
                KvMode::Kivi { bits, g } => {
                    let (kc, ksz) = kivi_quant(&k, upto, kvd, bits, g, true);
                    let (vc, vsz) = kivi_quant(&v, upto, kvd, bits, g, false);
                    bytes += 4 * (kc.len() + vc.len()) + 4 * (ksz.len() + vsz.len());
                    let p = [bits, g as u32, nkv as u32, hd as u32, upto as u32, 1];
                    let kl = Layer::Kivi {
                        kc: dev.upload_u32(&kc),
                        ksz: dev.upload_u32(&ksz),
                        vc: dev.upload_u32(&vc),
                        vsz: dev.upload_u32(&vsz),
                        p,
                    };
                    layers.push((kl, None));
                }
                KvMode::Tkv {
                    rel,
                    block,
                    lane_ch,
                } => {
                    let (cos, sin) = self.rope_tables(l);
                    let mut one = |x: &[f32], key: bool| -> Layer<D::Buffer> {
                        // [t][h][c] -> [h][t][c], K rotated back to pre-RoPE
                        let mut y = vec![0f32; x.len()];
                        for t in 0..upto {
                            for h in 0..nkv {
                                let src = &x[t * kvd + h * hd..][..hd];
                                let dst = &mut y[(h * upto + t) * hd..][..hd];
                                if key {
                                    for i in 0..half {
                                        let (cs, sn) = (cos[t * half + i], sin[t * half + i]);
                                        let (a, b) = (src[i], src[i + half]);
                                        dst[i] = a * cs + b * sn;
                                        dst[i + half] = b * cs - a * sn;
                                    }
                                } else {
                                    dst.copy_from_slice(src);
                                }
                            }
                        }
                        let mut sse = 0.0;
                        for h in 0..nkv {
                            for ch in 0..hd {
                                let col = |t: usize| y[(h * upto + t) * hd + ch] as f64;
                                let m = (0..upto).map(col).sum::<f64>() / upto as f64;
                                sse += (0..upto).map(|t| (col(t) - m).powi(2)).sum::<f64>();
                            }
                        }
                        let r = if key { 0.5 * rel } else { 2.0 * rel };
                        let step = r * (sse / y.len() as f64).sqrt() * 12f64.sqrt();
                        let mut p =
                            KvParams::new(if key { Kind::K } else { Kind::V }, nkv, hd, step);
                        p.pre_rope = false;
                        p.gpu = true;
                        p.block = block;
                        p.lane_ch = lane_ch;
                        let s = encode(p, &y, upto);
                        let ph = &phys;
                        let row = move |h: usize, t: usize| ph(t) + h * hd;
                        let mut b = Batch::new();
                        b.add(
                            &s,
                            &Placement {
                                row: &row,
                                rope_theta: key.then_some(0.0),
                                decode_only: opts.decode_only,
                            },
                            0,
                            upto,
                        )
                        .unwrap();
                        bytes += s.len();
                        Layer::Tkv {
                            stream: dev.upload_u32(&b.stream),
                            lanes: dev.upload_u32(&b.lanes),
                            chans: dev.upload_u32(&b.chans),
                            scratch: dev.alloc(b.scratch_words.max(1)),
                            n_lanes: b.n_lanes(),
                            rope_k: key,
                        }
                    };
                    let kl = one(&k, true);
                    let vl = one(&v, false);
                    layers.push((kl, Some(vl)));
                }
            }
        }
        let prior = matches!(mode, KvMode::Tkv { .. }).then(|| dev.upload_u32(&Batch::prior()));
        KvComp {
            layers,
            prior,
            upto,
            cached: opts.cached,
            decoded: (0..nl).map(|_| Default::default()).collect(),
            bytes,
            values: upto * nl * 2 * kvd,
        }
    }

    /// Decode layer `l` of `kc` into the pool (call before that layer's attention).
    pub(crate) fn restore_kv(
        &self,
        kc: &KvComp<D::Buffer>,
        l: usize,
        kp: &D::Buffer,
        vp: &D::Buffer,
        table: &D::Buffer,
    ) {
        if kc.cached && kc.decoded[l].swap(true, std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        let dev = &self.dev;
        let (cos, sin) = self.rope_buffers(l);
        let run = |lay: &Layer<D::Buffer>, out: &D::Buffer| match lay {
            Layer::Tkv {
                stream,
                lanes,
                chans,
                scratch,
                n_lanes,
                rope_k,
            } => dev.tkv_decode(
                stream,
                lanes,
                chans,
                kc.prior.as_ref().unwrap(),
                scratch,
                out,
                rope_k.then_some((cos, sin)),
                *n_lanes,
            ),
            Layer::Kivi {
                kc: c,
                ksz,
                vc,
                vsz,
                p,
            } => {
                let sh = BLOCK.trailing_zeros();
                dev.kivi_dequant(c, ksz, kp, Some((table, sh)), *p);
                let mut pv = *p;
                pv[5] = 0;
                dev.kivi_dequant(vc, vsz, vp, Some((table, sh)), pv);
            }
        };
        let (k, v) = &kc.layers[l];
        run(k, kp);
        if let Some(v) = v {
            run(v, vp);
        }
        let _ = blocks::BLOCK;
    }
}
