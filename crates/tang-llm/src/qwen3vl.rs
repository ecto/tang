//! Qwen3-VL's vision encoder: a SigLIP-style ViT over dynamic-resolution 16×16 patches (learned
//! positions resampled to the image's grid, plus 2D axial RoPE), a 2×2 spatial-merge projector
//! into the language model's width, and DeepStack: features from intermediate blocks, merged the
//! same way, that the language model adds to the image tokens' hidden states after its first
//! layers.
//!
//! The encoder knows nothing about the language model beyond its width, so other decoders that
//! use this tower (the Qwen3.8-Flash-Next mmproj, `qwen3vl_merger`) can share it. Image
//! placement (M-RoPE, DeepStack injection) lives with the decoder ([`crate::model`]).

use crate::weights::Weights;
use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use tang_compute::ComputeDevice;

/// `vision_config` of a Qwen3-VL checkpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct QwenVisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    #[serde(alias = "num_attention_heads")]
    pub num_heads: usize,
    #[serde(default = "d16")]
    pub patch_size: usize,
    #[serde(default = "d2")]
    pub spatial_merge_size: usize,
    #[serde(default = "d2")]
    pub temporal_patch_size: usize,
    pub out_hidden_size: usize,
    #[serde(default = "d2304")]
    pub num_position_embeddings: usize,
    #[serde(default)]
    pub deepstack_visual_indexes: Vec<usize>,
    /// Axial RoPE base (`rope_parameters.rope_theta`; 10000 in every released checkpoint).
    #[serde(default = "d10k")]
    pub rope_theta: f32,
}

fn d16() -> usize {
    16
}
fn d2() -> usize {
    2
}
fn d2304() -> usize {
    2304
}
fn d10k() -> f32 {
    10_000.
}

impl QwenVisionConfig {
    pub fn from_value(v: &serde_json::Value) -> Result<Self> {
        let mut c: Self = serde_json::from_value(v.clone())?;
        if let Some(t) = v["rope_parameters"]["rope_theta"].as_f64() {
            c.rope_theta = t as f32;
        }
        Ok(c)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }
}

/// How images are sized before the tower (`preprocessor_config.json`).
#[derive(Debug, Clone, Copy)]
pub struct Sizing {
    pub patch: usize,
    pub merge: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
}

impl Sizing {
    /// Qwen3-VL's defaults (`shortest_edge` 65536, `longest_edge` 16777216).
    pub fn new(cfg: &QwenVisionConfig) -> Self {
        Self {
            patch: cfg.patch_size,
            merge: cfg.spatial_merge_size,
            min_pixels: 65_536,
            max_pixels: 16_777_216,
        }
    }

    /// Read `size` from a `preprocessor_config.json`, if there is one.
    pub fn with_preprocessor(mut self, bytes: Option<&[u8]>) -> Self {
        let Some(v) = bytes.and_then(|b| serde_json::from_slice::<serde_json::Value>(b).ok())
        else {
            return self;
        };
        let size = &v["size"];
        if let Some(n) = size["shortest_edge"].as_u64().or(v["min_pixels"].as_u64()) {
            self.min_pixels = n as usize;
        }
        if let Some(n) = size["longest_edge"].as_u64().or(v["max_pixels"].as_u64()) {
            self.max_pixels = n as usize;
        }
        self
    }

    /// The resized `(height, width)` in pixels: multiples of `patch * merge`, aspect kept,
    /// area within `[min_pixels, max_pixels]` (transformers' `smart_resize`).
    pub fn resized(&self, height: usize, width: usize) -> Result<(usize, usize)> {
        ensure!(height > 0 && width > 0, "empty image");
        let (h, w) = (height as f64, width as f64);
        ensure!(
            h.max(w) / h.min(w) <= 200.,
            "image aspect ratio must be under 200"
        );
        let f = (self.patch * self.merge) as f64;
        // Python's round() is round-half-to-even.
        let round = |x: f64| x.round_ties_even();
        let (mut hb, mut wb) = (round(h / f) * f, round(w / f) * f);
        if hb * wb > self.max_pixels as f64 {
            let beta = (h * w / self.max_pixels as f64).sqrt();
            hb = f.max((h / beta / f).floor() * f);
            wb = f.max((w / beta / f).floor() * f);
        } else if hb * wb < self.min_pixels as f64 {
            let beta = (self.min_pixels as f64 / (h * w)).sqrt();
            hb = (h * beta / f).ceil() * f;
            wb = (w * beta / f).ceil() * f;
        }
        Ok((hb as usize, wb as usize))
    }

    /// Image tokens (after merging) for an image of this size.
    pub fn tokens(&self, height: usize, width: usize) -> Result<usize> {
        let (h, w) = self.resized(height, width)?;
        let m = self.patch * self.merge;
        Ok((h / m) * (w / m))
    }
}

/// A preprocessed image: patches `[n, 3 * patch * patch]` in spatial-merge-block order, each
/// flattened (channel, row, column), and the patch grid.
pub struct Patches {
    pub data: Vec<f32>,
    pub grid_h: usize,
    pub grid_w: usize,
}

impl Patches {
    pub fn count(&self) -> usize {
        self.grid_h * self.grid_w
    }
}

/// Decode an image (PNG, JPEG, WebP, GIF), resize it as the Qwen3-VL processor does (bicubic,
/// antialiased), normalize to [-1, 1] and cut it into patches.
pub fn preprocess(bytes: &[u8], sizing: &Sizing) -> Result<Patches> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| anyhow::anyhow!("couldn't read the image: {e}"))?
        .to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let (rh, rw) = sizing.resized(h, w)?;
    let px = resize_bicubic(img.as_raw(), h, w, rh, rw);
    Ok(patchify(&px, rh, rw, sizing.patch, sizing.merge))
}

/// Size an image from its header alone (to count its tokens before decoding it).
pub fn image_size(bytes: &[u8]) -> Result<(usize, usize)> {
    let r = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .context("couldn't read the image")?;
    let (w, h) = r
        .into_dimensions()
        .map_err(|e| anyhow::anyhow!("couldn't read the image: {e}"))?;
    Ok((h as usize, w as usize))
}

/// `[h, w, 3]` floats in [0, 255] → patches of `(x / 255 - 0.5) / 0.5`.
pub fn patchify(px: &[f32], h: usize, w: usize, patch: usize, merge: usize) -> Patches {
    let (gh, gw) = (h / patch, w / patch);
    let mut data = Vec::with_capacity(gh * gw * 3 * patch * patch);
    for by in 0..gh / merge {
        for bx in 0..gw / merge {
            for my in 0..merge {
                for mx in 0..merge {
                    let (py, pxx) = (by * merge + my, bx * merge + mx);
                    for c in 0..3 {
                        for ky in 0..patch {
                            let row = (py * patch + ky) * w + pxx * patch;
                            for kx in 0..patch {
                                let v = px[(row + kx) * 3 + c];
                                data.push((v / 255. - 0.5) / 0.5);
                            }
                        }
                    }
                }
            }
        }
    }
    Patches {
        data,
        grid_h: gh,
        grid_w: gw,
    }
}

/// Antialiased bicubic resize of `[h, w, 3]` u8 to `[rh, rw, 3]` floats in [0, 255], as
/// torchvision does it on float images (PIL's separable filter, `a = -0.5`, support scaled by
/// the downscale factor, horizontal pass first, no intermediate rounding), clamped to [0, 255]
/// and rounded the way a uint8 round trip would.
pub fn resize_bicubic(src: &[u8], h: usize, w: usize, rh: usize, rw: usize) -> Vec<f32> {
    let src: Vec<f32> = src.iter().map(|&b| b as f32).collect();
    let horiz = filter_axis(&src, h, w, rw, true);
    let out = filter_axis(&horiz, h, rw, rh, false);
    out.into_iter()
        .map(|v| v.clamp(0., 255.).round_ties_even())
        .collect()
}

/// Taps (first source index and weights) per output position, for a `len` → `out` resample.
fn taps(len: usize, out: usize) -> Vec<(usize, Vec<f32>)> {
    let cubic = |x: f64| {
        let a = -0.5;
        let x = x.abs();
        if x < 1. {
            ((a + 2.) * x - (a + 3.)) * x * x + 1.
        } else if x < 2. {
            (((x - 5.) * x + 8.) * x - 4.) * a
        } else {
            0.
        }
    };
    let scale = len as f64 / out as f64;
    let fs = scale.max(1.);
    let support = 2. * fs;
    (0..out)
        .map(|i| {
            let center = (i as f64 + 0.5) * scale;
            let lo = ((center - support + 0.5).floor().max(0.)) as usize;
            let hi = ((center + support + 0.5).floor() as usize).min(len);
            let mut ws: Vec<f64> = (lo..hi)
                .map(|j| cubic((j as f64 - center + 0.5) / fs))
                .collect();
            let sum: f64 = ws.iter().sum();
            if sum != 0. {
                ws.iter_mut().for_each(|x| *x /= sum);
            }
            (lo, ws.into_iter().map(|x| x as f32).collect())
        })
        .collect()
}

fn filter_axis(src: &[f32], h: usize, w: usize, out: usize, horizontal: bool) -> Vec<f32> {
    let (len, rows) = if horizontal { (w, h) } else { (h, w) };
    if len == out {
        return src.to_vec();
    }
    let t = taps(len, out);
    let (oh, ow) = if horizontal { (h, out) } else { (out, w) };
    let mut dst = vec![0f32; oh * ow * 3];
    for r in 0..rows {
        for (o, (lo, ws)) in t.iter().enumerate() {
            let mut acc = [0f32; 3];
            for (k, &wt) in ws.iter().enumerate() {
                let j = lo + k;
                let s = if horizontal { r * w + j } else { j * w + r } * 3;
                for c in 0..3 {
                    acc[c] += src[s + c] * wt;
                }
            }
            let d = if horizontal { r * ow + o } else { o * ow + r } * 3;
            dst[d..d + 3].copy_from_slice(&acc);
        }
    }
    dst
}

struct Block<B> {
    norm1: (B, B),
    q: (B, B),
    k: (B, B),
    v: (B, B),
    proj: (B, B),
    norm2: (B, B),
    fc1: (B, B),
    fc2: (B, B),
}

/// Patch merger: LayerNorm (over one patch, or over the merged 2×2 block for DeepStack), then
/// `fc1` (4h → 4h), exact GELU, `fc2` (4h → out).
struct Merger<B> {
    norm: (B, B),
    postshuffle: bool,
    fc1: (B, B),
    fc2: (B, B),
}

/// An image through the tower.
pub struct Encoded<B> {
    /// `[tokens, out_hidden]`, merged-block order (row-major over the merged grid).
    pub embeds: B,
    /// One `[tokens, out_hidden]` per DeepStack layer, in order.
    pub deepstack: Vec<B>,
    /// Merged grid (rows, columns); `tokens = rows * columns`.
    pub grid: (usize, usize),
}

impl<B> Encoded<B> {
    pub fn tokens(&self) -> usize {
        self.grid.0 * self.grid.1
    }
}

pub struct QwenVision<B> {
    pub cfg: QwenVisionConfig,
    pub sizing: Sizing,
    /// Patch embedding `[hidden, 3 * p * p]` (the temporal kernel summed: a still image is
    /// its frame twice), and bias.
    patch: (B, B),
    /// Learned position table `[side * side, hidden]`, on the host (resampled per image).
    pos: Vec<f32>,
    blocks: Vec<Block<B>>,
    merger: Merger<B>,
    deepstack: Vec<Merger<B>>,
}

impl<B> QwenVision<B> {
    /// Load from `prefix` (`model.visual` in HF checkpoints, `vision_tower` in MLX ones).
    /// Matrices go up as bf16 when `bf16`, else f32; vectors stay f32.
    pub fn load<D: ComputeDevice<Buffer = B>>(
        dev: &D,
        w: &Weights,
        prefix: &str,
        cfg: QwenVisionConfig,
        bf16: bool,
    ) -> Result<Self> {
        let h = cfg.hidden_size;
        let f32v = |n: &str| -> Result<B> { Ok(dev.upload(&w.f32(n)?.0)) };
        let mat = |data: &[f32]| -> B {
            if bf16 {
                let bits: Vec<u16> = data.iter().map(|&x| tang_compute::f32_to_bf16(x)).collect();
                dev.upload_bf16(&bits)
            } else {
                dev.upload(data)
            }
        };
        let lin = |n: &str| -> Result<(B, B)> {
            Ok((
                mat(&w.f32(&format!("{n}.weight"))?.0),
                f32v(&format!("{n}.bias"))?,
            ))
        };
        let norm = |n: &str| -> Result<(B, B)> {
            Ok((f32v(&format!("{n}.weight"))?, f32v(&format!("{n}.bias"))?))
        };
        let blocks = (0..cfg.depth)
            .map(|l| {
                let b = format!("{prefix}.blocks.{l}");
                // Fused qkv `[3h, h]`: rows are q, then k, then v.
                let (qkv, _) = w.f32(&format!("{b}.attn.qkv.weight"))?;
                let (qb, _) = w.f32(&format!("{b}.attn.qkv.bias"))?;
                let part = |i: usize| {
                    (
                        mat(&qkv[i * h * h..(i + 1) * h * h]),
                        dev.upload(&qb[i * h..(i + 1) * h]),
                    )
                };
                Ok(Block {
                    norm1: norm(&format!("{b}.norm1"))?,
                    q: part(0),
                    k: part(1),
                    v: part(2),
                    proj: lin(&format!("{b}.attn.proj"))?,
                    norm2: norm(&format!("{b}.norm2"))?,
                    fc1: lin(&format!("{b}.mlp.linear_fc1"))?,
                    fc2: lin(&format!("{b}.mlp.linear_fc2"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let merger = |n: &str, postshuffle: bool| -> Result<Merger<B>> {
            Ok(Merger {
                norm: norm(&format!("{n}.norm"))?,
                postshuffle,
                fc1: lin(&format!("{n}.linear_fc1"))?,
                fc2: lin(&format!("{n}.linear_fc2"))?,
            })
        };
        let deepstack = (0..cfg.deepstack_visual_indexes.len())
            .map(|i| merger(&format!("{prefix}.deepstack_merger_list.{i}"), true))
            .collect::<Result<Vec<_>>>()?;
        let (pw, shape) = w.f32(&format!("{prefix}.patch_embed.proj.weight"))?;
        let patch = (
            mat(&fold_patch_weight(&pw, &shape, &cfg)?),
            f32v(&format!("{prefix}.patch_embed.proj.bias"))?,
        );
        let (pos, ps) = w.f32(&format!("{prefix}.pos_embed.weight"))?;
        ensure!(
            ps == [cfg.num_position_embeddings, h],
            "pos_embed is {ps:?}, expected [{}, {h}]",
            cfg.num_position_embeddings
        );
        Ok(Self {
            sizing: Sizing::new(&cfg),
            patch,
            pos,
            blocks,
            merger: merger(&format!("{prefix}.merger"), false)?,
            deepstack,
            cfg,
        })
    }

    /// Encode one image's patches.
    pub fn encode<D: ComputeDevice<Buffer = B>>(&self, dev: &D, p: &Patches) -> Encoded<B> {
        self.encode_traced(dev, p, None)
    }

    /// [`encode`](Self::encode), recording the hidden state after the patch embedding and
    /// after every block (for parity checks).
    pub fn encode_traced<D: ComputeDevice<Buffer = B>>(
        &self,
        dev: &D,
        p: &Patches,
        mut trace: Option<&mut Vec<Vec<f32>>>,
    ) -> Encoded<B> {
        let c = &self.cfg;
        let (h, nh, hd, ps, m) = (
            c.hidden_size,
            c.num_heads,
            c.head_dim(),
            c.patch_size,
            c.spatial_merge_size,
        );
        let n = p.count();
        assert_eq!(p.data.len(), n * 3 * ps * ps, "patch data size");
        assert!(
            p.grid_h.is_multiple_of(m) && p.grid_w.is_multiple_of(m),
            "grid must be a multiple of the merge size"
        );
        let eps = 1e-6;

        let x = dev.upload(&p.data);
        let mut x = dev.linear(&x, &self.patch.0, n, 3 * ps * ps, h);
        x = dev.bias_add(&x, &self.patch.1, n * h, h);
        let pos = dev.upload(&self.positions(p.grid_h, p.grid_w));
        x = dev.add_tensors_buf(&x, &pos, n * h);
        let (cos, sin) = self.rope_tables(p.grid_h, p.grid_w);
        let (cos, sin) = (dev.upload(&cos), dev.upload(&sin));
        if let Some(t) = trace.as_deref_mut() {
            t.push(dev.download(&x));
        }

        let lin = |x: &B, (w, b): &(B, B), rows: usize, k: usize, out: usize| {
            let y = dev.linear(x, w, rows, k, out);
            dev.bias_add(&y, b, rows * out, out)
        };
        let mut deepstack = Vec::new();
        for (l, b) in self.blocks.iter().enumerate() {
            let a = dev.layer_norm(&x, &b.norm1.0, &b.norm1.1, n, h, eps);
            let q = lin(&a, &b.q, n, h, h);
            let k = lin(&a, &b.k, n, h, h);
            let v = lin(&a, &b.v, n, h, h);
            let q = dev.rope_half_cached(&q, &cos, &sin, n, nh, hd, 0);
            let k = dev.rope_half_cached(&k, &cos, &sin, n, nh, hd, 0);
            let att = dev.attention_full(&q, &k, &v, n, nh, hd);
            x = dev.add_tensors_buf(&x, &lin(&att, &b.proj, n, h, h), n * h);
            let a = dev.layer_norm(&x, &b.norm2.0, &b.norm2.1, n, h, eps);
            let f = lin(&a, &b.fc1, n, h, c.intermediate_size);
            let f = dev.gelu_tanh(&f, n * c.intermediate_size);
            x = dev.add_tensors_buf(&x, &lin(&f, &b.fc2, n, c.intermediate_size, h), n * h);
            if let Some(i) = c.deepstack_visual_indexes.iter().position(|&d| d == l) {
                deepstack.push(self.merge(dev, &self.deepstack[i], &x, n));
            }
            if let Some(t) = trace.as_deref_mut() {
                t.push(dev.download(&x));
            }
            dev.flush();
        }
        Encoded {
            embeds: self.merge(dev, &self.merger, &x, n),
            deepstack,
            grid: (p.grid_h / m, p.grid_w / m),
        }
    }

    fn merge<D: ComputeDevice<Buffer = B>>(&self, dev: &D, mg: &Merger<B>, x: &B, n: usize) -> B {
        let c = &self.cfg;
        let unit = c.spatial_merge_size * c.spatial_merge_size;
        let (h, rows) = (c.hidden_size, n / unit);
        let wide = h * unit;
        // Patches are already in merge-block order: `unit` consecutive rows make one token.
        let x = if mg.postshuffle {
            dev.layer_norm(x, &mg.norm.0, &mg.norm.1, rows, wide, 1e-6)
        } else {
            dev.layer_norm(x, &mg.norm.0, &mg.norm.1, n, h, 1e-6)
        };
        let y = dev.linear(&x, &mg.fc1.0, rows, wide, wide);
        let y = dev.bias_add(&y, &mg.fc1.1, rows * wide, wide);
        let y = dev.gelu_erf(&y, rows * wide);
        let y = dev.linear(&y, &mg.fc2.0, rows, wide, c.out_hidden_size);
        dev.bias_add(&y, &mg.fc2.1, rows * c.out_hidden_size, c.out_hidden_size)
    }

    /// The learned positions bilinearly resampled (align corners) from the square table to a
    /// `gh × gw` grid, `[gh * gw, hidden]` in merge-block order.
    fn positions(&self, gh: usize, gw: usize) -> Vec<f32> {
        let h = self.cfg.hidden_size;
        let side = (self.cfg.num_position_embeddings as f64).sqrt() as usize;
        let axis = |i: usize, size: usize| -> [(usize, f32); 2] {
            let src = i as f32 * (side - 1) as f32 / (size.max(2) - 1) as f32;
            let f = src.floor();
            let d = src - f;
            let lo = (f as usize).min(side - 1);
            [(lo, 1. - d), ((lo + 1).min(side - 1), d)]
        };
        let mut out = vec![0f32; gh * gw * h];
        for (t, (r, col)) in merge_order(gh, gw, self.cfg.spatial_merge_size).enumerate() {
            let dst = &mut out[t * h..(t + 1) * h];
            for (ri, rw) in axis(r, gh) {
                for (ci, cw) in axis(col, gw) {
                    let wt = rw * cw;
                    if wt == 0. {
                        continue;
                    }
                    let src = &self.pos[(ri * side + ci) * h..][..h];
                    for (d, s) in dst.iter_mut().zip(src) {
                        *d += s * wt;
                    }
                }
            }
        }
        out
    }

    /// Axial RoPE `[n, hd / 2]` (cos, sin): the first `hd / 4` frequencies turn with the
    /// patch's row, the rest with its column.
    fn rope_tables(&self, gh: usize, gw: usize) -> (Vec<f32>, Vec<f32>) {
        let hd = self.cfg.head_dim();
        let quarter = hd / 4;
        let inv: Vec<f32> = (0..quarter)
            .map(|i| 1. / self.cfg.rope_theta.powf((2 * i) as f32 / (hd / 2) as f32))
            .collect();
        let mut cos = Vec::with_capacity(gh * gw * hd / 2);
        let mut sin = Vec::with_capacity(gh * gw * hd / 2);
        for (r, c) in merge_order(gh, gw, self.cfg.spatial_merge_size) {
            for p in [r, c] {
                for &f in &inv {
                    let a = p as f32 * f;
                    cos.push(a.cos());
                    sin.push(a.sin());
                }
            }
        }
        (cos, sin)
    }
}

/// (row, column) of each patch, in spatial-merge-block order.
fn merge_order(gh: usize, gw: usize, m: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..gh / m).flat_map(move |by| {
        (0..gw / m).flat_map(move |bx| {
            (0..m).flat_map(move |my| (0..m).map(move |mx| (by * m + my, bx * m + mx)))
        })
    })
}

/// The patch conv `[hidden, 3, t, p, p]` (HF) or `[hidden, t, p, p, 3]` (MLX) as a matrix
/// `[hidden, 3 * p * p]` over (channel, row, column), the temporal taps summed.
fn fold_patch_weight(w: &[f32], shape: &[usize], c: &QwenVisionConfig) -> Result<Vec<f32>> {
    let (h, t, p) = (c.hidden_size, c.temporal_patch_size, c.patch_size);
    ensure!(
        w.len() == h * 3 * t * p * p,
        "patch_embed weight is {shape:?}"
    );
    let mlx = shape.len() == 5 && shape[4] == 3 && shape[1] != 3;
    let mut out = vec![0f32; h * 3 * p * p];
    for o in 0..h {
        for ch in 0..3 {
            for ti in 0..t {
                for y in 0..p {
                    for x in 0..p {
                        let src = if mlx {
                            (((o * t + ti) * p + y) * p + x) * 3 + ch
                        } else {
                            (((o * 3 + ch) * t + ti) * p + y) * p + x
                        };
                        out[((o * 3 + ch) * p + y) * p + x] += w[src];
                    }
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sizing() -> Sizing {
        Sizing {
            patch: 16,
            merge: 2,
            min_pixels: 65_536,
            max_pixels: 16_777_216,
        }
    }

    #[test]
    fn smart_resize_matches_transformers() {
        // Values from transformers' smart_resize(h, w, factor=32, 65536, 16777216).
        let s = sizing();
        assert_eq!(s.resized(1080, 1920).unwrap(), (1088, 1920));
        assert_eq!(s.resized(100, 100).unwrap(), (256, 256));
        assert_eq!(s.resized(48, 400).unwrap(), (96, 768));
        assert_eq!(s.tokens(1080, 1920).unwrap(), 34 * 60);
        let small = Sizing {
            max_pixels: 512 * 512,
            ..s
        };
        assert_eq!(small.resized(1080, 1920).unwrap(), (384, 672));
    }

    #[test]
    fn merge_order_is_block_major() {
        let v: Vec<_> = merge_order(2, 4, 2).collect();
        assert_eq!(
            v,
            [
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
                (0, 2),
                (0, 3),
                (1, 2),
                (1, 3)
            ]
        );
    }

    #[test]
    fn resize_identity_and_constant() {
        let src: Vec<u8> = (0..4 * 6 * 3).map(|i| i as u8).collect();
        let same = resize_bicubic(&src, 4, 6, 4, 6);
        assert!(same.iter().zip(&src).all(|(a, &b)| *a == b as f32));
        let flat = vec![77u8; 30 * 40 * 3];
        let out = resize_bicubic(&flat, 30, 40, 64, 96);
        assert!(out.iter().all(|&v| v == 77.));
    }
}
