//! Z-Image-Turbo single-image DiT, following diffusers' basic (non-Omni) path.
//! Matmuls/attention use the selected ComputeDevice; host packing and modulation
//! are deliberately unfused until tensor parity is established.
use crate::{model::Dtype, weights::Weights};
use anyhow::{ensure, Result};
use serde::Deserialize;
use std::path::Path;
use tang_compute::{ComputeBuffer, ComputeDevice};

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub n_layers: usize,
    pub n_refiner_layers: usize,
    pub in_channels: usize,
    pub cap_feat_dim: usize,
    pub all_patch_size: Vec<usize>,
    pub all_f_patch_size: Vec<usize>,
    pub norm_eps: f32,
    pub qk_norm: bool,
    pub rope_theta: f64,
    pub t_scale: f32,
    pub axes_dims: Vec<usize>,
    pub axes_lens: Vec<usize>,
}
impl Config {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.dim > 0 && self.n_heads > 0 && self.dim % self.n_heads == 0,
            "invalid attention dimensions"
        );
        ensure!(
            self.n_heads == self.n_kv_heads,
            "Z-Image requires equal Q/KV heads"
        );
        ensure!(
            self.all_patch_size == [2] && self.all_f_patch_size == [1],
            "only 2x2 single-frame patches supported"
        );
        ensure!(
            self.axes_dims.len() == 3
                && self.axes_lens.len() == 3
                && self.axes_dims.iter().all(|d| *d > 0 && d % 2 == 0)
                && self.axes_dims.iter().sum::<usize>() == self.dim / self.n_heads,
            "invalid RoPE axes"
        );
        ensure!(
            self.norm_eps > 0.
                && self.norm_eps.is_finite()
                && self.rope_theta > 0.
                && self.rope_theta.is_finite(),
            "invalid normalization/RoPE parameters"
        );
        ensure!(
            self.n_layers > 0 && self.cap_feat_dim > 0 && self.in_channels > 0,
            "invalid model dimensions"
        );
        Ok(())
    }
}
struct Linear<B> {
    weight: B,
    bias: Option<B>,
    input: usize,
    output: usize,
}
impl<B> Linear<B> {
    fn run<D: ComputeDevice<Buffer = B>>(&self, dev: &D, x: &B, rows: usize) -> B {
        let out = dev.linear(x, &self.weight, rows, self.input, self.output);
        match &self.bias {
            Some(b) => dev.bias_add(&out, b, rows * self.output, self.output),
            None => out,
        }
    }
}
struct Block<B> {
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    out: Linear<B>,
    q_norm: Option<B>,
    k_norm: Option<B>,
    an1: B,
    an2: B,
    fn1: B,
    fn2: B,
    w1: Linear<B>,
    w2: Linear<B>,
    w3: Linear<B>,
    modulation: Option<Linear<B>>,
}
pub struct DiT<B> {
    pub cfg: Config,
    x: Linear<B>,
    cap_norm: B,
    cap: Linear<B>,
    time1: Linear<B>,
    time2: Linear<B>,
    noise: Vec<Block<B>>,
    context: Vec<Block<B>>,
    layers: Vec<Block<B>>,
    x_pad: Vec<f32>,
    cap_pad: Vec<f32>,
    final_mod: Linear<B>,
    final_out: Linear<B>,
    norm_ones: B,
    norm_zeros: B,
}
fn silu<D: ComputeDevice>(dev: &D, x: &D::Buffer) -> D::Buffer {
    dev.swiglu_fused_buf(x, &dev.upload(&vec![1.; x.len()]), x.len())
}
fn multiply_rows<D: ComputeDevice>(dev: &D, x: &D::Buffer, v: &[f32], rows: usize) -> D::Buffer {
    let repeated = dev.upload(&v.repeat(rows));
    dev.elementwise(&[x, &repeated], x.len(), &|a| a[0] * a[1])
}
impl<B: ComputeBuffer> DiT<B> {
    pub fn load<D: ComputeDevice<Buffer = B>>(dev: &D, dir: &Path, dtype: Dtype) -> Result<Self> {
        let cfg: Config = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        cfg.validate()?;
        ensure!(dtype != Dtype::Q4, "diffusion Q4 weights are not supported");
        let w = Weights::open(dir)?;
        let vector = |name: &str, len: usize| -> Result<B> {
            let (v, shape) = w.f32(name)?;
            ensure!(shape == [len], "wrong tensor shape for {name}: {shape:?}");
            Ok(dev.upload(&v))
        };
        let linear = |name: &str, input: usize, output: usize, bias: bool| -> Result<Linear<B>> {
            let key = format!("{name}.weight");
            let (v, shape) = w.f32(&key)?;
            ensure!(
                shape == [output, input],
                "wrong tensor shape for {key}: {shape:?}"
            );
            let weight = match dtype {
                Dtype::F32 => dev.upload(&v),
                Dtype::Bf16 => dev.upload_bf16(&w.bf16(&key)?),
                Dtype::Q4 => unreachable!(),
            };
            Ok(Linear {
                weight,
                bias: if bias {
                    Some(vector(&format!("{name}.bias"), output)?)
                } else {
                    None
                },
                input,
                output,
            })
        };
        let dim = cfg.dim;
        let hd = dim / cfg.n_heads;
        let ff = dim * 8 / 3;
        let ad = dim.min(256);
        let block = |p: &str, modulated: bool| -> Result<Block<B>> {
            Ok(Block {
                q: linear(&format!("{p}.attention.to_q"), dim, dim, false)?,
                k: linear(&format!("{p}.attention.to_k"), dim, dim, false)?,
                v: linear(&format!("{p}.attention.to_v"), dim, dim, false)?,
                out: linear(&format!("{p}.attention.to_out.0"), dim, dim, false)?,
                q_norm: if cfg.qk_norm {
                    Some(vector(&format!("{p}.attention.norm_q.weight"), hd)?)
                } else {
                    None
                },
                k_norm: if cfg.qk_norm {
                    Some(vector(&format!("{p}.attention.norm_k.weight"), hd)?)
                } else {
                    None
                },
                an1: vector(&format!("{p}.attention_norm1.weight"), dim)?,
                an2: vector(&format!("{p}.attention_norm2.weight"), dim)?,
                fn1: vector(&format!("{p}.ffn_norm1.weight"), dim)?,
                fn2: vector(&format!("{p}.ffn_norm2.weight"), dim)?,
                w1: linear(&format!("{p}.feed_forward.w1"), dim, ff, false)?,
                w2: linear(&format!("{p}.feed_forward.w2"), ff, dim, false)?,
                w3: linear(&format!("{p}.feed_forward.w3"), dim, ff, false)?,
                modulation: if modulated {
                    Some(linear(
                        &format!("{p}.adaLN_modulation.0"),
                        ad,
                        4 * dim,
                        true,
                    )?)
                } else {
                    None
                },
            })
        };
        let blocks = |p: &str, n: usize, m: bool| -> Result<Vec<Block<B>>> {
            (0..n).map(|i| block(&format!("{p}.{i}"), m)).collect()
        };
        let x_pad = w.f32("x_pad_token")?;
        ensure!(x_pad.1 == [1, dim], "invalid image pad token");
        let cap_pad = w.f32("cap_pad_token")?;
        ensure!(cap_pad.1 == [1, dim], "invalid caption pad token");
        Ok(Self {
            x: linear("all_x_embedder.2-1", cfg.in_channels * 4, dim, true)?,
            cap_norm: vector("cap_embedder.0.weight", cfg.cap_feat_dim)?,
            cap: linear("cap_embedder.1", cfg.cap_feat_dim, dim, true)?,
            time1: linear("t_embedder.mlp.0", 256, 1024, true)?,
            time2: linear("t_embedder.mlp.2", 1024, ad, true)?,
            noise: blocks("noise_refiner", cfg.n_refiner_layers, true)?,
            context: blocks("context_refiner", cfg.n_refiner_layers, false)?,
            layers: blocks("layers", cfg.n_layers, true)?,
            x_pad: x_pad.0,
            cap_pad: cap_pad.0,
            final_mod: linear("all_final_layer.2-1.adaLN_modulation.1", ad, dim, true)?,
            final_out: linear("all_final_layer.2-1.linear", dim, cfg.in_channels * 4, true)?,
            norm_ones: dev.upload(&vec![1.; dim]),
            norm_zeros: dev.upload(&vec![0.; dim]),
            cfg,
        })
    }
    fn block<D: ComputeDevice<Buffer = B>>(
        &self,
        dev: &D,
        b: &Block<B>,
        mut x: B,
        rows: usize,
        cos: &B,
        sin: &B,
        time: &B,
    ) -> B {
        let c = &self.cfg;
        let d = c.dim;
        let hd = d / c.n_heads;
        let modulation = b
            .modulation
            .as_ref()
            .map(|m| dev.download(&m.run(dev, time, 1)));
        let mut a = dev.rms_norm(&x, &b.an1, rows, d, c.norm_eps);
        if let Some(m) = &modulation {
            a = multiply_rows(
                dev,
                &a,
                &m[..d].iter().map(|v| 1. + v).collect::<Vec<_>>(),
                rows,
            );
        }
        let mut q = b.q.run(dev, &a, rows);
        let mut k = b.k.run(dev, &a, rows);
        let v = b.v.run(dev, &a, rows);
        if let Some(w) = &b.q_norm {
            q = dev.rms_norm(&q, w, rows * c.n_heads, hd, 1e-5);
        }
        if let Some(w) = &b.k_norm {
            k = dev.rms_norm(&k, w, rows * c.n_heads, hd, 1e-5);
        }
        q = dev.rope_forward_cached(&q, cos, sin, rows, c.n_heads, hd, 0);
        k = dev.rope_forward_cached(&k, cos, sin, rows, c.n_heads, hd, 0);
        let attention = dev.attention_full(&q, &k, &v, rows, c.n_heads, hd);
        let mut a = dev.rms_norm(
            &b.out.run(dev, &attention, rows),
            &b.an2,
            rows,
            d,
            c.norm_eps,
        );
        if let Some(m) = &modulation {
            a = multiply_rows(
                dev,
                &a,
                &m[d..2 * d].iter().map(|v| v.tanh()).collect::<Vec<_>>(),
                rows,
            );
        }
        x = dev.add_tensors_buf(&x, &a, rows * d);
        let mut a = dev.rms_norm(&x, &b.fn1, rows, d, c.norm_eps);
        if let Some(m) = &modulation {
            a = multiply_rows(
                dev,
                &a,
                &m[2 * d..3 * d].iter().map(|v| 1. + v).collect::<Vec<_>>(),
                rows,
            );
        }
        let gate = b.w1.run(dev, &a, rows);
        let up = b.w3.run(dev, &a, rows);
        let activated = dev.swiglu_fused_buf(&gate, &up, rows * b.w1.output);
        let mut a = dev.rms_norm(
            &b.w2.run(dev, &activated, rows),
            &b.fn2,
            rows,
            d,
            c.norm_eps,
        );
        if let Some(m) = &modulation {
            a = multiply_rows(
                dev,
                &a,
                &m[3 * d..].iter().map(|v| v.tanh()).collect::<Vec<_>>(),
                rows,
            );
        }
        dev.add_tensors_buf(&x, &a, rows * d)
    }
    fn rope<D: ComputeDevice<Buffer = B>>(
        &self,
        dev: &D,
        positions: &[[usize; 3]],
    ) -> Result<(B, B)> {
        let mut cos = Vec::new();
        let mut sin = Vec::new();
        for pos in positions {
            for axis in 0..3 {
                ensure!(
                    pos[axis] < self.cfg.axes_lens[axis],
                    "position exceeds RoPE axis {axis}"
                );
                let dim = self.cfg.axes_dims[axis];
                for i in (0..dim).step_by(2) {
                    // diffusers computes frequencies in f64, rounds the angle to f32,
                    // then computes complex64 polar coordinates.
                    let angle = (pos[axis] as f64
                        * self.cfg.rope_theta.powf(-(i as f64) / dim as f64))
                        as f32;
                    cos.push(angle.cos());
                    sin.push(angle.sin());
                }
            }
        }
        Ok((dev.upload(&cos), dev.upload(&sin)))
    }
    /// One image's velocity prediction. Latents are channel-first `[C,H,W]`;
    /// caption features are unpadded `[caption_tokens,cap_feat_dim]`. t is normalized.
    pub fn forward<D: ComputeDevice<Buffer = B>>(
        &self,
        dev: &D,
        latent: &[f32],
        height: usize,
        width: usize,
        caption: &B,
        caption_tokens: usize,
        t: f32,
    ) -> Result<Vec<f32>> {
        self.forward_trace(
            dev,
            latent,
            height,
            width,
            caption,
            caption_tokens,
            t,
            &mut |_, _| {},
        )
    }
    pub fn forward_trace<D: ComputeDevice<Buffer = B>>(
        &self,
        dev: &D,
        latent: &[f32],
        height: usize,
        width: usize,
        caption: &B,
        caption_tokens: usize,
        t: f32,
        trace: &mut dyn FnMut(&str, &B),
    ) -> Result<Vec<f32>> {
        let c = &self.cfg;
        ensure!(
            height > 0 && width > 0 && height % 2 == 0 && width % 2 == 0,
            "latent dimensions must be positive/even"
        );
        ensure!(
            latent.len() == c.in_channels * height * width
                && caption_tokens > 0
                && caption.len() == caption_tokens * c.cap_feat_dim,
            "invalid input shape"
        );
        ensure!(t.is_finite(), "invalid timestep");
        let cap_rows = caption_tokens.next_multiple_of(32);
        let image_rows = height / 2 * (width / 2);
        let x_rows = image_rows.next_multiple_of(32);
        let patches = patchify(latent, c.in_channels, height, width);
        let mut packed = patches.clone();
        while packed.len() < x_rows * c.in_channels * 4 {
            packed.extend_from_slice(&patches[patches.len() - c.in_channels * 4..]);
        }
        let mut x = dev.download(&self.x.run(dev, &dev.upload(&packed), x_rows));
        for row in image_rows..x_rows {
            x[row * c.dim..(row + 1) * c.dim].copy_from_slice(&self.x_pad);
        }
        let mut positions = Vec::new();
        for y in 0..height / 2 {
            for x in 0..width / 2 {
                positions.push([cap_rows + 1, y, x]);
            }
        }
        positions.resize(x_rows, [0, 0, 0]);
        let (cos, sin) = self.rope(dev, &positions)?;
        let freqs: Vec<f32> = (0..128)
            .map(|i| (-10000f64.ln() * i as f64 / 128.).exp() as f32)
            .collect();
        let mut embedding: Vec<f32> = freqs.iter().map(|f| (t * c.t_scale * f).cos()).collect();
        embedding.extend(freqs.iter().map(|f| (t * c.t_scale * f).sin()));
        let time = self.time2.run(
            dev,
            &silu(dev, &self.time1.run(dev, &dev.upload(&embedding), 1)),
            1,
        );
        let mut x = dev.upload(&x);
        trace("image_embed", &x);
        for (i, b) in self.noise.iter().enumerate() {
            x = self.block(dev, b, x, x_rows, &cos, &sin, &time);
            trace(&format!("noise_refiner.{i}"), &x);
        }
        let mut cap = dev.download(caption);
        let last = cap[cap.len() - c.cap_feat_dim..].to_vec();
        for _ in caption_tokens..cap_rows {
            cap.extend_from_slice(&last);
        }
        let cap = dev.rms_norm(
            &dev.upload(&cap),
            &self.cap_norm,
            cap_rows,
            c.cap_feat_dim,
            c.norm_eps,
        );
        let mut cap = dev.download(&self.cap.run(dev, &cap, cap_rows));
        for row in caption_tokens..cap_rows {
            cap[row * c.dim..(row + 1) * c.dim].copy_from_slice(&self.cap_pad);
        }
        let cap_positions: Vec<_> = (1..=cap_rows).map(|p| [p, 0, 0]).collect();
        let (cap_cos, cap_sin) = self.rope(dev, &cap_positions)?;
        let mut cap = dev.upload(&cap);
        trace("caption_embed", &cap);
        for (i, b) in self.context.iter().enumerate() {
            cap = self.block(dev, b, cap, cap_rows, &cap_cos, &cap_sin, &time);
            trace(&format!("context_refiner.{i}"), &cap);
        }
        let mut joined = dev.download(&x);
        joined.extend(dev.download(&cap));
        positions.extend(cap_positions);
        let (cos, sin) = self.rope(dev, &positions)?;
        let rows = x_rows + cap_rows;
        let mut x = dev.upload(&joined);
        for (i, b) in self.layers.iter().enumerate() {
            x = self.block(dev, b, x, rows, &cos, &sin, &time);
            trace(&format!("layers.{i}"), &x);
        }
        let x = dev.layer_norm(&x, &self.norm_ones, &self.norm_zeros, rows, c.dim, 1e-6);
        let scale = dev
            .download(&self.final_mod.run(dev, &silu(dev, &time), 1))
            .iter()
            .map(|v| 1. + v)
            .collect::<Vec<_>>();
        let x = multiply_rows(dev, &x, &scale, rows);
        let out = self.final_out.run(dev, &x, rows);
        trace("final", &out);
        Ok(unpatchify(
            &dev.download(&out)[..image_rows * c.in_channels * 4],
            c.in_channels,
            height,
            width,
        ))
    }
}
fn patchify(x: &[f32], channels: usize, h: usize, w: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    for y in (0..h).step_by(2) {
        for xx in (0..w).step_by(2) {
            for dy in 0..2 {
                for dx in 0..2 {
                    for c in 0..channels {
                        out.push(x[(c * h + y + dy) * w + xx + dx]);
                    }
                }
            }
        }
    }
    out
}
fn unpatchify(x: &[f32], channels: usize, h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0.; x.len()];
    let mut i = 0;
    for y in (0..h).step_by(2) {
        for xx in (0..w).step_by(2) {
            for dy in 0..2 {
                for dx in 0..2 {
                    for c in 0..channels {
                        out[(c * h + y + dy) * w + xx + dx] = x[i];
                        i += 1;
                    }
                }
            }
        }
    }
    out
}
