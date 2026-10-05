//! FLUX/Z-Image VAE decoder. Metal uses implicit tiled convolution, normalization
//! and nearest upsampling on device. Portable convolution uses bounded host im2col.
use crate::weights::Weights;
use anyhow::{ensure, Result};
use serde::Deserialize;
use std::path::Path;
use tang_compute::{ComputeBuffer, ComputeDevice};
#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub latent_channels: usize,
    pub out_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub norm_num_groups: usize,
    pub act_fn: String,
    pub up_block_types: Vec<String>,
    pub use_post_quant_conv: bool,
    pub mid_block_add_attention: bool,
    pub scaling_factor: f32,
    pub shift_factor: f32,
}
struct Conv<B> {
    weight: B,
    bias: B,
    input: usize,
    output: usize,
    kernel: usize,
}
struct Norm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}
struct Res<B> {
    n1: Norm,
    n2: Norm,
    c1: Conv<B>,
    c2: Conv<B>,
    shortcut: Option<Conv<B>>,
}
struct Attention<B> {
    norm: Norm,
    q: (B, B),
    k: (B, B),
    v: (B, B),
    out: (B, B),
    channels: usize,
}
struct Up<B> {
    res: Vec<Res<B>>,
    upsample: Option<Conv<B>>,
}
pub struct Vae<B> {
    pub cfg: Config,
    input: Conv<B>,
    mid1: Res<B>,
    mid2: Res<B>,
    attention: Attention<B>,
    up: Vec<Up<B>>,
    norm: Norm,
    output: Conv<B>,
}
fn silu<D: ComputeDevice>(dev: &D, x: &D::Buffer) -> D::Buffer {
    dev.silu_buf(x, x.len())
}
fn norm<D: ComputeDevice>(
    dev: &D,
    n: &Norm,
    x: &D::Buffer,
    h: usize,
    w: usize,
    groups: usize,
) -> D::Buffer {
    dev.group_norm_affine(
        x,
        &dev.upload(&n.weight),
        &dev.upload(&n.bias),
        n.weight.len(),
        h * w,
        groups,
        1e-6,
    )
}
impl<B: ComputeBuffer> Conv<B> {
    fn run<D: ComputeDevice<Buffer = B>>(&self, dev: &D, x: &B, h: usize, w: usize) -> B {
        dev.conv2d_nchw(
            x,
            &self.weight,
            &self.bias,
            self.input,
            self.output,
            h,
            w,
            self.kernel,
        )
    }
}
impl<B: ComputeBuffer> Res<B> {
    fn run<D: ComputeDevice<Buffer = B>>(
        &self,
        dev: &D,
        x: &B,
        h: usize,
        w: usize,
        groups: usize,
    ) -> B {
        let y = self
            .c1
            .run(dev, &silu(dev, &norm(dev, &self.n1, x, h, w, groups)), h, w);
        let y = self.c2.run(
            dev,
            &silu(dev, &norm(dev, &self.n2, &y, h, w, groups)),
            h,
            w,
        );
        let skip = self.shortcut.as_ref().map(|c| c.run(dev, x, h, w));
        dev.add_tensors_buf(skip.as_ref().unwrap_or(x), &y, y.len())
    }
}
impl<B: ComputeBuffer> Attention<B> {
    fn run<D: ComputeDevice<Buffer = B>>(
        &self,
        dev: &D,
        x: &B,
        h: usize,
        w: usize,
        groups: usize,
    ) -> B {
        let n = h * w;
        let normalized = norm(dev, &self.norm, x, h, w, groups);
        let rows = dev.transpose_2d(&normalized, self.channels, n);
        let linear = |pair: &(B, B), x: &B| {
            let y = dev.linear(x, &pair.0, n, self.channels, self.channels);
            dev.bias_add(&y, &pair.1, n * self.channels, self.channels)
        };
        let (q, k, v) = (
            linear(&self.q, &rows),
            linear(&self.k, &rows),
            linear(&self.v, &rows),
        );
        let attention = if self.channels <= 128 {
            dev.attention_full(&q, &k, &v, n, 1, self.channels)
        } else {
            // VAE uses one 512-wide head; tang's decoder attention kernels
            // stop at head_dim 256. The general GEMM/softmax path has no such limit.
            let kt = dev.transpose_2d(&k, n, self.channels);
            let mut scores = dev.matmul(&q, &kt, n, self.channels, n);
            dev.scale_buffer(&mut scores, 1. / (self.channels as f32).sqrt());
            let probabilities = dev.softmax(&scores, n, n);
            dev.matmul(&probabilities, &v, n, n, self.channels)
        };
        let y = linear(&self.out, &attention);
        let y = dev.transpose_2d(&y, n, self.channels);
        dev.add_tensors_buf(x, &y, x.len())
    }
}
impl<B: ComputeBuffer> Vae<B> {
    pub fn load<D: ComputeDevice<Buffer = B>>(dev: &D, dir: &Path) -> Result<Self> {
        let cfg: Config = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        ensure!(
            cfg.act_fn == "silu" && !cfg.use_post_quant_conv && cfg.mid_block_add_attention,
            "unsupported VAE decoder variant"
        );
        ensure!(
            !cfg.block_out_channels.is_empty()
                && cfg.block_out_channels.len() == cfg.up_block_types.len()
                && cfg.up_block_types.iter().all(|s| s == "UpDecoderBlock2D"),
            "unsupported VAE up blocks"
        );
        ensure!(
            cfg.norm_num_groups > 0
                && cfg
                    .block_out_channels
                    .iter()
                    .all(|c| *c > 0 && c % cfg.norm_num_groups == 0),
            "invalid group normalization channels"
        );
        ensure!(
            cfg.scaling_factor > 0.
                && cfg.scaling_factor.is_finite()
                && cfg.shift_factor.is_finite(),
            "invalid latent scaling"
        );
        let w = Weights::open(dir)?;
        let vector = |name: &str, n: usize| -> Result<Vec<f32>> {
            let (v, s) = w.f32(name)?;
            ensure!(s == [n], "invalid shape {name}: {s:?}");
            Ok(v)
        };
        let norm = |name: &str, n: usize| -> Result<Norm> {
            Ok(Norm {
                weight: vector(&format!("{name}.weight"), n)?,
                bias: vector(&format!("{name}.bias"), n)?,
            })
        };
        let conv = |name: &str, input: usize, output: usize, kernel: usize| -> Result<Conv<B>> {
            let (v, s) = w.f32(&format!("{name}.weight"))?;
            ensure!(
                s == [output, input, kernel, kernel],
                "invalid convolution {name}: {s:?}"
            );
            Ok(Conv {
                weight: dev.upload(&v),
                bias: dev.upload(&vector(&format!("{name}.bias"), output)?),
                input,
                output,
                kernel,
            })
        };
        let res = |name: &str, input: usize, output: usize| -> Result<Res<B>> {
            Ok(Res {
                n1: norm(&format!("{name}.norm1"), input)?,
                n2: norm(&format!("{name}.norm2"), output)?,
                c1: conv(&format!("{name}.conv1"), input, output, 3)?,
                c2: conv(&format!("{name}.conv2"), output, output, 3)?,
                shortcut: if input != output {
                    Some(conv(&format!("{name}.conv_shortcut"), input, output, 1)?)
                } else {
                    None
                },
            })
        };
        let channels = *cfg.block_out_channels.last().unwrap();
        let linear = |name: &str| -> Result<(B, B)> {
            let (v, s) = w.f32(&format!("{name}.weight"))?;
            ensure!(s == [channels, channels], "invalid attention matrix {name}");
            Ok((
                dev.upload(&v),
                dev.upload(&vector(&format!("{name}.bias"), channels)?),
            ))
        };
        let attention = Attention {
            norm: norm("decoder.mid_block.attentions.0.group_norm", channels)?,
            q: linear("decoder.mid_block.attentions.0.to_q")?,
            k: linear("decoder.mid_block.attentions.0.to_k")?,
            v: linear("decoder.mid_block.attentions.0.to_v")?,
            out: linear("decoder.mid_block.attentions.0.to_out.0")?,
            channels,
        };
        let mut up = Vec::new();
        let mut prev = channels;
        for (i, &output) in cfg.block_out_channels.iter().rev().enumerate() {
            let mut blocks = Vec::new();
            for j in 0..=cfg.layers_per_block {
                blocks.push(res(
                    &format!("decoder.up_blocks.{i}.resnets.{j}"),
                    prev,
                    output,
                )?);
                prev = output;
            }
            up.push(Up {
                res: blocks,
                upsample: if i + 1 < cfg.block_out_channels.len() {
                    Some(conv(
                        &format!("decoder.up_blocks.{i}.upsamplers.0.conv"),
                        output,
                        output,
                        3,
                    )?)
                } else {
                    None
                },
            });
        }
        Ok(Self {
            input: conv("decoder.conv_in", cfg.latent_channels, channels, 3)?,
            mid1: res("decoder.mid_block.resnets.0", channels, channels)?,
            mid2: res("decoder.mid_block.resnets.1", channels, channels)?,
            attention,
            up,
            norm: norm("decoder.conv_norm_out", prev)?,
            output: conv("decoder.conv_out", prev, cfg.out_channels, 3)?,
            cfg,
        })
    }
    /// Decode unscaled channel-first latents to raw [-1,1] pixels. The pipeline
    /// applies latent/scaling_factor + shift_factor before this call.
    pub fn decode<D: ComputeDevice<Buffer = B>>(
        &self,
        dev: &D,
        latent: &[f32],
        h: usize,
        w: usize,
    ) -> Result<Vec<f32>> {
        self.decode_trace(dev, latent, h, w, &mut |_, _| {})
    }
    pub fn decode_trace<D: ComputeDevice<Buffer = B>>(
        &self,
        dev: &D,
        latent: &[f32],
        mut h: usize,
        mut w: usize,
        trace: &mut dyn FnMut(&str, &B),
    ) -> Result<Vec<f32>> {
        ensure!(
            h > 0 && w > 0 && latent.len() == self.cfg.latent_channels * h * w,
            "invalid latent shape"
        );
        let groups = self.cfg.norm_num_groups;
        let mut x = self.input.run(dev, &dev.upload(latent), h, w);
        trace("conv_in", &x);
        x = self.mid1.run(dev, &x, h, w, groups);
        dev.sync();
        trace("mid.resnets.0", &x);
        x = self.attention.run(dev, &x, h, w, groups);
        dev.sync();
        trace("mid.attention", &x);
        x = self.mid2.run(dev, &x, h, w, groups);
        dev.sync();
        trace("mid.resnets.1", &x);
        for (i, up) in self.up.iter().enumerate() {
            for (j, r) in up.res.iter().enumerate() {
                x = r.run(dev, &x, h, w, groups);
                // Bound retained GPU activations now that convolution needs no host sync.
                dev.sync();
                trace(&format!("up.{i}.resnets.{j}"), &x);
            }
            if let Some(c) = &up.upsample {
                let enlarged = dev.upsample_nearest_2x(&x, x.len() / (h * w), h, w);
                h *= 2;
                w *= 2;
                x = c.run(dev, &enlarged, h, w);
                dev.sync();
                trace(&format!("up.{i}.upsample"), &x);
            }
        }
        x = silu(dev, &norm(dev, &self.norm, &x, h, w, groups));
        x = self.output.run(dev, &x, h, w);
        trace("output", &x);
        Ok(dev.download(&x))
    }
}
