//! Gemma 3's vision side: a SigLIP encoder (patch embedding, bidirectional transformer with
//! LayerNorms and biases) and the multimodal projector (4×4 average pooling to 256 tokens, a
//! `(1 + w)` RMSNorm and a projection into the language model's embedding space).

use crate::config::VisionConfig;
use crate::weights::Weights;
use anyhow::Result;
use tang_compute::ComputeDevice;

struct Layer<B> {
    ln1: (B, B),
    q: (B, B),
    k: (B, B),
    v: (B, B),
    o: (B, B),
    ln2: (B, B),
    fc1: (B, B),
    fc2: (B, B),
}

pub struct Vision<B> {
    pub cfg: VisionConfig,
    /// Patch embedding as a matrix `[hidden, patch*patch*3]` (from `[out, kh, kw, in]`), and bias.
    patch: (B, B),
    /// Learned position embeddings, `[patches, hidden]`.
    pos: B,
    layers: Vec<Layer<B>>,
    post: (B, B),
    /// Projector: RMSNorm weight (with the 1 folded in) and `[text_hidden, hidden]`.
    mm_norm: B,
    mm_proj: B,
    /// Language model width, and image tokens per image (after pooling).
    text_hidden: usize,
    pub tokens: usize,
}

impl<B> Vision<B> {
    pub fn load<D: ComputeDevice<Buffer = B>>(
        dev: &D,
        w: &Weights,
        cfg: VisionConfig,
        text_hidden: usize,
        tokens: usize,
    ) -> Result<Self> {
        let p = "vision_tower.vision_model";
        let f32v = |n: &str| -> Result<B> { Ok(dev.upload(&w.f32(n)?.0)) };
        let mat = |n: &str| -> Result<B> { Ok(dev.upload_bf16(&w.bf16(n)?)) };
        let lin = |n: &str| -> Result<(B, B)> {
            Ok((mat(&format!("{n}.weight"))?, f32v(&format!("{n}.bias"))?))
        };
        let norm = |n: &str| -> Result<(B, B)> {
            Ok((f32v(&format!("{n}.weight"))?, f32v(&format!("{n}.bias"))?))
        };
        let layers = (0..cfg.num_hidden_layers)
            .map(|l| {
                let q = format!("{p}.encoder.layers.{l}");
                Ok(Layer {
                    ln1: norm(&format!("{q}.layer_norm1"))?,
                    q: lin(&format!("{q}.self_attn.q_proj"))?,
                    k: lin(&format!("{q}.self_attn.k_proj"))?,
                    v: lin(&format!("{q}.self_attn.v_proj"))?,
                    o: lin(&format!("{q}.self_attn.out_proj"))?,
                    ln2: norm(&format!("{q}.layer_norm2"))?,
                    fc1: lin(&format!("{q}.mlp.fc1"))?,
                    fc2: lin(&format!("{q}.mlp.fc2"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        // MLX stores the patch conv as [out, kh, kw, in], which flattens to exactly the
        // (kh, kw, channel) order `patches` produces.
        let patch = (
            f32v(&format!("{p}.embeddings.patch_embedding.weight"))?,
            f32v(&format!("{p}.embeddings.patch_embedding.bias"))?,
        );
        let mut mm_norm = w.f32("multi_modal_projector.mm_soft_emb_norm.weight")?.0;
        mm_norm.iter_mut().for_each(|x| *x += 1.0);
        // Stored `[vision_hidden, text_hidden]`; linear wants `[out, in]`.
        let (proj, shape) = w.f32("multi_modal_projector.mm_input_projection_weight")?;
        let (vin, vout) = (shape[0], shape[1]);
        let mut t = vec![0.0f32; proj.len()];
        for i in 0..vin {
            for o in 0..vout {
                t[o * vin + i] = proj[i * vout + o];
            }
        }
        Ok(Self {
            patch,
            pos: f32v(&format!("{p}.embeddings.position_embedding.weight"))?,
            layers,
            post: norm(&format!("{p}.post_layernorm"))?,
            mm_norm: dev.upload(&mm_norm),
            mm_proj: dev.upload(&t),
            text_hidden,
            tokens,
            cfg,
        })
    }

    /// An image (`[size, size, 3]` row-major, normalized to [-1, 1]) as `tokens` embeddings in
    /// the language model's space, `[tokens, text_hidden]`, scaled so that the decoder's input
    /// embedding scale leaves them as the projector produced them divided by sqrt(hidden)
    /// (as the reference merges them).
    pub fn encode<D: ComputeDevice<Buffer = B>>(
        &self,
        dev: &D,
        pixels: &[f32],
        embed_scale: f32,
    ) -> B {
        let mut out = self.project(dev, pixels);
        // The reference divides by sqrt(hidden) before the decoder multiplies every input
        // embedding by its (bf16-rounded) scale; embeddings here are already scaled.
        dev.scale_buffer(&mut out, embed_scale / (self.text_hidden as f32).sqrt());
        out
    }

    /// The projector's output for an image, `[tokens, text_hidden]` (unscaled).
    pub fn project<D: ComputeDevice<Buffer = B>>(&self, dev: &D, pixels: &[f32]) -> B {
        let c = &self.cfg;
        let (side, ps, h) = (c.image_size / c.patch_size, c.patch_size, c.hidden_size);
        let n = side * side;
        let (nh, eps) = (c.num_attention_heads, c.layer_norm_eps);
        let hd = h / nh;

        let patches = dev.upload(&patches(pixels, c.image_size, ps));
        let mut x = dev.linear(&patches, &self.patch.0, n, ps * ps * 3, h);
        x = dev.bias_add(&x, &self.patch.1, n * h, h);
        x = dev.add_tensors_buf(&x, &self.pos, n * h);

        let lin = |x: &B, (w, b): &(B, B), k: usize, out: usize| {
            let y = dev.linear(x, w, n, k, out);
            dev.bias_add(&y, b, n * out, out)
        };
        for l in &self.layers {
            let a = dev.layer_norm(&x, &l.ln1.0, &l.ln1.1, n, h, eps);
            let (q, k, v) = (
                lin(&a, &l.q, h, h),
                lin(&a, &l.k, h, h),
                lin(&a, &l.v, h, h),
            );
            let att = dev.attention_full(&q, &k, &v, n, nh, hd);
            x = dev.add_tensors_buf(&x, &lin(&att, &l.o, h, h), n * h);
            let m = dev.layer_norm(&x, &l.ln2.0, &l.ln2.1, n, h, eps);
            let f = lin(&m, &l.fc1, h, c.intermediate_size);
            let f = dev.gelu_tanh(&f, n * c.intermediate_size);
            x = dev.add_tensors_buf(&x, &lin(&f, &l.fc2, c.intermediate_size, h), n * h);
            dev.flush();
        }
        let x = dev.layer_norm(&x, &self.post.0, &self.post.1, n, h, eps);

        // Average-pool the patch grid down to `tokens` (e.g. 64×64 → 16×16).
        let per_side = (self.tokens as f64).sqrt() as usize;
        let k = side / per_side;
        let feats = dev.download(&x);
        let mut pooled = vec![0.0f32; per_side * per_side * h];
        let inv = 1.0 / (k * k) as f32;
        for py in 0..side {
            for px in 0..side {
                let t = (py / k) * per_side + px / k;
                let src = &feats[(py * side + px) * h..][..h];
                let dst = &mut pooled[t * h..][..h];
                for (d, s) in dst.iter_mut().zip(src) {
                    *d += s * inv;
                }
            }
        }
        let m = per_side * per_side;
        let pooled = dev.upload(&pooled);
        let normed = dev.rms_norm(&pooled, &self.mm_norm, m, h, eps);
        dev.linear(&normed, &self.mm_proj, m, h, self.text_hidden)
    }
}

/// Decode an image (PNG, JPEG, WebP, GIF) and prepare it for the tower: RGB, resized to
/// `size`×`size` (bilinear), scaled to [-1, 1], `[size, size, 3]` row-major.
pub fn preprocess(bytes: &[u8], size: usize) -> Result<Vec<f32>> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| anyhow::anyhow!("couldn't read the image: {e}"))?
        .to_rgb8();
    let img = image::imageops::resize(
        &img,
        size as u32,
        size as u32,
        image::imageops::FilterType::Triangle,
    );
    Ok(img
        .into_raw()
        .into_iter()
        .map(|b| (b as f32 / 255.0 - 0.5) / 0.5)
        .collect())
}

/// Non-overlapping `ps`×`ps` patches of an `[size, size, 3]` image, row-major over the patch
/// grid, each flattened in (row, column, channel) order.
fn patches(pixels: &[f32], size: usize, ps: usize) -> Vec<f32> {
    let side = size / ps;
    let mut out = Vec::with_capacity(side * side * ps * ps * 3);
    for py in 0..side {
        for px in 0..side {
            for ky in 0..ps {
                let row = (py * ps + ky) * size + px * ps;
                out.extend_from_slice(&pixels[row * 3..(row + ps) * 3]);
            }
        }
    }
    out
}
