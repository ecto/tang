//! Qwen3-family decoder (also Llama-shaped models: q/k norm is used only when present), and
//! Gemma 3: GeGLU, `(1 + w)` RMSNorms around both attention and MLP, a scaled embedding, and
//! local sliding-window layers (own RoPE base) between global ones (scaled RoPE). And gpt-oss:
//! Q/K/V and output biases, attention sinks, alternating sliding-window and full layers, YaRN,
//! and a routed MoE ([`Mlp::Moe`], with MXFP4 experts as the checkpoints ship them).

use crate::blocks::{self, Shared, BLOCK};
use crate::config::Config;
use crate::weights::Weights;
use anyhow::Result;
use std::path::Path;
use tang_compute::moe::{ExpertFmt, Mxfp4, Router};
use tang_compute::ComputeDevice;

/// How matrices are stored on device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    Bf16,
    /// 4-bit affine groups of 64 (quantized at load unless the checkpoint already is).
    Q4,
}

struct Layer<B> {
    attn_norm: B,
    /// Q, K and V projections stacked: `[(nh + 2*nkv) * hd, hidden]`.
    wqkv: B,
    wo: B,
    q_norm: Option<B>,
    k_norm: Option<B>,
    mlp_norm: B,
    mlp: Mlp<B>,
    /// gpt-oss: biases on the stacked Q/K/V projection and the output projection, and one
    /// attention-sink logit per head.
    bqkv: Option<B>,
    bo: Option<B>,
    sinks: Option<B>,
    /// Gemma: norms on the attention and MLP outputs, before the residual adds.
    post_attn_norm: Option<B>,
    post_mlp_norm: Option<B>,
    /// Sliding window (0: global attention), and which RoPE tables (0 global, 1 local).
    window: usize,
    rope: usize,
}

enum Mlp<B> {
    Dense {
        /// Gate and up projections stacked: `[2 * ff, hidden]`.
        w_gate_up: B,
        w_down: B,
    },
    Moe(Moe<B>),
}

/// A routed mixture of experts. Expert matrices are stacked `[experts, n, k]` in `fmt`; each
/// expert's gate/up rows are gate then up (gpt-oss interleaves them; we split at load).
struct Moe<B> {
    router: B,
    router_bias: Option<B>,
    route: Router,
    experts: usize,
    top_k: usize,
    ff: usize,
    fmt: ExpertFmt,
    gate_up: B,
    gate_up_bias: Option<B>,
    down: B,
    down_bias: Option<B>,
    /// gpt-oss's clamped SwiGLU `(alpha, limit)`; plain SwiGLU when `None`.
    clamp: Option<(f32, f32)>,
}

pub struct Model<D: ComputeDevice> {
    pub cfg: Config,
    pub dev: D,
    embed: D::Buffer,
    layers: Vec<Layer<D::Buffer>>,
    norm: D::Buffer,
    lm_head: Option<D::Buffer>,
    /// RoPE tables: global, and (Gemma) local.
    ropes: Vec<(D::Buffer, D::Buffer)>,
    /// Gemma scales embeddings by sqrt(hidden), rounded to bf16 as the reference does.
    embed_scale: Option<f32>,
    /// The vision tower, for multimodal checkpoints.
    pub vision: Option<crate::vision::Vision<D::Buffer>>,
    max_ctx: usize,
    /// Every cache's K and V blocks (see [`crate::blocks`]): bf16 (half the memory of f32;
    /// attention still accumulates in f32) unless the device has no bf16 storage, or with
    /// `TANG_KV_F32=1`.
    pool: Shared<D::Buffer>,
    /// Faster implementations registered for this device and model (see [`crate::ext`]).
    ext: crate::ext::Ext<D>,
}

pub use crate::blocks::Cache;

/// One layer's weights (see [`Model::layer_weights`]). Matrices are in the load dtype, shaped
/// as [`Layer`]'s fields describe; norms are f32 (Gemma's `1 + w` already folded in).
pub struct LayerWeights<'a, B> {
    pub attn_norm: &'a B,
    pub wqkv: &'a B,
    pub wo: &'a B,
    pub q_norm: Option<&'a B>,
    pub k_norm: Option<&'a B>,
    pub mlp_norm: &'a B,
    pub w_gate_up: &'a B,
    pub w_down: &'a B,
    pub post_attn_norm: Option<&'a B>,
    pub post_mlp_norm: Option<&'a B>,
    /// Sliding window (0: global).
    pub window: usize,
    /// This layer's RoPE (cos, sin) tables.
    pub rope: (&'a B, &'a B),
}

impl<D: ComputeDevice> Model<D> {
    /// Load a checkpoint directory. Matrices are kept in `dtype` on device; norms stay f32.
    pub fn load(dev: D, dir: &Path, max_ctx: usize, dtype: Dtype) -> Result<Self> {
        let cfg = Config::from_json(&std::fs::read(dir.join("config.json"))?)?;
        cfg.query_rescale()?;
        let w = Weights::open(dir)?;
        let gemma = cfg.is_gemma();
        // Multimodal checkpoints keep the language model under `language_model.`.
        let lm = if w.has("language_model.model.embed_tokens.weight")
            || w.has("language_model.model.embed_tokens.scales")
        {
            "language_model."
        } else {
            ""
        };
        // Gemma's RMSNorm scales by (1 + w): fold the 1 in at load.
        let vec = |name: &str| -> Result<D::Buffer> {
            let mut v = w.f32(name)?.0;
            if gemma {
                v.iter_mut().for_each(|x| *x += 1.0);
            }
            Ok(dev.upload(&v))
        };
        let quant = cfg.quantization.clone();
        if let Some(q) = &quant {
            anyhow::ensure!(
                q.bits == 4,
                "{}-bit checkpoints aren't supported yet (4-bit only)",
                q.bits
            );
        }
        // Matrices stacked along the output dimension (rows share K, so row-major data just
        // concatenates), for fused projections like Q/K/V.
        let up_cat = |names: &[String]| -> Result<D::Buffer> {
            let base = |n: &String| n.strip_suffix(".weight").unwrap_or(n).to_string();
            // Pre-quantized (MLX): `X.weight` (packed u32) with `X.scales` and `X.biases`.
            if let Some(q) = quant
                .as_ref()
                .filter(|_| w.has(&format!("{}.scales", base(&names[0]))))
            {
                let (mut p, mut sc, mut bi) = (Vec::new(), Vec::new(), Vec::new());
                for n in names {
                    p.extend(w.u32(n)?);
                    sc.extend(w.bf16(&format!("{}.scales", base(n)))?);
                    bi.extend(w.bf16(&format!("{}.biases", base(n)))?);
                }
                return Ok(dev.upload_q4(&p, &sc, &bi, q.group_size));
            }
            match dtype {
                Dtype::Bf16 => {
                    let mut all = Vec::new();
                    for n in names {
                        all.extend(w.bf16(n)?);
                    }
                    Ok(dev.upload_bf16(&all))
                }
                Dtype::F32 | Dtype::Q4 => {
                    let mut all = Vec::new();
                    for n in names {
                        all.extend(w.f32(n)?.0);
                    }
                    if dtype == Dtype::F32 {
                        return Ok(dev.upload(&all));
                    }
                    let (p, s, b) = crate::weights::quantize_q4(&all, 64);
                    Ok(dev.upload_q4(&p, &s, &b, 64))
                }
            }
        };
        let up = |name: &str| up_cat(&[name.to_string()]);
        let opt = |name: &str| -> Result<Option<D::Buffer>> {
            if w.has(name) {
                up(name).map(Some)
            } else {
                Ok(None)
            }
        };
        let opt_vec = |name: &str| -> Result<Option<D::Buffer>> {
            if w.has(name) {
                vec(name).map(Some)
            } else {
                Ok(None)
            }
        };
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let p = format!("{lm}model.layers.{l}");
            let norm = |n: &str| vec(&format!("{p}.{n}.weight"));
            let window = cfg.window(l);
            layers.push(Layer {
                attn_norm: norm("input_layernorm")?,
                wqkv: up_cat(&["q", "k", "v"].map(|x| format!("{p}.self_attn.{x}_proj.weight")))?,
                wo: up(&format!("{p}.self_attn.o_proj.weight"))?,
                q_norm: opt_vec(&format!("{p}.self_attn.q_norm.weight"))?,
                k_norm: opt_vec(&format!("{p}.self_attn.k_norm.weight"))?,
                // Gemma norms the MLP input with its own weight; Qwen reuses the post-attention one.
                mlp_norm: norm(if gemma {
                    "pre_feedforward_layernorm"
                } else {
                    "post_attention_layernorm"
                })?,
                mlp: if cfg.num_local_experts.is_some() {
                    Mlp::Moe(load_moe(&dev, &w, &cfg, &p, dtype)?)
                } else {
                    Mlp::Dense {
                        w_gate_up: up_cat(
                            &["gate", "up"].map(|x| format!("{p}.mlp.{x}_proj.weight")),
                        )?,
                        w_down: up(&format!("{p}.mlp.down_proj.weight"))?,
                    }
                },
                bqkv: if w.has(&format!("{p}.self_attn.q_proj.bias")) {
                    let mut b = Vec::new();
                    for x in ["q", "k", "v"] {
                        b.extend(w.f32(&format!("{p}.self_attn.{x}_proj.bias"))?.0);
                    }
                    Some(dev.upload(&b))
                } else {
                    None
                },
                bo: opt_vec(&format!("{p}.self_attn.o_proj.bias"))?,
                sinks: opt_vec(&format!("{p}.self_attn.sinks"))?,
                post_attn_norm: if gemma {
                    Some(norm("post_attention_layernorm")?)
                } else {
                    None
                },
                post_mlp_norm: if gemma {
                    Some(norm("post_feedforward_layernorm")?)
                } else {
                    None
                },
                window,
                rope: usize::from(gemma && window > 0),
            });
        }
        let embed = up(&format!("{lm}model.embed_tokens.weight"))?;
        let head_name = format!("{lm}lm_head.weight");
        let lm_head = if cfg.tie_word_embeddings || !w.has(&head_name) {
            None
        } else {
            opt(&head_name)?
        };
        let norm = vec(&format!("{lm}model.norm.weight"))?;

        let max_ctx = max_ctx.min(cfg.max_position_embeddings);
        let hd = cfg.head_dim();
        let scale = cfg.rope_scaling.as_ref().map_or(1.0, |s| s.factor);
        let mut ropes = Vec::new();
        let (cos, sin) = match cfg.rope_scaling.as_ref() {
            Some(y) if y.rope_type.as_deref() == Some("yarn") => {
                let (inv, mscale) = yarn(hd, cfg.rope_theta, y, cfg.max_position_embeddings);
                rope_tables_inv(&inv, max_ctx, mscale)
            }
            _ => rope_tables(hd, max_ctx, cfg.rope_theta, scale),
        };
        ropes.push((dev.upload(&cos), dev.upload(&sin)));
        if gemma {
            let base = cfg.rope_local_base_freq.unwrap_or(10_000.0);
            let (cos, sin) = rope_tables(hd, max_ctx, base, 1.0);
            ropes.push((dev.upload(&cos), dev.upload(&sin)));
        }
        let embed_scale = gemma.then(|| bf16_round((cfg.hidden_size as f32).sqrt()));
        let dev_bf16 = dev.bf16_storage();
        let (n_layers, kv_dim) = (layers.len(), cfg.kv_dim());
        let vision = match &cfg.vision {
            Some(v) if w.has("vision_tower.vision_model.post_layernorm.weight") => {
                Some(crate::vision::Vision::load(
                    &dev,
                    &w,
                    v.clone(),
                    cfg.hidden_size,
                    cfg.mm_tokens_per_image,
                )?)
            }
            _ => None,
        };
        let quant = if quant.is_some() || dtype == Dtype::Q4 {
            "q4"
        } else if dtype == Dtype::Bf16 {
            "bf16"
        } else {
            "f32"
        };
        let target = crate::ext::target(&dev, &cfg, quant);
        Ok(Self {
            ext: crate::ext::Ext::resolve(target),
            cfg,
            dev,
            embed,
            layers,
            norm,
            lm_head,
            ropes,
            embed_scale,
            vision,
            max_ctx,
            pool: new_pool(
                n_layers,
                kv_dim,
                dev_bf16 && !std::env::var("TANG_KV_F32").is_ok_and(|v| v == "1"),
                usize::MAX,
            ),
        })
    }

    /// Keep KV in f32 instead of bf16 (for comparing the two), in a new pool: caches made
    /// before keep the old one.
    pub fn set_kv_f32(&mut self, f32: bool) {
        let max = blocks::lock(&self.pool).stats().max;
        self.pool = new_pool(
            self.layers.len(),
            self.cfg.kv_dim(),
            !f32 && self.dev.bf16_storage(),
            max,
        );
    }

    /// Whether caches hold bf16 (else f32).
    pub fn kv_bf16(&self) -> bool {
        blocks::lock(&self.pool).bf16()
    }

    /// The pool every cache's blocks come from.
    pub fn pool(&self) -> &Shared<D::Buffer> {
        &self.pool
    }

    /// Cap the device memory KV blocks may take (`None`: no cap).
    pub fn set_kv_budget(&self, bytes: Option<usize>) {
        let mut pool = blocks::lock(&self.pool);
        let max = bytes.map_or(usize::MAX, |b| (b / pool.block_bytes()).max(1));
        pool.set_max_blocks(max);
    }

    /// What this model resolved from the extension registry (see [`crate::ext`]).
    pub fn ext(&self) -> &crate::ext::Ext<D> {
        &self.ext
    }

    /// Resolve the registry again (after registering something since load).
    pub fn resolve_ext(&mut self) {
        let t = self.ext.target.clone();
        self.ext = crate::ext::Ext::resolve(t);
    }

    /// Remove the resolved extensions, leaving the built-in kernels (and the target).
    pub fn take_ext(&mut self) -> crate::ext::Ext<D> {
        let t = self.ext.target.clone();
        std::mem::replace(
            &mut self.ext,
            crate::ext::Ext {
                target: t,
                ..Default::default()
            },
        )
    }

    pub fn set_ext(&mut self, e: crate::ext::Ext<D>) {
        self.ext = e;
    }

    /// Layer `l`'s weights, read-only, for step overrides that run the layers themselves.
    /// `None` for a layer these don't describe: a MoE layer, or attention with biases or sinks
    /// (gpt-oss).
    pub fn layer_weights(&self, l: usize) -> Option<LayerWeights<'_, D::Buffer>> {
        let w = &self.layers[l];
        let Mlp::Dense { w_gate_up, w_down } = &w.mlp else {
            return None;
        };
        if w.bqkv.is_some() || w.bo.is_some() || w.sinks.is_some() {
            return None;
        }
        Some(LayerWeights {
            attn_norm: &w.attn_norm,
            wqkv: &w.wqkv,
            wo: &w.wo,
            q_norm: w.q_norm.as_ref(),
            k_norm: w.k_norm.as_ref(),
            mlp_norm: &w.mlp_norm,
            w_gate_up,
            w_down,
            post_attn_norm: w.post_attn_norm.as_ref(),
            post_mlp_norm: w.post_mlp_norm.as_ref(),
            window: w.window,
            rope: self
                .ropes
                .get(w.rope)
                .map(|(c, s)| (c, s))
                .expect("rope table"),
        })
    }

    /// The embedding table, final norm and output head (the embedding when tied).
    pub fn io_weights(&self) -> (&D::Buffer, &D::Buffer, &D::Buffer) {
        (
            &self.embed,
            &self.norm,
            self.lm_head.as_ref().unwrap_or(&self.embed),
        )
    }

    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    /// Device memory a [`Cache`] of `rows` positions takes (K and V for every layer, in
    /// whole blocks).
    pub fn cache_bytes(&self, rows: usize) -> usize {
        rows.min(self.max_ctx).div_ceil(BLOCK) * blocks::lock(&self.pool).block_bytes()
    }

    /// Device memory per position: every layer's K and V, 2 bytes an element in bf16, else 4.
    pub fn row_bytes(&self) -> usize {
        self.row_elems() * if self.kv_bf16() { 2 } else { 4 }
    }

    pub fn new_cache(&self) -> Cache<D::Buffer> {
        Cache::new(self.pool.clone())
    }

    /// Elements per position in [`read_rows`](Self::read_rows): every layer's K and V.
    pub fn row_elems(&self) -> usize {
        2 * self.layers.len() * self.cfg.kv_dim()
    }

    /// Positions `from..to` of `cache` as bf16 bits (exact for a bf16 cache, rounded from an f32
    /// one), position-major: for each, every layer's K row then V row.
    pub fn read_rows(&self, cache: &Cache<D::Buffer>, from: usize, to: usize) -> Vec<u16> {
        let pool = blocks::lock(cache.pool());
        let mut out = Vec::with_capacity((to - from) * self.row_elems());
        let mut p = from;
        while p < to {
            let n = (BLOCK - p % BLOCK).min(to - p);
            out.extend(pool.read_block(&self.dev, cache.table[p / BLOCK], p % BLOCK, n));
            p += n;
        }
        out
    }

    /// Append positions to `cache`: `tokens`, with their rows as [`read_rows`](Self::read_rows)
    /// gives them.
    pub fn write_rows(&self, cache: &mut Cache<D::Buffer>, tokens: &[u32], rows: &[u16]) {
        let n = tokens.len();
        if n == 0 {
            return;
        }
        let pool = cache.pool().clone();
        let mut pool = blocks::lock(&pool);
        let (from, to) = (cache.len, cache.len + n);
        blocks::prepare(&self.dev, &mut pool, cache, from, to);
        let row = self.row_elems();
        let mut p = from;
        while p < to {
            let k = (BLOCK - p % BLOCK).min(to - p);
            let part = &rows[(p - from) * row..(p - from + k) * row];
            pool.write_block(&self.dev, cache.table[p / BLOCK], p % BLOCK, part);
            p += k;
        }
        cache.len = to;
        cache.tokens.extend_from_slice(tokens);
    }

    /// Run `tokens` through the model after what's already in `cache`, append them to it, and
    /// return logits: for the last token only, or for every token if `all` is set.
    pub fn forward(
        &self,
        tokens: &[u32],
        cache: &mut Cache<D::Buffer>,
        all: bool,
    ) -> Result<Vec<f32>> {
        let x = self.embed(tokens)?;
        self.forward_layers(x, tokens, cache, all, true, false)
    }

    /// An image's embeddings, ready to stand in for its placeholder tokens.
    pub fn encode_image(&self, pixels: &[f32]) -> Result<D::Buffer> {
        let v = self
            .vision
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("this model has no vision tower"))?;
        Ok(v.encode(&self.dev, pixels, self.embed_scale.unwrap_or(1.0)))
    }

    /// `forward` with images: each run of image placeholder tokens takes the next image's
    /// embeddings, and attends bidirectionally within itself (text stays causal).
    pub fn forward_images(
        &self,
        tokens: &[u32],
        images: &[D::Buffer],
        cache: &mut Cache<D::Buffer>,
        all: bool,
    ) -> Result<Vec<f32>> {
        let Some(img_tok) = self.cfg.image_token.filter(|_| !images.is_empty()) else {
            return self.forward(tokens, cache, all);
        };
        let h = self.cfg.hidden_size;
        let mut x = self.embed(tokens)?;
        // Runs of (start, len, is_image).
        let mut runs: Vec<(usize, usize, bool)> = Vec::new();
        for (i, &t) in tokens.iter().enumerate() {
            let img = t == img_tok;
            match runs.last_mut() {
                Some((_, len, kind)) if *kind == img => *len += 1,
                _ => runs.push((i, 1, img)),
            }
        }
        let mut next = images.iter();
        for &(start, len, img) in &runs {
            if img {
                let feat = next
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("more image placeholders than images"))?;
                anyhow::ensure!(
                    len == self.vision.as_ref().map_or(0, |v| v.tokens),
                    "an image takes {} placeholders, got {len}",
                    self.vision.as_ref().map_or(0, |v| v.tokens)
                );
                self.dev.write_into(&mut x, start * h, feat);
            }
        }
        // Text runs go in chunks (bounded scratch memory); an image block never splits.
        let mut pieces: Vec<(usize, usize, bool)> = Vec::new();
        for &(start, len, img) in &runs {
            if img {
                pieces.push((start, len, true));
            } else {
                let mut s = start;
                while s < start + len {
                    let n = (start + len - s).min(512);
                    pieces.push((s, n, false));
                    s += n;
                }
            }
        }
        let mut out = Vec::new();
        for (r, &(start, len, img)) in pieces.iter().enumerate() {
            let seg = self.dev.slice_buffer(&x, start * h, len * h);
            let last = r + 1 == pieces.len();
            let logits =
                self.forward_layers(seg, &tokens[start..start + len], cache, all, !img, true)?;
            if all {
                out.extend(logits);
            } else if last {
                out = logits;
            }
        }
        Ok(out)
    }

    /// Token embeddings (scaled, for Gemma), `[tokens, hidden]` on device.
    pub fn embed(&self, tokens: &[u32]) -> Result<D::Buffer> {
        anyhow::ensure!(!tokens.is_empty(), "no tokens");
        let ids = self.dev.upload_u32(tokens);
        let mut x = self
            .dev
            .embedding(&self.embed, &ids, tokens.len(), self.cfg.hidden_size);
        if let Some(scale) = self.embed_scale {
            self.dev.scale_buffer(&mut x, scale);
        }
        Ok(x)
    }

    /// Qwen conditioning used by diffusion pipelines: hidden_states[-2], before
    /// the final decoder layer, final RMSNorm, and language-model head.
    /// Returns a device-resident `[tokens.len(), hidden_size]` tensor. This uses
    /// a separate cache so conditioning never changes a chat conversation's KV.
    pub fn encode_penultimate(&self, tokens: &[u32]) -> Result<D::Buffer> {
        anyhow::ensure!(!tokens.is_empty(), "no tokens");
        anyhow::ensure!(tokens.len() <= self.max_ctx, "context full");
        anyhow::ensure!(!self.layers.is_empty(), "encoder has no layers");
        let mut x = self.embed(tokens)?;
        let mut cache = self.new_cache();
        self.decode_layers(
            &mut x,
            &mut [(&mut cache, tokens.len())],
            true,
            self.layers.len() - 1,
            true,
        )?;
        Ok(x)
    }

    /// Like `forward`, from hidden states instead of token ids (so image features can stand
    /// in for placeholder tokens). `tokens` are what the cache records for these positions.
    pub fn forward_hidden(
        &self,
        x: D::Buffer,
        tokens: &[u32],
        cache: &mut Cache<D::Buffer>,
        all: bool,
        causal: bool,
    ) -> Result<Vec<f32>> {
        self.forward_layers(x, tokens, cache, all, causal, false)
    }

    /// `forward_hidden`; `bound` waits for the GPU every few layers on large pieces, so
    /// multimodal and conditioning prefill don't retain every temporary at once.
    fn forward_layers(
        &self,
        mut x: D::Buffer,
        tokens: &[u32],
        cache: &mut Cache<D::Buffer>,
        all: bool,
        causal: bool,
        bound: bool,
    ) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let dev = &self.dev;
        let (s, pos) = (tokens.len(), cache.len);
        anyhow::ensure!(s > 0, "no tokens");
        anyhow::ensure!(
            pos + s <= self.max_ctx,
            "context full ({} tokens)",
            self.max_ctx
        );
        let (h, eps) = (c.hidden_size, c.rms_norm_eps);
        self.decode_layers(
            &mut x,
            &mut [(&mut *cache, s)],
            causal,
            self.layers.len(),
            bound,
        )?;
        let (rows, x) = if all {
            (s, x)
        } else {
            (1, dev.slice_buffer(&x, (s - 1) * h, h))
        };
        let x = dev.rms_norm(&x, &self.norm, rows, h, eps);
        let head = self.lm_head.as_ref().unwrap_or(&self.embed);
        let logits = dev.linear(&x, head, rows, h, c.vocab_size);
        let out = dev.download(&logits);
        cache.len += s;
        cache.tokens.extend_from_slice(tokens);
        Ok(out)
    }
}

impl<D: ComputeDevice> Model<D> {
    /// Several sequences' tokens through the model in one pass, each after what its own cache
    /// holds (see [`Segment`]). The projections and MLP run once over every segment's rows, so
    /// the weights are read once for the batch; RoPE, the KV write and attention run per
    /// segment over its own cache. Returns each segment's logits: its last row's, or every
    /// row's if its `all` is set. A segment's logits are what [`forward`](Self::forward) would
    /// return for it alone, up to the matmul kernel's summation order for this many rows.
    pub fn forward_batch(&self, segs: &mut [Segment<'_, D::Buffer>]) -> Result<Vec<Vec<f32>>> {
        if let Some(r) = self.ext.step.as_ref().and_then(|e| (e.imp.0)(self, segs)) {
            return r;
        }
        self.forward_batch_builtin(segs)
    }

    /// [`forward_batch`](Self::forward_batch) without a step override (what one falls back to).
    pub fn forward_batch_builtin(
        &self,
        segs: &mut [Segment<'_, D::Buffer>],
    ) -> Result<Vec<Vec<f32>>> {
        let c = &self.cfg;
        let dev = &self.dev;
        anyhow::ensure!(!segs.is_empty(), "no segments");
        for g in segs.iter() {
            anyhow::ensure!(!g.tokens.is_empty(), "no tokens");
            anyhow::ensure!(
                g.cache.len + g.tokens.len() <= self.max_ctx,
                "context full ({} tokens)",
                self.max_ctx
            );
        }
        // One-token segments first, so `decode_layers` can attend for them in one go, then the
        // other short ones (verify windows), then prefill chunks (see `row_groups`).
        let mut order: Vec<usize> = (0..segs.len()).collect();
        order.sort_by_key(|&i| {
            let n = segs[i].tokens.len();
            (n > SHORT_ROWS, n != 1)
        });
        let tokens: Vec<u32> = order
            .iter()
            .flat_map(|&i| segs[i].tokens.iter().copied())
            .collect();
        let mut x = self.embed(&tokens)?;
        let mut slots: Vec<Option<&mut Segment<'_, D::Buffer>>> =
            segs.iter_mut().map(Some).collect();
        let mut ordered: Vec<&mut Segment<'_, D::Buffer>> =
            order.iter().map(|&i| slots[i].take().unwrap()).collect();
        let mut parts: Vec<(&mut Cache<D::Buffer>, usize)> = ordered
            .iter_mut()
            .map(|g| (&mut *g.cache, g.tokens.len()))
            .collect();
        self.decode_layers(&mut x, &mut parts, true, self.layers.len(), false)?;
        drop(parts);
        // The rows whose logits are wanted, gathered so the head runs once.
        let h = c.hidden_size;
        let (mut at, mut rows) = (0usize, Vec::new());
        for g in ordered.iter() {
            let s = g.tokens.len();
            if g.all {
                rows.push((at, s));
            } else {
                rows.push((at + s - 1, 1));
            }
            at += s;
        }
        let n: usize = rows.iter().map(|r| r.1).sum();
        let picked = if n == tokens.len() {
            x
        } else {
            let mut picked = dev.alloc_uninit_f32(n * h);
            let mut o = 0;
            for &(r, len) in &rows {
                let part = dev.slice_buffer(&x, r * h, len * h);
                dev.write_into(&mut picked, o * h, &part);
                o += len;
            }
            picked
        };
        let picked = dev.rms_norm(&picked, &self.norm, n, h, c.rms_norm_eps);
        let head = self.lm_head.as_ref().unwrap_or(&self.embed);
        // In pieces of at most SHORT_ROWS rows, so each row's logits come from the kernel a
        // decode alone would use.
        let pieces: Vec<(usize, usize)> = (0..n)
            .step_by(SHORT_ROWS)
            .map(|r| (r, SHORT_ROWS.min(n - r)))
            .collect();
        let logits = dev.download(&self.linear_rows(&picked, head, &pieces, h, c.vocab_size));
        let v = c.vocab_size;
        let mut out = vec![Vec::new(); order.len()];
        let mut o = 0;
        for ((g, &(_, len)), &i) in ordered.iter_mut().zip(&rows).zip(&order) {
            out[i] = logits[o * v..(o + len) * v].to_vec();
            o += len;
            g.cache.len += g.tokens.len();
            g.cache.tokens.extend_from_slice(g.tokens);
        }
        Ok(out)
    }
}

/// Parts of a batch at most this long (decode, verify windows) share their matmuls; longer
/// ones (prefill chunks) get their own. The 4-bit CUDA kernel for up to this many rows sums
/// each row the same way whatever the count, so a row's numbers don't depend on its batch.
const SHORT_ROWS: usize = 8;

/// The row ranges a batch's matmuls run over, for parts `lens` long in order (short ones
/// first): the short parts together, then each long one alone, at the size it has when its
/// sequence runs alone.
fn row_groups(lens: &[usize]) -> Vec<(usize, usize)> {
    let short: usize = lens.iter().take_while(|&&n| n <= SHORT_ROWS).sum();
    let mut groups = Vec::new();
    if short > 0 {
        groups.push((0, short));
    }
    let mut at = short;
    for &n in lens.iter().skip_while(|&&n| n <= SHORT_ROWS) {
        groups.push((at, n));
        at += n;
    }
    groups
}

impl<D: ComputeDevice> Model<D> {
    /// `x · wᵀ` over `x`'s rows, a separate matmul for each `(first row, rows)` group.
    fn linear_rows(
        &self,
        x: &D::Buffer,
        w: &D::Buffer,
        groups: &[(usize, usize)],
        k: usize,
        n: usize,
    ) -> D::Buffer {
        let dev = &self.dev;
        let rows: usize = groups.iter().map(|g| g.1).sum();
        let ops = &self.ext.ops;
        if groups.len() == 1 {
            return ops.linear(dev, x, w, rows, k, n);
        }
        let mut out = dev.alloc_uninit_f32(rows * n);
        for &(r, m) in groups {
            let part = dev.slice_buffer(x, r * k, m * k);
            dev.write_into(&mut out, r * n, &ops.linear(dev, &part, w, m, k, n));
        }
        out
    }
}

/// One sequence's part of [`Model::forward_batch`]: tokens to run after what `cache` holds.
pub struct Segment<'a, B> {
    pub tokens: &'a [u32],
    pub cache: &'a mut Cache<B>,
    /// Logits for every token, not just the last.
    pub all: bool,
}

impl<D: ComputeDevice> Model<D> {
    /// The decoder layers over hidden states `x` (`[S, hidden]`): the rows of each
    /// `(cache, s)` part in turn, `s` of them at that cache's positions `cache.len..`.
    fn decode_layers(
        &self,
        x: &mut D::Buffer,
        parts: &mut [(&mut Cache<D::Buffer>, usize)],
        causal: bool,
        layer_limit: usize,
        bound: bool,
    ) -> Result<()> {
        let c = &self.cfg;
        let dev = &self.dev;
        let ops = &self.ext.ops;
        let (h, hd, nh, nkv) = (
            c.hidden_size,
            c.head_dim(),
            c.num_attention_heads,
            c.kv_heads(),
        );
        let (qd, kvd, ff) = (c.q_dim(), c.kv_dim(), c.intermediate_size);
        let eps = c.rms_norm_eps;
        let query_scale = c.query_rescale()?;
        let s: usize = parts.iter().map(|p| p.1).sum();
        let shared = parts[0].0.pool().clone();
        let mut pool = blocks::lock(&shared);
        // Each part's block table, and where its rows start in `x`.
        let mut seqs = Vec::with_capacity(parts.len());
        let mut row = 0;
        for (cache, n) in parts.iter_mut() {
            let pos = cache.len;
            let ids = blocks::prepare(dev, &mut pool, cache, pos, pos + *n);
            let table = dev.upload_u32(&ids);
            seqs.push((ids, table, pos, *n, row));
            row += *n;
        }
        let groups = row_groups(&seqs.iter().map(|q| q.3).collect::<Vec<_>>());
        // Leading one-token parts (batched decode) attend in one call per layer.
        let singles = if causal {
            seqs.iter().take_while(|q| q.3 == 1).count()
        } else {
            0
        };
        let fused = (singles >= 2).then(|| {
            let mut all = Vec::new();
            let mut offsets = Vec::with_capacity(singles);
            for q in &seqs[..singles] {
                offsets.push(all.len() as u32);
                all.extend_from_slice(&q.0);
            }
            let pos: Vec<usize> = seqs[..singles].iter().map(|q| q.2).collect();
            let pos32: Vec<u32> = pos.iter().map(|&p| p as u32).collect();
            (
                dev.upload_u32(&all),
                dev.upload_u32(&offsets),
                dev.upload_u32(&pos32),
                pos,
            )
        });
        let row_pages: Vec<tang_compute::Pages<D::Buffer>> = seqs
            [..if fused.is_some() { singles } else { 0 }]
            .iter()
            .map(|q| tang_compute::Pages {
                ids: &q.0,
                table: &q.1,
                block: BLOCK,
            })
            .collect();
        for (l, w) in self.layers.iter().take(layer_limit).enumerate() {
            let (kp, vp) = pool.layer(l);
            let a = ops.rms_norm(dev, x, &w.attn_norm, s, h, eps);
            let mut qkv = self.linear_rows(&a, &w.wqkv, &groups, h, qd + 2 * kvd);
            if let Some(b) = &w.bqkv {
                qkv = dev.bias_add(&qkv, b, s * (qd + 2 * kvd), qd + 2 * kvd);
            }
            let (cos, sin) = &self.ropes[w.rope];
            // Rows the fused call covers (all of them, or a leading run), then the rest one
            // part at a time.
            // Sink layers fuse with any window (their kernel takes one); the rest only without.
            let fuse = fused
                .as_ref()
                .filter(|_| w.window == 0 || w.sinks.is_some());
            let skip = if fuse.is_some() { singles } else { 0 };
            let one = seqs.len() - skip == 1 && skip == 0;
            let mut att = if one || skip == seqs.len() {
                None
            } else {
                Some(dev.alloc_uninit_f32(s * qd))
            };
            let mut single = None;
            if let Some((tables, offsets, positions, pos)) = fuse {
                let w3 = qd + 2 * kvd;
                let lead;
                let rows_qkv = if skip == seqs.len() {
                    &qkv
                } else {
                    lead = dev.slice_buffer(&qkv, 0, skip * w3);
                    &lead
                };
                let rows = tang_compute::PagedRows {
                    pages: &row_pages,
                    pos,
                    tables,
                    offsets,
                    positions,
                };
                let o = match &w.sinks {
                    Some(sinks) => dev.attention_paged_rows_sinks(
                        rows_qkv,
                        cos,
                        sin,
                        kp,
                        vp,
                        &rows,
                        (nh, nkv, hd),
                        w.window,
                        sinks,
                    ),
                    None => dev.attention_paged_rows(
                        rows_qkv,
                        w.q_norm.as_ref(),
                        w.k_norm.as_ref(),
                        cos,
                        sin,
                        kp,
                        vp,
                        &rows,
                        (nh, nkv, hd),
                        eps,
                        query_scale,
                    ),
                };
                match att.as_mut() {
                    Some(all) => dev.write_into(all, 0, &o),
                    None => single = Some(o),
                }
            }
            for (ids, table, pos, n, row) in &seqs[skip..] {
                let pages = tang_compute::Pages {
                    ids,
                    table,
                    block: BLOCK,
                };
                let part;
                let qkv = if one {
                    &qkv
                } else {
                    let w3 = qd + 2 * kvd;
                    part = dev.slice_buffer(&qkv, row * w3, n * w3);
                    &part
                };
                let mut q = dev.attention_prep_paged(
                    qkv,
                    w.q_norm.as_ref(),
                    w.k_norm.as_ref(),
                    cos,
                    sin,
                    kp,
                    vp,
                    &pages,
                    *n,
                    (nh, nkv, hd),
                    *pos,
                    eps,
                );
                if query_scale != 1.0 {
                    dev.scale_buffer(&mut q, query_scale);
                }
                let o = match &w.sinks {
                    Some(sinks) => dev.kv_attention_paged_sinks(
                        &q,
                        kp,
                        vp,
                        &pages,
                        *pos,
                        *n,
                        (nh, nkv, hd),
                        w.window,
                        causal,
                        sinks,
                    ),
                    None => ops.kv_attention_paged(
                        dev,
                        &crate::ext::PagedAttention {
                            q: &q,
                            k_pool: kp,
                            v_pool: vp,
                            pages: &pages,
                            start: *pos,
                            rows: *n,
                            heads: (nh, nkv, hd),
                            window: w.window,
                            causal,
                        },
                    ),
                };
                match att.as_mut() {
                    Some(all) => dev.write_into(all, row * qd, &o),
                    None => single = Some(o),
                }
            }
            let att = att.or(single).expect("a part");
            let mut o = self.linear_rows(&att, &w.wo, &groups, qd, h);
            if let Some(b) = &w.bo {
                o = dev.bias_add(&o, b, s * h, h);
            }
            if let Some(n) = &w.post_attn_norm {
                o = ops.rms_norm(dev, &o, n, s, h, eps);
            }
            *x = dev.add_tensors_buf(x, &o, s * h);

            let m = ops.rms_norm(dev, x, &w.mlp_norm, s, h, eps);
            let mut d = match &w.mlp {
                Mlp::Dense { w_gate_up, w_down } => {
                    let gu = self.linear_rows(&m, w_gate_up, &groups, h, 2 * ff);
                    let act = if c.is_gemma() {
                        dev.geglu_split(&gu, s, ff)
                    } else {
                        dev.swiglu_split(&gu, s, ff)
                    };
                    self.linear_rows(&act, w_down, &groups, ff, h)
                }
                Mlp::Moe(e) => self.moe(e, &m, &groups, s),
            };
            if let Some(n) = &w.post_mlp_norm {
                d = ops.rms_norm(dev, &d, n, s, h, eps);
            }
            *x = dev.add_tensors_buf(x, &d, s * h);
            // Large multimodal and conditioning pieces retain every temporary until their
            // command buffers are released. Bound that retention across deep models;
            // text prefill and decode keep the asynchronous path.
            if l % 4 == 3 {
                if bound && s > 128 {
                    dev.sync();
                } else {
                    dev.flush();
                }
            }
        }
        Ok(())
    }
}

impl<D: ComputeDevice> Model<D> {
    /// A routed MoE over normed hidden states `m` (`[s, hidden]`). Each row's experts run on
    /// that row alone, so a row's output doesn't depend on what else is in the batch.
    fn moe(
        &self,
        e: &Moe<D::Buffer>,
        m: &D::Buffer,
        groups: &[(usize, usize)],
        s: usize,
    ) -> D::Buffer {
        let dev = &self.dev;
        let h = self.cfg.hidden_size;
        let (k, ff) = (e.top_k, e.ff);
        let mut logits = self.linear_rows(m, &e.router, groups, h, e.experts);
        if let Some(b) = &e.router_bias {
            logits = dev.bias_add(&logits, b, s * e.experts, e.experts);
        }
        let (ids, wts) = dev.moe_route(&logits, s, e.experts, k, e.route);
        let gu = dev.moe_linear(
            m,
            &e.gate_up,
            e.fmt,
            e.experts,
            e.gate_up_bias.as_ref(),
            &ids,
            k,
            h,
            2 * ff,
            false,
        );
        let act = match e.clamp {
            Some((alpha, limit)) => dev.swiglu_clamped(&gu, s * k, ff, alpha, limit),
            None => dev.swiglu_split(&gu, s * k, ff),
        };
        let parts = dev.moe_linear(
            &act,
            &e.down,
            e.fmt,
            e.experts,
            e.down_bias.as_ref(),
            &ids,
            k,
            ff,
            h,
            true,
        );
        dev.moe_combine(&parts, &wts, s, k, h)
    }
}

/// Load layer `p`'s MoE: gpt-oss (`mlp.router`, `mlp.experts.*`, MXFP4 `_blocks`/`_scales` or
/// dense `[experts, in, out]` as transformers saves it).
fn load_moe<D: ComputeDevice>(
    dev: &D,
    w: &Weights,
    cfg: &Config,
    p: &str,
    dtype: Dtype,
) -> Result<Moe<D::Buffer>> {
    let (h, ff) = (cfg.hidden_size, cfg.intermediate_size);
    let experts = cfg.num_local_experts.unwrap_or(0);
    let top_k = cfg.num_experts_per_tok.unwrap_or(0);
    anyhow::ensure!(
        cfg.is_gpt_oss() && experts > 0 && top_k > 0 && top_k <= experts,
        "MoE layers are only supported for gpt-oss so far"
    );
    let x = format!("{p}.mlp.experts");
    // gpt-oss interleaves gate and up rows (gate 2j, up 2j + 1): gate rows first, then up.
    let split: Vec<usize> = (0..ff)
        .map(|j| 2 * j)
        .chain((0..ff).map(|j| 2 * j + 1))
        .collect();
    let per_expert = |v: &[f32], n: usize, perm: Option<&[usize]>| -> Vec<f32> {
        let Some(perm) = perm else { return v.to_vec() };
        let mut out = Vec::with_capacity(v.len());
        for e in 0..experts {
            for &r in perm {
                out.extend_from_slice(
                    &v[(e * n + r) * (v.len() / experts / n)..][..v.len() / experts / n],
                );
            }
        }
        out
    };
    let bias = |name: &str, n: usize, perm: Option<&[usize]>| -> Result<Option<D::Buffer>> {
        let name = format!("{x}.{name}");
        if !w.has(&name) {
            return Ok(None);
        }
        Ok(Some(dev.upload(&per_expert(&w.f32(&name)?.0, n, perm))))
    };
    let (fmt, gate_up, down) = if w.has(&format!("{x}.gate_up_proj_blocks")) {
        let mx = |name: &str, n: usize, k: usize, perm: Option<&[usize]>| -> Result<D::Buffer> {
            let (blocks, _) = w.bytes(&format!("{x}.{name}_blocks"))?;
            let (scales, _) = w.bytes(&format!("{x}.{name}_scales"))?;
            let mut packed = Mxfp4::pack(&blocks, &scales, experts, n, k);
            if let Some(perm) = perm {
                packed = Mxfp4::permute_rows(&packed, experts, n, k, perm);
            }
            Ok(dev.upload_bytes(&packed))
        };
        (
            ExpertFmt::Mxfp4,
            mx("gate_up_proj", 2 * ff, h, Some(&split))?,
            mx("down_proj", h, ff, None)?,
        )
    } else {
        // transformers' layout: `[experts, in, out]`; to `[experts, out, in]`.
        let dense = |name: &str, k: usize, n: usize, perm: Option<&[usize]>| -> Result<D::Buffer> {
            let (v, _) = w.f32(&format!("{x}.{name}"))?;
            let mut t = vec![0f32; v.len()];
            for e in 0..experts {
                for i in 0..k {
                    for o in 0..n {
                        t[(e * n + o) * k + i] = v[(e * k + i) * n + o];
                    }
                }
            }
            let t = per_expert(&t, n, perm);
            Ok(match dtype {
                Dtype::Bf16 => dev.upload_bf16(
                    &t.iter()
                        .map(|&f| tang_compute::f32_to_bf16(f))
                        .collect::<Vec<_>>(),
                ),
                _ => dev.upload(&t),
            })
        };
        let fmt = if dtype == Dtype::Bf16 && dev.bf16_storage() {
            ExpertFmt::Bf16
        } else {
            ExpertFmt::F32
        };
        (
            fmt,
            dense("gate_up_proj", h, 2 * ff, Some(&split))?,
            dense("down_proj", ff, h, None)?,
        )
    };
    Ok(Moe {
        router: dev.upload(&w.f32(&format!("{p}.mlp.router.weight"))?.0),
        router_bias: if w.has(&format!("{p}.mlp.router.bias")) {
            Some(dev.upload(&w.f32(&format!("{p}.mlp.router.bias"))?.0))
        } else {
            None
        },
        route: Router::TopKSoftmax,
        experts,
        top_k,
        ff,
        fmt,
        gate_up,
        gate_up_bias: bias("gate_up_proj_bias", 2 * ff, Some(&split))?,
        down,
        down_bias: bias("down_proj_bias", h, None)?,
        clamp: Some((
            cfg.swiglu_alpha.unwrap_or(1.702),
            cfg.swiglu_limit.unwrap_or(7.0),
        )),
    })
}

/// YaRN inverse frequencies and attention factor (transformers' `_compute_yarn_parameters`).
fn yarn(
    head_dim: usize,
    theta: f32,
    y: &crate::config::RopeScaling,
    max_pos: usize,
) -> (Vec<f64>, f64) {
    let (dim, base, factor) = (head_dim as f64, theta as f64, y.factor as f64);
    let orig = y.original_max_position_embeddings.unwrap_or(max_pos) as f64;
    let mscale = |s: f64| if s <= 1.0 { 1.0 } else { 0.1 * s.ln() + 1.0 };
    let attention = y.attention_factor.map_or(mscale(factor), |a| a as f64);
    let (fast, slow) = (
        y.beta_fast.unwrap_or(32.0) as f64,
        y.beta_slow.unwrap_or(1.0) as f64,
    );
    let corr =
        |rot: f64| dim * (orig / (rot * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base.ln());
    let (mut lo, mut hi) = (corr(fast), corr(slow));
    if y.truncate.unwrap_or(true) {
        (lo, hi) = (lo.floor(), hi.ceil());
    }
    let (lo, mut hi) = (lo.max(0.0), hi.min(dim - 1.0));
    if lo == hi {
        hi += 0.001;
    }
    let inv = (0..head_dim / 2)
        .map(|i| {
            let pos = base.powf(2.0 * i as f64 / dim);
            let ramp = ((i as f64 - lo) / (hi - lo)).clamp(0.0, 1.0);
            let extra = 1.0 - ramp;
            1.0 / (factor * pos) * (1.0 - extra) + 1.0 / pos * extra
        })
        .collect();
    (inv, attention)
}

/// cos/sin tables `[max_pos, inv.len()]` from inverse frequencies, both scaled by `mscale`.
fn rope_tables_inv(inv: &[f64], max_pos: usize, mscale: f64) -> (Vec<f32>, Vec<f32>) {
    let mut cos = Vec::with_capacity(max_pos * inv.len());
    let mut sin = Vec::with_capacity(max_pos * inv.len());
    for p in 0..max_pos {
        for &f in inv {
            let a = p as f64 * f;
            cos.push((a.cos() * mscale) as f32);
            sin.push((a.sin() * mscale) as f32);
        }
    }
    (cos, sin)
}

fn new_pool<B>(layers: usize, kv_dim: usize, bf16: bool, max_blocks: usize) -> Shared<B> {
    std::sync::Arc::new(std::sync::Mutex::new(blocks::Pool::new(
        layers, kv_dim, bf16, max_blocks,
    )))
}

/// Round to the nearest bfloat16, as Gemma's reference does for its embedding scale.
fn bf16_round(x: f32) -> f32 {
    let b = x.to_bits();
    f32::from_bits((b + 0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
}

/// cos/sin for half-split RoPE, `[max_pos, head_dim / 2]`, positions divided by `scale`
/// (linear RoPE scaling).
fn rope_tables(head_dim: usize, max_pos: usize, theta: f32, scale: f32) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = Vec::with_capacity(max_pos * half);
    let mut sin = Vec::with_capacity(max_pos * half);
    for p in 0..max_pos {
        for i in 0..half {
            let inv = (theta as f64).powf(-(2.0 * i as f64) / head_dim as f64);
            let a = p as f64 / scale as f64 * inv;
            cos.push(a.cos() as f32);
            sin.push(a.sin() as f32);
        }
    }
    (cos, sin)
}

#[cfg(test)]
mod conditioning_tests {
    use super::*;
    use tang_compute::CpuDevice;

    fn tiny_model() -> Model<CpuDevice> {
        let dev = CpuDevice::new();
        let cfg = Config::from_json(br#"{"model_type":"qwen3","hidden_size":4,"intermediate_size":4,"num_hidden_layers":2,"num_attention_heads":1,"num_key_value_heads":1,"vocab_size":2,"rms_norm_eps":0.000001,"rope_theta":10000}"#).unwrap();
        let layer = |scale: f32| Layer {
            attn_norm: dev.upload(&[1.; 4]),
            wqkv: dev.upload(&[0.; 48]),
            wo: dev.upload(&[0.; 16]),
            q_norm: None,
            k_norm: None,
            mlp_norm: dev.upload(&[1.; 4]),
            mlp: Mlp::Dense {
                w_gate_up: dev.upload(&[
                    1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0.,
                    0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
                ]),
                w_down: dev.upload(&[
                    scale, 0., 0., 0., 0., scale, 0., 0., 0., 0., scale, 0., 0., 0., 0., scale,
                ]),
            },
            bqkv: None,
            bo: None,
            sinks: None,
            post_attn_norm: None,
            post_mlp_norm: None,
            window: 0,
            rope: 0,
        };
        let layers = vec![layer(1.), layer(2.)];
        let (cos, sin) = rope_tables(4, 2048, 10000., 1.);
        Model {
            cfg,
            embed: dev.upload(&[1., 2., 3., 4., 4., 3., 2., 1.]),
            layers,
            norm: dev.upload(&[10.; 4]),
            lm_head: Some(dev.upload(&[0.; 8])),
            ropes: vec![(dev.upload(&cos), dev.upload(&sin))],
            embed_scale: None,
            vision: None,
            max_ctx: 2048,
            pool: new_pool(2, 4, false, usize::MAX),
            ext: Default::default(),
            dev,
        }
    }

    /// `tiny_model` with every projection filled in, so attention and the head matter.
    fn mixed_model() -> Model<CpuDevice> {
        let mut m = tiny_model();
        let dev = CpuDevice::new();
        let mut seed = 1u32;
        let mut fill = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
                })
                .collect()
        };
        for l in &mut m.layers {
            l.wqkv = dev.upload(&fill(48));
            l.wo = dev.upload(&fill(16));
        }
        m.lm_head = Some(dev.upload(&fill(8)));
        m
    }

    #[test]
    fn a_batched_forward_matches_each_segment_alone() {
        let m = mixed_model();
        // One-token segments among longer ones: they're attended for together.
        let prefixes: [&[u32]; 5] = [&[0, 1, 1], &[1], &[1, 0, 0, 1, 0], &[0, 0], &[1, 1, 1]];
        let feeds: [&[u32]; 5] = [&[1, 0], &[0], &[1, 1, 0], &[1], &[0]];
        let all = [true, false, false, false, true];
        let mut alone = Vec::new();
        for i in 0..5 {
            let mut c = m.new_cache();
            m.forward(prefixes[i], &mut c, false).unwrap();
            alone.push((
                m.forward(feeds[i], &mut c, all[i]).unwrap(),
                c.tokens.clone(),
            ));
        }
        let mut caches: Vec<_> = prefixes
            .iter()
            .map(|p| {
                let mut c = m.new_cache();
                m.forward(p, &mut c, false).unwrap();
                c
            })
            .collect();
        let mut segs: Vec<Segment<_>> = caches
            .iter_mut()
            .enumerate()
            .map(|(i, cache)| Segment {
                tokens: feeds[i],
                cache,
                all: all[i],
            })
            .collect();
        let got = m.forward_batch(&mut segs).unwrap();
        drop(segs);
        for i in 0..5 {
            assert_eq!(got[i].len(), alone[i].0.len());
            for (a, b) in got[i].iter().zip(&alone[i].0) {
                assert!((a - b).abs() < 1e-5, "segment {i}: {a} vs {b}");
            }
            assert_eq!(caches[i].tokens, alone[i].1);
            assert_eq!(caches[i].len, alone[i].1.len());
        }
    }

    #[test]
    fn explicit_attention_scalar_matches_rescaled_query_projection() {
        let make = |scalar, projection_scale: f32| {
            let mut model = tiny_model();
            model.cfg.query_pre_attn_scalar = scalar;
            for layer in &mut model.layers {
                let mut qkv = vec![0.; 48];
                let mut identity = vec![0.; 16];
                for i in 0..4 {
                    qkv[i * 4 + i] = projection_scale;
                    qkv[16 + i * 4 + i] = 1.;
                    qkv[32 + i * 4 + i] = 1.;
                    identity[i * 4 + i] = 1.;
                }
                layer.wqkv = model.dev.upload(&qkv);
                layer.wo = model.dev.upload(&identity);
            }
            model.lm_head = Some(model.dev.upload(&[1., 0., 0., 0., 0., 1., 0., 0.]));
            let mut cache = model.new_cache();
            model.forward(&[0, 1, 0], &mut cache, true).unwrap()
        };
        let actual = make(Some(8.), 1.);
        let reference = make(None, (4f32 / 8.).sqrt());
        let baseline = make(None, 1.);
        assert!(actual
            .iter()
            .zip(&reference)
            .all(|(a, b)| (a - b).abs() < 1e-5));
        assert!(actual
            .iter()
            .zip(&baseline)
            .any(|(a, b)| (a - b).abs() > 1e-3));
        let mut cfg = tiny_model().cfg;
        for invalid in [0., -1., f32::NAN, f32::INFINITY] {
            cfg.query_pre_attn_scalar = Some(invalid);
            assert!(cfg.query_rescale().is_err());
        }
        cfg.head_dim = Some(128);
        cfg.query_pre_attn_scalar = Some(168.);
        assert!(
            (cfg.query_rescale().unwrap() / 128f32.sqrt() - 168f32.sqrt().recip()).abs() < 1e-7
        );
    }

    #[test]
    fn penultimate_is_pre_final_layer_without_norm_or_head() {
        let model = tiny_model();
        let result = model
            .dev
            .download(&model.encode_penultimate(&[0, 1]).unwrap());
        for (row, original) in [[1f32, 2., 3., 4.], [4., 3., 2., 1.]].iter().enumerate() {
            let denom = (original.iter().map(|x| x * x).sum::<f32>() / 4. + 0.000001).sqrt();
            for i in 0..4 {
                let x = original[i] / denom;
                let expected = original[i] + x * x / (1. + (-x).exp());
                assert!((result[row * 4 + i] - expected).abs() < 1e-5);
            }
        }
        let mut cache = model.new_cache();
        assert_eq!(
            model.forward(&[0, 1], &mut cache, true).unwrap(),
            vec![0.; 4]
        );
        assert!(model.encode_penultimate(&[]).is_err());
        assert!(model.encode_penultimate(&[0; 2049]).is_err());
    }

    #[test]
    fn multimodal_last_logits_match_all_logits_across_text_chunks() {
        let mut model = tiny_model();
        model.cfg.vocab_size = 3;
        model.cfg.image_token = Some(2);
        model.embed = model
            .dev
            .upload(&[1., 2., 3., 4., 4., 3., 2., 1., 0., 0., 0., 0.]);
        model.lm_head = Some(
            model
                .dev
                .upload(&[1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.]),
        );
        model.vision = Some(crate::vision::Vision::encoded_fixture(&model.dev, 4));
        let mut tokens: Vec<u32> = (0..600).map(|i| i % 2).collect();
        tokens.splice(590..590, [2; 4]);
        tokens.splice(50..50, [2; 4]);
        let features = [model.dev.upload(&[6.; 16]), model.dev.upload(&[9.; 16])];
        let mut full_cache = model.new_cache();
        let all = model
            .forward_images(&tokens, &features, &mut full_cache, true)
            .unwrap();
        let mut last_cache = model.new_cache();
        let last = model
            .forward_images(&tokens, &features, &mut last_cache, false)
            .unwrap();
        assert_eq!(all.len(), tokens.len() * 3);
        assert_eq!(last.len(), 3);
        assert!(last
            .iter()
            .zip(&all[all.len() - 3..])
            .all(|(a, b)| (a - b).abs() < 1e-5));
        assert_eq!(full_cache.tokens, tokens);
        assert_eq!(last_cache.tokens, tokens);
    }
}
