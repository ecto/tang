//! Qwen3-family decoder (also Llama-shaped models: q/k norm is used only when present), and
//! Gemma 3: GeGLU, `(1 + w)` RMSNorms around both attention and MLP, a scaled embedding, and
//! local sliding-window layers (own RoPE base) between global ones (scaled RoPE).

use crate::config::Config;
use crate::weights::Weights;
use anyhow::Result;
use std::path::Path;
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
    /// Gate and up projections stacked: `[2 * ff, hidden]`.
    w_gate_up: B,
    w_down: B,
    /// Gemma: norms on the attention and MLP outputs, before the residual adds.
    post_attn_norm: Option<B>,
    post_mlp_norm: Option<B>,
    /// Sliding window (0: global attention), and which RoPE tables (0 global, 1 local).
    window: usize,
    rope: usize,
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
    /// New caches keep K and V in bf16 (half the memory of f32; attention still accumulates in
    /// f32). Off where the device has no bf16 storage, or with `TANG_KV_F32=1`.
    kv_bf16: bool,
}

/// Per-sequence attention state: K and V for every layer, `[cap, kv_dim]`, on device (bf16, or
/// f32; see [`Model::set_kv_f32`]). It starts small and doubles as the sequence grows, up to
/// the context window.
/// Truncating keeps a prefix (later positions are simply overwritten), which is how a new
/// request reuses the part of the conversation it shares with the last one.
pub struct Cache<B> {
    k: Vec<B>,
    v: Vec<B>,
    /// Positions allocated (a multiple of 32).
    cap: usize,
    /// Device memory per position, all layers' K and V.
    row_bytes: usize,
    /// K and V are bf16 (else f32).
    bf16: bool,
    /// Tokens already in the cache.
    pub len: usize,
    pub tokens: Vec<u32>,
}

/// Positions a new cache starts with.
const FIRST_ROWS: usize = 4096;

impl<B> Cache<B> {
    /// Device memory it holds now.
    pub fn bytes(&self) -> usize {
        self.cap * self.row_bytes
    }

    pub fn truncate(&mut self, len: usize) {
        self.len = self.len.min(len);
        self.tokens.truncate(self.len);
    }
}

impl<D: ComputeDevice> Model<D> {
    /// Load a checkpoint directory. Matrices are kept in `dtype` on device; norms stay f32.
    pub fn load(dev: D, dir: &Path, max_ctx: usize, dtype: Dtype) -> Result<Self> {
        let cfg = Config::from_json(&std::fs::read(dir.join("config.json"))?)?;
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
                w_gate_up: up_cat(&["gate", "up"].map(|x| format!("{p}.mlp.{x}_proj.weight")))?,
                w_down: up(&format!("{p}.mlp.down_proj.weight"))?,
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
        let (cos, sin) = rope_tables(hd, max_ctx, cfg.rope_theta, scale);
        ropes.push((dev.upload(&cos), dev.upload(&sin)));
        if gemma {
            let base = cfg.rope_local_base_freq.unwrap_or(10_000.0);
            let (cos, sin) = rope_tables(hd, max_ctx, base, 1.0);
            ropes.push((dev.upload(&cos), dev.upload(&sin)));
        }
        let embed_scale = gemma.then(|| bf16_round((cfg.hidden_size as f32).sqrt()));
        let dev_bf16 = dev.bf16_storage();
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
        Ok(Self {
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
            kv_bf16: dev_bf16 && !std::env::var("TANG_KV_F32").is_ok_and(|v| v == "1"),
        })
    }

    /// Keep caches made from now on in f32 instead of bf16 (for comparing the two).
    pub fn set_kv_f32(&mut self, f32: bool) {
        self.kv_bf16 = !f32 && self.dev.bf16_storage();
    }

    /// Whether new caches hold bf16 (else f32).
    pub fn kv_bf16(&self) -> bool {
        self.kv_bf16
    }

    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    /// Device memory a [`Cache`] of `rows` positions takes (K and V for every layer).
    pub fn cache_bytes(&self, rows: usize) -> usize {
        rows.min(self.max_ctx).next_multiple_of(32) * self.row_bytes()
    }

    /// Device memory per position: every layer's K and V, 2 bytes an element in bf16, else 4.
    pub fn row_bytes(&self) -> usize {
        self.row_elems() * if self.kv_bf16 { 2 } else { 4 }
    }

    /// One layer's K or V buffer of `n` elements, bf16 or f32.
    fn kv_alloc(&self, bf16: bool, n: usize) -> D::Buffer {
        if bf16 {
            self.dev.alloc_bf16(n)
        } else {
            self.dev.alloc(n)
        }
    }

    pub fn new_cache(&self) -> Cache<D::Buffer> {
        // Rounded up so tiled attention can read whole 32-row blocks.
        let cap = FIRST_ROWS.min(self.max_ctx).next_multiple_of(32);
        let n = cap * self.cfg.kv_dim();
        Cache {
            k: (0..self.layers.len())
                .map(|_| self.kv_alloc(self.kv_bf16, n))
                .collect(),
            v: (0..self.layers.len())
                .map(|_| self.kv_alloc(self.kv_bf16, n))
                .collect(),
            bf16: self.kv_bf16,
            cap,
            row_bytes: self.row_bytes(),
            len: 0,
            tokens: Vec::new(),
        }
    }

    /// Elements per position in [`read_rows`](Self::read_rows): every layer's K and V.
    pub fn row_elems(&self) -> usize {
        2 * self.layers.len() * self.cfg.kv_dim()
    }

    /// Positions `from..to` of `cache` as bf16 bits (exact for a bf16 cache, rounded from an f32
    /// one), position-major: for each, every layer's K row then V row.
    pub fn read_rows(&self, cache: &Cache<D::Buffer>, from: usize, to: usize) -> Vec<u16> {
        let kvd = self.cfg.kv_dim();
        let n = to - from;
        let mut out = vec![0u16; n * self.row_elems()];
        if n == 0 {
            return out;
        }
        let bufs = cache.k.iter().zip(&cache.v).flat_map(|(k, v)| [k, v]);
        for (j, buf) in bufs.enumerate() {
            let part = self
                .dev
                .download_bf16(&self.dev.slice_buffer(buf, from * kvd, n * kvd));
            for p in 0..n {
                let at = (p * self.layers.len() * 2 + j) * kvd;
                out[at..at + kvd].copy_from_slice(&part[p * kvd..(p + 1) * kvd]);
            }
        }
        out
    }

    /// Append positions to `cache`: `tokens`, with their rows as [`read_rows`](Self::read_rows)
    /// gives them.
    pub fn write_rows(&self, cache: &mut Cache<D::Buffer>, tokens: &[u32], rows: &[u16]) {
        let kvd = self.cfg.kv_dim();
        let n = tokens.len();
        if n == 0 {
            return;
        }
        self.reserve(cache, cache.len + n);
        let at = cache.len * kvd;
        let layers = self.layers.len();
        let bufs = cache
            .k
            .iter_mut()
            .zip(cache.v.iter_mut())
            .flat_map(|(k, v)| [k, v]);
        for (j, buf) in bufs.enumerate() {
            let mut part = Vec::with_capacity(n * kvd);
            for p in 0..n {
                let from = (p * layers * 2 + j) * kvd;
                part.extend_from_slice(&rows[from..from + kvd]);
            }
            // Widened on the way in if the cache is f32.
            self.dev.write_into(buf, at, &self.dev.upload_bf16(&part));
        }
        cache.len += n;
        cache.tokens.extend_from_slice(tokens);
    }

    /// Make room in `cache` for `rows` positions: double it (at least to `rows`, at most to
    /// the context window) and copy what's there.
    fn reserve(&self, cache: &mut Cache<D::Buffer>, rows: usize) {
        if rows <= cache.cap {
            return;
        }
        let cap = rows
            .max(cache.cap * 2)
            .min(self.max_ctx)
            .max(rows)
            .next_multiple_of(32);
        let kvd = self.cfg.kv_dim();
        let used = cache.len * kvd;
        for buf in cache.k.iter_mut().chain(cache.v.iter_mut()) {
            let mut grown = self.kv_alloc(cache.bf16, cap * kvd);
            if used > 0 {
                let old = self.dev.slice_buffer(buf, 0, used);
                self.dev.write_into(&mut grown, 0, &old);
            }
            *buf = grown;
        }
        cache.cap = cap;
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
        self.forward_hidden(x, tokens, cache, all, true)
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
                self.forward_hidden(seg, &tokens[start..start + len], cache, all || last, !img)?;
            if all {
                out.extend(logits);
            } else if last {
                out = logits;
            }
        }
        if !all {
            let v = self.cfg.vocab_size;
            out = out.split_off(out.len() - v);
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

    /// Like `forward`, from hidden states instead of token ids (so image features can stand
    /// in for placeholder tokens). `tokens` are what the cache records for these positions.
    pub fn forward_hidden(
        &self,
        mut x: D::Buffer,
        tokens: &[u32],
        cache: &mut Cache<D::Buffer>,
        all: bool,
        causal: bool,
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
        self.decode_layers(&mut x, cache, s, pos, causal)?;
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
    /// The decoder layers over hidden states `x` (`[s, hidden]`) at positions `pos..`.
    fn decode_layers(
        &self,
        x: &mut D::Buffer,
        cache: &mut Cache<D::Buffer>,
        s: usize,
        pos: usize,
        causal: bool,
    ) -> Result<()> {
        let c = &self.cfg;
        let dev = &self.dev;
        let (h, hd, nh, nkv) = (
            c.hidden_size,
            c.head_dim(),
            c.num_attention_heads,
            c.kv_heads(),
        );
        let (qd, kvd, ff) = (c.q_dim(), c.kv_dim(), c.intermediate_size);
        let eps = c.rms_norm_eps;
        self.reserve(cache, pos + s);
        for (l, w) in self.layers.iter().enumerate() {
            let a = dev.rms_norm(x, &w.attn_norm, s, h, eps);
            let qkv = dev.linear(&a, &w.wqkv, s, h, qd + 2 * kvd);
            let (cos, sin) = &self.ropes[w.rope];
            let q = dev.attention_prep(
                &qkv,
                w.q_norm.as_ref(),
                w.k_norm.as_ref(),
                cos,
                sin,
                &mut cache.k[l],
                &mut cache.v[l],
                s,
                (nh, nkv, hd),
                pos,
                eps,
            );
            let att = dev.kv_attention_window(
                &q,
                &cache.k[l],
                &cache.v[l],
                pos,
                s,
                (nh, nkv, hd),
                w.window,
                causal,
            );
            let mut o = dev.linear(&att, &w.wo, s, qd, h);
            if let Some(n) = &w.post_attn_norm {
                o = dev.rms_norm(&o, n, s, h, eps);
            }
            *x = dev.add_tensors_buf(x, &o, s * h);

            let m = dev.rms_norm(x, &w.mlp_norm, s, h, eps);
            let gu = dev.linear(&m, &w.w_gate_up, s, h, 2 * ff);
            let act = if c.is_gemma() {
                dev.geglu_split(&gu, s, ff)
            } else {
                dev.swiglu_split(&gu, s, ff)
            };
            let mut d = dev.linear(&act, &w.w_down, s, ff, h);
            if let Some(n) = &w.post_mlp_norm {
                d = dev.rms_norm(&d, n, s, h, eps);
            }
            *x = dev.add_tensors_buf(x, &d, s * h);
            // Let the GPU start on what's encoded so far.
            if l % 4 == 3 {
                dev.flush();
            }
        }
        Ok(())
    }
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
