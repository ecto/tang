//! Qwen3-family decoder (also Llama-shaped models: q/k norm is used only when present).

use crate::config::Config;
use crate::weights::Weights;
use anyhow::Result;
use std::path::Path;
use tang_compute::ComputeDevice;

struct Layer<B> {
    attn_norm: B,
    wq: B,
    wk: B,
    wv: B,
    wo: B,
    q_norm: Option<B>,
    k_norm: Option<B>,
    mlp_norm: B,
    w_gate: B,
    w_up: B,
    w_down: B,
}

pub struct Model<D: ComputeDevice> {
    pub cfg: Config,
    pub dev: D,
    embed: D::Buffer,
    layers: Vec<Layer<D::Buffer>>,
    norm: D::Buffer,
    lm_head: Option<D::Buffer>,
    cos: D::Buffer,
    sin: D::Buffer,
    max_ctx: usize,
}

/// Per-sequence attention state: K and V for every layer, `[max_ctx, kv_dim]`, on device.
pub struct Cache<B> {
    k: Vec<B>,
    v: Vec<B>,
    /// Tokens already in the cache.
    pub len: usize,
    pub tokens: Vec<u32>,
}

impl<D: ComputeDevice> Model<D> {
    pub fn load(dev: D, dir: &Path, max_ctx: usize) -> Result<Self> {
        let cfg: Config = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        let w = Weights::open(dir)?;
        let up = |name: &str| -> Result<D::Buffer> { Ok(dev.upload(&w.f32(name)?.0)) };
        let opt = |name: &str| -> Result<Option<D::Buffer>> {
            if w.has(name) { up(name).map(Some) } else { Ok(None) }
        };
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{l}");
            layers.push(Layer {
                attn_norm: up(&format!("{p}.input_layernorm.weight"))?,
                wq: up(&format!("{p}.self_attn.q_proj.weight"))?,
                wk: up(&format!("{p}.self_attn.k_proj.weight"))?,
                wv: up(&format!("{p}.self_attn.v_proj.weight"))?,
                wo: up(&format!("{p}.self_attn.o_proj.weight"))?,
                q_norm: opt(&format!("{p}.self_attn.q_norm.weight"))?,
                k_norm: opt(&format!("{p}.self_attn.k_norm.weight"))?,
                mlp_norm: up(&format!("{p}.post_attention_layernorm.weight"))?,
                w_gate: up(&format!("{p}.mlp.gate_proj.weight"))?,
                w_up: up(&format!("{p}.mlp.up_proj.weight"))?,
                w_down: up(&format!("{p}.mlp.down_proj.weight"))?,
            });
        }
        let embed = up("model.embed_tokens.weight")?;
        let lm_head = if cfg.tie_word_embeddings { None } else { opt("lm_head.weight")? };
        let norm = up("model.norm.weight")?;

        let max_ctx = max_ctx.min(cfg.max_position_embeddings);
        let (cos, sin) = rope_tables(cfg.head_dim(), max_ctx, cfg.rope_theta);
        let (cos, sin) = (dev.upload(&cos), dev.upload(&sin));
        Ok(Self { cfg, dev, embed, layers, norm, lm_head, cos, sin, max_ctx })
    }

    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    pub fn new_cache(&self) -> Cache<D::Buffer> {
        let n = self.max_ctx * self.cfg.kv_dim();
        Cache {
            k: (0..self.layers.len()).map(|_| self.dev.alloc(n)).collect(),
            v: (0..self.layers.len()).map(|_| self.dev.alloc(n)).collect(),
            len: 0,
            tokens: Vec::new(),
        }
    }

    /// Run `tokens` through the model after what's already in `cache`, append them to it, and
    /// return logits: for the last token only, or for every token if `all` is set.
    pub fn forward(&self, tokens: &[u32], cache: &mut Cache<D::Buffer>, all: bool) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let dev = &self.dev;
        let (s, pos) = (tokens.len(), cache.len);
        anyhow::ensure!(s > 0, "no tokens");
        anyhow::ensure!(pos + s <= self.max_ctx, "context full ({} tokens)", self.max_ctx);
        let (h, hd, nh, nkv) = (c.hidden_size, c.head_dim(), c.num_attention_heads, c.kv_heads());
        let (qd, kvd, ff) = (c.q_dim(), c.kv_dim(), c.intermediate_size);
        let eps = c.rms_norm_eps;

        let ids = dev.upload_u32(tokens);
        let mut x = dev.embedding(&self.embed, &ids, s, h);
        for (l, w) in self.layers.iter().enumerate() {
            let a = dev.rms_norm(&x, &w.attn_norm, s, h, eps);
            let mut q = dev.linear(&a, &w.wq, s, h, qd);
            let mut k = dev.linear(&a, &w.wk, s, h, kvd);
            let v = dev.linear(&a, &w.wv, s, h, kvd);
            if let Some(n) = &w.q_norm {
                q = dev.rms_norm(&q, n, s * nh, hd, eps);
            }
            if let Some(n) = &w.k_norm {
                k = dev.rms_norm(&k, n, s * nkv, hd, eps);
            }
            let q = dev.rope_half_cached(&q, &self.cos, &self.sin, s, nh, hd, pos);
            let k = dev.rope_half_cached(&k, &self.cos, &self.sin, s, nkv, hd, pos);
            dev.write_into(&mut cache.k[l], pos * kvd, &k);
            dev.write_into(&mut cache.v[l], pos * kvd, &v);
            let att = dev.kv_attention(&q, &cache.k[l], &cache.v[l], pos, s, nh, nkv, hd);
            let o = dev.linear(&att, &w.wo, s, qd, h);
            x = dev.add_tensors_buf(&x, &o, s * h);

            let m = dev.rms_norm(&x, &w.mlp_norm, s, h, eps);
            let g = dev.linear(&m, &w.w_gate, s, h, ff);
            let u = dev.linear(&m, &w.w_up, s, h, ff);
            let act = dev.swiglu_fused_buf(&g, &u, s * ff);
            let d = dev.linear(&act, &w.w_down, s, ff, h);
            x = dev.add_tensors_buf(&x, &d, s * h);
        }
        let (rows, x) = if all { (s, x) } else { (1, dev.slice_buffer(&x, (s - 1) * h, h)) };
        let x = dev.rms_norm(&x, &self.norm, rows, h, eps);
        let head = self.lm_head.as_ref().unwrap_or(&self.embed);
        let logits = dev.linear(&x, head, rows, h, c.vocab_size);
        let out = dev.download(&logits);
        cache.len += s;
        cache.tokens.extend_from_slice(tokens);
        Ok(out)
    }
}

/// cos/sin for half-split RoPE, `[max_pos, head_dim / 2]`.
fn rope_tables(head_dim: usize, max_pos: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = Vec::with_capacity(max_pos * half);
    let mut sin = Vec::with_capacity(max_pos * half);
    for p in 0..max_pos {
        for i in 0..half {
            let inv = (theta as f64).powf(-(2.0 * i as f64) / head_dim as f64);
            let a = p as f64 * inv;
            cos.push(a.cos() as f32);
            sin.push(a.sin() as f32);
        }
    }
    (cos, sin)
}
