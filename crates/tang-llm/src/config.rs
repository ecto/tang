//! `config.json` for Qwen3-family checkpoints (Qwen3, and Llama-shaped models without q/k norm)
//! and Gemma 3 (whose multimodal checkpoints nest the language model under `text_config`).

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    #[serde(default)]
    pub head_dim: Option<usize>,
    pub vocab_size: usize,
    #[serde(default = "default_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub eos_token_id: Option<EosIds>,
    /// MLX-quantized checkpoints: `{"group_size": 64, "bits": 4}`.
    #[serde(default)]
    pub quantization: Option<Quantization>,

    // Gemma 3.
    /// Attention scale is `query_pre_attn_scalar^-0.5` (Gemma); default `head_dim^-0.5`.
    #[serde(default)]
    pub query_pre_attn_scalar: Option<f32>,
    /// Local layers attend to this many most recent positions.
    #[serde(default)]
    pub sliding_window: Option<usize>,
    /// Every `pattern`-th layer is global; the rest are local (sliding-window).
    #[serde(default)]
    pub sliding_window_pattern: Option<usize>,
    /// RoPE base for local layers (global layers use `rope_theta`).
    #[serde(default)]
    pub rope_local_base_freq: Option<f32>,
    /// Linear RoPE scaling for global layers: positions divided by `factor`.
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    /// The vision tower, for multimodal checkpoints.
    #[serde(skip)]
    pub vision: Option<VisionConfig>,
    /// Image placeholder token and tokens per image (multimodal checkpoints).
    #[serde(skip)]
    pub image_token: Option<u32>,
    #[serde(skip)]
    pub mm_tokens_per_image: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    #[serde(default = "one")]
    pub factor: f32,
    #[serde(default)]
    pub rope_type: Option<String>,
}

fn one() -> f32 {
    1.0
}

/// SigLIP vision tower (Gemma 3).
#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub image_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub patch_size: usize,
    #[serde(default = "vision_eps")]
    pub layer_norm_eps: f32,
}

fn vision_eps() -> f32 {
    1e-6
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: usize,
    pub bits: u32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EosIds {
    One(u32),
    Many(Vec<u32>),
}

fn default_eps() -> f32 {
    1e-6
}
fn default_theta() -> f32 {
    10_000.0
}
fn default_max_pos() -> usize {
    32_768
}

impl Config {
    /// Parse `config.json`. Gemma 3 multimodal checkpoints keep the language model in
    /// `text_config`, with transformers' Gemma3TextConfig defaults for what's left out.
    pub fn from_json(bytes: &[u8]) -> anyhow::Result<Self> {
        let top: serde_json::Value = serde_json::from_slice(bytes)?;
        let Some(text) = top.get("text_config").filter(|t| t.is_object()) else {
            return Ok(serde_json::from_value(top)?);
        };
        let mut t = serde_json::json!({
            "model_type": "gemma3_text",
            "hidden_size": 2304,
            "intermediate_size": 9216,
            "num_hidden_layers": 26,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "vocab_size": 262208,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1_000_000.0,
            "rope_local_base_freq": 10_000.0,
            "query_pre_attn_scalar": 256.0,
            "sliding_window": 4096,
            "sliding_window_pattern": 6,
            "max_position_embeddings": 131072,
        });
        for (k, v) in text.as_object().unwrap() {
            t[k] = v.clone();
        }
        for k in ["quantization", "eos_token_id"] {
            if let Some(v) = top.get(k) {
                t[k] = v.clone();
            }
        }
        let mut cfg: Config = serde_json::from_value(t)?;
        if let Some(v) = top.get("vision_config").filter(|v| v.is_object()) {
            cfg.vision = Some(serde_json::from_value(v.clone())?);
            cfg.image_token = top["image_token_index"].as_u64().map(|i| i as u32);
            cfg.mm_tokens_per_image = top["mm_tokens_per_image"].as_u64().unwrap_or(256) as usize;
        }
        Ok(cfg)
    }

    pub fn is_gemma(&self) -> bool {
        self.model_type.starts_with("gemma")
    }

    /// Sliding window for layer `l` (0 = global attention).
    pub fn window(&self, l: usize) -> usize {
        match (self.sliding_window, self.sliding_window_pattern) {
            (Some(w), Some(p)) if p > 1 && (l + 1) % p != 0 => w,
            _ => 0,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
    pub fn kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }
    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim()
    }
    pub fn kv_dim(&self) -> usize {
        self.kv_heads() * self.head_dim()
    }
    pub fn eos(&self) -> Vec<u32> {
        match &self.eos_token_id {
            Some(EosIds::One(id)) => vec![*id],
            Some(EosIds::Many(ids)) => ids.clone(),
            None => Vec::new(),
        }
    }
}
