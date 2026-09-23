//! `config.json` for Qwen3-family checkpoints (Qwen3, and Llama-shaped models without q/k norm).

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
