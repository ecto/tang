//! SmolLM2-135M inference demo.
//!
//! Downloads SmolLM2-135M from HuggingFace and generates text.
//!
//! ```sh
//! cargo run --example smollm -p tang-train --release
//! ```

use std::collections::HashMap;
use std::time::Instant;

use tang_tensor::{Shape, Tensor};
use tang_train::{Linear, Parameter, RotaryEmbedding};

use tang_infer::{KVCache, SamplingConfig, Sampler};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const VOCAB: usize = 49152;
const HIDDEN: usize = 576;
const LAYERS: usize = 30;
const HEADS: usize = 9;
const KV_HEADS: usize = 3;
const INTERMEDIATE: usize = 1536;
const HEAD_DIM: usize = 64;
const ROPE_THETA: f64 = 10000.0;
const EPS: f64 = 1e-5;
const MAX_SEQ: usize = 2048;

const KV_DIM: usize = KV_HEADS * HEAD_DIM; // 192
const HEADS_PER_KV: usize = HEADS / KV_HEADS; // 3

// ---------------------------------------------------------------------------
// RMSNorm (inline function, configurable eps)
// ---------------------------------------------------------------------------

fn rms_norm(x: &[f64], weight: &[f64], eps: f64) -> Vec<f64> {
    let n = x.len();
    let ss: f64 = x.iter().map(|v| v * v).sum::<f64>() / n as f64;
    let scale = 1.0 / (ss + eps).sqrt();
    x.iter()
        .zip(weight.iter())
        .map(|(&xi, &wi)| xi * scale * wi)
        .collect()
}

// ---------------------------------------------------------------------------
// LlamaAttention
// ---------------------------------------------------------------------------

struct LlamaAttention {
    q_proj: Linear<f64>,
    k_proj: Linear<f64>,
    v_proj: Linear<f64>,
    o_proj: Linear<f64>,
    rope: RotaryEmbedding<f64>,
}

impl LlamaAttention {
    fn new() -> Self {
        Self {
            q_proj: Linear::new(HIDDEN, HIDDEN, 0),
            k_proj: Linear::new(HIDDEN, KV_DIM, 0),
            v_proj: Linear::new(HIDDEN, KV_DIM, 0),
            o_proj: Linear::new(HIDDEN, HIDDEN, 0),
            rope: RotaryEmbedding::with_base(HEAD_DIM, MAX_SEQ, ROPE_THETA),
        }
    }

    fn forward(
        &self,
        x: &Tensor<f64>,      // [seq_len, hidden]
        layer_idx: usize,
        cache: &mut KVCache<f64>,
        pos_offset: usize,
    ) -> Tensor<f64> {
        let seq_len = x.shape()[0];

        // Project: x @ W^T
        let q_full = matmul_weight(x, &self.q_proj.weight.data); // [seq, hidden]
        let k_full = matmul_weight(x, &self.k_proj.weight.data); // [seq, kv_dim]
        let v_full = matmul_weight(x, &self.v_proj.weight.data); // [seq, kv_dim]

        // Apply RoPE per head
        let q_roped = apply_rope_heads(&q_full, &self.rope, HEADS, HEAD_DIM, pos_offset);
        let k_roped = apply_rope_heads(&k_full, &self.rope, KV_HEADS, HEAD_DIM, pos_offset);

        // Append to KV cache
        cache.append(layer_idx, &k_roped, &v_full);

        // Get full cached K, V
        let cached_k = cache.get_keys(layer_idx);   // [total_seq, kv_dim]
        let cached_v = cache.get_values(layer_idx);  // [total_seq, kv_dim]
        let total_seq = cached_k.shape()[0];

        // Grouped-query attention per head
        let mut out_data = vec![0.0f64; seq_len * HIDDEN];

        for h in 0..HEADS {
            let kv_h = h / HEADS_PER_KV;

            // Extract q for this head: [seq_len, head_dim]
            // Extract k, v for this kv group: [total_seq, head_dim]
            // scores = q @ k^T / sqrt(head_dim)
            let scale = 1.0 / (HEAD_DIM as f64).sqrt();

            for qi in 0..seq_len {
                let q_off = qi * HIDDEN + h * HEAD_DIM;

                // Compute attention scores
                let mut scores = vec![0.0f64; total_seq];
                for ki in 0..total_seq {
                    let k_off = ki * KV_DIM + kv_h * HEAD_DIM;
                    let mut dot = 0.0;
                    for d in 0..HEAD_DIM {
                        dot += q_roped.data()[q_off + d] * cached_k.data()[k_off + d];
                    }
                    scores[ki] = dot * scale;
                }

                // Causal mask: only during prefill (seq_len > 1)
                if seq_len > 1 {
                    let q_pos = pos_offset + qi;
                    for ki in 0..total_seq {
                        if ki > q_pos {
                            scores[ki] = f64::NEG_INFINITY;
                        }
                    }
                }

                // Softmax
                let max_s = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mut sum = 0.0;
                for s in scores.iter_mut() {
                    *s = (*s - max_s).exp();
                    sum += *s;
                }
                for s in scores.iter_mut() {
                    *s /= sum;
                }

                // Weighted sum of values
                let out_off = qi * HIDDEN + h * HEAD_DIM;
                for ki in 0..total_seq {
                    let v_off = ki * KV_DIM + kv_h * HEAD_DIM;
                    let w = scores[ki];
                    for d in 0..HEAD_DIM {
                        out_data[out_off + d] += w * cached_v.data()[v_off + d];
                    }
                }
            }
        }

        // o_proj
        let attn_out = Tensor::new(out_data, Shape::from_slice(&[seq_len, HIDDEN]));
        matmul_weight(&attn_out, &self.o_proj.weight.data)
    }
}

// ---------------------------------------------------------------------------
// LlamaMLP (SwiGLU)
// ---------------------------------------------------------------------------

struct LlamaMLP {
    gate_proj: Linear<f64>,
    up_proj: Linear<f64>,
    down_proj: Linear<f64>,
}

impl LlamaMLP {
    fn new() -> Self {
        Self {
            gate_proj: Linear::new(HIDDEN, INTERMEDIATE, 0),
            up_proj: Linear::new(HIDDEN, INTERMEDIATE, 0),
            down_proj: Linear::new(INTERMEDIATE, HIDDEN, 0),
        }
    }

    fn forward(&self, x: &Tensor<f64>) -> Tensor<f64> {
        let gate = matmul_weight(x, &self.gate_proj.weight.data); // [seq, inter]
        let up = matmul_weight(x, &self.up_proj.weight.data);     // [seq, inter]

        // SwiGLU: silu(gate) * up
        let hidden = gate
            .map(|v| v * (1.0 / (1.0 + (-v).exp()))) // silu
            .mul(&up);

        matmul_weight(&hidden, &self.down_proj.weight.data) // [seq, hidden]
    }
}

// ---------------------------------------------------------------------------
// LlamaBlock
// ---------------------------------------------------------------------------

struct LlamaBlock {
    attn: LlamaAttention,
    mlp: LlamaMLP,
    attn_norm_weight: Vec<f64>,
    ffn_norm_weight: Vec<f64>,
}

impl LlamaBlock {
    fn new() -> Self {
        Self {
            attn: LlamaAttention::new(),
            mlp: LlamaMLP::new(),
            attn_norm_weight: vec![1.0; HIDDEN],
            ffn_norm_weight: vec![1.0; HIDDEN],
        }
    }

    fn forward(
        &self,
        x: &Tensor<f64>,
        layer_idx: usize,
        cache: &mut KVCache<f64>,
        pos_offset: usize,
    ) -> Tensor<f64> {
        // Pre-norm + attention + residual
        let normed = apply_rms_norm(x, &self.attn_norm_weight);
        let attn_out = self.attn.forward(&normed, layer_idx, cache, pos_offset);
        let x2 = x.add(&attn_out);

        // Pre-norm + MLP + residual
        let normed2 = apply_rms_norm(&x2, &self.ffn_norm_weight);
        let mlp_out = self.mlp.forward(&normed2);
        x2.add(&mlp_out)
    }
}

// ---------------------------------------------------------------------------
// LlamaModel
// ---------------------------------------------------------------------------

struct LlamaModel {
    embed_tokens: Tensor<f64>,  // [vocab, hidden]
    blocks: Vec<LlamaBlock>,
    final_norm_weight: Vec<f64>,
    lm_head: Linear<f64>,
}

impl LlamaModel {
    fn new() -> Self {
        Self {
            embed_tokens: Tensor::zeros(Shape::from_slice(&[VOCAB, HIDDEN])),
            blocks: (0..LAYERS).map(|_| LlamaBlock::new()).collect(),
            final_norm_weight: vec![1.0; HIDDEN],
            lm_head: Linear::new(HIDDEN, VOCAB, 0),
        }
    }

    fn forward(
        &self,
        token_ids: &[usize],
        cache: &mut KVCache<f64>,
        pos_offset: usize,
    ) -> Tensor<f64> {
        let seq_len = token_ids.len();

        // Token embedding: manual lookup
        let mut embed_data = vec![0.0f64; seq_len * HIDDEN];
        let embed_raw = self.embed_tokens.data();
        for (i, &tok) in token_ids.iter().enumerate() {
            let src = tok * HIDDEN;
            embed_data[i * HIDDEN..(i + 1) * HIDDEN]
                .copy_from_slice(&embed_raw[src..src + HIDDEN]);
        }
        let mut hidden = Tensor::new(embed_data, Shape::from_slice(&[seq_len, HIDDEN]));

        // Transformer blocks
        for (i, block) in self.blocks.iter().enumerate() {
            hidden = block.forward(&hidden, i, cache, pos_offset);
        }

        // Final norm
        let normed = apply_rms_norm(&hidden, &self.final_norm_weight);

        // LM head: only last token's logits
        let last_row = if seq_len == 1 {
            normed
        } else {
            let start = (seq_len - 1) * HIDDEN;
            Tensor::new(
                normed.data()[start..start + HIDDEN].to_vec(),
                Shape::from_slice(&[1, HIDDEN]),
            )
        };

        // logits = last_row @ lm_head.weight^T -> [1, vocab]
        let logits_2d = matmul_weight(&last_row, &self.lm_head.weight.data);
        // Flatten to [vocab]
        Tensor::new(logits_2d.data().to_vec(), Shape::from_slice(&[VOCAB]))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute x @ W^T where W is [out, in] and x is [seq, in] -> [seq, out].
/// Uses DMat for fast multiplication.
fn matmul_weight(x: &Tensor<f64>, weight: &Tensor<f64>) -> Tensor<f64> {
    let wt = weight.transpose(); // [in, out]
    x.matmul(&wt)
}

/// Apply RoPE to multi-head tensor. Input [seq, heads*head_dim], output same shape.
fn apply_rope_heads(
    x: &Tensor<f64>,
    rope: &RotaryEmbedding<f64>,
    n_heads: usize,
    head_dim: usize,
    offset: usize,
) -> Tensor<f64> {
    let seq_len = x.shape()[0];
    let full_dim = n_heads * head_dim;
    let x_data = x.data();
    let mut out = vec![0.0f64; seq_len * full_dim];

    for h in 0..n_heads {
        // Extract [seq, head_dim] for this head
        let mut head_data = vec![0.0f64; seq_len * head_dim];
        for s in 0..seq_len {
            for d in 0..head_dim {
                head_data[s * head_dim + d] = x_data[s * full_dim + h * head_dim + d];
            }
        }
        let head_tensor = Tensor::new(head_data, Shape::from_slice(&[seq_len, head_dim]));
        let rotated = rope.apply(&head_tensor, offset);

        // Write back
        for s in 0..seq_len {
            for d in 0..head_dim {
                out[s * full_dim + h * head_dim + d] = rotated.data()[s * head_dim + d];
            }
        }
    }

    Tensor::new(out, Shape::from_slice(&[seq_len, full_dim]))
}

/// Apply RMS norm row-by-row to a [seq, hidden] tensor.
fn apply_rms_norm(x: &Tensor<f64>, weight: &[f64]) -> Tensor<f64> {
    let seq_len = x.shape()[0];
    let dim = x.shape()[1];
    let x_data = x.data();
    let mut out = vec![0.0f64; seq_len * dim];

    for s in 0..seq_len {
        let row = &x_data[s * dim..(s + 1) * dim];
        let normed = rms_norm(row, weight, EPS);
        out[s * dim..(s + 1) * dim].copy_from_slice(&normed);
    }

    Tensor::new(out, Shape::from_slice(&[seq_len, dim]))
}

// ---------------------------------------------------------------------------
// Weight loading
// ---------------------------------------------------------------------------

fn load_model(weights: &HashMap<String, Tensor<f64>>) -> LlamaModel {
    let mut model = LlamaModel::new();

    // embed_tokens
    if let Some(w) = weights.get("model.embed_tokens.weight") {
        model.embed_tokens = w.clone();
    }

    // layers
    for i in 0..LAYERS {
        let prefix = format!("model.layers.{i}");

        if let Some(w) = weights.get(&format!("{prefix}.self_attn.q_proj.weight")) {
            model.blocks[i].attn.q_proj.weight = Parameter::new(w.clone());
        }
        if let Some(w) = weights.get(&format!("{prefix}.self_attn.k_proj.weight")) {
            model.blocks[i].attn.k_proj.weight = Parameter::new(w.clone());
        }
        if let Some(w) = weights.get(&format!("{prefix}.self_attn.v_proj.weight")) {
            model.blocks[i].attn.v_proj.weight = Parameter::new(w.clone());
        }
        if let Some(w) = weights.get(&format!("{prefix}.self_attn.o_proj.weight")) {
            model.blocks[i].attn.o_proj.weight = Parameter::new(w.clone());
        }

        if let Some(w) = weights.get(&format!("{prefix}.mlp.gate_proj.weight")) {
            model.blocks[i].mlp.gate_proj.weight = Parameter::new(w.clone());
        }
        if let Some(w) = weights.get(&format!("{prefix}.mlp.up_proj.weight")) {
            model.blocks[i].mlp.up_proj.weight = Parameter::new(w.clone());
        }
        if let Some(w) = weights.get(&format!("{prefix}.mlp.down_proj.weight")) {
            model.blocks[i].mlp.down_proj.weight = Parameter::new(w.clone());
        }

        if let Some(w) = weights.get(&format!("{prefix}.input_layernorm.weight")) {
            model.blocks[i].attn_norm_weight = w.data().to_vec();
        }
        if let Some(w) = weights.get(&format!("{prefix}.post_attention_layernorm.weight")) {
            model.blocks[i].ffn_norm_weight = w.data().to_vec();
        }
    }

    // final norm
    if let Some(w) = weights.get("model.norm.weight") {
        model.final_norm_weight = w.data().to_vec();
    }

    // lm_head (may be tied to embed_tokens)
    if let Some(w) = weights.get("lm_head.weight") {
        model.lm_head.weight = Parameter::new(w.clone());
    } else {
        // Weight tying: share embed_tokens
        model.lm_head.weight = Parameter::new(model.embed_tokens.clone());
    }

    model
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("=== SmolLM2-135M on tang ===\n");

    let repo = "HuggingFaceTB/SmolLM2-135M";

    // Download model files
    println!("downloading model from {repo}...");
    let t0 = Instant::now();
    let files = tang_hub::download(repo).expect("failed to download model");

    // Load weights
    let mut weights = HashMap::new();
    for wf in &files.weight_files {
        let tensors = tang_safetensors::load(wf).expect("failed to load safetensors");
        weights.extend(tensors);
    }
    println!(
        "loaded {} tensors in {:.1}s",
        weights.len(),
        t0.elapsed().as_secs_f64()
    );

    // Load tokenizer — download manually if hf-hub didn't fetch it
    let tokenizer = match files.tokenizer {
        Some(ref p) => tokenizers::Tokenizer::from_file(p).expect("failed to parse tokenizer"),
        None => {
            // hf-hub 0.3 sometimes fails to resolve file URLs; download via curl
            let dest = files.path.join("tokenizer.json");
            if !dest.exists() {
                println!("downloading tokenizer.json...");
                let url = format!(
                    "https://huggingface.co/{}/resolve/main/tokenizer.json",
                    repo
                );
                let status = std::process::Command::new("curl")
                    .args(["-sL", &url, "-o"])
                    .arg(&dest)
                    .status()
                    .expect("curl not found");
                assert!(status.success(), "failed to download tokenizer.json");
            }
            tokenizers::Tokenizer::from_file(&dest).expect("failed to parse tokenizer")
        }
    };
    println!("tokenizer vocab: {}\n", tokenizer.get_vocab_size(true));

    // Build model
    let t1 = Instant::now();
    let model = load_model(&weights);
    println!("model loaded in {:.1}s", t1.elapsed().as_secs_f64());
    drop(weights); // free raw weight map

    // Count parameters
    let n_params: usize = VOCAB * HIDDEN  // embed_tokens
        + LAYERS * (
            HIDDEN * HIDDEN      // q_proj
            + HIDDEN * KV_DIM    // k_proj
            + HIDDEN * KV_DIM    // v_proj
            + HIDDEN * HIDDEN    // o_proj
            + HIDDEN * INTERMEDIATE  // gate
            + HIDDEN * INTERMEDIATE  // up
            + INTERMEDIATE * HIDDEN  // down
            + HIDDEN             // attn_norm
            + HIDDEN             // ffn_norm
        )
        + HIDDEN                 // final_norm
        + VOCAB * HIDDEN;        // lm_head
    println!("parameters: {:.1}M\n", n_params as f64 / 1e6);

    // Prompt
    let prompt = "The meaning of life is";
    let prompt_ids = tang_hub::encode(&tokenizer, prompt);
    let prompt_tokens: Vec<usize> = prompt_ids.data().iter().map(|&v| v as usize).collect();
    println!("prompt: \"{prompt}\"");
    println!("tokens: {:?}\n", prompt_tokens);

    // KV cache
    let mut cache = KVCache::new(LAYERS, MAX_SEQ, KV_HEADS, HEAD_DIM);

    // --- Prefill ---
    let t_gen = Instant::now();
    let logits = model.forward(&prompt_tokens, &mut cache, 0);
    let prefill_ms = t_gen.elapsed().as_millis();
    println!("prefill: {prefill_ms}ms ({} tokens)", prompt_tokens.len());

    // --- Generate ---
    let max_new = 20;
    let mut sampler = Sampler::with_seed(
        SamplingConfig::greedy(),
        42,
    );

    let mut all_tokens = prompt_tokens.clone();
    let mut next_logits = logits;
    let eos_id = tokenizer.token_to_id("</s>").map(|id| id as usize);

    for step in 0..max_new {
        let tok = sampler.sample(&next_logits, &all_tokens);

        if eos_id == Some(tok) {
            break;
        }

        all_tokens.push(tok);
        let pos = prompt_tokens.len() + step;

        // Single-token forward
        next_logits = model.forward(&[tok], &mut cache, pos);
    }

    let gen_ms = t_gen.elapsed().as_millis();
    let n_generated = all_tokens.len() - prompt_tokens.len();

    // Decode
    let token_ids_u32: Vec<u32> = all_tokens.iter().map(|&t| t as u32).collect();
    let output = tokenizer.decode(&token_ids_u32, true).unwrap_or_default();

    println!("\n--- output ---");
    println!("{output}");
    println!("--- end ---\n");
    println!(
        "generated {n_generated} tokens in {gen_ms}ms ({:.1} tok/s)",
        n_generated as f64 / (gen_ms as f64 / 1000.0)
    );
}
