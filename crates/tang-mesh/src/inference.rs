//! Distributed inference: pipeline execution across heterogeneous nodes.
//!
//! `InferenceServer` partitions a forward graph across mesh nodes,
//! sends subgraphs for local WGSL compilation, and pipelines activations
//! through the mesh for inference requests.

use std::collections::HashMap;

use tracing::info;

use crate::coordinator::Coordinator;
use crate::error::MeshError;
use crate::mesh::Mesh;
use crate::partition::{auto_partition, GraphPartition};
use crate::protocol::WireGraph;

/// A loaded model's pipeline state.
struct ModelPipeline {
    /// Partitions in pipeline order.
    partitions: Vec<GraphPartition>,
    /// Task IDs per partition (assigned during compilation).
    task_ids: Vec<u64>,
}

/// Distributed inference server.
///
/// Manages model loading (graph partitioning + compilation) and inference
/// (activation pipelining) across mesh nodes.
pub struct InferenceServer {
    coordinator: Coordinator,
    /// Loaded models, keyed by name.
    pipelines: HashMap<String, ModelPipeline>,
}

impl InferenceServer {
    /// Create a new inference server connected to the mesh.
    pub async fn new(mesh: &Mesh) -> Result<Self, MeshError> {
        let coordinator = Coordinator::new();
        coordinator.connect_to_mesh(mesh).await?;

        Ok(Self {
            coordinator,
            pipelines: HashMap::new(),
        })
    }

    /// Create from an existing coordinator (for testing with channel transport).
    pub fn from_coordinator(coordinator: Coordinator) -> Self {
        Self {
            coordinator,
            pipelines: HashMap::new(),
        }
    }

    /// Load a model across the mesh.
    ///
    /// 1. Partitions the forward graph across mesh nodes
    /// 2. Sends each subgraph to its assigned worker
    /// 3. Workers compile to local WGSL
    ///
    /// After loading, the model is ready for inference via `infer()`.
    pub async fn load_model(
        &mut self,
        name: &str,
        forward_graph: WireGraph,
        mesh: &Mesh,
    ) -> Result<(), MeshError> {
        info!("loading model '{}' across {} nodes", name, mesh.len());

        // Partition the graph across nodes
        let partitions = auto_partition(&forward_graph, mesh)?;

        // Compile each partition on its target worker
        let mut task_ids = Vec::with_capacity(partitions.len());
        for (i, partition) in partitions.iter().enumerate() {
            let task_id = self
                .coordinator
                .compile_on(partition.node, &partition.graph)
                .await?;
            info!(
                "stage {i}: compiled on {} ({} nodes, {} outputs)",
                partition.node,
                partition.graph.nodes.len(),
                partition.graph.outputs.len()
            );
            task_ids.push(task_id);
        }

        self.pipelines.insert(
            name.to_string(),
            ModelPipeline {
                partitions,
                task_ids,
            },
        );

        info!("model '{}' loaded successfully", name);
        Ok(())
    }

    /// Run inference through the model pipeline.
    ///
    /// Activations flow through the pipeline:
    /// ```text
    /// input → Node 0 (subgraph_0) → Node 1 (subgraph_1) → ... → output
    /// ```
    ///
    /// Weights are loaded once per model. Only activations flow per request.
    pub async fn infer(
        &self,
        name: &str,
        inputs: Vec<f32>,
    ) -> Result<Vec<f32>, MeshError> {
        let pipeline = self
            .pipelines
            .get(name)
            .ok_or_else(|| MeshError::ExecutionFailed(format!("model '{name}' not loaded")))?;

        let mut activations = inputs;

        for (stage, (partition, task_id)) in pipeline
            .partitions
            .iter()
            .zip(pipeline.task_ids.iter())
            .enumerate()
        {
            activations = self
                .coordinator
                .forward_on(partition.node, *task_id, stage as u32, activations)
                .await?;
        }

        Ok(activations)
    }

    /// Unload a model, freeing resources on workers.
    pub fn unload_model(&mut self, name: &str) -> bool {
        self.pipelines.remove(name).is_some()
    }

    /// List loaded models.
    pub fn loaded_models(&self) -> Vec<&str> {
        self.pipelines.keys().map(|s| s.as_str()).collect()
    }

    /// Shut down all workers.
    pub async fn shutdown(self) {
        self.coordinator.shutdown_all().await;
    }
}

// ---------------------------------------------------------------------------
// Coded inference (tensor-parallel via erasure coding)
// ---------------------------------------------------------------------------

use crate::coded::{
    decode_outputs, reshape_blocks_to_seq, CompressedGrad, Generator,
    GradientPolicy, Shard,
};
use crate::mesh::NodeId;

/// Non-linearity type for decode/recode barriers.
#[derive(Clone, Debug)]
pub enum Activation {
    Gelu,
    LayerNorm { eps: f32 },
    Softmax,
    /// RMSNorm: y = x / sqrt(mean(x^2) + eps), then y *= scale (if present).
    RMSNorm { eps: f32, scale: Option<Vec<f32>> },
}

impl Activation {
    /// Compute gradient: returns dL/dx given dL/dy and pre-activation x.
    fn backward(&self, grad_output: &[f32], pre_activation: &[f32]) -> Vec<f32> {
        match self {
            Activation::Gelu => grad_output
                .iter()
                .zip(pre_activation)
                .map(|(&dy, &x)| {
                    let x3 = x * x * x;
                    let inner = 0.7978845608 * (x + 0.044715 * x3);
                    let tanh_inner = inner.tanh();
                    let sech2 = 1.0 - tanh_inner * tanh_inner;
                    let d_inner = 0.7978845608 * (1.0 + 3.0 * 0.044715 * x * x);
                    let gelu_prime = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * d_inner;
                    dy * gelu_prime
                })
                .collect(),
            Activation::LayerNorm { eps } => {
                let n = pre_activation.len() as f32;
                let mean = pre_activation.iter().sum::<f32>() / n;
                let var =
                    pre_activation.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
                let inv_std = 1.0 / (var + eps).sqrt();
                let dy_mean = grad_output.iter().sum::<f32>() / n;
                let dy_xhat_mean: f32 = grad_output
                    .iter()
                    .zip(pre_activation)
                    .map(|(&dy, &x)| dy * (x - mean) * inv_std)
                    .sum::<f32>()
                    / n;
                grad_output
                    .iter()
                    .zip(pre_activation)
                    .map(|(&dy, &x)| {
                        let xhat = (x - mean) * inv_std;
                        inv_std * (dy - dy_mean - xhat * dy_xhat_mean)
                    })
                    .collect()
            }
            Activation::Softmax => {
                let max =
                    pre_activation.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = pre_activation.iter().map(|&x| (x - max).exp()).collect();
                let sum: f32 = exps.iter().sum();
                let y: Vec<f32> = exps.iter().map(|&e| e / sum).collect();
                let dot: f32 =
                    grad_output.iter().zip(&y).map(|(&dy, &yi)| dy * yi).sum();
                grad_output
                    .iter()
                    .zip(&y)
                    .map(|(&dy, &yi)| yi * (dy - dot))
                    .collect()
            }
            Activation::RMSNorm { eps, scale } => {
                // Per-row backward when scale is present.
                let features = scale.as_ref().map(|s| s.len()).unwrap_or(pre_activation.len());
                let rows = pre_activation.len() / features;
                let n = features as f32;
                let mut result = vec![0.0f32; pre_activation.len()];
                for row in 0..rows {
                    let off = row * features;
                    let pa = &pre_activation[off..off + features];
                    let go = &grad_output[off..off + features];
                    let ms = pa.iter().map(|x| x * x).sum::<f32>() / n;
                    let rms = (ms + eps).sqrt();
                    let inv_rms = 1.0 / rms;
                    let scaled_grad: Vec<f32> = if let Some(s) = scale {
                        go.iter().zip(s).map(|(&dy, &si)| dy * si).collect()
                    } else {
                        go.to_vec()
                    };
                    let dy_xhat_mean: f32 = scaled_grad
                        .iter()
                        .zip(pa)
                        .map(|(&dy, &x)| dy * (x * inv_rms))
                        .sum::<f32>()
                        / n;
                    for (i, (&dy, &x)) in scaled_grad.iter().zip(pa).enumerate() {
                        let xh = x * inv_rms;
                        result[off + i] = inv_rms * (dy - xh * dy_xhat_mean);
                    }
                }
                result
            }
        }
    }

    /// Apply activation in-place.
    pub fn apply(&self, data: &mut [f32]) {
        match self {
            Activation::Gelu => {
                for x in data.iter_mut() {
                    // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
                    let x3 = *x * *x * *x;
                    let inner = 0.7978845608 * (*x + 0.044715 * x3);
                    *x = 0.5 * *x * (1.0 + inner.tanh());
                }
            }
            Activation::LayerNorm { eps } => {
                let n = data.len() as f32;
                let mean = data.iter().sum::<f32>() / n;
                let var = data.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
                let inv_std = 1.0 / (var + eps).sqrt();
                for x in data.iter_mut() {
                    *x = (*x - mean) * inv_std;
                }
            }
            Activation::Softmax => {
                let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for x in data.iter_mut() {
                    *x = (*x - max).exp();
                    sum += *x;
                }
                for x in data.iter_mut() {
                    *x /= sum;
                }
            }
            Activation::RMSNorm { eps, scale } => {
                // When scale is present, apply per-row (features = scale.len()).
                let features = scale.as_ref().map(|s| s.len()).unwrap_or(data.len());
                let rows = data.len() / features;
                for row in 0..rows {
                    let chunk = &mut data[row * features..(row + 1) * features];
                    let n = features as f32;
                    let ms = chunk.iter().map(|x| x * x).sum::<f32>() / n;
                    let rms = (ms + eps).sqrt();
                    if let Some(s) = scale {
                        for (x, &si) in chunk.iter_mut().zip(s) {
                            *x = (*x / rms) * si;
                        }
                    } else {
                        for x in chunk.iter_mut() {
                            *x /= rms;
                        }
                    }
                }
            }
        }
    }
}

/// A layer in the coded inference pipeline.
#[derive(Clone, Debug)]
pub enum CodedLayer {
    /// Linear layer — stays fully coded (no decode needed).
    /// Optional bias added after the coded matmul.
    Linear { d_in: usize, d_out: usize, bias: Option<Vec<f32>> },
    /// Non-linearity — requires decode → apply → re-encode.
    Nonlinear(Activation),

    // -- Dataflow control --
    /// Push x.clone() onto the residual stack.
    SaveResidual,
    /// x += residual_stack.pop()
    AddResidual,

    // -- Compound coded operations --
    /// 3 coded linears from SAME input → concat [Q;K;V].
    /// Consumes 3 consecutive shard indices.
    /// Optional per-projection biases applied to Q, K, V before concatenation.
    QkvProject {
        d_model: usize,
        d_q: usize,     // n_heads * head_dim
        d_k: usize,     // n_kv_heads * head_dim
        d_v: usize,     // n_kv_heads * head_dim
        q_bias: Option<Vec<f32>>,
        k_bias: Option<Vec<f32>>,
        v_bias: Option<Vec<f32>>,
    },
    /// Grouped-query attention on decoded [Q;K;V] → [seq, d_model].
    Attention {
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    },
    /// 2 coded linears (gate + up) from SAME input + SiLU + elementwise mul.
    /// Consumes 2 consecutive shard indices.
    /// Optional per-projection biases applied to gate and up BEFORE silu*mul.
    SwiGluUp {
        d_model: usize,
        ff_dim: usize,
        gate_bias: Option<Vec<f32>>,
        up_bias: Option<Vec<f32>>,
    },
    /// Standalone bias add: x[t*dim + i] += bias[i], dim = bias.len().
    Bias(Vec<f32>),
    /// Interleaved RoPE — sits between QkvProject and Attention.
    /// Rotates Q and K heads, V passes through.
    RoPE {
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        cos_table: Vec<f32>,  // [max_seq_len, head_dim/2] flat
        sin_table: Vec<f32>,
        max_seq_len: usize,
    },
}

// ---------------------------------------------------------------------------
// Activation helpers
// ---------------------------------------------------------------------------

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
fn silu_backward(x: f32) -> f32 {
    let sig = 1.0 / (1.0 + (-x).exp());
    sig * (1.0 + x * (1.0 - sig))
}

/// Precompute interleaved RoPE cos/sin tables.
///
/// Returns `(cos_table, sin_table)` each of shape `[max_seq_len, head_dim/2]` flattened.
/// θ_i = 1 / base^(2*i / head_dim) for pair index i.
pub fn precompute_rope_tables(
    head_dim: usize,
    max_seq_len: usize,
    base: f64,
) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos_table = vec![0.0f32; max_seq_len * half];
    let mut sin_table = vec![0.0f32; max_seq_len * half];
    for t in 0..max_seq_len {
        for i in 0..half {
            let theta = (t as f64) / base.powf(2.0 * i as f64 / head_dim as f64);
            cos_table[t * half + i] = theta.cos() as f32;
            sin_table[t * half + i] = theta.sin() as f32;
        }
    }
    (cos_table, sin_table)
}

/// Apply interleaved RoPE in-place to Q and K within a `[seq_len, d_q+d_k+d_v]` buffer.
///
/// For each head's interleaved pairs `(x[2i], x[2i+1])`:
/// ```text
/// y_even = x_even * cos(θ) - x_odd * sin(θ)
/// y_odd  = x_even * sin(θ) + x_odd * cos(θ)
/// ```
pub fn apply_rope(
    qkv: &mut [f32],
    seq_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    cos_table: &[f32],
    sin_table: &[f32],
) {
    let d_q = n_heads * head_dim;
    let d_k = n_kv_heads * head_dim;
    let d_v = n_kv_heads * head_dim;
    let d_qkv = d_q + d_k + d_v;
    let half = head_dim / 2;

    for t in 0..seq_len {
        let base = t * d_qkv;
        // Rotate Q heads
        for h in 0..n_heads {
            let off = base + h * head_dim;
            for i in 0..half {
                let cos = cos_table[t * half + i];
                let sin = sin_table[t * half + i];
                let even = qkv[off + 2 * i];
                let odd = qkv[off + 2 * i + 1];
                qkv[off + 2 * i] = even * cos - odd * sin;
                qkv[off + 2 * i + 1] = even * sin + odd * cos;
            }
        }
        // Rotate K heads
        for h in 0..n_kv_heads {
            let off = base + d_q + h * head_dim;
            for i in 0..half {
                let cos = cos_table[t * half + i];
                let sin = sin_table[t * half + i];
                let even = qkv[off + 2 * i];
                let odd = qkv[off + 2 * i + 1];
                qkv[off + 2 * i] = even * cos - odd * sin;
                qkv[off + 2 * i + 1] = even * sin + odd * cos;
            }
        }
        // V passes through (no rotation)
    }
}

/// RoPE backward: transpose rotation (orthogonal inverse).
///
/// ```text
/// dx_even =  dy_even * cos + dy_odd * sin
/// dx_odd  = -dy_even * sin + dy_odd * cos
/// ```
pub fn rope_backward(
    grad: &mut [f32],
    seq_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    cos_table: &[f32],
    sin_table: &[f32],
) {
    let d_q = n_heads * head_dim;
    let d_k = n_kv_heads * head_dim;
    let d_v = n_kv_heads * head_dim;
    let d_qkv = d_q + d_k + d_v;
    let half = head_dim / 2;

    for t in 0..seq_len {
        let base = t * d_qkv;
        // Q heads
        for h in 0..n_heads {
            let off = base + h * head_dim;
            for i in 0..half {
                let cos = cos_table[t * half + i];
                let sin = sin_table[t * half + i];
                let dy_even = grad[off + 2 * i];
                let dy_odd = grad[off + 2 * i + 1];
                grad[off + 2 * i] = dy_even * cos + dy_odd * sin;
                grad[off + 2 * i + 1] = -dy_even * sin + dy_odd * cos;
            }
        }
        // K heads
        for h in 0..n_kv_heads {
            let off = base + d_q + h * head_dim;
            for i in 0..half {
                let cos = cos_table[t * half + i];
                let sin = sin_table[t * half + i];
                let dy_even = grad[off + 2 * i];
                let dy_odd = grad[off + 2 * i + 1];
                grad[off + 2 * i] = dy_even * cos + dy_odd * sin;
                grad[off + 2 * i + 1] = -dy_even * sin + dy_odd * cos;
            }
        }
        // V: gradient passes through unchanged
    }
}

/// Add bias to batched activations: x[t*dim + i] += bias[i].
/// dim = bias.len(), seq_len = x.len() / dim.
pub fn apply_bias(x: &mut [f32], bias: &[f32]) {
    let dim = bias.len();
    assert!(dim > 0 && x.len() % dim == 0);
    for chunk in x.chunks_mut(dim) {
        for (xi, &bi) in chunk.iter_mut().zip(bias) {
            *xi += bi;
        }
    }
}

// ---------------------------------------------------------------------------
// Attention helpers
// ---------------------------------------------------------------------------

/// Multi-head (GQA) attention forward.
///
/// Input: `qkv` is `[seq_len, d_q + d_k + d_v]` flattened (token-major).
/// Returns `[seq_len, n_heads * head_dim]` flattened.
///
/// Also returns `(attn_weights, q, k, v)` for backward.
pub fn attention_forward(
    qkv: &[f32],
    seq_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let d_q = n_heads * head_dim;
    let d_k = n_kv_heads * head_dim;
    let d_v = n_kv_heads * head_dim;
    let d_qkv = d_q + d_k + d_v;
    assert_eq!(qkv.len(), seq_len * d_qkv);

    let heads_per_kv = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Split Q, K, V: each [seq_len, d_*]
    let mut q = vec![0.0f32; seq_len * d_q];
    let mut k = vec![0.0f32; seq_len * d_k];
    let mut v = vec![0.0f32; seq_len * d_v];
    for t in 0..seq_len {
        let off = t * d_qkv;
        q[t * d_q..(t + 1) * d_q].copy_from_slice(&qkv[off..off + d_q]);
        k[t * d_k..(t + 1) * d_k].copy_from_slice(&qkv[off + d_q..off + d_q + d_k]);
        v[t * d_v..(t + 1) * d_v].copy_from_slice(&qkv[off + d_q + d_k..off + d_qkv]);
    }

    let d_model = n_heads * head_dim;
    let mut output = vec![0.0f32; seq_len * d_model];
    let mut all_attn_weights = vec![0.0f32; n_heads * seq_len * seq_len];

    for h in 0..n_heads {
        let kv_h = h / heads_per_kv;

        // scores[i,j] = q[i] · k[j] / sqrt(hd), with causal mask
        let mut scores = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                if j > i {
                    scores[i * seq_len + j] = f32::NEG_INFINITY;
                } else {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[i * d_q + h * head_dim + d]
                            * k[j * d_k + kv_h * head_dim + d];
                    }
                    scores[i * seq_len + j] = dot * scale;
                }
            }
        }

        // softmax per row
        for i in 0..seq_len {
            let row = &mut scores[i * seq_len..(i + 1) * seq_len];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in row.iter_mut() {
                *s = (*s - max).exp();
                sum += *s;
            }
            for s in row.iter_mut() {
                *s /= sum;
            }
        }

        // store attn weights for backward
        all_attn_weights[h * seq_len * seq_len..(h + 1) * seq_len * seq_len]
            .copy_from_slice(&scores);

        // out[i] = sum_j attn[i,j] * v[j]
        for i in 0..seq_len {
            for j in 0..seq_len {
                let a = scores[i * seq_len + j];
                for d in 0..head_dim {
                    output[i * d_model + h * head_dim + d] +=
                        a * v[j * d_v + kv_h * head_dim + d];
                }
            }
        }
    }

    (output, all_attn_weights, q, k, v)
}

/// Multi-head (GQA) attention backward.
///
/// `d_output`: gradient w.r.t. attention output `[seq_len, d_model]`.
/// Returns gradient w.r.t. qkv input `[seq_len, d_q + d_k + d_v]`.
fn attention_backward(
    d_output: &[f32],
    attn_weights: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let d_q = n_heads * head_dim;
    let d_k = n_kv_heads * head_dim;
    let d_v = n_kv_heads * head_dim;
    let d_qkv = d_q + d_k + d_v;
    let d_model = n_heads * head_dim;
    let heads_per_kv = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut dq = vec![0.0f32; seq_len * d_q];
    let mut dk = vec![0.0f32; seq_len * d_k];
    let mut dv = vec![0.0f32; seq_len * d_v];

    for h in 0..n_heads {
        let kv_h = h / heads_per_kv;
        let aw = &attn_weights[h * seq_len * seq_len..(h + 1) * seq_len * seq_len];

        // d_attn[i,j] from d_output and v
        // d_v[j] += sum_i attn[i,j] * d_output[i]
        // d_attn[i,j] = d_output[i] · v[j]
        let mut d_attn = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += d_output[i * d_model + h * head_dim + d]
                        * v[j * d_v + kv_h * head_dim + d];
                }
                d_attn[i * seq_len + j] = dot;

                // dv
                let a = aw[i * seq_len + j];
                for d in 0..head_dim {
                    dv[j * d_v + kv_h * head_dim + d] +=
                        a * d_output[i * d_model + h * head_dim + d];
                }
            }
        }

        // softmax backward: d_scores = attn * (d_attn - sum_j(attn * d_attn))
        let mut d_scores = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            let row_sum: f32 = (0..seq_len)
                .map(|j| aw[i * seq_len + j] * d_attn[i * seq_len + j])
                .sum();
            for j in 0..seq_len {
                d_scores[i * seq_len + j] =
                    aw[i * seq_len + j] * (d_attn[i * seq_len + j] - row_sum);
            }
        }

        // Scale backward
        for s in d_scores.iter_mut() {
            *s *= scale;
        }

        // dq[i] += sum_j d_scores[i,j] * k[j]
        // dk[j] += sum_i d_scores[i,j] * q[i]
        for i in 0..seq_len {
            for j in 0..=i {
                // causal: only j <= i
                let ds = d_scores[i * seq_len + j];
                for d in 0..head_dim {
                    dq[i * d_q + h * head_dim + d] +=
                        ds * k[j * d_k + kv_h * head_dim + d];
                    dk[j * d_k + kv_h * head_dim + d] +=
                        ds * q[i * d_q + h * head_dim + d];
                }
            }
        }
    }

    // Reassemble dqkv
    let mut dqkv = vec![0.0f32; seq_len * d_qkv];
    for t in 0..seq_len {
        let off = t * d_qkv;
        dqkv[off..off + d_q].copy_from_slice(&dq[t * d_q..(t + 1) * d_q]);
        dqkv[off + d_q..off + d_q + d_k].copy_from_slice(&dk[t * d_k..(t + 1) * d_k]);
        dqkv[off + d_q + d_k..off + d_qkv].copy_from_slice(&dv[t * d_v..(t + 1) * d_v]);
    }
    dqkv
}

/// Coded tensor-parallel inference server.
///
/// Orchestrates k-node groups for coded inference:
/// - Linear layers: broadcast x → collect k coded outputs (stays coded)
/// - Non-linearities: decode → apply → re-encode → broadcast
pub struct CodedInferenceServer {
    coordinator: Coordinator,
    generator: Generator,
    /// The k node indices forming the active group.
    group: Vec<NodeId>,
    /// Layer sequence for the model.
    layers: Vec<CodedLayer>,
    /// Gradient policy for learning during inference.
    policy: GradientPolicy,
}

impl CodedInferenceServer {
    /// Create a new coded inference server with the given k-node group.
    pub fn new(
        coordinator: Coordinator,
        generator: Generator,
        group: Vec<NodeId>,
        layers: Vec<CodedLayer>,
    ) -> Self {
        assert_eq!(
            group.len(),
            generator.k,
            "group size must equal k"
        );
        Self {
            coordinator,
            generator,
            group,
            layers,
            policy: GradientPolicy::default(),
        }
    }

    /// Run coded forward, collecting partial outputs from k nodes and decoding.
    ///
    /// Returns raw decoded output (block-major). For 1-D input this is
    /// already token-major. For batched input use `coded_forward_seq`.
    async fn coded_forward_layer(
        &self,
        linear_layer_idx: u32,
        x: &[f32],
        d_in: usize,
    ) -> Result<Vec<f32>, MeshError> {
        let k = self.generator.k;
        let mut coded_outputs = Vec::with_capacity(k);
        for node_id in &self.group {
            let result = self
                .coordinator
                .coded_forward_on(*node_id, linear_layer_idx, x.to_vec(), d_in as u32)
                .await?;
            coded_outputs.push((node_id.0 as usize, result));
        }
        let refs: Vec<(usize, &[f32])> = coded_outputs
            .iter()
            .map(|(i, v)| (*i, v.as_slice()))
            .collect();
        Ok(decode_outputs(&self.generator, &refs))
    }

    /// Coded forward with block→token reshape for batched input.
    ///
    /// Returns `[seq_len, d_out]` flattened (token-major).
    async fn coded_forward_seq(
        &self,
        linear_layer_idx: u32,
        x: &[f32],
        d_in: usize,
        d_out: usize,
    ) -> Result<Vec<f32>, MeshError> {
        let decoded = self.coded_forward_layer(linear_layer_idx, x, d_in).await?;
        let seq_len = x.len() / d_in;
        if seq_len <= 1 {
            Ok(decoded[..d_out].to_vec())
        } else {
            Ok(reshape_blocks_to_seq(&decoded, self.generator.k, seq_len)
                [..seq_len * d_out]
                .to_vec())
        }
    }

    /// Run coded inference through the layer sequence.
    pub async fn infer(&self, input: Vec<f32>) -> Result<Vec<f32>, MeshError> {
        let mut x = input;
        let mut linear_idx = 0u32;
        let mut residual_stack: Vec<Vec<f32>> = Vec::new();

        for layer in &self.layers {
            match layer {
                CodedLayer::Linear { d_in, d_out, bias } => {
                    x = self.coded_forward_seq(linear_idx, &x, *d_in, *d_out).await?;
                    if let Some(b) = bias {
                        apply_bias(&mut x, b);
                    }
                    linear_idx += 1;
                }
                CodedLayer::Nonlinear(activation) => {
                    activation.apply(&mut x);
                }
                CodedLayer::SaveResidual => {
                    residual_stack.push(x.clone());
                }
                CodedLayer::AddResidual => {
                    let res = residual_stack.pop().expect("residual stack underflow");
                    for (xi, ri) in x.iter_mut().zip(&res) {
                        *xi += ri;
                    }
                }
                CodedLayer::QkvProject { d_model, d_q, d_k, d_v, q_bias, k_bias, v_bias } => {
                    let seq_len = x.len() / d_model;
                    let mut q = self.coded_forward_seq(linear_idx, &x, *d_model, *d_q).await?;
                    let mut k = self.coded_forward_seq(linear_idx + 1, &x, *d_model, *d_k).await?;
                    let mut v = self.coded_forward_seq(linear_idx + 2, &x, *d_model, *d_v).await?;
                    linear_idx += 3;
                    if let Some(b) = q_bias { apply_bias(&mut q, b); }
                    if let Some(b) = k_bias { apply_bias(&mut k, b); }
                    if let Some(b) = v_bias { apply_bias(&mut v, b); }
                    // Concat [Q;K;V] per token: [seq, d_q+d_k+d_v]
                    let d_qkv = d_q + d_k + d_v;
                    let mut qkv = vec![0.0f32; seq_len * d_qkv];
                    for t in 0..seq_len {
                        qkv[t * d_qkv..t * d_qkv + d_q]
                            .copy_from_slice(&q[t * d_q..(t + 1) * d_q]);
                        qkv[t * d_qkv + d_q..t * d_qkv + d_q + d_k]
                            .copy_from_slice(&k[t * d_k..(t + 1) * d_k]);
                        qkv[t * d_qkv + d_q + d_k..t * d_qkv + d_qkv]
                            .copy_from_slice(&v[t * d_v..(t + 1) * d_v]);
                    }
                    x = qkv;
                }
                CodedLayer::Attention { n_heads, n_kv_heads, head_dim } => {
                    let d_qkv = (n_heads + 2 * n_kv_heads) * head_dim;
                    let seq_len = x.len() / d_qkv;
                    let (out, _, _, _, _) = attention_forward(
                        &x, seq_len, *n_heads, *n_kv_heads, *head_dim,
                    );
                    x = out;
                }
                CodedLayer::SwiGluUp { d_model, ff_dim, gate_bias, up_bias } => {
                    let mut gate = self.coded_forward_seq(linear_idx, &x, *d_model, *ff_dim).await?;
                    let mut up = self.coded_forward_seq(linear_idx + 1, &x, *d_model, *ff_dim).await?;
                    linear_idx += 2;
                    if let Some(b) = gate_bias { apply_bias(&mut gate, b); }
                    if let Some(b) = up_bias { apply_bias(&mut up, b); }
                    x = gate.iter().zip(&up).map(|(&g, &u)| silu(g) * u).collect();
                }
                CodedLayer::Bias(bias) => {
                    apply_bias(&mut x, bias);
                }
                CodedLayer::RoPE { n_heads, n_kv_heads, head_dim, cos_table, sin_table, .. } => {
                    let d_qkv = (n_heads + 2 * n_kv_heads) * head_dim;
                    let seq_len = x.len() / d_qkv;
                    apply_rope(
                        &mut x, seq_len, *n_heads, *n_kv_heads, *head_dim,
                        cos_table, sin_table,
                    );
                }
            }
        }
        Ok(x)
    }

    /// Run inference and learn: forward → MSE loss → backward → gossip updates.
    ///
    /// Returns `(output, mse_loss)`.
    pub async fn infer_and_learn(
        &self,
        input: Vec<f32>,
        target: &[f32],
    ) -> Result<(Vec<f32>, f32), MeshError> {
        let k = self.generator.k;

        // Count total linear (shard) layers for decoding
        let num_linear = self.count_linear_layers();

        // --- Forward pass (save state for backward) ---
        let mut x = input;
        let mut linear_idx = 0u32;
        let mut linear_inputs: Vec<Vec<f32>> = Vec::new();
        let mut pre_activations: Vec<Vec<f32>> = Vec::new();
        let mut residual_stack: Vec<Vec<f32>> = Vec::new();
        // State for compound layers
        let mut qkv_inputs: Vec<Vec<f32>> = Vec::new();
        let mut attn_cache: Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> = Vec::new();
        let mut swiglu_cache: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = Vec::new();

        for layer in &self.layers {
            match layer {
                CodedLayer::Linear { d_in, d_out, bias } => {
                    linear_inputs.push(x.clone());
                    x = self.coded_forward_seq(linear_idx, &x, *d_in, *d_out).await?;
                    if let Some(b) = bias {
                        apply_bias(&mut x, b);
                    }
                    linear_idx += 1;
                }
                CodedLayer::Nonlinear(activation) => {
                    pre_activations.push(x.clone());
                    activation.apply(&mut x);
                }
                CodedLayer::SaveResidual => {
                    residual_stack.push(x.clone());
                }
                CodedLayer::AddResidual => {
                    let res = residual_stack.pop().expect("residual stack underflow");
                    for (xi, ri) in x.iter_mut().zip(&res) {
                        *xi += ri;
                    }
                }
                CodedLayer::QkvProject { d_model, d_q, d_k, d_v, q_bias, k_bias, v_bias } => {
                    let seq_len = x.len() / d_model;
                    qkv_inputs.push(x.clone());
                    let mut q_out = self.coded_forward_seq(linear_idx, &x, *d_model, *d_q).await?;
                    let mut k_out = self.coded_forward_seq(linear_idx + 1, &x, *d_model, *d_k).await?;
                    let mut v_out = self.coded_forward_seq(linear_idx + 2, &x, *d_model, *d_v).await?;
                    linear_idx += 3;
                    if let Some(b) = q_bias { apply_bias(&mut q_out, b); }
                    if let Some(b) = k_bias { apply_bias(&mut k_out, b); }
                    if let Some(b) = v_bias { apply_bias(&mut v_out, b); }
                    let d_qkv = d_q + d_k + d_v;
                    let mut qkv = vec![0.0f32; seq_len * d_qkv];
                    for t in 0..seq_len {
                        qkv[t * d_qkv..t * d_qkv + d_q]
                            .copy_from_slice(&q_out[t * d_q..(t + 1) * d_q]);
                        qkv[t * d_qkv + d_q..t * d_qkv + d_q + d_k]
                            .copy_from_slice(&k_out[t * d_k..(t + 1) * d_k]);
                        qkv[t * d_qkv + d_q + d_k..t * d_qkv + d_qkv]
                            .copy_from_slice(&v_out[t * d_v..(t + 1) * d_v]);
                    }
                    x = qkv;
                }
                CodedLayer::Attention { n_heads, n_kv_heads, head_dim } => {
                    let d_qkv = (n_heads + 2 * n_kv_heads) * head_dim;
                    let seq_len = x.len() / d_qkv;
                    let (out, aw, q, kk, v) = attention_forward(
                        &x, seq_len, *n_heads, *n_kv_heads, *head_dim,
                    );
                    attn_cache.push((aw, q, kk, v));
                    x = out;
                }
                CodedLayer::SwiGluUp { d_model, ff_dim, gate_bias, up_bias } => {
                    swiglu_cache.push((vec![], vec![], x.clone())); // placeholder
                    let mut gate = self.coded_forward_seq(linear_idx, &x, *d_model, *ff_dim).await?;
                    let mut up = self.coded_forward_seq(linear_idx + 1, &x, *d_model, *ff_dim).await?;
                    linear_idx += 2;
                    // Apply biases BEFORE silu*mul
                    if let Some(b) = gate_bias { apply_bias(&mut gate, b); }
                    if let Some(b) = up_bias { apply_bias(&mut up, b); }
                    // Save raw gate and up (after bias) for backward
                    let cache_idx = swiglu_cache.len() - 1;
                    swiglu_cache[cache_idx].0 = gate.clone();
                    swiglu_cache[cache_idx].1 = up.clone();
                    x = gate.iter().zip(&up).map(|(&g, &u)| silu(g) * u).collect();
                }
                CodedLayer::Bias(bias) => {
                    apply_bias(&mut x, bias);
                }
                CodedLayer::RoPE { n_heads, n_kv_heads, head_dim, cos_table, sin_table, .. } => {
                    // No trainable params — just apply rotation (save pre-RoPE for backward)
                    pre_activations.push(x.clone());
                    let d_qkv = (n_heads + 2 * n_kv_heads) * head_dim;
                    let seq_len = x.len() / d_qkv;
                    apply_rope(
                        &mut x, seq_len, *n_heads, *n_kv_heads, *head_dim,
                        cos_table, sin_table,
                    );
                }
            }
        }

        let output = x;

        // --- MSE loss ---
        let n = output.len() as f32;
        let loss: f32 =
            output.iter().zip(target).map(|(o, t)| (o - t).powi(2)).sum::<f32>() / n;
        let mut grad: Vec<f32> =
            output.iter().zip(target).map(|(o, t)| 2.0 * (o - t) / n).collect();

        // --- Decode per-layer weights for backward ---
        let mut all_shards_by_node: Vec<(usize, Vec<Shard>)> = Vec::with_capacity(k);
        for node_id in &self.group {
            let shards = self.coordinator.request_shards_from(*node_id).await?;
            all_shards_by_node.push((node_id.0 as usize, shards));
        }

        let mut full_weights_per_layer: Vec<Vec<f32>> = Vec::with_capacity(num_linear);
        let mut version_per_layer: Vec<u64> = Vec::with_capacity(num_linear);
        for layer_idx in 0..num_linear {
            let shard_refs: Vec<(usize, &Shard)> = all_shards_by_node
                .iter()
                .map(|(node_idx, shards)| (*node_idx, &shards[layer_idx]))
                .collect();
            let weight_blocks = self.generator.decode(&shard_refs);
            let full_weights: Vec<f32> = weight_blocks.into_iter().flatten().collect();
            version_per_layer.push(shard_refs[0].1.version);
            full_weights_per_layer.push(full_weights);
        }

        // --- Backward pass ---
        let mut li = linear_inputs.len();
        let mut pi = pre_activations.len();
        let mut qi = qkv_inputs.len();
        let mut ai = attn_cache.len();
        let mut si = swiglu_cache.len();
        let mut residual_grad_stack: Vec<Vec<f32>> = Vec::new();
        let mut weight_grads: Vec<(usize, Vec<f32>)> = Vec::new();
        // Track linear_idx in reverse
        let mut rev_linear_idx = num_linear;

        for layer in self.layers.iter().rev() {
            match layer {
                CodedLayer::Nonlinear(activation) => {
                    pi -= 1;
                    grad = activation.backward(&grad, &pre_activations[pi]);
                }
                CodedLayer::Linear { d_in, d_out, .. } => {
                    li -= 1;
                    rev_linear_idx -= 1;
                    let x_in = &linear_inputs[li];
                    let full_w = &full_weights_per_layer[rev_linear_idx];
                    let seq_len = x_in.len() / d_in;

                    // dW = sum_t outer(grad[t], x[t])
                    let mut dw = vec![0.0f32; d_out * d_in];
                    for t in 0..seq_len {
                        let g_off = t * d_out;
                        let x_off = t * d_in;
                        for r in 0..*d_out {
                            for c in 0..*d_in {
                                dw[r * d_in + c] += grad[g_off + r] * x_in[x_off + c];
                            }
                        }
                    }
                    weight_grads.push((rev_linear_idx, dw));

                    // dx = W^T @ grad per token
                    let mut dx = vec![0.0f32; seq_len * d_in];
                    for t in 0..seq_len {
                        for c in 0..*d_in {
                            let mut sum = 0.0f32;
                            for r in 0..*d_out {
                                sum += full_w[r * d_in + c] * grad[t * d_out + r];
                            }
                            dx[t * d_in + c] = sum;
                        }
                    }
                    grad = dx;
                }
                CodedLayer::SaveResidual => {
                    let rg = residual_grad_stack.pop().expect("residual grad stack underflow");
                    for (gi, rgi) in grad.iter_mut().zip(&rg) {
                        *gi += rgi;
                    }
                }
                CodedLayer::AddResidual => {
                    residual_grad_stack.push(grad.clone());
                    // grad passes through unchanged
                }
                CodedLayer::QkvProject { d_model, d_q, d_k, d_v, .. } => {
                    qi -= 1;
                    rev_linear_idx -= 3;
                    let x_in = &qkv_inputs[qi];
                    let seq_len = x_in.len() / d_model;
                    let d_qkv = d_q + d_k + d_v;

                    // Split grad into dQ, dK, dV
                    let mut dq_flat = vec![0.0f32; seq_len * d_q];
                    let mut dk_flat = vec![0.0f32; seq_len * d_k];
                    let mut dv_flat = vec![0.0f32; seq_len * d_v];
                    for t in 0..seq_len {
                        let off = t * d_qkv;
                        dq_flat[t * d_q..(t + 1) * d_q]
                            .copy_from_slice(&grad[off..off + d_q]);
                        dk_flat[t * d_k..(t + 1) * d_k]
                            .copy_from_slice(&grad[off + d_q..off + d_q + d_k]);
                        dv_flat[t * d_v..(t + 1) * d_v]
                            .copy_from_slice(&grad[off + d_q + d_k..off + d_qkv]);
                    }

                    // Weight grads and input grads for each of Q, K, V projections
                    let mut dx = vec![0.0f32; seq_len * d_model];
                    for (proj_offset, (d_proj, d_grad)) in [
                        (*d_q, &dq_flat), (*d_k, &dk_flat), (*d_v, &dv_flat),
                    ].iter().enumerate() {
                        let widx = rev_linear_idx + proj_offset;
                        let full_w = &full_weights_per_layer[widx];
                        let mut dw = vec![0.0f32; d_proj * d_model];
                        for t in 0..seq_len {
                            for r in 0..*d_proj {
                                for c in 0..*d_model {
                                    dw[r * d_model + c] +=
                                        d_grad[t * d_proj + r] * x_in[t * d_model + c];
                                }
                            }
                        }
                        weight_grads.push((widx, dw));

                        // dx += W^T @ d_grad per token
                        for t in 0..seq_len {
                            for c in 0..*d_model {
                                let mut sum = 0.0f32;
                                for r in 0..*d_proj {
                                    sum += full_w[r * d_model + c] * d_grad[t * d_proj + r];
                                }
                                dx[t * d_model + c] += sum;
                            }
                        }
                    }
                    grad = dx;
                }
                CodedLayer::Attention { n_heads, n_kv_heads, head_dim } => {
                    ai -= 1;
                    let (ref aw, ref q, ref kk, ref v) = attn_cache[ai];
                    let d_model = n_heads * head_dim;
                    let seq_len = grad.len() / d_model;
                    grad = attention_backward(
                        &grad, aw, q, kk, v, seq_len, *n_heads, *n_kv_heads, *head_dim,
                    );
                }
                CodedLayer::SwiGluUp { d_model, ff_dim, .. } => {
                    si -= 1;
                    rev_linear_idx -= 2;
                    let (ref gate_raw, ref up_raw, ref x_in) = swiglu_cache[si];
                    let seq_len = x_in.len() / d_model;

                    // d_gate = grad * up * silu'(gate), d_up = grad * silu(gate)
                    let d_gate: Vec<f32> = grad.iter().zip(up_raw).zip(gate_raw)
                        .map(|((&dy, &u), &g)| dy * u * silu_backward(g))
                        .collect();
                    let d_up: Vec<f32> = grad.iter().zip(gate_raw)
                        .map(|(&dy, &g)| dy * silu(g))
                        .collect();

                    // Weight grads for gate and up projections
                    let mut dx = vec![0.0f32; seq_len * d_model];
                    for (proj_offset, d_grad) in [(0, &d_gate), (1, &d_up)] {
                        let widx = rev_linear_idx + proj_offset;
                        let full_w = &full_weights_per_layer[widx];
                        let mut dw = vec![0.0f32; ff_dim * d_model];
                        for t in 0..seq_len {
                            for r in 0..*ff_dim {
                                for c in 0..*d_model {
                                    dw[r * d_model + c] +=
                                        d_grad[t * ff_dim + r] * x_in[t * d_model + c];
                                }
                            }
                        }
                        weight_grads.push((widx, dw));

                        for t in 0..seq_len {
                            for c in 0..*d_model {
                                let mut sum = 0.0f32;
                                for r in 0..*ff_dim {
                                    sum += full_w[r * d_model + c] * d_grad[t * ff_dim + r];
                                }
                                dx[t * d_model + c] += sum;
                            }
                        }
                    }
                    grad = dx;
                }
                CodedLayer::Bias(_) => {
                    // Bias is frozen — gradient passes through unchanged.
                }
                CodedLayer::RoPE { n_heads, n_kv_heads, head_dim, cos_table, sin_table, .. } => {
                    // No trainable params — apply transpose rotation to gradient.
                    // We saved pre-RoPE state in pre_activations during forward.
                    pi -= 1;
                    let d_qkv = (n_heads + 2 * n_kv_heads) * head_dim;
                    let seq_len = grad.len() / d_qkv;
                    rope_backward(
                        &mut grad, seq_len, *n_heads, *n_kv_heads, *head_dim,
                        cos_table, sin_table,
                    );
                }
            }
        }

        // --- Encode and gossip weight gradients (per layer) ---
        for (layer_idx, dw) in &weight_grads {
            let version = version_per_layer[*layer_idx];
            let block_len = (dw.len() + k - 1) / k;
            let mut padded = dw.clone();
            padded.resize(block_len * k, 0.0);

            let block_grads: Vec<&[f32]> =
                (0..k).map(|j| &padded[j * block_len..(j + 1) * block_len]).collect();

            for node_id in &self.group {
                let coded_grad =
                    self.generator.encode_update(node_id.0 as usize, &block_grads);
                let compressed =
                    CompressedGrad::compress(&coded_grad, self.policy.top_k_ratio);
                self.coordinator
                    .coded_update_on(*node_id, *layer_idx as u32, compressed, version)
                    .await?;
            }
        }

        Ok((output, loss))
    }

    /// Count total linear (shard) layers including those inside compound layers.
    pub fn count_linear_layers(&self) -> usize {
        self.layers.iter().map(|l| match l {
            CodedLayer::Linear { .. } => 1,
            CodedLayer::QkvProject { .. } => 3,
            CodedLayer::SwiGluUp { .. } => 2,
            _ => 0,
        }).sum()
    }

    /// Gossip a coded gradient update for a specific layer to all nodes in the group.
    pub async fn gossip_update(
        &self,
        layer: u32,
        grad: &CompressedGrad,
        version: u64,
    ) -> Result<(), MeshError> {
        for node_id in &self.group {
            self.coordinator
                .coded_update_on(*node_id, layer, grad.clone(), version)
                .await?;
        }
        Ok(())
    }

    /// Access the gradient policy.
    pub fn policy(&self) -> &GradientPolicy {
        &self.policy
    }

    /// Mutable access to the gradient policy.
    pub fn policy_mut(&mut self) -> &mut GradientPolicy {
        &mut self.policy
    }
}

// ---------------------------------------------------------------------------
// Transformer builder
// ---------------------------------------------------------------------------

/// Configuration for building a coded transformer.
#[derive(Clone, Debug)]
pub struct TransformerConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ff_dim: usize,
    pub n_layers: usize,
    pub vocab_size: usize,
    pub eps: f32,
    /// If set, interleaved RoPE is applied after each QkvProject.
    pub rope_base: Option<f64>,
    /// Maximum sequence length for precomputing RoPE cos/sin tables.
    pub max_seq_len: usize,
}

/// Weights for a coded transformer.
///
/// Per-block ordering: `[wq, wk, wv, wo, gate, up, down]`.
pub struct TransformerWeights {
    /// Per-block: \[wq, wk, wv, wo, gate, up, down\] as flat f32 row-major.
    pub block_weights: Vec<[Vec<f32>; 7]>,
    /// Per-block: \[bq, bk, bv, bo, b_gate, b_up, b_down\].
    pub block_biases: Vec<[Vec<f32>; 7]>,
    /// Per-block: (ln1_scale, ln2_scale).
    pub block_norms: Vec<(Vec<f32>, Vec<f32>)>,
    /// Final layer norm scale.
    pub ln_final_scale: Vec<f32>,
    /// lm_head weight \[vocab_size, d_model\] row-major.
    pub lm_head_weight: Vec<f32>,
    /// lm_head bias \[vocab_size\].
    pub lm_head_bias: Vec<f32>,
    /// Embedding table \[vocab_size, d_model\] row-major (not coded).
    pub embed_table: Vec<f32>,
}

/// Build a full coded transformer layer sequence from config and weights.
///
/// Returns `(layers, weight_slices)` where:
/// - `layers`: the `CodedLayer` sequence (N blocks + ln_final + lm_head)
/// - `weight_slices`: all linear weight matrices in shard order (7*N + 1 entries)
///
/// The embedding table is NOT included in the coded layers — it runs on the
/// coordinator as a simple lookup.
pub fn build_coded_transformer(
    config: &TransformerConfig,
    weights: &TransformerWeights,
) -> (Vec<CodedLayer>, Vec<Vec<f32>>) {
    assert_eq!(weights.block_weights.len(), config.n_layers);
    assert_eq!(weights.block_biases.len(), config.n_layers);
    assert_eq!(weights.block_norms.len(), config.n_layers);

    let d_model = config.d_model;
    let d_q = config.n_heads * config.head_dim;
    let d_k = config.n_kv_heads * config.head_dim;
    let d_v = config.n_kv_heads * config.head_dim;
    let ff_dim = config.ff_dim;

    // Precompute RoPE tables once (shared across all layers)
    let rope_tables = config.rope_base.map(|base| {
        precompute_rope_tables(config.head_dim, config.max_seq_len, base)
    });

    let mut layers = Vec::new();
    let mut all_weights = Vec::new();

    for i in 0..config.n_layers {
        let [ref wq, ref wk, ref wv, ref wo, ref w_gate, ref w_up, ref w_down] =
            weights.block_weights[i];
        let [ref bq, ref bk, ref bv, ref bo, ref b_gate, ref b_up, ref b_down] =
            weights.block_biases[i];
        let (ref ln1_scale, ref ln2_scale) = weights.block_norms[i];

        // --- Attention half ---
        layers.push(CodedLayer::SaveResidual);
        layers.push(CodedLayer::Nonlinear(Activation::RMSNorm {
            eps: config.eps,
            scale: Some(ln1_scale.clone()),
        }));
        layers.push(CodedLayer::QkvProject {
            d_model,
            d_q,
            d_k,
            d_v,
            q_bias: opt_bias(bq),
            k_bias: opt_bias(bk),
            v_bias: opt_bias(bv),
        });
        all_weights.push(wq.clone()); // shard idx 7*i + 0
        all_weights.push(wk.clone()); // shard idx 7*i + 1
        all_weights.push(wv.clone()); // shard idx 7*i + 2
        // Insert RoPE between QkvProject and Attention
        if let Some((ref cos, ref sin)) = rope_tables {
            layers.push(CodedLayer::RoPE {
                n_heads: config.n_heads,
                n_kv_heads: config.n_kv_heads,
                head_dim: config.head_dim,
                cos_table: cos.clone(),
                sin_table: sin.clone(),
                max_seq_len: config.max_seq_len,
            });
        }
        layers.push(CodedLayer::Attention {
            n_heads: config.n_heads,
            n_kv_heads: config.n_kv_heads,
            head_dim: config.head_dim,
        });
        layers.push(CodedLayer::Linear {
            d_in: d_model,
            d_out: d_model,
            bias: opt_bias(bo),
        });
        all_weights.push(wo.clone()); // shard idx 7*i + 3
        layers.push(CodedLayer::AddResidual);

        // --- FFN half ---
        layers.push(CodedLayer::SaveResidual);
        layers.push(CodedLayer::Nonlinear(Activation::RMSNorm {
            eps: config.eps,
            scale: Some(ln2_scale.clone()),
        }));
        layers.push(CodedLayer::SwiGluUp {
            d_model,
            ff_dim,
            gate_bias: opt_bias(b_gate),
            up_bias: opt_bias(b_up),
        });
        all_weights.push(w_gate.clone()); // shard idx 7*i + 4
        all_weights.push(w_up.clone());   // shard idx 7*i + 5
        layers.push(CodedLayer::Linear {
            d_in: ff_dim,
            d_out: d_model,
            bias: opt_bias(b_down),
        });
        all_weights.push(w_down.clone()); // shard idx 7*i + 6
        layers.push(CodedLayer::AddResidual);
    }

    // ln_final + lm_head
    layers.push(CodedLayer::Nonlinear(Activation::RMSNorm {
        eps: config.eps,
        scale: Some(weights.ln_final_scale.clone()),
    }));
    layers.push(CodedLayer::Linear {
        d_in: d_model,
        d_out: config.vocab_size,
        bias: opt_bias(&weights.lm_head_bias),
    });
    all_weights.push(weights.lm_head_weight.clone()); // shard idx 7*N

    (layers, all_weights)
}

/// Embed tokens: lookup into `[vocab_size, d_model]` table.
///
/// Returns `[seq_len, d_model]` flattened.
pub fn embed_tokens(tokens: &[u32], embed_table: &[f32], d_model: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(tokens.len() * d_model);
    for &tok in tokens {
        let off = tok as usize * d_model;
        out.extend_from_slice(&embed_table[off..off + d_model]);
    }
    out
}

/// Convert a bias vec to `Option` — returns `None` if all zeros.
fn opt_bias(bias: &[f32]) -> Option<Vec<f32>> {
    if bias.iter().all(|&b| b == 0.0) {
        None
    } else {
        Some(bias.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coded::{CodedModel, Generator};
    use crate::mesh::NodeId;
    use crate::protocol::{WireGraph, WireNode, PROTOCOL_VERSION};
    use crate::worker::Worker;

    fn simple_add_graph() -> WireGraph {
        WireGraph {
            version: PROTOCOL_VERSION,
            nodes: vec![
                WireNode::Lit(0.0_f64.to_bits()),
                WireNode::Lit(1.0_f64.to_bits()),
                WireNode::Lit(2.0_f64.to_bits()),
                WireNode::Var(0),
                WireNode::Var(1),
                WireNode::Add(3, 4),
            ],
            outputs: vec![5],
            n_inputs: 2,
        }
    }

    #[tokio::test]
    async fn single_node_inference() {
        let coordinator = Coordinator::new();
        let worker = Worker::new();
        coordinator
            .add_worker(NodeId(0), worker.spawn_channel())
            .await;

        let mut server = InferenceServer::from_coordinator(coordinator);
        let graph = simple_add_graph();
        let mesh = Mesh::mock(1);
        server.load_model("test", graph, &mesh).await.unwrap();

        let result = server.infer("test", vec![3.0, 4.0]).await.unwrap();
        assert!((result[0] - 7.0).abs() < 1e-5);
    }

    /// Graph designed for a clean 2-stage pipeline split:
    /// Stage 0: Var(0), Var(1), Add(0,1)
    /// Stage 1: Lit(2.0), Mul(2,3) — depends on Add from stage 0
    /// Result: (x0 + x1) * 2.0
    fn pipeline_graph() -> WireGraph {
        WireGraph {
            version: PROTOCOL_VERSION,
            nodes: vec![
                WireNode::Var(0),
                WireNode::Var(1),
                WireNode::Add(0, 1),
                WireNode::Lit(2.0_f64.to_bits()),
                WireNode::Mul(2, 3),
            ],
            outputs: vec![4],
            n_inputs: 2,
        }
    }

    #[tokio::test]
    async fn multi_node_inference() {
        let coordinator = Coordinator::new();
        let w1 = Worker::new();
        let w2 = Worker::new();
        coordinator
            .add_worker(NodeId(0), w1.spawn_channel())
            .await;
        coordinator
            .add_worker(NodeId(1), w2.spawn_channel())
            .await;

        let mut server = InferenceServer::from_coordinator(coordinator);
        let graph = pipeline_graph();
        let mesh = Mesh::mock(2);
        server.load_model("test", graph, &mesh).await.unwrap();

        // (3.0 + 4.0) * 2.0 = 14.0
        let result = server.infer("test", vec![3.0, 4.0]).await.unwrap();
        assert!((result[0] - 14.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn unload_model() {
        let coordinator = Coordinator::new();
        let worker = Worker::new();
        coordinator
            .add_worker(NodeId(0), worker.spawn_channel())
            .await;

        let mut server = InferenceServer::from_coordinator(coordinator);
        let graph = simple_add_graph();
        let mesh = Mesh::mock(1);
        server.load_model("test", graph, &mesh).await.unwrap();

        assert_eq!(server.loaded_models().len(), 1);
        assert!(server.unload_model("test"));
        assert!(server.loaded_models().is_empty());

        // Infer after unload should fail
        let result = server.infer("test", vec![3.0, 4.0]).await;
        assert!(result.is_err());
    }

    // --- Coded inference e2e tests ---

    /// Helper: set up n workers with coded models from given layer weights.
    /// Returns (coordinator, workers) — workers kept alive so channels stay open.
    async fn setup_coded_workers_multi(
        layer_weights: &[&[f32]],
        g: &Generator,
        n: usize,
    ) -> (Coordinator, Vec<Worker>) {
        let coordinator = Coordinator::new();
        let mut workers = Vec::new();
        for i in 0..n {
            let worker = Worker::new();
            let model = CodedModel::from_layer_weights(layer_weights, g, i);
            worker.set_coded_model(model).await;
            coordinator
                .add_worker(NodeId(i as u32), worker.spawn_channel())
                .await;
            workers.push(worker);
        }
        (coordinator, workers)
    }

    /// Helper: set up n workers with coded models from single-layer weights.
    async fn setup_coded_workers(
        weights: &[f32],
        g: &Generator,
        n: usize,
    ) -> (Coordinator, Vec<Worker>) {
        setup_coded_workers_multi(&[weights], g, n).await
    }

    /// Uncoded matmul: W (d_out × d_in, row-major) @ x (d_in,) → (d_out,)
    fn matmul(w: &[f32], x: &[f32], d_out: usize, d_in: usize) -> Vec<f32> {
        (0..d_out)
            .map(|r| {
                (0..d_in).map(|c| w[r * d_in + c] * x[c]).sum::<f32>()
            })
            .collect()
    }

    #[tokio::test]
    async fn coded_inference_linear_matches_uncoded() {
        // W: 2×3 weight matrix
        let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32];
        let x = vec![1.0, 0.5, -1.0f32];
        let expected = matmul(&weights, &x, 2, 3);

        let g = Generator::cauchy(4, 2);
        let (coordinator, _workers) = setup_coded_workers(&weights, &g, 4).await;

        let server = CodedInferenceServer::new(
            coordinator,
            g,
            vec![NodeId(0), NodeId(1)], // k=2 group
            vec![CodedLayer::Linear { d_in: 3, d_out: 2, bias: None }],
        );

        let result = server.infer(x).await.unwrap();
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-3,
                "output[{i}]: {got} vs {exp}"
            );
        }
    }

    #[tokio::test]
    async fn coded_inference_different_k_group() {
        // Same weights, but use nodes 1,3 instead of 0,1
        let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32];
        let x = vec![2.0, -1.0, 0.5f32];
        let expected = matmul(&weights, &x, 2, 3);

        let g = Generator::cauchy(5, 2);
        let (coordinator, _workers) = setup_coded_workers(&weights, &g, 5).await;

        let server = CodedInferenceServer::new(
            coordinator,
            g,
            vec![NodeId(1), NodeId(3)], // non-contiguous group
            vec![CodedLayer::Linear { d_in: 3, d_out: 2, bias: None }],
        );

        let result = server.infer(x).await.unwrap();
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-3,
                "output[{i}]: {got} vs {exp}"
            );
        }
    }

    #[tokio::test]
    async fn coded_inference_linear_then_gelu() {
        let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32];
        let x = vec![1.0, 0.5, -1.0f32];

        // Uncoded reference: W @ x then GELU
        let mut expected = matmul(&weights, &x, 2, 3);
        Activation::Gelu.apply(&mut expected);

        let g = Generator::cauchy(3, 2);
        let (coordinator, _workers) = setup_coded_workers(&weights, &g, 3).await;

        let server = CodedInferenceServer::new(
            coordinator,
            g,
            vec![NodeId(0), NodeId(2)],
            vec![
                CodedLayer::Linear { d_in: 3, d_out: 2, bias: None },
                CodedLayer::Nonlinear(Activation::Gelu),
            ],
        );

        let result = server.infer(x).await.unwrap();
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-3,
                "output[{i}]: {got} vs {exp}"
            );
        }
    }

    #[tokio::test]
    async fn coded_inference_fault_tolerance() {
        // n=5, k=3 — any 3 nodes can serve inference
        // d_out=3 so it splits evenly into k=3 blocks (1 row each)
        let weights = vec![
            0.5, -0.3, 0.8, // row 0
            -0.1, 0.6, 0.2, // row 1
            0.3, -0.5, 0.4, // row 2
        ]; // 3×3
        let x = vec![1.0, 2.0, -0.5f32];
        let expected = matmul(&weights, &x, 3, 3);

        let g = Generator::cauchy(5, 3);
        let (coordinator, _workers) = setup_coded_workers(&weights, &g, 5).await;

        // Group uses nodes 0,2,4 (nodes 1,3 are "dead")
        let server = CodedInferenceServer::new(
            coordinator,
            g,
            vec![NodeId(0), NodeId(2), NodeId(4)],
            vec![CodedLayer::Linear { d_in: 3, d_out: 3, bias: None }],
        );

        let result = server.infer(x).await.unwrap();
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-2,
                "output[{i}]: {got} vs {exp}"
            );
        }
    }

    #[tokio::test]
    async fn coded_infer_and_learn_reduces_loss() {
        // Single linear layer: W (2×3), train toward target
        let weights = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6f32];
        let x = vec![1.0, 0.5, -1.0f32];
        let target = vec![1.0, -1.0f32];

        let g = Generator::cauchy(3, 2);
        let (coordinator, _workers) = setup_coded_workers(&weights, &g, 3).await;

        let server = CodedInferenceServer::new(
            coordinator,
            g.clone(),
            vec![NodeId(0), NodeId(1)],
            vec![CodedLayer::Linear { d_in: 3, d_out: 2, bias: None }],
        );

        // First inference + learn
        let (output1, loss1) = server
            .infer_and_learn(x.clone(), &target)
            .await
            .unwrap();

        // Output should be W @ x before any update
        let expected_output = matmul(&weights, &x, 2, 3);
        for (i, (&got, &exp)) in output1.iter().zip(expected_output.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-3,
                "output1[{i}]: {got} vs {exp}"
            );
        }

        // Loss should be positive
        assert!(loss1 > 0.0, "loss1 should be positive: {loss1}");

        // Second inference should produce different output (weights updated)
        let (output2, loss2) = server
            .infer_and_learn(x.clone(), &target)
            .await
            .unwrap();

        // Outputs should differ (weights changed)
        let diff: f32 = output1
            .iter()
            .zip(&output2)
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-6, "outputs should differ after learning: diff={diff}");

        // Loss should decrease (gradient step moves toward target)
        assert!(
            loss2 < loss1,
            "loss should decrease: {loss2} < {loss1}"
        );
    }

    #[tokio::test]
    async fn coded_infer_and_learn_with_gelu() {
        // Linear → GELU, then learn
        let weights = vec![0.5, -0.3, 0.8, -0.1, 0.6, 0.2f32]; // 2×3
        let x = vec![1.0, 2.0, -0.5f32];
        let target = vec![0.5, 0.5f32];

        let g = Generator::cauchy(4, 2);
        let (coordinator, _workers) = setup_coded_workers(&weights, &g, 4).await;

        let server = CodedInferenceServer::new(
            coordinator,
            g,
            vec![NodeId(0), NodeId(1)],
            vec![
                CodedLayer::Linear { d_in: 3, d_out: 2, bias: None },
                CodedLayer::Nonlinear(Activation::Gelu),
            ],
        );

        let (_output, loss1) = server
            .infer_and_learn(x.clone(), &target)
            .await
            .unwrap();
        assert!(loss1 > 0.0);

        let (_output, loss2) = server
            .infer_and_learn(x.clone(), &target)
            .await
            .unwrap();
        // Loss should decrease after one step
        assert!(
            loss2 < loss1,
            "loss should decrease: {loss2} < {loss1}"
        );
    }

    #[tokio::test]
    async fn coded_inference_two_linear_layers() {
        // Linear(3→2) → GELU → Linear(2→2): two coded linear layers
        let w1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32]; // 2×3
        let w2 = vec![0.5, -0.5, -0.3, 0.7f32]; // 2×2

        let x = vec![1.0, 0.5, -1.0f32];

        // Uncoded reference
        let mut h = matmul(&w1, &x, 2, 3);
        Activation::Gelu.apply(&mut h);
        let expected = matmul(&w2, &h, 2, 2);

        let g = Generator::cauchy(4, 2);
        let (coordinator, _workers) =
            setup_coded_workers_multi(&[&w1, &w2], &g, 4).await;

        let server = CodedInferenceServer::new(
            coordinator,
            g,
            vec![NodeId(0), NodeId(1)],
            vec![
                CodedLayer::Linear { d_in: 3, d_out: 2, bias: None },
                CodedLayer::Nonlinear(Activation::Gelu),
                CodedLayer::Linear { d_in: 2, d_out: 2, bias: None },
            ],
        );

        let result = server.infer(x).await.unwrap();
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-3,
                "output[{i}]: {got} vs {exp}"
            );
        }
    }

    #[tokio::test]
    async fn coded_infer_and_learn_two_layers_reduces_loss() {
        // Linear(3→2) → GELU → Linear(2→2), train both layers
        let w1 = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6f32]; // 2×3
        let w2 = vec![0.5, -0.5, -0.3, 0.7f32]; // 2×2
        let x = vec![1.0, 0.5, -1.0f32];
        let target = vec![1.0, -1.0f32];

        let g = Generator::cauchy(3, 2);
        let (coordinator, _workers) =
            setup_coded_workers_multi(&[&w1, &w2], &g, 3).await;

        let server = CodedInferenceServer::new(
            coordinator,
            g,
            vec![NodeId(0), NodeId(1)],
            vec![
                CodedLayer::Linear { d_in: 3, d_out: 2, bias: None },
                CodedLayer::Nonlinear(Activation::Gelu),
                CodedLayer::Linear { d_in: 2, d_out: 2, bias: None },
            ],
        );

        let (_output, loss1) = server
            .infer_and_learn(x.clone(), &target)
            .await
            .unwrap();
        assert!(loss1 > 0.0, "loss1 should be positive: {loss1}");

        let (_output, loss2) = server
            .infer_and_learn(x.clone(), &target)
            .await
            .unwrap();

        assert!(
            loss2 < loss1,
            "loss should decrease: {loss2} < {loss1}"
        );
    }

    #[tokio::test]
    async fn coded_per_layer_version_tracking() {
        // Two linear layers — verify each shard tracks version independently
        let w1 = vec![1.0, 2.0, 3.0, 4.0f32]; // 2×2
        let w2 = vec![0.5, -0.5, 0.3, 0.7f32]; // 2×2
        let x = vec![1.0, -1.0f32];
        let target = vec![0.0, 0.0f32];

        let g = Generator::cauchy(3, 2);
        let (coordinator, workers) =
            setup_coded_workers_multi(&[&w1, &w2], &g, 3).await;

        let server = CodedInferenceServer::new(
            coordinator,
            g,
            vec![NodeId(0), NodeId(1)],
            vec![
                CodedLayer::Linear { d_in: 2, d_out: 2, bias: None },
                CodedLayer::Linear { d_in: 2, d_out: 2, bias: None },
            ],
        );

        // Before learning, all versions are 0
        {
            let state = workers[0].state_ref().read().await;
            let model = state.coded_model_ref().unwrap();
            assert_eq!(model.shards[0].version, 0);
            assert_eq!(model.shards[1].version, 0);
        }

        server.infer_and_learn(x.clone(), &target).await.unwrap();

        // After learning, both layers should have version 1
        {
            let state = workers[0].state_ref().read().await;
            let model = state.coded_model_ref().unwrap();
            assert_eq!(model.shards[0].version, 1);
            assert_eq!(model.shards[1].version, 1);
        }
    }

    #[tokio::test]
    async fn activation_backward_gelu() {
        // Numerical gradient check for GELU backward
        let x = vec![-1.0, 0.0, 0.5, 1.0, 2.0f32];
        let dy = vec![1.0; 5];
        let dx = Activation::Gelu.backward(&dy, &x);

        let eps = 1e-4f32;
        for i in 0..x.len() {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[i] += eps;
            x_minus[i] -= eps;
            Activation::Gelu.apply(&mut x_plus);
            Activation::Gelu.apply(&mut x_minus);
            let numerical = (x_plus[i] - x_minus[i]) / (2.0 * eps);
            assert!(
                (dx[i] - numerical).abs() < 1e-3,
                "GELU grad[{i}]: analytical={} numerical={}",
                dx[i],
                numerical
            );
        }
    }

    #[tokio::test]
    async fn activation_backward_softmax() {
        let x = vec![1.0, 2.0, 3.0f32];
        let dy = vec![1.0, 0.0, 0.0]; // gradient only through first output
        let dx = Activation::Softmax.backward(&dy, &x);

        // Numerical check
        let eps = 1e-4f32;
        for i in 0..x.len() {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[i] += eps;
            x_minus[i] -= eps;
            Activation::Softmax.apply(&mut x_plus);
            Activation::Softmax.apply(&mut x_minus);
            // dy = [1,0,0] so only first output matters
            let numerical = (x_plus[0] - x_minus[0]) / (2.0 * eps);
            assert!(
                (dx[i] - numerical).abs() < 1e-3,
                "softmax grad[{i}]: analytical={} numerical={}",
                dx[i],
                numerical
            );
        }
    }

    // --- New: RMSNorm, Attention, Transformer block tests ---

    #[tokio::test]
    async fn activation_backward_rmsnorm() {
        let x = vec![0.5, -1.0, 0.3, 0.8, -0.2f32];
        let dy = vec![1.0; 5];
        let dx = Activation::RMSNorm { eps: 1e-5, scale: None }.backward(&dy, &x);

        // RMSNorm couples all elements, so numerical grad[i] = sum_j dy[j] * d(y_j)/d(x_i)
        let eps = 1e-4f32;
        for i in 0..x.len() {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[i] += eps;
            x_minus[i] -= eps;
            Activation::RMSNorm { eps: 1e-5, scale: None }.apply(&mut x_plus);
            Activation::RMSNorm { eps: 1e-5, scale: None }.apply(&mut x_minus);
            let numerical: f32 = x_plus.iter().zip(&x_minus)
                .zip(&dy)
                .map(|((&p, &m), &d)| d * (p - m) / (2.0 * eps))
                .sum();
            assert!(
                (dx[i] - numerical).abs() < 1e-3,
                "RMSNorm grad[{i}]: analytical={} numerical={}",
                dx[i],
                numerical
            );
        }
    }

    #[tokio::test]
    async fn attention_forward_basic() {
        // seq_len=3, n_heads=2, n_kv_heads=1, head_dim=4
        let n_heads = 2;
        let n_kv_heads = 1;
        let head_dim = 4;
        let seq_len = 3;
        let d_q = n_heads * head_dim; // 8
        let d_k = n_kv_heads * head_dim; // 4
        let d_v = n_kv_heads * head_dim; // 4
        let d_qkv = d_q + d_k + d_v; // 16
        let d_model = n_heads * head_dim; // 8

        // Deterministic input
        let qkv: Vec<f32> = (0..seq_len * d_qkv)
            .map(|i| ((i as f32) * 0.1 - 2.0) * 0.3)
            .collect();

        let (output, aw, _q, _k, _v) = attention_forward(
            &qkv, seq_len, n_heads, n_kv_heads, head_dim,
        );

        // Output shape check
        assert_eq!(output.len(), seq_len * d_model);

        // Attention weights: each head has seq_len × seq_len, rows should sum to 1
        for h in 0..n_heads {
            for i in 0..seq_len {
                let row_sum: f32 = (0..seq_len)
                    .map(|j| aw[h * seq_len * seq_len + i * seq_len + j])
                    .sum();
                assert!(
                    (row_sum - 1.0).abs() < 1e-5,
                    "head {h} row {i} sum = {row_sum}"
                );
            }
        }

        // Causal: attn[i,j] = 0 for j > i
        for h in 0..n_heads {
            for i in 0..seq_len {
                for j in (i + 1)..seq_len {
                    let w = aw[h * seq_len * seq_len + i * seq_len + j];
                    assert!(
                        w < 1e-6,
                        "causal violation: head {h} attn[{i},{j}] = {w}"
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn attention_backward_numerical() {
        let n_heads = 2;
        let n_kv_heads = 1;
        let head_dim = 4;
        let seq_len = 2;
        let d_qkv = (n_heads + 2 * n_kv_heads) * head_dim;
        let d_model = n_heads * head_dim;

        let qkv: Vec<f32> = (0..seq_len * d_qkv)
            .map(|i| ((i as f32) * 0.13 - 1.5) * 0.5)
            .collect();

        let dy = vec![1.0f32; seq_len * d_model];
        let (_, aw, q, k, v) = attention_forward(&qkv, seq_len, n_heads, n_kv_heads, head_dim);
        let dqkv = attention_backward(&dy, &aw, &q, &k, &v, seq_len, n_heads, n_kv_heads, head_dim);

        // Numerical gradient check
        let eps = 1e-4f32;
        // Check a subset of indices (all would be slow)
        for i in (0..qkv.len()).step_by(3) {
            let mut qkv_p = qkv.clone();
            let mut qkv_m = qkv.clone();
            qkv_p[i] += eps;
            qkv_m[i] -= eps;
            let (out_p, _, _, _, _) = attention_forward(&qkv_p, seq_len, n_heads, n_kv_heads, head_dim);
            let (out_m, _, _, _, _) = attention_forward(&qkv_m, seq_len, n_heads, n_kv_heads, head_dim);
            // dy = all ones, so numerical grad = sum of (out_p - out_m) / (2*eps)
            let numerical: f32 = out_p.iter().zip(&out_m)
                .map(|(p, m)| (p - m) / (2.0 * eps))
                .sum();
            assert!(
                (dqkv[i] - numerical).abs() < 1e-2,
                "attn grad[{i}]: analytical={} numerical={}",
                dqkv[i],
                numerical
            );
        }
    }

    /// Batched matmul: W (d_out × d_in) @ x [seq_len, d_in] → [seq_len, d_out]
    fn matmul_batched(w: &[f32], x: &[f32], d_out: usize, d_in: usize) -> Vec<f32> {
        let seq_len = x.len() / d_in;
        let mut out = vec![0.0f32; seq_len * d_out];
        for t in 0..seq_len {
            for r in 0..d_out {
                let mut sum = 0.0f32;
                for c in 0..d_in {
                    sum += w[r * d_in + c] * x[t * d_in + c];
                }
                out[t * d_out + r] = sum;
            }
        }
        out
    }

    #[tokio::test]
    async fn transformer_block_forward_coded_matches_uncoded() {
        // d_model=8, n_heads=2, n_kv_heads=1, head_dim=4, ff_dim=12, seq_len=3
        let d_model = 8;
        let n_heads = 2;
        let n_kv_heads = 1;
        let head_dim = 4;
        let ff_dim = 12;
        let seq_len = 3;
        let d_q = n_heads * head_dim;   // 8
        let d_k = n_kv_heads * head_dim; // 4
        let d_v = n_kv_heads * head_dim; // 4

        // Generate deterministic weights for 7 linear layers
        let mut rng_seed = 42u64;
        let mut rand_weight = |rows: usize, cols: usize| -> Vec<f32> {
            (0..rows * cols)
                .map(|_| {
                    rng_seed = rng_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((rng_seed >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 0.5
                })
                .collect::<Vec<f32>>()
        };

        let wq = rand_weight(d_q, d_model);     // 0
        let wk = rand_weight(d_k, d_model);     // 1
        let wv = rand_weight(d_v, d_model);     // 2
        let wo = rand_weight(d_model, d_model);  // 3
        let w_gate = rand_weight(ff_dim, d_model); // 4
        let w_up = rand_weight(ff_dim, d_model);   // 5
        let w_down = rand_weight(d_model, ff_dim);  // 6

        // Input: [seq_len, d_model]
        let input: Vec<f32> = (0..seq_len * d_model)
            .map(|i| (i as f32 * 0.1 - 1.0) * 0.3)
            .collect();

        // --- Uncoded reference ---
        let mut x = input.clone();

        // SaveResidual
        let residual1 = x.clone();
        // RMSNorm
        Activation::RMSNorm { eps: 1e-5, scale: None }.apply(&mut x);
        // QKV project
        let q_ref = matmul_batched(&wq, &x, d_q, d_model);
        let k_ref = matmul_batched(&wk, &x, d_k, d_model);
        let v_ref = matmul_batched(&wv, &x, d_v, d_model);
        // Concat [Q;K;V] per token
        let d_qkv = d_q + d_k + d_v;
        let mut qkv_ref = vec![0.0f32; seq_len * d_qkv];
        for t in 0..seq_len {
            qkv_ref[t * d_qkv..t * d_qkv + d_q].copy_from_slice(&q_ref[t * d_q..(t + 1) * d_q]);
            qkv_ref[t * d_qkv + d_q..t * d_qkv + d_q + d_k].copy_from_slice(&k_ref[t * d_k..(t + 1) * d_k]);
            qkv_ref[t * d_qkv + d_q + d_k..t * d_qkv + d_qkv].copy_from_slice(&v_ref[t * d_v..(t + 1) * d_v]);
        }
        // Attention
        let (attn_out, _, _, _, _) = attention_forward(&qkv_ref, seq_len, n_heads, n_kv_heads, head_dim);
        // Wo
        x = matmul_batched(&wo, &attn_out, d_model, d_model);
        // AddResidual
        for (xi, ri) in x.iter_mut().zip(&residual1) { *xi += ri; }
        // SaveResidual
        let residual2 = x.clone();
        // RMSNorm
        Activation::RMSNorm { eps: 1e-5, scale: None }.apply(&mut x);
        // SwiGLU
        let gate_ref = matmul_batched(&w_gate, &x, ff_dim, d_model);
        let up_ref = matmul_batched(&w_up, &x, ff_dim, d_model);
        let swiglu_out: Vec<f32> = gate_ref.iter().zip(&up_ref)
            .map(|(&g, &u)| silu(g) * u)
            .collect();
        // Down proj
        x = matmul_batched(&w_down, &swiglu_out, d_model, ff_dim);
        // AddResidual
        for (xi, ri) in x.iter_mut().zip(&residual2) { *xi += ri; }
        let expected = x;

        // --- Coded inference ---
        let all_weights: Vec<&[f32]> = vec![
            &wq, &wk, &wv, &wo, &w_gate, &w_up, &w_down,
        ];
        let g = Generator::cauchy(4, 2);
        let (coordinator, _workers) =
            setup_coded_workers_multi(&all_weights, &g, 4).await;

        let layers = vec![
            CodedLayer::SaveResidual,
            CodedLayer::Nonlinear(Activation::RMSNorm { eps: 1e-5, scale: None }),
            CodedLayer::QkvProject { d_model, d_q, d_k, d_v, q_bias: None, k_bias: None, v_bias: None },
            CodedLayer::Attention { n_heads, n_kv_heads, head_dim },
            CodedLayer::Linear { d_in: d_model, d_out: d_model, bias: None }, // wo
            CodedLayer::AddResidual,
            CodedLayer::SaveResidual,
            CodedLayer::Nonlinear(Activation::RMSNorm { eps: 1e-5, scale: None }),
            CodedLayer::SwiGluUp { d_model, ff_dim, gate_bias: None, up_bias: None },
            CodedLayer::Linear { d_in: ff_dim, d_out: d_model, bias: None }, // down
            CodedLayer::AddResidual,
        ];

        let server = CodedInferenceServer::new(
            coordinator,
            g,
            vec![NodeId(0), NodeId(1)],
            layers,
        );

        let result = server.infer(input).await.unwrap();
        assert_eq!(result.len(), expected.len());
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 0.05,
                "output[{i}]: {got} vs {exp} (diff={})",
                (got - exp).abs()
            );
        }
    }

    #[tokio::test]
    async fn transformer_block_learning_reduces_loss() {
        let d_model = 8;
        let n_heads = 2;
        let n_kv_heads = 1;
        let head_dim = 4;
        let ff_dim = 12;
        let seq_len = 2;
        let d_q = n_heads * head_dim;
        let d_k = n_kv_heads * head_dim;
        let d_v = n_kv_heads * head_dim;

        let mut rng_seed = 123u64;
        let mut rand_weight = |rows: usize, cols: usize| -> Vec<f32> {
            (0..rows * cols)
                .map(|_| {
                    rng_seed = rng_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((rng_seed >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 0.3
                })
                .collect::<Vec<f32>>()
        };

        let wq = rand_weight(d_q, d_model);
        let wk = rand_weight(d_k, d_model);
        let wv = rand_weight(d_v, d_model);
        let wo = rand_weight(d_model, d_model);
        let w_gate = rand_weight(ff_dim, d_model);
        let w_up = rand_weight(ff_dim, d_model);
        let w_down = rand_weight(d_model, ff_dim);

        let all_weights: Vec<&[f32]> = vec![
            &wq, &wk, &wv, &wo, &w_gate, &w_up, &w_down,
        ];

        let input: Vec<f32> = (0..seq_len * d_model)
            .map(|i| (i as f32 * 0.05 - 0.4) * 0.5)
            .collect();
        let target: Vec<f32> = (0..seq_len * d_model)
            .map(|i| (i as f32 * 0.02 + 0.1) * 0.3)
            .collect();

        let g = Generator::cauchy(4, 2);
        let (coordinator, _workers) =
            setup_coded_workers_multi(&all_weights, &g, 4).await;

        let layers = vec![
            CodedLayer::SaveResidual,
            CodedLayer::Nonlinear(Activation::RMSNorm { eps: 1e-5, scale: None }),
            CodedLayer::QkvProject { d_model, d_q, d_k, d_v, q_bias: None, k_bias: None, v_bias: None },
            CodedLayer::Attention { n_heads, n_kv_heads, head_dim },
            CodedLayer::Linear { d_in: d_model, d_out: d_model, bias: None },
            CodedLayer::AddResidual,
            CodedLayer::SaveResidual,
            CodedLayer::Nonlinear(Activation::RMSNorm { eps: 1e-5, scale: None }),
            CodedLayer::SwiGluUp { d_model, ff_dim, gate_bias: None, up_bias: None },
            CodedLayer::Linear { d_in: ff_dim, d_out: d_model, bias: None },
            CodedLayer::AddResidual,
        ];

        let server = CodedInferenceServer::new(
            coordinator,
            g,
            vec![NodeId(0), NodeId(1)],
            layers,
        );

        let (_, loss1) = server
            .infer_and_learn(input.clone(), &target)
            .await
            .unwrap();
        assert!(loss1 > 0.0, "loss1 should be positive: {loss1}");

        let (_, loss2) = server
            .infer_and_learn(input.clone(), &target)
            .await
            .unwrap();

        assert!(
            loss2 < loss1,
            "loss should decrease: loss2={loss2} vs loss1={loss1}"
        );
    }

    // -----------------------------------------------------------------------
    // New tests: bias, RMSNorm scale, build_coded_transformer
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn bias_forward_applied_correctly() {
        // Linear 2×3 + bias [10, 20] → output should be matmul + bias
        let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32]; // 2×3
        let bias = vec![10.0, 20.0f32];
        let x = vec![1.0, 0.5, -1.0f32];

        let mut expected = matmul(&weights, &x, 2, 3);
        for (e, &b) in expected.iter_mut().zip(&bias) {
            *e += b;
        }

        let g = Generator::cauchy(4, 2);
        let (coordinator, _workers) = setup_coded_workers(&weights, &g, 4).await;

        let server = CodedInferenceServer::new(
            coordinator,
            g,
            vec![NodeId(0), NodeId(1)],
            vec![CodedLayer::Linear {
                d_in: 3,
                d_out: 2,
                bias: Some(bias),
            }],
        );

        let result = server.infer(x).await.unwrap();
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-3,
                "output[{i}]: {got} vs {exp}"
            );
        }
    }

    #[tokio::test]
    async fn standalone_bias_layer() {
        // Just a bias layer applied to input
        let bias = vec![1.0, 2.0, 3.0f32];
        let x = vec![10.0, 20.0, 30.0f32];
        let expected: Vec<f32> = x.iter().zip(&bias).map(|(&a, &b)| a + b).collect();

        let g = Generator::cauchy(3, 2);
        let (coordinator, _workers) = setup_coded_workers_multi(&[], &g, 3).await;

        let server = CodedInferenceServer::new(
            coordinator,
            g,
            vec![NodeId(0), NodeId(1)],
            vec![CodedLayer::Bias(bias)],
        );

        let result = server.infer(x).await.unwrap();
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "output[{i}]: {got} vs {exp}"
            );
        }
    }

    #[tokio::test]
    async fn rmsnorm_with_scale() {
        // RMSNorm with learned scale should match manual computation
        let x = vec![1.0, -2.0, 3.0, -1.0f32];
        let scale = vec![0.5, 1.0, 2.0, 0.1f32];

        // Manual: rms = sqrt(mean(x^2) + eps)
        let ms = x.iter().map(|xi| xi * xi).sum::<f32>() / x.len() as f32;
        let rms = (ms + 1e-5f32).sqrt();
        let expected: Vec<f32> = x.iter().zip(&scale)
            .map(|(&xi, &si)| (xi / rms) * si)
            .collect();

        let mut data = x.clone();
        Activation::RMSNorm { eps: 1e-5, scale: Some(scale.clone()) }.apply(&mut data);

        for (i, (&got, &exp)) in data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "rmsnorm_scale[{i}]: {got} vs {exp}"
            );
        }
    }

    #[tokio::test]
    async fn rmsnorm_with_scale_backward_numerical() {
        let x = vec![0.5, -1.0, 0.3, 0.8f32];
        let scale = vec![2.0, 0.5, 1.5, 0.1f32];
        let dy = vec![1.0; 4];
        let activation = Activation::RMSNorm { eps: 1e-5, scale: Some(scale.clone()) };
        let dx = activation.backward(&dy, &x);

        let eps_fd = 1e-4f32;
        for i in 0..x.len() {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[i] += eps_fd;
            x_minus[i] -= eps_fd;
            activation.apply(&mut x_plus);
            activation.apply(&mut x_minus);
            let numerical: f32 = x_plus.iter().zip(&x_minus)
                .zip(&dy)
                .map(|((&p, &m), &d)| d * (p - m) / (2.0 * eps_fd))
                .sum();
            assert!(
                (dx[i] - numerical).abs() < 1e-3,
                "RMSNorm+scale grad[{i}]: analytical={} numerical={}",
                dx[i], numerical
            );
        }
    }

    #[tokio::test]
    async fn build_coded_transformer_layer_count() {
        let config = TransformerConfig {
            d_model: 16,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 8,
            ff_dim: 32,
            n_layers: 2,
            vocab_size: 64,
            eps: 1e-5,
            rope_base: None,
            max_seq_len: 0,
        };

        let mut rng_seed = 42u64;
        let mut rand_vec = |len: usize| -> Vec<f32> {
            (0..len)
                .map(|_| {
                    rng_seed = rng_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((rng_seed >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 0.3
                })
                .collect()
        };

        let d_q = config.n_heads * config.head_dim;
        let d_k = config.n_kv_heads * config.head_dim;

        let weights = TransformerWeights {
            block_weights: (0..2).map(|_| [
                rand_vec(d_q * 16),  // wq
                rand_vec(d_k * 16),  // wk
                rand_vec(d_k * 16),  // wv
                rand_vec(16 * 16),   // wo
                rand_vec(32 * 16),   // gate
                rand_vec(32 * 16),   // up
                rand_vec(16 * 32),   // down
            ]).collect(),
            block_biases: (0..2).map(|_| [
                rand_vec(d_q),   // bq
                rand_vec(d_k),   // bk
                rand_vec(d_k),   // bv
                rand_vec(16),    // bo
                rand_vec(32),    // b_gate
                rand_vec(32),    // b_up
                rand_vec(16),    // b_down
            ]).collect(),
            block_norms: (0..2).map(|_| (rand_vec(16), rand_vec(16))).collect(),
            ln_final_scale: rand_vec(16),
            lm_head_weight: rand_vec(64 * 16),
            lm_head_bias: rand_vec(64),
            embed_table: rand_vec(64 * 16),
        };

        let (layers, all_ws) = build_coded_transformer(&config, &weights);

        // 7 linear layers per block * 2 blocks + 1 lm_head = 15
        assert_eq!(all_ws.len(), 15);

        // Count shard layers via CodedLayer
        let shard_count: usize = layers.iter().map(|l| match l {
            CodedLayer::Linear { .. } => 1,
            CodedLayer::QkvProject { .. } => 3,
            CodedLayer::SwiGluUp { .. } => 2,
            _ => 0,
        }).sum();
        assert_eq!(shard_count, 15);
    }

    #[tokio::test]
    async fn build_coded_transformer_forward_matches_uncoded() {
        // Small transformer: 1 layer, d_model=8, n_heads=2, n_kv_heads=1,
        // head_dim=4, ff_dim=16, vocab=32
        let d_model = 8;
        let n_heads = 2;
        let n_kv_heads = 1;
        let head_dim = 4;
        let ff_dim = 16;
        let vocab_size = 32;
        let d_q = n_heads * head_dim;
        let d_k = n_kv_heads * head_dim;

        let mut rng_seed = 77u64;
        let mut rand_vec = |len: usize| -> Vec<f32> {
            (0..len)
                .map(|_| {
                    rng_seed = rng_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((rng_seed >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 0.3
                })
                .collect()
        };

        let config = TransformerConfig {
            d_model,
            n_heads,
            n_kv_heads,
            head_dim,
            ff_dim,
            n_layers: 1,
            vocab_size,
            eps: 1e-5,
            rope_base: None,
            max_seq_len: 0,
        };

        let bq = rand_vec(d_q);
        let bk = rand_vec(d_k);
        let bv = rand_vec(d_k);
        let bo = rand_vec(d_model);
        let b_gate = rand_vec(ff_dim);
        let b_up = rand_vec(ff_dim);
        let b_down = rand_vec(d_model);
        let ln1 = rand_vec(d_model);
        let ln2 = rand_vec(d_model);
        let ln_final = rand_vec(d_model);
        let lm_head_bias = rand_vec(vocab_size);

        let wq = rand_vec(d_q * d_model);
        let wk = rand_vec(d_k * d_model);
        let wv = rand_vec(d_k * d_model);
        let wo = rand_vec(d_model * d_model);
        let w_gate = rand_vec(ff_dim * d_model);
        let w_up = rand_vec(ff_dim * d_model);
        let w_down = rand_vec(d_model * ff_dim);
        let lm_head_w = rand_vec(vocab_size * d_model);
        let embed = rand_vec(vocab_size * d_model);

        let weights = TransformerWeights {
            block_weights: vec![[wq.clone(), wk.clone(), wv.clone(), wo.clone(),
                                 w_gate.clone(), w_up.clone(), w_down.clone()]],
            block_biases: vec![[bq.clone(), bk.clone(), bv.clone(), bo.clone(),
                                b_gate.clone(), b_up.clone(), b_down.clone()]],
            block_norms: vec![(ln1.clone(), ln2.clone())],
            ln_final_scale: ln_final.clone(),
            lm_head_weight: lm_head_w.clone(),
            lm_head_bias: lm_head_bias.clone(),
            embed_table: embed.clone(),
        };

        let (layers, all_ws) = build_coded_transformer(&config, &weights);

        // --- Uncoded reference ---
        let tokens = vec![1u32, 5, 3];
        let seq_len = tokens.len();
        let mut x = embed_tokens(&tokens, &embed, d_model);

        // Block 0: attn half
        let residual1 = x.clone();
        Activation::RMSNorm { eps: 1e-5, scale: Some(ln1.clone()) }.apply(&mut x);
        let mut q_ref = matmul_batched(&wq, &x, d_q, d_model);
        let mut k_ref = matmul_batched(&wk, &x, d_k, d_model);
        let mut v_ref = matmul_batched(&wv, &x, d_k, d_model);
        apply_bias(&mut q_ref, &bq);
        apply_bias(&mut k_ref, &bk);
        apply_bias(&mut v_ref, &bv);
        let d_qkv = d_q + d_k + d_k;
        let mut qkv = vec![0.0f32; seq_len * d_qkv];
        for t in 0..seq_len {
            qkv[t * d_qkv..t * d_qkv + d_q].copy_from_slice(&q_ref[t * d_q..(t + 1) * d_q]);
            qkv[t * d_qkv + d_q..t * d_qkv + d_q + d_k].copy_from_slice(&k_ref[t * d_k..(t + 1) * d_k]);
            qkv[t * d_qkv + d_q + d_k..t * d_qkv + d_qkv].copy_from_slice(&v_ref[t * d_k..(t + 1) * d_k]);
        }
        let (attn_out, _, _, _, _) = attention_forward(&qkv, seq_len, n_heads, n_kv_heads, head_dim);
        x = matmul_batched(&wo, &attn_out, d_model, d_model);
        apply_bias(&mut x, &bo);
        for (xi, ri) in x.iter_mut().zip(&residual1) { *xi += ri; }

        // Block 0: ffn half
        let residual2 = x.clone();
        Activation::RMSNorm { eps: 1e-5, scale: Some(ln2.clone()) }.apply(&mut x);
        let mut gate_ref = matmul_batched(&w_gate, &x, ff_dim, d_model);
        let mut up_ref = matmul_batched(&w_up, &x, ff_dim, d_model);
        apply_bias(&mut gate_ref, &b_gate);
        apply_bias(&mut up_ref, &b_up);
        let swiglu_out: Vec<f32> = gate_ref.iter().zip(&up_ref)
            .map(|(&g, &u)| silu(g) * u).collect();
        x = matmul_batched(&w_down, &swiglu_out, d_model, ff_dim);
        apply_bias(&mut x, &b_down);
        for (xi, ri) in x.iter_mut().zip(&residual2) { *xi += ri; }

        // ln_final + lm_head
        Activation::RMSNorm { eps: 1e-5, scale: Some(ln_final.clone()) }.apply(&mut x);
        x = matmul_batched(&lm_head_w, &x, vocab_size, d_model);
        apply_bias(&mut x, &lm_head_bias);
        let expected = x;

        // --- Coded inference ---
        let weight_refs: Vec<&[f32]> = all_ws.iter().map(|w| w.as_slice()).collect();
        let g = Generator::cauchy(4, 2);
        let (coordinator, _workers) =
            setup_coded_workers_multi(&weight_refs, &g, 4).await;

        let server = CodedInferenceServer::new(
            coordinator,
            g,
            vec![NodeId(0), NodeId(1)],
            layers,
        );

        let input = embed_tokens(&tokens, &embed, d_model);
        let result = server.infer(input).await.unwrap();

        assert_eq!(result.len(), expected.len());
        let max_diff: f32 = result.iter().zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 0.1,
            "max diff between coded and uncoded: {max_diff}"
        );
    }

    // -----------------------------------------------------------------------
    // RoPE tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn coded_rope_matches_uncoded() {
        // 1-layer transformer with RoPE: coded vs manual reference.
        let d_model = 8;
        let n_heads = 2;
        let n_kv_heads = 1;
        let head_dim = 4;
        let ff_dim = 16;
        let vocab_size = 32;
        let rope_base = 10000.0;
        let max_seq_len = 64;
        let d_q = n_heads * head_dim;
        let d_k = n_kv_heads * head_dim;

        let mut rng_seed = 99u64;
        let mut rand_vec = |len: usize| -> Vec<f32> {
            (0..len)
                .map(|_| {
                    rng_seed = rng_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((rng_seed >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 0.3
                })
                .collect()
        };

        let config = TransformerConfig {
            d_model, n_heads, n_kv_heads, head_dim, ff_dim,
            n_layers: 1, vocab_size, eps: 1e-5,
            rope_base: Some(rope_base),
            max_seq_len,
        };

        let bq = rand_vec(d_q);
        let bk = rand_vec(d_k);
        let bv = rand_vec(d_k);
        let bo = rand_vec(d_model);
        let b_gate = rand_vec(ff_dim);
        let b_up = rand_vec(ff_dim);
        let b_down = rand_vec(d_model);
        let ln1 = rand_vec(d_model);
        let ln2 = rand_vec(d_model);
        let ln_final = rand_vec(d_model);
        let lm_head_bias = rand_vec(vocab_size);

        let wq = rand_vec(d_q * d_model);
        let wk = rand_vec(d_k * d_model);
        let wv = rand_vec(d_k * d_model);
        let wo = rand_vec(d_model * d_model);
        let w_gate = rand_vec(ff_dim * d_model);
        let w_up = rand_vec(ff_dim * d_model);
        let w_down = rand_vec(d_model * ff_dim);
        let lm_head_w = rand_vec(vocab_size * d_model);
        let embed = rand_vec(vocab_size * d_model);

        let weights = TransformerWeights {
            block_weights: vec![[wq.clone(), wk.clone(), wv.clone(), wo.clone(),
                                 w_gate.clone(), w_up.clone(), w_down.clone()]],
            block_biases: vec![[bq.clone(), bk.clone(), bv.clone(), bo.clone(),
                                b_gate.clone(), b_up.clone(), b_down.clone()]],
            block_norms: vec![(ln1.clone(), ln2.clone())],
            ln_final_scale: ln_final.clone(),
            lm_head_weight: lm_head_w.clone(),
            lm_head_bias: lm_head_bias.clone(),
            embed_table: embed.clone(),
        };

        let (layers, all_ws) = build_coded_transformer(&config, &weights);

        // --- Uncoded reference (manually apply RoPE) ---
        let tokens = vec![1u32, 5, 3];
        let seq_len = tokens.len();
        let mut x = embed_tokens(&tokens, &embed, d_model);

        let residual1 = x.clone();
        Activation::RMSNorm { eps: 1e-5, scale: Some(ln1.clone()) }.apply(&mut x);
        let mut q_ref = matmul_batched(&wq, &x, d_q, d_model);
        let mut k_ref = matmul_batched(&wk, &x, d_k, d_model);
        let mut v_ref = matmul_batched(&wv, &x, d_k, d_model);
        apply_bias(&mut q_ref, &bq);
        apply_bias(&mut k_ref, &bk);
        apply_bias(&mut v_ref, &bv);

        // Concat QKV then apply RoPE
        let d_qkv = d_q + d_k + d_k;
        let mut qkv = vec![0.0f32; seq_len * d_qkv];
        for t in 0..seq_len {
            qkv[t * d_qkv..t * d_qkv + d_q].copy_from_slice(&q_ref[t * d_q..(t + 1) * d_q]);
            qkv[t * d_qkv + d_q..t * d_qkv + d_q + d_k].copy_from_slice(&k_ref[t * d_k..(t + 1) * d_k]);
            qkv[t * d_qkv + d_q + d_k..t * d_qkv + d_qkv].copy_from_slice(&v_ref[t * d_k..(t + 1) * d_k]);
        }

        let (cos_table, sin_table) = precompute_rope_tables(head_dim, max_seq_len, rope_base);
        apply_rope(&mut qkv, seq_len, n_heads, n_kv_heads, head_dim, &cos_table, &sin_table);

        let (attn_out, _, _, _, _) = attention_forward(&qkv, seq_len, n_heads, n_kv_heads, head_dim);
        x = matmul_batched(&wo, &attn_out, d_model, d_model);
        apply_bias(&mut x, &bo);
        for (xi, ri) in x.iter_mut().zip(&residual1) { *xi += ri; }

        let residual2 = x.clone();
        Activation::RMSNorm { eps: 1e-5, scale: Some(ln2.clone()) }.apply(&mut x);
        let mut gate_ref = matmul_batched(&w_gate, &x, ff_dim, d_model);
        let mut up_ref = matmul_batched(&w_up, &x, ff_dim, d_model);
        apply_bias(&mut gate_ref, &b_gate);
        apply_bias(&mut up_ref, &b_up);
        let swiglu_out: Vec<f32> = gate_ref.iter().zip(&up_ref)
            .map(|(&g, &u)| silu(g) * u).collect();
        x = matmul_batched(&w_down, &swiglu_out, d_model, ff_dim);
        apply_bias(&mut x, &b_down);
        for (xi, ri) in x.iter_mut().zip(&residual2) { *xi += ri; }

        Activation::RMSNorm { eps: 1e-5, scale: Some(ln_final.clone()) }.apply(&mut x);
        x = matmul_batched(&lm_head_w, &x, vocab_size, d_model);
        apply_bias(&mut x, &lm_head_bias);
        let expected = x;

        // --- Coded inference ---
        let weight_refs: Vec<&[f32]> = all_ws.iter().map(|w| w.as_slice()).collect();
        let g = Generator::cauchy(4, 2);
        let (coordinator, _workers) =
            setup_coded_workers_multi(&weight_refs, &g, 4).await;

        let server = CodedInferenceServer::new(
            coordinator,
            g,
            vec![NodeId(0), NodeId(1)],
            layers,
        );

        let input = embed_tokens(&tokens, &embed, d_model);
        let result = server.infer(input).await.unwrap();

        assert_eq!(result.len(), expected.len());
        let max_diff: f32 = result.iter().zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 0.1,
            "RoPE coded vs uncoded max diff: {max_diff}"
        );
    }

    #[test]
    fn rope_backward_numerical() {
        // Finite-difference gradient check through RoPE.
        let n_heads = 2;
        let n_kv_heads = 1;
        let head_dim = 4;
        let seq_len = 3;
        let rope_base = 10000.0;
        let max_seq_len = 64;
        let d_q = n_heads * head_dim;
        let d_k = n_kv_heads * head_dim;
        let d_v = n_kv_heads * head_dim;
        let d_qkv = d_q + d_k + d_v;

        let (cos_table, sin_table) = precompute_rope_tables(head_dim, max_seq_len, rope_base);

        // Input QKV
        let qkv: Vec<f32> = (0..seq_len * d_qkv)
            .map(|i| ((i as f32) * 0.13 - 1.5) * 0.5)
            .collect();

        // Forward
        let mut output = qkv.clone();
        apply_rope(&mut output, seq_len, n_heads, n_kv_heads, head_dim, &cos_table, &sin_table);

        // Backward with dy = 1.0 everywhere
        let dy = vec![1.0f32; seq_len * d_qkv];
        let mut grad = dy.clone();
        rope_backward(&mut grad, seq_len, n_heads, n_kv_heads, head_dim, &cos_table, &sin_table);

        // Numerical gradient check
        let eps = 1e-4f32;
        for i in (0..qkv.len()).step_by(3) {
            let mut qkv_p = qkv.clone();
            let mut qkv_m = qkv.clone();
            qkv_p[i] += eps;
            qkv_m[i] -= eps;
            apply_rope(&mut qkv_p, seq_len, n_heads, n_kv_heads, head_dim, &cos_table, &sin_table);
            apply_rope(&mut qkv_m, seq_len, n_heads, n_kv_heads, head_dim, &cos_table, &sin_table);
            // dy = all ones → numerical grad = sum of (out_p - out_m) / (2*eps)
            let numerical: f32 = qkv_p.iter().zip(&qkv_m)
                .map(|(p, m)| (p - m) / (2.0 * eps))
                .sum();
            assert!(
                (grad[i] - numerical).abs() < 1e-2,
                "RoPE grad[{i}]: analytical={} numerical={}",
                grad[i], numerical
            );
        }
    }
}
