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
    /// RMSNorm: y = x / sqrt(mean(x^2) + eps). No learned scale/bias.
    RMSNorm { eps: f32 },
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
            Activation::RMSNorm { eps } => {
                let n = pre_activation.len() as f32;
                let ms = pre_activation.iter().map(|x| x * x).sum::<f32>() / n;
                let rms = (ms + eps).sqrt();
                let inv_rms = 1.0 / rms;
                let x_hat: Vec<f32> = pre_activation.iter().map(|&x| x * inv_rms).collect();
                let dy_xhat_mean: f32 = grad_output
                    .iter()
                    .zip(&x_hat)
                    .map(|(&dy, &xh)| dy * xh)
                    .sum::<f32>()
                    / n;
                grad_output
                    .iter()
                    .zip(&x_hat)
                    .map(|(&dy, &xh)| inv_rms * (dy - xh * dy_xhat_mean))
                    .collect()
            }
        }
    }

    /// Apply activation in-place.
    fn apply(&self, data: &mut [f32]) {
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
            Activation::RMSNorm { eps } => {
                let n = data.len() as f32;
                let ms = data.iter().map(|x| x * x).sum::<f32>() / n;
                let rms = (ms + eps).sqrt();
                for x in data.iter_mut() {
                    *x /= rms;
                }
            }
        }
    }
}

/// A layer in the coded inference pipeline.
#[derive(Clone, Debug)]
pub enum CodedLayer {
    /// Linear layer — stays fully coded (no decode needed).
    Linear { d_in: usize, d_out: usize },
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
    QkvProject {
        d_model: usize,
        d_q: usize,     // n_heads * head_dim
        d_k: usize,     // n_kv_heads * head_dim
        d_v: usize,     // n_kv_heads * head_dim
    },
    /// Grouped-query attention on decoded [Q;K;V] → [seq, d_model].
    Attention {
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    },
    /// 2 coded linears (gate + up) from SAME input + SiLU + elementwise mul.
    /// Consumes 2 consecutive shard indices.
    SwiGluUp {
        d_model: usize,
        ff_dim: usize,
    },
}

// ---------------------------------------------------------------------------
// Activation helpers
// ---------------------------------------------------------------------------

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
fn silu_backward(x: f32) -> f32 {
    let sig = 1.0 / (1.0 + (-x).exp());
    sig * (1.0 + x * (1.0 - sig))
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
fn attention_forward(
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
                CodedLayer::Linear { d_in, d_out } => {
                    x = self.coded_forward_seq(linear_idx, &x, *d_in, *d_out).await?;
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
                CodedLayer::QkvProject { d_model, d_q, d_k, d_v } => {
                    let seq_len = x.len() / d_model;
                    let q = self.coded_forward_seq(linear_idx, &x, *d_model, *d_q).await?;
                    let k = self.coded_forward_seq(linear_idx + 1, &x, *d_model, *d_k).await?;
                    let v = self.coded_forward_seq(linear_idx + 2, &x, *d_model, *d_v).await?;
                    linear_idx += 3;
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
                CodedLayer::SwiGluUp { d_model, ff_dim } => {
                    let gate = self.coded_forward_seq(linear_idx, &x, *d_model, *ff_dim).await?;
                    let up = self.coded_forward_seq(linear_idx + 1, &x, *d_model, *ff_dim).await?;
                    linear_idx += 2;
                    x = gate.iter().zip(&up).map(|(&g, &u)| silu(g) * u).collect();
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
                CodedLayer::Linear { d_in, d_out } => {
                    linear_inputs.push(x.clone());
                    x = self.coded_forward_seq(linear_idx, &x, *d_in, *d_out).await?;
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
                CodedLayer::QkvProject { d_model, d_q, d_k, d_v } => {
                    let seq_len = x.len() / d_model;
                    qkv_inputs.push(x.clone());
                    let q_out = self.coded_forward_seq(linear_idx, &x, *d_model, *d_q).await?;
                    let k_out = self.coded_forward_seq(linear_idx + 1, &x, *d_model, *d_k).await?;
                    let v_out = self.coded_forward_seq(linear_idx + 2, &x, *d_model, *d_v).await?;
                    linear_idx += 3;
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
                CodedLayer::SwiGluUp { d_model, ff_dim } => {
                    swiglu_cache.push((vec![], vec![], x.clone())); // placeholder
                    let gate = self.coded_forward_seq(linear_idx, &x, *d_model, *ff_dim).await?;
                    let up = self.coded_forward_seq(linear_idx + 1, &x, *d_model, *ff_dim).await?;
                    linear_idx += 2;
                    // Save raw gate and up for backward
                    let cache_idx = swiglu_cache.len() - 1;
                    swiglu_cache[cache_idx].0 = gate.clone();
                    swiglu_cache[cache_idx].1 = up.clone();
                    x = gate.iter().zip(&up).map(|(&g, &u)| silu(g) * u).collect();
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
                CodedLayer::Linear { d_in, d_out } => {
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
                CodedLayer::QkvProject { d_model, d_q, d_k, d_v } => {
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
                CodedLayer::SwiGluUp { d_model, ff_dim } => {
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
    fn count_linear_layers(&self) -> usize {
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
            vec![CodedLayer::Linear { d_in: 3, d_out: 2 }],
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
            vec![CodedLayer::Linear { d_in: 3, d_out: 2 }],
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
                CodedLayer::Linear { d_in: 3, d_out: 2 },
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
            vec![CodedLayer::Linear { d_in: 3, d_out: 3 }],
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
            vec![CodedLayer::Linear { d_in: 3, d_out: 2 }],
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
                CodedLayer::Linear { d_in: 3, d_out: 2 },
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
                CodedLayer::Linear { d_in: 3, d_out: 2 },
                CodedLayer::Nonlinear(Activation::Gelu),
                CodedLayer::Linear { d_in: 2, d_out: 2 },
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
                CodedLayer::Linear { d_in: 3, d_out: 2 },
                CodedLayer::Nonlinear(Activation::Gelu),
                CodedLayer::Linear { d_in: 2, d_out: 2 },
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
                CodedLayer::Linear { d_in: 2, d_out: 2 },
                CodedLayer::Linear { d_in: 2, d_out: 2 },
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
        let dx = Activation::RMSNorm { eps: 1e-5 }.backward(&dy, &x);

        // RMSNorm couples all elements, so numerical grad[i] = sum_j dy[j] * d(y_j)/d(x_i)
        let eps = 1e-4f32;
        for i in 0..x.len() {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[i] += eps;
            x_minus[i] -= eps;
            Activation::RMSNorm { eps: 1e-5 }.apply(&mut x_plus);
            Activation::RMSNorm { eps: 1e-5 }.apply(&mut x_minus);
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
        Activation::RMSNorm { eps: 1e-5 }.apply(&mut x);
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
        Activation::RMSNorm { eps: 1e-5 }.apply(&mut x);
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
            CodedLayer::Nonlinear(Activation::RMSNorm { eps: 1e-5 }),
            CodedLayer::QkvProject { d_model, d_q, d_k, d_v },
            CodedLayer::Attention { n_heads, n_kv_heads, head_dim },
            CodedLayer::Linear { d_in: d_model, d_out: d_model }, // wo
            CodedLayer::AddResidual,
            CodedLayer::SaveResidual,
            CodedLayer::Nonlinear(Activation::RMSNorm { eps: 1e-5 }),
            CodedLayer::SwiGluUp { d_model, ff_dim },
            CodedLayer::Linear { d_in: ff_dim, d_out: d_model }, // down
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
            CodedLayer::Nonlinear(Activation::RMSNorm { eps: 1e-5 }),
            CodedLayer::QkvProject { d_model, d_q, d_k, d_v },
            CodedLayer::Attention { n_heads, n_kv_heads, head_dim },
            CodedLayer::Linear { d_in: d_model, d_out: d_model },
            CodedLayer::AddResidual,
            CodedLayer::SaveResidual,
            CodedLayer::Nonlinear(Activation::RMSNorm { eps: 1e-5 }),
            CodedLayer::SwiGluUp { d_model, ff_dim },
            CodedLayer::Linear { d_in: ff_dim, d_out: d_model },
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
}
