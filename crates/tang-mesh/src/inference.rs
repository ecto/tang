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
    decode_outputs, CompressedGrad, Generator, GradientPolicy, Shard,
};
use crate::mesh::NodeId;

/// Non-linearity type for decode/recode barriers.
#[derive(Clone, Debug)]
pub enum Activation {
    Gelu,
    LayerNorm { eps: f32 },
    Softmax,
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
    /// `linear_layer_idx` is the index into each node's shard vector.
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
            // Use NodeId value as generator row index
            coded_outputs.push((node_id.0 as usize, result));
        }
        let refs: Vec<(usize, &[f32])> = coded_outputs
            .iter()
            .map(|(i, v)| (*i, v.as_slice()))
            .collect();
        Ok(decode_outputs(&self.generator, &refs))
    }

    /// Run coded inference through the layer sequence.
    ///
    /// At linear layers, each node computes `shard @ x` in parallel (stays coded).
    /// At non-linearities, we decode → apply activation → re-encode.
    pub async fn infer(&self, input: Vec<f32>) -> Result<Vec<f32>, MeshError> {
        let mut x = input;
        let mut linear_idx = 0u32;
        for layer in &self.layers {
            match layer {
                CodedLayer::Linear { d_in, .. } => {
                    x = self.coded_forward_layer(linear_idx, &x, *d_in).await?;
                    linear_idx += 1;
                }
                CodedLayer::Nonlinear(activation) => {
                    activation.apply(&mut x);
                }
            }
        }
        Ok(x)
    }

    /// Run inference and learn: forward → MSE loss → backward → gossip updates.
    ///
    /// Every inference is a training step. One node (the coordinator) runs
    /// backprop on decoded activations, producing block gradients that are
    /// encoded and gossiped to all group members.
    ///
    /// Returns `(output, mse_loss)`.
    pub async fn infer_and_learn(
        &self,
        input: Vec<f32>,
        target: &[f32],
    ) -> Result<(Vec<f32>, f32), MeshError> {
        let k = self.generator.k;

        // Count linear layers for indexing
        let num_linear = self
            .layers
            .iter()
            .filter(|l| matches!(l, CodedLayer::Linear { .. }))
            .count();

        // --- Forward pass (save activations) ---
        let mut x = input;
        let mut linear_inputs: Vec<Vec<f32>> = Vec::new();
        let mut pre_activations: Vec<Vec<f32>> = Vec::new();
        let mut linear_idx = 0u32;

        for layer in &self.layers {
            match layer {
                CodedLayer::Linear { d_in, .. } => {
                    linear_inputs.push(x.clone());
                    x = self.coded_forward_layer(linear_idx, &x, *d_in).await?;
                    linear_idx += 1;
                }
                CodedLayer::Nonlinear(activation) => {
                    pre_activations.push(x.clone());
                    activation.apply(&mut x);
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
        // Request all shards (all layers) from k nodes
        let mut all_shards_by_node: Vec<(usize, Vec<Shard>)> = Vec::with_capacity(k);
        for node_id in &self.group {
            let shards = self.coordinator.request_shards_from(*node_id).await?;
            all_shards_by_node.push((node_id.0 as usize, shards));
        }

        // Decode weights per linear layer
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
        // weight_grads stored in reverse linear-layer order (last linear first)
        let mut weight_grads: Vec<(usize, Vec<f32>)> = Vec::new();

        for layer in self.layers.iter().rev() {
            match layer {
                CodedLayer::Nonlinear(activation) => {
                    pi -= 1;
                    grad = activation.backward(&grad, &pre_activations[pi]);
                }
                CodedLayer::Linear { d_in, d_out } => {
                    li -= 1;
                    let linear_layer_idx = li; // forward order index
                    let x_in = &linear_inputs[li];
                    let full_w = &full_weights_per_layer[linear_layer_idx];

                    // dL/dW = outer(grad, x_in) — (d_out × d_in) row-major
                    let mut dw = vec![0.0f32; d_out * d_in];
                    for r in 0..*d_out {
                        for c in 0..*d_in {
                            dw[r * d_in + c] = grad[r] * x_in[c];
                        }
                    }
                    weight_grads.push((linear_layer_idx, dw));

                    // dL/dx = W^T @ grad (propagate to earlier layers)
                    if li > 0 {
                        let mut dx = vec![0.0f32; *d_in];
                        for c in 0..*d_in {
                            for r in 0..*d_out {
                                dx[c] += full_w[r * d_in + c] * grad[r];
                            }
                        }
                        grad = dx;
                    }
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
}
