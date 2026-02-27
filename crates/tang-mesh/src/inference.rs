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

#[cfg(test)]
mod tests {
    use super::*;
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
}
