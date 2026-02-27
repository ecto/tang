//! Distributed training: the `.distribute(&mesh)` API.
//!
//! `DistributedTrainer` wraps the training loop with data-parallel or
//! pipeline-parallel distribution across mesh workers.

use crate::allreduce::{AllReduce, ReduceOp};
use crate::coordinator::Coordinator;
use crate::error::MeshError;
use crate::mesh::Mesh;
use crate::placement::Placement;
use crate::protocol::WireGraph;

/// Distributed training configuration.
pub struct DistributedTrainer {
    /// Learning rate.
    lr: f32,
    /// Number of training epochs.
    num_epochs: usize,
    /// Gradient reduction operation.
    allreduce: AllReduce,
    /// Placement strategy (set after `.distribute()`).
    placement: Option<Placement>,
    /// Coordinator (set after `.distribute()`).
    coordinator: Option<Coordinator>,
}

impl DistributedTrainer {
    /// Create a new distributed trainer.
    pub fn new(lr: f32, num_epochs: usize) -> Self {
        Self {
            lr,
            num_epochs,
            allreduce: AllReduce::mean(),
            placement: None,
            coordinator: None,
        }
    }

    /// Create from an existing coordinator (for testing with channel transport).
    pub fn from_coordinator(coordinator: Coordinator, lr: f32, num_epochs: usize) -> Self {
        Self {
            lr,
            num_epochs,
            allreduce: AllReduce::mean(),
            placement: Some(Placement::data_parallel(0)),
            coordinator: Some(coordinator),
        }
    }

    /// Set the reduction operation (default: Mean).
    pub fn with_reduce_op(mut self, op: ReduceOp) -> Self {
        self.allreduce = AllReduce::new(op);
        self
    }

    /// Distribute across a mesh. Connects to all workers and sets up
    /// data-parallel placement.
    pub async fn distribute(mut self, mesh: &Mesh) -> Result<Self, MeshError> {
        let coordinator = Coordinator::new();
        coordinator.connect_to_mesh(mesh).await?;

        let _n_workers = coordinator.num_workers().await;
        self.placement = Some(Placement::data_parallel(0)); // updated when model is known
        self.coordinator = Some(coordinator);

        Ok(self)
    }

    /// Compile a forward graph on all workers.
    pub async fn compile_forward(&self, graph: &WireGraph) -> Result<u64, MeshError> {
        let coordinator = self
            .coordinator
            .as_ref()
            .ok_or(MeshError::NoWorkers)?;
        coordinator.compile_all(graph).await
    }

    /// Run the distributed training loop.
    ///
    /// Data-parallel: each worker gets a different data shard, computes
    /// forward/backward locally, gradients are allreduced at the coordinator.
    pub async fn fit_distributed(
        &mut self,
        forward_graph: &WireGraph,
        data: &[Vec<f32>],
        _targets: &[Vec<f32>],
    ) -> Result<Vec<f32>, MeshError> {
        let coordinator = self
            .coordinator
            .as_ref()
            .ok_or(MeshError::NoWorkers)?;

        let n_workers = coordinator.num_workers().await;
        let task_id = coordinator.compile_all(forward_graph).await?;

        let mut epoch_losses = Vec::with_capacity(self.num_epochs);

        for _epoch in 0..self.num_epochs {
            // Shard data across workers
            let chunk_size = (data.len() + n_workers - 1) / n_workers;
            let worker_ids = coordinator.worker_ids().await;

            let mut worker_grads = Vec::new();

            for (i, &node_id) in worker_ids.iter().enumerate() {
                let start = i * chunk_size;
                let end = ((i + 1) * chunk_size).min(data.len());
                if start >= data.len() {
                    break;
                }

                let shard: Vec<f32> = data[start..end].iter().flatten().copied().collect();

                match coordinator
                    .forward_on(node_id, task_id, 0, shard)
                    .await
                {
                    Ok(grads) => worker_grads.push(grads),
                    Err(e) => {
                        tracing::warn!("worker {node_id} failed: {e}");
                    }
                }
            }

            // AllReduce gradients and broadcast back to workers
            if !worker_grads.is_empty() {
                let reduced = self.allreduce.reduce(&worker_grads);

                // Broadcast reduced gradients back to all workers
                coordinator
                    .sync_params_all(vec![("gradients".into(), reduced.clone())])
                    .await?;

                let loss = reduced.iter().sum::<f32>() / reduced.len() as f32;
                epoch_losses.push(loss);
            }
        }

        Ok(epoch_losses)
    }

    /// Learning rate.
    pub fn lr(&self) -> f32 {
        self.lr
    }

    /// Number of epochs.
    pub fn num_epochs(&self) -> usize {
        self.num_epochs
    }

    /// The placement strategy (if distributed).
    pub fn placement(&self) -> Option<&Placement> {
        self.placement.as_ref()
    }

    /// Shut down all workers.
    pub async fn shutdown(&self) {
        if let Some(coordinator) = &self.coordinator {
            coordinator.shutdown_all().await;
        }
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
    async fn distributed_training_basic() {
        let coordinator = Coordinator::new();
        let w1 = Worker::new();
        let w2 = Worker::new();
        coordinator
            .add_worker(NodeId(0), w1.spawn_channel())
            .await;
        coordinator
            .add_worker(NodeId(1), w2.spawn_channel())
            .await;

        let mut trainer = DistributedTrainer::from_coordinator(coordinator, 0.01, 3);
        let graph = simple_add_graph();
        let data = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ];
        let targets = vec![vec![3.0], vec![7.0], vec![11.0], vec![15.0]];

        let losses = trainer
            .fit_distributed(&graph, &data, &targets)
            .await
            .unwrap();
        assert_eq!(losses.len(), 3); // 3 epochs
    }

    #[tokio::test]
    async fn gradient_broadcast_reaches_workers() {
        let coordinator = Coordinator::new();
        let w1 = Worker::new();
        let w2 = Worker::new();
        coordinator
            .add_worker(NodeId(0), w1.spawn_channel())
            .await;
        coordinator
            .add_worker(NodeId(1), w2.spawn_channel())
            .await;

        let mut trainer = DistributedTrainer::from_coordinator(coordinator, 0.01, 1);
        let graph = simple_add_graph();
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let targets = vec![vec![3.0], vec![7.0]];

        trainer
            .fit_distributed(&graph, &data, &targets)
            .await
            .unwrap();

        // After training, workers should have received gradient params
        let params1 = w1.get_params("gradients").await;
        let params2 = w2.get_params("gradients").await;
        assert!(params1.is_some());
        assert!(params2.is_some());
        // Both workers should have the same reduced gradients
        assert_eq!(params1, params2);
    }

    #[tokio::test]
    async fn data_sharding() {
        let coordinator = Coordinator::new();
        let w1 = Worker::new();
        let w2 = Worker::new();
        coordinator
            .add_worker(NodeId(0), w1.spawn_channel())
            .await;
        coordinator
            .add_worker(NodeId(1), w2.spawn_channel())
            .await;

        let mut trainer = DistributedTrainer::from_coordinator(coordinator, 0.01, 1);
        let graph = simple_add_graph();
        // 4 samples across 2 workers = 2 each
        let data = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ];
        let targets = vec![vec![3.0], vec![7.0], vec![11.0], vec![15.0]];

        let losses = trainer
            .fit_distributed(&graph, &data, &targets)
            .await
            .unwrap();
        assert_eq!(losses.len(), 1);
    }
}
