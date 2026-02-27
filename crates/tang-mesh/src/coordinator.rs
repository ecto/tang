//! Coordinator: orchestrates workers in the mesh.
//!
//! The coordinator connects to all workers, distributes expression graphs,
//! collects results, and manages the distributed training/inference lifecycle.
//! The coordinator is also a worker — it runs its own GPU.

use std::collections::HashMap;
use std::sync::Arc;

use tarpc::context;
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::error::MeshError;
use crate::mesh::{Mesh, NodeId};
use crate::protocol::WireGraph;
use crate::transport::WorkerServiceClient;

/// Coordinator that manages a mesh of workers.
pub struct Coordinator {
    /// Connected worker clients.
    workers: Arc<RwLock<HashMap<NodeId, WorkerServiceClient>>>,
    /// Next task ID.
    next_task_id: Arc<std::sync::atomic::AtomicU64>,
}

impl Coordinator {
    /// Create a new coordinator.
    pub fn new() -> Self {
        Self {
            workers: Arc::new(RwLock::new(HashMap::new())),
            next_task_id: Arc::new(std::sync::atomic::AtomicU64::new(1)),
        }
    }

    /// Register a worker client directly (for in-process / channel transport).
    pub async fn add_worker(&self, node_id: NodeId, client: WorkerServiceClient) {
        self.workers.write().await.insert(node_id, client);
    }

    /// Connect to all workers in the mesh via iroh.
    pub async fn connect_to_mesh(&self, mesh: &Mesh) -> Result<(), MeshError> {
        for node in mesh.nodes() {
            match self.connect_to_worker(mesh, node.id).await {
                Ok(()) => info!("connected to {}", node.id),
                Err(e) => warn!("failed to connect to {}: {e}", node.id),
            }
        }

        let workers = self.workers.read().await;
        if workers.is_empty() {
            return Err(MeshError::NoWorkers);
        }

        info!("connected to {} workers", workers.len());
        Ok(())
    }

    /// Connect to a single worker via iroh QUIC.
    async fn connect_to_worker(&self, mesh: &Mesh, node_id: NodeId) -> Result<(), MeshError> {
        let node = mesh.node(node_id).ok_or(MeshError::NodeNotFound(node_id))?;

        let conn = mesh.transport().connect(node.iroh_id).await?;

        // Open a bidirectional stream and bridge to tarpc via QuicStream
        let (send, recv) = conn
            .open_bi()
            .await
            .map_err(|e| MeshError::Transport(e.to_string()))?;

        let stream = crate::transport::QuicStream::new(send, recv);
        let transport = crate::transport::tarpc_transport(stream);
        let client = WorkerServiceClient::new(
            tarpc::client::Config::default(),
            transport,
        )
        .spawn();

        self.workers.write().await.insert(node_id, client);
        Ok(())
    }

    /// Allocate a unique task ID.
    fn next_task_id(&self) -> u64 {
        self.next_task_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    /// Compile a graph on all workers.
    pub async fn compile_all(&self, graph: &WireGraph) -> Result<u64, MeshError> {
        let task_id = self.next_task_id();
        let workers = self.workers.read().await;

        if workers.is_empty() {
            return Err(MeshError::NoWorkers);
        }

        let mut errors = Vec::new();
        for (node_id, client) in workers.iter() {
            let client = client.clone();
            match client
                .compile_graph(context::current(), task_id, graph.clone())
                .await
            {
                Ok(Ok(())) => info!("compiled on {node_id}"),
                Ok(Err(e)) => {
                    warn!("compile failed on {node_id}: {e}");
                    errors.push((*node_id, e));
                }
                Err(e) => {
                    warn!("rpc failed to {node_id}: {e}");
                    errors.push((*node_id, e.to_string()));
                }
            }
        }

        if errors.len() == workers.len() {
            return Err(MeshError::CompileFailed(format!(
                "all workers failed: {:?}",
                errors
            )));
        }

        Ok(task_id)
    }

    /// Compile a graph on a specific worker.
    pub async fn compile_on(
        &self,
        node_id: NodeId,
        graph: &WireGraph,
    ) -> Result<u64, MeshError> {
        let task_id = self.next_task_id();
        let workers = self.workers.read().await;

        let client = workers
            .get(&node_id)
            .ok_or(MeshError::NodeNotFound(node_id))?;

        let client = client.clone();
        match client
            .compile_graph(context::current(), task_id, graph.clone())
            .await
        {
            Ok(Ok(())) => Ok(task_id),
            Ok(Err(e)) => Err(MeshError::CompileFailed(e)),
            Err(e) => Err(MeshError::Rpc(e.to_string())),
        }
    }

    /// Execute a compiled graph on a specific worker.
    pub async fn execute_on(
        &self,
        node_id: NodeId,
        task_id: u64,
        inputs: Vec<f32>,
    ) -> Result<Vec<f32>, MeshError> {
        let workers = self.workers.read().await;
        let client = workers
            .get(&node_id)
            .ok_or(MeshError::NodeNotFound(node_id))?;

        let client = client.clone();
        match client.execute(context::current(), task_id, inputs).await {
            Ok(Ok(result)) => Ok(result),
            Ok(Err(e)) => Err(MeshError::ExecutionFailed(e)),
            Err(e) => Err(MeshError::Rpc(e.to_string())),
        }
    }

    /// Forward activations through a pipeline stage on a specific worker.
    pub async fn forward_on(
        &self,
        node_id: NodeId,
        task_id: u64,
        stage: u32,
        data: Vec<f32>,
    ) -> Result<Vec<f32>, MeshError> {
        let workers = self.workers.read().await;
        let client = workers
            .get(&node_id)
            .ok_or(MeshError::NodeNotFound(node_id))?;

        let client = client.clone();
        match client
            .forward_activations(context::current(), task_id, stage, data)
            .await
        {
            Ok(Ok(result)) => Ok(result),
            Ok(Err(e)) => Err(MeshError::ExecutionFailed(e)),
            Err(e) => Err(MeshError::Rpc(e.to_string())),
        }
    }

    /// Sync parameters to all workers.
    pub async fn sync_params_all(
        &self,
        params: Vec<(String, Vec<f32>)>,
    ) -> Result<(), MeshError> {
        let workers = self.workers.read().await;

        for (node_id, client) in workers.iter() {
            let client = client.clone();
            match client
                .sync_params(context::current(), params.clone())
                .await
            {
                Ok(Ok(())) => {}
                Ok(Err(e)) => warn!("sync_params failed on {node_id}: {e}"),
                Err(e) => warn!("rpc failed to {node_id}: {e}"),
            }
        }

        Ok(())
    }

    /// Ping all workers, returning responsive node IDs.
    pub async fn ping_all(&self) -> Vec<NodeId> {
        let workers = self.workers.read().await;
        let mut alive = Vec::new();

        for (node_id, client) in workers.iter() {
            let client = client.clone();
            match client.ping(context::current(), 1).await {
                Ok(1) => alive.push(*node_id),
                _ => warn!("{node_id} did not respond to ping"),
            }
        }

        alive
    }

    /// Shut down all workers.
    pub async fn shutdown_all(&self) {
        let workers = self.workers.read().await;
        for (node_id, client) in workers.iter() {
            let client = client.clone();
            match client.shutdown(context::current()).await {
                Ok(Ok(())) => info!("{node_id} shut down"),
                _ => warn!("failed to shut down {node_id}"),
            }
        }
    }

    /// Number of connected workers.
    pub async fn num_workers(&self) -> usize {
        self.workers.read().await.len()
    }

    /// Connected worker IDs.
    pub async fn worker_ids(&self) -> Vec<NodeId> {
        self.workers.read().await.keys().copied().collect()
    }
}

impl Default for Coordinator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
    async fn coordinator_compile_and_execute() {
        let coordinator = Coordinator::new();

        // Spawn two in-process workers
        let worker1 = Worker::new();
        let worker2 = Worker::new();
        let client1 = worker1.spawn_channel();
        let client2 = worker2.spawn_channel();

        coordinator.add_worker(NodeId(0), client1).await;
        coordinator.add_worker(NodeId(1), client2).await;

        assert_eq!(coordinator.num_workers().await, 2);

        // Compile graph on all workers
        let graph = simple_add_graph();
        let task_id = coordinator.compile_all(&graph).await.unwrap();

        // Execute on worker 0: 3.0 + 4.0 = 7.0
        let result = coordinator
            .execute_on(NodeId(0), task_id, vec![3.0, 4.0])
            .await
            .unwrap();
        assert!((result[0] - 7.0).abs() < 1e-5);

        // Execute on worker 1: same result
        let result = coordinator
            .execute_on(NodeId(1), task_id, vec![3.0, 4.0])
            .await
            .unwrap();
        assert!((result[0] - 7.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn coordinator_ping() {
        let coordinator = Coordinator::new();
        let worker = Worker::new();
        let client = worker.spawn_channel();
        coordinator.add_worker(NodeId(0), client).await;

        let alive = coordinator.ping_all().await;
        assert_eq!(alive, vec![NodeId(0)]);
    }

    #[tokio::test]
    async fn coordinator_no_workers() {
        let coordinator = Coordinator::new();
        let graph = simple_add_graph();

        let result = coordinator.compile_all(&graph).await;
        assert!(result.is_err());
    }
}
