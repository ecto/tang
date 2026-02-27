//! Worker: receives expression graphs, compiles to WGSL, executes on local GPU.
//!
//! Each worker runs on a node in the mesh. It accepts iroh connections from
//! the coordinator, serves the `WorkerService` RPC interface, and maintains
//! a cache of compiled kernels (keyed by graph hash).

use std::collections::HashMap;
use std::sync::Arc;

use tarpc::context;
use tarpc::server::{BaseChannel, Channel};
use tokio::sync::RwLock;
use tracing::info;

use tang_gpu::{GpuBuffer, GpuDevice, KernelCache};

use crate::protocol::WireGraph;
use crate::transport::WorkerService;

/// A compiled graph ready for execution.
struct CompiledGraph {
    /// The original WireGraph (needed for CPU eval fallback).
    wire_graph: WireGraph,
    /// WGSL kernel source (compiled from WireGraph). Used for GPU dispatch.
    wgsl_source: String,
    /// Number of inputs expected (used for future input validation).
    #[allow(dead_code)]
    n_inputs: usize,
    /// Number of outputs produced.
    n_outputs: usize,
}

/// Shared worker state.
pub struct WorkerState {
    /// Compiled graphs, keyed by task_id.
    compiled: HashMap<u64, CompiledGraph>,
    /// Whether a shutdown has been requested.
    shutting_down: bool,
    /// GPU device (None if no GPU available — falls back to CPU).
    device: Option<Arc<GpuDevice>>,
    /// Compiled WGSL pipeline cache for GPU dispatch.
    cache: KernelCache,
    /// Synced parameter tensors, keyed by name.
    params: HashMap<String, Vec<f32>>,
}

/// A worker node that receives graphs, compiles, and executes them.
#[derive(Clone)]
pub struct Worker {
    state: Arc<RwLock<WorkerState>>,
}

impl Worker {
    /// Create a new worker.
    ///
    /// Tries to acquire a GPU device. Falls back to CPU-only mode if
    /// no GPU is available (e.g. CI, headless servers).
    pub fn new() -> Self {
        let device = match GpuDevice::new_sync() {
            Ok(dev) => {
                info!("GPU device acquired for worker");
                Some(Arc::new(dev))
            }
            Err(e) => {
                info!("no GPU available, using CPU fallback: {e}");
                None
            }
        };
        Self {
            state: Arc::new(RwLock::new(WorkerState {
                compiled: HashMap::new(),
                shutting_down: false,
                device,
                cache: KernelCache::new(),
                params: HashMap::new(),
            })),
        }
    }

    /// Spawn a worker serving a tarpc channel transport.
    ///
    /// Returns a client that can call this worker's RPC methods.
    pub fn spawn_channel(
        &self,
    ) -> crate::transport::WorkerServiceClient {
        let (client_transport, server_transport) =
            tarpc::transport::channel::unbounded();

        let server = BaseChannel::with_defaults(server_transport);
        let handler = WorkerHandler {
            state: self.state.clone(),
        };

        tokio::spawn(async move {
            use futures_util::StreamExt;
            server
                .execute(handler.serve())
                .for_each(|response| async move {
                    tokio::spawn(response);
                })
                .await;
        });

        crate::transport::WorkerServiceClient::new(
            tarpc::client::Config::default(),
            client_transport,
        )
        .spawn()
    }

    /// Serve a single incoming QUIC connection via tarpc.
    pub async fn serve_connection(
        &self,
        conn: iroh::endpoint::Connection,
    ) -> Result<(), crate::error::MeshError> {
        let (send, recv) = conn
            .accept_bi()
            .await
            .map_err(|e| crate::error::MeshError::Transport(e.to_string()))?;
        let stream = crate::transport::QuicStream::new(send, recv);
        let transport = crate::transport::tarpc_transport(stream);
        let server = BaseChannel::with_defaults(transport);
        let handler = WorkerHandler {
            state: self.state.clone(),
        };
        tokio::spawn(async move {
            use futures_util::StreamExt;
            server
                .execute(handler.serve())
                .for_each(|response| async move {
                    tokio::spawn(response);
                })
                .await;
        });
        Ok(())
    }

    /// Accept loop: serve tarpc over incoming QUIC connections.
    pub async fn serve(
        &self,
        transport: &crate::transport::MeshTransport,
    ) -> Result<(), crate::error::MeshError> {
        while let Some(incoming) = transport.accept().await {
            let conn = incoming
                .await
                .map_err(|e| crate::error::MeshError::Transport(e.to_string()))?;
            self.serve_connection(conn).await?;
        }
        Ok(())
    }

    /// Whether shutdown has been requested.
    pub async fn is_shutting_down(&self) -> bool {
        self.state.read().await.shutting_down
    }

    /// Number of compiled graphs.
    pub async fn num_compiled(&self) -> usize {
        self.state.read().await.compiled.len()
    }

    /// Retrieve synced parameters by name.
    pub async fn get_params(&self, name: &str) -> Option<Vec<f32>> {
        self.state.read().await.params.get(name).cloned()
    }

    /// Whether this worker has a GPU device.
    pub async fn has_gpu(&self) -> bool {
        self.state.read().await.device.is_some()
    }
}

impl Default for Worker {
    fn default() -> Self {
        Self::new()
    }
}

/// tarpc service implementation for workers.
#[derive(Clone)]
struct WorkerHandler {
    state: Arc<RwLock<WorkerState>>,
}

impl WorkerService for WorkerHandler {
    async fn compile_graph(
        self,
        _ctx: context::Context,
        task_id: u64,
        graph: WireGraph,
    ) -> Result<(), String> {
        // Check protocol version
        if graph.version != crate::protocol::PROTOCOL_VERSION {
            return Err(format!(
                "protocol version mismatch: expected {}, got {}",
                crate::protocol::PROTOCOL_VERSION,
                graph.version
            ));
        }

        // Compile WireGraph → WGSL
        let kernel = graph.to_wgsl();

        let n_inputs = kernel.n_inputs;
        let n_outputs = kernel.n_outputs;

        let compiled = CompiledGraph {
            wire_graph: graph,
            wgsl_source: kernel.source,
            n_inputs,
            n_outputs,
        };

        let mut state = self.state.write().await;
        state.compiled.insert(task_id, compiled);

        info!(
            "compiled graph for task {task_id}: {n_inputs} inputs, {n_outputs} outputs"
        );

        Ok(())
    }

    async fn execute(
        self,
        _ctx: context::Context,
        task_id: u64,
        inputs: Vec<f32>,
    ) -> Result<Vec<f32>, String> {
        let mut state = self.state.write().await;
        let compiled = state
            .compiled
            .get(&task_id)
            .ok_or_else(|| format!("no compiled graph for task {task_id}"))?;

        if let Some(device) = &state.device {
            // GPU dispatch path
            let device = device.clone();
            let wgsl = compiled.wgsl_source.clone();
            let n_outputs = compiled.n_outputs;
            let input_buf = GpuBuffer::from_slice(&device, &inputs);
            let output_buf = GpuBuffer::uninit(&device, n_outputs);
            state.cache.dispatch(&device, &wgsl, &input_buf, &output_buf, 1);
            state.cache.flush(&device);
            let results = output_buf.to_vec_sync(&device);
            Ok(results)
        } else {
            // CPU evaluation fallback using tang-expr's eval
            let (graph, outputs) = compiled.wire_graph.to_expr_graph();
            let inputs_f64: Vec<f64> = inputs.iter().map(|&x| x as f64).collect();
            let results: Vec<f64> = graph.eval_many(&outputs, &inputs_f64);
            Ok(results.into_iter().map(|x| x as f32).collect())
        }
    }

    async fn forward_activations(
        self,
        _ctx: context::Context,
        task_id: u64,
        _stage: u32,
        data: Vec<f32>,
    ) -> Result<Vec<f32>, String> {
        let mut state = self.state.write().await;
        let compiled = state
            .compiled
            .get(&task_id)
            .ok_or_else(|| format!("no compiled graph for task {task_id}"))?;

        if let Some(device) = &state.device {
            // GPU dispatch path
            let device = device.clone();
            let wgsl = compiled.wgsl_source.clone();
            let n_outputs = compiled.n_outputs;
            let input_buf = GpuBuffer::from_slice(&device, &data);
            let output_buf = GpuBuffer::uninit(&device, n_outputs);
            state.cache.dispatch(&device, &wgsl, &input_buf, &output_buf, 1);
            state.cache.flush(&device);
            let results = output_buf.to_vec_sync(&device);
            Ok(results)
        } else {
            // CPU evaluation of the subgraph with activation inputs
            let (graph, outputs) = compiled.wire_graph.to_expr_graph();
            let inputs_f64: Vec<f64> = data.iter().map(|&x| x as f64).collect();
            let results: Vec<f64> = graph.eval_many(&outputs, &inputs_f64);
            Ok(results.into_iter().map(|x| x as f32).collect())
        }
    }

    async fn sync_params(
        self,
        _ctx: context::Context,
        params: Vec<(String, Vec<f32>)>,
    ) -> Result<(), String> {
        let mut state = self.state.write().await;
        for (name, values) in params {
            state.params.insert(name, values);
        }
        info!("synced {} parameter tensors", state.params.len());
        Ok(())
    }

    async fn ping(self, _ctx: context::Context, seq: u64) -> u64 {
        seq
    }

    async fn shutdown(self, _ctx: context::Context) -> Result<(), String> {
        info!("shutdown requested");
        let mut state = self.state.write().await;
        state.shutting_down = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{WireGraph, WireNode, PROTOCOL_VERSION};

    #[tokio::test]
    async fn worker_compile_and_execute() {
        let worker = Worker::new();
        let client = worker.spawn_channel();

        // Build a simple graph: output = x0 + x1
        let graph = WireGraph {
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
        };

        // Compile
        let result = client
            .compile_graph(context::current(), 1, graph)
            .await
            .unwrap();
        assert!(result.is_ok());

        // Execute: 3.0 + 4.0 = 7.0
        let result = client
            .execute(context::current(), 1, vec![3.0, 4.0])
            .await
            .unwrap();
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!((output[0] - 7.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn worker_ping() {
        let worker = Worker::new();
        let client = worker.spawn_channel();

        let seq = client.ping(context::current(), 42).await.unwrap();
        assert_eq!(seq, 42);
    }

    #[tokio::test]
    async fn worker_shutdown() {
        let worker = Worker::new();
        let client = worker.spawn_channel();

        assert!(!worker.is_shutting_down().await);

        let result = client.shutdown(context::current()).await.unwrap();
        assert!(result.is_ok());

        assert!(worker.is_shutting_down().await);
    }

    #[tokio::test]
    async fn worker_version_mismatch() {
        let worker = Worker::new();
        let client = worker.spawn_channel();

        let graph = WireGraph {
            version: 999, // wrong version
            nodes: vec![],
            outputs: vec![],
            n_inputs: 0,
        };

        let result = client
            .compile_graph(context::current(), 1, graph)
            .await
            .unwrap();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("version mismatch"));
    }

    #[tokio::test]
    async fn worker_sync_params() {
        let worker = Worker::new();
        let client = worker.spawn_channel();

        let params = vec![
            ("weights".into(), vec![1.0, 2.0, 3.0]),
            ("bias".into(), vec![0.5]),
        ];

        let result = client
            .sync_params(context::current(), params)
            .await
            .unwrap();
        assert!(result.is_ok());

        assert_eq!(
            worker.get_params("weights").await,
            Some(vec![1.0, 2.0, 3.0])
        );
        assert_eq!(worker.get_params("bias").await, Some(vec![0.5]));
        assert_eq!(worker.get_params("nonexistent").await, None);
    }
}
