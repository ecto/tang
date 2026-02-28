//! Transport layer: iroh QUIC + tarpc RPC.
//!
//! `MeshTransport` wraps an iroh `Endpoint` for QUIC-based peer-to-peer
//! communication with NAT traversal. `WorkerService` defines the RPC interface
//! between coordinator and workers using tarpc. `QuicStream` bridges iroh's
//! split send/recv streams into a single `AsyncRead + AsyncWrite` for tarpc.

use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};

use iroh::endpoint::{Connection, Incoming, RecvStream, SendStream};
use iroh::Endpoint;
use pin_project::pin_project;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};

use crate::coded::{CompressedGrad, Shard};
use crate::error::MeshError;
use crate::protocol::WireGraph;

/// ALPN protocol identifier for tang-mesh.
pub const ALPN: &[u8] = b"tang-mesh/0";

/// Wraps an iroh QUIC endpoint with tang-mesh-specific connection management.
pub struct MeshTransport {
    endpoint: Endpoint,
}

impl MeshTransport {
    /// Create a new transport with default iroh configuration.
    pub async fn new() -> Result<Self, MeshError> {
        let endpoint = Endpoint::builder()
            .alpns(vec![ALPN.to_vec()])
            .bind()
            .await
            .map_err(|e| MeshError::Transport(e.to_string()))?;
        Ok(Self { endpoint })
    }

    /// Create from an existing endpoint.
    pub fn from_endpoint(endpoint: Endpoint) -> Self {
        Self { endpoint }
    }

    /// This endpoint's identity (public key).
    pub fn node_id(&self) -> iroh::EndpointId {
        self.endpoint.id()
    }

    /// Connect to a peer by their EndpointId.
    ///
    /// iroh handles NAT traversal, relay fallback, and encryption.
    pub async fn connect(&self, peer: iroh::EndpointId) -> Result<Connection, MeshError> {
        let conn = self
            .endpoint
            .connect(peer, ALPN)
            .await?;
        Ok(conn)
    }

    /// Accept an incoming connection.
    ///
    /// Returns `None` if the endpoint is closing.
    pub async fn accept(&self) -> Option<Incoming> {
        self.endpoint.accept().await
    }

    /// The underlying iroh endpoint.
    pub fn endpoint(&self) -> &Endpoint {
        &self.endpoint
    }

    /// Close the transport.
    pub async fn close(self) {
        self.endpoint.close().await;
    }
}

// ---------------------------------------------------------------------------
// QuicStream — bridge iroh QUIC to AsyncRead + AsyncWrite
// ---------------------------------------------------------------------------

/// Combines iroh QUIC send/recv streams into a single bidirectional stream
/// that implements `AsyncRead + AsyncWrite` for tarpc serde_transport.
#[pin_project]
pub struct QuicStream {
    #[pin]
    recv: RecvStream,
    #[pin]
    send: SendStream,
}

impl QuicStream {
    /// Create a new bidirectional QUIC stream.
    pub fn new(send: SendStream, recv: RecvStream) -> Self {
        Self { recv, send }
    }
}

impl AsyncRead for QuicStream {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        AsyncRead::poll_read(self.project().recv, cx, buf)
    }
}

impl AsyncWrite for QuicStream {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        AsyncWrite::poll_write(self.project().send, cx, buf)
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        AsyncWrite::poll_flush(self.project().send, cx)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        AsyncWrite::poll_shutdown(self.project().send, cx)
    }
}

/// Create a tarpc serde transport from a `QuicStream`.
///
/// Uses length-delimited framing + JSON codec for tarpc message exchange.
pub fn tarpc_transport<Item, SinkItem>(
    stream: QuicStream,
) -> tarpc::serde_transport::Transport<
    QuicStream,
    Item,
    SinkItem,
    tokio_serde::formats::Json<Item, SinkItem>,
>
where
    Item: for<'de> serde::Deserialize<'de>,
    SinkItem: serde::Serialize,
{
    tarpc::serde_transport::new(
        tokio_util::codec::length_delimited::Builder::new().new_framed(stream),
        tokio_serde::formats::Json::default(),
    )
}

// ---------------------------------------------------------------------------
// WorkerService — tarpc RPC interface
// ---------------------------------------------------------------------------

/// RPC service trait for worker nodes.
///
/// Coordinator calls these methods on each worker. tarpc gives us
/// cascading cancellation (dropped requests auto-cancel on workers)
/// and typed async methods.
#[tarpc::service]
pub trait WorkerService {
    /// Compile a WireGraph to a local WGSL kernel. Idempotent — same graph
    /// produces same compiled result (cached by graph hash).
    async fn compile_graph(task_id: u64, graph: WireGraph) -> Result<(), String>;

    /// Execute a previously compiled graph with the given inputs.
    async fn execute(task_id: u64, inputs: Vec<f32>) -> Result<Vec<f32>, String>;

    /// Forward activations through a pipeline stage.
    async fn forward_activations(
        task_id: u64,
        stage: u32,
        data: Vec<f32>,
    ) -> Result<Vec<f32>, String>;

    /// Synchronize parameter values to this worker.
    async fn sync_params(params: Vec<(String, Vec<f32>)>) -> Result<(), String>;

    /// Health check ping. Returns the same sequence number.
    async fn ping(seq: u64) -> u64;

    /// Graceful shutdown.
    async fn shutdown() -> Result<(), String>;

    // --- Coded Mesh RPCs ---

    /// Coded forward: compute `shard @ x` (partial coded matmul).
    /// `layer` identifies which layer's shard to use, `x` is the input vector,
    /// `d_in` is the input dimension for the matrix multiply.
    async fn coded_forward(layer: u32, x: Vec<f32>, d_in: u32) -> Result<Vec<f32>, String>;

    /// Apply a compressed coded gradient update to a specific layer.
    async fn coded_update(layer: u32, grad: CompressedGrad, version: u64) -> Result<(), String>;

    /// Request this node's current shards for all layers.
    async fn request_shards() -> Result<Vec<Shard>, String>;
}
