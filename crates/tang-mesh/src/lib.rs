//! tang-mesh — Distributed expression graph execution.
//!
//! Ships expression graphs over the wire instead of tensors. Each worker
//! compiles the received DAG to a fused WGSL kernel locally. Supports
//! both distributed training (gradient allreduce, data/pipeline parallelism)
//! and distributed inference (pipeline across heterogeneous devices).
//!
//! # Architecture
//!
//! ```text
//! Coordinator                    Workers
//! ┌───────────┐    WireGraph     ┌─────────┐
//! │           │ ──────────────→  │ compile  │
//! │ partition │    activations   │ to WGSL  │
//! │ + route   │ ←──────────────→ │ execute  │
//! │           │    gradients     │ on GPU   │
//! └───────────┘ ←───────────── → └─────────┘
//! ```
//!
//! # Quick start
//!
//! ```ignore
//! use tang_mesh::{Mesh, DistributedTrainer, InferenceServer};
//!
//! // Training
//! let mesh = Mesh::builder()
//!     .node(peer_id_1)
//!     .node(peer_id_2)
//!     .build().await?;
//!
//! let trainer = DistributedTrainer::new(0.01, 500)
//!     .distribute(&mesh).await?;
//!
//! // Inference
//! let server = InferenceServer::new(&mesh).await?;
//! server.load_model("my_model", forward_graph, &mesh).await?;
//! let output = server.infer("my_model", input_data).await?;
//! ```

pub mod allreduce;
pub mod coded;
pub mod coordinator;
pub mod distributed;
pub mod error;
pub mod fault;
pub mod inference;
pub mod mesh;
pub mod partition;
pub mod placement;
pub mod protocol;
pub mod transport;
pub mod worker;

// Re-exports
pub use coded::{
    CodedModel, CompressedGrad, Generator, GradientPolicy, Shard,
};
pub use allreduce::{AllReduce, ReduceOp};
pub use coordinator::Coordinator;
pub use distributed::DistributedTrainer;
pub use error::MeshError;
pub use fault::{FaultHandler, HealthMonitor, HealthState};
pub use inference::{Activation, CodedInferenceServer, CodedLayer, InferenceServer};
pub use mesh::{GpuBackend, GpuDeviceType, GpuInfo, Mesh, MeshBuilder, MeshNode, NodeId};
pub use partition::{auto_partition, partition, GraphPartition};
pub use placement::{Placement, ShardSpec, Strategy};
pub use protocol::{WireGraph, WireNode, PROTOCOL_VERSION};
pub use transport::{MeshTransport, QuicStream, ALPN};
pub use worker::Worker;
