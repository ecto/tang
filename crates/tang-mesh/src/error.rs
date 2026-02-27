//! Error types for tang-mesh.

use std::fmt;

/// Errors that can occur in distributed mesh operations.
#[derive(Debug)]
pub enum MeshError {
    /// Transport-level error (iroh/QUIC).
    Transport(String),
    /// RPC call failed.
    Rpc(String),
    /// Serialization/deserialization error.
    Serde(String),
    /// Protocol version mismatch.
    VersionMismatch { expected: u32, got: u32 },
    /// Node not found in mesh.
    NodeNotFound(crate::mesh::NodeId),
    /// Worker failed to compile a graph.
    CompileFailed(String),
    /// Worker failed to execute a graph.
    ExecutionFailed(String),
    /// Node is dead (health check failed).
    NodeDead(crate::mesh::NodeId),
    /// Graph partitioning failed.
    PartitionFailed(String),
    /// No workers available.
    NoWorkers,
}

impl fmt::Display for MeshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transport(msg) => write!(f, "transport error: {msg}"),
            Self::Rpc(msg) => write!(f, "rpc error: {msg}"),
            Self::Serde(msg) => write!(f, "serialization error: {msg}"),
            Self::VersionMismatch { expected, got } => {
                write!(f, "protocol version mismatch: expected {expected}, got {got}")
            }
            Self::NodeNotFound(id) => write!(f, "node {id} not found in mesh"),
            Self::CompileFailed(msg) => write!(f, "graph compilation failed: {msg}"),
            Self::ExecutionFailed(msg) => write!(f, "execution failed: {msg}"),
            Self::NodeDead(id) => write!(f, "node {id} is dead"),
            Self::PartitionFailed(msg) => write!(f, "partitioning failed: {msg}"),
            Self::NoWorkers => write!(f, "no workers available"),
        }
    }
}

impl std::error::Error for MeshError {}

impl From<iroh::endpoint::ConnectError> for MeshError {
    fn from(e: iroh::endpoint::ConnectError) -> Self {
        Self::Transport(e.to_string())
    }
}

impl From<iroh::endpoint::ConnectionError> for MeshError {
    fn from(e: iroh::endpoint::ConnectionError) -> Self {
        Self::Transport(e.to_string())
    }
}

impl From<postcard::Error> for MeshError {
    fn from(e: postcard::Error) -> Self {
        Self::Serde(e.to_string())
    }
}
