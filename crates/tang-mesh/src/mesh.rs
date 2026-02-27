//! Mesh topology: cluster description and builder.
//!
//! A `Mesh` describes the set of nodes participating in distributed computation.
//! Each `MeshNode` reports its iroh identity and per-GPU capabilities.

use std::fmt;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::error::MeshError;
use crate::transport::MeshTransport;

/// Unique identifier for a node within a mesh.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub u32);

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "node-{}", self.0)
    }
}

/// GPU capabilities for a single device.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuInfo {
    /// Adapter index on this node.
    pub index: u32,
    /// Human-readable GPU name.
    pub name: String,
    /// Total VRAM in bytes.
    pub vram_bytes: u64,
    /// Whether this is a discrete, integrated, or software GPU.
    pub device_type: GpuDeviceType,
    /// Graphics backend.
    pub backend: GpuBackend,
}

/// Simplified device type (avoids wgpu dependency in wire format).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuDeviceType {
    DiscreteGpu,
    IntegratedGpu,
    VirtualGpu,
    Cpu,
    Other,
}

/// Simplified backend (avoids wgpu dependency in wire format).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuBackend {
    Vulkan,
    Metal,
    Dx12,
    Gl,
    BrowserWebGpu,
    Other,
}

/// A node in the mesh with its identity and capabilities.
#[derive(Clone, Debug)]
pub struct MeshNode {
    /// Unique ID within this mesh.
    pub id: NodeId,
    /// iroh peer identity (public key, handles NAT traversal).
    pub iroh_id: iroh::EndpointId,
    /// Per-GPU capabilities on this node.
    pub gpus: Vec<GpuInfo>,
}

impl MeshNode {
    /// Total VRAM across all GPUs on this node.
    pub fn total_vram(&self) -> u64 {
        self.gpus.iter().map(|g| g.vram_bytes).sum()
    }
}

/// A mesh of distributed compute nodes.
pub struct Mesh {
    /// All nodes in the mesh.
    nodes: Vec<MeshNode>,
    /// Transport layer for communication (None in mock/test meshes).
    transport: Option<MeshTransport>,
    /// Interval between heartbeat pings.
    heartbeat_interval: Duration,
}

impl Mesh {
    /// Start building a mesh.
    pub fn builder() -> MeshBuilder {
        MeshBuilder::new()
    }

    /// Create a mock mesh with N nodes (for testing, no real transport).
    ///
    /// Uses deterministic keys derived from the node index.
    pub fn mock(n: usize) -> Self {
        let nodes = (0..n)
            .map(|i| {
                // Generate a valid ed25519 keypair deterministically from index
                let seed = [i as u8; 32];
                let secret = iroh::SecretKey::from_bytes(&seed);
                MeshNode {
                    id: NodeId(i as u32),
                    iroh_id: secret.public(),
                    gpus: Vec::new(),
                }
            })
            .collect();
        Self {
            nodes,
            transport: None,
            heartbeat_interval: Duration::from_secs(5),
        }
    }

    /// All nodes in the mesh.
    pub fn nodes(&self) -> &[MeshNode] {
        &self.nodes
    }

    /// Number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the mesh is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get node by ID.
    pub fn node(&self, id: NodeId) -> Option<&MeshNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// The transport layer.
    ///
    /// Panics if called on a mock mesh.
    pub fn transport(&self) -> &MeshTransport {
        self.transport.as_ref().expect("no transport on mock mesh")
    }

    /// Heartbeat interval.
    pub fn heartbeat_interval(&self) -> Duration {
        self.heartbeat_interval
    }

    /// This node's iroh EndpointId (our identity in the mesh).
    ///
    /// Panics if called on a mock mesh.
    pub fn local_node_id(&self) -> iroh::EndpointId {
        self.transport().node_id()
    }

    /// Total number of GPUs across all nodes.
    pub fn total_gpus(&self) -> usize {
        self.nodes.iter().map(|n| n.gpus.len()).sum()
    }

    /// Return a new mesh with the specified nodes removed.
    pub fn without_nodes(&self, exclude: &[NodeId]) -> Self {
        let nodes = self
            .nodes
            .iter()
            .filter(|n| !exclude.contains(&n.id))
            .cloned()
            .collect();
        Self {
            nodes,
            transport: None,
            heartbeat_interval: self.heartbeat_interval,
        }
    }
}

/// Builder for constructing a `Mesh`.
pub struct MeshBuilder {
    peers: Vec<iroh::EndpointId>,
    heartbeat_interval: Duration,
}

impl MeshBuilder {
    fn new() -> Self {
        Self {
            peers: Vec::new(),
            heartbeat_interval: Duration::from_secs(5),
        }
    }

    /// Add a peer by iroh NodeId.
    pub fn node(mut self, iroh_id: iroh::EndpointId) -> Self {
        self.peers.push(iroh_id);
        self
    }

    /// Set the heartbeat interval.
    pub fn heartbeat_interval(mut self, interval: Duration) -> Self {
        self.heartbeat_interval = interval;
        self
    }

    /// Build the mesh: creates transport, connects to peers, discovers GPUs.
    pub async fn build(self) -> Result<Mesh, MeshError> {
        let transport = MeshTransport::new().await?;

        let mut nodes = Vec::with_capacity(self.peers.len());
        for (i, iroh_id) in self.peers.into_iter().enumerate() {
            nodes.push(MeshNode {
                id: NodeId(i as u32),
                iroh_id,
                gpus: Vec::new(), // populated on handshake
            });
        }

        Ok(Mesh {
            nodes,
            transport: Some(transport),
            heartbeat_interval: self.heartbeat_interval,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_id_display() {
        assert_eq!(format!("{}", NodeId(0)), "node-0");
        assert_eq!(format!("{}", NodeId(42)), "node-42");
    }

    #[test]
    fn gpu_info_basics() {
        let gpu = GpuInfo {
            index: 0,
            name: "RTX 4090".into(),
            vram_bytes: 24 * 1024 * 1024 * 1024,
            device_type: GpuDeviceType::DiscreteGpu,
            backend: GpuBackend::Vulkan,
        };
        assert_eq!(gpu.vram_bytes, 24 * 1024 * 1024 * 1024);
    }

    #[test]
    fn mesh_node_total_vram() {
        let secret = iroh::SecretKey::from_bytes(&[42u8; 32]);
        let node = MeshNode {
            id: NodeId(0),
            iroh_id: secret.public(),
            gpus: vec![
                GpuInfo {
                    index: 0,
                    name: "GPU 0".into(),
                    vram_bytes: 8_000_000_000,
                    device_type: GpuDeviceType::DiscreteGpu,
                    backend: GpuBackend::Vulkan,
                },
                GpuInfo {
                    index: 1,
                    name: "GPU 1".into(),
                    vram_bytes: 16_000_000_000,
                    device_type: GpuDeviceType::DiscreteGpu,
                    backend: GpuBackend::Vulkan,
                },
            ],
        };
        assert_eq!(node.total_vram(), 24_000_000_000);
    }
}
