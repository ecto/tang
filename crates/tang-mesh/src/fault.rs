//! Fault tolerance: health monitoring and failure handling.
//!
//! `HealthMonitor` tracks node health via heartbeats.
//! `FaultHandler` handles node failure by re-partitioning and replaying.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::mesh::NodeId;

/// Health state of a node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthState {
    /// Node is responding to heartbeats.
    Healthy,
    /// Node missed heartbeats but not yet declared dead.
    Suspect,
    /// Node has been declared dead.
    Dead,
}

/// Per-node health tracking.
#[derive(Clone, Debug)]
struct NodeHealth {
    state: HealthState,
    last_seen: Instant,
    missed_beats: u32,
}

/// Health monitor that tracks node liveness via heartbeats.
pub struct HealthMonitor {
    /// Per-node health state.
    nodes: HashMap<NodeId, NodeHealth>,
    /// Number of missed heartbeats before a node is suspected.
    suspect_threshold: u32,
    /// Number of missed heartbeats before a node is declared dead.
    dead_threshold: u32,
    /// Expected heartbeat interval.
    heartbeat_interval: Duration,
}

impl HealthMonitor {
    /// Create a new health monitor.
    pub fn new(heartbeat_interval: Duration) -> Self {
        Self {
            nodes: HashMap::new(),
            suspect_threshold: 3,
            dead_threshold: 10,
            heartbeat_interval,
        }
    }

    /// Set the suspect threshold (missed heartbeats before Suspect).
    pub fn with_suspect_threshold(mut self, threshold: u32) -> Self {
        self.suspect_threshold = threshold;
        self
    }

    /// Set the dead threshold (missed heartbeats before Dead).
    pub fn with_dead_threshold(mut self, threshold: u32) -> Self {
        self.dead_threshold = threshold;
        self
    }

    /// Register a node for health tracking.
    pub fn register(&mut self, node_id: NodeId) {
        self.nodes.insert(
            node_id,
            NodeHealth {
                state: HealthState::Healthy,
                last_seen: Instant::now(),
                missed_beats: 0,
            },
        );
    }

    /// Record a successful heartbeat from a node.
    pub fn heartbeat(&mut self, node_id: NodeId) {
        if let Some(health) = self.nodes.get_mut(&node_id) {
            if health.state != HealthState::Healthy {
                info!("{node_id} recovered from {:?}", health.state);
            }
            health.state = HealthState::Healthy;
            health.last_seen = Instant::now();
            health.missed_beats = 0;
        }
    }

    /// Tick the monitor: check for missed heartbeats and update states.
    ///
    /// Call this periodically (e.g., every heartbeat interval).
    /// Returns nodes that transitioned to Dead since last tick.
    pub fn tick(&mut self) -> Vec<NodeId> {
        let now = Instant::now();
        let mut newly_dead = Vec::new();

        for (&node_id, health) in self.nodes.iter_mut() {
            if health.state == HealthState::Dead {
                continue;
            }

            let elapsed = now.duration_since(health.last_seen);
            let expected_beats =
                (elapsed.as_millis() / self.heartbeat_interval.as_millis().max(1)) as u32;

            if expected_beats > health.missed_beats {
                health.missed_beats = expected_beats;
            }

            let prev_state = health.state;

            if health.missed_beats >= self.dead_threshold {
                health.state = HealthState::Dead;
                if prev_state != HealthState::Dead {
                    warn!("{node_id} declared dead ({} missed heartbeats)", health.missed_beats);
                    newly_dead.push(node_id);
                }
            } else if health.missed_beats >= self.suspect_threshold {
                health.state = HealthState::Suspect;
                if prev_state == HealthState::Healthy {
                    warn!("{node_id} suspected ({} missed heartbeats)", health.missed_beats);
                }
            }
        }

        newly_dead
    }

    /// Get the health state of a node.
    pub fn state(&self, node_id: NodeId) -> Option<HealthState> {
        self.nodes.get(&node_id).map(|h| h.state)
    }

    /// Get all healthy nodes.
    pub fn healthy_nodes(&self) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter(|(_, h)| h.state == HealthState::Healthy)
            .map(|(&id, _)| id)
            .collect()
    }

    /// Get all dead nodes.
    pub fn dead_nodes(&self) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter(|(_, h)| h.state == HealthState::Dead)
            .map(|(&id, _)| id)
            .collect()
    }
}

/// Handles node failures by adjusting the computation graph.
pub struct FaultHandler;

impl FaultHandler {
    /// Handle a node failure.
    ///
    /// In the expression graph model, recovery is straightforward:
    /// 1. Remove dead nodes from the mesh
    /// 2. Re-partition the graph across surviving nodes
    /// 3. Recompile on affected nodes
    ///
    /// No weight movement needed — weights stay where they are.
    /// Only the computation assignment changes.
    pub fn handle_failure(
        dead_nodes: &[NodeId],
        graph: &crate::protocol::WireGraph,
        mesh: &crate::mesh::Mesh,
        strategy: &crate::placement::Strategy,
    ) -> Result<Vec<crate::partition::GraphPartition>, crate::error::MeshError> {
        info!("handling failure of {} node(s), re-partitioning", dead_nodes.len());

        let alive_mesh = mesh.without_nodes(dead_nodes);
        if alive_mesh.is_empty() {
            return Err(crate::error::MeshError::NoWorkers);
        }
        crate::partition::partition(graph, &alive_mesh, strategy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn health_lifecycle() {
        let mut monitor = HealthMonitor::new(Duration::from_millis(100))
            .with_suspect_threshold(2)
            .with_dead_threshold(5);

        let node = NodeId(0);
        monitor.register(node);

        assert_eq!(monitor.state(node), Some(HealthState::Healthy));

        // Heartbeat keeps it healthy
        monitor.heartbeat(node);
        let dead = monitor.tick();
        assert!(dead.is_empty());
        assert_eq!(monitor.state(node), Some(HealthState::Healthy));
    }

    #[test]
    fn healthy_nodes_list() {
        let mut monitor = HealthMonitor::new(Duration::from_secs(1));
        monitor.register(NodeId(0));
        monitor.register(NodeId(1));
        monitor.register(NodeId(2));

        let healthy = monitor.healthy_nodes();
        assert_eq!(healthy.len(), 3);
    }

    #[test]
    fn unregistered_node_returns_none() {
        let monitor = HealthMonitor::new(Duration::from_secs(1));
        assert_eq!(monitor.state(NodeId(99)), None);
    }

    #[test]
    fn fault_handler_filters_dead_nodes() {
        use crate::mesh::Mesh;
        use crate::placement::Strategy;
        use crate::protocol::{WireGraph, WireNode, PROTOCOL_VERSION};

        let mesh = Mesh::mock(3);
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

        // Kill node 1, re-partition across survivors
        let partitions = FaultHandler::handle_failure(
            &[NodeId(1)],
            &graph,
            &mesh,
            &Strategy::DataParallel,
        )
        .unwrap();

        // Should only have partitions for nodes 0 and 2
        assert_eq!(partitions.len(), 2);
        let partition_nodes: Vec<NodeId> = partitions.iter().map(|p| p.node).collect();
        assert!(partition_nodes.contains(&NodeId(0)));
        assert!(partition_nodes.contains(&NodeId(2)));
        assert!(!partition_nodes.contains(&NodeId(1)));
    }

    #[test]
    fn fault_handler_all_dead_returns_error() {
        use crate::mesh::Mesh;
        use crate::placement::Strategy;
        use crate::protocol::{WireGraph, PROTOCOL_VERSION};

        let mesh = Mesh::mock(2);
        let graph = WireGraph {
            version: PROTOCOL_VERSION,
            nodes: vec![],
            outputs: vec![],
            n_inputs: 0,
        };

        let result = FaultHandler::handle_failure(
            &[NodeId(0), NodeId(1)],
            &graph,
            &mesh,
            &Strategy::DataParallel,
        );
        assert!(result.is_err());
    }
}
