//! Placement strategy: how parameters and computation are distributed.
//!
//! Describes how a model's parameters are sharded across mesh nodes and
//! which parallelism strategy to use.

use serde::{Deserialize, Serialize};

use crate::mesh::NodeId;

/// How a single parameter tensor is distributed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ShardSpec {
    /// Full copy on every node (data parallelism).
    Replicated,
    /// Split along an axis across specific nodes (tensor parallelism).
    Split {
        axis: usize,
        nodes: Vec<NodeId>,
    },
}

/// Top-level parallelism strategy.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Strategy {
    /// Each node has a full model copy, trains on different data.
    DataParallel,
    /// Split tensors along an axis across nodes.
    TensorParallel { split_axis: usize },
    /// Different layers on different nodes, activations pipeline through.
    Pipeline { stage_assignments: Vec<(NodeId, Vec<u32>)> },
}

/// Complete placement plan for a model across a mesh.
#[derive(Clone, Debug)]
pub struct Placement {
    /// Per-parameter sharding specification.
    pub shards: Vec<ShardSpec>,
    /// Overall parallelism strategy.
    pub strategy: Strategy,
}

impl Placement {
    /// Data-parallel placement: replicate all parameters.
    pub fn data_parallel(n_params: usize) -> Self {
        Self {
            shards: vec![ShardSpec::Replicated; n_params],
            strategy: Strategy::DataParallel,
        }
    }

    /// Pipeline placement: assign stage ranges to nodes.
    pub fn pipeline(stage_assignments: Vec<(NodeId, Vec<u32>)>) -> Self {
        Self {
            shards: Vec::new(), // populated during graph partitioning
            strategy: Strategy::Pipeline { stage_assignments },
        }
    }

    /// Auto placement (currently returns data parallel).
    ///
    /// Future: Alpa-style hierarchical solver considering VRAM, bandwidth,
    /// intra-node vs inter-node topology.
    pub fn auto(n_params: usize) -> Self {
        // Start simple — data parallel is always correct.
        Self::data_parallel(n_params)
    }

    /// Whether this is a data-parallel placement.
    pub fn is_data_parallel(&self) -> bool {
        matches!(self.strategy, Strategy::DataParallel)
    }

    /// Whether this is a pipeline placement.
    pub fn is_pipeline(&self) -> bool {
        matches!(self.strategy, Strategy::Pipeline { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_parallel_placement() {
        let p = Placement::data_parallel(10);
        assert_eq!(p.shards.len(), 10);
        assert!(p.is_data_parallel());
        assert!(!p.is_pipeline());
        for s in &p.shards {
            assert_eq!(*s, ShardSpec::Replicated);
        }
    }

    #[test]
    fn pipeline_placement() {
        let p = Placement::pipeline(vec![
            (NodeId(0), vec![0, 1, 2]),
            (NodeId(1), vec![3, 4, 5]),
        ]);
        assert!(p.is_pipeline());
        assert!(!p.is_data_parallel());
    }

    #[test]
    fn auto_defaults_to_data_parallel() {
        let p = Placement::auto(5);
        assert!(p.is_data_parallel());
    }
}
