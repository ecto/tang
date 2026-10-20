//! The fusion pipeline every backend compiles the step through.

use tang_compile::tensor::{Op, TensorGraph, TensorId};

/// The fusion pipeline, keeping `barriers` materialized (tang-bench's
/// `compile_with_barriers`: the residual stream would otherwise be spliced
/// into every later reader, O(layers²) work and too many bindings).
pub fn compile(
    g: &TensorGraph,
    outputs: &[TensorId],
    barriers: &[TensorId],
) -> (TensorGraph, Vec<TensorId>) {
    let all: Vec<TensorId> = outputs.iter().chain(barriers).copied().collect();
    let (g, outs) = g.stabilize_softmax(&all);
    let (g, outs) = g.substitute_maps(&outs);
    let (g, outs) = g.eliminate_common(&outs);
    let (g, outs) = g.merge_siblings(&outs);
    let (g, outs) = g.relayout(&outs);
    let (g, outs) = g.merge_reduce_dims(&outs);
    let (mut g, mut outs) = g.split_reduce_dims(&outs);
    g.schedule_auto(&outs);
    outs.truncate(outputs.len());
    (g, outs)
}

pub fn input_node(g: &TensorGraph, k: usize) -> TensorId {
    let i = g
        .nodes()
        .iter()
        .position(|n| matches!(n.op, Op::Input(j) if j == k))
        .expect("every input survives the passes");
    TensorId::from_index(i)
}
