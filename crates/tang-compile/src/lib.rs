//! Graph optimization and compilation passes for tang.
//!
//! Two layers:
//!
//! - [`tensor`]: a tensor-level IR built from two ideas — affine
//!   [`tensor::View`]s and one compute op, the comprehension
//!   `out[i] = ⊕_k body(x[view(i,k)])` whose body and monoid are scalar
//!   [`tang_expr`] graphs. Fusion is substitution, flash attention is a
//!   rewrite to the online-softmax monoid, gradients are comprehensions,
//!   dims may be symbolic until [`tensor::TensorGraph::bind`], and
//!   schedules (inline / split / tile) lower to WGSL kernels.
//! - Scalar passes over [`tang_expr`] graphs:
//!
//! - **Constant folding** — evaluate pure-constant subgraphs at compile time
//! - **Operator fusion** — fuse sequential element-wise ops into single kernels
//! - **Dead code elimination** — remove unreachable nodes
//! - **Common subexpression elimination** — deduplicate identical subgraphs
//! - **Algebraic simplification** — apply identities (x*1=x, x+0=x, etc.)
//!
//! # Example
//!
//! ```ignore
//! use tang_compile::{OptimizationPass, PassManager, FuseElementwise, FoldConstants, EliminateDeadCode};
//!
//! let mut pm = PassManager::new();
//! pm.add(FoldConstants);
//! pm.add(FuseElementwise);
//! pm.add(EliminateDeadCode);
//! let optimized = pm.run(&graph, &outputs);
//! ```

mod fusion;
mod passes;
pub mod tensor;

pub use fusion::{FusedKernel, FusionGroup, FusionPlanner};
pub use passes::{
    EliminateDeadCode, FoldConstants, FuseElementwise, OptimizationPass, PassManager,
    SimplifyAlgebraic,
};
