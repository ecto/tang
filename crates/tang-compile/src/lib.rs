//! Graph optimization and compilation passes for tang.
//!
//! Builds on [`tang_expr`] to provide optimization passes for tensor
//! computation graphs:
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

pub use fusion::{FusedKernel, FusionGroup, FusionPlanner};
pub use passes::{
    EliminateDeadCode, FoldConstants, FuseElementwise, OptimizationPass, PassManager,
    SimplifyAlgebraic,
};
