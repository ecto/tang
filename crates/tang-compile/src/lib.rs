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
//! Attention written naively with a symbolic sequence length, rewritten
//! to one online-softmax reduction, bound to a size and evaluated:
//!
//! ```
//! use tang::Scalar;
//! use tang_compile::tensor::{Dim, Monoid, MonoidKind, TensorGraph};
//!
//! let (n, d) = (Dim::Sym("n"), Dim::Fixed(4));
//! let mut g = TensorGraph::new();
//! let (q, k, v) = (g.input(&[n, d]), g.input(&[n, d]), g.input(&[n, d]));
//! let kt = g.transpose(k, &[1, 0]).unwrap();
//! let s = g.matmul(q, kt).unwrap();
//! let m = g.reduce(s, 1, Monoid::max()).unwrap();
//! let mb = g.broadcast(m, &[n, n]).unwrap();
//! let p = g.map(&[s, mb], |x| (x[0] - x[1]).exp()).unwrap();
//! let l = g.reduce(p, 1, Monoid::sum()).unwrap();
//! let o = g.matmul(p, v).unwrap();
//! let lb = g.broadcast(l, &[n, d]).unwrap();
//! let y = g.map(&[o, lb], |x| x[0] / x[1]).unwrap();
//!
//! let (g, outs) = g.substitute_maps(&[y]);
//! let (g, outs) = g.stabilize_softmax(&outs);
//! assert!(g.reduction_kinds(&outs).contains(&MonoidKind::OnlineSoftmax));
//!
//! let mut g = g.bind(&[("n", 8)]).unwrap();
//! g.schedule_inline(&outs);
//! let x = vec![0.1; 32];
//! let out = g.eval(&[x.clone(), x.clone(), x], &outs).remove(0);
//! assert_eq!(out.len(), 32);
//! // `g.to_wgsl(&outs)` lowers the same graph to GPU kernels.
//! ```
//!
//! Scalar-level passes over [`tang_expr`] graphs:
//!
//! ```
//! use tang::Scalar;
//! use tang_compile::{EliminateDeadCode, FoldConstants, FuseElementwise, PassManager};
//! use tang_expr::{trace, ExprId};
//!
//! let (graph, y) = trace(|| {
//!     let x = ExprId::var(0);
//!     (x + ExprId::from_f64(2.0) * ExprId::from_f64(3.0)) * ExprId::from_f64(1.0)
//! });
//! let mut pm = PassManager::new();
//! pm.add(FoldConstants);
//! pm.add(FuseElementwise);
//! pm.add(EliminateDeadCode);
//! let (optimized, outs) = pm.run(&graph, &[y]);
//! let f: f64 = optimized.eval(outs[0], &[1.0]);
//! assert!((f - 7.0).abs() < 1e-12);
//! ```

mod fusion;
mod passes;
pub mod tensor;

pub use fusion::{FusedKernel, FusionGroup, FusionPlanner};
pub use passes::{
    EliminateDeadCode, FoldConstants, FuseElementwise, OptimizationPass, PassManager,
    SimplifyAlgebraic,
};
