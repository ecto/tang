//! Run the scalar optimization passes over a traced `tang-expr` graph and
//! compare node counts before and after.
//!
//! ```text
//! cargo run -p tang-compile --example passes
//! ```
use tang::Scalar;
use tang_compile::{
    EliminateDeadCode, FoldConstants, FuseElementwise, PassManager, SimplifyAlgebraic,
};
use tang_expr::{trace, ExprId};

fn main() {
    let (graph, y) = trace(|| {
        let x = ExprId::var(0);
        let c = ExprId::from_f64(2.0) * ExprId::from_f64(3.0);
        (x * ExprId::from_f64(1.0) + c).sin() + ExprId::from_f64(0.0)
    });

    let mut pm = PassManager::new();
    pm.add(FoldConstants);
    pm.add(SimplifyAlgebraic);
    pm.add(FuseElementwise);
    pm.add(EliminateDeadCode);
    let (opt, outs) = pm.run(&graph, &[y]);

    println!("nodes: {} -> {}", graph.len(), opt.len());
    for x in [0.0f64, 0.5, 1.0] {
        let a: f64 = graph.eval(y, &[x]);
        let b: f64 = opt.eval(outs[0], &[x]);
        println!("f({x}) = {a:.12} (optimized {b:.12})");
        assert!((a - b).abs() < 1e-12);
    }
}
