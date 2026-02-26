use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tang::Vec3;
use tang_expr::{trace, ExprGraph, ExprId};

// ---------------------------------------------------------------------------
// Helpers: build graphs for various expression sizes
// ---------------------------------------------------------------------------

/// Build a dot-product graph: a.dot(b) with 6 vars.
fn build_dot_product() -> (ExprGraph, ExprId) {
    let mut g = ExprGraph::new();
    let x0 = g.var(0);
    let x1 = g.var(1);
    let x2 = g.var(2);
    let x3 = g.var(3);
    let x4 = g.var(4);
    let x5 = g.var(5);
    let t0 = g.mul(x0, x3);
    let t1 = g.mul(x1, x4);
    let t2 = g.mul(x2, x5);
    let s01 = g.add(t0, t1);
    let dot = g.add(s01, t2);
    (g, dot)
}

/// Build a cross-product graph: a.cross(b) with 6 vars, returning 3 outputs.
fn build_cross_product() -> (ExprGraph, [ExprId; 3]) {
    let mut g = ExprGraph::new();
    let a0 = g.var(0);
    let a1 = g.var(1);
    let a2 = g.var(2);
    let b0 = g.var(3);
    let b1 = g.var(4);
    let b2 = g.var(5);
    // cross = (a1*b2 - a2*b1, a2*b0 - a0*b2, a0*b1 - a1*b0)
    let c0 = {
        let p = g.mul(a1, b2);
        let q = g.mul(a2, b1);
        let nq = g.neg(q);
        g.add(p, nq)
    };
    let c1 = {
        let p = g.mul(a2, b0);
        let q = g.mul(a0, b2);
        let nq = g.neg(q);
        g.add(p, nq)
    };
    let c2 = {
        let p = g.mul(a0, b1);
        let q = g.mul(a1, b0);
        let nq = g.neg(q);
        g.add(p, nq)
    };
    (g, [c0, c1, c2])
}

/// Build norm(cross(a, b)) — a complex expression with 6 vars.
fn build_norm_cross() -> (ExprGraph, ExprId) {
    let (mut g, [c0, c1, c2]) = build_cross_product();
    let c0_sq = g.mul(c0, c0);
    let c1_sq = g.mul(c1, c1);
    let c2_sq = g.mul(c2, c2);
    let s01 = g.add(c0_sq, c1_sq);
    let sum = g.add(s01, c2_sq);
    let norm = g.sqrt(sum);
    (g, norm)
}

/// Build x^2 (simple: 1 var).
fn build_x_squared() -> (ExprGraph, ExprId) {
    let mut g = ExprGraph::new();
    let x = g.var(0);
    let xx = g.mul(x, x);
    (g, xx)
}

/// Direct f64 dot product for baseline comparison.
#[inline]
fn direct_dot(inputs: &[f64; 6]) -> f64 {
    inputs[0] * inputs[3] + inputs[1] * inputs[4] + inputs[2] * inputs[5]
}

/// Direct f64 norm(cross(a,b)) for baseline comparison.
#[inline]
fn direct_norm_cross(inputs: &[f64; 6]) -> f64 {
    let a0 = inputs[0];
    let a1 = inputs[1];
    let a2 = inputs[2];
    let b0 = inputs[3];
    let b1 = inputs[4];
    let b2 = inputs[5];
    let c0 = a1 * b2 - a2 * b1;
    let c1 = a2 * b0 - a0 * b2;
    let c2 = a0 * b1 - a1 * b0;
    (c0 * c0 + c1 * c1 + c2 * c2).sqrt()
}

const DOT_INPUTS: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
const NORM_INPUTS: [f64; 6] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];

// ---------------------------------------------------------------------------
// 1. Graph construction benchmarks
// ---------------------------------------------------------------------------

fn bench_graph_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("expr_graph/construction");

    group.bench_function("dot_product_direct_api", |b| {
        b.iter(|| {
            let (g, dot) = build_dot_product();
            black_box((g, dot))
        })
    });

    group.bench_function("dot_product_trace_vec3", |b| {
        b.iter(|| {
            let (g, dot) = trace(|| {
                let a = Vec3::new(ExprId::var(0), ExprId::var(1), ExprId::var(2));
                let b = Vec3::new(ExprId::var(3), ExprId::var(4), ExprId::var(5));
                a.dot(b)
            });
            black_box((g, dot))
        })
    });

    group.bench_function("cross_product_direct_api", |b| {
        b.iter(|| {
            let (g, cross) = build_cross_product();
            black_box((g, cross))
        })
    });

    group.bench_function("cross_product_trace_vec3", |b| {
        b.iter(|| {
            let (g, cross) = trace(|| {
                let a = Vec3::new(ExprId::var(0), ExprId::var(1), ExprId::var(2));
                let b = Vec3::new(ExprId::var(3), ExprId::var(4), ExprId::var(5));
                let c = a.cross(b);
                (c.x, c.y, c.z)
            });
            black_box((g, cross))
        })
    });

    group.bench_function("norm_cross", |b| {
        b.iter(|| {
            let (g, nc) = build_norm_cross();
            black_box((g, nc))
        })
    });

    group.bench_function("norm_trace_vec3", |b| {
        b.iter(|| {
            let (g, n) = trace(|| {
                let v = Vec3::new(ExprId::var(0), ExprId::var(1), ExprId::var(2));
                v.norm()
            });
            black_box((g, n))
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 2. Symbolic differentiation benchmarks
// ---------------------------------------------------------------------------

fn bench_differentiation(c: &mut Criterion) {
    let mut group = c.benchmark_group("expr_graph/differentiation");

    // Simple: d/dx(x^2)
    group.bench_function("x_squared", |b| {
        b.iter_batched(
            build_x_squared,
            |(mut g, xx)| {
                let d = g.diff(xx, 0);
                black_box(d)
            },
            criterion::BatchSize::SmallInput,
        )
    });

    // Medium: d/dx0(dot_product) — 6 vars
    group.bench_function("dot_product_d_dx0", |b| {
        b.iter_batched(
            build_dot_product,
            |(mut g, dot)| {
                let d = g.diff(dot, 0);
                black_box(d)
            },
            criterion::BatchSize::SmallInput,
        )
    });

    // Medium: full gradient of dot product (all 6 partials)
    group.bench_function("dot_product_gradient_6", |b| {
        b.iter_batched(
            build_dot_product,
            |(mut g, dot)| {
                let grads: Vec<_> = (0..6).map(|i| g.diff(dot, i)).collect();
                black_box(grads)
            },
            criterion::BatchSize::SmallInput,
        )
    });

    // Complex: d/dx0(norm(cross(a,b)))
    group.bench_function("norm_cross_d_dx0", |b| {
        b.iter_batched(
            build_norm_cross,
            |(mut g, nc)| {
                let d = g.diff(nc, 0);
                black_box(d)
            },
            criterion::BatchSize::SmallInput,
        )
    });

    // Complex: full gradient of norm_cross (all 6 partials)
    group.bench_function("norm_cross_gradient_6", |b| {
        b.iter_batched(
            build_norm_cross,
            |(mut g, nc)| {
                let grads: Vec<_> = (0..6).map(|i| g.diff(nc, i)).collect();
                black_box(grads)
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 3. Simplification benchmarks
// ---------------------------------------------------------------------------

fn bench_simplification(c: &mut Criterion) {
    let mut group = c.benchmark_group("expr_graph/simplification");

    // Simplify derivative of x^2
    group.bench_function("d_x_squared", |b| {
        b.iter_batched(
            || {
                let (mut g, xx) = build_x_squared();
                let d = g.diff(xx, 0);
                (g, d)
            },
            |(mut g, d)| {
                let s = g.simplify(d);
                black_box(s)
            },
            criterion::BatchSize::SmallInput,
        )
    });

    // Simplify gradient of dot product
    group.bench_function("d_dot_product_gradient", |b| {
        b.iter_batched(
            || {
                let (mut g, dot) = build_dot_product();
                let grads: Vec<_> = (0..6).map(|i| g.diff(dot, i)).collect();
                (g, grads)
            },
            |(mut g, grads)| {
                let simplified: Vec<_> = grads.iter().map(|&d| g.simplify(d)).collect();
                black_box(simplified)
            },
            criterion::BatchSize::SmallInput,
        )
    });

    // Simplify gradient of norm_cross (expensive)
    group.bench_function("d_norm_cross_gradient", |b| {
        b.iter_batched(
            || {
                let (mut g, nc) = build_norm_cross();
                let grads: Vec<_> = (0..6).map(|i| g.diff(nc, i)).collect();
                (g, grads)
            },
            |(mut g, grads)| {
                let simplified: Vec<_> = grads.iter().map(|&d| g.simplify(d)).collect();
                black_box(simplified)
            },
            criterion::BatchSize::SmallInput,
        )
    });

    // Measure node count reduction
    group.bench_function("norm_cross_simplify_single", |b| {
        b.iter_batched(
            || {
                let (mut g, nc) = build_norm_cross();
                let d = g.diff(nc, 0);
                (g, d)
            },
            |(mut g, d)| {
                let before = g.len();
                let s = g.simplify(d);
                let after = g.len();
                black_box((s, before, after))
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 4. Compilation benchmarks
// ---------------------------------------------------------------------------

fn bench_compilation(c: &mut Criterion) {
    let mut group = c.benchmark_group("expr_graph/compilation");

    // Time to compile dot product
    group.bench_function("compile_dot", |b| {
        let (g, dot) = build_dot_product();
        b.iter(|| {
            let f = g.compile(dot);
            black_box(f)
        })
    });

    // Time to compile norm_cross
    group.bench_function("compile_norm_cross", |b| {
        let (g, nc) = build_norm_cross();
        b.iter(|| {
            let f = g.compile(nc);
            black_box(f)
        })
    });

    // Time to compile_many (cross product: 3 outputs)
    group.bench_function("compile_many_cross_3", |b| {
        let (g, cross) = build_cross_product();
        b.iter(|| {
            let f = g.compile_many(&cross);
            black_box(f)
        })
    });

    // Time to compile a simplified derivative
    group.bench_function("compile_simplified_d_norm_cross", |b| {
        let (mut g, nc) = build_norm_cross();
        let d = g.diff(nc, 0);
        let d = g.simplify(d);
        b.iter(|| {
            let f = g.compile(d);
            black_box(f)
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 5. Evaluation benchmarks: compiled vs interpreted vs direct
// ---------------------------------------------------------------------------

fn bench_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("expr_graph/evaluation");

    // --- Dot product ---
    let (g_dot, dot_expr) = build_dot_product();
    let dot_compiled = g_dot.compile(dot_expr);

    group.bench_function("dot/direct_f64", |b| {
        b.iter(|| black_box(direct_dot(black_box(&DOT_INPUTS))))
    });

    group.bench_function("dot/compiled", |b| {
        b.iter(|| black_box(dot_compiled(black_box(&DOT_INPUTS))))
    });

    group.bench_function("dot/interpreted_eval", |b| {
        b.iter(|| black_box(g_dot.eval::<f64>(dot_expr, black_box(&DOT_INPUTS))))
    });

    // --- norm(cross(a,b)) ---
    let (g_nc, nc_expr) = build_norm_cross();
    let nc_compiled = g_nc.compile(nc_expr);

    group.bench_function("norm_cross/direct_f64", |b| {
        b.iter(|| black_box(direct_norm_cross(black_box(&NORM_INPUTS))))
    });

    group.bench_function("norm_cross/compiled", |b| {
        b.iter(|| black_box(nc_compiled(black_box(&NORM_INPUTS))))
    });

    group.bench_function("norm_cross/interpreted_eval", |b| {
        b.iter(|| black_box(g_nc.eval::<f64>(nc_expr, black_box(&NORM_INPUTS))))
    });

    // --- Derivative evaluation ---
    let (mut g_deriv, nc2) = build_norm_cross();
    let d_nc = g_deriv.diff(nc2, 0);
    let d_nc_simp = g_deriv.simplify(d_nc);
    let d_nc_compiled = g_deriv.compile(d_nc_simp);

    group.bench_function("d_norm_cross/compiled_simplified", |b| {
        b.iter(|| black_box(d_nc_compiled(black_box(&NORM_INPUTS))))
    });

    group.bench_function("d_norm_cross/interpreted_simplified", |b| {
        b.iter(|| black_box(g_deriv.eval::<f64>(d_nc_simp, black_box(&NORM_INPUTS))))
    });

    group.bench_function("d_norm_cross/interpreted_unsimplified", |b| {
        b.iter(|| black_box(g_deriv.eval::<f64>(d_nc, black_box(&NORM_INPUTS))))
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 6. WGSL generation benchmarks
// ---------------------------------------------------------------------------

fn bench_wgsl(c: &mut Criterion) {
    let mut group = c.benchmark_group("expr_graph/wgsl");

    // Small: dot product
    group.bench_function("dot_product", |b| {
        let (g, dot) = build_dot_product();
        b.iter(|| {
            let k = g.to_wgsl(&[dot], 6);
            black_box(k)
        })
    });

    // Medium: cross product (3 outputs)
    group.bench_function("cross_product_3out", |b| {
        let (g, cross) = build_cross_product();
        b.iter(|| {
            let k = g.to_wgsl(&cross, 6);
            black_box(k)
        })
    });

    // Large: norm_cross + its gradient (7 outputs)
    group.bench_function("norm_cross_with_gradient", |b| {
        let (mut g, nc) = build_norm_cross();
        let grads: Vec<_> = (0..6).map(|i| g.diff(nc, i)).collect();
        let simplified: Vec<_> = grads.iter().map(|&d| g.simplify(d)).collect();
        let mut outputs = vec![nc];
        outputs.extend_from_slice(&simplified);
        b.iter(|| {
            let k = g.to_wgsl(&outputs, 6);
            black_box(k)
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 7. End-to-end pipeline: trace -> diff -> simplify -> compile -> eval
// ---------------------------------------------------------------------------

fn bench_end_to_end(c: &mut Criterion) {
    let mut group = c.benchmark_group("expr_graph/end_to_end");

    // Full pipeline for dot product derivative
    group.bench_function("dot_d_dx0_full_pipeline", |b| {
        b.iter(|| {
            // 1. Build graph
            let (mut g, dot) = build_dot_product();
            // 2. Differentiate
            let d = g.diff(dot, 0);
            // 3. Simplify
            let d = g.simplify(d);
            // 4. Compile
            let f = g.compile(d);
            // 5. Evaluate
            let result = f(&DOT_INPUTS);
            black_box(result)
        })
    });

    // Full pipeline for norm_cross derivative
    group.bench_function("norm_cross_d_dx0_full_pipeline", |b| {
        b.iter(|| {
            let (mut g, nc) = build_norm_cross();
            let d = g.diff(nc, 0);
            let d = g.simplify(d);
            let f = g.compile(d);
            let result = f(&NORM_INPUTS);
            black_box(result)
        })
    });

    // Full pipeline: trace Vec3 dot → gradient → simplify → compile_many → eval
    group.bench_function("trace_dot_gradient_full", |b| {
        b.iter(|| {
            let (mut g, dot) = trace(|| {
                let a = Vec3::new(ExprId::var(0), ExprId::var(1), ExprId::var(2));
                let b = Vec3::new(ExprId::var(3), ExprId::var(4), ExprId::var(5));
                a.dot(b)
            });
            let grads: Vec<_> = (0..6)
                .map(|i| {
                    let d = g.diff(dot, i);
                    g.simplify(d)
                })
                .collect();
            let f = g.compile_many(&grads);
            let mut out = [0.0f64; 6];
            f(&DOT_INPUTS, &mut out);
            black_box(out)
        })
    });

    // Full pipeline: trace Vec3 norm_cross → gradient → simplify → compile_many → eval
    group.bench_function("trace_norm_cross_gradient_full", |b| {
        b.iter(|| {
            let (mut g, nc) = trace(|| {
                let a = Vec3::new(ExprId::var(0), ExprId::var(1), ExprId::var(2));
                let b = Vec3::new(ExprId::var(3), ExprId::var(4), ExprId::var(5));
                a.cross(b).norm()
            });
            let grads: Vec<_> = (0..6)
                .map(|i| {
                    let d = g.diff(nc, i);
                    g.simplify(d)
                })
                .collect();
            let f = g.compile_many(&grads);
            let mut out = [0.0f64; 6];
            f(&NORM_INPUTS, &mut out);
            black_box(out)
        })
    });

    // Pipeline + WGSL generation
    group.bench_function("norm_cross_gradient_to_wgsl", |b| {
        b.iter(|| {
            let (mut g, nc) = build_norm_cross();
            let grads: Vec<_> = (0..6)
                .map(|i| {
                    let d = g.diff(nc, i);
                    g.simplify(d)
                })
                .collect();
            let kernel = g.to_wgsl(&grads, 6);
            black_box(kernel)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_graph_construction,
    bench_differentiation,
    bench_simplification,
    bench_compilation,
    bench_evaluation,
    bench_wgsl,
    bench_end_to_end,
);
criterion_main!(benches);
