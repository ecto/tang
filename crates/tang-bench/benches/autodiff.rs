use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tang::{Dual, Scalar, Vec3, Quat};

/// Geometric test function: rotate unit-x around z by theta, return norm.
/// Norm of a rotated unit vector is always 1.0, but the compiler can't know that.
fn geometric_fn<S: Scalar>(theta: S) -> S {
    let axis = Vec3::new(S::ZERO, S::ZERO, S::ONE);
    let q = Quat::from_axis_angle(axis, theta);
    let rotated = q.rotate(Vec3::new(S::ONE, S::ZERO, S::ZERO));
    rotated.norm()
}

/// A more interesting scalar function: sin(x)^2 + cos(x*2)
fn trig_fn<S: Scalar>(x: S) -> S {
    let s = x.sin();
    let c = (x * S::TWO).cos();
    s * s + c
}

fn forward_dual_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("forward_dual/scalar");

    group.bench_function("plain_f64", |b| {
        let x = 1.5f64;
        b.iter(|| black_box(trig_fn(black_box(x))))
    });

    group.bench_function("dual_f64", |b| {
        let x = Dual::var(1.5f64);
        b.iter(|| black_box(trig_fn(black_box(x))))
    });

    group.finish();
}

fn forward_dual_vec3_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("forward_dual/vec3_chain");

    group.bench_function("plain_f64", |b| {
        let a = Vec3::new(1.0f64, 2.0, 3.0);
        let v = Vec3::new(0.5, -1.0, 0.7);
        b.iter(|| {
            let c = black_box(a).cross(black_box(v));
            let n = c.normalize();
            black_box(n.dot(a))
        })
    });

    group.bench_function("dual_f64", |b| {
        let a = Vec3::new(Dual::var(1.0f64), Dual::constant(2.0), Dual::constant(3.0));
        let v = Vec3::new(Dual::constant(0.5), Dual::constant(-1.0), Dual::constant(0.7));
        b.iter(|| {
            let c = black_box(a).cross(black_box(v));
            let n = c.normalize();
            black_box(n.dot(a))
        })
    });

    group.finish();
}

fn forward_dual_geometric(c: &mut Criterion) {
    let mut group = c.benchmark_group("forward_dual/geometric");

    group.bench_function("plain_f64", |b| {
        let theta = 1.2f64;
        b.iter(|| black_box(geometric_fn(black_box(theta))))
    });

    group.bench_function("dual_f64", |b| {
        let theta = Dual::var(1.2f64);
        b.iter(|| black_box(geometric_fn(black_box(theta))))
    });

    group.finish();
}

fn reverse_grad_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("reverse_grad/scalar");

    group.bench_function("reverse_ad", |b| {
        b.iter(|| {
            let g = tang_ad::grad(|x| {
                let s = x[0].sin();
                let c = (&x[0] + &x[0]).cos();
                &(&s * &s) + &c
            }, &[1.5]);
            black_box(g)
        })
    });

    group.finish();
}

fn finite_diff_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("finite_diff/scalar");

    group.bench_function("central_diff", |b| {
        let x = 1.5f64;
        let h = 1e-8;
        b.iter(|| {
            let fp = trig_fn(black_box(x) + h);
            let fm = trig_fn(black_box(x) - h);
            black_box((fp - fm) / (2.0 * h))
        })
    });

    group.bench_function("forward_dual", |b| {
        let x = Dual::var(1.5f64);
        b.iter(|| black_box(trig_fn(black_box(x)).dual))
    });

    group.finish();
}

fn hessian_dual_dual(c: &mut Criterion) {
    let mut group = c.benchmark_group("hessian/dual_dual");

    group.bench_function("1d", |b| {
        b.iter(|| {
            let h = tang_ad::hessian(|x| {
                let s = x[0].sin();
                let two: Dual<Dual<f64>> = Scalar::from_f64(2.0);
                let c = (x[0] * two).cos();
                s * s + c
            }, &[1.5]);
            black_box(h)
        })
    });

    group.bench_function("2d", |b| {
        b.iter(|| {
            let h = tang_ad::hessian(|x| {
                let three: Dual<Dual<f64>> = Scalar::from_f64(3.0);
                x[0] * x[0] + x[0] * x[1] * three + x[1] * x[1]
            }, &[1.0, 2.0]);
            black_box(h)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    forward_dual_scalar,
    forward_dual_vec3_chain,
    forward_dual_geometric,
    reverse_grad_scalar,
    finite_diff_scalar,
    hessian_dual_dual,
);
criterion_main!(benches);
