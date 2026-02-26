use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tang::{Dual, Scalar, Vec3, Quat};
use tang_la::{DMat, DVec, Lu};

// ---------------------------------------------------------------------------
// 1. Rigid-body Euler step: pos += vel*dt, vel += g*dt, loss = |pos|²
// ---------------------------------------------------------------------------

/// Generic physics step — works with f64 or Dual<f64>.
fn euler_step<S: Scalar>(params: &[S; 6]) -> S {
    let dt: S = S::from_f64(0.01);
    let gravity = Vec3::new(S::ZERO, S::from_f64(-9.81), S::ZERO);

    let mut pos = Vec3::new(params[0], params[1], params[2]);
    let mut vel = Vec3::new(params[3], params[4], params[5]);

    vel = vel + gravity * dt;
    pos = pos + vel * dt;

    pos.norm_sq()
}

fn rigid_body_gradient(c: &mut Criterion) {
    let mut group = c.benchmark_group("differentiable/rigid_body_gradient");

    let params = [1.0, 2.0, 0.5, 0.1, -0.3, 0.7];

    // tang: forward-mode Dual — one pass per parameter
    group.bench_function("tang_dual", |b| {
        b.iter(|| {
            let mut grad = [0.0f64; 6];
            for i in 0..6 {
                let mut p: [Dual<f64>; 6] = params.map(Dual::constant);
                p[i] = Dual::var(params[i]);
                grad[i] = euler_step(black_box(&p)).dual;
            }
            black_box(grad)
        })
    });

    // Baseline: central finite differences (2 evaluations per parameter)
    group.bench_function("finite_diff", |b| {
        let h = 1e-8;
        b.iter(|| {
            let mut grad = [0.0f64; 6];
            for i in 0..6 {
                let mut pp = black_box(params);
                let mut pm = black_box(params);
                pp[i] += h;
                pm[i] -= h;
                grad[i] = (euler_step(&pp) - euler_step(&pm)) / (2.0 * h);
            }
            black_box(grad)
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 2. Forward kinematics Jacobian — 3-link planar arm
// ---------------------------------------------------------------------------

/// 3-link planar FK: chain of rotations around Z + translates along local X.
fn fk_planar<S: Scalar>(angles: &[S]) -> Vec<S> {
    let link_len: S = S::ONE;
    let z_axis = Vec3::new(S::ZERO, S::ZERO, S::ONE);

    let mut tip = Vec3::<S>::zero();
    let mut orientation = Quat::<S>::identity();

    for &angle in angles {
        let q = Quat::from_axis_angle(z_axis, angle);
        orientation = orientation.mul(&q);
        let local_x = Vec3::new(link_len, S::ZERO, S::ZERO);
        tip = tip + orientation.rotate(local_x);
    }

    vec![tip.x, tip.y, tip.z]
}

fn fk_jacobian(c: &mut Criterion) {
    let mut group = c.benchmark_group("differentiable/fk_jacobian");

    let angles = [0.3, -0.5, 0.8];

    // tang: jacobian_fwd — exact Jacobian via forward-mode AD
    group.bench_function("tang_jacobian_fwd", |b| {
        b.iter(|| {
            let j = tang_ad::jacobian_fwd(
                |x| fk_planar(black_box(x)),
                black_box(&angles),
            );
            black_box(j)
        })
    });

    // Baseline: finite-difference Jacobian (2n evaluations for central diff)
    group.bench_function("finite_diff", |b| {
        let h = 1e-8;
        b.iter(|| {
            let base = fk_planar(black_box(&angles));
            let n_out = base.len();
            let n_in = angles.len();
            let mut jac = vec![0.0f64; n_out * n_in];
            for j in 0..n_in {
                let mut ap = angles;
                let mut am = angles;
                ap[j] += h;
                am[j] -= h;
                let fp = fk_planar(&ap);
                let fm = fk_planar(&am);
                for i in 0..n_out {
                    jac[i * n_in + j] = (fp[i] - fm[i]) / (2.0 * h);
                }
            }
            black_box(jac)
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 3. Differentiable LU solve — Ax = b with A(t), get dx/dt
// ---------------------------------------------------------------------------

fn differentiable_lu_solve(c: &mut Criterion) {
    let mut group = c.benchmark_group("differentiable/lu_solve");

    // A(t) = [[2+t, 1], [1, 3+t]], b = [1, 2]
    // Solve Ax = b and differentiate x w.r.t. t.

    let t_val = 0.5f64;

    // tang: Dual<f64> flows through LU — just works
    group.bench_function("tang_dual_lu", |b| {
        b.iter(|| {
            let t = Dual::var(black_box(t_val));
            let two: Dual<f64> = Scalar::from_f64(2.0);
            let three: Dual<f64> = Scalar::from_f64(3.0);
            let one: Dual<f64> = Scalar::from_f64(1.0);

            let a = DMat::from_fn(2, 2, |i, j| match (i, j) {
                (0, 0) => two + t,
                (0, 1) => one,
                (1, 0) => one,
                (1, 1) => three + t,
                _ => unreachable!(),
            });
            let bv = DVec::from_fn(2, |i| if i == 0 { one } else { two });
            let lu = Lu::new(&a).unwrap();
            let x = lu.solve(&bv);
            black_box(x)
        })
    });

    // Baseline: finite-diff (solve twice, subtract, divide by h)
    group.bench_function("finite_diff", |b| {
        let h = 1e-8;
        b.iter(|| {
            let solve_at = |t: f64| {
                let a = DMat::from_fn(2, 2, |i, j| match (i, j) {
                    (0, 0) => 2.0 + t,
                    (0, 1) => 1.0,
                    (1, 0) => 1.0,
                    (1, 1) => 3.0 + t,
                    _ => unreachable!(),
                });
                let bv = DVec::from_fn(2, |i| if i == 0 { 1.0 } else { 2.0 });
                Lu::new(&a).unwrap().solve(&bv)
            };
            let xp = solve_at(black_box(t_val) + h);
            let xm = solve_at(black_box(t_val) - h);
            let dx = DVec::from_fn(2, |i| (xp[i] - xm[i]) / (2.0 * h));
            black_box(dx)
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 4. Hessian of Rosenbrock — second derivatives via Dual<Dual<f64>>
// ---------------------------------------------------------------------------

fn rosenbrock<S: Scalar>(x: &[S]) -> S {
    let one: S = S::ONE;
    let hundred: S = S::from_f64(100.0);
    let dx = one - x[0];
    let dy = x[1] - x[0] * x[0];
    dx * dx + hundred * dy * dy
}

fn hessian_rosenbrock(c: &mut Criterion) {
    let mut group = c.benchmark_group("differentiable/hessian_rosenbrock");

    let x0 = [0.5, 0.5];

    // tang: hessian() via Dual<Dual<f64>> — exact second derivatives
    group.bench_function("tang_hessian", |b| {
        b.iter(|| {
            let h = tang_ad::hessian(
                |x| rosenbrock(black_box(x)),
                black_box(&x0),
            );
            black_box(h)
        })
    });

    // Baseline: finite-diff Hessian (n² evaluations with central diff)
    group.bench_function("finite_diff", |b| {
        let h = 1e-5;
        let n = x0.len();
        b.iter(|| {
            let mut hess = vec![0.0f64; n * n];
            for i in 0..n {
                for j in 0..n {
                    let mut xpp = black_box(x0);
                    let mut xpm = black_box(x0);
                    let mut xmp = black_box(x0);
                    let mut xmm = black_box(x0);
                    xpp[i] += h; xpp[j] += h;
                    xpm[i] += h; xpm[j] -= h;
                    xmp[i] -= h; xmp[j] += h;
                    xmm[i] -= h; xmm[j] -= h;
                    hess[i * n + j] = (rosenbrock(&xpp) - rosenbrock(&xpm)
                        - rosenbrock(&xmp) + rosenbrock(&xmm))
                        / (4.0 * h * h);
                }
            }
            black_box(hess)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    rigid_body_gradient,
    fk_jacobian,
    differentiable_lu_solve,
    hessian_rosenbrock,
);
criterion_main!(benches);
