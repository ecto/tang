use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tang_bench::*;

// ============================================================
// Vec3
// ============================================================

fn vec3_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("vec3/dot");

    group.bench_function("tang_f32", |b| {
        let a = random_tang_vec3f32(1)[0];
        let v = random_tang_vec3f32(2)[1];
        b.iter(|| black_box(black_box(a).dot(black_box(v))))
    });

    group.bench_function("tang_f64", |b| {
        let a = random_tang_vec3f64(1)[0];
        let v = random_tang_vec3f64(2)[1];
        b.iter(|| black_box(black_box(a).dot(black_box(v))))
    });

    group.bench_function("nalgebra", |b| {
        let d = random_f64_triples(2);
        let a = nalgebra::Vector3::new(d[0][0], d[0][1], d[0][2]);
        let v = nalgebra::Vector3::new(d[1][0], d[1][1], d[1][2]);
        b.iter(|| black_box(black_box(a).dot(&black_box(v))))
    });

    group.bench_function("glam", |b| {
        let d = random_f32_triples(2);
        let a = glam::Vec3::new(d[0][0], d[0][1], d[0][2]);
        let v = glam::Vec3::new(d[1][0], d[1][1], d[1][2]);
        b.iter(|| black_box(black_box(a).dot(black_box(v))))
    });

    group.finish();
}

fn vec3_cross(c: &mut Criterion) {
    let mut group = c.benchmark_group("vec3/cross");

    group.bench_function("tang_f32", |b| {
        let a = random_tang_vec3f32(1)[0];
        let v = random_tang_vec3f32(2)[1];
        b.iter(|| black_box(black_box(a).cross(black_box(v))))
    });

    group.bench_function("tang_f64", |b| {
        let a = random_tang_vec3f64(1)[0];
        let v = random_tang_vec3f64(2)[1];
        b.iter(|| black_box(black_box(a).cross(black_box(v))))
    });

    group.bench_function("nalgebra", |b| {
        let d = random_f64_triples(2);
        let a = nalgebra::Vector3::new(d[0][0], d[0][1], d[0][2]);
        let v = nalgebra::Vector3::new(d[1][0], d[1][1], d[1][2]);
        b.iter(|| black_box(black_box(a).cross(&black_box(v))))
    });

    group.bench_function("glam", |b| {
        let d = random_f32_triples(2);
        let a = glam::Vec3::new(d[0][0], d[0][1], d[0][2]);
        let v = glam::Vec3::new(d[1][0], d[1][1], d[1][2]);
        b.iter(|| black_box(black_box(a).cross(black_box(v))))
    });

    group.finish();
}

fn vec3_length(c: &mut Criterion) {
    let mut group = c.benchmark_group("vec3/length");

    group.bench_function("tang_f32", |b| {
        let v = random_tang_vec3f32(1)[0];
        b.iter(|| black_box(black_box(v).norm()))
    });

    group.bench_function("tang_f64", |b| {
        let v = random_tang_vec3f64(1)[0];
        b.iter(|| black_box(black_box(v).norm()))
    });

    group.bench_function("nalgebra", |b| {
        let d = random_f64_triples(1);
        let v = nalgebra::Vector3::new(d[0][0], d[0][1], d[0][2]);
        b.iter(|| black_box(black_box(v).norm()))
    });

    group.bench_function("glam", |b| {
        let d = random_f32_triples(1);
        let v = glam::Vec3::new(d[0][0], d[0][1], d[0][2]);
        b.iter(|| black_box(black_box(v).length()))
    });

    group.finish();
}

fn vec3_normalize(c: &mut Criterion) {
    let mut group = c.benchmark_group("vec3/normalize");

    group.bench_function("tang_f32", |b| {
        let v = random_tang_vec3f32(1)[0];
        b.iter(|| black_box(black_box(v).normalize()))
    });

    group.bench_function("tang_f64", |b| {
        let v = random_tang_vec3f64(1)[0];
        b.iter(|| black_box(black_box(v).normalize()))
    });

    group.bench_function("nalgebra", |b| {
        let d = random_f64_triples(1);
        let v = nalgebra::Vector3::new(d[0][0], d[0][1], d[0][2]);
        b.iter(|| black_box(black_box(v).normalize()))
    });

    group.bench_function("glam", |b| {
        let d = random_f32_triples(1);
        let v = glam::Vec3::new(d[0][0], d[0][1], d[0][2]);
        b.iter(|| black_box(black_box(v).normalize()))
    });

    group.finish();
}

// ============================================================
// Mat3
// ============================================================

fn mat3_determinant(c: &mut Criterion) {
    let mut group = c.benchmark_group("mat3/determinant");

    group.bench_function("tang", |b| {
        let m = random_tang_mat3f64(1)[0];
        b.iter(|| black_box(black_box(m).determinant()))
    });

    group.bench_function("nalgebra", |b| {
        let d = random_f64_mat3s(1);
        let m = nalgebra::Matrix3::new(
            d[0][0], d[0][1], d[0][2], d[0][3], d[0][4], d[0][5], d[0][6], d[0][7], d[0][8],
        );
        b.iter(|| black_box(black_box(m).determinant()))
    });

    group.finish();
}

fn mat3_inverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("mat3/inverse");

    group.bench_function("tang", |b| {
        let m = tang::Mat3::new(1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0);
        b.iter(|| black_box(black_box(m).try_inverse()))
    });

    group.bench_function("nalgebra", |b| {
        let m = nalgebra::Matrix3::new(1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0);
        b.iter(|| black_box(black_box(m).try_inverse()))
    });

    group.finish();
}

fn mat3_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("mat3/mul");

    group.bench_function("tang", |b| {
        let ms = random_tang_mat3f64(2);
        let (a, v) = (ms[0], ms[1]);
        b.iter(|| black_box(black_box(a) * black_box(v)))
    });

    group.bench_function("nalgebra", |b| {
        let d = random_f64_mat3s(2);
        let a = nalgebra::Matrix3::new(
            d[0][0], d[0][1], d[0][2], d[0][3], d[0][4], d[0][5], d[0][6], d[0][7], d[0][8],
        );
        let v = nalgebra::Matrix3::new(
            d[1][0], d[1][1], d[1][2], d[1][3], d[1][4], d[1][5], d[1][6], d[1][7], d[1][8],
        );
        b.iter(|| black_box(black_box(a) * black_box(v)))
    });

    group.finish();
}

fn mat3_transpose(c: &mut Criterion) {
    let mut group = c.benchmark_group("mat3/transpose");

    group.bench_function("tang", |b| {
        let m = random_tang_mat3f64(1)[0];
        b.iter(|| black_box(black_box(m).transpose()))
    });

    group.bench_function("nalgebra", |b| {
        let d = random_f64_mat3s(1);
        let m = nalgebra::Matrix3::new(
            d[0][0], d[0][1], d[0][2], d[0][3], d[0][4], d[0][5], d[0][6], d[0][7], d[0][8],
        );
        b.iter(|| black_box(black_box(m).transpose()))
    });

    group.finish();
}

// ============================================================
// Mat4
// ============================================================

fn mat4_determinant(c: &mut Criterion) {
    let mut group = c.benchmark_group("mat4/determinant");

    group.bench_function("tang", |b| {
        let m = random_tang_mat4f64(1)[0];
        b.iter(|| black_box(black_box(m).determinant()))
    });

    group.bench_function("nalgebra", |b| {
        let d = random_f64_mat4s(1);
        let m = nalgebra::Matrix4::from_column_slice(&d[0]);
        b.iter(|| black_box(black_box(m).determinant()))
    });

    group.bench_function("glam", |b| {
        let d = random_f32_mat4s(1);
        let m = glam::Mat4::from_cols_array(&d[0]);
        b.iter(|| black_box(black_box(m).determinant()))
    });

    group.finish();
}

fn mat4_inverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("mat4/inverse");

    group.bench_function("tang", |b| {
        let m = tang::Mat4::new(
            1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 3.0, 2.0, 0.0, 1.0, 0.0, 1.0, 3.0, 0.0, 1.0,
        );
        b.iter(|| black_box(black_box(m).try_inverse()))
    });

    group.bench_function("nalgebra", |b| {
        let m = nalgebra::Matrix4::new(
            1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 3.0, 2.0, 0.0, 1.0, 0.0, 1.0, 3.0, 0.0, 1.0,
        );
        b.iter(|| black_box(black_box(m).try_inverse()))
    });

    group.bench_function("glam", |b| {
        let m = glam::Mat4::from_cols_array(&[
            1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 3.0, 2.0, 0.0, 1.0, 0.0, 1.0, 3.0, 0.0, 1.0,
        ]);
        b.iter(|| black_box(black_box(m).inverse()))
    });

    group.finish();
}

fn mat4_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("mat4/mul");

    group.bench_function("tang", |b| {
        let ms = random_tang_mat4f64(2);
        let (a, v) = (ms[0], ms[1]);
        b.iter(|| black_box(black_box(a) * black_box(v)))
    });

    group.bench_function("nalgebra", |b| {
        let d = random_f64_mat4s(2);
        let a = nalgebra::Matrix4::from_column_slice(&d[0]);
        let v = nalgebra::Matrix4::from_column_slice(&d[1]);
        b.iter(|| black_box(black_box(a) * black_box(v)))
    });

    group.bench_function("glam", |b| {
        let d = random_f32_mat4s(2);
        let a = glam::Mat4::from_cols_array(&d[0]);
        let v = glam::Mat4::from_cols_array(&d[1]);
        b.iter(|| black_box(black_box(a) * black_box(v)))
    });

    group.finish();
}

fn mat4_transpose(c: &mut Criterion) {
    let mut group = c.benchmark_group("mat4/transpose");

    group.bench_function("tang", |b| {
        let m = random_tang_mat4f64(1)[0];
        b.iter(|| black_box(black_box(m).transpose()))
    });

    group.bench_function("nalgebra", |b| {
        let d = random_f64_mat4s(1);
        let m = nalgebra::Matrix4::from_column_slice(&d[0]);
        b.iter(|| black_box(black_box(m).transpose()))
    });

    group.bench_function("glam", |b| {
        let d = random_f32_mat4s(1);
        let m = glam::Mat4::from_cols_array(&d[0]);
        b.iter(|| black_box(black_box(m).transpose()))
    });

    group.finish();
}

// ============================================================
// Quat
// ============================================================

fn quat_from_axis_angle(c: &mut Criterion) {
    let mut group = c.benchmark_group("quat/from_axis_angle");

    group.bench_function("tang", |b| {
        let axis = tang::Vec3::new(0.0, 0.0, 1.0);
        let angle = 1.2f64;
        b.iter(|| {
            black_box(tang::Quat::from_axis_angle(
                black_box(axis),
                black_box(angle),
            ))
        })
    });

    group.bench_function("nalgebra", |b| {
        let axis = nalgebra::Unit::new_normalize(nalgebra::Vector3::new(0.0, 0.0, 1.0));
        let angle = 1.2f64;
        b.iter(|| {
            black_box(nalgebra::UnitQuaternion::from_axis_angle(
                &black_box(axis),
                black_box(angle),
            ))
        })
    });

    group.bench_function("glam", |b| {
        let axis = glam::Vec3::new(0.0, 0.0, 1.0);
        let angle = 1.2f32;
        b.iter(|| {
            black_box(glam::Quat::from_axis_angle(
                black_box(axis),
                black_box(angle),
            ))
        })
    });

    group.finish();
}

fn quat_rotate_vec(c: &mut Criterion) {
    let mut group = c.benchmark_group("quat/rotate_vec");

    group.bench_function("tang", |b| {
        let q = tang::Quat::from_axis_angle(tang::Vec3::z(), 1.2);
        let v = tang::Vec3::new(1.0, 2.0, 3.0);
        b.iter(|| black_box(black_box(q).rotate(black_box(v))))
    });

    group.bench_function("nalgebra", |b| {
        let q = nalgebra::UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(nalgebra::Vector3::z()),
            1.2,
        );
        let v = nalgebra::Vector3::new(1.0, 2.0, 3.0);
        b.iter(|| black_box(black_box(q) * black_box(v)))
    });

    group.bench_function("glam", |b| {
        let q = glam::Quat::from_axis_angle(glam::Vec3::Z, 1.2);
        let v = glam::Vec3::new(1.0, 2.0, 3.0);
        b.iter(|| black_box(black_box(q) * black_box(v)))
    });

    group.finish();
}

fn quat_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("quat/mul");

    group.bench_function("tang", |b| {
        let qs = random_tang_quat(2);
        let (a, v) = (qs[0], qs[1]);
        b.iter(|| black_box(black_box(a).mul(&black_box(v))))
    });

    group.bench_function("nalgebra", |b| {
        let q1 = nalgebra::UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(nalgebra::Vector3::z()),
            1.2,
        );
        let q2 = nalgebra::UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(nalgebra::Vector3::x()),
            0.8,
        );
        b.iter(|| black_box(black_box(q1) * black_box(q2)))
    });

    group.bench_function("glam", |b| {
        let q1 = glam::Quat::from_axis_angle(glam::Vec3::Z, 1.2);
        let q2 = glam::Quat::from_axis_angle(glam::Vec3::X, 0.8);
        b.iter(|| black_box(black_box(q1) * black_box(q2)))
    });

    group.finish();
}

fn quat_slerp(c: &mut Criterion) {
    let mut group = c.benchmark_group("quat/slerp");

    group.bench_function("tang", |b| {
        let qs = random_tang_quat(2);
        let (a, v) = (qs[0], qs[1]);
        b.iter(|| black_box(black_box(a).slerp(&black_box(v), 0.5)))
    });

    group.bench_function("nalgebra", |b| {
        let q1 = nalgebra::UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(nalgebra::Vector3::z()),
            1.2,
        );
        let q2 = nalgebra::UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(nalgebra::Vector3::x()),
            0.8,
        );
        b.iter(|| black_box(black_box(q1).slerp(&black_box(q2), 0.5)))
    });

    group.bench_function("glam", |b| {
        let q1 = glam::Quat::from_axis_angle(glam::Vec3::Z, 1.2);
        let q2 = glam::Quat::from_axis_angle(glam::Vec3::X, 0.8);
        b.iter(|| black_box(black_box(q1).slerp(black_box(q2), 0.5)))
    });

    group.finish();
}

// ============================================================
// Workloads
// ============================================================

fn workload_euler_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("workload/euler_3d");
    let steps = 10_000;
    let dt = 0.001;

    group.bench_function("tang", |b| {
        b.iter(|| {
            let mut pos = tang::Vec3::new(0.0f64, 0.0, 0.0);
            let mut vel = tang::Vec3::new(1.0, 0.5, 0.2);
            let accel = tang::Vec3::new(0.0, -9.81, 0.0);
            for _ in 0..steps {
                vel += accel * dt;
                pos += vel * dt;
            }
            black_box(pos)
        })
    });

    group.bench_function("nalgebra", |b| {
        b.iter(|| {
            let mut pos = nalgebra::Vector3::new(0.0f64, 0.0, 0.0);
            let mut vel = nalgebra::Vector3::new(1.0, 0.5, 0.2);
            let accel = nalgebra::Vector3::new(0.0, -9.81, 0.0);
            for _ in 0..steps {
                vel += accel * dt;
                pos += vel * dt;
            }
            black_box(pos)
        })
    });

    group.bench_function("glam", |b| {
        b.iter(|| {
            let mut pos = glam::DVec3::new(0.0, 0.0, 0.0);
            let mut vel = glam::DVec3::new(1.0, 0.5, 0.2);
            let accel = glam::DVec3::new(0.0, -9.81, 0.0);
            for _ in 0..steps {
                vel += accel * dt;
                pos += vel * dt;
            }
            black_box(pos)
        })
    });

    group.finish();
}

fn workload_transform_points(c: &mut Criterion) {
    let mut group = c.benchmark_group("workload/transform_points");
    let n = 100_000;

    group.bench_function("tang", |b| {
        let m = tang::Mat4::rotation_z(0.5) * tang::Mat4::translation(1.0, 2.0, 3.0);
        let points: Vec<tang::Vec3<f64>> = random_tang_vec3f64(n);
        b.iter(|| {
            let mut sum = tang::Vec3::zero();
            for p in &points {
                sum += m.transform_vec(*p);
            }
            black_box(sum)
        })
    });

    group.bench_function("nalgebra", |b| {
        let rot = nalgebra::Rotation3::from_axis_angle(
            &nalgebra::Unit::new_normalize(nalgebra::Vector3::z()),
            0.5,
        );
        let m = nalgebra::Matrix4::new_translation(&nalgebra::Vector3::new(1.0, 2.0, 3.0))
            * rot.to_homogeneous();
        let data = random_f64_triples(n);
        let points: Vec<nalgebra::Vector3<f64>> = data
            .iter()
            .map(|d| nalgebra::Vector3::new(d[0], d[1], d[2]))
            .collect();
        b.iter(|| {
            let mut sum = nalgebra::Vector3::zeros();
            for p in &points {
                let v4 = nalgebra::Vector4::new(p.x, p.y, p.z, 0.0);
                let r = m * v4;
                sum += nalgebra::Vector3::new(r.x, r.y, r.z);
            }
            black_box(sum)
        })
    });

    group.bench_function("glam", |b| {
        let m = glam::Mat4::from_rotation_z(0.5f32)
            * glam::Mat4::from_translation(glam::Vec3::new(1.0, 2.0, 3.0));
        let data = random_f32_triples(n);
        let points: Vec<glam::Vec3> = data
            .iter()
            .map(|d| glam::Vec3::new(d[0], d[1], d[2]))
            .collect();
        b.iter(|| {
            let mut sum = glam::Vec3::ZERO;
            for p in &points {
                sum += m.transform_vector3(*p);
            }
            black_box(sum)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    vec3_dot,
    vec3_cross,
    vec3_length,
    vec3_normalize,
    mat3_determinant,
    mat3_inverse,
    mat3_mul,
    mat3_transpose,
    mat4_determinant,
    mat4_inverse,
    mat4_mul,
    mat4_transpose,
    quat_from_axis_angle,
    quat_rotate_vec,
    quat_mul,
    quat_slerp,
    workload_euler_3d,
    workload_transform_points,
);
criterion_main!(benches);
