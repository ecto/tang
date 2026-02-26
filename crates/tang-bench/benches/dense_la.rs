use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use tang_bench::*;

const SIZES: &[usize] = &[32, 64, 128, 256, 512];
const SVD_SIZES: &[usize] = &[32, 64, 128]; // Jacobi SVD is slow for large

fn gemm(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm");

    for &n in SIZES {
        group.throughput(Throughput::Elements((n * n * n) as u64));

        group.bench_with_input(BenchmarkId::new("tang", n), &n, |b, &n| {
            let a = random_dmat(n);
            let v = random_dmat(n);
            b.iter(|| black_box(a.mul_mat(&v)))
        });

        group.bench_with_input(BenchmarkId::new("nalgebra", n), &n, |b, &n| {
            let flat_a = random_f64_flat(n * n);
            let flat_b = random_f64_flat(n * n);
            let a = nalgebra::DMatrix::from_column_slice(n, n, &flat_a);
            let v = nalgebra::DMatrix::from_column_slice(n, n, &flat_b);
            b.iter(|| black_box(&a * &v))
        });
    }

    group.finish();
}

fn lu_solve(c: &mut Criterion) {
    let mut group = c.benchmark_group("lu_solve");

    for &n in SIZES {
        group.bench_with_input(BenchmarkId::new("tang", n), &n, |b, &n| {
            let a = random_dmat(n);
            let rhs = random_dvec(n);
            b.iter(|| {
                let lu = tang_la::Lu::new(&a).unwrap();
                black_box(lu.solve(&rhs))
            })
        });

        group.bench_with_input(BenchmarkId::new("nalgebra", n), &n, |b, &n| {
            let flat = random_f64_flat(n * n);
            let rhs_flat = random_f64_flat(n);
            let a = nalgebra::DMatrix::from_column_slice(n, n, &flat);
            let rhs = nalgebra::DVector::from_column_slice(&rhs_flat);
            b.iter(|| {
                let lu = a.clone().lu();
                black_box(lu.solve(&rhs))
            })
        });
    }

    group.finish();
}

fn cholesky_solve(c: &mut Criterion) {
    let mut group = c.benchmark_group("cholesky_solve");

    for &n in SIZES {
        group.bench_with_input(BenchmarkId::new("tang", n), &n, |b, &n| {
            let a = random_spd_dmat(n);
            let rhs = random_dvec(n);
            b.iter(|| {
                let ch = tang_la::Cholesky::new(&a).unwrap();
                black_box(ch.solve(&rhs))
            })
        });

        group.bench_with_input(BenchmarkId::new("nalgebra", n), &n, |b, &n| {
            let flat = random_spd_flat(n);
            let rhs_flat = random_f64_flat(n);
            let a = nalgebra::DMatrix::from_column_slice(n, n, &flat);
            let rhs = nalgebra::DVector::from_column_slice(&rhs_flat);
            b.iter(|| {
                let ch = nalgebra::linalg::Cholesky::new(a.clone()).unwrap();
                black_box(ch.solve(&rhs))
            })
        });
    }

    group.finish();
}

fn qr(c: &mut Criterion) {
    let mut group = c.benchmark_group("qr");

    for &n in SIZES {
        group.bench_with_input(BenchmarkId::new("tang", n), &n, |b, &n| {
            let a = random_dmat(n);
            b.iter(|| {
                let qr = tang_la::Qr::new(&a);
                black_box(qr.r())
            })
        });

        group.bench_with_input(BenchmarkId::new("nalgebra", n), &n, |b, &n| {
            let flat = random_f64_flat(n * n);
            let a = nalgebra::DMatrix::from_column_slice(n, n, &flat);
            b.iter(|| {
                let qr = a.clone().qr();
                black_box(qr.r())
            })
        });
    }

    group.finish();
}

fn svd(c: &mut Criterion) {
    let mut group = c.benchmark_group("svd");

    for &n in SVD_SIZES {
        group.bench_with_input(BenchmarkId::new("tang", n), &n, |b, &n| {
            let a = random_dmat(n);
            b.iter(|| black_box(tang_la::Svd::new(&a)))
        });

        group.bench_with_input(BenchmarkId::new("nalgebra", n), &n, |b, &n| {
            let flat = random_f64_flat(n * n);
            let a = nalgebra::DMatrix::from_column_slice(n, n, &flat);
            b.iter(|| black_box(a.clone().svd(true, true)))
        });
    }

    group.finish();
}

fn symm_eigen(c: &mut Criterion) {
    let mut group = c.benchmark_group("symm_eigen");

    for &n in SVD_SIZES {
        group.bench_with_input(BenchmarkId::new("tang", n), &n, |b, &n| {
            let a = random_spd_dmat(n);
            b.iter(|| black_box(tang_la::SymmetricEigen::new(&a)))
        });

        group.bench_with_input(BenchmarkId::new("nalgebra", n), &n, |b, &n| {
            let flat = random_spd_flat(n);
            let a = nalgebra::DMatrix::from_column_slice(n, n, &flat);
            b.iter(|| black_box(a.clone().symmetric_eigen()))
        });
    }

    group.finish();
}

criterion_group!(benches, gemm, lu_solve, cholesky_solve, qr, svd, symm_eigen);
criterion_main!(benches);
