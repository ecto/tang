//! Hand-written kernels vs the same math decomposed into tang IR.
//!
//! Each block runs three ways:
//! - `hand`   — the existing hand-written path (tang-train / tang-tensor)
//! - `ir`     — traced into a `tang_expr::ExprGraph`, compiled as-is
//! - `ir_opt` — same graph after the tang-compile pass pipeline
//!
//! Plus `build` groups that time trace + passes + compile, since that cost
//! is what grows with model size in a scalar IR.
//!
//! Blocks:
//! - `swiglu_act` — silu(g) * u, purely elementwise (IR's best case)
//! - `rmsnorm`    — a row reduction followed by elementwise scale
//! - `matvec`     — y = W x, the op that dominates real models
//!
//! Node counts are printed to stderr once per size.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tang::Scalar;
use tang_compile::{EliminateDeadCode, FoldConstants, PassManager, SimplifyAlgebraic};
use tang_expr::compile::CompiledMany;
use tang_expr::{trace, ExprGraph, ExprId};
use tang_tensor::{Shape, Tensor};
use tang_train::{Module, RMSNorm};

fn passes() -> PassManager {
    let mut pm = PassManager::new();
    pm.add(FoldConstants);
    pm.add(SimplifyAlgebraic);
    pm.add(EliminateDeadCode);
    pm
}

/// Compile a graph, optionally running the pass pipeline first.
fn compile(g: &ExprGraph, outs: &[ExprId], opt: bool) -> (CompiledMany, usize) {
    if !opt {
        return (g.compile_many(outs), g.live_set(outs).len());
    }
    let (g2, outs2) = passes().run(g, outs);
    (g2.compile_many(&outs2), g2.live_set(&outs2).len())
}

// ---------------------------------------------------------------------------
// SwiGLU activation: out = silu(g) * u
// ---------------------------------------------------------------------------

fn trace_swiglu_act() -> (ExprGraph, Vec<ExprId>) {
    let (g, out) = trace(|| {
        let gate = ExprId::var(0);
        let up = ExprId::var(1);
        let silu = gate / (ExprId::from_f64(1.0) + (-gate).exp());
        silu * up
    });
    (g, vec![out])
}

fn bench_swiglu_act(c: &mut Criterion) {
    let mut group = c.benchmark_group("kvi_swiglu_act");
    for &n in &[4_096usize, 65_536, 1_048_576] {
        let gate: Vec<f64> = (0..n).map(|i| (i as f64) * 1e-4 - 1.0).collect();
        let up: Vec<f64> = (0..n).map(|i| (i as f64) * 2e-4).collect();
        let mut out = vec![0.0f64; n];

        group.bench_function(BenchmarkId::new("hand", n), |b| {
            b.iter(|| {
                for i in 0..n {
                    let g = gate[i];
                    out[i] = g / (1.0 + (-g).exp()) * up[i];
                }
                black_box(&out);
            })
        });

        let (g, outs) = trace_swiglu_act();
        for opt in [false, true] {
            let (f, nodes) = compile(&g, &outs, opt);
            if n == 4_096 {
                eprintln!("swiglu_act opt={opt}: {nodes} live nodes per element");
            }
            let name = if opt { "ir_opt" } else { "ir" };
            group.bench_function(BenchmarkId::new(name, n), |b| {
                let mut o = [0.0f64; 1];
                b.iter(|| {
                    for i in 0..n {
                        f(&[gate[i], up[i]], &mut o);
                        out[i] = o[0];
                    }
                    black_box(&out);
                })
            });
        }
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// RMSNorm over one row of `dim`, weight = 1 (matches RMSNorm::new)
// ---------------------------------------------------------------------------

fn trace_rmsnorm(dim: usize) -> (ExprGraph, Vec<ExprId>) {
    trace(|| {
        let xs: Vec<ExprId> = (0..dim).map(|i| ExprId::var(i as u16)).collect();
        let mut sq = ExprId::from_f64(0.0);
        for &x in &xs {
            sq = sq + x * x;
        }
        let rms = (sq / ExprId::from_f64(dim as f64) + ExprId::from_f64(1e-6)).sqrt();
        let inv = ExprId::from_f64(1.0) / rms;
        let w = ExprId::from_f64(1.0);
        xs.iter().map(|&x| w * (x * inv)).collect()
    })
}

fn bench_rmsnorm(c: &mut Criterion) {
    let mut group = c.benchmark_group("kvi_rmsnorm");
    let batch = 16;
    for &dim in &[64usize, 576, 2048] {
        let input = Tensor::from_fn(Shape::from_slice(&[batch, dim]), |idx| {
            ((idx[0] * dim + idx[1]) as f64) * 0.01
        });
        let rows: Vec<f64> = input.data().to_vec();
        let mut out = vec![0.0f64; batch * dim];

        let mut norm = RMSNorm::<f64>::new(dim);
        group.bench_function(BenchmarkId::new("hand", format!("{batch}x{dim}")), |b| {
            b.iter(|| black_box(norm.forward(&input)))
        });

        let (g, outs) = trace_rmsnorm(dim);
        for opt in [false, true] {
            let (f, nodes) = compile(&g, &outs, opt);
            eprintln!("rmsnorm dim={dim} opt={opt}: {nodes} live nodes per row");
            let name = if opt { "ir_opt" } else { "ir" };
            group.bench_function(BenchmarkId::new(name, format!("{batch}x{dim}")), |b| {
                b.iter(|| {
                    for r in 0..batch {
                        f(
                            &rows[r * dim..(r + 1) * dim],
                            &mut out[r * dim..(r + 1) * dim],
                        );
                    }
                    black_box(&out);
                })
            });
        }

        group.bench_function(BenchmarkId::new("build", format!("{batch}x{dim}")), |b| {
            b.iter(|| {
                let (g, outs) = trace_rmsnorm(dim);
                black_box(compile(&g, &outs, true));
            })
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Matvec y = W x, W is n x n. W and x are both IR vars (n*n + n inputs),
// so n is capped by the u16 var index: n = 255 is the largest that fits.
// ---------------------------------------------------------------------------

fn trace_matvec(n: usize) -> (ExprGraph, Vec<ExprId>) {
    trace(|| {
        let w = |i: usize, j: usize| ExprId::var((i * n + j) as u16);
        let x: Vec<ExprId> = (0..n).map(|j| ExprId::var((n * n + j) as u16)).collect();
        (0..n)
            .map(|i| {
                let mut acc = ExprId::from_f64(0.0);
                for j in 0..n {
                    acc = acc + w(i, j) * x[j];
                }
                acc
            })
            .collect()
    })
}

fn bench_matvec(c: &mut Criterion) {
    let mut group = c.benchmark_group("kvi_matvec");
    for &n in &[16usize, 64, 128, 255] {
        let w = Tensor::from_fn(Shape::from_slice(&[n, n]), |idx| {
            ((idx[0] * n + idx[1]) as f64) * 1e-3
        });
        let x = Tensor::from_fn(Shape::from_slice(&[n, 1]), |idx| (idx[0] as f64) * 1e-2);

        group.bench_function(BenchmarkId::new("hand", n), |b| {
            b.iter(|| black_box(w.matmul(&x)))
        });

        let mut inputs: Vec<f64> = w.data().to_vec();
        inputs.extend_from_slice(x.data());
        let mut out = vec![0.0f64; n];

        let (g, outs) = trace_matvec(n);
        for opt in [false, true] {
            let (f, nodes) = compile(&g, &outs, opt);
            eprintln!("matvec n={n} opt={opt}: {nodes} live nodes");
            let name = if opt { "ir_opt" } else { "ir" };
            group.bench_function(BenchmarkId::new(name, n), |b| {
                b.iter(|| {
                    f(&inputs, &mut out);
                    black_box(&out);
                })
            });
        }

        group.bench_function(BenchmarkId::new("build", n), |b| {
            b.iter(|| {
                let (g, outs) = trace_matvec(n);
                black_box(compile(&g, &outs, true));
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_swiglu_act, bench_rmsnorm, bench_matvec);

// ---------------------------------------------------------------------------
// GPU: hand-written fused causal attention vs attention derived from tensor IR
// ---------------------------------------------------------------------------

mod gpu {
    use super::*;
    use tang_bench::ir_gpu::{
        causal_mask, flash_attention, flash_attention_split, flash_attention_tiled,
        naive_attention, Executable,
    };
    use tang_gpu::{kv_attention_fused, GpuDevice, GpuTensor, KernelCache};

    fn random(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = tang_bench::Rng::new(seed);
        (0..n).map(|_| rng.f32() * 2.0 - 1.0).collect()
    }

    pub fn bench_attention(c: &mut Criterion) {
        let Ok(device) = GpuDevice::new_sync() else {
            eprintln!("no GPU, skipping kvi_gpu_attention");
            return;
        };
        let mut cache = KernelCache::new();
        let mut group = c.benchmark_group("kvi_gpu_attention");
        group.sample_size(20);
        let d = 64;
        for &n in &[128usize, 512, 1024] {
            let (q, k, v) = (random(n * d, 1), random(n * d, 2), random(n * d, 3));
            let mask = causal_mask(n);

            let qt = GpuTensor::from_slice(&device, &q, &[n, d]);
            let kt = GpuTensor::from_slice(&device, &k, &[n, d]);
            let vt = GpuTensor::from_slice(&device, &v, &[n, d]);
            group.bench_function(BenchmarkId::new("hand", n), |b| {
                b.iter(|| {
                    let out =
                        kv_attention_fused(&device, &mut cache, &qt, &kt, &vt, n, n, 1, 1, d, 0);
                    cache.flush(&device);
                    black_box(out.to_vec_flushed(&device, &mut cache));
                })
            });

            for (name, (g, y)) in [
                ("ir_naive", naive_attention(n, d, d)),
                ("ir_flash", flash_attention(n, d, d)),
                ("ir_flash_split", flash_attention_split(n, d, d)),
                ("ir_flash_tiled", flash_attention_tiled(n, d, d)),
            ] {
                let exe = Executable::new(&device, g.to_wgsl(&[y]), &[&q, &k, &v, &mask]);
                group.bench_function(BenchmarkId::new(name, n), |b| {
                    b.iter(|| {
                        exe.run(&device, &mut cache);
                        cache.flush(&device);
                        black_box(exe.read(&device, y));
                    })
                });
            }
        }
        group.finish();
    }
}

mod gpu_matmul {
    use super::*;
    use tang_bench::ir_gpu::{matmul_candidates, Executable};
    use tang_gpu::{GpuDevice, GpuTensor, KernelCache};

    fn random(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = tang_bench::Rng::new(seed);
        (0..n).map(|_| rng.f32() * 2.0 - 1.0).collect()
    }

    pub fn bench_matmul(c: &mut Criterion) {
        let Ok(device) = GpuDevice::new_sync() else {
            eprintln!("no GPU, skipping kvi_gpu_matmul");
            return;
        };
        let mut cache = KernelCache::new();
        let mut group = c.benchmark_group("kvi_gpu_matmul");
        group.sample_size(20);
        for &n in &[256usize, 1024] {
            let (a, b) = (random(n * n, 1), random(n * n, 2));
            let at = GpuTensor::from_slice(&device, &a, &[n, n]);
            let bt = GpuTensor::from_slice(&device, &b, &[n, n]);
            group.bench_function(BenchmarkId::new("hand", n), |bch| {
                bch.iter(|| {
                    let out = tang_gpu::matmul::matmul(&device, &mut cache, &at, &bt);
                    black_box(out.to_vec_flushed(&device, &mut cache));
                })
            });
            for (name, g, c) in matmul_candidates(n, n, n) {
                let exe = Executable::new(&device, g.to_wgsl(&[c]), &[&a, &b]);
                group.bench_function(BenchmarkId::new(format!("ir_{name}"), n), |bch| {
                    bch.iter(|| {
                        exe.run(&device, &mut cache);
                        cache.flush(&device);
                        black_box(exe.read(&device, c));
                    })
                });
            }
        }
        group.finish();
    }
}

mod gpu_norm {
    use super::*;
    use tang_bench::ir_gpu::{rmsnorm_candidates, swiglu_graph, Executable};
    use tang_gpu::{GpuDevice, GpuTensor, KernelCache};

    fn random(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = tang_bench::Rng::new(seed);
        (0..n).map(|_| rng.f32() * 2.0 - 1.0).collect()
    }

    pub fn bench_rmsnorm_swiglu(c: &mut Criterion) {
        let Ok(device) = GpuDevice::new_sync() else {
            eprintln!("no GPU, skipping kvi_gpu_rmsnorm");
            return;
        };
        let mut cache = KernelCache::new();
        let mut group = c.benchmark_group("kvi_gpu_rmsnorm");
        group.sample_size(20);
        for &(rows, dim) in &[(128usize, 576usize), (2048, 2048)] {
            let x = random(rows * dim, 1);
            let xt = GpuTensor::from_slice(&device, &x, &[rows, dim]);
            let norm = tang_gpu::GpuRMSNorm::new(&device, dim, 1e-6);
            let id = format!("{rows}x{dim}");
            group.bench_function(BenchmarkId::new("hand", &id), |b| {
                b.iter(|| {
                    let out = norm.forward(&device, &mut cache, &xt);
                    black_box(out.to_vec_flushed(&device, &mut cache));
                })
            });
            for (name, g, y) in rmsnorm_candidates(rows, dim) {
                let exe = Executable::new(&device, g.to_wgsl(&[y]), &[&x]);
                group.bench_function(BenchmarkId::new(format!("ir_{name}"), &id), |b| {
                    b.iter(|| {
                        exe.run(&device, &mut cache);
                        cache.flush(&device);
                        black_box(exe.read(&device, y));
                    })
                });
            }
        }
        group.finish();

        let mut group = c.benchmark_group("kvi_gpu_swiglu");
        group.sample_size(20);
        for &n in &[65_536usize, 4_194_304] {
            let (gate, up) = (random(n, 2), random(n, 3));
            let gt = GpuTensor::from_slice(&device, &gate, &[n]);
            let ut = GpuTensor::from_slice(&device, &up, &[n]);
            group.bench_function(BenchmarkId::new("hand", n), |b| {
                b.iter(|| {
                    let out = tang_gpu::swiglu_fused_pub(&device, &mut cache, &gt, &ut);
                    black_box(out.to_vec_flushed(&device, &mut cache));
                })
            });
            let (g, y) = swiglu_graph(n);
            let exe = Executable::new(&device, g.to_wgsl(&[y]), &[&gate, &up]);
            group.bench_function(BenchmarkId::new("ir", n), |b| {
                b.iter(|| {
                    exe.run(&device, &mut cache);
                    cache.flush(&device);
                    black_box(exe.read(&device, y));
                })
            });
        }
        group.finish();
    }
}

criterion_group!(
    gpu_benches,
    gpu::bench_attention,
    gpu_matmul::bench_matmul,
    gpu_norm::bench_rmsnorm_swiglu
);
criterion_main!(benches, gpu_benches);
