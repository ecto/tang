//! Shared helpers for tang benchmarks: seeded RNG, input generators.

use tang::{Mat3, Mat4, Quat, Vec3};
use tang_la::{DMat, DVec};

/// Simple xoshiro256** PRNG for reproducible benchmarks (no rand dependency in lib).
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // SplitMix64 to expand seed into state
        let mut z = seed;
        let mut s = [0u64; 4];
        for slot in &mut s {
            z = z.wrapping_add(0x9e3779b97f4a7c15);
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            *slot = z ^ (z >> 31);
        }
        Self { s }
    }

    fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform f64 in [-1, 1]
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64) * 2.0 - 1.0
    }

    /// Uniform f32 in [-1, 1]
    pub fn f32(&mut self) -> f32 {
        self.f64() as f32
    }
}

pub fn make_rng() -> Rng {
    Rng::new(0xDEAD_BEEF_CAFE_BABE)
}

// --- tang generators ---

pub fn random_tang_vec3f32(n: usize) -> Vec<Vec3<f32>> {
    let mut rng = make_rng();
    (0..n)
        .map(|_| Vec3::new(rng.f32(), rng.f32(), rng.f32()))
        .collect()
}

pub fn random_tang_vec3f64(n: usize) -> Vec<Vec3<f64>> {
    let mut rng = make_rng();
    (0..n)
        .map(|_| Vec3::new(rng.f64(), rng.f64(), rng.f64()))
        .collect()
}

pub fn random_tang_mat3f64(n: usize) -> Vec<Mat3<f64>> {
    let mut rng = make_rng();
    (0..n)
        .map(|_| {
            Mat3::new(
                rng.f64(),
                rng.f64(),
                rng.f64(),
                rng.f64(),
                rng.f64(),
                rng.f64(),
                rng.f64(),
                rng.f64(),
                rng.f64(),
            )
        })
        .collect()
}

pub fn random_tang_mat4f64(n: usize) -> Vec<Mat4<f64>> {
    let mut rng = make_rng();
    (0..n)
        .map(|_| {
            Mat4::new(
                rng.f64(),
                rng.f64(),
                rng.f64(),
                rng.f64(),
                rng.f64(),
                rng.f64(),
                rng.f64(),
                rng.f64(),
                rng.f64(),
                rng.f64(),
                rng.f64(),
                rng.f64(),
                rng.f64(),
                rng.f64(),
                rng.f64(),
                rng.f64(),
            )
        })
        .collect()
}

pub fn random_tang_mat4f32(n: usize) -> Vec<Mat4<f32>> {
    let mut rng = make_rng();
    (0..n)
        .map(|_| {
            Mat4::new(
                rng.f32(),
                rng.f32(),
                rng.f32(),
                rng.f32(),
                rng.f32(),
                rng.f32(),
                rng.f32(),
                rng.f32(),
                rng.f32(),
                rng.f32(),
                rng.f32(),
                rng.f32(),
                rng.f32(),
                rng.f32(),
                rng.f32(),
                rng.f32(),
            )
        })
        .collect()
}

pub fn random_tang_quat(n: usize) -> Vec<Quat<f64>> {
    let mut rng = make_rng();
    (0..n)
        .map(|_| Quat::new(rng.f64(), rng.f64(), rng.f64(), rng.f64()).normalize())
        .collect()
}

// --- DMat/DVec generators ---

pub fn random_dmat(size: usize) -> DMat<f64> {
    let flat = random_f64_flat(size * size);
    DMat::from_fn(size, size, |i, j| flat[j * size + i])
}

pub fn random_dvec(size: usize) -> DVec<f64> {
    let flat = random_f64_flat(size);
    DVec::from_fn(size, |i| flat[i])
}

/// Symmetric positive-definite matrix: A^T * A + εI
pub fn random_spd_dmat(size: usize) -> DMat<f64> {
    let a = random_dmat(size);
    let at = a.transpose();
    let ata = at.mul_mat(&a);
    // ata + 0.1 * I to ensure positive definiteness
    let mut result = ata;
    for i in 0..size {
        result.set(i, i, result.get(i, i) + 0.1);
    }
    result
}

// --- nalgebra generators (used only in bench files via dev-dependencies) ---
// These are generic over the nalgebra types which are only available in benches.
// We provide the data as raw Vecs that bench code converts.

pub fn random_f64_triples(n: usize) -> Vec<[f64; 3]> {
    let mut rng = make_rng();
    (0..n).map(|_| [rng.f64(), rng.f64(), rng.f64()]).collect()
}

pub fn random_f32_triples(n: usize) -> Vec<[f32; 3]> {
    let mut rng = make_rng();
    (0..n).map(|_| [rng.f32(), rng.f32(), rng.f32()]).collect()
}

pub fn random_f64_quads(n: usize) -> Vec<[f64; 4]> {
    let mut rng = make_rng();
    (0..n)
        .map(|_| [rng.f64(), rng.f64(), rng.f64(), rng.f64()])
        .collect()
}

pub fn random_f64_mat4s(n: usize) -> Vec<[f64; 16]> {
    let mut rng = make_rng();
    (0..n)
        .map(|_| {
            let mut m = [0.0f64; 16];
            for v in &mut m {
                *v = rng.f64();
            }
            m
        })
        .collect()
}

pub fn random_f32_mat4s(n: usize) -> Vec<[f32; 16]> {
    let mut rng = make_rng();
    (0..n)
        .map(|_| {
            let mut m = [0.0f32; 16];
            for v in &mut m {
                *v = rng.f32();
            }
            m
        })
        .collect()
}

pub fn random_f64_mat3s(n: usize) -> Vec<[f64; 9]> {
    let mut rng = make_rng();
    (0..n)
        .map(|_| {
            let mut m = [0.0f64; 9];
            for v in &mut m {
                *v = rng.f64();
            }
            m
        })
        .collect()
}

pub fn random_f64_flat(size: usize) -> Vec<f64> {
    let mut rng = make_rng();
    (0..size).map(|_| rng.f64()).collect()
}

/// SPD matrix as flat column-major data (size x size)
pub fn random_spd_flat(size: usize) -> Vec<f64> {
    let mut rng = Rng::new(0xDEAD_BEEF_CAFE_BABE);
    // Generate random A
    let a: Vec<f64> = (0..size * size).map(|_| rng.f64()).collect();
    // Compute A^T * A + 0.1 * I
    let mut result = vec![0.0f64; size * size];
    for i in 0..size {
        for j in 0..size {
            let mut sum = 0.0;
            for k in 0..size {
                // a is column-major: a[k + i*size] = A[k][i]
                sum += a[i * size + k] * a[j * size + k]; // A^T[i][k] * A[k][j]
            }
            if i == j {
                sum += 0.1;
            }
            result[j * size + i] = sum; // column-major
        }
    }
    result
}

/// Run tang-compile tensor graphs on the GPU, for the kernels-vs-IR benches.
pub mod ir_gpu {
    use tang::Scalar;
    use tang_compile::tensor::wgsl::WgslProgram;
    use tang_compile::tensor::{Monoid, TensorGraph, TensorId};
    use tang_expr::ExprId;
    use tang_gpu::{GpuBuffer, GpuDevice, KernelCache};

    /// Buffers for a program: graph inputs in order, then materialized
    /// intermediates. Reused across runs.
    pub struct Executable {
        pub program: WgslProgram,
        /// `buffers[i]` backs `program.buffers[i].0`.
        pub buffers: Vec<GpuBuffer>,
    }

    impl Executable {
        pub fn new(device: &GpuDevice, program: WgslProgram, inputs: &[&[f32]]) -> Self {
            let buffers = program
                .buffers
                .iter()
                .map(|b| match b.input {
                    Some(k) => {
                        assert_eq!(inputs[k].len(), b.len, "input {k} has wrong length");
                        GpuBuffer::from_slice(device, inputs[k])
                    }
                    None => GpuBuffer::uninit(device, b.len),
                })
                .collect();
            Self { program, buffers }
        }

        fn buffer(&self, t: TensorId) -> &GpuBuffer {
            let i = self
                .program
                .buffers
                .iter()
                .position(|b| b.node == t)
                .unwrap();
            &self.buffers[i]
        }

        /// Dispatch every kernel in order.
        pub fn run(&self, device: &GpuDevice, cache: &mut KernelCache) {
            for k in &self.program.kernels {
                let mut bufs: Vec<&GpuBuffer> = k.reads.iter().map(|t| self.buffer(*t)).collect();
                bufs.push(self.buffer(k.node));
                let out_idx = bufs.len() - 1;
                cache.dispatch_buffers(
                    device,
                    &k.source,
                    &bufs,
                    &[out_idx],
                    &[k.points, 0, 0, 0],
                    k.count,
                    k.workgroup_size,
                );
            }
        }

        pub fn read(&self, device: &GpuDevice, t: TensorId) -> Vec<f32> {
            self.buffer(t).to_vec_sync(device)
        }
    }

    /// Additive causal mask: 0 where `j <= i`, a large negative otherwise.
    pub fn causal_mask(n: usize) -> Vec<f32> {
        (0..n * n)
            .map(|p| if p % n <= p / n { 0.0 } else { -1e30 })
            .collect()
    }

    /// Single-head causal `softmax(q·kᵀ/√d + mask)·v`, written naively.
    /// Inputs: q `[n,d]`, k `[n,d]`, v `[n,dv]`, mask `[n,n]`.
    pub fn naive_attention(n: usize, d: usize, dv: usize) -> (TensorGraph, TensorId) {
        let mut g = TensorGraph::new();
        let q = g.input(&[n, d]);
        let k = g.input(&[n, d]);
        let v = g.input(&[n, dv]);
        let mask = g.input(&[n, n]);
        let scale = 1.0 / (d as f64).sqrt();
        let qs = g.map(&[q], |x| x[0] * ExprId::from_f64(scale)).unwrap();
        let kt = g.transpose(k, &[1, 0]).unwrap();
        let s = g.matmul(qs, kt).unwrap();
        let s = g.map(&[s, mask], |x| x[0] + x[1]).unwrap();
        let m = g.reduce(s, 1, Monoid::max()).unwrap();
        let mb = g.broadcast(m, &[n, n]).unwrap();
        let p = g.map(&[s, mb], |x| (x[0] - x[1]).exp()).unwrap();
        let l = g.reduce(p, 1, Monoid::sum()).unwrap();
        let o = g.matmul(p, v).unwrap();
        let lb = g.broadcast(l, &[n, dv]).unwrap();
        let y = g.map(&[o, lb], |x| x[0] / x[1]).unwrap();
        (g, y)
    }

    /// Naive attention after the full pass pipeline: one online-softmax
    /// reduction with `q·k` recomputed inside, nothing n×n materialized.
    pub fn flash_attention(n: usize, d: usize, dv: usize) -> (TensorGraph, TensorId) {
        let (g, y) = naive_attention(n, d, dv);
        let (g, outs) = g.substitute_maps(&[y]);
        let (mut g, outs) = g.stabilize_softmax(&outs);
        g.schedule_inline(&outs);
        (g, outs[0])
    }

    /// [`flash_attention`] with the online-softmax reduction split across
    /// a 256-thread workgroup per query row.
    pub fn flash_attention_split(n: usize, d: usize, dv: usize) -> (TensorGraph, TensorId) {
        let (mut g, y) = flash_attention(n, d, dv);
        g.schedule_split(&[y], 256, 64);
        (g, y)
    }

    /// [`flash_attention`] with 64 query rows per workgroup and k/v
    /// staged in workgroup memory 32 keys at a time — flash attention's
    /// tiling, derived from the views.
    pub fn flash_attention_tiled(n: usize, d: usize, dv: usize) -> (TensorGraph, TensorId) {
        let (mut g, y) = flash_attention(n, d, dv);
        g.schedule_tile(&[y], 64, 32, 64);
        (g, y)
    }

    /// Try every candidate schedule on this device and keep the fastest.
    /// This is the whole point: the algorithm is fixed, the schedule is
    /// whatever the chip at hand prefers.
    pub fn autotune(
        device: &GpuDevice,
        cache: &mut KernelCache,
        candidates: Vec<(&'static str, TensorGraph, TensorId)>,
        inputs: &[&[f32]],
        iters: usize,
    ) -> (&'static str, Executable, TensorId, std::time::Duration) {
        let mut best: Option<(&'static str, Executable, TensorId, std::time::Duration)> = None;
        for (name, g, y) in candidates {
            let exe = Executable::new(device, g.to_wgsl(&[y]), inputs);
            exe.run(device, cache);
            cache.flush(device);
            let _ = exe.read(device, y); // warm pipelines
            let start = std::time::Instant::now();
            for _ in 0..iters {
                exe.run(device, cache);
            }
            cache.flush(device);
            let _ = exe.read(device, y);
            let t = start.elapsed() / iters as u32;
            if best.as_ref().is_none_or(|b| t < b.3) {
                best = Some((name, exe, y, t));
            }
        }
        best.expect("at least one candidate")
    }

    /// [`flash_attention`] under [`TensorGraph::schedule_auto`].
    pub fn flash_attention_auto(n: usize, d: usize, dv: usize) -> (TensorGraph, TensorId) {
        let (g, y) = naive_attention(n, d, dv);
        let (g, outs) = g.substitute_maps(&[y]);
        let (mut g, outs) = g.stabilize_softmax(&outs);
        g.schedule_auto(&outs);
        (g, outs[0])
    }

    /// All attention schedules the compiler currently knows how to produce.
    pub fn attention_candidates(
        n: usize,
        d: usize,
        dv: usize,
    ) -> Vec<(&'static str, TensorGraph, TensorId)> {
        [
            ("naive", naive_attention(n, d, dv)),
            ("flash", flash_attention(n, d, dv)),
            ("flash_split", flash_attention_split(n, d, dv)),
            ("flash_tiled", flash_attention_tiled(n, d, dv)),
            ("flash_auto", flash_attention_auto(n, d, dv)),
        ]
        .into_iter()
        .map(|(name, (g, y))| (name, g, y))
        .collect()
    }

    /// `[m,k] x [k,n]` under each schedule the compiler can produce.
    pub fn matmul_candidates(
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<(&'static str, TensorGraph, TensorId)> {
        let build = |schedule: &str| {
            let mut g = TensorGraph::new();
            let a = g.input(&[m, k]);
            let b = g.input(&[k, n]);
            let c = g.matmul(a, b).unwrap();
            match schedule {
                "split" => g.schedule_split(&[c], 64, 16),
                "tiled" => g.schedule_tile_2d(&[c], 16, 16, 32, 16),
                "auto" => g.schedule_auto(&[c]),
                _ => {}
            }
            (g, c)
        };
        ["naive", "split", "tiled", "auto"]
            .into_iter()
            .map(|name| {
                let (g, c) = build(name);
                (name, g, c)
            })
            .collect()
    }

    /// RMSNorm over `[rows, dim]` with weight = 1, under each schedule.
    pub fn rmsnorm_candidates(
        rows: usize,
        dim: usize,
    ) -> Vec<(&'static str, TensorGraph, TensorId)> {
        let build = |schedule: &str| {
            let mut g = TensorGraph::new();
            let x = g.input(&[rows, dim]);
            let sq = g.map(&[x], |v| v[0] * v[0]).unwrap();
            let sum = g.reduce(sq, 1, Monoid::sum()).unwrap();
            let inv = g
                .map(&[sum], |v| {
                    ExprId::from_f64(1.0)
                        / (v[0] / ExprId::from_f64(dim as f64) + ExprId::from_f64(1e-6)).sqrt()
                })
                .unwrap();
            let inv = g.broadcast(inv, &[rows, dim]).unwrap();
            let y = g.map(&[x, inv], |v| v[0] * v[1]).unwrap();
            let (mut g, outs) = g.substitute_maps(&[y]);
            match schedule {
                "split" => g.schedule_split(&outs, 256, 64),
                "tiled" => g.schedule_tile(&outs, 64, 32, 64),
                "auto" => g.schedule_auto(&outs),
                _ => {}
            }
            (g, outs[0])
        };
        ["naive", "split", "tiled", "auto"]
            .into_iter()
            .map(|name| {
                let (g, y) = build(name);
                (name, g, y)
            })
            .collect()
    }

    /// `silu(gate) * up`: pure elementwise, one map.
    pub fn swiglu_graph(n: usize) -> (TensorGraph, TensorId) {
        let mut g = TensorGraph::new();
        let gate = g.input(&[n]);
        let up = g.input(&[n]);
        let y = g
            .map(&[gate, up], |v| {
                v[0] / (ExprId::from_f64(1.0) + (-v[0]).exp()) * v[1]
            })
            .unwrap();
        (g, y)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use tang_gpu::kv_attention_fused;
        use tang_gpu::GpuTensor;

        #[test]
        fn derived_rmsnorm_and_swiglu_match_hand_kernels_on_gpu() {
            let Ok(device) = GpuDevice::new_sync() else {
                eprintln!("no GPU, skipping");
                return;
            };
            let mut cache = KernelCache::new();
            let (rows, dim) = (37, 100);
            let mut rng = crate::Rng::new(5);
            let x: Vec<f32> = (0..rows * dim).map(|_| rng.f32() * 2.0 - 1.0).collect();
            let xt = GpuTensor::from_slice(&device, &x, &[rows, dim]);
            let norm = tang_gpu::GpuRMSNorm::new(&device, dim, 1e-6);
            let want = norm
                .forward(&device, &mut cache, &xt)
                .to_vec_flushed(&device, &mut cache);
            for (name, g, y) in rmsnorm_candidates(rows, dim) {
                let exe = Executable::new(&device, g.to_wgsl(&[y]), &[&x]);
                exe.run(&device, &mut cache);
                cache.flush(&device);
                let got = exe.read(&device, y);
                for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                    assert!((a - b).abs() < 1e-4, "rmsnorm {name}[{i}]: {a} vs {b}");
                }
            }

            let n = 1000;
            let gate: Vec<f32> = (0..n).map(|_| rng.f32() * 4.0 - 2.0).collect();
            let up: Vec<f32> = (0..n).map(|_| rng.f32() * 2.0 - 1.0).collect();
            let gt = GpuTensor::from_slice(&device, &gate, &[n]);
            let ut = GpuTensor::from_slice(&device, &up, &[n]);
            let want = tang_gpu::swiglu_fused_pub(&device, &mut cache, &gt, &ut)
                .to_vec_flushed(&device, &mut cache);
            let (g, y) = swiglu_graph(n);
            let exe = Executable::new(&device, g.to_wgsl(&[y]), &[&gate, &up]);
            exe.run(&device, &mut cache);
            cache.flush(&device);
            let got = exe.read(&device, y);
            for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                assert!((a - b).abs() < 1e-4, "swiglu[{i}]: {a} vs {b}");
            }
        }

        #[test]
        fn derived_matmul_matches_hand_kernel_on_gpu() {
            let Ok(device) = GpuDevice::new_sync() else {
                eprintln!("no GPU, skipping");
                return;
            };
            let mut cache = KernelCache::new();
            let (m, k, n) = (70, 50, 90);
            let mut rng = crate::Rng::new(3);
            let a: Vec<f32> = (0..m * k).map(|_| rng.f32() * 2.0 - 1.0).collect();
            let b: Vec<f32> = (0..k * n).map(|_| rng.f32() * 2.0 - 1.0).collect();
            let at = GpuTensor::from_slice(&device, &a, &[m, k]);
            let bt = GpuTensor::from_slice(&device, &b, &[k, n]);
            let want = tang_gpu::matmul::matmul(&device, &mut cache, &at, &bt)
                .to_vec_flushed(&device, &mut cache);
            for (name, g, c) in matmul_candidates(m, k, n) {
                let exe = Executable::new(&device, g.to_wgsl(&[c]), &[&a, &b]);
                exe.run(&device, &mut cache);
                cache.flush(&device);
                let got = exe.read(&device, c);
                for (i, (x, y)) in got.iter().zip(&want).enumerate() {
                    assert!((x - y).abs() < 1e-3, "{name}[{i}]: {x} vs {y}");
                }
            }
        }

        #[test]
        fn derived_attention_matches_hand_kernel_on_gpu() {
            let Ok(device) = GpuDevice::new_sync() else {
                eprintln!("no GPU, skipping");
                return;
            };
            let mut cache = KernelCache::new();
            let (n, d) = (100, 16);
            let mut rng = crate::Rng::new(7);
            let q: Vec<f32> = (0..n * d).map(|_| rng.f32() * 2.0 - 1.0).collect();
            let k: Vec<f32> = (0..n * d).map(|_| rng.f32() * 2.0 - 1.0).collect();
            let v: Vec<f32> = (0..n * d).map(|_| rng.f32() * 2.0 - 1.0).collect();

            let qt = GpuTensor::from_slice(&device, &q, &[n, d]);
            let kt = GpuTensor::from_slice(&device, &k, &[n, d]);
            let vt = GpuTensor::from_slice(&device, &v, &[n, d]);
            let want = kv_attention_fused(&device, &mut cache, &qt, &kt, &vt, n, n, 1, 1, d, 0)
                .to_vec_flushed(&device, &mut cache);
            let mask = causal_mask(n);

            // CPU reference
            let mut cpu = vec![0.0f32; n * d];
            for i in 0..n {
                let sc: Vec<f32> = (0..n)
                    .map(|j| {
                        let dot = (0..d).map(|t| q[i * d + t] * k[j * d + t]).sum::<f32>();
                        if j <= i {
                            dot / (d as f32).sqrt()
                        } else {
                            -1e30
                        }
                    })
                    .collect();
                let m = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let w: Vec<f32> = sc.iter().map(|s| (s - m).exp()).collect();
                let l: f32 = w.iter().sum();
                for c in 0..d {
                    cpu[i * d + c] = (0..n).map(|j| w[j] * v[j * d + c]).sum::<f32>() / l;
                }
            }
            for (i, (a, b)) in want.iter().zip(&cpu).enumerate() {
                assert!((a - b).abs() < 1e-4, "hand[{i}]: {a} vs {b}");
            }
            for (name, g, y) in attention_candidates(n, d, d) {
                let program = g.to_wgsl(&[y]);
                let exe = Executable::new(&device, program, &[&q, &k, &v, &mask]);
                exe.run(&device, &mut cache);
                cache.flush(&device);
                let got = exe.read(&device, y);
                assert_eq!(got.len(), want.len());
                for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                    assert!((a - b).abs() < 1e-4, "{name}[{i}]: {a} vs {b}");
                }
            }

            let (name, exe, y, t) = autotune(
                &device,
                &mut cache,
                attention_candidates(n, d, d),
                &[&q, &k, &v, &mask],
                3,
            );
            eprintln!("autotune picked {name} at {t:?}");
            let got = exe.read(&device, y);
            for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                assert!((a - b).abs() < 1e-4, "auto[{i}]: {a} vs {b}");
            }
        }
    }
}
