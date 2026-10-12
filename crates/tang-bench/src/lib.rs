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
    use tang_compile::tensor::{Arg, Form, Monoid, ScheduleCandidate, TensorGraph, TensorId, View};
    use tang_expr::ExprId;
    use tang_gpu::{GpuBuffer, GpuDevice, KernelCache};

    /// Buffers for a program: graph inputs in order, then materialized
    /// intermediates. Reused across runs.
    pub struct Executable {
        pub program: WgslProgram,
        /// `buffers[i]` backs `program.buffers[i].0`.
        pub buffers: Vec<GpuBuffer>,
    }

    /// The program for `device`. Native simdgroup-matrix tiles (Metal,
    /// `TANG_NATIVE=1`) are correct but, as first cut, slower than the
    /// WGSL register tile on an M4 (see BENCHMARKS.md), so WGSL is the
    /// default everywhere.
    pub fn program_for(device: &GpuDevice, g: &TensorGraph, outputs: &[TensorId]) -> WgslProgram {
        if device.msl_passthrough && std::env::var("TANG_NATIVE").is_ok() {
            g.to_msl(outputs)
        } else {
            g.to_wgsl(outputs)
        }
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

        /// The buffer holding `t` (inputs included), for feeding one
        /// program's outputs into another's inputs.
        pub fn buffer(&self, t: TensorId) -> &GpuBuffer {
            let t = self.program.storage(t);
            let i = self
                .program
                .buffers
                .iter()
                .position(|b| b.node == t)
                .unwrap();
            &self.buffers[i]
        }

        /// Buffer of graph input `k`.
        pub fn input_buffer(&self, k: usize) -> &GpuBuffer {
            let i = self
                .program
                .buffers
                .iter()
                .position(|b| b.input == Some(k))
                .expect("input not bound");
            &self.buffers[i]
        }

        /// Dispatch every kernel in order as one submission.
        pub fn run(&self, device: &GpuDevice, cache: &mut KernelCache) {
            cache.begin_batch();
            for i in 0..self.program.kernels.len() {
                self.run_kernel(device, cache, i);
            }
            // leaving the cache in batch mode would make a later
            // `to_vec_sync` (the hand path's readbacks) wait on work that
            // was never submitted
            cache.end_batch(device);
        }

        /// Dispatch kernel `i` only (its inputs must already be computed).
        pub fn run_kernel(&self, device: &GpuDevice, cache: &mut KernelCache, i: usize) {
            for k in std::iter::once(&self.program.kernels[i]) {
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
        causal_mask_kv(n, n)
    }

    /// `[q_len, kv_len]` mask for queries at positions `kv_len - q_len ..`:
    /// query `i` sees keys `j <= kv_len - q_len + i`.
    pub fn causal_mask_kv(q_len: usize, kv_len: usize) -> Vec<f32> {
        let start = kv_len - q_len;
        (0..q_len * kv_len)
            .map(|p| {
                if p % kv_len <= start + p / kv_len {
                    0.0
                } else {
                    -1e30
                }
            })
            .collect()
    }

    /// `[q_len, kv_len]` additive causal mask from coordinates: query `i`
    /// (at position `kv_len - q_len + i`) sees keys `j` up to that position.
    pub fn causal_mask_graph(g: &mut TensorGraph, q_len: usize, kv_len: usize) -> TensorId {
        let i = g.iota(&[q_len, kv_len], 0);
        let j = g.iota(&[q_len, kv_len], 1);
        let start = (kv_len - q_len) as f64;
        g.map(&[i, j], |v| {
            let above = v[1] - v[0] - ExprId::from_f64(start); // > 0 when j > start + i
            tang_expr::with_graph(|eg| {
                let neg = eg.lit(-1e30);
                eg.select(above, neg, ExprId::ZERO)
            })
        })
        .unwrap()
    }

    /// Single-head causal `softmax(q·kᵀ/√d + mask)·v`, written naively,
    /// mask built in-graph. Inputs: q `[n,d]`, k `[n,d]`, v `[n,dv]`.
    pub fn naive_attention(n: usize, d: usize, dv: usize) -> (TensorGraph, TensorId) {
        let mut g = TensorGraph::new();
        let q = g.input(&[n, d]);
        let k = g.input(&[n, d]);
        let v = g.input(&[n, dv]);
        let mask = causal_mask_graph(&mut g, n, n);
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
        let (g, outs) = g.stabilize_softmax(&[y]);
        let (mut g, outs) = g.substitute_maps(&outs);
        g.schedule_inline_up_to(&outs, dv);
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
        // lanes over the value column, 4 rows per workgroup
        g.schedule_tile_2d(&[y], dv as u32, 4, 32, 64);
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
        let (g, outs) = g.stabilize_softmax(&[y]);
        let (g, outs) = g.substitute_maps(&outs);
        let (g, outs) = g.eliminate_common(&outs);
        let (g, outs) = g.merge_siblings(&outs);
        let (g, outs) = g.relayout(&outs);
        let (mut g, outs) = g.merge_reduce_dims(&outs);
        g.schedule_auto(&outs);
        (g, outs[0])
    }

    /// One kernel's winning thread mapping and its measured time.
    pub struct TunedKernel {
        pub node: TensorId,
        pub split: u32,
        pub tile: (u32, u32, u32),
        pub reg: (u32, u32),
        pub time: std::time::Duration,
    }

    /// Per-kernel autotuning: for every materialized comp, time each
    /// thread mapping from `schedule_candidates` on this device (with the
    /// graph's real inputs already computed) and keep the fastest. The
    /// algebra and materialization are left as scheduled; only `split`
    /// and `tile` change, so buffers stay valid across variants.
    pub fn autotune_kernels(
        device: &GpuDevice,
        cache: &mut KernelCache,
        mut g: TensorGraph,
        outputs: &[TensorId],
        inputs: &[&[f32]],
        iters: usize,
    ) -> (TensorGraph, Vec<TunedKernel>) {
        let base = Executable::new(device, g.to_wgsl(outputs), inputs);
        base.run(device, cache);
        cache.flush(device);
        let _ = base.read(device, outputs[0]);
        let mut log = Vec::new();
        for i in 0..base.program.kernels.len() {
            let node = base.program.kernels[i].node;
            let mut best: Option<(ScheduleCandidate, std::time::Duration)> = None;
            for (split, tile, reg) in g.schedule_candidates(node) {
                let mut variant = g.clone();
                variant.set_schedule(node, split, tile, reg);
                let exe = Executable::new(device, variant.to_wgsl(outputs), inputs);
                // inputs of kernel i: run everything before it once
                for j in 0..i {
                    exe.run_kernel(device, cache, j);
                }
                exe.run_kernel(device, cache, i);
                cache.flush(device);
                let _ = exe.read(device, node);
                // one timed run first: a variant far behind the best so
                // far is not worth `iters` more
                let probe = std::time::Instant::now();
                exe.run_kernel(device, cache, i);
                cache.flush(device);
                let _ = exe.read(device, node);
                let probe = probe.elapsed();
                if best.is_some_and(|b| probe > b.1 * 4) {
                    continue;
                }
                let start = std::time::Instant::now();
                for _ in 0..iters {
                    exe.run_kernel(device, cache, i);
                }
                cache.flush(device);
                let _ = exe.read(device, node);
                let t = start.elapsed() / iters as u32;
                if best.is_none_or(|b| t < b.1) {
                    best = Some(((split, tile, reg), t));
                }
            }
            let ((split, tile, reg), time) = best.unwrap();
            g.set_schedule(node, split, tile, reg);
            log.push(TunedKernel {
                node,
                split,
                tile,
                reg,
                time,
            });
        }
        (g, log)
    }

    /// Schedules chosen by [`autotune_kernels`], stored per (program,
    /// device) so the measurement is paid once per chip. Plain text:
    /// one `node split bx by chunk` line per kernel under a header line.
    pub struct TuningCache {
        pub path: std::path::PathBuf,
    }

    /// `(node index, split, tile, reg)` as stored in the cache.
    type Entry = (usize, u32, (u32, u32, u32), (u32, u32));

    impl TuningCache {
        pub fn new(path: impl Into<std::path::PathBuf>) -> Self {
            Self { path: path.into() }
        }

        /// Hash of the untuned kernels' sources plus the device name.
        fn key(device: &GpuDevice, g: &TensorGraph, outputs: &[TensorId]) -> u64 {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            device.name.hash(&mut h);
            let mut plain = g.clone();
            for k in g.to_wgsl(outputs).kernels {
                plain.set_schedule(k.node, 0, (0, 0, 0), (1, 1));
            }
            for k in plain.to_wgsl(outputs).kernels {
                k.source.hash(&mut h);
            }
            h.finish()
        }

        fn load(&self, key: u64) -> Option<Vec<Entry>> {
            let text = std::fs::read_to_string(&self.path).ok()?;
            let mut lines = text.lines();
            let header = lines.next()?;
            if header != format!("tang-tuning {key:016x}") {
                return None;
            }
            lines
                .map(|l| {
                    let v: Vec<u32> = l
                        .split_whitespace()
                        .filter_map(|x| x.parse().ok())
                        .collect();
                    match v.len() {
                        5 => Some((v[0] as usize, v[1], (v[2], v[3], v[4]), (1, 1))),
                        7 => Some((v[0] as usize, v[1], (v[2], v[3], v[4]), (v[5], v[6]))),
                        _ => None,
                    }
                })
                .collect()
        }

        fn store(&self, key: u64, log: &[TunedKernel]) {
            let mut text = format!("tang-tuning {key:016x}\n");
            for k in log {
                text.push_str(&format!(
                    "{} {} {} {} {} {} {}\n",
                    k.node.index(),
                    k.split,
                    k.tile.0,
                    k.tile.1,
                    k.tile.2,
                    k.reg.0,
                    k.reg.1
                ));
            }
            if let Some(dir) = self.path.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            if let Err(e) = std::fs::write(&self.path, text) {
                eprintln!("tuning cache: could not write {}: {e}", self.path.display());
            }
        }

        /// Tuned graph: from the cache if the (program, device) matches,
        /// otherwise measured now and cached.
        pub fn tune(
            &self,
            device: &GpuDevice,
            cache: &mut KernelCache,
            g: TensorGraph,
            outputs: &[TensorId],
            inputs: &[&[f32]],
        ) -> (TensorGraph, bool) {
            let key = Self::key(device, &g, outputs);
            if let Some(entries) = self.load(key) {
                let mut g = g;
                for (node, split, tile, reg) in entries {
                    g.set_schedule(TensorId::from_index(node), split, tile, reg);
                }
                return (g, true);
            }
            let (g, log) = autotune_kernels(device, cache, g, outputs, inputs, 10);
            self.store(key, &log);
            (g, false)
        }
    }

    /// One Adam step as three maps over `[n]`: returns `(m', v', p')`.
    /// Inputs: `p, g, m, v, bc` where `bc = [1 - β1ᵗ, 1 - β2ᵗ]` is a
    /// 2-element input so the step count changes without recompiling.
    pub fn adam_step(
        n: usize,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
    ) -> (TensorGraph, [TensorId; 3]) {
        let mut g = TensorGraph::new();
        let p = g.input(&[n]);
        let grad = g.input(&[n]);
        let m = g.input(&[n]);
        let v = g.input(&[n]);
        let bc = g.input(&[2]);
        let bc_view = |k: i64| View {
            forms: vec![Form {
                coefs: vec![0],
                offset: k,
            }],
        };
        let m2 = g
            .map(&[m, grad], |x| {
                x[0] * ExprId::from_f64(beta1) + x[1] * ExprId::from_f64(1.0 - beta1)
            })
            .unwrap();
        let v2 = g
            .map(&[v, grad], |x| {
                x[0] * ExprId::from_f64(beta2) + x[1] * x[1] * ExprId::from_f64(1.0 - beta2)
            })
            .unwrap();
        let id = |t: TensorId| Arg {
            src: t,
            view: View::identity(1, 1),
        };
        let p2 = g
            .comp(
                &[n],
                &[],
                vec![
                    id(p),
                    id(m2),
                    id(v2),
                    Arg {
                        src: bc,
                        view: bc_view(0),
                    },
                    Arg {
                        src: bc,
                        view: bc_view(1),
                    },
                ],
                None,
                |x| {
                    let m_hat = x[1] / x[3];
                    let v_hat = x[2] / x[4];
                    vec![
                        x[0] - ExprId::from_f64(lr) * m_hat
                            / (v_hat.sqrt() + ExprId::from_f64(eps)),
                    ]
                },
            )
            .unwrap();
        (g, [m2, v2, p2])
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
                "reg2" => g.set_schedule(c, 0, (16, 16, 32), (2, 2)),
                "reg4" => g.set_schedule(c, 0, (16, 16, 32), (4, 4)),
                "reg8" => g.set_schedule(c, 0, (16, 16, 32), (8, 8)),
                "reg4c64" => g.set_schedule(c, 0, (16, 16, 64), (4, 4)),
                "reg8x4" => g.set_schedule(c, 0, (16, 16, 32), (8, 4)),
                "reg4x8" => g.set_schedule(c, 0, (16, 16, 32), (4, 8)),
                "reg4c16" => g.set_schedule(c, 0, (16, 16, 16), (4, 4)),
                "reg8c16" => g.set_schedule(c, 0, (16, 16, 16), (8, 8)),
                "reg8c8" => g.set_schedule(c, 0, (16, 16, 8), (8, 8)),
                "reg8b8" => g.set_schedule(c, 0, (8, 8, 32), (8, 8)),
                "reg4b32" => g.set_schedule(c, 0, (32, 8, 32), (4, 4)),
                "auto" => g.schedule_auto(&[c]),
                _ => {}
            }
            (g, c)
        };
        [
            "naive", "split", "tiled", "reg2", "reg4", "reg8", "reg4c64", "reg8x4", "reg4x8",
            "reg4c16", "reg8c16", "reg8c8", "reg8b8", "reg4b32", "auto",
        ]
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
        fn derived_adam_matches_hand_kernel_on_gpu() {
            let Ok(device) = GpuDevice::new_sync() else {
                eprintln!("no GPU, skipping");
                return;
            };
            let mut cache = KernelCache::new();
            let n = 1000;
            let mut rng = crate::Rng::new(8);
            let mut rand = |s: f32| {
                (0..n)
                    .map(|_| (rng.f32() * 2.0 - 1.0) * s)
                    .collect::<Vec<f32>>()
            };
            let (p, g, m, v) = (
                rand(1.0),
                rand(0.1),
                rand(0.01),
                rand(0.001).iter().map(|x| x.abs()).collect::<Vec<f32>>(),
            );
            let (lr, b1, b2, eps, bc1, bc2) =
                (1e-3f32, 0.9f32, 0.999f32, 1e-8f32, 0.19f32, 0.002f32);

            let pb = GpuBuffer::from_slice(&device, &p);
            let gb = GpuBuffer::from_slice(&device, &g);
            let mb = GpuBuffer::from_slice(&device, &m);
            let vb = GpuBuffer::from_slice(&device, &v);
            cache.dispatch_adam(
                &device, &pb, &gb, &mb, &vb, n as u32, lr, b1, b2, eps, bc1, bc2,
            );
            cache.flush(&device);
            let want = [
                mb.to_vec_sync(&device),
                vb.to_vec_sync(&device),
                pb.to_vec_sync(&device),
            ];

            let (graph, outs) = adam_step(n, lr as f64, b1 as f64, b2 as f64, eps as f64);
            let bc = [bc1, bc2];
            let exe = Executable::new(&device, graph.to_wgsl(&outs), &[&p, &g, &m, &v, &bc]);
            exe.run(&device, &mut cache);
            cache.flush(&device);
            for (k, o) in outs.iter().enumerate() {
                let got = exe.read(&device, *o);
                for (i, (a, b)) in got.iter().zip(&want[k]).enumerate() {
                    assert!(
                        (a - b).abs() < 1e-5 * (1.0 + b.abs()),
                        "adam out {k}[{i}]: {a} vs {b}"
                    );
                }
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

        /// On Metal, the tiled matmul (and its bias epilogue) lowers to
        /// simdgroup matrices; same answer as the WGSL tile.
        #[test]
        fn simdgroup_matrix_matmul_matches_on_metal() {
            let Ok(device) = GpuDevice::new_sync() else {
                return;
            };
            if !device.msl_passthrough {
                eprintln!("no MSL passthrough on {}, skipping", device.name);
                return;
            }
            // ragged shapes take the staged path, full tiles the direct one
            for (m, k, n) in [(130, 70, 200), (128, 64, 192)] {
                check_simdgroup_matmul(&device, m, k, n);
            }
        }

        fn check_simdgroup_matmul(device: &GpuDevice, m: usize, k: usize, n: usize) {
            let mut cache = KernelCache::new();
            let mut rng = crate::Rng::new(5);
            let a: Vec<f32> = (0..m * k).map(|_| rng.f32() * 2.0 - 1.0).collect();
            let b: Vec<f32> = (0..k * n).map(|_| rng.f32() * 2.0 - 1.0).collect();
            let bias: Vec<f32> = (0..n).map(|_| rng.f32()).collect();
            // y = a·bᵀ-free plain matmul + bias, then silu: the bias and
            // activation are the tile's epilogue
            let mut g = TensorGraph::new();
            let ai = g.input(&[m, k]);
            let bi = g.input(&[k, n]);
            let bb = g.input(&[n]);
            let c = g.matmul(ai, bi).unwrap();
            let bb1 = g.reshape(bb, &[1, n]).unwrap();
            let bbc = g.broadcast(bb1, &[m, n]).unwrap();
            let y = g
                .map(&[c, bbc], |v| {
                    let s = v[0] + v[1];
                    s / (ExprId::from_f64(1.0) + (-s).exp())
                })
                .unwrap();
            let (mut g, outs) = g.substitute_maps(&[y]);
            g.schedule_auto(&outs);
            let f64s = |v: &[f32]| v.iter().map(|&x| x as f64).collect::<Vec<f64>>();
            let want = g.eval(&[f64s(&a), f64s(&b), f64s(&bias)], &outs).remove(0);
            let prog = g.to_msl(&outs);
            assert!(
                prog.kernels
                    .iter()
                    .any(|k| k.source.starts_with("#include")),
                "expected a native kernel"
            );
            let exe = Executable::new(device, prog, &[&a, &b, &bias]);
            exe.run(device, &mut cache);
            cache.flush(device);
            let got = exe.read(device, outs[0]);
            assert_eq!(got.len(), want.len());
            for (i, (&x, &y)) in got.iter().zip(&want).enumerate() {
                assert!((x as f64 - y).abs() < 2e-3, "{m}x{k}x{n} [{i}]: {x} vs {y}");
            }
        }

        #[test]
        fn derived_attention_matches_hand_kernel_on_gpu() {
            let Ok(device) = GpuDevice::new_sync() else {
                eprintln!("no GPU, skipping");
                return;
            };
            let mut cache = KernelCache::new();
            for (n, d) in [(100, 16), (512, 64), (1024, 64)] {
                check_attention_size(&device, &mut cache, n, d);
            }
        }

        fn check_attention_size(device: &GpuDevice, cache: &mut KernelCache, n: usize, d: usize) {
            let mut rng = crate::Rng::new(7);
            let q: Vec<f32> = (0..n * d).map(|_| rng.f32() * 2.0 - 1.0).collect();
            let k: Vec<f32> = (0..n * d).map(|_| rng.f32() * 2.0 - 1.0).collect();
            let v: Vec<f32> = (0..n * d).map(|_| rng.f32() * 2.0 - 1.0).collect();

            let qt = GpuTensor::from_slice(device, &q, &[n, d]);
            let kt = GpuTensor::from_slice(device, &k, &[n, d]);
            let vt = GpuTensor::from_slice(device, &v, &[n, d]);
            let want = kv_attention_fused(device, cache, &qt, &kt, &vt, n, n, 1, 1, d, 0)
                .to_vec_flushed(device, cache);

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
                let exe = Executable::new(device, program, &[&q, &k, &v]);
                exe.run(device, cache);
                cache.flush(device);
                let got = exe.read(device, y);
                assert_eq!(got.len(), want.len());
                for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                    assert!((a - b).abs() < 1e-4, "{name}[{i}]: {a} vs {b}");
                }
            }

            let (name, exe, y, t) = autotune(
                device,
                cache,
                attention_candidates(n, d, d),
                &[&q, &k, &v],
                3,
            );
            eprintln!("autotune picked {name} at {t:?}");
            let got = exe.read(device, y);
            for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                assert!((a - b).abs() < 1e-4, "auto[{i}]: {a} vs {b}");
            }
        }
    }
}

/// A whole transformer block derived from tensor IR, mirroring the
/// hand-written `gpu_transformer_layer` prefill path in `gpu_inference`.

/// Running a [`TensorGraph::to_cuda`](tang_compile::tensor::TensorGraph::to_cuda)
/// program on an NVIDIA GPU through the CUDA driver (cudarc): nvrtc
/// compiles each kernel — wmma tensor-core tiles and transpiled WGSL
/// alike — and the kernels launch in program order on one stream.
#[cfg(feature = "cuda")]
pub mod ir_cuda {
    use std::sync::Arc;

    use cudarc::driver::{
        CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
    };
    use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
    use tang_compile::tensor::wgsl::WgslProgram;
    use tang_compile::tensor::TensorId;

    pub struct CudaExecutable {
        pub program: WgslProgram,
        pub stream: Arc<CudaStream>,
        buffers: Vec<CudaSlice<f32>>,
        functions: Vec<CudaFunction>,
    }

    impl CudaExecutable {
        /// Compile every kernel and allocate every buffer; inputs uploaded.
        pub fn new(ctx: &Arc<CudaContext>, program: WgslProgram, inputs: &[&[f32]]) -> Self {
            let stream = ctx.default_stream();
            let buffers = program
                .buffers
                .iter()
                .map(|b| match b.input {
                    Some(k) => stream.memcpy_stod(inputs[k]).expect("upload"),
                    None => stream.alloc_zeros::<f32>(b.len.max(1)).expect("alloc"),
                })
                .collect();
            let include = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
            let functions = program
                .kernels
                .iter()
                .map(|k| {
                    let opts = CompileOptions {
                        arch: Some("sm_86"),
                        include_paths: vec![format!("{include}/include")],
                        ..Default::default()
                    };
                    let ptx = compile_ptx_with_opts(&k.source, opts)
                        .unwrap_or_else(|e| panic!("nvrtc: {e:?}\n{}", k.source));
                    let module = ctx.load_module(ptx).expect("load module");
                    module
                        .load_function(tang_compile::tensor::cuda::CUDA_ENTRY)
                        .expect("entry")
                })
                .collect();
            Self {
                program,
                stream,
                buffers,
                functions,
            }
        }

        fn index(&self, t: TensorId) -> usize {
            let t = self.program.storage(t);
            self.program
                .buffers
                .iter()
                .position(|b| b.node == t)
                .expect("buffer")
        }

        /// Launch every kernel in order (asynchronously on the stream).
        pub fn run(&self) {
            for i in 0..self.program.kernels.len() {
                self.run_kernel(i);
            }
        }

        /// Launch kernel `i` only.
        pub fn run_kernel(&self, i: usize) {
            for (k, f) in std::iter::once((&self.program.kernels[i], &self.functions[i])) {
                let groups = k.count.div_ceil(k.workgroup_size);
                let cfg = LaunchConfig {
                    grid_dim: (groups, 1, 1),
                    block_dim: (k.workgroup_size, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut launch = self.stream.launch_builder(f);
                for t in &k.reads {
                    launch.arg(&self.buffers[self.index(*t)]);
                }
                launch.arg(&self.buffers[self.index(k.node)]);
                launch.arg(&k.points);
                unsafe { launch.launch(cfg) }.expect("launch");
            }
        }

        pub fn read(&self, t: TensorId) -> Vec<f32> {
            self.stream.synchronize().expect("sync");
            self.stream
                .memcpy_dtov(&self.buffers[self.index(t)])
                .expect("download")
        }

        /// Milliseconds per run over `iters` launches of the whole program
        /// (wall clock around a stream sync; launches are asynchronous).
        pub fn time(&self, iters: usize) -> f64 {
            self.time_with(iters, |e| e.run())
        }

        /// Milliseconds per launch of kernel `i` alone.
        pub fn time_kernel(&self, i: usize, iters: usize) -> f64 {
            self.time_with(iters, |e| e.run_kernel(i))
        }

        fn time_with(&self, iters: usize, f: impl Fn(&Self)) -> f64 {
            for _ in 0..3 {
                f(self);
            }
            self.stream.synchronize().expect("sync");
            let t = std::time::Instant::now();
            for _ in 0..iters {
                f(self);
            }
            self.stream.synchronize().expect("sync");
            t.elapsed().as_secs_f64() * 1e3 / iters as f64
        }
    }
}

pub mod ir_block {
    use tang::Scalar;
    use tang_compile::tensor::{numel, Arg, Form, Monoid, TensorGraph, TensorId, View};
    use tang_expr::ExprId;

    pub struct BlockDims {
        /// query tokens
        pub seq: usize,
        /// keys/values attended to. `== seq`: prefill, k/v computed from
        /// `x`. `> seq`: decode, rotated k/v caches are graph inputs and
        /// the queries sit at positions `kv_len - seq ..`.
        pub kv_len: usize,
        /// interleaved RoPE on q/k (the inference block); off matches the
        /// hand-written trainable block, which has neither rope nor biases
        pub rope: bool,
        /// biases on every projection
        pub bias: bool,
        pub d_model: usize,
        pub n_heads: usize,
        pub n_kv_heads: usize,
        pub head_dim: usize,
        pub ff_dim: usize,
    }

    /// Input order for the graph's buffers.
    pub const INPUTS: [&str; 17] = [
        "x", "wq", "bq", "wk", "bk", "wv", "bv", "wo", "bo", "w_gate", "b_gate", "w_up", "b_up",
        "w_down", "b_down", "cos", "sin",
    ];
    // Decode adds `k_cache`, `v_cache` (rotated, `[kv_len, kv_dim]`) after
    // `sin`. The causal mask is built in-graph from coordinates.

    fn unit(rank: usize, d: usize) -> Form {
        let mut f = Form {
            coefs: vec![0; rank],
            offset: 0,
        };
        f.coefs[d] = 1;
        f
    }

    fn form(rank: usize, terms: &[(usize, i64)], offset: i64) -> Form {
        let mut f = Form {
            coefs: vec![0; rank],
            offset,
        };
        for &(d, c) in terms {
            f.coefs[d] += c;
        }
        f
    }

    /// `x · wᵀ + b` for `w: [out, in]`, `b: [out]` (`b = None`: no bias).
    fn linear(g: &mut TensorGraph, x: TensorId, w: TensorId, b: Option<TensorId>) -> TensorId {
        let wt = g.transpose(w, &[1, 0]).unwrap();
        let y = g.matmul(x, wt).unwrap();
        let Some(b) = b else { return y };
        let shape = g.shape(y).to_vec();
        let rank = shape.len();
        let args = vec![
            Arg {
                src: y,
                view: View::identity(rank, rank),
            },
            Arg {
                src: b,
                view: View {
                    forms: vec![unit(rank, 1)],
                },
            },
        ];
        g.comp(&shape, &[], args, None, |v| vec![v[0] + v[1]])
            .unwrap()
    }

    fn rmsnorm(g: &mut TensorGraph, x: TensorId) -> TensorId {
        let shape = g.shape(x).to_vec();
        let dim = shape[1].n();
        let sq = g.map(&[x], |v| v[0] * v[0]).unwrap();
        let sum = g.reduce(sq, 1, Monoid::sum()).unwrap();
        let inv = g
            .map(&[sum], |v| {
                ExprId::from_f64(1.0)
                    / (v[0] / ExprId::from_f64(dim as f64) + ExprId::from_f64(1e-6)).sqrt()
            })
            .unwrap();
        let inv = g.broadcast(inv, &shape).unwrap();
        g.map(&[x, inv], |v| v[0] * v[1]).unwrap()
    }

    /// Interleaved RoPE on `x: [seq, heads*head_dim]` as two tensors of
    /// even/odd rotated components, `[seq, heads, head_dim/2]`, scaled.
    #[allow(clippy::too_many_arguments)]
    fn rope_split(
        g: &mut TensorGraph,
        x: TensorId,
        heads: usize,
        head_dim: usize,
        cos: TensorId,
        sin: TensorId,
        scale: f64,
        rope: bool,
    ) -> (TensorId, TensorId) {
        let seq = g.shape(x)[0].n();
        let half = head_dim / 2;
        let out = [seq, heads, half];
        // x viewed as [s, h, i, parity]: every read below is a pick view,
        // which is what makes the block differentiable through rope.
        let x = g.reshape(x, &[seq, heads, half, 2]).unwrap();
        let xv = |parity: i64| View {
            forms: vec![unit(3, 0), unit(3, 1), unit(3, 2), form(3, &[], parity)],
        };
        let tv = View {
            forms: vec![unit(3, 0), unit(3, 2)],
        };
        if !rope {
            // no rotation: just the even/odd halves, scaled
            let halves = [0i64, 1].map(|parity| {
                g.comp(
                    &out,
                    &[],
                    vec![Arg {
                        src: x,
                        view: xv(parity),
                    }],
                    None,
                    |v| vec![v[0] * ExprId::from_f64(scale)],
                )
                .unwrap()
            });
            return (halves[0], halves[1]);
        }
        let args = |g: &TensorGraph| {
            let _ = g;
            vec![
                Arg {
                    src: x,
                    view: xv(0),
                },
                Arg {
                    src: x,
                    view: xv(1),
                },
                Arg {
                    src: cos,
                    view: tv.clone(),
                },
                Arg {
                    src: sin,
                    view: tv.clone(),
                },
            ]
        };
        // Literals must be created inside the body closure: `from_f64`
        // writes to the graph being traced.
        let even = g
            .comp(&out, &[], args(g), None, |v| {
                vec![(v[0] * v[2] - v[1] * v[3]) * ExprId::from_f64(scale)]
            })
            .unwrap();
        let odd = g
            .comp(&out, &[], args(g), None, |v| {
                vec![(v[0] * v[3] + v[1] * v[2]) * ExprId::from_f64(scale)]
            })
            .unwrap();
        (even, odd)
    }

    /// Intermediates of the IR block, for stage-by-stage comparison.
    pub struct BlockNodes {
        pub normed: TensorId,
        pub q: TensorId,
        pub qe: TensorId,
        pub attn: TensorId,
        pub proj: TensorId,
        pub out: TensorId,
    }

    /// Build the block. Returns the graph and its output `[seq, d_model]`.
    pub fn transformer_block(dims: &BlockDims) -> (TensorGraph, TensorId) {
        let (g, n) = transformer_block_nodes(dims);
        (g, n.out)
    }

    pub fn transformer_block_nodes(dims: &BlockDims) -> (TensorGraph, BlockNodes) {
        let BlockDims {
            seq,
            kv_len,
            rope,
            bias,
            d_model,
            n_heads,
            n_kv_heads,
            head_dim,
            ff_dim,
        } = *dims;
        let decode = kv_len != seq;
        let b = |t: TensorId| bias.then_some(t);
        let groups = n_kv_heads;
        let per_group = n_heads / n_kv_heads;
        let half = head_dim / 2;
        let kv_dim = n_kv_heads * head_dim;

        let mut g = TensorGraph::new();
        let x = g.input(&[seq, d_model]);
        let wq = g.input(&[d_model, d_model]);
        let bq = g.input(&[d_model]);
        let wk = g.input(&[kv_dim, d_model]);
        let bk = g.input(&[kv_dim]);
        let wv = g.input(&[kv_dim, d_model]);
        let bv = g.input(&[kv_dim]);
        let wo = g.input(&[d_model, d_model]);
        let bo = g.input(&[d_model]);
        let w_gate = g.input(&[ff_dim, d_model]);
        let b_gate = g.input(&[ff_dim]);
        let w_up = g.input(&[ff_dim, d_model]);
        let b_up = g.input(&[ff_dim]);
        let w_down = g.input(&[d_model, ff_dim]);
        let b_down = g.input(&[d_model]);
        let cos = g.input(&[seq, half]);
        let sin = g.input(&[seq, half]);

        let caches = decode.then(|| (g.input(&[kv_len, kv_dim]), g.input(&[kv_len, kv_dim])));

        let normed = rmsnorm(&mut g, x);
        let q = linear(&mut g, normed, wq, b(bq));
        let scale = 1.0 / (head_dim as f64).sqrt();
        let (qe, qo) = rope_split(&mut g, q, n_heads, head_dim, cos, sin, scale, rope);
        let (ke, ko, v) = match caches {
            // cache rows are already rotated: split even/odd by view only
            Some((k_cache, v_cache)) => {
                let k4 = g.reshape(k_cache, &[kv_len, n_kv_heads, half, 2]).unwrap();
                let halves = [0i64, 1].map(|parity| {
                    let view = View {
                        forms: vec![unit(3, 0), unit(3, 1), unit(3, 2), form(3, &[], parity)],
                    };
                    g.comp(
                        &[kv_len, n_kv_heads, half],
                        &[],
                        vec![Arg { src: k4, view }],
                        None,
                        |v| vec![v[0]],
                    )
                    .unwrap()
                });
                (halves[0], halves[1], v_cache)
            }
            None => {
                let k = linear(&mut g, normed, wk, b(bk));
                let v = linear(&mut g, normed, wv, b(bv));
                let (ke, ko) = rope_split(&mut g, k, n_kv_heads, head_dim, cos, sin, 1.0, rope);
                (ke, ko, v)
            }
        };

        // scores[gr, r, i, j] = Σ_i' qe[i, 3g+r, i']·ke[j, g, i'] + qo·ko  (+ causal mask)
        // space: (g, r, i, j, i')
        let r5 = 5;
        // heads as [group, member]: q read through a reshape, pick views only
        let qe = g.reshape(qe, &[seq, groups, per_group, half]).unwrap();
        let qo = g.reshape(qo, &[seq, groups, per_group, half]).unwrap();
        let qv = View {
            forms: vec![unit(r5, 2), unit(r5, 0), unit(r5, 1), unit(r5, 4)],
        };
        let kv = View {
            forms: vec![unit(r5, 3), unit(r5, 0), unit(r5, 4)],
        };
        let s_out = [groups, per_group, seq, kv_len];
        let scores = g
            .comp(
                &s_out,
                &[half],
                vec![
                    Arg {
                        src: qe,
                        view: qv.clone(),
                    },
                    Arg {
                        src: ke,
                        view: kv.clone(),
                    },
                    Arg { src: qo, view: qv },
                    Arg { src: ko, view: kv },
                ],
                Some(Monoid::sum()),
                |a| vec![a[0] * a[1] + a[2] * a[3]],
            )
            .unwrap();
        let mask = super::ir_gpu::causal_mask_graph(&mut g, seq, kv_len);
        let r4 = 4;
        let masked = g
            .comp(
                &s_out,
                &[],
                vec![
                    Arg {
                        src: scores,
                        view: View::identity(r4, r4),
                    },
                    Arg {
                        src: mask,
                        view: View {
                            forms: vec![unit(r4, 2), unit(r4, 3)],
                        },
                    },
                ],
                None,
                |a| vec![a[0] + a[1]],
            )
            .unwrap();
        let m = g.reduce(masked, 3, Monoid::max()).unwrap();
        let mb = g.broadcast(m, &s_out).unwrap();
        let p = g.map(&[masked, mb], |a| (a[0] - a[1]).exp()).unwrap();
        let l = g.reduce(p, 3, Monoid::sum()).unwrap();
        // o[g, r, i, c] = Σ_j p[g,r,i,j] · v[j, g*head_dim + c]; space (g, r, i, c, j)
        let o_out = [groups, per_group, seq, head_dim];
        let v3 = g.reshape(v, &[kv_len, groups, head_dim]).unwrap();
        let o = g
            .comp(
                &o_out,
                &[kv_len],
                vec![
                    Arg {
                        src: p,
                        view: View {
                            forms: vec![unit(5, 0), unit(5, 1), unit(5, 2), unit(5, 4)],
                        },
                    },
                    Arg {
                        src: v3,
                        view: View {
                            forms: vec![unit(5, 4), unit(5, 0), unit(5, 3)],
                        },
                    },
                ],
                Some(Monoid::sum()),
                |a| vec![a[0] * a[1]],
            )
            .unwrap();
        let lb = g
            .comp(
                &o_out,
                &[],
                vec![Arg {
                    src: l,
                    view: View {
                        forms: vec![
                            unit(4, 0),
                            unit(4, 1),
                            unit(4, 2),
                            Form {
                                coefs: vec![0; 4],
                                offset: 0,
                            },
                        ],
                    },
                }],
                None,
                |a| vec![a[0]],
            )
            .unwrap();
        let y = g.map(&[o, lb], |a| a[0] / a[1]).unwrap();

        // proj[i, n] = Σ_{g,r,c} y[g,r,i,c] · wo[n, (3g+r)*head_dim + c] + bo[n]
        // space (i, n, g, r, c)
        let wo4 = g
            .reshape(wo, &[d_model, groups, per_group, head_dim])
            .unwrap();
        let proj = g
            .comp(
                &[seq, d_model],
                &[groups, per_group, head_dim],
                vec![
                    Arg {
                        src: y,
                        view: View {
                            forms: vec![unit(5, 2), unit(5, 3), unit(5, 0), unit(5, 4)],
                        },
                    },
                    Arg {
                        src: wo4,
                        view: View {
                            forms: vec![unit(5, 1), unit(5, 2), unit(5, 3), unit(5, 4)],
                        },
                    },
                ],
                Some(Monoid::sum()),
                |a| vec![a[0] * a[1]],
            )
            .unwrap();
        let bob = if bias {
            g.broadcast_bias(proj, bo)
        } else {
            proj
        };
        let res1 = g.map(&[x, bob], |a| a[0] + a[1]).unwrap();
        let normed2 = rmsnorm(&mut g, res1);
        let gate = linear(&mut g, normed2, w_gate, b(b_gate));
        let up = linear(&mut g, normed2, w_up, b(b_up));
        let act = g
            .map(&[gate, up], |a| {
                a[0] / (ExprId::from_f64(1.0) + (-a[0]).exp()) * a[1]
            })
            .unwrap();
        let ffn = linear(&mut g, act, w_down, b(b_down));
        let out = g.map(&[res1, ffn], |a| a[0] + a[1]).unwrap();
        (
            g,
            BlockNodes {
                normed,
                q,
                qe,
                attn: y,
                proj: bob,
                out,
            },
        )
    }

    trait BiasExt {
        fn broadcast_bias(&mut self, x: TensorId, b: TensorId) -> TensorId;
    }

    impl BiasExt for TensorGraph {
        fn broadcast_bias(&mut self, x: TensorId, b: TensorId) -> TensorId {
            let shape = self.shape(x).to_vec();
            let args = vec![
                Arg {
                    src: x,
                    view: View::identity(2, 2),
                },
                Arg {
                    src: b,
                    view: View {
                        forms: vec![unit(2, 1)],
                    },
                },
            ];
            self.comp(&shape, &[], args, None, |v| vec![v[0] + v[1]])
                .unwrap()
        }
    }

    /// `loss = mean((block(x) - target)²)` with gradients for every weight.
    /// Returns the graph, `[loss, dwq, dwk, dwv, dwo, dgate, dup, ddown]`
    /// and the index of the `target` input (last).
    pub fn train_loss_graph(dims: &BlockDims) -> (TensorGraph, Vec<TensorId>, usize) {
        let (mut g, out) = transformer_block(dims);
        let shape = g.shape(out).to_vec();
        let target = g.input(&shape);
        let n = numel(&shape) as f64;
        let sq = g
            .map(&[out, target], |v| (v[0] - v[1]) * (v[0] - v[1]))
            .unwrap();
        let flat = g.reshape(sq, &[numel(&shape)]).unwrap();
        let sum = g.reduce(flat, 0, Monoid::sum()).unwrap();
        let loss = g.map(&[sum], |v| v[0] / ExprId::from_f64(n)).unwrap();
        let wrt: Vec<TensorId> = [1usize, 3, 5, 7, 9, 11, 13]
            .iter()
            .map(|&i| TensorId::from_index(i))
            .collect();
        let grads = g.backward(loss, &wrt).unwrap();
        let mut outs = vec![loss];
        outs.extend(grads);
        let target_idx = INPUTS.len() + if dims.kv_len != dims.seq { 2 } else { 0 };
        (g, outs, target_idx)
    }

    /// Full compiler pipeline for the block.
    pub fn compiled_block(dims: &BlockDims) -> (TensorGraph, TensorId) {
        let (g, out) = transformer_block(dims);
        let (g, outs) = g.stabilize_softmax(&[out]);
        let (g, outs) = g.substitute_maps(&outs);
        let (g, outs) = g.eliminate_common(&outs);
        let (g, outs) = g.merge_siblings(&outs);
        let (g, outs) = g.relayout(&outs);
        let (mut g, outs) = g.merge_reduce_dims(&outs);
        g.schedule_auto(&outs);
        (g, outs[0])
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// The whole block is differentiable: gradients of `sum(out)`
        /// w.r.t. the input and a weight match finite differences.
        #[test]
        fn block_backward_matches_finite_differences() {
            let dims = BlockDims {
                seq: 3,
                kv_len: 3,
                rope: true,
                bias: true,
                d_model: 8,
                n_heads: 2,
                n_kv_heads: 1,
                head_dim: 4,
                ff_dim: 6,
            };
            let (mut g, out) = transformer_block(&dims);
            let x = TensorId::from_index(0);
            let wq = TensorId::from_index(1);
            let grads = g.backward(out, &[x, wq]).unwrap();

            let mut rng = crate::Rng::new(5);
            let mut inputs: Vec<Vec<f64>> = Vec::new();
            let sizes = [
                3 * 8, // x
                8 * 8,
                8, // wq, bq
                4 * 8,
                4, // wk, bk
                4 * 8,
                4, // wv, bv
                8 * 8,
                8, // wo, bo
                6 * 8,
                6, // gate
                6 * 8,
                6, // up
                8 * 6,
                8, // down
                3 * 2,
                3 * 2, // cos, sin
            ];
            for n in sizes {
                inputs.push((0..n).map(|_| rng.f64() * 0.8 - 0.4).collect());
            }
            // real rope tables for positions 0..3
            for pos in 0..3 {
                for i in 0..2 {
                    let theta = pos as f64 / 10000f64.powf(2.0 * i as f64 / 4.0);
                    inputs[15][pos * 2 + i] = theta.cos();
                    inputs[16][pos * 2 + i] = theta.sin();
                }
            }

            let f = |inp: &[Vec<f64>]| g.eval(inp, &[out]).remove(0).iter().sum::<f64>();
            let got = g.eval(&inputs, &grads);
            for (which, idx) in [(0usize, 0usize), (1, 1)] {
                let h = 1e-5;
                for i in 0..inputs[idx].len() {
                    let mut plus = inputs.clone();
                    plus[idx][i] += h;
                    let mut minus = inputs.clone();
                    minus[idx][i] -= h;
                    let fd = (f(&plus) - f(&minus)) / (2.0 * h);
                    let an = got[which][i];
                    assert!(
                        (an - fd).abs() <= 1e-4 * (1.0 + fd.abs()),
                        "input {idx}[{i}]: analytic {an} vs fd {fd}"
                    );
                }
            }
        }
    }
}

/// Hand-written vs derived transformer block on the GPU.
pub mod ir_block_gpu {
    use super::ir_block::{compiled_block, transformer_block, BlockDims};
    use super::ir_gpu::{program_for, Executable};
    use tang_gpu::nn::bias_add;
    use tang_gpu::*;

    /// The hand path's parameters, plus CPU copies for the IR graph.
    pub struct Params {
        pub ln1: GpuRMSNorm,
        pub ln2: GpuRMSNorm,
        pub wq: GpuLinear,
        pub wk: GpuLinear,
        pub wv: GpuLinear,
        pub wo: GpuLinear,
        pub gate: GpuLinear,
        pub up: GpuLinear,
        pub down: GpuLinear,
        pub rope: GpuInterleavedRoPE,
        /// Rotated k/v caches `[kv_len, kv_dim]` for decode dims.
        pub k_cache: Option<GpuTensor>,
        pub v_cache: Option<GpuTensor>,
        pub dims: BlockDims,
        /// IR inputs in `ir_block::INPUTS` order (x first).
        pub cpu: Vec<Vec<f32>>,
    }

    pub fn params(
        device: &GpuDevice,
        cache: &mut KernelCache,
        dims: BlockDims,
        x: Vec<f32>,
    ) -> Params {
        let BlockDims {
            seq,
            kv_len,
            d_model,
            n_kv_heads,
            head_dim,
            ff_dim,
            ..
        } = dims;
        let kv = n_kv_heads * head_dim;
        let wq = GpuLinear::kaiming(device, d_model, d_model, 42);
        let wk = GpuLinear::kaiming(device, d_model, kv, 43);
        let wv = GpuLinear::kaiming(device, d_model, kv, 44);
        let wo = GpuLinear::kaiming(device, d_model, d_model, 45);
        let gate = GpuLinear::kaiming(device, d_model, ff_dim, 46);
        let up = GpuLinear::kaiming(device, d_model, ff_dim, 47);
        let down = GpuLinear::kaiming(device, ff_dim, d_model, 48);
        let rope = GpuInterleavedRoPE::new(device, head_dim, kv_len.max(16), 10000.0);
        let mut cpu = vec![x];
        for l in [&wq, &wk, &wv, &wo, &gate, &up, &down] {
            cpu.push(l.weight.to_vec_flushed(device, cache));
            cpu.push(l.bias.to_vec_flushed(device, cache));
        }
        let half = head_dim / 2;
        let start = kv_len - seq;
        cpu.push(
            rope.cos_table.to_vec_flushed(device, cache)[start * half..kv_len * half].to_vec(),
        );
        cpu.push(
            rope.sin_table.to_vec_flushed(device, cache)[start * half..kv_len * half].to_vec(),
        );
        let (k_cache, v_cache) = if kv_len != seq {
            let mut rng = crate::Rng::new(77);
            let kc: Vec<f32> = (0..kv_len * kv).map(|_| rng.f32() * 2.0 - 1.0).collect();
            let vc: Vec<f32> = (0..kv_len * kv).map(|_| rng.f32() * 2.0 - 1.0).collect();
            cpu.push(kc.clone());
            cpu.push(vc.clone());
            (Some(kc), Some(vc))
        } else {
            (None, None)
        };
        Params {
            k_cache: k_cache.map(|kc| GpuTensor::from_slice(device, &kc, &[kv_len, kv])),
            v_cache: v_cache.map(|vc| GpuTensor::from_slice(device, &vc, &[kv_len, kv])),
            ln1: GpuRMSNorm::new(device, d_model, 1e-6),
            ln2: GpuRMSNorm::new(device, d_model, 1e-6),
            wq,
            wk,
            wv,
            wo,
            gate,
            up,
            down,
            rope,
            dims,
            cpu,
        }
    }

    pub struct HandNodes {
        pub normed: GpuTensor,
        pub q: GpuTensor,
        pub qr: GpuTensor,
        pub attn: GpuTensor,
        pub proj: GpuTensor,
        pub out: GpuTensor,
    }

    /// The hand-written prefill path from `gpu_inference`, without the KV cache.
    pub fn hand_forward(
        device: &GpuDevice,
        cache: &mut KernelCache,
        p: &Params,
        input: &GpuTensor,
    ) -> GpuTensor {
        hand_forward_nodes(device, cache, p, input).out
    }

    pub fn hand_forward_nodes(
        device: &GpuDevice,
        cache: &mut KernelCache,
        p: &Params,
        input: &GpuTensor,
    ) -> HandNodes {
        let d = &p.dims;
        let linear = |cache: &mut KernelCache, l: &GpuLinear, x: &GpuTensor| {
            let wt = l.weight.transpose_gpu(device, cache);
            let out = matmul::matmul(device, cache, x, &wt);
            bias_add(device, cache, &out, &l.bias)
        };
        let start = d.kv_len - d.seq;
        let normed = p.ln1.forward(device, cache, input);
        let q = linear(cache, &p.wq, &normed);
        let q3 = q
            .clone_gpu_batched(device, cache)
            .reshape(&[d.seq, d.n_heads, d.head_dim]);
        let qr = p.rope.forward(device, cache, &q3, start);
        let q_flat = qr.reshape(&[d.seq, d.n_heads * d.head_dim]);
        let (k_flat, v) = match (&p.k_cache, &p.v_cache) {
            (Some(kc), Some(vc)) => (
                kc.clone_gpu_batched(device, cache),
                vc.clone_gpu_batched(device, cache),
            ),
            _ => {
                let k = linear(cache, &p.wk, &normed);
                let v = linear(cache, &p.wv, &normed);
                let k3 = k.reshape(&[d.seq, d.n_kv_heads, d.head_dim]);
                let kr = p.rope.forward(device, cache, &k3, 0);
                (kr.reshape(&[d.seq, d.n_kv_heads * d.head_dim]), v)
            }
        };
        let attn = kv_attention_fused(
            device,
            cache,
            &q_flat,
            &k_flat,
            &v,
            d.seq,
            d.kv_len,
            d.n_heads,
            d.n_kv_heads,
            d.head_dim,
            start,
        );
        let proj = linear(cache, &p.wo, &attn);
        let res1 = add_tensors(device, cache, input, &proj);
        let normed2 = p.ln2.forward(device, cache, &res1);
        let gate = linear(cache, &p.gate, &normed2);
        let up = linear(cache, &p.up, &normed2);
        let act = swiglu_fused_pub(device, cache, &gate, &up);
        let ffn = linear(cache, &p.down, &act);
        let out = add_tensors(device, cache, &res1, &ffn);
        HandNodes {
            normed,
            q,
            qr: q_flat,
            attn,
            proj,
            out,
        }
    }

    pub fn ir_executable(
        device: &GpuDevice,
        p: &Params,
        scheduled: bool,
    ) -> (Executable, tang_compile::tensor::TensorId) {
        let (g, y) = if scheduled {
            compiled_block(&p.dims)
        } else {
            let (g, y) = transformer_block(&p.dims);
            let (g, outs) = g.substitute_maps(&[y]);
            (g, outs[0])
        };
        let inputs: Vec<&[f32]> = p.cpu.iter().map(|v| v.as_slice()).collect();
        (
            Executable::new(device, program_for(device, &g, &[y]), &inputs),
            y,
        )
    }

    /// tang-gpu's hand-written trainable block (`GpuTrainTransformerBlock`:
    /// no rope, no biases) loaded with the same weights as `p`.
    pub fn hand_train_block(
        device: &GpuDevice,
        cache: &mut KernelCache,
        p: &Params,
    ) -> GpuTrainTransformerBlock {
        let d = &p.dims;
        assert!(
            !d.rope && !d.bias,
            "the hand trainable block has no rope/biases"
        );
        let mut attn =
            GpuCausalAttention::zeros(device, d.d_model, d.n_heads, d.n_kv_heads, d.head_dim);
        attn.wq = p.wq.weight.clone_gpu_batched(device, cache);
        attn.wk = p.wk.weight.clone_gpu_batched(device, cache);
        attn.wv = p.wv.weight.clone_gpu_batched(device, cache);
        attn.wo = p.wo.weight.clone_gpu_batched(device, cache);
        let mut ffn = GpuSwiGLU::zeros(device, d.d_model, d.ff_dim);
        ffn.gate_proj = p.gate.weight.clone_gpu_batched(device, cache);
        ffn.up_proj = p.up.weight.clone_gpu_batched(device, cache);
        ffn.down_proj = p.down.weight.clone_gpu_batched(device, cache);
        GpuTrainTransformerBlock::new(
            GpuRMSNorm::new(device, d.d_model, 1e-6),
            attn,
            GpuRMSNorm::new(device, d.d_model, 1e-6),
            ffn,
        )
    }

    /// Forward + backward of the derived block as one program: outputs
    /// are `[out, dx, dwq, dwk, dwv, dwo, dgate, dup, ddown]` for
    /// `loss = sum(out)`.
    pub fn ir_train_executable(
        device: &GpuDevice,
        p: &Params,
    ) -> (Executable, Vec<tang_compile::tensor::TensorId>) {
        use tang_compile::tensor::TensorId;
        let (mut g, out) = transformer_block(&p.dims);
        let wrt: Vec<TensorId> = [0usize, 1, 3, 5, 7, 9, 11, 13]
            .iter()
            .map(|&i| TensorId::from_index(i))
            .collect();
        let grads = g.backward(out, &wrt).unwrap();
        let mut outs = vec![out];
        outs.extend(grads);
        let (g, outs) = g.substitute_maps(&outs);
        let (g, outs) = g.eliminate_common(&outs);
        let (g, outs) = g.merge_siblings(&outs);
        let (g, outs) = g.relayout(&outs);
        let (mut g, outs) = g.merge_reduce_dims(&outs);
        g.schedule_auto(&outs);
        let inputs: Vec<&[f32]> = p.cpu.iter().map(|v| v.as_slice()).collect();
        (
            Executable::new(device, program_for(device, &g, &outs), &inputs),
            outs,
        )
    }

    /// The block under the per-kernel autotuner, cached in
    /// `target/tang-tuning-block-<seq>.txt` for this device.
    pub fn ir_executable_tuned(
        device: &GpuDevice,
        cache: &mut KernelCache,
        p: &Params,
    ) -> (Executable, tang_compile::tensor::TensorId) {
        let (g, y) = compiled_block(&p.dims);
        let inputs: Vec<&[f32]> = p.cpu.iter().map(|v| v.as_slice()).collect();
        let path = format!(
            "{}/target/tang-tuning-block-{}-{}.txt",
            env!("CARGO_MANIFEST_DIR"),
            p.dims.seq,
            p.dims.kv_len
        );
        let tc = super::ir_gpu::TuningCache::new(path);
        let (g, from_cache) = tc.tune(device, cache, g, &[y], &inputs);
        eprintln!(
            "block seq {}: schedule {}",
            p.dims.seq,
            if from_cache {
                "from tuning cache"
            } else {
                "measured and cached"
            }
        );
        (Executable::new(device, g.to_wgsl(&[y]), &inputs), y)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Each hand-written stage against a plain CPU implementation.
        #[test]
        fn hand_block_pieces_vs_cpu() {
            let Ok(device) = GpuDevice::new_sync() else {
                eprintln!("no GPU, skipping");
                return;
            };
            let mut cache = KernelCache::new();
            let (seq, heads, kv_heads, hd) = (8usize, 6usize, 2usize, 32usize);
            let mut rng = crate::Rng::new(3);
            let q: Vec<f32> = (0..seq * heads * hd)
                .map(|_| rng.f32() * 2.0 - 1.0)
                .collect();
            let k: Vec<f32> = (0..seq * kv_heads * hd)
                .map(|_| rng.f32() * 2.0 - 1.0)
                .collect();
            let v: Vec<f32> = (0..seq * kv_heads * hd)
                .map(|_| rng.f32() * 2.0 - 1.0)
                .collect();
            let rope = GpuInterleavedRoPE::new(&device, hd, 64, 10000.0);
            let cos = rope.cos_table.to_vec_flushed(&device, &mut cache);
            let sin = rope.sin_table.to_vec_flushed(&device, &mut cache);
            let half = hd / 2;

            // rope
            let q3 = GpuTensor::from_slice(&device, &q, &[seq, heads, hd]);
            let qr = rope
                .forward(&device, &mut cache, &q3, 0)
                .to_vec_flushed(&device, &mut cache);
            let mut qr_cpu = vec![0.0f32; q.len()];
            for s in 0..seq {
                for h in 0..heads {
                    for i in 0..half {
                        let b = (s * heads + h) * hd;
                        let (c, sn) = (cos[s * half + i], sin[s * half + i]);
                        qr_cpu[b + 2 * i] = q[b + 2 * i] * c - q[b + 2 * i + 1] * sn;
                        qr_cpu[b + 2 * i + 1] = q[b + 2 * i] * sn + q[b + 2 * i + 1] * c;
                    }
                }
            }
            let err = qr
                .iter()
                .zip(&qr_cpu)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            eprintln!("hand rope max err {err}");
            assert!(err < 1e-4, "hand rope");

            // gqa causal attention on unrotated q/k
            let qt = GpuTensor::from_slice(&device, &q, &[seq, heads * hd]);
            let kt = GpuTensor::from_slice(&device, &k, &[seq, kv_heads * hd]);
            let vt = GpuTensor::from_slice(&device, &v, &[seq, kv_heads * hd]);
            let attn = kv_attention_fused(
                &device, &mut cache, &qt, &kt, &vt, seq, seq, heads, kv_heads, hd, 0,
            )
            .to_vec_flushed(&device, &mut cache);
            let per_group = heads / kv_heads;
            let scale = 1.0 / (hd as f32).sqrt();
            let mut attn_cpu = vec![0.0f32; seq * heads * hd];
            for i in 0..seq {
                for h in 0..heads {
                    let g = h / per_group;
                    let sc: Vec<f32> = (0..=i)
                        .map(|j| {
                            (0..hd)
                                .map(|d| {
                                    q[i * heads * hd + h * hd + d]
                                        * k[j * kv_heads * hd + g * hd + d]
                                })
                                .sum::<f32>()
                                * scale
                        })
                        .collect();
                    let m = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let w: Vec<f32> = sc.iter().map(|s| (s - m).exp()).collect();
                    let l: f32 = w.iter().sum();
                    for d in 0..hd {
                        attn_cpu[i * heads * hd + h * hd + d] = (0..=i)
                            .map(|j| w[j] * v[j * kv_heads * hd + g * hd + d])
                            .sum::<f32>()
                            / l;
                    }
                }
            }
            let (err, at) = attn
                .iter()
                .zip(&attn_cpu)
                .enumerate()
                .map(|(i, (a, b))| ((a - b).abs(), i))
                .fold((0.0, 0), |m, x| if x.0 > m.0 { x } else { m });
            eprintln!(
                "hand gqa attention max err {err} at {at} (row {}, head {})",
                at / (heads * hd),
                (at % (heads * hd)) / hd
            );
            assert!(err < 1e-4, "hand gqa attention");
        }

        /// Stage-by-stage: hand GPU intermediates vs the f64 interpreter.
        #[test]
        fn hand_block_stages_vs_f64() {
            let Ok(device) = GpuDevice::new_sync() else {
                eprintln!("no GPU, skipping");
                return;
            };
            let mut cache = KernelCache::new();
            let dims = BlockDims {
                seq: 12,
                kv_len: 12,
                rope: true,
                bias: true,
                d_model: 96,
                n_heads: 6,
                n_kv_heads: 2,
                head_dim: 16,
                ff_dim: 64,
            };
            let mut rng = crate::Rng::new(11);
            let x: Vec<f32> = (0..dims.seq * dims.d_model)
                .map(|_| rng.f32() * 2.0 - 1.0)
                .collect();
            let input = GpuTensor::from_slice(&device, &x, &[dims.seq, dims.d_model]);
            let p = params(&device, &mut cache, dims, x);
            let h = hand_forward_nodes(&device, &mut cache, &p, &input);
            let (g, n) = super::super::ir_block::transformer_block_nodes(&p.dims);
            let f64_inputs: Vec<Vec<f64>> = p
                .cpu
                .iter()
                .map(|v| v.iter().map(|x| *x as f64).collect())
                .collect();
            let cpu = g.eval(&f64_inputs, &[n.normed, n.q, n.qe, n.attn, n.proj, n.out]);
            let d = &p.dims;
            let err = |a: &[f32], b: &[f64]| {
                a.iter()
                    .zip(b)
                    .map(|(a, b)| (a - *b as f32).abs())
                    .fold(0.0f32, f32::max)
            };
            eprintln!(
                "normed err {}",
                err(&h.normed.to_vec_flushed(&device, &mut cache), &cpu[0])
            );
            eprintln!(
                "q err {}",
                err(&h.q.to_vec_flushed(&device, &mut cache), &cpu[1])
            );
            let qr = h.qr.to_vec_flushed(&device, &mut cache);
            let scale = 1.0 / (d.head_dim as f32).sqrt();
            let half = d.head_dim / 2;
            let mut qe_hand = vec![0.0f32; d.seq * d.n_heads * half];
            for s in 0..d.seq {
                for hh in 0..d.n_heads {
                    for i in 0..half {
                        qe_hand[(s * d.n_heads + hh) * half + i] =
                            qr[s * d.n_heads * d.head_dim + hh * d.head_dim + 2 * i] * scale;
                    }
                }
            }
            eprintln!("qe err {}", err(&qe_hand, &cpu[2]));
            // rope formula on the CPU, applied to the hand's own q with the hand's table
            let q_hand = h.q.to_vec_flushed(&device, &mut cache);
            let (cos, sin) = (&p.cpu[15], &p.cpu[16]);
            let mut qr_cpu = vec![0.0f32; q_hand.len()];
            for s in 0..d.seq {
                for hh in 0..d.n_heads {
                    for i in 0..half {
                        let b = s * d.n_heads * d.head_dim + hh * d.head_dim;
                        let (c, sn) = (cos[s * half + i], sin[s * half + i]);
                        qr_cpu[b + 2 * i] = q_hand[b + 2 * i] * c - q_hand[b + 2 * i + 1] * sn;
                        qr_cpu[b + 2 * i + 1] = q_hand[b + 2 * i] * sn + q_hand[b + 2 * i + 1] * c;
                    }
                }
            }
            let (e, at) = qr
                .iter()
                .zip(&qr_cpu)
                .enumerate()
                .map(|(i, (a, b))| ((a - b).abs(), i))
                .fold((0.0f32, 0), |m, x| if x.0 > m.0 { x } else { m });
            eprintln!(
                "hand qr vs cpu rope(hand q): err {e} at {at} (row {}): gpu {} cpu {}",
                at / (d.n_heads * d.head_dim),
                qr[at],
                qr_cpu[at]
            );
            let attn = h.attn.to_vec_flushed(&device, &mut cache);
            let per = d.n_heads / d.n_kv_heads;
            let mut attn_ir = vec![0.0f64; attn.len()];
            for g_ in 0..d.n_kv_heads {
                for r in 0..per {
                    for s in 0..d.seq {
                        for c in 0..d.head_dim {
                            let src = ((g_ * per + r) * d.seq + s) * d.head_dim + c;
                            attn_ir[s * d.n_heads * d.head_dim + (g_ * per + r) * d.head_dim + c] =
                                cpu[3][src];
                        }
                    }
                }
            }
            let (e, at) = attn
                .iter()
                .zip(&attn_ir)
                .enumerate()
                .map(|(i, (a, b))| ((a - *b as f32).abs(), i))
                .fold((0.0f32, 0), |m, x| if x.0 > m.0 { x } else { m });
            eprintln!(
                "attn err {e} at {at} (row {}, head {})",
                at / (d.n_heads * d.head_dim),
                (at % (d.n_heads * d.head_dim)) / d.head_dim
            );
            eprintln!(
                "proj err {}",
                err(&h.proj.to_vec_flushed(&device, &mut cache), &cpu[4])
            );
            eprintln!(
                "out err {}",
                err(&h.out.to_vec_flushed(&device, &mut cache), &cpu[5])
            );
        }

        /// The block's backward graph, lowered and run on the GPU, matches
        /// the f64 interpreter. No hand-written backward kernel anywhere.
        #[test]
        fn derived_block_backward_on_gpu() {
            let Ok(device) = GpuDevice::new_sync() else {
                eprintln!("no GPU, skipping");
                return;
            };
            let mut cache = KernelCache::new();
            let dims = BlockDims {
                seq: 12,
                kv_len: 12,
                rope: true,
                bias: true,
                d_model: 64,
                n_heads: 4,
                n_kv_heads: 2,
                head_dim: 16,
                ff_dim: 48,
            };
            let mut rng = crate::Rng::new(31);
            let x: Vec<f32> = (0..dims.seq * dims.d_model)
                .map(|_| rng.f32() * 2.0 - 1.0)
                .collect();
            let p = params(&device, &mut cache, dims, x);
            let (mut g, out) = transformer_block(&p.dims);
            use tang_compile::tensor::TensorId;
            let (xi, wq) = (TensorId::from_index(0), TensorId::from_index(1));
            let grads = g.backward(out, &[xi, wq]).unwrap();

            let f64_inputs: Vec<Vec<f64>> = p
                .cpu
                .iter()
                .map(|v| v.iter().map(|x| *x as f64).collect())
                .collect();
            let want = g.eval(&f64_inputs, &grads);

            let (g, outs) = g.substitute_maps(&grads);
            let (g, outs) = g.eliminate_common(&outs);
            let (g, outs) = g.merge_siblings(&outs);
            let (g, outs) = g.relayout(&outs);
            let (mut g, outs) = g.merge_reduce_dims(&outs);
            g.schedule_auto(&outs);
            let program = g.to_wgsl(&outs);
            let n_kernels = program.kernels.len();
            let inputs: Vec<&[f32]> = p.cpu.iter().map(|v| v.as_slice()).collect();
            let exe = Executable::new(&device, program, &inputs);
            let start = std::time::Instant::now();
            exe.run(&device, &mut cache);
            cache.flush(&device);
            let got: Vec<Vec<f32>> = outs.iter().map(|o| exe.read(&device, *o)).collect();
            eprintln!(
                "block backward: {n_kernels} kernels, {:?} incl. readback",
                start.elapsed()
            );
            for (k, (gw, gg)) in want.iter().zip(&got).enumerate() {
                let scale = gw.iter().fold(0.0f64, |m, v| m.max(v.abs())) as f32;
                let err = gg
                    .iter()
                    .zip(gw)
                    .map(|(a, b)| (a - *b as f32).abs())
                    .fold(0.0f32, f32::max);
                eprintln!("grad {k}: max err {err} (scale {scale})");
                assert!(err < 2e-3 * scale.max(1.0), "grad {k} diverges");
            }
        }

        /// Hand-written trainable block (forward_train + backward) vs the
        /// derived forward + backward on identical weights.
        #[test]
        fn derived_train_step_matches_hand_train_block() {
            let Ok(device) = GpuDevice::new_sync() else {
                eprintln!("no GPU, skipping");
                return;
            };
            let mut cache = KernelCache::new();
            let dims = BlockDims {
                seq: 10,
                kv_len: 10,
                rope: false,
                bias: false,
                d_model: 48,
                n_heads: 4,
                n_kv_heads: 2,
                head_dim: 12,
                ff_dim: 32,
            };
            let mut rng = crate::Rng::new(41);
            let x: Vec<f32> = (0..dims.seq * dims.d_model)
                .map(|_| rng.f32() * 2.0 - 1.0)
                .collect();
            let input = GpuTensor::from_slice(&device, &x, &[dims.seq, dims.d_model]);
            let p = params(&device, &mut cache, dims, x);

            let mut hand = hand_train_block(&device, &mut cache, &p);
            let out_hand = hand
                .forward_train(&device, &mut cache, &input)
                .to_vec_flushed(&device, &mut cache);
            let ones = GpuTensor::from_slice(
                &device,
                &vec![1.0f32; p.dims.seq * p.dims.d_model],
                &[p.dims.seq, p.dims.d_model],
            );
            let dx_hand = hand
                .backward(&device, &mut cache, &ones)
                .to_vec_flushed(&device, &mut cache);
            // gradients(): ln1 weight, wq, wk, wv, wo, ln2 weight, gate, up, down
            let hand_grads: Vec<Vec<f32>> = hand
                .gradients()
                .iter()
                .map(|g| {
                    g.map(|t| t.to_vec_flushed(&device, &mut cache))
                        .unwrap_or_default()
                })
                .collect();

            let (exe, outs) = ir_train_executable(&device, &p);
            exe.run(&device, &mut cache);
            cache.flush(&device);
            let got: Vec<Vec<f32>> = outs.iter().map(|o| exe.read(&device, *o)).collect();

            let check = |name: &str, a: &[f32], b: &[f32]| {
                assert_eq!(a.len(), b.len(), "{name} length");
                let scale = b.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let err = a
                    .iter()
                    .zip(b)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0f32, f32::max);
                eprintln!("{name}: max err {err} (scale {scale})");
                assert!(err < 2e-3 * scale.max(1.0), "{name} diverges");
            };
            check("out", &got[0], &out_hand);
            check("dx", &got[1], &dx_hand);
            check("dwq", &got[2], &hand_grads[1]);
            check("dwk", &got[3], &hand_grads[2]);
            check("dwv", &got[4], &hand_grads[3]);
            check("dwo", &got[5], &hand_grads[4]);
            check("dgate", &got[6], &hand_grads[6]);
            check("dup", &got[7], &hand_grads[7]);
            check("ddown", &got[8], &hand_grads[8]);
        }

        #[test]
        fn derived_decode_block_matches_hand() {
            let Ok(device) = GpuDevice::new_sync() else {
                eprintln!("no GPU, skipping");
                return;
            };
            let mut cache = KernelCache::new();
            let dims = BlockDims {
                seq: 2,
                kv_len: 40,
                rope: true,
                bias: true,
                d_model: 192,
                n_heads: 6,
                n_kv_heads: 2,
                head_dim: 32,
                ff_dim: 256,
            };
            let mut rng = crate::Rng::new(21);
            let x: Vec<f32> = (0..dims.seq * dims.d_model)
                .map(|_| rng.f32() * 2.0 - 1.0)
                .collect();
            let input = GpuTensor::from_slice(&device, &x, &[dims.seq, dims.d_model]);
            let p = params(&device, &mut cache, dims, x);
            let want =
                hand_forward(&device, &mut cache, &p, &input).to_vec_flushed(&device, &mut cache);
            let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            for scheduled in [false, true] {
                let (exe, y) = ir_executable(&device, &p, scheduled);
                exe.run(&device, &mut cache);
                cache.flush(&device);
                let got = exe.read(&device, y);
                let err = got
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                eprintln!("decode ir(scheduled={scheduled}) vs hand: max err {err}");
                assert!(err < 1e-3 * scale.max(1.0));
            }
        }

        #[test]
        fn derived_block_matches_hand_block_on_gpu() {
            let Ok(device) = GpuDevice::new_sync() else {
                eprintln!("no GPU, skipping");
                return;
            };
            let mut cache = KernelCache::new();
            let dims = BlockDims {
                seq: 24,
                kv_len: 24,
                rope: true,
                bias: true,
                d_model: 192,
                n_heads: 6,
                n_kv_heads: 2,
                head_dim: 32,
                ff_dim: 256,
            };
            let mut rng = crate::Rng::new(9);
            let x: Vec<f32> = (0..dims.seq * dims.d_model)
                .map(|_| rng.f32() * 2.0 - 1.0)
                .collect();
            let input = GpuTensor::from_slice(&device, &x, &[dims.seq, dims.d_model]);
            let p = params(&device, &mut cache, dims, x);
            let want =
                hand_forward(&device, &mut cache, &p, &input).to_vec_flushed(&device, &mut cache);
            // f64 reference from the interpreter on the unscheduled graph
            let (g, y) = transformer_block(&p.dims);
            let f64_inputs: Vec<Vec<f64>> = p
                .cpu
                .iter()
                .map(|v| v.iter().map(|x| *x as f64).collect())
                .collect();
            let cpu = g.eval(&f64_inputs, &[y]).remove(0);
            let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let worst = |got: &[f32]| {
                got.iter()
                    .zip(&cpu)
                    .enumerate()
                    .map(|(i, (a, b))| ((a - *b as f32).abs(), i))
                    .fold((0.0f32, 0usize), |m, x| if x.0 > m.0 { x } else { m })
            };
            let (hand_err, hand_i) = worst(&want);
            eprintln!("hand vs f64: max err {hand_err} at {hand_i} (scale {scale})");
            {
                let (exe, y) = ir_executable_tuned(&device, &mut cache, &p);
                exe.run(&device, &mut cache);
                cache.flush(&device);
                let got = exe.read(&device, y);
                let (ir_err, _) = worst(&got);
                eprintln!("ir(tuned) vs f64: max err {ir_err}");
                assert!(
                    ir_err < 2e-3 * scale.max(1.0),
                    "tuned ir diverges from f64 reference"
                );
            }
            for scheduled in [false, true] {
                let (exe, y) = ir_executable(&device, &p, scheduled);
                exe.run(&device, &mut cache);
                cache.flush(&device);
                let got = exe.read(&device, y);
                assert_eq!(got.len(), want.len());
                let (ir_err, ir_i) = worst(&got);
                eprintln!("ir(scheduled={scheduled}) vs f64: max err {ir_err} at {ir_i}");
                assert!(
                    ir_err < 2e-3 * scale.max(1.0),
                    "ir diverges from f64 reference"
                );
            }
            assert!(
                hand_err < 2e-3 * scale.max(1.0),
                "hand diverges from f64 reference"
            );
        }
    }
}
