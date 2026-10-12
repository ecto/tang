//! One runner for tensor-IR programs on every backend.
//!
//! A [`TensorGraph`] lowers to a [`WgslProgram`] (kernels + buffer specs)
//! for wgpu (`to_wgsl`, or MSL passthrough) or for CUDA (`to_cuda`: wmma
//! tensor-core tiles plus transpiled WGSL). [`Runtime`] picks the backend
//! once — CUDA when the `cuda` feature is on and an NVIDIA device with
//! nvrtc is present, wgpu otherwise — and [`Program`] owns the buffers of
//! one lowered program on that backend.
//!
//! # Interop
//!
//! A program lives entirely on its runtime's backend: every buffer it
//! reads or writes is allocated there. Programs on the same runtime hand
//! tensors to each other with device-side copies ([`Program::copy`]);
//! crossing backends (or reaching the CPU) is an explicit host round trip
//! ([`Program::read`] / [`Program::write`]). On wgpu, [`Program::wgpu_buffer`]
//! exposes the buffers to tang-gpu's hand kernels exactly as before; on
//! CUDA the hand kernels (`llm.rs`, `nn.rs`, …) do not run, so a code path
//! that wants CUDA must be expressed as IR programs end to end — the
//! derived training step and Adam are (`tang-bench/examples/train_loop.rs`).
//! There is no zero-copy Vulkan/CUDA sharing.
//!
//! # Memory plans
//!
//! By default every materialized tensor gets its own allocation. A
//! [`BufferPlan`] maps the program's buffers onto fewer slots (tensors
//! with disjoint lifetimes sharing one allocation); [`Program::with_plan`]
//! allocates the slots instead. A whole-model memory planner only has to
//! produce that mapping.

use std::cell::{RefCell, RefMut};

use tang_compile::tensor::precision::{f16_bits, f16_from_bits};
use tang_compile::tensor::wgsl::WgslProgram;
use tang_compile::tensor::{Format, TensorGraph, TensorId};

use crate::{GpuBuffer, GpuDevice, GpuError, KernelCache};

#[cfg(feature = "cuda")]
use crate::cuda::CudaDevice;

/// Which backend a [`Runtime`] runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    Wgpu,
    Cuda,
}

/// A device plus what running programs on it needs.
pub enum Runtime {
    Wgpu {
        device: GpuDevice,
        cache: RefCell<KernelCache>,
    },
    #[cfg(feature = "cuda")]
    Cuda(CudaDevice),
}

impl Runtime {
    /// CUDA when compiled with `cuda` and an NVIDIA device with nvrtc is
    /// present, else wgpu. `TANG_BACKEND=wgpu` or `=cuda` forces one.
    pub fn select() -> Result<Self, GpuError> {
        let want = std::env::var("TANG_BACKEND").ok();
        if want.as_deref() != Some("wgpu") {
            #[cfg(feature = "cuda")]
            if let Some(rt) = Self::cuda() {
                return Ok(rt);
            }
            if want.as_deref() == Some("cuda") {
                return Err(GpuError::NoAdapter);
            }
        }
        Self::wgpu()
    }

    pub fn wgpu() -> Result<Self, GpuError> {
        Ok(Self::from_wgpu(GpuDevice::new_sync()?))
    }

    /// Wrap an existing wgpu device (its own kernel cache).
    pub fn from_wgpu(device: GpuDevice) -> Self {
        Self::Wgpu {
            device,
            cache: RefCell::new(KernelCache::new()),
        }
    }

    /// CUDA device 0, if a driver, a device and nvrtc are all present.
    #[cfg(feature = "cuda")]
    pub fn cuda() -> Option<Self> {
        if !CudaDevice::has_nvrtc() {
            return None;
        }
        CudaDevice::new(0).map(Self::Cuda)
    }

    pub fn kind(&self) -> BackendKind {
        match self {
            Self::Wgpu { .. } => BackendKind::Wgpu,
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => BackendKind::Cuda,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Self::Wgpu { device, .. } => &device.name,
            #[cfg(feature = "cuda")]
            Self::Cuda(d) => &d.name,
        }
    }

    /// The wgpu device and kernel cache, for hand kernels.
    pub fn wgpu_parts(&self) -> Option<(&GpuDevice, RefMut<'_, KernelCache>)> {
        match self {
            Self::Wgpu { device, cache } => Some((device, cache.borrow_mut())),
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => None,
        }
    }

    #[cfg(feature = "cuda")]
    pub fn cuda_device(&self) -> Option<&CudaDevice> {
        match self {
            Self::Cuda(d) => Some(d),
            _ => None,
        }
    }

    /// `g`'s program for this backend. wgpu: WGSL, or native MSL tiles
    /// with `TANG_NATIVE=1` on Metal. CUDA: `to_cuda` (tensor cores on).
    pub fn lower(&self, g: &TensorGraph, outputs: &[TensorId]) -> WgslProgram {
        match self {
            Self::Wgpu { device, .. } => {
                if device.msl_passthrough && std::env::var("TANG_NATIVE").is_ok() {
                    g.to_msl_mma(outputs)
                } else {
                    g.to_wgsl(outputs)
                }
            }
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => g.to_cuda(outputs),
        }
    }

    /// Submit pending work and wait for it.
    pub fn sync(&self) {
        match self {
            Self::Wgpu { device, cache } => {
                cache.borrow_mut().flush(device);
                device.device.poll(wgpu::PollType::Wait).ok();
            }
            #[cfg(feature = "cuda")]
            Self::Cuda(d) => d.sync(),
        }
    }
}

/// Program buffers → allocations. `slot[i]` is the slot backing
/// `program.buffers[i]`; each slot is allocated once with `slot_len`
/// f32 words. Tensors sharing a slot must have disjoint live ranges and
/// graph inputs should keep slots of their own (the planner's job).
#[derive(Clone, Debug)]
pub struct BufferPlan {
    pub slot: Vec<usize>,
    pub slot_len: Vec<usize>,
}

impl BufferPlan {
    /// One slot per buffer — what [`Program::new`] uses.
    pub fn identity(program: &WgslProgram) -> Self {
        Self {
            slot: (0..program.buffers.len()).collect(),
            slot_len: program.buffers.iter().map(words).collect(),
        }
    }
}

/// f32 words a buffer needs (f16 packs two per word on wgpu; CUDA
/// programs keep f32 buffers).
fn words(b: &tang_compile::tensor::wgsl::BufferSpec) -> usize {
    match b.format {
        Format::F16 => b.len.div_ceil(2),
        _ => b.len,
    }
    .max(1)
}

enum Storage {
    Wgpu(Vec<GpuBuffer>),
    #[cfg(feature = "cuda")]
    Cuda {
        slots: Vec<cudarc::driver::CudaSlice<f32>>,
        functions: Vec<cudarc::driver::CudaFunction>,
    },
}

/// A lowered program with its buffers on one backend; reusable across runs.
pub struct Program {
    pub program: WgslProgram,
    storage: Storage,
    /// `slot[i]` backs `program.buffers[i]`.
    slot: Vec<usize>,
    /// Per kernel: slots of its reads then its output.
    args: Vec<Vec<usize>>,
}

/// What [`Program::tuned`] did.
#[derive(Clone, Copy, Debug, Default)]
pub struct TuneReport {
    /// Kernels with a tensor-core (native) version that kept it.
    pub native_kept: usize,
    /// Kernels with a native version at all.
    pub native_total: usize,
    /// The choices came from the tuning cache file.
    pub from_cache: bool,
}

impl Program {
    /// Lower `g` for `rt` and allocate it; `inputs` in graph input order.
    pub fn compile(rt: &Runtime, g: &TensorGraph, outputs: &[TensorId], inputs: &[&[f32]]) -> Self {
        Self::new(rt, rt.lower(g, outputs), inputs)
    }

    /// Allocate every buffer (one per tensor) and upload the inputs.
    pub fn new(rt: &Runtime, program: WgslProgram, inputs: &[&[f32]]) -> Self {
        let plan = BufferPlan::identity(&program);
        Self::with_plan(rt, program, inputs, &plan)
    }

    /// Allocate the slots of `plan` and upload the inputs.
    pub fn with_plan(
        rt: &Runtime,
        program: WgslProgram,
        inputs: &[&[f32]],
        plan: &BufferPlan,
    ) -> Self {
        assert_eq!(plan.slot.len(), program.buffers.len(), "plan size");
        for (i, b) in program.buffers.iter().enumerate() {
            assert!(words(b) <= plan.slot_len[plan.slot[i]], "slot too small");
            if let Some(k) = b.input {
                assert_eq!(inputs[k].len(), b.len, "input {k} has wrong length");
            }
        }
        let storage = match rt {
            Runtime::Wgpu { device, .. } => {
                let mut slots: Vec<GpuBuffer> = plan
                    .slot_len
                    .iter()
                    .map(|&n| GpuBuffer::uninit(device, n))
                    .collect();
                for (i, b) in program.buffers.iter().enumerate() {
                    let Some(k) = b.input else { continue };
                    let s = plan.slot[i];
                    // a whole-slot input is uploaded as its own buffer, as
                    // the old executor did
                    let fresh = match b.format {
                        Format::F16 => GpuBuffer::from_u32_slice(device, &pack_f16(inputs[k])),
                        Format::F32 => GpuBuffer::from_slice(device, inputs[k]),
                        Format::BF16 => unreachable!("bf16 is interpreter-only"),
                    };
                    if fresh.len() == plan.slot_len[s] {
                        slots[s] = fresh;
                    } else {
                        write_wgpu(device, &slots[s], b.format, inputs[k]);
                    }
                }
                Storage::Wgpu(slots)
            }
            #[cfg(feature = "cuda")]
            Runtime::Cuda(d) => {
                let mut slots: Vec<_> = plan
                    .slot_len
                    .iter()
                    .map(|&n| d.stream.alloc_zeros::<f32>(n).expect("alloc"))
                    .collect();
                for (i, b) in program.buffers.iter().enumerate() {
                    if let Some(k) = b.input {
                        let mut view = slots[plan.slot[i]].slice_mut(0..b.len);
                        d.stream.memcpy_htod(inputs[k], &mut view).expect("upload");
                    }
                }
                let functions = program
                    .kernels
                    .iter()
                    .map(|k| d.function(&k.source))
                    .collect();
                Storage::Cuda { slots, functions }
            }
        };
        let mut p = Self {
            program,
            storage,
            slot: plan.slot.clone(),
            args: Vec::new(),
        };
        p.args = p
            .program
            .kernels
            .iter()
            .map(|k| {
                k.reads
                    .iter()
                    .chain(std::iter::once(&k.node))
                    .map(|&t| p.slot_of(t))
                    .collect()
            })
            .collect();
        p
    }

    fn buffer_index(&self, t: TensorId) -> usize {
        let t = self.program.storage(t);
        self.program
            .buffers
            .iter()
            .position(|b| b.node == t)
            .expect("tensor has no buffer")
    }

    fn slot_of(&self, t: TensorId) -> usize {
        self.slot[self.buffer_index(t)]
    }

    /// The tensor bound to graph input `k`.
    pub fn input(&self, k: usize) -> TensorId {
        self.program
            .buffers
            .iter()
            .find(|b| b.input == Some(k))
            .expect("input not bound")
            .node
    }

    /// Format of the buffer holding `t`.
    pub fn format(&self, t: TensorId) -> Format {
        self.program.buffers[self.buffer_index(t)].format
    }

    /// The wgpu buffer holding `t`, for tang-gpu's hand kernels.
    pub fn wgpu_buffer(&self, t: TensorId) -> Option<&GpuBuffer> {
        match &self.storage {
            Storage::Wgpu(s) => Some(&s[self.slot_of(t)]),
            #[cfg(feature = "cuda")]
            _ => None,
        }
    }

    /// Run every kernel in order (one submission on wgpu, asynchronous
    /// launches on the CUDA stream).
    pub fn run(&self, rt: &Runtime) {
        match rt {
            Runtime::Wgpu { device, cache } => {
                let mut cache = cache.borrow_mut();
                cache.begin_batch();
                for i in 0..self.program.kernels.len() {
                    self.run_wgpu(device, &mut cache, i);
                }
                cache.end_batch(device);
            }
            #[cfg(feature = "cuda")]
            Runtime::Cuda(d) => {
                for i in 0..self.program.kernels.len() {
                    self.run_cuda(d, i);
                }
            }
        }
    }

    /// Run kernel `i` only (its inputs must already be computed).
    pub fn run_kernel(&self, rt: &Runtime, i: usize) {
        match rt {
            Runtime::Wgpu { device, cache } => self.run_wgpu(device, &mut cache.borrow_mut(), i),
            #[cfg(feature = "cuda")]
            Runtime::Cuda(d) => self.run_cuda(d, i),
        }
    }

    fn run_wgpu(&self, device: &GpuDevice, cache: &mut KernelCache, i: usize) {
        #[allow(irrefutable_let_patterns)]
        let Storage::Wgpu(slots) = &self.storage
        else {
            panic!("program and runtime are on different backends")
        };
        let k = &self.program.kernels[i];
        let bufs: Vec<&GpuBuffer> = self.args[i].iter().map(|&s| &slots[s]).collect();
        cache.dispatch_buffers(
            device,
            &k.source,
            &bufs,
            &[bufs.len() - 1],
            &[k.points, 0, 0, 0],
            k.count,
            k.workgroup_size,
        );
    }

    #[cfg(feature = "cuda")]
    fn run_cuda(&self, d: &CudaDevice, i: usize) {
        use cudarc::driver::{LaunchConfig, PushKernelArg};
        let Storage::Cuda { slots, functions } = &self.storage else {
            panic!("program and runtime are on different backends")
        };
        let k = &self.program.kernels[i];
        let cfg = LaunchConfig {
            grid_dim: (k.count.div_ceil(k.workgroup_size), 1, 1),
            block_dim: (k.workgroup_size, 1, 1),
            shared_mem_bytes: crate::cuda::dynamic_smem(&k.source),
        };
        let mut launch = d.stream.launch_builder(&functions[i]);
        for &s in &self.args[i] {
            launch.arg(&slots[s]);
        }
        launch.arg(&k.points);
        unsafe { launch.launch(cfg) }.expect("launch");
    }

    /// Read `t` back as f32 (waits for queued work; f16 widened exactly).
    pub fn read(&self, rt: &Runtime, t: TensorId) -> Vec<f32> {
        let b = &self.program.buffers[self.buffer_index(t)];
        match (rt, &self.storage) {
            (Runtime::Wgpu { device, cache }, Storage::Wgpu(s)) => {
                cache.borrow_mut().flush(device);
                let raw = s[self.slot_of(t)].to_vec_sync(device);
                match b.format {
                    Format::F16 => unpack_f16(&raw, b.len),
                    _ => raw[..b.len].to_vec(),
                }
            }
            #[cfg(feature = "cuda")]
            (Runtime::Cuda(d), Storage::Cuda { slots, .. }) => {
                let mut out = vec![0.0f32; b.len];
                d.stream
                    .memcpy_dtoh(&slots[self.slot_of(t)].slice(0..b.len), &mut out)
                    .expect("download");
                d.sync();
                out
            }
            #[allow(unreachable_patterns)]
            _ => panic!("program and runtime are on different backends"),
        }
    }

    /// Overwrite `t` (usually an input) from the host.
    pub fn write(&self, rt: &Runtime, t: TensorId, data: &[f32]) {
        let b = &self.program.buffers[self.buffer_index(t)];
        assert_eq!(data.len(), b.len, "write length");
        let s = self.slot_of(t);
        match (rt, &self.storage) {
            (Runtime::Wgpu { device, cache }, Storage::Wgpu(slots)) => {
                cache.borrow_mut().flush(device);
                write_wgpu(device, &slots[s], b.format, data);
            }
            #[cfg(feature = "cuda")]
            (Runtime::Cuda(d), Storage::Cuda { slots, .. }) => {
                use cudarc::driver::DevicePtr;
                let (dst, _sync) = slots[s].device_ptr(&d.stream);
                unsafe {
                    cudarc::driver::result::memcpy_htod_async(dst, data, d.stream.cu_stream())
                }
                .expect("upload");
                // `data` is borrowed only for this call
                d.sync();
            }
            #[allow(unreachable_patterns)]
            _ => panic!("program and runtime are on different backends"),
        }
    }

    /// Device-side copy of `src`'s tensor `src_t` into this program's
    /// tensor `dst` (same runtime; queued in order with the kernels).
    pub fn copy(&self, rt: &Runtime, dst: TensorId, src: &Program, src_t: TensorId) {
        let (db, sb) = (
            &self.program.buffers[self.buffer_index(dst)],
            &src.program.buffers[src.buffer_index(src_t)],
        );
        assert_eq!((db.len, db.format), (sb.len, sb.format), "copy shape");
        let (ds, ss) = (self.slot_of(dst), src.slot_of(src_t));
        match (rt, &self.storage, &src.storage) {
            (Runtime::Wgpu { device, cache }, Storage::Wgpu(d), Storage::Wgpu(s)) => {
                let mut cache = cache.borrow_mut();
                if d[ds].len() == s[ss].len() {
                    d[ds].copy_from(device, &mut cache, &s[ss]);
                } else {
                    copy_wgpu(device, &mut cache, &d[ds], &s[ss], words(db));
                }
            }
            #[cfg(feature = "cuda")]
            (
                Runtime::Cuda(dev),
                Storage::Cuda { slots: d, .. },
                Storage::Cuda { slots: s, .. },
            ) => {
                use cudarc::driver::DevicePtr;
                let (dp, _a) = d[ds].device_ptr(&dev.stream);
                let (sp, _b) = s[ss].device_ptr(&dev.stream);
                unsafe {
                    cudarc::driver::result::memcpy_dtod_async(
                        dp,
                        sp,
                        db.len * 4,
                        dev.stream.cu_stream(),
                    )
                }
                .expect("dtod");
            }
            #[allow(unreachable_patterns)]
            _ => panic!("programs and runtime are on different backends"),
        }
    }

    /// Milliseconds per run of the whole program: the minimum over five
    /// batches of `iters` (a shared GPU only ever adds time).
    pub fn time(&self, rt: &Runtime, iters: usize) -> f64 {
        time_with(rt, iters, || self.run(rt))
    }

    /// Milliseconds per launch of kernel `i` alone.
    pub fn time_kernel(&self, rt: &Runtime, i: usize, iters: usize) -> f64 {
        time_with(rt, iters, || {
            self.run_kernel(rt, i);
            #[allow(irrefutable_let_patterns)]
            if let Runtime::Wgpu { device, cache } = rt {
                cache.borrow_mut().flush(device);
            }
        })
    }

    /// Lower `g` and tune it for `rt`. On CUDA, every kernel with a
    /// tensor-core lowering is timed against its transpiled-WGSL twin and
    /// the faster one kept; the choices are stored in `cache` (keyed by the
    /// device and both programs' sources) so the measurement is paid once
    /// per chip. On wgpu this is [`Program::compile`] (the WGSL schedule
    /// autotuner, `tang_bench::ir_gpu::TuningCache`, runs on the graph
    /// before lowering).
    pub fn tuned(
        rt: &Runtime,
        g: &TensorGraph,
        outputs: &[TensorId],
        inputs: &[&[f32]],
        cache: Option<&std::path::Path>,
    ) -> (Self, TuneReport) {
        match rt {
            Runtime::Wgpu { .. } => {
                let _ = cache;
                (Self::compile(rt, g, outputs, inputs), TuneReport::default())
            }
            #[cfg(feature = "cuda")]
            Runtime::Cuda(_) => Self::tuned_cuda(rt, g, outputs, inputs, cache),
        }
    }

    #[cfg(feature = "cuda")]
    fn tuned_cuda(
        rt: &Runtime,
        g: &TensorGraph,
        outputs: &[TensorId],
        inputs: &[&[f32]],
        cache: Option<&std::path::Path>,
    ) -> (Self, TuneReport) {
        let native_p = g.to_cuda_with(outputs, true);
        let plain_p = g.to_cuda_with(outputs, false);
        assert_eq!(native_p.kernels.len(), plain_p.kernels.len());
        let differs: Vec<usize> = (0..native_p.kernels.len())
            .filter(|&i| native_p.kernels[i].source != plain_p.kernels[i].source)
            .collect();
        let key = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            rt.name().hash(&mut h);
            for k in native_p.kernels.iter().chain(&plain_p.kernels) {
                k.source.hash(&mut h);
            }
            h.finish()
        };
        let header = format!("tang-cuda-tuning {key:016x}");
        let cached: Option<Vec<usize>> = cache.and_then(|p| {
            let text = std::fs::read_to_string(p).ok()?;
            let mut lines = text.lines();
            (lines.next()? == header).then(|| lines.filter_map(|l| l.trim().parse().ok()).collect())
        });
        let from_cache = cached.is_some();
        let (mut native_p, mut plain_p) = (native_p, plain_p);
        let plain_kernels: Vec<usize> = match cached {
            Some(v) => v,
            None => {
                let native = Self::new(rt, native_p, inputs);
                let plain = Self::new(rt, plain_p, inputs);
                // every kernel's inputs exist once a whole run has gone through
                native.run(rt);
                plain.run(rt);
                let v: Vec<usize> = differs
                    .iter()
                    .copied()
                    .filter(|&i| plain.time_kernel(rt, i, 10) < native.time_kernel(rt, i, 10))
                    .collect();
                if let Some(p) = cache {
                    let mut text = header.clone();
                    for i in &v {
                        text.push_str(&format!("\n{i}"));
                    }
                    text.push('\n');
                    if let Some(dir) = p.parent() {
                        let _ = std::fs::create_dir_all(dir);
                    }
                    if let Err(e) = std::fs::write(p, text) {
                        eprintln!("tuning cache: could not write {}: {e}", p.display());
                    }
                }
                native_p = native.program;
                plain_p = plain.program;
                v
            }
        };
        let mut program = native_p;
        for &i in &plain_kernels {
            std::mem::swap(&mut program.kernels[i], &mut plain_p.kernels[i]);
        }
        let report = TuneReport {
            native_kept: differs.len() - plain_kernels.len(),
            native_total: differs.len(),
            from_cache,
        };
        (Self::new(rt, program, inputs), report)
    }
}

fn time_with(rt: &Runtime, iters: usize, f: impl Fn()) -> f64 {
    for _ in 0..3 {
        f();
    }
    rt.sync();
    (0..5)
        .map(|_| {
            let t = std::time::Instant::now();
            for _ in 0..iters {
                f();
            }
            rt.sync();
            t.elapsed().as_secs_f64() * 1e3 / iters as f64
        })
        .fold(f64::INFINITY, f64::min)
}

/// Write `data` at the start of `buf` (which may be a larger slot).
fn write_wgpu(device: &GpuDevice, buf: &GpuBuffer, format: Format, data: &[f32]) {
    match format {
        Format::F16 => {
            let words = pack_f16(data);
            device
                .queue
                .write_buffer(&buf.buffer, 0, bytemuck::cast_slice(&words));
        }
        _ => device
            .queue
            .write_buffer(&buf.buffer, 0, bytemuck::cast_slice(data)),
    }
}

/// Copy the first `words` of `src` into `dst` (buffers of different sizes).
fn copy_wgpu(
    device: &GpuDevice,
    cache: &mut KernelCache,
    dst: &GpuBuffer,
    src: &GpuBuffer,
    words: usize,
) {
    let mut enc = device
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_buffer_to_buffer(&src.buffer, 0, &dst.buffer, 0, (words * 4) as u64);
    cache.enqueue(enc.finish());
}

/// f32 values → IEEE f16 (round to nearest even), two per u32 word,
/// little end first — the byte layout of a WGSL `array<f16>`.
pub fn pack_f16(v: &[f32]) -> Vec<u32> {
    v.chunks(2)
        .map(|p| {
            let lo = f16_bits(p[0] as f64) as u32;
            let hi = p.get(1).map_or(0, |&x| f16_bits(x as f64) as u32);
            lo | (hi << 16)
        })
        .collect()
}

/// Inverse of [`pack_f16`]: the first `n` halves of `words` (read back as
/// f32 bit patterns), widened to f32.
pub fn unpack_f16(words: &[f32], n: usize) -> Vec<f32> {
    words
        .iter()
        .flat_map(|w| {
            let b = w.to_bits();
            [b as u16, (b >> 16) as u16]
        })
        .take(n)
        .map(|h| f16_from_bits(h) as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matmul_graph() -> (TensorGraph, TensorId) {
        let mut g = TensorGraph::new();
        let a = g.input(&[8, 4]);
        let b = g.input(&[4, 6]);
        let c = g.matmul(a, b).unwrap();
        let (g, outs) = g.substitute_maps(&[c]);
        (g, outs[0])
    }

    fn inputs() -> (Vec<f32>, Vec<f32>) {
        (
            (0..32).map(|i| i as f32 * 0.1 - 1.0).collect(),
            (0..24).map(|i| (i % 5) as f32 * 0.2).collect(),
        )
    }

    #[test]
    fn program_matches_eval() {
        let Ok(rt) = Runtime::select() else { return };
        let (g, c) = matmul_graph();
        let (a, b) = inputs();
        let p = Program::compile(&rt, &g, &[c], &[&a, &b]);
        p.run(&rt);
        let got = p.read(&rt, c);
        let want = g.eval(
            &[
                a.iter().map(|&x| x as f64).collect(),
                b.iter().map(|&x| x as f64).collect(),
            ],
            &[c],
        );
        for (x, y) in got.iter().zip(&want[0]) {
            assert!((*x as f64 - y).abs() < 1e-3, "{x} vs {y}");
        }
    }

    #[test]
    fn write_copy_and_plan() {
        let Ok(rt) = Runtime::select() else { return };
        let (g, c) = matmul_graph();
        let (a, b) = inputs();
        let program = rt.lower(&g, &[c]);
        // all buffers in one oversized slot each: exercises slot != buffer sizes
        let mut plan = BufferPlan::identity(&program);
        for n in &mut plan.slot_len {
            *n += 3;
        }
        let p = Program::with_plan(&rt, program, &[&a, &b], &plan);
        p.run(&rt);
        let first = p.read(&rt, c);
        let q = Program::compile(&rt, &g, &[c], &[&a, &b]);
        q.write(&rt, q.input(0), &vec![0.0; a.len()]);
        q.run(&rt);
        assert!(q.read(&rt, c).iter().all(|&x| x == 0.0));
        q.copy(&rt, q.input(0), &p, p.input(0));
        q.run(&rt);
        assert_eq!(q.read(&rt, c), first);
    }
}
