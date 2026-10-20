//! GPU decoder for tc-gpu streams ([`crate::gpuwav`]) — wgpu + hand-written
//! WGSL compute kernels (`gpuwav.wgsl`). Output is bit-identical to
//! [`crate::gpuwav::decode_box_parsed`]: every kernel is integer only.
//!
//! Kernels, in order:
//! 1. `decode` — one workgroup per brick, one invocation per rANS lane:
//!    coefficient decode pass by pass (barrier between passes), then
//!    in-place fixed-point dequantization.
//! 2. `lift` — one dispatch per (level, axis), one invocation per line:
//!    inverse CDF 9/7 integer lifting.
//! 3. `outliers` — one invocation per brick: decodes the max-abs correction
//!    lane and adds `k·corr` in the integer domain.
//! 4. `finalize` — one invocation per cell: `(v + mean) as f32 · 2^e`.
//! 5. `escapes` — raw `f32` overrides.

use crate::gpuwav::{self, Parsed, ESC0, LOW0, MAG0, NKEY, NMODEL, NTAP, SGN0, SIG0};
use std::time::Instant;
use wgpu::util::DeviceExt;

const SHS: usize = 68;
const BS: usize = 20;
const BRS: usize = 8;
const NPARAM: usize = 32;
/// Largest lane count (one workgroup per brick, one invocation per lane).
pub const MAX_LANES: usize = 256;

/// A GPU device with the tc-gpu pipelines for one lane count / line length.
pub struct GpuDecoder {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    pipes: Vec<(String, wgpu::ComputePipeline)>,
    /// Adapter description.
    pub name: String,
    lanes: usize,
    maxl: usize,
}

/// Stream state uploaded to the GPU.
pub struct Prepared {
    meta: wgpu::Buffer,
    stream: wgpu::Buffer,
    models: wgpu::Buffer,
    coef: wgpu::Buffer,
    out: wgpu::Buffer,
    params: Vec<u32>,
    nbricks: usize,
    batch: usize,
    maxlev: usize,
    maxlines: usize,
    ncells: usize,
    corr: bool,
    nesc: usize,
}

/// Wall-clock split of a GPU decode.
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuTimes {
    /// Header parse + host tables.
    pub parse_s: f64,
    /// Buffer creation + upload.
    pub upload_s: f64,
    /// Kernel execution (submit → done).
    pub kernels_s: f64,
    /// Readback to host.
    pub readback_s: f64,
}

fn shader_src(lanes: usize, maxl: usize) -> String {
    let k = gpuwav::lift_consts();
    let head = format!(
        "const LANESu: u32 = {lanes}u;\nconst NMODELu: u32 = {NMODEL}u;\nconst SIG0u: u32 = {SIG0}u;\nconst MAG0u: u32 = {MAG0}u;\nconst LOW0u: u32 = {LOW0}u;\nconst SGN0u: u32 = {SGN0}u;\nconst ESC0u: u32 = {ESC0}u;\nconst KB_u: u32 = {}u;\nconst NE_u: u32 = {}u;\nconst NA_u: u32 = 12u;\nconst NPB_u: u32 = 5u;\nconst BSu: u32 = {BS}u;\nconst SHSu: u32 = {SHS}u;\nconst BRSu: u32 = {BRS}u;\n",
        crate::coefcode::KB,
        crate::coefcode::NE
    );
    let body = include_str!("gpuwav.wgsl")
        .replace("LANES_WG", &lanes.to_string())
        .replace("MAXL_N", &maxl.max(2).to_string())
        .replace("KK_LIST", &k.iter().map(|v| format!("{v}")).collect::<Vec<_>>().join(", "));
    head + &body
}

impl GpuDecoder {
    /// Open the default high-performance adapter and build the pipelines
    /// for `lanes` lanes per brick and lines up to `maxl` samples.
    pub fn new(lanes: usize, maxl: usize) -> Result<Self, String> {
        pollster::block_on(Self::new_async(lanes, maxl))
    }

    async fn new_async(lanes: usize, maxl: usize) -> Result<Self, String> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| format!("adapter: {e}"))?;
        let al = adapter.limits();
        let limits = wgpu::Limits {
            max_buffer_size: al.max_buffer_size,
            max_storage_buffer_binding_size: al.max_storage_buffer_binding_size,
            max_compute_workgroups_per_dimension: al.max_compute_workgroups_per_dimension,
            ..Default::default()
        };
        if lanes == 0 || lanes > MAX_LANES {
            return Err(format!("tc-gpu GPU decode supports 1..={MAX_LANES} lanes"));
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("tc-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| format!("device: {e}"))?;
        let info = adapter.get_info();
        let name = format!("{} ({:?})", info.name, info.backend);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tc-gpu"),
            source: wgpu::ShaderSource::Wgsl(shader_src(lanes, maxl).into()),
        });
        let st = |b: u32, ro: bool| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: ro },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tc-gpu"),
            entries: &[
                st(0, true),
                st(1, true),
                st(2, false),
                st(3, false),
                st(4, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tc-gpu"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipes = ["decode", "lift", "outliers", "finalize", "escapes", "decode3", "dequant3", "outliers3"]
            .iter()
            .map(|&e| {
                let p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(e),
                    layout: Some(&pl),
                    module: &module,
                    entry_point: Some(e),
                    compilation_options: Default::default(),
                    cache: None,
                });
                (e.to_string(), p)
            })
            .collect();
        Ok(Self { device, queue, layout, pipes, name, lanes, maxl })
    }

    fn pipe(&self, n: &str) -> &wgpu::ComputePipeline {
        &self.pipes.iter().find(|p| p.0 == n).unwrap().1
    }

    /// Lanes this decoder was built for.
    pub fn lanes(&self) -> usize {
        self.lanes
    }

    /// Build host tables and upload a parsed stream.
    pub fn prepare(&self, ps: &Parsed, bytes: &[u8], times: &mut GpuTimes) -> Prepared {
        let t0 = Instant::now();
        assert_eq!(ps.lanes, self.lanes, "decoder built for a different lane count");
        assert!(!ps.adaptive, "the GPU decoder takes static-model streams (GpuConfig::adaptive = false)");
        let mut meta: Vec<u32> = Vec::new();
        let off_shapes = meta.len();
        let mut band_off = 0usize;
        let mut maxlev = 0usize;
        let mut maxlines = 1usize;
        for sh in &ps.shapes {
            let mut r = vec![0u32; SHS];
            for a in 0..3 {
                r[a] = sh.dims[a] as u32;
                assert!(sh.dims[a] <= self.maxl.max(2) || sh.dims.iter().filter(|&&d| d > 1).count() == 0, "line too long for this decoder");
            }
            r[3] = sh.bands.len() as u32;
            r[4] = band_off as u32;
            r[5] = sh.p.nlev() as u32;
            maxlev = maxlev.max(sh.p.nlev());
            for (l, ln) in sh.p.lens.iter().enumerate().take(15) {
                for a in 0..3 {
                    r[6 + l * 3 + a] = ln[a] as u32;
                }
            }
            for (l, s) in sh.p.split.iter().enumerate().take(15) {
                r[6 + 45 + l] = (s[0] as u32) | (s[1] as u32) << 1 | (s[2] as u32) << 2;
            }
            maxlines = maxlines.max(sh.dims[0] * sh.dims[1]).max(sh.dims[0] * sh.dims[2]).max(sh.dims[1] * sh.dims[2]);
            band_off += sh.bands.len();
            meta.extend(r);
        }
        let off_bands = meta.len();
        for sh in &ps.shapes {
            for b in &sh.bands {
                let mut r = vec![0u32; BS];
                for a in 0..3 {
                    r[a] = b.start[a] as u32;
                    r[3 + a] = b.end[a] as u32;
                }
                r[6] = b.class as u32;
                r[7] = b.parent.map_or(u32::MAX, |p| p as u32);
                r[8] = b.pshift[0] | b.pshift[1] << 1 | b.pshift[2] << 2;
                r[9] = b.npass as u32;
                for p in 0..8 {
                    r[10 + p] = b.pkey[p] as u32;
                }
                r[18] = b.m as u32;
                r[19] = b.sh;
                meta.extend(r);
            }
        }
        let off_preds = meta.len();
        assert_eq!(ps.preds.len(), NKEY);
        for p in &ps.preds {
            meta.extend(p.iter().map(|&v| v as u32));
            debug_assert_eq!(p.len(), NTAP);
        }
        let off_init = meta.len();
        meta.extend_from_slice(&ps.init);
        let off_dl = meta.len();
        meta.extend(ps.dl.iter().map(|&v| v as u32));
        let off_bricks = meta.len();
        let nbricks = ps.bricks.len();
        meta.resize(off_bricks + nbricks * BRS, 0);
        let mut cbase = 0usize;
        let mut escs: Vec<u32> = Vec::new();
        for (bi, br) in ps.bricks.iter().enumerate() {
            let sh = &ps.shapes[br.shape];
            let ltab = meta.len();
            meta.extend(br.lane_word.iter().map(|&w| w as u32));
            let r = off_bricks + bi * BRS;
            for a in 0..3 {
                meta[r + a] = br.origin[a] as u32;
            }
            meta[r + 3] = br.shape as u32;
            meta[r + 4] = cbase as u32;
            meta[r + 5] = ltab as u32;
            let d = sh.dims;
            for &(li, bits) in &br.esc {
                let (z, y, x) = (li / (d[1] * d[2]), (li / d[2]) % d[1], li % d[2]);
                let g = ((br.origin[0] + z) * ps.d3[1] + br.origin[1] + y) * ps.d3[2] + br.origin[2] + x;
                escs.push(g as u32);
                escs.push(bits);
            }
            cbase += sh.dims.iter().product::<usize>();
        }
        let off_esc = meta.len();
        meta.extend_from_slice(&escs);
        let ncells: usize = ps.d3.iter().product();
        assert_eq!(cbase, ncells);
        let mut params = vec![0u32; NPARAM];
        params[0] = off_shapes as u32;
        params[1] = off_bands as u32;
        params[2] = off_preds as u32;
        params[3] = off_init as u32;
        params[4] = off_dl as u32;
        params[5] = off_bricks as u32;
        params[6] = off_esc as u32;
        params[7] = (escs.len() / 2) as u32;
        params[8] = nbricks as u32;
        params[13] = ps.mean_fix as u32;
        params[14] = ps.scale().to_bits();
        params[15] = ps.corr as u32;
        for a in 0..3 {
            params[16 + a] = ps.d3[a] as u32;
        }
        params[19] = ps.bdims[0] as u32;
        params[20] = ps.bdims[1] as u32;
        params[21] = ps.bdims[2] as u32;
        for a in 0..3 {
            params[22 + a] = ps.d3[a].div_ceil(ps.bdims[a]) as u32;
        }
        params[25] = ncells as u32;
        let mut sw: Vec<u32> = Vec::with_capacity(bytes.len() / 4 + 1);
        for c in bytes.chunks(4) {
            let mut b = [0u8; 4];
            b[..c.len()].copy_from_slice(c);
            sw.push(u32::from_le_bytes(b));
        }
        times.parse_s += t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let batch = nbricks.clamp(1, 65535);
        let dev = &self.device;
        let init = |label: &str, data: &[u32], usage: wgpu::BufferUsages| {
            dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: as_bytes(data), usage })
        };
        let meta_b = init("meta", &meta, wgpu::BufferUsages::STORAGE);
        let stream_b = init("stream", &sw, wgpu::BufferUsages::STORAGE);
        let mk = |label: &str, n: u64, usage: wgpu::BufferUsages| {
            dev.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: n.max(4), usage, mapped_at_creation: false })
        };
        // static context models need no per-lane state; the binding is kept
        // for an adaptive-model kernel
        let models = mk("models", 4, wgpu::BufferUsages::STORAGE);
        let coef = mk("coef", ncells as u64 * 4, wgpu::BufferUsages::STORAGE);
        let out = mk("out", ncells as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        self.queue.submit([]);
        let _ = self.device.poll(wgpu::PollType::Wait);
        times.upload_s += t1.elapsed().as_secs_f64();
        Prepared {
            meta: meta_b,
            stream: stream_b,
            models,
            coef,
            out,
            params,
            nbricks,
            batch,
            maxlev,
            maxlines,
            ncells,
            corr: ps.corr != 0,
            nesc: escs.len() / 2,
        }
    }

    fn bind(&self, pr: &Prepared, params: &[u32]) -> wgpu::BindGroup {
        let ub = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: as_bytes(params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: pr.meta.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pr.stream.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pr.models.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: pr.coef.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: pr.out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: ub.as_entire_binding() },
            ],
        })
    }

    /// Run every kernel; returns the kernel wall time in seconds.
    pub fn run(&self, pr: &Prepared) -> f64 {
        let mut jobs: Vec<(&str, Vec<u32>, [u32; 3])> = Vec::new();
        let mut b0 = 0;
        while b0 < pr.nbricks {
            let n = pr.batch.min(pr.nbricks - b0);
            let mut p = pr.params.clone();
            p[9] = b0 as u32;
            p[10] = n as u32;
            jobs.push(("decode", p, [n as u32, 1, 1]));
            b0 += n;
        }
        for l in (0..pr.maxlev).rev() {
            for ax in [2u32, 1, 0] {
                let mut p = pr.params.clone();
                p[11] = l as u32;
                p[12] = ax;
                jobs.push(("lift", p, [pr.maxlines.div_ceil(64) as u32, pr.nbricks as u32, 1]));
            }
        }
        if pr.corr {
            jobs.push(("outliers", pr.params.clone(), [pr.nbricks.div_ceil(64) as u32, 1, 1]));
        }
        let groups = pr.ncells.div_ceil(256);
        jobs.push(("finalize", pr.params.clone(), [groups.min(65535) as u32, groups.div_ceil(65535) as u32, 1]));
        if pr.nesc > 0 {
            jobs.push(("escapes", pr.params.clone(), [pr.nesc.div_ceil(64) as u32, 1, 1]));
        }
        self.submit(pr, &jobs)
    }

    fn submit(&self, pr: &Prepared, jobs: &[(&str, Vec<u32>, [u32; 3])]) -> f64 {
        let binds: Vec<wgpu::BindGroup> = jobs.iter().map(|j| self.bind(pr, &j.1)).collect();
        if std::env::var("TC_GPU_PROFILE").is_ok() {
            let mut acc: Vec<(&str, f64)> = Vec::new();
            for (j, bg) in jobs.iter().zip(&binds) {
                let t0 = Instant::now();
                let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                    pass.set_pipeline(self.pipe(j.0));
                    pass.set_bind_group(0, bg, &[]);
                    pass.dispatch_workgroups(j.2[0], j.2[1], j.2[2]);
                }
                self.queue.submit([enc.finish()]);
                let _ = self.device.poll(wgpu::PollType::Wait);
                let dt = t0.elapsed().as_secs_f64();
                match acc.iter_mut().find(|a| a.0 == j.0) {
                    Some(a) => a.1 += dt,
                    None => acc.push((j.0, dt)),
                }
            }
            eprintln!("tc-gpu kernel times: {acc:?}");
        }
        let t0 = Instant::now();
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for (j, bg) in jobs.iter().zip(&binds) {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(self.pipe(j.0));
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(j.2[0], j.2[1], j.2[2]);
        }
        self.queue.submit([enc.finish()]);
        let _ = self.device.poll(wgpu::PollType::Wait);
        t0.elapsed().as_secs_f64()
    }

    /// Copy the decoded `f32` bits back to the host.
    pub fn readback(&self, pr: &Prepared) -> Vec<u32> {
        let size = pr.ncells as u64 * 4;
        let st = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: size.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&pr.out, 0, &st, 0, size);
        self.queue.submit([enc.finish()]);
        let slice = st.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        let _ = self.device.poll(wgpu::PollType::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let mut v = vec![0u32; pr.ncells];
        for (o, c) in v.iter_mut().zip(data.as_chunks::<4>().0) {
            *o = u32::from_le_bytes(*c);
        }
        drop(data);
        st.unmap();
        v
    }

    /// Decode a whole stream; returns `f32` bits (field layout) and timings.
    pub fn decode(&self, bytes: &[u8]) -> Result<(Vec<u32>, GpuTimes), crate::bitio::DecodeError> {
        let mut t = GpuTimes::default();
        let t0 = Instant::now();
        let ps = gpuwav::parse(bytes)?;
        t.parse_s = t0.elapsed().as_secs_f64();
        let pr = self.prepare(&ps, bytes, &mut t);
        t.kernels_s = self.run(&pr);
        let t1 = Instant::now();
        let v = self.readback(&pr);
        t.readback_s = t1.elapsed().as_secs_f64();
        Ok((v, t))
    }
}

impl GpuDecoder {
    /// Decode a whole v3 (code-block) stream; returns `f32` bits and timings.
    pub fn decode3(&self, bytes: &[u8]) -> Result<(Vec<u32>, GpuTimes), crate::bitio::DecodeError> {
        let mut t = GpuTimes::default();
        let t0 = Instant::now();
        let ps = crate::gpuwav3::parse3(bytes)?;
        t.parse_s = t0.elapsed().as_secs_f64();
        let pr = self.prepare3(&ps, bytes, &mut t);
        t.kernels_s = self.run3(&pr);
        let t1 = Instant::now();
        let v = self.readback(&pr.base);
        t.readback_s = t1.elapsed().as_secs_f64();
        Ok((v, t))
    }

    /// Upload a parsed v3 stream.
    pub fn prepare3(&self, ps: &crate::gpuwav3::Parsed3, bytes: &[u8], times: &mut GpuTimes) -> Prepared3 {
        let t0 = Instant::now();
        let sh = &ps.sh;
        assert!(sh.dims.iter().all(|&d| d <= self.maxl.max(2)), "field axis longer than this decoder's line buffer (maxl)");
        let mut meta: Vec<u32> = Vec::new();
        // shape record (whole field)
        let mut r = vec![0u32; SHS];
        for a in 0..3 {
            r[a] = sh.dims[a] as u32;
        }
        r[3] = sh.bands.len() as u32;
        r[5] = sh.p.nlev() as u32;
        for (l, ln) in sh.p.lens.iter().enumerate().take(15) {
            for a in 0..3 {
                r[6 + l * 3 + a] = ln[a] as u32;
            }
        }
        for (l, s) in sh.p.split.iter().enumerate().take(15) {
            r[6 + 45 + l] = (s[0] as u32) | (s[1] as u32) << 1 | (s[2] as u32) << 2;
        }
        let off_shapes = meta.len();
        meta.extend(r);
        let off_bands = meta.len();
        for (bi, b) in sh.bands.iter().enumerate() {
            let pb = &sh.p.bands[bi];
            let mut r = vec![0u32; BS];
            for a in 0..3 {
                r[a] = b.start[a] as u32;
                r[3 + a] = b.end[a] as u32;
            }
            r[6] = b.class as u32;
            r[7] = b.parent.map_or(u32::MAX, |p| p as u32);
            r[8] = b.pshift[0] | b.pshift[1] << 1 | b.pshift[2] << 2;
            r[9] = 1;
            r[10] = crate::gpuwav3::key3(pb.level, pb.mask) as u32;
            r[18] = b.m as u32;
            r[19] = b.sh;
            meta.extend(r);
        }
        let off_preds = meta.len();
        for p in &ps.preds {
            meta.extend(p.iter().map(|&v| v as u32));
        }
        let off_init = meta.len();
        meta.extend_from_slice(&ps.init);
        let off_dl = meta.len();
        meta.extend(ps.dl.iter().map(|&v| v as u32));
        let off_bricks = meta.len();
        meta.extend([0, 0, 0, 0, 0, 0, 0, 0]);
        let off_blocks = meta.len();
        for (i, b) in ps.blocks.iter().enumerate() {
            meta.push(ps.block_band[i] as u32);
            meta.extend(b.start.iter().map(|&v| v as u32));
            meta.extend(b.end.iter().map(|&v| v as u32));
            meta.push(ps.block_words[i].0 as u32);
            meta.push(ps.block_words[i].1 as u32);
        }
        let off_tiles = meta.len();
        let mut escs: Vec<u32> = Vec::new();
        let d3 = ps.d3;
        for t in &ps.tiles {
            meta.extend(t.origin.iter().map(|&v| v as u32));
            meta.extend(t.dims.iter().map(|&v| v as u32));
            meta.push(t.words.0 as u32);
            meta.push(t.words.1 as u32);
            for &(li, bits) in &t.esc {
                let (z, y, x) = (li / (t.dims[1] * t.dims[2]), (li / t.dims[2]) % t.dims[1], li % t.dims[2]);
                escs.push((((t.origin[0] + z) * d3[1] + t.origin[1] + y) * d3[2] + t.origin[2] + x) as u32);
                escs.push(bits);
            }
        }
        let off_esc = meta.len();
        meta.extend_from_slice(&escs);
        let ncells: usize = d3.iter().product();
        let mut params = vec![0u32; NPARAM];
        params[0] = off_shapes as u32;
        params[1] = off_bands as u32;
        params[2] = off_preds as u32;
        params[3] = off_init as u32;
        params[4] = off_dl as u32;
        params[5] = off_bricks as u32;
        params[6] = off_esc as u32;
        params[7] = (escs.len() / 2) as u32;
        params[8] = 1;
        params[13] = ps.mean_fix as u32;
        params[14] = ps.scale().to_bits();
        params[15] = ps.corr as u32;
        for a in 0..3 {
            params[16 + a] = d3[a] as u32;
            params[19 + a] = d3[a] as u32;
            params[22 + a] = 1;
        }
        params[25] = ncells as u32;
        params[26] = off_blocks as u32;
        params[27] = ps.resctx as u32;
        params[28] = off_tiles as u32;
        params[29] = ps.tiles.len() as u32;
        let mut sw: Vec<u32> = Vec::with_capacity(bytes.len() / 4 + 1);
        for c in bytes.chunks(4) {
            let mut b = [0u8; 4];
            b[..c.len()].copy_from_slice(c);
            sw.push(u32::from_le_bytes(b));
        }
        times.parse_s += t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let dev = &self.device;
        let init = |label: &str, data: &[u32], usage: wgpu::BufferUsages| {
            dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: as_bytes(data), usage })
        };
        let mk = |label: &str, n: u64, usage: wgpu::BufferUsages| {
            dev.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: n.max(4), usage, mapped_at_creation: false })
        };
        let base = Prepared {
            meta: init("meta", &meta, wgpu::BufferUsages::STORAGE),
            stream: init("stream", &sw, wgpu::BufferUsages::STORAGE),
            models: mk("residuals", ncells as u64 * 4, wgpu::BufferUsages::STORAGE),
            coef: mk("coef", ncells as u64 * 4, wgpu::BufferUsages::STORAGE),
            out: mk("out", ncells as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC),
            params,
            nbricks: 1,
            batch: 1,
            maxlev: sh.p.nlev(),
            maxlines: (d3[0] * d3[1]).max(d3[0] * d3[2]).max(d3[1] * d3[2]),
            ncells,
            corr: ps.corr != 0,
            nesc: escs.len() / 2,
        };
        let _ = self.device.poll(wgpu::PollType::Wait);
        times.upload_s += t1.elapsed().as_secs_f64();
        Prepared3 { base, groups: ps.level_groups(), nblocks: ps.blocks.len(), ntiles: ps.tiles.len() }
    }

    /// Run the v3 kernels; returns the kernel wall time.
    pub fn run3(&self, pr: &Prepared3) -> f64 {
        let b = &pr.base;
        let mut jobs: Vec<(&str, Vec<u32>, [u32; 3])> = Vec::new();
        for &(g0, g1) in &pr.groups {
            let n = g1 - g0;
            let mut p = b.params.clone();
            p[9] = g0 as u32;
            p[10] = n as u32;
            jobs.push(("decode3", p, [n.div_ceil(64) as u32, 1, 1]));
        }
        let mut p = b.params.clone();
        p[10] = pr.nblocks as u32;
        jobs.push(("dequant3", p, [pr.nblocks.div_ceil(64) as u32, 1, 1]));
        for l in (0..b.maxlev).rev() {
            for ax in [2u32, 1, 0] {
                let mut p = b.params.clone();
                p[11] = l as u32;
                p[12] = ax;
                jobs.push(("lift", p, [b.maxlines.div_ceil(64) as u32, 1, 1]));
            }
        }
        if b.corr {
            jobs.push(("outliers3", b.params.clone(), [pr.ntiles.div_ceil(64) as u32, 1, 1]));
        }
        let groups = b.ncells.div_ceil(256);
        jobs.push(("finalize", b.params.clone(), [groups.min(65535) as u32, groups.div_ceil(65535) as u32, 1]));
        if b.nesc > 0 {
            jobs.push(("escapes", b.params.clone(), [b.nesc.div_ceil(64) as u32, 1, 1]));
        }
        self.submit(b, &jobs)
    }
}

/// A v3 stream uploaded to the GPU.
pub struct Prepared3 {
    base: Prepared,
    groups: Vec<(usize, usize)>,
    nblocks: usize,
    ntiles: usize,
}

fn as_bytes(v: &[u32]) -> &[u8] {
    // u32 has no padding; reinterpreting as bytes is sound
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::ErrorBound;
    use crate::gpuwav::{decode_box_parsed, encode_f32, parse, GpuConfig};

    fn field(d: [usize; 3]) -> Vec<f32> {
        let mut v = Vec::new();
        for i in 0..d[0] {
            for j in 0..d[1] {
                for k in 0..d[2] {
                    let (x, y, z) = (i as f64, j as f64, k as f64);
                    v.push(((0.3 * x).sin() + (0.11 * y * z / 8.0).cos() + 0.01 * x * y - ((k * 7 + j * 3) % 11) as f64 * 0.003 + 3.0) as f32);
                }
            }
        }
        v
    }

    #[test]
    fn gpu_matches_cpu() {
        let Ok(g) = GpuDecoder::new(8, 128) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let d = [20, 37, 45];
        let x = field(d);
        for bound in [ErrorBound::Psnr(50.0), ErrorBound::Psnr(100.0), ErrorBound::MaxAbs(1e-3), ErrorBound::MaxAbs(1e-5)] {
            for brick in [8, 16, 64] {
                let cfg = GpuConfig { brick, lanes: 8, ..GpuConfig::new(bound) };
                let (bytes, _) = encode_f32(&x, &d, &cfg);
                let ps = parse(&bytes).unwrap();
                let cpu = decode_box_parsed(&ps, &bytes, [0; 3], ps.d3, 1).unwrap();
                let (gpu, _) = g.decode(&bytes).unwrap();
                let bad = cpu.iter().zip(&gpu).filter(|(a, b)| a != b).count();
                assert_eq!(bad, 0, "{bound:?} brick {brick}: {bad} cells differ");
            }
        }
        let x2: Vec<f32> = (0..70 * 90).map(|i| ((i % 90) as f32 * 0.07).sin() * ((i / 90) as f32 * 0.05).cos()).collect();
        let (bytes, _) = encode_f32(&x2, &[70, 90], &GpuConfig { brick: 32, lanes: 8, ..GpuConfig::new(ErrorBound::Psnr(70.0)) });
        let ps = parse(&bytes).unwrap();
        let cpu = decode_box_parsed(&ps, &bytes, [0; 3], ps.d3, 1).unwrap();
        assert_eq!(g.decode(&bytes).unwrap().0, cpu);
        // v3 code-blocks
        use crate::gpuwav3::{decode3_box, encode3_f32, Gpu3Config};
        for bound in [ErrorBound::Psnr(50.0), ErrorBound::Psnr(100.0), ErrorBound::MaxAbs(1e-3), ErrorBound::MaxAbs(1e-5)] {
            for (block, resctx) in [(8, true), (16, false), (64, true)] {
                let cfg = Gpu3Config { block, resctx, tile: 16, ..Gpu3Config::new(bound) };
                let (bytes, _) = encode3_f32(&x, &d, &cfg, 2);
                let cpu = decode3_box(&bytes, [0; 3], [d[0], d[1], d[2]], 1).unwrap().bits;
                let (gpu, _) = g.decode3(&bytes).unwrap();
                let bad = cpu.iter().zip(&gpu).filter(|(a, b)| a != b).count();
                assert_eq!(bad, 0, "v3 {bound:?} block {block}: {bad} cells differ");
            }
        }
        let (bytes, _) = encode3_f32(&x2, &[70, 90], &Gpu3Config { block2d: 16, tile: 4, ..Gpu3Config::new(ErrorBound::MaxAbs(1e-4)) }, 1);
        let cpu = decode3_box(&bytes, [0; 3], [1, 70, 90], 1).unwrap().bits;
        assert_eq!(g.decode3(&bytes).unwrap().0, cpu);
    }
}
