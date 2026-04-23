//! GPU-accelerated brute-force operator search.
//!
//! Given a shape table (from `shape_bytecode`), dispatch a WGSL
//! compute kernel that evaluates every (shape, assignment) tree over
//! the alphabet `{6 atoms, 10 unary ops, 5 binary ops}` at five complex
//! test-point pairs, matches against the 31 standard targets, watches
//! for `(1+ε)^(1/ε)` limit-regime pow calls, and emits only the
//! candidates that (a) use both x and y, (b) produce finite values, and
//! (c) hit at least `target_threshold` targets at the primary test
//! point.
//!
//! Only available under `--features gpu`.

use crate::shape_bytecode::{ShapeInfoRecord, ShapeTable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Raw WGSL kernel source. See comments below for the computation.
pub const KERNEL_WGSL: &str = include_str!("gpu_search.wgsl");

/// One hit record emitted by the GPU kernel.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuHit {
    pub shape_idx: u32,
    pub assignment_lo: u32,
    pub assignment_hi: u32,
    pub target_bits: u32,
    pub artifact: u32,
    pub value_re: f32,
    pub value_im: f32,
    pub size: u32,
}

/// Uniform parameters for each dispatch.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DispatchParams {
    shape_idx: u32,
    base_assignment_lo: u32,
    base_assignment_hi: u32,
    n_threads: u32,
    target_threshold: u32,
    limit_threshold_exp: i32, // negative exponent, e.g. -5 for 1e-5
    max_hits: u32,
    _pad: u32,
}

/// The 5 transcendental-independent test-point pairs (as 20 f32s).
/// Must match `op_enum::DedupSet::TEST_PAIRS`.
const TEST_POINTS_F32: [f32; 20] = [
    // γ, A
    0.5772156649015329,
    0.0,
    1.2824271291006226,
    0.0,
    // A, G
    1.2824271291006226,
    0.0,
    0.9159655941772190,
    0.0,
    // G, γ
    0.9159655941772190,
    0.0,
    0.5772156649015329,
    0.0,
    // complex pair 1
    1.5,
    0.3,
    0.7,
    -0.4,
    // complex pair 2
    0.4,
    1.1,
    -0.6,
    0.8,
];

/// 31 standard targets, primary test point is γ (first of test_points).
/// Each target is a (re, im) pair — 62 f32s total.
fn targets_f32() -> Vec<f32> {
    use std::f64::consts::{E, PI};
    let g = 0.5772156649015329_f64;
    let gc = num_complex::Complex::new(g, 0.0);
    let raw: [(f64, f64); 31] = [
        // Constants
        (0.0, 0.0),
        (1.0, 0.0),
        (-1.0, 0.0),
        (2.0, 0.0),
        (-2.0, 0.0),
        (0.5, 0.0),
        (E, 0.0),
        (-E, 0.0),
        (1.0 / E, 0.0),
        (E * E, 0.0),
        (PI, 0.0),
        (PI / 2.0, 0.0),
        (2.0 * PI, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (0.0, PI),
        // Functions of x
        (g, 0.0),
        (gc.exp().re, gc.exp().im),
        (gc.ln().re, gc.ln().im),
        (-g, 0.0),
        (1.0 / g, 0.0),
        (g * g, 0.0),
        (gc.sqrt().re, gc.sqrt().im),
        (g + 1.0, 0.0),
        (g - 1.0, 0.0),
        (2.0 * g, 0.0),
        (E * g, 0.0),
        (gc.exp().exp().re, gc.exp().exp().im),
        (gc.ln().ln().re, gc.ln().ln().im),
        (g.sin(), 0.0),
        (g.cos(), 0.0),
    ];
    let mut out = Vec::with_capacity(62);
    for (re, im) in raw {
        out.push(re as f32);
        out.push(im as f32);
    }
    out
}

pub const TARGET_NAMES: [&str; 31] = [
    "0",
    "1",
    "-1",
    "2",
    "-2",
    "1/2",
    "e",
    "-e",
    "1/e",
    "e^2",
    "pi",
    "pi/2",
    "2pi",
    "i",
    "-i",
    "i*pi",
    "x",
    "exp(x)",
    "ln(x)",
    "-x",
    "1/x",
    "x^2",
    "sqrt(x)",
    "x+1",
    "x-1",
    "2x",
    "e*x",
    "exp(exp(x))",
    "ln(ln(x))",
    "sin(x)",
    "cos(x)",
];

pub struct GpuSearcher {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,

    shape_bytecode_buf: wgpu::Buffer,
    shape_info_buf: wgpu::Buffer,
    test_points_buf: wgpu::Buffer,
    targets_buf: wgpu::Buffer,
    hit_buf: wgpu::Buffer,
    hit_count_buf: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    hit_count_readback: wgpu::Buffer,
    hit_readback: wgpu::Buffer,

    pub shape_table: Arc<ShapeTable>,
    pub max_hits: u32,
}

impl GpuSearcher {
    pub fn new(
        shape_table: Arc<ShapeTable>,
        max_hits: u32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or("no GPU adapter")?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("sheffer-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    // Max allowed by wgpu is 2^31 - 1 bytes per binding.
                    max_storage_buffer_binding_size: 2_147_483_647,
                    max_buffer_size: 2_147_483_647,
                    max_compute_invocations_per_workgroup: 256,
                    ..wgpu::Limits::default()
                },
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))?;

        // Compile shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sheffer-gpu kernel"),
            source: wgpu::ShaderSource::Wgsl(KERNEL_WGSL.into()),
        });

        // Bind group layout: 7 bindings (params uniform + 6 storage buffers)
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sheffer-gpu bgl"),
            entries: &[
                // 0: params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: shape_bytecodes (storage read)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2: shape_info (storage read)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3: test_points (storage read)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 4: targets (storage read)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 5: hit_buffer (storage read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 6: hit_count (storage atomic)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sheffer-gpu pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sheffer-gpu pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Upload shape bytecodes
        let shape_bytecode_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shape_bytecodes"),
            contents: bytemuck::cast_slice(&shape_table.all_instrs),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Pack shape_info into bytes
        let shape_info_bytes = bytemuck_shape_info(&shape_table.shape_info);
        let shape_info_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shape_info"),
            contents: &shape_info_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });

        let test_points_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test_points"),
            contents: bytemuck::cast_slice(&TEST_POINTS_F32),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let targets = targets_f32();
        let targets_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("targets"),
            contents: bytemuck::cast_slice(&targets),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let hit_buf_size = (max_hits as u64) * (std::mem::size_of::<GpuHit>() as u64);
        let hit_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hit_buffer"),
            size: hit_buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let hit_count_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hit_count"),
            contents: bytemuck::cast_slice(&[0u32]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<DispatchParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let hit_count_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hit_count_readback"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let hit_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hit_readback"),
            size: hit_buf_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            shape_bytecode_buf,
            shape_info_buf,
            test_points_buf,
            targets_buf,
            hit_buf,
            hit_count_buf,
            params_buf,
            hit_count_readback,
            hit_readback,
            shape_table,
            max_hits,
        })
    }

    /// Dispatch the kernel over one shape's full assignment space in
    /// chunks of up to `chunk_threads`. Hits are accumulated in
    /// `self.hit_buf`; caller reads them back after the full search.
    pub fn dispatch_shape(
        &mut self,
        shape_idx: u32,
        target_threshold: u32,
        chunk_threads: u32,
        limit_threshold_exp: i32,
    ) {
        let info = &self.shape_table.shape_info[shape_idx as usize];
        let assignment_count =
            (info.assignment_count_hi as u64) << 32 | (info.assignment_count_lo as u64);
        let mut offset: u64 = 0;
        while offset < assignment_count {
            let n = std::cmp::min(chunk_threads as u64, assignment_count - offset) as u32;
            let params = DispatchParams {
                shape_idx,
                base_assignment_lo: offset as u32,
                base_assignment_hi: (offset >> 32) as u32,
                n_threads: n,
                target_threshold,
                limit_threshold_exp,
                max_hits: self.max_hits,
                _pad: 0,
            };
            self.queue
                .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sheffer-gpu bg"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.shape_bytecode_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.shape_info_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.test_points_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.targets_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.hit_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: self.hit_count_buf.as_entire_binding(),
                    },
                ],
            });

            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("sheffer-gpu enc"),
                });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("sheffer-gpu pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                let wg = n.div_ceil(256);
                pass.dispatch_workgroups(wg, 1, 1);
            }
            self.queue.submit(std::iter::once(enc.finish()));
            offset += n as u64;
        }
    }

    /// Read back the final hit count and the corresponding slice of
    /// hit records from the GPU.
    pub fn read_hits(&mut self) -> Vec<GpuHit> {
        // First, copy hit_count into readback buffer and read it.
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sheffer-gpu readback-enc"),
            });
        enc.copy_buffer_to_buffer(&self.hit_count_buf, 0, &self.hit_count_readback, 0, 4);
        self.queue.submit(std::iter::once(enc.finish()));

        let slice = self.hit_count_readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let count = {
            let data = slice.get_mapped_range();
            let arr: [u32; 1] = *bytemuck::from_bytes::<[u32; 1]>(&data);
            arr[0]
        };
        self.hit_count_readback.unmap();

        let count_capped = std::cmp::min(count, self.max_hits);
        if count_capped == 0 {
            return Vec::new();
        }

        let byte_count = count_capped as u64 * std::mem::size_of::<GpuHit>() as u64;
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sheffer-gpu hit-readback-enc"),
            });
        enc.copy_buffer_to_buffer(&self.hit_buf, 0, &self.hit_readback, 0, byte_count);
        self.queue.submit(std::iter::once(enc.finish()));

        let slice = self.hit_readback.slice(..byte_count);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let hits: Vec<GpuHit> = {
            let data = slice.get_mapped_range();
            bytemuck::cast_slice::<u8, GpuHit>(&data).to_vec()
        };
        self.hit_readback.unmap();
        hits
    }

    pub fn reset_hit_count(&mut self) {
        self.queue
            .write_buffer(&self.hit_count_buf, 0, bytemuck::cast_slice(&[0u32]));
    }
}

fn bytemuck_shape_info(info: &[ShapeInfoRecord]) -> Vec<u8> {
    // ShapeInfoRecord is 8 u32s = 32 bytes. Re-cast directly.
    let mut out = Vec::with_capacity(info.len() * 32);
    for r in info {
        out.extend_from_slice(&r.bytecode_offset.to_le_bytes());
        out.extend_from_slice(&r.bytecode_len.to_le_bytes());
        out.extend_from_slice(&r.n_atoms.to_le_bytes());
        out.extend_from_slice(&r.n_unary.to_le_bytes());
        out.extend_from_slice(&r.n_binary.to_le_bytes());
        out.extend_from_slice(&r.assignment_count_lo.to_le_bytes());
        out.extend_from_slice(&r.assignment_count_hi.to_le_bytes());
        out.extend_from_slice(&r.size.to_le_bytes());
    }
    out
}
