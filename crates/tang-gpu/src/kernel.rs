//! Kernel cache: compile WGSL to pipeline, dispatch.

use std::collections::HashMap;

use crate::buffer::GpuBuffer;
use crate::device::GpuDevice;

/// A cached compute pipeline.
pub(crate) struct CachedPipeline {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

/// Command buffers a batch holds before it is submitted anyway.
const MAX_PENDING: usize = 32;

/// Cache of compiled WGSL compute pipelines, keyed by source hash.
/// Supports command batching: when `batching` is true, dispatches are
/// accumulated and submitted together on `flush()`.
pub struct KernelCache {
    pipelines: HashMap<u64, CachedPipeline>,
    /// Pending command buffers to be submitted together.
    pending: Vec<wgpu::CommandBuffer>,
    /// When true, dispatches are batched instead of submitted immediately.
    batching: bool,
}

impl KernelCache {
    /// Create an empty kernel cache.
    pub fn new() -> Self {
        Self {
            pipelines: HashMap::new(),
            pending: Vec::new(),
            batching: false,
        }
    }

    /// Enable command batching. Dispatches will accumulate until `flush()`.
    pub fn begin_batch(&mut self) {
        self.batching = true;
    }

    /// Submit what is pending and go back to submitting every dispatch
    /// as it is issued.
    pub fn end_batch(&mut self, device: &GpuDevice) {
        self.flush(device);
        self.batching = false;
    }

    /// Submit all pending command buffers to the GPU queue.
    /// Must be called before any buffer readback (to_vec_sync).
    pub fn flush(&mut self, device: &GpuDevice) {
        if !self.pending.is_empty() {
            device.queue.submit(self.pending.drain(..));
        }
    }

    /// Enqueue a command buffer for batched submission.
    pub fn enqueue(&mut self, cmd: wgpu::CommandBuffer) {
        self.pending.push(cmd);
    }

    /// Submit or enqueue a command buffer depending on batching mode.
    pub(crate) fn submit_or_enqueue(&mut self, device: &GpuDevice, cmd: wgpu::CommandBuffer) {
        if self.batching {
            self.pending.push(cmd);
            // Metal allows 64 live command buffers per queue and blocks
            // forever creating the 65th if none can complete: an unflushed
            // batch must not grow past that.
            if self.pending.len() >= MAX_PENDING {
                self.flush(device);
            }
        } else {
            device.queue.submit(std::iter::once(cmd));
        }
    }

    /// Get or compile a pipeline with the standard 3-binding layout
    /// (read storage, read-write storage, uniform). Public for custom dispatch patterns.
    pub(crate) fn get_or_compile_custom(
        &mut self,
        device: &GpuDevice,
        wgsl: &str,
        hash: u64,
    ) -> &CachedPipeline {
        self.pipelines
            .entry(hash)
            .or_insert_with(|| Self::compile_standard_3(device, wgsl))
    }

    fn compile_standard_3(device: &GpuDevice, wgsl: &str) -> CachedPipeline {
        let module = device
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("tang-gpu kernel"),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("tang-gpu bgl"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
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

        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("tang-gpu pipeline layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("tang-gpu pipeline"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        CachedPipeline {
            pipeline,
            bind_group_layout,
        }
    }

    /// Get or compile a pipeline with 5-binding layout:
    /// 0=read, 1=read, 2=read, 3=read_write, 4=uniform.
    /// Used for layer norm (input, weight, bias, output, params).
    pub(crate) fn get_or_compile_5bind(
        &mut self,
        device: &GpuDevice,
        wgsl: &str,
        hash: u64,
    ) -> &CachedPipeline {
        self.pipelines.entry(hash).or_insert_with(|| {
            let module = device
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("tang-gpu kernel 5bind"),
                    source: wgpu::ShaderSource::Wgsl(wgsl.into()),
                });

            fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
                wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }
            }

            let bind_group_layout =
                device
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("tang-gpu bgl 5bind"),
                        entries: &[
                            storage_entry(0, true),
                            storage_entry(1, true),
                            storage_entry(2, true),
                            storage_entry(3, false),
                            wgpu::BindGroupLayoutEntry {
                                binding: 4,
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

            let pipeline_layout =
                device
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("tang-gpu pipeline layout 5bind"),
                        bind_group_layouts: &[&bind_group_layout],
                        push_constant_ranges: &[],
                    });

            let pipeline =
                device
                    .device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("tang-gpu pipeline 5bind"),
                        layout: Some(&pipeline_layout),
                        module: &module,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        cache: None,
                    });

            CachedPipeline {
                pipeline,
                bind_group_layout,
            }
        })
    }

    /// Get or compile a pipeline for the given WGSL source.
    fn get_or_compile(&mut self, device: &GpuDevice, wgsl: &str) -> &CachedPipeline {
        let hash = Self::hash_wgsl(wgsl);
        self.pipelines
            .entry(hash)
            .or_insert_with(|| Self::compile_standard_3(device, wgsl))
    }

    /// Dispatch a compute kernel.
    ///
    /// `count` is the number of work items (threads).
    /// The kernel reads from `inputs` and writes to `output`.
    pub fn dispatch(
        &mut self,
        device: &GpuDevice,
        wgsl: &str,
        inputs: &GpuBuffer,
        output: &GpuBuffer,
        count: u32,
    ) {
        let workgroup_size = 256u32;
        let cached = self.get_or_compile(device, wgsl);

        // Create params uniform buffer (count + 3 padding u32s)
        let params_data: [u32; 4] = [count, 0, 0, 0];
        use wgpu::util::DeviceExt;
        let params_buf = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tang-gpu params"),
                contents: bytemuck::cast_slice(&params_data),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tang-gpu bind group"),
            layout: &cached.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: inputs.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tang-gpu dispatch"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tang-gpu compute"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&cached.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let n_workgroups = count.div_ceil(workgroup_size);
            pass.dispatch_workgroups(n_workgroups, 1, 1);
        }

        self.submit_or_enqueue(device, encoder.finish());
    }

    /// Get or compile a pipeline with 4-binding layout:
    /// 0=read, 1=read, 2=read_write, 3=uniform. Public for custom dispatch.
    pub(crate) fn get_or_compile_rr_w(
        &mut self,
        device: &GpuDevice,
        wgsl: &str,
        hash: u64,
    ) -> &CachedPipeline {
        self.pipelines.entry(hash).or_insert_with(|| {
            let module = device
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("tang-gpu kernel rr_w"),
                    source: wgpu::ShaderSource::Wgsl(wgsl.into()),
                });

            fn bgl_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
                wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }
            }

            let bind_group_layout =
                device
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("tang-gpu bgl rr_w cached"),
                        entries: &[
                            bgl_entry(0, true),
                            bgl_entry(1, true),
                            bgl_entry(2, false),
                            wgpu::BindGroupLayoutEntry {
                                binding: 3,
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

            let pipeline_layout =
                device
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("tang-gpu pipeline layout rr_w"),
                        bind_group_layouts: &[&bind_group_layout],
                        push_constant_ranges: &[],
                    });

            let pipeline =
                device
                    .device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("tang-gpu pipeline rr_w"),
                        layout: Some(&pipeline_layout),
                        module: &module,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        cache: None,
                    });

            CachedPipeline {
                pipeline,
                bind_group_layout,
            }
        })
    }

    /// Dispatch with 2 read-only inputs + 1 read-write output + uniform params.
    /// Layout: binding 0 = read A, 1 = read B, 2 = read_write out, 3 = uniform.
    pub fn dispatch_rr_w(
        &mut self,
        device: &GpuDevice,
        wgsl: &str,
        input_a: &GpuBuffer,
        input_b: &GpuBuffer,
        output: &GpuBuffer,
        params: &[u32; 4],
    ) {
        let count = params[0];
        let workgroup_size = 256u32;
        let hash = Self::hash_wgsl(wgsl);

        let cached = self.pipelines.entry(hash).or_insert_with(|| {
            let module = device
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("tang-gpu kernel"),
                    source: wgpu::ShaderSource::Wgsl(wgsl.into()),
                });

            let bind_group_layout =
                device
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("tang-gpu bgl rr_w"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
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
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 3,
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

            let pipeline_layout =
                device
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("tang-gpu pipeline layout"),
                        bind_group_layouts: &[&bind_group_layout],
                        push_constant_ranges: &[],
                    });

            let pipeline =
                device
                    .device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("tang-gpu pipeline"),
                        layout: Some(&pipeline_layout),
                        module: &module,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        cache: None,
                    });

            CachedPipeline {
                pipeline,
                bind_group_layout,
            }
        });

        use wgpu::util::DeviceExt;
        let params_buf = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tang-gpu params"),
                contents: bytemuck::cast_slice(params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tang-gpu bind group"),
            layout: &cached.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_a.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_b.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tang-gpu dispatch"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tang-gpu compute"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&cached.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let n_workgroups = count.div_ceil(workgroup_size);
            pass.dispatch_workgroups(n_workgroups, 1, 1);
        }

        self.submit_or_enqueue(device, encoder.finish());
    }

    /// Dispatch with custom params (4 u32s). First element is used as the dispatch count.
    pub fn dispatch_with_params(
        &mut self,
        device: &GpuDevice,
        wgsl: &str,
        inputs: &GpuBuffer,
        output: &GpuBuffer,
        params: &[u32; 4],
    ) {
        let count = params[0];
        let workgroup_size = 256u32;
        let cached = self.get_or_compile(device, wgsl);

        use wgpu::util::DeviceExt;
        let params_buf = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tang-gpu params"),
                contents: bytemuck::cast_slice(params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tang-gpu bind group"),
            layout: &cached.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: inputs.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tang-gpu dispatch"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tang-gpu compute"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&cached.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let n_workgroups = count.div_ceil(workgroup_size);
            pass.dispatch_workgroups(n_workgroups, 1, 1);
        }

        self.submit_or_enqueue(device, encoder.finish());
    }

    /// Dispatch Adam optimizer kernel: updates params, m, v in-place on GPU.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_adam(
        &mut self,
        device: &GpuDevice,
        params: &GpuBuffer,
        grads: &GpuBuffer,
        m: &GpuBuffer,
        v: &GpuBuffer,
        count: u32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        bc1: f32,
        bc2: f32,
    ) {
        let wgsl = r#"// Adam optimizer: updates params, m, v in-place

struct Params {
    count: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    bc1: f32,
    bc2: f32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> params: array<f32>;
@group(0) @binding(1) var<storage, read> grads: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.count) { return; }
    let g = grads[idx];
    m[idx] = p.beta1 * m[idx] + (1.0 - p.beta1) * g;
    v[idx] = p.beta2 * v[idx] + (1.0 - p.beta2) * g * g;
    let m_hat = m[idx] / p.bc1;
    let v_hat = v[idx] / p.bc2;
    params[idx] = params[idx] - p.lr * m_hat / (sqrt(v_hat) + p.eps);
}
"#;

        let workgroup_size = 256u32;
        let hash = Self::hash_wgsl(wgsl);

        let cached = self.pipelines.entry(hash).or_insert_with(|| {
            let module = device
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("tang-gpu adam kernel"),
                    source: wgpu::ShaderSource::Wgsl(wgsl.into()),
                });

            let entries: Vec<wgpu::BindGroupLayoutEntry> = vec![
                // 0: params (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: grads (read)
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
                // 2: m (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3: v (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 4: uniform params
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ];

            let bind_group_layout =
                device
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("tang-gpu adam bgl"),
                        entries: &entries,
                    });

            let pipeline_layout =
                device
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("tang-gpu adam pipeline layout"),
                        bind_group_layouts: &[&bind_group_layout],
                        push_constant_ranges: &[],
                    });

            let pipeline =
                device
                    .device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("tang-gpu adam pipeline"),
                        layout: Some(&pipeline_layout),
                        module: &module,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        cache: None,
                    });

            CachedPipeline {
                pipeline,
                bind_group_layout,
            }
        });

        // Pack uniform: count as u32, then f32 fields, then pad
        // Layout matches the WGSL struct (u32, f32, f32, f32, f32, f32, f32, u32)
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct AdamParams {
            count: u32,
            lr: f32,
            beta1: f32,
            beta2: f32,
            eps: f32,
            bc1: f32,
            bc2: f32,
            _pad: u32,
        }
        let uniform_data = AdamParams {
            count,
            lr,
            beta1,
            beta2,
            eps,
            bc1,
            bc2,
            _pad: 0,
        };

        use wgpu::util::DeviceExt;
        let uniform_buf = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tang-gpu adam params"),
                contents: bytemuck::bytes_of(&uniform_data),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tang-gpu adam bind group"),
            layout: &cached.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grads.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: m.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: v.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tang-gpu adam dispatch"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tang-gpu adam compute"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&cached.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let n_workgroups = count.div_ceil(workgroup_size);
            pass.dispatch_workgroups(n_workgroups, 1, 1);
        }

        self.submit_or_enqueue(device, encoder.finish());
    }

    /// Dispatch a kernel over arbitrary storage buffers.
    ///
    /// Binding `i` is `buffers[i]` (read-only unless listed in `writable`);
    /// the last binding is a uniform of `params`. Launches
    /// `count.div_ceil(workgroup_size)` workgroups along x.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_buffers(
        &mut self,
        device: &GpuDevice,
        wgsl: &str,
        buffers: &[&GpuBuffer],
        writable: &[usize],
        params: &[u32; 4],
        count: u32,
        workgroup_size: u32,
    ) {
        let mut bindings: Vec<BindingSpec> = (0..buffers.len())
            .map(|i| BindingSpec::Storage {
                read_only: !writable.contains(&i),
            })
            .collect();
        bindings.push(BindingSpec::Uniform);
        let hash = Self::hash_wgsl(wgsl);
        let cached = self.get_or_compile_dynamic_wg(device, wgsl, hash, &bindings, workgroup_size);

        use wgpu::util::DeviceExt;
        let params_buf = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tang-gpu params"),
                contents: bytemuck::cast_slice(params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let mut entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.buffer.as_entire_binding(),
            })
            .collect();
        entries.push(wgpu::BindGroupEntry {
            binding: buffers.len() as u32,
            resource: params_buf.as_entire_binding(),
        });
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tang-gpu bind group"),
            layout: &cached.bind_group_layout,
            entries: &entries,
        });

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tang-gpu dispatch"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tang-gpu compute"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&cached.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // Grids wider than 65535 workgroups wrap into y; kernels
            // linearize with `x + y * 65535`.
            let groups = count.div_ceil(workgroup_size);
            pass.dispatch_workgroups(groups.min(65535), groups.div_ceil(65535), 1);
        }
        self.submit_or_enqueue(device, encoder.finish());
    }

    /// The pipeline and bind group layout [`dispatch_buffers`] would use
    /// for `wgsl` over `n` storage buffers (then the uniform).
    pub(crate) fn pipeline(
        &mut self,
        device: &GpuDevice,
        wgsl: &str,
        n: usize,
        writable: &[usize],
        workgroup_size: u32,
    ) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
        let mut bindings: Vec<BindingSpec> = (0..n)
            .map(|i| BindingSpec::Storage {
                read_only: !writable.contains(&i),
            })
            .collect();
        bindings.push(BindingSpec::Uniform);
        let hash = Self::hash_wgsl(wgsl);
        let c = self.get_or_compile_dynamic_wg(device, wgsl, hash, &bindings, workgroup_size);
        (c.pipeline.clone(), c.bind_group_layout.clone())
    }

    /// A bind group of `buffers` then a uniform holding `params`.
    pub(crate) fn bind(
        &mut self,
        device: &GpuDevice,
        layout: &wgpu::BindGroupLayout,
        buffers: &[&GpuBuffer],
        params: &[u32; 4],
    ) -> wgpu::BindGroup {
        use wgpu::util::DeviceExt;
        let params_buf = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tang-gpu params"),
                contents: bytemuck::cast_slice(params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let mut entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.buffer.as_entire_binding(),
            })
            .collect();
        entries.push(wgpu::BindGroupEntry {
            binding: buffers.len() as u32,
            resource: params_buf.as_entire_binding(),
        });
        device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tang-gpu bind group"),
            layout,
            entries: &entries,
        })
    }

    /// Get or compile a pipeline with an arbitrary binding layout.
    ///
    /// Each `BindingSpec` describes one binding: its type and read-only flag.
    /// The last entry should typically be Uniform for params.
    pub(crate) fn get_or_compile_dynamic(
        &mut self,
        device: &GpuDevice,
        wgsl: &str,
        hash: u64,
        bindings: &[BindingSpec],
    ) -> &CachedPipeline {
        // WGSL carries its own workgroup size; the value only matters
        // for native MSL, which goes through `dispatch_buffers`
        self.get_or_compile_dynamic_wg(device, wgsl, hash, bindings, 1)
    }

    /// `get_or_compile_dynamic` with the threadgroup size a native
    /// (MSL) kernel is dispatched with.
    pub(crate) fn get_or_compile_dynamic_wg(
        &mut self,
        device: &GpuDevice,
        wgsl: &str,
        hash: u64,
        bindings: &[BindingSpec],
        workgroup_size: u32,
    ) -> &CachedPipeline {
        self.pipelines
            .entry(hash)
            .or_insert_with(|| Self::compile_dynamic(device, wgsl, bindings, workgroup_size))
    }

    fn compile_dynamic(
        device: &GpuDevice,
        wgsl: &str,
        bindings: &[BindingSpec],
        workgroup_size: u32,
    ) -> CachedPipeline {
        // Native MSL (from tang-compile's simdgroup-matrix lowering)
        // goes straight to the Metal compiler; bindings map to buffer
        // indices in order, which is how the kernel declares them.
        // Adreno's Vulkan driver miscompiles naga's SPIR-V for some tang
        // kernels (a RoPE map in Qwen decode came out wrong on the K1's
        // Adreno 740, right on NVIDIA and Metal); through naga's GLSL writer
        // and glslang it is right. Default there; TANG_WGSL_VIA_GLSL=0/1 forces.
        let via_glsl = !wgsl.starts_with('#')
            && device.spirv_passthrough
            && match std::env::var("TANG_WGSL_VIA_GLSL").as_deref() {
                Ok("1") => true,
                Ok("0") => false,
                _ => device.name.contains("Adreno"),
            };
        let module = if via_glsl {
            // a WGSL kernel through naga's GLSL writer and shaderc instead
            // of naga's SPIR-V writer
            let spirv = compile_glsl(&wgsl_to_glsl(wgsl));
            unsafe {
                device.device.create_shader_module_passthrough(
                    wgpu::ShaderModuleDescriptorPassthrough::SpirV(
                        wgpu::ShaderModuleDescriptorSpirV {
                            label: Some("tang-gpu kernel wgsl via glsl"),
                            source: spirv.into(),
                        },
                    ),
                )
            }
        } else if wgsl.starts_with("#version") {
            // GLSL compute (tang-compile's Vulkan decode lowering):
            // compiled to SPIR-V here and passed straight to the driver;
            // bindings are declared in the order `bindings` lists them.
            let spirv = compile_glsl(wgsl);
            unsafe {
                device.device.create_shader_module_passthrough(
                    wgpu::ShaderModuleDescriptorPassthrough::SpirV(
                        wgpu::ShaderModuleDescriptorSpirV {
                            label: Some("tang-gpu kernel glsl"),
                            source: spirv.into(),
                        },
                    ),
                )
            }
        } else if wgsl.starts_with("#include") {
            unsafe {
                device.device.create_shader_module_passthrough(
                    wgpu::ShaderModuleDescriptorPassthrough::Msl(wgpu::ShaderModuleDescriptorMsl {
                        label: Some("tang-gpu kernel msl"),
                        source: wgsl.into(),
                        entry_point: "main0".into(),
                        // the threadgroup size (wgpu-hal passes it to
                        // Metal as threads-per-threadgroup)
                        num_workgroups: (workgroup_size, 1, 1),
                    }),
                )
            }
        } else {
            device
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("tang-gpu kernel dynamic"),
                    source: wgpu::ShaderSource::Wgsl(wgsl.into()),
                })
        };

        let entries: Vec<wgpu::BindGroupLayoutEntry> = bindings
            .iter()
            .enumerate()
            .map(|(i, spec)| wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: match spec {
                    BindingSpec::Storage { read_only } => wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: *read_only,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    BindingSpec::Uniform => wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
                count: None,
            })
            .collect();

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("tang-gpu bgl dynamic"),
                    entries: &entries,
                });

        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("tang-gpu pipeline layout dynamic"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("tang-gpu pipeline dynamic"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(if wgsl.starts_with("#include") {
                    "main0"
                } else {
                    // WGSL and GLSL kernels both enter at `main`
                    "main"
                }),
                compilation_options: Default::default(),
                cache: None,
            });

        CachedPipeline {
            pipeline,
            bind_group_layout,
        }
    }

    fn hash_wgsl(wgsl: &str) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        wgsl.hash(&mut hasher);
        hasher.finish()
    }
}

/// Binding type specification for dynamic pipeline compilation.
pub(crate) enum BindingSpec {
    /// Storage buffer (read-only or read-write).
    Storage { read_only: bool },
    /// Uniform buffer.
    Uniform,
}

impl Default for KernelCache {
    fn default() -> Self {
        Self::new()
    }
}

/// GLSL compute source to SPIR-V words (Vulkan 1.1), with shaderc, through
/// the on-disk cache ([`spirv_cache_dir`]): a decode program is ~500
/// kernels, minutes of CPU on a phone-class SoC, so each compiles once.
/// `TANG_COMPILE_DUTY` (0..1) paces compiles that do run, for boards that
/// overheat at full load.
#[cfg(feature = "glsl")]
fn compile_glsl(src: &str) -> Vec<u32> {
    let level = if std::env::var("TANG_GLSL_OPT").as_deref() == Ok("0") {
        shaderc::OptimizationLevel::Zero
    } else {
        shaderc::OptimizationLevel::Performance
    };
    let paths = spirv_cache_dir().map(|dir| {
        let stem = format!("{:016x}", fnv64(&[src, "vulkan1.1", &format!("{level:?}")]));
        (
            dir.join(format!("{stem}.spv")),
            dir.join(format!("{stem}.comp")),
        )
    });
    if let Some((spv, comp)) = &paths {
        if std::fs::read_to_string(comp).is_ok_and(|s| s == src) {
            if let Ok(bytes) = std::fs::read(spv) {
                if bytes.len() % 4 == 0 && !bytes.is_empty() {
                    return bytes
                        .chunks_exact(4)
                        .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
                        .collect();
                }
            }
        }
    }
    let t = std::time::Instant::now();
    let compiler = shaderc::Compiler::new().expect("shaderc");
    let mut opts = shaderc::CompileOptions::new().expect("shaderc options");
    opts.set_target_env(
        shaderc::TargetEnv::Vulkan,
        shaderc::EnvVersion::Vulkan1_1 as u32,
    );
    // TANG_GLSL_OPT=0 hands the driver unoptimized SPIR-V
    opts.set_optimization_level(level);
    let words = match compiler.compile_into_spirv(
        src,
        shaderc::ShaderKind::Compute,
        "tang.comp",
        "main",
        Some(&opts),
    ) {
        Ok(a) => a.as_binary().to_vec(),
        Err(e) => panic!("GLSL kernel failed to compile: {e}\n{src}"),
    };
    if let Some((spv, comp)) = &paths {
        if let Some(dir) = spv.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        // the binary first: a reader trusts the pair only once the source matches
        if std::fs::write(spv, bytes).is_ok() {
            let _ = std::fs::write(comp, src);
        }
    }
    if let Some(d) = std::env::var("TANG_COMPILE_DUTY")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .map(|d| d.clamp(0.01, 1.0))
    {
        std::thread::sleep(t.elapsed().mul_f64(1.0 / d - 1.0));
    }
    words
}

/// The on-disk SPIR-V cache: `$TANG_SPIRV_CACHE`, else
/// `$XDG_CACHE_HOME/tang/spirv` or `~/.cache/tang/spirv`; `None` (no disk
/// cache) when `TANG_SPIRV_CACHE` is empty or there is no home.
#[cfg(feature = "glsl")]
pub fn spirv_cache_dir() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;
    match std::env::var_os("TANG_SPIRV_CACHE") {
        Some(d) if d.is_empty() => None,
        Some(d) => Some(PathBuf::from(d)),
        None => std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
            .map(|c| c.join("tang/spirv")),
    }
}

/// FNV-1a over `parts`, each followed by a 0 byte.
#[cfg(feature = "glsl")]
fn fnv64(parts: &[&str]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for p in parts {
        for b in p.bytes().chain([0]) {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// Without the `glsl` feature there is no GLSL compiler to hand a kernel to.
#[cfg(not(feature = "glsl"))]
fn compile_glsl(_src: &str) -> Vec<u32> {
    panic!("a GLSL kernel needs tang-gpu built with the `glsl` feature")
}

/// A WGSL compute kernel as desktop GLSL 450 (naga's GLSL writer), its
/// `@group(0) @binding(i)` resources at `binding = i`.
#[cfg(feature = "glsl")]
fn wgsl_to_glsl(wgsl: &str) -> String {
    use naga::back::glsl;
    let module = naga::front::wgsl::parse_str(wgsl).expect("WGSL kernel parses");
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("WGSL kernel validates");
    let mut binding_map = glsl::BindingMap::default();
    for (_, gv) in module.global_variables.iter() {
        if let Some(rb) = &gv.binding {
            binding_map.insert(rb.clone(), rb.binding as u8);
        }
    }
    let options = glsl::Options {
        version: glsl::Version::Desktop(450),
        writer_flags: glsl::WriterFlags::empty(),
        binding_map,
        zero_initialize_workgroup_memory: true,
    };
    let pipeline = glsl::PipelineOptions {
        shader_stage: naga::ShaderStage::Compute,
        entry_point: "main".into(),
        multiview: None,
    };
    let mut out = String::new();
    glsl::Writer::new(
        &mut out,
        &module,
        &info,
        &options,
        &pipeline,
        naga::proc::BoundsCheckPolicies::default(),
    )
    .expect("GLSL writer")
    .write()
    .expect("GLSL from WGSL");
    out
}

#[cfg(not(feature = "glsl"))]
fn wgsl_to_glsl(_wgsl: &str) -> String {
    panic!("TANG_WGSL_VIA_GLSL needs tang-gpu built with the `glsl` feature")
}
