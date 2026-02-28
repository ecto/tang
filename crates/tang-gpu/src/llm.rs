//! GPU-accelerated components for LLM inference.
//!
//! Provides the core building blocks for running transformer-based language
//! models on GPU via wgpu compute shaders:
//!
//! - [`GpuEmbedding`]: token embedding lookup
//! - [`GpuRMSNorm`]: RMS normalization
//! - [`GpuSwiGLU`]: SwiGLU feed-forward network
//! - [`GpuRoPE`]: rotary position embeddings
//! - [`GpuCausalAttention`]: grouped-query attention with causal mask
//! - [`GpuKVCache`]: key-value cache for autoregressive inference

use crate::buffer::GpuBuffer;
use crate::device::GpuDevice;
use crate::kernel::{BindingSpec, KernelCache};
use crate::tensor::GpuTensor;

use std::hash::{Hash, Hasher};

fn hash_wgsl(wgsl: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    wgsl.hash(&mut hasher);
    hasher.finish()
}

// ---------------------------------------------------------------------------
// GpuEmbedding
// ---------------------------------------------------------------------------

/// Token embedding lookup on GPU.
///
/// Stores an embedding table of shape `[vocab_size, dim]` and looks up
/// rows by token ID. Input: buffer of u32 token IDs. Output: `[seq_len, dim]`.
pub struct GpuEmbedding {
    /// Embedding weight table [vocab_size, dim].
    pub weight: GpuTensor,
    pub vocab_size: usize,
    pub dim: usize,
}

impl GpuEmbedding {
    /// Create an embedding layer from a weight table.
    pub fn new(device: &GpuDevice, weight: &[f32], vocab_size: usize, dim: usize) -> Self {
        assert_eq!(weight.len(), vocab_size * dim);
        Self {
            weight: GpuTensor::from_slice(device, weight, &[vocab_size, dim]),
            vocab_size,
            dim,
        }
    }

    /// Create an embedding layer with zero weights.
    pub fn zeros(device: &GpuDevice, vocab_size: usize, dim: usize) -> Self {
        Self::new(device, &vec![0.0f32; vocab_size * dim], vocab_size, dim)
    }

    /// Look up embeddings for a sequence of token IDs.
    ///
    /// `token_ids`: u32 buffer of length `seq_len`.
    /// Returns: `GpuTensor` of shape `[seq_len, dim]`.
    pub fn forward(
        &self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        token_ids: &GpuBuffer,
        seq_len: usize,
    ) -> GpuTensor {
        let out = GpuTensor::uninit(device, &[seq_len, self.dim]);

        let wgsl = r#"// Embedding lookup: output[i, :] = weight[token_ids[i], :]

struct Params {
    seq_len: u32,
    dim: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> token_ids: array<u32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.seq_len * params.dim;
    if (idx >= total) { return; }
    let row = idx / params.dim;
    let col = idx % params.dim;
    let tok = token_ids[row];
    output[idx] = weight[tok * params.dim + col];
}
"#;

        let params: [u32; 4] = [seq_len as u32, self.dim as u32, 0, 0];

        let hash = hash_wgsl(wgsl);
        let bindings = [
            BindingSpec::Storage { read_only: true },  // token_ids
            BindingSpec::Storage { read_only: true },  // weight
            BindingSpec::Storage { read_only: false },  // output
            BindingSpec::Uniform,                       // params
        ];
        let cached = cache.get_or_compile_dynamic(device, wgsl, hash, &bindings);

        use wgpu::util::DeviceExt;
        let params_buf = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("embedding params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("embedding bind group"),
            layout: &cached.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: token_ids.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.weight.buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.buffer.buffer.as_entire_binding(),
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
                label: Some("embedding dispatch"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("embedding compute"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&cached.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let total = (seq_len * self.dim) as u32;
            pass.dispatch_workgroups(total.div_ceil(256), 1, 1);
        }

        cache.submit_or_enqueue(device, encoder.finish());
        out
    }
}

// ---------------------------------------------------------------------------
// GpuRMSNorm
// ---------------------------------------------------------------------------

/// RMS normalization: `x * rsqrt(mean(x^2) + eps) * weight`.
///
/// Operates on the last dimension. One workgroup per normalization group
/// with shared-memory reduction for the RMS statistic.
pub struct GpuRMSNorm {
    /// Scale parameter [dim].
    pub weight: GpuTensor,
    pub eps: f32,
    pub dim: usize,
}

impl GpuRMSNorm {
    /// Create RMS norm with ones for weight.
    pub fn new(device: &GpuDevice, dim: usize, eps: f32) -> Self {
        let ones = vec![1.0f32; dim];
        Self {
            weight: GpuTensor::from_slice(device, &ones, &[dim]),
            eps,
            dim,
        }
    }

    /// Apply RMS normalization on GPU.
    pub fn forward(
        &self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        input: &GpuTensor,
    ) -> GpuTensor {
        let n_groups = input.numel() / self.dim;
        let out = GpuTensor::uninit(device, input.shape());

        let wg_size = (self.dim as u32).next_power_of_two().min(256);

        let wgsl = format!(
            r#"// RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight
// One workgroup per normalization group

struct Params {{
    n_groups: u32,
    dim: u32,
    eps: f32,
    _pad: u32,
}}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = {wg_size}u;
var<workgroup> wg_buf: array<f32, {wg_size}>;

@compute @workgroup_size({wg_size})
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    let group = wg_id.x;
    if (group >= params.n_groups) {{ return; }}
    let tid = lid.x;
    let base = group * params.dim;

    // Compute mean(x^2)
    var local_sum: f32 = 0.0;
    for (var i = tid; i < params.dim; i = i + WG) {{
        let val = input[base + i];
        local_sum = local_sum + val * val;
    }}
    wg_buf[tid] = local_sum;
    workgroupBarrier();

    for (var stride: u32 = WG / 2u; stride > 0u; stride = stride / 2u) {{
        if (tid < stride) {{
            wg_buf[tid] = wg_buf[tid] + wg_buf[tid + stride];
        }}
        workgroupBarrier();
    }}
    let rms = sqrt(wg_buf[0] / f32(params.dim) + params.eps);
    let inv_rms = 1.0 / rms;
    workgroupBarrier();

    // Normalize and scale
    for (var i = tid; i < params.dim; i = i + WG) {{
        output[base + i] = input[base + i] * inv_rms * weight[i];
    }}
}}
"#,
            wg_size = wg_size,
        );

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct RmsParams {
            n_groups: u32,
            dim: u32,
            eps: f32,
            _pad: u32,
        }
        let uniform = RmsParams {
            n_groups: n_groups as u32,
            dim: self.dim as u32,
            eps: self.eps,
            _pad: 0,
        };

        let hash = hash_wgsl(&wgsl);
        let bindings = [
            BindingSpec::Storage { read_only: true },   // input
            BindingSpec::Storage { read_only: true },   // weight
            BindingSpec::Storage { read_only: false },   // output
            BindingSpec::Uniform,                        // params
        ];
        let cached = cache.get_or_compile_dynamic(device, &wgsl, hash, &bindings);

        use wgpu::util::DeviceExt;
        let params_buf = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rmsnorm params"),
                contents: bytemuck::bytes_of(&uniform),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rmsnorm bind group"),
            layout: &cached.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.weight.buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.buffer.buffer.as_entire_binding(),
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
                label: Some("rmsnorm dispatch"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rmsnorm compute"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&cached.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(n_groups as u32, 1, 1);
        }

        cache.submit_or_enqueue(device, encoder.finish());
        out
    }
}

// ---------------------------------------------------------------------------
// GpuSwiGLU
// ---------------------------------------------------------------------------

/// SwiGLU feed-forward network: `down_proj(silu(gate_proj(x)) * up_proj(x))`.
///
/// gate_proj: [dim, hidden_dim], up_proj: [dim, hidden_dim], down_proj: [hidden_dim, dim].
pub struct GpuSwiGLU {
    /// Gate projection [hidden_dim, dim] (stored transposed for matmul).
    pub gate_proj: GpuTensor,
    /// Up projection [hidden_dim, dim].
    pub up_proj: GpuTensor,
    /// Down projection [dim, hidden_dim].
    pub down_proj: GpuTensor,
    pub dim: usize,
    pub hidden_dim: usize,
}

impl GpuSwiGLU {
    /// Create a SwiGLU layer with given weights.
    ///
    /// Weight shapes: gate/up = `[hidden_dim, dim]`, down = `[dim, hidden_dim]`.
    pub fn new(
        device: &GpuDevice,
        gate: &[f32],
        up: &[f32],
        down: &[f32],
        dim: usize,
        hidden_dim: usize,
    ) -> Self {
        assert_eq!(gate.len(), hidden_dim * dim);
        assert_eq!(up.len(), hidden_dim * dim);
        assert_eq!(down.len(), dim * hidden_dim);
        Self {
            gate_proj: GpuTensor::from_slice(device, gate, &[hidden_dim, dim]),
            up_proj: GpuTensor::from_slice(device, up, &[hidden_dim, dim]),
            down_proj: GpuTensor::from_slice(device, down, &[dim, hidden_dim]),
            dim,
            hidden_dim,
        }
    }

    /// Create a SwiGLU layer with zero weights.
    pub fn zeros(device: &GpuDevice, dim: usize, hidden_dim: usize) -> Self {
        Self::new(
            device,
            &vec![0.0f32; hidden_dim * dim],
            &vec![0.0f32; hidden_dim * dim],
            &vec![0.0f32; dim * hidden_dim],
            dim,
            hidden_dim,
        )
    }

    /// Forward pass: `down_proj(silu(gate_proj(x)) * up_proj(x))`.
    ///
    /// Input: `[seq_len, dim]` or `[dim]`. Output: same shape as input.
    pub fn forward(
        &self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        input: &GpuTensor,
    ) -> GpuTensor {
        use crate::matmul::matmul;

        // Ensure 2D for matmul
        let was_1d = input.ndim() == 1;
        let input_2d = if was_1d {
            GpuTensor {
                buffer: input.buffer.clone_gpu_batched(device, cache),
                shape: vec![1, input.numel()],
            }
        } else {
            GpuTensor {
                buffer: input.buffer.clone_gpu_batched(device, cache),
                shape: input.shape().to_vec(),
            }
        };

        // gate = input @ gate_proj^T -> [seq_len, hidden_dim]
        let gate_t = self.gate_proj.transpose_gpu(device, cache);
        let gate = matmul(device, cache, &input_2d, &gate_t);

        // up = input @ up_proj^T -> [seq_len, hidden_dim]
        let up_t = self.up_proj.transpose_gpu(device, cache);
        let up = matmul(device, cache, &input_2d, &up_t);

        // silu(gate) * up -> fused in one kernel
        let activated = swiglu_fused(device, cache, &gate, &up);

        // down = activated @ down_proj^T -> [seq_len, dim]
        let down_t = self.down_proj.transpose_gpu(device, cache);
        let result = matmul(device, cache, &activated, &down_t);

        if was_1d {
            GpuTensor {
                buffer: result.buffer,
                shape: vec![self.dim],
            }
        } else {
            result
        }
    }
}

/// Fused SiLU(gate) * up kernel.
fn swiglu_fused(
    device: &GpuDevice,
    cache: &mut KernelCache,
    gate: &GpuTensor,
    up: &GpuTensor,
) -> GpuTensor {
    assert_eq!(gate.numel(), up.numel());
    let numel = gate.numel() as u32;
    let out = GpuTensor::uninit(device, gate.shape());

    let wgsl = r#"// Fused SiLU(gate) * up: output[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i]

struct Params {
    count: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) { return; }
    let g = gate[idx];
    let silu_g = g / (1.0 + exp(-g));
    output[idx] = silu_g * up[idx];
}
"#;

    cache.dispatch_rr_w(device, wgsl, &gate.buffer, &up.buffer, &out.buffer, &[numel, 0, 0, 0]);
    out
}

// ---------------------------------------------------------------------------
// GpuRoPE
// ---------------------------------------------------------------------------

/// Rotary position embeddings (RoPE) with configurable frequency base.
///
/// Applies rotation to pairs of dimensions in the input. Supports
/// base frequencies up to 500K for long-context models.
pub struct GpuRoPE {
    /// Precomputed cos table [max_seq_len, head_dim/2].
    pub cos_table: GpuTensor,
    /// Precomputed sin table [max_seq_len, head_dim/2].
    pub sin_table: GpuTensor,
    pub head_dim: usize,
    pub max_seq_len: usize,
}

impl GpuRoPE {
    /// Create RoPE with precomputed frequency tables.
    ///
    /// `base`: frequency base (10000.0 standard, up to 500000.0 for long context).
    /// `head_dim`: dimension of each attention head (must be even).
    /// `max_seq_len`: maximum sequence length to precompute.
    pub fn new(device: &GpuDevice, head_dim: usize, max_seq_len: usize, base: f32) -> Self {
        assert_eq!(head_dim % 2, 0, "RoPE head_dim must be even");
        let half = head_dim / 2;

        let mut cos_data = vec![0.0f32; max_seq_len * half];
        let mut sin_data = vec![0.0f32; max_seq_len * half];

        for pos in 0..max_seq_len {
            for i in 0..half {
                let freq = 1.0 / base.powf(2.0 * i as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                cos_data[pos * half + i] = angle.cos();
                sin_data[pos * half + i] = angle.sin();
            }
        }

        Self {
            cos_table: GpuTensor::from_slice(device, &cos_data, &[max_seq_len, half]),
            sin_table: GpuTensor::from_slice(device, &sin_data, &[max_seq_len, half]),
            head_dim,
            max_seq_len,
        }
    }

    /// Apply rotary embeddings to input tensor.
    ///
    /// Input: `[seq_len, n_heads, head_dim]`. Output: same shape.
    /// `start_pos`: position offset (for KV cache continuation).
    pub fn forward(
        &self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        input: &GpuTensor,
        start_pos: usize,
    ) -> GpuTensor {
        assert_eq!(input.ndim(), 3, "RoPE input must be [seq_len, n_heads, head_dim]");
        let seq_len = input.shape()[0];
        let n_heads = input.shape()[1];
        let head_dim = input.shape()[2];
        assert_eq!(head_dim, self.head_dim);
        assert!(
            start_pos + seq_len <= self.max_seq_len,
            "RoPE: position {} + seq_len {} exceeds max {}",
            start_pos,
            seq_len,
            self.max_seq_len
        );

        let half = head_dim / 2;
        let out = GpuTensor::uninit(device, input.shape());

        let wgsl = r#"// RoPE: rotate pairs of dimensions using precomputed cos/sin
// Input/output: [seq_len, n_heads, head_dim]
// cos/sin tables: [max_seq_len, head_dim/2]

struct Params {
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
    half_dim: u32,
    start_pos: u32,
    max_seq_len: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> cos_table: array<f32>;
@group(0) @binding(2) var<storage, read> sin_table: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.seq_len * params.n_heads * params.half_dim;
    if (idx >= total) { return; }

    // Decode: idx -> (pos_in_seq, head, pair)
    let pair = idx % params.half_dim;
    let remainder = idx / params.half_dim;
    let head = remainder % params.n_heads;
    let pos_in_seq = remainder / params.n_heads;
    let pos = pos_in_seq + params.start_pos;

    // Input indices for the pair
    let base_idx = (pos_in_seq * params.n_heads + head) * params.head_dim;
    let i0 = base_idx + pair;
    let i1 = base_idx + pair + params.half_dim;

    // Lookup cos/sin for this position and dimension pair
    let table_idx = pos * params.half_dim + pair;
    let c = cos_table[table_idx];
    let s = sin_table[table_idx];

    let x0 = input[i0];
    let x1 = input[i1];

    // Apply rotation
    output[i0] = x0 * c - x1 * s;
    output[i1] = x0 * s + x1 * c;
}
"#;

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct RopeParams {
            seq_len: u32,
            n_heads: u32,
            head_dim: u32,
            half_dim: u32,
            start_pos: u32,
            max_seq_len: u32,
            _pad2: u32,
            _pad3: u32,
        }
        let uniform = RopeParams {
            seq_len: seq_len as u32,
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
            half_dim: half as u32,
            start_pos: start_pos as u32,
            max_seq_len: self.max_seq_len as u32,
            _pad2: 0,
            _pad3: 0,
        };

        let hash = hash_wgsl(wgsl);
        let bindings = [
            BindingSpec::Storage { read_only: true },   // input
            BindingSpec::Storage { read_only: true },   // cos_table
            BindingSpec::Storage { read_only: true },   // sin_table
            BindingSpec::Storage { read_only: false },   // output
            BindingSpec::Uniform,                        // params
        ];
        let cached = cache.get_or_compile_dynamic(device, wgsl, hash, &bindings);

        use wgpu::util::DeviceExt;
        let params_buf = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rope params"),
                contents: bytemuck::bytes_of(&uniform),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rope bind group"),
            layout: &cached.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.cos_table.buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.sin_table.buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out.buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rope dispatch"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rope compute"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&cached.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let total = (seq_len * n_heads * half) as u32;
            pass.dispatch_workgroups(total.div_ceil(256), 1, 1);
        }

        cache.submit_or_enqueue(device, encoder.finish());
        out
    }
}

// ---------------------------------------------------------------------------
// GpuInterleavedRoPE
// ---------------------------------------------------------------------------

/// Rotary position embeddings with interleaved pair convention.
///
/// Pairs `(d, d+1)` for `d = 0, 2, 4, ...` — matching the CPU `RotaryEmbedding`
/// convention used by gaia/tang-train. Use this when weights were trained with
/// interleaved pairs rather than the halved convention of [`GpuRoPE`].
pub struct GpuInterleavedRoPE {
    /// Precomputed cos table `[max_seq_len, head_dim/2]`.
    pub cos_table: GpuTensor,
    /// Precomputed sin table `[max_seq_len, head_dim/2]`.
    pub sin_table: GpuTensor,
    pub head_dim: usize,
    pub max_seq_len: usize,
}

impl GpuInterleavedRoPE {
    /// Create with precomputed frequency tables matching CPU interleaved convention.
    ///
    /// `base`: frequency base (e.g. 500000.0 for long-context models).
    /// `head_dim`: dimension of each attention head (must be even).
    /// `max_seq_len`: maximum sequence length to precompute.
    pub fn new(device: &GpuDevice, head_dim: usize, max_seq_len: usize, base: f64) -> Self {
        assert_eq!(head_dim % 2, 0, "RoPE head_dim must be even");
        let half = head_dim / 2;

        // theta_i = pos / base^(2i/dim) — identical to CPU RotaryEmbedding::with_base
        let mut cos_data = vec![0.0f32; max_seq_len * half];
        let mut sin_data = vec![0.0f32; max_seq_len * half];

        for pos in 0..max_seq_len {
            for i in 0..half {
                let theta = pos as f64 / base.powf(2.0 * i as f64 / head_dim as f64);
                cos_data[pos * half + i] = theta.cos() as f32;
                sin_data[pos * half + i] = theta.sin() as f32;
            }
        }

        Self {
            cos_table: GpuTensor::from_slice(device, &cos_data, &[max_seq_len, half]),
            sin_table: GpuTensor::from_slice(device, &sin_data, &[max_seq_len, half]),
            head_dim,
            max_seq_len,
        }
    }

    /// Apply interleaved RoPE to input tensor.
    ///
    /// Input: `[seq_len, n_heads, head_dim]`. Output: same shape.
    /// `start_pos`: position offset (for KV cache continuation).
    ///
    /// For each position and head, rotates interleaved pairs:
    /// - `output[2i]   = x[2i] * cos - x[2i+1] * sin`
    /// - `output[2i+1] = x[2i] * sin + x[2i+1] * cos`
    pub fn forward(
        &self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        input: &GpuTensor,
        start_pos: usize,
    ) -> GpuTensor {
        assert_eq!(input.ndim(), 3, "RoPE input must be [seq_len, n_heads, head_dim]");
        let seq_len = input.shape()[0];
        let n_heads = input.shape()[1];
        let head_dim = input.shape()[2];
        assert_eq!(head_dim, self.head_dim);
        assert!(
            start_pos + seq_len <= self.max_seq_len,
            "RoPE: position {} + seq_len {} exceeds max {}",
            start_pos, seq_len, self.max_seq_len
        );

        let half = head_dim / 2;
        let out = GpuTensor::uninit(device, input.shape());

        let wgsl = r#"// Interleaved RoPE: rotate pairs (2i, 2i+1)

struct Params {
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
    half_dim: u32,
    start_pos: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> cos_table: array<f32>;
@group(0) @binding(2) var<storage, read> sin_table: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.seq_len * params.n_heads * params.half_dim;
    if (idx >= total) { return; }

    let pair = idx % params.half_dim;
    let remainder = idx / params.half_dim;
    let head = remainder % params.n_heads;
    let pos_in_seq = remainder / params.n_heads;
    let pos = pos_in_seq + params.start_pos;

    let base_idx = (pos_in_seq * params.n_heads + head) * params.head_dim;
    let i0 = base_idx + pair * 2u;
    let i1 = base_idx + pair * 2u + 1u;

    let table_idx = pos * params.half_dim + pair;
    let c = cos_table[table_idx];
    let s = sin_table[table_idx];

    let x0 = input[i0];
    let x1 = input[i1];

    output[i0] = x0 * c - x1 * s;
    output[i1] = x0 * s + x1 * c;
}
"#;

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct RopeParams {
            seq_len: u32,
            n_heads: u32,
            head_dim: u32,
            half_dim: u32,
            start_pos: u32,
            _pad1: u32,
            _pad2: u32,
            _pad3: u32,
        }
        let uniform = RopeParams {
            seq_len: seq_len as u32,
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
            half_dim: half as u32,
            start_pos: start_pos as u32,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };

        let hash = hash_wgsl(wgsl);
        let bindings = [
            BindingSpec::Storage { read_only: true },
            BindingSpec::Storage { read_only: true },
            BindingSpec::Storage { read_only: true },
            BindingSpec::Storage { read_only: false },
            BindingSpec::Uniform,
        ];
        let cached = cache.get_or_compile_dynamic(device, wgsl, hash, &bindings);

        use wgpu::util::DeviceExt;
        let params_buf = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("interleaved rope params"),
                contents: bytemuck::bytes_of(&uniform),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("interleaved rope bind group"),
            layout: &cached.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.cos_table.buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.sin_table.buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out.buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("interleaved rope dispatch"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("interleaved rope compute"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&cached.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let total = (seq_len * n_heads * half) as u32;
            pass.dispatch_workgroups(total.div_ceil(256), 1, 1);
        }

        cache.submit_or_enqueue(device, encoder.finish());
        out
    }
}

// ---------------------------------------------------------------------------
// GpuKVCache
// ---------------------------------------------------------------------------

/// Simple KV cache for autoregressive inference.
///
/// Stores key and value tensors and supports appending new entries
/// and retrieving the full sequence. Fixed maximum sequence length.
pub struct GpuKVCache {
    /// Key cache [max_seq_len, n_kv_heads, head_dim].
    pub keys: GpuBuffer,
    /// Value cache [max_seq_len, n_kv_heads, head_dim].
    pub values: GpuBuffer,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    /// Current number of cached positions.
    pub len: usize,
}

impl GpuKVCache {
    /// Create an empty KV cache.
    pub fn new(device: &GpuDevice, n_kv_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        let size = max_seq_len * n_kv_heads * head_dim;
        Self {
            keys: GpuBuffer::uninit(device, size),
            values: GpuBuffer::uninit(device, size),
            n_kv_heads,
            head_dim,
            max_seq_len,
            len: 0,
        }
    }

    /// Append new key and value entries to the cache.
    ///
    /// `new_keys`, `new_values`: `[new_seq_len, n_kv_heads, head_dim]`.
    pub fn append(
        &mut self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        new_keys: &GpuTensor,
        new_values: &GpuTensor,
    ) {
        let new_seq = new_keys.shape()[0];
        assert!(
            self.len + new_seq <= self.max_seq_len,
            "KV cache overflow: {} + {} > {}",
            self.len,
            new_seq,
            self.max_seq_len
        );

        let row_size = self.n_kv_heads * self.head_dim;
        let offset_bytes = (self.len * row_size * std::mem::size_of::<f32>()) as u64;
        let copy_bytes = (new_seq * row_size * std::mem::size_of::<f32>()) as u64;

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("kv cache append"),
            });

        encoder.copy_buffer_to_buffer(
            &new_keys.buffer.buffer,
            0,
            &self.keys.buffer,
            offset_bytes,
            copy_bytes,
        );
        encoder.copy_buffer_to_buffer(
            &new_values.buffer.buffer,
            0,
            &self.values.buffer,
            offset_bytes,
            copy_bytes,
        );

        cache.submit_or_enqueue(device, encoder.finish());
        self.len += new_seq;
    }

    /// Get cached keys as a tensor of shape `[current_len, n_kv_heads, head_dim]`.
    pub fn get_keys(&self, device: &GpuDevice) -> GpuTensor {
        let row_size = self.n_kv_heads * self.head_dim;
        let total = self.len * row_size;
        let data = self.keys.to_vec_sync(device);
        GpuTensor::from_slice(device, &data[..total], &[self.len, self.n_kv_heads, self.head_dim])
    }

    /// Get cached values as a tensor of shape `[current_len, n_kv_heads, head_dim]`.
    pub fn get_values(&self, device: &GpuDevice) -> GpuTensor {
        let row_size = self.n_kv_heads * self.head_dim;
        let total = self.len * row_size;
        let data = self.values.to_vec_sync(device);
        GpuTensor::from_slice(device, &data[..total], &[self.len, self.n_kv_heads, self.head_dim])
    }

    /// Get cached keys as a tensor, staying on GPU (no CPU roundtrip).
    ///
    /// Returns `[current_len, n_kv_heads, head_dim]`.
    pub fn get_keys_gpu(
        &self,
        device: &GpuDevice,
        cache: &mut KernelCache,
    ) -> GpuTensor {
        let row_size = self.n_kv_heads * self.head_dim;
        let total = self.len * row_size;
        let dst = GpuBuffer::uninit(device, total);
        let copy_bytes = (total * std::mem::size_of::<f32>()) as u64;

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("kv cache get_keys_gpu"),
            });
        encoder.copy_buffer_to_buffer(&self.keys.buffer, 0, &dst.buffer, 0, copy_bytes);
        cache.submit_or_enqueue(device, encoder.finish());

        GpuTensor {
            buffer: dst,
            shape: vec![self.len, self.n_kv_heads, self.head_dim],
        }
    }

    /// Get cached values as a tensor, staying on GPU (no CPU roundtrip).
    ///
    /// Returns `[current_len, n_kv_heads, head_dim]`.
    pub fn get_values_gpu(
        &self,
        device: &GpuDevice,
        cache: &mut KernelCache,
    ) -> GpuTensor {
        let row_size = self.n_kv_heads * self.head_dim;
        let total = self.len * row_size;
        let dst = GpuBuffer::uninit(device, total);
        let copy_bytes = (total * std::mem::size_of::<f32>()) as u64;

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("kv cache get_values_gpu"),
            });
        encoder.copy_buffer_to_buffer(&self.values.buffer, 0, &dst.buffer, 0, copy_bytes);
        cache.submit_or_enqueue(device, encoder.finish());

        GpuTensor {
            buffer: dst,
            shape: vec![self.len, self.n_kv_heads, self.head_dim],
        }
    }

    /// Reset the cache (set length to 0).
    pub fn clear(&mut self) {
        self.len = 0;
    }
}

// ---------------------------------------------------------------------------
// GpuCausalAttention
// ---------------------------------------------------------------------------

/// Grouped-query attention with causal mask.
///
/// Supports variable numbers of query heads per KV head (GQA).
/// Uses a fused attention kernel with causal masking for autoregressive
/// inference.
pub struct GpuCausalAttention {
    /// Query projection [n_heads * head_dim, dim].
    pub wq: GpuTensor,
    /// Key projection [n_kv_heads * head_dim, dim].
    pub wk: GpuTensor,
    /// Value projection [n_kv_heads * head_dim, dim].
    pub wv: GpuTensor,
    /// Output projection [dim, n_heads * head_dim].
    pub wo: GpuTensor,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub dim: usize,
}

impl GpuCausalAttention {
    /// Create attention with zero-initialized projections.
    pub fn zeros(
        device: &GpuDevice,
        dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        assert_eq!(n_heads % n_kv_heads, 0, "n_heads must be divisible by n_kv_heads");
        let q_size = n_heads * head_dim * dim;
        let kv_size = n_kv_heads * head_dim * dim;
        let o_size = dim * n_heads * head_dim;
        Self {
            wq: GpuTensor::from_slice(device, &vec![0.0f32; q_size], &[n_heads * head_dim, dim]),
            wk: GpuTensor::from_slice(device, &vec![0.0f32; kv_size], &[n_kv_heads * head_dim, dim]),
            wv: GpuTensor::from_slice(device, &vec![0.0f32; kv_size], &[n_kv_heads * head_dim, dim]),
            wo: GpuTensor::from_slice(device, &vec![0.0f32; o_size], &[dim, n_heads * head_dim]),
            n_heads,
            n_kv_heads,
            head_dim,
            dim,
        }
    }

    /// Forward pass with grouped-query attention and causal masking.
    ///
    /// Input: `[seq_len, dim]`. Output: `[seq_len, dim]`.
    ///
    /// The causal mask ensures position `i` can only attend to positions `<= i`.
    /// KV heads are shared across query head groups (GQA).
    pub fn forward(
        &self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        input: &GpuTensor,
    ) -> GpuTensor {
        use crate::matmul::matmul;

        assert_eq!(input.ndim(), 2, "attention input must be [seq_len, dim]");
        let seq_len = input.shape()[0];

        // Q = input @ Wq^T -> [seq_len, n_heads * head_dim]
        let wq_t = self.wq.transpose_gpu(device, cache);
        let q_flat = matmul(device, cache, input, &wq_t);

        // K = input @ Wk^T -> [seq_len, n_kv_heads * head_dim]
        let wk_t = self.wk.transpose_gpu(device, cache);
        let k_flat = matmul(device, cache, input, &wk_t);

        // V = input @ Wv^T -> [seq_len, n_kv_heads * head_dim]
        let wv_t = self.wv.transpose_gpu(device, cache);
        let v_flat = matmul(device, cache, input, &wv_t);

        // Fused causal attention with GQA
        let attn_out = causal_attention_fused(
            device,
            cache,
            &q_flat,
            &k_flat,
            &v_flat,
            seq_len,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
        );

        // Output projection: attn_out @ Wo^T -> [seq_len, dim]
        let wo_t = self.wo.transpose_gpu(device, cache);
        matmul(device, cache, &attn_out, &wo_t)
    }
}

/// Fused causal attention with grouped-query support.
///
/// Q: [seq_len, n_heads * head_dim], K: [seq_len, n_kv_heads * head_dim],
/// V: [seq_len, n_kv_heads * head_dim] -> output: [seq_len, n_heads * head_dim].
fn causal_attention_fused(
    device: &GpuDevice,
    cache: &mut KernelCache,
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    seq_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> GpuTensor {
    let heads_per_group = n_heads / n_kv_heads;
    let out = GpuTensor::uninit(device, &[seq_len, n_heads * head_dim]);

    // Use a workgroup per (query_pos, head) pair.
    // Each workgroup computes attention for one query position and one head.
    let wg_size = (seq_len as u32).next_power_of_two().min(256).max(1);

    let wgsl = format!(
        r#"// Fused causal GQA attention
// One workgroup per (query_pos, head) pair
// Each thread covers one or more key positions

struct Params {{
    seq_len: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    heads_per_group: u32,
    scale: f32,
    _pad2: u32,
    _pad3: u32,
}}

@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const WG: u32 = {wg_size}u;
var<workgroup> wg_scores: array<f32, {wg_size}>;
var<workgroup> wg_max: array<f32, {wg_size}>;
var<workgroup> wg_sum: array<f32, {wg_size}>;

@compute @workgroup_size({wg_size})
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    let query_pos = wg_id.x;
    let head = wg_id.y;
    if (query_pos >= params.seq_len || head >= params.n_heads) {{ return; }}
    let tid = lid.x;
    let kv_head = head / params.heads_per_group;

    // Q offset for this head: Q[query_pos, head * head_dim .. (head+1) * head_dim]
    let q_base = query_pos * params.n_heads * params.head_dim + head * params.head_dim;
    // K/V row stride
    let kv_stride = params.n_kv_heads * params.head_dim;

    // Phase 1: Compute dot products Q . K for each key position
    // Each thread handles a subset of key positions
    var local_max: f32 = -3.402823e+38;
    for (var kp = tid; kp < params.seq_len; kp = kp + WG) {{
        if (kp <= query_pos) {{
            // Causal: only attend to positions <= query_pos
            let k_base = kp * kv_stride + kv_head * params.head_dim;
            var dot: f32 = 0.0;
            for (var d: u32 = 0u; d < params.head_dim; d = d + 1u) {{
                dot = dot + Q[q_base + d] * K[k_base + d];
            }}
            let score = dot * params.scale;
            wg_scores[kp % WG] = score;
            local_max = max(local_max, score);
        }}
    }}
    wg_max[tid] = local_max;
    workgroupBarrier();

    // Reduce max
    for (var stride: u32 = WG / 2u; stride > 0u; stride = stride / 2u) {{
        if (tid < stride) {{
            wg_max[tid] = max(wg_max[tid], wg_max[tid + stride]);
        }}
        workgroupBarrier();
    }}
    let row_max = wg_max[0];
    workgroupBarrier();

    // Phase 2: Compute exp(score - max) and accumulate weighted V
    // For small seq_len, we can store all scores and do two passes.
    // For larger seq_len, we use online softmax.

    // Initialize output accumulator (per-thread partial sums)
    // Since head_dim could be large, we accumulate across all positions per dimension

    // Simple approach: iterate key positions, accumulate weighted V
    var local_exp_sum: f32 = 0.0;

    // Output accumulator: we need head_dim values.
    // We iterate over key positions and accumulate on the fly.
    // Since we only have WG shared memory slots, and head_dim could be > WG,
    // we compute in two passes: first get softmax weights, then weighted sum.

    // Pass 1: compute exp weights and sum
    for (var kp = tid; kp < params.seq_len; kp = kp + WG) {{
        if (kp <= query_pos) {{
            let k_base = kp * kv_stride + kv_head * params.head_dim;
            var dot: f32 = 0.0;
            for (var d: u32 = 0u; d < params.head_dim; d = d + 1u) {{
                dot = dot + Q[q_base + d] * K[k_base + d];
            }}
            let w = exp(dot * params.scale - row_max);
            wg_scores[kp % WG] = w;
            local_exp_sum = local_exp_sum + w;
        }}
    }}
    wg_sum[tid] = local_exp_sum;
    workgroupBarrier();

    // Reduce sum
    for (var stride: u32 = WG / 2u; stride > 0u; stride = stride / 2u) {{
        if (tid < stride) {{
            wg_sum[tid] = wg_sum[tid] + wg_sum[tid + stride];
        }}
        workgroupBarrier();
    }}
    let total_sum = wg_sum[0];
    let inv_sum = 1.0 / total_sum;
    workgroupBarrier();

    // Phase 3: Compute weighted V sum, each thread handles a subset of head_dim
    let out_base = query_pos * params.n_heads * params.head_dim + head * params.head_dim;
    for (var d = tid; d < params.head_dim; d = d + WG) {{
        var acc: f32 = 0.0;
        for (var kp: u32 = 0u; kp <= query_pos; kp = kp + 1u) {{
            let v_base = kp * kv_stride + kv_head * params.head_dim;
            // Recompute weight (avoids needing seq_len shared memory)
            let k_base = kp * kv_stride + kv_head * params.head_dim;
            var dot: f32 = 0.0;
            for (var dd: u32 = 0u; dd < params.head_dim; dd = dd + 1u) {{
                dot = dot + Q[q_base + dd] * K[k_base + dd];
            }}
            let w = exp(dot * params.scale - row_max) * inv_sum;
            acc = acc + w * V[v_base + d];
        }}
        output[out_base + d] = acc;
    }}
}}
"#,
        wg_size = wg_size,
    );

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct AttnParams {
        seq_len: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        heads_per_group: u32,
        scale: f32,
        _pad2: u32,
        _pad3: u32,
    }
    let scale = 1.0 / (head_dim as f32).sqrt();
    let uniform = AttnParams {
        seq_len: seq_len as u32,
        n_heads: n_heads as u32,
        n_kv_heads: n_kv_heads as u32,
        head_dim: head_dim as u32,
        heads_per_group: heads_per_group as u32,
        scale,
        _pad2: 0,
        _pad3: 0,
    };

    let hash = hash_wgsl(&wgsl);
    let bindings = [
        BindingSpec::Storage { read_only: true },   // Q
        BindingSpec::Storage { read_only: true },   // K
        BindingSpec::Storage { read_only: true },   // V
        BindingSpec::Storage { read_only: false },   // output
        BindingSpec::Uniform,                        // params
    ];
    let cached = cache.get_or_compile_dynamic(device, &wgsl, hash, &bindings);

    use wgpu::util::DeviceExt;
    let params_buf = device
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("causal attn params"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("causal attn bind group"),
        layout: &cached.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: q.buffer.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: k.buffer.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: v.buffer.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: out.buffer.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("causal attn dispatch"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("causal attn compute"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&cached.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // One workgroup per (query_pos, head) pair
        pass.dispatch_workgroups(seq_len as u32, n_heads as u32, 1);
    }

    cache.submit_or_enqueue(device, encoder.finish());
    out
}

/// Fused causal attention with separate Q and KV sequence lengths.
///
/// Supports KV-cached inference where Q has `q_len` new positions and
/// K/V have `kv_len` total cached positions (including the new ones).
///
/// - `q`: `[q_len, n_heads * head_dim]`
/// - `k`: `[kv_len, n_kv_heads * head_dim]`
/// - `v`: `[kv_len, n_kv_heads * head_dim]`
/// - `start_pos`: absolute position of the first query token. Query at batch
///   index `i` can attend to key positions `0..=start_pos+i`.
///
/// Returns: `[q_len, n_heads * head_dim]`.
pub fn kv_attention_fused(
    device: &GpuDevice,
    cache: &mut KernelCache,
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    q_len: usize,
    kv_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    start_pos: usize,
) -> GpuTensor {
    let heads_per_group = n_heads / n_kv_heads;
    let out = GpuTensor::uninit(device, &[q_len, n_heads * head_dim]);

    let wg_size = (kv_len as u32).next_power_of_two().min(256).max(1);

    let wgsl = format!(
        r#"// Fused causal GQA attention with separate Q/KV lengths
// One workgroup per (query_pos, head) pair

struct Params {{
    q_len: u32,
    kv_len: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    heads_per_group: u32,
    scale: f32,
    start_pos: u32,
}}

@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const WG: u32 = {wg_size}u;
var<workgroup> wg_max: array<f32, {wg_size}>;
var<workgroup> wg_sum: array<f32, {wg_size}>;

@compute @workgroup_size({wg_size})
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    let query_pos = wg_id.x;
    let head = wg_id.y;
    if (query_pos >= params.q_len || head >= params.n_heads) {{ return; }}
    let tid = lid.x;
    let kv_head = head / params.heads_per_group;

    let q_base = query_pos * params.n_heads * params.head_dim + head * params.head_dim;
    let kv_stride = params.n_kv_heads * params.head_dim;
    let max_attend = params.start_pos + query_pos;

    // Phase 1: compute dot products and find max
    var local_max: f32 = -3.402823e+38;
    for (var kp = tid; kp < params.kv_len; kp = kp + WG) {{
        if (kp <= max_attend) {{
            let k_base = kp * kv_stride + kv_head * params.head_dim;
            var dot: f32 = 0.0;
            for (var d: u32 = 0u; d < params.head_dim; d = d + 1u) {{
                dot = dot + Q[q_base + d] * K[k_base + d];
            }}
            local_max = max(local_max, dot * params.scale);
        }}
    }}
    wg_max[tid] = local_max;
    workgroupBarrier();

    for (var stride: u32 = WG / 2u; stride > 0u; stride = stride / 2u) {{
        if (tid < stride) {{
            wg_max[tid] = max(wg_max[tid], wg_max[tid + stride]);
        }}
        workgroupBarrier();
    }}
    let row_max = wg_max[0];
    workgroupBarrier();

    // Phase 2: compute exp weights and sum
    var local_exp_sum: f32 = 0.0;
    for (var kp = tid; kp < params.kv_len; kp = kp + WG) {{
        if (kp <= max_attend) {{
            let k_base = kp * kv_stride + kv_head * params.head_dim;
            var dot: f32 = 0.0;
            for (var d: u32 = 0u; d < params.head_dim; d = d + 1u) {{
                dot = dot + Q[q_base + d] * K[k_base + d];
            }}
            local_exp_sum = local_exp_sum + exp(dot * params.scale - row_max);
        }}
    }}
    wg_sum[tid] = local_exp_sum;
    workgroupBarrier();

    for (var stride: u32 = WG / 2u; stride > 0u; stride = stride / 2u) {{
        if (tid < stride) {{
            wg_sum[tid] = wg_sum[tid] + wg_sum[tid + stride];
        }}
        workgroupBarrier();
    }}
    let total_sum = wg_sum[0];
    let inv_sum = 1.0 / total_sum;
    workgroupBarrier();

    // Phase 3: compute weighted V sum
    let out_base = query_pos * params.n_heads * params.head_dim + head * params.head_dim;
    for (var d = tid; d < params.head_dim; d = d + WG) {{
        var acc: f32 = 0.0;
        for (var kp: u32 = 0u; kp <= max_attend; kp = kp + 1u) {{
            let v_base = kp * kv_stride + kv_head * params.head_dim;
            let k_base = kp * kv_stride + kv_head * params.head_dim;
            var dot: f32 = 0.0;
            for (var dd: u32 = 0u; dd < params.head_dim; dd = dd + 1u) {{
                dot = dot + Q[q_base + dd] * K[k_base + dd];
            }}
            let w = exp(dot * params.scale - row_max) * inv_sum;
            acc = acc + w * V[v_base + d];
        }}
        output[out_base + d] = acc;
    }}
}}
"#,
        wg_size = wg_size,
    );

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct KvAttnParams {
        q_len: u32,
        kv_len: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        heads_per_group: u32,
        scale: f32,
        start_pos: u32,
    }
    let scale = 1.0 / (head_dim as f32).sqrt();
    let uniform = KvAttnParams {
        q_len: q_len as u32,
        kv_len: kv_len as u32,
        n_heads: n_heads as u32,
        n_kv_heads: n_kv_heads as u32,
        head_dim: head_dim as u32,
        heads_per_group: heads_per_group as u32,
        scale,
        start_pos: start_pos as u32,
    };

    let hash = hash_wgsl(&wgsl);
    let bindings = [
        BindingSpec::Storage { read_only: true },  // Q
        BindingSpec::Storage { read_only: true },  // K
        BindingSpec::Storage { read_only: true },  // V
        BindingSpec::Storage { read_only: false }, // output
        BindingSpec::Uniform,                      // params
    ];
    let cached = cache.get_or_compile_dynamic(device, &wgsl, hash, &bindings);

    use wgpu::util::DeviceExt;
    let params_buf = device
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("kv attn params"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("kv attn bind group"),
        layout: &cached.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: q.buffer.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: k.buffer.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: v.buffer.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: out.buffer.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("kv attn dispatch"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("kv attn compute"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&cached.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(q_len as u32, n_heads as u32, 1);
    }

    cache.submit_or_enqueue(device, encoder.finish());
    out
}

/// Fused SiLU(gate) * up kernel (public version for use outside this module).
///
/// `gate` and `up` must have the same number of elements.
/// Returns `output[i] = silu(gate[i]) * up[i]`.
pub fn swiglu_fused_pub(
    device: &GpuDevice,
    cache: &mut KernelCache,
    gate: &GpuTensor,
    up: &GpuTensor,
) -> GpuTensor {
    swiglu_fused(device, cache, gate, up)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_device() -> GpuDevice {
        GpuDevice::new_sync().expect("GPU device required for tests")
    }

    #[test]
    fn embedding_lookup() {
        let device = get_device();
        let mut cache = KernelCache::new();

        // vocab=3, dim=4. Embeddings:
        // 0 -> [1, 0, 0, 0]
        // 1 -> [0, 1, 0, 0]
        // 2 -> [0, 0, 1, 0]
        let weight = vec![
            1.0, 0.0, 0.0, 0.0, // token 0
            0.0, 1.0, 0.0, 0.0, // token 1
            0.0, 0.0, 1.0, 0.0, // token 2
        ];
        let emb = GpuEmbedding::new(&device, &weight, 3, 4);

        // Look up tokens [2, 0, 1]
        let token_ids: Vec<u32> = vec![2, 0, 1];
        use wgpu::util::DeviceExt;
        let id_buf = GpuBuffer {
            buffer: device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("token ids"),
                    contents: bytemuck::cast_slice(&token_ids),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                }),
            len: token_ids.len(),
        };

        let out = emb.forward(&device, &mut cache, &id_buf, 3);
        let result = out.to_vec_sync(&device);

        assert_eq!(out.shape(), &[3, 4]);
        // Token 2 -> [0, 0, 1, 0]
        assert_eq!(&result[0..4], &[0.0, 0.0, 1.0, 0.0]);
        // Token 0 -> [1, 0, 0, 0]
        assert_eq!(&result[4..8], &[1.0, 0.0, 0.0, 0.0]);
        // Token 1 -> [0, 1, 0, 0]
        assert_eq!(&result[8..12], &[0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn rmsnorm_unit_weight() {
        let device = get_device();
        let mut cache = KernelCache::new();

        let rms = GpuRMSNorm::new(&device, 4, 1e-6);
        let input = GpuTensor::from_slice(&device, &[1.0, 2.0, 3.0, 4.0], &[4]);
        let output = rms.forward(&device, &mut cache, &input);
        let result = output.to_vec_sync(&device);

        // RMS = sqrt(mean([1,4,9,16]) + eps) = sqrt(7.5 + eps)
        // Expected: x / RMS
        let rms_val = (7.5f32 + 1e-6).sqrt();
        for (i, &val) in result.iter().enumerate() {
            let expected = (i + 1) as f32 / rms_val;
            assert!(
                (val - expected).abs() < 1e-4,
                "rmsnorm[{i}] = {val}, expected {expected}"
            );
        }
    }

    #[test]
    fn rmsnorm_batched() {
        let device = get_device();
        let mut cache = KernelCache::new();

        let rms = GpuRMSNorm::new(&device, 3, 1e-6);
        // 2 rows of 3 elements
        let input = GpuTensor::from_slice(&device, &[3.0, 4.0, 0.0, 1.0, 1.0, 1.0], &[2, 3]);
        let output = rms.forward(&device, &mut cache, &input);
        let result = output.to_vec_sync(&device);

        // Row 0: RMS = sqrt((9+16+0)/3 + eps)
        let rms0 = (25.0f32 / 3.0 + 1e-6).sqrt();
        assert!((result[0] - 3.0 / rms0).abs() < 1e-4);
        assert!((result[1] - 4.0 / rms0).abs() < 1e-4);
        assert!(result[2].abs() < 1e-4);

        // Row 1: RMS = sqrt((1+1+1)/3 + eps) = sqrt(1 + eps) ~= 1
        let rms1 = (1.0f32 + 1e-6).sqrt();
        for i in 3..6 {
            assert!(
                (result[i] - 1.0 / rms1).abs() < 1e-4,
                "rmsnorm[{i}] = {}, expected {}",
                result[i],
                1.0 / rms1
            );
        }
    }

    #[test]
    fn swiglu_fused_activation() {
        let device = get_device();
        let mut cache = KernelCache::new();

        // Test the fused SiLU(gate) * up kernel directly
        let gate = GpuTensor::from_slice(&device, &[0.0, 1.0, -1.0, 2.0], &[4]);
        let up = GpuTensor::from_slice(&device, &[1.0, 1.0, 1.0, 1.0], &[4]);
        let out = swiglu_fused(&device, &mut cache, &gate, &up);
        let result = out.to_vec_sync(&device);

        // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
        let expected: Vec<f32> = [0.0, 1.0, -1.0, 2.0]
            .iter()
            .map(|&x| x / (1.0 + (-x as f32).exp()))
            .collect();

        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-4,
                "swiglu[{i}] = {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn swiglu_layer_identity_like() {
        let device = get_device();
        let mut cache = KernelCache::new();

        // With zero weights, output should be zero
        let ffn = GpuSwiGLU::zeros(&device, 4, 8);
        let input = GpuTensor::from_slice(&device, &[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let output = ffn.forward(&device, &mut cache, &input);
        let result = output.to_vec_sync(&device);

        for (i, &val) in result.iter().enumerate() {
            assert!(val.abs() < 1e-6, "swiglu zero output[{i}] = {val}");
        }
    }

    #[test]
    fn rope_identity_at_pos_zero() {
        let device = get_device();
        let mut cache = KernelCache::new();

        // At position 0, all angles are 0, so cos=1, sin=0 -> no rotation
        let rope = GpuRoPE::new(&device, 4, 32, 10000.0);
        let input = GpuTensor::from_slice(
            &device,
            &[1.0, 2.0, 3.0, 4.0], // 1 seq, 1 head, dim=4
            &[1, 1, 4],
        );
        let output = rope.forward(&device, &mut cache, &input, 0);
        let result = output.to_vec_sync(&device);

        // At pos 0: cos=1, sin=0 for all freqs
        // output[0] = x[0]*1 - x[2]*0 = x[0]
        // output[2] = x[0]*0 + x[2]*1 = x[2]
        assert!((result[0] - 1.0).abs() < 1e-5, "rope[0] = {}", result[0]);
        assert!((result[1] - 2.0).abs() < 1e-5, "rope[1] = {}", result[1]);
        assert!((result[2] - 3.0).abs() < 1e-5, "rope[2] = {}", result[2]);
        assert!((result[3] - 4.0).abs() < 1e-5, "rope[3] = {}", result[3]);
    }

    #[test]
    fn rope_rotation_at_pos_nonzero() {
        let device = get_device();
        let mut cache = KernelCache::new();

        let head_dim = 4;
        let base = 10000.0f32;
        let rope = GpuRoPE::new(&device, head_dim, 32, base);

        // Input: [1, 0, 0, 0] at position 5
        let input = GpuTensor::from_slice(
            &device,
            &[1.0, 0.0, 0.0, 0.0],
            &[1, 1, 4],
        );
        let output = rope.forward(&device, &mut cache, &input, 5);
        let result = output.to_vec_sync(&device);

        // At pos 5, pair 0: angle = 5 * 1/10000^(0/4) = 5.0
        let angle0 = 5.0f32 * 1.0 / base.powf(0.0 / 4.0);
        // x[0]=1, x[2]=0: out[0] = 1*cos(a) - 0*sin(a) = cos(a)
        //                   out[2] = 1*sin(a) + 0*cos(a) = sin(a)
        assert!(
            (result[0] - angle0.cos()).abs() < 1e-4,
            "rope pos5[0] = {}, expected {}",
            result[0],
            angle0.cos()
        );
        assert!(
            (result[2] - angle0.sin()).abs() < 1e-4,
            "rope pos5[2] = {}, expected {}",
            result[2],
            angle0.sin()
        );
    }

    #[test]
    fn rope_high_base() {
        let device = get_device();
        let mut cache = KernelCache::new();

        // Test with 500K base (long context)
        let rope = GpuRoPE::new(&device, 4, 64, 500000.0);
        let input = GpuTensor::from_slice(&device, &[1.0, 1.0, 1.0, 1.0], &[1, 1, 4]);
        let output = rope.forward(&device, &mut cache, &input, 10);
        let result = output.to_vec_sync(&device);

        // Just verify it produces valid (non-NaN) output
        for (i, &val) in result.iter().enumerate() {
            assert!(!val.is_nan(), "rope 500K base output[{i}] is NaN");
            assert!(val.is_finite(), "rope 500K base output[{i}] is infinite");
        }
    }

    #[test]
    fn kv_cache_append_and_retrieve() {
        let device = get_device();
        let mut cache = KernelCache::new();

        let mut kv = GpuKVCache::new(&device, 2, 4, 32); // 2 heads, dim 4, max 32

        // Append 2 positions
        let k1 = GpuTensor::from_slice(
            &device,
            &[
                1.0, 0.0, 0.0, 0.0, // head 0
                0.0, 1.0, 0.0, 0.0, // head 1
                0.0, 0.0, 1.0, 0.0, // head 0
                0.0, 0.0, 0.0, 1.0, // head 1
            ],
            &[2, 2, 4],
        );
        let v1 = GpuTensor::from_slice(
            &device,
            &[
                1.0, 1.0, 1.0, 1.0,
                2.0, 2.0, 2.0, 2.0,
                3.0, 3.0, 3.0, 3.0,
                4.0, 4.0, 4.0, 4.0,
            ],
            &[2, 2, 4],
        );

        kv.append(&device, &mut cache, &k1, &v1);
        assert_eq!(kv.len, 2);

        // Append 1 more position
        let k2 = GpuTensor::from_slice(
            &device,
            &[5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0],
            &[1, 2, 4],
        );
        let v2 = GpuTensor::from_slice(
            &device,
            &[7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0],
            &[1, 2, 4],
        );

        kv.append(&device, &mut cache, &k2, &v2);
        assert_eq!(kv.len, 3);

        // Retrieve and verify
        cache.flush(&device);
        let keys = kv.get_keys(&device);
        let key_data = keys.to_vec_sync(&device);
        assert_eq!(keys.shape(), &[3, 2, 4]);
        assert_eq!(&key_data[0..4], &[1.0, 0.0, 0.0, 0.0]); // pos 0, head 0
        assert_eq!(&key_data[16..20], &[5.0, 5.0, 5.0, 5.0]); // pos 2, head 0

        let values = kv.get_values(&device);
        let val_data = values.to_vec_sync(&device);
        assert_eq!(values.shape(), &[3, 2, 4]);
        assert_eq!(&val_data[16..20], &[7.0, 7.0, 7.0, 7.0]); // pos 2, head 0
    }

    #[test]
    fn kv_cache_clear() {
        let device = get_device();
        let mut cache = KernelCache::new();

        let mut kv = GpuKVCache::new(&device, 1, 2, 16);
        let k = GpuTensor::from_slice(&device, &[1.0, 2.0], &[1, 1, 2]);
        let v = GpuTensor::from_slice(&device, &[3.0, 4.0], &[1, 1, 2]);
        kv.append(&device, &mut cache, &k, &v);
        assert_eq!(kv.len, 1);

        kv.clear();
        assert_eq!(kv.len, 0);
    }

    #[test]
    fn causal_attention_zeros() {
        let device = get_device();
        let mut cache = KernelCache::new();

        // With zero weights, output should be zero
        let attn = GpuCausalAttention::zeros(&device, 8, 2, 2, 4);
        let input = GpuTensor::from_slice(
            &device,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
              8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            &[2, 8],
        );
        let output = attn.forward(&device, &mut cache, &input);
        let result = output.to_vec_sync(&device);

        assert_eq!(output.shape(), &[2, 8]);
        for (i, &val) in result.iter().enumerate() {
            assert!(
                val.abs() < 1e-4 || val.is_nan(),
                "causal_attn zeros output[{i}] = {val}"
            );
        }
    }

    #[test]
    fn causal_attention_identity_kv() {
        let device = get_device();
        let mut cache = KernelCache::new();

        // seq_len=1 with identity-like weights to verify the pipeline works
        let dim = 4;
        let n_heads = 1;
        let n_kv_heads = 1;
        let head_dim = 4;

        // Set Wq, Wk, Wv to identity, Wo to identity
        let eye4 = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];

        let attn = GpuCausalAttention {
            wq: GpuTensor::from_slice(&device, &eye4, &[n_heads * head_dim, dim]),
            wk: GpuTensor::from_slice(&device, &eye4, &[n_kv_heads * head_dim, dim]),
            wv: GpuTensor::from_slice(&device, &eye4, &[n_kv_heads * head_dim, dim]),
            wo: GpuTensor::from_slice(&device, &eye4, &[dim, n_heads * head_dim]),
            n_heads,
            n_kv_heads,
            head_dim,
            dim,
        };

        // Single token: self-attention on 1 position just returns V (after softmax weight = 1.0)
        let input = GpuTensor::from_slice(&device, &[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let output = attn.forward(&device, &mut cache, &input);
        let result = output.to_vec_sync(&device);

        assert_eq!(output.shape(), &[1, 4]);
        // With identity projections and single token: output = V = input
        for (i, (&got, &expected)) in result.iter().zip([1.0, 2.0, 3.0, 4.0].iter()).enumerate() {
            assert!(
                (got - expected).abs() < 0.1,
                "causal_attn identity[{i}] = {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn causal_attention_mask_works() {
        let device = get_device();
        let mut cache = KernelCache::new();

        let dim = 4;
        let n_heads = 1;
        let n_kv_heads = 1;
        let head_dim = 4;

        let eye4 = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];

        let attn = GpuCausalAttention {
            wq: GpuTensor::from_slice(&device, &eye4, &[n_heads * head_dim, dim]),
            wk: GpuTensor::from_slice(&device, &eye4, &[n_kv_heads * head_dim, dim]),
            wv: GpuTensor::from_slice(&device, &eye4, &[n_kv_heads * head_dim, dim]),
            wo: GpuTensor::from_slice(&device, &eye4, &[dim, n_heads * head_dim]),
            n_heads,
            n_kv_heads,
            head_dim,
            dim,
        };

        // Two tokens: position 0 can only see itself, position 1 sees both
        let input = GpuTensor::from_slice(
            &device,
            &[
                1.0, 0.0, 0.0, 0.0, // token 0
                0.0, 0.0, 0.0, 1.0, // token 1
            ],
            &[2, 4],
        );
        let output = attn.forward(&device, &mut cache, &input);
        let result = output.to_vec_sync(&device);

        assert_eq!(output.shape(), &[2, 4]);

        // Position 0: only attends to itself -> output ~= V[0] = [1, 0, 0, 0]
        assert!(
            (result[0] - 1.0).abs() < 0.1,
            "pos0[0] = {}, expected ~1.0",
            result[0]
        );

        // Position 1: attends to both -> weighted average of V[0] and V[1]
        // The exact values depend on the attention weights, but it should be
        // some mix of [1,0,0,0] and [0,0,0,1]
        let row1_sum: f32 = result[4..8].iter().sum();
        assert!(
            (row1_sum - 1.0).abs() < 0.2,
            "pos1 sum = {row1_sum}, expected ~1.0"
        );
    }

    #[test]
    fn causal_attention_gqa() {
        let device = get_device();
        let mut cache = KernelCache::new();

        // 4 query heads, 2 KV heads (GQA ratio 2:1)
        let dim = 8;
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 2;

        let attn = GpuCausalAttention::zeros(&device, dim, n_heads, n_kv_heads, head_dim);
        let input = GpuTensor::from_slice(
            &device,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[1, 8],
        );

        // Should not panic -- just verifying GQA dispatch works
        let output = attn.forward(&device, &mut cache, &input);
        assert_eq!(output.shape(), &[1, 8]);
    }
}
