//! Neural network layers: attention, layer norm, softmax, activations.

use crate::device::GpuDevice;
use crate::kernel::KernelCache;
use crate::matmul::matmul;
use crate::realize::map_elementwise;
use crate::tensor::GpuTensor;

/// Layer normalization.
pub struct GpuLayerNorm {
    /// Scale parameter [dim].
    pub weight: GpuTensor,
    /// Bias parameter [dim].
    pub bias: GpuTensor,
    pub eps: f32,
    pub dim: usize,
}

impl GpuLayerNorm {
    /// Create layer norm with ones for weight and zeros for bias.
    pub fn new(device: &GpuDevice, dim: usize, eps: f32) -> Self {
        let ones = vec![1.0f32; dim];
        let zeros = vec![0.0f32; dim];
        Self {
            weight: GpuTensor::from_slice(device, &ones, &[dim]),
            bias: GpuTensor::from_slice(device, &zeros, &[dim]),
            eps,
            dim,
        }
    }

    /// Apply layer normalization on GPU.
    ///
    /// Uses a single WGSL kernel: one workgroup per group (row),
    /// shared memory reductions for mean and variance.
    pub fn forward(&self, device: &GpuDevice, cache: &mut KernelCache, input: &GpuTensor) -> GpuTensor {
        let n_groups = input.numel() / self.dim;
        let out = GpuTensor::uninit(device, input.shape());

        let wg_size = (self.dim as u32).next_power_of_two().min(256);

        let wgsl = format!(
            r#"// LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
// One workgroup per normalization group

struct Params {{
    n_groups: u32,
    dim: u32,
    eps: f32,
    _pad: u32,
}}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

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

    // Phase 1: Compute mean
    var local_sum: f32 = 0.0;
    for (var i = tid; i < params.dim; i = i + WG) {{
        local_sum = local_sum + input[base + i];
    }}
    wg_buf[tid] = local_sum;
    workgroupBarrier();

    for (var stride: u32 = WG / 2u; stride > 0u; stride = stride / 2u) {{
        if (tid < stride) {{
            wg_buf[tid] = wg_buf[tid] + wg_buf[tid + stride];
        }}
        workgroupBarrier();
    }}
    let mean = wg_buf[0] / f32(params.dim);
    workgroupBarrier();

    // Phase 2: Compute variance
    var local_var: f32 = 0.0;
    for (var i = tid; i < params.dim; i = i + WG) {{
        let diff = input[base + i] - mean;
        local_var = local_var + diff * diff;
    }}
    wg_buf[tid] = local_var;
    workgroupBarrier();

    for (var stride: u32 = WG / 2u; stride > 0u; stride = stride / 2u) {{
        if (tid < stride) {{
            wg_buf[tid] = wg_buf[tid] + wg_buf[tid + stride];
        }}
        workgroupBarrier();
    }}
    let variance = wg_buf[0] / f32(params.dim);
    let inv_std = 1.0 / sqrt(variance + params.eps);
    workgroupBarrier();

    // Phase 3: Normalize and apply weight/bias
    for (var i = tid; i < params.dim; i = i + WG) {{
        output[base + i] = (input[base + i] - mean) * inv_std * weight[i] + bias[i];
    }}
}}
"#,
            wg_size = wg_size,
        );

        // Custom 5-binding layout: input, weight, bias, output, params
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct LnParams {
            n_groups: u32,
            dim: u32,
            eps: f32,
            _pad: u32,
        }
        let uniform = LnParams {
            n_groups: n_groups as u32,
            dim: self.dim as u32,
            eps: self.eps,
            _pad: 0,
        };

        let hash = {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            wgsl.hash(&mut hasher);
            hasher.finish()
        };

        let cached = cache.get_or_compile_5bind(device, &wgsl, hash);

        use wgpu::util::DeviceExt;
        let params_buf = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("layernorm params"),
                contents: bytemuck::bytes_of(&uniform),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("layernorm bind group"),
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
                    resource: self.bias.buffer.buffer.as_entire_binding(),
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
                label: Some("layernorm dispatch"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("layernorm compute"),
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

/// Softmax along the last dimension, entirely on GPU.
///
/// Uses a single WGSL kernel with workgroup-level reductions:
/// one workgroup per row, shared memory for max and sum.
pub fn softmax(device: &GpuDevice, cache: &mut KernelCache, input: &GpuTensor) -> GpuTensor {
    let shape = input.shape();
    let last_dim = *shape.last().unwrap();
    let n_rows = input.numel() / last_dim;
    let out = GpuTensor::uninit(device, shape);

    // Workgroup size: min(256, last_dim rounded up to power of 2)
    let wg_size = (last_dim as u32).next_power_of_two().min(256);

    let wgsl = format!(
        r#"// Softmax along last dim: one workgroup per row
// Numerically stable: subtract max, then exp/sum

struct Params {{
    n_rows: u32,
    row_len: u32,
    _pad2: u32,
    _pad3: u32,
}}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const WG: u32 = {wg_size}u;
var<workgroup> wg_buf: array<f32, {wg_size}>;

@compute @workgroup_size({wg_size})
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    let row = wg_id.x;
    if (row >= params.n_rows) {{ return; }}
    let tid = lid.x;
    let base = row * params.row_len;

    // Phase 1: Find max (parallel reduction)
    var local_max: f32 = -3.402823e+38;
    for (var i = tid; i < params.row_len; i = i + WG) {{
        local_max = max(local_max, input[base + i]);
    }}
    wg_buf[tid] = local_max;
    workgroupBarrier();

    // Tree reduction for max
    for (var stride: u32 = WG / 2u; stride > 0u; stride = stride / 2u) {{
        if (tid < stride) {{
            wg_buf[tid] = max(wg_buf[tid], wg_buf[tid + stride]);
        }}
        workgroupBarrier();
    }}
    let row_max = wg_buf[0];
    workgroupBarrier();

    // Phase 2: Compute exp(x - max) and sum
    var local_sum: f32 = 0.0;
    for (var i = tid; i < params.row_len; i = i + WG) {{
        let val = exp(input[base + i] - row_max);
        output[base + i] = val;
        local_sum = local_sum + val;
    }}
    wg_buf[tid] = local_sum;
    workgroupBarrier();

    // Tree reduction for sum
    for (var stride: u32 = WG / 2u; stride > 0u; stride = stride / 2u) {{
        if (tid < stride) {{
            wg_buf[tid] = wg_buf[tid] + wg_buf[tid + stride];
        }}
        workgroupBarrier();
    }}
    let row_sum = wg_buf[0];
    workgroupBarrier();

    // Phase 3: Normalize
    let inv_sum = 1.0 / row_sum;
    for (var i = tid; i < params.row_len; i = i + WG) {{
        output[base + i] = output[base + i] * inv_sum;
    }}
}}
"#,
        wg_size = wg_size,
    );

    let params: [u32; 4] = [n_rows as u32, last_dim as u32, 0, 0];

    // Need custom dispatch: n_rows workgroups, not count/256
    let hash = {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        wgsl.hash(&mut hasher);
        hasher.finish()
    };

    // Compile pipeline with custom layout (input read, output rw, uniform)
    let cached = cache.get_or_compile_custom(device, &wgsl, hash);

    use wgpu::util::DeviceExt;
    let params_buf = device
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("softmax params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("softmax bind group"),
        layout: &cached.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.buffer.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out.buffer.buffer.as_entire_binding(),
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
            label: Some("softmax dispatch"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("softmax compute"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&cached.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // One workgroup per row
        pass.dispatch_workgroups(n_rows as u32, 1, 1);
    }

    cache.submit_or_enqueue(device, encoder.finish());
    out
}

/// GELU activation via tang-expr fused kernel.
pub fn gelu(device: &GpuDevice, cache: &mut KernelCache, input: &GpuTensor) -> GpuTensor {
    // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    map_elementwise(device, cache, &[input], |args| {
        use tang::Scalar;
        let x = args[0];
        let half = ExprId::from_f64(0.5);
        let one = ExprId::from_f64(1.0);
        let coeff = ExprId::from_f64(0.044715);
        let sqrt_2_over_pi = ExprId::from_f64(0.7978845608028654); // sqrt(2/π)

        let x3 = x * x * x;
        let inner = sqrt_2_over_pi * (x + coeff * x3);
        half * x * (one + inner.tanh())
    })
}

use tang_expr::ExprId;

/// Add bias [cols] to each row of a matrix [rows, cols] on GPU.
pub fn bias_add(
    device: &GpuDevice,
    cache: &mut KernelCache,
    matrix: &GpuTensor,
    bias: &GpuTensor,
) -> GpuTensor {
    assert_eq!(matrix.ndim(), 2, "bias_add: matrix must be 2D");
    let rows = matrix.shape()[0];
    let cols = matrix.shape()[1];
    assert_eq!(bias.numel(), cols, "bias_add: bias length must match cols");

    let numel = (rows * cols) as u32;
    let out = GpuTensor::uninit(device, matrix.shape());

    let wgsl = r#"// Bias add: output[i] = matrix[i] + bias[i % cols]

struct Params {
    count: u32,
    cols: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> matrix: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) { return; }
    let j = idx % params.cols;
    output[idx] = matrix[idx] + bias[j];
}
"#;

    cache.dispatch_rr_w(device, wgsl, &matrix.buffer, &bias.buffer, &out.buffer, &[numel, cols as u32, 0, 0]);
    out
}

/// ReLU activation via tang-expr fused kernel.
pub fn relu(device: &GpuDevice, cache: &mut KernelCache, input: &GpuTensor) -> GpuTensor {
    map_elementwise(device, cache, &[input], |args| {
        use tang::Scalar;
        let x = args[0];
        let zero = ExprId::from_f64(0.0);
        x.max(zero)
    })
}

/// ReLU backward: grad_input = grad_output * (input > 0), on GPU via WGSL select().
pub fn relu_backward(
    device: &GpuDevice,
    cache: &mut KernelCache,
    input: &GpuTensor,
    grad_output: &GpuTensor,
) -> GpuTensor {
    assert_eq!(input.numel(), grad_output.numel());
    let numel = input.numel() as u32;
    let out = GpuTensor::uninit(device, input.shape());

    let wgsl = r#"// ReLU backward: output = select(0.0, grad, input > 0.0)

struct Params {
    count: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) { return; }
    output[idx] = select(0.0, grad[idx], input[idx] > 0.0);
}
"#;

    cache.dispatch_rr_w(device, wgsl, &input.buffer, &grad_output.buffer, &out.buffer, &[numel, 0, 0, 0]);
    out
}

/// Multi-head attention.
pub struct GpuAttention {
    /// Query projection [dim, dim].
    pub wq: GpuTensor,
    /// Key projection [dim, dim].
    pub wk: GpuTensor,
    /// Value projection [dim, dim].
    pub wv: GpuTensor,
    /// Output projection [dim, dim].
    pub wo: GpuTensor,
    pub n_heads: usize,
    pub dim: usize,
    pub head_dim: usize,
}

impl GpuAttention {
    /// Create attention with zero-initialized projections.
    pub fn zeros(device: &GpuDevice, dim: usize, n_heads: usize) -> Self {
        assert_eq!(dim % n_heads, 0);
        let head_dim = dim / n_heads;
        let z = vec![0.0f32; dim * dim];
        Self {
            wq: GpuTensor::from_slice(device, &z, &[dim, dim]),
            wk: GpuTensor::from_slice(device, &z, &[dim, dim]),
            wv: GpuTensor::from_slice(device, &z, &[dim, dim]),
            wo: GpuTensor::from_slice(device, &z, &[dim, dim]),
            n_heads,
            dim,
            head_dim,
        }
    }

    /// Forward pass: scaled dot-product attention, entirely on GPU.
    ///
    /// input: [seq_len, dim] → output: [seq_len, dim]
    pub fn forward(
        &self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        input: &GpuTensor,
    ) -> GpuTensor {
        // Q = input @ Wq^T, K = input @ Wk^T, V = input @ Wv^T
        let q = matmul(device, cache, input, &self.wq);
        let k = matmul(device, cache, input, &self.wk);
        let v = matmul(device, cache, input, &self.wv);

        // Transpose K on GPU: [seq_len, dim] -> [dim, seq_len]
        let kt = k.transpose_gpu(device, cache);

        // scores = Q @ K^T
        let scores = matmul(device, cache, &q, &kt);

        // Scale on GPU: scores / sqrt(head_dim)
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let scores_scaled = scores.scale(device, cache, scale);

        // Softmax on GPU
        let attn = softmax(device, cache, &scores_scaled);

        // attn @ V
        let attn_out = matmul(device, cache, &attn, &v);

        // Output projection
        matmul(device, cache, &attn_out, &self.wo)
    }
}

/// Transformer block: attention + FFN with residual connections.
pub struct GpuTransformerBlock {
    pub attn: GpuAttention,
    pub ffn_up: GpuTensor,   // [hidden_dim, dim]
    pub ffn_down: GpuTensor, // [dim, hidden_dim]
    pub ln1: GpuLayerNorm,
    pub ln2: GpuLayerNorm,
    pub dim: usize,
    pub hidden_dim: usize,
}

impl GpuTransformerBlock {
    /// Create a transformer block with zero-initialized weights.
    pub fn zeros(device: &GpuDevice, dim: usize, hidden_dim: usize, n_heads: usize) -> Self {
        Self {
            attn: GpuAttention::zeros(device, dim, n_heads),
            ffn_up: GpuTensor::from_slice(
                device,
                &vec![0.0f32; hidden_dim * dim],
                &[hidden_dim, dim],
            ),
            ffn_down: GpuTensor::from_slice(
                device,
                &vec![0.0f32; dim * hidden_dim],
                &[dim, hidden_dim],
            ),
            ln1: GpuLayerNorm::new(device, dim, 1e-5),
            ln2: GpuLayerNorm::new(device, dim, 1e-5),
            dim,
            hidden_dim,
        }
    }

    /// Forward pass.
    pub fn forward(
        &self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        input: &GpuTensor,
    ) -> GpuTensor {
        // Pre-norm architecture
        let normed = self.ln1.forward(device, cache, input);
        let attn_out = self.attn.forward(device, cache, &normed);

        // Residual connection: x + attn(ln1(x))
        let residual1 = add_tensors(device, cache, input, &attn_out);

        // FFN
        let normed2 = self.ln2.forward(device, cache, &residual1);
        let hidden = matmul(device, cache, &normed2, &self.ffn_up);
        let activated = gelu(device, cache, &hidden);
        let ffn_out = matmul(device, cache, &activated, &self.ffn_down);

        // Residual connection
        add_tensors(device, cache, &residual1, &ffn_out)
    }
}

/// Element-wise tensor addition on GPU.
pub fn add_tensors(device: &GpuDevice, cache: &mut KernelCache, a: &GpuTensor, b: &GpuTensor) -> GpuTensor {
    assert_eq!(a.numel(), b.numel(), "add_tensors: shape mismatch");
    let numel = a.numel() as u32;
    let out = GpuTensor::uninit(device, a.shape());

    let wgsl = r#"// Element-wise add: output[i] = a[i] + b[i]

struct Params {
    count: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) { return; }
    output[idx] = a[idx] + b[idx];
}
"#;

    cache.dispatch_rr_w(device, wgsl, &a.buffer, &b.buffer, &out.buffer, &[numel, 0, 0, 0]);
    out
}
