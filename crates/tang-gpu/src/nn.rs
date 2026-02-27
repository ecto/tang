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

    /// Apply layer normalization.
    pub fn forward(&self, device: &GpuDevice, input: &GpuTensor) -> GpuTensor {
        // LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
        let data = input.buffer.to_vec_sync(device);
        let w = self.weight.buffer.to_vec_sync(device);
        let b = self.bias.buffer.to_vec_sync(device);

        let numel = data.len();
        let n_groups = numel / self.dim;
        let mut out = vec![0.0f32; numel];

        for g in 0..n_groups {
            let offset = g * self.dim;
            let group = &data[offset..offset + self.dim];

            let mean: f32 = group.iter().sum::<f32>() / self.dim as f32;
            let var: f32 =
                group.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / self.dim as f32;
            let inv_std = 1.0 / (var + self.eps).sqrt();

            for i in 0..self.dim {
                out[offset + i] = (group[i] - mean) * inv_std * w[i] + b[i];
            }
        }

        GpuTensor::from_slice(device, &out, input.shape())
    }
}

/// Softmax along the last dimension (on CPU, uploaded to GPU).
pub fn softmax(device: &GpuDevice, input: &GpuTensor) -> GpuTensor {
    let data = input.buffer.to_vec_sync(device);
    let shape = input.shape();
    let last_dim = *shape.last().unwrap();
    let n_groups = data.len() / last_dim;
    let mut out = vec![0.0f32; data.len()];

    for g in 0..n_groups {
        let offset = g * last_dim;
        let group = &data[offset..offset + last_dim];

        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = group.iter().map(|&x| (x - max_val).exp()).sum();

        for i in 0..last_dim {
            out[offset + i] = (group[i] - max_val).exp() / exp_sum;
        }
    }

    GpuTensor::from_slice(device, &out, shape)
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

    /// Forward pass for single-token attention (seq_len=1 simplification).
    pub fn forward(
        &self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        input: &GpuTensor,
    ) -> GpuTensor {
        // input: [seq_len, dim]
        // Q = input @ Wq^T, K = input @ Wk^T, V = input @ Wv^T
        let q = matmul(device, cache, input, &self.wq);
        let k = matmul(device, cache, input, &self.wk);
        let v = matmul(device, cache, input, &self.wv);

        // Scaled dot-product attention (single head, simplified)
        // scores = Q @ K^T / sqrt(head_dim)
        let seq_len = input.shape()[0];

        // Transpose K
        let k_data = k.buffer.to_vec_sync(device);
        let mut kt_data = vec![0.0f32; k_data.len()];
        for i in 0..seq_len {
            for j in 0..self.dim {
                kt_data[j * seq_len + i] = k_data[i * self.dim + j];
            }
        }
        let kt = GpuTensor::from_slice(device, &kt_data, &[self.dim, seq_len]);

        let scores = matmul(device, cache, &q, &kt);
        let scale = 1.0 / (self.head_dim as f32).sqrt();

        // Scale scores
        let scores_data = scores.buffer.to_vec_sync(device);
        let scaled: Vec<f32> = scores_data.iter().map(|&s| s * scale).collect();
        let scores_scaled = GpuTensor::from_slice(device, &scaled, &[seq_len, seq_len]);

        // Softmax
        let attn = softmax(device, &scores_scaled);

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
        let normed = self.ln1.forward(device, input);
        let attn_out = self.attn.forward(device, cache, &normed);

        // Residual connection: x + attn(ln1(x))
        let residual1 = add_tensors(device, input, &attn_out);

        // FFN
        let normed2 = self.ln2.forward(device, &residual1);
        let hidden = matmul(device, cache, &normed2, &self.ffn_up);
        let activated = gelu(device, cache, &hidden);
        let ffn_out = matmul(device, cache, &activated, &self.ffn_down);

        // Residual connection
        add_tensors(device, &residual1, &ffn_out)
    }
}

/// Element-wise tensor addition (on CPU for now).
pub fn add_tensors(device: &GpuDevice, a: &GpuTensor, b: &GpuTensor) -> GpuTensor {
    let a_data = a.buffer.to_vec_sync(device);
    let b_data = b.buffer.to_vec_sync(device);
    assert_eq!(a_data.len(), b_data.len());
    let out: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| x + y)
        .collect();
    GpuTensor::from_slice(device, &out, a.shape())
}
