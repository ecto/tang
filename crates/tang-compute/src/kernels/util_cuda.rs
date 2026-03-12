//! Utility CUDA kernels: column extraction, type conversion, fused RMSNorm+residual.

/// Extract a contiguous column range from a row-major matrix.
/// input: [batch, total_cols], output: [batch, col_count]
/// Extracts columns [col_start, col_start + col_count) from each row.
/// Grid: (ceil(batch * col_count / 256)), Block: (256)
pub const EXTRACT_COLUMNS_CUDA: &str = r#"
extern "C" __global__ void extract_columns(
    const float* __restrict__ input,
    float* __restrict__ output,
    const unsigned int batch,
    const unsigned int total_cols,
    const unsigned int col_start,
    const unsigned int col_count)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = batch * col_count;
    if (idx >= total) return;
    unsigned int row = idx / col_count;
    unsigned int col = idx % col_count;
    output[idx] = input[row * total_cols + col_start + col];
}
"#;

/// bf16 variant of extract_columns.
/// Grid: (ceil(batch * col_count / 256)), Block: (256)
pub const EXTRACT_COLUMNS_BF16_CUDA: &str = r#"
extern "C" __global__ void extract_columns_bf16(
    const unsigned short* __restrict__ input,
    unsigned short* __restrict__ output,
    const unsigned int batch,
    const unsigned int total_cols,
    const unsigned int col_start,
    const unsigned int col_count)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = batch * col_count;
    if (idx >= total) return;
    unsigned int row = idx / col_count;
    unsigned int col = idx % col_count;
    output[idx] = input[row * total_cols + col_start + col];
}
"#;

/// Convert bf16 buffer to f32 on GPU (no CPU round-trip).
/// Grid: (ceil(n / 256)), Block: (256)
pub const BF16_TO_F32_CUDA: &str = r#"
extern "C" __global__ void bf16_to_f32(
    const unsigned short* __restrict__ input,
    float* __restrict__ output,
    const unsigned int n)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = __int_as_float(((unsigned int)input[idx]) << 16);
}
"#;

/// Convert f32 buffer to bf16 on GPU (no CPU round-trip).
/// Grid: (ceil(n / 256)), Block: (256)
pub const F32_TO_BF16_CUDA: &str = r#"
extern "C" __global__ void f32_to_bf16(
    const float* __restrict__ input,
    unsigned short* __restrict__ output,
    const unsigned int n)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    unsigned int bits = __float_as_int(input[idx]);
    unsigned int lsb = (bits >> 16) & 1;
    bits += 0x7FFF + lsb;
    output[idx] = (unsigned short)(bits >> 16);
}
"#;

/// Broadcast bias addition: out[i] = matrix[i] + bias[i % dim].
/// Grid: (ceil(numel / 256)), Block: (256)
pub const BIAS_ADD_CUDA: &str = r#"
extern "C" __global__ void bias_add(
    const float* __restrict__ matrix,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const unsigned int numel,
    const unsigned int dim)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    output[idx] = matrix[idx] + bias[idx % dim];
}
"#;

/// Broadcast bias addition for bf16: out[i] = matrix[i] + bias[i % dim], all bf16.
/// Reads bf16, computes in f32, writes bf16.
/// Grid: (ceil(numel / 256)), Block: (256)
pub const BIAS_ADD_BF16_CUDA: &str = r#"
extern "C" __global__ void bias_add_bf16(
    const unsigned short* __restrict__ matrix,
    const unsigned short* __restrict__ bias,
    unsigned short* __restrict__ output,
    const unsigned int numel,
    const unsigned int dim)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    float m = __int_as_float(((unsigned int)matrix[idx]) << 16);
    float b = __int_as_float(((unsigned int)bias[idx % dim]) << 16);
    float result = m + b;
    unsigned int bits = __float_as_int(result);
    output[idx] = (unsigned short)((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}
"#;

/// Fused bf16 residual add + RMS normalization.
/// output = rms_norm(input + residual, weight, eps), all bf16 I/O.
/// Also writes the pre-norm sum to `sum_out` for backward.
/// Grid: (n_groups), Block: (min(dim, 256))
pub const RMS_NORM_RESIDUAL_BF16_CUDA: &str = r#"
extern "C" __global__ void rms_norm_residual_bf16(
    const unsigned short* __restrict__ input,
    const unsigned short* __restrict__ residual,
    const unsigned short* __restrict__ weight,
    unsigned short* __restrict__ output,
    unsigned short* __restrict__ sum_out,
    const unsigned int n_groups,
    const unsigned int dim,
    const float eps)
{
    __shared__ float shared[256];

    unsigned int group = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int tg_size = blockDim.x;
    if (group >= n_groups) return;

    unsigned int base = group * dim;

    // Phase 1: compute sum and sum of squares, store pre-norm sum as bf16
    float local_sq = 0.0f;
    for (unsigned int i = tid; i < dim; i += tg_size) {
        float vi = __int_as_float(((unsigned int)input[base + i]) << 16);
        float vr = __int_as_float(((unsigned int)residual[base + i]) << 16);
        float v = vi + vr;
        unsigned int sbits = __float_as_int(v);
        sum_out[base + i] = (unsigned short)((sbits + 0x7FFF + ((sbits >> 16) & 1)) >> 16);
        local_sq += v * v;
    }
    shared[tid] = local_sq;
    __syncthreads();

    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    float inv_rms = rsqrtf(shared[0] / float(dim) + eps);

    // Phase 2: normalize — recompute sum from inputs (avoids bf16 truncation loss)
    for (unsigned int i = tid; i < dim; i += tg_size) {
        float vi = __int_as_float(((unsigned int)input[base + i]) << 16);
        float vr = __int_as_float(((unsigned int)residual[base + i]) << 16);
        float w = __int_as_float(((unsigned int)weight[i]) << 16);
        float result = (vi + vr) * inv_rms * w;
        unsigned int rbits = __float_as_int(result);
        output[base + i] = (unsigned short)((rbits + 0x7FFF + ((rbits >> 16) & 1)) >> 16);
    }
}
"#;

/// Fused residual add + RMS normalization.
/// output = rms_norm(input + residual, weight, eps)
/// Also writes the pre-norm sum (input + residual) to `sum_out` for backward.
/// Grid: (n_groups), Block: (min(dim, 256))
pub const RMS_NORM_RESIDUAL_CUDA: &str = r#"
extern "C" __global__ void rms_norm_residual(
    const float* __restrict__ input,
    const float* __restrict__ residual,
    const float* __restrict__ weight,
    float* __restrict__ output,
    float* __restrict__ sum_out,
    const unsigned int n_groups,
    const unsigned int dim,
    const float eps)
{
    __shared__ float shared[256];

    unsigned int group = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int tg_size = blockDim.x;
    if (group >= n_groups) return;

    unsigned int base = group * dim;

    // Phase 1: compute sum and sum of squares
    float local_sq = 0.0f;
    for (unsigned int i = tid; i < dim; i += tg_size) {
        float v = input[base + i] + residual[base + i];
        sum_out[base + i] = v;
        local_sq += v * v;
    }
    shared[tid] = local_sq;
    __syncthreads();

    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    float inv_rms = rsqrtf(shared[0] / float(dim) + eps);

    // Phase 2: normalize
    for (unsigned int i = tid; i < dim; i += tg_size) {
        output[base + i] = sum_out[base + i] * inv_rms * weight[i];
    }
}
"#;

// ---------- RoPE kernels ----------
// Interleaved rotary position embedding: rotates pairs (2i, 2i+1).
// Each thread handles one (position, head, pair) element.

/// RoPE forward f32: out[base+2i] = x0*cos - x1*sin, out[base+2i+1] = x0*sin + x1*cos.
/// Grid: (ceil(seq_len * n_heads * half_dim / 256)), Block: (256)
pub const ROPE_FORWARD_CUDA: &str = r#"
extern "C" __global__ void rope_forward(
    const float* __restrict__ input,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    float* __restrict__ output,
    const unsigned int seq_len,
    const unsigned int n_heads,
    const unsigned int head_dim,
    const unsigned int half_dim,
    const unsigned int start_pos)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = seq_len * n_heads * half_dim;
    if (idx >= total) return;

    unsigned int i = idx % half_dim;
    unsigned int remaining = idx / half_dim;
    unsigned int h = remaining % n_heads;
    unsigned int s = remaining / n_heads;
    unsigned int pos = start_pos + s;

    float cos_val = cos_table[pos * half_dim + i];
    float sin_val = sin_table[pos * half_dim + i];

    unsigned int base = (s * n_heads + h) * head_dim;
    float x0 = input[base + 2*i];
    float x1 = input[base + 2*i + 1];
    output[base + 2*i] = x0 * cos_val - x1 * sin_val;
    output[base + 2*i + 1] = x0 * sin_val + x1 * cos_val;
}
"#;

/// RoPE backward f32: reverse rotation (transpose of rotation matrix).
/// Grid: (ceil(seq_len * n_heads * half_dim / 256)), Block: (256)
pub const ROPE_BACKWARD_CUDA: &str = r#"
extern "C" __global__ void rope_backward(
    const float* __restrict__ grad_output,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    float* __restrict__ grad_input,
    const unsigned int seq_len,
    const unsigned int n_heads,
    const unsigned int head_dim,
    const unsigned int half_dim,
    const unsigned int start_pos)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = seq_len * n_heads * half_dim;
    if (idx >= total) return;

    unsigned int i = idx % half_dim;
    unsigned int remaining = idx / half_dim;
    unsigned int h = remaining % n_heads;
    unsigned int s = remaining / n_heads;
    unsigned int pos = start_pos + s;

    float cos_val = cos_table[pos * half_dim + i];
    float sin_val = sin_table[pos * half_dim + i];

    unsigned int base = (s * n_heads + h) * head_dim;
    float g0 = grad_output[base + 2*i];
    float g1 = grad_output[base + 2*i + 1];
    grad_input[base + 2*i] = g0 * cos_val + g1 * sin_val;
    grad_input[base + 2*i + 1] = -g0 * sin_val + g1 * cos_val;
}
"#;

/// RoPE forward bf16: bf16 I/O, f32 compute.
/// Grid: (ceil(seq_len * n_heads * half_dim / 256)), Block: (256)
pub const ROPE_FORWARD_BF16_CUDA: &str = r#"
extern "C" __global__ void rope_forward_bf16(
    const unsigned short* __restrict__ input,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    unsigned short* __restrict__ output,
    const unsigned int seq_len,
    const unsigned int n_heads,
    const unsigned int head_dim,
    const unsigned int half_dim,
    const unsigned int start_pos)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = seq_len * n_heads * half_dim;
    if (idx >= total) return;

    unsigned int i = idx % half_dim;
    unsigned int remaining = idx / half_dim;
    unsigned int h = remaining % n_heads;
    unsigned int s = remaining / n_heads;
    unsigned int pos = start_pos + s;

    float cos_val = cos_table[pos * half_dim + i];
    float sin_val = sin_table[pos * half_dim + i];

    unsigned int base = (s * n_heads + h) * head_dim;
    float x0 = __int_as_float(((unsigned int)input[base + 2*i]) << 16);
    float x1 = __int_as_float(((unsigned int)input[base + 2*i + 1]) << 16);
    float r0 = x0 * cos_val - x1 * sin_val;
    float r1 = x0 * sin_val + x1 * cos_val;
    unsigned int b0 = __float_as_int(r0);
    unsigned int b1 = __float_as_int(r1);
    output[base + 2*i] = (unsigned short)((b0 + 0x7FFF + ((b0 >> 16) & 1)) >> 16);
    output[base + 2*i + 1] = (unsigned short)((b1 + 0x7FFF + ((b1 >> 16) & 1)) >> 16);
}
"#;

/// RoPE backward bf16: bf16 I/O, f32 compute.
/// Grid: (ceil(seq_len * n_heads * half_dim / 256)), Block: (256)
pub const ROPE_BACKWARD_BF16_CUDA: &str = r#"
extern "C" __global__ void rope_backward_bf16(
    const unsigned short* __restrict__ grad_output,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    unsigned short* __restrict__ grad_input,
    const unsigned int seq_len,
    const unsigned int n_heads,
    const unsigned int head_dim,
    const unsigned int half_dim,
    const unsigned int start_pos)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = seq_len * n_heads * half_dim;
    if (idx >= total) return;

    unsigned int i = idx % half_dim;
    unsigned int remaining = idx / half_dim;
    unsigned int h = remaining % n_heads;
    unsigned int s = remaining / n_heads;
    unsigned int pos = start_pos + s;

    float cos_val = cos_table[pos * half_dim + i];
    float sin_val = sin_table[pos * half_dim + i];

    unsigned int base = (s * n_heads + h) * head_dim;
    float g0 = __int_as_float(((unsigned int)grad_output[base + 2*i]) << 16);
    float g1 = __int_as_float(((unsigned int)grad_output[base + 2*i + 1]) << 16);
    float r0 = g0 * cos_val + g1 * sin_val;
    float r1 = -g0 * sin_val + g1 * cos_val;
    unsigned int b0 = __float_as_int(r0);
    unsigned int b1 = __float_as_int(r1);
    grad_input[base + 2*i] = (unsigned short)((b0 + 0x7FFF + ((b0 >> 16) & 1)) >> 16);
    grad_input[base + 2*i + 1] = (unsigned short)((b1 + 0x7FFF + ((b1 >> 16) & 1)) >> 16);
}
"#;

// ---------- Fused bf16 elementwise kernels ----------
// Read bf16, compute in f32, write bf16 in a single kernel.
// Eliminates separate bf16→f32 and f32→bf16 conversion passes.

/// Fused bf16 add: out[i] = a[i] + b[i], all bf16.
/// Grid: (ceil(n / 256)), Block: (256)
pub const ADD_TENSORS_BF16_CUDA: &str = r#"
extern "C" __global__ void add_tensors_bf16(
    const unsigned short* __restrict__ a,
    const unsigned short* __restrict__ b,
    unsigned short* __restrict__ out,
    const unsigned int n)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float va = __int_as_float(((unsigned int)a[idx]) << 16);
    float vb = __int_as_float(((unsigned int)b[idx]) << 16);
    float sum = va + vb;
    unsigned int bits = __float_as_int(sum);
    out[idx] = (unsigned short)((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}
"#;

/// Fused bf16 SwiGLU: out[i] = silu(gate[i]) * up[i], all bf16.
/// silu(x) = x / (1 + exp(-x))
/// Grid: (ceil(n / 256)), Block: (256)
pub const SWIGLU_FUSED_BF16_CUDA: &str = r#"
extern "C" __global__ void swiglu_fused_bf16(
    const unsigned short* __restrict__ gate,
    const unsigned short* __restrict__ up,
    unsigned short* __restrict__ out,
    const unsigned int n)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float g = __int_as_float(((unsigned int)gate[idx]) << 16);
    float u = __int_as_float(((unsigned int)up[idx]) << 16);
    float sig = 1.0f / (1.0f + expf(-g));
    float result = g * sig * u;
    unsigned int bits = __float_as_int(result);
    out[idx] = (unsigned short)((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}
"#;

/// Fused bf16 SwiGLU backward: computes grad_gate and grad_up.
/// grad_up = grad * silu(gate)
/// grad_gate = grad * up * dsilu(gate), dsilu(x) = sigmoid(x)*(1+x*(1-sigmoid(x)))
/// Grid: (ceil(n / 256)), Block: (256)
pub const SWIGLU_BACKWARD_BF16_CUDA: &str = r#"
extern "C" __global__ void swiglu_backward_bf16(
    const unsigned short* __restrict__ grad,
    const unsigned short* __restrict__ gate,
    const unsigned short* __restrict__ up,
    unsigned short* __restrict__ grad_gate,
    unsigned short* __restrict__ grad_up,
    const unsigned int n)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float g_val = __int_as_float(((unsigned int)grad[idx]) << 16);
    float gate_val = __int_as_float(((unsigned int)gate[idx]) << 16);
    float up_val = __int_as_float(((unsigned int)up[idx]) << 16);
    float sig = 1.0f / (1.0f + expf(-gate_val));
    float silu = gate_val * sig;
    // grad_up = grad * silu(gate)
    unsigned int gu_bits = __float_as_int(g_val * silu);
    grad_up[idx] = (unsigned short)((gu_bits + 0x7FFF + ((gu_bits >> 16) & 1)) >> 16);
    // dsilu = sigmoid * (1 + gate * (1 - sigmoid))
    float dsilu = sig * (1.0f + gate_val * (1.0f - sig));
    unsigned int gg_bits = __float_as_int(g_val * up_val * dsilu);
    grad_gate[idx] = (unsigned short)((gg_bits + 0x7FFF + ((gg_bits >> 16) & 1)) >> 16);
}
"#;
