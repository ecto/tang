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
