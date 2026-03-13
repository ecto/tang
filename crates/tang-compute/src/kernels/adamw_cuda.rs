//! CUDA AdamW optimizer kernel.

/// AdamW step: updates param, m, v in-place.
/// Grid: (ceil(n / 256)), Block: (256)
pub const ADAMW_STEP_CUDA: &str = r#"
extern "C" __global__ void adamw_step(
    float* __restrict__ param,
    const float* __restrict__ grad,
    float* __restrict__ m,
    float* __restrict__ v,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float weight_decay,
    const float beta1_pow,
    const float beta2_pow,
    const unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float g = grad[i];
    float p = param[i];

    // Decoupled weight decay
    p -= lr * weight_decay * p;

    // Moment updates
    float mi = beta1 * m[i] + (1.0f - beta1) * g;
    float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = mi;
    v[i] = vi;

    // Bias-corrected update
    float m_hat = mi / (1.0f - beta1_pow);
    float v_hat = vi / (1.0f - beta2_pow);
    p -= lr * m_hat / (sqrtf(v_hat) + eps);

    param[i] = p;
}
"#;

/// AdamW step for bf16 params with f32 grad/m/v.
/// Reads bf16 param → f32 compute → writes bf16 param.
/// Grid: (ceil(n / 256)), Block: (256)
pub const ADAMW_STEP_BF16_CUDA: &str = r#"
__device__ float bf16_to_float(unsigned short bits) {
    unsigned int f = ((unsigned int)bits) << 16;
    return __int_as_float(f);
}
__device__ unsigned short float_to_bf16(float f) {
    unsigned int bits = __float_as_uint(f);
    unsigned int lsb = (bits >> 16) & 1;
    bits = bits + 0x7FFF + lsb;
    return (unsigned short)(bits >> 16);
}

extern "C" __global__ void adamw_step_bf16(
    unsigned short* __restrict__ param,
    const float* __restrict__ grad,
    float* __restrict__ m,
    float* __restrict__ v,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float weight_decay,
    const float beta1_pow,
    const float beta2_pow,
    const unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float g = grad[i];
    float p = bf16_to_float(param[i]);

    p -= lr * weight_decay * p;

    float mi = beta1 * m[i] + (1.0f - beta1) * g;
    float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = mi;
    v[i] = vi;

    float m_hat = mi / (1.0f - beta1_pow);
    float v_hat = vi / (1.0f - beta2_pow);
    p -= lr * m_hat / (sqrtf(v_hat) + eps);

    param[i] = float_to_bf16(p);
}
"#;

/// Element-wise addition: f32_dst[i] += bf16_src[i] (bf16 → f32 on the fly).
/// Grid: (ceil(n / 256)), Block: (256)
pub const ADD_ASSIGN_BF16_TO_F32_CUDA: &str = r#"
extern "C" __global__ void add_assign_bf16_to_f32(
    float* __restrict__ dst,
    const unsigned short* __restrict__ src,
    const unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int f = ((unsigned int)src[i]) << 16;
    dst[i] += __int_as_float(f);
}
"#;

/// Element-wise in-place addition: dst[i] += src[i].
/// Grid: (ceil(n / 256)), Block: (256)
pub const ADD_ASSIGN_CUDA: &str = r#"
extern "C" __global__ void add_assign(
    float* __restrict__ dst,
    const float* __restrict__ src,
    const unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] += src[i];
}
"#;

/// Element-wise in-place addition: bf16 dst[i] += bf16 src[i].
/// Reads bf16, computes in f32, writes bf16.
/// Grid: (ceil(n / 256)), Block: (256)
pub const ADD_ASSIGN_BF16_CUDA: &str = r#"
extern "C" __global__ void add_assign_bf16(
    unsigned short* __restrict__ dst,
    const unsigned short* __restrict__ src,
    const unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float d = __int_as_float(((unsigned int)dst[i]) << 16);
    float s = __int_as_float(((unsigned int)src[i]) << 16);
    float result = d + s;
    unsigned int bits = __float_as_int(result);
    dst[i] = (unsigned short)((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}
"#;

/// Element-wise in-place addition: bf16 dst[i] += f32 src[i].
/// Reads bf16 dst, f32 src, computes in f32, writes bf16.
/// Grid: (ceil(n / 256)), Block: (256)
pub const ADD_ASSIGN_F32_TO_BF16_CUDA: &str = r#"
extern "C" __global__ void add_assign_f32_to_bf16(
    unsigned short* __restrict__ dst,
    const float* __restrict__ src,
    const unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float d = __int_as_float(((unsigned int)dst[i]) << 16);
    float result = d + src[i];
    unsigned int bits = __float_as_int(result);
    dst[i] = (unsigned short)((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}
"#;

/// Zero buffer: dst[i] = 0.
/// Grid: (ceil(n / 256)), Block: (256)
pub const ZERO_BUFFER_CUDA: &str = r#"
extern "C" __global__ void zero_buffer(
    float* __restrict__ dst,
    const unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = 0.0f;
}
"#;

/// Parallel reduction: sum of squares → atomicAdd into output[0].
/// Multiple blocks contribute via atomicAdd, so output must be zero-initialized.
/// Grid: (ceil(n / (256*4))), Block: (256) — each thread handles 4 elements.
pub const REDUCE_SUM_SQ_CUDA: &str = r#"
extern "C" __global__ void reduce_sum_sq(
    const float* __restrict__ input,
    float* __restrict__ output,
    const unsigned int n)
{
    __shared__ float shared[256];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 4 + tid;

    float local_sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        unsigned int i = idx + k * blockDim.x;
        if (i < n) {
            float v = input[i];
            local_sum += v * v;
        }
    }
    shared[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(output, shared[0]);
}
"#;

/// In-place scale: data[i] *= scale.
/// Grid: (ceil(n / 256)), Block: (256)
pub const SCALE_BUFFER_CUDA: &str = r#"
extern "C" __global__ void scale_buffer(
    float* __restrict__ data,
    const float scale,
    const unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    data[i] *= scale;
}
"#;

/// Fused multi-buffer sum-of-squares: processes all grad buffers in one kernel.
/// Takes array of buffer pointers and array of cumulative offsets.
/// Grid: (ceil(total_elems / 1024)), Block: (256), 4 elements/thread.
pub const MULTI_BUFFER_SUM_SQ_CUDA: &str = r#"
extern "C" __global__ void multi_buffer_sum_sq(
    const unsigned long long* __restrict__ ptrs,
    const unsigned int* __restrict__ offsets,
    const unsigned int n_bufs,
    const unsigned int total_n,
    float* __restrict__ output)
{
    __shared__ float shared[256];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 4 + tid;

    float local_sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        unsigned int gi = idx + k * blockDim.x;
        if (gi < total_n) {
            // Binary search for which buffer this index falls in
            unsigned int lo = 0, hi = n_bufs;
            while (lo < hi) {
                unsigned int mid = (lo + hi) / 2;
                if (offsets[mid + 1] <= gi) lo = mid + 1;
                else hi = mid;
            }
            const float* buf = (const float*)ptrs[lo];
            unsigned int local_idx = gi - offsets[lo];
            float v = buf[local_idx];
            local_sum += v * v;
        }
    }
    shared[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(output, shared[0]);
}
"#;

/// Fused multi-buffer scale: scales all grad buffers in one kernel.
/// Same layout as multi_buffer_sum_sq.
pub const MULTI_BUFFER_SCALE_CUDA: &str = r#"
extern "C" __global__ void multi_buffer_scale(
    const unsigned long long* __restrict__ ptrs,
    const unsigned int* __restrict__ offsets,
    const unsigned int n_bufs,
    const unsigned int total_n,
    const float scale)
{
    unsigned int gi = blockIdx.x * blockDim.x + threadIdx.x;
    if (gi >= total_n) return;

    // Binary search for buffer
    unsigned int lo = 0, hi = n_bufs;
    while (lo < hi) {
        unsigned int mid = (lo + hi) / 2;
        if (offsets[mid + 1] <= gi) lo = mid + 1;
        else hi = mid;
    }
    float* buf = (float*)ptrs[lo];
    unsigned int local_idx = gi - offsets[lo];
    buf[local_idx] *= scale;
}
"#;
