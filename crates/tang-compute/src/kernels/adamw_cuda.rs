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
