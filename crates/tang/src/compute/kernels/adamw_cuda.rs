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
