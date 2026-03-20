//! MSL AdamW optimizer kernel for Metal backend.

/// AdamW step: updates param, m, v in-place.
/// Dispatch: threadgroups = ceil(n / 256), threads_per_threadgroup = 256.
/// Hyperparams passed via buffer: [lr, beta1, beta2, eps, wd, beta1_pow, beta2_pow, n].
pub const ADAMW_STEP_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void adamw_step(
    device float* param [[buffer(0)]],
    device const float* grad [[buffer(1)]],
    device float* m [[buffer(2)]],
    device float* v [[buffer(3)]],
    device const float* hparams [[buffer(4)]],  // [lr, beta1, beta2, eps, wd, beta1_pow, beta2_pow]
    device const uint* count [[buffer(5)]],      // [n]
    uint gid [[thread_position_in_grid]])
{
    uint n = count[0];
    if (gid >= n) return;

    float lr = hparams[0];
    float beta1 = hparams[1];
    float beta2 = hparams[2];
    float eps = hparams[3];
    float wd = hparams[4];
    float beta1_pow = hparams[5];
    float beta2_pow = hparams[6];

    float g = grad[gid];
    float p = param[gid];

    // Decoupled weight decay
    p -= lr * wd * p;

    // Moment updates
    float mi = beta1 * m[gid] + (1.0f - beta1) * g;
    float vi = beta2 * v[gid] + (1.0f - beta2) * g * g;
    m[gid] = mi;
    v[gid] = vi;

    // Bias-corrected update
    float m_hat = mi / (1.0f - beta1_pow);
    float v_hat = vi / (1.0f - beta2_pow);
    p -= lr * m_hat / (sqrt(v_hat) + eps);

    param[gid] = p;
}
"#;

/// In-place element-wise addition: dst[i] += src[i].
pub const ADD_ASSIGN_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void add_assign(
    device float* dst [[buffer(0)]],
    device const float* src [[buffer(1)]],
    device const uint* params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params[0]) return;
    dst[gid] += src[gid];
}
"#;

/// Parallel reduction: sum of squares → atomic_fetch_add into output[0].
pub const REDUCE_SUM_SQ_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void reduce_sum_sq(
    device const float* input [[buffer(0)]],
    device atomic_float* output [[buffer(1)]],
    device const uint* params [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint n = params[0];
    threadgroup float shared[256];

    float local_sum = 0.0f;
    // Each thread handles 4 elements for coalescing
    uint base = gid * 4;
    for (uint k = 0; k < 4; k++) {
        uint i = base + k;
        if (i < n) {
            float val = input[i];
            local_sum += val * val;
        }
    }
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        atomic_fetch_add_explicit(output, shared[0], memory_order_relaxed);
    }
}
"#;

/// In-place scale: buf[i] *= scale.
pub const SCALE_BUFFER_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void scale_buffer(
    device float* data [[buffer(0)]],
    device const float* scale_buf [[buffer(1)]],
    device const uint* params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params[0]) return;
    data[gid] *= scale_buf[0];
}
"#;
