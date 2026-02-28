//! MSL reduction kernels: softmax, rms_norm.

/// Row-wise softmax: each threadgroup handles one row.
/// params: [n_rows, row_len]
pub const SOFTMAX_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint WG = 256;

kernel void softmax(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const uint* params [[buffer(2)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint n_rows = params[0];
    uint row_len = params[1];
    if (tg_id >= n_rows) return;

    uint base = tg_id * row_len;

    threadgroup float shared[WG];

    // Phase 1: find max
    float local_max = -INFINITY;
    for (uint i = tid; i < row_len; i += tg_size) {
        local_max = max(local_max, input[base + i]);
    }
    shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce max
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = max(shared[tid], shared[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared[0];

    // Phase 2: compute exp and sum
    float local_sum = 0.0f;
    for (uint i = tid; i < row_len; i += tg_size) {
        float val = exp(input[base + i] - row_max);
        output[base + i] = val;
        local_sum += val;
    }
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce sum
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_sum = shared[0];

    // Phase 3: normalize
    float inv_sum = 1.0f / row_sum;
    for (uint i = tid; i < row_len; i += tg_size) {
        output[base + i] *= inv_sum;
    }
}
"#;

/// RMS normalization: out = x * weight / sqrt(mean(x^2) + eps).
/// Each threadgroup handles one group of `dim` elements.
/// params: [n_groups, dim, eps_bits]
pub const RMS_NORM_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint WG = 256;

kernel void rms_norm(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const uint* params [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint n_groups = params[0];
    uint dim = params[1];
    float eps = as_type<float>(params[2]);
    if (tg_id >= n_groups) return;

    uint base = tg_id * dim;

    threadgroup float shared[WG];

    // Compute sum of squares
    float local_sq = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        float v = input[base + i];
        local_sq += v * v;
    }
    shared[tid] = local_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms = rsqrt(shared[0] / float(dim) + eps);

    // Normalize
    for (uint i = tid; i < dim; i += tg_size) {
        output[base + i] = input[base + i] * rms * weight[i];
    }
}
"#;
