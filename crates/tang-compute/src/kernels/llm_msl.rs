//! Kernels for autoregressive LLM inference: decode-time GEMV and half-split RoPE.

/// `y[r, n] = sum_k x[r, k] * W[n, k]` for a handful of rows (`M <= 8`), W stored `[N, K]`.
///
/// Decode is memory-bound: every weight is read exactly once, by one simdgroup per output
/// column, in float4s, and reused for all `M` rows.
/// Dispatch: threadgroups = ceil(N / 8), threads_per_threadgroup = 256 (8 simdgroups).
pub const GEMV_BT_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gemv_bt(
    device const float* X [[buffer(0)]],   // [M, K]
    device const float* W [[buffer(1)]],   // [N, K]
    device float* Y [[buffer(2)]],         // [M, N]
    device const uint* params [[buffer(3)]],  // [M, K, N]
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint M = params[0];
    uint K = params[1];
    uint N = params[2];
    uint n = tg * 8 + sg;
    if (n >= N) return;

    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    device const float4* w4 = (device const float4*)(W + (ulong)n * K);
    uint K4 = K / 4;
    for (uint i = lane; i < K4; i += 32) {
        float4 w = w4[i];
        for (uint r = 0; r < M; r++) {
            float4 x = ((device const float4*)(X + r * K))[i];
            acc[r] += dot(w, x);
        }
    }
    // Tail when K isn't a multiple of 4.
    for (uint i = K4 * 4 + lane; i < K; i += 32) {
        float w = W[(ulong)n * K + i];
        for (uint r = 0; r < M; r++) acc[r] += w * X[r * K + i];
    }
    for (uint r = 0; r < M; r++) {
        float s = simd_sum(acc[r]);
        if (lane == 0) Y[r * N + n] = s;
    }
}
"#;

/// Half-split RoPE: rotates `(x[i], x[i + half])` by the angle for `start_pos + s`.
/// Dispatch: one thread per (s, h, i < half).
pub const ROPE_HALF_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void rope_half(
    device const float* X [[buffer(0)]],
    device const float* COS [[buffer(1)]],
    device const float* SIN [[buffer(2)]],
    device float* Y [[buffer(3)]],
    device const uint* params [[buffer(4)]],  // [seq_len, n_heads, head_dim, start_pos]
    uint gid [[thread_position_in_grid]])
{
    uint seq_len = params[0];
    uint n_heads = params[1];
    uint head_dim = params[2];
    uint start_pos = params[3];
    uint half_dim = head_dim / 2;
    if (gid >= seq_len * n_heads * half_dim) return;

    uint i = gid % half_dim;
    uint sh = gid / half_dim;       // s * n_heads + h
    uint s = sh / n_heads;
    uint base = sh * head_dim;
    uint t = (start_pos + s) * half_dim + i;
    float c = COS[t];
    float sn = SIN[t];
    float x0 = X[base + i];
    float x1 = X[base + i + half_dim];
    Y[base + i] = x0 * c - x1 * sn;
    Y[base + i + half_dim] = x1 * c + x0 * sn;
}
"#;
