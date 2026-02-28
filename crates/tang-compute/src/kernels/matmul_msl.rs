//! Hand-optimized MSL matmul kernel using simdgroup_matrix for Apple Silicon.

/// MSL matmul kernel using simdgroup_matrix_multiply_accumulate.
///
/// A: [M, K], B: [K, N], C: [M, N], row-major.
/// Uses 8x8 simdgroup tiles for hardware-accelerated matrix multiply.
///
/// Dispatch: threadgroups = ceil(M/32) × ceil(N/32), threads_per_threadgroup = 32×4 (128).
/// Each threadgroup computes a 32×32 tile of C using 4 simdgroups.
pub const MATMUL_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;
using namespace metal::simdgroup;

// Each threadgroup: 32x32 tile of C
// Each simdgroup: 8x8 accumulators tiled across the 32x32 block
// Threadgroup layout: 128 threads = 4 simdgroups of 32 threads

constant uint TILE = 32;
constant uint BK = 8;

kernel void matmul(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    device const uint* params [[buffer(3)]],    // [M, K, N]
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint M = params[0];
    uint K = params[1];
    uint N = params[2];

    // Each simdgroup handles a 8x32 strip of the 32x32 tile
    uint row_base = tg_pos.y * TILE + sg_id * 8;
    uint col_base = tg_pos.x * TILE;

    // Accumulate 4 8x8 sub-tiles across the column dimension
    simdgroup_float8x8 acc[4];
    for (int i = 0; i < 4; i++) {
        acc[i] = simdgroup_float8x8(0);
    }

    // Walk along K dimension in steps of BK
    for (uint kb = 0; kb < K; kb += BK) {
        // Load A tile: 8 rows × BK cols
        simdgroup_float8x8 a_tile;
        simdgroup_load(a_tile, A + row_base * K + kb, K);

        // Load 4 B tiles: BK rows × 8 cols each
        for (int j = 0; j < 4; j++) {
            simdgroup_float8x8 b_tile;
            simdgroup_load(b_tile, B + kb * N + (col_base + j * 8), N);
            simdgroup_multiply_accumulate(acc[j], a_tile, b_tile, acc[j]);
        }
    }

    // Store results
    for (int j = 0; j < 4; j++) {
        if (row_base < M && (col_base + j * 8) < N) {
            simdgroup_store(acc[j], C + row_base * N + (col_base + j * 8), N);
        }
    }
}
"#;

/// Simple matmul fallback for non-simdgroup devices or small matrices.
/// Uses a naive per-thread approach.
pub const MATMUL_NAIVE_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void matmul_naive(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    device const uint* params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint M = params[0];
    uint K = params[1];
    uint N = params[2];

    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (uint i = 0; i < K; i++) {
        sum += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
}
"#;
