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

/// C = A @ B^T. B stored as [N, K] row-major.
/// Dispatch: threadgroups = ceil(M/32) × ceil(N/32), threads_per_threadgroup = 128.
pub const MATMUL_BT_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint TILE = 32;
constant uint BK = 8;

kernel void matmul_bt(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],    // [N, K] row-major
    device float* C [[buffer(2)]],
    device const uint* params [[buffer(3)]],    // [M, K, N]
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint M = params[0];
    uint K = params[1];
    uint N = params[2];

    uint row_base = tg_pos.y * TILE + sg_id * 8;
    uint col_base = tg_pos.x * TILE;

    simdgroup_float8x8 acc[4];
    for (int i = 0; i < 4; i++) acc[i] = simdgroup_float8x8(0);

    for (uint kb = 0; kb < K; kb += BK) {
        simdgroup_float8x8 a_tile;
        simdgroup_load(a_tile, A + row_base * K + kb, K);

        for (int j = 0; j < 4; j++) {
            // B is [N,K]: row (col_base+j*8) has stride K
            simdgroup_float8x8 b_tile;
            simdgroup_load(b_tile, B + (col_base + j * 8) * K + kb, K, ulong2(0), true);
            simdgroup_multiply_accumulate(acc[j], a_tile, b_tile, acc[j]);
        }
    }

    for (int j = 0; j < 4; j++) {
        if (row_base < M && (col_base + j * 8) < N) {
            simdgroup_store(acc[j], C + row_base * N + (col_base + j * 8), N);
        }
    }
}
"#;

/// C = A^T @ B. A stored as [K, M] row-major.
/// Dispatch: threadgroups = ceil(M/32) × ceil(N/32), threads_per_threadgroup = 128.
pub const MATMUL_AT_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint TILE = 32;
constant uint BK = 8;

kernel void matmul_at(
    device const float* A [[buffer(0)]],    // [K, M] row-major
    device const float* B [[buffer(1)]],    // [K, N] row-major
    device float* C [[buffer(2)]],
    device const uint* params [[buffer(3)]],    // [M, K, N]
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint M = params[0];
    uint K = params[1];
    uint N = params[2];

    uint row_base = tg_pos.y * TILE + sg_id * 8;
    uint col_base = tg_pos.x * TILE;

    simdgroup_float8x8 acc[4];
    for (int i = 0; i < 4; i++) acc[i] = simdgroup_float8x8(0);

    for (uint kb = 0; kb < K; kb += BK) {
        // A is [K,M]: column row_base has stride M, need transpose
        simdgroup_float8x8 a_tile;
        simdgroup_load(a_tile, A + kb * M + row_base, M, ulong2(0), true);

        for (int j = 0; j < 4; j++) {
            simdgroup_float8x8 b_tile;
            simdgroup_load(b_tile, B + kb * N + (col_base + j * 8), N);
            simdgroup_multiply_accumulate(acc[j], a_tile, b_tile, acc[j]);
        }
    }

    for (int j = 0; j < 4; j++) {
        if (row_base < M && (col_base + j * 8) < N) {
            simdgroup_store(acc[j], C + row_base * N + (col_base + j * 8), N);
        }
    }
}
"#;

/// C += A @ B (accumulate into existing C).
/// Dispatch: threadgroups = ceil(M/32) × ceil(N/32), threads_per_threadgroup = 128.
pub const MATMUL_ACC_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint TILE = 32;
constant uint BK = 8;

kernel void matmul_acc(
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

    uint row_base = tg_pos.y * TILE + sg_id * 8;
    uint col_base = tg_pos.x * TILE;

    // Initialize accumulators from existing C
    simdgroup_float8x8 acc[4];
    for (int j = 0; j < 4; j++) {
        if (row_base < M && (col_base + j * 8) < N) {
            simdgroup_load(acc[j], C + row_base * N + (col_base + j * 8), N);
        } else {
            acc[j] = simdgroup_float8x8(0);
        }
    }

    for (uint kb = 0; kb < K; kb += BK) {
        simdgroup_float8x8 a_tile;
        simdgroup_load(a_tile, A + row_base * K + kb, K);

        for (int j = 0; j < 4; j++) {
            simdgroup_float8x8 b_tile;
            simdgroup_load(b_tile, B + kb * N + (col_base + j * 8), N);
            simdgroup_multiply_accumulate(acc[j], a_tile, b_tile, acc[j]);
        }
    }

    for (int j = 0; j < 4; j++) {
        if (row_base < M && (col_base + j * 8) < N) {
            simdgroup_store(acc[j], C + row_base * N + (col_base + j * 8), N);
        }
    }
}
"#;

/// C += A^T @ B. A stored as [K, M] row-major.
/// Dispatch: threadgroups = ceil(M/32) × ceil(N/32), threads_per_threadgroup = 128.
pub const MATMUL_ACC_AT_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint TILE = 32;
constant uint BK = 8;

kernel void matmul_acc_at(
    device const float* A [[buffer(0)]],    // [K, M] row-major
    device const float* B [[buffer(1)]],    // [K, N] row-major
    device float* C [[buffer(2)]],
    device const uint* params [[buffer(3)]],    // [M, K, N]
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint M = params[0];
    uint K = params[1];
    uint N = params[2];

    uint row_base = tg_pos.y * TILE + sg_id * 8;
    uint col_base = tg_pos.x * TILE;

    // Initialize accumulators from existing C
    simdgroup_float8x8 acc[4];
    for (int j = 0; j < 4; j++) {
        if (row_base < M && (col_base + j * 8) < N) {
            simdgroup_load(acc[j], C + row_base * N + (col_base + j * 8), N);
        } else {
            acc[j] = simdgroup_float8x8(0);
        }
    }

    for (uint kb = 0; kb < K; kb += BK) {
        simdgroup_float8x8 a_tile;
        simdgroup_load(a_tile, A + kb * M + row_base, M, ulong2(0), true);

        for (int j = 0; j < 4; j++) {
            simdgroup_float8x8 b_tile;
            simdgroup_load(b_tile, B + kb * N + (col_base + j * 8), N);
            simdgroup_multiply_accumulate(acc[j], a_tile, b_tile, acc[j]);
        }
    }

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

/// Implicit im2col GEMM with bounded threadgroup tiles and direct NCHW output.
/// Four SIMD groups compute a 32-spatial by 32-output-channel tile.
pub const CONV2D_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void conv2d_nchw(
    device const float* x [[buffer(0)]], device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]], device float* out [[buffer(3)]],
    device const uint* p [[buffer(4)]], uint2 tile [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    uint input = p[0], output = p[1], h = p[2], w = p[3], ks = p[4];
    uint spatial = h*w, kk = ks*ks, k = input*kk;
    uint row_base = tile.y*32, col_base = tile.x*32;
    threadgroup float a[32*8], b[32*8], result[32*32];
    simdgroup_float8x8 acc[4];
    for (uint j = 0; j < 4; j++) acc[j] = simdgroup_float8x8(0);
    for (uint kb = 0; kb < k; kb += 8) {
        for (uint i = tid; i < 256; i += 128) {
            uint row = row_base+i/8, kval = kb+i%8, col = col_base+i/8;
            float v = 0;
            if (row < spatial && kval < k) {
                int yy = int(row/w)+int((kval%kk)/ks)-int(ks/2);
                int xx = int(row%w)+int(kval%ks)-int(ks/2);
                if (yy >= 0 && yy < int(h) && xx >= 0 && xx < int(w))
                    v = x[(kval/kk*h+uint(yy))*w+uint(xx)];
            }
            a[i] = v;
            b[i] = col < output && kval < k ? weight[col*k+kval] : 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 av;
        simdgroup_load(av, a+sg*64, 8);
        for (uint j = 0; j < 4; j++) {
            simdgroup_float8x8 bv;
            simdgroup_load(bv, b+j*64, 8, ulong2(0), true);
            simdgroup_multiply_accumulate(acc[j], av, bv, acc[j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint j = 0; j < 4; j++) simdgroup_store(acc[j], result+sg*8*32+j*8, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 1024; i += 128) {
        uint row = row_base+i/32, col = col_base+i%32;
        if (row < spatial && col < output) out[col*spatial+row] = result[i]+bias[col];
    }
}
"#;
