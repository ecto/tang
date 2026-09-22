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

/// `out = a + b` and `out = silu(gate) * up`, elementwise.
pub const ELEMENTWISE_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void add_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* Y [[buffer(2)]],
    device const uint* params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < params[0]) Y[gid] = A[gid] + B[gid];
}

kernel void swiglu_f32(
    device const float* G [[buffer(0)]],
    device const float* U [[buffer(1)]],
    device float* Y [[buffer(2)]],
    device const uint* params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params[0]) return;
    float g = G[gid];
    Y[gid] = g / (1.0f + exp(-g)) * U[gid];
}
"#;

/// bfloat16-weight variants: decode GEMV, tiled prefill matmul, and embedding lookup.
/// Activations stay f32; weights are widened in registers (bf16 -> f32 is a 16-bit shift).
pub const BF16_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

inline float bf(ushort b) { return as_type<float>(uint(b) << 16); }
inline float4 bf4(ushort4 b) { return float4(bf(b.x), bf(b.y), bf(b.z), bf(b.w)); }

// Same shape as gemv_bt: one simdgroup per output column, M <= 8 rows.
kernel void gemv_bt_bf16(
    device const float* X [[buffer(0)]],
    device const ushort* W [[buffer(1)]],
    device float* Y [[buffer(2)]],
    device const uint* params [[buffer(3)]],
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
    device const ushort4* w4 = (device const ushort4*)(W + (ulong)n * K);
    uint K4 = K / 4;
    for (uint i = lane; i < K4; i += 32) {
        float4 w = bf4(w4[i]);
        for (uint r = 0; r < M; r++) {
            acc[r] += dot(w, ((device const float4*)(X + r * K))[i]);
        }
    }
    for (uint i = K4 * 4 + lane; i < K; i += 32) {
        float w = bf(W[(ulong)n * K + i]);
        for (uint r = 0; r < M; r++) acc[r] += w * X[r * K + i];
    }
    for (uint r = 0; r < M; r++) {
        float s = simd_sum(acc[r]);
        if (lane == 0) Y[r * N + n] = s;
    }
}

// C[M,N] = A[M,K] @ B[N,K]^T with B in bf16. 32x32 output tile per threadgroup of 128 threads
// (4 simdgroups, 8 rows each); A and B tiles staged through threadgroup memory, so any M, N, K
// work (out-of-range elements load as zero, stores are bounds-checked).
constant uint T = 32;

kernel void matmul_bt_bf16(
    device const float* A [[buffer(0)]],
    device const ushort* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    device const uint* params [[buffer(3)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    uint M = params[0];
    uint K = params[1];
    uint N = params[2];
    uint row0 = tg_pos.y * T;
    uint col0 = tg_pos.x * T;

    threadgroup float As[T * T];
    threadgroup float Bs[T * T];

    simdgroup_float8x8 acc[4];
    for (int j = 0; j < 4; j++) acc[j] = simdgroup_float8x8(0);

    for (uint kb = 0; kb < K; kb += T) {
        for (uint e = 0; e < 8; e++) {
            uint idx = tid * 8 + e;
            uint r = idx / T, kk = idx % T;
            uint k = kb + kk;
            uint row = row0 + r, col = col0 + r;
            As[idx] = (row < M && k < K) ? A[(ulong)row * K + k] : 0.0f;
            Bs[idx] = (col < N && k < K) ? bf(B[(ulong)col * K + k]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ks = 0; ks < T; ks += 8) {
            simdgroup_float8x8 a;
            simdgroup_load(a, As + sg * 8 * T + ks, T);
            for (int j = 0; j < 4; j++) {
                simdgroup_float8x8 b;
                simdgroup_load(b, Bs + j * 8 * T + ks, T, ulong2(0), true);
                simdgroup_multiply_accumulate(acc[j], a, b, acc[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Stage through threadgroup memory for bounds-checked stores.
    for (int j = 0; j < 4; j++) simdgroup_store(acc[j], As + sg * 8 * T + j * 8, T);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = 0; e < 8; e++) {
        uint idx = tid * 8 + e;
        uint r = idx / T, c = idx % T;
        if (row0 + r < M && col0 + c < N) C[(ulong)(row0 + r) * N + col0 + c] = As[idx];
    }
}

kernel void embedding_bf16(
    device const ushort* W [[buffer(0)]],
    device const uint* ids [[buffer(1)]],
    device float* Y [[buffer(2)]],
    device const uint* params [[buffer(3)]],  // [seq_len, dim]
    uint gid [[thread_position_in_grid]])
{
    uint dim = params[1];
    if (gid >= params[0] * dim) return;
    uint s = gid / dim, d = gid % dim;
    Y[gid] = bf(W[(ulong)ids[s] * dim + d]);
}
"#;

/// `dst[dst_off + i] = src[src_off + i]` — a compute-encoder copy, so small copies (KV cache
/// appends, row slices) don't force an encoder switch.
pub const COPY_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void copy_f32(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    device const uint* params [[buffer(2)]],  // [src_off, dst_off, n]
    uint gid [[thread_position_in_grid]])
{
    if (gid < params[2]) dst[params[1] + gid] = src[params[0] + gid];
}
"#;

/// Split-KV ("flash decoding") attention over a KV cache, for decode and prefill.
///
/// Pass 1 (`attn_partial`): threadgroup (head, split, qi), 8 simdgroups. Each simdgroup runs an
/// online softmax over its share of the split's keys with no barriers: lanes hold D/32 dims
/// of q and of the accumulator, a key's score is one `simd_sum`. Simdgroups then merge, and the
/// threadgroup writes (max, sum, acc[D]) for its split.
/// Pass 2 (`attn_combine`): per (qi, head), rescale and sum the splits.
///
/// Layouts: q/out `[q_len, n_heads*D]`, K/V `[pos, n_kv_heads*D]`; query qi attends to keys
/// `0 ..= cache_start + qi`. D must be a multiple of 32, at most 256.
pub const FLASH_DECODE_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

#define MAXV 8   // max D/32

kernel void attn_partial(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* P [[buffer(3)]],            // [q_len, n_heads, n_splits, D + 2]
    device const uint* params [[buffer(4)]],  // [cache_start, q_len, n_heads, n_kv_heads, D, n_splits, split_len]
    uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint cache_start = params[0];
    uint n_heads = params[2];
    uint n_kv = params[3];
    uint D = params[4];
    uint n_splits = params[5];
    uint split_len = params[6];
    uint head = tg.x, split = tg.y, qi = tg.z;
    uint kv_head = head / (n_heads / n_kv);
    uint kv_dim = n_kv * D;
    uint per = D / 32;

    uint attend = cache_start + qi + 1;
    uint j0 = split * split_len;
    uint j1 = min(j0 + split_len, attend);

    float q[MAXV], acc[MAXV];
    float scale = rsqrt(float(D));
    device const float* qp = Q + (ulong)qi * n_heads * D + head * D;
    for (uint v = 0; v < per; v++) { q[v] = qp[v * 32 + lane] * scale; acc[v] = 0; }
    float m = -INFINITY, l = 0;

    for (uint j = j0 + sg; j < j1; j += 8) {
        device const float* kp = K + (ulong)j * kv_dim + kv_head * D;
        float s = 0;
        for (uint v = 0; v < per; v++) s += q[v] * kp[v * 32 + lane];
        s = simd_sum(s);
        float m2 = max(m, s);
        float c = exp(m - m2), p = exp(s - m2);
        l = l * c + p;
        device const float* vp = V + (ulong)j * kv_dim + kv_head * D;
        for (uint v = 0; v < per; v++) acc[v] = acc[v] * c + p * vp[v * 32 + lane];
        m = m2;
    }

    // Merge the 8 simdgroups.
    threadgroup float sm[8], sl[8];
    threadgroup float sacc[8 * 256];
    if (lane == 0) { sm[sg] = m; sl[sg] = l; }
    for (uint v = 0; v < per; v++) sacc[sg * D + v * 32 + lane] = acc[v];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0) return;

    float M = -INFINITY;
    for (uint g = 0; g < 8; g++) M = max(M, sm[g]);
    float L = 0;
    float out[MAXV];
    for (uint v = 0; v < per; v++) out[v] = 0;
    for (uint g = 0; g < 8; g++) {
        float c = (sm[g] == -INFINITY) ? 0.0f : exp(sm[g] - M);
        L += sl[g] * c;
        for (uint v = 0; v < per; v++) out[v] += sacc[g * D + v * 32 + lane] * c;
    }
    device float* pp = P + (((ulong)qi * n_heads + head) * n_splits + split) * (D + 2);
    for (uint v = 0; v < per; v++) pp[2 + v * 32 + lane] = out[v];
    if (lane == 0) { pp[0] = M; pp[1] = L; }
}

kernel void attn_combine(
    device const float* P [[buffer(0)]],
    device float* O [[buffer(1)]],
    device const uint* params [[buffer(2)]],  // same as attn_partial
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    uint n_heads = params[2];
    uint D = params[4];
    uint n_splits = params[5];
    uint head = tg.x, qi = tg.y;
    device const float* base = P + ((ulong)qi * n_heads + head) * n_splits * (D + 2);

    float M = -INFINITY;
    for (uint s = 0; s < n_splits; s++) M = max(M, base[s * (D + 2)]);
    float L = 0;
    for (uint s = 0; s < n_splits; s++) {
        float m = base[s * (D + 2)];
        if (m != -INFINITY) L += base[s * (D + 2) + 1] * exp(m - M);
    }
    for (uint d = tid; d < D; d += 32) {
        float o = 0;
        for (uint s = 0; s < n_splits; s++) {
            float m = base[s * (D + 2)];
            if (m != -INFINITY) o += base[s * (D + 2) + 2 + d] * exp(m - M);
        }
        O[(ulong)qi * n_heads * D + head * D + d] = o / L;
    }
}
"#;
