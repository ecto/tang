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
/// `0 ..= cache_start + qi`, or with a sliding window (`params[7] > 0`) only the last `window`
/// of them. Splits start at key `params[8]` (decode skips keys no window can see).
/// D must be a multiple of 32, at most 256.
pub const FLASH_DECODE_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

#define MAXV 8   // max D/32

kernel void attn_partial(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* P [[buffer(3)]],            // [q_len, n_heads, n_splits, D + 2]
    device const uint* params [[buffer(4)]],  // [cache_start, q_len, n_heads, n_kv_heads, D, n_splits, split_len, window, base]
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
    uint window = params[7], base = params[8];
    uint head = tg.x, split = tg.y, qi = tg.z;
    uint kv_head = head / (n_heads / n_kv);
    uint kv_dim = n_kv * D;
    uint per = D / 32;

    uint attend = cache_start + qi + 1;
    uint lo = (window > 0 && attend > window) ? attend - window : 0;
    uint s0 = base + split * split_len;
    uint j0 = max(s0, lo);
    uint j1 = min(s0 + split_len, attend);

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

// Decode (q_len == 1): lanes score different keys. Threadgroup (head, split), 8 simdgroups;
// each takes 32-key blocks: lane t scores key j0+t against q (from threadgroup memory), the
// block's max/sum is one simd reduction, and V is accumulated with lanes owning output dims
// and p broadcast by shuffles. Same partial layout as attn_partial. D % 4 == 0, D <= 256.
kernel void attn_decode(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* P [[buffer(3)]],
    device const uint* params [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint cache_start = params[0];
    uint n_heads = params[2], n_kv = params[3], D = params[4];
    uint n_splits = params[5], split_len = params[6];
    uint window = params[7], base = params[8];
    uint head = tg.x, split = tg.y;
    uint kv_head = head / (n_heads / n_kv);
    uint kv_dim = n_kv * D;
    uint per = (D + 31) / 32;
    uint attend = cache_start + 1;
    uint lo = (window > 0 && attend > window) ? attend - window : 0;
    uint s0 = base + split * split_len;
    uint j0 = max(s0, lo);
    uint j1 = min(s0 + split_len, attend);

    threadgroup float4 qs[64];
    float scale = rsqrt(float(D));
    for (uint i = tid; i < D / 4; i += 256) qs[i] = ((device const float4*)(Q + head * D))[i] * scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m = -INFINITY, l = 0;
    float acc[8];
    for (uint v = 0; v < per; v++) acc[v] = 0;
    for (uint b = j0 + sg * 32; b < j1; b += 256) {
        uint j = b + lane;
        bool valid = j < j1;
        float s = -INFINITY;
        if (valid) {
            device const float4* kp = (device const float4*)(K + (ulong)j * kv_dim + kv_head * D);
            float4 d4 = 0;
            for (uint i = 0; i < D / 4; i++) d4 += qs[i] * kp[i];
            s = d4.x + d4.y + d4.z + d4.w;
        }
        float bm = simd_max(s);
        float mn = max(m, bm);
        float c = (m == -INFINITY) ? 0.0f : exp(m - mn);
        float p = valid ? exp(s - mn) : 0.0f;
        l = l * c + simd_sum(p);
        m = mn;
        for (uint v = 0; v < per; v++) acc[v] *= c;
        uint n = min(32u, j1 - b);
        for (uint t = 0; t < n; t++) {
            float pt = simd_shuffle(p, t);
            device const float* vp = V + (ulong)(b + t) * kv_dim + kv_head * D;
            for (uint v = 0; v < per; v++) {
                uint d = v * 32 + lane;
                if (d < D) acc[v] += pt * vp[d];
            }
        }
    }

    threadgroup float sm[8], sl[8];
    threadgroup float sacc[8 * 256];
    if (lane == 0) { sm[sg] = m; sl[sg] = l; }
    for (uint v = 0; v < per; v++) { uint d = v * 32 + lane; if (d < D) sacc[sg * D + d] = acc[v]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0) return;
    float M = -INFINITY;
    for (uint g = 0; g < 8; g++) M = max(M, sm[g]);
    float L = 0;
    float out[8];
    for (uint v = 0; v < per; v++) out[v] = 0;
    for (uint g = 0; g < 8; g++) {
        float c = (sm[g] == -INFINITY) ? 0.0f : exp(sm[g] - M);
        L += sl[g] * c;
        for (uint v = 0; v < per; v++) { uint d = v * 32 + lane; if (d < D) out[v] += sacc[g * D + d] * c; }
    }
    device float* pp = P + ((ulong)head * n_splits + split) * (D + 2);
    for (uint v = 0; v < per; v++) { uint d = v * 32 + lane; if (d < D) pp[2 + d] = out[v]; }
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

/// 4-bit affine weights (MLX layout, see `Kind::Q4`): decode GEMV, tiled prefill matmul, and
/// embedding lookup. `group` must be a multiple of 32 and K a multiple of `group`.
pub const Q4_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

inline float bf(ushort b) { return as_type<float>(uint(b) << 16); }

// Eight nibbles of `w` (low first) against eight activations.
inline float dot8(uint w, float4 a, float4 b) {
    return float(w & 0xf) * a.x + float((w >> 4) & 0xf) * a.y
         + float((w >> 8) & 0xf) * a.z + float((w >> 12) & 0xf) * a.w
         + float((w >> 16) & 0xf) * b.x + float((w >> 20) & 0xf) * b.y
         + float((w >> 24) & 0xf) * b.z + float(w >> 28) * b.w;
}

// Each simdgroup computes 4 output columns; each lane holds 16 activations in registers and
// applies them to all 4, so activation traffic is amortized. Needs K % 16 == 0.
// Dispatch: threadgroups = ceil(N / 8), 64 threads (2 simdgroups).
kernel void gemv_q4(
    device const float* X [[buffer(0)]],
    device const uint* W [[buffer(1)]],
    device const ushort* S [[buffer(2)]],
    device const ushort* B [[buffer(3)]],
    device float* Y [[buffer(4)]],
    device const uint* params [[buffer(5)]],  // [M, K, N, group]
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint M = params[0], K = params[1], N = params[2], group = params[3];
    uint n0 = tg * 8 + sg * 4;
    if (n0 >= N) return;
    uint rows = min(4u, N - n0);
    uint G = K / group;
    uint words = K / 8;

    for (uint r = 0; r < M; r++) {
        float acc[4] = {0, 0, 0, 0};
        for (uint k = lane * 16; k < K; k += 512) {
            device const float4* xp = (device const float4*)(X + r * K + k);
            float4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
            float4 xs4 = x0 + x1 + x2 + x3;
            float xs = xs4.x + xs4.y + xs4.z + xs4.w;
            for (uint j = 0; j < rows; j++) {
                uint n = n0 + j;
                uint2 w = *(device const uint2*)(W + (ulong)n * words + k / 8);
                uint g = n * G + k / group;
                float d = dot8(w.x, x0, x1) + dot8(w.y, x2, x3);
                acc[j] += bf(S[g]) * d + bf(B[g]) * xs;
            }
        }
        for (uint j = 0; j < rows; j++) {
            float v = simd_sum(acc[j]);
            if (lane == 0) Y[r * N + n0 + j] = v;
        }
    }
}

inline float q4_at(device const uint* W, device const ushort* S, device const ushort* B,
                   uint row, uint k, uint K, uint group) {
    uint q = (W[(ulong)row * (K / 8) + k / 8] >> (4 * (k % 8))) & 0xf;
    uint g = row * (K / group) + k / group;
    return bf(S[g]) * float(q) + bf(B[g]);
}

// Same tiling as matmul_bt_bf16, dequantizing B into threadgroup memory.
constant uint T = 32;

kernel void matmul_bt_q4(
    device const float* A [[buffer(0)]],
    device const uint* W [[buffer(1)]],
    device const ushort* S [[buffer(2)]],
    device const ushort* Bz [[buffer(3)]],
    device float* C [[buffer(4)]],
    device const uint* params [[buffer(5)]],  // [M, K, N, group]
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    uint M = params[0], K = params[1], N = params[2], group = params[3];
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
            uint row = row0 + r;
            As[idx] = (row < M && k < K) ? A[(ulong)row * K + k] : 0.0f;
        }
        {
            // One packed word (8 weights) per thread: column tid/4, weights (tid%4)*8 .. +8.
            uint c = tid / 4, k = kb + (tid % 4) * 8;
            uint col = col0 + c;
            threadgroup float* dst = Bs + c * T + (tid % 4) * 8;
            if (col < N && k < K) {
                uint w = W[(ulong)col * (K / 8) + k / 8];
                uint g = col * (K / group) + k / group;
                float sc = bf(S[g]), bi = bf(Bz[g]);
                for (uint e = 0; e < 8; e++) dst[e] = sc * float((w >> (4 * e)) & 0xf) + bi;
            } else {
                for (uint e = 0; e < 8; e++) dst[e] = 0.0f;
            }
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
    for (int j = 0; j < 4; j++) simdgroup_store(acc[j], As + sg * 8 * T + j * 8, T);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = 0; e < 8; e++) {
        uint idx = tid * 8 + e;
        uint r = idx / T, c = idx % T;
        if (row0 + r < M && col0 + c < N) C[(ulong)(row0 + r) * N + col0 + c] = As[idx];
    }
}

kernel void embedding_q4(
    device const uint* W [[buffer(0)]],
    device const ushort* S [[buffer(1)]],
    device const ushort* B [[buffer(2)]],
    device const uint* ids [[buffer(3)]],
    device float* Y [[buffer(4)]],
    device const uint* params [[buffer(5)]],  // [seq_len, dim, group]
    uint gid [[thread_position_in_grid]])
{
    uint dim = params[1];
    if (gid >= params[0] * dim) return;
    uint s = gid / dim, d = gid % dim;
    Y[gid] = q4_at(W, S, B, ids[s], d, dim, params[2]);
}
"#;

/// Fused decoder-layer glue: attention prologue and split SwiGLU.
///
/// `attention_prep`: one simdgroup per (token, head) over q, k and v heads of a fused QKV row.
/// q/k heads get the optional RMS norm and half-split RoPE; q goes to `Qo`, k and v straight
/// into the caches. Each lane holds `hd/32` elements; element `lane + 32v` pairs with
/// `lane + 32(v + hd/64)` for RoPE, so `hd` must be a multiple of 64 (at most 256).
pub const FUSED_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void attention_prep(
    device const float* QKV [[buffer(0)]],
    device const float* QN [[buffer(1)]],
    device const float* KN [[buffer(2)]],
    device const float* COS [[buffer(3)]],
    device const float* SIN [[buffer(4)]],
    device float* Qo [[buffer(5)]],
    device float* KC [[buffer(6)]],
    device float* VC [[buffer(7)]],
    device const uint* params [[buffer(8)]],  // [seq, nh, nkv, hd, pos, eps bits, has_qn, has_kn]
    uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint nh = params[1], nkv = params[2], hd = params[3], pos = params[4];
    float eps = as_type<float>(params[5]);
    uint s = tg.y, h = tg.x;
    uint qd = nh * hd, kvd = nkv * hd, row = qd + 2 * kvd;
    uint per = hd / 32, half_v = per / 2, half_dim = hd / 2;
    device const float* src = QKV + (ulong)s * row + h * hd;  // heads are contiguous: q.., k.., v..

    float x[8];
    for (uint v = 0; v < per; v++) x[v] = src[v * 32 + lane];

    if (h >= nh + nkv) {  // v head: straight into the cache
        uint kh = h - nh - nkv;
        device float* dst = VC + (ulong)(pos + s) * kvd + kh * hd;
        for (uint v = 0; v < per; v++) dst[v * 32 + lane] = x[v];
        return;
    }
    bool is_q = h < nh;
    bool has_norm = is_q ? params[6] != 0 : params[7] != 0;
    if (has_norm) {
        device const float* w = is_q ? QN : KN;
        float ss = 0;
        for (uint v = 0; v < per; v++) ss += x[v] * x[v];
        float r = rsqrt(simd_sum(ss) / float(hd) + eps);
        for (uint v = 0; v < per; v++) x[v] *= r * w[v * 32 + lane];
    }
    float y[8];
    uint t = (pos + s) * half_dim;
    for (uint v = 0; v < half_v; v++) {
        uint i = v * 32 + lane;
        float c = COS[t + i], sn = SIN[t + i];
        float x0 = x[v], x1 = x[v + half_v];
        y[v] = x0 * c - x1 * sn;
        y[v + half_v] = x1 * c + x0 * sn;
    }
    device float* dst = is_q ? Qo + (ulong)s * qd + h * hd
                             : KC + (ulong)(pos + s) * kvd + (h - nh) * hd;
    for (uint v = 0; v < per; v++) dst[v * 32 + lane] = y[v];
}

// GeGLU (Gemma): gelu_tanh(gate) * up over a fused gate/up projection.
kernel void geglu_split(
    device const float* GU [[buffer(0)]],
    device float* Y [[buffer(1)]],
    device const uint* params [[buffer(2)]],  // [rows, ff]
    uint gid [[thread_position_in_grid]])
{
    uint ff = params[1];
    if (gid >= params[0] * ff) return;
    uint r = gid / ff, i = gid % ff;
    float g = GU[(ulong)r * 2 * ff + i];
    // Fast-math tanh overflows (inf/inf) for large inputs; it's ±1 well before ±10.
    float t = tanh(clamp(0.7978845608f * (g + 0.044715f * g * g * g), -10.0f, 10.0f));
    Y[gid] = 0.5f * g * (1.0f + t) * GU[(ulong)r * 2 * ff + ff + i];
}

kernel void swiglu_split(
    device const float* GU [[buffer(0)]],
    device float* Y [[buffer(1)]],
    device const uint* params [[buffer(2)]],  // [rows, ff]
    uint gid [[thread_position_in_grid]])
{
    uint ff = params[1];
    if (gid >= params[0] * ff) return;
    uint r = gid / ff, i = gid % ff;
    float g = GU[(ulong)r * 2 * ff + i];
    Y[gid] = g / (1.0f + exp(-g)) * GU[(ulong)r * 2 * ff + ff + i];
}
"#;

/// Tiled causal flash attention for prefill, on simdgroup matrices.
///
/// Threadgroup = (32-query block, head), 4 simdgroups × 8 query rows. Per 32-key block, each
/// simdgroup computes S = Q·Kᵀ (8×32), runs the online softmax on it in threadgroup scratch,
/// rescales its O accumulator (8×D, as D/8 8×8 tiles) by diag(α) and adds P·V. K/V tiles are
/// read straight from the cache with `simdgroup_load`. Blocks past the diagonal are skipped.
/// Requires: q rows padded to a multiple of 32; K/V caches readable to a multiple of 32 rows
/// past the end; D a multiple of 8, at most 128.
pub const FLASH_PREFILL_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

#define BQ 32
#define BK 32
#define MAXT 16   // D / 8

kernel void attn_prefill(
    device const float* Q [[buffer(0)]],     // [q_pad, nh*D]
    device const float* K [[buffer(1)]],     // [.., nkv*D]
    device const float* V [[buffer(2)]],
    device float* O [[buffer(3)]],           // [q_pad, nh*D]
    device const uint* params [[buffer(4)]], // [cache_start, q_len, nh, nkv, D]
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint cache_start = params[0], q_len = params[1], nh = params[2], nkv = params[3], D = params[4];
    uint qb = tg.x, head = tg.y;
    uint kvh = head / (nh / nkv);
    uint qs = nh * D, ks = nkv * D;
    uint r0 = qb * BQ + sg * 8;                  // first query row of this simdgroup
    uint T = D / 8;
    float scale = rsqrt(float(D));

    threadgroup float scratch[4][8 * BK];
    threadgroup float diag[4][64];
    threadgroup float* S = scratch[sg];
    threadgroup float* Dg = diag[sg];

    simdgroup_float8x8 o[MAXT];
    for (uint t = 0; t < T; t++) o[t] = simdgroup_float8x8(0);

    // Row bookkeeping: lane handles row lane/4, columns (lane%4)*8 .. +8.
    uint row = lane / 4, part = lane % 4;
    float m = -INFINITY, l = 0;
    uint abs_row = cache_start + r0 + row;       // absolute position of this query
    uint last = min(cache_start + r0 + 7, cache_start + q_len - 1);  // last position any row here needs

    device const float* qp = Q + (ulong)r0 * qs + head * D;
    for (uint j0 = 0; j0 <= last; j0 += BK) {
        // S = Q Kᵀ for 8 rows × 32 keys.
        simdgroup_float8x8 s[4];
        for (uint c = 0; c < 4; c++) s[c] = simdgroup_float8x8(0);
        for (uint d = 0; d < D; d += 8) {
            simdgroup_float8x8 a;
            simdgroup_load(a, qp + d, qs);
            for (uint c = 0; c < 4; c++) {
                simdgroup_float8x8 b;
                simdgroup_load(b, K + (ulong)(j0 + c * 8) * ks + kvh * D + d, ks, ulong2(0), true);
                simdgroup_multiply_accumulate(s[c], a, b, s[c]);
            }
        }
        for (uint c = 0; c < 4; c++) simdgroup_store(s[c], S + c * 8, BK);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax on this lane's 8 columns of its row.
        float v[8];
        float bm = -INFINITY;
        for (uint i = 0; i < 8; i++) {
            uint j = j0 + part * 8 + i;
            v[i] = (j <= abs_row) ? S[row * BK + part * 8 + i] * scale : -INFINITY;
            bm = max(bm, v[i]);
        }
        bm = max(bm, simd_shuffle_xor(bm, 1));
        bm = max(bm, simd_shuffle_xor(bm, 2));
        float mn = max(m, bm);
        float alpha = (m == -INFINITY) ? 0.0f : exp(m - mn);
        float ps = 0;
        for (uint i = 0; i < 8; i++) {
            float p = (v[i] == -INFINITY) ? 0.0f : exp(v[i] - mn);
            S[row * BK + part * 8 + i] = p;
            ps += p;
        }
        ps += simd_shuffle_xor(ps, 1);
        ps += simd_shuffle_xor(ps, 2);
        l = l * alpha + ps;
        m = mn;
        // diag(alpha)
        for (uint i = lane; i < 64; i += 32) Dg[i] = 0;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (part == 0) Dg[row * 8 + row] = alpha;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 da;
        simdgroup_load(da, Dg, 8);
        simdgroup_float8x8 p[4];
        for (uint c = 0; c < 4; c++) simdgroup_load(p[c], S + c * 8, BK);
        for (uint t = 0; t < T; t++) {
            simdgroup_float8x8 acc;
            simdgroup_multiply(acc, da, o[t]);
            for (uint c = 0; c < 4; c++) {
                simdgroup_float8x8 vt;
                simdgroup_load(vt, V + (ulong)(j0 + c * 8) * ks + kvh * D + t * 8, ks);
                simdgroup_multiply_accumulate(acc, p[c], vt, acc);
            }
            o[t] = acc;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    // O /= l
    for (uint i = lane; i < 64; i += 32) Dg[i] = 0;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (part == 0) Dg[row * 8 + row] = (l > 0) ? 1.0f / l : 0.0f;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 dl;
    simdgroup_load(dl, Dg, 8);
    device float* op = O + (ulong)r0 * qs + head * D;
    for (uint t = 0; t < T; t++) {
        simdgroup_float8x8 r;
        simdgroup_multiply(r, dl, o[t]);
        simdgroup_store(r, op + t * 8, qs);
    }
}
"#;
