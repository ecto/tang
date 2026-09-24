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
/// of them. Splits start at key `params[8]` (decode skips keys no window can see). With
/// `params[9]` set, attention is bidirectional within the batch: every query sees keys up to
/// `cache_start + q_len` (image tokens, vision encoders). D at most 256.
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
    uint window = params[7], base = params[8], bidir = params[9];
    uint head = tg.x, split = tg.y, qi = tg.z;
    uint kv_head = head / (n_heads / n_kv);
    uint kv_dim = n_kv * D;
    uint per = (D + 31) / 32;

    uint qpos = cache_start + qi + 1;
    uint attend = bidir ? cache_start + params[1] : qpos;
    uint lo = (window > 0 && qpos > window) ? qpos - window : 0;
    uint s0 = base + split * split_len;
    uint j0 = max(s0, lo);
    uint j1 = min(s0 + split_len, attend);

    float q[MAXV], acc[MAXV];
    float scale = rsqrt(float(D));
    device const float* qp = Q + (ulong)qi * n_heads * D + head * D;
    for (uint v = 0; v < per; v++) {
        uint d = v * 32 + lane;
        q[v] = d < D ? qp[d] * scale : 0.0f;
        acc[v] = 0;
    }
    float m = -INFINITY, l = 0;

    for (uint j = j0 + sg; j < j1; j += 8) {
        device const float* kp = K + (ulong)j * kv_dim + kv_head * D;
        float s = 0;
        for (uint v = 0; v < per; v++) { uint d = v * 32 + lane; if (d < D) s += q[v] * kp[d]; }
        s = simd_sum(s);
        float m2 = max(m, s);
        float c = exp(m - m2), p = exp(s - m2);
        l = l * c + p;
        device const float* vp = V + (ulong)j * kv_dim + kv_head * D;
        for (uint v = 0; v < per; v++) { uint d = v * 32 + lane; if (d < D) acc[v] = acc[v] * c + p * vp[d]; }
        m = m2;
    }

    // Merge the 8 simdgroups.
    threadgroup float sm[8], sl[8];
    threadgroup float sacc[8 * 256];
    if (lane == 0) { sm[sg] = m; sl[sg] = l; }
    for (uint v = 0; v < per; v++) { uint d = v * 32 + lane; if (d < D) sacc[sg * D + d] = acc[v]; }
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
        for (uint v = 0; v < per; v++) { uint d = v * 32 + lane; if (d < D) out[v] += sacc[g * D + d] * c; }
    }
    device float* pp = P + (((ulong)qi * n_heads + head) * n_splits + split) * (D + 2);
    for (uint v = 0; v < per; v++) { uint d = v * 32 + lane; if (d < D) pp[2 + d] = out[v]; }
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

/// Split-KV attention for a few queries (speculative-decoding verify, short incremental
/// prefills), `attn_multi_*`: threadgroup (kv head, split, row group). A threadgroup's rows are
/// (query, head) pairs sharing one KV head (row f -> query f / gqa, head kvh * gqa + f % gqa),
/// so each K/V row of the split is read once for all of them. Keys come in tiles of BK,
/// row-major in threadgroup memory (stride D + 4), with the next tile's K and V prefetched into
/// registers: scores with one key per lane, an online softmax per row in registers
/// (simdgroup w owns rows w*RW ..), then P·V with lanes over head dims. Writes unnormalized
/// partials for `FLASH_DECODE_MSL`'s `attn_combine` (same params). D % 4 == 0, D <= 256.
pub const FLASH_MULTI_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

template <int R, int BK, int DMAX, int NW>
inline void attn_multi_body(
    device const float* Q, device const float* K, device const float* V, device float* P,
    device const uint* params, threadgroup float* Qs, threadgroup float* KVs,
    threadgroup float* Ps, uint3 tg, uint tid, uint warp, uint lane)
{
    constexpr int RW = R / NW;             // rows per simdgroup
    constexpr int SEGS = 32 / BK;          // key segments per simdgroup
    constexpr int RPT = RW / SEGS;         // rows per lane in the score phase
    constexpr int NV = DMAX / 32;          // head dims per lane in the PV phase
    constexpr int PS = R + 4;              // Ps row stride
    constexpr int NT = 32 * NW;            // threads
    constexpr int NL = BK * DMAX / 4 / NT; // float4 loads per thread per tile

    uint cache_start = params[0], q_len = params[1], nh = params[2], nkv = params[3];
    uint D = params[4], n_splits = params[5], split_len = params[6];
    uint window = params[7], base = params[8], bidir = params[9];
    uint kvh = tg.x, split = tg.y, f0 = tg.z * R;
    uint gqa = nh / nkv, rows = q_len * gqa;
    uint LD = D + 4, D4 = D / 4;
    ulong kvd = (ulong)nkv * D;
    uint total = cache_start + q_len;
    float scale = rsqrt(float(D));

    for (uint i = tid; i < R * D4; i += NT) {
        uint r = i / D4, d = (i % D4) * 4, f = f0 + r;
        float4 v = 0.0f;
        if (f < rows) {
            uint qi = f / gqa, head = kvh * gqa + f % gqa;
            v = *(device const float4*)(Q + ((ulong)qi * nh + head) * D + d) * scale;
        }
        *(threadgroup float4*)(Qs + r * LD + d) = v;
    }

    uint f_last = min(f0 + R, rows) - 1;
    uint first = cache_start + f0 / gqa + 1;
    uint lo_blk = (window > 0 && first > window) ? first - window : 0;
    uint hi_blk = bidir ? total : cache_start + f_last / gqa + 1;
    uint s0 = base + split * split_len;
    uint jb = max(s0, lo_blk), je = min(s0 + split_len, hi_blk);

    uint c = lane % BK, seg = lane / BK;
    uint r0 = warp * RW + seg * RPT;
    uint lo[RPT], hi[RPT];
    float m[RPT], l[RPT];
    for (int i = 0; i < RPT; i++) {
        uint f = f0 + r0 + i;
        lo[i] = 0; hi[i] = 0;
        if (f < rows) {
            uint qpos = cache_start + f / gqa + 1;
            hi[i] = bidir ? total : qpos;
            lo[i] = (window > 0 && qpos > window) ? qpos - window : 0;
        }
        m[i] = -INFINITY; l[i] = 0.0f;
    }
    float acc[RW][NV];
    for (int i = 0; i < RW; i++)
        for (int v = 0; v < NV; v++) acc[i][v] = 0.0f;

    float4 kreg[NL], vreg[NL];
    if (jb < je) {
        for (int t = 0; t < NL; t++) {
            uint i = tid + NT * t, cc = i / D4, d = (i % D4) * 4, j = jb + cc;
            bool ok = cc < (uint)BK && j < je;
            ulong o = (ulong)j * kvd + (ulong)kvh * D + d;
            kreg[t] = ok ? *(device const float4*)(K + o) : float4(0.0f);
            vreg[t] = ok ? *(device const float4*)(V + o) : float4(0.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j0 = jb; j0 < je; j0 += BK) {
        bool more = j0 + BK < je;
        for (int t = 0; t < NL; t++) {
            uint i = tid + NT * t, cc = i / D4, d = (i % D4) * 4;
            if (cc < (uint)BK) *(threadgroup float4*)(KVs + cc * LD + d) = kreg[t];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (more) {
            for (int t = 0; t < NL; t++) {
                uint i = tid + NT * t, cc = i / D4, d = (i % D4) * 4, j = j0 + BK + cc;
                kreg[t] = (cc < (uint)BK && j < je)
                    ? *(device const float4*)(K + (ulong)j * kvd + (ulong)kvh * D + d) : float4(0.0f);
            }
        }

        float s[RPT];
        for (int i = 0; i < RPT; i++) s[i] = 0.0f;
        threadgroup const float* kr = KVs + c * LD;
        for (uint d = 0; d < D; d += 4) {
            float4 k4 = *(threadgroup const float4*)(kr + d);
            for (int i = 0; i < RPT; i++) s[i] += dot(*(threadgroup const float4*)(Qs + (r0 + i) * LD + d), k4);
        }
        uint j = j0 + c;
        float alpha[RPT];
        for (int i = 0; i < RPT; i++) {
            if (!(j < je && j >= lo[i] && j < hi[i])) s[i] = -INFINITY;
            float mx = s[i];
            for (ushort o = BK / 2; o > 0; o >>= 1) mx = max(mx, simd_shuffle_xor(mx, o));
            float mn = max(m[i], mx);
            float p = (s[i] == -INFINITY) ? 0.0f : exp(s[i] - mn);
            alpha[i] = (m[i] == -INFINITY) ? 0.0f : exp(m[i] - mn);
            float ps = p;
            for (ushort o = BK / 2; o > 0; o >>= 1) ps += simd_shuffle_xor(ps, o);
            l[i] = l[i] * alpha[i] + ps;
            m[i] = mn;
            Ps[c * PS + r0 + i] = p;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int t = 0; t < NL; t++) {
            uint i = tid + NT * t, cc = i / D4, d = (i % D4) * 4;
            if (cc < (uint)BK) *(threadgroup float4*)(KVs + cc * LD + d) = vreg[t];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (more) {
            for (int t = 0; t < NL; t++) {
                uint i = tid + NT * t, cc = i / D4, d = (i % D4) * 4, j = j0 + BK + cc;
                vreg[t] = (cc < (uint)BK && j < je)
                    ? *(device const float4*)(V + (ulong)j * kvd + (ulong)kvh * D + d) : float4(0.0f);
            }
        }

        for (int i = 0; i < RW; i++) {
            float a = simd_shuffle(alpha[i % RPT], (ushort)((i / RPT) * BK));
            for (int v = 0; v < NV; v++) acc[i][v] *= a;
        }
        threadgroup const float* pw = Ps + warp * RW;
        for (int cc = 0; cc < BK; cc++) {
            float p[RW];
            for (int i = 0; i < RW; i++) p[i] = pw[cc * PS + i];
            threadgroup const float* vr = KVs + cc * LD;
            for (int v = 0; v < NV; v++) {
                uint d = v * 32 + lane;
                float x = (d < D) ? vr[d] : 0.0f;
                for (int i = 0; i < RW; i++) acc[i][v] += p[i] * x;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int i = 0; i < RW; i++) {
        ushort src = (ushort)((i / RPT) * BK);
        float mi = simd_shuffle(m[i % RPT], src);
        float li = simd_shuffle(l[i % RPT], src);
        uint f = f0 + warp * RW + i;
        if (f >= rows) continue;
        uint qi = f / gqa, head = kvh * gqa + f % gqa;
        device float* pp = P + (((ulong)qi * nh + head) * n_splits + split) * (D + 2);
        for (int v = 0; v < NV; v++) {
            uint d = v * 32 + lane;
            if (d < D) pp[2 + d] = acc[i][v];
        }
        if (lane == 0) { pp[0] = mi; pp[1] = li; }
    }
}

#define ATTN_MULTI(NAME, R, BK, DMAX, NW)                                                    \
kernel void NAME(                                                                            \
    device const float* Q [[buffer(0)]], device const float* K [[buffer(1)]],                \
    device const float* V [[buffer(2)]], device float* P [[buffer(3)]],                      \
    device const uint* params [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],      \
    uint tid [[thread_index_in_threadgroup]], uint warp [[simdgroup_index_in_threadgroup]],  \
    uint lane [[thread_index_in_simdgroup]])                                                 \
{                                                                                            \
    threadgroup float Qs[R * (DMAX + 4)];                                                    \
    threadgroup float KVs[BK * (DMAX + 4)];                                                  \
    threadgroup float Ps[BK * (R + 4)];                                                      \
    attn_multi_body<R, BK, DMAX, NW>(Q, K, V, P, params, Qs, KVs, Ps, tg, tid, warp, lane); \
}

ATTN_MULTI(attn_multi_r8, 8, 32, 128, 8)
ATTN_MULTI(attn_multi_r16, 16, 32, 128, 8)
ATTN_MULTI(attn_multi_r32, 32, 16, 128, 8)
ATTN_MULTI(attn_multi_d256, 8, 16, 256, 4)
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

// A few rows at once (M <= 8, speculative-decoding verify and short prefills): like gemv_q4,
// but each lane widens its 16 weights of each of the 4 columns once and applies them to every
// row, so the weights are read and unpacked once for the whole batch. Needs K % 16 == 0.
// Dispatch: threadgroups = ceil(N / 8), 64 threads (2 simdgroups).
kernel void gemv_q4_rows(
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
    uint cols = min(4u, N - n0);
    uint G = K / group;
    uint words = K / 8;

    float acc[4][8];
    for (uint j = 0; j < 4; j++)
        for (uint r = 0; r < 8; r++) acc[j][r] = 0.0f;
    for (uint k = lane * 16; k < K; k += 512) {
        float4 w[4][4];
        float sc[4], bi[4];
        for (uint j = 0; j < 4; j++) {
            uint n = n0 + min(j, cols - 1);
            uint2 q = *(device const uint2*)(W + (ulong)n * words + k / 8);
            uint g = n * G + k / group;
            sc[j] = bf(S[g]);
            bi[j] = bf(B[g]);
            w[j][0] = float4(float(q.x & 0xf), float((q.x >> 4) & 0xf), float((q.x >> 8) & 0xf), float((q.x >> 12) & 0xf));
            w[j][1] = float4(float((q.x >> 16) & 0xf), float((q.x >> 20) & 0xf), float((q.x >> 24) & 0xf), float(q.x >> 28));
            w[j][2] = float4(float(q.y & 0xf), float((q.y >> 4) & 0xf), float((q.y >> 8) & 0xf), float((q.y >> 12) & 0xf));
            w[j][3] = float4(float((q.y >> 16) & 0xf), float((q.y >> 20) & 0xf), float((q.y >> 24) & 0xf), float(q.y >> 28));
        }
        for (uint r = 0; r < 8; r++) {
            if (r >= M) break;
            device const float4* xp = (device const float4*)(X + (ulong)r * K + k);
            float4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
            float4 xs4 = x0 + x1 + x2 + x3;
            float xs = xs4.x + xs4.y + xs4.z + xs4.w;
            for (uint j = 0; j < 4; j++) {
                float d = dot(w[j][0], x0) + dot(w[j][1], x1) + dot(w[j][2], x2) + dot(w[j][3], x3);
                acc[j][r] += sc[j] * d + bi[j] * xs;
            }
        }
    }
    for (uint r = 0; r < M; r++) {
        for (uint j = 0; j < cols; j++) {
            float v = simd_sum(acc[j][r]);
            if (lane == 0) Y[(ulong)r * N + n0 + j] = v;
        }
    }
}

// Small batches (2..32 rows) with simdgroup matrices: threadgroup = 8 simdgroups over 128
// output columns (simdgroup s owns column blocks 2s, 2s+1) and every row (RB blocks of 8).
// K goes in 32-deep steps: the weight tile is widened into threadgroup memory once per step
// and shared by all rows, X is staged beside it, and each thread's global loads for the steps
// ahead (QP of them) are already in flight while the current step multiplies.
// With a K split (grid y > 1) each split writes its own [M, N] slice for `sum_splits`.
// Needs K % 32 == 0, group % 32 == 0. params: [M, K, N, group, steps per split].
// Dispatch: threadgroups (ceil(N / 128), splits), 256 threads.
#define QS_BN 128
#define QS_BK 32
#define QS_LD 36
#define QP 2

template <int RB>
inline void qmm_small_body(
    device const float* X, device const uint* W, device const ushort* S, device const ushort* B,
    device float* Y, uint M, uint K, uint N, uint group, uint k_steps, uint2 tg, uint tid, uint sg,
    threadgroup float* Ws, threadgroup float* Xs)
{
    // K split tg.y covers steps [tg.y * k_steps, ..) and writes its own [M, N] slice of Y.
    uint n0 = tg.x * QS_BN;
    uint st0 = tg.y * k_steps;
    Y += (ulong)tg.y * M * N;
    uint wc = tid >> 1, part = tid & 1, wn = n0 + wc;
    uint words = K / 8, G = K / group;
    uint steps = min(K / QS_BK - st0, k_steps);
    // X loader: rows tid / 32 + 8 i, column tid % 32.
    uint xk = tid & 31, xr = tid >> 5;

    uint2 wq[QP];
    float xv[QP][RB], wsc[QP], wbi[QP];
    auto fetch = [&](uint slot, uint st) {
        uint k0 = (st0 + st) * QS_BK;
        bool ok = wn < N && st < steps;
        wq[slot] = ok ? *(device const uint2*)(W + (ulong)wn * words + k0 / 8 + part * 2) : uint2(0);
        uint g = wn * G + (k0 + part * 16) / group;
        wsc[slot] = ok ? bf(S[g]) : 0.0f;
        wbi[slot] = ok ? bf(B[g]) : 0.0f;
        for (int i = 0; i < RB; i++) {
            uint r = xr + 8 * i;
            xv[slot][i] = (r < M && st < steps) ? X[(ulong)r * K + k0 + xk] : 0.0f;
        }
    };
    simdgroup_float8x8 acc[RB][2];
    for (int i = 0; i < RB; i++) { acc[i][0] = simdgroup_float8x8(0); acc[i][1] = simdgroup_float8x8(0); }

    for (uint p = 0; p < QP; p++) fetch(p, p);
    for (uint st = 0; st < steps; st++) {
        uint slot = st % QP;
        {
            float sc = wsc[slot], bi = wbi[slot];
            uint2 q = wq[slot];
            threadgroup float4* dst = (threadgroup float4*)(Ws + wc * QS_LD + part * 16);
            dst[0] = float4(float(q.x & 0xf), float((q.x >> 4) & 0xf), float((q.x >> 8) & 0xf), float((q.x >> 12) & 0xf)) * sc + bi;
            dst[1] = float4(float((q.x >> 16) & 0xf), float((q.x >> 20) & 0xf), float((q.x >> 24) & 0xf), float(q.x >> 28)) * sc + bi;
            dst[2] = float4(float(q.y & 0xf), float((q.y >> 4) & 0xf), float((q.y >> 8) & 0xf), float((q.y >> 12) & 0xf)) * sc + bi;
            dst[3] = float4(float((q.y >> 16) & 0xf), float((q.y >> 20) & 0xf), float((q.y >> 24) & 0xf), float(q.y >> 28)) * sc + bi;
            for (int i = 0; i < RB; i++) Xs[(xr + 8 * i) * QS_LD + xk] = xv[slot][i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        fetch(slot, st + QP);
        for (uint ks = 0; ks < QS_BK; ks += 8) {
            simdgroup_float8x8 b0, b1;
            simdgroup_load(b0, Ws + (2 * sg) * 8 * QS_LD + ks, QS_LD, ulong2(0), true);
            simdgroup_load(b1, Ws + (2 * sg + 1) * 8 * QS_LD + ks, QS_LD, ulong2(0), true);
            for (int i = 0; i < RB; i++) {
                simdgroup_float8x8 a;
                simdgroup_load(a, Xs + i * 8 * QS_LD + ks, QS_LD);
                simdgroup_multiply_accumulate(acc[i][0], a, b0, acc[i][0]);
                simdgroup_multiply_accumulate(acc[i][1], a, b1, acc[i][1]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Through threadgroup memory ([RB * 8][128]) so partial tiles can be clipped.
    for (int i = 0; i < RB; i++) {
        simdgroup_store(acc[i][0], Ws + i * 8 * QS_BN + (2 * sg) * 8, QS_BN);
        simdgroup_store(acc[i][1], Ws + i * 8 * QS_BN + (2 * sg + 1) * 8, QS_BN);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < RB * 8 * QS_BN; e += 256) {
        uint r = e / QS_BN, c = e % QS_BN;
        if (r < M && n0 + c < N) Y[(ulong)r * N + n0 + c] = Ws[e];
    }
}

#define QMM_SMALL(NAME, RB)                                                                   \
kernel void NAME(                                                                             \
    device const float* X [[buffer(0)]], device const uint* W [[buffer(1)]],                  \
    device const ushort* S [[buffer(2)]], device const ushort* B [[buffer(3)]],               \
    device float* Y [[buffer(4)]], device const uint* params [[buffer(5)]],                   \
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],      \
    uint sg [[simdgroup_index_in_threadgroup]])                                               \
{                                                                                             \
    threadgroup float Ws[QS_BN * QS_LD];                                                      \
    threadgroup float Xs[RB * 8 * QS_LD];                                                     \
    qmm_small_body<RB>(X, W, S, B, Y, params[0], params[1], params[2], params[3], params[4],  \
                       tg, tid, sg, Ws, Xs);                                                  \
}

QMM_SMALL(qmm_small_8, 1)
QMM_SMALL(qmm_small_16, 2)
QMM_SMALL(qmm_small_32, 4)

// Y[i] = sum over splits of P[s * n + i]. params: [n, splits].
kernel void sum_splits(
    device const float* P [[buffer(0)]], device float* Y [[buffer(1)]],
    device const uint* params [[buffer(2)]], uint gid [[thread_position_in_grid]])
{
    uint n = params[0], splits = params[1];
    if (gid >= n) return;
    float s = 0.0f;
    for (uint k = 0; k < splits; k++) s += P[(ulong)k * n + gid];
    Y[gid] = s;
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

// GELU (tanh approximation), elementwise.
kernel void gelu_tanh(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    device const uint* params [[buffer(2)]],  // [n]
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params[0]) return;
    float g = X[gid];
    float t = tanh(clamp(0.7978845608f * (g + 0.044715f * g * g * g), -10.0f, 10.0f));
    Y[gid] = 0.5f * g * (1.0f + t);
}

// LayerNorm with weight and bias: one threadgroup (256 threads) per row.
kernel void layer_norm(
    device const float* X [[buffer(0)]],
    device const float* W [[buffer(1)]],
    device const float* B [[buffer(2)]],
    device float* Y [[buffer(3)]],
    device const float* eps [[buffer(4)]],
    device const uint* params [[buffer(5)]],  // [rows, dim]
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint dim = params[1];
    device const float* x = X + (ulong)row * dim;
    threadgroup float red[8];
    float s = 0;
    for (uint i = tid; i < dim; i += 256) s += x[i];
    s = simd_sum(s);
    if (lane == 0) red[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = 0;
    for (uint g = 0; g < 8; g++) mean += red[g];
    mean /= float(dim);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float v = 0;
    for (uint i = tid; i < dim; i += 256) { float d = x[i] - mean; v += d * d; }
    v = simd_sum(v);
    if (lane == 0) red[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float var = 0;
    for (uint g = 0; g < 8; g++) var += red[g];
    float inv = rsqrt(var / float(dim) + eps[0]);
    for (uint i = tid; i < dim; i += 256) Y[(ulong)row * dim + i] = (x[i] - mean) * inv * W[i] + B[i];
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
    device const uint* params [[buffer(4)]], // [cache_start, q_len, nh, nkv, D, bidirectional]
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint cache_start = params[0], q_len = params[1], nh = params[2], nkv = params[3], D = params[4];
    bool bidir = params[5] != 0;
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
    uint end = cache_start + q_len - 1;          // last key in the cache
    // The last key this row may see: itself (causal) or the batch's end (bidirectional).
    uint abs_row = bidir ? end : cache_start + r0 + row;
    uint last = bidir ? end : min(cache_start + r0 + 7, end);  // last position any row here needs

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
