//! CUDA kernels for autoregressive LLM inference, mirroring `llm_msl`: decode GEMVs over bf16
//! and MLX 4-bit weights, weight dequantization for prefill GEMMs, embedding lookups, the fused
//! attention prologue, activations, LayerNorm, and split-KV / tiled attention with sliding
//! windows and bidirectional blocks. Activations are f32 throughout.

/// Prepends the helpers every kernel source below shares.
macro_rules! with_common {
    ($body:literal) => {
        concat!(
            r#"
__device__ __forceinline__ float bf(unsigned short b) {
    return __uint_as_float(((unsigned int)b) << 16);
}
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
#define NEG_INF __uint_as_float(0xff800000u)
typedef unsigned long long u64;
"#,
            $body
        )
    };
}

/// `y[r, n] = sum_k x[r, k] * W[n, k]` for a handful of rows (`M <= 8`), W stored `[N, K]` in
/// bf16 (`gemv_bf16`) or MLX 4-bit groups (`gemv_q4`: packed `[N, K/8]` u32, low nibble first,
/// bf16 scales and biases `[N, K/group]`, `w = scale * q + bias`).
///
/// Decode is memory-bound: one warp per output column reads its weight row once, coalesced,
/// and applies it to all `M` rows. Launch: grid ceil(N / 8), block 256 (8 warps).
pub const GEMV_CUDA: &str = with_common!(
    r#"
extern "C" __global__ void gemv_bf16(
    const float* __restrict__ X, const unsigned short* __restrict__ W, float* __restrict__ Y,
    unsigned int M, unsigned int K, unsigned int N)
{
    unsigned int lane = threadIdx.x & 31;
    unsigned int n = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (n >= N) return;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const unsigned short* w = W + (u64)n * K;
    if ((K & 7) == 0) {
        const uint4* w8 = (const uint4*)w;
        for (unsigned int i = lane; i < K / 8; i += 32) {
            uint4 p = w8[i];
            float wv[8] = {bf(p.x & 0xffff), bf(p.x >> 16), bf(p.y & 0xffff), bf(p.y >> 16),
                           bf(p.z & 0xffff), bf(p.z >> 16), bf(p.w & 0xffff), bf(p.w >> 16)};
            #pragma unroll
            for (unsigned int r = 0; r < 8; r++) {
                if (r < M) {
                    const float4* x4 = (const float4*)(X + (u64)r * K + i * 8);
                    float4 a = x4[0], b = x4[1];
                    acc[r] += wv[0] * a.x + wv[1] * a.y + wv[2] * a.z + wv[3] * a.w
                            + wv[4] * b.x + wv[5] * b.y + wv[6] * b.z + wv[7] * b.w;
                }
            }
        }
    } else {
        for (unsigned int i = lane; i < K; i += 32) {
            float wv = bf(w[i]);
            #pragma unroll
            for (unsigned int r = 0; r < 8; r++) if (r < M) acc[r] += wv * X[(u64)r * K + i];
        }
    }
    #pragma unroll
    for (unsigned int r = 0; r < 8; r++) {
        if (r < M) {
            float s = warp_sum(acc[r]);
            if (lane == 0) Y[(u64)r * N + n] = s;
        }
    }
}

// Each lane takes one packed word (8 weights) per step; K % 8 == 0 and group % 8 == 0.
extern "C" __global__ void gemv_q4(
    const float* __restrict__ X, const unsigned int* __restrict__ Wq,
    const unsigned short* __restrict__ S, const unsigned short* __restrict__ B,
    float* __restrict__ Y, unsigned int M, unsigned int K, unsigned int N, unsigned int group)
{
    unsigned int lane = threadIdx.x & 31;
    unsigned int n = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (n >= N) return;
    unsigned int words = K / 8, G = K / group;
    const unsigned int* w = Wq + (u64)n * words;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (unsigned int i = lane; i < words; i += 32) {
        unsigned int q = w[i];
        u64 g = (u64)n * G + (i * 8) / group;
        float s = bf(S[g]), b = bf(B[g]);
        float qf[8];
        #pragma unroll
        for (int e = 0; e < 8; e++) qf[e] = (float)((q >> (4 * e)) & 0xf);
        #pragma unroll
        for (unsigned int r = 0; r < 8; r++) {
            if (r < M) {
                const float4* x4 = (const float4*)(X + (u64)r * K + i * 8);
                float4 a = x4[0], c = x4[1];
                float d = qf[0] * a.x + qf[1] * a.y + qf[2] * a.z + qf[3] * a.w
                        + qf[4] * c.x + qf[5] * c.y + qf[6] * c.z + qf[7] * c.w;
                float xs = (a.x + a.y + a.z + a.w) + (c.x + c.y + c.z + c.w);
                acc[r] += s * d + b * xs;
            }
        }
    }
    #pragma unroll
    for (unsigned int r = 0; r < 8; r++) {
        if (r < M) {
            float v = warp_sum(acc[r]);
            if (lane == 0) Y[(u64)r * N + n] = v;
        }
    }
}

// A few rows (2..32: speculative-decoding verify, short incremental prefills), one output
// column per thread: each thread streams its column's weights (64 K per step, prefetched a
// step ahead with the step's activations: 128 K, 64 or 256 bytes per column), widens each weight once, and applies it to every row,
// whose activations come from shared memory as broadcasts (Xs[k][row]). Rows are padded to
// MR (8, 16 or 32). A K split (gridDim.y > 1) writes its own [M, N] slice of Y for
// `sum_splits`. Needs K % 128 == 0 (and group % 64 == 0 for 4-bit).
// Launch: grid (ceil(N / 256), splits), block 256.
template <int MR, bool IS_Q4, int KC>
__device__ __forceinline__ void gemm_cols_body(
    const float* __restrict__ X, const unsigned int* __restrict__ Wq,
    const unsigned short* __restrict__ S, const unsigned short* __restrict__ B,
    const unsigned short* __restrict__ Wb, float* __restrict__ Y,
    unsigned int M, unsigned int K, unsigned int N, unsigned int group, unsigned int k_split)
{
    __shared__ __align__(16) float Xs[KC * MR];
    constexpr int NW = IS_Q4 ? KC / 32 : KC / 8;   // uint4 per column per step
    unsigned int tid = threadIdx.x, n = blockIdx.x * 256 + tid;
    unsigned int kb = blockIdx.y * k_split, ke = min(kb + k_split, K);
    Y += (u64)blockIdx.y * M * N;
    unsigned int nc = min(n, N - 1);
    unsigned int G = K / group;

    uint4 wr[NW];
    float sc[NW / 2 + 1], bi[NW / 2 + 1];
    auto fetch = [&](unsigned int k0) {
        #pragma unroll
        for (int u = 0; u < NW; u++)
            wr[u] = IS_Q4 ? *(const uint4*)(Wq + (u64)nc * (K / 8) + k0 / 8 + u * 4)
                          : *(const uint4*)(Wb + (u64)nc * K + k0 + u * 8);
        if (IS_Q4) {
            #pragma unroll
            for (int u = 0; u < NW / 2; u++) {
                u64 g = (u64)nc * G + (k0 + 64 * u) / group;
                sc[u] = bf(S[g]); bi[u] = bf(B[g]);
            }
        }
    };
    float acc[MR];
    #pragma unroll
    for (int r = 0; r < MR; r++) acc[r] = 0.0f;

    // X loader: element tid + 256 * i of the [MR, KC] chunk.
    constexpr int NX = KC * MR / 256;
    float xr[NX];
    auto fetch_x = [&](unsigned int k0) {
        #pragma unroll
        for (int i = 0; i < NX; i++) {
            unsigned int e = tid + 256 * i, r = e / KC, kk = e % KC;
            xr[i] = (r < M) ? X[(u64)r * K + k0 + kk] : 0.0f;
        }
    };
    if (kb < ke) { fetch(kb); fetch_x(kb); }
    for (unsigned int k0 = kb; k0 < ke; k0 += KC) {
        #pragma unroll
        for (int i = 0; i < NX; i++) {
            unsigned int e = tid + 256 * i, r = e / KC, kk = e % KC;
            Xs[kk * MR + r] = xr[i];
        }
        uint4 w4[NW];
        #pragma unroll
        for (int u = 0; u < NW; u++) w4[u] = wr[u];
        float s4[NW / 2 + 1], b4[NW / 2 + 1];
        #pragma unroll
        for (int u = 0; u < NW / 2 + 1; u++) { s4[u] = sc[u]; b4[u] = bi[u]; }
        __syncthreads();
        if (k0 + KC < ke) { fetch(k0 + KC); fetch_x(k0 + KC); }
        #pragma unroll
        for (int u = 0; u < NW; u++) {
            unsigned int wv[4] = {w4[u].x, w4[u].y, w4[u].z, w4[u].w};
            #pragma unroll
            for (int e = 0; e < (IS_Q4 ? 32 : 8); e++) {
                float w;
                if (IS_Q4) w = fmaf((float)((wv[e / 8] >> (4 * (e % 8))) & 0xf), s4[u / 2], b4[u / 2]);
                else w = bf((unsigned short)(wv[e / 2] >> (16 * (e % 2))));
                const float4* xr = (const float4*)(Xs + (u * (IS_Q4 ? 32 : 8) + e) * MR);
                #pragma unroll
                for (int r4 = 0; r4 < MR / 4; r4++) {
                    float4 x = xr[r4];
                    acc[4 * r4] += w * x.x;
                    acc[4 * r4 + 1] += w * x.y;
                    acc[4 * r4 + 2] += w * x.z;
                    acc[4 * r4 + 3] += w * x.w;
                }
            }
        }
        __syncthreads();
    }
    if (n < N) {
        #pragma unroll
        for (int r = 0; r < MR; r++)
            if (r < M) Y[(u64)r * N + n] = acc[r];
    }
}

#define GEMM_COLS(NAME, MR)                                                                   \
extern "C" __global__ void __launch_bounds__(256) NAME##_q4(                                  \
    const float* __restrict__ X, const unsigned int* __restrict__ Wq,                         \
    const unsigned short* __restrict__ S, const unsigned short* __restrict__ B,               \
    float* __restrict__ Y, unsigned int M, unsigned int K, unsigned int N, unsigned int group, \
    unsigned int k_split)                                                                     \
{                                                                                             \
    gemm_cols_body<MR, true, 128>(X, Wq, S, B, nullptr, Y, M, K, N, group, k_split);               \
}                                                                                             \
extern "C" __global__ void __launch_bounds__(256) NAME##_bf16(                                \
    const float* __restrict__ X, const unsigned short* __restrict__ W, float* __restrict__ Y, \
    unsigned int M, unsigned int K, unsigned int N, unsigned int k_split)                     \
{                                                                                             \
    gemm_cols_body<MR, false, 128>(X, nullptr, nullptr, nullptr, W, Y, M, K, N, 64, k_split);      \
}

GEMM_COLS(gemm_cols8, 8)
GEMM_COLS(gemm_cols16, 16)
GEMM_COLS(gemm_cols32, 32)

// Small-batch GEMM (9..32 rows: verify steps, short incremental prefills) straight off packed
// weights: block (128 output columns, K split), 256 threads, 64-deep K chunks. The weight tile
// is widened into shared memory once per chunk and shared by every row, X is staged transposed;
// each thread owns 4 rows x 4 columns (rows (tid / 32) * 4 .., columns tid % 32 + 32 * i).
// The next chunk's global loads are issued before the current chunk's math. With a K split
// (gridDim.y > 1) each split writes its own `[M, N]` slice of Y for `sum_splits`.
// Needs M <= 32 and K % 64 == 0 (and group % 8 == 0 for 4-bit).
#define GS_BN 128
#define GS_BK 64
#define GS_XS 36
template <bool IS_Q4>
__device__ __forceinline__ void gemm_small_body(
    const float* __restrict__ X, const unsigned int* __restrict__ Wq,
    const unsigned short* __restrict__ S, const unsigned short* __restrict__ B,
    const unsigned short* __restrict__ Wb, float* __restrict__ Y,
    unsigned int M, unsigned int K, unsigned int N, unsigned int group, unsigned int k_split)
{
    __shared__ __align__(16) float Ws[GS_BK * GS_BN];
    __shared__ __align__(16) float Xs[GS_BK * GS_XS];
    unsigned int tid = threadIdx.x;
    unsigned int n0 = blockIdx.x * GS_BN;
    unsigned int kb = blockIdx.y * k_split, ke = min(kb + k_split, K);
    Y += (u64)blockIdx.y * M * N;

    // Loader roles: weight column wc (two threads per column, 32 k each), X column xk.
    unsigned int wc = tid >> 1, part = tid & 1, wn = n0 + wc;
    unsigned int xk = tid & 63, xr = (tid >> 6) * 8;
    uint4 wreg[IS_Q4 ? 1 : 4];
    float xreg[8];

    auto fetch = [&](unsigned int k0) {
        if (IS_Q4) {
            wreg[0] = (wn < N) ? *(const uint4*)(Wq + (u64)wn * (K / 8) + k0 / 8 + part * 4)
                               : make_uint4(0, 0, 0, 0);
        } else {
            #pragma unroll
            for (int u = 0; u < (IS_Q4 ? 1 : 4); u++)
                wreg[u] = (wn < N) ? *(const uint4*)(Wb + (u64)wn * K + k0 + part * 32 + u * 8)
                                   : make_uint4(0, 0, 0, 0);
        }
        #pragma unroll
        for (int i = 0; i < 8; i++)
            xreg[i] = (xr + i < M) ? X[(u64)(xr + i) * K + k0 + xk] : 0.0f;
    };
    auto stage = [&](unsigned int k0) {
        if (IS_Q4) {
            unsigned int w4[4] = {wreg[0].x, wreg[0].y, wreg[0].z, wreg[0].w};
            #pragma unroll
            for (int u = 0; u < 4; u++) {
                unsigned int kk = part * 32 + u * 8;
                float sc = 0.0f, bi = 0.0f;
                if (wn < N) {
                    u64 g = (u64)wn * (K / group) + (k0 + kk) / group;
                    sc = bf(S[g]); bi = bf(B[g]);
                }
                #pragma unroll
                for (int e = 0; e < 8; e++)
                    Ws[(kk + e) * GS_BN + wc] = sc * (float)((w4[u] >> (4 * e)) & 0xf) + bi;
            }
        } else {
            #pragma unroll
            for (int u = 0; u < (IS_Q4 ? 1 : 4); u++) {
                unsigned int w4[4] = {wreg[u].x, wreg[u].y, wreg[u].z, wreg[u].w};
                unsigned int kk = part * 32 + u * 8;
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    Ws[(kk + 2 * e) * GS_BN + wc] = bf(w4[e] & 0xffff);
                    Ws[(kk + 2 * e + 1) * GS_BN + wc] = bf(w4[e] >> 16);
                }
            }
        }
        #pragma unroll
        for (int i = 0; i < 8; i++) Xs[xk * GS_XS + xr + i] = xreg[i];
    };

    unsigned int tc = tid & 31, tr = (tid >> 5) * 4;
    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;

    if (kb < ke) fetch(kb);
    for (unsigned int k0 = kb; k0 < ke; k0 += GS_BK) {
        stage(k0);
        __syncthreads();
        if (k0 + GS_BK < ke) fetch(k0 + GS_BK);
        #pragma unroll 8
        for (int kk = 0; kk < GS_BK; kk++) {
            float4 x = *(const float4*)(Xs + kk * GS_XS + tr);
            const float* w = Ws + kk * GS_BN + tc;
            float wv[4] = {w[0], w[32], w[64], w[96]};
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                acc[0][j] += x.x * wv[j];
                acc[1][j] += x.y * wv[j];
                acc[2][j] += x.z * wv[j];
                acc[3][j] += x.w * wv[j];
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        if (tr + i >= M) break;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            unsigned int n = n0 + tc + 32 * j;
            if (n < N) Y[(u64)(tr + i) * N + n] = acc[i][j];
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) gemm_small_q4(
    const float* __restrict__ X, const unsigned int* __restrict__ Wq,
    const unsigned short* __restrict__ S, const unsigned short* __restrict__ B,
    float* __restrict__ Y, unsigned int M, unsigned int K, unsigned int N, unsigned int group,
    unsigned int k_split)
{
    gemm_small_body<true>(X, Wq, S, B, nullptr, Y, M, K, N, group, k_split);
}

extern "C" __global__ void __launch_bounds__(256) gemm_small_bf16(
    const float* __restrict__ X, const unsigned short* __restrict__ W, float* __restrict__ Y,
    unsigned int M, unsigned int K, unsigned int N, unsigned int k_split)
{
    gemm_small_body<false>(X, nullptr, nullptr, nullptr, W, Y, M, K, N, 8, k_split);
}

// Y[i] = sum over splits of P[s * n + i].
extern "C" __global__ void sum_splits(
    const float* __restrict__ P, float* __restrict__ Y, unsigned int n, unsigned int splits)
{
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = 0.0f;
    for (unsigned int k = 0; k < splits; k++) s += P[(u64)k * n + i];
    Y[i] = s;
}
"#
);

/// Widen rows `row0 .. row0 + rows` of a `[N, K]` weight to f32 (`[rows, K]`), for prefill
/// GEMMs through cuBLAS. `dequant_bf16`: one thread per element; `dequant_q4`: one thread per
/// packed word. Launch: 256 threads, enough blocks to cover the range.
pub const DEQUANT_CUDA: &str = with_common!(
    r#"
extern "C" __global__ void dequant_bf16(
    const unsigned short* __restrict__ W, float* __restrict__ Y,
    unsigned int row0, unsigned int rows, unsigned int K)
{
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)rows * K) return;
    Y[i] = bf(W[(u64)row0 * K + i]);
}

extern "C" __global__ void dequant_q4(
    const unsigned int* __restrict__ Wq, const unsigned short* __restrict__ S,
    const unsigned short* __restrict__ B, float* __restrict__ Y,
    unsigned int row0, unsigned int rows, unsigned int K, unsigned int group)
{
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;   // word within the chunk
    unsigned int words = K / 8;
    if (i >= (u64)rows * words) return;
    u64 r = (u64)row0 + i / words;
    unsigned int k = (unsigned int)(i % words) * 8;
    unsigned int q = Wq[r * words + k / 8];
    u64 g = r * (K / group) + k / group;
    float s = bf(S[g]), b = bf(B[g]);
    float* y = Y + i * 8;
    #pragma unroll
    for (int e = 0; e < 8; e++) y[e] = s * (float)((q >> (4 * e)) & 0xf) + b;
}
"#
);

/// Embedding lookups from bf16 and 4-bit tables into f32 rows. `ids` are u32 bit patterns.
/// Launch: one thread per output element.
pub const EMBED_CUDA: &str = with_common!(
    r#"
extern "C" __global__ void embedding_bf16(
    const unsigned short* __restrict__ W, const unsigned int* __restrict__ ids,
    float* __restrict__ Y, unsigned int seq_len, unsigned int dim)
{
    u64 gid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (u64)seq_len * dim) return;
    u64 s = gid / dim, d = gid % dim;
    Y[gid] = bf(W[(u64)ids[s] * dim + d]);
}

extern "C" __global__ void embedding_q4(
    const unsigned int* __restrict__ Wq, const unsigned short* __restrict__ S,
    const unsigned short* __restrict__ B, const unsigned int* __restrict__ ids,
    float* __restrict__ Y, unsigned int seq_len, unsigned int dim, unsigned int group)
{
    u64 gid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (u64)seq_len * dim) return;
    u64 s = gid / dim;
    unsigned int d = (unsigned int)(gid % dim);
    u64 row = ids[s];
    unsigned int q = (Wq[row * (dim / 8) + d / 8] >> (4 * (d % 8))) & 0xf;
    u64 g = row * (dim / group) + d / group;
    Y[gid] = bf(S[g]) * (float)q + bf(B[g]);
}
"#
);

/// Decoder-layer glue: fused attention prologue, half-split RoPE, activations, residual add,
/// LayerNorm.
///
/// `attention_prep`: one warp per (head, token) over the q, k and v heads of a fused QKV row
/// (grid `(nh + 2*nkv, seq)`, block 32). q/k heads get the optional per-head RMS norm and
/// half-split RoPE at `pos + s`; q goes to `Qo`, k and v straight into the caches at `pos + s`.
/// Any even head dim up to 256.
pub const FUSED_CUDA: &str = with_common!(
    r#"
extern "C" __global__ void attention_prep(
    const float* __restrict__ QKV, const float* __restrict__ QN, const float* __restrict__ KN,
    const float* __restrict__ COS, const float* __restrict__ SIN,
    float* __restrict__ Qo, float* __restrict__ KC, float* __restrict__ VC,
    unsigned int nh, unsigned int nkv, unsigned int hd, unsigned int pos, float eps,
    unsigned int has_qn, unsigned int has_kn)
{
    __shared__ float xs[256];
    unsigned int h = blockIdx.x, s = blockIdx.y, lane = threadIdx.x;
    unsigned int qd = nh * hd, kvd = nkv * hd, row = qd + 2 * kvd, half = hd / 2;
    const float* src = QKV + (u64)s * row + (u64)h * hd;  // heads are contiguous: q.., k.., v..

    if (h >= nh + nkv) {  // v head: straight into the cache
        float* dst = VC + (u64)(pos + s) * kvd + (u64)(h - nh - nkv) * hd;
        for (unsigned int d = lane; d < hd; d += 32) dst[d] = src[d];
        return;
    }
    bool is_q = h < nh;
    bool has_norm = is_q ? has_qn != 0 : has_kn != 0;
    float ss = 0.0f;
    for (unsigned int d = lane; d < hd; d += 32) { float v = src[d]; xs[d] = v; ss += v * v; }
    if (has_norm) {
        const float* w = is_q ? QN : KN;
        float r = rsqrtf(warp_sum(ss) / (float)hd + eps);
        for (unsigned int d = lane; d < hd; d += 32) xs[d] *= r * w[d];
    }
    __syncwarp();
    u64 t = (u64)(pos + s) * half;
    float* dst = is_q ? Qo + (u64)s * qd + (u64)h * hd
                      : KC + (u64)(pos + s) * kvd + (u64)(h - nh) * hd;
    for (unsigned int i = lane; i < half; i += 32) {
        float c = COS[t + i], sn = SIN[t + i];
        float x0 = xs[i], x1 = xs[i + half];
        dst[i] = x0 * c - x1 * sn;
        dst[i + half] = x1 * c + x0 * sn;
    }
}

// Half-split RoPE: rotates (x[i], x[i + half]) by the angle for start_pos + s. One thread per
// (s, h, i < half).
extern "C" __global__ void rope_half(
    const float* __restrict__ X, const float* __restrict__ COS, const float* __restrict__ SIN,
    float* __restrict__ Y, unsigned int seq_len, unsigned int n_heads, unsigned int head_dim,
    unsigned int start_pos)
{
    unsigned int half = head_dim / 2;
    u64 gid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (u64)seq_len * n_heads * half) return;
    u64 i = gid % half, sh = gid / half, s = sh / n_heads;
    u64 base = sh * head_dim;
    u64 t = (start_pos + s) * half + i;
    float c = COS[t], sn = SIN[t];
    float x0 = X[base + i], x1 = X[base + i + half];
    Y[base + i] = x0 * c - x1 * sn;
    Y[base + i + half] = x1 * c + x0 * sn;
}

extern "C" __global__ void add_f32(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ Y, unsigned int n)
{
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) Y[i] = A[i] + B[i];
}

// GELU, tanh approximation. The tanh argument is clamped as on Metal (it's ±1 well before ±10).
__device__ __forceinline__ float gelu(float g) {
    float t = tanhf(fminf(fmaxf(0.7978845608f * (g + 0.044715f * g * g * g), -10.0f), 10.0f));
    return 0.5f * g * (1.0f + t);
}

extern "C" __global__ void gelu_tanh(const float* __restrict__ X, float* __restrict__ Y, unsigned int n)
{
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) Y[i] = gelu(X[i]);
}

// Over a fused gate/up projection `[rows, 2*ff]`: gelu_tanh(gate) * up and silu(gate) * up.
extern "C" __global__ void geglu_split(
    const float* __restrict__ GU, float* __restrict__ Y, unsigned int rows, unsigned int ff)
{
    u64 gid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (u64)rows * ff) return;
    u64 r = gid / ff, i = gid % ff;
    Y[gid] = gelu(GU[r * 2 * ff + i]) * GU[r * 2 * ff + ff + i];
}

extern "C" __global__ void swiglu_split(
    const float* __restrict__ GU, float* __restrict__ Y, unsigned int rows, unsigned int ff)
{
    u64 gid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (u64)rows * ff) return;
    u64 r = gid / ff, i = gid % ff;
    float g = GU[r * 2 * ff + i];
    Y[gid] = g / (1.0f + expf(-g)) * GU[r * 2 * ff + ff + i];
}

// LayerNorm with weight and bias: one block of 256 threads per row.
extern "C" __global__ void layer_norm(
    const float* __restrict__ X, const float* __restrict__ W, const float* __restrict__ B,
    float* __restrict__ Y, unsigned int dim, float eps)
{
    __shared__ float red[8];
    unsigned int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const float* x = X + (u64)blockIdx.x * dim;
    float s = 0.0f;
    for (unsigned int i = tid; i < dim; i += 256) s += x[i];
    s = warp_sum(s);
    if (lane == 0) red[warp] = s;
    __syncthreads();
    float mean = 0.0f;
    for (int g = 0; g < 8; g++) mean += red[g];
    mean /= (float)dim;
    __syncthreads();
    float v = 0.0f;
    for (unsigned int i = tid; i < dim; i += 256) { float d = x[i] - mean; v += d * d; }
    v = warp_sum(v);
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float var = 0.0f;
    for (int g = 0; g < 8; g++) var += red[g];
    float inv = rsqrtf(var / (float)dim + eps);
    float* y = Y + (u64)blockIdx.x * dim;
    for (unsigned int i = tid; i < dim; i += 256) y[i] = (x[i] - mean) * inv * W[i] + B[i];
}
"#
);

/// Attention over a KV cache: q/out `[q_len, nh*D]`, K/V `[pos, nkv*D]` (GQA: `nh % nkv == 0`).
/// Query `qi` sits at position `cache_start + qi` and sees keys `0 ..= cache_start + qi`, or
/// with a sliding window (`window > 0`) only the last `window` of those; with `bidir` it sees
/// keys up to `cache_start + q_len` instead (image tokens, vision encoders). D at most 256.
///
/// Split-KV ("flash decoding"), for decode and short batches:
/// `attn_partial`: grid (head, split, qi), block 256 (8 warps). Each warp runs an online softmax
/// over its share of the split's keys (lanes hold D/32 dims; a score is one warp sum), then the
/// warps merge and write (max, sum, acc[D]) for the split. Splits start at key `base` (decode
/// skips keys no window can see).
/// `attn_combine`: grid (head, qi), block 32; rescales and sums the splits.
///
/// Tiled prefill, `attn_prefill`: grid (ceil(q_len / BQ), head), block 256. A block holds BQ
/// query rows (16, or 32 when D <= 128) in shared memory and walks 16-key tiles: S = Q·Kᵀ for
/// the tile, an online-softmax update per row, then O = diag(α)·O + P·V, so each K/V row is
/// read once per BQ queries.
pub const ATTENTION_CUDA: &str = with_common!(
    r#"
#define MAXV 8   // max D / 32

extern "C" __global__ void attn_partial(
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
    float* __restrict__ P,   // [q_len, nh, n_splits, D + 2]
    unsigned int cache_start, unsigned int q_len, unsigned int nh, unsigned int nkv,
    unsigned int D, unsigned int n_splits, unsigned int split_len, unsigned int window,
    unsigned int base, unsigned int bidir)
{
    unsigned int head = blockIdx.x, split = blockIdx.y, qi = blockIdx.z;
    unsigned int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    unsigned int kvh = head / (nh / nkv);
    u64 kvd = (u64)nkv * D;
    unsigned int per = (D + 31) / 32;

    unsigned int qpos = cache_start + qi + 1;
    unsigned int attend = bidir ? cache_start + q_len : qpos;
    unsigned int lo = (window > 0 && qpos > window) ? qpos - window : 0;
    unsigned int s0 = base + split * split_len;
    unsigned int j0 = max(s0, lo);
    unsigned int j1 = min(s0 + split_len, attend);

    float q[MAXV], acc[MAXV];
    float scale = rsqrtf((float)D);
    const float* qp = Q + ((u64)qi * nh + head) * D;
    #pragma unroll
    for (unsigned int v = 0; v < MAXV; v++) {
        unsigned int d = v * 32 + lane;
        q[v] = (v < per && d < D) ? qp[d] * scale : 0.0f;
        acc[v] = 0.0f;
    }
    float m = NEG_INF, l = 0.0f;

    for (unsigned int j = j0 + warp; j < j1; j += 8) {
        const float* kp = K + (u64)j * kvd + (u64)kvh * D;
        float s = 0.0f;
        #pragma unroll
        for (unsigned int v = 0; v < MAXV; v++) {
            unsigned int d = v * 32 + lane;
            if (v < per && d < D) s += q[v] * kp[d];
        }
        s = warp_sum(s);
        float m2 = fmaxf(m, s);
        float c = expf(m - m2), p = expf(s - m2);
        l = l * c + p;
        const float* vp = V + (u64)j * kvd + (u64)kvh * D;
        #pragma unroll
        for (unsigned int v = 0; v < MAXV; v++) {
            unsigned int d = v * 32 + lane;
            if (v < per && d < D) acc[v] = acc[v] * c + p * vp[d];
        }
        m = m2;
    }

    // Merge the 8 warps.
    __shared__ float sm[8], sl[8];
    __shared__ float sacc[8 * 256];
    if (lane == 0) { sm[warp] = m; sl[warp] = l; }
    #pragma unroll
    for (unsigned int v = 0; v < MAXV; v++) {
        unsigned int d = v * 32 + lane;
        if (v < per && d < D) sacc[warp * D + d] = acc[v];
    }
    __syncthreads();
    if (warp != 0) return;

    float M = NEG_INF;
    for (int g = 0; g < 8; g++) M = fmaxf(M, sm[g]);
    float L = 0.0f;
    float out[MAXV];
    #pragma unroll
    for (unsigned int v = 0; v < MAXV; v++) out[v] = 0.0f;
    for (int g = 0; g < 8; g++) {
        float c = (sm[g] == NEG_INF) ? 0.0f : expf(sm[g] - M);
        L += sl[g] * c;
        #pragma unroll
        for (unsigned int v = 0; v < MAXV; v++) {
            unsigned int d = v * 32 + lane;
            if (v < per && d < D) out[v] += sacc[g * D + d] * c;
        }
    }
    float* pp = P + (((u64)qi * nh + head) * n_splits + split) * (D + 2);
    #pragma unroll
    for (unsigned int v = 0; v < MAXV; v++) {
        unsigned int d = v * 32 + lane;
        if (v < per && d < D) pp[2 + d] = out[v];
    }
    if (lane == 0) { pp[0] = M; pp[1] = L; }
}

extern "C" __global__ void attn_combine(
    const float* __restrict__ P, float* __restrict__ O,
    unsigned int nh, unsigned int D, unsigned int n_splits)
{
    unsigned int head = blockIdx.x, qi = blockIdx.y;
    const float* base = P + ((u64)qi * nh + head) * n_splits * (D + 2);
    float M = NEG_INF;
    for (unsigned int s = 0; s < n_splits; s++) M = fmaxf(M, base[s * (D + 2)]);
    float L = 0.0f;
    for (unsigned int s = 0; s < n_splits; s++) {
        float m = base[s * (D + 2)];
        if (m != NEG_INF) L += base[s * (D + 2) + 1] * expf(m - M);
    }
    for (unsigned int d = threadIdx.x; d < D; d += blockDim.x) {
        float o = 0.0f;
        for (unsigned int s = 0; s < n_splits; s++) {
            float m = base[s * (D + 2)];
            if (m != NEG_INF) o += base[s * (D + 2) + 2 + d] * expf(m - M);
        }
        O[((u64)qi * nh + head) * D + d] = o / L;
    }
}

#define BK 16

extern "C" __global__ void attn_prefill(
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
    float* __restrict__ O, unsigned int cache_start, unsigned int q_len, unsigned int nh,
    unsigned int nkv, unsigned int D, unsigned int window, unsigned int bidir, unsigned int BQ)
{
    __shared__ float Qs[32 * 129];        // [BQ][D + 1]; BQ * (D + 1) <= 4128
    __shared__ float KVs[BK * 257];       // [BK][D + 1], K then V
    __shared__ float Ps[32 * BK];         // [BQ][BK] scores, then probabilities
    __shared__ float m_s[32], l_s[32], a_s[32];

    unsigned int tid = threadIdx.x;
    unsigned int head = blockIdx.y, q0 = blockIdx.x * BQ;
    unsigned int kvh = head / (nh / nkv);
    u64 qs = (u64)nh * D, kvd = (u64)nkv * D;
    unsigned int ld = D + 1;
    unsigned int total = cache_start + q_len;
    float scale = rsqrtf((float)D);

    for (unsigned int i = tid; i < BQ * D; i += 256) {
        unsigned int r = i / D, d = i % D;
        Qs[r * ld + d] = (q0 + r < q_len) ? Q[(u64)(q0 + r) * qs + (u64)head * D + d] * scale : 0.0f;
    }
    if (tid < BQ) { m_s[tid] = NEG_INF; l_s[tid] = 0.0f; }

    // Keys any row of this block can see.
    unsigned int r_last = min(q0 + BQ, q_len) - 1;
    unsigned int k_end = bidir ? total : cache_start + r_last + 1;
    unsigned int first = cache_start + q0 + 1;
    unsigned int k_begin = (window > 0 && first > window) ? first - window : 0;

    // Accumulator ownership: row tid / tpr, dims tid % tpr + tpr * i.
    unsigned int tpr = 256 / BQ;
    unsigned int ar = tid / tpr, ad = tid % tpr;
    float acc[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) acc[i] = 0.0f;
    unsigned int per_score = BQ * BK / 256;
    __syncthreads();

    for (unsigned int j0 = k_begin; j0 < k_end; j0 += BK) {
        for (unsigned int i = tid; i < BK * D; i += 256) {
            unsigned int c = i / D, d = i % D, j = j0 + c;
            KVs[c * ld + d] = (j < total) ? K[(u64)j * kvd + (u64)kvh * D + d] : 0.0f;
        }
        __syncthreads();
        for (unsigned int t = 0; t < per_score; t++) {
            unsigned int e = tid + 256 * t;
            unsigned int r = e / BK, c = e % BK;
            unsigned int qi = q0 + r, j = j0 + c;
            unsigned int qpos = cache_start + qi + 1;
            unsigned int attend = bidir ? total : qpos;
            unsigned int lo = (window > 0 && qpos > window) ? qpos - window : 0;
            float s = NEG_INF;
            if (qi < q_len && j < attend && j >= lo) {
                s = 0.0f;
                const float* qr = Qs + r * ld;
                const float* kr = KVs + c * ld;
                for (unsigned int d = 0; d < D; d++) s += qr[d] * kr[d];
            }
            Ps[r * BK + c] = s;
        }
        __syncthreads();
        if (tid < BQ) {
            float* pr = Ps + tid * BK;
            float m = m_s[tid], bm = NEG_INF;
            for (int c = 0; c < BK; c++) bm = fmaxf(bm, pr[c]);
            float mn = fmaxf(m, bm);
            float alpha = (m == NEG_INF) ? 0.0f : expf(m - mn);
            float ps = 0.0f;
            for (int c = 0; c < BK; c++) {
                float p = (pr[c] == NEG_INF) ? 0.0f : expf(pr[c] - mn);
                pr[c] = p;
                ps += p;
            }
            l_s[tid] = l_s[tid] * alpha + ps;
            m_s[tid] = mn;
            a_s[tid] = alpha;
        }
        for (unsigned int i = tid; i < BK * D; i += 256) {
            unsigned int c = i / D, d = i % D, j = j0 + c;
            // Overwrites K: every score read of K finished before the barrier above.
            KVs[c * ld + d] = (j < total) ? V[(u64)j * kvd + (u64)kvh * D + d] : 0.0f;
        }
        __syncthreads();
        float alpha = a_s[ar];
        const float* pr = Ps + ar * BK;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            unsigned int d = ad + tpr * i;
            if (d < D) {
                float o = acc[i] * alpha;
                for (int c = 0; c < BK; c++) o += pr[c] * KVs[c * ld + d];
                acc[i] = o;
            }
        }
        __syncthreads();
    }

    unsigned int qi = q0 + ar;
    if (qi >= q_len) return;
    float l = l_s[ar];
    float inv = l > 0.0f ? 1.0f / l : 0.0f;
    float* op = O + (u64)qi * qs + (u64)head * D;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        unsigned int d = ad + tpr * i;
        if (d < D) op[d] = acc[i] * inv;
    }
}

#undef BK

// Split-KV attention for a few queries (speculative-decoding verify, short incremental
// prefills): block (kv head, split, row group). A block's rows are (query, head) pairs that
// share one KV head (row f -> query f / gqa, head kvh * gqa + f % gqa), so each K/V row of the
// split is read once for every query and every head of the GQA group. Keys come in tiles of
// BK, row-major in shared memory (stride D + 4 keeps float4 accesses conflict-free), with the
// next tile's K and V prefetched into registers while the current one is used: scores with
// one key per lane, an online softmax per row in registers (warp w owns rows w*RW ..), then
// P·V with lanes over head dims. Writes unnormalized partials in `attn_partial`'s layout for
// `attn_combine`. Needs D % 4 == 0.
template <int R, int BK, int DMAX>
__device__ __forceinline__ void attn_multi_body(
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
    float* __restrict__ P, unsigned int cache_start, unsigned int q_len, unsigned int nh,
    unsigned int nkv, unsigned int D, unsigned int n_splits, unsigned int split_len,
    unsigned int window, unsigned int base, unsigned int bidir,
    float* Qs, float* KVs, float* Ps)
{
    constexpr int RW = R / 8;              // rows per warp
    constexpr int SEGS = 32 / BK;          // key segments per warp
    constexpr int RPT = RW / SEGS;         // rows per lane in the score phase
    constexpr int NV = DMAX / 32;          // head dims per lane in the PV phase
    constexpr int PS = R + 4;              // Ps row stride
    constexpr int NL = BK * DMAX / 4 / 256;  // float4 loads per thread per tile

    unsigned int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    unsigned int kvh = blockIdx.x, split = blockIdx.y, f0 = blockIdx.z * R;
    unsigned int gqa = nh / nkv, rows = q_len * gqa;
    unsigned int LD = D + 4, D4 = D / 4;
    u64 kvd = (u64)nkv * D;
    unsigned int total = cache_start + q_len;
    float scale = rsqrtf((float)D);

    for (unsigned int i = tid; i < R * D4; i += 256) {
        unsigned int r = i / D4, d = (i % D4) * 4, f = f0 + r;
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (f < rows) {
            unsigned int qi = f / gqa, head = kvh * gqa + f % gqa;
            v = *(const float4*)(Q + ((u64)qi * nh + head) * D + d);
            v.x *= scale; v.y *= scale; v.z *= scale; v.w *= scale;
        }
        *(float4*)(Qs + r * LD + d) = v;
    }

    // Keys any row of the block can see, clipped to this split.
    unsigned int f_last = min(f0 + R, rows) - 1;
    unsigned int first = cache_start + f0 / gqa + 1;
    unsigned int lo_blk = (window > 0 && first > window) ? first - window : 0;
    unsigned int hi_blk = bidir ? total : cache_start + f_last / gqa + 1;
    unsigned int s0 = base + split * split_len;
    unsigned int jb = max(s0, lo_blk), je = min(s0 + split_len, hi_blk);

    unsigned int c = lane % BK, seg = lane / BK;
    unsigned int r0 = warp * RW + seg * RPT;
    unsigned int lo[RPT], hi[RPT];
    float m[RPT], l[RPT];
    #pragma unroll
    for (int i = 0; i < RPT; i++) {
        unsigned int f = f0 + r0 + i;
        lo[i] = 0; hi[i] = 0;
        if (f < rows) {
            unsigned int qpos = cache_start + f / gqa + 1;
            hi[i] = bidir ? total : qpos;
            lo[i] = (window > 0 && qpos > window) ? qpos - window : 0;
        }
        m[i] = NEG_INF; l[i] = 0.0f;
    }
    float acc[RW][NV];
    #pragma unroll
    for (int i = 0; i < RW; i++)
        #pragma unroll
        for (int v = 0; v < NV; v++) acc[i][v] = 0.0f;

    float4 kreg[NL], vreg[NL];
    auto fetch = [&](float4* reg, const float* src, unsigned int j0) {
        #pragma unroll
        for (int t = 0; t < NL; t++) {
            unsigned int i = tid + 256 * t, cc = i / D4, d = (i % D4) * 4, j = j0 + cc;
            reg[t] = (cc < BK && j < je) ? *(const float4*)(src + (u64)j * kvd + (u64)kvh * D + d)
                                         : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    };
    auto stage = [&](const float4* reg) {
        #pragma unroll
        for (int t = 0; t < NL; t++) {
            unsigned int i = tid + 256 * t, cc = i / D4, d = (i % D4) * 4;
            if (cc < BK) *(float4*)(KVs + cc * LD + d) = reg[t];
        }
    };
    if (jb < je) { fetch(kreg, K, jb); fetch(vreg, V, jb); }
    __syncthreads();

    for (unsigned int j0 = jb; j0 < je; j0 += BK) {
        bool more = j0 + BK < je;
        stage(kreg);
        __syncthreads();
        if (more) fetch(kreg, K, j0 + BK);

        float s[RPT];
        #pragma unroll
        for (int i = 0; i < RPT; i++) s[i] = 0.0f;
        const float* kr = KVs + c * LD;
        for (unsigned int d = 0; d < D; d += 4) {
            float4 k4 = *(const float4*)(kr + d);
            #pragma unroll
            for (int i = 0; i < RPT; i++) {
                float4 q4 = *(const float4*)(Qs + (r0 + i) * LD + d);
                s[i] += q4.x * k4.x + q4.y * k4.y + q4.z * k4.z + q4.w * k4.w;
            }
        }
        unsigned int j = j0 + c;
        float alpha[RPT];
        #pragma unroll
        for (int i = 0; i < RPT; i++) {
            if (!(j < je && j >= lo[i] && j < hi[i])) s[i] = NEG_INF;
            float mx = s[i];
            #pragma unroll
            for (int o = BK / 2; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            float mn = fmaxf(m[i], mx);
            float p = (s[i] == NEG_INF) ? 0.0f : __expf(s[i] - mn);
            alpha[i] = (m[i] == NEG_INF) ? 0.0f : __expf(m[i] - mn);
            float ps = p;
            #pragma unroll
            for (int o = BK / 2; o > 0; o >>= 1) ps += __shfl_xor_sync(0xffffffffu, ps, o);
            l[i] = l[i] * alpha[i] + ps;
            m[i] = mn;
            Ps[c * PS + r0 + i] = p;
        }
        __syncthreads();   // every warp is done with K
        stage(vreg);
        __syncthreads();
        if (more) fetch(vreg, V, j0 + BK);

        #pragma unroll
        for (int i = 0; i < RW; i++) {
            float a = __shfl_sync(0xffffffffu, alpha[i % RPT], (i / RPT) * BK);
            #pragma unroll
            for (int v = 0; v < NV; v++) acc[i][v] *= a;
        }
        const float* pw = Ps + warp * RW;
        #pragma unroll 4
        for (int cc = 0; cc < BK; cc++) {
            float p[RW];
            #pragma unroll
            for (int i = 0; i < RW; i++) p[i] = pw[cc * PS + i];
            const float* vr = KVs + cc * LD;
            #pragma unroll
            for (int v = 0; v < NV; v++) {
                unsigned int d = v * 32 + lane;
                float x = (d < D) ? vr[d] : 0.0f;
                #pragma unroll
                for (int i = 0; i < RW; i++) acc[i][v] += p[i] * x;
            }
        }
        __syncthreads();   // before the next tile overwrites V and Ps
    }

    #pragma unroll
    for (int i = 0; i < RW; i++) {
        unsigned int src = (i / RPT) * BK;
        float mi = __shfl_sync(0xffffffffu, m[i % RPT], src);
        float li = __shfl_sync(0xffffffffu, l[i % RPT], src);
        unsigned int f = f0 + warp * RW + i;
        if (f >= rows) continue;
        unsigned int qi = f / gqa, head = kvh * gqa + f % gqa;
        float* pp = P + (((u64)qi * nh + head) * n_splits + split) * (D + 2);
        #pragma unroll
        for (int v = 0; v < NV; v++) {
            unsigned int d = v * 32 + lane;
            if (d < D) pp[2 + d] = acc[i][v];
        }
        if (lane == 0) { pp[0] = mi; pp[1] = li; }
    }
}

#define ATTN_MULTI(NAME, R, BK, DMAX)                                                         \
extern "C" __global__ void __launch_bounds__(256) NAME(                                      \
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,    \
    float* __restrict__ P, unsigned int cache_start, unsigned int q_len, unsigned int nh,     \
    unsigned int nkv, unsigned int D, unsigned int n_splits, unsigned int split_len,          \
    unsigned int window, unsigned int base, unsigned int bidir)                               \
{                                                                                             \
    __shared__ __align__(16) float Qs[R * (DMAX + 4)];                                        \
    __shared__ __align__(16) float KVs[BK * (DMAX + 4)];                                      \
    __shared__ __align__(16) float Ps[BK * (R + 4)];                                          \
    attn_multi_body<R, BK, DMAX>(Q, K, V, P, cache_start, q_len, nh, nkv, D, n_splits,        \
                                 split_len, window, base, bidir, Qs, KVs, Ps);                \
}

ATTN_MULTI(attn_multi_r8, 8, 32, 128)
ATTN_MULTI(attn_multi_r16, 16, 32, 128)
ATTN_MULTI(attn_multi_r32, 32, 32, 128)
ATTN_MULTI(attn_multi_d256, 16, 16, 256)
"#
);
