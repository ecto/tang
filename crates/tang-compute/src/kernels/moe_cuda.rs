//! CUDA kernels for MoE on the dense decoder path and gpt-oss's attention sinks (see
//! [`crate::moe`] for the semantics each one matches).

/// Shared helpers, prepended to every source here.
macro_rules! with_prelude {
    ($body:literal) => {
        concat!(
            r#"
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
#define NEG_INF __uint_as_float(0xff800000u)
typedef unsigned long long u64;
"#,
            $body
        )
    };
}

/// Causal attention with sinks over a paged KV cache (read through `kv_source`'s `kv_t`,
/// `kv1` and `kv_row`), split over keys ("flash decoding") and grouped by KV head.
///
/// `attn_sink_partial`: grid `(rows, nkv, max_splits)`, block `32 * G` (`G = nh / nkv <= 16`,
/// warp `g` is query head `kvh * G + g`). Row `r` is at position `RP[r]` (or `start + r`
/// without `RP`) of the cache whose block table starts at `T + RT[r]` (or `T`), and sees keys
/// `lo ..= pos` (`lo`: the window's first key, or 0). It splits them by its own span alone,
/// `ceil(span / 256)` splits (at most 64), so its numbers don't depend on the other rows. The
/// block stages 32-key tiles of K and V in shared memory for its `G` heads; lane `j` scores
/// key `j`, lanes hold `D / 32` output dims. Splits past the row's count write an empty slot.
///
/// `attn_sink_combine`: grid `(rows, nh)`, block 32. Merges the splits and the head's sink
/// logit (a key with a zero value): `o = Σ_s e^{m_s - M} acc_s / (Σ_s e^{m_s - M} l_s +
/// e^{sink - M})`.
pub const ATTN_SINKS_CUDA: &str = with_prelude!(
    r#"
#define TK 32
#define MAXD 128
#define MAXG 16

__device__ __forceinline__ unsigned int sink_splits(unsigned int span) {
    return min(max((span + 255u) / 256u, 1u), 64u);
}

extern "C" __global__ void attn_sink_partial(
    const float* __restrict__ Q, const kv_t* __restrict__ K, const kv_t* __restrict__ V,
    float* __restrict__ P,   // [rows, nh, max_splits, D + 2]
    unsigned int start, unsigned int nh, unsigned int nkv, unsigned int D,
    unsigned int max_splits, unsigned int window, const unsigned int* __restrict__ T,
    unsigned int sh, const unsigned int* __restrict__ RP, const unsigned int* __restrict__ RT,
    unsigned int has_rows)
{
    __shared__ float Ks[TK * (MAXD + 1)];
    __shared__ float Vs[TK * MAXD];
    __shared__ float Qs[MAXG * MAXD];
    unsigned int r = blockIdx.x, kvh = blockIdx.y, split = blockIdx.z;
    unsigned int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, G = nh / nkv;
    unsigned int head = kvh * G + warp;
    unsigned int pos = has_rows ? RP[r] : start + r;
    const unsigned int* tb = has_rows ? T + RT[r] : T;
    unsigned int end = pos + 1;
    unsigned int lo = (window > 0 && end > window) ? end - window : 0;
    unsigned int span = end - lo, mine = sink_splits(span);
    unsigned int len = (span + mine - 1) / mine;
    unsigned int j0 = split < mine ? lo + split * len : end;
    unsigned int j1 = split < mine ? min(j0 + len, end) : end;
    u64 kvd = (u64)nkv * D;
    unsigned int per = (D + 31) / 32;

    float scale = rsqrtf((float)D);
    const float* qp = Q + ((u64)r * nh + head) * D;
    for (unsigned int d = lane; d < D; d += 32) Qs[warp * D + d] = qp[d] * scale;
    float acc[MAXD / 32];
    #pragma unroll
    for (unsigned int v = 0; v < MAXD / 32; v++) acc[v] = 0.0f;
    float m = NEG_INF, l = 0.0f;

    for (unsigned int t0 = j0; t0 < j1; t0 += TK) {
        unsigned int n = min((unsigned int)TK, j1 - t0);
        __syncthreads();
        for (unsigned int i = threadIdx.x; i < n * D; i += blockDim.x) {
            unsigned int j = i / D, d = i - j * D;
            u64 row = kv_row(tb, sh, t0 + j) * kvd + (u64)kvh * D + d;
            Ks[j * (D + 1) + d] = kv1(K + row);
            Vs[j * D + d] = kv1(V + row);
        }
        __syncthreads();
        float s = NEG_INF;
        if (lane < n) {
            s = 0.0f;
            for (unsigned int d = 0; d < D; d++) s += Qs[warp * D + d] * Ks[lane * (D + 1) + d];
        }
        float m2 = fmaxf(m, warp_max(s));
        float c = expf(m - m2);
        float p = lane < n ? expf(s - m2) : 0.0f;
        l = l * c + warp_sum(p);
        #pragma unroll
        for (unsigned int v = 0; v < MAXD / 32; v++) acc[v] *= c;
        for (unsigned int j = 0; j < n; j++) {
            float pj = __shfl_sync(0xffffffffu, p, j);
            #pragma unroll
            for (unsigned int v = 0; v < MAXD / 32; v++) {
                unsigned int d = v * 32 + lane;
                if (v < per && d < D) acc[v] += pj * Vs[j * D + d];
            }
        }
        m = m2;
    }

    float* pp = P + (((u64)r * nh + head) * max_splits + split) * (D + 2);
    #pragma unroll
    for (unsigned int v = 0; v < MAXD / 32; v++) {
        unsigned int d = v * 32 + lane;
        if (v < per && d < D) pp[2 + d] = acc[v];
    }
    if (lane == 0) { pp[0] = m; pp[1] = l; }
}

extern "C" __global__ void attn_sink_combine(
    const float* __restrict__ P, const float* __restrict__ S, float* __restrict__ O,
    unsigned int nh, unsigned int D, unsigned int max_splits)
{
    unsigned int r = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
    const float* pp = P + ((u64)r * nh + head) * max_splits * (D + 2);
    float sink = S[head];
    float M = sink;
    for (unsigned int s = 0; s < max_splits; s++) M = fmaxf(M, pp[s * (D + 2)]);
    float L = expf(sink - M);
    for (unsigned int s = 0; s < max_splits; s++) {
        float ms = pp[s * (D + 2)];
        if (ms != NEG_INF) L += pp[s * (D + 2) + 1] * expf(ms - M);
    }
    float inv = 1.0f / L;
    for (unsigned int d = lane; d < D; d += 32) {
        float o = 0.0f;
        for (unsigned int s = 0; s < max_splits; s++) {
            float ms = pp[s * (D + 2)];
            if (ms != NEG_INF) o += pp[s * (D + 2) + 2 + d] * expf(ms - M);
        }
        O[((u64)r * nh + head) * D + d] = o * inv;
    }
}
"#
);

/// MoE routing, expert GEMV and the elementwise pieces.
///
/// `moe_route`: grid `rows`, block 32. Top `k` (at most 8) of `E` (at most 256) logits by
/// (value desc, index asc), weighted by a softmax over the k (`mode` 0) or over all `E`
/// (`mode` 1, renormalised over the k when `renorm`). Ids go out as u32 bits.
///
/// `moe_sort`: one block of 1024. Groups the `picks` routed ids by expert: `OFF[e]..OFF[e+1]`
/// of `LIST` are the picks of expert `e` (in no particular order: each pick's result doesn't
/// depend on the others).
///
/// `moe_gemv_mxfp4`: grid `(ceil(N / 8), E)`, block 256. Warp `w` owns output row
/// `n = blockIdx.x * 8 + w` of expert `e = blockIdx.y` and applies it to every pick of `e`, up
/// to 8 at a time, so each weight row is read once per 8 picks. Lane `b` takes the row's MXFP4
/// blocks `b, b + 32, ...`: 16 bytes of nibbles and an E8M0 scale; per block `acc += scale ·
/// Σ value · x` in a fixed order, then one warp sum, so a pick's numbers don't depend on how
/// many other picks share its expert. `x` is `[rows, K]` read by pick `p` at row `p / k_top`,
/// or with `per_pick` its own row `p`.
pub const MOE_CUDA: &str = with_prelude!(
    r#"
extern "C" __global__ void moe_route(
    const float* __restrict__ L, unsigned int* __restrict__ IDS, float* __restrict__ W,
    unsigned int E, unsigned int k, unsigned int mode, unsigned int renorm)
{
    unsigned int r = blockIdx.x, lane = threadIdx.x;
    const float* l = L + (u64)r * E;
    float v[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        unsigned int e = i * 32 + lane;
        v[i] = e < E ? l[e] : NEG_INF;
    }
    float top[8];
    unsigned int ids[8];
    for (unsigned int t = 0; t < k; t++) {
        float best = NEG_INF;
        unsigned int bi = 0xffffffffu;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            unsigned int e = i * 32 + lane;
            if (e < E && (v[i] > best || (v[i] == best && e < bi))) { best = v[i]; bi = e; }
        }
        for (int o = 16; o > 0; o >>= 1) {
            float ob = __shfl_xor_sync(0xffffffffu, best, o);
            unsigned int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ob > best || (ob == best && oi < bi)) { best = ob; bi = oi; }
        }
        top[t] = best;
        ids[t] = bi;
        #pragma unroll
        for (int i = 0; i < 8; i++)
            if (i * 32 + lane == bi) v[i] = NEG_INF;
    }
    if (lane != 0) return;
    float mx = top[0], z = 0.0f;
    if (mode == 0) {
        for (unsigned int t = 0; t < k; t++) z += expf(top[t] - mx);
    } else {
        for (unsigned int e = 0; e < E; e++) z += expf(l[e] - mx);
    }
    float w[8], s = 0.0f;
    for (unsigned int t = 0; t < k; t++) { w[t] = expf(top[t] - mx) / z; s += w[t]; }
    for (unsigned int t = 0; t < k; t++) {
        IDS[(u64)r * k + t] = ids[t];
        W[(u64)r * k + t] = (mode == 1 && renorm) ? w[t] / s : w[t];
    }
}

extern "C" __global__ void moe_sort(
    const unsigned int* __restrict__ IDS, unsigned int* __restrict__ OFF,
    unsigned int* __restrict__ LIST, unsigned int picks, unsigned int E)
{
    __shared__ unsigned int cnt[257], cur[256];
    for (unsigned int e = threadIdx.x; e <= E; e += blockDim.x) cnt[e] = 0;
    __syncthreads();
    for (unsigned int p = threadIdx.x; p < picks; p += blockDim.x) atomicAdd(&cnt[IDS[p] + 1], 1u);
    __syncthreads();
    if (threadIdx.x == 0) {
        for (unsigned int e = 1; e <= E; e++) cnt[e] += cnt[e - 1];
    }
    __syncthreads();
    for (unsigned int e = threadIdx.x; e <= E; e += blockDim.x) {
        OFF[e] = cnt[e];
        if (e < E) cur[e] = cnt[e];
    }
    __syncthreads();
    for (unsigned int p = threadIdx.x; p < picks; p += blockDim.x)
        LIST[atomicAdd(&cur[IDS[p]], 1u)] = p;
}

#define ROWS 4    // output rows per warp: each input block loaded once feeds all of them

__device__ __forceinline__ float e8m0f(unsigned int s) {
    return s == 0 ? __uint_as_float(1u << 22) : __uint_as_float(s << 23);
}

// Grid (ceil(N / (8 * ROWS)), E), block 256. Warp w owns rows n0 .. n0 + ROWS of expert e and
// takes its picks one at a time; lane b sums blocks b, b + 32, ... of each row, in that order,
// so a pick's numbers are the same whatever else shares its expert.
extern "C" __global__ void moe_gemv_mxfp4(
    const float* __restrict__ X, const unsigned char* __restrict__ Wt,
    const float* __restrict__ B, const unsigned int* __restrict__ OFF,
    const unsigned int* __restrict__ LIST, float* __restrict__ Y,
    unsigned int k_top, unsigned int K, unsigned int N, unsigned int E,
    unsigned int per_pick, unsigned int has_bias)
{
    __shared__ float lut[16];
    if (threadIdx.x < 16) {
        const float f[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                             -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
        lut[threadIdx.x] = f[threadIdx.x];
    }
    __syncthreads();
    unsigned int e = blockIdx.y, lane = threadIdx.x & 31;
    unsigned int n0 = (blockIdx.x * 8 + (threadIdx.x >> 5)) * ROWS;
    unsigned int p0 = OFF[e], p1 = OFF[e + 1];
    if (n0 >= N || p0 == p1) return;
    unsigned int nb = K / 32, rows = min((unsigned int)ROWS, N - n0);
    const unsigned char* wb = Wt + ((u64)e * N + n0) * (K / 2);
    const unsigned char* ws = Wt + (u64)E * N * (K / 2) + ((u64)e * N + n0) * nb;
    for (unsigned int c = p0; c < p1; c++) {
        unsigned int p = LIST[c];
        const float* x = X + (u64)(per_pick ? p : p / k_top) * K;
        float acc[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; r++) acc[r] = 0.0f;
        for (unsigned int b = lane; b < nb; b += 32) {
            const float4* x4 = (const float4*)(x + (u64)b * 32);
            float xv[32];
            #pragma unroll
            for (int t = 0; t < 8; t++) {
                float4 v = x4[t];
                xv[4 * t] = v.x; xv[4 * t + 1] = v.y; xv[4 * t + 2] = v.z; xv[4 * t + 3] = v.w;
            }
            #pragma unroll
            for (int r = 0; r < ROWS; r++) {
                if (r >= rows) break;
                uint4 q = *(const uint4*)(wb + ((u64)r * nb + b) * 16);
                float sc = e8m0f(ws[(u64)r * nb + b]);
                unsigned int qs[4] = {q.x, q.y, q.z, q.w};
                float d = 0.0f;
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    #pragma unroll
                    for (int t = 0; t < 8; t++) d += lut[(qs[j] >> (4 * t)) & 15u] * xv[8 * j + t];
                }
                acc[r] += sc * d;
            }
        }
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            if (r >= rows) break;
            float s = warp_sum(acc[r]);
            if (lane == 0)
                Y[(u64)p * N + n0 + r] = s + (has_bias ? B[(u64)e * N + n0 + r] : 0.0f);
        }
    }
}

// `moe_gemv_mxfp4` with each pick's input row staged in shared memory (dynamic, K floats)
// instead of read through L1 by every warp. Same lanes, blocks and summation order, so the same
// bits.
extern "C" __global__ void moe_gemv_mxfp4_smem(
    const float* __restrict__ X, const unsigned char* __restrict__ Wt,
    const float* __restrict__ B, const unsigned int* __restrict__ OFF,
    const unsigned int* __restrict__ LIST, float* __restrict__ Y,
    unsigned int k_top, unsigned int K, unsigned int N, unsigned int E,
    unsigned int per_pick, unsigned int has_bias)
{
    extern __shared__ float xs[];
    __shared__ float lut[16];
    if (threadIdx.x < 16) {
        const float f[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                             -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
        lut[threadIdx.x] = f[threadIdx.x];
    }
    unsigned int e = blockIdx.y, lane = threadIdx.x & 31;
    unsigned int p0 = OFF[e], p1 = OFF[e + 1];
    if (p0 == p1) return;   // uniform across the block
    unsigned int n0 = (blockIdx.x * 8 + (threadIdx.x >> 5)) * ROWS;
    bool live = n0 < N;
    unsigned int nb = K / 32, rows = live ? min((unsigned int)ROWS, N - n0) : 0;
    const unsigned char* wb = Wt + ((u64)e * N + n0) * (K / 2);
    const unsigned char* ws = Wt + (u64)E * N * (K / 2) + ((u64)e * N + n0) * nb;
    for (unsigned int c = p0; c < p1; c++) {
        unsigned int p = LIST[c];
        const float4* x4g = (const float4*)(X + (u64)(per_pick ? p : p / k_top) * K);
        __syncthreads();
        for (unsigned int i = threadIdx.x; i < K / 4; i += blockDim.x) ((float4*)xs)[i] = x4g[i];
        __syncthreads();
        if (!live) continue;
        float acc[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; r++) acc[r] = 0.0f;
        for (unsigned int b = lane; b < nb; b += 32) {
            const float4* x4 = (const float4*)(xs + b * 32);
            float xv[32];
            #pragma unroll
            for (int t = 0; t < 8; t++) {
                float4 v = x4[t];
                xv[4 * t] = v.x; xv[4 * t + 1] = v.y; xv[4 * t + 2] = v.z; xv[4 * t + 3] = v.w;
            }
            #pragma unroll
            for (int r = 0; r < ROWS; r++) {
                if (r >= rows) break;
                uint4 q = *(const uint4*)(wb + ((u64)r * nb + b) * 16);
                float sc = e8m0f(ws[(u64)r * nb + b]);
                unsigned int qs[4] = {q.x, q.y, q.z, q.w};
                float d = 0.0f;
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    #pragma unroll
                    for (int t = 0; t < 8; t++) d += lut[(qs[j] >> (4 * t)) & 15u] * xv[8 * j + t];
                }
                acc[r] += sc * d;
            }
        }
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            if (r >= rows) break;
            float s = warp_sum(acc[r]);
            if (lane == 0)
                Y[(u64)p * N + n0 + r] = s + (has_bias ? B[(u64)e * N + n0 + r] : 0.0f);
        }
    }
}

// Prefill: picks grouped by expert run as one GEMM per expert (see `moe_linear_llm`).
// `moe_gather`: row i of XG is the input row of pick LIST[i]; grid (picks), block 256.
extern "C" __global__ void moe_gather(
    const float* __restrict__ X, const unsigned int* __restrict__ LIST, float* __restrict__ XG,
    unsigned int k_top, unsigned int K, unsigned int per_pick)
{
    unsigned int i = blockIdx.x, p = LIST[i];
    const float* x = X + (u64)(per_pick ? p : p / k_top) * K;
    for (unsigned int c = threadIdx.x; c < K; c += blockDim.x) XG[(u64)i * K + c] = x[c];
}

// Expert `e`'s MXFP4 matrix `[N, K]` as f32; grid (ceil(N * K / 2 / 256)), block 256: a
// thread per byte (two weights), so neighbouring threads write neighbouring float2s.
__device__ __forceinline__ float e2m1(unsigned int q) {
    unsigned int e = (q >> 1) & 3u, m = q & 1u;
    float v = e == 0 ? 0.5f * (float)m : __uint_as_float(((e + 126u) << 23) | (m << 22));
    return (q & 8u) ? -v : v;
}

extern "C" __global__ void moe_dequant_mxfp4(
    const unsigned char* __restrict__ Wt, float* __restrict__ O, unsigned int e,
    unsigned int K, unsigned int N, unsigned int E)
{
    u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x, nbytes = (u64)N * (K / 2);
    if (t >= nbytes) return;
    unsigned int q = Wt[(u64)e * nbytes + t];
    unsigned int s = Wt[(u64)E * nbytes + (u64)e * (nbytes / 16) + t / 16];
    float sc = s == 0 ? __uint_as_float(1u << 22) : __uint_as_float(s << 23);
    ((float2*)O)[t] = make_float2(e2m1(q & 15u) * sc, e2m1(q >> 4) * sc);
}

// Y[LIST[i]] = YG[i] + bias of pick LIST[i]'s expert; grid (picks), block 256.
extern "C" __global__ void moe_scatter(
    const float* __restrict__ YG, const unsigned int* __restrict__ LIST,
    const unsigned int* __restrict__ IDS, const float* __restrict__ B, float* __restrict__ Y,
    unsigned int N, unsigned int has_bias)
{
    unsigned int i = blockIdx.x, p = LIST[i], e = IDS[p];
    for (unsigned int c = threadIdx.x; c < N; c += blockDim.x)
        Y[(u64)p * N + c] = YG[(u64)i * N + c] + (has_bias ? B[(u64)e * N + c] : 0.0f);
}

extern "C" __global__ void swiglu_clamped(
    const float* __restrict__ GU, float* __restrict__ O, unsigned int rows, unsigned int ff,
    float alpha, float limit)
{
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)rows * ff) return;
    u64 r = i / ff, c = i - r * ff;
    float g = fminf(GU[r * 2 * ff + c], limit);
    float u = fminf(fmaxf(GU[r * 2 * ff + ff + c], -limit), limit);
    O[i] = (u + 1.0f) * (g / (1.0f + expf(-alpha * g)));
}

extern "C" __global__ void moe_combine(
    const float* __restrict__ Pt, const float* __restrict__ W, float* __restrict__ Y,
    unsigned int rows, unsigned int k, unsigned int h)
{
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (u64)rows * h) return;
    u64 r = i / h, c = i - r * h;
    float y = 0.0f;
    for (unsigned int t = 0; t < k; t++) y += W[r * k + t] * Pt[(r * k + t) * h + c];
    Y[i] = y;
}
"#
);
