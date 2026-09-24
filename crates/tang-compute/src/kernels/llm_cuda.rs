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
"#
);
