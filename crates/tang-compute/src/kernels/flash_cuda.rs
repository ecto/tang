//! CUDA kernels for the Flash-Next decode ops (`crate::flash` has the contracts, `cpu::flash`
//! the reference). One NVRTC module; every kernel is `fl_*` so names never collide with the
//! other LLM kernels in the per-name function cache. Shapes are the model's, baked in as
//! constants; the window size `T` (1..=8) is a kernel argument, per-window values come from the
//! `win` record in device memory.

pub const FLASH_CUDA: &str = r#"
typedef unsigned long long u64;
#define NEG_INF __uint_as_float(0xff800000u)

#define HIDDEN 2560
#define HC 4
#define HC_LR 320
#define GDN_HK 16
#define GDN_HV 48
#define GDN_D 128
#define GDN_CONV 10240
#define GDN_V 6144
#define GDN_Z 10240
#define GDN_A 16384
#define GDN_B 16432
#define QSA_HEADS 24
#define QSA_KV 2
#define QSA_D 256
#define IDX_HEADS 4
#define IDX_D 128
#define QSA_WIDTH 2051
#define QSA_K 12288
#define QSA_V 12800
#define QSA_IQ 13312
#define QSA_IK 13824
#define QSA_CHUNK 64
#define QSA_NCH 33
#define TOPK 10
#define FF 640
#define PLAN_CAP 88
#define PLAN_GP 4
#define PLAN_GS (PLAN_GP + 2 * PLAN_CAP)
#define PLAN_ET (PLAN_GS + PLAN_CAP + 1)
#define PLAN_ED (PLAN_ET + PLAN_CAP)
#define PLAN_MISS (PLAN_ED + PLAN_CAP)
#define SHARED_ROW 80
#define BLOB_UP 460800
#define BLOB_DOWN 921600
#define INV_SQRT_D 0.088388346f

__device__ __forceinline__ float bf(unsigned short b) { return __uint_as_float(((unsigned int)b) << 16); }
__device__ __forceinline__ unsigned short to_bf16(float x) {
    unsigned int b = __float_as_uint(x);
    return (unsigned short)((b + 0x7fffu + ((b >> 16) & 1u)) >> 16);
}
// IEEE half to float, exact for normals, subnormals and zero (no FTZ in this module).
__device__ __forceinline__ float h2f(unsigned short h) {
    float f = __uint_as_float(((unsigned int)(h & 0x7fffu)) << 13) * 5.192296858534828e+33f;
    return (h & 0x8000u) ? -f : f;
}
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ int warp_isum(int v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

// ---- int8 activations ([QAct]) ----

// One lane per element of a 32-wide chunk: amax, d = amax / 127, q = round(x / d), the chunk's
// code sum, and the code stored at its permuted slot. `Q` is the [QAct] buffer for `m` rows of
// `k`; this writes row `r`, chunk `c`.
__device__ __forceinline__ void quant_chunk(float v, unsigned int* Q, unsigned int m, unsigned int k,
                                            unsigned int r, unsigned int c, unsigned int lane) {
    float a = warp_max(fabsf(v));
    float d = a / 127.0f;
    int q = 0;
    if (d != 0.0f) q = (int)fminf(fmaxf(roundf(v / d), -127.0f), 127.0f);
    int hx = warp_isum(q);
    unsigned int h = lane >> 4, b = (lane & 15) >> 2, f = lane & 3;
    unsigned char* qb = (unsigned char*)(Q + (u64)r * (k / 4) + c * 8);
    qb[(4 * h + f) * 4 + b] = (unsigned char)(signed char)q;
    if (lane == 0) {
        unsigned int nch = k / 32;
        Q[(u64)m * k / 4 + (u64)r * nch + c] = __float_as_uint(d);
        Q[(u64)m * k / 4 + (u64)m * nch + (u64)r * nch + c] = (unsigned int)hx;
    }
}

// Grid (ceil(K / 256), M), block 256: a warp per chunk.
extern "C" __global__ void fl_quantize(const float* __restrict__ X, unsigned int* __restrict__ Q,
                                       unsigned int M, unsigned int K) {
    unsigned int lane = threadIdx.x & 31, c = blockIdx.x * 8 + (threadIdx.x >> 5), r = blockIdx.y;
    if (c >= K / 32) return;
    quant_chunk(X[(u64)r * K + c * 32 + lane], Q, M, K, r, c, lane);
}

// ---- Q2_0 dot products ----

// Eight dp4a operands for a chunk's two code words: m[4h + f] byte b = code of element 16h+4b+f.
__device__ __forceinline__ void expand(uint2 w, int m[8]) {
    #pragma unroll
    for (int f = 0; f < 4; f++) {
        m[f] = (int)((w.x >> (2 * f)) & 0x03030303u);
        m[4 + f] = (int)((w.y >> (2 * f)) & 0x03030303u);
    }
}
// Σ code · q over one chunk; `xw` points at the chunk's 8 activation words.
__device__ __forceinline__ int chunk_dot(const int m[8], const unsigned int* __restrict__ xw) {
    uint4 a = ((const uint4*)xw)[0], b = ((const uint4*)xw)[1];
    int s = 0;
    s = __dp4a(m[0], (int)a.x, s); s = __dp4a(m[1], (int)a.y, s);
    s = __dp4a(m[2], (int)a.z, s); s = __dp4a(m[3], (int)a.w, s);
    s = __dp4a(m[4], (int)b.x, s); s = __dp4a(m[5], (int)b.y, s);
    s = __dp4a(m[6], (int)b.z, s); s = __dp4a(m[7], (int)b.w, s);
    return s;
}

// y[t, o] = Σ_chunks fma(d_w d_x, Σ code·q − Σ q, acc). R rows per warp, 8 warps per block;
// the weight is the repacked [N, K] matrix at W (codes, then fp16 scales); XQ is [QAct] for
// m = MT rows of K. Lanes stride the chunks; every row's chunk words are loaded before use.
template <int T, int R>
__device__ __forceinline__ void q2_gemv_body(const unsigned int* __restrict__ XQ,
                                             const unsigned char* __restrict__ W,
                                             float* __restrict__ Y, unsigned int K, unsigned int N,
                                             unsigned int MT) {
    unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    unsigned int o0 = (blockIdx.x * 8 + warp) * R;
    if (o0 >= N) return;
    unsigned int nch = K / 32, kb = K / 4;
    const unsigned char* codes = W;
    const unsigned short* sc = (const unsigned short*)(W + (u64)N * kb);
    const unsigned int* xs = XQ + (u64)MT * kb;
    const unsigned int* xh = xs + (u64)MT * nch;
    float acc[R][T];
    #pragma unroll
    for (int r = 0; r < R; r++)
        #pragma unroll
        for (int t = 0; t < T; t++) acc[r][t] = 0.0f;
    for (unsigned int c = lane; c < nch; c += 32) {
        uint2 cw[R];
        float dw[R];
        #pragma unroll
        for (int r = 0; r < R; r++) {
            unsigned int o = min(o0 + r, N - 1);
            cw[r] = *(const uint2*)(codes + (u64)o * kb + c * 8);
            dw[r] = h2f(sc[(u64)o * (K / 64) + c / 2]);
        }
        #pragma unroll
        for (int r = 0; r < R; r++) {
            int m[8];
            expand(cw[r], m);
            #pragma unroll
            for (int t = 0; t < T; t++) {
                int s = chunk_dot(m, XQ + (u64)t * kb + c * 8);
                float dx = __uint_as_float(xs[(u64)t * nch + c]);
                int hx = (int)xh[(u64)t * nch + c];
                acc[r][t] = __fmaf_rn(dw[r] * dx, (float)(s - hx), acc[r][t]);
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < R; r++) {
        #pragma unroll
        for (int t = 0; t < T; t++) {
            float v = warp_sum(acc[r][t]);
            if (lane == 0 && o0 + r < N) Y[(u64)t * N + o0 + r] = v;
        }
    }
}

#define Q2_GEMV(T) \
extern "C" __global__ void __launch_bounds__(256) fl_q2_gemv_t##T( \
    const unsigned int* __restrict__ XQ, const unsigned char* __restrict__ W, \
    float* __restrict__ Y, unsigned int K, unsigned int N) { \
    q2_gemv_body<T, 2>(XQ, W, Y, K, N, T); \
}
Q2_GEMV(1) Q2_GEMV(2) Q2_GEMV(3) Q2_GEMV(4) Q2_GEMV(5) Q2_GEMV(6) Q2_GEMV(7) Q2_GEMV(8)

// ---- hyper-connections ----

// R[t][c] += y[t] * 2σ(inj[t][c] / 4). Grid (HIDDEN / 256, HC, T).
extern "C" __global__ void fl_hc_write(float* __restrict__ R, const float* __restrict__ Y,
                                       const float* __restrict__ I) {
    unsigned int d = blockIdx.x * 256 + threadIdx.x, c = blockIdx.y, t = blockIdx.z;
    float g = 2.0f / (1.0f + expf(-(I[t * HC + c] * 0.25f)));
    u64 i = ((u64)t * HC + c) * HIDDEN + d;
    R[i] = __fmaf_rn(Y[(u64)t * HIDDEN + d], g, R[i]);
}

// Per (stream, token): the pending write (when `apply`), then xn = R · rsqrt(mean R² + eps) · w.
// Grid (HC, T), block 256.
extern "C" __global__ void fl_hc_norm(float* __restrict__ R, const float* __restrict__ Yp,
                                      const float* __restrict__ Ip, unsigned int apply,
                                      const float* __restrict__ Wn, float* __restrict__ XN, float eps) {
    __shared__ float red[8];
    unsigned int c = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
    float* row = R + ((u64)t * HC + c) * HIDDEN;
    float g = apply ? 2.0f / (1.0f + expf(-(Ip[t * HC + c] * 0.25f))) : 0.0f;
    float v[10], ss = 0.0f;
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        unsigned int d = tid + 256 * i;
        float x = row[d];
        if (apply) {
            x = __fmaf_rn(Yp[(u64)t * HIDDEN + d], g, x);
            row[d] = x;
        }
        v[i] = x;
        ss += x * x;
    }
    ss = warp_sum(ss);
    if ((tid & 31) == 0) red[tid >> 5] = ss;
    __syncthreads();
    ss = 0.0f;
    #pragma unroll
    for (int w = 0; w < 8; w++) ss += red[w];
    float rs = 1.0f / sqrtf(ss / (float)HIDDEN + eps);
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        unsigned int d = tid + 256 * i;
        XN[(u64)t * HC * HIDDEN + c * HIDDEN + d] = v[i] * rs * Wn[c * HIDDEN + d];
    }
}

// lo[t][k] = silu((down[k] · xn[t]) / 4) for k < HC_LR, inj[t][c] = inject[c] · xn[t] for the
// next HC rows. Grid (HC_LR + HC or HC_LR), block 256: the 8 warps split K = 10240.
extern "C" __global__ void __launch_bounds__(256) fl_hc_down(
    const float* __restrict__ XN, const unsigned short* __restrict__ Wd,
    const unsigned short* __restrict__ Wi, float* __restrict__ LO, float* __restrict__ INJ,
    unsigned int T) {
    __shared__ float red[8][8];
    unsigned int row = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const unsigned short* w = (row < HC_LR ? Wd + (u64)row * HC * HIDDEN : Wi + (u64)(row - HC_LR) * HC * HIDDEN)
                              + warp * 1280;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (unsigned int i = lane; i < 160; i += 32) {
        uint4 p = ((const uint4*)w)[i];
        float wv[8] = {bf(p.x & 0xffff), bf(p.x >> 16), bf(p.y & 0xffff), bf(p.y >> 16),
                       bf(p.z & 0xffff), bf(p.z >> 16), bf(p.w & 0xffff), bf(p.w >> 16)};
        #pragma unroll
        for (unsigned int t = 0; t < 8; t++) {
            if (t < T) {
                const float4* x4 = (const float4*)(XN + (u64)t * HC * HIDDEN + warp * 1280 + i * 8);
                float4 a = x4[0], b = x4[1];
                acc[t] += wv[0] * a.x + wv[1] * a.y + wv[2] * a.z + wv[3] * a.w
                        + wv[4] * b.x + wv[5] * b.y + wv[6] * b.z + wv[7] * b.w;
            }
        }
    }
    #pragma unroll
    for (unsigned int t = 0; t < 8; t++) {
        if (t < T) {
            float s = warp_sum(acc[t]);
            if (lane == 0) red[warp][t] = s;
        }
    }
    __syncthreads();
    if (threadIdx.x < T) {
        unsigned int t = threadIdx.x;
        float v = 0.0f;
        for (int k = 0; k < 8; k++) v += red[k][t];
        if (row < HC_LR) LO[t * HC_LR + row] = silu(v * 0.25f);
        else INJ[t * HC + row - HC_LR] = v;
    }
}

// x[t][d] = (Σ_c xn[t][c][d] · σ(up[c·HIDDEN + d] · lo[t])) / 4. Grid HIDDEN / 8, block 256:
// a warp per d.
extern "C" __global__ void __launch_bounds__(256) fl_hc_up(
    const float* __restrict__ XN, const float* __restrict__ LO, const unsigned short* __restrict__ Wu,
    float* __restrict__ X, unsigned int T) {
    __shared__ __align__(16) float lo[8 * HC_LR];
    for (unsigned int i = threadIdx.x; i < T * HC_LR; i += 256) lo[i] = LO[i];
    __syncthreads();
    unsigned int lane = threadIdx.x & 31, d = blockIdx.x * 8 + (threadIdx.x >> 5);
    float acc[HC][8];
    #pragma unroll
    for (int c = 0; c < HC; c++)
        #pragma unroll
        for (int t = 0; t < 8; t++) acc[c][t] = 0.0f;
    #pragma unroll
    for (int c = 0; c < HC; c++) {
        const uint4* w = (const uint4*)(Wu + (u64)(c * HIDDEN + d) * HC_LR);
        for (unsigned int p = lane; p < HC_LR / 8; p += 32) {
            uint4 q = w[p];
            float wv[8] = {bf(q.x & 0xffff), bf(q.x >> 16), bf(q.y & 0xffff), bf(q.y >> 16),
                           bf(q.z & 0xffff), bf(q.z >> 16), bf(q.w & 0xffff), bf(q.w >> 16)};
            #pragma unroll
            for (unsigned int t = 0; t < 8; t++) {
                if (t < T) {
                    const float4* l4 = (const float4*)(lo + t * HC_LR + p * 8);
                    float4 a = l4[0], b = l4[1];
                    acc[c][t] += wv[0] * a.x + wv[1] * a.y + wv[2] * a.z + wv[3] * a.w
                               + wv[4] * b.x + wv[5] * b.y + wv[6] * b.z + wv[7] * b.w;
                }
            }
        }
    }
    #pragma unroll
    for (unsigned int t = 0; t < 8; t++) {
        if (t < T) {
            float x = 0.0f;
            #pragma unroll
            for (int c = 0; c < HC; c++) {
                float g = sigm(warp_sum(acc[c][t]));
                x = __fmaf_rn(XN[(u64)t * HC * HIDDEN + c * HIDDEN + d], g, x);
            }
            if (lane == 0) X[(u64)t * HIDDEN + d] = x * 0.25f;
        }
    }
}

// ---- Gated DeltaNet ----

// Grid GDN_CONV / 128 (one block per 128-channel head), block 128.
extern "C" __global__ void fl_gdn_conv(const float* __restrict__ P, unsigned int stride,
                                       const float* __restrict__ H0, const float* __restrict__ Wc,
                                       float* __restrict__ H, unsigned int T, float eps) {
    __shared__ float red[4];
    unsigned int head = blockIdx.x, j = threadIdx.x, ch = head * 128 + j;
    float4 w = ((const float4*)Wc)[ch];
    float e[11];
    e[0] = H0[ch]; e[1] = H0[GDN_CONV + ch]; e[2] = H0[2 * GDN_CONV + ch];
    #pragma unroll
    for (unsigned int t = 0; t < 8; t++) e[3 + t] = t < T ? P[(u64)t * stride + ch] : 0.0f;
    #pragma unroll
    for (unsigned int t = 0; t < 8; t++) {
        if (t >= T) break;
        float acc = e[t] * w.x;
        acc = __fmaf_rn(e[t + 1], w.y, acc);
        acc = __fmaf_rn(e[t + 2], w.z, acc);
        acc = __fmaf_rn(e[t + 3], w.w, acc);
        float v = silu(acc);
        if (head < 2 * GDN_HK) {
            float ss = warp_sum(v * v);
            if ((j & 31) == 0) red[j >> 5] = ss;
            __syncthreads();
            ss = ((red[0] + red[1]) + red[2]) + red[3];
            __syncthreads();
            v *= 1.0f / sqrtf(ss + eps);
        }
        H[(u64)t * GDN_CONV + ch] = v;
    }
}

// hist <- the last 3 rows of [hist | qkv_0 .. qkv_{n-1}], n = min(win[1], T). Grid 40, block 256.
extern "C" __global__ void fl_gdn_conv_commit(float* __restrict__ H0, const float* __restrict__ P,
                                              unsigned int stride, const unsigned int* __restrict__ win,
                                              unsigned int T) {
    unsigned int ch = blockIdx.x * 256 + threadIdx.x;
    if (ch >= GDN_CONV) return;
    unsigned int n = min(win[1], T);
    float old[3] = {H0[ch], H0[GDN_CONV + ch], H0[2 * GDN_CONV + ch]};
    float nw[3];
    #pragma unroll
    for (unsigned int s = 0; s < 3; s++) {
        unsigned int e = n + s;
        nw[s] = e < 3 ? old[e] : P[(u64)(e - 3) * stride + ch];
    }
    #pragma unroll
    for (unsigned int s = 0; s < 3; s++) H0[s * GDN_CONV + ch] = nw[s];
}

// The recurrence + gated norm. Grid GDN_HV (a block per v head), block 512 = 4 row groups of 32
// state rows × 128 columns; each thread holds its 32 state values in registers for the window.
// `commit`: run min(win[1], T) tokens and write the state; else run T, state untouched. Same
// code path either way, so commit's outputs are bitwise verify's.
extern "C" __global__ void __launch_bounds__(512) fl_gdn_step(
    float* __restrict__ S, const float* __restrict__ Hc, const float* __restrict__ P,
    unsigned int stride, const float* __restrict__ DT, const float* __restrict__ SA,
    const float* __restrict__ NW, float* __restrict__ Y, unsigned int T,
    const unsigned int* __restrict__ win, unsigned int commit, float eps) {
    __shared__ float sq[8][128], sk[8][128], sv[8][128], sz[8][128];
    __shared__ float sg[8], sb[8];
    __shared__ float red[4][128];
    __shared__ float red2[4];
    unsigned int hv = blockIdx.x, hk = hv % GDN_HK, tid = threadIdx.x, j = tid & 127, rg = tid >> 7;
    unsigned int n = commit ? min(win[1], T) : T;
    for (unsigned int i = tid; i < n * 128; i += 512) {
        unsigned int t = i >> 7, jj = i & 127;
        const float* hh = Hc + (u64)t * GDN_CONV;
        sq[t][jj] = hh[hk * 128 + jj];
        sk[t][jj] = hh[GDN_HK * 128 + hk * 128 + jj];
        sv[t][jj] = hh[2 * GDN_HK * 128 + hv * 128 + jj];
        sz[t][jj] = P[(u64)t * stride + GDN_Z + hv * 128 + jj];
    }
    if (tid < n) {
        float a = P[(u64)tid * stride + GDN_A + hv] + DT[hv];
        float sp = a > 20.0f ? a : log1pf(expf(a));
        sg[tid] = expf(sp * SA[hv]);
        sb[tid] = sigm(P[(u64)tid * stride + GDN_B + hv]);
    }
    float st[32];
    #pragma unroll
    for (int r = 0; r < 32; r++) st[r] = S[((u64)(rg * 32 + r) * GDN_HV + hv) * 128 + j];
    __syncthreads();
    for (unsigned int t = 0; t < n; t++) {
        float g = sg[t];
        float p = 0.0f;
        #pragma unroll
        for (int r = 0; r < 32; r++) {
            st[r] *= g;
            p = __fmaf_rn(st[r], sk[t][rg * 32 + r], p);
        }
        red[rg][j] = p;
        __syncthreads();
        float skj = ((red[0][j] + red[1][j]) + red[2][j]) + red[3][j];
        float d = (sv[t][j] - skj) * sb[t];
        p = 0.0f;
        #pragma unroll
        for (int r = 0; r < 32; r++) {
            st[r] = __fmaf_rn(sk[t][rg * 32 + r], d, st[r]);
            p = __fmaf_rn(st[r], sq[t][rg * 32 + r], p);
        }
        __syncthreads();
        red[rg][j] = p;
        __syncthreads();
        float o = (((red[0][j] + red[1][j]) + red[2][j]) + red[3][j]) * INV_SQRT_D;
        if (rg == 0) {
            float ss = warp_sum(o * o);
            if ((j & 31) == 0) red2[j >> 5] = ss;
        }
        __syncthreads();
        if (rg == 0) {
            float ss = ((red2[0] + red2[1]) + red2[2]) + red2[3];
            float rs = 1.0f / sqrtf(ss / 128.0f + eps);
            Y[(u64)t * GDN_V + hv * 128 + j] = o * rs * NW[j] * sigm(sz[t][j]);
        }
        __syncthreads();
    }
    if (commit) {
        #pragma unroll
        for (int r = 0; r < 32; r++) S[((u64)(rg * 32 + r) * GDN_HV + hv) * 128 + j] = st[r];
    }
}

// ---- MoE ----

// Grid T, block 32 (a warp per token): top-10 by (logit desc, index asc), then the f64 softmax
// weights renormalised over the ten with the 2^-14 clamp.
extern "C" __global__ void fl_router_topk(const float* __restrict__ L, unsigned int stride,
                                          unsigned int NE, unsigned int* __restrict__ IDS,
                                          float* __restrict__ W) {
    unsigned int t = blockIdx.x, lane = threadIdx.x;
    const float* l = L + (u64)t * stride;
    float v[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        unsigned int e = lane + 32 * i;
        v[i] = e < NE ? l[e] : NEG_INF;
    }
    unsigned int used = 0;
    unsigned int sel[TOPK];
    float selv[TOPK];
    for (int k = 0; k < TOPK; k++) {
        float bv = NEG_INF;
        unsigned int bi = 0xffffffffu;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            unsigned int e = lane + 32 * i;
            if (!((used >> i) & 1) && e < NE && (bi == 0xffffffffu || v[i] > bv)) { bv = v[i]; bi = e; }
        }
        for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            unsigned int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (oi != 0xffffffffu && (bi == 0xffffffffu || ov > bv || (ov == bv && oi < bi))) { bv = ov; bi = oi; }
        }
        if ((bi & 31) == lane) used |= 1u << (bi >> 5);
        sel[k] = bi;
        selv[k] = bv;
    }
    double m = (double)selv[0];
    double z = 0.0;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        unsigned int e = lane + 32 * i;
        if (e < NE) z += exp((double)v[i] - m);
    }
    for (int o = 16; o > 0; o >>= 1) z += __shfl_xor_sync(0xffffffffu, z, o);
    if (lane == 0) {
        double p[TOPK], sum = 0.0;
        for (int k = 0; k < TOPK; k++) { p[k] = exp((double)selv[k] - m) / z; sum += p[k]; }
        double den = fmax(sum, 1.0 / 16384.0);
        for (int k = 0; k < TOPK; k++) {
            IDS[t * TOPK + k] = sel[k];
            W[t * TOPK + k] = (float)(p[k] / den);
        }
    }
}

// The [MoePlan] from router ids and a residency table (EXPERTS addresses, word pairs). Grid 1,
// block 128.
extern "C" __global__ void fl_moe_plan(const unsigned int* __restrict__ IDS, const unsigned int* __restrict__ TABLE,
                                       u64 shared, unsigned int* __restrict__ PLAN, unsigned int T) {
    __shared__ unsigned int ids[PLAN_CAP], first[PLAN_CAP], gid[PLAN_CAP], start[PLAN_CAP + 1];
    __shared__ unsigned int n_routed_groups;
    unsigned int n = T * TOPK, tid = threadIdx.x;
    if (tid < n) ids[tid] = IDS[tid];
    __syncthreads();
    if (tid < n) {
        unsigned int f = tid;
        for (unsigned int j = 0; j < tid; j++) if (ids[j] == ids[tid]) { f = j; break; }
        first[tid] = f;
    }
    __syncthreads();
    if (tid == 0) {
        unsigned int ng = 0, nm = 0, cnt[PLAN_CAP];
        for (unsigned int i = 0; i < n; i++) {
            if (first[i] != i) continue;
            unsigned int e = ids[i];
            u64 p = (u64)TABLE[2 * e] | ((u64)TABLE[2 * e + 1] << 32);
            if (p == 0) { PLAN[PLAN_MISS + nm++] = e; gid[i] = 0xffffffffu; continue; }
            gid[i] = ng;
            cnt[ng] = 0;
            PLAN[PLAN_GP + 2 * ng] = (unsigned int)p;
            PLAN[PLAN_GP + 2 * ng + 1] = (unsigned int)(p >> 32);
            ng++;
        }
        for (unsigned int i = 0; i < n; i++) {
            unsigned int g = gid[first[i]];
            if (g != 0xffffffffu) cnt[g]++;
        }
        unsigned int s = 0;
        for (unsigned int g = 0; g < ng; g++) { start[g] = s; PLAN[PLAN_GS + g] = s; s += cnt[g]; }
        n_routed_groups = ng;
        if (shared != 0) {
            PLAN[PLAN_GP + 2 * ng] = (unsigned int)shared;
            PLAN[PLAN_GP + 2 * ng + 1] = (unsigned int)(shared >> 32);
            start[ng] = s;
            PLAN[PLAN_GS + ng] = s;
            s += T;
            ng++;
        }
        PLAN[PLAN_GS + ng] = s;
        PLAN[0] = ng; PLAN[1] = s; PLAN[2] = nm; PLAN[3] = 0;
    }
    __syncthreads();
    if (tid < n) {
        unsigned int g = gid[first[tid]];
        if (g != 0xffffffffu) {
            unsigned int rank = 0;
            for (unsigned int j = 0; j < tid; j++) rank += ids[j] == ids[tid];
            unsigned int e = start[g] + rank;
            PLAN[PLAN_ET + e] = tid / TOPK;
            PLAN[PLAN_ED + e] = tid;
        }
    }
    if (shared != 0 && tid < T) {
        unsigned int e = start[n_routed_groups] + tid;
        PLAN[PLAN_ET + e] = tid;
        PLAN[PLAN_ED + e] = SHARED_ROW + tid;
    }
}

__device__ __forceinline__ const unsigned char* plan_blob(const unsigned int* PLAN, unsigned int g) {
    return (const unsigned char*)((u64)PLAN[PLAN_GP + 2 * g] | ((u64)PLAN[PLAN_GP + 2 * g + 1] << 32));
}

// Gate and up rows of each planned expert for its tokens, then h = silu(gate)·up quantized per
// [QAct] into HQ (m = PLAN_CAP rows of FF, row = entry). Grid (FF / 32, PLAN_CAP), block 256:
// a block is one group's 32 consecutive h rows (one quantization chunk), a warp 4 of them
// (gate and up: 8 weight rows), applied to all of the group's entries (at most 8).
extern "C" __global__ void __launch_bounds__(256) fl_moe_gu(
    const unsigned int* __restrict__ XQ, unsigned int T, const unsigned int* __restrict__ PLAN,
    unsigned int* __restrict__ HQ) {
    __shared__ float hv[8][32][2];
    __shared__ unsigned int tok[8];
    unsigned int g = blockIdx.y;
    if (g >= PLAN[0]) return;
    const unsigned char* blob = plan_blob(PLAN, g);
    unsigned int e0 = PLAN[PLAN_GS + g], ne = PLAN[PLAN_GS + g + 1] - e0;
    unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (threadIdx.x < ne) tok[threadIdx.x] = PLAN[PLAN_ET + e0 + threadIdx.x];
    __syncthreads();
    const unsigned int kb = HIDDEN / 4, nch = HIDDEN / 32;
    const unsigned int* xs = XQ + (u64)T * kb;
    const unsigned int* xh = xs + (u64)T * nch;
    unsigned int r0 = blockIdx.x * 32 + warp * 4;
    float acc[8][8];
    #pragma unroll
    for (int q = 0; q < 8; q++)
        #pragma unroll
        for (int e = 0; e < 8; e++) acc[q][e] = 0.0f;
    for (unsigned int c = lane; c < nch; c += 32) {
        uint2 cw[8];
        float dw[8];
        #pragma unroll
        for (int q = 0; q < 8; q++) {
            const unsigned char* mat = blob + (q & 1 ? BLOB_UP : 0);
            unsigned int r = r0 + (q >> 1);
            cw[q] = *(const uint2*)(mat + (u64)r * kb + c * 8);
            dw[q] = h2f(((const unsigned short*)(mat + (u64)FF * kb))[r * (HIDDEN / 64) + c / 2]);
        }
        #pragma unroll
        for (int q = 0; q < 8; q++) {
            int m[8];
            expand(cw[q], m);
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                if (e < ne) {
                    unsigned int t = tok[e];
                    int s = chunk_dot(m, XQ + (u64)t * kb + c * 8);
                    float dx = __uint_as_float(xs[(u64)t * nch + c]);
                    int hx = (int)xh[(u64)t * nch + c];
                    acc[q][e] = __fmaf_rn(dw[q] * dx, (float)(s - hx), acc[q][e]);
                }
            }
        }
    }
    #pragma unroll
    for (int q = 0; q < 8; q++)
        #pragma unroll
        for (int e = 0; e < 8; e++)
            if (e < ne) {
                float s = warp_sum(acc[q][e]);
                if (lane == 0) hv[e][warp * 4 + (q >> 1)][q & 1] = s;
            }
    __syncthreads();
    if (warp < ne) {
        float h = silu(hv[warp][lane][0]) * hv[warp][lane][1];
        quant_chunk(h, HQ, PLAN_CAP, FF, e0 + warp, blockIdx.x, lane);
    }
}

// Down rows of each planned expert against its entries' HQ rows, into PARTS[dst]. Grid
// (HIDDEN / 64, PLAN_CAP), block 256: a warp takes 8 rows, lanes the 20 chunks of a row.
extern "C" __global__ void __launch_bounds__(256) fl_moe_down(
    const unsigned int* __restrict__ HQ, const unsigned int* __restrict__ PLAN, float* __restrict__ PARTS) {
    __shared__ unsigned int dst[8];
    unsigned int g = blockIdx.y;
    if (g >= PLAN[0]) return;
    const unsigned char* mat = plan_blob(PLAN, g) + BLOB_DOWN;
    unsigned int e0 = PLAN[PLAN_GS + g], ne = PLAN[PLAN_GS + g + 1] - e0;
    unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (threadIdx.x < ne) dst[threadIdx.x] = PLAN[PLAN_ED + e0 + threadIdx.x];
    __syncthreads();
    const unsigned int kb = FF / 4, nch = FF / 32;
    const unsigned int* xs = HQ + (u64)PLAN_CAP * kb;
    const unsigned int* xh = xs + (u64)PLAN_CAP * nch;
    unsigned int r0 = blockIdx.x * 64 + warp * 8;
    float acc[8][8];
    #pragma unroll
    for (int q = 0; q < 8; q++)
        #pragma unroll
        for (int e = 0; e < 8; e++) acc[q][e] = 0.0f;
    if (lane < nch) {
        unsigned int c = lane;
        uint2 cw[8];
        float dw[8];
        #pragma unroll
        for (int q = 0; q < 8; q++) {
            cw[q] = *(const uint2*)(mat + (u64)(r0 + q) * kb + c * 8);
            dw[q] = h2f(((const unsigned short*)(mat + (u64)HIDDEN * kb))[(r0 + q) * (FF / 64) + c / 2]);
        }
        #pragma unroll
        for (int q = 0; q < 8; q++) {
            int m[8];
            expand(cw[q], m);
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                if (e < ne) {
                    unsigned int row = e0 + e;
                    int s = chunk_dot(m, HQ + (u64)row * kb + c * 8);
                    float dx = __uint_as_float(xs[(u64)row * nch + c]);
                    int hx = (int)xh[(u64)row * nch + c];
                    acc[q][e] = __fmaf_rn(dw[q] * dx, (float)(s - hx), acc[q][e]);
                }
            }
        }
    }
    #pragma unroll
    for (int q = 0; q < 8; q++)
        #pragma unroll
        for (int e = 0; e < 8; e++)
            if (e < ne) {
                float s = warp_sum(acc[q][e]);
                if (lane == 0) PARTS[(u64)dst[e] * HIDDEN + r0 + q] = s;
            }
}

// y[t] = Σ_i w[t][i] parts[t·10 + i] (+ σ(logit[t][sg]) parts[SHARED_ROW + t]). Grid
// (HIDDEN / 256, T).
extern "C" __global__ void fl_moe_combine(const float* __restrict__ PARTS, const float* __restrict__ W,
                                          const float* __restrict__ L, unsigned int stride, int sg,
                                          float* __restrict__ Y) {
    unsigned int t = blockIdx.y, d = blockIdx.x * 256 + threadIdx.x;
    float acc = 0.0f;
    #pragma unroll
    for (int i = 0; i < TOPK; i++) acc = __fmaf_rn(W[t * TOPK + i], PARTS[(u64)(t * TOPK + i) * HIDDEN + d], acc);
    if (sg >= 0) acc = __fmaf_rn(sigm(L[(u64)t * stride + sg]), PARTS[(u64)(SHARED_ROW + t) * HIDDEN + d], acc);
    Y[(u64)t * HIDDEN + d] = acc;
}

// ---- QSA ----

// RMSNorm then NeoX rope on the first 64 dims of a D-wide head (D = 256 or 128), by the whole
// 256-thread block. `buf` is shared scratch of 256 floats; returns this thread's output (j < D).
__device__ float norm_rope(const float* __restrict__ src, const float* __restrict__ w, unsigned int D,
                           const float* __restrict__ COS, const float* __restrict__ SIN, unsigned int pos,
                           float eps, float* buf, float* red) {
    unsigned int j = threadIdx.x;
    float x = j < D ? src[j] : 0.0f;
    float ss = warp_sum(x * x);
    if ((j & 31) == 0) red[j >> 5] = ss;
    __syncthreads();
    ss = 0.0f;
    for (int k = 0; k < 8; k++) ss += red[k];
    float rs = 1.0f / sqrtf(ss / (float)D + eps);
    float y = j < D ? x * rs * w[j] : 0.0f;
    buf[j] = y;
    __syncthreads();
    if (j < 32) {
        float c = COS[pos * 32 + j], s = SIN[pos * 32 + j];
        y = __fmaf_rn(buf[j], c, -(buf[j + 32] * s));
    } else if (j < 64) {
        float c = COS[pos * 32 + j - 32], s = SIN[pos * 32 + j - 32];
        y = __fmaf_rn(buf[j], c, buf[j - 32] * s);
    }
    __syncthreads();
    return y;
}

// Grid (34, T), block 256. Jobs: 0..23 q heads, 24..25 k heads, 26..27 v heads, 28..31 indexer
// q heads, 32 raw indexer key into the ring, 33 pool the block this token completes.
extern "C" __global__ void fl_qsa_prep(
    const float* __restrict__ P, unsigned int stride, const unsigned int* __restrict__ win,
    const float* __restrict__ QN, const float* __restrict__ KN, const float* __restrict__ IQN,
    const float* __restrict__ IKN, const float* __restrict__ COS, const float* __restrict__ SIN,
    float* __restrict__ Q, unsigned short* __restrict__ KC, unsigned short* __restrict__ VC,
    float* __restrict__ RING, float* __restrict__ POOLED, unsigned int T, float eps) {
    __shared__ float buf[256], red[8], mean[128];
    unsigned int job = blockIdx.x, t = blockIdx.y, j = threadIdx.x;
    unsigned int pos0 = win[0], pos = pos0 + t;
    const float* p = P + (u64)t * stride;
    if (job < 24) {
        float y = norm_rope(p + job * 512, QN, 256, COS, SIN, pos, eps, buf, red);
        Q[((u64)t * QSA_HEADS + job) * QSA_D + j] = y;
    } else if (job < 26) {
        unsigned int g = job - 24;
        float y = norm_rope(p + QSA_K + g * 256, KN, 256, COS, SIN, pos, eps, buf, red);
        KC[((u64)pos * QSA_KV + g) * QSA_D + j] = to_bf16(y);
    } else if (job < 28) {
        unsigned int g = job - 26;
        VC[((u64)pos * QSA_KV + g) * QSA_D + j] = to_bf16(p[QSA_V + g * 256 + j]);
    } else if (job < 32) {
        unsigned int h = job - 28;
        float y = norm_rope(p + QSA_IQ + h * 128, IQN, 128, COS, SIN, pos, eps, buf, red);
        if (j < 128) Q[(u64)T * QSA_HEADS * QSA_D + ((u64)t * IDX_HEADS + h) * IDX_D + j] = y;
    } else if (job == 32) {
        if (j < 128) RING[(pos % 16) * IDX_D + j] = p[QSA_IK + j];
    } else {
        if (pos % 4 != 3) return;
        unsigned int b = pos / 4, c0 = 4 * b;
        if (j < 128) {
            float r[4];
            for (int i = 0; i < 4; i++) {
                unsigned int c = c0 + i;
                r[i] = c >= pos0 ? P[(u64)(c - pos0) * stride + QSA_IK + j] : RING[(c % 16) * IDX_D + j];
            }
            mean[j] = (((r[0] + r[1]) + r[2]) + r[3]) * 0.25f;
        }
        __syncthreads();
        float y = norm_rope(mean, IKN, 128, COS, SIN, c0, eps, buf, red);
        if (j < 128) POOLED[(u64)b * IDX_D + j] = y;
    }
}

// Block scores: grid (ceil(MAXB / 8), T), block 256, a warp per pooled block.
extern "C" __global__ void fl_qsa_scores(const float* __restrict__ POOLED, const float* __restrict__ IQ,
                                         const unsigned int* __restrict__ win, float* __restrict__ SC,
                                         unsigned int MAXB) {
    unsigned int t = blockIdx.y, lane = threadIdx.x & 31;
    unsigned int b = blockIdx.x * 8 + (threadIdx.x >> 5);
    unsigned int n_bid = (win[0] + t + 1) / 4;
    if (b >= n_bid || b >= MAXB) return;
    float4 pk = ((const float4*)(POOLED + (u64)b * IDX_D))[lane];
    float total = 0.0f;
    #pragma unroll
    for (int h = 0; h < IDX_HEADS; h++) {
        float4 q = ((const float4*)(IQ + ((u64)t * IDX_HEADS + h) * IDX_D))[lane];
        float a = q.x * pk.x;
        a = __fmaf_rn(q.y, pk.y, a);
        a = __fmaf_rn(q.z, pk.z, a);
        a = __fmaf_rn(q.w, pk.w, a);
        a = warp_sum(a);
        total += a > 0.0f ? a : 0.0f;
    }
    if (lane == 0) SC[(u64)t * MAXB + b] = total;
}

__device__ __forceinline__ unsigned int ordered(float x) {
    unsigned int b = __float_as_uint(x);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// Inclusive block scan of one value per thread (1024 threads).
__device__ unsigned int block_scan(unsigned int v, unsigned int* sc) {
    unsigned int tid = threadIdx.x;
    sc[tid] = v;
    __syncthreads();
    for (unsigned int o = 1; o < 1024; o <<= 1) {
        unsigned int a = tid >= o ? sc[tid - o] : 0;
        __syncthreads();
        sc[tid] += a;
        __syncthreads();
    }
    unsigned int r = sc[tid];
    __syncthreads();
    return r;
}

#define SEL_MAXB 8192
// Top cells per token. Grid T, block 1024.
extern "C" __global__ void __launch_bounds__(1024) fl_qsa_select(
    const float* __restrict__ SC, const unsigned int* __restrict__ win, unsigned int* __restrict__ IDS,
    unsigned int MAXB) {
    __shared__ unsigned int keys[SEL_MAXB];
    __shared__ unsigned int hist[256];
    __shared__ unsigned int scan[1024];
    __shared__ unsigned int sh_digit, sh_k;
    unsigned int t = blockIdx.x, tid = threadIdx.x;
    unsigned int n_kv = win[0] + t + 1;
    unsigned int* out = IDS + (u64)t * QSA_WIDTH;
    if (n_kv <= QSA_WIDTH) {
        for (unsigned int i = tid; i < n_kv; i += 1024) out[i] = i;
        return;
    }
    unsigned int n_bid = n_kv / 4, tail = n_kv % 4;
    unsigned int R = QSA_WIDTH - tail, full = R / 4, rem = R % 4, need = full + (rem > 0);
    for (unsigned int b = tid; b < n_bid; b += 1024) keys[b] = ordered(SC[(u64)t * MAXB + b]);
    __syncthreads();
    unsigned int prefix = 0, pmask = 0, k = need;
    for (int pass = 0; pass < 4; pass++) {
        unsigned int shift = 24 - 8 * pass;
        for (unsigned int i = tid; i < 256; i += 1024) hist[i] = 0;
        __syncthreads();
        for (unsigned int b = tid; b < n_bid; b += 1024)
            if ((keys[b] & pmask) == prefix) atomicAdd(&hist[(keys[b] >> shift) & 255u], 1u);
        __syncthreads();
        if (tid == 0) {
            unsigned int acc = 0;
            for (int dg = 255; dg >= 0; dg--) {
                if (acc + hist[dg] >= k) { sh_digit = dg; sh_k = k - acc; break; }
                acc += hist[dg];
            }
        }
        __syncthreads();
        prefix |= sh_digit << shift;
        pmask |= 255u << shift;
        k = sh_k;
        __syncthreads();
    }
    unsigned int v = prefix, take_eq = k;
    unsigned int per = (n_bid + 1023) / 1024, b0 = min(tid * per, n_bid), b1 = min(b0 + per, n_bid);
    unsigned int eq = 0;
    for (unsigned int b = b0; b < b1; b++) eq += keys[b] == v;
    unsigned int eq_before = block_scan(eq, scan) - eq;
    unsigned int cells = 0, r = eq_before;
    for (unsigned int b = b0; b < b1; b++) {
        unsigned int key = keys[b];
        unsigned int c = 0;
        if (key > v) c = 4;
        else if (key == v) {
            if (r < take_eq) c = (r == take_eq - 1 && rem > 0) ? rem : 4;
            r++;
        }
        cells += c;
    }
    unsigned int off = block_scan(cells, scan) - cells;
    r = eq_before;
    for (unsigned int b = b0; b < b1; b++) {
        unsigned int key = keys[b];
        unsigned int c = 0;
        if (key > v) c = 4;
        else if (key == v) {
            if (r < take_eq) c = (r == take_eq - 1 && rem > 0) ? rem : 4;
            r++;
        }
        for (unsigned int i = 0; i < c; i++) out[off++] = b * 4 + i;
    }
    if (tid == 0)
        for (unsigned int i = 0; i < tail; i++) out[QSA_WIDTH - tail + i] = n_bid * 4 + i;
}

// Split-K attention: grid (QSA_NCH, QSA_KV, T), block 256. A block reads its chunk of 64
// selected cells' K and V once for the 12 q heads that share the kv head; writes per head
// [m, l, acc[256]] partials.
extern "C" __global__ void __launch_bounds__(256) fl_qsa_attend(
    const float* __restrict__ Q, const unsigned short* __restrict__ KC, const unsigned short* __restrict__ VC,
    const unsigned int* __restrict__ IDS, const unsigned int* __restrict__ win, float* __restrict__ PART) {
    __shared__ __align__(16) float qs[12][256];
    __shared__ float s[12][64];
    __shared__ unsigned int cells[64];
    unsigned int ch = blockIdx.x, g = blockIdx.y, t = blockIdx.z, tid = threadIdx.x;
    unsigned int lane = tid & 31, warp = tid >> 5;
    unsigned int n_sel = min(win[0] + t + 1, (unsigned int)QSA_WIDTH);
    unsigned int i0 = ch * QSA_CHUNK;
    if (i0 >= n_sel) return;
    unsigned int n = min((unsigned int)QSA_CHUNK, n_sel - i0);
    for (unsigned int i = tid; i < 12 * 256; i += 256)
        (&qs[0][0])[i] = Q[((u64)t * QSA_HEADS + g * 12) * QSA_D + i];
    if (tid < n) cells[tid] = IDS[(u64)t * QSA_WIDTH + i0 + tid];
    __syncthreads();
    for (unsigned int i = warp; i < n; i += 8) {
        uint4 kq = ((const uint4*)(KC + ((u64)cells[i] * QSA_KV + g) * QSA_D))[lane];
        float kv[8] = {bf(kq.x & 0xffff), bf(kq.x >> 16), bf(kq.y & 0xffff), bf(kq.y >> 16),
                       bf(kq.z & 0xffff), bf(kq.z >> 16), bf(kq.w & 0xffff), bf(kq.w >> 16)};
        #pragma unroll
        for (int h = 0; h < 12; h++) {
            const float4* q4 = (const float4*)(&qs[h][lane * 8]);
            float4 a = q4[0], b = q4[1];
            float d = kv[0] * a.x + kv[1] * a.y + kv[2] * a.z + kv[3] * a.w
                    + kv[4] * b.x + kv[5] * b.y + kv[6] * b.z + kv[7] * b.w;
            d = warp_sum(d);
            if (lane == 0) s[h][i] = d * 0.0625f;
        }
    }
    __syncthreads();
    for (unsigned int h = warp; h < 12; h += 8) {
        float a = lane < n ? s[h][lane] : NEG_INF, b = lane + 32 < n ? s[h][lane + 32] : NEG_INF;
        float m = warp_max(fmaxf(a, b));
        float pa = lane < n ? expf(a - m) : 0.0f, pb = lane + 32 < n ? expf(b - m) : 0.0f;
        if (lane < n) s[h][lane] = pa;
        if (lane + 32 < n) s[h][lane + 32] = pb;
        float l = warp_sum(pa + pb);
        if (lane == 0) {
            float* part = PART + (((u64)t * QSA_HEADS + g * 12 + h) * QSA_NCH + ch) * (QSA_D + 2);
            part[0] = m;
            part[1] = l;
        }
    }
    __syncthreads();
    float acc[12];
    #pragma unroll
    for (int h = 0; h < 12; h++) acc[h] = 0.0f;
    for (unsigned int i = 0; i < n; i++) {
        float vv = bf(VC[((u64)cells[i] * QSA_KV + g) * QSA_D + tid]);
        #pragma unroll
        for (int h = 0; h < 12; h++) acc[h] = __fmaf_rn(s[h][i], vv, acc[h]);
    }
    #pragma unroll
    for (int h = 0; h < 12; h++)
        PART[(((u64)t * QSA_HEADS + g * 12 + h) * QSA_NCH + ch) * (QSA_D + 2) + 2 + tid] = acc[h];
}

// Merge the chunks and apply the sigmoid gate. Grid (QSA_HEADS, T), block 256.
extern "C" __global__ void fl_qsa_merge(const float* __restrict__ PART, const float* __restrict__ P,
                                        unsigned int stride, const unsigned int* __restrict__ win,
                                        float* __restrict__ OUT) {
    unsigned int h = blockIdx.x, t = blockIdx.y, j = threadIdx.x;
    unsigned int n_sel = min(win[0] + t + 1, (unsigned int)QSA_WIDTH);
    unsigned int nch = (n_sel + QSA_CHUNK - 1) / QSA_CHUNK;
    const float* part = PART + ((u64)t * QSA_HEADS + h) * QSA_NCH * (QSA_D + 2);
    float M = NEG_INF;
    for (unsigned int c = 0; c < nch; c++) M = fmaxf(M, part[c * (QSA_D + 2)]);
    float L = 0.0f, o = 0.0f;
    for (unsigned int c = 0; c < nch; c++) {
        float f = expf(part[c * (QSA_D + 2)] - M);
        L += part[c * (QSA_D + 2) + 1] * f;
        o += part[c * (QSA_D + 2) + 2 + j] * f;
    }
    float gate = P[(u64)t * stride + h * 512 + 256 + j];
    OUT[(u64)t * QSA_HEADS * QSA_D + h * QSA_D + j] = o / L * sigm(gate);
}
"#;
