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
#define GU_CODES 0
#define GU_SCALES 819200
#define DOWN_CODES 921600
#define DOWN_SCALES 1331200
#define INV_SQRT_D __int_as_float(0x3db504f3)

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
__device__ __forceinline__ int chunk_dot(const int* m, const unsigned int* __restrict__ xw) {
    uint4 a = ((const uint4*)xw)[0], b = ((const uint4*)xw)[1];
    int s = 0;
    s = __dp4a(m[0], (int)a.x, s); s = __dp4a(m[1], (int)a.y, s);
    s = __dp4a(m[2], (int)a.z, s); s = __dp4a(m[3], (int)a.w, s);
    s = __dp4a(m[4], (int)b.x, s); s = __dp4a(m[5], (int)b.y, s);
    s = __dp4a(m[6], (int)b.z, s); s = __dp4a(m[7], (int)b.w, s);
    return s;
}

// Multi-column GEMV, Y[t, o] = W[o] · x[t] for T <= 8 columns, three weight formats:
//   FMT 0: bf16 W [N][K] against f32 X [T][K];
//   FMT 1: repacked Q2_0 against int8 XQ ([QAct], m = T): per chunk fma(d_w d_x, S − hx, acc);
//   FMT 2: repacked Q4X (MLX affine, group 64) against int8 XQ: per chunk
//          fma(d_x, fma(s, S, b · hx), acc).
// Each warp owns R = 4 consecutive rows and loads one 16-byte weight vector per row per step,
// applying it to every column, so each activation load serves four rows. KS warps split K for
// one row group (KS = 1, 2, 4, 8; 8 / KS row groups per 256-thread block) and reduce through
// shared memory, which keeps short-N / long-K shapes (10240 -> 320) on many SMs.
#define GR 4
template <int FMT, int T>
__device__ __forceinline__ void gemv_body(const float* __restrict__ X, const unsigned int* __restrict__ XQ,
                                          const unsigned char* __restrict__ W, float* __restrict__ Y,
                                          unsigned int K, unsigned int N, unsigned int KS) {
    __shared__ float red[8][GR * T];
    const unsigned int EPV = FMT == 0 ? 8 : (FMT == 1 ? 64 : (FMT == 2 ? 32 : 16));
    unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    unsigned int kw = warp % KS, rg = warp / KS, groups = 8 / KS;
    unsigned int row0 = (blockIdx.x * groups + rg) * GR;
    unsigned int nv = K / EPV, v0 = kw * nv / KS, v1 = (kw + 1) * nv / KS;
    u64 rowbytes = FMT == 0 ? (u64)K * 2 : (FMT == 1 ? K / 4 : (FMT == 2 ? K / 2 : K));
    const unsigned short* sc = (const unsigned short*)(W + (u64)N * rowbytes);
    const unsigned short* bi = sc + (u64)N * (K / 64);
    unsigned int kb = K / 4, nch = K / 32;
    const unsigned int* xs = XQ + (u64)T * kb;
    const unsigned int* xh = xs + (u64)T * nch;
    float acc[GR][T];
    #pragma unroll
    for (int r = 0; r < GR; r++)
        #pragma unroll
        for (int t = 0; t < T; t++) acc[r][t] = 0.0f;
    if (row0 < N) {
        for (unsigned int v = v0 + lane; v < v1; v += 32) {
            uint4 wv[GR];
            float ws[GR], wb[GR];
            #pragma unroll
            for (int r = 0; r < GR; r++) {
                unsigned int o = min(row0 + r, N - 1);
                wv[r] = ((const uint4*)(W + (u64)o * rowbytes))[v];
                if (FMT == 1) ws[r] = h2f(sc[(u64)o * (K / 64) + v]);
                if (FMT == 3) ws[r] = h2f(sc[(u64)o * (K / 32) + v / 2]);
                if (FMT == 2) {
                    ws[r] = bf(sc[(u64)o * (K / 64) + v / 2]);
                    wb[r] = bf(bi[(u64)o * (K / 64) + v / 2]);
                }
            }
            // dp4a operands, expanded once per row and reused for every column.
            int m[GR][16];
            if (FMT == 1) {
                #pragma unroll
                for (int r = 0; r < GR; r++) {
                    expand(make_uint2(wv[r].x, wv[r].y), &m[r][0]);
                    expand(make_uint2(wv[r].z, wv[r].w), &m[r][8]);
                }
            } else if (FMT == 2) {
                #pragma unroll
                for (int r = 0; r < GR; r++) {
                    unsigned int q[4] = {wv[r].x, wv[r].y, wv[r].z, wv[r].w};
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        m[r][2 * i] = (int)(q[i] & 0x0f0f0f0fu);
                        m[r][2 * i + 1] = (int)((q[i] >> 4) & 0x0f0f0f0fu);
                    }
                }
            }
            #pragma unroll
            for (int t = 0; t < T; t++) {
                if (FMT == 0) {
                    const float4* x4 = (const float4*)(X + (u64)t * K + v * 8);
                    float4 a = x4[0], b = x4[1];
                    #pragma unroll
                    for (int r = 0; r < GR; r++) {
                        uint4 p = wv[r];
                        acc[r][t] += bf(p.x & 0xffff) * a.x + bf(p.x >> 16) * a.y + bf(p.y & 0xffff) * a.z
                                   + bf(p.y >> 16) * a.w + bf(p.z & 0xffff) * b.x + bf(p.z >> 16) * b.y
                                   + bf(p.w & 0xffff) * b.z + bf(p.w >> 16) * b.w;
                    }
                } else if (FMT == 1) {
                    const unsigned int* xw = XQ + (u64)t * kb + v * 16;
                    float dx0 = __uint_as_float(xs[(u64)t * nch + 2 * v]);
                    float dx1 = __uint_as_float(xs[(u64)t * nch + 2 * v + 1]);
                    int hx0 = (int)xh[(u64)t * nch + 2 * v], hx1 = (int)xh[(u64)t * nch + 2 * v + 1];
                    #pragma unroll
                    for (int r = 0; r < GR; r++) {
                        int s0 = chunk_dot(&m[r][0], xw);
                        int s1 = chunk_dot(&m[r][8], xw + 8);
                        acc[r][t] = __fmaf_rn(ws[r] * dx0, (float)(s0 - hx0), acc[r][t]);
                        acc[r][t] = __fmaf_rn(ws[r] * dx1, (float)(s1 - hx1), acc[r][t]);
                    }
                } else if (FMT == 3) {
                    // Half a chunk: 16 int8 weights (activation order) against 4 activation words.
                    uint4 xw = *(const uint4*)(XQ + (u64)t * kb + v * 4);
                    float dx = __uint_as_float(xs[(u64)t * nch + v / 2]);
                    #pragma unroll
                    for (int r = 0; r < GR; r++) {
                        int S = __dp4a((int)wv[r].x, (int)xw.x, 0);
                        S = __dp4a((int)wv[r].y, (int)xw.y, S);
                        S = __dp4a((int)wv[r].z, (int)xw.z, S);
                        S = __dp4a((int)wv[r].w, (int)xw.w, S);
                        acc[r][t] = __fmaf_rn(ws[r] * dx, (float)S, acc[r][t]);
                    }
                } else {
                    const unsigned int* xw = XQ + (u64)t * kb + v * 8;
                    float dx = __uint_as_float(xs[(u64)t * nch + v]);
                    int hx = (int)xh[(u64)t * nch + v];
                    #pragma unroll
                    for (int r = 0; r < GR; r++) {
                        int S = chunk_dot(&m[r][0], xw);
                        float val = __fmaf_rn(ws[r], (float)S, wb[r] * (float)hx);
                        acc[r][t] = __fmaf_rn(dx, val, acc[r][t]);
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < GR; r++)
        #pragma unroll
        for (int t = 0; t < T; t++) {
            float v = warp_sum(acc[r][t]);
            if (KS == 1) {
                if (lane == 0 && row0 + r < N) Y[(u64)t * N + row0 + r] = v;
            } else if (lane == 0) {
                red[warp][r * T + t] = v;
            }
        }
    if (KS == 1) return;
    __syncthreads();
    unsigned int i = threadIdx.x;
    if (i < groups * GR * T) {
        unsigned int g = i / (GR * T), rt = i % (GR * T), r = rt / T, t = rt % T;
        unsigned int o = (blockIdx.x * groups + g) * GR + r;
        float v = 0.0f;
        for (unsigned int k = 0; k < KS; k++) v += red[g * KS + k][rt];
        if (o < N) Y[(u64)t * N + o] = v;
    }
}

#define GEMV(FMT, NAME, T) \
extern "C" __global__ void __launch_bounds__(256) fl_##NAME##_t##T( \
    const float* __restrict__ X, const unsigned int* __restrict__ XQ, \
    const unsigned char* __restrict__ W, float* __restrict__ Y, unsigned int K, unsigned int N, \
    unsigned int KS) { \
    gemv_body<FMT, T>(X, XQ, W, Y, K, N, KS); \
}
#define GEMV_ALL(FMT, NAME) GEMV(FMT, NAME, 1) GEMV(FMT, NAME, 2) GEMV(FMT, NAME, 3) GEMV(FMT, NAME, 4) \
    GEMV(FMT, NAME, 5) GEMV(FMT, NAME, 6) GEMV(FMT, NAME, 7) GEMV(FMT, NAME, 8)
GEMV_ALL(0, bf16_gemv)
GEMV_ALL(1, q2_gemv)
GEMV_ALL(2, q4x_gemv)
GEMV_ALL(3, q8x_gemv)

// ---- hyper-connections ----

// R[t][c] += y[t] * 2σ(inj[t][c] / 4). Grid (HIDDEN / 256, HC, T).
extern "C" __global__ void fl_hc_write(float* __restrict__ R, const float* __restrict__ Y,
                                       const float* __restrict__ I) {
    unsigned int d = blockIdx.x * 256 + threadIdx.x, c = blockIdx.y, t = blockIdx.z;
    float g = 2.0f / (1.0f + expf(-(I[t * HC + c] * 0.25f)));
    u64 i = ((u64)t * HC + c) * HIDDEN + d;
    R[i] = __fmaf_rn(Y[(u64)t * HIDDEN + d], g, R[i]);
}

// Per (stream, token): the pending write, then xn = R · rsqrt(mean R² + eps) · w. `mode` 0: no
// write; 1: R += Yp · 2σ(Ip / 4); 2: the same with Yp the MoE combine of PARTS (router weights
// Wr, shared-gate logit column `sg` of L, -1 for none), computed here exactly as fl_moe_combine
// does. Grid (HC, T), block 256.
extern "C" __global__ void fl_hc_norm(float* __restrict__ R, const float* __restrict__ Yp,
                                      const float* __restrict__ Ip, unsigned int mode,
                                      const float* __restrict__ Wn, float* __restrict__ XN, float eps,
                                      const float* __restrict__ PARTS, const float* __restrict__ Wr,
                                      const float* __restrict__ L, unsigned int stride, int sg) {
    __shared__ float red[8];
    unsigned int c = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
    float* row = R + ((u64)t * HC + c) * HIDDEN;
    float g = mode ? 2.0f / (1.0f + expf(-(Ip[t * HC + c] * 0.25f))) : 0.0f;
    float gs = (mode == 2 && sg >= 0) ? sigm(L[(u64)t * stride + sg]) : 0.0f;
    float v[10], ss = 0.0f;
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        unsigned int d = tid + 256 * i;
        float x = row[d];
        if (mode) {
            float y;
            if (mode == 1) {
                y = Yp[(u64)t * HIDDEN + d];
            } else {
                y = 0.0f;
                #pragma unroll
                for (int k = 0; k < TOPK; k++)
                    y = __fmaf_rn(Wr[t * TOPK + k], PARTS[(u64)(t * TOPK + k) * HIDDEN + d], y);
                if (sg >= 0) y = __fmaf_rn(gs, PARTS[(u64)(SHARED_ROW + t) * HIDDEN + d], y);
            }
            x = __fmaf_rn(y, g, x);
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
// next HC rows. Grid ceil(rows / 4), block 256: a block takes 4 rows (so each xn load serves
// four), its 8 warps split K = 10240 and reduce through shared memory.
extern "C" __global__ void __launch_bounds__(256) fl_hc_down(
    const float* __restrict__ XN, const unsigned short* __restrict__ Wd,
    const unsigned short* __restrict__ Wi, float* __restrict__ LO, float* __restrict__ INJ,
    unsigned int T, unsigned int rows) {
    __shared__ float red[8][4][8];
    unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    unsigned int row0 = blockIdx.x * 4;
    const unsigned short* w[4];
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        unsigned int row = min(row0 + r, rows - 1);
        w[r] = (row < HC_LR ? Wd + (u64)row * HC * HIDDEN : Wi + (u64)(row - HC_LR) * HC * HIDDEN) + warp * 1280;
    }
    float acc[4][8];
    #pragma unroll
    for (int r = 0; r < 4; r++)
        #pragma unroll
        for (int t = 0; t < 8; t++) acc[r][t] = 0.0f;
    for (int j = 0; j < 5; j++) {
        unsigned int i = lane + 32 * j;
        uint4 pq[4];
        #pragma unroll
        for (int r = 0; r < 4; r++) pq[r] = ((const uint4*)w[r])[i];
        #pragma unroll
        for (unsigned int t = 0; t < 8; t++) {
            if (t < T) {
                const float4* x4 = (const float4*)(XN + (u64)t * HC * HIDDEN + warp * 1280 + i * 8);
                float4 a = x4[0], b = x4[1];
                #pragma unroll
                for (int r = 0; r < 4; r++) {
                    uint4 p = pq[r];
                    acc[r][t] += bf(p.x & 0xffff) * a.x + bf(p.x >> 16) * a.y + bf(p.y & 0xffff) * a.z
                               + bf(p.y >> 16) * a.w + bf(p.z & 0xffff) * b.x + bf(p.z >> 16) * b.y
                               + bf(p.w & 0xffff) * b.z + bf(p.w >> 16) * b.w;
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < 4; r++)
        #pragma unroll
        for (unsigned int t = 0; t < 8; t++)
            if (t < T) {
                float s = warp_sum(acc[r][t]);
                if (lane == 0) red[warp][r][t] = s;
            }
    __syncthreads();
    if (threadIdx.x < 4 * T) {
        unsigned int r = threadIdx.x / T, t = threadIdx.x % T, row = row0 + r;
        if (row < rows) {
            float v = 0.0f;
            for (int k = 0; k < 8; k++) v += red[k][r][t];
            if (row < HC_LR) LO[t * HC_LR + row] = silu(v * 0.25f);
            else INJ[t * HC + row - HC_LR] = v;
        }
    }
}

// x[t][d] = (Σ_c xn[t][c][d] · σ(up[c·HIDDEN + d] · lo[t])) / 4; with `quant`, x also as int8
// activations ([QAct], m = T rows of HIDDEN). Grid HIDDEN / 32, block 1024: a warp per d, so a
// block's 32 outputs are one quantization chunk.
template <int T>
__device__ __forceinline__ void hc_up_body(const float* __restrict__ XN, const float* __restrict__ LO,
                                           const unsigned short* __restrict__ Wu, float* __restrict__ X,
                                           unsigned int* __restrict__ XQ, unsigned int quant) {
    __shared__ __align__(16) float lo[T * HC_LR];
    __shared__ float xs[T][32];
    for (unsigned int i = threadIdx.x; i < T * HC_LR; i += 1024) lo[i] = LO[i];
    __syncthreads();
    unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, d = blockIdx.x * 32 + warp;
    // Repacked up rows: d's 160 vectors are [p][c]; lane l reads stream l % 4, groups l / 4 + 8j.
    const uint4* w = (const uint4*)(Wu + (u64)d * HC * HC_LR);
    uint4 q[5];
    #pragma unroll
    for (int j = 0; j < 5; j++) q[j] = w[lane + 32 * j];
    float acc[T];
    #pragma unroll
    for (int t = 0; t < T; t++) acc[t] = 0.0f;
    #pragma unroll
    for (int j = 0; j < 5; j++) {
        unsigned int p = (lane >> 2) + 8 * j;
        float wv[8] = {bf(q[j].x & 0xffff), bf(q[j].x >> 16), bf(q[j].y & 0xffff), bf(q[j].y >> 16),
                       bf(q[j].z & 0xffff), bf(q[j].z >> 16), bf(q[j].w & 0xffff), bf(q[j].w >> 16)};
        #pragma unroll
        for (int t = 0; t < T; t++) {
            const float4* l4 = (const float4*)(lo + t * HC_LR + p * 8);
            float4 a = l4[0], b = l4[1];
            acc[t] += wv[0] * a.x + wv[1] * a.y + wv[2] * a.z + wv[3] * a.w
                    + wv[4] * b.x + wv[5] * b.y + wv[6] * b.z + wv[7] * b.w;
        }
    }
    #pragma unroll
    for (int t = 0; t < T; t++) {
        float v = acc[t];
        v += __shfl_xor_sync(0xffffffffu, v, 4);
        v += __shfl_xor_sync(0xffffffffu, v, 8);
        v += __shfl_xor_sync(0xffffffffu, v, 16);
        float g = sigm(v);
        float g1 = __shfl_sync(0xffffffffu, g, 1), g2 = __shfl_sync(0xffffffffu, g, 2), g3 = __shfl_sync(0xffffffffu, g, 3);
        if (lane == 0) {
            const float* xn = XN + (u64)t * HC * HIDDEN + d;
            float x = __fmaf_rn(xn[0], g, 0.0f);
            x = __fmaf_rn(xn[HIDDEN], g1, x);
            x = __fmaf_rn(xn[2 * HIDDEN], g2, x);
            x = __fmaf_rn(xn[3 * HIDDEN], g3, x);
            x *= 0.25f;
            X[(u64)t * HIDDEN + d] = x;
            xs[t][warp] = x;
        }
    }
    if (!quant) return;
    __syncthreads();
    if (warp < T) quant_chunk(xs[warp][lane], XQ, T, HIDDEN, warp, blockIdx.x, lane);
}
#define HC_UP(T) \
extern "C" __global__ void __launch_bounds__(1024) fl_hc_up_t##T( \
    const float* __restrict__ XN, const float* __restrict__ LO, const unsigned short* __restrict__ Wu, \
    float* __restrict__ X, unsigned int* __restrict__ XQ, unsigned int quant) { \
    hc_up_body<T>(XN, LO, Wu, X, XQ, quant); \
}
HC_UP(1) HC_UP(2) HC_UP(3) HC_UP(4) HC_UP(5) HC_UP(6) HC_UP(7) HC_UP(8)

// ---- grid barrier and completion counters ----

// All blocks of a co-resident grid meet here. `bar` = [count, generation], self-resetting, so a
// captured graph can replay it.
__device__ void grid_bar(unsigned int* bar, unsigned int nb) {
    __syncthreads();
    if (threadIdx.x == 0) {
        volatile unsigned int* gen = bar + 1;
        unsigned int g = *gen;
        __threadfence();
        if (atomicAdd(bar, 1u) == nb - 1) {
            atomicExch(bar, 0u);
            __threadfence();
            atomicAdd(bar + 1, 1u);
        } else {
            while (*gen == g) {}
        }
        __threadfence();
    }
    __syncthreads();
}

// True in exactly one block: the last of `nb` to arrive at `cnt` (which it resets). Everything
// the other blocks wrote before arriving is visible to it.
__device__ bool last_block(unsigned int* cnt, unsigned int nb) {
    __shared__ unsigned int is_last;
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        unsigned int prev = atomicAdd(cnt, 1u);
        is_last = prev == nb - 1;
        if (is_last) atomicExch(cnt, 0u);
    }
    __syncthreads();
    if (is_last) __threadfence();
    return is_last;
}

// The whole hyper-connection read in one launch on a co-resident grid (512 threads a block):
// phase 0 the pending write and per-stream norm (fl_hc_norm's arithmetic), barrier, phase 1 the
// down / inject rows (two rows per block step, 16 warps splitting K), barrier, phase 2 the up
// projection and mean (fl_hc_up's arithmetic: a warp per output, a block per 32-wide
// quantization chunk). XN and LO are the hc scratch.
template <int T>
__device__ __forceinline__ void hc_fused_body(
    float* __restrict__ R, const float* __restrict__ Yp, const float* __restrict__ Ip, unsigned int mode,
    const float* __restrict__ Wn, float eps, const float* __restrict__ PARTS, const float* __restrict__ Wr,
    const float* __restrict__ L, unsigned int stride, int sg, const unsigned short* __restrict__ Wd,
    const unsigned short* __restrict__ Wi, unsigned int rows, const unsigned short* __restrict__ Wu,
    float* __restrict__ X, unsigned int* __restrict__ XQ, unsigned int quant, float* __restrict__ INJ,
    float* __restrict__ XN, float* __restrict__ LO, unsigned int* __restrict__ bar) {
    __shared__ float red[16][2 * T];
    __shared__ __align__(16) float lo[T * HC_LR];
    __shared__ float xs[T][32];
    unsigned int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nb = gridDim.x;
    // Phase 0: per (stream, token) row.
    for (unsigned int item = blockIdx.x; item < HC * T; item += nb) {
        unsigned int c = item % HC, t = item / HC;
        float* row = R + ((u64)t * HC + c) * HIDDEN;
        float g = mode ? 2.0f / (1.0f + expf(-(Ip[t * HC + c] * 0.25f))) : 0.0f;
        float gs = (mode == 2 && sg >= 0) ? sigm(L[(u64)t * stride + sg]) : 0.0f;
        float v[5], ss = 0.0f;
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            unsigned int d = tid + 512 * i;
            float x = row[d];
            if (mode) {
                float y;
                if (mode == 1) {
                    y = Yp[(u64)t * HIDDEN + d];
                } else {
                    y = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < TOPK; k++)
                        y = __fmaf_rn(Wr[t * TOPK + k], PARTS[(u64)(t * TOPK + k) * HIDDEN + d], y);
                    if (sg >= 0) y = __fmaf_rn(gs, PARTS[(u64)(SHARED_ROW + t) * HIDDEN + d], y);
                }
                x = __fmaf_rn(y, g, x);
                row[d] = x;
            }
            v[i] = x;
            ss += x * x;
        }
        ss = warp_sum(ss);
        if (lane == 0) red[warp][0] = ss;
        __syncthreads();
        ss = 0.0f;
        for (int w = 0; w < 16; w++) ss += red[w][0];
        float rs = 1.0f / sqrtf(ss / (float)HIDDEN + eps);
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            unsigned int d = tid + 512 * i;
            XN[(u64)t * HC * HIDDEN + c * HIDDEN + d] = v[i] * rs * Wn[c * HIDDEN + d];
        }
        __syncthreads();
    }
    grid_bar(bar, nb);
    // Phase 1: rows (r, r + 1) per step; warp w takes K slice [640 w, 640 w + 640).
    for (unsigned int r0 = 2 * blockIdx.x; r0 < rows; r0 += 2 * nb) {
        const unsigned short* w[2];
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            unsigned int row = min(r0 + r, rows - 1);
            w[r] = (row < HC_LR ? Wd + (u64)row * HC * HIDDEN : Wi + (u64)(row - HC_LR) * HC * HIDDEN) + warp * 640;
        }
        float acc[2][T];
        #pragma unroll
        for (int r = 0; r < 2; r++)
            #pragma unroll
            for (int t = 0; t < T; t++) acc[r][t] = 0.0f;
        for (unsigned int i = lane; i < 80; i += 32) {
            uint4 p0 = ((const uint4*)w[0])[i], p1 = ((const uint4*)w[1])[i];
            #pragma unroll
            for (int t = 0; t < T; t++) {
                const float4* x4 = (const float4*)(XN + (u64)t * HC * HIDDEN + warp * 640 + i * 8);
                float4 a = x4[0], b = x4[1];
                acc[0][t] += bf(p0.x & 0xffff) * a.x + bf(p0.x >> 16) * a.y + bf(p0.y & 0xffff) * a.z
                           + bf(p0.y >> 16) * a.w + bf(p0.z & 0xffff) * b.x + bf(p0.z >> 16) * b.y
                           + bf(p0.w & 0xffff) * b.z + bf(p0.w >> 16) * b.w;
                acc[1][t] += bf(p1.x & 0xffff) * a.x + bf(p1.x >> 16) * a.y + bf(p1.y & 0xffff) * a.z
                           + bf(p1.y >> 16) * a.w + bf(p1.z & 0xffff) * b.x + bf(p1.z >> 16) * b.y
                           + bf(p1.w & 0xffff) * b.z + bf(p1.w >> 16) * b.w;
            }
        }
        #pragma unroll
        for (int r = 0; r < 2; r++)
            #pragma unroll
            for (int t = 0; t < T; t++) {
                float s = warp_sum(acc[r][t]);
                if (lane == 0) red[warp][r * T + t] = s;
            }
        __syncthreads();
        if (tid < 2 * T) {
            unsigned int r = tid / T, t = tid % T, row = r0 + r;
            if (row < rows) {
                float v = 0.0f;
                for (int k = 0; k < 16; k++) v += red[k][r * T + t];
                if (row < HC_LR) LO[t * HC_LR + row] = silu(v * 0.25f);
                else INJ[t * HC + row - HC_LR] = v;
            }
        }
        __syncthreads();
    }
    grid_bar(bar, nb);
    // Phase 2: 32 outputs per block step, two per warp.
    if (blockIdx.x >= HIDDEN / 32) return;
    for (unsigned int i = tid; i < T * HC_LR; i += 512) lo[i] = LO[i];
    __syncthreads();
    for (unsigned int q = blockIdx.x; q < HIDDEN / 32; q += nb) {
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            unsigned int dl = warp + 16 * half, d = q * 32 + dl;
            const uint4* w = (const uint4*)(Wu + (u64)d * HC * HC_LR);
            uint4 qv[5];
            #pragma unroll
            for (int j = 0; j < 5; j++) qv[j] = w[lane + 32 * j];
            float acc[T];
            #pragma unroll
            for (int t = 0; t < T; t++) acc[t] = 0.0f;
            #pragma unroll
            for (int j = 0; j < 5; j++) {
                unsigned int p = (lane >> 2) + 8 * j;
                float wv[8] = {bf(qv[j].x & 0xffff), bf(qv[j].x >> 16), bf(qv[j].y & 0xffff), bf(qv[j].y >> 16),
                               bf(qv[j].z & 0xffff), bf(qv[j].z >> 16), bf(qv[j].w & 0xffff), bf(qv[j].w >> 16)};
                #pragma unroll
                for (int t = 0; t < T; t++) {
                    const float4* l4 = (const float4*)(lo + t * HC_LR + p * 8);
                    float4 a = l4[0], b = l4[1];
                    acc[t] += wv[0] * a.x + wv[1] * a.y + wv[2] * a.z + wv[3] * a.w
                            + wv[4] * b.x + wv[5] * b.y + wv[6] * b.z + wv[7] * b.w;
                }
            }
            #pragma unroll
            for (int t = 0; t < T; t++) {
                float v = acc[t];
                v += __shfl_xor_sync(0xffffffffu, v, 4);
                v += __shfl_xor_sync(0xffffffffu, v, 8);
                v += __shfl_xor_sync(0xffffffffu, v, 16);
                float g = sigm(v);
                float g1 = __shfl_sync(0xffffffffu, g, 1), g2 = __shfl_sync(0xffffffffu, g, 2), g3 = __shfl_sync(0xffffffffu, g, 3);
                if (lane == 0) {
                    const float* xn = XN + (u64)t * HC * HIDDEN + d;
                    float x = __fmaf_rn(xn[0], g, 0.0f);
                    x = __fmaf_rn(xn[HIDDEN], g1, x);
                    x = __fmaf_rn(xn[2 * HIDDEN], g2, x);
                    x = __fmaf_rn(xn[3 * HIDDEN], g3, x);
                    x *= 0.25f;
                    X[(u64)t * HIDDEN + d] = x;
                    xs[t][dl] = x;
                }
            }
        }
        __syncthreads();
        if (quant && warp < T) quant_chunk(xs[warp][lane], XQ, T, HIDDEN, warp, q, lane);
        __syncthreads();
    }
}
#define HC_FUSED(T) \
extern "C" __global__ void __launch_bounds__(512, 2) fl_hc_fused_t##T( \
    float* R, const float* Yp, const float* Ip, unsigned int mode, const float* Wn, float eps, \
    const float* PARTS, const float* Wr, const float* L, unsigned int stride, int sg, \
    const unsigned short* Wd, const unsigned short* Wi, unsigned int rows, const unsigned short* Wu, \
    float* X, unsigned int* XQ, unsigned int quant, float* INJ, float* XN, float* LO, unsigned int* bar) { \
    hc_fused_body<T>(R, Yp, Ip, mode, Wn, eps, PARTS, Wr, L, stride, sg, Wd, Wi, rows, Wu, X, XQ, quant, \
                     INJ, XN, LO, bar); \
}
HC_FUSED(1) HC_FUSED(2) HC_FUSED(3) HC_FUSED(4) HC_FUSED(5) HC_FUSED(6) HC_FUSED(7) HC_FUSED(8)

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
    const unsigned int* __restrict__ win, unsigned int commit, float eps,
    unsigned int* __restrict__ YQ, unsigned int quant) {
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
            float y = o * rs * NW[j] * sigm(sz[t][j]);
            Y[(u64)t * GDN_V + hv * 128 + j] = y;
            if (quant) quant_chunk(y, YQ, T, GDN_V, t, hv * 4 + (j >> 5), j & 31);
        }
        __syncthreads();
    }
    if (commit) {
        #pragma unroll
        for (int r = 0; r < 32; r++) S[((u64)(rg * 32 + r) * GDN_HV + hv) * 128 + j] = st[r];
    }
}

// ---- MoE ----

// One warp, one token: top-10 by (logit desc, index asc) of `l`, weights e_k / Σ_top e_j with
// e_k = exp(l_k − l_max) in f64 summed in rank order. Lane k < 10 returns rank k.
__device__ __forceinline__ void router_warp(const float* __restrict__ l, unsigned int NE, unsigned int lane,
                                            unsigned int* id_out, float* w_out) {
    float v[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        unsigned int e = lane + 32 * i;
        v[i] = e < NE ? l[e] : NEG_INF;
    }
    unsigned int used = 0, my_id = 0;
    float my_v = 0.0f, top = 0.0f;
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
        if (k == 0) top = bv;
        if (lane == k) { my_id = bi; my_v = bv; }
    }
    double e = lane < TOPK ? exp((double)my_v - (double)top) : 0.0;
    double sum = 0.0;
    for (int k = 0; k < TOPK; k++) sum += __shfl_sync(0xffffffffu, e, k);
    *id_out = my_id;
    *w_out = (float)(e / sum);
}

// Grid T, block 32 (a warp per token).
extern "C" __global__ void fl_router_topk(const float* __restrict__ L, unsigned int stride,
                                          unsigned int NE, unsigned int* __restrict__ IDS,
                                          float* __restrict__ W) {
    unsigned int t = blockIdx.x, lane = threadIdx.x, id;
    float w;
    router_warp(L + (u64)t * stride, NE, lane, &id, &w);
    if (lane < TOPK) {
        IDS[t * TOPK + lane] = id;
        W[t * TOPK + lane] = w;
    }
}

// Inclusive scan of one value per thread over the first 128 threads of the block (every thread
// of the block must call it).
__device__ unsigned int scan128(unsigned int v, unsigned int* sc) {
    unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int o = 1; o < 32; o <<= 1) {
        unsigned int n = __shfl_up_sync(0xffffffffu, v, o);
        if (lane >= o) v += n;
    }
    if (lane == 31 && warp < 4) sc[warp] = v;
    __syncthreads();
    unsigned int off = 0;
    for (unsigned int w = 0; w < warp && w < 4; w++) off += sc[w];
    __syncthreads();
    return v + off;
}

// The [MoePlan] from router ids `ids` (shared memory, T·TOPK of them, visible to the block) and
// a residency table (EXPERTS addresses, word pairs). Needs at least 128 threads; every step is
// parallel over the routed entries (two scans), no serial loop.
__device__ void plan_core(const unsigned int* ids, const unsigned int* __restrict__ TABLE, u64 shared,
                          unsigned int* __restrict__ PLAN, unsigned int T) {
    __shared__ unsigned int first[PLAN_CAP], gid[PLAN_CAP], start[PLAN_CAP + 1];
    __shared__ unsigned int sc[4];
    __shared__ unsigned int tot_g, tot_e, tot_m;
    unsigned int n = T * TOPK, tid = threadIdx.x;
    u64 p = 0;
    unsigned int me = 0, f = tid;
    if (tid < n) {
        me = ids[tid];
        p = (u64)TABLE[2 * me] | ((u64)TABLE[2 * me + 1] << 32);
        for (unsigned int j = 0; j < tid; j++) if (ids[j] == me) { f = j; break; }
        first[tid] = f;
    }
    bool lead = tid < n && f == tid;
    bool res = lead && p != 0, miss = lead && p == 0;
    unsigned int cnt = 0;
    if (res) for (unsigned int j = tid; j < n; j++) cnt += ids[j] == me;
    // Group index of a resident leader, its entries' start, and the missing-list index.
    unsigned int g_incl = scan128(res ? 1u : 0u, sc);
    unsigned int s_incl = scan128(cnt, sc);
    unsigned int m_incl = scan128(miss ? 1u : 0u, sc);
    if (tid == 127) { tot_g = g_incl; tot_e = s_incl; tot_m = m_incl; }
    if (res) {
        unsigned int g = g_incl - 1, st = s_incl - cnt;
        gid[tid] = g;
        start[g] = st;
        PLAN[PLAN_GP + 2 * g] = (unsigned int)p;
        PLAN[PLAN_GP + 2 * g + 1] = (unsigned int)(p >> 32);
        PLAN[PLAN_GS + g] = st;
    } else if (lead) {
        gid[tid] = 0xffffffffu;
    }
    if (miss) PLAN[PLAN_MISS + m_incl - 1] = me;
    __syncthreads();
    unsigned int ng = tot_g, s = tot_e;
    if (tid < n) {
        unsigned int g = gid[first[tid]];
        if (g != 0xffffffffu) {
            unsigned int rank = 0;
            for (unsigned int j = 0; j < tid; j++) rank += ids[j] == me;
            unsigned int e = start[g] + rank;
            PLAN[PLAN_ET + e] = tid / TOPK;
            PLAN[PLAN_ED + e] = tid;
        }
    }
    if (shared != 0) {
        if (tid < T) {
            PLAN[PLAN_ET + s + tid] = tid;
            PLAN[PLAN_ED + s + tid] = SHARED_ROW + tid;
        }
        if (tid == 0) {
            PLAN[PLAN_GP + 2 * ng] = (unsigned int)shared;
            PLAN[PLAN_GP + 2 * ng + 1] = (unsigned int)(shared >> 32);
            PLAN[PLAN_GS + ng] = s;
        }
        ng += 1;
        s += T;
    }
    if (tid == 0) {
        PLAN[PLAN_GS + ng] = s;
        PLAN[0] = ng; PLAN[1] = s; PLAN[2] = tot_m; PLAN[3] = 0;
    }
}

// Grid 1, block 128.
extern "C" __global__ void fl_moe_plan(const unsigned int* __restrict__ IDS, const unsigned int* __restrict__ TABLE,
                                       u64 shared, unsigned int* __restrict__ PLAN, unsigned int T) {
    __shared__ unsigned int ids[PLAN_CAP];
    if (threadIdx.x < T * TOPK) ids[threadIdx.x] = IDS[threadIdx.x];
    __syncthreads();
    plan_core(ids, TABLE, shared, PLAN, T);
}

// Router and plan in one launch: warp t routes token t (ids and weights to IDS / W), then the
// block builds the plan from those ids, or from FORCED when `forced` (a recorded routing, for
// replays and benchmarks). Grid 1, block 256.
extern "C" __global__ void fl_moe_route(const float* __restrict__ L, unsigned int stride, unsigned int NE,
                                        const unsigned int* __restrict__ FORCED, unsigned int forced,
                                        const unsigned int* __restrict__ TABLE, u64 shared,
                                        unsigned int* __restrict__ IDS, float* __restrict__ W,
                                        unsigned int* __restrict__ PLAN, unsigned int T) {
    __shared__ unsigned int ids[PLAN_CAP];
    unsigned int lane = threadIdx.x & 31, t = threadIdx.x >> 5;
    if (t < T) {
        unsigned int id;
        float w;
        router_warp(L + (u64)t * stride, NE, lane, &id, &w);
        if (lane < TOPK) {
            IDS[t * TOPK + lane] = id;
            W[t * TOPK + lane] = w;
            if (!forced) ids[t * TOPK + lane] = id;
        }
    }
    if (forced && threadIdx.x < T * TOPK) ids[threadIdx.x] = FORCED[threadIdx.x];
    __syncthreads();
    plan_core(ids, TABLE, shared, PLAN, T);
}

__device__ __forceinline__ const unsigned char* plan_blob(const unsigned int* PLAN, unsigned int g) {
    return (const unsigned char*)((u64)PLAN[PLAN_GP + 2 * g] | ((u64)PLAN[PLAN_GP + 2 * g + 1] << 32));
}

// exp with the pinned operation sequence of flash::pexp (bitwise the CPU's).
__device__ __forceinline__ float pexpf(float x) {
    x = fminf(fmaxf(x, -87.0f), 88.0f);
    float n = rintf(x * __int_as_float(0x3fb8aa3b));
    float r = __fmaf_rn(-n, __int_as_float(0x35bfbe8e), __fmaf_rn(-n, __int_as_float(0x3f317200), x));
    float p = __int_as_float(0x39500d01);
    p = __fmaf_rn(p, r, __int_as_float(0x3ab60b61));
    p = __fmaf_rn(p, r, __int_as_float(0x3c088889));
    p = __fmaf_rn(p, r, __int_as_float(0x3d2aaaab));
    p = __fmaf_rn(p, r, __int_as_float(0x3e2aaaab));
    p = __fmaf_rn(p, r, 0.5f);
    p = __fmaf_rn(p, r, 1.0f);
    p = __fmaf_rn(p, r, 1.0f);
    return p * __int_as_float(((int)n + 127) << 23);
}
__device__ __forceinline__ float psilu(float x) { return x / (1.0f + pexpf(-x)); }

// Σ code · q over half a chunk: 4 code bytes (16 weights) against one [QAct] uint4 (the half's 4
// activation words), field f of the code bytes pairing with word f.
__device__ __forceinline__ int half_dot(unsigned int w, uint4 x) {
    int s = __dp4a((int)(w & 0x03030303u), (int)x.x, 0);
    s = __dp4a((int)((w >> 2) & 0x03030303u), (int)x.y, s);
    s = __dp4a((int)((w >> 4) & 0x03030303u), (int)x.z, s);
    s = __dp4a((int)((w >> 6) & 0x03030303u), (int)x.w, s);
    return s;
}

// One 16-row expert tile against up to NE entries, in the pinned order of ExpertBlob::ORDER:
// lane = (r = lane / 4, quarter b = lane % 4) owns slots 2b, 2b + 1 (one chunk, both halves) of
// rows r and r + 8, and runs their chains over the rows' 128-weight groups: one 8-byte load per
// row and group (the warp reads two 256-byte lines), each activation load serving both rows.
// The tree: slot m + slot m + 4 is a shuffle with lane b ^ 2, then (l0 + l2) + (l1 + l3) with
// lane b ^ 1. On return, lanes with b = 0 hold rows r (o0) and r + 8 (o1). Entry e reads row
// row[e] of the [QAct] at X (m_rows rows, kb words wide).
template <int G, int NE>
__device__ __forceinline__ void tile_dot(const unsigned char* __restrict__ codes, const unsigned char* __restrict__ scales,
                                         const unsigned int* __restrict__ X, unsigned int kb, unsigned int m_rows,
                                         const unsigned int row[NE], unsigned int ne, unsigned int lane,
                                         float o0[NE], float o1[NE]) {
    const unsigned int nch = kb / 8;
    unsigned int r = lane >> 2, b = lane & 3;
    const unsigned int* xp[NE];
    const unsigned int* sp[NE];
    #pragma unroll
    for (int e = 0; e < NE; e++) {
        xp[e] = X + (u64)row[e] * kb + b * 8;
        sp[e] = X + (u64)m_rows * kb + (u64)row[e] * nch + b;
    }
    const unsigned int hoff = m_rows * nch;
    float a[NE][2][2];
    #pragma unroll
    for (int e = 0; e < NE; e++)
        #pragma unroll
        for (int q = 0; q < 2; q++) { a[e][q][0] = 0.0f; a[e][q][1] = 0.0f; }
    constexpr int B = G % 2 == 0 ? 2 : 5;
    for (int g0 = 0; g0 < G; g0 += B) {
        uint2 w[B][2];
        unsigned short ds[B][2];
        #pragma unroll
        for (int q = 0; q < B; q++)
            #pragma unroll
            for (int rr = 0; rr < 2; rr++) {
                w[q][rr] = *(const uint2*)(codes + (g0 + q) * 512 + (r + 8 * rr) * 32 + b * 8);
                ds[q][rr] = *(const unsigned short*)(scales + (g0 + q) * 64 + (r + 8 * rr) * 4 + (b >> 1) * 2);
            }
        #pragma unroll
        for (int q = 0; q < B; q++) {
            int op[2][2][4];
            float d[2];
            #pragma unroll
            for (int rr = 0; rr < 2; rr++) {
                d[rr] = h2f(ds[q][rr]);
                #pragma unroll
                for (int f = 0; f < 4; f++) {
                    op[rr][0][f] = (int)((w[q][rr].x >> (2 * f)) & 0x03030303u);
                    op[rr][1][f] = (int)((w[q][rr].y >> (2 * f)) & 0x03030303u);
                }
            }
            unsigned int c4 = 4 * (g0 + q);
            #pragma unroll
            for (int e = 0; e < NE; e++) {
                if (e < ne) {
                    const uint4* xq = (const uint4*)(xp[e] + c4 * 8);
                    uint4 x0 = xq[0], x1 = xq[1];
                    float dx = __uint_as_float(sp[e][c4]);
                    int nh = -(int)sp[e][hoff + c4];
                    #pragma unroll
                    for (int rr = 0; rr < 2; rr++) {
                        int s0 = __dp4a(op[rr][0][0], (int)x0.x, nh);
                        s0 = __dp4a(op[rr][0][1], (int)x0.y, s0);
                        s0 = __dp4a(op[rr][0][2], (int)x0.z, s0);
                        s0 = __dp4a(op[rr][0][3], (int)x0.w, s0);
                        int s1 = __dp4a(op[rr][1][0], (int)x1.x, 0);
                        s1 = __dp4a(op[rr][1][1], (int)x1.y, s1);
                        s1 = __dp4a(op[rr][1][2], (int)x1.z, s1);
                        s1 = __dp4a(op[rr][1][3], (int)x1.w, s1);
                        float dd = d[rr] * dx;
                        a[e][rr][0] = __fmaf_rn((float)s0, dd, a[e][rr][0]);
                        a[e][rr][1] = __fmaf_rn((float)s1, dd, a[e][rr][1]);
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int e = 0; e < NE; e++) {
        if (e < ne) {
            #pragma unroll
            for (int rr = 0; rr < 2; rr++) {
                float l0 = a[e][rr][0] + __shfl_xor_sync(0xffffffffu, a[e][rr][0], 2);
                float l1 = a[e][rr][1] + __shfl_xor_sync(0xffffffffu, a[e][rr][1], 2);
                float x = l0 + __shfl_xor_sync(0xffffffffu, l0, 1);
                float y = l1 + __shfl_xor_sync(0xffffffffu, l1, 1);
                if (rr == 0) o0[e] = x + y; else o1[e] = x + y;
            }
        }
    }
}

// Gate and up rows of every planned expert for its tokens, h = psilu(gate)·up quantized per
// [QAct] into HQ (m = PLAN_CAP rows of FF, row = entry). Work items are (32-row h chunk, group),
// 20 per group, one per 128-thread block in turn: each warp takes one 16-row gu tile (8 h rows),
// the block gathers the chunk's h in shared memory and quantizes it.
template <int NE>
__device__ __forceinline__ void moe_gu_body(const unsigned int* __restrict__ XQ, unsigned int T,
                                            const unsigned int* __restrict__ PLAN, unsigned int* __restrict__ HQ) {
    __shared__ float hs[NE][32];
    unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const unsigned int kb = HIDDEN / 4;
    unsigned int items = (FF / 32) * PLAN[0];
    for (unsigned int item = blockIdx.x; item < items; item += gridDim.x) {
        unsigned int q = item % (FF / 32), g = item / (FF / 32), tile = 4 * q + warp;
        const unsigned char* blob = plan_blob(PLAN, g);
        unsigned int e0 = PLAN[PLAN_GS + g], ne = PLAN[PLAN_GS + g + 1] - e0;
        unsigned int tok[NE];
        #pragma unroll
        for (int e = 0; e < NE; e++) tok[e] = e < ne ? PLAN[PLAN_ET + e0 + e] : 0;
        float a0[NE], a1[NE];
        tile_dot<HIDDEN / 128, NE>(blob + GU_CODES + (u64)tile * (HIDDEN / 128) * 512,
                                   blob + GU_SCALES + (u64)tile * (HIDDEN / 128) * 64, XQ, kb, T, tok, ne, lane, a0, a1);
        __syncthreads();
        #pragma unroll
        for (int e = 0; e < NE; e++) {
            if (e < ne) {
                // Lane 4s holds gu rows s (a0) and s + 8 (a1): gate (s even) or up (s odd) of h
                // rows 8·tile + s / 2 and 8·tile + 4 + s / 2.
                float u0 = __shfl_down_sync(0xffffffffu, a0[e], 4);
                float u1 = __shfl_down_sync(0xffffffffu, a1[e], 4);
                if ((lane & 7) == 0) {
                    hs[e][8 * warp + (lane >> 3)] = psilu(a0[e]) * u0;
                    hs[e][8 * warp + 4 + (lane >> 3)] = psilu(a1[e]) * u1;
                }
            }
        }
        __syncthreads();
        for (unsigned int e = warp; e < ne; e += 4) quant_chunk(hs[e][lane], HQ, PLAN_CAP, FF, e0 + e, q, lane);
    }
}

#define MOE_GU(T) \
extern "C" __global__ void __launch_bounds__(128) fl_moe_gu_t##T( \
    const unsigned int* __restrict__ XQ, unsigned int Tm, const unsigned int* __restrict__ PLAN, \
    unsigned int* __restrict__ HQ) { \
    moe_gu_body<T>(XQ, Tm, PLAN, HQ); \
}
MOE_GU(1) MOE_GU(2) MOE_GU(3) MOE_GU(4) MOE_GU(5) MOE_GU(6) MOE_GU(7) MOE_GU(8)

// Down rows of every planned expert against its entries' quantized h (HQ), into PARTS[dst].
// Work items are (16-row down tile, group), 160 per group, one per warp in turn.
template <int NE>
__device__ __forceinline__ void moe_down_body(const unsigned int* __restrict__ HQ, const unsigned int* __restrict__ PLAN,
                                              float* __restrict__ PARTS) {
    unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const unsigned int kb = FF / 4;
    unsigned int items = (HIDDEN / 16) * PLAN[0];
    for (unsigned int item = blockIdx.x * 8 + warp; item < items; item += gridDim.x * 8) {
        unsigned int tile = item % (HIDDEN / 16), g = item / (HIDDEN / 16);
        const unsigned char* blob = plan_blob(PLAN, g);
        unsigned int e0 = PLAN[PLAN_GS + g], ne = PLAN[PLAN_GS + g + 1] - e0;
        unsigned int ents[NE];
        #pragma unroll
        for (int e = 0; e < NE; e++) ents[e] = e0 + e;
        float a0[NE], a1[NE];
        tile_dot<FF / 128, NE>(blob + DOWN_CODES + (u64)tile * (FF / 128) * 512,
                               blob + DOWN_SCALES + (u64)tile * (FF / 128) * 64, HQ, kb, PLAN_CAP, ents, ne, lane,
                               a0, a1);
        unsigned int my_dst = lane < ne ? PLAN[PLAN_ED + e0 + lane] : 0;
        #pragma unroll
        for (int e = 0; e < NE; e++) {
            unsigned int dst = __shfl_sync(0xffffffffu, my_dst, e);
            if (e < ne && (lane & 3) == 0) {
                float* out = PARTS + (u64)dst * HIDDEN + 16 * tile + (lane >> 2);
                out[0] = a0[e];
                out[8] = a1[e];
            }
        }
    }
}

#define MOE_DOWN(T) \
extern "C" __global__ void __launch_bounds__(256, 3) fl_moe_down_t##T( \
    const unsigned int* __restrict__ HQ, const unsigned int* __restrict__ PLAN, float* __restrict__ PARTS) { \
    moe_down_body<T>(HQ, PLAN, PARTS); \
}
MOE_DOWN(1) MOE_DOWN(2) MOE_DOWN(3) MOE_DOWN(4) MOE_DOWN(5) MOE_DOWN(6) MOE_DOWN(7) MOE_DOWN(8)

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
    if (lane == 0) SC[(u64)t * MAXB + b] = total * INV_SQRT_D;
}

__device__ __forceinline__ unsigned int ordered(float x) {
    unsigned int b = __float_as_uint(x);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// Inclusive block scan of one value per thread (1024 threads): warp scans, then a scan of the
// 32 warp totals.
__device__ unsigned int block_scan(unsigned int v, unsigned int* sc) {
    unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int o = 1; o < 32; o <<= 1) {
        unsigned int n = __shfl_up_sync(0xffffffffu, v, o);
        if (lane >= o) v += n;
    }
    if (lane == 31) sc[warp] = v;
    __syncthreads();
    if (warp == 0) {
        unsigned int w = sc[lane];
        for (int o = 1; o < 32; o <<= 1) {
            unsigned int n = __shfl_up_sync(0xffffffffu, w, o);
            if (lane >= o) w += n;
        }
        sc[lane] = w;
    }
    __syncthreads();
    unsigned int r = v + (warp > 0 ? sc[warp - 1] : 0);
    __syncthreads();
    return r;
}

#define SEL_MAXB 8192
#define SEL_BLOCKS 512
// Cells token t selects: every cell up to 512 complete blocks, else 512 blocks plus the tail.
__device__ __forceinline__ unsigned int qsa_n_sel(unsigned int n_kv) {
    return n_kv / 4 <= SEL_BLOCKS ? n_kv : 4 * SEL_BLOCKS + n_kv % 4;
}
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
    unsigned int n_bid = n_kv / 4, tail = n_kv % 4;
    if (n_bid <= SEL_BLOCKS) {
        for (unsigned int i = tid; i < n_kv; i += 1024) out[i] = i;
        return;
    }
    // The top 512 complete blocks by (score desc, block asc), then the tail's cells.
    const unsigned int need = SEL_BLOCKS, rem = 0;
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
        if (tid < 32) {
            // Lane l owns digits 255 - 8l .. 248 - 8l: a warp scan finds the digit holding the
            // k-th largest key.
            unsigned int loc[8], sum = 0;
            #pragma unroll
            for (int i = 0; i < 8; i++) { loc[i] = hist[255 - 8 * tid - i]; sum += loc[i]; }
            unsigned int incl = sum;
            for (int o = 1; o < 32; o <<= 1) {
                unsigned int n = __shfl_up_sync(0xffffffffu, incl, o);
                if (tid >= o) incl += n;
            }
            unsigned int acc = incl - sum;
            if (acc < k && incl >= k) {
                for (int i = 0; i < 8; i++) {
                    if (acc + loc[i] >= k) { sh_digit = 255 - 8 * tid - i; sh_k = k - acc; break; }
                    acc += loc[i];
                }
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
        for (unsigned int i = 0; i < tail; i++) out[4 * SEL_BLOCKS + i] = n_bid * 4 + i;
}

// Split-K attention: grid (QSA_NCH, QSA_KV, T), block 256. A block reads its chunk of
// QSA_CHUNK selected cells' K and V once for the 12 q heads that share the kv head: scores a
// thread per (cell, 3 heads), each reading its cell's whole key row; softmax a warp per head;
// p·V a thread per (dim pair, half of the cells). Writes per head [m, l, acc[256]] partials.
extern "C" __global__ void __launch_bounds__(256) fl_qsa_attend(
    const float* __restrict__ Q, const unsigned short* __restrict__ KC, const unsigned short* __restrict__ VC,
    const unsigned int* __restrict__ IDS, const unsigned int* __restrict__ win, float* __restrict__ PART) {
    __shared__ __align__(16) float qs[12][256];
    __shared__ float s[12][QSA_CHUNK];
    __shared__ unsigned int cells[QSA_CHUNK];
    __shared__ __align__(8) float half[12][256];
    unsigned int ch = blockIdx.x, g = blockIdx.y, t = blockIdx.z, tid = threadIdx.x;
    unsigned int lane = tid & 31, warp = tid >> 5;
    unsigned int n_sel = qsa_n_sel(win[0] + t + 1);
    unsigned int i0 = ch * QSA_CHUNK;
    if (i0 >= n_sel) return;
    unsigned int n = min((unsigned int)QSA_CHUNK, n_sel - i0);
    for (unsigned int i = tid; i < 12 * 256 / 4; i += 256)
        ((float4*)&qs[0][0])[i] = ((const float4*)(Q + ((u64)t * QSA_HEADS + g * 12) * QSA_D))[i];
    if (tid < n) cells[tid] = IDS[(u64)t * QSA_WIDTH + i0 + tid];
    __syncthreads();
    {
        unsigned int c = tid % QSA_CHUNK, hg = tid / QSA_CHUNK;  // 4 groups of 3 heads
        if (c < n) {
            const uint4* kr = (const uint4*)(KC + ((u64)cells[c] * QSA_KV + g) * QSA_D);
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
            #pragma unroll 4
            for (int v = 0; v < 32; v++) {
                uint4 kq = kr[v];
                float kv[8] = {bf(kq.x & 0xffff), bf(kq.x >> 16), bf(kq.y & 0xffff), bf(kq.y >> 16),
                               bf(kq.z & 0xffff), bf(kq.z >> 16), bf(kq.w & 0xffff), bf(kq.w >> 16)};
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    a0 = __fmaf_rn(kv[e], qs[3 * hg][8 * v + e], a0);
                    a1 = __fmaf_rn(kv[e], qs[3 * hg + 1][8 * v + e], a1);
                    a2 = __fmaf_rn(kv[e], qs[3 * hg + 2][8 * v + e], a2);
                }
            }
            s[3 * hg][c] = a0 * 0.0625f;
            s[3 * hg + 1][c] = a1 * 0.0625f;
            s[3 * hg + 2][c] = a2 * 0.0625f;
        }
    }
    __syncthreads();
    for (unsigned int h = warp; h < 12; h += 8) {
        float a = NEG_INF, b = NEG_INF;
        if (lane < n) a = s[h][lane];
        if (QSA_CHUNK > 32 && lane + 32 < n) b = s[h][lane + 32];
        float m = warp_max(fmaxf(a, b));
        float pa = lane < n ? expf(a - m) : 0.0f, pb = (QSA_CHUNK > 32 && lane + 32 < n) ? expf(b - m) : 0.0f;
        if (lane < n) s[h][lane] = pa;
        if (QSA_CHUNK > 32 && lane + 32 < n) s[h][lane + 32] = pb;
        float l = warp_sum(pa + pb);
        if (lane == 0) {
            float* part = PART + (((u64)t * QSA_HEADS + g * 12 + h) * QSA_NCH + ch) * (QSA_D + 2);
            part[0] = m;
            part[1] = l;
        }
    }
    __syncthreads();
    // p·V: thread (dims 2j, 2j+1; cells of parity `side`), then the two halves added.
    unsigned int j = tid & 127, side = tid >> 7;
    float acc0[12], acc1[12];
    #pragma unroll
    for (int h = 0; h < 12; h++) { acc0[h] = 0.0f; acc1[h] = 0.0f; }
    for (unsigned int i = side; i < n; i += 2) {
        unsigned int vv = ((const unsigned int*)(VC + ((u64)cells[i] * QSA_KV + g) * QSA_D))[j];
        float v0 = bf(vv & 0xffff), v1 = bf(vv >> 16);
        #pragma unroll
        for (int h = 0; h < 12; h++) {
            acc0[h] = __fmaf_rn(s[h][i], v0, acc0[h]);
            acc1[h] = __fmaf_rn(s[h][i], v1, acc1[h]);
        }
    }
    if (side == 1) {
        #pragma unroll
        for (int h = 0; h < 12; h++) { half[h][2 * j] = acc0[h]; half[h][2 * j + 1] = acc1[h]; }
    }
    __syncthreads();
    if (side == 0) {
        #pragma unroll
        for (int h = 0; h < 12; h++) {
            float* part = PART + (((u64)t * QSA_HEADS + g * 12 + h) * QSA_NCH + ch) * (QSA_D + 2) + 2;
            part[2 * j] = acc0[h] + half[h][2 * j];
            part[2 * j + 1] = acc1[h] + half[h][2 * j + 1];
        }
    }
}

// Merge the chunks and apply the sigmoid gate; with `OQ`, also the gated output as int8
// activations ([QAct], m = T rows of QSA_HEADS · QSA_D). Grid (QSA_HEADS, T), block 256.
extern "C" __global__ void fl_qsa_merge(const float* __restrict__ PART, const float* __restrict__ P,
                                        unsigned int stride, const unsigned int* __restrict__ win,
                                        float* __restrict__ OUT, unsigned int* __restrict__ OQ,
                                        unsigned int quant, unsigned int T) {
    __shared__ float f[QSA_NCH];
    __shared__ float sL;
    unsigned int h = blockIdx.x, t = blockIdx.y, j = threadIdx.x;
    unsigned int n_sel = qsa_n_sel(win[0] + t + 1);
    unsigned int nch = (n_sel + QSA_CHUNK - 1) / QSA_CHUNK;
    const float* part = PART + ((u64)t * QSA_HEADS + h) * QSA_NCH * (QSA_D + 2);
    if (j < 32) {
        // Chunk weights exp(m_c − M) and L = Σ l_c · w_c, by one warp.
        float M = NEG_INF;
        for (unsigned int c = j; c < nch; c += 32) M = fmaxf(M, part[c * (QSA_D + 2)]);
        M = warp_max(M);
        float L = 0.0f;
        for (unsigned int c = j; c < nch; c += 32) {
            float w = expf(part[c * (QSA_D + 2)] - M);
            f[c] = w;
            L += part[c * (QSA_D + 2) + 1] * w;
        }
        L = warp_sum(L);
        if (j == 0) sL = L;
    }
    __syncthreads();
    float o = 0.0f, L = sL;
    for (unsigned int c = 0; c < nch; c++) o = __fmaf_rn(part[c * (QSA_D + 2) + 2 + j], f[c], o);
    float gate = P[(u64)t * stride + h * 512 + 256 + j];
    float y = o / L * sigm(gate);
    OUT[(u64)t * QSA_HEADS * QSA_D + h * QSA_D + j] = y;
    if (quant) quant_chunk(y, OQ, T, QSA_HEADS * QSA_D, t, h * 8 + (j >> 5), j & 31);
}
"#;
