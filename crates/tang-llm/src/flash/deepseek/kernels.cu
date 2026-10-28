// DeepSeek-V4.1-Flash device kernels, correctness first: each mirrors one op of the CPU
// reference (forward.rs / quant.rs), including where it rounds to bf16, so every kernel has a
// CPU twin to test against. Activations are f32 holding bf16-rounded values; weights stay in
// their checkpoint format and are decoded in the inner loop.
//
// Weight formats (struct DsW.fmt):
//   0 f32 [rows][cols]
//   1 bf16 [rows][cols] (u16)
//   2 fp8 e4m3 [rows][cols] + e8m0 scales [ceil(rows/32)][ceil(cols/32)]
//   3 fp4 e2m1 packed two to a byte along cols, low nibble first, + e8m0 scales [rows][cols/32]

// NVRTC has no <math.h> macros
#define DS_INF __int_as_float(0x7f800000)

struct DsW {
    unsigned long long w;
    unsigned long long s;
    int rows;
    int cols;
    int fmt;
};

__device__ __forceinline__ float bf(float x) {
    unsigned int b = __float_as_uint(x);
    if ((b & 0x7f800000u) == 0x7f800000u && (b & 0x7fffffu)) return x;  // NaN
    b += 0x7fffu + ((b >> 16) & 1u);
    return __uint_as_float(b & 0xffff0000u);
}

__device__ __forceinline__ float e4m3(unsigned char v) {
    float sign = (v & 0x80) ? -1.f : 1.f;
    int e = (v >> 3) & 0xf, m = v & 7;
    if (e == 0) return sign * (float)m * 0.001953125f;  // 2^-9
    return sign * (1.f + m / 8.f) * exp2f((float)(e - 7));
}

__device__ __forceinline__ float e8m0(unsigned char v) {
    if (v == 0) return 5.877471754111438e-39f;  // 2^-127
    return __uint_as_float(((unsigned int)v) << 23);
}

__device__ __forceinline__ float fp4(unsigned char code) {
    const float g[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    float m = g[code & 7];
    return (code & 8) ? -m : m;
}

__device__ __forceinline__ float pow2_ceil(float x) {
    unsigned int b = __float_as_uint(x);
    unsigned int e = (b >> 23) & 0xff;
    return __uint_as_float((e + ((b & 0x7fffffu) != 0)) << 23);
}

__device__ __forceinline__ float round_e4m3(float x) {
    float a = fabsf(x);
    if (a == 0.f) return 0.f * x;
    float step;
    if (a >= 0.015625f) {  // 2^-6: three mantissa bits in this binade
        int e = (int)((__float_as_uint(a) >> 23) & 0xff) - 127;
        step = exp2f((float)(e - 3));
    } else {
        step = 0.001953125f;
    }
    float q = fminf(rintf(a / step) * step, 448.f);
    return copysignf(q, x);
}

__device__ __forceinline__ float round_fp4(float x) {
    const float g[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    float a = fminf(fabsf(x), 6.f);
    int best = 0;
    for (int c = 1; c < 8; c++) {
        float d = fabsf(a - g[c]), db = fabsf(a - g[best]);
        if (d < db || (d == db && (c % 2 == 0) && (best % 2 == 1))) best = c;
    }
    return copysignf(g[best], x);
}

__device__ __forceinline__ float wget(const DsW& w, int r, int c) {
    switch (w.fmt) {
        case 0:
            return ((const float*)w.w)[(long long)r * w.cols + c];
        case 1: {
            unsigned short h = ((const unsigned short*)w.w)[(long long)r * w.cols + c];
            return __uint_as_float(((unsigned int)h) << 16);
        }
        case 2: {
            unsigned char v = ((const unsigned char*)w.w)[(long long)r * w.cols + c];
            int sc = (w.cols + 31) / 32;
            return e4m3(v) * e8m0(((const unsigned char*)w.s)[(long long)(r / 32) * sc + c / 32]);
        }
        default: {
            unsigned char v = ((const unsigned char*)w.w)[(long long)r * (w.cols / 2) + c / 2];
            unsigned char code = (c & 1) ? (v >> 4) : (v & 0xf);
            return fp4(code) * e8m0(((const unsigned char*)w.s)[(long long)r * (w.cols / 32) + c / 32]);
        }
    }
}

// y[t][n] = Σ_k x[t][k] w[n][k], one warp per output; rounded to bf16 when `round`.
extern "C" __global__ void ds_linear(DsW w, const float* x, float* y, int t, int round) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
    if (warp >= t * w.rows) return;
    int ti = warp / w.rows, n = warp % w.rows;
    const float* xr = x + (long long)ti * w.cols;
    float s = 0.f;
    for (int k = lane; k < w.cols; k += 32) s = fmaf(xr[k], wget(w, n, k), s);
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0) y[(long long)ti * w.rows + n] = round ? bf(s) : s;
}

// ds_linear for t <= 8 rows, one warp per output column n across all t rows: the weight row
// is read once, not t times. Bit-identical to ds_linear (same per-lane order and shuffle tree).
extern "C" __global__ void ds_linear_t8(DsW w, const float* x, float* y, int t, int round) {
    int n = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
    if (n >= w.rows) return;
    float s[8];
#pragma unroll
    for (int i = 0; i < 8; i++) s[i] = 0.f;
    int k = lane;
    if (w.fmt <= 1) {
        // loads for U steps issued ahead of the in-order FMAs (memory-level parallelism)
        const int U = 8;
        for (; k + 32 * (U - 1) < w.cols; k += 32 * U) {
            float wv[U];
#pragma unroll
            for (int u = 0; u < U; u++) wv[u] = wget(w, n, k + 32 * u);
#pragma unroll
            for (int i = 0; i < 8; i++) {
                if (i >= t) break;
                const float* xr = x + (long long)i * w.cols + k;
                float xv[U];
#pragma unroll
                for (int u = 0; u < U; u++) xv[u] = xr[32 * u];
#pragma unroll
                for (int u = 0; u < U; u++) s[i] = fmaf(xv[u], wv[u], s[i]);
            }
        }
    }
    for (; k < w.cols; k += 32) {
        float wv = wget(w, n, k);
#pragma unroll
        for (int i = 0; i < 8; i++)
            if (i < t) s[i] = fmaf(x[(long long)i * w.cols + k], wv, s[i]);
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
        if (i >= t) break;
        float v = s[i];
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) y[(long long)i * w.rows + n] = round ? bf(v) : v;
    }
}

// In place: fp8 quant-dequant per 32 with power-of-two scales (warp per group, lane per value).
extern "C" __global__ void ds_fp8_qdq(float* v, int n_groups) {
    int g = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (g >= n_groups) return;
    float* p = v + (long long)g * 32 + lane;
    float x = *p, amax = fabsf(x);
#pragma unroll
    for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    float s = pow2_ceil(fmaxf(amax, 1e-4f) / 448.f);
    *p = bf(round_e4m3(fminf(fmaxf(x / s, -448.f), 448.f)) * s);
}

// In place: fp4 quant-dequant per `block`, e4m3 scales when `e4m3_scale`, else power of two.
extern "C" __global__ void ds_fp4_qdq(float* v, int n_groups, int block, int e4m3_scale) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_groups) return;
    float* p = v + (long long)g * block;
    float amax = 0.f;
    for (int i = 0; i < block; i++) amax = fmaxf(amax, fabsf(p[i]));
    float s = e4m3_scale ? round_e4m3(fmaxf(amax, 6.f * 0.001953125f) / 6.f)
                         : pow2_ceil(fmaxf(amax, 6.f * 1.1754944e-38f) / 6.f);
    for (int i = 0; i < block; i++) p[i] = bf(round_fp4(fminf(fmaxf(p[i] / s, -6.f), 6.f)) * s);
}

// In place: RMSNorm of rows of n with weight w (any format, one row), bf16 out. Block per row.
extern "C" __global__ void ds_rms(float* x, DsW w, int n, float eps) {
    __shared__ float red[32];
    float* r = x + (long long)blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += r[i] * r[i];
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = s;
    __syncthreads();
    if (threadIdx.x < 32) {
        s = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : 0.f;
        for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (threadIdx.x == 0) red[0] = s;
    }
    __syncthreads();
    float rs = 1.f / sqrtf(red[0] / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) r[i] = bf(wget(w, 0, i) * (r[i] * rs));
}

// In place: rotate the last rd of each d-wide vector; row r (of per_row vectors) at position
// pos0 + r * pstride, or pos_fixed when >= 0. tab is [S][rd/2] of (cos, sin). One thread per pair.
extern "C" __global__ void ds_rope(float* v, const float2* tab, int rows, int per_row, int d, int rd,
                                   int pos0, int pstride, int pos_fixed, int inverse) {
    int half = rd / 2;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)rows * per_row * half) return;
    int j = i % half;
    long long vec = i / half;
    int r = vec / per_row;
    int pos = pos_fixed >= 0 ? pos_fixed : pos0 + r * pstride;
    float2 cs = tab[(long long)pos * half + j];
    float s = inverse ? -cs.y : cs.y;
    float* t = v + vec * d + (d - rd);
    float a = t[2 * j], b = t[2 * j + 1];
    t[2 * j] = bf(a * cs.x - b * s);
    t[2 * j + 1] = bf(a * s + b * cs.x);
}

// Gathered attention with a per-head sink: block per (t, h), d threads (d <= 1024).
// q [T][H][d], kv [n][d], idxs [T][k] (-1 = none), o [T][H][d] (bf16-rounded).
// Scores live in shared memory (k floats), so k <= smem / 4.
extern "C" __global__ void ds_sparse_attn(const float* q, const float* kv, const float* sink,
                                          const int* idxs, float* o, int h, int d, int k, float scale) {
    extern __shared__ float sc[];
    __shared__ float red[32];
    __shared__ float den_s;
    int t = blockIdx.x / h, hi = blockIdx.x % h;
    const float* qv = q + ((long long)t * h + hi) * d;
    const int* ix = idxs + (long long)t * k;
    for (int j = 0; j < k; j++) {
        int r = ix[j];
        float part = 0.f;
        if (r >= 0)
            for (int i = threadIdx.x; i < d; i += blockDim.x) part += qv[i] * kv[(long long)r * d + i];
        for (int o2 = 16; o2; o2 >>= 1) part += __shfl_xor_sync(0xffffffffu, part, o2);
        if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = part;
        __syncthreads();
        if (threadIdx.x == 0) {
            float s = 0.f;
            for (int w = 0; w < (blockDim.x + 31) / 32; w++) s += red[w];
            sc[j] = r >= 0 ? s * scale : -DS_INF;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float mx = -DS_INF;
        for (int j = 0; j < k; j++) mx = fmaxf(mx, sc[j]);
        mx = fmaxf(mx, -1e30f);
        float den = 0.f;
        for (int j = 0; j < k; j++) {
            float p = expf(sc[j] - mx);
            den += p;
            sc[j] = p;
        }
        den_s = den + expf(sink[hi] - mx);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j < k; j++) {
            int r = ix[j];
            if (r >= 0) acc += bf(sc[j]) * kv[(long long)r * d + i];
        }
        o[((long long)t * h + hi) * d + i] = bf(acc / den_s);
    }
}

// mHC mixes for one token per thread: mixes [T][mix] (already scaled by the row's rsqrt) ->
// pre [T][hc], post [T][hc], comb [T][hc][hc]. hc <= 8.
__device__ __forceinline__ void sinkhorn_row(const float* m, const float* scale, const float* base, float* pre,
                                             float* post, float* comb, int ti, int hc, int iters, float eps) {
    float c[64];
    for (int j = 0; j < hc; j++) {
        pre[ti * hc + j] = 1.f / (1.f + expf(-(m[j] * scale[0] + base[j]))) + eps;
        post[ti * hc + j] = 2.f / (1.f + expf(-(m[hc + j] * scale[1] + base[hc + j])));
    }
    for (int i = 0; i < hc * hc; i++) c[i] = m[2 * hc + i] * scale[2] + base[2 * hc + i];
    for (int j = 0; j < hc; j++) {
        float mx = -DS_INF, s = 0.f;
        for (int k = 0; k < hc; k++) mx = fmaxf(mx, c[j * hc + k]);
        for (int k = 0; k < hc; k++) { c[j * hc + k] = expf(c[j * hc + k] - mx); s += c[j * hc + k]; }
        for (int k = 0; k < hc; k++) c[j * hc + k] = c[j * hc + k] / s + eps;
    }
    for (int it = 0; it < iters; it++) {
        if (it > 0)
            for (int j = 0; j < hc; j++) {
                float s = 0.f;
                for (int k = 0; k < hc; k++) s += c[j * hc + k];
                for (int k = 0; k < hc; k++) c[j * hc + k] /= s + eps;
            }
        for (int k = 0; k < hc; k++) {
            float s = 0.f;
            for (int j = 0; j < hc; j++) s += c[j * hc + k];
            for (int j = 0; j < hc; j++) c[j * hc + k] /= s + eps;
        }
    }
    for (int i = 0; i < hc * hc; i++) comb[(long long)ti * hc * hc + i] = c[i];
}

extern "C" __global__ void ds_sinkhorn(const float* mixes, const float* scale, const float* base,
                                       float* pre, float* post, float* comb, int t, int hc, int iters, float eps) {
    int ti = blockIdx.x * blockDim.x + threadIdx.x;
    if (ti >= t) return;
    int mix = (2 + hc) * hc;
    sinkhorn_row(mixes + (long long)ti * mix, scale, base, pre, post, comb, ti, hc, iters, eps);
}

// ds_hc_scale then ds_sinkhorn in one launch: block per token (blockDim a multiple of 32,
// <= 1024), the row's rsqrt as ds_hc_scale (same reduction order), the scaled mixes
// (bf-exact: one f32 product each, as ds_hc_scale stores them) staged in shared memory, then
// thread 0 runs the token's Sinkhorn. `mixes` is left unscaled.
extern "C" __global__ void ds_hc_sinkhorn(const float* x, const float* mixes, int n, float neps, const float* scale,
                                          const float* base, float* pre, float* post, float* comb, int hc, int iters,
                                          float eps) {
    __shared__ float red[32];
    __shared__ float sm[80];
    int mix = (2 + hc) * hc;
    const float* r = x + (long long)blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += r[i] * r[i];
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float a = 0.f;
        for (int w = 0; w < blockDim.x / 32; w++) a += red[w];
        red[0] = 1.f / sqrtf(a / n + neps);
    }
    __syncthreads();
    for (int m = threadIdx.x; m < mix; m += blockDim.x) sm[m] = __fmul_rn(mixes[(long long)blockIdx.x * mix + m], red[0]);
    __syncthreads();
    if (threadIdx.x == 0) sinkhorn_row(sm, scale, base, pre, post, comb, blockIdx.x, hc, iters, eps);
}

// ---- the rest of the forward -------------------------------------------------------------

extern "C" __global__ void ds_copy(float* dst, const float* src, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// mixes[t][m] *= rsqrt(mean(x[t]^2) + eps), x rows of n. Block per row.
extern "C" __global__ void ds_hc_scale(const float* x, float* mixes, int n, int mix, float eps) {
    __shared__ float red[32];
    const float* r = x + (long long)blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += r[i] * r[i];
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float a = 0.f;
        for (int w = 0; w < blockDim.x / 32; w++) a += red[w];
        red[0] = 1.f / sqrtf(a / n + eps);
    }
    __syncthreads();
    for (int m = threadIdx.x; m < mix; m += blockDim.x) mixes[(long long)blockIdx.x * mix + m] *= red[0];
}

// out[t][i] = bf(Σ_c pre[t][c] x[t][c][i]). Thread per (t, i).
extern "C" __global__ void ds_hc_pre(const float* x, const float* pre, float* out, int t, int hc, int d) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)t * d) return;
    int ti = i / d, k = i % d;
    float s = 0.f;
    for (int c = 0; c < hc; c++) s += pre[ti * hc + c] * x[((long long)ti * hc + c) * d + k];
    out[i] = bf(s);
}

// out[t][k][i] = bf(post[t][k] h[t][i] + Σ_j comb[t][j][k] res[t][j][i]). Thread per element.
extern "C" __global__ void ds_hc_post(const float* h, const float* res, const float* post, const float* comb,
                                      float* out, int t, int hc, int d) {
    long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= (long long)t * hc * d) return;
    int i = e % d;
    int k = (e / d) % hc;
    int ti = e / ((long long)d * hc);
    float acc = 0.f;
    for (int j = 0; j < hc; j++) acc += comb[((long long)ti * hc + j) * hc + k] * res[((long long)ti * hc + j) * d + i];
    out[e] = bf(post[ti * hc + k] * h[(long long)ti * d + i] + acc);
}

// out[t][c][i] = table[ids[t]][i] for every copy c (bf16 table). Thread per element.
extern "C" __global__ void ds_embed(const unsigned short* table, const int* ids, float* out, int t, int copies, int d) {
    long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= (long long)t * copies * d) return;
    int i = e % d;
    int ti = e / ((long long)d * copies);
    out[e] = __uint_as_float(((unsigned int)table[(long long)ids[ti] * d + i]) << 16);
}

// out[t][i] = bf(mean_c x[t][c][i]).
extern "C" __global__ void ds_mean_hc(const float* x, float* out, int t, int hc, int d) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)t * d) return;
    int ti = i / d, k = i % d;
    float s = 0.f;
    for (int c = 0; c < hc; c++) s += x[((long long)ti * hc + c) * d + k];
    out[i] = bf(s / hc);
}

__device__ __forceinline__ float swiglu1(float gv, float uv, const float* w, float lim) {
    if (lim > 0.f) {
        gv = fminf(gv, lim);
        uv = fminf(fmaxf(uv, -lim), lim);
    }
    float v = gv / (1.f + expf(-gv)) * uv;
    return bf(w ? *w * v : v);
}

// h[r][j] = bf(w[r] * silu(min(g, lim)) * clamp(u, ±lim)); w null for the shared expert.
extern "C" __global__ void ds_swiglu(const float* g, const float* u, const float* w, float* h, long long n, int inter, float lim) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    h[i] = swiglu1(g[i], u[i], w ? w + i / inter : 0, lim);
}

// The router, one block per token: raw [t][ne] gate logits (f32) -> ids/weights [t][k].
// sqrt(softplus) scores, top-k of score + bias (ties to the lower id), weights normalized and
// scaled. k <= 16; dynamic shared memory: ne floats (score + bias) + ne bytes (used flags).
// Each round is a block argmax; the weight sum runs serially in selection order, as before.
__device__ __forceinline__ bool router_better(float v, int e, float bv, int be) {
    return be < 0 || (e >= 0 && (v > bv || (v == bv && e < be)));
}

extern "C" __global__ void ds_router(const float* raw, const float* bias, int* ids, float* wts, int t, int ne, int k,
                                     int norm, float route_scale) {
    extern __shared__ float sh[];
    float* sv = sh;
    unsigned char* used = (unsigned char*)(sh + ne);
    __shared__ float rv[32];
    __shared__ int re[32];
    __shared__ int sel[16];
    int ti = blockIdx.x, lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    const float* r = raw + (long long)ti * ne;
    for (int e = threadIdx.x; e < ne; e += blockDim.x) {
        float x = r[e];
        float sc = sqrtf(x > 20.f ? x : log1pf(expf(x)));
        sv[e] = sc + bias[e];
        used[e] = 0;
    }
    __syncthreads();
    for (int j = 0; j < k; j++) {
        int be = -1;
        float bv = 0.f;
        for (int e = threadIdx.x; e < ne; e += blockDim.x)
            if (!used[e] && router_better(sv[e], e, bv, be)) { be = e; bv = sv[e]; }
        for (int o = 16; o; o >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            int oe = __shfl_xor_sync(0xffffffffu, be, o);
            if (router_better(ov, oe, bv, be)) { be = oe; bv = ov; }
        }
        if (lane == 0) { rv[wid] = bv; re[wid] = be; }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int q = 1; q < nw; q++)
                if (router_better(rv[q], re[q], bv, be)) { be = re[q]; bv = rv[q]; }
            sel[j] = be;
            used[be] = 1;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float w[16], sum = 0.f;
        for (int j = 0; j < k; j++) {
            float x = r[sel[j]];
            w[j] = sqrtf(x > 20.f ? x : log1pf(expf(x)));
            sum += w[j];
        }
        for (int j = 0; j < k; j++) {
            float v = (norm && k > 1) ? w[j] / (sum + 1e-20f) : w[j];
            ids[ti * k + j] = sel[j];
            wts[ti * k + j] = v * route_scale;
        }
    }
}

// out[j][i] = x[rows[j]][i].
extern "C" __global__ void ds_gather_rows(const float* x, const int* rows, float* out, int n, int d) {
    long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= (long long)n * d) return;
    out[e] = x[(long long)rows[e / d] * d + e % d];
}

// y[rows[j]][i] += v[j][i] (rows unique within one launch).
extern "C" __global__ void ds_scatter_add(float* y, const float* v, const int* rows, int n, int d) {
    long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= (long long)n * d) return;
    y[(long long)rows[e / d] * d + e % d] += v[e];
}

// y[i] += v[i].
extern "C" __global__ void ds_add(float* y, const float* v, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += v[i];
}

// y[i] = bf(y[i] + v[i]).
extern "C" __global__ void ds_add_round(float* y, const float* v, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = bf(y[i] + v[i]);
}

// v[i] = bf(v[i] * s).
extern "C" __global__ void ds_scale_round(float* v, float s, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] = bf(v[i] * s);
}

// v[i] = bf(v[i]).
extern "C" __global__ void ds_round(float* v, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] = bf(v[i]);
}

// The ratio-r compressor's pooling: out[g][c] = Σ_j kv[g*r+j][c] softmax_j(sc[g*r+j][c]).
extern "C" __global__ void ds_pool(const float* kv, const float* sc, float* out, int groups, int r, int hd) {
    long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= (long long)groups * hd) return;
    int g = e / hd, c = e % hd;
    float mx = -DS_INF;
    for (int j = 0; j < r; j++) mx = fmaxf(mx, sc[((long long)g * r + j) * hd + c]);
    float s = 0.f;
    for (int j = 0; j < r; j++) s += expf(sc[((long long)g * r + j) * hd + c] - mx);
    float acc = 0.f;
    for (int j = 0; j < r; j++) {
        long long o = ((long long)g * r + j) * hd + c;
        acc += kv[o] * (expf(sc[o] - mx) / s);
    }
    out[e] = acc;
}

// The indexer's scores, warp per (t, j): score = bf(Σ_h bf(relu(bf(q_h·k_j)) * w[t][h])).
extern "C" __global__ void ds_index_score(const float* q, const float* ik, const float* w, float* score,
                                          int t, int width, int nh, int ihd) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
    if (warp >= t * width) return;
    int ti = warp / width, j = warp % width;
    const float* kj = ik + (long long)j * ihd;
    float s = 0.f;
    for (int h = 0; h < nh; h++) {
        const float* qv = q + ((long long)ti * nh + h) * ihd;
        float d = 0.f;
        for (int i = lane; i < ihd; i += 32) d += qv[i] * kj[i];
        for (int o = 16; o; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
        s += bf(fmaxf(bf(d), 0.f) * w[ti * nh + h]);
    }
    if (lane == 0) score[warp] = bf(s);
}

// wo_a, block-diagonal over groups: mid[t][g][r] = bf(Σ_i o[t][g][i] w[g*ol + r][i]).
// Warp per output.
extern "C" __global__ void ds_wo_a(const float* o, DsW w, float* mid, int t, int g, int ol) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
    if (warp >= t * g * ol) return;
    int r = warp % ol, gi = (warp / ol) % g, ti = warp / (ol * g);
    int gd = w.cols;
    const float* ov = o + ((long long)ti * g + gi) * gd;
    float s = 0.f;
    for (int i = lane; i < gd; i += 32) s += ov[i] * bf(wget(w, gi * ol + r, i));
    for (int o2 = 16; o2; o2 >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o2);
    if (lane == 0) mid[warp] = bf(s);
}

// Engram's gate, block per (t, c): out = bf(h + sigmoid(signed sqrt(normalized h·qk·key)) val).
// kv [t][(hc + 1) d] (keys, then the shared value), qk [hc][d] = q_weight * k_weight.
extern "C" __global__ void ds_engram_gate(const float* x, const float* kv, const float* qk, float* out,
                                          int hc, int d, float eps) {
    __shared__ float red[3][32];
    int ti = blockIdx.x / hc, c = blockIdx.x % hc;
    const float* h = x + ((long long)ti * hc + c) * d;
    const float* key = kv + (long long)ti * (hc + 1) * d + (long long)c * d;
    const float* val = kv + (long long)ti * (hc + 1) * d + (long long)hc * d;
    float hh = 0.f, kk = 0.f, dot = 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        hh += h[i] * h[i];
        kk += key[i] * key[i];
        dot += h[i] * qk[c * d + i] * key[i];
    }
    for (int o = 16; o; o >>= 1) {
        hh += __shfl_xor_sync(0xffffffffu, hh, o);
        kk += __shfl_xor_sync(0xffffffffu, kk, o);
        dot += __shfl_xor_sync(0xffffffffu, dot, o);
    }
    if ((threadIdx.x & 31) == 0) { red[0][threadIdx.x / 32] = hh; red[1][threadIdx.x / 32] = kk; red[2][threadIdx.x / 32] = dot; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float a = 0.f, b = 0.f, e = 0.f;
        for (int w = 0; w < blockDim.x / 32; w++) { a += red[0][w]; b += red[1][w]; e += red[2][w]; }
        float rstd = 1.f / sqrtf(a / d + eps) * (1.f / sqrtf(b / d + eps));
        float dt = e * rstd * powf((float)d, -0.5f);
        float gs = sqrtf(fmaxf(fabsf(dt), 1e-6f));
        red[0][0] = 1.f / (1.f + expf(-copysignf(gs, dt)));
    }
    __syncthreads();
    float gate = red[0][0];
    float* o = out + ((long long)ti * hc + c) * d;
    for (int i = threadIdx.x; i < d; i += blockDim.x) o[i] = bf(h[i] + gate * val[i]);
}

// ---- fast GEMV (decode, t <= 8 per launch) for fp8 (fmt 2) and fp4 (fmt 3) weights ----
//
// Bit-identical to ds_linear: lane l of the warp owning a row accumulates x[k] * w[k] for
// k = l, l + 32, ... in increasing k with fmaf, then the same xor butterfly. The weights are
// read with one 16-byte load per lane (32 fp4 / 16 fp8 values), decoded in registers and
// transposed through shared memory so each lane gets its k = l (mod 32) column. Each warp owns
// LR = 4 weight rows (8 lanes load each row's consecutive 128 bytes per tile), so one
// activation read serves 4 rows, and all t <= TM token rows share each weight read.
// `qdq` fuses ds_fp8_qdq: each 32-group is exactly one warp-wide column step, so its amax is a
// warp max and the result is bit-identical to quantizing first.
// GU: rows 0,1 come from w (w1) and rows 2,3 from w3 at the same output rows; the epilogue
// emits h = swiglu1(bf(g), bf(u)) (ds_swiglu), `rw` the per-token route weights or null.

#define LR 4
#define LU 4  // tiles in flight per lane

__device__ __forceinline__ float fp4v(unsigned int c) {
    unsigned int m = c & 7, e = m >> 1;
    unsigned int b = e ? (((e + 126u) << 23) | ((m & 1u) << 22)) : ((m & 1u) ? (126u << 23) : 0u);
    return __uint_as_float(b | ((c & 8u) << 28));
}

__device__ __forceinline__ float e4m3v(unsigned int v) {
    unsigned int e = (v >> 3) & 0xf, m = v & 7;
    float a = e ? __uint_as_float(((e + 120u) << 23) | (m << 20)) : (float)m * 0.001953125f;
    return (v & 0x80) ? -a : a;
}

template <int FMT, int TM, int GU>
__device__ __forceinline__ void lin_fast(DsW w, DsW w3, const float* __restrict__ x, float* __restrict__ y, int t,
                                         int round, int qdq, const float* __restrict__ rw, float lim,
                                         const int* __restrict__ xi) {
    // chunks (32 columns) per tile per row: 8 lanes x 16 bytes
    const int CT = FMT == 3 ? 8 : 4;
    __shared__ float sh[4][LR][8][33];
    int wib = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float(*S)[8][33] = sh[wib];
    const int OUT = GU ? LR / 2 : LR;
    int n0 = (blockIdx.x * (blockDim.x >> 5) + wib) * OUT;
    int cols = w.cols, nch = cols / 32;
    int r = lane >> 3, sub = lane & 7;
    // the row this lane loads
    int nr = GU ? n0 + (r & 1) : n0 + r;
    const DsW& wr = (GU && r >= 2) ? w3 : w;
    bool rv = nr < w.rows;
    long long rbytes = FMT == 3 ? cols / 2 : cols;
    const unsigned char* wp = (const unsigned char*)wr.w + (rv ? (long long)nr * rbytes : 0);
    const unsigned char* sp = (const unsigned char*)wr.s;
    // x row of token ti: xi[ti] when gathering (grouped MoE), else ti
    const float* xr[TM];
#pragma unroll
    for (int ti = 0; ti < TM; ti++) xr[ti] = x + (long long)(xi && ti < t ? xi[ti] : ti) * cols;
    float acc[LR][TM];
#pragma unroll
    for (int i = 0; i < LR; i++)
#pragma unroll
        for (int j = 0; j < TM; j++) acc[i][j] = 0.f;
    int ntiles = (nch + CT - 1) / CT;
    for (int tg = 0; tg < ntiles; tg += LU) {
        uint4 buf[LU];
        float scl[LU];
#pragma unroll
        for (int u = 0; u < LU; u++) {
            int tile = tg + u;
            // fp4: this lane's chunk; fp8: this lane's half chunk
            int c = FMT == 3 ? tile * 8 + sub : (tile * 8 + sub) >> 1;
            bool ok = rv && tile < ntiles && c < nch;
            buf[u] = make_uint4(0, 0, 0, 0);
            scl[u] = 0.f;
            if (ok) {
                buf[u] = __ldg((const uint4*)(wp + (long long)(tile * 8 + sub) * 16));
                unsigned char sv = FMT == 3 ? __ldg(sp + (long long)nr * nch + c) : __ldg(sp + (long long)(nr / 32) * nch + c);
                scl[u] = e8m0(sv);
            }
        }
#pragma unroll
        for (int u = 0; u < LU; u++) {
            int tile = tg + u;
            if (tile >= ntiles) break;
            unsigned int q[4] = {buf[u].x, buf[u].y, buf[u].z, buf[u].w};
            if (FMT == 3) {
                float* d = &S[r][sub][0];
#pragma unroll
                for (int i = 0; i < 4; i++)
#pragma unroll
                    for (int b = 0; b < 8; b++) d[i * 8 + b] = fp4v((q[i] >> (4 * b)) & 15u) * scl[u];
            } else {
                float* d = &S[r][sub >> 1][(sub & 1) * 16];
#pragma unroll
                for (int i = 0; i < 4; i++)
#pragma unroll
                    for (int b = 0; b < 4; b++) d[i * 4 + b] = e4m3v((q[i] >> (8 * b)) & 255u) * scl[u];
            }
            __syncwarp();
            int nj = min(CT, nch - tile * CT);
            for (int j = 0; j < nj; j++) {
                int k = (tile * CT + j) * 32 + lane;
#pragma unroll
                for (int ti = 0; ti < TM; ti++) {
                    if (ti < t) {
                        float xv = xr[ti][k];
                        if (qdq) {
                            float a = fabsf(xv);
#pragma unroll
                            for (int o = 16; o; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
                            float s = pow2_ceil(fmaxf(a, 1e-4f) / 448.f);
                            xv = bf(round_e4m3(fminf(fmaxf(xv / s, -448.f), 448.f)) * s);
                        }
#pragma unroll
                        for (int i = 0; i < LR; i++) acc[i][ti] = fmaf(xv, S[i][j][lane], acc[i][ti]);
                    }
                }
            }
            __syncwarp();
        }
    }
#pragma unroll
    for (int i = 0; i < LR; i++)
#pragma unroll
        for (int ti = 0; ti < TM; ti++)
#pragma unroll
            for (int o = 16; o; o >>= 1) acc[i][ti] += __shfl_xor_sync(0xffffffffu, acc[i][ti], o);
    if (lane != 0) return;
#pragma unroll
    for (int i = 0; i < OUT; i++) {
        int n = n0 + i;
        if (n >= w.rows) break;
#pragma unroll
        for (int ti = 0; ti < TM; ti++) {
            if (ti >= t) break;
            if (GU) {
                y[(long long)ti * w.rows + n] = swiglu1(bf(acc[i][ti]), bf(acc[i + 2][ti]), rw ? rw + ti : 0, lim);
            } else {
                float s = acc[i][ti];
                y[(long long)ti * w.rows + n] = round ? bf(s) : s;
            }
        }
    }
}

#define DS_LIN(F, T, G)                                                                                       \
    extern "C" __global__ void __launch_bounds__(128) ds_lin_f##F##_t##T##_g##G(                             \
        DsW w, DsW w3, const float* x, float* y, int t, int round, int qdq, const float* rw, float lim) {     \
        lin_fast<F, T, G>(w, w3, x, y, t, round, qdq, rw, lim, 0);                                            \
    }
#define DS_LIN_T(F, G) DS_LIN(F, 1, G) DS_LIN(F, 2, G) DS_LIN(F, 4, G) DS_LIN(F, 8, G)
DS_LIN_T(2, 0)
DS_LIN_T(3, 0)
DS_LIN_T(2, 1)
DS_LIN_T(3, 1)

// ---- grouped MoE: one launch per projection for every routed expert of a layer ----
//
// blockIdx.y is a slot of `elist` (ds_moe_plan: the live local experts ascending, -1 past
// them); blockIdx.x tiles the weight rows exactly as ds_lin_*. Expert e's rows are its CSR
// segment off[e]..off[e+1] (ds_route), at most T of them. Both are lin_fast, bit-identical to
// the per-expert ds_linear_csr loop: GU gathers its x rows through `rows` (xq: x already
// fp8-quantized) into h [t*k][inter] at the CSR positions, with SwiGLU and the route weights;
// down maps hq (h quantized) to per-slot partials part [t*k][d], summed by ds_moe_reduce.
#define DS_MOE(F, T)                                                                                          \
    extern "C" __global__ void __launch_bounds__(128) ds_moe_gu_f##F##_t##T(                                 \
        const DsW* w1s, const DsW* w3s, const int* elist, const int* off, const int* rows, const float* rw,  \
        const float* xq, float* h, float lim) {                                                               \
        int e = elist[blockIdx.y];                                                                            \
        if (e < 0) return;                                                                                    \
        int o = off[e], n = off[e + 1] - o;                                                                   \
        DsW w = w1s[e];                                                                                       \
        lin_fast<F, T, 1>(w, w3s[e], xq, h + (long long)o * w.rows, n, 1, 0, rw + o, lim, rows + o);          \
    }                                                                                                         \
    extern "C" __global__ void __launch_bounds__(128) ds_moe_down_f##F##_t##T(                               \
        const DsW* w2s, const int* elist, const int* off, const float* hq, float* part) {                    \
        int e = elist[blockIdx.y];                                                                            \
        if (e < 0) return;                                                                                    \
        int o = off[e], n = off[e + 1] - o;                                                                   \
        DsW w = w2s[e];                                                                                       \
        lin_fast<F, T, 0>(w, w, hq + (long long)o * w.cols, part + (long long)o * w.rows, n, 1, 0, 0, 0.f, 0); \
    }
#define DS_MOE_T(F) DS_MOE(F, 1) DS_MOE(F, 2) DS_MOE(F, 4) DS_MOE(F, 8)
DS_MOE_T(2)
DS_MOE_T(3)

// ---- tensor-core K-split GEMV (the default fast path; DS_EXACT=1 selects lin_fast above) ----
//
// Not bit-exact with ds_linear (the reduction order differs); outputs agree with it to within
// one bf16 ulp in rare elements. The activation is fp8-quantized per 32-group while it is
// staged in shared memory (bit-identical to ds_fp8_qdq; idempotent on an already quantized
// activation), and kept as its e4m3 value in f16 plus the group scale. Weights decode with a
// few integer ops to f16 (e4m3 is f16 × 2^8, e2m1 is f16 × 2^14, both exact), so each 32-column
// group of a 16-row × 8-token tile is two mma.m16n8k16 (f16 in, f32 accumulate), scaled by
// weight scale × activation scale into f32 accumulators. A warp owns 16 weight rows (GU: rows
// 0-7 of w1 and the same rows of w3); a block of 8 warps is RW = 8 / S row tiles × S K-slices,
// the slices meeting in shared memory in a fixed order (deterministic). Within an mma's k step
// columns are permuted freely (A and B agree), so each lane's weight bytes are contiguous.

#define KS_UG 8  // groups in flight per warp

__device__ __forceinline__ unsigned short f2h_exact(float v) {
    unsigned int b = __float_as_uint(v);
    unsigned int sg = (b >> 16) & 0x8000u, a = b & 0x7fffffffu;
    if (a == 0) return (unsigned short)sg;
    return (unsigned short)(sg | (((a >> 23) - 112u) << 10) | ((a >> 13) & 0x3ffu));
}

// two e4m3 in bits 8-15 and 24-31 -> f16x2 of value / 2^8
__device__ __forceinline__ unsigned int f8h(unsigned int u) { return (u & 0x80008000u) | ((u >> 1) & 0x3f803f80u); }
// two e2m1 in bits 12-15 and 28-31 -> f16x2 of value / 2^14
__device__ __forceinline__ unsigned int f4h(unsigned int u) { return (u & 0x80008000u) | ((u >> 3) & 0x0e000e00u); }

__device__ __forceinline__ void mma16816(float* c, unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
                                         unsigned int b0, unsigned int b1) {
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

// __CUDA_ARCH__ of the module, so the host can fall back to the exact kernels below sm_80.
extern "C" __global__ void ds_arch(int* out) { *out = __CUDA_ARCH__; }

// e4m3-valued float -> its e4m3 byte (|v| <= 448, on the grid)
__device__ __forceinline__ unsigned int e4m3_byte(float v) {
    unsigned int b = __float_as_uint(v), sg = (b >> 24) & 0x80u, a = b & 0x7fffffffu;
    if (a == 0) return sg;
    int e = (int)(a >> 23) - 127;
    if (e < -6) return sg | (unsigned int)(__uint_as_float(a) * 512.f);  // subnormal: m * 2^-9
    return sg | ((unsigned int)(e + 7) << 3) | ((a >> 20) & 7u);
}

// The activation, quantized once: codes [t][K] (e4m3 bytes) and scales [t][K/32] (pow2), bit
// for bit ds_fp8_qdq's. Warp per 32-group.
extern "C" __global__ void ds_ks_quant(const float* x, unsigned char* codes, float* sx, int n_groups) {
    int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (gw >= n_groups) return;
    float xv = x[(long long)gw * 32 + lane];
    float a = fabsf(xv);
#pragma unroll
    for (int o = 16; o; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    float s = pow2_ceil(fmaxf(a, 1e-4f) / 448.f);
    codes[(long long)gw * 32 + lane] = (unsigned char)e4m3_byte(round_e4m3(fminf(fmaxf(xv / s, -448.f), 448.f)));
    if (lane == 0) sx[gw] = s;
}

// Persistent blocks: each stages the whole quantized activation ([t][K] codes, row pitch
// P = K + 32 bytes for conflict-free 8-byte reads, and scales) once, then takes work items
// (row group br, K slice sb) = blockIdx.x, += gridDim.x. With SB > 1 slices, partials go to
// part[sb][t][rows (× 2 for GU)] and the last block of a row group (cnt) sums them in order.
template <int FMT, int GU>
__device__ __forceinline__ void lin_mma(DsW w, DsW w3, const unsigned char* __restrict__ xc,
                                        const float* __restrict__ xsc, float* __restrict__ y, int t, int round,
                                        const float* __restrict__ rw, float lim, int S, int SB, int nbr, float* part,
                                        int* cnt) {
    extern __shared__ __align__(16) unsigned char smem[];
    __shared__ float red[8][4][32];
    __shared__ int last;
    int cols = w.cols, nch = cols / 32, P = ((cols + 127) / 128) * 128 + 32;
    unsigned char* xs = smem;           // [t][P]
    float* sx = (float*)(smem + t * P);  // [t][nch], scale × 2^8 (code -> f16) × weight POW
    const float POW = (FMT == 3 ? 16384.f : 256.f) * 256.f;
    for (int e = threadIdx.x; e < t * cols / 16; e += blockDim.x) {
        int ti = e / (cols / 16), c = e % (cols / 16);
        *(uint4*)(xs + ti * P + c * 16) = __ldg((const uint4*)(xc + (long long)ti * cols) + c);
    }
    for (int e = threadIdx.x; e < t * nch; e += blockDim.x) sx[e] = __ldg(xsc + e) * POW;
    __syncthreads();
    int wib = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
    int RW = 8 / S, rwi = wib / S, si = wib % S;
    const int OUT = GU ? 8 : 16;
    const int R2 = GU ? 2 * w.rows : w.rows;
    long long rbytes = FMT == 3 ? cols / 2 : cols;
    int gpb = (nch + SB - 1) / SB;
    for (int item = blockIdx.x; item < nbr * SB; item += gridDim.x) {
        int br = item / SB, sb = item % SB;
        int n0 = (br * RW + rwi) * OUT;
        int ra = n0 + g, rb = GU ? n0 + g : n0 + g + 8;
        bool va = ra < w.rows, vb = rb < w.rows;
        const unsigned char* pa =
            (const unsigned char*)w.w + (va ? (long long)ra * rbytes : 0) + (FMT == 3 ? 4 : 8) * tig;
        const unsigned char* pb =
            (const unsigned char*)(GU ? w3.w : w.w) + (vb ? (long long)rb * rbytes : 0) + (FMT == 3 ? 4 : 8) * tig;
        const unsigned char* sa =
            (const unsigned char*)w.s + (FMT == 3 ? (long long)ra * nch : (long long)(ra / 32) * nch);
        const unsigned char* sbp =
            (const unsigned char*)(GU ? w3.s : w.s) + (FMT == 3 ? (long long)rb * nch : (long long)(rb / 32) * nch);
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        int glo = sb * gpb, ghi = min(nch, glo + gpb);
        for (int q0 = glo + si; q0 < ghi; q0 += KS_UG * S) {
            uint2 la[KS_UG], lb[KS_UG];
            unsigned char ca[KS_UG], cb[KS_UG];
#pragma unroll
            for (int u = 0; u < KS_UG; u++) {
                int G = q0 + u * S;
                la[u] = lb[u] = make_uint2(0, 0);
                ca[u] = cb[u] = 0;
                if (G < ghi) {
                    if (va) {
                        if (FMT == 3) la[u].x = __ldg((const unsigned int*)(pa + (long long)G * 16));
                        else la[u] = __ldg((const uint2*)(pa + (long long)G * 32));
                        ca[u] = __ldg(sa + G);
                    }
                    if (vb) {
                        if (FMT == 3) lb[u].x = __ldg((const unsigned int*)(pb + (long long)G * 16));
                        else lb[u] = __ldg((const uint2*)(pb + (long long)G * 32));
                        cb[u] = __ldg(sbp + G);
                    }
                }
            }
#pragma unroll
            for (int u = 0; u < KS_UG; u++) {
                int G = q0 + u * S;
                if (G >= ghi) break;
                uint2 X = g < t ? *(const uint2*)(xs + g * P + G * 32 + 8 * tig) : make_uint2(0, 0);
                float c[4] = {0.f, 0.f, 0.f, 0.f};
                if (FMT == 3) {
                    unsigned int wa = la[u].x, wb = lb[u].x;
                    mma16816(c, f4h(wa << 12), f4h(wb << 12), f4h(wa << 8), f4h(wb << 8),
                             f8h(__byte_perm(X.x, X.y, 0x4000)), f8h(__byte_perm(X.x, X.y, 0x5010)));
                    mma16816(c, f4h(wa << 4), f4h(wb << 4), f4h(wa), f4h(wb), f8h(__byte_perm(X.x, X.y, 0x6020)),
                             f8h(__byte_perm(X.x, X.y, 0x7030)));
                } else {
                    mma16816(c, f8h(la[u].x << 8), f8h(lb[u].x << 8), f8h(la[u].x), f8h(lb[u].x), f8h(X.x << 8),
                             f8h(X.x));
                    mma16816(c, f8h(la[u].y << 8), f8h(lb[u].y << 8), f8h(la[u].y), f8h(lb[u].y), f8h(X.y << 8),
                             f8h(X.y));
                }
                float x0 = 2 * tig < t ? sx[2 * tig * nch + G] : 0.f;
                float x1 = 2 * tig + 1 < t ? sx[(2 * tig + 1) * nch + G] : 0.f;
                float swa = e8m0(ca[u]), swb = e8m0(cb[u]);
                acc[0] = fmaf(c[0], swa * x0, acc[0]);
                acc[1] = fmaf(c[1], swa * x1, acc[1]);
                acc[2] = fmaf(c[2], swb * x0, acc[2]);
                acc[3] = fmaf(c[3], swb * x1, acc[3]);
            }
        }
        __syncthreads();  // red reuse across items
#pragma unroll
        for (int i = 0; i < 4; i++) red[wib][i][lane] = acc[i];
        __syncthreads();
        if (SB > 1) {
            for (int e = threadIdx.x; e < RW * OUT * t; e += blockDim.x) {
                int j = e % t, o = (e / t) % OUT, rg = e / (t * OUT);
                int n = (br * RW + rg) * OUT + o;
                if (n >= w.rows) continue;
                int ln = (o & 7) * 4 + j / 2, i = j & 1;
                float a = 0.f, b = 0.f;
                for (int s = 0; s < S; s++) {
                    a += red[rg * S + s][GU || o < 8 ? i : 2 + i][ln];
                    if (GU) b += red[rg * S + s][2 + i][ln];
                }
                float* pp = part + ((long long)sb * t + j) * R2;
                pp[n] = a;
                if (GU) pp[w.rows + n] = b;
            }
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0) {
                last = atomicAdd(&cnt[br], 1) == SB - 1;
                if (last) cnt[br] = 0;
            }
            __syncthreads();
            if (!last) continue;
            __threadfence();
        }
        for (int e = threadIdx.x; e < RW * OUT * t; e += blockDim.x) {
            int j = e % t, o = (e / t) % OUT, rg = e / (t * OUT);
            int n = (br * RW + rg) * OUT + o;
            if (n >= w.rows) continue;
            // C fragment: (row g, token 2tig + i) in acc[i], (row g + 8, ...) in acc[2 + i]
            int ln = (o & 7) * 4 + j / 2, i = j & 1;
            float a = 0.f, b = 0.f;
            if (SB > 1) {
                for (int q = 0; q < SB; q++) {
                    const float* pp = part + ((long long)q * t + j) * R2;
                    a += __ldcg(pp + n);
                    if (GU) b += __ldcg(pp + w.rows + n);
                }
            } else {
                for (int s = 0; s < S; s++) {
                    a += red[rg * S + s][GU || o < 8 ? i : 2 + i][ln];
                    if (GU) b += red[rg * S + s][2 + i][ln];
                }
            }
            if (GU) y[(long long)j * w.rows + n] = swiglu1(bf(a), bf(b), rw ? rw + j : 0, lim);
            else y[(long long)j * w.rows + n] = round ? bf(a) : a;
        }
    }
}

#define DS_KS(F, G)                                                                                             \
    extern "C" __global__ void __launch_bounds__(256) ds_ks_f##F##_g##G(                                        \
        DsW w, DsW w3, const unsigned char* xc, const float* xsc, float* y, int t, int round, const float* rw,  \
        float lim, int S, int SB, int nbr, float* part, int* cnt) {                                             \
        lin_mma<F, G>(w, w3, xc, xsc, y, t, round, rw, lim, S, SB, nbr, part, cnt);                             \
    }
DS_KS(2, 0)
DS_KS(3, 0)
DS_KS(2, 1)
DS_KS(3, 1)

// ---- tiled sparse attention (default; DS_EXACT=1 selects ds_sparse_attn) ---------------------
//
// Three launches over a scratch buffer, same numerics as ds_sparse_attn up to summation order:
// 1. ds_attn_qk: block per (t, 64-head chunk, 32-key tile, 64-wide d slice): the key tile and
//    the heads' q slice in shared memory, every head scores every key of the tile (4 heads ×
//    2 keys a thread); partial dot products to part[t][d/64][h][k].
// 2. ds_attn_soft: warp per (t, h): score = Σ partials × scale (-inf for idx -1), softmax with
//    the sink in the denominator; P = bf(p) to pr[t][h][k], den to den[t][h].
// 3. ds_attn_pv: block per (t, 8 heads, 64-wide d slice), 4 key slices × 64 dims: each kv
//    value is read once for 8 heads; o = bf(Σ P kv / den).

extern "C" __global__ void __launch_bounds__(256) ds_attn_qk(const float* q, const float* kv, const int* idxs,
                                                             float* part, int h, int d, int k) {
    __shared__ float Qs[64][65];
    __shared__ float Ks[32][65];
    int nkt = (k + 31) / 32, nds = d / 64, nhc = h / 64;
    int b = blockIdx.x;
    int ds = b % nds;
    int kt = (b / nds) % nkt;
    int hc = (b / (nds * nkt)) % nhc;
    int ti = b / (nds * nkt * nhc);
    int tid = threadIdx.x;
    const int* ix = idxs + (long long)ti * k;
    for (int e = tid; e < 64 * 64; e += 256) {
        int hh = e / 64, i = e % 64;
        Qs[hh][i] = q[((long long)ti * h + hc * 64 + hh) * d + ds * 64 + i];
    }
    for (int e = tid; e < 32 * 64; e += 256) {
        int kk = e / 64, i = e % 64;
        int j = kt * 32 + kk;
        int r = j < k ? ix[j] : -1;
        Ks[kk][i] = r >= 0 ? kv[(long long)r * d + ds * 64 + i] : 0.f;
    }
    __syncthreads();
    int kq = tid & 15, hq = tid >> 4;
    float a[4][2] = {{0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};
#pragma unroll 8
    for (int i = 0; i < 64; i++) {
        float k0 = Ks[kq * 2][i], k1 = Ks[kq * 2 + 1][i];
#pragma unroll
        for (int m = 0; m < 4; m++) {
            float qv = Qs[hq * 4 + m][i];
            a[m][0] = fmaf(qv, k0, a[m][0]);
            a[m][1] = fmaf(qv, k1, a[m][1]);
        }
    }
#pragma unroll
    for (int m = 0; m < 4; m++)
#pragma unroll
        for (int c = 0; c < 2; c++) {
            int j = kt * 32 + kq * 2 + c;
            if (j < k) part[(((long long)ti * nds + ds) * h + hc * 64 + hq * 4 + m) * k + j] = a[m][c];
        }
}

extern "C" __global__ void ds_attn_soft(const float* part, const float* sink, const int* idxs, float* pr,
                                        float* den, int t, int h, int d, int k, float scale) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (warp >= t * h) return;
    int ti = warp / h, hi = warp % h, nds = d / 64;
    const int* ix = idxs + (long long)ti * k;
    float* p = pr + (long long)warp * k;
    float mx = -DS_INF;
    for (int j = lane; j < k; j += 32) {
        float s = 0.f;
        for (int c = 0; c < nds; c++) s += part[(((long long)ti * nds + c) * h + hi) * k + j];
        s = ix[j] >= 0 ? s * scale : -DS_INF;
        p[j] = s;
        mx = fmaxf(mx, s);
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    mx = fmaxf(mx, -1e30f);
    float sum = 0.f;
    for (int j = lane; j < k; j += 32) {
        float e = expf(p[j] - mx);
        sum += e;
        p[j] = bf(e);
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if (lane == 0) den[warp] = sum + expf(sink[hi] - mx);
}

extern "C" __global__ void __launch_bounds__(256) ds_attn_pv(const float* pr, const float* den, const float* kv,
                                                             const int* idxs, float* o, int h, int d, int k) {
    extern __shared__ float P[];  // [8][k], then the k indices
    __shared__ float red[4][8][64];
    int* Is = (int*)(P + 8 * k);
    int nds = d / 64, hg = h / 8;
    int b = blockIdx.x;
    int ds = b % nds, g = (b / nds) % hg, ti = b / (nds * hg);
    int tid = threadIdx.x;
    for (int e = tid; e < 8 * k; e += 256) P[e] = pr[((long long)ti * h + g * 8) * k + e];
    for (int e = tid; e < k; e += 256) Is[e] = idxs[(long long)ti * k + e];
    __syncthreads();
    int ks = tid >> 6, di = tid & 63;
    const float* kc = kv + ds * 64 + di;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int j0 = ks; j0 < k; j0 += 32) {
        float v[8];
#pragma unroll
        for (int u = 0; u < 8; u++) {
            int j = j0 + 4 * u;
            int r = j < k ? Is[j] : -1;
            v[u] = r >= 0 ? kc[(long long)r * d] : 0.f;
        }
#pragma unroll
        for (int u = 0; u < 8; u++) {
            int j = j0 + 4 * u;
            if (j < k)
#pragma unroll
                for (int m = 0; m < 8; m++) acc[m] = fmaf(P[m * k + j], v[u], acc[m]);
        }
    }
#pragma unroll
    for (int m = 0; m < 8; m++) red[ks][m][di] = acc[m];
    __syncthreads();
    for (int e = tid; e < 8 * 64; e += 256) {
        int m = e / 64, c = e % 64;
        float s = red[0][m][c] + red[1][m][c] + red[2][m][c] + red[3][m][c];
        int hi = g * 8 + m;
        o[((long long)ti * h + hi) * d + ds * 64 + c] = bf(s / den[ti * h + hi]);
    }
}
