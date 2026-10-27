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
    for (int k = lane; k < w.cols; k += 32) s += xr[k] * wget(w, n, k);
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0) y[(long long)ti * w.rows + n] = round ? bf(s) : s;
}

// In place: fp8 quant-dequant per 32 with power-of-two scales (one thread per group).
extern "C" __global__ void ds_fp8_qdq(float* v, int n_groups) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_groups) return;
    float* p = v + (long long)g * 32;
    float amax = 0.f;
    for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(p[i]));
    float s = pow2_ceil(fmaxf(amax, 1e-4f) / 448.f);
    for (int i = 0; i < 32; i++) p[i] = bf(round_e4m3(fminf(fmaxf(p[i] / s, -448.f), 448.f)) * s);
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
    float rs = rsqrtf(red[0] / n + eps);
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
extern "C" __global__ void ds_sinkhorn(const float* mixes, const float* scale, const float* base,
                                       float* pre, float* post, float* comb, int t, int hc, int iters, float eps) {
    int ti = blockIdx.x * blockDim.x + threadIdx.x;
    if (ti >= t) return;
    int mix = (2 + hc) * hc;
    const float* m = mixes + (long long)ti * mix;
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
