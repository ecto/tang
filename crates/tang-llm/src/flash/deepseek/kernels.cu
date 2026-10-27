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

// The router for one token per thread: raw [t][ne] gate logits (f32) -> ids/weights [t][k].
// sqrt(softplus) scores, top-k of score + bias (ties to the lower id), weights normalized and
// scaled. k <= 16.
extern "C" __global__ void ds_router(const float* raw, const float* bias, int* ids, float* wts, int t, int ne, int k,
                                     int norm, float route_scale) {
    int ti = blockIdx.x * blockDim.x + threadIdx.x;
    if (ti >= t) return;
    const float* r = raw + (long long)ti * ne;
    int sel[16];
    for (int j = 0; j < k; j++) {
        int best = -1;
        float bv = 0.f;
        for (int e = 0; e < ne; e++) {
            bool used = false;
            for (int q = 0; q < j; q++) used |= sel[q] == e;
            if (used) continue;
            float x = r[e];
            float sc = sqrtf(x > 20.f ? x : log1pf(expf(x)));
            float v = sc + bias[e];
            if (best < 0 || v > bv) { best = e; bv = v; }
        }
        sel[j] = best;
    }
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
                                         int round, int qdq, const float* __restrict__ rw, float lim) {
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
                        float xv = x[(long long)ti * cols + k];
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
        lin_fast<F, T, G>(w, w3, x, y, t, round, qdq, rw, lim);                                               \
    }
#define DS_LIN_T(F, G) DS_LIN(F, 1, G) DS_LIN(F, 2, G) DS_LIN(F, 4, G) DS_LIN(F, 8, G)
DS_LIN_T(2, 0)
DS_LIN_T(3, 0)
DS_LIN_T(2, 1)
DS_LIN_T(3, 1)
