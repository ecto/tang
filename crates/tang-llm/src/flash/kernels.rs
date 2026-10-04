//! The engine's own small CUDA kernels: what the Flash-Next window needs beyond the
//! `ComputeDevice` ops (`tang_compute::flash`). All take raw device addresses and read
//! per-window values from the `win` record, so they capture into the window's graph.

/// CUDA C, compiled by NVRTC at load (`CudaComputeDevice::custom_func`).
pub const SRC: &str = r#"
#define HIDDEN 2560
#define HC 4
#define PLE_RING 16

__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }

// Sum over a block of up to 1024 threads; every thread gets the total.
__device__ float block_sum(float v, float* sh) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) sh[w] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    float s = 0.f;
    for (int i = 0; i < nw; i++) s += sh[i];
    return s;
}

// r[t][c][d] = emb[t][d]: the token embedding copied into every hyper-connection stream.
extern "C" __global__ void fe_embed(float* r, const float* emb) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x, c = blockIdx.y, t = blockIdx.z;
    r[((size_t)t * HC + c) * HIDDEN + d] = emb[(size_t)t * HIDDEN + d];
}

// The n-gram (PLE) block for a window of t tokens, one block per stream c:
//   key = rmsnorm(key_raw[c]) * nk[c];  q = rmsnorm(r[c]) * nq[c];  s = <key, q> / sqrt(HIDDEN)
//   gate = sigmoid(sign(s) * sqrt(max(|s|, 1e-6)));  gated = val * gate
//   nrm = rmsnorm(gated) * nc[c] -> ring[pos % 16];  r[c] += gated + silu(conv over nrm at
//   pos - 9, pos - 6, pos - 3, pos (taps 0..3), zero before position 0)
// The ring is positional, so a rejected draft's rows are rewritten before anything reads them.
extern "C" __global__ void fe_ple(float* r, const float* key, const float* val, const float* nk,
                                  const float* nq, const float* nc, const float* conv,
                                  float* ring, const unsigned* win, int t, float eps) {
    __shared__ float sh[32];
    const int c = blockIdx.x;
    const unsigned pos0 = win[0];
    for (int tt = 0; tt < t; tt++) {
        const int pos = (int)pos0 + tt;
        const float* kr = key + (size_t)tt * HC * HIDDEN + c * HIDDEN;
        float* rr = r + ((size_t)tt * HC + c) * HIDDEN;
        const float* vv = val + (size_t)tt * HIDDEN;
        float sk = 0.f, sq = 0.f;
        for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) {
            sk += kr[i] * kr[i];
            sq += rr[i] * rr[i];
        }
        sk = block_sum(sk, sh);
        sq = block_sum(sq, sh);
        const float ik = rsqrtf(sk / HIDDEN + eps), iq = rsqrtf(sq / HIDDEN + eps);
        float d = 0.f;
        for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x)
            d += (kr[i] * ik * nk[c * HIDDEN + i]) * (rr[i] * iq * nq[c * HIDDEN + i]);
        d = block_sum(d, sh) / sqrtf((float)HIDDEN);
        const float sg = d > 0.f ? 1.f : (d < 0.f ? -1.f : 0.f);
        const float gate = sigmoid_f(sg * sqrtf(fmaxf(fabsf(d), 1e-6f)));
        float sgs = 0.f;
        for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) {
            const float g = vv[i] * gate;
            sgs += g * g;
        }
        sgs = block_sum(sgs, sh);
        const float ig = rsqrtf(sgs / HIDDEN + eps);
        float* slot = ring + (size_t)(pos % PLE_RING) * HC * HIDDEN + c * HIDDEN;
        for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) {
            const float g = vv[i] * gate;
            slot[i] = g * ig * nc[c * HIDDEN + i];
        }
        // Each thread reads back only the channels it wrote (same i stride), so no barrier.
        for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) {
            const int ch = c * HIDDEN + i;
            float acc = 0.f;
            for (int k = 0; k < 4; k++) {
                const int p = pos - (3 - k) * 3;
                if (p >= 0) acc += conv[k + 4 * ch] * ring[(size_t)(p % PLE_RING) * HC * HIDDEN + ch];
            }
            rr[i] += vv[i] * gate + silu_f(acc);
        }
        __syncthreads();
    }
}

// Shared expert middle: h = silu(gu[t][j]) * gu[t][ff + j] (gate rows first, then up rows),
// quantized per the QAct contract (32-wide chunks, d = amax/127, round half away, chunk-permuted
// codes, scales, sums) into xq = QAct { m: t, k: ff }. One warp per (chunk, token).
extern "C" __global__ void fe_silu_q(const float* gu, unsigned* xq, float* hf, int ff, int t) {
    const int ch = blockIdx.x, tt = blockIdx.y, lane = threadIdx.x;
    const int e = ch * 32 + lane;
    const float g = gu[(size_t)tt * 2 * ff + e], u = gu[(size_t)tt * 2 * ff + ff + e];
    const float v = silu_f(g) * u;
    hf[(size_t)tt * ff + e] = v;
    float amax = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = amax / 127.0f;
    int q = 0;
    if (d != 0.f) q = (int)fminf(fmaxf(roundf(v / d), -127.f), 127.f);
    int s = q;
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    // word 4h + f of the chunk holds elements 16h + 4b + f in byte b
    const int h = (lane >> 2) & 1, f = lane & 3;
    unsigned w = 0;
    for (int b = 0; b < 4; b++) {
        const int src = 16 * h + 4 * b + f;
        const unsigned qb = (unsigned)(unsigned char)(signed char)__shfl_sync(0xffffffffu, q, src);
        w |= qb << (8 * b);
    }
    const size_t m = t, k = ff;
    if (lane < 8) xq[(size_t)tt * k / 4 + ch * 8 + (4 * h + f)] = w;
    if (lane == 0) {
        xq[m * k / 4 + (size_t)tt * k / 32 + ch] = __float_as_uint(d);
        xq[m * k / 4 + m * k / 32 + (size_t)tt * k / 32 + ch] = (unsigned)s;
    }
}

// dst[i] = src[i] for n floats.
extern "C" __global__ void fe_copy(float* dst, const float* src, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// Scatter a [t][w] block into columns off..off + w of a [t][stride] buffer.
extern "C" __global__ void fe_scatter_cols(float* dst, int stride, int off, const float* src, int w) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x, tt = blockIdx.y;
    if (j < w) dst[(size_t)tt * stride + off + j] = src[(size_t)tt * w + j];
}

// Greedy argmax in two passes: fe_argmax1 grid (64, t) writes each slice's (value, index) to
// `part [t][64][2]`; fe_argmax2 (one block of 64 per token) reduces them. Ties go to the lower id.
__device__ __forceinline__ void amax_merge(float& v, int& i, float ov, int oi) {
    if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
}
// Philox4x32-10, first output word, for counter (c0, c1, 0, 0) and key (k0, 0).
__device__ __forceinline__ unsigned philox(unsigned c0, unsigned c1, unsigned k0) {
    unsigned x0 = c0, x1 = c1, x2 = 0, x3 = 0, k1 = 0;
    #pragma unroll
    for (int r = 0; r < 10; r++) {
        const unsigned lo0 = 0xD2511F53u * x0, hi0 = __umulhi(0xD2511F53u, x0);
        const unsigned lo1 = 0xCD9E8D57u * x2, hi1 = __umulhi(0xCD9E8D57u, x2);
        const unsigned y0 = hi1 ^ x1 ^ k0, y2 = hi0 ^ x3 ^ k1;
        x0 = y0; x1 = lo1; x2 = y2; x3 = lo0;
        k0 += 0x9E3779B9u; k1 += 0xBB67AE85u;
    }
    return x0;
}

// Sampling is the Gumbel-max trick on the window's logits: token = argmax_i (logit_i / T + g_i),
// g_i = -log(-log(u_i)), u_i from Philox(seed, position, i). A function of (seed, position) and
// the logits only, so drafts never change what is sampled. T = 0: plain argmax.
extern "C" __global__ void fe_argmax1(const float* logits, int n, float* part, const unsigned* ctl) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    const float* l = logits + (size_t)blockIdx.y * n;
    const int per = (n + gridDim.x - 1) / gridDim.x, lo = blockIdx.x * per, hi = min(n, lo + per);
    float v = __int_as_float(0xff800000);
    int idx = 0x7fffffff;
    const float temp = __uint_as_float(ctl[4]);
    const unsigned pos = ctl[0] + blockIdx.y, seed = ctl[3];
    if (temp > 0.f) {
        const float it = 1.0f / temp;
        for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
            const float u = ((float)(philox((unsigned)i, pos, seed) >> 8) + 0.5f) * 5.9604644775390625e-8f;
            amax_merge(v, idx, l[i] * it - logf(-logf(u)), i);
        }
    } else {
        for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) amax_merge(v, idx, l[i], i);
    }
    for (int o = 16; o > 0; o >>= 1) amax_merge(v, idx, __shfl_xor_sync(0xffffffffu, v, o), __shfl_xor_sync(0xffffffffu, idx, o));
    const int w = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) { bv[w] = v; bi[w] = idx; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 1; i < (int)(blockDim.x >> 5); i++) amax_merge(v, idx, bv[i], bi[i]);
        float* p = part + ((size_t)blockIdx.y * gridDim.x + blockIdx.x) * 2;
        p[0] = v;
        p[1] = __int_as_float(idx);
    }
}
extern "C" __global__ void fe_argmax2(const float* part, int nparts, unsigned* ids) {
    const float* p = part + (size_t)blockIdx.x * nparts * 2;
    float v = __int_as_float(0xff800000);
    int idx = 0x7fffffff;
    for (int i = threadIdx.x; i < nparts; i += 32) amax_merge(v, idx, p[2 * i], __float_as_int(p[2 * i + 1]));
    for (int o = 16; o > 0; o >>= 1) amax_merge(v, idx, __shfl_xor_sync(0xffffffffu, v, o), __shfl_xor_sync(0xffffffffu, idx, o));
    if (threadIdx.x == 0) ids[blockIdx.x] = (unsigned)idx;
}
"#;

/// Native GGUF-type GEMV with f32 activations: `out[t][off + r] = W_seg[r] · x[t]` for every
/// row of every segment (a stacked projection whose parts have different GGUF types), one warp
/// per row, each lane decoding 8 consecutive weights per 256-wide step. Types: BF16, Q4_0, Q5_0,
/// Q8_0, Q2_0, IQ4_NL, IQ4_XS, Q3_K, Q4_K, Q5_K, Q6_K (the dequantizers of `gguf.rs`, which are
/// bit-exact against gguf-py). Lossless with respect to the file; the only rounding is the f32
/// accumulation.
pub const GEMV_SRC: &str = r#"
struct Seg { unsigned long long w; int type; int rows; int row_bytes; int out_off; };

// Small int -> float without I2F (quarter rate on sm_86): magic-number add.
__device__ __forceinline__ float i2f(int q) { return __int_as_float(0x4B400000 + q) - 12582912.0f; }

// IQ4_NL grid value for a nibble, from registers (a __constant__ table serializes when lanes
// index it differently).
__device__ __forceinline__ float iq4(int q) {
    const unsigned t0 = 0xBFAD9881u, t1 = 0xF6EADDCFu, t2 = 0x26190D01u, t3 = 0x71594535u;
    const unsigned w = (q & 8) ? ((q & 4) ? t3 : t2) : ((q & 4) ? t1 : t0);
    return i2f((int)(signed char)(w >> (8 * (q & 3))));
}

__device__ __forceinline__ float h2f(unsigned short h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}
__device__ __forceinline__ unsigned short rd16(const unsigned char* p) {
    return (unsigned short)p[0] | ((unsigned short)p[1] << 8);
}
// 8 consecutive bytes at any alignment (shared memory: three aligned 32-bit loads, two funnel
// shifts).
__device__ __forceinline__ unsigned long long rd64(const unsigned char* p) {
    const unsigned long long a = (unsigned long long)p;
    const unsigned* q = (const unsigned*)(a & ~3ull);
    const unsigned sh = (unsigned)(a & 3) * 8;
    const unsigned w0 = q[0], w1 = q[1], w2 = q[2];
    const unsigned lo = __funnelshift_r(w0, w1, sh), hi = __funnelshift_r(w1, w2, sh);
    return (unsigned long long)lo | ((unsigned long long)hi << 32);
}
__device__ __forceinline__ void scale_min_k4(int j, const unsigned char* q, int& s, int& m) {
    if (j < 4) { s = q[j] & 63; m = q[j + 4] & 63; }
    else { s = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

// The 8 weights e0..e0 + 8 of a row (e0 % 8 == 0). Shift-based selects, no lane-dependent
// branches (lanes of a warp sit in different quarters of a block).
template <int TY>
__device__ __forceinline__ void decode8(const unsigned char* row, int e0, float* v) {
    if (TY == 30) {  // BF16
        const uint4 u = *(const uint4*)(row + 2 * e0);
        const unsigned w[4] = {u.x, u.y, u.z, u.w};
        #pragma unroll
        for (int i = 0; i < 4; i++) { v[2 * i] = __uint_as_float(w[i] << 16); v[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u); }
    } else if (TY == 8) {  // Q8_0
        const unsigned char* b = row + (e0 / 32) * 34;
        const float d = h2f(rd16(b));
        const unsigned long long q = rd64(b + 2 + e0 % 32);
        #pragma unroll
        for (int i = 0; i < 8; i++) v[i] = d * i2f((int)(signed char)(q >> (8 * i)));
    } else if (TY == 2 || TY == 20) {  // Q4_0, IQ4_NL
        const unsigned char* b = row + (e0 / 32) * 18;
        const float d = h2f(rd16(b));
        const int j = e0 % 32, sh = (j >> 4) * 4;
        const unsigned long long q = rd64(b + 2 + (j & 15));
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int n = (int)(q >> (8 * i + sh)) & 0xf;
            v[i] = TY == 2 ? d * i2f(n - 8) : d * iq4(n);
        }
    } else if (TY == 6) {  // Q5_0
        const unsigned char* b = row + (e0 / 32) * 22;
        const float d = h2f(rd16(b));
        const unsigned qh = (unsigned)b[2] | ((unsigned)b[3] << 8) | ((unsigned)b[4] << 16) | ((unsigned)b[5] << 24);
        const int j0 = e0 % 32, hi = j0 >> 4, jl = j0 & 15;
        const unsigned long long q = rd64(b + 6 + jl);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int n = ((int)(q >> (8 * i + 4 * hi)) & 0xf) | (((qh >> (j0 + i)) & 1) << 4);
            v[i] = d * i2f(n - 16);
        }
    } else if (TY == 42) {  // Q2_0: 64 in 18 B, (code - 1) * d
        const unsigned char* b = row + (e0 / 64) * 18;
        const float d = h2f(rd16(b));
        const int j = e0 % 64;
        const unsigned q = (unsigned)b[2 + j / 4] | ((unsigned)b[3 + j / 4] << 8);
        #pragma unroll
        for (int i = 0; i < 8; i++) v[i] = d * i2f((int)((q >> (2 * i)) & 3) - 1);
    } else if (TY == 23) {  // IQ4_XS
        const unsigned char* b = row + (e0 / 256) * 136;
        const int e = e0 % 256, ib = e / 32, j = e % 32, sh = (j >> 4) * 4;
        const float d = h2f(rd16(b));
        const unsigned shh = rd16(b + 2);
        const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) | (((shh >> (2 * ib)) & 3) << 4);
        const float dl = d * i2f(ls - 32);
        const unsigned long long q = rd64(b + 8 + ib * 16 + (j & 15));
        #pragma unroll
        for (int i = 0; i < 8; i++) v[i] = dl * iq4((int)(q >> (8 * i + sh)) & 0xf);
    } else if (TY == 11) {  // Q3_K: hmask[32] qs[64] scales[12] d
        const unsigned char* b = row + (e0 / 256) * 110;
        const int e = e0 % 256, n = e / 128, jj = (e % 128) / 32, l = e % 32;
        const int is = n * 8 + jj * 2 + l / 16;
        const unsigned char* sc = b + 96;
        const int lo = (sc[is & 7] >> (4 * (is >> 3))) & 0xF;
        const int hi = (sc[8 + is % 4] >> (2 * (is / 4))) & 3;
        const float dl = h2f(rd16(b + 108)) * i2f((lo | (hi << 4)) - 32);
        const int m = n * 4 + jj;
        const unsigned long long q = rd64(b + 32 + n * 32 + l);
        const unsigned long long hm = rd64(b + l);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int qv = (int)((q >> (8 * i + 2 * jj)) & 3) - (((hm >> (8 * i + m)) & 1) ? 0 : 4);
            v[i] = dl * i2f(qv);
        }
    } else if (TY == 12 || TY == 13) {  // Q4_K, Q5_K
        const int bb = TY == 12 ? 144 : 176;
        const unsigned char* b = row + (e0 / 256) * bb;
        const float d = h2f(rd16(b)), dmin = h2f(rd16(b + 2));
        const int e = e0 % 256, j = e / 64, w = e % 64, l = w % 32, up = w >> 5;
        int s, m;
        scale_min_k4(2 * j + up, b + 4, s, m);
        const float d1 = d * i2f(s), m1 = dmin * i2f(m);
        const unsigned long long q = rd64(b + (TY == 12 ? 16 : 48) + j * 32 + l);
        const unsigned long long qh = TY == 13 ? rd64(b + 16 + l) : 0ull;
        const int hb = 2 * j + up;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int qv = (int)(q >> (8 * i + 4 * up)) & 0xf;
            if (TY == 13) qv |= (int)((qh >> (8 * i + hb)) & 1) << 4;
            v[i] = fmaf(d1, i2f(qv), -m1);
        }
    } else if (TY == 14) {  // Q6_K: ql[128] qh[64] scales[16] d
        const unsigned char* b = row + (e0 / 256) * 210;
        const float d = h2f(rd16(b + 208));
        const int e = e0 % 256, n = e / 128, r = e % 128, qt = r / 32, l = r % 32;
        const float dl = d * i2f((int)(signed char)b[192 + n * 8 + l / 16 + 2 * qt]);
        const unsigned long long ql = rd64(b + n * 64 + l + 32 * (qt & 1));
        const unsigned long long qh = rd64(b + 128 + n * 32 + l);
        const int ls = 4 * (qt >> 1), hs = 2 * qt;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int qv = ((int)(ql >> (8 * i + ls)) & 0xf) | (((int)(qh >> (8 * i + hs)) & 3) << 4);
            v[i] = dl * i2f(qv - 32);
        }
    }
}

template <int T, int TY>
__device__ __forceinline__ void rows_dot(const unsigned char* w, const float* x, int k, float* acc) {
    const int lane = threadIdx.x & 31;
    for (int e0 = lane * 8; e0 < k; e0 += 256) {
        float v[8];
        decode8<TY>(w, e0, v);
        #pragma unroll
        for (int t = 0; t < T; t++) {
            const float4 a = *(const float4*)(x + (size_t)t * k + e0);
            const float4 c = *(const float4*)(x + (size_t)t * k + e0 + 4);
            float s = acc[t];
            s = fmaf(v[0], a.x, s); s = fmaf(v[1], a.y, s); s = fmaf(v[2], a.z, s); s = fmaf(v[3], a.w, s);
            s = fmaf(v[4], c.x, s); s = fmaf(v[5], c.y, s); s = fmaf(v[6], c.z, s); s = fmaf(v[7], c.w, s);
            acc[t] = s;
        }
    }
}

// ---- int8-activation path (dp4a) ----

#define XQ8_T 8

// IQ4_NL grid for four nibbles (bytes 0..15) at once.
__device__ __forceinline__ int iq4w(unsigned n) {
    const unsigned sel = (n & 0x7u) | ((n >> 4) & 0x70u) | ((n >> 8) & 0x700u) | ((n >> 12) & 0x7000u);
    const unsigned lo = __byte_perm(0xBFAD9881u, 0xF6EADDCFu, sel);
    const unsigned hi = __byte_perm(0x26190D01u, 0x71594535u, sel);
    const unsigned m = ((n >> 3) & 0x01010101u) * 0xFFu;
    return (int)((lo & ~m) | (hi & m));
}
// Four 1-bit fields (bits 0..3 of b) into bit 0 of four bytes.
__device__ __forceinline__ unsigned spread1(unsigned b) {
    return (b & 1u) | ((b & 2u) << 7) | ((b & 4u) << 14) | ((b & 8u) << 21);
}
// Four 2-bit fields (bits 0..7 of c) into four bytes.
__device__ __forceinline__ unsigned spread2(unsigned c) {
    return (c & 3u) | ((c & 0xCu) << 6) | ((c & 0x30u) << 12) | ((c & 0xC0u) << 18);
}

// The 8 weights e0..e0 + 8 as signed int8 lanes of (w0, w1), with weight = dw · code − mw.
template <int TY>
__device__ __forceinline__ void decode8i(const unsigned char* row, int e0, int& w0, int& w1, float& dw, float& mw) {
    mw = 0.f;
    if (TY == 8) {
        const unsigned char* b = row + (e0 / 32) * 34;
        dw = h2f(rd16(b));
        const unsigned long long q = rd64(b + 2 + e0 % 32);
        w0 = (int)(unsigned)q; w1 = (int)(unsigned)(q >> 32);
    } else if (TY == 2 || TY == 20) {
        const unsigned char* b = row + (e0 / 32) * 18;
        dw = h2f(rd16(b));
        const int j = e0 % 32, sh = (j >> 4) * 4;
        const unsigned long long q = rd64(b + 2 + (j & 15));
        const unsigned n0 = ((unsigned)q >> sh) & 0x0F0F0F0Fu, n1 = ((unsigned)(q >> 32) >> sh) & 0x0F0F0F0Fu;
        if (TY == 2) { w0 = (int)__vsub4(n0, 0x08080808u); w1 = (int)__vsub4(n1, 0x08080808u); }
        else { w0 = iq4w(n0); w1 = iq4w(n1); }
    } else if (TY == 6) {
        const unsigned char* b = row + (e0 / 32) * 22;
        dw = h2f(rd16(b));
        const unsigned qh = (unsigned)b[2] | ((unsigned)b[3] << 8) | ((unsigned)b[4] << 16) | ((unsigned)b[5] << 24);
        const int j0 = e0 % 32, sh = (j0 >> 4) * 4;
        const unsigned long long q = rd64(b + 6 + (j0 & 15));
        const unsigned h = qh >> j0;
        const unsigned n0 = (((unsigned)q >> sh) & 0x0F0F0F0Fu) | (spread1(h & 0xF) << 4);
        const unsigned n1 = (((unsigned)(q >> 32) >> sh) & 0x0F0F0F0Fu) | (spread1((h >> 4) & 0xF) << 4);
        w0 = (int)__vsub4(n0, 0x10101010u); w1 = (int)__vsub4(n1, 0x10101010u);
    } else if (TY == 42) {
        const unsigned char* b = row + (e0 / 64) * 18;
        dw = h2f(rd16(b));
        const int j = e0 % 64;
        const unsigned q = (unsigned)b[2 + j / 4] | ((unsigned)b[3 + j / 4] << 8);
        w0 = (int)__vsub4(spread2(q & 0xFF), 0x01010101u);
        w1 = (int)__vsub4(spread2(q >> 8), 0x01010101u);
    } else if (TY == 23) {
        const unsigned char* b = row + (e0 / 256) * 136;
        const int e = e0 % 256, ib = e / 32, j = e % 32, sh = (j >> 4) * 4;
        const unsigned shh = rd16(b + 2);
        const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) | (((shh >> (2 * ib)) & 3) << 4);
        dw = h2f(rd16(b)) * i2f(ls - 32);
        const unsigned long long q = rd64(b + 8 + ib * 16 + (j & 15));
        w0 = iq4w(((unsigned)q >> sh) & 0x0F0F0F0Fu);
        w1 = iq4w(((unsigned)(q >> 32) >> sh) & 0x0F0F0F0Fu);
    } else if (TY == 11) {
        const unsigned char* b = row + (e0 / 256) * 110;
        const int e = e0 % 256, n = e / 128, jj = (e % 128) / 32, l = e % 32;
        const int is = n * 8 + jj * 2 + l / 16;
        const unsigned char* sc = b + 96;
        const int lo = (sc[is & 7] >> (4 * (is >> 3))) & 0xF;
        const int hi = (sc[8 + is % 4] >> (2 * (is / 4))) & 3;
        dw = h2f(rd16(b + 108)) * i2f((lo | (hi << 4)) - 32);
        const int m = n * 4 + jj;
        const unsigned long long q = rd64(b + 32 + n * 32 + l);
        const unsigned long long hm = rd64(b + l);
        const unsigned c0 = ((unsigned)q >> (2 * jj)) & 0x03030303u, c1 = ((unsigned)(q >> 32) >> (2 * jj)) & 0x03030303u;
        const unsigned h0 = ((unsigned)hm >> m) & 0x01010101u, h1 = ((unsigned)(hm >> 32) >> m) & 0x01010101u;
        w0 = (int)__vsub4(c0, (h0 ^ 0x01010101u) << 2);
        w1 = (int)__vsub4(c1, (h1 ^ 0x01010101u) << 2);
    } else if (TY == 12 || TY == 13) {
        const int bb = TY == 12 ? 144 : 176;
        const unsigned char* b = row + (e0 / 256) * bb;
        const int e = e0 % 256, j = e / 64, w = e % 64, l = w % 32, up = w >> 5;
        int s, m;
        scale_min_k4(2 * j + up, b + 4, s, m);
        dw = h2f(rd16(b)) * i2f(s);
        mw = h2f(rd16(b + 2)) * i2f(m);
        const unsigned long long q = rd64(b + (TY == 12 ? 16 : 48) + j * 32 + l);
        unsigned n0 = ((unsigned)q >> (4 * up)) & 0x0F0F0F0Fu, n1 = ((unsigned)(q >> 32) >> (4 * up)) & 0x0F0F0F0Fu;
        if (TY == 13) {
            const unsigned long long qh = rd64(b + 16 + l);
            const int hb = 2 * j + up;
            n0 |= (((unsigned)qh >> hb) & 0x01010101u) << 4;
            n1 |= (((unsigned)(qh >> 32) >> hb) & 0x01010101u) << 4;
        }
        w0 = (int)n0; w1 = (int)n1;
    } else if (TY == 14) {
        const unsigned char* b = row + (e0 / 256) * 210;
        const int e = e0 % 256, n = e / 128, r = e % 128, qt = r / 32, l = r % 32;
        dw = h2f(rd16(b + 208)) * i2f((int)(signed char)b[192 + n * 8 + l / 16 + 2 * qt]);
        const unsigned long long ql = rd64(b + n * 64 + l + 32 * (qt & 1));
        const unsigned long long qh = rd64(b + 128 + n * 32 + l);
        const int ls = 4 * (qt >> 1), hs = 2 * qt;
        const unsigned n0 = (((unsigned)ql >> ls) & 0x0F0F0F0Fu) | ((((unsigned)qh >> hs) & 0x03030303u) << 4);
        const unsigned n1 = (((unsigned)(ql >> 32) >> ls) & 0x0F0F0F0Fu) | ((((unsigned)(qh >> 32) >> hs) & 0x03030303u) << 4);
        w0 = (int)__vsub4(n0, 0x20202020u); w1 = (int)__vsub4(n1, 0x20202020u);
    }
}

// xq: int8 codes [XQ8_T][k] (plain order), then f32 scales [XQ8_T][k / 32].
template <int T, int TY>
__device__ __forceinline__ void rows_dot8(const unsigned char* w, const unsigned char* xq, int k, float* acc) {
    const int lane = threadIdx.x & 31;
    const float* xd = (const float*)(xq + (size_t)XQ8_T * k);
    for (int e0 = lane * 8; e0 < k; e0 += 256) {
        int w0, w1;
        float dw, mw;
        decode8i<TY>(w, e0, w0, w1, dw, mw);
        #pragma unroll
        for (int t = 0; t < T; t++) {
            const uint2 xv = *(const uint2*)(xq + (size_t)t * k + e0);
            const float dx = xd[t * (k / 32) + e0 / 32];
            const int s = __dp4a(w0, (int)xv.x, __dp4a(w1, (int)xv.y, 0));
            if (TY == 12 || TY == 13) {
                const int sx = __dp4a(0x01010101, (int)xv.x, __dp4a(0x01010101, (int)xv.y, 0));
                acc[t] = fmaf(dx, fmaf(dw, i2f(s), -mw * i2f(sx)), acc[t]);
            } else {
                acc[t] = fmaf(dx * dw, i2f(s), acc[t]);
            }
        }
    }
}

// x [t][k] f32 -> xq (int8 plain codes + per-32 scales; d = amax/127, round half away, as QAct).
extern "C" __global__ void fe_q8(const float* x, unsigned char* xq, int k) {
    const int c = blockIdx.x, t = blockIdx.y, lane = threadIdx.x;
    const float v = x[(size_t)t * k + c * 32 + lane];
    float amax = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = amax / 127.0f;
    const int q = d == 0.f ? 0 : (int)fminf(fmaxf(roundf(v / d), -127.f), 127.f);
    xq[(size_t)t * k + c * 32 + lane] = (unsigned char)(signed char)q;
    if (lane == 0) ((float*)(xq + (size_t)XQ8_T * k))[t * (k / 32) + c] = d;
}

// Two rows per warp (both staged in shared memory before either is decoded, so twice the bytes
// are in flight), four warps per block; int8 activations for quantized segments, f32 for BF16.
template <int T>
__device__ void gemv8_body(const Seg* segs, int nseg, const float* x, const unsigned char* xq, int k, float* out, int ostride, int row16) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    extern __shared__ uint4 dsm[];
    const unsigned char* w[2];
    Seg sg[2];
    int rr[2];
    bool ok[2];
    #pragma unroll
    for (int j = 0; j < 2; j++) {
        int row = (blockIdx.x * 4 + warp) * 2 + j, s = 0;
        while (s < nseg && row >= segs[s].rows) { row -= segs[s].rows; s++; }
        ok[j] = s < nseg;
        rr[j] = row;
        w[j] = 0;
        if (!ok[j]) continue;
        sg[j] = segs[s];
        const unsigned char* src = (const unsigned char*)sg[j].w + (size_t)row * sg[j].row_bytes;
        w[j] = src;
        if (sg[j].type != 30) {
            uint4* sm = dsm + (warp * 2 + j) * row16;
            const unsigned long long a = (unsigned long long)src & ~15ull;
            const int delta = (int)((unsigned long long)src - a);
            const int n16 = (delta + sg[j].row_bytes + 15) >> 4;
            for (int i = lane; i < n16; i += 32) sm[i] = __ldg((const uint4*)a + i);
            w[j] = (const unsigned char*)sm + delta;
        }
    }
    __syncwarp();
    #pragma unroll
    for (int j = 0; j < 2; j++) {
        if (!ok[j]) continue;
        float acc[T];
        #pragma unroll
        for (int t = 0; t < T; t++) acc[t] = 0.f;
        switch (sg[j].type) {
            case 30: rows_dot<T, 30>(w[j], x, k, acc); break;
            case 8: rows_dot8<T, 8>(w[j], xq, k, acc); break;
            case 2: rows_dot8<T, 2>(w[j], xq, k, acc); break;
            case 20: rows_dot8<T, 20>(w[j], xq, k, acc); break;
            case 6: rows_dot8<T, 6>(w[j], xq, k, acc); break;
            case 42: rows_dot8<T, 42>(w[j], xq, k, acc); break;
            case 23: rows_dot8<T, 23>(w[j], xq, k, acc); break;
            case 11: rows_dot8<T, 11>(w[j], xq, k, acc); break;
            case 12: rows_dot8<T, 12>(w[j], xq, k, acc); break;
            case 13: rows_dot8<T, 13>(w[j], xq, k, acc); break;
            case 14: rows_dot8<T, 14>(w[j], xq, k, acc); break;
            default: break;
        }
        #pragma unroll
        for (int t = 0; t < T; t++) {
            float a = acc[t];
            for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
            if (lane == 0) out[(size_t)t * ostride + sg[j].out_off + rr[j]] = a;
        }
    }
}


#define GEMV8_T(T) extern "C" __global__ void __launch_bounds__(128) fe_gemv8_t##T( \
    const Seg* segs, int nseg, const float* x, const unsigned char* xq, int k, float* out, int ostride, int row16) { \
    gemv8_body<T>(segs, nseg, x, xq, k, out, ostride, row16); }
GEMV8_T(1) GEMV8_T(2) GEMV8_T(3) GEMV8_T(4) GEMV8_T(5) GEMV8_T(6) GEMV8_T(7) GEMV8_T(8)

__device__ __forceinline__ void cp16(void* dst, const void* src) {
    const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(d), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_wait1() { asm volatile("cp.async.wait_group 1;" ::: "memory"); }

// Row `row` of the stacked weight: its segment and byte address.
__device__ __forceinline__ const Seg* find_seg(const Seg* segs, int nseg, int& row) {
    int s = 0;
    while (s < nseg && row >= segs[s].rows) { row -= segs[s].rows; s++; }
    return s < nseg ? segs + s : 0;
}

// Async-stage one row (aligned 16-byte cp.async) into `sm`; returns the offset of its first
// byte there. BF16 rows aren't staged (read straight from global).
__device__ __forceinline__ int stage_row(const Seg* sg, int row, uint4* sm, int lane) {
    if (!sg || sg->type == 30) return 0;
    const unsigned char* src = (const unsigned char*)sg->w + (size_t)row * sg->row_bytes;
    const unsigned long long a = (unsigned long long)src & ~15ull;
    const int delta = (int)((unsigned long long)src - a);
    const int n16 = (delta + sg->row_bytes + 15) >> 4;
    for (int i = lane; i < n16; i += 32) cp16(sm + i, (const uint4*)a + i);
    return delta;
}

// Each warp walks rows warp_id, warp_id + n_warps, ...; the next row's bytes stream into the
// other half of its double buffer (cp.async) while it decodes the current one.
template <int T>
__device__ void gemv_body(const Seg* segs, int nseg, const float* x, int k, float* out, int ostride, int row16, int total) {
    extern __shared__ uint4 dsm[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nw = gridDim.x * (blockDim.x >> 5);
    uint4* buf[2] = {dsm + (2 * warp) * row16, dsm + (2 * warp + 1) * row16};
    int row = blockIdx.x * (blockDim.x >> 5) + warp;
    int r0 = row;
    const Seg* sg = row < total ? find_seg(segs, nseg, r0) : 0;
    int delta = stage_row(sg, r0, buf[0], lane);
    cp_commit();
    for (int it = 0; row < total; it++, row += nw) {
        const int nrow = row + nw;
        int r1 = nrow;
        const Seg* ns = nrow < total ? find_seg(segs, nseg, r1) : 0;
        const int ndelta = stage_row(ns, r1, buf[(it + 1) & 1], lane);
        cp_commit();
        cp_wait1();
        __syncwarp();
        const Seg cur = *sg;
        const unsigned char* w = cur.type == 30
            ? (const unsigned char*)cur.w + (size_t)r0 * cur.row_bytes
            : (const unsigned char*)buf[it & 1] + delta;
        float acc[T];
        #pragma unroll
        for (int t = 0; t < T; t++) acc[t] = 0.f;
        switch (cur.type) {
            case 30: rows_dot<T, 30>(w, x, k, acc); break;
            case 8: rows_dot<T, 8>(w, x, k, acc); break;
            case 2: rows_dot<T, 2>(w, x, k, acc); break;
            case 20: rows_dot<T, 20>(w, x, k, acc); break;
            case 6: rows_dot<T, 6>(w, x, k, acc); break;
            case 42: rows_dot<T, 42>(w, x, k, acc); break;
            case 23: rows_dot<T, 23>(w, x, k, acc); break;
            case 11: rows_dot<T, 11>(w, x, k, acc); break;
            case 12: rows_dot<T, 12>(w, x, k, acc); break;
            case 13: rows_dot<T, 13>(w, x, k, acc); break;
            case 14: rows_dot<T, 14>(w, x, k, acc); break;
            default: break;
        }
        #pragma unroll
        for (int t = 0; t < T; t++) {
            float v = acc[t];
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0) out[(size_t)t * ostride + cur.out_off + r0] = v;
        }
        __syncwarp();  // everyone is done with buf[it & 1] before it is refilled
        sg = ns;
        r0 = r1;
        delta = ndelta;
    }
}

#define GEMV_T(T) extern "C" __global__ void __launch_bounds__(128) fe_gemv_t##T( \
    const Seg* segs, int nseg, const float* x, int k, float* out, int ostride, int row16, int total) { \
    gemv_body<T>(segs, nseg, x, k, out, ostride, row16, total); }
GEMV_T(1) GEMV_T(2) GEMV_T(3) GEMV_T(4) GEMV_T(5) GEMV_T(6) GEMV_T(7) GEMV_T(8)
"#;
