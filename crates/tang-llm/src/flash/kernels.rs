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

// Greedy: ids[t] = argmax_j logits[t][j] (ties to the lower id). One block per token.
extern "C" __global__ void fe_argmax(const float* logits, int n, unsigned* ids) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    const float* l = logits + (size_t)blockIdx.x * n;
    float v = __int_as_float(0xff800000);
    int idx = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float x = l[i];
        if (x > v) { v = x; idx = i; }
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, o);
        const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
        if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
    }
    const int w = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) { bv[w] = v; bi[w] = idx; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 1; i < (int)(blockDim.x >> 5); i++)
            if (bv[i] > v || (bv[i] == v && bi[i] < idx)) { v = bv[i]; idx = bi[i]; }
        ids[blockIdx.x] = (unsigned)idx;
    }
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

__constant__ signed char IQ4NL[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

__device__ __forceinline__ float h2f(unsigned short h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}
__device__ __forceinline__ unsigned short rd16(const unsigned char* p) {
    return (unsigned short)p[0] | ((unsigned short)p[1] << 8);
}
__device__ __forceinline__ void scale_min_k4(int j, const unsigned char* q, int& s, int& m) {
    if (j < 4) { s = q[j] & 63; m = q[j + 4] & 63; }
    else { s = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

// The 8 weights e0..e0 + 8 of a row (e0 % 8 == 0).
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
        const int j = e0 % 32;
        #pragma unroll
        for (int i = 0; i < 8; i++) v[i] = d * (float)(signed char)b[2 + j + i];
    } else if (TY == 2 || TY == 20) {  // Q4_0, IQ4_NL
        const unsigned char* b = row + (e0 / 32) * 18;
        const float d = h2f(rd16(b));
        const int j = e0 % 32;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int q = j < 16 ? (b[2 + j + i] & 0xf) : (b[2 + j - 16 + i] >> 4);
            v[i] = TY == 2 ? d * (float)(q - 8) : d * (float)IQ4NL[q];
        }
    } else if (TY == 6) {  // Q5_0
        const unsigned char* b = row + (e0 / 32) * 22;
        const float d = h2f(rd16(b));
        const unsigned qh = (unsigned)b[2] | ((unsigned)b[3] << 8) | ((unsigned)b[4] << 16) | ((unsigned)b[5] << 24);
        const int j0 = e0 % 32;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int q;
            if (j0 < 16) { const int j = j0 + i; q = (b[6 + j] & 0xf) | (((qh >> j) << 4) & 0x10); }
            else { const int j = j0 - 16 + i; q = (b[6 + j] >> 4) | ((qh >> (j + 12)) & 0x10); }
            v[i] = d * (float)(q - 16);
        }
    } else if (TY == 42) {  // Q2_0: 64 in 18 B, (code - 1) * d
        const unsigned char* b = row + (e0 / 64) * 18;
        const float d = h2f(rd16(b));
        const int j = e0 % 64;
        #pragma unroll
        for (int i = 0; i < 8; i++) v[i] = d * (float)((int)((b[2 + (j + i) / 4] >> (2 * ((j + i) % 4))) & 3) - 1);
    } else if (TY == 23) {  // IQ4_XS
        const unsigned char* b = row + (e0 / 256) * 136;
        const int e = e0 % 256, ib = e / 32, j = e % 32;
        const float d = h2f(rd16(b));
        const unsigned sh = rd16(b + 2);
        const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4);
        const float dl = d * (float)(ls - 32);
        const unsigned char* q = b + 8 + ib * 16;
        #pragma unroll
        for (int i = 0; i < 8; i++) v[i] = dl * (float)IQ4NL[j < 16 ? (q[j + i] & 0xf) : (q[j - 16 + i] >> 4)];
    } else if (TY == 11) {  // Q3_K: hmask[32] qs[64] scales[12] d
        const unsigned char* b = row + (e0 / 256) * 110;
        const int e = e0 % 256, n = e / 128, jj = (e % 128) / 32, l = e % 32;
        const int is = n * 8 + jj * 2 + l / 16;
        const unsigned char* sc = b + 96;
        const int lo = is < 8 ? (sc[is] & 0xF) : (sc[is - 8] >> 4);
        const int hi = (sc[8 + is % 4] >> (2 * (is / 4))) & 3;
        const float dl = h2f(rd16(b + 108)) * (float)((lo | (hi << 4)) - 32);
        const unsigned char m = (unsigned char)(1 << (n * 4 + jj));
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int li = l + i;
            const int q = (int)((b[32 + n * 32 + li] >> (2 * jj)) & 3) - ((b[li] & m) ? 0 : 4);
            v[i] = dl * (float)q;
        }
    } else if (TY == 12 || TY == 13) {  // Q4_K, Q5_K
        const int bb = TY == 12 ? 144 : 176;
        const unsigned char* b = row + (e0 / 256) * bb;
        const float d = h2f(rd16(b)), dmin = h2f(rd16(b + 2));
        const int e = e0 % 256, j = e / 64, w = e % 64, l = w % 32;
        int s, m;
        scale_min_k4(2 * j + (w >= 32), b + 4, s, m);
        const float d1 = d * (float)s, m1 = dmin * (float)m;
        const unsigned char* q = b + (TY == 12 ? 16 : 48) + j * 32;
        const unsigned char u = (unsigned char)((w < 32 ? 1 : 2) << (2 * j));
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int qv = w < 32 ? (q[l + i] & 0xf) : (q[l + i] >> 4);
            if (TY == 13 && (b[16 + l + i] & u)) qv += 16;
            v[i] = d1 * (float)qv - m1;
        }
    } else if (TY == 14) {  // Q6_K: ql[128] qh[64] scales[16] d
        const unsigned char* b = row + (e0 / 256) * 210;
        const float d = h2f(rd16(b + 208));
        const int e = e0 % 256, n = e / 128, r = e % 128, qt = r / 32, l = r % 32;
        const unsigned char* ql = b + n * 64;
        const unsigned char* qh = b + 128 + n * 32;
        const float dl = d * (float)(signed char)b[192 + n * 8 + l / 16 + 2 * qt];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int li = l + i;
            int q;
            if (qt == 0) q = (ql[li] & 0xf) | ((qh[li] & 3) << 4);
            else if (qt == 1) q = (ql[li + 32] & 0xf) | (((qh[li] >> 2) & 3) << 4);
            else if (qt == 2) q = (ql[li] >> 4) | (((qh[li] >> 4) & 3) << 4);
            else q = (ql[li + 32] >> 4) | (((qh[li] >> 6) & 3) << 4);
            v[i] = dl * (float)(q - 32);
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

template <int T>
__device__ void gemv_body(const Seg* segs, int nseg, const float* x, int k, float* out, int ostride) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp, s = 0;
    while (s < nseg && row >= segs[s].rows) { row -= segs[s].rows; s++; }
    if (s >= nseg) return;
    const Seg sg = segs[s];
    const unsigned char* w = (const unsigned char*)sg.w + (size_t)row * sg.row_bytes;
    float acc[T];
    #pragma unroll
    for (int t = 0; t < T; t++) acc[t] = 0.f;
    switch (sg.type) {
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
        float a = acc[t];
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (lane == 0) out[(size_t)t * ostride + sg.out_off + row] = a;
    }
}

#define GEMV_T(T) extern "C" __global__ void __launch_bounds__(256) fe_gemv_t##T( \
    const Seg* segs, int nseg, const float* x, int k, float* out, int ostride) { \
    gemv_body<T>(segs, nseg, x, k, out, ostride); }
GEMV_T(1) GEMV_T(2) GEMV_T(3) GEMV_T(4) GEMV_T(5) GEMV_T(6) GEMV_T(7) GEMV_T(8)
"#;
