//! Top-k / top-p / presence-penalty sampling for the Flash engine's head, in the window graph.
//!
//! Per row (a window position), deterministically from that row's logits: subtract the presence
//! penalty from every token already in the output (the accepted output before the window, and
//! the window's own output tokens up to this row), keep the `top_k` best (ties to the lower id,
//! at most [`MAX_K`]), then the smallest prefix whose softmax mass at the temperature reaches
//! `top_p`, and take the Gumbel-max over it with the same `Philox(token, position, seed)` noise
//! as `fe_argmax1`. Greedy (temperature 0) is the first of the list. A sample is a function of
//! (seed, position, logits, output so far) only, so speculation stays exact: a draft is kept
//! only when it equals the sample, and then the window's earlier output tokens are the real ones.

use anyhow::{anyhow, ensure, Result};
use tang_compute::cuda::CudaBuffer;
use tang_compute::flash::MAX_T;
use tang_compute::{ComputeDevice, CudaComputeDevice};
use tang_moe::gpu::{self, Gpu, Module, Stream};

type B = CudaBuffer;

/// The most candidates kept per row.
pub const MAX_K: usize = 64;
/// Slices per row in the first pass.
const SLICES: usize = 64;

pub const SRC: &str = r#"
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
// (v, i) ranks before (ov, oi): larger value, then lower id.
__device__ __forceinline__ bool before(float v, int i, float ov, int oi) { return v > ov || (v == ov && i < oi); }
__device__ __forceinline__ void best(float& v, int& i, float ov, int oi) { if (before(ov, oi, v, i)) { v = ov; i = oi; } }

// params: [0] top_k, [1] top_p (f32 bits), [2] presence penalty (f32 bits), [3] rows whose
// input is an output token start at this row (>= t: none), [4..12] the window's input tokens.
// seen[v]: 1 if token v is in the accepted output before the window.
__device__ __forceinline__ float pen_logit(const float* l, int i, int row, const unsigned* params, const unsigned* seen) {
    const float pen = __uint_as_float(params[2]);
    float x = l[i];
    if (pen != 0.f) {
        bool s = seen[i] != 0u;
        for (int r = (int)params[3]; r <= row && !s; r++) s = params[4 + r] == (unsigned)i;
        if (s) x -= pen;
    }
    return x;
}

// Pass 1, grid (SLICES, t), block 1024: each slice's k best as (value, id) pairs, best first.
extern "C" __global__ void fs_topk1(const float* logits, int n, float* part, const unsigned* params, const unsigned* seen) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    const int row = blockIdx.y, k = min((int)params[0], 64);
    const float* l = logits + (size_t)row * n;
    const int per = (n + gridDim.x - 1) / gridDim.x, lo = blockIdx.x * per, hi = min(n, lo + per);
    float* out = part + ((size_t)row * gridDim.x + blockIdx.x) * 64 * 2;
    float pv = __int_as_float(0x7f800000);
    int pi = -1;
    for (int j = 0; j < k; j++) {
        float v = __int_as_float(0xff800000);
        int idx = 0x7fffffff;
        for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
            const float x = pen_logit(l, i, row, params, seen);
            if (before(pv, pi, x, i)) best(v, idx, x, i);
        }
        for (int o = 16; o > 0; o >>= 1) best(v, idx, __shfl_xor_sync(0xffffffffu, v, o), __shfl_xor_sync(0xffffffffu, idx, o));
        const int w = threadIdx.x >> 5;
        if ((threadIdx.x & 31) == 0) { bv[w] = v; bi[w] = idx; }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int q = 1; q < (int)(blockDim.x >> 5); q++) best(v, idx, bv[q], bi[q]);
            bv[0] = v; bi[0] = idx;
            out[2 * j] = v;
            out[2 * j + 1] = __int_as_float(idx);
        }
        __syncthreads();
        pv = bv[0]; pi = bi[0];
        __syncthreads();
    }
}

// Pass 2, grid (t), block 32: merge the slices' lists into the row's k best, cut at top_p, and
// take the Gumbel-max (or the first, greedy).
extern "C" __global__ void fs_topk2(const float* part, int nslices, const unsigned* params, const unsigned* ctl, unsigned* ids) {
    __shared__ float tv[64];
    __shared__ int ti[64];
    const int row = blockIdx.x, k = min((int)params[0], 64);
    const float* p = part + (size_t)row * nslices * 64 * 2;
    const int nc = nslices * k;
    float pv = __int_as_float(0x7f800000);
    int pi = -1;
    for (int j = 0; j < k; j++) {
        float v = __int_as_float(0xff800000);
        int idx = 0x7fffffff;
        for (int c = threadIdx.x; c < nc; c += 32) {
            const int s = c / k, q = c % k;
            const float x = p[(s * 64 + q) * 2];
            const int xi = __float_as_int(p[(s * 64 + q) * 2 + 1]);
            if (xi != 0x7fffffff && before(pv, pi, x, xi)) best(v, idx, x, xi);
        }
        for (int o = 16; o > 0; o >>= 1) best(v, idx, __shfl_xor_sync(0xffffffffu, v, o), __shfl_xor_sync(0xffffffffu, idx, o));
        if (threadIdx.x == 0) { tv[j] = v; ti[j] = idx; }
        pv = v; pi = idx;
    }
    __syncwarp();
    if (threadIdx.x != 0) return;
    const float temp = __uint_as_float(ctl[4]);
    int pick = ti[0];
    if (temp > 0.f) {
        const float it = 1.0f / temp, top_p = __uint_as_float(params[1]);
        int keep = 0;
        while (keep < k && ti[keep] != 0x7fffffff) keep++;
        float sum = 0.f;
        for (int j = 0; j < keep; j++) sum += expf((tv[j] - tv[0]) * it);
        float cum = 0.f;
        int cut = keep;
        for (int j = 0; j < keep; j++) {
            cum += expf((tv[j] - tv[0]) * it);
            if (cum >= top_p * sum) { cut = j + 1; break; }
        }
        const unsigned pos = ctl[0] + row, seed = ctl[3];
        float v = __int_as_float(0xff800000);
        int idx = 0x7fffffff;
        for (int j = 0; j < cut; j++) {
            const float u = ((float)(philox((unsigned)ti[j], pos, seed) >> 8) + 0.5f) * 5.9604644775390625e-8f;
            best(v, idx, tv[j] * it - logf(-logf(u)), ti[j]);
        }
        pick = idx;
    }
    ids[row] = (unsigned)pick;
}
"#;

