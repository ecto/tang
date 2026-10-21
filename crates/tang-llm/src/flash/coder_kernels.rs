//! The CUDA kernels of the Qwen3-Coder-Next engine ([`super::coder`]): what a plain-residual,
//! indexer-free, all-VRAM qwen3next window needs beyond the native GEMV (`kernels::GEMV_SRC`,
//! compiled into the same module) and the GDN kernels of the flash module.
//!
//! Every kernel's arithmetic for a token is independent of the window width `t` (the same
//! reduction orders at any `t`), so a token's outputs are bitwise the same in a T=1 window, a
//! T=8 window and a verify window (`flash-tcheck`).

/// The `#define` prelude for a model's shapes.
pub fn prelude(c: &Shapes) -> String {
    format!(
        "#define CX_HID {hid}\n#define CX_NH {nh}\n#define CX_NKV {nkv}\n#define CX_HD {hd}\n\
         #define CX_ROT {rot}\n#define CX_G {g}\n#define CX_QK {qk}\n#define CX_QV {qv}\n\
         #define CX_FF {ff}\n#define CX_TOPK {topk}\n#define CX_NE {ne}\n#define CX_NP {np}\n\
         #define CX_CHUNK {chunk}\n",
        hid = c.hidden,
        nh = c.n_head,
        nkv = c.n_kv,
        hd = c.head_dim,
        rot = c.n_rot,
        g = c.n_head / c.n_kv,
        qk = c.n_head * 2 * c.head_dim,
        qv = c.n_head * 2 * c.head_dim + c.n_kv * c.head_dim,
        ff = c.ff,
        topk = c.topk,
        ne = c.n_expert,
        np = super::coder::MAX_T * (c.topk + 1),
        chunk = ATTN_CHUNK,
    )
}

/// Positions per split-K attention block.
pub const ATTN_CHUNK: usize = 64;

/// The shapes the kernels are compiled for.
#[derive(Clone, Debug)]
pub struct Shapes {
    pub hidden: usize,
    pub n_head: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub n_rot: usize,
    pub ff: usize,
    pub topk: usize,
    pub n_expert: usize,
}

pub const SRC: &str = r#"
#define MAXT 8
#define NEG_INF_F __int_as_float(0xff800000)

__device__ __forceinline__ float sigm_f(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float silu_f2(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ unsigned short f2bf(float x) {
    unsigned int b = __float_as_uint(x);
    return (unsigned short)((b + 0x7fffu + ((b >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ float bf2f(unsigned short b) { return __uint_as_float(((unsigned int)b) << 16); }

// Sum over a block (blockDim.x a multiple of 32, at most 1024); every thread gets the total, in
// a fixed order.
__device__ float cx_bsum(float v, float* sh) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) sh[w] = v;
    __syncthreads();
    float s = 0.f;
    for (int i = 0; i < (int)(blockDim.x >> 5); i++) s += sh[i];
    return s;
}

// x = rmsnorm(r) * w and its int8 codes xq (fe_q8's layout and rule: codes [XQ8_T][n], then
// per-32 f32 scales), after r += y when `add`. Grid t, block 1024 (or 256); n % blockDim == 0, n <= 16 blockDim.
__device__ void norm_q8_body(float* rt, float (&v)[16], int nv, const float* w, float* x, unsigned char* xq,
                             int n, float eps, int t) {
    __shared__ float sh[32];
    float ss = 0.f;
    for (int j = 0; j < nv; j++) ss = fmaf(v[j], v[j], ss);
    ss = cx_bsum(ss, sh);
    const float inv = 1.0f / sqrtf(ss / (float)n + eps);
    const int lane = threadIdx.x & 31;
    float* xd = (float*)(xq + (size_t)XQ8_T * n);
    for (int j = 0; j < nv; j++) {
        const int i = threadIdx.x + j * blockDim.x;
        const float y = v[j] * inv * w[i];
        x[(size_t)t * n + i] = y;
        float amax = fabsf(y);
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float d = amax / 127.0f;
        const int q = d == 0.f ? 0 : (int)fminf(fmaxf(roundf(y / d), -127.f), 127.f);
        xq[(size_t)t * n + i] = (unsigned char)(signed char)q;
        if (lane == 0) xd[t * (n / 32) + i / 32] = d;
    }
}
extern "C" __global__ void cx_norm_q8(float* r, const float* y, const float* w, float* x, unsigned char* xq,
                                      int n, float eps, int add) {
    const int t = blockIdx.x, nv = n / blockDim.x;
    float* rt = r + (size_t)t * n;
    float v[16];
    for (int j = 0; j < nv; j++) {
        const int i = threadIdx.x + j * blockDim.x;
        float a = rt[i];
        if (add) { a += y[(size_t)t * n + i]; rt[i] = a; }
        v[j] = a;
    }
    norm_q8_body(rt, v, nv, w, x, xq, n, eps, t);
}

// r += Σ_k w_k parts[t·K + k] (rank order) + sigmoid(lg[t][NE]) parts[T·K + t] (the shared
// expert), y (when given) gets that sum alone; then x, xq = the next norm of r (as
// cx_norm_q8). Grid t, block 1024.
extern "C" __global__ void cx_moe_add_norm(float* r, const float* parts, const float* wts, const float* lg, int ls,
                                           int T, float* y, const float* w, float* x, unsigned char* xq, float eps) {
    const int t = blockIdx.x, n = CX_HID, nv = CX_HID / (int)blockDim.x;
    float* rt = r + (size_t)t * n;
    const float g = sigm_f(lg[(size_t)t * ls + CX_NE]);
    float v[16];
    for (int j = 0; j < nv; j++) {
        const int i = threadIdx.x + j * blockDim.x;
        float acc = 0.f;
        for (int k = 0; k < CX_TOPK; k++) acc += wts[t * CX_TOPK + k] * parts[((size_t)t * CX_TOPK + k) * n + i];
        acc += g * parts[((size_t)T * CX_TOPK + t) * n + i];
        if (y) y[(size_t)t * n + i] = acc;
        const float a = rt[i] + acc;
        rt[i] = a;
        v[j] = a;
    }
    norm_q8_body(rt, v, nv, w, x, xq, n, eps, t);
}

// Softmax over the router logits, top CX_TOPK (ties to the lower id), renormalised in f64.
// Grid t, block 32. Logits row stride `ls`; writes ids [t][TOPK], w [t][TOPK].
extern "C" __global__ void cx_route(const float* lg, int ls, unsigned* ids, float* wts, float scale) {
    const int t = blockIdx.x, lane = threadIdx.x;
    const float* l = lg + (size_t)t * ls;
    const int PER = (CX_NE + 31) / 32;
    float v[PER];
    float mx = NEG_INF_F;
    for (int i = 0; i < PER; i++) {
        const int e = lane + 32 * i;
        v[i] = e < CX_NE ? l[e] : NEG_INF_F;
        mx = fmaxf(mx, v[i]);
    }
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    // p_e = exp(l_e - max) / sum over all e; renormalised over the top-k the sum cancels.
    const double sum = 1.0;
    unsigned used = 0;
    unsigned my_id = 0;
    double my_p = 0.0, wsum = 0.0;
    for (int k = 0; k < CX_TOPK; k++) {
        float bv = NEG_INF_F;
        int bi = 0x7fffffff;
        for (int i = 0; i < PER; i++) {
            const int e = lane + 32 * i;
            if (e < CX_NE && !((used >> i) & 1u) && (v[i] > bv || (v[i] == bv && e < bi))) { bv = v[i]; bi = e; }
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if ((bi & 31) == lane) used |= 1u << (bi >> 5);
        const double p = exp((double)bv - (double)mx) / sum;
        wsum += p;
        if (lane == k) { my_id = (unsigned)bi; my_p = p; }
    }
    // (flash-ref clamps Σ top p at 6.1e-5; the top-1 term alone is p_max / sum >= 1 / NE here.)
    if (lane < CX_TOPK) {
        float w = (float)(my_p / wsum);
        if (scale != 0.f && scale != 1.f) w *= scale;
        ids[t * CX_TOPK + lane] = my_id;
        wts[t * CX_TOPK + lane] = w;
    }
}

// One row of a GGUF-typed weight against int8 codes `xc` (plain order) with per-32 scales `xd`.
template <int TY>
__device__ __forceinline__ float cx_dot8(const unsigned char* w, const unsigned char* xc, const float* xd, int k) {
    const int lane = threadIdx.x & 31;
    float acc = 0.f;
    for (int e0 = lane * 8; e0 < k; e0 += 256) {
        int w0, w1;
        float dw, mw;
        decode8i<TY>(w, e0, w0, w1, dw, mw);
        const uint2 xv = *(const uint2*)(xc + e0);
        const float dx = xd[e0 / 32];
        const int s = __dp4a(w0, (int)xv.x, __dp4a(w1, (int)xv.y, 0));
        if (TY == 12 || TY == 13) {
            const int sx = __dp4a(0x01010101, (int)xv.x, __dp4a(0x01010101, (int)xv.y, 0));
            acc = fmaf(dx, fmaf(dw, i2f(s), -mw * i2f(sx)), acc);
        } else {
            acc = fmaf(dx * dw, i2f(s), acc);
        }
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    return acc;
}
__device__ __forceinline__ float cx_dot(int ty, const unsigned char* w, const unsigned char* xc, const float* xd, int k) {
    switch (ty) {
        case 8: return cx_dot8<8>(w, xc, xd, k);
        case 2: return cx_dot8<2>(w, xc, xd, k);
        case 20: return cx_dot8<20>(w, xc, xd, k);
        case 6: return cx_dot8<6>(w, xc, xd, k);
        case 42: return cx_dot8<42>(w, xc, xd, k);
        case 23: return cx_dot8<23>(w, xc, xd, k);
        case 11: return cx_dot8<11>(w, xc, xd, k);
        case 12: return cx_dot8<12>(w, xc, xd, k);
        case 13: return cx_dot8<13>(w, xc, xd, k);
        case 14: return cx_dot8<14>(w, xc, xd, k);
        default: return 0.f;
    }
}

struct ExW { unsigned long long base; unsigned long long estride; int ty; int row_bytes; };

// Stage `nb` bytes at `src` into this warp's shared buffer with aligned 16-byte loads; returns
// the row's first byte there.
__device__ __forceinline__ const unsigned char* cx_stage(const unsigned char* src, int nb, uint4* sm, int lane) {
    const unsigned long long a = (unsigned long long)src & ~15ull;
    const int delta = (int)((unsigned long long)src - a);
    const int n16 = (delta + nb + 15) >> 4;
    for (int i = lane; i < n16; i += 32) sm[i] = __ldg((const uint4*)a + i);
    return (const unsigned char*)sm + delta;
}

// Experts' gate and up rows, two rows a warp (both staged in shared memory before either is
// decoded), 8 warps a block. Pairs p < T·K are (token p / K, rank p % K) with the routed expert
// ids[p]; p >= T·K is the shared expert for token p − T·K. out[p][r]: r < FF gate, else up.
// Grid (2 FF / 16, T · (K + 1)), block 256, dynamic smem 16 × r16 × 16 bytes.
extern "C" __global__ void __launch_bounds__(256) cx_moe_gu(ExW g, ExW u, ExW sg, ExW su, const unsigned* ids,
                                                            const unsigned char* xq, float* out, int T, int r16) {
    extern __shared__ uint4 dsm[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, p = blockIdx.y;
    const bool shared = p >= T * CX_TOPK;
    const int t = shared ? p - T * CX_TOPK : p / CX_TOPK;
    const unsigned e = shared ? 0u : ids[p];
    const unsigned char* w[2];
    int ty[2], rr[2];
    #pragma unroll
    for (int j = 0; j < 2; j++) {
        const int r = (blockIdx.x * 8 + warp) * 2 + j;
        rr[j] = r;
        const ExW& m = shared ? (r < CX_FF ? sg : su) : (r < CX_FF ? g : u);
        const int ro = r < CX_FF ? r : r - CX_FF;
        ty[j] = m.ty;
        const unsigned char* src = (const unsigned char*)m.base + (size_t)e * m.estride + (size_t)ro * m.row_bytes;
        w[j] = cx_stage(src, m.row_bytes, dsm + (warp * 2 + j) * r16, lane);
    }
    __syncwarp();
    const float* xd = (const float*)(xq + (size_t)XQ8_T * CX_HID);
    #pragma unroll
    for (int j = 0; j < 2; j++) {
        const float a = cx_dot(ty[j], w[j], xq + (size_t)t * CX_HID, xd + t * (CX_HID / 32), CX_HID);
        if (lane == 0) out[(size_t)p * 2 * CX_FF + rr[j]] = a;
    }
}

// h = silu(gate) * up per pair, then int8 (fe_q8's rule) into hq: codes [CX_NP][FF], then
// scales [CX_NP][FF / 32]. Grid (FF / 32, T · (K + 1)), block 32.
extern "C" __global__ void cx_silu_q(const float* gu, unsigned char* hq) {
    const int c = blockIdx.x, p = blockIdx.y, lane = threadIdx.x;
    const float g = gu[(size_t)p * 2 * CX_FF + c * 32 + lane];
    const float u = gu[(size_t)p * 2 * CX_FF + CX_FF + c * 32 + lane];
    const float v = silu_f2(g) * u;
    float amax = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = amax / 127.0f;
    const int q = d == 0.f ? 0 : (int)fminf(fmaxf(roundf(v / d), -127.f), 127.f);
    hq[(size_t)p * CX_FF + c * 32 + lane] = (unsigned char)(signed char)q;
    if (lane == 0) ((float*)(hq + (size_t)CX_NP * CX_FF))[p * (CX_FF / 32) + c] = d;
}

// Experts' down rows (shared expert for p >= T·K): out[p][r], two rows a warp, staged.
// Grid (HID / 16, T · (K + 1)), block 256, dynamic smem 16 × r16 × 16 bytes.
extern "C" __global__ void __launch_bounds__(256) cx_moe_down(ExW d, ExW sd, const unsigned* ids,
                                                              const unsigned char* hq, float* out, int T, int r16) {
    extern __shared__ uint4 dsm[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, p = blockIdx.y;
    const bool shared = p >= T * CX_TOPK;
    const ExW& m = shared ? sd : d;
    const unsigned e = shared ? 0u : ids[p];
    const unsigned char* w[2];
    int rr[2];
    #pragma unroll
    for (int j = 0; j < 2; j++) {
        const int r = (blockIdx.x * 8 + warp) * 2 + j;
        rr[j] = r;
        const unsigned char* src = (const unsigned char*)m.base + (size_t)e * m.estride + (size_t)r * m.row_bytes;
        w[j] = cx_stage(src, m.row_bytes, dsm + (warp * 2 + j) * r16, lane);
    }
    __syncwarp();
    const float* hd = (const float*)(hq + (size_t)CX_NP * CX_FF);
    #pragma unroll
    for (int j = 0; j < 2; j++) {
        const float a = cx_dot(m.ty, w[j], hq + (size_t)p * CX_FF, hd + p * (CX_FF / 32), CX_FF);
        if (lane == 0) out[(size_t)p * CX_HID + rr[j]] = a;
    }
}

// ---- K-split dense GEMV ----
// A block of 8 warps handles 2 rows; the 4 warps of a row each take a contiguous quarter of k
// (k % 1024 == 0) and their partial sums are added in warp order. Same segments, outputs and
// activation formats as fe_gemv8 (int8 codes for quantized rows, f32 for F32/BF16 rows), so
// it drops in for it; at T = 1 a 2048-row matrix gets 8192 warps instead of 1024.
template <int T, int TY>
__device__ __forceinline__ void dotk8(const unsigned char* w, const unsigned char* xq, int k, int k0, int k1, float* acc) {
    const int lane = threadIdx.x & 31;
    const float* xd = (const float*)(xq + (size_t)XQ8_T * k);
    for (int e0 = k0 + lane * 8; e0 < k1; e0 += 256) {
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
template <int T, int TY>
__device__ __forceinline__ void dotkf(const unsigned char* w, const float* x, int k, int k0, int k1, float* acc) {
    const int lane = threadIdx.x & 31;
    for (int e0 = k0 + lane * 8; e0 < k1; e0 += 256) {
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
__device__ void gemvk_body(const Seg* segs, int nseg, const float* x, const unsigned char* xq, int k, float* out, int ostride) {
    __shared__ float part[8][MAXT];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int j = warp >> 2, q = warp & 3;
    int row = blockIdx.x * 2 + j, s = 0;
    while (s < nseg && row >= segs[s].rows) { row -= segs[s].rows; s++; }
    float acc[T];
    #pragma unroll
    for (int t = 0; t < T; t++) acc[t] = 0.f;
    const bool ok = s < nseg;
    if (ok) {
        const Seg sg = segs[s];
        const unsigned char* w = (const unsigned char*)sg.w + (size_t)row * sg.row_bytes;
        const int k0 = q * (k / 4), k1 = k0 + k / 4;
        switch (sg.type) {
            case 30: dotkf<T, 30>(w, x, k, k0, k1, acc); break;
            case 0: dotkf<T, 0>(w, x, k, k0, k1, acc); break;
            case 8: dotk8<T, 8>(w, xq, k, k0, k1, acc); break;
            case 2: dotk8<T, 2>(w, xq, k, k0, k1, acc); break;
            case 20: dotk8<T, 20>(w, xq, k, k0, k1, acc); break;
            case 6: dotk8<T, 6>(w, xq, k, k0, k1, acc); break;
            case 42: dotk8<T, 42>(w, xq, k, k0, k1, acc); break;
            case 23: dotk8<T, 23>(w, xq, k, k0, k1, acc); break;
            case 11: dotk8<T, 11>(w, xq, k, k0, k1, acc); break;
            case 12: dotk8<T, 12>(w, xq, k, k0, k1, acc); break;
            case 13: dotk8<T, 13>(w, xq, k, k0, k1, acc); break;
            case 14: dotk8<T, 14>(w, xq, k, k0, k1, acc); break;
            default: break;
        }
    }
    #pragma unroll
    for (int t = 0; t < T; t++) {
        float a = acc[t];
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (lane == 0) part[warp][t] = a;
    }
    __syncthreads();
    if (ok && q == 0 && lane < T) {
        const float v = ((part[4 * j][lane] + part[4 * j + 1][lane]) + part[4 * j + 2][lane]) + part[4 * j + 3][lane];
        out[(size_t)lane * ostride + segs[s].out_off + row] = v;
    }
}
#define GEMVK_T(T) extern "C" __global__ void __launch_bounds__(256) cx_gemvk_t##T( \
    const Seg* segs, int nseg, const float* x, const unsigned char* xq, int k, float* out, int ostride) { \
    gemvk_body<T>(segs, nseg, x, xq, k, out, ostride); }
GEMVK_T(1) GEMVK_T(2) GEMVK_T(3) GEMVK_T(4) GEMVK_T(5) GEMVK_T(6) GEMVK_T(7) GEMVK_T(8)

// Gated attention prep: q heads (rmsnorm * qn, NeoX rope on CX_ROT dims) into q [t][NH][HD];
// k heads (rmsnorm * kn, rope) and v into the bf16 caches at position ctl[0] + t.
// Grid (t, NH + NKV), block HD.
extern "C" __global__ void cx_attn_prep(const float* proj, int stride, const float* qn, const float* kn,
                                        const float* rc, const float* rs, const unsigned* ctl, float* q,
                                        unsigned short* kc, unsigned short* vc, float eps) {
    __shared__ float sh[32];
    __shared__ float sx[CX_HD];
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const unsigned pos = ctl[0] + t;
    const float* row = proj + (size_t)t * stride;
    const bool isq = h < CX_NH;
    const int g = h - CX_NH;
    float x = isq ? row[h * 2 * CX_HD + d] : row[CX_QK + g * CX_HD + d];
    const float ss = cx_bsum(x * x, sh);
    x = x * (1.0f / sqrtf(ss / (float)CX_HD + eps)) * (isq ? qn[d] : kn[d]);
    sx[d] = x;
    __syncthreads();
    if (d < CX_ROT) {
        const int half = CX_ROT / 2, i = d % half;
        const float c = rc[(size_t)pos * half + i], s = rs[(size_t)pos * half + i];
        x = d < half ? sx[d] * c - sx[d + half] * s : sx[d - half] * s + sx[d] * c;
    }
    if (isq) {
        q[((size_t)t * CX_NH + h) * CX_HD + d] = x;
    } else {
        const size_t o = ((size_t)pos * CX_NKV + g) * CX_HD + d;
        kc[o] = f2bf(x);
        vc[o] = f2bf(row[CX_QV + g * CX_HD + d]);
    }
}

// Split-K causal attention: block (token c, kv group g, chunk j of CX_CHUNK positions) runs the
// CX_G query heads of group g over its chunk (4 warps; lane owns HD / 32 dims) and writes per
// head (m, l, acc[HD]) to part[c][g][j][G][HD + 2]. Chunks past the token's position write l = 0.
#define CX_DL (CX_HD / 32)
extern "C" __global__ void __launch_bounds__(128) cx_attn_part(const float* q, const unsigned short* kc,
                                                               const unsigned short* vc, const unsigned* ctl,
                                                               float* part, int nchunk) {
    __shared__ float wm[4][CX_G], wl[4][CX_G];
    __shared__ float wacc[4][CX_HD];
    const int c = blockIdx.x, g = blockIdx.y, j = blockIdx.z;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int pos = (int)ctl[0] + c;
    float* out = part + (((size_t)c * CX_NKV + g) * nchunk + j) * CX_G * (CX_HD + 2);
    const int lo = j * CX_CHUNK;
    if (lo > pos) {
        if (threadIdx.x < CX_G) out[threadIdx.x * (CX_HD + 2) + 1] = 0.f;
        return;
    }
    const float scale = 1.0f / sqrtf((float)CX_HD);
    float qv[CX_G][CX_DL];
    for (int h = 0; h < CX_G; h++)
        for (int i = 0; i < CX_DL; i++) qv[h][i] = q[((size_t)c * CX_NH + g * CX_G + h) * CX_HD + lane * CX_DL + i] * scale;
    float m[CX_G], l[CX_G], acc[CX_G][CX_DL];
    for (int h = 0; h < CX_G; h++) {
        m[h] = NEG_INF_F;
        l[h] = 0.f;
        for (int i = 0; i < CX_DL; i++) acc[h][i] = 0.f;
    }
    const int hi = min(pos + 1, lo + CX_CHUNK);
    for (int p = lo + warp; p < hi; p += 4) {
        const unsigned short* kr = kc + ((size_t)p * CX_NKV + g) * CX_HD + lane * CX_DL;
        const unsigned short* vr = vc + ((size_t)p * CX_NKV + g) * CX_HD + lane * CX_DL;
        float kv[CX_DL], vv[CX_DL];
        #pragma unroll
        for (int i = 0; i < CX_DL; i++) { kv[i] = bf2f(kr[i]); vv[i] = bf2f(vr[i]); }
        #pragma unroll
        for (int h = 0; h < CX_G; h++) {
            float sc = 0.f;
            #pragma unroll
            for (int i = 0; i < CX_DL; i++) sc = fmaf(qv[h][i], kv[i], sc);
            for (int o = 16; o > 0; o >>= 1) sc += __shfl_xor_sync(0xffffffffu, sc, o);
            const float mn = fmaxf(m[h], sc), corr = expf(m[h] - mn), pr = expf(sc - mn);
            #pragma unroll
            for (int i = 0; i < CX_DL; i++) acc[h][i] = fmaf(pr, vv[i], acc[h][i] * corr);
            l[h] = l[h] * corr + pr;
            m[h] = mn;
        }
    }
    for (int h = 0; h < CX_G; h++)
        if (lane == 0) { wm[warp][h] = m[h]; wl[warp][h] = l[h]; }
    for (int h = 0; h < CX_G; h++) {
        __syncthreads();
        for (int i = 0; i < CX_DL; i++) wacc[warp][lane * CX_DL + i] = acc[h][i];
        __syncthreads();
        float M = NEG_INF_F;
        for (int w = 0; w < 4; w++) if (wl[w][h] > 0.f) M = fmaxf(M, wm[w][h]);
        float L = 0.f;
        for (int w = 0; w < 4; w++) if (wl[w][h] > 0.f) L += wl[w][h] * expf(wm[w][h] - M);
        for (int d = threadIdx.x; d < CX_HD; d += 128) {
            float o = 0.f;
            for (int w = 0; w < 4; w++) if (wl[w][h] > 0.f) o += wacc[w][d] * expf(wm[w][h] - M);
            out[h * (CX_HD + 2) + 2 + d] = o;
        }
        if (threadIdx.x == 0) { out[h * (CX_HD + 2)] = M; out[h * (CX_HD + 2) + 1] = L; }
    }
}

// Merge each (token, head)'s chunks and apply the sigmoid gate (the second half of the head's
// q projection). Grid (t, NH), block HD.
extern "C" __global__ void cx_attn_merge(const float* part, int nchunk, const float* proj, int stride, float* out) {
    const int c = blockIdx.x, h = blockIdx.y, g = h / CX_G, hh = h % CX_G, d = threadIdx.x;
    const size_t hs = (size_t)CX_G * (CX_HD + 2);
    const float* pp = part + ((size_t)c * CX_NKV + g) * nchunk * hs + hh * (CX_HD + 2);
    float M = NEG_INF_F;
    for (int j = 0; j < nchunk; j++) if (pp[j * hs + 1] > 0.f) M = fmaxf(M, pp[j * hs]);
    float L = 0.f, o = 0.f;
    for (int j = 0; j < nchunk; j++) {
        const float* qq = pp + j * hs;
        if (qq[1] > 0.f) {
            const float f = expf(qq[0] - M);
            L += qq[1] * f;
            o += qq[2 + d] * f;
        }
    }
    const float gt = proj[(size_t)c * stride + h * 2 * CX_HD + CX_HD + d];
    out[((size_t)c * CX_NH + h) * CX_HD + d] = (o / L) * sigm_f(gt);
}
"#;
