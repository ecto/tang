// GLM-5.3-Flash MoE kernels (NVRTC). Self-contained: the K-quant decoders (Q4_K/Q5_K/Q6_K,
// types 12/13/14) and the int8 activation rule are copied from the Coder-Next expert kernels
// so the GLM-5 geometry (hidden 4096, ff 2048, 288 experts top-8, 1 shared) can reuse them.
//
// Per token: the router is a [288][4096] GEMV in f32; selection on scores + bias, weights the
// unbiased scores, normalised and scaled (all in the router kernel, one warp per token).
// The experts run per (token, expert) pair like Coder-Next: int8-quantised activations, one
// warp per output row group over K-quant weight rows, with the GLM-5 swiglu clamp fused into
// the silu+quant step. The shared expert is the (K+1)-th "pair", as in Coder-Next.
//
// Launch plan (to wire under the flash engine, W6):
//   1. g5_route:        grid (T), block 32.  ids [T][8] (u32, sentinel allowed), wts [T][8].
//   2. fe_q8-style:     x -> xq codes [T][4096] + scales [T][128] (reuse cx_norm_q8's rule).
//   3. g5_moe_gu:       grid (ff/(8R), T*(K+1)), block 256; out gu [T*(K+1)][2*ff] f32.
//   4. g5_silu_q_clamp: grid (ff/32, T*(K+1)), block 32; hq codes + scales (limit clamp fused).
//   5. g5_moe_down:     grid (hid/(8R), T*(K+1)), block 256; y [T][4096], weights applied here.
//
// The sentinel expert id (== n_expert) reads garbage rows; g5_moe_down multiplies its pair by
// the routed weight 0 — but the cleanest contract is: the router never emits it (HF pads with
// it only for top_k > n_expert), so kernels assert ids < n_expert.

#define G5_HID 4096
#define G5_NE 288
#define G5_TOPK 8
#define G5_FF 2048
#define G5_FF_SH 2048
#define G5_NP (G5_TOPK + 1)
#define NEG_INF_F __int_as_float(0xff800000)
#define G5_LIMIT 10.0f

__device__ __forceinline__ float g5_sig(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float g5_silu(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float i2f(int q) { return __int_as_float(0x4B400000 + q) - 12582912.0f; }
__device__ __forceinline__ unsigned short rd16(const unsigned char* p) {
    return (unsigned short)p[0] | ((unsigned short)p[1] << 8);
}
__device__ __forceinline__ unsigned long long rd64(const unsigned char* p) {
    const unsigned long long a = (unsigned long long)p;
    const unsigned* q = (const unsigned*)(a & ~3ull);
    const unsigned sh = (unsigned)(a & 3) * 8;
    const unsigned w0 = q[0], w1 = q[1], w2 = q[2];
    const unsigned lo = __funnelshift_r(w0, w1, sh), hi = __funnelshift_r(w1, w2, sh);
    return (unsigned long long)lo | ((unsigned long long)hi << 32);
}
__device__ __forceinline__ float h2f(unsigned short h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}
__device__ __forceinline__ void scale_min_k4(int j, const unsigned char* q, int& s, int& m) {
    if (j < 4) { s = q[j] & 63; m = q[j + 4] & 63; }
    else { s = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

// The 8 weights e0..e0+8 as signed int8 lanes of (w0, w1), weight = dw*code - mw
// (types 12/13 only here; the launcher never hands the kernels anything else, plus Q8_0 for
// the shared/dense path).
template <int TY>
__device__ __forceinline__ void decode8i(const unsigned char* row, int e0, int& w0, int& w1, float& dw, float& mw) {
    mw = 0.f;
    if (TY == 8) {
        const unsigned char* b = row + (e0 / 32) * 34;
        dw = h2f(rd16(b));
        const unsigned long long q = rd64(b + 2 + e0 % 32);
        w0 = (int)(unsigned)q; w1 = (int)(unsigned)(q >> 32);
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

// One row of a GGUF-typed weight against int8 codes xc (plain order) with per-32 f32 scales.
template <int TY>
__device__ __forceinline__ float g5_dot8(const unsigned char* w, const unsigned char* xc, const float* xd, int k) {
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

struct G5ExW { const unsigned char* base; unsigned long long estride; int ty; int row_bytes; };

__device__ __forceinline__ const unsigned char* g5_stage(const unsigned char* src, int nb, uint4* sm, int lane) {
    const unsigned long long a = (unsigned long long)src & ~15ull;
    const int delta = (int)((unsigned long long)src - a);
    const int n16 = (delta + nb + 15) >> 4;
    for (int i = lane; i < n16; i += 32) sm[i] = __ldg((const uint4*)a + i);
    return (const unsigned char*)sm + delta;
}

// Router: one warp per token. logits = W x (f32 GEMV), scores = sigmoid(logits),
// selection by scores + bias, weights the unbiased scores of the selected experts,
// renormalised (denominator clamped at 2^-14), scaled by `scale`. ids/wts out [T][TOPK].
// PER = ceil(NE/32) loads per lane.
extern "C" __global__ void g5_route(const float* w, const float* bias, const float* x,
                                    unsigned* ids, float* wts, int T, float scale) {
    const int t = blockIdx.x, lane = threadIdx.x;
    const float* xt = x + (size_t)t * G5_HID;
    float v[9];  // ceil(288/32) = 9
    float mx = NEG_INF_F;
    #pragma unroll
    for (int i = 0; i < 9; i++) {
        const int e = lane + 32 * i;
        float s = NEG_INF_F;
        if (e < G5_NE) {
            const float* wr = w + (size_t)e * G5_HID;
            float acc = 0.f;
            for (int j = lane; j < G5_HID; j += 32) acc = fmaf(xt[j], wr[j], acc);
            for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
            s = g5_sig(acc);
        }
        v[i] = s;
        mx = fmaxf(mx, s);
    }
    unsigned used = 0;
    float my_w = 0.f, wsum = 0.f;
    unsigned my_id = G5_NE;
    for (int k = 0; k < G5_TOPK; k++) {
        float bv = NEG_INF_F;
        int bi = 0x7fffffff;
        #pragma unroll
        for (int i = 0; i < 9; i++) {
            const int e = lane + 32 * i;
            if (e < G5_NE && !((used >> i) & 1u)) {
                const float sb = v[i] + __ldg(bias + e);
                if (sb > bv || (sb == bv && e < bi)) { bv = sb; bi = e; }
            }
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (bi != 0x7fffffff && (bi & 31) == lane) used |= 1u << (bi >> 5);
        // the weight is the unbiased score of expert bi, which lives in lane bi&31, slot bi>>5
        float mine;
        if (bi == 0x7fffffff) mine = 0.f;
        else {
            const int slot = bi >> 5;
            mine = __shfl_sync(0xffffffffu, v[slot], bi & 31);
        }
        wsum += mine;
        if (lane == k) { my_id = bi == 0x7fffffff ? (unsigned)G5_NE : (unsigned)bi; my_w = mine; }
    }
    if (lane < G5_TOPK) {
        float w = my_w / fmaxf(wsum, 6.103515625e-5f);
        if (scale != 0.f && scale != 1.f) w *= scale;
        ids[(size_t)t * G5_TOPK + lane] = my_id;
        wts[(size_t)t * G5_TOPK + lane] = w;
    }
}

// Gate|up for pair p (routed expert for p < T*K, shared expert for p == T*K... here pairs are
// (token, rank) so p in [0, T*K) plus T*K shared slots p in [T*K, T*(K+1)) whose id is
// implicit). R rows a warp, staged into shared memory. out gu [T*(K+1)][2*ff].
// Grid (ff/(8R), T*(K+1)), block 256, smem 8R rows.
template <int R, bool SHARED_PAIR>
__device__ __forceinline__ void g5_gu_body(const G5ExW& g, const G5ExW& sg, const unsigned* ids,
                                           const unsigned char* xq, float* out, int T, int r16) {
    extern __shared__ uint4 sm[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int p = blockIdx.y;                    // pair index
    const bool shared = p >= T * G5_TOPK;        // shared-expert pairs come after routed
    const G5ExW& m = shared ? sg : g;
    const unsigned e = shared ? 0u : ids[p];
    const unsigned char* w[R];
    int rr[R];
    #pragma unroll
    for (int j = 0; j < R; j++) {
        const int r = (blockIdx.x * 8 + warp) * R + j;
        rr[j] = r;
        const unsigned char* src = m.base + (size_t)e * m.estride + (size_t)r * m.row_bytes;
        w[j] = g5_stage(src, m.row_bytes, sm + (warp * R + j) * r16, lane);
    }
    __syncwarp();
    const float* xd = (const float*)(xq + (size_t)T * G5_HID);
    const unsigned char* xcodes = shared ? xq + (size_t)(p - T * G5_TOPK) * G5_HID
                                         : xq + (size_t)(p / G5_TOPK) * G5_HID;
    const float* xs = shared ? xd + (size_t)(p - T * G5_TOPK) * (G5_HID / 32)
                             : xd + (size_t)(p / G5_TOPK) * (G5_HID / 32);
    #pragma unroll
    for (int j = 0; j < R; j++) {
        const float a = m.ty == 8 ? g5_dot8<8>(w[j], xcodes, xs, G5_HID)
                    : m.ty == 12 ? g5_dot8<12>(w[j], xcodes, xs, G5_HID)
                    : m.ty == 13 ? g5_dot8<13>(w[j], xcodes, xs, G5_HID)
                                 : g5_dot8<14>(w[j], xcodes, xs, G5_HID);
        if (lane == 0) out[(size_t)p * 2 * G5_FF + rr[j]] = a;
    }
}
extern "C" __global__ void __launch_bounds__(256) g5_moe_gu(const G5ExW g, const G5ExW sg,
                                                           const unsigned* ids, const unsigned char* xq,
                                                           float* out, int T, int r16) {
    g5_gu_body<2, false>(g, sg, ids, xq, out, T, r16);
}

// silu(clamp(gate)) * clamp(up), then int8 (d = amax/127, round half away) into hq:
// codes [NP_ALL][ff] then scales [NP_ALL][ff/32], where NP_ALL = P_NP * G5_NP (passed in:
// pairs per token is K+1; the host allocates hq for T*(K+1) pairs). The swiglu clamp
// (limit) is the only difference from Coder-Next's cx_silu_q. Grid (ff/32, T*(K+1)), block 32.
extern "C" __global__ void g5_silu_q_clamp(const float* gu, unsigned char* hq, float limit, int np_all, int pbase) {
    const int c = blockIdx.x, p = pbase + blockIdx.y, lane = threadIdx.x;
    const float g = fminf(gu[(size_t)p * 2 * G5_FF + c * 32 + lane], limit);
    const float u = fminf(fmaxf(gu[(size_t)p * 2 * G5_FF + G5_FF + c * 32 + lane], -limit), limit);
    const float v = g5_silu(g) * u;
    float amax = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = amax / 127.0f;
    const int q = d == 0.f ? 0 : (int)fminf(fmaxf(roundf(v / d), -127.f), 127.f);
    hq[(size_t)p * G5_FF + c * 32 + lane] = (unsigned char)(signed char)q;
    if (lane == 0) ((float*)(hq + (size_t)np_all * G5_FF))[(size_t)p * (G5_FF / 32) + c] = d;
}

// Down rows for pair p (routed expert for p < T*K, shared for p >= T*K, id implicit 0),
// R rows a warp, staged; per-pair rows yp [T*(K+1)][hid]. The combination (sum of weighted
// pair rows) is g5_moe_combine. Grid (hid/(8R), T*(K+1)), block 256, smem 8R rows.
template <int R>
__device__ __forceinline__ void g5_down_body(const G5ExW d, const G5ExW sd, const unsigned* ids,
                                             const unsigned char* hq, float* yp, int T, int r16, int np_all) {
    extern __shared__ uint4 sm[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int p = blockIdx.y;
    const bool shared = p >= T * G5_TOPK;
    const G5ExW& m = shared ? sd : d;
    const unsigned e = shared ? 0u : ids[p];
    const unsigned char* w[R];
    int rr[R];
    #pragma unroll
    for (int j = 0; j < R; j++) {
        const int r = (blockIdx.x * 8 + warp) * R + j;
        rr[j] = r;
        const unsigned char* src = m.base + (size_t)e * m.estride + (size_t)r * m.row_bytes;
        w[j] = g5_stage(src, m.row_bytes, sm + (warp * R + j) * r16, lane);
    }
    __syncwarp();
    const unsigned char* hcodes = hq + (size_t)p * G5_FF;
    const float* hd = (const float*)(hq + (size_t)np_all * G5_FF) + (size_t)p * (G5_FF / 32);
    #pragma unroll
    for (int j = 0; j < R; j++) {
        const float a = m.ty == 8 ? g5_dot8<8>(w[j], hcodes, hd, G5_FF)
                    : m.ty == 12 ? g5_dot8<12>(w[j], hcodes, hd, G5_FF)
                    : m.ty == 13 ? g5_dot8<13>(w[j], hcodes, hd, G5_FF)
                                 : g5_dot8<14>(w[j], hcodes, hd, G5_FF);
        if (lane == 0) yp[(size_t)p * G5_HID + rr[j]] = a;
    }
}
extern "C" __global__ void __launch_bounds__(256) g5_moe_down(const G5ExW d, const G5ExW sd,
                                                             const unsigned* ids, const unsigned char* hq,
                                                             float* yp, int T, int r16, int np_all) {
    g5_down_body<2>(d, sd, ids, hq, yp, T, r16, np_all);
}

// y[t][i] = sum_k wts[t][k] * yp[t*K + k][i] + yp[T*K + t][i]  (shared expert, weight 1).
// Grid (hid/256, T), block 256.
extern "C" __global__ void g5_moe_combine(const float* yp, const float* wts, float* y, int T) {
    const int i = blockIdx.x * 256 + threadIdx.x, t = blockIdx.y;
    if (i >= G5_HID) return;
    float acc = yp[(size_t)(T * G5_TOPK + t) * G5_HID + i];
    #pragma unroll
    for (int k = 0; k < G5_TOPK; k++)
        acc = fmaf(wts[(size_t)t * G5_TOPK + k], yp[(size_t)(t * G5_TOPK + k) * G5_HID + i], acc);
    y[(size_t)t * G5_HID + i] = acc;
}
