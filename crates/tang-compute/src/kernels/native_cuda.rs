//! CUDA source for the native-type (NatX, `crate::flash_native`) int8-activation GEMV. One
//! module per weight type (`native_source(ty)` prepends `#define TY <ggml id>`), so only the
//! types a model uses are compiled. Kernels `fl_nat<ty>_t<T>`, T = 1..8 columns.

/// The kernel body; compiled with `TY` defined.
pub const NATIVE_CUDA: &str = r#"
typedef unsigned long long u64;

__device__ __forceinline__ float h2f(unsigned short h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
// IQ4_NL grid for four nibbles (bytes 0..15) at once.
__device__ __forceinline__ unsigned int iq4w(unsigned int n) {
    const unsigned int sel = (n & 0x7u) | ((n >> 4) & 0x70u) | ((n >> 8) & 0x700u) | ((n >> 12) & 0x7000u);
    const unsigned int lo = __byte_perm(0xBFAD9881u, 0xF6EADDCFu, sel);
    const unsigned int hi = __byte_perm(0x26190D01u, 0x71594535u, sel);
    const unsigned int m = ((n >> 3) & 0x01010101u) * 0xFFu;
    return (lo & ~m) | (hi & m);
}

#define LB ((TY == 42 || TY == 11) ? 8 : 16)
#define HB ((TY == 6 || TY == 13 || TY == 11) ? 4 : (TY == 14 ? 8 : 0))
#define HAS_MIN (TY == 12 || TY == 13)
#define PER16 (TY == 11 || TY == 14)
#ifndef NAT_PF
#define NAT_PF(T) (T == 1)
#endif
#define SPC (TY == 23 ? 1 : ((TY == 11 || TY == 12 || TY == 13 || TY == 14) ? 2 : 0))


// The eight dp4a operands (int8 lanes in activation order) of one chunk from its loaded words:
// low plane l[0..LB/4), extra-bit words h0, h1.
__device__ __forceinline__ void operands(const unsigned int* l, unsigned int h0, unsigned int h1, int op[8]) {
    if (LB == 16) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            op[2 * i] = (int)(l[i] & 0x0f0f0f0fu);
            op[2 * i + 1] = (int)((l[i] >> 4) & 0x0f0f0f0fu);
        }
    } else {
        #pragma unroll
        for (int f = 0; f < 4; f++) {
            op[f] = (int)((l[0] >> (2 * f)) & 0x03030303u);
            op[4 + f] = (int)((l[1] >> (2 * f)) & 0x03030303u);
        }
    }
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        unsigned int v = (unsigned int)op[j];
        if (HB > 0) {
            unsigned int b0 = (h0 >> j) & 0x01010101u;
            v |= b0 << (TY == 11 ? 2 : 4);
            if (HB == 8) v |= ((h1 >> j) & 0x01010101u) << 5;
        }
        if (TY == 2) v = __vsub4(v, 0x08080808u);
        else if (TY == 6) v = __vsub4(v, 0x10101010u);
        else if (TY == 42) v = __vsub4(v, 0x01010101u);
        else if (TY == 11) v = __vsub4(v, 0x04040404u);
        else if (TY == 14) v = __vsub4(v, 0x20202020u);
        else if (TY == 20 || TY == 23) v = iq4w(v);
        op[j] = (int)v;
    }
}

// One chunk of one row as loaded: low plane, extra-bit words, scale words. All of a step's loads
// are issued before any of them is used (and the next step's before this one's math), so a warp
// keeps two steps of weights in flight instead of waiting on the scale loads after the codes.
struct Raw { uint4 l; unsigned int h0, h1, s0, s1; };

// A row's planes: low codes, extra bits, scales.
struct RowP { const unsigned char* l; const unsigned char* h; const unsigned char* s; const unsigned char* d; };

__device__ __forceinline__ Raw load_raw(const RowP& p, unsigned int c) {
    Raw r;
    if (LB == 16) r.l = __ldg((const uint4*)p.l + c);
    else { uint2 l2 = __ldg((const uint2*)p.l + c); r.l = make_uint4(l2.x, l2.y, 0, 0); }
    r.h0 = r.h1 = 0;
    if (HB == 4) r.h0 = __ldg((const unsigned int*)p.h + c);
    if (HB == 8) { uint2 h = __ldg((const uint2*)p.h + c); r.h0 = h.x; r.h1 = h.y; }
    r.s1 = 0;
    const unsigned char* S = p.s;
    if (TY == 2 || TY == 6 || TY == 20) r.s0 = __ldg((const unsigned short*)S + c);
    else if (TY == 42) r.s0 = __ldg((const unsigned short*)S + c / 2);
    else {
        // Super-block types: the chunk's SC entry, then its 256-block's SD word.
        if (TY == 23) r.s1 = __ldg(S + c);
        else r.s1 = __ldg((const unsigned short*)S + c);
        r.s0 = __ldg((const unsigned int*)p.d + c / 8);
    }
    return r;
}

// Scales of chunk c from its scale words: sc0, sc1 (per half) and the min.
__device__ __forceinline__ void chunk_scales(const Raw& r, unsigned int c, float& sc0, float& sc1, float& mn) {
    mn = 0.0f;
    if (TY == 2 || TY == 6 || TY == 20 || TY == 42) {
        sc0 = sc1 = h2f((unsigned short)r.s0);
    } else if (TY == 23) {
        sc0 = sc1 = h2f((unsigned short)r.s0) * (float)(signed char)r.s1;
    } else if (TY == 11 || TY == 14) {
        float d = h2f((unsigned short)r.s0);
        sc0 = d * (float)(signed char)r.s1;
        sc1 = d * (float)(signed char)(r.s1 >> 8);
    } else {
        sc0 = sc1 = h2f((unsigned short)r.s0) * (float)(r.s1 & 0xffu);
        mn = h2f((unsigned short)(r.s0 >> 16)) * (float)(r.s1 >> 8);
    }
}

// Y[t, o] = W[o] · x̂[t]: GR rows a warp, one 32-weight chunk a lane per step, KS warps
// splitting K (8 / KS row groups a 256-thread block), as the other multi-column GEMVs. Blocks
// stride over the NB row tiles, the grid capped at what is co-resident (no partial last wave).
template <int T, int GR>
__device__ __forceinline__ void nat_tile(unsigned int bx, const unsigned int* __restrict__ XQ, const unsigned char* __restrict__ W,
                                         float* __restrict__ Y, unsigned int K, unsigned int N, unsigned int KS,
                                         u64 hoff, u64 soff, unsigned int LPR, float (*red)[2][GR * T]) {
    // Each LPR-lane group (a half-warp or the whole warp) takes GR rows; its lanes stride the
    // chunks. Half-warps when a warp's chunk range is not a whole number of 32-lane steps (K = 2560
    // is 80 chunks, five 16-lane steps).
    unsigned int lane = threadIdx.x % LPR, half = (threadIdx.x & 31) / LPR, warp = threadIdx.x >> 5, nh = 32 / LPR;
    unsigned int kw = warp % KS, rg = warp / KS, groups = 8 / KS;
    unsigned int row0 = (bx * groups + rg) * nh * GR + half * GR;
    unsigned int nch = K / 32, c0 = kw * nch / KS, c1 = (kw + 1) * nch / KS;
    const unsigned int kb = K / 4;
    const unsigned int* xs = XQ + (u64)T * kb;
    const unsigned int* xh = xs + (u64)T * nch;
    u64 sbytes = TY == 42 ? K / 64 * 2 : K / 32 * 2;
    float acc[GR][T];
    #pragma unroll
    for (int r = 0; r < GR; r++)
        #pragma unroll
        for (int t = 0; t < T; t++) acc[r][t] = 0.0f;
    if (row0 < N) {
        RowP rp[GR];
        #pragma unroll
        for (int r = 0; r < GR; r++) {
            u64 o = min(row0 + r, N - 1);
            rp[r].l = W + o * nch * LB;
            rp[r].h = W + hoff + o * nch * HB;
            if (SPC == 0) {
                rp[r].s = W + soff + o * sbytes;
                rp[r].d = rp[r].s;
            } else {
                rp[r].s = W + soff + o * nch * SPC;
                rp[r].d = W + soff + (u64)N * nch * SPC + o * (K / 256) * 4;
            }
        }
        unsigned int cc = c0 + lane;
        constexpr bool PF = NAT_PF(T);
        Raw cur[GR];
        if (PF && cc < c1) {
            #pragma unroll
            for (int r = 0; r < GR; r++) cur[r] = load_raw(rp[r], cc);
        }
        for (; cc < c1; cc += LPR) {
            unsigned int cn = cc + LPR;
            Raw nxt[GR];
            if (!PF) {
                #pragma unroll
                for (int r = 0; r < GR; r++) cur[r] = load_raw(rp[r], cc);
            } else if (cn < c1) {
                #pragma unroll
                for (int r = 0; r < GR; r++) nxt[r] = load_raw(rp[r], cn);
            }
            const unsigned int c = cc;
#ifdef NAT_LOADONLY
            #pragma unroll
            for (int r = 0; r < GR; r++)
                acc[r][0] += __uint_as_float((cur[r].l.x ^ cur[r].l.y ^ cur[r].l.z ^ cur[r].l.w ^ cur[r].h0 ^ cur[r].h1 ^ cur[r].s0 ^ cur[r].s1) & 0x3fffffffu);
#else
            int op[GR][8];
            float sc0[GR], sc1[GR], mn[GR];
            #pragma unroll
            for (int r = 0; r < GR; r++) {
                unsigned int lw[4] = {cur[r].l.x, cur[r].l.y, cur[r].l.z, cur[r].l.w};
                operands(lw, cur[r].h0, cur[r].h1, op[r]);
                chunk_scales(cur[r], c, sc0[r], sc1[r], mn[r]);
            }
            #pragma unroll
            for (int t = 0; t < T; t++) {
                const uint4* xw = (const uint4*)(XQ + (u64)t * kb + c * 8);
                uint4 xa = xw[0], xb = xw[1];
                float dx = __uint_as_float(xs[(u64)t * nch + c]);
                float hx = HAS_MIN ? (float)(int)xh[(u64)t * nch + c] : 0.0f;
                #pragma unroll
                for (int r = 0; r < GR; r++) {
                    int s0 = __dp4a(op[r][0], (int)xa.x, 0);
                    s0 = __dp4a(op[r][1], (int)xa.y, s0);
                    s0 = __dp4a(op[r][2], (int)xa.z, s0);
                    s0 = __dp4a(op[r][3], (int)xa.w, s0);
                    int s1 = __dp4a(op[r][4], (int)xb.x, 0);
                    s1 = __dp4a(op[r][5], (int)xb.y, s1);
                    s1 = __dp4a(op[r][6], (int)xb.z, s1);
                    s1 = __dp4a(op[r][7], (int)xb.w, s1);
                    float v;
                    if (PER16) v = __fmaf_rn(sc1[r], (float)s1, sc0[r] * (float)s0);
                    else if (HAS_MIN) v = __fmaf_rn(sc0[r], (float)(s0 + s1), -(mn[r] * hx));
                    else v = sc0[r] * (float)(s0 + s1);
                    acc[r][t] = __fmaf_rn(dx, v, acc[r][t]);
                }
            }
#endif
            #pragma unroll
            for (int r = 0; r < GR; r++) if (PF) cur[r] = nxt[r];
        }
    }
    #pragma unroll
    for (int r = 0; r < GR; r++)
        #pragma unroll
        for (int t = 0; t < T; t++) {
            float v = acc[r][t];
            for (unsigned int o = LPR / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (KS == 1) {
                if (lane == 0 && row0 + r < N) Y[(u64)t * N + row0 + r] = v;
            } else if (lane == 0) {
                red[warp][half][r * T + t] = v;
            }
        }
    if (KS == 1) return;
    __syncthreads();
    unsigned int i = threadIdx.x;
    if (i < groups * nh * GR * T) {
        unsigned int g = i / (nh * GR * T), hrt = i % (nh * GR * T), hf = hrt / (GR * T), rt = hrt % (GR * T);
        unsigned int r = rt / T, t = rt % T;
        unsigned int o = (bx * groups + g) * nh * GR + hf * GR + r;
        float v = 0.0f;
        for (unsigned int k = 0; k < KS; k++) v += red[g * KS + k][hf][rt];
        if (o < N) Y[(u64)t * N + o] = v;
    }
}

#ifndef NAT_GR
#define NAT_GR(T) (T == 1 ? 2 : 4)
#endif
#define NAT(T) \
extern "C" __global__ void __launch_bounds__(256) fl_nat_t##T( \
    const unsigned int* __restrict__ XQ, const unsigned char* __restrict__ W, float* __restrict__ Y, \
    unsigned int K, unsigned int N, unsigned int KS, u64 hoff, u64 soff, unsigned int LPR, unsigned int NB) { \
    constexpr int GR = NAT_GR(T); \
    __shared__ float red[8][2][GR * T]; \
    for (unsigned int bx = blockIdx.x; bx < NB; bx += gridDim.x) { \
        if (bx != blockIdx.x) __syncthreads(); \
        nat_tile<T, GR>(bx, XQ, W, Y, K, N, KS, hoff, soff, LPR, red); \
    } \
}
NAT(1) NAT(2) NAT(3) NAT(4) NAT(5) NAT(6) NAT(7) NAT(8)
"#;
