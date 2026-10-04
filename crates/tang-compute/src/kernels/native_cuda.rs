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
// Per-chunk scale bytes of the super-block types' SC plane.
#define SPC ((TY == 12 || TY == 13) ? 2 : 0)

// The eight dp4a operands (int8 lanes in activation order) of one chunk.
__device__ __forceinline__ void operands(const unsigned char* L, const unsigned char* H, int op[8]) {
    if (LB == 16) {
        uint4 q = *(const uint4*)L;
        unsigned int w[4] = {q.x, q.y, q.z, q.w};
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            op[2 * i] = (int)(w[i] & 0x0f0f0f0fu);
            op[2 * i + 1] = (int)((w[i] >> 4) & 0x0f0f0f0fu);
        }
    } else {
        uint2 q = *(const uint2*)L;
        #pragma unroll
        for (int f = 0; f < 4; f++) {
            op[f] = (int)((q.x >> (2 * f)) & 0x03030303u);
            op[4 + f] = (int)((q.y >> (2 * f)) & 0x03030303u);
        }
    }
    unsigned int h0 = 0, h1 = 0;
    if (HB == 4) h0 = *(const unsigned int*)H;
    if (HB == 8) { uint2 hh = *(const uint2*)H; h0 = hh.x; h1 = hh.y; }
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

// Scales of chunk c of row o (global chunk g = o · K/32 + c): sc0, sc1 (per half) and the min.
// S is the scale plane: per-block f16 (Q4_0 / Q5_0 / IQ4_NL / Q2_0), 12 / 20 B headers per 256
// (IQ4_XS, Q3_K, Q6_K; row stride sbytes), or for Q4_K / Q5_K the per-chunk SC plane with D the
// per-256 SD plane (module docs of flash_native).
__device__ __forceinline__ void chunk_scales(const unsigned char* S, const unsigned char* D, u64 sbytes, unsigned int o,
                                             unsigned int c, u64 g, float& sc0, float& sc1, float& mn) {
    mn = 0.0f;
    if (TY == 2 || TY == 6 || TY == 20) {
        sc0 = sc1 = h2f(__ldg((const unsigned short*)S + g));
    } else if (TY == 42) {
        sc0 = sc1 = h2f(__ldg((const unsigned short*)S + g / 2));
    } else if (TY == 23) {
        const unsigned char* hd = S + o * sbytes + (c / 8) * 12;
        sc0 = sc1 = h2f(*(const unsigned short*)hd) * (float)(signed char)hd[4 + c % 8];
    } else if (TY == 11 || TY == 14) {
        const unsigned char* hd = S + o * sbytes + (c / 8) * 20;
        float d = h2f(*(const unsigned short*)hd);
        sc0 = d * (float)(signed char)hd[4 + 2 * (c % 8)];
        sc1 = d * (float)(signed char)hd[5 + 2 * (c % 8)];
    } else {
        unsigned int w = __ldg((const unsigned short*)S + g);
        unsigned int dd = __ldg((const unsigned int*)D + g / 8);
        sc0 = sc1 = h2f((unsigned short)dd) * (float)(w & 0xffu);
        mn = h2f((unsigned short)(dd >> 16)) * (float)(w >> 8);
    }
}

// Y[t, o] = W[o] · x̂[t]: GR rows a warp, one 32-weight chunk a lane per step, KS warps
// splitting K (8 / KS row groups a 256-thread block), as the other multi-column GEMVs.
template <int T, int GR>
__device__ __forceinline__ void nat_body(const unsigned int* __restrict__ XQ, const unsigned char* __restrict__ W,
                                         float* __restrict__ Y, unsigned int K, unsigned int N, unsigned int KS,
                                         u64 hoff, u64 soff) {
    __shared__ float red[8][GR * T];
    unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    unsigned int kw = warp % KS, rg = warp / KS, groups = 8 / KS;
    unsigned int row0 = (blockIdx.x * groups + rg) * GR;
    unsigned int nch = K / 32, c0 = kw * nch / KS, c1 = (kw + 1) * nch / KS;
    const unsigned int kb = K / 4;
    const unsigned int* xs = XQ + (u64)T * kb;
    const unsigned int* xh = xs + (u64)T * nch;
    const unsigned char* S = W + soff;
    const unsigned char* D = S + (u64)N * nch * SPC;
    const u64 sbytes = TY == 23 ? K / 256 * 12 : K / 256 * 20;
    float acc[GR][T];
    #pragma unroll
    for (int r = 0; r < GR; r++)
        #pragma unroll
        for (int t = 0; t < T; t++) acc[r][t] = 0.0f;
    if (row0 < N) {
        for (unsigned int c = c0 + lane; c < c1; c += 32) {
            int op[GR][8];
            float sc0[GR], sc1[GR], mn[GR];
            #pragma unroll
            for (int r = 0; r < GR; r++) {
                unsigned int o = min(row0 + r, N - 1);
                u64 ch = (u64)o * nch + c;
                operands(W + ch * LB, W + hoff + ch * (HB ? HB : 1), op[r]);
                chunk_scales(S, D, sbytes, o, c, ch, sc0[r], sc1[r], mn[r]);
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

#define NAT(T) \
extern "C" __global__ void __launch_bounds__(256) fl_nat_t##T( \
    const unsigned int* __restrict__ XQ, const unsigned char* __restrict__ W, float* __restrict__ Y, \
    unsigned int K, unsigned int N, unsigned int KS, u64 hoff, u64 soff) { \
    nat_body<T, (T == 1 ? 2 : 4)>(XQ, W, Y, K, N, KS, hoff, soff); \
}
NAT(1) NAT(2) NAT(3) NAT(4) NAT(5) NAT(6) NAT(7) NAT(8)
"#;
