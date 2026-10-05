//! CUDA source for the native-type (NatX, `crate::flash_native`) int8-activation GEMV. One
//! module per weight type (`#define NAT_TY <ggml id>` prepended), so only the types a model
//! uses are compiled; kernels `fl_nat_t<T>`, T = 1..8 columns. With `#define NAT_STACK`, the
//! module instead holds `fl_natst_t<T>`: every segment of a stacked projection (any mix of types,
//! bf16 included) in one launch.

/// The kernel source (see the module docs for its two forms).
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

#define LB ((TY == 42 || TY == 11) ? 8 : 16)
#define HB ((TY == 6 || TY == 13 || TY == 11) ? 4 : (TY == 14 ? 8 : 0))
#define HAS_MIN (TY == 12 || TY == 13)
#define PER16 (TY == 11 || TY == 14)
// Per-chunk scale bytes of the super-block types' SC plane.
#define SPC ((TY == 12 || TY == 13) ? 2 : 0)

// Rows a warp (NatType::rows_per_warp): 2 for one column; for more, per type. Bits don't depend
// on it (the K split is chosen from (n, k) alone); throughput does.
// Measured at T=4 (flash-kernel-bench native): 2 rows for the IQ4 types and Q2_0 (+30..80 GB/s),
// 4 for the rest (+40..150).
#define NAT_GR_WIDE(ty) (((ty) == 20 || (ty) == 23 || (ty) == 42) ? 2 : 4)
__host__ __device__ constexpr int nat_gr(int ty, int t) { return t == 1 ? 2 : (t <= 8 ? NAT_GR_WIDE(ty) : 1); }

// The eight dp4a operands (int8 lanes in activation order) of one chunk.
template <int TY>
__device__ __forceinline__ void operands(const unsigned char* L, const unsigned char* H, int op[8]) {
    if (TY == 20 || TY == 23) {
        // IQ4: both nibble words of a code word through the 16-entry grid with byte permutes
        // (llama.cpp's get_int_from_table_16): the low 3 bits pick among 8 entries, bit 3
        // between the two halves of the table.
        uint4 q = *(const uint4*)L;
        unsigned int w[4] = {q.x, q.y, q.z, q.w};
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const unsigned int hl = 0x32103210u | ((w[i] & 0x88888888u) >> 1);
            unsigned int tmp[2];
            #pragma unroll
            for (int k = 0; k < 2; k++) {
                const unsigned int sh = 16 * k;
                const unsigned int lo = __byte_perm(0xBFAD9881u, 0xF6EADDCFu, w[i] >> sh);
                const unsigned int hi = __byte_perm(0x26190D01u, 0x71594535u, w[i] >> sh);
                tmp[k] = __byte_perm(lo, hi, hl >> sh);
            }
            op[2 * i] = (int)__byte_perm(tmp[0], tmp[1], 0x6420);
            op[2 * i + 1] = (int)__byte_perm(tmp[0], tmp[1], 0x7531);
        }
        return;
    }
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
        // Stored value v (unsigned, below 0x80) minus the type's offset as signed bytes:
        // (v + 0x80 - off) ^ 0x80, no carries between bytes.
        if (TY == 2) v = (v + 0x78787878u) ^ 0x80808080u;
        else if (TY == 6) v = (v + 0x70707070u) ^ 0x80808080u;
        else if (TY == 42) v = (v + 0x7F7F7F7Fu) ^ 0x80808080u;
        else if (TY == 11) v = (v + 0x7C7C7C7Cu) ^ 0x80808080u;
        else if (TY == 14) v = (v + 0x60606060u) ^ 0x80808080u;
        op[j] = (int)v;
    }
}

// Scales of chunk c of row o (global chunk g = o · K/32 + c): sc0, sc1 (per half) and the min.
// S is the scale plane: per-block f16 (Q4_0 / Q5_0 / IQ4_NL / Q2_0), 12 / 20 B headers per 256
// (IQ4_XS, Q3_K, Q6_K; row stride sbytes), or for Q4_K / Q5_K the per-chunk SC plane with D the
// per-256 SD plane (module docs of flash_native).
template <int TY>
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
template <int TY, int T, int GR>
__device__ __forceinline__ void nat_body(unsigned int bx, const unsigned int* __restrict__ XQ, const unsigned char* __restrict__ W,
                                         float* __restrict__ Y, unsigned int K, unsigned int N, unsigned int KS,
                                         u64 hoff, u64 soff, unsigned int OS, unsigned int M = T, unsigned int t0 = 0) {
    // Columns t0 .. t0 + T of an M-column window (a token tile); a column past M is computed on
    // column M - 1 and not written.
    __shared__ float red[8][GR * T];
    unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    unsigned int kw = warp % KS, rg = warp / KS, groups = 8 / KS;
    unsigned int row0 = (bx * groups + rg) * GR;
    unsigned int nch = K / 32, c0 = kw * nch / KS, c1 = (kw + 1) * nch / KS;
    const unsigned int kb = K / 4;
    const unsigned int* xs = XQ + (u64)M * kb;
    const unsigned int* xh = xs + (u64)M * nch;
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
                operands<TY>(W + ch * LB, W + hoff + ch * (HB ? HB : 1), op[r]);
                chunk_scales<TY>(S, D, sbytes, o, c, ch, sc0[r], sc1[r], mn[r]);
            }
            #pragma unroll
            for (int t = 0; t < T; t++) {
                const unsigned int tc = min(t0 + t, M - 1);
                const uint4* xw = (const uint4*)(XQ + (u64)tc * kb + c * 8);
                uint4 xa = xw[0], xb = xw[1];
                float dx = __uint_as_float(xs[(u64)tc * nch + c]);
                float hx = HAS_MIN ? (float)(int)xh[(u64)tc * nch + c] : 0.0f;
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
                if (lane == 0 && row0 + r < N && t0 + t < M) Y[(u64)(t0 + t) * OS + row0 + r] = v;
            } else if (lane == 0) {
                red[warp][r * T + t] = v;
            }
        }
    if (KS == 1) return;
    __syncthreads();
    unsigned int i = threadIdx.x;
    if (i < groups * GR * T) {
        unsigned int g = i / (GR * T), rt = i % (GR * T), r = rt / T, t = rt % T;
        unsigned int o = (bx * groups + g) * GR + r;
        float v = 0.0f;
        for (unsigned int k = 0; k < KS; k++) v += red[g * KS + k][rt];
        if (o < N && t0 + t < M) Y[(u64)(t0 + t) * OS + o] = v;
    }
}

#ifndef NAT_STACK
#define NAT(T) \
extern "C" __global__ void __launch_bounds__(256) fl_nat_t##T( \
    const unsigned int* __restrict__ XQ, const unsigned char* __restrict__ W, float* __restrict__ Y, \
    unsigned int K, unsigned int N, unsigned int KS, u64 hoff, u64 soff, unsigned int OS) { \
    nat_body<NAT_TY, T, nat_gr(NAT_TY, T)>(blockIdx.x, XQ, W, Y, K, N, KS, hoff, soff, OS); \
}
NAT(1) NAT(2) NAT(3) NAT(4) NAT(5) NAT(6) NAT(7) NAT(8)
// Wide windows (prefill): token tiles of 8 in turn over the same rows (the tile's weights are
// re-read from L1/L2, not DRAM); per-token arithmetic is the T=8 kernel's.
extern "C" __global__ void __launch_bounds__(256) fl_natw(
    const unsigned int* __restrict__ XQ, const unsigned char* __restrict__ W, float* __restrict__ Y,
    unsigned int K, unsigned int N, unsigned int KS, u64 hoff, u64 soff, unsigned int OS, unsigned int M) {
    for (unsigned int t0 = 0; t0 < M; t0 += 8) {
        if (t0) __syncthreads();
        nat_body<NAT_TY, 8, nat_gr(NAT_TY, 8)>(blockIdx.x, XQ, W, Y, K, N, KS, hoff, soff, OS, M, t0);
    }
}
// Variants of fl_natw: token tile TT, GR rows a warp (A/B: TANG_FLASH_NATW=TTxGR).
#define NATWV(TT, GR) \
extern "C" __global__ void __launch_bounds__(256) fl_natw_##TT##x##GR( \
    const unsigned int* __restrict__ XQ, const unsigned char* __restrict__ W, float* __restrict__ Y, \
    unsigned int K, unsigned int N, unsigned int KS, u64 hoff, u64 soff, unsigned int OS, unsigned int M) { \
    for (unsigned int t0 = 0; t0 < M; t0 += TT) { \
        if (t0) __syncthreads(); \
        nat_body<NAT_TY, TT, GR>(blockIdx.x, XQ, W, Y, K, N, KS, hoff, soff, OS, M, t0); \
    } \
}
NATWV(8, 2) NATWV(8, 4) NATWV(8, 8) NATWV(16, 2) NATWV(16, 4) NATWV(12, 4) NATWV(32, 1) NATWV(32, 2)
// Wide windows on int8 tensor cores (prefill). A warp owns 16 rows (mma m16) of its K slice;
// tokens go in tiles of 32 (four n8 tiles). The integer chunk sums come from mma.sync (exact);
// the float steps are nat_body's: per (row, token) the chunks a lane would own (c0 + l + 32j,
// "slot" l) chain in order from 0, and the 32 slot partials merge as warp_sum's xor tree does
// (bit-reversed slot order, a binary-counter stack), then the KS sum in order. Bits equal the
// T=1..8 kernels'.
__device__ __forceinline__ void mma_k32(int c[4], const unsigned int a[4], const unsigned int b[2]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(0), "r"(0), "r"(0), "r"(0));
}
__device__ __forceinline__ void mma_k16(int c[4], unsigned int a0, unsigned int a1, unsigned int b0) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(0), "r"(0), "r"(0), "r"(0));
}
#define NM_NT 4
extern "C" __global__ void __launch_bounds__(256) fl_natmma(
    const unsigned int* __restrict__ XQ, const unsigned char* __restrict__ W, float* __restrict__ Y,
    unsigned int K, unsigned int N, unsigned int KS, u64 hoff, u64 soff, unsigned int OS, unsigned int M) {
    constexpr int TY = NAT_TY;
    __shared__ float red[8][16][8 * NM_NT];
    const unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, gid = lane >> 2, tig = lane & 3;
    const unsigned int kw = warp % KS, rg = warp / KS, groups = 8 / KS;
    const unsigned int row0 = (blockIdx.x * groups + rg) * 16;
    const unsigned int nch = K / 32, c0 = kw * nch / KS, c1 = (kw + 1) * nch / KS;
    const unsigned int kb = K / 4;
    const unsigned int* xs = XQ + (u64)M * kb;
    const unsigned int* xh = xs + (u64)M * nch;
    const unsigned char* S = W + soff;
    const unsigned char* D = S + (u64)N * nch * SPC;
    const u64 sbytes = TY == 23 ? K / 256 * 12 : K / 256 * 20;
    const unsigned int o[2] = {min(row0 + gid, N - 1), min(row0 + gid + 8, N - 1)};
    for (unsigned int t0 = 0; t0 < M; t0 += 8 * NM_NT) {
        // Tokens of this thread: B column gid of n tile nt; C columns 2 tig, 2 tig + 1.
        unsigned int tb[NM_NT], te[NM_NT][2];
        #pragma unroll
        for (int nt = 0; nt < NM_NT; nt++) {
            tb[nt] = min(t0 + 8 * nt + gid, M - 1);
            te[nt][0] = min(t0 + 8 * nt + 2 * tig, M - 1);
            te[nt][1] = min(t0 + 8 * nt + 2 * tig + 1, M - 1);
        }
        float st[5][NM_NT][4];
        float v[NM_NT][4];
        #pragma unroll 1
        for (unsigned int i = 0; i < 32; i++) {
            const unsigned int l = __brev(i) >> 27;
            float cur[NM_NT][4];
            #pragma unroll
            for (int nt = 0; nt < NM_NT; nt++)
                #pragma unroll
                for (int k = 0; k < 4; k++) cur[nt][k] = 0.0f;
            if (row0 < N) {
                #pragma unroll 1
                for (unsigned int c = c0 + l; c < c1; c += 32) {
                    int op[2][8];
                    float sc0[2], sc1[2], mn[2];
                    #pragma unroll
                    for (int r = 0; r < 2; r++) {
                        u64 ch = (u64)o[r] * nch + c;
                        operands<TY>(W + ch * LB, W + hoff + ch * (HB ? HB : 1), op[r]);
                        chunk_scales<TY>(S, D, sbytes, o[r], c, ch, sc0[r], sc1[r], mn[r]);
                    }
                    #pragma unroll
                    for (int nt = 0; nt < NM_NT; nt++) {
                        const unsigned int* xb = XQ + (u64)tb[nt] * kb + c * 8;
                        unsigned int b[2] = {__ldg(xb + tig), __ldg(xb + 4 + tig)};
                        int C0[4], C1[4];
                        if (PER16) {
                            mma_k16(C0, (unsigned int)op[0][tig], (unsigned int)op[1][tig], b[0]);
                            mma_k16(C1, (unsigned int)op[0][4 + tig], (unsigned int)op[1][4 + tig], b[1]);
                        } else {
                            unsigned int a[4] = {(unsigned int)op[0][tig], (unsigned int)op[1][tig],
                                                 (unsigned int)op[0][4 + tig], (unsigned int)op[1][4 + tig]};
                            mma_k32(C0, a, b);
                        }
                        #pragma unroll
                        for (int ec = 0; ec < 2; ec++) {
                            const unsigned int t = te[nt][ec];
                            float dx = __uint_as_float(__ldg(xs + (u64)t * nch + c));
                            float hx = HAS_MIN ? (float)(int)__ldg(xh + (u64)t * nch + c) : 0.0f;
                            #pragma unroll
                            for (int r = 0; r < 2; r++) {
                                float vv;
                                if (PER16) vv = __fmaf_rn(sc1[r], (float)C1[2 * r + ec], sc0[r] * (float)C0[2 * r + ec]);
                                else if (HAS_MIN) vv = __fmaf_rn(sc0[r], (float)C0[2 * r + ec], -(mn[r] * hx));
                                else vv = sc0[r] * (float)C0[2 * r + ec];
                                cur[nt][2 * r + ec] = __fmaf_rn(dx, vv, cur[nt][2 * r + ec]);
                            }
                        }
                    }
                }
            }
            // Binary-counter merge: slot partials pair up as warp_sum's xor tree.
            #pragma unroll
            for (int nt = 0; nt < NM_NT; nt++)
                #pragma unroll
                for (int k = 0; k < 4; k++) v[nt][k] = cur[nt][k];
            #pragma unroll
            for (int lv = 0; lv < 5; lv++) {
                if ((i >> lv) & 1) {
                    #pragma unroll
                    for (int nt = 0; nt < NM_NT; nt++)
                        #pragma unroll
                        for (int k = 0; k < 4; k++) v[nt][k] = st[lv][nt][k] + v[nt][k];
                } else {
                    #pragma unroll
                    for (int nt = 0; nt < NM_NT; nt++)
                        #pragma unroll
                        for (int k = 0; k < 4; k++) st[lv][nt][k] = v[nt][k];
                    break;
                }
            }
        }
        // v now holds the full warp sums (after i = 31).
        if (KS > 1) __syncthreads();
        #pragma unroll
        for (int nt = 0; nt < NM_NT; nt++)
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                const unsigned int r = gid + 8 * (k >> 1), tl = 8 * nt + 2 * tig + (k & 1);
                if (KS == 1) {
                    if (row0 + r < N && t0 + tl < M) Y[(u64)(t0 + tl) * OS + row0 + r] = v[nt][k];
                } else {
                    red[warp][r][tl] = v[nt][k];
                }
            }
        if (KS == 1) continue;
        __syncthreads();
        for (unsigned int idx = threadIdx.x; idx < groups * 16 * 8 * NM_NT; idx += 256) {
            unsigned int g = idx / (16 * 8 * NM_NT), r = (idx / (8 * NM_NT)) % 16, tl = idx % (8 * NM_NT);
            unsigned int oo = (blockIdx.x * groups + g) * 16 + r;
            float sum = 0.0f;
            for (unsigned int k = 0; k < KS; k++) sum += red[g * KS + k][r][tl];
            if (oo < N && t0 + tl < M) Y[(u64)(t0 + tl) * OS + oo] = sum;
        }
    }
}

// Wide windows, decoded-weight reuse: one row a warp; a lane decodes NB of its chunks into
// registers once, then runs them over every column of the launch (up to 32, accumulators in
// shared memory, TT columns at a time in registers). Each column's chain is still its chunks in
// order, then warp_sum, then the KS sum: bits equal the T=1..8 kernels'.
#define NW_NB 4
#define NW_TT 16
#define NW_MAX 32
extern "C" __global__ void __launch_bounds__(256) fl_natw2(
    const unsigned int* __restrict__ XQ, const unsigned char* __restrict__ W, float* __restrict__ Y,
    unsigned int K, unsigned int N, unsigned int KS, u64 hoff, u64 soff, unsigned int OS, unsigned int M,
    unsigned int t0, unsigned int MT) {
    constexpr int TY = NAT_TY;
    __shared__ float accs[NW_MAX][256];
    __shared__ float red[8][NW_MAX];
    unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    unsigned int kw = warp % KS, rg = warp / KS, groups = 8 / KS;
    unsigned int o = blockIdx.x * groups + rg;
    unsigned int nch = K / 32, c0 = kw * nch / KS, c1 = (kw + 1) * nch / KS;
    const unsigned int kb = K / 4;
    const unsigned int* xs = XQ + (u64)M * kb;
    const unsigned int* xh = xs + (u64)M * nch;
    const unsigned char* S = W + soff;
    const unsigned char* D = S + (u64)N * nch * SPC;
    const u64 sbytes = TY == 23 ? K / 256 * 12 : K / 256 * 20;
    for (unsigned int t = 0; t < MT; t++) accs[t][threadIdx.x] = 0.0f;
    unsigned int orow = min(o, N - 1);
    if (o < N) {
        for (unsigned int cb = c0 + lane; cb < c1; cb += 32 * NW_NB) {
            int op[NW_NB][8];
            float sc0[NW_NB], sc1[NW_NB], mn[NW_NB];
            #pragma unroll
            for (int b = 0; b < NW_NB; b++) {
                unsigned int c = min(cb + b * 32, c1 - 1);
                u64 ch = (u64)orow * nch + c;
                operands<TY>(W + ch * LB, W + hoff + ch * (HB ? HB : 1), op[b]);
                chunk_scales<TY>(S, D, sbytes, orow, c, ch, sc0[b], sc1[b], mn[b]);
            }
            for (unsigned int tt = 0; tt < MT; tt += NW_TT) {
                float acc[NW_TT];
                #pragma unroll
                for (int t = 0; t < NW_TT; t++) acc[t] = tt + t < MT ? accs[tt + t][threadIdx.x] : 0.0f;
                #pragma unroll
                for (int b = 0; b < NW_NB; b++) {
                    unsigned int c = cb + b * 32;
                    if (c >= c1) break;
                    #pragma unroll
                    for (int t = 0; t < NW_TT; t++) {
                        const unsigned int tc = t0 + min(tt + t, MT - 1);
                        const uint4* xw = (const uint4*)(XQ + (u64)tc * kb + c * 8);
                        uint4 xa = xw[0], xb = xw[1];
                        float dx = __uint_as_float(xs[(u64)tc * nch + c]);
                        float hx = HAS_MIN ? (float)(int)xh[(u64)tc * nch + c] : 0.0f;
                        int s0 = __dp4a(op[b][0], (int)xa.x, 0);
                        s0 = __dp4a(op[b][1], (int)xa.y, s0);
                        s0 = __dp4a(op[b][2], (int)xa.z, s0);
                        s0 = __dp4a(op[b][3], (int)xa.w, s0);
                        int s1 = __dp4a(op[b][4], (int)xb.x, 0);
                        s1 = __dp4a(op[b][5], (int)xb.y, s1);
                        s1 = __dp4a(op[b][6], (int)xb.z, s1);
                        s1 = __dp4a(op[b][7], (int)xb.w, s1);
                        float v;
                        if (PER16) v = __fmaf_rn(sc1[b], (float)s1, sc0[b] * (float)s0);
                        else if (HAS_MIN) v = __fmaf_rn(sc0[b], (float)(s0 + s1), -(mn[b] * hx));
                        else v = sc0[b] * (float)(s0 + s1);
                        acc[t] = __fmaf_rn(dx, v, acc[t]);
                    }
                }
                #pragma unroll
                for (int t = 0; t < NW_TT; t++)
                    if (tt + t < MT) accs[tt + t][threadIdx.x] = acc[t];
            }
        }
    }
    for (unsigned int t = 0; t < MT; t++) {
        float v = warp_sum(accs[t][threadIdx.x]);
        if (KS == 1) {
            if (lane == 0 && o < N) Y[(u64)(t0 + t) * OS + o] = v;
        } else if (lane == 0) {
            red[warp][t] = v;
        }
    }
    if (KS == 1) return;
    __syncthreads();
    for (unsigned int i = threadIdx.x; i < groups * MT; i += 256) {
        unsigned int g = i / MT, t = i % MT;
        unsigned int oo = blockIdx.x * groups + g;
        float v = 0.0f;
        for (unsigned int k = 0; k < KS; k++) v += red[g * KS + k][t];
        if (oo < N) Y[(u64)(t0 + t) * OS + oo] = v;
    }
}
// Prefill-only widths (flash-serve).
NAT(16) NAT(32) NAT(64)
#else
// ---- stacked: every segment of a stacked projection in one launch ----

__device__ __forceinline__ float bfv(unsigned int b) { return __uint_as_float(b << 16); }
// acc + (w · (a, b)) for 8 bf16 weights in p, one pinned chain (flash_cuda's dot8_rn).
__device__ __forceinline__ float dot8_rn(uint4 p, float4 a, float4 b, float acc) {
    float s = __fmul_rn(bfv(p.x & 0xffff), a.x);
    s = __fmaf_rn(bfv(p.x >> 16), a.y, s); s = __fmaf_rn(bfv(p.y & 0xffff), a.z, s);
    s = __fmaf_rn(bfv(p.y >> 16), a.w, s); s = __fmaf_rn(bfv(p.z & 0xffff), b.x, s);
    s = __fmaf_rn(bfv(p.z >> 16), b.y, s); s = __fmaf_rn(bfv(p.w & 0xffff), b.z, s);
    s = __fmaf_rn(bfv(p.w >> 16), b.w, s);
    return __fadd_rn(acc, s);
}

// The bf16 GEMV (flash_cuda gemv_body<0>: f32 activations, 8 weights a lane step), one tile.
template <int T, int GR>
__device__ __forceinline__ void bf16_body(unsigned int bx, const float* __restrict__ X, const unsigned char* __restrict__ W,
                                          float* __restrict__ Y, unsigned int K, unsigned int N, unsigned int KS,
                                          unsigned int OS) {
    __shared__ float red[8][GR * T];
    unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    unsigned int kw = warp % KS, rg = warp / KS, groups = 8 / KS;
    unsigned int row0 = (bx * groups + rg) * GR;
    unsigned int nv = K / 8, v0 = kw * nv / KS, v1 = (kw + 1) * nv / KS;
    float acc[GR][T];
    #pragma unroll
    for (int r = 0; r < GR; r++)
        #pragma unroll
        for (int t = 0; t < T; t++) acc[r][t] = 0.0f;
    if (row0 < N) {
        for (unsigned int v = v0 + lane; v < v1; v += 32) {
            uint4 wv[GR];
            #pragma unroll
            for (int r = 0; r < GR; r++) {
                unsigned int o = min(row0 + r, N - 1);
                wv[r] = ((const uint4*)(W + (u64)o * K * 2))[v];
            }
            #pragma unroll
            for (int t = 0; t < T; t++) {
                const float4* x4 = (const float4*)(X + (u64)t * K + v * 8);
                float4 a = x4[0], b = x4[1];
                #pragma unroll
                for (int r = 0; r < GR; r++) acc[r][t] = dot8_rn(wv[r], a, b, acc[r][t]);
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < GR; r++)
        #pragma unroll
        for (int t = 0; t < T; t++) {
            float v = warp_sum(acc[r][t]);
            if (KS == 1) {
                if (lane == 0 && row0 + r < N) Y[(u64)t * OS + row0 + r] = v;
            } else if (lane == 0) {
                red[warp][r * T + t] = v;
            }
        }
    if (KS == 1) return;
    __syncthreads();
    unsigned int i = threadIdx.x;
    if (i < groups * GR * T) {
        unsigned int g = i / (GR * T), rt = i % (GR * T), r = rt / T, t = rt % T;
        unsigned int o = (bx * groups + g) * GR + r;
        float v = 0.0f;
        for (unsigned int k = 0; k < KS; k++) v += red[g * KS + k][rt];
        if (o < N) Y[(u64)t * OS + o] = v;
    }
}

// Segment table entry (flash::NatStack): 8 words.
struct StSeg { u64 w; u64 hoff; u64 soff; unsigned int ty, n, ks, off; };

// Blocks of segment s at window width T: rows per block (2 or 4 rows a warp) · 8 / KS.
template <int T>
__device__ __forceinline__ unsigned int st_blocks(const StSeg& s) {
    unsigned int gr = s.ty == 0 ? (T == 1 ? 2 : (T <= 8 ? 4 : 1)) : nat_gr(s.ty, T);
    return (s.n + gr * 8 / s.ks - 1) / (gr * 8 / s.ks);
}

template <int T>
__device__ __forceinline__ void stack_body(const unsigned int* __restrict__ XQ, const float* __restrict__ X,
                                           const StSeg* __restrict__ SEGS, unsigned int nseg, float* __restrict__ Y,
                                           unsigned int K, unsigned int OS) {
    constexpr int GB = T == 1 ? 2 : (T <= 8 ? 4 : 1);  // the bf16 tile's rows a warp (gemv_rows)
    unsigned int b = blockIdx.x, s = 0;
    for (; s < nseg; s++) {
        unsigned int nb = st_blocks<T>(SEGS[s]);
        if (b < nb) break;
        b -= nb;
    }
    if (s >= nseg) return;
    const StSeg sg = SEGS[s];
    const unsigned char* W = (const unsigned char*)sg.w;
    float* y = Y + sg.off;
    switch (sg.ty) {
        case 2: nat_body<2, T, nat_gr(2, T)>(b, XQ, W, y, K, sg.n, sg.ks, sg.hoff, sg.soff, OS); break;
        case 6: nat_body<6, T, nat_gr(6, T)>(b, XQ, W, y, K, sg.n, sg.ks, sg.hoff, sg.soff, OS); break;
        case 42: nat_body<42, T, nat_gr(42, T)>(b, XQ, W, y, K, sg.n, sg.ks, sg.hoff, sg.soff, OS); break;
        case 20: nat_body<20, T, nat_gr(20, T)>(b, XQ, W, y, K, sg.n, sg.ks, sg.hoff, sg.soff, OS); break;
        case 23: nat_body<23, T, nat_gr(23, T)>(b, XQ, W, y, K, sg.n, sg.ks, sg.hoff, sg.soff, OS); break;
        case 11: nat_body<11, T, nat_gr(11, T)>(b, XQ, W, y, K, sg.n, sg.ks, sg.hoff, sg.soff, OS); break;
        case 12: nat_body<12, T, nat_gr(12, T)>(b, XQ, W, y, K, sg.n, sg.ks, sg.hoff, sg.soff, OS); break;
        case 13: nat_body<13, T, nat_gr(13, T)>(b, XQ, W, y, K, sg.n, sg.ks, sg.hoff, sg.soff, OS); break;
        case 14: nat_body<14, T, nat_gr(14, T)>(b, XQ, W, y, K, sg.n, sg.ks, sg.hoff, sg.soff, OS); break;
        default: bf16_body<T, GB>(b, X, W, y, K, sg.n, sg.ks, OS); break;
    }
}

#define NATST(T) \
extern "C" __global__ void __launch_bounds__(256) fl_natst_t##T( \
    const unsigned int* __restrict__ XQ, const float* __restrict__ X, const StSeg* __restrict__ SEGS, \
    unsigned int nseg, float* __restrict__ Y, unsigned int K, unsigned int OS) { \
    stack_body<T>(XQ, X, SEGS, nseg, Y, K, OS); \
}
NATST(1) NATST(2) NATST(3) NATST(4) NATST(5) NATST(6) NATST(7) NATST(8)
NATST(16) NATST(32) NATST(64)
#endif
"#;
