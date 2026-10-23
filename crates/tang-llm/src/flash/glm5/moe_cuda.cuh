// GLM-5.3-Flash MoE, CUDA (NVRTC). NAIVE ON PURPOSE (GATES.md: WORK -> RIGHT -> FAST): one thread
// per output value, plain loops, weights dequantised element by element straight from their GGUF
// bytes, f32 activations and accumulation. No shared memory, no warp shuffles, no int8 activations.
// Optimisation ideas are listed under "later" in moe.rs.
//
// The file only uses __global__/__device__, blockIdx/blockDim/threadIdx .x, expf/fminf/fmaxf and
// __uint_as_float, so scripts/glm5_moe_cuda_host.cpp can compile it as host C++ and execute every
// kernel with an emulated grid (that is how it is checked on a Mac with no CUDA).
//
// Weight view G5W: matrix e of a weight tensor starts at base + e * estride; row r of it at
// + r * row_bytes; ty is the ggml type id (0 F32, 1 F16, 8 Q8_0, 12 Q4_K, 13 Q5_K, 14 Q6_K). A 2-D
// tensor is estride 0. GGUF fused experts [in, out, E] have estride = out * row_bytes.
//
// One MoE layer (T tokens, K = top_k, x [T][hid] f32 on device):
//   g5_router_logits  grid ceil(T*ne/128)          s    [T][ne]   sigmoid(router . x)
//   g5_router_select  grid ceil(T/128)             ids  [T][K] u32, wts [T][K] f32
//   g5_expert_gu      grid ceil(T*K*ff/128)        h    [T*K][ff] silu(min(g,L)) * clamp(u,±L)
//   g5_expert_down    grid ceil(T*K*hid/128)       yp   [T*K][hid]
//   shared expert:    g5_expert_gu / g5_expert_down with ids = 0 (pair = token, expert 0) -> ysh
//   g5_combine        grid ceil(T*hid/128)         y    [T][hid] = sum_k wts * yp + ysh
// Dense FFN (first 3 layers): g5_expert_gu + g5_expert_down with ids = 0, writing y directly.
// All kernels: block 128 (any size works; each thread bounds-checks its index).

struct G5W {
    unsigned long long base;
    unsigned long long estride;
    unsigned row_bytes;
    unsigned ty;
};

__device__ __forceinline__ unsigned g5_u16(const unsigned char* p) { return (unsigned)p[0] | ((unsigned)p[1] << 8); }
__device__ __forceinline__ unsigned g5_u32(const unsigned char* p) {
    return (unsigned)p[0] | ((unsigned)p[1] << 8) | ((unsigned)p[2] << 16) | ((unsigned)p[3] << 24);
}

// IEEE half -> float by bits (no intrinsics, so the host build is the same code).
__device__ __forceinline__ float g5_h2f(unsigned h) {
    const unsigned sign = (h & 0x8000u) << 16;
    int ex = (int)((h >> 10) & 0x1f);
    unsigned man = h & 0x3ffu;
    if (ex == 0x1f) return __uint_as_float(sign | 0x7f800000u | (man << 13));
    if (ex == 0) {
        if (man == 0) return __uint_as_float(sign);
        ex = 1;
        while (!(man & 0x400u)) { man <<= 1; ex--; }
        man &= 0x3ffu;
    }
    return __uint_as_float(sign | ((unsigned)(ex + 112) << 23) | (man << 13));
}

// ggml get_scale_min_k4
__device__ __forceinline__ void g5_scale_min_k4(int j, const unsigned char* q, int* d, int* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Element j of a row of type ty (ggml dequantize_row_* semantics). Unknown types give NaN.
__device__ float g5_deq(unsigned ty, const unsigned char* row, int j) {
    if (ty == 0) return __uint_as_float(g5_u32(row + 4 * j));
    if (ty == 1) return g5_h2f(g5_u16(row + 2 * j));
    if (ty == 8) {  // Q8_0: 32 per block, 34 B: f16 d, int8 qs[32]
        const unsigned char* b = row + (j / 32) * 34;
        return g5_h2f(g5_u16(b)) * (float)(signed char)b[2 + j % 32];
    }
    if (ty == 12 || ty == 13) {  // Q4_K 144 B / Q5_K 176 B: f16 d, f16 dmin, scales[12], (qh[32]), qs[128]
        const unsigned char* b = row + (j / 256) * (ty == 12 ? 144 : 176);
        const int e = j % 256, c = e / 64, w = e % 64, hi = w / 32, l = w % 32;
        int sc, mn;
        g5_scale_min_k4(2 * c + hi, b + 4, &sc, &mn);
        const unsigned char* qs = b + (ty == 12 ? 16 : 48);
        int q = (qs[c * 32 + l] >> (4 * hi)) & 0xF;
        if (ty == 13) q += ((b[16 + l] >> (2 * c + hi)) & 1) << 4;
        return g5_h2f(g5_u16(b)) * (float)sc * (float)q - g5_h2f(g5_u16(b + 2)) * (float)mn;
    }
    if (ty == 14) {  // Q6_K 210 B: ql[128], qh[64], int8 scales[16], f16 d
        const unsigned char* b = row + (j / 256) * 210;
        const int e = j % 256, n = e / 128, r = e % 128, qd = r / 32, l = r % 32;
        // slice n: ql_n = b + n*64, qh_n = b + 128 + n*32. Per (l, qd):
        //   qd 0: lo = ql_n[l] lo nibble  qd 1: lo = ql_n[l+32] lo nibble
        //   qd 2: lo = ql_n[l] hi nibble  qd 3: lo = ql_n[l+32] hi nibble
        //   h2 = qh_n[l] >> (2*qd) & 3; the value's scale row is sc[2*qd + l/16].
        const unsigned char* ql = b + n * 64;
        const unsigned char* qh = b + 128 + n * 32;
        const int lo = (ql[l + (qd & 1) * 32] >> ((qd & 2) ? 4 : 0)) & 0xF;
        const int h2 = (qh[l] >> (2 * qd)) & 3;
        const int sc = (signed char)b[192 + n * 8 + (qd * 2 + l / 16)];
        return g5_h2f(g5_u16(b + 208)) * (float)sc * (float)((lo | (h2 << 4)) - 32);
    }
    return __uint_as_float(0x7fc00000u);
}

// dot(row r of matrix e of w, x[0..k))
__device__ float g5_dot(const G5W w, unsigned e, int r, const float* x, int k) {
    const unsigned char* row = (const unsigned char*)(w.base + (unsigned long long)e * w.estride +
                                                      (unsigned long long)r * w.row_bytes);
    float acc = 0.f;
    for (int j = 0; j < k; j++) acc += g5_deq(w.ty, row, j) * x[j];
    return acc;
}

// s[t][e] = sigmoid(dot(router row e, x[t]))
extern "C" __global__ void g5_router_logits(const G5W w, const float* x, float* s, int T, int ne, int hid) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * ne) return;
    const int t = i / ne, e = i % ne;
    const float a = g5_dot(w, 0u, e, x + (long long)t * hid, hid);
    s[i] = 1.0f / (1.0f + expf(-a));
}

// noaux_tc selection, one thread per token: choice = s + bias (bias may be null); optional group
// limit (group score = sum of its top-2 choice values, keep topk_group groups, lower index wins
// ties); top-k on choice (lower index wins ties); weights = unbiased s of the chosen, divided by
// (sum + 1e-20) if norm, times scale (HF Glm5NextTextTopkRouter). n_group <= 64.
extern "C" __global__ void g5_router_select(const float* s, const float* bias, unsigned* ids, float* wts,
                                            int T, int ne, int k, int n_group, int topk_group, int norm,
                                            float scale) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    const float* st = s + (long long)t * ne;
    unsigned* id = ids + (long long)t * k;
    float* wt = wts + (long long)t * k;
    const int gs = n_group > 1 ? ne / n_group : ne;
    unsigned long long gmask = 1ull;
    if (n_group > 1) {
        gmask = 0ull;
        for (int pick = 0; pick < topk_group; pick++) {
            float best = -INFINITY;
            int bg = -1;
            for (int g = 0; g < n_group; g++) {
                if ((gmask >> g) & 1ull) continue;
                float a = -INFINITY, b2 = -INFINITY;
                for (int e = g * gs; e < (g + 1) * gs; e++) {
                    const float v = st[e] + (bias ? bias[e] : 0.f);
                    if (v > a) { b2 = a; a = v; } else if (v > b2) b2 = v;
                }
                if (bg < 0 || a + b2 > best) { best = a + b2; bg = g; }
            }
            gmask |= 1ull << bg;
        }
    }
    float wsum = 0.f;
    for (int j = 0; j < k; j++) {
        float best = -INFINITY;
        int bi = -1;
        for (int e = 0; e < ne; e++) {
            if (!((gmask >> (e / gs)) & 1ull)) continue;
            bool taken = false;
            for (int q = 0; q < j; q++) taken = taken || id[q] == (unsigned)e;
            if (taken) continue;
            const float v = st[e] + (bias ? bias[e] : 0.f);
            if (bi < 0 || v > best) { best = v; bi = e; }
        }
        id[j] = bi < 0 ? (unsigned)ne : (unsigned)bi;
        wt[j] = bi < 0 ? 0.f : st[bi];
        wsum += wt[j];
    }
    for (int j = 0; j < k; j++) {
        if (norm) wt[j] = wt[j] / (wsum + 1e-20f);
        wt[j] = wt[j] * scale;
    }
}

// h[p][r] = silu(min(gate_r . x, L)) * clamp(up_r . x, -L, L) for pair p. With ids: pair p is
// (token p / k, expert ids[p]); an id >= ne (the "no expert" sentinel) gives 0. Without ids
// (null): pair p is token p on matrix 0 (shared expert, dense FFN). L = INFINITY disables.
extern "C" __global__ void g5_expert_gu(const G5W gate, const G5W up, const unsigned* ids, int k, int ne,
                                        const float* x, float* h, int npairs, int ff, int hid, float limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npairs * ff) return;
    const int p = i / ff, r = i % ff;
    const unsigned e = ids ? ids[p] : 0u;
    const int t = ids ? p / k : p;
    if (ids && e >= (unsigned)ne) { h[i] = 0.f; return; }
    const float* xt = x + (long long)t * hid;
    const float g = fminf(g5_dot(gate, e, r, xt, hid), limit);
    const float u = fminf(fmaxf(g5_dot(up, e, r, xt, hid), -limit), limit);
    h[i] = g / (1.0f + expf(-g)) * u;
}

// yp[p][c] = down_c . h[p] (pairs as in g5_expert_gu)
extern "C" __global__ void g5_expert_down(const G5W down, const unsigned* ids, int ne, const float* h,
                                          float* yp, int npairs, int ff, int hid) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npairs * hid) return;
    const int p = i / hid, c = i % hid;
    const unsigned e = ids ? ids[p] : 0u;
    if (ids && e >= (unsigned)ne) { yp[i] = 0.f; return; }
    yp[i] = g5_dot(down, e, c, h + (long long)p * ff, ff);
}

// y[t][c] = sum_j wts[t][j] * yp[t*k + j][c] (+ ysh[t][c] if ysh is not null)
extern "C" __global__ void g5_combine(const float* yp, const float* wts, const float* ysh, float* y,
                                      int T, int k, int hid) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * hid) return;
    const int t = i / hid, c = i % hid;
    float acc = 0.f;
    for (int j = 0; j < k; j++) acc += wts[t * k + j] * yp[((long long)t * k + j) * hid + c];
    if (ysh) acc += ysh[i];
    y[i] = acc;
}
