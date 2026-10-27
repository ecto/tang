
// y[t*YSTRIDE + r] = dot(W row r of matrix e, x + t*COLS). Call per projection with the
// destination base pointer and stride (ROWS for a plain layout, 3*DI for stacked qkv).
extern "C" __global__ void k5_gemv(const G5W w, unsigned e, const float* x, float* y,
                                   unsigned ystride, unsigned T, unsigned ROWS, unsigned COLS) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned t = i / ROWS, r = i % ROWS;
    if (t >= T) return;
    y[(unsigned long long)t * ystride + r] = g5_dot(w, e, r, x + (unsigned long long)t * COLS, COLS);
}

// g1 in place: g1[i] = gate_form(g1[i] + dt[chan]) over nh*hd per token.
extern "C" __global__ void k5_gate_finish(float* g1, const float* dt, const float* rate,
                                          unsigned NH, unsigned HD, float LB, unsigned T) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * NH * HD) return;
    unsigned chan = i % (NH * HD);
    unsigned h = chan / HD;
    float pre = g1[i] + dt[chan];
    float out;
    if (LB == LB) { // NaN means absent: the softplus form
        out = LB / (1.f + expf(-rate[h] * pre));
    } else {
        float v = rate[h] * pre; // softplus: log(1+e^v), pass-through above 20 (ggml)
        out = -(v > 20.f ? v : logf(1.f + expf(v)));
    }
    g1[i] = out;
}

// beta[t][h] = sigmoid(dot(b_proj row h, x[t])).
extern "C" __global__ void k5_beta(const G5W bp, const float* x, float* beta,
                                   unsigned T, unsigned HID, unsigned NH) {
    // one WARP per (t, h): the naive form gave one thread a serial 4096-element
    // dequant+dot (1.28 ms/launch, 45.8 ms per decode token across 34 layers, ~0 B read
    // because each thread streams its row alone). Warp lanes split the row 128-wide in
    // 8-element steps with the shuffle reduce — the same shape as k5_gemv8's dot.
    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const unsigned i = blockIdx.x * (blockDim.x >> 5) + warp;
    const unsigned t = i / NH, h = i % NH;
    if (t >= T) return;
    float a = 0.f;
    for (int e0 = lane * 8; e0 < (int)HID; e0 += 256) {
        #pragma unroll
        for (int q = 0; q < 8; q++) a = fmaf(g5_deq(bp.ty, bp.base + (unsigned long long)h * bp.row_bytes, e0 + q), x[(size_t)t * HID + e0 + q], a);
    }
    for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
    if (lane == 0) beta[(unsigned long long)t * NH + h] = 1.f / (1.f + expf(-a));
}

// y[t][c] = rms_norm(o[t][h..])[c] * on[c] * sigmoid(g2[t][c]), h = c / HD.
extern "C" __global__ void k5_gated_norm(const float* o, const float* on, const float* g2,
                                         float* y, unsigned NH, unsigned HD, float EPS,
                                         unsigned T) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * NH * HD) return;
    unsigned t = i / (NH * HD), c = i % (NH * HD);
    unsigned h = c / HD, j = c % HD;
    const float* ot = o + ((unsigned long long)t * NH + h) * HD;
    double s = 0.0; // the host's rms_norm_mul: f64 sum of squares
    for (unsigned d = 0; d < HD; d++) { double v = (double)ot[d]; s += v * v; }
    float r = (float)(1.0 / sqrt((s / (double)HD) + (double)EPS));
    y[i] = ot[j] * r * on[j] / (1.f + expf(-g2[i]));
}




// ---------------------------------------------------------------------------
// k5_gemv8: the dequant-reuse GEMV for Q8_0 rows (docs/models/glm5-speed-plan.md step 1)
// — the Flash engine's fe_gemv8 shape (kernels.rs gemv8_body) cut down to the G5W view's
// one-type matrices. SCOPE: the KDA projections, all Q8_0 in the UD-Q4_K_XL GGUF
// (attn_q/k/v/output, ssm_f_a/f_b, ssm_beta, ssm_g_a/g_b). The MoE experts (Q4_K/Q5_K)
// stay on g5_expert_gu for now.
//
// Block 128 = K5_WARPS warps, ONE row per warp (grid.x covers ROWS/K5_WARPS blocks,
// grid.y = T). The row is staged into shared memory with aligned uint4 loads (the whole
// 8.7 KB row in flight vs the naive one-thread serial read); then the 32 lanes walk it
// in 8-element steps: e0 = lane*8 + 256*i, each step dp4a's 8 contiguous row bytes
// (Q8_0: block at e0/32, values at bytes 2 + e0%32 .. +8) against 8 int8 activation
// codes, scaled by that block's f16 d and the activation block's d. Shared use: 4 warps
// x 8.7 KB = 34.8 KB (under the 48 KB default; two rows a warp would need a dynamic-
// smem opt-in — later, if the measure says so).
//
// Activations: xq = fe_q8's layout — plain int8 codes [T][COLS], f32 per-32 scales
// [T][COLS/32], d = amax/127. NOT the f32 serial rounding: parity is the engine test's
// 1e-3 bar + the real-model ids (fe_gemv8's bar). F32 rows never come through here
// (the small per-layer f32 tensors are not GEMV weights).
#ifndef K5_HOST_CHECK
#define K5_WARPS 4

// int -> float via a magic add (kernels.rs i2f): exact for |q| < 2^22.
__device__ __forceinline__ float k5_i2f(int q) { return __int_as_float(0x4B400000 + q) - 12582912.0f; }

// x [T][COLS] f32 -> xq: plain int8 codes [T][COLS] then per-32 f32 scales [T][COLS/32]
// (fe_q8's rule: d = amax/127 over each 32, round half away). Grid (COLS/32, T), block 32.
// MAXT = the scratch's row cap (the codes array is sized MAXT rows; scales follow it).
extern "C" __global__ void k5_q8(const float* x, unsigned char* xq, unsigned COLS, unsigned MAXT) {
    const unsigned t = blockIdx.y, c = blockIdx.x, lane = threadIdx.x;
    const float v = x[(size_t)t * COLS + c * 32 + lane];
    float amax = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = amax / 127.0f;
    const int q = d == 0.f ? 0 : (int)fminf(fmaxf(roundf(v / d), -127.f), 127.f);
    xq[(size_t)t * COLS + c * 32 + lane] = (unsigned char)(signed char)q;
    if (lane == 0) ((float*)(xq + (size_t)MAXT * COLS))[t * (COLS / 32) + c] = d;
}

extern "C" __global__ void __launch_bounds__(128) k5_gemv8(
    const G5W w, unsigned e, const unsigned char* xq, const float* xd,
    float* y, unsigned ystride, unsigned T, unsigned ROWS, unsigned COLS) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    extern __shared__ uint4 dsm[]; // per warp: one row, ceil(row_bytes/16)+1 uint4s
    const unsigned rb = w.row_bytes;
    const int row16 = (int)((rb + 15) >> 4);
    const unsigned long long mbase = w.base + (unsigned long long)e * w.estride;
    const int t = (int)blockIdx.y;
    if (t >= (int)T) return;
    const unsigned char* xrow = xq + (size_t)t * COLS;

    {
        const unsigned row = blockIdx.x * K5_WARPS + warp;
        if (row >= ROWS) return;
        const unsigned char* src = (const unsigned char*)(mbase + (unsigned long long)row * rb);
        // +1 granule per warp: an unaligned row (delta > 0) spans row16+1 granules; spacing
        // warps by row16 let each warp's staging overwrite the next warp's first granule
        uint4* sm = dsm + (size_t)warp * (row16 + 1);
        const unsigned long long a = (unsigned long long)src & ~15ull;
        const int delta = (int)((unsigned long long)src - a);
        const int n16 = (delta + (int)rb + 15) >> 4;
        for (int i = lane; i < n16; i += 32) sm[i] = __ldg((const uint4*)a + i);
        // lanes read granules other lanes staged: the warp must sync first (missing in the
        // first 3090 run — the shadow check failed 68/68 windows at qkv.q)
        __syncwarp();
        const unsigned char* rowp = (const unsigned char*)sm + delta;
        float acc = 0.f;
        for (int e0 = lane * 8; e0 < (int)COLS; e0 += 256) {
            const unsigned char* b = rowp + (e0 / 32) * 34;
            const float dw = g5_h2f(g5_u16(b));
            // Q8_0 blocks are 34 B: b + 2 + e0%32 is NOT 8-byte aligned — assemble the two
            // 4-byte words from bytes (a direct u64 load faults: MISALIGNED/ILLEGAL_ADDRESS)
            const unsigned char* qb = b + 2 + (e0 % 32);
            const int w0 = (int)((unsigned)qb[0] | ((unsigned)qb[1] << 8) | ((unsigned)qb[2] << 16) | ((unsigned)qb[3] << 24));
            const int w1 = (int)((unsigned)qb[4] | ((unsigned)qb[5] << 8) | ((unsigned)qb[6] << 16) | ((unsigned)qb[7] << 24));
            const uint2 xv = *(const uint2*)(xrow + e0);
            const float dx = xd[(size_t)t * (COLS / 32) + e0 / 32];
            const int s = __dp4a(w0, (int)xv.x, __dp4a(w1, (int)xv.y, 0));
            acc = fmaf(dx * dw, k5_i2f(s), acc);
        }
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) y[(unsigned long long)t * ystride + row] = acc;
        __syncwarp();
    }
}
#endif // K5_HOST_CHECK
