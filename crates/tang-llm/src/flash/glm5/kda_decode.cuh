
// ---------------------------------------------------------------------------
// The fused KDA DECODE step (T = 1; docs/models/glm5-speed-plan.md step 2, kda_decode.rs).
// Compiled after the KDA prelude (KDA_H/KDA_D/KDA_KC/KDA_CONV) and moe_cuda.cuh (G5W,
// g5_dot). Every value is computed with the SAME expression as the naive kernel it
// replaces (k5_gemv / k5_beta / k5_gate_finish / kda_step / kda_conv_commit /
// k5_gated_norm), so the fused step equals the naive one to the bit; only the launch
// count changes (13 launches + 3 host round-trips -> 4 launches, no host traffic between).

// k5d_proj: every GEMV over the token's x, one thread per output row.
// rows [0, 3DI): q | k | v -> QKV (the stacked layout); [3DI, 3DI+HD): f_a -> G1;
// [.., +NH): beta (sigmoid) -> B; [.., +HD): g_a -> G2A. R0 = the first row this launch
// covers (3*DI when k5_gemv8 already wrote q/k/v). SKIP bit0: f_a is on k5_gemv8 (the
// naive path's per-matrix rule, kda_dev window()) — skip it here, the gemv8 launch wrote
// it; bit1: same for g_a. The f32-dequant g5_dot and gemv8's int8 dp4a round DIFFERENTLY,
// so a group must use the same kernel on both paths or the outputs diverge (the forget
// gate then amplifies the difference through the state — the first real-weights run).
// Grid ceil((3DI+2HD+NH-R0)/128).
extern "C" __global__ void k5d_proj(const G5W wq, const G5W wk, const G5W wv, const G5W fa,
                                    const G5W bp, const G5W ga, const float* x, float* qkv,
                                    float* g1, float* beta, float* g2a, unsigned DI,
                                    unsigned HD, unsigned NH, unsigned HID, unsigned R0,
                                    unsigned SKIP) {
    unsigned i = R0 + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < DI) { qkv[i] = g5_dot(wq, 0u, i, x, HID); return; }
    i -= DI;
    if (i < DI) { qkv[DI + i] = g5_dot(wk, 0u, i, x, HID); return; }
    i -= DI;
    if (i < DI) { qkv[2 * DI + i] = g5_dot(wv, 0u, i, x, HID); return; }
    i -= DI;
    if (i < HD) { if (!(SKIP & 1u)) g1[i] = g5_dot(fa, 0u, i, x, HID); return; }
    i -= HD;
    if (i < NH) { float a = g5_dot(bp, 0u, i, x, HID); beta[i] = 1.f / (1.f + expf(-a)); return; }
    i -= NH;
    if (i < HD) { if (!(SKIP & 2u)) g2a[i] = g5_dot(ga, 0u, i, x, HID); }
}

// k5d_gate2: the two hd-input GEMVs. rows [0, DI): f_b(g1) + dt_bias + the gate form
// (k5_gate_finish's expression) -> G; [DI, 2DI): g_b(g2a) -> G2. Grid ceil(2DI/128).
extern "C" __global__ void k5d_gate2(const G5W fb, const G5W gb, const float* g1,
                                     const float* g2a, const float* dt, const float* rate,
                                     float* g, float* g2, unsigned NH, unsigned HD, float LB) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x, DI = NH * HD;
    if (i < DI) {
        // the naive path stores the GEMV result as a float first, then adds dt: same here
        float raw = g5_dot(fb, 0u, i, g1, HD);
        unsigned h = i / HD;
        float pre = raw + dt[i];
        float out;
        if (LB == LB) {
            out = LB / (1.f + expf(-rate[h] * pre));
        } else {
            float v = rate[h] * pre;
            out = -(v > 20.f ? v : logf(1.f + expf(v)));
        }
        g[i] = out;
        return;
    }
    i -= DI;
    if (i < DI) g2[i] = g5_dot(gb, 0u, i, g2a, HD);
}

// k5d_step: kda_step (fused conv + delta rule, verbatim) + the T=1 conv-history commit
// (each thread owns its 3 channels: it read their history in the conv, then shifts it —
// no other thread touches those channels) + the gated output norm (k5_gated_norm's f64
// sum of squares in the same d order). Grid KDA_H, block KDA_D. Writes O (pre-norm, the
// debug tap) and Y (normed, gated).
extern "C" __global__ void k5d_step(const float* __restrict__ QKV, float* __restrict__ H0,
                                    const float* __restrict__ W, const float* __restrict__ G,
                                    const float* __restrict__ B, float* __restrict__ S,
                                    float* __restrict__ O, const float* __restrict__ ON,
                                    const float* __restrict__ G2, float* __restrict__ Y,
                                    float EPS) {
    unsigned int h = blockIdx.x, x = threadIdx.x, D = KDA_D;
    __shared__ float q[KDA_D], k[KDA_D], sk[KDA_D], os[KDA_D];
    float qv = 0.0f, kv = 0.0f, vv = 0.0f;
    const unsigned int qc = h * D + x, kch = KDA_H * D + h * D + x, vch = 2 * KDA_H * D + h * D + x;
    {
        float acc = 0.0f;
        for (unsigned int i = 0; i < KDA_KC; i++) {
            float e = i < KDA_KC - 1 ? H0[qc * (KDA_KC - 1) + i] : QKV[qc];
            acc = __fmaf_rn(W[qc * KDA_KC + i], e, acc);
        }
        qv = acc / (1.0f + expf(-acc));
        acc = 0.0f;
        for (unsigned int i = 0; i < KDA_KC; i++) {
            float e = i < KDA_KC - 1 ? H0[kch * (KDA_KC - 1) + i] : QKV[kch];
            acc = __fmaf_rn(W[kch * KDA_KC + i], e, acc);
        }
        kv = acc / (1.0f + expf(-acc));
        acc = 0.0f;
        for (unsigned int i = 0; i < KDA_KC; i++) {
            float e = i < KDA_KC - 1 ? H0[vch * (KDA_KC - 1) + i] : QKV[vch];
            acc = __fmaf_rn(W[vch * KDA_KC + i], e, acc);
        }
        vv = acc / (1.0f + expf(-acc));
    }
    // the commit (kda_conv_commit at T=1): history <- [hist[1..], x]
    {
        const unsigned int chs[3] = {qc, kch, vch};
        for (unsigned int c = 0; c < 3; c++) {
            float* hp = H0 + chs[c] * (KDA_KC - 1);
            for (unsigned int s = 0; s < KDA_KC - 1; s++)
                hp[s] = s + 1 < KDA_KC - 1 ? hp[s + 1] : QKV[chs[c]];
        }
    }
    q[x] = qv; k[x] = kv;
    __syncthreads();
    float qs = 0.0f, ks = 0.0f;
    for (unsigned int a = 0; a < D; a++) { qs += q[a] * q[a]; ks += k[a] * k[a]; }
    qs = rsqrtf(qs + 1e-6f); ks = rsqrtf(ks + 1e-6f);
    float scale = rsqrtf((float)D);
    __syncthreads(); // every thread has read q[]/k[] raw before any rescales them
    q[x] *= qs * scale; k[x] *= ks;
    __syncthreads();
    float p = 0.0f;
    for (unsigned int a = 0; a < D; a++) {
        float* sp = S + ((unsigned long long)h * D + a) * D + x;
        float s = (*sp) * expf(G[h * D + a]);
        *sp = s;
        p = __fmaf_rn(k[a], s, p);
    }
    sk[x] = p;
    __syncthreads();
    float delta = B[h] * (vv - sk[x]);
    float o = 0.0f;
    for (unsigned int a = 0; a < D; a++) {
        float* sp = S + ((unsigned long long)h * D + a) * D + x;
        float s = __fmaf_rn(k[a], delta, *sp);
        *sp = s;
        o = __fmaf_rn(q[a], s, o);
    }
    O[h * D + x] = o;
    // gated norm over the head (k5_gated_norm: f64 mean of squares, d ascending)
    os[x] = o;
    __syncthreads();
    double ss = 0.0;
    for (unsigned int d = 0; d < D; d++) { double v = (double)os[d]; ss += v * v; }
    float r = (float)(1.0 / sqrt((ss / (double)D) + (double)EPS));
    Y[h * D + x] = os[x] * r * ON[x] / (1.f + expf(-G2[h * D + x]));
}
