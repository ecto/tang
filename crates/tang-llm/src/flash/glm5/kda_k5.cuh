
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
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned t = i / NH, h = i % NH;
    if (t >= T) return;
    float a = g5_dot(bp, 0u, h, x + (unsigned long long)t * HID, HID);
    beta[(unsigned long long)t * NH + h] = 1.f / (1.f + expf(-a));
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
