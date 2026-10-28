// Device-side selection (appended to kernels.cu in one NVRTC module): the indexer's masks,
// candidate blocks and top-k, MoE routing lists, and attention index rows, so a decode step
// has no host round trip for them. Each mirrors forward.rs (`indexer`, `candidates`,
// `topk_stable`, `moe`): ties to the LOWEST index, selections emitted in ascending order.

#define DS_SEL_THREADS 512

// Order-preserving key: a > b as floats <=> key(a) > key(b). ±0 share a key (they tie in
// Rust's partial_cmp).
__device__ __forceinline__ unsigned int ds_key(float x) {
    if (x == 0.f) x = 0.f;
    unsigned int u = __float_as_uint(x);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Prefix lens of row `ti`: causal prefill sees (ti + 1) / r compressed entries, decode all.
__device__ __forceinline__ int ds_len(int ti, int width, int causal, int r) {
    return causal ? (ti + 1) / r : width;
}

// Block-wide exclusive prefix of `flag` (blockDim.x == DS_SEL_THREADS); returns the block
// total in *total. `ws` holds DS_SEL_THREADS / 32 + 1 ints.
__device__ int ds_block_scan(int flag, int* ws, int* total) {
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    unsigned int b = __ballot_sync(0xffffffffu, flag);
    int pre = __popc(b & ((1u << lane) - 1u));
    __syncthreads();
    if (lane == 0) ws[w] = __popc(b);
    __syncthreads();
    if (threadIdx.x == 0) {
        int s = 0;
        for (int i = 0; i < nw; i++) {
            int c = ws[i];
            ws[i] = s;
            s += c;
        }
        ws[nw] = s;
    }
    __syncthreads();
    pre += ws[w];
    *total = ws[nw];
    return pre;
}

// One block: the `k` largest of v[0..n) (ties to the lowest index), written to sel[0..k)
// in ascending index order. Radix select on the keys finds the k-th largest key T; then
// everything above T plus the first (k - #above) equal to T, in index order.
__device__ void ds_select_row(const float* v, int n, int k, int* sel) {
    __shared__ int hist[256];
    __shared__ int ws[DS_SEL_THREADS / 32 + 1];
    __shared__ unsigned int s_prefix;
    __shared__ int s_kk;
    if (k <= 0) return;
    unsigned int prefix = 0, mask = 0;
    int kk = k;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            unsigned int key = ds_key(v[i]);
            if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255u], 1);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int b = 255;
            for (; b > 0; b--) {
                if (hist[b] >= kk) break;
                kk -= hist[b];
            }
            s_prefix = prefix | ((unsigned int)b << shift);
            s_kk = kk;
        }
        __syncthreads();
        prefix = s_prefix;
        kk = s_kk;
        mask |= 255u << shift;
        __syncthreads();
    }
    // prefix = T exactly; kk = how many of the keys equal to T to take
    int base = 0, eq_seen = 0;
    for (int c0 = 0; c0 < n; c0 += blockDim.x) {
        int i = c0 + threadIdx.x;
        unsigned int key = i < n ? ds_key(v[i]) : 0u;
        int eq = i < n && key == prefix;
        int eq_tot;
        int eq_pre = ds_block_scan(eq, ws, &eq_tot);
        int take = i < n && (key > prefix || (eq && eq_seen + eq_pre < kk));
        int tot;
        int pos = ds_block_scan(take, ws, &tot);
        if (take) sel[base + pos] = i;
        base += tot;
        eq_seen += eq_tot;
    }
    __syncthreads();
}

// In place on score [t][width]: -inf past each row's causal prefix (when `causal`), and
// where the candidate mask cand [t][width] (u8; null for none) is 0.
extern "C" __global__ void ds_mask_scores(float* score, const unsigned char* cand, int t, int width, int causal,
                                          int r) {
    long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= (long long)t * width) return;
    int ti = (int)(e / width), j = (int)(e % width);
    if ((causal && j >= ds_len(ti, width, causal, r)) || (cand && !cand[e])) score[e] = -DS_INF;
}

// Candidate blocks, one block per row: block maxima of `bs` (scratch sc [t][nb]), the block
// holding the row's newest position pinned to +inf, the top `kb` blocks kept (-inf blocks
// never), expanded to a mask cand [t][width].
extern "C" __global__ void ds_candidates(const float* score, float* sc, int* scratch, unsigned char* cand, int t,
                                         int width, int causal, int r, int bs, int kb) {
    int ti = blockIdx.x;
    if (ti >= t) return;
    int nb = (width + bs - 1) / bs;
    const float* row = score + (long long)ti * width;
    float* s = sc + (long long)ti * nb;
    int* sel = scratch + (long long)ti * nb;
    int len = ds_len(ti, width, causal, r);
    int last = len >= 1 ? (len - 1) / bs : -1;
    for (int b = threadIdx.x; b < nb; b += blockDim.x) {
        float m = -DS_INF;
        int hi = min((b + 1) * bs, width);
        for (int j = b * bs; j < hi; j++) m = fmaxf(m, row[j]);
        s[b] = b == last ? DS_INF : m;
    }
    __syncthreads();
    int k = min(kb, nb);
    ds_select_row(s, nb, k, sel);
    unsigned char* out = cand + (long long)ti * width;
    for (int j = threadIdx.x; j < width; j += blockDim.x) out[j] = 0;
    __syncthreads();
    for (int q = 0; q < k; q++) {
        int b = sel[q];
        if (!(s[b] > -DS_INF)) continue;
        int hi = min((b + 1) * bs, width);
        for (int j = b * bs + threadIdx.x; j < hi; j += blockDim.x) out[j] = 1;
    }
}

// Indexer top-k, one block per row: the k largest of score [t][width] in ascending index
// order, as out[ti * stride + q] = idx + offset, or -1 when idx is past the row's prefix.
// `scratch` holds [t][k] ints.
extern "C" __global__ void ds_topk(const float* score, int* scratch, int* out, int t, int width, int k, int stride,
                                   int offset, int causal, int r) {
    int ti = blockIdx.x;
    if (ti >= t) return;
    int* sel = scratch + (long long)ti * k;
    ds_select_row(score + (long long)ti * width, width, k, sel);
    int len = ds_len(ti, width, causal, r);
    for (int q = threadIdx.x; q < k; q += blockDim.x) {
        int i = sel[q];
        out[(long long)ti * stride + q] = i < len ? i + offset : -1;
    }
}

// Attention window rows into out [t][stride] at column 0:
//   mode 0, prefill: row e sees positions max(e - win + 1, 0) .. e (n = min(t, win) columns,
//     -1 past e);
//   mode 1, decode at sp: the ring oldest first, -1 for slots not yet written;
//   mode 3, verify window at sp (kv = the pre-chunk ring then the chunk's t rows): row e sees
//     positions max(sp + e - win + 1, 0) .. sp + e ascending, -1 past them;
//   mode 2, DSpark at sp: ring slots 0 .. min(win, sp + 1), then win .. win + b (the block).
extern "C" __global__ void ds_win_idx(int* out, int t, int stride, int n, int mode, int sp, int win, int b) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= t * n) return;
    int row = e / n, j = e % n, v;
    if (mode == 0) {
        int lo = max(row - win + 1, 0);
        v = lo + j > row ? -1 : lo + j;
    } else if (mode == 1) {
        int i = (sp % win + 1 + j) % win;
        v = i > sp ? -1 : i;
    } else if (mode == 3) {
        int q = sp + row, cnt = min(win, q + 1), pos = q + 1 - cnt + j;
        v = j >= cnt ? -1 : (pos < sp ? pos % win : win + pos - sp);
    } else {
        int n1 = min(win, sp + 1);
        v = j < n1 ? j : win + (j - n1);
    }
    out[(long long)row * stride + j] = v;
}

// dst[row * dstride + col + j] = src[row * k + j], j < k.
extern "C" __global__ void ds_idx_cat(int* dst, const int* src, int t, int k, int dstride, int col) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= t * k) return;
    dst[(long long)(e / k) * dstride + col + e % k] = src[e];
}

// MoE routing lists, one block: ids/wts [t][k] -> per-expert CSR, rows ascending within an
// expert: counts[ne], off[ne + 1], rows[t * k] (token), rw[t * k] (route weight).
extern "C" __global__ void ds_route(const int* ids, const float* wts, int* counts, int* off, int* rows, float* rw,
                                    int t, int k, int ne) {
    int n = t * k;
    for (int e = threadIdx.x; e < ne; e += blockDim.x) {
        int c = 0;
        for (int i = 0; i < n; i++) c += ids[i] == e;
        counts[e] = c;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int s = 0;
        for (int e = 0; e < ne; e++) {
            off[e] = s;
            s += counts[e];
        }
        off[ne] = s;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < ne; e += blockDim.x) {
        int o = off[e];
        for (int i = 0; i < n; i++)
            if (ids[i] == e) {
                rows[o] = i / k;
                rw[o] = wts[i];
                o++;
            }
    }
}

// Expert e's rows of x [.][d] into xs [cap][d] and its route weights into w [cap]; rows past
// the expert's count are zero (cap is the host's bound, t).
extern "C" __global__ void ds_gather_csr(const float* x, const int* off, const int* rows, const float* rw, float* xs,
                                         float* w, int e, int cap, int d) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)cap * d) return;
    int j = (int)(i / d), n = off[e + 1] - off[e];
    bool live = j < n;
    xs[i] = live ? x[(long long)rows[off[e] + j] * d + i % d] : 0.f;
    if (i % d == 0) w[j] = live ? rw[off[e] + j] : 0.f;
}

// y[rows[j]] += v[j] for expert e's rows only.
extern "C" __global__ void ds_scatter_csr(float* y, const float* v, const int* off, const int* rows, int e, int cap,
                                          int d) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int n = off[e + 1] - off[e];
    if (i >= (long long)n * d) return;
    y[(long long)rows[off[e] + i / d] * d + i % d] += v[i];
}

// ds_linear limited to expert e's row count (rows past it are left unwritten).
extern "C" __global__ void ds_linear_csr(DsW w, const float* x, float* y, int t, int round, const int* off, int e) {
    int n_rows = min(t, off[e + 1] - off[e]);
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
    if (warp >= n_rows * w.rows) return;
    int ti = warp / w.rows, n = warp % w.rows;
    const float* xr = x + (long long)ti * w.cols;
    float s = 0.f;
    for (int k = lane; k < w.cols; k += 32) s += xr[k] * wget(w, n, k);
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0) y[(long long)ti * w.rows + n] = round ? bf(s) : s;
}

// Grouped MoE plan, one thread: elist[0..cap) = the experts with rows that this rank owns
// (mask[e] != 0), ascending, then -1.
extern "C" __global__ void ds_moe_plan(const int* counts, const int* mask, int* elist, int ne, int cap) {
    if (threadIdx.x || blockIdx.x) return;
    int j = 0;
    for (int e = 0; e < ne && j < cap; e++)
        if (counts[e] > 0 && mask[e]) elist[j++] = e;
    for (; j < cap; j++) elist[j] = -1;
}

// y [t][d] = Σ over token ti's local experts in ascending id of part[slot] (slot = the token's
// CSR position under that expert): the per-expert loop's scatter order, from 0.
extern "C" __global__ void ds_moe_reduce(float* y, const float* part, const int* ids, const int* off, const int* rows,
                                         const int* mask, int t, int k, int d) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)t * d) return;
    int ti = (int)(i / d), c = (int)(i % d);
    const int* id = ids + (long long)ti * k;
    float s = 0.f;
    int done = -1;
    for (int q = 0; q < k; q++) {
        int best = 0x7fffffff;
        for (int r = 0; r < k; r++)
            if (id[r] > done && id[r] < best) best = id[r];
        if (best == 0x7fffffff) break;
        done = best;
        if (!mask[best]) continue;
        int o = off[best], n = off[best + 1] - o, p = o;
        for (int j = 0; j < n; j++)
            if (rows[o + j] == ti) {
                p = o + j;
                break;
            }
        s += part[(long long)p * d + c];
    }
    y[i] = s;
}

// n ints set to v (a verify window's [t][k] selection, -1 = no entry).
extern "C" __global__ void ds_fill_i32(int* p, int v, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

// n bytes (candidate mask rows).
extern "C" __global__ void ds_copy_u8(unsigned char* dst, const unsigned char* src, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// DSpark's greedy Markov draw on the device: row += bias (when given; the Markov bias, as
// ds_add), ids[slot] = argmax(row) (the first maximum, as the host `argmax`), or with
// row == nullptr keep ids[slot] (first set to set[0] when given); then, when memb != nullptr, the
// Markov embedding row of that id: memb[0..rank] = tab[id * rank ..]. One block of 1024.
extern "C" __global__ void ds_spark_pick(float* row, int v, int* ids, int slot, const float* tab, int rank,
                                         float* memb, const float* bias, const int* set) {
    __shared__ float sv[1024];
    __shared__ int si[1024];
    __shared__ int pick;
    int tid = threadIdx.x;
    if (row) {
        float bv = -DS_INF;
        int bi = 0x7fffffff;
        for (int j = tid; j < v; j += blockDim.x) {
            float x = row[j];
            if (bias) row[j] = x = x + bias[j];
            if (x > bv) {
                bv = x;
                bi = j;
            }
        }
        sv[tid] = bv;
        si[tid] = bi;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                float ov = sv[tid + s];
                int oi = si[tid + s];
                if (ov > sv[tid] || (ov == sv[tid] && oi < si[tid])) {
                    sv[tid] = ov;
                    si[tid] = oi;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            int a = si[0] == 0x7fffffff ? 0 : si[0];
            ids[slot] = a;
            pick = a;
        }
    } else if (tid == 0) {
        if (set) ids[slot] = set[0];
        pick = ids[slot];
    }
    __syncthreads();
    if (memb)
        for (int j = tid; j < rank; j += blockDim.x) memb[j] = tab[(long long)pick * rank + j];
}

// ---- DSpark's sampled Markov draw on the device (`SpecSampler::sample`, same transform) ----

// The 15 sums of a thread's `loc` over the block (1024 threads), in a fixed order (so every
// EP rank gets the same bits); the results in `out` for all threads.
__device__ void ds_ss_sum15(double* loc, double (*red)[15], double* out) {
    int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    for (int k = 0; k < 15; k++) {
        double x = loc[k];
        for (int o = 16; o > 0; o >>= 1) x += __shfl_down_sync(0xffffffff, x, o);
        if (lane == 0) red[wid][k] = x;
    }
    __syncthreads();
    if (tid < 15) {
        double s = 0.0;
        for (int w = 0; w < (int)(blockDim.x >> 5); w++) s += red[w][tid];
        out[tid] = s;
    }
    __syncthreads();
}

// The largest key k (w's bits: w >= 0, so they order as w does) with G(k) = sum of w over
// keys >= k at least `target` and positive, 4 bits a pass.
__device__ unsigned long long ds_ss_search(const double* w, int v, double target, double (*red)[15],
                                  double* out) {
    unsigned long long kap = 0;
    double loc[15];
    for (int sh = 60; sh >= 0; sh -= 4) {
        for (int k = 0; k < 15; k++) loc[k] = 0.0;
        for (int j = threadIdx.x; j < v; j += blockDim.x) {
            unsigned long long key = (unsigned long long)__double_as_longlong(w[j]);
            double x = w[j];
            for (int k = 0; k < 15; k++)
                if (key >= (kap | ((unsigned long long)(k + 1) << sh))) loc[k] += x;
        }
        ds_ss_sum15(loc, red, out);
        int best = 0;
        for (int k = 0; k < 15; k++)
            if (out[k] >= target && out[k] > 0.0) best = k + 1;
        kap |= (unsigned long long)best << sh;
        __syncthreads();
    }
    return kap;
}

// G(> kap) and the number of keys == kap.
__device__ void ds_ss_above(const double* w, int v, unsigned long long kap, double (*red)[15], double* out,
                            double* above, int* ties) {
    double loc[15];
    for (int k = 0; k < 15; k++) loc[k] = 0.0;
    for (int j = threadIdx.x; j < v; j += blockDim.x) {
        unsigned long long key = (unsigned long long)__double_as_longlong(w[j]);
        if (key > kap) loc[0] += w[j];
        if (key == kap) loc[1] += 1.0;
    }
    ds_ss_sum15(loc, red, out);
    *above = out[0];
    *ties = (int)out[1];
    __syncthreads();
}

// The n-th (1-based) token, in id order, with key == kap.
__device__ int ds_ss_nth(const double* w, int v, unsigned long long kap, int n, int* cnt, int* res) {
    int tid = threadIdx.x, nt = blockDim.x;
    int cs = (v + nt - 1) / nt, lo = tid * cs, hi = min(v, lo + cs);
    int c = 0;
    for (int j = lo; j < hi; j++) c += (unsigned long long)__double_as_longlong(w[j]) == kap;
    cnt[tid] = c;
    if (tid == 0) *res = -1;
    __syncthreads();
    if (tid == 0) {
        int acc = 0;
        for (int i = 0; i < nt; i++) {
            int x = cnt[i];
            cnt[i] = acc;
            acc += x;
        }
    }
    __syncthreads();
    int before = cnt[tid];
    if (before < n && before + c >= n) {
        int k = before;
        for (int j = lo; j < hi; j++)
            if ((unsigned long long)__double_as_longlong(w[j]) == kap && ++k == n) {
                *res = j;
                break;
            }
    }
    __syncthreads();
    return *res;
}

// ids[slot] = a draw from SpecSampler::dist(row) (temperature `temp` > 0, top-p `top_p`):
// w = exp((row - max) / temp) in f64, the smallest prefix by decreasing w (ties by id) holding
// top_p of the mass, one uniform from the splitmix64 state *rng (advanced), the first prefix
// token whose cumulative mass reaches it. `w` is [v] f64 scratch; `bias` (when given) is
// first added to row, as ds_spark_pick. Then, as ds_spark_pick, the
// Markov embedding row of the draw. One block of 1024.
extern "C" __global__ void __launch_bounds__(1024) ds_spark_sample(float* row, int v, int* ids, int slot,
                                           const float* tab, int rank, float* memb, float temp,
                                           float top_p, unsigned long long* rng, double* w,
                                           const float* bias) {
    __shared__ double red[32][15];
    __shared__ double out[15];
    __shared__ float smx[32];
    __shared__ int cnt[1024];
    __shared__ int res;
    __shared__ double sz, su;
    __shared__ int pick;
    int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    float mx = -DS_INF;
    for (int j = tid; j < v; j += blockDim.x) {
        float x = row[j];
        if (bias) row[j] = x = x + bias[j];
        mx = fmaxf(mx, x);
    }
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, o));
    if (lane == 0) smx[wid] = mx;
    __syncthreads();
    if (tid == 0) {
        float m = smx[0];
        for (int i = 1; i < (int)(blockDim.x >> 5); i++) m = fmaxf(m, smx[i]);
        smx[0] = m;
    }
    __syncthreads();
    mx = smx[0];
    double loc[15];
    for (int k = 0; k < 15; k++) loc[k] = 0.0;
    for (int j = tid; j < v; j += blockDim.x) {
        double x = exp((double)__fdiv_rn(__fsub_rn(row[j], mx), temp));
        w[j] = x;
        loc[0] += x;
    }
    __syncthreads();
    ds_ss_sum15(loc, red, out);
    if (tid == 0) {
        sz = out[0];
        unsigned long long s = *rng + 0x9E3779B97F4A7C15ull;
        *rng = s;
        unsigned long long z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        su = (double)(z >> 11) / (double)(1ull << 53);
    }
    __syncthreads();
    double z = sz, u = su;
    // the kept prefix: keys above kc, then the first m ties at kc
    unsigned long long kc = 0;
    double above = 0.0, total = z;
    int ties = 0, m = 0;
    bool cut = top_p < 1.0f;
    if (cut) {
        double tgt = (double)top_p * z;
        kc = ds_ss_search(w, v, tgt, red, out);
        ds_ss_above(w, v, kc, red, out, &above, &ties);
        double wc = __longlong_as_double((long long)kc);
        m = wc > 0.0 ? (int)ceil((tgt - above) / wc) : ties;
        m = max(1, min(m, ties));
        total = above + m * wc;
    }
    double r = fmin(u * total, total);
    int tok;
    if (cut && r > above) {
        double wc = __longlong_as_double((long long)kc);
        int n = wc > 0.0 ? (int)ceil((r - above) / wc) : m;
        tok = ds_ss_nth(w, v, kc, max(1, min(n, m)), cnt, &res);
    } else {
        unsigned long long kd = ds_ss_search(w, v, r, red, out);
        double ab;
        int nt;
        ds_ss_above(w, v, kd, red, out, &ab, &nt);
        double wd = __longlong_as_double((long long)kd);
        int n = wd > 0.0 ? (int)ceil((r - ab) / wd) : nt;
        tok = ds_ss_nth(w, v, kd, max(1, min(n, nt)), cnt, &res);
    }
    if (tid == 0) {
        if (tok < 0) tok = 0;
        ids[slot] = tok;
        pick = tok;
    }
    __syncthreads();
    if (memb)
        for (int j = tid; j < rank; j += blockDim.x) memb[j] = tab[(long long)pick * rank + j];
}
