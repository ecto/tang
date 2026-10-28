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

// The 8-row tiles of the elist experts' CSR segments, one thread: tiles[0..cap) = e | c << 16
// for each slot's expert e and chunk c < ceil(rows / 8), then -1 (cap >= slots + rows / 8).
extern "C" __global__ void ds_moe_plan_tiles(const int* elist, const int* off, int* tiles, int slots, int cap) {
    if (threadIdx.x || blockIdx.x) return;
    int j = 0;
    for (int s = 0; s < slots; s++) {
        int e = elist[s];
        if (e < 0) break;
        int nc = (off[e + 1] - off[e] + 7) / 8;
        for (int c = 0; c < nc && j < cap; c++) tiles[j++] = e | (c << 16);
    }
    for (; j < cap; j++) tiles[j] = -1;
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

// DSpark's greedy Markov draw on the device: ids[slot] = argmax(row) (the first maximum, as
// the host `argmax`), or with row == nullptr keep ids[slot]; then, when memb != nullptr, the
// Markov embedding row of that id: memb[0..rank] = tab[id * rank ..]. One block of 1024.
extern "C" __global__ void ds_spark_pick(const float* row, int v, int* ids, int slot, const float* tab, int rank,
                                         float* memb) {
    __shared__ float sv[1024];
    __shared__ int si[1024];
    __shared__ int pick;
    int tid = threadIdx.x;
    if (row) {
        float bv = -DS_INF;
        int bi = 0x7fffffff;
        for (int j = tid; j < v; j += blockDim.x) {
            float x = row[j];
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
        pick = ids[slot];
    }
    __syncthreads();
    if (memb)
        for (int j = tid; j < rank; j += blockDim.x) memb[j] = tab[(long long)pick * rank + j];
}
