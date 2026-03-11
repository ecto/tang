//! CUDA attention kernels: causal self-attention and KV-cached attention.

/// Causal self-attention kernel with GQA support.
/// Grid: (seq_len, n_heads), Block: (min(seq_len, 256))
pub const CAUSAL_ATTENTION_CUDA: &str = r#"
#define MAX_SEQ 2048
extern "C" __global__ void causal_attention(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    const unsigned int seq_len,
    const unsigned int n_heads,
    const unsigned int n_kv_heads,
    const unsigned int head_dim)
{
    __shared__ float scores[MAX_SEQ];
    __shared__ float reduce[256];

    unsigned int pos = blockIdx.x;
    unsigned int head = blockIdx.y;
    unsigned int tid = threadIdx.x;
    unsigned int tg_size = blockDim.x;

    if (pos >= seq_len || head >= n_heads) return;

    unsigned int total_dim = n_heads * head_dim;
    unsigned int kv_dim = n_kv_heads * head_dim;
    unsigned int heads_per_kv = n_heads / n_kv_heads;
    unsigned int kv_head = head / heads_per_kv;
    unsigned int q_off = head * head_dim;
    unsigned int kv_off = kv_head * head_dim;
    float scale = rsqrtf(float(head_dim));
    unsigned int attend_len = pos + 1;

    // Step 1: compute scores
    float local_max = -1e38f;
    for (unsigned int j = tid; j < attend_len; j += tg_size) {
        float dot = 0.0f;
        for (unsigned int d = 0; d < head_dim; d++) {
            dot += Q[pos * total_dim + q_off + d] * K[j * kv_dim + kv_off + d];
        }
        float s = dot * scale;
        scores[j] = s;
        local_max = fmaxf(local_max, s);
    }
    reduce[tid] = local_max;
    __syncthreads();
    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] = fmaxf(reduce[tid], reduce[tid + s]);
        __syncthreads();
    }
    float row_max = reduce[0];
    __syncthreads();  // barrier before reusing reduce[]

    // Step 2: exp + sum
    float local_sum = 0.0f;
    for (unsigned int j = tid; j < attend_len; j += tg_size) {
        float val = expf(scores[j] - row_max);
        scores[j] = val;
        local_sum += val;
    }
    reduce[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] += reduce[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / reduce[0];

    // Step 3: weighted V
    for (unsigned int d = tid; d < head_dim; d += tg_size) {
        float val = 0.0f;
        for (unsigned int j = 0; j < attend_len; j++) {
            val += scores[j] * inv_sum * V[j * kv_dim + kv_off + d];
        }
        output[pos * total_dim + q_off + d] = val;
    }
}
"#;

/// KV-cached attention for autoregressive decoding (single query).
/// Grid: (n_heads), Block: (min(total_len, 256))
pub const KV_ATTENTION_CUDA: &str = r#"
#define MAX_SEQ 2048
extern "C" __global__ void kv_attention(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    const unsigned int cache_len,
    const unsigned int n_heads,
    const unsigned int n_kv_heads,
    const unsigned int head_dim)
{
    __shared__ float scores[MAX_SEQ];
    __shared__ float reduce[256];

    unsigned int head = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int tg_size = blockDim.x;
    if (head >= n_heads) return;

    unsigned int kv_dim = n_kv_heads * head_dim;
    unsigned int heads_per_kv = n_heads / n_kv_heads;
    unsigned int kv_head = head / heads_per_kv;
    unsigned int q_off = head * head_dim;
    unsigned int kv_off = kv_head * head_dim;
    float scale = rsqrtf(float(head_dim));

    // Compute scores
    float local_max = -1e38f;
    for (unsigned int j = tid; j < cache_len; j += tg_size) {
        float dot = 0.0f;
        for (unsigned int d = 0; d < head_dim; d++) {
            dot += Q[q_off + d] * K[j * kv_dim + kv_off + d];
        }
        float s = dot * scale;
        scores[j] = s;
        local_max = fmaxf(local_max, s);
    }
    reduce[tid] = local_max;
    __syncthreads();
    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] = fmaxf(reduce[tid], reduce[tid + s]);
        __syncthreads();
    }
    float row_max = reduce[0];
    __syncthreads();  // barrier before reusing reduce[]

    // Exp + sum
    float local_sum = 0.0f;
    for (unsigned int j = tid; j < cache_len; j += tg_size) {
        float val = expf(scores[j] - row_max);
        scores[j] = val;
        local_sum += val;
    }
    reduce[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] += reduce[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / reduce[0];

    // Weighted V
    for (unsigned int d = tid; d < head_dim; d += tg_size) {
        float val = 0.0f;
        for (unsigned int j = 0; j < cache_len; j++) {
            val += scores[j] * inv_sum * V[j * kv_dim + kv_off + d];
        }
        output[q_off + d] = val;
    }
}
"#;

/// KV-cached attention for batched prefill.
/// Grid: (q_len, n_heads), Block: (min(max_attend, 256))
pub const KV_ATTENTION_PREFILL_CUDA: &str = r#"
#define MAX_SEQ 2048
extern "C" __global__ void kv_attention_prefill(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    const unsigned int cache_start,
    const unsigned int q_len,
    const unsigned int n_heads,
    const unsigned int n_kv_heads,
    const unsigned int head_dim)
{
    __shared__ float scores[MAX_SEQ];
    __shared__ float reduce[256];

    unsigned int qi = blockIdx.x;
    unsigned int head = blockIdx.y;
    unsigned int tid = threadIdx.x;
    unsigned int tg_size = blockDim.x;

    if (qi >= q_len || head >= n_heads) return;

    unsigned int total_dim = n_heads * head_dim;
    unsigned int kv_dim = n_kv_heads * head_dim;
    unsigned int heads_per_kv = n_heads / n_kv_heads;
    unsigned int kv_head = head / heads_per_kv;
    unsigned int q_off = qi * total_dim + head * head_dim;
    unsigned int kv_off = kv_head * head_dim;
    float scale = rsqrtf(float(head_dim));
    unsigned int attend_len = cache_start + qi + 1;

    // Compute scores
    float local_max = -1e38f;
    for (unsigned int j = tid; j < attend_len; j += tg_size) {
        float dot = 0.0f;
        for (unsigned int d = 0; d < head_dim; d++) {
            dot += Q[q_off + d] * K[j * kv_dim + kv_off + d];
        }
        float s = dot * scale;
        scores[j] = s;
        local_max = fmaxf(local_max, s);
    }
    reduce[tid] = local_max;
    __syncthreads();
    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] = fmaxf(reduce[tid], reduce[tid + s]);
        __syncthreads();
    }
    float row_max = reduce[0];
    __syncthreads();  // barrier before reusing reduce[]

    // Exp + sum
    float local_sum = 0.0f;
    for (unsigned int j = tid; j < attend_len; j += tg_size) {
        float val = expf(scores[j] - row_max);
        scores[j] = val;
        local_sum += val;
    }
    reduce[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] += reduce[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / reduce[0];

    // Weighted V
    unsigned int out_off = qi * total_dim + head * head_dim;
    for (unsigned int d = tid; d < head_dim; d += tg_size) {
        float val = 0.0f;
        for (unsigned int j = 0; j < attend_len; j++) {
            val += scores[j] * inv_sum * V[j * kv_dim + kv_off + d];
        }
        output[out_off + d] = val;
    }
}
"#;

/// FlashAttention-style causal self-attention with GQA support.
/// Uses online softmax and K/V tiling to avoid O(N) shared memory.
/// Grid: (seq_len, n_heads), Block: (min(attend_len, MAX_THREADS))
///
/// Instead of materializing the full N-length score vector in shared memory,
/// this kernel iterates over K/V in tiles of TILE_KV and maintains running
/// softmax statistics (row max `m` and row sum `l`). Each output dimension
/// accumulates its weighted-V result in registers across tiles.
///
/// Threading model (same as the naive kernel):
///   - Phase 1 (score computation): threads split over K positions in the tile.
///     Each thread computes the full dot product for its assigned positions.
///   - Phase 2 (V accumulation): threads split over output dimensions d.
///     Each thread accumulates P[j] * V[j,d] for its assigned dimensions.
///
/// Shared memory: TILE_KV scores + MAX_THREADS reduce scratch = ~1.5 KB
///   (vs 8 KB for scores[2048] in the naive kernel)
///
/// Dynamic shared memory layout (bytes, at launch):
///   float tile_scores[TILE_KV]  — attention scores for current tile
///   float reduce[MAX_THREADS]   — scratch for parallel reductions
///   float output_acc[head_dim]  — running output accumulator (shared across phases)
/// Total: (TILE_KV + MAX_THREADS + head_dim) * 4 bytes
///   e.g. TILE_KV=64, MAX_THREADS=256, head_dim=128: (64+256+128)*4 = 1792 bytes
pub const CAUSAL_ATTENTION_FLASH_CUDA: &str = r#"
#define FA_TILE_KV 64

extern "C" __global__ void causal_attention_flash(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    const unsigned int seq_len,
    const unsigned int n_heads,
    const unsigned int n_kv_heads,
    const unsigned int head_dim)
{
    // Dynamic shared memory (allocated at launch):
    //   float tile_scores[FA_TILE_KV]
    //   float reduce[blockDim.x]
    //   float out_acc[head_dim]
    extern __shared__ float smem[];
    float* tile_scores = smem;
    float* reduce      = smem + FA_TILE_KV;
    float* out_acc     = smem + FA_TILE_KV + blockDim.x;

    const unsigned int pos     = blockIdx.x;
    const unsigned int head    = blockIdx.y;
    const unsigned int tid     = threadIdx.x;
    const unsigned int tg_size = blockDim.x;

    if (pos >= seq_len || head >= n_heads) return;

    const unsigned int total_dim    = n_heads * head_dim;
    const unsigned int kv_dim       = n_kv_heads * head_dim;
    const unsigned int heads_per_kv = n_heads / n_kv_heads;
    const unsigned int kv_head      = head / heads_per_kv;
    const unsigned int q_off        = head * head_dim;
    const unsigned int kv_off       = kv_head * head_dim;
    const float scale               = rsqrtf((float)head_dim);
    const unsigned int attend_len   = pos + 1;

    // Initialize output accumulator to zero
    for (unsigned int d = tid; d < head_dim; d += tg_size) {
        out_acc[d] = 0.0f;
    }
    __syncthreads();

    // Running online-softmax state (uniform across all threads via reductions)
    float row_m = -1e38f;  // running global max
    float row_l = 0.0f;    // running global sum of exp(score - row_m)

    // Iterate over K/V in tiles
    const unsigned int n_tiles = (attend_len + FA_TILE_KV - 1) / FA_TILE_KV;

    for (unsigned int tile = 0; tile < n_tiles; tile++) {
        const unsigned int tile_start = tile * FA_TILE_KV;
        const unsigned int tile_end   = min(tile_start + (unsigned int)FA_TILE_KV, attend_len);
        const unsigned int tile_len   = tile_end - tile_start;

        // ================================================================
        // Phase 1: Compute attention scores for this tile
        // Threads split over K positions (j) within the tile.
        // Each thread computes full dot product for its positions.
        // ================================================================
        float local_max = -1e38f;
        for (unsigned int j = tid; j < tile_len; j += tg_size) {
            float dot = 0.0f;
            const unsigned int k_base = (tile_start + j) * kv_dim + kv_off;
            const unsigned int q_base = pos * total_dim + q_off;
            for (unsigned int d = 0; d < head_dim; d++) {
                dot += Q[q_base + d] * K[k_base + d];
            }
            float s = dot * scale;
            tile_scores[j] = s;
            local_max = fmaxf(local_max, s);
        }

        // ---- Parallel max-reduce to find tile_max ----
        reduce[tid] = local_max;
        __syncthreads();
        for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
            if (tid < s) reduce[tid] = fmaxf(reduce[tid], reduce[tid + s]);
            __syncthreads();
        }
        float tile_max = reduce[0];
        __syncthreads();  // barrier before reusing reduce[]

        // ---- Online softmax: merge tile stats with running stats ----
        float m_new = fmaxf(row_m, tile_max);
        float rescale = expf(row_m - m_new);  // factor to rescale old accumulator

        // ---- Compute exp(score - m_new) in-place, and parallel sum-reduce ----
        float local_sum = 0.0f;
        for (unsigned int j = tid; j < tile_len; j += tg_size) {
            float p = expf(tile_scores[j] - m_new);
            tile_scores[j] = p;
            local_sum += p;
        }
        reduce[tid] = local_sum;
        __syncthreads();
        for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
            if (tid < s) reduce[tid] += reduce[tid + s];
            __syncthreads();
        }
        float tile_sum = reduce[0];

        // ================================================================
        // Phase 2: Accumulate P @ V for this tile
        // Threads split over output dimensions (d).
        // Must first rescale old accumulator, then add new contribution.
        // ================================================================
        for (unsigned int d = tid; d < head_dim; d += tg_size) {
            // Rescale old accumulator for updated max
            float old_val = out_acc[d] * rescale;

            // Accumulate new tile's contribution
            float new_val = 0.0f;
            for (unsigned int j = 0; j < tile_len; j++) {
                new_val += tile_scores[j] * V[(tile_start + j) * kv_dim + kv_off + d];
            }

            out_acc[d] = old_val + new_val;
        }

        // Update running softmax statistics
        row_l = row_l * rescale + tile_sum;
        row_m = m_new;

        __syncthreads();
    }

    // ---- Final normalization: O = out_acc / row_l ----
    float inv_l = (row_l > 0.0f) ? (1.0f / row_l) : 0.0f;
    for (unsigned int d = tid; d < head_dim; d += tg_size) {
        output[pos * total_dim + q_off + d] = out_acc[d] * inv_l;
    }
}
"#;

// ---- bf16 variants ----

/// bf16 causal self-attention. Q,K,V,output are bf16. Shared memory stays f32.
pub const CAUSAL_ATTENTION_BF16_CUDA: &str = r#"
#define MAX_SEQ 2048
__device__ float bf16_to_float(unsigned short bits) {
    return __int_as_float(((unsigned int)bits) << 16);
}
__device__ unsigned short float_to_bf16(float val) {
    unsigned int bits = __float_as_int(val);
    unsigned int lsb = (bits >> 16) & 1;
    bits += 0x7FFF + lsb;
    return (unsigned short)(bits >> 16);
}

extern "C" __global__ void causal_attention_bf16(
    const unsigned short* __restrict__ Q,
    const unsigned short* __restrict__ K,
    const unsigned short* __restrict__ V,
    unsigned short* __restrict__ output,
    const unsigned int seq_len,
    const unsigned int n_heads,
    const unsigned int n_kv_heads,
    const unsigned int head_dim)
{
    __shared__ float scores[MAX_SEQ];
    __shared__ float reduce[256];

    unsigned int pos = blockIdx.x;
    unsigned int head = blockIdx.y;
    unsigned int tid = threadIdx.x;
    unsigned int tg_size = blockDim.x;

    if (pos >= seq_len || head >= n_heads) return;

    unsigned int total_dim = n_heads * head_dim;
    unsigned int kv_dim = n_kv_heads * head_dim;
    unsigned int heads_per_kv = n_heads / n_kv_heads;
    unsigned int kv_head = head / heads_per_kv;
    unsigned int q_off = head * head_dim;
    unsigned int kv_off = kv_head * head_dim;
    float scale = rsqrtf(float(head_dim));
    unsigned int attend_len = pos + 1;

    // Step 1: compute scores
    float local_max = -1e38f;
    for (unsigned int j = tid; j < attend_len; j += tg_size) {
        float dot = 0.0f;
        for (unsigned int d = 0; d < head_dim; d++) {
            dot += bf16_to_float(Q[pos * total_dim + q_off + d]) * bf16_to_float(K[j * kv_dim + kv_off + d]);
        }
        float s = dot * scale;
        scores[j] = s;
        local_max = fmaxf(local_max, s);
    }
    reduce[tid] = local_max;
    __syncthreads();
    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] = fmaxf(reduce[tid], reduce[tid + s]);
        __syncthreads();
    }
    float row_max = reduce[0];
    __syncthreads();  // barrier before reusing reduce[]

    // Step 2: exp + sum
    float local_sum = 0.0f;
    for (unsigned int j = tid; j < attend_len; j += tg_size) {
        float val = expf(scores[j] - row_max);
        scores[j] = val;
        local_sum += val;
    }
    reduce[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] += reduce[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / reduce[0];

    // Step 3: weighted V
    for (unsigned int d = tid; d < head_dim; d += tg_size) {
        float val = 0.0f;
        for (unsigned int j = 0; j < attend_len; j++) {
            val += scores[j] * inv_sum * bf16_to_float(V[j * kv_dim + kv_off + d]);
        }
        output[pos * total_dim + q_off + d] = float_to_bf16(val);
    }
}
"#;

/// bf16 KV-cached attention for single-query decode.
pub const KV_ATTENTION_BF16_CUDA: &str = r#"
#define MAX_SEQ 2048
__device__ float bf16_to_float(unsigned short bits) {
    return __int_as_float(((unsigned int)bits) << 16);
}
__device__ unsigned short float_to_bf16(float val) {
    unsigned int bits = __float_as_int(val);
    unsigned int lsb = (bits >> 16) & 1;
    bits += 0x7FFF + lsb;
    return (unsigned short)(bits >> 16);
}

extern "C" __global__ void kv_attention_bf16(
    const unsigned short* __restrict__ Q,
    const unsigned short* __restrict__ K,
    const unsigned short* __restrict__ V,
    unsigned short* __restrict__ output,
    const unsigned int cache_len,
    const unsigned int n_heads,
    const unsigned int n_kv_heads,
    const unsigned int head_dim)
{
    __shared__ float scores[MAX_SEQ];
    __shared__ float reduce[256];

    unsigned int head = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int tg_size = blockDim.x;
    if (head >= n_heads) return;

    unsigned int kv_dim = n_kv_heads * head_dim;
    unsigned int heads_per_kv = n_heads / n_kv_heads;
    unsigned int kv_head = head / heads_per_kv;
    unsigned int q_off = head * head_dim;
    unsigned int kv_off = kv_head * head_dim;
    float scale = rsqrtf(float(head_dim));

    float local_max = -1e38f;
    for (unsigned int j = tid; j < cache_len; j += tg_size) {
        float dot = 0.0f;
        for (unsigned int d = 0; d < head_dim; d++) {
            dot += bf16_to_float(Q[q_off + d]) * bf16_to_float(K[j * kv_dim + kv_off + d]);
        }
        float s = dot * scale;
        scores[j] = s;
        local_max = fmaxf(local_max, s);
    }
    reduce[tid] = local_max;
    __syncthreads();
    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] = fmaxf(reduce[tid], reduce[tid + s]);
        __syncthreads();
    }
    float row_max = reduce[0];
    __syncthreads();  // barrier before reusing reduce[]

    float local_sum = 0.0f;
    for (unsigned int j = tid; j < cache_len; j += tg_size) {
        float val = expf(scores[j] - row_max);
        scores[j] = val;
        local_sum += val;
    }
    reduce[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] += reduce[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / reduce[0];

    for (unsigned int d = tid; d < head_dim; d += tg_size) {
        float val = 0.0f;
        for (unsigned int j = 0; j < cache_len; j++) {
            val += scores[j] * inv_sum * bf16_to_float(V[j * kv_dim + kv_off + d]);
        }
        output[q_off + d] = float_to_bf16(val);
    }
}
"#;

/// bf16 KV-cached attention for batched prefill.
pub const KV_ATTENTION_PREFILL_BF16_CUDA: &str = r#"
#define MAX_SEQ 2048
__device__ float bf16_to_float(unsigned short bits) {
    return __int_as_float(((unsigned int)bits) << 16);
}
__device__ unsigned short float_to_bf16(float val) {
    unsigned int bits = __float_as_int(val);
    unsigned int lsb = (bits >> 16) & 1;
    bits += 0x7FFF + lsb;
    return (unsigned short)(bits >> 16);
}

extern "C" __global__ void kv_attention_prefill_bf16(
    const unsigned short* __restrict__ Q,
    const unsigned short* __restrict__ K,
    const unsigned short* __restrict__ V,
    unsigned short* __restrict__ output,
    const unsigned int cache_start,
    const unsigned int q_len,
    const unsigned int n_heads,
    const unsigned int n_kv_heads,
    const unsigned int head_dim)
{
    __shared__ float scores[MAX_SEQ];
    __shared__ float reduce[256];

    unsigned int qi = blockIdx.x;
    unsigned int head = blockIdx.y;
    unsigned int tid = threadIdx.x;
    unsigned int tg_size = blockDim.x;

    if (qi >= q_len || head >= n_heads) return;

    unsigned int total_dim = n_heads * head_dim;
    unsigned int kv_dim = n_kv_heads * head_dim;
    unsigned int heads_per_kv = n_heads / n_kv_heads;
    unsigned int kv_head = head / heads_per_kv;
    unsigned int q_off = qi * total_dim + head * head_dim;
    unsigned int kv_off = kv_head * head_dim;
    float scale = rsqrtf(float(head_dim));
    unsigned int attend_len = cache_start + qi + 1;

    float local_max = -1e38f;
    for (unsigned int j = tid; j < attend_len; j += tg_size) {
        float dot = 0.0f;
        for (unsigned int d = 0; d < head_dim; d++) {
            dot += bf16_to_float(Q[q_off + d]) * bf16_to_float(K[j * kv_dim + kv_off + d]);
        }
        float s = dot * scale;
        scores[j] = s;
        local_max = fmaxf(local_max, s);
    }
    reduce[tid] = local_max;
    __syncthreads();
    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] = fmaxf(reduce[tid], reduce[tid + s]);
        __syncthreads();
    }
    float row_max = reduce[0];
    __syncthreads();  // barrier before reusing reduce[]

    float local_sum = 0.0f;
    for (unsigned int j = tid; j < attend_len; j += tg_size) {
        float val = expf(scores[j] - row_max);
        scores[j] = val;
        local_sum += val;
    }
    reduce[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] += reduce[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / reduce[0];

    unsigned int out_off = qi * total_dim + head * head_dim;
    for (unsigned int d = tid; d < head_dim; d += tg_size) {
        float val = 0.0f;
        for (unsigned int j = 0; j < attend_len; j++) {
            val += scores[j] * inv_sum * bf16_to_float(V[j * kv_dim + kv_off + d]);
        }
        output[out_off + d] = float_to_bf16(val);
    }
}
"#;
