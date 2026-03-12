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
    const unsigned int head_dim,
    const unsigned int batch_size)
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
    const unsigned int batch   = blockIdx.z;
    const unsigned int tid     = threadIdx.x;
    const unsigned int tg_size = blockDim.x;

    if (pos >= seq_len || head >= n_heads || batch >= batch_size) return;

    const unsigned int total_dim    = n_heads * head_dim;
    const unsigned int kv_dim       = n_kv_heads * head_dim;
    const unsigned int heads_per_kv = n_heads / n_kv_heads;
    const unsigned int kv_head      = head / heads_per_kv;
    const unsigned int q_off        = batch * seq_len * total_dim + head * head_dim;
    const unsigned int kv_off       = batch * seq_len * kv_dim + kv_head * head_dim;
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

/// Tensor-core FlashAttention forward kernel using wmma (m16n16k16 bf16→f32).
///
/// Native bf16 I/O — no external conversion passes needed.
/// Uses online softmax (Milakov-Gimelshein) with KV tiling.
///
/// Grid: (ceil(seq_len/16), n_heads, batch_size)
/// Block: (128) = 4 warps
///
/// Each block processes 16 Q rows against all KV in tiles of 16.
/// 4 warps split the head_dim k-steps for Q×K^T, then cooperate on softmax + P×V.
///
/// Dynamic shared memory layout (for head_dim=64):
///   bf16 Q_smem[16×64]    = 2048B   — Q tile (loaded once)
///   bf16 KV_smem[16×64]   = 2048B   — K or V tile (reloaded per KV tile)
///   f32  S[16×16]          = 1024B   — score tile (Q×K^T result)
///   f32  O_acc[16×64]      = 4096B   — output accumulator
///   f32  row_m[16]         = 64B     — running row max
///   f32  row_l[16]         = 64B     — running row sum
///   bf16 P_bf16[16×16]     = 512B    — softmax probs in bf16 for wmma
///   f32  S_warp[4][16×16]  = 4096B   — per-warp wmma scratch
///   Total: ~14KB
///
/// head_dim must be a multiple of 16 (e.g. 64, 128).
pub const CAUSAL_ATTENTION_FLASH_TC_CUDA: &str = r#"
#include <mma.h>
using namespace nvcuda;

#define TC_TILE_Q  16
#define TC_TILE_KV 16
#define TC_BLOCK   128
#define TC_WARPS   4

extern "C" __global__ void causal_attention_flash_tc(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ output,
    const unsigned int seq_len,
    const unsigned int n_heads,
    const unsigned int n_kv_heads,
    const unsigned int head_dim,
    const unsigned int batch_size)
{
    extern __shared__ char smem_raw[];

    const unsigned int q_tile_idx = blockIdx.x;
    const unsigned int head       = blockIdx.y;
    const unsigned int batch      = blockIdx.z;
    const unsigned int tid        = threadIdx.x;
    const unsigned int warp_id    = tid / 32;

    const unsigned int q_start = q_tile_idx * TC_TILE_Q;
    if (q_start >= seq_len) return;
    const unsigned int q_end = min(q_start + (unsigned int)TC_TILE_Q, seq_len);

    const unsigned int hd           = head_dim;
    const unsigned int total_dim    = n_heads * hd;
    const unsigned int kv_dim       = n_kv_heads * hd;
    const unsigned int heads_per_kv = n_heads / n_kv_heads;
    const unsigned int kv_head      = head / heads_per_kv;
    const float scale               = rsqrtf((float)hd);

    // Global memory pointers for this batch/head
    const __nv_bfloat16* Q_g = Q + (unsigned long long)batch * seq_len * total_dim;
    const __nv_bfloat16* K_g = K + (unsigned long long)batch * seq_len * kv_dim;
    const __nv_bfloat16* V_g = V + (unsigned long long)batch * seq_len * kv_dim;
    __nv_bfloat16*       O_g = output + (unsigned long long)batch * seq_len * total_dim;

    // Shared memory layout (all offsets in bytes)
    __nv_bfloat16* Q_smem  = (__nv_bfloat16*)smem_raw;
    __nv_bfloat16* KV_smem = Q_smem + TC_TILE_Q * hd;
    float* S               = (float*)(KV_smem + TC_TILE_KV * hd);
    float* O_acc           = S + TC_TILE_Q * TC_TILE_KV;
    float* row_m           = O_acc + TC_TILE_Q * hd;
    float* row_l           = row_m + TC_TILE_Q;
    __nv_bfloat16* P_bf16  = (__nv_bfloat16*)(row_l + TC_TILE_Q);
    float* S_warp          = (float*)(P_bf16 + TC_TILE_Q * TC_TILE_KV);
    // S_warp layout: [4][256] = 4 warps × (16×16) floats

    // Initialize
    for (unsigned int i = tid; i < TC_TILE_Q * hd; i += TC_BLOCK)
        O_acc[i] = 0.0f;
    if (tid < TC_TILE_Q) {
        row_m[tid] = -1e38f;
        row_l[tid] = 0.0f;
    }

    // Load Q tile: Q_smem[row, d] = Q_g[(q_start+row)*total_dim + head*hd + d]
    for (unsigned int i = tid; i < TC_TILE_Q * hd; i += TC_BLOCK) {
        unsigned int row = i / hd;
        unsigned int col = i % hd;
        Q_smem[i] = (q_start + row < seq_len)
            ? Q_g[(q_start + row) * total_dim + head * hd + col]
            : __float2bfloat16(0.0f);
    }
    __syncthreads();

    // KV tile loop (causal: attend up to q_end-1)
    const unsigned int max_attend = q_end;
    const unsigned int n_kv_tiles = (max_attend + TC_TILE_KV - 1) / TC_TILE_KV;
    const unsigned int n_k_steps  = hd / 16;

    for (unsigned int kv_tile = 0; kv_tile < n_kv_tiles; kv_tile++) {
        const unsigned int kv_start = kv_tile * TC_TILE_KV;
        const unsigned int kv_end_t = min(kv_start + (unsigned int)TC_TILE_KV, max_attend);
        const unsigned int kv_len   = kv_end_t - kv_start;

        // Load K tile into KV_smem
        for (unsigned int i = tid; i < TC_TILE_KV * hd; i += TC_BLOCK) {
            unsigned int row = i / hd;
            unsigned int col = i % hd;
            KV_smem[i] = (row < kv_len)
                ? K_g[(kv_start + row) * kv_dim + kv_head * hd + col]
                : __float2bfloat16(0.0f);
        }
        __syncthreads();

        // ---- Q×K^T via wmma: S[16×16] = sum_k Q[16×16] × K[16×16]^T ----
        // Each warp handles k_steps strided by 4, accumulates partial S in a fragment,
        // stores to S_warp[warp_id], then all threads reduce into S.
        {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
            wmma::fill_fragment(acc_frag, 0.0f);

            for (unsigned int ks = warp_id; ks < n_k_steps; ks += TC_WARPS) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
                wmma::load_matrix_sync(a_frag, Q_smem + ks * 16, hd);
                wmma::load_matrix_sync(b_frag, KV_smem + ks * 16, hd);
                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }

            // Store per-warp result to scratch
            wmma::store_matrix_sync(S_warp + warp_id * 256, acc_frag, 16, wmma::mem_row_major);
        }
        __syncthreads();

        // Reduce across warps into S
        {
            unsigned int active_warps = min((unsigned int)TC_WARPS, n_k_steps);
            for (unsigned int i = tid; i < TC_TILE_Q * TC_TILE_KV; i += TC_BLOCK) {
                float sum = 0.0f;
                for (unsigned int w = 0; w < active_warps; w++)
                    sum += S_warp[w * 256 + i];
                S[i] = sum;
            }
        }
        __syncthreads();

        // ---- Apply scale + causal mask ----
        for (unsigned int i = tid; i < TC_TILE_Q * TC_TILE_KV; i += TC_BLOCK) {
            unsigned int q_row = i / TC_TILE_KV;
            unsigned int k_col = i % TC_TILE_KV;
            unsigned int gq = q_start + q_row;
            unsigned int gk = kv_start + k_col;
            if (gq >= seq_len || k_col >= kv_len || gk > gq)
                S[i] = -1e38f;
            else
                S[i] *= scale;
        }
        __syncthreads();

        // ---- Online softmax per row (8 threads per row via warp shuffles) ----
        {
            const unsigned int tpr = TC_BLOCK / TC_TILE_Q;  // 8 threads per row
            const unsigned int my_row  = tid / tpr;
            const unsigned int my_lane = tid % tpr;

            if (my_row < TC_TILE_Q) {
                // Row max
                float lmax = -1e38f;
                for (unsigned int j = my_lane; j < TC_TILE_KV; j += tpr)
                    lmax = fmaxf(lmax, S[my_row * TC_TILE_KV + j]);
                for (unsigned int off = tpr / 2; off > 0; off >>= 1)
                    lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffff, lmax, off));
                float tile_max = lmax;

                float old_m = row_m[my_row];
                float new_m = fmaxf(old_m, tile_max);
                float rescale = expf(old_m - new_m);

                // Exp + sum
                float lsum = 0.0f;
                for (unsigned int j = my_lane; j < TC_TILE_KV; j += tpr) {
                    float p = expf(S[my_row * TC_TILE_KV + j] - new_m);
                    S[my_row * TC_TILE_KV + j] = p;
                    lsum += p;
                }
                for (unsigned int off = tpr / 2; off > 0; off >>= 1)
                    lsum += __shfl_xor_sync(0xffffffff, lsum, off);

                // Rescale O_acc
                for (unsigned int d = my_lane; d < hd; d += tpr)
                    O_acc[my_row * hd + d] *= rescale;

                if (my_lane == 0) {
                    row_l[my_row] = row_l[my_row] * rescale + lsum;
                    row_m[my_row] = new_m;
                }
            }
        }
        __syncthreads();

        // ---- Convert S to bf16 P_bf16 (separate buffer, no aliasing) ----
        for (unsigned int i = tid; i < TC_TILE_Q * TC_TILE_KV; i += TC_BLOCK)
            P_bf16[i] = __float2bfloat16(S[i]);

        // ---- Load V tile into KV_smem ----
        for (unsigned int i = tid; i < TC_TILE_KV * hd; i += TC_BLOCK) {
            unsigned int row = i / hd;
            unsigned int col = i % hd;
            KV_smem[i] = (row < kv_len)
                ? V_g[(kv_start + row) * kv_dim + kv_head * hd + col]
                : __float2bfloat16(0.0f);
        }
        __syncthreads();

        // ---- O_acc += P × V via wmma ----
        // P is [16,16] bf16, V is [16,hd] bf16, O_acc is [16,hd] f32
        // Split d_steps (hd/16) across warps
        {
            const unsigned int n_d_steps = hd / 16;
            for (unsigned int ds = warp_id; ds < n_d_steps; ds += TC_WARPS) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> p_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> v_frag;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frag;

                wmma::load_matrix_sync(o_frag, O_acc + ds * 16, hd, wmma::mem_row_major);
                wmma::load_matrix_sync(p_frag, P_bf16, TC_TILE_KV);
                wmma::load_matrix_sync(v_frag, KV_smem + ds * 16, hd);
                wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
                wmma::store_matrix_sync(O_acc + ds * 16, o_frag, hd, wmma::mem_row_major);
            }
        }
        __syncthreads();
    }

    // ---- Final normalization: O = O_acc / row_l → write bf16 to global ----
    for (unsigned int i = tid; i < TC_TILE_Q * hd; i += TC_BLOCK) {
        unsigned int row = i / hd;
        unsigned int gq = q_start + row;
        if (gq < seq_len) {
            unsigned int col = i % hd;
            float inv_l = (row_l[row] > 0.0f) ? (1.0f / row_l[row]) : 0.0f;
            O_g[gq * total_dim + head * hd + col] = __float2bfloat16(O_acc[i] * inv_l);
        }
    }
}
"#;
