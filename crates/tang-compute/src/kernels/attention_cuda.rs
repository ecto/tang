//! CUDA attention kernels: causal self-attention and KV-cached attention.

/// Causal self-attention kernel with GQA support.
/// Grid: (seq_len, n_heads), Block: (min(seq_len, 256))
pub const CAUSAL_ATTENTION_CUDA: &str = r#"
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
    __shared__ float scores[256];
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
        scores[j % 256] = s;
        local_max = fmaxf(local_max, s);
    }
    reduce[tid] = local_max;
    __syncthreads();
    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] = fmaxf(reduce[tid], reduce[tid + s]);
        __syncthreads();
    }
    float row_max = reduce[0];

    // Step 2: exp + sum
    float local_sum = 0.0f;
    for (unsigned int j = tid; j < attend_len; j += tg_size) {
        float val = expf(scores[j % 256] - row_max);
        scores[j % 256] = val;
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
            val += scores[j % 256] * inv_sum * V[j * kv_dim + kv_off + d];
        }
        output[pos * total_dim + q_off + d] = val;
    }
}
"#;

/// KV-cached attention for autoregressive decoding (single query).
/// Grid: (n_heads), Block: (min(total_len, 256))
pub const KV_ATTENTION_CUDA: &str = r#"
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
    __shared__ float scores[256];
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
        scores[j % 256] = s;
        local_max = fmaxf(local_max, s);
    }
    reduce[tid] = local_max;
    __syncthreads();
    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] = fmaxf(reduce[tid], reduce[tid + s]);
        __syncthreads();
    }
    float row_max = reduce[0];

    // Exp + sum
    float local_sum = 0.0f;
    for (unsigned int j = tid; j < cache_len; j += tg_size) {
        float val = expf(scores[j % 256] - row_max);
        scores[j % 256] = val;
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
            val += scores[j % 256] * inv_sum * V[j * kv_dim + kv_off + d];
        }
        output[q_off + d] = val;
    }
}
"#;

/// KV-cached attention for batched prefill.
/// Grid: (q_len, n_heads), Block: (min(max_attend, 256))
pub const KV_ATTENTION_PREFILL_CUDA: &str = r#"
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
    __shared__ float scores[256];
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
        scores[j % 256] = s;
        local_max = fmaxf(local_max, s);
    }
    reduce[tid] = local_max;
    __syncthreads();
    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] = fmaxf(reduce[tid], reduce[tid + s]);
        __syncthreads();
    }
    float row_max = reduce[0];

    // Exp + sum
    float local_sum = 0.0f;
    for (unsigned int j = tid; j < attend_len; j += tg_size) {
        float val = expf(scores[j % 256] - row_max);
        scores[j % 256] = val;
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
            val += scores[j % 256] * inv_sum * V[j * kv_dim + kv_off + d];
        }
        output[out_off + d] = val;
    }
}
"#;

// ---- bf16 variants ----

/// bf16 causal self-attention. Q,K,V,output are bf16. Shared memory stays f32.
pub const CAUSAL_ATTENTION_BF16_CUDA: &str = r#"
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
    __shared__ float scores[256];
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
        scores[j % 256] = s;
        local_max = fmaxf(local_max, s);
    }
    reduce[tid] = local_max;
    __syncthreads();
    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] = fmaxf(reduce[tid], reduce[tid + s]);
        __syncthreads();
    }
    float row_max = reduce[0];

    // Step 2: exp + sum
    float local_sum = 0.0f;
    for (unsigned int j = tid; j < attend_len; j += tg_size) {
        float val = expf(scores[j % 256] - row_max);
        scores[j % 256] = val;
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
            val += scores[j % 256] * inv_sum * bf16_to_float(V[j * kv_dim + kv_off + d]);
        }
        output[pos * total_dim + q_off + d] = float_to_bf16(val);
    }
}
"#;

/// bf16 KV-cached attention for single-query decode.
pub const KV_ATTENTION_BF16_CUDA: &str = r#"
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
    __shared__ float scores[256];
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
        scores[j % 256] = s;
        local_max = fmaxf(local_max, s);
    }
    reduce[tid] = local_max;
    __syncthreads();
    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] = fmaxf(reduce[tid], reduce[tid + s]);
        __syncthreads();
    }
    float row_max = reduce[0];

    float local_sum = 0.0f;
    for (unsigned int j = tid; j < cache_len; j += tg_size) {
        float val = expf(scores[j % 256] - row_max);
        scores[j % 256] = val;
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
            val += scores[j % 256] * inv_sum * bf16_to_float(V[j * kv_dim + kv_off + d]);
        }
        output[q_off + d] = float_to_bf16(val);
    }
}
"#;

/// bf16 KV-cached attention for batched prefill.
pub const KV_ATTENTION_PREFILL_BF16_CUDA: &str = r#"
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
    __shared__ float scores[256];
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
        scores[j % 256] = s;
        local_max = fmaxf(local_max, s);
    }
    reduce[tid] = local_max;
    __syncthreads();
    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] = fmaxf(reduce[tid], reduce[tid + s]);
        __syncthreads();
    }
    float row_max = reduce[0];

    float local_sum = 0.0f;
    for (unsigned int j = tid; j < attend_len; j += tg_size) {
        float val = expf(scores[j % 256] - row_max);
        scores[j % 256] = val;
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
            val += scores[j % 256] * inv_sum * bf16_to_float(V[j * kv_dim + kv_off + d]);
        }
        output[out_off + d] = float_to_bf16(val);
    }
}
"#;
