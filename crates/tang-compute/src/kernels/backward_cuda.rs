//! CUDA backward kernels for training: transpose, softmax_backward, rms_norm_backward,
//! embedding_backward, cross_entropy_forward_backward, causal_attention_backward.

/// On-device 2D transpose: [rows, cols] → [cols, rows].
/// Grid: (ceil(cols/16), ceil(rows/16)), Block: (16, 16)
pub const TRANSPOSE_2D_CUDA: &str = r#"
extern "C" __global__ void transpose_2d(
    const float* __restrict__ input,
    float* __restrict__ output,
    const unsigned int rows,
    const unsigned int cols)
{
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= rows || c >= cols) return;
    output[c * rows + r] = input[r * cols + c];
}
"#;

/// Softmax backward: grad_input[i,j] = sm[i,j] * (grad[i,j] - dot(sm[i,:], grad[i,:])).
/// One block per row.
/// Grid: (n_rows), Block: (min(row_len, 256))
pub const SOFTMAX_BACKWARD_CUDA: &str = r#"
extern "C" __global__ void softmax_backward(
    const float* __restrict__ sm,
    const float* __restrict__ grad_out,
    float* __restrict__ grad_in,
    const unsigned int n_rows,
    const unsigned int row_len)
{
    __shared__ float shared[256];

    unsigned int row = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int tg_size = blockDim.x;
    if (row >= n_rows) return;

    unsigned int base = row * row_len;

    // Compute dot(sm, grad) for this row
    float local_dot = 0.0f;
    for (unsigned int i = tid; i < row_len; i += tg_size) {
        local_dot += sm[base + i] * grad_out[base + i];
    }
    shared[tid] = local_dot;
    __syncthreads();

    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    float row_dot = shared[0];

    // grad_input = sm * (grad - dot)
    for (unsigned int i = tid; i < row_len; i += tg_size) {
        grad_in[base + i] = sm[base + i] * (grad_out[base + i] - row_dot);
    }
}
"#;

/// RMS norm backward.
/// Outputs: grad_input [n_groups * dim], grad_weight [dim] (atomically accumulated).
/// Grid: (n_groups), Block: (min(dim, 256))
pub const RMS_NORM_BACKWARD_CUDA: &str = r#"
extern "C" __global__ void rms_norm_backward(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ grad_out,
    float* __restrict__ grad_input,
    float* __restrict__ grad_weight,
    const unsigned int n_groups,
    const unsigned int dim,
    const float eps)
{
    __shared__ float shared[256];

    unsigned int group = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int tg_size = blockDim.x;
    if (group >= n_groups) return;

    unsigned int base = group * dim;

    // Phase 1: compute sum of squares
    float local_sq = 0.0f;
    for (unsigned int i = tid; i < dim; i += tg_size) {
        float v = input[base + i];
        local_sq += v * v;
    }
    shared[tid] = local_sq;
    __syncthreads();

    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    float rms_sq = shared[0] / float(dim) + eps;
    float inv_rms = rsqrtf(rms_sq);
    __syncthreads();  // barrier before Phase 2 reuses shared[]

    // Phase 2: compute sum(x * w * grad_out)
    float local_xwg = 0.0f;
    for (unsigned int i = tid; i < dim; i += tg_size) {
        local_xwg += input[base + i] * weight[i] * grad_out[base + i];
    }
    shared[tid] = local_xwg;
    __syncthreads();

    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    float sum_xwg = shared[0];

    // Phase 3: compute grad_input and accumulate grad_weight
    for (unsigned int i = tid; i < dim; i += tg_size) {
        float x = input[base + i];
        float go = grad_out[base + i];
        float w = weight[i];

        grad_input[base + i] = w * inv_rms * go
            - x * inv_rms * inv_rms * inv_rms / float(dim) * sum_xwg;

        // Atomic accumulate grad_weight across groups
        atomicAdd(&grad_weight[i], x * inv_rms * go);
    }
}
"#;

/// Embedding backward: scatter-add grad_output into grad_weight.
/// Grid: (ceil(seq_len/256)), Block: (256)
pub const EMBEDDING_BACKWARD_CUDA: &str = r#"
extern "C" __global__ void embedding_backward(
    const float* __restrict__ grad_out,
    const unsigned int* __restrict__ ids,
    float* __restrict__ grad_weight,
    const unsigned int vocab_size,
    const unsigned int seq_len,
    const unsigned int dim)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= seq_len) return;

    unsigned int id = ids[gid];
    for (unsigned int d = 0; d < dim; d++) {
        atomicAdd(&grad_weight[id * dim + d], grad_out[gid * dim + d]);
    }
}
"#;

/// Fused cross-entropy forward + backward.
/// One block per position (row).
/// Grid: (n_positions), Block: (min(vocab_size, 256))
pub const CROSS_ENTROPY_CUDA: &str = r#"
extern "C" __global__ void cross_entropy_fwd_bwd(
    const float* __restrict__ logits,
    const unsigned int* __restrict__ targets,
    float* __restrict__ grad,
    float* __restrict__ loss_out,
    const unsigned int n_pos,
    const unsigned int vocab,
    const unsigned int pad_id,
    const unsigned int count)
{
    __shared__ float shared[256];

    unsigned int row = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int tg_size = blockDim.x;
    if (row >= n_pos) return;
    if (count == 0) return;

    unsigned int target = targets[row];
    unsigned int base = row * vocab;

    if (target == pad_id) {
        // Zero out gradient for padded positions
        for (unsigned int i = tid; i < vocab; i += tg_size) {
            grad[base + i] = 0.0f;
        }
        return;
    }

    // Phase 1: find max
    float local_max = -1e38f;
    for (unsigned int i = tid; i < vocab; i += tg_size) {
        local_max = fmaxf(local_max, logits[base + i]);
    }
    shared[tid] = local_max;
    __syncthreads();
    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        __syncthreads();
    }
    float row_max = shared[0];
    __syncthreads();  // barrier before Phase 2 reuses shared[]

    // Phase 2: exp sum
    float local_sum = 0.0f;
    for (unsigned int i = tid; i < vocab; i += tg_size) {
        local_sum += expf(logits[base + i] - row_max);
    }
    shared[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    float row_sum = shared[0];

    // Loss contribution from this position (only one thread writes)
    if (tid == 0) {
        float log_prob = (logits[base + target] - row_max) - logf(row_sum);
        atomicAdd(&loss_out[0], -log_prob / float(count));
    }

    // Gradient: (softmax - one_hot) / count
    float inv_count = 1.0f / float(count);
    float inv_sum = 1.0f / row_sum;
    for (unsigned int i = tid; i < vocab; i += tg_size) {
        float sm = expf(logits[base + i] - row_max) * inv_sum;
        float one_hot = (i == target) ? 1.0f : 0.0f;
        grad[base + i] = (sm - one_hot) * inv_count;
    }
}
"#;

/// Causal attention backward with GQA support.
/// Recomputes softmax probs from Q,K, then computes grad_Q, grad_K, grad_V.
///
/// Grid: (seq_len, n_heads), Block: (min(head_dim, 256))
/// grad_K and grad_V use atomicAdd (multiple positions/heads write to same KV row).
///
/// Max sequence length: 2048 (limited by shared memory).
pub const CAUSAL_ATTENTION_BACKWARD_CUDA: &str = r#"
#define MAX_SEQ 2048

extern "C" __global__ void causal_attention_backward(
    const float* __restrict__ grad_out,
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ grad_Q,
    float* __restrict__ grad_K,
    float* __restrict__ grad_V,
    const unsigned int seq_len,
    const unsigned int n_heads,
    const unsigned int n_kv_heads,
    const unsigned int head_dim)
{
    __shared__ float probs[MAX_SEQ];
    __shared__ float grad_s[MAX_SEQ];
    __shared__ float partials[256];

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

    // ---- Phase 1: Compute attention scores ----
    for (unsigned int j = 0; j < attend_len; j++) {
        float local_dot = 0.0f;
        for (unsigned int d = tid; d < head_dim; d += tg_size) {
            local_dot += Q[pos * total_dim + q_off + d] * K[j * kv_dim + kv_off + d];
        }
        partials[tid] = local_dot;
        __syncthreads();
        for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
            if (tid < s) partials[tid] += partials[tid + s];
            __syncthreads();
        }
        if (tid == 0) {
            probs[j] = partials[0] * scale;
        }
        __syncthreads();
    }

    // ---- Phase 2: Softmax ----
    // Find max
    float local_max = -1e38f;
    for (unsigned int j = tid; j < attend_len; j += tg_size) {
        local_max = fmaxf(local_max, probs[j]);
    }
    partials[tid] = local_max;
    __syncthreads();
    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) partials[tid] = fmaxf(partials[tid], partials[tid + s]);
        __syncthreads();
    }
    float row_max = partials[0];
    __syncthreads();

    // Exp
    for (unsigned int j = tid; j < attend_len; j += tg_size) {
        probs[j] = expf(probs[j] - row_max);
    }
    __syncthreads();

    // Sum
    float local_sum = 0.0f;
    for (unsigned int j = tid; j < attend_len; j += tg_size) {
        local_sum += probs[j];
    }
    partials[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) partials[tid] += partials[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / partials[0];
    __syncthreads();

    // Normalize
    for (unsigned int j = tid; j < attend_len; j += tg_size) {
        probs[j] *= inv_sum;
    }
    __syncthreads();

    // ---- Phase 3: grad_P[j] = dot(grad_out[pos], V[j]) ----
    for (unsigned int j = 0; j < attend_len; j++) {
        float local_dot = 0.0f;
        for (unsigned int d = tid; d < head_dim; d += tg_size) {
            local_dot += grad_out[pos * total_dim + q_off + d] * V[j * kv_dim + kv_off + d];
        }
        partials[tid] = local_dot;
        __syncthreads();
        for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
            if (tid < s) partials[tid] += partials[tid + s];
            __syncthreads();
        }
        if (tid == 0) {
            grad_s[j] = partials[0];
        }
        __syncthreads();
    }

    // ---- Phase 4: Softmax backward ----
    // dot_pq = sum(P[j] * grad_P[j])
    float local_dpq = 0.0f;
    for (unsigned int j = tid; j < attend_len; j += tg_size) {
        local_dpq += probs[j] * grad_s[j];
    }
    partials[tid] = local_dpq;
    __syncthreads();
    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) partials[tid] += partials[tid + s];
        __syncthreads();
    }
    float dot_pq = partials[0];
    __syncthreads();

    // grad_S[j] = P[j] * (grad_P[j] - dot_pq)
    for (unsigned int j = tid; j < attend_len; j += tg_size) {
        grad_s[j] = probs[j] * (grad_s[j] - dot_pq);
    }
    __syncthreads();

    // ---- Phase 5: Accumulate gradients ----
    // grad_Q[pos, d] = scale * sum_j(grad_S[j] * K[j, d])
    for (unsigned int d = tid; d < head_dim; d += tg_size) {
        float acc = 0.0f;
        for (unsigned int j = 0; j < attend_len; j++) {
            acc += grad_s[j] * K[j * kv_dim + kv_off + d];
        }
        grad_Q[pos * total_dim + q_off + d] = acc * scale;
    }

    // grad_K[j, d] += scale * grad_S[j] * Q[pos, d]  — atomic
    // grad_V[j, d] += P[j] * grad_out[pos, d]        — atomic
    for (unsigned int j = 0; j < attend_len; j++) {
        float gs = grad_s[j];
        float p = probs[j];
        for (unsigned int d = tid; d < head_dim; d += tg_size) {
            atomicAdd(&grad_K[j * kv_dim + kv_off + d],
                gs * Q[pos * total_dim + q_off + d] * scale);
            atomicAdd(&grad_V[j * kv_dim + kv_off + d],
                p * grad_out[pos * total_dim + q_off + d]);
        }
    }
}
"#;

// ---- bf16 variants ----
// bf16 inputs/outputs use unsigned short*, gradient accumulation buffers stay float*.

/// bf16 transpose: just rearranges u16 values, no conversion needed.
pub const TRANSPOSE_2D_BF16_CUDA: &str = r#"
extern "C" __global__ void transpose_2d_bf16(
    const unsigned short* __restrict__ input,
    unsigned short* __restrict__ output,
    const unsigned int rows,
    const unsigned int cols)
{
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= rows || c >= cols) return;
    output[c * rows + r] = input[r * cols + c];
}
"#;

/// bf16 softmax backward. sm/grad_out/grad_in are bf16.
pub const SOFTMAX_BACKWARD_BF16_CUDA: &str = r#"
__device__ float bf16_to_float(unsigned short bits) {
    return __int_as_float(((unsigned int)bits) << 16);
}
__device__ unsigned short float_to_bf16(float val) {
    unsigned int bits = __float_as_int(val);
    unsigned int lsb = (bits >> 16) & 1;
    bits += 0x7FFF + lsb;
    return (unsigned short)(bits >> 16);
}

extern "C" __global__ void softmax_backward_bf16(
    const unsigned short* __restrict__ sm,
    const unsigned short* __restrict__ grad_out,
    unsigned short* __restrict__ grad_in,
    const unsigned int n_rows,
    const unsigned int row_len)
{
    __shared__ float shared[256];

    unsigned int row = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int tg_size = blockDim.x;
    if (row >= n_rows) return;

    unsigned int base = row * row_len;

    float local_dot = 0.0f;
    for (unsigned int i = tid; i < row_len; i += tg_size) {
        local_dot += bf16_to_float(sm[base + i]) * bf16_to_float(grad_out[base + i]);
    }
    shared[tid] = local_dot;
    __syncthreads();

    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    float row_dot = shared[0];

    for (unsigned int i = tid; i < row_len; i += tg_size) {
        float s = bf16_to_float(sm[base + i]);
        grad_in[base + i] = float_to_bf16(s * (bf16_to_float(grad_out[base + i]) - row_dot));
    }
}
"#;

/// bf16 RMS norm backward. input/weight/grad_out/grad_input are bf16, grad_weight stays f32 (atomic).
pub const RMS_NORM_BACKWARD_BF16_CUDA: &str = r#"
__device__ float bf16_to_float(unsigned short bits) {
    return __int_as_float(((unsigned int)bits) << 16);
}
__device__ unsigned short float_to_bf16(float val) {
    unsigned int bits = __float_as_int(val);
    unsigned int lsb = (bits >> 16) & 1;
    bits += 0x7FFF + lsb;
    return (unsigned short)(bits >> 16);
}

extern "C" __global__ void rms_norm_backward_bf16(
    const unsigned short* __restrict__ input,
    const unsigned short* __restrict__ weight,
    const unsigned short* __restrict__ grad_out,
    unsigned short* __restrict__ grad_input,
    float* __restrict__ grad_weight,
    const unsigned int n_groups,
    const unsigned int dim,
    const float eps)
{
    __shared__ float shared[256];

    unsigned int group = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int tg_size = blockDim.x;
    if (group >= n_groups) return;

    unsigned int base = group * dim;

    // Phase 1: sum of squares
    float local_sq = 0.0f;
    for (unsigned int i = tid; i < dim; i += tg_size) {
        float v = bf16_to_float(input[base + i]);
        local_sq += v * v;
    }
    shared[tid] = local_sq;
    __syncthreads();

    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    float rms_sq = shared[0] / float(dim) + eps;
    float inv_rms = rsqrtf(rms_sq);
    __syncthreads();  // barrier before Phase 2 reuses shared[]

    // Phase 2: sum(x * w * grad_out)
    float local_xwg = 0.0f;
    for (unsigned int i = tid; i < dim; i += tg_size) {
        local_xwg += bf16_to_float(input[base + i]) * bf16_to_float(weight[i]) * bf16_to_float(grad_out[base + i]);
    }
    shared[tid] = local_xwg;
    __syncthreads();

    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    float sum_xwg = shared[0];

    // Phase 3: compute grad_input (bf16) and accumulate grad_weight (f32 atomic)
    for (unsigned int i = tid; i < dim; i += tg_size) {
        float x = bf16_to_float(input[base + i]);
        float go = bf16_to_float(grad_out[base + i]);
        float w = bf16_to_float(weight[i]);

        float gi = w * inv_rms * go - x * inv_rms * inv_rms * inv_rms / float(dim) * sum_xwg;
        grad_input[base + i] = float_to_bf16(gi);

        atomicAdd(&grad_weight[i], x * inv_rms * go);
    }
}
"#;

/// bf16 embedding backward. grad_out is bf16, ids is u32 (as f32 bits), grad_weight stays f32 (atomic).
pub const EMBEDDING_BACKWARD_BF16_CUDA: &str = r#"
__device__ float bf16_to_float(unsigned short bits) {
    return __int_as_float(((unsigned int)bits) << 16);
}

extern "C" __global__ void embedding_backward_bf16(
    const unsigned short* __restrict__ grad_out,
    const unsigned int* __restrict__ ids,
    float* __restrict__ grad_weight,
    const unsigned int vocab_size,
    const unsigned int seq_len,
    const unsigned int dim)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= seq_len) return;

    unsigned int id = ids[gid];
    for (unsigned int d = 0; d < dim; d++) {
        atomicAdd(&grad_weight[id * dim + d], bf16_to_float(grad_out[gid * dim + d]));
    }
}
"#;

/// bf16 fused cross-entropy. logits is bf16, targets is u32, grad is bf16, loss_out stays f32.
pub const CROSS_ENTROPY_BF16_CUDA: &str = r#"
__device__ float bf16_to_float(unsigned short bits) {
    return __int_as_float(((unsigned int)bits) << 16);
}
__device__ unsigned short float_to_bf16(float val) {
    unsigned int bits = __float_as_int(val);
    unsigned int lsb = (bits >> 16) & 1;
    bits += 0x7FFF + lsb;
    return (unsigned short)(bits >> 16);
}

extern "C" __global__ void cross_entropy_fwd_bwd_bf16(
    const unsigned short* __restrict__ logits,
    const unsigned int* __restrict__ targets,
    unsigned short* __restrict__ grad,
    float* __restrict__ loss_out,
    const unsigned int n_pos,
    const unsigned int vocab,
    const unsigned int pad_id,
    const unsigned int count)
{
    __shared__ float shared[256];

    unsigned int row = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int tg_size = blockDim.x;
    if (row >= n_pos) return;
    if (count == 0) return;

    unsigned int target = targets[row];
    unsigned int base = row * vocab;

    if (target == pad_id) {
        for (unsigned int i = tid; i < vocab; i += tg_size) {
            grad[base + i] = float_to_bf16(0.0f);
        }
        return;
    }

    // Phase 1: find max
    float local_max = -1e38f;
    for (unsigned int i = tid; i < vocab; i += tg_size) {
        local_max = fmaxf(local_max, bf16_to_float(logits[base + i]));
    }
    shared[tid] = local_max;
    __syncthreads();
    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        __syncthreads();
    }
    float row_max = shared[0];
    __syncthreads();  // barrier before Phase 2 reuses shared[]

    // Phase 2: exp sum
    float local_sum = 0.0f;
    for (unsigned int i = tid; i < vocab; i += tg_size) {
        local_sum += expf(bf16_to_float(logits[base + i]) - row_max);
    }
    shared[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    float row_sum = shared[0];

    if (tid == 0) {
        float log_prob = (bf16_to_float(logits[base + target]) - row_max) - logf(row_sum);
        atomicAdd(&loss_out[0], -log_prob / float(count));
    }

    float inv_count = 1.0f / float(count);
    float inv_sum = 1.0f / row_sum;
    for (unsigned int i = tid; i < vocab; i += tg_size) {
        float sm = expf(bf16_to_float(logits[base + i]) - row_max) * inv_sum;
        float one_hot = (i == target) ? 1.0f : 0.0f;
        grad[base + i] = float_to_bf16((sm - one_hot) * inv_count);
    }
}
"#;

// ---------------------------------------------------------------------------
// FlashAttention-2 backward kernels (tiled, no O(N) shared memory)
// ---------------------------------------------------------------------------

/// Precompute D[i,h] = sum_d(dO[i,h,d] * O[i,h,d]) for the flash backward.
/// Grid: (seq_len, n_heads), Block: (min(head_dim, 256))
pub const FLASH_ATTN_BWD_PRECOMPUTE_D_CUDA: &str = r#"
extern "C" __global__ void flash_attn_bwd_precompute_d(
    const float* __restrict__ dO,
    const float* __restrict__ O,
    float* __restrict__ D,
    const unsigned int seq_len,
    const unsigned int n_heads,
    const unsigned int head_dim,
    const unsigned int batch_size)
{
    extern __shared__ float reduce[];

    const unsigned int pos   = blockIdx.x;
    const unsigned int head  = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    const unsigned int tid   = threadIdx.x;
    const unsigned int tg_size = blockDim.x;

    if (pos >= seq_len || head >= n_heads || batch >= batch_size) return;

    const unsigned int total_dim = n_heads * head_dim;
    const unsigned int base = batch * seq_len * total_dim + pos * total_dim + head * head_dim;

    float local_sum = 0.0f;
    for (unsigned int d = tid; d < head_dim; d += tg_size) {
        local_sum += dO[base + d] * O[base + d];
    }
    reduce[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] += reduce[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        D[batch * seq_len * n_heads + pos * n_heads + head] = reduce[0];
    }
}
"#;

/// FlashAttention-2 style causal attention backward with GQA support (v3).
///
/// Grid: (n_kv_tiles, n_heads, Q_SPLIT)
///   - blockIdx.x = KV tile index
///   - blockIdx.y = Q head index (NOT KV head — eliminates heads_per_kv serial loop)
///   - blockIdx.z = Q-split index (each handles seq_len/Q_SPLIT Q positions)
/// Block: (128, 1, 1)
///
/// v3: Q[i] and dO[i] cached in shared memory per Q iteration.
/// K/V read from global (L2-cached naturally for 32×64 tiles).
/// Shared memory kept small for high occupancy.
///
/// Dynamic shared memory layout:
///   float dK_acc[TILE_KV * head_dim]  — local grad_K accumulator
///   float dV_acc[TILE_KV * head_dim]  — local grad_V accumulator
///   float scores[TILE_KV]             — softmax probs for one Q row
///   float reduce[blockDim.x]          — reduction scratch
///   float Q_cache[head_dim]           — cached Q[i] for current Q position
///   float dO_cache[head_dim]          — cached dO[i] for current Q position
pub const CAUSAL_ATTENTION_BACKWARD_FLASH_CUDA: &str = r#"
#define FA_BWD_TILE 32

extern "C" __global__ void causal_attention_backward_flash(
    const float* __restrict__ dO,
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ D,
    float* __restrict__ grad_Q,
    float* __restrict__ grad_K,
    float* __restrict__ grad_V,
    const unsigned int seq_len,
    const unsigned int n_heads,
    const unsigned int n_kv_heads,
    const unsigned int head_dim,
    const unsigned int q_split,
    const unsigned int batch_size,
    const unsigned int n_kv_tiles)
{
    extern __shared__ float smem[];

    // Decode batch from blockIdx.x: (tile_idx, batch) packed together
    const unsigned int combined = blockIdx.x;
    const unsigned int tile_idx = combined % n_kv_tiles;
    const unsigned int batch    = combined / n_kv_tiles;
    const unsigned int head     = blockIdx.y;
    const unsigned int q_chunk  = blockIdx.z;
    const unsigned int tid      = threadIdx.x;
    const unsigned int tg_size  = blockDim.x;

    if (batch >= batch_size) return;

    const unsigned int tile_start = tile_idx * FA_BWD_TILE;
    const unsigned int tile_end   = min(tile_start + (unsigned int)FA_BWD_TILE, seq_len);
    const unsigned int tile_len   = tile_end - tile_start;

    const unsigned int total_dim    = n_heads * head_dim;
    const unsigned int kv_dim       = n_kv_heads * head_dim;
    const unsigned int heads_per_kv = n_heads / n_kv_heads;
    const unsigned int kv_head      = head / heads_per_kv;
    const unsigned int q_off        = batch * seq_len * total_dim + head * head_dim;
    const unsigned int kv_off       = batch * seq_len * kv_dim + kv_head * head_dim;
    const unsigned int d_off        = batch * seq_len * n_heads;
    const float scale               = rsqrtf((float)head_dim);

    // Shared memory layout (small footprint for high occupancy)
    float* dK_acc   = smem;                                              // [TILE, head_dim]
    float* dV_acc   = dK_acc  + FA_BWD_TILE * head_dim;                  // [TILE, head_dim]
    float* scores   = dV_acc  + FA_BWD_TILE * head_dim;                  // [FA_BWD_TILE]
    float* reduce   = scores  + FA_BWD_TILE;                             // [tg_size]
    float* Q_cache  = reduce  + tg_size;                                 // [head_dim]
    float* dO_cache = Q_cache + head_dim;                                // [head_dim]

    // Initialize dK_acc and dV_acc to zero
    for (unsigned int idx = tid; idx < tile_len * head_dim; idx += tg_size) {
        dK_acc[idx] = 0.0f;
        dV_acc[idx] = 0.0f;
    }
    __syncthreads();

    // Compute Q range for this q_chunk
    // Causal: only Q positions i >= tile_start can attend to this KV tile
    const unsigned int q_total = (seq_len > tile_start) ? (seq_len - tile_start) : 0;
    const unsigned int chunk_size = (q_total + q_split - 1) / q_split;
    const unsigned int q_start = tile_start + q_chunk * chunk_size;
    const unsigned int q_end   = min(q_start + chunk_size, seq_len);

    // Inner loop over Q positions in this chunk
    for (unsigned int i = q_start; i < q_end; i++) {
        // How many KV positions in this tile does position i attend to?
        unsigned int attend_in_tile = min(i + 1 - tile_start, tile_len);

        // Cache Q[i] and dO[i] in shared memory (avoids redundant global reads across phases)
        for (unsigned int d = tid; d < head_dim; d += tg_size) {
            Q_cache[d]  = Q[i * total_dim + q_off + d];
            dO_cache[d] = dO[i * total_dim + q_off + d];
        }
        __syncthreads();

        // ── Phase 1: Compute scores S[j] = Q[i] . K[tile_start+j] * scale ──
        float local_max = -1e38f;
        for (unsigned int j = tid; j < attend_in_tile; j += tg_size) {
            float dot = 0.0f;
            const unsigned int k_base = (tile_start + j) * kv_dim + kv_off;
            for (unsigned int d = 0; d < head_dim; d++) {
                dot += Q_cache[d] * K[k_base + d];
            }
            float s = dot * scale;
            scores[j] = s;
            local_max = fmaxf(local_max, s);
        }
        // Parallel max-reduce
        reduce[tid] = local_max;
        __syncthreads();
        for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
            if (tid < s) reduce[tid] = fmaxf(reduce[tid], reduce[tid + s]);
            __syncthreads();
        }
        float row_max = reduce[0];
        __syncthreads();

        // Exp and sum
        float local_sum = 0.0f;
        for (unsigned int j = tid; j < attend_in_tile; j += tg_size) {
            float p = expf(scores[j] - row_max);
            scores[j] = p;
            local_sum += p;
        }
        reduce[tid] = local_sum;
        __syncthreads();
        for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
            if (tid < s) reduce[tid] += reduce[tid + s];
            __syncthreads();
        }
        float inv_sum = (reduce[0] > 0.0f) ? (1.0f / reduce[0]) : 0.0f;

        // Normalize: P[j] = exp(S[j] - max) / sum
        for (unsigned int j = tid; j < attend_in_tile; j += tg_size) {
            scores[j] *= inv_sum;
        }
        __syncthreads();

        // ── Phase 2a: Accumulate dV ──
        // dV[tile_j, d] += P[j] * dO[i, head, d]
        for (unsigned int d = tid; d < head_dim; d += tg_size) {
            float do_val = dO_cache[d];
            for (unsigned int j = 0; j < attend_in_tile; j++) {
                dV_acc[j * head_dim + d] += scores[j] * do_val;
            }
        }
        __syncthreads();

        // ── Phase 2b: Compute dS[j] = P[j] * (dP[j] - D[i,head]) ──
        float d_val = D[d_off + i * n_heads + head];
        for (unsigned int j = tid; j < attend_in_tile; j += tg_size) {
            float dp = 0.0f;
            const unsigned int v_base = (tile_start + j) * kv_dim + kv_off;
            for (unsigned int d = 0; d < head_dim; d++) {
                dp += dO_cache[d] * V[v_base + d];
            }
            scores[j] = scores[j] * (dp - d_val);  // scores now holds dS[j]
        }
        __syncthreads();

        // ── Phase 2c: Accumulate dQ and dK ──
        for (unsigned int d = tid; d < head_dim; d += tg_size) {
            float dq_acc = 0.0f;
            float q_val = Q_cache[d];
            for (unsigned int j = 0; j < attend_in_tile; j++) {
                float ds = scores[j];
                dq_acc += ds * K[(tile_start + j) * kv_dim + kv_off + d];
                dK_acc[j * head_dim + d] += ds * q_val;
            }
            atomicAdd(&grad_Q[i * total_dim + q_off + d], dq_acc * scale);
        }
        __syncthreads();
    }

    // ── Write dK_acc and dV_acc to global via atomicAdd ──
    // (needed because multiple Q_SPLIT blocks contribute to the same KV tile×head)
    for (unsigned int idx = tid; idx < tile_len * head_dim; idx += tg_size) {
        unsigned int j = idx / head_dim;
        unsigned int d = idx % head_dim;
        atomicAdd(&grad_K[(tile_start + j) * kv_dim + kv_off + d], dK_acc[j * head_dim + d] * scale);
        atomicAdd(&grad_V[(tile_start + j) * kv_dim + kv_off + d], dV_acc[j * head_dim + d]);
    }
}
"#;

/// bf16 D-precompute for flash attention backward using tensor cores.
/// Reads bf16 dO and O, accumulates in f32, writes f32 D.
/// Grid: (seq_len, n_heads, batch_size), Block: (min(head_dim, 256))
pub const FLASH_ATTN_BWD_PRECOMPUTE_D_BF16_CUDA: &str = r#"
#include <cuda_bf16.h>

extern "C" __global__ void flash_attn_bwd_precompute_d_bf16(
    const __nv_bfloat16* __restrict__ dO,
    const __nv_bfloat16* __restrict__ O,
    float* __restrict__ D,
    const unsigned int seq_len,
    const unsigned int n_heads,
    const unsigned int head_dim,
    const unsigned int batch_size)
{
    extern __shared__ float reduce[];

    const unsigned int pos   = blockIdx.x;
    const unsigned int head  = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    const unsigned int tid   = threadIdx.x;
    const unsigned int tg_size = blockDim.x;

    if (pos >= seq_len || head >= n_heads || batch >= batch_size) return;

    const unsigned int total_dim = n_heads * head_dim;
    const unsigned long long base = (unsigned long long)batch * seq_len * total_dim
        + pos * total_dim + head * head_dim;

    float local_sum = 0.0f;
    for (unsigned int d = tid; d < head_dim; d += tg_size) {
        local_sum += __bfloat162float(dO[base + d]) * __bfloat162float(O[base + d]);
    }
    reduce[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] += reduce[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        D[batch * seq_len * n_heads + pos * n_heads + head] = reduce[0];
    }
}
"#;

/// Tensor-core flash attention backward (bf16 native, wmma m16n16k16).
///
/// Grid: (n_kv_tiles * batch_size, n_heads, q_split)
/// Block: (128) = 4 warps
/// TILE_Q = 16, TILE_KV = 16
///
/// Shared memory layout (~22KB for head_dim=64):
///   K_smem[16×hd] bf16, V_smem[16×hd] bf16,
///   Q_smem[16×hd] bf16, dO_smem[16×hd] bf16,
///   S[16×16] f32, dK_acc[16×hd] f32, dV_acc[16×hd] f32,
///   row_m[16] f32, row_l[16] f32, D_cache[16] f32,
///   P_bf16[16×16] bf16, dS_bf16[16×16] bf16,
///   S_warp[4][256] f32
///
/// All 5 matmuls use wmma m16n16k16 with bf16 A/B → f32 C accumulation.
/// Gradient outputs (grad_Q/K/V) are f32 for precision.
pub const CAUSAL_ATTENTION_BACKWARD_FLASH_TC_CUDA: &str = r#"
#include <mma.h>
using namespace nvcuda;

#define TC_BWD_TILE_Q  16
#define TC_BWD_TILE_KV 16
#define TC_BWD_BLOCK   128
#define TC_BWD_WARPS   4

extern "C" __global__ void causal_attention_backward_flash_tc(
    const __nv_bfloat16* __restrict__ dO,
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const float* __restrict__ D,
    float* __restrict__ grad_Q,
    float* __restrict__ grad_K,
    float* __restrict__ grad_V,
    const unsigned int seq_len,
    const unsigned int n_heads,
    const unsigned int n_kv_heads,
    const unsigned int head_dim,
    const unsigned int q_split,
    const unsigned int batch_size,
    const unsigned int n_kv_tiles)
{
    extern __shared__ char smem_raw[];

    // Decode grid indices
    const unsigned int combined = blockIdx.x;
    const unsigned int tile_idx = combined % n_kv_tiles;
    const unsigned int batch    = combined / n_kv_tiles;
    const unsigned int head     = blockIdx.y;
    const unsigned int q_chunk  = blockIdx.z;
    const unsigned int tid      = threadIdx.x;
    const unsigned int warp_id  = tid / 32;

    if (batch >= batch_size) return;

    const unsigned int hd           = head_dim;
    const unsigned int total_dim    = n_heads * hd;
    const unsigned int kv_dim       = n_kv_heads * hd;
    const unsigned int heads_per_kv = n_heads / n_kv_heads;
    const unsigned int kv_head      = head / heads_per_kv;
    const float scale               = rsqrtf((float)hd);

    const unsigned int tile_start = tile_idx * TC_BWD_TILE_KV;
    const unsigned int tile_end   = min(tile_start + (unsigned int)TC_BWD_TILE_KV, seq_len);
    const unsigned int tile_len   = tile_end - tile_start;

    // Global memory offsets for this batch
    const __nv_bfloat16* Q_g  = Q  + (unsigned long long)batch * seq_len * total_dim;
    const __nv_bfloat16* K_g  = K  + (unsigned long long)batch * seq_len * kv_dim;
    const __nv_bfloat16* V_g  = V  + (unsigned long long)batch * seq_len * kv_dim;
    const __nv_bfloat16* dO_g = dO + (unsigned long long)batch * seq_len * total_dim;
    const float* D_g          = D  + (unsigned long long)batch * seq_len * n_heads;
    float* gQ_g = grad_Q + (unsigned long long)batch * seq_len * total_dim;
    float* gK_g = grad_K + (unsigned long long)batch * seq_len * kv_dim;
    float* gV_g = grad_V + (unsigned long long)batch * seq_len * kv_dim;

    // ---- Shared memory layout ----
    __nv_bfloat16* K_smem  = (__nv_bfloat16*)smem_raw;                                      // [16, hd]
    __nv_bfloat16* V_smem  = K_smem + TC_BWD_TILE_KV * hd;                                  // [16, hd]
    __nv_bfloat16* Q_smem  = V_smem + TC_BWD_TILE_KV * hd;                                  // [16, hd]
    __nv_bfloat16* dO_smem = Q_smem + TC_BWD_TILE_Q * hd;                                   // [16, hd]
    float* S               = (float*)(dO_smem + TC_BWD_TILE_Q * hd);                        // [16, 16]
    float* dK_acc          = S + TC_BWD_TILE_Q * TC_BWD_TILE_KV;                             // [16, hd]
    float* dV_acc          = dK_acc + TC_BWD_TILE_KV * hd;                                   // [16, hd]
    float* row_m           = dV_acc + TC_BWD_TILE_KV * hd;                                   // [16]
    float* row_l           = row_m + TC_BWD_TILE_Q;                                          // [16]
    float* D_cache         = row_l + TC_BWD_TILE_Q;                                          // [16]
    __nv_bfloat16* P_bf16  = (__nv_bfloat16*)(D_cache + TC_BWD_TILE_Q);                     // [16, 16]
    __nv_bfloat16* dS_bf16 = P_bf16 + TC_BWD_TILE_Q * TC_BWD_TILE_KV;                      // [16, 16]
    float* S_warp          = (float*)(dS_bf16 + TC_BWD_TILE_Q * TC_BWD_TILE_KV);            // [4][256]

    const unsigned int n_k_steps = hd / 16;

    // ---- Phase 0: Load K_smem, V_smem, zero dK_acc, dV_acc ----
    for (unsigned int i = tid; i < TC_BWD_TILE_KV * hd; i += TC_BWD_BLOCK) {
        unsigned int row = i / hd;
        unsigned int col = i % hd;
        __nv_bfloat16 kval = (row < tile_len)
            ? K_g[(tile_start + row) * kv_dim + kv_head * hd + col]
            : __float2bfloat16(0.0f);
        __nv_bfloat16 vval = (row < tile_len)
            ? V_g[(tile_start + row) * kv_dim + kv_head * hd + col]
            : __float2bfloat16(0.0f);
        K_smem[i] = kval;
        V_smem[i] = vval;
    }
    for (unsigned int i = tid; i < TC_BWD_TILE_KV * hd; i += TC_BWD_BLOCK) {
        dK_acc[i] = 0.0f;
        dV_acc[i] = 0.0f;
    }
    __syncthreads();

    // Compute Q range for this q_chunk (causal: only Q >= tile_start)
    const unsigned int q_total = (seq_len > tile_start) ? (seq_len - tile_start) : 0;
    const unsigned int chunk_size = (q_total + q_split - 1) / q_split;
    const unsigned int q_start = tile_start + q_chunk * chunk_size;
    const unsigned int q_end   = min(q_start + chunk_size, seq_len);

    // ---- Phase 1: Loop over Q tiles in this chunk ----
    // Process Q positions in tiles of TC_BWD_TILE_Q
    for (unsigned int q_tile_start = q_start; q_tile_start < q_end; q_tile_start += TC_BWD_TILE_Q) {
        unsigned int q_tile_end = min(q_tile_start + (unsigned int)TC_BWD_TILE_Q, q_end);

        // How many KV positions in this tile can the first Q row attend?
        // If q_tile_start < tile_start, skip (fully masked). This shouldn't happen
        // due to q_start >= tile_start from causality.
        if (q_tile_start + TC_BWD_TILE_Q <= tile_start && q_tile_start + TC_BWD_TILE_Q <= tile_start) {
            continue; // Fully masked — no Q in this tile attends to any K in the KV tile
        }

        // Step 1: Load Q_smem, dO_smem, D_cache
        for (unsigned int i = tid; i < TC_BWD_TILE_Q * hd; i += TC_BWD_BLOCK) {
            unsigned int row = i / hd;
            unsigned int col = i % hd;
            unsigned int gq = q_tile_start + row;
            Q_smem[i]  = (gq < seq_len)
                ? Q_g[gq * total_dim + head * hd + col]
                : __float2bfloat16(0.0f);
            dO_smem[i] = (gq < seq_len)
                ? dO_g[gq * total_dim + head * hd + col]
                : __float2bfloat16(0.0f);
        }
        if (tid < TC_BWD_TILE_Q) {
            unsigned int gq = q_tile_start + tid;
            D_cache[tid] = (gq < seq_len) ? D_g[gq * n_heads + head] : 0.0f;
        }
        __syncthreads();

        // Step 2: S = Q × K^T via wmma [16×16]
        // Each warp handles k-steps strided by 4, accumulates partial S in a fragment
        {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
            wmma::fill_fragment(acc_frag, 0.0f);

            for (unsigned int ks = warp_id; ks < n_k_steps; ks += TC_BWD_WARPS) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
                // Q_smem[16, hd] row_major, offset by ks*16 columns
                wmma::load_matrix_sync(a_frag, Q_smem + ks * 16, hd);
                // K_smem[16, hd] row_major, loaded as col_major → K^T
                wmma::load_matrix_sync(b_frag, K_smem + ks * 16, hd);
                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }
            wmma::store_matrix_sync(S_warp + warp_id * 256, acc_frag, 16, wmma::mem_row_major);
        }
        __syncthreads();

        // Reduce across warps into S
        {
            unsigned int active_warps = min((unsigned int)TC_BWD_WARPS, n_k_steps);
            for (unsigned int i = tid; i < TC_BWD_TILE_Q * TC_BWD_TILE_KV; i += TC_BWD_BLOCK) {
                float sum = 0.0f;
                for (unsigned int w = 0; w < active_warps; w++)
                    sum += S_warp[w * 256 + i];
                S[i] = sum;
            }
        }
        __syncthreads();

        // Apply scale + causal mask
        for (unsigned int i = tid; i < TC_BWD_TILE_Q * TC_BWD_TILE_KV; i += TC_BWD_BLOCK) {
            unsigned int q_row = i / TC_BWD_TILE_KV;
            unsigned int k_col = i % TC_BWD_TILE_KV;
            unsigned int gq = q_tile_start + q_row;
            unsigned int gk = tile_start + k_col;
            if (gq >= seq_len || k_col >= tile_len || gk > gq)
                S[i] = -1e38f;
            else
                S[i] *= scale;
        }
        __syncthreads();

        // Step 3: Softmax (8 threads/row, warp shuffles) — S now holds P
        {
            const unsigned int tpr = TC_BWD_BLOCK / TC_BWD_TILE_Q;  // 8 threads per row
            const unsigned int my_row  = tid / tpr;
            const unsigned int my_lane = tid % tpr;

            if (my_row < TC_BWD_TILE_Q) {
                // Row max
                float lmax = -1e38f;
                for (unsigned int j = my_lane; j < TC_BWD_TILE_KV; j += tpr)
                    lmax = fmaxf(lmax, S[my_row * TC_BWD_TILE_KV + j]);
                for (unsigned int off = tpr / 2; off > 0; off >>= 1)
                    lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffff, lmax, off));

                // Store row stats
                if (my_lane == 0) {
                    row_m[my_row] = lmax;
                }
            }
        }
        __syncthreads();

        {
            const unsigned int tpr = TC_BWD_BLOCK / TC_BWD_TILE_Q;
            const unsigned int my_row  = tid / tpr;
            const unsigned int my_lane = tid % tpr;

            if (my_row < TC_BWD_TILE_Q) {
                float lmax = row_m[my_row];
                float lsum = 0.0f;
                for (unsigned int j = my_lane; j < TC_BWD_TILE_KV; j += tpr) {
                    float p = expf(S[my_row * TC_BWD_TILE_KV + j] - lmax);
                    S[my_row * TC_BWD_TILE_KV + j] = p;
                    lsum += p;
                }
                for (unsigned int off = tpr / 2; off > 0; off >>= 1)
                    lsum += __shfl_xor_sync(0xffffffff, lsum, off);

                if (my_lane == 0) {
                    row_l[my_row] = lsum;
                }
            }
        }
        __syncthreads();

        // Normalize P and convert to bf16
        for (unsigned int i = tid; i < TC_BWD_TILE_Q * TC_BWD_TILE_KV; i += TC_BWD_BLOCK) {
            unsigned int row = i / TC_BWD_TILE_KV;
            float inv_l = (row_l[row] > 0.0f) ? (1.0f / row_l[row]) : 0.0f;
            float p = S[i] * inv_l;
            S[i] = p;
            P_bf16[i] = __float2bfloat16(p);
        }
        __syncthreads();

        // Step 4: dV_acc += P^T × dO via wmma
        // P^T[16,16] col_major × dO[16,hd] row_major → [16,hd] f32
        {
            const unsigned int n_d_steps = hd / 16;
            for (unsigned int ds = warp_id; ds < n_d_steps; ds += TC_BWD_WARPS) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> pt_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> do_frag;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> dv_frag;

                wmma::load_matrix_sync(dv_frag, dV_acc + ds * 16, hd, wmma::mem_row_major);
                wmma::load_matrix_sync(pt_frag, P_bf16, TC_BWD_TILE_KV);
                wmma::load_matrix_sync(do_frag, dO_smem + ds * 16, hd);
                wmma::mma_sync(dv_frag, pt_frag, do_frag, dv_frag);
                wmma::store_matrix_sync(dV_acc + ds * 16, dv_frag, hd, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // Step 5: dP = dO × V^T via wmma [16×16], then fuse dS = P * (dP - D)
        // Same pattern as Q×K^T: 4 warps split k-steps, reduce via S_warp → S
        {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
            wmma::fill_fragment(acc_frag, 0.0f);

            for (unsigned int ks = warp_id; ks < n_k_steps; ks += TC_BWD_WARPS) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> do_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> v_frag;
                wmma::load_matrix_sync(do_frag, dO_smem + ks * 16, hd);
                wmma::load_matrix_sync(v_frag, V_smem + ks * 16, hd);
                wmma::mma_sync(acc_frag, do_frag, v_frag, acc_frag);
            }
            wmma::store_matrix_sync(S_warp + warp_id * 256, acc_frag, 16, wmma::mem_row_major);
        }
        __syncthreads();

        // Reduce dP across warps, fuse dS = P * (dP - D), convert to bf16
        {
            unsigned int active_warps = min((unsigned int)TC_BWD_WARPS, n_k_steps);
            for (unsigned int i = tid; i < TC_BWD_TILE_Q * TC_BWD_TILE_KV; i += TC_BWD_BLOCK) {
                unsigned int row = i / TC_BWD_TILE_KV;
                float dp = 0.0f;
                for (unsigned int w = 0; w < active_warps; w++)
                    dp += S_warp[w * 256 + i];
                float ds = S[i] * (dp - D_cache[row]);  // S still holds P from softmax
                S[i] = ds;                               // Reuse S for dS
                dS_bf16[i] = __float2bfloat16(ds);
            }
        }
        __syncthreads();

        // Step 6: dQ += dS × K via wmma, atomicAdd to global grad_Q (f32)
        // dS[16,16] row × K_block[16,16] row → [16,16] f32, per d-step
        // Each warp handles d-steps strided by WARPS. After each wmma, the warp
        // stores to its S_warp slot and immediately atomicAdds to global.
        {
            const unsigned int n_d_steps = hd / 16;
            const unsigned int lane = tid % 32;
            for (unsigned int ds = warp_id; ds < n_d_steps; ds += TC_BWD_WARPS) {
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> dq_frag;
                wmma::fill_fragment(dq_frag, 0.0f);

                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> ds_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> k_frag;
                wmma::load_matrix_sync(ds_frag, dS_bf16, TC_BWD_TILE_KV);
                wmma::load_matrix_sync(k_frag, K_smem + ds * 16, hd);
                wmma::mma_sync(dq_frag, ds_frag, k_frag, dq_frag);

                // store_matrix_sync guarantees warp-level visibility
                wmma::store_matrix_sync(S_warp + warp_id * 256, dq_frag, 16, wmma::mem_row_major);

                // Each lane in the warp writes its portion to global
                for (unsigned int i = lane; i < 256; i += 32) {
                    unsigned int row = i / 16;
                    unsigned int col = i % 16;
                    unsigned int gq = q_tile_start + row;
                    if (gq < seq_len) {
                        atomicAdd(&gQ_g[gq * total_dim + head * hd + ds * 16 + col],
                            S_warp[warp_id * 256 + i] * scale);
                    }
                }
            }
        }
        __syncthreads();

        // Step 7: dK_acc += dS^T × Q via wmma
        // dS^T[16,16] col × Q[16,hd] row → [16,hd] f32 accumulated into dK_acc
        {
            const unsigned int n_d_steps = hd / 16;
            for (unsigned int ds = warp_id; ds < n_d_steps; ds += TC_BWD_WARPS) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> dst_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> q_frag;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> dk_frag;

                wmma::load_matrix_sync(dk_frag, dK_acc + ds * 16, hd, wmma::mem_row_major);
                wmma::load_matrix_sync(dst_frag, dS_bf16, TC_BWD_TILE_KV);
                wmma::load_matrix_sync(q_frag, Q_smem + ds * 16, hd);
                wmma::mma_sync(dk_frag, dst_frag, q_frag, dk_frag);
                wmma::store_matrix_sync(dK_acc + ds * 16, dk_frag, hd, wmma::mem_row_major);
            }
        }
        __syncthreads();
    }

    // ---- Phase 2: atomicAdd dK_acc × scale and dV_acc to global grad_K, grad_V ----
    for (unsigned int i = tid; i < tile_len * hd; i += TC_BWD_BLOCK) {
        unsigned int j = i / hd;
        unsigned int d = i % hd;
        atomicAdd(&gK_g[(tile_start + j) * kv_dim + kv_head * hd + d], dK_acc[j * hd + d] * scale);
        atomicAdd(&gV_g[(tile_start + j) * kv_dim + kv_head * hd + d], dV_acc[j * hd + d]);
    }
}
"#;

/// bf16 causal attention backward. grad_out/Q/K/V are bf16, grad_Q is bf16,
/// grad_K/grad_V stay f32 (atomic accumulation for precision).
pub const CAUSAL_ATTENTION_BACKWARD_BF16_CUDA: &str = r#"
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

extern "C" __global__ void causal_attention_backward_bf16(
    const unsigned short* __restrict__ grad_out,
    const unsigned short* __restrict__ Q,
    const unsigned short* __restrict__ K,
    const unsigned short* __restrict__ V,
    unsigned short* __restrict__ grad_Q,
    float* __restrict__ grad_K,
    float* __restrict__ grad_V,
    const unsigned int seq_len,
    const unsigned int n_heads,
    const unsigned int n_kv_heads,
    const unsigned int head_dim)
{
    __shared__ float probs[MAX_SEQ];
    __shared__ float grad_sp[MAX_SEQ];
    __shared__ float partials[256];

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

    // Phase 1: attention scores
    for (unsigned int j = 0; j < attend_len; j++) {
        float local_dot = 0.0f;
        for (unsigned int d = tid; d < head_dim; d += tg_size) {
            local_dot += bf16_to_float(Q[pos * total_dim + q_off + d]) * bf16_to_float(K[j * kv_dim + kv_off + d]);
        }
        partials[tid] = local_dot;
        __syncthreads();
        for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
            if (tid < s) partials[tid] += partials[tid + s];
            __syncthreads();
        }
        if (tid == 0) probs[j] = partials[0] * scale;
        __syncthreads();
    }

    // Phase 2: softmax
    float local_max = -1e38f;
    for (unsigned int j = tid; j < attend_len; j += tg_size) {
        local_max = fmaxf(local_max, probs[j]);
    }
    partials[tid] = local_max;
    __syncthreads();
    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) partials[tid] = fmaxf(partials[tid], partials[tid + s]);
        __syncthreads();
    }
    float row_max = partials[0];
    __syncthreads();

    for (unsigned int j = tid; j < attend_len; j += tg_size) {
        probs[j] = expf(probs[j] - row_max);
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (unsigned int j = tid; j < attend_len; j += tg_size) {
        local_sum += probs[j];
    }
    partials[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) partials[tid] += partials[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / partials[0];
    __syncthreads();

    for (unsigned int j = tid; j < attend_len; j += tg_size) {
        probs[j] *= inv_sum;
    }
    __syncthreads();

    // Phase 3: grad_P[j] = dot(grad_out[pos], V[j])
    for (unsigned int j = 0; j < attend_len; j++) {
        float local_dot = 0.0f;
        for (unsigned int d = tid; d < head_dim; d += tg_size) {
            local_dot += bf16_to_float(grad_out[pos * total_dim + q_off + d]) * bf16_to_float(V[j * kv_dim + kv_off + d]);
        }
        partials[tid] = local_dot;
        __syncthreads();
        for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
            if (tid < s) partials[tid] += partials[tid + s];
            __syncthreads();
        }
        if (tid == 0) grad_sp[j] = partials[0];
        __syncthreads();
    }

    // Phase 4: softmax backward
    float local_dpq = 0.0f;
    for (unsigned int j = tid; j < attend_len; j += tg_size) {
        local_dpq += probs[j] * grad_sp[j];
    }
    partials[tid] = local_dpq;
    __syncthreads();
    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) partials[tid] += partials[tid + s];
        __syncthreads();
    }
    float dot_pq = partials[0];
    __syncthreads();

    for (unsigned int j = tid; j < attend_len; j += tg_size) {
        grad_sp[j] = probs[j] * (grad_sp[j] - dot_pq);
    }
    __syncthreads();

    // Phase 5: accumulate gradients
    // grad_Q: bf16 output (direct write, unique per threadgroup)
    for (unsigned int d = tid; d < head_dim; d += tg_size) {
        float acc = 0.0f;
        for (unsigned int j = 0; j < attend_len; j++) {
            acc += grad_sp[j] * bf16_to_float(K[j * kv_dim + kv_off + d]);
        }
        grad_Q[pos * total_dim + q_off + d] = float_to_bf16(acc * scale);
    }

    // grad_K/grad_V: f32 output (atomic accumulation)
    for (unsigned int j = 0; j < attend_len; j++) {
        float gs = grad_sp[j];
        float p = probs[j];
        for (unsigned int d = tid; d < head_dim; d += tg_size) {
            atomicAdd(&grad_K[j * kv_dim + kv_off + d],
                gs * bf16_to_float(Q[pos * total_dim + q_off + d]) * scale);
            atomicAdd(&grad_V[j * kv_dim + kv_off + d],
                p * bf16_to_float(grad_out[pos * total_dim + q_off + d]));
        }
    }
}
"#;
