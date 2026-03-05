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
