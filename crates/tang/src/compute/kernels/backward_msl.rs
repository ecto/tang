//! MSL backward kernels for training: transpose, softmax_backward, rms_norm_backward,
//! embedding_backward, cross_entropy_forward_backward.

/// On-device 2D transpose: [rows, cols] → [cols, rows].
/// params: [rows, cols]
pub const TRANSPOSE_2D_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void transpose_2d(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const uint* params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint rows = params[0];
    uint cols = params[1];
    uint r = gid.y;
    uint c = gid.x;
    if (r >= rows || c >= cols) return;
    output[c * rows + r] = input[r * cols + c];
}
"#;

/// Softmax backward: grad_input[i,j] = sm[i,j] * (grad[i,j] - dot(sm[i,:], grad[i,:])).
/// One threadgroup per row.
/// params: [n_rows, row_len]
pub const SOFTMAX_BACKWARD_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint WG = 256;

kernel void softmax_backward(
    device const float* sm [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device float* grad_in [[buffer(2)]],
    device const uint* params [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint n_rows = params[0];
    uint row_len = params[1];
    if (tg_id >= n_rows) return;

    uint base = tg_id * row_len;

    threadgroup float shared[WG];

    // Compute dot(sm, grad) for this row
    float local_dot = 0.0f;
    for (uint i = tid; i < row_len; i += tg_size) {
        local_dot += sm[base + i] * grad_out[base + i];
    }
    shared[tid] = local_dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_dot = shared[0];

    // grad_input = sm * (grad - dot)
    for (uint i = tid; i < row_len; i += tg_size) {
        grad_in[base + i] = sm[base + i] * (grad_out[base + i] - row_dot);
    }
}
"#;

/// RMS norm backward.
/// params: [n_groups, dim, eps_bits]
/// Outputs: grad_input [n_groups * dim], grad_weight [dim] (atomically accumulated).
pub const RMS_NORM_BACKWARD_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint WG = 256;

kernel void rms_norm_backward(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* grad_out [[buffer(2)]],
    device float* grad_input [[buffer(3)]],
    device atomic_float* grad_weight [[buffer(4)]],
    device const uint* params [[buffer(5)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint n_groups = params[0];
    uint dim = params[1];
    float eps = as_type<float>(params[2]);
    if (tg_id >= n_groups) return;

    uint base = tg_id * dim;

    threadgroup float shared[WG];

    // Phase 1: compute sum of squares
    float local_sq = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        float v = input[base + i];
        local_sq += v * v;
    }
    shared[tid] = local_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms_sq = shared[0] / float(dim) + eps;
    float inv_rms = rsqrt(rms_sq);

    // Phase 2: compute sum(x * w * grad_out)
    float local_xwg = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        local_xwg += input[base + i] * weight[i] * grad_out[base + i];
    }
    shared[tid] = local_xwg;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_xwg = shared[0];

    // Phase 3: compute grad_input and accumulate grad_weight
    for (uint i = tid; i < dim; i += tg_size) {
        float x = input[base + i];
        float go = grad_out[base + i];
        float w = weight[i];

        grad_input[base + i] = w * inv_rms * go
            - x * inv_rms * inv_rms * inv_rms / float(dim) * sum_xwg;

        // Atomic accumulate grad_weight across groups
        atomic_fetch_add_explicit(&grad_weight[i], x * inv_rms * go, memory_order_relaxed);
    }
}
"#;

/// Embedding backward: scatter-add grad_output into grad_weight.
/// params: [vocab_size, seq_len, dim]
pub const EMBEDDING_BACKWARD_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void embedding_backward(
    device const float* grad_out [[buffer(0)]],
    device const uint* ids [[buffer(1)]],
    device atomic_float* grad_weight [[buffer(2)]],
    device const uint* params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint seq_len = params[1];
    uint dim = params[2];
    if (gid >= seq_len) return;

    uint id = ids[gid];
    for (uint d = 0; d < dim; d++) {
        atomic_fetch_add_explicit(&grad_weight[id * dim + d],
            grad_out[gid * dim + d], memory_order_relaxed);
    }
}
"#;

/// Fused cross-entropy forward + backward.
/// One threadgroup per position (row).
/// params: [n_positions, vocab_size, pad_id, count_ptr_offset]
/// output buffer layout: [loss (1 float), grad (n_positions * vocab_size)]
pub const CROSS_ENTROPY_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint WG = 256;

kernel void cross_entropy_fwd_bwd(
    device const float* logits [[buffer(0)]],
    device const uint* targets [[buffer(1)]],
    device float* grad [[buffer(2)]],
    device atomic_float* loss_out [[buffer(3)]],
    device const uint* params [[buffer(4)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint n_pos = params[0];
    uint vocab = params[1];
    uint pad_id = params[2];
    uint count = params[3]; // pre-computed non-pad count
    if (tg_id >= n_pos) return;
    if (count == 0) return;

    uint target = targets[tg_id];
    uint base = tg_id * vocab;

    if (target == pad_id) {
        // Zero out gradient for padded positions
        for (uint i = tid; i < vocab; i += tg_size) {
            grad[base + i] = 0.0f;
        }
        return;
    }

    threadgroup float shared[WG];

    // Phase 1: find max
    float local_max = -INFINITY;
    for (uint i = tid; i < vocab; i += tg_size) {
        local_max = max(local_max, logits[base + i]);
    }
    shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = max(shared[tid], shared[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared[0];

    // Phase 2: exp sum
    float local_sum = 0.0f;
    for (uint i = tid; i < vocab; i += tg_size) {
        local_sum += exp(logits[base + i] - row_max);
    }
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_sum = shared[0];

    // Loss contribution from this position (only one thread writes)
    if (tid == 0) {
        float log_prob = (logits[base + target] - row_max) - log(row_sum);
        atomic_fetch_add_explicit(&loss_out[0], -log_prob / float(count), memory_order_relaxed);
    }

    // Gradient: (softmax - one_hot) / count
    float inv_count = 1.0f / float(count);
    float inv_sum = 1.0f / row_sum;
    for (uint i = tid; i < vocab; i += tg_size) {
        float sm = exp(logits[base + i] - row_max) * inv_sum;
        float one_hot = (i == target) ? 1.0f : 0.0f;
        grad[base + i] = (sm - one_hot) * inv_count;
    }
}
"#;

/// Causal attention backward with GQA support.
/// Recomputes softmax probs from Q,K, then computes grad_Q, grad_K, grad_V.
///
/// Grid: (seq_len, n_heads) threadgroups. Threads parallelize over head_dim.
/// grad_K and grad_V use atomic_fetch_add (multiple positions/heads write to same KV row).
/// params: [seq_len, n_heads, n_kv_heads, head_dim]
///
/// Max sequence length: 2048 (limited by threadgroup memory).
pub const CAUSAL_ATTENTION_BACKWARD_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint MAX_SEQ = 2048;

kernel void causal_attention_backward(
    device const float* grad_out [[buffer(0)]],
    device const float* Q [[buffer(1)]],
    device const float* K [[buffer(2)]],
    device const float* V [[buffer(3)]],
    device float* grad_Q [[buffer(4)]],
    device atomic_float* grad_K [[buffer(5)]],
    device atomic_float* grad_V [[buffer(6)]],
    device const uint* params [[buffer(7)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint2 tg_size2 [[threads_per_threadgroup]])
{
    uint tid = tid2.x;
    uint tg_size = tg_size2.x;
    uint seq_len = params[0];
    uint n_heads = params[1];
    uint n_kv_heads = params[2];
    uint head_dim = params[3];
    uint total_dim = n_heads * head_dim;
    uint kv_dim = n_kv_heads * head_dim;
    uint heads_per_kv = n_heads / n_kv_heads;

    uint pos = tg_id.x;
    uint head = tg_id.y;
    if (pos >= seq_len || head >= n_heads) return;

    uint kv_head = head / heads_per_kv;
    uint q_off = head * head_dim;
    uint kv_off = kv_head * head_dim;
    float scale = rsqrt(float(head_dim));
    uint attend_len = pos + 1;

    // Threadgroup memory
    threadgroup float probs[MAX_SEQ];
    threadgroup float grad_s[MAX_SEQ];
    threadgroup float partials[256];

    // ---- Phase 1: Compute attention scores ----
    for (uint j = 0; j < attend_len; j++) {
        float local_dot = 0.0f;
        for (uint d = tid; d < head_dim; d += tg_size) {
            local_dot += Q[pos * total_dim + q_off + d] * K[j * kv_dim + kv_off + d];
        }
        partials[tid] = local_dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tg_size / 2; s > 0; s >>= 1) {
            if (tid < s) partials[tid] += partials[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            probs[j] = partials[0] * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- Phase 2: Softmax ----
    // Find max
    float local_max = -INFINITY;
    for (uint j = tid; j < attend_len; j += tg_size) {
        local_max = max(local_max, probs[j]);
    }
    partials[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) partials[tid] = max(partials[tid], partials[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = partials[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Exp
    for (uint j = tid; j < attend_len; j += tg_size) {
        probs[j] = exp(probs[j] - row_max);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Sum
    float local_sum = 0.0f;
    for (uint j = tid; j < attend_len; j += tg_size) {
        local_sum += probs[j];
    }
    partials[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) partials[tid] += partials[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / partials[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Normalize
    for (uint j = tid; j < attend_len; j += tg_size) {
        probs[j] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Phase 3: grad_P[j] = dot(grad_out[pos], V[j]) ----
    for (uint j = 0; j < attend_len; j++) {
        float local_dot = 0.0f;
        for (uint d = tid; d < head_dim; d += tg_size) {
            local_dot += grad_out[pos * total_dim + q_off + d] * V[j * kv_dim + kv_off + d];
        }
        partials[tid] = local_dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tg_size / 2; s > 0; s >>= 1) {
            if (tid < s) partials[tid] += partials[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            grad_s[j] = partials[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- Phase 4: Softmax backward ----
    // dot_pq = sum(P[j] * grad_P[j])
    float local_dpq = 0.0f;
    for (uint j = tid; j < attend_len; j += tg_size) {
        local_dpq += probs[j] * grad_s[j];
    }
    partials[tid] = local_dpq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) partials[tid] += partials[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float dot_pq = partials[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // grad_S[j] = P[j] * (grad_P[j] - dot_pq)
    for (uint j = tid; j < attend_len; j += tg_size) {
        grad_s[j] = probs[j] * (grad_s[j] - dot_pq);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Phase 5: Accumulate gradients ----
    // grad_Q[pos, d] = scale * sum_j(grad_S[j] * K[j, d])  — direct write, unique per threadgroup
    for (uint d = tid; d < head_dim; d += tg_size) {
        float acc = 0.0f;
        for (uint j = 0; j < attend_len; j++) {
            acc += grad_s[j] * K[j * kv_dim + kv_off + d];
        }
        grad_Q[pos * total_dim + q_off + d] = acc * scale;
    }

    // grad_K[j, d] += scale * grad_S[j] * Q[pos, d]  — atomic (multiple pos/heads write same j)
    // grad_V[j, d] += P[j] * grad_out[pos, d]        — atomic
    for (uint j = 0; j < attend_len; j++) {
        float gs = grad_s[j];
        float p = probs[j];
        for (uint d = tid; d < head_dim; d += tg_size) {
            atomic_fetch_add_explicit(&grad_K[j * kv_dim + kv_off + d],
                gs * Q[pos * total_dim + q_off + d] * scale, memory_order_relaxed);
            atomic_fetch_add_explicit(&grad_V[j * kv_dim + kv_off + d],
                p * grad_out[pos * total_dim + q_off + d], memory_order_relaxed);
        }
    }
}
"#;
