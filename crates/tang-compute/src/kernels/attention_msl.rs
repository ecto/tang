//! MSL attention kernels using online softmax (flash attention algorithm).
//!
//! Streams through K/V positions using online softmax rescaling.
//! No shared-memory score buffer — O(head_dim) threadgroup memory only.
//! No sequence length cap (works for any seq_len).

/// Flash causal self-attention kernel with GQA support.
/// Q: [seq_len, n_heads * head_dim], K,V: [seq_len, n_kv_heads * head_dim]
/// Output: [seq_len, n_heads * head_dim]
/// params: [seq_len, n_heads, n_kv_heads, head_dim]
///
/// Each threadgroup handles one (position, head) pair.
/// Threads parallelize over head_dim for dot-product reduction and V accumulation.
/// K/V positions are streamed sequentially with online softmax.
pub const CAUSAL_ATTENTION_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void causal_attention(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* output [[buffer(3)]],
    device const uint* params [[buffer(4)]],
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

    // Threadgroup memory for online softmax
    threadgroup float out_accum[256];  // head_dim <= 256
    threadgroup float partials[256];   // for dot-product reduction
    threadgroup float tg_max[1];
    threadgroup float tg_sum[1];

    for (uint d = tid; d < head_dim; d += tg_size) {
        out_accum[d] = 0.0f;
    }
    if (tid == 0) {
        tg_max[0] = -INFINITY;
        tg_sum[0] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stream through K/V positions
    for (uint j = 0; j < attend_len; j++) {
        // Parallel dot product: Q[pos] . K[j]
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
        float score = partials[0] * scale;

        // Online softmax update
        float prev_max = tg_max[0];
        float new_max = max(prev_max, score);
        float exp_score = exp(score - new_max);
        float rescale = exp(prev_max - new_max);

        if (tid == 0) {
            tg_max[0] = new_max;
            tg_sum[0] = tg_sum[0] * rescale + exp_score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Rescale accumulator and add new V contribution
        for (uint d = tid; d < head_dim; d += tg_size) {
            out_accum[d] = out_accum[d] * rescale + exp_score * V[j * kv_dim + kv_off + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize by sum
    float inv_sum = 1.0f / tg_sum[0];
    for (uint d = tid; d < head_dim; d += tg_size) {
        output[pos * total_dim + q_off + d] = out_accum[d] * inv_sum;
    }
}
"#;

/// Flash KV-cached attention for autoregressive decoding (single query).
/// q: [1, n_heads * head_dim]
/// k_cache, v_cache: [total_len, n_kv_heads * head_dim]
/// output: [1, n_heads * head_dim]
/// params: [total_len, n_heads, n_kv_heads, head_dim]
///
/// Each threadgroup handles one head. Streams through all cache positions
/// using online softmax — no sequence length cap.
pub const KV_ATTENTION_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kv_attention(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* output [[buffer(3)]],
    device const uint* params [[buffer(4)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint cache_len = params[0];
    uint n_heads = params[1];
    uint n_kv_heads = params[2];
    uint head_dim = params[3];
    uint kv_dim = n_kv_heads * head_dim;
    uint heads_per_kv = n_heads / n_kv_heads;

    uint head = tg_id;
    if (head >= n_heads) return;

    uint kv_head = head / heads_per_kv;
    uint q_off = head * head_dim;
    uint kv_off = kv_head * head_dim;
    float scale = rsqrt(float(head_dim));

    threadgroup float out_accum[256];
    threadgroup float partials[256];
    threadgroup float tg_max[1];
    threadgroup float tg_sum[1];

    for (uint d = tid; d < head_dim; d += tg_size) {
        out_accum[d] = 0.0f;
    }
    if (tid == 0) {
        tg_max[0] = -INFINITY;
        tg_sum[0] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = 0; j < cache_len; j++) {
        float local_dot = 0.0f;
        for (uint d = tid; d < head_dim; d += tg_size) {
            local_dot += Q[q_off + d] * K[j * kv_dim + kv_off + d];
        }
        partials[tid] = local_dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tg_size / 2; s > 0; s >>= 1) {
            if (tid < s) partials[tid] += partials[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float score = partials[0] * scale;

        float prev_max = tg_max[0];
        float new_max = max(prev_max, score);
        float exp_score = exp(score - new_max);
        float rescale = exp(prev_max - new_max);

        if (tid == 0) {
            tg_max[0] = new_max;
            tg_sum[0] = tg_sum[0] * rescale + exp_score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint d = tid; d < head_dim; d += tg_size) {
            out_accum[d] = out_accum[d] * rescale + exp_score * V[j * kv_dim + kv_off + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_sum = 1.0f / tg_sum[0];
    for (uint d = tid; d < head_dim; d += tg_size) {
        output[q_off + d] = out_accum[d] * inv_sum;
    }
}
"#;

/// Flash KV-cached attention for batched prefill (multi-query).
/// q: [q_len, n_heads * head_dim]
/// k_cache, v_cache: [cache_start + q_len, n_kv_heads * head_dim]
/// output: [q_len, n_heads * head_dim]
/// params: [cache_start, q_len, n_heads, n_kv_heads, head_dim]
///
/// Grid: (q_len, n_heads) threadgroups. Each handles one (qi, head) pair.
/// Causal mask: query qi attends to positions 0..cache_start+qi+1.
/// Uses online softmax — no sequence length cap.
pub const KV_ATTENTION_PREFILL_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kv_attention_prefill(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* output [[buffer(3)]],
    device const uint* params [[buffer(4)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint2 tg_size2 [[threads_per_threadgroup]])
{
    uint tid = tid2.x;
    uint tg_size = tg_size2.x;
    uint cache_start = params[0];
    uint q_len = params[1];
    uint n_heads = params[2];
    uint n_kv_heads = params[3];
    uint head_dim = params[4];
    uint total_dim = n_heads * head_dim;
    uint kv_dim = n_kv_heads * head_dim;
    uint heads_per_kv = n_heads / n_kv_heads;

    uint qi = tg_id.x;
    uint head = tg_id.y;
    if (qi >= q_len || head >= n_heads) return;

    uint kv_head = head / heads_per_kv;
    uint q_off = qi * total_dim + head * head_dim;
    uint kv_off = kv_head * head_dim;
    float scale = rsqrt(float(head_dim));
    uint attend_len = cache_start + qi + 1;

    threadgroup float out_accum[256];
    threadgroup float partials[256];
    threadgroup float tg_max[1];
    threadgroup float tg_sum[1];

    for (uint d = tid; d < head_dim; d += tg_size) {
        out_accum[d] = 0.0f;
    }
    if (tid == 0) {
        tg_max[0] = -INFINITY;
        tg_sum[0] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = 0; j < attend_len; j++) {
        float local_dot = 0.0f;
        for (uint d = tid; d < head_dim; d += tg_size) {
            local_dot += Q[q_off + d] * K[j * kv_dim + kv_off + d];
        }
        partials[tid] = local_dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = tg_size / 2; s > 0; s >>= 1) {
            if (tid < s) partials[tid] += partials[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float score = partials[0] * scale;

        float prev_max = tg_max[0];
        float new_max = max(prev_max, score);
        float exp_score = exp(score - new_max);
        float rescale = exp(prev_max - new_max);

        if (tid == 0) {
            tg_max[0] = new_max;
            tg_sum[0] = tg_sum[0] * rescale + exp_score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint d = tid; d < head_dim; d += tg_size) {
            out_accum[d] = out_accum[d] * rescale + exp_score * V[j * kv_dim + kv_off + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_sum = 1.0f / tg_sum[0];
    uint out_off = qi * total_dim + head * head_dim;
    for (uint d = tid; d < head_dim; d += tg_size) {
        output[out_off + d] = out_accum[d] * inv_sum;
    }
}
"#;
