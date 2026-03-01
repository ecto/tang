//! MSL attention kernels: causal self-attention and KV-cached attention.

/// Causal self-attention kernel.
/// Q, K, V: [seq_len, n_heads * head_dim]
/// Output: [seq_len, n_heads * head_dim]
/// params: [seq_len, n_heads, head_dim]
///
/// Each threadgroup handles one (position, head) pair.
pub const CAUSAL_ATTENTION_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint WG = 256;

kernel void causal_attention(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* output [[buffer(3)]],
    device const uint* params [[buffer(4)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint seq_len = params[0];
    uint n_heads = params[1];
    uint head_dim = params[2];
    uint total_dim = n_heads * head_dim;

    uint pos = tg_id.x;        // which position
    uint head = tg_id.y;       // which head
    if (pos >= seq_len || head >= n_heads) return;

    uint h_off = head * head_dim;
    float scale = rsqrt(float(head_dim));

    // Causal: attend to positions [0..pos]
    uint attend_len = pos + 1;

    threadgroup float scores[WG];
    threadgroup float shared[WG];

    // Step 1: compute attention scores (dot products)
    // Each thread handles a subset of attended positions
    float local_max = -INFINITY;
    for (uint j = tid; j < attend_len; j += tg_size) {
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot += Q[pos * total_dim + h_off + d] * K[j * total_dim + h_off + d];
        }
        scores[j % WG] = dot * scale;
        local_max = max(local_max, dot * scale);
    }

    // For longer sequences, we'd need a proper reduction here.
    // This works for attend_len <= WG.
    shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = max(shared[tid], shared[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared[0];

    // Step 2: exp and sum
    float local_sum = 0.0f;
    for (uint j = tid; j < attend_len; j += tg_size) {
        float val = exp(scores[j % WG] - row_max);
        scores[j % WG] = val;
        local_sum += val;
    }
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_sum = shared[0];
    float inv_sum = 1.0f / row_sum;

    // Step 3: weighted sum of V
    for (uint d = tid; d < head_dim; d += tg_size) {
        float val = 0.0f;
        for (uint j = 0; j < attend_len; j++) {
            val += scores[j % WG] * inv_sum * V[j * total_dim + h_off + d];
        }
        output[pos * total_dim + h_off + d] = val;
    }
}
"#;

/// KV-cached attention for autoregressive decoding (single query).
/// q: [1, n_heads * head_dim]
/// k_cache, v_cache: [total_len, n_kv_heads * head_dim]
/// output: [1, n_heads * head_dim]
/// params: [total_len, n_heads, n_kv_heads, head_dim]
///
/// Each threadgroup handles one head. Attends to all total_len positions.
pub const KV_ATTENTION_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint WG = 256;

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

    threadgroup float shared[WG];

    // Compute scores
    float local_max = -INFINITY;
    for (uint j = tid; j < cache_len; j += tg_size) {
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot += Q[q_off + d] * K[j * kv_dim + kv_off + d];
        }
        float score = dot * scale;
        shared[j % WG] = score;
        local_max = max(local_max, score);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Find global max (simplified for cache_len <= WG)
    threadgroup float reduce[WG];
    reduce[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] = max(reduce[tid], reduce[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = reduce[0];

    // Exp + sum
    float local_sum = 0.0f;
    for (uint j = tid; j < cache_len; j += tg_size) {
        float val = exp(shared[j % WG] - row_max);
        shared[j % WG] = val;
        local_sum += val;
    }
    reduce[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] += reduce[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / reduce[0];

    // Weighted sum of V
    for (uint d = tid; d < head_dim; d += tg_size) {
        float val = 0.0f;
        for (uint j = 0; j < cache_len; j++) {
            val += shared[j % WG] * inv_sum * V[j * kv_dim + kv_off + d];
        }
        output[q_off + d] = val;
    }
}
"#;

/// KV-cached attention for batched prefill.
/// q: [q_len, n_heads * head_dim]
/// k_cache, v_cache: [cache_start + q_len, n_kv_heads * head_dim]
/// output: [q_len, n_heads * head_dim]
/// params: [cache_start, q_len, n_heads, n_kv_heads, head_dim]
///
/// Grid: (q_len, n_heads) threadgroups. Each handles one (qi, head) pair.
/// Causal mask: query qi attends to positions 0..cache_start+qi+1.
pub const KV_ATTENTION_PREFILL_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint WG = 256;

kernel void kv_attention_prefill(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* output [[buffer(3)]],
    device const uint* params [[buffer(4)]],
    uint3 tg_id [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 tg_size3 [[threads_per_threadgroup]])
{
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
    uint tid = tid3.x;
    uint tg_size = tg_size3.x;
    if (qi >= q_len || head >= n_heads) return;

    uint kv_head = head / heads_per_kv;
    uint q_off = qi * total_dim + head * head_dim;
    uint kv_off = kv_head * head_dim;
    float scale = rsqrt(float(head_dim));
    uint attend_len = cache_start + qi + 1;

    threadgroup float scores[WG];
    threadgroup float reduce[WG];

    // Compute scores
    float local_max = -INFINITY;
    for (uint j = tid; j < attend_len; j += tg_size) {
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot += Q[q_off + d] * K[j * kv_dim + kv_off + d];
        }
        float s = dot * scale;
        scores[j % WG] = s;
        local_max = max(local_max, s);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Global max
    reduce[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] = max(reduce[tid], reduce[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = reduce[0];

    // Exp + sum
    float local_sum = 0.0f;
    for (uint j = tid; j < attend_len; j += tg_size) {
        float val = exp(scores[j % WG] - row_max);
        scores[j % WG] = val;
        local_sum += val;
    }
    reduce[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] += reduce[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / reduce[0];

    // Weighted sum of V
    uint out_off = qi * total_dim + head * head_dim;
    for (uint d = tid; d < head_dim; d += tg_size) {
        float val = 0.0f;
        for (uint j = 0; j < attend_len; j++) {
            val += scores[j % WG] * inv_sum * V[j * kv_dim + kv_off + d];
        }
        output[out_off + d] = val;
    }
}
"#;
