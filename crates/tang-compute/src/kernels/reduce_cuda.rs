//! CUDA reduction kernels: softmax, rms_norm, embedding_gather, reduce_sum.

/// Row-wise softmax. Each block handles one row.
/// Grid: (n_rows), Block: (min(row_len, 256))
pub const SOFTMAX_CUDA: &str = r#"
extern "C" __global__ void softmax(
    const float* __restrict__ input,
    float* __restrict__ output,
    const unsigned int n_rows,
    const unsigned int row_len)
{
    __shared__ float shared[256];

    unsigned int row = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int tg_size = blockDim.x;
    if (row >= n_rows) return;

    unsigned int base = row * row_len;

    // Phase 1: max
    float local_max = -1e38f;
    for (unsigned int i = tid; i < row_len; i += tg_size) {
        local_max = fmaxf(local_max, input[base + i]);
    }
    shared[tid] = local_max;
    __syncthreads();

    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }
    float row_max = shared[0];
    __syncthreads();  // barrier before Phase 2 reuses shared[]

    // Phase 2: exp and sum
    float local_sum = 0.0f;
    for (unsigned int i = tid; i < row_len; i += tg_size) {
        float val = expf(input[base + i] - row_max);
        output[base + i] = val;
        local_sum += val;
    }
    shared[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }

    // Phase 3: normalize
    float inv_sum = 1.0f / shared[0];
    for (unsigned int i = tid; i < row_len; i += tg_size) {
        output[base + i] *= inv_sum;
    }
}
"#;

/// RMS normalization. Each block handles one group.
/// Grid: (n_groups), Block: (min(dim, 256))
pub const RMS_NORM_CUDA: &str = r#"
extern "C" __global__ void rms_norm(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
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

    // Sum of squares
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

    float rms = rsqrtf(shared[0] / float(dim) + eps);

    // Normalize
    for (unsigned int i = tid; i < dim; i += tg_size) {
        output[base + i] = input[base + i] * rms * weight[i];
    }
}
"#;

/// bf16 row-wise softmax. Input/output are bf16, shared memory stays f32.
pub const SOFTMAX_BF16_CUDA: &str = r#"
__device__ float bf16_to_float(unsigned short bits) {
    return __int_as_float(((unsigned int)bits) << 16);
}
__device__ unsigned short float_to_bf16(float val) {
    unsigned int bits = __float_as_int(val);
    unsigned int lsb = (bits >> 16) & 1;
    bits += 0x7FFF + lsb;
    return (unsigned short)(bits >> 16);
}

extern "C" __global__ void softmax_bf16(
    const unsigned short* __restrict__ input,
    unsigned short* __restrict__ output,
    const unsigned int n_rows,
    const unsigned int row_len)
{
    __shared__ float shared[256];

    unsigned int row = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int tg_size = blockDim.x;
    if (row >= n_rows) return;

    unsigned int base = row * row_len;

    // Phase 1: max
    float local_max = -1e38f;
    for (unsigned int i = tid; i < row_len; i += tg_size) {
        local_max = fmaxf(local_max, bf16_to_float(input[base + i]));
    }
    shared[tid] = local_max;
    __syncthreads();

    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }
    float row_max = shared[0];
    __syncthreads();  // barrier before Phase 2 reuses shared[]

    // Phase 2: exp and sum
    float local_sum = 0.0f;
    for (unsigned int i = tid; i < row_len; i += tg_size) {
        float val = expf(bf16_to_float(input[base + i]) - row_max);
        output[base + i] = float_to_bf16(val);
        local_sum += val;
    }
    shared[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }

    // Phase 3: normalize
    float inv_sum = 1.0f / shared[0];
    for (unsigned int i = tid; i < row_len; i += tg_size) {
        output[base + i] = float_to_bf16(bf16_to_float(output[base + i]) * inv_sum);
    }
}
"#;

/// bf16 RMS normalization. Input/weight/output are bf16, shared memory stays f32.
pub const RMS_NORM_BF16_CUDA: &str = r#"
__device__ float bf16_to_float(unsigned short bits) {
    return __int_as_float(((unsigned int)bits) << 16);
}
__device__ unsigned short float_to_bf16(float val) {
    unsigned int bits = __float_as_int(val);
    unsigned int lsb = (bits >> 16) & 1;
    bits += 0x7FFF + lsb;
    return (unsigned short)(bits >> 16);
}

extern "C" __global__ void rms_norm_bf16(
    const unsigned short* __restrict__ input,
    const unsigned short* __restrict__ weight,
    unsigned short* __restrict__ output,
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

    // Sum of squares
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

    float rms = rsqrtf(shared[0] / float(dim) + eps);

    // Normalize
    for (unsigned int i = tid; i < dim; i += tg_size) {
        output[base + i] = float_to_bf16(bf16_to_float(input[base + i]) * rms * bf16_to_float(weight[i]));
    }
}
"#;

/// Embedding gather: output[i, d] = weight[ids[i], d].
/// Each thread handles one element of the [seq_len, dim] output.
/// Grid: (ceil(seq_len*dim / 256)), Block: (256)
///
/// `ids` buffer stores u32 token IDs as f32 bit patterns (via upload_u32).
/// The kernel casts the pointer to `unsigned int*` to read them directly.
pub const EMBEDDING_GATHER_CUDA: &str = r#"
extern "C" __global__ void embedding_gather(
    const float* __restrict__ weight,
    const unsigned int* __restrict__ ids,
    float* __restrict__ output,
    const unsigned int seq_len,
    const unsigned int dim)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = seq_len * dim;
    if (idx >= total) return;

    unsigned int pos = idx / dim;
    unsigned int d = idx % dim;
    unsigned int token_id = ids[pos];
    output[idx] = weight[token_id * dim + d];
}
"#;

/// bf16 embedding gather: weight is bf16, ids is u32, output is bf16.
/// Each thread copies one bf16 element (no conversion needed, just a gather).
/// Grid: (ceil(seq_len*dim / 256)), Block: (256)
pub const EMBEDDING_GATHER_BF16_CUDA: &str = r#"
extern "C" __global__ void embedding_gather_bf16(
    const unsigned short* __restrict__ weight,
    const unsigned int* __restrict__ ids,
    unsigned short* __restrict__ output,
    const unsigned int seq_len,
    const unsigned int dim)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = seq_len * dim;
    if (idx >= total) return;

    unsigned int pos = idx / dim;
    unsigned int d = idx % dim;
    unsigned int token_id = ids[pos];
    output[idx] = weight[token_id * dim + d];
}
"#;

/// Reduce-sum along an arbitrary axis.
/// Given tensor with shape decomposed into (outer, axis_len, inner),
/// each block handles one output element at (outer_idx, inner_idx),
/// summing axis_len values with shared-memory parallel reduction.
///
/// Grid: (outer * inner), Block: (min(axis_len, 256) rounded to power-of-two)
/// Params: input, output, outer, axis_len, inner
pub const REDUCE_SUM_CUDA: &str = r#"
extern "C" __global__ void reduce_sum(
    const float* __restrict__ input,
    float* __restrict__ output,
    const unsigned int outer,
    const unsigned int axis_len,
    const unsigned int inner)
{
    __shared__ float shared[256];

    unsigned int out_idx = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int tg_size = blockDim.x;

    unsigned int total_out = outer * inner;
    if (out_idx >= total_out) return;

    unsigned int o = out_idx / inner;
    unsigned int i = out_idx % inner;

    // Each thread accumulates a partial sum over the reduction axis
    float local_sum = 0.0f;
    unsigned int base = o * axis_len * inner + i;
    for (unsigned int a = tid; a < axis_len; a += tg_size) {
        local_sum += input[base + a * inner];
    }
    shared[tid] = local_sum;
    __syncthreads();

    // Tree reduction in shared memory
    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[out_idx] = shared[0];
    }
}
"#;

/// Reduce-sum along an axis, accumulating into output (output[i] += ...).
/// Same structure as reduce_sum but uses += instead of =.
/// Grid: (outer * inner), Block: (min(axis_len, 256) rounded to power-of-two)
pub const REDUCE_SUM_ACCUMULATE_CUDA: &str = r#"
extern "C" __global__ void reduce_sum_accumulate(
    const float* __restrict__ input,
    float* __restrict__ output,
    const unsigned int outer,
    const unsigned int axis_len,
    const unsigned int inner)
{
    __shared__ float shared[256];

    unsigned int out_idx = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int tg_size = blockDim.x;

    unsigned int total_out = outer * inner;
    if (out_idx >= total_out) return;

    unsigned int o = out_idx / inner;
    unsigned int i = out_idx % inner;

    float local_sum = 0.0f;
    unsigned int base = o * axis_len * inner + i;
    for (unsigned int a = tid; a < axis_len; a += tg_size) {
        local_sum += input[base + a * inner];
    }
    shared[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[out_idx] += shared[0];
    }
}
"#;

/// Reduce-sum along an axis, reading bf16 input, accumulating into f32 output.
/// Fused bf16→f32 conversion + reduction — avoids separate convert-then-reduce.
/// Grid: (outer * inner), Block: (min(axis_len, 256) rounded to power-of-two)
pub const REDUCE_SUM_ACCUMULATE_BF16_CUDA: &str = r#"
__device__ float bf16_to_float_rsa(unsigned short bits) {
    unsigned int x = ((unsigned int)bits) << 16;
    float result;
    asm("mov.b32 %0, %1;" : "=f"(result) : "r"(x));
    return result;
}
extern "C" __global__ void reduce_sum_accumulate_bf16(
    const unsigned short* __restrict__ input,
    float* __restrict__ output,
    const unsigned int outer,
    const unsigned int axis_len,
    const unsigned int inner)
{
    __shared__ float shared[256];

    unsigned int out_idx = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int tg_size = blockDim.x;

    unsigned int total_out = outer * inner;
    if (out_idx >= total_out) return;

    unsigned int o = out_idx / inner;
    unsigned int i = out_idx % inner;

    float local_sum = 0.0f;
    unsigned int base = o * axis_len * inner + i;
    for (unsigned int a = tid; a < axis_len; a += tg_size) {
        local_sum += bf16_to_float_rsa(input[base + a * inner]);
    }
    shared[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[out_idx] += shared[0];
    }
}
"#;

/// bf16 reduce-sum along an arbitrary axis.
/// Input/output are bf16 (unsigned short), reduction in f32.
pub const REDUCE_SUM_BF16_CUDA: &str = r#"
__device__ float bf16_to_float(unsigned short bits) {
    return __int_as_float(((unsigned int)bits) << 16);
}
__device__ unsigned short float_to_bf16(float val) {
    unsigned int bits = __float_as_int(val);
    unsigned int lsb = (bits >> 16) & 1;
    bits += 0x7FFF + lsb;
    return (unsigned short)(bits >> 16);
}

extern "C" __global__ void reduce_sum_bf16(
    const unsigned short* __restrict__ input,
    unsigned short* __restrict__ output,
    const unsigned int outer,
    const unsigned int axis_len,
    const unsigned int inner)
{
    __shared__ float shared[256];

    unsigned int out_idx = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int tg_size = blockDim.x;

    unsigned int total_out = outer * inner;
    if (out_idx >= total_out) return;

    unsigned int o = out_idx / inner;
    unsigned int i = out_idx % inner;

    // Each thread accumulates a partial sum in f32
    float local_sum = 0.0f;
    unsigned int base = o * axis_len * inner + i;
    for (unsigned int a = tid; a < axis_len; a += tg_size) {
        local_sum += bf16_to_float(input[base + a * inner]);
    }
    shared[tid] = local_sum;
    __syncthreads();

    // Tree reduction in shared memory
    for (unsigned int s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[out_idx] = float_to_bf16(shared[0]);
    }
}
"#;
