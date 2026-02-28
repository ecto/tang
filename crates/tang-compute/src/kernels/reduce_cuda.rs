//! CUDA reduction kernels: softmax, rms_norm.

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
