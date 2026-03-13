//! Fused norm-relative Gaussian noise injection kernel.
//!
//! Per-row: compute L2 norm, generate Gaussian noise via splitmix64 + Box-Muller,
//! then in-place: `hidden[i] += epsilon * ||row|| * noise[i]`.
//! One block per row, shared-memory reduction for norm.

/// Fused norm-relative noise injection kernel (f32).
/// Grid: (rows), Block: (min(cols, 256)), Shared: block_dim * 4 bytes.
pub const NORM_RELATIVE_NOISE_F32_CUDA: &str = r#"
extern "C" __global__ void norm_relative_noise_f32(
    float* __restrict__ data,
    const float epsilon,
    const unsigned long long seed,
    const unsigned int rows,
    const unsigned int cols)
{
    const unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int block_size = blockDim.x;
    float* row_data = data + (unsigned long long)row * cols;

    // Phase 1: compute row L2 norm via shared-memory reduction
    extern __shared__ float sdata[];
    float local_sq = 0.0f;
    for (unsigned int i = tid; i < cols; i += block_size) {
        float v = row_data[i];
        local_sq += v * v;
    }
    sdata[tid] = local_sq;
    __syncthreads();

    // Tree reduction
    for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float row_norm = sqrtf(sdata[0]);

    // Phase 2: generate noise and add in-place
    float scale = epsilon * row_norm;
    for (unsigned int i = tid; i < cols; i += block_size) {
        // Counter-based PRNG: splitmix64
        unsigned long long key = seed ^ ((unsigned long long)row * cols + i);
        key += 0x9e3779b97f4a7c15ULL;
        key = (key ^ (key >> 30)) * 0xbf58476d1ce4e5b9ULL;
        key = (key ^ (key >> 27)) * 0x94d049bb133111ebULL;
        key = key ^ (key >> 31);

        // Second independent hash for Box-Muller pair
        unsigned long long key2 = (seed ^ ((unsigned long long)row * cols + i)) + 0x6a09e667f3bcc908ULL;
        key2 += 0x9e3779b97f4a7c15ULL;
        key2 = (key2 ^ (key2 >> 30)) * 0xbf58476d1ce4e5b9ULL;
        key2 = (key2 ^ (key2 >> 27)) * 0x94d049bb133111ebULL;
        key2 = key2 ^ (key2 >> 31);

        // Box-Muller transform: two uniform -> one Gaussian
        // Map to (0, 1) avoiding exact 0
        float u1 = ((float)(key & 0x7FFFFFFFU) + 1.0f) / 2147483649.0f;
        float u2 = ((float)(key2 & 0x7FFFFFFFU) + 1.0f) / 2147483649.0f;
        float noise = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853071795864f * u2);

        row_data[i] += scale * noise;
    }
}
"#;

/// Fused norm-relative noise injection kernel (bf16).
/// Grid: (rows), Block: (min(cols, 256)), Shared: block_dim * 4 bytes.
pub const NORM_RELATIVE_NOISE_BF16_CUDA: &str = r#"
__device__ float bf16_to_float(unsigned short bits) {
    return __int_as_float(((unsigned int)bits) << 16);
}

__device__ unsigned short float_to_bf16(float val) {
    unsigned int bits = __float_as_int(val);
    unsigned int lsb = (bits >> 16) & 1;
    bits += 0x7FFF + lsb;
    return (unsigned short)(bits >> 16);
}

extern "C" __global__ void norm_relative_noise_bf16(
    unsigned short* __restrict__ data,
    const float epsilon,
    const unsigned long long seed,
    const unsigned int rows,
    const unsigned int cols)
{
    const unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int block_size = blockDim.x;
    unsigned short* row_data = data + (unsigned long long)row * cols;

    // Phase 1: compute row L2 norm in f32 via shared-memory reduction
    extern __shared__ float sdata[];
    float local_sq = 0.0f;
    for (unsigned int i = tid; i < cols; i += block_size) {
        float v = bf16_to_float(row_data[i]);
        local_sq += v * v;
    }
    sdata[tid] = local_sq;
    __syncthreads();

    for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float row_norm = sqrtf(sdata[0]);

    // Phase 2: generate noise and add in-place (compute in f32, store bf16)
    float scale = epsilon * row_norm;
    for (unsigned int i = tid; i < cols; i += block_size) {
        unsigned long long key = seed ^ ((unsigned long long)row * cols + i);
        key += 0x9e3779b97f4a7c15ULL;
        key = (key ^ (key >> 30)) * 0xbf58476d1ce4e5b9ULL;
        key = (key ^ (key >> 27)) * 0x94d049bb133111ebULL;
        key = key ^ (key >> 31);

        unsigned long long key2 = (seed ^ ((unsigned long long)row * cols + i)) + 0x6a09e667f3bcc908ULL;
        key2 += 0x9e3779b97f4a7c15ULL;
        key2 = (key2 ^ (key2 >> 30)) * 0xbf58476d1ce4e5b9ULL;
        key2 = (key2 ^ (key2 >> 27)) * 0x94d049bb133111ebULL;
        key2 = key2 ^ (key2 >> 31);

        float u1 = ((float)(key & 0x7FFFFFFFU) + 1.0f) / 2147483649.0f;
        float u2 = ((float)(key2 & 0x7FFFFFFFU) + 1.0f) / 2147483649.0f;
        float noise = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853071795864f * u2);

        float val = bf16_to_float(row_data[i]);
        row_data[i] = float_to_bf16(val + scale * noise);
    }
}
"#;
