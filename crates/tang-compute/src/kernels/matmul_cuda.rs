//! Hand-optimized CUDA matmul kernel.

/// Tiled CUDA matmul kernel using shared memory.
///
/// A: [M, K], B: [K, N], C: [M, N], row-major.
/// Uses shared memory tiling with 16×16 tiles.
///
/// Dispatch: grid = ceil(N/16) × ceil(M/16), block = 16×16.
pub const MATMUL_CUDA: &str = r#"
#define TILE 16

extern "C" __global__ void matmul(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const unsigned int M,
    const unsigned int K,
    const unsigned int N)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    unsigned int row = blockIdx.y * TILE + threadIdx.y;
    unsigned int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (unsigned int t = 0; t < (K + TILE - 1) / TILE; t++) {
        unsigned int ak = t * TILE + threadIdx.x;
        unsigned int bk = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && ak < K) ? A[row * K + ak] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bk < K && col < N) ? B[bk * N + col] : 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE; i++) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
"#;
