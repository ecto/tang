//! CUDA C sources (compiled by NVRTC at runtime).

/// Applies residency changes to the device tables: `upd[i] = {key, slot, ptr}`.
pub const CACHE: &str = r#"
struct Upd { unsigned int key; int slot; unsigned long long ptr; };

extern "C" __global__ void apply_updates(const Upd* __restrict__ upd, int n,
                                         int* __restrict__ resid,
                                         unsigned long long* __restrict__ ptrs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Upd u = upd[i];
    resid[u.key] = u.slot;
    ptrs[u.key] = u.ptr;
}

// out[k] = sum of the 32-bit words of blob k, read through the pointer table (one block per key).
extern "C" __global__ void blob_sums(const unsigned long long* __restrict__ ptrs, int words,
                                     unsigned int* __restrict__ out) {
    const unsigned int* p = (const unsigned int*)ptrs[blockIdx.x];
    unsigned int s = 0;
    for (int i = threadIdx.x; i < words; i += blockDim.x) s += p[i];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    __shared__ unsigned int part[32];
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int t = 0;
        for (int w = 0; w < (int)(blockDim.x >> 5); w++) t += part[w];
        out[blockIdx.x] = t;
    }
}
"#;

/// The expert-shaped work for the benchmarks.
///
/// `q2_expert` is a T=1 matvec over one Q2_0-like expert blob per `blockIdx.y`, laid out as we
/// would repack it: 76,800 groups of 64 2-bit codes (16 B each; gate+up is 1280 rows × 40
/// groups, then down is 2560 rows × 10 groups), then one 16-bit scale per group (read as
/// bf16 here; fp16 costs the same). Total 1,382,400 B, the Flash-Next blob size. Each group is
/// `d · (Σ c·x − Σ x)` with 16 dp4a; x is int8 in shared memory. Grid: (320, n_blobs), 256
/// threads: blocks 0..160 do gate+up (a warp per row), 160..320 do down (half a warp per row).
///
/// Code layout in a group: byte b of 32-bit word j holds, at bits 2s..2s+1, the code for
/// x[16j + 4s + b]. So `(w_j >> 2s) & 0x03030303` lines up with the int8x4 word x[16j+4s..].
pub const BENCH: &str = r#"
#define GU_ROWS 1280
#define GU_G 40
#define DN_ROWS 2560
#define DN_G 10
#define N_GROUPS 76800
#define SCALE_OFF 1228800
#define XPAD 17

__device__ __forceinline__ float group_dot(uint4 c, const int* x, float sx, unsigned short d) {
    int s = 0;
    unsigned int w[4] = {c.x, c.y, c.z, c.w};
#pragma unroll
    for (int j = 0; j < 4; j++) {
#pragma unroll
        for (int sh = 0; sh < 4; sh++) {
            s = __dp4a((int)((w[j] >> (2 * sh)) & 0x03030303u), x[4 * j + sh], s);
        }
    }
    return __uint_as_float(((unsigned int)d) << 16) * ((float)s - sx);
}

extern "C" __global__ void __launch_bounds__(256)
q2_expert(const unsigned long long* __restrict__ ptrs, const int* __restrict__ ids,
          const signed char* __restrict__ xq, const float* __restrict__ xsum,
          float* __restrict__ out) {
    __shared__ int xs[GU_G * XPAD];
    __shared__ float sx[GU_G];
    for (int i = threadIdx.x; i < GU_G * 16; i += blockDim.x)
        xs[(i >> 4) * XPAD + (i & 15)] = ((const int*)xq)[i];
    if (threadIdx.x < GU_G) sx[threadIdx.x] = xsum[threadIdx.x];
    __syncthreads();

    const int b = blockIdx.y;
    const unsigned char* blob = (const unsigned char*)ptrs[ids ? ids[b] : b];
    const uint4* codes = (const uint4*)blob;
    const unsigned short* scales = (const unsigned short*)(blob + SCALE_OFF);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float acc = 0.f;
    if (blockIdx.x < GU_ROWS / 8) {
        const int row = blockIdx.x * 8 + warp;
        for (int g = lane; g < GU_G; g += 32) {
            const int gi = row * GU_G + g;
            acc += group_dot(codes[gi], &xs[g * XPAD], sx[g], scales[gi]);
        }
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) out[(size_t)b * (GU_ROWS + DN_ROWS) + row] = acc;
    } else {
        const int k = lane & 15;
        const int row = (blockIdx.x - GU_ROWS / 8) * 16 + warp * 2 + (lane >> 4);
        if (k < DN_G) {
            const int gi = GU_ROWS * GU_G + row * DN_G + k;
            acc = group_dot(codes[gi], &xs[k * XPAD], sx[k], scales[gi]);
        }
        for (int o = 8; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (k == 0) out[(size_t)b * (GU_ROWS + DN_ROWS) + GU_ROWS + row] = acc;
    }
}

// Raw streaming read of whole blobs (16 B per load), for the bandwidth ceiling.
extern "C" __global__ void stream_sum(const unsigned long long* __restrict__ ptrs,
                                      const int* __restrict__ ids, int vecs,
                                      unsigned int* __restrict__ out) {
    const uint4* p = (const uint4*)ptrs[ids ? ids[blockIdx.y] : blockIdx.y];
    unsigned int s = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vecs; i += gridDim.x * blockDim.x) {
        uint4 v = p[i];
        s ^= v.x + v.y + v.z + v.w;
    }
    if (s == 0x12345678u) out[blockIdx.y] = s;  // keep the loads
}

// Graph overhead: one small kernel per step. `step` lives in device memory.
extern "C" __global__ void tiny(float* buf, const int* step, int i) {
    buf[i * 32 + threadIdx.x] += (float)(*step + i);
}

// A "layer-sized" small kernel: 64K floats, one read-modify-write each.
extern "C" __global__ void medium(float* buf, const int* step, int i) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    buf[t] = buf[t] * 0.999f + (float)(*step + i) * 1e-3f;
}

// The same work as K `tiny` launches in one kernel (a block-wide barrier between steps).
extern "C" __global__ void tiny_fused(float* buf, const int* step, int k) {
    for (int i = 0; i < k; i++) {
        buf[i * 32 + threadIdx.x] += (float)(*step + i);
        __syncthreads();
    }
}

// Ping-pong with a host thread through mapped memory, `rounds` times.
extern "C" __global__ void pingpong(volatile unsigned int* flag, volatile unsigned int* ack,
                                    int rounds) {
    for (unsigned int r = 1; r <= (unsigned int)rounds; r++) {
        while (*flag < r) { }
        __threadfence_system();
        *ack = r;
        __threadfence_system();
    }
}

// Spin until *flag >= target, then exit.
extern "C" __global__ void wait_flag(volatile unsigned int* flag, unsigned int target) {
    while (*flag < target) { }
}
"#;
