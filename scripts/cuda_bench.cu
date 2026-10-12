// Times tang-compile's derived tensor-core matmul kernel against cuBLAS on
// the same device. Build:
//   cargo run -q --release -p tang-bench --example dump_cuda_mma -- M K N > /tmp/tang_kernel.cu
//   nvcc -O3 -arch=sm_86 -I/tmp -o /tmp/cuda_bench scripts/cuda_bench.cu -lcublas
//   /tmp/cuda_bench
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "tang_kernel.cu"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

static float rnd(unsigned& s) { s = s * 1664525u + 1013904223u; return ((s >> 8) & 0xffff) / 65535.0f * 2.0f - 1.0f; }

int main() {
    const int M = TANG_M, K = TANG_K, N = TANG_N;
    unsigned seed = 7;
    std::vector<float> ha(M * K), hb(K * N), hbias(N);
    for (auto& x : ha) x = rnd(seed);
    for (auto& x : hb) x = rnd(seed);
    for (auto& x : hbias) x = rnd(seed);
    float *a, *b, *bias, *out, *ref;
    CK(cudaMalloc(&a, ha.size() * 4)); CK(cudaMalloc(&b, hb.size() * 4));
    CK(cudaMalloc(&bias, hbias.size() * 4));
    CK(cudaMalloc(&out, (size_t)TANG_OUT_LEN * 4)); CK(cudaMalloc(&ref, (size_t)M * N * 4));
    CK(cudaMemcpy(a, ha.data(), ha.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b, hb.data(), hb.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(bias, hbias.data(), hbias.size() * 4, cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    const int iters = 50;
    auto launch = [&]() { TANG_LAUNCH(a, b, bias, out); };
    for (int i = 0; i < 5; i++) launch();
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(t0));
    for (int i = 0; i < iters; i++) launch();
    CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
    float ms_tang; CK(cudaEventElapsedTime(&ms_tang, t0, t1)); ms_tang /= iters;

    // cuBLAS, row-major C = A·B as column-major C^T = B^T·A^T
    cublasHandle_t h; cublasCreate(&h);
    const float one = 1.0f, zero = 0.0f;
    auto gemm = [&]() {
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, b, N, a, K, &zero, ref, N);
    };
    cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
    for (int i = 0; i < 5; i++) gemm();
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(t0));
    for (int i = 0; i < iters; i++) gemm();
    CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
    float ms_fp32; CK(cudaEventElapsedTime(&ms_fp32, t0, t1)); ms_fp32 /= iters;
    std::vector<float> href((size_t)M * N), hout(TANG_OUT_LEN);
    CK(cudaMemcpy(href.data(), ref, href.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hout.data(), out, hout.size() * 4, cudaMemcpyDeviceToHost));

    cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH);
    for (int i = 0; i < 5; i++) gemm();
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(t0));
    for (int i = 0; i < iters; i++) gemm();
    CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
    float ms_tf32; CK(cudaEventElapsedTime(&ms_tf32, t0, t1)); ms_tf32 /= iters;

    // error vs fp32 cuBLAS (TF32 inputs: expect ~1e-3 relative)
    double max_err = 0, max_ref = 0;
    for (size_t i = 0; i < href.size(); i++) {
        float r = href[i];
#if TANG_EPILOGUE
        float s = r + hbias[i % N];
        r = s / (1.0f + expf(-s));
#endif
        max_err = fmax(max_err, fabs(r - hout[i]));
        max_ref = fmax(max_ref, fabs(r));
    }
    double tflops = 2.0 * M * N * K / (ms_tang * 1e-3) / 1e12;
    printf("%dx%dx%d epilogue=%d: tang wmma(tf32) %.3f ms (%.1f TFLOPS)  cublas fp32 %.3f ms  cublas tf32 %.3f ms  max err %.2e (max |ref| %.1f)\n",
           M, K, N, (int)TANG_EPILOGUE, ms_tang, tflops, ms_fp32, ms_tf32, max_err, max_ref);
    return 0;
}
