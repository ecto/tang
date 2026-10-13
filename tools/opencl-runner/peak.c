// peak: clpeak-style fp32/fp16 FMA throughput and global-memory bandwidth,
// so tang's numbers can be stated as % of what this GPU actually does.
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>

static const char *SRC =
    "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
    "__kernel void f32(__global float* o, float a) {\n"
    "  float4 x = (float4)(get_global_id(0), 1, 2, 3), y = x * 0.5f;\n"
    "  for (int i = 0; i < 256; i++) { x = mad(x, y, a); y = mad(y, x, a); }\n"
    "  o[get_global_id(0)] = x.s0 + x.s1 + x.s2 + x.s3 + y.s0;\n"
    "}\n"
    "__kernel void f16(__global half* o, float af) {\n"
    "  half a = (half)af; half8 x = (half8)(get_global_id(0)), y = x * (half)0.5f;\n"
    "  for (int i = 0; i < 256; i++) { x = mad(x, y, a); y = mad(y, x, a); }\n"
    "  o[get_global_id(0)] = x.s0 + x.s7 + y.s3;\n"
    "}\n"
    "__kernel void copy(__global const float4* a, __global float4* b) {\n"
    "  b[get_global_id(0)] = a[get_global_id(0)];\n"
    "}\n";

static double run(cl_command_queue q, cl_kernel k, size_t n) {
    double best = 1e30;
    for (int r = 0; r < 10; r++) {
        cl_event e;
        size_t l = 256;
        clEnqueueNDRangeKernel(q, k, 1, NULL, &n, &l, 0, NULL, &e);
        clWaitForEvents(1, &e);
        cl_ulong s, t;
        clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_START, 8, &s, NULL);
        clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END, 8, &t, NULL);
        if ((t - s) * 1e-9 < best) best = (t - s) * 1e-9;
        clReleaseEvent(e);
    }
    return best;
}

int main(void) {
    cl_platform_id p;
    cl_device_id d;
    clGetPlatformIDs(1, &p, NULL);
    clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &d, NULL);
    cl_uint mhz;
    clGetDeviceInfo(d, CL_DEVICE_MAX_CLOCK_FREQUENCY, 4, &mhz, NULL);
    cl_context c = clCreateContext(NULL, 1, &d, NULL, NULL, NULL);
    cl_queue_properties qp[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue q = clCreateCommandQueueWithProperties(c, d, qp, NULL);
    cl_program pr = clCreateProgramWithSource(c, 1, &SRC, NULL, NULL);
    if (clBuildProgram(pr, 1, &d, "", NULL, NULL)) {
        printf("build failed\n");
        return 1;
    }
    size_t n = 1 << 20, bytes = 64 << 20;
    cl_mem o = clCreateBuffer(c, CL_MEM_READ_WRITE, n * 4, NULL, NULL);
    cl_mem a = clCreateBuffer(c, CL_MEM_READ_WRITE, bytes, NULL, NULL);
    cl_mem b = clCreateBuffer(c, CL_MEM_READ_WRITE, bytes, NULL, NULL);
    float one = 1.0f;
    cl_kernel k32 = clCreateKernel(pr, "f32", NULL), k16 = clCreateKernel(pr, "f16", NULL),
              kc = clCreateKernel(pr, "copy", NULL);
    clSetKernelArg(k32, 0, sizeof o, &o);
    clSetKernelArg(k32, 1, 4, &one);
    clSetKernelArg(k16, 0, sizeof o, &o);
    clSetKernelArg(k16, 1, 4, &one);
    clSetKernelArg(kc, 0, sizeof a, &a);
    clSetKernelArg(kc, 1, sizeof b, &b);
    double t32 = run(q, k32, n), t16 = run(q, k16, n), tc = run(q, kc, bytes / 16);
    printf("max clock %u MHz\n", mhz);
    printf("fp32 FMA: %.0f GFLOP/s\n", n * 256.0 * 2 * 4 * 2 / t32 / 1e9);
    printf("fp16 FMA: %.0f GFLOP/s\n", n * 256.0 * 2 * 8 * 2 / t16 / 1e9);
    printf("copy: %.1f GB/s\n", 2.0 * bytes / tc / 1e9);
    return 0;
}
