// launch: the per-launch overhead floor of the first OpenCL GPU.
//
//   launch [n=200]
//
// One tiny kernel (one work-group of 64, one store). Reports the median and
// minimum wall time of enqueue + clFinish for 1 launch, the wall time per
// launch of `n` back-to-back launches before one clFinish, and the
// profiling-event duration (device time) of one launch.
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static int cmp(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 200;
    cl_platform_id plat;
    cl_device_id dev;
    clGetPlatformIDs(1, &plat, NULL);
    clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    cl_int err;
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    cl_queue_properties qp[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, qp, &err);
    const char *src =
        "__kernel void k(__global float* o) { if (get_global_id(0) == 0) o[0] += 1.0f; }";
    cl_program p = clCreateProgramWithSource(ctx, 1, &src, NULL, &err);
    if (clBuildProgram(p, 1, &dev, "", NULL, NULL) != CL_SUCCESS) {
        fprintf(stderr, "build failed\n");
        return 1;
    }
    cl_kernel k = clCreateKernel(p, "k", &err);
    cl_mem m = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 4, NULL, &err);
    clSetKernelArg(k, 0, sizeof m, &m);
    size_t g = 64, l = 64;
    double *t = malloc(n * sizeof *t), *d = malloc(n * sizeof *d);
    for (int i = 0; i < 20; i++) clEnqueueNDRangeKernel(q, k, 1, NULL, &g, &l, 0, NULL, NULL);
    clFinish(q);
    for (int i = 0; i < n; i++) {
        cl_event e;
        double t0 = now();
        clEnqueueNDRangeKernel(q, k, 1, NULL, &g, &l, 0, NULL, &e);
        clFinish(q);
        t[i] = (now() - t0) * 1e6;
        cl_ulong s0, s1;
        clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_START, sizeof s0, &s0, NULL);
        clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END, sizeof s1, &s1, NULL);
        d[i] = (s1 - s0) * 1e-3;
        clReleaseEvent(e);
    }
    qsort(t, n, sizeof *t, cmp);
    qsort(d, n, sizeof *d, cmp);
    double t0 = now();
    for (int i = 0; i < n; i++) clEnqueueNDRangeKernel(q, k, 1, NULL, &g, &l, 0, NULL, NULL);
    clFinish(q);
    double batched = (now() - t0) * 1e6 / n;
    // the same, but clFlush and spin on the event instead of clFinish
    double *sp = malloc(n * sizeof *sp);
    for (int i = 0; i < n; i++) {
        cl_event e;
        double t1 = now();
        clEnqueueNDRangeKernel(q, k, 1, NULL, &g, &l, 0, NULL, &e);
        clFlush(q);
        cl_int st = CL_QUEUED;
        while (st != CL_COMPLETE && st >= 0)
            clGetEventInfo(e, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof st, &st, NULL);
        sp[i] = (now() - t1) * 1e6;
        clReleaseEvent(e);
    }
    qsort(sp, n, sizeof *sp, cmp);
    printf("single launch + flush + spin: median %.1f us, min %.1f us, p95 %.1f us\n",
           sp[n / 2], sp[0], sp[n * 95 / 100]);
    printf("single launch + finish: median %.1f us, min %.1f us, p95 %.1f us\n", t[n / 2], t[0],
           t[n * 95 / 100]);
    printf("device time of one launch: median %.1f us\n", d[n / 2]);
    printf("%d back-to-back launches, one finish: %.1f us per launch\n", n, batched);
    return 0;
}
