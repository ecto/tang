// tang_cl: run a tang program exported as OpenCL C (`opencl_export`) on
// the first OpenCL GPU, with no Rust toolchain on the device.
//
//   tang_cl DIR [budget_seconds=2] [max_reps=50] [build options...]
//
// DIR/manifest.txt (written by `cargo run -p tang-bench --example
// opencl_export`):
//
//   tang-opencl 1
//   buffers N
//   b <bytes> <input k | -1>              one per buffer, in order
//   kernels M
//   k <file.cl> <wg> <count> <points> <nargs> <buf>...   reads, then out
//   outputs P
//   o <buf> <file.bin>
//   candidates C                          optional: schedule candidates
//   c <base k> <file.cl> <wg> <count> <points> <nargs> <buf>...
//
// Inputs are DIR/in<k>.bin, raw bytes in the buffer's storage format.
// Kernels are built once per distinct source (canonical sources repeat),
// the program runs `reps` times (until `budget_seconds` of wall time or
// `max_reps`), and per-kernel / whole-program times are the minimum over
// repeats of OpenCL profiling events. Writes DIR/<output files> and
// DIR/times.txt (`kernel i ms` lines, then `program ms`).
//
// Candidates (from `opencl_export tune-export`) are alternative kernels
// for base kernel `k` that write the same buffer. After the program runs,
// each is timed (min of up to 5 runs; one run if it is > 4x the base) and
// its output compared with the base kernel's; DIR/tune.txt gets `device
// <name>` then `c i ms rel_err` lines. The SoC temperature is checked
// before every candidate: at >= 82 C the run pauses until <= 78 C.
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHECK(e, what)                                                              \
    do {                                                                            \
        cl_int err_ = (e);                                                          \
        if (err_ != CL_SUCCESS) {                                                   \
            fprintf(stderr, "%s failed: %d (%s:%d)\n", what, err_, __FILE__, __LINE__); \
            exit(1);                                                                \
        }                                                                           \
    } while (0)

static char *slurp(const char *dir, const char *name, size_t *len) {
    char path[4096];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    rewind(f);
    char *buf = malloc(n + 1);
    if (fread(buf, 1, n, f) != (size_t)n) {
        fprintf(stderr, "short read %s\n", path);
        exit(1);
    }
    buf[n] = 0;
    fclose(f);
    if (len) *len = n;
    return buf;
}

static int max_temp(void) {
    int best = 0;
    for (int z = 0; z < 128; z++) {
        char path[128];
        snprintf(path, sizeof path, "/sys/class/thermal/thermal_zone%d/temp", z);
        FILE *f = fopen(path, "r");
        if (!f) continue;
        int t;
        if (fscanf(f, "%d", &t) == 1 && t > best && t < 200000) best = t;
        fclose(f);
    }
    return best;
}

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

typedef struct {
    char file[256];
    unsigned wg, count, points, nargs;
    int *args;
    cl_kernel kernel;
} Kernel;

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s DIR [budget_s] [max_reps] [build opts]\n", argv[0]);
        return 2;
    }
    const char *dir = argv[1];
    double budget = argc > 2 ? atof(argv[2]) : 2.0;
    int max_reps = argc > 3 ? atoi(argv[3]) : 50;
    char opts[1024] = "-cl-std=CL3.0";
    for (int i = 4; i < argc; i++) {
        strncat(opts, " ", sizeof opts - strlen(opts) - 1);
        strncat(opts, argv[i], sizeof opts - strlen(opts) - 1);
    }

    char *manifest = slurp(dir, "manifest.txt", NULL);
    char *cur = manifest;
    int nbuf = 0, nk = 0, nout = 0, used = 0, version = 0;
    if (sscanf(cur, "tang-opencl %d\nbuffers %d%n", &version, &nbuf, &used) != 2 || version != 1) {
        fprintf(stderr, "bad manifest header\n");
        return 1;
    }
    cur += used;
    size_t *bytes = calloc(nbuf, sizeof *bytes);
    int *input = calloc(nbuf, sizeof *input);
    for (int i = 0; i < nbuf; i++) {
        if (sscanf(cur, " b %zu %d%n", &bytes[i], &input[i], &used) != 2) {
            fprintf(stderr, "bad buffer line %d\n", i);
            return 1;
        }
        cur += used;
    }
    sscanf(cur, " kernels %d%n", &nk, &used);
    cur += used;
    Kernel *ks = calloc(nk, sizeof *ks);
    for (int i = 0; i < nk; i++) {
        Kernel *k = &ks[i];
        if (sscanf(cur, " k %255s %u %u %u %u%n", k->file, &k->wg, &k->count, &k->points,
                   &k->nargs, &used) != 5) {
            fprintf(stderr, "bad kernel line %d\n", i);
            return 1;
        }
        cur += used;
        k->args = calloc(k->nargs, sizeof(int));
        for (unsigned a = 0; a < k->nargs; a++) {
            sscanf(cur, " %d%n", &k->args[a], &used);
            cur += used;
        }
    }
    sscanf(cur, " outputs %d%n", &nout, &used);
    cur += used;
    int *out_buf = calloc(nout, sizeof(int));
    char (*out_file)[256] = calloc(nout, 256);
    for (int i = 0; i < nout; i++) {
        sscanf(cur, " o %d %255s%n", &out_buf[i], out_file[i], &used);
        cur += used;
    }
    int nc = 0;
    if (sscanf(cur, " candidates %d%n", &nc, &used) == 1) cur += used;
    else nc = 0;
    Kernel *cs = calloc(nc ? nc : 1, sizeof *cs);
    int *cbase = calloc(nc ? nc : 1, sizeof(int));
    for (int i = 0; i < nc; i++) {
        Kernel *k = &cs[i];
        if (sscanf(cur, " c %d %255s %u %u %u %u%n", &cbase[i], k->file, &k->wg, &k->count,
                   &k->points, &k->nargs, &used) != 6) {
            fprintf(stderr, "bad candidate line %d\n", i);
            return 1;
        }
        cur += used;
        k->args = calloc(k->nargs, sizeof(int));
        for (unsigned a = 0; a < k->nargs; a++) {
            sscanf(cur, " %d%n", &k->args[a], &used);
            cur += used;
        }
    }

    cl_platform_id plat;
    cl_device_id dev;
    CHECK(clGetPlatformIDs(1, &plat, NULL), "clGetPlatformIDs");
    CHECK(clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL), "clGetDeviceIDs");
    char name[256];
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof name, name, NULL);
    cl_int err;
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    CHECK(err, "clCreateContext");
    cl_queue_properties qp[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, qp, &err);
    CHECK(err, "clCreateCommandQueue");
    printf("device: %s, %d buffers, %d kernels, opts '%s'\n", name, nbuf, nk, opts);

    // buffers: inputs uploaded, the rest zeroed
    cl_mem *mem = calloc(nbuf, sizeof *mem);
    size_t total = 0;
    for (int i = 0; i < nbuf; i++) {
        size_t n = bytes[i] < 4 ? 4 : bytes[i];
        total += n;
        mem[i] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n, NULL, &err);
        CHECK(err, "clCreateBuffer");
        if (input[i] >= 0) {
            char fname[64];
            snprintf(fname, sizeof fname, "in%d.bin", input[i]);
            size_t len;
            char *data = slurp(dir, fname, &len);
            if (len != bytes[i]) {
                fprintf(stderr, "%s: %zu bytes, want %zu\n", fname, len, bytes[i]);
                return 1;
            }
            CHECK(clEnqueueWriteBuffer(q, mem[i], CL_TRUE, 0, len, data, 0, NULL, NULL),
                  "write input");
            free(data);
        } else {
            unsigned zero = 0;
            CHECK(clEnqueueFillBuffer(q, mem[i], &zero, 4, 0, n, 0, NULL, NULL), "fill");
        }
    }
    printf("buffers: %.1f MiB\n", total / 1048576.0);

    // build each distinct source once
    char **srcs = calloc(nk, sizeof(char *));
    cl_program *progs = calloc(nk, sizeof(cl_program));
    double t_build = now();
    int distinct = 0;
    for (int i = 0; i < nk; i++) {
        srcs[i] = slurp(dir, ks[i].file, NULL);
        int same = -1;
        for (int j = 0; j < i; j++)
            if (strcmp(srcs[i], srcs[j]) == 0) {
                same = j;
                break;
            }
        if (same >= 0) {
            progs[i] = progs[same];
        } else {
            distinct++;
            const char *s = srcs[i];
            progs[i] = clCreateProgramWithSource(ctx, 1, &s, NULL, &err);
            CHECK(err, "clCreateProgramWithSource");
            err = clBuildProgram(progs[i], 1, &dev, opts, NULL, NULL);
            if (err != CL_SUCCESS) {
                size_t ln;
                clGetProgramBuildInfo(progs[i], dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &ln);
                char *log = malloc(ln + 1);
                clGetProgramBuildInfo(progs[i], dev, CL_PROGRAM_BUILD_LOG, ln, log, NULL);
                fprintf(stderr, "build %s failed (%d):\n%.4000s\n", ks[i].file, err, log);
                return 1;
            }
        }
        ks[i].kernel = clCreateKernel(progs[i], "main0", &err);
        CHECK(err, "clCreateKernel");
        for (unsigned a = 0; a < ks[i].nargs; a++)
            CHECK(clSetKernelArg(ks[i].kernel, a, sizeof(cl_mem), &mem[ks[i].args[a]]), "arg");
        CHECK(clSetKernelArg(ks[i].kernel, ks[i].nargs, sizeof(unsigned), &ks[i].points),
              "arg count");
    }
    printf("built %d distinct kernels in %.2f s\n", distinct, now() - t_build);

    // run
    double *best = malloc(nk * sizeof(double));
    for (int i = 0; i < nk; i++) best[i] = 1e30;
    double best_prog = 1e30, best_wall = 1e30;
    cl_event *ev = calloc(nk, sizeof(cl_event));
    double t0 = now();
    int reps = 0;
    while (reps < max_reps && (reps < 2 || now() - t0 < budget)) {
        double w0 = now();
        for (int i = 0; i < nk; i++) {
            size_t local = ks[i].wg;
            size_t global = (ks[i].count + local - 1) / local * local;
            CHECK(clEnqueueNDRangeKernel(q, ks[i].kernel, 1, NULL, &global, &local, 0, NULL,
                                         &ev[i]),
                  ks[i].file);
        }
        CHECK(clFinish(q), "clFinish");
        double wall = now() - w0;
        cl_ulong first = 0, last = 0;
        for (int i = 0; i < nk; i++) {
            cl_ulong s, e;
            clGetEventProfilingInfo(ev[i], CL_PROFILING_COMMAND_START, sizeof s, &s, NULL);
            clGetEventProfilingInfo(ev[i], CL_PROFILING_COMMAND_END, sizeof e, &e, NULL);
            double ms = (e - s) * 1e-6;
            if (ms < best[i]) best[i] = ms;
            if (i == 0) first = s;
            last = e;
            clReleaseEvent(ev[i]);
        }
        double prog = (last - first) * 1e-6;
        if (reps > 0) {  // the first run warms caches and the clock
            if (prog < best_prog) best_prog = prog;
            if (wall * 1e3 < best_wall) best_wall = wall * 1e3;
        }
        reps++;
    }

    double sum = 0;
    char tpath[4096];
    snprintf(tpath, sizeof tpath, "%s/times.txt", dir);
    FILE *tf = fopen(tpath, "w");
    for (int i = 0; i < nk; i++) {
        sum += best[i];
        fprintf(tf, "kernel %d %.6f\n", i, best[i]);
    }
    fprintf(tf, "program %.6f\nwall %.6f\nreps %d\n", best_prog, best_wall, reps);
    fclose(tf);
    printf("reps %d: program %.3f ms (events), wall %.3f ms, sum of kernels %.3f ms\n", reps,
           best_prog, best_wall, sum);
    for (int i = 0; i < nk; i++)
        printf("  k%-3d %9.3f ms  wg %4u  groups %6u  %s\n", i, best[i], ks[i].wg,
               (ks[i].count + ks[i].wg - 1) / ks[i].wg, ks[i].file);

    if (nc > 0) {
        char tp[4096];
        snprintf(tp, sizeof tp, "%s/tune.txt", dir);
        FILE *tu = fopen(tp, "w");
        fprintf(tu, "device %s\n", name);
        double gpu_ms = 0;
        float *ref = NULL, *got = NULL;
        int ref_k = -1;
        for (int c = 0; c < nc; c++) {
            Kernel *k = &cs[c];
            int b = cbase[c];
            int ob = k->args[k->nargs - 1];
            size_t n = bytes[ob];
            if (b != ref_k) {
                free(ref);
                free(got);
                ref = malloc(n);
                got = malloc(n);
                CHECK(clEnqueueReadBuffer(q, mem[ob], CL_TRUE, 0, n, ref, 0, NULL, NULL), "snap");
                ref_k = b;
            }
            int t = max_temp();
            if (t >= 82000 || gpu_ms > 1500) {
                if (t >= 82000) printf("hot (%d mC), pausing\n", t);
                do {
                    struct timespec ts = {3, 0};
                    nanosleep(&ts, NULL);
                    t = max_temp();
                } while (t > 78000);
                gpu_ms = 0;
            }
            const char *src = slurp(dir, k->file, NULL);
            cl_program pr = clCreateProgramWithSource(ctx, 1, &src, NULL, &err);
            CHECK(err, "candidate program");
            if (clBuildProgram(pr, 1, &dev, opts, NULL, NULL) != CL_SUCCESS) {
                fprintf(tu, "c %d build-failed\n", c);
                clReleaseProgram(pr);
                continue;
            }
            cl_kernel kk = clCreateKernel(pr, "main0", &err);
            CHECK(err, "candidate kernel");
            for (unsigned a = 0; a < k->nargs; a++)
                CHECK(clSetKernelArg(kk, a, sizeof(cl_mem), &mem[k->args[a]]), "arg");
            CHECK(clSetKernelArg(kk, k->nargs, sizeof(unsigned), &k->points), "arg count");
            double bestc = 1e30;
            for (int r = 0; r < 6; r++) {
                size_t local = k->wg;
                size_t global = (k->count + local - 1) / local * local;
                cl_event e1;
                CHECK(clEnqueueNDRangeKernel(q, kk, 1, NULL, &global, &local, 0, NULL, &e1),
                      k->file);
                CHECK(clFinish(q), "clFinish");
                cl_ulong s0, s1;
                clGetEventProfilingInfo(e1, CL_PROFILING_COMMAND_START, sizeof s0, &s0, NULL);
                clGetEventProfilingInfo(e1, CL_PROFILING_COMMAND_END, sizeof s1, &s1, NULL);
                clReleaseEvent(e1);
                double ms = (s1 - s0) * 1e-6;
                gpu_ms += ms;
                if (r > 0 && ms < bestc) bestc = ms;
                if (r == 0 && ms > 4 * best[b] + 0.05) {
                    bestc = ms;
                    break;
                }
            }
            CHECK(clEnqueueReadBuffer(q, mem[ob], CL_TRUE, 0, n, got, 0, NULL, NULL), "cand out");
            double e = 0, sc = 0;
            for (size_t i = 0; i < n / 4; i++) {
                double d = (double)got[i] - (double)ref[i];
                if (d < 0) d = -d;
                if (d != d) d = 1e30;
                if (d > e) e = d;
                double a = ref[i] < 0 ? -ref[i] : ref[i];
                if (a > sc) sc = a;
            }
            fprintf(tu, "c %d %.6f %.3e\n", c, bestc, e / (sc > 1e-30 ? sc : 1e-30));
            clReleaseKernel(kk);
            clReleaseProgram(pr);
            free((void *)src);
            // restore the base kernel's output after its last candidate
            if (c + 1 == nc || cbase[c + 1] != b) {
                size_t local = ks[b].wg;
                size_t global = (ks[b].count + local - 1) / local * local;
                CHECK(clEnqueueNDRangeKernel(q, ks[b].kernel, 1, NULL, &global, &local, 0, NULL,
                                             NULL),
                      "restore");
                CHECK(clFinish(q), "clFinish");
            }
        }
        fclose(tu);
        printf("timed %d candidates, max temp %d mC\n", nc, max_temp());
    }

    for (int i = 0; i < nout; i++) {
        size_t n = bytes[out_buf[i]];
        char *data = malloc(n);
        CHECK(clEnqueueReadBuffer(q, mem[out_buf[i]], CL_TRUE, 0, n, data, 0, NULL, NULL),
              "read output");
        char path[4096];
        snprintf(path, sizeof path, "%s/%s", dir, out_file[i]);
        FILE *f = fopen(path, "wb");
        fwrite(data, 1, n, f);
        fclose(f);
        free(data);
    }
    return 0;
}
