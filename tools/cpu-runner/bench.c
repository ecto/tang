// Per-step latency of a `to_c` program (crates/tang-bench/examples/cpu_export.rs).
//
//   gcc -O3 -mcpu=native -ffp-contract=fast -fno-strict-aliasing \
//       -DMODEL='"CASE/model.c"' -o bench bench.c -lm
//   ./bench CASE_DIR [STEPS] [CORE]
//
// Reads CASE_DIR/in{k}.bin (f32; converted to each input's storage format),
// pins to CORE (sched_setaffinity, Linux), tries SCHED_FIFO (works only
// with the privilege; reported, never required), runs 500 warm-up steps (at most 0.5 s),
// then up to STEPS back-to-back single-threaded steps capped at 4 s, and
// prints min / median / p99 / p99.9 / max per-step latency (CLOCK_MONOTONIC
// around each tang_run). Writes CASE_DIR/out{i}.bin (f32) from the last step
// (inputs are fixed, so every step's output is the same).
#define _GNU_SOURCE
#include <errno.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include MODEL

static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}

static int cmp(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

static uint16_t f2h(float f) {
    tang_f16 h = (tang_f16)f;
    uint16_t u;
    memcpy(&u, &h, 2);
    return u;
}

static float h2f(uint16_t u) {
    tang_f16 h;
    memcpy(&h, &u, 2);
    return (float)h;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s CASE_DIR [STEPS] [CORE]\n", argv[0]);
        return 2;
    }
    const char* dir = argv[1];
    long steps = argc > 2 ? atol(argv[2]) : 20000;
    int core = argc > 3 ? atoi(argv[3]) : -1;
    char path[4096];
#ifdef __linux__
    if (core >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(core, &set);
        if (sched_setaffinity(0, sizeof set, &set) != 0) perror("sched_setaffinity");
    }
    struct sched_param sp = {.sched_priority = 10};
    int fifo = sched_setscheduler(0, SCHED_FIFO, &sp) == 0;
    printf("pinned core %d, SCHED_FIFO %s\n", core, fifo ? "on" : strerror(errno));
#else
    (void)core;
#endif
    for (int k = 0; k < tang_n_inputs(); k++) {
        snprintf(path, sizeof path, "%s/in%d.bin", dir, k);
        FILE* f = fopen(path, "rb");
        if (!f) { perror(path); return 1; }
        size_t n = tang_input_len(k);
        float* tmp = malloc(n * 4);
        if (fread(tmp, 4, n, f) != n) { fprintf(stderr, "%s: short\n", path); return 1; }
        fclose(f);
        void* dst = tang_input(k);
        switch (tang_input_format(k)) {
        case 0: memcpy(dst, tmp, n * 4); break;
        case 1: for (size_t i = 0; i < n; i++) ((uint16_t*)dst)[i] = f2h(tmp[i]); break;
        default: for (size_t i = 0; i < n; i++) ((uint16_t*)dst)[i] = tang_f2b(tmp[i]); break;
        }
        free(tmp);
    }
    // warm-up: 500 steps or 0.5 s, whichever is first
    double w0 = now_ns();
    int warm = 0;
    for (; warm < 500 && now_ns() - w0 < 5e8; warm++) tang_run();
    double* t = malloc(steps * sizeof *t);
    double t0 = now_ns();
    long n = 0;
    for (; n < steps; n++) {
        double a = now_ns();
        tang_run();
        double b = now_ns();
        t[n] = b - a;
        if (b - t0 > 4e9) { n++; break; }
    }
    double total = now_ns() - t0;
    qsort(t, n, sizeof *t, cmp);
    double sum = 0;
    for (long i = 0; i < n; i++) sum += t[i];
#define Q(p) t[(long)((p) * (n - 1))]
    printf("%s: %ld steps  min %.2f  med %.2f  p99 %.2f  p99.9 %.2f  max %.2f  mean %.2f us  (%.2f s)\n",
           dir, n, t[0] / 1e3, Q(0.5) / 1e3, Q(0.99) / 1e3, Q(0.999) / 1e3, t[n - 1] / 1e3,
           sum / n / 1e3, total / 1e9);
#ifdef TANG_PROFILE
    {
        double all = 0;
        for (int k = 0; k < tang_n_kernels(); k++) all += tang_prof_ns[k];
        for (int k = 0; k < tang_n_kernels(); k++)
            printf("  kernel %2d: %8.2f us/step (%4.1f%%)\n", k, tang_prof_ns[k] / (n + warm) / 1e3,
                   100 * tang_prof_ns[k] / all);
    }
#endif
    for (int i = 0; i < tang_n_outputs(); i++) {
        snprintf(path, sizeof path, "%s/out%d.bin", dir, i);
        FILE* f = fopen(path, "wb");
        size_t len = tang_output_len(i);
        for (size_t j = 0; j < len; j++) {
            float v;
            switch (tang_output_format(i)) {
            case 0: v = ((float*)tang_output(i))[j]; break;
            case 1: v = h2f(((uint16_t*)tang_output(i))[j]); break;
            default: v = tang_b2f(((uint16_t*)tang_output(i))[j]); break;
            }
            fwrite(&v, 4, 1, f);
        }
        fclose(f);
    }
    free(t);
    return 0;
}
