// Hand-written GRU step (batch 1, obs 64, hidden 256), the sanity floor
// for the `to_c` lowering, in two loop orders:
// - `dot`: the textbook loop, a dot product per output over the column of
//   W ([in, hidden] row-major, as tang stores it): strided loads;
// - `axpy`: the same sums with k outside j, so the inner loop streams a
//   row of W (what a careful hand writer does; gcc vectorizes it).
//
//   gcc -O3 -mcpu=native -o naive_gru naive_gru.c -lm
//   ./naive_gru CASE_DIR [STEPS] [CORE]     # CASE_DIR = an exported gru_b1
//
// Inputs as cpu_export writes them: x, h, then per gate r, z, n:
// W_i, W_h, b_i, b_h. Writes CASE_DIR/out_naive.bin.
#define _GNU_SOURCE
#include <math.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define OBS 64
#define H 256

static float x[OBS], h[H], hn[H];
static float wi[3][OBS * H], wh[3][H * H], bi[3][H], bh[3][H];

static void load(const char* dir, int k, float* dst, size_t n) {
    char p[4096];
    snprintf(p, sizeof p, "%s/in%d.bin", dir, k);
    FILE* f = fopen(p, "rb");
    if (!f || fread(dst, 4, n, f) != n) { fprintf(stderr, "%s\n", p); exit(1); }
    fclose(f);
}

static float g[3][2][H];

static void gates_dot(void) {
    for (int q = 0; q < 3; q++)
        for (int j = 0; j < H; j++) {
            float si = bi[q][j], sh = bh[q][j];
            for (int k = 0; k < OBS; k++) si += x[k] * wi[q][k * H + j];
            for (int k = 0; k < H; k++) sh += h[k] * wh[q][k * H + j];
            g[q][0][j] = si;
            g[q][1][j] = sh;
        }
}

static void gates_axpy(void) {
    for (int q = 0; q < 3; q++) {
        float* restrict si = g[q][0];
        float* restrict sh = g[q][1];
        for (int j = 0; j < H; j++) si[j] = bi[q][j], sh[j] = bh[q][j];
        for (int k = 0; k < OBS; k++)
            for (int j = 0; j < H; j++) si[j] += x[k] * wi[q][k * H + j];
        for (int k = 0; k < H; k++)
            for (int j = 0; j < H; j++) sh[j] += h[k] * wh[q][k * H + j];
    }
}

static void (*gates)(void) = gates_dot;

static void step(void) {
    gates();
    for (int j = 0; j < H; j++) {
        float r = 1.0f / (1.0f + expf(-(g[0][0][j] + g[0][1][j])));
        float z = 1.0f / (1.0f + expf(-(g[1][0][j] + g[1][1][j])));
        float n = tanhf(g[2][0][j] + r * g[2][1][j]);
        hn[j] = (1.0f - z) * n + z * h[j];
    }
}

static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}

static int cmp(const void* a, const void* b) {
    double u = *(const double*)a, v = *(const double*)b;
    return (u > v) - (u < v);
}

int main(int argc, char** argv) {
    const char* dir = argv[1];
    long steps = argc > 2 ? atol(argv[2]) : 20000;
#ifdef __linux__
    if (argc > 3) {
        cpu_set_t s;
        CPU_ZERO(&s);
        CPU_SET(atoi(argv[3]), &s);
        sched_setaffinity(0, sizeof s, &s);
    }
#endif
    load(dir, 0, x, OBS);
    load(dir, 1, h, H);
    for (int q = 0; q < 3; q++) {
        load(dir, 2 + 4 * q, wi[q], OBS * H);
        load(dir, 3 + 4 * q, wh[q], H * H);
        load(dir, 4 + 4 * q, bi[q], H);
        load(dir, 5 + 4 * q, bh[q], H);
    }
    double* t = malloc(steps * sizeof *t);
    for (int mode = 0; mode < 2; mode++) {
        gates = mode ? gates_axpy : gates_dot;
        for (int i = 0; i < 500; i++) step();
        double t0 = now_ns();
        long n = 0;
        for (; n < steps; n++) {
            double a = now_ns();
            step();
            double b = now_ns();
            t[n] = b - a;
            if (b - t0 > 2e9) { n++; break; }
        }
        qsort(t, n, sizeof *t, cmp);
#define Q(p) t[(long)((p) * (n - 1))]
        printf("naive gru %s: %ld steps  min %.2f  med %.2f  p99 %.2f  p99.9 %.2f  max %.2f us\n",
               mode ? "axpy" : "dot", n, t[0] / 1e3, Q(0.5) / 1e3, Q(0.99) / 1e3, Q(0.999) / 1e3,
               t[n - 1] / 1e3);
    }
    char p[4096];
    snprintf(p, sizeof p, "%s/out_naive.bin", dir);
    FILE* f = fopen(p, "wb");
    fwrite(hn, 4, H, f);
    fclose(f);
    return 0;
}
