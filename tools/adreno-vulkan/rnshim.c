// Vulkan on Adreno boards whose kernel exposes the GPU only as kgsl (the
// Booster K1's QCS8550, Ubuntu 22.04): Qualcomm's libvulkan_adreno.so opens
// /dev/dri/renderD128 inside vkEnumerateInstanceVersion, fails, and
// vkCreateInstance returns -1. The render node is not needed for compute:
// this LD_PRELOAD shim answers that one open with RNSHIM_TARGET (default
// /dev/null) and everything else passes through. Only the process it is
// preloaded into is affected.
//
//   gcc -O2 -shared -fPIC rnshim.c -o rnshim.so -ldl
//   LD_PRELOAD=./rnshim.so NODEVICE_SELECT=1 \
//     VK_ICD_FILENAMES=/usr/local/share/vulkan/icd.d/adrenovk.json \
//     TANG_BACKEND=wgpu ./qwen_decode tiny
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

static int (*real_open64)(const char *, int, ...);

int open64(const char *path, int flags, ...) {
    if (!real_open64) real_open64 = dlsym(RTLD_NEXT, "open64");
    mode_t mode = 0;
    if (flags & O_CREAT) {
        va_list ap;
        va_start(ap, flags);
        mode = va_arg(ap, int);
        va_end(ap);
    }
    if (path && strcmp(path, "/dev/dri/renderD128") == 0) {
        const char *t = getenv("RNSHIM_TARGET");
        if (!t) t = "/dev/null";
        int fd = real_open64(t, O_RDWR | O_CLOEXEC, 0);
        return fd;
    }
    return real_open64(path, flags, mode);
}

int open(const char *path, int flags, ...) {
    mode_t mode = 0;
    if (flags & O_CREAT) {
        va_list ap;
        va_start(ap, flags);
        mode = va_arg(ap, int);
        va_end(ap);
    }
    return open64(path, flags, mode);
}
