/* LD_PRELOAD shim: make eglQueryDevicesEXT report only Mesa's software (llvmpipe) device, so
 * that a renderer which always takes EGL device 0 (habitat-sim) runs on the CPU.  Used when
 * the GPU is unavailable.  Build: gcc -shared -fPIC -o egl_software_only.so egl_software_only.c -ldl
 * Use:   __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json \
 *        LIBGL_ALWAYS_SOFTWARE=1 LD_PRELOAD=.../egl_software_only.so python ...            */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <string.h>
#include <stdio.h>

typedef void *EGLDeviceEXT;
typedef unsigned int EGLBoolean;
typedef int EGLint;
typedef void (*fn_t)(void);
typedef EGLBoolean (*query_devices_t)(EGLint, EGLDeviceEXT *, EGLint *);
typedef const char *(*query_string_t)(EGLDeviceEXT, EGLint);
#define EGL_EXTENSIONS 0x3055

static fn_t (*real_get_proc)(const char *) = NULL;
static query_devices_t real_query = NULL;
static query_string_t real_string = NULL;

static EGLBoolean filtered_query(EGLint max, EGLDeviceEXT *out, EGLint *n) {
    EGLDeviceEXT all[16];
    EGLint total = 0, kept = 0;
    if (!real_query(16, all, &total)) return 0;
    for (EGLint i = 0; i < total; i++) {
        const char *ext = real_string ? real_string(all[i], EGL_EXTENSIONS) : NULL;
        if (ext && strstr(ext, "EGL_MESA_device_software")) {
            if (out && kept < max) out[kept] = all[i];
            kept++;
        }
    }
    *n = out ? (kept < max ? kept : max) : kept;
    return 1;
}

fn_t eglGetProcAddress(const char *name) {
    if (!real_get_proc) real_get_proc = dlsym(RTLD_NEXT, "eglGetProcAddress");
    fn_t f = real_get_proc(name);
    if (f && strcmp(name, "eglQueryDevicesEXT") == 0) {
        real_query = (query_devices_t)f;
        real_string = (query_string_t)real_get_proc("eglQueryDeviceStringEXT");
        return (fn_t)filtered_query;
    }
    return f;
}
