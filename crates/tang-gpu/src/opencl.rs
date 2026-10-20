//! OpenCL device for [`program`](crate::program) (feature `opencl`): GPUs
//! with an OpenCL driver and no CUDA/Metal, and where Vulkan dispatch is
//! expensive — the Booster K1's Adreno 740 (QCS8550), where a back-to-back
//! OpenCL launch costs tens of µs against Vulkan's ~150.
//!
//! The CUDA backend's structure, with OpenCL's runtime compiler in nvrtc's
//! place:
//!
//! - **libOpenCL is `dlopen`ed** on first use, so nothing links it. Tried in
//!   order: `$TANG_OPENCL_LIB`, `libOpenCL.so`, Qualcomm's
//!   `/opt/qti/usr/lib/libOpenCL.so.adreno`, `libOpenCL.so.1`, Apple's
//!   framework. (On the K1 `libOpenCL.so` is Qualcomm's library;
//!   `libOpenCL.so.1` is ocl-icd with no vendor file, which finds nothing.)
//! - **Device binaries are cached on disk**: each distinct kernel source is
//!   built once (`clBuildProgram`, 0.1–0.4 s on the Adreno) and its binary
//!   (`CL_PROGRAM_BINARIES`) kept under `$TANG_CL_CACHE` (default
//!   `~/.cache/tang-cl`), keyed by device, driver, options and source.
//! - **One context, one in-order queue**: launches only enqueue; reads
//!   block. Kernels carry their arguments (set once per program).
//! - **Decode kernels' warp shuffles** ([`ClDevice::shfl`]): the driver's
//!   subgroup shuffle when it has one (`cl_khr_subgroup_shuffle`,
//!   `cl_qcom_subgroup_shuffle`), else local memory; `TANG_CL_SHFL=khr|qcom|local`
//!   overrides.
//!
//! Limits seen on the Adreno 740: `CL_DEVICE_MAX_MEM_ALLOC_SIZE` 256 MiB
//! per buffer is enforced (larger allocations fail with -61), but the
//! advertised 1 GiB `CL_DEVICE_GLOBAL_MEM_SIZE` is not a total limit
//! (3 GiB of 128 MiB buffers allocate and hold data).

use std::ffi::{c_char, c_void, CString};
use std::path::PathBuf;
use std::ptr::{null, null_mut};

type H = *mut c_void;
type Int = i32;

const CL_DEVICE_TYPE_GPU: u64 = 1 << 2;
const CL_DEVICE_NAME: u32 = 0x102B;
const CL_DRIVER_VERSION: u32 = 0x102D;
const CL_DEVICE_EXTENSIONS: u32 = 0x1030;
const CL_MEM_READ_WRITE: u64 = 1;
const CL_PROGRAM_BINARY_SIZES: u32 = 0x1165;
const CL_PROGRAM_BINARIES: u32 = 0x1166;
const CL_PROGRAM_BUILD_LOG: u32 = 0x1183;
const CL_PROFILING_COMMAND_START: u32 = 0x1282;
const CL_PROFILING_COMMAND_END: u32 = 0x1283;
// A device is used from one thread at a time; kernel builds borrow it from
// a few (OpenCL's program and kernel calls are thread-safe).
unsafe impl Send for ClDevice {}
unsafe impl Sync for ClDevice {}

// cl_qcom_perf_hint / cl_qcom_priority_hint context properties
const CL_CONTEXT_PERF_HINT_QCOM: isize = 0x40C2;
const CL_PERF_HINT_HIGH_QCOM: isize = 0x40C3;
const CL_CONTEXT_PRIORITY_HINT_QCOM: isize = 0x40C9;
const CL_PRIORITY_HINT_HIGH_QCOM: isize = 0x40CA;

#[allow(non_snake_case)]
struct Api {
    _lib: libloading::Library,
    clGetPlatformIDs: unsafe extern "C" fn(u32, *mut H, *mut u32) -> Int,
    clGetDeviceIDs: unsafe extern "C" fn(H, u64, u32, *mut H, *mut u32) -> Int,
    clGetDeviceInfo: unsafe extern "C" fn(H, u32, usize, *mut c_void, *mut usize) -> Int,
    clCreateContext: unsafe extern "C" fn(
        *const isize,
        u32,
        *const H,
        *const c_void,
        *const c_void,
        *mut Int,
    ) -> H,
    clCreateCommandQueue: unsafe extern "C" fn(H, H, u64, *mut Int) -> H,
    clCreateBuffer: unsafe extern "C" fn(H, u64, usize, *mut c_void, *mut Int) -> H,
    clEnqueueWriteBuffer:
        unsafe extern "C" fn(H, H, u32, usize, usize, *const c_void, u32, *const H, *mut H) -> Int,
    clEnqueueReadBuffer:
        unsafe extern "C" fn(H, H, u32, usize, usize, *mut c_void, u32, *const H, *mut H) -> Int,
    clEnqueueCopyBuffer:
        unsafe extern "C" fn(H, H, H, usize, usize, usize, u32, *const H, *mut H) -> Int,
    clEnqueueFillBuffer: unsafe extern "C" fn(
        H,
        H,
        *const c_void,
        usize,
        usize,
        usize,
        u32,
        *const H,
        *mut H,
    ) -> Int,
    clCreateProgramWithSource:
        unsafe extern "C" fn(H, u32, *const *const c_char, *const usize, *mut Int) -> H,
    clCreateProgramWithBinary: unsafe extern "C" fn(
        H,
        u32,
        *const H,
        *const usize,
        *const *const u8,
        *mut Int,
        *mut Int,
    ) -> H,
    clBuildProgram:
        unsafe extern "C" fn(H, u32, *const H, *const c_char, *const c_void, *const c_void) -> Int,
    clGetProgramBuildInfo: unsafe extern "C" fn(H, H, u32, usize, *mut c_void, *mut usize) -> Int,
    clGetProgramInfo: unsafe extern "C" fn(H, u32, usize, *mut c_void, *mut usize) -> Int,
    clCreateKernel: unsafe extern "C" fn(H, *const c_char, *mut Int) -> H,
    clSetKernelArg: unsafe extern "C" fn(H, u32, usize, *const c_void) -> Int,
    clEnqueueNDRangeKernel: unsafe extern "C" fn(
        H,
        H,
        u32,
        *const usize,
        *const usize,
        *const usize,
        u32,
        *const H,
        *mut H,
    ) -> Int,
    clFlush: unsafe extern "C" fn(H) -> Int,
    clFinish: unsafe extern "C" fn(H) -> Int,
    clWaitForEvents: unsafe extern "C" fn(u32, *const H) -> Int,
    clGetEventProfilingInfo: unsafe extern "C" fn(H, u32, usize, *mut c_void, *mut usize) -> Int,
    clReleaseEvent: unsafe extern "C" fn(H) -> Int,
    clReleaseMemObject: unsafe extern "C" fn(H) -> Int,
    clReleaseKernel: unsafe extern "C" fn(H) -> Int,
    clReleaseProgram: unsafe extern "C" fn(H) -> Int,
}

fn check(e: Int, what: &str) {
    assert!(e == 0, "OpenCL {what}: error {e}");
}

fn load_api() -> Result<Api, String> {
    let mut names: Vec<String> = Vec::new();
    if let Ok(p) = std::env::var("TANG_OPENCL_LIB") {
        names.push(p);
    }
    names.extend(
        [
            "libOpenCL.so",
            "/opt/qti/usr/lib/libOpenCL.so.adreno",
            "libOpenCL.so.1",
            "/System/Library/Frameworks/OpenCL.framework/OpenCL",
        ]
        .map(String::from),
    );
    let mut errs = Vec::new();
    for n in &names {
        let lib = match unsafe { libloading::Library::new(n) } {
            Ok(l) => l,
            Err(e) => {
                errs.push(format!("{n}: {e}"));
                continue;
            }
        };
        macro_rules! sym {
            ($name:ident) => {
                *unsafe { lib.get(concat!(stringify!($name), "\0").as_bytes()) }
                    .map_err(|e| format!("{n}: {}: {e}", stringify!($name)))?
            };
        }
        return Ok(Api {
            clGetPlatformIDs: sym!(clGetPlatformIDs),
            clGetDeviceIDs: sym!(clGetDeviceIDs),
            clGetDeviceInfo: sym!(clGetDeviceInfo),
            clCreateContext: sym!(clCreateContext),
            clCreateCommandQueue: sym!(clCreateCommandQueue),
            clCreateBuffer: sym!(clCreateBuffer),
            clEnqueueWriteBuffer: sym!(clEnqueueWriteBuffer),
            clEnqueueReadBuffer: sym!(clEnqueueReadBuffer),
            clEnqueueCopyBuffer: sym!(clEnqueueCopyBuffer),
            clEnqueueFillBuffer: sym!(clEnqueueFillBuffer),
            clCreateProgramWithSource: sym!(clCreateProgramWithSource),
            clCreateProgramWithBinary: sym!(clCreateProgramWithBinary),
            clBuildProgram: sym!(clBuildProgram),
            clGetProgramBuildInfo: sym!(clGetProgramBuildInfo),
            clGetProgramInfo: sym!(clGetProgramInfo),
            clCreateKernel: sym!(clCreateKernel),
            clSetKernelArg: sym!(clSetKernelArg),
            clEnqueueNDRangeKernel: sym!(clEnqueueNDRangeKernel),
            clFlush: sym!(clFlush),
            clFinish: sym!(clFinish),
            clWaitForEvents: sym!(clWaitForEvents),
            clGetEventProfilingInfo: sym!(clGetEventProfilingInfo),
            clReleaseEvent: sym!(clReleaseEvent),
            clReleaseMemObject: sym!(clReleaseMemObject),
            clReleaseKernel: sym!(clReleaseKernel),
            clReleaseProgram: sym!(clReleaseProgram),
            _lib: lib,
        });
    }
    Err(format!("no OpenCL library ({})", errs.join("; ")))
}

/// How decode kernels' `TANG_SHFL_XOR` is defined on this device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shfl {
    Khr,
    Qcom,
    Local,
}

/// An OpenCL GPU: one context and one in-order (profiling) queue.
pub struct ClDevice {
    api: Api,
    device: H,
    context: H,
    queue: H,
    pub name: String,
    pub driver: String,
    pub extensions: String,
    pub shfl: Shfl,
}

/// A device buffer (f32 words), released on drop.
pub struct ClMem {
    mem: H,
    pub words: usize,
    release: unsafe extern "C" fn(H) -> Int,
}

impl Drop for ClMem {
    fn drop(&mut self) {
        unsafe { (self.release)(self.mem) };
    }
}

/// One kernel object per launch site (its arguments are its own) and the
/// built programs behind them, released together.
pub struct ClKernels {
    kernels: Vec<H>,
    programs: Vec<H>,
    release_k: unsafe extern "C" fn(H) -> Int,
    release_p: unsafe extern "C" fn(H) -> Int,
}

impl ClKernels {
    pub fn len(&self) -> usize {
        self.kernels.len()
    }
    pub fn is_empty(&self) -> bool {
        self.kernels.is_empty()
    }
}

impl Drop for ClKernels {
    fn drop(&mut self) {
        unsafe {
            for &k in &self.kernels {
                (self.release_k)(k);
            }
            for &p in &self.programs {
                (self.release_p)(p);
            }
        }
    }
}

impl ClDevice {
    /// The first GPU of the first platform that has one.
    pub fn new() -> Option<Self> {
        match Self::try_new() {
            Ok(d) => Some(d),
            Err(e) => {
                if std::env::var("TANG_BACKEND").as_deref() == Ok("opencl") {
                    eprintln!("tang-gpu: {e}");
                }
                None
            }
        }
    }

    fn try_new() -> Result<Self, String> {
        let api = load_api()?;
        unsafe {
            let mut plats = [null_mut(); 8];
            let mut np = 0u32;
            if (api.clGetPlatformIDs)(8, plats.as_mut_ptr(), &mut np) != 0 {
                return Err("clGetPlatformIDs failed".into());
            }
            let mut device = null_mut();
            for &p in &plats[..np as usize] {
                if (api.clGetDeviceIDs)(p, CL_DEVICE_TYPE_GPU, 1, &mut device, null_mut()) == 0 {
                    break;
                }
            }
            if device.is_null() {
                return Err(format!("OpenCL: no GPU on {np} platform(s)"));
            }
            let info = |param: u32| {
                let mut b = vec![0u8; 1 << 14];
                let mut n = 0usize;
                (api.clGetDeviceInfo)(
                    device,
                    param,
                    b.len(),
                    b.as_mut_ptr() as *mut c_void,
                    &mut n,
                );
                b.truncate(n.saturating_sub(1));
                String::from_utf8_lossy(&b).into_owned()
            };
            let (name, driver, extensions) = (
                info(CL_DEVICE_NAME),
                info(CL_DRIVER_VERSION),
                info(CL_DEVICE_EXTENSIONS),
            );
            let mut err = 0;
            let hints = name.contains("Adreno")
                && std::env::var("TANG_CL_HINTS").map_or(true, |v| v != "0");
            let props = [
                CL_CONTEXT_PERF_HINT_QCOM,
                CL_PERF_HINT_HIGH_QCOM,
                CL_CONTEXT_PRIORITY_HINT_QCOM,
                CL_PRIORITY_HINT_HIGH_QCOM,
                0,
            ];
            let mut context = if hints {
                (api.clCreateContext)(props.as_ptr(), 1, &device, null(), null(), &mut err)
            } else {
                null_mut()
            };
            if context.is_null() {
                context = (api.clCreateContext)(null(), 1, &device, null(), null(), &mut err);
            }
            if err != 0 {
                return Err(format!("clCreateContext: {err}"));
            }
            // CL_QUEUE_PROFILING_ENABLE: event times for time_kernel
            let queue = (api.clCreateCommandQueue)(context, device, 2, &mut err);
            if err != 0 {
                return Err(format!("clCreateCommandQueue: {err}"));
            }
            let shfl = match std::env::var("TANG_CL_SHFL").as_deref() {
                Ok("khr") => Shfl::Khr,
                Ok("qcom") => Shfl::Qcom,
                Ok("local") => Shfl::Local,
                _ if extensions.contains("cl_khr_subgroup_shuffle") => Shfl::Khr,
                _ if extensions.contains("cl_qcom_subgroup_shuffle") => Shfl::Qcom,
                _ => Shfl::Local,
            };
            Ok(Self {
                api,
                device,
                context,
                queue,
                name,
                driver,
                extensions,
                shfl,
            })
        }
    }

    /// `source` with the device's `TANG_SHFL_XOR` in place of a decode
    /// kernel's marker (other sources unchanged).
    pub fn prepare(&self, source: &str) -> String {
        use tang_compile::tensor::opencl_gemv::{
            OPENCL_SHFL_KHR, OPENCL_SHFL_LOCAL, OPENCL_SHFL_QCOM, SHFL_MARKER,
        };
        match source.strip_prefix(SHFL_MARKER) {
            None => source.to_string(),
            Some(rest) => {
                let def = match self.shfl {
                    Shfl::Khr => OPENCL_SHFL_KHR,
                    Shfl::Qcom => OPENCL_SHFL_QCOM,
                    Shfl::Local => OPENCL_SHFL_LOCAL,
                };
                format!("{def}{rest}")
            }
        }
    }

    fn build_log(&self, p: H) -> String {
        let mut b = vec![0u8; 1 << 16];
        let mut n = 0usize;
        unsafe {
            (self.api.clGetProgramBuildInfo)(
                p,
                self.device,
                CL_PROGRAM_BUILD_LOG,
                b.len(),
                b.as_mut_ptr() as *mut c_void,
                &mut n,
            )
        };
        b.truncate(n.saturating_sub(1));
        String::from_utf8_lossy(&b).into_owned()
    }

    /// A built program for `source` (prepared): from the binary cache,
    /// else compiled and cached.
    fn program(&self, source: &str) -> Result<H, String> {
        use std::hash::{Hash, Hasher};
        let opts = std::env::var("TANG_CL_OPTS").unwrap_or_default();
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (&self.name, &self.driver, &opts, source).hash(&mut h);
        let dir = cache_dir();
        let file = dir.join(format!("{:016x}.bin", h.finish()));
        let copts = CString::new(opts).unwrap();
        let mut err = 0;
        if let Ok(bin) = std::fs::read(&file) {
            let (len, ptr) = (bin.len(), bin.as_ptr());
            let mut status = 0;
            let p = unsafe {
                (self.api.clCreateProgramWithBinary)(
                    self.context,
                    1,
                    &self.device,
                    &len,
                    &ptr,
                    &mut status,
                    &mut err,
                )
            };
            if err == 0
                && status == 0
                && unsafe {
                    (self.api.clBuildProgram)(p, 1, &self.device, copts.as_ptr(), null(), null())
                } == 0
            {
                return Ok(p);
            }
            if !p.is_null() {
                unsafe { (self.api.clReleaseProgram)(p) };
            }
            let _ = std::fs::remove_file(&file);
        }
        let src = CString::new(source).map_err(|e| e.to_string())?;
        let sp = src.as_ptr();
        let p =
            unsafe { (self.api.clCreateProgramWithSource)(self.context, 1, &sp, null(), &mut err) };
        if err != 0 {
            return Err(format!("clCreateProgramWithSource: {err}"));
        }
        if unsafe { (self.api.clBuildProgram)(p, 1, &self.device, copts.as_ptr(), null(), null()) }
            != 0
        {
            let log = self.build_log(p);
            unsafe { (self.api.clReleaseProgram)(p) };
            return Err(format!(
                "clBuildProgram failed:\n{log}\n--- source ---\n{source}"
            ));
        }
        let mut len = 0usize;
        unsafe {
            (self.api.clGetProgramInfo)(
                p,
                CL_PROGRAM_BINARY_SIZES,
                std::mem::size_of::<usize>(),
                &mut len as *mut usize as *mut c_void,
                null_mut(),
            )
        };
        if len > 0 && std::fs::create_dir_all(&dir).is_ok() {
            let mut bin = vec![0u8; len];
            let mut bp = bin.as_mut_ptr();
            if unsafe {
                (self.api.clGetProgramInfo)(
                    p,
                    CL_PROGRAM_BINARIES,
                    std::mem::size_of::<*mut u8>(),
                    &mut bp as *mut *mut u8 as *mut c_void,
                    null_mut(),
                )
            } == 0
            {
                let tmp = file.with_extension(format!("{}", std::process::id()));
                if std::fs::write(&tmp, &bin).is_ok() {
                    let _ = std::fs::rename(&tmp, &file);
                }
            }
        }
        Ok(p)
    }

    /// One kernel object per source (entry `main0`), the distinct sources
    /// built on a few threads (`TANG_CL_BUILD_THREADS`, default 3).
    pub fn kernels<'a>(&self, sources: impl Iterator<Item = &'a str>) -> ClKernels {
        let prepared: Vec<String> = sources.map(|s| self.prepare(s)).collect();
        let mut uniq: Vec<&str> = prepared.iter().map(String::as_str).collect();
        uniq.sort_unstable();
        uniq.dedup();
        struct Send2(H);
        unsafe impl Send for Send2 {}
        let n = std::env::var("TANG_CL_BUILD_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3usize)
            .max(1);
        let chunk = uniq.len().div_ceil(n).max(1);
        let built: Vec<Send2> = std::thread::scope(|s| {
            let hs: Vec<_> = uniq
                .chunks(chunk)
                .map(|c| {
                    s.spawn(move || {
                        c.iter()
                            .map(|src| Send2(self.program(src).unwrap_or_else(|e| panic!("{e}"))))
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });
        let programs: Vec<H> = built.into_iter().map(|b| b.0).collect();
        let entry = CString::new("main0").unwrap();
        let kernels = prepared
            .iter()
            .map(|src| {
                let p = programs[uniq.binary_search(&src.as_str()).unwrap()];
                let mut err = 0;
                let k = unsafe { (self.api.clCreateKernel)(p, entry.as_ptr(), &mut err) };
                check(err, "clCreateKernel");
                k
            })
            .collect();
        ClKernels {
            kernels,
            programs,
            release_k: self.api.clReleaseKernel,
            release_p: self.api.clReleaseProgram,
        }
    }

    /// A zeroed buffer of `words` f32 words.
    pub fn alloc_zeros(&self, words: usize) -> ClMem {
        let bytes = words.max(1) * 4;
        let mut err = 0;
        let mem = unsafe {
            (self.api.clCreateBuffer)(self.context, CL_MEM_READ_WRITE, bytes, null_mut(), &mut err)
        };
        check(err, &format!("clCreateBuffer {bytes} B"));
        let z = 0f32;
        check(
            unsafe {
                (self.api.clEnqueueFillBuffer)(
                    self.queue,
                    mem,
                    &z as *const f32 as *const c_void,
                    4,
                    0,
                    bytes,
                    0,
                    null(),
                    null_mut(),
                )
            },
            "fill",
        );
        ClMem {
            mem,
            words: words.max(1),
            release: self.api.clReleaseMemObject,
        }
    }

    /// Host → device, blocking (`data` is borrowed only for the call).
    pub fn write(&self, m: &ClMem, data: &[f32]) {
        assert!(data.len() <= m.words);
        check(
            unsafe {
                (self.api.clEnqueueWriteBuffer)(
                    self.queue,
                    m.mem,
                    1,
                    0,
                    data.len() * 4,
                    data.as_ptr() as *const c_void,
                    0,
                    null(),
                    null_mut(),
                )
            },
            "write",
        );
    }

    /// Device → host, the first `n` words (waits for queued work).
    pub fn read(&self, m: &ClMem, n: usize) -> Vec<f32> {
        let mut v = vec![0f32; n];
        check(
            unsafe {
                (self.api.clEnqueueReadBuffer)(
                    self.queue,
                    m.mem,
                    1,
                    0,
                    n * 4,
                    v.as_mut_ptr() as *mut c_void,
                    0,
                    null(),
                    null_mut(),
                )
            },
            "read",
        );
        v
    }

    /// Device-side copy of `words` words, queued in order.
    pub fn copy(&self, dst: &ClMem, src: &ClMem, words: usize) {
        check(
            unsafe {
                (self.api.clEnqueueCopyBuffer)(
                    self.queue,
                    src.mem,
                    dst.mem,
                    0,
                    0,
                    words * 4,
                    0,
                    null(),
                    null_mut(),
                )
            },
            "copy",
        );
    }

    /// Bind `bufs` then the `u32` `count` as `k`'s arguments.
    pub fn set_args(&self, ks: &ClKernels, i: usize, bufs: &[&ClMem], count: u32) {
        let k = ks.kernels[i];
        for (a, b) in bufs.iter().enumerate() {
            check(
                unsafe {
                    (self.api.clSetKernelArg)(
                        k,
                        a as u32,
                        std::mem::size_of::<H>(),
                        &b.mem as *const H as *const c_void,
                    )
                },
                "arg",
            );
        }
        check(
            unsafe {
                (self.api.clSetKernelArg)(
                    k,
                    bufs.len() as u32,
                    4,
                    &count as *const u32 as *const c_void,
                )
            },
            "arg count",
        );
    }

    /// Enqueue `k` over `items` work-items in groups of `group`; with
    /// `timed`, wait and return its device milliseconds.
    pub fn launch(&self, ks: &ClKernels, i: usize, items: usize, group: usize, timed: bool) -> f64 {
        let global = items.div_ceil(group) * group;
        let mut ev = null_mut();
        check(
            unsafe {
                (self.api.clEnqueueNDRangeKernel)(
                    self.queue,
                    ks.kernels[i],
                    1,
                    null(),
                    &global,
                    &group,
                    0,
                    null(),
                    if timed { &mut ev } else { null_mut() },
                )
            },
            "enqueue",
        );
        if !timed {
            return 0.0;
        }
        unsafe {
            (self.api.clWaitForEvents)(1, &ev);
            let q = |w: u32| {
                let mut v = 0u64;
                (self.api.clGetEventProfilingInfo)(
                    ev,
                    w,
                    8,
                    &mut v as *mut u64 as *mut c_void,
                    null_mut(),
                );
                v
            };
            let ms = q(CL_PROFILING_COMMAND_END).saturating_sub(q(CL_PROFILING_COMMAND_START))
                as f64
                * 1e-6;
            (self.api.clReleaseEvent)(ev);
            ms
        }
    }

    pub fn flush(&self) {
        unsafe { (self.api.clFlush)(self.queue) };
    }

    pub fn sync(&self) {
        unsafe { (self.api.clFinish)(self.queue) };
    }
}

fn cache_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("TANG_CL_CACHE") {
        return PathBuf::from(d);
    }
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    home.join(".cache/tang-cl")
}
