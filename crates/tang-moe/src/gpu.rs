//! A thin layer over the CUDA driver API (through cudarc's `sys`), for what tang-compute's
//! single-stream device doesn't expose: several streams, events, raw device addresses, mapped
//! host memory and graph capture.
//!
//! Everything here assumes the context is current on the calling thread
//! ([`Gpu::bind`]). Handles free themselves on drop.

use std::ffi::{c_void, CString};
use std::sync::Arc;

use cudarc::driver::{sys, CudaContext};

#[derive(Clone, Debug)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<cudarc::driver::DriverError> for Error {
    fn from(e: cudarc::driver::DriverError) -> Self {
        Error(format!("{e:?}"))
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// Turn a `CUresult` into a `Result`, naming the call.
pub fn check(r: sys::CUresult, what: &str) -> Result<()> {
    if r == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(Error(format!("{what}: {r:?}")))
    }
}

macro_rules! ck {
    ($e:expr) => {
        $crate::gpu::check(unsafe { $e }, stringify!($e))
    };
}
pub(crate) use ck;

pub struct Gpu {
    pub ctx: Arc<CudaContext>,
}

impl Gpu {
    /// Device 0's primary context. Errors (instead of panicking) when there is no driver.
    pub fn new() -> Result<Self> {
        let ctx = std::panic::catch_unwind(|| CudaContext::new(0))
            .map_err(|_| Error("no CUDA driver".into()))??;
        Ok(Gpu { ctx })
    }

    /// For tests: the GPU, or `None` with a note (panics if `TANG_REQUIRE_CUDA=1`).
    pub fn for_test() -> Option<Self> {
        match Gpu::new() {
            Ok(g) => Some(g),
            Err(e) if std::env::var("TANG_REQUIRE_CUDA").is_ok_and(|v| v == "1") => {
                panic!("TANG_REQUIRE_CUDA=1 but no CUDA device: {e}")
            }
            Err(e) => {
                eprintln!("no CUDA device ({e}): skipping");
                None
            }
        }
    }

    pub fn bind(&self) -> Result<()> {
        Ok(self.ctx.bind_to_thread()?)
    }

    /// `(free, total)` device memory in bytes.
    pub fn mem_info(&self) -> Result<(usize, usize)> {
        let (mut free, mut total) = (0usize, 0usize);
        ck!(sys::cuMemGetInfo_v2(&mut free, &mut total))?;
        Ok((free, total))
    }

    pub fn attribute(&self, a: sys::CUdevice_attribute) -> Result<i32> {
        Ok(self.ctx.attribute(a)?)
    }

    /// Compile CUDA C with NVRTC for this device's architecture and load it.
    pub fn module(&self, src: &str) -> Result<Module> {
        let major =
            self.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let minor =
            self.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        let arch: &'static str = Box::leak(format!("sm_{major}{minor}").into_boxed_str());
        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some(arch),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(src, opts)
            .map_err(|e| Error(format!("nvrtc: {e:?}")))?;
        let src = CString::new(ptx.to_src()).unwrap();
        let mut m: sys::CUmodule = std::ptr::null_mut();
        ck!(sys::cuModuleLoadData(&mut m, src.as_ptr() as *const c_void))?;
        Ok(Module(m))
    }
}

pub struct Module(sys::CUmodule);

impl Module {
    pub fn func(&self, name: &str) -> Result<sys::CUfunction> {
        let c = CString::new(name).unwrap();
        let mut f: sys::CUfunction = std::ptr::null_mut();
        ck!(sys::cuModuleGetFunction(&mut f, self.0, c.as_ptr()))?;
        Ok(f)
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        unsafe {
            sys::cuModuleUnload(self.0);
        }
    }
}

pub struct Stream(pub sys::CUstream);

impl Stream {
    /// A non-blocking stream (doesn't synchronise with the legacy default stream; capturable).
    pub fn new() -> Result<Self> {
        let mut s: sys::CUstream = std::ptr::null_mut();
        ck!(sys::cuStreamCreate(
            &mut s,
            sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32
        ))?;
        Ok(Stream(s))
    }

    pub fn sync(&self) -> Result<()> {
        ck!(sys::cuStreamSynchronize(self.0))
    }

    /// True when all work is done (`cuStreamQuery`).
    pub fn idle(&self) -> bool {
        unsafe { sys::cuStreamQuery(self.0) == sys::CUresult::CUDA_SUCCESS }
    }

    pub fn wait(&self, e: &Event) -> Result<()> {
        ck!(sys::cuStreamWaitEvent(self.0, e.0, 0))
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe {
            sys::cuStreamDestroy_v2(self.0);
        }
    }
}

pub struct Event(pub sys::CUevent);

impl Event {
    pub fn new(timing: bool) -> Result<Self> {
        let flags = if timing {
            sys::CUevent_flags::CU_EVENT_DEFAULT
        } else {
            sys::CUevent_flags::CU_EVENT_DISABLE_TIMING
        };
        let mut e: sys::CUevent = std::ptr::null_mut();
        ck!(sys::cuEventCreate(&mut e, flags as u32))?;
        Ok(Event(e))
    }

    pub fn record(&self, s: &Stream) -> Result<()> {
        ck!(sys::cuEventRecord(self.0, s.0))
    }

    pub fn sync(&self) -> Result<()> {
        ck!(sys::cuEventSynchronize(self.0))
    }

    pub fn done(&self) -> bool {
        unsafe { sys::cuEventQuery(self.0) == sys::CUresult::CUDA_SUCCESS }
    }

    /// Milliseconds from `start` to this event.
    pub fn since(&self, start: &Event) -> Result<f32> {
        let mut ms = 0f32;
        ck!(sys::cuEventElapsedTime(&mut ms, start.0, self.0))?;
        Ok(ms)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        unsafe {
            sys::cuEventDestroy_v2(self.0);
        }
    }
}

/// A raw device allocation.
pub struct DevBuf {
    pub ptr: u64,
    pub len: usize,
}

impl DevBuf {
    pub fn alloc(len: usize) -> Result<Self> {
        let mut p: sys::CUdeviceptr = 0;
        ck!(sys::cuMemAlloc_v2(&mut p, len.max(1)))?;
        Ok(DevBuf { ptr: p, len })
    }

    pub fn zeroed(len: usize) -> Result<Self> {
        let b = Self::alloc(len)?;
        ck!(sys::cuMemsetD8_v2(b.ptr, 0, len.max(1)))?;
        Ok(b)
    }

    pub fn from_slice<T: Copy>(data: &[T]) -> Result<Self> {
        let b = Self::alloc(std::mem::size_of_val(data))?;
        b.write(0, data)?;
        Ok(b)
    }

    /// Synchronous copy of `data` to byte offset `off`.
    pub fn write<T: Copy>(&self, off: usize, data: &[T]) -> Result<()> {
        let n = std::mem::size_of_val(data);
        assert!(off + n <= self.len);
        if n == 0 {
            return Ok(());
        }
        ck!(sys::cuMemcpyHtoD_v2(
            self.ptr + off as u64,
            data.as_ptr() as *const c_void,
            n
        ))
    }

    /// Async copy from pageable `data` (the driver stages it before returning).
    pub fn write_async<T: Copy>(&self, off: usize, data: &[T], s: &Stream) -> Result<()> {
        let n = std::mem::size_of_val(data);
        assert!(off + n <= self.len);
        if n == 0 {
            return Ok(());
        }
        ck!(sys::cuMemcpyHtoDAsync_v2(
            self.ptr + off as u64,
            data.as_ptr() as *const c_void,
            n,
            s.0
        ))
    }

    pub fn read<T: Copy + Default>(&self, off: usize, n: usize) -> Result<Vec<T>> {
        let mut v = vec![T::default(); n];
        let bytes = n * std::mem::size_of::<T>();
        assert!(off + bytes <= self.len);
        if bytes > 0 {
            ck!(sys::cuMemcpyDtoH_v2(
                v.as_mut_ptr() as *mut c_void,
                self.ptr + off as u64,
                bytes
            ))?;
        }
        Ok(v)
    }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        unsafe {
            sys::cuMemFree_v2(self.ptr);
        }
    }
}

/// Launch `f`. `args` are pointers to each kernel argument's value.
///
/// # Safety
/// `args` must match the kernel's parameter list in count and types, and every device
/// address passed must be valid for what the kernel does with it.
pub unsafe fn launch(
    f: sys::CUfunction,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    smem: u32,
    s: &Stream,
    args: &mut [*mut c_void],
) -> Result<()> {
    check(
        sys::cuLaunchKernel(
            f,
            grid.0,
            grid.1,
            grid.2,
            block.0,
            block.1,
            block.2,
            smem,
            s.0,
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        ),
        "cuLaunchKernel",
    )
}

/// Kernel arguments: `args![a, b, c]` gives the `&mut [*mut c_void]` that [`launch`] takes.
/// The values must outlive the launch call (they are copied at launch).
#[macro_export]
macro_rules! args {
    ($($a:expr),* $(,)?) => {
        &mut [$(&$a as *const _ as *mut ::std::ffi::c_void),*]
    };
}

/// An instantiated CUDA graph.
pub struct Graph {
    graph: sys::CUgraph,
    exec: sys::CUgraphExec,
}

impl Graph {
    /// Record everything `body` enqueues on `s` (which must not be the legacy default stream)
    /// and instantiate it.
    pub fn capture(s: &Stream, body: impl FnOnce(&Stream) -> Result<()>) -> Result<Self> {
        ck!(sys::cuStreamBeginCapture_v2(
            s.0,
            sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL
        ))?;
        let body_result = body(s);
        let mut graph: sys::CUgraph = std::ptr::null_mut();
        let end = ck!(sys::cuStreamEndCapture(s.0, &mut graph));
        body_result?;
        end?;
        let mut exec: sys::CUgraphExec = std::ptr::null_mut();
        ck!(sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0))?;
        ck!(sys::cuGraphUpload(exec, s.0))?;
        Ok(Graph { graph, exec })
    }

    pub fn launch(&self, s: &Stream) -> Result<()> {
        ck!(sys::cuGraphLaunch(self.exec, s.0))
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe {
            sys::cuGraphExecDestroy(self.exec);
            sys::cuGraphDestroy(self.graph);
        }
    }
}
