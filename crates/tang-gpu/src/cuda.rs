//! CUDA device for tensor-IR programs (feature `cuda`).
//!
//! The driver (`libcuda`) and nvrtc are loaded at run time by cudarc, so
//! a build with this feature runs on machines without NVIDIA hardware:
//! [`CudaDevice::new`] returns `None` there and
//! [`Runtime::select`](crate::program::Runtime::select) falls back to wgpu.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

/// The cudarc version this backend is built on, for callers that need
/// the driver directly.
pub use cudarc;
use cudarc::driver::{CudaContext, CudaFunction, CudaStream};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

/// One CUDA context and its stream, plus compiled kernels keyed by source
/// (a program lowered twice for tuning compiles each kernel once).
pub struct CudaDevice {
    pub ctx: Arc<CudaContext>,
    /// Every launch, copy and readback goes on this stream, in order.
    pub stream: Arc<CudaStream>,
    /// Device name, e.g. `"NVIDIA GeForce RTX 3090 (CUDA sm_86)"`.
    pub name: String,
    /// nvrtc target, e.g. `"sm_86"`, from the device's compute capability.
    pub arch: &'static str,
    functions: RefCell<HashMap<String, CudaFunction>>,
}

/// `#define TANG_SMEM n` in a kernel's source: its dynamic shared memory
/// in bytes (0 when absent).
pub fn dynamic_smem(src: &str) -> u32 {
    src.lines()
        .find_map(|l| l.strip_prefix("#define TANG_SMEM "))
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0)
}

impl CudaDevice {
    /// Device `ordinal`, or `None` when there is no driver or no device.
    /// Never panics: cudarc panics when `libcuda` is missing, and that is
    /// caught here.
    pub fn new(ordinal: usize) -> Option<Self> {
        let ctx = std::panic::catch_unwind(|| CudaContext::new(ordinal).ok())
            .ok()
            .flatten()?;
        use cudarc::driver::sys::CUdevice_attribute as A;
        let major = ctx
            .attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .ok()?;
        let minor = ctx
            .attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .ok()?;
        let arch: &'static str = Box::leak(format!("sm_{major}{minor}").into_boxed_str());
        let name = format!("{} (CUDA {arch})", ctx.name().ok()?);
        let stream = ctx.default_stream();
        Some(Self {
            ctx,
            stream,
            name,
            arch,
            functions: RefCell::new(HashMap::new()),
        })
    }

    /// nvrtc can be loaded too (a driver without the toolkit cannot
    /// compile programs).
    pub fn has_nvrtc() -> bool {
        std::panic::catch_unwind(|| {
            let (mut major, mut minor) = (0, 0);
            let r = unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor) };
            r == cudarc::nvrtc::sys::nvrtcResult::NVRTC_SUCCESS
        })
        .unwrap_or(false)
    }

    /// The program entry point of `source`, compiled with nvrtc (cached).
    /// Kernels that want more than 48 KB of dynamic shared memory (their
    /// `TANG_SMEM`) get the attribute set.
    pub fn function(&self, source: &str) -> CudaFunction {
        if let Some(f) = self.functions.borrow().get(source) {
            return f.clone();
        }
        let include = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
        let opts = CompileOptions {
            arch: Some(self.arch),
            include_paths: vec![format!("{include}/include")],
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(source, opts)
            .unwrap_or_else(|e| panic!("nvrtc: {e:?}\n{source}"));
        let module = self.ctx.load_module(ptx).expect("load module");
        let f = module
            .load_function(tang_compile::tensor::cuda::CUDA_ENTRY)
            .expect("entry");
        let smem = dynamic_smem(source);
        if smem > 48 * 1024 {
            f.set_attribute(
                cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                smem as i32,
            )
            .expect("smem attribute");
        }
        self.functions
            .borrow_mut()
            .insert(source.to_string(), f.clone());
        f
    }

    /// Wait for everything queued on the stream.
    pub fn sync(&self) {
        self.stream.synchronize().expect("cuda sync");
    }
}
