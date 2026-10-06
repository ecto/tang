//! Pinned, device-mapped host memory for expert blobs.
//!
//! [`HostArena::new`] maps anonymous memory, trying `MAP_HUGETLB` first and falling back to
//! ordinary pages with `madvise(MADV_HUGEPAGE)` (transparent huge pages). Nothing is touched:
//! with the `cuda` feature, [`HostArena::register`] pins it with
//! `cuMemHostRegister(PORTABLE | DEVICEMAP)` *before first touch*, so the driver's own faulting
//! picks up huge pages. If one registration of the whole range is refused it falls back to
//! several smaller ones. Each registered region has a device-visible address
//! (`cuMemHostGetDevicePointer`), so kernels read the arena directly over PCIe.

use std::io;

/// How the arena's pages are backed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backing {
    /// `MAP_HUGETLB`: 2 MiB pages from the reserved pool (needs `vm.nr_hugepages`).
    HugeTlb,
    /// Ordinary mapping with `MADV_HUGEPAGE`; the kernel uses 2 MiB pages where it can.
    Thp,
    /// Ordinary 4 KiB pages (THP refused or not asked for).
    Small,
}

#[derive(Clone, Copy, Debug)]
pub struct ArenaOptions {
    pub try_hugetlb: bool,
    pub thp: bool,
}

impl Default for ArenaOptions {
    fn default() -> Self {
        ArenaOptions {
            try_hugetlb: true,
            thp: true,
        }
    }
}

/// One registered range of the arena and its device-visible address.
#[derive(Clone, Copy, Debug)]
pub struct Region {
    pub offset: usize,
    pub len: usize,
    pub device: u64,
}

pub struct HostArena {
    ptr: *mut u8,
    len: usize,
    backing: Backing,
    regions: Vec<Region>,
    #[cfg(feature = "cuda")]
    ctx: Option<std::sync::Arc<cudarc::driver::CudaContext>>,
}

// The arena is plain memory; callers coordinate access to its bytes.
unsafe impl Send for HostArena {}
unsafe impl Sync for HostArena {}

const HUGE: usize = 2 << 20;

impl HostArena {
    /// Map `len` bytes (rounded up to 2 MiB), untouched.
    pub fn new(len: usize, opts: ArenaOptions) -> io::Result<Self> {
        let len = len.div_ceil(HUGE) * HUGE;
        let prot = libc::PROT_READ | libc::PROT_WRITE;
        let base = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS;
        #[cfg(target_os = "linux")]
        if opts.try_hugetlb {
            let p = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    prot,
                    base | libc::MAP_HUGETLB,
                    -1,
                    0,
                )
            };
            if p != libc::MAP_FAILED {
                return Ok(Self::from_raw(p as *mut u8, len, Backing::HugeTlb));
            }
        }
        let p = unsafe { libc::mmap(std::ptr::null_mut(), len, prot, base, -1, 0) };
        if p == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        #[allow(unused_mut)]
        let mut backing = Backing::Small;
        #[cfg(target_os = "linux")]
        if opts.thp && unsafe { libc::madvise(p, len, libc::MADV_HUGEPAGE) } == 0 {
            backing = Backing::Thp;
        }
        let _ = opts;
        Ok(Self::from_raw(p as *mut u8, len, backing))
    }

    fn from_raw(ptr: *mut u8, len: usize, backing: Backing) -> Self {
        HostArena {
            ptr,
            len,
            backing,
            regions: Vec::new(),
            #[cfg(feature = "cuda")]
            ctx: None,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn backing(&self) -> Backing {
        self.backing
    }

    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// The whole arena as bytes. Touching pages before [`register`](Self::register) defeats
    /// the point of registering first.
    ///
    /// # Safety
    /// The caller must not create overlapping mutable views, or write bytes a GPU kernel or
    /// copy is reading.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn bytes_mut(&self) -> &mut [u8] {
        std::slice::from_raw_parts_mut(self.ptr, self.len)
    }

    pub fn regions(&self) -> &[Region] {
        &self.regions
    }

    pub fn is_registered(&self) -> bool {
        !self.regions.is_empty()
    }

    /// Device-visible address of byte `offset`, if that byte is in a registered region.
    pub fn device_ptr(&self, offset: usize) -> Option<u64> {
        let i = self.regions.partition_point(|r| r.offset + r.len <= offset);
        let r = self.regions.get(i)?;
        (offset >= r.offset).then(|| r.device + (offset - r.offset) as u64)
    }

    /// Bytes of this mapping backed by huge pages, from `/proc/self/smaps` (Linux only).
    pub fn huge_bytes(&self) -> Option<usize> {
        #[cfg(target_os = "linux")]
        {
            let smaps = std::fs::read_to_string("/proc/self/smaps").ok()?;
            let (lo, hi) = (self.ptr as usize, self.ptr as usize + self.len);
            let mut total = 0usize;
            let mut inside = false;
            for line in smaps.lines() {
                if let Some((range, _)) = line.split_once(' ') {
                    if let Some((a, b)) = range.split_once('-') {
                        if let (Ok(a), Ok(b)) =
                            (usize::from_str_radix(a, 16), usize::from_str_radix(b, 16))
                        {
                            inside = a >= lo && b <= hi;
                            continue;
                        }
                    }
                }
                if !inside {
                    continue;
                }
                let kb = |l: &str| {
                    l.split_whitespace()
                        .nth(1)
                        .and_then(|v| v.parse::<usize>().ok())
                        .unwrap_or(0)
                        * 1024
                };
                if line.starts_with("AnonHugePages:")
                    || line.starts_with("Private_Hugetlb:")
                    || line.starts_with("Shared_Hugetlb:")
                {
                    total += kb(line);
                }
            }
            Some(total)
        }
        #[cfg(not(target_os = "linux"))]
        None
    }
}

impl Drop for HostArena {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        self.unregister();
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.len);
        }
    }
}

/// What [`HostArena::register`] did.
#[cfg(feature = "cuda")]
#[derive(Clone, Debug)]
pub struct RegisterReport {
    /// Size of each registered region (all equal unless it fell back).
    pub chunk: usize,
    pub regions: usize,
    pub seconds: f64,
    /// Chunk sizes tried and refused, with the driver's error.
    pub refused: Vec<(usize, String)>,
}

#[cfg(feature = "cuda")]
impl HostArena {
    /// Pin and device-map the arena. Tries one region, then each of `fallback_chunks` in turn
    /// (e.g. `[4 GiB, 1 GiB, 256 MiB]`). On failure nothing stays registered.
    pub fn register(
        &mut self,
        ctx: &std::sync::Arc<cudarc::driver::CudaContext>,
        fallback_chunks: &[usize],
    ) -> crate::gpu::Result<RegisterReport> {
        use cudarc::driver::sys;
        assert!(self.regions.is_empty(), "arena already registered");
        ctx.bind_to_thread().map_err(crate::gpu::Error::from)?;
        self.ctx = Some(ctx.clone());
        let flags = sys::CU_MEMHOSTREGISTER_PORTABLE | sys::CU_MEMHOSTREGISTER_DEVICEMAP;
        let t0 = std::time::Instant::now();
        let mut refused = Vec::new();
        let mut tries = vec![self.len];
        tries.extend(fallback_chunks.iter().copied().filter(|&c| c < self.len));
        for chunk in tries {
            let chunk = chunk.div_ceil(HUGE) * HUGE;
            let mut ok = true;
            let mut off = 0;
            while off < self.len {
                let n = chunk.min(self.len - off);
                let p = unsafe { self.ptr.add(off) } as *mut std::ffi::c_void;
                let r = unsafe { sys::cuMemHostRegister_v2(p, n, flags) };
                if r != sys::CUresult::CUDA_SUCCESS {
                    refused.push((chunk, format!("{r:?}")));
                    ok = false;
                    break;
                }
                let mut dev: sys::CUdeviceptr = 0;
                let r = unsafe { sys::cuMemHostGetDevicePointer_v2(&mut dev, p, 0) };
                self.regions.push(Region {
                    offset: off,
                    len: n,
                    device: dev,
                });
                if r != sys::CUresult::CUDA_SUCCESS {
                    refused.push((chunk, format!("device pointer: {r:?}")));
                    ok = false;
                    break;
                }
                off += n;
            }
            if ok {
                return Ok(RegisterReport {
                    chunk,
                    regions: self.regions.len(),
                    seconds: t0.elapsed().as_secs_f64(),
                    refused,
                });
            }
            self.unregister();
        }
        Err(crate::gpu::Error(format!(
            "cuMemHostRegister refused every chunk size: {refused:?}"
        )))
    }

    /// Unpin all regions (also done on drop).
    pub fn unregister(&mut self) {
        if self.regions.is_empty() {
            return;
        }
        if let Some(ctx) = &self.ctx {
            let _ = ctx.bind_to_thread();
        }
        for r in self.regions.drain(..) {
            unsafe {
                cudarc::driver::sys::cuMemHostUnregister(self.ptr.add(r.offset) as *mut _);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_and_rounds_to_2mib() {
        let a = HostArena::new(3 << 20, ArenaOptions::default()).unwrap();
        assert_eq!(a.len(), 4 << 20);
        let b = unsafe { a.bytes_mut() };
        b[0] = 1;
        b[a.len() - 1] = 2;
        assert_eq!(b[0] + b[a.len() - 1], 3);
        assert!(a.device_ptr(0).is_none());
    }

    #[test]
    fn device_ptr_finds_region() {
        let mut a = HostArena::new(8 << 20, ArenaOptions::default()).unwrap();
        a.regions = vec![
            Region {
                offset: 0,
                len: 4 << 20,
                device: 0x1000_0000,
            },
            Region {
                offset: 4 << 20,
                len: 4 << 20,
                device: 0x9000_0000,
            },
        ];
        assert_eq!(a.device_ptr(5), Some(0x1000_0005));
        assert_eq!(a.device_ptr((4 << 20) + 1), Some(0x9000_0001));
        assert_eq!(a.device_ptr(8 << 20), None);
        a.regions.clear(); // not really registered
    }
}
