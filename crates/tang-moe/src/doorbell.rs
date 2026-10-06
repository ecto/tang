//! The per-layer GPU↔CPU handoff through mapped host memory.
//!
//! One [`Mailbox`] (a few MB of registered, device-mapped host memory) serves every layer of
//! every window. For layer `l` of window `w` the sequence number is `w · 64 + l + 1`. Inside the
//! window's graph, per layer:
//!
//! 1. **GPU publishes** (kernel `db_publish`, or the real router writing directly): the window's
//!    token count `t` into `HDR_T`, the routed ids `[t][TOPK]` into `IDS`, the token
//!    activations as `QAct { m: t, k: HIDDEN }` words into `XQ`; `__threadfence_system()`; then
//!    `SEQ = seq`.
//! 2. **Host** (spinning on `SEQ`): reads ids, classifies them against the residency table,
//!    writes the [`MoePlan`] into `PLAN` and raises `FLAG_A = seq` *before* any CPU work.
//! 3. **GPU** waits on `FLAG_A` (`db_wait`), then runs the grouped expert kernel over the plan
//!    (VRAM hits and the shared expert), concurrently with:
//! 4. **Host** computes the missed experts into `ROWS` (row `dst`, `HIDDEN` floats each, the
//!    same row numbering as `parts`), writes `CPU_ROWS = [n, dst...]`, raises `FLAG_B = seq`.
//! 5. **GPU** waits on `FLAG_B`, copies the `n` listed rows from `ROWS` into `parts`
//!    (`db_copy_rows`), then combines as usual.
//!
//! **Watchdog.** A device spin on a flag the host never raises hangs the GPU. The host polls
//! `cuStreamQuery` every ~2 ms while waiting for `SEQ`; if the stream has gone idle or failed,
//! it gives up, and [`Mailbox::abort`] raises both flags to `u32::MAX` so any spinning kernel
//! falls through. No host callbacks (`cuLaunchHostFunc`) are used, so nothing on this path
//! shares a lock with the driver.

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use cudarc::driver::sys;

use crate::arena::{ArenaOptions, HostArena};
use crate::contract::{MoePlan, QAct, HIDDEN, MAX_T, TOPK};
use crate::gpu::{Error, Gpu, Result, Stream};
use crate::miss::{build_plan, MissExec, MissJob, MissTimes};

/// Word offsets in the mailbox. Each host-written flag has a 256-byte region to itself, so it
/// never shares a GPU cache line with anything the GPU writes (a precaution; the device waits
/// also use `ld.acquire.sys`).
pub struct Mb;

/// Widest window the mailbox carries (wide prefill windows; decode stays at `MAX_T`).
pub const WIDE_T: usize = 64;
/// Row capacity at `WIDE_T`: every routed slot plus the shared expert's rows.
pub const WIDE_CAP: usize = WIDE_T * TOPK + WIDE_T;

impl Mb {
    pub const SEQ: usize = 0;
    pub const FLAG_A: usize = 64;
    pub const FLAG_B: usize = 128;
    pub const HDR_T: usize = 192;
    pub const HDR_LAYER: usize = 193;
    pub const IDS: usize = 256;
    pub const XQ: usize = (Self::IDS + WIDE_T * TOPK).next_multiple_of(64);
    pub const XQ_WORDS: usize = QAct {
        m: WIDE_T,
        k: HIDDEN,
    }
    .words();
    pub const PLAN: usize = (Self::XQ + Self::XQ_WORDS).next_multiple_of(16);
    pub const CPU_ROWS: usize = (Self::PLAN + MoePlan::WORDS).next_multiple_of(16);
    pub const ROWS: usize = (Self::CPU_ROWS + 1 + WIDE_CAP).next_multiple_of(1024);
    pub const WORDS: usize = Self::ROWS + WIDE_CAP * HIDDEN;
}

/// Sequence number of layer `l` of window `w`.
pub fn seq(win: u32, layer: usize) -> u32 {
    win.wrapping_mul(64).wrapping_add(layer as u32 + 1)
}

pub struct Mailbox {
    arena: HostArena,
    base: *mut u32,
    /// Poll the stream this often while waiting.
    pub watchdog: Duration,
}

unsafe impl Send for Mailbox {}

impl Mailbox {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let mut arena = HostArena::new(Mb::WORDS * 4, ArenaOptions::default())
            .map_err(|e| Error(format!("mailbox mmap: {e}")))?;
        arena.register(&gpu.ctx, &[])?;
        let base = arena.as_ptr() as *mut u32;
        unsafe { std::ptr::write_bytes(base, 0, Mb::WORDS) };
        Ok(Mailbox {
            arena,
            base,
            watchdog: Duration::from_millis(
                std::env::var("TANG_MOE_WATCHDOG_MS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(2),
            ),
        })
    }

    /// Device address of word 0.
    pub fn device(&self) -> u64 {
        self.arena.device_ptr(0).unwrap()
    }

    fn atom(&self, w: usize) -> &AtomicU32 {
        unsafe { &*(self.base.add(w) as *const AtomicU32) }
    }

    pub fn words(&self, at: usize, n: usize) -> &[u32] {
        assert!(at + n <= Mb::WORDS);
        unsafe { std::slice::from_raw_parts(self.base.add(at), n) }
    }

    #[allow(clippy::mut_from_ref)]
    fn words_mut(&self, at: usize, n: usize) -> &mut [u32] {
        assert!(at + n <= Mb::WORDS);
        unsafe { std::slice::from_raw_parts_mut(self.base.add(at), n) }
    }

    /// Spin until `SEQ` reaches `target`. Every `watchdog` interval, check that `stream` is
    /// still running; error out (and [`abort`](Self::abort)) if it finished or failed.
    pub fn wait_seq(&self, target: u32, stream: &Stream) -> Result<()> {
        let mut last = Instant::now();
        loop {
            for _ in 0..256 {
                if self
                    .atom(Mb::SEQ)
                    .load(Ordering::Acquire)
                    .wrapping_sub(target) as i32
                    >= 0
                {
                    return Ok(());
                }
                std::hint::spin_loop();
            }
            if last.elapsed() >= self.watchdog {
                last = Instant::now();
                let r = unsafe { sys::cuStreamQuery(stream.0) };
                if r != sys::CUresult::CUDA_ERROR_NOT_READY {
                    // Re-check: the publish may have landed just before the stream finished.
                    if self
                        .atom(Mb::SEQ)
                        .load(Ordering::Acquire)
                        .wrapping_sub(target) as i32
                        >= 0
                    {
                        return Ok(());
                    }
                    self.abort();
                    return Err(Error(format!(
                        "watchdog: stream is {r:?} but seq {target} never arrived"
                    )));
                }
            }
        }
    }

    /// The published request: `(t, ids [t·TOPK], xq words of QAct { m: t, k: HIDDEN })`.
    pub fn request(&self) -> (usize, &[u32], &[u32]) {
        let t = (self.words(Mb::HDR_T, 1)[0] as usize).min(WIDE_T);
        (
            t,
            self.words(Mb::IDS, t * TOPK),
            self.words(Mb::XQ, QAct { m: t, k: HIDDEN }.words()),
        )
    }

    /// The plan area, to fill before [`raise_a`](Self::raise_a).
    #[allow(clippy::mut_from_ref)]
    pub fn plan_mut(&self) -> &mut [u32] {
        self.words_mut(Mb::PLAN, MoePlan::WORDS)
    }

    pub fn raise_a(&self, seq: u32) {
        self.atom(Mb::FLAG_A).store(seq, Ordering::Release);
    }

    /// Start of `ROWS` (`PARTS_ROWS × HIDDEN` floats).
    pub fn rows_ptr(&self) -> *mut f32 {
        unsafe { self.base.add(Mb::ROWS) as *mut f32 }
    }

    /// List the rows written and raise `FLAG_B`.
    pub fn raise_b(&self, seq: u32, dsts: &[u32]) {
        let w = self.words_mut(Mb::CPU_ROWS, 1 + WIDE_CAP);
        w[0] = dsts.len() as u32;
        w[1..1 + dsts.len()].copy_from_slice(dsts);
        self.atom(Mb::FLAG_B).store(seq, Ordering::Release);
    }

    /// Release any kernel spinning on either flag.
    pub fn abort(&self) {
        self.atom(Mb::FLAG_A).store(u32::MAX, Ordering::SeqCst);
        self.atom(Mb::FLAG_B).store(u32::MAX, Ordering::SeqCst);
    }

    /// Clear flags and sequence (between benchmark runs, with the GPU idle).
    pub fn reset(&self) {
        for w in [Mb::SEQ, Mb::FLAG_A, Mb::FLAG_B] {
            self.atom(w).store(0, Ordering::SeqCst);
        }
    }
}

/// Host-side timings of one served layer, µs from `SEQ` observed.
#[derive(Clone, Copy, Debug, Default)]
pub struct LayerTimes {
    /// Spent waiting for `SEQ` (GPU work before routing, from the host's view).
    pub wait_us: f64,
    /// `SEQ` seen → `FLAG_A` raised (classify + plan).
    pub plan_us: f64,
    /// `FLAG_A` → `FLAG_B` (CPU experts).
    pub cpu_us: f64,
    pub missed: usize,
    pub rows: usize,
    pub exec: MissTimes,
}

/// Serve one layer: wait for the GPU's request, publish the plan, compute the misses, publish
/// the rows. `addr(e)` is expert `e`'s device address if resident (0 if not); `host(e)` its
/// host blob. `dsts` is scratch.
#[allow(clippy::too_many_arguments)]
pub fn serve_layer<'b>(
    mb: &Mailbox,
    seq: u32,
    stream: &Stream,
    exec: &mut MissExec,
    addr: &dyn Fn(u32) -> u64,
    host: &dyn Fn(u32) -> &'b [u8],
    shared: u64,
) -> Result<LayerTimes> {
    let t0 = Instant::now();
    mb.wait_seq(seq, stream)?;
    let t1 = Instant::now();
    let (t, ids, xq) = mb.request();
    let missed = build_plan(ids, t, addr, shared, mb.plan_mut());
    mb.raise_a(seq);
    let t2 = Instant::now();
    let jobs: Vec<MissJob> = missed
        .iter()
        .map(|m| MissJob {
            blob: host(m.expert),
            toks: &m.toks,
        })
        .collect();
    // SAFETY: ROWS holds PARTS_ROWS rows; dst < t·TOPK ≤ PARTS_ROWS; the GPU reads them only
    // after FLAG_B.
    let exec_t = unsafe { exec.run(xq, t, &jobs, mb.rows_ptr()) };
    let dsts: Vec<u32> = missed
        .iter()
        .flat_map(|m| m.toks.iter().map(|&(_, d)| d as u32))
        .collect();
    mb.raise_b(seq, &dsts);
    let t3 = Instant::now();
    Ok(LayerTimes {
        wait_us: (t1 - t0).as_secs_f64() * 1e6,
        plan_us: (t2 - t1).as_secs_f64() * 1e6,
        cpu_us: (t3 - t2).as_secs_f64() * 1e6,
        missed: missed.len(),
        rows: dsts.len(),
        exec: exec_t,
    })
}

/// The device side: CUDA C for the doorbell kernels and the stand-in GPU work. Word offsets
/// are passed in as `#define`s generated from [`Mb`].
pub fn kernels_src() -> String {
    format!(
        r#"
#define MB_SEQ {seq}
#define MB_FLAG_A {fa}
#define MB_FLAG_B {fb}
#define MB_HDR_T {ht}
#define MB_HDR_LAYER {hl}
#define MB_IDS {ids}
#define MB_XQ {xq}
#define MB_PLAN {plan}
#define MB_CPU_ROWS {cr}
#define MB_ROWS {rows}
#define HIDDEN {hidden}
#define TOPK {topk}
#define GROUP_PTR {gp}

__device__ __forceinline__ unsigned seq_of(const unsigned* win, int layer) {{
    return win[0] * 64u + (unsigned)layer + 1u;
}}

// Publish layer `layer`'s ids and activations (one block), then bump SEQ.
extern "C" __global__ void db_publish(unsigned* mb, const unsigned* ids, const unsigned* xq,
                                      int xq_words, const unsigned* win, int layer, int t) {{
    for (int i = threadIdx.x; i < t * TOPK; i += blockDim.x) mb[MB_IDS + i] = ids[i];
    for (int i = threadIdx.x; i < xq_words; i += blockDim.x) mb[MB_XQ + i] = xq[i];
    if (threadIdx.x == 0) {{ mb[MB_HDR_T] = t; mb[MB_HDR_LAYER] = layer; }}
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {{
        *(volatile unsigned*)&mb[MB_SEQ] = seq_of(win, layer);
        __threadfence_system();
    }}
}}

__device__ __forceinline__ unsigned ld_acquire_sys(const unsigned* p) {{
    unsigned v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}}

// Thread 0 spins until mb[flag] reaches this layer's sequence number; then the block copies
// `n` words from mb[src..] into VRAM `dst`. Control words are staged once like this because
// every warp that reads mapped memory pays a separate, serialised PCIe read (measured ~1 µs
// each: 2,816 blocks reading three plan words took 29 ms).
extern "C" __global__ void db_wait(const unsigned* mb, int flag, const unsigned* win, int layer,
                                   int src, int n, unsigned* dst) {{
    if (threadIdx.x == 0) {{
        const unsigned target = seq_of(win, layer);
        while ((int)(ld_acquire_sys(mb + flag) - target) < 0) {{ }}
    }}
    __syncthreads();
    volatile const unsigned* m = mb + src;
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = m[i];
}}

// Copy the host-computed rows into parts (one block per listed row). `list` is the CPU_ROWS
// words staged into VRAM by db_wait: [n, dst...].
extern "C" __global__ void db_copy_rows(const unsigned* mb, const unsigned* list, float* parts) {{
    const unsigned n = list[0];
    if (blockIdx.x >= n) return;
    const unsigned dst = list[1 + blockIdx.x];
    const float4* src = (const float4*)(mb + MB_ROWS + (size_t)dst * HIDDEN);
    float4* out = (float4*)(parts + (size_t)dst * HIDDEN);
    for (int i = threadIdx.x; i < HIDDEN / 4; i += blockDim.x) out[i] = src[i];
}}

// Debug: record the GPU's global timer (ns) into buf[i].
extern "C" __global__ void stamp(unsigned long long* buf, int i) {{
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    buf[i] = t;
}}

// Stand-in for the layer's dense work: stream `vecs` 16-byte vectors of VRAM.
extern "C" __global__ void dense_read(const uint4* p, long long vecs, unsigned* sink) {{
    unsigned s = 0;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < vecs;
         i += (long long)gridDim.x * blockDim.x) {{
        uint4 v = p[i];
        s ^= v.x + v.y + v.z + v.w;
    }}
    if (s == 0x12345678u) sink[0] = s;
}}

// Stand-in for the grouped expert kernel: read every planned group's 1.38 MB blob once
// (grid.y = group capacity; blocks past n_groups exit). `plan` is in VRAM.
extern "C" __global__ void plan_hits(const unsigned* plan, unsigned* sink) {{
    const unsigned* p = plan;
    const unsigned g = blockIdx.y;
    if (g >= p[0]) return;
    const unsigned long long a = (unsigned long long)p[GROUP_PTR + 2 * g] |
                                 ((unsigned long long)p[GROUP_PTR + 2 * g + 1] << 32);
    const uint4* blob = (const uint4*)a;
    const int vecs = 1382400 / 16;
    unsigned s = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vecs; i += gridDim.x * blockDim.x) {{
        uint4 v = blob[i];
        s ^= v.x + v.y + v.z + v.w;
    }}
    if (s == 0x12345678u) sink[0] = s;
}}
"#,
        seq = Mb::SEQ,
        fa = Mb::FLAG_A,
        fb = Mb::FLAG_B,
        ht = Mb::HDR_T,
        hl = Mb::HDR_LAYER,
        ids = Mb::IDS,
        xq = Mb::XQ,
        plan = Mb::PLAN,
        cr = Mb::CPU_ROWS,
        rows = Mb::ROWS,
        hidden = HIDDEN,
        topk = TOPK,
        gp = MoePlan::GROUP_PTR,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_is_aligned_and_disjoint() {
        const { assert!(Mb::IDS + WIDE_T * TOPK <= Mb::XQ) };
        // Host-written flags 256 bytes clear of everything else.
        for f in [Mb::FLAG_A, Mb::FLAG_B] {
            for other in [Mb::SEQ, Mb::FLAG_A, Mb::FLAG_B, Mb::HDR_T, Mb::IDS] {
                assert!(f == other || f.abs_diff(other) >= 64);
            }
        }
        const { assert!(Mb::PLAN >= Mb::XQ + Mb::XQ_WORDS) };
        const { assert!(Mb::CPU_ROWS >= Mb::PLAN + MoePlan::WORDS) };
        const { assert!(Mb::ROWS.is_multiple_of(4)) }; // float4 copies
        assert_eq!(seq(1, 0), 65);
    }
}
