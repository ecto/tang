//! The Flash-Next decode engine on CUDA: weights in VRAM in the kernels' formats, routed experts
//! split between VRAM slots and a pinned host arena (tang-moe `ResidentCache`), misses computed
//! on the CPU through the per-layer doorbell (tang-moe `doorbell`), and a whole verify window
//! (48 layers, embedding to argmax) as one CUDA graph per window size.
//!
//! A window is `t` = 1..=8 consecutive tokens at positions `pos0..pos0 + t`. Without drafts
//! (prefill, and plain decode) every token is kept, so GDN layers run the step in `Commit` mode
//! with `n_keep = t` inside the window: the same outputs `ReadOnly` would give, and the state
//! written in the same pass (contract in `tang_compute::flash::GdnMode`).
//!
//! Per layer, on the stream:
//!
//! ```text
//! hc read (+ pending MoE write) → Q4X input GEMV ‖ bf16 side GEMV (GDN a|b, QSA indexer) → mixer
//! → Q4X output GEMV → hc read (+ mixer write) → bf16 router GEMV → top-10
//! → db_publish (ids, int8 activations → mailbox) → shared expert (Q4X) → db_wait A (plan)
//! → grouped experts over VRAM hits → db_wait B (CPU rows) → copy rows
//! ```
//!
//! and on the host, per layer: wait for the publish, plan, raise A, compute the misses on the
//! P-cores, raise B.

use super::kernels;
use super::ngram::NgramTable;
use super::pack::{self, Entry, Fmt};
use super::reference::Hparams;
use crate::gguf::{GgmlType, Gguf};
use anyhow::{anyhow, bail, ensure, Context, Result};
use std::mem::ManuallyDrop;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tang_compute::cuda::CudaBuffer;
use tang_compute::flash::shape::*;
use tang_compute::flash::{
    self as fl, ExpertBlob, GdnMode, GdnParams, HcPending, HcWeights, MoePlan, QAct, QsaCache,
    QsaNorms, MAX_T,
};
use tang_compute::{ComputeDevice, CudaComputeDevice};
use tang_moe::arena::{ArenaOptions, HostArena};
use tang_moe::doorbell::{self, Mailbox, Mb};
use tang_moe::gpu::{self, Gpu, Graph, Module, Stream};
use tang_moe::miss::{build_plan, MissExec, MissJob};
use tang_moe::policy::{AdaptParams, Geometry};
use tang_moe::pool::Pool;
use tang_moe::q2cpu::Isa;
use tang_moe::resident::Loc;

type B = CudaBuffer;
type Fun = cudarc::driver::sys::CUfunction;

/// Widest prefill window (`prefill_wide`): scratch, staging and plans are sized for it.
pub const WIDE: usize = MoePlan::WIDE_MAX;

/// Router rows: 512 experts and the shared expert's gate.
const ROUTER_ROWS: usize = EXPERTS + 1;
const EPS: f32 = 1e-6;

/// Where routed experts are computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertMode {
    /// VRAM slots + pinned host arena (resident mode); misses on the CPU.
    Cache,
    /// Every routed expert on the CPU, from the repacked file (parity checks).
    AllCpu,
}

#[derive(Clone, Debug)]
pub struct Opts {
    pub max_ctx: usize,
    /// VRAM expert slots; `None` sizes them from free memory.
    pub slots: Option<usize>,
    /// VRAM to leave free after everything (MB).
    pub reserve_mb: usize,
    /// VRAM held back for the MTP layer (MB).
    pub mtp_mb: usize,
    /// Exchange scratch slots for cache adaptation.
    pub scratch_slots: usize,
    /// Seed the cache from this routing profile (u32 count per key, key order).
    pub profile: Option<PathBuf>,
    pub experts: ExpertMode,
    /// Adapt the cache between windows.
    pub adapt: bool,
    /// Record GPU timestamps around the doorbell waits (the split).
    pub split: bool,
    /// The MTP GGUF (drafting); loaded before the expert slots are sized.
    pub mtp: Option<PathBuf>,
}

impl Default for Opts {
    fn default() -> Self {
        Opts {
            max_ctx: 8192,
            slots: None,
            reserve_mb: 700,
            mtp_mb: 1000,
            scratch_slots: 96,
            profile: None,
            experts: ExpertMode::Cache,
            adapt: true,
            split: false,
            mtp: None,
        }
    }
}

/// One segment of a [`Dw::Kn`] stack.
enum KSeg {
    Nat(B, tang_compute::flash_native::NatType),
    /// bf16 rows through the kernel track's strided bf16 GEMV (f32 activations).
    Bf(B),
    Own(Dw),
}

/// Launch handles for the native GEMV (`kernels::GEMV_SRC`), one per window size.
#[derive(Clone, Copy)]
struct NativeK {
    f: [Fun; MAX_T],
    /// `fe_gemv8_t*` (int8 activations) and `fe_q8`.
    f8: [Fun; MAX_T],

    q8: Fun,
    /// Scratch for the int8 activations (`XQ8_T × 6144` codes + scales).
    xq8: u64,
    stream: cudarc::driver::sys::CUstream,
}

/// Bytes of the `fe_q8` scratch for widths up to `k`.
const XQ8_BYTES: usize = MAX_T * 6144 + MAX_T * 6144 / 32 * 4;

/// A dense GEMV weight: Q4X (int8 activations), bf16 (f32 activations), or native GGUF
/// segments (`fe_gemv`, f32 activations).
enum Dw {
    /// The kernel track's native GEMV (`native_linear_out_into`, NatX repack) per segment of a
    /// stacked projection (rows, output column), and the stack's width.
    Kn(Vec<(KSeg, usize, usize)>, usize),
    /// `TANG_FLASH_KSTACK=1`: all of a Kn weight's segments in one launch (`native_stack_into`).
    St(tang_compute::flash_native::NatStack<B>),
    Q4x(B),
    Bf16(B),
    Native {
        _w: B,
        segs: B,
        nseg: i32,
        rows: usize,
        /// 16-byte units of shared memory per warp (the largest staged row + slack).
        row16: i32,
    },
}

/// Grid (persistent: as many 4-warp blocks as fit at once, at most one row per warp) and
/// dynamic shared memory (two staged rows per warp) of a native GEMV launch.
fn gemv_grid(n: usize, row16: i32) -> (u32, u32) {
    let smem = 4 * 2 * 16 * row16 as usize;
    let per_sm = (100 * 1024 / smem.max(1)).clamp(1, 12);
    let grid = (82 * per_sm).min(n.div_ceil(4));
    (grid as u32, smem as u32)
}

/// Shared memory per warp for a native GEMV's staged rows (bf16 rows aren't staged).
fn row16(segs: &[[u64; 5]]) -> i32 {
    let rb = segs
        .iter()
        .filter(|s| s[0] != 30 && s[0] != 0)
        .map(|s| s[2])
        .max()
        .unwrap_or(0);
    ((rb + 15).div_ceil(16) + 1) as i32
}

impl Dw {
    #[allow(clippy::too_many_arguments)]
    fn apply(
        &self,
        dev: &CudaComputeDevice,
        nk: &NativeK,
        x: &B,
        xq: &B,
        out: &mut B,
        t: usize,
        k: usize,
        n: usize,
    ) {
        match self {
            Dw::Bf16(b) => dev.linear_into(x, b, out, t, k, n),
            Dw::Q4x(b) => dev.q4x_linear_into(xq, b, out, t, k, n),
            Dw::Kn(segs, width) => {
                assert_eq!(*width, n, "native GEMV rows");
                for (seg, rows, off) in segs {
                    match seg {
                        KSeg::Nat(b, ty) => {
                            dev.native_linear_out_into(*ty, xq, b, out, *off, n, t, k, *rows)
                        }
                        KSeg::Bf(b) => dev.bf16_linear_out_into(x, b, out, *off, n, t, k, *rows),
                        KSeg::Own(d) => d.native_ptr(
                            dev,
                            nk,
                            dev.buffer_addr(x),
                            dev.buffer_addr(out) + (*off * 4) as u64,
                            t,
                            k,
                            n,
                        ),
                    }
                }
            }
            Dw::St(st) => {
                assert_eq!((st.n, st.k), (n, k), "stacked GEMV shape");
                dev.native_stack_into(st, x, xq, out, t);
            }
            Dw::Native { rows, .. } => {
                assert_eq!(*rows, n, "native GEMV rows");
                self.native_ptr(dev, nk, dev.buffer_addr(x), dev.buffer_addr(out), t, k, n);
            }
        }
    }

    /// Native GEMV on raw addresses: `t` rows of f32 `x` (`k` wide) at `xp`, outputs
    /// `out[t · ostride + seg offset + row]` at `op`.
    #[allow(clippy::too_many_arguments)]
    fn native_ptr(
        &self,
        dev: &CudaComputeDevice,
        nk: &NativeK,
        xp: u64,
        op: u64,
        t: usize,
        k: usize,
        ostride: usize,
    ) {
        let Dw::Native {
            segs,
            nseg,
            rows,
            row16,
            ..
        } = self
        else {
            panic!("native_ptr on a non-native weight")
        };
        // Wider than MAX_T (wide prefill): slices of MAX_T tokens (each row's arithmetic doesn't
        // depend on the width).
        if t > MAX_T {
            for i in (0..t).step_by(MAX_T) {
                let n = MAX_T.min(t - i);
                self.native_ptr(
                    dev,
                    nk,
                    xp + (i * k * 4) as u64,
                    op + (i * ostride * 4) as u64,
                    n,
                    k,
                    ostride,
                );
            }
            return;
        }
        let (sp, ns, os, r16, xq, ki) = (
            dev.buffer_addr(segs),
            *nseg,
            ostride as i32,
            *row16,
            nk.xq8,
            k as i32,
        );
        let s = ManuallyDrop::new(Stream(nk.stream));
        unsafe {
            gpu::launch(
                nk.q8,
                ((k / 32) as u32, t as u32, 1),
                (32, 1, 1),
                0,
                &s,
                tang_moe::args![xp, xq, ki],
            )
            .expect("fe_q8 launch");
            gpu::launch(
                nk.f8[t - 1],
                (rows.div_ceil(8) as u32, 1, 1),
                (128, 1, 1),
                (8 * 16 * r16) as u32,
                &s,
                tang_moe::args![sp, ns, xp, xq, ki, op, os, r16],
            )
            .expect("fe_gemv launch")
        };
    }

    fn native(&self) -> bool {
        matches!(self, Dw::Native { .. } | Dw::Kn(..) | Dw::St(..))
    }
}

struct Hc {
    norm: B,
    down: B,
    up: B,
    inject: Option<B>,
    /// The matrices are `fl::hc_q8` buffers (default; `TANG_FLASH_HC_Q8=0` keeps bf16).
    q8: bool,
}

impl Hc {
    fn w(&self) -> HcWeights<'_, B> {
        HcWeights {
            norm: &self.norm,
            down: &self.down,
            up: &self.up,
            inject: self.inject.as_ref(),
            q8: self.q8,
        }
    }
}

struct Gdn {
    conv: B,
    dt: B,
    a: B,
    norm: B,
    state: B,
    hist: B,
    /// This layer's window inputs, kept for a later commit: stacked projection and conv output.
    proj: B,
    h: B,
}

struct Qsa {
    qn: B,
    kn: B,
    iqn: B,
    ikn: B,
    k: B,
    v: B,
    ring: B,
    pooled: B,
    proj: B,
}

struct Layer {
    hc_a: Hc,
    hc_f: Hc,
    w_in: Dw,
    in_rows: usize,
    side: B,
    side_rows: usize,
    w_out: Dw,
    router: Dw,
    sh_gu: Dw,
    sh_down: Dw,
    gdn: Option<Gdn>,
    qsa: Option<Qsa>,
}

struct Ple {
    layer: usize,
    /// `ple_key` as a native Q2_0 segment (`fe_gemv8`), so a token's PLE output doesn't depend
    /// on the window width.
    key: Dw,
    value: Dw,
    nk: B,
    nq: B,
    nc: B,
    conv: B,
    ring: B,
}

/// Per-window scratch, sized for `MAX_T`.
struct Scratch {
    /// `[pos0, n_keep, window counter]` (the `Win` record, then the doorbell's counter).
    ctl: B,
    emb: B,
    ple_e: B,
    ple_eq: B,
    ple_key: B,
    ple_val: B,
    r: B,
    x: B,
    x2: B,
    xq: B,
    inj_a: B,
    inj_f: B,
    hc: B,
    side: B,
    y: B,
    yq: B,
    mix: B,
    q: B,
    scores: B,
    sel: B,
    attn_s: B,
    attn: B,
    logits: B,
    ids: B,
    w: B,
    plan: B,
    list: B,
    moe: B,
    parts: B,
    gu: B,
    hq: B,
    hf: B,
    shy: B,
    head: B,
    out_ids: B,
    amax: B,
    stamps: B,
    /// PCIe share staging (`TANG_FLASH_PCIE_STAGE`): STAGE_MAX expert blobs and the source list.
    stage: Option<(B, B)>,
}

/// Experts per layer the staged PCIe share brings into VRAM before the MoE kernel.
const STAGE_MAX: usize = 32;

/// Raw kernels: the engine's and the doorbell's.
struct Kern {
    _m: Module,
    _db: Module,
    embed: Fun,
    ple: Fun,
    silu_q: Fun,
    copy: Fun,
    scatter: Fun,
    argmax: Fun,
    argmax2: Fun,
    pcie: Fun,
    pcie_staged: Fun,
    stage_copy: Fun,
    wait_if: Fun,
    m_embed: Fun,
    m_cat: Fun,
    m_prep: Fun,
    m_attn: Fun,
    m_attn_part: Fun,
    m_attn_merge: Fun,
    m_moe: Fun,
    m_amax1: Fun,
    m_amax2: Fun,
    m_next: Fun,
    m_short1: Fun,
    m_short2a: Fun,
    m_short2b: Fun,
    publish: Fun,
    wait: Fun,
    copy_rows: Fun,
    stamp: Fun,
}

enum Experts {
    Resident(Box<tang_moe::cache::ResidentCache>),
    Cpu(memmap2::Mmap),
}

/// Pinned staging shared with the graph: inputs copied in by the window's first nodes, the
/// argmax ids copied out by its last.
struct Io {
    arena: HostArena,
}

impl Io {
    /// The window's control words (`Scratch::ctl`): pos0, n_keep, counter, seed, temperature.
    const CTL: usize = 0;
    /// The same for the commit graph (its own copy: the next window restages CTL before the
    /// commit's copy has run).
    const CCTL: usize = 64;
    /// Raised by the n-gram reader when the window's PLE rows are staged (own cache line).
    const PLE_FLAG: usize = 128;
    const EMB: usize = 256;
    const PLE: usize = Self::EMB + WIDE * HIDDEN * 4;
    const IDS: usize = Self::PLE + WIDE * HIDDEN * 4;
    const STAMPS: usize = Self::IDS + 64;
    /// Every layer's routed ids `[48][MAX_T × TOPK]` u32, written by the graph.
    const LIDS: usize = Self::STAMPS + 48 * 8 * 8 + 64;
    /// MTP inputs: cell tokens (8 u32), then at +64 the control record [pos0, cells].
    const MTP_IN: usize = Self::LIDS + 48 * WIDE * TOPK * 4;
    /// MTP outputs per step: drafts (8 u32), then at +32 probabilities (8 f32).
    const MTP_OUT: usize = Self::MTP_IN + 128;
    /// Teacher-forced MTP cells' token embeddings (8 × 2560 f32).
    const MTP_EMB: usize = Self::MTP_OUT + 64 * 8;
    const BYTES: usize = Self::MTP_EMB + MAX_T * HIDDEN * 4;

    fn ptr(&self, off: usize) -> *mut u8 {
        unsafe { self.arena.as_ptr().add(off) }
    }
    fn f32s(&self, off: usize, n: usize) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr(off) as *mut f32, n) }
    }
    fn u32s(&self, off: usize, n: usize) -> &mut [u32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr(off) as *mut u32, n) }
    }
}

/// Timing of one window.
#[derive(Clone, Copy, Debug, Default)]
pub struct WinStats {
    pub t: usize,
    pub wall_ms: f64,
    pub host_prep_ms: f64,
    /// Host: sum over layers of SEQ → FLAG_A (classify + plan).
    pub plan_ms: f64,
    /// Host: sum over layers of FLAG_A → FLAG_B (CPU experts).
    pub cpu_ms: f64,
    /// GPU: time spent in `db_wait` for the plan / for CPU rows (with `split`).
    pub gpu_wait_a_ms: f64,
    pub gpu_wait_b_ms: f64,
    /// GPU: window start to end (with `split`).
    pub gpu_ms: f64,
    /// Host: launch to the window's last layer served; served to stream drained; drained to
    /// return (cache boundary, tables).
    pub serve_ms: f64,
    /// Host time to launch the commit / to run the cache boundary after a verify window.
    pub commit_host_ms: f64,
    pub boundary_ms: f64,
    pub drain_ms: f64,
    pub post_ms: f64,
    pub routed: usize,
    pub distinct: usize,
    /// Layers the GPU passed without waiting for the host (no CPU misses).
    pub skipped: usize,
    /// Distinct experts the GPU streamed from mapped host memory (PCIe share).
    pub pcie: usize,
    pub missed: usize,
    pub swaps: usize,
}

/// One probe point of a debug forward: the last token's tensors.
#[derive(Default)]
pub struct Probe {
    pub post_mixer: Vec<Vec<f32>>,
    pub post_moe: Vec<Vec<f32>>,
    pub mixer_out: Vec<Vec<f32>>,
    pub moe_out: Vec<Vec<f32>>,
    pub router_ids: Vec<Vec<u32>>,
    pub qsa_sel: Vec<Option<Vec<u32>>>,
    pub ple_out: Vec<f32>,
    pub final_x: Vec<f32>,
}

pub struct Engine {
    pub dev: CudaComputeDevice,
    pub hp: Hparams,
    pub opts: Opts,
    g: Gguf,
    ngram: std::sync::Arc<NgramTable>,
    layers: Vec<Layer>,
    out_hc: Hc,
    head: Dw,
    ple: Ple,
    rope: (B, B),
    s: Scratch,
    io: Io,
    mb: Mailbox,
    gpu: Gpu,
    k: Kern,
    nk: NativeK,
    _xq8: B,
    _gemv: Module,
    experts: Experts,
    exec: MissExec,
    stream: ManuallyDrop<Stream>,
    graphs: Vec<Option<Graph>>,
    commit_graphs: Vec<Option<Graph>>,
    counter: u32,
    /// Sampling: Gumbel-max with `Philox(seed, position)` noise at `temperature` (0: greedy).
    pub seed: u32,
    pub temperature: f32,
    gather_err: std::sync::Arc<std::sync::Mutex<Option<String>>>,
    ngram_pending: bool,
    /// The sequence so far (PLE needs its predecessors).
    pub tokens: Vec<u32>,
    /// Routing counts per key, for `--dump-routing`.
    pub routing: Vec<u32>,
    /// Last window's per-layer distinct routed experts (keys), for the cache.
    keys: Vec<u32>,
    /// Per layer, the device residency table `moe_route_into` plans from (`EXPERTS` addresses,
    /// 0 = on the host), mirrored from the cache's table after each boundary.
    tables: Vec<B>,
    /// Per layer, the mapped host address of every host-resident expert (PCIe share).
    host_tables: Vec<B>,
    /// Pinned (so the copies stay asynchronous) mirror of `host_tables`.
    host_addrs: HostArena,
    /// The addresses `tables` holds now (host copy, to find layers that changed).
    table_addrs: Vec<u64>,
    host_plan: Vec<u32>,
    pub use_graphs: bool,
    pub last: WinStats,
    pub n_slots: usize,
    pub load_report: String,
    mtp: Option<Box<Mtp>>,
    pub last_mtp_ms: f64,
    probe_buf: Option<B>,
    train_dump_prompt: bool,
    train_dump: Option<(
        std::io::BufWriter<std::fs::File>,
        std::io::BufWriter<std::fs::File>,
    )>,
    pub last_commit_ms: f64,
    pub last_mtp_gpu_ms: f64,
    defer_boundary: bool,
    pending_boundary: bool,
    /// At most this many missed experts per layer the GPU streams from the mapped host arena
    /// instead of the CPU computing them (`TANG_FLASH_PCIE_CAP`).
    pub pcie_cap: usize,
    /// A second stream: the MTP draft overlaps the commit.
    side: Stream,
    /// Run the MTP after every window (keeps its K/V cache complete) and keep its drafts.
    pub use_mtp: bool,
    /// The last MTP pass's chain: (draft, probability) × 3.
    pub mtp_last: Vec<(u32, f32)>,
    /// Top-k / top-p / presence-penalty sampling in the head (`enable_sampler`); `None`: argmax.
    sampler: Option<Box<super::sampler::Sampler>>,
    /// Enqueueing a wide prefill window (`prefill_wide`): no doorbell, misses over PCIe, no head.
    wide: bool,
    /// Captured wide prefill windows by width.
    wide_graphs: std::collections::HashMap<usize, Graph>,
}

fn upload_entry(dev: &CudaComputeDevice, e: &Entry, b: &[u8]) -> B {
    match e.fmt {
        Fmt::F32 => dev.upload_f32(
            &b.chunks(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<_>>(),
        ),
        Fmt::Bf16 => dev.upload_bf16(
            &b.chunks(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect::<Vec<_>>(),
        ),
        Fmt::Q4x | Fmt::HcQ8 => dev.upload_bytes(b),
        // 16 bytes of slack: fe_gemv stages rows with aligned 16-byte loads.
        Fmt::Native => {
            let mut v = b.to_vec();
            v.extend_from_slice(&[0u8; 16]);
            dev.upload_bytes(&v)
        }
        Fmt::Q2Raw => dev.upload_q2(b, e.n, e.k),
    }
}

impl Engine {
    /// Load the model at `path` (first GGUF shard) onto the GPU.
    pub fn load(path: &Path, opts: Opts) -> Result<Self> {
        let t0 = Instant::now();
        let g = Gguf::open(path)?;
        let hp = Hparams::from_gguf(&g)?;
        ensure!(
            hp.n_embd == HIDDEN && hp.n_layer == 48 && hp.hc == HC && hp.n_expert == EXPERTS,
            "not the Flash-Next geometry the kernels are built for"
        );
        let plep = hp.ple.clone().context("no PLE block")?;
        let dir = pack::cache_dir(path)?;
        let warm = dir
            .join(format!("dense.{}.json", pack::dense_tag()))
            .exists()
            && dir.join(pack::EXPERTS_FILE).exists();
        let (entries, dense_path) =
            pack::dense(&g, &dir, hp.n_layer, &hp.is_recurrent, Some(plep.layer))?;
        let experts_path = pack::experts(&g, &dir, hp.n_layer)?;
        let t_pack = t0.elapsed().as_secs_f64();

        let dev = CudaComputeDevice::new().map_err(|e| anyhow!("CUDA: {e:?}"))?;
        let gpu = Gpu {
            ctx: dev.cuda_context().clone(),
        };
        gpu.bind().map_err(|e| anyhow!("{e}"))?;
        let (free0, _) = gpu.mem_info().map_err(|e| anyhow!("{e}"))?;

        // ---- dense weights
        let t1 = Instant::now();
        let by = pack::by_name(&entries);
        let df = std::fs::File::open(&dense_path)?;
        let get = |name: &str| -> Result<B> {
            let e = by
                .get(name)
                .with_context(|| format!("pack has no {name}"))?;
            let b = pack::read_entry(&df, e)?;
            Ok(upload_entry(&dev, e, &b))
        };
        let dw = |name: &str| -> Result<Dw> {
            let e = by
                .get(name)
                .with_context(|| format!("pack has no {name}"))?;
            let w = get(name)?;
            Ok(match e.fmt {
                Fmt::Bf16 => Dw::Bf16(w),
                Fmt::Q4x => Dw::Q4x(w),
                Fmt::Native if std::env::var("TANG_FLASH_KNATIVE").map_or(true, |v| v != "0") => {
                    // The kernel track's native GEMV per segment (NatX repack); bf16 segments
                    // through fe_gemv.
                    let raw = pack::read_entry(&df, e)?;
                    drop(w);
                    let mut segs = Vec::new();
                    for &[ty, rows, rb, off, out] in &e.segs {
                        let (rows, rb, off, out) =
                            (rows as usize, rb as usize, off as usize, out as usize);
                        let bytes = &raw[off..off + rows * rb];
                        let seg = match tang_compute::flash_native::NatType::from_ggml(ty as u32) {
                            Some(nt) => KSeg::Nat(dev.upload_native(nt, bytes, rows, e.k), nt),
                            None if ty == 30 && std::env::var("TANG_FLASH_BF16_OWN").is_err() => {
                                KSeg::Bf(
                                    dev.upload_bf16(
                                        &bytes
                                            .chunks(2)
                                            .map(|c| u16::from_le_bytes([c[0], c[1]]))
                                            .collect::<Vec<_>>(),
                                    ),
                                )
                            }
                            None => {
                                let wb = super::mtp_gpu::upload_padded(&dev, bytes);
                                let p = dev.buffer_addr(&wb);
                                KSeg::Own(Dw::Native {
                                    segs: dev.upload_u32(&[
                                        p as u32,
                                        (p >> 32) as u32,
                                        ty as u32,
                                        rows as u32,
                                        rb as u32,
                                        0,
                                    ]),
                                    _w: wb,
                                    nseg: 1,
                                    rows,
                                    row16: row16(&[[ty, rows as u64, rb as u64, 0, 0]]),
                                })
                            }
                        };
                        segs.push((seg, rows, out));
                    }
                    let stackable = segs
                        .iter()
                        .all(|(s, _, _)| matches!(s, KSeg::Nat(..) | KSeg::Bf(_)));
                    if stackable && !std::env::var("TANG_FLASH_KSTACK").is_ok_and(|v| v == "0") {
                        use tang_compute::flash_native::NatSeg;
                        let st: Vec<NatSeg<B>> = segs
                            .into_iter()
                            .map(|(s, rows, off)| match s {
                                KSeg::Nat(w, ty) => NatSeg {
                                    ty: Some(ty),
                                    w,
                                    rows,
                                    off,
                                },
                                KSeg::Bf(w) => NatSeg {
                                    ty: None,
                                    w,
                                    rows,
                                    off,
                                },
                                KSeg::Own(_) => unreachable!(),
                            })
                            .collect();
                        Dw::St(dev.native_stack(st, e.n, e.k))
                    } else {
                        Dw::Kn(segs, e.n)
                    }
                }
                Fmt::Native => {
                    let base = dev.buffer_addr(&w);
                    let words: Vec<u32> = e
                        .segs
                        .iter()
                        .flat_map(|&[ty, rows, rb, off, out]| {
                            let p = base + off;
                            [
                                p as u32,
                                (p >> 32) as u32,
                                ty as u32,
                                rows as u32,
                                rb as u32,
                                out as u32,
                            ]
                        })
                        .collect();
                    Dw::Native {
                        _w: w,
                        segs: dev.upload_u32(&words),
                        nseg: e.segs.len() as i32,
                        rows: e.n,
                        row16: row16(&e.segs),
                    }
                }
                f => bail!("{name}: unexpected format {f:?}"),
            })
        };
        // A bf16 matrix through fe_gemv's pinned fma chains (the kernel track's bf16 GEMV
        // contracts differently per window width).
        let bf16_native = |name: &str| -> Result<Dw> {
            let e = by
                .get(name)
                .with_context(|| format!("pack has no {name}"))?;
            ensure!(e.fmt == Fmt::Bf16, "{name} isn't bf16");
            let raw = pack::read_entry(&df, e)?;
            if std::env::var("TANG_FLASH_BF16_OWN").is_err() {
                let w = dev.upload_bf16(
                    &raw.chunks(2)
                        .map(|c| u16::from_le_bytes([c[0], c[1]]))
                        .collect::<Vec<_>>(),
                );
                return Ok(Dw::Kn(vec![(KSeg::Bf(w), e.n, 0)], e.n));
            }
            let w = super::mtp_gpu::upload_padded(&dev, &raw);
            let p = dev.buffer_addr(&w);
            let rb = (e.k * 2) as u64;
            Ok(Dw::Native {
                segs: dev.upload_u32(&[p as u32, (p >> 32) as u32, 30, e.n as u32, rb as u32, 0]),
                _w: w,
                nseg: 1,
                rows: e.n,
                row16: row16(&[[30, e.n as u64, rb, 0, 0]]),
            })
        };
        // Hyper-connection weights at 8 bits (fl::hc_q8: Q8_0-style, f16 scale per 32 along each GGUF
        // row), quantized here from the packed bf16 (the packed up is hc_up_repack'ed: back to GGUF
        // order first). Default; TANG_FLASH_HC_Q8=0 keeps bf16.
        let hc_q8 = !std::env::var("TANG_FLASH_HC_Q8").is_ok_and(|v| v == "0");
        let hc = |key: &str, inject: bool| -> Result<Hc> {
            if hc_q8 {
                let q8 = |part: &str, n: usize, k: usize| -> Result<B> {
                    let name = format!("{key}.{part}.q8");
                    let e = by
                        .get(name.as_str())
                        .with_context(|| format!("pack has no {name}"))?;
                    ensure!((e.n, e.k) == (n, k), "{name}: {}x{} vs {n}x{k}", e.n, e.k);
                    Ok(dev.upload_bytes(&pack::read_entry(&df, e)?))
                };
                let w = HC * HIDDEN;
                return Ok(Hc {
                    norm: get(&format!("{key}.norm"))?,
                    down: q8("down", HC_LR, w)?,
                    up: q8("up", w, HC_LR)?,
                    inject: if inject {
                        Some(q8("inject", HC, w)?)
                    } else {
                        None
                    },
                    q8: true,
                });
            }
            Ok(Hc {
                norm: get(&format!("{key}.norm"))?,
                down: get(&format!("{key}.down"))?,
                up: get(&format!("{key}.up"))?,
                inject: if inject {
                    Some(get(&format!("{key}.inject"))?)
                } else {
                    None
                },
                q8: false,
            })
        };
        let max_ctx = opts.max_ctx.next_multiple_of(16);
        let mut layers = Vec::with_capacity(hp.n_layer);
        for l in 0..hp.n_layer {
            let k = format!("L{l:02}");
            let rec = hp.is_recurrent[l];
            let (in_rows, side_rows) = if rec {
                (GDN_PROJ, 2 * GDN_HV)
            } else {
                (QSA_PROJ, IDX_HEADS * IDX_D + IDX_D)
            };
            let gdn = if rec {
                Some(Gdn {
                    conv: get(&format!("{k}.conv"))?,
                    dt: get(&format!("{k}.dt"))?,
                    a: get(&format!("{k}.a"))?,
                    norm: get(&format!("{k}.norm"))?,
                    state: dev.alloc_f32(fl::GDN_STATE),
                    hist: dev.alloc_f32(fl::GDN_HIST),
                    proj: dev.alloc_f32(WIDE * GDN_PROJ),
                    h: dev.alloc_f32(WIDE * GDN_CONV),
                })
            } else {
                None
            };
            let qsa = if !rec {
                Some(Qsa {
                    qn: get(&format!("{k}.qn"))?,
                    kn: get(&format!("{k}.kn"))?,
                    iqn: get(&format!("{k}.iqn"))?,
                    ikn: get(&format!("{k}.ikn"))?,
                    k: dev.alloc_bf16(max_ctx * QSA_KV * QSA_D),
                    v: dev.alloc_bf16(max_ctx * QSA_KV * QSA_D),
                    ring: dev.alloc_f32(QSA_RING * IDX_D),
                    pooled: dev.alloc_f32(max_ctx / IDX_BLOCK * IDX_D),
                    proj: dev.alloc_f32(WIDE * QSA_PROJ),
                })
            } else {
                None
            };
            layers.push(Layer {
                hc_a: hc(&format!("{k}.hc_attn"), true)?,
                hc_f: hc(&format!("{k}.hc_ffn"), true)?,
                w_in: dw(&format!("{k}.w_in"))?,
                in_rows,
                side: get(&format!("{k}.side"))?,
                side_rows,
                w_out: dw(&format!("{k}.w_out"))?,
                router: bf16_native(&format!("{k}.router"))?,
                sh_gu: dw(&format!("{k}.sh_gu"))?,
                sh_down: dw(&format!("{k}.sh_down"))?,
                gdn,
                qsa,
            });
        }
        let ple = Ple {
            layer: plep.layer,
            key: {
                let e = by.get("ple.key").context("pack has no ple.key")?;
                ensure!(e.fmt == Fmt::Q2Raw, "ple.key must be Q2_0");
                let raw = pack::read_entry(&std::fs::File::open(&dense_path)?, e)?;
                let nt = tang_compute::flash_native::NatType::Q2_0;
                Dw::Kn(
                    vec![(KSeg::Nat(dev.upload_native(nt, &raw, e.n, e.k), nt), e.n, 0)],
                    e.n,
                )
            },
            value: bf16_native("ple.value")?,
            nk: get("ple.nk")?,
            nq: get("ple.nq")?,
            nc: get("ple.nc")?,
            conv: get("ple.conv")?,
            ring: dev.alloc_f32(kernels::PLE_RING * HC * HIDDEN),
        };
        let out_hc = hc("out_hc", false)?;
        let head = dw("head")?;
        drop(df);
        let rope = {
            let (c, s) = fl::rope_table(max_ctx, hp.rope_base as f64);
            (dev.upload_f32(&c), dev.upload_f32(&s))
        };
        let z = |n: usize| dev.alloc_f32(n);
        let s = Scratch {
            ctl: z(8),
            emb: z(WIDE * HIDDEN),
            ple_e: z(WIDE * HIDDEN),
            ple_eq: z(QAct { m: WIDE, k: HIDDEN }.words()),
            ple_key: z(WIDE * HC * HIDDEN),
            ple_val: z(WIDE * HIDDEN),
            r: z(WIDE * HC * HIDDEN),
            x: z(WIDE * HIDDEN),
            x2: z(WIDE * HIDDEN),
            xq: z(QAct { m: WIDE, k: HIDDEN }.words()),
            inj_a: z(WIDE * HC),
            inj_f: z(WIDE * HC),
            hc: z(fl::hc_scratch_words(WIDE)),
            side: z(WIDE * (IDX_HEADS * IDX_D + IDX_D)),
            y: z(WIDE * GDN_V),
            yq: z(QAct { m: WIDE, k: GDN_V }.words()),
            mix: z(WIDE * HIDDEN),
            q: z(fl::qsa_q_words(WIDE)),
            scores: z(WIDE * max_ctx / IDX_BLOCK),
            sel: z(WIDE * QSA_WIDTH),
            attn_s: z(fl::qsa_attend_scratch_words(WIDE)),
            attn: z(WIDE * QSA_OUT),
            logits: z(WIDE * ROUTER_ROWS),
            ids: z(WIDE * TOPK),
            w: z(WIDE * TOPK),
            plan: z(MoePlan::words(WIDE)),
            list: z(1 + MoePlan::cap(WIDE)),
            moe: z(MoePlan::scratch_words_for(WIDE)),
            parts: z(MoePlan::parts_rows(WIDE) * HIDDEN),
            gu: z(WIDE * 2 * FF),
            hf: z(WIDE * FF),
            hq: z(QAct { m: WIDE, k: FF }.words()),
            shy: z(WIDE * HIDDEN),
            head: z(MAX_T * hp.n_vocab),
            out_ids: z(MAX_T),
            amax: z(MAX_T * 64 * 2),
            stamps: z(48 * 8 * 2),
            stage: std::env::var("TANG_FLASH_PCIE_STAGE")
                .is_ok_and(|v| v == "1")
                .then(|| (z(STAGE_MAX * ExpertBlob::BYTES / 4), z(2 * (1 + STAGE_MAX)))),
        };
        dev.sync();
        let t_dense = t1.elapsed().as_secs_f64();
        let (free1, _) = gpu.mem_info().map_err(|e| anyhow!("{e}"))?;

        // ---- kernels, mailbox, staging
        let m = gpu.module(kernels::SRC).map_err(|e| anyhow!("{e}"))?;
        let db = gpu
            .module(&doorbell::kernels_src())
            .map_err(|e| anyhow!("{e}"))?;
        let f = |m: &Module, n: &str| m.func(n).map_err(|e| anyhow!("{e}"));
        let gm = gpu.module(kernels::GEMV_SRC).map_err(|e| anyhow!("{e}"))?;
        let names = [
            "fe_gemv_t1",
            "fe_gemv_t2",
            "fe_gemv_t3",
            "fe_gemv_t4",
            "fe_gemv_t5",
            "fe_gemv_t6",
            "fe_gemv_t7",
            "fe_gemv_t8",
        ];
        let mut fs = [std::ptr::null_mut(); MAX_T];
        for (i, n) in names.iter().enumerate() {
            fs[i] = f(&gm, n)?;
        }
        let mut f8s = [std::ptr::null_mut(); MAX_T];
        for (i, slot) in f8s.iter_mut().enumerate() {
            *slot = f(
                &gm,
                [
                    "fe_gemv8_t1",
                    "fe_gemv8_t2",
                    "fe_gemv8_t3",
                    "fe_gemv8_t4",
                    "fe_gemv8_t5",
                    "fe_gemv8_t6",
                    "fe_gemv8_t7",
                    "fe_gemv8_t8",
                ][i],
            )?;
        }
        for &f8 in &f8s {
            // Rows of up to ~12 KB staged per warp (Q8_0 at k = 6144 is 6.5 KB; 8 warps).
            unsafe {
                gpu::check(
                    cudarc::driver::sys::cuFuncSetAttribute(
                        f8,
                        cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        99 * 1024,
                    ),
                    "cuFuncSetAttribute",
                )
                .map_err(|e| anyhow!("{e}"))?;
            }
        }
        let xq8_buf = dev.alloc_f32(XQ8_BYTES / 4);
        let nk = NativeK {
            f: fs,
            f8: f8s,
            q8: f(&gm, "fe_q8")?,
            xq8: dev.buffer_addr(&xq8_buf),
            stream: dev.cu_stream(),
        };
        let k = Kern {
            m_embed: f(&gm, "fe_embed_q3k")?,
            m_cat: f(&gm, "fe_mtp_cat")?,
            m_prep: f(&gm, "fe_mtp_prep")?,
            m_attn: f(&gm, "fe_mtp_attn")?,
            m_attn_part: f(&gm, "fe_mtp_attn_part")?,
            m_attn_merge: f(&gm, "fe_mtp_attn_merge")?,
            m_moe: f(&gm, "fe_moe_rows")?,
            m_amax1: f(&gm, "fe_amaxp1")?,
            m_amax2: f(&gm, "fe_amaxp2")?,
            m_next: f(&gm, "fe_mtp_next")?,
            m_short1: f(&gm, "fe_short1")?,
            m_short2a: f(&gm, "fe_short2a")?,
            m_short2b: f(&gm, "fe_short2b")?,
            embed: f(&m, "fe_embed")?,
            ple: f(&m, "fe_ple")?,
            silu_q: f(&m, "fe_silu_q")?,
            copy: f(&m, "fe_copy")?,
            scatter: f(&m, "fe_scatter_cols")?,
            argmax: f(&m, "fe_argmax1")?,
            argmax2: f(&m, "fe_argmax2")?,
            pcie: f(&m, "fe_pcie_patch")?,
            pcie_staged: f(&m, "fe_pcie_patch_staged")?,
            stage_copy: f(&m, "fe_stage_copy")?,
            wait_if: f(&m, "fe_db_wait_if")?,
            publish: f(&db, "db_publish")?,
            wait: f(&db, "db_wait")?,
            copy_rows: f(&db, "db_copy_rows")?,
            stamp: f(&db, "stamp")?,
            _m: m,
            _db: db,
        };
        let mb = Mailbox::new(&gpu).map_err(|e| anyhow!("{e}"))?;
        let mut io = Io {
            arena: HostArena::new(
                Io::BYTES,
                ArenaOptions {
                    try_hugetlb: false,
                    thp: false,
                },
            )?,
        };
        io.arena
            .register(&gpu.ctx, &[])
            .map_err(|e| anyhow!("{e}"))?;
        let stream = ManuallyDrop::new(Stream(dev.cu_stream()));

        // ---- MTP (before sizing the expert slots, so its VRAM is accounted for)
        let mtp_loaded = match &opts.mtp {
            Some(p) => {
                let t = Instant::now();
                let (free_a, _) = gpu.mem_info().map_err(|e| anyhow!("{e}"))?;
                let m = Mtp::load(&dev, &g, p, max_ctx, hp.n_vocab)?;
                let (free_b, _) = gpu.mem_info().map_err(|e| anyhow!("{e}"))?;
                eprintln!(
                    "flash: MTP layer loaded in {:.1} s, {:.2} GB VRAM (experts {})",
                    t.elapsed().as_secs_f64(),
                    (free_a - free_b) as f64 / 1e9,
                    match m.ex_ty {
                        8 => "Q8_0",
                        42 => "Q2_0",
                        _ => "Q4_0",
                    }
                );
                Some(m)
            }
            None => None,
        };
        let mtp_reserve = if mtp_loaded.is_some() { 0 } else { opts.mtp_mb };

        // ---- experts
        let t2 = Instant::now();
        let geo = Geometry::FLASH_NEXT_Q2_0;
        let blob = ExpertBlob::BYTES;
        let (experts, n_slots) = match opts.experts {
            ExpertMode::AllCpu => {
                let f = std::fs::File::open(&experts_path)?;
                (Experts::Cpu(unsafe { memmap2::Mmap::map(&f)? }), 0)
            }
            ExpertMode::Cache => {
                let (free, _) = gpu.mem_info().map_err(|e| anyhow!("{e}"))?;
                let budget = free.saturating_sub((opts.reserve_mb + mtp_reserve) << 20);
                let fit = (budget / blob).saturating_sub(opts.scratch_slots);
                let n_slots = opts.slots.unwrap_or(fit).min(fit).min(geo.keys());
                let ranking = Self::ranking(opts.profile.as_deref(), hp.n_layer)?;
                let ef = std::fs::File::open(&experts_path)?;
                let mut done = 0u64;
                let fill = |key: u32, dst: &mut [u8]| {
                    let off = key as u64 * blob as u64;
                    pack::pread(&ef, off, dst).expect("reading experts.bin");
                    if off + blob as u64 - done >= 256 << 20 {
                        pack::drop_cache(&ef, done, off + blob as u64 - done);
                        done = off + blob as u64;
                    }
                };
                let params = if opts.adapt {
                    // TANG_FLASH_ADAPT=every,max_swaps (default 4,96).
                    let mut p = AdaptParams::default();
                    if let Ok(v) = std::env::var("TANG_FLASH_ADAPT") {
                        let f: Vec<&str> = v.split(',').collect();
                        if let Some(e) = f.first().and_then(|x| x.parse().ok()) {
                            p.every = e;
                        }
                        if let Some(m) = f.get(1).and_then(|x| x.parse().ok()) {
                            p.max_swaps = m;
                        }
                    }
                    p
                } else {
                    AdaptParams {
                        every: 0,
                        ..AdaptParams::default()
                    }
                };
                let rc = tang_moe::cache::ResidentCache::new_fill(
                    &gpu,
                    geo,
                    params,
                    n_slots,
                    opts.scratch_slots,
                    ranking,
                    fill,
                )
                .map_err(|e| anyhow!("{e}"))?;
                pack::drop_cache(&ef, 0, (geo.keys() * blob) as u64);
                (Experts::Resident(Box::new(rc)), n_slots)
            }
        };
        let t_experts = t2.elapsed().as_secs_f64();
        let (free2, _) = gpu.mem_info().map_err(|e| anyhow!("{e}"))?;
        let tables: Vec<B> = (0..hp.n_layer)
            .map(|_| dev.alloc_f32(2 * EXPERTS))
            .collect();
        let host_tables: Vec<B> = (0..hp.n_layer)
            .map(|_| dev.alloc_f32(2 * EXPERTS))
            .collect();
        let host_addrs = {
            let mut a = HostArena::new(
                48 * EXPERTS * 8,
                ArenaOptions {
                    try_hugetlb: false,
                    thp: false,
                },
            )?;
            a.register(&gpu.ctx, &[]).map_err(|e| anyhow!("{e}"))?;
            unsafe { std::slice::from_raw_parts_mut(a.as_ptr() as *mut u64, 48 * EXPERTS) }
                .fill(u64::MAX);
            a
        };
        let mut exec = MissExec::new(
            Pool::new(&Pool::default_cpus(), std::time::Duration::from_millis(20)),
            Isa::detect(),
        );
        exec.tiled = true;
        let ngram = std::sync::Arc::new(NgramTable::open(&g, &plep)?);
        let load_report = format!(
            "load: {} ({}) pack {:.1}s, dense upload {:.1}s ({:.2} GB VRAM), experts {:.1}s: {} VRAM slots ({:.2} GB) + {} scratch, host arena {:.2} GB; VRAM free {:.2} -> {:.2} GB; total {:.1}s",
            if warm { "warm" } else { "cold" },
            dir.display(),
            t_pack,
            t_dense,
            (free0 - free1) as f64 / 1e9,
            t_experts,
            n_slots,
            (n_slots * blob) as f64 / 1e9,
            if opts.experts == ExpertMode::Cache { opts.scratch_slots } else { 0 },
            if opts.experts == ExpertMode::Cache {
                ((geo.keys() - n_slots) * blob) as f64 / 1e9
            } else {
                0.0
            },
            free0 as f64 / 1e9,
            free2 as f64 / 1e9,
            t0.elapsed().as_secs_f64()
        );
        let mut eng = Engine {
            dev,
            hp,
            opts,
            g,
            ngram,
            layers,
            out_hc,
            head,
            ple,
            rope,
            s,
            io,
            mb,
            gpu,
            k,
            nk,
            _xq8: xq8_buf,
            _gemv: gm,
            experts,
            exec,
            stream,
            graphs: (0..2 * (MAX_T + 1)).map(|_| None).collect(),
            commit_graphs: (0..=MAX_T).map(|_| None).collect(),
            counter: 1,
            seed: 0,
            temperature: 0.0,
            gather_err: Default::default(),
            ngram_pending: false,
            tokens: Vec::new(),
            routing: vec![0; 48 * EXPERTS],
            keys: Vec::new(),
            host_tables,
            host_addrs,
            tables,
            table_addrs: vec![u64::MAX; 48 * EXPERTS],
            host_plan: vec![0; MoePlan::WORDS],
            use_graphs: true,
            last: WinStats::default(),
            n_slots,
            load_report,
            mtp: None,
            last_mtp_ms: 0.0,
            train_dump: None,
            train_dump_prompt: false,
            probe_buf: None,
            last_commit_ms: 0.0,
            last_mtp_gpu_ms: 0.0,
            defer_boundary: false,
            pending_boundary: false,
            pcie_cap: std::env::var("TANG_FLASH_PCIE_CAP")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            side: Stream::new().map_err(|e| anyhow!("{e}"))?,
            use_mtp: false,
            mtp_last: Vec::new(),
            sampler: None,
            wide: false,
            wide_graphs: Default::default(),
        };
        if let Some(m) = mtp_loaded {
            eng.mtp = Some(Box::new(m));
        }
        eng.sync_tables()?;
        eng.dev.sync();
        Ok(eng)
    }

    /// Hottest first: a saved profile's order, then (and without one) layer-uniform order so
    /// every layer gets the same share of slots.
    fn ranking(profile: Option<&Path>, n_layer: usize) -> Result<Vec<u32>> {
        let uniform: Vec<u32> = (0..EXPERTS)
            .flat_map(|e| (0..n_layer).map(move |l| (l * EXPERTS + e) as u32))
            .collect();
        let Some(p) = profile else {
            return Ok(uniform);
        };
        let b = std::fs::read(p).with_context(|| format!("reading profile {}", p.display()))?;
        ensure!(
            b.len() == n_layer * EXPERTS * 4,
            "profile {} has the wrong size",
            p.display()
        );
        let counts: Vec<u32> = b
            .chunks(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut r = uniform.clone();
        // Stable: ties keep layer-uniform order.
        r.sort_by_key(|&k| std::cmp::Reverse(counts[k as usize]));
        Ok(r)
    }

    /// Write the routing counts (u32 per key) for seeding a later load.
    pub fn save_routing(&self, path: &Path) -> Result<()> {
        let b: Vec<u8> = self.routing.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(path, b)?;
        Ok(())
    }

    /// Start a new sequence (GDN state and history to zero; positional state is rewritten).
    pub fn reset(&mut self) {
        for l in &mut self.layers {
            if let Some(g) = l.gdn.as_mut() {
                self.dev.zero_buffer(&mut g.state);
                self.dev.zero_buffer(&mut g.hist);
            }
        }
        self.dev.sync();
        self.tokens.clear();
    }

    fn addr(b: &B, dev: &CudaComputeDevice) -> u64 {
        dev.buffer_addr(b)
    }

    // ---- launch helpers

    unsafe fn launch(
        &self,
        f: Fun,
        grid: (u32, u32, u32),
        block: u32,
        args: &mut [*mut std::ffi::c_void],
    ) {
        gpu::launch(f, grid, (block, 1, 1), 0, &self.stream, args).expect("launch");
    }

    fn stamp(&self, i: usize) {
        if !self.opts.split {
            return;
        }
        let p = Self::addr(&self.s.stamps, &self.dev);
        let i = i as i32;
        unsafe { self.launch(self.k.stamp, (1, 1, 1), 1, tang_moe::args![p, i]) };
    }

    /// Device address of byte `off` of the pinned staging (registered device-mapped).
    fn io_dev(&self, off: usize) -> u64 {
        self.io.arena.device_ptr(off).expect("io arena is mapped")
    }

    /// `bytes` (a multiple of 4) from `src` to `dst` with a kernel on `stream`: device or
    /// mapped host addresses alike. Used instead of copy-engine memcpys on the decode path, so
    /// a window never queues behind the expert cache's swap copies on the copy engines.
    fn kcopy_on(&self, stream: cudarc::driver::sys::CUstream, dst: u64, src: u64, bytes: usize) {
        let n = (bytes / 4) as i32;
        let s = ManuallyDrop::new(Stream(stream));
        unsafe {
            gpu::launch(
                self.k.copy,
                ((n as u32).div_ceil(256).max(1), 1, 1),
                (256, 1, 1),
                0,
                &s,
                tang_moe::args![dst, src, n],
            )
            .expect("kcopy")
        };
    }

    fn kcopy(&self, dst: u64, src: u64, bytes: usize) {
        self.kcopy_on(self.stream.0, dst, src, bytes);
    }

    /// Copy the window's inputs from pinned staging (graph nodes).
    fn enqueue_inputs(&self, t: usize) {
        self.kcopy(self.dev.buffer_addr(&self.s.ctl), self.io_dev(Io::CTL), 32);
        self.kcopy(
            self.dev.buffer_addr(&self.s.emb),
            self.io_dev(Io::EMB),
            t * HIDDEN * 4,
        );
    }

    fn enqueue_outputs(&self, t: usize) {
        self.kcopy(
            self.io_dev(Io::IDS),
            self.dev.buffer_addr(&self.s.out_ids),
            t * 4,
        );
        if self.opts.split {
            self.kcopy(
                self.io_dev(Io::STAMPS),
                self.dev.buffer_addr(&self.s.stamps),
                48 * 8 * 8,
            );
        }
    }

    /// The embedding into every stream.
    fn enqueue_embed(&mut self, t: usize) {
        let (r, e) = (
            Self::addr(&self.s.r, &self.dev),
            Self::addr(&self.s.emb, &self.dev),
        );
        self.stamp(0);
        unsafe {
            self.launch(
                self.k.embed,
                ((HIDDEN / 256) as u32, HC as u32, t as u32),
                256,
                tang_moe::args![r, e],
            )
        };
    }

    /// Apply layer `l`'s MoE write (combine + hc write) explicitly.
    fn apply_moe(&mut self, t: usize) {
        let s = &mut self.s;
        self.dev.moe_combine_into(
            &s.parts,
            &s.w,
            &s.logits,
            ROUTER_ROWS,
            Some(EXPERTS),
            &mut s.mix,
            t,
        );
        self.dev.hc_write(&mut s.r, &s.mix, &s.inj_f, t);
    }

    fn enqueue_ple(&mut self, t: usize) {
        // The n-gram rows are read on the host while layer 0 runs: wait for them, then copy.
        {
            let (mb, flag, ctlw, layer, src, n, dst) = (
                self.io.arena.device_ptr(0).expect("io arena is mapped"),
                (Io::PLE_FLAG / 4) as i32,
                self.dev.buffer_addr(&self.s.ctl) + 8,
                63i32,
                0i32,
                0i32,
                self.dev.buffer_addr(&self.s.out_ids),
            );
            unsafe {
                gpu::launch(
                    self.k.wait,
                    (1, 1, 1),
                    (32, 1, 1),
                    0,
                    &self.stream,
                    tang_moe::args![mb, flag, ctlw, layer, src, n, dst],
                )
                .expect("launch");
            }
            self.kcopy(
                self.dev.buffer_addr(&self.s.ple_e),
                self.io_dev(Io::PLE),
                t * HIDDEN * 4,
            );
        }
        let dev = &self.dev;
        let s = &mut self.s;
        dev.quantize_act_into(&s.ple_e, &mut s.ple_eq, t, HIDDEN);
        self.ple.key.apply(
            dev,
            &self.nk,
            &s.ple_e,
            &s.ple_eq,
            &mut s.ple_key,
            t,
            HIDDEN,
            HC * HIDDEN,
        );
        self.ple.value.apply(
            dev,
            &self.nk,
            &s.ple_e,
            &s.ple_eq,
            &mut s.ple_val,
            t,
            HIDDEN,
            HIDDEN,
        );
        let a = |b: &B| dev.buffer_addr(b);
        let (r, key, val, nk, nq, nc, conv, ring, ctl) = (
            a(&s.r),
            a(&s.ple_key),
            a(&s.ple_val),
            a(&self.ple.nk),
            a(&self.ple.nq),
            a(&self.ple.nc),
            a(&self.ple.conv),
            a(&self.ple.ring),
            a(&s.ctl),
        );
        let (ti, eps) = (t as i32, EPS);
        unsafe {
            self.launch(
                self.k.ple,
                (HC as u32, 1, 1),
                1024,
                tang_moe::args![r, key, val, nk, nq, nc, conv, ring, ctl, ti, eps],
            )
        };
    }

    /// Layer `l` of a window of `t` (every token kept). `fused`: the previous layer's MoE write
    /// is applied inside this layer's first hyper-connection read.
    fn enqueue_layer(&mut self, l: usize, t: usize, fused: bool, verify: bool) {
        let dev = &self.dev;
        let share = self.pcie_share();
        let s = &mut self.s;
        let layer = &mut self.layers[l];
        let pending = fused.then_some(HcPending::Moe {
            parts: &s.parts,
            w: &s.w,
            logits: &s.logits,
            stride: ROUTER_ROWS,
            sg: Some(EXPERTS),
            inj: &s.inj_f,
        });
        dev.hc_read_into(
            &mut s.r,
            pending,
            &layer.hc_a.w(),
            &mut s.x,
            Some(&mut s.xq),
            Some(&mut s.inj_a),
            &mut s.hc,
            t,
            EPS,
        );
        let a = |b: &B| dev.buffer_addr(b);
        let proj = match (&mut layer.gdn, &mut layer.qsa) {
            (Some(g), _) => &mut g.proj,
            (_, Some(q)) => &mut q.proj,
            _ => unreachable!(),
        };
        layer
            .w_in
            .apply(dev, &self.nk, &s.x, &s.xq, proj, t, HIDDEN, layer.in_rows);
        if !layer.w_in.native() {
            dev.linear_into(&s.x, &layer.side, &mut s.side, t, HIDDEN, layer.side_rows);
            let (dst, stride, off, src, w) = (
                a(proj),
                layer.in_rows as i32,
                if layer.gdn.is_some() { GDN_A } else { QSA_IQ } as i32,
                a(&s.side),
                layer.side_rows as i32,
            );
            unsafe {
                gpu::launch(
                    self.k.scatter,
                    ((w as u32).div_ceil(128), t as u32, 1),
                    (128, 1, 1),
                    0,
                    &self.stream,
                    tang_moe::args![dst, stride, off, src, w],
                )
                .expect("launch")
            };
        }
        if let Some(g) = layer.gdn.as_mut() {
            dev.gdn_conv_into(&g.proj, GDN_PROJ, &g.hist, &g.conv, &mut g.h, t, EPS);
            let p = GdnParams {
                conv: &g.conv,
                dt_bias: &g.dt,
                ssm_a: &g.a,
                norm: &g.norm,
            };
            dev.gdn_step(
                &mut g.state,
                &g.h,
                &g.proj,
                GDN_PROJ,
                &p,
                &mut s.y,
                Some(&mut s.yq),
                t,
                if verify {
                    GdnMode::ReadOnly
                } else {
                    GdnMode::Commit { win: &s.ctl }
                },
                EPS,
            );
            if !verify {
                dev.gdn_conv_commit(&mut g.hist, &g.proj, GDN_PROJ, &s.ctl, t);
            }
            layer
                .w_out
                .apply(dev, &self.nk, &s.y, &s.yq, &mut s.mix, t, GDN_V, HIDDEN);
        }
        if let Some(q) = layer.qsa.as_mut() {
            let norms = QsaNorms {
                q: &q.qn,
                k: &q.kn,
                iq: &q.iqn,
                ik: &q.ikn,
            };
            let cache = QsaCache {
                k: &mut q.k,
                v: &mut q.v,
                ring: &mut q.ring,
                pooled: &mut q.pooled,
            };
            dev.qsa_prep(
                &q.proj,
                QSA_PROJ,
                &s.ctl,
                &norms,
                (&self.rope.0, &self.rope.1),
                &mut s.q,
                cache,
                t,
                EPS,
            );
            let max_blocks = self.opts.max_ctx.next_multiple_of(16) / IDX_BLOCK;
            dev.qsa_select_into(
                &q.pooled,
                &s.q,
                &s.ctl,
                &mut s.scores,
                &mut s.sel,
                max_blocks,
                t,
            );
            dev.qsa_attend_into(
                &s.q,
                &q.k,
                &q.v,
                &s.sel,
                &q.proj,
                QSA_PROJ,
                &s.ctl,
                &mut s.attn_s,
                &mut s.attn,
                Some(&mut s.yq),
                t,
            );
            layer.w_out.apply(
                dev, &self.nk, &s.attn, &s.yq, &mut s.mix, t, QSA_OUT, HIDDEN,
            );
        }
        dev.hc_read_into(
            &mut s.r,
            Some(HcPending::Write {
                y: &s.mix,
                inj: &s.inj_a,
            }),
            &layer.hc_f.w(),
            &mut s.x2,
            Some(&mut s.xq),
            Some(&mut s.inj_f),
            &mut s.hc,
            t,
            EPS,
        );
        layer.router.apply(
            dev,
            &self.nk,
            &s.x2,
            &s.xq,
            &mut s.logits,
            t,
            HIDDEN,
            ROUTER_ROWS,
        );
        // Top-10 and the plan over VRAM-resident experts, on the GPU from its residency table;
        // the host computes the same misses from its own view of the table (both change only
        // between windows).
        dev.moe_route_into(
            &s.logits,
            ROUTER_ROWS,
            EXPERTS,
            None,
            &self.tables[l],
            0,
            &mut s.ids,
            &mut s.w,
            &mut s.plan,
            t,
        );
        {
            let (dst, src, n) = (
                self.io
                    .arena
                    .device_ptr(Io::LIDS + l * 4 * WIDE * TOPK)
                    .expect("mapped"),
                a(&s.ids),
                (t * TOPK) as i32,
            );
            unsafe {
                gpu::launch(
                    self.k.copy,
                    (1, 1, 1),
                    (256, 1, 1),
                    0,
                    &self.stream,
                    tang_moe::args![dst, src, n],
                )
                .expect("launch")
            };
        }
        let wide_pcie = self.wide && std::env::var("TANG_FLASH_WIDE_PCIE").is_ok_and(|v| v == "1");
        if self.pcie_cap > 0 || wide_pcie {
            let cap = if wide_pcie {
                i32::MAX
            } else {
                share as i32
            };
            let (plan, ids, ht, ti) = (a(&s.plan), a(&s.ids), a(&self.host_tables[l]), t as i32);
            // The plan's layout at this width (MoePlan's offsets with cap(t)).
            let c = MoePlan::cap(t) as i32;
            let gp = MoePlan::GROUP_PTR as i32;
            let (gs, et) = (gp + 2 * c, gp + 3 * c + 1);
            let (ed, mi) = (et + c, et + 2 * c);
            if let (Some((sb, lb)), false) = (&s.stage, wide_pcie) {
                let (stage, blob, srcs, smax) =
                    (a(sb), ExpertBlob::BYTES as u64, a(lb), STAGE_MAX as i32);
                unsafe {
                    gpu::launch(
                        self.k.pcie_staged,
                        (1, 1, 1),
                        (32, 1, 1),
                        0,
                        &self.stream,
                        tang_moe::args![plan, ids, ht, ti, cap, gp, gs, et, ed, mi, stage, blob, srcs, smax],
                    )
                    .expect("launch");
                    gpu::launch(
                        self.k.stage_copy,
                        (STAGE_MAX as u32, 8, 1),
                        (256, 1, 1),
                        0,
                        &self.stream,
                        tang_moe::args![srcs, stage, blob],
                    )
                    .expect("launch")
                };
            } else {
            unsafe {
                gpu::launch(
                    self.k.pcie,
                    (1, 1, 1),
                    (32, 1, 1),
                    0,
                    &self.stream,
                    tang_moe::args![plan, ids, ht, ti, cap, gp, gs, et, ed, mi],
                )
                .expect("launch")
            };
            }
        }
        // Publish ids and activations; the host computes the misses while the GPU runs the
        // hits and the shared expert (not in a wide prefill window: misses go over PCIe).
        let mbp = self.mb.device();
        let ctlw = a(&s.ctl) + 8; // window counter word
        let li = l as i32;
        if !wide_pcie {
            {
                let (ids, xq, xw, ti) = (
                    a(&s.ids),
                    a(&s.xq),
                    QAct { m: t, k: HIDDEN }.words() as i32,
                    t as i32,
                );
                unsafe {
                    gpu::launch(
                        self.k.publish,
                        (1, 1, 1),
                        (512, 1, 1),
                        0,
                        &self.stream,
                        tang_moe::args![mbp, ids, xq, xw, ctlw, li, ti],
                    )
                    .expect("launch")
                };
            }
        }
        // SAFETY: every group address in the plan is a live VRAM slot or scratch blob.
        unsafe { dev.moe_grouped_into(&s.xq, &s.plan, &mut s.moe, &mut s.parts, t) };
        layer
            .sh_gu
            .apply(dev, &self.nk, &s.x2, &s.xq, &mut s.gu, t, HIDDEN, 2 * FF);
        {
            let (gu, hq, hf, ff, ti) = (a(&s.gu), a(&s.hq), a(&s.hf), FF as i32, t as i32);
            unsafe {
                gpu::launch(
                    self.k.silu_q,
                    ((FF / 32) as u32, t as u32, 1),
                    (32, 1, 1),
                    0,
                    &self.stream,
                    tang_moe::args![gu, hq, hf, ff, ti],
                )
                .expect("launch")
            };
        }
        layer
            .sh_down
            .apply(dev, &self.nk, &s.hf, &s.hq, &mut s.shy, t, FF, HIDDEN);
        {
            let (dst, src, n) = (
                a(&s.parts) + (MoePlan::shared_row(t) * HIDDEN * 4) as u64,
                a(&s.shy),
                (t * HIDDEN) as i32,
            );
            unsafe {
                gpu::launch(
                    self.k.copy,
                    ((n as u32).div_ceil(256), 1, 1),
                    (256, 1, 1),
                    0,
                    &self.stream,
                    tang_moe::args![dst, src, n],
                )
                .expect("launch")
            };
        }
        if wide_pcie {
            return;
        }
        let st = 1 + 4 * l;
        for i in [st, st + 1] {
            if self.opts.split {
                let (p, i) = (a(&s.stamps), i as i32);
                unsafe {
                    gpu::launch(
                        self.k.stamp,
                        (1, 1, 1),
                        (1, 1, 1),
                        0,
                        &self.stream,
                        tang_moe::args![p, i],
                    )
                    .expect("launch")
                };
            }
        }
        if self.opts.split {
            let (p, i) = (a(&s.stamps), st as i32 + 2);
            unsafe {
                gpu::launch(
                    self.k.stamp,
                    (1, 1, 1),
                    (1, 1, 1),
                    0,
                    &self.stream,
                    tang_moe::args![p, i],
                )
                .expect("launch")
            };
        }
        {
            let (flag, src, n, dst) = (
                Mb::FLAG_B as i32,
                Mb::CPU_ROWS as i32,
                (1 + MoePlan::cap(t)) as i32,
                a(&s.list),
            );
            let (plan, streamed) = (a(&s.plan), share as i32);
            unsafe {
                gpu::launch(
                    self.k.wait_if,
                    (1, 1, 1),
                    (256, 1, 1),
                    0,
                    &self.stream,
                    tang_moe::args![mbp, flag, ctlw, li, src, n, dst, plan, streamed],
                )
                .expect("launch")
            };
        }
        if self.opts.split {
            let (p, i) = (a(&s.stamps), st as i32 + 3);
            unsafe {
                gpu::launch(
                    self.k.stamp,
                    (1, 1, 1),
                    (1, 1, 1),
                    0,
                    &self.stream,
                    tang_moe::args![p, i],
                )
                .expect("launch")
            };
        }
        {
            let (list, parts) = (a(&s.list), a(&s.parts));
            unsafe {
                gpu::launch(
                    self.k.copy_rows,
                    (MoePlan::cap(t) as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    &self.stream,
                    tang_moe::args![mbp, list, parts],
                )
                .expect("launch")
            };
        }
    }

    /// The final read, the head and the argmax.
    fn enqueue_head(&mut self, t: usize, fused: bool) {
        let dev = &self.dev;
        let s = &mut self.s;
        let pending = fused.then_some(HcPending::Moe {
            parts: &s.parts,
            w: &s.w,
            logits: &s.logits,
            stride: ROUTER_ROWS,
            sg: Some(EXPERTS),
            inj: &s.inj_f,
        });
        dev.hc_read_into(
            &mut s.r,
            pending,
            &self.out_hc.w(),
            &mut s.x,
            Some(&mut s.xq),
            None,
            &mut s.hc,
            t,
            EPS,
        );
        let v = self.hp.n_vocab;
        self.head
            .apply(dev, &self.nk, &s.x, &s.xq, &mut s.head, t, HIDDEN, v);
        let (lg, ids, n, part, np, ctl) = (
            dev.buffer_addr(&s.head),
            dev.buffer_addr(&s.out_ids),
            v as i32,
            dev.buffer_addr(&s.amax),
            64i32,
            dev.buffer_addr(&s.ctl),
        );
        if let Some(sp) = self.sampler.as_ref() {
            sp.enqueue(dev, &self.stream, &s.head, &s.ctl, &s.out_ids, t);
        } else {
            unsafe {
                gpu::launch(
                    self.k.argmax,
                    (64, t as u32, 1),
                    (1024, 1, 1),
                    0,
                    &self.stream,
                    tang_moe::args![lg, n, part, ctl],
                )
                .expect("launch");
                gpu::launch(
                    self.k.argmax2,
                    (t as u32, 1, 1),
                    (32, 1, 1),
                    0,
                    &self.stream,
                    tang_moe::args![part, np, ids],
                )
                .expect("launch");
            }
        }
        self.stamp(1 + 4 * 48);
    }

    /// Everything a window enqueues, layer by layer; `between(l)` runs on the host after layer
    /// `l` is enqueued (eager mode serves it there).
    fn enqueue_window(
        &mut self,
        t: usize,
        unfused: bool,
        verify: bool,
        between: &mut dyn FnMut(&mut Self, usize) -> Result<()>,
    ) -> Result<()> {
        self.enqueue_inputs(t);
        self.enqueue_embed(t);
        let pl = self.ple.layer;
        for l in 0..self.layers.len() {
            if l == pl {
                if l > 0 {
                    self.apply_moe(t);
                }
                self.enqueue_ple(t);
            }
            let fused = l > 0 && l != pl && !unfused;
            if !fused && l > 0 && l != pl {
                self.apply_moe(t);
            }
            self.enqueue_layer(l, t, fused, verify);
            between(self, l)?;
        }
        if unfused {
            self.apply_moe(t);
        }
        if self.wide {
            // Prefill: no logits (the prompt's last window runs the head); the last layer's MoE
            // write is left pending (nothing reads the residual after it).
            return Ok(());
        }
        self.enqueue_head(t, !unfused);
        self.enqueue_outputs(t);
        Ok(())
    }

    /// The commit after a verify window: replay the first `n_keep` (from `Io::CCTL`) tokens'
    /// recurrence into every GDN layer's state and conv history.
    fn enqueue_commit(&mut self, t: usize) {
        self.kcopy(self.dev.buffer_addr(&self.s.ctl), self.io_dev(Io::CCTL), 32);
        let dev = &self.dev;
        let s = &mut self.s;
        for layer in &mut self.layers {
            if let Some(g) = layer.gdn.as_mut() {
                let p = GdnParams {
                    conv: &g.conv,
                    dt_bias: &g.dt,
                    ssm_a: &g.a,
                    norm: &g.norm,
                };
                dev.gdn_step(
                    &mut g.state,
                    &g.h,
                    &g.proj,
                    GDN_PROJ,
                    &p,
                    &mut s.y,
                    None,
                    t,
                    GdnMode::Commit { win: &s.ctl },
                    EPS,
                );
                dev.gdn_conv_commit(&mut g.hist, &g.proj, GDN_PROJ, &s.ctl, t);
            }
        }
    }

    // ---- host side

    /// Fill the pinned staging for tokens `self.tokens[pos0..pos0 + t]` (control words and
    /// embeddings; the n-gram rows come from [`start_gather`](Self::start_gather)).
    fn stage(&mut self, pos0: usize, t: usize) -> Result<()> {
        let ctl = self.io.u32s(Io::CTL, 5);
        ctl[0] = pos0 as u32;
        ctl[1] = t as u32;
        ctl[2] = self.counter;
        ctl[3] = self.seed;
        ctl[4] = self.temperature.to_bits();
        if let Some(sp) = self.sampler.as_mut().filter(|_| t <= MAX_T) {
            sp.stage(&self.dev, &self.tokens, pos0, t)?;
        }
        let emb_t = self.g.info("token_embd.weight")?.clone();
        let emb = self.io.f32s(Io::EMB, t * HIDDEN);
        for i in 0..t {
            let row = self.g.rows(&emb_t, self.tokens[pos0 + i] as usize, 1)?;
            emb[i * HIDDEN..(i + 1) * HIDDEN].copy_from_slice(&row);
        }
        Ok(())
    }

    /// Read the window's n-gram rows on the E-core pool into `Io::PLE`, then raise
    /// `Io::PLE_FLAG` (the graph waits for it before the PLE block, after layer 0).
    fn start_gather(&mut self, pos0: usize, t: usize) {
        struct P(*mut u8);
        unsafe impl Send for P {}
        let (ple, flag) = (P(self.io.ptr(Io::PLE)), P(self.io.ptr(Io::PLE_FLAG)));
        let toks: Vec<u32> = self.tokens[..pos0 + t].to_vec();
        let ng = self.ngram.clone();
        let err = self.gather_err.clone();
        let target = self.counter.wrapping_mul(64).wrapping_add(64);
        self.ngram_pending = true;
        ng.clone().spawn(move || {
            let (ple, flag) = (ple, flag);
            let out = unsafe { std::slice::from_raw_parts_mut(ple.0 as *mut f32, t * HIDDEN) };
            if let Err(e) = ng.gather(&toks, pos0, pos0 + t, out) {
                *err.lock().unwrap() = Some(format!("{e:#}"));
            }
            let f = unsafe { &*(flag.0 as *const std::sync::atomic::AtomicU32) };
            f.store(target, std::sync::atomic::Ordering::Release);
        });
    }

    /// Serve layer `l` of the current window: plan, raise A, compute misses, raise B.
    fn serve(&mut self, l: usize, st: &mut WinStats) -> Result<()> {
        let seq = doorbell::seq(self.counter, l);
        self.mb
            .wait_seq(seq, &self.stream)
            .map_err(|e| anyhow!("layer {l}: {e}"))?;
        // The GPU only waits on layers with CPU misses; if it has already published a later
        // layer, this one had none.
        if self.mb.words(Mb::SEQ, 1)[0] != seq {
            st.skipped += 1;
            return Ok(());
        }
        let t0 = Instant::now();
        let (t, ids, xq) = self.mb.request();
        let ids: Vec<u32> = ids.to_vec();
        let base = (l * EXPERTS) as u32;
        let experts = &self.experts;
        let tab = &self.table_addrs;
        let addr = |e: u32| -> u64 {
            match experts {
                Experts::Resident(_) => tab[(base + e) as usize],
                Experts::Cpu(_) => 0,
            }
        };
        let mut missed = if t <= MAX_T {
            build_plan(&ids, t, addr, 0, &mut self.host_plan)
        } else {
            // Wide prefill window: the missed experts in order of first appearance.
            let n = t * TOPK;
            let mut seen: Vec<u32> = Vec::new();
            let mut m = Vec::new();
            for &e in &ids[..n] {
                if seen.contains(&e) {
                    continue;
                }
                seen.push(e);
                if addr(e) == 0 {
                    let toks = (0..n)
                        .filter(|&j| ids[j] == e)
                        .map(|j| (j / TOPK, j))
                        .collect();
                    m.push(tang_moe::miss::Missed { expert: e, toks });
                }
            }
            m
        };
        let streamed = if matches!(experts, Experts::Resident(_)) {
            self.pcie_share().min(missed.len())
        } else {
            0
        };
        missed.drain(..streamed);
        let t1 = Instant::now();
        let blob = ExpertBlob::BYTES;
        let jobs: Vec<MissJob> = missed
            .iter()
            .map(|m| MissJob {
                blob: match experts {
                    Experts::Resident(rc) => rc
                        .host_blob(base + m.expert)
                        .expect("missed expert has a host copy"),
                    Experts::Cpu(map) => {
                        let o = (base + m.expert) as usize * blob;
                        &map[o..o + blob]
                    }
                },
                toks: &m.toks,
            })
            .collect();
        if t <= MAX_T {
            // SAFETY: ROWS holds PARTS_ROWS rows; the GPU reads them only after FLAG_B.
            unsafe { self.exec.run(xq, t, &jobs, self.mb.rows_ptr()) };
        } else {
            // Wide: in slices of MAX_T tokens (the executor's width), each with its own
            // activations repacked and the jobs' tokens that fall in it; rows land at their
            // window-wide dst. Each row's arithmetic is the T <= 8 path's.
            let qa = tang_moe::contract::QAct { m: t, k: HIDDEN };
            for s0 in (0..t).step_by(MAX_T) {
                let n = MAX_T.min(t - s0);
                let qs = tang_moe::contract::QAct { m: n, k: HIDDEN };
                let mut sub = vec![0u32; qs.words()];
                let (cw, sw) = (HIDDEN / 4, HIDDEN / 32);
                sub[..n * cw].copy_from_slice(&xq[qa.codes(s0)..qa.codes(s0) + n * cw]);
                sub[qs.scales(0)..qs.scales(0) + n * sw]
                    .copy_from_slice(&xq[qa.scales(s0)..qa.scales(s0) + n * sw]);
                sub[qs.sums(0)..qs.sums(0) + n * sw]
                    .copy_from_slice(&xq[qa.sums(s0)..qa.sums(s0) + n * sw]);
                let toks: Vec<Vec<(usize, usize)>> = jobs
                    .iter()
                    .map(|j| {
                        j.toks
                            .iter()
                            .filter(|&&(tk, _)| tk >= s0 && tk < s0 + n)
                            .map(|&(tk, d)| (tk - s0, d))
                            .collect()
                    })
                    .collect();
                let sj: Vec<MissJob> = jobs
                    .iter()
                    .zip(&toks)
                    .filter(|(_, tk)| !tk.is_empty())
                    .map(|(j, tk)| MissJob {
                        blob: j.blob,
                        toks: tk,
                    })
                    .collect();
                // SAFETY: as above; ROWS holds the wide plan's rows (tang-moe `WIDE_CAP`).
                unsafe { self.exec.run(&sub, n, &sj, self.mb.rows_ptr()) };
            }
        }
        let dsts: Vec<u32> = missed
            .iter()
            .flat_map(|m| m.toks.iter().map(|&(_, d)| d as u32))
            .collect();
        self.mb.raise_b(seq, &dsts);
        let t2 = Instant::now();
        st.plan_ms += (t1 - t0).as_secs_f64() * 1e3;
        st.cpu_ms += (t2 - t1).as_secs_f64() * 1e3;
        st.missed += missed.len();
        st.pcie += streamed;
        Ok(())
    }

    /// Run one window over `self.tokens[pos0..pos0 + t]` (all kept). Returns the argmax of each
    /// position's logits. With `probe`, runs eagerly and unfused and records the last token's
    /// intermediates.
    pub fn window(
        &mut self,
        pos0: usize,
        t: usize,
        mut probe: Option<&mut Probe>,
    ) -> Result<Vec<u32>> {
        self.window_mode(pos0, t, false, probe.take())
    }

    /// [`window`](Self::window) as a verify window (`verify`): the GDN state is only read; call
    /// [`commit`](Self::commit) with the number of tokens kept before the next window.
    pub fn window_mode(
        &mut self,
        pos0: usize,
        t: usize,
        verify: bool,
        mut probe: Option<&mut Probe>,
    ) -> Result<Vec<u32>> {
        ensure!((1..=MAX_T).contains(&t) && pos0 + t <= self.tokens.len());
        ensure!(
            pos0 + t <= self.opts.max_ctx,
            "context {} > max {}",
            pos0 + t,
            self.opts.max_ctx
        );
        let w0 = Instant::now();
        let mut st = WinStats {
            t,
            ..Default::default()
        };
        self.stage(pos0, t)?;
        st.host_prep_ms = w0.elapsed().as_secs_f64() * 1e3;
        self.keys.clear();
        let gi = t + if verify { MAX_T + 1 } else { 0 };
        let graph_ready = self.graphs[gi].is_some();
        if probe.is_none() && self.use_graphs && graph_ready {
            self.graphs[gi]
                .as_ref()
                .unwrap()
                .launch(&self.stream)
                .map_err(|e| anyhow!("{e}"))?;
            self.start_gather(pos0, t);
            self.probe_traffic()?;
            for l in 0..self.layers.len() {
                self.serve(l, &mut st)?;
            }
        } else {
            let unfused = probe.is_some();
            self.start_gather(pos0, t);
            let mut f = |me: &mut Self, l: usize| -> Result<()> {
                me.serve(l, &mut st)?;
                if let Some(p) = probe.as_deref_mut() {
                    me.probe_layer(l, t, p)?;
                }
                Ok(())
            };
            self.enqueue_window(t, unfused, verify, &mut f)?;
        }
        let ts = Instant::now();
        st.serve_ms = (ts - w0).as_secs_f64() * 1e3 - st.host_prep_ms;
        self.dev.sync();
        // Routing of every layer (copied out by the graph): cache usage and statistics.
        for l in 0..self.layers.len() {
            let mut d: Vec<u32> = self
                .io
                .u32s(Io::LIDS + l * 4 * WIDE * TOPK, t * TOPK)
                .to_vec();
            d.sort_unstable();
            d.dedup();
            st.routed += t * TOPK;
            st.distinct += d.len();
            let base = (l * EXPERTS) as u32;
            for &e in &d {
                self.routing[(base + e) as usize] += 1;
                self.keys.push(base + e);
            }
        }
        let td = Instant::now();
        st.drain_ms = (td - ts).as_secs_f64() * 1e3;
        if let Some(e) = self.gather_err.lock().unwrap().take() {
            bail!("n-gram rows: {e}");
        }
        let ids = self.io.u32s(Io::IDS, t).to_vec();
        if let Some(p) = probe {
            p.final_x = self.dev.download(&self.s.x)[(t - 1) * HIDDEN..t * HIDDEN].to_vec();
        }
        if self.opts.split {
            let stamps: Vec<u64> = (0..2 + 4 * 48)
                .map(|i| unsafe { *(self.io.ptr(Io::STAMPS + 8 * i) as *const u64) })
                .collect();
            st.gpu_ms = (stamps[1 + 4 * 48] - stamps[0]) as f64 / 1e6;
            for l in 0..48 {
                let b = 1 + 4 * l;
                st.gpu_wait_a_ms += (stamps[b + 1] - stamps[b]) as f64 / 1e6;
                st.gpu_wait_b_ms += (stamps[b + 3] - stamps[b + 2]) as f64 / 1e6;
            }
        }
        // Cache bookkeeping between windows (deferred by `verify` until the MTP and the
        // commit are launched).
        if self.defer_boundary {
            self.pending_boundary = true;
        } else {
            st.swaps = self.boundary()?;
        }
        self.counter = self.counter.wrapping_add(1);
        st.wall_ms = w0.elapsed().as_secs_f64() * 1e3;
        st.post_ms = td.elapsed().as_secs_f64() * 1e3;
        self.last = st;
        // Capture this size's graph now that every kernel is loaded.
        if self.use_graphs && !graph_ready {
            self.capture(t, verify)?;
        }
        Ok(ids)
    }

    /// After a verify window of `t`: keep the first `n_keep` tokens' recurrent state. Async on
    /// the stream (the next window is ordered after it).
    pub fn commit(&mut self, t: usize, n_keep: usize) -> Result<()> {
        ensure!((1..=t).contains(&n_keep));
        let c = self.io.u32s(Io::CCTL, 2);
        c[0] = 0;
        c[1] = n_keep as u32;
        if self.use_graphs {
            if self.commit_graphs[t].is_none() {
                self.enqueue_commit(t);
                self.dev.sync();
                let stream = ManuallyDrop::new(Stream(self.dev.cu_stream()));
                let me = self as *mut Self;
                let g = Graph::capture(&stream, |_| {
                    // SAFETY: `self` is not otherwise touched during capture.
                    unsafe { &mut *me }.enqueue_commit(t);
                    Ok(())
                })
                .map_err(|e| anyhow!("capture commit T={t}: {e}"))?;
                self.commit_graphs[t] = Some(g);
                return Ok(());
            }
            self.commit_graphs[t]
                .as_ref()
                .unwrap()
                .launch(&self.stream)
                .map_err(|e| anyhow!("{e}"))?;
        } else {
            self.enqueue_commit(t);
        }
        Ok(())
    }

    /// The cache's between-windows step: record usage, publish table changes, start swaps.
    fn boundary(&mut self) -> Result<usize> {
        self.pending_boundary = false;
        let mut swaps = 0;
        if let Experts::Resident(rc) = &mut self.experts {
            rc.record(&self.keys);
            swaps = rc.boundary(&self.stream).map_err(|e| anyhow!("{e}"))?;
        }
        self.sync_tables()?;
        Ok(swaps)
    }

    #[allow(clippy::mut_from_ref)]
    fn host_addrs_pinned(&self) -> &mut [u64] {
        unsafe {
            std::slice::from_raw_parts_mut(self.host_addrs.as_ptr() as *mut u64, 48 * EXPERTS)
        }
    }

    /// Copy the cache's residency table into the per-layer plan tables where it changed
    /// (stream-ordered after the cache's own publish).
    fn sync_tables(&mut self) -> Result<()> {
        let Experts::Resident(rc) = &self.experts else {
            return Ok(());
        };
        for l in 0..self.layers.len() {
            let base = l * EXPERTS;
            let mut dirty = false;
            for e in 0..EXPERTS {
                let key = (base + e) as u32;
                let a = rc.addr(key);
                let hd = if a == 0 {
                    rc.host_device_addr(key).unwrap_or(0)
                } else {
                    0
                };
                let ha = self.host_addrs_pinned();
                if ha[base + e] != hd {
                    ha[base + e] = hd;
                    dirty = true;
                }
                if self.table_addrs[base + e] != a {
                    self.table_addrs[base + e] = a;
                    dirty = true;
                }
            }
            if dirty {
                self.kcopy(
                    self.dev.buffer_addr(&self.tables[l]),
                    rc.device_addrs() + (base * 8) as u64,
                    EXPERTS * 8,
                );
                self.kcopy(
                    self.dev.buffer_addr(&self.host_tables[l]),
                    self.host_addrs.device_ptr(base * 8).expect("mapped"),
                    EXPERTS * 8,
                );
            }
        }
        Ok(())
    }

    fn capture(&mut self, t: usize, verify: bool) -> Result<()> {
        let stream = ManuallyDrop::new(Stream(self.dev.cu_stream()));
        let me = self as *mut Self;
        let g = Graph::capture(&stream, |_| {
            // SAFETY: `self` is not otherwise touched during capture.
            let me = unsafe { &mut *me };
            me.enqueue_window(t, false, verify, &mut |_, _| Ok(()))
                .map_err(|e| tang_moe::gpu::Error(format!("{e}")))
        })
        .map_err(|e| anyhow!("capture T={t}: {e}"))?;
        self.graphs[t + if verify { MAX_T + 1 } else { 0 }] = Some(g);
        Ok(())
    }

    fn probe_layer(&mut self, l: usize, t: usize, p: &mut Probe) -> Result<()> {
        // Eager and unfused: after `serve`, this layer's parts are complete once the stream
        // drains; apply the MoE write on a copy to read post_moe without disturbing the stream.
        self.dev.sync();
        let last = t - 1;
        let row = |v: Vec<f32>, w: usize| v[last * w..(last + 1) * w].to_vec();
        p.mixer_out
            .push(row(self.dev.download(&self.s.mix), HIDDEN));
        let ids = self.dev.download(&self.s.ids);
        p.router_ids.push(
            ids[last * TOPK..(last + 1) * TOPK]
                .iter()
                .map(|v| v.to_bits())
                .collect(),
        );
        if self.layers[l].qsa.is_some() {
            let sel: Vec<u32> = self.dev.download(&self.s.sel)
                [last * QSA_WIDTH..(last + 1) * QSA_WIDTH]
                .iter()
                .map(|v| v.to_bits())
                .collect();
            let n = (self.ctl_pos0() + last + 1).min(QSA_WIDTH);
            p.qsa_sel.push(Some(sel[..n].to_vec()));
        } else {
            p.qsa_sel.push(None);
        }
        // post_mixer: r with the mixer write applied is what hc_ffn read saw; r itself now
        // holds it (the write was applied inside that read).
        p.post_mixer
            .push(row(self.dev.download(&self.s.r), HC * HIDDEN));
        let mut y = self.dev.alloc_f32(MAX_T * HIDDEN);
        let s = &self.s;
        self.dev.moe_combine_into(
            &s.parts,
            &s.w,
            &s.logits,
            ROUTER_ROWS,
            Some(EXPERTS),
            &mut y,
            t,
        );
        let yv = self.dev.download(&y);
        p.moe_out.push(row(yv.clone(), HIDDEN));
        let mut r = self.dev.download(&self.s.r);
        let inj = self.dev.download(&self.s.inj_f);
        for c in 0..HC {
            let wgt = 2.0 / (1.0 + (-inj[last * HC + c] / HC as f32).exp());
            for d in 0..HIDDEN {
                r[(last * HC + c) * HIDDEN + d] += yv[last * HIDDEN + d] * wgt;
            }
        }
        p.post_moe.push(row(r, HC * HIDDEN));
        Ok(())
    }

    fn ctl_pos0(&self) -> usize {
        self.io.u32s(Io::CTL, 1)[0] as usize
    }

    /// Download the last window's logits (`[t][vocab]`).
    pub fn logits(&self, t: usize) -> Vec<f32> {
        let v = self.hp.n_vocab;
        self.dev.download(&self.s.head)[..t * v].to_vec()
    }

    /// Hit rate etc. of the cache so far.
    pub fn cache_stats(&self) -> Option<tang_moe::policy::Stats> {
        match &self.experts {
            Experts::Resident(rc) => Some(rc.policy.lfu.stats()),
            Experts::Cpu(_) => None,
        }
    }

    /// Where each key is served from right now (`true` = GPU).
    pub fn on_gpu(&self, key: u32) -> bool {
        match &self.experts {
            Experts::Resident(rc) => rc.policy.loc(key) != Loc::Host(0) && rc.addr(key) != 0,
            Experts::Cpu(_) => false,
        }
    }

    /// Feed `ids` as a prompt (windows of up to `MAX_T`), then return the greedy next token.
    /// `logits_cb(pos0, t, logits)` sees every window's logits when given.
    pub fn prefill(
        &mut self,
        ids: &[u32],
        chunk: usize,
        mut logits_cb: Option<&mut dyn FnMut(usize, usize, &[f32])>,
    ) -> Result<u32> {
        ensure!(!ids.is_empty(), "empty prompt");
        let pos_start = self.tokens.len();
        self.tokens.extend_from_slice(ids);
        let mut pos = pos_start;
        let mut next = 0;
        while pos < self.tokens.len() {
            let t = chunk.min(self.tokens.len() - pos).clamp(1, MAX_T);
            let out = self.window(pos, t, None)?;
            if let Some(cb) = logits_cb.as_deref_mut() {
                cb(pos, t, &self.logits(t));
            }
            let nexts: Vec<u32> = (0..t)
                .map(|i| self.tokens.get(pos + i + 1).copied().unwrap_or(out[t - 1]))
                .collect();
            // Training pairs from generated text only: of the prompt, just its last position
            // (whose next token is the first generated one).
            if self.train_dump_prompt || pos + t == self.tokens.len() {
                self.dump_train(pos, &nexts, if self.train_dump_prompt { 0 } else { t - 1 })?;
            }
            if self.use_mtp && self.mtp.is_some() {
                self.mtp_last = self.mtp_draft(pos, &nexts)?;
            }
            next = out[t - 1];
            pos += t;
        }
        Ok(next)
    }

    /// `TANG_FLASH_PROBE_MB=N`: a contention probe for lookahead prefetch. Alongside each graph
    /// window, a kernel on the side stream copies N MB of host-resident expert blobs from the
    /// mapped arena into a VRAM scratch buffer (the traffic a prefetch of predicted misses would
    /// make), so the window's slowdown can be measured.
    fn probe_traffic(&mut self) -> Result<()> {
        let mb: usize = std::env::var("TANG_FLASH_PROBE_MB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        if mb == 0 {
            return Ok(());
        }
        let Experts::Resident(rc) = &self.experts else {
            return Ok(());
        };
        let blob = ExpertBlob::BYTES;
        let n = (mb << 20) / blob;
        if self.probe_buf.is_none() {
            self.probe_buf = Some(self.dev.alloc_f32(n.max(1) * blob / 4));
        }
        let dst0 = self.dev.buffer_addr(self.probe_buf.as_ref().unwrap());
        let mut srcs = Vec::new();
        let mut key = (self.counter as usize * 97) % (48 * EXPERTS);
        while srcs.len() < n {
            if let Some(a) = rc.host_device_addr(key as u32) {
                srcs.push(a);
            }
            key = (key + 1) % (48 * EXPERTS);
        }
        for (i, &src) in srcs.iter().enumerate() {
            self.kcopy_on(self.side.0, dst0 + (i * blob) as u64, src, blob);
        }
        Ok(())
    }

    /// `--dump-mtp-train`: for each kept position `pos0 + i`, the final 4-stream residual (the
    /// MTP's `h`, fp16 `[4][2560]`) to `h.f16`, and `(position, token, next token)` as three u32
    /// to `ids.u32`: what a distillation job needs to fine-tune the MTP layer on this text.
    /// Rows before `from` (prompt positions) are skipped.
    fn dump_train(&mut self, pos0: usize, next: &[u32], from: usize) -> Result<()> {
        use std::io::Write;
        if self.train_dump.is_none() {
            return Ok(());
        }
        let n = next.len();
        let r = self.dev.download(&self.s.r);
        let (hw, iw) = self.train_dump.as_mut().unwrap();
        for (i, &nx) in next.iter().enumerate().take(n).skip(from) {
            let row = &r[i * HC * HIDDEN..(i + 1) * HC * HIDDEN];
            let b: Vec<u8> = row
                .iter()
                .flat_map(|&x| fl::f32_to_f16(x).to_le_bytes())
                .collect();
            hw.write_all(&b)?;
            let p = pos0 + i;
            for v in [p as u32, self.tokens[p], nx] {
                iw.write_all(&v.to_le_bytes())?;
            }
        }
        Ok(())
    }

    /// Start writing MTP training pairs to `dir` (`h.f16`, `ids.u32`).
    pub fn dump_mtp_train(&mut self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        let f = |n: &str| -> Result<std::io::BufWriter<std::fs::File>> {
            Ok(std::io::BufWriter::new(
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(dir.join(n))?,
            ))
        };
        self.train_dump = Some((f("h.f16")?, f("ids.u32")?));
        Ok(())
    }

    /// Include every prompt position in a separate residual sidecar. Generation dumps
    /// retain their existing default of recording only the last prompt position.
    pub fn dump_prompt_positions(&mut self, enabled: bool) {
        self.train_dump_prompt = enabled;
    }

    pub fn finish_mtp_train_dump(&mut self) -> Result<()> {
        use std::io::Write;
        if let Some((mut h, mut ids)) = self.train_dump.take() {
            h.flush()?;
            ids.flush()?;
        }
        self.train_dump_prompt = false;
        Ok(())
    }

    /// One speculative step: feed `cur` (the last sampled token) and `drafts` as a window, keep
    /// the drafts the model itself samples (exact match), commit the kept tokens' recurrent
    /// state, and return the sampled tokens of the kept positions (1 + accepted drafts; the
    /// last is the next `cur`). Without drafts this is a plain T=1 window.
    pub fn verify(&mut self, cur: u32, drafts: &[u32]) -> Result<Vec<u32>> {
        let pos = self.tokens.len();
        let t = 1 + drafts.len();
        ensure!(t <= MAX_T, "{} drafts", drafts.len());
        self.tokens.push(cur);
        self.tokens.extend_from_slice(drafts);
        // With a captured MTP graph, the draft runs on a second stream while the commit and the
        // cache's bookkeeping run on the main one.
        self.defer_boundary = true;
        let kept = if t == 1 {
            vec![self.window(pos, 1, None)?[0]]
        } else {
            let targets = self.window_mode(pos, t, true, None)?;
            let mut n = 1;
            while n < t && drafts[n - 1] == targets[n - 1] {
                n += 1;
            }
            targets[..n].to_vec()
        };
        self.defer_boundary = false;
        let n = kept.len();
        self.dump_train(pos, &kept, 0)?;
        self.last_mtp_ms = 0.0;
        let t0 = Instant::now();
        let overlap = self.use_mtp
            && self.use_graphs
            && self.mtp.as_ref().is_some_and(|m| m.graphs[n].is_some());
        if overlap {
            self.mtp_stage(pos, &kept);
            let ev = gpu::Event::new(false).map_err(|e| anyhow!("{e}"))?;
            ev.record(&self.stream).map_err(|e| anyhow!("{e}"))?;
            self.side.wait(&ev).map_err(|e| anyhow!("{e}"))?;
            self.mtp.as_ref().unwrap().graphs[n]
                .as_ref()
                .unwrap()
                .launch(&self.side)
                .map_err(|e| anyhow!("{e}"))?;
        }
        let tc = Instant::now();
        if t > 1 {
            self.commit(t, n)?;
            self.tokens.truncate(pos + n);
        }
        let tb = Instant::now();
        if self.pending_boundary {
            self.last.swaps = self.boundary()?;
        }
        self.last.commit_host_ms = (tb - tc).as_secs_f64() * 1e3;
        self.last.boundary_ms = tb.elapsed().as_secs_f64() * 1e3;
        if overlap {
            self.side.sync().map_err(|e| anyhow!("{e}"))?;
            self.mtp_last = self.mtp_read(n);
            self.last_mtp_ms = t0.elapsed().as_secs_f64() * 1e3;
        } else if self.use_mtp && self.mtp.is_some() {
            self.mtp_last = self.mtp_draft(pos, &kept)?;
        }
        Ok(kept)
    }

    /// Append `tok` and run it as a T=1 window; returns the greedy next token.
    pub fn step(&mut self, tok: u32) -> Result<u32> {
        self.tokens.push(tok);
        let pos = self.tokens.len() - 1;
        Ok(self.window(pos, 1, None)?[0])
    }

    /// Token `id`'s text (byte-level BPE pieces from GGUF metadata, best effort).
    pub fn vocab(&self) -> Result<Vec<String>> {
        let v = self.g.meta("tokenizer.ggml.tokens")?;
        let a = v.as_array().context("tokens")?;
        Ok(a.iter()
            .map(|x| x.as_str().unwrap_or("").to_string())
            .collect())
    }

    pub fn ngram_stats(&self) -> (u64, u64) {
        use std::sync::atomic::Ordering::Relaxed;
        (
            self.ngram.hits.load(Relaxed),
            self.ngram.reads.load(Relaxed),
        )
    }

    pub fn gguf(&self) -> &Gguf {
        &self.g
    }

    /// Fail loudly if a key isn't where the plan says (debug).
    pub fn check(&self) -> Result<()> {
        if self.layers.is_empty() {
            bail!("no layers");
        }
        Ok(())
    }
}

/// `flash-gemv-check`: the native GEMV against the CPU dequantizer on one real tensor of every
/// GGUF type the dense pack uses, at T = 1, 3 and 8. Prints the worst relative error per type.
pub fn gemv_check(path: &Path) -> Result<()> {
    use crate::gguf::GgmlType;
    let g = Gguf::open(path)?;
    let dev = CudaComputeDevice::new().map_err(|e| anyhow!("CUDA: {e:?}"))?;
    let gpu = Gpu {
        ctx: dev.cuda_context().clone(),
    };
    gpu.bind().map_err(|e| anyhow!("{e}"))?;
    let gm = gpu.module(kernels::GEMV_SRC).map_err(|e| anyhow!("{e}"))?;
    let stream = ManuallyDrop::new(Stream(dev.cu_stream()));
    let mut seen = std::collections::BTreeSet::new();
    let mut rng = 0x9e3779b97f4a7c15u64;
    let mut rnd = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        ((rng >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    for t in &g.tensors {
        if t.dims.len() != 2
            || t.name.contains("_exps")
            || t.name.starts_with("per_layer")
            || t.name == "token_embd.weight"
            || t.name.contains("hc_")
        {
            continue;
        }
        let id = match t.ty {
            GgmlType::Q4_0 => 2,
            GgmlType::Q5_0 => 6,
            GgmlType::Q8_0 => 8,
            GgmlType::Q3K => 11,
            GgmlType::Q4K => 12,
            GgmlType::Q5K => 13,
            GgmlType::Q6K => 14,
            GgmlType::Iq4Nl => 20,
            GgmlType::Iq4Xs => 23,
            GgmlType::Bf16 => 30,
            GgmlType::Q2_0 => 42,
            _ => continue,
        };
        let k = t.row_len();
        if !seen.insert((id, k)) {
            continue;
        }
        let n = t.n_rows().min(512);
        let rb = t.row_bytes()?;
        let mut rawp = g.bytes(t)[..n * rb].to_vec();
        rawp.extend_from_slice(&[0u8; 16]);
        let raw = &rawp[..n * rb];
        let w = dev.upload_bytes(&rawp);
        let p = dev.buffer_addr(&w);
        let segs = dev.upload_u32(&[p as u32, (p >> 32) as u32, id, n as u32, rb as u32, 0]);
        let r16 = row16(&[[id as u64, n as u64, rb as u64, 0, 0]]);
        let (grid, smem) = gemv_grid(n, r16);
        let mut wf = vec![0f32; n * k];
        crate::gguf::dequantize(t.ty, raw, &mut wf)?;
        let mut worst = 0f64;
        for tt in [1usize, 3, 8] {
            let xv: Vec<f32> = (0..tt * k).map(|_| rnd()).collect();
            let x = dev.upload_f32(&xv);
            let out = dev.alloc_f32(tt * n);
            let f = gm
                .func(&format!("fe_gemv_t{tt}"))
                .map_err(|e| anyhow!("{e}"))?;
            let (sp, ns, xp, ki, op, os) = (
                dev.buffer_addr(&segs),
                1i32,
                dev.buffer_addr(&x),
                k as i32,
                dev.buffer_addr(&out),
                n as i32,
            );
            unsafe {
                {
                    let total = n as i32;
                    gpu::launch(
                        f,
                        (grid, 1, 1),
                        (128, 1, 1),
                        smem,
                        &stream,
                        tang_moe::args![sp, ns, xp, ki, op, os, r16, total],
                    )
                }
                .map_err(|e| anyhow!("{e}"))?
            };
            dev.sync();
            let got = dev.download(&out);
            for ti in 0..tt {
                for r in 0..n {
                    let want: f64 = (0..k)
                        .map(|i| wf[r * k + i] as f64 * xv[ti * k + i] as f64)
                        .sum();
                    let mag: f64 = (0..k)
                        .map(|i| (wf[r * k + i] as f64 * xv[ti * k + i] as f64).abs())
                        .sum();
                    let e = (got[ti * n + r] as f64 - want).abs() / mag.max(1e-30);
                    worst = worst.max(e);
                }
            }
        }
        // The int8-activation kernel against dequant(W) · dequant(q8(x)).
        let q8 = gm.func("fe_q8").map_err(|e| anyhow!("{e}"))?;
        let xqb = dev.alloc_f32(XQ8_BYTES / 4);
        let mut worst8 = 0f64;
        for tt in [1usize, 3, 8] {
            let xv: Vec<f32> = (0..tt * k).map(|_| rnd()).collect();
            let x = dev.upload_f32(&xv);
            let out = dev.alloc_f32(tt * n);
            let (f, grid) = (
                gm.func(&format!("fe_gemv8_t{tt}"))
                    .map_err(|e| anyhow!("{e}"))?,
                n.div_ceil(8) as u32,
            );
            let (sp, ns, xp, xq, ki, op, os) = (
                dev.buffer_addr(&segs),
                1i32,
                dev.buffer_addr(&x),
                dev.buffer_addr(&xqb),
                k as i32,
                dev.buffer_addr(&out),
                n as i32,
            );
            unsafe {
                gpu::launch(
                    q8,
                    ((k / 32) as u32, tt as u32, 1),
                    (32, 1, 1),
                    0,
                    &stream,
                    tang_moe::args![xp, xq, ki],
                )
                .map_err(|e| anyhow!("{e}"))?;
                gpu::launch(
                    f,
                    (grid, 1, 1),
                    (128, 1, 1),
                    8 * 16 * r16 as u32,
                    &stream,
                    tang_moe::args![sp, ns, xp, xq, ki, op, os, r16],
                )
                .map_err(|e| anyhow!("{e}"))?
            };
            dev.sync();
            let got = dev.download(&out);
            let qw: Vec<u8> = dev
                .download(&xqb)
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect();
            let xd = |ti: usize, i: usize| -> f64 {
                let q = qw[ti * k + i] as i8 as f64;
                let o = MAX_T * k + 4 * (ti * (k / 32) + i / 32);
                q * f32::from_le_bytes([qw[o], qw[o + 1], qw[o + 2], qw[o + 3]]) as f64
            };
            for ti in 0..tt {
                for r in 0..n {
                    let (mut want, mut mag) = (0f64, 0f64);
                    for i in 0..k {
                        let v = wf[r * k + i] as f64
                            * if id == 30 {
                                xv[ti * k + i] as f64
                            } else {
                                xd(ti, i)
                            };
                        want += v;
                        mag += v.abs();
                    }
                    worst8 = worst8.max((got[ti * n + r] as f64 - want).abs() / mag.max(1e-30));
                }
            }
        }
        // Bandwidth over the whole tensor at T = 1 and 4.
        let nf = t.n_rows();
        let mut full = g.bytes(t).to_vec();
        full.extend_from_slice(&[0u8; 16]);
        let wfull = dev.upload_bytes(&full);
        let pf = dev.buffer_addr(&wfull);
        let segf = dev.upload_u32(&[pf as u32, (pf >> 32) as u32, id, nf as u32, rb as u32, 0]);
        let mut gbs = vec![];
        for tt in [1usize, 4] {
            let x = dev.upload_f32(&vec![0.5f32; tt * k]);
            let out = dev.alloc_f32(tt * nf);
            let (f, gridf) = (
                gm.func(&format!("fe_gemv8_t{tt}"))
                    .map_err(|e| anyhow!("{e}"))?,
                nf.div_ceil(8) as u32,
            );
            let (sp, ns, xp, xq, ki, op, os) = (
                dev.buffer_addr(&segf),
                1i32,
                dev.buffer_addr(&x),
                dev.buffer_addr(&xqb),
                k as i32,
                dev.buffer_addr(&out),
                nf as i32,
            );
            let mut run = || unsafe {
                gpu::launch(
                    q8,
                    ((k / 32) as u32, tt as u32, 1),
                    (32, 1, 1),
                    0,
                    &stream,
                    tang_moe::args![xp, xq, ki],
                )
                .unwrap();
                gpu::launch(
                    f,
                    (gridf, 1, 1),
                    (128, 1, 1),
                    8 * 16 * r16 as u32,
                    &stream,
                    tang_moe::args![sp, ns, xp, xq, ki, op, os, r16],
                )
                .unwrap()
            };
            run();
            let ms = dev.event_ms(&mut || {
                for _ in 0..20 {
                    run()
                }
            }) / 20.0;
            gbs.push(t.nbytes as f64 / (ms as f64 * 1e6));
        }
        println!(
            "{:<8} k={k:<5} {:<34} rows {n}: f32 path {worst:.1e}, int8 path {worst8:.1e} {}  int8 {:.0} / {:.0} GB/s at T=1/4 ({:.1} MB)",
            t.ty.name(),
            t.name,
            if worst < 1e-5 && worst8 < 1e-5 { "ok" } else { "FAIL" },
            gbs[0],
            gbs[1],
            t.nbytes as f64 / 1e6
        );
    }
    Ok(())
}

impl Engine {
    /// Per-op window-width check on layer `l`'s real weights and random inputs: for each op,
    /// does token 7 of a T=8 call equal a T=1 call on the same token (GDN: eight committed
    /// T=1 steps vs one read-only T=8 walk)? Prints mismatching element counts.
    pub fn op_tcheck(&mut self, l: usize) -> Result<()> {
        let mut rng = 0x1234_5678_9abc_def1u64;
        let mut rnd = |n: usize, sc: f32| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    (((rng >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * sc
                })
                .collect()
        };
        let dev = &self.dev;
        let diff = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count()
        };
        let t = 8;
        // hc read
        let r8 = rnd(t * HC * HIDDEN, 1.0);
        let r1 = r8[7 * HC * HIDDEN..].to_vec();
        let (mut x8, mut x1) = (dev.alloc_f32(t * HIDDEN), dev.alloc_f32(HIDDEN));
        let (mut q8, mut q1) = (
            dev.alloc_f32(QAct { m: t, k: HIDDEN }.words()),
            dev.alloc_f32(QAct { m: 1, k: HIDDEN }.words()),
        );
        let (mut i8b, mut i1b) = (dev.alloc_f32(t * HC), dev.alloc_f32(HC));
        let mut hc = dev.alloc_f32(fl::hc_scratch_words(t));
        let (mut rb8, mut rb1) = (dev.upload_f32(&r8), dev.upload_f32(&r1));
        let w = self.layers[l].hc_a.w();
        dev.hc_read_into(
            &mut rb8,
            None,
            &w,
            &mut x8,
            Some(&mut q8),
            Some(&mut i8b),
            &mut hc,
            t,
            EPS,
        );
        dev.hc_read_into(
            &mut rb1,
            None,
            &w,
            &mut x1,
            Some(&mut q1),
            Some(&mut i1b),
            &mut hc,
            1,
            EPS,
        );
        let (xa, xb) = (dev.download(&x8), dev.download(&x1));
        println!(
            "hc_read x: {} of {} differ",
            diff(&xa[7 * HIDDEN..], &xb),
            HIDDEN
        );
        // GEMVs on identical inputs
        let xv = rnd(t * HIDDEN, 1.0);
        let x8 = dev.upload_f32(&xv);
        let x1 = dev.upload_f32(&xv[7 * HIDDEN..]);
        let mut xq8 = dev.alloc_f32(QAct { m: t, k: HIDDEN }.words());
        let mut xq1 = dev.alloc_f32(QAct { m: 1, k: HIDDEN }.words());
        dev.quantize_act_into(&x8, &mut xq8, t, HIDDEN);
        dev.quantize_act_into(&x1, &mut xq1, 1, HIDDEN);
        let n = self.layers[l].in_rows;
        let (mut o8, mut o1) = (dev.alloc_f32(t * n), dev.alloc_f32(n));
        self.layers[l]
            .w_in
            .apply(dev, &self.nk, &x8, &xq8, &mut o8, t, HIDDEN, n);
        self.layers[l]
            .w_in
            .apply(dev, &self.nk, &x1, &xq1, &mut o1, 1, HIDDEN, n);
        let (a, b) = (dev.download(&o8), dev.download(&o1));
        println!("w_in GEMV: {} of {n} differ", diff(&a[7 * n..8 * n], &b));
        let (mut l8, mut l1) = (dev.alloc_f32(t * ROUTER_ROWS), dev.alloc_f32(ROUTER_ROWS));
        self.layers[l]
            .router
            .apply(dev, &self.nk, &x8, &xq8, &mut l8, t, HIDDEN, ROUTER_ROWS);
        self.layers[l]
            .router
            .apply(dev, &self.nk, &x1, &xq1, &mut l1, 1, HIDDEN, ROUTER_ROWS);
        let (a, b) = (dev.download(&l8), dev.download(&l1));
        println!(
            "router bf16 GEMV: {} of {ROUTER_ROWS} differ",
            diff(&a[7 * ROUTER_ROWS..], &b)
        );
        // PLE value GEMV (bf16, 2560 x 2560)
        let (mut v8, mut v1) = (dev.alloc_f32(t * HIDDEN), dev.alloc_f32(HIDDEN));
        self.ple
            .value
            .apply(dev, &self.nk, &x8, &xq8, &mut v8, t, HIDDEN, HIDDEN);
        self.ple
            .value
            .apply(dev, &self.nk, &x1, &xq1, &mut v1, 1, HIDDEN, HIDDEN);
        let (a, b) = (dev.download(&v8), dev.download(&v1));
        println!(
            "ple value bf16 GEMV: {} of {HIDDEN} differ",
            diff(&a[7 * HIDDEN..], &b)
        );
        // MoE combine + hc write
        {
            let parts_v = rnd(MoePlan::PARTS_ROWS * HIDDEN, 1.0);
            let wv = rnd(t * TOPK, 1.0);
            let lv = rnd(t * ROUTER_ROWS, 1.0);
            let injv = rnd(t * HC, 1.0);
            let mut p1 = parts_v.clone();
            for k in 0..TOPK {
                let (src, dst) = ((7 * TOPK + k) * HIDDEN, k * HIDDEN);
                let row = parts_v[src..src + HIDDEN].to_vec();
                p1[dst..dst + HIDDEN].copy_from_slice(&row);
            }
            let row = parts_v
                [(MoePlan::SHARED_ROW + 7) * HIDDEN..(MoePlan::SHARED_ROW + 8) * HIDDEN]
                .to_vec();
            p1[MoePlan::SHARED_ROW * HIDDEN..(MoePlan::SHARED_ROW + 1) * HIDDEN]
                .copy_from_slice(&row);
            let (pb8, pb1) = (dev.upload_f32(&parts_v), dev.upload_f32(&p1));
            let (w8, w1) = (dev.upload_f32(&wv), dev.upload_f32(&wv[7 * TOPK..]));
            let (lg8, lg1) = (dev.upload_f32(&lv), dev.upload_f32(&lv[7 * ROUTER_ROWS..]));
            let (in8, in1) = (dev.upload_f32(&injv), dev.upload_f32(&injv[7 * HC..]));
            let (mut y8, mut y1) = (dev.alloc_f32(t * HIDDEN), dev.alloc_f32(HIDDEN));
            dev.moe_combine_into(&pb8, &w8, &lg8, ROUTER_ROWS, Some(EXPERTS), &mut y8, t);
            dev.moe_combine_into(&pb1, &w1, &lg1, ROUTER_ROWS, Some(EXPERTS), &mut y1, 1);
            let (a, b) = (dev.download(&y8), dev.download(&y1));
            println!(
                "moe combine: {} of {HIDDEN} differ",
                diff(&a[7 * HIDDEN..], &b)
            );
            let (mut ra, mut rb) = (dev.upload_f32(&r8), dev.upload_f32(&r1));
            dev.hc_write(&mut ra, &y8, &in8, t);
            dev.hc_write(&mut rb, &y1, &in1, 1);
            let (a, b) = (dev.download(&ra), dev.download(&rb));
            println!(
                "hc write: {} of {} differ",
                diff(&a[7 * HC * HIDDEN..], &b),
                HC * HIDDEN
            );
        }
        // GDN: eight committed T=1 steps vs one read-only T=8 walk from the same state.
        if self.layers[l].gdn.is_some() {
            let pv = rnd(t * GDN_PROJ, 1.0);
            let st0 = rnd(fl::GDN_STATE, 0.05);
            let h0 = rnd(fl::GDN_HIST, 0.5);
            let g = self.layers[l].gdn.as_mut().unwrap();
            let p = GdnParams {
                conv: &g.conv,
                dt_bias: &g.dt,
                ssm_a: &g.a,
                norm: &g.norm,
            };
            let mut st = dev.upload_f32(&st0);
            let hist = dev.upload_f32(&h0);
            let proj8 = dev.upload_f32(&pv);
            let mut h8 = dev.alloc_f32(t * GDN_CONV);
            let mut y8 = dev.alloc_f32(t * GDN_V);
            let mut yq8 = dev.alloc_f32(QAct { m: t, k: GDN_V }.words());
            dev.gdn_conv_into(&proj8, GDN_PROJ, &hist, p.conv, &mut h8, t, EPS);
            dev.gdn_step(
                &mut st,
                &h8,
                &proj8,
                GDN_PROJ,
                &p,
                &mut y8,
                Some(&mut yq8),
                t,
                GdnMode::ReadOnly,
                EPS,
            );
            let ya = dev.download(&y8);
            let ha = dev.download(&h8);
            let mut st1 = dev.upload_f32(&st0);
            let mut hist1 = dev.upload_f32(&h0);
            let ctl = dev.upload_u32(&[0, 1, 0, 0]);
            let (mut yb, mut hb) = (vec![], vec![]);
            for i in 0..t {
                let proj1 = dev.upload_f32(&pv[i * GDN_PROJ..(i + 1) * GDN_PROJ]);
                let mut h1 = dev.alloc_f32(GDN_CONV);
                let mut y1 = dev.alloc_f32(GDN_V);
                let mut yq1 = dev.alloc_f32(QAct { m: 1, k: GDN_V }.words());
                dev.gdn_conv_into(&proj1, GDN_PROJ, &hist1, p.conv, &mut h1, 1, EPS);
                dev.gdn_step(
                    &mut st1,
                    &h1,
                    &proj1,
                    GDN_PROJ,
                    &p,
                    &mut y1,
                    Some(&mut yq1),
                    1,
                    GdnMode::Commit { win: &ctl },
                    EPS,
                );
                dev.gdn_conv_commit(&mut hist1, &proj1, GDN_PROJ, &ctl, 1);
                yb = dev.download(&y1);
                hb = dev.download(&h1);
            }
            println!(
                "gdn conv h (token 7): {} of {GDN_CONV} differ",
                diff(&ha[7 * GDN_CONV..], &hb)
            );
            println!(
                "gdn step y (token 7): {} of {GDN_V} differ",
                diff(&ya[7 * GDN_V..], &yb)
            );
        }
        Ok(())
    }
}

/// The MTP layer's weights and buffers ([`super::mtp_gpu`] for what it computes).
struct Mtp {
    eh: Dw,
    enorm: B,
    hnorm: B,
    hc_a: Hc,
    hc_f: Hc,
    hc_head: Hc,
    w_in: Dw,
    qn: B,
    kn: B,
    w_out: Dw,
    router: Dw,
    sh_gu: Dw,
    sh_down: Dw,
    /// Routed experts: ggml type, the three [512 × rows × k] tensors, their row bytes.
    ex_ty: i32,
    ex: [B; 3],
    ex_rb: [usize; 3],
    /// The main model's token embedding (Q3_K) for the chain's draft tokens.
    /// The main embedding's rows for the drafter's vocabulary (chain cells' draft tokens);
    /// teacher-forced cells' rows come from the host.
    embed: B,
    embed_rb: usize,
    emb_lo: i32,
    emb_hi: i32,
    /// The draft head: the main head's rows for tokens [0, n_lo) and [hi_base, vocab)
    /// (`TANG_FLASH_MTP_VOCAB`, e.g. 32768; default 0 = the whole head: on code, 32768 costs d1 97% -> 91%). A drafter's vocabulary
    /// changes acceptance only.
    dhead: Option<(Dw, usize, usize)>,
    /// Default (`TANG_FLASH_MTP_HEAD2=0` disables): a Q2_0 copy of the draft head that shortlists 256 rows, rescored
    /// exactly from the draft head's own (Q5_K) rows: (Q2 head, exact rows buffer, row bytes).
    head2: Option<(Dw, B, usize)>,
    short: B,
    kc: B,
    vc: B,
    /// Split-K attention partials: [8 cells][2 groups][nchunk][12 heads][258].
    part: B,
    nchunk: usize,
    h: B,
    toks: B,
    /// One control record per step: [pos0, cells].
    ctl: Vec<B>,
    e: B,
    cat: B,
    r: B,
    proj: B,
    q: B,
    attn: B,
    gu: B,
    hq: B,
    hf: B,
    logits: B,
    amax: B,
    drafts: Vec<B>,
    probs: Vec<B>,
    graphs: Vec<Option<Graph>>,
    steps: usize,
    graph_banks: std::collections::HashMap<usize, Vec<Option<Graph>>>,
}

impl Mtp {
    fn load(
        dev: &CudaComputeDevice,
        main: &Gguf,
        path: &Path,
        max_ctx: usize,
        vocab: usize,
    ) -> Result<Self> {
        use super::mtp_gpu::{self as m, raw};
        let (g, l) = m::open(path)?;
        let t = |n: &str| g.info(&format!("blk.{l}.{n}"));
        let native = |parts: &[(&str, usize)], rows_total: usize| -> Result<Dw> {
            let mut bytes = Vec::new();
            let mut words = Vec::new();
            let mut meta = Vec::new();
            for &(n, off) in parts {
                let r = raw(&g, t(n)?)?;
                while bytes.len() % 16 != 0 {
                    bytes.push(0);
                }
                meta.push((bytes.len() as u64, r.ty, r.rows, r.rb, off));
                bytes.extend_from_slice(&r.bytes);
            }
            let w = m::upload_padded(dev, &bytes);
            let base = dev.buffer_addr(&w);
            let mut seg_meta = Vec::new();
            let mut rows = 0;
            for (o, ty, r, rb, off) in meta {
                let p = base + o;
                words.extend_from_slice(&[
                    p as u32,
                    (p >> 32) as u32,
                    ty as u32,
                    r as u32,
                    rb as u32,
                    off as u32,
                ]);
                seg_meta.push([ty, r as u64, rb as u64, 0, off as u64]);
                rows += r;
            }
            ensure!(rows <= rows_total, "{parts:?}: {rows} rows > {rows_total}");
            Ok(Dw::Native {
                segs: dev.upload_u32(&words),
                _w: w,
                nseg: parts.len() as i32,
                rows: rows_total,
                row16: row16(&seg_meta),
            })
        };
        let f32v = |n: &str| -> Result<B> { Ok(dev.upload_f32(&g.dequantize(t(n)?)?)) };
        let hc = |pre: &str, inject: bool| -> Result<Hc> {
            let up = m::bf16(&g, t(&format!("{pre}up.weight"))?)?;
            Ok(Hc {
                norm: f32v(&format!("{pre}norm.weight"))?,
                down: dev.upload_bf16(&m::bf16(&g, t(&format!("{pre}down.weight"))?)?),
                up: dev.upload_bf16(&fl::hc_up_repack(&up)),
                inject: if inject {
                    Some(dev.upload_bf16(&m::bf16(&g, t(&format!("{pre}inject.weight"))?)?))
                } else {
                    None
                },
                q8: false,
            })
        };
        let q8 = std::env::var("TANG_FLASH_MTP_Q8").is_ok_and(|v| v == "1");
        // TANG_FLASH_MTP_Q2=1: experts requantized to Q2_0 (0.67 GB instead of 1.42).
        let q2 = !q8 && std::env::var("TANG_FLASH_MTP_Q2").is_ok_and(|v| v == "1");
        // The drafter's vocabulary: token ids [0, n) and the specials from 248044 on
        // (`TANG_FLASH_MTP_VOCAB`, default 106000: same acceptance as the full head on chat and code, 65536 costs code d1 96 -> 92%; 0 = all).
        let vocab_lo: Option<usize> = match std::env::var("TANG_FLASH_MTP_VOCAB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
        {
            Some(0) => None,
            Some(n) if n < 248_044 => Some(n),
            Some(_) => None,
            None => Some(106_000),
        };
        let mut ex: Vec<B> = Vec::new();
        let mut ex_rb = [0usize; 3];
        for (i, n) in [
            "ffn_gate_exps.weight",
            "ffn_up_exps.weight",
            "ffn_down_exps.weight",
        ]
        .iter()
        .enumerate()
        {
            let ti = t(n)?;
            ensure!(
                ti.ty == GgmlType::Q8_0,
                "{n}: {:?} MTP experts (Q8_0 expected)",
                ti.ty
            );
            let k = ti.row_len();
            let b = if q8 {
                g.bytes(ti).to_vec()
            } else if q2 {
                m::q8_to_q2_0(g.bytes(ti))?
            } else {
                m::q8_to_q4_0(g.bytes(ti))?
            };
            ex_rb[i] = if q8 {
                k / 32 * 34
            } else if q2 {
                k / 64 * 18
            } else {
                k / 32 * 18
            };
            ex.push(m::upload_padded(dev, &b));
        }
        let emb_t = main.info("token_embd.weight")?;
        ensure!(
            emb_t.ty == GgmlType::Q3K,
            "main embedding is {:?} (Q3_K expected)",
            emb_t.ty
        );
        let z = |n: usize| dev.alloc_f32(n);
        let pairs = MAX_T * TOPK;
        Ok(Mtp {
            eh: native(&[("nextn.eh_proj.weight", 0)], HIDDEN)?,
            enorm: f32v("nextn.enorm.weight")?,
            hnorm: f32v("nextn.hnorm.weight")?,
            hc_a: hc("hc_attn_", true)?,
            hc_f: hc("hc_ffn_", true)?,
            hc_head: hc("nextn.hc_head_", false)?,
            w_in: native(
                &[
                    ("attn_q.weight", 0),
                    ("attn_k.weight", QSA_HEADS * 2 * QSA_D),
                    ("attn_v.weight", QSA_HEADS * 2 * QSA_D + QSA_KV * QSA_D),
                ],
                super::mtp_gpu::MTP_PROJ,
            )?,
            qn: f32v("attn_q_norm.weight")?,
            kn: f32v("attn_k_norm.weight")?,
            w_out: native(&[("attn_output.weight", 0)], HIDDEN)?,
            router: native(
                &[
                    ("ffn_gate_inp.weight", 0),
                    ("ffn_gate_inp_shexp.weight", EXPERTS),
                ],
                ROUTER_ROWS,
            )?,
            sh_gu: native(
                &[("ffn_gate_shexp.weight", 0), ("ffn_up_shexp.weight", FF)],
                2 * FF,
            )?,
            sh_down: native(&[("ffn_down_shexp.weight", 0)], HIDDEN)?,
            ex_ty: if q8 {
                8
            } else if q2 {
                42
            } else {
                2
            },
            ex: [ex.remove(0), ex.remove(0), ex.remove(0)],
            ex_rb,
            embed: {
                let all = main.bytes(emb_t);
                let rb = emb_t.row_bytes()?;
                match vocab_lo {
                    Some(n) => {
                        let mut b = all[..n * rb].to_vec();
                        b.extend_from_slice(&all[248_044.min(vocab) * rb..vocab * rb]);
                        m::upload_padded(dev, &b)
                    }
                    None => m::upload_padded(dev, all),
                }
            },
            emb_lo: vocab_lo.map_or(vocab as i32, |n| n as i32),
            emb_hi: 248_044.min(vocab) as i32,
            dhead: {
                if let Some(n_lo) = vocab_lo {
                    let ht = main.info("output.weight")?;
                    let rb = ht.row_bytes()?;
                    let hi_base = 248_044.min(vocab);
                    let all = main.bytes(ht);
                    let mut b = all[..n_lo * rb].to_vec();
                    b.extend_from_slice(&all[hi_base * rb..vocab * rb]);
                    let w = m::upload_padded(dev, &b);
                    let p = dev.buffer_addr(&w);
                    let rows = n_lo + vocab - hi_base;
                    let ty = m::ggml_id(ht.ty)?;
                    Some((
                        Dw::Native {
                            segs: dev.upload_u32(&[
                                p as u32,
                                (p >> 32) as u32,
                                ty as u32,
                                rows as u32,
                                rb as u32,
                                0,
                            ]),
                            _w: w,
                            nseg: 1,
                            rows,
                            row16: row16(&[[ty, rows as u64, rb as u64, 0, 0]]),
                        },
                        n_lo,
                        hi_base,
                    ))
                } else {
                    None
                }
            },
            head2: if vocab_lo.is_some()
                && !std::env::var("TANG_FLASH_MTP_HEAD2").is_ok_and(|v| v == "0")
            {
                let n_lo =
                    vocab_lo.context("TANG_FLASH_MTP_HEAD2 needs a pruned draft vocabulary")?;
                let ht = main.info("output.weight")?;
                ensure!(ht.ty == GgmlType::Q5K, "head2 assumes a Q5_K head");
                let rb = ht.row_bytes()?;
                let hi_base = 248_044.min(vocab);
                let all = main.bytes(ht);
                let mut b = all[..n_lo * rb].to_vec();
                b.extend_from_slice(&all[hi_base * rb..vocab * rb]);
                let rows = n_lo + vocab - hi_base;
                let mut f = vec![0f32; rows * HIDDEN];
                use rayon::prelude::*;
                f.par_chunks_mut(HIDDEN * 1024).enumerate().try_for_each(
                    |(i, o)| -> Result<()> {
                        let r0 = i * 1024;
                        let n = o.len() / HIDDEN;
                        crate::gguf::dequantize(ht.ty, &b[r0 * rb..(r0 + n) * rb], o)
                    },
                )?;
                let rows = rows;
                let q2 = m::f32_to_q2_0(&f);
                let q2rb = HIDDEN / 64 * 18;
                let w = m::upload_padded(dev, &q2);
                let p = dev.buffer_addr(&w);
                let exact = m::upload_padded(dev, &b);
                Some((
                    Dw::Native {
                        segs: dev.upload_u32(&[
                            p as u32,
                            (p >> 32) as u32,
                            42,
                            rows as u32,
                            q2rb as u32,
                            0,
                        ]),
                        _w: w,
                        nseg: 1,
                        rows,
                        row16: row16(&[[42, rows as u64, q2rb as u64, 0, 0]]),
                    },
                    exact,
                    rb,
                ))
            } else {
                None
            },
            short: z(64 * 10 + 256),
            embed_rb: emb_t.row_bytes()?,
            kc: z(max_ctx * QSA_KV * QSA_D),
            vc: z(max_ctx * QSA_KV * QSA_D),
            part: z(MAX_T * QSA_KV * max_ctx.div_ceil(32) * 12 * 258),
            nchunk: max_ctx.div_ceil(32),
            h: z(MAX_T * HC * HIDDEN),
            toks: z(MAX_T),
            ctl: (0..super::mtp_gpu::MAX_STEPS).map(|_| z(4)).collect(),
            steps: super::mtp_gpu::steps(),
            graph_banks: Default::default(),
            e: z(MAX_T * HIDDEN),
            cat: z(MAX_T * HC * 2 * HIDDEN),
            r: z(MAX_T * HC * HIDDEN),
            proj: z(MAX_T * super::mtp_gpu::MTP_PROJ),
            q: z(MAX_T * QSA_HEADS * QSA_D),
            attn: z(MAX_T * QSA_OUT),
            gu: z(pairs * 2 * FF),
            hq: z(QAct { m: pairs, k: FF }.words()),
            hf: z(pairs * FF),
            logits: z(MAX_T * vocab),
            amax: z(MAX_T * 64 * 3),
            drafts: (0..super::mtp_gpu::MAX_STEPS).map(|_| z(MAX_T)).collect(),
            probs: (0..super::mtp_gpu::MAX_STEPS).map(|_| z(MAX_T)).collect(),
            graphs: (0..=MAX_T).map(|_| None).collect(),
        })
    }
}

impl Engine {
    /// MTP cells for `c` = 1..=8 positions starting at the control record of step `step`.
    fn enqueue_mtp_cells(&mut self, step: usize, c: usize) {
        let dev = &self.dev;
        let nk = self.nk;
        let st = ManuallyDrop::new(Stream(nk.stream));
        let mt = self.mtp.as_mut().expect("MTP loaded");
        let s = &mut self.s;
        let a = |b: &B| dev.buffer_addr(b);
        let launch = |f: Fun,
                      grid: (u32, u32, u32),
                      block: u32,
                      args: &mut [*mut std::ffi::c_void]| unsafe {
            gpu::launch(f, grid, (block, 1, 1), 0, &st, args).expect("mtp launch")
        };
        let ctl = a(&mt.ctl[step]);
        // embedding of the cells' tokens (step 0: staged by the host), then [e ; hn] per stream
        {
            let (tb, rb, tk, e, k, lo, hi) = (
                a(&mt.embed),
                mt.embed_rb as i32,
                a(&mt.toks),
                a(&mt.e),
                HIDDEN as i32,
                mt.emb_lo,
                mt.emb_hi,
            );
            if step > 0 {
                launch(
                    self.k.m_embed,
                    (c as u32, 1, 1),
                    256,
                    tang_moe::args![tb, rb, tk, e, k, lo, hi],
                );
            }
            let (en, h, hn, cat, eps) = (a(&mt.enorm), a(&mt.h), a(&mt.hnorm), a(&mt.cat), EPS);
            launch(
                self.k.m_cat,
                ((c * HC) as u32, 1, 1),
                256,
                tang_moe::args![e, en, h, hn, cat, eps],
            );
        }
        // R = eh_proj [e ; hn[c]], 4c rows of 5120, eight at a time
        let rows = c * HC;
        for r0 in (0..rows).step_by(MAX_T) {
            let n = (rows - r0).min(MAX_T);
            mt.eh.native_ptr(
                dev,
                &nk,
                a(&mt.cat) + (r0 * 2 * HIDDEN * 4) as u64,
                a(&mt.r) + (r0 * HIDDEN * 4) as u64,
                n,
                2 * HIDDEN,
                HIDDEN,
            );
        }
        dev.hc_read_into(
            &mut mt.r,
            None,
            &mt.hc_a.w(),
            &mut s.x,
            Some(&mut s.xq),
            Some(&mut s.inj_a),
            &mut s.hc,
            c,
            EPS,
        );
        let mp = super::mtp_gpu::MTP_PROJ;
        mt.w_in
            .native_ptr(dev, &nk, a(&s.x), a(&mt.proj), c, HIDDEN, mp);
        {
            let (pr, stride, qn, kn, cs, sn, q, kc, vc, eps) = (
                a(&mt.proj),
                mp as i32,
                a(&mt.qn),
                a(&mt.kn),
                a(&self.rope.0),
                a(&self.rope.1),
                a(&mt.q),
                a(&mt.kc),
                a(&mt.vc),
                EPS,
            );
            launch(
                self.k.m_prep,
                (c as u32, 26, 1),
                256,
                tang_moe::args![pr, stride, qn, kn, cs, sn, ctl, q, kc, vc, eps],
            );
            let (out, part, nch) = (a(&mt.attn), a(&mt.part), mt.nchunk as i32);
            launch(
                self.k.m_attn_part,
                (c as u32, QSA_KV as u32, mt.nchunk as u32),
                128,
                tang_moe::args![q, kc, vc, ctl, part, nch],
            );
            launch(
                self.k.m_attn_merge,
                (c as u32, QSA_HEADS as u32, 1),
                256,
                tang_moe::args![part, nch, pr, stride, out],
            );
        }
        mt.w_out
            .native_ptr(dev, &nk, a(&mt.attn), a(&s.mix), c, QSA_OUT, HIDDEN);
        dev.hc_read_into(
            &mut mt.r,
            Some(HcPending::Write {
                y: &s.mix,
                inj: &s.inj_a,
            }),
            &mt.hc_f.w(),
            &mut s.x2,
            Some(&mut s.xq),
            Some(&mut s.inj_f),
            &mut s.hc,
            c,
            EPS,
        );
        mt.router
            .native_ptr(dev, &nk, a(&s.x2), a(&s.logits), c, HIDDEN, ROUTER_ROWS);
        dev.router_topk_into(&s.logits, ROUTER_ROWS, EXPERTS, &mut s.ids, &mut s.w, c);
        // routed experts: gate | up rows per (token, rank), SwiGLU, down rows into parts
        {
            let pairs = (c * TOPK) as u32;
            let (ids, x2, gu, hf, hq, parts) = (
                a(&s.ids),
                a(&s.x2),
                a(&mt.gu),
                a(&mt.hf),
                a(&mt.hq),
                a(&s.parts),
            );
            let ty = mt.ex_ty;
            for (i, ooff) in [(0usize, 0i32), (1, FF as i32)] {
                let (base, es, rb, rows, xd, k, os) = (
                    a(&mt.ex[i]),
                    (FF * mt.ex_rb[i]) as u64,
                    mt.ex_rb[i] as i32,
                    FF as i32,
                    TOPK as i32,
                    HIDDEN as i32,
                    (2 * FF) as i32,
                );
                launch(
                    self.k.m_moe,
                    ((FF / 8) as u32, pairs, 1),
                    256,
                    tang_moe::args![ty, base, es, rb, rows, ids, x2, xd, k, gu, os, ooff],
                );
            }
            let (ff, np) = (FF as i32, pairs as i32);
            launch(
                self.k.silu_q,
                ((FF / 32) as u32, pairs, 1),
                32,
                tang_moe::args![gu, hq, hf, ff, np],
            );
            let (base, es, rb, rows, xd, k, os, ooff) = (
                a(&mt.ex[2]),
                (HIDDEN * mt.ex_rb[2]) as u64,
                mt.ex_rb[2] as i32,
                HIDDEN as i32,
                1i32,
                FF as i32,
                HIDDEN as i32,
                0i32,
            );
            launch(
                self.k.m_moe,
                ((HIDDEN / 8) as u32, pairs, 1),
                256,
                tang_moe::args![ty, base, es, rb, rows, ids, hf, xd, k, parts, os, ooff],
            );
        }
        // shared expert into parts' shared rows
        mt.sh_gu
            .native_ptr(dev, &nk, a(&s.x2), a(&s.gu), c, HIDDEN, 2 * FF);
        {
            let (gu, hq, hf, ff, ti) = (a(&s.gu), a(&s.hq), a(&s.hf), FF as i32, c as i32);
            launch(
                self.k.silu_q,
                ((FF / 32) as u32, c as u32, 1),
                32,
                tang_moe::args![gu, hq, hf, ff, ti],
            );
        }
        mt.sh_down.native_ptr(
            dev,
            &nk,
            a(&s.hf),
            a(&s.parts) + (MoePlan::SHARED_ROW * HIDDEN * 4) as u64,
            c,
            FF,
            HIDDEN,
        );
        dev.hc_read_into(
            &mut mt.r,
            Some(HcPending::Moe {
                parts: &s.parts,
                w: &s.w,
                logits: &s.logits,
                stride: ROUTER_ROWS,
                sg: Some(EXPERTS),
                inj: &s.inj_f,
            }),
            &mt.hc_head.w(),
            &mut s.x,
            Some(&mut s.xq),
            None,
            &mut s.hc,
            c,
            EPS,
        );
        // Only the last cell's draft is used: the head for that row alone.
        let xl = a(&s.x) + ((c - 1) * HIDDEN * 4) as u64;
        if let (Some((h2, exact, rb)), Some((_, n_lo, hi))) = (&mt.head2, &mt.dhead) {
            let rows = n_lo + self.hp.n_vocab - hi;
            h2.native_ptr(dev, &nk, xl, a(&mt.logits), 1, HIDDEN, rows);
            let (lg, n, part) = (a(&mt.logits), rows as i32, a(&mt.short));
            launch(
                self.k.m_short1,
                (64, 1, 1),
                1024,
                tang_moe::args![lg, n, part],
            );
            let (np, hd, rbi, ids, pr, nl, hb) = (
                64i32,
                a(exact),
                *rb as i32,
                a(&mt.drafts[step]),
                a(&mt.probs[step]),
                *n_lo as i32,
                *hi as i32,
            );
            let ex = part + (64 * 10 * 4) as u64;
            launch(
                self.k.m_short2a,
                (32, 1, 1),
                256,
                tang_moe::args![part, np, hd, rbi, xl, ex],
            );
            launch(
                self.k.m_short2b,
                (1, 1, 1),
                256,
                tang_moe::args![part, np, ex, ids, pr, nl, hb],
            );
            return;
        }
        let (v, n_lo, hi_base) = match &mt.dhead {
            Some((h, n_lo, hi)) => {
                let rows = n_lo + self.hp.n_vocab - hi;
                h.native_ptr(dev, &nk, xl, a(&mt.logits), 1, HIDDEN, rows);
                (rows, *n_lo as i32, *hi as i32)
            }
            None => {
                let v = self.hp.n_vocab;
                self.head
                    .native_ptr(dev, &nk, xl, a(&mt.logits), 1, HIDDEN, v);
                (v, v as i32, 0)
            }
        };
        {
            let (lg, n, part, np, ids, pr) = (
                a(&mt.logits),
                v as i32,
                a(&mt.amax),
                64i32,
                a(&mt.drafts[step]),
                a(&mt.probs[step]),
            );
            launch(
                self.k.m_amax1,
                (64, 1, 1),
                1024,
                tang_moe::args![lg, n, part],
            );
            launch(
                self.k.m_amax2,
                (1, 1, 1),
                32,
                tang_moe::args![part, np, ids, pr, n_lo, hi_base],
            );
        }
    }

    /// The whole MTP pass for `c` teacher-forced cells (inputs staged in `Io::MTP_IN`): cells,
    /// then two chain steps, then the drafts and probabilities to `Io::MTP_OUT`.
    fn enqueue_mtp(&mut self, c: usize) {
        let mt = self.mtp.as_ref().expect("MTP loaded");
        let a = |b: &B| self.dev.buffer_addr(b);
        self.kcopy(a(&mt.toks), self.io_dev(Io::MTP_IN), 4 * MAX_T);
        self.kcopy(a(&mt.ctl[0]), self.io_dev(Io::MTP_IN + 64), 16);
        self.kcopy(a(&mt.h), a(&self.s.r), c * HC * HIDDEN * 4);
        self.kcopy(a(&mt.e), self.io_dev(Io::MTP_EMB), c * HIDDEN * 4);
        self.enqueue_mtp_cells(0, c);
        let steps = self.mtp.as_ref().unwrap().steps;
        for step in 1..steps {
            {
                let mt = self.mtp.as_ref().unwrap();
                let a = |b: &B| self.dev.buffer_addr(b);
                let (r, h, d, tk, cp, cn) = (
                    a(&mt.r),
                    a(&mt.h),
                    a(&mt.drafts[step - 1]),
                    a(&mt.toks),
                    a(&mt.ctl[step - 1]),
                    a(&mt.ctl[step]),
                );
                unsafe {
                    gpu::launch(
                        self.k.m_next,
                        (20, 1, 1),
                        (512, 1, 1),
                        0,
                        &self.stream,
                        tang_moe::args![r, h, d, tk, cp, cn],
                    )
                    .expect("launch")
                };
            }
            self.enqueue_mtp_cells(step, 1);
        }
        let mt = self.mtp.as_ref().unwrap();
        for step in 0..steps {
            self.kcopy(
                self.io_dev(Io::MTP_OUT + 64 * step),
                self.dev.buffer_addr(&mt.drafts[step]),
                4 * MAX_T,
            );
            self.kcopy(
                self.io_dev(Io::MTP_OUT + 64 * step + 32),
                self.dev.buffer_addr(&mt.probs[step]),
                4 * MAX_T,
            );
        }
    }

    /// Capture every graph decoding can use (windows and commits of T = 1..8, the MTP pass for
    /// 1..8 cells) on a throwaway sequence, so no capture lands inside a timed run.
    pub fn warm(&mut self) -> Result<()> {
        let use_mtp = self.use_mtp;
        self.use_mtp = self.mtp.is_some();
        self.reset();
        let ids: Vec<u32> = (0..16).map(|i| 1000 + i).collect();
        let mut cur = self.prefill(&ids, MAX_T, None)?;
        for t in 1..=MAX_T {
            let d: Vec<u32> = (0..t - 1).map(|i| 7 + i as u32).collect();
            cur = *self.verify(cur, &d)?.last().unwrap();
        }
        if self.mtp.is_some() {
            let pos = self.tokens.len() - 1;
            for c in 1..=MAX_T {
                self.mtp_draft(pos.saturating_sub(c), &vec![cur; c])?;
                self.mtp_draft(pos.saturating_sub(c), &vec![cur; c])?;
            }
        }
        self.use_mtp = use_mtp;
        self.reset();
        Ok(())
    }

    fn mtp_stage(&mut self, pos0: usize, next: &[u32]) {
        let c = next.len();
        if let Ok(emb_t) = self.g.info("token_embd.weight").cloned() {
            let e = self.io.f32s(Io::MTP_EMB, c * HIDDEN);
            for (i, &tok) in next.iter().enumerate() {
                if let Ok(row) = self.g.rows(&emb_t, tok as usize, 1) {
                    e[i * HIDDEN..(i + 1) * HIDDEN].copy_from_slice(&row);
                }
            }
        }
        let tk = self.io.u32s(Io::MTP_IN, MAX_T);
        tk[..c].copy_from_slice(next);
        let ctl = self.io.u32s(Io::MTP_IN + 64, 2);
        ctl[0] = pos0 as u32;
        ctl[1] = c as u32;
    }

    fn mtp_read(&self, _c: usize) -> Vec<(u32, f32)> {
        (0..self.mtp.as_ref().map_or(0, |m| m.steps))
            .map(|step| {
                let d = self.io.u32s(Io::MTP_OUT + 64 * step, MAX_T)[0];
                let p = f32::from_bits(self.io.u32s(Io::MTP_OUT + 64 * step + 32, MAX_T)[0]);
                (d, p)
            })
            .collect()
    }

    /// Switch chain length between requests. Captured MTP graphs include chain length;
    /// retain a separate bank per length so panel runs never reuse a mismatched graph.
    pub fn set_mtp_steps(&mut self, steps: usize) -> Result<()> {
        ensure!(
            (1..=super::mtp_gpu::MAX_STEPS).contains(&steps),
            "invalid MTP steps"
        );
        self.side.sync().map_err(|e| anyhow!("{e}"))?;
        self.dev.sync();
        let m = self.mtp.as_mut().context("MTP layer not loaded")?;
        if m.steps != steps {
            let next = m
                .graph_banks
                .remove(&steps)
                .unwrap_or_else(|| (0..=MAX_T).map(|_| None).collect());
            let previous = std::mem::replace(&mut m.graphs, next);
            m.graph_banks.insert(m.steps, previous);
            m.steps = steps;
        }
        Ok(())
    }

    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    /// Run the MTP over the `c` positions `pos0..pos0 + c` the last window kept (their final
    /// residuals are still in the window's `r`), with `next[i]` the token after position
    /// `pos0 + i`. Returns the chain's drafts and probabilities (3 each).
    pub fn mtp_draft(&mut self, pos0: usize, next: &[u32]) -> Result<Vec<(u32, f32)>> {
        let c = next.len();
        ensure!((1..=MAX_T).contains(&c) && self.mtp.is_some());
        let t0 = Instant::now();
        self.mtp_stage(pos0, next);
        let ready = self.mtp.as_ref().unwrap().graphs[c].is_some();
        if std::env::var("TANG_FLASH_MTP_TIMING").is_ok() {
            self.dev.sync();
            self.last_commit_ms = t0.elapsed().as_secs_f64() * 1e3;
        }
        if self.use_graphs && ready && std::env::var("TANG_FLASH_MTP_TIMING").is_ok() {
            let g = self.mtp.as_ref().unwrap().graphs[c].as_ref().unwrap() as *const Graph;
            let stream = ManuallyDrop::new(Stream(self.dev.cu_stream()));
            self.last_mtp_gpu_ms = self
                .dev
                .event_ms(&mut || unsafe { &*g }.launch(&stream).expect("mtp graph"))
                as f64;
        } else if self.use_graphs && ready {
            self.mtp.as_ref().unwrap().graphs[c]
                .as_ref()
                .unwrap()
                .launch(&self.stream)
                .map_err(|e| anyhow!("{e}"))?;
        } else {
            self.enqueue_mtp(c);
        }
        self.dev.sync();
        let out = self.mtp_read(c);
        self.last_mtp_ms = t0.elapsed().as_secs_f64() * 1e3;
        if self.use_graphs && !ready {
            let stream = ManuallyDrop::new(Stream(self.dev.cu_stream()));
            let me = self as *mut Self;
            let g = Graph::capture(&stream, |_| {
                // SAFETY: `self` is not otherwise touched during capture.
                unsafe { &mut *me }.enqueue_mtp(c);
                Ok(())
            })
            .map_err(|e| anyhow!("capture MTP c={c}: {e}"))?;
            self.mtp.as_mut().unwrap().graphs[c] = Some(g);
        }
        Ok(out)
    }
}

/// The running (non-positional) state after `pos` tokens: GDN state and conv history, the QSA
/// indexer-key rings and the PLE ring (`docs/strata.md`, "State, KV, conversations"). Positional
/// state (QSA K/V, pooled indexer keys, MTP K/V) is rewritten before it is read, so restoring
/// this plus the token prefix resumes the sequence at `pos` (the server's prefix reuse).
pub struct RunningState {
    pub pos: usize,
    gdn: Vec<(Vec<f32>, Vec<f32>)>,
    qsa_ring: Vec<Vec<f32>>,
    ple_ring: Vec<f32>,
}

impl RunningState {
    /// Host bytes held.
    pub fn bytes(&self) -> usize {
        4 * (self
            .gdn
            .iter()
            .map(|(a, b)| a.len() + b.len())
            .sum::<usize>()
            + self.qsa_ring.iter().map(Vec::len).sum::<usize>()
            + self.ple_ring.len())
    }

    /// Every buffer, in a fixed order (for saving to disk).
    pub fn parts(&self) -> Vec<&[f32]> {
        let mut v: Vec<&[f32]> = Vec::new();
        for (s, h) in &self.gdn {
            v.push(s);
            v.push(h);
        }
        v.extend(self.qsa_ring.iter().map(Vec::as_slice));
        v.push(&self.ple_ring);
        v
    }

    /// GDN layers in the state (the first `2 ×` this many [`parts`](Self::parts) are theirs).
    pub fn gdn_layers(&self) -> usize {
        self.gdn.len()
    }

    /// The inverse of [`parts`](Self::parts), for a model with `gdn_layers` GDN layers.
    pub fn from_parts(pos: usize, gdn_layers: usize, mut parts: Vec<Vec<f32>>) -> Result<Self> {
        ensure!(
            parts.len() > 2 * gdn_layers,
            "state has {} parts",
            parts.len()
        );
        let ple_ring = parts.pop().unwrap();
        let qsa_ring = parts.split_off(2 * gdn_layers);
        let mut gdn = Vec::new();
        let mut it = parts.into_iter();
        while let (Some(s), Some(h)) = (it.next(), it.next()) {
            gdn.push((s, h));
        }
        Ok(RunningState {
            pos,
            gdn,
            qsa_ring,
            ple_ring,
        })
    }
}

impl Engine {
    /// Copy the running state out (after the last window's commit). The sequence is `tokens`.
    pub fn save_running(&mut self) -> Result<RunningState> {
        self.side.sync().map_err(|e| anyhow!("{e}"))?;
        self.dev.sync();
        let dev = &self.dev;
        let gdn = self
            .layers
            .iter()
            .filter_map(|l| l.gdn.as_ref())
            .map(|g| (dev.download(&g.state), dev.download(&g.hist)))
            .collect();
        let qsa_ring = self
            .layers
            .iter()
            .filter_map(|l| l.qsa.as_ref())
            .map(|q| dev.download(&q.ring))
            .collect();
        Ok(RunningState {
            pos: self.tokens.len(),
            gdn,
            qsa_ring,
            ple_ring: dev.download(&self.ple.ring),
        })
    }

    /// Resume at `s.pos` with `prefix` (`s.pos` tokens, the sequence `s` was saved after). MTP
    /// drafts are dropped until the next window.
    pub fn restore_running(&mut self, s: &RunningState, prefix: &[u32]) -> Result<()> {
        ensure!(
            prefix.len() == s.pos,
            "prefix of {} tokens for a state at {}",
            prefix.len(),
            s.pos
        );
        self.side.sync().map_err(|e| anyhow!("{e}"))?;
        self.dev.sync();
        let mut gi = 0;
        let mut qi = 0;
        for l in &mut self.layers {
            if let Some(g) = l.gdn.as_mut() {
                let (st, h) = s.gdn.get(gi).context("state has fewer GDN layers")?;
                self.dev.upload_into_f32(&mut g.state, st);
                self.dev.upload_into_f32(&mut g.hist, h);
                gi += 1;
            }
            if let Some(q) = l.qsa.as_mut() {
                self.dev.upload_into_f32(
                    &mut q.ring,
                    s.qsa_ring.get(qi).context("state has fewer QSA layers")?,
                );
                qi += 1;
            }
        }
        ensure!(
            gi == s.gdn.len() && qi == s.qsa_ring.len(),
            "state is for another model"
        );
        self.dev.upload_into_f32(&mut self.ple.ring, &s.ple_ring);
        self.dev.sync();
        self.tokens.clear();
        self.tokens.extend_from_slice(prefix);
        self.mtp_last.clear();
        Ok(())
    }
}

/// Positional state (QSA K/V and pooled indexer keys, MTP K/V) for positions `from..to`, as
/// raw bytes per buffer: what a [`RunningState`] needs besides itself when another sequence has
/// since overwritten those positions.
pub struct Positional {
    pub from: usize,
    pub to: usize,
    bufs: Vec<Vec<u8>>,
}

impl Positional {
    pub fn bytes(&self) -> usize {
        self.bufs.iter().map(Vec::len).sum()
    }

    pub fn parts(&self) -> &[Vec<u8>] {
        &self.bufs
    }

    pub fn from_parts(from: usize, to: usize, bufs: Vec<Vec<u8>>) -> Self {
        Positional { from, to, bufs }
    }
}

impl Engine {
    /// Every positional buffer: (device address, positions per row, row bytes).
    fn positional_bufs(&self) -> Vec<(u64, usize, usize)> {
        let a = |b: &B| self.dev.buffer_addr(b);
        let mut v = Vec::new();
        for q in self.layers.iter().filter_map(|l| l.qsa.as_ref()) {
            v.push((a(&q.k), 1, QSA_KV * QSA_D * 2));
            v.push((a(&q.v), 1, QSA_KV * QSA_D * 2));
            v.push((a(&q.pooled), IDX_BLOCK, IDX_D * 4));
        }
        if let Some(m) = self.mtp.as_ref() {
            v.push((a(&m.kc), 1, QSA_KV * QSA_D * 4));
            v.push((a(&m.vc), 1, QSA_KV * QSA_D * 4));
        }
        v
    }

    /// Copy positions `from..to` of the positional state out.
    pub fn save_positional(&mut self, from: usize, to: usize) -> Result<Positional> {
        ensure!(from <= to && to <= self.opts.max_ctx);
        self.side.sync().map_err(|e| anyhow!("{e}"))?;
        self.dev.sync();
        let mut bufs = Vec::new();
        for (addr, per, row) in self.positional_bufs() {
            let (r0, r1) = (from / per, to / per);
            let mut host = vec![0u8; (r1 - r0) * row];
            if !host.is_empty() {
                let r = unsafe {
                    cudarc::driver::sys::cuMemcpyDtoH_v2(
                        host.as_mut_ptr() as *mut _,
                        addr + (r0 * row) as u64,
                        host.len(),
                    )
                };
                ensure!(
                    r == cudarc::driver::sys::CUresult::CUDA_SUCCESS,
                    "cuMemcpyDtoH: {r:?}"
                );
            }
            bufs.push(host);
        }
        Ok(Positional { from, to, bufs })
    }

    /// Write positions `from.max(p.from)..p.to` of `p` back.
    pub fn restore_positional(&mut self, p: &Positional, from: usize) -> Result<()> {
        let bufs = self.positional_bufs();
        ensure!(
            bufs.len() == p.bufs.len(),
            "positional state is for another model"
        );
        self.side.sync().map_err(|e| anyhow!("{e}"))?;
        self.dev.sync();
        let from = from.max(p.from);
        for ((addr, per, row), host) in bufs.into_iter().zip(&p.bufs) {
            let (r0, r1, base) = (from / per, p.to / per, p.from / per);
            ensure!(
                host.len() == (r1 - base) * row,
                "positional state size differs"
            );
            if r1 > r0 {
                let src = &host[(r0 - base) * row..];
                let r = unsafe {
                    cudarc::driver::sys::cuMemcpyHtoD_v2(
                        addr + (r0 * row) as u64,
                        src.as_ptr() as *const _,
                        (r1 - r0) * row,
                    )
                };
                ensure!(
                    r == cudarc::driver::sys::CUresult::CUDA_SUCCESS,
                    "cuMemcpyHtoD: {r:?}"
                );
            }
        }
        Ok(())
    }
}

impl Engine {
    /// Experts per layer the GPU reads from the host arena itself (`pcie_cap`, at most
    /// `STAGE_MAX` when staged).
    fn pcie_share(&self) -> usize {
        if self.s.stage.is_some() {
            self.pcie_cap.min(STAGE_MAX)
        } else {
            self.pcie_cap
        }
    }

    /// Forget a captured window graph (recaptured on its next use), e.g. after changing
    /// `pcie_cap`, which graphs bake in.
    pub fn drop_window_graph(&mut self, t: usize, verify: bool) {
        if let Some(g) = self.graphs.get_mut(t + if verify { MAX_T + 1 } else { 0 }) {
            *g = None;
        }
    }
}

impl Engine {
    /// Sample in the head with top-k / top-p / a presence penalty (`flash::sampler`) instead of
    /// the plain Gumbel-max over the whole vocabulary. Window graphs are recaptured.
    pub fn enable_sampler(&mut self) -> Result<()> {
        if self.sampler.is_none() {
            self.sampler = Some(Box::new(super::sampler::Sampler::new(
                &self.gpu,
                &self.dev,
                self.hp.n_vocab,
            )?));
            for g in self.graphs.iter_mut() {
                *g = None;
            }
        }
        Ok(())
    }

    pub fn sampler_mut(&mut self) -> Option<&mut super::sampler::Sampler> {
        self.sampler.as_deref_mut()
    }

    /// A new generation's output starts at position `pos` (penalties count output from there).
    pub fn sampler_begin(&mut self, pos: usize) -> Result<()> {
        match self.sampler.as_mut() {
            Some(s) => s.begin(&self.dev, pos),
            None => Ok(()),
        }
    }
}

impl Engine {
    /// A prefill window of `t` (`MAX_T` < t <= `WIDE`) tokens `tokens[pos0..pos0 + t]`, every
    /// token kept, with its own graph per width (decode's graphs untouched). Routed experts
    /// missing from VRAM are read by the GPU from the mapped host arena (no doorbell, no CPU
    /// rows), and there is no head: for every prompt window but the last.
    pub fn prefill_wide(&mut self, pos0: usize, t: usize) -> Result<WinStats> {
        ensure!(t > MAX_T && t <= WIDE && pos0 + t <= self.tokens.len());
        ensure!(
            pos0 + t <= self.opts.max_ctx,
            "context {} > max {}",
            pos0 + t,
            self.opts.max_ctx
        );
        let w0 = Instant::now();
        let mut st = WinStats {
            t,
            ..Default::default()
        };
        self.stage(pos0, t)?;
        self.keys.clear();
        self.wide = true;
        let r = (|| -> Result<()> {
            if let Some(g) = self.wide_graphs.get(&t) {
                g.launch(&self.stream).map_err(|e| anyhow!("{e}"))?;
                self.start_gather(pos0, t);
                for l in 0..self.layers.len() {
                    self.serve(l, &mut st)?;
                }
                Ok(())
            } else {
                self.start_gather(pos0, t);
                let mut f = |me: &mut Self, l: usize| me.serve(l, &mut st);
                self.enqueue_window(t, false, false, &mut f)
            }
        })();
        self.wide = false;
        r?;
        self.dev.sync();
        if let Some(e) = self.gather_err.lock().unwrap().take() {
            bail!("n-gram rows: {e}");
        }
        // Routing from the ids copied out per layer: statistics and the cache's adaptation.
        for l in 0..self.layers.len() {
            let mut d: Vec<u32> = self
                .io
                .u32s(Io::LIDS + l * 4 * WIDE * TOPK, t * TOPK)
                .to_vec();
            d.sort_unstable();
            d.dedup();
            st.routed += t * TOPK;
            st.distinct += d.len();
            let base = (l * EXPERTS) as u32;
            for &e in &d {
                self.routing[(base + e) as usize] += 1;
                self.keys.push(base + e);
            }
        }
        st.swaps = self.boundary()?;
        self.counter = self.counter.wrapping_add(1);
        if self.use_graphs && !self.wide_graphs.contains_key(&t) {
            let stream = ManuallyDrop::new(Stream(self.dev.cu_stream()));
            let me = self as *mut Self;
            self.wide = true;
            let g = Graph::capture(&stream, |_| {
                // SAFETY: `self` is not otherwise touched during capture.
                let me = unsafe { &mut *me };
                me.enqueue_window(t, false, false, &mut |_, _| Ok(()))
                    .map_err(|e| tang_moe::gpu::Error(format!("{e}")))
            });
            self.wide = false;
            self.wide_graphs
                .insert(t, g.map_err(|e| anyhow!("capture wide T={t}: {e}"))?);
        }
        st.wall_ms = w0.elapsed().as_secs_f64() * 1e3;
        Ok(st)
    }
}
