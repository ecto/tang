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
use crate::gguf::Gguf;
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
        }
    }
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
    let rb = segs.iter().filter(|s| s[0] != 30).map(|s| s[2]).max().unwrap_or(0);
    ((rb + 15).div_ceil(16) + 1) as i32
}

impl Dw {
    #[allow(clippy::too_many_arguments)]
    fn apply(&self, dev: &CudaComputeDevice, nk: &NativeK, x: &B, xq: &B, out: &mut B, t: usize, k: usize, n: usize) {
        match self {
            Dw::Bf16(b) => dev.linear_into(x, b, out, t, k, n),
            Dw::Q4x(b) => dev.q4x_linear_into(xq, b, out, t, k, n),
            Dw::Native { segs, nseg, rows, row16, .. } => {
                assert_eq!(*rows, n, "native GEMV rows");
                let (sp, ns, xp, ki, op, os) = (
                    dev.buffer_addr(segs),
                    *nseg,
                    dev.buffer_addr(x),
                    k as i32,
                    dev.buffer_addr(out),
                    n as i32,
                );
                let s = ManuallyDrop::new(Stream(nk.stream));
                let r16 = *row16;
                let xq = nk.xq8;
                unsafe {
                    let (ki, ti) = (k as i32, t as i32);
                    let _ = ti;
                    gpu::launch(nk.q8, ((k / 32) as u32, t as u32, 1), (32, 1, 1), 0, &s, tang_moe::args![xp, xq, ki])
                        .expect("fe_q8 launch");
                    gpu::launch(
                        nk.f8[t - 1],
                        (n.div_ceil(8) as u32, 1, 1),
                        (128, 1, 1),
                        (8 * 16 * r16) as u32,
                        &s,
                        tang_moe::args![sp, ns, xp, xq, ki, op, os, r16],
                    )
                    .expect("fe_gemv launch")
                };
            }
        }
    }

    fn native(&self) -> bool {
        matches!(self, Dw::Native { .. })
    }
}

struct Hc {
    norm: B,
    down: B,
    up: B,
    inject: Option<B>,
}

impl Hc {
    fn w(&self) -> HcWeights<'_, B> {
        HcWeights {
            norm: &self.norm,
            down: &self.down,
            up: &self.up,
            inject: self.inject.as_ref(),
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
    router: B,
    sh_gu: Dw,
    sh_down: Dw,
    gdn: Option<Gdn>,
    qsa: Option<Qsa>,
}

struct Ple {
    layer: usize,
    key: B,
    key_q2: bool,
    value: B,
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
    stamps: B,
}

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
    const CTL: usize = 0;
    const EMB: usize = 64;
    const PLE: usize = Self::EMB + MAX_T * HIDDEN * 4;
    const IDS: usize = Self::PLE + MAX_T * HIDDEN * 4;
    const STAMPS: usize = Self::IDS + 64;
    const BYTES: usize = Self::STAMPS + 48 * 8 * 8;

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
    pub routed: usize,
    pub distinct: usize,
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
    ngram: NgramTable,
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
    counter: u32,
    /// The sequence so far (PLE needs its predecessors).
    pub tokens: Vec<u32>,
    /// Routing counts per key, for `--dump-routing`.
    pub routing: Vec<u32>,
    /// Last window's per-layer distinct routed experts (keys), for the cache.
    keys: Vec<u32>,
    /// Per layer, the device residency table `moe_route_into` plans from (`EXPERTS` addresses,
    /// 0 = on the host), mirrored from the cache's table after each boundary.
    tables: Vec<B>,
    /// The addresses `tables` holds now (host copy, to find layers that changed).
    table_addrs: Vec<u64>,
    host_plan: Vec<u32>,
    pub use_graphs: bool,
    pub last: WinStats,
    pub n_slots: usize,
    pub load_report: String,
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
        Fmt::Q4x => dev.upload_bytes(b),
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
        let warm = dir.join(format!("dense.{}.json", pack::dense_tag())).exists()
            && dir.join("experts.bin").exists();
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
            let e = by.get(name).with_context(|| format!("pack has no {name}"))?;
            let b = pack::read_entry(&df, e)?;
            Ok(upload_entry(&dev, e, &b))
        };
        let dw = |name: &str| -> Result<Dw> {
            let e = by.get(name).with_context(|| format!("pack has no {name}"))?;
            let w = get(name)?;
            Ok(match e.fmt {
                Fmt::Bf16 => Dw::Bf16(w),
                Fmt::Q4x => Dw::Q4x(w),
                Fmt::Native => {
                    let base = dev.buffer_addr(&w);
                    let words: Vec<u32> = e
                        .segs
                        .iter()
                        .flat_map(|&[ty, rows, rb, off, out]| {
                            let p = base + off;
                            [p as u32, (p >> 32) as u32, ty as u32, rows as u32, rb as u32, out as u32]
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
        let hc = |key: &str, inject: bool| -> Result<Hc> {
            Ok(Hc {
                norm: get(&format!("{key}.norm"))?,
                down: get(&format!("{key}.down"))?,
                up: get(&format!("{key}.up"))?,
                inject: if inject {
                    Some(get(&format!("{key}.inject"))?)
                } else {
                    None
                },
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
                    proj: dev.alloc_f32(MAX_T * GDN_PROJ),
                    h: dev.alloc_f32(MAX_T * GDN_CONV),
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
                    ring: dev.alloc_f32(16 * IDX_D),
                    pooled: dev.alloc_f32(max_ctx / IDX_BLOCK * IDX_D),
                    proj: dev.alloc_f32(MAX_T * QSA_PROJ),
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
                router: get(&format!("{k}.router"))?,
                sh_gu: dw(&format!("{k}.sh_gu"))?,
                sh_down: dw(&format!("{k}.sh_down"))?,
                gdn,
                qsa,
            });
        }
        let ple = Ple {
            layer: plep.layer,
            key_q2: by.get("ple.key").is_some_and(|e| e.fmt == Fmt::Q2Raw),
            key: get("ple.key")?,
            value: get("ple.value")?,
            nk: get("ple.nk")?,
            nq: get("ple.nq")?,
            nc: get("ple.nc")?,
            conv: get("ple.conv")?,
            ring: dev.alloc_f32(16 * HC * HIDDEN),
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
            ctl: z(4),
            emb: z(MAX_T * HIDDEN),
            ple_e: z(MAX_T * HIDDEN),
            ple_eq: z(QAct { m: MAX_T, k: HIDDEN }.words()),
            ple_key: z(MAX_T * HC * HIDDEN),
            ple_val: z(MAX_T * HIDDEN),
            r: z(MAX_T * HC * HIDDEN),
            x: z(MAX_T * HIDDEN),
            x2: z(MAX_T * HIDDEN),
            xq: z(QAct { m: MAX_T, k: HIDDEN }.words()),
            inj_a: z(MAX_T * HC),
            inj_f: z(MAX_T * HC),
            hc: z(fl::hc_scratch_words(MAX_T)),
            side: z(MAX_T * (IDX_HEADS * IDX_D + IDX_D)),
            y: z(MAX_T * GDN_V),
            yq: z(QAct { m: MAX_T, k: GDN_V }.words()),
            mix: z(MAX_T * HIDDEN),
            q: z(fl::qsa_q_words(MAX_T)),
            scores: z(MAX_T * max_ctx / IDX_BLOCK),
            sel: z(MAX_T * QSA_WIDTH),
            attn_s: z(fl::qsa_attend_scratch_words(MAX_T)),
            attn: z(MAX_T * QSA_OUT),
            logits: z(MAX_T * ROUTER_ROWS),
            ids: z(MAX_T * TOPK),
            w: z(MAX_T * TOPK),
            plan: z(MoePlan::WORDS),
            list: z(1 + MoePlan::CAP),
            moe: z(MoePlan::scratch_words()),
            parts: z(MoePlan::PARTS_ROWS * HIDDEN),
            gu: z(MAX_T * 2 * FF),
            hf: z(MAX_T * FF),
            hq: z(QAct { m: MAX_T, k: FF }.words()),
            shy: z(MAX_T * HIDDEN),
            head: z(MAX_T * hp.n_vocab),
            out_ids: z(MAX_T),
            stamps: z(48 * 8 * 2),
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
        let k = Kern {
            embed: f(&m, "fe_embed")?,
            ple: f(&m, "fe_ple")?,
            silu_q: f(&m, "fe_silu_q")?,
            copy: f(&m, "fe_copy")?,
            scatter: f(&m, "fe_scatter_cols")?,
            argmax: f(&m, "fe_argmax")?,
            publish: f(&db, "db_publish")?,
            wait: f(&db, "db_wait")?,
            copy_rows: f(&db, "db_copy_rows")?,
            stamp: f(&db, "stamp")?,
            _m: m,
            _db: db,
        };
        let gm = gpu.module(kernels::GEMV_SRC).map_err(|e| anyhow!("{e}"))?;
        let names = ["fe_gemv_t1", "fe_gemv_t2", "fe_gemv_t3", "fe_gemv_t4", "fe_gemv_t5", "fe_gemv_t6", "fe_gemv_t7", "fe_gemv_t8"];
        let mut fs = [std::ptr::null_mut(); MAX_T];
        for (i, n) in names.iter().enumerate() {
            fs[i] = f(&gm, n)?;
        }
        let mut f8s = [std::ptr::null_mut(); MAX_T];
        for (i, slot) in f8s.iter_mut().enumerate() {
            *slot = f(&gm, ["fe_gemv8_t1", "fe_gemv8_t2", "fe_gemv8_t3", "fe_gemv8_t4", "fe_gemv8_t5", "fe_gemv8_t6", "fe_gemv8_t7", "fe_gemv8_t8"][i])?;
        }
        let xq8_buf = dev.alloc_f32(XQ8_BYTES / 4);
        let nk = NativeK {
            f: fs,
            f8: f8s,
            q8: f(&gm, "fe_q8")?,
            xq8: dev.buffer_addr(&xq8_buf),
            stream: dev.cu_stream(),
        };
        let mb = Mailbox::new(&gpu).map_err(|e| anyhow!("{e}"))?;
        let mut io = Io {
            arena: HostArena::new(Io::BYTES, ArenaOptions { try_hugetlb: false, thp: false })?,
        };
        io.arena
            .register(&gpu.ctx, &[])
            .map_err(|e| anyhow!("{e}"))?;
        let stream = ManuallyDrop::new(Stream(dev.cu_stream()));

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
                let budget = free.saturating_sub((opts.reserve_mb + opts.mtp_mb) << 20);
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
                    AdaptParams::default()
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
        let tables: Vec<B> = (0..hp.n_layer).map(|_| dev.alloc_f32(2 * EXPERTS)).collect();
        let mut exec = MissExec::new(Pool::new(&Pool::default_cpus(), std::time::Duration::from_millis(20)), Isa::detect());
        exec.tiled = true;
        let ngram = NgramTable::open(&g, &plep)?;
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
            graphs: (0..=MAX_T).map(|_| None).collect(),
            counter: 1,
            tokens: Vec::new(),
            routing: vec![0; 48 * EXPERTS],
            keys: Vec::new(),
            tables,
            table_addrs: vec![u64::MAX; 48 * EXPERTS],
            host_plan: vec![0; MoePlan::WORDS],
            use_graphs: true,
            last: WinStats::default(),
            n_slots,
            load_report,
        };
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
        ensure!(b.len() == n_layer * EXPERTS * 4, "profile {} has the wrong size", p.display());
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

    unsafe fn launch(&self, f: Fun, grid: (u32, u32, u32), block: u32, args: &mut [*mut std::ffi::c_void]) {
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

    /// Copy the window's inputs from pinned staging (graph nodes).
    fn enqueue_inputs(&self, t: usize) {
        let st = self.stream.0;
        let cp = |dst: &B, off: usize, bytes: usize| unsafe {
            gpu::check(
                cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                    Self::addr(dst, &self.dev),
                    self.io.ptr(off) as *const std::ffi::c_void,
                    bytes,
                    st,
                ),
                "cuMemcpyHtoDAsync",
            )
            .expect("input copy")
        };
        cp(&self.s.ctl, Io::CTL, 16);
        cp(&self.s.emb, Io::EMB, t * HIDDEN * 4);
        cp(&self.s.ple_e, Io::PLE, t * HIDDEN * 4);
    }

    fn enqueue_outputs(&self, t: usize) {
        unsafe {
            gpu::check(
                cudarc::driver::sys::cuMemcpyDtoHAsync_v2(
                    self.io.ptr(Io::IDS) as *mut std::ffi::c_void,
                    Self::addr(&self.s.out_ids, &self.dev),
                    t * 4,
                    self.stream.0,
                ),
                "cuMemcpyDtoHAsync",
            )
            .expect("output copy");
            if self.opts.split {
                gpu::check(
                    cudarc::driver::sys::cuMemcpyDtoHAsync_v2(
                        self.io.ptr(Io::STAMPS) as *mut std::ffi::c_void,
                        Self::addr(&self.s.stamps, &self.dev),
                        48 * 8 * 8,
                        self.stream.0,
                    ),
                    "cuMemcpyDtoHAsync",
                )
                .expect("stamp copy");
            }
        }
    }

    /// The embedding into every stream.
    fn enqueue_embed(&mut self, t: usize) {
        let (r, e) = (Self::addr(&self.s.r, &self.dev), Self::addr(&self.s.emb, &self.dev));
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
        let dev = &self.dev;
        let s = &mut self.s;
        dev.quantize_act_into(&s.ple_e, &mut s.ple_eq, t, HIDDEN);
        if self.ple.key_q2 {
            dev.q2_linear_into(&s.ple_eq, &self.ple.key, &mut s.ple_key, t, HIDDEN, HC * HIDDEN);
        } else {
            dev.q4x_linear_into(&s.ple_eq, &self.ple.key, &mut s.ple_key, t, HIDDEN, HC * HIDDEN);
        }
        dev.linear_into(&s.ple_e, &self.ple.value, &mut s.ple_val, t, HIDDEN, HIDDEN);
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
    fn enqueue_layer(&mut self, l: usize, t: usize, fused: bool) {
        let dev = &self.dev;
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
        layer.w_in.apply(dev, &self.nk, &s.x, &s.xq, proj, t, HIDDEN, layer.in_rows);
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
                GdnMode::Commit { win: &s.ctl },
                EPS,
            );
            dev.gdn_conv_commit(&mut g.hist, &g.proj, GDN_PROJ, &s.ctl, t);
            layer.w_out.apply(dev, &self.nk, &s.y, &s.yq, &mut s.mix, t, GDN_V, HIDDEN);
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
            dev.qsa_select_into(&q.pooled, &s.q, &s.ctl, &mut s.scores, &mut s.sel, max_blocks, t);
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
            layer.w_out.apply(dev, &self.nk, &s.attn, &s.yq, &mut s.mix, t, QSA_OUT, HIDDEN);
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
        dev.linear_into(&s.x2, &layer.router, &mut s.logits, t, HIDDEN, ROUTER_ROWS);
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
        // Publish ids and activations; the host computes the misses while the GPU runs the
        // hits and the shared expert.
        let mbp = self.mb.device();
        let ctlw = a(&s.ctl) + 8; // window counter word
        let li = l as i32;
        {
            let (ids, xq, xw, ti) = (a(&s.ids), a(&s.xq), QAct { m: t, k: HIDDEN }.words() as i32, t as i32);
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
        // SAFETY: every group address in the plan is a live VRAM slot or scratch blob.
        unsafe { dev.moe_grouped_into(&s.xq, &s.plan, &mut s.moe, &mut s.parts, t) };
        layer.sh_gu.apply(dev, &self.nk, &s.x2, &s.xq, &mut s.gu, t, HIDDEN, 2 * FF);
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
        layer.sh_down.apply(dev, &self.nk, &s.hf, &s.hq, &mut s.shy, t, FF, HIDDEN);
        {
            let (dst, src, n) = (
                a(&s.parts) + (MoePlan::SHARED_ROW * HIDDEN * 4) as u64,
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
        let st = 1 + 4 * l;
        for i in [st, st + 1] {
            if self.opts.split {
                let (p, i) = (a(&s.stamps), i as i32);
                unsafe { gpu::launch(self.k.stamp, (1, 1, 1), (1, 1, 1), 0, &self.stream, tang_moe::args![p, i]).expect("launch") };
            }
        }
        if self.opts.split {
            let (p, i) = (a(&s.stamps), st as i32 + 2);
            unsafe { gpu::launch(self.k.stamp, (1, 1, 1), (1, 1, 1), 0, &self.stream, tang_moe::args![p, i]).expect("launch") };
        }
        {
            let (flag, src, n, dst) = (
                Mb::FLAG_B as i32,
                Mb::CPU_ROWS as i32,
                (1 + MoePlan::CAP) as i32,
                a(&s.list),
            );
            unsafe {
                gpu::launch(
                    self.k.wait,
                    (1, 1, 1),
                    (256, 1, 1),
                    0,
                    &self.stream,
                    tang_moe::args![mbp, flag, ctlw, li, src, n, dst],
                )
                .expect("launch")
            };
        }
        if self.opts.split {
            let (p, i) = (a(&s.stamps), st as i32 + 3);
            unsafe { gpu::launch(self.k.stamp, (1, 1, 1), (1, 1, 1), 0, &self.stream, tang_moe::args![p, i]).expect("launch") };
        }
        {
            let (list, parts) = (a(&s.list), a(&s.parts));
            unsafe {
                gpu::launch(
                    self.k.copy_rows,
                    (MoePlan::CAP as u32, 1, 1),
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
        self.head.apply(dev, &self.nk, &s.x, &s.xq, &mut s.head, t, HIDDEN, v);
        let (lg, ids, n) = (dev.buffer_addr(&s.head), dev.buffer_addr(&s.out_ids), v as i32);
        unsafe {
            gpu::launch(
                self.k.argmax,
                (t as u32, 1, 1),
                (1024, 1, 1),
                0,
                &self.stream,
                tang_moe::args![lg, n, ids],
            )
            .expect("launch")
        };
        self.stamp(1 + 4 * 48);
    }

    /// Everything a window enqueues, layer by layer; `between(l)` runs on the host after layer
    /// `l` is enqueued (eager mode serves it there).
    fn enqueue_window(&mut self, t: usize, unfused: bool, between: &mut dyn FnMut(&mut Self, usize) -> Result<()>) -> Result<()> {
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
            self.enqueue_layer(l, t, fused);
            between(self, l)?;
        }
        if unfused {
            self.apply_moe(t);
        }
        self.enqueue_head(t, !unfused);
        self.enqueue_outputs(t);
        Ok(())
    }

    // ---- host side

    /// Fill the pinned staging for tokens `self.tokens[pos0..pos0 + t]`.
    fn stage(&mut self, pos0: usize, t: usize) -> Result<()> {
        let ctl = self.io.u32s(Io::CTL, 3);
        ctl[0] = pos0 as u32;
        ctl[1] = t as u32;
        ctl[2] = self.counter;
        let emb_t = self.g.info("token_embd.weight")?.clone();
        let emb = self.io.f32s(Io::EMB, t * HIDDEN);
        for i in 0..t {
            let row = self.g.rows(&emb_t, self.tokens[pos0 + i] as usize, 1)?;
            emb[i * HIDDEN..(i + 1) * HIDDEN].copy_from_slice(&row);
        }
        let ple = self.io.f32s(Io::PLE, t * HIDDEN);
        self.ngram.gather(&self.tokens, pos0, pos0 + t, ple)?;
        Ok(())
    }

    /// Serve layer `l` of the current window: plan, raise A, compute misses, raise B.
    fn serve(&mut self, l: usize, st: &mut WinStats) -> Result<()> {
        let seq = doorbell::seq(self.counter, l);
        self.mb.wait_seq(seq, &self.stream).map_err(|e| anyhow!("layer {l}: {e}"))?;
        let t0 = Instant::now();
        let (t, ids, xq) = self.mb.request();
        let ids: Vec<u32> = ids.to_vec();
        let base = (l * EXPERTS) as u32;
        let experts = &self.experts;
        let addr = |e: u32| -> u64 {
            match experts {
                Experts::Resident(rc) => rc.addr(base + e),
                Experts::Cpu(_) => 0,
            }
        };
        let missed = build_plan(&ids, t, addr, 0, &mut self.host_plan);
        let t1 = Instant::now();
        let blob = ExpertBlob::BYTES;
        let jobs: Vec<MissJob> = missed
            .iter()
            .map(|m| MissJob {
                blob: match experts {
                    Experts::Resident(rc) => rc.host_blob(base + m.expert).expect("missed expert has a host copy"),
                    Experts::Cpu(map) => {
                        let o = (base + m.expert) as usize * blob;
                        &map[o..o + blob]
                    }
                },
                toks: &m.toks,
            })
            .collect();
        // SAFETY: ROWS holds PARTS_ROWS rows; the GPU reads them only after FLAG_B.
        unsafe { self.exec.run(xq, t, &jobs, self.mb.rows_ptr()) };
        let dsts: Vec<u32> = missed
            .iter()
            .flat_map(|m| m.toks.iter().map(|&(_, d)| d as u32))
            .collect();
        self.mb.raise_b(seq, &dsts);
        let t2 = Instant::now();
        st.plan_ms += (t1 - t0).as_secs_f64() * 1e3;
        st.cpu_ms += (t2 - t1).as_secs_f64() * 1e3;
        let mut distinct: Vec<u32> = ids[..t * TOPK].to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        st.routed += t * TOPK;
        st.distinct += distinct.len();
        st.missed += missed.len();
        for &e in &distinct {
            self.routing[(base + e) as usize] += 1;
            self.keys.push(base + e);
        }
        Ok(())
    }

    /// Run one window over `self.tokens[pos0..pos0 + t]` (all kept). Returns the argmax of each
    /// position's logits. With `probe`, runs eagerly and unfused and records the last token's
    /// intermediates.
    pub fn window(&mut self, pos0: usize, t: usize, mut probe: Option<&mut Probe>) -> Result<Vec<u32>> {
        ensure!((1..=MAX_T).contains(&t) && pos0 + t <= self.tokens.len());
        ensure!(pos0 + t <= self.opts.max_ctx, "context {} > max {}", pos0 + t, self.opts.max_ctx);
        let w0 = Instant::now();
        let mut st = WinStats {
            t,
            ..Default::default()
        };
        self.stage(pos0, t)?;
        st.host_prep_ms = w0.elapsed().as_secs_f64() * 1e3;
        self.keys.clear();
        let graph_ready = self.graphs[t].is_some();
        if probe.is_none() && self.use_graphs && graph_ready {
            self.graphs[t]
                .as_ref()
                .unwrap()
                .launch(&self.stream)
                .map_err(|e| anyhow!("{e}"))?;
            for l in 0..self.layers.len() {
                self.serve(l, &mut st)?;
            }
        } else {
            let unfused = probe.is_some();
            let mut f = |me: &mut Self, l: usize| -> Result<()> {
                me.serve(l, &mut st)?;
                if let Some(p) = probe.as_deref_mut() {
                    me.probe_layer(l, t, p)?;
                }
                Ok(())
            };
            self.enqueue_window(t, unfused, &mut f)?;
        }
        self.dev.sync();
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
        // Cache bookkeeping between windows.
        if let Experts::Resident(rc) = &mut self.experts {
            rc.record(&self.keys);
            st.swaps = rc.boundary(&self.stream).map_err(|e| anyhow!("{e}"))?;
        }
        self.sync_tables()?;
        self.counter = self.counter.wrapping_add(1);
        st.wall_ms = w0.elapsed().as_secs_f64() * 1e3;
        self.last = st;
        // Capture this size's graph now that every kernel is loaded.
        if self.use_graphs && !graph_ready {
            self.capture(t)?;
        }
        Ok(ids)
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
                let a = rc.addr((base + e) as u32);
                if self.table_addrs[base + e] != a {
                    self.table_addrs[base + e] = a;
                    dirty = true;
                }
            }
            if dirty {
                unsafe {
                    gpu::check(
                        cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                            self.dev.buffer_addr(&self.tables[l]),
                            rc.device_addrs() + (base * 8) as u64,
                            EXPERTS * 8,
                            self.stream.0,
                        ),
                        "table copy",
                    )
                    .map_err(|e| anyhow!("{e}"))?;
                }
            }
        }
        Ok(())
    }

    fn capture(&mut self, t: usize) -> Result<()> {
        let stream = ManuallyDrop::new(Stream(self.dev.cu_stream()));
        let me = self as *mut Self;
        let g = Graph::capture(&stream, |_| {
            // SAFETY: `self` is not otherwise touched during capture.
            let me = unsafe { &mut *me };
            me.enqueue_window(t, false, &mut |_, _| Ok(()))
                .map_err(|e| tang_moe::gpu::Error(format!("{e}")))
        })
        .map_err(|e| anyhow!("capture T={t}: {e}"))?;
        self.graphs[t] = Some(g);
        Ok(())
    }

    fn probe_layer(&mut self, l: usize, t: usize, p: &mut Probe) -> Result<()> {
        // Eager and unfused: after `serve`, this layer's parts are complete once the stream
        // drains; apply the MoE write on a copy to read post_moe without disturbing the stream.
        self.dev.sync();
        let last = t - 1;
        let row = |v: Vec<f32>, w: usize| v[last * w..(last + 1) * w].to_vec();
        p.mixer_out.push(row(self.dev.download(&self.s.mix), HIDDEN));
        let ids = self.dev.download(&self.s.ids);
        p.router_ids
            .push(ids[last * TOPK..(last + 1) * TOPK].iter().map(|v| v.to_bits()).collect());
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
        p.post_mixer.push(row(self.dev.download(&self.s.r), HC * HIDDEN));
        let mut y = self.dev.alloc_f32(MAX_T * HIDDEN);
        let s = &self.s;
        self.dev.moe_combine_into(&s.parts, &s.w, &s.logits, ROUTER_ROWS, Some(EXPERTS), &mut y, t);
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
            next = out[t - 1];
            pos += t;
        }
        Ok(next)
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
        Ok(a.iter().map(|x| x.as_str().unwrap_or("").to_string()).collect())
    }

    pub fn ngram_stats(&self) -> (u64, u64) {
        use std::sync::atomic::Ordering::Relaxed;
        (self.ngram.hits.load(Relaxed), self.ngram.reads.load(Relaxed))
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
        if t.dims.len() != 2 || t.name.contains("_exps") || t.name.starts_with("per_layer") || t.name == "token_embd.weight" || t.name.contains("hc_") {
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
            let f = gm.func(&format!("fe_gemv_t{tt}")).map_err(|e| anyhow!("{e}"))?;
            let (sp, ns, xp, ki, op, os) = (dev.buffer_addr(&segs), 1i32, dev.buffer_addr(&x), k as i32, dev.buffer_addr(&out), n as i32);
            unsafe {
                { let total = n as i32; gpu::launch(f, (grid, 1, 1), (128, 1, 1), smem, &stream, tang_moe::args![sp, ns, xp, ki, op, os, r16, total]) }
                    .map_err(|e| anyhow!("{e}"))?
            };
            dev.sync();
            let got = dev.download(&out);
            for ti in 0..tt {
                for r in 0..n {
                    let want: f64 = (0..k).map(|i| wf[r * k + i] as f64 * xv[ti * k + i] as f64).sum();
                    let mag: f64 = (0..k).map(|i| (wf[r * k + i] as f64 * xv[ti * k + i] as f64).abs()).sum();
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
            let f = gm.func(&format!("fe_gemv8_t{tt}")).map_err(|e| anyhow!("{e}"))?;
            let (sp, ns, xp, xq, ki, op, os) = (dev.buffer_addr(&segs), 1i32, dev.buffer_addr(&x), dev.buffer_addr(&xqb), k as i32, dev.buffer_addr(&out), n as i32);
            unsafe {
                gpu::launch(q8, ((k / 32) as u32, tt as u32, 1), (32, 1, 1), 0, &stream, tang_moe::args![xp, xq, ki]).map_err(|e| anyhow!("{e}"))?;
                gpu::launch(f, (n.div_ceil(8) as u32, 1, 1), (128, 1, 1), 8 * 16 * r16 as u32, &stream, tang_moe::args![sp, ns, xp, xq, ki, op, os, r16])
                    .map_err(|e| anyhow!("{e}"))?
            };
            dev.sync();
            let got = dev.download(&out);
            let qw: Vec<u8> = dev.download(&xqb).iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
            let xd = |ti: usize, i: usize| -> f64 {
                let q = qw[ti * k + i] as i8 as f64;
                let o = MAX_T * k + 4 * (ti * (k / 32) + i / 32);
                q * f32::from_le_bytes([qw[o], qw[o + 1], qw[o + 2], qw[o + 3]]) as f64
            };
            for ti in 0..tt {
                for r in 0..n {
                    let (mut want, mut mag) = (0f64, 0f64);
                    for i in 0..k {
                        let v = wf[r * k + i] as f64 * if id == 30 { xv[ti * k + i] as f64 } else { xd(ti, i) };
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
            let f = gm.func(&format!("fe_gemv8_t{tt}")).map_err(|e| anyhow!("{e}"))?;
            let (sp, ns, xp, xq, ki, op, os) = (dev.buffer_addr(&segf), 1i32, dev.buffer_addr(&x), dev.buffer_addr(&xqb), k as i32, dev.buffer_addr(&out), nf as i32);
            let mut run = || unsafe {
                gpu::launch(q8, ((k / 32) as u32, tt as u32, 1), (32, 1, 1), 0, &stream, tang_moe::args![xp, xq, ki]).unwrap();
                gpu::launch(f, (nf.div_ceil(8) as u32, 1, 1), (128, 1, 1), 8 * 16 * r16 as u32, &stream, tang_moe::args![sp, ns, xp, xq, ki, op, os, r16]).unwrap()
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
