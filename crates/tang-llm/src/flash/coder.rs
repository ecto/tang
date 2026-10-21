//! The Qwen3-Coder-Next / Qwen3-Next decode engine on CUDA (GGUF architecture `qwen3next`).
//!
//! The same window model as the Flash-Next [`Engine`](super::engine::Engine): `t` = 1..=8
//! consecutive tokens at positions `pos0..pos0 + t`, plain windows commit every token's GDN
//! recurrence in the window, verify windows only read it and [`Coder::commit`] replays the kept
//! prefix. What differs is the model:
//!
//! - one residual stream with pre-norms (`r += mixer(rmsnorm(r))`, `r += moe(rmsnorm(r))`)
//!   instead of hyper-connections;
//! - full-attention layers are gated GQA over every earlier cell (no indexer), split-K;
//! - every routed expert is VRAM-resident in its GGUF format (an 80 GB card holds the
//!   UD-Q4_K_XL file), so there is no doorbell, no CPU miss path and no PLE;
//! - GDN v heads are grouped over k heads (`h / r`) and gated with SiLU (the flash module
//!   compiled for [`Geometry`] with `gdn_grouped`).
//!
//! Dense weights stay in their GGUF types through the native GEMV (`kernels::GEMV_SRC`,
//! int8 activations), so the engine reads exactly the bytes `flash-ref` and llama.cpp read.
//!
//! Per layer, on the stream:
//!
//! ```text
//! rmsnorm → stacked input GEMV → GDN conv + step | attention prep + split-K + gated merge
//! → output GEMV → r += mix, rmsnorm → router GEMV (+ shared gate row) → top-10
//! → int8 → routed gate|up → silu·up → int8 → routed down ‖ shared expert → r += Σ w·y + g·shared
//! ```

use super::coder_kernels::{self as ck, Shapes};
use super::engine::{Probe, WinStats};
use super::kernels;
use super::reference::{Arch, Hparams};
use crate::gguf::{self, GgmlType, Gguf, TensorInfo};
use anyhow::{anyhow, ensure, Context, Result};
use std::collections::HashMap;
use std::mem::ManuallyDrop;
use std::path::Path;
use std::time::Instant;
use tang_compute::cuda::CudaBuffer;
use tang_compute::flash::{self as fl, Geometry};
use tang_compute::{ComputeDevice, CudaComputeDevice};
use tang_moe::arena::{ArenaOptions, HostArena};
use tang_moe::gpu::{self, Gpu, Graph, Module, Stream};

type B = CudaBuffer;
type Fun = cudarc::driver::sys::CUfunction;
/// A window's logits as they are computed: `(pos0, t, [t][vocab])`.
pub type LogitsCb<'a> = &'a mut dyn FnMut(usize, usize, &[f32]);

/// Widest window.
pub const MAX_T: usize = fl::MAX_T;
/// Sequences one engine holds at once (slot 0 is the one the single-sequence API uses).
pub const MAX_SLOTS: usize = 16;
/// u32s of the per-row record staged with each window: positions, slots, then segments.
const ROWS_U32: usize = 6 * MAX_T;

/// Options of a [`Coder`] load.
#[derive(Clone, Debug)]
pub struct CoderOpts {
    pub max_ctx: usize,
    /// An MTP sidecar GGUF (`scripts/qwen3next_mtp_sidecar.py`): its one MTP layer drafts the
    /// next tokens. Default: `TANG_CODER_MTP`.
    pub mtp: Option<std::path::PathBuf>,
}

impl Default for CoderOpts {
    fn default() -> Self {
        CoderOpts {
            max_ctx: 8192,
            mtp: None,
        }
    }
}

/// `TANG_CODER_GEMV=v1`: dense GEMVs on `fe_gemv8` (two rows a warp) instead of the K-split
/// `cx_gemvk` (A/B).
fn env_is(key: &str, val: &str) -> bool {
    std::env::var(key).is_ok_and(|v| v == val)
}

/// Kernel variants for A/B on one build (the defaults are the measured winners; the losers are
/// in `docs/speed-log.md` under Dead ends): `TANG_CODER_KQ` warps a row in the K-split GEMV
/// (1, 2, 4 or 8; default 2), `TANG_CODER_FASTK` the K-quant expert dot (0: 8 weights a lane,
/// 1: 32 a lane, 2: also two down rows a warp; default 2), `TANG_CODER_SHPAR=1` the shared
/// expert on a parallel graph branch.
#[derive(Clone, Copy, Debug)]
struct Variants {
    kq: usize,
    fastk: i32,
    shared_par: bool,
    /// Grouped expert kernels (each distinct expert read once a window), `TANG_CODER_GROUPED=1`.
    /// Bitwise the same outputs as the per-pair ones. Off by default: at decode widths (≤ 8 rows ×
    /// top-10 of 512 experts) pairs rarely share an expert, and on an H200 the plan and the
    /// serialized groups cost more than they save (8 streams: 10.98 ms a step per-pair, 12.49
    /// grouped). They're for wide windows (prefill), where many rows share experts.
    grouped: bool,
    /// Per-pair expert kernels: 0 stage rows in shared memory (`cx_moe_gu` / `cx_moe_down`),
    /// 2 or 4 read them from global memory with that many rows a warp (`_d2` / `_d4`), 8
    /// double-buffer them with cp.async (`_db`); `usize::MAX` (the default) picks by window:
    /// `_d2` up to 4 rows, `_d4` past that. All bitwise equal. On an H200 (UD-Q4_K_XL, 1k
    /// context), step ms at 1/2/4/8 streams: staged 4.08/5.03/6.92/11.00, d2 3.82/4.58/6.09/9.41,
    /// d4 4.04/4.84/6.15/9.12, db 4.63/5.52/7.34/10.57.
    /// `TANG_CODER_MOE`. Bitwise the same outputs.
    moe: usize,
}

impl Default for Variants {
    fn default() -> Self {
        Variants {
            kq: 2,
            fastk: 2,
            shared_par: false,
            grouped: false,
            moe: usize::MAX,
        }
    }
}

impl Variants {
    fn from_env() -> Self {
        let num = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<i64>().ok());
        let d = Variants::default();
        Variants {
            kq: num("TANG_CODER_KQ").map_or(d.kq, |v| v as usize),
            fastk: num("TANG_CODER_FASTK").map_or(d.fastk, |v| v as i32),
            shared_par: env_is("TANG_CODER_SHPAR", "1"),
            grouped: env_is("TANG_CODER_GROUPED", "1"),
            moe: num("TANG_CODER_MOE").map_or(d.moe, |v| v as usize),
        }
    }
}

fn gemv_v1() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("TANG_CODER_GEMV").is_ok_and(|v| v == "v1"))
}

/// The ggml type id the native GEMV decodes, if it does.
fn native_id(ty: GgmlType) -> Option<u32> {
    use GgmlType::*;
    Some(match ty {
        F32 => 0,
        Q4_0 => 2,
        Q5_0 => 6,
        Q8_0 => 8,
        Q3K => 11,
        Q4K => 12,
        Q5K => 13,
        Q6K => 14,
        Iq4Nl => 20,
        Iq4Xs => 23,
        Bf16 => 30,
        Q2_0 => 42,
        _ => return None,
    })
}

/// Q8_0 blocks (f16 scale, 32 int8 codes) of `v` (`v.len() % 32 == 0`).
fn quantize_q8_0(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() / 32 * 34);
    for b in v.chunks(32) {
        let amax = b.iter().fold(0f32, |m, x| m.max(x.abs()));
        let d = amax / 127.0;
        out.extend_from_slice(&gguf::f32_to_f16(d).to_le_bytes());
        for &x in b {
            let q = if d == 0.0 {
                0
            } else {
                (x / d).round().clamp(-127.0, 127.0) as i8
            };
            out.push(q as u8);
        }
    }
    out
}

/// A dense weight for the native GEMV: one or more GGUF tensors stacked by rows.
/// The MTP layer's index in [`Coder::lay`] (it lives outside `layers`).
const MTP_L: usize = usize::MAX;

/// The MTP layer's input (`cx_mtp_in`: `[rmsnorm(emb) * we | rmsnorm(h) * wh]` per row, grid
/// `(t, 2)`) and its pick (`cx_mtp_pick`: argmax and its softmax probability per row of logits,
/// grid `t`).
const MTP_SRC: &str = r#"
__device__ float mtp_block_sum(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
    if (threadIdx.x < 32)
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (threadIdx.x == 0) red[0] = v;
    __syncthreads();
    return red[0];
}

extern "C" __global__ void __launch_bounds__(1024) cx_mtp_in(const float* emb, const float* h,
        const float* we, const float* wh, float eps, int n, float* out) {
    __shared__ float red[32];
    const int c = blockIdx.x, half = blockIdx.y;
    const float* src = (half ? h : emb) + (size_t)c * n;
    const float* w = half ? wh : we;
    float* dst = out + (size_t)c * 2 * n + (size_t)half * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += src[i] * src[i];
    const float r = rsqrtf(mtp_block_sum(ss, red) / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i] * r * w[i];
}

extern "C" __global__ void __launch_bounds__(1024) cx_mtp_pick(const float* lg, int v,
        unsigned* ids, float* prob) {
    __shared__ float red[32];
    __shared__ float bm[32];
    __shared__ int bi[32];
    const float* row = lg + (size_t)blockIdx.x * v;
    float m = -3.0e38f;
    int mi = 0;
    for (int i = threadIdx.x; i < v; i += blockDim.x)
        if (row[i] > m) { m = row[i]; mi = i; }
    for (int o = 16; o > 0; o >>= 1) {
        const float om = __shfl_xor_sync(0xffffffffu, m, o);
        const int oi = __shfl_xor_sync(0xffffffffu, mi, o);
        if (om > m || (om == m && oi < mi)) { m = om; mi = oi; }
    }
    if ((threadIdx.x & 31) == 0) { bm[threadIdx.x >> 5] = m; bi[threadIdx.x >> 5] = mi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < (blockDim.x >> 5); w++)
            if (bm[w] > bm[0] || (bm[w] == bm[0] && bi[w] < bi[0])) { bm[0] = bm[w]; bi[0] = bi[w]; }
    }
    __syncthreads();
    const float top = bm[0];
    float se = 0.f;
    for (int i = threadIdx.x; i < v; i += blockDim.x) se += __expf(row[i] - top);
    se = mtp_block_sum(se, red);
    if (threadIdx.x == 0) { ids[blockIdx.x] = (unsigned)bi[0]; prob[blockIdx.x] = 1.f / se; }
}
"#;

/// The multi-token-prediction layer (an MTP sidecar): a gated-attention MoE layer whose input
/// is `fc([rmsnorm(embed(next token)) | rmsnorm(final hidden)])`, its output normed and put
/// through the model's head. Run after each window on the window's final hidden rows, then
/// chained on its own output, it drafts the next tokens.
struct Mtp {
    layer: Layer,
    fc: Dense,
    enorm: B,
    hnorm: B,
    onorm: B,
    /// Final hidden rows (normed): the last window's, then the MTP layer's own last row.
    mh: B,
    emb: B,
    cat: B,
    ids: B,
    prob: B,
    f_in: Fun,
    f_pick: Fun,
    _m: Module,
}

struct Dense {
    _w: Vec<B>,
    segs: B,
    nseg: i32,
    rows: usize,
    k: usize,
    row16: i32,
    /// Weight bytes in VRAM.
    bytes: usize,
}

/// A routed-expert tensor (`[n_expert][rows][k]`, GGUF type) in VRAM, as the kernels take it.
#[repr(C)]
#[derive(Clone, Copy)]
struct ExW {
    base: u64,
    estride: u64,
    ty: i32,
    row_bytes: i32,
}

struct Experts {
    _w: B,
    w: ExW,
}

struct GdnL {
    conv: B,
    dt: B,
    a: B,
    norm: B,
    state: B,
    hist: B,
    /// The window's stacked projection and conv output, kept for a later commit.
    proj: B,
    /// Every slot's `state` and `hist` addresses (u64 each; slot 0: the two above).
    stab: B,
    htab: B,
}

struct AttnL {
    qn: B,
    kn: B,
    k: B,
    v: B,
    proj: B,
    /// Every slot's K and V cache addresses (u64 each; slot 0: `k` and `v`).
    ktab: B,
    vtab: B,
}

struct Layer {
    attn_norm: B,
    post_norm: B,
    w_in: Dense,
    w_out: Dense,
    router: Dense,
    sh_gate: Experts,
    sh_up: Experts,
    sh_down: Experts,
    /// Shared-memory 16-byte units a staged expert row needs (gate|up, down).
    r16_gu: i32,
    r16_down: i32,
    gate: Experts,
    up: Experts,
    down: Experts,
    gdn: Option<GdnL>,
    attn: Option<AttnL>,
}

struct Kern {
    _mods: Vec<Module>,
    q8: Fun,
    gemv8: [Fun; MAX_T],
    gemvk: [Fun; MAX_T],
    gemvk8: [Fun; MAX_T],
    gemvk2: [Fun; MAX_T],
    gemvk1: [Fun; MAX_T],
    norm_q8: Fun,
    route: Fun,
    moe_gu: Fun,
    silu_q: Fun,
    moe_down: Fun,
    moe_plan: Fun,
    moe_gu_d: [Fun; 2],
    moe_down_d: [Fun; 2],
    moe_gu_db: Fun,
    moe_down_db: Fun,
    moe_gu_g: Fun,
    moe_down_g: Fun,
    moe_add_norm: Fun,
    prep: Fun,
    part: Fun,
    merge: Fun,
    gdn_step: Fun,
    gdn_commit: Fun,
    gdn_step_seg: Fun,
    gdn_commit_seg: Fun,
    copy: Fun,
    argmax1: Fun,
    argmax2: Fun,
}

/// Per-window scratch, sized for `MAX_T`.
struct Scratch {
    ctl: B,
    /// Per-row record (`ROWS_U32`): row positions, row slots, then `[start, len, slot, keep]`
    /// per segment (a slot's consecutive rows).
    rows: B,
    r: B,
    x: B,
    xq8: B,
    mxq: B,
    mix: B,
    y: B,
    q: B,
    part: B,
    attn: B,
    lg: B,
    ids: B,
    w: B,
    egu: B,
    hq: B,
    /// Grouped-expert plans (`cx_moe_plan`): one for the main stream, one for the side stream.
    plan: B,
    parts: B,
    moe_y: B,
    head: B,
    out_ids: B,
    amax: B,
}

/// Pinned staging: control words and embeddings in, sampled ids out.
struct Io {
    arena: HostArena,
}

impl Io {
    const CTL: usize = 0;
    const CCTL: usize = 64;
    const IDS: usize = 128;
    const EMB: usize = 256;

    fn bytes(hidden: usize) -> usize {
        Self::rows(hidden) + ROWS_U32 * 4
    }
    /// Where the per-row record sits (after the embeddings).
    fn rows(hidden: usize) -> usize {
        Self::EMB + MAX_T * hidden * 4
    }
    fn ptr(&self, off: usize) -> *mut u8 {
        unsafe { self.arena.as_ptr().add(off) }
    }
    #[allow(clippy::mut_from_ref)]
    fn f32s(&self, off: usize, n: usize) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr(off) as *mut f32, n) }
    }
    #[allow(clippy::mut_from_ref)]
    fn u32s(&self, off: usize, n: usize) -> &mut [u32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr(off) as *mut u32, n) }
    }
}

/// A sequence past slot 0: its own recurrent state, conv history and K/V caches (per layer:
/// `gdn` for GDN layers, `kv` for attention layers), and its tokens.
struct Slot {
    gdn: Vec<Option<(B, B)>>,
    kv: Vec<Option<(B, B)>>,
    tokens: Vec<u32>,
}

pub struct Coder {
    pub dev: CudaComputeDevice,
    gpu: Gpu,
    /// Top-k / top-p / presence sampling in the head (`enable_sampler`); `None`: Gumbel-max.
    sampler: Option<Box<super::sampler::Sampler>>,
    pub hp: Hparams,
    g: Gguf,
    geo: Geometry,
    eps: f32,
    layers: Vec<Layer>,
    out_norm: B,
    head: Dense,
    rope: (B, B),
    s: Scratch,
    io: Io,
    k: Kern,
    stream: ManuallyDrop<Stream>,
    /// The shared expert's branch (`shared_par`), and its fork/join events.
    side: Stream,
    ev_fork: gpu::Event,
    ev_join: gpu::Event,
    /// Window graphs by (t, verify, attention chunks, segments).
    graphs: HashMap<(usize, bool, usize, usize, bool), Graph>,
    /// The staged window is greedy (temperature 0, no presence penalty): the head takes the
    /// plain argmax even with the sampler enabled (its top-1 is the argmax then, ties to the
    /// lower id either way). Part of the graph key.
    greedy: std::cell::Cell<bool>,
    commit_graphs: Vec<Option<Graph>>,
    max_ctx: usize,
    counter: u32,
    pub tokens: Vec<u32>,
    /// Sequences past slot 0: their state buffers and tokens (`None`: free).
    slots: Vec<Option<Slot>>,
    pub seed: u32,
    pub temperature: f32,
    pub use_graphs: bool,
    pub last: WinStats,
    pub last_commit_ms: f64,
    pub load_report: String,
    /// `TANG_CODER_PROF=1` (eager windows): GPU time per stage, by name, summed.
    prof: Option<std::cell::RefCell<Prof>>,
    /// Stage sweep: enqueue only this stage's launches (`stage_sweep`).
    only: std::cell::Cell<Option<&'static str>>,
    var: Variants,
    mtp: Option<Mtp>,
    /// MTP pass graphs by (rows, attention chunks).
    mtp_graphs: HashMap<(usize, usize), Graph>,
    /// Draft with the MTP layer after each window (when loaded and greedy).
    pub use_mtp: bool,
    /// Chain length (`TANG_CODER_MTP_K`, default 2).
    pub mtp_steps: usize,
    /// The latest MTP chain: draft tokens and their probabilities, for the next verify.
    pub mtp_last: Vec<(u32, f32)>,
    pub last_mtp_ms: f64,
}

#[derive(Default)]
struct Prof {
    marks: Vec<(&'static str, gpu::Event)>,
    total: std::collections::BTreeMap<&'static str, (f64, usize)>,
}

impl Coder {
    /// Load a `qwen3next` GGUF (first shard) onto the GPU.
    pub fn load(path: &Path, opts: CoderOpts) -> Result<Self> {
        let t0 = Instant::now();
        let g = Gguf::open(path)?;
        let hp = Hparams::from_gguf(&g)?;
        ensure!(hp.arch == Arch::Qwen3Next, "not a qwen3next GGUF");
        ensure!(
            hp.ssm_d_state == 128 && hp.ssm_d_conv == 4,
            "GDN state {} / conv {}: the kernels are built for 128 / 4",
            hp.ssm_d_state,
            hp.ssm_d_conv
        );
        ensure!(
            hp.head_dim == 256 && hp.n_rot == 64 && hp.n_head % hp.n_head_kv == 0,
            "attention head_dim {} rot {}: the kernels are built for 256 / 64",
            hp.head_dim,
            hp.n_rot
        );
        ensure!(
            hp.n_embd.is_multiple_of(256) && hp.n_ff_exp.is_multiple_of(256),
            "widths must be multiples of 256"
        );
        let sh = Shapes {
            hidden: hp.n_embd,
            n_head: hp.n_head,
            n_kv: hp.n_head_kv,
            head_dim: hp.head_dim,
            n_rot: hp.n_rot,
            ff: hp.n_ff_exp,
            topk: hp.n_expert_used,
            n_expert: hp.n_expert,
        };
        // The flash module's GDN kernels for this model's heads; the rest of its geometry is
        // Flash-Next's (unused here, kept so every kernel of the module still compiles).
        let geo = Geometry {
            gdn_hk: hp.ssm_k_heads,
            gdn_hv: hp.ssm_v_heads,
            gdn_d: hp.ssm_d_state,
            gdn_grouped: true,
            ..Geometry::FLASH_NEXT
        };
        let max_ctx = opts.max_ctx.next_multiple_of(ck::attn_chunk());
        let eps = hp.eps;

        let dev = CudaComputeDevice::new().map_err(|e| anyhow!("CUDA: {e:?}"))?;
        let gpu = Gpu {
            ctx: dev.cuda_context().clone(),
        };
        gpu.bind().map_err(|e| anyhow!("{e}"))?;
        let (free0, _) = gpu.mem_info().map_err(|e| anyhow!("{e}"))?;

        // ---- weights
        let t1 = Instant::now();
        let info = |name: &str| -> Result<TensorInfo> { Ok(g.info(name)?.clone()) };
        let vecf = |name: &str| -> Result<B> { Ok(dev.upload_f32(&g.tensor_f32(name)?)) };
        let mut expert_bytes = 0usize;
        let mut layers = Vec::with_capacity(hp.n_layer);
        let (hk, hv) = (hp.ssm_k_heads, hp.ssm_v_heads);
        let r = hv / hk;
        // ssm_ba rows per k head g: [b of v heads g·r..g·r + r | a of the same]; the flash
        // kernels read [a (hv) | b (hv)] in v-head order.
        let ba_perm: Vec<usize> = (0..2 * hv)
            .map(|i| {
                let (is_b, h) = (i >= hv, i % hv);
                let (gk, j) = (h / r, h % r);
                gk * 2 * r + if is_b { j } else { r + j }
            })
            .collect();
        for l in 0..hp.n_layer {
            let layer = load_layer(
                &dev,
                &g,
                &hp,
                &geo,
                &format!("blk.{l}."),
                hp.is_recurrent[l],
                &ba_perm,
                max_ctx,
            )?;
            for e in [&layer.gate, &layer.up, &layer.down] {
                expert_bytes += e.w.estride as usize * hp.n_expert;
            }
            layers.push(layer);
            if l % 8 == 7 {
                eprintln!(
                    "coder: {} of {} layers in VRAM ({:.0} s)",
                    l + 1,
                    hp.n_layer,
                    t1.elapsed().as_secs_f64()
                );
            }
        }
        let ff_sh = hp.n_ff_exp;
        let out_norm = vecf("output_norm.weight")?;
        let head_name = if g.get("output.weight").is_some() {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let head = upload_dense(&dev, &g, &[(info(head_name)?, None)])?;
        ensure!(head.rows == hp.n_vocab, "head rows {}", head.rows);
        let rope = {
            let (c, s) = fl::rope_table(max_ctx, hp.rope_base as f64);
            (dev.upload_f32(&c), dev.upload_f32(&s))
        };
        let mtp_path = opts
            .mtp
            .clone()
            .or_else(|| std::env::var_os("TANG_CODER_MTP").map(Into::into));
        let mtp_parts = match &mtp_path {
            Some(p) => {
                let mg = Gguf::open(p).with_context(|| format!("MTP sidecar {}", p.display()))?;
                let layer = load_layer(&dev, &mg, &hp, &geo, "mtp.", false, &ba_perm, max_ctx)?;
                let fc = upload_dense(&dev, &mg, &[(mg.info("mtp.fc.weight")?.clone(), None)])?;
                ensure!(
                    fc.rows == hp.n_embd && fc.k == 2 * hp.n_embd,
                    "MTP fc {}x{}",
                    fc.rows,
                    fc.k
                );
                let v = |name: &str| -> Result<B> { Ok(dev.upload_f32(&mg.tensor_f32(name)?)) };
                Some((
                    layer,
                    fc,
                    v("mtp.enorm.weight")?,
                    v("mtp.hnorm.weight")?,
                    v("mtp.norm.weight")?,
                ))
            }
            None => None,
        };

        // ---- scratch
        let n = hp.n_embd;
        let kmax = [2 * n, geo.gdn_v(), hp.n_head * hp.head_dim, ff_sh]
            .into_iter()
            .max()
            .unwrap();
        let np = MAX_T * (hp.n_expert_used + 1);
        let nchunk_max = max_ctx / ck::attn_chunk();
        let z = |n: usize| dev.alloc_f32(n);
        let xq_words = |k: usize| (MAX_T * k + MAX_T * k / 32 * 4).div_ceil(4);
        let s = Scratch {
            ctl: z(8),
            rows: z(ROWS_U32),
            r: z(MAX_T * n),
            x: z(MAX_T * n),
            xq8: z(xq_words(kmax)),
            mxq: z(xq_words(n)),
            mix: z(MAX_T * n),
            y: z(MAX_T * geo.gdn_v()),
            q: z(MAX_T * hp.n_head * hp.head_dim),
            part: z(MAX_T
                * hp.n_head_kv
                * nchunk_max
                * (hp.n_head / hp.n_head_kv)
                * (hp.head_dim + 2)),
            attn: z(MAX_T * hp.n_head * hp.head_dim),
            lg: z(MAX_T * (hp.n_expert + 1)),
            ids: z(MAX_T * hp.n_expert_used),
            w: z(MAX_T * hp.n_expert_used),
            egu: z(np * 2 * hp.n_ff_exp),
            plan: z(2 * (2 + 2 * np)),
            hq: z((np * hp.n_ff_exp + np * hp.n_ff_exp / 32 * 4).div_ceil(4)),
            parts: z(np * n),
            moe_y: z(MAX_T * n),
            head: z(MAX_T * hp.n_vocab),
            out_ids: z(MAX_T),
            amax: z(MAX_T * 64 * 2),
        };
        dev.sync();
        let t_weights = t1.elapsed().as_secs_f64();
        let (free1, _) = gpu.mem_info().map_err(|e| anyhow!("{e}"))?;

        // ---- kernels
        let t2 = Instant::now();
        let src = format!("{}{}{}", ck::prelude(&sh), kernels::GEMV_SRC, ck::SRC);
        let cm = gpu
            .module(&src)
            .map_err(|e| anyhow!("coder kernels: {e}"))?;
        // GDN_SLOTS: the per-segment GDN kernels (several sequences a window); Flash-Next builds
        // without it, so its module is unchanged.
        let fm = gpu
            .module(&format!("#define GDN_SLOTS 1\n{}", geo.cuda_source()))
            .map_err(|e| anyhow!("flash kernels: {e}"))?;
        let em = gpu.module(kernels::SRC).map_err(|e| anyhow!("{e}"))?;
        let f = |m: &Module, n: &str| m.func(n).map_err(|e| anyhow!("{n}: {e}"));
        let mut gemv8 = [std::ptr::null_mut(); MAX_T];
        for (i, slot) in gemv8.iter_mut().enumerate() {
            *slot = f(&cm, &format!("fe_gemv8_t{}", i + 1))?;
            unsafe {
                gpu::check(
                    cudarc::driver::sys::cuFuncSetAttribute(
                        *slot,
                        cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        99 * 1024,
                    ),
                    "cuFuncSetAttribute",
                )
                .map_err(|e| anyhow!("{e}"))?;
            }
        }
        let mut gemvk = [std::ptr::null_mut(); MAX_T];
        for (i, slot) in gemvk.iter_mut().enumerate() {
            *slot = f(&cm, &format!("cx_gemvk_t{}", i + 1))?;
        }
        let k = Kern {
            q8: f(&cm, "fe_q8")?,
            gemv8,
            gemvk,
            gemvk1: {
                let mut a = [std::ptr::null_mut(); MAX_T];
                for (i, slot) in a.iter_mut().enumerate() {
                    *slot = f(&cm, &format!("cx_gemvk1_t{}", i + 1))?;
                }
                a
            },
            gemvk2: {
                let mut a = [std::ptr::null_mut(); MAX_T];
                for (i, slot) in a.iter_mut().enumerate() {
                    *slot = f(&cm, &format!("cx_gemvk2_t{}", i + 1))?;
                }
                a
            },
            gemvk8: {
                let mut a = [std::ptr::null_mut(); MAX_T];
                for (i, slot) in a.iter_mut().enumerate() {
                    *slot = f(&cm, &format!("cx_gemvk8_t{}", i + 1))?;
                }
                a
            },
            norm_q8: f(&cm, "cx_norm_q8")?,
            route: f(&cm, "cx_route")?,
            moe_gu: f(&cm, "cx_moe_gu")?,
            silu_q: f(&cm, "cx_silu_q")?,
            moe_down: f(&cm, "cx_moe_down")?,
            moe_plan: f(&cm, "cx_moe_plan")?,
            moe_gu_d: [f(&cm, "cx_moe_gu_d2")?, f(&cm, "cx_moe_gu_d4")?],
            moe_down_d: [f(&cm, "cx_moe_down_d2")?, f(&cm, "cx_moe_down_d4")?],
            moe_gu_db: f(&cm, "cx_moe_gu_db")?,
            moe_down_db: f(&cm, "cx_moe_down_db")?,
            moe_gu_g: f(&cm, "cx_moe_gu_g")?,
            moe_down_g: f(&cm, "cx_moe_down_g")?,
            moe_add_norm: f(&cm, "cx_moe_add_norm")?,
            prep: f(&cm, "cx_attn_prep")?,
            part: f(&cm, "cx_attn_part")?,
            merge: f(&cm, "cx_attn_merge")?,
            gdn_step: f(&fm, "fl_gdn_conv_step")?,
            gdn_commit: f(&fm, "fl_gdn_conv_commit")?,
            gdn_step_seg: f(&fm, "fl_gdn_conv_step_seg")?,
            gdn_commit_seg: f(&fm, "fl_gdn_conv_commit_seg")?,
            copy: f(&em, "fe_copy")?,
            argmax1: f(&em, "fe_argmax1")?,
            argmax2: f(&em, "fe_argmax2")?,
            _mods: vec![cm, fm, em],
        };
        // Up to 200 KB of staged expert rows, capped at what the device allows a block (99 KB on
        // sm_86/89, 227 KB on sm_90), so the engine also loads on smaller GPUs.
        let smem_max = {
            let mut v = 0i32;
            unsafe {
                cudarc::driver::sys::cuDeviceGetAttribute(
                    &mut v,
                    cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                    0,
                )
            };
            if v > 0 {
                v.min(200 * 1024)
            } else {
                99 * 1024
            }
        };
        for f in [
            k.moe_gu,
            k.moe_down,
            k.moe_gu_g,
            k.moe_down_g,
            k.moe_gu_db,
            k.moe_down_db,
        ] {
            unsafe {
                gpu::check(
                    cudarc::driver::sys::cuFuncSetAttribute(
                        f,
                        cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        smem_max,
                    ),
                    "cuFuncSetAttribute",
                )
                .map_err(|e| anyhow!("{e}"))?;
            }
        }
        let mtp = match mtp_parts {
            Some((layer, fc, enorm, hnorm, onorm)) => {
                let m = gpu
                    .module(MTP_SRC)
                    .map_err(|e| anyhow!("MTP kernels: {e}"))?;
                Some(Mtp {
                    layer,
                    fc,
                    enorm,
                    hnorm,
                    onorm,
                    mh: dev.alloc_f32(MAX_T * n),
                    emb: dev.alloc_f32(MAX_T * n),
                    cat: dev.alloc_f32(MAX_T * 2 * n),
                    ids: dev.alloc_f32(MAX_T),
                    prob: dev.alloc_f32(MAX_T),
                    f_in: f(&m, "cx_mtp_in")?,
                    f_pick: f(&m, "cx_mtp_pick")?,
                    _m: m,
                })
            }
            None => None,
        };
        let t_kernels = t2.elapsed().as_secs_f64();

        let mut io = Io {
            arena: HostArena::new(
                Io::bytes(n),
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
        let load_report = format!(
            "coder: {} layers ({} GDN, {} attention), {} experts top-{}, hidden {}, vocab {}; \
             {:.1} GB VRAM ({:.1} GB experts), weights {:.0} s, kernels {:.0} s, total {:.0} s",
            hp.n_layer,
            hp.is_recurrent.iter().filter(|&&x| x).count(),
            hp.is_recurrent.iter().filter(|&&x| !x).count(),
            hp.n_expert,
            hp.n_expert_used,
            n,
            hp.n_vocab,
            (free0 - free1) as f64 / 1e9,
            expert_bytes as f64 / 1e9,
            t_weights,
            t_kernels,
            t0.elapsed().as_secs_f64()
        );
        let c = Coder {
            dev,
            gpu,
            sampler: None,
            hp,
            g,
            geo,
            eps,
            layers,
            out_norm,
            head,
            rope,
            s,
            io,
            k,
            stream,
            side: Stream::new().map_err(|e| anyhow!("{e}"))?,
            ev_fork: gpu::Event::new(false).map_err(|e| anyhow!("{e}"))?,
            ev_join: gpu::Event::new(false).map_err(|e| anyhow!("{e}"))?,
            graphs: HashMap::new(),
            commit_graphs: (0..=MAX_T).map(|_| None).collect(),
            max_ctx,
            counter: 0,
            tokens: Vec::new(),
            greedy: std::cell::Cell::new(true),
            slots: (0..MAX_SLOTS).map(|_| None).collect(),
            seed: 0,
            temperature: 0.0,
            use_graphs: true,
            last: WinStats::default(),
            last_commit_ms: 0.0,
            load_report,
            prof: std::env::var("TANG_CODER_PROF")
                .is_ok_and(|v| v == "1")
                .then(|| std::cell::RefCell::new(Prof::default())),
            only: std::cell::Cell::new(None),
            var: Variants::from_env(),
            use_mtp: mtp.is_some(),
            mtp,
            mtp_graphs: HashMap::new(),
            mtp_steps: std::env::var("TANG_CODER_MTP_K")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2)
                .clamp(1, MAX_T - 1),
            mtp_last: Vec::new(),
            last_mtp_ms: 0.0,
        };
        let mut c = c;
        c.write_slot_tables();
        Ok(c)
    }

    // ---- sequence slots (several sequences in one window)

    /// Rewrite every layer's slot tables from the live slots (slot 0: the layer's own buffers;
    /// a free slot: 0).
    fn write_slot_tables(&mut self) {
        let nl = self.layers.len();
        for l in 0..nl {
            let mut ta = [0u64; MAX_SLOTS];
            let mut tb = [0u64; MAX_SLOTS];
            let layer = &self.layers[l];
            let (a0, b0) = match (&layer.gdn, &layer.attn) {
                (Some(g), _) => (self.a(&g.state), self.a(&g.hist)),
                (_, Some(at)) => (self.a(&at.k), self.a(&at.v)),
                _ => continue,
            };
            ta[0] = a0;
            tb[0] = b0;
            for (i, sl) in self.slots.iter().enumerate().skip(1) {
                if let Some(sl) = sl {
                    let pair = if layer.gdn.is_some() {
                        &sl.gdn[l]
                    } else {
                        &sl.kv[l]
                    };
                    if let Some((a, b)) = pair {
                        ta[i] = self.a(a);
                        tb[i] = self.a(b);
                    }
                }
            }
            let words = |t: &[u64; MAX_SLOTS]| -> Vec<f32> {
                t.iter()
                    .flat_map(|&v| [f32::from_bits(v as u32), f32::from_bits((v >> 32) as u32)])
                    .collect()
            };
            let (wa, wb) = (words(&ta), words(&tb));
            let layer = &mut self.layers[l];
            let (da, db) = match (&mut layer.gdn, &mut layer.attn) {
                (Some(g), _) => (&mut g.stab, &mut g.htab),
                (_, Some(at)) => (&mut at.ktab, &mut at.vtab),
                _ => continue,
            };
            self.dev.upload_into_f32(da, &wa);
            self.dev.upload_into_f32(db, &wb);
        }
        // The MTP layer drafts for slot 0 only.
        let mk = self
            .mtp
            .as_ref()
            .and_then(|m| m.layer.attn.as_ref())
            .map(|at| (self.a(&at.k), self.a(&at.v)));
        if let Some((k, v)) = mk {
            let word = |v: u64| -> Vec<f32> {
                let mut w = vec![0f32; 2 * MAX_SLOTS];
                w[0] = f32::from_bits(v as u32);
                w[1] = f32::from_bits((v >> 32) as u32);
                w
            };
            let at = self.mtp.as_mut().unwrap().layer.attn.as_mut().unwrap();
            self.dev.upload_into_f32(&mut at.ktab, &word(k));
            self.dev.upload_into_f32(&mut at.vtab, &word(v));
        }
        self.dev.sync();
    }

    /// Switch the per-pair expert kernels (`TANG_CODER_MOE`'s values) without reloading; drops
    /// the captured window graphs, which hold the old kernels.
    pub fn set_moe_variant(&mut self, v: usize) {
        self.var.moe = v;
        self.graphs.clear();
        self.commit_graphs.iter_mut().for_each(|g| *g = None);
    }

    /// A free sequence slot (1..MAX_SLOTS), its state zeroed and no tokens.
    pub fn alloc_slot(&mut self) -> Result<usize> {
        let i = (1..MAX_SLOTS)
            .find(|&i| self.slots[i].is_none())
            .ok_or_else(|| anyhow!("all {} slots in use", MAX_SLOTS))?;
        let (hv, conv, kv) = (
            self.geo.gdn_hv,
            self.geo.gdn_conv(),
            self.max_ctx * self.hp.n_head_kv * self.hp.head_dim,
        );
        let mut sl = Slot {
            gdn: Vec::with_capacity(self.layers.len()),
            kv: Vec::with_capacity(self.layers.len()),
            tokens: Vec::new(),
        };
        for layer in &self.layers {
            sl.gdn.push(layer.gdn.as_ref().map(|_| {
                (
                    self.dev.alloc_f32(hv * 128 * 128),
                    self.dev.alloc_f32(3 * conv),
                )
            }));
            sl.kv.push(
                layer
                    .attn
                    .as_ref()
                    .map(|_| (self.dev.alloc_bf16(kv), self.dev.alloc_bf16(kv))),
            );
        }
        self.slots[i] = Some(sl);
        self.write_slot_tables();
        Ok(i)
    }

    /// Release slot `i` (not 0) and its memory.
    pub fn free_slot(&mut self, i: usize) {
        if i > 0 && i < MAX_SLOTS && self.slots[i].take().is_some() {
            self.write_slot_tables();
        }
    }

    /// Slot `i`'s tokens (slot 0: [`tokens`](Self::tokens)).
    pub fn slot_tokens(&self, i: usize) -> &[u32] {
        if i == 0 {
            &self.tokens
        } else {
            self.slots[i].as_ref().map_or(&[], |s| &s.tokens)
        }
    }

    fn slot_tokens_mut(&mut self, i: usize) -> Result<&mut Vec<u32>> {
        if i == 0 {
            return Ok(&mut self.tokens);
        }
        self.slots
            .get_mut(i)
            .and_then(|s| s.as_mut())
            .map(|s| &mut s.tokens)
            .ok_or_else(|| anyhow!("slot {i} isn't allocated"))
    }

    /// Stage a window of rows `(slot, position, token)` in segments `[start, len, slot, keep]`.
    fn stage_rows(&mut self, rows: &[(usize, usize, u32)], segs: &[[u32; 4]]) -> Result<()> {
        let t = rows.len();
        ensure!((1..=MAX_T).contains(&t) && segs.len() <= MAX_T);
        let ctl = self.io.u32s(Io::CTL, 5);
        ctl[0] = rows[0].1 as u32;
        ctl[1] = t as u32;
        ctl[2] = self.counter;
        ctl[3] = self.seed;
        ctl[4] = self.temperature.to_bits();
        let n = self.hp.n_embd;
        let emb_t = self.g.info("token_embd.weight")?.clone();
        let emb = self.io.f32s(Io::EMB, t * n);
        for (i, r) in rows.iter().enumerate() {
            let row = self.g.rows(&emb_t, r.2 as usize, 1)?;
            emb[i * n..(i + 1) * n].copy_from_slice(&row);
        }
        let rec = self.io.u32s(Io::rows(n), ROWS_U32);
        rec.fill(0);
        for (i, r) in rows.iter().enumerate() {
            rec[i] = r.1 as u32;
            rec[MAX_T + i] = r.0 as u32;
        }
        for (b, sg) in segs.iter().enumerate() {
            rec[2 * MAX_T + 4 * b..2 * MAX_T + 4 * b + 4].copy_from_slice(sg);
        }
        Ok(())
    }

    /// Feed `ids` to slot `slot` (windows of up to `MAX_T` of its rows) and return the next
    /// token. Greedy; the sampler isn't per slot yet.
    pub fn prefill_slot(&mut self, slot: usize, ids: &[u32]) -> Result<u32> {
        ensure!(!ids.is_empty(), "empty prompt");
        ensure!(self.sampler.is_none(), "batched windows are greedy for now");
        let mut pos = self.slot_tokens(slot).len();
        self.slot_tokens_mut(slot)?.extend_from_slice(ids);
        let end = pos + ids.len();
        ensure!(end <= self.max_ctx, "context {end} > max {}", self.max_ctx);
        let mut next = 0;
        while pos < end {
            let t = (end - pos).min(MAX_T);
            let toks = &self.slot_tokens(slot)[pos..pos + t];
            let rows: Vec<(usize, usize, u32)> = toks
                .iter()
                .enumerate()
                .map(|(i, &k)| (slot, pos + i, k))
                .collect();
            let w0 = Instant::now();
            self.stage_rows(&rows, &[[0, t as u32, slot as u32, t as u32]])?;
            let nchunk = self.nchunk(pos + t);
            let st = WinStats {
                t,
                ..Default::default()
            };
            next = self.run_staged(st, w0, t, false, nchunk, 1, None)?[t - 1];
            pos += t;
        }
        Ok(next)
    }

    /// One decode step for several sequences at once: `(slot, token)` per row, distinct
    /// slots, at most `MAX_T`. Feeds each slot its token and returns each row's next token
    /// (greedy). A slot's result is the one a window of that slot alone would give.
    pub fn decode_batch(&mut self, rows: &[(usize, u32)]) -> Result<Vec<u32>> {
        let b = rows.len();
        ensure!((1..=MAX_T).contains(&b), "{b} rows");
        ensure!(self.sampler.is_none(), "batched windows are greedy for now");
        let mut staged = Vec::with_capacity(b);
        let mut segs = Vec::with_capacity(b);
        let mut end = 0;
        for (i, &(slot, tok)) in rows.iter().enumerate() {
            ensure!(
                rows[..i].iter().all(|r| r.0 != slot),
                "slot {slot} twice in one step"
            );
            let toks = self.slot_tokens_mut(slot)?;
            let pos = toks.len();
            toks.push(tok);
            end = end.max(pos + 1);
            staged.push((slot, pos, tok));
            segs.push([i as u32, 1, slot as u32, 1]);
        }
        ensure!(end <= self.max_ctx, "context {end} > max {}", self.max_ctx);
        let w0 = Instant::now();
        self.stage_rows(&staged, &segs)?;
        let nchunk = self.nchunk(end);
        let st = WinStats {
            t: b,
            ..Default::default()
        };
        self.run_staged(st, w0, b, false, nchunk, b, None)
    }

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

    // ---- launch helpers

    fn a(&self, b: &B) -> u64 {
        self.dev.buffer_addr(b)
    }

    unsafe fn launch(
        &self,
        f: Fun,
        grid: (u32, u32, u32),
        block: u32,
        args: &mut [*mut std::ffi::c_void],
    ) {
        if self.skip(self.kname(f)) {
            return;
        }
        gpu::launch(f, grid, (block, 1, 1), 0, &self.stream, args).expect("launch");
        if self.prof.is_some() {
            self.mark(self.kname(f));
        }
    }

    /// A launch with dynamic shared memory on any stream.
    unsafe fn launch_on(
        &self,
        stream: &Stream,
        f: Fun,
        grid: (u32, u32, u32),
        block: u32,
        smem: u32,
        args: &mut [*mut std::ffi::c_void],
    ) {
        if self.skip(self.kname(f)) {
            return;
        }
        gpu::launch(f, grid, (block, 1, 1), smem, stream, args).expect("launch");
        if self.prof.is_some() && std::ptr::eq(stream, &*self.stream) {
            self.mark(self.kname(f));
        }
    }

    /// Stage sweep: is `name` filtered out?
    fn skip(&self, name: &'static str) -> bool {
        self.only.get().is_some_and(|o| o != name)
    }

    fn kname(&self, f: Fun) -> &'static str {
        let k = &self.k;
        [
            (k.norm_q8, "norm_q8"),
            (k.route, "route"),
            (k.moe_gu, "moe_gu"),
            (k.silu_q, "silu_q"),
            (k.moe_down, "moe_down"),
            (k.moe_gu_d[0], "moe_gu"),
            (k.moe_gu_d[1], "moe_gu"),
            (k.moe_down_d[0], "moe_down"),
            (k.moe_down_d[1], "moe_down"),
            (k.moe_gu_db, "moe_gu"),
            (k.moe_down_db, "moe_down"),
            (k.moe_plan, "route"),
            (k.moe_gu_g, "moe_gu"),
            (k.moe_down_g, "moe_down"),
            (k.moe_add_norm, "moe_add_norm"),
            (k.prep, "attn_prep"),
            (k.part, "attn_part"),
            (k.merge, "attn_merge"),
            (k.gdn_step, "gdn_conv_step"),
            (k.gdn_commit, "gdn_commit"),
            (k.gdn_step_seg, "gdn_conv_step"),
            (k.gdn_commit_seg, "gdn_commit"),
            (k.copy, "copy"),
            (k.argmax1, "argmax"),
            (k.argmax2, "argmax"),
            (k.q8, "moe_q8"),
        ]
        .into_iter()
        .find(|(g, _)| *g == f)
        .map_or("other", |(_, n)| n)
    }

    /// Profiling: everything enqueued since the previous mark is charged to `name`.
    fn mark(&self, name: &'static str) {
        if let Some(p) = &self.prof {
            let ev = gpu::Event::new(true).expect("event");
            ev.record(&self.stream).expect("record");
            p.borrow_mut().marks.push((name, ev));
        }
    }

    /// Fold the window's marks into the totals.
    fn prof_collect(&self) {
        let Some(p) = &self.prof else { return };
        let mut p = p.borrow_mut();
        let marks = std::mem::take(&mut p.marks);
        for w in marks.windows(2) {
            let ms = w[1].1.since(&w[0].1).unwrap_or(0.0) as f64;
            let e = p.total.entry(w[1].0).or_default();
            e.0 += ms;
            e.1 += 1;
        }
    }

    /// The profile so far, by stage (`TANG_CODER_PROF=1`).
    pub fn prof_report(&self) -> Option<String> {
        let p = self.prof.as_ref()?.borrow();
        let tot: f64 = p.total.values().map(|v| v.0).sum();
        let mut v: Vec<_> = p.total.iter().collect();
        v.sort_by(|a, b| b.1 .0.total_cmp(&a.1 .0));
        Some(
            v.iter()
                .map(|(k, (ms, n))| {
                    format!("  {k:14} {ms:9.2} ms  x{n:<6} {:5.1}%\n", 100.0 * ms / tot)
                })
                .collect(),
        )
    }

    fn io_dev(&self, off: usize) -> u64 {
        self.io.arena.device_ptr(off).expect("io arena is mapped")
    }

    /// `bytes` (a multiple of 4) from `src` to `dst` with a kernel (device or mapped host).
    fn kcopy(&self, dst: u64, src: u64, bytes: usize) {
        let n = (bytes / 4) as i32;
        unsafe {
            self.launch(
                self.k.copy,
                ((n as u32).div_ceil(256).max(1), 1, 1),
                256,
                tang_moe::args![dst, src, n],
            )
        };
    }

    /// `out[t][off + row] = W · x[t]` for every row of `w` (f32 `x`, `w.k` wide), outputs
    /// `ostride` apart; `x`'s int8 codes are made here (fe_q8) unless `xq` already holds them.
    #[allow(clippy::too_many_arguments)]
    fn gemv(
        &self,
        name: &'static str,
        w: &Dense,
        x: u64,
        xq: Option<u64>,
        out: u64,
        t: usize,
        ostride: usize,
    ) {
        if self.skip(name) {
            return;
        }
        let (sp, ns, os, r16, ki) = (self.a(&w.segs), w.nseg, ostride as i32, w.row16, w.k as i32);
        let xq = match xq {
            Some(q) => q,
            None => {
                let q = self.a(&self.s.xq8);
                unsafe {
                    gpu::launch(
                        self.k.q8,
                        ((w.k / 32) as u32, t as u32, 1),
                        (32, 1, 1),
                        0,
                        &self.stream,
                        tang_moe::args![x, q, ki],
                    )
                    .expect("fe_q8");
                }
                q
            }
        };
        let kq = self.var.kq;
        if w.k.is_multiple_of(kq * 256) && !gemv_v1() {
            let f = match kq {
                8 => self.k.gemvk8[t - 1],
                2 => self.k.gemvk2[t - 1],
                1 => self.k.gemvk1[t - 1],
                _ => self.k.gemvk[t - 1],
            };
            unsafe {
                gpu::launch(
                    f,
                    (w.rows.div_ceil(8 / kq) as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    &self.stream,
                    tang_moe::args![sp, ns, x, xq, ki, out, os],
                )
                .expect("cx_gemvk")
            };
        } else {
            unsafe {
                gpu::launch(
                    self.k.gemv8[t - 1],
                    (w.rows.div_ceil(8) as u32, 1, 1),
                    (128, 1, 1),
                    (8 * 16 * r16) as u32,
                    &self.stream,
                    tang_moe::args![sp, ns, x, xq, ki, out, os, r16],
                )
                .expect("fe_gemv8")
            };
        }
        self.mark(name);
    }

    /// `r += y` (when given), then `s.x = rmsnorm(r) * w` and its int8 codes in `s.mxq`.
    fn norm_q8(&self, add: Option<u64>, w: &B, t: usize) {
        let (r, y, wp, x, xq, n, eps, ad) = (
            self.a(&self.s.r),
            add.unwrap_or(0),
            self.a(w),
            self.a(&self.s.x),
            self.a(&self.s.mxq),
            self.hp.n_embd as i32,
            self.eps,
            add.is_some() as i32,
        );
        unsafe {
            self.launch(
                self.k.norm_q8,
                (t as u32, 1, 1),
                1024,
                tang_moe::args![r, y, wp, x, xq, n, eps, ad],
            )
        };
    }

    /// Split-K chunks a window at `pos0 + t` runs over: a power-of-two bucket of the context,
    /// so a window graph is reused until the context doubles.
    fn nchunk(&self, end: usize) -> usize {
        let bucket = end.next_power_of_two().max(4 * ck::attn_chunk());
        bucket.min(self.max_ctx) / ck::attn_chunk()
    }

    // ---- the window

    fn enqueue_inputs(&self, t: usize) {
        self.mark("start");
        self.kcopy(self.a(&self.s.ctl), self.io_dev(Io::CTL), 32);
        self.kcopy(
            self.a(&self.s.rows),
            self.io_dev(Io::rows(self.hp.n_embd)),
            ROWS_U32 * 4,
        );
        self.kcopy(
            self.a(&self.s.r),
            self.io_dev(Io::EMB),
            t * self.hp.n_embd * 4,
        );
    }

    /// Decoder layer `l`, or the MTP layer for `MTP_L`.
    fn lay(&self, l: usize) -> &Layer {
        match (l, &self.mtp) {
            (MTP_L, Some(m)) => &m.layer,
            _ => &self.layers[l],
        }
    }

    /// Layer `l`'s mixer (input `s.x` / `s.mxq`) into `s.mix`.
    fn enqueue_mixer(&self, l: usize, t: usize, verify: bool, nchunk: usize, nseg: usize) {
        let layer = self.lay(l);
        let s = &self.s;
        let (x, xq) = (self.a(&s.x), Some(self.a(&s.mxq)));
        let eps = self.eps;
        let n = self.hp.n_embd;
        if let Some(g) = &layer.gdn {
            let stride = layer.w_in.rows;
            self.gemv("gdn_in", &layer.w_in, x, xq, self.a(&g.proj), t, stride);
            self.gdn_step_seg(g, stride, nseg, !verify);
            if !verify {
                self.gdn_commit_seg(g, stride, nseg);
            }
            self.gemv(
                "gdn_out",
                &layer.w_out,
                self.a(&s.y),
                None,
                self.a(&s.mix),
                t,
                n,
            );
        }
        if let Some(at) = &layer.attn {
            let stride = layer.w_in.rows;
            self.gemv("attn_in", &layer.w_in, x, xq, self.a(&at.proj), t, stride);
            let hp = &self.hp;
            let (proj, st, qn, kn, rc, rs, q, kt, vt, rows) = (
                self.a(&at.proj),
                stride as i32,
                self.a(&at.qn),
                self.a(&at.kn),
                self.a(&self.rope.0),
                self.a(&self.rope.1),
                self.a(&s.q),
                self.a(&at.ktab),
                self.a(&at.vtab),
                self.a(&s.rows),
            );
            let (part, nc, attn) = (self.a(&s.part), nchunk as i32, self.a(&s.attn));
            unsafe {
                self.launch(
                    self.k.prep,
                    (t as u32, (hp.n_head + hp.n_head_kv) as u32, 1),
                    hp.head_dim as u32,
                    tang_moe::args![proj, st, qn, kn, rc, rs, rows, q, kt, vt, eps],
                );
                self.launch(
                    self.k.part,
                    (t as u32, hp.n_head_kv as u32, nchunk as u32),
                    128,
                    tang_moe::args![q, kt, vt, rows, part, nc],
                );
                self.launch(
                    self.k.merge,
                    (t as u32, hp.n_head as u32, 1),
                    hp.head_dim as u32,
                    tang_moe::args![part, nc, proj, st, attn],
                );
            }
            self.gemv("attn_out", &layer.w_out, attn, None, self.a(&s.mix), t, n);
        }
    }

    /// The GDN conv and recurrence in one launch (`commit`: write the state for `win[1]`
    /// tokens), from the stacked projection and the conv history.
    fn gdn_step(&self, g: &GdnL, stride: usize, t: usize, commit: bool) {
        let (st, p, sd, dt, sa, nw, y, tu, win, cm, eps, yq, quant, h0, wc) = (
            self.a(&g.state),
            self.a(&g.proj),
            stride as u32,
            self.a(&g.dt),
            self.a(&g.a),
            self.a(&g.norm),
            self.a(&self.s.y),
            t as u32,
            self.a(&self.s.ctl),
            commit as u32,
            self.eps,
            0u64,
            0u32,
            self.a(&g.hist),
            self.a(&g.conv),
        );
        unsafe {
            self.launch(
                self.k.gdn_step,
                (self.geo.gdn_hv as u32, 1, 1),
                512,
                tang_moe::args![st, p, sd, dt, sa, nw, y, tu, win, cm, eps, yq, quant, h0, wc],
            )
        };
    }

    fn gdn_commit(&self, g: &GdnL, stride: usize, t: usize) {
        let (h0, p, sd, win, tu) = (
            self.a(&g.hist),
            self.a(&g.proj),
            stride as u32,
            self.a(&self.s.ctl),
            t as u32,
        );
        unsafe {
            self.launch(
                self.k.gdn_commit,
                (self.geo.gdn_conv().div_ceil(256) as u32, 1, 1),
                256,
                tang_moe::args![h0, p, sd, win, tu],
            )
        };
    }

    /// [`gdn_step`](Self::gdn_step) for every segment of the window (`s.rows`): each segment's
    /// rows run against its slot's state and history.
    fn gdn_step_seg(&self, g: &GdnL, stride: usize, nseg: usize, commit: bool) {
        let (stab, htab, p, sd, dt, sa, nw, y, seg, cm, eps, wc) = (
            self.a(&g.stab),
            self.a(&g.htab),
            self.a(&g.proj),
            stride as u32,
            self.a(&g.dt),
            self.a(&g.a),
            self.a(&g.norm),
            self.a(&self.s.y),
            self.a(&self.s.rows) + (2 * MAX_T * 4) as u64,
            commit as u32,
            self.eps,
            self.a(&g.conv),
        );
        unsafe {
            self.launch(
                self.k.gdn_step_seg,
                (self.geo.gdn_hv as u32, nseg as u32, 1),
                512,
                tang_moe::args![stab, htab, p, sd, dt, sa, nw, y, seg, cm, eps, wc],
            )
        };
    }

    fn gdn_commit_seg(&self, g: &GdnL, stride: usize, nseg: usize) {
        let (htab, p, sd, seg) = (
            self.a(&g.htab),
            self.a(&g.proj),
            stride as u32,
            self.a(&self.s.rows) + (2 * MAX_T * 4) as u64,
        );
        unsafe {
            self.launch(
                self.k.gdn_commit_seg,
                (self.geo.gdn_conv().div_ceil(256) as u32, nseg as u32, 1),
                256,
                tang_moe::args![htab, p, sd, seg],
            )
        };
    }

    /// Layer `l`'s MoE (input `s.x` / `s.mxq`): router, top-k, routed and shared experts in the
    /// same launches, then `r +=` the result and the next norm (`next`: the next layer's
    /// `attn_norm`, or `output_norm`) into `s.x` / `s.mxq`. `s.moe_y` gets the MoE output when
    /// `keep_y`.
    fn enqueue_moe(&self, l: usize, t: usize, next: &B, keep_y: bool) {
        let layer = self.lay(l);
        let (s, hp) = (&self.s, &self.hp);
        let (ne, kk) = (hp.n_expert, hp.n_expert_used);
        let (x, mxq) = (self.a(&s.x), self.a(&s.mxq));
        let ls = ne + 1;
        let (lg, lsi, ids, w, scale) = (
            self.a(&s.lg),
            ls as i32,
            self.a(&s.ids),
            self.a(&s.w),
            hp.expert_weights_scale,
        );
        let (r, yo, wn, xo, eps, ti) = (
            self.a(&s.r),
            if keep_y { self.a(&s.moe_y) } else { 0u64 },
            self.a(next),
            x,
            self.eps,
            t as i32,
        );
        let routed = t * kk;
        if self.var.shared_par {
            // The shared expert doesn't depend on the routing: fork it onto the side stream
            // before the router, join before the combine.
            self.fork();
            self.experts(layer, t, routed, t, &self.side);
            self.gemv("router", &layer.router, x, Some(mxq), lg, t, ls);
            unsafe {
                self.launch(
                    self.k.route,
                    (t as u32, 1, 1),
                    32,
                    tang_moe::args![lg, lsi, ids, w, scale],
                );
            }
            self.experts(layer, t, 0, routed, &self.stream);
            self.join();
        } else {
            self.gemv("router", &layer.router, x, Some(mxq), lg, t, ls);
            unsafe {
                self.launch(
                    self.k.route,
                    (t as u32, 1, 1),
                    32,
                    tang_moe::args![lg, lsi, ids, w, scale],
                );
            }
            self.experts(layer, t, 0, routed + t, &self.stream);
        }
        let parts = self.a(&s.parts);
        unsafe {
            self.launch(
                self.k.moe_add_norm,
                (t as u32, 1, 1),
                1024,
                tang_moe::args![r, parts, w, lg, lsi, ti, yo, wn, xo, mxq, eps],
            );
        }
    }

    /// Expert pairs `pbase..pbase + n` (routed pairs first, then the shared expert per token):
    /// gate|up rows, SiLU+int8, down rows, on `stream`.
    fn experts(&self, layer: &Layer, t: usize, pbase: usize, n: usize, stream: &Stream) {
        if n == 0 {
            return;
        }
        let (s, hp) = (&self.s, &self.hp);
        let (nh, ff) = (hp.n_embd, hp.n_ff_exp);
        let (gw, uw, dw) = (layer.gate.w, layer.up.w, layer.down.w);
        let (sg, su, sd) = (layer.sh_gate.w, layer.sh_up.w, layer.sh_down.w);
        let (ids, mxq, egu, hq, parts) = (
            self.a(&s.ids),
            self.a(&s.mxq),
            self.a(&s.egu),
            self.a(&s.hq),
            self.a(&s.parts),
        );
        let (ti, r16g, r16d, fastk, pb) = (
            t as i32,
            layer.r16_gu,
            layer.r16_down,
            self.var.fastk,
            pbase as i32,
        );
        let np = n as u32;
        if self.var.grouped {
            // One plan region per stream (the side stream's call is the one past the routed pairs).
            let pw = 2 + 2 * MAX_T * (hp.n_expert_used + 1);
            let plan = self.a(&s.plan) + if pbase > 0 { (pw * 4) as u64 } else { 0 };
            let ni = n as i32;
            let groups = n.min(hp.n_expert + 1) as u32;
            unsafe {
                self.launch_on(
                    stream,
                    self.k.moe_plan,
                    (1, 1, 1),
                    128,
                    0,
                    tang_moe::args![ids, ti, pb, ni, plan],
                );
                self.launch_on(
                    stream,
                    self.k.moe_gu_g,
                    ((2 * ff / 16) as u32, groups, 1),
                    256,
                    (16 * 16 * r16g) as u32,
                    tang_moe::args![gw, uw, sg, su, ids, mxq, egu, ti, r16g, fastk, plan],
                );
                self.launch_on(
                    stream,
                    self.k.silu_q,
                    ((ff / 32) as u32, np, 1),
                    32,
                    0,
                    tang_moe::args![egu, hq, pb],
                );
                self.launch_on(
                    stream,
                    self.k.moe_down_g,
                    ((nh / 16) as u32, groups, 1),
                    256,
                    (16 * 16 * r16d) as u32,
                    tang_moe::args![dw, sd, ids, hq, parts, ti, r16d, fastk, plan],
                );
            }
            return;
        }
        // Staged (8 warps × 2 rows, rows in shared memory) or direct (8 warps × R rows).
        let moe = match self.var.moe {
            usize::MAX if t <= 4 => 2,
            usize::MAX => 4,
            v => v,
        };
        let (gu_f, down_f, rows, smem_g, smem_d) = match moe {
            2 => (self.k.moe_gu_d[0], self.k.moe_down_d[0], 16, 0, 0),
            4 => (self.k.moe_gu_d[1], self.k.moe_down_d[1], 32, 0, 0),
            // Double-buffered: 8 warps × 2 rows × 4 groups a block, two buffers.
            8 => (
                self.k.moe_gu_db,
                self.k.moe_down_db,
                64,
                (2 * 16 * 16 * r16g) as u32,
                (2 * 16 * 16 * r16d) as u32,
            ),
            _ => (
                self.k.moe_gu,
                self.k.moe_down,
                16,
                (16 * 16 * r16g) as u32,
                (16 * 16 * r16d) as u32,
            ),
        };
        unsafe {
            self.launch_on(
                stream,
                gu_f,
                ((2 * ff / rows) as u32, np, 1),
                256,
                smem_g,
                tang_moe::args![gw, uw, sg, su, ids, mxq, egu, ti, r16g, fastk, pb],
            );
            self.launch_on(
                stream,
                self.k.silu_q,
                ((ff / 32) as u32, np, 1),
                32,
                0,
                tang_moe::args![egu, hq, pb],
            );
            self.launch_on(
                stream,
                down_f,
                ((nh / rows) as u32, np, 1),
                256,
                smem_d,
                tang_moe::args![dw, sd, ids, hq, parts, ti, r16d, fastk, pb],
            );
        }
    }

    /// Side stream: start after everything enqueued so far on the main stream.
    fn fork(&self) {
        self.ev_fork.record(&self.stream).expect("record");
        self.side.wait(&self.ev_fork).expect("wait");
    }

    /// Main stream: wait for everything enqueued so far on the side stream.
    fn join(&self) {
        self.ev_join.record(&self.side).expect("record");
        self.stream.wait(&self.ev_join).expect("wait");
    }

    fn enqueue_head(&self, t: usize) {
        let v = self.hp.n_vocab;
        self.gemv(
            "head",
            &self.head,
            self.a(&self.s.x),
            Some(self.a(&self.s.mxq)),
            self.a(&self.s.head),
            t,
            v,
        );
        let (lg, ids, n, part, np, ctl) = (
            self.a(&self.s.head),
            self.a(&self.s.out_ids),
            v as i32,
            self.a(&self.s.amax),
            64i32,
            self.a(&self.s.ctl),
        );
        if let Some(sp) = self.sampler.as_ref().filter(|_| !self.greedy.get()) {
            if !self.skip("argmax") {
                sp.enqueue(
                    &self.dev,
                    &self.stream,
                    &self.s.head,
                    &self.s.ctl,
                    &self.s.out_ids,
                    t,
                );
            }
            self.kcopy(self.io_dev(Io::IDS), ids, t * 4);
            return;
        }
        unsafe {
            self.launch(
                self.k.argmax1,
                (64, t as u32, 1),
                1024,
                tang_moe::args![lg, n, part, ctl],
            );
            self.launch(
                self.k.argmax2,
                (t as u32, 1, 1),
                32,
                tang_moe::args![part, np, ids],
            );
        }
        self.kcopy(self.io_dev(Io::IDS), ids, t * 4);
    }

    /// Everything a window enqueues; with `probe`, eager, syncing after each layer to read its
    /// last token's tensors.
    fn enqueue_window(
        &self,
        t: usize,
        verify: bool,
        nchunk: usize,
        nseg: usize,
        mut probe: Option<&mut Probe>,
    ) -> Result<()> {
        self.enqueue_inputs(t);
        let n = self.hp.n_embd;
        let last = |b: &B| -> Vec<f32> { self.dev.download(b)[(t - 1) * n..t * n].to_vec() };
        let nl = self.layers.len();
        self.norm_q8(None, &self.layers[0].attn_norm, t);
        for l in 0..nl {
            let layer = &self.layers[l];
            self.enqueue_mixer(l, t, verify, nchunk, nseg);
            self.norm_q8(Some(self.a(&self.s.mix)), &layer.post_norm, t);
            if let Some(p) = probe.as_deref_mut() {
                self.dev.sync();
                p.mixer_out.push(last(&self.s.mix));
                p.post_mixer.push(last(&self.s.r));
            }
            let next = if l + 1 < nl {
                &self.layers[l + 1].attn_norm
            } else {
                &self.out_norm
            };
            self.enqueue_moe(l, t, next, probe.is_some());
            if let Some(p) = probe.as_deref_mut() {
                self.dev.sync();
                p.moe_out.push(last(&self.s.moe_y));
                p.post_moe.push(last(&self.s.r));
                let k = self.hp.n_expert_used;
                let ids = self.dev.download(&self.s.ids);
                p.router_ids.push(
                    ids[(t - 1) * k..t * k]
                        .iter()
                        .map(|v| v.to_bits())
                        .collect(),
                );
                p.qsa_sel.push(None);
            }
        }
        self.enqueue_head(t);
        if let Some(m) = &self.mtp {
            self.kcopy(self.a(&m.mh), self.a(&self.s.x), t * n * 4);
        }
        Ok(())
    }

    /// The MTP layer on `t` staged rows: input from `m.emb` (staged) and `m.mh`, attention over
    /// `nchunk` chunks, then the head; ids and probabilities to the pinned `IDS` words, and the
    /// last row's normed output into `m.mh` row 0 (the next chained step's input).
    fn enqueue_mtp(&self, t: usize, nchunk: usize) {
        let m = self.mtp.as_ref().expect("MTP layer");
        let (s, n) = (&self.s, self.hp.n_embd);
        self.kcopy(self.a(&s.rows), self.io_dev(Io::rows(n)), ROWS_U32 * 4);
        self.kcopy(self.a(&m.emb), self.io_dev(Io::EMB), t * n * 4);
        let (emb, mh, we, wh, eps, ni, cat) = (
            self.a(&m.emb),
            self.a(&m.mh),
            self.a(&m.enorm),
            self.a(&m.hnorm),
            self.eps,
            n as i32,
            self.a(&m.cat),
        );
        unsafe {
            self.launch(
                m.f_in,
                (t as u32, 2, 1),
                1024,
                tang_moe::args![emb, mh, we, wh, eps, ni, cat],
            )
        };
        self.gemv("mtp_fc", &m.fc, cat, None, self.a(&s.r), t, n);
        self.norm_q8(None, &m.layer.attn_norm, t);
        self.enqueue_mixer(MTP_L, t, true, nchunk, 1);
        self.norm_q8(Some(self.a(&s.mix)), &m.layer.post_norm, t);
        self.enqueue_moe(MTP_L, t, &m.onorm, false);
        let v = self.hp.n_vocab;
        self.gemv(
            "mtp_head",
            &self.head,
            self.a(&s.x),
            Some(self.a(&s.mxq)),
            self.a(&s.head),
            t,
            v,
        );
        let (lg, vi, ids, prob) = (self.a(&s.head), v as i32, self.a(&m.ids), self.a(&m.prob));
        unsafe {
            self.launch(
                m.f_pick,
                (t as u32, 1, 1),
                1024,
                tang_moe::args![lg, vi, ids, prob],
            )
        };
        self.kcopy(self.io_dev(Io::IDS), ids, t * 4);
        self.kcopy(self.io_dev(Io::IDS + 64), prob, t * 4);
        self.kcopy(mh, self.a(&s.x) + ((t - 1) * n * 4) as u64, n * 4);
    }

    fn enqueue_commit(&self, t: usize) {
        self.kcopy(self.a(&self.s.ctl), self.io_dev(Io::CCTL), 32);
        for layer in &self.layers {
            if let Some(g) = &layer.gdn {
                self.gdn_step(g, layer.w_in.rows, t, true);
                self.gdn_commit(g, layer.w_in.rows, t);
            }
        }
    }

    // ---- host side

    fn stage(&mut self, pos0: usize, t: usize) -> Result<()> {
        let ctl = self.io.u32s(Io::CTL, 5);
        ctl[0] = pos0 as u32;
        ctl[1] = t as u32;
        ctl[2] = self.counter;
        ctl[3] = self.seed;
        ctl[4] = self.temperature.to_bits();
        if let Some(sp) = self.sampler.as_mut() {
            sp.stage(&self.dev, &self.tokens, pos0, t)?;
        }
        let n = self.hp.n_embd;
        let emb_t = self.g.info("token_embd.weight")?.clone();
        let emb = self.io.f32s(Io::EMB, t * n);
        for i in 0..t {
            let row = self.g.rows(&emb_t, self.tokens[pos0 + i] as usize, 1)?;
            emb[i * n..(i + 1) * n].copy_from_slice(&row);
        }
        // One sequence (slot 0): rows at pos0.., one segment keeping all of them.
        let rows = self.io.u32s(Io::rows(n), ROWS_U32);
        rows.fill(0);
        for i in 0..t {
            rows[i] = (pos0 + i) as u32;
        }
        rows[2 * MAX_T..2 * MAX_T + 4].copy_from_slice(&[0, t as u32, 0, t as u32]);
        Ok(())
    }

    /// The MTP layer on `next.len()` rows at positions `pos0 + 1..`: row `i` takes the final
    /// hidden state `m.mh` holds for position `pos0 + i` and the token `next[i]` at
    /// `pos0 + i + 1`, and guesses the token at `pos0 + i + 2`. Returns each row's guess and its
    /// probability.
    fn mtp_pass(&mut self, pos0: usize, next: &[u32]) -> Result<Vec<(u32, f32)>> {
        let (t, n) = (next.len(), self.hp.n_embd);
        ensure!((1..=MAX_T).contains(&t) && pos0 + t < self.max_ctx);
        let emb_t = self.g.info("token_embd.weight")?.clone();
        let emb = self.io.f32s(Io::EMB, t * n);
        for (i, &tok) in next.iter().enumerate() {
            let row = self.g.rows(&emb_t, tok as usize, 1)?;
            emb[i * n..(i + 1) * n].copy_from_slice(&row);
        }
        let rows = self.io.u32s(Io::rows(n), ROWS_U32);
        rows.fill(0);
        for (i, r) in rows.iter_mut().take(t).enumerate() {
            *r = (pos0 + 1 + i) as u32;
        }
        rows[2 * MAX_T..2 * MAX_T + 4].copy_from_slice(&[0, t as u32, 0, t as u32]);
        let nchunk = self.nchunk(pos0 + t + 1);
        if self.use_graphs && self.prof.is_none() {
            let key = (t, nchunk);
            if let Some(g) = self.mtp_graphs.get(&key) {
                g.launch(&self.stream).map_err(|e| anyhow!("{e}"))?;
            } else {
                self.enqueue_mtp(t, nchunk);
                self.dev.sync();
                let out = self.mtp_read(t);
                let stream = ManuallyDrop::new(Stream(self.dev.cu_stream()));
                let me = self as *const Self;
                let gr = Graph::capture(&stream, |_| {
                    // SAFETY: `self` is not otherwise touched during capture.
                    unsafe { &*me }.enqueue_mtp(t, nchunk);
                    Ok(())
                })
                .map_err(|e| anyhow!("capture MTP T={t}: {e}"))?;
                self.mtp_graphs.insert(key, gr);
                return Ok(out);
            }
        } else {
            self.enqueue_mtp(t, nchunk);
        }
        self.dev.sync();
        Ok(self.mtp_read(t))
    }

    fn mtp_read(&self, t: usize) -> Vec<(u32, f32)> {
        let ids = self.io.u32s(Io::IDS, t);
        let p = self.io.f32s(Io::IDS + 64, t);
        ids.iter().zip(p.iter()).map(|(&d, &p)| (d, p)).collect()
    }

    /// After a window over positions `pos..pos + next.len()` (whose final hidden rows `m.mh`
    /// holds), feed the MTP layer the tokens that followed them (`next`), so its K/V cache
    /// covers those positions; with `chain`, then draft `mtp_steps` tokens past the last one
    /// into `mtp_last`.
    fn mtp_after(&mut self, pos: usize, next: &[u32], chain: bool) -> Result<()> {
        self.mtp_last.clear();
        if self.mtp.is_none() || !self.use_mtp || !self.greedy.get() || next.is_empty() {
            return Ok(());
        }
        let t0 = Instant::now();
        let end = pos + next.len();
        if end + self.mtp_steps + 1 >= self.max_ctx {
            return Ok(());
        }
        let out = self.mtp_pass(pos, next)?;
        if chain {
            let mut d = *out.last().unwrap();
            self.mtp_last.push(d);
            for j in 1..self.mtp_steps {
                d = self.mtp_pass(end + j - 1, &[d.0])?[0];
                self.mtp_last.push(d);
            }
        }
        self.last_mtp_ms = t0.elapsed().as_secs_f64() * 1e3;
        Ok(())
    }

    /// After a [`window`](Self::window) over `pos0..pos0 + next.len()`: feed the MTP layer
    /// the tokens that followed each row, drafting a chain when it reaches the sequence's end.
    pub fn mtp_feed(&mut self, pos0: usize, next: &[u32]) -> Result<()> {
        let end = pos0 + next.len() >= self.tokens.len();
        self.mtp_after(pos0, next, end)
    }

    pub fn window(&mut self, pos0: usize, t: usize, probe: Option<&mut Probe>) -> Result<Vec<u32>> {
        self.window_mode(pos0, t, false, probe)
    }

    /// [`window`](Self::window) as a verify window (`verify`): the GDN state is only read; call
    /// [`commit`](Self::commit) with the number of tokens kept before the next window.
    pub fn window_mode(
        &mut self,
        pos0: usize,
        t: usize,
        verify: bool,
        probe: Option<&mut Probe>,
    ) -> Result<Vec<u32>> {
        ensure!((1..=MAX_T).contains(&t) && pos0 + t <= self.tokens.len());
        ensure!(
            pos0 + t <= self.max_ctx,
            "context {} > max {}",
            pos0 + t,
            self.max_ctx
        );
        let w0 = Instant::now();
        let mut st = WinStats {
            t,
            ..Default::default()
        };
        self.stage(pos0, t)?;
        st.host_prep_ms = w0.elapsed().as_secs_f64() * 1e3;
        let nchunk = self.nchunk(pos0 + t);
        self.run_staged(st, w0, t, verify, nchunk, 1, probe)
    }

    /// Run the window `stage` / `stage_rows` staged: `t` rows in `nseg` segments, attention over
    /// `nchunk` chunks. A graph per (t, verify, nchunk, nseg), captured on first use.
    #[allow(clippy::too_many_arguments)]
    fn run_staged(
        &mut self,
        st: WinStats,
        w0: Instant,
        t: usize,
        verify: bool,
        nchunk: usize,
        nseg: usize,
        mut probe: Option<&mut Probe>,
    ) -> Result<Vec<u32>> {
        self.greedy.set(
            self.temperature == 0.0 && self.sampler.as_ref().is_none_or(|s| s.presence == 0.0),
        );
        let key = (t, verify, nchunk, nseg, self.greedy.get());
        if probe.is_none() && self.use_graphs {
            if !self.graphs.contains_key(&key) {
                // First use: run eagerly (loads every kernel), then capture.
                self.enqueue_window(t, verify, nchunk, nseg, None)?;
                self.dev.sync();
                let ids = self.io.u32s(Io::IDS, t).to_vec();
                // Undo nothing: the eager run *was* this window. Capture for next time without
                // running it (capture records, it doesn't execute).
                let stream = ManuallyDrop::new(Stream(self.dev.cu_stream()));
                let me = self as *const Self;
                let gr = Graph::capture(&stream, |_| {
                    // SAFETY: `self` is not otherwise touched during capture.
                    unsafe { &*me }
                        .enqueue_window(t, verify, nchunk, nseg, None)
                        .map_err(|e| tang_moe::gpu::Error(format!("{e}")))
                })
                .map_err(|e| anyhow!("capture T={t}: {e}"))?;
                self.graphs.insert(key, gr);
                return Ok(self.finish(st, w0, ids));
            }
            self.graphs[&key]
                .launch(&self.stream)
                .map_err(|e| anyhow!("{e}"))?;
        } else {
            self.enqueue_window(t, verify, nchunk, nseg, probe.as_deref_mut())?;
        }
        self.dev.sync();
        self.prof_collect();
        let ids = self.io.u32s(Io::IDS, t).to_vec();
        if let Some(p) = probe {
            let n = self.hp.n_embd;
            p.final_x = self.dev.download(&self.s.x)[(t - 1) * n..t * n].to_vec();
        }
        Ok(self.finish(st, w0, ids))
    }

    fn finish(&mut self, mut st: WinStats, w0: Instant, ids: Vec<u32>) -> Vec<u32> {
        st.routed = st.t * self.hp.n_expert_used * self.hp.n_layer;
        st.wall_ms = w0.elapsed().as_secs_f64() * 1e3;
        self.counter = self.counter.wrapping_add(1);
        self.last = st;
        ids
    }

    /// After a verify window of `t`: keep the first `n_keep` tokens' recurrent state.
    pub fn commit(&mut self, t: usize, n_keep: usize) -> Result<()> {
        ensure!((1..=t).contains(&n_keep));
        let c = self.io.u32s(Io::CCTL, 2);
        c[0] = 0;
        c[1] = n_keep as u32;
        if !self.use_graphs {
            self.enqueue_commit(t);
            return Ok(());
        }
        if self.commit_graphs[t].is_none() {
            self.enqueue_commit(t);
            self.dev.sync();
            let stream = ManuallyDrop::new(Stream(self.dev.cu_stream()));
            let me = self as *const Self;
            let gr = Graph::capture(&stream, |_| {
                // SAFETY: `self` is not otherwise touched during capture.
                unsafe { &*me }.enqueue_commit(t);
                Ok(())
            })
            .map_err(|e| anyhow!("capture commit T={t}: {e}"))?;
            self.commit_graphs[t] = Some(gr);
            return Ok(());
        }
        self.commit_graphs[t]
            .as_ref()
            .unwrap()
            .launch(&self.stream)
            .map_err(|e| anyhow!("{e}"))
    }

    /// Download the last window's logits (`[t][vocab]`).
    pub fn logits(&self, t: usize) -> Vec<f32> {
        let v = self.hp.n_vocab;
        self.dev.download(&self.s.head)[..t * v].to_vec()
    }

    /// Feed `ids` as a prompt (windows of up to `MAX_T`), then return the next token.
    pub fn prefill(
        &mut self,
        ids: &[u32],
        chunk: usize,
        mut logits_cb: Option<LogitsCb<'_>>,
    ) -> Result<u32> {
        ensure!(!ids.is_empty(), "empty prompt");
        let mut pos = self.tokens.len();
        self.tokens.extend_from_slice(ids);
        let mut next = 0;
        while pos < self.tokens.len() {
            let t = chunk.min(self.tokens.len() - pos).clamp(1, MAX_T);
            let out = self.window(pos, t, None)?;
            if let Some(cb) = logits_cb.as_deref_mut() {
                cb(pos, t, &self.logits(t));
            }
            next = out[t - 1];
            if self.mtp.is_some() {
                let mut nx = self.tokens[pos + 1..pos + t].to_vec();
                let last = pos + t == self.tokens.len();
                nx.push(if last { next } else { self.tokens[pos + t] });
                self.mtp_after(pos, &nx, last)?;
            }
            pos += t;
        }
        // Profile decode only.
        if let Some(p) = &self.prof {
            p.borrow_mut().total.clear();
        }
        Ok(next)
    }

    /// Run `cur` and `drafts` as one verify window; returns the kept tokens (the accepted
    /// drafts' successors plus the first correction), and commits them.
    pub fn verify(&mut self, cur: u32, drafts: &[u32]) -> Result<Vec<u32>> {
        let pos = self.tokens.len();
        let t = 1 + drafts.len();
        ensure!(t <= MAX_T, "{} drafts", drafts.len());
        self.tokens.push(cur);
        self.tokens.extend_from_slice(drafts);
        if t == 1 {
            let next = self.window(pos, 1, None)?[0];
            self.mtp_after(pos, &[next], true)?;
            return Ok(vec![next]);
        }
        let targets = self.window_mode(pos, t, true, None)?;
        let mut n = 1;
        while n < t && drafts[n - 1] == targets[n - 1] {
            n += 1;
        }
        let tc = Instant::now();
        self.commit(t, n)?;
        self.tokens.truncate(pos + n);
        self.last_commit_ms = tc.elapsed().as_secs_f64() * 1e3;
        self.mtp_after(pos, &targets[..n], true)?;
        Ok(targets[..n].to_vec())
    }

    /// Append `tok` and run it as a T=1 window; returns the next token.
    pub fn step(&mut self, tok: u32) -> Result<u32> {
        self.tokens.push(tok);
        let pos = self.tokens.len() - 1;
        let next = self.window(pos, 1, None)?[0];
        self.mtp_after(pos, &[next], true)?;
        Ok(next)
    }

    /// An MTP layer is loaded (`CoderOpts::mtp`).
    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    pub fn gguf(&self) -> &Gguf {
        &self.g
    }

    pub fn vocab(&self) -> Result<Vec<String>> {
        let v = self.g.meta("tokenizer.ggml.tokens")?;
        let a = v.as_array().context("tokens")?;
        Ok(a.iter()
            .map(|x| x.as_str().unwrap_or("").to_string())
            .collect())
    }

    /// Time each stage of a T=1 window alone: a graph of that stage's launches for every layer
    /// (the window with every other launch filtered out), replayed back to back. Returns
    /// (stage, µs a step, bytes a step), the whole window first. Run after a decode (the
    /// routing buffers then hold real expert ids).
    pub fn stage_sweep(&mut self) -> Result<Vec<(&'static str, f64, f64)>> {
        ensure!(!self.tokens.is_empty(), "decode something first");
        let pos0 = self.tokens.len() - 1;
        self.stage(pos0, 1)?;
        let nchunk = self.nchunk(pos0 + 1);
        self.enqueue_window(1, false, nchunk, 1, None)?;
        self.dev.sync();
        let hp = &self.hp;
        let lb = |f: &dyn Fn(&Layer) -> f64| -> f64 { self.layers.iter().map(f).sum() };
        let ex = |e: &Experts, n: usize| e.w.estride as f64 * n as f64;
        let k = hp.n_expert_used;
        let stages: Vec<(&'static str, f64)> = vec![
            ("", self.step_bytes(pos0 + 1)),
            (
                "gdn_in",
                lb(&|l| {
                    if l.gdn.is_some() {
                        dense_bytes(&l.w_in)
                    } else {
                        0.0
                    }
                }),
            ),
            ("gdn_conv_step", 0.0),
            ("gdn_commit", 0.0),
            (
                "gdn_out",
                lb(&|l| {
                    if l.gdn.is_some() {
                        dense_bytes(&l.w_out)
                    } else {
                        0.0
                    }
                }),
            ),
            (
                "attn_in",
                lb(&|l| {
                    if l.attn.is_some() {
                        dense_bytes(&l.w_in)
                    } else {
                        0.0
                    }
                }),
            ),
            ("attn_prep", 0.0),
            (
                "attn_part",
                lb(&|l| {
                    if l.attn.is_some() {
                        (pos0 * hp.n_head_kv * hp.head_dim * 4) as f64
                    } else {
                        0.0
                    }
                }),
            ),
            ("attn_merge", 0.0),
            (
                "attn_out",
                lb(&|l| {
                    if l.attn.is_some() {
                        dense_bytes(&l.w_out)
                    } else {
                        0.0
                    }
                }),
            ),
            ("norm_q8", 0.0),
            ("router", lb(&|l| dense_bytes(&l.router))),
            ("route", 0.0),
            (
                "moe_gu",
                lb(&|l| ex(&l.gate, k) + ex(&l.up, k) + ex(&l.sh_gate, 1) + ex(&l.sh_up, 1)),
            ),
            ("silu_q", 0.0),
            ("moe_down", lb(&|l| ex(&l.down, k) + ex(&l.sh_down, 1))),
            ("moe_add_norm", 0.0),
            ("head", dense_bytes(&self.head)),
            ("argmax", 0.0),
        ];
        const REPS: usize = 4;
        const RUNS: usize = 10;
        let mut out = Vec::new();
        for (name, bytes) in stages {
            self.only
                .set(if name.is_empty() { None } else { Some(name) });
            let stream = ManuallyDrop::new(Stream(self.dev.cu_stream()));
            let me = self as *const Self;
            let gr = Graph::capture(&stream, |_| {
                for _ in 0..REPS {
                    // SAFETY: `self` is not otherwise touched during capture.
                    unsafe { &*me }
                        .enqueue_window(1, false, nchunk, 1, None)
                        .map_err(|e| tang_moe::gpu::Error(format!("{e}")))?;
                }
                Ok(())
            });
            self.only.set(None);
            let gr = gr.map_err(|e| anyhow!("capture {name}: {e}"))?;
            gr.launch(&self.stream).map_err(|e| anyhow!("{e}"))?;
            let (e0, e1) = (
                gpu::Event::new(true).map_err(|e| anyhow!("{e}"))?,
                gpu::Event::new(true).map_err(|e| anyhow!("{e}"))?,
            );
            e0.record(&self.stream).map_err(|e| anyhow!("{e}"))?;
            for _ in 0..RUNS {
                gr.launch(&self.stream).map_err(|e| anyhow!("{e}"))?;
            }
            e1.record(&self.stream).map_err(|e| anyhow!("{e}"))?;
            e1.sync().map_err(|e| anyhow!("{e}"))?;
            let us = e1.since(&e0).map_err(|e| anyhow!("{e}"))? as f64 * 1e3 / (REPS * RUNS) as f64;
            out.push((if name.is_empty() { "window" } else { name }, us, bytes));
        }
        Ok(out)
    }

    /// Switch kernel variants (`kq=N`, `fastk=N`, `shpar=1`, comma-separated; empty for the
    /// defaults) and drop the captured graphs.
    pub fn set_variants(&mut self, spec: &str) -> Result<()> {
        let mut v = Variants::default();
        for kv in spec.split(',').filter(|x| !x.is_empty()) {
            match kv.split_once('=') {
                Some(("kq", n)) => v.kq = n.parse()?,
                Some(("fastk", n)) => v.fastk = n.parse()?,
                Some(("shpar", n)) => v.shared_par = n == "1",
                _ => anyhow::bail!("unknown variant {kv}"),
            }
        }
        self.var = v;
        self.dev.sync();
        self.graphs.clear();
        Ok(())
    }

    /// Bytes one decode step reads from VRAM at context `ctx`: every dense weight, the
    /// routed experts a token uses (top-k per layer, at the layer's average expert size), and
    /// the attention cache. The speed-of-light denominator.
    pub fn step_bytes(&self, ctx: usize) -> f64 {
        let hp = &self.hp;
        let mut b = 0f64;
        for l in &self.layers {
            for d in [&l.w_in, &l.w_out, &l.router] {
                b += dense_bytes(d);
            }
            for e in [&l.sh_gate, &l.sh_up, &l.sh_down] {
                b += e.w.estride as f64;
            }
            for e in [&l.gate, &l.up, &l.down] {
                b += e.w.estride as f64 * hp.n_expert_used as f64;
            }
            if l.attn.is_some() {
                b += (ctx * hp.n_head_kv * hp.head_dim * 2 * 2) as f64;
            }
            if l.gdn.is_some() {
                // the recurrent state, read and written
                b += (2 * self.geo.gdn_hv * 128 * 128 * 4) as f64;
            }
        }
        b + dense_bytes(&self.head)
    }
}

/// One decoder layer's weights and state buffers from `g`, its tensors named `{pre}…` (a GDN
/// layer when `rec`, else gated attention), with a K/V cache of `max_ctx`.
#[allow(clippy::too_many_arguments)]
fn load_layer(
    dev: &CudaComputeDevice,
    g: &Gguf,
    hp: &Hparams,
    geo: &Geometry,
    pre: &str,
    rec: bool,
    ba_perm: &[usize],
    max_ctx: usize,
) -> Result<Layer> {
    let info = |name: &str| -> Result<TensorInfo> { Ok(g.info(name)?.clone()) };
    let vecf = |name: &str| -> Result<B> { Ok(dev.upload_f32(&g.tensor_f32(name)?)) };
    let hv = hp.ssm_v_heads;
    let t = |s: &str| format!("{pre}{s}");
    let (w_in, gdn, attn) = if rec {
        let w_in = upload_dense(
            &dev,
            &g,
            &[
                (info(&t("attn_qkv.weight"))?, None),
                (info(&t("attn_gate.weight"))?, None),
                (info(&t("ssm_ba.weight"))?, Some(&ba_perm[..])),
            ],
        )?;
        ensure!(
            w_in.rows == geo.gdn_b() + hv,
            "{pre} GDN input rows {}",
            w_in.rows
        );
        let gd = GdnL {
            conv: vecf(&t("ssm_conv1d.weight"))?,
            dt: vecf(&t("ssm_dt.bias"))?,
            a: vecf(&t("ssm_a"))?,
            norm: vecf(&t("ssm_norm.weight"))?,
            state: dev.alloc_f32(hv * 128 * 128),
            hist: dev.alloc_f32(3 * geo.gdn_conv()),
            proj: dev.alloc_f32(MAX_T * w_in.rows),
            stab: dev.alloc_f32(2 * MAX_SLOTS),
            htab: dev.alloc_f32(2 * MAX_SLOTS),
        };
        (w_in, Some(gd), None)
    } else {
        let w_in = upload_dense(
            &dev,
            &g,
            &[
                (info(&t("attn_q.weight"))?, None),
                (info(&t("attn_k.weight"))?, None),
                (info(&t("attn_v.weight"))?, None),
            ],
        )?;
        ensure!(
            w_in.rows == (2 * hp.n_head + 2 * hp.n_head_kv) * hp.head_dim,
            "{pre} attention input rows {}",
            w_in.rows
        );
        let at = AttnL {
            qn: vecf(&t("attn_q_norm.weight"))?,
            kn: vecf(&t("attn_k_norm.weight"))?,
            k: dev.alloc_bf16(max_ctx * hp.n_head_kv * hp.head_dim),
            v: dev.alloc_bf16(max_ctx * hp.n_head_kv * hp.head_dim),
            proj: dev.alloc_f32(MAX_T * w_in.rows),
            ktab: dev.alloc_f32(2 * MAX_SLOTS),
            vtab: dev.alloc_f32(2 * MAX_SLOTS),
        };
        (w_in, None, Some(at))
    };
    let w_out = upload_dense(
        &dev,
        &g,
        &[(
            info(&t(if rec {
                "ssm_out.weight"
            } else {
                "attn_output.weight"
            }))?,
            None,
        )],
    )?;
    let router = upload_dense(
        &dev,
        &g,
        &[
            (info(&t("ffn_gate_inp.weight"))?, None),
            (info(&t("ffn_gate_inp_shexp.weight"))?, None),
        ],
    )?;
    ensure!(router.rows == hp.n_expert + 1, "{pre} router rows");
    let sh_gate = upload_experts(&dev, &g, &info(&t("ffn_gate_shexp.weight"))?)?;
    let sh_up = upload_experts(&dev, &g, &info(&t("ffn_up_shexp.weight"))?)?;
    let sh_down = upload_experts(&dev, &g, &info(&t("ffn_down_shexp.weight"))?)?;
    ensure!(
        sh_gate.w.estride as usize / sh_gate.w.row_bytes as usize == hp.n_ff_exp,
        "{pre}: the shared expert's FF must equal the routed experts' ({})",
        hp.n_ff_exp
    );
    let gate = upload_experts(&dev, &g, &info(&t("ffn_gate_exps.weight"))?)?;
    let up = upload_experts(&dev, &g, &info(&t("ffn_up_exps.weight"))?)?;
    let down = upload_experts(&dev, &g, &info(&t("ffn_down_exps.weight"))?)?;
    Ok(Layer {
        attn_norm: vecf(&t("attn_norm.weight"))?,
        post_norm: vecf(&t("post_attention_norm.weight"))?,
        w_in,
        w_out,
        router,
        r16_gu: [&gate, &up, &sh_gate, &sh_up]
            .iter()
            .map(|e| (e.w.row_bytes + 15) / 16 + 1)
            .max()
            .unwrap(),
        r16_down: [&down, &sh_down]
            .iter()
            .map(|e| (e.w.row_bytes + 15) / 16 + 1)
            .max()
            .unwrap(),
        sh_gate,
        sh_up,
        sh_down,
        gate,
        up,
        down,
        gdn,
        attn,
    })
}

fn dense_bytes(d: &Dense) -> f64 {
    d.bytes as f64
}

/// Stack `parts` (tensor, optional row permutation: output row `i` is input row `perm[i]`) by
/// rows into one native-GEMV weight. Types the GEMV doesn't decode are dequantized to f32.
fn upload_dense(
    dev: &CudaComputeDevice,
    g: &Gguf,
    parts: &[(TensorInfo, Option<&[usize]>)],
) -> Result<Dense> {
    let k = parts[0].0.row_len();
    let mut ws = Vec::new();
    let mut words = Vec::new();
    let mut segs16 = Vec::new();
    let mut off = 0usize;
    let mut total = 0usize;
    for (t, perm) in parts {
        ensure!(
            t.row_len() == k,
            "{}: {} wide, want {k}",
            t.name,
            t.row_len()
        );
        let rows = t.n_rows();
        let (ty, rb, mut bytes) = match native_id(t.ty) {
            Some(id) => (id, t.row_bytes()?, g.bytes(t).to_vec()),
            None => {
                let v = g.dequantize(t)?;
                (0, k * 4, v.iter().flat_map(|x| x.to_le_bytes()).collect())
            }
        };
        if let Some(p) = perm {
            ensure!(
                p.len() == rows,
                "{}: permutation of {} rows",
                t.name,
                p.len()
            );
            let src = bytes.clone();
            for (i, &j) in p.iter().enumerate() {
                bytes[i * rb..(i + 1) * rb].copy_from_slice(&src[j * rb..(j + 1) * rb]);
            }
        }
        // TANG_CODER_ROUTER16=1: F32 rows (the router) as bf16, half the bytes.
        let (ty, rb, bytes) = if ty == 0 && env_is("TANG_CODER_ROUTER16", "1") {
            let b16: Vec<u8> = bytes
                .chunks(4)
                .flat_map(|c| {
                    let x = f32::from_le_bytes([c[0], c[1], c[2], c[3]]).to_bits();
                    (((x + 0x7fff + ((x >> 16) & 1)) >> 16) as u16).to_le_bytes()
                })
                .collect();
            (30, k * 2, b16)
        } else if ty == 8 && k.is_multiple_of(1024) && !env_is("TANG_CODER_Q8X", "0") {
            // Q8X (type 108, default; TANG_CODER_Q8X=0 keeps Q8_0): [k codes][k / 32 f16 scales] a
            // row, lossless.
            let nb = k / 32;
            let mut out = Vec::with_capacity(rows * (k + 2 * nb));
            for r in 0..rows {
                let row = &bytes[r * rb..(r + 1) * rb];
                for b in 0..nb {
                    out.extend_from_slice(&row[b * 34 + 2..b * 34 + 34]);
                }
                for b in 0..nb {
                    out.extend_from_slice(&row[b * 34..b * 34 + 2]);
                }
            }
            (108, k + 2 * nb, out)
        } else {
            (ty, rb, bytes)
        };
        let mut bytes = bytes;
        // 16 bytes of slack: the GEMV stages rows with aligned 16-byte loads.
        total += rows * rb;
        bytes.extend_from_slice(&[0u8; 16]);
        let w = dev.upload_bytes(&bytes);
        let p = dev.buffer_addr(&w);
        words.extend_from_slice(&[
            p as u32,
            (p >> 32) as u32,
            ty,
            rows as u32,
            rb as u32,
            off as u32,
        ]);
        if ty != 0 && ty != 30 && ty != 108 {
            segs16.push(rb);
        }
        ws.push(w);
        off += rows;
    }
    let rb = segs16.into_iter().max().unwrap_or(0);
    Ok(Dense {
        segs: dev.upload_u32(&words),
        nseg: parts.len() as i32,
        rows: off,
        k,
        row16: ((rb + 15).div_ceil(16) + 1) as i32,
        bytes: total,
        _w: ws,
    })
}

/// A `[n_expert][rows][k]` expert tensor into VRAM in its GGUF type (Q8_0 when the expert
/// kernels don't decode it).
fn upload_experts(dev: &CudaComputeDevice, g: &Gguf, t: &TensorInfo) -> Result<Experts> {
    ensure!(
        t.dims.len() == 3 || t.dims.len() == 2,
        "{}: not an expert stack or matrix",
        t.name
    );
    let (k, rows) = (t.dims[0] as usize, t.dims[1] as usize);
    let ne = t.dims.get(2).map_or(1, |&d| d as usize);
    let (ty, rb, mut bytes) = match native_id(t.ty).filter(|&id| id != 0 && id != 30) {
        Some(id) => (id, t.row_bytes()?, g.bytes(t).to_vec()),
        None => {
            let mut out = Vec::with_capacity(ne * rows * k / 32 * 34);
            for e in 0..ne {
                let v = if ne == 1 {
                    g.dequantize(t)?
                } else {
                    g.expert(t, e)?
                };
                out.extend_from_slice(&quantize_q8_0(&v));
            }
            (8, k / 32 * 34, out)
        }
    };
    ensure!(
        bytes.len() == ne * rows * rb,
        "{}: {} bytes, want {}",
        t.name,
        bytes.len(),
        ne * rows * rb
    );
    bytes.extend_from_slice(&[0u8; 16]);
    let w = dev.upload_bytes(&bytes);
    let base = dev.buffer_addr(&w);
    if t.ty != GgmlType::from_id(ty) {
        eprintln!("coder: {} requantized to Q8_0", t.name);
    }
    Ok(Experts {
        w: ExW {
            base,
            estride: (rows * rb) as u64,
            ty: ty as i32,
            row_bytes: rb as i32,
        },
        _w: w,
    })
}

/// Is the GGUF at `path` a `qwen3next` model (this engine) rather than Flash-Next?
pub fn is_qwen3next(path: &Path) -> Result<bool> {
    let g = Gguf::open(path)?;
    Ok(g.meta_str("general.architecture")
        .is_ok_and(|a| a == "qwen3next"))
}

// ---- serving: states, sampler, warm-up

impl Coder {
    /// Copy the running state out: every GDN layer's recurrent state and conv history (the
    /// attention K/V is positional, see [`save_positional`](Self::save_positional)).
    pub fn save_running(&mut self) -> Result<super::engine::RunningState> {
        self.dev.sync();
        let mut parts = Vec::new();
        for g in self.layers.iter().filter_map(|l| l.gdn.as_ref()) {
            parts.push(self.dev.download(&g.state));
            parts.push(self.dev.download(&g.hist));
        }
        let n = parts.len() / 2;
        parts.push(Vec::new());
        super::engine::RunningState::from_parts(self.tokens.len(), n, parts)
    }

    /// Resume at `s.pos` with `prefix` (the `s.pos` tokens `s` was saved after).
    pub fn restore_running(
        &mut self,
        s: &super::engine::RunningState,
        prefix: &[u32],
    ) -> Result<()> {
        ensure!(
            prefix.len() == s.pos,
            "prefix of {} tokens for a state at {}",
            prefix.len(),
            s.pos
        );
        let parts = s.parts();
        let gdn: Vec<&mut GdnL> = self
            .layers
            .iter_mut()
            .filter_map(|l| l.gdn.as_mut())
            .collect();
        ensure!(
            s.gdn_layers() == gdn.len() && parts.len() == 2 * gdn.len() + 1,
            "state is for another model"
        );
        self.dev.sync();
        for (i, g) in gdn.into_iter().enumerate() {
            self.dev.upload_into_f32(&mut g.state, parts[2 * i]);
            self.dev.upload_into_f32(&mut g.hist, parts[2 * i + 1]);
        }
        self.dev.sync();
        self.tokens.clear();
        self.tokens.extend_from_slice(prefix);
        Ok(())
    }

    /// The positional buffers (attention K and V per layer): device address, bytes a position.
    fn positional_bufs(&self) -> Vec<(u64, usize)> {
        let row = self.hp.n_head_kv * self.hp.head_dim * 2;
        self.layers
            .iter()
            .chain(self.mtp.iter().map(|m| &m.layer))
            .filter_map(|l| l.attn.as_ref())
            .flat_map(|a| [(self.a(&a.k), row), (self.a(&a.v), row)])
            .collect()
    }

    /// Copy positions `from..to` of the positional state out.
    pub fn save_positional(&mut self, from: usize, to: usize) -> Result<super::engine::Positional> {
        ensure!(from <= to && to <= self.max_ctx);
        self.dev.sync();
        let mut bufs = Vec::new();
        for (addr, row) in self.positional_bufs() {
            let mut host = vec![0u8; (to - from) * row];
            if !host.is_empty() {
                let r = unsafe {
                    cudarc::driver::sys::cuMemcpyDtoH_v2(
                        host.as_mut_ptr() as *mut _,
                        addr + (from * row) as u64,
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
        Ok(super::engine::Positional::from_parts(from, to, bufs))
    }

    /// Write positions `from.max(p.from)..p.to` of `p` back.
    pub fn restore_positional(&mut self, p: &super::engine::Positional, from: usize) -> Result<()> {
        let bufs = self.positional_bufs();
        ensure!(
            bufs.len() == p.parts().len(),
            "positional state is for another model"
        );
        self.dev.sync();
        let from = from.max(p.from);
        for ((addr, row), host) in bufs.into_iter().zip(p.parts()) {
            ensure!(
                host.len() == (p.to - p.from) * row,
                "positional state size differs"
            );
            if p.to > from {
                let src = &host[(from - p.from) * row..];
                let r = unsafe {
                    cudarc::driver::sys::cuMemcpyHtoD_v2(
                        addr + (from * row) as u64,
                        src.as_ptr() as *const _,
                        (p.to - from) * row,
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

    /// Sample in the head with top-k / top-p / a presence penalty (`flash::sampler`) instead of
    /// the Gumbel-max over the whole vocabulary. Window graphs are recaptured.
    pub fn enable_sampler(&mut self) -> Result<()> {
        if self.sampler.is_none() {
            self.sampler = Some(Box::new(super::sampler::Sampler::new(
                &self.gpu,
                &self.dev,
                self.hp.n_vocab,
            )?));
            self.dev.sync();
            self.graphs.clear();
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

    /// Capture every window width's graphs (plain and verify) and the commits on a throwaway
    /// sequence, then reset.
    pub fn warm(&mut self) -> Result<()> {
        self.reset();
        let ids: Vec<u32> = (0..16).map(|i| 1000 + i).collect();
        let mut cur = self.prefill(&ids, MAX_T, None)?;
        for t in 1..=MAX_T {
            let d: Vec<u32> = (0..t - 1).map(|i| 7 + i as u32).collect();
            cur = *self.verify(cur, &d)?.last().unwrap();
        }
        self.reset();
        Ok(())
    }
}
