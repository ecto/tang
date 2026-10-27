//! The forward on the device (M2): [`super::forward`]'s control flow over [`super::gpu`]'s
//! kernels, so every step has a CPU twin and the fixture checks the whole.
//!
//! On the device: every GEMM, norm, RoPE, quant-dequant, attention, mHC, the router, expert
//! MLPs and Engram's gate, and selection (kernels_select.cu, kernels_graph.cu): the
//! indexer's masks, candidate blocks and top-k, the per-expert routing lists (CSR; experts
//! launch for every id and exit early when empty), and the attention index rows. On the
//! host: Engram's hashing and row gather (the tables stay mapped on the host: 101 GB each
//! for the real model), uploaded through pinned staging; DSpark's left-to-right Markov
//! sampling; and, only when `DS_EXPERT_SLOTS` bounds the expert LRU, the routed-expert
//! counts. Weights upload lazily on first use, in their checkpoint format.
//!
//! Scratch and syncs ([`super::graph`]): a step's temporaries come from a per-model arena
//! released at the next step's start, so nothing is allocated or freed on the token path and
//! one stream orders all reuse. The host syncs only where it reads: the logits and DSpark
//! hiddens at the end of a step, `DS_TAP`, a forward hook, and the LRU's expert counts.
//!
//! CUDA graphs (`DS_GRAPH`, on by default): a decode step (`sp > 0`: t = 1, or a verify
//! window) is shape-stable. Everything that depends on the position reads it from a device
//! parameter (`P[0]` = the start position, copied in by the graph's first node) through
//! kernels_graph.cu; widths that grow with the position (the indexer's `(pos + 1) / r`,
//! candidate masks, top-k) are launched for their maximum (`max_seq_len / r`, `index_topk`)
//! and bounded on the device, and short top-k rows are padded with -1 holes, which the
//! attention skips exactly. So the step is captured once per t (after a warm-up run that
//! uploads the weights and sizes the arena) and replayed per token. Not captured: prefill,
//! DSpark's `forward_spec`, and steps with the expert LRU (`DS_EXPERT_SLOTS`), a host-side
//! expert-parallel `moe_reduce` (one that syncs: `reduce_graphable` unset), a hook,
//! `force` or `DS_TAP`; they run the same code directly. An NCCL `moe_reduce` only enqueues
//! (`reduce_graphable`), so it is captured as a graph node with the step.

use super::forward::{argmax, Model, SpecPick, SpecStep};
use super::gpu::{blocks, DevW, DsW, Kernels};
use super::graph::{Arena, Buf, GKernels, Staging, Upload, SEL};
use super::quant::{e4m3_to_f32, e8m0_to_f32};
use anyhow::{bail, Context, Result};
use std::cell::RefCell;
use std::collections::HashMap;
use std::mem::ManuallyDrop;
use std::path::Path;
use std::rc::Rc;
use tang_moe::args;
use tang_moe::gpu::{DevBuf, Gpu, Graph, Stream};

const B256: (u32, u32, u32) = (256, 1, 1);

/// Stands in for the start position in a decode step's host code, which must not depend on
/// it (the kernels read it from `P[0]`): any host use gives wrong answers the tests catch.
const DYN: usize = 1 << 40;

fn b(d: &DevBuf) -> Buf {
    Buf::from(d)
}

#[derive(Default)]
struct ExpertCache {
    map: HashMap<String, (Rc<DevW>, u64)>,
    tick: u64,
    misses: usize,
}

/// A layer's routed experts for the grouped MoE kernels: device arrays of [`DsW`] (w1, w3,
/// w2) and the ownership mask, indexed by expert; non-local experts are zeroed and masked off.
struct MoeTab {
    w1: DevBuf,
    w3: DevBuf,
    w2: DevBuf,
    mask: DevBuf,
    fmt: i32,
    inter: usize,
    local: usize,
    _keep: Vec<Rc<DevW>>,
}

/// Which attention a block runs.
enum Attn {
    Main,
    Spark(Buf, usize),
}

/// mHC coefficients on the device: pre [T][hc], post [T][hc], comb [T][hc][hc].
struct Mix {
    pre: Buf,
    post: Buf,
    comb: Buf,
}

/// Where RoPE reads a row's position: from the host, or from `P[0] + add` on the device.
#[derive(Clone, Copy)]
enum At {
    Host {
        pos0: usize,
        stride: usize,
        fixed: Option<usize>,
    },
    /// Row r at `P[0] + add + r * stride`, or (`fixed_r` > 0) a compressed group's
    /// `P[0] + add + 1 - fixed_r`.
    Dev {
        add: usize,
        stride: usize,
        fixed_r: usize,
    },
}

/// A step's host inputs, staged.
struct Inputs {
    ids: Buf,
    one_hot: Buf,
    engram: Vec<Buf>,
    copies: Vec<Upload>,
}

/// What a step leaves for the host: last (or all) rows' logits, the DSpark target hiddens,
/// and the ratio-r compressors' partial-group rows (layer, row in the step, kv, score).
#[derive(Clone)]
struct StepOut {
    logits: Buf,
    rows: usize,
    mains: Vec<Buf>,
    jrows: Vec<(usize, usize, Buf, Buf)>,
}

struct Captured {
    g: Graph,
    out: StepOut,
    gen: (u64, u64),
}

/// Graph counters (tests and the benchmark).
#[derive(Default, Clone, Copy, Debug)]
pub struct GraphStats {
    pub direct: usize,
    pub captures: usize,
    pub replays: usize,
}

pub struct GpuModel {
    /// The CPU reference: config, the mapped checkpoint, Engram hashing, host-side selection.
    pub cpu: Model,
    w: RefCell<HashMap<String, Rc<DevW>>>,
    experts: RefCell<ExpertCache>,
    /// Routed-expert tensors kept on the device (`DS_EXPERT_SLOTS`; 0 = all).
    pub expert_slots: usize,
    /// The fast GEMV path (`ds_lin_*`: fused activation quant and SwiGLU), bit-identical to
    /// the naive `ds_linear` path kept for A/B (`DS_SLOW=1` turns it off).
    pub fast: bool,
    /// Capture decode steps as CUDA graphs (`DS_GRAPH=0` turns it off; see the module doc).
    pub graph: bool,
    pub graph_stats: GraphStats,
    tabs: Vec<DevBuf>,
    win: Vec<DevBuf>,
    ckv: HashMap<usize, DevBuf>,
    idxk: HashMap<usize, DevBuf>,
    cst: HashMap<usize, (DevBuf, DevBuf)>,
    qk: RefCell<HashMap<usize, Rc<DevBuf>>>,
    sh_ckv: usize,
    sh_idxk: usize,
    /// The last index source's selection: [t][k] positions (offset into the attention's
    /// keys), and k.
    topk: Option<(Buf, usize)>,
    /// The candidate mask: u8 [t][stride], and the stride.
    cand: Option<(Buf, usize)>,
    /// This step's device parameters: `P[0]` = its start position.
    p: Buf,
    /// This step's compressor rows (see [`StepOut`]).
    jrows: Vec<(usize, usize, Buf, Buf)>,
    arena: RefCell<Arena>,
    stage: RefCell<Staging>,
    graphs: HashMap<(usize, bool), Captured>,
    warm: HashMap<(usize, bool), (u64, u64)>,
    /// Debugging: each backbone layer's input stream and pre-mix (as [`Model::force`]).
    pub force: Option<Vec<(Vec<f32>, Vec<f32>)>>,
    /// Intermediates read back by name when `DS_TAP` is set (as [`Model::taps`]).
    pub taps: RefCell<Vec<(String, Vec<f32>)>>,
    /// While a [`GpuSnapshot`] is open: whether it is, the steps run under it, and the
    /// compressors' partial-group writes (as [`Model::snapshot`]): lid, position, kv row,
    /// score row (arena views of the one step run under it).
    journal_on: bool,
    journal_steps: usize,
    journal: Vec<(usize, usize, Buf, Buf)>,
    snap: Option<(Vec<DevBuf>, HashMap<usize, (DevBuf, DevBuf)>)>,
    /// Expert parallelism ([`super::ep`]): the routed experts this rank runs. Others are
    /// skipped in `moe` and never uploaded. All of them when unsharded.
    pub owns_expert: Box<dyn Fn(usize) -> bool>,
    /// Expert parallelism: combines the routed partial sum `y` ([t][d] f32, the buffer and its
    /// element count) across ranks in place, before the shared expert is added. A host
    /// callback enqueueing on the stream: a step that has one is not captured (to capture
    /// it, the callback's NCCL all-reduce would have to be captured with the step).
    pub moe_reduce: Option<MoeReduce>,
    /// Grouped MoE (a fixed handful of launches per layer, whatever the expert count) for
    /// steps of <= 8 tokens with every expert resident; `DS_MOE_LOOP=1` keeps the per-expert
    /// loop. Bit-identical either way.
    pub moe_grouped: bool,
    moe_tabs: RefCell<HashMap<usize, Option<Rc<MoeTab>>>>,
    /// `moe_reduce` only enqueues on the stream (no host sync, no allocation), so a step
    /// that has one can be captured with it (NCCL's all-reduce becomes a graph node). Every
    /// rank then captures and replays its collectives in the same order.
    pub reduce_graphable: bool,
    /// Skip the logits/hiddens read-back (and its sync): `forward` returns empty vectors.
    /// For expert-parallel ranks whose output nobody reads.
    pub skip_readback: bool,
    /// The step timeline ([`super::obs`]; `DS_OBS`).
    pub obs: super::obs::Obs,
    // last: dropped after every buffer and graph above (see `Drop`)
    k: Kernels,
    gk: GKernels,
    s: Stream,
    _g: Gpu,
}

impl Drop for GpuModel {
    /// Drain the stream and destroy the graphs before any memory they use is freed.
    fn drop(&mut self) {
        let _ = self.s.sync();
        self.graphs.clear();
    }
}

/// As [`super::forward::Snapshot`], on the device: views of the model's one snapshot store
/// (a newer snapshot overwrites it).
pub struct GpuSnapshot {
    win: Vec<Buf>,
    cst: HashMap<usize, (Buf, Buf)>,
}

/// See [`GpuModel::moe_reduce`].
pub type MoeReduce = Box<dyn Fn(&DevBuf, usize, &Stream) -> Result<()>>;

impl GpuModel {
    pub fn open(dir: &Path) -> Result<Self> {
        Self::open_on(dir, 0)
    }

    /// [`Self::open`] on device `ordinal`, binding its context to the calling thread (the
    /// model is not `Send`: it lives on that thread).
    pub fn open_on(dir: &Path, ordinal: usize) -> Result<Self> {
        let g = Gpu::at(ordinal)?;
        g.bind()?;
        let k = Kernels::new(&g)?;
        let gk = GKernels::new(&g)?;
        // blocking: weight uploads (pageable `cuMemcpyHtoD`) finish their DMA on the legacy
        // stream, which only a blocking stream orders behind
        let s = Stream::blocking()?;
        let cpu = Model::open(dir)?;
        let c = cpu.cfg.clone();
        let (hd, sl) = (c.head_dim, c.max_seq_len);
        let upf = |v: &[f32]| -> Result<DevBuf> { Ok(DevBuf::from_slice(v)?) };
        let tabs = cpu
            .freqs
            .iter()
            .map(|t| upf(&t.iter().flat_map(|&(a, b)| [a, b]).collect::<Vec<f32>>()))
            .collect::<Result<_>>()?;
        let n_all = c.n_layers + c.n_mtp_layers;
        let win = (0..n_all)
            .map(|_| upf(&vec![0f32; c.window_size * hd]))
            .collect::<Result<_>>()?;
        let mut ckv = HashMap::new();
        let mut idxk = HashMap::new();
        let mut cst = HashMap::new();
        for &l in &c.kv_source_layers {
            let r = c.compress_ratios[l];
            ckv.insert(l, upf(&vec![0f32; sl / r * hd])?);
            idxk.insert(l, upf(&vec![0f32; sl / r * c.index_head_dim])?);
            if r > 1 {
                cst.insert(
                    l,
                    (
                        upf(&vec![0f32; r * hd])?,
                        upf(&vec![f32::NEG_INFINITY; r * hd])?,
                    ),
                );
            }
        }
        Ok(Self {
            cpu,
            k,
            gk,
            s,
            _g: g,
            w: RefCell::new(HashMap::new()),
            experts: RefCell::new(ExpertCache::default()),
            // an expert MLP holds three tensors at once
            expert_slots: match std::env::var("DS_EXPERT_SLOTS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
            {
                Some(0) | None => 0,
                Some(n) => n.max(3),
            },
            fast: std::env::var_os("DS_SLOW").is_none(),
            graph: std::env::var("DS_GRAPH").map_or(true, |v| v != "0"),
            graph_stats: GraphStats::default(),
            tabs,
            win,
            ckv,
            idxk,
            cst,
            qk: RefCell::new(HashMap::new()),
            sh_ckv: 0,
            sh_idxk: 0,
            topk: None,
            cand: None,
            p: Buf { ptr: 0, len: 0 },
            jrows: Vec::new(),
            arena: RefCell::new(Arena::default()),
            stage: RefCell::new(Staging::default()),
            graphs: HashMap::new(),
            warm: HashMap::new(),
            force: None,
            taps: RefCell::new(Vec::new()),
            journal_on: false,
            journal_steps: 0,
            journal: Vec::new(),
            snap: None,
            owns_expert: Box::new(|_| true),
            moe_reduce: None,
            moe_grouped: std::env::var("DS_MOE_LOOP").map_or(true, |v| v != "1"),
            moe_tabs: RefCell::new(HashMap::new()),
            reduce_graphable: false,
            skip_readback: false,
            obs: Default::default(),
        })
    }

    /// Forget the sequence: the next forward must start at position 0. The caches need no
    /// clearing (nothing past a sequence's own positions is ever read); the ratio-2
    /// compressors' partial groups do (in place: captured graphs hold their addresses).
    pub fn reset(&mut self) -> Result<()> {
        let hd = self.cpu.cfg.head_dim;
        for (&l, (a, bb)) in self.cst.iter() {
            let n = self.cpu.cfg.compress_ratios[l] * hd;
            self.fill(a.ptr, 0, n)?;
            self.fill(bb.ptr, f32::NEG_INFINITY.to_bits() as i32, n)?;
        }
        self.sh_ckv = 0;
        self.sh_idxk = 0;
        self.topk = None;
        self.cand = None;
        Ok(())
    }

    // scratch

    /// `n` uninitialized f32 (or i32) of this step's scratch.
    fn fbuf(&self, n: usize) -> Result<Buf> {
        self.arena.borrow_mut().alloc(n.max(1) * 4)
    }

    fn bytes(&self, n: usize) -> Result<Buf> {
        self.arena.borrow_mut().alloc(n.max(1))
    }

    /// `n` zeroed f32 of scratch (a kernel: `DevBuf::zeroed`'s memset is on the NULL stream).
    fn zbuf(&self, n: usize) -> Result<Buf> {
        let o = self.fbuf(n)?;
        self.fill(o.ptr, 0, n)?;
        Ok(o)
    }

    fn fill(&self, p: u64, v: i32, n: usize) -> Result<()> {
        let nn = n as i64;
        unsafe {
            self.k.go(
                "ds_fill_i32",
                blocks(n, 256),
                B256,
                0,
                &self.s,
                args![p, v, nn],
            )
        }
    }

    /// Stage `v` for upload: scratch on the device, filled by the returned copy.
    fn stage<T: Copy>(&self, v: &[T]) -> Result<(Buf, Upload)> {
        let n = std::mem::size_of_val(v);
        let d = self.arena.borrow_mut().alloc(n)?;
        let h = self.stage.borrow_mut().alloc(n)?;
        unsafe { std::ptr::copy_nonoverlapping(v.as_ptr() as *const u8, h, n) };
        Ok((
            d,
            Upload {
                dst: d.ptr,
                src: h,
                n,
            },
        ))
    }

    /// `v` on the device now (a direct step's upload).
    fn up<T: Copy>(&self, v: &[T]) -> Result<Buf> {
        let (d, u) = self.stage(v)?;
        u.enqueue(&self.s)?;
        Ok(d)
    }

    /// Start a step: drain the stream (its staging may still be read), then release the
    /// last step's scratch and staging.
    fn begin_step(&mut self) -> Result<()> {
        self.sync()?;
        self.arena.borrow_mut().reset()?;
        self.stage.borrow_mut().reset()?;
        let g = self.gen();
        self.graphs.retain(|_, c| c.gen == g);
        Ok(())
    }

    /// Drain the stream (counted: [`super::obs`]).
    fn sync(&self) -> Result<()> {
        super::obs::sync();
        Ok(self.s.sync()?)
    }

    /// Open a timeline span ([`super::obs::Obs::span`]).
    #[inline(always)]
    fn span(&self, name: &'static str) -> u32 {
        self.obs.span(&self.s, name, None)
    }

    #[inline(always)]
    fn end(&self, id: u32) {
        self.obs.end(&self.s, id)
    }

    fn gen(&self) -> (u64, u64) {
        (self.arena.borrow().gen, self.stage.borrow().gen)
    }

    // weights

    fn dw(&self, name: &str) -> Result<Rc<DevW>> {
        if name.contains(".ffn.experts.") {
            return self.expert_w(name);
        }
        if let Some(w) = self.w.borrow().get(name) {
            return Ok(w.clone());
        }
        let w = Rc::new(DevW::upload(&self.cpu.st, name)?);
        self.w.borrow_mut().insert(name.to_string(), w.clone());
        Ok(w)
    }

    /// A routed expert's weight through the LRU of `expert_slots` tensors (unbounded when 0),
    /// uploaded from the mapped checkpoint on a miss. Evicting frees device memory a queued
    /// kernel may still read, so it drains the stream first (the LRU's only sync besides
    /// its counts read-back).
    fn expert_w(&self, name: &str) -> Result<Rc<DevW>> {
        let mut ex = self.experts.borrow_mut();
        ex.tick += 1;
        let tick = ex.tick;
        if let Some((w, t)) = ex.map.get_mut(name) {
            *t = tick;
            return Ok(w.clone());
        }
        if self.expert_slots > 0 && ex.map.len() >= self.expert_slots {
            let oldest = ex
                .map
                .iter()
                .min_by_key(|(_, (_, t))| *t)
                .map(|(k, _)| k.clone())
                .unwrap();
            self.sync()?;
            ex.map.remove(&oldest);
        }
        ex.misses += 1;
        super::obs::expert_upload();
        let w = Rc::new(DevW::upload(&self.cpu.st, name)?);
        ex.map.insert(name.to_string(), (w.clone(), tick));
        Ok(w)
    }

    /// Routed-expert tensors uploaded so far (cache misses).
    pub fn expert_misses(&self) -> usize {
        self.experts.borrow().misses
    }

    fn tap(&self, name: &str, b: Buf, n: usize) -> Result<()> {
        if std::env::var_os("DS_TAP").is_some() {
            let v = self.read(b, n)?;
            self.taps.borrow_mut().push((name.to_string(), v));
        }
        Ok(())
    }

    fn read(&self, b: Buf, n: usize) -> Result<Vec<f32>> {
        self.sync()?;
        super::obs::d2h(n * 4);
        Ok(b.dev().read::<f32>(0, n)?)
    }

    fn copy(&self, dst: u64, src: u64, n: usize) -> Result<()> {
        let nn = n as i64;
        unsafe {
            self.k.go(
                "ds_copy",
                blocks(n, 256),
                B256,
                0,
                &self.s,
                args![dst, src, nn],
            )
        }
    }

    /// `linear`: quantized weights get an fp8-quantized copy of the activation; bf16 out.
    fn lin(&self, x: Buf, t: usize, name: &str) -> Result<Buf> {
        let w = self.dw(name)?;
        let (k, n) = (w.v.cols as usize, w.v.rows as usize);
        let y = self.fbuf(t * n)?;
        if self.fast && Kernels::fast_ok(&w.v) {
            let xq = self.qdq(x, t * k)?;
            self.k
                .linear_fast(&self.s, &w.v, None, xq.ptr, y.ptr, t, true, false, 0, 0.0)?;
        } else if w.v.fmt >= 2 {
            let xq = self.qdq(x, t * k)?;
            self.k.linear(&self.s, &w.v, xq.ptr, y.ptr, t, true)?;
        } else {
            self.k.linear(&self.s, &w.v, x.ptr, y.ptr, t, true)?;
        }
        Ok(y)
    }

    /// An f32 GEMM with no activation quantization or rounding (router, mHC, compressor > 1,
    /// the head).
    fn raw(&self, x: u64, t: usize, name: &str) -> Result<Buf> {
        let w = self.dw(name)?;
        let y = self.fbuf(t * w.v.rows as usize)?;
        self.k.linear(&self.s, &w.v, x, y.ptr, t, false)?;
        Ok(y)
    }

    fn rms(&self, x: Buf, rows: usize, name: &str) -> Result<()> {
        let w = self.dw(name)?;
        self.k
            .rms(&self.s, x.ptr, &w.v, rows, self.cpu.cfg.norm_eps)
    }

    #[allow(clippy::too_many_arguments)]
    fn rope(
        &self,
        x: Buf,
        rows: usize,
        per_row: usize,
        d: usize,
        lid: usize,
        at: At,
        inverse: bool,
    ) -> Result<()> {
        let rd = self.cpu.cfg.rope_head_dim;
        let tab = self.tabs[lid].ptr;
        match at {
            At::Host {
                pos0,
                stride,
                fixed,
            } => self.k.rope(
                &self.s, x.ptr, tab, rows, per_row, d, rd, pos0, stride, fixed, inverse,
            ),
            At::Dev {
                add,
                stride,
                fixed_r,
            } => {
                let n = rows * per_row * rd / 2;
                let (r, pr, di, rdi, p, a, ps, fx, inv) = (
                    rows as i32,
                    per_row as i32,
                    d as i32,
                    rd as i32,
                    self.p.ptr,
                    add as i32,
                    stride as i32,
                    fixed_r as i32,
                    inverse as i32,
                );
                unsafe {
                    self.gk.go(
                        "dg_rope",
                        blocks(n, 256),
                        B256,
                        0,
                        &self.s,
                        args![x.ptr, tab, r, pr, di, rdi, p, a, ps, fx, inv],
                    )
                }
            }
        }
    }

    // attention

    fn attention(&mut self, p: &str, lid: usize, x: Buf, t: usize, sp: usize) -> Result<Buf> {
        if sp == 0 {
            self.attention_prefill(p, lid, x, t)
        } else {
            self.attention_dyn(p, lid, x, t)
        }
    }

    fn attention_prefill(&mut self, p: &str, lid: usize, x: Buf, t: usize) -> Result<Buf> {
        let c = self.cpu.cfg.clone();
        let (h, hd) = (c.n_heads, c.head_dim);
        let at0 = At::Host {
            pos0: 0,
            stride: 1,
            fixed: None,
        };
        let s = self.span("q");
        let qr = self.lin(x, t, &format!("{p}.attn.wq_a.weight"))?;
        self.rms(qr, t, &format!("{p}.attn.q_norm.weight"))?;
        self.tap(&format!("{p}.qr"), qr, t * c.q_lora_rank)?;
        let q = self.lin(qr, t, &format!("{p}.attn.wq_b.weight"))?;
        self.tap(&format!("{p}.q0"), q, t * h * hd)?;
        self.rope(q, t, h, hd, lid, at0, false)?;
        self.tap(&format!("{p}.q"), q, t * h * hd)?;
        self.end(s);
        let s = self.span("window_kv");
        let kv = self.lin(x, t, &format!("{p}.attn.wkv.weight"))?;
        self.rms(kv, t, &format!("{p}.attn.kv_norm.weight"))?;
        self.rope(kv, t, 1, hd, lid, at0, false)?;
        self.k.fp8_qdq(&self.s, kv.ptr, t * hd)?;
        self.ring_seed(lid, kv, t)?;
        self.tap(&format!("{p}.wkv"), kv, t * hd)?;
        self.end(s);
        let nw = t.min(c.window_size);
        let (mut kc, mut kv1) = (0, 0);
        if c.compress_ratios[lid] > 0 {
            let s = self.span("compress");
            let (src, k) = self.compress_kv_prefill(p, lid, x, qr, t)?;
            kc = k;
            kv1 = self.ckv[&src].ptr;
            self.end(s);
        }
        let s = self.span("sparse_attn");
        let idx = self.win_idx(t, nw + kc, nw, 0, 0, 0)?;
        if kc > 0 {
            let tk = self.topk.as_ref().context("no top-k")?.0;
            self.idx_cat(idx, tk, t, kc, nw + kc, nw)?;
        }
        self.end(s);
        self.attn_out(p, q, (kv, t, kv1), idx, t, nw + kc, lid, at0)
    }

    /// The decode step's attention (t = 1, or a verify window), position-free on the host.
    fn attention_dyn(&mut self, p: &str, lid: usize, x: Buf, t: usize) -> Result<Buf> {
        let c = self.cpu.cfg.clone();
        let (h, hd, win) = (c.n_heads, c.head_dim, c.window_size);
        let at = At::Dev {
            add: 0,
            stride: 1,
            fixed_r: 0,
        };
        let s = self.span("q");
        let qr = self.lin(x, t, &format!("{p}.attn.wq_a.weight"))?;
        self.rms(qr, t, &format!("{p}.attn.q_norm.weight"))?;
        let q = self.lin(qr, t, &format!("{p}.attn.wq_b.weight"))?;
        self.rope(q, t, h, hd, lid, at, false)?;
        self.end(s);
        let s = self.span("window_kv");
        let kv = self.lin(x, t, &format!("{p}.attn.wkv.weight"))?;
        self.rms(kv, t, &format!("{p}.attn.kv_norm.weight"))?;
        self.rope(kv, t, 1, hd, lid, at, false)?;
        self.k.fp8_qdq(&self.s, kv.ptr, t * hd)?;
        let ring = b(&self.win[lid]);
        let (kv0, split, mode) = if t > 1 {
            // a verify window (as forward.rs): the pre-chunk ring then the chunk's rows
            let all = self.fbuf((win + t) * hd)?;
            self.copy(all.ptr, ring.ptr, win * hd)?;
            self.copy(all.ptr + (win * hd * 4) as u64, kv.ptr, t * hd)?;
            self.ring_write(ring, kv, t, hd, win, 0)?;
            (all, win + t, 3)
        } else {
            self.ring_write(ring, kv, 1, hd, win, 0)?;
            (ring, win, 1)
        };
        self.end(s);
        let (mut kc, mut kv1) = (0, 0);
        if c.compress_ratios[lid] > 0 {
            let s = self.span("compress");
            let src = self.compress_dyn(p, lid, x, qr, t, split)?;
            kc = self.topk.as_ref().context("no top-k")?.1;
            kv1 = self.ckv[&src].ptr;
            self.end(s);
        }
        let s = self.span("sparse_attn");
        let idx = self.fbuf(t * (win + kc))?;
        let (ti, st, ni, mi, pp, wi) = (
            t as i32,
            (win + kc) as i32,
            win as i32,
            mode as i32,
            self.p.ptr,
            win as i32,
        );
        unsafe {
            self.gk.go(
                "dg_win_idx",
                blocks(t * win, 256),
                B256,
                0,
                &self.s,
                args![idx.ptr, ti, st, ni, mi, pp, wi],
            )?
        };
        if kc > 0 {
            let tk = self.topk.as_ref().unwrap().0;
            self.idx_cat(idx, tk, t, kc, win + kc, win)?;
        }
        self.end(s);
        self.attn_out(p, q, (kv0, split, kv1), idx, t, win + kc, lid, at)
    }

    fn idx_cat(&self, idx: Buf, src: Buf, t: usize, k: usize, st: usize, col: usize) -> Result<()> {
        let (ti, ki, sti, ci) = (t as i32, k as i32, st as i32, col as i32);
        unsafe {
            self.k.go(
                "ds_idx_cat",
                blocks(t * k, 256),
                B256,
                0,
                &self.s,
                args![idx.ptr, src.ptr, ti, ki, sti, ci],
            )
        }
    }

    /// `t` rows of `hd` into a ring of `win` slots at positions `P[0] + add + e`.
    fn ring_write(
        &self,
        ring: Buf,
        src: Buf,
        t: usize,
        hd: usize,
        win: usize,
        add: usize,
    ) -> Result<()> {
        let (ti, hdi, wi, pp, ai) = (t as i32, hd as i32, win as i32, self.p.ptr, add as i32);
        unsafe {
            self.gk.go(
                "dg_ring_write",
                blocks(t * hd, 256),
                B256,
                0,
                &self.s,
                args![ring.ptr, src.ptr, ti, hdi, wi, pp, ai],
            )
        }
    }

    /// One row of `n` into row `(P[0] + add) / r` of `dst`.
    fn row_write(&self, dst: u64, src: Buf, n: usize, add: usize, r: usize) -> Result<()> {
        let (ni, pp, ai, ri) = (n as i32, self.p.ptr, add as i32, r as i32);
        unsafe {
            self.gk.go(
                "dg_row_write",
                blocks(n, 256),
                B256,
                0,
                &self.s,
                args![dst, src.ptr, ni, pp, ai, ri],
            )
        }
    }

    /// Attention index rows [t][stride] with the window's `n` columns filled (`ds_win_idx`).
    fn win_idx(
        &self,
        t: usize,
        stride: usize,
        n: usize,
        mode: i32,
        sp: usize,
        bb: usize,
    ) -> Result<Buf> {
        let idx = self.fbuf(t.max(1) * stride.max(1))?;
        let (ti, si, ni, spi, wi, bi) = (
            t as i32,
            stride as i32,
            n as i32,
            sp as i32,
            self.cpu.cfg.window_size as i32,
            bb as i32,
        );
        unsafe {
            self.k.go(
                "ds_win_idx",
                blocks(t * n, 256),
                B256,
                0,
                &self.s,
                args![idx.ptr, ti, si, ni, mode, spi, wi, bi],
            )?
        };
        Ok(idx)
    }

    /// Sparse attention over keys `kv` = (rows below `split` here, the rest from `kv1`), then
    /// the output projection.
    #[allow(clippy::too_many_arguments)]
    fn attn_out(
        &self,
        p: &str,
        q: Buf,
        kv: (Buf, usize, u64),
        di: Buf,
        t: usize,
        k: usize,
        lid: usize,
        at: At,
    ) -> Result<Buf> {
        let c = &self.cpu.cfg;
        let (h, hd, g, ol) = (c.n_heads, c.head_dim, c.o_groups, c.o_lora_rank);
        let s = self.span("sparse_attn");
        let sink = self.dw(&format!("{p}.attn.attn_sink"))?;
        let o = self.fbuf(t * h * hd)?;
        let (kv0, split, kv1) = (kv.0.ptr, kv.1 as i32, kv.2);
        let (hi, dd, ki, scale) = (h as i32, hd as i32, k as i32, (hd as f32).powf(-0.5));
        let threads = hd.clamp(32, 1024) as u32;
        unsafe {
            self.gk.go(
                "dg_sparse_attn",
                ((t * h) as u32, 1, 1),
                (threads, 1, 1),
                (k * 4) as u32,
                &self.s,
                args![q.ptr, kv0, split, kv1, sink.v.w, di.ptr, o.ptr, hi, dd, ki, scale],
            )?
        };
        self.tap(&format!("{p}.sa"), o, t * h * hd)?;
        self.end(s);
        let s = self.span("wo");
        self.rope(o, t, h, hd, lid, at, true)?;
        self.k.fp8_qdq(&self.s, o.ptr, t * h * hd)?;
        let wo_a = self.dw(&format!("{p}.attn.wo_a.weight"))?;
        let mid = self.fbuf(t * g * ol)?;
        let (wv, ti, gi, oli) = (wo_a.v, t as i32, g as i32, ol as i32);
        unsafe {
            self.k.go(
                "ds_wo_a",
                blocks(t * g * ol * 32, 256),
                B256,
                0,
                &self.s,
                args![o.ptr, wv, mid.ptr, ti, gi, oli],
            )?
        };
        let y = self.lin(mid, t, &format!("{p}.attn.wo_b.weight"));
        self.end(s);
        y
    }

    fn ring_seed(&self, lid: usize, kv: Buf, t: usize) -> Result<()> {
        let (hd, win) = (self.cpu.cfg.head_dim, self.cpu.cfg.window_size);
        let ring = self.win[lid].ptr;
        if t <= win {
            return self.copy(ring, kv.ptr, t * hd);
        }
        let cut = t % win;
        let tail = kv.ptr + ((t - win) * hd * 4) as u64;
        self.copy(ring + (cut * hd * 4) as u64, tail, (win - cut) * hd)?;
        self.copy(ring, tail + ((win - cut) * hd * 4) as u64, cut * hd)
    }

    fn pool(&self, kvp: u64, scp: u64, groups: usize, r: usize) -> Result<Buf> {
        let hd = self.cpu.cfg.head_dim;
        let out = self.fbuf(groups * hd)?;
        let (gi, ri, hdi) = (groups as i32, r as i32, hd as i32);
        unsafe {
            self.k.go(
                "ds_pool",
                blocks(groups * hd, 256),
                B256,
                0,
                &self.s,
                args![kvp, scp, out.ptr, gi, ri, hdi],
            )?
        };
        let n = (groups * hd) as i64;
        unsafe {
            self.k.go(
                "ds_round",
                blocks(groups * hd, 256),
                B256,
                0,
                &self.s,
                args![out.ptr, n],
            )?
        };
        Ok(out)
    }

    /// Prefill: the source's new compressed latents (pre-RoPE) and how many.
    fn compressor_prefill(
        &mut self,
        p: &str,
        lid: usize,
        x: Buf,
        t: usize,
    ) -> Result<Option<(Buf, usize)>> {
        let c = self.cpu.cfg.clone();
        let (r, hd) = (c.compress_ratios[lid], c.head_dim);
        let norm = format!("{p}.attn.compressor.norm.weight");
        if r == 1 {
            let kv = self.lin(x, t, &format!("{p}.attn.compressor.wkv.weight"))?;
            self.rms(kv, t, &norm)?;
            return Ok(Some((kv, t)));
        }
        let kv = self.raw(x.ptr, t, &format!("{p}.attn.compressor.wkv.weight"))?;
        let sc = self.raw(x.ptr, t, &format!("{p}.attn.compressor.wgate.weight"))?;
        let (kst, sst) = {
            let (a, bb) = &self.cst[&lid];
            (a.ptr, bb.ptr)
        };
        let row = (hd * 4) as u64;
        if t < r {
            self.copy(kst, kv.ptr, t * hd)?;
            self.copy(sst, sc.ptr, t * hd)?;
            return Ok(None);
        }
        let rem = t % r;
        let cut = t - rem;
        if rem > 0 {
            self.copy(kst, kv.ptr + cut as u64 * row, rem * hd)?;
            self.copy(sst, sc.ptr + cut as u64 * row, rem * hd)?;
        }
        let out = self.pool(kv.ptr, sc.ptr, cut / r, r)?;
        self.rms(out, cut / r, &norm)?;
        Ok(Some((out, cut / r)))
    }

    /// Prefill: compressed KV and selection; the cache's source layer and top-k width.
    fn compress_kv_prefill(
        &mut self,
        p: &str,
        lid: usize,
        x: Buf,
        qr: Buf,
        t: usize,
    ) -> Result<(usize, usize)> {
        let c = self.cpu.cfg.clone();
        let (r, hd) = (c.compress_ratios[lid], c.head_dim);
        let clen = t / r;
        let mut latent = None;
        if c.is_kv_source(lid) {
            latent = self.compressor_prefill(p, lid, x, t)?;
            self.sh_ckv = lid;
        }
        if c.is_index_source(lid) {
            self.topk = if clen == 0 {
                None
            } else {
                let s = self.span("indexer");
                let r = self.indexer_prefill(p, lid, x, qr, latent, t)?;
                self.end(s);
                Some(r)
            };
        }
        let kc = self.topk.as_ref().map_or(0, |(_, k)| *k);
        if let Some((lat, n)) = latent {
            let at = At::Host {
                pos0: 0,
                stride: r,
                fixed: None,
            };
            self.rope(lat, n, 1, hd, lid, at, false)?;
            self.k.fp4_qdq(&self.s, lat.ptr, n * hd, 16, true)?;
            self.copy(self.ckv[&lid].ptr, lat.ptr, n * hd)?;
        }
        Ok((self.sh_ckv, kc))
    }

    fn indexer_prefill(
        &mut self,
        p: &str,
        lid: usize,
        x: Buf,
        qr: Buf,
        latent: Option<(Buf, usize)>,
        t: usize,
    ) -> Result<(Buf, usize)> {
        let c = self.cpu.cfg.clone();
        let (r, nh, ihd) = (c.compress_ratios[lid], c.index_n_heads, c.index_head_dim);
        let owns = c.is_kv_source(lid);
        if let (true, Some((lat, n))) = (owns, latent) {
            let k = self.lin(lat, n, &format!("{p}.attn.indexer.wk.weight"))?;
            self.rms(k, n, &format!("{p}.attn.indexer.k_norm.weight"))?;
            let at = At::Host {
                pos0: 0,
                stride: r,
                fixed: None,
            };
            self.rope(k, n, 1, ihd, lid, at, false)?;
            self.k.fp4_qdq(&self.s, k.ptr, n * ihd, 32, false)?;
            self.copy(self.idxk[&lid].ptr, k.ptr, n * ihd)?;
        }
        if owns {
            self.sh_idxk = lid;
        }
        let q = self.lin(qr, t, &format!("{p}.attn.indexer.wq_b.weight"))?;
        let at0 = At::Host {
            pos0: 0,
            stride: 1,
            fixed: None,
        };
        self.rope(q, t, nh, ihd, lid, at0, false)?;
        self.k.fp4_qdq(&self.s, q.ptr, t * nh * ihd, 32, false)?;
        let width = t / r;
        let wts = self.index_weights(p, x, t)?;
        let score = self.fbuf(t * width)?;
        let (ti, wi, nhi, ihdi) = (t as i32, width as i32, nh as i32, ihd as i32);
        let ik = self.idxk[&self.sh_idxk].ptr;
        unsafe {
            self.k.go(
                "ds_index_score",
                blocks(t * width * 32, 256),
                B256,
                0,
                &self.s,
                args![q.ptr, ik, wts.ptr, score.ptr, ti, wi, nhi, ihdi],
            )?
        };
        // masks, candidate blocks, top-k (as forward.rs), all on the device
        self.k
            .mask_scores(&self.s, score.ptr, 0, t, width, true, r)?;
        let src = c.candidate_source_layer;
        if lid as i64 == src {
            let cand = self.bytes(t * width)?;
            let (bs, kb) = (c.candidate_block_size, c.candidate_topk_blocks);
            let nb = width.div_ceil(bs);
            let (sc, sel) = (self.fbuf(t * nb)?, self.fbuf(t * nb)?);
            let (ci, ri, bsi, kbi) = (1i32, r as i32, bs as i32, kb as i32);
            unsafe {
                self.k.go(
                    "ds_candidates",
                    (t.max(1) as u32, 1, 1),
                    SEL,
                    0,
                    &self.s,
                    args![score.ptr, sc.ptr, sel.ptr, cand.ptr, ti, wi, ci, ri, bsi, kbi],
                )?
            };
            self.cand = Some((cand, width));
        } else if src >= 0 && (src as usize) < lid {
            let (cand, cw) = self.cand.context("no candidate mask")?;
            anyhow::ensure!(cw == width, "candidate mask width {cw} != {width}");
            self.k
                .mask_scores(&self.s, score.ptr, cand.ptr, t, width, false, r)?;
        }
        let k = c.index_topk.min(width);
        let out = self.fbuf(t * k)?;
        let scratch = self.fbuf(t * k)?;
        let (ki, si, oi, ci, ri) = (k as i32, k as i32, t as i32, 1i32, r as i32);
        unsafe {
            self.k.go(
                "ds_topk",
                (t.max(1) as u32, 1, 1),
                SEL,
                0,
                &self.s,
                args![score.ptr, scratch.ptr, out.ptr, ti, wi, ki, si, oi, ci, ri],
            )?
        };
        if std::env::var_os("DS_TAP").is_some() {
            self.sync()?;
            super::obs::d2h(t * k * 4);
            let v = out.dev().read::<i32>(0, t * k)?;
            self.taps
                .borrow_mut()
                .push((format!("{p}.topk"), v.iter().map(|&i| i as f32).collect()));
        }
        Ok((out, k))
    }

    /// The indexer's per-head weights, scaled and rounded.
    fn index_weights(&self, p: &str, x: Buf, t: usize) -> Result<Buf> {
        let (nh, ihd) = (self.cpu.cfg.index_n_heads, self.cpu.cfg.index_head_dim);
        let wts = self.lin(x, t, &format!("{p}.attn.indexer.weights_proj.weight"))?;
        let ws = ((ihd as f64).powf(-0.5) * (nh as f64).powf(-0.5)) as f32;
        let nw = (t * nh) as i64;
        unsafe {
            self.k.go(
                "ds_scale_round",
                blocks(t * nh, 256),
                B256,
                0,
                &self.s,
                args![wts.ptr, ws, nw],
            )?
        };
        Ok(wts)
    }

    /// A decode step's compressed KV and selection (as forward.rs `compress_kv_steps`: one
    /// position at a time), position-free on the host: the ratio-r compressor pools its
    /// partial group every position and writes it to the group's cache row, which is past
    /// every reader's width until the group completes and rewrites it. An index source
    /// writes [t][index_topk] rows (-1 holes past each row's min(k, width)); the candidate
    /// source a [t][max_seq_len / r] mask. Returns the cache's source layer.
    fn compress_dyn(
        &mut self,
        p: &str,
        lid: usize,
        x: Buf,
        qr: Buf,
        t: usize,
        offset: usize,
    ) -> Result<usize> {
        let c = self.cpu.cfg.clone();
        let (r, hd, d, ql) = (c.compress_ratios[lid], c.head_dim, c.dim, c.q_lora_rank);
        let (is_kv, is_idx) = (c.is_kv_source(lid), c.is_index_source(lid));
        let (wmax, kmax) = (c.max_seq_len / r, c.index_topk);
        let src_l = c.candidate_source_layer;
        let makes_cand = lid as i64 == src_l;
        let eats_cand = is_idx && src_l >= 0 && (src_l as usize) < lid;
        let tk = if is_idx {
            Some(self.fbuf(t * kmax)?)
        } else {
            None
        };
        let cand_out = if makes_cand {
            Some(self.bytes(t * wmax)?)
        } else {
            None
        };
        let cand_in = if eats_cand {
            Some(self.cand.context("no candidate mask")?)
        } else {
            None
        };
        for e in 0..t {
            let (xe, qe) = (x.at(e * d * 4), qr.at(e * ql * 4));
            let latent = if is_kv {
                self.sh_ckv = lid;
                Some(self.compressor_dyn(p, lid, xe, e)?)
            } else {
                None
            };
            if let Some(tk) = tk {
                let s = self.span("indexer");
                self.indexer_dyn(
                    p,
                    lid,
                    xe,
                    qe,
                    latent,
                    e,
                    offset,
                    tk.at(e * kmax * 4),
                    cand_out.map(|cb| cb.at(e * wmax)),
                    cand_in.map(|(cb, st)| cb.at(e * st)),
                )?;
                self.end(s);
            }
            if let Some(lat) = latent {
                let at = At::Dev {
                    add: e,
                    stride: 0,
                    fixed_r: r,
                };
                self.rope(lat, 1, 1, hd, lid, at, false)?;
                self.k.fp4_qdq(&self.s, lat.ptr, hd, 16, true)?;
                self.row_write(self.ckv[&lid].ptr, lat, hd, e, r)?;
            }
        }
        if let Some(tk) = tk {
            self.topk = Some((tk, kmax));
        }
        if let Some(cb) = cand_out {
            self.cand = Some((cb, wmax));
        }
        Ok(self.sh_ckv)
    }

    /// One position's compressed latent (pre-RoPE): see [`Self::compress_dyn`].
    fn compressor_dyn(&mut self, p: &str, lid: usize, xe: Buf, e: usize) -> Result<Buf> {
        let c = &self.cpu.cfg;
        let (r, hd) = (c.compress_ratios[lid], c.head_dim);
        let norm = format!("{p}.attn.compressor.norm.weight");
        if r == 1 {
            let kv = self.lin(xe, 1, &format!("{p}.attn.compressor.wkv.weight"))?;
            self.rms(kv, 1, &norm)?;
            return Ok(kv);
        }
        let kv = self.raw(xe.ptr, 1, &format!("{p}.attn.compressor.wkv.weight"))?;
        let sc = self.raw(xe.ptr, 1, &format!("{p}.attn.compressor.wgate.weight"))?;
        let (kst, sst) = {
            let (a, bb) = &self.cst[&lid];
            (b(a), b(bb))
        };
        self.ring_write(kst, kv, 1, hd, r, e)?;
        self.ring_write(sst, sc, 1, hd, r, e)?;
        self.jrows.push((lid, e, kv, sc));
        let out = self.pool(kst.ptr, sst.ptr, 1, r)?;
        self.rms(out, 1, &norm)?;
        Ok(out)
    }

    /// One position's indexer (see [`Self::compress_dyn`]).
    #[allow(clippy::too_many_arguments)]
    fn indexer_dyn(
        &mut self,
        p: &str,
        lid: usize,
        xe: Buf,
        qe: Buf,
        latent: Option<Buf>,
        e: usize,
        offset: usize,
        out: Buf,
        cand_out: Option<Buf>,
        cand_in: Option<Buf>,
    ) -> Result<()> {
        let c = self.cpu.cfg.clone();
        let (r, nh, ihd) = (c.compress_ratios[lid], c.index_n_heads, c.index_head_dim);
        let (wmax, kmax) = (c.max_seq_len / r, c.index_topk);
        let owns = c.is_kv_source(lid);
        if let (true, Some(lat)) = (owns, latent) {
            let k = self.lin(lat, 1, &format!("{p}.attn.indexer.wk.weight"))?;
            self.rms(k, 1, &format!("{p}.attn.indexer.k_norm.weight"))?;
            let at = At::Dev {
                add: e,
                stride: 0,
                fixed_r: r,
            };
            self.rope(k, 1, 1, ihd, lid, at, false)?;
            self.k.fp4_qdq(&self.s, k.ptr, ihd, 32, false)?;
            self.row_write(self.idxk[&lid].ptr, k, ihd, e, r)?;
        }
        if owns {
            self.sh_idxk = lid;
        }
        let q = self.lin(qe, 1, &format!("{p}.attn.indexer.wq_b.weight"))?;
        let at = At::Dev {
            add: e,
            stride: 1,
            fixed_r: 0,
        };
        self.rope(q, 1, nh, ihd, lid, at, false)?;
        self.k.fp4_qdq(&self.s, q.ptr, nh * ihd, 32, false)?;
        let wts = self.index_weights(p, xe, 1)?;
        let score = self.fbuf(wmax)?;
        let ik = self.idxk[&self.sh_idxk].ptr;
        let (pp, ai, ri, nhi, ihdi) = (self.p.ptr, e as i32, r as i32, nh as i32, ihd as i32);
        unsafe {
            self.gk.go(
                "dg_index_score",
                blocks(wmax * 32, 256),
                B256,
                0,
                &self.s,
                args![q.ptr, ik, wts.ptr, score.ptr, pp, ai, ri, nhi, ihdi],
            )?
        };
        if let Some(cb) = cand_out {
            let (bs, kb) = (c.candidate_block_size, c.candidate_topk_blocks);
            let nb = wmax.div_ceil(bs);
            let (sc, sel) = (self.fbuf(nb)?, self.fbuf(nb)?);
            let (bsi, kbi) = (bs as i32, kb as i32);
            unsafe {
                self.gk.go(
                    "dg_candidates",
                    (1, 1, 1),
                    SEL,
                    0,
                    &self.s,
                    args![score.ptr, sc.ptr, sel.ptr, cb.ptr, pp, ai, ri, bsi, kbi],
                )?
            };
        } else if let Some(cb) = cand_in {
            unsafe {
                self.gk.go(
                    "dg_mask_scores",
                    blocks(wmax, 256),
                    B256,
                    0,
                    &self.s,
                    args![score.ptr, cb.ptr, pp, ai, ri],
                )?
            };
        }
        let sel = self.fbuf(kmax)?;
        let (ki, oi) = (kmax as i32, offset as i32);
        unsafe {
            self.gk.go(
                "dg_topk",
                (1, 1, 1),
                SEL,
                0,
                &self.s,
                args![score.ptr, sel.ptr, out.ptr, pp, ai, ri, ki, oi],
            )
        }
    }

    // moe

    /// An fp8 quant-dequantized copy of `x` (n values). Kept out of the GEMV: fused into its
    /// column loop (`qdq` in `ds_lin_*`, bit-identical) it costs 2-8x the GEMV at t = 1..8,
    /// since every warp redoes it.
    fn qdq(&self, x: Buf, n: usize) -> Result<Buf> {
        let xq = self.fbuf(n)?;
        self.copy(xq.ptr, x.ptr, n)?;
        self.k.fp8_qdq(&self.s, xq.ptr, n)?;
        Ok(xq)
    }

    fn swiglu_into(&self, g: Buf, u: Buf, w: u64, h: Buf, n: usize, inter: usize) -> Result<()> {
        let (nn, ii, lim) = ((n * inter) as i64, inter as i32, self.cpu.cfg.swiglu_limit);
        unsafe {
            self.k.go(
                "ds_swiglu",
                blocks(n * inter, 256),
                B256,
                0,
                &self.s,
                args![g.ptr, u.ptr, w, h.ptr, nn, ii, lim],
            )
        }
    }

    fn mlp(&self, p: &str, x: Buf, n: usize, w: u64) -> Result<Buf> {
        let (n1, n3) = (format!("{p}.w1.weight"), format!("{p}.w3.weight"));
        if self.fast {
            let (w1, w3) = (self.dw(&n1)?, self.dw(&n3)?);
            if Kernels::fast_ok(&w1.v) && Kernels::fast_ok(&w3.v) {
                let h = self.fbuf(n * w1.v.rows as usize)?;
                let xq = self.qdq(x, n * w1.v.cols as usize)?;
                self.k.linear_fast(
                    &self.s,
                    &w1.v,
                    Some(&w3.v),
                    xq.ptr,
                    h.ptr,
                    n,
                    true,
                    false,
                    w,
                    self.cpu.cfg.swiglu_limit,
                )?;
                return self.lin(h, n, &format!("{p}.w2.weight"));
            }
        }
        let g = self.lin(x, n, &n1)?;
        let u = self.lin(x, n, &n3)?;
        let inter = self.dw(&n1)?.v.rows as usize;
        let h = self.fbuf(n * inter)?;
        self.swiglu_into(g, u, w, h, n, inter)?;
        self.lin(h, n, &format!("{p}.w2.weight"))
    }

    /// `lin` over at most expert `e`'s row count (from the device CSR `off`) of `t` rows,
    /// into `y`, with `xq` as the activation-quant scratch.
    #[allow(clippy::too_many_arguments)]
    fn lin_csr(
        &self,
        x: Buf,
        t: usize,
        name: &str,
        off: Buf,
        e: usize,
        y: Buf,
        xq: Buf,
    ) -> Result<()> {
        let w = self.dw(name)?;
        let k = w.v.cols as usize;
        let xp = if w.v.fmt >= 2 {
            self.copy(xq.ptr, x.ptr, t * k)?;
            self.k.fp8_qdq(&self.s, xq.ptr, t * k)?;
            xq.ptr
        } else {
            x.ptr
        };
        let n = w.v.rows as usize;
        let (wv, ti, r, ei) = (w.v, t as i32, 1i32, e as i32);
        unsafe {
            self.k.go(
                "ds_linear_csr",
                blocks(t * n * 32, 256),
                B256,
                0,
                &self.s,
                args![wv, xp, y.ptr, ti, r, off.ptr, ei],
            )
        }
    }

    fn moe(&self, p: &str, x: Buf, t: usize, lid: usize) -> Result<Buf> {
        let c = &self.cpu.cfg;
        let (ne, topk) = c.moe(lid);
        let d = c.dim;
        let s = self.span("router");
        let raw = self.raw(x.ptr, t, &format!("{p}.ffn.gate.weight"))?;
        let bias = self.dw(&format!("{p}.ffn.gate.bias"))?;
        let (ids, wts) = (self.fbuf(t * topk)?, self.fbuf(t * topk)?);
        let (ti, nei, ki, norm, rs) = (
            t as i32,
            ne as i32,
            topk as i32,
            c.norm_topk_prob as i32,
            c.route_scale,
        );
        unsafe {
            self.k.go(
                "ds_router",
                blocks(t, 64),
                (64, 1, 1),
                0,
                &self.s,
                args![raw.ptr, bias.v.w, ids.ptr, wts.ptr, ti, nei, ki, norm, rs],
            )?
        };
        self.end(s);
        // per-expert CSR on the device
        let s = self.span("route");
        let (counts, off) = (self.fbuf(ne)?, self.fbuf(ne + 1)?);
        let (rows, rw) = (self.fbuf(t * topk)?, self.fbuf(t * topk)?);
        self.k.route(
            &self.s, ids.ptr, wts.ptr, counts.ptr, off.ptr, rows.ptr, rw.ptr, t, topk, ne,
        )?;
        let tab = if self.moe_grouped && self.expert_slots == 0 && t <= 8 {
            self.moe_tab(p, lid, ne)?
        } else {
            None
        };
        self.end(s);
        if let Some(tab) = tab {
            let y = self.moe_grouped_routed(&tab, x, t, topk, ne, ids, counts, off, rows, rw)?;
            return self.moe_finish(p, x, t, y);
        }
        // the expert LRU uploads on the host's say: it alone reads the counts back, to skip
        // (and not upload) unrouted experts
        let live: Vec<bool> = if self.expert_slots > 0 {
            self.sync()?;
            super::obs::d2h(ne * 4);
            counts
                .dev()
                .read::<i32>(0, ne)?
                .iter()
                .map(|&n| n > 0)
                .collect()
        } else {
            vec![true; ne]
        };
        let s = self.span("experts");
        let y = self.zbuf(t * d)?;
        // one set of expert scratch, reused expert after expert (the stream orders it)
        let inter =
            (0..ne)
                .find(|&e| (self.owns_expert)(e))
                .map_or(Ok(1), |e| -> Result<usize> {
                    Ok(self
                        .cpu
                        .st
                        .info(&format!("{p}.ffn.experts.{e}.w1.weight"))?
                        .shape[0])
                })?;
        let (xs, w, out) = (self.fbuf(t * d)?, self.fbuf(t)?, self.fbuf(t * d)?);
        let (g, u, h) = (
            self.fbuf(t * inter)?,
            self.fbuf(t * inter)?,
            self.fbuf(t * inter)?,
        );
        let xq = self.fbuf(t * d.max(inter))?;
        let (ei_t, di) = (t as i32, d as i32);
        for e in (0..ne).filter(|&e| live[e] && (self.owns_expert)(e)) {
            let ei = e as i32;
            unsafe {
                self.k.go(
                    "ds_gather_csr",
                    blocks(t * d, 256),
                    B256,
                    0,
                    &self.s,
                    args![x.ptr, off.ptr, rows.ptr, rw.ptr, xs.ptr, w.ptr, ei, ei_t, di],
                )?
            };
            let pe = format!("{p}.ffn.experts.{e}");
            let n1 = format!("{pe}.w1.weight");
            self.lin_csr(xs, t, &n1, off, e, g, xq)?;
            self.lin_csr(xs, t, &format!("{pe}.w3.weight"), off, e, u, xq)?;
            self.swiglu_into(g, u, w.ptr, h, t, inter)?;
            self.lin_csr(h, t, &format!("{pe}.w2.weight"), off, e, out, xq)?;
            unsafe {
                self.k.go(
                    "ds_scatter_csr",
                    blocks(t * d, 256),
                    B256,
                    0,
                    &self.s,
                    args![y.ptr, out.ptr, off.ptr, rows.ptr, ei, ei_t, di],
                )?
            };
        }
        self.end(s);
        self.moe_finish(p, x, t, y)
    }

    /// The routed experts' partial sum y [t][d] by the grouped kernels: plan (live local
    /// experts), gate/up + SwiGLU, quantize, down to per-slot partials, ordered reduce.
    #[allow(clippy::too_many_arguments)]
    fn moe_grouped_routed(
        &self,
        tab: &MoeTab,
        x: Buf,
        t: usize,
        topk: usize,
        ne: usize,
        ids: Buf,
        counts: Buf,
        off: Buf,
        rows: Buf,
        rw: Buf,
    ) -> Result<Buf> {
        let d = self.cpu.cfg.dim;
        let (n, inter) = (t * topk, tab.inter);
        let slots = n.min(tab.local);
        let s = self.span("grouped_gu");
        let elist = self.fbuf(slots)?;
        let (nei, ci) = (ne as i32, slots as i32);
        unsafe {
            self.k.go(
                "ds_moe_plan",
                (1, 1, 1),
                (32, 1, 1),
                0,
                &self.s,
                args![counts.ptr, tab.mask.ptr, elist.ptr, nei, ci],
            )?
        };
        let xq = self.qdq(x, t * d)?;
        let h = self.fbuf(n * inter)?;
        let lim = self.cpu.cfg.swiglu_limit;
        self.k.moe_gu(
            &self.s, tab.fmt, tab.w1.ptr, tab.w3.ptr, elist.ptr, off.ptr, rows.ptr, rw.ptr, xq.ptr,
            h.ptr, t, slots, inter, lim,
        )?;
        self.k.fp8_qdq(&self.s, h.ptr, n * inter)?;
        self.end(s);
        let s = self.span("grouped_down");
        let part = self.fbuf(n * d)?;
        self.k.moe_down(
            &self.s, tab.fmt, tab.w2.ptr, elist.ptr, off.ptr, h.ptr, part.ptr, t, slots, d,
        )?;
        let y = self.fbuf(t * d)?;
        let (ti, ki, di) = (t as i32, topk as i32, d as i32);
        unsafe {
            self.k.go(
                "ds_moe_reduce",
                blocks(t * d, 256),
                B256,
                0,
                &self.s,
                args![
                    y.ptr,
                    part.ptr,
                    ids.ptr,
                    off.ptr,
                    rows.ptr,
                    tab.mask.ptr,
                    ti,
                    ki,
                    di
                ],
            )?
        };
        self.end(s);
        Ok(y)
    }

    /// Layer `lid`'s grouped-MoE table, built (uploading every local expert) on first use;
    /// None when the experts are not one `linear_fast` format (the loop then runs).
    fn moe_tab(&self, p: &str, lid: usize, ne: usize) -> Result<Option<Rc<MoeTab>>> {
        if let Some(t) = self.moe_tabs.borrow().get(&lid) {
            return Ok(t.clone());
        }
        let zero = DsW::default();
        let (mut w1, mut w3, mut w2, mut mask) = (
            vec![zero; ne],
            vec![zero; ne],
            vec![zero; ne],
            vec![0i32; ne],
        );
        let mut keep = Vec::new();
        let mut ok = true;
        let mut shape: Option<(i32, i32, i32)> = None;
        for e in (0..ne).filter(|&e| (self.owns_expert)(e)) {
            let pe = format!("{p}.ffn.experts.{e}");
            let ws = [
                self.dw(&format!("{pe}.w1.weight"))?,
                self.dw(&format!("{pe}.w3.weight"))?,
                self.dw(&format!("{pe}.w2.weight"))?,
            ];
            let (a, b, c) = (ws[0].v, ws[1].v, ws[2].v);
            let sh = (a.fmt, a.rows, a.cols);
            ok &= Kernels::fast_ok(&a)
                && Kernels::fast_ok(&c)
                && (b.fmt, b.rows, b.cols) == sh
                && (c.fmt, c.rows, c.cols) == (a.fmt, a.cols, a.rows)
                && *shape.get_or_insert(sh) == sh;
            (w1[e], w3[e], w2[e], mask[e]) = (a, b, c, 1);
            keep.extend(ws);
        }
        let tab = match shape {
            Some((fmt, inter, _)) if ok => Some(Rc::new(MoeTab {
                w1: DevBuf::from_slice(&w1)?,
                w3: DevBuf::from_slice(&w3)?,
                w2: DevBuf::from_slice(&w2)?,
                mask: DevBuf::from_slice(&mask)?,
                fmt,
                inter: inter as usize,
                local: mask.iter().filter(|&&m| m != 0).count(),
                _keep: keep,
            })),
            _ => None,
        };
        self.moe_tabs.borrow_mut().insert(lid, tab.clone());
        Ok(tab)
    }

    /// The routed sum `y` [t][d] -> the MoE output: the EP reduce hook, then the shared
    /// expert added and the sum bf16-rounded once.
    fn moe_finish(&self, p: &str, x: Buf, t: usize, y: Buf) -> Result<Buf> {
        let d = self.cpu.cfg.dim;
        if let Some(reduce) = &self.moe_reduce {
            let s = self.span("reduce");
            reduce(&y.dev(), t * d, &self.s)?;
            self.end(s);
        }
        let s = self.span("shared");
        let sh = self.mlp(&format!("{p}.ffn.shared_experts"), x, t, 0)?;
        self.end(s);
        let s = self.span("finish");
        let nn = (t * d) as i64;
        unsafe {
            self.k.go(
                "ds_add_round",
                blocks(t * d, 256),
                B256,
                0,
                &self.s,
                args![y.ptr, sh.ptr, nn],
            )?
        };
        self.end(s);
        Ok(y)
    }

    // mhc and blocks

    fn hc_mixes(&self, p: &str, kind: &str, x: Buf, t: usize) -> Result<Mix> {
        let c = &self.cpu.cfg;
        let (hc, d) = (c.hc_mult, c.dim);
        let mix = (2 + hc) * hc;
        let mixes = self.raw(x.ptr, t, &format!("{p}.hc_{kind}_fn"))?;
        let (n, mi, eps) = ((hc * d) as i32, mix as i32, c.norm_eps);
        unsafe {
            self.k.go(
                "ds_hc_scale",
                (t as u32, 1, 1),
                B256,
                0,
                &self.s,
                args![x.ptr, mixes.ptr, n, mi, eps],
            )?
        };
        let m = Mix {
            pre: self.fbuf(t * hc)?,
            post: self.fbuf(t * hc)?,
            comb: self.fbuf(t * hc * hc)?,
        };
        let sc = self.dw(&format!("{p}.hc_{kind}_scale"))?;
        let base = self.dw(&format!("{p}.hc_{kind}_base"))?;
        self.k.sinkhorn(
            &self.s,
            mixes.ptr,
            sc.v.w,
            base.v.w,
            m.pre.ptr,
            m.post.ptr,
            m.comb.ptr,
            t,
            hc,
            c.hc_sinkhorn_iters,
            c.hc_eps,
        )?;
        Ok(m)
    }

    fn hc_pre(&self, x: Buf, pre: Buf, t: usize) -> Result<Buf> {
        let (hc, d) = (self.cpu.cfg.hc_mult, self.cpu.cfg.dim);
        let out = self.fbuf(t * d)?;
        let (ti, hci, di) = (t as i32, hc as i32, d as i32);
        unsafe {
            self.k.go(
                "ds_hc_pre",
                blocks(t * d, 256),
                B256,
                0,
                &self.s,
                args![x.ptr, pre.ptr, out.ptr, ti, hci, di],
            )?
        };
        Ok(out)
    }

    fn hc_post(&self, h: Buf, res: Buf, m: &Mix, t: usize) -> Result<Buf> {
        let (hc, d) = (self.cpu.cfg.hc_mult, self.cpu.cfg.dim);
        let out = self.fbuf(t * hc * d)?;
        let (ti, hci, di) = (t as i32, hc as i32, d as i32);
        unsafe {
            self.k.go(
                "ds_hc_post",
                blocks(t * hc * d, 256),
                B256,
                0,
                &self.s,
                args![h.ptr, res.ptr, m.post.ptr, m.comb.ptr, out.ptr, ti, hci, di],
            )?
        };
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn block(
        &mut self,
        p: &str,
        lid: usize,
        x: Buf,
        t: usize,
        sp: usize,
        pre_mix: Buf,
        attn: Attn,
    ) -> Result<(Buf, Buf)> {
        let d = self.cpu.cfg.dim;
        let s = self.span("hc");
        let am = self.hc_mixes(p, "attn", x, t)?;
        let h = self.hc_pre(x, pre_mix, t)?;
        self.tap(&format!("{p}.attn_pre"), h, t * d)?;
        self.rms(h, t, &format!("{p}.attn_norm.weight"))?;
        self.tap(&format!("{p}.attn_in"), h, t * d)?;
        self.end(s);
        let s = self.span("attn");
        let h = match attn {
            Attn::Main => self.attention(p, lid, h, t, sp)?,
            Attn::Spark(main_x, mt) => self.dspark_attention(p, lid, h, t, sp, main_x, mt)?,
        };
        self.end(s);
        self.tap(&format!("{p}.attn"), h, t * d)?;
        let s = self.span("hc");
        let x = self.hc_post(h, x, &am, t)?;
        let fm = self.hc_mixes(p, "ffn", x, t)?;
        let h = self.hc_pre(x, am.pre, t)?;
        self.rms(h, t, &format!("{p}.ffn_norm.weight"))?;
        self.end(s);
        self.tap(&format!("{p}.moe_in"), h, t * d)?;
        let s = self.span("moe");
        let h = self.moe(p, h, t, lid)?;
        self.end(s);
        self.tap(&format!("{p}.moe"), h, t * d)?;
        let s = self.span("hc");
        let x = self.hc_post(h, x, &fm, t)?;
        self.end(s);
        Ok((x, fm.pre))
    }

    // engram

    /// Rows of an engram table, decoded on the host from the mapped checkpoint (fp8 with a
    /// per-row × 32 e8m0 scale): only the rows a step touches are ever read.
    fn engram_rows(&self, name: &str, rows: &[i64]) -> Result<Vec<f32>> {
        let info = self.cpu.st.info(name)?;
        let ehd = info.shape[1];
        let w = self.cpu.st.bytes(name)?;
        let s = self
            .cpu
            .st
            .bytes(&format!("{}scale", name.strip_suffix("weight").unwrap()))?;
        let mut out = Vec::with_capacity(rows.len() * ehd);
        for &r in rows {
            let r = r as usize;
            for i in 0..ehd {
                out.push(e4m3_to_f32(w[r * ehd + i]) * e8m0_to_f32(s[r * (ehd / 32) + i / 32]));
            }
        }
        Ok(out)
    }

    /// Engram's gate over `rows` (the step's staged table rows for this layer).
    fn engram(&self, p: &str, lid: usize, x: Buf, rows: Buf, t: usize) -> Result<Buf> {
        let c = &self.cpu.cfg;
        let (hc, d) = (c.hc_mult, c.dim);
        self.tap(&format!("{p}.eg_rows"), rows, rows.len / 4)?;
        let kv = self.lin(rows, t, &format!("{p}.engram.wkv.weight"))?;
        self.tap(&format!("{p}.eg_kv"), kv, t * (hc + 1) * d)?;
        let qk = {
            let cached = self.qk.borrow().get(&lid).cloned();
            match cached {
                Some(q) => q,
                None => {
                    let (qw, kw) = (
                        self.cpu.w(&format!("{p}.engram.q_weight"))?,
                        self.cpu.w(&format!("{p}.engram.k_weight"))?,
                    );
                    let v: Vec<f32> = qw.data.iter().zip(&kw.data).map(|(a, b)| a * b).collect();
                    let bq = Rc::new(DevBuf::from_slice(&v)?);
                    self.qk.borrow_mut().insert(lid, bq.clone());
                    bq
                }
            }
        };
        let out = self.fbuf(t * hc * d)?;
        let (hci, di, eps) = (hc as i32, d as i32, c.norm_eps);
        unsafe {
            self.k.go(
                "ds_engram_gate",
                ((t * hc) as u32, 1, 1),
                B256,
                0,
                &self.s,
                args![x.ptr, kv.ptr, qk.ptr, out.ptr, hci, di, eps],
            )?
        };
        Ok(out)
    }

    // the forward

    fn embed_hc(&self, ids: Buf, t: usize, copies: usize) -> Result<Buf> {
        let d = self.cpu.cfg.dim;
        let e = self.dw("embed.weight")?;
        let out = self.fbuf(t * copies * d)?;
        let (ti, ci, dd) = (t as i32, copies as i32, d as i32);
        unsafe {
            self.k.go(
                "ds_embed",
                blocks(t * copies * d, 256),
                B256,
                0,
                &self.s,
                args![e.v.w, ids.ptr, out.ptr, ti, ci, dd],
            )?
        };
        Ok(out)
    }

    fn one_hot(&self, t: usize) -> Vec<f32> {
        let hc = self.cpu.cfg.hc_mult;
        (0..t * hc).map(|i| (i % hc == 0) as u8 as f32).collect()
    }

    /// Stage a step's host inputs: P, the ids, the mHC one-hot and the Engram rows. The
    /// first scratch of a step, so a graph's copies land at the same addresses every time.
    fn inputs(
        &mut self,
        ids: &[usize],
        sp: usize,
        hashes: Option<&[Vec<Vec<i64>>]>,
    ) -> Result<Inputs> {
        let c = self.cpu.cfg.clone();
        let t = ids.len();
        let mut copies = Vec::new();
        let (p, u) = self.stage(&[sp as i32, 0, 0, 0])?;
        copies.push(u);
        self.p = p;
        let (idb, u) = self.stage(&ids.iter().map(|&i| i as i32).collect::<Vec<_>>())?;
        copies.push(u);
        let (one_hot, u) = self.stage(&self.one_hot(t))?;
        copies.push(u);
        let mut engram = Vec::new();
        for (j, &l) in c.engram_layer_ids.iter().enumerate() {
            let hs = hashes.context("hashes")?;
            let flat: Vec<i64> = hs.iter().flat_map(|ht| ht[j].iter().copied()).collect();
            let rows = self.engram_rows(&format!("layers.{l}.engram.embed.weight"), &flat)?;
            let (bb, u) = self.stage(&rows)?;
            copies.push(u);
            engram.push(bb);
        }
        Ok(Inputs {
            ids: idb,
            one_hot,
            engram,
            copies,
        })
    }

    /// As [`Model::forward`]: the last position's logits and the DSpark target hiddens.
    /// `hook` (when given) sees the engram hashes and each block's stream, read back.
    pub fn forward(
        &mut self,
        ids: &[usize],
        sp: usize,
        hook: Option<&mut dyn FnMut(&str, &[f32])>,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        self.forward_impl(ids, sp, hook, false)
    }

    /// As [`Model::forward_all`]: every position's logits [T][V] (a verify window).
    pub fn forward_all(&mut self, ids: &[usize], sp: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        self.forward_impl(ids, sp, None, true)
    }

    /// Whether a step at `sp` with these settings is captured (see the module doc).
    fn graphable(&self, sp: usize, hook: bool) -> bool {
        self.graph
            && sp > 0
            && !hook
            && self.force.is_none()
            && self.expert_slots == 0
            && (self.moe_reduce.is_none() || self.reduce_graphable)
            && std::env::var_os("DS_TAP").is_none()
            && !self.obs.direct()
    }

    fn forward_impl(
        &mut self,
        ids: &[usize],
        sp: usize,
        mut hook: Option<&mut dyn FnMut(&str, &[f32])>,
        all: bool,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let c = self.cpu.cfg.clone();
        let (d, t, v) = (c.dim, ids.len(), c.vocab_size);
        self.begin_step()?;
        let kind = match (sp, all && t > 1) {
            (0, _) => "prefill",
            (_, true) => "verify",
            _ => "decode",
        };
        self.obs.begin(&self.s, kind, t, sp);
        let hashes = if c.engram_layer_ids.is_empty() {
            None
        } else {
            Some(self.cpu.engram_hash(ids, sp))
        };
        if let (Some(hs), Some(h)) = (&hashes, hook.as_mut()) {
            h(
                "hash",
                &hs.iter()
                    .flatten()
                    .flatten()
                    .map(|&v| v as f32)
                    .collect::<Vec<_>>(),
            );
        }
        let inp = self.inputs(ids, sp, hashes.as_deref())?;
        // host code of a decode step never sees the position (see `DYN`)
        let bsp = if sp == 0 { 0 } else { DYN };
        let key = (t, all);
        let gen = self.gen();
        if std::env::var_os("DS_GRAPH_DEBUG").is_some() {
            eprintln!("step t={t} sp={sp} gen={gen:?} {:?}", self.graph_stats);
        }
        let out = if self.graphable(sp, hook.is_some()) {
            if let Some(cg) = self.graphs.get(&key) {
                self.obs.path(true, false);
                super::obs::replay();
                super::obs::h2d(inp.copies.iter().map(|u| u.n).sum());
                cg.g.launch(&self.s)?;
                self.graph_stats.replays += 1;
                cg.out.clone()
            } else if self.warm.get(&key) == Some(&gen) {
                // warmed up at this generation: every weight is resident and the arena fits
                // the step, so the capture allocates nothing
                let s = ManuallyDrop::new(Stream(self.s.0));
                // no timeline events inside the capture
                self.obs.path(true, true);
                super::obs::capture();
                self.arena.borrow_mut().frozen = true;
                let mut out = None;
                let g = Graph::capture(&s, |s| {
                    let r = (|| -> Result<StepOut> {
                        for u in &inp.copies {
                            u.enqueue(s)?;
                        }
                        self.body(&inp, t, bsp, all, None)
                    })();
                    let ok = r.is_ok();
                    out = Some(r);
                    if ok {
                        Ok(())
                    } else {
                        Err(tang_moe::gpu::Error("capture body failed".into()))
                    }
                });
                self.arena.borrow_mut().frozen = false;
                let out = out.context("capture")?.context("captured step")?;
                let g = g?;
                g.launch(&self.s)?;
                self.graph_stats.captures += 1;
                self.graphs.insert(
                    key,
                    Captured {
                        g,
                        out: out.clone(),
                        gen,
                    },
                );
                out
            } else {
                self.obs.path(false, false);
                let out = self.direct(&inp, t, bsp, all, None)?;
                // a step that grew the arena moves it at the next start: warm up again
                self.warm.insert(key, self.gen_after());
                out
            }
        } else {
            self.obs.path(false, false);
            self.direct(&inp, t, bsp, all, hook)?
        };
        self.obs.gpu_end(&self.s, self.arena.borrow().capacity());
        if self.journal_on && sp > 0 {
            self.journal_steps += 1;
            self.journal = out
                .jrows
                .iter()
                .map(|&(l, e, kv, sc)| (l, sp + e, kv, sc))
                .collect();
        }
        if self.skip_readback {
            // its events complete by the next step's start, which emits it
            self.obs.finish();
            return Ok((Vec::new(), Vec::new()));
        }
        self.sync()?;
        super::obs::d2h((out.rows * v + out.mains.len() * t * d) * 4);
        let logits = out.logits.dev().read::<f32>(0, out.rows * v)?;
        let n = out.mains.len();
        let mut main = vec![0f32; t * d * n];
        for (j, m) in out.mains.iter().enumerate() {
            let mv = m.dev().read::<f32>(0, t * d)?;
            for ti in 0..t {
                main[(ti * n + j) * d..(ti * n + j + 1) * d]
                    .copy_from_slice(&mv[ti * d..(ti + 1) * d]);
            }
        }
        self.obs.finish();
        Ok((logits, main))
    }

    /// The generation the next step will start at (one more when this one overflowed).
    fn gen_after(&self) -> (u64, u64) {
        let (a, s) = (self.arena.borrow(), self.stage.borrow());
        (a.gen + a.overflowed() as u64, s.gen + s.overflowed() as u64)
    }

    fn direct(
        &mut self,
        inp: &Inputs,
        t: usize,
        sp: usize,
        all: bool,
        hook: Option<&mut dyn FnMut(&str, &[f32])>,
    ) -> Result<StepOut> {
        for u in &inp.copies {
            u.enqueue(&self.s)?;
        }
        self.graph_stats.direct += 1;
        self.body(inp, t, sp, all, hook)
    }

    /// The step on the device, from staged inputs: no host sync, no host-side position when
    /// `sp` is [`DYN`].
    fn body(
        &mut self,
        inp: &Inputs,
        t: usize,
        sp: usize,
        all: bool,
        mut hook: Option<&mut dyn FnMut(&str, &[f32])>,
    ) -> Result<StepOut> {
        let c = self.cpu.cfg.clone();
        let (hc, d) = (c.hc_mult, c.dim);
        self.jrows.clear();
        let sp_embed = self.span("embed");
        let mut h = self.embed_hc(inp.ids, t, hc)?;
        self.end(sp_embed);
        let mut pre_mix = inp.one_hot;
        let mut mains: Vec<Buf> = Vec::new();
        for i in 0..c.n_layers {
            let p = format!("layers.{i}");
            let sp_layer = self.obs.span(&self.s, "layer", Some(i));
            if let Some(f) = &self.force {
                let (a, bb) = (f[i].0.clone(), f[i].1.clone());
                h = self.up(&a)?;
                pre_mix = self.up(&bb)?;
            }
            if let Some(j) = c.engram_layer_ids.iter().position(|&l| l == i) {
                let se = self.span("engram");
                h = self.engram(&p, i, h, inp.engram[j], t)?;
                self.end(se);
            }
            if c.dspark_target_layer_ids.contains(&i) {
                let sh = self.span("hc");
                let m = self.fbuf(t * d)?;
                let (ti, hci, di) = (t as i32, hc as i32, d as i32);
                unsafe {
                    self.k.go(
                        "ds_mean_hc",
                        blocks(t * d, 256),
                        B256,
                        0,
                        &self.s,
                        args![h.ptr, m.ptr, ti, hci, di],
                    )?
                };
                mains.push(m);
                self.end(sh);
            }
            let (nh, npm) = self.block(&p, i, h, t, sp, pre_mix, Attn::Main)?;
            h = nh;
            pre_mix = npm;
            self.end(sp_layer);
            if let Some(hk) = hook.as_mut() {
                hk(&format!("layer{i}"), &self.read(h, t * hc * d)?);
            }
        }
        let sp_head = self.span("head");
        let x = self.hc_pre(h, pre_mix, t)?;
        self.rms(x, t, "norm.weight")?;
        let rows = if all { t } else { 1 };
        let logits = self.raw(x.ptr + ((t - rows) * d * 4) as u64, rows, "head.weight")?;
        self.end(sp_head);
        Ok(StepOut {
            logits,
            rows,
            mains,
            jrows: std::mem::take(&mut self.jrows),
        })
    }

    // dspark

    #[allow(clippy::too_many_arguments)]
    fn dspark_attention(
        &mut self,
        p: &str,
        lid: usize,
        x: Buf,
        bl: usize,
        sp: usize,
        main_x: Buf,
        t: usize,
    ) -> Result<Buf> {
        let c = self.cpu.cfg.clone();
        let (h, hd, win) = (c.n_heads, c.head_dim, c.window_size);
        let at = |pos0| At::Host {
            pos0,
            stride: 1,
            fixed: None,
        };
        let mkv = self.lin(main_x, t, &format!("{p}.attn.wkv.weight"))?;
        self.rms(mkv, t, &format!("{p}.attn.kv_norm.weight"))?;
        self.rope(mkv, t, 1, hd, lid, at(sp), false)?;
        self.k.fp8_qdq(&self.s, mkv.ptr, t * hd)?;
        if sp == 0 {
            self.ring_seed(lid, mkv, t)?;
            return self.fbuf(1);
        }
        let pos0 = sp + t;
        let qr = self.lin(x, bl, &format!("{p}.attn.wq_a.weight"))?;
        self.rms(qr, bl, &format!("{p}.attn.q_norm.weight"))?;
        let q = self.lin(qr, bl, &format!("{p}.attn.wq_b.weight"))?;
        self.rope(q, bl, h, hd, lid, at(pos0), false)?;
        let kv = self.lin(x, bl, &format!("{p}.attn.wkv.weight"))?;
        self.rms(kv, bl, &format!("{p}.attn.kv_norm.weight"))?;
        self.rope(kv, bl, 1, hd, lid, at(pos0), false)?;
        self.k.fp8_qdq(&self.s, kv.ptr, bl * hd)?;
        // the ring holds positions < sp + t once the t main rows are written (mode 2 counts
        // `sp + 1` ring rows, so pass the last written position)
        let nk = win.min(sp + t) + bl;
        let idx = self.win_idx(bl, nk, nk, 2, sp + t - 1, bl)?;
        let ring = self.win[lid].ptr;
        for i in 0..t {
            let s = ((sp + i) % win * hd * 4) as u64;
            self.copy(ring + s, mkv.ptr + (i * hd * 4) as u64, hd)?;
        }
        let all = self.fbuf((win + bl) * hd)?;
        self.copy(all.ptr, ring, win * hd)?;
        self.copy(all.ptr + (win * hd * 4) as u64, kv.ptr, bl * hd)?;
        self.attn_out(p, q, (all, win + bl, 0), idx, bl, nk, lid, at(pos0))
    }

    /// As [`Model::forward_spec`]: greedy drafts.
    pub fn forward_spec(
        &mut self,
        token: usize,
        main_hidden: &[f32],
        sp: usize,
    ) -> Result<Option<super::forward::Draft>> {
        self.forward_spec_with(token, main_hidden, sp, &mut argmax)
    }

    /// As [`Model::forward_spec_with`]: each draft is `draw(logits)` (Markov bias included).
    /// Not captured: it runs directly, then samples on the host.
    pub fn forward_spec_with(
        &mut self,
        token: usize,
        main_hidden: &[f32],
        sp: usize,
        draw: &mut dyn FnMut(&[f32]) -> usize,
    ) -> Result<Option<super::forward::Draft>> {
        let c = self.cpu.cfg.clone();
        let (hc, d, blk, n_mtp, v) = (
            c.hc_mult,
            c.dim,
            c.dspark_block_size,
            c.n_mtp_layers,
            c.vocab_size,
        );
        self.begin_step()?;
        self.obs.begin(&self.s, "draft", blk, sp);
        let t = main_hidden.len() / (d * c.dspark_target_layer_ids.len());
        let mh = self.up(main_hidden)?;
        let main_x = self.lin(mh, t, "mtp.0.main_proj.weight")?;
        self.rms(main_x, t, "mtp.0.main_norm.weight")?;
        let mut ids = vec![c.dspark_noise_token_id as i32; blk];
        ids[0] = token as i32;
        let idb = self.up(&ids)?;
        let mut h = self.embed_hc(idb, blk, hc)?;
        let mut pre_mix = self.up(&self.one_hot(blk))?;
        for s in 0..n_mtp {
            let (lid, p) = (c.n_layers + s, format!("mtp.{s}"));
            if sp == 0 {
                self.dspark_attention(&p, lid, h, blk, 0, main_x, t)?;
                continue;
            }
            let (nh, npm) = self.block(&p, lid, h, blk, sp, pre_mix, Attn::Spark(main_x, t))?;
            h = nh;
            pre_mix = npm;
        }
        if sp == 0 {
            self.obs.gpu_end(&self.s, self.arena.borrow().capacity());
            self.obs.finish();
            return Ok(None);
        }
        let last = format!("mtp.{}", n_mtp - 1);
        let x = self.hc_pre(h, pre_mix, blk)?;
        let xh = self.read(x, blk * d)?;
        self.rms(x, blk, &format!("{last}.norm.weight"))?;
        let dlogits = self.raw(x.ptr, blk, "head.weight")?;
        // the Markov head samples left to right: each position's bias (its predecessor's
        // Markov embedding through the head, a rank x V GEMV) on the device, the row read back
        // for the draw; the confidence head (a dot product) on the host
        let rank = c.dspark_markov_rank;
        let memb_w = self.cpu.w(&format!("{last}.markov_head.embed.weight"))?;
        let mhead = format!("{last}.markov_head.head.weight");
        let proj = self.cpu.w(&format!("{last}.confidence_head.proj.weight"))?;
        let mut logits = vec![0f32; blk * v];
        let mut out = vec![token];
        let mut conf = Vec::with_capacity(blk);
        for i in 0..blk {
            let memb = &memb_w.data[out[i] * rank..(out[i] + 1) * rank];
            let mx = self.up(memb)?;
            let bias = self.raw(mx.ptr, 1, &mhead)?;
            let row = dlogits.at(i * v * 4);
            let vn = v as i64;
            unsafe {
                self.k.go(
                    "ds_add",
                    blocks(v, 256),
                    B256,
                    0,
                    &self.s,
                    args![row.ptr, bias.ptr, vn],
                )?
            };
            let li = &mut logits[i * v..(i + 1) * v];
            li.copy_from_slice(&self.read(row, v)?);
            out.push(draw(li));
            let feat: Vec<f32> = xh[i * d..(i + 1) * d].iter().chain(memb).copied().collect();
            conf.push(super::forward::matmul(&feat, 1, &proj)[0]);
        }
        self.obs.gpu_end(&self.s, self.arena.borrow().capacity());
        self.obs.finish();
        Ok(Some(super::forward::Draft {
            ids: out,
            logits,
            conf,
        }))
    }

    // speculative decoding

    /// As [`Model::snapshot`], into the model's one snapshot store.
    pub fn snapshot(&mut self) -> Result<GpuSnapshot> {
        let c = self.cpu.cfg.clone();
        let hd = c.head_dim;
        if self.snap.is_none() {
            let z = |n: usize| -> Result<DevBuf> { Ok(DevBuf::alloc(n * 4)?) };
            let win = (0..c.n_layers)
                .map(|_| z(c.window_size * hd))
                .collect::<Result<_>>()?;
            let mut cst = HashMap::new();
            for &l in self.cst.keys() {
                let n = c.compress_ratios[l] * hd;
                cst.insert(l, (z(n)?, z(n)?));
            }
            self.snap = Some((win, cst));
        }
        let (sw, sc) = self.snap.as_ref().unwrap();
        let (sw, sc): (Vec<Buf>, HashMap<usize, (Buf, Buf)>) = (
            sw.iter().map(b).collect(),
            sc.iter().map(|(&l, (a, bb))| (l, (b(a), b(bb)))).collect(),
        );
        for (l, dst) in sw.iter().enumerate() {
            self.copy(dst.ptr, self.win[l].ptr, c.window_size * hd)?;
        }
        for (l, (a, bb)) in &sc {
            let n = c.compress_ratios[*l] * hd;
            self.copy(a.ptr, self.cst[l].0.ptr, n)?;
            self.copy(bb.ptr, self.cst[l].1.ptr, n)?;
        }
        self.journal_on = true;
        self.journal_steps = 0;
        self.journal.clear();
        Ok(GpuSnapshot { win: sw, cst: sc })
    }

    /// As [`Model::rewind`]: back to only positions `< from` fed. At most one forward may
    /// run under the snapshot (the journal lives in that step's scratch).
    pub fn rewind(&mut self, snap: &GpuSnapshot, from: usize, end: usize) -> Result<()> {
        if self.journal_steps > 1 {
            bail!("rewind: {} forwards under one snapshot", self.journal_steps);
        }
        let c = self.cpu.cfg.clone();
        let (win, hd) = (c.window_size, c.head_dim);
        for (l, old) in snap.win.iter().enumerate() {
            for pos in from..end.min(from + win) {
                let o = ((pos % win) * hd * 4) as u64;
                self.copy(self.win[l].ptr + o, old.ptr + o, hd)?;
            }
        }
        for (l, (a, bb)) in &snap.cst {
            let n = c.compress_ratios[*l] * hd;
            let (ka, kb) = (self.cst[l].0.ptr, self.cst[l].1.ptr);
            self.copy(ka, a.ptr, n)?;
            self.copy(kb, bb.ptr, n)?;
        }
        let journal = std::mem::take(&mut self.journal);
        for (lid, pos, kv, sc) in &journal {
            if *pos < from {
                let o = ((pos % c.compress_ratios[*lid]) * hd * 4) as u64;
                let (ka, kb) = (self.cst[lid].0.ptr, self.cst[lid].1.ptr);
                self.copy(ka + o, kv.ptr, hd)?;
                self.copy(kb + o, sc.ptr, hd)?;
            }
        }
        self.journal_on = false;
        Ok(())
    }

    /// As [`Model::decode_spec`].
    pub fn decode_spec(
        &mut self,
        token: usize,
        main: &[f32],
        sp: usize,
        edit: &mut dyn FnMut(usize, &mut [usize]),
        pick: &mut dyn SpecPick,
    ) -> Result<SpecStep> {
        let c = self.cpu.cfg.clone();
        let (v, w) = (c.vocab_size, c.dim * c.dspark_target_layer_ids.len());
        let pos0 = sp + main.len() / w;
        let (start, t0) = (std::time::SystemTime::now(), std::time::Instant::now());
        let draft = self
            .forward_spec_with(token, main, sp, &mut |q: &[f32]| pick.draft(q))?
            .context("decode_spec needs sp > 0")?;
        let mut ids = draft.ids;
        edit(pos0 + 1, &mut ids[1..]);
        let t1 = std::time::Instant::now();
        let snap = self.snapshot()?;
        let (logits, mains) = self.forward_all(&ids, pos0)?;
        let n = ids.len();
        let (k, tokens) = super::forward::verify_rows(&logits, &draft.logits, &ids, v, pick);
        let t2 = std::time::Instant::now();
        self.rewind(&snap, pos0 + k + 1, pos0 + n)?;
        let ms = |a: std::time::Instant, b: std::time::Instant| (b - a).as_secs_f64() * 1e3;
        if super::obs::level() > 0 || crate::telemetry::ds_enabled() {
            super::obs::spec(super::obs::SpecRec {
                drafted: n - 1,
                accepted: k,
                draft_ms: ms(t0, t1),
                verify_ms: ms(t1, t2),
                rewind_ms: ms(t2, std::time::Instant::now()),
                rank: self.obs.rank.get(),
                start,
            });
        }
        Ok(SpecStep {
            tokens,
            drafted: n - 1,
            accepted: k,
            main: mains[..(k + 1) * w].to_vec(),
            sp: pos0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::forward::tests::tiny;
    use super::*;

    fn load_ref(name: &str) -> Option<Vec<f32>> {
        let b = std::fs::read(format!("{}/ref/{name}.bin", tiny())).ok()?;
        Some(
            b.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        )
    }

    fn rel(a: &[f32], b: &[f32]) -> f32 {
        let m = b.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-12);
        a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / m
    }

    fn kl(p_logits: &[f32], q_logits: &[f32]) -> f64 {
        let ls = |l: &[f32]| {
            let m = l.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
            let z = l.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln() + m;
            l.iter().map(|&x| x as f64 - z).collect::<Vec<f64>>()
        };
        let (p, q) = (ls(p_logits), ls(q_logits));
        p.iter().zip(&q).map(|(a, b)| a.exp() * (a - b)).sum()
    }

    fn argmax(v: &[f32]) -> usize {
        (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b })
    }

    fn open() -> Option<(GpuModel, serde_json::Value)> {
        Gpu::for_test()?;
        let dir = Path::new(&tiny()).join("hf");
        if !Path::new(&tiny()).join("ref/step0_in0.bin").exists() {
            eprintln!("skip: no {} (run scripts/deepseek_tiny.py)", tiny());
            return None;
        }
        let meta = serde_json::from_str(
            &std::fs::read_to_string(format!("{}/ref/meta.json", tiny())).unwrap(),
        )
        .unwrap();
        Some((GpuModel::open(&dir).unwrap(), meta))
    }

    /// Step 0 intermediates, device against the CPU reference on the same forced input.
    #[test]
    #[ignore]
    fn gpu_tiny_taps() {
        let Some((mut m, meta)) = open() else { return };
        std::env::set_var("DS_TAP", "1");
        let n_layers = m.cpu.cfg.n_layers;
        let ids: Vec<usize> = meta["steps"][0]["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let force: Vec<(Vec<f32>, Vec<f32>)> = (0..n_layers)
            .map(|i| {
                (
                    load_ref(&format!("step0_in{i}")).unwrap(),
                    load_ref(&format!("step0_pm{i}")).unwrap(),
                )
            })
            .collect();
        let mut cpu = Model::open(&Path::new(&tiny()).join("hf")).unwrap();
        cpu.force = Some(force.clone());
        cpu.forward(&ids, 0, &mut |_, _| {}).unwrap();
        m.force = Some(force);
        m.forward(&ids, 0, None).unwrap();
        let want: HashMap<String, Vec<f32>> = cpu.taps.borrow().iter().cloned().collect();
        for (name, v) in m.taps.borrow().iter() {
            if let Some(w) = want.get(name) {
                eprintln!("{name}: rel {:.2e}", rel(v, w));
            }
        }
    }

    /// The device forward teacher-forced against the fixture, as the CPU reference's test.
    #[test]
    fn gpu_tiny_teacher_forced() {
        let Some((mut m, meta)) = open() else { return };
        let n_layers = m.cpu.cfg.n_layers;
        for (s, step) in meta["steps"].as_array().unwrap().iter().enumerate() {
            let ids: Vec<usize> = step["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let sp = step["start_pos"].as_u64().unwrap() as usize;
            m.force = Some(
                (0..n_layers)
                    .map(|i| {
                        (
                            load_ref(&format!("step{s}_in{i}")).unwrap(),
                            load_ref(&format!("step{s}_pm{i}")).unwrap(),
                        )
                    })
                    .collect(),
            );
            let mut got: HashMap<String, Vec<f32>> = HashMap::new();
            let mut hook = |n: &str, t: &[f32]| {
                got.insert(n.to_string(), t.to_vec());
            };
            let (logits, _) = m.forward(&ids, sp, Some(&mut hook)).unwrap();
            let rels: Vec<f32> = (0..n_layers)
                .map(|i| {
                    rel(
                        &got[&format!("layer{i}")],
                        &load_ref(&format!("step{s}_layer{i}")).unwrap(),
                    )
                })
                .collect();
            let k = kl(&load_ref(&format!("step{s}_logits")).unwrap(), &logits);
            let main = load_ref(&format!("step{s}_main_hidden")).unwrap();
            let next = step["next_id"].as_u64().unwrap() as usize;
            let draft = m.forward_spec(next, &main, sp).unwrap();
            let spec = match (draft, step["spec_ids"].as_array()) {
                (Some(d), Some(want)) => {
                    let want: Vec<usize> =
                        want.iter().map(|v| v.as_u64().unwrap() as usize).collect();
                    let r = load_ref(&format!("step{s}_spec_logits")).unwrap();
                    format!(
                        "spec {} kl0 {:.0e}",
                        if d.ids == want {
                            "same".into()
                        } else {
                            format!("{:?} want {want:?}", d.ids)
                        },
                        kl(
                            &r[..m.cpu.cfg.vocab_size],
                            &d.logits[..m.cpu.cfg.vocab_size]
                        )
                    )
                }
                _ => String::new(),
            };
            eprintln!(
                "step {s}: [{}] KL {k:.1e} top1 {} {spec}",
                rels.iter()
                    .map(|r| format!("{r:.1e}"))
                    .collect::<Vec<_>>()
                    .join(" "),
                argmax(&logits) == next
            );
            assert!(k < 1e-3, "step {s}: KL {k}");
            for (i, r) in rels.iter().enumerate() {
                assert!(*r < 5e-2, "step {s} layer {i}: rel {r}");
            }
        }
    }

    /// The expert LRU changes where weights live, never the math: a 3-slot cache gives the
    /// unbounded run's logits bit for bit.
    #[test]
    fn gpu_tiny_expert_cache_is_exact() {
        let Some((mut a, meta)) = open() else { return };
        let mut b = GpuModel::open(&Path::new(&tiny()).join("hf")).unwrap();
        b.expert_slots = 3;
        for step in meta["steps"].as_array().unwrap().iter().take(4) {
            let ids: Vec<usize> = step["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let sp = step["start_pos"].as_u64().unwrap() as usize;
            let (la, _) = a.forward(&ids, sp, None).unwrap();
            let (lb, _) = b.forward(&ids, sp, None).unwrap();
            assert_eq!(la, lb, "step at {sp}");
        }
        eprintln!(
            "misses: unbounded {} vs 3 slots {}",
            a.expert_misses(),
            b.expert_misses()
        );
        assert!(b.expert_misses() > a.expert_misses());
    }

    /// The grouped MoE kernels give the per-expert loop's logits and hiddens bit for bit, on
    /// decode steps and 6-token verify windows (prefill takes the loop in both).
    #[test]
    fn gpu_tiny_moe_grouped_matches_loop() {
        let Some((_, meta)) = open() else { return };
        let dir = Path::new(&tiny()).join("hf");
        let ids = prompt(&meta);
        let (mut a, mut b) = (GpuModel::open(&dir).unwrap(), GpuModel::open(&dir).unwrap());
        a.moe_grouped = true;
        b.moe_grouped = false;
        let (v, n) = (a.cpu.cfg.vocab_size, ids.len());
        let (la, _) = a.forward(&ids, 0, None).unwrap();
        let (lb, _) = b.forward(&ids, 0, None).unwrap();
        assert_eq!(la, lb, "prefill");
        let mut tok = argmax(&la);
        let mut pos = n;
        for i in 0..6 {
            let (la, ma) = a.forward(&[tok], pos, None).unwrap();
            let (lb, mb) = b.forward(&[tok], pos, None).unwrap();
            assert_eq!(la, lb, "decode step {i}");
            assert_eq!(ma, mb, "decode step {i} hiddens");
            tok = argmax(&la);
            pos += 1;
        }
        for round in 0..3 {
            let win: Vec<usize> = (0..6).map(|i| (tok + 31 * i + round) % v).collect();
            let (la, ma) = a.forward_all(&win, pos).unwrap();
            let (lb, mb) = b.forward_all(&win, pos).unwrap();
            assert_eq!(la, lb, "verify window {round}");
            assert_eq!(ma, mb, "verify window {round} hiddens");
            pos += 6;
            tok = argmax(&la[5 * v..]);
        }
        assert!(
            a.moe_tabs.borrow().values().any(|t| t.is_some()),
            "grouped path never taken"
        );
    }

    /// Free-running on the device, prefill then greedy decode; informational past top-1.
    #[test]
    fn gpu_tiny_free_running() {
        let Some((mut m, meta)) = open() else { return };
        let steps = meta["steps"].as_array().unwrap();
        let mut top1 = 0;
        for (s, step) in steps.iter().enumerate() {
            let ids: Vec<usize> = step["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let sp = step["start_pos"].as_u64().unwrap() as usize;
            let (logits, main) = m.forward(&ids, sp, None).unwrap();
            let next = step["next_id"].as_u64().unwrap() as usize;
            top1 += (argmax(&logits) == next) as usize;
            m.forward_spec(argmax(&logits), &main, sp).unwrap();
            eprintln!(
                "step {s}: KL {:.2e} top1 {}",
                kl(&load_ref(&format!("step{s}_logits")).unwrap(), &logits),
                argmax(&logits) == next
            );
        }
        assert!(top1 + 1 >= steps.len(), "top-1 {top1}/{}", steps.len());
    }

    fn prompt(meta: &serde_json::Value) -> Vec<usize> {
        meta["steps"][0]["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect()
    }

    /// As forward.rs `tiny_chunk_and_rewind_exact`, on the device.
    #[test]
    fn gpu_tiny_chunk_and_rewind_exact() {
        let Some((mut a, meta)) = open() else { return };
        let ids = prompt(&meta);
        let (v, n) = (a.cpu.cfg.vocab_size, ids.len());
        a.forward(&ids, 0, None).unwrap();
        let stream: Vec<usize> = (0..14).map(|i| (37 * i + 11) % v).collect();
        let mut want = Vec::new();
        for (i, &t) in stream.iter().enumerate() {
            want.push(a.forward(&[t], n + i, None).unwrap().0);
        }
        drop(a);
        for keep in 1..=6 {
            let mut b = GpuModel::open(&Path::new(&tiny()).join("hf")).unwrap();
            b.forward(&ids, 0, None).unwrap();
            b.forward(&stream[..1], n, None).unwrap();
            let snap = b.snapshot().unwrap();
            let (lw, _) = b.forward_all(&stream[1..7], n + 1).unwrap();
            for e in 0..6 {
                assert_eq!(lw[e * v..(e + 1) * v], want[1 + e][..], "window row {e}");
            }
            b.rewind(&snap, n + 1 + keep, n + 7).unwrap();
            for i in 1 + keep..stream.len() {
                let (l, _) = b.forward(&[stream[i]], n + i, None).unwrap();
                assert_eq!(l, want[i], "keep {keep} step {i}");
            }
        }
    }

    /// Graphs change how a decode step is launched, never the math: captured and replayed
    /// decode steps and verify windows give the direct run's logits bit for bit.
    #[test]
    fn gpu_tiny_graph_matches_direct() {
        let Some((_, meta)) = open() else { return };
        let dir = Path::new(&tiny()).join("hf");
        let ids = prompt(&meta);
        let (mut a, mut b) = (GpuModel::open(&dir).unwrap(), GpuModel::open(&dir).unwrap());
        a.graph = true;
        b.graph = false;
        let (v, n) = (a.cpu.cfg.vocab_size, ids.len());
        let (la, _) = a.forward(&ids, 0, None).unwrap();
        let (lb, _) = b.forward(&ids, 0, None).unwrap();
        assert_eq!(la, lb, "prefill");
        let mut tok = argmax(&la);
        let mut pos = n;
        for i in 0..10 {
            let (la, ma) = a.forward(&[tok], pos, None).unwrap();
            let (lb, mb) = b.forward(&[tok], pos, None).unwrap();
            assert_eq!(la, lb, "decode step {i}");
            assert_eq!(ma, mb, "decode step {i} hiddens");
            tok = argmax(&la);
            pos += 1;
        }
        for round in 0..4 {
            let win: Vec<usize> = (0..6).map(|i| (tok + 31 * i + round) % v).collect();
            let (sa, sb) = (a.snapshot().unwrap(), b.snapshot().unwrap());
            let (la, ma) = a.forward_all(&win, pos).unwrap();
            let (lb, mb) = b.forward_all(&win, pos).unwrap();
            assert_eq!(la, lb, "verify window {round}");
            assert_eq!(ma, mb, "verify window {round} hiddens");
            let keep = 1 + round;
            a.rewind(&sa, pos + keep, pos + 6).unwrap();
            b.rewind(&sb, pos + keep, pos + 6).unwrap();
            pos += keep;
            let (la, _) = a.forward(&[tok], pos, None).unwrap();
            let (lb, _) = b.forward(&[tok], pos, None).unwrap();
            assert_eq!(la, lb, "decode after rewind {round}");
            tok = argmax(&la);
            pos += 1;
        }
        eprintln!("graph: {:?}, direct: {:?}", a.graph_stats, b.graph_stats);
        assert!(a.graph_stats.replays >= 8, "{:?}", a.graph_stats);
        assert_eq!(b.graph_stats.captures + b.graph_stats.replays, 0);
    }

    /// `DS_OBS=2` on a direct step: one `layer` span per backbone layer, and the ops inside
    /// the layers plus `embed` and `head` account for the step's device time (within 20%).
    #[test]
    fn gpu_tiny_obs_layers_sum_to_step() {
        use super::super::obs;
        let Some((mut m, meta)) = open() else { return };
        m.obs.over.set(Some((2, 1, true)));
        obs::collect(true);
        let ids = prompt(&meta);
        let (mut l, _) = m.forward(&ids, 0, None).unwrap();
        let mut pos = ids.len();
        for _ in 0..5 {
            l = m.forward(&[argmax(&l)], pos, None).unwrap().0;
            pos += 1;
        }
        let recs: Vec<_> = obs::take()
            .into_iter()
            .filter(|r| r.kind == "decode" && r.t == 1 && r.sp >= ids.len())
            .collect();
        obs::collect(false);
        assert_eq!(recs.len(), 5);
        let nl = m.cpu.cfg.n_layers;
        for r in &recs {
            assert_eq!(r.path, "direct");
            assert!(
                r.counts.launches > 0 && r.counts.syncs >= 1,
                "{:?}",
                r.counts
            );
            assert_eq!(r.spans.iter().filter(|s| s.name == "layer").count(), nl);
            let top: f64 = r
                .spans
                .iter()
                .filter(|s| s.name == "embed" || s.name == "head")
                .map(|s| s.gpu_ms)
                .sum();
            let ops = obs::layer_ops_ms(r) + top;
            let ratio = ops / r.gpu_ms;
            eprintln!(
                "step {} gpu {:.3} ms, ops {:.3} ms ({:.1}%), covered {:.3} ms, launches {}",
                r.step,
                r.gpu_ms,
                ops,
                100.0 * ratio,
                obs::covered_ms(r),
                r.counts.launches
            );
            for (n, ms, k) in obs::breakdown(r) {
                eprintln!("  {n:<26} {ms:>8.3} ms x{k}");
            }
            assert!(
                (0.8..=1.2).contains(&ratio),
                "ops {ops} vs gpu {}",
                r.gpu_ms
            );
        }
    }

    /// The timeline doesn't change the math, and graph capture works with it on.
    #[test]
    fn gpu_tiny_obs_graph_exact() {
        let Some((mut a, meta)) = open() else { return };
        let mut b = GpuModel::open(&Path::new(&tiny()).join("hf")).unwrap();
        b.obs.over.set(Some((2, 1, false)));
        let ids = prompt(&meta);
        let (la, _) = a.forward(&ids, 0, None).unwrap();
        let (lb, _) = b.forward(&ids, 0, None).unwrap();
        assert_eq!(la, lb);
        let (mut ta, mut pos) = (argmax(&la), ids.len());
        for _ in 0..6 {
            let la = a.forward(&[ta], pos, None).unwrap().0;
            let lb = b.forward(&[ta], pos, None).unwrap().0;
            assert_eq!(la, lb, "at {pos}");
            ta = argmax(&la);
            pos += 1;
        }
        assert!(
            b.graph_stats.captures >= 1 && b.graph_stats.replays >= 2,
            "{:?}",
            b.graph_stats
        );
    }

    /// Timeline overhead: decode us/step with it off, at 1 and at 2 (graph path), and off vs 2
    /// on the direct path.
    #[test]
    #[ignore]
    fn gpu_tiny_obs_overhead() {
        let Some((_, meta)) = open() else { return };
        let dir = Path::new(&tiny()).join("hf");
        let ids = prompt(&meta);
        // the fixture's context is short: every round restarts the sequence
        let ctx = GpuModel::open(&dir).unwrap().cpu.cfg.max_seq_len;
        let steps = ctx.saturating_sub(ids.len() + 8).max(8);
        for (lvl, direct) in [(0, false), (1, false), (2, false), (0, true), (2, true)] {
            let mut m = GpuModel::open(&dir).unwrap();
            m.obs.over.set(Some((lvl, 1, direct)));
            let (l, _) = m.forward(&ids, 0, None).unwrap();
            let mut tok = argmax(&l);
            let mut pos = ids.len();
            for _ in 0..4 {
                tok = argmax(&m.forward(&[tok], pos, None).unwrap().0);
                pos += 1;
            }
            let mut best = f64::MAX;
            for _ in 0..10 {
                let t0 = std::time::Instant::now();
                for _ in 0..steps {
                    tok = argmax(&m.forward(&[tok], pos, None).unwrap().0);
                    pos += 1;
                }
                best = best.min(t0.elapsed().as_secs_f64() * 1e6 / steps as f64);
                m.reset().unwrap();
                pos = 0;
                let (l, _) = m.forward(&ids, 0, None).unwrap();
                tok = argmax(&l);
                pos += ids.len();
            }
            eprintln!("DS_OBS={lvl} direct={direct}: {best:.0} us/step");
        }
    }

    /// Decode-step time on the tiny model, graphs on vs off (launch overhead is the whole
    /// story at this size). `cargo test ... gpu_tiny_graph_bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn gpu_tiny_graph_bench() {
        let Some((_, meta)) = open() else { return };
        let dir = Path::new(&tiny()).join("hf");
        let ids = prompt(&meta);
        let steps = 50;
        for graph in [false, true] {
            let mut m = GpuModel::open(&dir).unwrap();
            m.graph = graph;
            let (l, _) = m.forward(&ids, 0, None).unwrap();
            let mut tok = argmax(&l);
            let mut pos = ids.len();
            // warm up: weights, arena, capture
            for _ in 0..4 {
                tok = argmax(&m.forward(&[tok], pos, None).unwrap().0);
                pos += 1;
            }
            let t0 = std::time::Instant::now();
            for _ in 0..steps {
                tok = argmax(&m.forward(&[tok], pos, None).unwrap().0);
                pos += 1;
            }
            let us = t0.elapsed().as_secs_f64() * 1e6 / steps as f64;
            eprintln!("graph {graph}: {us:.0} us/step ({:?})", m.graph_stats);
        }
    }

    /// As forward.rs `tiny_spec_matches_greedy`, on the device.
    #[test]
    fn gpu_tiny_spec_matches_greedy() {
        let Some((mut m, meta)) = open() else { return };
        let ids = prompt(&meta);
        let (v, n, len) = (m.cpu.cfg.vocab_size, ids.len(), 40);
        let (mut l, _) = m.forward(&ids, 0, None).unwrap();
        let mut want = Vec::new();
        while want.len() < len {
            want.push(argmax(&l));
            l = m
                .forward(&[*want.last().unwrap()], n + want.len() - 1, None)
                .unwrap()
                .0;
        }
        drop(m);
        let (oracle, oracle2) = (want.clone(), want.clone());
        let mut rng = 1u64;
        type Edit = Box<dyn FnMut(usize, &mut [usize])>;
        let runs: Vec<(&str, Edit)> = vec![
            ("dspark", Box::new(|_, _: &mut [usize]| {})),
            (
                "oracle",
                Box::new(move |pos, d: &mut [usize]| {
                    for (i, x) in d.iter_mut().enumerate() {
                        if let Some(&w) = oracle.get(pos + i - n) {
                            *x = w;
                        }
                    }
                }),
            ),
            (
                "wrong",
                Box::new(move |pos, d: &mut [usize]| {
                    for (i, x) in d.iter_mut().enumerate() {
                        if let Some(&w) = oracle2.get(pos + i - n) {
                            *x = w;
                        }
                        rng = rng
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        if (rng >> 33) % 5 == 0 {
                            *x = (*x + 1 + (rng >> 40) as usize % (v - 1)) % v;
                        }
                    }
                }),
            ),
        ];
        for (name, mut edit) in runs {
            let mut m = GpuModel::open(&Path::new(&tiny()).join("hf")).unwrap();
            let (l, main) = m.forward(&ids, 0, None).unwrap();
            let t0 = argmax(&l);
            m.forward_spec(t0, &main, 0).unwrap();
            let (l, mut main) = m.forward(&[t0], n, None).unwrap();
            let mut out = vec![t0, argmax(&l)];
            let (mut sp, mut drafted, mut accepted) = (n, 0, 0);
            while out.len() < len {
                let s = m
                    .decode_spec(
                        *out.last().unwrap(),
                        &main,
                        sp,
                        &mut *edit,
                        &mut super::super::forward::Greedy,
                    )
                    .unwrap();
                drafted += s.drafted;
                accepted += s.accepted;
                out.extend(s.tokens);
                main = s.main;
                sp = s.sp;
            }
            out.truncate(len);
            eprintln!(
                "{name}: accepted {accepted}/{drafted} ({:.1}%)",
                100.0 * accepted as f64 / drafted.max(1) as f64
            );
            assert_eq!(out, want, "{name}");
        }
    }
}
