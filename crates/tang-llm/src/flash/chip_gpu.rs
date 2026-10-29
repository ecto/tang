//! The chip window on CUDA: the compiled store layers ([`super::chip`], emitted by `tang-chip`)
//! run on the device as an engine-internal window with its own per-slot GDN and conv state.
//! A store tool call becomes chip windows: the call tokens and every token the circuit owns go
//! through the chip stack only (never the base layers, the LM head or the KV cache), and the
//! circuit's printed tokens come back to the caller. See `docs/design/gdn-chip.md` ("Engine").
//!
//! Per token, on one stream (each kernel returns at once when the op is over or flagged):
//!
//! ```text
//! embed(cur) → 12 × [ rmsnorm → CSR qkv|z|a|b → conv + SiLU (history shift) → L2 q,k
//!                     → GDN step + gated norm per v head → CSR out (+= x)
//!                     → rmsnorm → CSR gate|up → silu·up → CSR down (+= x) ]
//!            → pick: decode the residual, choose the next token (circuit's or queued)
//! ```
//!
//! A window is [`WINDOW`] such tokens captured as one CUDA graph per slot; the host reads the
//! control block between windows. The arithmetic is chip.rs's [`Numerics::Bf16`] term for term:
//! every CSR row sums its entries in column order from −0.0 (Rust's `f32` `Sum`); mul and add
//! are separate roundings (`__fmul_rn`/`__fadd_rn`: no FMA contraction, as Rust); `exp`/`log`
//! go through `double` (correctly rounded, as glibc's `expf`/`logf` are on these inputs);
//! the GDN recurrence is `reference::gdn_head_step`'s loop order per value column. bf16
//! rounding points (chip.rs `Layer::step`): norm outputs; each projection output; the conv
//! sum and its SiLU; β; the GDN core output; the gated norm's normed value, its gain product and
//! the SiLU(z) product; the out projection and the residual after it; gate and up outputs,
//! SiLU(gate) and the product; the down projection and the residual after it. Two reductions
//! are in a different order than the CPU's: the RMSNorm sums of squares (a block tree, not a
//! sequential sum). On a chip token that sum is `m · 2^80` (the ballast) plus terms far below
//! its half-ulp (2^62), so every order gives the same f32; the parity test checks it bitwise.

use std::collections::HashMap;
use std::ffi::c_void;

use anyhow::{anyhow, bail, ensure, Context, Result};
use tang_moe::gpu::{self, DevBuf, Gpu, Graph, Module, Stream};

use super::chip::{ChipPack, ChipState, PackedCsr, Shape};
use super::memserve::{OpOut, StoreOp};

/// Tokens per window (one graph launch, one host read).
pub const WINDOW: usize = 8;
/// Forced tokens an op may queue.
const QCAP: usize = 4096;
/// Tokens an op may feed (the CPU driver's runaway bound is 10,000).
const LOGCAP: usize = 8192;
/// Control words before the queue.
const CTL: usize = 16;

/// The control block: `[done, cur, qhead, qlen, nfed, err, _, err_arg, cur_from_chip, …]`,
/// then the queue, the fed log and each fed token's source (1: the circuit's).
const C_DONE: usize = 0;
const C_CUR: usize = 1;
const C_QHEAD: usize = 2;
const C_QLEN: usize = 3;
const C_NFED: usize = 4;
const C_ERR: usize = 5;
const C_ARG: usize = 7;
const C_SRC: usize = 8;

const SRC: &str = r#"
typedef unsigned int u32;
typedef unsigned long long u64;
#define STOP(c) ((c)[0] | (c)[5])

__device__ __forceinline__ float bf(float x) {
    if (x != x) return x;
    u32 u = __float_as_uint(x);
    u = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
    return __uint_as_float(u);
}
__device__ __forceinline__ float ex(float x) { return (float)exp((double)x); }
__device__ __forceinline__ float lg(float x) { return (float)log((double)x); }
__device__ __forceinline__ float silu_r(float x) { return __fdiv_rn(x, __fadd_rn(1.0f, ex(-x))); }
__device__ __forceinline__ float sigm_r(float x) { return __fdiv_rn(1.0f, __fadd_rn(1.0f, ex(-x))); }
__device__ __forceinline__ float softplus_r(float x) { return x > 20.0f ? x : lg(__fadd_rn(1.0f, ex(x))); }

extern "C" __global__ void chip_embed(u32* ctl, const int* tokmap, u32 ntok, const float* rows,
                                      u32 hidden, float* x) {
    if (STOP(ctl)) return;
    u32 tok = ctl[1];
    int r = tok < ntok ? tokmap[tok] : -1;
    if (r < 0) {
        if (blockIdx.x == 0 && threadIdx.x == 0) { ctl[7] = tok; ctl[5] = 1; }
        return;
    }
    for (u32 i = blockIdx.x * blockDim.x + threadIdx.x; i < hidden; i += gridDim.x * blockDim.x)
        x[i] = rows[(u64)r * hidden + i];
}

// One block of 1024: out = bf((x · 1/sqrt(mean(x²) + eps)) · w).
extern "C" __global__ void __launch_bounds__(1024) chip_rmsnorm(const u32* ctl, const float* x,
        const float* w, float* out, u32 n, float eps) {
    if (STOP(ctl)) return;
    __shared__ float red[1024];
    __shared__ float rr;
    float s = 0.0f;
    for (u32 i = threadIdx.x; i < n; i += 1024) s = __fadd_rn(s, __fmul_rn(x[i], x[i]));
    red[threadIdx.x] = s;
    __syncthreads();
    for (u32 o = 512; o > 0; o >>= 1) {
        if (threadIdx.x < o) red[threadIdx.x] = __fadd_rn(red[threadIdx.x], red[threadIdx.x + o]);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float ms = __fdiv_rn(red[0], (float)n);
        rr = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(ms, eps)));
    }
    __syncthreads();
    for (u32 i = threadIdx.x; i < n; i += 1024) out[i] = bf(__fmul_rn(__fmul_rn(x[i], rr), w[i]));
}

// y = bf(W x) row by row in column order from -0.0; with res: res = bf(res + y) instead.
extern "C" __global__ void chip_csr(const u32* ctl, const u32* ptr, const u32* idx, const float* val,
                                    const float* x, float* y, u32 rows, float* res) {
    if (STOP(ctl)) return;
    u32 r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    float acc = -0.0f;
    for (u32 e = ptr[r], b = ptr[r + 1]; e < b; e++) acc = __fadd_rn(acc, __fmul_rn(val[e], x[idx[e]]));
    float v = bf(acc);
    if (res) res[r] = bf(__fadd_rn(res[r], v));
    else y[r] = v;
}

// Causal conv over [history | this token] then SiLU, per channel; the history shifts.
extern "C" __global__ void chip_conv(const u32* ctl, const float* P, const float* W, float* H,
                                     float* C, u32 ch) {
    if (STOP(ctl)) return;
    u32 j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ch) return;
    float q = P[j];
    const float* w = W + 4 * j;
    float acc = __fmul_rn(w[3], q);
    for (u32 i = 0; i < 3; i++) acc = __fadd_rn(acc, __fmul_rn(H[i * ch + j], w[i]));
    C[j] = bf(silu_r(bf(acc)));
    H[j] = H[ch + j];
    H[ch + j] = H[2 * ch + j];
    H[2 * ch + j] = q;
}

// q and k heads L2-normed (the sum of squares in double, in order), q scaled. Block per head.
extern "C" __global__ void chip_l2(const u32* ctl, float* C, u32 hk, float eps, float qscale) {
    if (STOP(ctl)) return;
    __shared__ float r;
    float* v = C + blockIdx.x * 128;
    if (threadIdx.x == 0) {
        double s = -0.0;
        for (u32 i = 0; i < 128; i++) s = __dadd_rn(s, __dmul_rn((double)v[i], (double)v[i]));
        r = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn((float)s, eps)));
    }
    __syncthreads();
    float y = __fmul_rn(v[threadIdx.x], r);
    if (blockIdx.x < hk) y = __fmul_rn(y, qscale);
    v[threadIdx.x] = y;
}

// One v head per block, one value column per thread: gdn_head_step's loops (decay, kᵀS, delta,
// qᵀS) in its order, then the gated RMSNorm. P = [qkv | z | a | b].
extern "C" __global__ void __launch_bounds__(128) chip_gdn(const u32* ctl, const float* C,
        const float* P, float* S, float* O, const float* DT, const float* AN, const float* NW,
        u32 hk, u32 hv, u32 conv_dim, float eps) {
    if (STOP(ctl)) return;
    __shared__ float sq[128], sk[128], so[128];
    __shared__ float rr;
    u32 vh = blockIdx.x, b = threadIdx.x, kh = vh / (hv / hk), kd = hk * 128;
    sq[b] = C[kh * 128 + b];
    sk[b] = C[kd + kh * 128 + b];
    float v = C[2 * kd + vh * 128 + b];
    u32 vd = hv * 128;
    float g = __fmul_rn(AN[vh], softplus_r(__fadd_rn(P[conv_dim + vd + vh], DT[vh])));
    float decay = ex(g);
    float beta = bf(sigm_r(P[conv_dim + vd + hv + vh]));
    __syncthreads();
    float* st = S + (u64)vh * 128 * 128 + b;
    float acc = 0.0f;
    for (u32 a = 0; a < 128; a++) {
        float e = __fmul_rn(st[a * 128], decay);
        st[a * 128] = e;
        acc = __fadd_rn(acc, __fmul_rn(e, sk[a]));
    }
    float d = __fmul_rn(__fsub_rn(v, acc), beta);
    float o = 0.0f;
    for (u32 a = 0; a < 128; a++) {
        float e = __fadd_rn(st[a * 128], __fmul_rn(sk[a], d));
        st[a * 128] = e;
        o = __fadd_rn(o, __fmul_rn(e, sq[a]));
    }
    o = bf(o);
    so[b] = o;
    __syncthreads();
    if (b == 0) {
        float ms = -0.0f;
        for (u32 i = 0; i < 128; i++) ms = __fadd_rn(ms, __fmul_rn(so[i], so[i]));
        ms = __fdiv_rn(ms, 128.0f);
        rr = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(ms, eps)));
    }
    __syncthreads();
    float n = bf(__fmul_rn(bf(__fmul_rn(o, rr)), NW[b]));
    O[vh * 128 + b] = bf(__fmul_rn(n, silu_r(P[conv_dim + vh * 128 + b])));
}

extern "C" __global__ void chip_act(const u32* ctl, const float* GU, float* A, u32 inter) {
    if (STOP(ctl)) return;
    u32 u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= inter) return;
    A[u] = bf(__fmul_rn(bf(silu_r(GU[u])), GU[inter + u]));
}

// One thread: log the fed token, decode the residual (chip::Decoder), choose the next token
// (chip::pick_next). Errors: 2 no token for the code, 3 log full, 4 chip token without successor.
extern "C" __global__ void chip_pick(u32* ctl, const float* x, const u32* fam, u32 nfam, u32 nfl,
        const u32* printable, u32 nprint, u32 v0, const u32* code, u32 ncode, const u32* idc,
        u32 nidc, u32 qcap, u32 logcap) {
    if (STOP(ctl)) return;
    u32* Q = ctl + 16;
    u32* LOG = Q + qcap;
    u32* SRCF = LOG + logcap;
    u32 cur = ctl[1], n = ctl[4];
    if (n >= logcap) { ctl[5] = 3; return; }
    LOG[n] = cur;
    SRCF[n] = ctl[8];
    ctl[4] = n + 1;
    int has = 0;
    u32 t = 0;
    int anyid = 0;
    for (u32 i = 0; i < nidc; i++) anyid |= x[idc[i]] != 0.0f;
    if (anyid) {
        has = 1;
        for (u32 i = 0; i < nidc; i++) if (x[idc[i]] > 0.0f) t += 1u << i;
    } else {
        int anyc = 0;
        for (u32 i = 0; i < ncode; i++) anyc |= x[code[i]] != 0.0f;
        if (anyc) {
            u32 fid = 0, p = 0;
            for (u32 i = 0; i < nfam; i++) fid += (u32)(x[code[i]] > 0.0f) << i;
            for (u32 i = nfam; i < ncode; i++) p += (u32)(x[code[i]] > 0.0f) << (i - nfam);
            if (fid < nfl && fam[3 * fid] == 0 && p < nprint) { t = printable[p]; has = 1; }
            else if (fid < nfl && fam[3 * fid] == 1) { t = v0 + fam[3 * fid + 1]; has = 1; }
            else if (fid < nfl && fam[3 * fid] == 2 && p < fam[3 * fid + 2]) { t = v0 + fam[3 * fid + 1] + p; has = 1; }
            else { ctl[7] = fid | (p << 8); ctl[5] = 2; return; }
        }
    }
    if (cur >= v0) {
        if (!has) { ctl[5] = 4; return; }
        ctl[1] = t; ctl[8] = 1;
    } else if (has) {
        ctl[1] = t; ctl[8] = 1;
    } else if (ctl[2] < ctl[3]) {
        ctl[1] = Q[ctl[2]]; ctl[2] += 1; ctl[8] = 0;
    } else {
        ctl[0] = 1;
    }
}

extern "C" __global__ void chip_fill(float* p, u32 n, float v) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}
"#;

type Fun = cudarc::driver::sys::CUfunction;

struct Funs {
    embed: Fun,
    rmsnorm: Fun,
    csr: Fun,
    conv: Fun,
    l2: Fun,
    gdn: Fun,
    act: Fun,
    pick: Fun,
    fill: Fun,
}

struct DevCsr {
    rows: usize,
    ptr: DevBuf,
    idx: DevBuf,
    val: DevBuf,
}

impl DevCsr {
    fn new(p: &PackedCsr) -> Result<Self> {
        Ok(Self {
            rows: p.rows,
            ptr: up(&p.ptr)?,
            idx: up(&p.idx)?,
            val: up(&p.val)?,
        })
    }
}

struct DevLayer {
    proj: DevCsr,
    out: DevCsr,
    gu: DevCsr,
    down: DevCsr,
    in_norm: DevBuf,
    post_norm: DevBuf,
    conv: DevBuf,
    dt_bias: DevBuf,
    a_neg: DevBuf,
    gnorm: DevBuf,
}

/// One slot's chip state: per layer the conv history `[3][conv_dim]` and the recurrent state
/// `[v_heads][d_k][d_v]` (chip.rs [`ChipState`]'s layout).
struct SlotState {
    conv: Vec<DevBuf>,
    rec: Vec<DevBuf>,
}

/// Scratch for one token: the residual, a norm output, the stacked projections, the conv
/// output, the GDN output, gate|up and the activation.
struct Scratch {
    x: DevBuf,
    h: DevBuf,
    p: DevBuf,
    c: DevBuf,
    o: DevBuf,
    gu: DevBuf,
    act: DevBuf,
}

fn up<T: Copy>(v: &[T]) -> Result<DevBuf> {
    DevBuf::from_slice(v).map_err(|e| anyhow!("{e}"))
}

fn ge<T>(r: gpu::Result<T>) -> Result<T> {
    r.map_err(|e| anyhow!("{e}"))
}

/// What one op fed through the chip stack.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChipOut {
    /// Every token fed (forced call tokens, chip records, circuit-printed text), in order.
    pub fed: Vec<u32>,
    /// Whether each fed token was the circuit's (else the caller's queue).
    pub from_chip: Vec<bool>,
    /// First chip token id.
    pub v0: u32,
}

impl ChipOut {
    /// The circuit's printed answer: the text tokens (below `v0`) it produced.
    pub fn printed(&self) -> Vec<u32> {
        self.fed
            .iter()
            .zip(&self.from_chip)
            .filter(|(&t, &c)| c && t < self.v0)
            .map(|(&t, _)| t)
            .collect()
    }
}

/// Store ops as chip tokens and back: the tool-call text in the host tokenizer, and the
/// circuit's printed answer parsed into an [`OpOut`]. Tokenizer-specific (the store's id codes
/// and printable tables are token ids), so it comes with the model.
pub trait ChipCodec: Send {
    fn encode(&self, op: &StoreOp) -> Result<Vec<u32>>;
    fn decode(&self, op: &StoreOp, out: &ChipOut) -> Result<OpOut>;
}

/// The chip stack on the device, with per-slot state.
pub struct ChipGpu {
    pub shape: Shape,
    eps: f32,
    stream: Stream,
    _module: Module,
    f: Funs,
    layers: Vec<DevLayer>,
    rows: DevBuf,
    tokmap: DevBuf,
    ntok: u32,
    fam: DevBuf,
    nfl: u32,
    nfam: u32,
    printable: DevBuf,
    nprint: u32,
    pub v0: u32,
    code: DevBuf,
    ncode: u32,
    idc: DevBuf,
    nidc: u32,
    s: Scratch,
    ctl: DevBuf,
    slots: HashMap<usize, SlotState>,
    graphs: HashMap<usize, Graph>,
    /// Run windows as CUDA graphs (else eager launches).
    pub use_graphs: bool,
    /// Device bytes: weights (CSR + vectors), embedding rows + token map, decoder.
    pub weight_bytes: usize,
    pub row_bytes: usize,
}

impl ChipGpu {
    /// Upload `pack` (the context of `gpu` is bound to this thread).
    pub fn new(gpu: &Gpu, pack: &ChipPack) -> Result<Self> {
        ge(gpu.bind())?;
        let sh = pack.shape;
        ensure!(
            sh.d_state == 128 && sh.d_conv == 4,
            "the chip window is written for d_state 128, conv 4"
        );
        let module = ge(gpu.module(SRC))?;
        let fun = |n: &str| ge(module.func(n));
        let f = Funs {
            embed: fun("chip_embed")?,
            rmsnorm: fun("chip_rmsnorm")?,
            csr: fun("chip_csr")?,
            conv: fun("chip_conv")?,
            l2: fun("chip_l2")?,
            gdn: fun("chip_gdn")?,
            act: fun("chip_act")?,
            pick: fun("chip_pick")?,
            fill: fun("chip_fill")?,
        };
        let layers = pack
            .layers
            .iter()
            .map(|l| -> Result<DevLayer> {
                Ok(DevLayer {
                    proj: DevCsr::new(&l.proj)?,
                    out: DevCsr::new(&l.out)?,
                    gu: DevCsr::new(&l.gu)?,
                    down: DevCsr::new(&l.down)?,
                    in_norm: up(&l.in_norm)?,
                    post_norm: up(&l.post_norm)?,
                    conv: up(&l.conv)?,
                    dt_bias: up(&l.dt_bias)?,
                    a_neg: up(&l.a_neg)?,
                    gnorm: up(&l.gnorm)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let d = &pack.dec;
        let z = |n: usize| -> Result<DevBuf> { ge(DevBuf::zeroed(4 * n)) };
        let (w, r) = pack.bytes();
        let proj_rows = sh.conv_dim() + sh.value_dim() + 2 * sh.v_heads;
        let s = Scratch {
            x: z(sh.hidden)?,
            h: z(sh.hidden)?,
            p: z(proj_rows)?,
            c: z(sh.conv_dim())?,
            o: z(sh.value_dim())?,
            gu: z(2 * sh.inter)?,
            act: z(sh.inter)?,
        };
        let me = Self {
            shape: sh,
            eps: pack.eps,
            stream: ge(Stream::new())?,
            _module: module,
            f,
            layers,
            rows: up(&pack.rows)?,
            tokmap: up(&pack.tokmap)?,
            ntok: pack.tokmap.len() as u32,
            fam: up(&d.fam)?,
            nfl: (d.fam.len() / 3) as u32,
            nfam: d.nfam,
            printable: up(&d.printable)?,
            nprint: d.printable.len() as u32,
            v0: d.v0,
            code: up(&d.code_dims)?,
            ncode: d.code_dims.len() as u32,
            idc: up(&d.id_code_dims)?,
            nidc: d.id_code_dims.len() as u32,
            s,
            ctl: z(CTL + QCAP + 2 * LOGCAP)?,
            slots: HashMap::new(),
            graphs: HashMap::new(),
            use_graphs: true,
            weight_bytes: w,
            row_bytes: r,
        };
        ge(gpu::check(
            unsafe { cudarc::driver::sys::cuCtxSynchronize() },
            "cuCtxSynchronize",
        ))?;
        Ok(me)
    }

    /// Device bytes of one slot's state.
    pub fn slot_bytes(&self) -> usize {
        let sh = &self.shape;
        4 * self.layers.len() * (3 * sh.conv_dim() + sh.v_heads * 128 * 128)
    }

    fn slot(&mut self, slot: usize) -> Result<&SlotState> {
        if !self.slots.contains_key(&slot) {
            let sh = self.shape;
            let n = self.layers.len();
            let mut st = SlotState {
                conv: Vec::with_capacity(n),
                rec: Vec::with_capacity(n),
            };
            for _ in 0..n {
                st.conv.push(ge(DevBuf::zeroed(4 * 3 * sh.conv_dim()))?);
                st.rec.push(ge(DevBuf::zeroed(4 * sh.v_heads * 128 * 128))?);
            }
            self.slots.insert(slot, st);
        }
        Ok(&self.slots[&slot])
    }

    /// Drop slot `slot`'s chip state (and its window graph).
    pub fn free(&mut self, slot: usize) {
        self.graphs.remove(&slot);
        self.slots.remove(&slot);
    }

    /// Zero every slot's chip state.
    pub fn zero_all(&mut self) -> Result<()> {
        for st in self.slots.values() {
            for b in st.conv.iter().chain(&st.rec) {
                ge(gpu::check(
                    unsafe { cudarc::driver::sys::cuMemsetD8Async(b.ptr, 0, b.len, self.stream.0) },
                    "cuMemsetD8Async",
                ))?;
            }
        }
        ge(self.stream.sync())
    }

    /// Slot `slot`'s chip state on the host.
    pub fn state(&mut self, slot: usize) -> Result<ChipState> {
        ge(self.stream.sync())?;
        let st = self.slot(slot)?;
        Ok(ChipState {
            conv: st
                .conv
                .iter()
                .map(|b| ge(b.read::<f32>(0, b.len / 4)))
                .collect::<Result<_>>()?,
            rec: st
                .rec
                .iter()
                .map(|b| ge(b.read::<f32>(0, b.len / 4)))
                .collect::<Result<_>>()?,
        })
    }

    /// One layer's state of slot `slot`: `(conv history, recurrent state)`.
    pub fn layer_state(&mut self, slot: usize, l: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        ge(self.stream.sync())?;
        let st = self.slot(slot)?;
        Ok((
            ge(st.conv[l].read::<f32>(0, st.conv[l].len / 4))?,
            ge(st.rec[l].read::<f32>(0, st.rec[l].len / 4))?,
        ))
    }

    /// Replace slot `slot`'s chip state.
    pub fn set_state(&mut self, slot: usize, s: &ChipState) -> Result<()> {
        ge(self.stream.sync())?;
        let st = self.slot(slot)?;
        ensure!(s.conv.len() == st.conv.len() && s.rec.len() == st.rec.len());
        for (b, v) in st.conv.iter().zip(&s.conv).chain(st.rec.iter().zip(&s.rec)) {
            ensure!(v.len() * 4 == b.len, "chip state size");
            ge(b.write(0, v))?;
        }
        ge(gpu::check(
            unsafe { cudarc::driver::sys::cuCtxSynchronize() },
            "cuCtxSynchronize",
        ))
    }

    unsafe fn go(&self, f: Fun, grid: u32, block: u32, args: &mut [*mut c_void]) -> Result<()> {
        ge(gpu::launch(
            f,
            (grid, 1, 1),
            (block, 1, 1),
            0,
            &self.stream,
            args,
        ))
    }

    fn csr(&self, m: &DevCsr, x: u64, y: u64, res: u64) -> Result<()> {
        let ctl = self.ctl.ptr;
        let rows = m.rows as u32;
        unsafe {
            self.go(
                self.f.csr,
                rows.div_ceil(128),
                128,
                tang_moe::args![ctl, m.ptr.ptr, m.idx.ptr, m.val.ptr, x, y, rows, res],
            )
        }
    }

    fn enqueue_embed(&self) -> Result<()> {
        let (ctl, h) = (self.ctl.ptr, self.shape.hidden as u32);
        unsafe {
            self.go(
                self.f.embed,
                h.div_ceil(256),
                256,
                tang_moe::args![
                    ctl,
                    self.tokmap.ptr,
                    self.ntok,
                    self.rows.ptr,
                    h,
                    self.s.x.ptr
                ],
            )
        }
    }

    /// Layer `l` on the residual `s.x`, with `st`'s state for that layer.
    fn enqueue_layer(&self, l: usize, conv: u64, rec: u64) -> Result<()> {
        let sh = &self.shape;
        let w = &self.layers[l];
        let s = &self.s;
        let ctl = self.ctl.ptr;
        let (h, eps) = (sh.hidden as u32, self.eps);
        let (cd, hk, hv) = (sh.conv_dim() as u32, sh.k_heads as u32, sh.v_heads as u32);
        let qscale = 1.0f32 / (sh.d_state as f32).sqrt();
        let l2eps = 1e-6f32;
        let inter = sh.inter as u32;
        unsafe {
            self.go(
                self.f.rmsnorm,
                1,
                1024,
                tang_moe::args![ctl, s.x.ptr, w.in_norm.ptr, s.h.ptr, h, eps],
            )?;
            self.csr(&w.proj, s.h.ptr, s.p.ptr, 0)?;
            self.go(
                self.f.conv,
                cd.div_ceil(256),
                256,
                tang_moe::args![ctl, s.p.ptr, w.conv.ptr, conv, s.c.ptr, cd],
            )?;
            self.go(
                self.f.l2,
                2 * hk,
                128,
                tang_moe::args![ctl, s.c.ptr, hk, l2eps, qscale],
            )?;
            self.go(
                self.f.gdn,
                hv,
                128,
                tang_moe::args![
                    ctl,
                    s.c.ptr,
                    s.p.ptr,
                    rec,
                    s.o.ptr,
                    w.dt_bias.ptr,
                    w.a_neg.ptr,
                    w.gnorm.ptr,
                    hk,
                    hv,
                    cd,
                    eps
                ],
            )?;
            self.csr(&w.out, s.o.ptr, 0, s.x.ptr)?;
            self.go(
                self.f.rmsnorm,
                1,
                1024,
                tang_moe::args![ctl, s.x.ptr, w.post_norm.ptr, s.h.ptr, h, eps],
            )?;
            self.csr(&w.gu, s.h.ptr, s.gu.ptr, 0)?;
            self.go(
                self.f.act,
                inter.div_ceil(256),
                256,
                tang_moe::args![ctl, s.gu.ptr, s.act.ptr, inter],
            )?;
            self.csr(&w.down, s.act.ptr, 0, s.x.ptr)?;
        }
        Ok(())
    }

    fn enqueue_pick(&self) -> Result<()> {
        let ctl = self.ctl.ptr;
        let (q, lc) = (QCAP as u32, LOGCAP as u32);
        unsafe {
            self.go(
                self.f.pick,
                1,
                1,
                tang_moe::args![
                    ctl,
                    self.s.x.ptr,
                    self.fam.ptr,
                    self.nfam,
                    self.nfl,
                    self.printable.ptr,
                    self.nprint,
                    self.v0,
                    self.code.ptr,
                    self.ncode,
                    self.idc.ptr,
                    self.nidc,
                    q,
                    lc
                ],
            )
        }
    }

    /// Slot `slot`'s state pointers per layer.
    fn ptrs(&mut self, slot: usize) -> Result<Vec<(u64, u64)>> {
        let st = self.slot(slot)?;
        Ok(st
            .conv
            .iter()
            .zip(&st.rec)
            .map(|(c, r)| (c.ptr, r.ptr))
            .collect())
    }

    fn enqueue_window(&self, ptrs: &[(u64, u64)]) -> Result<()> {
        for _ in 0..WINDOW {
            self.enqueue_embed()?;
            for (l, &(c, r)) in ptrs.iter().enumerate() {
                self.enqueue_layer(l, c, r)?;
            }
            self.enqueue_pick()?;
        }
        Ok(())
    }

    fn write_ctl(&self, words: &[u32]) -> Result<()> {
        ge(self.ctl.write_async(0, words, &self.stream))
    }

    fn read_ctl(&self) -> Result<Vec<u32>> {
        ge(self.stream.sync())?;
        ge(self.ctl.read::<u32>(0, CTL))
    }

    fn check(&self, c: &[u32]) -> Result<()> {
        match c[C_ERR] {
            0 => Ok(()),
            1 => bail!("chip window: token {} has no embedding row", c[C_ARG]),
            2 => bail!(
                "chip window: the circuit predicts no token (family {}, payload {})",
                c[C_ARG] & 0xff,
                c[C_ARG] >> 8
            ),
            3 => bail!("chip window: runaway op ({LOGCAP} tokens)"),
            4 => bail!("chip window: a chip token without a successor"),
            e => bail!("chip window: error {e}"),
        }
    }

    /// One op from slot `slot`'s state: `forced` (the tool-call tokens) are fed until the
    /// circuit owns a position, then its tokens until it hands the position back with the
    /// queue empty ([`super::chip::Driver::op`]), in windows of [`WINDOW`] tokens.
    pub fn op(&mut self, slot: usize, forced: &[u32]) -> Result<ChipOut> {
        ensure!(
            !forced.is_empty() && forced.len() <= QCAP,
            "{} forced tokens",
            forced.len()
        );
        let ptrs = self.ptrs(slot)?;
        let mut words = vec![0u32; CTL + forced.len() - 1];
        words[C_CUR] = forced[0];
        words[C_QLEN] = (forced.len() - 1) as u32;
        words[CTL..].copy_from_slice(&forced[1..]);
        self.write_ctl(&words)?;
        if self.use_graphs && !self.graphs.contains_key(&slot) {
            let g = {
                let me = &*self;
                ge(Graph::capture(&me.stream, |_| {
                    me.enqueue_window(&ptrs)
                        .map_err(|e| gpu::Error(format!("{e}")))
                }))?
            };
            self.graphs.insert(slot, g);
        }
        loop {
            match self.graphs.get(&slot).filter(|_| self.use_graphs) {
                Some(g) => ge(g.launch(&self.stream))?,
                None => self.enqueue_window(&ptrs)?,
            }
            let c = self.read_ctl()?;
            self.check(&c)?;
            if c[C_DONE] != 0 {
                let n = c[C_NFED] as usize;
                let log = ge(self.ctl.read::<u32>(4 * (CTL + QCAP), n))?;
                let src = ge(self.ctl.read::<u32>(4 * (CTL + QCAP + LOGCAP), n))?;
                return Ok(ChipOut {
                    fed: log,
                    from_chip: src.iter().map(|&s| s != 0).collect(),
                    v0: self.v0,
                });
            }
        }
    }

    // ---- verification hooks (one token, one layer at a time; `chip_gpu` tests)

    /// Fill every scratch buffer with NaN (outputs nothing writes then read NaN).
    pub fn poison(&self) -> Result<()> {
        let s = &self.s;
        for b in [&s.x, &s.h, &s.p, &s.c, &s.o, &s.gu, &s.act] {
            let n = (b.len / 4) as u32;
            let nan = f32::NAN;
            unsafe {
                self.go(
                    self.f.fill,
                    n.div_ceil(256),
                    256,
                    tang_moe::args![b.ptr, n, nan],
                )?;
            }
        }
        Ok(())
    }

    /// Start a single-token step: the control block for token `tok` with an empty queue.
    pub fn begin_token(&self, tok: u32) -> Result<()> {
        let mut w = vec![0u32; CTL];
        w[C_CUR] = tok;
        self.write_ctl(&w)
    }

    /// The embedding of the current token into the residual.
    pub fn run_embed(&self) -> Result<()> {
        self.enqueue_embed()
    }

    /// Layer `l` on the residual with slot `slot`'s state.
    pub fn run_layer(&mut self, slot: usize, l: usize) -> Result<()> {
        let p = self.ptrs(slot)?[l];
        self.enqueue_layer(l, p.0, p.1)
    }

    /// The decoder and the pick on the residual: `(circuit's token or None, error)`.
    pub fn run_pick(&self) -> Result<Option<u32>> {
        self.enqueue_pick()?;
        let c = self.read_ctl()?;
        self.check(&c)?;
        Ok((c[C_DONE] == 0).then_some(c[C_CUR]))
    }

    /// The residual.
    pub fn x(&self) -> Result<Vec<f32>> {
        ge(self.stream.sync())?;
        ge(self.s.x.read::<f32>(0, self.shape.hidden))
    }

    /// Set the residual.
    pub fn set_x(&self, x: &[f32]) -> Result<()> {
        ge(self.s.x.write_async(0, x, &self.stream))?;
        ge(self.stream.sync())
    }

    pub fn sync(&self) -> Result<()> {
        ge(self.stream.sync())
    }

    /// The window's stream.
    pub fn stream(&self) -> &Stream {
        &self.stream
    }
}

/// A [`ChipCodec`] from fixed token sequences: `encode` looks the op up by kind and name
/// (`(kind, name) → call tokens`), `decode` reads the printed answer between the response
/// markers. For the bundled trace (Qwen3.5 text ids); a real one is built from the host
/// tokenizer.
pub struct TableCodec {
    pub calls: Vec<((String, Vec<u8>), Vec<u32>)>,
    /// The tool response's open and close tokens, and the separator (newline) token.
    pub open: u32,
    pub close: u32,
    pub sep: u32,
    /// Token → bytes for file names in a listing.
    pub vocab: HashMap<u32, Vec<u8>>,
}

impl TableCodec {
    fn key(op: &StoreOp) -> (String, Vec<u8>) {
        match op {
            StoreOp::Put { name, .. } => ("put".into(), name.clone()),
            StoreOp::Get { name } => ("get".into(), name.clone()),
            StoreOp::Delete { name } => ("delete".into(), name.clone()),
            StoreOp::List => ("list".into(), vec![]),
        }
    }

    /// The printed body between `open` and `close`, separators trimmed.
    fn body(&self, p: &[u32]) -> Result<Vec<u32>> {
        let a = p
            .iter()
            .position(|&t| t == self.open)
            .context("no tool response")?;
        let b = p[a..]
            .iter()
            .position(|&t| t == self.close)
            .context("unterminated tool response")?
            + a;
        let mut body = &p[a + 1..b];
        while body.first() == Some(&self.sep) {
            body = &body[1..];
        }
        while body.last() == Some(&self.sep) {
            body = &body[..body.len() - 1];
        }
        Ok(body.to_vec())
    }
}

impl ChipCodec for TableCodec {
    fn encode(&self, op: &StoreOp) -> Result<Vec<u32>> {
        let k = Self::key(op);
        self.calls
            .iter()
            .find(|(c, _)| *c == k)
            .map(|(_, t)| t.clone())
            .with_context(|| format!("no call tokens for {op:?}"))
    }

    fn decode(&self, op: &StoreOp, out: &ChipOut) -> Result<OpOut> {
        let body = self.body(&out.printed())?;
        Ok(match op {
            StoreOp::Put { .. } => OpOut::Put,
            StoreOp::Get { .. } => OpOut::Got(Some(body)),
            StoreOp::Delete { .. } => OpOut::Deleted(true),
            StoreOp::List => {
                let mut files = vec![];
                for line in body.split(|&t| t == self.sep).filter(|l| !l.is_empty()) {
                    let mut name = vec![];
                    for t in line {
                        name.extend(self.vocab.get(t).with_context(|| format!("token {t}"))?);
                    }
                    files.push((name, 0));
                }
                OpOut::Listed(files)
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::chip::{Driver, Numerics, RetargetFixture};
    use super::*;
    use std::time::Instant;

    fn gpu() -> Option<Gpu> {
        let g = Gpu::for_test();
        if g.is_none() {
            eprintln!("================ chip_gpu: NO CUDA DEVICE, SKIPPED ================");
        }
        g
    }

    fn bits_eq(a: &[f32], b: &[f32]) -> usize {
        a.iter()
            .zip(b)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count()
    }

    /// put/get/list from zero state through the device chip stack against chip.rs on the same
    /// emitted weights. (1) Shadow + poison, one token and one layer at a time: before each
    /// token the scratch is NaN; after each layer the residual and that layer's conv history
    /// and recurrent state must equal the CPU layer step's bitwise (the CPU starts every layer
    /// from the device's input, so a mismatch is attributed to its layer and resynced). The
    /// decoded token must equal the CPU decoder's, and the fed sequence the trace's. (2) The
    /// production path (graph windows of 8) on a fresh slot: every fed token, and after each op
    /// every layer's full state bitwise equal to the CPU driver's and the store heads to the
    /// trace's. (3) Timing and memory.
    #[test]
    fn chip_gpu_put_get_list_matches_cpu() {
        let Some(g) = gpu() else { return };
        let t0 = Instant::now();
        let fxt = RetargetFixture::new().unwrap();
        let (model, dec) = (&fxt.model, &fxt.dec);
        let pack = ChipPack::new(model, dec, &fxt.row_tokens).unwrap();
        let mut chip = ChipGpu::new(&g, &pack).unwrap();
        let sh = model.shape;
        let nl = model.layers.len();
        eprintln!(
            "emitted + uploaded in {:?}: weights {:.2} MB ({} nnz), rows {:.2} MB, slot state {:.2} MB",
            t0.elapsed(),
            chip.weight_bytes as f64 / 1e6,
            pack.layers
                .iter()
                .map(|l| l.proj.nnz() + l.out.nnz() + l.gu.nnz() + l.down.nnz())
                .sum::<usize>(),
            chip.row_bytes as f64 / 1e6,
            chip.slot_bytes() as f64 / 1e6
        );

        // (1) shadow + poison
        let mut cpu = model.state();
        let (mut tokens, mut mism) = (0usize, vec![]);
        let mut snaps = vec![];
        for (label, want, nf) in &fxt.ops {
            let mut q: std::collections::VecDeque<u32> = want[1..*nf].iter().copied().collect();
            let mut cur = want[0];
            let mut fed = vec![];
            loop {
                chip.poison().unwrap();
                chip.begin_token(cur).unwrap();
                chip.run_embed().unwrap();
                let mut x = model.embed(cur as usize);
                let gx = chip.x().unwrap();
                assert_eq!(bits_eq(&gx, &x), 0, "{label} tok {}: embedding", fed.len());
                for l in 0..nl {
                    chip.run_layer(1, l).unwrap();
                    model.layers[l].step(
                        &sh,
                        model.eps,
                        Numerics::Bf16,
                        &mut x,
                        &mut cpu.conv[l],
                        &mut cpu.rec[l],
                    );
                    let gx = chip.x().unwrap();
                    let (gc, gr) = chip.layer_state(1, l).unwrap();
                    let (dx, dc, dr) = (
                        bits_eq(&gx, &x),
                        bits_eq(&gc, &cpu.conv[l]),
                        bits_eq(&gr, &cpu.rec[l]),
                    );
                    let nan = gx.iter().filter(|v| v.is_nan()).count();
                    if dx + dc + dr + nan > 0 {
                        mism.push(format!(
                            "{label} tok {} ({cur}) layer {l}: residual {dx} ({nan} NaN), conv {dc}, rec {dr} floats differ",
                            fed.len()
                        ));
                        x = gx;
                        cpu.conv[l] = gc;
                        cpu.rec[l] = gr;
                    }
                }
                fed.push(cur);
                tokens += 1;
                let ct = pack.dec.token(&x).unwrap();
                let gt = chip.run_pick().unwrap();
                let next = super::super::chip::pick_next(pack.dec.v0, cur, ct, &mut q).unwrap();
                // run_pick sees an empty queue: it agrees with the CPU whenever the circuit
                // owns the next position
                if ct.is_some() || cur >= pack.dec.v0 {
                    assert_eq!(gt, ct, "{label} tok {}: decoded token", fed.len());
                } else {
                    assert_eq!(gt, None, "{label} tok {}: decoded token", fed.len());
                }
                match next {
                    Some((t, _)) => cur = t,
                    None => break,
                }
            }
            assert_eq!(&fed, want, "{label}: fed tokens (shadow)");
            snaps.push(cpu.clone());
        }
        eprintln!(
            "shadow: {tokens} tokens × {nl} layers, {} mismatching layer steps",
            mism.len()
        );
        for m in mism.iter().take(20) {
            eprintln!("  {m}");
        }
        assert!(mism.is_empty(), "first divergence: {}", mism[0]);

        // (2) production windows on a fresh slot, against the CPU driver
        let mut drv = Driver {
            model,
            dec,
            num: Numerics::Bf16,
            state: model.state(),
        };
        for (i, (label, want, nf)) in fxt.ops.iter().enumerate() {
            let t = Instant::now();
            let out = chip.op(2, &want[..*nf]).unwrap();
            let dt = t.elapsed();
            assert_eq!(&out.fed, want, "{label}: fed tokens (windows)");
            assert_eq!(drv.op(&want[..*nf]).unwrap(), *want);
            let gs = chip.state(2).unwrap();
            for l in 0..nl {
                assert_eq!(
                    bits_eq(&gs.conv[l], &drv.state.conv[l]),
                    0,
                    "{label} l{l} conv"
                );
                assert_eq!(
                    bits_eq(&gs.rec[l], &drv.state.rec[l]),
                    0,
                    "{label} l{l} rec"
                );
                assert_eq!(
                    bits_eq(&gs.rec[l], &snaps[i].rec[l]),
                    0,
                    "{label} l{l} vs shadow"
                );
            }
            for (l, vh, w) in fxt.store_states(label) {
                assert_eq!(
                    bits_eq(drv.state.head(&sh, l, vh), &w),
                    0,
                    "{label} trace l{l} v{vh}"
                );
            }
            let printed = out.printed();
            eprintln!(
                "{label}: {} tokens fed, {} printed, states bitwise equal (first run {:?})",
                out.fed.len(),
                printed.len(),
                dt
            );
        }

        // (3) timing: each op from the state before it, repeated on scratch slots
        let mut before = model.state();
        for (label, want, nf) in &fxt.ops {
            let reps = 20;
            for (mode, graphs) in [("graph", true), ("eager", false)] {
                chip.use_graphs = graphs;
                let mut tot = 0f64;
                for _ in 0..reps {
                    chip.set_state(3, &before).unwrap();
                    let t = Instant::now();
                    let out = chip.op(3, &want[..*nf]).unwrap();
                    tot += t.elapsed().as_secs_f64();
                    assert_eq!(&out.fed, want);
                }
                let per_op = tot / reps as f64;
                let per_tok = per_op / want.len() as f64;
                eprintln!(
                    "timing {label} ({mode}): {:.0} µs/op, {} tokens, {:.1} µs/token, {:.2} µs/token/layer",
                    per_op * 1e6,
                    want.len(),
                    per_tok * 1e6,
                    per_tok * 1e6 / nl as f64
                );
            }
            chip.use_graphs = true;
            let mut d = Driver {
                model,
                dec,
                num: Numerics::Bf16,
                state: before.clone(),
            };
            d.op(&want[..*nf]).unwrap();
            before = d.state;
        }
    }

    /// The chip window inside a Coder engine (`TANG_LLM_TEST_GGUF`: any `qwen3next` GGUF; its
    /// base layers only host the window). On a slot with a prompt: put/get/list through
    /// `chip_op` give the trace's tokens, the slot's tokens are unchanged, and its greedy
    /// continuation equals a twin slot's that never ran the chip (base layers, KV and GDN
    /// state untouched). `store_op_at` with a table codec returns the ops' results.
    #[test]
    fn chip_gpu_in_coder() {
        let Ok(path) = std::env::var("TANG_LLM_TEST_GGUF") else {
            assert!(
                std::env::var("TANG_REQUIRE_CUDA").is_err(),
                "TANG_REQUIRE_CUDA=1 needs TANG_LLM_TEST_GGUF"
            );
            eprintln!("================ chip_gpu_in_coder: no TANG_LLM_TEST_GGUF, SKIPPED ================");
            return;
        };
        let mut e = super::super::coder::Coder::load(
            std::path::Path::new(&path),
            super::super::coder::CoderOpts {
                max_ctx: 1024,
                mtp: None,
            },
        )
        .unwrap();
        let fxt = RetargetFixture::new().unwrap();
        let pack = ChipPack::new(&fxt.model, &fxt.dec, &fxt.row_tokens).unwrap();
        let call = |i: usize| fxt.ops[i].1[..fxt.ops[i].2].to_vec();
        let name = b"a.txt".to_vec();
        let codec = TableCodec {
            calls: vec![
                (("put".into(), name.clone()), call(0)),
                (("get".into(), name.clone()), call(1)),
                (("list".into(), vec![]), call(2)),
            ],
            open: 248066,
            close: 248067,
            sep: 198,
            vocab: [(64, b"a".to_vec()), (3778, b".txt".to_vec())].into(),
        };
        e.load_chip(&pack, Some(Box::new(codec))).unwrap();
        let (a, b, c) = (
            e.alloc_slot().unwrap(),
            e.alloc_slot().unwrap(),
            e.alloc_slot().unwrap(),
        );
        let prompt: Vec<u32> = (1000..1040).collect();
        let na = e.prefill_slot(a, &prompt).unwrap();
        let nb = e.prefill_slot(b, &prompt).unwrap();
        assert_eq!(na, nb);
        let toks = e.slot_tokens(a).to_vec();
        for (label, want, nf) in &fxt.ops {
            let t = Instant::now();
            let out = e.chip_op(a, &want[..*nf]).unwrap();
            assert_eq!(&out.fed, want, "{label}");
            eprintln!(
                "coder chip_op {label}: {} tokens in {:?}, printed {:?}",
                out.fed.len(),
                t.elapsed(),
                out.printed()
            );
        }
        assert_eq!(
            e.slot_tokens(a),
            &toks[..],
            "chip_op changed the slot's tokens"
        );
        let (mut xa, mut xb) = (na, nb);
        for i in 0..8 {
            xa = e.prefill_slot(a, &[xa]).unwrap();
            xb = e.prefill_slot(b, &[xb]).unwrap();
            assert_eq!(xa, xb, "continuation step {i} differs after chip ops");
        }
        let ops = [
            StoreOp::Put {
                name: name.clone(),
                ids: vec![],
            },
            StoreOp::Get { name: name.clone() },
            StoreOp::List,
        ];
        let got: Vec<OpOut> = ops.iter().map(|op| e.store_op_at(c, op).unwrap()).collect();
        assert_eq!(
            got,
            vec![
                OpOut::Put,
                OpOut::Got(Some(vec![8029, 13053, 20956, 9197, 30183])),
                OpOut::Listed(vec![(name.clone(), 0)]),
            ]
        );
        eprintln!("coder store_op_at through the chip window: {got:?}");
    }
}
