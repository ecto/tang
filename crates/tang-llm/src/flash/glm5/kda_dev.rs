//! The KDA layers resident on the GPU: W4's conv + delta-rule core ([`DevKda`]) plus the
//! projections, gates and gated output norm as device kernels — the engine's wiring of
//! everything W4 built.
//!
//! Before this module the engine's `kda_gpu` (engine.rs) staged every weight per window
//! (`DevBuf::from_slice` of `w.q`/`k`/`v`/… on the host) and ran the projections as the
//! CPU f64 `matmul` — ~8 GFLOP of scalar host work per 64-token window per layer, 34
//! layers: the 22 s/window profile. The fix is the DevMoe shape (W5):
//!
//! - weights uploaded ONCE at engine load, in their GGUF storage format (Q8_0 stays Q8_0;
//!   [`super::moe::gpu::upload`] + the `G5W` view, dequantised element-wise in the
//!   kernel) — 67 MB/layer instead of 268 MB f32;
//! - one [`DevLayer`] per KDA layer holding the conv history + recurrent state + scratch
//!   between windows on its own persistent stream;
//! - per window the host traffic is `x` in, the core I/O W4's launchers define, and `y`
//!   out — no weight uploads, no per-window allocations.
//!
//! Kernels (K5_SRC, appended to [`super::moe::SRC`] so W5's dequant table is reused):
//! `k5_gemv` (one thread per output, G5W row dot, destination base + stride so the q/k/v
//! projections write the stacked q|k|v layout directly), `k5_gate_finish` (dt_bias + the
//! safe-gate/softplus form), `k5_beta` (the β GEMV + sigmoid), `k5_gated_norm` (the host
//! `rms_norm_mul`'s f64 mean-of-squares exactly, times `o_norm` and `sigmoid(g2)`).
//!
//! State ownership: the device holds the live state. The engine's host
//! `(ConvState, KdaState)` stays authoritative only across a CPU-fallback window
//! (`TANG_GLM5_GPU_KDA=0`, a layer past [`Self::max_t`]): [`Self::read_back`] pulls the
//! device state before the CPU runs, and [`Self::seed`] pushes it back before the GPU
//! resumes (the engine tracks this with a per-layer `host_fresh` flag). Decode also
//! updates the host conv copy as a side effect (W4's design); the recurrent state does
//! not. `read_back`/`seed` are cheap (288 KiB + 4 MiB) and only run on the fallback path.
//!
//! Later (FAST, after the gates): keep qkv/g/beta on the device across the core call
//! (extends DevKda's launchers to take device pointers — kills the per-chunk host
//! round-trip); tensor-core GEMV for prefill; CUDA graphs.
#![allow(clippy::too_many_arguments)]
use super::kda::{ConvState, KdaConfig, KdaState};

/// warps per `k5_gemv8` block (the kernel's K5_WARPS; one row a warp).
const K5_WARPS: usize = 4;
use anyhow::{ensure, Context, Result};
use tang_moe::gpu::{DevBuf, Gpu, Stream};
use tang_moe::{args, gpu::launch};

use super::kda_cuda::cuda::{DevKda, Kernels};

/// The KDA projection kernels, appended to [`super::moe::SRC`] (the G5W dequant table is
/// defined there). NAIVE ON PURPOSE (GATES.md): one thread per output value, plain loops,
/// f32 accumulate. All layouts match the host 1:1 so parity is a value compare:
///
/// - `k5_gemv(W, E, X, Y, YSTRIDE, T, ROWS, COLS)`: `y[t*ystride + r] = dot(W row r of
///   matrix e, x[t])` — the host `matmul`'s f32 form (the host accumulates f64; parity
///   comes from the engine test's 1e-3 bar). `ystride = ROWS` is the plain layout; the
///   q/k/v projections write the stacked q|k|v layout directly (`ystride = 3·d_inner`,
///   base offsets `q=0, k=d_inner, v=2·d_inner` — the host's interleave loop, gone).
///   Grid `ceil(T*ROWS/128)`, block 128.
/// - `k5_gate_finish(G1, DT, RATE, NH, HD, LB, T)`: in place on the forget gate's raw
///   GEMV output `g1[t][nh*hd]`: `+ dt_bias[chan]`, then `lb·σ(rate[h]·pre)` when
///   `lb >= 0` (GLM-5.3-Flash passes −5.0), else the Qwen3-Next softplus form — the
///   caller passes a NaN for `None` (safe: a real lower bound is −5, never NaN).
/// - `k5_beta(BP, X, BETA, T, HID, NH)`: `beta[t][h] = σ(dot(b_proj row h, x[t]))`.
/// - `k5_gated_norm(O, ON, G2, Y, NH, HD, EPS, T)`: the host's `rms_norm_mul` (f64 sum
///   of squares over the head, the exact form) times `o_norm[j]`, then
///   `y[t][c] = that · σ(g2[t][c])`, where `g2 = g_b(g_a(x))` was produced by two
///   ordinary `k5_gemv` launches into the `g2` scratch (`g_a`: `[t][hd]`, then `g_b`
///   in place on a separate buffer, so no read/write race).


/// Everything of one KDA layer on one GPU, built once at engine load.
pub struct DevLayer {
    /// The conv + delta-rule core (W4's `DevKda`): state lives on device between windows.
    core: DevKda,
    /// Projection weights in GGUF storage format: q/k/v `[d_inner][hidden]`,
    /// o_proj `[hidden][d_inner]`.
    wq: super::moe::gpu::DevW,
    wk: super::moe::gpu::DevW,
    wv: super::moe::gpu::DevW,
    wo: super::moe::gpu::DevW,
    /// The gates' GEMV weights: f_a `[hd][hidden]`, f_b `[d_inner][hd]`,
    /// b_proj `[nh][hidden]`, g_a `[hd][hidden]`, g_b `[d_inner][hd]`.
    fa: super::moe::gpu::DevW,
    fb: super::moe::gpu::DevW,
    bproj: super::moe::gpu::DevW,
    ga: super::moe::gpu::DevW,
    gb: super::moe::gpu::DevW,
    /// Small exact vectors: dt_bias `[d_inner]`, decay_rate `[nh]` (= `−ssm_a`),
    /// o_norm `[hd]`.
    dt: DevBuf,
    rate: DevBuf,
    on: DevBuf,
    /// Per-window scratch (device, persistent): `x` `[t][hidden]`, the stacked qkv
    /// `[t][3·d_inner]`, the forget gate's raw GEMV `g1` `[t][d_inner]`, the `f_b` GEMV
    /// output `g` (separate from `g1`: the GEMV writes a buffer it does not read),
    /// `beta` `[t][nh]`, the core output `o` `[t][d_inner]`, `g2` `[t][d_inner]`, the
    /// gated-norm output, and the layer output `y` `[t][hidden]`.
    xb: DevBuf,
    qkvb: DevBuf,
    g1b: DevBuf,
    gb_out: DevBuf,
    bb: DevBuf,
    ob: DevBuf,
    g2a: DevBuf,
    g2b: DevBuf,
    yb: DevBuf,
    outb: DevBuf,
    /// int8 activations for `k5_gemv8` (codes + scales; see the xq8 field).
    xq8: DevBuf,
    /// The layer's persistent stream (reads host-staged buffers, hence blocking).
    s: Stream,
    kers: Kernels,
    /// The projection kernels (`K5_SRC` compiled per arch; the G5W dequant table comes
    /// from the MoE source it is appended to).
    gv: super::moe::gpu::Kernels5,
    cfg: KdaConfig,
    /// The layer's conv weights on the host (for the <64-token CPU tail path).
    conv_w: Vec<f32>,
    hidden: usize,
    d_inner: usize,
    nh: usize,
    hd: usize,
    max_t: usize,
    /// `TANG_GLM5_KDA_DEBUG`: when set by the engine for this window, [`Self::window`]
    /// poisons its scratch with NaN first and leaves every intermediate in [`Self::taps`]
    /// for the shadow check ([`super::kda_debug`]).
    pub debug: bool,
    pub taps: Option<super::kda_debug::KdaTaps>,
    /// `TANG_GLM5_GEMV8=1` at construction: the q/k/v projections on k5_gemv8 (a field,
    /// not an env read per window, so a test can A/B one layer in-process).
    pub gemv8: bool,
}

/// The G5W descriptor of one `KdaLayerW` matrix, from the GGUF's raw tensor (the weights
/// stay quantised on the device; the kernel dequantises element-wise). `rows × cols` in
/// HF orientation (`[out][in]`), which for GGUF ne dims `[cols, rows]` is
/// `rows = n_rows()`, `cols = row_len()`.
fn grid(n: usize) -> (u32, u32, u32) {
    (n.div_ceil(128) as u32, 1, 1)
}

fn dev_w(g: &crate::gguf::Gguf, name: &str) -> Result<super::moe::gpu::DevW> {
    let t = g.info(name).with_context(|| format!("k5 {name}"))?;
    super::moe::gpu::upload(&super::moe::W::Gguf(g, t), 1, t.n_rows(), t.row_len())
}

impl DevLayer {
    /// Build from the GGUF directly (the device reads its own bytes; the CPU loader's
    /// dequantised copy stays for the fallback path and the parity tests).
    /// `conv`/`state` seed the device (the engine passes the zeros it would give the CPU
    /// path at load). `g` must already be bound to this device.
    #[allow(clippy::too_many_arguments)]
    pub fn new_gguf(
        g: &Gpu,
        gguf: &crate::gguf::Gguf,
        l: usize,
        cfg: &KdaConfig,
        hidden: usize,
        conv: &ConvState,
        state: &KdaState,
        max_t: usize,
    ) -> Result<Self> {
        let (nh, hd) = (cfg.heads, cfg.head_dim);
        let d_inner = nh * hd;
        let t = |s: &str| format!("blk.{l}.{s}");
        let gerr = |e: tang_moe::gpu::Error| anyhow::anyhow!("k5 upload: {e}");
        let kers = super::kda_cuda::cuda::load(g, cfg)?;
        let gv = super::moe::gpu::Kernels5::new(g)?;
        let upf = |name: &str| -> Result<DevBuf> {
            let ti = gguf.info(&t(name))?;
            ensure!(
                ti.ty == crate::gguf::GgmlType::F32,
                "{}: want F32 for the small vectors",
                t(name)
            );
            let v = gguf.tensor_f32(&t(name))?;
            DevBuf::from_slice(&v).map_err(gerr)
        };
        let conv_w = {
            // the converter splits HF's fused qkv conv per q|k|v; the CPU loader stacks
            // them in q,k,v order — the core kernel indexes the same stack
            let mut c = Vec::with_capacity(3 * d_inner * cfg.d_conv);
            for n in ["ssm_conv1d_q.weight", "ssm_conv1d_k.weight", "ssm_conv1d_v.weight"] {
                c.extend_from_slice(&gguf.tensor_f32(&t(n))?);
            }
            c
        };
        let rate = {
            // GGUF `ssm_a` = `-exp(A_log)`; the gate's `rate` = `exp(A_log)` = `−ssm_a`
            let r = gguf.tensor_f32(&t("ssm_a"))?;
            let neg: Vec<f32> = r.iter().map(|&a| -a).collect();
            DevBuf::from_slice(&neg).map_err(gerr)
        };
        let slots = 1;
        Ok(Self {
            core: DevKda::new(g, &kers, cfg, &conv_w, conv, state, slots)?,
            wq: dev_w(gguf, &t("attn_q.weight"))?,
            wk: dev_w(gguf, &t("attn_k.weight"))?,
            wv: dev_w(gguf, &t("attn_v.weight"))?,
            wo: dev_w(gguf, &t("attn_output.weight"))?,
            fa: dev_w(gguf, &t("ssm_f_a.weight"))?,
            fb: dev_w(gguf, &t("ssm_f_b.weight"))?,
            bproj: dev_w(gguf, &t("ssm_beta.weight"))?,
            ga: dev_w(gguf, &t("ssm_g_a.weight"))?,
            gb: dev_w(gguf, &t("ssm_g_b.weight"))?,
            dt: upf("ssm_dt.bias")?,
            rate: rate?,
            on: upf("ssm_norm.weight")?,
            xb: DevBuf::zeroed_elems::<f32>(max_t * hidden).map_err(gerr)?,
            qkvb: DevBuf::zeroed_elems::<f32>(max_t * 3 * d_inner).map_err(gerr)?,
            g1b: DevBuf::zeroed_elems::<f32>(max_t * hd).map_err(gerr)?,
            gb_out: DevBuf::zeroed_elems::<f32>(max_t * d_inner).map_err(gerr)?,
            bb: DevBuf::zeroed_elems::<f32>(max_t * nh).map_err(gerr)?,
            ob: DevBuf::zeroed_elems::<f32>(max_t * d_inner).map_err(gerr)?,
            g2a: DevBuf::zeroed_elems::<f32>(max_t * hd).map_err(gerr)?,
            g2b: DevBuf::zeroed_elems::<f32>(max_t * d_inner).map_err(gerr)?,
            yb: DevBuf::zeroed_elems::<f32>(max_t * d_inner).map_err(gerr)?,
            outb: DevBuf::zeroed_elems::<f32>(max_t * hidden).map_err(gerr)?,
            // TANG_GLM5_GEMV8: int8 codes [t][hidden] + f32 per-32 scales [t][hidden/32]
            // (k5_q8's layout; the codes+scales in one buffer, scales after T*hidden codes)
            xq8: DevBuf::zeroed_elems::<u8>(max_t * hidden + max_t * hidden.div_ceil(32) * 4)
                .map_err(gerr)?,
            s: Stream::blocking().map_err(gerr)?,
            kers,
            gv,
            cfg: *cfg,
            conv_w,
            hidden,
            d_inner,
            nh,
            hd,
            max_t,
            debug: false,
            taps: None,
            gemv8: std::env::var("TANG_GLM5_GEMV8").is_ok_and(|v| v == "1"),
        })
    }

    /// Pull the device's conv + recurrent state into the host copies (before a CPU
    /// fallback window on this layer; cheap, 288 KiB + 4 MiB).
    pub fn read_back(&self, conv: &mut ConvState, state: &mut KdaState) -> Result<()> {
        self.s.sync()?;
        self.core.read_conv(conv)?;
        self.core.read_state(state)?;
        Ok(())
    }

    /// Push the host copies back onto the device (after a CPU fallback window; the device
    /// takes over from the host's values).
    pub fn seed(&self, conv: &ConvState, state: &KdaState) -> Result<()> {
        self.core.h0.write(0, &conv.0)?;
        self.core.st.write(0, &state.0)?;
        Ok(())
    }

    /// Run one window (`x` `[t][hidden]`, `t ≤ max_t`) through the layer. The host's
    /// `conv` is updated in place (W4's launchers keep the host conv copy fresh); the
    /// recurrent state lives only on the device except across the CPU tail, which reads
    /// it back and seeds it forward.
    pub fn window(&mut self, x: &[f32], t: usize, conv: &mut ConvState) -> Result<Vec<f32>> {
        ensure!(x.len() == t * self.hidden, "k5 layer input");
        ensure!(t <= self.max_t, "k5 window {t} > {max}", max = self.max_t);
        let (nh, hd, di, hidden) = (self.nh, self.hd, self.d_inner, self.hidden);
        let lb = match self.cfg.gate_lower_bound {
            Some(v) => v,
            None => f32::NAN, // the kernel's softplus-form sentinel
        };
        if std::env::var("TANG_GLM5_KDA_DUMP").is_ok_and(|v| v == "1") {
            // the layer input, matching kda::kda_layer's `kda cpu x` line: if x already
            // differs, the divergence is upstream of KDA (not a staging bug)
            eprintln!("kda gpu x[0..8]: {:?}", &x[..8.min(x.len())]);
        }
        self.xb.write(0, x)?;
        if self.debug && super::kda_debug::poison_on() {
            // NaN-fill every output scratch this window writes: a region no kernel
            // touches then reads NaN in the shadow check, not a plausible 0.0
            let nan = |n: usize| vec![f32::NAN; n];
            self.qkvb.write(0, &nan(t * 3 * di))?;
            self.g1b.write(0, &nan(t * hd))?;
            self.gb_out.write(0, &nan(t * di))?;
            self.bb.write(0, &nan(t * nh))?;
            self.g2a.write(0, &nan(t * hd))?;
            self.g2b.write(0, &nan(t * di))?;
            self.yb.write(0, &nan(t * di))?;
            self.outb.write(0, &nan(t * hidden))?;
            // the core's own scratch (conv output, core output) for the device paths
            let c = t.min(64);
            self.core.yb.write(0, &nan(c * 3 * di))?;
            self.core.ob.write(0, &nan(c * di))?;
        }
        let mut tp = self.debug.then(super::kda_debug::KdaTaps::default);
        // projections → the stacked q|k|v layout (base offsets q=0, k=di, v=2·di FLOATS).
        // The device pointer is a BYTE address: the offsets must be scaled by
        // size_of::<f32>(). Unscaled (the GLM-5.3 staging bug, Oct 2026) k landed di/4 and
        // v di/2 floats into the q block — q corrupted past 2048, half of k and all of v
        // never written (zeros on a fresh buffer → the core output was exactly 0).
        let qkv = self.qkvb.ptr;
        let f = std::mem::size_of::<f32>() as u64;
        // TANG_GLM5_GEMV8=1: the dequant-reuse GEMV (k5_gemv8, docs/models/glm5-speed-plan.md
        // step 1) for the Q8_0 projections. x is quantised once per window (k5_q8), then
        // every GEMV reads the int8 codes; one row per warp, the whole 8.7 KB row in
        // flight through shared memory. The gates' small GEMVs (f_a: 128 outs, beta: 64)
        // stay on k5_gemv — too few rows to amortise the staging.
        // gemv8 is Q8_0-only (its dp4a reads 34-byte Q8_0 blocks): any other weight type —
        // e.g. the f32 tiny fixture — must take k5_gemv, or the kernel reads f32 rows as
        // Q8_0 blocks (NaN logits on the first tiny run)
        let gemv8 = self.gemv8 && [&self.wq, &self.wk, &self.wv].iter().all(|w| w.w.ty == 8);
        unsafe {
            if gemv8 {
                launch(
                    self.gv.q8,
                    ((hidden / 32) as u32, t as u32, 1),
                    (32, 1, 1),
                    0,
                    &self.s,
                    args![self.xb.ptr, self.xq8.ptr, hidden as u32, self.max_t as u32],  // MAXT: the codes array's row cap (scales follow it), NOT t
                )?;
                let codes = self.xq8.ptr;
                let scales = codes + (self.max_t * hidden) as u64;
                for (w, off) in [(&self.wq, 0u64), (&self.wk, di as u64 * f), (&self.wv, 2 * di as u64 * f)] {
                    // dynamic shared memory: one staged row per warp, 16-byte granules
                    // (the kernel's `extern __shared__ uint4 dsm[]`); 0 here wrote past
                    // the allocation — CUDA_ERROR_ILLEGAL_ADDRESS on the first real run
                    let smem = (K5_WARPS * ((w.w.row_bytes as usize).div_ceil(16) + 1) * 16) as u32;
                    launch(
                        self.gv.gemv8,
                        (di.div_ceil(K5_WARPS) as u32, t as u32, 1),
                        (128, 1, 1),
                        smem,
                        &self.s,
                        args![w.w, 0u32, codes, scales, qkv + off, (3 * di) as u32,
                            t as u32, di as u32, hidden as u32],
                    )?;
                }
            } else {
                for (w, off) in [(&self.wq, 0u64), (&self.wk, di as u64 * f), (&self.wv, 2 * di as u64 * f)] {
                    launch(
                        self.gv.gemv,
                        grid(t * di),
                        (128, 1, 1),
                        0,
                        &self.s,
                        args![
                            w.w, 0u32, self.xb.ptr, qkv + off, (3 * di) as u32, t as u32,
                            di as u32, hidden as u32
                        ],
                    )?;
                }
            }
            // forget gate: f_a → g1, f_b: g1 → g (out of place), finish in place on g
            launch(
                self.gv.gemv,
                grid(t * hd),
                (128, 1, 1),
                0,
                &self.s,
                args![self.fa.w, 0u32, self.xb.ptr, self.g1b.ptr, hd as u32, t as u32,
                    hd as u32, hidden as u32],
            )?;
            launch(
                self.gv.gemv,
                grid(t * di),
                (128, 1, 1),
                0,
                &self.s,
                args![self.fb.w, 0u32, self.g1b.ptr, self.gb_out.ptr, di as u32, t as u32,
                    di as u32, hd as u32],
            )?;
            launch(
                self.gv.gate,
                grid(t * di),
                (128, 1, 1),
                0,
                &self.s,
                args![self.gb_out.ptr, self.dt.ptr, self.rate.ptr, nh as u32, hd as u32,
                    lb, t as u32],
            )?;
            // beta gate
            launch(
                self.gv.beta,
                grid(t * nh),
                (128, 1, 1),
                0,
                &self.s,
                args![self.bproj.w, self.xb.ptr, self.bb.ptr, t as u32, hidden as u32,
                    nh as u32],
            )?;
        }
        self.s.sync()?;
        // the core (W4's launchers stage their inputs from host): qkv, g, beta
        let qkv_h = self.qkvb.read::<f32>(0, t * 3 * di)?;
        let g_h = self.gb_out.read::<f32>(0, t * di)?;
        let beta_h = self.bb.read::<f32>(0, t * nh)?;
        if let Some(tp) = tp.as_mut() {
            tp.qkv_pre = qkv_h.clone();
            tp.g = g_h.clone();
            tp.beta = beta_h.clone();
            // the device conv path's output is only observable per 64-chunk (core.yb);
            // NaN marks rows no path filled (decode fuses the conv into the step kernel)
            if t > 1 {
                tp.conv_out = vec![f32::NAN; t * 3 * di];
            }
        }
        if std::env::var("TANG_GLM5_KDA_DUMP").is_ok_and(|v| v == "1") {
            // the staging probe (matches kda::kda_layer's CPU dump; layout/units equal):
            // qkv[0..16] (token 0), g[0..8], beta[0..4], the device state's first 8,
            // and the host conv copy's first 4
            let st_h = self.core.st.read::<f32>(0, 8).unwrap_or_else(|e| {
                // surface read errors loudly: a silent empty vec here looks like a zero
                // state and wasted a diagnosis cycle (every state line printed 0.0)
                eprintln!("kda gpu state READ ERROR: {e}");
                Vec::new()
            });
            eprintln!(
                "kda gpu qkv[0..16]: {:?}\n kda gpu qkv_k[0..8]: {:?}\n kda gpu qkv_v[0..8]: {:?}\n kda gpu g[0..8]: {:?}\n kda gpu beta[0..4]: {:?}\n kda gpu state[0..8]: {:?}\n kda gpu conv_host[0..4]: {:?}",
                &qkv_h[..16.min(qkv_h.len())],
                &qkv_h[di..di + 8.min(qkv_h.len() - di)],
                &qkv_h[2 * di..2 * di + 8.min(qkv_h.len() - 2 * di)],
                &g_h[..8.min(g_h.len())],
                &beta_h[..4.min(beta_h.len())],
                &st_h,
                &conv.0[..4.min(conv.0.len())],
            );
        }
        // W4's core decode/prefill index state.head(cfg, slot, h) on the host even when the
        // device owns the state — hand it a correctly-sized zero buffer so the indexing is
        // in-bounds (the values are ignored: the device-side state is authoritative).
        let mut dummy_state = KdaState(vec![0f32; self.cfg.heads * self.cfg.head_dim * self.cfg.head_dim]);
        let mut o = vec![0f32; t * di];
        if t == 1 {
            o.copy_from_slice(&self.core.decode(
                &self.kers, &self.cfg, &qkv_h, &g_h, &beta_h, conv, &mut dummy_state, 1,
            )?);
        } else {
            let mut off = 0usize;
            while off + 64 <= t {
                let o64 = self.core.prefill(
                    &self.kers,
                    &self.cfg,
                    &qkv_h[off * 3 * di..(off + 64) * 3 * di],
                    &g_h[off * di..(off + 64) * di],
                    &beta_h[off * nh..(off + 64) * nh],
                    conv,
                    &mut dummy_state,
                    1,
                    64,
                )?;
                o[off * di..(off + 64) * di].copy_from_slice(&o64);
                if let Some(tp) = tp.as_mut() {
                    let y64 = self.core.yb.read::<f32>(0, 64 * 3 * di)?;
                    tp.conv_out[off * 3 * di..(off + 64) * 3 * di].copy_from_slice(&y64);
                }
                off += 64;
            }
            // a tail shorter than one chunk: the CPU core on the device's state —
            // read it back, run the CPU chunk, seed the result forward (tails < 64 only)
            if off < t {
                let rem = t - off;
                let (mut hconv, mut hstate) = (
                    conv.clone(),
                    KdaState(vec![
                        0f32;
                        self.cfg.heads
                            * self.cfg.head_dim
                            * self.cfg.head_dim
                    ],
                ));
                self.read_back(&mut hconv, &mut hstate)?;
                let yc = super::kda::conv1d_prefill(
                    &qkv_h[off * 3 * di..t * 3 * di],
                    &self.conv_w,
                    &mut hconv,
                    &self.cfg,
                    1,
                    rem,
                )?;
                let oc = super::kda::kda_chunk(
                    &yc,
                    &g_h[off * di..t * di],
                    &beta_h[off * nh..t * nh],
                    &mut hstate,
                    &self.cfg,
                    1,
                    rem,
                    64,
                )?;
                o[off * di..t * di].copy_from_slice(&oc);
                if let Some(tp) = tp.as_mut() {
                    tp.conv_out[off * 3 * di..t * 3 * di].copy_from_slice(&yc);
                }
                self.seed(&hconv, &hstate)?;
                *conv = hconv;
            }
        }
        // gated output norm (device) then o_proj
        if std::env::var("TANG_GLM5_KDA_DUMP").is_ok_and(|v| v == "1") {
            // the core output pre-norm (the CPU tail path's `o`): y-divergence with a
            // matching o pins the bug to the gated norm or the o_proj GEMV
            eprintln!("kda gpu o[0..8]: {:?}", &o[..8.min(o.len())]);
        }
        self.ob.write(0, &o)?;
        unsafe {
            // g2 = g_b(g_a(x)): g_a → g2a, g_b: g2a → g2b (out of place)
            launch(
                self.gv.gemv,
                grid(t * hd),
                (128, 1, 1),
                0,
                &self.s,
                args![self.ga.w, 0u32, self.xb.ptr, self.g2a.ptr, hd as u32, t as u32,
                    hd as u32, hidden as u32],
            )?;
            launch(
                self.gv.gemv,
                grid(t * di),
                (128, 1, 1),
                0,
                &self.s,
                args![self.gb.w, 0u32, self.g2a.ptr, self.g2b.ptr, di as u32, t as u32,
                    di as u32, hd as u32],
            )?;
            launch(
                self.gv.norm,
                grid(t * di),
                (128, 1, 1),
                0,
                &self.s,
                args![self.ob.ptr, self.on.ptr, self.g2b.ptr, self.yb.ptr, nh as u32,
                    hd as u32, self.cfg.eps, t as u32],
            )?;
            launch(
                self.gv.gemv,
                grid(t * hidden),
                (128, 1, 1),
                0,
                &self.s,
                args![self.wo.w, 0u32, self.yb.ptr, self.outb.ptr, hidden as u32,
                    t as u32, hidden as u32, di as u32],
            )?;
        }
        self.s.sync()?;
        let out = self.outb.read::<f32>(0, t * hidden)?;
        if let Some(mut tp) = tp {
            tp.o = o;
            tp.g2 = self.g2b.read::<f32>(0, t * di)?;
            tp.y_norm = self.yb.read::<f32>(0, t * di)?;
            tp.out = out.clone();
            self.taps = Some(tp);
        }
        if std::env::var("TANG_GLM5_KDA_DUMP").is_ok_and(|v| v == "1") {
            // the layer OUTPUT (post gated-norm + o_proj): pins the divergence to the
            // core (ob/yb) or the output path (norm/wo) when x already matches
            eprintln!("kda gpu y[0..8]: {:?}", &out[..8.min(out.len())]);
        }
        Ok(out)
    }
}

/// k5_gemv8 vs k5_gemv on one real-shaped layer of the tiny GLM-5 fixture, in-process:
/// the same x through the same DevLayer with the projection kernel flipped, the device
/// taps compared in full. Isolates the kernel from the engine (the gemv8 bring-up's first
/// three failures — no smem, misaligned u64, missing __syncwarp + overlapping staging,
/// and the k5_q8 MAXT argument — would each have shown here as a q/k/v divergence).
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    // the Q8_0 fixture: gemv8 only takes Q8_0 rows (the f32 fixture routes to k5_gemv)
    const TINY: &str = "/private/tmp/glm5-tiny/glm5-tiny-q8_0.gguf";

    #[test]
    fn gemv8_matches_gemv_on_tiny_layer() {
        let required = std::env::var("TANG_REQUIRE_CUDA").is_ok_and(|v| v == "1");
        let Some(g) = Gpu::for_test() else {
            assert!(!required, "no CUDA device but TANG_REQUIRE_CUDA=1");
            return;
        };
        if !Path::new(TINY).exists() {
            assert!(!required, "tiny fixture absent but TANG_REQUIRE_CUDA=1");
            return;
        }
        let r = super::super::forward::Glm5Ref::open(Path::new(TINY)).unwrap();
        let hp = &r.hp;
        let l = (0..hp.n_layer).find(|&l| hp.is_kda[l]).unwrap();
        let cfg = KdaConfig {
            heads: hp.n_head,
            head_dim: hp.kda_head_dim,
            d_conv: hp.ssm_d_conv,
            gate_lower_bound: hp.kda_gate_lower_bound,
            eps: hp.eps,
        };
        let conv = ConvState::zeros(&cfg, 1);
        let state = KdaState::zeros(&cfg, 1);
        g.bind().unwrap();
        let mut layer = DevLayer::new_gguf(&g, &r.g, l, &cfg, hp.n_embd, &conv, &state, 64).unwrap();
        let t = 20usize;
        let mut seed = 7u64;
        let x: Vec<f32> = (0..t * hp.n_embd)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect();
        let run = |layer: &mut DevLayer, on: bool| {
            layer.gemv8 = on;
            layer.debug = true;
            let mut c = ConvState::zeros(&cfg, 1);
            layer.seed(&c, &KdaState::zeros(&cfg, 1)).unwrap();
            let y = layer.window(&x, t, &mut c).unwrap();
            layer.debug = false;
            (y, layer.taps.take().unwrap())
        };
        assert!(
            [&layer.wq, &layer.wk, &layer.wv].iter().all(|w| w.w.ty == 8),
            "fixture q/k/v must be Q8_0 for this test to exercise k5_gemv8"
        );
        let (y0, t0) = run(&mut layer, false);
        let (y1, t1) = run(&mut layer, true);
        let di = cfg.heads * cfg.head_dim;
        for (sec, name) in [(0, "q"), (1, "k"), (2, "v")] {
            let a = super::super::kda_debug::section(&t0.qkv_pre, t, di, sec);
            let b = super::super::kda_debug::section(&t1.qkv_pre, t, di, sec);
            let s = super::super::kda_debug::stat(&b, &a, 1e-2);
            eprintln!("qkv.{name}: rel {:.3e} max {:.3e} nonfinite {} zeros {}/{}", s.rel_l2, s.max_abs, s.nonfinite_gpu, s.zeros_gpu, s.zeros_cpu);
            // int8 activations vs f32: the fe_gemv8 band, not bitwise
            assert!(s.nonfinite_gpu == 0 && s.rel_l2 < 1e-2, "gemv8 qkv.{name} diverges: {s:?}");
        }
        let s = super::super::kda_debug::stat(&y1, &y0, 1e-2);
        eprintln!("layer out: rel {:.3e} max {:.3e}", s.rel_l2, s.max_abs);
        assert!(s.nonfinite_gpu == 0 && s.rel_l2 < 2e-2, "gemv8 layer output diverges: {s:?}");
    }
}
