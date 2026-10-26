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
    /// `TANG_GLM5_KDA_PF2=1` (read at load): the device-resident core — conv in place on
    /// `qkvb`, the parallel chunked prefill ([`super::kda_cuda::PF2_SRC`]) for any `t`
    /// (no CPU tail), the decode step from device pointers; qkv/g/beta never visit the
    /// host. Off: the original host-staged path (64-chunks on the device + CPU tail).
    pf2: bool,
    /// `TANG_GLM5_KDA_DEBUG` / `TANG_KERNEL_VERIFY`: when the engine sets a probe for
    /// this window, [`Self::window`] poisons its scratch with NaN first (per the probe) and
    /// leaves every intermediate in the probe's taps for the shadow check
    /// ([`super::kda_debug`], names in [`super::kda_debug::tap`]). `None`: zero cost.
    pub probe: Option<tang_verify::Probe>,
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
        let pf2 = std::env::var("TANG_GLM5_KDA_PF2").is_ok_and(|v| v == "1");
        let mut core = DevKda::new(g, &kers, cfg, &conv_w, conv, state, slots)?;
        if pf2 {
            core.enable_pf2(max_t)?;
        }
        Ok(Self {
            core,
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
            pf2,
            probe: None,
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
        let mut tp = self.probe.take();
        if let Some(p) = &tp {
            // NaN-fill every output scratch this window writes (whole buffers): a region
            // no kernel touches then reads NaN in the shadow check, not a plausible 0.0.
            // Includes the core's own scratch (conv output, core output) and PF2's
            // workspace (the pf_state temporaries).
            p.poison(&[
                &self.qkvb, &self.g1b, &self.gb_out, &self.bb, &self.g2a, &self.g2b, &self.yb,
                &self.outb, &self.ob, &self.core.yb, &self.core.ob, &self.core.ws,
            ])?;
        }
        let prof = std::env::var("TANG_GLM5_PROF").is_ok_and(|v| v == "1");
        let t_win = std::time::Instant::now();
        // projections → the stacked q|k|v layout (base offsets q=0, k=di, v=2·di FLOATS).
        // The device pointer is a BYTE address: the offsets must be scaled by
        // size_of::<f32>(). Unscaled (the GLM-5.3 staging bug, Oct 2026) k landed di/4 and
        // v di/2 floats into the q block — q corrupted past 2048, half of k and all of v
        // never written (zeros on a fresh buffer → the core output was exactly 0).
        let qkv = self.qkvb.ptr;
        let f = std::mem::size_of::<f32>() as u64;
        unsafe {
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
        if self.pf2 {
            if prof {
                self.s.sync()?;
                eprintln!("glm5: prof kda proj: {:.2} ms", t_win.elapsed().as_secs_f32() * 1e3);
            }
            let t_core = std::time::Instant::now();
            self.core_pf2(t, conv, &mut tp)?;
            if prof {
                self.s.sync()?;
                eprintln!("glm5: prof kda core: {:.2} ms", t_core.elapsed().as_secs_f32() * 1e3);
            }
            return self.finish(t, None, tp);
        }
        self.s.sync()?;
        if prof {
            eprintln!("glm5: prof kda proj: {:.2} ms", t_win.elapsed().as_secs_f32() * 1e3);
        }
        let t_core = std::time::Instant::now();
        // the core (W4's launchers stage their inputs from host): qkv, g, beta
        let qkv_h = self.qkvb.read::<f32>(0, t * 3 * di)?;
        let g_h = self.gb_out.read::<f32>(0, t * di)?;
        let beta_h = self.bb.read::<f32>(0, t * nh)?;
        if let Some(p) = tp.as_mut() {
            use super::kda_debug::tap;
            p.put(tap::QKV, qkv_h.clone());
            p.put(tap::G, g_h.clone());
            p.put(tap::BETA, beta_h.clone());
            // the device conv path's output is only observable per 64-chunk (core.yb),
            // filled below; decode fuses the conv into the step kernel (no conv tap)
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
                if let Some(p) = tp.as_mut() {
                    p.read_at(super::kda_debug::tap::CONV, off * 3 * di, &self.core.yb, 64 * 3 * di)?;
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
                if let Some(p) = tp.as_mut() {
                    p.taps.put_at(super::kda_debug::tap::CONV, off * 3 * di, &yc);
                }
                self.seed(&hconv, &hstate)?;
                *conv = hconv;
            }
        }
        self.ob.write(0, &o)?;
        if prof {
            eprintln!("glm5: prof kda core: {:.2} ms", t_core.elapsed().as_secs_f32() * 1e3);
        }
        self.finish(t, Some(o), tp)
    }

    /// The PF2 core on the device: qkv (pre-conv, in `qkvb`), g (`gb_out`) and beta (`bb`)
    /// straight from the projection kernels into the core, the core output into `ob` —
    /// all on the layer stream, no host round trip. Under the shadow check the inputs and
    /// the conv output (in place in `qkvb`) are read back as taps, and the host conv copy
    /// is mirrored from the device (this path doesn't maintain it: the device is
    /// authoritative and `read_back` syncs it before any CPU use).
    fn core_pf2(&mut self, t: usize, conv: &mut ConvState, tp: &mut Option<tang_verify::Probe>) -> Result<()> {
        let (nh, di) = (self.nh, self.d_inner);
        if let Some(p) = tp.as_mut() {
            use super::kda_debug::tap;
            self.s.sync()?;
            p.read(tap::QKV, &self.qkvb, t * 3 * di)?;
            p.read(tap::G, &self.gb_out, t * di)?;
            p.read(tap::BETA, &self.bb, t * nh)?;
        }
        if t == 1 {
            self.core.decode_dev(&self.kers, &self.s, self.qkvb.ptr, self.gb_out.ptr, self.bb.ptr, self.ob.ptr)?;
        } else {
            self.core.prefill_dev(
                &self.kers,
                &self.s,
                self.qkvb.ptr,
                3 * di,
                self.gb_out.ptr,
                di,
                self.bb.ptr,
                nh,
                self.ob.ptr,
                t,
            )?;
        }
        if let Some(p) = tp.as_mut() {
            use super::kda_debug::tap;
            self.s.sync()?;
            // the conv ran in place on qkvb — the post-conv qkv IS the conv output tap
            if t > 1 {
                p.read_at(tap::CONV, 0, &self.qkvb, t * 3 * di)?;
            }
            self.core.read_conv(conv)?;
        }
        Ok(())
    }

    /// The gated output norm + o_proj from the core output in `ob` (`o`: the host copy
    /// when the host-staged path produced it), then the taps.
    fn finish(&mut self, t: usize, o: Option<Vec<f32>>, tp: Option<tang_verify::Probe>) -> Result<Vec<f32>> {
        let (nh, hd, di, hidden) = (self.nh, self.hd, self.d_inner, self.hidden);
        let prof = std::env::var("TANG_GLM5_PROF").is_ok_and(|v| v == "1");
        let t_out = std::time::Instant::now();
        // gated output norm (device) then o_proj
        if std::env::var("TANG_GLM5_KDA_DUMP").is_ok_and(|v| v == "1") {
            // the core output pre-norm (the CPU tail path's `o`): y-divergence with a
            // matching o pins the bug to the gated norm or the o_proj GEMV
            let o = match &o {
                Some(o) => o.clone(),
                None => self.ob.read::<f32>(0, t * di)?,
            };
            eprintln!("kda gpu o[0..8]: {:?}", &o[..8.min(o.len())]);
        }
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
        if prof {
            eprintln!("glm5: prof kda out: {:.2} ms", t_out.elapsed().as_secs_f32() * 1e3);
        }
        let out = self.outb.read::<f32>(0, t * hidden)?;
        if let Some(mut p) = tp {
            use super::kda_debug::tap;
            // the host-staged path hands back the core output it read; PF2 leaves it in
            // `ob` (device-resident) — read it here for the tap either way
            let o = match o {
                Some(o) => o,
                None => self.ob.read::<f32>(0, t * di)?,
            };
            p.put(tap::O, o);
            p.read(tap::G2, &self.g2b, t * di)?;
            p.read(tap::Y_NORM, &self.yb, t * di)?;
            p.put(tap::OUT, out.clone());
            self.probe = Some(p);
        }
        if std::env::var("TANG_GLM5_KDA_DUMP").is_ok_and(|v| v == "1") {
            // the layer OUTPUT (post gated-norm + o_proj): pins the divergence to the
            // core (ob/yb) or the output path (norm/wo) when x already matches
            eprintln!("kda gpu y[0..8]: {:?}", &out[..8.min(out.len())]);
        }
        Ok(out)
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::super::kda::{ConvState, KdaState};
    use super::DevLayer;

    fn rel(a: &[f32], b: &[f32]) -> f64 {
        let (mut d, mut n) = (0f64, 0f64);
        for (x, y) in a.iter().zip(b) {
            d += ((x - y) as f64).powi(2);
            n += (*y as f64).powi(2);
        }
        if a.iter().any(|x| !x.is_finite()) {
            return f64::INFINITY;
        }
        (d / n.max(1e-30)).sqrt()
    }

    /// Real GLM-5.3 KDA layer 0 (`TANG_GLM5_BENCH_GGUF` = the first shard), one resident
    /// `DevLayer` per path: the original host-staged core (64-chunks + CPU tail) vs PF2.
    /// Times whole windows (proj + core + out, `TANG_GLM5_PROF=1` prints the split) at
    /// 20 / 512 / 2048 tokens, checks PF2's layer output against the original path's at
    /// every length, and both against the CPU `kda_layer` at 20 (the oracle). Each
    /// window starts from the state the previous one left (a running prefill).
    /// Ignored: `TANG_GLM5_BENCH_GGUF=… cargo test … -- --ignored glm5_kda_layer_bench_real`.
    #[test]
    #[ignore]
    fn glm5_kda_layer_bench_real() {
        let Ok(path) = std::env::var("TANG_GLM5_BENCH_GGUF") else {
            eprintln!("SKIP: TANG_GLM5_BENCH_GGUF unset");
            return;
        };
        let g = tang_moe::gpu::Gpu::for_test().expect("CUDA GPU");
        g.bind().unwrap();
        let r = super::super::forward::Glm5Ref::open(std::path::Path::new(&path)).unwrap();
        let l = 0usize;
        let (w, cfg) = r.kda_weights(l).unwrap();
        let hidden = r.hp.n_embd;
        let max_t = 2048usize;
        let mk = |pf2: bool| {
            // DevLayer reads the flag at load
            std::env::set_var("TANG_GLM5_KDA_PF2", if pf2 { "1" } else { "0" });
            let d = DevLayer::new_gguf(&g, &r.g, l, &cfg, hidden, &ConvState::zeros(&cfg, 1),
                &KdaState::zeros(&cfg, 1), max_t).unwrap();
            std::env::remove_var("TANG_GLM5_KDA_PF2");
            d
        };
        let (mut old, mut new) = (mk(false), mk(true));
        let (mut c_old, mut c_new) = (ConvState::zeros(&cfg, 1), ConvState::zeros(&cfg, 1));
        let (mut c_cpu, mut s_cpu) = (ConvState::zeros(&cfg, 1), KdaState::zeros(&cfg, 1));
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        for t in [20usize, 512, 2048] {
            // a normed-hidden-like input: uniform with unit RMS
            let x: Vec<f32> = (0..t * hidden)
                .map(|_| {
                    seed ^= seed >> 12;
                    seed ^= seed << 25;
                    seed ^= seed >> 27;
                    ((seed.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 3.46
                })
                .collect();
            let t0 = std::time::Instant::now();
            let y_old = old.window(&x, t, &mut c_old).unwrap();
            let ms_old = t0.elapsed().as_secs_f64() * 1e3;
            // PF2: best of 3 from the same state (read back, re-seed): the first window
            // pays the module's first-launch costs
            let (mut c0, mut s0) = (c_new.clone(), KdaState::zeros(&cfg, 1));
            new.read_back(&mut c0, &mut s0).unwrap();
            let mut best = f64::MAX;
            let mut y_new = Vec::new();
            for _ in 0..3 {
                new.seed(&c0, &s0).unwrap();
                let t1 = std::time::Instant::now();
                y_new = new.window(&x, t, &mut c_new).unwrap();
                best = best.min(t1.elapsed().as_secs_f64() * 1e3);
            }
            let e = rel(&y_new, &y_old);
            // the states the two paths carry forward
            let (mut co, mut so) = (c_old.clone(), KdaState::zeros(&cfg, 1));
            old.read_back(&mut co, &mut so).unwrap();
            let (mut cn, mut sn) = (c_new.clone(), KdaState::zeros(&cfg, 1));
            new.read_back(&mut cn, &mut sn).unwrap();
            let es = rel(&sn.0, &so.0);
            let ec = rel(&cn.0, &co.0);
            println!(
                "kda layer {l} T={t}: window old {ms_old:.1} ms, pf2 {best:.2} ms ({:.0}x); x34 layers: old {:.0} ms, pf2 {:.0} ms; pf2 vs old: out rel {e:.2e} state {es:.2e} conv {ec:.2e}",
                ms_old / best,
                ms_old * 34.0,
                best * 34.0
            );
            assert!(e < 1e-3 && es < 1e-3 && ec < 1e-6, "pf2 diverges from the original path at T={t}");
            if t == 20 {
                let y_cpu = super::super::kda::kda_layer(&x, &w, &cfg, &mut c_cpu, &mut s_cpu, 1, t).unwrap();
                let (ecpu, escpu) = (rel(&y_new, &y_cpu), rel(&sn.0, &s_cpu.0));
                println!("kda layer {l} T={t}: pf2 vs CPU kda_layer: out rel {ecpu:.2e} state {escpu:.2e}");
                assert!(ecpu < 1e-3 && escpu < 1e-3, "pf2 vs CPU at T={t}");
            }
        }
    }
}
