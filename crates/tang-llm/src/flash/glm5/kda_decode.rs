//! The fused KDA decode step + its CUDA graph (docs/models/glm5-speed-plan.md step 2).
//!
//! A child module of [`super`] (`kda_dev`) so it reads [`DevLayer`]'s fields directly and
//! keeps its diff to three lines there. At T = 1 the naive [`DevLayer::window`] issues 13
//! launches on two streams with three host round-trips between them (qkv/g/beta read back
//! and re-staged for W4's `DevKda::decode`, the core output `o` read back and re-staged
//! for the norm, the 4 MiB dummy state zeroed per call, three stream syncs). Here:
//!
//! 1. `k5d_proj`  — q | k | v, f_a, beta (+σ), g_a: every GEMV over `x`, one launch
//! 2. `k5d_gate2` — f_b(g1) + dt + gate form, and g_b(g2a): one launch
//! 3. `k5d_step`  — conv + delta rule + T=1 conv commit + gated norm: one launch
//! 4. `k5_gemv`   — o_proj (the existing kernel)
//!
//! all on the layer's stream, device pointers only — the host writes `x` before and reads
//! `out` after. That sequence has fixed pointers and shapes, so with
//! `TANG_GLM5_KDA_GRAPH=1` it is captured once (after one eager run) and replayed as a
//! single `cuGraphLaunch` per layer per token. With `TANG_GLM5_GEMV8=1` the q/k/v rows
//! go through `k5_q8` + `k5_gemv8` instead (kda_dev's Q8_0-only rule; the graph is keyed
//! on the choice).
//!
//! Env (default OFF until verified on the real model):
//! - `TANG_GLM5_KDA_FUSED=1` — the fused launches, eager
//! - `TANG_GLM5_KDA_GRAPH=1` — the fused launches, graph-captured (implies FUSED)
//!
//! Numerics: each fused kernel evaluates the naive kernel's expressions in the same
//! order, so the outputs equal the naive path's to the bit — with one deliberate
//! exception: `k5d_step` adds the `__syncthreads()` the naive `kda_step` lacks between the
//! q/k norm reductions and the in-place rescale (a cross-warp read-after-write race in
//! W4's kernel; benign whenever the 4 warps happen to finish the reduction together).
//!
//! Host conv copy: the naive decode shifts the engine's host `ConvState` as a side effect.
//! Nothing reads it while the device is fresh (`kda_mixer` reads back before any CPU
//! window), and the prefill-chunk path already leaves it stale — so the fused path
//! skips the shift (it would cost a qkv read-back) EXCEPT in a debug window, where the
//! shadow check compares it (`conv_host`).
#![allow(clippy::too_many_arguments)]
use super::DevLayer;
use super::super::kda::ConvState;
use anyhow::{ensure, Context, Result};
use cudarc::driver::sys;
use tang_moe::gpu::{launch, Event, Graph, Gpu, Module, Stream};
use tang_moe::args;

/// Which decode path a call takes (env-selected by [`Mode::from_env`]; tests pass it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Naive,
    Fused,
    Graph,
}

impl Mode {
    pub fn from_env() -> Self {
        let on = |k: &str| std::env::var(k).is_ok_and(|v| v == "1");
        if on("TANG_GLM5_KDA_GRAPH") {
            Mode::Graph
        } else if on("TANG_GLM5_KDA_FUSED") {
            Mode::Fused
        } else {
            Mode::Naive
        }
    }
}

/// The fused kernels of one layer's device + the captured graph.
pub struct Fast {
    _m: Module,
    proj: sys::CUfunction,
    gate2: sys::CUfunction,
    step: sys::CUfunction,
    /// The captured decode step and the `TANG_GLM5_GEMV8` value it was captured with.
    graph: Option<(Graph, bool)>,
}

impl Fast {
    pub fn new(g: &Gpu, cfg: &super::KdaConfig) -> Result<Self> {
        let src = format!(
            "{}\n{}\n{}",
            super::super::kda_cuda::prelude(cfg),
            super::super::moe::SRC,
            include_str!("kda_decode.cuh")
        );
        let m = g.module(&src).map_err(|e| anyhow::anyhow!("kda_decode nvrtc: {e}"))?;
        let f = |n: &str| m.func(n).map_err(|e| anyhow::anyhow!("kda_decode {n}: {e}"));
        Ok(Fast { proj: f("k5d_proj")?, gate2: f("k5d_gate2")?, step: f("k5d_step")?, graph: None, _m: m })
    }
}

fn grid(n: usize) -> (u32, u32, u32) {
    (n.div_ceil(128) as u32, 1, 1)
}

impl DevLayer {
    /// Enqueue the fused decode step on `s` (device pointers only: valid inside a capture).
    /// `ev` (eager only, the bench's per-stage breakdown): 5 events recorded before the
    /// projections and after each of proj / gate2 / step / o_proj.
    fn enqueue_decode(&self, s: &Stream, gemv8: bool, ev: Option<&[Event]>) -> Result<()> {
        let mark = |i: usize| -> Result<()> {
            if let Some(e) = ev {
                e[i].record(s).map_err(|e| anyhow::anyhow!("{e}"))?;
            }
            Ok(())
        };
        mark(0)?;
        let fast = self.fast.as_ref().context("kda fused decode: kernels not built")?;
        let (nh, hd, di, hidden) = (self.nh, self.hd, self.d_inner, self.hidden);
        let lb = self.cfg.gate_lower_bound.unwrap_or(f32::NAN);
        let f = std::mem::size_of::<f32>() as u64;
        let qkv = self.qkvb.ptr;
        let rows = 3 * di + 2 * hd + nh;
        // the per-matrix gemv8 rule (kda_dev window()): f_a / g_a go through k5_gemv8 when
        // Q8_0 — their f32-dequant g5_dot rounds differently, so k5d_proj must SKIP exactly
        // the groups gemv8 wrote (the forget gate amplifies any mismatch into the state)
        let g8f = gemv8 && self.fa.w.ty == 8;
        let g8ga = gemv8 && self.ga.w.ty == 8;
        let skip = (g8f as u32) | ((g8ga as u32) << 1);
        unsafe {
            let r0 = if gemv8 {
                // the dequant-reuse GEMV for q/k/v (kda_dev's TANG_GLM5_GEMV8 path, T = 1)
                launch(
                    self.gv.q8,
                    ((hidden / 32) as u32, 1, 1),
                    (32, 1, 1),
                    0,
                    s,
                    args![self.xb.ptr, self.xq8.ptr, hidden as u32, self.max_t as u32],
                )?;
                let codes = self.xq8.ptr;
                let scales = codes + (self.max_t * hidden) as u64;
                for (w, off) in [(&self.wq, 0u64), (&self.wk, di as u64 * f), (&self.wv, 2 * di as u64 * f)] {
                    // the same dynamic smem as kda_dev's gemv8 launch (keep the two in step)
                    let smem = (super::K5_WARPS * ((w.w.row_bytes as usize).div_ceil(16) + 1) * 16) as u32;
                    launch(
                        self.gv.gemv8,
                        (di.div_ceil(super::K5_WARPS) as u32, 1, 1),
                        (128, 1, 1),
                        smem,
                        s,
                        args![w.w, 0u32, codes, scales, qkv + off, (3 * di) as u32, 1u32,
                            di as u32, hidden as u32],
                    )?;
                }
                // f_a / g_a read the SAME x codes — strictly AFTER k5_q8 above: before it,
                // xq8 still holds the previous step's o_proj codes (yb, d_inner wide), a
                // stale read that diverged every output on the first real-weights rerun
                if g8f {
                    self.gemv8_launch(&self.fa, codes, hidden, self.g1b.ptr, hd, hd, 1)?;
                }
                if g8ga {
                    self.gemv8_launch(&self.ga, codes, hidden, self.g2a.ptr, hd, hd, 1)?;
                }
                3 * di
            } else {
                0
            };
            launch(
                fast.proj,
                grid(rows - r0),
                (128, 1, 1),
                0,
                s,
                args![self.wq.w, self.wk.w, self.wv.w, self.fa.w, self.bproj.w, self.ga.w,
                    self.xb.ptr, qkv, self.g1b.ptr, self.bb.ptr, self.g2a.ptr, di as u32,
                    hd as u32, nh as u32, hidden as u32, r0 as u32, skip],
            )?;
            mark(1)?;
            launch(
                fast.gate2,
                grid(2 * di),
                (128, 1, 1),
                0,
                s,
                args![self.fb.w, self.gb.w, self.g1b.ptr, self.g2a.ptr, self.dt.ptr,
                    self.rate.ptr, self.gb_out.ptr, self.g2b.ptr, nh as u32, hd as u32, lb],
            )?;
            mark(2)?;
            launch(
                fast.step,
                (nh as u32, 1, 1),
                (hd as u32, 1, 1),
                0,
                s,
                args![qkv, self.core.h0.ptr, self.core.wconv.ptr, self.gb_out.ptr,
                    self.bb.ptr, self.core.st.ptr, self.ob.ptr, self.on.ptr, self.g2b.ptr,
                    self.yb.ptr, self.cfg.eps],
            )?;
            mark(3)?;
            // o_proj: gemv8 quantises yb (d_inner wide) first — the SAME xq8 the q/k/v
            // pre-pass used (nothing reads x's codes after the q/k/v GEMVs, so overwriting
            // them here is safe; kda_dev's finish() does the same)
            let g8o = gemv8 && self.wo.w.ty == 8;
            if g8o {
                launch(
                    self.gv.q8,
                    ((di / 32) as u32, 1, 1),
                    (32, 1, 1),
                    0,
                    s,
                    args![self.yb.ptr, self.xq8.ptr, di as u32, self.max_t as u32],
                )?;
                let codes = self.xq8.ptr;
                let scales = codes + (self.max_t * di) as u64;
                let smem = (super::K5_WARPS * ((self.wo.w.row_bytes as usize).div_ceil(16) + 1) * 16) as u32;
                launch(
                    self.gv.gemv8,
                    (hidden.div_ceil(super::K5_WARPS) as u32, 1, 1),
                    (128, 1, 1),
                    smem,
                    s,
                    args![self.wo.w, 0u32, codes, scales, self.outb.ptr, hidden as u32, 1u32,
                        hidden as u32, di as u32],
                )?;
            } else {
                launch(
                    self.gv.gemv,
                    grid(hidden),
                    (128, 1, 1),
                    0,
                    s,
                    args![self.wo.w, 0u32, self.yb.ptr, self.outb.ptr, hidden as u32, 1u32,
                        hidden as u32, di as u32],
                )?;
            }
            mark(4)?;
        }
        Ok(())
    }

    /// One decode token through the fused step (`mode` Fused or Graph). Same contract as
    /// [`DevLayer::window`] at `t = 1`, including the shadow-check probe (the poisoned
    /// buffers get their taps read out of the device, `out`/`conv_host` put on the host).
    pub(super) fn decode_fast(&mut self, x: &[f32], conv: &mut ConvState, mode: Mode) -> Result<Vec<f32>> {
        ensure!(x.len() == self.hidden, "k5 decode input");
        ensure!(mode != Mode::Naive, "decode_fast: naive mode");
        let (nh, di, hidden) = (self.nh, self.d_inner, self.hidden);
        // kda_dev's per-matrix rule (the x pre-pass): gemv8 only when q/k/v are all Q8_0
        // (the kernel reads 34-byte Q8_0 blocks; the f32 tiny fixture must take k5_gemv)
        let g8 = |w: &super::super::moe::gpu::DevW| self.gemv8 && w.w.ty == 8;
        let gemv8 = g8(&self.wq) && g8(&self.wk) && g8(&self.wv);
        // the core's prefill path uses its own stream: make sure it has drained (it syncs
        // at the end of every call, so this is free; it keeps the ordering explicit)
        self.core.s.sync()?;
        // synchronous HtoD on the legacy stream; self.s is BLOCKING, so the launches (or
        // the graph) below are ordered after it
        self.xb.write(0, x)?;
        let mut tp = self.probe.take();
        if let Some(p) = &tp {
            // NaN-fill every output scratch this window writes (whole buffers): a region
            // no kernel touches then reads NaN in the shadow check, not a plausible 0.0.
            // (The PF2 workspace `core.ws` is prefill-only, not touched at t = 1.)
            p.poison(&[
                &self.qkvb, &self.g1b, &self.gb_out, &self.bb, &self.g2a, &self.g2b, &self.yb,
                &self.outb, &self.ob,
            ])?;
        }
        match mode {
            Mode::Graph if self.fast.as_ref().and_then(|f| f.graph.as_ref()).is_some_and(|g| g.1 == gemv8) => {
                let f = self.fast.as_ref().unwrap();
                f.graph.as_ref().unwrap().0.launch(&self.s).map_err(|e| anyhow::anyhow!("kda graph launch: {e}"))?;
            }
            Mode::Graph => {
                // first token (or GEMV8 toggled): run eagerly, then capture for the next
                // tokens (capture records, it does not execute)
                self.enqueue_decode(&self.s, gemv8, None)?;
                self.s.sync()?;
                if let Some(f) = self.fast.as_mut() {
                    f.graph = None;
                }
                let me = &*self;
                let gr = Graph::capture(&me.s, |s| {
                    me.enqueue_decode(s, gemv8, None).map_err(|e| tang_moe::gpu::Error(format!("{e:#}")))
                })
                .map_err(|e| anyhow::anyhow!("kda decode capture: {e}"))?;
                self.fast.as_mut().unwrap().graph = Some((gr, gemv8));
            }
            _ => self.enqueue_decode(&self.s, gemv8, None)?,
        }
        self.s.sync()?;
        let out = self.outb.read::<f32>(0, hidden)?;
        let Some(mut p) = tp else {
            // no probe: keep the pre-probe behavior — the host conv copy is NOT shifted
            // (that would cost a qkv read-back per token per layer; nothing reads it while
            // the device is fresh — `read_back` syncs it before any CPU use)
            return Ok(out);
        };
        use super::super::kda_debug::tap;
        // the host conv shift (DevKda::decode's side effect), from qkv PRE-conv (the same
        // qkv the step kernel consumed)
        let qkv_h = self.qkvb.read::<f32>(0, 3 * di)?;
        let m = self.cfg.d_conv - 1;
        let ch = 3 * di;
        let mut next = conv.0.clone();
        for c in 0..ch {
            for s2 in 0..m {
                next[c * m + s2] = if s2 + 1 < m { conv.0[c * m + s2 + 1] } else { qkv_h[c] };
            }
        }
        *conv = ConvState(next);
        p.put(tap::QKV, qkv_h);
        p.put(tap::G, self.gb_out.read::<f32>(0, di)?);
        p.put(tap::BETA, self.bb.read::<f32>(0, nh)?);
        p.put(tap::O, self.ob.read::<f32>(0, di)?);
        p.put(tap::G2, self.g2b.read::<f32>(0, di)?);
        p.put(tap::Y_NORM, self.yb.read::<f32>(0, di)?);
        p.put(tap::OUT, out.clone());
        self.probe = Some(p);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::super::kda::{ConvState, KdaConfig, KdaState};
    use std::path::Path;

    const TINY: &str = "/private/tmp/glm5-tiny/glm5-tiny-f32.gguf";

    fn poison(d: &DevLayer) {
        let (di, hd, nh) = (d.d_inner, d.hd, d.nh);
        let nan = |n: usize| vec![f32::NAN; n];
        for (b, n) in [(&d.qkvb, 3 * di), (&d.g1b, hd), (&d.gb_out, di), (&d.bb, nh), (&d.g2a, hd), (&d.g2b, di), (&d.ob, di), (&d.yb, di), (&d.outb, d.hidden)] {
            b.write(0, &nan(n)).unwrap();
        }
    }

    /// The fused decode (eager and graph-replayed) equals the naive decode on the tiny
    /// model's first KDA layer: 6 tokens of carried state, output and device state
    /// compared every step. Bit-equality is expected (same expressions) up to the naive
    /// step kernel's sync race, so the bar is 1e-5 relative.
    #[test]
    fn kda_fused_decode_matches_naive() {
        let required = std::env::var("TANG_REQUIRE_CUDA").is_ok_and(|v| v == "1");
        if !Path::new(TINY).exists() {
            assert!(!required, "TANG_REQUIRE_CUDA=1 but the tiny fixture is absent");
            eprintln!("SKIP kda_fused_decode_matches_naive: no fixture");
            return;
        }
        let Some(g) = Gpu::for_test() else { return };
        g.bind().unwrap();
        let gguf = crate::gguf::Gguf::open(Path::new(TINY)).unwrap();
        let hp = super::super::super::Hparams::from_gguf(&gguf).unwrap();
        let l = (0..hp.n_layer).find(|&l| hp.is_kda[l]).unwrap();
        let cfg = KdaConfig {
            heads: hp.n_head,
            head_dim: hp.kda_head_dim,
            d_conv: hp.ssm_d_conv,
            gate_lower_bound: hp.kda_gate_lower_bound,
            eps: hp.eps,
        };
        let n = hp.n_embd;
        // `a` has no fused kernels, so its window() is the naive path whatever the env says
        let mk = |fast: bool| {
            let mut d = DevLayer::new_gguf(&g, &gguf, l, &cfg, n, &ConvState::zeros(&cfg, 1), &KdaState::zeros(&cfg, 1), 8).unwrap();
            d.fast = fast.then(|| Fast::new(&g, &cfg).unwrap());
            d
        };
        let (mut a, mut b, mut c) = (mk(false), mk(true), mk(true));
        let mut seed = 7u64;
        let mut rnd = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let (mut ca, mut cb, mut cc) = (ConvState::zeros(&cfg, 1), ConvState::zeros(&cfg, 1), ConvState::zeros(&cfg, 1));
        let rel = |p: &[f32], q: &[f32]| {
            let num: f64 = p.iter().zip(q).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
            let den: f64 = q.iter().map(|y| (*y as f64).powi(2)).sum();
            (num / den.max(1e-30)).sqrt()
        };
        for step in 0..6 {
            let x: Vec<f32> = (0..n).map(|_| rnd()).collect();
            let ya = a.window(&x, 1, &mut ca).unwrap();
            // NaN-poison every buffer the fused step writes before each call: a graph node
            // that didn't replay (or a stale pointer) then reads NaN, not last step's values
            for d in [&b, &c] {
                poison(d);
            }
            let yb = b.decode_fast(&x, &mut cb, Mode::Fused).unwrap();
            let yc = c.decode_fast(&x, &mut cc, Mode::Graph).unwrap();
            let (mut sa, mut sb, mut sc) = (KdaState::zeros(&cfg, 1), KdaState::zeros(&cfg, 1), KdaState::zeros(&cfg, 1));
            let (mut ha, mut hb, mut hc) = (ca.clone(), cb.clone(), cc.clone());
            a.read_back(&mut ha, &mut sa).unwrap();
            b.read_back(&mut hb, &mut sb).unwrap();
            c.read_back(&mut hc, &mut sc).unwrap();
            let (r_fb, r_gr) = (rel(&yb, &ya), rel(&yc, &ya));
            let (s_fb, s_gr) = (rel(&sb.0, &sa.0), rel(&sc.0, &sa.0));
            let (h_fb, h_gr) = (rel(&hb.0, &ha.0), rel(&hc.0, &ha.0));
            let bits = ya == yb && ya == yc;
            eprintln!(
                "step {step}: out rel fused {r_fb:.2e} graph {r_gr:.2e}; state {s_fb:.2e} {s_gr:.2e}; conv {h_fb:.2e} {h_gr:.2e}; bit-equal {bits}"
            );
            assert!(ya.iter().all(|v| v.is_finite()) && ya.iter().any(|&v| v != 0.0), "naive output degenerate");
            for r in [r_fb, r_gr, s_fb, s_gr, h_fb, h_gr] {
                assert!(r <= 1e-5, "fused decode diverges from naive: {r:.3e} at step {step}");
            }
        }
        assert!(c.fast.as_ref().unwrap().graph.is_some(), "graph never captured");
    }

    /// REAL WEIGHTS, one KDA layer (TANG_GLM5_REAL_GGUF = shard 1; TANG_GLM5_BENCH_LAYER,
    /// default the first KDA layer): naive vs fused vs graph decode on the same random
    /// token stream — per-step bit-equality of the outputs and the final device state,
    /// then wall-clock µs per decode step of each (host-timed, x in → y out, i.e. what
    /// the engine's `kda mixer` stage pays per layer). Skips without the env var.
    #[test]
    fn kda_decode_real_bench() {
        let Ok(real) = std::env::var("TANG_GLM5_REAL_GGUF") else {
            eprintln!("skip: TANG_GLM5_REAL_GGUF not set");
            return;
        };
        let g = Gpu::for_test().expect("GPU");
        g.bind().unwrap();
        let gguf = crate::gguf::Gguf::open(Path::new(&real)).unwrap();
        let hp = super::super::super::Hparams::from_gguf(&gguf).unwrap();
        let l = std::env::var("TANG_GLM5_BENCH_LAYER")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| (0..hp.n_layer).find(|&l| hp.is_kda[l]).unwrap());
        assert!(hp.is_kda[l], "layer {l} is not KDA");
        let cfg = KdaConfig {
            heads: hp.n_head,
            head_dim: hp.kda_head_dim,
            d_conv: hp.ssm_d_conv,
            gate_lower_bound: hp.kda_gate_lower_bound,
            eps: hp.eps,
        };
        let n = hp.n_embd;
        let mk = |fast: bool| {
            let mut d = DevLayer::new_gguf(&g, &gguf, l, &cfg, n, &ConvState::zeros(&cfg, 1), &KdaState::zeros(&cfg, 1), 64).unwrap();
            d.fast = fast.then(|| Fast::new(&g, &cfg).unwrap());
            d
        };
        let mut seed = 11u64;
        let mut rnd = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let steps = 64usize;
        let xs: Vec<Vec<f32>> = (0..steps).map(|_| (0..n).map(|_| rnd()).collect()).collect();
        let mut outs: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut states = Vec::new();
        for mode in [Mode::Naive, Mode::Fused, Mode::Graph] {
            let mut d = mk(mode != Mode::Naive);
            let mut conv = ConvState::zeros(&cfg, 1);
            let mut ys = Vec::with_capacity(steps);
            // 8 warm-up steps (the graph captures on its first), then the timed rest
            let mut t0 = std::time::Instant::now();
            for (i, x) in xs.iter().enumerate() {
                if i == 8 {
                    t0 = std::time::Instant::now();
                }
                ys.push(match mode {
                    Mode::Naive => d.window(x, 1, &mut conv).unwrap(),
                    m => d.decode_fast(x, &mut conv, m).unwrap(),
                });
            }
            let us = t0.elapsed().as_secs_f64() * 1e6 / (steps - 8) as f64;
            eprintln!("kda decode layer {l} {mode:?}: {us:.0} us/step (real weights, {} steps timed)", steps - 8);
            let mut st = KdaState::zeros(&cfg, 1);
            let mut hc = conv.clone();
            d.read_back(&mut hc, &mut st).unwrap();
            outs.push(ys);
            states.push((hc, st));
        }
        {
            // per-stage GPU time of the fused step (eager, events between the launches)
            let d = mk(true);
            let ev: Vec<Event> = (0..5).map(|_| Event::new(true).unwrap()).collect();
            let gemv8 = d.gemv8 && [&d.wq, &d.wk, &d.wv].iter().all(|w| w.w.ty == 8);
            let mut acc = [0f32; 4];
            for x in &xs {
                d.xb.write(0, x).unwrap();
                d.enqueue_decode(&d.s, gemv8, Some(&ev)).unwrap();
                d.s.sync().unwrap();
                for i in 0..4 {
                    acc[i] += ev[i + 1].since(&ev[i]).unwrap();
                }
            }
            let k = xs.len() as f32;
            eprintln!(
                "kda decode layer {l} fused stages (GPU us/step, gemv8 {gemv8}): proj {:.0}, gate2 {:.0}, step {:.0}, o_proj {:.0}",
                acc[0] * 1e3 / k, acc[1] * 1e3 / k, acc[2] * 1e3 / k, acc[3] * 1e3 / k
            );
        }
        for (mi, name) in [(1, "fused"), (2, "graph")] {
            let diff = outs[mi]
                .iter()
                .zip(&outs[0])
                .flat_map(|(a, b)| a.iter().zip(b))
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let sdiff = states[mi].1 .0.iter().zip(&states[0].1 .0).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            let cdiff = states[mi].0 .0.iter().zip(&states[0].0 .0).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            eprintln!("{name} vs naive: {diff} output values differ (bits), state {sdiff}, conv {cdiff}");
            assert_eq!((diff, sdiff, cdiff), (0, 0, 0), "{name} decode is not bit-equal to naive on real weights");
        }
    }
}
