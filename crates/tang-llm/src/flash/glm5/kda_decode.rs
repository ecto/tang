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
//! go through `k5_q8` + `k5_gemv8` instead (the graph is keyed on that flag).
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
use tang_moe::gpu::{launch, Graph, Gpu, Module, Stream};
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
    fn enqueue_decode(&self, s: &Stream, gemv8: bool) -> Result<()> {
        let fast = self.fast.as_ref().context("kda fused decode: kernels not built")?;
        let (nh, hd, di, hidden) = (self.nh, self.hd, self.d_inner, self.hidden);
        let lb = self.cfg.gate_lower_bound.unwrap_or(f32::NAN);
        let f = std::mem::size_of::<f32>() as u64;
        let qkv = self.qkvb.ptr;
        let rows = 3 * di + 2 * hd + nh;
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
                    hd as u32, nh as u32, hidden as u32, r0 as u32],
            )?;
            launch(
                fast.gate2,
                grid(2 * di),
                (128, 1, 1),
                0,
                s,
                args![self.fb.w, self.gb.w, self.g1b.ptr, self.g2a.ptr, self.dt.ptr,
                    self.rate.ptr, self.gb_out.ptr, self.g2b.ptr, nh as u32, hd as u32, lb],
            )?;
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
        Ok(())
    }

    /// One decode token through the fused step (`mode` Fused or Graph). Same contract as
    /// [`DevLayer::window`] at `t = 1`, including the debug taps.
    pub(super) fn decode_fast(&mut self, x: &[f32], conv: &mut ConvState, mode: Mode) -> Result<Vec<f32>> {
        ensure!(x.len() == self.hidden, "k5 decode input");
        ensure!(mode != Mode::Naive, "decode_fast: naive mode");
        let (nh, di, hidden) = (self.nh, self.d_inner, self.hidden);
        let gemv8 = std::env::var("TANG_GLM5_GEMV8").is_ok_and(|v| v == "1");
        // the core's prefill path uses its own stream: make sure it has drained (it syncs
        // at the end of every call, so this is free; it keeps the ordering explicit)
        self.core.s.sync()?;
        // synchronous HtoD on the legacy stream; self.s is BLOCKING, so the launches (or
        // the graph) below are ordered after it
        self.xb.write(0, x)?;
        if self.debug && super::super::kda_debug::poison_on() {
            let nan = |n: usize| vec![f32::NAN; n];
            self.qkvb.write(0, &nan(3 * di))?;
            self.g1b.write(0, &nan(self.hd))?;
            self.gb_out.write(0, &nan(di))?;
            self.bb.write(0, &nan(nh))?;
            self.g2a.write(0, &nan(self.hd))?;
            self.g2b.write(0, &nan(di))?;
            self.ob.write(0, &nan(di))?;
            self.yb.write(0, &nan(di))?;
            self.outb.write(0, &nan(hidden))?;
        }
        match mode {
            Mode::Graph if self.fast.as_ref().and_then(|f| f.graph.as_ref()).is_some_and(|g| g.1 == gemv8) => {
                let f = self.fast.as_ref().unwrap();
                f.graph.as_ref().unwrap().0.launch(&self.s).map_err(|e| anyhow::anyhow!("kda graph launch: {e}"))?;
            }
            Mode::Graph => {
                // first token (or GEMV8 toggled): run eagerly, then capture for the next
                // tokens (capture records, it does not execute)
                self.enqueue_decode(&self.s, gemv8)?;
                self.s.sync()?;
                if let Some(f) = self.fast.as_mut() {
                    f.graph = None;
                }
                let me = &*self;
                let gr = Graph::capture(&me.s, |s| {
                    me.enqueue_decode(s, gemv8).map_err(|e| tang_moe::gpu::Error(format!("{e:#}")))
                })
                .map_err(|e| anyhow::anyhow!("kda decode capture: {e}"))?;
                self.fast.as_mut().unwrap().graph = Some((gr, gemv8));
            }
            _ => self.enqueue_decode(&self.s, gemv8)?,
        }
        self.s.sync()?;
        let out = self.outb.read::<f32>(0, hidden)?;
        if self.debug {
            let qkv_h = self.qkvb.read::<f32>(0, 3 * di)?;
            // the host conv shift (DevKda::decode's side effect), for the conv_host check
            let m = self.cfg.d_conv - 1;
            let ch = 3 * di;
            let mut next = conv.0.clone();
            for c in 0..ch {
                for s2 in 0..m {
                    next[c * m + s2] = if s2 + 1 < m { conv.0[c * m + s2 + 1] } else { qkv_h[c] };
                }
            }
            conv.0 = next;
            let tp = super::super::kda_debug::KdaTaps {
                qkv_pre: qkv_h,
                g: self.gb_out.read::<f32>(0, di)?,
                beta: self.bb.read::<f32>(0, nh)?,
                o: self.ob.read::<f32>(0, di)?,
                g2: self.g2b.read::<f32>(0, di)?,
                y_norm: self.yb.read::<f32>(0, di)?,
                out: out.clone(),
                ..Default::default()
            };
            self.taps = Some(tp);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::super::kda::{ConvState, KdaConfig, KdaState};
    use std::path::Path;

    const TINY: &str = "/private/tmp/glm5-tiny/glm5-tiny-f32.gguf";

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
}
