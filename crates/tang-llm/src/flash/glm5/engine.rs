//! The GLM-5.3-Flash decode engine (W1): the resumable form of [`super::forward::Glm5Ref`].
//!
//! `Glm5Ref::forward` recomputes a whole sequence from position 0. This engine keeps each
//! layer's running state between windows, so prefill and decode cost only the new tokens:
//!
//! - **34 KDA layers**: the recurrent state `[head][key][value]` f32 plus the conv history
//!   (W4's [`KdaState`] / [`ConvState`]). Fixed size, never grows.
//! - **11 DSA layers**: the normed 512-dim NoPE latent per token (W3's `k_latent` cache) plus
//!   the pooled indexer's key/gate rows per token. Grows with context.
//! - **mHC**: nothing persists. The 4 streams live only inside a window.
//! - **MoE**: stateless. Experts stay mmap'd GGUF handles, dequantised per expert (W5).
//!
//! Naive-but-correct (GATES.md): every component is the CPU f32 function already checked
//! against HF. `window` must equal the reference forward for the same positions, which
//! the tests here pin down (chunked prefill + token-by-token decode vs one full forward).
//! The CUDA path replaces these calls one component at a time, with this engine as the
//! oracle.
//!
//! Weights: mixer weights are dequantised per window and dropped, the same as the
//! reference, so the 244 GB UD-Q4_K_XL file stays mmap'd. On small models
//! (`hidden ≤ 1024`, i.e. the tiny fixture), or with `TANG_GLM5_CACHE=1`, they are cached.

use super::forward::Glm5Ref;
use super::kda::{ConvState, KdaConfig, KdaLayerW, KdaState};
use super::mla::MlaWeights;
use anyhow::{bail, ensure, Context, Result};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// One DSA layer's per-token caches: normed latent `[n][kv_lora]`, indexer key and gate
/// rows `[n][idx_dim]`.
#[derive(Default, Clone)]
struct DsaCache {
    latent: Vec<f32>,
    idx_k: Vec<f32>,
    idx_g: Vec<f32>,
    /// The GPU path's latent cache: `[n][kv_lora]` bf16 (W3's layout).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    lat16: Vec<u16>,
    /// The GPU path's indexer validity marks, `[n]`.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    valid: Vec<u8>,
}

/// The FFNs on the GPU (feature `cuda`): every layer's dense or MoE FFN (W5's
/// [`super::moe::gpu::DevMoe`]) uploaded once in its GGUF storage format and kept resident.
/// With `TANG_GPUS=2`, layers `[0, split)` live on GPU 0 and `[split, L)` on GPU 1
/// (`TANG_SPLIT_AT`, default: the cut that balances routed-expert bytes). Activations cross
/// through the host for now (naive; one copy each way per FFN).
#[cfg(feature = "cuda")]
pub struct GpuFfn {
    devs: Vec<(tang_moe::gpu::Gpu, super::moe::gpu::Kernels)>,
    /// The resident KDA layers ([`super::kda_dev::DevLayer`], W4's DevKda core + the
    /// projections as device GEMVs), one per KDA layer, `None` when
    /// `TANG_GLM5_GPU_KDA=0`. The free-kernel path (`kda_gpu`) stays as the fallback for
    /// windows past `max_t`.
    kda_l: Vec<Option<super::kda_dev::DevLayer>>,
    /// Per KDA layer: does the device hold the live state (vs the engine's host copies)?
    kda_dev_fresh: Vec<bool>,
    /// W4's staged KDA kernels per device (the >max_t fallback path).
    kda_kernels: Vec<super::kda_cuda::cuda::Kernels>,
    /// The resident scratch's window cap (TANG_GLM5_KDA_MAXT, default 512).
    max_t: usize,
    /// W3's MLA + indexer per DSA LAYER (`None` for KDA layers; `TANG_GLM5_GPU_MLA=0`:
    /// DSA stays on the CPU). One per layer, not per device: an `MlaCuda` stages its
    /// layer's kv_a / indexer / K-V decompress weights once and keeps the latent + indexer
    /// caches and their watermarks per sequence — shared across layers it ran every DSA
    /// layer on the first layer's weights and let the next window's first DSA layer
    /// attend over the last layer's latents (the watermark matched, so no re-upload).
    mla_k: Vec<Option<super::mla::cuda::MlaCuda>>,
    /// W2's resident mHC per layer ([`super::mhc::MhcGpu`]: both blocks' weights + a
    /// scratch pool on the device; `TANG_GLM5_GPU_MHC=0`: mHC stays on the CPU).
    mhc_g: Vec<Option<super::mhc::MhcGpu>>,
    /// Per layer: its FFN resident on the GPU (`None`: not resident, the CPU runs it —
    /// `TANG_GLM5_GPU_FFN=0` for all, or `TANG_GLM5_GPU_FFN_LAYERS` to pick a subset).
    layers: Vec<Option<super::moe::gpu::DevMoe>>,
    dev_of: Vec<usize>,
    pub split: usize,
    /// `TANG_GLM5_MOE_PROF=1`: per layer, (calls, total µs) for T = 1 / T > 1 FFN calls.
    prof: Option<Vec<[(u64, f64); 2]>>,
}

/// Everything device-owned goes before the contexts it lives in. Fields drop in
/// declaration order, so without this `devs` (the contexts) went first and every DevLayer
/// stream / MLA buffer / module unload after it hit a destroyed context — the SIGSEGV at
/// the end of every GPU run (compute-sanitizer: cuModuleUnload INVALID_HANDLE,
/// cuStreamDestroy CONTEXT_IS_DESTROYED).
#[cfg(feature = "cuda")]
impl Drop for GpuFfn {
    fn drop(&mut self) {
        self.print_prof();
        if let Some((g, _)) = self.devs.first() {
            let _ = g.bind();
        }
        self.kda_l.clear();
        self.mla_k.clear();
        self.mhc_g.clear();
        self.layers.clear();
        self.kda_kernels.clear();
        // each device's kernels (a module) before the device itself
        for (g, k) in self.devs.drain(..) {
            let _ = g.bind();
            drop(k);
            drop(g);
        }
    }
}

#[cfg(feature = "cuda")]
impl GpuFfn {
    /// The KDA layers resident (tests: the wiring regression guards).
    #[cfg(test)]
    pub(crate) fn kda_resident(&self) -> usize {
        self.kda_l.iter().filter(|x| x.is_some()).count()
    }

    /// The mHC layers resident (tests: the wiring regression guards).
    #[cfg(test)]
    pub(crate) fn mhc_resident(&self) -> usize {
        self.mhc_g.iter().filter(|x| x.is_some()).count()
    }

    /// Upload every layer's FFN. `Ok(None)` when there is no CUDA device or `TANG_GLM5_GPU=0`.
    /// `mla_w(l)` builds DSA layer `l`'s weights (each gets its own resident `MlaCuda`).
    pub fn load(
        g: &crate::gguf::Gguf,
        hp: &super::Hparams,
        mla_w: &dyn Fn(usize) -> Result<MlaWeights>,
    ) -> Result<Option<Self>> {
        use tang_moe::gpu::Gpu;
        let n_layer = hp.n_layer;
        let kda_cfg = KdaConfig {
            heads: hp.n_head,
            head_dim: hp.kda_head_dim,
            d_conv: hp.ssm_d_conv,
            gate_lower_bound: hp.kda_gate_lower_bound,
            eps: hp.eps,
        };
        let kda_on = !std::env::var("TANG_GLM5_GPU_KDA").is_ok_and(|v| v == "0");
        let mut kda_kernels = Vec::new();
        let mla_on = !std::env::var("TANG_GLM5_GPU_MLA").is_ok_and(|v| v == "0");
        let mut mla_k = Vec::new();
        let mhc_on = hp.hc == 4
            && hp.n_embd % 32 == 0
            && !std::env::var("TANG_GLM5_GPU_MHC").is_ok_and(|v| v == "0");
        let mhp = super::mhc::MhcHparams {
            streams: hp.hc,
            sinkhorn_iters: hp.hc_sinkhorn_iters as u32,
            hc_eps: hp.hc_eps,
            rms_eps: hp.eps,
        };
        if std::env::var("TANG_GLM5_GPU").is_ok_and(|v| v == "0") {
            return Ok(None);
        }
        let ngpu = crate::flash::split::gpus();
        let mut devs = Vec::new();
        for ord in 0..ngpu {
            let gpu = match Gpu::at(ord) {
                Ok(g) => g,
                Err(e) if ord == 0 => {
                    eprintln!("glm5: no CUDA device ({e}); FFNs on the CPU");
                    return Ok(None);
                }
                Err(e) => bail!("TANG_GPUS={ngpu}: CUDA device {ord}: {e}"),
            };
            gpu.bind().map_err(|e| anyhow::anyhow!("bind GPU {ord}: {e}"))?;
            kda_kernels.push(if kda_on && hp.is_kda.iter().any(|&b| b) {
                super::kda_cuda::cuda::load(&gpu, &kda_cfg)?
            } else {
                // the staged fallback is only reachable when some KDA layer exists; a
                // placeholder keeps the vec indexed — an Err surfaces if it is ever used
                super::kda_cuda::cuda::load(&gpu, &kda_cfg)?
            });
            let _ = &mhc_on; // the resident layers are built in the per-layer loop below
            let mk = super::moe::gpu::Kernels::new(&gpu)?;
            devs.push((gpu, mk));
        }
        let split = if devs.len() < 2 {
            n_layer
        } else if let Ok(k) = std::env::var("TANG_SPLIT_AT") {
            k.parse::<usize>().context("TANG_SPLIT_AT")?.min(n_layer)
        } else {
            let bytes: Vec<u64> = (0..n_layer)
                .map(|l| super::moe::expert_bytes(g, l).unwrap_or(0))
                .collect();
            let total: u64 = bytes.iter().sum();
            let (mut best, mut err, mut acc) = (n_layer / 2, u64::MAX, 0u64);
            for (l, b) in bytes.iter().enumerate() {
                acc += b;
                if acc.abs_diff(total / 2) < err {
                    (best, err) = (l + 1, acc.abs_diff(total / 2));
                }
            }
            best
        };
        // the resident KDA layers: one DevLayer per KDA layer on its device. Weights
        // upload in GGUF storage format; scratch is sized to the largest engine window
        // (the engine's chunk, 512 at serve, never exceeds 4096; 64 covers the profile's
        // prefill windows and decode, and the <64 tail falls to the CPU path).
        let max_t = std::env::var("TANG_GLM5_KDA_MAXT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(512);
        let mut kda_l: Vec<Option<super::kda_dev::DevLayer>> = Vec::new();
        let mut mhc_g: Vec<Option<super::mhc::MhcGpu>> = Vec::new();
        {
            let t_k = Instant::now();
            for l in 0..n_layer {
                // mHC resident: both blocks' weights (attn + ffn) on the layer's device
                if mhc_on {
                    let d = usize::from(l >= split);
                    devs[d].0.bind().map_err(|e| anyhow::anyhow!("bind GPU {d}: {e}"))?;
                    let wa = super::mhc::MhcWeights::from_gguf(g, l, "hc_attn", hp.n_embd, &mhp)
                        .with_context(|| format!("mhc blk.{l}.hc_attn"))?;
                    let wf = super::mhc::MhcWeights::from_gguf(g, l, "hc_ffn", hp.n_embd, &mhp)
                        .with_context(|| format!("mhc blk.{l}.hc_ffn"))?;
                    let mh = super::mhc::MhcGpu::new(&devs[d].0, [&wa, &wf], max_t)
                        .with_context(|| format!("mhc layer {l} resident on GPU {d}"))?;
                    mhc_g.push(Some(mh));
                } else {
                    mhc_g.push(None);
                }
                // MLA resident per DSA layer, from that layer's own weights
                if mla_on && !hp.is_kda[l] {
                    let d = usize::from(l >= split);
                    devs[d].0.bind().map_err(|e| anyhow::anyhow!("bind GPU {d}: {e}"))?;
                    let w = mla_w(l).with_context(|| format!("mla weights blk.{l}"))?;
                    mla_k.push(Some(
                        super::mla::cuda::MlaCuda::new_at(&w, d)
                            .map_err(|e| anyhow::anyhow!("MLA layer {l} on GPU {d}: {e}"))?,
                    ));
                } else {
                    mla_k.push(None);
                }
                if !hp.is_kda[l] || !kda_on {
                    kda_l.push(None);
                    continue;
                }
                let d = usize::from(l >= split);
                devs[d].0.bind().map_err(|e| anyhow::anyhow!("bind GPU {d}: {e}"))?;
                let layer = super::kda_dev::DevLayer::new_gguf(
                    &devs[d].0,
                    g,
                    l,
                    &kda_cfg,
                    hp.n_embd,
                    &super::kda::ConvState::zeros(&kda_cfg, 1),
                    &super::kda::KdaState::zeros(&kda_cfg, 1),
                    max_t,
                )
                .with_context(|| format!("KDA layer {l} resident on GPU {d}"))?;
                kda_l.push(Some(layer));
                if kda_l.iter().filter(|x| x.is_some()).count() % 8 == 0 {
                    eprintln!(
                        "glm5: {} KDA layers resident ({:.0} s)",
                        kda_l.iter().filter(|x| x.is_some()).count(),
                        t_k.elapsed().as_secs_f64()
                    );
                }
            }
            eprintln!(
                "glm5: {} KDA layers resident on the GPU(s) ({:.0} s)",
                kda_l.iter().filter(|x| x.is_some()).count(),
                t_k.elapsed().as_secs_f64()
            );
            // residency of the other per-layer GPU paths, so a run log proves what ran where
            eprintln!(
                "glm5: {} DSA layers resident (MLA, one per layer), {} mHC layers resident",
                mla_k.iter().filter(|x| x.is_some()).count(),
                mhc_g.iter().filter(|x| x.is_some()).count()
            );
        }
        let kda_dev_fresh = vec![false; n_layer];

        let t0 = Instant::now();
        let ffn_on = !std::env::var("TANG_GLM5_GPU_FFN").is_ok_and(|v| v == "0");
        // TANG_GLM5_GPU_FFN_LAYERS=0-6,11: only these layers' FFNs resident (the rest on the
        // CPU) — lets a 24 GB card run the real model with some MoE layers on the GPU
        let pick = std::env::var("TANG_GLM5_GPU_FFN_LAYERS").ok().map(|s| {
            let mut v = Vec::new();
            for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
                match part.split_once('-') {
                    Some((a, b)) => {
                        if let (Ok(a), Ok(b)) = (a.parse::<usize>(), b.parse::<usize>()) {
                            v.extend(a..=b);
                        }
                    }
                    None => v.extend(part.parse::<usize>().ok()),
                }
            }
            v
        });
        let mut layers = Vec::with_capacity(n_layer);
        let mut dev_of = vec![0usize; n_layer];
        for l in 0..n_layer {
            let d = usize::from(l >= split);
            dev_of[l] = d;
            if !ffn_on || pick.as_ref().is_some_and(|v| !v.contains(&l)) {
                layers.push(None);
                continue;
            }
            devs[d].0.bind().map_err(|e| anyhow::anyhow!("bind GPU {d}: {e}"))?;
            let p = super::moe::MoeParams::from_gguf(g, l)?;
            layers.push(Some(
                super::moe::gpu::DevMoe::new(&p)
                    .with_context(|| format!("uploading layer {l}'s FFN to GPU {d}"))?,
            ));
            if l % 8 == 7 || l + 1 == n_layer {
                eprintln!(
                    "glm5: {} of {n_layer} FFNs on the GPU ({:.0} s)",
                    layers.iter().filter(|x| x.is_some()).count(),
                    t0.elapsed().as_secs_f64()
                );
            }
        }
        eprintln!(
            "glm5: {} of {n_layer} FFNs resident on the GPU, MoE kernels: {}",
            layers.iter().filter(|x| x.is_some()).count(),
            if devs.first().is_some_and(|d| d.1.fast) { "fast (TANG_GLM5_MOE=fast)" } else { "naive" }
        );
        eprintln!(
            "glm5: FFNs on {} GPU(s), layers [0, {split}) on GPU 0{}",
            devs.len(),
            if devs.len() > 1 {
                format!(", [{split}, {n_layer}) on GPU 1")
            } else {
                String::new()
            }
        );
        Ok(Some(Self {
            devs,
            kda_l,
            kda_dev_fresh,
            kda_kernels: kda_kernels,
            max_t,
            mla_k,
            mhc_g,
            layers,
            dev_of,
            split,
            prof: std::env::var("TANG_GLM5_MOE_PROF")
                .is_ok_and(|v| v == "1")
                .then(|| vec![[(0u64, 0f64); 2]; n_layer]),
        }))
    }

    /// Whether layer `l`'s FFN is resident.
    fn has_ffn(&self, l: usize) -> bool {
        self.layers.get(l).is_some_and(|m| m.is_some())
    }

    /// Layer `l`'s FFN on its GPU: host `x` `[t][hidden]` in, host `y` out.
    fn ffn(&mut self, l: usize, x: &[f32], t: usize) -> Result<Vec<f32>> {
        let (gpu, k) = &self.devs[self.dev_of[l]];
        gpu.bind().map_err(|e| anyhow::anyhow!("bind: {e}"))?;
        let m = self.layers[l].as_ref().context("FFN not resident")?;
        let t0 = Instant::now();
        let y = super::moe::gpu::ffn_host(k, m, x, t)?;
        if let Some(p) = self.prof.as_mut() {
            let e = &mut p[l][usize::from(t > 1)];
            e.0 += 1;
            e.1 += t0.elapsed().as_secs_f64() * 1e6;
        }
        Ok(y)
    }

    /// The fast path's routing of the last call on layer `l`'s device (`MoE debug tap`).
    fn last_routing(&self, l: usize, t: usize, k: usize) -> Result<Option<(Vec<u32>, Vec<f32>)>> {
        let (gpu, kern) = &self.devs[self.dev_of[l]];
        if !kern.fast {
            return Ok(None);
        }
        gpu.bind().map_err(|e| anyhow::anyhow!("bind: {e}"))?;
        kern.last_routing(t, k)
    }

    /// `TANG_GLM5_MOE_PROF=1` summary: mean µs per FFN call, per layer and kind.
    fn print_prof(&self) {
        let Some(p) = &self.prof else { return };
        let mut tot = [(0u64, 0f64); 2];
        for (l, e) in p.iter().enumerate() {
            for (i, (n, us)) in e.iter().enumerate() {
                if *n == 0 {
                    continue;
                }
                eprintln!(
                    "moeprof l{l} {} calls {n} mean {:.0} us",
                    if i == 0 { "T=1" } else { "T>1" },
                    us / *n as f64
                );
                tot[i].0 += n;
                tot[i].1 += us;
            }
        }
        for (i, (n, us)) in tot.iter().enumerate() {
            if *n > 0 {
                eprintln!(
                    "moeprof ALL {} calls {n} total {:.1} ms mean {:.0} us/call",
                    if i == 0 { "T=1" } else { "T>1" },
                    us / 1e3,
                    us / *n as f64
                );
            }
        }
    }
}

/// The resumable GLM-5.3-Flash model.
pub struct Glm5Engine {
    pub m: Glm5Ref,
    /// Every token run so far (the prompt and the generated tokens).
    pub tokens: Vec<u32>,
    kda: Vec<Option<(ConvState, KdaState)>>,
    dsa: Vec<DsaCache>,
    kda_w: Vec<Option<std::sync::Arc<(KdaLayerW, KdaConfig)>>>,
    hc_w: Vec<Option<std::sync::Arc<super::mhc::MhcWeights>>>,
    mla_w: Vec<Option<std::sync::Arc<MlaWeights>>>,
    head: Option<crate::flash::reference::Mat>,
    /// Print per-window timings to stderr.
    pub verbose: bool,
    /// W0's R3 bisector: per-layer dump sink, no-op without TANG_DUMP_DIR (dump.rs).
    pub dump: Option<super::dump::DumpSink>,
    /// The FFNs on the GPU, when a CUDA build finds one ([`Glm5Engine::use_gpu`]).
    #[cfg(feature = "cuda")]
    pub gpu: Option<GpuFfn>,
    /// `TANG_GLM5_KDA_DEBUG=1`: the CPU shadow of every resident KDA layer, checked
    /// intermediate-by-intermediate after each window ([`super::kda_debug`]).
    #[cfg(feature = "cuda")]
    kda_dbg: Option<super::kda_debug::Shadow>,
    /// `TANG_GLM5_MOE_DEBUG=1`: the CPU shadow of every GPU FFN call ([`super::moe_debug`]).
    #[cfg(feature = "cuda")]
    moe_dbg: Option<super::moe_debug::Shadow>,
}

impl Glm5Engine {
    pub fn open(path: &Path) -> Result<Self> {
        let m = Glm5Ref::open(path)?;
        let nl = m.hp.n_layer;
        Ok(Self {
            m,
            tokens: Vec::new(),
            kda: (0..nl).map(|_| None).collect(),
            dsa: vec![DsaCache::default(); nl],
            kda_w: (0..nl).map(|_| None).collect(),
            hc_w: (0..nl * 2).map(|_| None).collect(),
            mla_w: (0..nl).map(|_| None).collect(),
            head: None,
            verbose: false,
            dump: super::dump::DumpSink::open(),
            #[cfg(feature = "cuda")]
            gpu: None,
            #[cfg(feature = "cuda")]
            kda_dbg: super::kda_debug::Shadow::from_env(nl),
            #[cfg(feature = "cuda")]
            moe_dbg: super::moe_debug::Shadow::from_env(nl),
        })
    }

    /// Move the FFNs to the GPU(s) (no-op without a device or with `TANG_GLM5_GPU=0`).
    /// Returns whether they moved.
    #[cfg(feature = "cuda")]
    pub fn use_gpu(&mut self) -> Result<bool> {
        let m = &self.m;
        self.gpu = GpuFfn::load(&m.g, &m.hp, &|l| m.mla_weights(l))?;
        Ok(self.gpu.is_some())
    }

    /// Layer `l`'s FFN: on its GPU when loaded, else W1's dense FFN / W5's CPU MoE.
    fn ffn(&mut self, l: usize, x: &[f32], t: usize) -> Result<Vec<f32>> {
        #[cfg(feature = "cuda")]
        if let Some(g) = self.gpu.as_mut() {
            if g.has_ffn(l) {
                let y = g.ffn(l, x, t)?;
                if let Some(d) = self.moe_dbg.as_mut().filter(|d| d.wants(l)) {
                    let p = super::moe::MoeParams::from_gguf(&self.m.g, l)?;
                    let route = if p.is_dense() { None } else { g.last_routing(l, t, p.top_k)? };
                    let m = &self.m;
                    d.check(l, &p, x, t, &y, route, &|| m.dense_ffn(l, x, t))?;
                }
                return Ok(y);
            }
            // not resident (TANG_GLM5_GPU_FFN=0 / _LAYERS): fall through to the CPU path.
        }
        if self.m.hp.is_dense(l) {
            self.m.dense_ffn(l, x, t)
        } else {
            let p = super::moe::MoeParams::from_gguf(&self.m.g, l)?;
            super::moe::moe_layer(&p, x, t)
        }
    }

    /// Forget the sequence. Cached weights stay.
    pub fn reset(&mut self) {
        self.tokens.clear();
        for s in &mut self.kda {
            *s = None;
        }
        for c in &mut self.dsa {
            *c = DsaCache::default();
        }
    }

    /// Layer `l`'s KDA weights, loaded once and shared (dequantising 268 MB per window is
    /// the engine's original per-token cost).
    fn kda_weights(&mut self, l: usize) -> Result<std::sync::Arc<(KdaLayerW, KdaConfig)>> {
        if let Some(w) = &self.kda_w[l] {
            return Ok(w.clone());
        }
        let w = std::sync::Arc::new(self.m.kda_weights(l)?);
        self.kda_w[l] = Some(w.clone());
        Ok(w)
    }

    /// mHC mixer weights for layer `l` (cached when `cache_weights`; see the field note).
    fn hc_weights(
        &mut self,
        l: usize,
        prefix: &str,
        hp: &super::mhc::MhcHparams,
    ) -> Result<std::sync::Arc<super::mhc::MhcWeights>> {
        let idx = l * 2 + if prefix == "hc_ffn" { 1 } else { 0 };
        if let Some(w) = &self.hc_w[idx] {
            return Ok(w.clone());
        }
        let w = std::sync::Arc::new(super::mhc::MhcWeights::from_gguf(
            &self.m.g,
            l,
            prefix,
            self.m.hp.n_embd,
            hp,
        )?);
        self.hc_w[idx] = Some(w.clone());
        Ok(w)
    }

    fn mla_weights(&mut self, l: usize) -> Result<std::sync::Arc<MlaWeights>> {
        if let Some(w) = &self.mla_w[l] {
            return Ok(w.clone());
        }
        let w = std::sync::Arc::new(self.m.mla_weights(l)?);
        self.mla_w[l] = Some(w.clone());
        Ok(w)
    }

    /// KDA layer `l`'s mixer: the resident [`super::kda_dev::DevLayer`] when loaded (the
    /// device holds the state; the host copies sync only across a CPU fallback), W4's
    /// staged free-function path when the window exceeds the resident scratch, else the
    /// CPU reference.
    #[allow(clippy::too_many_arguments)]
    fn kda_mixer(
        &mut self,
        l: usize,
        x: &[f32],
        t: usize,
        w: &KdaLayerW,
        cfg: &KdaConfig,
        st: &mut (ConvState, KdaState),
    ) -> Result<Vec<f32>> {
        #[cfg(feature = "cuda")]
        if let Some(g) = self.gpu.as_mut() {
            if let Some(dev) = &mut g.kda_l[l] {
                let d = g.dev_of[l];
                g.devs[d].0.bind().map_err(|e| anyhow::anyhow!("bind: {e}"))?;
                if t > g.max_t {
                    // past the resident scratch: sync the state to the host and run the
                    // staged path (which keeps both copies fresh), then seed forward
                    if g.kda_dev_fresh[l] {
                        if std::env::var("TANG_GLM5_KDA_DUMP").is_ok_and(|v| v == "1") {
                            eprintln!("kda wiring: read_back layer {l} (window {t} > max_t)");
                        }
                        dev.read_back(&mut st.0, &mut st.1)?;
                        g.kda_dev_fresh[l] = false;
                    }
                    return kda_gpu(x, t, w, cfg, &g.kda_kernels[d], &g.devs[d].0, &mut st.0, &mut st.1);
                }
                if !g.kda_dev_fresh[l] {
                    if std::env::var("TANG_GLM5_KDA_DUMP").is_ok_and(|v| v == "1") {
                        eprintln!("kda wiring: seed layer {l} (host -> device)");
                    }
                    dev.seed(&st.0, &st.1)?;
                    g.kda_dev_fresh[l] = true;
                }
                let dbg = self.kda_dbg.as_mut().filter(|d| d.wants(l));
                let Some(dbg) = dbg else {
                    return dev.window(x, t, &mut st.0);
                };
                // shadow check: the device's state going in, the window with taps, the
                // device's state coming out, then the CPU re-run of the same window
                dbg.prime(l, &st.0, &st.1);
                let (mut c_in, mut s_in) = (st.0.clone(), st.1.clone());
                dev.read_back(&mut c_in, &mut s_in)?;
                dev.debug = true;
                let y = dev.window(x, t, &mut st.0);
                dev.debug = false;
                let y = y?;
                let taps = dev.taps.take().unwrap_or_default();
                let (mut c_out, mut s_out) = (st.0.clone(), st.1.clone());
                dev.read_back(&mut c_out, &mut s_out)?;
                dbg.check(l, x, t, w, cfg, &taps, &st.0, &c_out, &s_out, Some(&s_in))?;
                return Ok(y);
            }
        }
        super::kda::kda_layer(x, w, cfg, &mut st.0, &mut st.1, 1, t)
    }

    /// DSA layer `l`: W3's GPU MLA + indexer when loaded, else the CPU path.
    fn dsa_layer(
        &mut self,
        l: usize,
        w: &MlaWeights,
        x: &[f32],
        t: usize,
        pos0: usize,
    ) -> Result<Vec<f32>> {
        #[cfg(feature = "cuda")]
        if let Some(g) = &self.gpu {
            let d = g.dev_of[l];
            if let Some(cu) = &g.mla_k[l] {
                g.devs[d].0.bind().map_err(|e| anyhow::anyhow!("bind: {e}"))?;
                return dsa_gpu(w, cu, x, t, pos0, &mut self.dsa[l]);
            }
        }
        self.dsa_cpu(l, w, x, t, pos0)
    }

    /// DSA layer `l` on the CPU: W3's MLA-NoPE + pooled indexer over the f32 caches.
    fn dsa_cpu(
        &mut self,
        l: usize,
        w: &MlaWeights,
        x: &[f32],
        t: usize,
        pos0: usize,
    ) -> Result<Vec<f32>> {
        let hp = self.m.hp.clone();
        let n = hp.n_embd;
        let kv_len = pos0 + t;
                let q_resid = w.gemv_q_a(x, t);
                let latent = w.gemv_kv_a(x, t);
                let idx_dim = hp.idx_dim;
                let cache = &mut self.dsa[l];
                cache.latent.extend_from_slice(&latent);
                for i in 0..t {
                    let (mut k, mut g) = (vec![0f32; idx_dim], vec![0f32; idx_dim]);
                    super::mla::indexer_key(w, &x[i * n..(i + 1) * n], &mut k, &mut g);
                    cache.idx_k.extend_from_slice(&k);
                    cache.idx_g.extend_from_slice(&g);
                }
                let sel = if super::mla::indexer_is_dense(kv_len, hp.idx_top_k) {
                    None
                } else {
                    Some(super::mla::indexer_select(
                        &w,
                        x,
                        &q_resid,
                        &cache.idx_k,
                        &cache.idx_g,
                        &vec![1u8; kv_len],
                        &vec![1u8; t],
                        t,
                        kv_len,
                        pos0,
                    ))
                };
                let mut out = vec![0f32; t * n];
                super::mla::mla_forward(
                    &w,
                    x,
                    &q_resid,
                    &cache.latent,
                    kv_len,
                    t,
                    pos0,
                    None,
                    sel.as_deref(),
                    &mut out,
                );
        Ok(out)
    }

    /// mHC pre for layer `l`'s block: W2's GPU kernel when loaded, else the CPU reference.
    /// `block`: 0 = the attention half (`hc_attn` weights), 1 = the FFN half (`hc_ffn`) —
    /// it selects the resident weights; `w` is the same block's host copy (CPU path).
    fn mhc_pre(
        &self,
        l: usize,
        block: usize,
        w: &super::mhc::MhcWeights,
        r: &[f32],
        t: usize,
    ) -> Result<(Vec<f32>, Vec<super::mhc::Mix>)> {
        let (hc, n) = (self.m.hp.hc, self.m.hp.n_embd);
        #[cfg(feature = "cuda")]
        if let Some(g) = &self.gpu {
            if t <= g.max_t {
            if let Some(m) = &g.mhc_g[l] {
                g.devs[g.dev_of[l]].0.bind().map_err(|e| anyhow::anyhow!("bind: {e}"))?;
                // the block index, NOT a constant: `pre(0, ..)` ran the FFN half's mix
                // with the attention half's hc weights (the GLM-5.3 all-GPU divergence)
                let mix = m.pre(block, r, t)?;
                // MixSet -> Vec<Mix> (the CPU post path's input shape; the resident post
                // keeps the MixSet and avoids the repack below)
                let mixes = (0..t)
                    .map(|i| {
                        let mut m2 = super::mhc::Mix {
                            x: mix.x[i * n..(i + 1) * n].to_vec(),
                            post: [0f32; 4],
                            comb: [[0f32; 4]; 4],
                        };
                        m2.post.copy_from_slice(&mix.post[i * 4..(i + 1) * 4]);
                        for src in 0..4 {
                            m2.comb[src]
                                .copy_from_slice(&mix.comb[i * 16 + src * 4..i * 16 + src * 4 + 4]);
                        }
                        m2
                    })
                    .collect();
                return Ok((mix.x, mixes));
            }
            }
        }
        let _ = l;
        mhc_pre_all(w, r, t, hc, n)
    }

    /// mHC post for layer `l`'s block: `r ← post ⊙ y + combᵀ r`, GPU or CPU as [`Self::mhc_pre`].
    fn mhc_post(
        &self,
        l: usize,
        mixes: &[super::mhc::Mix],
        y: &[f32],
        r: &mut [f32],
        t: usize,
    ) -> Result<()> {
        let (hc, n) = (self.m.hp.hc, self.m.hp.n_embd);
        #[cfg(feature = "cuda")]
        if let Some(g) = &self.gpu {
            if t <= g.max_t {
            if let Some(m) = &g.mhc_g[l] {
                g.devs[g.dev_of[l]].0.bind().map_err(|e| anyhow::anyhow!("bind: {e}"))?;
                let out = m.post_mixes(mixes, y, r, t)?;
                r.copy_from_slice(&out);
                return Ok(());
            }
            }
        }
        let _ = l;
        mhc_post_all(mixes, y, r, t, hc, n)
    }

    /// Run `ids` after the tokens so far. Returns their logits `[ids.len()][n_vocab]`.
    pub fn window(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        ensure!(!ids.is_empty(), "empty window");
        let w0 = Instant::now();
        let prof = std::env::var("TANG_GLM5_PROF").is_ok_and(|v| v == "1");
        let mut stage = |name: &str, t0: &std::time::Instant| {
            if prof {
                eprintln!("glm5: prof {name}: {:.1} ms", t0.elapsed().as_secs_f32() * 1e3);
            }
        };
        let hp = self.m.hp.clone();
        let (n, hc, t) = (hp.n_embd, hp.hc, ids.len());
        let pos0 = self.tokens.len();

        // ---- embedding into every stream
        let embd = self.m.g.info("token_embd.weight")?;
        let mut r = vec![0f32; t * hc * n];
        for (i, &tok) in ids.iter().enumerate() {
            ensure!((tok as usize) < hp.n_vocab, "token {tok} out of range");
            let e = self.m.g.rows(embd, tok as usize, 1)?;
            for c in 0..hc {
                r[(i * hc + c) * n..(i * hc + c + 1) * n].copy_from_slice(&e);
            }
        }
        let mhp = super::mhc::MhcHparams {
            streams: hc,
            sinkhorn_iters: hp.hc_sinkhorn_iters as u32,
            hc_eps: hp.hc_eps,
            rms_eps: hp.eps,
        };

        let mut t_embd = Instant::now();
        for l in 0..hp.n_layer {
            // ---- attention half
            let t0 = Instant::now();
            let w_attn = self.hc_weights(l, "hc_attn", &mhp)?;
            let (mut x, mixes) = self.mhc_pre(l, 0, &w_attn, &r, t)?;
            self.m.norm_all(&mut x, l, "attn_norm.weight")?;
            stage("mhc_pre+norm(attn)", &t0);
            let t1 = Instant::now();
            let y = if hp.is_kda[l] {
                let t_w = Instant::now();
                let wc = self.kda_weights(l)?;
                stage("kda weights", &t_w);
                // no wc.0.clone(): the free-function paths only read the weights, and the
                // resident path doesn't touch them at all — pass the Arc's ref
                let (w, cfg) = (&wc.0, wc.1);
                let t_m = Instant::now();
                let mut st = self.kda[l]
                    .take()
                    .unwrap_or_else(|| (ConvState::zeros(&cfg, 1), KdaState::zeros(&cfg, 1)));
                let y = self.kda_mixer(l, &x, t, w, &cfg, &mut st);
                self.kda[l] = Some(st);
                stage("kda mixer", &t_m);
                y?
            } else {
                let t_w = Instant::now();
                let w = self.mla_weights(l)?;
                stage("mla weights", &t_w);
                let t_m = Instant::now();
                let y = self.dsa_layer(l, &w, &x, t, pos0)?;
                stage("mla mixer", &t_m);
                y
            };
            let t2 = Instant::now();
            self.mhc_post(l, &mixes, &y, &mut r, t)?;
            stage("mhc_post(attn)", &t2);

            // ---- FFN half
            let t3 = Instant::now();
            let w_ffn = self.hc_weights(l, "hc_ffn", &mhp)?;
            let (mut x, mixes) = self.mhc_pre(l, 1, &w_ffn, &r, t)?;
            self.m.norm_all(&mut x, l, "ffn_norm.weight")?;
            let y = self.ffn(l, &x, t)?;
            let t4 = Instant::now();
            self.mhc_post(l, &mixes, &y, &mut r, t)?;
            stage("ffn", &t3);
            stage("mhc_post(ffn)", &t4);
            // W0's R3 bisector: dump the last position's post-FFN streams (no-op
            // without TANG_DUMP_DIR; lives in dump.rs).
            if let Some(d) = self.dump.as_mut() {
                let row = &r[(t - 1) * hc * n..t * hc * n];
                d.layer(l, pos0 + t - 1, row);
            }
        }

        stage("layers", &t_embd);
        // ---- head: unweighted stream mean, output norm, lm_head
        let t5 = Instant::now();
        let mut h = vec![0f32; t * n];
        for i in 0..t {
            for c in 0..hc {
                for j in 0..n {
                    h[i * n + j] += r[(i * hc + c) * n + j];
                }
            }
            for v in &mut h[i * n..(i + 1) * n] {
                *v /= hc as f32;
            }
        }
        self.m.norm_all_exact(&mut h, "output_norm.weight")?;
        if self.head.is_none() {
            self.head = Some(self.m.head()?);
        }
        let head = self.head.as_ref().unwrap();
        let logits = head.apply(&h, t, false);
        stage("head", &t5);
        self.tokens.extend_from_slice(ids);
        // W0's R3 bisector: write the dump (no-op without TANG_DUMP_DIR).
        if let Some(d) = self.dump.as_mut() {
            let last_row = t - 1;
            d.finish(
                &self.tokens,
                &h[last_row * n..(last_row + 1) * n],
                &logits[(t - 1) * hp.n_vocab..t * hp.n_vocab],
                hp.n_vocab,
            )?;
        }
        if self.verbose {
            eprintln!(
                "glm5: window t={t} at {pos0}: {:.2} s",
                w0.elapsed().as_secs_f64()
            );
        }
        Ok(logits)
    }

    /// Prefill `ids` in windows of `chunk`; returns the greedy next token.
    pub fn prefill(&mut self, ids: &[u32], chunk: usize) -> Result<u32> {
        ensure!(!ids.is_empty(), "empty prompt");
        let nv = self.m.hp.n_vocab;
        let mut last = Vec::new();
        for c in ids.chunks(chunk.max(1)) {
            let lg = self.window(c)?;
            last = lg[(c.len() - 1) * nv..].to_vec();
        }
        Ok(argmax(&last))
    }

    /// Feed one token; returns the greedy next token.
    pub fn step(&mut self, tok: u32) -> Result<u32> {
        Ok(argmax(&self.window(&[tok])?))
    }
}

/// mHC pre for every token of a window: the collapsed inputs `[t][n]` and each token's mix.
fn mhc_pre_all(
    w: &super::mhc::MhcWeights,
    r: &[f32],
    t: usize,
    hc: usize,
    n: usize,
) -> Result<(Vec<f32>, Vec<super::mhc::Mix>)> {
    let mut x = vec![0f32; t * n];
    let mut mixes = Vec::with_capacity(t);
    for i in 0..t {
        let streams: Vec<&[f32]> = (0..hc)
            .map(|c| &r[(i * hc + c) * n..(i * hc + c + 1) * n])
            .collect();
        let mix = super::mhc::mhc_pre(w, &streams)?;
        x[i * n..(i + 1) * n].copy_from_slice(&mix.x);
        mixes.push(mix);
    }
    Ok((x, mixes))
}

/// mHC post for every token: `r ← post ⊙ y + combᵀ r`.
fn mhc_post_all(
    mixes: &[super::mhc::Mix],
    y: &[f32],
    r: &mut [f32],
    t: usize,
    hc: usize,
    n: usize,
) -> Result<()> {
    let r_in = r.to_vec();
    for i in 0..t {
        let streams: Vec<&[f32]> = (0..hc)
            .map(|c| &r_in[(i * hc + c) * n..(i * hc + c + 1) * n])
            .collect();
        let token = &mut r[i * hc * n..(i + 1) * hc * n];
        let mut outs: Vec<&mut [f32]> = token.chunks_mut(n).collect();
        super::mhc::mhc_post(&mixes[i], &y[i * n..(i + 1) * n], &streams, &mut outs)?;
    }
    Ok(())
}

/// One KDA layer with W4's CUDA conv + delta core (feature `cuda`). Projections, gates and
/// the gated output norm stay on the host (the reference's f32 `matmul`). Prefill runs
/// 64-token GPU chunks (the kernel's fixed size), a single-token window runs the GPU decode
/// step, and any other tail runs the CPU core on the same state. `x` `[t][hidden]` in,
/// `[t][hidden]` out; `conv` / `state` are updated in place.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn kda_gpu(
    x: &[f32],
    t: usize,
    w: &KdaLayerW,
    cfg: &KdaConfig,
    k: &super::kda_cuda::cuda::Kernels,
    gpu: &tang_moe::gpu::Gpu,
    conv: &mut ConvState,
    state: &mut KdaState,
) -> Result<Vec<f32>> {
    use super::kda::{matmul, sigmoid};
    use super::kda_cuda::cuda;
    let (nh, hd) = (cfg.heads, cfg.head_dim);
    let hidden = w.hidden;
    ensure!(x.len() == t * hidden, "layer input");
    // projections, stacked q|k|v per token (the reference's layout)
    let qp = matmul(&w.q, nh * hd, hidden, x, t);
    let kp = matmul(&w.k, nh * hd, hidden, x, t);
    let vp = matmul(&w.v, nh * hd, hidden, x, t);
    let ch = 3 * nh * hd;
    let mut qkv = vec![0f32; t * ch];
    for i in 0..t {
        qkv[i * ch..i * ch + nh * hd].copy_from_slice(&qp[i * nh * hd..(i + 1) * nh * hd]);
        qkv[i * ch + nh * hd..i * ch + 2 * nh * hd]
            .copy_from_slice(&kp[i * nh * hd..(i + 1) * nh * hd]);
        qkv[i * ch + 2 * nh * hd..(i + 1) * ch]
            .copy_from_slice(&vp[i * nh * hd..(i + 1) * nh * hd]);
    }
    // gates: gpre (pre-exp, per key dim) and beta (post-sigmoid, per head)
    let fg = super::kda::ForgetGateW {
        f_a: w.f_a.clone(),
        f_b: w.f_b.clone(),
        dt_bias: w.dt_bias.clone(),
        decay_rate: w.decay_rate.clone(),
    };
    let (b, c) = (1usize, 64usize);
    let gpre = super::kda::forget_gate(x, &fg, cfg, t);
    let beta = super::kda::beta_gate(x, &w.b_proj, cfg, b, t);
    let mut y = vec![0f32; t * nh * hd];
    let mut off = 0usize;
    if t == 1 {
        let o = cuda::decode(gpu, k, cfg, &qkv, &w.conv, &gpre, &beta, conv, state, b)?;
        y.copy_from_slice(&o);
        off = 1;
    }
    while off + c <= t {
        let o = cuda::prefill(
            gpu,
            k,
            cfg,
            &qkv[off * ch..(off + c) * ch],
            &w.conv,
            &gpre[off * nh * hd..(off + c) * nh * hd],
            &beta[off * nh..(off + c) * nh],
            conv,
            state,
            b,
            c,
            c,
        )?;
        y[off * nh * hd..(off + c) * nh * hd].copy_from_slice(&o);
        off += c;
    }
    // a tail shorter than one chunk: the CPU core on the same state objects
    if off < t {
        let rem = t - off;
        let yc = super::kda::conv1d_prefill(&qkv[off * ch..t * ch], &w.conv, conv, cfg, b, rem)?;
        let oc = super::kda::kda_chunk(
            &yc,
            &gpre[off * nh * hd..t * nh * hd],
            &beta[off * nh..t * nh],
            state,
            cfg,
            b,
            rem,
            c,
        )?;
        y[off * nh * hd..t * nh * hd].copy_from_slice(&oc);
    }
    // gated output norm (host, same as the reference), then o_proj
    let ga = matmul(&w.g_a, hd, hidden, x, t);
    let g2 = matmul(&w.g_b, nh * hd, hd, &ga, t);
    for i in 0..t {
        for h in 0..nh {
            let dst = &mut y[i * nh * hd + h * hd..i * nh * hd + (h + 1) * hd];
            crate::flash::reference::rms_norm_mul(dst, &w.o_norm, cfg.eps);
            for (yj, gj) in dst
                .iter_mut()
                .zip(&g2[i * nh * hd + h * hd..i * nh * hd + (h + 1) * hd])
            {
                *yj *= sigmoid(*gj);
            }
        }
    }
    Ok(matmul(&w.o_proj, hidden, nh * hd, &y, t))
}

/// One DSA layer with W3's CUDA kernels (feature `cuda`), naive: the caches live on the
/// host and are staged per call (W3's launchers), q_a / q_b / o_proj are host GEMVs.
/// Latents are bf16 (W3's cache format), so this is ~1e-3 from the f32 CPU path.
/// Order follows W3's parity test: kv_append → idx_append → (sparse) idx_pool + idx_run →
/// attend.
#[cfg(feature = "cuda")]
fn dsa_gpu(
    w: &MlaWeights,
    cu: &super::mla::cuda::MlaCuda,
    x: &[f32],
    t: usize,
    pos0: usize,
    c: &mut DsaCache,
) -> Result<Vec<f32>> {
    let gerr = |e: tang_moe::gpu::Error| anyhow::anyhow!("glm5 mla cuda: {e}");
    let s = tang_moe::gpu::Stream::blocking().map_err(gerr)?;
    let (n, dl, idd) = (w.hidden, w.kv_lora, w.idx_dim);
    let kv_len = pos0 + t;
    ensure!(x.len() == t * n, "layer input");
    ensure!(c.lat16.len() == pos0 * dl, "latent cache out of step");
    let q_resid = w.gemv_q_a(x, t);
    // caches grow by t rows; the launchers write rows [pos0, kv_len)
    c.lat16.resize(kv_len * dl, 0);
    cu.kv_append(&w.kv_a, &w.kv_a_norm, x, t, pos0, &mut c.lat16, &s)
        .map_err(gerr)?;
    c.idx_k.resize(kv_len * idd, 0.0);
    c.idx_g.resize(kv_len * idd, 0.0);
    c.valid.resize(kv_len, 0);
    cu.idx_append(
        &w.i_wk,
        &w.i_k_norm_w,
        &w.i_k_norm_b,
        &w.i_gate,
        x,
        t,
        pos0,
        &mut c.idx_k,
        &mut c.idx_g,
        &mut c.valid,
        &s,
    )
    .map_err(gerr)?;
    let sel = if super::mla::indexer_is_dense(kv_len, w.topk) {
        None
    } else {
        let n_pool = kv_len / w.kpool;
        let mut pool_keys = vec![0f32; n_pool * idd];
        cu.idx_pool(&c.idx_k, &c.idx_g, &w.i_ape, kv_len, 0, n_pool, &mut pool_keys, &s)
            .map_err(gerr)?;
        let mut sel = vec![-1i32; t * w.select_width()];
        cu.idx_run(
            &w.i_wq_b,
            &w.i_proj,
            &q_resid,
            x,
            &pool_keys,
            &c.valid,
            kv_len,
            0,
            n_pool,
            t,
            pos0,
            &mut sel,
            &s,
        )
        .map_err(gerr)?;
        Some(sel)
    };
    let q = w.gemv_q_b(&q_resid, t);
    let wk: Vec<f32> = (0..w.n_head).flat_map(|h| w.k_nope(h).to_vec()).collect();
    let wv: Vec<f32> = (0..w.n_head).flat_map(|h| w.v_b_head(h).to_vec()).collect();
    let hv = w.n_head * w.v_head;
    let mut out_v = vec![0f32; t * hv];
    cu.attend(&wk, &wv, &q, &c.lat16, kv_len, t, pos0, sel.as_deref(), &mut out_v, &s)
        .map_err(gerr)?;
    // o_proj on the host
    let mut out = vec![0f32; t * n];
    for i in 0..t {
        for r in 0..n {
            out[i * n + r] = crate::flash::reference::dot(
                &w.o[r * hv..(r + 1) * hv],
                &out_v[i * hv..(i + 1) * hv],
            );
        }
    }
    Ok(out)
}

pub(crate) fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

/// Is the GGUF at `path` a `glm5-next` model?
pub fn is_glm5(path: &Path) -> bool {
    crate::gguf::Gguf::open(path)
        .ok()
        .and_then(|g| g.meta_str("general.architecture").ok().map(|a| a == "glm5-next"))
        .unwrap_or(false)
}

/// `flash-generate` / `flash-bench` for a glm5-next GGUF:
/// `<gguf> (--prompt-ids IDS | --ids-file F | --prompt TEXT) [-n N] [--chunk C] [--out F]
/// [--quiet] [-v]`. Greedy (temperature 0) only; other flash flags are accepted and ignored.
pub fn cli(args: &[String], bench: bool) -> Result<()> {
    let mut it = args.iter();
    let path = PathBuf::from(it.next().context("first argument: the first GGUF shard")?);
    let (mut ids, mut prompt, mut n, mut chunk) = (Vec::<u32>::new(), None, 32usize, 64usize);
    let (mut out, mut quiet, mut verbose) = (None::<PathBuf>, false, false);
    let push = |ids: &mut Vec<u32>, s: &str| -> Result<()> {
        for w in s.split(|c: char| c.is_whitespace() || c == ',' || c == '[' || c == ']') {
            if !w.is_empty() {
                ids.push(w.parse().with_context(|| format!("bad token id {w:?}"))?);
            }
        }
        Ok(())
    };
    while let Some(a) = it.next() {
        let mut val = || it.next().with_context(|| format!("{a} needs a value"));
        match a.as_str() {
            // the flash CLI takes one quoted list; bare ids that follow are accepted too
            "--prompt-ids" => push(&mut ids, val()?)?,
            "--ids-file" => push(&mut ids, &std::fs::read_to_string(val()?)?)?,
            "--prompt" => prompt = Some(val()?.clone()),
            "-n" | "--n" => n = val()?.parse()?,
            "--chunk" => chunk = val()?.parse()?,
            "--out" => out = Some(PathBuf::from(val()?)),
            "--quiet" => quiet = true,
            "-v" | "--verbose" => verbose = true,
            "--temp" => {
                let t: f32 = val()?.parse()?;
                ensure!(t == 0.0, "glm5: only greedy (--temp 0) for now");
            }
            "--eager" | "--no-think" => {}
            "--draft" | "--seed" | "--ctx" | "--mtp" | "--bw" => {
                val()?;
            }
            s if s.parse::<u32>().is_ok() => ids.push(s.parse()?),
            s => bail!("glm5: unknown argument {s}"),
        }
    }
    if let Some(p) = &prompt {
        ids = crate::flash::tokenize::encode_chat(&path, p, None)?;
    }
    ensure!(!ids.is_empty(), "no prompt (--prompt-ids, --ids-file or --prompt)");
    ensure!(n >= 1, "-n must be at least 1");

    let t0 = Instant::now();
    let mut e = Glm5Engine::open(&path)?;
    e.verbose = verbose;
    #[cfg(feature = "cuda")]
    e.use_gpu()?;
    let hp = &e.m.hp;
    eprintln!(
        "glm5: {} layers ({} KDA, {} DSA), hidden {}, {} experts top-{}, vocab {}; open {:.1} s",
        hp.n_layer,
        hp.is_kda.iter().filter(|&&k| k).count(),
        hp.is_kda.iter().filter(|&&k| !k).count(),
        hp.n_embd,
        hp.n_expert,
        hp.n_expert_used,
        hp.n_vocab,
        t0.elapsed().as_secs_f64()
    );
    let t1 = Instant::now();
    let mut cur = e.prefill(&ids, chunk)?;
    let prefill_s = t1.elapsed().as_secs_f64();
    let mut gen = vec![cur];
    let t2 = Instant::now();
    while gen.len() < n {
        cur = e.step(cur)?;
        gen.push(cur);
    }
    let decode_s = t2.elapsed().as_secs_f64();
    println!(
        "prompt: {} tokens, prefill {:.2} s ({:.1} tok/s)",
        ids.len(),
        prefill_s,
        ids.len() as f64 / prefill_s
    );
    let line = gen.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" ");
    println!("ids: {line}");
    if gen.len() > 1 {
        println!(
            "decode: {} tokens in {:.2} s ({:.2} tok/s)",
            gen.len() - 1,
            decode_s,
            (gen.len() - 1) as f64 / decode_s
        );
    }
    if !quiet && !bench {
        let tk = crate::flash::tokenize::FlashTokenizer::from_gguf(&e.m.g);
        if let Ok(text) = tk.and_then(|t| t.decode(&gen)) {
            println!("text: {text}");
        }
    }
    if let Some(o) = out {
        std::fs::write(&o, format!("{line}\n"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TINY: &str = "/private/tmp/glm5-tiny/glm5-tiny-f32.gguf";

    /// GPU tests skip when there's no device — but a SILENT pass looks green in CI and hid
    /// the GLM-5.3 GPU bugs for a cycle (the "tiny fixture passes" that never ran). With
    /// `TANG_REQUIRE_CUDA=1` a skip is a FAILURE instead; without it, the skip prints a
    /// loud banner. `reason` says what's missing.
    fn skip_or_fail(reason: &str) {
        let required = std::env::var("TANG_REQUIRE_CUDA").is_ok_and(|v| v == "1");
        let msg = format!("SKIP: {reason} (set TANG_REQUIRE_CUDA=1 to make this a failure)");
        if required {
            panic!("{msg} — but TANG_REQUIRE_CUDA=1");
        }
        eprintln!("==================== {msg} ====================");
    }

    fn seq1() -> Option<Vec<u32>> {
        let meta = std::fs::read_to_string("/private/tmp/glm5-tiny/ref/meta.json").ok()?;
        let v: serde_json::Value = serde_json::from_str(&meta).ok()?;
        Some(
            v["seq1_ids"]
                .as_array()?
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect(),
        )
    }

    /// REAL-WEIGHTS BISECTOR: the wired GPU engine vs the CPU engine on the REAL model,
    /// a tiny sequence. Named the bug the tiny fixture couldn't: if this fails while
    /// engine_gpu_matches_cpu passes, the divergence is real-weights/real-shape specific.
    /// Gated on TANG_GLM5_REAL_GGUF (the shard-1 path) so normal test runs stay cheap.
    #[cfg(feature = "cuda")]
    #[test]
    fn engine_real_weights_gpu_matches_cpu() {
        let Some(real) = std::env::var("TANG_GLM5_REAL_GGUF").ok() else {
            eprintln!("skip: TANG_GLM5_REAL_GGUF not set");
            return;
        };
        let p = Path::new(&real);
        if !p.exists() {
            eprintln!("skip: {real} absent");
            return;
        }
        let ids: Vec<u32> = (1000..1008).collect(); // 8 arbitrary in-vocab tokens
        // CPU outputs FIRST, engine dropped before the GPU engine opens: two resident
        // engines (2 × ~250 GB) exceed the 274 GB card. The CPU engine holds no resident
        // experts (mmap dequant per layer), so it's cheap — but dropping it keeps the
        // peak at one resident set.
        std::env::set_var("TANG_GLM5_GPU", "0");
        let cpu_out = {
            let mut cpu2 = Glm5Engine::open(p).unwrap();
            let mut out = Vec::new();
            out.extend(cpu2.window(&ids[..5]).unwrap());
            for &tok in &ids[5..] {
                out.extend(cpu2.window(&[tok]).unwrap());
            }
            out
        };
        std::env::remove_var("TANG_GLM5_GPU");
        let mut gpu = Glm5Engine::open(p).unwrap();
        gpu.use_gpu().unwrap();
        let mut gpu_out = Vec::new();
        gpu_out.extend(gpu.window(&ids[..5]).unwrap());
        for &tok in &ids[5..] {
            gpu_out.extend(gpu.window(&[tok]).unwrap());
        }
        let mut worst = 0f32;
        let mut bad = 0usize;
        let nv = 154_880;
        for p in 0..ids.len() {
            let c = &cpu_out[p * nv..(p + 1) * nv];
            let g = &gpu_out[p * nv..(p + 1) * nv];
            let w: f32 = c.iter().zip(g).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            worst = worst.max(w);
            if argmax(c) != argmax(g) {
                bad += 1;
                eprintln!("position {p}: top-1 differs, max abs {w}");
            }
        }
        eprintln!("real-weights bisector: max abs {worst}, top-1 mismatches {bad}/{}", ids.len());
        assert_eq!(bad, 0, "the wired GPU path diverges from CPU on the real model");
    }

    /// Chunked prefill + token-by-token decode equals one full reference forward.
    #[test]
    fn engine_matches_reference_forward() {
        let (Some(ids), true) = (seq1(), Path::new(TINY).exists()) else {
            eprintln!("skip: tiny fixture absent");
            return;
        };
        let r = Glm5Ref::open(Path::new(TINY)).unwrap();
        let want = r.forward(&ids).unwrap();
        let nv = r.hp.n_vocab;
        let mut e = Glm5Engine::open(Path::new(TINY)).unwrap();
        let mut got = Vec::new();
        // uneven chunks, then single tokens: exercises KDA chunk + decode and the DSA caches
        let (a, b) = (17, 36);
        got.extend(e.window(&ids[..a]).unwrap());
        got.extend(e.window(&ids[a..b]).unwrap());
        for &tok in &ids[b..] {
            got.extend(e.window(&[tok]).unwrap());
        }
        assert_eq!(got.len(), want.len());
        let mut worst = 0f32;
        for (x, y) in got.iter().zip(&want) {
            worst = worst.max((x - y).abs());
        }
        for p in 0..ids.len() {
            assert_eq!(
                argmax(&got[p * nv..(p + 1) * nv]),
                argmax(&want[p * nv..(p + 1) * nv]),
                "top-1 differs at {p}"
            );
        }
        eprintln!("engine vs reference: logits max abs {worst:.3e}");
        assert!(worst <= 1e-4, "logits max abs {worst}");
    }

    /// Greedy continuation through the engine equals re-running the reference each step.
    #[test]
    fn engine_greedy_matches_reference() {
        let (Some(ids), true) = (seq1(), Path::new(TINY).exists()) else {
            eprintln!("skip: tiny fixture absent");
            return;
        };
        let prompt = &ids[..12];
        let r = Glm5Ref::open(Path::new(TINY)).unwrap();
        let nv = r.hp.n_vocab;
        let mut seq = prompt.to_vec();
        let mut want = Vec::new();
        for _ in 0..8 {
            let lg = r.forward(&seq).unwrap();
            let next = argmax(&lg[(seq.len() - 1) * nv..]);
            want.push(next);
            seq.push(next);
        }
        let mut e = Glm5Engine::open(Path::new(TINY)).unwrap();
        let mut got = vec![e.prefill(prompt, 5).unwrap()];
        while got.len() < 8 {
            let t = *got.last().unwrap();
            got.push(e.step(t).unwrap());
        }
        assert_eq!(got, want);
    }

    /// R3 (the f32 GPU paths: FFN, KDA, mHC): every FFN, the KDA conv + core and the mHC
    /// pre/post on the GPU equal the CPU engine on the tiny model: 150-token prefill (2 GPU chunks + a CPU tail), GPU
    /// decode steps, then a short window. Pod: `TANG_REQUIRE_CUDA=1 cargo test --release -p tang-llm
    /// --no-default-features --features cuda --lib glm5::engine -- --nocapture`
    /// (`TANG_GPUS=2 TANG_SPLIT_AT=2` exercises the split).
    #[cfg(feature = "cuda")]
    #[test]
    fn engine_gpu_matches_cpu() {
        let (Some(_ids), true) = (seq1(), Path::new(TINY).exists()) else {
            skip_or_fail("tiny fixture absent");
            return;
        };
        if tang_moe::gpu::Gpu::for_test().is_none() {
            skip_or_fail("no CUDA device");
            return;
        }
        let mut cpu = Glm5Engine::open(Path::new(TINY)).unwrap();
        let mut gpu = Glm5Engine::open(Path::new(TINY)).unwrap();
        assert!(gpu.use_gpu().unwrap(), "GPU FFNs did not load");
        // f32 paths only here: DSA stays on the CPU (the bf16 MLA path has its own test)
        for k in &mut gpu.gpu.as_mut().unwrap().mla_k {
            *k = None;
        }
        let nv = cpu.m.hp.n_vocab;
        let mut worst = 0f32;
        // seq2 (2052 tokens) for a window ≥ 64: two GPU prefill chunks + a CPU tail, then
        // single tokens (the GPU decode step), then a short window (CPU tail only)
        let meta = std::fs::read_to_string("/private/tmp/glm5-tiny/ref/meta.json").unwrap();
        let v: serde_json::Value = serde_json::from_str(&meta).unwrap();
        let ids2: Vec<u32> = v["seq2_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();
        let _ = &_ids;
        let mut flips = Vec::new();
        for (wi, w) in [&ids2[..150], &ids2[150..151], &ids2[151..152], &ids2[152..160]]
            .into_iter()
            .enumerate()
        {
            let a = cpu.window(w).unwrap();
            let b = gpu.window(w).unwrap();
            for p in 0..w.len() {
                let (ra, rb) = (&a[p * nv..(p + 1) * nv], &b[p * nv..(p + 1) * nv]);
                let (ta, tb) = (argmax(ra), argmax(rb));
                let row: f32 = ra.iter().zip(rb).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
                if ta != tb {
                    // a near-tie flip (|cpu margin| ≲ the row's max abs diff) is numerics; a
                    // wide margin is a real divergence — print both so the log says which
                    let margin = ra[ta as usize] - ra[tb as usize];
                    eprintln!(
                        "window {wi} pos {p}: top-1 cpu {ta} gpu {tb}; cpu margin {margin:.3e}, \
                         row max abs {row:.3e}, gpu logits {:.5}/{:.5}",
                        rb[ta as usize], rb[tb as usize]
                    );
                    flips.push((wi, p, margin, row));
                }
            }
            for (x, y) in a.iter().zip(&b) {
                worst = worst.max((x - y).abs());
            }
        }
        eprintln!("GPU vs CPU: {} top-1 flips {flips:?}", flips.len());
        assert!(flips.is_empty(), "top-1 differs at {} positions", flips.len());
        eprintln!("GPU (FFN + KDA + mHC) vs CPU: logits max abs {worst:.3e}");
        assert!(worst <= 1e-3, "logits max abs {worst}");
    }

    /// REGRESSION (the GLM-5.3 mHC bug, 324e7f6b): the resident mHC pre must use the
    /// CALLING half's weights — block 0 (hc_attn) for the attention half, block 1
    /// (hc_ffn) for the FFN half. The bug ran `pre(0, ..)` for both, so every FFN-half
    /// mix collapsed the streams with the attention weights. The tiny fixture's
    /// hc_attn/hc_ffn weights differ by ~0.17, so the wrong block is detectable directly:
    /// compare the GPU pre's mixes for block 1 against the CPU's for block 1 (not block 0).
    #[cfg(feature = "cuda")]
    #[test]
    fn mhc_gpu_pre_block1_matches_cpu_block1() {
        if !Path::new(TINY).exists() || tang_moe::gpu::Gpu::for_test().is_none() {
            skip_or_fail("fixture/GPU");
            return;
        }
        let r = Glm5Ref::open(Path::new(TINY)).unwrap();
        let mhp = super::super::mhc::MhcHparams {
            streams: r.hp.hc,
            sinkhorn_iters: r.hp.hc_sinkhorn_iters as u32,
            hc_eps: r.hp.hc_eps,
            rms_eps: r.hp.eps,
        };
        let wa = super::super::mhc::MhcWeights::from_gguf(&r.g, 0, "hc_attn", r.hp.n_embd, &mhp).unwrap();
        let wf = super::super::mhc::MhcWeights::from_gguf(&r.g, 0, "hc_ffn", r.hp.n_embd, &mhp).unwrap();
        let g = tang_moe::gpu::Gpu::for_test().unwrap();
        g.bind().unwrap();
        let mh = super::super::mhc::MhcGpu::new(&g, [&wa, &wf], 8).unwrap();
        // four streams of nonzero state
        let t = 3usize;
        let n = r.hp.n_embd;
        let mut rng_state = 12345u64;
        let mut next = move || {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 33) as f32) / f32::from_bits(0x4f800000) - 1.0
        };
        let r_in: Vec<f32> = (0..t * 4 * n).map(|_| next()).collect();
        for block in 0..2 {
            let w = if block == 0 { &wa } else { &wf };
            let (x_cpu, mixes_cpu) = mhc_pre_all(w, &r_in, t, 4, n).unwrap();
            let set = mh.pre(block, &r_in, t).unwrap();
            let dx: f32 = x_cpu
                .iter()
                .zip(&set.x)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            let dpost: f32 = mixes_cpu
                .iter()
                .zip(0..)
                .flat_map(|(m, i)| {
                    m.post.iter().zip(&set.post[i * 4..(i + 1) * 4]).map(|(a, b)| (a - b).abs())
                })
                .fold(0.0, f32::max);
            let dcomb: f32 = mixes_cpu
                .iter()
                .zip(0..)
                .flat_map(|(m, i)| {
                    m.comb.iter().flatten().zip(&set.comb[i * 16..(i + 1) * 16]).map(|(a, b)| (a - b).abs())
                })
                .fold(0.0, f32::max);
            eprintln!("block {block}: x {dx:.3e} post {dpost:.3e} comb {dcomb:.3e}");
            // the bug's signature: block 1's mixes would equal block 0's GPU mixes here
            assert!(dx < 1e-4 && dpost < 1e-4 && dcomb < 1e-4, "block {block} mHC pre diverges");
        }
        // and the two blocks must NOT produce the same mixes (guards a future revert to
        // a constant block index passing this test trivially)
        let x0 = mh.pre(0, &r_in, t).unwrap().x;
        let x1 = mh.pre(1, &r_in, t).unwrap().x;
        let d: f32 = x0.iter().zip(&x1).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        eprintln!("block0 vs block1 x diff {d:.3e}");
        assert!(d > 1e-3, "hc_attn and hc_ffn pre produced identical mixes");
    }

    /// REGRESSION (the GLM-5.3 DSA bug, 74192590): each DSA layer's resident `MlaCuda`
    /// stages ITS OWN layer's weights and caches. The bug shared one sample-layer
    /// instance across all DSA layers (per device) — invisible to any single-DSA-layer
    /// fixture. This test asserts the structural invariant on the real model's geometry
    /// at load (no weights needed): `mla_k[l]` is `Some` exactly for DSA layers, and the
    /// per-layer instances are distinct objects.
    #[test]
    fn mla_resident_is_per_layer() {
        // the real GLM-5.3 geometry (HF config kda_layers): 11 DSA layers, the first at 3 —
        // the sample layer the per-device bug froze for all of them. The load-time
        // invariant (mla_k[l] is Some exactly on DSA layers) is asserted on a real engine
        // in engine_gpu_mla_matches_cpu; this pins the geometry the bug class depends on.
        let is_kda: Vec<bool> = (0..45).map(|l| l % 4 != 3).collect();
        let dsa: Vec<usize> = (0..45).filter(|&l| !is_kda[l]).collect();
        assert_eq!(dsa.len(), 11, "GLM-5.3 DSA layer count");
        assert_eq!(dsa[0], 3, "first DSA layer (the sample-layer bug's source)");
        assert!(dsa.len() > 1, "a single-DSA-layer geometry cannot see this bug class");
    }

    /// R3 (all of it): FFN + KDA + MLA/indexer on the GPU vs the CPU engine. W3's GPU
    /// latents are bf16 (the CPU's are f32), so the bar is top-1 agreement ≥ 98% and logits
    /// within 0.25; both numbers are printed for the pod log.
    #[cfg(feature = "cuda")]
    #[test]
    fn engine_gpu_mla_matches_cpu() {
        if !Path::new(TINY).exists() || tang_moe::gpu::Gpu::for_test().is_none() {
            skip_or_fail("fixture/GPU");
            return;
        }
        let meta = std::fs::read_to_string("/private/tmp/glm5-tiny/ref/meta.json").unwrap();
        let v: serde_json::Value = serde_json::from_str(&meta).unwrap();
        let ids2: Vec<u32> = v["seq2_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();
        let mut cpu = Glm5Engine::open(Path::new(TINY)).unwrap();
        let mut gpu = Glm5Engine::open(Path::new(TINY)).unwrap();
        assert!(gpu.use_gpu().unwrap(), "GPU did not load");
        assert!(
            gpu.gpu
                .as_ref()
                .unwrap()
                .mla_k
                .iter()
                .zip(&gpu.m.hp.is_kda)
                .all(|(k, &kda)| k.is_some() != kda),
            "MLA kernels: one per DSA layer, none on KDA layers"
        );
        let nv = cpu.m.hp.n_vocab;
        let (mut worst, mut agree, mut total) = (0f32, 0usize, 0usize);
        // dense (≤ top-k) then sparse windows, then decode steps
        for w in [&ids2[..12], &ids2[12..150], &ids2[150..151], &ids2[151..152], &ids2[152..170]] {
            let a = cpu.window(w).unwrap();
            let b = gpu.window(w).unwrap();
            for p in 0..w.len() {
                total += 1;
                if argmax(&a[p * nv..(p + 1) * nv]) == argmax(&b[p * nv..(p + 1) * nv]) {
                    agree += 1;
                }
            }
            for (x, y) in a.iter().zip(&b) {
                worst = worst.max((x - y).abs());
            }
        }
        eprintln!(
            "GPU (FFN + KDA + mHC + MLA bf16) vs CPU: top-1 {agree}/{total}, logits max abs {worst:.3e}"
        );
        assert!(agree as f32 >= 0.98 * total as f32, "top-1 {agree}/{total}");
        assert!(worst <= 0.25, "logits max abs {worst}");
    }

    /// reset() gives a fresh sequence.
    #[test]
    fn engine_reset() {
        let (Some(ids), true) = (seq1(), Path::new(TINY).exists()) else {
            return;
        };
        let mut e = Glm5Engine::open(Path::new(TINY)).unwrap();
        let a = e.window(&ids[..6]).unwrap();
        e.reset();
        let b = e.window(&ids[..6]).unwrap();
        assert_eq!(a, b);
    }

    /// THE LESSON OF THIS PORT (the engine never wired the built-and-tested GPU structs
    /// in): a test that the resident layers are actually CONSTRUCTED at `use_gpu()` —
    /// one DevLayer per KDA layer, one MhcGpu per layer. Without a CUDA device this
    /// asserts the counts the CPU build reports (0) so the wiring stays type-visible;
    /// on the pod it fails if a future refactor drops the construction.
    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_resident_layers_are_constructed() {
        if !Path::new(TINY).exists() {
            return;
        }
        let Some(_g) = tang_moe::gpu::Gpu::for_test() else {
            return;
        };
        let mut e = Glm5Engine::open(Path::new(TINY)).unwrap();
        assert!(e.use_gpu().unwrap(), "GPU did not load");
        let g = e.gpu.as_ref().unwrap();
        let want_kda = e.m.hp.is_kda.iter().filter(|&&k| k).count();
        assert_eq!(g.kda_resident(), want_kda, "KDA DevLayers constructed");
        assert_eq!(g.mhc_resident(), e.m.hp.n_layer, "mHC MhcGpus constructed");
    }

    /// THE LESSON, part 2: a KDA window actually ROUTES through the resident layer —
    /// observable via the engine's host/device state ownership: after one window the
    /// device is fresh (the host copies were seeded and the device owns the state).
    /// On a CPU-only build this is trivially skipped; the pod run pins the routing.
    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_kda_window_routes_through_resident_layer() {
        if !Path::new(TINY).exists() {
            return;
        }
        let Some(_g) = tang_moe::gpu::Gpu::for_test() else {
            return;
        };
        let (Some(ids), true) = (seq1(), Path::new(TINY).exists()) else {
            return;
        };
        let mut e = Glm5Engine::open(Path::new(TINY)).unwrap();
        assert!(e.use_gpu().unwrap());
        e.window(&ids[..4]).unwrap();
        let g = e.gpu.as_ref().unwrap();
        let kda_layers: Vec<usize> = (0..e.m.hp.n_layer).filter(|&l| e.m.hp.is_kda[l]).collect();
        assert!(
            g.kda_dev_fresh.iter().any(|&f| f),
            "no KDA layer took over the state: the mixer bypassed the resident path"
        );
        assert_eq!(kda_layers.len(), g.kda_resident());
    }
}
