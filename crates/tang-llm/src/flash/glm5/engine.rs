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
    /// W4's KDA kernels per device (`TANG_GLM5_GPU_KDA=0`: KDA stays on the CPU).
    kda_k: Vec<Option<super::kda_cuda::cuda::Kernels>>,
    /// W3's MLA + indexer kernels per device (`TANG_GLM5_GPU_MLA=0`: DSA stays on the CPU).
    mla_k: Vec<Option<super::mla::cuda::MlaCuda>>,
    layers: Vec<super::moe::gpu::DevMoe>,
    dev_of: Vec<usize>,
    pub split: usize,
}

#[cfg(feature = "cuda")]
impl GpuFfn {
    /// Upload every layer's FFN. `Ok(None)` when there is no CUDA device or `TANG_GLM5_GPU=0`.
    pub fn load(
        g: &crate::gguf::Gguf,
        hp: &super::Hparams,
        mla_sample: Option<&MlaWeights>,
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
        let mut kda_k = Vec::new();
        let mla_on = !std::env::var("TANG_GLM5_GPU_MLA").is_ok_and(|v| v == "0");
        let mut mla_k = Vec::new();
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
            let k = super::moe::gpu::Kernels::new(&gpu)?;
            kda_k.push(if kda_on && hp.is_kda.iter().any(|&b| b) {
                Some(super::kda_cuda::cuda::load(&gpu, &kda_cfg)?)
            } else {
                None
            });
            mla_k.push(match (mla_on, mla_sample) {
                (true, Some(w)) => Some(
                    super::mla::cuda::MlaCuda::new_at(w, ord)
                        .map_err(|e| anyhow::anyhow!("MLA kernels on GPU {ord}: {e}"))?,
                ),
                _ => None,
            });
            devs.push((gpu, k));
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
        let t0 = Instant::now();
        let mut layers = Vec::with_capacity(n_layer);
        let mut dev_of = Vec::with_capacity(n_layer);
        for l in 0..n_layer {
            let d = usize::from(l >= split);
            devs[d].0.bind().map_err(|e| anyhow::anyhow!("bind GPU {d}: {e}"))?;
            let p = super::moe::MoeParams::from_gguf(g, l)?;
            layers.push(
                super::moe::gpu::DevMoe::new(&p)
                    .with_context(|| format!("uploading layer {l}'s FFN to GPU {d}"))?,
            );
            dev_of.push(d);
            if l % 8 == 7 || l + 1 == n_layer {
                eprintln!(
                    "glm5: {} of {n_layer} FFNs on the GPU ({:.0} s)",
                    l + 1,
                    t0.elapsed().as_secs_f64()
                );
            }
        }
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
            kda_k,
            mla_k,
            layers,
            dev_of,
            split,
        }))
    }

    /// Layer `l`'s FFN on its GPU: host `x` `[t][hidden]` in, host `y` out.
    fn ffn(&self, l: usize, x: &[f32], t: usize) -> Result<Vec<f32>> {
        let (gpu, k) = &self.devs[self.dev_of[l]];
        gpu.bind().map_err(|e| anyhow::anyhow!("bind: {e}"))?;
        super::moe::gpu::ffn_host(k, &self.layers[l], x, t)
    }
}

/// The resumable GLM-5.3-Flash model.
pub struct Glm5Engine {
    pub m: Glm5Ref,
    /// Every token run so far (the prompt and the generated tokens).
    pub tokens: Vec<u32>,
    kda: Vec<Option<(ConvState, KdaState)>>,
    dsa: Vec<DsaCache>,
    cache_weights: bool,
    kda_w: Vec<Option<(KdaLayerW, KdaConfig)>>,
    mla_w: Vec<Option<MlaWeights>>,
    head: Option<crate::flash::reference::Mat>,
    /// Print per-window timings to stderr.
    pub verbose: bool,
    /// The FFNs on the GPU, when a CUDA build finds one ([`Glm5Engine::use_gpu`]).
    #[cfg(feature = "cuda")]
    pub gpu: Option<GpuFfn>,
}

impl Glm5Engine {
    pub fn open(path: &Path) -> Result<Self> {
        let m = Glm5Ref::open(path)?;
        let nl = m.hp.n_layer;
        let cache_weights = m.hp.n_embd <= 1024
            || std::env::var("TANG_GLM5_CACHE").is_ok_and(|v| v == "1");
        Ok(Self {
            m,
            tokens: Vec::new(),
            kda: (0..nl).map(|_| None).collect(),
            dsa: vec![DsaCache::default(); nl],
            cache_weights,
            kda_w: (0..nl).map(|_| None).collect(),
            mla_w: (0..nl).map(|_| None).collect(),
            head: None,
            verbose: false,
            #[cfg(feature = "cuda")]
            gpu: None,
        })
    }

    /// Move the FFNs to the GPU(s) (no-op without a device or with `TANG_GLM5_GPU=0`).
    /// Returns whether they moved.
    #[cfg(feature = "cuda")]
    pub fn use_gpu(&mut self) -> Result<bool> {
        let sample = match self.m.hp.is_kda.iter().position(|&k| !k) {
            Some(l) => Some(self.m.mla_weights(l)?),
            None => None,
        };
        self.gpu = GpuFfn::load(&self.m.g, &self.m.hp, sample.as_ref())?;
        Ok(self.gpu.is_some())
    }

    /// Layer `l`'s FFN: on its GPU when loaded, else W1's dense FFN / W5's CPU MoE.
    fn ffn(&self, l: usize, x: &[f32], t: usize) -> Result<Vec<f32>> {
        #[cfg(feature = "cuda")]
        if let Some(g) = &self.gpu {
            return g.ffn(l, x, t);
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

    fn kda_weights(&mut self, l: usize) -> Result<(KdaLayerW, KdaConfig)> {
        if let Some(w) = &self.kda_w[l] {
            return Ok(w.clone());
        }
        let w = self.m.kda_weights(l)?;
        if self.cache_weights {
            self.kda_w[l] = Some(w.clone());
        }
        Ok(w)
    }

    fn mla_weights(&mut self, l: usize) -> Result<MlaWeights> {
        if let Some(w) = &self.mla_w[l] {
            return Ok(w.clone());
        }
        let w = self.m.mla_weights(l)?;
        if self.cache_weights {
            self.mla_w[l] = Some(w.clone());
        }
        Ok(w)
    }

    /// KDA layer `l`'s mixer: W4's GPU conv + core when loaded, else the CPU reference.
    fn kda_mixer(
        &self,
        l: usize,
        x: &[f32],
        t: usize,
        w: &KdaLayerW,
        cfg: &KdaConfig,
        st: &mut (ConvState, KdaState),
    ) -> Result<Vec<f32>> {
        #[cfg(feature = "cuda")]
        if let Some(g) = &self.gpu {
            let d = g.dev_of[l];
            if let Some(k) = &g.kda_k[d] {
                let gpu = &g.devs[d].0;
                gpu.bind().map_err(|e| anyhow::anyhow!("bind: {e}"))?;
                return kda_gpu(x, t, w, cfg, k, gpu, &mut st.0, &mut st.1);
            }
        }
        let _ = l;
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
            if let Some(cu) = &g.mla_k[d] {
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

    /// Run `ids` after the tokens so far. Returns their logits `[ids.len()][n_vocab]`.
    pub fn window(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        ensure!(!ids.is_empty(), "empty window");
        let w0 = Instant::now();
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

        for l in 0..hp.n_layer {
            // ---- attention half
            let w_attn = super::mhc::MhcWeights::from_gguf(&self.m.g, l, "hc_attn", n, &mhp)?;
            let (mut x, mixes) = mhc_pre_all(&w_attn, &r, t, hc, n)?;
            self.m.norm_all(&mut x, l, "attn_norm.weight")?;
            let y = if hp.is_kda[l] {
                let (w, cfg) = self.kda_weights(l)?;
                let mut st = self.kda[l]
                    .take()
                    .unwrap_or_else(|| (ConvState::zeros(&cfg, 1), KdaState::zeros(&cfg, 1)));
                let y = self.kda_mixer(l, &x, t, &w, &cfg, &mut st);
                self.kda[l] = Some(st);
                y?
            } else {
                let w = self.mla_weights(l)?;
                self.dsa_layer(l, &w, &x, t, pos0)?
            };
            mhc_post_all(&mixes, &y, &mut r, t, hc, n)?;

            // ---- FFN half
            let w_ffn = super::mhc::MhcWeights::from_gguf(&self.m.g, l, "hc_ffn", n, &mhp)?;
            let (mut x, mixes) = mhc_pre_all(&w_ffn, &r, t, hc, n)?;
            self.m.norm_all(&mut x, l, "ffn_norm.weight")?;
            let y = self.ffn(l, &x, t)?;
            mhc_post_all(&mixes, &y, &mut r, t, hc, n)?;
        }

        // ---- head: unweighted stream mean, output norm, lm_head
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
        self.tokens.extend_from_slice(ids);
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
    let s = tang_moe::gpu::Stream::new().map_err(gerr)?;
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

    /// R3 (FFN + KDA, the f32 GPU paths): every FFN and the KDA conv + core on the GPU equal
    /// the CPU engine on the tiny model: 150-token prefill (2 GPU chunks + a CPU tail), GPU
    /// decode steps, then a short window. Pod: `TANG_REQUIRE_CUDA=1 cargo test --release -p tang-llm
    /// --no-default-features --features cuda --lib glm5::engine -- --nocapture`
    /// (`TANG_GPUS=2 TANG_SPLIT_AT=2` exercises the split).
    #[cfg(feature = "cuda")]
    #[test]
    fn engine_gpu_matches_cpu() {
        let (Some(ids), true) = (seq1(), Path::new(TINY).exists()) else {
            eprintln!("skip: tiny fixture absent");
            return;
        };
        if tang_moe::gpu::Gpu::for_test().is_none() {
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
        let _ = &ids;
        for w in [&ids2[..150], &ids2[150..151], &ids2[151..152], &ids2[152..160]] {
            let a = cpu.window(w).unwrap();
            let b = gpu.window(w).unwrap();
            for p in 0..w.len() {
                assert_eq!(
                    argmax(&a[p * nv..(p + 1) * nv]),
                    argmax(&b[p * nv..(p + 1) * nv]),
                    "top-1 differs"
                );
            }
            for (x, y) in a.iter().zip(&b) {
                worst = worst.max((x - y).abs());
            }
        }
        eprintln!("GPU (FFN + KDA) vs CPU: logits max abs {worst:.3e}");
        assert!(worst <= 1e-3, "logits max abs {worst}");
    }

    /// R3 (all of it): FFN + KDA + MLA/indexer on the GPU vs the CPU engine. W3's GPU
    /// latents are bf16 (the CPU's are f32), so the bar is top-1 agreement ≥ 98% and logits
    /// within 0.25; both numbers are printed for the pod log.
    #[cfg(feature = "cuda")]
    #[test]
    fn engine_gpu_mla_matches_cpu() {
        if !Path::new(TINY).exists() || tang_moe::gpu::Gpu::for_test().is_none() {
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
            gpu.gpu.as_ref().unwrap().mla_k.iter().all(|k| k.is_some()),
            "MLA kernels did not load"
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
            "GPU (FFN + KDA + MLA bf16) vs CPU: top-1 {agree}/{total}, logits max abs {worst:.3e}"
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
}
