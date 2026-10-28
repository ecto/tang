//! GLM-5.3-Flash MTP ("NextN") drafting + greedy draft-and-verify decode, CPU reference.
//!
//! The unsloth GGUF ships the NextN block as `blk.{n_layer}` (`blk.45`): a DSA (MLA-NoPE +
//! indexer) attention, a 288-expert MoE + shared expert, and `nextn.{eh_proj, enorm, hnorm,
//! shared_head_norm}`. Embedding and LM head are the trunk's (`token_embd`, `output`). There
//! are **no** mHC tensors: plain residual. Per llama.cpp `glm5-next.cpp` `graph_mtp`:
//!
//! ```text
//! cell i (position i):  h_i = trunk's post-output_norm hidden at i (t_h_nextn), tok = x_{i+1}
//! x    = eh_proj @ [rmsnorm(embed(tok))*enorm ; rmsnorm(h_i)*hnorm]       (8192 -> 4096)
//! x   += dsa(rmsnorm(x)*attn_norm)            own latent/indexer cache, one row per cell
//! x   += moe(rmsnorm(x)*ffn_norm) + shexp(..)
//! hn   = rmsnorm(x)*shared_head_norm           the next chain step's h (t_h_nextn)
//! logits = output @ hn                          predicts x_{i+2}
//! ```
//!
//! Decode: trunk window -> teacher-forced MTP cells for the window's positions -> chain
//! `k` drafts (temporary cells, truncated after) -> one trunk window `[cur, d1..dk]` verifies.
//! Greedy acceptance keeps the longest prefix where `d_{j+1} == argmax(row j)`. A partial
//! accept rewinds the trunk ([`Glm5Engine::snapshot`]/`restore`) and re-runs the accepted
//! prefix (the KDA recurrence has no per-token checkpoints yet), so the pass count below is
//! the honest one; `ideal` counts verify passes only (what per-token KDA checkpoints buy).

use super::engine::{argmax, Glm5Engine};
use super::mla::MlaWeights;
use crate::flash::reference::{rms_norm_mul, Mat};
use anyhow::{ensure, Result};
#[cfg(feature = "cuda")]
use anyhow::Context;
use rayon::prelude::*;
use std::path::Path;
use std::time::Instant;

/// The NextN block's weights plus its own per-cell DSA cache.
pub struct Mtp {
    pub layer: usize,
    n: usize,
    eps: f32,
    eh: Mat,
    enorm: Vec<f32>,
    hnorm: Vec<f32>,
    shnorm: Vec<f32>,
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    mla: MlaWeights,
    head: Mat,
    idx_top_k: usize,
    // per-cell caches (positions 0..len)
    latent: Vec<f32>,
    idx_k: Vec<f32>,
    idx_g: Vec<f32>,
    /// The device home of this block (`TANG_GLM5_MTP_DEV`, default on with a GPU engine):
    /// its own MLA caches + projections, its MoE FFN resident, the trunk's device LM head.
    #[cfg(feature = "cuda")]
    dev: Option<MtpDev>,
}

/// The NextN block on the device: the same pieces a trunk DSA layer uses (W3's `MlaCuda`
/// with its own latent / indexer caches, the resident projections `MlaProj`, the W5 FFN
/// `DevMoe`), called through their host-in / host-out entry points.
#[cfg(feature = "cuda")]
struct MtpDev {
    cu: super::mla::cuda::MlaCuda,
    p: super::mla_dev::MlaProj,
    c: super::engine::DsaCache,
    moe: super::moe::gpu::DevMoe,
}

/// What a batch of cells gives back.
pub struct CellOut {
    /// `hn` rows `[t][n]` (the next chain step's `h`).
    pub h: Vec<f32>,
    /// argmax draft per cell.
    pub top: Vec<u32>,
}

impl Mtp {
    pub fn load(e: &Glm5Engine) -> Result<Self> {
        let (m, hp) = (&e.m, &e.m.hp);
        ensure!(hp.n_nextn >= 1, "no NextN layer (nextn_predict_layers = 0)");
        let l = hp.n_layer;
        let v = |s: &str| m.vec(&super::t(l, s));
        let eh = m.mat(l, "nextn.eh_proj.weight")?;
        ensure!(
            eh.cols == 2 * hp.n_embd && eh.rows == hp.n_embd,
            "eh_proj is {}x{}",
            eh.rows,
            eh.cols
        );
        Ok(Self {
            layer: l,
            n: hp.n_embd,
            eps: hp.eps,
            eh,
            enorm: v("nextn.enorm.weight")?,
            hnorm: v("nextn.hnorm.weight")?,
            shnorm: v("nextn.shared_head_norm.weight")?,
            attn_norm: v("attn_norm.weight")?,
            ffn_norm: v("ffn_norm.weight")?,
            mla: m.mla_weights(l)?,
            head: m.head()?,
            idx_top_k: hp.idx_top_k,
            latent: Vec::new(),
            idx_k: Vec::new(),
            idx_g: Vec::new(),
            #[cfg(feature = "cuda")]
            dev: None,
        })
        .and_then(|mut me| {
            #[cfg(feature = "cuda")]
            me.load_dev(e)?;
            Ok(me)
        })
    }

    /// Put the block on the engine's GPU 0 (no-op without one, or `TANG_GLM5_MTP_DEV=0`).
    #[cfg(feature = "cuda")]
    fn load_dev(&mut self, e: &Glm5Engine) -> Result<()> {
        if std::env::var("TANG_GLM5_MTP_DEV").is_ok_and(|v| v == "0") {
            return Ok(());
        }
        let Some(gf) = e.gpu.as_ref() else { return Ok(()) };
        let (gpu, _) = gf.dev0();
        gpu.bind().map_err(|er| anyhow::anyhow!("bind GPU 0 (mtp): {er}"))?;
        let g = &e.m.g;
        let cu = super::mla::cuda::MlaCuda::new_at(&self.mla, 0)
            .map_err(|er| anyhow::anyhow!("mtp MlaCuda: {er}"))?;
        let p = super::mla_dev::MlaProj::new(gpu, g, self.layer, &self.mla)?;
        let mp = super::moe::MoeParams::from_gguf(g, self.layer)?;
        let moe = super::moe::gpu::DevMoe::new(&mp)?;
        self.dev = Some(MtpDev { cu, p, c: Default::default(), moe });
        eprintln!("glm5: MTP layer blk.{} resident on GPU 0 (DSA + FFN + head)", self.layer);
        Ok(())
    }

    /// Drop the device home: the host reference path (tests compare the two).
    pub fn host_only(mut self) -> Self {
        #[cfg(feature = "cuda")]
        {
            self.dev = None;
        }
        self
    }

    /// Host LM-head logits of `hn` rows `[t][n]` (tests: the draft margin).
    pub fn host_logits(&self, hn: &[f32]) -> Vec<f32> {
        self.head.apply(hn, hn.len() / self.n, false)
    }

    /// Is the block on the device?
    pub fn on_device(&self) -> bool {
        #[cfg(feature = "cuda")]
        return self.dev.is_some();
        #[cfg(not(feature = "cuda"))]
        false
    }

    /// Cells in the cache (= the next cell's position).
    pub fn len(&self) -> usize {
        #[cfg(feature = "cuda")]
        if let Some(d) = &self.dev {
            return d.c.valid_len();
        }
        self.latent.len() / self.mla.kv_lora
    }

    /// Drop cells at positions `>= pos`.
    pub fn truncate(&mut self, pos: usize) {
        #[cfg(feature = "cuda")]
        if let Some(d) = &mut self.dev {
            d.c.truncate_rows(pos, self.mla.kv_lora, self.mla.idx_dim);
        }
        self.latent.truncate(pos * self.mla.kv_lora);
        self.idx_k.truncate(pos * self.mla.idx_dim);
        self.idx_g.truncate(pos * self.mla.idx_dim);
    }

    /// Run `toks.len()` cells at positions `len()..`, cell `i` pairing `h[i]` with `toks[i]`.
    pub fn cells(&mut self, e: &Glm5Engine, h: &[f32], toks: &[u32]) -> Result<CellOut> {
        let g = &e.m.g;
        let (n, t, eps) = (self.n, toks.len(), self.eps);
        ensure!(
            h.len() == t * n,
            "mtp: h is {} floats, want {}",
            h.len(),
            t * n
        );
        let pos0 = self.len();
        let embd = g.info("token_embd.weight")?;
        // ---- [enorm(e) ; hnorm(h)] -> eh_proj
        let mut cat = vec![0f32; t * 2 * n];
        for (i, &tok) in toks.iter().enumerate() {
            let row = &mut cat[i * 2 * n..(i + 1) * 2 * n];
            row[..n].copy_from_slice(&g.rows(embd, tok as usize, 1)?);
            rms_norm_mul(&mut row[..n], &self.enorm, eps);
            row[n..].copy_from_slice(&h[i * n..(i + 1) * n]);
            rms_norm_mul(&mut row[n..], &self.hnorm, eps);
        }
        let mut x = self.eh.apply(&cat, t, false);
        // ---- attention half (the engine's dsa_cpu, on this block's own cache)
        let mut xa = x.clone();
        xa.par_chunks_mut(n)
            .for_each(|r| rms_norm_mul(r, &self.attn_norm, eps));
        #[cfg(feature = "cuda")]
        let y = match self.dev.as_mut() {
            Some(d) => {
                let (gpu, _) = e.gpu.as_ref().context("mtp: device block without a GPU")?.dev0();
                gpu.bind().map_err(|er| anyhow::anyhow!("bind (mtp): {er}"))?;
                d.cu.set_seq(0);
                super::engine::dsa_gpu_resident(self.layer, &self.mla, &d.cu, &d.p, &xa, t, pos0, &mut d.c, None)?
            }
            None => self.dsa_host(&xa, t, pos0),
        };
        #[cfg(not(feature = "cuda"))]
        let y = self.dsa_host(&xa, t, pos0);
        x.iter_mut().zip(&y).for_each(|(a, b)| *a += b);
        // ---- FFN half
        let mut xf = x.clone();
        xf.par_chunks_mut(n)
            .for_each(|r| rms_norm_mul(r, &self.ffn_norm, eps));
        #[cfg(feature = "cuda")]
        let y = match self.dev.as_ref() {
            Some(d) => {
                let (_, k) = e.gpu.as_ref().context("mtp: no GPU")?.dev0();
                super::moe::gpu::ffn_host(k, &d.moe, &xf, t)?
            }
            None => {
                let p = super::moe::MoeParams::from_gguf(g, self.layer)?;
                super::moe::moe_layer(&p, &xf, t)?
            }
        };
        #[cfg(not(feature = "cuda"))]
        let y = {
            let p = super::moe::MoeParams::from_gguf(g, self.layer)?;
            super::moe::moe_layer(&p, &xf, t)?
        };
        x.iter_mut().zip(&y).for_each(|(a, b)| *a += b);
        // ---- shared head
        x.par_chunks_mut(n)
            .for_each(|r| rms_norm_mul(r, &self.shnorm, eps));
        let nv = self.head.rows;
        #[cfg(feature = "cuda")]
        if let Some(gf) = e.gpu.as_ref().filter(|_| self.dev.is_some()) {
            let mut top = Vec::with_capacity(t);
            for i in 0..t {
                match gf.head_logits(&x[i * n..(i + 1) * n], n, nv)? {
                    Some(lg) => top.push(argmax(&lg)),
                    None => top.push(argmax(&self.head.apply(&x[i * n..(i + 1) * n], 1, false))),
                }
            }
            return Ok(CellOut { h: x, top });
        }
        let logits = self.head.apply(&x, t, false);
        let top = (0..t)
            .map(|i| argmax(&logits[i * nv..(i + 1) * nv]))
            .collect();
        Ok(CellOut { h: x, top })
    }
}

impl Mtp {
    /// The attention half on the host (f32 caches).
    fn dsa_host(&mut self, xa: &[f32], t: usize, pos0: usize) -> Vec<f32> {
        let n = self.n;
        let xa = xa.to_vec();
        let w = &self.mla;
        let kv_len = pos0 + t;
        let q_resid = w.gemv_q_a(&xa, t);
        self.latent.extend_from_slice(&w.gemv_kv_a(&xa, t));
        let idd = w.idx_dim;
        for i in 0..t {
            let (mut k, mut gg) = (vec![0f32; idd], vec![0f32; idd]);
            super::mla::indexer_key(w, &xa[i * n..(i + 1) * n], &mut k, &mut gg);
            self.idx_k.extend_from_slice(&k);
            self.idx_g.extend_from_slice(&gg);
        }
        // index_share_for_mtp_iteration is about reusing the trunk's top-k across MTP
        // iterations; below idx_top_k the indexer is dense either way.
        let sel = if super::mla::indexer_is_dense(kv_len, self.idx_top_k) {
            None
        } else {
            Some(super::mla::indexer_select(
                w,
                &xa,
                &q_resid,
                &self.idx_k,
                &self.idx_g,
                &vec![1u8; kv_len],
                &vec![1u8; t],
                t,
                kv_len,
                pos0,
            ))
        };
        let mut y = vec![0f32; t * n];
        super::mla::mla_forward(
            w,
            &xa,
            &q_resid,
            &self.latent,
            kv_len,
            t,
            pos0,
            None,
            sel.as_deref(),
            &mut y,
        );
        y
    }
}

/// Something that proposes draft tokens. `observe` gets every committed trunk window's
/// `h` rows with the token that FOLLOWS each position; `draft` then proposes up to `k`
/// tokens after the last committed one.
pub trait Drafter {
    fn observe(&mut self, e: &Glm5Engine, h: &[f32], next_toks: &[u32]) -> Result<()>;
    fn draft(&mut self, e: &Glm5Engine, k: usize) -> Result<Vec<u32>>;
}

/// The MTP drafter: teacher-forced cells for committed positions, chained cells for drafts.
pub struct MtpDrafter {
    pub mtp: Mtp,
    /// the last committed cell's (hn, argmax)
    last: Option<(Vec<f32>, u32)>,
}

impl MtpDrafter {
    pub fn new(mtp: Mtp) -> Self {
        Self { mtp, last: None }
    }
}

impl Drafter for MtpDrafter {
    fn observe(&mut self, e: &Glm5Engine, h: &[f32], next_toks: &[u32]) -> Result<()> {
        let o = self.mtp.cells(e, h, next_toks)?;
        let n = self.mtp.n;
        let t = next_toks.len();
        self.last = Some((o.h[(t - 1) * n..].to_vec(), o.top[t - 1]));
        Ok(())
    }

    fn draft(&mut self, e: &Glm5Engine, k: usize) -> Result<Vec<u32>> {
        let Some((mut h, mut d)) = self.last.clone() else {
            return Ok(vec![]);
        };
        let keep = self.mtp.len();
        let mut out = vec![d];
        while out.len() < k {
            let o = self.mtp.cells(e, &h, &[d])?;
            h = o.h;
            d = o.top[0];
            out.push(d);
        }
        self.mtp.truncate(keep);
        Ok(out)
    }
}

/// Decode statistics.
#[derive(Default, Debug, Clone)]
pub struct Stats {
    /// verify windows run (one per loop iteration)
    pub verify: usize,
    /// re-run windows after a partial accept (rewind)
    pub rerun: usize,
    /// drafts reached / accepted at each depth (index 0 = first draft; depth d is reached
    /// when every earlier draft in its window was accepted)
    pub proposed: Vec<usize>,
    pub accepted: Vec<usize>,
    /// tokens produced by the decode loop (excluding the prefill's token)
    pub tokens: usize,
    pub draft_s: f64,
    pub trunk_s: f64,
}

impl Stats {
    pub fn report(&self) -> String {
        let mut s = String::new();
        for (d, (p, a)) in self.proposed.iter().zip(&self.accepted).enumerate() {
            s += &format!(
                "mtp: depth {} accept {}/{} = {:.1}%\n",
                d + 1,
                a,
                p,
                100.0 * *a as f64 / (*p).max(1) as f64
            );
        }
        let passes = self.verify + self.rerun;
        s += &format!(
            "mtp: {} tokens, {} verify + {} rerun trunk passes: {:.3} tok/pass (ideal, verify only: {:.3}); draft {:.1} s, trunk {:.1} s\n",
            self.tokens,
            self.verify,
            self.rerun,
            self.tokens as f64 / passes.max(1) as f64,
            self.tokens as f64 / self.verify.max(1) as f64,
            self.draft_s,
            self.trunk_s
        );
        s
    }
}

/// Greedy draft-and-verify: prefill `prompt`, then generate until `n` tokens (the first is
/// the prefill's). Output equals plain greedy decode whatever the drafter proposes.
pub fn generate(
    e: &mut Glm5Engine,
    dr: &mut dyn Drafter,
    prompt: &[u32],
    n: usize,
    chunk: usize,
    k: usize,
) -> Result<(Vec<u32>, Stats)> {
    ensure!(
        !prompt.is_empty() && n >= 1 && k >= 1,
        "mtp generate: bad args"
    );
    let nv = e.m.hp.n_vocab;
    let mut st = Stats {
        proposed: vec![0; k],
        accepted: vec![0; k],
        ..Default::default()
    };
    // ---- prefill, collecting h for every prompt position
    let mut hs = Vec::new();
    let mut last = Vec::new();
    for c in prompt.chunks(chunk.max(1)) {
        let lg = e.window(c)?;
        hs.extend_from_slice(&e.last_h);
        last = lg[(c.len() - 1) * nv..].to_vec();
    }
    let mut cur = argmax(&last);
    let mut gen = vec![cur];
    let mut nexts: Vec<u32> = prompt[1..].to_vec();
    nexts.push(cur);
    let td = Instant::now();
    dr.observe(e, &hs, &nexts)?;
    st.draft_s += td.elapsed().as_secs_f64();
    while gen.len() < n {
        let td = Instant::now();
        // never draft past n: the window is cur + drafts
        let kk = k.min(n - gen.len());
        let mut drafts = dr.draft(e, kk)?;
        drafts.truncate(kk);
        st.draft_s += td.elapsed().as_secs_f64();
        let tt = Instant::now();
        let snap = e.snapshot()?;
        let mut win = vec![cur];
        win.extend_from_slice(&drafts);
        let lg = e.window(&win)?;
        st.verify += 1;
        let g: Vec<u32> = (0..win.len())
            .map(|j| argmax(&lg[j * nv..(j + 1) * nv]))
            .collect();
        let mut a = 0;
        while a < drafts.len() && drafts[a] == g[a] {
            a += 1;
        }
        // conditional per depth: depth d counts only when depths < d were accepted
        for d in 0..drafts.len().min(a + 1) {
            st.proposed[d] += 1;
            if d < a {
                st.accepted[d] += 1;
            }
        }
        let mut h = e.last_h.clone();
        if a < drafts.len() {
            // rewind and re-run the accepted prefix (no per-token KDA checkpoints yet)
            e.restore(snap);
            let lg2 = e.window(&win[..a + 1])?;
            st.rerun += 1;
            let g2: Vec<u32> = (0..=a)
                .map(|j| argmax(&lg2[j * nv..(j + 1) * nv]))
                .collect();
            if g2[..] != g[..=a] {
                eprintln!(
                    "mtp: WARNING re-run argmax {:?} != verify {:?} (window-size numerics)",
                    g2,
                    &g[..=a]
                );
            }
            h = e.last_h.clone();
            // the re-run is the authority: its state is what the next window continues
            let new = g2;
            st.trunk_s += tt.elapsed().as_secs_f64();
            let td = Instant::now();
            dr.observe(e, &h, &new)?;
            st.draft_s += td.elapsed().as_secs_f64();
            cur = *new.last().unwrap();
            gen.extend_from_slice(&new);
            st.tokens += new.len();
            continue;
        }
        st.trunk_s += tt.elapsed().as_secs_f64();
        let new = g[..=a].to_vec();
        h.truncate((a + 1) * e.m.hp.n_embd);
        let td = Instant::now();
        dr.observe(e, &h, &new)?;
        st.draft_s += td.elapsed().as_secs_f64();
        cur = *new.last().unwrap();
        gen.extend_from_slice(&new);
        st.tokens += new.len();
    }
    gen.truncate(n);
    // the last window may run past n; count only the kept tokens
    st.tokens = gen.len() - 1;
    Ok((gen, st))
}

/// `flash-generate ... --mtp K` (or `TANG_GLM5_MTP=K`): the MTP decode, same output lines as
/// the plain CLI plus the acceptance report.
/// Default for `TANG_GLM5_KDA_WIN_STEP` when the env is unset: [`cli_run`] sets it to
/// the draft depth + 1 (the verify window length), so MTP's KDA windows run as serial
/// t=1 steps and the verify is exact against plain decode. Plain decode leaves it 0.
pub static WIN_STEP_DEFAULT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// The effective KDA window-step threshold: `TANG_GLM5_KDA_WIN_STEP` if set and parses,
/// else [`WIN_STEP_DEFAULT`].
pub fn win_step() -> usize {
    std::env::var("TANG_GLM5_KDA_WIN_STEP")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or_else(|| WIN_STEP_DEFAULT.load(std::sync::atomic::Ordering::Relaxed))
}

pub fn cli_run(
    e: &mut Glm5Engine,
    ids: &[u32],
    n: usize,
    chunk: usize,
    k: usize,
    out: Option<&Path>,
    quiet: bool,
) -> Result<()> {
    WIN_STEP_DEFAULT.store(k + 1, std::sync::atomic::Ordering::Relaxed);
    eprintln!("glm5: KDA win_step={} (TANG_GLM5_KDA_WIN_STEP overrides)", win_step());
    let t0 = Instant::now();
    let mut dr = MtpDrafter::new(Mtp::load(e)?);
    eprintln!(
        "glm5: MTP layer blk.{} loaded, k={k}, {:.1} s",
        dr.mtp.layer,
        t0.elapsed().as_secs_f64()
    );
    let t1 = Instant::now();
    let (gen, st) = generate(e, &mut dr, ids, n, chunk, k)?;
    let s = t1.elapsed().as_secs_f64();
    println!("prompt: {} tokens", ids.len());
    let line = gen
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(" ");
    println!("ids: {line}");
    println!("decode: {} tokens in {:.2} s (incl. prefill)", gen.len(), s);
    print!("{}", st.report());
    if !quiet {
        let tk = crate::flash::tokenize::FlashTokenizer::from_gguf(&e.m.g);
        if let Ok(text) = tk.and_then(|t| t.decode(&gen)) {
            println!("text: {text}");
        }
    }
    if let Some(o) = out {
        std::fs::write(o, format!("{line}\n"))?;
    }
    Ok(())
}
