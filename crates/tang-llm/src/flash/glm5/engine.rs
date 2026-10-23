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
        })
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

    /// Run `ids` after the tokens so far. Returns their logits `[ids.len()][n_vocab]`.
    pub fn window(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        ensure!(!ids.is_empty(), "empty window");
        let w0 = Instant::now();
        let hp = self.m.hp.clone();
        let (n, hc, t) = (hp.n_embd, hp.hc, ids.len());
        let pos0 = self.tokens.len();
        let kv_len = pos0 + t;

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
                let st = self.kda[l].get_or_insert_with(|| {
                    (ConvState::zeros(&cfg, 1), KdaState::zeros(&cfg, 1))
                });
                super::kda::kda_layer(&x, &w, &cfg, &mut st.0, &mut st.1, 1, t)?
            } else {
                let w = self.mla_weights(l)?;
                let q_resid = w.gemv_q_a(&x, t);
                let latent = w.gemv_kv_a(&x, t);
                let idx_dim = hp.idx_dim;
                let cache = &mut self.dsa[l];
                cache.latent.extend_from_slice(&latent);
                for i in 0..t {
                    let (mut k, mut g) = (vec![0f32; idx_dim], vec![0f32; idx_dim]);
                    super::mla::indexer_key(&w, &x[i * n..(i + 1) * n], &mut k, &mut g);
                    cache.idx_k.extend_from_slice(&k);
                    cache.idx_g.extend_from_slice(&g);
                }
                let sel = if super::mla::indexer_is_dense(kv_len, hp.idx_top_k) {
                    None
                } else {
                    Some(super::mla::indexer_select(
                        &w,
                        &x,
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
                    &x,
                    &q_resid,
                    &cache.latent,
                    kv_len,
                    t,
                    pos0,
                    None,
                    sel.as_deref(),
                    &mut out,
                );
                out
            };
            mhc_post_all(&mixes, &y, &mut r, t, hc, n)?;

            // ---- FFN half
            let w_ffn = super::mhc::MhcWeights::from_gguf(&self.m.g, l, "hc_ffn", n, &mhp)?;
            let (mut x, mixes) = mhc_pre_all(&w_ffn, &r, t, hc, n)?;
            self.m.norm_all(&mut x, l, "ffn_norm.weight")?;
            let y = if hp.is_dense(l) {
                self.m.dense_ffn(l, &x, t)?
            } else {
                let p = super::moe::MoeParams::from_gguf(&self.m.g, l)?;
                super::moe::moe_layer(&p, &x, t)?
            };
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
