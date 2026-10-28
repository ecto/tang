//! Chunk-splice probe: can a KDA chunk's affine state map be precomputed without context?
//!
//! For fixed per-token (k, v, β, g) the KDA update is affine in S, so running a chunk's KDA
//! from the *oracle* `S_start` with the chunk's *context-free* inputs applies exactly the
//! affine map a precomputed (A, B) would. One layer-major CPU pass carries every run in
//! lockstep (each layer's weights are read once): the oracle over the whole text, per spliced
//! chunk two context-free "alone" runs (zero context; true context up to the K-th KDA layer),
//! and three 64-token continuations whose KDA state at the chunk end is the zero-context
//! splice, the K splice, or the chunk skipped (`S_end = S_start`, a control). MLA rows of the
//! continuations attend over the oracle prefix, so only the recurrent state is spliced.
//!
//! ```text
//! SPLICE_GGUF=<first shard> [SPLICE_TEXT=file] [SPLICE_CHUNKS=4,8,12] [SPLICE_C=64]
//! [SPLICE_K=4] cargo test -p tang-llm --release --lib splice_probe -- --ignored --nocapture
//! ```

use super::forward::Glm5Ref;
use super::kda::{kda_layer, ConvState, KdaState};
use super::mhc::{mhc_post, mhc_pre, MhcHparams, MhcWeights, Mix};
use super::mla::MlaWeights;
use anyhow::Result;
use rayon::prelude::*;

/// The ways a run's rows get their attention-half mixer.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Oracle,
    /// Context-free chunk; `true` = seeded from the oracle up to the K-th KDA layer.
    Alone(usize, bool),
    /// Continuation after chunk `c`; 0 = zero-ctx splice, 1 = K splice, 2 = skip control.
    Cont(usize, u8),
}

/// Oracle `(S_start, S_end, conv_end)` of one spliced chunk.
type Snap = (KdaState, KdaState, ConvState);

struct Run {
    kind: Kind,
    row0: usize,
    len: usize,
}

fn rel(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut n) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        d += ((x - y) as f64).powi(2);
        n += (*y as f64).powi(2);
    }
    (d / n.max(1e-30)).sqrt()
}

fn log_softmax(l: &[f32]) -> Vec<f64> {
    let m = l.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
    let z = l.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln() + m;
    l.iter().map(|&v| v as f64 - z).collect()
}

/// mHC pre over rows `[r0, r0+len)` of `r` (`[row][hc][n]`).
fn pre(
    w: &MhcWeights,
    r: &[f32],
    rows: std::ops::Range<usize>,
    n: usize,
) -> Result<(Vec<f32>, Vec<Mix>)> {
    let (mut x, mut mixes) = (Vec::new(), Vec::new());
    for t in rows {
        let s: Vec<&[f32]> = (0..4)
            .map(|c| &r[(t * 4 + c) * n..(t * 4 + c + 1) * n])
            .collect();
        let m = mhc_pre(w, &s)?;
        x.extend_from_slice(&m.x);
        mixes.push(m);
    }
    Ok((x, mixes))
}

fn post(mixes: &[Mix], y: &[f32], r: &mut [f32], r0: usize, n: usize) -> Result<()> {
    for (i, m) in mixes.iter().enumerate() {
        let tok = &mut r[(r0 + i) * 4 * n..(r0 + i + 1) * 4 * n];
        let r_in = tok.to_vec();
        let s: Vec<&[f32]> = r_in.chunks(n).collect();
        let mut outs: Vec<&mut [f32]> = tok.chunks_mut(n).collect();
        mhc_post(m, &y[i * n..(i + 1) * n], &s, &mut outs)?;
    }
    Ok(())
}

/// Dense-causal MLA over `x` (`[t][n]`), as the reference forward runs it below `idx_top_k`.
fn mla(w: &MlaWeights, x: &[f32], t: usize) -> Result<Vec<f32>> {
    let (q, kl) = (w.gemv_q_a(x, t), w.gemv_kv_a(x, t));
    let mut out = vec![0f32; t * w.hidden];
    super::mla::mla_forward_absorbed(w, x, &q, &kl, t, t, 0, None, None, &mut out);
    Ok(out)
}

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

#[test]
#[ignore = "needs SPLICE_GGUF (GLM-5 GGUF); a long CPU run on the real model"]
fn splice_probe() -> Result<()> {
    let m = Glm5Ref::open(std::path::Path::new(&std::env::var("SPLICE_GGUF")?))?;
    let hp = &m.hp;
    let (n, cs, kk) = (hp.n_embd, env("SPLICE_C", 64usize), env("SPLICE_K", 4usize));
    let chunks: Vec<usize> = std::env::var("SPLICE_CHUNKS")
        .unwrap_or("4,8,12".into())
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let t_len = (chunks.iter().max().unwrap() + 2) * cs;
    let text = match std::env::var("SPLICE_TEXT") {
        Ok(p) => std::fs::read_to_string(p)?,
        Err(_) => [
            include_str!("../../../../../README.md"),
            include_str!("kda.rs"),
        ]
        .concat(),
    };
    let mut ids = crate::flash::tokenize::FlashTokenizer::from_gguf(&m.g)?.encode(&text)?;
    anyhow::ensure!(
        ids.len() >= t_len,
        "text is {} tokens, need {t_len}",
        ids.len()
    );
    ids.truncate(t_len);
    let kda_layers: Vec<usize> = (0..hp.n_layer).filter(|&l| hp.is_kda[l]).collect();
    if !super::mla::indexer_is_dense(t_len, hp.idx_top_k) {
        eprintln!("warning: {t_len} tokens > indexer top-k; MLA runs dense (fixture only)");
    }
    let l_k = kda_layers[kk.min(kda_layers.len() - 1)]; // input of this layer is the oracle's
    eprintln!("{t_len} tokens, chunk {cs}, splice {chunks:?}, K={kk} (oracle up to layer {l_k})");

    // ---- the runs and their rows in one residual buffer
    let mut runs = vec![Run {
        kind: Kind::Oracle,
        row0: 0,
        len: t_len,
    }];
    let mut rows = t_len;
    for &c in &chunks {
        let kinds = [
            Kind::Alone(c, false),
            Kind::Alone(c, true),
            Kind::Cont(c, 0),
            Kind::Cont(c, 1),
            Kind::Cont(c, 2),
        ];
        for kind in kinds {
            runs.push(Run {
                kind,
                row0: rows,
                len: cs,
            });
            rows += cs;
        }
    }
    let src = |kind: Kind| match kind {
        Kind::Oracle => 0,
        Kind::Alone(c, _) => c * cs,
        Kind::Cont(c, _) => (c + 1) * cs,
    };
    let embd = m.g.info("token_embd.weight")?;
    let mut r = vec![0f32; rows * 4 * n];
    for run in &runs {
        for i in 0..run.len {
            let e = m.g.rows(embd, ids[src(run.kind) + i] as usize, 1)?;
            for c in 0..4 {
                let o = ((run.row0 + i) * 4 + c) * n;
                r[o..o + n].copy_from_slice(&e);
            }
        }
    }
    // per spliced chunk, per KDA layer: (oracle S_end, zero splice, K splice, skip)
    let mut errs: Vec<Vec<[f64; 3]>> = vec![vec![]; chunks.len()];
    let mhp = MhcHparams {
        streams: 4,
        sinkhorn_iters: hp.hc_sinkhorn_iters as u32,
        hc_eps: hp.hc_eps,
        rms_eps: hp.eps,
    };
    let start = std::time::Instant::now();

    for l in 0..hp.n_layer {
        // the K runs see the oracle's residual up to (and into) layer l_k
        if l <= l_k {
            for run in runs
                .iter()
                .filter(|r| matches!(r.kind, Kind::Alone(_, true)))
            {
                let (a, b) = (src(run.kind) * 4 * n, run.row0 * 4 * n);
                r.copy_within(a..a + cs * 4 * n, b);
            }
        }
        // --- attention half
        let w = MhcWeights::from_gguf(&m.g, l, "hc_attn", n, &mhp)?;
        let (mut x, mixes) = pre(&w, &r, 0..rows, n)?;
        m.norm_all(&mut x, l, "attn_norm.weight")?;
        let mut y = vec![0f32; rows * n];
        let kda = if hp.is_kda[l] {
            Some(m.kda_weights(l)?)
        } else {
            None
        };
        let mlaw = if hp.is_kda[l] {
            None
        } else {
            Some(m.mla_weights(l)?)
        };
        let slice = |ri: usize| &x[runs[ri].row0 * n..(runs[ri].row0 + runs[ri].len) * n];
        let mut outs: Vec<Vec<f32>> = vec![vec![]; runs.len()];
        if let Some(w) = &mlaw {
            // MLA: continuations attend over the oracle prefix
            outs = (0..runs.len())
                .into_par_iter()
                .map(|ri| {
                    let p = match runs[ri].kind {
                        Kind::Cont(c, _) => (c + 1) * cs,
                        _ => 0,
                    };
                    let xx = [&x[..p * n], slice(ri)].concat();
                    Ok(mla(w, &xx, p + runs[ri].len)?[p * n..].to_vec())
                })
                .collect::<Result<_>>()?;
        }
        if let Some((kw, cfg)) = &kda {
            let cp = |s: &KdaState, c: &ConvState| (KdaState(s.0.clone()), ConvState(c.0.clone()));
            // 1: the oracle in chunks (snapshotting S_start, S_end, conv_end of each spliced
            //    chunk) alongside every context-free chunk's own output
            let (oracle, alone) = rayon::join(
                || -> Result<(Vec<f32>, Vec<Snap>)> {
                    let (mut conv, mut st) = (ConvState::zeros(cfg, 1), KdaState::zeros(cfg, 1));
                    let (mut out, mut snap) = (vec![], vec![]);
                    for c in 0..t_len / cs {
                        let s0 = KdaState(st.0.clone());
                        let xs = &slice(0)[c * cs * n..(c + 1) * cs * n];
                        out.extend(kda_layer(xs, kw, cfg, &mut conv, &mut st, 1, cs)?);
                        if chunks.contains(&c) {
                            let (s1, cv) = cp(&st, &conv);
                            snap.push((s0, s1, cv));
                        }
                    }
                    Ok((out, snap))
                },
                || -> Result<Vec<Vec<f32>>> {
                    (0..runs.len())
                        .into_par_iter()
                        .map(|ri| match runs[ri].kind {
                            Kind::Alone(..) => {
                                let mut conv = ConvState::zeros(cfg, 1);
                                let mut st = KdaState::zeros(cfg, 1);
                                kda_layer(slice(ri), kw, cfg, &mut conv, &mut st, 1, cs)
                            }
                            _ => Ok(vec![]),
                        })
                        .collect()
                },
            );
            let (o, snap) = oracle?;
            for (ri, a) in alone?.into_iter().enumerate() {
                outs[ri] = a;
            }
            outs[0] = o;
            // 2: S_end per (chunk, variant): the chunk's context-free affine map applied to
            //    the oracle S_start (zero splice; K splice, oracle below l_k), or skipped
            let ends: Vec<(KdaState, ConvState)> = (0..chunks.len() * 3)
                .into_par_iter()
                .map(|j| -> Result<_> {
                    let ((s0, s1, cv), v) = (&snap[j / 3], j % 3);
                    match v {
                        2 => return Ok(cp(s0, cv)),
                        1 if l < l_k => return Ok(cp(s1, cv)),
                        _ => {}
                    }
                    let ri = runs
                        .iter()
                        .position(|q| q.kind == Kind::Alone(chunks[j / 3], v == 1))
                        .unwrap();
                    let (mut st, mut conv) = (KdaState(s0.0.clone()), ConvState::zeros(cfg, 1));
                    kda_layer(slice(ri), kw, cfg, &mut conv, &mut st, 1, cs)?;
                    Ok((st, conv))
                })
                .collect::<Result<_>>()?;
            for (ci, e) in errs.iter_mut().enumerate() {
                e.push([0, 1, 2].map(|v| rel(&ends[ci * 3 + v].0 .0, &snap[ci].1 .0)));
            }
            // 3: the continuations from those states
            let cont: Vec<Vec<f32>> = (0..runs.len())
                .into_par_iter()
                .map(|ri| match runs[ri].kind {
                    Kind::Cont(c, v) => {
                        let ci = chunks.iter().position(|&q| q == c).unwrap();
                        let (mut st, mut conv) =
                            cp(&ends[ci * 3 + v as usize].0, &ends[ci * 3 + v as usize].1);
                        kda_layer(slice(ri), kw, cfg, &mut conv, &mut st, 1, cs)
                    }
                    _ => Ok(vec![]),
                })
                .collect::<Result<_>>()?;
            for (ri, c) in cont.into_iter().enumerate().filter(|(_, c)| !c.is_empty()) {
                outs[ri] = c;
            }
        }
        for (run, o) in runs.iter().zip(&outs) {
            y[run.row0 * n..(run.row0 + run.len) * n].copy_from_slice(o);
        }
        drop((kda, mlaw));
        post(&mixes, &y, &mut r, 0, n)?;

        // --- FFN half: per token, so every run's rows go through together
        let w = MhcWeights::from_gguf(&m.g, l, "hc_ffn", n, &mhp)?;
        let (mut x, mixes) = pre(&w, &r, 0..rows, n)?;
        m.norm_all(&mut x, l, "ffn_norm.weight")?;
        let y = if hp.is_dense(l) {
            m.dense_ffn(l, &x, rows)?
        } else {
            super::moe::moe_layer(&super::moe::MoeParams::from_gguf(&m.g, l)?, &x, rows)?
        };
        post(&mixes, &y, &mut r, 0, n)?;
        eprintln!(
            "layer {l:2} {} total {:.0}s",
            if hp.is_kda[l] { "kda" } else { "dsa" },
            start.elapsed().as_secs_f64()
        );
    }

    // ---- head on the rows that matter: oracle [c_end, c_end+C) and each continuation
    let head = m.head()?;
    let logits = |row0: usize| -> Result<Vec<f32>> {
        let mut h = vec![0f32; cs * n];
        for i in 0..cs {
            for c in 0..4 {
                for j in 0..n {
                    h[i * n + j] += r[((row0 + i) * 4 + c) * n + j] / 4.0;
                }
            }
        }
        m.norm_all_exact(&mut h, "output_norm.weight")?;
        Ok(head.apply(&h, cs, false))
    };
    let nv = hp.n_vocab;
    let g = kda_layers.len();
    let groups = [
        (0, g / 3, "early"),
        (g / 3, 2 * g / 3, "mid"),
        (2 * g / 3, g, "late"),
    ];
    println!(
        "\n| chunk | variant | state rel-L2 early | mid | late | KL mean | KL max | top1 agree |"
    );
    println!("|---|---|---|---|---|---|---|---|");
    for (ci, &c) in chunks.iter().enumerate() {
        let lo = logits((c + 1) * cs)?;
        for (v, name) in [
            (0u8, "zero-ctx splice"),
            (1, "K splice"),
            (2, "skip (control)"),
        ] {
            let row0 = runs
                .iter()
                .find(|q| q.kind == Kind::Cont(c, v))
                .unwrap()
                .row0;
            let ls = logits(row0)?;
            let (mut kl_sum, mut kl_max, mut agree) = (0f64, 0f64, 0);
            for i in 0..cs {
                let (p, q) = (
                    log_softmax(&lo[i * nv..(i + 1) * nv]),
                    log_softmax(&ls[i * nv..(i + 1) * nv]),
                );
                let kl: f64 = p.iter().zip(&q).map(|(a, b)| a.exp() * (a - b)).sum();
                kl_sum += kl;
                kl_max = kl_max.max(kl);
                let am = |v: &[f64]| (0..nv).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap();
                agree += (am(&p) == am(&q)) as usize;
            }
            let ge: Vec<String> = groups
                .iter()
                .map(|&(a, b, _)| {
                    let s = &errs[ci][a..b];
                    format!(
                        "{:.3}",
                        s.iter().map(|e| e[v as usize]).sum::<f64>() / s.len().max(1) as f64
                    )
                })
                .collect();
            println!(
                "| {c} | {name} | {} | {:.4} | {:.4} | {agree}/{cs} |",
                ge.join(" | "),
                kl_sum / cs as f64,
                kl_max
            );
        }
    }
    for (ci, &c) in chunks.iter().enumerate() {
        let per: Vec<String> = errs[ci]
            .iter()
            .map(|e| format!("{:.2}/{:.2}/{:.2}", e[0], e[1], e[2]))
            .collect();
        println!(
            "chunk {c} per-KDA-layer rel-L2 (zero/K/skip): {}",
            per.join(" ")
        );
    }
    println!("runtime {:.0}s", start.elapsed().as_secs_f64());
    Ok(())
}
