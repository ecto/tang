//! The MTP (multi-token prediction) draft layer of Qwen3.8-Flash-Next, on the f32 reference.
//!
//! It ships separately (unsloth's `MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf`): one block, `blk.48`,
//! plus its own copies of `token_embd` and `output`. Cell `i` pairs the main model's final 4-stream
//! residual at position `i` (`h_i`, before the output hyper-connection read) with the token at
//! position `i + 1`, at rope position `i`, and predicts the token at `i + 2`:
//!
//! ```text
//! e      = rmsnorm(embed(tok)) * enorm                                  (2560)
//! hn[c]  = rmsnorm(h[c]) * hnorm[c]       per stream (llama.cpp); or rmsnorm over all 10240 (Strata)
//! R[c]   = eh_proj @ [e ; hn[c]]          eh_proj is [5120 -> 2560], applied per stream
//! x, inj = hc_read(R, hc_attn_*);  R = hc_write(R, attn(x), inj)       dense causal attention, own K/V
//! x, inj = hc_read(R, hc_ffn_*);   R = hc_write(R, moe(x), inj)        512 experts, top-10, shared
//! logits = output @ hc_read(R, nextn.hc_head_*)
//! ```
//!
//! The attention is the QSA block's (`[q | gate]` per head, q/k RMSNorm, NeoX rope on 64 dims,
//! 2 KV heads, sigmoid output gate) with `compress_ratios[48] = 0`, i.e. dense: no indexer. The
//! layer's output residual `R` is the next draft step's `h`: a chain draft at depth `d` from start
//! cell `i` is a cell at position `i + d - 1` with the previous draft as its token, attending to the
//! teacher-forced cells `0..=i` and the chain's own earlier cells.
//!
//! There is no llama.cpp oracle for this, so it's validated by acceptance ([`eval`]).

use super::reference::{hc_write, rms_norm_mul, rope_neox, sigmoid, Dump, FlashRef};
use anyhow::{ensure, Context, Result};
use rayon::prelude::*;
use std::fmt::Write as _;
use std::path::Path;

/// The MTP block and where it takes its embedding and head from.
pub struct Mtp {
    /// The MTP GGUF as a reference model (its `blk.48`, embedding and head).
    pub m: FlashRef,
    pub layer: usize,
    /// RMS over all 10240 of `h` (Strata / vLLM) instead of per 2560 stream (llama.cpp).
    pub joint_hnorm: bool,
    /// Use the main model's embedding and head instead of the MTP file's own copies.
    pub main_head: bool,
}

/// K/V rows of every MTP cell computed so far, `[n][nkv][hd]`.
#[derive(Default)]
struct Kv {
    k: Vec<f32>,
    v: Vec<f32>,
}

/// One MTP cell to run: its input residual, token, rope position, and which stored K/V rows (plus
/// itself, appended last) it attends to.
struct Cell<'a> {
    h: &'a [f32],
    tok: u32,
    pos: usize,
    visible: Vec<usize>,
}

/// What a batch of cells produces.
struct Out {
    /// Output residual per cell, `[hc][n_embd]`.
    r: Vec<f32>,
    /// Argmax token and its softmax probability per cell.
    top: Vec<(u32, f32)>,
    /// Index of each cell's K/V row in the store.
    kv_row: Vec<usize>,
}

impl Mtp {
    pub fn open(path: &Path) -> Result<Self> {
        let m = FlashRef::open(path)?;
        let layer =
            m.g.meta_u64("qwen4exp.block_count")
                .context("MTP block count")? as usize
                - 1;
        ensure!(
            m.g.get(&format!("blk.{layer}.nextn.eh_proj.weight"))
                .is_some(),
            "{}: no blk.{layer}.nextn.eh_proj.weight, not an MTP file",
            path.display()
        );
        Ok(Self {
            m,
            layer,
            joint_hnorm: false,
            main_head: false,
        })
    }

    fn name(&self, s: &str) -> String {
        format!("blk.{}.{s}", self.layer)
    }

    /// Run a batch of cells. New K/V rows are appended to `kv` *before* attention, so a cell sees
    /// itself; `visible` lists the earlier rows it may also see.
    fn run(
        &self,
        main: &FlashRef,
        cells: &[Cell],
        kv: &mut Kv,
        dump: Option<&mut Dump>,
    ) -> Result<Out> {
        let hp = &self.m.hp;
        let (n, hc, eps) = (hp.n_embd, hp.hc, hp.eps);
        let (nh, nkv, hd) = (hp.n_head, hp.n_head_kv, hp.head_dim);
        let nc = cells.len();
        let emb_src = if self.main_head { &main.g } else { &self.m.g };

        // ---- inputs: e = rms(embed) * enorm, hn = rms(h) * hnorm, R[c] = eh_proj [e ; hn[c]]
        // The GGUF stores both gammas already folded (HF's Gemma-style 1 + w): adding 1 again drops
        // depth-1 acceptance on chat2 from 0.761 to 0.610.
        let enorm = self.m.vec(&self.name("nextn.enorm.weight"))?;
        let hnorm = self.m.vec(&self.name("nextn.hnorm.weight"))?;
        let embd = emb_src.info("token_embd.weight")?;
        let mut cat = vec![0f32; nc * hc * 2 * n];
        for (i, c) in cells.iter().enumerate() {
            ensure!(c.h.len() == hc * n, "MTP h is {} wide", c.h.len());
            let mut e = emb_src.rows(embd, c.tok as usize, 1)?;
            rms_norm_mul(&mut e, &enorm, eps);
            let mut hn = c.h.to_vec();
            if self.joint_hnorm {
                rms_norm_mul(&mut hn, &hnorm, eps);
            } else {
                for s in 0..hc {
                    rms_norm_mul(&mut hn[s * n..(s + 1) * n], &hnorm[s * n..(s + 1) * n], eps);
                }
            }
            for s in 0..hc {
                let row = &mut cat[(i * hc + s) * 2 * n..(i * hc + s + 1) * 2 * n];
                // [e ; hn]: the embedding half first (swapped, acceptance is 0)
                row[..n].copy_from_slice(&e);
                row[n..].copy_from_slice(&hn[s * n..(s + 1) * n]);
            }
        }
        let eh = self.m.load(&self.name("nextn.eh_proj.weight"))?;
        ensure!(
            eh.cols == 2 * n && eh.rows == n,
            "eh_proj is {}x{}",
            eh.rows,
            eh.cols
        );
        let mut r = eh.apply(&cat, nc * hc, false); // [nc][hc][n]

        // ---- attention
        let (x, inj) = self.m.hc_read(&r, nc, &self.name("hc_attn_"), true)?;
        let q_full = self
            .m
            .load(&self.name("attn_q.weight"))?
            .apply(&x, nc, false);
        let mut kc = self
            .m
            .load(&self.name("attn_k.weight"))?
            .apply(&x, nc, false);
        let vc = self
            .m
            .load(&self.name("attn_v.weight"))?
            .apply(&x, nc, false);
        let qn = self.m.vec(&self.name("attn_q_norm.weight"))?;
        let kn = self.m.vec(&self.name("attn_k_norm.weight"))?;
        let mut q = vec![0f32; nc * nh * hd];
        let mut gate = vec![0f32; nc * nh * hd];
        for (t, c) in cells.iter().enumerate() {
            for h in 0..nh {
                let src = &q_full[(t * nh + h) * 2 * hd..(t * nh + h + 1) * 2 * hd];
                let dq = &mut q[(t * nh + h) * hd..(t * nh + h + 1) * hd];
                dq.copy_from_slice(&src[..hd]);
                rms_norm_mul(dq, &qn, eps);
                rope_neox(dq, c.pos, hp.n_rot, hp.rope_base);
                gate[(t * nh + h) * hd..(t * nh + h + 1) * hd].copy_from_slice(&src[hd..]);
            }
            for g in 0..nkv {
                let dk = &mut kc[(t * nkv + g) * hd..(t * nkv + g + 1) * hd];
                rms_norm_mul(dk, &kn, eps);
                rope_neox(dk, c.pos, hp.n_rot, hp.rope_base);
            }
        }
        let base = kv.k.len() / (nkv * hd);
        kv.k.extend_from_slice(&kc);
        kv.v.extend_from_slice(&vc);
        let kv_row: Vec<usize> = (0..nc).map(|t| base + t).collect();
        let kq_scale = 1.0 / (hd as f32).sqrt();
        let group = nh / nkv;
        let (kk, vv) = (&kv.k, &kv.v);
        let attn: Vec<f32> = cells
            .par_iter()
            .enumerate()
            .flat_map_iter(|(t, c)| {
                let mut rows = c.visible.clone();
                rows.push(base + t);
                let mut out = vec![0f32; nh * hd];
                let mut w = vec![0f32; rows.len()];
                for h in 0..nh {
                    let g = h / group;
                    let qh = &q[(t * nh + h) * hd..(t * nh + h + 1) * hd];
                    let mut mx = f32::NEG_INFINITY;
                    for (j, &rr) in rows.iter().enumerate() {
                        w[j] = super::reference::dot(
                            qh,
                            &kk[(rr * nkv + g) * hd..(rr * nkv + g + 1) * hd],
                        ) * kq_scale;
                        mx = mx.max(w[j]);
                    }
                    let mut sum = 0f64;
                    for v in w.iter_mut() {
                        *v = (*v - mx).exp();
                        sum += *v as f64;
                    }
                    let oh = &mut out[h * hd..(h + 1) * hd];
                    for (j, &rr) in rows.iter().enumerate() {
                        let p = (w[j] as f64 / sum) as f32;
                        for (o, v) in oh
                            .iter_mut()
                            .zip(&vv[(rr * nkv + g) * hd..(rr * nkv + g + 1) * hd])
                        {
                            *o += p * v;
                        }
                    }
                    for (o, gv) in oh
                        .iter_mut()
                        .zip(&gate[(t * nh + h) * hd..(t * nh + h + 1) * hd])
                    {
                        *o *= sigmoid(*gv);
                    }
                }
                out
            })
            .collect();
        let y = self
            .m
            .load(&self.name("attn_output.weight"))?
            .apply(&attn, nc, false);
        hc_write(&mut r, &y, &inj, nc, hc, n);

        // ---- MoE
        let (x, inj) = self.m.hc_read(&r, nc, &self.name("hc_ffn_"), true)?;
        let y = self.m.moe(self.layer, &x, nc, None)?;
        hc_write(&mut r, &y, &inj, nc, hc, n);

        // ---- head
        let (xh, _) = self
            .m
            .hc_read(&r, nc, &self.name("nextn.hc_head_"), false)?;
        let head = if self.main_head {
            main.load("output.weight")?
        } else {
            self.m.load("output.weight")?
        };
        let mut top = Vec::with_capacity(nc);
        for c0 in (0..nc).step_by(64) {
            let c1 = (c0 + 64).min(nc);
            let logits = head.apply(&xh[c0 * n..c1 * n], c1 - c0, false);
            for lg in logits.chunks(head.rows) {
                let t = super::reference::top_logprobs(lg, 1)[0];
                top.push((t.0, t.1.exp() as f32));
            }
        }
        if let Some(d) = dump {
            let t = nc - 1;
            d.f32(
                &format!("mtp.x_attn.{}", cells[t].pos),
                &[n],
                &x[t * n..(t + 1) * n],
            )?;
            d.f32(
                &format!("mtp.r_out.{}", cells[t].pos),
                &[hc, n],
                &r[t * hc * n..(t + 1) * hc * n],
            )?;
            d.f32(
                &format!("mtp.final_x.{}", cells[t].pos),
                &[n],
                &xh[t * n..(t + 1) * n],
            )?;
        }
        Ok(Out { r, top, kv_row })
    }
}

/// A chain alive at some depth: start cell, residual, draft token, its probability, the K/V
/// rows of the chain's cells so far.
type Chain = (usize, Vec<f32>, u32, f32, Vec<usize>);

/// Per-depth acceptance of chained MTP drafts against the main model's own greedy tokens.
#[derive(Default, Clone)]
pub struct Acceptance {
    /// Chains evaluated at this depth (the previous drafts were accepted and the target is known).
    pub tried: Vec<usize>,
    pub accepted: Vec<usize>,
    /// Starts that began a chain (depth-1 denominators), for cumulative rates.
    pub starts: usize,
    /// (draft prob bucket lower edge, tried, accepted) over every depth.
    pub calib: Vec<(f32, usize, usize)>,
}

/// Teacher-forced acceptance on one sequence.
///
/// `tokens` is the sequence, `h` the main model's final residuals at every position, `greedy[j]`
/// the main model's argmax after position `j`. For each start `i` in `from..`, the chain is
/// `d1 = MTP(h_i, tokens[i+1])`, `d2 = MTP(r1, d1)` at position `i+1`, … The depth-`k` target is the
/// main model's greedy token at position `i+k` *given the drafts*, which this one forward pass
/// knows only when the earlier drafts equal the sequence's own tokens: so depth `k` counts where
/// `d1..d(k-1)` were accepted and `greedy[i+j] == tokens[i+j+1]` for `j < k` (always true on the
/// model's own greedy text, which is why `--gen` sequences are the clean case).
#[allow(clippy::too_many_arguments)]
pub fn eval(
    mtp: &Mtp,
    main: &FlashRef,
    tokens: &[u32],
    h: &[f32],
    greedy: &[u32],
    from: usize,
    depth: usize,
    mut dump: Option<&mut Dump>,
) -> Result<Acceptance> {
    let hp = &mtp.m.hp;
    let w = hp.hc * hp.n_embd;
    let t_len = tokens.len();
    ensure!(
        t_len >= 3 && h.len() == t_len * w && greedy.len() == t_len,
        "MTP eval inputs"
    );
    // teacher-forced cells 0..t_len-1: (h_j, tokens[j+1]) at position j
    let tf: Vec<Cell> = (0..t_len - 1)
        .map(|j| Cell {
            h: &h[j * w..(j + 1) * w],
            tok: tokens[j + 1],
            pos: j,
            visible: (0..j).collect(),
        })
        .collect();
    let mut kv = Kv::default();
    let out = mtp.run(main, &tf, &mut kv, dump.as_deref_mut())?;

    let mut acc = Acceptance {
        tried: vec![0; depth],
        accepted: vec![0; depth],
        ..Default::default()
    };
    let mut calib = [(0usize, 0usize); 10];
    // chains alive at this depth: (start i, residual, draft token, kv rows of the chain so far)
    let starts: Vec<usize> = (from..t_len.saturating_sub(2)).collect();
    acc.starts = starts.len();
    let mut alive: Vec<Chain> = starts
        .iter()
        .map(|&i| {
            (
                i,
                out.r[i * w..(i + 1) * w].to_vec(),
                out.top[i].0,
                out.top[i].1,
                vec![],
            )
        })
        .collect();
    for d in 1..=depth {
        // depth d drafts the token at position i+d+1; its target is greedy[i+d]
        let mut next = Vec::new();
        for (i, r, tok, p, chain) in alive {
            if i + d >= t_len {
                continue;
            }
            // target known only if earlier drafts reproduced the sequence
            if (1..d).any(|j| greedy[i + j] != tokens[i + j + 1]) {
                continue;
            }
            let ok = tok == greedy[i + d];
            acc.tried[d - 1] += 1;
            let b = ((p * 10.0) as usize).min(9);
            calib[b].0 += 1;
            if ok {
                acc.accepted[d - 1] += 1;
                calib[b].1 += 1;
                // the next target is known only if this draft is also the sequence's token
                if d < depth && i + d + 1 < t_len && greedy[i + d] == tokens[i + d + 1] {
                    next.push((i, r, tok, chain));
                }
            }
        }
        if d == depth || next.is_empty() {
            break;
        }
        // run the depth d+1 cells: residual of the previous cell, token = accepted draft,
        // position i+d, attending to tf cells 0..=i and this chain's earlier cells
        let cells: Vec<Cell> = next
            .iter()
            .map(|(i, r, tok, chain)| {
                let mut vis: Vec<usize> = (0..=*i).collect();
                vis.extend(chain);
                Cell {
                    h: r,
                    tok: *tok,
                    pos: i + d,
                    visible: vis,
                }
            })
            .collect();
        let o = mtp.run(main, &cells, &mut kv, None)?;
        alive = next
            .iter()
            .enumerate()
            .map(|(k, (i, _, _, chain))| {
                let mut ch = chain.clone();
                ch.push(o.kv_row[k]);
                (
                    *i,
                    o.r[k * w..(k + 1) * w].to_vec(),
                    o.top[k].0,
                    o.top[k].1,
                    ch,
                )
            })
            .collect();
    }
    acc.calib = calib
        .iter()
        .enumerate()
        .map(|(b, &(t, a))| (b as f32 / 10.0, t, a))
        .collect();
    if let Some(d) = dump {
        let last = t_len - 2;
        d.u32("mtp.tf_top", &[out.top[last].0])?;
        d.f32("mtp.tf_top_p", &[1], &[out.top[last].1])?;
        d.u32("mtp.tf_tokens_in", &[tokens[last + 1]])?;
    }
    Ok(acc)
}

impl Acceptance {
    pub fn merge(&mut self, o: &Acceptance) {
        if self.tried.is_empty() {
            *self = o.clone();
            return;
        }
        for d in 0..self.tried.len() {
            self.tried[d] += o.tried[d];
            self.accepted[d] += o.accepted[d];
        }
        self.starts += o.starts;
        for (a, b) in self.calib.iter_mut().zip(&o.calib) {
            a.1 += b.1;
            a.2 += b.2;
        }
    }

    pub fn report(&self) -> String {
        let mut s = String::new();
        let mut cum = 1.0f64;
        let _ = writeln!(
            s,
            "depth | tried | accepted | conditional | cumulative (of {} starts)",
            self.starts
        );
        for d in 0..self.tried.len() {
            let c = self.accepted[d] as f64 / self.tried[d].max(1) as f64;
            cum *= c;
            let _ = writeln!(
                s,
                "{} | {} | {} | {:.3} | {:.3}",
                d + 1,
                self.tried[d],
                self.accepted[d],
                c,
                cum
            );
        }
        let _ = writeln!(s, "draft prob | tried | accepted | rate");
        for &(lo, t, a) in &self.calib {
            if t > 0 {
                let _ = writeln!(
                    s,
                    "[{:.1},{:.1}) | {t} | {a} | {:.3}",
                    lo,
                    lo + 0.1,
                    a as f64 / t as f64
                );
            }
        }
        s
    }
}

/// `tang-llm flash-mtp <main gguf> <mtp gguf> --ids-file F [--from I] [--depth D] [--joint-hnorm]
/// [--main-head] [--dump DIR]`: teacher-forced chained-draft acceptance (see [`eval`]).
/// `TANG_MTP_HCACHE=<file>` saves the main model's residuals and greedy tokens on the first run
/// and reads them back after, so MTP-only experiments skip the main forward.
pub fn cli(args: &[String]) -> Result<()> {
    let usage = "usage: flash-mtp <main gguf> <mtp gguf> --ids-file F [--from I] [--depth D] [--joint-hnorm] [--main-head] [--dump DIR] [-v]";
    let mut it = args.iter();
    let main_path = it.next().context(usage)?;
    let mtp_path = it.next().context(usage)?;
    let (mut ids, mut from, mut depth) = (Vec::<u32>::new(), 0usize, 3usize);
    let (mut joint, mut main_head, mut verbose) = (false, false, false);
    let mut dump_dir = None;
    while let Some(a) = it.next() {
        match a.as_str() {
            "--ids-file" => {
                let f = it.next().context(usage)?;
                for w in std::fs::read_to_string(f)?.split(|c: char| c.is_whitespace() || c == ',')
                {
                    if !w.is_empty() {
                        ids.push(w.parse().with_context(|| format!("bad id {w:?}"))?);
                    }
                }
            }
            "--from" => from = it.next().context(usage)?.parse()?,
            "--depth" => depth = it.next().context(usage)?.parse()?,
            "--joint-hnorm" => joint = true,
            "--main-head" => main_head = true,
            "--dump" => dump_dir = Some(std::path::PathBuf::from(it.next().context(usage)?)),
            "-v" => verbose = true,
            s => anyhow::bail!("unknown argument {s:?}; {usage}"),
        }
    }
    let mut main = FlashRef::open(Path::new(main_path))?;
    main.verbose = verbose;
    let mut mtp = Mtp::open(Path::new(mtp_path))?;
    mtp.joint_hnorm = joint;
    mtp.main_head = main_head;
    let mut dump = dump_dir.as_deref().map(Dump::new).transpose()?;
    let cache = std::env::var("TANG_MTP_HCACHE").ok();
    let (h, greedy) = match cache.as_deref().filter(|p| Path::new(p).exists()) {
        Some(p) => {
            let b = std::fs::read(p)?;
            let f: Vec<f32> = b
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            let w = main.hp.hc * main.hp.n_embd;
            let t = ids.len();
            (
                f[..t * w].to_vec(),
                f[t * w..].iter().map(|v| v.to_bits()).collect::<Vec<u32>>(),
            )
        }
        None => {
            let (tops, h) = main.forward_res(&ids, 0, 1, dump.as_mut())?;
            let greedy: Vec<u32> = tops.iter().map(|t| t.top[0].0).collect();
            if let Some(p) = &cache {
                let mut b: Vec<u8> = h.iter().flat_map(|v| v.to_le_bytes()).collect();
                b.extend(greedy.iter().flat_map(|v| v.to_le_bytes()));
                std::fs::write(p, b)?;
            }
            (h, greedy)
        }
    };
    let acc = eval(&mtp, &main, &ids, &h, &greedy, from, depth, dump.as_mut())?;
    if let Some(d) = dump.as_mut() {
        d.finish(&ids, ids.len() - 1)?;
    }
    print!("{}", acc.report());
    Ok(())
}

/// CPU oracle for the offline trainer's recursive mask (full-precision Q8 experts).
#[cfg(feature = "mtp-train")]
pub(crate) fn training_oracle(
    main: &FlashRef,
    mtp: &Mtp,
    h: &[f32],
    tokens: &[u32],
    pos: &[usize],
) -> Result<Vec<(Vec<f32>, Vec<(u32, f32)>)>> {
    let width = mtp.m.hp.hc * mtp.m.hp.n_embd;
    let len = pos.len();
    let mut residual = h.to_vec();
    let mut kv = Kv::default();
    let mut counts = Vec::new();
    let mut result = Vec::new();
    for depth in 1..=3 {
        let rows = len - depth;
        let cells: Vec<_> = (0..rows)
            .map(|i| {
                let mut visible: Vec<_> = (0..=i).collect();
                if depth == 1 {
                    visible.pop();
                } else {
                    let mut offset = counts[0];
                    for &count in &counts[1..] {
                        visible.push(offset + i);
                        offset += count;
                    }
                }
                Cell {
                    h: &residual[i * width..(i + 1) * width],
                    tok: tokens[i + depth],
                    pos: pos[i] + depth - 1,
                    visible,
                }
            })
            .collect();
        let out = mtp.run(main, &cells, &mut kv, None)?;
        result.push((out.r.clone(), out.top));
        residual = out.r;
        counts.push(rows);
    }
    Ok(result)
}
