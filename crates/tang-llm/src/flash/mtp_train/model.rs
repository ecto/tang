use super::{
    data::Batch,
    tape::{Id, Tape},
    weights::Weights,
};
use crate::flash::reference::Hparams;
use anyhow::{ensure, Result};
use std::collections::BTreeMap;
use tang_compute::ComputeDevice;

pub struct CellOut {
    pub r: Id,
    pub k: Id,
    pub v: Id,
    pub logits: Id,
}
fn linear<D: ComputeDevice>(
    t: &mut Tape<D>,
    w: &Weights,
    ids: &BTreeMap<String, Id>,
    x: Id,
    name: &str,
    rows: usize,
) -> Id {
    let mat = &w.dense[name];
    t.linear(x, ids[name], rows, mat.cols, mat.rows)
}
fn hc_read<D: ComputeDevice>(
    t: &mut Tape<D>,
    w: &Weights,
    ids: &BTreeMap<String, Id>,
    r: Id,
    rows: usize,
    pre: &str,
    hp: &Hparams,
    inject: bool,
) -> (Id, Option<Id>) {
    let n = hp.n_embd;
    let hc = hp.hc;
    let xn = t.norm(r, ids[&format!("{pre}norm.weight")], n, hp.eps);
    let lo = linear(t, w, ids, xn, &format!("{pre}down.weight"), rows);
    let lo = t.scale(lo, 1.0 / hc as f32);
    let lo = t.nonlinear(lo, true);
    let gate = linear(t, w, ids, lo, &format!("{pre}up.weight"), rows);
    let gate = t.nonlinear(gate, false);
    let mixed = t.mul(xn, gate);
    let mut streams = Vec::new();
    for c in 0..hc {
        streams.push(
            t.gather(
                mixed,
                (0..rows)
                    .flat_map(|r| (0..n).map(move |j| (r * hc + c) * n + j))
                    .collect(),
            ),
        );
    }
    let mut x = streams[0];
    for &s in &streams[1..] {
        x = t.add(x, s);
    }
    let x = t.scale(x, 1.0 / hc as f32);
    let inj = inject.then(|| linear(t, w, ids, xn, &format!("{pre}inject.weight"), rows));
    (x, inj)
}
fn hc_write<D: ComputeDevice>(
    t: &mut Tape<D>,
    r: Id,
    y: Id,
    inj: Id,
    rows: usize,
    hp: &Hparams,
) -> Id {
    let g = t.scale(inj, 1.0 / hp.hc as f32);
    let g = t.nonlinear(g, false);
    let g = t.scale(g, 2.0);
    let g = t.gather(
        g,
        (0..rows * hp.hc)
            .flat_map(|i| std::iter::repeat_n(i, hp.n_embd))
            .collect(),
    );
    let y = t.gather(
        y,
        (0..rows)
            .flat_map(|r| {
                (0..hp.hc).flat_map(move |_| (0..hp.n_embd).map(move |j| r * hp.n_embd + j))
            })
            .collect(),
    );
    let y = t.mul(y, g);
    t.add(r, y)
}
fn moe<D: ComputeDevice>(
    t: &mut Tape<D>,
    w: &mut Weights,
    ids: &BTreeMap<String, Id>,
    x: Id,
    rows: usize,
    hp: &Hparams,
) -> Result<Id> {
    let (n, ff, k, ne) = (hp.n_embd, hp.n_ff_exp, hp.n_expert_used, hp.n_expert);
    let logits = linear(t, w, ids, x, "ffn_gate_inp.weight", rows);
    let mut selected = Vec::new();
    let mut groups: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
    for r in 0..rows {
        let lg = &t.data(logits)[r * ne..(r + 1) * ne];
        let mut idx: Vec<_> = (0..ne).collect();
        idx.sort_by(|&a, &b| lg[b].total_cmp(&lg[a]).then(a.cmp(&b)));
        idx.truncate(k);
        for (rank, &e) in idx.iter().enumerate() {
            selected.push(r * ne + e);
            groups.entry(e).or_default().push((r, rank));
        }
    }
    let route = t.gather(logits, selected);
    let route = t.softmax(route, k);
    let route = t.scale(
        route,
        if hp.expert_weights_scale == 0.0 {
            1.0
        } else {
            hp.expert_weights_scale
        },
    );
    let mut y = t.constant(vec![0.0; rows * n]);
    for (e, pairs) in groups {
        let xe = t.gather(
            x,
            pairs
                .iter()
                .flat_map(|&(r, _)| (0..n).map(move |j| r * n + j))
                .collect(),
        );
        let gate = t.leaf(w.expert("ffn_gate_exps.weight", e)?, false);
        let up = t.leaf(w.expert("ffn_up_exps.weight", e)?, false);
        let down = t.leaf(w.expert("ffn_down_exps.weight", e)?, false);
        let gv = t.linear(xe, gate, pairs.len(), n, ff);
        let gv = t.nonlinear(gv, true);
        let uv = t.linear(xe, up, pairs.len(), n, ff);
        let h = t.mul(gv, uv);
        let ye = t.linear(h, down, pairs.len(), ff, n);
        let rw = t.gather(
            route,
            pairs
                .iter()
                .flat_map(|&(r, rank)| std::iter::repeat_n(r * k + rank, n))
                .collect(),
        );
        let ye = t.mul(ye, rw);
        let zero = t.constant(vec![0.0; n]);
        let padded = t.join(&[ye, zero]);
        let mut inverse = vec![pairs.len(); rows];
        for (j, &(r, _)) in pairs.iter().enumerate() {
            inverse[r] = j;
        }
        let ye = t.gather(
            padded,
            inverse
                .into_iter()
                .flat_map(|j| (0..n).map(move |d| j * n + d))
                .collect(),
        );
        y = t.add(y, ye);
    }
    let gate = linear(t, w, ids, x, "ffn_gate_shexp.weight", rows);
    let gate = t.nonlinear(gate, true);
    let up = linear(t, w, ids, x, "ffn_up_shexp.weight", rows);
    let h = t.mul(gate, up);
    let sh = linear(t, w, ids, h, "ffn_down_shexp.weight", rows);
    let sg = linear(t, w, ids, x, "ffn_gate_inp_shexp.weight", rows);
    let sg = t.nonlinear(sg, false);
    let sg = t.gather(
        sg,
        (0..rows).flat_map(|r| std::iter::repeat_n(r, n)).collect(),
    );
    let sh = t.mul(sh, sg);
    Ok(t.add(y, sh))
}
#[allow(clippy::too_many_arguments)]
fn cell<D: ComputeDevice>(
    t: &mut Tape<D>,
    w: &mut Weights,
    ids: &BTreeMap<String, Id>,
    h: Id,
    tokens: &[u32],
    pos: &[usize],
    previous: &[(Id, Id)],
    visible: Vec<Vec<usize>>,
    hp: &Hparams,
) -> Result<CellOut> {
    let (n, hc, rows) = (hp.n_embd, hp.hc, tokens.len());
    let ti = w.main.info("token_embd.weight")?;
    let mut emb = Vec::with_capacity(rows * n);
    for &tok in tokens {
        emb.extend(w.main.rows(ti, tok as usize, 1)?);
    }
    let e = t.constant(emb);
    let e = t.norm(e, ids["nextn.enorm.weight"], n, hp.eps);
    let hn = t.norm(h, ids["nextn.hnorm.weight"], n, hp.eps);
    let packed = t.join(&[e, hn]);
    let mut cat = Vec::with_capacity(rows * hc * 2 * n);
    for r in 0..rows {
        for c in 0..hc {
            cat.extend((0..n).map(|j| r * n + j));
            cat.extend((0..n).map(|j| rows * n + (r * hc + c) * n + j));
        }
    }
    let cat = t.gather(packed, cat);
    let r = linear(t, w, ids, cat, "nextn.eh_proj.weight", rows * hc);
    let (x, inj) = hc_read(t, w, ids, r, rows, "hc_attn_", hp, true);
    let qf = linear(t, w, ids, x, "attn_q.weight", rows);
    let nh = hp.n_head;
    let nkv = hp.n_head_kv;
    let hd = hp.head_dim;
    let q = t.gather(
        qf,
        (0..rows * nh)
            .flat_map(|i| (0..hd).map(move |j| i * 2 * hd + j))
            .collect(),
    );
    let gate = t.gather(
        qf,
        (0..rows * nh)
            .flat_map(|i| (0..hd).map(move |j| i * 2 * hd + hd + j))
            .collect(),
    );
    let q = t.norm(q, ids["attn_q_norm.weight"], hd, hp.eps);
    let q = t.rope(q, nh, hd, hp.n_rot, hp.rope_base, pos);
    let k = linear(t, w, ids, x, "attn_k.weight", rows);
    let k = t.norm(k, ids["attn_k_norm.weight"], hd, hp.eps);
    let k = t.rope(k, nkv, hd, hp.n_rot, hp.rope_base, pos);
    let v = linear(t, w, ids, x, "attn_v.weight", rows);
    let ks: Vec<_> = previous
        .iter()
        .map(|p| p.0)
        .chain(std::iter::once(k))
        .collect();
    let vs: Vec<_> = previous
        .iter()
        .map(|p| p.1)
        .chain(std::iter::once(v))
        .collect();
    let allk = t.join(&ks);
    let allv = t.join(&vs);
    let a = t.attention(q, allk, allv, nh, nkv, hd, visible);
    let gate = t.nonlinear(gate, false);
    let a = t.mul(a, gate);
    let a = linear(t, w, ids, a, "attn_output.weight", rows);
    let r = hc_write(t, r, a, inj.unwrap(), rows, hp);
    let (x, inj) = hc_read(t, w, ids, r, rows, "hc_ffn_", hp, true);
    let y = moe(t, w, ids, x, rows, hp)?;
    let r = hc_write(t, r, y, inj.unwrap(), rows, hp);
    let (x, _) = hc_read(t, w, ids, r, rows, "nextn.hc_head_", hp, false);
    let logits = linear(t, w, ids, x, "output.weight", rows);
    Ok(CellOut { r, k, v, logits })
}
/// Depth k uses MTP residuals from k-1 and teacher token i+k. Attention sees teacher cells
/// through i and only its own preceding recursive cells, exactly the inference chain mask.
pub fn forward_full<D: ComputeDevice>(
    t: &mut Tape<D>,
    w: &mut Weights,
    ids: &BTreeMap<String, Id>,
    b: &Batch,
    hp: &Hparams,
) -> Result<Vec<CellOut>> {
    let len = b.pos.len();
    ensure!(
        len >= 4,
        "three-step training needs at least four residual rows"
    );
    let width = hp.n_embd * hp.hc;
    let mut residual = t.constant(b.h.clone());
    let mut prev = Vec::new();
    let mut logits = Vec::new();
    let mut counts = Vec::new();
    for depth in 1..=3 {
        let rows = len - depth;
        let h = t.gather(residual, (0..rows * width).collect());
        let tokens = &b.tokens[depth..depth + rows];
        let pos: Vec<_> = b.pos[..rows].iter().map(|&p| p + depth - 1).collect();
        let mut visible = Vec::new();
        for i in 0..rows {
            let mut mask: Vec<_> = (0..=i).collect();
            if depth > 1 {
                let mut offset = counts[0];
                for &count in &counts[1..] {
                    mask.push(offset + i);
                    offset += count;
                }
                mask.push(counts.iter().sum::<usize>() + i);
            }
            visible.push(mask);
        }
        let out = cell(t, w, ids, h, tokens, &pos, &prev, visible, hp)?;
        residual = out.r;
        prev.push((out.k, out.v));
        logits.push(out);
        counts.push(rows);
    }
    Ok(logits)
}
pub fn objective<D: ComputeDevice>(
    t: &Tape<D>,
    w: &Weights,
    logits: &[Id],
    b: &Batch,
    beta: f64,
    burn: usize,
) -> Result<([f64; 3], Vec<(Id, Vec<f32>)>, [usize; 3], [usize; 3])> {
    let denom = 1.0 + beta + beta * beta;
    let mut losses = [0.0; 3];
    let mut hits = [0; 3];
    let mut totals = [0; 3];
    let mut seeds = Vec::new();
    let vocab = w.vocab.len();
    for (d, &id) in logits.iter().enumerate() {
        let rows = t.data(id).len() / vocab;
        ensure!(burn < rows, "burn-in leaves no loss rows");
        let mut g = vec![0.0; rows * vocab];
        for r in burn..rows {
            let label = b.tokens[r + d + 2];
            let target = w.vocab.binary_search(&label).map_err(|_| {
                anyhow::anyhow!("label {label} excluded from vocabulary; use --vocab 0")
            })?;
            let row = &t.data(id)[r * vocab..(r + 1) * vocab];
            let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let z = row.iter().map(|&v| (v as f64 - mx).exp()).sum::<f64>();
            losses[d] += mx + z.ln() - row[target] as f64;
            let best = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(&a.0)))
                .unwrap()
                .0;
            hits[d] += usize::from(best == target);
            totals[d] += 1;
            let weight = beta.powi(d as i32) / denom / (rows - burn) as f64;
            for j in 0..vocab {
                g[r * vocab + j] =
                    (((row[j] as f64 - mx).exp() / z - f64::from(j == target)) * weight) as f32;
            }
        }
        losses[d] /= totals[d] as f64;
        seeds.push((id, g));
    }
    Ok((losses, seeds, hits, totals))
}

pub fn forward<D: ComputeDevice>(
    t: &mut Tape<D>,
    w: &mut Weights,
    ids: &BTreeMap<String, Id>,
    b: &Batch,
    hp: &Hparams,
) -> Result<Vec<Id>> {
    Ok(forward_full(t, w, ids, b, hp)?
        .iter()
        .map(|o| o.logits)
        .collect())
}
