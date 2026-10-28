//! Lowering: a scheduled [`Graph`] to Qwen3.5 `linear_attention` decoder-layer weights, a port
//! of qwenputer's `chipformer/q35/emit.py` (MIT, Copyright (c) 2026 Divyansh Agrawal). Each
//! lowering rule cites the python line it ports.
//!
//! Residual layout of the chip region, starting at `hidden_base` (the host model's own dims come
//! first): `[ballast × m] [tau, unused] [slots…] [unused]` (emit.py:1-12). Chip tokens carry the
//! ballast `B = 2^40` in every ballast dim; with `m · B² / hidden` a power of 4 every RMSNorm on a
//! chip token divides by exactly `√(m / hidden) · B`, so a residual value `x` reaches the
//! projections as `x · c`, `c = √(hidden / m) / B` a power of 2, and weights carry `1 / c`. The IR
//! constant ONE is the normed ballast.
//!
//! Lowering rules (the ops `emit_layers` handles):
//! - **linear form → projection row** (`row`, emit.py:160): `w · coef / c` per dim; a constant
//!   is split into ≤ m bf16 parts on the ballast dims (`bf16_parts`, emit.py:146).
//! - **head placement**: the j-th head at sublayer `2L + 1` is key head j of layer L; its x-th
//!   value head is v head `j · r + x`, `r = v_heads / k_heads` (emit.py:212-240).
//! - **q / k / v channels → `in_proj_qkv` rows + conv taps** (emit.py:233-244).
//! - **β pinning**: `b = 128` (β = sigmoid(128) = 1) or `256 g − 128` for a 0/1 write gate
//!   (emit.py:246-247).
//! - **decay**: `a` row as is (`dt_bias = −110`, `A_log = 0`: `a ∈ {0, 110, 238}` gives
//!   `exp(g) ∈ {1, ½, 0}`) (emit.py:248, 284-285).
//! - **output gates** `z` rows; **outputs** `out_proj[slot(d), vh·128 + col] += w`
//!   (emit.py:249-252).
//! - **FFN unit** `coef · a · relu(b)` → `gate = K·b`, `up = a / K`, `down[slot(out)] = coef`
//!   (emit.py:255-269).
//! - **slot handover**: `−old · relu(ONE)` units (`subtract`, `clears`) (emit.py:259-262).
//! - **exit**: the last MLP clears every live dim but `keep`, and the ballast (emit.py:263-275).
//! - **constants**: zero (`1 + w = 1`) RMSNorm weights, `dt_bias = −110`, `A_log = 0`, gated
//!   norm 1 (emit.py:277-291).
//! - **embedding rows**: ballast plus token features (emit.py:298).

use std::collections::HashMap;

use anyhow::{bail, ensure, Result};

use crate::alloc::{allocate, Alloc};
use crate::ir::{Expr, Graph, Node, HD, K, ONE};

/// The host model's GDN layer shape and where the chip goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChipShape {
    /// The host model's hidden size; the chip dims follow it.
    pub hidden_base: usize,
    /// Chip dims appended (ballast + slots + padding); `None`: the smallest that fits
    /// ([`pick_hidden`]).
    pub hidden_chip: Option<usize>,
    /// Ballast dims for a given `hidden_chip` (`None` with `hidden_chip: None`).
    pub ballast: Option<usize>,
    pub k_heads: usize,
    pub v_heads: usize,
    /// GDN head dim; the IR is written for 128.
    pub d_state: usize,
    pub d_conv: usize,
    /// MLP width of the chip layers.
    pub inter: usize,
    /// Layers to emit (≥ the circuit's; extra layers are empty).
    pub n_layers: usize,
    /// First chip token id in the model vocabulary.
    pub vocab_offset: u32,
}

impl ChipShape {
    /// qwenputer's Filesystem checkpoint: Qwen3.5-4B (2560, 16 k / 32 v heads) widened to 4608.
    pub const QWEN35_4B: ChipShape = ChipShape {
        hidden_base: 2560,
        hidden_chip: Some(2048),
        ballast: Some(18),
        k_heads: 16,
        v_heads: 32,
        d_state: 128,
        d_conv: 4,
        inter: 9216,
        n_layers: 12,
        vocab_offset: 248320,
    };
}

/// q35/merge.py:161 `pick_hidden`: the smallest `(hidden, m)` with `hidden = m · 4^k ≥ base +
/// m + n_chip`, a multiple of 128, `3 ≤ m ≤ 64` (`m · B² / hidden` must be a power of 4).
pub fn pick_hidden(base: usize, n_chip: usize) -> (usize, usize) {
    let mut best: Option<(usize, usize)> = None;
    for m in 3..65 {
        let mut h = m;
        while h < base + m + n_chip || h % 128 != 0 {
            h *= 4;
        }
        if best.is_none_or(|(bh, bm)| h < bh || (h == bh && m > bm)) {
            best = Some((h, m));
        }
    }
    best.unwrap()
}

/// emit.py:24 `Layout`.
#[derive(Clone, Debug)]
pub struct Layout {
    pub hidden: usize,
    /// First chip dim.
    pub off: usize,
    /// Ballast dims.
    pub m: usize,
    pub log2_ballast: i32,
    /// IR dim → residual slot.
    pub slots: HashMap<u32, usize>,
}

impl Layout {
    pub fn ballast(&self) -> f64 {
        2f64.powi(self.log2_ballast)
    }
    /// emit.py:36: `c`, the normed value of a residual 1 on a chip token.
    pub fn scale(&self) -> Result<f64> {
        let mean = self.m as f64 * self.ballast().powi(2) / self.hidden as f64;
        let log4 = mean.ln() / 4f64.ln();
        ensure!(
            (log4 - log4.round()).abs() < 1e-12,
            "m · B² / hidden must be a power of 4"
        );
        Ok(2f64.powi(-(log4.round() as i32)))
    }
    /// emit.py:47: residual index of IR dim `d ≥ 1`.
    pub fn dim(&self, d: u32) -> usize {
        self.off + self.m + 1 + self.slots[&d]
    }
}

/// A weight matrix as its nonzero entries, `(row, col, value)` sorted row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct Sparse {
    pub rows: usize,
    pub cols: usize,
    pub entries: Vec<(u32, u32, f32)>,
}

impl Sparse {
    pub fn nnz(&self) -> usize {
        self.entries.len()
    }
    /// Dense row-major (for small shapes and checks).
    pub fn to_dense(&self) -> Vec<f32> {
        let mut d = vec![0f32; self.rows * self.cols];
        for &(r, c, v) in &self.entries {
            d[r as usize * self.cols + c as usize] = v;
        }
        d
    }
}

/// An f32 accumulator with torch's `t[i] += x` semantics (the f64 value cast to f32, added in
/// f32) and `t[i] = x`.
struct Acc {
    rows: usize,
    cols: usize,
    m: HashMap<(u32, u32), f32>,
}

impl Acc {
    fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            m: HashMap::new(),
        }
    }
    fn add(&mut self, r: usize, c: usize, v: f64) {
        debug_assert!(
            r < self.rows && c < self.cols,
            "({r}, {c}) outside {}×{}",
            self.rows,
            self.cols
        );
        *self.m.entry((r as u32, c as u32)).or_default() += v as f32;
    }
    fn set(&mut self, r: usize, c: usize, v: f64) {
        debug_assert!(r < self.rows && c < self.cols);
        self.m.insert((r as u32, c as u32), v as f32);
    }
    fn finish(self, name: &str) -> Result<Sparse> {
        let mut entries: Vec<(u32, u32, f32)> = self
            .m
            .into_iter()
            .filter(|e| e.1 != 0.0)
            .map(|((r, c), v)| (r, c, v))
            .collect();
        entries.sort_unstable_by_key(|e| (e.0, e.1));
        for &(r, c, v) in &entries {
            check_bf16(name, r, c, v)?;
        }
        Ok(Sparse {
            rows: self.rows,
            cols: self.cols,
            entries,
        })
    }
}

/// Round to bf16, nearest-even (`ir.py` `bf16`).
pub fn round_bf16(x: f32) -> f32 {
    let u = x.to_bits();
    f32::from_bits(u.wrapping_add(0x7FFF + ((u >> 16) & 1)) & 0xFFFF_0000)
}

/// emit.py:175 `check_bf16`: every weight must be a finite bf16 value (stored exactly).
fn check_bf16(name: &str, r: u32, c: u32, v: f32) -> Result<()> {
    if round_bf16(v) != v || !v.is_finite() {
        bail!("{name}: weight {v} at ({r}, {c}) is not a bf16 value");
    }
    Ok(())
}

/// emit.py:146 `bf16_parts`: x as ≤ n bf16 values summing to it exactly (torch's double →
/// bf16 goes through f32).
fn bf16_parts(mut x: f64, n: usize) -> Result<Vec<f64>> {
    let mut parts = vec![];
    for _ in 0..n {
        if x == 0.0 {
            break;
        }
        let p = round_bf16(x as f32) as f64;
        parts.push(p);
        x -= p;
    }
    ensure!(x == 0.0, "constant needs more than {n} bf16 parts");
    Ok(parts)
}

/// emit.py:160 `row`: add `w · e` as projection row `r` (input side: the normed residual).
fn row(e: &Expr, lay: &Layout, c: f64, out: &mut Acc, r: usize, w: f64) -> Result<()> {
    for &(d, coef) in &e.0 {
        if d == ONE {
            for (i, p) in bf16_parts(w * coef / (lay.ballast() * c), lay.m)?
                .into_iter()
                .enumerate()
            {
                out.add(r, lay.off + i, p);
            }
        } else {
            out.add(r, lay.dim(d), w * coef / c);
        }
    }
    Ok(())
}

/// One emitted decoder layer, HF tensor by tensor (emit.py:276-291). Norm weights are HF's raw
/// zero-centred values (`1 + w` applied by the model).
#[derive(Clone, Debug)]
pub struct LayerWeights {
    pub input_layernorm: Vec<f32>,
    pub post_attention_layernorm: Vec<f32>,
    pub in_proj_qkv: Sparse,
    pub in_proj_z: Sparse,
    pub in_proj_a: Sparse,
    pub in_proj_b: Sparse,
    /// `[conv_dim][d_conv]`, tap `d_conv − 1` the current token (HF `[C, 1, d_conv]`).
    pub conv1d: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub a_log: Vec<f32>,
    pub norm: Vec<f32>,
    pub out_proj: Sparse,
    pub gate_proj: Sparse,
    pub up_proj: Sparse,
    pub down_proj: Sparse,
}

/// A tensor of [`LayerWeights`] by its HF name (under `model.layers.{i}.`).
pub enum TensorRef<'a> {
    Sparse(&'a Sparse),
    Dense(&'a [f32]),
}

impl LayerWeights {
    pub fn tensors(&self) -> Vec<(&'static str, TensorRef<'_>)> {
        use TensorRef::{Dense as D, Sparse as S};
        vec![
            ("input_layernorm.weight", D(&self.input_layernorm)),
            (
                "post_attention_layernorm.weight",
                D(&self.post_attention_layernorm),
            ),
            ("linear_attn.in_proj_qkv.weight", S(&self.in_proj_qkv)),
            ("linear_attn.in_proj_z.weight", S(&self.in_proj_z)),
            ("linear_attn.in_proj_a.weight", S(&self.in_proj_a)),
            ("linear_attn.in_proj_b.weight", S(&self.in_proj_b)),
            ("linear_attn.conv1d.weight", D(&self.conv1d)),
            ("linear_attn.dt_bias", D(&self.dt_bias)),
            ("linear_attn.A_log", D(&self.a_log)),
            ("linear_attn.norm.weight", D(&self.norm)),
            ("linear_attn.out_proj.weight", S(&self.out_proj)),
            ("mlp.gate_proj.weight", S(&self.gate_proj)),
            ("mlp.up_proj.weight", S(&self.up_proj)),
            ("mlp.down_proj.weight", S(&self.down_proj)),
        ]
    }
    pub fn nnz(&self) -> usize {
        self.tensors()
            .iter()
            .map(|(_, t)| match t {
                TensorRef::Sparse(s) => s.nnz(),
                TensorRef::Dense(_) => 0,
            })
            .sum()
    }
}

/// The emitted chip stack.
#[derive(Clone, Debug)]
pub struct Compiled {
    pub shape: ChipShape,
    pub layout: Layout,
    pub alloc: Alloc,
    pub layers: Vec<LayerWeights>,
    /// Hidden size of the layers (`hidden_base + hidden_chip`).
    pub hidden: usize,
    /// Residual indices the output decoder reads: the output code, the store's id code.
    pub code_dims: Vec<usize>,
    pub id_code_dims: Vec<usize>,
    pub tau_dim: usize,
    pub mu_dim: usize,
}

impl Compiled {
    /// emit.py:298 `embedding_rows` for one token given its IR features `(dim, value)`: the
    /// ballast, then each feature at its slot. The host's own dims (`0..hidden_base`) are left
    /// zero: the caller puts the host embedding there (the norms ignore it next to the ballast).
    pub fn embedding_row(&self, features: &[(u32, f64)]) -> Result<Vec<f32>> {
        let lay = &self.layout;
        let mut e = vec![0f32; self.hidden];
        e[lay.off..lay.off + lay.m].fill(lay.ballast() as f32);
        for &(d, v) in features {
            let i = lay.dim(d);
            e[i] = v as f32;
            check_bf16("embedding", 0, i as u32, e[i])?;
        }
        Ok(e)
    }
}

/// Emit the chip layers of `g` for `shape` (emit.py:196 `emit_layers` with the store build's
/// allocation, q35/merge.py / rv/merge.py:138-141 `allocate(c, None, keep, inter)` and
/// `Layout(hidden, off = base, m, slots)`).
pub fn emit(g: &Graph, shape: &ChipShape) -> Result<Compiled> {
    let (nk, nv) = (shape.k_heads, shape.v_heads);
    ensure!(
        shape.d_state == HD,
        "the IR is written for 128-dim heads, not {}",
        shape.d_state
    );
    ensure!(shape.d_conv == 4, "the IR's conv taps are 4 wide");
    // emit.py:209: value head i reads key head i / r; a circuit head uses value heads j·r + x
    let r = nv / nk;
    ensure!(r * nk == nv, "v heads {nv} not a multiple of k heads {nk}");
    let n_circ = g.n_layers();
    ensure!(
        shape.n_layers >= n_circ,
        "the circuit needs {n_circ} layers, shape has {}",
        shape.n_layers
    );
    ensure!(
        g.target.key_heads <= nk && g.target.intermediate <= shape.inter,
        "the graph was scheduled for {:?}; the shape has {nk} k heads, MLP {}",
        g.target,
        shape.inter
    );

    let alloc = allocate(g, &g.keep, g.target.intermediate);
    let (hidden, m) = match (shape.hidden_chip, shape.ballast) {
        (Some(hc), Some(m)) => (shape.hidden_base + hc, m),
        (None, None) => pick_hidden(shape.hidden_base, 1 + alloc.n_slots),
        _ => bail!("give hidden_chip and ballast together"),
    };
    ensure!(
        shape.hidden_base + m + 1 + alloc.n_slots <= hidden,
        "{} slots + {m} ballast do not fit in {} chip dims",
        alloc.n_slots,
        hidden - shape.hidden_base
    );
    let lay = Layout {
        hidden,
        off: shape.hidden_base,
        m,
        log2_ballast: 40,
        slots: alloc.slots.clone(),
    };
    let c = lay.scale()?;
    let h = hidden;
    let conv_dim = 2 * nk * HD + nv * HD;

    // emit.py:205-211: the exit MLP clears every live dim but `keep`
    let is_exit = |n: &Node| n.name().starts_with("exit.");
    let exit_level = g
        .nodes
        .iter()
        .filter(|n| is_exit(n))
        .map(|n| n.level())
        .max();
    if let Some(el) = exit_level {
        ensure!(
            g.nodes.iter().all(|n| is_exit(n) || n.level() < el),
            "exit nodes must be last"
        );
    }
    let exit_clear: Vec<u32> = alloc
        .final_
        .iter()
        .copied()
        .filter(|d| !g.keep.contains(d))
        .collect();
    let one = Expr::dim(ONE, 1.0);

    let mut layers = Vec::with_capacity(shape.n_layers);
    for l in 0..shape.n_layers {
        let lvl_head = 2 * l as u32 + 1;
        let lvl_ffn = 2 * l as u32 + 2;
        let name = |t: &str| format!("layers.{l}.{t}");
        let mut qkv = Acc::new(conv_dim, h);
        let mut convw = vec![0f32; conv_dim * shape.d_conv];
        let mut zw = Acc::new(nv * HD, h);
        let mut bw = Acc::new(nv, h);
        let mut aw = Acc::new(nv, h);
        let mut ow = Acc::new(h, nv * HD);
        let mut gate = Acc::new(shape.inter, h);
        let mut up = Acc::new(shape.inter, h);
        let mut down = Acc::new(h, shape.inter);

        // emit.py:226-252: heads
        let heads = g.nodes.iter().filter_map(|n| match n {
            Node::Head(hd) if hd.level == lvl_head => Some(hd),
            _ => None,
        });
        for (j, n) in heads.enumerate() {
            ensure!(j < nk, "layer {l}: more than {nk} heads");
            ensure!(
                n.vheads.len() <= r,
                "{}: {} value heads, {r} per key head",
                n.name,
                n.vheads.len()
            );
            let conv_set = |convw: &mut Vec<f32>, ch: usize, taps: &[f64; 4]| {
                for (lag, &w) in taps.iter().enumerate() {
                    convw[ch * shape.d_conv + shape.d_conv - 1 - lag] = w as f32;
                    // emit.py:237
                }
            };
            for (base, chans) in [(j * HD, &n.q), (nk * HD + j * HD, &n.k)] {
                for (i, ch) in chans {
                    row(&ch.expr, &lay, c, &mut qkv, base + i, 1.0)?; // emit.py:235
                    conv_set(&mut convw, base + i, &ch.taps);
                }
            }
            for (x, vhd) in n.vheads.iter().enumerate() {
                let vh = j * r + x; // emit.py:240
                let base = 2 * nk * HD + vh * HD;
                for (i, ch) in &vhd.v {
                    row(&ch.expr, &lay, c, &mut qkv, base + i, 1.0)?; // emit.py:242
                    conv_set(&mut convw, base + i, &ch.taps);
                }
                // emit.py:246: β = sigmoid(128) = 1, or a write gate sigmoid(256 g − 128)
                let beta = match &vhd.beta {
                    None => Expr::dim(ONE, 128.0),
                    Some(b) => b.scale(256.0).add(&one.scale(-128.0)),
                };
                row(&beta, &lay, c, &mut bw, vh, 1.0)?;
                row(&vhd.a, &lay, c, &mut aw, vh, 1.0)?; // emit.py:248
                for (col, e) in &vhd.z {
                    row(e, &lay, c, &mut zw, vh * HD + col, 1.0)?; // emit.py:250
                }
                for &(col, d, w) in &vhd.out {
                    ow.add(lay.dim(d), vh * HD + col, w); // emit.py:252
                }
            }
        }

        // emit.py:254-264: MLP units (coef, a, b, output dim)
        let mut units: Vec<(f64, Expr, Expr, u32)> = vec![];
        for n in &g.nodes {
            let Node::Ffn(f) = n else { continue };
            if f.level != lvl_ffn {
                continue;
            }
            units.extend(
                f.terms
                    .iter()
                    .map(|(k, a, b)| (*k, a.clone(), b.clone(), f.out)),
            );
            if let Some(&old) = alloc.subtract.get(&f.out) {
                units.push((-1.0, Expr::dim(old, 1.0), one.clone(), f.out)); // emit.py:260
            }
        }
        for &o in alloc.clears.get(&lvl_ffn).into_iter().flatten() {
            units.push((-1.0, Expr::dim(o, 1.0), one.clone(), o)); // emit.py:261
        }
        let exit_here = exit_level == Some(lvl_ffn);
        if exit_here {
            units.extend(
                exit_clear
                    .iter()
                    .map(|&o| (-1.0, Expr::dim(o, 1.0), one.clone(), o)),
            );
        }
        ensure!(
            units.len() + if exit_here { m } else { 0 } <= shape.inter,
            "layer {l}: {} units",
            units.len()
        );
        for (u, (coef, a, b, d)) in units.iter().enumerate() {
            row(b, &lay, c, &mut gate, u, K)?; // emit.py:267
            row(a, &lay, c, &mut up, u, 1.0 / K)?; // emit.py:268
            down.set(lay.dim(*d), u, *coef); // emit.py:269
        }
        if exit_here {
            // emit.py:270-275: the ballast goes too, −B per ballast dim
            for i in 0..m {
                let u = units.len() + i;
                row(&Expr::dim(ONE, K), &lay, c, &mut gate, u, 1.0)?;
                up.set(u, lay.off + i, (lay.ballast() / K) / (lay.ballast() * c));
                down.set(lay.off + i, u, -1.0);
            }
        }
        for v in &convw {
            check_bf16(&name("conv1d"), 0, 0, *v)?;
        }
        layers.push(LayerWeights {
            input_layernorm: vec![0.0; h],
            post_attention_layernorm: vec![0.0; h],
            in_proj_qkv: qkv.finish(&name("in_proj_qkv"))?,
            in_proj_z: zw.finish(&name("in_proj_z"))?,
            in_proj_a: aw.finish(&name("in_proj_a"))?,
            in_proj_b: bw.finish(&name("in_proj_b"))?,
            conv1d: convw,
            dt_bias: vec![-110.0; nv],
            a_log: vec![0.0; nv],
            norm: vec![1.0; HD],
            out_proj: ow.finish(&name("out_proj"))?,
            gate_proj: gate.finish(&name("gate_proj"))?,
            up_proj: up.finish(&name("up_proj"))?,
            down_proj: down.finish(&name("down_proj"))?,
        });
    }
    let code_dims = g.code.iter().map(|&d| lay.dim(d)).collect();
    let id_code_dims = g.id_code.iter().map(|&d| lay.dim(d)).collect();
    let (tau_dim, mu_dim) = (lay.dim(g.tau), lay.dim(g.mu));
    Ok(Compiled {
        shape: *shape,
        layout: lay,
        alloc,
        layers,
        hidden,
        code_dims,
        id_code_dims,
        tau_dim,
        mu_dim,
    })
}
