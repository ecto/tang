//! Compiled store layers ("chip layers") run in tang's CPU reference math: phase 2 of zero-context
//! compiled memory, where the model itself reads and writes the in-state file store.
//!
//! The layers come from qwenputer (divagr.com/blog/qwenputer, github.com/divagr18/qwenputer,
//! MIT, Copyright (c) Divyansh Agrawal): a compiler that writes a circuit straight into the
//! weights of stock Qwen3.5 Gated DeltaNet decoder layers. Its Filesystem checkpoint
//! (`divagr1925/Qwen3.5-4B-Qwenputer-Filesystem`, `experimental/riscv` on branch `rv-store`)
//! puts 12 such layers in front of Qwen3.5-4B's 32: hidden 4608 = Qwen's 2560 + computer dims
//! 2560..4607, GDN 16 k heads / 32 v heads / d 128. The store tables are GDN heads with one-hot
//! keys, β pinned to sigmoid(±128) and g pinned to exactly 0 (A_log = 0, dt_bias = −110), so a
//! write overwrites one 128-float row and nothing ever decays. See `docs/design/gdn-chip.md`.
//!
//! This module is the Qwen3.5 decoder layer (`linear_attention` + dense SwiGLU MLP, zero-centred
//! RMSNorms) for any GDN shape ([`Shape`]), decoded one token at a time with persistent state
//! ([`ChipState`]), its recurrence being [`reference::gdn_head_step`], the same function the
//! Qwen3-Next reference forward steps through. [`Numerics::Bf16`] rounds where HF's bf16 modules
//! round, to reproduce the author's driver; [`Numerics::F32`] is tang's plain f32 math.
//!
//! [`Driver`] is the author's teacher-forced driver (`forced` in the reproduction's
//! `repro.py`, after `RvChat`): prompt through the chip layers from zero state, the circuit's
//! initial head states injected (`hf.inject`), then per op the forced call tokens until the
//! circuit owns the position; chip tokens and circuit-printed text are fed back.

use std::collections::VecDeque;
use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
use rayon::prelude::*;
use safetensors::{Dtype, SafeTensors};

use super::reference::{dot, gdn_head_step, l2_norm, round_bf16, sigmoid, silu, softplus};

/// GDN and layer geometry of a Qwen3.5-family decoder layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub hidden: usize,
    pub k_heads: usize,
    pub v_heads: usize,
    pub d_state: usize,
    pub d_conv: usize,
    pub inter: usize,
}

impl Shape {
    /// Qwen3.5-4B widened by the Filesystem checkpoint's computer dims.
    pub const QWENPUTER_4B: Shape = Shape {
        hidden: 4608,
        k_heads: 16,
        v_heads: 32,
        d_state: 128,
        d_conv: 4,
        inter: 9216,
    };

    pub fn key_dim(&self) -> usize {
        self.k_heads * self.d_state
    }
    pub fn value_dim(&self) -> usize {
        self.v_heads * self.d_state
    }
    /// Channels of the fused q|k|v projection and the conv.
    pub fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }
    /// The k head v head `vh` reads (HF `repeat_interleave`: grouped).
    pub fn k_head(&self, vh: usize) -> usize {
        vh / (self.v_heads / self.k_heads)
    }
}

/// Where activations are rounded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Numerics {
    /// Everything f32 (tang's reference).
    F32,
    /// Round to bf16 wherever HF's bf16 Qwen3.5 modules hold a bf16 tensor: linear outputs, the
    /// conv output and its SiLU, norm outputs, the residual, the GDN core output.
    Bf16,
}

impl Numerics {
    #[inline]
    fn r(self, x: f32) -> f32 {
        match self {
            Numerics::F32 => x,
            Numerics::Bf16 => round_bf16(x),
        }
    }
    fn rv(self, x: &mut [f32]) {
        if self == Numerics::Bf16 {
            x.iter_mut().for_each(|v| *v = round_bf16(*v));
        }
    }
}

/// `y = W x` for a weight that is zero outside `W[r0..r1, c0..c1]` (the compiled layers use a
/// few hundred of their inputs): only the block is kept. Exactly the dense product wherever the
/// f32 sum is exact, which is all the circuit's arithmetic.
#[derive(Clone, Debug)]
pub struct Block {
    pub rows: usize,
    pub cols: usize,
    pub r0: usize,
    pub r1: usize,
    pub c0: usize,
    pub c1: usize,
    pub data: Vec<f32>,
}

impl Block {
    /// From a dense row-major `[rows, cols]` matrix.
    pub fn from_dense(rows: usize, cols: usize, w: &[f32]) -> Self {
        assert_eq!(w.len(), rows * cols);
        let (mut r0, mut r1, mut c0, mut c1) = (rows, 0, cols, 0);
        for r in 0..rows {
            let row = &w[r * cols..(r + 1) * cols];
            if let Some(a) = row.iter().position(|&v| v != 0.0) {
                let b = row.iter().rposition(|&v| v != 0.0).unwrap();
                r0 = r0.min(r);
                r1 = r + 1;
                c0 = c0.min(a);
                c1 = c1.max(b + 1);
            }
        }
        if r1 == 0 {
            return Self {
                rows,
                cols,
                r0: 0,
                r1: 0,
                c0: 0,
                c1: 0,
                data: vec![],
            };
        }
        let mut data = Vec::with_capacity((r1 - r0) * (c1 - c0));
        for r in r0..r1 {
            data.extend_from_slice(&w[r * cols + c0..r * cols + c1]);
        }
        Self {
            rows,
            cols,
            r0,
            r1,
            c0,
            c1,
            data,
        }
    }

    /// Multiply-adds per application.
    pub fn macs(&self) -> usize {
        self.data.len()
    }

    pub fn apply(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.cols);
        let mut y = vec![0f32; self.rows];
        let w = self.c1 - self.c0;
        if w == 0 {
            return y;
        }
        let xs = &x[self.c0..self.c1];
        y[self.r0..self.r1]
            .par_iter_mut()
            .with_min_len(64)
            .enumerate()
            .for_each(|(i, out)| *out = dot(&self.data[i * w..(i + 1) * w], xs));
        y
    }
}

/// `y = W x` over `W`'s nonzero entries only (compressed rows): the compiled layers emitted by
/// `tang-chip` are ~0.05% dense, so this is what the emitter's weights run as. Same sums as
/// [`Block`] on the circuit's (exact) arithmetic.
#[derive(Clone, Debug)]
pub struct Csr {
    pub rows: usize,
    pub cols: usize,
    /// Row `r`'s entries are `idx[ptr[r]..ptr[r + 1]]`, `val[..]`.
    pub ptr: Vec<u32>,
    pub idx: Vec<u32>,
    pub val: Vec<f32>,
}

impl Csr {
    pub fn from_sparse(s: &tang_chip::Sparse) -> Self {
        let mut ptr = vec![0u32; s.rows + 1];
        for &(r, _, _) in &s.entries {
            ptr[r as usize + 1] += 1;
        }
        for r in 0..s.rows {
            ptr[r + 1] += ptr[r];
        }
        Self {
            rows: s.rows,
            cols: s.cols,
            ptr,
            idx: s.entries.iter().map(|e| e.1).collect(),
            val: s.entries.iter().map(|e| e.2).collect(),
        }
    }

    pub fn apply(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.cols);
        (0..self.rows)
            .map(|r| {
                let (a, b) = (self.ptr[r] as usize, self.ptr[r + 1] as usize);
                self.idx[a..b]
                    .iter()
                    .zip(&self.val[a..b])
                    .map(|(&c, &w)| w * x[c as usize])
                    .sum()
            })
            .collect()
    }
}

/// A layer matrix: a dense nonzero block (the checkpoint's tensors) or compressed rows (the
/// emitter's).
#[derive(Clone, Debug)]
pub enum Mat {
    Block(Block),
    Csr(Csr),
}

impl Mat {
    pub fn apply(&self, x: &[f32]) -> Vec<f32> {
        match self {
            Mat::Block(b) => b.apply(x),
            Mat::Csr(c) => c.apply(x),
        }
    }
    /// Multiply-adds per application.
    pub fn macs(&self) -> usize {
        match self {
            Mat::Block(b) => b.macs(),
            Mat::Csr(c) => c.val.len(),
        }
    }
}

impl From<Block> for Mat {
    fn from(b: Block) -> Self {
        Mat::Block(b)
    }
}

/// One Qwen3.5 `linear_attention` decoder layer, norms with the +1 folded in.
pub struct Layer {
    pub in_norm: Vec<f32>,
    pub qkv: Mat,
    pub z: Mat,
    pub a: Mat,
    pub b: Mat,
    /// `[conv_dim][d_conv]`, tap `d_conv − 1` is the current token.
    pub conv: Vec<f32>,
    pub dt_bias: Vec<f32>,
    /// `−exp(A_log)`.
    pub a_neg: Vec<f32>,
    pub gnorm: Vec<f32>,
    pub out: Mat,
    pub post_norm: Vec<f32>,
    pub gate: Mat,
    pub up: Mat,
    pub down: Mat,
}

impl Layer {
    /// Multiply-adds of the nonzero blocks per token (the recurrence and conv not counted).
    pub fn macs(&self) -> usize {
        [
            &self.qkv, &self.z, &self.a, &self.b, &self.out, &self.gate, &self.up, &self.down,
        ]
        .iter()
        .map(|m| m.macs())
        .sum()
    }
}

impl Layer {
    /// A layer emitted by `tang-chip` (HF's zero-centred norms get their `+ 1` here).
    pub fn from_compiled(w: &tang_chip::LayerWeights) -> Self {
        let m = |s: &tang_chip::Sparse| Mat::Csr(Csr::from_sparse(s));
        let plus1 = |v: &[f32]| v.iter().map(|x| 1.0 + x).collect();
        Layer {
            in_norm: plus1(&w.input_layernorm),
            qkv: m(&w.in_proj_qkv),
            z: m(&w.in_proj_z),
            a: m(&w.in_proj_a),
            b: m(&w.in_proj_b),
            conv: w.conv1d.clone(),
            dt_bias: w.dt_bias.clone(),
            a_neg: w.a_log.iter().map(|v| -v.exp()).collect(),
            gnorm: w.norm.clone(),
            out: m(&w.out_proj),
            post_norm: plus1(&w.post_attention_layernorm),
            gate: m(&w.gate_proj),
            up: m(&w.up_proj),
            down: m(&w.down_proj),
        }
    }
}

impl Shape {
    /// The layer shape of an emitted chip stack.
    pub fn of_compiled(c: &tang_chip::Compiled) -> Shape {
        Shape {
            hidden: c.hidden,
            k_heads: c.shape.k_heads,
            v_heads: c.shape.v_heads,
            d_state: c.shape.d_state,
            d_conv: c.shape.d_conv,
            inter: c.shape.inter,
        }
    }
}

/// The recurrent state of a stack of layers: per layer the last `d_conv − 1` conv inputs and
/// every v head's `[d_k][d_v]` state (HF's `recurrent_states[0][0, vh]`).
#[derive(Clone)]
pub struct ChipState {
    pub conv: Vec<Vec<f32>>,
    pub rec: Vec<Vec<f32>>,
}

impl ChipState {
    pub fn zeros(shape: &Shape, n_layer: usize) -> Self {
        let s = shape.d_state;
        Self {
            conv: vec![vec![0f32; (shape.d_conv - 1) * shape.conv_dim()]; n_layer],
            rec: vec![vec![0f32; shape.v_heads * s * s]; n_layer],
        }
    }
    /// Value head `vh`'s state in layer `l`, `[d_k][d_v]`.
    pub fn head(&self, shape: &Shape, l: usize, vh: usize) -> &[f32] {
        let ss = shape.d_state * shape.d_state;
        &self.rec[l][vh * ss..(vh + 1) * ss]
    }
    pub fn head_mut(&mut self, shape: &Shape, l: usize, vh: usize) -> &mut [f32] {
        let ss = shape.d_state * shape.d_state;
        &mut self.rec[l][vh * ss..(vh + 1) * ss]
    }
}

/// HF's `Qwen3_5RMSNorm` with the `1 + w` folded: `x / rms(x) · w`, the mean in f32.
fn rms_norm(x: &[f32], w: &[f32], eps: f32, num: Numerics) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let r = 1.0 / (ms + eps).sqrt();
    x.iter().zip(w).map(|(v, g)| num.r(v * r * g)).collect()
}

impl Layer {
    /// One token through the layer, in place on the residual `x`.
    pub fn step(
        &self,
        sh: &Shape,
        eps: f32,
        num: Numerics,
        x: &mut [f32],
        conv: &mut [f32],
        rec: &mut [f32],
    ) {
        let (s, hk, hv, dc) = (sh.d_state, sh.k_heads, sh.v_heads, sh.d_conv);
        let (kd, ch) = (sh.key_dim(), sh.conv_dim());
        let h = rms_norm(x, &self.in_norm, eps, num);
        let mut qkv = self.qkv.apply(&h);
        num.rv(&mut qkv);
        let mut z = self.z.apply(&h);
        num.rv(&mut z);
        let mut a = self.a.apply(&h);
        num.rv(&mut a);
        let mut b = self.b.apply(&h);
        num.rv(&mut b);

        // causal conv over [conv history | this token], SiLU; then shift the history
        let mut c = vec![0f32; ch];
        for (j, out) in c.iter_mut().enumerate() {
            let w = &self.conv[j * dc..(j + 1) * dc];
            let mut acc = w[dc - 1] * qkv[j];
            for (i, wi) in w[..dc - 1].iter().enumerate() {
                acc += conv[i * ch + j] * wi;
            }
            *out = num.r(silu(num.r(acc)));
        }
        conv.copy_within(ch.., 0);
        conv[(dc - 2) * ch..].copy_from_slice(&qkv);

        // q, k L2-normed per head (f32), q / sqrt(s)
        let qscale = 1.0 / (s as f32).sqrt();
        for head in 0..2 * hk {
            l2_norm(&mut c[head * s..(head + 1) * s], 1e-6);
        }
        c[..kd].iter_mut().for_each(|v| *v *= qscale);

        let mut o = vec![0f32; sh.value_dim()];
        rec.par_chunks_mut(s * s)
            .zip(o.par_chunks_mut(s))
            .enumerate()
            .for_each(|(vh, (st, ot))| {
                let kh = sh.k_head(vh);
                let q = &c[kh * s..(kh + 1) * s];
                let k = &c[kd + kh * s..kd + (kh + 1) * s];
                let v = &c[2 * kd + vh * s..2 * kd + (vh + 1) * s];
                let g = self.a_neg[vh] * softplus(a[vh] + self.dt_bias[vh]);
                let beta = num.r(sigmoid(b[vh]));
                let mut sk = vec![0f32; s];
                gdn_head_step(st, q, k, v, g.exp(), beta, &mut sk, ot);
            });
        num.rv(&mut o);
        let _ = hv;

        // gated RMSNorm per head: w · norm(o) (rounded) · silu(z)
        for (ot, zt) in o.chunks_mut(s).zip(z.chunks(s)) {
            let ms = ot.iter().map(|v| v * v).sum::<f32>() / s as f32;
            let r = 1.0 / (ms + eps).sqrt();
            for ((v, g), zz) in ot.iter_mut().zip(&self.gnorm).zip(zt) {
                let n = num.r(num.r(*v * r) * g);
                *v = num.r(n * silu(*zz));
            }
        }
        let mut y = self.out.apply(&o);
        num.rv(&mut y);
        for (xv, yv) in x.iter_mut().zip(&y) {
            *xv = num.r(*xv + yv);
        }

        // SwiGLU MLP
        let h = rms_norm(x, &self.post_norm, eps, num);
        let mut gt = self.gate.apply(&h);
        num.rv(&mut gt);
        let mut up = self.up.apply(&h);
        num.rv(&mut up);
        let act: Vec<f32> = gt
            .iter()
            .zip(&up)
            .map(|(g, u)| num.r(num.r(silu(*g)) * u))
            .collect();
        let mut d = self.down.apply(&act);
        num.rv(&mut d);
        for (xv, dv) in x.iter_mut().zip(&d) {
            *xv = num.r(*xv + dv);
        }
    }
}

/// A stack of compiled layers plus the token embedding they read.
pub struct ChipModel {
    pub shape: Shape,
    pub eps: f32,
    pub layers: Vec<Layer>,
    embed: Embed,
}

enum Embed {
    /// A memory-mapped bf16 `[n_vocab, hidden]` table.
    Bf16 {
        map: memmap2::Mmap,
        offset: usize,
        n_vocab: usize,
    },
    Dense(Vec<f32>),
    /// Rows for the tokens a run uses (the emitter's rows), by model token id.
    Rows(std::collections::HashMap<usize, Vec<f32>>),
}

fn to_f32(t: &safetensors::tensor::TensorView<'_>) -> Result<Vec<f32>> {
    let d = t.data();
    Ok(match t.dtype() {
        Dtype::BF16 => d
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
            .collect(),
        Dtype::F32 => d
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        other => bail!("unsupported dtype {other:?}"),
    })
}

impl ChipModel {
    /// Layers `0..n_layer` (`model.layers.{i}.*`) and `model.embed_tokens.weight` of an HF
    /// Qwen3.5 safetensors file.
    pub fn open(path: &Path, shape: Shape, n_layer: usize, eps: f32) -> Result<Self> {
        let file = std::fs::File::open(path).with_context(|| format!("{}", path.display()))?;
        let map = unsafe { memmap2::Mmap::map(&file)? };
        let st = SafeTensors::deserialize(&map)?;
        let get = |n: &str| -> Result<(Vec<usize>, Vec<f32>)> {
            let t = st.tensor(n).with_context(|| n.to_string())?;
            Ok((t.shape().to_vec(), to_f32(&t)?))
        };
        let (h, ch) = (shape.hidden, shape.conv_dim());
        let layers = (0..n_layer)
            .map(|l| -> Result<Layer> {
                let p = format!("model.layers.{l}.");
                let mat = |n: &str, rows: usize, cols: usize| -> Result<Mat> {
                    let (sh, w) = get(&format!("{p}{n}"))?;
                    ensure!(sh == [rows, cols], "{p}{n}: {sh:?} != [{rows}, {cols}]");
                    Ok(Block::from_dense(rows, cols, &w).into())
                };
                let vec1 = |n: &str, len: usize, plus_one: bool| -> Result<Vec<f32>> {
                    let (_, w) = get(&format!("{p}{n}"))?;
                    ensure!(w.len() == len, "{p}{n}: len {}", w.len());
                    Ok(if plus_one {
                        w.iter().map(|v| 1.0 + v).collect()
                    } else {
                        w
                    })
                };
                let conv = vec1("linear_attn.conv1d.weight", ch * shape.d_conv, false)?;
                Ok(Layer {
                    in_norm: vec1("input_layernorm.weight", h, true)?,
                    qkv: mat("linear_attn.in_proj_qkv.weight", ch, h)?,
                    z: mat("linear_attn.in_proj_z.weight", shape.value_dim(), h)?,
                    a: mat("linear_attn.in_proj_a.weight", shape.v_heads, h)?,
                    b: mat("linear_attn.in_proj_b.weight", shape.v_heads, h)?,
                    conv,
                    dt_bias: vec1("linear_attn.dt_bias", shape.v_heads, false)?,
                    a_neg: vec1("linear_attn.A_log", shape.v_heads, false)?
                        .iter()
                        .map(|v| -v.exp())
                        .collect(),
                    gnorm: vec1("linear_attn.norm.weight", shape.d_state, false)?,
                    out: mat("linear_attn.out_proj.weight", h, shape.value_dim())?,
                    post_norm: vec1("post_attention_layernorm.weight", h, true)?,
                    gate: mat("mlp.gate_proj.weight", shape.inter, h)?,
                    up: mat("mlp.up_proj.weight", shape.inter, h)?,
                    down: mat("mlp.down_proj.weight", h, shape.inter)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let e = st.tensor("model.embed_tokens.weight")?;
        ensure!(e.dtype() == Dtype::BF16 && e.shape()[1] == h, "embedding");
        let n_vocab = e.shape()[0];
        let offset = e.data().as_ptr() as usize - map.as_ptr() as usize;
        drop(st);
        Ok(Self {
            shape,
            eps,
            layers,
            embed: Embed::Bf16 {
                map,
                offset,
                n_vocab,
            },
        })
    }

    /// From layers already in memory and a dense f32 embedding `[n_vocab][hidden]`.
    pub fn from_parts(shape: Shape, eps: f32, layers: Vec<Layer>, embed: Vec<f32>) -> Self {
        Self {
            shape,
            eps,
            layers,
            embed: Embed::Dense(embed),
        }
    }

    /// An emitted chip stack with embedding rows for the tokens `rows` names.
    pub fn from_compiled(
        c: &tang_chip::Compiled,
        eps: f32,
        rows: std::collections::HashMap<usize, Vec<f32>>,
    ) -> Self {
        Self {
            shape: Shape::of_compiled(c),
            eps,
            layers: c.layers.iter().map(Layer::from_compiled).collect(),
            embed: Embed::Rows(rows),
        }
    }

    pub fn embed(&self, tok: usize) -> Vec<f32> {
        let h = self.shape.hidden;
        match &self.embed {
            Embed::Bf16 {
                map,
                offset,
                n_vocab,
            } => {
                assert!(tok < *n_vocab, "token {tok} out of the embedding");
                let b = &map[offset + tok * h * 2..offset + (tok + 1) * h * 2];
                b.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
                    .collect()
            }
            Embed::Dense(e) => e[tok * h..(tok + 1) * h].to_vec(),
            Embed::Rows(r) => r
                .get(&tok)
                .unwrap_or_else(|| panic!("token {tok} has no row"))
                .clone(),
        }
    }

    pub fn state(&self) -> ChipState {
        ChipState::zeros(&self.shape, self.layers.len())
    }

    /// One token through every layer; the residual after the last.
    pub fn step(&self, tok: usize, st: &mut ChipState, num: Numerics) -> Vec<f32> {
        let mut x = self.embed(tok);
        for (l, layer) in self.layers.iter().enumerate() {
            layer.step(
                &self.shape,
                self.eps,
                num,
                &mut x,
                &mut st.conv[l],
                &mut st.rec[l],
            );
        }
        x
    }

    /// Multiply-adds per token of the nonzero blocks, all layers.
    pub fn macs(&self) -> usize {
        self.layers.iter().map(|l| l.macs()).sum()
    }
}

/// One output family of the circuit's code (`core.OUT_NAMES`).
#[derive(Clone, Debug)]
pub enum Family {
    /// `Q`: a printable Qwen token, payload indexes `printable`.
    Q,
    Control(u32),
    Range {
        base: u32,
        size: u32,
    },
}

/// `RvChat.code_token`: the model token the circuit predicts from the residual after the chip
/// layers, `None` where Qwen owns the position.
#[derive(Clone, Debug)]
pub struct Decoder {
    pub nfam: usize,
    pub families: Vec<Family>,
    pub printable: Vec<u32>,
    /// First chip token id in the model vocabulary.
    pub v0: u32,
    pub code_dims: Vec<usize>,
    pub id_code_dims: Vec<usize>,
}

impl Decoder {
    pub fn token(&self, hc: &[f32]) -> Result<Option<u32>> {
        let idc: Vec<f32> = self.id_code_dims.iter().map(|&d| hc[d]).collect();
        if idc.iter().any(|&v| v != 0.0) {
            return Ok(Some(
                idc.iter()
                    .enumerate()
                    .filter(|(_, v)| **v > 0.0)
                    .map(|(i, _)| 1u32 << i)
                    .sum(),
            ));
        }
        let code: Vec<f32> = self.code_dims.iter().map(|&d| hc[d]).collect();
        if code.iter().all(|&v| v == 0.0) {
            return Ok(None);
        }
        let bit = |i: usize| (code[i] > 0.0) as u32;
        let fid: u32 = (0..self.nfam).map(|i| bit(i) << i).sum();
        let p: u32 = (self.nfam..code.len())
            .map(|i| bit(i) << (i - self.nfam))
            .sum();
        let t = match self.families.get(fid as usize) {
            Some(Family::Q) => match self.printable.get(p as usize) {
                Some(&q) => return Ok(Some(q)),
                None => None,
            },
            Some(Family::Control(t)) => Some(*t),
            Some(Family::Range { base, size }) if p < *size => Some(base + p),
            _ => None,
        };
        match t {
            Some(t) => Ok(Some(self.v0 + t)),
            None => bail!("the circuit predicts no token (family {fid}, payload {p})"),
        }
    }
}

/// The author's teacher-forced driver over the chip layers.
pub struct Driver<'a> {
    pub model: &'a ChipModel,
    pub dec: &'a Decoder,
    pub num: Numerics,
    pub state: ChipState,
}

impl<'a> Driver<'a> {
    /// The prompt through the chip layers from zero state, then `inject`: for every `(layer, v
    /// head)` with `mask` set, the head's state replaced by `init` (`[layer][vh][d_k][d_v]`).
    /// The conv windows keep the prompt's last tokens, as in HF.
    pub fn start(
        model: &'a ChipModel,
        dec: &'a Decoder,
        num: Numerics,
        prompt: &[u32],
        init: &[f32],
        mask: &[u8],
    ) -> Self {
        let mut state = model.state();
        for &t in prompt {
            model.step(t as usize, &mut state, num);
        }
        let ss = model.shape.d_state * model.shape.d_state;
        let hv = model.shape.v_heads;
        for (l, rec) in state.rec.iter_mut().enumerate() {
            for vh in 0..hv {
                if mask[l * hv + vh] != 0 {
                    let i = (l * hv + vh) * ss;
                    rec[vh * ss..(vh + 1) * ss].copy_from_slice(&init[i..i + ss]);
                }
            }
        }
        Self {
            model,
            dec,
            num,
            state,
        }
    }

    /// One op: feed the forced tokens until the circuit owns a position, then its tokens until it
    /// gives the position back with the queue empty. Returns every token fed.
    pub fn op(&mut self, forced: &[u32]) -> Result<Vec<u32>> {
        let mut queue: VecDeque<u32> = forced.iter().copied().collect();
        let mut nxt = queue.pop_front().context("empty op")?;
        let mut fed = vec![];
        loop {
            ensure!(fed.len() < 10_000, "runaway op");
            let chip_in = nxt >= self.dec.v0;
            let hc = self.model.step(nxt as usize, &mut self.state, self.num);
            fed.push(nxt);
            let ct = self.dec.token(&hc)?;
            if chip_in {
                nxt = ct.context("a chip token without a successor")?;
                continue;
            }
            if let Some(t) = ct {
                nxt = t;
                continue;
            }
            match queue.pop_front() {
                Some(t) => nxt = t,
                None => break,
            }
        }
        Ok(fed)
    }
}

// --- device packing (the CUDA chip window, `chip_gpu`) ------------------------------------

/// Several CSR matrices with the same column space stacked row-wise: what one device GEMV
/// launch runs (row `r` of the stack is row `r - offset` of its source). Each row's entries
/// stay in column order, so a row's sum is [`Csr::apply`]'s, term for term.
#[derive(Clone, Debug, PartialEq)]
pub struct PackedCsr {
    pub rows: usize,
    pub ptr: Vec<u32>,
    pub idx: Vec<u32>,
    pub val: Vec<f32>,
}

impl PackedCsr {
    pub fn stack(mats: &[&Mat]) -> Result<Self> {
        let mut p = PackedCsr {
            rows: 0,
            ptr: vec![0],
            idx: vec![],
            val: vec![],
        };
        for m in mats {
            let Mat::Csr(c) = m else {
                bail!("the device chip window runs emitted (CSR) layers, not dense blocks")
            };
            let base = p.idx.len() as u32;
            p.ptr.extend(c.ptr[1..].iter().map(|v| v + base));
            p.idx.extend_from_slice(&c.idx);
            p.val.extend_from_slice(&c.val);
            p.rows += c.rows;
        }
        Ok(p)
    }
    pub fn nnz(&self) -> usize {
        self.val.len()
    }
    /// Device bytes (ptr + idx + val).
    pub fn bytes(&self) -> usize {
        4 * (self.ptr.len() + self.idx.len() + self.val.len())
    }
}

/// One chip layer as the device runs it: the four input projections stacked (`qkv | z | a |
/// b`, one launch), the output projection, gate|up stacked, down; the small vectors as they
/// are.
#[derive(Clone, Debug)]
pub struct PackedLayer {
    pub proj: PackedCsr,
    pub out: PackedCsr,
    pub gu: PackedCsr,
    pub down: PackedCsr,
    pub in_norm: Vec<f32>,
    pub post_norm: Vec<f32>,
    pub conv: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub a_neg: Vec<f32>,
    pub gnorm: Vec<f32>,
}

/// [`Decoder`] as flat arrays: families as `[kind, a, b]` (kind 0 `Q`, 1 `Control(a)`, 2
/// `Range { base: a, size: b }`).
#[derive(Clone, Debug)]
pub struct PackedDecoder {
    pub nfam: u32,
    pub fam: Vec<u32>,
    pub printable: Vec<u32>,
    pub v0: u32,
    pub code_dims: Vec<u32>,
    pub id_code_dims: Vec<u32>,
}

impl PackedDecoder {
    pub fn new(d: &Decoder) -> Self {
        let fam = d
            .families
            .iter()
            .flat_map(|f| match f {
                Family::Q => [0, 0, 0],
                Family::Control(t) => [1, *t, 0],
                Family::Range { base, size } => [2, *base, *size],
            })
            .collect();
        Self {
            nfam: d.nfam as u32,
            fam,
            printable: d.printable.clone(),
            v0: d.v0,
            code_dims: d.code_dims.iter().map(|&v| v as u32).collect(),
            id_code_dims: d.id_code_dims.iter().map(|&v| v as u32).collect(),
        }
    }

    /// What the device's `chip_pick` decodes, on the host (`Err` where it flags an error):
    /// [`Decoder::token`] over the packed arrays.
    pub fn token(&self, hc: &[f32]) -> Result<Option<u32>> {
        if self.id_code_dims.iter().any(|&d| hc[d as usize] != 0.0) {
            return Ok(Some(
                self.id_code_dims
                    .iter()
                    .enumerate()
                    .filter(|(_, &d)| hc[d as usize] > 0.0)
                    .map(|(i, _)| 1u32 << i)
                    .sum(),
            ));
        }
        if self.code_dims.iter().all(|&d| hc[d as usize] == 0.0) {
            return Ok(None);
        }
        let bit = |i: usize| (hc[self.code_dims[i] as usize] > 0.0) as u32;
        let nf = self.nfam as usize;
        let fid: u32 = (0..nf).map(|i| bit(i) << i).sum();
        let p: u32 = (nf..self.code_dims.len()).map(|i| bit(i) << (i - nf)).sum();
        let f = self.fam.get(3 * fid as usize..3 * fid as usize + 3);
        match f {
            Some(&[0, _, _]) if (p as usize) < self.printable.len() => {
                Ok(Some(self.printable[p as usize]))
            }
            Some(&[1, t, _]) => Ok(Some(self.v0 + t)),
            Some(&[2, base, size]) if p < size => Ok(Some(self.v0 + base + p)),
            _ => bail!("the circuit predicts no token (family {fid}, payload {p})"),
        }
    }
}

/// An emitted chip stack flattened for upload: layers, the embedding rows of the tokens it is
/// fed (`tokmap[token]` = row, −1 none), the decoder.
pub struct ChipPack {
    pub shape: Shape,
    pub eps: f32,
    pub layers: Vec<PackedLayer>,
    pub rows: Vec<f32>,
    pub tokmap: Vec<i32>,
    pub dec: PackedDecoder,
}

impl ChipPack {
    /// `tokens`: every token id the window may be fed (their rows come from `model.embed`).
    pub fn new(model: &ChipModel, dec: &Decoder, tokens: &[usize]) -> Result<Self> {
        let layers = model
            .layers
            .iter()
            .map(|l| -> Result<PackedLayer> {
                Ok(PackedLayer {
                    proj: PackedCsr::stack(&[&l.qkv, &l.z, &l.a, &l.b])?,
                    out: PackedCsr::stack(&[&l.out])?,
                    gu: PackedCsr::stack(&[&l.gate, &l.up])?,
                    down: PackedCsr::stack(&[&l.down])?,
                    in_norm: l.in_norm.clone(),
                    post_norm: l.post_norm.clone(),
                    conv: l.conv.clone(),
                    dt_bias: l.dt_bias.clone(),
                    a_neg: l.a_neg.clone(),
                    gnorm: l.gnorm.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let n = tokens.iter().max().map_or(0, |m| m + 1);
        let mut tokmap = vec![-1i32; n];
        let mut rows = Vec::with_capacity(tokens.len() * model.shape.hidden);
        for (i, &t) in tokens.iter().enumerate() {
            tokmap[t] = i as i32;
            rows.extend(model.embed(t));
        }
        Ok(Self {
            shape: model.shape,
            eps: model.eps,
            layers,
            rows,
            tokmap,
            dec: PackedDecoder::new(dec),
        })
    }

    /// Device bytes of the weights (CSR blocks, vectors), the embedding rows and the decoder.
    pub fn bytes(&self) -> (usize, usize) {
        let w = self
            .layers
            .iter()
            .map(|l| {
                l.proj.bytes()
                    + l.out.bytes()
                    + l.gu.bytes()
                    + l.down.bytes()
                    + 4 * (l.in_norm.len()
                        + l.post_norm.len()
                        + l.conv.len()
                        + l.dt_bias.len()
                        + l.a_neg.len()
                        + l.gnorm.len())
            })
            .sum();
        (w, 4 * (self.rows.len() + self.tokmap.len()))
    }
}

/// The driver's rule for the next token after `cur` (whose residual decoded to `ct`), as the
/// device's `chip_pick` applies it: a chip token is always followed by the circuit's; a text
/// token by the circuit's when it owns the position, else by the next queued (forced) token;
/// `None`: the op is over. Same as [`Driver::op`].
pub fn pick_next(
    v0: u32,
    cur: u32,
    ct: Option<u32>,
    queue: &mut VecDeque<u32>,
) -> Result<Option<(u32, bool)>> {
    if cur >= v0 {
        return Ok(Some((
            ct.context("a chip token without a successor")?,
            true,
        )));
    }
    if let Some(t) = ct {
        return Ok(Some((t, true)));
    }
    Ok(queue.pop_front().map(|t| (t, false)))
}

/// The retarget fixture (`chip_retarget_coder_next`, the GPU chip window's oracle): the store
/// program emitted at Coder-Next-like shapes (base 2048, 16 k / 32 v, chip tokens at 262144),
/// the put/get/list trace remapped to those ids, embedding rows with a fixed pseudo-random
/// host part.
pub struct RetargetFixture {
    pub compiled: tang_chip::Compiled,
    pub model: ChipModel,
    pub dec: Decoder,
    /// `(op, tokens fed, forced prefix length)`.
    pub ops: Vec<(String, Vec<u32>, usize)>,
    /// Token ids with embedding rows.
    pub row_tokens: Vec<usize>,
    /// The trace's store heads and states (`store_heads`, `states` of the bundled fixture).
    pub fx: serde_json::Value,
}

impl RetargetFixture {
    pub fn new() -> Result<Self> {
        let g = tang_chip::store_program()?;
        let shape = tang_chip::ChipShape {
            hidden_base: 2048,
            hidden_chip: None,
            ballast: None,
            k_heads: 16,
            v_heads: 32,
            d_state: 128,
            d_conv: 4,
            inter: 9216,
            n_layers: g.n_layers(),
            vocab_offset: 262_144,
        };
        let c = tang_chip::emit(&g, &shape)?;
        let fx: serde_json::Value =
            serde_json::from_str(&tang_chip::gunzip(tang_chip::TRACE_PUT_GET_LIST)?)?;
        let v0_old = fx["v0"].as_u64().context("v0")? as u32;
        let remap = |t: u32| {
            if t >= v0_old {
                t - v0_old + shape.vocab_offset
            } else {
                t
            }
        };
        // the host part of each row: a fixed pseudo-random pattern (must survive untouched)
        let host =
            |t: u32, i: usize| ((t as usize * 7919 + i * 104_729) % 2001) as f32 / 1000.0 - 1.0;
        let mut rows = std::collections::HashMap::new();
        let mut row_tokens = vec![];
        for (t, feats) in fx["rows"].as_object().context("rows")? {
            let t: u32 = t.parse()?;
            let f: Vec<(u32, f64)> = feats
                .as_array()
                .context("row")?
                .iter()
                .map(|p| (p[0].as_u64().unwrap() as u32, p[1].as_f64().unwrap()))
                .collect();
            let mut r = c.embedding_row(&f)?;
            for (i, v) in r[..shape.hidden_base].iter_mut().enumerate() {
                *v = round_bf16(host(t, i));
            }
            rows.insert(remap(t) as usize, r);
            row_tokens.push(remap(t) as usize);
        }
        row_tokens.sort_unstable();
        let model = ChipModel::from_compiled(&c, 1e-6, rows);
        let d = &fx["decoder"];
        let us = |v: &serde_json::Value| v.as_u64().unwrap() as u32;
        let dec = Decoder {
            nfam: d["nfam"].as_u64().context("nfam")? as usize,
            families: d["families"]
                .as_array()
                .context("families")?
                .iter()
                .map(|f| match f["kind"].as_str().unwrap() {
                    "q" => Family::Q,
                    "control" => Family::Control(us(&f["token"])),
                    _ => Family::Range {
                        base: us(&f["base"]),
                        size: us(&f["size"]),
                    },
                })
                .collect(),
            printable: d["printable"]
                .as_array()
                .context("printable")?
                .iter()
                .map(us)
                .collect(),
            v0: shape.vocab_offset,
            code_dims: c.code_dims.clone(),
            id_code_dims: c.id_code_dims.clone(),
        };
        let ops = fx["ops"]
            .as_array()
            .context("ops")?
            .iter()
            .map(|op| {
                (
                    op["op"].as_str().unwrap().to_string(),
                    op["tokens"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| remap(us(v)))
                        .collect(),
                    op["forced"].as_u64().unwrap() as usize,
                )
            })
            .collect();
        Ok(Self {
            compiled: c,
            model,
            dec,
            ops,
            row_tokens,
            fx,
        })
    }

    /// The trace's store states after op `label`: `(layer, v head, [d_k][d_v])`.
    pub fn store_states(&self, label: &str) -> Vec<(usize, usize, Vec<f32>)> {
        let sh = self.model.shape;
        let r = sh.v_heads / sh.k_heads;
        let mut out = vec![];
        for (name, h) in self.fx["store_heads"].as_object().unwrap() {
            let l = h["layer"].as_u64().unwrap() as usize;
            let kh = h["key_head"].as_u64().unwrap() as usize;
            let nvh = h["value_heads"].as_array().unwrap().len();
            for x in 0..nvh {
                let st = &self.fx["states"][label][name][x];
                let mut want = vec![0f32; sh.d_state * sh.d_state];
                for (i, rr) in st["rows"].as_array().unwrap().iter().enumerate() {
                    let rr = rr.as_u64().unwrap() as usize;
                    for (j, v) in st["vals"][i].as_array().unwrap().iter().enumerate() {
                        want[rr * sh.d_state + j] = v.as_f64().unwrap() as f32;
                    }
                }
                out.push((l, kh * r + x, want));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::TensorView;
    use std::collections::HashMap;
    use std::path::PathBuf;

    /// A layer whose GDN head 0 is a one-hot RAM (the store tables' construction): v head 0's
    /// key is one-hot on the row named by input dim 1, its value is input dims 2..4, β = 1, g =
    /// 0; zero MLP and out projection. Writing then reading back is exact in both numerics.
    #[test]
    fn chip_one_hot_ram_layer() {
        let sh = Shape {
            hidden: 8,
            k_heads: 1,
            v_heads: 2,
            d_state: 4,
            d_conv: 4,
            inter: 4,
        };
        let (h, ch, s) = (sh.hidden, sh.conv_dim(), sh.d_state);
        let mut qkv = vec![0f32; ch * h];
        // k row r (r = 0..4) reads input dim 4 + r (a one-hot address); q likewise
        for r in 0..s {
            qkv[r * h + 4 + r] = 64.0; // large: the L2 norm's eps then rounds away
            qkv[(s + r) * h + 4 + r] = 64.0;
        }
        // v head 0 value columns 0..2 read input dims 0..2 (scaled up: silu(x) ~ x for x >> 0)
        for j in 0..2 {
            qkv[(2 * s + j) * h + j] = 64.0;
        }
        let mut conv = vec![0f32; ch * sh.d_conv];
        for c in 0..ch {
            conv[c * sh.d_conv + sh.d_conv - 1] = 1.0;
        }
        let zero = |r, c| -> Mat { Block::from_dense(r, c, &vec![0f32; r * c]).into() };
        let mut b = vec![0f32; sh.v_heads * h];
        b[3] = 1.0; // β logit = 128 · dim 3 (the write enable)
        b.iter_mut().for_each(|v| *v *= 128.0);
        let layer = Layer {
            in_norm: vec![1.0; h],
            qkv: Block::from_dense(ch, h, &qkv).into(),
            z: zero(sh.value_dim(), h),
            a: zero(sh.v_heads, h),
            b: Block::from_dense(sh.v_heads, h, &b).into(),
            conv,
            dt_bias: vec![-110.0; sh.v_heads],
            a_neg: vec![-1.0; sh.v_heads],
            gnorm: vec![1.0; s],
            out: zero(h, sh.value_dim()),
            post_norm: vec![1.0; h],
            gate: zero(sh.inter, h),
            up: zero(sh.inter, h),
            down: zero(h, sh.inter),
        };
        for num in [Numerics::F32, Numerics::Bf16] {
            let mut conv_st = vec![0f32; (sh.d_conv - 1) * ch];
            let mut rec = vec![0f32; sh.v_heads * s * s];
            // write (3, 5) to row 2, then (1, 2) to row 0, then a write-disabled token on row 2;
            // a stored value is what the conv hands the head: silu(64 · normed input)
            let mut want = vec![];
            for (val, row, we) in [
                ([3.0, 5.0], 2, 1.0),
                ([1.0, 2.0], 0, 1.0),
                ([7.0, 7.0], 2, -1.0),
            ] {
                let mut x = vec![0f32; h];
                x[..2].copy_from_slice(&val);
                x[3] = we;
                x[4 + row] = 1.0;
                let n = rms_norm(&x, &vec![1.0; h], 1e-6, num);
                want.push([0, 1].map(|j| num.r(silu(num.r(64.0 * n[j])))));
                layer.step(&sh, 1e-6, num, &mut x, &mut conv_st, &mut rec);
            }
            let st = &rec[..s * s];
            let v = |r: usize, j: usize| st[r * s + j];
            assert_eq!([v(2, 0), v(2, 1)], want[0], "{num:?}");
            assert_eq!([v(0, 0), v(0, 1)], want[1], "{num:?}");
            assert!(
                st[s..2 * s].iter().all(|&e| e == 0.0) && st[3 * s..].iter().all(|&e| e == 0.0)
            );
        }
    }

    // --- milestone 1: the author's checkpoint against the reproduction's traces -------------

    /// `$TANG_CHIP_DIR` (default `/private/tmp/claude-501/qwp-ckpt`): `chip.safetensors` (layers
    /// 0..11 + embedding of the checkpoint), `prep/inject.safetensors`, `prep/decode.json`;
    /// `$TANG_CHIP_TRACES` (default `/private/tmp/claude-501/qwenputer-traces`).
    fn dirs() -> Option<(PathBuf, PathBuf)> {
        let d = PathBuf::from(
            std::env::var("TANG_CHIP_DIR").unwrap_or("/private/tmp/claude-501/qwp-ckpt".into()),
        );
        let t = PathBuf::from(
            std::env::var("TANG_CHIP_TRACES")
                .unwrap_or("/private/tmp/claude-501/qwenputer-traces".into()),
        );
        (d.join("chip.safetensors").exists() && t.join("trace_put_get_list.safetensors").exists())
            .then_some((d, t))
    }

    fn load_tensors(p: &Path) -> HashMap<String, (Vec<usize>, Vec<u8>, Dtype)> {
        let bytes = std::fs::read(p).unwrap();
        let st = SafeTensors::deserialize(&bytes).unwrap();
        st.tensors()
            .into_iter()
            .map(|(n, t): (String, TensorView)| {
                (n, (t.shape().to_vec(), t.data().to_vec(), t.dtype()))
            })
            .collect()
    }

    fn f32s(b: &[u8]) -> Vec<f32> {
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    }
    fn i64s(b: &[u8]) -> Vec<i64> {
        b.as_chunks::<8>()
            .0
            .iter()
            .map(|c| i64::from_le_bytes(*c))
            .collect()
    }
    fn i32s(b: &[u8]) -> Vec<i32> {
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes(*c))
            .collect()
    }

    fn decoder(dir: &Path) -> Decoder {
        let j: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("prep/decode.json")).unwrap())
                .unwrap();
        let us = |k: &str| -> Vec<usize> {
            j[k].as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect()
        };
        let families = j["families"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| match f["kind"].as_str().unwrap() {
                "q" => Family::Q,
                "control" => Family::Control(f["token"].as_u64().unwrap() as u32),
                _ => Family::Range {
                    base: f["base"].as_u64().unwrap() as u32,
                    size: f["size"].as_u64().unwrap() as u32,
                },
            })
            .collect();
        Decoder {
            nfam: j["nfam"].as_u64().unwrap() as usize,
            families,
            printable: us("printable").into_iter().map(|v| v as u32).collect(),
            v0: j["v0"].as_u64().unwrap() as u32,
            code_dims: us("code_dims"),
            id_code_dims: us("id_code_dims"),
        }
    }

    /// Milestone 1. The checkpoint's 12 compiled layers, run token by token in this module's
    /// reference math, driven like the author's driver over put("a.txt", 5 tokens), get, list:
    /// every fed token (forced call tokens, chip records, circuit-printed text) must equal the
    /// reproduction's, and after each op every store head's state must equal the dumped one.
    fn milestone1(num: Numerics, tol: f32, emitted: bool) {
        let Some((dir, traces)) = dirs() else {
            eprintln!("chip milestone 1: checkpoint or traces missing, skipped");
            return;
        };
        let t0 = std::time::Instant::now();
        let mut model =
            ChipModel::open(&dir.join("chip.safetensors"), Shape::QWENPUTER_4B, 12, 1e-6).unwrap();
        if emitted {
            // milestone 2: the layers tang-chip emits for these shapes replace the checkpoint's
            let g = tang_chip::store_program().unwrap();
            let c = tang_chip::emit(&g, &tang_chip::ChipShape::QWEN35_4B).unwrap();
            assert_eq!(Shape::of_compiled(&c), Shape::QWENPUTER_4B);
            model.layers = c.layers.iter().map(Layer::from_compiled).collect();
        }
        eprintln!(
            "loaded in {:?}; {:.1} M MACs/token in nonzero blocks",
            t0.elapsed(),
            model.macs() as f64 / 1e6
        );
        let dec = decoder(&dir);
        let inj = load_tensors(&dir.join("prep/inject.safetensors"));
        let init = f32s(&inj["init"].1);
        let mask = inj["inject_mask"].1.clone();
        let tr = load_tensors(&traces.join("trace_put_get_list.safetensors"));
        let ops_json: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(traces.join("trace_put_get_list.json")).unwrap(),
        )
        .unwrap();
        let layout: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(traces.join("layout.json")).unwrap())
                .unwrap();
        let prompt: Vec<u32> = i64s(&tr["prompt_ids"].1)
            .iter()
            .map(|&v| v as u32)
            .collect();

        let t0 = std::time::Instant::now();
        let mut drv = Driver::start(&model, &dec, num, &prompt, &init, &mask);
        eprintln!("prompt ({} tokens) in {:?}", prompt.len(), t0.elapsed());
        let sh = model.shape;
        let mut worst = 0f32;
        for op in ops_json.as_array().unwrap() {
            let label = op["op"].as_str().unwrap();
            let want: Vec<u32> = i64s(&tr[&format!("{label}/tokens")].1)
                .iter()
                .map(|&v| v as u32)
                .collect();
            let nf = op["forced_tokens"].as_u64().unwrap() as usize;
            let t0 = std::time::Instant::now();
            let fed = drv.op(&want[..nf]).unwrap();
            eprintln!("{label}: {} tokens in {:?}", fed.len(), t0.elapsed());
            assert_eq!(fed, want, "{num:?} {label}: fed tokens differ");
            // store heads
            let mut n_rows = 0;
            for (name, h) in layout["store_heads"].as_object().unwrap() {
                let l = h["layer"].as_u64().unwrap() as usize;
                for (x, vh) in h["value_heads"].as_array().unwrap().iter().enumerate() {
                    let vh = vh.as_u64().unwrap() as usize;
                    let ours = drv.state.head(&sh, l, vh);
                    let rows = i32s(&tr[&format!("{label}/{name}/v{x}/rows")].1);
                    let vals = f32s(&tr[&format!("{label}/{name}/v{x}/vals")].1);
                    let mut want = vec![0f32; sh.d_state * sh.d_state];
                    for (i, &r) in rows.iter().enumerate() {
                        want[r as usize * sh.d_state..(r as usize + 1) * sh.d_state]
                            .copy_from_slice(&vals[i * sh.d_state..(i + 1) * sh.d_state]);
                    }
                    n_rows += rows.len();
                    for (i, (a, b)) in ours.iter().zip(&want).enumerate() {
                        let d = (a - b).abs();
                        worst = worst.max(d);
                        assert!(
                            d <= tol * b.abs().max(1.0),
                            "{num:?} {label} {name} v{x} [{}][{}]: {a} vs {b}",
                            i / sh.d_state,
                            i % sh.d_state
                        );
                    }
                }
            }
            eprintln!("{label}: store heads match ({n_rows} nonzero rows)");
        }
        eprintln!("{num:?}: worst store-state difference {worst:e}");
    }

    #[test]
    #[ignore = "needs the 9 GB checkpoint slice; run with --ignored"]
    fn chip_milestone1_bf16() {
        milestone1(Numerics::Bf16, 0.0, false);
    }

    #[test]
    #[ignore = "needs the 9 GB checkpoint slice; run with --ignored"]
    fn chip_milestone1_f32() {
        milestone1(Numerics::F32, 1e-3, false);
    }

    /// Milestone 2: the same trace through the layers `tang-chip` emits (sparse) in place of the
    /// checkpoint's; tokens and store states bitwise as in milestone 1.
    #[test]
    #[ignore = "needs the 9 GB checkpoint slice; run with --ignored"]
    fn chip_milestone2_emitted_bf16() {
        milestone1(Numerics::Bf16, 0.0, true);
    }

    // --- milestone 2: retarget -------------------------------------------------------------

    /// Coder-Next-like shapes: hidden 2048 with the chip dims appended (auto-sized), 16 k / 32 v
    /// heads, d 128, the circuit's 12 layers, chip tokens at a different vocabulary offset. The
    /// put/get/list trace (as IR features, bundled in tang-chip) runs through the emitted layers
    /// from zero state with no prompt and no injected tables (the store needs neither): every
    /// fed token and every store head's state must equal the reproduction's, and the host's dims
    /// and the v heads the circuit does not use must stay untouched.
    #[test]
    fn chip_retarget_coder_next() {
        let t0 = std::time::Instant::now();
        let fxt = RetargetFixture::new().unwrap();
        let (c, model, dec) = (&fxt.compiled, &fxt.model, &fxt.dec);
        let shape = c.shape;
        assert_eq!(
            (c.hidden, c.layout.m),
            (3968, 62),
            "1920 chip dims: 62 ballast + tau + 1806 slots + 51 spare"
        );
        eprintln!("emitted in {:?}, {} MACs/token", t0.elapsed(), model.macs());
        let sh = model.shape;
        let mut drv = Driver {
            model,
            dec,
            num: Numerics::Bf16,
            state: model.state(),
        };
        // heads the store uses, and every head any circuit node uses
        let mut used = std::collections::HashSet::new();
        for (l, lw) in c.layers.iter().enumerate() {
            for &(row, _, _) in &lw.in_proj_qkv.entries {
                if let Some(v) = (row as usize).checked_sub(2 * sh.key_dim()) {
                    used.insert((l, v / sh.d_state));
                }
            }
        }
        // no layer writes a host dim or reads one
        for lw in &c.layers {
            let hb = shape.hidden_base as u32;
            assert!(lw.out_proj.entries.iter().all(|e| e.0 >= hb));
            assert!(lw.down_proj.entries.iter().all(|e| e.0 >= hb));
            for m in [&lw.in_proj_qkv, &lw.in_proj_z, &lw.in_proj_a, &lw.in_proj_b] {
                assert!(m.entries.iter().all(|e| e.1 >= hb));
            }
            assert!(lw
                .gate_proj
                .entries
                .iter()
                .chain(&lw.up_proj.entries)
                .all(|e| e.1 >= hb));
        }
        for (label, want, nf) in &fxt.ops {
            let fed = drv.op(&want[..*nf]).unwrap();
            assert_eq!(&fed, want, "{label}: fed tokens differ");
            // the host dims come out as embedded (checked on the op's last token; the weights
            // above show it holds for every token)
            let t = *fed.last().unwrap() as usize;
            let mut st = drv.state.clone();
            let hc = model.step(t, &mut st, Numerics::Bf16);
            assert_eq!(
                &hc[..shape.hidden_base],
                &model.embed(t)[..shape.hidden_base]
            );
            let states = fxt.store_states(label);
            for (l, vh, want) in &states {
                let ours = drv.state.head(&sh, *l, *vh);
                assert!(
                    ours.iter()
                        .zip(want)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{label} l{l} v{vh}: store state differs"
                );
            }
            for l in 0..model.layers.len() {
                for vh in 0..sh.v_heads {
                    if !used.contains(&(l, vh)) {
                        assert!(drv.state.head(&sh, l, vh).iter().all(|&v| v == 0.0));
                    }
                }
            }
            eprintln!(
                "{label}: {} tokens, store heads exact ({} heads)",
                fed.len(),
                states.len()
            );
        }
        eprintln!("retarget run in {:?}", t0.elapsed());
    }

    /// The device packing on the host: the stacked CSR rows are the layer's rows, the packed
    /// decoder decodes every residual of the trace as [`Decoder::token`] does, and
    /// [`pick_next`] replays [`Driver::op`]'s token sequence.
    #[test]
    fn chip_pack_decoder_and_pick() {
        let fxt = RetargetFixture::new().unwrap();
        let (model, dec) = (&fxt.model, &fxt.dec);
        let pack = ChipPack::new(model, dec, &fxt.row_tokens).unwrap();
        let x: Vec<f32> = (0..model.shape.hidden)
            .map(|i| ((i * 37) % 11) as f32 - 5.0)
            .collect();
        let l0 = &model.layers[0];
        let want: Vec<f32> = [&l0.qkv, &l0.z, &l0.a, &l0.b]
            .iter()
            .flat_map(|m| m.apply(&x))
            .collect();
        let p = &pack.layers[0].proj;
        let got: Vec<f32> = (0..p.rows)
            .map(|r| {
                let (a, b) = (p.ptr[r] as usize, p.ptr[r + 1] as usize);
                p.idx[a..b]
                    .iter()
                    .zip(&p.val[a..b])
                    .map(|(&c, &w)| w * x[c as usize])
                    .sum()
            })
            .collect();
        assert!(got
            .iter()
            .zip(&want)
            .all(|(a, b)| a.to_bits() == b.to_bits()));
        let mut st = model.state();
        for (label, want, nf) in &fxt.ops {
            let mut q: VecDeque<u32> = want[1..*nf].iter().copied().collect();
            let mut cur = want[0];
            let mut fed = vec![];
            loop {
                let hc = model.step(cur as usize, &mut st, Numerics::Bf16);
                fed.push(cur);
                let ct = pack.dec.token(&hc).unwrap();
                assert_eq!(ct, dec.token(&hc).unwrap(), "{label} token {}", fed.len());
                match pick_next(pack.dec.v0, cur, ct, &mut q).unwrap() {
                    Some((t, _)) => cur = t,
                    None => break,
                }
            }
            assert_eq!(&fed, want, "{label}");
        }
        let (w, e) = pack.bytes();
        eprintln!(
            "pack: {:.1} MB weights, {:.1} MB rows",
            w as f64 / 1e6,
            e as f64 / 1e6
        );
    }
}
