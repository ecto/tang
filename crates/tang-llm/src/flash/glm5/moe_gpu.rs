//! GPU launcher for the naive GLM-5 MoE kernels ([`super::SRC`]): weights uploaded in their
//! storage format (GGUF bytes as-is, or f32), f32 activations. Correctness first; see "Later"
//! in the module docs for the fast path.
//!
//! Pod parity test (1×H200 or any CUDA card):
//! `TANG_REQUIRE_CUDA=1 cargo test --release -p tang-llm --features cuda --lib flash::glm5::moe`

use super::{MoeParams, W};
use crate::gguf::{GgmlType, TensorInfo};
use anyhow::{bail, ensure, Context, Result};
use tang_moe::gpu::{self, DevBuf, Gpu, Module, Stream};

type Func = cudarc::driver::sys::CUfunction;

/// `struct G5W` in moe_cuda.cuh (same field order and widths: 8 + 8 + 4 + 4 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct G5W {
    pub base: u64,
    pub estride: u64,
    pub row_bytes: u32,
    pub ty: u32,
}

pub struct Kernels {
    _m: Module,
    logits: Func,
    select: Func,
    gu: Func,
    down: Func,
    combine: Func,
}

impl Kernels {
    pub fn new(g: &Gpu) -> Result<Self> {
        let m = g.module(super::SRC)?;
        Ok(Kernels {
            logits: m.func("g5_router_logits")?,
            select: m.func("g5_router_select")?,
            gu: m.func("g5_expert_gu")?,
            down: m.func("g5_expert_down")?,
            combine: m.func("g5_combine")?,
            _m: m,
        })
    }
}

/// One weight on the device, with its kernel view.
pub struct DevW {
    pub buf: DevBuf,
    pub w: G5W,
}

fn ggml_id(t: &TensorInfo) -> Result<u32> {
    Ok(match t.ty {
        GgmlType::F32 => 0,
        GgmlType::F16 => 1,
        GgmlType::Q5_0 => 6,
        GgmlType::Q8_0 => 8,
        GgmlType::Q4K => 12,
        GgmlType::Q5K => 13,
        GgmlType::Q6K => 14,
        other => bail!(
            "{}: {} has no decoder in moe_cuda.cuh",
            t.name,
            other.name()
        ),
    })
}

/// Upload `w`, which holds `n_mat` matrices of `rows × cols`.
pub fn upload(w: &W, n_mat: usize, rows: usize, cols: usize) -> Result<DevW> {
    match *w {
        W::F32(v) => {
            ensure!(
                v.len() == n_mat * rows * cols,
                "f32 weight: {} values, want {n_mat}x{rows}x{cols}",
                v.len()
            );
            let buf = DevBuf::from_slice(v)?;
            let w = G5W {
                base: buf.ptr,
                estride: (rows * cols * 4) as u64,
                row_bytes: (cols * 4) as u32,
                ty: 0,
            };
            Ok(DevW { buf, w })
        }
        W::Gguf(g, t) => {
            ensure!(
                t.dims[0] as usize == cols && t.dims.get(1).copied().unwrap_or(1) as usize == rows,
                "{}: dims {:?}, want [{cols}, {rows}, ..]",
                t.name,
                t.dims
            );
            let rb = t.row_bytes()?;
            let buf = DevBuf::from_slice(g.bytes(t))?;
            let w = G5W {
                base: buf.ptr,
                estride: (rows * rb) as u64,
                row_bytes: rb as u32,
                ty: ggml_id(t)?,
            };
            Ok(DevW { buf, w })
        }
    }
}

/// A layer's FFN on the device: geometry (weight handles cleared) plus uploaded weights.
pub struct DevMoe {
    pub p: MoeParams<'static>,
    pub router: Option<DevW>,
    pub bias: Option<DevBuf>,
    pub exps: Option<(DevW, DevW, DevW)>,
    pub sh: Option<(DevW, DevW, DevW)>,
    pub dense: Option<(DevW, DevW, DevW)>,
}

type Mlp = (DevW, DevW, DevW);

fn up3(
    g: Option<W>,
    u: Option<W>,
    d: Option<W>,
    n: usize,
    ff: usize,
    hid: usize,
) -> Result<Option<Mlp>> {
    let (Some(g), Some(u), Some(d)) = (g, u, d) else {
        return Ok(None);
    };
    Ok(Some((
        upload(&g, n, ff, hid)?,
        upload(&u, n, ff, hid)?,
        upload(&d, n, hid, ff)?,
    )))
}

impl DevMoe {
    pub fn new(p: &MoeParams) -> Result<Self> {
        let (hid, e) = (p.hidden, p.n_expert);
        let bias = match p.bias {
            Some(b) => Some(DevBuf::from_slice(&b.vec(e)?)?),
            None => None,
        };
        let geom = MoeParams {
            hidden: p.hidden,
            n_expert: p.n_expert,
            top_k: p.top_k,
            n_group: p.n_group,
            topk_group: p.topk_group,
            ff_exp: p.ff_exp,
            ff_shexp: p.ff_shexp,
            ff_dense: p.ff_dense,
            scale: p.scale,
            norm_topk_prob: p.norm_topk_prob,
            limit_exp: p.limit_exp,
            limit_shexp: p.limit_shexp,
            ..Default::default()
        };
        Ok(DevMoe {
            p: geom,
            router: p.router.map(|w| upload(&w, 1, e, hid)).transpose()?,
            bias,
            exps: up3(p.gate_exps, p.up_exps, p.down_exps, e, p.ff_exp, hid)?,
            sh: up3(p.sh_gate, p.sh_up, p.sh_down, 1, p.ff_shexp, hid)?,
            dense: up3(p.dense_gate, p.dense_up, p.dense_down, 1, p.ff_dense, hid)?,
        })
    }
}

const BLOCK: (u32, u32, u32) = (128, 1, 1);

fn grid(n: usize) -> (u32, u32, u32) {
    (n.div_ceil(128) as u32, 1, 1)
}

/// `out = MLP(x)` per token on matrix 0 of `m` (shared expert or dense FFN).
#[allow(clippy::too_many_arguments)]
fn mlp(
    k: &Kernels,
    s: &Stream,
    m: &Mlp,
    ff: usize,
    hid: usize,
    lim: f32,
    x: u64,
    out: u64,
    t_len: usize,
) -> Result<()> {
    let h = DevBuf::alloc(t_len * ff * 4)?;
    let (null, one, zero) = (0u64, 1i32, 0i32);
    let (t, ffi, hidi) = (t_len as i32, ff as i32, hid as i32);
    let (gw, uw, dw, hp) = (m.0.w, m.1.w, m.2.w, h.ptr);
    unsafe {
        // TANG_GLM5_TRACE: matrix 0's weight bytes (estride = one matrix)
        gpu::trace::bytes(gw.estride + uw.estride);
        gpu::launch(
            k.gu,
            grid(t_len * ff),
            BLOCK,
            0,
            s,
            tang_moe::args![gw, uw, null, one, zero, x, hp, t, ffi, hidi, lim],
        )?;
        gpu::trace::bytes(dw.estride);
        gpu::launch(
            k.down,
            grid(t_len * hid),
            BLOCK,
            0,
            s,
            tang_moe::args![dw, null, zero, hp, out, t, ffi, hidi],
        )?;
    }
    s.sync()?;
    Ok(())
}

/// `y = ffn(x)` for `t_len` tokens (dense or MoE); `x`, `y` are device `[t_len][hidden]` f32.
/// Synchronous (returns after the stream drains).
pub fn ffn(k: &Kernels, s: &Stream, m: &DevMoe, x: u64, y: u64, t_len: usize) -> Result<()> {
    let p = &m.p;
    let hid = p.hidden;
    use super::super::trace::span;
    if let Some(d) = &m.dense {
        let _tr = span("dense");
        return mlp(k, s, d, p.ff_dense, hid, p.limit_shexp, x, y, t_len);
    }
    let router = m.router.as_ref().context("no router")?;
    let (gw, uw, dw) = m
        .exps
        .as_ref()
        .map(|(g, u, d)| (g.w, u.w, d.w))
        .context("no routed experts")?;
    let np = t_len * p.top_k;
    let sc = DevBuf::alloc(t_len * p.n_expert * 4)?;
    let ids = DevBuf::alloc(np * 4)?;
    let wts = DevBuf::alloc(np * 4)?;
    let h = DevBuf::alloc(np * p.ff_exp * 4)?;
    let yp = DevBuf::alloc(np * hid * 4)?;
    let ysh = match &m.sh {
        Some(sh) => {
            let _tr = span("moe.shared");
            let b = DevBuf::alloc(t_len * hid * 4)?;
            mlp(k, s, sh, p.ff_shexp, hid, p.limit_shexp, x, b.ptr, t_len)?;
            Some(b)
        }
        None => None,
    };
    let (t, ne, kk, hidi, ff, npi) = (
        t_len as i32,
        p.n_expert as i32,
        p.top_k as i32,
        hid as i32,
        p.ff_exp as i32,
        np as i32,
    );
    let (ng, tg, norm, scale) = (
        p.n_group as i32,
        p.topk_group as i32,
        p.norm_topk_prob as i32,
        p.scale,
    );
    let bias = m.bias.as_ref().map_or(0u64, |b| b.ptr);
    let yshp = ysh.as_ref().map_or(0u64, |b| b.ptr);
    let (rw, scp, idp, wtp, hp, ypp, lim) = (
        router.w,
        sc.ptr,
        ids.ptr,
        wts.ptr,
        h.ptr,
        yp.ptr,
        p.limit_exp,
    );
    // the experts a window can touch: every routed pair, at most every expert (an upper
    // bound on the bytes for prefill; exact for decode's distinct top-k)
    let n_touch = np.min(p.n_expert) as u64;
    unsafe {
        let tr = span("moe.router");
        gpu::trace::bytes(rw.estride);
        gpu::launch(
            k.logits,
            grid(t_len * p.n_expert),
            BLOCK,
            0,
            s,
            tang_moe::args![rw, x, scp, t, ne, hidi],
        )?;
        gpu::launch(
            k.select,
            grid(t_len),
            BLOCK,
            0,
            s,
            tang_moe::args![scp, bias, idp, wtp, t, ne, kk, ng, tg, norm, scale],
        )?;
        drop(tr);
        let tr = span("moe.experts");
        gpu::trace::bytes(n_touch * (gw.estride + uw.estride));
        gpu::launch(
            k.gu,
            grid(np * p.ff_exp),
            BLOCK,
            0,
            s,
            tang_moe::args![gw, uw, idp, kk, ne, x, hp, npi, ff, hidi, lim],
        )?;
        gpu::trace::bytes(n_touch * dw.estride);
        gpu::launch(
            k.down,
            grid(np * hid),
            BLOCK,
            0,
            s,
            tang_moe::args![dw, idp, ne, hp, ypp, npi, ff, hidi],
        )?;
        drop(tr);
        let _tr = span("moe.combine");
        gpu::launch(
            k.combine,
            grid(t_len * hid),
            BLOCK,
            0,
            s,
            tang_moe::args![ypp, wtp, yshp, y, t, kk, hidi],
        )?;
    }
    let _tr = span("sync");
    s.sync()?;
    Ok(())
}

/// Convenience for tests and W1's first wiring: host `x` in, host `y` out.
pub fn ffn_host(k: &Kernels, m: &DevMoe, x: &[f32], t_len: usize) -> Result<Vec<f32>> {
    ensure!(x.len() == t_len * m.p.hidden, "ffn_host: x size");
    let s = Stream::blocking()?;
    let xd = DevBuf::from_slice(x)?;
    let yd = DevBuf::zeroed(x.len() * 4)?;
    ffn(k, &s, m, xd.ptr, yd.ptr, t_len)?;
    Ok(yd.read::<f32>(0, x.len())?)
}

/// The KDA projection kernels ([`super::K5_SRC`] = the MoE source + `kda_k5.cuh`, so
/// `g5_dot`/`g5_deq` are in scope): `gemv` (one thread per output), `gate` (the forget
/// gate's finish: dt_bias + the safe-gate/softplus form), `beta` (the β GEMV + sigmoid),
/// `norm` (the gated output norm). Compiled once per arch; one instance serves every
/// KDA layer.
pub struct Kernels5 {
    _m: Module,
    pub gemv: Func,
    pub gemv8: Func,
    pub q8: Func,
    pub gate: Func,
    pub beta: Func,
    pub norm: Func,
}

impl Kernels5 {
    pub fn new(g: &Gpu) -> Result<Self> {
        let m = g.module(super::K5_SRC)?;
        Ok(Kernels5 {
            gemv: m.func("k5_gemv")?,
            gemv8: m.func("k5_gemv8")?,
            q8: m.func("k5_q8")?,
            gate: m.func("k5_gate_finish")?,
            beta: m.func("k5_beta")?,
            norm: m.func("k5_gated_norm")?,
            _m: m,
        })
    }
}
