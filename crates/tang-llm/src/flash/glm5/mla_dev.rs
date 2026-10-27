//! Device-resident MLA projections (`TANG_GLM5_MLA_RESIDENT=1`, default off): q_a (+ its
//! RMSNorm), q_b, o_proj and the indexer's `wq_b` / `weights_proj` GEMVs on the device, on
//! the fast MoE path's `g5f_gemv` (weights in their GGUF type, two-level int8
//! activations). The naive DSA path ran all five on the host (an f32 loop over the
//! dequantised weights) — ~93% of GPU-side decode time on the H200 profile.
//!
//! One window ([`MlaProj::window`]): `x` up once; q_a → norm → q_b and the indexer's iq/hw
//! stay on the device and feed [`MlaCuda::idx_run_dev`] / [`MlaCuda::attend_dev`]
//! directly; o_proj reads the attention output in place; `y` comes down once. The latent and
//! indexer caches still append from the host `x` (their launchers stage it; small).

use super::mla::cuda::MlaCuda;
use super::mla::MlaWeights;
use super::moe::gpu::{upload, Act, DevW, Kernels5};
use anyhow::{ensure, Context, Result};
use std::cell::RefCell;
use tang_moe::gpu::{DevBuf, Gpu, Stream};

/// Per-window device scratch (grows by token count).
struct Scratch {
    cap: usize,
    x: DevBuf,
    xq: DevBuf,
    xq2: DevBuf,
    xd: DevBuf,
    qr: DevBuf,
    qrq: DevBuf,
    qrq2: DevBuf,
    qrd: DevBuf,
    q: DevBuf,
    iq: DevBuf,
    hw: DevBuf,
    oq: DevBuf,
    oq2: DevBuf,
    od: DevBuf,
    y: DevBuf,
}

/// One DSA layer's resident projections.
pub struct MlaProj {
    k: Kernels5,
    q_a: DevW,
    q_b: DevW,
    o: DevW,
    iq_b: DevW,
    ip: DevW,
    qan: DevBuf,
    hid: usize,
    ql: usize,
    nhdk: usize,
    nhdv: usize,
    iqr: usize,
    ih: usize,
    eps: f32,
    sc: RefCell<Option<Scratch>>,
}

/// What [`MlaProj::window`] hands back besides `y`: the selection it used, and (debug only)
/// the device's q_resid / q.
pub struct ProjOut {
    pub y: Vec<f32>,
    pub sel: Option<Vec<i32>>,
    pub q_resid: Option<Vec<f32>>,
    pub q: Option<Vec<f32>>,
}

fn dev_w(g: &crate::gguf::Gguf, name: &str) -> Result<DevW> {
    let t = g.info(name).with_context(|| format!("mla_dev {name}"))?;
    upload(&super::moe::W::Gguf(g, t), 1, t.n_rows(), t.row_len())
}

impl MlaProj {
    /// Upload layer `l`'s projections from the GGUF (bound to the layer's device).
    pub fn new(g: &Gpu, gguf: &crate::gguf::Gguf, l: usize, w: &MlaWeights) -> Result<Self> {
        let t = |s: &str| super::t(l, s);
        let p = MlaProj {
            k: Kernels5::new(g)?,
            q_a: dev_w(gguf, &t("attn_q_a.weight"))?,
            q_b: dev_w(gguf, &t("attn_q_b.weight"))?,
            o: dev_w(gguf, &t("attn_output.weight"))?,
            iq_b: dev_w(gguf, &t("indexer.attn_q_b.weight"))?,
            ip: dev_w(gguf, &t("indexer.proj.weight"))?,
            qan: DevBuf::from_slice(&w.q_a_norm).map_err(|e| anyhow::anyhow!("q_a_norm: {e}"))?,
            hid: w.hidden,
            ql: w.q_lora,
            nhdk: w.n_head * w.qk_nope,
            nhdv: w.n_head * w.v_head,
            iqr: w.idx_heads * w.idx_dim,
            ih: w.idx_heads,
            eps: w.eps,
            sc: RefCell::new(None),
        };
        for (n, dw, cols) in [
            ("q_a", &p.q_a, p.hid),
            ("q_b", &p.q_b, p.ql),
            ("o", &p.o, p.nhdv),
            ("iq_b", &p.iq_b, p.ql),
            ("ip", &p.ip, p.hid),
        ] {
            // g5f_gemv's activation stride must be a multiple of 32; every row length here
            // is one (real model and the tiny fixtures alike)
            let _ = dw;
            ensure!(cols % 32 == 0, "mla_dev {n}: cols {cols} not a multiple of 32");
        }
        Ok(p)
    }

    fn scratch(&self, t: usize) -> Result<std::cell::RefMut<'_, Scratch>> {
        let mut s = self.sc.borrow_mut();
        if s.as_ref().is_none_or(|s| s.cap < t) {
            let cap = t.max(8);
            let f = |n: usize| DevBuf::zeroed(n * 4).map_err(|e| anyhow::anyhow!("mla_dev scratch: {e}"));
            let b = |n: usize| DevBuf::zeroed(n).map_err(|e| anyhow::anyhow!("mla_dev scratch: {e}"));
            let (h, ql, ov) = (self.hid, self.ql, self.nhdv);
            *s = Some(Scratch {
                cap,
                x: f(cap * h)?,
                xq: b(cap * h)?,
                xq2: b(cap * h)?,
                xd: f(cap * h / 32)?,
                qr: f(cap * ql)?,
                qrq: b(cap * ql)?,
                qrq2: b(cap * ql)?,
                qrd: f(cap * ql / 32)?,
                q: f(cap * self.nhdk)?,
                iq: f(cap * self.iqr)?,
                hw: f(cap * self.ih)?,
                oq: b(cap * ov)?,
                oq2: b(cap * ov)?,
                od: f(cap * ov / 32)?,
                y: f(cap * self.hid)?,
            });
        }
        Ok(std::cell::RefMut::map(s, |s| s.as_mut().unwrap()))
    }

    /// int8 codes of `x [t][cols]` when any of `ws` is quantised (F32/F16 rows read `xf`).
    #[allow(clippy::too_many_arguments)]
    fn act(&self, s: &Stream, ws: &[&DevW], xf: u64, xq: u64, xq2: u64, xd: u64, cols: usize, t: usize) -> Result<Act> {
        let quant = ws.iter().any(|w| w.w.ty > 1);
        if quant {
            self.k.fast.q8(s, xf, xq, xq2, xd, cols, t)?;
        }
        Ok(Act {
            xf,
            xq: if quant { xq } else { 0 },
            xq2: if quant { xq2 } else { 0 },
            xd: if quant { xd } else { 0 },
            xs: cols,
        })
    }

    /// One DSA window on the device: the caches append from `x` (host), the indexer runs
    /// when sparse, attention + o_proj; returns `y [t][hidden]`.
    #[allow(clippy::too_many_arguments)]
    pub fn window(
        &self,
        cu: &MlaCuda,
        w: &MlaWeights,
        x: &[f32],
        t: usize,
        pos0: usize,
        c_lat: &mut [u16],
        c_k: &mut [f32],
        c_g: &mut [f32],
        c_valid: &mut [u8],
        pool_keys_out: Option<&mut Vec<f32>>,
        debug: bool,
        s: &Stream,
    ) -> Result<ProjOut> {
        let gerr = |e: tang_moe::gpu::Error| anyhow::anyhow!("glm5 mla_dev: {e}");
        let kv_len = pos0 + t;
        let sc = self.scratch(t)?;
        let (h, ql) = (self.hid, self.ql);
        sc.x.write(0, x).map_err(gerr)?;
        let ax = self.act(s, &[&self.q_a, &self.ip], sc.x.ptr, sc.xq.ptr, sc.xq2.ptr, sc.xd.ptr, h, t)?;
        // q_resid = rms_norm(q_a x) * q_a_norm
        self.k.fast.gemv(s, self.q_a.w, 0, ax, sc.qr.ptr, ql, t, ql, h)?;
        cu.rms_norm_dev(sc.qr.ptr, self.qan.ptr, ql, t, self.eps, s).map_err(gerr)?;
        let aq = self.act(s, &[&self.q_b, &self.iq_b], sc.qr.ptr, sc.qrq.ptr, sc.qrq2.ptr, sc.qrd.ptr, ql, t)?;
        self.k.fast.gemv(s, self.q_b.w, 0, aq, sc.q.ptr, self.nhdk, t, self.nhdk, ql)?;

        // caches (host x staged by the launchers; also refresh the engine's mirrors)
        cu.kv_append(&w.kv_a, &w.kv_a_norm, x, t, pos0, c_lat, s).map_err(gerr)?;
        cu.idx_append(&w.i_wk, &w.i_k_norm_w, &w.i_k_norm_b, &w.i_gate, x, t, pos0, c_k, c_g, c_valid, s)
            .map_err(gerr)?;
        let sel = if super::mla::indexer_is_dense(kv_len, w.topk) {
            None
        } else {
            let n_pool = kv_len / w.kpool;
            let mut pool_keys = vec![0f32; n_pool * w.idx_dim];
            cu.idx_pool(c_k, c_g, &w.i_ape, kv_len, 0, n_pool, &mut pool_keys, s).map_err(gerr)?;
            self.k.fast.gemv(s, self.iq_b.w, 0, aq, sc.iq.ptr, self.iqr, t, self.iqr, ql)?;
            self.k.fast.gemv(s, self.ip.w, 0, ax, sc.hw.ptr, self.ih, t, self.ih, h)?;
            let mut sel = vec![-1i32; t * w.select_width()];
            cu.idx_run_dev(sc.iq.ptr, sc.hw.ptr, kv_len, 0, n_pool, t, pos0, &mut sel, s).map_err(gerr)?;
            if let Some(o) = pool_keys_out {
                *o = pool_keys;
            }
            Some(sel)
        };
        let ov = cu.attend_dev(sc.q.ptr, kv_len, t, pos0, sel.as_deref(), s).map_err(gerr)?;
        // o_proj straight off the attention output ([t][nh*dv])
        let ao = self.act(s, &[&self.o], ov, sc.oq.ptr, sc.oq2.ptr, sc.od.ptr, self.nhdv, t)?;
        self.k.fast.gemv(s, self.o.w, 0, ao, sc.y.ptr, h, t, h, self.nhdv)?;
        s.sync().map_err(gerr)?;
        let y = sc.y.read::<f32>(0, t * h).map_err(gerr)?;
        let (q_resid, q) = if debug {
            (
                Some(sc.qr.read::<f32>(0, t * ql).map_err(gerr)?),
                Some(sc.q.read::<f32>(0, t * self.nhdk).map_err(gerr)?),
            )
        } else {
            (None, None)
        };
        Ok(ProjOut { y, sel, q_resid, q })
    }
}
