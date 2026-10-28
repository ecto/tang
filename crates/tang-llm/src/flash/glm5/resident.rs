//! The device-resident decode step (`TANG_GLM5_RESIDENT`, default on when every stage is
//! resident; `=0` forces the host-staged path).
//!
//! One decode token, slot 0, one device, every stage on ONE stream (the fused mHC's):
//!
//! - embed: the host gathers the token's row and `MhcFused::begin` uploads the 4 streams
//!   (the step's only H2D);
//! - every half block: `pre_dev` writes the normed `x` straight into the consumer's input
//!   buffer (KDA `xb`, the MLA scratch `x`, or [`Resident::x`] for the FFN); the consumer
//!   runs async on the same stream (KDA: its captured decode graph; DSA:
//!   `MlaProj::window_dev`, `kv_len`/`pos0` passed as per-call kernel args; FFN:
//!   `ffn_async`); `post_dev` hands the producer's output pointer to the next pre;
//! - head: `end_dev` (last post + stream mean) → `rs_norm` (output_norm, f64 sum like the
//!   host) → k5_q8 + k5_gemv8 (the LM head) → `rs_argmax` (first max, like `argmax`) →
//!   ONE sync and a 4-byte read: the next token.
//!
//! So per token: 1 H2D, 1 D2H, 1 host sync (was ~182 / 226 / ~226). The DSA host mirrors
//! are only resized; [`DsaCache::stale_from`] marks them and the host paths refresh them
//! from the device before any host use.

use super::*;
use tang_moe::gpu::{DevBuf, Module};

const SRC: &str = r#"
// output_norm: one block (1024 threads), x[n] -> y[n] = x * inv * w; the sum of squares
// in f64 (the host's inv_rms), mean in f64, cast to f32, + eps, 1/sqrt in f32.
extern "C" __global__ void rs_norm(const float* x, const float* w, float* y, int n, float eps) {
    __shared__ double red[1024];
    double s = 0.0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += (double)x[i] * (double)x[i];
    red[threadIdx.x] = s;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
        __syncthreads();
    }
    const float inv = 1.0f / sqrtf((float)(red[0] / (double)n) + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = x[i] * inv * w[i];
}

// argmax over v[n]: the FIRST maximum (strict >, like the host's argmax). One block.
extern "C" __global__ void rs_argmax(const float* v, int n, unsigned* out) {
    __shared__ float bv[1024];
    __shared__ int bi[1024];
    float best = v[0];
    int idx = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float x = v[i];
        if (x > best || (x == best && i < idx)) { best = x; idx = i; }
    }
    bv[threadIdx.x] = best;
    bi[threadIdx.x] = idx;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) {
            const float a = bv[threadIdx.x], b = bv[threadIdx.x + o];
            const int ia = bi[threadIdx.x], ib = bi[threadIdx.x + o];
            if (b > a || (b == a && ib < ia)) { bv[threadIdx.x] = b; bi[threadIdx.x] = ib; }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[0] = (unsigned)bi[0];
}
"#;

/// The resident step's own device state: the FFN's x / y, output_norm's weight, the token,
/// and (when there is no Q8_0 [`HeadDev`], e.g. the f32 tiny fixture) its own LM head on
/// the generic `k5_gemv`: weights, the normed row, the logits.
pub(super) struct Resident {
    m: Module,
    x: DevBuf,
    y: DevBuf,
    norm_w: DevBuf,
    tok: DevBuf,
    head: Option<(super::super::moe::gpu::DevW, super::super::moe::gpu::Kernels5, DevBuf, DevBuf)>,
}

impl Resident {
    fn new(gpu: &tang_moe::gpu::Gpu, g: &crate::gguf::Gguf, norm_w: &[f32], own_head: bool) -> Result<Self> {
        let ge = |e: tang_moe::gpu::Error| anyhow::anyhow!("glm5 resident: {e}");
        let n = norm_w.len();
        let head = if own_head {
            let info = match g.get("output.weight") {
                Some(t) => t,
                None => g.info("token_embd.weight")?,
            };
            let (rows, cols) = (info.dims[1] as usize, info.dims[0] as usize);
            ensure!(cols == n, "lm head: {cols} cols, hidden {n}");
            let dw = super::super::moe::gpu::upload(&super::super::moe::W::Gguf(g, info), 1, rows, cols)?;
            let k5 = super::super::moe::gpu::Kernels5::new(gpu)?;
            Some((dw, k5, DevBuf::zeroed_elems::<f32>(n).map_err(ge)?, DevBuf::zeroed_elems::<f32>(rows).map_err(ge)?))
        } else {
            None
        };
        Ok(Self {
            m: gpu.module(SRC).map_err(ge)?,
            x: DevBuf::zeroed_elems::<f32>(n).map_err(ge)?,
            y: DevBuf::zeroed_elems::<f32>(n).map_err(ge)?,
            norm_w: DevBuf::from_slice(norm_w).map_err(ge)?,
            tok: DevBuf::zeroed_elems::<u32>(1).map_err(ge)?,
            head,
        })
    }
}

/// `TANG_GLM5_RESIDENT=0`: force the host-staged decode.
fn resident_env() -> bool {
    !std::env::var("TANG_GLM5_RESIDENT").is_ok_and(|v| v == "0")
}

impl Glm5Engine {
    /// Can slot `slot`'s next decode token take the resident step? Every stage must be
    /// resident on one device, no shadow / dump, slot 0 of a single-slot engine, and every
    /// KDA layer's device state live.
    fn resident_ready(&self, slot: usize) -> bool {
        if !resident_env() || slot != 0 || self.n_slots != 1 || self.dump.is_some() {
            return false;
        }
        if self.kda_dbg.is_some() || self.moe_dbg.is_some() || self.mla_dbg.is_some() {
            return false;
        }
        let Some(g) = self.gpu.as_ref() else { return false };
        if g.devs.len() != 1 || !g.devs[0].1.fast || g.mhc_f.is_none() {
            return false;
        }
        let hp = &self.m.hp;
        (0..hp.n_layer).all(|l| {
            g.mhc_g[l].is_some()
                && g.has_ffn(l)
                && if hp.is_kda[l] {
                    g.kda_l[l].as_ref().is_some_and(|k| k.resident_ok()) && g.kda_dev_fresh[slot][l]
                } else {
                    g.mla_k[l].is_some() && g.mla_p[l].is_some()
                }
        })
    }

    /// Before a host path touches DSA layer `l`'s mirrors: pull the rows the resident step
    /// left on the device only.
    pub(super) fn refresh_stale_dsa(&mut self, slot: usize, l: usize) -> Result<()> {
        let Some(from) = self.dsa[slot][l].stale_from else { return Ok(()) };
        let Some(g) = self.gpu.as_ref() else { return Ok(()) };
        let Some(cu) = g.mla_k[l].as_ref() else { return Ok(()) };
        g.devs[g.dev_of[l]].0.bind().map_err(|e| anyhow::anyhow!("bind: {e}"))?;
        cu.set_seq(slot as u64);
        let c = &mut self.dsa[slot][l];
        let rows = c.valid.len();
        if rows > from {
            cu.refresh_mirrors(from, rows - from, &mut c.lat16, &mut c.idx_k, &mut c.idx_g, &mut c.valid)
                .map_err(|e| anyhow::anyhow!("glm5 dsa mirror refresh: {e}"))?;
        }
        c.stale_from = None;
        Ok(())
    }

    /// One greedy decode token on the resident step, or `None` when it doesn't apply
    /// (the caller runs the host path).
    pub(super) fn step_resident(&mut self, slot: usize, tok: u32) -> Result<Option<u32>> {
        if !self.resident_ready(slot) {
            return Ok(None);
        }
        let hp = self.m.hp.clone();
        let (n, hc) = (hp.n_embd, hp.hc);
        ensure!((tok as usize) < hp.n_vocab, "token {tok} out of range");
        let pos0 = self.slot_tokens(slot).len();
        use super::super::trace::{span, span_l};
        let tr_root = super::super::trace::window_begin(1, pos0);
        let tr = span("embed");
        let embd = self.m.g.info("token_embd.weight")?;
        let e = self.m.g.rows(embd, tok as usize, 1)?;
        let mut r = vec![0f32; hc * n];
        for c in 0..hc {
            r[c * n..(c + 1) * n].copy_from_slice(&e);
        }
        // the DSA weights (Arc'd; the window_dev path reads only scalars + i_ape)
        let mut mla_w: Vec<Option<std::sync::Arc<MlaWeights>>> = vec![None; hp.n_layer];
        for l in 0..hp.n_layer {
            if !hp.is_kda[l] {
                mla_w[l] = Some(self.mla_weights(l)?);
            }
        }
        let norm_w = if self.gpu.as_ref().is_some_and(|g| g.resident.is_none()) {
            Some(self.m.vec("output_norm.weight")?)
        } else {
            None
        };
        let g = self.gpu.as_mut().context("resident step without a GPU")?;
        g.devs[0].0.bind().map_err(|e| anyhow::anyhow!("bind: {e}"))?;
        if let Some(w) = norm_w {
            let own = g.head_dev.is_none();
            g.resident = Some(Resident::new(&g.devs[0].0, &self.m.g, &w, own)?);
        }
        let ge = |e: tang_moe::gpu::Error| anyhow::anyhow!("glm5 resident: {e}");
        let mut f = g.mhc_f.take().context("fused mHC: no instance")?;
        let out = (|| -> Result<u32> {
            f.begin(&r, 1)?;
            drop(tr);
            let res = g.resident.as_ref().unwrap();
            for l in 0..hp.n_layer {
                let _tr_l = span_l("layer", l);
                let mh = g.mhc_g[l].as_ref().context("mhc not resident")?;
                // ---- attention half
                if hp.is_kda[l] {
                    let k = g.kda_l[l].as_mut().context("kda not resident")?;
                    let tr = span("mhc_pre.attn");
                    f.pre_dev(mh, l, 0, k.x_ptr())?;
                    drop(tr);
                    let tr = span("kda");
                    k.decode_dev(f.stream())?;
                    drop(tr);
                    f.post_dev(k.y_ptr())?;
                } else {
                    let (cu, p) = (g.mla_k[l].as_ref().unwrap(), g.mla_p[l].as_ref().unwrap());
                    let w = mla_w[l].as_ref().unwrap();
                    cu.set_seq(slot as u64);
                    let tr = span("mhc_pre.attn");
                    f.pre_dev(mh, l, 0, p.x_ptr(1)?)?;
                    drop(tr);
                    let tr = span("dsa");
                    let c = &mut self.dsa[slot][l];
                    let (dl, idd) = (w.kv_lora, w.idx_dim);
                    ensure!(c.lat16.len() == pos0 * dl, "latent cache out of step");
                    let kv_len = pos0 + 1;
                    c.lat16.resize(kv_len * dl, 0);
                    c.idx_k.resize(kv_len * idd, 0.0);
                    c.idx_g.resize(kv_len * idd, 0.0);
                    c.valid.resize(kv_len, 0);
                    c.stale_from = Some(c.stale_from.map_or(pos0, |s| s.min(pos0)));
                    let y = p.window_dev(cu, w, 1, pos0, &c.lat16, &c.idx_k, &c.idx_g, &c.valid, f.stream())?;
                    drop(tr);
                    f.post_dev(y)?;
                }
                // ---- FFN half
                let tr = span("mhc_pre.ffn");
                f.pre_dev(mh, l, 1, res.x.ptr)?;
                drop(tr);
                let tr = span("ffn");
                let m = g.layers[l].as_ref().context("FFN not resident")?;
                super::super::moe::gpu::ffn_async(&g.devs[0].1, f.stream(), m, res.x.ptr, res.y.ptr, 1)?;
                drop(tr);
                f.post_dev(res.y.ptr)?;
            }
            let tr_head = span("head");
            let h = f.end_dev()?;
            let s = f.stream();
            let nv = hp.n_vocab;
            let nf = res.m.func("rs_norm").map_err(ge)?;
            let af = res.m.func("rs_argmax").map_err(ge)?;
            let (nw, ni, eps) = (res.norm_w.ptr, n as i32, hp.eps);
            let logits = unsafe {
                if let Some(hd) = g.head_dev.as_ref() {
                    let xf = hd.xf.ptr;
                    launch(nf, (1, 1, 1), (1024, 1, 1), 0, s, args![h, nw, xf, ni, eps])?;
                    launch(hd.k5.q8, ((n / 32) as u32, 1, 1), (32, 1, 1), 0, s, args![hd.xf.ptr, hd.xq8.ptr, n as u32, 1u32])?;
                    launch(
                        hd.k5.gemv8,
                        (nv.div_ceil(K5_WARPS) as u32, 1, 1),
                        (128, 1, 1),
                        (K5_WARPS * ((hd.dw.w.row_bytes as usize).div_ceil(16) + 1) * 16) as u32,
                        s,
                        args![hd.dw.w, 0u32, hd.xq8.ptr, hd.xq8.ptr + n as u64, hd.out.ptr,
                            nv as u32, 1u32, nv as u32, n as u32],
                    )?;
                    hd.out.ptr
                } else {
                    let (dw, k5, xb, ob) = res.head.as_ref().context("resident: no LM head")?;
                    let xf = xb.ptr;
                    launch(nf, (1, 1, 1), (1024, 1, 1), 0, s, args![h, nw, xf, ni, eps])?;
                    launch(
                        k5.gemv,
                        (nv.div_ceil(128) as u32, 1, 1),
                        (128, 1, 1),
                        0,
                        s,
                        args![dw.w, 0u32, xb.ptr, ob.ptr, nv as u32, 1u32, nv as u32, n as u32],
                    )?;
                    ob.ptr
                }
            };
            let (nvi, tp) = (nv as i32, res.tok.ptr);
            unsafe { launch(af, (1, 1, 1), (1024, 1, 1), 0, s, args![logits, nvi, tp])? };
            let tr = span("sync+read");
            s.sync().map_err(ge)?;
            let t = res.tok.read::<u32>(0, 1).map_err(ge)?[0];
            drop(tr);
            drop(tr_head);
            Ok(t)
        })();
        g.mhc_f = Some(f);
        let next = out?;
        self.slot_tokens_mut(slot).push(tok);
        super::super::trace::window_end(tr_root, 1, pos0);
        Ok(Some(next))
    }
}
