//! MoE on the dense path and attention sinks on CUDA (kernels in `kernels::moe_cuda`; the
//! semantics are [`crate::moe`]'s).

use cudarc::driver::{LaunchConfig, PushKernelArg};

use super::{CudaBuffer, CudaComputeDevice, CudaStorage};
use crate::kernels::{llm_cuda, moe_cuda};

/// Rows a launch of the sink kernels covers (bounds the split partials' scratch).
const SINK_ROWS: usize = 64;

fn grid(g: (usize, usize, usize), threads: u32) -> LaunchConfig {
    LaunchConfig {
        block_dim: (threads, 1, 1),
        grid_dim: (g.0 as u32, g.1 as u32, g.2 as u32),
        shared_mem_bytes: 0,
    }
}

fn per_elem(n: usize) -> LaunchConfig {
    grid((n.div_ceil(256), 1, 1), 256)
}

/// The sink-attention source for f32 or bf16 caches, kernels suffixed `_f32` / `_bf16`.
fn sinks_source(bf16: bool) -> &'static str {
    static SRC: std::sync::OnceLock<[&'static str; 2]> = std::sync::OnceLock::new();
    let build = |bf16: bool| -> &'static str {
        let head = if bf16 {
            llm_cuda::KV_BF16_CUDA
        } else {
            llm_cuda::KV_F32_CUDA
        };
        let s = moe_cuda::ATTN_SINKS_CUDA
            .replace("attn_sink_partial(", if bf16 { "attn_sink_partial_bf16(" } else { "attn_sink_partial_f32(" });
        Box::leak(format!("{head}{}{s}", llm_cuda::KV_ROW_CUDA).into_boxed_str())
    };
    SRC.get_or_init(|| [build(false), build(true)])[bf16 as usize]
}

impl CudaComputeDevice {
    /// `buf` as an f32 buffer: itself, or a widened copy of bf16 activations.
    fn moe_f32<'a>(&self, buf: &'a CudaBuffer, tmp: &'a mut Option<CudaBuffer>) -> &'a CudaBuffer {
        if buf.is_bf16() {
            tmp.insert(self.convert_bf16_to_f32(buf))
        } else {
            buf
        }
    }

    /// Causal sink attention for `q_len` rows at positions `start..` (one sequence), or one row
    /// each from several (`rows`: positions and offsets into `table`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_sinks_llm(
        &self,
        q: &CudaBuffer,
        k: &CudaBuffer,
        v: &CudaBuffer,
        table: &CudaBuffer,
        sh: u32,
        start: usize,
        q_len: usize,
        (nh, nkv, d): (usize, usize, usize),
        window: usize,
        sinks: &CudaBuffer,
        rows: Option<(&CudaBuffer, &CudaBuffer, &[usize])>,
    ) -> Option<CudaBuffer> {
        let g = nh / nkv;
        if d > 128 || nh % nkv != 0 || g > 16 || k.is_bf16() != v.is_bf16() {
            return None;
        }
        if !matches!(k.storage(), CudaStorage::F32(_) | CudaStorage::Bf16(_)) {
            return None;
        }
        let (mut tq, mut ts) = (None, None);
        let q = self.moe_f32(q, &mut tq);
        let sinks = self.moe_f32(sinks, &mut ts);
        let bf16 = k.is_bf16();
        let src = sinks_source(bf16);
        let part = self.llm_func(
            src,
            if bf16 {
                "attn_sink_partial_bf16"
            } else {
                "attn_sink_partial_f32"
            },
        );
        let comb = self.llm_func(sinks_source(false), "attn_sink_combine");
        let mut out = self.pool_alloc_uninit_f32(q_len * nh * d);
        let splits = |end: usize| {
            let lo = if window > 0 { end.saturating_sub(window) } else { 0 };
            (end - lo).div_ceil(256).clamp(1, 64)
        };
        let (nh_u, nkv_u, d_u, win) = (nh as u32, nkv as u32, d as u32, window as u32);
        for r0 in (0..q_len).step_by(SINK_ROWS) {
            let n = SINK_ROWS.min(q_len - r0);
            let max_splits = match rows {
                Some((_, _, pos)) => pos[r0..r0 + n].iter().map(|&p| splits(p + 1)).max(),
                None => Some(splits(start + r0 + n)),
            }
            .unwrap_or(1);
            let mut partial = self.pool_alloc_uninit_f32(n * nh * max_splits * (d + 2));
            let qs = crate::ComputeDevice::slice_buffer(self, q, r0 * nh * d, n * nh * d);
            let (st, ms) = ((start + r0) as u32, max_splits as u32);
            let has_rows = rows.is_some() as u32;
            let (rp, rt) = match rows {
                Some((rp, rt, _)) => (
                    crate::ComputeDevice::slice_buffer(self, rp, r0, n),
                    crate::ComputeDevice::slice_buffer(self, rt, r0, n),
                ),
                None => (crate::ComputeDevice::slice_buffer(self, table, 0, 1), crate::ComputeDevice::slice_buffer(self, table, 0, 1)),
            };
            let mut lb = self.stream.launch_builder(&part);
            lb.arg(qs.f32_data());
            match (k.storage(), v.storage()) {
                (CudaStorage::F32(ks), CudaStorage::F32(vs)) => {
                    lb.arg(ks);
                    lb.arg(vs);
                }
                (CudaStorage::Bf16(ks), CudaStorage::Bf16(vs)) => {
                    lb.arg(ks);
                    lb.arg(vs);
                }
                _ => unreachable!(),
            }
            unsafe {
                lb.arg(partial.f32_data_mut())
                    .arg(&st)
                    .arg(&nh_u)
                    .arg(&nkv_u)
                    .arg(&d_u)
                    .arg(&ms)
                    .arg(&win)
                    .arg(table.f32_data())
                    .arg(&sh)
                    .arg(rp.f32_data())
                    .arg(rt.f32_data())
                    .arg(&has_rows)
                    .launch(grid((n, nkv, max_splits), 32 * g as u32))
                    .unwrap();
            }
            let mut o = self.pool_alloc_uninit_f32(n * nh * d);
            unsafe {
                self.stream
                    .launch_builder(&comb)
                    .arg(partial.f32_data())
                    .arg(sinks.f32_data())
                    .arg(o.f32_data_mut())
                    .arg(&nh_u)
                    .arg(&d_u)
                    .arg(&ms)
                    .launch(grid((n, nh, 1), 32))
                    .unwrap();
            }
            crate::ComputeDevice::write_into(self, &mut out, r0 * nh * d, &o);
        }
        Some(out)
    }

    pub(super) fn moe_route_llm(
        &self,
        logits: &CudaBuffer,
        rows: usize,
        experts: usize,
        k: usize,
        router: crate::moe::Router,
    ) -> Option<(CudaBuffer, CudaBuffer)> {
        if experts > 256 || k > 8 || k == 0 {
            return None;
        }
        let mut tl = None;
        let logits = self.moe_f32(logits, &mut tl);
        let (mode, renorm) = match router {
            crate::moe::Router::TopKSoftmax => (0u32, 0u32),
            crate::moe::Router::SoftmaxTopK { renorm } => (1, renorm as u32),
        };
        let f = self.llm_func(moe_cuda::MOE_CUDA, "moe_route");
        let (mut ids, mut w) = (
            self.pool_alloc_uninit_f32(rows * k),
            self.pool_alloc_uninit_f32(rows * k),
        );
        let (e_u, k_u) = (experts as u32, k as u32);
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(logits.f32_data())
                .arg(ids.f32_data_mut())
                .arg(w.f32_data_mut())
                .arg(&e_u)
                .arg(&k_u)
                .arg(&mode)
                .arg(&renorm)
                .launch(grid((rows, 1, 1), 32))
                .unwrap();
        }
        Some((ids, w))
    }

    /// The MXFP4 expert GEMV; None for other formats (the host default runs them).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn moe_linear_llm(
        &self,
        x: &CudaBuffer,
        w: &CudaBuffer,
        fmt: crate::moe::ExpertFmt,
        experts: usize,
        bias: Option<&CudaBuffer>,
        ids: &CudaBuffer,
        k_top: usize,
        k: usize,
        n: usize,
        per_pick: bool,
    ) -> Option<CudaBuffer> {
        if fmt != crate::moe::ExpertFmt::Mxfp4 || !k.is_multiple_of(32) || experts > 256 {
            return None;
        }
        let picks = ids.len;
        let mut tx = None;
        let x = self.moe_f32(x, &mut tx);
        let sort = self.llm_func(moe_cuda::MOE_CUDA, "moe_sort");
        let gemv = self.llm_func(moe_cuda::MOE_CUDA, "moe_gemv_mxfp4");
        let (mut off, mut list) = (
            self.pool_alloc_uninit_f32(experts + 1),
            self.pool_alloc_uninit_f32(picks),
        );
        let (p_u, e_u) = (picks as u32, experts as u32);
        unsafe {
            self.stream
                .launch_builder(&sort)
                .arg(ids.f32_data())
                .arg(off.f32_data_mut())
                .arg(list.f32_data_mut())
                .arg(&p_u)
                .arg(&e_u)
                .launch(grid((1, 1, 1), 1024))
                .unwrap();
        }
        let mut y = self.pool_alloc_uninit_f32(picks * n);
        let (kt, k_u, n_u, pp, hb) = (
            k_top as u32,
            k as u32,
            n as u32,
            per_pick as u32,
            bias.is_some() as u32,
        );
        let b = bias.unwrap_or(x);
        unsafe {
            self.stream
                .launch_builder(&gemv)
                .arg(x.f32_data())
                .arg(w.f32_data())
                .arg(b.f32_data())
                .arg(off.f32_data())
                .arg(list.f32_data())
                .arg(y.f32_data_mut())
                .arg(&kt)
                .arg(&k_u)
                .arg(&n_u)
                .arg(&e_u)
                .arg(&pp)
                .arg(&hb)
                .launch(grid((n.div_ceil(8), experts, 1), 256))
                .unwrap();
        }
        Some(y)
    }

    pub(super) fn swiglu_clamped_llm(
        &self,
        gu: &CudaBuffer,
        rows: usize,
        ff: usize,
        alpha: f32,
        limit: f32,
    ) -> CudaBuffer {
        let mut t = None;
        let gu = self.moe_f32(gu, &mut t);
        let f = self.llm_func(moe_cuda::MOE_CUDA, "swiglu_clamped");
        let mut out = self.pool_alloc_uninit_f32(rows * ff);
        let (r, f_u) = (rows as u32, ff as u32);
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(gu.f32_data())
                .arg(out.f32_data_mut())
                .arg(&r)
                .arg(&f_u)
                .arg(&alpha)
                .arg(&limit)
                .launch(per_elem(rows * ff))
                .unwrap();
        }
        out
    }

    pub(super) fn moe_combine_llm(
        &self,
        parts: &CudaBuffer,
        w: &CudaBuffer,
        rows: usize,
        k: usize,
        h: usize,
    ) -> CudaBuffer {
        let mut t = None;
        let parts = self.moe_f32(parts, &mut t);
        let f = self.llm_func(moe_cuda::MOE_CUDA, "moe_combine");
        let mut y = self.pool_alloc_uninit_f32(rows * h);
        let (r, k_u, h_u) = (rows as u32, k as u32, h as u32);
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(parts.f32_data())
                .arg(w.f32_data())
                .arg(y.f32_data_mut())
                .arg(&r)
                .arg(&k_u)
                .arg(&h_u)
                .launch(per_elem(rows * h))
                .unwrap();
        }
        y
    }
}
