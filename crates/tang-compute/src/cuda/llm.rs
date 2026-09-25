//! LLM inference on CUDA: the `ComputeDevice` methods tang-llm's forward pass uses, over the
//! kernels in `kernels::llm_cuda`. Same semantics as the Metal backend: activations f32,
//! weights f32, bf16 or MLX 4-bit; attention with sliding windows and bidirectional blocks for
//! any head dim up to 256.

use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{Gemm, GemmConfig};
use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};

use super::{CudaBuffer, CudaComputeDevice, CudaStorage, Q4Weight};
use crate::kernels::llm_cuda;

/// One thread per element, 256 per block.
fn per_elem(n: usize) -> LaunchConfig {
    LaunchConfig {
        block_dim: (256, 1, 1),
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn blocks(grid: (usize, usize, usize), threads: u32) -> LaunchConfig {
    LaunchConfig {
        block_dim: (threads, 1, 1),
        grid_dim: (grid.0 as u32, grid.1 as u32, grid.2 as u32),
        shared_mem_bytes: 0,
    }
}

/// Dequantized weight rows staged per prefill GEMM (floats): bounds scratch memory.
const DEQUANT_CHUNK: usize = 32 << 20;

impl CudaComputeDevice {
    /// Kernel `name` from `source` (one of `llm_cuda`'s), compiled once.
    fn llm_func(&self, source: &str, name: &'static str) -> CudaFunction {
        if let Some(f) = self.llm_funcs.borrow().get(name) {
            return f.clone();
        }
        let (_module, f) = self.get_func(source, name);
        self.llm_funcs.borrow_mut().insert(name, f.clone());
        f
    }

    /// `buf` as f32 on device: itself, or a widened copy of bf16 activations.
    fn as_f32<'a>(&self, buf: &'a CudaBuffer, tmp: &'a mut Option<CudaBuffer>) -> &'a CudaBuffer {
        if buf.is_bf16() {
            tmp.insert(self.convert_bf16_to_f32(buf))
        } else {
            buf
        }
    }

    /// Back to the device's activation precision.
    fn finish(&self, out: CudaBuffer) -> CudaBuffer {
        if self.mixed_precision {
            self.convert_f32_to_bf16(&out)
        } else {
            out
        }
    }

    pub(super) fn upload_q4_impl(
        &self,
        packed: &[u32],
        scales: &[u16],
        biases: &[u16],
        group: usize,
    ) -> CudaBuffer {
        assert!(group % 8 == 0, "4-bit groups must be a multiple of 8");
        let q = Q4Weight {
            packed: self.stream.memcpy_stod(packed).unwrap(),
            scales: self.stream.memcpy_stod(scales).unwrap(),
            biases: self.stream.memcpy_stod(biases).unwrap(),
            group,
        };
        Self::make_buf_unpooled(CudaStorage::Q4(Box::new(q)), packed.len() * 8)
    }

    pub(super) fn upload_bf16_impl(&self, bits: &[u16]) -> CudaBuffer {
        let slice = self.stream.memcpy_stod(bits).unwrap();
        Self::make_buf_unpooled(CudaStorage::Bf16(slice), bits.len())
    }

    /// `embedding` from bf16 or 4-bit weights into f32 rows. None for other weights.
    pub(super) fn embedding_llm(
        &self,
        weight: &CudaBuffer,
        ids: &CudaBuffer,
        seq_len: usize,
        dim: usize,
    ) -> Option<CudaBuffer> {
        let total = seq_len * dim;
        let (s, d) = (seq_len as u32, dim as u32);
        let mut out = self.pool_alloc_uninit_f32(total);
        match weight.storage() {
            CudaStorage::Q4(q) => {
                let f = self.llm_func(llm_cuda::EMBED_CUDA, "embedding_q4");
                let group = q.group as u32;
                unsafe {
                    self.stream
                        .launch_builder(&f)
                        .arg(&q.packed)
                        .arg(&q.scales)
                        .arg(&q.biases)
                        .arg(ids.f32_data())
                        .arg(out.f32_data_mut())
                        .arg(&s)
                        .arg(&d)
                        .arg(&group)
                        .launch(per_elem(total))
                        .unwrap();
                }
            }
            // bf16 activations (mixed precision) keep the training path.
            CudaStorage::Bf16(w) if !self.mixed_precision => {
                let f = self.llm_func(llm_cuda::EMBED_CUDA, "embedding_bf16");
                unsafe {
                    self.stream
                        .launch_builder(&f)
                        .arg(w)
                        .arg(ids.f32_data())
                        .arg(out.f32_data_mut())
                        .arg(&s)
                        .arg(&d)
                        .launch(per_elem(total))
                        .unwrap();
                }
            }
            _ => return None,
        }
        Some(self.finish(out))
    }

    /// `linear` over bf16 or 4-bit weights with f32 activations. None for other combinations.
    /// Decode (`m <= 8`) is a GEMV straight off the packed weights; larger batches widen the
    /// weights to f32 a chunk of rows at a time and run cuBLAS SGEMM into the output columns.
    pub(super) fn linear_llm(
        &self,
        x: &CudaBuffer,
        w: &CudaBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) -> Option<CudaBuffer> {
        let q4 = matches!(w.storage(), CudaStorage::Q4(_));
        if !q4 && !(w.is_bf16() && !x.is_bf16()) {
            return None;
        }
        let mut tmp = None;
        let x = self.as_f32(x, &mut tmp);
        let mut out = self.pool_alloc_uninit_f32(m * n);
        let (mu, ku, nu) = (m as u32, k as u32, n as u32);
        if m <= 8 && k % 8 == 0 {
            let cfg = blocks((n.div_ceil(8), 1, 1), 256);
            match w.storage() {
                CudaStorage::Q4(q) => {
                    let f = self.llm_func(llm_cuda::GEMV_CUDA, "gemv_q4");
                    let group = q.group as u32;
                    unsafe {
                        self.stream
                            .launch_builder(&f)
                            .arg(x.f32_data())
                            .arg(&q.packed)
                            .arg(&q.scales)
                            .arg(&q.biases)
                            .arg(out.f32_data_mut())
                            .arg(&mu)
                            .arg(&ku)
                            .arg(&nu)
                            .arg(&group)
                            .launch(cfg)
                            .unwrap();
                    }
                }
                _ => {
                    let f = self.llm_func(llm_cuda::GEMV_CUDA, "gemv_bf16");
                    unsafe {
                        self.stream
                            .launch_builder(&f)
                            .arg(x.f32_data())
                            .arg(w.bf16_data())
                            .arg(out.f32_data_mut())
                            .arg(&mu)
                            .arg(&ku)
                            .arg(&nu)
                            .launch(cfg)
                            .unwrap();
                    }
                }
            }
            return Some(self.finish(out));
        }

        let chunk = (DEQUANT_CHUNK / k).clamp(1, n);
        let mut scratch = self.pool_alloc_uninit_f32(chunk * k);
        let mut row0 = 0;
        while row0 < n {
            let rows = chunk.min(n - row0);
            self.dequant_rows(w, scratch.f32_data_mut(), row0, rows, k);
            // Column-major view: out^T[row0.., :] = W_chunk (rows x k) · x^T (k x m).
            let mut dst = out.f32_data_mut().slice_mut(row0..);
            unsafe {
                self.cublas
                    .gemm(
                        GemmConfig {
                            transa: cublasOperation_t::CUBLAS_OP_T,
                            transb: cublasOperation_t::CUBLAS_OP_N,
                            m: rows as i32,
                            n: m as i32,
                            k: k as i32,
                            alpha: 1.0f32,
                            lda: k as i32,
                            ldb: k as i32,
                            beta: 0.0f32,
                            ldc: n as i32,
                        },
                        scratch.f32_data(),
                        x.f32_data(),
                        &mut dst,
                    )
                    .expect("cuBLAS sgemm (dequantized weights) failed");
            }
            row0 += rows;
        }
        Some(self.finish(out))
    }

    /// Widen weight rows `row0 .. row0 + rows` (bf16 or 4-bit) into `dst` as `[rows, k]` f32.
    fn dequant_rows(
        &self,
        w: &CudaBuffer,
        dst: &mut CudaSlice<f32>,
        row0: usize,
        rows: usize,
        k: usize,
    ) {
        let (r0, r, ku) = (row0 as u32, rows as u32, k as u32);
        match w.storage() {
            CudaStorage::Q4(q) => {
                let f = self.llm_func(llm_cuda::DEQUANT_CUDA, "dequant_q4");
                let group = q.group as u32;
                unsafe {
                    self.stream
                        .launch_builder(&f)
                        .arg(&q.packed)
                        .arg(&q.scales)
                        .arg(&q.biases)
                        .arg(dst)
                        .arg(&r0)
                        .arg(&r)
                        .arg(&ku)
                        .arg(&group)
                        .launch(per_elem(rows * k / 8))
                        .unwrap();
                }
            }
            CudaStorage::Bf16(s) => {
                let f = self.llm_func(llm_cuda::DEQUANT_CUDA, "dequant_bf16");
                unsafe {
                    self.stream
                        .launch_builder(&f)
                        .arg(s)
                        .arg(dst)
                        .arg(&r0)
                        .arg(&r)
                        .arg(&ku)
                        .launch(per_elem(rows * k))
                        .unwrap();
                }
            }
            CudaStorage::F32(_) => unreachable!(),
        }
    }

    /// Fused attention prologue (see `ComputeDevice::attention_prep`). None when the shape or
    /// precision isn't covered (bf16 activations, odd or > 256 head dims).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_prep_llm(
        &self,
        qkv: &CudaBuffer,
        q_norm: Option<&CudaBuffer>,
        k_norm: Option<&CudaBuffer>,
        cos: &CudaBuffer,
        sin: &CudaBuffer,
        k_cache: &mut CudaBuffer,
        v_cache: &mut CudaBuffer,
        seq: usize,
        (nh, nkv, hd): (usize, usize, usize),
        pos: usize,
        eps: f32,
    ) -> Option<CudaBuffer> {
        if qkv.is_bf16() || k_cache.is_bf16() || v_cache.is_bf16() || hd % 2 != 0 || hd > 256 {
            return None;
        }
        let kvd = nkv * hd;
        assert!(
            k_cache.len >= (pos + seq) * kvd && v_cache.len >= (pos + seq) * kvd,
            "KV cache too small"
        );
        let mut q = self.pool_alloc_uninit_f32(seq * nh * hd);
        let f = self.llm_func(llm_cuda::FUSED_CUDA, "attention_prep");
        let qn = q_norm.unwrap_or(qkv);
        let kn = k_norm.unwrap_or(qkv);
        let (nh_u, nkv_u, hd_u, pos_u) = (nh as u32, nkv as u32, hd as u32, pos as u32);
        let (has_qn, has_kn) = (q_norm.is_some() as u32, k_norm.is_some() as u32);
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(qkv.f32_data())
                .arg(qn.f32_data())
                .arg(kn.f32_data())
                .arg(cos.f32_data())
                .arg(sin.f32_data())
                .arg(q.f32_data_mut())
                .arg(k_cache.f32_data_mut())
                .arg(v_cache.f32_data_mut())
                .arg(&nh_u)
                .arg(&nkv_u)
                .arg(&hd_u)
                .arg(&pos_u)
                .arg(&eps)
                .arg(&has_qn)
                .arg(&has_kn)
                .launch(blocks((nh + 2 * nkv, seq, 1), 32))
                .unwrap();
        }
        Some(q)
    }

    /// Attention over a KV cache (see `ComputeDevice::kv_attention_window`): split-KV for
    /// decode, tiled for batches. `bidir`: every query sees the whole batch.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_llm(
        &self,
        q: &CudaBuffer,
        k: &CudaBuffer,
        v: &CudaBuffer,
        cache_start: usize,
        q_len: usize,
        (nh, nkv, d): (usize, usize, usize),
        window: usize,
        bidir: bool,
    ) -> CudaBuffer {
        assert!(
            d <= 256 && nh % nkv == 0,
            "attention needs head_dim at most 256 and n_heads a multiple of n_kv_heads"
        );
        let (mut tq, mut tk, mut tv) = (None, None, None);
        let (q, k, v) = (
            self.as_f32(q, &mut tq),
            self.as_f32(k, &mut tk),
            self.as_f32(v, &mut tv),
        );
        let longest = cache_start + q_len;
        assert!(k.len >= longest * nkv * d && v.len >= longest * nkv * d);
        let mut out = self.pool_alloc_uninit_f32(q_len * nh * d);
        let (cs, ql, nh_u, nkv_u, d_u, win, bi) = (
            cache_start as u32,
            q_len as u32,
            nh as u32,
            nkv as u32,
            d as u32,
            window as u32,
            bidir as u32,
        );

        if q_len > 1 {
            let bq: u32 = if d <= 128 { 32 } else { 16 };
            let f = self.llm_func(llm_cuda::ATTENTION_CUDA, "attn_prefill");
            unsafe {
                self.stream
                    .launch_builder(&f)
                    .arg(q.f32_data())
                    .arg(k.f32_data())
                    .arg(v.f32_data())
                    .arg(out.f32_data_mut())
                    .arg(&cs)
                    .arg(&ql)
                    .arg(&nh_u)
                    .arg(&nkv_u)
                    .arg(&d_u)
                    .arg(&win)
                    .arg(&bi)
                    .arg(&bq)
                    .launch(blocks((q_len.div_ceil(bq as usize), nh, 1), 256))
                    .unwrap();
            }
            return self.finish(out);
        }

        // Decode: split the keys a window can see across blocks, then combine.
        let base = if window > 0 {
            longest.saturating_sub(window)
        } else {
            0
        };
        let span = longest - base;
        let n_splits = span.div_ceil(256).clamp(1, 64);
        let split_len = span.div_ceil(n_splits);
        let mut partial = self.pool_alloc_uninit_f32(nh * n_splits * (d + 2));
        let (ns, sl, base_u) = (n_splits as u32, split_len as u32, base as u32);
        let f = self.llm_func(llm_cuda::ATTENTION_CUDA, "attn_partial");
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(q.f32_data())
                .arg(k.f32_data())
                .arg(v.f32_data())
                .arg(partial.f32_data_mut())
                .arg(&cs)
                .arg(&ql)
                .arg(&nh_u)
                .arg(&nkv_u)
                .arg(&d_u)
                .arg(&ns)
                .arg(&sl)
                .arg(&win)
                .arg(&base_u)
                .arg(&bi)
                .launch(blocks((nh, n_splits, 1), 256))
                .unwrap();
        }
        let f = self.llm_func(llm_cuda::ATTENTION_CUDA, "attn_combine");
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(partial.f32_data())
                .arg(out.f32_data_mut())
                .arg(&nh_u)
                .arg(&d_u)
                .arg(&ns)
                .launch(blocks((nh, 1, 1), 32))
                .unwrap();
        }
        self.finish(out)
    }

    /// Elementwise kernels from `FUSED_CUDA` taking `(in, out, a[, b])` with f32 buffers.
    fn fused_unary(
        &self,
        name: &'static str,
        x: &CudaBuffer,
        n_out: usize,
        a: u32,
        b: Option<u32>,
    ) -> CudaBuffer {
        let mut tmp = None;
        let x = self.as_f32(x, &mut tmp);
        let mut out = self.pool_alloc_uninit_f32(n_out);
        let f = self.llm_func(llm_cuda::FUSED_CUDA, name);
        unsafe {
            let mut l = self.stream.launch_builder(&f);
            l.arg(x.f32_data()).arg(out.f32_data_mut()).arg(&a);
            if let Some(b) = b.as_ref() {
                l.arg(b);
            }
            l.launch(per_elem(n_out)).unwrap();
        }
        self.finish(out)
    }

    pub(super) fn gelu_tanh_llm(&self, x: &CudaBuffer, n: usize) -> CudaBuffer {
        self.fused_unary("gelu_tanh", x, n, n as u32, None)
    }

    pub(super) fn gated_split_llm(
        &self,
        name: &'static str,
        gu: &CudaBuffer,
        rows: usize,
        ff: usize,
    ) -> CudaBuffer {
        self.fused_unary(name, gu, rows * ff, rows as u32, Some(ff as u32))
    }

    /// `a + b` over f32 buffers. None for bf16 inputs.
    pub(super) fn add_llm(&self, a: &CudaBuffer, b: &CudaBuffer, n: usize) -> Option<CudaBuffer> {
        if a.is_bf16() || b.is_bf16() {
            return None;
        }
        let mut out = self.pool_alloc_uninit_f32(n);
        let f = self.llm_func(llm_cuda::FUSED_CUDA, "add_f32");
        let n_u = n as u32;
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(a.f32_data())
                .arg(b.f32_data())
                .arg(out.f32_data_mut())
                .arg(&n_u)
                .launch(per_elem(n))
                .unwrap();
        }
        Some(out)
    }

    pub(super) fn layer_norm_llm(
        &self,
        x: &CudaBuffer,
        w: &CudaBuffer,
        b: &CudaBuffer,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> CudaBuffer {
        let (mut tx, mut tw, mut tb) = (None, None, None);
        let (x, w, b) = (
            self.as_f32(x, &mut tx),
            self.as_f32(w, &mut tw),
            self.as_f32(b, &mut tb),
        );
        let mut out = self.pool_alloc_uninit_f32(rows * dim);
        let f = self.llm_func(llm_cuda::FUSED_CUDA, "layer_norm");
        let dim_u = dim as u32;
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(x.f32_data())
                .arg(w.f32_data())
                .arg(b.f32_data())
                .arg(out.f32_data_mut())
                .arg(&dim_u)
                .arg(&eps)
                .launch(blocks((rows, 1, 1), 256))
                .unwrap();
        }
        self.finish(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn rope_half_llm(
        &self,
        input: &CudaBuffer,
        cos: &CudaBuffer,
        sin: &CudaBuffer,
        seq_len: usize,
        n_heads: usize,
        head_dim: usize,
        start_pos: usize,
    ) -> CudaBuffer {
        let mut tmp = None;
        let x = self.as_f32(input, &mut tmp);
        let mut out = self.pool_alloc_uninit_f32(x.len);
        // Heads past `seq_len * n_heads` (if any) pass through unrotated, as on the host.
        if out.len > seq_len * n_heads * head_dim {
            self.stream
                .memcpy_dtod(x.f32_data(), out.f32_data_mut())
                .unwrap();
        }
        let f = self.llm_func(llm_cuda::FUSED_CUDA, "rope_half");
        let (s, h, d, p) = (
            seq_len as u32,
            n_heads as u32,
            head_dim as u32,
            start_pos as u32,
        );
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(x.f32_data())
                .arg(cos.f32_data())
                .arg(sin.f32_data())
                .arg(out.f32_data_mut())
                .arg(&s)
                .arg(&h)
                .arg(&d)
                .arg(&p)
                .launch(per_elem(seq_len * n_heads * head_dim / 2))
                .unwrap();
        }
        self.finish(out)
    }
}

/// Parity against `CpuDevice` (the trait's portable reference implementations). Each test
/// skips, with a note, when there's no CUDA driver or GPU.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ComputeBuffer, ComputeDevice, CpuDevice};

    /// Runs `check` on the GPU, or skips when there's no CUDA device (unless
    /// `TANG_REQUIRE_CUDA=1`, for the GPU box, where skipping would hide a broken setup).
    fn on_gpu(check: fn(&CudaComputeDevice)) {
        match std::panic::catch_unwind(CudaComputeDevice::new) {
            Ok(Ok(dev)) => check(&dev),
            _ if std::env::var("TANG_REQUIRE_CUDA").is_ok_and(|v| v == "1") => {
                panic!("TANG_REQUIRE_CUDA=1 but no CUDA device")
            }
            _ => eprintln!("no CUDA device: skipping"),
        }
    }

    /// Deterministic values in [-scale/2, scale/2).
    fn vals(n: usize, seed: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i * 7919 + seed * 104_729) % 1009) as f32 / 1009.0 - 0.5)
            .map(|v| v * scale)
            .collect()
    }

    fn close(got: &[f32], want: &[f32], tol: f32, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= tol * (1.0 + w.abs()),
                "{what} at {i}: {g} vs {w}"
            );
        }
    }

    fn bf16_bits(v: &[f32]) -> Vec<u16> {
        v.iter().map(|&x| super::super::f32_to_bf16(x)).collect()
    }

    /// Random MLX-style 4-bit weights for an `[n, k]` matrix.
    fn q4(n: usize, k: usize, group: usize) -> (Vec<u32>, Vec<u16>, Vec<u16>) {
        let packed = (0..n * k / 8)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761) ^ 0x5a5a_a5a5)
            .collect();
        let scales = bf16_bits(&vals(n * k / group, 1, 0.02));
        let biases = bf16_bits(&vals(n * k / group, 2, 0.1));
        (packed, scales, biases)
    }

    fn q4_and_bf16_linear_and_embedding_vs_cpu<D: ComputeDevice>(g: &D) {
        let c = CpuDevice::new();
        for &(m, k, n, group) in &[(1, 256, 96, 64), (8, 512, 40, 32), (37, 128, 72, 128)] {
            let x = vals(m * k, 3, 2.0);
            let (p, s, b) = q4(n, k, group);
            let (gw, cw) = (
                g.upload_q4(&p, &s, &b, group),
                c.upload_q4(&p, &s, &b, group),
            );
            close(&gw.to_vec(), &c.download(&cw), 0.0, "q4 to_vec");
            let got = g.download(&g.linear(&g.upload(&x), &gw, m, k, n));
            let want = c.download(&c.linear(&c.upload(&x), &cw, m, k, n));
            close(&got, &want, 1e-4, &format!("q4 linear {m}x{k}x{n}"));

            let wb = bf16_bits(&vals(n * k, 4, 1.0));
            let (gb, cb) = (g.upload_bf16(&wb), c.upload_bf16(&wb));
            let got = g.download(&g.linear(&g.upload(&x), &gb, m, k, n));
            let want = c.download(&c.linear(&c.upload(&x), &cb, m, k, n));
            close(&got, &want, 1e-4, &format!("bf16 linear {m}x{k}x{n}"));

            let ids: Vec<u32> = (0..5).map(|i| (i * 13 % n) as u32).collect();
            for (gw, cw) in [(&gw, &cw), (&gb, &cb)] {
                let got = g.download(&g.embedding(gw, &g.upload_u32(&ids), 5, k));
                let want = c.download(&c.embedding(cw, &c.upload_u32(&ids), 5, k));
                close(&got, &want, 0.0, "embedding");
            }
        }
        // K not a multiple of 8 (bf16 goes through the dequantized GEMM).
        let (m, k, n) = (2, 30, 17);
        let x = vals(m * k, 5, 1.0);
        let wb = bf16_bits(&vals(n * k, 6, 1.0));
        let got = g.download(&g.linear(&g.upload(&x), &g.upload_bf16(&wb), m, k, n));
        let want = c.download(&c.linear(&c.upload(&x), &c.upload_bf16(&wb), m, k, n));
        close(&got, &want, 1e-4, "bf16 linear, odd K");
    }

    /// Several dequant chunks: many output rows against a long K.
    fn q4_linear_chunked_vs_cpu<D: ComputeDevice>(g: &D) {
        let c = CpuDevice::new();
        let (m, k, n) = (9, 4096, DEQUANT_CHUNK / 4096 * 2 + 40);
        let x = vals(m * k, 7, 1.0);
        let (p, s, b) = q4(n, k, 64);
        let got = g.download(&g.linear(&g.upload(&x), &g.upload_q4(&p, &s, &b, 64), m, k, n));
        let want = c.download(&c.linear(&c.upload(&x), &c.upload_q4(&p, &s, &b, 64), m, k, n));
        close(&got, &want, 1e-3, "chunked q4 linear");
    }

    fn attention_prep_vs_cpu<D: ComputeDevice>(g: &D) {
        let c = CpuDevice::new();
        for &(nh, nkv, hd, seq, pos, norms) in &[
            (4, 2, 128, 3, 5, true),
            (8, 4, 256, 1, 9, true),
            (2, 1, 64, 4, 0, false),
            (2, 2, 72, 2, 1, true),
        ] {
            let qkv = vals(seq * (nh + 2 * nkv) * hd, 8, 2.0);
            let (qn, kn) = (vals(hd, 9, 1.0), vals(hd, 10, 1.0));
            let max = 16;
            let (cos, sin) = (vals(max * hd / 2, 11, 2.0), vals(max * hd / 2, 12, 2.0));
            let cache = max * nkv * hd;
            let run = |d: &dyn Fn() -> (Vec<f32>, Vec<f32>, Vec<f32>)| d();
            let got = run(&|| {
                let (mut kc, mut vc) = (g.alloc(cache), g.alloc(cache));
                let (qn, kn) = (g.upload(&qn), g.upload(&kn));
                let q = g.attention_prep(
                    &g.upload(&qkv),
                    norms.then_some(&qn),
                    norms.then_some(&kn),
                    &g.upload(&cos),
                    &g.upload(&sin),
                    &mut kc,
                    &mut vc,
                    seq,
                    (nh, nkv, hd),
                    pos,
                    1e-6,
                );
                (g.download(&q), g.download(&kc), g.download(&vc))
            });
            let want = run(&|| {
                let (mut kc, mut vc) = (c.alloc(cache), c.alloc(cache));
                let (qn, kn) = (c.upload(&qn), c.upload(&kn));
                let q = c.attention_prep(
                    &c.upload(&qkv),
                    norms.then_some(&qn),
                    norms.then_some(&kn),
                    &c.upload(&cos),
                    &c.upload(&sin),
                    &mut kc,
                    &mut vc,
                    seq,
                    (nh, nkv, hd),
                    pos,
                    1e-6,
                );
                (c.download(&q), c.download(&kc), c.download(&vc))
            });
            let what = format!("attention_prep nh={nh} nkv={nkv} hd={hd}");
            close(&got.0, &want.0, 1e-4, &format!("{what} q"));
            close(&got.1, &want.1, 1e-4, &format!("{what} k cache"));
            close(&got.2, &want.2, 0.0, &format!("{what} v cache"));
        }
    }

    fn attention_vs_cpu<D: ComputeDevice>(g: &D) {
        let c = CpuDevice::new();
        // (nh, nkv, hd, cache_start, q_len, window, causal)
        let cases = [
            (4, 2, 128, 0, 1, 0, true),     // first decode step
            (4, 2, 128, 999, 1, 0, true),   // long decode: several splits
            (8, 4, 256, 700, 1, 512, true), // Gemma local layer, decode past the window
            (4, 1, 64, 0, 45, 0, true),     // prefill
            (4, 2, 128, 30, 70, 0, true),   // prefill after a cached prefix
            (4, 2, 256, 20, 50, 16, true),  // windowed prefill, D = 256
            (4, 4, 72, 3, 33, 0, false),    // image block: bidirectional within the batch
            (2, 2, 72, 0, 1, 0, false),
        ];
        for &(nh, nkv, hd, cs, ql, window, causal) in &cases {
            let total = cs + ql;
            let q = vals(ql * nh * hd, 13, 2.0);
            let k = vals(total * nkv * hd, 14, 2.0);
            let v = vals(total * nkv * hd, 15, 2.0);
            let got = g.download(&g.kv_attention_window(
                &g.upload(&q),
                &g.upload(&k),
                &g.upload(&v),
                cs,
                ql,
                (nh, nkv, hd),
                window,
                causal,
            ));
            let want = c.download(&c.kv_attention_window(
                &c.upload(&q),
                &c.upload(&k),
                &c.upload(&v),
                cs,
                ql,
                (nh, nkv, hd),
                window,
                causal,
            ));
            let what = format!("attention nh={nh} nkv={nkv} hd={hd} start={cs} len={ql} window={window} causal={causal}");
            close(&got, &want, 1e-4, &what);
        }
        // Vision tower shape (SigLIP: 16 heads of 72), smaller grid.
        let (n, nh, hd) = (100, 16, 72);
        let (q, k, v) = (
            vals(n * nh * hd, 16, 2.0),
            vals(n * nh * hd, 17, 2.0),
            vals(n * nh * hd, 18, 2.0),
        );
        let got =
            g.download(&g.attention_full(&g.upload(&q), &g.upload(&k), &g.upload(&v), n, nh, hd));
        let want =
            c.download(&c.attention_full(&c.upload(&q), &c.upload(&k), &c.upload(&v), n, nh, hd));
        close(&got, &want, 1e-4, "attention_full");
    }

    fn elementwise_and_norms_vs_cpu<D: ComputeDevice>(g: &D) {
        let c = CpuDevice::new();
        let (rows, ff, dim) = (3, 300, 1152);
        // Large gate values exercise the tanh clamp.
        let gu = vals(rows * 2 * ff, 19, 40.0);
        close(
            &g.download(&g.geglu_split(&g.upload(&gu), rows, ff)),
            &c.download(&c.geglu_split(&c.upload(&gu), rows, ff)),
            1e-5,
            "geglu_split",
        );
        close(
            &g.download(&g.swiglu_split(&g.upload(&gu), rows, ff)),
            &c.download(&c.swiglu_split(&c.upload(&gu), rows, ff)),
            1e-5,
            "swiglu_split",
        );
        close(
            &g.download(&g.gelu_tanh(&g.upload(&gu), gu.len())),
            &c.download(&c.gelu_tanh(&c.upload(&gu), gu.len())),
            1e-5,
            "gelu_tanh",
        );
        let a = vals(rows * dim, 20, 3.0);
        let b = vals(rows * dim, 21, 3.0);
        close(
            &g.download(&g.add_tensors_buf(&g.upload(&a), &g.upload(&b), a.len())),
            &c.download(&c.add_tensors_buf(&c.upload(&a), &c.upload(&b), a.len())),
            0.0,
            "add",
        );
        let (w, bias) = (vals(dim, 22, 1.0), vals(dim, 23, 1.0));
        close(
            &g.download(&g.layer_norm(
                &g.upload(&a),
                &g.upload(&w),
                &g.upload(&bias),
                rows,
                dim,
                1e-6,
            )),
            &c.download(&c.layer_norm(
                &c.upload(&a),
                &c.upload(&w),
                &c.upload(&bias),
                rows,
                dim,
                1e-6,
            )),
            1e-4,
            "layer_norm",
        );
        close(
            &g.download(&g.rms_norm(&g.upload(&a), &g.upload(&w), rows, dim, 1e-6)),
            &c.download(&c.rms_norm(&c.upload(&a), &c.upload(&w), rows, dim, 1e-6)),
            1e-4,
            "rms_norm",
        );
        let (seq, nh, hd, pos) = (3, 4, 128, 7);
        let x = vals(seq * nh * hd, 24, 2.0);
        let (cos, sin) = (vals(16 * hd / 2, 25, 2.0), vals(16 * hd / 2, 26, 2.0));
        close(
            &g.download(&g.rope_half_cached(
                &g.upload(&x),
                &g.upload(&cos),
                &g.upload(&sin),
                seq,
                nh,
                hd,
                pos,
            )),
            &c.download(&c.rope_half_cached(
                &c.upload(&x),
                &c.upload(&cos),
                &c.upload(&sin),
                seq,
                nh,
                hd,
                pos,
            )),
            1e-5,
            "rope_half",
        );
    }

    #[test]
    fn cuda_q4_and_bf16_linear_and_embedding_vs_cpu() {
        on_gpu(q4_and_bf16_linear_and_embedding_vs_cpu::<CudaComputeDevice>);
    }

    #[test]
    fn cuda_q4_linear_chunked_vs_cpu() {
        on_gpu(q4_linear_chunked_vs_cpu::<CudaComputeDevice>);
    }

    #[test]
    fn cuda_attention_prep_vs_cpu() {
        on_gpu(attention_prep_vs_cpu::<CudaComputeDevice>);
    }

    #[test]
    fn cuda_attention_vs_cpu() {
        on_gpu(attention_vs_cpu::<CudaComputeDevice>);
    }

    #[test]
    fn cuda_elementwise_and_norms_vs_cpu() {
        on_gpu(elementwise_and_norms_vs_cpu::<CudaComputeDevice>);
    }

    /// The same checks on Metal, so the harness itself is exercised on a Mac.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_q4_and_bf16_linear_and_embedding_vs_cpu() {
        q4_and_bf16_linear_and_embedding_vs_cpu(
            &crate::MetalDevice::new().expect("no Metal device"),
        );
    }

    /// The same checks on Metal, so the harness itself is exercised on a Mac.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_q4_linear_chunked_vs_cpu() {
        q4_linear_chunked_vs_cpu(&crate::MetalDevice::new().expect("no Metal device"));
    }

    /// The same checks on Metal, so the harness itself is exercised on a Mac.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_attention_prep_vs_cpu() {
        attention_prep_vs_cpu(&crate::MetalDevice::new().expect("no Metal device"));
    }

    /// The same checks on Metal, so the harness itself is exercised on a Mac.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_attention_vs_cpu() {
        attention_vs_cpu(&crate::MetalDevice::new().expect("no Metal device"));
    }

    /// The same checks on Metal, so the harness itself is exercised on a Mac.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_elementwise_and_norms_vs_cpu() {
        elementwise_and_norms_vs_cpu(&crate::MetalDevice::new().expect("no Metal device"));
    }
}
