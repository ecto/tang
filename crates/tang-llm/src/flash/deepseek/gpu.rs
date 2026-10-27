//! Device side of the port (M2): weights uploaded in their checkpoint format, and launchers for
//! [`SRC`]'s kernels. Correctness first; every launcher is tested against its CPU twin in
//! [`super::forward`].
//!
//! Pod or mew (`flock ~/tang-gpu.lock` around GPU runs):
//! `DS41_TINY=/tmp/ds41-tiny TANG_REQUIRE_CUDA=1 cargo test --release --no-default-features
//! --features cuda --lib flash::deepseek::gpu`

use super::st::{Dtype, St};
use anyhow::{bail, Context, Result};
use tang_moe::gpu::{self, DevBuf, Gpu, Module, Stream};

pub const SRC: &str = include_str!("kernels.cu");

type Func = cudarc::driver::sys::CUfunction;

/// `struct DsW` in kernels.cu (8 + 8 + 4 + 4 + 4, padded to 32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DsW {
    pub w: u64,
    pub s: u64,
    pub rows: i32,
    pub cols: i32,
    pub fmt: i32,
    _pad: i32,
}

/// A weight on the device and its kernel view.
pub struct DevW {
    _w: DevBuf,
    _s: Option<DevBuf>,
    pub v: DsW,
}

impl DevW {
    /// Upload `name` from the checkpoint as stored: f32, bf16, fp8 with 32 × 32 block scales,
    /// or I8-packed fp4 with per-row × 32 scales.
    pub fn upload(st: &St, name: &str) -> Result<Self> {
        let info = st.info(name)?.clone();
        let (rows, cols) = match info.shape.len() {
            1 => (1, info.shape[0]),
            _ => (info.shape[0], info.shape[1..].iter().product()),
        };
        let scale = || -> Result<DevBuf> {
            let sname = format!(
                "{}scale",
                name.strip_suffix("weight")
                    .context("quantized non-weight")?
            );
            Ok(DevBuf::from_slice(st.bytes(&sname)?)?)
        };
        let (fmt, cols, s) = match info.dtype {
            Dtype::F32 => (0, cols, None),
            Dtype::BF16 => (1, cols, None),
            Dtype::F8E4M3 => {
                let ss = st
                    .info(&format!("{}scale", name.strip_suffix("weight").unwrap()))?
                    .shape
                    .clone();
                if ss != [rows.div_ceil(32), cols.div_ceil(32)] {
                    bail!("{name}: per-row fp8 scales {ss:?} are not a GEMM weight");
                }
                (2, cols, Some(scale()?))
            }
            Dtype::I8 | Dtype::U8 => (3, cols * 2, Some(scale()?)),
            d => bail!("{name}: {d:?} is not a weight format"),
        };
        let w = DevBuf::from_slice(st.bytes(name)?)?;
        let v = DsW {
            w: w.ptr,
            s: s.as_ref().map_or(0, |b| b.ptr),
            rows: rows as i32,
            cols: cols as i32,
            fmt,
            _pad: 0,
        };
        Ok(DevW { _w: w, _s: s, v })
    }
}

pub struct Kernels {
    _m: Module,
    linear: Func,
    fp8_qdq: Func,
    fp4_qdq: Func,
    rms: Func,
    rope: Func,
    sparse_attn: Func,
    sinkhorn: Func,
    /// Every other kernel, by name, for [`Kernels::go`].
    more: std::collections::HashMap<&'static str, Func>,
}

/// The kernels launched through [`Kernels::go`].
const MORE: &[&str] = &[
    "ds_copy",
    "ds_hc_scale",
    "ds_hc_pre",
    "ds_hc_post",
    "ds_embed",
    "ds_mean_hc",
    "ds_swiglu",
    "ds_router",
    "ds_gather_rows",
    "ds_scatter_add",
    "ds_add_round",
    "ds_scale_round",
    "ds_round",
    "ds_pool",
    "ds_index_score",
    "ds_wo_a",
    "ds_engram_gate",
];

pub(crate) fn blocks(n: usize, per: usize) -> (u32, u32, u32) {
    // at least one block: every kernel bounds-checks, and a zero grid is a launch error
    (n.div_ceil(per).max(1) as u32, 1, 1)
}

impl Kernels {
    pub fn new(g: &Gpu) -> Result<Self> {
        let m = g.module(SRC)?;
        Ok(Kernels {
            linear: m.func("ds_linear")?,
            fp8_qdq: m.func("ds_fp8_qdq")?,
            fp4_qdq: m.func("ds_fp4_qdq")?,
            rms: m.func("ds_rms")?,
            rope: m.func("ds_rope")?,
            sparse_attn: m.func("ds_sparse_attn")?,
            sinkhorn: m.func("ds_sinkhorn")?,
            more: MORE
                .iter()
                .map(|&n| Ok((n, m.func(n)?)))
                .collect::<Result<_>>()?,
            _m: m,
        })
    }

    /// Launch kernel `name` (one of [`MORE`]).
    ///
    /// # Safety
    /// As [`gpu::launch`]: `args` must match the kernel's parameters.
    pub unsafe fn go(
        &self,
        name: &str,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        smem: u32,
        s: &Stream,
        args: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        let f = *self
            .more
            .get(name)
            .with_context(|| format!("no kernel {name}"))?;
        Ok(gpu::launch(f, grid, block, smem, s, args)?)
    }

    /// `y = x · wᵀ` for `t` rows (`x` already activation-quantized when the weight is).
    pub fn linear(&self, s: &Stream, w: &DsW, x: u64, y: u64, t: usize, round: bool) -> Result<()> {
        let (wv, ti, r) = (*w, t as i32, round as i32);
        unsafe {
            gpu::launch(
                self.linear,
                blocks(t * w.rows as usize * 32, 256),
                (256, 1, 1),
                0,
                s,
                tang_moe::args![wv, x, y, ti, r],
            )?
        }
        Ok(())
    }

    pub fn fp8_qdq(&self, s: &Stream, v: u64, n: usize) -> Result<()> {
        let g = (n / 32) as i32;
        unsafe {
            gpu::launch(
                self.fp8_qdq,
                blocks(n / 32, 128),
                (128, 1, 1),
                0,
                s,
                tang_moe::args![v, g],
            )?
        }
        Ok(())
    }

    pub fn fp4_qdq(
        &self,
        s: &Stream,
        v: u64,
        n: usize,
        block: usize,
        e4m3_scale: bool,
    ) -> Result<()> {
        let (g, b, e) = ((n / block) as i32, block as i32, e4m3_scale as i32);
        unsafe {
            gpu::launch(
                self.fp4_qdq,
                blocks(n / block, 128),
                (128, 1, 1),
                0,
                s,
                tang_moe::args![v, g, b, e],
            )?
        }
        Ok(())
    }

    pub fn rms(&self, s: &Stream, x: u64, w: &DsW, rows: usize, eps: f32) -> Result<()> {
        let (wv, n) = (*w, w.cols);
        unsafe {
            gpu::launch(
                self.rms,
                (rows as u32, 1, 1),
                (256, 1, 1),
                0,
                s,
                tang_moe::args![x, wv, n, eps],
            )?
        }
        Ok(())
    }

    /// Rotate the last `rd` of each `d`-wide vector, row `r` at `pos0 + r * pstride` (or
    /// `fixed` for every row).
    #[allow(clippy::too_many_arguments)]
    pub fn rope(
        &self,
        s: &Stream,
        v: u64,
        tab: u64,
        rows: usize,
        per_row: usize,
        d: usize,
        rd: usize,
        pos0: usize,
        pstride: usize,
        fixed: Option<usize>,
        inverse: bool,
    ) -> Result<()> {
        let n = rows * per_row * rd / 2;
        let (r, pr, di, rdi, p0, ps, fx, inv) = (
            rows as i32,
            per_row as i32,
            d as i32,
            rd as i32,
            pos0 as i32,
            pstride as i32,
            fixed.map_or(-1, |f| f as i32),
            inverse as i32,
        );
        unsafe {
            gpu::launch(
                self.rope,
                blocks(n, 256),
                (256, 1, 1),
                0,
                s,
                tang_moe::args![v, tab, r, pr, di, rdi, p0, ps, fx, inv],
            )?
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sparse_attn(
        &self,
        s: &Stream,
        q: u64,
        kv: u64,
        sink: u64,
        idxs: u64,
        o: u64,
        t: usize,
        h: usize,
        d: usize,
        k: usize,
        scale: f32,
    ) -> Result<()> {
        let (hi, di, ki) = (h as i32, d as i32, k as i32);
        let threads = d.clamp(32, 1024) as u32;
        unsafe {
            gpu::launch(
                self.sparse_attn,
                ((t * h) as u32, 1, 1),
                (threads, 1, 1),
                (k * 4) as u32,
                s,
                tang_moe::args![q, kv, sink, idxs, o, hi, di, ki, scale],
            )?
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sinkhorn(
        &self,
        s: &Stream,
        mixes: u64,
        scale: u64,
        base: u64,
        pre: u64,
        post: u64,
        comb: u64,
        t: usize,
        hc: usize,
        iters: usize,
        eps: f32,
    ) -> Result<()> {
        let (ti, hci, it) = (t as i32, hc as i32, iters as i32);
        unsafe {
            gpu::launch(
                self.sinkhorn,
                blocks(t, 64),
                (64, 1, 1),
                0,
                s,
                tang_moe::args![mixes, scale, base, pre, post, comb, ti, hci, it, eps],
            )?
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::forward::{self, tests::tiny, Model};
    use super::super::quant::{bf, fp4_qdq, fp8_qdq};
    use super::*;
    use std::path::Path;

    /// Deterministic bf16-valued test data.
    fn data(n: usize, seed: u64, amp: f32) -> Vec<f32> {
        let mut x = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                bf(((x >> 33) as f32 / (1u64 << 31) as f32 * 2.0 - 1.0) * amp)
            })
            .collect()
    }

    fn rel(a: &[f32], b: &[f32]) -> f32 {
        let m = b.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-12);
        a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / m
    }

    fn setup() -> Option<(Gpu, Kernels, Stream, Model)> {
        let g = Gpu::for_test()?;
        let dir = Path::new(&tiny()).join("hf");
        if !dir.join("model.safetensors").exists() {
            eprintln!("skip: no {} (run scripts/deepseek_tiny.py)", tiny());
            return None;
        }
        g.bind().unwrap();
        let k = Kernels::new(&g).unwrap();
        let s = Stream::blocking().unwrap();
        Some((g, k, s, Model::open(&dir).unwrap()))
    }

    #[test]
    fn quant_kernels_are_bit_exact() {
        let Some((_g, k, s, _m)) = setup() else {
            return;
        };
        let x = data(32 * 64, 1, 3.0);
        let dev = DevBuf::from_slice(&x).unwrap();
        k.fp8_qdq(&s, dev.ptr, x.len()).unwrap();
        s.sync().unwrap();
        let mut want = x.clone();
        fp8_qdq(&mut want, 32);
        assert_eq!(dev.read::<f32>(0, x.len()).unwrap(), want, "fp8");
        for (block, e4) in [(32, false), (16, true)] {
            let dev = DevBuf::from_slice(&x).unwrap();
            k.fp4_qdq(&s, dev.ptr, x.len(), block, e4).unwrap();
            s.sync().unwrap();
            let mut want = x.clone();
            fp4_qdq(&mut want, block, e4);
            assert_eq!(
                dev.read::<f32>(0, x.len()).unwrap(),
                want,
                "fp4 block {block}"
            );
        }
    }

    /// Every weight format against `Model::linear` (which quantizes the activation itself).
    #[test]
    fn linear_matches_cpu() {
        let Some((_g, k, s, m)) = setup() else { return };
        let t = 5;
        for name in [
            "layers.0.attn.wq_b.weight",           // fp8
            "layers.0.ffn.experts.3.w2.weight",    // fp4
            "layers.2.attn.compressor.wkv.weight", // bf16
            "layers.0.ffn.gate.weight",            // bf16 router
            "layers.1.engram.wkv.weight",
            "layers.0.attn.wo_b.weight",
            "layers.0.ffn.shared_experts.w2.weight",
            "mtp.0.main_proj.weight",
        ] {
            let w = DevW::upload(&m.st, name).unwrap();
            let cols = w.v.cols as usize;
            let rows = w.v.rows as usize;
            let x = data(t * cols, 7, 2.0);
            let want = m.linear(&x, t, name).unwrap();
            let dx = DevBuf::from_slice(&x).unwrap();
            if w.v.fmt >= 2 {
                k.fp8_qdq(&s, dx.ptr, x.len()).unwrap();
            }
            let dy = DevBuf::alloc((t * rows) * 4).unwrap();
            k.linear(&s, &w.v, dx.ptr, dy.ptr, t, true).unwrap();
            s.sync().unwrap();
            let got = dy.read::<f32>(0, t * rows).unwrap();
            let r = rel(&got, &want);
            eprintln!("{name}: rel {r:.2e}");
            assert!(r < 1e-2, "{name}: rel {r}");
        }
    }

    #[test]
    fn rms_rope_attn_sinkhorn_match_cpu() {
        let Some((_g, k, s, m)) = setup() else { return };
        let c = m.cfg.clone();
        let (t, h, d, rd) = (6, c.n_heads, c.head_dim, c.rope_head_dim);

        // rms
        let x = data(t * c.dim, 3, 4.0);
        let mut want = x.clone();
        m.rms(&mut want, "layers.0.attn_norm.weight").unwrap();
        let dx = DevBuf::from_slice(&x).unwrap();
        let nw = DevW::upload(&m.st, "layers.0.attn_norm.weight").unwrap();
        k.rms(&s, dx.ptr, &nw.v, t, c.norm_eps).unwrap();
        s.sync().unwrap();
        let r = rel(&dx.read::<f32>(0, x.len()).unwrap(), &want);
        assert!(r < 1e-2, "rms rel {r}");

        // rope, forward and inverse, at an offset
        let tab = &m.freqs[2];
        let flat: Vec<f32> = tab.iter().flat_map(|&(a, b)| [a, b]).collect();
        let dtab = DevBuf::from_slice(&flat).unwrap();
        let q = data(t * h * d, 4, 2.0);
        for inverse in [false, true] {
            let mut want = q.clone();
            forward::rope(&mut want, d, rd, h, tab, |i| 9 + i, inverse);
            let dq = DevBuf::from_slice(&q).unwrap();
            k.rope(&s, dq.ptr, dtab.ptr, t, h, d, rd, 9, 1, None, inverse)
                .unwrap();
            s.sync().unwrap();
            let r = rel(&dq.read::<f32>(0, q.len()).unwrap(), &want);
            assert!(r < 1e-2, "rope inverse={inverse} rel {r}");
        }

        // sparse attention with -1 holes and a sink
        let n = 20;
        let kv = data(n * d, 5, 1.0);
        let sink = data(h, 6, 1.0);
        let idxs: Vec<Vec<i32>> = (0..t)
            .map(|ti| {
                (0..8)
                    .map(|j| {
                        if (ti + j) % 5 == 0 {
                            -1
                        } else {
                            ((ti * 3 + j * 2) % n) as i32
                        }
                    })
                    .collect()
            })
            .collect();
        let want = forward::sparse_attn(&q, h, d, &kv, &sink, &idxs, (d as f32).powf(-0.5));
        let (dq, dkv, ds) = (
            DevBuf::from_slice(&q).unwrap(),
            DevBuf::from_slice(&kv).unwrap(),
            DevBuf::from_slice(&sink).unwrap(),
        );
        let flat_idx: Vec<i32> = idxs.iter().flatten().copied().collect();
        let di = DevBuf::from_slice(&flat_idx).unwrap();
        let o = DevBuf::alloc((t * h * d) * 4).unwrap();
        k.sparse_attn(
            &s,
            dq.ptr,
            dkv.ptr,
            ds.ptr,
            di.ptr,
            o.ptr,
            t,
            h,
            d,
            8,
            (d as f32).powf(-0.5),
        )
        .unwrap();
        s.sync().unwrap();
        let r = rel(&o.read::<f32>(0, t * h * d).unwrap(), &want);
        assert!(r < 1e-2, "sparse_attn rel {r}");

        // sinkhorn
        let hc = c.hc_mult;
        let mix = (2 + hc) * hc;
        let mixes = data(t * mix, 8, 2.0);
        let scale = vec![0.7f32, 1.1, 0.9];
        let base = data(mix, 9, 0.5);
        let (dm, dsc, db) = (
            DevBuf::from_slice(&mixes).unwrap(),
            DevBuf::from_slice(&scale).unwrap(),
            DevBuf::from_slice(&base).unwrap(),
        );
        let (pre, post, comb) = (
            DevBuf::alloc((t * hc) * 4).unwrap(),
            DevBuf::alloc((t * hc) * 4).unwrap(),
            DevBuf::alloc((t * hc * hc) * 4).unwrap(),
        );
        k.sinkhorn(
            &s,
            dm.ptr,
            dsc.ptr,
            db.ptr,
            pre.ptr,
            post.ptr,
            comb.ptr,
            t,
            hc,
            c.hc_sinkhorn_iters,
            c.hc_eps,
        )
        .unwrap();
        s.sync().unwrap();
        let (gp, gq, gc) = (
            pre.read::<f32>(0, t * hc).unwrap(),
            post.read::<f32>(0, t * hc).unwrap(),
            comb.read::<f32>(0, t * hc * hc).unwrap(),
        );
        for ti in 0..t {
            let (wp, wq, wc) = forward::sinkhorn(
                &mixes[ti * mix..(ti + 1) * mix],
                &scale,
                &base,
                hc,
                c.hc_sinkhorn_iters,
                c.hc_eps,
            );
            assert!(rel(&gp[ti * hc..(ti + 1) * hc], &wp) < 1e-5);
            assert!(rel(&gq[ti * hc..(ti + 1) * hc], &wq) < 1e-5);
            assert!(
                rel(&gc[ti * hc * hc..(ti + 1) * hc * hc], &wc) < 1e-5,
                "sinkhorn comb"
            );
        }
    }
}
