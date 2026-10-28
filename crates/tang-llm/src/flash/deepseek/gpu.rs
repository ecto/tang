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

/// kernels.cu, then the device-side selection (one module, so the latter can use the former's
/// helpers).
pub const SRC: &str = concat!(
    include_str!("kernels.cu"),
    include_str!("kernels_select.cu")
);

/// Threads per block of the selection kernels (`DS_SEL_THREADS` in kernels_select.cu).
const SEL: (u32, u32, u32) = (512, 1, 1);

type Func = cudarc::driver::sys::CUfunction;

/// `struct DsW` in kernels.cu (8 + 8 + 4 + 4 + 4, padded to 32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
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
    /// `ds_lin_f{fmt}_t{tm}_g{gu}`, by (fmt - 2, log2 tm, gu).
    fast: [[[Func; 2]; 4]; 2],
    /// `ds_ks_f{fmt}_g{gu}`, the tensor-core K-split GEMV, by (fmt - 2, gu).
    ks: [[Func; 2]; 2],
    /// `ds_moe_{gu,down}_f{fmt}_t{tm}`, the grouped MoE projections, by (fmt - 2, log2 tm,
    /// down).
    moe: [[[Func; 2]; 4]; 2],
    /// `ds_moe_ks_{gu,down}_f{fmt}`, the grouped MoE on the K-split path, by (fmt - 2, down).
    moe_ks: [[Func; 2]; 2],
    attn_qk: Func,
    attn_soft: Func,
    attn_pv: Func,
    /// Scratch for the tiled attention (partial scores, probabilities, denominators).
    scratch: DevBuf,
    /// K-split partials across blocks [SB][t][rows (× 2 for GU)] and per-row-group arrival
    /// counters (left at zero by the last block). One launch at a time per `Kernels`.
    ks_part: DevBuf,
    ks_cnt: DevBuf,
    /// The quantized activation for the K-split GEMV: e4m3 codes [t][K], then scales.
    ks_x: DevBuf,
    ks_quant: Func,
    /// SM count and the opt-in dynamic shared memory limit per block.
    sms: usize,
    smem_max: usize,
    /// `DS_EXACT=1`: the bit-exact kernels (`ds_lin_*`, `ds_sparse_attn`) instead of the
    /// K-split GEMV and the tiled attention.
    pub exact: bool,
    /// Every other kernel, by name, for [`Kernels::go`].
    more: std::collections::HashMap<&'static str, Func>,
}

/// The kernels launched through [`Kernels::go`].
const MORE: &[&str] = &[
    "ds_copy",
    "ds_hc_scale",
    "ds_hc_sinkhorn",
    "ds_hc_pre",
    "ds_hc_post",
    "ds_embed",
    "ds_mean_hc",
    "ds_swiglu",
    "ds_router",
    "ds_linear_t8",
    "ds_head_bf16",
    "ds_argmax_rows",
    "ds_gather_rows",
    "ds_scatter_add",
    "ds_add",
    "ds_add_round",
    "ds_scale_round",
    "ds_round",
    "ds_pool",
    "ds_index_score",
    "ds_wo_a",
    "ds_perm3",
    "ds_engram_gate",
    "ds_win_idx",
    "ds_idx_cat",
    "ds_gather_csr",
    "ds_scatter_csr",
    "ds_linear_csr",
    "ds_mask_scores",
    "ds_candidates",
    "ds_topk",
    "ds_route",
    "ds_fill_i32",
    "ds_copy_u8",
    "ds_spark_pick",
    "ds_moe_plan",
    "ds_moe_reduce",
];

/// Floats of attention scratch: partial scores [t][d/64][h][k], then P [t][h][k] and den
/// [t][h]. Larger problems fall back to `ds_sparse_attn`.
const ATTN_SCRATCH: usize = 6 << 20;

/// Floats of cross-block K-split partials, and row-group counters.
const KS_PART: usize = 1 << 21;
const KS_CNT: usize = 1 << 13;
/// Bytes for the quantized activation: 8 tokens × 16384 columns, codes and scales.
const KS_X: usize = 8 * 16384 * 2;
/// Row groups each grouped K-split block takes (blockIdx.x strides them).
const MOE_KS_ITEMS: usize = 4;

/// `__CUDA_ARCH__` the module was built for.
fn arch(m: &Module) -> Result<i32> {
    let f = m.func("ds_arch")?;
    let out = DevBuf::alloc(4)?;
    let s = Stream::blocking()?;
    unsafe { gpu::launch(f, (1, 1, 1), (1, 1, 1), 0, &s, tang_moe::args![out.ptr])? };
    s.sync()?;
    Ok(out.read::<i32>(0, 1)?[0])
}

fn lin_table(m: &Module, pre: &str) -> Result<[[[Func; 2]; 4]; 2]> {
    let mut a = [[[std::ptr::null_mut(); 2]; 4]; 2];
    for (fi, af) in a.iter_mut().enumerate() {
        for (ti, at) in af.iter_mut().enumerate() {
            for (gi, ag) in at.iter_mut().enumerate() {
                *ag = m.func(&format!("{pre}_f{}_t{}_g{gi}", fi + 2, 1 << ti))?;
            }
        }
    }
    Ok(a)
}

fn moe_table(m: &Module) -> Result<[[[Func; 2]; 4]; 2]> {
    let mut a = [[[std::ptr::null_mut(); 2]; 4]; 2];
    for (fi, af) in a.iter_mut().enumerate() {
        for (ti, at) in af.iter_mut().enumerate() {
            for (di, ad) in at.iter_mut().enumerate() {
                let kind = ["gu", "down"][di];
                *ad = m.func(&format!("ds_moe_{kind}_f{}_t{}", fi + 2, 1 << ti))?;
            }
        }
    }
    Ok(a)
}

/// The K split for `tiles` 16-row warp tiles with `groups` 32-column groups: S slices within a
/// block of 8 warps (RW = 8 / S tiles a block) and SB blocks along K, aiming at ≥ 2 blocks of 8
/// warps per SM while keeping ≥ 4 groups a warp. Returns (S, row-group blocks, SB).
fn ks_split(tiles: usize, groups: usize, sms: usize) -> (usize, usize, usize) {
    let target = 6 * sms;
    let mut sp = 1;
    while sp < 8 && tiles.div_ceil(8 / sp) < target && groups >= sp * 2 * 4 {
        sp *= 2;
    }
    let br = tiles.div_ceil(8 / sp).max(1);
    let mut sb = 1;
    while br * sb < target && groups >= (sb + 1) * sp * 4 && sb < 16 {
        sb += 1;
    }
    (sp, br, sb)
}

/// The `ds_lin_*` / `ds_moe_*` row-tile template index for `t` <= 8 rows.
fn tm_index(t: usize) -> Result<usize> {
    Ok(match t {
        1 => 0,
        2 => 1,
        3 | 4 => 2,
        5..=8 => 3,
        _ => bail!("tm_index: {t} rows"),
    })
}

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
            fast: lin_table(&m, "ds_lin")?,
            moe: moe_table(&m)?,
            moe_ks: [
                [m.func("ds_moe_ks_gu_f2")?, m.func("ds_moe_ks_down_f2")?],
                [m.func("ds_moe_ks_gu_f3")?, m.func("ds_moe_ks_down_f3")?],
            ],
            ks: [
                [m.func("ds_ks_f2_g0")?, m.func("ds_ks_f2_g1")?],
                [m.func("ds_ks_f3_g0")?, m.func("ds_ks_f3_g1")?],
            ],
            attn_qk: m.func("ds_attn_qk")?,
            attn_soft: m.func("ds_attn_soft")?,
            attn_pv: m.func("ds_attn_pv")?,
            scratch: DevBuf::alloc(ATTN_SCRATCH * 4)?,
            ks_part: DevBuf::alloc(KS_PART * 4)?,
            ks_cnt: DevBuf::from_slice(&vec![0i32; KS_CNT])?,
            ks_x: DevBuf::alloc(KS_X)?,
            ks_quant: m.func("ds_ks_quant")?,
            sms: g.attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            )? as usize,
            smem_max: {
                let mx = g.attribute(
                    cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                )?;
                for n in [
                    "ds_ks_f2_g0",
                    "ds_ks_f2_g1",
                    "ds_ks_f3_g0",
                    "ds_ks_f3_g1",
                    "ds_moe_ks_gu_f2",
                    "ds_moe_ks_gu_f3",
                    "ds_moe_ks_down_f2",
                    "ds_moe_ks_down_f3",
                ] {
                    let r = unsafe {
                        cudarc::driver::sys::cuFuncSetAttribute(
                            m.func(n)?,
                            cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                            mx - 4352, // static: red + last
                        )
                    };
                    gpu::check(r, "cuFuncSetAttribute")?;
                }
                (mx - 4352) as usize
            },
            // the mma path needs sm_80 (an arch-less NVRTC build may target less)
            exact: std::env::var("DS_EXACT").is_ok_and(|v| v == "1") || arch(&m)? < 800,
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
        Ok(super::obs::launch(f, grid, block, smem, s, args)?)
    }

    /// In place: -inf past each row's causal prefix (`causal`, ratio `r`) and where the u8
    /// candidate mask `cand` (0 for none) is 0. As forward.rs `indexer`.
    #[allow(clippy::too_many_arguments)]
    pub fn mask_scores(
        &self,
        s: &Stream,
        score: u64,
        cand: u64,
        t: usize,
        width: usize,
        causal: bool,
        r: usize,
    ) -> Result<()> {
        let (ti, wi, ci, ri) = (t as i32, width as i32, causal as i32, r as i32);
        unsafe {
            self.go(
                "ds_mask_scores",
                blocks(t * width, 256),
                (256, 1, 1),
                0,
                s,
                tang_moe::args![score, cand, ti, wi, ci, ri],
            )
        }
    }

    /// Candidate blocks into the u8 mask `cand` [t][width]. As [`super::forward::Model::candidates`].
    #[allow(clippy::too_many_arguments)]
    pub fn candidates(
        &self,
        s: &Stream,
        score: u64,
        cand: u64,
        t: usize,
        width: usize,
        causal: bool,
        r: usize,
        bs: usize,
        kb: usize,
    ) -> Result<()> {
        let nb = width.div_ceil(bs);
        let sc = DevBuf::alloc(t.max(1) * nb.max(1) * 4)?;
        let sel = DevBuf::alloc(t.max(1) * nb.max(1) * 4)?;
        let (ti, wi, ci, ri, bsi, kbi) = (
            t as i32,
            width as i32,
            causal as i32,
            r as i32,
            bs as i32,
            kb as i32,
        );
        unsafe {
            self.go(
                "ds_candidates",
                (t.max(1) as u32, 1, 1),
                SEL,
                0,
                s,
                tang_moe::args![score, sc.ptr, sel.ptr, cand, ti, wi, ci, ri, bsi, kbi],
            )?
        };
        // scratch drops here: drain first
        s.sync()?;
        Ok(())
    }

    /// Top-`k` per row of `score` [t][width] (ties to the lowest index), ascending, into
    /// `out[ti * stride ..]` as `idx + offset`, or -1 past the row's prefix.
    #[allow(clippy::too_many_arguments)]
    pub fn topk(
        &self,
        s: &Stream,
        score: u64,
        out: u64,
        t: usize,
        width: usize,
        k: usize,
        stride: usize,
        offset: usize,
        causal: bool,
        r: usize,
    ) -> Result<()> {
        let scratch = DevBuf::alloc(t.max(1) * k.max(1) * 4)?;
        let (ti, wi, ki, si, oi, ci, ri) = (
            t as i32,
            width as i32,
            k as i32,
            stride as i32,
            offset as i32,
            causal as i32,
            r as i32,
        );
        unsafe {
            self.go(
                "ds_topk",
                (t.max(1) as u32, 1, 1),
                SEL,
                0,
                s,
                tang_moe::args![score, scratch.ptr, out, ti, wi, ki, si, oi, ci, ri],
            )?
        };
        s.sync()?;
        Ok(())
    }

    /// The MoE router: raw gate logits [t][ne] (f32) and bias [ne] -> ids/weights [t][k]
    /// (sqrt-softplus scores, top-k of score + bias ties to the lower id, weights normalized
    /// when `norm` and scaled by `scale`).
    #[allow(clippy::too_many_arguments)]
    pub fn router(
        &self,
        s: &Stream,
        raw: u64,
        bias: u64,
        ids: u64,
        wts: u64,
        t: usize,
        ne: usize,
        k: usize,
        norm: bool,
        scale: f32,
    ) -> Result<()> {
        let (ti, nei, ki, ni) = (t as i32, ne as i32, k as i32, norm as i32);
        unsafe {
            self.go(
                "ds_router",
                (t as u32, 1, 1),
                (128, 1, 1),
                (ne * 5) as u32,
                s,
                tang_moe::args![raw, bias, ids, wts, ti, nei, ki, ni, scale],
            )
        }
    }

    /// Router ids/weights [t][k] to per-expert CSR: counts [ne], off [ne + 1], rows and
    /// route weights [t * k], rows ascending within an expert.
    #[allow(clippy::too_many_arguments)]
    pub fn route(
        &self,
        s: &Stream,
        ids: u64,
        wts: u64,
        counts: u64,
        off: u64,
        rows: u64,
        rw: u64,
        t: usize,
        k: usize,
        ne: usize,
    ) -> Result<()> {
        let (ti, ki, nei) = (t as i32, k as i32, ne as i32);
        unsafe {
            self.go(
                "ds_route",
                (1, 1, 1),
                (256, 1, 1),
                0,
                s,
                tang_moe::args![ids, wts, counts, off, rows, rw, ti, ki, nei],
            )
        }
    }

    /// `y = x · wᵀ` for `t` rows (`x` already activation-quantized when the weight is).
    pub fn linear(&self, s: &Stream, w: &DsW, x: u64, y: u64, t: usize, round: bool) -> Result<()> {
        let (wv, ti, r) = (*w, t as i32, round as i32);
        if (2..=8).contains(&t) {
            return unsafe {
                self.go(
                    "ds_linear_t8",
                    blocks(w.rows as usize * 32, 256),
                    (256, 1, 1),
                    0,
                    s,
                    tang_moe::args![wv, x, y, ti, r],
                )
            };
        }
        unsafe {
            super::obs::launch(
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

    /// The vocab head, `y` [t][rows] = `x` [t][cols] · `w`ᵀ (no rounding): `ds_head_bf16` for
    /// bf16 weights in 8-row chunks, so each row's logits are the same at every `t`;
    /// [`Kernels::linear`] otherwise.
    pub fn head(&self, s: &Stream, w: &DsW, x: u64, y: u64, t: usize) -> Result<()> {
        if w.fmt != 1 || w.cols % 8 != 0 || x % 16 != 0 {
            return self.linear(s, w, x, y, t, false);
        }
        let (rows, cols) = (w.rows as usize, w.cols as usize);
        for i0 in (0..t).step_by(8) {
            let ti = (t - i0).min(8) as i32;
            let (xp, yp) = (x + (i0 * cols * 4) as u64, y + (i0 * rows * 4) as u64);
            unsafe {
                self.go(
                    "ds_head_bf16",
                    blocks(rows.div_ceil(4) * 32, 256),
                    (256, 1, 1),
                    0,
                    s,
                    tang_moe::args![w.w, w.rows, w.cols, xp, yp, ti],
                )?
            };
        }
        Ok(())
    }

    /// Whether [`Kernels::linear_fast`] takes `w`: fp8 or fp4 with whole 32-column groups.
    pub fn fast_ok(w: &DsW) -> bool {
        w.fmt >= 2 && w.cols % 32 == 0 && w.cols > 0
    }

    /// `linear` for fp8/fp4 weights ([`Kernels::fast_ok`]), bit-identical to [`Kernels::linear`]
    /// (after [`Kernels::fp8_qdq`] when `qdq`), at near memory bandwidth for small `t`: weights
    /// are read once per 8 token rows. With `w3` (same shape and format as `w`), the fused
    /// SwiGLU: `y` = `ds_swiglu(bf(x·w1ᵀ), bf(x·w3ᵀ), rw, lim)`, `rw` per-token weights or 0.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_fast(
        &self,
        s: &Stream,
        w: &DsW,
        w3: Option<&DsW>,
        x: u64,
        y: u64,
        t: usize,
        round: bool,
        qdq: bool,
        rw: u64,
        lim: f32,
    ) -> Result<()> {
        if !Self::fast_ok(w) {
            bail!("linear_fast: fmt {} cols {}", w.fmt, w.cols);
        }
        if let Some(w3) = w3 {
            if (w3.fmt, w3.rows, w3.cols) != (w.fmt, w.rows, w.cols) {
                bail!("linear_fast: w1/w3 mismatch");
            }
        }
        if !self.exact {
            return self.linear_ks(s, w, w3, x, y, t, round, qdq, rw, lim);
        }
        self.linear_exact(s, w, w3, x, y, t, round, qdq, rw, lim)
    }

    /// The bit-exact `ds_lin_*` kernels behind [`Kernels::linear_fast`] under `DS_EXACT=1`.
    #[allow(clippy::too_many_arguments)]
    fn linear_exact(
        &self,
        s: &Stream,
        w: &DsW,
        w3: Option<&DsW>,
        x: u64,
        y: u64,
        t: usize,
        round: bool,
        qdq: bool,
        rw: u64,
        lim: f32,
    ) -> Result<()> {
        let (rows, cols) = (w.rows as usize, w.cols as usize);
        let gu = w3.is_some() as usize;
        let per_warp = if gu == 1 { 2 } else { 4 };
        let grid = blocks(rows.div_ceil(per_warp), 4);
        let (r, q) = (round as i32, qdq as i32);
        let mut t0 = 0;
        while t0 < t {
            let n = (t - t0).min(8);
            let tm = match n {
                1 => 0,
                2 => 1,
                3 | 4 => 2,
                _ => 3,
            };
            let f = self.fast[w.fmt as usize - 2][tm][gu];
            let (wv, w3v) = (*w, *w3.unwrap_or(w));
            let (xo, yo, ni) = (
                x + (t0 * cols * 4) as u64,
                y + (t0 * rows * 4) as u64,
                n as i32,
            );
            let rwo = if rw == 0 { 0 } else { rw + (t0 * 4) as u64 };
            unsafe {
                super::obs::launch(
                    f,
                    grid,
                    (128, 1, 1),
                    0,
                    s,
                    tang_moe::args![wv, w3v, xo, yo, ni, r, q, rwo, lim],
                )?
            }
            t0 += n;
        }
        Ok(())
    }

    /// The K-split GEMV behind [`Kernels::linear_fast`] (unless `DS_EXACT=1`): same contract,
    /// outputs within one bf16 ulp of the exact kernels in rare elements.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_ks(
        &self,
        s: &Stream,
        w: &DsW,
        w3: Option<&DsW>,
        x: u64,
        y: u64,
        t: usize,
        round: bool,
        qdq: bool,
        rw: u64,
        lim: f32,
    ) -> Result<()> {
        if !Self::fast_ok(w) {
            bail!("linear_ks: fmt {} cols {}", w.fmt, w.cols);
        }
        if let Some(w3) = w3 {
            if (w3.fmt, w3.rows, w3.cols) != (w.fmt, w.rows, w.cols) {
                bail!("linear_ks: w1/w3 mismatch");
            }
        }
        let (rows, cols) = (w.rows as usize, w.cols as usize);
        let gu = w3.is_some() as usize;
        let out = if gu == 1 { 8 } else { 16 };
        let pitch = cols.div_ceil(128) * 128 + 32;
        let nch = cols / 32;
        if 8 * pitch + 8 * nch * 4 > self.smem_max || cols > 16384 {
            // too wide to stage: the exact kernels
            return self.linear_exact(s, w, w3, x, y, t, round, qdq, rw, lim);
        }
        let r = round as i32;
        let _ = qdq; // quantized either way (idempotent on a quantized activation)
        let f = self.ks[w.fmt as usize - 2][gu];
        let (codes, scales) = (self.ks_x.ptr, self.ks_x.ptr + (8 * 16384) as u64);
        let mut t0 = 0;
        while t0 < t {
            let n = (t - t0).min(8);
            let ng = (n * nch) as i32;
            let xo = x + (t0 * cols * 4) as u64;
            unsafe {
                super::obs::launch(
                    self.ks_quant,
                    blocks(n * nch * 32, 256),
                    (256, 1, 1),
                    0,
                    s,
                    tang_moe::args![xo, codes, scales, ng],
                )?
            }
            let smem = n * pitch + n * nch * 4;
            let (sp, br, mut sb) = ks_split(rows.div_ceil(out), nch, self.sms);
            if sb * n * rows * (1 + gu) > KS_PART || br > KS_CNT {
                sb = 1;
            }
            let mut per_sm = 0i32;
            unsafe {
                gpu::check(
                    cudarc::driver::sys::cuOccupancyMaxActiveBlocksPerMultiprocessor(
                        &mut per_sm,
                        f,
                        256,
                        smem,
                    ),
                    "occupancy",
                )?
            };
            let grid = (br * sb).min(self.sms * per_sm.max(1) as usize);
            let (wv, w3v) = (*w, *w3.unwrap_or(w));
            let (yo, ni) = (y + (t0 * rows * 4) as u64, n as i32);
            let rwo = if rw == 0 { 0 } else { rw + (t0 * 4) as u64 };
            let (si, sbi, bri) = (sp as i32, sb as i32, br as i32);
            let (part, cnt) = (self.ks_part.ptr, self.ks_cnt.ptr);
            unsafe {
                super::obs::launch(
                    f,
                    (grid as u32, 1, 1),
                    (256, 1, 1),
                    smem as u32,
                    s,
                    tang_moe::args![
                        wv, w3v, codes, scales, yo, ni, r, rwo, lim, si, sbi, bri, part, cnt
                    ],
                )?
            }
            t0 += n;
        }
        Ok(())
    }

    /// Whether the grouped K-split kernels take `t` <= 8 tokens per expert over `cols`
    /// columns, with `n` activation rows to quantize into `ks_x`.
    fn moe_ks_ok(&self, t: usize, cols: usize, n: usize) -> bool {
        let pitch = cols.div_ceil(128) * 128 + 32;
        (1..=8).contains(&t)
            && cols % 32 == 0
            && cols > 0
            && n * cols <= 8 * 16384
            && t * pitch + t * (cols / 32) * 4 <= self.smem_max
    }

    /// `ds_ks_quant` of `n` rows × `cols` at `x` into `ks_x`: (codes, scales).
    fn ks_quantize(&self, s: &Stream, x: u64, n: usize, cols: usize) -> Result<(u64, u64)> {
        let (codes, scales) = (self.ks_x.ptr, self.ks_x.ptr + (8 * 16384) as u64);
        let ng = (n * cols / 32) as i32;
        unsafe {
            super::obs::launch(
                self.ks_quant,
                blocks(n * cols, 256),
                (256, 1, 1),
                0,
                s,
                tang_moe::args![x, codes, scales, ng],
            )?
        }
        Ok((codes, scales))
    }

    /// The grouped K-split shape for one expert's `rows` × `cols` (`out` rows a warp tile):
    /// S K-slices a block (≥ 8 groups a warp), its nbr row groups, and the blocks per slot,
    /// each striding over ~4 row groups so the staged activation amortizes over weights.
    fn moe_ks_grid(rows: usize, out: usize, cols: usize) -> (usize, usize, usize) {
        let nch = cols / 32;
        let mut sp = 1;
        while sp < 8 && nch >= sp * 2 * 8 {
            sp *= 2;
        }
        let nbr = rows.div_ceil(out).div_ceil(8 / sp).max(1);
        (sp, nbr, nbr.div_ceil(MOE_KS_ITEMS))
    }

    /// Grouped MoE gate/up + SwiGLU for every routed expert of a layer in one launch (see
    /// `DS_MOE` in kernels.cu): `w1s`/`w3s` device arrays of [`DsW`] per expert (format
    /// `fmt`, `inter` rows), `elist` [slots] from `ds_moe_plan`, the CSR of [`Kernels::route`],
    /// `xq` the fp8-quantized activation [t][d], into `h` [t * k][inter] at CSR positions.
    /// Bit-identical to the per-expert `ds_linear_csr` + `ds_swiglu` loop. `t` <= 8.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_gu(
        &self,
        s: &Stream,
        fmt: i32,
        w1s: u64,
        w3s: u64,
        elist: u64,
        off: u64,
        rows: u64,
        rw: u64,
        xq: u64,
        h: u64,
        t: usize,
        slots: usize,
        inter: usize,
        d: usize,
        lim: f32,
    ) -> Result<()> {
        if !self.exact && self.moe_ks_ok(t, d, t) {
            let f = self.moe_ks[fmt as usize - 2][0];
            let (codes, scales) = self.ks_quantize(s, xq, t, d)?;
            let (sp, nbr, gx) = Self::moe_ks_grid(inter, 8, d);
            let (si, nbri) = (sp as i32, nbr as i32);
            unsafe {
                super::obs::launch(
                    f,
                    (gx as u32, slots.max(1) as u32, 1),
                    (256, 1, 1),
                    (t * (d.div_ceil(128) * 128 + 32) + t * (d / 32) * 4) as u32,
                    s,
                    tang_moe::args![
                        w1s, w3s, elist, off, rows, rw, codes, scales, h, lim, si, nbri
                    ],
                )?
            }
            return Ok(());
        }
        let f = self.moe[fmt as usize - 2][tm_index(t)?][0];
        let grid = (inter.div_ceil(2).div_ceil(4) as u32, slots.max(1) as u32, 1);
        unsafe {
            super::obs::launch(
                f,
                grid,
                (128, 1, 1),
                0,
                s,
                tang_moe::args![w1s, w3s, elist, off, rows, rw, xq, h, lim],
            )?
        }
        Ok(())
    }

    /// Grouped MoE down projection: `hq` [t * k][inter] (quantized `h`) to per-slot partials
    /// `part` [t * k][d], bf16-rounded, for `ds_moe_reduce`.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_down(
        &self,
        s: &Stream,
        fmt: i32,
        w2s: u64,
        elist: u64,
        off: u64,
        hq: u64,
        part: u64,
        t: usize,
        slots: usize,
        d: usize,
        inter: usize,
        topk: usize,
    ) -> Result<()> {
        let n = t * topk;
        if !self.exact && self.moe_ks_ok(t, inter, n) {
            let f = self.moe_ks[fmt as usize - 2][1];
            let (codes, scales) = self.ks_quantize(s, hq, n, inter)?;
            let (sp, nbr, gx) = Self::moe_ks_grid(d, 16, inter);
            let (si, nbri) = (sp as i32, nbr as i32);
            unsafe {
                super::obs::launch(
                    f,
                    (gx as u32, slots.max(1) as u32, 1),
                    (256, 1, 1),
                    (t * (inter.div_ceil(128) * 128 + 32) + t * (inter / 32) * 4) as u32,
                    s,
                    tang_moe::args![w2s, elist, off, codes, scales, part, si, nbri],
                )?
            }
            return Ok(());
        }
        let f = self.moe[fmt as usize - 2][tm_index(t)?][1];
        let grid = (d.div_ceil(4).div_ceil(4) as u32, slots.max(1) as u32, 1);
        unsafe {
            super::obs::launch(
                f,
                grid,
                (128, 1, 1),
                0,
                s,
                tang_moe::args![w2s, elist, off, hq, part],
            )?
        }
        Ok(())
    }

    pub fn fp8_qdq(&self, s: &Stream, v: u64, n: usize) -> Result<()> {
        let g = (n / 32) as i32;
        unsafe {
            super::obs::launch(
                self.fp8_qdq,
                blocks(n, 256),
                (256, 1, 1),
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
            super::obs::launch(
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
            super::obs::launch(
                self.rms,
                (rows as u32, 1, 1),
                (1024, 1, 1),
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
            super::obs::launch(
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
        split: usize,
        kv1: u64,
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
        let sp = split.min(i32::MAX as usize) as i32;
        let need = t * h * k * (d / 64 + 1) + t * h;
        if !self.exact && h % 64 == 0 && d % 64 == 0 && k > 0 && k <= 960 && need <= ATTN_SCRATCH {
            let part = self.scratch.ptr;
            let pr = part + (t * h * k * (d / 64) * 4) as u64;
            let den = pr + (t * h * k * 4) as u64;
            let ti = t as i32;
            unsafe {
                super::obs::launch(
                    self.attn_qk,
                    ((t * (h / 64) * k.div_ceil(32) * (d / 64)) as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    s,
                    tang_moe::args![q, kv, sp, kv1, idxs, part, hi, di, ki],
                )?;
                super::obs::launch(
                    self.attn_soft,
                    blocks(t * h * 32, 256),
                    (256, 1, 1),
                    0,
                    s,
                    tang_moe::args![part, sink, idxs, pr, den, ti, hi, di, ki, scale],
                )?;
                super::obs::launch(
                    self.attn_pv,
                    ((t * (h / 8) * (d / 64)) as u32, 1, 1),
                    (256, 1, 1),
                    (9 * k * 4) as u32,
                    s,
                    tang_moe::args![pr, den, kv, sp, kv1, idxs, o, hi, di, ki],
                )?;
            }
            return Ok(());
        }
        let threads = d.clamp(32, 1024) as u32;
        unsafe {
            super::obs::launch(
                self.sparse_attn,
                ((t * h) as u32, 1, 1),
                (threads, 1, 1),
                (k * 4) as u32,
                s,
                tang_moe::args![q, kv, sp, kv1, sink, idxs, o, hi, di, ki, scale],
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
            super::obs::launch(
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
            // the t <= 8 kernel is bit-identical to one row at a time
            for ti in 0..t {
                k.linear(
                    &s,
                    &w.v,
                    dx.ptr + (ti * cols * 4) as u64,
                    dy.ptr + (ti * rows * 4) as u64,
                    1,
                    true,
                )
                .unwrap();
            }
            s.sync().unwrap();
            let one = dy.read::<f32>(0, t * rows).unwrap();
            assert!(
                got.iter()
                    .zip(&one)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{name}: t8 != t1"
            );
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
            usize::MAX,
            0,
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

    /// Activations with group magnitudes spread over a few binades, so the fused activation
    /// quant sees varied scales.
    fn acts(n: usize, seed: u64) -> Vec<f32> {
        let mut x = data(n, seed, 2.0);
        for (g, c) in x.chunks_mut(32).enumerate() {
            let m = 2f32.powi((g % 7) as i32 - 3);
            for v in c {
                *v = bf(*v * m);
            }
        }
        x
    }

    /// The slow path: fp8_qdq on a copy, then ds_linear (bf16 out).
    fn slow(k: &Kernels, s: &Stream, w: &DsW, x: &[f32], t: usize) -> Vec<f32> {
        let dx = DevBuf::from_slice(x).unwrap();
        k.fp8_qdq(s, dx.ptr, x.len()).unwrap();
        let dy = DevBuf::alloc(t * w.rows as usize * 4).unwrap();
        k.linear(s, w, dx.ptr, dy.ptr, t, true).unwrap();
        s.sync().unwrap();
        dy.read::<f32>(0, t * w.rows as usize).unwrap()
    }

    /// The fast-path bar against the exact kernels: bit for bit under `DS_EXACT=1`; otherwise
    /// ≥ 99.5% of elements equal and every other one a neighbouring bf16 value.
    fn check(k: &Kernels, got: &[f32], want: &[f32], what: &str) {
        if k.exact {
            assert_eq!(got, want, "{what}");
            return;
        }
        assert_eq!(got.len(), want.len());
        let mut diff = 0;
        for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
            if a.to_bits() == b.to_bits() {
                continue;
            }
            diff += 1;
            let (ua, ub) = ((a.to_bits() >> 16) as i32, (b.to_bits() >> 16) as i32);
            let near = (a.signum() == b.signum() && (ua - ub).abs() == 1) || (a == 0.0 && b == 0.0);
            assert!(near, "{what}: [{i}] {a} vs {b} is not a bf16 neighbour");
        }
        assert!(
            diff as f64 <= 0.005 * got.len() as f64,
            "{what}: {diff} of {} differ",
            got.len()
        );
    }

    /// The fast GEMV against the slow path, bit for bit, for every fp8/fp4 weight shape in
    /// the fixture, with the activation quant fused and not, over decode and verify widths
    /// (and t > 8, which runs in launches of 8).
    #[test]
    fn fast_linear_matches_slow() {
        let Some((_g, k, s, m)) = setup() else { return };
        let mut seen = std::collections::HashSet::new();
        let mut names: Vec<&str> = m.st.names().filter(|n| n.ends_with("weight")).collect();
        names.sort();
        let mut checked = 0;
        for name in names {
            let info = m.st.info(name).unwrap();
            if !matches!(info.dtype, Dtype::F8E4M3 | Dtype::I8 | Dtype::U8) || info.shape.len() != 2
            {
                continue;
            }
            if !seen.insert((info.dtype as u8, info.shape.clone())) {
                continue;
            }
            let Ok(w) = DevW::upload(&m.st, name) else {
                continue;
            };
            assert!(Kernels::fast_ok(&w.v), "{name}");
            let (rows, cols) = (w.v.rows as usize, w.v.cols as usize);
            for t in [1, 2, 3, 4, 5, 6, 8, 11] {
                let x = acts(t * cols, 11 + t as u64);
                let want = slow(&k, &s, &w.v, &x, t);
                let dy = DevBuf::alloc(t * rows * 4).unwrap();
                // fused qdq
                let dx = DevBuf::from_slice(&x).unwrap();
                k.linear_fast(&s, &w.v, None, dx.ptr, dy.ptr, t, true, true, 0, 0.0)
                    .unwrap();
                s.sync().unwrap();
                check(
                    &k,
                    &dy.read::<f32>(0, t * rows).unwrap(),
                    &want,
                    &format!("{name} t {t} fused"),
                );
                // pre-quantized
                k.fp8_qdq(&s, dx.ptr, x.len()).unwrap();
                k.linear_fast(&s, &w.v, None, dx.ptr, dy.ptr, t, true, false, 0, 0.0)
                    .unwrap();
                s.sync().unwrap();
                check(
                    &k,
                    &dy.read::<f32>(0, t * rows).unwrap(),
                    &want,
                    &format!("{name} t {t}"),
                );
                // unrounded f32 against unrounded ds_linear
                let dq = DevBuf::from_slice(&x).unwrap();
                k.fp8_qdq(&s, dq.ptr, x.len()).unwrap();
                k.linear(&s, &w.v, dq.ptr, dy.ptr, t, false).unwrap();
                s.sync().unwrap();
                let wf = dy.read::<f32>(0, t * rows).unwrap();
                k.linear_fast(&s, &w.v, None, dq.ptr, dy.ptr, t, false, false, 0, 0.0)
                    .unwrap();
                s.sync().unwrap();
                let r = rel(&dy.read::<f32>(0, t * rows).unwrap(), &wf);
                assert!(r < 1e-5, "{name} t {t}: f32 rel {r:e}");
            }
            eprintln!("{name} fmt {} [{rows}, {cols}]: exact", w.v.fmt);
            checked += 1;
        }
        assert!(checked >= 8, "only {checked} shapes");
    }

    /// The fused w1/w3 GEMV + SwiGLU against two slow GEMVs and ds_swiglu, bit for bit.
    #[test]
    fn fast_swiglu_matches_slow() {
        let Some((_g, k, s, m)) = setup() else { return };
        let lim = m.cfg.swiglu_limit;
        for p in ["layers.0.ffn.experts.0", "layers.0.ffn.shared_experts"] {
            let w1 = DevW::upload(&m.st, &format!("{p}.w1.weight")).unwrap();
            let w3 = DevW::upload(&m.st, &format!("{p}.w3.weight")).unwrap();
            let (inter, cols) = (w1.v.rows as usize, w1.v.cols as usize);
            for t in [1, 2, 5, 6, 8, 11] {
                for routed in [false, true] {
                    let x = acts(t * cols, 21 + t as u64);
                    let g = DevBuf::from_slice(&slow(&k, &s, &w1.v, &x, t)).unwrap();
                    let u = DevBuf::from_slice(&slow(&k, &s, &w3.v, &x, t)).unwrap();
                    let rw = DevBuf::from_slice(&data(t, 5, 1.5)).unwrap();
                    let rwp = if routed { rw.ptr } else { 0 };
                    let h = DevBuf::alloc(t * inter * 4).unwrap();
                    let (nn, ii) = ((t * inter) as i64, inter as i32);
                    unsafe {
                        k.go(
                            "ds_swiglu",
                            blocks(t * inter, 256),
                            (256, 1, 1),
                            0,
                            &s,
                            tang_moe::args![g.ptr, u.ptr, rwp, h.ptr, nn, ii, lim],
                        )
                        .unwrap()
                    };
                    s.sync().unwrap();
                    let want = h.read::<f32>(0, t * inter).unwrap();
                    let dx = DevBuf::from_slice(&x).unwrap();
                    k.linear_fast(
                        &s,
                        &w1.v,
                        Some(&w3.v),
                        dx.ptr,
                        h.ptr,
                        t,
                        true,
                        true,
                        rwp,
                        lim,
                    )
                    .unwrap();
                    s.sync().unwrap();
                    check(
                        &k,
                        &h.read::<f32>(0, t * inter).unwrap(),
                        &want,
                        &format!("{p} t {t} routed {routed}"),
                    );
                }
            }
            eprintln!("{p} fmt {}: swiglu exact", w1.v.fmt);
        }
    }

    /// [`check`]'s bar for random weights, where a sum can cancel to far below its terms:
    /// ≤ 0.5% of elements differ, each a bf16 neighbour or within 2^-8 of the output's scale
    /// (the K-split's reduction order against the exact kernels').
    fn check_abs(got: &[f32], want: &[f32], what: &str) {
        assert_eq!(got.len(), want.len());
        let m = want.iter().fold(0f32, |m, x| m.max(x.abs()));
        let mut diff = 0;
        for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
            if a.to_bits() == b.to_bits() {
                continue;
            }
            diff += 1;
            let (ua, ub) = ((a.to_bits() >> 16) as i32, (b.to_bits() >> 16) as i32);
            let near =
                (a.signum() == b.signum() && (ua - ub).abs() == 1) || (a - b).abs() <= m / 256.0;
            assert!(near, "{what}: [{i}] {a} vs {b} (scale {m})");
        }
        assert!(
            diff as f64 <= 0.005 * got.len() as f64,
            "{what}: {diff} of {} differ",
            got.len()
        );
    }

    /// Random fp8 (fmt 2) or fp4 (fmt 3) weight [rows, cols], kept alive in `keep`.
    fn rand_w(fmt: i32, rows: usize, cols: usize, seed: u64, keep: &mut Vec<DevBuf>) -> DsW {
        let mut x = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
        let mut rb = |n: usize| -> Vec<u8> {
            (0..n)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    x as u8
                })
                .collect()
        };
        let mut wb = rb(if fmt == 3 {
            rows * cols / 2
        } else {
            rows * cols
        });
        if fmt == 2 {
            for b in &mut wb {
                if *b & 0x7f == 0x7f {
                    *b = 0x10; // no NaN codes
                }
            }
        }
        let ns = if fmt == 3 {
            rows * cols / 32
        } else {
            rows.div_ceil(32) * cols / 32
        };
        let sb: Vec<u8> = rb(ns).iter().map(|b| 118 + b % 8).collect();
        let (w, sc) = (
            DevBuf::from_slice(&wb).unwrap(),
            DevBuf::from_slice(&sb).unwrap(),
        );
        let v = DsW {
            w: w.ptr,
            s: sc.ptr,
            rows: rows as i32,
            cols: cols as i32,
            fmt,
            _pad: 0,
        };
        keep.push(w);
        keep.push(sc);
        v
    }

    /// The grouped K-split MoE kernels against the grouped exact ones (bit-identical to the
    /// per-expert loop), h and the down partials, under the fast-path bar, for fp8 and fp4,
    /// decode and verify widths, distinct and shared experts.
    #[test]
    fn moe_grouped_ks_matches_exact() {
        let Some(g) = Gpu::for_test() else { return };
        g.bind().unwrap();
        let mut k = Kernels::new(&g).unwrap();
        if k.exact {
            eprintln!("exact build: nothing to compare");
            return;
        }
        let s = Stream::blocking().unwrap();
        let (ne, topk, d, inter) = (32usize, 4usize, 1024usize, 512usize);
        for fmt in [2, 3] {
            let mut keep = Vec::new();
            let (mut w1, mut w3, mut w2) = (vec![], vec![], vec![]);
            for e in 0..ne as u64 {
                w1.push(rand_w(fmt, inter, d, 3 * e + 1, &mut keep));
                w3.push(rand_w(fmt, inter, d, 3 * e + 2, &mut keep));
                w2.push(rand_w(fmt, d, inter, 3 * e + 3, &mut keep));
            }
            let (w1, w3, w2) = (
                DevBuf::from_slice(&w1).unwrap(),
                DevBuf::from_slice(&w3).unwrap(),
                DevBuf::from_slice(&w2).unwrap(),
            );
            let mask = DevBuf::from_slice(&vec![1i32; ne]).unwrap();
            for t in [1, 2, 3, 6, 8] {
                for shared in [false, true] {
                    let n = t * topk;
                    let ids: Vec<i32> = (0..n)
                        .map(|i| {
                            let (ti, j) = (i / topk, i % topk);
                            if shared {
                                (j * 5 + 1) as i32
                            } else {
                                ((ti * 7 + j * 3 + 2) % ne) as i32
                            }
                        })
                        .collect();
                    let wts = DevBuf::from_slice(&data(n, 9 + t as u64, 1.0)).unwrap();
                    let ids = DevBuf::from_slice(&ids).unwrap();
                    let (counts, off) = (
                        DevBuf::alloc(ne * 4).unwrap(),
                        DevBuf::alloc((ne + 1) * 4).unwrap(),
                    );
                    let (rows, rw) = (DevBuf::alloc(n * 4).unwrap(), DevBuf::alloc(n * 4).unwrap());
                    k.route(
                        &s, ids.ptr, wts.ptr, counts.ptr, off.ptr, rows.ptr, rw.ptr, t, topk, ne,
                    )
                    .unwrap();
                    let x = DevBuf::from_slice(&data(t * d, 3 + t as u64, 2.0)).unwrap();
                    k.fp8_qdq(&s, x.ptr, t * d).unwrap();
                    let slots = n.min(ne);
                    let elist = DevBuf::alloc(slots * 4).unwrap();
                    let (nei, ci) = (ne as i32, slots as i32);
                    unsafe {
                        k.go(
                            "ds_moe_plan",
                            (1, 1, 1),
                            (32, 1, 1),
                            0,
                            &s,
                            tang_moe::args![counts.ptr, mask.ptr, elist.ptr, nei, ci],
                        )
                        .unwrap();
                    }
                    let mut gu = |exact: bool| -> Vec<f32> {
                        k.exact = exact;
                        let h = DevBuf::from_slice(&vec![0f32; n * inter]).unwrap();
                        k.moe_gu(
                            &s, fmt, w1.ptr, w3.ptr, elist.ptr, off.ptr, rows.ptr, rw.ptr, x.ptr,
                            h.ptr, t, slots, inter, d, 10.0,
                        )
                        .unwrap();
                        s.sync().unwrap();
                        h.read::<f32>(0, n * inter).unwrap()
                    };
                    let (he, hk) = (gu(true), gu(false));
                    // down from the same (exact, quantized) h on both paths
                    let hx = DevBuf::from_slice(&he).unwrap();
                    k.fp8_qdq(&s, hx.ptr, n * inter).unwrap();
                    let mut down = |exact: bool| -> Vec<f32> {
                        k.exact = exact;
                        let part = DevBuf::from_slice(&vec![0f32; n * d]).unwrap();
                        k.moe_down(
                            &s, fmt, w2.ptr, elist.ptr, off.ptr, hx.ptr, part.ptr, t, slots, d,
                            inter, topk,
                        )
                        .unwrap();
                        s.sync().unwrap();
                        part.read::<f32>(0, n * d).unwrap()
                    };
                    let (pe, pk) = (down(true), down(false));
                    k.exact = false;
                    let what = format!("fmt {fmt} t {t} shared {shared}");
                    check_abs(&hk, &he, &format!("{what} gu"));
                    check_abs(&pk, &pe, &format!("{what} down"));
                    assert!(he.iter().any(|&v| v != 0.0) && pe.iter().any(|&v| v != 0.0));
                }
            }
            eprintln!("fmt {fmt}: grouped K-split within the fast-path bar");
        }
    }

    /// One layer's routed MoE by the grouped kernels at the real shapes (384 experts, top-6,
    /// d = 5120, inter = 2304, fp4) with random weights: plan, GU, h quant, down, reduce
    /// (manual: `--ignored --nocapture moe_grouped_bench`). Bytes = the live experts' weights
    /// and scales.
    #[test]
    #[ignore]
    fn moe_grouped_bench() {
        let Some(g) = Gpu::for_test() else { return };
        g.bind().unwrap();
        let mut k = Kernels::new(&g).unwrap();
        let s = Stream::blocking().unwrap();
        let peak: f64 = std::env::var("DS_PEAK_GBS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(936.0);
        let (ne, topk, d, inter) = (384usize, 6usize, 5120usize, 2304usize);
        let rand_bytes = |n: usize, seed: u64| -> Vec<u8> {
            let mut x = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
            (0..n)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    x as u8
                })
                .collect()
        };
        let mut keep = Vec::new();
        let mut mk = |rows: usize, cols: usize, seed: u64| -> (DsW, usize) {
            let wb = rand_bytes(rows * cols / 2, seed);
            let sb: Vec<u8> = rand_bytes(rows * cols / 32, seed + 1)
                .iter()
                .map(|b| 118 + b % 8)
                .collect();
            let (w, sc) = (
                DevBuf::from_slice(&wb).unwrap(),
                DevBuf::from_slice(&sb).unwrap(),
            );
            let v = DsW {
                w: w.ptr,
                s: sc.ptr,
                rows: rows as i32,
                cols: cols as i32,
                fmt: 3,
                _pad: 0,
            };
            keep.push((w, sc));
            (v, wb.len() + sb.len())
        };
        // 36 distinct experts with weights (every t = 6 slot distinct: the worst case)
        let used: Vec<usize> = (0..36).map(|i| (i * 67 + 5) % ne).collect();
        let (mut w1, mut w3, mut w2) = (
            vec![DsW::default(); ne],
            vec![DsW::default(); ne],
            vec![DsW::default(); ne],
        );
        let mut per = 0;
        for (i, &e) in used.iter().enumerate() {
            let a = mk(inter, d, 10 * i as u64 + 1);
            let b = mk(inter, d, 10 * i as u64 + 2);
            let c = mk(d, inter, 10 * i as u64 + 3);
            (w1[e], w3[e], w2[e]) = (a.0, b.0, c.0);
            per = a.1 + b.1 + c.1;
        }
        let (w1, w3, w2) = (
            DevBuf::from_slice(&w1).unwrap(),
            DevBuf::from_slice(&w3).unwrap(),
            DevBuf::from_slice(&w2).unwrap(),
        );
        let mask = DevBuf::from_slice(&vec![1i32; ne]).unwrap();
        eprintln!(
            "\n| t | routing | live experts | exact µs | exact GB/s | K-split µs | K-split GB/s | % of {peak} |\n|---|---|---|---|---|---|---|---|"
        );
        for (t, label) in [(1, "6 experts"), (6, "36 distinct"), (6, "6 shared")] {
            let n = t * topk;
            let ids: Vec<i32> = (0..n)
                .map(|i| {
                    if label == "6 shared" {
                        used[i % topk] as i32
                    } else {
                        used[i] as i32
                    }
                })
                .collect();
            let live = if label == "36 distinct" { n } else { topk };
            let wts = DevBuf::from_slice(&vec![0.2f32; n]).unwrap();
            let ids = DevBuf::from_slice(&ids).unwrap();
            let (counts, off) = (
                DevBuf::alloc(ne * 4).unwrap(),
                DevBuf::alloc((ne + 1) * 4).unwrap(),
            );
            let (rows, rw) = (DevBuf::alloc(n * 4).unwrap(), DevBuf::alloc(n * 4).unwrap());
            k.route(
                &s, ids.ptr, wts.ptr, counts.ptr, off.ptr, rows.ptr, rw.ptr, t, topk, ne,
            )
            .unwrap();
            let x = DevBuf::from_slice(&data(t * d, 3, 2.0)).unwrap();
            k.fp8_qdq(&s, x.ptr, t * d).unwrap();
            let (elist, h, part, y) = (
                DevBuf::alloc(n * 4).unwrap(),
                DevBuf::alloc(n * inter * 4).unwrap(),
                DevBuf::alloc(n * d * 4).unwrap(),
                DevBuf::alloc(t * d * 4).unwrap(),
            );
            let slots = n;
            let mut run = |exact: bool| {
                k.exact = exact;
                let (nei, ci) = (ne as i32, slots as i32);
                let (ti, ki, di) = (t as i32, topk as i32, d as i32);
                unsafe {
                    k.go(
                        "ds_moe_plan",
                        (1, 1, 1),
                        (32, 1, 1),
                        0,
                        &s,
                        tang_moe::args![counts.ptr, mask.ptr, elist.ptr, nei, ci],
                    )
                    .unwrap();
                }
                k.moe_gu(
                    &s, 3, w1.ptr, w3.ptr, elist.ptr, off.ptr, rows.ptr, rw.ptr, x.ptr, h.ptr, t,
                    slots, inter, d, 10.0,
                )
                .unwrap();
                k.fp8_qdq(&s, h.ptr, n * inter).unwrap();
                k.moe_down(
                    &s, 3, w2.ptr, elist.ptr, off.ptr, h.ptr, part.ptr, t, slots, d, inter, topk,
                )
                .unwrap();
                unsafe {
                    k.go(
                        "ds_moe_reduce",
                        blocks(t * d, 256),
                        (256, 1, 1),
                        0,
                        &s,
                        tang_moe::args![
                            y.ptr, part.ptr, ids.ptr, off.ptr, rows.ptr, mask.ptr, ti, ki, di
                        ],
                    )
                    .unwrap();
                }
            };
            let mut time = |exact: bool| {
                run(exact);
                s.sync().unwrap();
                let iters = 20;
                let t0 = std::time::Instant::now();
                for _ in 0..iters {
                    run(exact);
                }
                s.sync().unwrap();
                t0.elapsed().as_secs_f64() / iters as f64
            };
            let (te, tk) = (time(true), time(false));
            let (ge, gk) = (
                (live * per) as f64 / te / 1e9,
                (live * per) as f64 / tk / 1e9,
            );
            eprintln!(
                "| {t} | {label} | {live} | {:.0} | {ge:.0} | {:.0} | {gk:.0} | {:.0}% |",
                te * 1e6,
                tk * 1e6,
                100.0 * gk / peak
            );
        }
    }

    /// Bandwidth on the real model's shapes with random weights (manual: `--ignored
    /// --nocapture fast_linear_bench`). Bytes = weights + scales read once per launch.
    /// The vocab head at the real shape (129,280 x 5120 bf16, t = 1 and the 6-token verify),
    /// CUDA-event timed, plus a bit-identity check of the R-rows-per-warp kernel against
    /// `ds_linear_t8` (`cargo test ... --ignored --nocapture head_bench`).
    #[test]
    #[ignore]
    fn head_bench() {
        let Some(g) = Gpu::for_test() else { return };
        g.bind().unwrap();
        let k = Kernels::new(&g).unwrap();
        let s = Stream::blocking().unwrap();
        let (rows, cols) = (129_280usize, 5120usize);
        let wb: Vec<u16> = data(rows * cols, 7, 0.05)
            .iter()
            .map(|v| (v.to_bits() >> 16) as u16)
            .collect();
        let wd = DevBuf::from_slice(&wb).unwrap();
        let w = DsW {
            w: wd.ptr,
            s: 0,
            rows: rows as i32,
            cols: cols as i32,
            fmt: 1,
            _pad: 0,
        };
        let gb = (rows * cols * 2) as f64 / 1e9;
        let ev = |f: &dyn Fn()| -> f64 {
            f();
            s.sync().unwrap();
            let (a, b) = (
                gpu::Event::new(true).unwrap(),
                gpu::Event::new(true).unwrap(),
            );
            let n = 20;
            a.record(&s).unwrap();
            for _ in 0..n {
                f();
            }
            b.record(&s).unwrap();
            b.sync().unwrap();
            b.since(&a).unwrap() as f64 * 1e3 / n as f64
        };
        for t in [1usize, 2, 6, 8] {
            let x = DevBuf::from_slice(&data(t * cols, 3, 2.0)).unwrap();
            let (y0, y1) = (
                DevBuf::alloc(t * rows * 4).unwrap(),
                DevBuf::alloc(t * rows * 4).unwrap(),
            );
            let base = ev(&|| k.linear(&s, &w, x.ptr, y0.ptr, t, false).unwrap());
            let rr = ev(&|| k.head(&s, &w, x.ptr, y1.ptr, t).unwrap());
            // each row of the t-row head equals that row run alone
            let y2 = DevBuf::alloc(rows * 4).unwrap();
            let mut same = true;
            let all = y1.read::<f32>(0, t * rows).unwrap();
            for i in 0..t {
                k.head(&s, &w, x.ptr + (i * cols * 4) as u64, y2.ptr, 1)
                    .unwrap();
                let one = y2.read::<f32>(0, rows).unwrap();
                same &= one
                    .iter()
                    .zip(&all[i * rows..])
                    .all(|(p, q)| p.to_bits() == q.to_bits());
            }
            let base_v = y0.read::<f32>(0, t * rows).unwrap();
            let md = base_v
                .iter()
                .zip(&all)
                .map(|(p, q)| (p - q).abs())
                .fold(0f32, f32::max);
            eprintln!(
                "head t={t}: linear {base:.0} us ({:.0} GB/s), ds_head_bf16 {rr:.0} us ({:.0} GB/s), rows t-independent {same}, max |diff| vs linear {md:.1e}",
                gb / base * 1e6,
                gb / rr * 1e6
            );
        }
    }

    #[test]
    #[ignore]
    fn fast_linear_bench() {
        let Some(_g) = Gpu::for_test() else { return };
        _g.bind().unwrap();
        let mut k = Kernels::new(&_g).unwrap();
        let s = Stream::blocking().unwrap();
        let peak: f64 = std::env::var("DS_PEAK_GBS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(936.0);
        let rand_bytes = |n: usize, seed: u64| -> Vec<u8> {
            let mut x = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
            (0..n)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    x as u8
                })
                .collect()
        };
        let mk = |fmt: i32, rows: usize, cols: usize, seed: u64| -> (DevBuf, DevBuf, DsW, usize) {
            let mut wb = rand_bytes(
                if fmt == 3 {
                    rows * cols / 2
                } else {
                    rows * cols
                },
                seed,
            );
            if fmt == 2 {
                for b in &mut wb {
                    if *b & 0x7f == 0x7f {
                        *b = 0x10; // no NaN codes
                    }
                }
            }
            let ns = if fmt == 3 {
                rows * cols / 32
            } else {
                rows.div_ceil(32) * cols / 32
            };
            let sb: Vec<u8> = rand_bytes(ns, seed + 1)
                .iter()
                .map(|b| 118 + b % 8)
                .collect();
            let (w, sc) = (
                DevBuf::from_slice(&wb).unwrap(),
                DevBuf::from_slice(&sb).unwrap(),
            );
            let v = DsW {
                w: w.ptr,
                s: sc.ptr,
                rows: rows as i32,
                cols: cols as i32,
                fmt,
                _pad: 0,
            };
            (w, sc, v, wb.len() + sb.len())
        };
        // Host-timed loop of 50 calls: includes launch overhead (~2 µs a launch here, see the
        // rope row), which a CUDA graph mostly removes.
        let time = |f: &dyn Fn()| -> f64 {
            f();
            s.sync().unwrap();
            let n = 50;
            let t0 = std::time::Instant::now();
            for _ in 0..n {
                f();
            }
            s.sync().unwrap();
            t0.elapsed().as_secs_f64() / n as f64
        };
        // Weights cycle through copies totalling > 4× L2 (6 MB on the 3090), so every launch
        // reads DRAM. Bytes = weights + scales.
        let copies = |bytes: usize| (24usize << 20).div_ceil(bytes).max(1);
        {
            // calibration: what a plain copy reaches here (read + write bytes)
            let n = 32usize << 20;
            let (a, b) = (DevBuf::alloc(n * 4).unwrap(), DevBuf::alloc(n * 4).unwrap());
            let nn = n as i64;
            let tc = time(&|| unsafe {
                k.go(
                    "ds_copy",
                    blocks(n, 256),
                    (256, 1, 1),
                    0,
                    &s,
                    tang_moe::args![b.ptr, a.ptr, nn],
                )
                .unwrap()
            });
            eprintln!(
                "calibration: ds_copy 128 MB -> {:.0} GB/s",
                2.0 * (n * 4) as f64 / tc / 1e9
            );
        }
        let kinds: &[(&str, i32, usize, usize, bool)] = &[
            ("wq_a", 2, 1280, 5120, false),
            ("wq_b", 2, 32768, 1280, false),
            ("wkv", 2, 512, 5120, false),
            ("wo_a", 2, 8192, 4096, false),
            ("wo_b", 2, 5120, 8192, false),
            ("expert w1", 3, 2304, 5120, false),
            ("expert w2", 3, 5120, 2304, false),
            ("expert w1+w3", 3, 2304, 5120, true),
        ];
        eprintln!(
            "\n| weight | fmt | [rows, cols] | t | exact µs | exact GB/s | K-split µs | K-split GB/s | % of {peak} |"
        );
        eprintln!("|---|---|---|---|---|---|---|---|---|");
        // t = 1 K-split µs per kind, for the per-token estimate
        let mut t1 = std::collections::HashMap::new();
        for &(label, fmt, rows, cols, gu) in kinds {
            let one = mk(fmt, rows, cols, 1).3 * if gu { 2 } else { 1 };
            let nc = copies(one);
            let ws: Vec<_> = (0..nc * if gu { 2 } else { 1 })
                .map(|i| mk(fmt, rows, cols, rows as u64 + cols as u64 + i as u64))
                .collect();
            for t in [1, 2, 6, 8] {
                let x = DevBuf::from_slice(&data(t * cols, 3, 2.0)).unwrap();
                let y = DevBuf::alloc(t * rows * 4).unwrap();
                let i = std::cell::Cell::new(0usize);
                let mut run = |exact: bool| {
                    k.exact = exact;
                    let f = || {
                        let c = i.get() % nc;
                        i.set(i.get() + 1);
                        let (w, w3) = if gu {
                            (&ws[2 * c].2, Some(&ws[2 * c + 1].2))
                        } else {
                            (&ws[c].2, None)
                        };
                        k.linear_fast(&s, w, w3, x.ptr, y.ptr, t, true, true, 0, 10.0)
                            .unwrap()
                    };
                    time(&f)
                };
                let (te, tk) = (run(true), run(false));
                k.exact = false;
                let (ge, gk) = (one as f64 / te / 1e9, one as f64 / tk / 1e9);
                eprintln!(
                    "| {label} | {} | [{}{rows}, {cols}] | {t} | {:.1} | {ge:.0} | {:.1} | {gk:.0} | {:.0}% |",
                    if fmt == 2 { "fp8" } else { "fp4" },
                    if gu { "2×" } else { "" },
                    te * 1e6,
                    tk * 1e6,
                    100.0 * gk / peak
                );
                if t == 1 {
                    t1.insert(label, (tk, one));
                }
            }
        }

        // The small ops at d = 5120, hc = 4, 64 heads × 512 (t = 1, and 6).
        eprintln!("\n| op | t | µs |\n|---|---|---|");
        let d = 5120;
        let nw = DevBuf::from_slice(&vec![0x3f80u16; d]).unwrap();
        let nv = DsW {
            w: nw.ptr,
            s: 0,
            rows: 1,
            cols: d as i32,
            fmt: 1,
            _pad: 0,
        };
        let tab: Vec<f32> = (0..4096 * 32)
            .flat_map(|i| [(i as f32).cos(), (i as f32).sin()])
            .collect();
        let dtab = DevBuf::from_slice(&tab).unwrap();
        let mut small = 0.0;
        for t in [1, 6] {
            let xb = DevBuf::from_slice(&data(t * 4 * d, 3, 2.0)).unwrap();
            let pre = DevBuf::from_slice(&data(t * 4, 4, 1.0)).unwrap();
            let comb = DevBuf::from_slice(&data(t * 16, 5, 1.0)).unwrap();
            let hb = DevBuf::from_slice(&data(t * d, 6, 1.0)).unwrap();
            let out = DevBuf::alloc(t * 4 * d * 4).unwrap();
            let qb = DevBuf::from_slice(&data(t * 64 * 512, 7, 1.0)).unwrap();
            let (ti, hc, di) = (t as i32, 4i32, d as i32);
            let ops: Vec<(&str, Box<dyn Fn()>)> = vec![
                (
                    "rms [t, 5120]",
                    Box::new(|| k.rms(&s, xb.ptr, &nv, t, 1e-6).unwrap()),
                ),
                (
                    "fp8_qdq [t, 5120]",
                    Box::new(|| k.fp8_qdq(&s, xb.ptr, t * d).unwrap()),
                ),
                (
                    "rope q [t, 64, 512] rd 64",
                    Box::new(|| {
                        k.rope(&s, qb.ptr, dtab.ptr, t, 64, 512, 64, 9, 1, None, false)
                            .unwrap()
                    }),
                ),
                (
                    "hc_pre [t, 4, 5120]",
                    Box::new(|| unsafe {
                        k.go(
                            "ds_hc_pre",
                            blocks(t * d, 256),
                            (256, 1, 1),
                            0,
                            &s,
                            tang_moe::args![xb.ptr, pre.ptr, out.ptr, ti, hc, di],
                        )
                        .unwrap()
                    }),
                ),
                (
                    "hc_post [t, 4, 5120]",
                    Box::new(|| unsafe {
                        k.go(
                            "ds_hc_post",
                            blocks(t * 4 * d, 256),
                            (256, 1, 1),
                            0,
                            &s,
                            tang_moe::args![hb.ptr, xb.ptr, pre.ptr, comb.ptr, out.ptr, ti, hc, di],
                        )
                        .unwrap()
                    }),
                ),
            ];
            for (name, f) in &ops {
                let us = time(f.as_ref()) * 1e6;
                if t == 1 {
                    small += us;
                }
                eprintln!("| {name} | {t} | {us:.1} |");
            }
        }

        // Per-token estimate at the real shapes, t = 1: 40 layers × (dense attention
        // projections + 6 routed + 1 shared expert), the GEMVs at their measured K-split time.
        let l = 40.0;
        let dense = ["wq_a", "wq_b", "wkv", "wo_a", "wo_b"];
        let (mut us, mut bytes) = (0.0, 0.0);
        for n in dense {
            let (tk, b) = t1[n];
            us += tk * 1e6;
            bytes += b as f64;
        }
        for n in ["expert w1+w3", "expert w2"] {
            let (tk, b) = t1[n];
            us += 7.0 * tk * 1e6;
            bytes += 7.0 * b as f64;
        }
        // the bf16 router [384, 5120] isn't on the fast path; count its bytes at peak
        let router = 384.0 * 5120.0 * 2.0;
        let (us_l, by_l) = (us + router / peak / 1e3, bytes + router);
        let gb = l * by_l / 1e9;
        eprintln!(
            "\nper layer: {:.1} MB, {:.1} µs GEMV (+ {small:.1} µs of small ops, launched ~4× a layer)",
            by_l / 1e6,
            us_l
        );
        eprintln!(
            "per token (40 layers, no head/embed/MTP): {gb:.2} GB; GEMV-only measured {:.0} tok/s",
            1e6 / (l * us_l)
        );
        for (hw, bw) in [("3090", peak), ("H200", 4800.0), ("B300", 8000.0)] {
            eprintln!(
                "{hw}: memory-bound ceiling {:.0} tok/s at 100% of {bw} GB/s; {:.0} tok/s at the measured {:.0}%",
                bw / gb,
                bw / gb * (by_l / us_l / 1e3) / peak,
                100.0 * (by_l / us_l / 1e3) / peak
            );
        }

        // attention at the real head shapes
        eprintln!("\n| attention h 64 d 512 | t | k | exact µs | tiled µs | tiled kv GB/s |\n|---|---|---|---|---|---|");
        for (t, kk) in [(1, 640), (6, 640), (5, 640)] {
            let [q, kv, sink, ix] = attn_case(t, kk);
            let o = DevBuf::alloc(t * 64 * 512 * 4).unwrap();
            let mut run = |exact: bool| {
                k.exact = exact;
                time(&|| {
                    k.sparse_attn(
                        &s,
                        q.ptr,
                        kv.ptr,
                        usize::MAX,
                        0,
                        sink.ptr,
                        ix.ptr,
                        o.ptr,
                        t,
                        64,
                        512,
                        kk,
                        0.04,
                    )
                    .unwrap()
                })
            };
            let (te, tt) = (run(true), run(false));
            k.exact = false;
            eprintln!(
                "| | {t} | {kk} | {:.1} | {:.1} | {:.0} |",
                te * 1e6,
                tt * 1e6,
                (t * kk * 512 * 4) as f64 / tt / 1e9
            );
        }
    }

    /// Real-model attention shapes (64 heads × 512, 1 kv head) with `k` gathered keys, -1
    /// holes and a sink; returns (q, kv, sink, idxs) on the device.
    fn attn_case(t: usize, k: usize) -> [DevBuf; 4] {
        let (h, d, n) = (64, 512, 4096);
        let idxs: Vec<i32> = (0..t * k)
            .map(|i| {
                if i % 37 == 5 {
                    -1
                } else {
                    ((i * 7919 + 13) % n) as i32
                }
            })
            .collect();
        [
            DevBuf::from_slice(&data(t * h * d, 31, 2.0)).unwrap(),
            DevBuf::from_slice(&data(n * d, 32, 1.0)).unwrap(),
            DevBuf::from_slice(&data(h, 33, 1.0)).unwrap(),
            DevBuf::from_slice(&idxs).unwrap(),
        ]
    }

    /// The tiled attention against `ds_sparse_attn` (DS_EXACT=1) at the real head shapes.
    #[test]
    fn tiled_attn_matches_exact() {
        let Some(g) = Gpu::for_test() else { return };
        g.bind().unwrap();
        let mut k = Kernels::new(&g).unwrap();
        let s = Stream::blocking().unwrap();
        let (h, d) = (64, 512);
        for (t, kk) in [(1, 640), (6, 640), (5, 100), (2, 960)] {
            let [q, kv, sink, ix] = attn_case(t, kk);
            let o = DevBuf::alloc(t * h * d * 4).unwrap();
            // the tiled run reads its keys split in two (window rows, then the rest)
            let run = |k: &Kernels, split: usize| {
                k.sparse_attn(
                    &s,
                    q.ptr,
                    kv.ptr,
                    split,
                    kv.ptr + (split.min(4096) * d * 4) as u64,
                    sink.ptr,
                    ix.ptr,
                    o.ptr,
                    t,
                    h,
                    d,
                    kk,
                    (d as f32).powf(-0.5),
                )
                .unwrap();
                s.sync().unwrap();
                o.read::<f32>(0, t * h * d).unwrap()
            };
            k.exact = true;
            let want = run(&k, usize::MAX);
            k.exact = false;
            let got = run(&k, 134);
            k.exact = false;
            // Reordered sums flip bf(p) of a few keys by an ulp, and outputs that cancel to
            // near zero then miss by more than an ulp of themselves: ≥ 99% equal, normwise within
            // two ulps of the largest output.
            let (mut eq, mut ulp1) = (0, 0);
            for (a, b) in got.iter().zip(&want) {
                let du = ((a.to_bits() >> 16) as i32 - (b.to_bits() >> 16) as i32).abs();
                eq += (du == 0) as usize;
                ulp1 += (du == 1) as usize;
            }
            let r = rel(&got, &want);
            eprintln!(
                "attn t {t} k {kk}: {:.2}% equal, {:.2}% 1-ulp, normwise rel {r:.2e}",
                100.0 * eq as f64 / got.len() as f64,
                100.0 * ulp1 as f64 / got.len() as f64
            );
            assert!(r < 8e-3, "attn t {t} k {kk}: rel {r}");
            assert!(
                eq as f64 >= 0.9 * got.len() as f64,
                "attn t {t} k {kk}: {eq} equal"
            );
        }
    }

    /// Scores with heavy ties (few levels, ±0, -inf) so tie-breaking is exercised.
    fn tied(n: usize, seed: u64) -> Vec<f32> {
        data(n, seed, 1.0)
            .into_iter()
            .enumerate()
            .map(|(i, x)| match i % 11 {
                0 => f32::NEG_INFINITY,
                1 => -0.0,
                2 => 0.0,
                _ => (x * 4.0).round() / 4.0,
            })
            .collect()
    }

    fn lens(t: usize, width: usize, causal: bool, r: usize) -> Vec<usize> {
        (0..t)
            .map(|ti| if causal { (ti + 1) / r } else { width })
            .collect()
    }

    #[test]
    fn select_kernels_match_cpu() {
        let Some((_g, k, s, m)) = setup() else { return };
        // mask + top-k
        for (t, width, kk, causal, r, off) in [
            (1, 1, 1, false, 1, 0),
            (3, 7, 6, false, 1, 8),
            (16, 8, 6, true, 2, 16),
            (5, 600, 64, true, 1, 3),
            (2, 5000, 512, false, 4, 128),
            (1, 16384, 512, false, 1, 128),
        ] {
            let x = tied(t * width, (width + kk) as u64);
            let ls = lens(t, width, causal, r);
            let mut want_sc = x.clone();
            if causal {
                for ti in 0..t {
                    for j in ls[ti]..width {
                        want_sc[ti * width + j] = f32::NEG_INFINITY;
                    }
                }
            }
            let dx = DevBuf::from_slice(&x).unwrap();
            k.mask_scores(&s, dx.ptr, 0, t, width, causal, r).unwrap();
            let out = DevBuf::alloc(t * kk * 4).unwrap();
            k.topk(&s, dx.ptr, out.ptr, t, width, kk, kk, off, causal, r)
                .unwrap();
            s.sync().unwrap();
            let got_sc = dx.read::<f32>(0, t * width).unwrap();
            assert!(
                got_sc
                    .iter()
                    .zip(&want_sc)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "mask t{t} w{width}"
            );
            let got = out.read::<i32>(0, t * kk).unwrap();
            for ti in 0..t {
                let mut sel = forward::topk_stable(&want_sc[ti * width..(ti + 1) * width], kk);
                sel.sort();
                let want: Vec<i32> = sel
                    .into_iter()
                    .map(|i| if i < ls[ti] { (i + off) as i32 } else { -1 })
                    .collect();
                assert_eq!(
                    &got[ti * kk..(ti + 1) * kk],
                    &want[..],
                    "topk t{t} w{width} row {ti}"
                );
            }
        }
        // candidate blocks, then the candidate mask
        let (bs, kb) = (m.cfg.candidate_block_size, m.cfg.candidate_topk_blocks);
        for (t, width, causal, r) in [
            (16, 16, true, 1),
            (1, 17, false, 1),
            (12, 6, true, 2),
            (1, 3000, false, 1),
        ] {
            let ls = lens(t, width, causal, r);
            let mut x = tied(t * width, width as u64);
            if causal {
                for ti in 0..t {
                    for j in ls[ti]..width {
                        x[ti * width + j] = f32::NEG_INFINITY;
                    }
                }
            }
            let want = m.candidates(&x, width, &ls);
            let dx = DevBuf::from_slice(&x).unwrap();
            let cand = DevBuf::alloc(t * width).unwrap();
            k.candidates(&s, dx.ptr, cand.ptr, t, width, causal, r, bs, kb)
                .unwrap();
            let got = cand.read::<u8>(0, t * width).unwrap();
            let flat: Vec<u8> = want.iter().flatten().map(|&b| b as u8).collect();
            assert_eq!(got, flat, "candidates t{t} w{width}");
            let y = tied(t * width, 99);
            let dy = DevBuf::from_slice(&y).unwrap();
            k.mask_scores(&s, dy.ptr, cand.ptr, t, width, false, r)
                .unwrap();
            s.sync().unwrap();
            let got = dy.read::<f32>(0, t * width).unwrap();
            for (i, (&g, &v)) in got.iter().zip(&y).enumerate() {
                let w = if flat[i] == 1 { v } else { f32::NEG_INFINITY };
                assert_eq!(g.to_bits(), w.to_bits(), "cand mask {i}");
            }
        }
        // routing lists
        for (t, kk, ne) in [(1, 2, 8), (37, 6, 64), (300, 8, 256)] {
            let mut ids = Vec::new();
            let mut wts = Vec::new();
            for ti in 0..t {
                // distinct per token: 3 is coprime to every ne here
                let used: Vec<usize> = (0..kk).map(|j| (ti * 7 + j * 3) % ne).collect();
                for (j, &e) in used.iter().enumerate() {
                    ids.push(e as i32);
                    wts.push((ti * 10 + j) as f32);
                }
            }
            let (di, dw) = (
                DevBuf::from_slice(&ids).unwrap(),
                DevBuf::from_slice(&wts).unwrap(),
            );
            let counts = DevBuf::alloc(ne * 4).unwrap();
            let off = DevBuf::alloc((ne + 1) * 4).unwrap();
            let rows = DevBuf::alloc(t * kk * 4).unwrap();
            let rw = DevBuf::alloc(t * kk * 4).unwrap();
            k.route(
                &s, di.ptr, dw.ptr, counts.ptr, off.ptr, rows.ptr, rw.ptr, t, kk, ne,
            )
            .unwrap();
            s.sync().unwrap();
            let mut routes: Vec<Vec<(i32, f32)>> = vec![Vec::new(); ne];
            for ti in 0..t {
                for j in 0..kk {
                    routes[ids[ti * kk + j] as usize].push((ti as i32, wts[ti * kk + j]));
                }
            }
            let (gc, go, gr, gw) = (
                counts.read::<i32>(0, ne).unwrap(),
                off.read::<i32>(0, ne + 1).unwrap(),
                rows.read::<i32>(0, t * kk).unwrap(),
                rw.read::<f32>(0, t * kk).unwrap(),
            );
            for e in 0..ne {
                assert_eq!(gc[e] as usize, routes[e].len());
                let o = go[e] as usize;
                assert_eq!(go[e + 1] as usize, o + routes[e].len());
                for (q, &(r, w)) in routes[e].iter().enumerate() {
                    assert_eq!((gr[o + q], gw[o + q]), (r, w), "route e{e}");
                }
            }
        }
    }

    /// The MoE router on the real shape (manual: `--ignored --nocapture router_bench`):
    /// bf16 gate [384, 5120] GEMV, `ds_router`, `route`, each timed alone.
    #[test]
    #[ignore]
    fn router_bench() {
        let Some(g) = Gpu::for_test() else { return };
        g.bind().unwrap();
        let k = Kernels::new(&g).unwrap();
        let s = Stream::blocking().unwrap();
        let (ne, topk, d, t) = (384usize, 6usize, 5120usize, 6usize);
        let wb: Vec<u16> = data(ne * d, 5, 0.05)
            .iter()
            .map(|v| (v.to_bits() >> 16) as u16)
            .collect();
        let wdev = DevBuf::from_slice(&wb).unwrap();
        let w = DsW {
            w: wdev.ptr,
            s: 0,
            rows: ne as i32,
            cols: d as i32,
            fmt: 1,
            _pad: 0,
        };
        let x = DevBuf::from_slice(&data(t * d, 7, 1.0)).unwrap();
        let bias = DevBuf::from_slice(&data(ne, 9, 0.1)).unwrap();
        let raw = DevBuf::alloc(t * ne * 4).unwrap();
        let (ids, wts) = (
            DevBuf::alloc(t * topk * 4).unwrap(),
            DevBuf::alloc(t * topk * 4).unwrap(),
        );
        let (counts, off) = (
            DevBuf::alloc(ne * 4).unwrap(),
            DevBuf::alloc((ne + 1) * 4).unwrap(),
        );
        let (rows, rw) = (
            DevBuf::alloc(t * topk * 4).unwrap(),
            DevBuf::alloc(t * topk * 4).unwrap(),
        );
        let time = |name: &str, f: &dyn Fn()| {
            f();
            s.sync().unwrap();
            let iters = 200;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                f();
            }
            s.sync().unwrap();
            let us = t0.elapsed().as_secs_f64() / iters as f64 * 1e6;
            eprintln!("router_bench {name}: {us:.1} us");
        };
        time("gate", &|| {
            k.linear(&s, &w, x.ptr, raw.ptr, t, false).unwrap();
        });
        time("ds_router", &|| {
            k.router(
                &s, raw.ptr, bias.ptr, ids.ptr, wts.ptr, t, ne, topk, true, 1.5,
            )
            .unwrap();
        });
        time("route", &|| {
            k.route(
                &s, ids.ptr, wts.ptr, counts.ptr, off.ptr, rows.ptr, rw.ptr, t, topk, ne,
            )
            .unwrap();
        });
    }

    /// `ds_router` against the CPU router math on exact ties (coarse raw values shared by
    /// many experts, so the order rests on the lower-id rule) and on random logits.
    #[test]
    fn router_matches_cpu() {
        let Some(g) = Gpu::for_test() else { return };
        g.bind().unwrap();
        let k = Kernels::new(&g).unwrap();
        let s = Stream::blocking().unwrap();
        for (t, ne, kk, norm, coarse) in [
            (1, 8, 1, true, false),
            (6, 384, 6, true, true),
            (6, 384, 6, true, false),
            (5, 256, 8, false, true),
            (40, 1000, 16, true, true),
        ] {
            let mut raw = data(t * ne, (ne + kk) as u64, 3.0);
            let mut bias = data(ne, 11, 0.5);
            if coarse {
                raw.iter_mut().for_each(|v| *v = (*v * 2.0).round() / 2.0);
                bias.iter_mut().for_each(|v| *v = (*v * 4.0).round() / 4.0);
            }
            let (dr, db) = (
                DevBuf::from_slice(&raw).unwrap(),
                DevBuf::from_slice(&bias).unwrap(),
            );
            let (ids, wts) = (
                DevBuf::alloc(t * kk * 4).unwrap(),
                DevBuf::alloc(t * kk * 4).unwrap(),
            );
            k.router(&s, dr.ptr, db.ptr, ids.ptr, wts.ptr, t, ne, kk, norm, 1.5)
                .unwrap();
            s.sync().unwrap();
            let (gi, gw) = (
                ids.read::<i32>(0, t * kk).unwrap(),
                wts.read::<f32>(0, t * kk).unwrap(),
            );
            for ti in 0..t {
                let row: Vec<f32> = raw[ti * ne..(ti + 1) * ne]
                    .iter()
                    .map(|&x| forward::softplus(x).sqrt())
                    .collect();
                let biased: Vec<f32> = row.iter().zip(&bias).map(|(a, b)| a + b).collect();
                let sel = forward::topk_stable(&biased, kk);
                let want: Vec<i32> = sel.iter().map(|&e| e as i32).collect();
                assert_eq!(
                    &gi[ti * kk..(ti + 1) * kk],
                    &want[..],
                    "ids t{t} ne{ne} row {ti}"
                );
                let mut w: Vec<f32> = sel.iter().map(|&e| row[e]).collect();
                if norm && kk > 1 {
                    let sm: f32 = w.iter().sum::<f32>() + 1e-20;
                    w.iter_mut().for_each(|v| *v /= sm);
                }
                for (j, &wv) in w.iter().enumerate() {
                    let gv = gw[ti * kk + j];
                    assert!(
                        (gv - wv * 1.5).abs() <= 1e-5 * (1.0 + gv.abs()),
                        "wts row {ti}"
                    );
                }
            }
        }
    }

    /// The attention block of a verify step at the real shapes (t 6, 64 heads × 512, window
    /// 128 + top-k 512, dim 5120, pos ~2000): wo_a, sparse attention (the decode kernel and
    /// the tiled one), the indexer's score + top-k. `-- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn attn_block_bench() {
        let Some(g) = Gpu::for_test() else { return };
        g.bind().unwrap();
        let mut k = Kernels::new(&g).unwrap();
        let gk = super::super::graph::GKernels::new(&g).unwrap();
        let s = Stream::blocking().unwrap();
        let ev = |f: &dyn Fn()| -> f64 {
            f();
            s.sync().unwrap();
            let n = 50;
            let (a, b) = (
                gpu::Event::new(true).unwrap(),
                gpu::Event::new(true).unwrap(),
            );
            a.record(&s).unwrap();
            for _ in 0..n {
                f();
            }
            b.record(&s).unwrap();
            s.sync().unwrap();
            b.since(&a).unwrap() as f64 * 1e3 / n as f64
        };
        let (t, h, d, g8, ol) = (6usize, 64usize, 512usize, 8usize, 1024usize);
        // wo_a: fp8 [8 × 1024, 4096]
        let (rows, cols) = (g8 * ol, h * d / g8);
        let mut wb: Vec<u8> = (0..rows * cols)
            .map(|i| (i * 2654435761 >> 13) as u8)
            .collect();
        wb.iter_mut()
            .filter(|b| **b & 0x7f == 0x7f)
            .for_each(|b| *b = 0x10);
        let sb: Vec<u8> = (0..rows.div_ceil(32) * cols / 32)
            .map(|i| 118 + (i % 8) as u8)
            .collect();
        let (w, sc) = (
            DevBuf::from_slice(&wb).unwrap(),
            DevBuf::from_slice(&sb).unwrap(),
        );
        let wv = DsW {
            w: w.ptr,
            s: sc.ptr,
            rows: rows as i32,
            cols: cols as i32,
            fmt: 2,
            _pad: 0,
        };
        let o = DevBuf::from_slice(&data(t * h * d, 3, 1.0)).unwrap();
        let mid = DevBuf::alloc(t * rows * 4).unwrap();
        let (ti, gi, oli) = (t as i32, g8 as i32, ol as i32);
        let us = ev(&|| unsafe {
            k.go(
                "ds_wo_a",
                ((g8 * ol / 16) as u32, 1, 1),
                (256, 1, 1),
                0,
                &s,
                tang_moe::args![o.ptr, wv, mid.ptr, ti, gi, oli],
            )
            .unwrap()
        });
        eprintln!("attn_block_bench wo_a exact t {t}: {us:.1} us");
        // the K-split path as engine.rs `wo_a`: group-major copy, a GEMV a group, back
        let og = DevBuf::alloc(t * h * d * 4).unwrap();
        let yg = DevBuf::alloc(t * rows * 4).unwrap();
        let perm = |src: u64, dst: u64, a: usize, b: usize, c: usize| unsafe {
            let (ai, bi, ci) = (a as i32, b as i32, c as i32);
            k.go(
                "ds_perm3",
                blocks(a * b * c, 256),
                (256, 1, 1),
                0,
                &s,
                tang_moe::args![src, dst, ai, bi, ci],
            )
            .unwrap()
        };
        let us = ev(&|| {
            perm(o.ptr, og.ptr, t, g8, cols);
            for gi in 0..g8 {
                let mut v = wv;
                v.rows = ol as i32;
                v.w += (gi * ol * cols) as u64;
                v.s += ((gi * ol / 32) * (cols / 32)) as u64;
                k.linear_fast(
                    &s,
                    &v,
                    None,
                    og.ptr + (gi * t * cols * 4) as u64,
                    yg.ptr + (gi * t * ol * 4) as u64,
                    t,
                    true,
                    false,
                    0,
                    0.0,
                )
                .unwrap();
            }
            perm(yg.ptr, mid.ptr, g8, t, ol);
        });
        eprintln!("attn_block_bench wo_a K-split t {t}: {us:.1} us");
        // sparse attention: window 128 + 6 chunk rows, then 512 compressed rows
        let (win, kc) = (128 + t, 512usize);
        let kk = win + kc;
        let q = DevBuf::from_slice(&data(t * h * d, 5, 1.0)).unwrap();
        let kv0 = DevBuf::from_slice(&data(win * d, 6, 1.0)).unwrap();
        let kv1 = DevBuf::from_slice(&data(2048 * d, 7, 1.0)).unwrap();
        let sink = DevBuf::from_slice(&data(h, 8, 1.0)).unwrap();
        let ix: Vec<i32> = (0..t * kk)
            .map(|i| {
                let j = i % kk;
                if j < win {
                    j as i32
                } else {
                    (win + (j * 7 + i / kk) % 500) as i32
                }
            })
            .collect();
        let ix = DevBuf::from_slice(&ix).unwrap();
        let ob = DevBuf::alloc(t * h * d * 4).unwrap();
        let (hi, di, ki, split, scale) = (h as i32, d as i32, kk as i32, win as i32, 0.04f32);
        let us = ev(&|| unsafe {
            gk.go(
                "dg_sparse_attn",
                ((t * h) as u32, 1, 1),
                (512, 1, 1),
                (kk * 4) as u32,
                &s,
                tang_moe::args![
                    q.ptr, kv0.ptr, split, kv1.ptr, sink.ptr, ix.ptr, ob.ptr, hi, di, ki, scale
                ],
            )
            .unwrap()
        });
        eprintln!("attn_block_bench dg_sparse_attn t {t} k {kk}: {us:.1} us");
        k.exact = false;
        let us = ev(&|| {
            k.sparse_attn(
                &s, q.ptr, kv0.ptr, win, kv1.ptr, sink.ptr, ix.ptr, ob.ptr, t, h, d, kk, scale,
            )
            .unwrap()
        });
        eprintln!("attn_block_bench tiled sparse_attn t {t} k {kk}: {us:.1} us");
        // indexer at pos 2000, ratio 4: 64 heads × 128, width 500, top-512
        let (nh, ihd, r, wmax, kmax) = (64usize, 128usize, 4usize, 4096usize, 512usize);
        let qi = DevBuf::from_slice(&data(t * nh * ihd, 9, 1.0)).unwrap();
        let ik = DevBuf::from_slice(&data(wmax * ihd, 10, 1.0)).unwrap();
        let wts = DevBuf::from_slice(&data(t * nh, 11, 1.0)).unwrap();
        let score = DevBuf::alloc(t * wmax * 4).unwrap();
        let sel = DevBuf::alloc(t * kmax * 4).unwrap();
        let out = DevBuf::alloc(t * kmax * 4).unwrap();
        let pp = DevBuf::from_slice(&[2000i32]).unwrap();
        let (ri, nhi, ihdi, kmi, offi) = (r as i32, nh as i32, ihd as i32, kmax as i32, win as i32);
        let wi = wmax as i32;
        let us = ev(&|| unsafe {
            let sg = blocks(wmax * 32, 256);
            gk.go(
                "dg_index_score",
                (sg.0, t as u32, 1),
                (256, 1, 1),
                0,
                &s,
                tang_moe::args![
                    qi.ptr, ik.ptr, wts.ptr, score.ptr, pp.ptr, 0i32, ri, nhi, ihdi, wi
                ],
            )
            .unwrap();
            gk.go(
                "dg_topk",
                (t as u32, 1, 1),
                super::super::graph::SEL,
                0,
                &s,
                tang_moe::args![score.ptr, sel.ptr, out.ptr, pp.ptr, 0i32, ri, kmi, offi, wi],
            )
            .unwrap();
        });
        eprintln!("attn_block_bench indexer score+topk t {t} batched (per layer): {us:.1} us");
    }

    /// `ds_wo_a`'s fp8 fast path against its generic (wget) path on the same weights
    /// dequantized to f32: identical sums, so bit-equal. t 9 crosses the 8-token pass.
    #[test]
    fn wo_a_fp8_matches_f32() {
        let Some(g) = Gpu::for_test() else { return };
        g.bind().unwrap();
        let k = Kernels::new(&g).unwrap();
        let s = Stream::blocking().unwrap();
        let (t, gn, ol, gd) = (9usize, 2usize, 48usize, 96usize);
        let rows = gn * ol;
        let mut wb: Vec<u8> = (0..rows * gd)
            .map(|i| (i * 2654435761 >> 11) as u8)
            .collect();
        wb.iter_mut()
            .filter(|b| **b & 0x7f == 0x7f)
            .for_each(|b| *b = 0x08);
        let nsc = gd.div_ceil(32);
        let sb: Vec<u8> = (0..rows.div_ceil(32) * nsc)
            .map(|i| 120 + (i % 9) as u8)
            .collect();
        let wf: Vec<f32> = (0..rows * gd)
            .map(|i| {
                let (r, c) = (i / gd, i % gd);
                super::super::quant::e4m3_to_f32(wb[i])
                    * super::super::quant::e8m0_to_f32(sb[(r / 32) * nsc + c / 32])
            })
            .collect();
        let (dw, ds, df) = (
            DevBuf::from_slice(&wb).unwrap(),
            DevBuf::from_slice(&sb).unwrap(),
            DevBuf::from_slice(&wf).unwrap(),
        );
        let o = DevBuf::from_slice(&data(t * gn * gd, 3, 2.0)).unwrap();
        let run = |w: DsW| {
            let mid = DevBuf::alloc(t * rows * 4).unwrap();
            let (ti, gi, oli) = (t as i32, gn as i32, ol as i32);
            unsafe {
                k.go(
                    "ds_wo_a",
                    ((gn * ol / 16) as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    &s,
                    tang_moe::args![o.ptr, w, mid.ptr, ti, gi, oli],
                )
                .unwrap()
            };
            s.sync().unwrap();
            mid.read::<f32>(0, t * rows).unwrap()
        };
        let (rw, cw) = (rows as i32, gd as i32);
        let got = run(DsW {
            w: dw.ptr,
            s: ds.ptr,
            rows: rw,
            cols: cw,
            fmt: 2,
            _pad: 0,
        });
        let want = run(DsW {
            w: df.ptr,
            s: 0,
            rows: rw,
            cols: cw,
            fmt: 0,
            _pad: 0,
        });
        // a CPU check of the sum too (any order: within an ulp or so of bf16)
        for ti in 0..t {
            for gi in 0..gn {
                for r in 0..ol {
                    let row = gi * ol + r;
                    let ov =
                        &data(t * gn * gd, 3, 2.0)[(ti * gn + gi) * gd..(ti * gn + gi + 1) * gd];
                    let cpu: f32 = ov
                        .iter()
                        .zip(&wf[row * gd..(row + 1) * gd])
                        .map(|(a, b)| a * bf(*b))
                        .sum();
                    let gv = got[(ti * gn + gi) * ol + r];
                    assert!(
                        (gv - cpu).abs() <= 2e-2 * (1.0 + cpu.abs()),
                        "t {ti} g {gi} r {r}: {gv} vs {cpu}"
                    );
                }
            }
        }
        assert!(
            got.iter()
                .zip(&want)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "fp8 path != f32 path"
        );
    }
}
