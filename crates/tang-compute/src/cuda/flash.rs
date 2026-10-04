//! The Flash-Next decode ops on CUDA (kernels in `kernels::flash_cuda`, contracts in
//! `crate::flash`). Every launch here writes into caller buffers and takes per-window values
//! from the `win` record, so a window of them can be captured into one CUDA graph.

use cudarc::driver::{CudaFunction, DevicePtr, LaunchConfig, PushKernelArg};

use super::{CudaBuffer, CudaComputeDevice, CudaStorage, Q2Weight};
use crate::flash::shape::*;
use crate::flash::{
    GdnMode, GdnParams, HcPending, HcWeights, MoePlan, QAct, QsaCache, QsaNorms, MAX_T,
};
use crate::kernels::flash_cuda::FLASH_CUDA;

/// A null device pointer for kernel arguments a mode never reads.
const NULL: u64 = 0;

/// Words of the per-device sync counters, and each fused kernel's slot in them.
const SYNC_WORDS: usize = 256;
const SYNC_HC: usize = 0;

/// `TANG_FLASH_UNFUSED=1`: the multi-launch kernels instead of the barrier-merged ones (A/B).
fn unfused() -> bool {
    static U: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *U.get_or_init(|| std::env::var("TANG_FLASH_UNFUSED").is_ok_and(|v| v == "1"))
}

fn grid(g: (usize, usize, usize), threads: u32) -> LaunchConfig {
    LaunchConfig {
        block_dim: (threads, 1, 1),
        grid_dim: (g.0 as u32, g.1 as u32, g.2 as u32),
        shared_mem_bytes: 0,
    }
}

const GEMV_FMTS: [&str; 4] = ["bf16_gemv", "q2_gemv", "q4x_gemv", "q8x_gemv"];
const GEMV_NAMES: [[&str; 8]; 4] = [
    [
        "fl_bf16_gemv_t1",
        "fl_bf16_gemv_t2",
        "fl_bf16_gemv_t3",
        "fl_bf16_gemv_t4",
        "fl_bf16_gemv_t5",
        "fl_bf16_gemv_t6",
        "fl_bf16_gemv_t7",
        "fl_bf16_gemv_t8",
    ],
    [
        "fl_q2_gemv_t1",
        "fl_q2_gemv_t2",
        "fl_q2_gemv_t3",
        "fl_q2_gemv_t4",
        "fl_q2_gemv_t5",
        "fl_q2_gemv_t6",
        "fl_q2_gemv_t7",
        "fl_q2_gemv_t8",
    ],
    [
        "fl_q4x_gemv_t1",
        "fl_q4x_gemv_t2",
        "fl_q4x_gemv_t3",
        "fl_q4x_gemv_t4",
        "fl_q4x_gemv_t5",
        "fl_q4x_gemv_t6",
        "fl_q4x_gemv_t7",
        "fl_q4x_gemv_t8",
    ],
    [
        "fl_q8x_gemv_t1",
        "fl_q8x_gemv_t2",
        "fl_q8x_gemv_t3",
        "fl_q8x_gemv_t4",
        "fl_q8x_gemv_t5",
        "fl_q8x_gemv_t6",
        "fl_q8x_gemv_t7",
        "fl_q8x_gemv_t8",
    ],
];

impl CudaComputeDevice {
    /// Kernel `name` of `FLASH_CUDA` (compiled once for sm_86, as the tensor-core kernels are:
    /// `__dp4a` needs an explicit architecture).
    fn fl(&self, name: &'static str) -> CudaFunction {
        if let Some(f) = self.llm_funcs.borrow().get(name) {
            return f.clone();
        }
        let (_module, f) = self.get_func_with_arch(FLASH_CUDA, name, "sm_86");
        self.llm_funcs.borrow_mut().insert(name, f.clone());
        f
    }

    pub(super) fn upload_bytes_impl(&self, bytes: &[u8]) -> CudaBuffer {
        self.upload_f32(&crate::flash::words_of(&crate::flash::bytes_to_words(
            bytes,
        )))
    }

    pub(super) fn alloc_bf16_impl(&self, len: usize) -> CudaBuffer {
        let slice = self.stream.alloc_zeros::<u16>(len).unwrap();
        Self::make_buf_unpooled(CudaStorage::Bf16(slice), len)
    }

    pub(super) fn buffer_addr_impl(&self, buf: &CudaBuffer) -> u64 {
        let (p, _sync) = match buf.storage() {
            CudaStorage::F32(s) => s.device_ptr(&self.stream),
            CudaStorage::Bf16(s) => s.device_ptr(&self.stream),
            CudaStorage::Q2(q) => q.data.device_ptr(&self.stream),
            CudaStorage::Q4(q) => q.packed.device_ptr(&self.stream),
        };
        p
    }

    pub(super) fn upload_q2_impl(&self, raw: &[u8], n: usize, k: usize) -> CudaBuffer {
        let bytes = crate::flash::q2_repack(raw, n, k);
        let len = bytes.len().div_ceil(4);
        let data = self.stream.memcpy_stod(&bytes).unwrap();
        Self::make_buf_unpooled(CudaStorage::Q2(Box::new(Q2Weight { data, n, k })), len)
    }

    /// `linear` into `out` for `m <= 8` with the bf16 / 4-bit GEMVs; anything else allocates
    /// through `linear` and copies.
    pub(super) fn linear_into_impl(
        &self,
        x: &CudaBuffer,
        w: &CudaBuffer,
        out: &mut CudaBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        use crate::ComputeDevice;
        let direct = m <= 8 && k.is_multiple_of(8) && !x.is_bf16() && !out.is_bf16();
        let (mu, ku, nu) = (m as u32, k as u32, n as u32);
        let cfg = grid((n.div_ceil(8), 1, 1), 256);
        match w.storage() {
            CudaStorage::Q4(q) if direct => {
                let f = self.llm_func(crate::kernels::llm_cuda::GEMV_CUDA, "gemv_q4");
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
            CudaStorage::Bf16(s) if direct && k.is_multiple_of(8) && m >= 1 => {
                self.gemv_rows("bf16_gemv", Some(x), None, s, out, m, k, n);
            }
            _ => {
                let y = self.linear(x, w, m, k, n);
                self.write_into(out, 0, &y);
            }
        }
    }

    pub(super) fn quantize_act_impl(
        &self,
        x: &CudaBuffer,
        xq: &mut CudaBuffer,
        m: usize,
        k: usize,
    ) {
        assert!(k.is_multiple_of(32) && xq.len >= QAct { m, k }.words());
        let f = self.fl("fl_quantize");
        let (mu, ku) = (m as u32, k as u32);
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(x.f32_data())
                .arg(xq.f32_data_mut())
                .arg(&mu)
                .arg(&ku)
                .launch(grid((k.div_ceil(256), m, 1), 256))
                .unwrap();
        }
    }

    /// Launch `fl_{name}_t{m}` (the multi-column GEMV) for `[n, k]` weights at `w`.
    #[allow(clippy::too_many_arguments)]
    fn gemv_rows(
        &self,
        name: &str,
        x: Option<&CudaBuffer>,
        xq: Option<&CudaBuffer>,
        w: &cudarc::driver::CudaSlice<impl cudarc::driver::DeviceRepr>,
        out: &mut CudaBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        assert!((1..=MAX_T).contains(&m) && out.len >= m * n);
        // Split K over 1..8 warps until there are ~2 blocks per SM, keeping 32+ weight vectors
        // per warp (4 rows a warp, 8 / ks row groups a 256-thread block).
        let epv = match name {
            "bf16_gemv" => 8,
            "q2_gemv" => 64,
            "q8x_gemv" => 16,
            _ => 32,
        };
        let mut ks = 1;
        while ks < 8
            && n.div_ceil(4 * 8 / ks) < 2 * super::llm::sm_count()
            && k / epv / (2 * ks) >= 32
        {
            ks *= 2;
        }
        let f = self.fl(GEMV_NAMES[GEMV_FMTS.iter().position(|&f| f == name).unwrap()][m - 1]);
        let (ku, nu, ksu) = (k as u32, n as u32, ks as u32);
        let any = xq.or(x).unwrap();
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(x.unwrap_or(any).f32_data())
                .arg(xq.unwrap_or(any).f32_data())
                .arg(w)
                .arg(out.f32_data_mut())
                .arg(&ku)
                .arg(&nu)
                .arg(&ksu)
                .launch(grid((n.div_ceil(4 * 8 / ks), 1, 1), 256))
                .unwrap();
        }
    }

    pub(super) fn q2_linear_impl(
        &self,
        xq: &CudaBuffer,
        w: &CudaBuffer,
        out: &mut CudaBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        let CudaStorage::Q2(q) = w.storage() else {
            panic!("q2_linear_into needs a Q2 weight (upload_q2)")
        };
        assert!(q.n == n && q.k == k);
        self.gemv_rows("q2_gemv", None, Some(xq), &q.data, out, m, k, n);
    }

    pub(super) fn q4x_linear_impl(
        &self,
        xq: &CudaBuffer,
        w: &CudaBuffer,
        out: &mut CudaBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        assert!(
            w.len * 4 >= n * k / 2 + 4 * n * k / 64,
            "q4x_linear_into: weight size"
        );
        self.gemv_rows("q4x_gemv", None, Some(xq), w.f32_data(), out, m, k, n);
    }

    pub(super) fn q8x_linear_impl(
        &self,
        xq: &CudaBuffer,
        w: &CudaBuffer,
        out: &mut CudaBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        assert!(
            w.len * 4 >= n * k + 2 * n * k / 32,
            "q8x_linear_into: weight size"
        );
        self.gemv_rows("q8x_gemv", None, Some(xq), w.f32_data(), out, m, k, n);
    }

    pub(super) fn hc_write_impl(
        &self,
        r: &mut CudaBuffer,
        y: &CudaBuffer,
        inj: &CudaBuffer,
        t: usize,
    ) {
        let f = self.fl("fl_hc_write");
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(r.f32_data_mut())
                .arg(y.f32_data())
                .arg(inj.f32_data())
                .launch(grid((HIDDEN / 256, HC, t), 256))
                .unwrap();
        }
    }

    /// The self-resetting counters the fused kernels synchronize through (zeroed once).
    fn sync_words(&self) -> std::cell::RefMut<'_, cudarc::driver::CudaSlice<u32>> {
        let mut s = self.flash_sync.borrow_mut();
        if s.is_none() {
            *s = Some(self.stream.alloc_zeros::<u32>(SYNC_WORDS).unwrap());
        }
        std::cell::RefMut::map(s, |s| s.as_mut().unwrap())
    }

    /// Blocks for a grid that must be co-resident (it meets at `grid_bar`): up to `per_sm`
    /// per SM, as the occupancy calculator allows.
    fn coresident(&self, f: &CudaFunction, threads: u32, per_sm: u32) -> usize {
        let occ = f
            .occupancy_max_active_blocks_per_multiprocessor(threads, 0, None)
            .unwrap();
        assert!(occ >= 1, "fused kernel does not fit on an SM");
        occ.min(per_sm) as usize * super::llm::sm_count()
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn hc_read_impl(
        &self,
        r: &mut CudaBuffer,
        pending: Option<HcPending<'_, CudaBuffer>>,
        w: &HcWeights<'_, CudaBuffer>,
        x: &mut CudaBuffer,
        xq: Option<&mut CudaBuffer>,
        inj: Option<&mut CudaBuffer>,
        scratch: &mut CudaBuffer,
        t: usize,
        eps: f32,
    ) {
        assert!((1..=MAX_T).contains(&t) && scratch.len >= crate::flash::hc_scratch_words(t));
        if !unfused() {
            return self.hc_fused_impl(r, pending, w, x, xq, inj, scratch, t, eps);
        }
        let xn_len = t * HC * HIDDEN;
        let (mut xn, mut lo) = scratch.f32_data_mut().split_at_mut(xn_len);
        // Unused pointer arguments get `w.norm`; the kernel never reads them in that mode.
        let any = w.norm;
        let (mode, yp, ip, parts, wr, lg, stride, sg) = match pending {
            None => (0u32, any, any, any, any, any, 0usize, -1i32),
            Some(HcPending::Write { y, inj }) => (1, y, inj, any, any, any, 0, -1),
            Some(HcPending::Moe {
                parts,
                w: wr,
                logits,
                stride,
                sg,
                inj,
            }) => (
                2,
                any,
                inj,
                parts,
                wr,
                logits,
                stride,
                sg.map_or(-1, |c| c as i32),
            ),
        };
        let su = stride as u32;
        let f = self.fl("fl_hc_norm");
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(r.f32_data_mut())
                .arg(yp.f32_data())
                .arg(ip.f32_data())
                .arg(&mode)
                .arg(w.norm.f32_data())
                .arg(&mut xn)
                .arg(&eps)
                .arg(parts.f32_data())
                .arg(wr.f32_data())
                .arg(lg.f32_data())
                .arg(&su)
                .arg(&sg)
                .launch(grid((HC, t, 1), 256))
                .unwrap();
        }
        let tu = t as u32;
        let f = self.fl("fl_hc_down");
        let rows = HC_LR + if w.inject.is_some() { HC } else { 0 };
        let wi = w.inject.unwrap_or(w.down);
        unsafe {
            let mut l = self.stream.launch_builder(&f);
            l.arg(&mut xn)
                .arg(w.down.bf16_data())
                .arg(wi.bf16_data())
                .arg(&mut lo);
            // Without injection rows the kernel never writes INJ; any pointer will do.
            match inj {
                Some(i) => l.arg(i.f32_data_mut()),
                None => l.arg(r.f32_data_mut()),
            };
            let ru = rows as u32;
            l.arg(&tu)
                .arg(&ru)
                .launch(grid((rows.div_ceil(4), 1, 1), 256))
                .unwrap();
        }
        const UP: [&str; 8] = [
            "fl_hc_up_t1",
            "fl_hc_up_t2",
            "fl_hc_up_t3",
            "fl_hc_up_t4",
            "fl_hc_up_t5",
            "fl_hc_up_t6",
            "fl_hc_up_t7",
            "fl_hc_up_t8",
        ];
        let f = self.fl(UP[t - 1]);
        let quant = xq.is_some() as u32;
        unsafe {
            let mut l = self.stream.launch_builder(&f);
            l.arg(&mut xn).arg(&mut lo).arg(w.up.bf16_data());
            match xq {
                Some(q) => {
                    assert!(q.len >= QAct { m: t, k: HIDDEN }.words());
                    l.arg(x.f32_data_mut()).arg(q.f32_data_mut())
                }
                None => l.arg(x.f32_data_mut()).arg(&NULL),
            };
            l.arg(&quant)
                .launch(grid((HIDDEN / 32, 1, 1), 1024))
                .unwrap();
        }
    }

    /// `hc_read_impl` as one launch of `fl_hc_fused_t{t}` on a co-resident grid.
    #[allow(clippy::too_many_arguments)]
    fn hc_fused_impl(
        &self,
        r: &mut CudaBuffer,
        pending: Option<HcPending<'_, CudaBuffer>>,
        w: &HcWeights<'_, CudaBuffer>,
        x: &mut CudaBuffer,
        xq: Option<&mut CudaBuffer>,
        inj: Option<&mut CudaBuffer>,
        scratch: &mut CudaBuffer,
        t: usize,
        eps: f32,
    ) {
        const NAMES: [&str; 8] = [
            "fl_hc_fused_t1",
            "fl_hc_fused_t2",
            "fl_hc_fused_t3",
            "fl_hc_fused_t4",
            "fl_hc_fused_t5",
            "fl_hc_fused_t6",
            "fl_hc_fused_t7",
            "fl_hc_fused_t8",
        ];
        let f = self.fl(NAMES[t - 1]);
        let blocks = self.coresident(&f, 512, 2);
        let (mut xn, mut lo) = scratch.f32_data_mut().split_at_mut(t * HC * HIDDEN);
        let any = w.norm;
        let (mode, yp, ip, parts, wr, lg, stride, sg) = match pending {
            None => (0u32, any, any, any, any, any, 0usize, -1i32),
            Some(HcPending::Write { y, inj }) => (1, y, inj, any, any, any, 0, -1),
            Some(HcPending::Moe {
                parts,
                w: wr,
                logits,
                stride,
                sg,
                inj,
            }) => (
                2,
                any,
                inj,
                parts,
                wr,
                logits,
                stride,
                sg.map_or(-1, |c| c as i32),
            ),
        };
        let rows = (HC_LR + if w.inject.is_some() { HC } else { 0 }) as u32;
        let wi = w.inject.unwrap_or(w.down);
        let (su, quant) = (stride as u32, xq.is_some() as u32);
        let mut bar = self.sync_words();
        let mut bar = bar.slice_mut(SYNC_HC..SYNC_HC + 2);
        unsafe {
            let mut l = self.stream.launch_builder(&f);
            l.arg(r.f32_data_mut())
                .arg(yp.f32_data())
                .arg(ip.f32_data())
                .arg(&mode)
                .arg(w.norm.f32_data())
                .arg(&eps)
                .arg(parts.f32_data())
                .arg(wr.f32_data())
                .arg(lg.f32_data())
                .arg(&su)
                .arg(&sg)
                .arg(w.down.bf16_data())
                .arg(wi.bf16_data())
                .arg(&rows)
                .arg(w.up.bf16_data())
                .arg(x.f32_data_mut());
            match xq {
                Some(q) => {
                    assert!(q.len >= QAct { m: t, k: HIDDEN }.words());
                    l.arg(q.f32_data_mut())
                }
                None => l.arg(&NULL),
            };
            l.arg(&quant);
            match inj {
                Some(i) => l.arg(i.f32_data_mut()),
                None => l.arg(&NULL),
            };
            l.arg(&mut xn)
                .arg(&mut lo)
                .arg(&mut bar)
                .launch(grid((blocks, 1, 1), 512))
                .unwrap();
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn gdn_conv_impl(
        &self,
        proj: &CudaBuffer,
        stride: usize,
        hist: &CudaBuffer,
        conv: &CudaBuffer,
        h: &mut CudaBuffer,
        t: usize,
        eps: f32,
    ) {
        assert!((1..=MAX_T).contains(&t));
        let f = self.fl("fl_gdn_conv");
        let (s, tu) = (stride as u32, t as u32);
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(proj.f32_data())
                .arg(&s)
                .arg(hist.f32_data())
                .arg(conv.f32_data())
                .arg(h.f32_data_mut())
                .arg(&tu)
                .arg(&eps)
                .launch(grid((GDN_CONV / 128, 1, 1), 128))
                .unwrap();
        }
    }

    pub(super) fn gdn_conv_commit_impl(
        &self,
        hist: &mut CudaBuffer,
        proj: &CudaBuffer,
        stride: usize,
        win: &CudaBuffer,
        t: usize,
    ) {
        let f = self.fl("fl_gdn_conv_commit");
        let (s, tu) = (stride as u32, t as u32);
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(hist.f32_data_mut())
                .arg(proj.f32_data())
                .arg(&s)
                .arg(win.f32_data())
                .arg(&tu)
                .launch(grid((GDN_CONV.div_ceil(256), 1, 1), 256))
                .unwrap();
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn gdn_step_impl(
        &self,
        state: &mut CudaBuffer,
        h: &CudaBuffer,
        proj: &CudaBuffer,
        stride: usize,
        p: &GdnParams<'_, CudaBuffer>,
        y: &mut CudaBuffer,
        yq: Option<&mut CudaBuffer>,
        t: usize,
        mode: GdnMode<'_, CudaBuffer>,
        eps: f32,
    ) {
        assert!((1..=MAX_T).contains(&t));
        let quant = yq.is_some() as u32;
        let (win, commit) = match mode {
            GdnMode::ReadOnly => (h, 0u32),
            GdnMode::Commit { win } => (win, 1u32),
        };
        let f = self.fl("fl_gdn_step");
        let (s, tu) = (stride as u32, t as u32);
        unsafe {
            let mut l = self.stream.launch_builder(&f);
            l.arg(state.f32_data_mut())
                .arg(h.f32_data())
                .arg(proj.f32_data())
                .arg(&s)
                .arg(p.dt_bias.f32_data())
                .arg(p.ssm_a.f32_data())
                .arg(p.norm.f32_data())
                .arg(y.f32_data_mut())
                .arg(&tu)
                .arg(win.f32_data())
                .arg(&commit)
                .arg(&eps);
            match yq {
                Some(q) => l.arg(q.f32_data_mut()),
                None => l.arg(&NULL),
            };
            l.arg(&quant).launch(grid((GDN_HV, 1, 1), 512)).unwrap();
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn gdn_conv_step_impl(
        &self,
        state: &mut CudaBuffer,
        proj: &CudaBuffer,
        stride: usize,
        hist: &CudaBuffer,
        p: &GdnParams<'_, CudaBuffer>,
        y: &mut CudaBuffer,
        yq: Option<&mut CudaBuffer>,
        t: usize,
        mode: GdnMode<'_, CudaBuffer>,
        eps: f32,
    ) {
        assert!((1..=MAX_T).contains(&t));
        let quant = yq.is_some() as u32;
        let (win, commit) = match mode {
            GdnMode::ReadOnly => (proj, 0u32),
            GdnMode::Commit { win } => (win, 1u32),
        };
        let f = self.fl("fl_gdn_conv_step");
        let (s, tu) = (stride as u32, t as u32);
        unsafe {
            let mut l = self.stream.launch_builder(&f);
            l.arg(state.f32_data_mut())
                .arg(proj.f32_data())
                .arg(&s)
                .arg(p.dt_bias.f32_data())
                .arg(p.ssm_a.f32_data())
                .arg(p.norm.f32_data())
                .arg(y.f32_data_mut())
                .arg(&tu)
                .arg(win.f32_data())
                .arg(&commit)
                .arg(&eps);
            match yq {
                Some(q) => l.arg(q.f32_data_mut()),
                None => l.arg(&NULL),
            };
            l.arg(&quant)
                .arg(hist.f32_data())
                .arg(p.conv.f32_data())
                .launch(grid((GDN_HV, 1, 1), 512))
                .unwrap();
        }
    }

    pub(super) fn router_topk_impl(
        &self,
        logits: &CudaBuffer,
        stride: usize,
        n_expert: usize,
        ids: &mut CudaBuffer,
        w: &mut CudaBuffer,
        t: usize,
    ) {
        assert!((TOPK..=512).contains(&n_expert));
        let f = self.fl("fl_router_topk");
        let (s, ne) = (stride as u32, n_expert as u32);
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(logits.f32_data())
                .arg(&s)
                .arg(&ne)
                .arg(ids.f32_data_mut())
                .arg(w.f32_data_mut())
                .launch(grid((t, 1, 1), 32))
                .unwrap();
        }
    }

    pub(super) fn moe_plan_impl(
        &self,
        ids: &CudaBuffer,
        table: &CudaBuffer,
        shared: u64,
        plan: &mut CudaBuffer,
        t: usize,
    ) {
        assert!((1..=MAX_T).contains(&t) && plan.len >= MoePlan::WORDS && table.len >= 2 * EXPERTS);
        let f = self.fl("fl_moe_plan");
        let tu = t as u32;
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(ids.f32_data())
                .arg(table.f32_data())
                .arg(&shared)
                .arg(plan.f32_data_mut())
                .arg(&tu)
                .launch(grid((1, 1, 1), 128))
                .unwrap();
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn moe_route_impl(
        &self,
        logits: &CudaBuffer,
        stride: usize,
        n_expert: usize,
        forced: Option<&CudaBuffer>,
        table: &CudaBuffer,
        shared: u64,
        ids: &mut CudaBuffer,
        w: &mut CudaBuffer,
        plan: &mut CudaBuffer,
        t: usize,
    ) {
        assert!((1..=MAX_T).contains(&t) && (TOPK..=512).contains(&n_expert));
        assert!(plan.len >= MoePlan::WORDS && table.len >= 2 * EXPERTS);
        let f = self.fl("fl_moe_route");
        let (s, ne, tu, fo) = (
            stride as u32,
            n_expert as u32,
            t as u32,
            forced.is_some() as u32,
        );
        unsafe {
            let mut l = self.stream.launch_builder(&f);
            l.arg(logits.f32_data()).arg(&s).arg(&ne);
            match forced {
                Some(b) => l.arg(b.f32_data()),
                None => l.arg(&NULL),
            };
            l.arg(&fo)
                .arg(table.f32_data())
                .arg(&shared)
                .arg(ids.f32_data_mut())
                .arg(w.f32_data_mut())
                .arg(plan.f32_data_mut())
                .arg(&tu)
                .launch(grid((1, 1, 1), 256))
                .unwrap();
        }
    }

    pub(super) fn moe_grouped_impl(
        &self,
        xq: &CudaBuffer,
        plan: &CudaBuffer,
        scratch: &mut CudaBuffer,
        parts: &mut CudaBuffer,
        t: usize,
    ) {
        assert!(
            scratch.len >= MoePlan::scratch_words() && parts.len >= MoePlan::PARTS_ROWS * HIDDEN
        );
        const GU: [&str; 8] = [
            "fl_moe_gu_t1",
            "fl_moe_gu_t2",
            "fl_moe_gu_t3",
            "fl_moe_gu_t4",
            "fl_moe_gu_t5",
            "fl_moe_gu_t6",
            "fl_moe_gu_t7",
            "fl_moe_gu_t8",
        ];
        const DOWN: [&str; 8] = [
            "fl_moe_down_t1",
            "fl_moe_down_t2",
            "fl_moe_down_t3",
            "fl_moe_down_t4",
            "fl_moe_down_t5",
            "fl_moe_down_t6",
            "fl_moe_down_t7",
            "fl_moe_down_t8",
        ];
        // A group has at most t entries (an expert appears once per token; the shared group has t).
        let tu = t as u32;
        let f = self.fl(GU[t - 1]);
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(xq.f32_data())
                .arg(&tu)
                .arg(plan.f32_data())
                .arg(scratch.f32_data_mut())
                .launch(grid((16 * super::llm::sm_count(), 1, 1), 128))
                .unwrap();
        }
        let f = self.fl(DOWN[t - 1]);
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(scratch.f32_data())
                .arg(plan.f32_data())
                .arg(parts.f32_data_mut())
                .launch(grid((8 * super::llm::sm_count(), 1, 1), 256))
                .unwrap();
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn moe_combine_impl(
        &self,
        parts: &CudaBuffer,
        w: &CudaBuffer,
        logits: &CudaBuffer,
        stride: usize,
        sg: Option<usize>,
        y: &mut CudaBuffer,
        t: usize,
    ) {
        let f = self.fl("fl_moe_combine");
        let (s, sgi) = (stride as u32, sg.map_or(-1i32, |c| c as i32));
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(parts.f32_data())
                .arg(w.f32_data())
                .arg(logits.f32_data())
                .arg(&s)
                .arg(&sgi)
                .arg(y.f32_data_mut())
                .launch(grid((HIDDEN / 256, t, 1), 256))
                .unwrap();
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn qsa_prep_impl(
        &self,
        proj: &CudaBuffer,
        stride: usize,
        win: &CudaBuffer,
        norms: &QsaNorms<'_, CudaBuffer>,
        rope: (&CudaBuffer, &CudaBuffer),
        q: &mut CudaBuffer,
        cache: QsaCache<'_, CudaBuffer>,
        t: usize,
        eps: f32,
    ) {
        assert!((1..=MAX_T).contains(&t) && q.len >= crate::flash::qsa_q_words(t));
        let f = self.fl("fl_qsa_prep");
        let (s, tu) = (stride as u32, t as u32);
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(proj.f32_data())
                .arg(&s)
                .arg(win.f32_data())
                .arg(norms.q.f32_data())
                .arg(norms.k.f32_data())
                .arg(norms.iq.f32_data())
                .arg(norms.ik.f32_data())
                .arg(rope.0.f32_data())
                .arg(rope.1.f32_data())
                .arg(q.f32_data_mut())
                .arg(cache.k.bf16_data_mut())
                .arg(cache.v.bf16_data_mut())
                .arg(cache.ring.f32_data_mut())
                .arg(cache.pooled.f32_data_mut())
                .arg(&tu)
                .arg(&eps)
                .launch(grid((34, t, 1), 256))
                .unwrap();
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn qsa_select_impl(
        &self,
        pooled: &CudaBuffer,
        q: &CudaBuffer,
        win: &CudaBuffer,
        scores: &mut CudaBuffer,
        ids: &mut CudaBuffer,
        max_blocks: usize,
        t: usize,
    ) {
        assert!(max_blocks <= 8192, "qsa_select: context above 32K cells");
        assert!(scores.len >= t * max_blocks && ids.len >= t * QSA_WIDTH);
        let iq = q.f32_data().slice(t * QSA_HEADS * QSA_D..);
        let mb = max_blocks as u32;
        let f = self.fl("fl_qsa_scores");
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(pooled.f32_data())
                .arg(&iq)
                .arg(win.f32_data())
                .arg(scores.f32_data_mut())
                .arg(&mb)
                .launch(grid((max_blocks.div_ceil(8), t, 1), 256))
                .unwrap();
        }
        let f = self.fl("fl_qsa_select");
        let mask = 0u32;
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(scores.f32_data())
                .arg(win.f32_data())
                .arg(ids.f32_data_mut())
                .arg(&mb)
                .arg(&NULL)
                .arg(&mask)
                .launch(grid((t, 1, 1), 1024))
                .unwrap();
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn qsa_select_union_impl(
        &self,
        pooled: &CudaBuffer,
        q: &CudaBuffer,
        win: &CudaBuffer,
        scores: &mut CudaBuffer,
        ids: &mut CudaBuffer,
        union: &mut CudaBuffer,
        max_blocks: usize,
        t: usize,
    ) {
        assert!(max_blocks <= 8192 && union.len >= crate::flash::qsa_union_words(max_blocks, t));
        assert!(scores.len >= t * max_blocks && ids.len >= t * QSA_WIDTH);
        let iq = q.f32_data().slice(t * QSA_HEADS * QSA_D..);
        let mb = max_blocks as u32;
        let f = self.fl("fl_qsa_scores");
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(pooled.f32_data())
                .arg(&iq)
                .arg(win.f32_data())
                .arg(scores.f32_data_mut())
                .arg(&mb)
                .launch(grid((max_blocks.div_ceil(8), t, 1), 256))
                .unwrap();
        }
        let cap = crate::flash::qsa_union_cap(t);
        let (mut bm, mut rest) = union.f32_data_mut().split_at_mut(max_blocks);
        let (mut ub, mut rest) = rest.split_at_mut(cap);
        let (mut um, mut uc) = rest.split_at_mut(cap);
        let f = self.fl("fl_qsa_select");
        let mask = 1u32;
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(scores.f32_data())
                .arg(win.f32_data())
                .arg(ids.f32_data_mut())
                .arg(&mb)
                .arg(&mut bm)
                .arg(&mask)
                .launch(grid((t, 1, 1), 1024))
                .unwrap();
        }
        let f = self.fl("fl_qsa_union");
        let tu = t as u32;
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(&mut bm)
                .arg(win.f32_data())
                .arg(&tu)
                .arg(&mut ub)
                .arg(&mut um)
                .arg(&mut uc)
                .launch(grid((1, 1, 1), 1024))
                .unwrap();
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn qsa_attend_union_impl(
        &self,
        q: &CudaBuffer,
        k_cache: &CudaBuffer,
        v_cache: &CudaBuffer,
        union: &CudaBuffer,
        max_blocks: usize,
        proj: &CudaBuffer,
        stride: usize,
        win: &CudaBuffer,
        scratch: &mut CudaBuffer,
        out: &mut CudaBuffer,
        outq: Option<&mut CudaBuffer>,
        t: usize,
    ) {
        assert!(scratch.len >= crate::flash::qsa_union_scratch_words(t));
        let cap = crate::flash::qsa_union_cap(t);
        let nchu = cap.div_ceil(16);
        let u = union.f32_data();
        let (ub, um, uc) = (
            u.slice(max_blocks..max_blocks + cap),
            u.slice(max_blocks + cap..max_blocks + 2 * cap),
            u.slice(max_blocks + 2 * cap..max_blocks + 2 * cap + 1),
        );
        let (tu, nu) = (t as u32, nchu as u32);
        let f = self.fl("fl_qsa_attend_union");
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(q.f32_data())
                .arg(k_cache.bf16_data())
                .arg(v_cache.bf16_data())
                .arg(&ub)
                .arg(&um)
                .arg(&uc)
                .arg(win.f32_data())
                .arg(scratch.f32_data_mut())
                .arg(&tu)
                .arg(&nu)
                .launch(grid((nchu, QSA_KV, 1), 256))
                .unwrap();
        }
        let f = self.fl("fl_qsa_union_merge");
        let (s, quant) = (stride as u32, outq.is_some() as u32);
        unsafe {
            let mut l = self.stream.launch_builder(&f);
            l.arg(scratch.f32_data())
                .arg(proj.f32_data())
                .arg(&s)
                .arg(&uc)
                .arg(out.f32_data_mut());
            match outq {
                Some(q) => l.arg(q.f32_data_mut()),
                None => l.arg(&NULL),
            };
            l.arg(&quant)
                .arg(&tu)
                .arg(&nu)
                .launch(grid((QSA_HEADS, t, 1), 256))
                .unwrap();
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn qsa_attend_impl(
        &self,
        q: &CudaBuffer,
        k_cache: &CudaBuffer,
        v_cache: &CudaBuffer,
        ids: &CudaBuffer,
        proj: &CudaBuffer,
        stride: usize,
        win: &CudaBuffer,
        scratch: &mut CudaBuffer,
        out: &mut CudaBuffer,
        outq: Option<&mut CudaBuffer>,
        t: usize,
    ) {
        assert!(scratch.len >= crate::flash::qsa_attend_scratch_words(t));
        let nch = QSA_WIDTH.div_ceil(crate::flash::QSA_CHUNK);
        let f = self.fl("fl_qsa_attend");
        unsafe {
            self.stream
                .launch_builder(&f)
                .arg(q.f32_data())
                .arg(k_cache.bf16_data())
                .arg(v_cache.bf16_data())
                .arg(ids.f32_data())
                .arg(win.f32_data())
                .arg(scratch.f32_data_mut())
                .launch(grid((nch, QSA_KV, t), 256))
                .unwrap();
        }
        let f = self.fl("fl_qsa_merge");
        let s = stride as u32;
        let quant = outq.is_some() as u32;
        let tu = t as u32;
        unsafe {
            let mut l = self.stream.launch_builder(&f);
            l.arg(scratch.f32_data())
                .arg(proj.f32_data())
                .arg(&s)
                .arg(win.f32_data())
                .arg(out.f32_data_mut());
            match outq {
                Some(q) => l.arg(q.f32_data_mut()),
                None => l.arg(&NULL),
            };
            l.arg(&quant)
                .arg(&tu)
                .launch(grid((QSA_HEADS, t, 1), 256))
                .unwrap();
        }
    }
}

/// GPU timing for benchmarks (`flash-kernel-bench`).
impl CudaComputeDevice {
    /// GPU milliseconds between CUDA events recorded before and after `f` (whatever `f`
    /// enqueues on this device's stream, including any idle time between its launches).
    pub fn event_ms(&self, f: &mut dyn FnMut()) -> f32 {
        use cudarc::driver::sys::CUevent_flags;
        let a = self
            .ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .unwrap();
        let b = self
            .ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .unwrap();
        a.record(&self.stream).unwrap();
        f();
        b.record(&self.stream).unwrap();
        a.elapsed_ms(&b).unwrap()
    }

    /// Capture `f` into a CUDA graph (run it eagerly once first, so modules are loaded and
    /// pools warm), then return the graph.
    pub fn capture(&self, f: &mut dyn FnMut()) -> cudarc::driver::CudaGraph {
        f();
        self.stream.synchronize().unwrap();
        self.begin_capture();
        f();
        self.end_capture().expect("empty graph")
    }
}

/// Parity against `CpuDevice` (the reference in `cpu::flash`) for windows of 1, 2, 4 and 8
/// tokens. Skips without a GPU unless `TANG_REQUIRE_CUDA=1`. Run in release: the reference is
/// plain loops at the model's real shapes.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash::{self as fl, u32s, ExpertBlob, HcPending};
    use crate::{ComputeBuffer, ComputeDevice, CpuDevice};

    fn on_gpu(check: fn(&CudaComputeDevice)) {
        match std::panic::catch_unwind(CudaComputeDevice::new) {
            Ok(Ok(dev)) => check(&dev),
            _ if std::env::var("TANG_REQUIRE_CUDA").is_ok_and(|v| v == "1") => {
                panic!("TANG_REQUIRE_CUDA=1 but no CUDA device")
            }
            _ => eprintln!("no CUDA device: skipping"),
        }
    }

    const TS: [usize; 4] = [1, 2, 4, 8];

    /// xorshift values in [-scale, scale).
    struct Rng(u64);
    impl Rng {
        fn u(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn f(&mut self, scale: f32) -> f32 {
            ((self.u() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
        }
        fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
            (0..n).map(|_| self.f(scale)).collect()
        }
        fn bf16(&mut self, n: usize, scale: f32) -> Vec<u16> {
            (0..n)
                .map(|_| (self.f(scale).to_bits() >> 16) as u16)
                .collect()
        }
        /// A random GGUF Q2_0 matrix.
        fn q2(&mut self, n: usize, k: usize) -> Vec<u8> {
            let mut b = Vec::with_capacity(fl::q2_bytes(n, k));
            for _ in 0..n * k / 64 {
                let d = fl::f32_to_f16(0.01 + 0.03 * (self.f(1.0) + 1.0) / 2.0);
                b.extend(d.to_le_bytes());
                for _ in 0..16 {
                    b.push(self.u() as u8);
                }
            }
            b
        }
    }

    /// |got − want| <= tol · (scale + |want|), `scale` the mean |want|.
    fn close(got: &[f32], want: &[f32], tol: f32, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        let scale = want.iter().map(|v| v.abs()).sum::<f32>() / want.len().max(1) as f32;
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= tol * (scale + w.abs()),
                "{what} at {i}: {g} vs {w} (scale {scale})"
            );
        }
    }

    fn same_bits(got: &[f32], want: &[f32], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(g.to_bits() == w.to_bits(), "{what} at {i}: {g} vs {w}");
        }
    }

    fn win<D: ComputeDevice>(d: &D, pos0: usize, n_keep: usize) -> D::Buffer {
        d.upload_u32(&[pos0 as u32, n_keep as u32, 0])
    }

    // ---- Q2 / int8 / dense GEMVs ----

    fn q2_linear_vs_cpu<D: ComputeDevice>(g: &D) {
        let c = CpuDevice::new();
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for &(k, n) in &[(2560, 300), (640, 2560), (10240, 64)] {
            let raw = rng.q2(n, k);
            let (gw, cw) = (g.upload_q2(&raw, n, k), c.upload_q2(&raw, n, k));
            same_bits(&gw.to_vec(), &c.download(&cw), "q2 to_vec");
            for t in TS {
                let mut x = rng.vec(t * k, 3.0);
                x[5] = 0.0; // a zero chunk element, and below a whole zero chunk
                for v in x.iter_mut().skip(64).take(32) {
                    *v = 0.0;
                }
                let qa = QAct { m: t, k };
                let (mut gq, mut cq) = (g.alloc_f32(qa.words()), c.alloc_f32(qa.words()));
                g.quantize_act_into(&g.upload_f32(&x), &mut gq, t, k);
                c.quantize_act_into(&c.upload_f32(&x), &mut cq, t, k);
                same_bits(
                    &g.download(&gq),
                    &c.download(&cq),
                    &format!("quantize t={t} k={k}"),
                );
                let (mut gy, mut cy) = (g.alloc_f32(t * n), c.alloc_f32(t * n));
                g.q2_linear_into(&gq, &gw, &mut gy, t, k, n);
                c.q2_linear_into(&cq, &cw, &mut cy, t, k, n);
                close(
                    &g.download(&gy),
                    &c.download(&cy),
                    1e-5,
                    &format!("q2 linear t={t} {k}->{n}"),
                );
            }
        }
    }

    fn q4x_linear_vs_cpu<D: ComputeDevice>(g: &D) {
        let c = CpuDevice::new();
        let mut rng = Rng(0x51);
        for &(k, n) in &[(2560, 300), (6144, 129), (10240, 40)] {
            let packed: Vec<u32> = (0..n * k / 8).map(|_| rng.u() as u32).collect();
            let sc: Vec<u16> = (0..n * k / 64)
                .map(|_| (rng.f(0.02).to_bits() >> 16) as u16)
                .collect();
            let bi: Vec<u16> = (0..n * k / 64)
                .map(|_| (rng.f(0.1).to_bits() >> 16) as u16)
                .collect();
            let (gw, cw) = (
                g.upload_q4x(&packed, &sc, &bi, n, k),
                c.upload_q4x(&packed, &sc, &bi, n, k),
            );
            // The same weights through tang-Q4 `linear` (f32 activations) agree to int8 rounding.
            let cq4 = c.upload_q4(&packed, &sc, &bi, 64);
            for t in TS {
                let x = rng.vec(t * k, 2.0);
                let qa = QAct { m: t, k };
                let (mut gq, mut cq) = (g.alloc_f32(qa.words()), c.alloc_f32(qa.words()));
                g.quantize_act_into(&g.upload_f32(&x), &mut gq, t, k);
                c.quantize_act_into(&c.upload_f32(&x), &mut cq, t, k);
                let (mut gy, mut cy) = (g.alloc_f32(t * n), c.alloc_f32(t * n));
                g.q4x_linear_into(&gq, &gw, &mut gy, t, k, n);
                c.q4x_linear_into(&cq, &cw, &mut cy, t, k, n);
                let want = c.download(&cy);
                close(
                    &g.download(&gy),
                    &want,
                    1e-5,
                    &format!("q4x linear t={t} {k}->{n}"),
                );
                let f32_path = c.download(&c.linear(&c.upload_f32(&x), &cq4, t, k, n));
                close(&want, &f32_path, 2e-2, &format!("q4x vs f32 q4 t={t}"));
            }
        }
    }

    fn q8x_linear_vs_cpu<D: ComputeDevice>(g: &D) {
        let c = CpuDevice::new();
        let mut rng = Rng(0x88);
        for &(k, n) in &[(2560, 300), (6144, 129), (10240, 40)] {
            let mut raw = Vec::with_capacity(n * k / 32 * fl::Q8_BLOCK_BYTES);
            for _ in 0..n * k / 32 {
                raw.extend(fl::f32_to_f16(0.001 + 0.002 * (rng.f(1.0) + 1.0)).to_le_bytes());
                raw.extend((0..32).map(|_| rng.u() as u8));
            }
            let (gw, cw) = (g.upload_q8x(&raw, n, k), c.upload_q8x(&raw, n, k));
            for t in TS {
                let x = rng.vec(t * k, 2.0);
                let qa = QAct { m: t, k };
                let (mut gq, mut cq) = (g.alloc_f32(qa.words()), c.alloc_f32(qa.words()));
                g.quantize_act_into(&g.upload_f32(&x), &mut gq, t, k);
                c.quantize_act_into(&c.upload_f32(&x), &mut cq, t, k);
                let (mut gy, mut cy) = (g.alloc_f32(t * n), c.alloc_f32(t * n));
                g.q8x_linear_into(&gq, &gw, &mut gy, t, k, n);
                c.q8x_linear_into(&cq, &cw, &mut cy, t, k, n);
                let want = c.download(&cy);
                close(
                    &g.download(&gy),
                    &want,
                    1e-5,
                    &format!("q8x linear t={t} {k}->{n}"),
                );
                // Against the dequantized weights and the f32 activations: int8 rounding only.
                let mut deq = vec![0f32; n * k];
                for (b, blk) in raw.chunks(fl::Q8_BLOCK_BYTES).enumerate() {
                    let d = fl::f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                    for i in 0..32 {
                        deq[b * 32 + i] = d * blk[2 + i] as i8 as f32;
                    }
                }
                let f32_path =
                    c.download(&c.linear(&c.upload_f32(&x), &c.upload_f32(&deq), t, k, n));
                close(&want, &f32_path, 2e-2, &format!("q8x vs f32 t={t}"));
            }
        }
    }

    fn linear_into_vs_cpu<D: ComputeDevice>(g: &D) {
        let c = CpuDevice::new();
        let mut rng = Rng(7);
        let (k, n) = (2560, 333);
        let wb = rng.bf16(n * k, 0.5);
        let packed: Vec<u32> = (0..n * k / 8).map(|_| rng.u() as u32).collect();
        let (sc, bi) = (rng.bf16(n * k / 64, 0.02), rng.bf16(n * k / 64, 0.1));
        for t in TS {
            let x = rng.vec(t * k, 1.0);
            for (gw, cw) in [
                (g.upload_bf16(&wb), c.upload_bf16(&wb)),
                (
                    g.upload_q4(&packed, &sc, &bi, 64),
                    c.upload_q4(&packed, &sc, &bi, 64),
                ),
            ] {
                let mut gy = g.alloc_f32(t * n);
                g.linear_into(&g.upload_f32(&x), &gw, &mut gy, t, k, n);
                let cy = c.linear(&c.upload_f32(&x), &cw, t, k, n);
                close(
                    &g.download(&gy),
                    &c.download(&cy),
                    1e-4,
                    &format!("linear_into t={t}"),
                );
            }
        }
    }

    // ---- hyper-connections ----

    /// (norm, down, up, inject, r0) on the host.
    type HcWeightsHost<'a> = (&'a [f32], &'a [u16], &'a [u16], &'a [u16], &'a [f32]);
    /// (y, inj, parts, router weights, logits) on the host.
    type HcInputsHost<'a> = (&'a [f32], &'a [f32], &'a [f32], &'a [f32], &'a [f32]);

    fn hc_vs_cpu<D: ComputeDevice>(g: &D) {
        let c = CpuDevice::new();
        let mut rng = Rng(11);
        let w = HC * HIDDEN;
        let norm: Vec<f32> = rng.vec(w, 1.0).iter().map(|v| 1.0 + 0.2 * v).collect();
        let (down, up, inject) = (
            rng.bf16(HC_LR * w, 0.05),
            fl::hc_up_repack(&rng.bf16(w * HC_LR, 0.2)),
            rng.bf16(HC * w, 0.05),
        );
        let ls = EXPERTS + 1;
        // (inject rows, pending: 0 none, 1 write, 2 MoE write)
        for t in TS {
            for (with_inject, mode) in [(true, 0), (true, 1), (false, 1), (true, 2)] {
                let r0 = rng.vec(t * w, 2.0);
                let (y, injp) = (rng.vec(t * HIDDEN, 1.0), rng.vec(t * HC, 3.0));
                let parts = rng.vec(MoePlan::PARTS_ROWS * HIDDEN, 1.0);
                let (rw, logits) = (rng.vec(t * TOPK, 0.3), rng.vec(t * ls, 2.0));
                // (x, inj, r, xq) after one read on `d`; `split` does the write as its own op.
                fn go<D: ComputeDevice>(
                    d: &D,
                    split: bool,
                    a: HcWeightsHost,
                    b: HcInputsHost,
                    with_inject: bool,
                    mode: u32,
                    t: usize,
                ) -> [Vec<f32>; 4] {
                    let (norm, down, up, inject, r0) = a;
                    let (y, injp, parts, rw, logits) = b;
                    let ls = EXPERTS + 1;
                    let (n, dn, u, i) = (
                        d.upload_f32(norm),
                        d.upload_bf16(down),
                        d.upload_bf16(up),
                        d.upload_bf16(inject),
                    );
                    let hw = HcWeights {
                        norm: &n,
                        down: &dn,
                        up: &u,
                        inject: with_inject.then_some(&i),
                    };
                    let mut r = d.upload_f32(r0);
                    let (yb, ib) = (d.upload_f32(y), d.upload_f32(injp));
                    let (pb, wb, lb) =
                        (d.upload_f32(parts), d.upload_f32(rw), d.upload_f32(logits));
                    let pending = match mode {
                        0 => None,
                        1 => Some(HcPending::Write { y: &yb, inj: &ib }),
                        _ => Some(HcPending::Moe {
                            parts: &pb,
                            w: &wb,
                            logits: &lb,
                            stride: ls,
                            sg: Some(EXPERTS),
                            inj: &ib,
                        }),
                    };
                    let pending = if split && mode > 0 {
                        let mut yc = d.alloc_f32(t * HIDDEN);
                        let yref = if mode == 2 {
                            d.moe_combine_into(&pb, &wb, &lb, ls, Some(EXPERTS), &mut yc, t);
                            &yc
                        } else {
                            &yb
                        };
                        d.hc_write(&mut r, yref, &ib, t);
                        None
                    } else {
                        pending
                    };
                    let (mut x, mut inj, mut sc, mut xq) = (
                        d.alloc_f32(t * HIDDEN),
                        d.alloc_f32(t * HC),
                        d.alloc_f32(fl::hc_scratch_words(t)),
                        d.alloc_f32(QAct { m: t, k: HIDDEN }.words()),
                    );
                    d.hc_read_into(
                        &mut r,
                        pending,
                        &hw,
                        &mut x,
                        Some(&mut xq),
                        with_inject.then_some(&mut inj),
                        &mut sc,
                        t,
                        1e-6,
                    );
                    [
                        d.download(&x),
                        d.download(&inj),
                        d.download(&r),
                        d.download(&xq),
                    ]
                }
                let a = (&norm[..], &down[..], &up[..], &inject[..], &r0[..]);
                let b = (&y[..], &injp[..], &parts[..], &rw[..], &logits[..]);
                let got = go(g, false, a, b, with_inject, mode, t);
                let want = go(&c, false, a, b, with_inject, mode, t);
                let what = format!("hc t={t} inject={with_inject} pending={mode}");
                close(&got[2], &want[2], 1e-6, &format!("{what} R"));
                close(&got[0], &want[0], 2e-4, &format!("{what} x"));
                if with_inject {
                    close(&got[1], &want[1], 2e-4, &format!("{what} inj"));
                }
                // The fused int8 output is the quantization of x.
                let mut q = g.alloc_f32(QAct { m: t, k: HIDDEN }.words());
                g.quantize_act_into(&g.upload_f32(&got[0]), &mut q, t, HIDDEN);
                same_bits(&got[3], &g.download(&q), &format!("{what} xq"));
                if mode > 0 {
                    // Fused write-then-read is bitwise the separate ops.
                    let split = go(g, true, a, b, with_inject, mode, t);
                    same_bits(&split[2], &got[2], &format!("{what} unfused R"));
                    same_bits(&split[0], &got[0], &format!("{what} unfused x"));
                }
            }
        }
    }

    // ---- GDN ----

    struct GdnCase {
        proj: Vec<f32>,
        hist: Vec<f32>,
        conv: Vec<f32>,
        dt: Vec<f32>,
        a: Vec<f32>,
        norm: Vec<f32>,
        state: Vec<f32>,
    }

    fn gdn_case(rng: &mut Rng, t: usize) -> GdnCase {
        let mut proj = rng.vec(t * GDN_PROJ, 1.0);
        for tt in 0..t {
            for h in 0..GDN_HV {
                proj[tt * GDN_PROJ + GDN_A + h] = rng.f(3.0);
                proj[tt * GDN_PROJ + GDN_B + h] = rng.f(3.0);
            }
        }
        // One head's a well past the softplus cutoff.
        proj[GDN_A + 5] = 30.0;
        GdnCase {
            proj,
            hist: rng.vec(fl::GDN_HIST, 1.0),
            conv: rng.vec(GDN_CONV * GDN_TAPS, 0.6),
            dt: rng.vec(GDN_HV, 1.0),
            a: rng
                .vec(GDN_HV, 1.0)
                .iter()
                .map(|v| -(v.abs() + 0.05))
                .collect(),
            norm: rng.vec(GDN_D, 1.0).iter().map(|v| 1.0 + 0.3 * v).collect(),
            state: rng.vec(fl::GDN_STATE, 0.3),
        }
    }

    /// (h, y readonly, y commit, state after commit, hist after commit)
    fn gdn_run<D: ComputeDevice>(d: &D, cs: &GdnCase, t: usize, n_keep: usize) -> [Vec<f32>; 5] {
        let (proj, hist, conv) = (
            d.upload_f32(&cs.proj),
            d.upload_f32(&cs.hist),
            d.upload_f32(&cs.conv),
        );
        let (dt, a, norm) = (
            d.upload_f32(&cs.dt),
            d.upload_f32(&cs.a),
            d.upload_f32(&cs.norm),
        );
        let p = GdnParams {
            conv: &conv,
            dt_bias: &dt,
            ssm_a: &a,
            norm: &norm,
        };
        let mut state = d.upload_f32(&cs.state);
        let mut h = d.alloc_f32(t * GDN_CONV);
        d.gdn_conv_into(&proj, GDN_PROJ, &hist, &conv, &mut h, t, 1e-6);
        let mut y = d.alloc_f32(t * GDN_V);
        let mut yq = d.alloc_f32(QAct { m: t, k: GDN_V }.words());
        d.gdn_step(
            &mut state,
            &h,
            &proj,
            GDN_PROJ,
            &p,
            &mut y,
            Some(&mut yq),
            t,
            GdnMode::ReadOnly,
            1e-6,
        );
        let mut q = d.alloc_f32(QAct { m: t, k: GDN_V }.words());
        d.quantize_act_into(&y, &mut q, t, GDN_V);
        same_bits(&d.download(&yq), &d.download(&q), "gdn fused int8 output");
        // Conv and step in one launch: bitwise the two.
        let mut yf = d.alloc_f32(t * GDN_V);
        d.gdn_conv_step(
            &mut state,
            &proj,
            GDN_PROJ,
            &hist,
            &p,
            &mut yf,
            None,
            t,
            GdnMode::ReadOnly,
            1e-6,
        );
        same_bits(&d.download(&yf), &d.download(&y), "gdn conv+step fused");
        let unchanged = d.download(&state);
        same_bits(&unchanged, &cs.state, "readonly leaves the state");
        let w = win(d, 0, n_keep);
        let mut yc = d.alloc_f32(t * GDN_V);
        d.gdn_step(
            &mut state,
            &h,
            &proj,
            GDN_PROJ,
            &p,
            &mut yc,
            None,
            t,
            GdnMode::Commit { win: &w },
            1e-6,
        );
        let mut hist2 = d.upload_f32(&cs.hist);
        d.gdn_conv_commit(&mut hist2, &proj, GDN_PROJ, &w, t);
        [
            d.download(&h),
            d.download(&y),
            d.download(&yc),
            d.download(&state),
            d.download(&hist2),
        ]
    }

    fn gdn_vs_cpu<D: ComputeDevice>(g: &D) {
        let c = CpuDevice::new();
        let mut rng = Rng(23);
        for t in TS {
            let cs = gdn_case(&mut rng, t);
            for n_keep in [0, 1, t.div_ceil(2), t, t + 3] {
                let got = gdn_run(g, &cs, t, n_keep);
                let want = gdn_run(&c, &cs, t, n_keep);
                let what = format!("gdn t={t} n_keep={n_keep}");
                close(&got[0], &want[0], 1e-5, &format!("{what} conv"));
                close(&got[1], &want[1], 2e-4, &format!("{what} y"));
                close(&got[3], &want[3], 1e-4, &format!("{what} state"));
                same_bits(&got[4], &want[4], &format!("{what} conv commit"));
                let k = n_keep.min(t);
                // Commit replays exactly what verify computed.
                same_bits(
                    &got[2][..k * GDN_V],
                    &got[1][..k * GDN_V],
                    &format!("{what} commit y"),
                );
                if k < t {
                    // And the next window, from the committed state, continues bitwise.
                    let mut rest = GdnCase {
                        proj: cs.proj[k * GDN_PROJ..].to_vec(),
                        state: got[3].clone(),
                        ..gdn_case(&mut Rng(1), 1)
                    };
                    rest.hist = cs.hist.clone();
                    let (proj, h) = (
                        g.upload_f32(&rest.proj),
                        g.upload_f32(&got[0][k * GDN_CONV..]),
                    );
                    let (dt, a, norm, conv) = (
                        g.upload_f32(&cs.dt),
                        g.upload_f32(&cs.a),
                        g.upload_f32(&cs.norm),
                        g.upload_f32(&cs.conv),
                    );
                    let p = GdnParams {
                        conv: &conv,
                        dt_bias: &dt,
                        ssm_a: &a,
                        norm: &norm,
                    };
                    let mut state = g.upload_f32(&got[3]);
                    let mut y = g.alloc_f32((t - k) * GDN_V);
                    g.gdn_step(
                        &mut state,
                        &h,
                        &proj,
                        GDN_PROJ,
                        &p,
                        &mut y,
                        None,
                        t - k,
                        GdnMode::ReadOnly,
                        1e-6,
                    );
                    same_bits(
                        &g.download(&y),
                        &got[1][k * GDN_V..],
                        &format!("{what} continue"),
                    );
                }
            }
        }
    }

    // ---- MoE ----

    fn router_vs_cpu<D: ComputeDevice>(g: &D) {
        let c = CpuDevice::new();
        let mut rng = Rng(31);
        let stride = EXPERTS + 1;
        for t in TS {
            let mut l = rng.vec(t * stride, 4.0);
            // Ties straddling the cut, and a run of equal logits.
            for e in [3, 40, 77, 300, 301] {
                l[e] = 5.5;
            }
            if t > 1 {
                l[stride + 9] = 9.0;
                l[stride + 10] = 9.0;
            }
            let (mut gi, mut gw, mut ci, mut cw) = (
                g.alloc_f32(t * TOPK),
                g.alloc_f32(t * TOPK),
                c.alloc_f32(t * TOPK),
                c.alloc_f32(t * TOPK),
            );
            g.router_topk_into(&g.upload_f32(&l), stride, EXPERTS, &mut gi, &mut gw, t);
            c.router_topk_into(&c.upload_f32(&l), stride, EXPERTS, &mut ci, &mut cw, t);
            assert_eq!(
                u32s(&g.download(&gi)),
                u32s(&c.download(&ci)),
                "router ids t={t}"
            );
            close(
                &g.download(&gw),
                &c.download(&cw),
                1e-6,
                &format!("router w t={t}"),
            );
        }
    }

    fn moe_vs_cpu<D: ComputeDevice>(g: &D) {
        let c = CpuDevice::new();
        let mut rng = Rng(41);
        // Eight distinct blobs; experts map onto them, some not resident.
        let blobs: Vec<Vec<u8>> = (0..8)
            .map(|_| {
                ExpertBlob::from_gguf(
                    &rng.q2(FF, HIDDEN),
                    &rng.q2(FF, HIDDEN),
                    &rng.q2(HIDDEN, FF),
                )
            })
            .collect();
        let shared_blob = ExpertBlob::from_gguf(
            &rng.q2(FF, HIDDEN),
            &rng.q2(FF, HIDDEN),
            &rng.q2(HIDDEN, FF),
        );
        let (gb, cb): (Vec<_>, Vec<_>) = blobs
            .iter()
            .map(|b| (g.upload_bytes(b), c.upload_bytes(b)))
            .unzip();
        let (gs, cs) = (g.upload_bytes(&shared_blob), c.upload_bytes(&shared_blob));
        let table = |d: &dyn Fn(usize) -> u64| -> Vec<u64> {
            (0..EXPERTS)
                .map(|e| if e % 7 == 3 { 0 } else { d(e % 8) })
                .collect()
        };
        let gt = table(&|i| g.buffer_addr(&gb[i]));
        let ct = table(&|i| c.buffer_addr(&cb[i]));
        let tw = |t: &[u64]| -> Vec<u32> {
            t.iter()
                .flat_map(|&p| [p as u32, (p >> 32) as u32])
                .collect()
        };
        for t in TS {
            // Few distinct experts per window so groups have several entries.
            let mut ids = vec![];
            for tt in 0..t {
                let mut row: Vec<u32> = vec![];
                while row.len() < TOPK {
                    let e = (rng.u() % 24) as u32 * 5 + (tt as u32 % 2);
                    if !row.contains(&e) {
                        row.push(e);
                    }
                }
                ids.extend(row);
            }
            let x = rng.vec(t * HIDDEN, 1.0);
            let w = rng.vec(t * TOPK, 0.3);
            let logits = rng.vec(t * (EXPERTS + 1), 2.0);
            for shared in [false, true] {
                // Plan: the device kernel against the reference on the same table.
                let gsh = if shared { g.buffer_addr(&gs) } else { 0 };
                let mut gplan = g.alloc_f32(MoePlan::WORDS);
                g.moe_plan_into(
                    &g.upload_u32(&ids),
                    &g.upload_u32(&tw(&gt)),
                    gsh,
                    &mut gplan,
                    t,
                );
                let want = crate::cpu::flash::moe_plan(&ids, &gt, gsh, t);
                let got = u32s(&g.download(&gplan));
                let (ng, ne, nm) = (want[0] as usize, want[1] as usize, want[2] as usize);
                assert_eq!(&got[..3], &want[..3], "plan header t={t}");
                assert_eq!(
                    got[MoePlan::GROUP_PTR..][..2 * ng],
                    want[MoePlan::GROUP_PTR..][..2 * ng],
                    "plan ptrs"
                );
                assert_eq!(
                    got[MoePlan::GROUP_START..][..ng + 1],
                    want[MoePlan::GROUP_START..][..ng + 1],
                    "plan starts"
                );
                assert_eq!(
                    got[MoePlan::ENT_TOK..][..ne],
                    want[MoePlan::ENT_TOK..][..ne],
                    "plan tok"
                );
                assert_eq!(
                    got[MoePlan::ENT_DST..][..ne],
                    want[MoePlan::ENT_DST..][..ne],
                    "plan dst"
                );
                assert_eq!(
                    got[MoePlan::MISSING..][..nm],
                    want[MoePlan::MISSING..][..nm],
                    "plan missing"
                );
                assert!(nm > 0 && ng > 0);

                // The fused route: router ids/weights as router_topk, plan as moe_plan on them,
                // and with forced ids, the plan above.
                let gl = g.upload_f32(&logits);
                let (mut ri, mut rw, mut rp) = (
                    g.alloc_f32(t * TOPK),
                    g.alloc_f32(t * TOPK),
                    g.alloc_f32(MoePlan::WORDS),
                );
                let gtab = g.upload_u32(&tw(&gt));
                g.moe_route_into(
                    &gl,
                    EXPERTS + 1,
                    EXPERTS,
                    None,
                    &gtab,
                    gsh,
                    &mut ri,
                    &mut rw,
                    &mut rp,
                    t,
                );
                let (mut ti, mut tw2) = (g.alloc_f32(t * TOPK), g.alloc_f32(t * TOPK));
                g.router_topk_into(&gl, EXPERTS + 1, EXPERTS, &mut ti, &mut tw2, t);
                same_bits(&g.download(&ri), &g.download(&ti), "route ids");
                same_bits(&g.download(&rw), &g.download(&tw2), "route w");
                let want_r = crate::cpu::flash::moe_plan(&u32s(&g.download(&ti)), &gt, gsh, t);
                let got_r = u32s(&g.download(&rp));
                let ner = want_r[1] as usize;
                assert_eq!(&got_r[..3], &want_r[..3], "route plan header");
                assert_eq!(
                    got_r[MoePlan::ENT_DST..][..ner],
                    want_r[MoePlan::ENT_DST..][..ner],
                    "route plan dst"
                );
                let gids = g.upload_u32(&ids);
                g.moe_route_into(
                    &gl,
                    EXPERTS + 1,
                    EXPERTS,
                    Some(&gids),
                    &gtab,
                    gsh,
                    &mut ri,
                    &mut rw,
                    &mut rp,
                    t,
                );
                let got_f = u32s(&g.download(&rp));
                assert_eq!(&got_f[..3], &want[..3], "forced plan header");
                assert_eq!(
                    got_f[MoePlan::ENT_TOK..][..ne],
                    want[MoePlan::ENT_TOK..][..ne],
                    "forced plan tok"
                );

                // Grouped experts and combine, each device with its own addresses.
                let run = |d: &dyn Fn() -> (Vec<f32>, Vec<f32>)| d();
                macro_rules! go {
                    ($d:expr, $tab:expr, $sh:expr) => {{
                        let d = $d;
                        let sh = if shared { $sh } else { 0 };
                        let mut plan = d.alloc_f32(MoePlan::WORDS);
                        d.moe_plan_into(
                            &d.upload_u32(&ids),
                            &d.upload_u32(&tw($tab)),
                            sh,
                            &mut plan,
                            t,
                        );
                        let mut xq = d.alloc_f32(QAct { m: t, k: HIDDEN }.words());
                        d.quantize_act_into(&d.upload_f32(&x), &mut xq, t, HIDDEN);
                        let (mut sc, mut parts) = (
                            d.alloc_f32(MoePlan::scratch_words()),
                            d.alloc_f32(MoePlan::PARTS_ROWS * HIDDEN),
                        );
                        unsafe { d.moe_grouped_into(&xq, &plan, &mut sc, &mut parts, t) };
                        let mut y = d.alloc_f32(t * HIDDEN);
                        d.moe_combine_into(
                            &parts,
                            &d.upload_f32(&w),
                            &d.upload_f32(&logits),
                            EXPERTS + 1,
                            shared.then_some(EXPERTS),
                            &mut y,
                            t,
                        );
                        (d.download(&parts), d.download(&y))
                    }};
                }
                let gotp = run(&|| go!(g, &gt, g.buffer_addr(&gs)));
                let wantp = run(&|| go!(&c, &ct, c.buffer_addr(&cs)));
                // h is requantized to int8 between gate/up and down, so a summation-order
                // difference can move one code by 1 at a rounding tie: a few outputs may move by
                // d_h · w; nearly all agree to fp32 noise.
                // The expert order is pinned (ExpertBlob::ORDER): GPU rows are the reference's
                // bit for bit, so a CPU-served miss and a VRAM hit agree exactly.
                let what = format!("moe parts t={t} shared={shared}");
                same_bits(&gotp.0, &wantp.0, &what);
                // Combine on the same parts.
                let mut cy = c.alloc_f32(t * HIDDEN);
                let ls = EXPERTS + 1;
                let sg = shared.then_some(EXPERTS);
                let (cp, cw, cl) = (
                    c.upload_f32(&gotp.0),
                    c.upload_f32(&w),
                    c.upload_f32(&logits),
                );
                c.moe_combine_into(&cp, &cw, &cl, ls, sg, &mut cy, t);
                let what = format!("moe combine t={t} shared={shared}");
                close(&gotp.1, &c.download(&cy), 1e-5, &what);
            }
        }
    }

    // ---- QSA ----

    fn qsa_prep_vs_cpu<D: ComputeDevice>(g: &D) {
        let c = CpuDevice::new();
        let mut rng = Rng(53);
        let max_ctx = 128;
        let (cos, sin) = fl::rope_table(max_ctx, 1e7);
        let norms: Vec<Vec<f32>> = [QSA_D, QSA_D, IDX_D, IDX_D]
            .iter()
            .map(|&n| rng.vec(n, 1.0).iter().map(|v| 1.0 + 0.3 * v).collect())
            .collect();
        for t in TS {
            // A run of windows from position 0, so blocks complete inside and across windows.
            let windows: Vec<(usize, Vec<f32>)> = (0..6)
                .map(|i| (i * t, rng.vec(t * QSA_PROJ, 1.0)))
                .collect();
            let run = |d: &dyn Fn() -> [Vec<f32>; 5]| d();
            macro_rules! go {
                ($d:expr) => {{
                    let d = $d;
                    let nb: Vec<_> = norms.iter().map(|v| d.upload_f32(v)).collect();
                    let qn = QsaNorms {
                        q: &nb[0],
                        k: &nb[1],
                        iq: &nb[2],
                        ik: &nb[3],
                    };
                    let (cb, sb) = (d.upload_f32(&cos), d.upload_f32(&sin));
                    let (mut kc, mut vc) = (
                        d.alloc_bf16(max_ctx * QSA_KV * QSA_D),
                        d.alloc_bf16(max_ctx * QSA_KV * QSA_D),
                    );
                    let (mut ring, mut pooled) =
                        (d.alloc_f32(16 * IDX_D), d.alloc_f32(max_ctx / 4 * IDX_D));
                    let mut q = d.alloc_f32(fl::qsa_q_words(t));
                    for (pos0, proj) in &windows {
                        let cache = QsaCache {
                            k: &mut kc,
                            v: &mut vc,
                            ring: &mut ring,
                            pooled: &mut pooled,
                        };
                        d.qsa_prep(
                            &d.upload_f32(proj),
                            QSA_PROJ,
                            &win(d, *pos0, 0),
                            &qn,
                            (&cb, &sb),
                            &mut q,
                            cache,
                            t,
                            1e-6,
                        );
                    }
                    [
                        d.download(&q),
                        d.download(&kc),
                        d.download(&vc),
                        d.download(&ring),
                        d.download(&pooled),
                    ]
                }};
            }
            let got = run(&|| go!(g));
            let want = run(&|| go!(&c));
            let n = 6 * t;
            let what = format!("qsa prep t={t}");
            close(&got[0], &want[0], 1e-4, &format!("{what} q"));
            close(
                &got[1][..n * 512],
                &want[1][..n * 512],
                1e-2,
                &format!("{what} k cache"),
            );
            same_bits(
                &got[2][..n * 512],
                &want[2][..n * 512],
                &format!("{what} v cache"),
            );
            same_bits(&got[3], &want[3], &format!("{what} ring"));
            close(
                &got[4][..n / 4 * IDX_D],
                &want[4][..n / 4 * IDX_D],
                1e-4,
                &format!("{what} pooled"),
            );
        }
    }

    fn qsa_select_vs_cpu<D: ComputeDevice>(g: &D) {
        let c = CpuDevice::new();
        let mut rng = Rng(59);
        let max_blocks = 1024;
        let pooled = rng.vec(max_blocks * IDX_D, 1.0);
        for t in TS {
            let mut q = rng.vec(fl::qsa_q_words(t), 1.0);
            // Make heads share signs so plenty of scores tie at a few values.
            for v in q.iter_mut().skip(t * QSA_HEADS * QSA_D) {
                *v = (*v * 4.0).round() / 4.0;
            }
            for &pos0 in &[0usize, 2040, 2048, 2049, 2050, 2051, 2052, 3000, 4093 - t] {
                if pos0 + t > max_blocks * 4 {
                    continue;
                }
                let run = |d: &dyn Fn() -> (Vec<f32>, Vec<f32>)| d();
                macro_rules! go {
                    ($d:expr) => {{
                        let d = $d;
                        let (mut sc, mut ids) =
                            (d.alloc_f32(t * max_blocks), d.alloc_f32(t * QSA_WIDTH));
                        d.qsa_select_into(
                            &d.upload_f32(&pooled),
                            &d.upload_f32(&q),
                            &win(d, pos0, 0),
                            &mut sc,
                            &mut ids,
                            max_blocks,
                            t,
                        );
                        (d.download(&sc), d.download(&ids))
                    }};
                }
                let got = run(&|| go!(g));
                let want = run(&|| go!(&c));
                for tt in 0..t {
                    let n_kv = pos0 + tt + 1;
                    let nb = n_kv / 4;
                    same_bits(
                        &got.0[tt * max_blocks..tt * max_blocks + nb],
                        &want.0[tt * max_blocks..tt * max_blocks + nb],
                        &format!("qsa scores t={t} pos0={pos0} row {tt}"),
                    );
                    let w = crate::cpu::flash::qsa_n_sel(n_kv);
                    assert_eq!(
                        u32s(&got.1[tt * QSA_WIDTH..tt * QSA_WIDTH + w]),
                        u32s(&want.1[tt * QSA_WIDTH..tt * QSA_WIDTH + w]),
                        "qsa ids t={t} pos0={pos0} row {tt}"
                    );
                }
            }
        }
    }

    fn qsa_attend_vs_cpu<D: ComputeDevice>(g: &D) {
        let c = CpuDevice::new();
        let mut rng = Rng(61);
        let max_ctx = 4096;
        let kv: Vec<f32> = rng
            .vec(max_ctx * QSA_KV * QSA_D, 1.0)
            .iter()
            .map(|&v| fl::bf16_round(v))
            .collect();
        let vv: Vec<f32> = rng
            .vec(max_ctx * QSA_KV * QSA_D, 1.0)
            .iter()
            .map(|&v| fl::bf16_round(v))
            .collect();
        let to_bits =
            |v: &[f32]| -> Vec<u16> { v.iter().map(|x| (x.to_bits() >> 16) as u16).collect() };
        for t in TS {
            let q = rng.vec(fl::qsa_q_words(t), 1.0);
            let proj = rng.vec(t * QSA_PROJ, 2.0);
            for &pos0 in &[0usize, 63, 2045, 3500] {
                // Arbitrary ascending selections.
                let mut ids = vec![0u32; t * QSA_WIDTH];
                for tt in 0..t {
                    let n_kv = pos0 + tt + 1;
                    let w = crate::cpu::flash::qsa_n_sel(n_kv);
                    let mut cells: Vec<u32> = (0..n_kv as u32).collect();
                    while cells.len() > w {
                        let i = (rng.u() as usize) % cells.len();
                        cells.remove(i);
                    }
                    ids[tt * QSA_WIDTH..tt * QSA_WIDTH + w].copy_from_slice(&cells);
                }
                let gk = g.upload_bf16(&to_bits(&kv));
                let gv = g.upload_bf16(&to_bits(&vv));
                let mut gs = g.alloc_f32(fl::qsa_attend_scratch_words(t));
                let mut go = g.alloc_f32(t * QSA_OUT);
                let mut gq = g.alloc_f32(QAct { m: t, k: QSA_OUT }.words());
                g.qsa_attend_into(
                    &g.upload_f32(&q),
                    &gk,
                    &gv,
                    &g.upload_u32(&ids),
                    &g.upload_f32(&proj),
                    QSA_PROJ,
                    &win(g, pos0, 0),
                    &mut gs,
                    &mut go,
                    Some(&mut gq),
                    t,
                );
                let mut qq = g.alloc_f32(QAct { m: t, k: QSA_OUT }.words());
                g.quantize_act_into(&go, &mut qq, t, QSA_OUT);
                same_bits(&g.download(&gq), &g.download(&qq), "qsa fused int8 output");
                let mut co = c.alloc_f32(t * QSA_OUT);
                let mut cs = c.alloc_f32(1);
                c.qsa_attend_into(
                    &c.upload_f32(&q),
                    &c.upload_f32(&kv),
                    &c.upload_f32(&vv),
                    &c.upload_u32(&ids),
                    &c.upload_f32(&proj),
                    QSA_PROJ,
                    &win(&c, pos0, 0),
                    &mut cs,
                    &mut co,
                    None,
                    t,
                );
                close(
                    &g.download(&go),
                    &c.download(&co),
                    1e-4,
                    &format!("qsa attend t={t} pos0={pos0}"),
                );
            }
        }
    }

    fn qsa_union_vs_cpu<D: ComputeDevice>(g: &D) {
        let c = CpuDevice::new();
        let mut rng = Rng(67);
        let (max_ctx, max_blocks) = (4096, 1024);
        let to_bits =
            |v: &[f32]| -> Vec<u16> { v.iter().map(|x| (x.to_bits() >> 16) as u16).collect() };
        let kv: Vec<f32> = rng
            .vec(max_ctx * QSA_KV * QSA_D, 1.0)
            .iter()
            .map(|&v| fl::bf16_round(v))
            .collect();
        let vv: Vec<f32> = rng
            .vec(max_ctx * QSA_KV * QSA_D, 1.0)
            .iter()
            .map(|&v| fl::bf16_round(v))
            .collect();
        let pooled = rng.vec(max_blocks * IDX_D, 1.0);
        let (gk, gv, gp) = (
            g.upload_bf16(&to_bits(&kv)),
            g.upload_bf16(&to_bits(&vv)),
            g.upload_f32(&pooled),
        );
        let mut un = g.alloc_f32(fl::qsa_union_words(max_blocks, 8));
        for t in TS {
            let q = rng.vec(fl::qsa_q_words(t), 1.0);
            let proj = rng.vec(t * QSA_PROJ, 2.0);
            for &pos0 in &[0usize, 61, 2047, 2049, 3001, 4095 - t] {
                let w = win(g, pos0, 0);
                let gq = g.upload_f32(&q);
                let (mut sc, mut ids) = (g.alloc_f32(t * max_blocks), g.alloc_f32(t * QSA_WIDTH));
                g.qsa_select_union_into(&gp, &gq, &w, &mut sc, &mut ids, &mut un, max_blocks, t);
                let mut s = g.alloc_f32(fl::qsa_union_scratch_words(t));
                let (mut out, mut oq) = (
                    g.alloc_f32(t * QSA_OUT),
                    g.alloc_f32(QAct { m: t, k: QSA_OUT }.words()),
                );
                let gpr = g.upload_f32(&proj);
                g.qsa_attend_union_into(
                    &gq,
                    &gk,
                    &gv,
                    &ids,
                    &un,
                    max_blocks,
                    &gpr,
                    QSA_PROJ,
                    &w,
                    &mut s,
                    &mut out,
                    Some(&mut oq),
                    t,
                );
                let mut co = c.alloc_f32(t * QSA_OUT);
                let mut cs = c.alloc_f32(1);
                let cw = win(&c, pos0, 0);
                let (cq, ck, cv, ci, cp) = (
                    c.upload_f32(&q),
                    c.upload_f32(&kv),
                    c.upload_f32(&vv),
                    c.upload_f32(&g.download(&ids)),
                    c.upload_f32(&proj),
                );
                c.qsa_attend_into(
                    &cq, &ck, &cv, &ci, &cp, QSA_PROJ, &cw, &mut cs, &mut co, None, t,
                );
                close(
                    &g.download(&out),
                    &c.download(&co),
                    1e-4,
                    &format!("qsa union attend t={t} pos0={pos0}"),
                );
                let mut qq = g.alloc_f32(QAct { m: t, k: QSA_OUT }.words());
                g.quantize_act_into(&out, &mut qq, t, QSA_OUT);
                same_bits(&g.download(&oq), &g.download(&qq), "qsa union int8 output");
            }
        }
        // The mask region is left clear for the next use.
        let left = g.download(&un);
        assert!(
            left[..max_blocks].iter().all(|v| v.to_bits() == 0),
            "union masks not cleared"
        );
    }

    #[test]
    fn cuda_qsa_union_vs_cpu() {
        on_gpu(qsa_union_vs_cpu::<CudaComputeDevice>);
    }

    #[test]
    fn cuda_q2_linear_vs_cpu() {
        on_gpu(q2_linear_vs_cpu::<CudaComputeDevice>);
    }

    #[test]
    fn cuda_q4x_linear_vs_cpu() {
        on_gpu(q4x_linear_vs_cpu::<CudaComputeDevice>);
    }

    #[test]
    fn cuda_q8x_linear_vs_cpu() {
        on_gpu(q8x_linear_vs_cpu::<CudaComputeDevice>);
    }

    #[test]
    fn cuda_linear_into_vs_cpu() {
        on_gpu(linear_into_vs_cpu::<CudaComputeDevice>);
    }

    #[test]
    fn cuda_hc_vs_cpu() {
        on_gpu(hc_vs_cpu::<CudaComputeDevice>);
    }

    #[test]
    fn cuda_gdn_vs_cpu() {
        on_gpu(gdn_vs_cpu::<CudaComputeDevice>);
    }

    #[test]
    fn cuda_router_vs_cpu() {
        on_gpu(router_vs_cpu::<CudaComputeDevice>);
    }

    #[test]
    fn cuda_moe_vs_cpu() {
        on_gpu(moe_vs_cpu::<CudaComputeDevice>);
    }

    #[test]
    fn cuda_qsa_prep_vs_cpu() {
        on_gpu(qsa_prep_vs_cpu::<CudaComputeDevice>);
    }

    #[test]
    fn cuda_qsa_select_vs_cpu() {
        on_gpu(qsa_select_vs_cpu::<CudaComputeDevice>);
    }

    #[test]
    fn cuda_qsa_attend_vs_cpu() {
        on_gpu(qsa_attend_vs_cpu::<CudaComputeDevice>);
    }
}
