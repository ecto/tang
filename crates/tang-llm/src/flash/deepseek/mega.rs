//! The DS_MEGA lane: measuring what a step's structure costs (graph nodes, launches, grid
//! barriers) and the persistent dense-glue kernel that replaces a layer's small launches
//! (`mg_glue`, see kernels_mega.cu). Opt-in with `DS_MEGA=1`.

use super::gpu::DsW;
use anyhow::{ensure, Result};
use cudarc::driver::sys;
use tang_moe::gpu::{self, DevBuf, Gpu, Module, Stream};

/// kernels.cu (for its device helpers and `lin_mma`), then kernels_mega.cu.
pub const SRC: &str = concat!(include_str!("kernels.cu"), include_str!("kernels_mega.cu"));

/// Whether the opt-in mega paths are on (`DS_MEGA=1`).
pub fn on() -> bool {
    std::env::var("DS_MEGA").is_ok_and(|v| v == "1")
}

/// `mg_glue` launches so far (enqueued or captured), for tests.
pub static LAUNCHES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// `struct MgArgs` in kernels_mega.cu.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct MgArgs {
    pub h: u64,
    pub h2: u64,
    pub res: u64,
    pub post: u64,
    pub comb: u64,
    pub xo: u64,
    pub hpart: u64,
    pub pre_i: u64,
    pub hin: u64,
    pub scale: u64,
    pub base: u64,
    pub pre_o: u64,
    pub post_o: u64,
    pub comb_o: u64,
    pub xc: u64,
    pub xs: u64,
    pub ring: u64,
    pub all: u64,
    pub qr: u64,
    pub kv: u64,
    pub qc: u64,
    pub qsc: u64,
    pub tab: u64,
    pub p: u64,
    pub q: u64,
    pub part1: u64,
    pub part2: u64,
    pub part3: u64,
    pub cnt1: u64,
    pub cnt2: u64,
    pub cnt3: u64,
    pub gbar: u64,
    pub hbar: u64,
    pub w: DsW,
    pub nw: DsW,
    pub wqa: DsW,
    pub wkv: DsW,
    pub wqb: DsW,
    pub qn: DsW,
    pub kn: DsW,
    pub t: i32,
    pub hc: i32,
    pub d: i32,
    pub iters: i32,
    pub gx: i32,
    pub rd: i32,
    pub win: i32,
    pub ncopy: i32,
    pub eps: f32,
    pub neps: f32,
    pub reps: f32,
    pub qeps: f32,
    pub s1: i32,
    pub sb1: i32,
    pub nb1: i32,
    pub s2: i32,
    pub sb2: i32,
    pub nb2: i32,
    pub s3: i32,
    pub sb3: i32,
    pub nb3: i32,
    pub pf_bytes: i32,
}

/// gpu.rs's `ks_split` (the K-split GEMV's work split), so the partial sums, and with them
/// the results, are those of `Kernels::ks_gemv`.
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

const PART: usize = 1 << 21;
const CNT: usize = 1 << 13;

/// The megakernel's module and persistent scratch.
pub struct Mega {
    _m: Module,
    f: [sys::CUfunction; 2],
    sms: usize,
    part: [DevBuf; 3],
    cnt: [DevBuf; 3],
    bar: DevBuf,
    /// Blocks per SM (`DS_MEGA_BPS`, capped by the occupancy).
    bps: usize,
    pf: i32,
}

impl Mega {
    pub fn new(g: &Gpu) -> Result<Self> {
        let m = g.module(SRC)?;
        let f = [m.func("mg_glue_f2")?, m.func("mg_glue_f3")?];
        let sms = g.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?
            as usize;
        let env = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        Ok(Mega {
            _m: m,
            f,
            sms,
            part: [
                DevBuf::zeroed(PART * 4)?,
                DevBuf::zeroed(PART * 4)?,
                DevBuf::zeroed(PART * 4)?,
            ],
            cnt: [
                DevBuf::zeroed(CNT * 4)?,
                DevBuf::zeroed(CNT * 4)?,
                DevBuf::zeroed(CNT * 4)?,
            ],
            bar: DevBuf::zeroed(64)?,
            bps: env("DS_MEGA_BPS", 2),
            pf: (env("DS_MEGA_PF_MB", 16) << 20) as i32,
        })
    }

    /// Whether `mg_glue` takes these shapes.
    pub fn ok(&self, a: &MgArgs) -> bool {
        let f = a.wqa.fmt;
        let ci = 256 / a.hc.max(1);
        (1..=8).contains(&a.t)
            && (f == 2 || f == 3)
            && a.wkv.fmt == f
            && a.wqb.fmt == f
            && 256 % a.hc == 0
            && ci % 32 == 0
            && a.d % ci == 0
            && a.w.rows <= 80
            && (a.gx as usize) < self.sms
    }

    /// Launch `mg_glue` on `a` (the scratch, barrier and split fields are filled here).
    pub fn launch(&self, s: &Stream, mut a: MgArgs) -> Result<()> {
        ensure!(self.ok(&a), "mg_glue: unsupported shapes");
        let t = a.t as usize;
        let mut smem = 0usize;
        let mut split = |w: &DsW| -> (i32, i32, i32) {
            let (rows, cols) = (w.rows as usize, w.cols as usize);
            let nch = cols / 32;
            let pitch = cols.div_ceil(128) * 128 + 32;
            smem = smem.max(t * pitch + t * nch * 4);
            let (sp, br, mut sb) = ks_split(rows.div_ceil(16), nch, self.sms);
            if sb * t * rows > PART || br > CNT {
                sb = 1;
            }
            (sp as i32, sb as i32, br as i32)
        };
        (a.s1, a.sb1, a.nb1) = split(&a.wqa);
        (a.s2, a.sb2, a.nb2) = split(&a.wkv);
        (a.s3, a.sb3, a.nb3) = split(&a.wqb);
        ensure!(smem <= 48 * 1024, "mg_glue: {smem} B of shared memory");
        [a.part1, a.part2, a.part3] = [self.part[0].ptr, self.part[1].ptr, self.part[2].ptr];
        [a.cnt1, a.cnt2, a.cnt3] = [self.cnt[0].ptr, self.cnt[1].ptr, self.cnt[2].ptr];
        a.gbar = self.bar.ptr;
        a.hbar = self.bar.ptr + 32;
        a.pf_bytes = self.pf;
        let f = self.f[(a.wqa.fmt - 2) as usize];
        let mut per_sm = 0i32;
        gpu::check(
            unsafe { sys::cuOccupancyMaxActiveBlocksPerMultiprocessor(&mut per_sm, f, 256, smem) },
            "occupancy",
        )?;
        ensure!(per_sm >= 1, "mg_glue: not resident");
        let grid = self.sms * self.bps.min(per_sm as usize).max(1);
        LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        unsafe {
            super::obs::launch(
                f,
                (grid as u32, 1, 1),
                (256, 1, 1),
                smem as u32,
                s,
                tang_moe::args![a],
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tang_moe::args;
    use tang_moe::gpu::{Event, Graph};

    fn time(s: &Stream, reps: usize, mut f: impl FnMut() -> Result<()>) -> Result<f32> {
        let (a, b) = (Event::new(true)?, Event::new(true)?);
        f()?;
        s.sync()?;
        let mut best = f32::MAX;
        for _ in 0..reps {
            a.record(s)?;
            f()?;
            b.record(s)?;
            b.sync()?;
            best = best.min(b.since(&a)?);
        }
        Ok(best * 1000.0) // µs
    }

    /// Launch floor on this GPU: µs per node of a captured chain of N dependent trivial
    /// kernels, vs plain stream launches, and a persistent kernel's grid barrier. Run with
    /// `--ignored --nocapture` (one GPU; CUDA_VISIBLE_DEVICES picks it).
    #[test]
    #[ignore]
    fn mega_launch_floor() -> Result<()> {
        let Some(g) = Gpu::for_test() else {
            return Ok(());
        };
        g.bind()?;
        let m = g.module(SRC)?;
        let tick = m.func("mg_tick")?;
        let bar = m.func("mg_barrier_loop")?;
        let sms =
            g.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as u32;
        let s = Stream::new()?;
        let p = DevBuf::zeroed(4 * 4096)?;
        let cnt = DevBuf::zeroed(8)?;
        let src = DevBuf::zeroed(256)?;
        let dst = DevBuf::zeroed(256)?;
        println!("sms {sms}");
        let chain = |s: &Stream, n: usize, nb: u32, memcpy: bool| -> gpu::Result<()> {
            for i in 0..n {
                if memcpy && i % 10 == 9 {
                    gpu::check(
                        unsafe { sys::cuMemcpyDtoDAsync_v2(dst.ptr, src.ptr, 64, s.0) },
                        "memcpy",
                    )?;
                    continue;
                }
                unsafe { gpu::launch(tick, (nb, 1, 1), (128, 1, 1), 0, s, args![p.ptr])? };
            }
            Ok(())
        };
        println!("| N | graph 1-block | graph {sms}-block | graph {sms}-block + memcpy/10 | stream 1-block | stream {sms}-block |");
        println!("|---|---|---|---|---|---|");
        for n in [300usize, 600, 1200, 2400] {
            let mut row = vec![];
            for (nb, mc) in [(1, false), (sms, false), (sms, true)] {
                let gr = Graph::capture(&s, |s| chain(s, n, nb, mc))?;
                let us = time(&s, 20, || Ok(gr.launch(&s)?))?;
                row.push(us / n as f32);
            }
            for nb in [1, sms] {
                let us = time(&s, 10, || Ok(chain(&s, n, nb, false)?))?;
                row.push(us / n as f32);
            }
            println!(
                "| {n} | {} |",
                row.iter()
                    .map(|v| format!("{v:.2}"))
                    .collect::<Vec<_>>()
                    .join(" | ")
            );
        }
        println!("\n| blocks x threads | µs per grid barrier |");
        println!("|---|---|");
        for (nb, th) in [
            (sms, 256u32),
            (sms, 1024),
            (2 * sms, 256),
            (64, 256),
            (8, 256),
        ] {
            let r1 = 1i32;
            let r = 1000i32;
            let one = time(&s, 20, || unsafe {
                Ok(gpu::launch(
                    bar,
                    (nb, 1, 1),
                    (th, 1, 1),
                    0,
                    &s,
                    args![p.ptr, cnt.ptr, r1],
                )?)
            })?;
            let many = time(&s, 20, || unsafe {
                Ok(gpu::launch(
                    bar,
                    (nb, 1, 1),
                    (th, 1, 1),
                    0,
                    &s,
                    args![p.ptr, cnt.ptr, r],
                )?)
            })?;
            println!("| {nb} x {th} | {:.2} |", (many - one) / (r - 1) as f32);
        }
        Ok(())
    }
}
