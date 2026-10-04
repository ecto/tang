//! Serve one layer's missed experts on the CPU.
//!
//! [`build_plan`] dedupes a window's `t × TOPK` routed ids into distinct experts, writes the
//! GPU's [`MoePlan`] for the resident ones and returns the missed ones with their token lists.
//! [`MissExec::run`] computes the missed experts, two-phase and row-split across the pool:
//!
//! - **Phase A:** every gate/up row pair of every missed expert, split into ~3 ranges per
//!   thread. Each 32-byte weight load is dotted against all of that expert's tokens. Writes
//!   `h = silu(g) · u` (fp32).
//! - Quantize each `h` row (640) per the [`QAct`] contract (calling thread).
//! - **Phase B:** every down row of every missed expert, split the same way. Writes fp32 rows
//!   (not router-weighted) to `out[dst · HIDDEN ..]`, one per (expert, token).

use std::time::Instant;

use crate::contract::{silu, ExpertBlob, MoePlan, QAct, FF, HIDDEN, MAX_T, TOPK};
use crate::pool::Pool;
use crate::q2cpu::{rows_in, Isa, Layout, XPrep};

/// A missed expert and the routed occurrences it must serve.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Missed {
    /// Expert id within the layer.
    pub expert: u32,
    /// `(token, parts row)` per occurrence, in routing order; the row is `token · TOPK + rank`.
    pub toks: Vec<(usize, usize)>,
}

/// Build the plan for router ids `[t][TOPK]` into `plan` (`MoePlan::WORDS`), given each
/// expert's device address (`addr(e)`, 0 = not resident) and an optional shared expert, and
/// return the missed experts in order of first appearance. Matches the kernel track's
/// `moe_plan` word for word.
pub fn build_plan(
    ids: &[u32],
    t: usize,
    addr: impl Fn(u32) -> u64,
    shared: u64,
    plan: &mut [u32],
) -> Vec<Missed> {
    let n = t * TOPK;
    assert!(t <= MAX_T && ids.len() >= n && plan.len() >= MoePlan::WORDS);
    plan[..MoePlan::WORDS].fill(0);
    let mut seen: Vec<u32> = Vec::with_capacity(n);
    let mut groups: Vec<(u64, u32)> = Vec::new();
    let mut missed: Vec<Missed> = Vec::new();
    for &e in &ids[..n] {
        if seen.contains(&e) {
            continue;
        }
        seen.push(e);
        let p = addr(e);
        if p == 0 {
            missed.push(Missed {
                expert: e,
                toks: (0..n)
                    .filter(|&j| ids[j] == e)
                    .map(|j| (j / TOPK, j))
                    .collect(),
            });
        } else {
            groups.push((p, e));
        }
    }
    let mut ne = 0usize;
    let mut put_group =
        |g: usize, p: u64, ents: &mut dyn Iterator<Item = (usize, usize)>, plan: &mut [u32]| {
            plan[MoePlan::GROUP_PTR + 2 * g] = p as u32;
            plan[MoePlan::GROUP_PTR + 2 * g + 1] = (p >> 32) as u32;
            plan[MoePlan::GROUP_START + g] = ne as u32;
            for (tok, dst) in ents {
                plan[MoePlan::ENT_TOK + ne] = tok as u32;
                plan[MoePlan::ENT_DST + ne] = dst as u32;
                ne += 1;
            }
        };
    for (g, &(p, e)) in groups.iter().enumerate() {
        let mut it = (0..n).filter(|&j| ids[j] == e).map(|j| (j / TOPK, j));
        put_group(g, p, &mut it, plan);
    }
    let mut ng = groups.len();
    if shared != 0 {
        let mut it = (0..t).map(|tt| (tt, MoePlan::SHARED_ROW + tt));
        put_group(ng, shared, &mut it, plan);
        ng += 1;
    }
    plan[0] = ng as u32;
    plan[1] = ne as u32;
    plan[2] = missed.len() as u32;
    plan[MoePlan::GROUP_START + ng] = ne as u32;
    for (i, m) in missed.iter().enumerate() {
        plan[MoePlan::MISSING + i] = m.expert;
    }
    missed
}

/// Where a missed expert's weights are, for [`MissExec::run`].
pub struct MissJob<'a> {
    pub blob: &'a [u8],
    pub toks: &'a [(usize, usize)],
}

/// Timings of one [`MissExec::run`], in µs.
#[derive(Clone, Copy, Debug, Default)]
pub struct MissTimes {
    pub prep_us: f64,
    pub phase_a_us: f64,
    pub quant_us: f64,
    pub phase_b_us: f64,
}

impl MissTimes {
    pub fn total(&self) -> f64 {
        self.prep_us + self.phase_a_us + self.quant_us + self.phase_b_us
    }
}

/// Max distinct missed experts per call (every routed slot of a full window).
pub const MAX_MISSED: usize = MAX_T * TOPK;

struct SendPtr<T>(*mut T);
impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SendPtr<T> {}
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
impl<T> SendPtr<T> {
    fn get(self) -> *mut T {
        self.0
    }
}

pub struct MissExec {
    pub pool: Pool,
    pub isa: Isa,
    /// Ranges per thread per phase.
    pub split: usize,
    /// Blobs are tang-compute's tiled `flash::ExpertBlob` (the GPU kernels' layout, so one
    /// copy serves VRAM slots and the host arena) instead of `contract::ExpertBlob`.
    pub tiled: bool,
    x: Vec<XPrep>,
    h: Vec<f32>,
    hprep: Vec<XPrep>,
}

impl MissExec {
    pub fn new(pool: Pool, isa: Isa) -> Self {
        MissExec {
            pool,
            isa,
            split: 3,
            tiled: false,
            x: (0..MAX_T).map(|_| XPrep::new(HIDDEN)).collect(),
            h: vec![0.0; MAX_MISSED * MAX_T * FF],
            hprep: (0..MAX_MISSED * MAX_T).map(|_| XPrep::new(FF)).collect(),
        }
    }

    /// Compute `jobs` for the `t` tokens of `xq` (`QAct { m: t, k: HIDDEN }`), writing row
    /// `dst` of `out` (`HIDDEN` floats per row) for every `(token, dst)`.
    ///
    /// # Safety
    /// `out` must be valid for writes of `(max dst + 1) · HIDDEN` floats, and nothing else may
    /// access those rows during the call.
    pub unsafe fn run(
        &mut self,
        xq: &[u32],
        t: usize,
        jobs: &[MissJob],
        out: *mut f32,
    ) -> MissTimes {
        let mut tm = MissTimes::default();
        if jobs.is_empty() {
            return tm;
        }
        assert!(jobs.len() <= MAX_MISSED && t <= MAX_T);
        let t0 = Instant::now();
        let lx = QAct { m: t, k: HIDDEN };
        let mut used = [false; MAX_T];
        for j in jobs {
            assert!(
                j.blob.len() >= ExpertBlob::BYTES && !j.toks.is_empty() && j.toks.len() <= MAX_T
            );
            for &(tok, _) in j.toks {
                used[tok] = true;
            }
        }
        for (tok, &u) in used.iter().enumerate().take(t) {
            if u {
                self.x[tok].load(xq, lx, tok);
            }
        }
        tm.prep_us = t0.elapsed().as_secs_f64() * 1e6;

        let threads = self.pool.threads();
        let isa = self.isa;
        let tiled = self.tiled;
        let nj = jobs.len();
        let ranges = |total: usize| (self.split * threads).min(total.div_ceil(8)).max(1);

        // Phase A: gate and up rows, h = silu(g)·u.
        let t1 = Instant::now();
        let total_a = nj * FF;
        let na = ranges(total_a);
        let xs = &self.x;
        let h = SendPtr(self.h.as_mut_ptr());
        self.pool.run(na, &|r| {
            let (lo, hi) = (r * total_a / na, (r + 1) * total_a / na);
            let mut row = lo;
            while row < hi {
                let (e, r0) = (row / FF, row % FF);
                let r1 = (r0 + (hi - row)).min(FF).min(r0 + 32);
                let j = &jobs[e];
                let nt = j.toks.len();
                let mut tk: [&XPrep; MAX_T] = [&xs[0]; MAX_T];
                for (i, &(tok, _)) in j.toks.iter().enumerate() {
                    tk[i] = &xs[tok];
                }
                let toks = &tk[..nt];
                let (mut g, mut u) = ([0f32; 32 * MAX_T], [0f32; 32 * MAX_T]);
                let (gl, gw, ul, uw) = if tiled {
                    (
                        Layout::GuTiled { up: false },
                        &j.blob[..],
                        Layout::GuTiled { up: true },
                        &j.blob[..],
                    )
                } else {
                    (
                        Layout::Plain,
                        &j.blob[ExpertBlob::GATE..ExpertBlob::UP],
                        Layout::Plain,
                        &j.blob[ExpertBlob::UP..ExpertBlob::DOWN],
                    )
                };
                rows_in(isa, gl, gw, FF, HIDDEN, r0, r1, toks, &mut g);
                rows_in(isa, ul, uw, FF, HIDDEN, r0, r1, toks, &mut u);
                for rr in r0..r1 {
                    for ti in 0..nt {
                        let v = silu(g[(rr - r0) * nt + ti]) * u[(rr - r0) * nt + ti];
                        // SAFETY: (e, ti, rr) is written by exactly one range.
                        unsafe { *h.get().add((e * MAX_T + ti) * FF + rr) = v };
                    }
                }
                row += r1 - r0;
            }
        });
        tm.phase_a_us = t1.elapsed().as_secs_f64() * 1e6;

        // Quantize the intermediates, one job per (expert, token).
        let t2 = Instant::now();
        let pairs: Vec<(usize, usize)> = jobs
            .iter()
            .enumerate()
            .flat_map(|(e, j)| (0..j.toks.len()).map(move |ti| (e, ti)))
            .collect();
        let hsrc = &self.h;
        let hp = SendPtr(self.hprep.as_mut_ptr());
        self.pool.run(pairs.len(), &|i| {
            let (e, ti) = pairs[i];
            let row = &hsrc[(e * MAX_T + ti) * FF..(e * MAX_T + ti + 1) * FF];
            // SAFETY: each (e, ti) slot is written by one job.
            unsafe { (*hp.get().add(e * MAX_T + ti)).quantize(row) };
        });
        tm.quant_us = t2.elapsed().as_secs_f64() * 1e6;

        // Phase B: down rows into out[dst].
        let t3 = Instant::now();
        let total_b = nj * HIDDEN;
        let nb = ranges(total_b);
        let hp = &self.hprep;
        let out = SendPtr(out);
        self.pool.run(nb, &|r| {
            let (lo, hi) = (r * total_b / nb, (r + 1) * total_b / nb);
            let mut row = lo;
            while row < hi {
                let (e, r0) = (row / HIDDEN, row % HIDDEN);
                let r1 = (r0 + (hi - row)).min(HIDDEN).min(r0 + 64);
                let j = &jobs[e];
                let nt = j.toks.len();
                let mut hk: [&XPrep; MAX_T] = [&hp[0]; MAX_T];
                for (ti, h) in hk.iter_mut().enumerate().take(nt) {
                    *h = &hp[e * MAX_T + ti];
                }
                let hs = &hk[..nt];
                let mut y = [0f32; 64 * MAX_T];
                let (dl, dw) = if tiled {
                    (Layout::DownTiled, &j.blob[..])
                } else {
                    (Layout::Plain, &j.blob[ExpertBlob::DOWN..ExpertBlob::BYTES])
                };
                rows_in(isa, dl, dw, HIDDEN, FF, r0, r1, hs, &mut y);
                for (ti, &(_, dst)) in j.toks.iter().enumerate() {
                    for rr in r0..r1 {
                        // SAFETY: caller guarantees the rows; (dst, rr) has one writer.
                        unsafe { *out.get().add(dst * HIDDEN + rr) = y[(rr - r0) * nt + ti] };
                    }
                }
                row += r1 - r0;
            }
        });
        tm.phase_b_us = t3.elapsed().as_secs_f64() * 1e6;
        tm
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::expert_ref;
    use crate::q2cpu::tests::{acts, blob, Rng};
    use std::time::Duration;

    #[test]
    fn plan_dedupes_and_lists_misses() {
        // t = 2; expert 7 routed by both tokens and missing; 3 resident (both tokens).
        let mut ids = vec![0u32; 2 * TOPK];
        for (j, id) in ids.iter_mut().enumerate() {
            *id = 100 + j as u32;
        }
        ids[0] = 7;
        ids[TOPK + 4] = 7;
        ids[1] = 3;
        ids[TOPK] = 3;
        let addr = |e: u32| {
            if e == 7 || e >= 118 {
                0
            } else {
                0x1000 + e as u64
            }
        };
        let mut plan = vec![0u32; MoePlan::WORDS];
        let missed = build_plan(&ids, 2, addr, 0xbeef, &mut plan);
        assert_eq!(
            missed[0],
            Missed {
                expert: 7,
                toks: vec![(0, 0), (1, TOPK + 4)]
            }
        );
        assert_eq!(
            missed.iter().map(|m| m.expert).collect::<Vec<_>>(),
            vec![7, 118, 119]
        );
        assert_eq!(plan[2], 3);
        // Group 0 is expert 3 with entries (0,1) and (1,10).
        assert_eq!(plan[MoePlan::GROUP_PTR], 0x1003);
        assert_eq!(plan[MoePlan::ENT_TOK], 0);
        assert_eq!(plan[MoePlan::ENT_DST], 1);
        assert_eq!(plan[MoePlan::ENT_TOK + 1], 1);
        assert_eq!(plan[MoePlan::ENT_DST + 1], TOPK as u32);
        let ng = plan[0] as usize;
        // Shared expert last, one entry per token.
        assert_eq!(plan[MoePlan::GROUP_PTR + 2 * (ng - 1)], 0xbeef);
        assert_eq!(plan[MoePlan::GROUP_START + ng], plan[1]);
        let routed_resident = 2 * TOPK - 2 - 2; // minus 7 twice, 118, 119
        assert_eq!(plan[1] as usize, routed_resident + 2);
    }

    #[test]
    fn exec_matches_reference_expert() {
        let mut rng = Rng(3);
        let blobs: Vec<Vec<u8>> = (0..3).map(|_| blob(&mut rng)).collect();
        let t = 3;
        let xq = acts(&mut rng, t, HIDDEN);
        let toks: Vec<Vec<(usize, usize)>> = vec![
            vec![(0, 0), (2, 2 * TOPK + 5)],
            vec![(1, TOPK + 1)],
            vec![(0, 3), (1, TOPK + 3), (2, 2 * TOPK)],
        ];
        let jobs: Vec<MissJob> = blobs
            .iter()
            .zip(&toks)
            .map(|(b, t)| MissJob { blob: b, toks: t })
            .collect();
        let n = std::thread::available_parallelism()
            .map_or(2, |n| n.get())
            .min(4);
        for isa in Isa::available() {
            let pool = Pool::new(&(0..n).collect::<Vec<_>>(), Duration::from_millis(2));
            let mut ex = MissExec::new(pool, isa);
            let mut out = vec![f32::NAN; MoePlan::PARTS_ROWS * HIDDEN];
            unsafe { ex.run(&xq, t, &jobs, out.as_mut_ptr()) };
            for (b, tl) in blobs.iter().zip(&toks) {
                for &(tok, dst) in tl {
                    let want = expert_ref(b, &xq, t, tok);
                    let got = &out[dst * HIDDEN..(dst + 1) * HIDDEN];
                    let scale = want.iter().fold(0f32, |a, v| a.max(v.abs()));
                    // h is requantized, so a near-tie in rounding one intermediate code can move
                    // an output by about one int8 step of h times a weight: allow 1% of max |y|.
                    let bad = got
                        .iter()
                        .zip(&want)
                        .filter(|(g, w)| (*g - *w).abs() > 0.01 * scale)
                        .count();
                    assert!(bad == 0, "{isa:?}: {bad} rows off (tok {tok} dst {dst})");
                }
            }
        }
    }
}
