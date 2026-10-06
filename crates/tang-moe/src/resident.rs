//! Resident mode: the host arena holds only the experts that are *not* in VRAM, so RAM use is
//! `keys − slots` blobs (≈ 16.9 GB for Flash-Next on mew) instead of all 34 GB.
//!
//! A swap is then an exchange: the incoming expert's bytes go from its host slot `H` into the
//! victim's VRAM slot `S`, and the victim's bytes must land in `H`. Neither copy may overwrite
//! a place something is still served from, so each exchange goes through a VRAM scratch slot
//! `X`, one step per window boundary:
//!
//! | stage | copy in flight (side stream) | victim served from | incoming served from |
//! |---|---|---|---|
//! | 1 save | `S → X` | `S` (GPU) | `H` (CPU) |
//! | 2 fill | `H → S` | `X` (GPU) | `H` (CPU) |
//! | 3 write back | `X → H` | `X` (GPU) | `S` (GPU, admitted) |
//! | done | — | `H` (CPU) | `S` (GPU) |
//!
//! The victim leaves its slot as soon as its bytes are safe in scratch (it never stops being
//! servable, which a resident-mode victim can't be, as it has no host copy yet); the incoming
//! expert is admitted when its copy lands. A boundary is a point where no window is in flight;
//! table changes are published on the main stream there, and the side-stream copies issued at
//! that boundary start only after the publish (an event), so no copy ever writes a location a
//! window reads. A swap into a free slot skips stages 1 and 3 and frees its host slot.
//!
//! [`ResidentPolicy`] is the state machine (CPU-only, tested here against a simulated memory);
//! the GPU executes its [`Step`]s.

use crate::policy::{AdaptParams, CachePolicy, Geometry, Swap};

/// Where a key is served from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Loc {
    Slot(u32),
    Scratch(u32),
    Host(u32),
}

impl Loc {
    /// On the GPU (VRAM slot or scratch).
    pub fn on_gpu(self) -> bool {
        !matches!(self, Loc::Host(_))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CopyOp {
    SlotToScratch { slot: u32, scratch: u32 },
    HostToSlot { host: u32, slot: u32 },
    ScratchToHost { scratch: u32, host: u32 },
}

/// One copy to enqueue on the side stream, for exchange `ex`. Report `ex` as landed at a later
/// boundary once the copy has completed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Step {
    pub ex: u32,
    pub op: CopyOp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Save,
    Fill,
    WriteBack,
}

#[derive(Clone, Copy, Debug)]
struct Exchange {
    id: u32,
    swap: Swap,
    host: u32,
    scratch: Option<u32>,
    stage: Stage,
}

pub struct ResidentPolicy {
    pub lfu: CachePolicy,
    loc: Vec<Loc>,
    free_scratch: Vec<u32>,
    free_host: Vec<u32>,
    ex: Vec<Exchange>,
    next_id: u32,
    dirty: Vec<u32>,
}

impl ResidentPolicy {
    /// `ranking` (hottest first) fills the `n_slots` VRAM slots; every other key gets a host
    /// slot, in key order. Returns the policy and each key's initial location (for loading).
    pub fn new(
        geo: Geometry,
        n_slots: usize,
        n_scratch: usize,
        params: AdaptParams,
        ranking: impl IntoIterator<Item = u32>,
    ) -> Self {
        let mut lfu = CachePolicy::new(geo, n_slots, params);
        let fills = lfu.seed(ranking);
        for f in &fills {
            lfu.admit(f);
        }
        lfu.drain_dirty();
        let mut loc = vec![Loc::Host(0); geo.keys()];
        let mut h = 0u32;
        for (k, l) in loc.iter_mut().enumerate() {
            *l = match lfu.slot_of(k as u32) {
                Some(s) => Loc::Slot(s),
                None => {
                    h += 1;
                    Loc::Host(h - 1)
                }
            };
        }
        ResidentPolicy {
            lfu,
            loc,
            free_scratch: (0..n_scratch as u32).rev().collect(),
            free_host: Vec::new(),
            ex: Vec::new(),
            next_id: 0,
            dirty: Vec::new(),
        }
    }

    /// Host slots needed: one per key not seeded into VRAM.
    pub fn host_slots(&self) -> usize {
        self.loc
            .iter()
            .filter(|l| matches!(l, Loc::Host(_)))
            .count()
    }

    pub fn loc(&self, key: u32) -> Loc {
        self.loc[key as usize]
    }

    pub fn locs(&self) -> &[Loc] {
        &self.loc
    }

    pub fn in_flight(&self) -> usize {
        self.ex.len()
    }

    pub fn record(&mut self, keys: &[u32]) {
        self.lfu.record(keys);
    }

    /// Keys whose location changed since the last drain.
    pub fn drain_dirty(&mut self) -> Vec<u32> {
        let mut d = std::mem::take(&mut self.dirty);
        d.sort_unstable();
        d.dedup();
        d
    }

    fn set(&mut self, key: u32, l: Loc) {
        self.loc[key as usize] = l;
        self.dirty.push(key);
    }

    /// Between windows: advance the exchanges in `landed` (ids whose copy completed), then
    /// start new ones if an adaptation is due. Publish [`drain_dirty`](Self::drain_dirty)
    /// before enqueuing the returned copies.
    pub fn boundary(&mut self, landed: &[u32]) -> Vec<Step> {
        let mut steps = Vec::new();
        let mut i = 0;
        while i < self.ex.len() {
            let e = self.ex[i];
            if !landed.contains(&e.id) {
                i += 1;
                continue;
            }
            let victim = e.swap.victim;
            match e.stage {
                Stage::Save => {
                    let x = e.scratch.unwrap();
                    self.set(victim.unwrap(), Loc::Scratch(x));
                    self.ex[i].stage = Stage::Fill;
                    steps.push(Step {
                        ex: e.id,
                        op: CopyOp::HostToSlot {
                            host: e.host,
                            slot: e.swap.slot,
                        },
                    });
                    i += 1;
                }
                Stage::Fill => {
                    self.lfu.admit(&e.swap);
                    self.lfu.drain_dirty();
                    self.set(e.swap.incoming, Loc::Slot(e.swap.slot));
                    match e.scratch {
                        Some(x) => {
                            self.ex[i].stage = Stage::WriteBack;
                            steps.push(Step {
                                ex: e.id,
                                op: CopyOp::ScratchToHost {
                                    scratch: x,
                                    host: e.host,
                                },
                            });
                            i += 1;
                        }
                        None => {
                            self.free_host.push(e.host);
                            self.ex.swap_remove(i);
                        }
                    }
                }
                Stage::WriteBack => {
                    self.set(victim.unwrap(), Loc::Host(e.host));
                    self.lfu.set_busy(victim.unwrap(), false);
                    self.free_scratch.push(e.scratch.unwrap());
                    self.ex.swap_remove(i);
                }
            }
        }

        if self.lfu.end_window() && !self.free_scratch.is_empty() {
            for swap in self.lfu.plan_adapt_n(self.free_scratch.len()) {
                let Loc::Host(host) = self.loc[swap.incoming as usize] else {
                    unreachable!("adapt picked a key already on the GPU")
                };
                let id = self.next_id;
                self.next_id = self.next_id.wrapping_add(1);
                let (stage, scratch, op) = match swap.victim {
                    Some(v) => {
                        // Not a candidate again until its bytes are back on the host.
                        self.lfu.set_busy(v, true);
                        let x = self.free_scratch.pop().unwrap();
                        (
                            Stage::Save,
                            Some(x),
                            CopyOp::SlotToScratch {
                                slot: swap.slot,
                                scratch: x,
                            },
                        )
                    }
                    None => (
                        Stage::Fill,
                        None,
                        CopyOp::HostToSlot {
                            host,
                            slot: swap.slot,
                        },
                    ),
                };
                self.ex.push(Exchange {
                    id,
                    swap,
                    host,
                    scratch,
                    stage,
                });
                steps.push(Step { ex: id, op });
            }
            self.lfu.drain_dirty();
        }
        steps
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::q2cpu::tests::Rng;

    /// Simulated memory: each place holds the key whose bytes are there (or NONE).
    const NONE: u32 = u32::MAX;
    struct Mem {
        slot: Vec<u32>,
        scratch: Vec<u32>,
        host: Vec<u32>,
    }

    impl Mem {
        fn at(&self, l: Loc) -> u32 {
            match l {
                Loc::Slot(s) => self.slot[s as usize],
                Loc::Scratch(x) => self.scratch[x as usize],
                Loc::Host(h) => self.host[h as usize],
            }
        }
        fn dst(op: CopyOp) -> Loc {
            match op {
                CopyOp::SlotToScratch { scratch, .. } => Loc::Scratch(scratch),
                CopyOp::HostToSlot { slot, .. } => Loc::Slot(slot),
                CopyOp::ScratchToHost { host, .. } => Loc::Host(host),
            }
        }
        fn apply(&mut self, op: CopyOp) {
            let v = match op {
                CopyOp::SlotToScratch { slot, .. } => self.slot[slot as usize],
                CopyOp::HostToSlot { host, .. } => self.host[host as usize],
                CopyOp::ScratchToHost { scratch, .. } => self.scratch[scratch as usize],
            };
            match Self::dst(op) {
                Loc::Slot(s) => self.slot[s as usize] = v,
                Loc::Scratch(x) => self.scratch[x as usize] = v,
                Loc::Host(h) => self.host[h as usize] = v,
            }
        }
    }

    fn run(seed: u64, n_scratch: usize) {
        let geo = Geometry {
            layers: 2,
            experts: 32,
            blob_bytes: 1,
        };
        let params = AdaptParams {
            every: 2,
            max_swaps: 6,
            ..AdaptParams::default()
        };
        let mut p = ResidentPolicy::new(geo, 12, n_scratch, params, 0..64);
        let mut mem = Mem {
            slot: vec![NONE; 12],
            scratch: vec![NONE; n_scratch],
            host: vec![NONE; p.host_slots()],
        };
        for k in 0..geo.keys() as u32 {
            match p.loc(k) {
                Loc::Slot(s) => mem.slot[s as usize] = k,
                Loc::Host(h) => mem.host[h as usize] = k,
                Loc::Scratch(_) => unreachable!(),
            }
        }
        let mut rng = Rng(seed);
        // Copies in flight: (exchange, op, boundaries until it lands).
        let mut flight: Vec<(u32, CopyOp, u32)> = Vec::new();
        let mut swaps_done = 0;
        for w in 0..600 {
            // The routing drifts: a hot window of 10 keys moves every 100 windows.
            let base = (w / 100) * 9;
            let keys: Vec<u32> = (0..30)
                .map(|_| ((base + (rng.next() % 10) as usize) % geo.keys()) as u32)
                .collect();
            // A window runs: everything it reads must hold the right bytes, and no in-flight
            // copy may write a location being served.
            for &k in &keys {
                assert_eq!(mem.at(p.loc(k)), k, "window {w}: key {k} at {:?}", p.loc(k));
            }
            for (_, op, _) in &flight {
                let d = Mem::dst(*op);
                assert!(
                    !p.locs().contains(&d),
                    "window {w}: copy {op:?} writes a served location"
                );
            }
            p.record(&keys);
            // Boundary: some copies land.
            let mut landed = Vec::new();
            flight.retain_mut(|(ex, op, left)| {
                if *left == 0 {
                    mem.apply(*op);
                    landed.push(*ex);
                    false
                } else {
                    *left -= 1;
                    true
                }
            });
            let before = p.in_flight();
            let steps = p.boundary(&landed);
            swaps_done += before + steps.len();
            p.drain_dirty();
            for s in steps {
                flight.push((s.ex, s.op, (rng.next() % 3) as u32));
            }
            // After the publish, every key is servable from its location.
            for k in 0..geo.keys() as u32 {
                assert_eq!(mem.at(p.loc(k)), k, "after boundary {w}: key {k}");
            }
            // Locations are distinct.
            let mut ls = p.locs().to_vec();
            ls.sort_by_key(|l| format!("{l:?}"));
            ls.dedup();
            assert_eq!(ls.len(), geo.keys());
        }
        assert!(swaps_done > 50, "only {swaps_done} steps");
        p.lfu.check();
    }

    #[test]
    fn exchanges_never_expose_a_wrong_or_overwritten_blob() {
        for seed in 1..6 {
            run(seed, 4);
        }
    }

    #[test]
    fn works_with_a_single_scratch_slot() {
        run(99, 1);
    }
}
