//! Which experts live in VRAM: the residency table and the decayed-LFU adapter.
//!
//! This is the CPU half of [`ExpertCache`](crate::ExpertCache), with no GPU in it, so it is
//! tested on any machine. The GPU half moves bytes and mirrors the tables to the device; it
//! drives this through four calls:
//!
//! - [`CachePolicy::seed`] fills free slots from a ranking (a routing profile) at load.
//! - [`CachePolicy::record`] counts routed experts after each window.
//! - [`CachePolicy::end_window`] says when to adapt; [`CachePolicy::plan_adapt`] picks swaps.
//!   A victim is non-resident as soon as it is picked, so no window reads a slot being
//!   overwritten. Its incoming expert is admitted with [`CachePolicy::admit`] once its copy
//!   has landed.
//! - [`CachePolicy::drain_dirty`] lists keys whose device entries must be rewritten.

/// Shape of the expert set. A key is `layer * experts + expert`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub layers: usize,
    pub experts: usize,
    /// Bytes of one (layer, expert) blob, and of one VRAM slot.
    pub blob_bytes: usize,
}

impl Geometry {
    /// Qwen3.8-Flash-Next at Q2_0: 48 layers × 512 experts, 1,382,400 B per blob
    /// (gate+up 1280 × 2560 and down 2560 × 640, 64 weights in 18 B).
    pub const FLASH_NEXT_Q2_0: Geometry = Geometry {
        layers: 48,
        experts: 512,
        blob_bytes: 1_382_400,
    };

    pub fn keys(&self) -> usize {
        self.layers * self.experts
    }

    pub fn key(&self, layer: usize, expert: usize) -> u32 {
        debug_assert!(layer < self.layers && expert < self.experts);
        (layer * self.experts + expert) as u32
    }
}

/// Decayed-LFU parameters (Strata's defaults).
#[derive(Clone, Copy, Debug)]
pub struct AdaptParams {
    /// Adapt every this many windows (0 = never).
    pub every: u32,
    /// At most this many swaps per adaptation.
    pub max_swaps: usize,
    /// A candidate needs at least this decayed count.
    pub threshold: f32,
    /// ...and must beat its victim's count by this factor.
    pub margin: f32,
    /// Counts are multiplied by this after each adaptation.
    pub decay: f32,
}

impl Default for AdaptParams {
    fn default() -> Self {
        AdaptParams {
            every: 4,
            max_swaps: 96,
            threshold: 2.0,
            margin: 1.5,
            decay: 0.7,
        }
    }
}

/// One planned move: `incoming`'s blob goes into `slot`. `victim` is the key that held the slot
/// (`None` for a slot that was free).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Swap {
    pub incoming: u32,
    pub victim: Option<u32>,
    pub slot: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Slot {
    Free,
    /// Being filled with this key; not readable yet.
    Filling(u32),
    Resident(u32),
}

/// Hit and miss counts since creation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Stats {
    pub hits: u64,
    pub misses: u64,
    pub swaps: u64,
    pub adaptations: u64,
}

impl Stats {
    pub fn hit_rate(&self) -> f64 {
        let n = self.hits + self.misses;
        if n == 0 {
            0.0
        } else {
            self.hits as f64 / n as f64
        }
    }
}

pub struct CachePolicy {
    geo: Geometry,
    params: AdaptParams,
    /// key → slot, or -1. This is what the device table mirrors.
    resident: Vec<i32>,
    slots: Vec<Slot>,
    /// key is the target of an in-flight fill.
    filling: Vec<bool>,
    /// key may not be picked as a candidate (e.g. a resident-mode victim still moving out).
    busy: Vec<bool>,
    usage: Vec<f32>,
    windows: u32,
    dirty: Vec<u32>,
    stats: Stats,
}

impl CachePolicy {
    pub fn new(geo: Geometry, n_slots: usize, params: AdaptParams) -> Self {
        CachePolicy {
            geo,
            params,
            resident: vec![-1; geo.keys()],
            slots: vec![Slot::Free; n_slots],
            filling: vec![false; geo.keys()],
            busy: vec![false; geo.keys()],
            usage: vec![0.0; geo.keys()],
            windows: 0,
            dirty: Vec::new(),
            stats: Stats::default(),
        }
    }

    pub fn geometry(&self) -> Geometry {
        self.geo
    }

    pub fn n_slots(&self) -> usize {
        self.slots.len()
    }

    pub fn stats(&self) -> Stats {
        self.stats
    }

    /// The host copy of the residency table: key → slot or -1.
    pub fn residency(&self) -> &[i32] {
        &self.resident
    }

    pub fn slot_of(&self, key: u32) -> Option<u32> {
        let s = self.resident[key as usize];
        (s >= 0).then_some(s as u32)
    }

    pub fn usage(&self, key: u32) -> f32 {
        self.usage[key as usize]
    }

    /// Claim free slots for keys in `ranking` order (hottest first), skipping keys already
    /// resident or filling. Returns the fills; each must be [`admit`](Self::admit)ted once
    /// its bytes are in the slot.
    pub fn seed(&mut self, ranking: impl IntoIterator<Item = u32>) -> Vec<Swap> {
        let free: Vec<u32> = (0..self.slots.len() as u32)
            .filter(|&s| self.slots[s as usize] == Slot::Free)
            .collect();
        let mut free = free.into_iter();
        let mut out = Vec::new();
        for key in ranking {
            if self.resident[key as usize] >= 0 || self.filling[key as usize] {
                continue;
            }
            let Some(slot) = free.next() else { break };
            self.slots[slot as usize] = Slot::Filling(key);
            self.filling[key as usize] = true;
            out.push(Swap {
                incoming: key,
                victim: None,
                slot,
            });
        }
        out
    }

    /// The copy for `swap` has landed: its key becomes resident.
    pub fn admit(&mut self, swap: &Swap) {
        let slot = swap.slot as usize;
        assert_eq!(
            self.slots[slot],
            Slot::Filling(swap.incoming),
            "admit of a slot that is not filling with this key"
        );
        self.slots[slot] = Slot::Resident(swap.incoming);
        self.filling[swap.incoming as usize] = false;
        self.resident[swap.incoming as usize] = slot as i32;
        self.dirty.push(swap.incoming);
    }

    /// Count one window's routed experts (keys may repeat across tokens; each occurrence
    /// counts, as each one is a read). Updates hit/miss stats against current residency.
    pub fn record(&mut self, keys: &[u32]) {
        for &k in keys {
            self.usage[k as usize] += 1.0;
            if self.resident[k as usize] >= 0 {
                self.stats.hits += 1;
            } else {
                self.stats.misses += 1;
            }
        }
    }

    /// Marks the end of a window. True when an adaptation is due.
    pub fn end_window(&mut self) -> bool {
        self.windows += 1;
        self.params.every > 0 && self.windows.is_multiple_of(self.params.every)
    }

    /// Pick up to `max_swaps` swaps: the most-used non-resident keys (count ≥ threshold) take
    /// free slots first, then the least-used resident keys' slots, while the candidate's count
    /// is ≥ margin × the victim's. Victims become non-resident now. Then decay all counts.
    pub fn plan_adapt(&mut self) -> Vec<Swap> {
        self.plan_adapt_n(self.params.max_swaps)
    }

    /// [`plan_adapt`](Self::plan_adapt) with at most `max` swaps (≤ `max_swaps`).
    pub fn plan_adapt_n(&mut self, max: usize) -> Vec<Swap> {
        let p = self.params;
        let mut cands: Vec<u32> = (0..self.geo.keys() as u32)
            .filter(|&k| {
                self.resident[k as usize] < 0
                    && !self.filling[k as usize]
                    && !self.busy[k as usize]
                    && self.usage[k as usize] >= p.threshold
            })
            .collect();
        cands.sort_by(|a, b| {
            self.usage[*b as usize]
                .total_cmp(&self.usage[*a as usize])
                .then(a.cmp(b))
        });
        cands.truncate(p.max_swaps.min(max));

        let mut free: Vec<u32> = (0..self.slots.len() as u32)
            .filter(|&s| self.slots[s as usize] == Slot::Free)
            .collect();
        free.reverse(); // pop() hands out the lowest slot first
        let mut victims: Vec<(u32, u32)> = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(s, st)| match st {
                Slot::Resident(k) => Some((*k, s as u32)),
                _ => None,
            })
            .collect();
        victims.sort_by(|a, b| {
            self.usage[a.0 as usize]
                .total_cmp(&self.usage[b.0 as usize])
                .then(a.0.cmp(&b.0))
        });
        let mut victims = victims.into_iter();

        let mut out = Vec::new();
        for c in cands {
            let cu = self.usage[c as usize];
            if let Some(slot) = free.pop() {
                out.push(Swap {
                    incoming: c,
                    victim: None,
                    slot,
                });
                continue;
            }
            // Candidates descend and victims ascend, so the first failure ends the pairing.
            let Some((v, slot)) = victims.next() else {
                break;
            };
            let vu = self.usage[v as usize];
            if !(cu >= p.margin * vu && cu > vu) {
                break;
            }
            out.push(Swap {
                incoming: c,
                victim: Some(v),
                slot,
            });
        }

        for s in &out {
            if let Some(v) = s.victim {
                self.resident[v as usize] = -1;
                self.dirty.push(v);
            }
            self.slots[s.slot as usize] = Slot::Filling(s.incoming);
            self.filling[s.incoming as usize] = true;
        }
        for u in &mut self.usage {
            *u *= p.decay;
        }
        self.stats.swaps += out.len() as u64;
        self.stats.adaptations += 1;
        out
    }

    /// Exclude (or re-admit) `key` as a swap candidate.
    pub fn set_busy(&mut self, key: u32, busy: bool) {
        self.busy[key as usize] = busy;
    }

    /// Keys whose residency changed since the last drain (deduplicated, ascending).
    pub fn drain_dirty(&mut self) -> Vec<u32> {
        let mut d = std::mem::take(&mut self.dirty);
        d.sort_unstable();
        d.dedup();
        d
    }

    /// Check internal consistency (tests and debug builds).
    pub fn check(&self) {
        let mut seen = vec![false; self.geo.keys()];
        for (s, st) in self.slots.iter().enumerate() {
            match *st {
                Slot::Resident(k) => {
                    assert_eq!(self.resident[k as usize], s as i32, "slot {s} key {k}");
                    assert!(!seen[k as usize], "key {k} in two slots");
                    seen[k as usize] = true;
                }
                Slot::Filling(k) => {
                    assert!(self.filling[k as usize]);
                    assert_eq!(self.resident[k as usize], -1);
                    assert!(!seen[k as usize], "key {k} in two slots");
                    seen[k as usize] = true;
                }
                Slot::Free => {}
            }
        }
        for (k, &s) in self.resident.iter().enumerate() {
            if s >= 0 {
                assert_eq!(self.slots[s as usize], Slot::Resident(k as u32));
            }
        }
    }
}

/// Per-(layer, expert) device address of a blob: its VRAM slot if resident, else its
/// device-mapped host address. Kernels do one load from this table and don't care where the
/// blob lives.
pub struct PointerTable {
    /// Device-visible address of each key's host copy (0 if it has none).
    pub host: Vec<u64>,
    /// Device address of slot 0; slots are `slot_bytes` apart.
    pub vram_base: u64,
    pub slot_bytes: u64,
}

impl PointerTable {
    pub fn entry(&self, key: u32, slot: i32) -> u64 {
        if slot >= 0 {
            self.vram_base + slot as u64 * self.slot_bytes
        } else {
            self.host[key as usize]
        }
    }

    /// The whole table for a residency vector.
    pub fn build(&self, residency: &[i32]) -> Vec<u64> {
        residency
            .iter()
            .enumerate()
            .map(|(k, &s)| self.entry(k as u32, s))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GEO: Geometry = Geometry {
        layers: 2,
        experts: 8,
        blob_bytes: 64,
    };

    fn admit_all(p: &mut CachePolicy, swaps: &[Swap]) {
        for s in swaps {
            p.admit(s);
        }
    }

    #[test]
    fn seed_fills_slots_in_rank_order() {
        let mut p = CachePolicy::new(GEO, 3, AdaptParams::default());
        let fills = p.seed([5, 5, 9, 2, 7]);
        assert_eq!(
            fills
                .iter()
                .map(|s| (s.incoming, s.slot))
                .collect::<Vec<_>>(),
            vec![(5, 0), (9, 1), (2, 2)]
        );
        // Filling keys are not resident yet.
        assert_eq!(p.slot_of(5), None);
        admit_all(&mut p, &fills);
        assert_eq!(p.slot_of(9), Some(1));
        assert_eq!(p.drain_dirty(), vec![2, 5, 9]);
        assert!(p.drain_dirty().is_empty());
        p.check();
    }

    #[test]
    fn record_counts_hits_and_misses() {
        let mut p = CachePolicy::new(GEO, 2, AdaptParams::default());
        let f = p.seed([1, 2]);
        admit_all(&mut p, &f);
        p.record(&[1, 1, 2, 3]);
        assert_eq!(p.stats().hits, 3);
        assert_eq!(p.stats().misses, 1);
        assert_eq!(p.usage(1), 2.0);
    }

    #[test]
    fn adapt_every_n_windows() {
        let mut p = CachePolicy::new(GEO, 2, AdaptParams::default());
        let due: Vec<bool> = (0..8).map(|_| p.end_window()).collect();
        assert_eq!(due, [false, false, false, true, false, false, false, true]);
    }

    #[test]
    fn adapt_swaps_hot_misses_for_cold_residents() {
        let mut p = CachePolicy::new(GEO, 2, AdaptParams::default());
        let f = p.seed([0, 1]);
        admit_all(&mut p, &f);
        p.drain_dirty();
        // 0 is hot, 1 is cold, 4 is hot and missing, 6 is below threshold.
        p.record(&[0; 10]);
        p.record(&[1]);
        p.record(&[4; 5]);
        p.record(&[6]);
        let swaps = p.plan_adapt();
        assert_eq!(
            swaps,
            vec![Swap {
                incoming: 4,
                victim: Some(1),
                slot: 1
            }]
        );
        // The victim is out at once; the incoming key is not in until admitted.
        assert_eq!(p.slot_of(1), None);
        assert_eq!(p.slot_of(4), None);
        assert_eq!(p.drain_dirty(), vec![1]);
        p.check();
        p.admit(&swaps[0]);
        assert_eq!(p.slot_of(4), Some(1));
        assert_eq!(p.drain_dirty(), vec![4]);
        // Counts decayed by 0.7.
        assert!((p.usage(0) - 7.0).abs() < 1e-5);
        p.check();
    }

    #[test]
    fn margin_blocks_marginal_swaps() {
        let mut p = CachePolicy::new(GEO, 1, AdaptParams::default());
        let f = p.seed([0]);
        admit_all(&mut p, &f);
        p.record(&[0; 4]);
        p.record(&[3; 5]); // 5 < 1.5 × 4
        assert!(p.plan_adapt().is_empty());
        assert_eq!(p.slot_of(0), Some(0));
        // After decay: 0 → 2.8, 3 → 3.5. Add 2 to 3: 5.5 ≥ 4.2.
        p.record(&[3; 2]);
        let s = p.plan_adapt();
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].victim, Some(0));
    }

    #[test]
    fn free_slots_are_used_before_eviction_and_swaps_capped() {
        let params = AdaptParams {
            max_swaps: 3,
            ..AdaptParams::default()
        };
        let mut p = CachePolicy::new(GEO, 4, params);
        let f = p.seed([0, 1]);
        admit_all(&mut p, &f);
        for k in 8..14 {
            p.record(&[k; 10]);
        }
        let s = p.plan_adapt();
        assert_eq!(s.len(), 3);
        assert_eq!(s[0].victim, None);
        assert_eq!(s[1].victim, None);
        assert!(s[2].victim.is_some());
        p.check();
    }

    #[test]
    fn filling_keys_are_not_candidates_again() {
        let mut p = CachePolicy::new(GEO, 2, AdaptParams::default());
        p.record(&[5; 10]);
        let s1 = p.plan_adapt();
        assert_eq!(s1.len(), 1);
        p.record(&[5; 10]);
        assert!(p.plan_adapt().is_empty());
        p.admit(&s1[0]);
        p.check();
    }

    #[test]
    fn hot_set_converges() {
        // Skewed routing over 16 keys with 6 slots: the cache ends full and holding the top 4
        // (5 and 6 are within the 1.5 margin of each other, so either may stay).
        let mut p = CachePolicy::new(GEO, 6, AdaptParams::default());
        let mut rng = 0x9e37_79b9_u32;
        let mut inflight: Vec<Swap> = Vec::new();
        for _ in 0..400 {
            let mut keys = Vec::new();
            for _ in 0..20 {
                rng ^= rng << 13;
                rng ^= rng >> 17;
                rng ^= rng << 5;
                let u = (rng % 1000) as f32 / 1000.0;
                keys.push(((u * u * u) * 16.0) as u32);
            }
            p.record(&keys);
            for s in inflight.drain(..) {
                p.admit(&s);
            }
            if p.end_window() {
                inflight = p.plan_adapt();
            }
            p.check();
        }
        for s in inflight.drain(..) {
            p.admit(&s);
        }
        let res: Vec<u32> = (0..16).filter(|&k| p.slot_of(k).is_some()).collect();
        assert_eq!(res.len(), 6, "{res:?}");
        assert!((0..4).all(|k| res.contains(&k)), "{res:?}");
    }

    #[test]
    fn pointer_table_picks_slot_or_host() {
        let t = PointerTable {
            host: (0..4).map(|k| 0x7000_0000 + k * 0x100).collect(),
            vram_base: 0x1000,
            slot_bytes: 0x40,
        };
        assert_eq!(
            t.build(&[-1, 2, -1, 0]),
            vec![0x7000_0000, 0x1080, 0x7000_0200, 0x1000]
        );
    }
}
