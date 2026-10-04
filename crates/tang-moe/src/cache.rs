//! The VRAM expert cache: one arena of blob-sized slots, the residency and pointer tables on
//! the device, and swaps on a side stream.
//!
//! Ordering, so a window never reads a slot that is being overwritten:
//!
//! - All device-table writes happen on the *main* stream, between windows
//!   ([`ExpertCache::between_windows`]). Windows (captured graphs) only read the tables.
//! - A swap's victim is pointed back at its host copy in that update. The side stream waits on
//!   an event recorded after the update, then copies the incoming blob into the slot. Windows
//!   enqueued later run concurrently with the copy but no longer reference the slot.
//! - When the copy's event has completed, the next `between_windows` admits the incoming key
//!   and points its entry at the slot.
//!
//! This is the "full arena" mode: every expert has a host copy in a registered
//! [`HostArena`]. Resident mode (the arena holds only experts not in VRAM) turns each swap
//! into an exchange (slot → scratch → victim's new host place) and is not built yet.

use std::collections::VecDeque;
use std::ffi::c_void;

use cudarc::driver::sys;

use crate::arena::HostArena;
use crate::args;
use crate::gpu::{ck, launch, DevBuf, Event, Gpu, Module, Result, Stream};
use crate::kernels;
use crate::policy::{AdaptParams, CachePolicy, Geometry, PointerTable, Swap};

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Upd {
    key: u32,
    slot: i32,
    ptr: u64,
}

/// How [`ExpertCache::new`] sized the slot arena.
#[derive(Clone, Copy, Debug)]
pub struct SizeReport {
    pub free_before: usize,
    pub reserve: usize,
    pub slots_first: usize,
    pub free_after_touch: usize,
    pub slots: usize,
    pub free_final: usize,
}

pub struct ExpertCache {
    pub policy: CachePolicy,
    table: PointerTable,
    /// Host virtual address of each key's blob (for copies).
    host_src: Vec<usize>,
    vram: DevBuf,
    d_resid: DevBuf,
    d_ptrs: DevBuf,
    d_upd: DevBuf,
    side: Stream,
    inflight: VecDeque<(Event, Vec<Swap>)>,
    apply: sys::CUfunction,
    _module: Module,
}

impl ExpertCache {
    /// Allocate the device tables, then size the slot arena last from free memory minus
    /// `reserve`, touch it, re-measure, and shrink if the touch cost more than expected.
    /// `offsets[key]` is the byte offset of each key's blob in `host` (registered).
    pub fn new(
        gpu: &Gpu,
        geo: Geometry,
        params: AdaptParams,
        host: &HostArena,
        offsets: &[usize],
        reserve: usize,
        max_slots: Option<usize>,
    ) -> Result<(Self, SizeReport)> {
        assert_eq!(offsets.len(), geo.keys());
        gpu.bind()?;
        let blob = geo.blob_bytes;
        let host_dev: Vec<u64> = offsets
            .iter()
            .map(|&o| {
                host.device_ptr(o)
                    .expect("expert blob outside the registered arena")
            })
            .collect();
        let host_src: Vec<usize> = offsets
            .iter()
            .map(|&o| host.as_ptr() as usize + o)
            .collect();

        let module = gpu.module(kernels::CACHE)?;
        let apply = module.func("apply_updates")?;
        let keys = geo.keys();
        let d_resid = DevBuf::from_slice(&vec![-1i32; keys])?;
        let d_ptrs = DevBuf::from_slice(&host_dev)?;
        let d_upd = DevBuf::alloc(keys * std::mem::size_of::<Upd>())?;

        let (free_before, _) = gpu.mem_info()?;
        let budget = free_before.saturating_sub(reserve);
        let mut slots = (budget / blob)
            .min(max_slots.unwrap_or(usize::MAX))
            .min(keys);
        let slots_first = slots;
        let alloc = |n: usize| -> Result<DevBuf> {
            let b = DevBuf::alloc(n * blob)?;
            ck!(sys::cuMemsetD8_v2(b.ptr, 0, n * blob))?;
            ck!(sys::cuCtxSynchronize())?;
            Ok(b)
        };
        let mut vram = loop {
            match alloc(slots) {
                Ok(b) => break b,
                Err(e) if slots > 64 => {
                    eprintln!("slot arena of {slots} refused ({e}); shrinking");
                    slots -= 64;
                }
                Err(e) => return Err(e),
            }
        };
        let (free_after_touch, _) = gpu.mem_info()?;
        if free_after_touch < reserve && slots > 0 && max_slots.is_none() {
            let short = (reserve - free_after_touch).div_ceil(blob);
            slots = slots.saturating_sub(short);
            drop(vram);
            vram = alloc(slots)?;
        }
        let (free_final, _) = gpu.mem_info()?;

        let table = PointerTable {
            host: host_dev,
            vram_base: vram.ptr,
            slot_bytes: blob as u64,
        };
        let cache = ExpertCache {
            policy: CachePolicy::new(geo, slots, params),
            table,
            host_src,
            vram,
            d_resid,
            d_ptrs,
            d_upd,
            side: Stream::new()?,
            inflight: VecDeque::new(),
            apply,
            _module: module,
        };
        let report = SizeReport {
            free_before,
            reserve,
            slots_first,
            free_after_touch,
            slots,
            free_final,
        };
        Ok((cache, report))
    }

    /// Device address of the `u64[layers*experts]` pointer table.
    pub fn device_ptrs(&self) -> u64 {
        self.d_ptrs.ptr
    }

    /// Device address of the `i32[layers*experts]` residency table (slot or -1).
    pub fn device_residency(&self) -> u64 {
        self.d_resid.ptr
    }

    pub fn vram_base(&self) -> u64 {
        self.vram.ptr
    }

    pub fn swaps_in_flight(&self) -> usize {
        self.inflight.iter().map(|(_, s)| s.len()).sum()
    }

    /// Fill free slots from `ranking` (hottest first), synchronously, and publish the tables.
    pub fn seed(&mut self, ranking: impl IntoIterator<Item = u32>, main: &Stream) -> Result<()> {
        let fills = self.policy.seed(ranking);
        for s in &fills {
            self.copy_in(s, &self.side)?;
        }
        self.side.sync()?;
        for s in &fills {
            self.policy.admit(s);
        }
        self.flush(main)?;
        main.sync()
    }

    /// Count a finished window's routed keys.
    pub fn record(&mut self, keys: &[u32]) {
        self.policy.record(keys);
    }

    /// Call on the main stream between windows: admit landed copies, adapt if due, publish
    /// table changes, and start new copies on the side stream. Returns swaps started.
    pub fn between_windows(&mut self, main: &Stream) -> Result<usize> {
        while self.inflight.front().is_some_and(|(e, _)| e.done()) {
            let (_, swaps) = self.inflight.pop_front().unwrap();
            for s in &swaps {
                self.policy.admit(s);
            }
        }
        let swaps = if self.policy.end_window() {
            self.policy.plan_adapt()
        } else {
            Vec::new()
        };
        self.flush(main)?;
        if swaps.is_empty() {
            return Ok(0);
        }
        let published = Event::new(false)?;
        published.record(main)?;
        self.side.wait(&published)?;
        for s in &swaps {
            self.copy_in(s, &self.side)?;
        }
        let landed = Event::new(false)?;
        landed.record(&self.side)?;
        let n = swaps.len();
        self.inflight.push_back((landed, swaps));
        Ok(n)
    }

    /// Wait for all copies and admit them (then publish on `main`).
    pub fn settle(&mut self, main: &Stream) -> Result<()> {
        self.side.sync()?;
        while let Some((_, swaps)) = self.inflight.pop_front() {
            for s in &swaps {
                self.policy.admit(s);
            }
        }
        self.flush(main)
    }

    fn copy_in(&self, s: &Swap, on: &Stream) -> Result<()> {
        let blob = self.policy.geometry().blob_bytes;
        let dst = self.vram.ptr + s.slot as u64 * blob as u64;
        let src = self.host_src[s.incoming as usize] as *const c_void;
        ck!(sys::cuMemcpyHtoDAsync_v2(dst, src, blob, on.0))
    }

    /// Write dirty table entries on `main`.
    fn flush(&mut self, main: &Stream) -> Result<()> {
        let dirty = self.policy.drain_dirty();
        if dirty.is_empty() {
            return Ok(());
        }
        let res = self.policy.residency();
        let upd: Vec<Upd> = dirty
            .iter()
            .map(|&k| {
                let slot = res[k as usize];
                Upd {
                    key: k,
                    slot,
                    ptr: self.table.entry(k, slot),
                }
            })
            .collect();
        self.d_upd.write_async(0, &upd, main)?;
        let n = upd.len() as i32;
        unsafe {
            launch(
                self.apply,
                ((n as u32).div_ceil(256), 1, 1),
                (256, 1, 1),
                0,
                main,
                args![self.d_upd.ptr, n, self.d_resid.ptr, self.d_ptrs.ptr],
            )
        }
    }

    /// The device tables read back (tests).
    pub fn read_tables(&self) -> Result<(Vec<i32>, Vec<u64>)> {
        let n = self.policy.geometry().keys();
        Ok((self.d_resid.read(0, n)?, self.d_ptrs.read(0, n)?))
    }

    /// The host-side pointer table for the current residency (tests).
    pub fn expected_ptrs(&self) -> Vec<u64> {
        self.table.build(self.policy.residency())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::{ArenaOptions, HostArena};

    #[test]
    fn swaps_keep_tables_and_bytes_consistent() {
        let Some(gpu) = Gpu::for_test() else { return };
        gpu.bind().unwrap();
        let geo = Geometry {
            layers: 2,
            experts: 16,
            blob_bytes: 64 << 10,
        };
        let mut arena =
            HostArena::new(geo.keys() * geo.blob_bytes, ArenaOptions::default()).unwrap();
        arena.register(&gpu.ctx, &[1 << 20]).unwrap();
        let bytes = unsafe { arena.bytes_mut() };
        let words = geo.blob_bytes / 4;
        let mut want = vec![0u32; geo.keys()];
        for k in 0..geo.keys() {
            let blob = &mut bytes[k * geo.blob_bytes..(k + 1) * geo.blob_bytes];
            for (i, w) in blob.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                let v = (k as u32)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(i as u32 * 7);
                w.copy_from_slice(&v.to_le_bytes());
                want[k] = want[k].wrapping_add(v);
            }
        }
        let offsets: Vec<usize> = (0..geo.keys()).map(|k| k * geo.blob_bytes).collect();
        let (mut cache, rep) = ExpertCache::new(
            &gpu,
            geo,
            AdaptParams::default(),
            &arena,
            &offsets,
            0,
            Some(6),
        )
        .unwrap();
        assert_eq!(rep.slots, 6);
        let main = Stream::new().unwrap();
        cache.seed(0..6, &main).unwrap();

        let module = gpu.module(kernels::CACHE).unwrap();
        let sums = module.func("blob_sums").unwrap();
        let out = DevBuf::zeroed(geo.keys() * 4).unwrap();
        let check = |cache: &ExpertCache, main: &Stream| {
            unsafe {
                launch(
                    sums,
                    (geo.keys() as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    main,
                    args![cache.device_ptrs(), words as i32, out.ptr],
                )
            }
            .unwrap();
            main.sync().unwrap();
            let got: Vec<u32> = out.read(0, geo.keys()).unwrap();
            assert_eq!(got, want, "blob checksums through the pointer table");
            let (res, ptrs) = cache.read_tables().unwrap();
            assert_eq!(res, cache.policy.residency());
            assert_eq!(ptrs, cache.expected_ptrs());
        };
        check(&cache, &main);

        // Route mostly to keys 20..26 so they displace the seed.
        let mut started = 0;
        for w in 0..24u32 {
            let keys: Vec<u32> = (0..30).map(|i| 20 + (i + w) % 6).collect();
            cache.record(&keys);
            started += cache.between_windows(&main).unwrap();
            check(&cache, &main);
        }
        cache.settle(&main).unwrap();
        check(&cache, &main);
        assert!(started >= 6, "only {started} swaps");
        for k in 20..26 {
            assert!(cache.policy.slot_of(k).is_some(), "key {k} not resident");
        }
        cache.policy.check();
    }
}

/// Resident mode on the GPU: VRAM slots + scratch, a host arena holding only the experts not
/// in VRAM, exchanges driven by [`ResidentPolicy`]. The device table holds, per key, the
/// address of its blob if it is on the GPU and 0 if it is on the host (the "0 = not resident"
/// convention of the MoE plan builder, so `addr` can feed [`crate::miss::build_plan`]).
pub struct ResidentCache {
    pub policy: crate::resident::ResidentPolicy,
    blob: usize,
    vram: DevBuf,
    scratch: DevBuf,
    host: HostArena,
    d_resid: DevBuf,
    d_addr: DevBuf,
    d_upd: DevBuf,
    side: Stream,
    inflight: VecDeque<(Event, Vec<u32>)>,
    apply: sys::CUfunction,
    _module: Module,
}

impl ResidentCache {
    /// Allocate `n_slots` + `n_scratch` VRAM blobs and a registered host arena for the rest,
    /// and place every key: `ranking` (hottest first) into slots, the others into host slots.
    /// `src(key)` gives each key's bytes for the initial load.
    #[allow(clippy::too_many_arguments)]
    pub fn new<'a>(
        gpu: &Gpu,
        geo: Geometry,
        params: AdaptParams,
        n_slots: usize,
        n_scratch: usize,
        ranking: impl IntoIterator<Item = u32>,
        src: impl Fn(u32) -> &'a [u8],
    ) -> Result<Self> {
        Self::new_fill(gpu, geo, params, n_slots, n_scratch, ranking, |k, dst| {
            let bytes = src(k);
            assert_eq!(bytes.len(), dst.len());
            dst.copy_from_slice(bytes)
        })
    }

    /// [`new`](Self::new) with `fill(key, dst)` writing each key's bytes into `dst` (straight
    /// into the host arena for host keys, a staging buffer for slot keys), in key order, so a
    /// loader can stream from a file without holding or mapping the whole set.
    #[allow(clippy::too_many_arguments)]
    pub fn new_fill(
        gpu: &Gpu,
        geo: Geometry,
        params: AdaptParams,
        n_slots: usize,
        n_scratch: usize,
        ranking: impl IntoIterator<Item = u32>,
        mut fill: impl FnMut(u32, &mut [u8]),
    ) -> Result<Self> {
        use crate::arena::ArenaOptions;
        use crate::resident::{Loc, ResidentPolicy};
        gpu.bind()?;
        let blob = geo.blob_bytes;
        let policy = ResidentPolicy::new(geo, n_slots, n_scratch, params, ranking);
        let mut host = HostArena::new(policy.host_slots().max(1) * blob, ArenaOptions::default())
            .map_err(|e| crate::gpu::Error(format!("mmap: {e}")))?;
        host.register(&gpu.ctx, &[1 << 30, 256 << 20])?;
        let vram = DevBuf::alloc(n_slots.max(1) * blob)?;
        let scratch = DevBuf::alloc(n_scratch.max(1) * blob)?;
        let mut stage = vec![0u8; blob];
        for k in 0..geo.keys() as u32 {
            match policy.loc(k) {
                Loc::Slot(s) => {
                    fill(k, &mut stage);
                    vram.write(s as usize * blob, &stage)?
                }
                Loc::Host(h) => {
                    let hb = unsafe { host.bytes_mut() };
                    fill(k, &mut hb[h as usize * blob..(h as usize + 1) * blob]);
                }
                Loc::Scratch(_) => unreachable!(),
            }
        }
        let module = gpu.module(kernels::CACHE)?;
        let apply = module.func("apply_updates")?;
        let keys = geo.keys();
        let mut c = ResidentCache {
            policy,
            blob,
            vram,
            scratch,
            host,
            d_resid: DevBuf::from_slice(&vec![-1i32; keys])?,
            d_addr: DevBuf::zeroed(keys * 8)?,
            d_upd: DevBuf::alloc(keys * std::mem::size_of::<Upd>())?,
            side: Stream::new()?,
            inflight: VecDeque::new(),
            apply,
            _module: module,
        };
        let all: Vec<u32> = (0..keys as u32).collect();
        let s = Stream::new()?;
        c.publish(&all, &s)?;
        s.sync()?;
        Ok(c)
    }

    /// Device address of `u64[keys]`: blob address if on the GPU, 0 if on the host.
    pub fn device_addrs(&self) -> u64 {
        self.d_addr.ptr
    }

    /// Host-side view of the same table.
    pub fn addr(&self, key: u32) -> u64 {
        use crate::resident::Loc;
        match self.policy.loc(key) {
            Loc::Slot(s) => self.vram.ptr + s as u64 * self.blob as u64,
            Loc::Scratch(x) => self.scratch.ptr + x as u64 * self.blob as u64,
            Loc::Host(_) => 0,
        }
    }

    /// The host copy of a key that lives on the host.
    pub fn host_blob(&self, key: u32) -> Option<&[u8]> {
        use crate::resident::Loc;
        match self.policy.loc(key) {
            Loc::Host(h) => Some(unsafe {
                std::slice::from_raw_parts(
                    self.host.as_ptr().add(h as usize * self.blob),
                    self.blob,
                )
            }),
            _ => None,
        }
    }

    /// The device-mapped address of a host-resident key's blob (the arena is registered
    /// `DEVICEMAP`), for letting a kernel stream it over PCIe instead of the CPU computing it.
    pub fn host_device_addr(&self, key: u32) -> Option<u64> {
        use crate::resident::Loc;
        match self.policy.loc(key) {
            Loc::Host(h) => self.host.device_ptr(h as usize * self.blob),
            _ => None,
        }
    }

    pub fn record(&mut self, keys: &[u32]) {
        self.policy.record(keys);
    }

    /// Between windows, on the main stream: advance landed exchanges, publish table changes,
    /// then start this boundary's copies on the side stream. Returns copies started.
    pub fn boundary(&mut self, main: &Stream) -> Result<usize> {
        let mut landed = Vec::new();
        while self.inflight.front().is_some_and(|(e, _)| e.done()) {
            landed.extend(self.inflight.pop_front().unwrap().1);
        }
        let steps = self.policy.boundary(&landed);
        let dirty = self.policy.drain_dirty();
        self.publish(&dirty, main)?;
        if steps.is_empty() {
            return Ok(0);
        }
        let published = Event::new(false)?;
        published.record(main)?;
        self.side.wait(&published)?;
        let b = self.blob as u64;
        for st in &steps {
            use crate::resident::CopyOp::*;
            match st.op {
                SlotToScratch { slot, scratch } => ck!(sys::cuMemcpyDtoDAsync_v2(
                    self.scratch.ptr + scratch as u64 * b,
                    self.vram.ptr + slot as u64 * b,
                    self.blob,
                    self.side.0
                ))?,
                HostToSlot { host, slot } => ck!(sys::cuMemcpyHtoDAsync_v2(
                    self.vram.ptr + slot as u64 * b,
                    self.host.as_ptr().add(host as usize * self.blob) as *const c_void,
                    self.blob,
                    self.side.0
                ))?,
                ScratchToHost { scratch, host } => ck!(sys::cuMemcpyDtoHAsync_v2(
                    self.host.as_ptr().add(host as usize * self.blob) as *mut c_void,
                    self.scratch.ptr + scratch as u64 * b,
                    self.blob,
                    self.side.0
                ))?,
            }
        }
        let ev = Event::new(false)?;
        ev.record(&self.side)?;
        self.inflight
            .push_back((ev, steps.iter().map(|s| s.ex).collect()));
        Ok(steps.len())
    }

    /// Wait for in-flight copies (they are reported landed at the next boundary).
    pub fn sync_side(&self) -> Result<()> {
        self.side.sync()
    }

    fn publish(&mut self, keys: &[u32], main: &Stream) -> Result<()> {
        if keys.is_empty() {
            return Ok(());
        }
        let upd: Vec<Upd> = keys
            .iter()
            .map(|&k| {
                let a = self.addr(k);
                Upd {
                    key: k,
                    slot: if a == 0 { -1 } else { 0 },
                    ptr: a,
                }
            })
            .collect();
        self.d_upd.write_async(0, &upd, main)?;
        let n = upd.len() as i32;
        unsafe {
            launch(
                self.apply,
                ((n as u32).div_ceil(256), 1, 1),
                (256, 1, 1),
                0,
                main,
                args![self.d_upd.ptr, n, self.d_resid.ptr, self.d_addr.ptr],
            )
        }
    }

    /// The device address table read back (tests).
    pub fn read_addrs(&self) -> Result<Vec<u64>> {
        self.d_addr.read(0, self.policy.locs().len())
    }
}

#[cfg(test)]
mod resident_tests {
    use super::*;

    #[test]
    fn resident_exchanges_keep_every_blob_readable() {
        let Some(gpu) = Gpu::for_test() else { return };
        gpu.bind().unwrap();
        let geo = Geometry {
            layers: 2,
            experts: 32,
            blob_bytes: 64 << 10,
        };
        let words = geo.blob_bytes / 4;
        let content = |k: u32| -> Vec<u8> {
            (0..words)
                .flat_map(|i| (k.wrapping_mul(2_654_435_761) ^ (i as u32 * 7)).to_le_bytes())
                .collect()
        };
        let blobs: Vec<Vec<u8>> = (0..geo.keys() as u32).map(content).collect();
        let want: Vec<u32> = blobs
            .iter()
            .map(|b| {
                b.as_chunks::<4>()
                    .0
                    .iter()
                    .fold(0u32, |a, w| a.wrapping_add(u32::from_le_bytes(*w)))
            })
            .collect();
        let params = AdaptParams {
            every: 2,
            max_swaps: 6,
            ..AdaptParams::default()
        };
        let mut c =
            ResidentCache::new(&gpu, geo, params, 12, 4, 0..64, |k| &blobs[k as usize]).unwrap();
        let module = gpu.module(kernels::CACHE).unwrap();
        let sums = module.func("blob_sums").unwrap();
        let out = DevBuf::zeroed(geo.keys() * 4).unwrap();
        let main = Stream::new().unwrap();
        let mut rng = crate::q2cpu::tests::Rng(42);
        let mut started = 0;
        for w in 0..300 {
            // A "window": read every GPU-resident blob through the device table (keys on the
            // host point at a zero address and are skipped) and every host blob on the CPU.
            let addrs = c.read_addrs().unwrap();
            let gpu_keys: Vec<u32> = (0..geo.keys() as u32)
                .filter(|&k| addrs[k as usize] != 0)
                .collect();
            let table: Vec<u64> = gpu_keys.iter().map(|&k| addrs[k as usize]).collect();
            let d_table = DevBuf::from_slice(&table).unwrap();
            unsafe {
                launch(
                    sums,
                    (table.len() as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    &main,
                    args![d_table.ptr, words as i32, out.ptr],
                )
                .unwrap();
            }
            main.sync().unwrap();
            let got: Vec<u32> = out.read(0, table.len()).unwrap();
            for (i, &k) in gpu_keys.iter().enumerate() {
                assert_eq!(got[i], want[k as usize], "window {w}: GPU key {k}");
            }
            for k in 0..geo.keys() as u32 {
                if let Some(b) = c.host_blob(k) {
                    assert!(b == &blobs[k as usize][..], "window {w}: host key {k}");
                }
            }
            let base = (w / 60) * 9;
            let keys: Vec<u32> = (0..30)
                .map(|_| ((base + (rng.next() % 10) as usize) % geo.keys()) as u32)
                .collect();
            c.record(&keys);
            if rng.next().is_multiple_of(3) {
                c.sync_side().unwrap(); // sometimes let copies land before the boundary
            }
            started += c.boundary(&main).unwrap();
        }
        assert!(started > 30, "only {started} copies");
        assert_eq!(
            gpu_keys_count(&c),
            12 + c
                .policy
                .locs()
                .iter()
                .filter(|l| matches!(l, crate::resident::Loc::Scratch(_)))
                .count()
        );
    }

    fn gpu_keys_count(c: &ResidentCache) -> usize {
        c.policy.locs().iter().filter(|l| l.on_gpu()).count()
    }
}
