//! A key→value file store kept in dedicated Gated DeltaNet heads of one layer's recurrent state
//! (phase 1 of "zero-context compiled memory"; design and numbers in `docs/design/gdn-memstore.md`).
//!
//! The idea and the directory/content split are divagr.com/blog/qwenputer's filesystem (code:
//! github.com/divagr18/qwenputer, branch `rv-store`, `docs/q35-store.md` and
//! `chipformer/q35/store.py`): with decay g = 1, write strength β = 1 and a one-hot key, a GDN
//! token overwrites exactly one 128-float row of a head's state, and a one-hot query reads it
//! back. The post's own v1 store (MIT, `rv-store`) uses a 128-entry scanned directory and
//! chained 104-token chunks; this follows the post's 1,024-slot hashed directory instead and
//! keeps 18-bit token ids like its store. No code is ported. Here the engine (not the model) is the driver: every write and read is a
//! [`ComputeDevice::gdn_step`] window of at most [`MAX_T`] tokens whose inputs are built so the
//! kernel does exactly that, `Commit` for writes, `ReadOnly` for reads. Nothing pokes the state
//! buffer except [`MemStore::audit`] (a read-only integrity scan) and the head snapshot helpers.
//!
//! # Layout (state `[i 128][hv 48][j 128]` f32, v head `hv` on k head `hv % 16`)
//!
//! The store owns k heads 12..=15 and all three of their v heads, 12 v heads of 48:
//!
//! - **directory**: v heads `12..=15` and `28..=31` (groups 0 and 1), 1,024 slots. Slot `s` is
//!   v head `12 + s % 4 + 16 · ((s / 4) % 2)`, row `s / 8`: the 8 slots of a bucket `s / 8` share
//!   one row, so one token with the same one-hot query on the 4 store k heads reads a whole
//!   bucket (and the 4 content rows at that index).
//! - **content**: v heads `44..=47` (group 2), 512 rows: row `c` is v head `44 + c % 4`, row
//!   `c / 4`, 127 token ids per row.
//!
//! Every written row has a marker 1 in column 127 (an unwritten row is all zeros). The kernel's
//! output is `o · rsqrt(mean o² + eps) · norm · σ(z)`, which loses the row's scale; the marker
//! puts it back: `row_j = (y_j / norm_j) / (y_127 / norm_127)`.
//!
//! Directory slot columns: 0 status (0 empty, 1 live, 2 deleted), 1 name length in bytes, 2 file
//! length in tokens, 3 content rows used, `4..52` the name (2 bytes per float, 96 bytes),
//! `52..127` the file's content row numbers (75 rows, so 9,525 tokens per file).
//!
//! # Values
//!
//! Stored values are token ids (u32), one per float, below 2^18 (Qwen's 248,320-token vocabulary
//! fits; the post's store also carries 18 id bits). The state is exact for integers below 2^24,
//! but a read goes through the kernel's norm and a division, about 8 roundings (~5·10⁻⁷
//! relative), so a value must sit below ~2^20 to round back exactly; 2^18 leaves a 4× margin
//! (worst error 0.13). The same rule is why names pack 2 bytes, not 3, per float.
//!
//! # The driver discipline
//!
//! A store head is exact only while every token through it has g = 1 exactly and either β = 1
//! with a one-hot k, or k = 0. One model token with g < 1 scales every row by a non-power of two
//! and the head is never RAM again. The store's own windows pin `a = −40 − dt_bias` (g rounds to
//! 1.0 for any `ssm_a` ≥ −10⁴), `b = ±40` (β is 1.0 or 4·10⁻¹⁸, which with v = 0 leaves the row
//! bitwise alone) and zero q/k on the base model's k heads, so base heads are untouched (tested).
//! Model tokens are kept out by construction, see the design doc ("pinned rows"): the q/k/v/z
//! rows of the stacked projection and the conv taps for store k heads are zero, so a model token
//! has k = 0 there and cannot write. [`MemStore::audit`] detects a head that got poisoned anyway.

use std::collections::BTreeSet;

use anyhow::{bail, ensure, Result};
use tang_compute::flash::shape::{
    GDN_A, GDN_B, GDN_CONV, GDN_D, GDN_HK, GDN_HV, GDN_PROJ, GDN_V, GDN_Z,
};
use tang_compute::flash::{GdnMode, GdnParams, Win, GDN_STATE, MAX_T};
use tang_compute::ComputeDevice;

const D: usize = GDN_D;
/// The store's k heads.
pub const STORE_HK: [usize; 4] = [12, 13, 14, 15];
/// The store's v heads (every v head of [`STORE_HK`]).
pub const STORE_HV: [usize; 12] = [12, 13, 14, 15, 28, 29, 30, 31, 44, 45, 46, 47];
/// Directory slots.
pub const SLOTS: usize = 1024;
/// Content rows.
pub const ROWS: usize = 512;
/// Live files at most (load factor 1/2, as in the post).
pub const MAX_FILES: usize = 512;
/// Token ids per content row (column 127 is the marker).
pub const ROW_TOKENS: usize = D - 1;
/// Name bytes at most.
pub const MAX_NAME: usize = 96;
/// Content rows per file at most.
pub const MAX_FILE_ROWS: usize = D - 1 - COL_ROWS;
/// Tokens per file at most.
pub const MAX_FILE_TOKENS: usize = MAX_FILE_ROWS * ROW_TOKENS;
/// Every stored value is below this (see the module doc).
pub const VALUE_LIMIT: u32 = 1 << 18;

const BUCKETS: usize = SLOTS / 8;
const MARK: usize = D - 1;
const COL_STATUS: usize = 0;
const COL_NAMELEN: usize = 1;
const COL_LEN: usize = 2;
const COL_NROWS: usize = 3;
const COL_NAME: usize = 4;
const COL_ROWS: usize = COL_NAME + MAX_NAME / 2;
const LIVE: u32 = 1;
const DEAD: u32 = 2;
const PIN_A: f32 = -40.0;
const PIN_B: f32 = 40.0;
const PIN_Z: f32 = 40.0;

/// A read row: `((k head, row), [group] values)`.
type Decoded = ((usize, usize), [Vec<u32>; 3]);
/// One token's write on a k head: `(row, [group] values)`.
type RowWrite<'a> = (usize, [Option<&'a [u32]>; 3]);

/// A row's place: store k head index (0..4), v group (0..3), row.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Cell {
    kh: usize,
    group: usize,
    row: usize,
}

impl Cell {
    #[cfg_attr(not(test), allow(dead_code))]
    fn hv(self) -> usize {
        STORE_HK[self.kh] + GDN_HK * self.group
    }
    fn slot(s: usize) -> Self {
        Cell {
            kh: s % 4,
            group: (s / 4) % 2,
            row: s / 8,
        }
    }
    fn content(c: usize) -> Self {
        Cell {
            kh: c % 4,
            group: 2,
            row: c / 4,
        }
    }
}

/// What [`MemStore::capacity`] reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Capacity {
    /// Live files.
    pub files: usize,
    /// Live files at most.
    pub max_files: usize,
    /// Content rows in use.
    pub rows_used: usize,
    /// Tokens stored (sum of file lengths).
    pub tokens: usize,
    /// Tokens the content heads hold at most ([`ROWS`] · [`ROW_TOKENS`]).
    pub max_tokens: usize,
}

/// Kernel traffic so far.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Traffic {
    /// `gdn_step` calls (windows).
    pub steps: usize,
    /// Tokens in those windows.
    pub tokens: usize,
}

/// A decoded directory slot.
#[derive(Clone, Debug, Default)]
struct Slot {
    status: u32,
    name: Vec<u8>,
    len: usize,
    rows: Vec<usize>,
}

impl Slot {
    fn decode(r: &[u32]) -> Self {
        let nl = (r[COL_NAMELEN] as usize).min(MAX_NAME);
        let mut name = Vec::with_capacity(nl);
        for i in 0..nl {
            let w = r[COL_NAME + i / 2];
            name.push(if i % 2 == 0 { w & 0xff } else { w >> 8 } as u8);
        }
        let nr = (r[COL_NROWS] as usize).min(MAX_FILE_ROWS);
        Slot {
            status: r[COL_STATUS],
            name,
            len: r[COL_LEN] as usize,
            rows: r[COL_ROWS..COL_ROWS + nr]
                .iter()
                .map(|&x| x as usize)
                .collect(),
        }
    }

    fn encode(&self) -> Vec<u32> {
        let mut r = vec![0u32; ROW_TOKENS];
        r[COL_STATUS] = self.status;
        r[COL_NAMELEN] = self.name.len() as u32;
        r[COL_LEN] = self.len as u32;
        r[COL_NROWS] = self.rows.len() as u32;
        for (i, &b) in self.name.iter().enumerate() {
            r[COL_NAME + i / 2] |= (b as u32) << (8 * (i % 2));
        }
        for (i, &c) in self.rows.iter().enumerate() {
            r[COL_ROWS + i] = c as u32;
        }
        r
    }
}

/// Where a name probes to.
struct Probe {
    /// The live slot holding the name.
    found: Option<(usize, Slot)>,
    /// The first empty or deleted slot on the probe path.
    free: Option<usize>,
}

/// The FNV-1a hash of a name, its home slot.
fn home(name: &[u8]) -> usize {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in name {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    (h % SLOTS as u64) as usize
}

/// The store over one GDN layer's state buffer; the buffer itself is passed to each call (it
/// stays the engine's). The struct holds the driver's derived bookkeeping only (free rows, live
/// count), which [`open`](Self::open) rebuilds from the state by a directory scan.
pub struct MemStore<'p, Dev: ComputeDevice> {
    dev: &'p Dev,
    params: GdnParams<'p, Dev::Buffer>,
    dt: Vec<f32>,
    norm: Vec<f32>,
    eps: f32,
    win: Dev::Buffer,
    free: BTreeSet<usize>,
    files: usize,
    tokens: usize,
    traffic: Traffic,
}

impl<'p, Dev: ComputeDevice> MemStore<'p, Dev> {
    /// Attach to `state` (one layer's [`GDN_STATE`] floats) with that layer's params, rebuilding
    /// the bookkeeping by reading the whole directory through the kernel (16 windows).
    pub fn open(
        dev: &'p Dev,
        params: GdnParams<'p, Dev::Buffer>,
        eps: f32,
        state: &mut Dev::Buffer,
    ) -> Result<Self> {
        let dt = dev.download(params.dt_bias);
        let norm = dev.download(params.norm);
        ensure!(
            dt.len() >= GDN_HV && norm.len() >= D,
            "GDN params too short"
        );
        ensure!(
            norm.iter().all(|x| x.is_finite() && *x != 0.0),
            "ssm_norm has a zero or non-finite weight: the store can't read through it"
        );
        let mut s = MemStore {
            dev,
            params,
            dt,
            norm,
            eps,
            win: dev.upload_f32(&[0.0; Win::WORDS]),
            free: (0..ROWS).collect(),
            files: 0,
            tokens: 0,
            traffic: Traffic::default(),
        };
        for (_, slot) in s.scan(state)? {
            for &c in &slot.rows {
                ensure!(c < ROWS && s.free.remove(&c), "row {c} is claimed twice");
            }
            s.files += 1;
            s.tokens += slot.len;
        }
        Ok(s)
    }

    /// Files, rows and tokens.
    pub fn capacity(&self) -> Capacity {
        Capacity {
            files: self.files,
            max_files: MAX_FILES,
            rows_used: ROWS - self.free.len(),
            tokens: self.tokens,
            max_tokens: ROWS * ROW_TOKENS,
        }
    }

    /// Kernel windows and tokens spent so far.
    pub fn traffic(&self) -> Traffic {
        self.traffic
    }

    /// Store `data` under `name`, replacing any file of that name. Content rows are written
    /// before (in the same commit window as, at the latest) the directory slot.
    pub fn put(&mut self, state: &mut Dev::Buffer, name: &[u8], data: &[u32]) -> Result<()> {
        ensure!(
            !name.is_empty() && name.len() <= MAX_NAME,
            "name must be 1..={MAX_NAME} bytes"
        );
        if let Some(v) = data.iter().find(|&&v| v >= VALUE_LIMIT) {
            bail!("value {v} is not below 2^18");
        }
        let need = data.len().div_ceil(ROW_TOKENS);
        ensure!(
            need <= MAX_FILE_ROWS,
            "{} tokens is over the {MAX_FILE_TOKENS}-token file limit",
            data.len()
        );
        ensure!(
            need <= self.free.len(),
            "store full: {need} rows needed, {} free",
            self.free.len()
        );
        let p = self.probe(state, name)?;
        let (slot, old) = match (p.found, p.free) {
            (Some((s, old)), _) => (s, Some(old)),
            (None, Some(s)) if self.files < MAX_FILES => (s, None),
            (None, Some(_)) => bail!("store full: {MAX_FILES} files"),
            (None, None) => bail!("directory full"),
        };
        let rows: Vec<usize> = self.free.iter().take(need).copied().collect();
        let mut writes: Vec<(Cell, Vec<u32>)> = rows
            .iter()
            .zip(data.chunks(ROW_TOKENS))
            .map(|(&c, chunk)| (Cell::content(c), chunk.to_vec()))
            .collect();
        let entry = Slot {
            status: LIVE,
            name: name.to_vec(),
            len: data.len(),
            rows: rows.clone(),
        };
        writes.push((Cell::slot(slot), entry.encode()));
        self.write(state, &writes)?;
        for c in &rows {
            self.free.remove(c);
        }
        self.tokens += data.len();
        match old {
            Some(o) => {
                self.free.extend(o.rows);
                self.tokens -= o.len;
            }
            None => self.files += 1,
        }
        Ok(())
    }

    /// The file `name`, read back through the kernel.
    pub fn get(&mut self, state: &mut Dev::Buffer, name: &[u8]) -> Result<Option<Vec<u32>>> {
        let Some((_, slot)) = self.probe(state, name)?.found else {
            return Ok(None);
        };
        let cells: Vec<Cell> = slot.rows.iter().map(|&c| Cell::content(c)).collect();
        let got = self.read(state, &cells)?;
        let mut out: Vec<u32> = got.into_iter().flatten().collect();
        ensure!(
            out.len() >= slot.len,
            "file {:?} is short",
            String::from_utf8_lossy(name)
        );
        out.truncate(slot.len);
        Ok(Some(out))
    }

    /// Delete `name` (a tombstone in its slot); whether it was there.
    pub fn delete(&mut self, state: &mut Dev::Buffer, name: &[u8]) -> Result<bool> {
        let Some((s, old)) = self.probe(state, name)?.found else {
            return Ok(false);
        };
        let tomb = Slot {
            status: DEAD,
            ..Slot::default()
        };
        self.write(state, &[(Cell::slot(s), tomb.encode())])?;
        self.free.extend(old.rows);
        self.files -= 1;
        self.tokens -= old.len;
        Ok(true)
    }

    /// Every live file's name and length, in slot order (a full directory scan: 16 windows).
    pub fn list(&mut self, state: &mut Dev::Buffer) -> Result<Vec<(Vec<u8>, usize)>> {
        Ok(self
            .scan(state)?
            .into_iter()
            .map(|(_, s)| (s.name, s.len))
            .collect())
    }

    /// Live slots, by reading all 128 buckets.
    fn scan(&mut self, state: &mut Dev::Buffer) -> Result<Vec<(usize, Slot)>> {
        let rows: Vec<Vec<Cell>> = (0..BUCKETS).map(Self::bucket).collect();
        let flat: Vec<Cell> = rows.concat();
        let got = self.read(state, &flat)?;
        Ok(got
            .chunks(8)
            .enumerate()
            .flat_map(|(b, slots)| {
                slots
                    .iter()
                    .enumerate()
                    .map(move |(i, r)| (b * 8 + i, Slot::decode(r)))
            })
            .filter(|(_, s)| s.status == LIVE)
            .collect())
    }

    fn bucket(b: usize) -> Vec<Cell> {
        (b * 8..b * 8 + 8).map(Cell::slot).collect()
    }

    /// Linear probing from the name's home slot. Buckets (8 slots, one token each) are read
    /// lazily: a 1-token window for the home bucket, then 8-token windows, so a lookup at load
    /// 1/2 is one window and rarely two.
    fn probe(&mut self, state: &mut Dev::Buffer, name: &[u8]) -> Result<Probe> {
        let h = home(name);
        let mut read: Vec<Option<Vec<u32>>> = vec![None; SLOTS];
        let mut next_bucket = h / 8;
        let mut fetched = 0;
        let mut free = None;
        for pos in 0..SLOTS {
            let s = (h + pos) % SLOTS;
            if read[s].is_none() {
                let n = if fetched == 0 {
                    1
                } else {
                    MAX_T.min(BUCKETS - fetched)
                };
                let bs: Vec<usize> = (0..n).map(|i| (next_bucket + i) % BUCKETS).collect();
                let cells: Vec<Cell> = bs.iter().flat_map(|&b| Self::bucket(b)).collect();
                let got = self.read(state, &cells)?;
                for (k, r) in got.into_iter().enumerate() {
                    read[bs[k / 8] * 8 + k % 8] = Some(r);
                }
                next_bucket = (next_bucket + n) % BUCKETS;
                fetched += n;
            }
            let slot = Slot::decode(read[s].as_ref().unwrap());
            match slot.status {
                LIVE if slot.name == name => {
                    return Ok(Probe {
                        found: Some((s, slot)),
                        free,
                    })
                }
                LIVE => {}
                0 => {
                    return Ok(Probe {
                        found: None,
                        free: free.or(Some(s)),
                    })
                }
                _ => free = free.or(Some(s)),
            }
        }
        Ok(Probe { found: None, free })
    }

    /// One token's worth of base inputs: g pinned to 1 on every head, β ≈ 0 and z = 40.
    fn blank(&self) -> (Vec<f32>, Vec<f32>) {
        let mut p = vec![0f32; GDN_PROJ];
        for hv in 0..GDN_HV {
            p[GDN_A + hv] = PIN_A - self.dt[hv];
            p[GDN_B + hv] = -PIN_B;
        }
        for &hv in &STORE_HV {
            p[GDN_Z + hv * D..GDN_Z + (hv + 1) * D].fill(PIN_Z);
        }
        (vec![0f32; GDN_CONV], p)
    }

    /// Read `cells` (rows of 127 values) through `ReadOnly` windows. A token queries one row per
    /// store k head and returns that row of all three of its v heads.
    fn read(&mut self, state: &mut Dev::Buffer, cells: &[Cell]) -> Result<Vec<Vec<u32>>> {
        // per k head, the distinct rows wanted, in order
        let mut queues: [Vec<usize>; 4] = Default::default();
        for c in cells {
            if !queues[c.kh].contains(&c.row) {
                queues[c.kh].push(c.row);
            }
        }
        let ntok = queues.iter().map(Vec::len).max().unwrap_or(0);
        // decoded[(kh, row)] = [group][127]
        let mut decoded: Vec<Decoded> = Vec::new();
        for w0 in (0..ntok).step_by(MAX_T) {
            let t = (ntok - w0).min(MAX_T);
            let (bh, bp) = self.blank();
            let mut h = Vec::with_capacity(t * GDN_CONV);
            let mut p = Vec::with_capacity(t * GDN_PROJ);
            for tt in 0..t {
                let mut hh = bh.clone();
                for (kh, q) in queues.iter().enumerate() {
                    if let Some(&row) = q.get(w0 + tt) {
                        hh[STORE_HK[kh] * D + row] = 1.0;
                    }
                }
                h.extend_from_slice(&hh);
                p.extend_from_slice(&bp);
            }
            let y = self.step(state, &h, &p, t, None)?;
            for tt in 0..t {
                for (kh, q) in queues.iter().enumerate() {
                    if let Some(&row) = q.get(w0 + tt) {
                        let g = |group: usize| {
                            let hv = STORE_HK[kh] + GDN_HK * group;
                            self.decode(&y[tt * GDN_V + hv * D..tt * GDN_V + (hv + 1) * D], hv, row)
                        };
                        decoded.push(((kh, row), [g(0)?, g(1)?, g(2)?]));
                    }
                }
            }
        }
        Ok(cells
            .iter()
            .map(|c| {
                let (_, r) = decoded.iter().find(|(k, _)| *k == (c.kh, c.row)).unwrap();
                r[c.group].clone()
            })
            .collect())
    }

    /// A row's values from its kernel output (see the module doc), checked to be integers.
    fn decode(&self, y: &[f32], hv: usize, row: usize) -> Result<Vec<u32>> {
        let m = y[MARK] / self.norm[MARK];
        if m == 0.0 {
            ensure!(
                y.iter().all(|&x| x == 0.0),
                "head {hv} row {row}: unmarked row isn't empty (poisoned)"
            );
            return Ok(vec![0; ROW_TOKENS]);
        }
        (0..ROW_TOKENS)
            .map(|j| {
                let x = (y[j] / self.norm[j]) / m;
                let r = x.round();
                ensure!(
                    (x - r).abs() < 0.25 && (0.0..VALUE_LIMIT as f32).contains(&r),
                    "head {hv} row {row} col {j}: read {x}, not a stored value (poisoned)"
                );
                Ok(r as u32)
            })
            .collect()
    }

    /// Write rows through `Commit` windows: per token, one row index per store k head, written
    /// into any of its three v heads (β = 1 there, β ≈ 0 with v = 0 in the others).
    fn write(&mut self, state: &mut Dev::Buffer, writes: &[(Cell, Vec<u32>)]) -> Result<()> {
        // per k head, (row, [group] values)
        let mut queues: [Vec<RowWrite<'_>>; 4] = Default::default();
        for (c, v) in writes {
            debug_assert!(v.len() <= ROW_TOKENS);
            let q = &mut queues[c.kh];
            match q
                .iter_mut()
                .find(|(r, g)| *r == c.row && g[c.group].is_none())
            {
                Some((_, g)) => g[c.group] = Some(v),
                None => {
                    let mut g = [None; 3];
                    g[c.group] = Some(v.as_slice());
                    q.push((c.row, g));
                }
            }
        }
        let ntok = queues.iter().map(Vec::len).max().unwrap_or(0);
        for w0 in (0..ntok).step_by(MAX_T) {
            let t = (ntok - w0).min(MAX_T);
            let (bh, bp) = self.blank();
            let mut h = Vec::with_capacity(t * GDN_CONV);
            let mut p = Vec::with_capacity(t * GDN_PROJ);
            for tt in 0..t {
                let (mut hh, mut pp) = (bh.clone(), bp.clone());
                for (kh, q) in queues.iter().enumerate() {
                    let Some((row, groups)) = q.get(w0 + tt) else {
                        continue;
                    };
                    hh[GDN_HK * D + STORE_HK[kh] * D + row] = 1.0;
                    for (group, v) in groups.iter().enumerate() {
                        let Some(v) = v else { continue };
                        let hv = STORE_HK[kh] + GDN_HK * group;
                        pp[GDN_B + hv] = PIN_B;
                        let vo = 2 * GDN_HK * D + hv * D;
                        for (j, &x) in v.iter().enumerate() {
                            hh[vo + j] = x as f32;
                        }
                        hh[vo + MARK] = 1.0;
                    }
                }
                h.extend_from_slice(&hh);
                p.extend_from_slice(&pp);
            }
            self.step(state, &h, &p, t, Some(t))?;
        }
        Ok(())
    }

    /// One `gdn_step` window: `ReadOnly` when `keep` is `None`, else `Commit` of `keep` tokens.
    fn step(
        &mut self,
        state: &mut Dev::Buffer,
        h: &[f32],
        p: &[f32],
        t: usize,
        keep: Option<usize>,
    ) -> Result<Vec<f32>> {
        let dev = self.dev;
        let (hb, pb) = (dev.upload_f32(h), dev.upload_f32(p));
        let mut y = dev.alloc_f32(t * GDN_V);
        let mode = match keep {
            None => GdnMode::ReadOnly,
            Some(k) => {
                self.win = dev.upload_f32(&[0.0, f32::from_bits(k as u32), 0.0][..Win::WORDS]);
                GdnMode::Commit { win: &self.win }
            }
        };
        dev.gdn_step(
            state,
            &hb,
            &pb,
            GDN_PROJ,
            &self.params,
            &mut y,
            None,
            t,
            mode,
            self.eps,
        );
        self.traffic.steps += 1;
        self.traffic.tokens += t;
        Ok(dev.download(&y))
    }

    /// Integrity scan of the store heads (read-only, straight from the buffer): every value an
    /// integer in `[0, 2^18)`, every marker 0 or 1, every unmarked row all zero. Returns the
    /// poisoned v heads (empty when the store is intact).
    pub fn audit(dev: &Dev, state: &Dev::Buffer) -> Vec<usize> {
        let s = dev.download(state);
        STORE_HV
            .iter()
            .copied()
            .filter(|&hv| {
                (0..D).any(|i| {
                    let r = &s[(i * GDN_HV + hv) * D..(i * GDN_HV + hv + 1) * D];
                    let bad_val = r
                        .iter()
                        .any(|&x| x.fract() != 0.0 || !(0.0..VALUE_LIMIT as f32).contains(&x));
                    let m = r[MARK];
                    bad_val || (m != 0.0 && m != 1.0) || (m == 0.0 && r.iter().any(|&x| x != 0.0))
                })
            })
            .collect()
    }

    /// The store heads' rows out of a state (`[i][12 heads][j]`, 768 KiB): what a cross-session
    /// snapshot keeps, like the post's `store_snapshot` keeps only the persistent heads.
    pub fn snapshot_heads(state: &[f32]) -> Vec<f32> {
        assert_eq!(state.len(), GDN_STATE);
        let mut out = Vec::with_capacity(D * STORE_HV.len() * D);
        for i in 0..D {
            for &hv in &STORE_HV {
                out.extend_from_slice(&state[(i * GDN_HV + hv) * D..(i * GDN_HV + hv + 1) * D]);
            }
        }
        out
    }

    /// Splice a [`snapshot_heads`](Self::snapshot_heads) into `state` (a fresh conversation's
    /// layer state), leaving every other head as it is.
    pub fn restore_heads(state: &mut [f32], snap: &[f32]) {
        assert_eq!(state.len(), GDN_STATE);
        assert_eq!(snap.len(), D * STORE_HV.len() * D);
        for i in 0..D {
            for (n, &hv) in STORE_HV.iter().enumerate() {
                let src = &snap[(i * STORE_HV.len() + n) * D..(i * STORE_HV.len() + n + 1) * D];
                state[(i * GDN_HV + hv) * D..(i * GDN_HV + hv + 1) * D].copy_from_slice(src);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::time::Instant;
    use tang_compute::CpuDevice;

    const EPS: f32 = 1e-6;

    fn env(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(default)
    }

    /// xorshift64*.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
        fn f(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32
        }
    }

    struct Rig {
        dev: CpuDevice,
        conv: Vec<f32>,
        dt: Vec<f32>,
        ssm_a: Vec<f32>,
        norm: Vec<f32>,
    }

    /// Realistic-ish params: nonzero dt_bias, ssm_a in (−8, −0.5), norm in (0.5, 2).
    fn rig(seed: u64) -> Rig {
        let mut r = Rng(seed);
        Rig {
            dev: CpuDevice::new(),
            conv: vec![0.0],
            dt: (0..GDN_HV).map(|_| r.f() * 4.0 - 2.0).collect(),
            ssm_a: (0..GDN_HV).map(|_| -0.5 - 7.5 * r.f()).collect(),
            norm: (0..D).map(|_| 0.5 + 1.5 * r.f()).collect(),
        }
    }

    struct Bufs {
        conv: <CpuDevice as ComputeDevice>::Buffer,
        dt: <CpuDevice as ComputeDevice>::Buffer,
        ssm_a: <CpuDevice as ComputeDevice>::Buffer,
        norm: <CpuDevice as ComputeDevice>::Buffer,
    }

    impl Rig {
        fn bufs(&self) -> Bufs {
            Bufs {
                conv: self.dev.upload_f32(&self.conv),
                dt: self.dev.upload_f32(&self.dt),
                ssm_a: self.dev.upload_f32(&self.ssm_a),
                norm: self.dev.upload_f32(&self.norm),
            }
        }
    }

    impl Bufs {
        fn params(&self) -> GdnParams<'_, <CpuDevice as ComputeDevice>::Buffer> {
            GdnParams {
                conv: &self.conv,
                dt_bias: &self.dt,
                ssm_a: &self.ssm_a,
                norm: &self.norm,
            }
        }
    }

    /// A layer state whose base heads hold noise (what a conversation leaves) and whose store
    /// heads are zero.
    fn noisy_state(seed: u64) -> Vec<f32> {
        let mut r = Rng(seed);
        let mut s = vec![0f32; GDN_STATE];
        for i in 0..D {
            for hv in 0..GDN_HV {
                if STORE_HV.contains(&hv) {
                    continue;
                }
                for j in 0..D {
                    s[(i * GDN_HV + hv) * D + j] = r.f() * 2.0 - 1.0;
                }
            }
        }
        s
    }

    fn base_heads_eq(a: &[f32], b: &[f32]) {
        for i in 0..D {
            for hv in (0..GDN_HV).filter(|h| !STORE_HV.contains(h)) {
                for j in 0..D {
                    let k = (i * GDN_HV + hv) * D + j;
                    assert_eq!(
                        a[k].to_bits(),
                        b[k].to_bits(),
                        "base head {hv} row {i} col {j}"
                    );
                }
            }
        }
    }

    fn file(r: &mut Rng, len: usize) -> Vec<u32> {
        (0..len)
            .map(|_| r.below(VALUE_LIMIT as usize) as u32)
            .collect()
    }

    /// 512 files of mixed sizes (1..=127 tokens, one content row each: all 512 rows) fill the
    /// store to its file and row limit (default 128 files in debug, ~25 s; `TANG_MEMSTORE_FILES=512`
    /// for the full fill, run in release); every file reads back exactly, the 513th put fails, and
    /// the base heads are bitwise untouched.
    #[test]
    fn memstore_fills_to_capacity() {
        let t0 = Instant::now();
        let rig = rig(1);
        let b = rig.bufs();
        let before = noisy_state(2);
        let mut state = rig.dev.upload_f32(&before);
        let mut st = MemStore::open(&rig.dev, b.params(), EPS, &mut state).unwrap();
        let n = env("TANG_MEMSTORE_FILES", 128);
        let mut r = Rng(3);
        let mut want = Vec::new();
        for i in 0..n {
            let name = format!("notes/{i:03}-{}.md", r.below(1 << 20)).into_bytes();
            let data = {
                let n = 1 + r.below(ROW_TOKENS);
                file(&mut r, n)
            };
            st.put(&mut state, &name, &data).unwrap();
            want.push((name, data));
        }
        let cap = st.capacity();
        assert_eq!(cap.files, n);
        assert_eq!(cap.tokens, want.iter().map(|(_, d)| d.len()).sum::<usize>());
        let puts = st.traffic();
        for (name, data) in &want {
            assert_eq!(st.get(&mut state, name).unwrap().as_ref(), Some(data));
        }
        if n == MAX_FILES {
            assert!(st.put(&mut state, b"one-too-many", &[1]).is_err());
        }
        assert!(MemStore::<CpuDevice>::audit(&rig.dev, &state).is_empty());
        base_heads_eq(&rig.dev.download(&state), &before);
        let all = st.traffic();
        eprintln!(
            "memstore: {n} files, {} tokens; puts {} steps/{} tokens, gets {} steps/{} tokens, {:?}",
            cap.tokens,
            puts.steps,
            puts.tokens,
            all.steps - puts.steps,
            all.tokens - puts.tokens,
            t0.elapsed()
        );
    }

    /// Big files (multi-row, up to the per-file limit), overwrite with a different size, delete,
    /// and the free-row bookkeeping.
    #[test]
    fn memstore_overwrite_delete() {
        let rig = rig(4);
        let b = rig.bufs();
        let mut state = rig.dev.upload_f32(&noisy_state(5));
        let mut st = MemStore::open(&rig.dev, b.params(), EPS, &mut state).unwrap();
        let mut r = Rng(6);
        let big = file(&mut r, MAX_FILE_TOKENS);
        st.put(&mut state, b"big", &big).unwrap();
        assert_eq!(st.capacity().rows_used, MAX_FILE_ROWS);
        assert!(st
            .put(&mut state, b"too-big", &file(&mut r, MAX_FILE_TOKENS + 1))
            .is_err());
        assert!(st.put(&mut state, b"bad", &[VALUE_LIMIT]).is_err());
        let small = file(&mut r, 300);
        st.put(&mut state, b"small", &small).unwrap();
        assert_eq!(st.get(&mut state, b"big").unwrap(), Some(big));
        let big2 = file(&mut r, 1000);
        st.put(&mut state, b"big", &big2).unwrap();
        assert_eq!(st.capacity().rows_used, 8 + 3);
        assert_eq!(st.capacity().files, 2);
        assert_eq!(st.get(&mut state, b"big").unwrap(), Some(big2));
        st.put(&mut state, b"empty", &[]).unwrap();
        assert_eq!(st.get(&mut state, b"empty").unwrap(), Some(vec![]));
        assert!(st.delete(&mut state, b"big").unwrap());
        assert!(!st.delete(&mut state, b"big").unwrap());
        assert_eq!(st.get(&mut state, b"big").unwrap(), None);
        assert_eq!(st.get(&mut state, b"small").unwrap(), Some(small));
        assert_eq!(st.capacity().rows_used, 3);
        let mut names: Vec<_> = st.list(&mut state).unwrap();
        names.sort();
        assert_eq!(
            names,
            vec![(b"empty".to_vec(), 0), (b"small".to_vec(), 300)]
        );
    }

    /// Names that share a home slot (and fill its bucket, so the probe crosses into the next
    /// window and wraps the table's end) all store and read, and a delete in the middle of the
    /// chain keeps the names after it reachable; the tombstone is reused.
    #[test]
    fn memstore_collisions() {
        let rig = rig(7);
        let b = rig.bufs();
        let mut state = rig.dev.upload_f32(&noisy_state(8));
        let mut st = MemStore::open(&rig.dev, b.params(), EPS, &mut state).unwrap();
        // names homing to slot 1021 (3 slots before the end): the chain wraps to slot 0
        let mut names = Vec::new();
        let mut i = 0u64;
        while names.len() < 20 {
            let n = format!("c{i}").into_bytes();
            if home(&n) == SLOTS - 3 {
                names.push(n);
            }
            i += 1;
        }
        for (k, n) in names.iter().enumerate() {
            st.put(&mut state, n, &[k as u32, 7]).unwrap();
        }
        for (k, n) in names.iter().enumerate() {
            assert_eq!(st.get(&mut state, n).unwrap(), Some(vec![k as u32, 7]));
        }
        assert!(st.delete(&mut state, &names[2]).unwrap());
        for (k, n) in names.iter().enumerate().skip(3) {
            assert_eq!(st.get(&mut state, n).unwrap(), Some(vec![k as u32, 7]));
        }
        st.put(&mut state, &names[2], &[99]).unwrap();
        assert_eq!(st.get(&mut state, &names[2]).unwrap(), Some(vec![99]));
        assert_eq!(st.capacity().files, 20);
        let s = rig.dev.download(&state);
        // slots 1021..1023 then 0..: slot 0 is bucket 0 row 0 of v head 12, status live
        assert_eq!(s[12 * D + COL_STATUS], LIVE as f32);
    }

    /// A random workload of put / overwrite / get / delete / miss against a `HashMap`, with
    /// small files so the store churns through its slots (tombstones) and rows. Default 300
    /// operations (~20 s in debug, ~65 ms per window); `TANG_MEMSTORE_OPS=10000` runs the full 10k (release: see the design doc).
    #[test]
    fn memstore_random_workload() {
        let t0 = Instant::now();
        let rig = rig(9);
        let b = rig.bufs();
        let before = noisy_state(10);
        let mut state = rig.dev.upload_f32(&before);
        let mut st = MemStore::open(&rig.dev, b.params(), EPS, &mut state).unwrap();
        let ops = env("TANG_MEMSTORE_OPS", 300);
        let mut r = Rng(11);
        let mut model: HashMap<Vec<u8>, Vec<u32>> = HashMap::new();
        for _ in 0..ops {
            let name = format!("k{}", r.below(600)).into_bytes();
            match r.below(10) {
                0..=3 => {
                    let len = [0, 1, 50, 127, 128, 400][r.below(6)];
                    let data = file(&mut r, len);
                    let fits = model.contains_key(&name) || model.len() < MAX_FILES;
                    let rows = st.capacity().rows_used;
                    match st.put(&mut state, &name, &data) {
                        Ok(()) => {
                            model.insert(name, data);
                        }
                        Err(e) => assert!(
                            !fits || rows + len.div_ceil(ROW_TOKENS) > ROWS,
                            "put failed: {e}"
                        ),
                    }
                }
                4..=5 => {
                    assert_eq!(
                        st.delete(&mut state, &name).unwrap(),
                        model.remove(&name).is_some()
                    )
                }
                _ => assert_eq!(
                    st.get(&mut state, &name).unwrap().as_ref(),
                    model.get(&name)
                ),
            }
            assert_eq!(st.capacity().files, model.len());
        }
        let mut listed: Vec<_> = st.list(&mut state).unwrap();
        listed.sort();
        let mut want: Vec<_> = model.iter().map(|(k, v)| (k.clone(), v.len())).collect();
        want.sort();
        assert_eq!(listed, want);
        assert!(MemStore::<CpuDevice>::audit(&rig.dev, &state).is_empty());
        base_heads_eq(&rig.dev.download(&state), &before);
        eprintln!(
            "memstore: {ops} random ops, {:?} ({:?})",
            st.traffic(),
            t0.elapsed()
        );
    }

    /// Persistence: (1) the whole layer buffer, the bytes `save_running` carries, reloaded into
    /// a fresh device buffer, reopens to the same store; (2) only the store heads, spliced into
    /// another conversation's state (different base-head contents), read back exactly while
    /// leaving that conversation's base heads alone.
    #[test]
    fn memstore_persists() {
        let rig = rig(12);
        let b = rig.bufs();
        let mut state = rig.dev.upload_f32(&noisy_state(13));
        let mut r = Rng(14);
        let mut want = Vec::new();
        {
            let mut st = MemStore::open(&rig.dev, b.params(), EPS, &mut state).unwrap();
            for i in 0..40 {
                let name = format!("mem/{i}").into_bytes();
                let data = {
                    let n = r.below(700);
                    file(&mut r, n)
                };
                st.put(&mut state, &name, &data).unwrap();
                want.push((name, data));
            }
            st.delete(&mut state, b"mem/7").unwrap();
            want.remove(7);
        }
        let saved: Vec<f32> = rig.dev.download(&state); // RunningState's part for this layer
        let cap = |st: &MemStore<CpuDevice>| st.capacity();

        let mut fresh = rig.dev.upload_f32(&saved);
        let mut st = MemStore::open(&rig.dev, b.params(), EPS, &mut fresh).unwrap();
        assert_eq!(cap(&st).files, 39);
        for (name, data) in &want {
            assert_eq!(st.get(&mut fresh, name).unwrap().as_ref(), Some(data));
        }

        let snap = MemStore::<CpuDevice>::snapshot_heads(&saved);
        let other = noisy_state(15);
        let mut spliced = other.clone();
        MemStore::<CpuDevice>::restore_heads(&mut spliced, &snap);
        let mut s2 = rig.dev.upload_f32(&spliced);
        let mut st2 = MemStore::open(&rig.dev, b.params(), EPS, &mut s2).unwrap();
        assert_eq!(cap(&st2), cap(&st));
        for (name, data) in &want {
            assert_eq!(st2.get(&mut s2, name).unwrap().as_ref(), Some(data));
        }
        base_heads_eq(&rig.dev.download(&s2), &other);
    }

    /// One ordinary model token through the layer (real gates, random k on every head) poisons
    /// the store heads: `audit` names them, and a kernel read of a written row fails instead of
    /// returning wrong data (or, for a uniformly decayed row the ratio read survives, the audit
    /// is what catches it).
    #[test]
    fn memstore_detects_poison() {
        let rig = rig(16);
        let b = rig.bufs();
        let mut state = rig.dev.upload_f32(&noisy_state(17));
        let mut st = MemStore::open(&rig.dev, b.params(), EPS, &mut state).unwrap();
        st.put(&mut state, b"x", &[1, 2, 3]).unwrap();
        assert!(MemStore::<CpuDevice>::audit(&rig.dev, &state).is_empty());
        // a model token: g < 1, β ≈ 0.6, dense k and v
        let mut r = Rng(18);
        let h: Vec<f32> = (0..GDN_CONV).map(|_| r.f() - 0.5).collect();
        let mut p = vec![0f32; GDN_PROJ];
        for hv in 0..GDN_HV {
            p[GDN_A + hv] = 0.3;
            p[GDN_B + hv] = 0.4;
        }
        let win = rig.dev.upload_f32(&[0.0, f32::from_bits(1), 0.0]);
        let mut y = rig.dev.alloc_f32(GDN_V);
        rig.dev.gdn_step(
            &mut state,
            &rig.dev.upload_f32(&h),
            &rig.dev.upload_f32(&p),
            GDN_PROJ,
            &b.params(),
            &mut y,
            None,
            1,
            GdnMode::Commit { win: &win },
            EPS,
        );
        let bad = MemStore::<CpuDevice>::audit(&rig.dev, &state);
        assert_eq!(bad, STORE_HV.to_vec());
        assert!(
            st.get(&mut state, b"x").is_err()
                || st.get(&mut state, b"x").unwrap() != Some(vec![1, 2, 3])
        );
    }

    /// The kernel stored what the store claims: a written row is exactly the integers in the
    /// state buffer (value, marker), in the documented head and row, and nothing else of the
    /// store heads moved.
    #[test]
    fn memstore_rows_are_exact_in_state() {
        let rig = rig(19);
        let b = rig.bufs();
        let mut state = rig.dev.upload_f32(&vec![0f32; GDN_STATE]);
        let mut st = MemStore::open(&rig.dev, b.params(), EPS, &mut state).unwrap();
        let data: Vec<u32> = (0..200).map(|i| VALUE_LIMIT - 1 - i).collect();
        st.put(&mut state, b"exact", &data).unwrap();
        let s = rig.dev.download(&state);
        // rows 0 and 1 of the content space: v heads 44 and 45, state row 0
        for (c, chunk) in data.chunks(ROW_TOKENS).enumerate() {
            let hv = 44 + c;
            for (j, &v) in chunk.iter().enumerate() {
                assert_eq!(s[hv * D + j].to_bits(), (v as f32).to_bits());
            }
            assert_eq!(s[hv * D + MARK], 1.0);
        }
        let slot = home(b"exact");
        let c = Cell::slot(slot);
        let row = &s[(c.row * GDN_HV + c.hv()) * D..(c.row * GDN_HV + c.hv() + 1) * D];
        assert_eq!(row[COL_STATUS], 1.0);
        assert_eq!(row[COL_LEN], 200.0);
        assert_eq!(row[MARK], 1.0);
        let written = s.iter().filter(|&&x| x != 0.0).count();
        let want = data.iter().filter(|&&x| x != 0).count()
            + 2
            + row.iter().filter(|&&x| x != 0.0).count();
        assert_eq!(written, want);
    }
}
