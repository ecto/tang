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
//! # Layouts
//!
//! A [`Layout`] says how the GDN layer's heads are shaped (k heads, v heads, head dim, which k
//! head a v head reads) and which of them the store owns. There are two, picked from the state
//! size (or the engine's hparams, [`Layout::of_shape`]):
//!
//! - [`Layout::FLASH`]: 16 k / 48 v heads, tiled (v head `hv` on k head `hv % 16`, the
//!   Flash-Next kernel). The store owns k heads 12..=15 and all three of their v heads, 12 v
//!   heads of 48 (below).
//! - [`Layout::CODER`]: 16 k / 32 v heads, grouped (v head `hv` on k head `hv / 2`, qwen3next
//!   and Qwen3.5-4B). The store owns k heads 12..=15 and both of their v heads, `24..=31`, 8 of
//!   32: v head `2·kh` (group 0) is directory, `2·kh + 1` (group 1) content. The directory is
//!   one group, so it has 512 slots (256 files), 4 to a bucket; content is unchanged (512 rows).
//!   The Flash-Next device kernel only knows the tiled 48-head shape, so this layout's windows
//!   run through the reference recurrence (`reference::gdn_head_step`) on the host.
//!
//! ## The 48-head layout (state `[i 128][hv 48][j 128]` f32, v head `hv` on k head `hv % 16`)
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
use tang_compute::flash::shape::{GDN_D, GDN_HK, GDN_HV};
use tang_compute::flash::{GdnMode, GdnParams, Win, GDN_STATE, MAX_T};
use tang_compute::ComputeDevice;

use super::reference::{gdn_head_step, sigmoid, silu, softplus};

const D: usize = GDN_D;
/// The store's k heads (both layouts).
pub const STORE_HK: [usize; 4] = [12, 13, 14, 15];
/// The 48-head layout's store v heads (every v head of [`STORE_HK`], tiled).
pub const STORE_HV: [usize; 12] = [12, 13, 14, 15, 28, 29, 30, 31, 44, 45, 46, 47];
/// The 32-head layout's store v heads (every v head of [`STORE_HK`], grouped: `2·kh`, `2·kh + 1`).
pub const STORE_HV_32: [usize; 8] = [24, 25, 26, 27, 28, 29, 30, 31];
/// Directory slots of the 48-head layout ([`Layout::slots`] in general).
pub const SLOTS: usize = 1024;
/// Content rows (both layouts).
pub const ROWS: usize = 512;
/// Live files at most in the 48-head layout (load factor 1/2, as in the post).
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
/// 1 / sqrt(128) as f32 (the kernel's q scale).
const INV_SQRT_D: f32 = f32::from_bits(0x3db5_04f3);

/// Which k head a v head reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KMap {
    /// `hv % hk` (Flash-Next, GGUF order).
    Tiled,
    /// `hv / (hv_count / hk)` (qwen3next, Qwen3.5).
    Grouped,
}

/// The output gate on `z`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gate {
    Sigmoid,
    Silu,
}

/// A GDN layer's head shape and the store's place in it (see the module doc).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    /// For errors and JSON.
    pub name: &'static str,
    /// k heads.
    pub hk: usize,
    /// v heads.
    pub hv: usize,
    /// Head dim (state is `d × d` per v head).
    pub d: usize,
    pub kmap: KMap,
    pub gate: Gate,
    /// The store's k heads.
    pub store_hk: &'static [usize],
    /// The store's v heads, in snapshot order.
    pub store_hv: &'static [usize],
}

impl Layout {
    /// Flash-Next: 16 k / 48 v, tiled; 1,024 slots, 512 rows, 12 store heads.
    pub const FLASH: Layout = Layout {
        name: "16k/48v tiled",
        hk: GDN_HK,
        hv: GDN_HV,
        d: D,
        kmap: KMap::Tiled,
        gate: Gate::Sigmoid,
        store_hk: &STORE_HK,
        store_hv: &STORE_HV,
    };
    /// Qwen3-Coder-Next / Qwen3.5-4B: 16 k / 32 v, grouped; 512 slots, 512 rows, 8 store heads.
    pub const CODER: Layout = Layout {
        name: "16k/32v grouped",
        hk: 16,
        hv: 32,
        d: D,
        kmap: KMap::Grouped,
        gate: Gate::Silu,
        store_hk: &STORE_HK,
        store_hv: &STORE_HV_32,
    };
    /// Every layout.
    pub const ALL: [Layout; 2] = [Layout::FLASH, Layout::CODER];

    /// The layout for GDN layers of `hk` k / `hv` v heads.
    pub fn of_shape(hk: usize, hv: usize) -> Result<Layout> {
        Self::ALL
            .into_iter()
            .find(|l| l.hk == hk && l.hv == hv)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "the memstore has layouts for GDN layers of 16 k / 48 v (Flash-Next) and \
                     16 k / 32 v heads (Coder-Next, Qwen3.5); this model's have {hk} k / {hv} v"
                )
            })
    }

    /// The layout whose layer state is `n` floats.
    pub fn of_state_len(n: usize) -> Result<Layout> {
        Self::ALL
            .into_iter()
            .find(|l| l.state_len() == n)
            .ok_or_else(|| anyhow::anyhow!("a {n}-float GDN state matches no memstore layout"))
    }

    /// The layout whose store heads are `n` floats.
    pub fn of_heads_len(n: usize) -> Result<Layout> {
        Self::ALL
            .into_iter()
            .find(|l| l.head_floats() == n)
            .ok_or_else(|| anyhow::anyhow!("{n} store-head floats match no memstore layout"))
    }

    /// The k head v head `vh` reads (`Hparams::gdn_k_head`).
    pub fn k_head(&self, vh: usize) -> usize {
        match self.kmap {
            KMap::Tiled => vh % self.hk,
            KMap::Grouped => vh / (self.hv / self.hk),
        }
    }
    /// v heads per k head.
    pub fn groups(&self) -> usize {
        self.hv / self.hk
    }
    /// Groups holding the directory (all but the last, which is content).
    pub fn dir_groups(&self) -> usize {
        self.groups() - 1
    }
    /// v head of store k head `kh` (index into [`store_hk`](Self::store_hk)), group `g`.
    pub fn store_v(&self, kh: usize, g: usize) -> usize {
        let k = self.store_hk[kh];
        match self.kmap {
            KMap::Tiled => k + self.hk * g,
            KMap::Grouped => k * self.groups() + g,
        }
    }
    /// Slots in a bucket (one state row of every directory head).
    pub fn bucket_slots(&self) -> usize {
        self.store_hk.len() * self.dir_groups()
    }
    /// Directory slots.
    pub fn slots(&self) -> usize {
        self.bucket_slots() * self.d
    }
    /// Buckets.
    pub fn buckets(&self) -> usize {
        self.d
    }
    /// Live files at most (load factor 1/2).
    pub fn max_files(&self) -> usize {
        self.slots() / 2
    }
    /// Content rows.
    pub fn rows(&self) -> usize {
        self.store_hk.len() * self.d
    }
    /// Tokens the content heads hold at most.
    pub fn max_tokens(&self) -> usize {
        self.rows() * ROW_TOKENS
    }
    /// Floats in a layer state.
    pub fn state_len(&self) -> usize {
        self.d * self.hv * self.d
    }
    /// Floats in the store heads (a blob without its header).
    pub fn head_floats(&self) -> usize {
        self.d * self.store_hv.len() * self.d
    }
    /// Conv channels `[q | k | v]`.
    pub fn conv(&self) -> usize {
        2 * self.hk * self.d + self.hv * self.d
    }
    /// Offsets in the stacked projection `[qkv | z | a | b]`.
    pub fn z(&self) -> usize {
        self.conv()
    }
    pub fn a(&self) -> usize {
        self.z() + self.hv * self.d
    }
    pub fn b(&self) -> usize {
        self.a() + self.hv
    }
    /// Width of the stacked projection.
    pub fn proj(&self) -> usize {
        self.b() + self.hv
    }
    /// Kernel output width per token.
    pub fn v_width(&self) -> usize {
        self.hv * self.d
    }
    /// Whether the device's `gdn_step` (fixed to the Flash-Next shape) runs this layout.
    pub fn on_device(&self) -> bool {
        *self == Layout::FLASH
    }
}

/// A read row: `((k head, row), [group] values)`.
type Decoded = ((usize, usize), Vec<Vec<u32>>);
/// One token's write on a k head: `(row, [group] values)`.
type RowWrite<'a> = (usize, Vec<Option<&'a [u32]>>);

/// A row's place: store k head index (0..4), v group, row.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Cell {
    kh: usize,
    group: usize,
    row: usize,
}

impl Cell {
    #[cfg_attr(not(test), allow(dead_code))]
    fn hv(self, l: &Layout) -> usize {
        l.store_v(self.kh, self.group)
    }
    fn slot(l: &Layout, s: usize) -> Self {
        let n = l.store_hk.len();
        Cell {
            kh: s % n,
            group: (s / n) % l.dir_groups(),
            row: s / l.bucket_slots(),
        }
    }
    fn content(l: &Layout, c: usize) -> Self {
        let n = l.store_hk.len();
        Cell {
            kh: c % n,
            group: l.groups() - 1,
            row: c / n,
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

/// The FNV-1a hash of a name, its home slot among `slots`.
fn home(name: &[u8], slots: usize) -> usize {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in name {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    (h % slots as u64) as usize
}

/// The store over one GDN layer's state buffer; the buffer itself is passed to each call (it
/// stays the engine's). The struct holds the driver's derived bookkeeping only (free rows, live
/// count), which [`open`](Self::open) rebuilds from the state by a directory scan.
pub struct MemStore<'p, Dev: ComputeDevice> {
    dev: &'p Dev,
    layout: Layout,
    /// Run windows through the host reference recurrence instead of `dev.gdn_step`.
    host_ref: bool,
    params: GdnParams<'p, Dev::Buffer>,
    dt: Vec<f32>,
    ssm_a: Vec<f32>,
    norm: Vec<f32>,
    eps: f32,
    win: Dev::Buffer,
    free: BTreeSet<usize>,
    files: usize,
    tokens: usize,
    traffic: Traffic,
}

impl<'p, Dev: ComputeDevice> MemStore<'p, Dev> {
    /// Attach to `state` (one layer's state, whose size picks the [`Layout`]) with that layer's
    /// params, rebuilding the bookkeeping by reading the whole directory through the kernel
    /// (16 windows). The 48-head layout runs on `dev.gdn_step`; the 32-head one on the host
    /// reference recurrence (the device kernel is fixed to the 48-head shape).
    pub fn open(
        dev: &'p Dev,
        params: GdnParams<'p, Dev::Buffer>,
        eps: f32,
        state: &mut Dev::Buffer,
    ) -> Result<Self> {
        let layout = Layout::of_state_len(dev.download(state).len())?;
        Self::open_with(dev, layout, !layout.on_device(), params, eps, state)
    }

    /// [`open`](Self::open) with the layout and the kernel path given (`host_ref`: the host
    /// reference recurrence; otherwise `dev.gdn_step`, which needs [`Layout::on_device`]).
    pub fn open_with(
        dev: &'p Dev,
        layout: Layout,
        host_ref: bool,
        params: GdnParams<'p, Dev::Buffer>,
        eps: f32,
        state: &mut Dev::Buffer,
    ) -> Result<Self> {
        ensure!(layout.d == D, "the memstore needs head dim {D}");
        ensure!(
            host_ref || layout.on_device(),
            "the device's gdn_step only runs the {} layout, not {}",
            Layout::FLASH.name,
            layout.name
        );
        ensure!(
            dev.download(state).len() == layout.state_len(),
            "the state isn't a {} GDN layer state",
            layout.name
        );
        let dt = dev.download(params.dt_bias);
        let ssm_a = dev.download(params.ssm_a);
        let norm = dev.download(params.norm);
        ensure!(
            dt.len() >= layout.hv && norm.len() >= D && (!host_ref || ssm_a.len() >= layout.hv),
            "GDN params too short"
        );
        ensure!(
            norm.iter().all(|x| x.is_finite() && *x != 0.0),
            "ssm_norm has a zero or non-finite weight: the store can't read through it"
        );
        let rows = layout.rows();
        let mut s = MemStore {
            dev,
            layout,
            host_ref,
            params,
            dt,
            ssm_a,
            norm,
            eps,
            win: dev.upload_f32(&[0.0; Win::WORDS]),
            free: (0..rows).collect(),
            files: 0,
            tokens: 0,
            traffic: Traffic::default(),
        };
        for (_, slot) in s.scan(state)? {
            for &c in &slot.rows {
                ensure!(c < rows && s.free.remove(&c), "row {c} is claimed twice");
            }
            s.files += 1;
            s.tokens += slot.len;
        }
        Ok(s)
    }

    /// The layout.
    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// Files, rows and tokens.
    pub fn capacity(&self) -> Capacity {
        Capacity {
            files: self.files,
            max_files: self.layout.max_files(),
            rows_used: self.layout.rows() - self.free.len(),
            tokens: self.tokens,
            max_tokens: self.layout.max_tokens(),
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
        let max_files = self.layout.max_files();
        let p = self.probe(state, name)?;
        let (slot, old) = match (p.found, p.free) {
            (Some((s, old)), _) => (s, Some(old)),
            (None, Some(s)) if self.files < max_files => (s, None),
            (None, Some(_)) => bail!("store full: {max_files} files"),
            (None, None) => bail!("directory full"),
        };
        let l = self.layout;
        let rows: Vec<usize> = self.free.iter().take(need).copied().collect();
        let mut writes: Vec<(Cell, Vec<u32>)> = rows
            .iter()
            .zip(data.chunks(ROW_TOKENS))
            .map(|(&c, chunk)| (Cell::content(&l, c), chunk.to_vec()))
            .collect();
        let entry = Slot {
            status: LIVE,
            name: name.to_vec(),
            len: data.len(),
            rows: rows.clone(),
        };
        writes.push((Cell::slot(&l, slot), entry.encode()));
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
        let l = self.layout;
        let cells: Vec<Cell> = slot.rows.iter().map(|&c| Cell::content(&l, c)).collect();
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
        let l = self.layout;
        self.write(state, &[(Cell::slot(&l, s), tomb.encode())])?;
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
        let l = self.layout;
        let bs = l.bucket_slots();
        let flat: Vec<Cell> = (0..l.buckets()).flat_map(|b| Self::bucket(&l, b)).collect();
        let got = self.read(state, &flat)?;
        Ok(got
            .chunks(bs)
            .enumerate()
            .flat_map(|(b, slots)| {
                slots
                    .iter()
                    .enumerate()
                    .map(move |(i, r)| (b * bs + i, Slot::decode(r)))
            })
            .filter(|(_, s)| s.status == LIVE)
            .collect())
    }

    fn bucket(l: &Layout, b: usize) -> Vec<Cell> {
        let bs = l.bucket_slots();
        (b * bs..b * bs + bs).map(|s| Cell::slot(l, s)).collect()
    }

    /// Linear probing from the name's home slot. Buckets (one token each) are read lazily: a
    /// 1-token window for the home bucket, then 8-token windows, so a lookup at load 1/2 is
    /// one window and rarely two.
    fn probe(&mut self, state: &mut Dev::Buffer, name: &[u8]) -> Result<Probe> {
        let l = self.layout;
        let (slots, bs, buckets) = (l.slots(), l.bucket_slots(), l.buckets());
        let h = home(name, slots);
        let mut read: Vec<Option<Vec<u32>>> = vec![None; slots];
        let mut next_bucket = h / bs;
        let mut fetched = 0;
        let mut free = None;
        for pos in 0..slots {
            let s = (h + pos) % slots;
            if read[s].is_none() {
                let n = if fetched == 0 {
                    1
                } else {
                    MAX_T.min(buckets - fetched)
                };
                let bks: Vec<usize> = (0..n).map(|i| (next_bucket + i) % buckets).collect();
                let cells: Vec<Cell> = bks.iter().flat_map(|&b| Self::bucket(&l, b)).collect();
                let got = self.read(state, &cells)?;
                for (k, r) in got.into_iter().enumerate() {
                    read[bks[k / bs] * bs + k % bs] = Some(r);
                }
                next_bucket = (next_bucket + n) % buckets;
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
        let l = &self.layout;
        let mut p = vec![0f32; l.proj()];
        for hv in 0..l.hv {
            p[l.a() + hv] = PIN_A - self.dt[hv];
            p[l.b() + hv] = -PIN_B;
        }
        for &hv in l.store_hv {
            p[l.z() + hv * D..l.z() + (hv + 1) * D].fill(PIN_Z);
        }
        (vec![0f32; l.conv()], p)
    }

    /// Read `cells` (rows of 127 values) through `ReadOnly` windows. A token queries one row per
    /// store k head and returns that row of all of its v heads.
    fn read(&mut self, state: &mut Dev::Buffer, cells: &[Cell]) -> Result<Vec<Vec<u32>>> {
        let l = self.layout;
        // per k head, the distinct rows wanted, in order
        let mut queues: Vec<Vec<usize>> = vec![Vec::new(); l.store_hk.len()];
        for c in cells {
            if !queues[c.kh].contains(&c.row) {
                queues[c.kh].push(c.row);
            }
        }
        let ntok = queues.iter().map(Vec::len).max().unwrap_or(0);
        let vw = l.v_width();
        // decoded[(kh, row)] = [group][127]
        let mut decoded: Vec<Decoded> = Vec::new();
        for w0 in (0..ntok).step_by(MAX_T) {
            let t = (ntok - w0).min(MAX_T);
            let (bh, bp) = self.blank();
            let mut h = Vec::with_capacity(t * l.conv());
            let mut p = Vec::with_capacity(t * l.proj());
            for tt in 0..t {
                let mut hh = bh.clone();
                for (kh, q) in queues.iter().enumerate() {
                    if let Some(&row) = q.get(w0 + tt) {
                        hh[l.store_hk[kh] * D + row] = 1.0;
                    }
                }
                h.extend_from_slice(&hh);
                p.extend_from_slice(&bp);
            }
            let y = self.step(state, &h, &p, t, None)?;
            for tt in 0..t {
                for (kh, q) in queues.iter().enumerate() {
                    if let Some(&row) = q.get(w0 + tt) {
                        let groups = (0..l.groups())
                            .map(|group| {
                                let hv = l.store_v(kh, group);
                                self.decode(&y[tt * vw + hv * D..tt * vw + (hv + 1) * D], hv, row)
                            })
                            .collect::<Result<Vec<_>>>()?;
                        decoded.push(((kh, row), groups));
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
    /// into any of its v heads (β = 1 there, β ≈ 0 with v = 0 in the others).
    fn write(&mut self, state: &mut Dev::Buffer, writes: &[(Cell, Vec<u32>)]) -> Result<()> {
        let l = self.layout;
        // per k head, (row, [group] values)
        let mut queues: Vec<Vec<RowWrite<'_>>> = vec![Vec::new(); l.store_hk.len()];
        for (c, v) in writes {
            debug_assert!(v.len() <= ROW_TOKENS);
            let q = &mut queues[c.kh];
            match q
                .iter_mut()
                .find(|(r, g)| *r == c.row && g[c.group].is_none())
            {
                Some((_, g)) => g[c.group] = Some(v),
                None => {
                    let mut g = vec![None; l.groups()];
                    g[c.group] = Some(v.as_slice());
                    q.push((c.row, g));
                }
            }
        }
        let ntok = queues.iter().map(Vec::len).max().unwrap_or(0);
        for w0 in (0..ntok).step_by(MAX_T) {
            let t = (ntok - w0).min(MAX_T);
            let (bh, bp) = self.blank();
            let mut h = Vec::with_capacity(t * l.conv());
            let mut p = Vec::with_capacity(t * l.proj());
            for tt in 0..t {
                let (mut hh, mut pp) = (bh.clone(), bp.clone());
                for (kh, q) in queues.iter().enumerate() {
                    let Some((row, groups)) = q.get(w0 + tt) else {
                        continue;
                    };
                    hh[l.hk * D + l.store_hk[kh] * D + row] = 1.0;
                    for (group, v) in groups.iter().enumerate() {
                        let Some(v) = v else { continue };
                        let hv = l.store_v(kh, group);
                        pp[l.b() + hv] = PIN_B;
                        let vo = 2 * l.hk * D + hv * D;
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

    /// One window: `ReadOnly` when `keep` is `None`, else `Commit` of `keep` tokens.
    fn step(
        &mut self,
        state: &mut Dev::Buffer,
        h: &[f32],
        p: &[f32],
        t: usize,
        keep: Option<usize>,
    ) -> Result<Vec<f32>> {
        self.traffic.steps += 1;
        self.traffic.tokens += t;
        if self.host_ref {
            return Ok(self.host_step(state, h, p, t, keep));
        }
        let dev = self.dev;
        let (hb, pb) = (dev.upload_f32(h), dev.upload_f32(p));
        let mut y = dev.alloc_f32(t * self.layout.v_width());
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
            self.layout.proj(),
            &self.params,
            &mut y,
            None,
            t,
            mode,
            self.eps,
        );
        Ok(dev.download(&y))
    }

    /// [`step`](Self::step) on the host: the reference recurrence
    /// (`reference::gdn_head_step`) per v head, each reading k head [`Layout::k_head`], with
    /// the device kernel's gates (`g = exp(softplus(a + dt) · ssm_a)`, `β = σ(b)`), q scale
    /// and gated RMS norm. The state is downloaded and, for a commit, uploaded back.
    fn host_step(
        &self,
        state: &mut Dev::Buffer,
        h: &[f32],
        p: &[f32],
        t: usize,
        keep: Option<usize>,
    ) -> Vec<f32> {
        let l = &self.layout;
        let d = D;
        let mut s = self.dev.download(state);
        let n_run = keep.map_or(t, |k| k.min(t));
        let vw = l.v_width();
        let mut y = vec![0f32; t * vw];
        let (mut st, mut sk, mut o, mut q) = (
            vec![0f32; d * d],
            vec![0f32; d],
            vec![0f32; d],
            vec![0f32; d],
        );
        for hv in 0..l.hv {
            let kh = l.k_head(hv);
            for i in 0..d {
                st[i * d..(i + 1) * d]
                    .copy_from_slice(&s[(i * l.hv + hv) * d..(i * l.hv + hv + 1) * d]);
            }
            for tt in 0..n_run {
                let pp = &p[tt * l.proj()..(tt + 1) * l.proj()];
                let hh = &h[tt * l.conv()..(tt + 1) * l.conv()];
                let g = (softplus(pp[l.a() + hv] + self.dt[hv]) * self.ssm_a[hv]).exp();
                let beta = sigmoid(pp[l.b() + hv]);
                for (qa, &x) in q.iter_mut().zip(&hh[kh * d..(kh + 1) * d]) {
                    *qa = x * INV_SQRT_D;
                }
                let k = &hh[l.hk * d + kh * d..l.hk * d + (kh + 1) * d];
                let v = &hh[2 * l.hk * d + hv * d..2 * l.hk * d + (hv + 1) * d];
                gdn_head_step(&mut st, &q, k, v, g, beta, &mut sk, &mut o);
                let ss: f32 = o.iter().map(|x| x * x).sum();
                let rs = 1.0 / (ss / d as f32 + self.eps).sqrt();
                for j in 0..d {
                    let z = pp[l.z() + hv * d + j];
                    let gate = match l.gate {
                        Gate::Sigmoid => sigmoid(z),
                        Gate::Silu => silu(z),
                    };
                    y[tt * vw + hv * d + j] = o[j] * rs * self.norm[j] * gate;
                }
            }
            if keep.is_some() {
                for i in 0..d {
                    s[(i * l.hv + hv) * d..(i * l.hv + hv + 1) * d]
                        .copy_from_slice(&st[i * d..(i + 1) * d]);
                }
            }
        }
        if keep.is_some() {
            self.dev.upload_into_f32(state, &s);
        }
        y
    }

    /// Integrity scan of the store heads (read-only, straight from the buffer; the layout from
    /// its size): every value an integer in `[0, 2^18)`, every marker 0 or 1, every unmarked
    /// row all zero. Returns the poisoned v heads (empty when the store is intact; every store
    /// head when the state is no layout's size).
    pub fn audit(dev: &Dev, state: &Dev::Buffer) -> Vec<usize> {
        let s = dev.download(state);
        let Ok(l) = Layout::of_state_len(s.len()) else {
            return STORE_HV.to_vec();
        };
        l.store_hv
            .iter()
            .copied()
            .filter(|&hv| {
                (0..D).any(|i| {
                    let r = &s[(i * l.hv + hv) * D..(i * l.hv + hv + 1) * D];
                    let bad_val = r
                        .iter()
                        .any(|&x| x.fract() != 0.0 || !(0.0..VALUE_LIMIT as f32).contains(&x));
                    let m = r[MARK];
                    bad_val || (m != 0.0 && m != 1.0) || (m == 0.0 && r.iter().any(|&x| x != 0.0))
                })
            })
            .collect()
    }

    /// The store heads' rows out of a state (`[i][store heads][j]`: 768 KiB for the 48-head
    /// layout, 512 KiB for the 32-head one; the layout from the state's size): what a
    /// cross-session snapshot keeps, like the post's `store_snapshot` keeps only the persistent
    /// heads.
    pub fn snapshot_heads(state: &[f32]) -> Vec<f32> {
        let l = Layout::of_state_len(state.len()).expect("not a GDN layer state");
        let mut out = Vec::with_capacity(l.head_floats());
        for i in 0..D {
            for &hv in l.store_hv {
                out.extend_from_slice(&state[(i * l.hv + hv) * D..(i * l.hv + hv + 1) * D]);
            }
        }
        out
    }

    /// Splice a [`snapshot_heads`](Self::snapshot_heads) into `state` (a fresh conversation's
    /// layer state of the same layout), leaving every other head as it is.
    pub fn restore_heads(state: &mut [f32], snap: &[f32]) {
        let l = Layout::of_state_len(state.len()).expect("not a GDN layer state");
        assert_eq!(snap.len(), l.head_floats(), "store heads of another layout");
        let n = l.store_hv.len();
        for i in 0..D {
            for (k, &hv) in l.store_hv.iter().enumerate() {
                let src = &snap[(i * n + k) * D..(i * n + k + 1) * D];
                state[(i * l.hv + hv) * D..(i * l.hv + hv + 1) * D].copy_from_slice(src);
            }
        }
    }
}

const _: () = assert!(GDN_STATE == GDN_D * GDN_HV * GDN_D);

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

    /// Run `case` under every layout (the 48-head one on the CPU device's `gdn_step`, the
    /// 32-head one on the host reference recurrence).
    fn each_layout(case: impl Fn(Layout)) {
        for l in Layout::ALL {
            eprintln!("layout {}", l.name);
            case(l);
        }
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
    fn rig(seed: u64, l: Layout) -> Rig {
        let mut r = Rng(seed);
        Rig {
            dev: CpuDevice::new(),
            conv: vec![0.0],
            dt: (0..l.hv).map(|_| r.f() * 4.0 - 2.0).collect(),
            ssm_a: (0..l.hv).map(|_| -0.5 - 7.5 * r.f()).collect(),
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
    fn noisy_state(seed: u64, l: Layout) -> Vec<f32> {
        let mut r = Rng(seed);
        let mut s = vec![0f32; l.state_len()];
        for i in 0..D {
            for hv in 0..l.hv {
                if l.store_hv.contains(&hv) {
                    continue;
                }
                for j in 0..D {
                    s[(i * l.hv + hv) * D + j] = r.f() * 2.0 - 1.0;
                }
            }
        }
        s
    }

    fn base_heads_eq(a: &[f32], b: &[f32], l: Layout) {
        for i in 0..D {
            for hv in (0..l.hv).filter(|h| !l.store_hv.contains(h)) {
                for j in 0..D {
                    let k = (i * l.hv + hv) * D + j;
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

    /// Files of mixed sizes (1..=127 tokens, one content row each) fill the store toward its
    /// file and row limit (default 128 files in debug; `TANG_MEMSTORE_FILES=512` (48-head) or
    /// `256` (32-head) for the full fill, run in release); every file reads back exactly, the
    /// one past the limit fails, and the base heads are bitwise untouched.
    #[test]
    fn memstore_fills_to_capacity() {
        each_layout(|l| {
            let t0 = Instant::now();
            let rig = rig(1, l);
            let b = rig.bufs();
            let before = noisy_state(2, l);
            let mut state = rig.dev.upload_f32(&before);
            let mut st = MemStore::open(&rig.dev, b.params(), EPS, &mut state).unwrap();
            assert_eq!(st.layout(), l);
            let n = env("TANG_MEMSTORE_FILES", 128).min(l.max_files());
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
            if n == l.max_files() {
                assert!(st.put(&mut state, b"one-too-many", &[1]).is_err());
            }
            assert!(MemStore::<CpuDevice>::audit(&rig.dev, &state).is_empty());
            base_heads_eq(&rig.dev.download(&state), &before, l);
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
        });
    }

    /// Big files (multi-row, up to the per-file limit), overwrite with a different size, delete,
    /// and the free-row bookkeeping.
    #[test]
    fn memstore_overwrite_delete() {
        each_layout(|l| {
            let rig = rig(4, l);
            let b = rig.bufs();
            let mut state = rig.dev.upload_f32(&noisy_state(5, l));
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
        });
    }

    /// Names that share a home slot (and fill its bucket, so the probe crosses into the next
    /// window and wraps the table's end) all store and read, and a delete in the middle of the
    /// chain keeps the names after it reachable; the tombstone is reused.
    #[test]
    fn memstore_collisions() {
        each_layout(|l| {
            let rig = rig(7, l);
            let b = rig.bufs();
            let mut state = rig.dev.upload_f32(&noisy_state(8, l));
            let mut st = MemStore::open(&rig.dev, b.params(), EPS, &mut state).unwrap();
            // names homing 3 slots before the end: the chain wraps to slot 0
            let mut names = Vec::new();
            let mut i = 0u64;
            while names.len() < 20 {
                let n = format!("c{i}").into_bytes();
                if home(&n, l.slots()) == l.slots() - 3 {
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
            // the chain wrapped into slot 0: bucket 0 row 0 of the first directory head (v head
            // 12 tiled, 24 grouped), status live
            let hv0 = Cell::slot(&l, 0).hv(&l);
            assert_eq!(hv0, [12, 24][(l == Layout::CODER) as usize]);
            assert_eq!(s[hv0 * D + COL_STATUS], LIVE as f32);
        });
    }

    /// A random workload of put / overwrite / get / delete / miss against a `HashMap`, with
    /// small files so the store churns through its slots (tombstones) and rows. Default 300
    /// operations (~20 s in debug, ~65 ms per window); `TANG_MEMSTORE_OPS=10000` runs the full 10k (release: see the design doc).
    #[test]
    fn memstore_random_workload() {
        each_layout(|l| {
            let t0 = Instant::now();
            let rig = rig(9, l);
            let b = rig.bufs();
            let before = noisy_state(10, l);
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
                        let fits = model.contains_key(&name) || model.len() < l.max_files();
                        let rows = st.capacity().rows_used;
                        match st.put(&mut state, &name, &data) {
                            Ok(()) => {
                                model.insert(name, data);
                            }
                            Err(e) => assert!(
                                !fits || rows + len.div_ceil(ROW_TOKENS) > l.rows(),
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
            base_heads_eq(&rig.dev.download(&state), &before, l);
            eprintln!(
                "memstore: {ops} random ops, {:?} ({:?})",
                st.traffic(),
                t0.elapsed()
            );
        });
    }

    /// Persistence: (1) the whole layer buffer, the bytes `save_running` carries, reloaded into
    /// a fresh device buffer, reopens to the same store; (2) only the store heads, spliced into
    /// another conversation's state (different base-head contents), read back exactly while
    /// leaving that conversation's base heads alone.
    #[test]
    fn memstore_persists() {
        each_layout(|l| {
            let rig = rig(12, l);
            let b = rig.bufs();
            let mut state = rig.dev.upload_f32(&noisy_state(13, l));
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
            assert_eq!(snap.len(), l.head_floats());
            let other = noisy_state(15, l);
            let mut spliced = other.clone();
            MemStore::<CpuDevice>::restore_heads(&mut spliced, &snap);
            let mut s2 = rig.dev.upload_f32(&spliced);
            let mut st2 = MemStore::open(&rig.dev, b.params(), EPS, &mut s2).unwrap();
            assert_eq!(cap(&st2), cap(&st));
            for (name, data) in &want {
                assert_eq!(st2.get(&mut s2, name).unwrap().as_ref(), Some(data));
            }
            base_heads_eq(&rig.dev.download(&s2), &other, l);
        });
    }

    /// One ordinary model token through the layer (real gates, random k on every head) poisons
    /// the store heads: `audit` names them, and a kernel read of a written row fails instead of
    /// returning wrong data (or, for a uniformly decayed row the ratio read survives, the audit
    /// is what catches it).
    #[test]
    fn memstore_detects_poison() {
        each_layout(|l| {
            let rig = rig(16, l);
            let b = rig.bufs();
            let mut state = rig.dev.upload_f32(&noisy_state(17, l));
            let mut st = MemStore::open(&rig.dev, b.params(), EPS, &mut state).unwrap();
            st.put(&mut state, b"x", &[1, 2, 3]).unwrap();
            assert!(MemStore::<CpuDevice>::audit(&rig.dev, &state).is_empty());
            // a model token: g < 1, β ≈ 0.6, dense k and v
            let mut r = Rng(18);
            let h: Vec<f32> = (0..l.conv()).map(|_| r.f() - 0.5).collect();
            let mut p = vec![0f32; l.proj()];
            for hv in 0..l.hv {
                p[l.a() + hv] = 0.3;
                p[l.b() + hv] = 0.4;
            }
            if l.on_device() {
                let win = rig.dev.upload_f32(&[0.0, f32::from_bits(1), 0.0]);
                let mut y = rig.dev.alloc_f32(l.v_width());
                rig.dev.gdn_step(
                    &mut state,
                    &rig.dev.upload_f32(&h),
                    &rig.dev.upload_f32(&p),
                    l.proj(),
                    &b.params(),
                    &mut y,
                    None,
                    1,
                    GdnMode::Commit { win: &win },
                    EPS,
                );
            } else {
                st.step(&mut state, &h, &p, 1, Some(1)).unwrap();
            }
            let bad = MemStore::<CpuDevice>::audit(&rig.dev, &state);
            assert_eq!(bad, l.store_hv.to_vec());
            assert!(
                st.get(&mut state, b"x").is_err()
                    || st.get(&mut state, b"x").unwrap() != Some(vec![1, 2, 3])
            );
        });
    }

    /// The kernel stored what the store claims: a written row is exactly the integers in the
    /// state buffer (value, marker), in the documented head and row, and nothing else of the
    /// store heads moved. Content rows 0 and 1 are store k heads 12 and 13, group `last`: v
    /// heads 44 and 45 tiled (`12 + 16·2`), 25 and 27 grouped (`2·12 + 1`, `2·13 + 1`).
    #[test]
    fn memstore_rows_are_exact_in_state() {
        each_layout(|l| {
            let rig = rig(19, l);
            let b = rig.bufs();
            let mut state = rig.dev.upload_f32(&vec![0f32; l.state_len()]);
            let mut st = MemStore::open(&rig.dev, b.params(), EPS, &mut state).unwrap();
            let data: Vec<u32> = (0..200).map(|i| VALUE_LIMIT - 1 - i).collect();
            st.put(&mut state, b"exact", &data).unwrap();
            let s = rig.dev.download(&state);
            let heads = if l == Layout::CODER {
                [25, 27]
            } else {
                [44, 45]
            };
            for (c, chunk) in data.chunks(ROW_TOKENS).enumerate() {
                let hv = heads[c];
                assert_eq!(Cell::content(&l, c).hv(&l), hv);
                assert_eq!(l.k_head(hv), STORE_HK[c]);
                for (j, &v) in chunk.iter().enumerate() {
                    assert_eq!(s[hv * D + j].to_bits(), (v as f32).to_bits());
                }
                assert_eq!(s[hv * D + MARK], 1.0);
            }
            let slot = home(b"exact", l.slots());
            let c = Cell::slot(&l, slot);
            let row = &s[(c.row * l.hv + c.hv(&l)) * D..(c.row * l.hv + c.hv(&l) + 1) * D];
            assert_eq!(row[COL_STATUS], 1.0);
            assert_eq!(row[COL_LEN], 200.0);
            assert_eq!(row[MARK], 1.0);
            let written = s.iter().filter(|&&x| x != 0.0).count();
            let want = data.iter().filter(|&&x| x != 0).count()
                + 2
                + row.iter().filter(|&&x| x != 0.0).count();
            assert_eq!(written, want);
        });
    }

    fn hparams(
        arch: super::super::reference::Arch,
        hk: usize,
        hv: usize,
    ) -> super::super::reference::Hparams {
        super::super::reference::Hparams {
            arch,
            n_embd: 0,
            n_layer: 0,
            n_vocab: 0,
            eps: 1e-6,
            hc: 1,
            hc_lr: 0,
            n_head: 0,
            n_head_kv: 0,
            head_dim: 0,
            n_rot: 0,
            rope_base: 0.0,
            rope_sections: [0; 4],
            idx_heads: 0,
            idx_dim: 0,
            idx_top_k: 0,
            kpool: 0,
            ssm_d_conv: 4,
            ssm_d_state: 128,
            ssm_k_heads: hk,
            ssm_v_heads: hv,
            n_expert: 0,
            n_expert_used: 0,
            n_ff_exp: 0,
            expert_weights_scale: 0.0,
            is_recurrent: Vec::new(),
            ple: None,
        }
    }

    /// Each layout's v-to-k head rule is the model's (`Hparams::gdn_k_head`: qwen4exp tiled,
    /// qwen3next grouped), every store v head reads a store k head, and the capacities are the
    /// documented ones.
    #[test]
    fn memstore_layouts_follow_the_k_head_rule() {
        use super::super::reference::Arch;
        for (l, arch) in [
            (Layout::FLASH, Arch::Qwen4Exp),
            (Layout::CODER, Arch::Qwen3Next),
        ] {
            let hp = hparams(arch, l.hk, l.hv);
            for vh in 0..l.hv {
                assert_eq!(l.k_head(vh), hp.gdn_k_head(vh), "{} v head {vh}", l.name);
            }
            let mut seen = Vec::new();
            for kh in 0..l.store_hk.len() {
                for g in 0..l.groups() {
                    let hv = l.store_v(kh, g);
                    assert_eq!(
                        hp.gdn_k_head(hv),
                        l.store_hk[kh],
                        "{} kh {kh} g {g}",
                        l.name
                    );
                    seen.push(hv);
                }
            }
            seen.sort();
            let mut want = l.store_hv.to_vec();
            want.sort();
            assert_eq!(
                seen, want,
                "{}: store v heads are every v head of the store k heads",
                l.name
            );
            assert_eq!(Layout::of_shape(l.hk, l.hv).unwrap(), l);
            assert_eq!(Layout::of_state_len(l.state_len()).unwrap(), l);
        }
        let c = Layout::CODER;
        assert_eq!(
            (
                c.slots(),
                c.bucket_slots(),
                c.max_files(),
                c.rows(),
                c.max_tokens()
            ),
            (512, 4, 256, 512, 65_024)
        );
        assert_eq!(4 * c.head_floats(), 512 << 10);
        let f = Layout::FLASH;
        assert_eq!(
            (f.slots(), f.max_files(), f.rows()),
            (SLOTS, MAX_FILES, ROWS)
        );
        assert_eq!(f.proj(), tang_compute::flash::shape::GDN_PROJ);
        assert!(Layout::of_shape(16, 64).is_err());
    }

    /// The host reference recurrence is the device kernel for store windows: the same ops on
    /// the 48-head layout through both leave bitwise-equal states and read the same, so the
    /// 32-head layout (host-only) runs the device's arithmetic.
    #[test]
    fn memstore_host_ref_matches_device() {
        let l = Layout::FLASH;
        let rig = rig(20, l);
        let b = rig.bufs();
        let init = noisy_state(21, l);
        let (mut sd, mut sh) = (rig.dev.upload_f32(&init), rig.dev.upload_f32(&init));
        let mut dev = MemStore::open_with(&rig.dev, l, false, b.params(), EPS, &mut sd).unwrap();
        let mut host = MemStore::open_with(&rig.dev, l, true, b.params(), EPS, &mut sh).unwrap();
        let mut r = Rng(22);
        for i in 0..12 {
            let name = format!("f{}", i % 5).into_bytes();
            let data = {
                let n = r.below(400);
                file(&mut r, n)
            };
            dev.put(&mut sd, &name, &data).unwrap();
            host.put(&mut sh, &name, &data).unwrap();
            if i % 4 == 3 {
                assert_eq!(
                    dev.delete(&mut sd, b"f1").unwrap(),
                    host.delete(&mut sh, b"f1").unwrap()
                );
            }
            assert_eq!(
                dev.get(&mut sd, &name).unwrap(),
                host.get(&mut sh, &name).unwrap()
            );
        }
        let (a, c) = (rig.dev.download(&sd), rig.dev.download(&sh));
        assert!(a.iter().zip(&c).all(|(x, y)| x.to_bits() == y.to_bits()));
        assert_eq!(dev.list(&mut sd).unwrap(), host.list(&mut sh).unwrap());
        // the device path refuses the 32-head layout
        let lc = Layout::CODER;
        let rc = self::rig(23, lc);
        let bc = rc.bufs();
        let mut s32 = rc.dev.upload_f32(&vec![0f32; lc.state_len()]);
        assert!(MemStore::open_with(&rc.dev, lc, false, bc.params(), EPS, &mut s32).is_err());
    }
}
