//! Paged KV storage: every cache of a model keeps its keys and values in fixed blocks of
//! [`BLOCK`] positions from one [`Pool`], and a block of a prompt's prefix that some cache
//! already computed is shared instead of computed again. See `docs/paged-kv.md`.
//!
//! A block is *sealed* once its positions are final: it's then immutable, known by a hash
//! chained over its tokens and every token before it, and found by that hash. Sealed blocks
//! nobody holds stay cached until the pool needs their room (least recently used first).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};
use tang_compute::ComputeDevice;

/// Positions a block holds: a multiple of 32 (tiled attention reads 32-row tiles), a power of
/// two (rows are found by shift and mask), and large enough to keep tables and bookkeeping
/// small.
pub const BLOCK: usize = 256;

/// Blocks a pool starts with.
const FIRST_BLOCKS: usize = 16;

/// The hash a sequence's first block chains from.
pub const SEED: u64 = 0xcbf2_9ce4_8422_2325;

/// The id of a block holding `tokens` after a prefix whose last block's id is `parent` (or
/// [`SEED`]): FNV-1a over both, stable across builds (it also names blocks on disk).
pub fn chain(parent: u64, tokens: &[u32]) -> u64 {
    let mut h = SEED;
    for b in parent
        .to_le_bytes()
        .into_iter()
        .chain(tokens.iter().flat_map(|t| t.to_le_bytes()))
    {
        h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// The ids of the full blocks of `tokens`, in order.
pub fn chain_all(tokens: &[u32]) -> Vec<u64> {
    let mut h = SEED;
    tokens
        .chunks_exact(BLOCK)
        .map(|t| {
            h = chain(h, t);
            h
        })
        .collect()
}

#[derive(Debug, Clone, Default)]
struct Meta {
    /// Caches whose tables hold it.
    refs: u32,
    /// Sealed: its id (see [`chain`]). Unsealed blocks belong to one cache and change.
    hash: Option<u64>,
    /// The id of the block before it (sealed blocks).
    parent: u64,
    /// Its tokens (sealed blocks), to check a lookup by hash.
    tokens: Vec<u32>,
    /// Last attached or sealed (larger is later).
    used: u64,
    /// Written to the disk tier already.
    pub on_disk: bool,
}

/// Device memory for KV blocks: each layer's K and V buffers, `capacity * BLOCK` rows each.
pub struct Pool<B> {
    k: Vec<B>,
    v: Vec<B>,
    layers: usize,
    kv_dim: usize,
    bf16: bool,
    meta: Vec<Meta>,
    /// Unsealed blocks nobody holds.
    free: Vec<u32>,
    by_hash: HashMap<u64, u32>,
    clock: u64,
    /// Most blocks to hold (the memory budget); more only when every block is in use.
    max_blocks: usize,
    /// Over budget at least once (reported once).
    overrun: bool,
}

/// What a pool holds, in blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stats {
    /// Allocated on the device.
    pub capacity: usize,
    /// Held by caches.
    pub used: usize,
    /// Sealed, held by no cache, kept for sharing until their room is needed.
    pub cached: usize,
    /// The budget.
    pub max: usize,
}

impl<B> Pool<B> {
    /// An empty pool for `layers` layers of `kv_dim`-wide K and V, holding up to `max_blocks`.
    pub fn new(layers: usize, kv_dim: usize, bf16: bool, max_blocks: usize) -> Self {
        Self {
            k: Vec::new(),
            v: Vec::new(),
            layers,
            kv_dim,
            bf16,
            meta: Vec::new(),
            free: Vec::new(),
            by_hash: HashMap::new(),
            clock: 0,
            max_blocks: max_blocks.max(1),
            overrun: false,
        }
    }

    /// Device memory one block takes (all layers' K and V).
    pub fn block_bytes(&self) -> usize {
        2 * self.layers * self.kv_dim * BLOCK * if self.bf16 { 2 } else { 4 }
    }

    pub fn bf16(&self) -> bool {
        self.bf16
    }

    pub fn set_max_blocks(&mut self, n: usize) {
        self.max_blocks = n.max(1);
    }

    pub fn stats(&self) -> Stats {
        let used = self.meta.iter().filter(|m| m.refs > 0).count();
        let cached = self
            .meta
            .iter()
            .filter(|m| m.refs == 0 && m.hash.is_some())
            .count();
        Stats {
            capacity: self.meta.len(),
            used,
            cached,
            max: self.max_blocks,
        }
    }

    /// Blocks an allocation could still get within the budget: free, not yet allocated, or
    /// cached (evictable).
    pub fn available(&self) -> usize {
        let s = self.stats();
        s.max.saturating_sub(s.capacity) + self.free.len() + s.cached
    }

    /// Layer `l`'s K and V buffers.
    pub fn layer(&mut self, l: usize) -> (&mut B, &mut B) {
        (&mut self.k[l], &mut self.v[l])
    }

    fn touch(&mut self, b: u32) {
        self.clock += 1;
        self.meta[b as usize].used = self.clock;
    }

    /// One more holder of `b`.
    pub fn retain(&mut self, b: u32) {
        self.meta[b as usize].refs += 1;
        self.touch(b);
    }

    /// One holder of `b` fewer: an unsealed block nobody holds is free again; a sealed one
    /// stays cached.
    pub fn release(&mut self, b: u32) {
        let m = &mut self.meta[b as usize];
        debug_assert!(m.refs > 0, "releasing an unheld block");
        m.refs -= 1;
        if m.refs == 0 && m.hash.is_none() {
            self.free.push(b);
        }
    }

    /// Whether a cache holding `b` may write into it: unsealed, and held by it alone.
    pub fn writable(&self, b: u32) -> bool {
        let m = &self.meta[b as usize];
        m.hash.is_none() && m.refs == 1
    }

    /// The sealed block with id `hash` holding `tokens`, if the pool has it.
    pub fn find(&self, hash: u64, tokens: &[u32]) -> Option<u32> {
        let &b = self.by_hash.get(&hash)?;
        (self.meta[b as usize].tokens == tokens).then_some(b)
    }

    /// Sealed blocks following the block with id `parent`, with their tokens.
    pub fn children(&self, parent: u64) -> Vec<(u32, &[u32])> {
        self.by_hash
            .values()
            .filter(|&&b| self.meta[b as usize].parent == parent)
            .map(|&b| (b, &self.meta[b as usize].tokens[..]))
            .collect()
    }

    /// Seal `b` (a full block a cache holds alone) as `hash`, the block after `parent`,
    /// holding `tokens`. When the pool already has that block, `b` is released and the
    /// existing one returned (retained) instead.
    pub fn seal(&mut self, b: u32, hash: u64, parent: u64, tokens: &[u32]) -> u32 {
        if let Some(other) = self.find(hash, tokens) {
            if other != b {
                self.retain(other);
                self.release(b);
            }
            return other;
        }
        let m = &mut self.meta[b as usize];
        debug_assert!(m.hash.is_none());
        m.hash = Some(hash);
        m.parent = parent;
        m.tokens = tokens.to_vec();
        self.by_hash.insert(hash, b);
        self.touch(b);
        b
    }

    /// Drop every cached block (sealed, held by no cache): nothing is shared afterwards with
    /// what was computed before.
    pub fn forget(&mut self) {
        for b in 0..self.meta.len() {
            let m = &mut self.meta[b];
            if m.refs == 0 && m.hash.is_some() {
                *m = Meta::default();
                self.free.push(b as u32);
            }
        }
        self.by_hash
            .retain(|_, b| self.meta[*b as usize].hash.is_some());
    }

    /// The ids of every sealed block held, in no particular order.
    pub fn hashes(&self) -> Vec<u64> {
        self.by_hash.keys().copied().collect()
    }

    /// The id of sealed block `b`.
    pub fn hash(&self, b: u32) -> Option<u64> {
        self.meta[b as usize].hash
    }

    pub fn on_disk(&self, b: u32) -> bool {
        self.meta[b as usize].on_disk
    }

    pub fn set_on_disk(&mut self, b: u32) {
        self.meta[b as usize].on_disk = true;
    }
}

impl<B> Pool<B> {
    /// A block for a cache to write (held once). Free blocks first, then new room within the
    /// budget, then the least recently used cached block; past the budget only if every
    /// block is held.
    pub fn alloc<D: ComputeDevice<Buffer = B>>(&mut self, dev: &D) -> u32 {
        if self.free.is_empty() {
            if self.meta.len() < self.max_blocks {
                let want = (self.meta.len() * 2).max(FIRST_BLOCKS).min(self.max_blocks);
                self.grow(dev, want);
            } else if let Some(b) = self.lru_cached() {
                let m = &mut self.meta[b as usize];
                if let Some(h) = m.hash.take() {
                    self.by_hash.remove(&h);
                }
                *m = Meta::default();
                self.free.push(b);
            } else {
                if !self.overrun {
                    eprintln!(
                        "tang-llm: KV caches need more than the budget ({} blocks); growing past it",
                        self.max_blocks
                    );
                    self.overrun = true;
                }
                let want = self.meta.len() + self.meta.len().div_ceil(4).max(1);
                self.grow(dev, want);
            }
        }
        let b = self.free.pop().expect("a free block");
        self.meta[b as usize].refs = 1;
        self.touch(b);
        b
    }

    fn lru_cached(&self) -> Option<u32> {
        (0..self.meta.len())
            .filter(|&i| self.meta[i].refs == 0 && self.meta[i].hash.is_some())
            .min_by_key(|&i| self.meta[i].used)
            .map(|i| i as u32)
    }

    /// Room for `blocks` blocks: each layer's buffers are replaced in turn (copy, then drop the
    /// old one), so at most one layer's worth of memory is held twice.
    fn grow<D: ComputeDevice<Buffer = B>>(&mut self, dev: &D, blocks: usize) {
        let old = self.meta.len();
        if blocks <= old {
            return;
        }
        let rows = blocks * BLOCK * self.kv_dim;
        let bf16 = self.bf16;
        let alloc = |n| {
            if bf16 {
                dev.alloc_bf16(n)
            } else {
                dev.alloc(n)
            }
        };
        if self.k.is_empty() {
            self.k = (0..self.layers).map(|_| alloc(rows)).collect();
            self.v = (0..self.layers).map(|_| alloc(rows)).collect();
        } else {
            for buf in self.k.iter_mut().chain(self.v.iter_mut()) {
                let mut grown = alloc(rows);
                dev.write_into(&mut grown, 0, buf);
                *buf = grown;
            }
        }
        self.meta.resize(blocks, Meta::default());
        // Lowest ids first out of the free list.
        self.free.extend((old as u32..blocks as u32).rev());
    }

    /// Copy positions `0..n` of block `from` into block `to` (every layer).
    pub fn copy_rows<D: ComputeDevice<Buffer = B>>(
        &mut self,
        dev: &D,
        from: u32,
        to: u32,
        n: usize,
    ) {
        if n == 0 {
            return;
        }
        let kvd = self.kv_dim;
        let (src, dst) = (from as usize * BLOCK * kvd, to as usize * BLOCK * kvd);
        for buf in self.k.iter_mut().chain(self.v.iter_mut()) {
            let part = dev.slice_buffer(buf, src, n * kvd);
            dev.write_into(buf, dst, &part);
        }
    }

    /// Positions `start..start + n` of block `b` as bf16 bits, position-major (every layer's K
    /// row then V row), for the disk tier.
    pub fn read_block<D: ComputeDevice<Buffer = B>>(
        &self,
        dev: &D,
        b: u32,
        start: usize,
        n: usize,
    ) -> Vec<u16> {
        let kvd = self.kv_dim;
        let mut out = vec![0u16; n * 2 * self.layers * kvd];
        let at = (b as usize * BLOCK + start) * kvd;
        let bufs = self.k.iter().zip(&self.v).flat_map(|(k, v)| [k, v]);
        for (j, buf) in bufs.enumerate() {
            let part = dev.download_bf16(&dev.slice_buffer(buf, at, n * kvd));
            for p in 0..n {
                let o = (p * self.layers * 2 + j) * kvd;
                out[o..o + kvd].copy_from_slice(&part[p * kvd..(p + 1) * kvd]);
            }
        }
        out
    }

    /// Fill positions from `start` of block `b` with rows as [`read_block`](Self::read_block)
    /// gives them.
    pub fn write_block<D: ComputeDevice<Buffer = B>>(
        &mut self,
        dev: &D,
        b: u32,
        start: usize,
        rows: &[u16],
    ) {
        let kvd = self.kv_dim;
        let layers = self.layers;
        let n = rows.len() / (2 * layers * kvd);
        let at = (b as usize * BLOCK + start) * kvd;
        let bufs = self
            .k
            .iter_mut()
            .zip(self.v.iter_mut())
            .flat_map(|(k, v)| [k, v]);
        for (j, buf) in bufs.enumerate() {
            let mut part = Vec::with_capacity(n * kvd);
            for p in 0..n {
                let o = (p * layers * 2 + j) * kvd;
                part.extend_from_slice(&rows[o..o + kvd]);
            }
            dev.write_into(buf, at, &dev.upload_bf16(&part));
        }
    }
}

/// A pool shared by a model and its caches.
pub type Shared<B> = Arc<Mutex<Pool<B>>>;

pub fn lock<B>(p: &Shared<B>) -> MutexGuard<'_, Pool<B>> {
    p.lock().unwrap_or_else(|e| e.into_inner())
}

/// One sequence's attention state: the blocks holding its positions, in order. Truncating
/// keeps a prefix; positions are rewritten by the next forward (into a private copy of a block
/// that's sealed or shared).
pub struct Cache<B> {
    pool: Shared<B>,
    /// Blocks of positions `0..len`, `len.div_ceil(BLOCK)` of them (more during a forward).
    pub(crate) table: Vec<u32>,
    /// Tokens already in the cache.
    pub len: usize,
    pub tokens: Vec<u32>,
}

impl<B> Cache<B> {
    pub fn new(pool: Shared<B>) -> Self {
        Self {
            pool,
            table: Vec::new(),
            len: 0,
            tokens: Vec::new(),
        }
    }

    pub fn pool(&self) -> &Shared<B> {
        &self.pool
    }

    /// The blocks it holds.
    pub fn blocks(&self) -> &[u32] {
        &self.table
    }

    /// Device memory of the blocks it holds (shared ones included).
    pub fn bytes(&self) -> usize {
        self.table.len() * lock(&self.pool).block_bytes()
    }

    pub fn truncate(&mut self, len: usize) {
        self.len = self.len.min(len);
        self.tokens.truncate(self.len);
        let keep = self.len.div_ceil(BLOCK);
        if self.table.len() > keep {
            let mut pool = lock(&self.pool);
            for b in self.table.drain(keep..) {
                pool.release(b);
            }
        }
    }

    /// Replace what it holds with `blocks` (retained here) holding `tokens`'s first
    /// `blocks.len() * BLOCK` positions.
    pub fn attach(&mut self, blocks: &[u32], tokens: &[u32]) {
        self.truncate(0);
        let mut pool = lock(&self.pool);
        for &b in blocks {
            pool.retain(b);
        }
        drop(pool);
        self.table = blocks.to_vec();
        self.len = blocks.len() * BLOCK;
        self.tokens = tokens[..self.len].to_vec();
    }

    /// Append block `b` (held for it already) holding `tokens` at its first positions: the
    /// cache must end on a block boundary.
    pub fn push_block(&mut self, b: u32, tokens: &[u32]) {
        assert_eq!(self.len % BLOCK, 0, "push_block mid-block");
        assert_eq!(self.table.len(), self.len / BLOCK);
        self.table.push(b);
        self.len += tokens.len();
        self.tokens.extend_from_slice(tokens);
    }

    /// Seal its full blocks among the first `upto` positions, deduplicating against blocks
    /// the pool already has. Returns the ids of its sealed blocks, in order.
    pub fn seal(&mut self, upto: usize) -> Vec<u64> {
        let full = upto.min(self.len) / BLOCK;
        let mut pool = lock(&self.pool);
        let mut parent = SEED;
        let mut ids = Vec::with_capacity(full);
        for i in 0..full {
            let toks = &self.tokens[i * BLOCK..(i + 1) * BLOCK];
            let h = chain(parent, toks);
            let b = self.table[i];
            match pool.hash(b) {
                Some(have) => debug_assert_eq!(have, h, "a sealed block changed"),
                None => self.table[i] = pool.seal(b, h, parent, toks),
            }
            ids.push(h);
            parent = h;
        }
        ids
    }
}

impl<B> Drop for Cache<B> {
    fn drop(&mut self) {
        let mut pool = lock(&self.pool);
        for &b in &self.table {
            pool.release(b);
        }
    }
}

/// Make `cache`'s blocks ready for positions `pos..end` to be written: allocate blocks past
/// its table, and give it a private copy of the block holding `pos` if that one is sealed or
/// shared (rows before `pos` copied). Returns the table.
pub fn prepare<D: ComputeDevice>(
    dev: &D,
    pool: &mut Pool<D::Buffer>,
    cache: &mut Cache<D::Buffer>,
    pos: usize,
    end: usize,
) -> Vec<u32> {
    let first = pos / BLOCK;
    // Blocks at or past `pos` that it can't write: a fresh one each (copying the rows before
    // `pos` into the first).
    for i in first..cache.table.len() {
        let b = cache.table[i];
        if pool.writable(b) {
            continue;
        }
        let fresh = pool.alloc(dev);
        if i == first {
            pool.copy_rows(dev, b, fresh, pos % BLOCK);
        }
        pool.release(b);
        cache.table[i] = fresh;
    }
    while cache.table.len() < end.div_ceil(BLOCK) {
        let b = pool.alloc(dev);
        cache.table.push(b);
    }
    cache.table.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tang_compute::CpuDevice;

    fn pool(max: usize) -> (CpuDevice, Shared<<CpuDevice as ComputeDevice>::Buffer>) {
        (
            CpuDevice::new(),
            Arc::new(Mutex::new(Pool::new(2, 4, false, max))),
        )
    }

    /// A cache holding `tokens`, its blocks marked with the token values.
    fn filled(
        dev: &CpuDevice,
        p: &Shared<<CpuDevice as ComputeDevice>::Buffer>,
        tokens: &[u32],
    ) -> Cache<<CpuDevice as ComputeDevice>::Buffer> {
        let mut c = Cache::new(p.clone());
        let mut pool = lock(p);
        prepare(dev, &mut pool, &mut c, 0, tokens.len());
        drop(pool);
        c.len = tokens.len();
        c.tokens = tokens.to_vec();
        c
    }

    fn toks(n: usize, salt: u32) -> Vec<u32> {
        (0..n as u32).map(|i| i * 3 + salt).collect()
    }

    #[test]
    fn sealing_shares_equal_blocks_and_frees_duplicates() {
        let (dev, p) = pool(64);
        let a_toks = toks(2 * BLOCK + 10, 0);
        let mut a = filled(&dev, &p, &a_toks);
        let ids = a.seal(a.len);
        assert_eq!(ids, chain_all(&a_toks));
        // The same prefix computed again by another cache collapses onto a's blocks.
        let mut b = filled(&dev, &p, &a_toks[..2 * BLOCK]);
        let dup = b.blocks().to_vec();
        b.seal(b.len);
        assert_eq!(b.blocks(), &a.blocks()[..2]);
        assert_ne!(dup, b.blocks());
        let s = lock(&p).stats();
        assert_eq!(s.used, 3, "a's two shared blocks and its tail");
        drop(a);
        drop(b);
        let s = lock(&p).stats();
        assert_eq!((s.used, s.cached), (0, 2));
    }

    #[test]
    fn writing_into_a_shared_block_copies_it_first() {
        let (dev, p) = pool(64);
        let t = toks(BLOCK * 2, 1);
        let mut a = filled(&dev, &p, &t);
        a.seal(a.len);
        let mut b = Cache::new(p.clone());
        b.attach(a.blocks(), &t);
        // Rewind into the second block and write: b gets its own copy, a keeps its block.
        b.truncate(BLOCK + 5);
        let mut pool = lock(&p);
        prepare(&dev, &mut pool, &mut b, BLOCK + 5, BLOCK + 9);
        drop(pool);
        assert_eq!(b.blocks()[0], a.blocks()[0]);
        assert_ne!(b.blocks()[1], a.blocks()[1]);
        assert!(lock(&p).writable(b.blocks()[1]));
    }

    #[test]
    fn eviction_takes_the_least_recently_used_cached_block() {
        let (dev, p) = pool(3);
        let (t1, t2) = (toks(BLOCK, 5), toks(BLOCK, 7));
        let mut a = filled(&dev, &p, &t1);
        a.seal(a.len);
        let mut b = filled(&dev, &p, &t2);
        b.seal(b.len);
        let (ha, hb) = (chain_all(&t1)[0], chain_all(&t2)[0]);
        drop(a);
        drop(b);
        // Full and both cached: a new block evicts a's (older), b's stays findable.
        let c = filled(&dev, &p, &toks(2 * BLOCK, 9));
        assert!(lock(&p).find(ha, &t1).is_none());
        assert!(lock(&p).find(hb, &t2).is_some());
        assert_eq!(lock(&p).stats().capacity, 3);
        drop(c);
    }

    #[test]
    fn past_the_budget_only_when_everything_is_held() {
        let (dev, p) = pool(2);
        let a = filled(&dev, &p, &toks(3 * BLOCK, 0));
        let s = lock(&p).stats();
        assert!(s.capacity >= 3 && s.used == 3);
        drop(a);
    }
}
