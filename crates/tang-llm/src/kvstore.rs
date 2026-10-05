//! KV blocks on disk, so a conversation that lost its blocks (or the server restarting) costs a
//! read instead of a prefill.
//!
//! A sealed block (see [`crate::blocks`]) is a file named by its hash, `blocks/<hash>.kvb`: its
//! tokens and its positions' keys and values (bf16, position-major: every layer's K row then V
//! row). It's written once, whichever conversation computed it, and any conversation whose
//! prompt has that block reads it back. A conversation's last, partial block goes to
//! `<key-hash>.tail` with the conversation's tokens, rewritten each turn. Files are written
//! whole and renamed into place, so a reader never sees half of one. Exact for bf16 caches:
//! rows are stored as the cache holds them. The least recently used files go when the
//! directory passes its budget (files from the older per-conversation format, `.kv` and
//! `.json`, are only ever trimmed).
//!
//! [`Writer`] does the writing on its own thread, so a turn never waits on the disk.

use anyhow::{bail, Context, Result};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const BLOCK_MAGIC: &[u8; 4] = b"TKVB";
const TAIL_MAGIC: &[u8; 4] = b"TKVT";
/// Format version of both kinds of file (bf16 rows).
const VERSION: u32 = 1;

pub struct Store {
    dir: PathBuf,
    /// Most bytes of files to keep.
    budget: u64,
    /// bf16 elements per position.
    row: usize,
}

/// FNV-1a: stable across builds, unlike std's hasher.
fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |h, &b| {
        (h ^ b as u64).wrapping_mul(0x100000001b3)
    })
}

fn put_u32(out: &mut Vec<u8>, x: u32) {
    out.extend_from_slice(&x.to_le_bytes());
}

/// Little-endian reads from a byte slice, failing on a short one.
struct Bytes<'a>(&'a [u8]);

impl Bytes<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        anyhow::ensure!(self.0.len() >= n, "truncated file");
        let (a, b) = self.0.split_at(n);
        self.0 = b;
        Ok(a)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u32s(&mut self, n: usize) -> Result<Vec<u32>> {
        Ok(self
            .take(n * 4)?
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }

    fn u16s(&mut self, n: usize) -> Result<Vec<u16>> {
        Ok(self
            .take(n * 2)?
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect())
    }
}

/// Write `bytes` to `path` whole: a temporary file renamed into place.
fn write_whole(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    let mut f = fs::File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
    f.write_all(bytes)?;
    drop(f);
    fs::rename(&tmp, path)?;
    Ok(())
}

/// Mark `path` as just used (the budget drops the least recently used first).
fn touch(path: &Path) {
    if let Ok(f) = fs::File::options().append(true).open(path) {
        let _ = f.set_modified(std::time::SystemTime::now());
    }
}

impl Store {
    /// A store in `dir` (one per model and weight format) holding positions of `row` bf16
    /// elements.
    pub fn new(dir: PathBuf, budget: u64, row: usize) -> Result<Self> {
        fs::create_dir_all(dir.join("blocks"))
            .with_context(|| format!("creating {}", dir.display()))?;
        Ok(Self { dir, budget, row })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The ids of the blocks on disk (from their file names).
    pub fn block_hashes(dir: &Path) -> Vec<u64> {
        let Ok(rd) = fs::read_dir(dir.join("blocks")) else {
            return Vec::new();
        };
        rd.filter_map(|e| {
            let name = e.ok()?.file_name();
            let hex = name.to_str()?.strip_suffix(".kvb")?;
            u64::from_str_radix(hex, 16).ok()
        })
        .collect()
    }

    fn block_path(&self, hash: u64) -> PathBuf {
        self.dir.join("blocks").join(format!("{hash:016x}.kvb"))
    }

    fn tail_path(&self, key: &str) -> PathBuf {
        self.dir.join(format!("{:016x}.tail", fnv(key.as_bytes())))
    }

    /// Whether block `hash` is on disk.
    pub fn has_block(&self, hash: u64) -> bool {
        self.block_path(hash).exists()
    }

    /// Block `hash`'s rows, if it's on disk and holds `tokens`.
    pub fn read_block(&self, hash: u64, tokens: &[u32]) -> Result<Option<Vec<u16>>> {
        let path = self.block_path(hash);
        let Ok(bytes) = fs::read(&path) else {
            return Ok(None);
        };
        let mut b = Bytes(&bytes);
        if b.take(4)? != BLOCK_MAGIC || b.u32()? != VERSION || b.u32()? as usize != self.row {
            return Ok(None);
        }
        let n = b.u32()? as usize;
        if b.u32s(n)? != tokens {
            return Ok(None);
        }
        let rows = b.u16s(n * self.row)?;
        touch(&path);
        Ok(Some(rows))
    }

    /// Put block `hash` (holding `tokens`, with their `rows`) on disk, unless it's there.
    pub fn write_block(&self, hash: u64, tokens: &[u32], rows: &[u16]) -> Result<()> {
        debug_assert_eq!(rows.len(), tokens.len() * self.row);
        let path = self.block_path(hash);
        if path.exists() {
            touch(&path);
            return Ok(());
        }
        let mut out = Vec::with_capacity(16 + tokens.len() * 4 + rows.len() * 2);
        out.extend_from_slice(BLOCK_MAGIC);
        put_u32(&mut out, VERSION);
        put_u32(&mut out, self.row as u32);
        put_u32(&mut out, tokens.len() as u32);
        tokens.iter().for_each(|&t| put_u32(&mut out, t));
        rows.iter()
            .for_each(|r| out.extend_from_slice(&r.to_le_bytes()));
        write_whole(&path, &out)?;
        self.trim(&path);
        Ok(())
    }

    /// `key`'s tail: the conversation's tokens, and where the tail's rows start (a block
    /// boundary).
    pub fn tail(&self, key: &str) -> Option<(Vec<u32>, usize)> {
        let mut f = fs::File::open(self.tail_path(key)).ok()?;
        let mut head = vec![0u8; 16 + key.len() + 8];
        f.read_exact(&mut head).ok()?;
        let mut b = Bytes(&head);
        let ok = b.take(4).ok()? == TAIL_MAGIC
            && b.u32().ok()? == VERSION
            && b.u32().ok()? as usize == self.row
            && b.u32().ok()? as usize == key.len()
            && b.take(key.len()).ok()? == key.as_bytes();
        if !ok {
            return None;
        }
        let (n, start) = (b.u32().ok()? as usize, b.u32().ok()? as usize);
        let mut toks = vec![0u8; n * 4];
        f.read_exact(&mut toks).ok()?;
        Some((Bytes(&toks).u32s(n).ok()?, start))
    }

    /// The rows of `key`'s tail, positions `start..start + n`.
    pub fn read_tail(&self, key: &str, n: usize) -> Result<Vec<u16>> {
        let path = self.tail_path(key);
        let bytes = fs::read(&path)?;
        let mut b = Bytes(&bytes);
        b.take(4 + 4 + 4)?;
        let klen = b.u32()? as usize;
        b.take(klen)?;
        let (count, start) = (b.u32()? as usize, b.u32()? as usize);
        b.take(count * 4)?;
        if start + n > count {
            bail!("the tail has {} positions, not {n}", count - start);
        }
        let rows = b.u16s(n * self.row)?;
        touch(&path);
        Ok(rows)
    }

    /// Record `key`'s conversation as `tokens`, whose positions from `start` (a block
    /// boundary) on are `rows`.
    pub fn write_tail(&self, key: &str, tokens: &[u32], start: usize, rows: &[u16]) -> Result<()> {
        debug_assert_eq!(rows.len(), (tokens.len() - start) * self.row);
        let mut out = Vec::with_capacity(24 + key.len() + tokens.len() * 4 + rows.len() * 2);
        out.extend_from_slice(TAIL_MAGIC);
        put_u32(&mut out, VERSION);
        put_u32(&mut out, self.row as u32);
        put_u32(&mut out, key.len() as u32);
        out.extend_from_slice(key.as_bytes());
        put_u32(&mut out, tokens.len() as u32);
        put_u32(&mut out, start as u32);
        tokens.iter().for_each(|&t| put_u32(&mut out, t));
        rows.iter()
            .for_each(|r| out.extend_from_slice(&r.to_le_bytes()));
        let path = self.tail_path(key);
        write_whole(&path, &out)?;
        self.trim(&path);
        Ok(())
    }

    /// Drop the least recently used files past the budget, never `keep`.
    fn trim(&self, keep: &Path) {
        let list = |d: &Path| -> Vec<PathBuf> {
            fs::read_dir(d)
                .map(|r| r.filter_map(|e| e.ok().map(|e| e.path())).collect())
                .unwrap_or_default()
        };
        let mut files: Vec<(std::time::SystemTime, u64, PathBuf)> = list(&self.dir)
            .into_iter()
            .chain(list(&self.dir.join("blocks")))
            .filter(|p| {
                p.extension()
                    .is_some_and(|x| x == "kvb" || x == "tail" || x == "kv")
            })
            .filter_map(|p| {
                let m = fs::metadata(&p).ok()?;
                Some((m.modified().ok()?, m.len(), p))
            })
            .collect();
        let mut total: u64 = files.iter().map(|f| f.1).sum();
        files.sort();
        for (_, len, p) in files {
            if total <= self.budget {
                break;
            }
            if p == keep {
                continue;
            }
            if p.extension().is_some_and(|x| x == "kv") {
                let _ = fs::remove_file(p.with_extension("json"));
            }
            let _ = fs::remove_file(&p);
            total -= len;
        }
    }
}

/// A save waiting for the writer thread.
enum Job {
    Block {
        hash: u64,
        tokens: Vec<u32>,
        rows: Vec<u16>,
    },
    Tail {
        key: String,
        tokens: Vec<u32>,
        start: usize,
        rows: Vec<u16>,
    },
    /// Answer once every earlier write is done.
    Flush(std::sync::mpsc::SyncSender<()>),
}

/// A [`Store`] whose writes happen on a thread of their own, in order.
pub struct Writer {
    store: std::sync::Arc<Store>,
    jobs: std::sync::mpsc::Sender<Job>,
}

impl Writer {
    pub fn new(store: Store) -> Self {
        let store = std::sync::Arc::new(store);
        let (jobs, rx) = std::sync::mpsc::channel::<Job>();
        let s = store.clone();
        std::thread::spawn(move || {
            for job in rx {
                let done = match job {
                    Job::Block { hash, tokens, rows } => s.write_block(hash, &tokens, &rows),
                    Job::Tail {
                        key,
                        tokens,
                        start,
                        rows,
                    } => s.write_tail(&key, &tokens, start, &rows),
                    Job::Flush(done) => {
                        let _ = done.send(());
                        Ok(())
                    }
                };
                if let Err(e) = done {
                    eprintln!("tang-llm: saving the KV cache: {e:#}");
                }
            }
        });
        Self { store, jobs }
    }

    /// The store, for reads. Files appear whole once written; one still queued isn't there
    /// yet ([`flush`](Self::flush) first where that matters).
    pub fn store(&self) -> &Store {
        &self.store
    }

    /// [`Store::write_block`], in the background.
    pub fn write_block(&self, hash: u64, tokens: Vec<u32>, rows: Vec<u16>) {
        let _ = self.jobs.send(Job::Block { hash, tokens, rows });
    }

    /// [`Store::write_tail`], in the background.
    pub fn write_tail(&self, key: &str, tokens: Vec<u32>, start: usize, rows: Vec<u16>) {
        let _ = self.jobs.send(Job::Tail {
            key: key.to_string(),
            tokens,
            start,
            rows,
        });
    }

    /// Wait for writes already sent.
    pub fn flush(&self) {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        if self.jobs.send(Job::Flush(tx)).is_ok() {
            let _ = rx.recv();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Store;

    fn store(budget: u64, name: &str) -> (Store, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("tang-kvstore-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        (Store::new(dir.clone(), budget, 2).unwrap(), dir)
    }

    #[test]
    fn blocks_are_written_once_and_read_by_hash() {
        let (s, dir) = store(1 << 20, "blocks");
        assert!(!s.has_block(7));
        s.write_block(7, &[1, 2], &[1, 2, 3, 4]).unwrap();
        assert!(s.has_block(7));
        assert_eq!(s.read_block(7, &[1, 2]).unwrap(), Some(vec![1, 2, 3, 4]));
        // Other tokens under the same hash: not this block.
        assert_eq!(s.read_block(7, &[1, 3]).unwrap(), None);
        // A second write of the same block leaves the first.
        s.write_block(7, &[1, 2], &[9, 9, 9, 9]).unwrap();
        assert_eq!(s.read_block(7, &[1, 2]).unwrap(), Some(vec![1, 2, 3, 4]));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn tails_are_per_conversation_and_rewritten() {
        let (s, dir) = store(1 << 20, "tails");
        assert!(s.tail("a").is_none());
        s.write_tail("a", &[1, 2, 3], 1, &[5, 6, 7, 8]).unwrap();
        assert_eq!(s.tail("a"), Some((vec![1, 2, 3], 1)));
        assert_eq!(s.read_tail("a", 2).unwrap(), vec![5, 6, 7, 8]);
        assert_eq!(s.read_tail("a", 1).unwrap(), vec![5, 6]);
        s.write_tail("a", &[1, 9], 1, &[1, 1]).unwrap();
        assert_eq!(s.tail("a"), Some((vec![1, 9], 1)));
        assert!(s.tail("b").is_none());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn the_writer_writes_in_order_and_flush_waits() {
        let (s, dir) = store(1 << 21, "writer");
        let w = super::Writer::new(s);
        w.write_tail("a", vec![1, 2], 0, vec![1, 2, 3, 4]);
        w.write_tail("a", vec![1, 3], 1, vec![5, 6]);
        w.write_block(3, vec![1], vec![7, 8]);
        w.flush();
        assert_eq!(w.store().tail("a"), Some((vec![1, 3], 1)));
        assert_eq!(w.store().read_tail("a", 1).unwrap(), vec![5, 6]);
        assert!(w.store().has_block(3));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn trims_the_least_recently_used() {
        // A block file is 16 + 2 * 4 + 4 * 2 = 32 bytes; room for one.
        let (s, dir) = store(40, "trim");
        s.write_block(1, &[1, 2], &[0; 4]).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        s.write_block(2, &[1, 2], &[0; 4]).unwrap();
        assert!(!s.has_block(1));
        assert!(s.has_block(2));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn old_per_conversation_files_are_only_trimmed() {
        let (s, dir) = store(40, "old");
        std::fs::write(dir.join("0123456789abcdef.kv"), [0u8; 64]).unwrap();
        std::fs::write(dir.join("0123456789abcdef.json"), b"{}").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        s.write_block(1, &[1, 2], &[0; 4]).unwrap();
        assert!(!dir.join("0123456789abcdef.kv").exists());
        assert!(!dir.join("0123456789abcdef.json").exists());
        assert!(s.has_block(1));
        let _ = std::fs::remove_dir_all(dir);
    }
}
