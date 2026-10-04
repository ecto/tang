//! KV caches on disk, so a conversation that lost its slot (or the server restarting) costs a
//! read instead of a prefill.
//!
//! Each conversation (`prompt_cache_key`) is two files: `<hash>.kv`, its positions' keys and
//! values (bf16, position-major: every layer's K row then V row), and `<hash>.json`, the tokens
//! those positions hold (and the row format: files from before bf16, which held f32, are
//! ignored and rewritten). Saving after a turn appends only the new positions; a conversation
//! that was rewound is cut back to what it still shares first. Exact for bf16 caches: rows are
//! stored as the cache holds them. The oldest files go when the directory passes its budget.
//!
//! [`Writer`] does the writing on its own thread, so a turn never waits on the disk.

use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::fs::{self, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub struct Store {
    dir: PathBuf,
    /// Most bytes of `.kv` files to keep.
    budget: u64,
    /// Elements per position.
    row: usize,
}

/// What `.json` files record about their `.kv` file's rows; anything else is ignored.
const FORMAT: &str = "bf16";

impl Store {
    /// A store in `dir` (one per model and weight format) holding positions of `row` bf16
    /// elements.
    pub fn new(dir: PathBuf, budget: u64, row: usize) -> Result<Self> {
        fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        Ok(Self { dir, budget, row })
    }

    fn paths(&self, key: &str) -> (PathBuf, PathBuf) {
        // FNV-1a: stable across builds, unlike std's hasher.
        let h = key.bytes().fold(0xcbf29ce484222325u64, |h, b| {
            (h ^ b as u64).wrapping_mul(0x100000001b3)
        });
        let base = self.dir.join(format!("{h:016x}"));
        (base.with_extension("json"), base.with_extension("kv"))
    }

    fn row_bytes(&self) -> u64 {
        self.row as u64 * 2
    }

    /// The tokens whose positions are on disk for `key` (empty if none).
    pub fn tokens(&self, key: &str) -> Vec<u32> {
        let (meta, data) = self.paths(key);
        let Some(v) = fs::read(&meta)
            .ok()
            .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
        else {
            return Vec::new();
        };
        if v["key"] != key || v["row"].as_u64() != Some(self.row as u64) || v["format"] != FORMAT {
            return Vec::new();
        }
        let rows = fs::metadata(&data).map_or(0, |m| m.len() / self.row_bytes()) as usize;
        let mut t: Vec<u32> = v["tokens"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|t| t.as_u64().map(|t| t as u32))
            .collect();
        t.truncate(rows);
        t
    }

    /// Positions `from..to` for `key`.
    pub fn read(&self, key: &str, from: usize, to: usize) -> Result<Vec<u16>> {
        let (_, data) = self.paths(key);
        let mut f = fs::File::open(&data)?;
        f.seek(SeekFrom::Start(from as u64 * self.row_bytes()))?;
        let mut bytes = vec![0u8; (to - from) * self.row * 2];
        f.read_exact(&mut bytes)?;
        // Touch it: the budget drops the least recently used first.
        let _ = f.set_modified(std::time::SystemTime::now());
        Ok(bytes
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect())
    }

    /// Record that `key`'s cache holds `tokens`, whose positions from `from` on are `rows`
    /// (positions before `from` are already on disk).
    pub fn write(&self, key: &str, tokens: &[u32], from: usize, rows: &[u16]) -> Result<()> {
        debug_assert_eq!(rows.len(), (tokens.len() - from) * self.row);
        let (meta, data) = self.paths(key);
        let mut f = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&data)?;
        // Data first, then the tokens that vouch for it: a crash between leaves the old
        // tokens, which never claim more rows than the file has.
        f.set_len(from as u64 * self.row_bytes())?;
        f.seek(SeekFrom::End(0))?;
        let mut bytes = Vec::with_capacity(rows.len() * 2);
        for x in rows {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        f.write_all(&bytes)?;
        let tmp = meta.with_extension("json.tmp");
        fs::write(
            &tmp,
            serde_json::to_vec(
                &json!({ "key": key, "row": self.row, "format": FORMAT, "tokens": tokens }),
            )?,
        )?;
        fs::rename(&tmp, &meta)?;
        self.trim(&data);
        Ok(())
    }

    /// Drop the least recently used conversations past the budget, never `keep`.
    fn trim(&self, keep: &Path) {
        let Ok(dir) = fs::read_dir(&self.dir) else {
            return;
        };
        let mut files: Vec<(std::time::SystemTime, u64, PathBuf)> = dir
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "kv"))
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
            let _ = fs::remove_file(p.with_extension("json"));
            let _ = fs::remove_file(&p);
            total -= len;
        }
    }
}

/// A save waiting for the writer thread.
enum Job {
    Write {
        key: String,
        tokens: Vec<u32>,
        from: usize,
        rows: Vec<u16>,
    },
    /// Answer once every earlier write is done.
    Flush(std::sync::mpsc::SyncSender<()>),
}

/// A [`Store`] whose writes happen on a thread of their own, in order.
pub struct Writer {
    store: std::sync::Arc<Store>,
    jobs: std::sync::mpsc::Sender<Job>,
    /// Writes sent and not yet done.
    pending: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl Writer {
    pub fn new(store: Store) -> Self {
        use std::sync::atomic::Ordering;
        let store = std::sync::Arc::new(store);
        let pending = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (jobs, rx) = std::sync::mpsc::channel::<Job>();
        let (s, p) = (store.clone(), pending.clone());
        std::thread::spawn(move || {
            for job in rx {
                match job {
                    Job::Write {
                        key,
                        tokens,
                        from,
                        rows,
                    } => {
                        if let Err(e) = s.write(&key, &tokens, from, &rows) {
                            eprintln!("tang-llm: saving the KV cache: {e:#}");
                        }
                        p.fetch_sub(1, Ordering::Release);
                    }
                    Job::Flush(done) => {
                        let _ = done.send(());
                    }
                }
            }
        });
        Self {
            store,
            jobs,
            pending,
        }
    }

    /// The store, for reads. A read right after [`write`](Self::write) may not see it yet:
    /// [`flush`](Self::flush) first where that matters.
    pub fn store(&self) -> &Store {
        &self.store
    }

    /// [`Store::write`], in the background.
    pub fn write(&self, key: &str, tokens: Vec<u32>, from: usize, rows: Vec<u16>) {
        self.pending
            .fetch_add(1, std::sync::atomic::Ordering::Acquire);
        let job = Job::Write {
            key: key.to_string(),
            tokens,
            from,
            rows,
        };
        if self.jobs.send(job).is_err() {
            self.pending
                .fetch_sub(1, std::sync::atomic::Ordering::Release);
        }
    }

    /// Wait for writes already sent (returns at once when there are none).
    pub fn flush(&self) {
        if self.pending.load(std::sync::atomic::Ordering::Acquire) == 0 {
            return;
        }
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        if self.jobs.send(Job::Flush(tx)).is_ok() {
            let _ = rx.recv();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Store;

    fn store(budget: u64) -> (Store, std::path::PathBuf) {
        let dir =
            std::env::temp_dir().join(format!("tang-kvstore-{}-{}", std::process::id(), budget));
        let _ = std::fs::remove_dir_all(&dir);
        (Store::new(dir.clone(), budget, 2).unwrap(), dir)
    }

    #[test]
    fn appends_reads_and_rewinds() {
        let (s, dir) = store(1 << 20);
        assert!(s.tokens("a").is_empty());
        s.write("a", &[1, 2], 0, &[1, 2, 3, 4]).unwrap();
        s.write("a", &[1, 2, 3], 2, &[5, 6]).unwrap();
        assert_eq!(s.tokens("a"), vec![1, 2, 3]);
        assert_eq!(s.read("a", 1, 3).unwrap(), vec![3, 4, 5, 6]);
        // Rewound to one shared token, then a different continuation.
        s.write("a", &[1, 9], 1, &[7, 8]).unwrap();
        assert_eq!(s.tokens("a"), vec![1, 9]);
        assert_eq!(s.read("a", 0, 2).unwrap(), vec![1, 2, 7, 8]);
        assert!(s.tokens("b").is_empty());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn ignores_files_from_before_bf16() {
        let (s, dir) = store(1 << 22);
        // What the f32 format left: no "format", 4 bytes an element.
        let (meta, data) = s.paths("a");
        std::fs::write(&data, [0u8; 16]).unwrap();
        let old = serde_json::json!({ "key": "a", "row": 2, "tokens": [1, 2] });
        std::fs::write(&meta, serde_json::to_vec(&old).unwrap()).unwrap();
        assert!(s.tokens("a").is_empty());
        // Saving starts the file over.
        s.write("a", &[1], 0, &[7, 8]).unwrap();
        assert_eq!(s.tokens("a"), vec![1]);
        assert_eq!(std::fs::metadata(&data).unwrap().len(), 4);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn the_writer_writes_in_order_and_flush_waits() {
        let (s, dir) = store(1 << 21);
        let w = super::Writer::new(s);
        w.write("a", vec![1, 2], 0, vec![1, 2, 3, 4]);
        w.write("a", vec![1, 3], 1, vec![5, 6]);
        w.flush();
        assert_eq!(w.store().tokens("a"), vec![1, 3]);
        assert_eq!(w.store().read("a", 0, 2).unwrap(), vec![1, 2, 5, 6]);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn trims_the_least_recently_used() {
        // Two positions of two elements: 8 bytes a conversation; room for one.
        let (s, dir) = store(10);
        s.write("a", &[1, 2], 0, &[0; 4]).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        s.write("b", &[1, 2], 0, &[0; 4]).unwrap();
        assert!(s.tokens("a").is_empty());
        assert_eq!(s.tokens("b"), vec![1, 2]);
        let _ = std::fs::remove_dir_all(dir);
    }
}
