//! KV caches on disk, so a conversation that lost its slot (or the server restarting) costs a
//! read instead of a prefill.
//!
//! Each conversation (`prompt_cache_key`) is two files: `<hash>.kv`, its positions' keys and
//! values (f32, position-major: every layer's K row then V row), and `<hash>.json`, the tokens
//! those positions hold. Saving after a turn appends only the new positions; a conversation
//! that was rewound is cut back to what it still shares first. Exact: rows are stored as
//! computed. The oldest files go when the directory passes its budget.

use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::fs::{self, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub struct Store {
    dir: PathBuf,
    /// Most bytes of `.kv` files to keep.
    budget: u64,
    /// Floats per position.
    row: usize,
}

impl Store {
    /// A store in `dir` (one per model and weight format) holding positions of `row` floats.
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
        self.row as u64 * 4
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
        if v["key"] != key || v["row"].as_u64() != Some(self.row as u64) {
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
    pub fn read(&self, key: &str, from: usize, to: usize) -> Result<Vec<f32>> {
        let (_, data) = self.paths(key);
        let mut f = fs::File::open(&data)?;
        f.seek(SeekFrom::Start(from as u64 * self.row_bytes()))?;
        let mut bytes = vec![0u8; (to - from) * self.row * 4];
        f.read_exact(&mut bytes)?;
        // Touch it: the budget drops the least recently used first.
        let _ = f.set_modified(std::time::SystemTime::now());
        Ok(bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect())
    }

    /// Record that `key`'s cache holds `tokens`, whose positions from `from` on are `rows`
    /// (positions before `from` are already on disk).
    pub fn write(&self, key: &str, tokens: &[u32], from: usize, rows: &[f32]) -> Result<()> {
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
        let mut bytes = Vec::with_capacity(rows.len() * 4);
        for x in rows {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        f.write_all(&bytes)?;
        let tmp = meta.with_extension("json.tmp");
        fs::write(
            &tmp,
            serde_json::to_vec(&json!({ "key": key, "row": self.row, "tokens": tokens }))?,
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
        s.write("a", &[1, 2], 0, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        s.write("a", &[1, 2, 3], 2, &[5.0, 6.0]).unwrap();
        assert_eq!(s.tokens("a"), vec![1, 2, 3]);
        assert_eq!(s.read("a", 1, 3).unwrap(), vec![3.0, 4.0, 5.0, 6.0]);
        // Rewound to one shared token, then a different continuation.
        s.write("a", &[1, 9], 1, &[7.0, 8.0]).unwrap();
        assert_eq!(s.tokens("a"), vec![1, 9]);
        assert_eq!(s.read("a", 0, 2).unwrap(), vec![1.0, 2.0, 7.0, 8.0]);
        assert!(s.tokens("b").is_empty());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn trims_the_least_recently_used() {
        // Two positions of two floats: 16 bytes a conversation; room for one.
        let (s, dir) = store(20);
        s.write("a", &[1, 2], 0, &[0.0; 4]).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        s.write("b", &[1, 2], 0, &[0.0; 4]).unwrap();
        assert!(s.tokens("a").is_empty());
        assert_eq!(s.tokens("b"), vec![1, 2]);
        let _ = std::fs::remove_dir_all(dir);
    }
}
