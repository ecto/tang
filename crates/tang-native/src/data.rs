//! Token shards: the corpus as u32 little-endian files, every document
//! starting with BOS. `tokenize` (feature `tokenize`) writes them from text
//! or FineWeb-Edu parquet (feature `parquet`); [`Shards`] samples training
//! windows from them.
//!
//! Layout: `<dir>/train_00000.bin`, … and `<dir>/val_00000.bin`, …. 0.5% of
//! documents, chosen by a hash of their index, go to the validation split,
//! so a rerun over the same inputs makes the same split.

use crate::train::Rng;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Which split a document belongs to: deterministic, ~0.5% validation.
pub fn is_validation(doc: u64) -> bool {
    let mut x = doc.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    (x ^ (x >> 31)).is_multiple_of(200)
}

/// Appends tokens to fixed-size shard files.
pub struct ShardWriter {
    dir: PathBuf,
    prefix: &'static str,
    per_shard: usize,
    buf: Vec<u32>,
    shards: usize,
    pub tokens: u64,
}

impl ShardWriter {
    pub fn new(dir: &Path, prefix: &'static str, per_shard: usize) -> io::Result<ShardWriter> {
        std::fs::create_dir_all(dir)?;
        Ok(ShardWriter {
            dir: dir.to_path_buf(),
            prefix,
            per_shard,
            buf: Vec::with_capacity(per_shard.min(1 << 24)),
            shards: 0,
            tokens: 0,
        })
    }

    pub fn push(&mut self, ids: &[u32]) -> io::Result<()> {
        for &t in ids {
            self.buf.push(t);
            if self.buf.len() == self.per_shard {
                self.flush()?;
            }
        }
        self.tokens += ids.len() as u64;
        Ok(())
    }

    fn flush(&mut self) -> io::Result<()> {
        if self.buf.is_empty() {
            return Ok(());
        }
        let path = self
            .dir
            .join(format!("{}_{:05}.bin", self.prefix, self.shards));
        let mut f = io::BufWriter::new(File::create(path)?);
        for t in &self.buf {
            f.write_all(&t.to_le_bytes())?;
        }
        f.flush()?;
        self.shards += 1;
        self.buf.clear();
        Ok(())
    }

    pub fn finish(mut self) -> io::Result<(usize, u64)> {
        self.flush()?;
        Ok((self.shards, self.tokens))
    }
}

/// A split's shards, sampled as random windows.
pub struct Shards {
    files: Vec<(PathBuf, u64)>,
    total: u64,
}

impl Shards {
    /// Every `<prefix>_*.bin` in `dir`.
    pub fn open(dir: &Path, prefix: &str) -> io::Result<Shards> {
        let mut files: Vec<(PathBuf, u64)> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with(&format!("{prefix}_")) && n.ends_with(".bin"))
            })
            .map(|p| {
                let len = std::fs::metadata(&p).map(|m| m.len() / 4).unwrap_or(0);
                (p, len)
            })
            .collect();
        files.sort();
        let total = files.iter().map(|f| f.1).sum();
        if files.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("no {prefix}_*.bin shards in {}", dir.display()),
            ));
        }
        Ok(Shards { files, total })
    }

    pub fn tokens(&self) -> u64 {
        self.total
    }

    /// `batch` windows of `seq + 1` tokens: `(tokens, targets)` of
    /// `batch·seq` each, sequence-major. Windows start at a uniform random
    /// token (files weighted by size).
    pub fn sample(
        &self,
        rng: &mut Rng,
        batch: usize,
        seq: usize,
    ) -> io::Result<(Vec<u32>, Vec<u32>)> {
        let mut x = Vec::with_capacity(batch * seq);
        let mut y = Vec::with_capacity(batch * seq);
        for _ in 0..batch {
            let w = self.window(rng, seq + 1)?;
            x.extend_from_slice(&w[..seq]);
            y.extend_from_slice(&w[1..]);
        }
        Ok((x, y))
    }

    /// The `i`-th of the split's consecutive windows (deterministic eval).
    pub fn window_at(&self, i: u64, len: usize) -> io::Result<Option<Vec<u32>>> {
        let mut start = i * len as u64;
        for (p, n) in &self.files {
            if start + len as u64 <= *n {
                return read_at(p, start, len).map(Some);
            }
            start = start.saturating_sub(*n);
        }
        Ok(None)
    }

    fn window(&self, rng: &mut Rng, len: usize) -> io::Result<Vec<u32>> {
        let span: u64 = self
            .files
            .iter()
            .map(|f| f.1.saturating_sub(len as u64))
            .sum();
        if span == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "shards shorter than one window",
            ));
        }
        let mut at = ((rng.uniform() + 1.0) * 0.5 * span as f64) as u64 % span;
        for (p, n) in &self.files {
            let s = n.saturating_sub(len as u64);
            if at < s {
                return read_at(p, at, len);
            }
            at -= s;
        }
        unreachable!("at < span")
    }
}

fn read_at(p: &Path, start: u64, len: usize) -> io::Result<Vec<u32>> {
    let mut f = File::open(p)?;
    f.seek(SeekFrom::Start(start * 4))?;
    let mut b = vec![0u8; len * 4];
    f.read_exact(&mut b)?;
    Ok(b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect())
}

/// Tokenize documents into train/val shards. Inputs: `.txt` files (one
/// document each) and, with the `parquet` feature, `.parquet` files whose
/// `text` column holds one document per row (FineWeb-Edu).
#[cfg(feature = "tokenize")]
pub fn tokenize(
    tokenizer: &Path,
    inputs: &[PathBuf],
    out: &Path,
    per_shard: usize,
    max_tokens: Option<u64>,
) -> Result<(u64, u64), String> {
    let tok = tokenizers::Tokenizer::from_file(tokenizer).map_err(|e| format!("tokenizer: {e}"))?;
    let bos = tok
        .token_to_id("<|begin_of_text|>")
        .or_else(|| tok.token_to_id("<s>"))
        .ok_or("the tokenizer has no BOS token")?;
    let mut train = ShardWriter::new(out, "train", per_shard).map_err(|e| e.to_string())?;
    let mut val = ShardWriter::new(out, "val", per_shard).map_err(|e| e.to_string())?;
    let mut doc = 0u64;
    let mut emit =
        |text: &str, train: &mut ShardWriter, val: &mut ShardWriter| -> Result<bool, String> {
            let enc = tok
                .encode(text, false)
                .map_err(|e| format!("encode: {e}"))?;
            let mut ids = vec![bos];
            ids.extend_from_slice(enc.get_ids());
            if is_validation(doc) {
                val.push(&ids).map_err(|e| e.to_string())?;
            } else {
                train.push(&ids).map_err(|e| e.to_string())?;
            }
            doc += 1;
            Ok(max_tokens.is_some_and(|m| train.tokens >= m))
        };
    'files: for p in inputs {
        match p.extension().and_then(|e| e.to_str()) {
            Some("txt") | Some("md") => {
                let text =
                    std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
                if emit(&text, &mut train, &mut val)? {
                    break 'files;
                }
            }
            #[cfg(feature = "parquet")]
            Some("parquet") => {
                use parquet::file::reader::{FileReader, SerializedFileReader};
                use parquet::record::RowAccessor;
                let f = File::open(p).map_err(|e| format!("{}: {e}", p.display()))?;
                let r =
                    SerializedFileReader::new(f).map_err(|e| format!("{}: {e}", p.display()))?;
                let fields = r.metadata().file_metadata().schema().get_fields();
                let col = fields
                    .iter()
                    .position(|f| f.name() == "text")
                    .ok_or_else(|| format!("{}: no `text` column", p.display()))?;
                for row in r.get_row_iter(None).map_err(|e| e.to_string())? {
                    let row = row.map_err(|e| e.to_string())?;
                    let text = row.get_string(col).map_err(|e| e.to_string())?;
                    if emit(text, &mut train, &mut val)? {
                        break 'files;
                    }
                }
            }
            _ => return Err(format!("{}: expected .txt, .md or .parquet", p.display())),
        }
    }
    let (_, tt) = train.finish().map_err(|e| e.to_string())?;
    let (_, vt) = val.finish().map_err(|e| e.to_string())?;
    Ok((tt, vt))
}
