//! Fast boot caches for the flash engines: the post-requantization, device-layout weights and
//! the compiled kernels written to disk once and read back with large sequential reads on later
//! boots (`docs/fast-boot.md`).
//!
//! - `TANG_WEIGHT_CACHE=<dir>`: after the first [`super::coder::Coder::load`] of a GGUF, the
//!   exactly-rebuilt device bytes (including every requantization and row permutation) go to one
//!   blob file under the dir; later boots stream that file into VRAM instead of re-reading and
//!   re-transforming the GGUF. `0` disables it. Default off: nothing is written unless the dir
//!   is named.
//! - `TANG_KERNEL_CACHE=<dir>`: the compiled CUDA modules (PTX text, plus the CUBIN when nvrtc
//!   makes one for this arch) cached by source hash. `0` disables. Default off.

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

/// Current cache format version. Bump when any layout here changes.
pub const FORMAT: u32 = 1;

/// What every cache file starts with.
const MAGIC: &[u8; 8] = b"tangbc\0\x01";

/// The tang version + this format: the part of the cache key that isn't the model or source.
pub fn version_tag() -> String {
    format!("tang-{}-v{}", env!("CARGO_PKG_VERSION"), FORMAT)
}

/// An env var that names a directory; `0` or unset means off.
fn env_dir(var: &str) -> Result<Option<PathBuf>> {
    match std::env::var_os(var) {
        None => Ok(None),
        Some(v) if v == "0" => Ok(None),
        Some(v) => {
            let p = PathBuf::from(&v);
            if p.as_os_str().is_empty() {
                bail!("{var} is empty");
            }
            Ok(Some(p))
        }
    }
}

/// `TANG_WEIGHT_CACHE`, or `None` when unset/`0`.
pub fn weight_cache_dir() -> Result<Option<PathBuf>> {
    env_dir("TANG_WEIGHT_CACHE")
}

/// `TANG_KERNEL_CACHE`, or `None` when unset/`0`.
pub fn kernel_cache_dir() -> Result<Option<PathBuf>> {
    env_dir("TANG_KERNEL_CACHE")
}

/// `sha256` of `src` as hex, to name a kernel cache entry.
pub fn source_hash(src: &str) -> String {
    let mut h = Sha256::new();
    h.update(src.as_bytes());
    hex(&h.finalize())
}

fn hex(d: &[u8]) -> String {
    let mut s = String::with_capacity(d.len() * 2);
    for b in d {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// The cache key for a model: the content hash of every shard, plus the tang version, format and
/// `extra` (the GPU arch and any option that changes the device layout). Content, not
/// name/mtime: moving or re-copying a model keeps its cache, a changed file invalidates it.
pub fn model_key(paths: &[&Path], extra: &str) -> Result<String> {
    let mut h = Sha256::new();
    h.update(MAGIC);
    h.update(version_tag().as_bytes());
    h.update([0]);
    h.update(extra.as_bytes());
    h.update([0]);
    for p in paths {
        let f = fs::File::open(p).with_context(|| format!("hashing {}", p.display()))?;
        let mut r = std::io::BufReader::with_capacity(1 << 20, f);
        let mut b = [0u8; 1 << 16];
        loop {
            let n = r.read(&mut b)?;
            if n == 0 {
                break;
            }
            h.update(&b[..n]);
        }
    }
    Ok(hex(&h.finalize())[..40].to_string())
}

/// The weight blob for `key`: `<dir>/weights-<key>.bin`.
pub fn weight_blob_path(dir: &Path, key: &str) -> PathBuf {
    dir.join(format!("weights-{key}.bin"))
}

/// A weight cache being built during a first load: every device-side byte buffer, in load order.
/// Recording is zero-cost when the cache is off; a write error kills the recording, never the
/// boot (the next boot rebuilds).
pub struct WeightWriter {
    file: Option<BufWriter<fs::File>>,
    tmp: PathBuf,
    final_path: PathBuf,
    n: usize,
    bytes: u64,
    broken: Option<anyhow::Error>,
}

impl WeightWriter {
    /// Start a blob for `key` when the cache is on and the blob isn't there yet. When it's off
    /// (or already cached) the result still accepts [`Self::record`] as a no-op, so the load
    /// code stays one-shaped.
    pub fn create(dir: Option<&Path>, key: &str) -> Result<Self> {
        let (file, tmp, final_path) = match dir {
            Some(d) => {
                fs::create_dir_all(d)?;
                let p = weight_blob_path(d, key);
                if p.exists() {
                    (None, PathBuf::new(), p)
                } else {
                    let t = p.with_extension("tmp");
                    match Self::start(&t, key) {
                        Ok(w) => (Some(w), t, p),
                        Err(e) => {
                            eprintln!("boot cache: {e:#} (weights will not be cached)");
                            (None, PathBuf::new(), p)
                        }
                    }
                }
            }
            None => (None, PathBuf::new(), PathBuf::new()),
        };
        Ok(WeightWriter {
            file,
            tmp,
            final_path,
            n: 0,
            bytes: 0,
            broken: None,
        })
    }

    fn start(tmp: &Path, key: &str) -> Result<BufWriter<fs::File>> {
        let f = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(tmp)
            .with_context(|| format!("creating {}", tmp.display()))?;
        let mut w = BufWriter::with_capacity(4 << 20, f);
        w.write_all(MAGIC)?;
        w.write_all(&FORMAT.to_le_bytes())?;
        let v = version_tag();
        w.write_all(&(v.len() as u32).to_le_bytes())?;
        w.write_all(v.as_bytes())?;
        w.write_all(&(key.len() as u32).to_le_bytes())?;
        w.write_all(key.as_bytes())?;
        Ok(w)
    }

    pub fn is_on(&self) -> bool {
        self.file.is_some() && self.broken.is_none()
    }

    /// Queue one device buffer's bytes, in load order.
    pub fn record(&mut self, bytes: &[u8]) {
        if self.broken.is_some() || self.file.is_none() {
            return;
        }
        self.n += 1;
        self.bytes += bytes.len() as u64;
        if let Some(w) = self.file.as_mut() {
            if let Err(e) = w
                .write_all(&(bytes.len() as u64).to_le_bytes())
                .and_then(|_| w.write_all(bytes))
            {
                self.break_off(e.into());
            }
        }
    }

    fn break_off(&mut self, e: anyhow::Error) {
        self.file = None;
        let _ = fs::remove_file(&self.tmp);
        self.broken = Some(e);
    }

    /// Flush, fsync, rename into place. Returns whether a blob was written.
    pub fn finish(mut self) -> Result<bool> {
        match self.file.take() {
            None => {
                if let Some(e) = self.broken {
                    eprintln!("boot cache: weights not cached ({e:#})");
                }
                Ok(false)
            }
            Some(w) => {
                let n = self.n;
                let bytes = self.bytes;
                let (tmp, final_path) = (self.tmp.clone(), self.final_path.clone());
                let res = w
                    .into_inner()
                    .map_err(|e| e.into_error())
                    .and_then(|f| f.sync_all())
                    .with_context(|| format!("writing {}", tmp.display()))
                    .and_then(|()| fs::rename(&tmp, &final_path).map_err(Into::into));
                if let Err(e) = res {
                    let _ = fs::remove_file(&tmp);
                    eprintln!("boot cache: weights not cached ({e:#})");
                    return Ok(false);
                }
                eprintln!(
                    "boot cache: wrote {} ({} buffers, {:.2} GB)",
                    final_path.display(),
                    n,
                    bytes as f64 / 1e9
                );
                Ok(true)
            }
        }
    }
}

/// A weight cache being read during a later boot: hands out the recorded buffers in write order,
/// one large sequential slice at a time.
#[derive(Debug)]
pub struct WeightReader {
    map: memmap2::Mmap,
    data: usize,
    end: usize,
    /// How many buffers have been handed out (the load order must match the write order).
    pub pos: usize,
    /// The blob's path, for load reports.
    pub path: PathBuf,
}

impl WeightReader {
    /// Open the blob for `key` and check its header. Any error means "not usable": the caller
    /// falls back to a full load.
    pub fn open(dir: &Path, key: &str) -> Result<Self> {
        let p = weight_blob_path(dir, key);
        let f = fs::File::open(&p).with_context(|| format!("opening {}", p.display()))?;
        // Safety: the file's name is the model's content key; it isn't modified while we run.
        let map = unsafe { memmap2::Mmap::map(&f)? };
        let mut c = &map[..];
        if c.len() < MAGIC.len() + 4 || &c[..MAGIC.len()] != MAGIC {
            bail!("{}: not a tang boot cache", p.display());
        }
        c = &c[MAGIC.len()..];
        let format = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        c = &c[4..];
        if format != FORMAT {
            bail!("{}: format {format} (want {FORMAT})", p.display());
        }
        let u32_at = |c: &[u8]| u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        let vl = u32_at(c) as usize;
        c = &c[4..];
        let v =
            std::str::from_utf8(&c[..vl]).with_context(|| format!("{}: version", p.display()))?;
        c = &c[vl..];
        if v != version_tag() {
            bail!("{}: built by {v} (want {})", p.display(), version_tag());
        }
        let kl = u32_at(c) as usize;
        c = &c[4..];
        let k = std::str::from_utf8(&c[..kl]).with_context(|| format!("{}: key", p.display()))?;
        if k != key {
            bail!("{}: key mismatch", p.display());
        }
        // magic(8) format(4) version_len(4) version key_len(4) key
        let data = MAGIC.len() + 4 + 4 + vl + 4 + kl;
        let end = map.len();
        anyhow::ensure!(end >= data, "{}: truncated", p.display());
        Ok(WeightReader {
            map,
            data,
            end,
            pos: 0,
            path: p,
        })
    }

    /// The next recorded buffer's bytes, in write order. A length mismatch (a code change that
    /// reordered the load) is an error, not a panic: the caller falls back to a full load.
    pub fn next(&mut self, len: usize) -> Result<&[u8]> {
        let pos = self.pos;
        let (n, b) = self.next_bytes()?;
        anyhow::ensure!(n == len, "buffer {pos} is {n} bytes (want {len})");
        Ok(b)
    }

    /// The rest of the file as raw bytes (for a caller that streams it itself).
    pub fn rest(&self) -> &[u8] {
        &self.map[self.data..self.end]
    }

    fn next_bytes(&mut self) -> Result<(usize, &[u8])> {
        use std::borrow::Borrow;
        anyhow::ensure!(self.data + 8 <= self.end, "weight cache truncated");
        let (map, end) = (self.map.borrow(), self.end);
        let n = u64::from_le_bytes(map[self.data..self.data + 8].try_into().unwrap()) as usize;
        self.data += 8;
        anyhow::ensure!(self.data + n <= end, "weight cache truncated");
        let b = &map[self.data..self.data + n];
        self.data += n;
        self.pos += 1;
        Ok((n, b))
    }
}

/// One compiled kernel source: the PTX text, plus the CUBIN when nvrtc made one for this arch.
#[derive(PartialEq, Eq, Debug)]
pub struct KernelImage {
    pub ptx: String,
    pub cubin: Option<Vec<u8>>,
}

impl KernelImage {
    fn entry(dir: &Path, hash: &str) -> PathBuf {
        dir.join(format!("k-{hash}.bin"))
    }

    /// The cached image for `hash`, or `None` when there isn't one.
    pub fn read(dir: &Path, hash: &str) -> Result<Option<KernelImage>> {
        let bytes = match fs::read(Self::entry(dir, hash)) {
            Ok(b) => b,
            Err(_) => return Ok(None),
        };
        let mut c = &bytes[..];
        if c.len() < MAGIC.len() + 4 || &c[..MAGIC.len()] != MAGIC {
            bail!("kernel cache entry: not a tang boot cache");
        }
        c = &c[MAGIC.len()..];
        let format = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        c = &c[4..];
        if format != FORMAT {
            bail!("kernel cache entry: format {format} (want {FORMAT})");
        }
        let u32_at = |c: &[u8]| u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        let pl = u32_at(c) as usize;
        c = &c[4..];
        let ptx = String::from_utf8(c[..pl].to_vec()).context("kernel cache entry: ptx")?;
        c = &c[pl..];
        let cubin = match c.first() {
            Some(1) => {
                let cl = u32_at(&c[1..]) as usize;
                c = &c[5..];
                Some(c[..cl].to_vec())
            }
            _ => None,
        };
        Ok(Some(KernelImage { ptx, cubin }))
    }

    /// Store the image for `hash` (best effort; errors go to the log, not the caller).
    pub fn write(&self, dir: &Path, hash: &str) -> Result<()> {
        fs::create_dir_all(dir)?;
        let p = Self::entry(dir, hash);
        let tmp = p.with_extension("tmp");
        let mut w = BufWriter::new(fs::File::create(&tmp)?);
        w.write_all(MAGIC)?;
        w.write_all(&FORMAT.to_le_bytes())?;
        w.write_all(&(self.ptx.len() as u32).to_le_bytes())?;
        w.write_all(self.ptx.as_bytes())?;
        match &self.cubin {
            Some(c) => {
                w.write_all(&[1u8])?;
                w.write_all(&(c.len() as u32).to_le_bytes())?;
                w.write_all(c)?;
            }
            None => w.write_all(&[0u8])?,
        }
        w.into_inner()?.sync_all()?;
        fs::rename(&tmp, &p)?;
        Ok(())
    }
}

/// Q8_0 blocks (f16 scale, 32 int8 codes) of `v` (`v.len() % 32 == 0`) — the requantization
/// `upload_experts` performs for types the expert kernels don't decode.
pub fn quantize_q8_0(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() / 32 * 34);
    for b in v.chunks(32) {
        let amax = b.iter().fold(0f32, |m, x| m.max(x.abs()));
        let d = amax / 127.0;
        out.extend_from_slice(&crate::gguf::f32_to_f16(d).to_le_bytes());
        for &x in b {
            let q = if d == 0.0 {
                0
            } else {
                (x / d).round().clamp(-127.0, 127.0) as i8
            };
            out.push(q as u8);
        }
    }
    out
}

impl WeightReader {
    /// Delete the blob (a stale one, after its read failed: the caller reloads cold and
    /// rewrites it). Best effort.
    pub fn remove(&self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("tang-bootcache-{}", name));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn key_tracks_content_not_name() {
        let d = tmpdir("key");
        let a = d.join("a.gguf");
        let b = d.join("b.gguf");
        std::fs::write(&a, b"hello world").unwrap();
        std::fs::write(&b, b"hello world").unwrap();
        let ka = model_key(&[&a], "sm_90").unwrap();
        let kb = model_key(&[&b], "sm_90").unwrap();
        assert_eq!(ka, kb, "same content under another name: same key");
        std::fs::write(&b, b"hello world ").unwrap();
        assert_ne!(ka, model_key(&[&b], "sm_90").unwrap());
        // the layout-changing extra is in the key
        assert_ne!(ka, model_key(&[&a], "sm_100").unwrap());
        // a second shard is in the key
        std::fs::write(d.join("s.gguf"), b"x").unwrap();
        assert_ne!(ka, model_key(&[&a, &d.join("s.gguf")], "sm_90").unwrap());
    }

    #[test]
    fn blob_round_trip_and_invalidations() {
        let d = tmpdir("blob");
        let k = "deadbeef";
        {
            let mut w = WeightWriter::create(Some(&d), k).unwrap();
            assert!(w.is_on());
            w.record(b"aaaa");
            w.record(b"bbbbbb");
            assert!(w.finish().unwrap());
        }
        // the blob exists: a second create records nothing
        {
            let mut w = WeightWriter::create(Some(&d), k).unwrap();
            assert!(!w.is_on());
            w.record(b"zz");
            assert!(!w.finish().unwrap());
        }
        // read back, in order
        {
            let mut r = WeightReader::open(&d, k).unwrap();
            assert_eq!(r.next(4).unwrap(), b"aaaa");
            assert_eq!(r.next(6).unwrap(), b"bbbbbb");
            assert!(r.next(1).is_err(), "past the end");
        }
        // wrong length: an error to fall back on, not a panic
        {
            let mut r = WeightReader::open(&d, k).unwrap();
            assert!(r.next(5).is_err());
        }
        // wrong key
        assert!(WeightReader::open(&d, "other").is_err());
        // a truncated tail is caught on read (the header is intact)...
        let p = weight_blob_path(&d, k);
        let full = std::fs::read(&p).unwrap();
        std::fs::write(&p, &full[..full.len() - 3]).unwrap();
        {
            let mut r = WeightReader::open(&d, k).unwrap();
            assert!(r.next(4).is_ok(), "the first buffer is whole");
            assert!(r.next(6).is_err(), "the tail is gone");
        }
        // ...and a stale format is rejected at open
        std::fs::write(&p, b"tangbc\0\x02junk").unwrap();
        assert!(WeightReader::open(&d, k).is_err());
    }

    #[test]
    fn kernel_entry_round_trip() {
        let d = tmpdir("kernel");
        let h = source_hash("extern \"C\" __global__ void k() {}");
        assert!(KernelImage::read(&d, &h).unwrap().is_none());
        let img = KernelImage {
            ptx: "// ptx".into(),
            cubin: Some(vec![1, 2, 3, 4]),
        };
        img.write(&d, &h).unwrap();
        assert_eq!(KernelImage::read(&d, &h).unwrap().unwrap(), img);
        let none = KernelImage {
            ptx: "x".into(),
            cubin: None,
        };
        none.write(&d, "aaaa").unwrap();
        assert_eq!(KernelImage::read(&d, "aaaa").unwrap().unwrap(), none);
    }

    // ---- the device-layout functions, on a tiny hand-built GGUF ----

    fn ty_id(ty: crate::gguf::GgmlType) -> u32 {
        use crate::gguf::GgmlType::*;
        match ty {
            F32 => 0,
            F16 => 1,
            Q4_0 => 2,
            Q8_0 => 8,
            Bf16 => 30,
            other => panic!("test type {other:?}"),
        }
    }

    /// Write a minimal single-tensor-per-entry GGUF v3 (the same header `Gguf::open` parses):
    /// `meta` then `tensors` (`name`, `dims`, `type`, `data`), 32-byte aligned data.
    fn write_gguf(
        path: &Path,
        meta: &[(&str, crate::gguf::Value)],
        tensors: &[(String, Vec<u64>, crate::gguf::GgmlType, Vec<u8>)],
    ) {
        use crate::gguf::Value;
        // Header length depends only on names/dims (offsets are fixed-width), so assemble
        // once with zero offsets, find the data start, then rewrite with real offsets.
        let header = |offsets: &[u64]| -> Vec<u8> {
            let mut b = Vec::new();
            b.extend_from_slice(b"GGUF");
            b.extend_from_slice(&3u32.to_le_bytes());
            b.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
            b.extend_from_slice(&(meta.len() as u64).to_le_bytes());
            for (k, v) in meta {
                let k = k.as_bytes();
                b.extend_from_slice(&(k.len() as u64).to_le_bytes());
                b.extend_from_slice(k);
                let (ty, val): (u32, Vec<u8>) = match v {
                    Value::Str(s) => {
                        let s = s.as_bytes();
                        (
                            8,
                            s.len()
                                .to_le_bytes()
                                .iter()
                                .chain(s.iter())
                                .copied()
                                .collect(),
                        )
                    }
                    Value::U32(x) => (4, x.to_le_bytes().to_vec()),
                    Value::U64(x) => (10, x.to_le_bytes().to_vec()),
                    Value::F32(x) => (6, x.to_le_bytes().to_vec()),
                    Value::F64(x) => (12, x.to_le_bytes().to_vec()),
                    Value::Bool(x) => (7, vec![*x as u8]),
                    other => panic!("test meta {other:?}"),
                };
                b.extend_from_slice(&ty.to_le_bytes());
                b.extend_from_slice(&val);
            }
            for ((name, dims, t, _), &off) in tensors.iter().zip(offsets) {
                let n = name.as_bytes();
                b.extend_from_slice(&(n.len() as u64).to_le_bytes());
                b.extend_from_slice(n);
                b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
                for d in dims {
                    b.extend_from_slice(&d.to_le_bytes());
                }
                b.extend_from_slice(&ty_id(*t).to_le_bytes());
                b.extend_from_slice(&off.to_le_bytes());
            }
            b
        };
        let zeros = vec![0u64; tensors.len()];
        let head = header(&zeros);
        let data_start = (head.len()).div_ceil(32) * 32;
        let mut offs = Vec::with_capacity(tensors.len());
        let mut cursor = data_start;
        for (_, _, t, data) in tensors {
            offs.push((cursor - data_start) as u64);
            cursor += data.len();
            let _ = t;
        }
        let mut head = header(&offs);
        head.resize(data_start, 0);
        let mut file = head;
        for (_, _, _, data) in tensors {
            file.extend_from_slice(data);
        }
        std::fs::write(path, file).unwrap();
    }

    /// f32 rows as GGUF F32 tensor data.
    fn f32_data(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    /// Q8_0 blocks of `v` (deterministic scale from the amax — matches `quantize_q8_0`).
    fn q8_0_data(v: &[f32]) -> Vec<u8> {
        super::quantize_q8_0(v)
    }

    /// A cold load's device bytes must be bit-identical when re-read from a warm blob — the
    /// whole point of the weight cache (`docs/fast-boot.md`). CPU-only: the loaders' pure
    /// byte-building, a tiny GGUF, a blob written then read back.
    #[test]
    fn warm_blob_is_bit_identical_to_cold_load() {
        let d = tmpdir("bit-identical");
        // A tiny "layer": an F32 dense weight (k=64) to stack, a Q8_0 dense weight with
        // k=1024 (the Q8X repack path), a Q8_0 expert stack [4][8][64] (native), an F32
        // expert stack [2][8][32] (requantized to Q8_0), and an f32 norm.
        let gen = |seed: u64, n: usize| -> Vec<f32> {
            let mut x = seed;
            (0..n)
                .map(|_| {
                    x = x
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((x >> 33) as i32 % 97) as f32 / 24.0 - 2.0
                })
                .collect()
        };
        let dense_f32 = gen(1, 64 * 4); // [4, 64]
        let dense_q8: Vec<f32> = gen(2, 1024 * 2); // [2, 1024]
        let exp_native: Vec<f32> = gen(3, 4 * 8 * 64); // [64, 8, 4] (dims[0] fastest)
        let exp_f32: Vec<f32> = gen(4, 32 * 8 * 2); // [32, 8, 2]
        let norm = gen(5, 64);
        let gguf = d.join("tiny.gguf");
        write_gguf(
            &gguf,
            &[(
                "general.architecture",
                crate::gguf::Value::Str("test".into()),
            )],
            &[
                (
                    "w1.weight".into(),
                    vec![64, 4],
                    crate::gguf::GgmlType::F32,
                    f32_data(&dense_f32),
                ),
                (
                    "w2.weight".into(),
                    vec![1024, 2],
                    crate::gguf::GgmlType::Q8_0,
                    q8_0_data(&dense_q8),
                ),
                (
                    "e1.weight".into(),
                    vec![64, 8, 4],
                    crate::gguf::GgmlType::Q8_0,
                    q8_0_data(&exp_native),
                ),
                (
                    "e2.weight".into(),
                    vec![32, 8, 2],
                    crate::gguf::GgmlType::F32,
                    f32_data(&exp_f32),
                ),
                (
                    "norm.weight".into(),
                    vec![64],
                    crate::gguf::GgmlType::F32,
                    f32_data(&norm),
                ),
            ],
        );
        let g = crate::gguf::Gguf::open(&gguf).expect("open tiny gguf");
        // The GPU loaders' upload order, as pure functions: dense stack (2 segments), the
        // experts, the norm.
        let cold = |mut w: Option<&mut super::WeightWriter>| -> Vec<Vec<u8>> {
            let mut out = Vec::new();
            let mut record = |bytes: &[u8]| {
                if let Some(w) = w.as_deref_mut() {
                    w.record(bytes);
                }
            };
            for bytes in [
                super::dense_bytes(&g, g.info("w1.weight").unwrap(), None).unwrap(),
                super::dense_bytes(&g, g.info("w2.weight").unwrap(), None).unwrap(),
                super::experts_bytes(&g, g.info("e1.weight").unwrap()).unwrap(),
                super::experts_bytes(&g, g.info("e2.weight").unwrap()).unwrap(),
                super::f32_bytes(&g, "norm.weight").unwrap(),
            ] {
                record(&bytes);
                out.push(bytes);
            }
            out
        };
        // cold load: build and record
        let key = "test-key";
        {
            let mut w = super::WeightWriter::create(Some(&d), key).unwrap();
            assert!(w.is_on());
            let bytes = cold(Some(&mut w));
            assert_eq!(bytes.len(), 5);
            // the layouts the loaders would compute agree with the built bytes
            let (ty, rb, want) = super::dense_layout(g.info("w1.weight").unwrap());
            assert_eq!((ty, rb), (0, 64 * 4));
            assert_eq!(bytes[0].len(), want);
            let (ty, rb, want) = super::dense_layout(g.info("w2.weight").unwrap());
            assert_eq!((ty, rb), (108, 1024 + 2 * 32));
            assert_eq!(bytes[1].len(), want);
            let (ty, rb, want) = super::experts_layout(g.info("e1.weight").unwrap());
            assert_eq!((ty, rb), (8, 64 / 32 * 34));
            assert_eq!(bytes[2].len(), want);
            let (ty, rb, want) = super::experts_layout(g.info("e2.weight").unwrap());
            assert_eq!((ty, rb), (8, 32 / 32 * 34));
            assert_eq!(bytes[3].len(), want);
            assert!(w.finish().unwrap());
        }
        // warm load: read the blob back, buffer for buffer, bit for bit
        let fresh = cold(None);
        {
            let mut r = super::WeightReader::open(&d, key).unwrap();
            for (i, want) in fresh.iter().enumerate() {
                let got = r
                    .next(want.len())
                    .unwrap_or_else(|e| panic!("buffer {i}: {e}"));
                assert_eq!(got, &want[..], "buffer {i} differs");
            }
        }
    }

    /// The Q8X repack is lossless: codes and scales reordered, not re-quantized.
    #[test]
    fn q8x_repack_is_lossless() {
        let d = tmpdir("q8x");
        let v: Vec<f32> = (0..1024)
            .map(|i| ((i * 37) % 101) as f32 / 50.0 - 1.0)
            .collect();
        let gguf = d.join("q8.gguf");
        write_gguf(
            &gguf,
            &[],
            &[(
                "w.weight".into(),
                vec![1024, 1],
                crate::gguf::GgmlType::Q8_0,
                q8_0_data(&v),
            )],
        );
        let g = crate::gguf::Gguf::open(&gguf).unwrap();
        let t = g.info("w.weight").unwrap();
        let bytes = super::dense_bytes(&g, t, None).unwrap();
        let (ty, rb, want) = super::dense_layout(t);
        assert_eq!((ty, rb, want), (108, 1024 + 64, 1024 + 64 + 16));
        // codes first, then the f16 scales, of every block
        let raw = q8_0_data(&v);
        let nb = 1024 / 32;
        let mut expect = Vec::new();
        for b in 0..nb {
            expect.extend_from_slice(&raw[b * 34 + 2..b * 34 + 34]);
        }
        for b in 0..nb {
            expect.extend_from_slice(&raw[b * 34..b * 34 + 2]);
        }
        assert_eq!(&bytes[..expect.len()], &expect[..]);
        assert_eq!(&bytes[expect.len()..], &[0u8; 16]);
    }

    /// A row permutation lands as rows, in the permuted order.
    #[test]
    fn permuted_rows() {
        let d = tmpdir("perm");
        let v: Vec<f32> = (0..8)
            .map(|r| (0..32).map(move |c| r as f32 + c as f32 / 100.0))
            .flatten()
            .collect();
        let gguf = d.join("p.gguf");
        write_gguf(
            &gguf,
            &[],
            &[(
                "ba.weight".into(),
                vec![32, 8],
                crate::gguf::GgmlType::F32,
                f32_data(&v),
            )],
        );
        let g = crate::gguf::Gguf::open(&gguf).unwrap();
        let t = g.info("ba.weight").unwrap();
        let perm: Vec<usize> = (0..8).map(|i| (i * 3) % 8).collect();
        let bytes = super::dense_bytes(&g, t, Some(&perm)).unwrap();
        for (i, &j) in perm.iter().enumerate() {
            let row = &bytes[i * 128..(i + 1) * 128];
            let want = &f32_data(&v)[j * 128..(j + 1) * 128];
            assert_eq!(row, want, "row {i}");
        }
    }

    #[test]
    fn env_dirs() {
        // cache dirs are opt-in; `0` and unset both mean off
        let key = "TANG_WEIGHT_CACHE_BOOTCACHE_TEST";
        std::env::remove_var(key);
        assert!(env_dir(key).unwrap().is_none());
        std::env::set_var(key, "0");
        assert!(env_dir(key).unwrap().is_none());
        std::env::set_var(key, "/tmp/x-bootcache");
        assert_eq!(
            env_dir(key).unwrap().unwrap(),
            Path::new("/tmp/x-bootcache")
        );
        std::env::remove_var(key);
    }
}

// -------------------------------------------------------------------------------------------
// The Coder's device weight layout, GPU-free
// -------------------------------------------------------------------------------------------
//
// The exact bytes a `Coder::load` puts in VRAM, as pure functions of the GGUF's tensor
// directory (no CUDA, no VRAM): the flash engine's loaders (`upload_dense` /
// `upload_experts`) are thin `dev.upload_bytes(...)` wrappers around these, so
//
// - the warm-boot read order and lengths are exactly the blob's, by construction;
// - the CPU test below proves a blob read back is bit-identical to a fresh build;
// - the kernel side of the engine never needs the GPU to know the layout.

/// The ggml type id the native GEMV decodes, if it does (mirrors `coder::native_id`).
pub fn native_id(ty: crate::gguf::GgmlType) -> Option<u32> {
    use crate::gguf::GgmlType::*;
    Some(match ty {
        F32 => 0,
        Q4_0 => 2,
        Q5_0 => 6,
        Q8_0 => 8,
        Q3K => 11,
        Q4K => 12,
        Q5K => 13,
        Q6K => 14,
        Iq4Nl => 20,
        Iq4Xs => 23,
        Bf16 => 30,
        Q2_0 => 42,
        _ => return None,
    })
}

fn env_is(key: &str, val: &str) -> bool {
    std::env::var(key).is_ok_and(|v| v == val)
}

/// One stacked dense segment's device layout: `(type id, row bytes, upload length)`, the exact
/// bytes the native GEMV reads plus 16 bytes of row-staging slack. Pure — warm boots rely on
/// cold and warm agreeing here. `TANG_CODER_ROUTER16` / `TANG_CODER_Q8X` are part of the shape.
pub fn dense_layout(t: &crate::gguf::TensorInfo) -> (u32, usize, usize) {
    let k = t.row_len();
    let rows = t.n_rows();
    let (ty, rb) = match native_id(t.ty) {
        Some(id) => (id, t.row_bytes().unwrap_or(0)),
        None => (0, k * 4),
    };
    if ty == 0 && env_is("TANG_CODER_ROUTER16", "1") {
        // F32 rows (the router) as bf16, half the bytes.
        let rb = k * 2;
        (30, rb, rows * rb + 16)
    } else if ty == 8 && k.is_multiple_of(1024) && !env_is("TANG_CODER_Q8X", "0") {
        // Q8X (type 108, default; TANG_CODER_Q8X=0 keeps Q8_0): [k codes][k / 32 f16 scales] a
        // row, lossless.
        let rb = k + 2 * (k / 32);
        (108, rb, rows * rb + 16)
    } else {
        (ty, rb, rows * rb + 16)
    }
}

/// The cold-path device bytes of one stacked dense segment (with an optional row permutation,
/// output row `i` = input row `perm[i]`): native bytes, or dequantized f32 for types the GEMV
/// doesn't decode, permuted, converted per `dense_layout`'s type, then 16 slack bytes.
pub fn dense_bytes(
    g: &crate::gguf::Gguf,
    t: &crate::gguf::TensorInfo,
    perm: Option<&[usize]>,
) -> Result<Vec<u8>> {
    let (ty, _rb, _len) = dense_layout(t);
    let k_ty = native_id(t.ty);
    let mut bytes = match k_ty {
        Some(id) if id != 0 => g.bytes(t).to_vec(),
        _ => g
            .dequantize(t)?
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect(),
    };
    if let Some(p) = perm {
        let rb = t.row_bytes()?;
        anyhow::ensure!(
            p.len() == t.n_rows(),
            "{}: permutation of {} rows",
            t.name,
            p.len()
        );
        let src = bytes.clone();
        for (i, &j) in p.iter().enumerate() {
            bytes[i * rb..(i + 1) * rb].copy_from_slice(&src[j * rb..(j + 1) * rb]);
        }
    }
    // (router16: any source `dense_layout` turned into bf16 rows — F32 or an undecodable
    // type dequantized to f32 — narrows to bf16; natively bf16 rows pass through.)
    if ty == 30 && k_ty != Some(30) {
        bytes = bytes
            .chunks(4)
            .flat_map(|c| {
                let x = f32::from_le_bytes([c[0], c[1], c[2], c[3]]).to_bits();
                (((x + 0x7fff + ((x >> 16) & 1)) >> 16) as u16).to_le_bytes()
            })
            .collect();
    } else if ty == 108 {
        let k = t.row_len();
        let nb = k / 32;
        let rb = 34 * nb;
        let mut out = Vec::with_capacity(t.n_rows() * (k + 2 * nb));
        for r in 0..t.n_rows() {
            let row = &bytes[r * rb..(r + 1) * rb];
            for b in 0..nb {
                out.extend_from_slice(&row[b * 34 + 2..b * 34 + 34]);
            }
            for b in 0..nb {
                out.extend_from_slice(&row[b * 34..b * 34 + 2]);
            }
        }
        bytes = out;
    }
    // 16 bytes of slack: the GEMV stages rows with aligned 16-byte loads.
    bytes.extend_from_slice(&[0u8; 16]);
    Ok(bytes)
}

/// An expert stack's device layout: the GGUF type when the expert kernels decode it, else Q8_0
/// (`(type id, row bytes, upload length)`, slack included).
pub fn experts_layout(t: &crate::gguf::TensorInfo) -> (u32, usize, usize) {
    let (k, rows, ne) = expert_dims(t);
    match native_id(t.ty).filter(|&id| id != 0 && id != 30) {
        Some(id) => (
            id,
            t.row_bytes().unwrap_or(0),
            ne * rows * t.row_bytes().unwrap_or(0) + 16,
        ),
        None => {
            let rb = k / 32 * 34;
            (8, rb, ne * rows * rb + 16)
        }
    }
}

/// `(k, rows, n_expert)` of an expert tensor.
pub fn expert_dims(t: &crate::gguf::TensorInfo) -> (usize, usize, usize) {
    let (k, rows) = (t.dims[0] as usize, t.dims[1] as usize);
    let ne = t.dims.get(2).map_or(1, |&d| d as usize);
    (k, rows, ne)
}

/// The cold-path device bytes of an expert stack: the GGUF's bytes when the kernels decode the
/// type, else every expert dequantized and requantized to Q8_0 (what the log line
/// "requantized to Q8_0" is about), plus 16 slack bytes.
pub fn experts_bytes(g: &crate::gguf::Gguf, t: &crate::gguf::TensorInfo) -> Result<Vec<u8>> {
    let (k, rows, ne) = expert_dims(t);
    let mut bytes = match native_id(t.ty).filter(|&id| id != 0 && id != 30) {
        Some(_) => g.bytes(t).to_vec(),
        None => {
            let mut out = Vec::with_capacity(ne * rows * k / 32 * 34);
            for e in 0..ne {
                let v = if ne == 1 {
                    g.dequantize(t)?
                } else {
                    g.expert(t, e)?
                };
                out.extend_from_slice(&quantize_q8_0(&v));
            }
            out
        }
    };
    bytes.extend_from_slice(&[0u8; 16]);
    Ok(bytes)
}

/// One f32 tensor's device bytes (norms, biases, conv rows), dequantized from whatever type the
/// GGUF holds.
pub fn f32_bytes(g: &crate::gguf::Gguf, name: &str) -> Result<Vec<u8>> {
    let v = g.tensor_f32(name)?;
    Ok(v.iter().flat_map(|x| x.to_le_bytes()).collect())
}
