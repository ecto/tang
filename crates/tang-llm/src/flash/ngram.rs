//! The n-gram (PLE) table: 320M rows of 160 IQ4_NL values (90 B) in a 26.8 GiB tensor that
//! never goes through the page cache. A token needs 16 rows; each is one `O_DIRECT` read of the
//! 4 KiB page (or two) holding it, issued in parallel, behind a small row cache.

use crate::flash::reference::PleParams;
use crate::gguf::{dequantize, GgmlType, Gguf, TensorInfo};
use anyhow::{ensure, Context, Result};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::sync::Mutex;

const PAGE: usize = 4096;

/// A page-aligned buffer for `O_DIRECT`.
struct Aligned {
    ptr: *mut u8,
    len: usize,
}

impl Aligned {
    fn new(len: usize) -> Self {
        let layout = std::alloc::Layout::from_size_align(len, PAGE).unwrap();
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null());
        Aligned { ptr, len }
    }
    fn bytes(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for Aligned {
    fn drop(&mut self) {
        let layout = std::alloc::Layout::from_size_align(self.len, PAGE).unwrap();
        unsafe { std::alloc::dealloc(self.ptr, layout) };
    }
}

pub struct NgramTable {
    file: File,
    offset: u64,
    row_bytes: usize,
    n_rows: usize,
    width: usize,
    ty: GgmlType,
    p: PleParams,
    cache: Mutex<HashMap<u64, Box<[u8]>>>,
    /// Rows kept before the cache is cleared.
    pub cache_rows: usize,
    pub hits: std::sync::atomic::AtomicU64,
    pub reads: std::sync::atomic::AtomicU64,
    /// Readers, on the E-cores when the CPU has them (the P-cores spin for expert misses).
    io: rayon::ThreadPool,
}

/// E-core CPUs of a hybrid Intel part (empty elsewhere).
fn ecores() -> Vec<usize> {
    let Ok(s) = std::fs::read_to_string("/sys/devices/cpu_atom/cpus") else {
        return Vec::new();
    };
    let mut v = Vec::new();
    for part in s.trim().split(',') {
        if let Some((a, b)) = part.split_once('-') {
            if let (Ok(a), Ok(b)) = (a.parse::<usize>(), b.parse::<usize>()) {
                v.extend(a..=b);
            }
        } else if let Ok(a) = part.parse() {
            v.push(a);
        }
    }
    v
}

fn pin(cpu: usize) {
    #[cfg(target_os = "linux")]
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(cpu, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
    let _ = cpu;
}

impl NgramTable {
    pub fn open(g: &Gguf, p: &PleParams) -> Result<Self> {
        let t: &TensorInfo = g.info("per_layer_token_embd.weight")?;
        ensure!(t.row_len() == p.head_dim, "PLE table rows are {} wide", t.row_len());
        let path = g.shard_paths()[t.shard].to_path_buf();
        #[cfg(target_os = "linux")]
        let direct = libc::O_DIRECT;
        #[cfg(not(target_os = "linux"))]
        let direct = 0;
        let file = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(direct)
            .open(&path)
            .or_else(|_| File::open(&path))
            .with_context(|| format!("opening {}", path.display()))?;
        Ok(NgramTable {
            file,
            offset: t.offset,
            row_bytes: t.row_bytes()?,
            n_rows: t.n_rows(),
            width: t.row_len(),
            ty: t.ty,
            p: p.clone(),
            cache: Mutex::new(HashMap::new()),
            cache_rows: 1 << 18,
            hits: 0.into(),
            reads: 0.into(),
            io: {
                let e = ecores();
                rayon::ThreadPoolBuilder::new()
                    .num_threads(16)
                    .start_handler(move |i| {
                        if !e.is_empty() {
                            pin(e[i % e.len()]);
                        }
                    })
                    .build()?
            },
        })
    }

    fn read_raw(&self, row: u64) -> Result<Box<[u8]>> {
        use std::sync::atomic::Ordering::Relaxed;
        if let Some(b) = self.cache.lock().unwrap().get(&row) {
            self.hits.fetch_add(1, Relaxed);
            return Ok(b.clone());
        }
        self.reads.fetch_add(1, Relaxed);
        ensure!((row as usize) < self.n_rows, "n-gram row {row} out of range");
        let at = self.offset + row * self.row_bytes as u64;
        let start = at / PAGE as u64 * PAGE as u64;
        let end = (at + self.row_bytes as u64).div_ceil(PAGE as u64) * PAGE as u64;
        let mut buf = Aligned::new((end - start) as usize);
        let b = buf.bytes();
        let mut got = 0;
        while got < b.len() {
            let n = self.file.read_at(&mut b[got..], start + got as u64)?;
            ensure!(n > 0, "short read of the n-gram table");
            got += n;
        }
        let o = (at - start) as usize;
        let raw: Box<[u8]> = b[o..o + self.row_bytes].into();
        let mut c = self.cache.lock().unwrap();
        if c.len() >= self.cache_rows {
            c.clear();
        }
        c.insert(row, raw.clone());
        Ok(raw)
    }

    /// The gathered PLE inputs (`n_heads × head_dim` = 2560 floats each) for tokens `i0..i1`
    /// of `tokens` (the sequence so far), into `out[(i - i0) * 2560 ..]`. All rows of the window
    /// are read in parallel.
    pub fn gather(&self, tokens: &[u32], i0: usize, i1: usize, out: &mut [f32]) -> Result<()> {
        let per = self.p.n_heads();
        let rows: Vec<u64> = (i0..i1).flat_map(|i| self.p.rows(tokens, i)).collect();
        ensure!(out.len() >= rows.len() * self.width && rows.len() == (i1 - i0) * per, "PLE gather");
        let raws: Vec<Result<Box<[u8]>>> =
            self.io.install(|| rows.par_iter().map(|&r| self.read_raw(r)).collect());
        for (h, raw) in raws.into_iter().enumerate() {
            dequantize(self.ty, &raw?, &mut out[h * self.width..(h + 1) * self.width])?;
        }
        Ok(())
    }
}
