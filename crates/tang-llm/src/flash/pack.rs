//! Repacking Flash-Next's GGUF into the device formats the kernels take
//! (`tang_compute::flash`), once, with the result cached on disk under
//! `~/.cache/tang/flash/<hash>/`:
//!
//! - `dense.bin` + `dense.json`: every non-expert tensor the window reads, already in its device
//!   bytes (Q4X, bf16, f32, raw Q2_0), by name.
//! - `experts.bin`: all 48 × 512 routed experts as tang-compute `ExpertBlob`s (1,382,400 B), in
//!   key order (`layer · 512 + expert`).
//!
//! **Formats and what they cost.** Kept native (lossless): every bf16 tensor (hyper-connections,
//! router and shared gate, `ssm_alpha/beta`, indexer projections, `ple_value`), every f32 tensor,
//! the routed experts and `ple_key` (Q2_0). Requantized to Q4X (tang-Q4 affine, group 64, bf16
//! scale and bias, 4.5 bpw; the only dense format with an int8-activation GEMV): the GDN and QSA
//! input/output projections, the shared experts and the LM head, whose GGUF types are IQ4_XS,
//! Q2_0, Q3_K, Q4_K, Q5_K, Q6_K, Q4_0, Q5_0, Q8_0 and IQ4_NL depending on the layer. That is
//! dequant → requant, a second rounding on top of the file's; the group's scale and bias are fit
//! by least squares (not min/max), which roughly halves the added error. The end-to-end cost is
//! measured by the parity checks (`flash-parity`).

use crate::gguf::{GgmlType, Gguf, TensorInfo};
use anyhow::{bail, ensure, Context, Result};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use tang_compute::flash::shape::*;
use tang_compute::flash::{self as fl, ExpertBlob};

/// Bump when any packed format changes.
pub const FORMAT: &str = "flash-pack-v1";

/// A packed tensor's device format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Fmt {
    /// `[n][k]` Q4X ([`fl::q4x_repack`] bytes).
    Q4x,
    /// `[n][k]` bf16 bits.
    Bf16,
    /// f32 values.
    F32,
    /// Raw GGUF Q2_0 blocks `[n][k]` (`upload_q2` repacks).
    Q2Raw,
    /// Native GGUF tensors for `fe_gemv` (`Entry::segs` says where each starts).
    Native,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Entry {
    pub name: String,
    pub fmt: Fmt,
    pub n: usize,
    pub k: usize,
    pub offset: u64,
    pub len: u64,
    /// Native segments: `[ggml type, rows, row bytes, byte offset in the entry, output column]`.
    #[serde(default)]
    pub segs: Vec<[u64; 5]>,
}

/// Where the repack of `gguf` is cached.
pub fn cache_dir(first_shard: &Path) -> Result<PathBuf> {
    use std::hash::{Hash, Hasher};
    let canon = std::fs::canonicalize(first_shard)?;
    let meta = std::fs::metadata(&canon)?;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    FORMAT.hash(&mut h);
    canon.hash(&mut h);
    meta.len().hash(&mut h);
    meta.modified()?.hash(&mut h);
    let home = std::env::var_os("HOME").context("HOME")?;
    let base = std::env::var_os("TANG_FLASH_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(home).join(".cache/tang/flash"));
    Ok(base.join(format!("{:016x}", h.finish())))
}

// ---- Q4 with a least-squares fit ----

fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return ((b >> 16) | 0x40) as u16;
    }
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn from_bf16(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// One 64-weight group: `(q, scale bits, bias bits)` with `w ≈ scale · q + bias`, `q` in 0..16.
/// Starts from min/max and refines (scale, bias) by least squares on the current assignment,
/// keeping whichever stored (bf16-rounded) pair has the smallest squared error.
fn q4_group(w: &[f32], q: &mut [u8; 64]) -> (u16, u16) {
    let lo = w.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = w.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let assign = |s: f32, b: f32, q: &mut [u8; 64]| -> f64 {
        let mut e = 0f64;
        for (j, &v) in w.iter().enumerate() {
            let qi = ((v - b) / s).round().clamp(0.0, 15.0);
            q[j] = qi as u8;
            let d = (s * qi + b - v) as f64;
            e += d * d;
        }
        e
    };
    let mut s = from_bf16(bf16_bits(((hi - lo) / 15.0).max(1e-8)));
    let mut b = from_bf16(bf16_bits(lo));
    let mut best = (assign(s, b, q), bf16_bits(s), bf16_bits(b));
    let mut cur = *q;
    for _ in 0..4 {
        // Least squares for (s, b) given q.
        let n = w.len() as f64;
        let (mut sq, mut sqq, mut sv, mut sqv) = (0f64, 0f64, 0f64, 0f64);
        for (j, &v) in w.iter().enumerate() {
            let qi = cur[j] as f64;
            sq += qi;
            sqq += qi * qi;
            sv += v as f64;
            sqv += qi * v as f64;
        }
        let den = n * sqq - sq * sq;
        if den.abs() < 1e-20 {
            break;
        }
        let ns = (n * sqv - sq * sv) / den;
        let nb = (sv - ns * sq) / n;
        if ns <= 0.0 {
            break;
        }
        s = from_bf16(bf16_bits(ns as f32).max(1));
        b = from_bf16(bf16_bits(nb as f32));
        let e = assign(s, b, &mut cur);
        if e < best.0 {
            best = (e, bf16_bits(s), bf16_bits(b));
            *q = cur;
        } else {
            break;
        }
    }
    (best.1, best.2)
}

/// Quantize `[n][k]` f32 rows straight into Q4X bytes (the layout of [`fl::q4x_repack`]).
pub fn q4x_pack(w: &[f32], n: usize, k: usize) -> Vec<u8> {
    assert!(k.is_multiple_of(64) && w.len() == n * k);
    let gpr = k / 64;
    // Per row: codes (k/2 bytes), scales (gpr), biases (gpr).
    let rows: Vec<(Vec<u8>, Vec<u16>, Vec<u16>)> = (0..n)
        .into_par_iter()
        .map(|r| {
            let mut codes = vec![0u8; k / 2];
            let (mut sc, mut bi) = (Vec::with_capacity(gpr), Vec::with_capacity(gpr));
            let mut q = [0u8; 64];
            for g in 0..gpr {
                let (s, b) = q4_group(&w[r * k + g * 64..r * k + g * 64 + 64], &mut q);
                sc.push(s);
                bi.push(b);
                for (j, &qi) in q.iter().enumerate() {
                    let e = g * 64 + j;
                    let (byte, sh) = fl::q4x_slot(e % 32);
                    codes[(e / 32) * 16 + byte] |= qi << sh;
                }
            }
            (codes, sc, bi)
        })
        .collect();
    let mut out = Vec::with_capacity(n * k / 2 + 4 * n * gpr);
    for (c, _, _) in &rows {
        out.extend_from_slice(c);
    }
    for (_, s, _) in &rows {
        for v in s {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    for (_, _, b) in &rows {
        for v in b {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

/// Dequantized Q4X rows (tests and error reports).
pub fn q4x_unpack(b: &[u8], n: usize, k: usize) -> Vec<f32> {
    let gpr = k / 64;
    let (sp, bp) = (n * k / 2, n * k / 2 + 2 * n * gpr);
    let mut out = vec![0f32; n * k];
    for r in 0..n {
        for e in 0..k {
            let (byte, sh) = fl::q4x_slot(e % 32);
            let q = (b[r * k / 2 + (e / 32) * 16 + byte] >> sh) & 0xf;
            let gi = r * gpr + e / 64;
            let s = from_bf16(u16::from_le_bytes([b[sp + 2 * gi], b[sp + 2 * gi + 1]]));
            let bb = from_bf16(u16::from_le_bytes([b[bp + 2 * gi], b[bp + 2 * gi + 1]]));
            out[r * k + e] = s * q as f32 + bb;
        }
    }
    out
}

// ---- dense pack ----

/// Rows of an (up to 2-D) GGUF tensor: `(n rows, k wide)`.
fn nk(t: &TensorInfo) -> (usize, usize) {
    (t.n_rows(), t.row_len())
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// A tensor's values as bf16 bits (exact for BF16 tensors; rounded otherwise).
fn bf16_of(g: &Gguf, t: &TensorInfo) -> Result<Vec<u8>> {
    if t.ty == GgmlType::Bf16 {
        return Ok(g.bytes(t).to_vec());
    }
    Ok(g
        .dequantize(t)?
        .iter()
        .flat_map(|&x| bf16_bits(x).to_le_bytes())
        .collect())
}

/// One packing job: a name and how to build its bytes.
enum Job {
    /// Rows of several tensors stacked (missing tail rows zero), requantized to Q4X.
    Q4Stack {
        names: Vec<String>,
        rows: usize,
        class: &'static str,
        /// The same rows as native GGUF segments (tensor, output column), when the class is
        /// packed native.
        segs: Vec<(String, usize)>,
    },
    /// Rows of several tensors stacked, bf16.
    Bf16Stack { names: Vec<String> },
    /// A tensor as f32.
    F32(String),
    /// A Q2_0 tensor's raw blocks.
    Q2(String),
    /// The hyper-connection up projection, bf16, repacked by `hc_up_repack`.
    HcUp(String),
}

fn jobs(g: &Gguf, n_layer: usize, is_rec: &[bool], ple_layer: Option<usize>) -> Result<Vec<(String, Job)>> {
    let b = |l: usize, s: &str| format!("blk.{l}.{s}");
    let mut v: Vec<(String, Job)> = Vec::new();
    let hc = |v: &mut Vec<(String, Job)>, key: &str, pre: &str, inject: bool| {
        v.push((format!("{key}.norm"), Job::F32(format!("{pre}norm.weight"))));
        v.push((
            format!("{key}.down"),
            Job::Bf16Stack {
                names: vec![format!("{pre}down.weight")],
            },
        ));
        v.push((format!("{key}.up"), Job::HcUp(format!("{pre}up.weight"))));
        if inject {
            v.push((
                format!("{key}.inject"),
                Job::Bf16Stack {
                    names: vec![format!("{pre}inject.weight")],
                },
            ));
        }
    };
    for l in 0..n_layer {
        let k = format!("L{l:02}");
        hc(&mut v, &format!("{k}.hc_attn"), &b(l, "hc_attn_"), true);
        hc(&mut v, &format!("{k}.hc_ffn"), &b(l, "hc_ffn_"), true);
        if is_rec[l] {
            v.push((
                format!("{k}.w_in"),
                Job::Q4Stack {
                    names: vec![b(l, "attn_qkv.weight"), b(l, "attn_gate.weight")],
                    rows: GDN_PROJ,
                    class: "w_in",
                    segs: vec![(b(l, "attn_qkv.weight"), 0), (b(l, "attn_gate.weight"), GDN_Z), (b(l, "ssm_alpha.weight"), GDN_A), (b(l, "ssm_beta.weight"), GDN_B)],
                },
            ));
            v.push((
                format!("{k}.side"),
                Job::Bf16Stack {
                    names: vec![b(l, "ssm_alpha.weight"), b(l, "ssm_beta.weight")],
                },
            ));
            v.push((
                format!("{k}.w_out"),
                Job::Q4Stack {
                    names: vec![b(l, "ssm_out.weight")],
                    rows: HIDDEN,
                    class: "w_out",
                    segs: vec![(b(l, "ssm_out.weight"), 0)],
                },
            ));
            for (n, t) in [
                ("conv", "ssm_conv1d.weight"),
                ("dt", "ssm_dt.bias"),
                ("a", "ssm_a"),
                ("norm", "ssm_norm.weight"),
            ] {
                v.push((format!("{k}.{n}"), Job::F32(b(l, t))));
            }
        } else {
            v.push((
                format!("{k}.w_in"),
                Job::Q4Stack {
                    names: vec![
                        b(l, "attn_q.weight"),
                        b(l, "attn_k.weight"),
                        b(l, "attn_v.weight"),
                    ],
                    rows: QSA_PROJ,
                    class: "w_in",
                    segs: vec![(b(l, "attn_q.weight"), 0), (b(l, "attn_k.weight"), QSA_K), (b(l, "attn_v.weight"), QSA_V), (b(l, "indexer.q_proj.weight"), QSA_IQ), (b(l, "indexer.k_proj.weight"), QSA_IK)],
                },
            ));
            v.push((
                format!("{k}.side"),
                Job::Bf16Stack {
                    names: vec![b(l, "indexer.q_proj.weight"), b(l, "indexer.k_proj.weight")],
                },
            ));
            v.push((
                format!("{k}.w_out"),
                Job::Q4Stack {
                    names: vec![b(l, "attn_output.weight")],
                    rows: HIDDEN,
                    class: "w_out",
                    segs: vec![(b(l, "attn_output.weight"), 0)],
                },
            ));
            for (n, t) in [
                ("qn", "attn_q_norm.weight"),
                ("kn", "attn_k_norm.weight"),
                ("iqn", "indexer.q_norm.weight"),
                ("ikn", "indexer.k_norm.weight"),
            ] {
                v.push((format!("{k}.{n}"), Job::F32(b(l, t))));
            }
        }
        v.push((
            format!("{k}.router"),
            Job::Bf16Stack {
                names: vec![b(l, "ffn_gate_inp.weight"), b(l, "ffn_gate_inp_shexp.weight")],
            },
        ));
        v.push((
            format!("{k}.sh_gu"),
            Job::Q4Stack {
                names: vec![b(l, "ffn_gate_shexp.weight"), b(l, "ffn_up_shexp.weight")],
                rows: 2 * FF,
                class: "sh",
                segs: vec![(b(l, "ffn_gate_shexp.weight"), 0), (b(l, "ffn_up_shexp.weight"), FF)],
            },
        ));
        v.push((
            format!("{k}.sh_down"),
            Job::Q4Stack {
                names: vec![b(l, "ffn_down_shexp.weight")],
                rows: HIDDEN,
                class: "sh",
                segs: vec![(b(l, "ffn_down_shexp.weight"), 0)],
            },
        ));
    }
    if let Some(l) = ple_layer {
        let key = b(l, "ple_key.weight");
        if g.info(&key)?.ty == GgmlType::Q2_0 {
            v.push(("ple.key".into(), Job::Q2(key)));
        } else {
            v.push((
                "ple.key".into(),
                Job::Q4Stack {
                    names: vec![key],
                    rows: HC * HIDDEN,
                    class: "ple",
                    segs: vec![],
                },
            ));
        }
        v.push((
            "ple.value".into(),
            Job::Bf16Stack {
                names: vec![b(l, "ple_value.weight")],
            },
        ));
        for (n, t) in [
            ("nk", "ple_norm_key.weight"),
            ("nq", "ple_norm_query.weight"),
            ("nc", "ple_norm_conv.weight"),
            ("conv", "ple_conv1d.weight"),
        ] {
            v.push((format!("ple.{n}"), Job::F32(b(l, t))));
        }
    }
    hc(&mut v, "out_hc", "output_hc_", false);
    v.push((
        "head".into(),
        Job::Q4Stack {
            names: vec!["output.weight".into()],
            rows: g.info("output.weight")?.n_rows(),
            class: "head",
            segs: vec![("output.weight".to_string(), 0)],
        },
    ));
    Ok(v)
}

/// Which requantized classes (`w_in`, `w_out`, `sh`, `ple`, `head`, or `all`) are kept bf16
/// instead of Q4X: `TANG_FLASH_BF16=w_in,sh`. A precision experiment knob (2-3.5× the bytes).
pub fn bf16_classes() -> Vec<String> {
    std::env::var("TANG_FLASH_BF16")
        .map(|v| v.split(',').filter(|s| !s.is_empty()).map(String::from).collect())
        .unwrap_or_default()
}

/// The dense pack's file tag for the current precision choice.
pub fn dense_tag() -> String {
    let mut c = bf16_classes();
    c.sort();
    let mut n = native_classes();
    n.sort();
    let mut tag = if c.is_empty() {
        "q4x".to_string()
    } else {
        format!("bf16-{}", c.join("-"))
    };
    if !n.is_empty() {
        tag += &format!(".native-{}", n.join("-"));
    }
    tag
}

/// Classes kept as native GGUF types (`fe_gemv`, lossless): `TANG_FLASH_NATIVE=w_in,w_out,sh,head`
/// (the default; `none` for Q4X everywhere).
pub fn native_classes() -> Vec<String> {
    let v = std::env::var("TANG_FLASH_NATIVE").unwrap_or_else(|_| "w_in,w_out,sh,head".into());
    v.split(',').filter(|s| !s.is_empty() && *s != "none").map(String::from).collect()
}

fn ggml_id(t: GgmlType) -> Result<u64> {
    use GgmlType::*;
    Ok(match t {
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
        other => bail!("no native GEMV for {other:?}"),
    })
}

type Built = (Fmt, usize, usize, Vec<u8>, Vec<[u64; 5]>);

fn run_job(g: &Gguf, job: &Job) -> Result<Built> {
    if let Job::Q4Stack {
        segs, class, rows, ..
    } = job
    {
        if native_classes().iter().any(|c| c == class) && !segs.is_empty() {
            let mut out: Vec<u8> = Vec::new();
            let mut meta = Vec::new();
            let mut k = 0;
            let mut total = 0;
            for (name, off) in segs {
                let t = g.info(name)?;
                let (tn, tk) = nk(t);
                ensure!(k == 0 || k == tk, "{name}: width {tk} vs {k}");
                ensure!(tk % 256 == 0 || t.ty.geometry().is_some_and(|(b, _)| b <= 64), "{name}: {tk} wide");
                k = tk;
                total += tn;
                while out.len() % 16 != 0 {
                    out.push(0);
                }
                meta.push([ggml_id(t.ty)?, tn as u64, t.row_bytes()? as u64, out.len() as u64, *off as u64]);
                out.extend_from_slice(g.bytes(t));
            }
            ensure!(total <= *rows, "{segs:?}: more rows than {rows}");
            return Ok((Fmt::Native, total, k, out, meta));
                }
    }
    let (f, n, k, b) = run_job_plain(g, job)?;
    Ok((f, n, k, b, Vec::new()))
}

fn run_job_plain(g: &Gguf, job: &Job) -> Result<(Fmt, usize, usize, Vec<u8>)> {
    Ok(match job {
        Job::F32(n) => {
            let t = g.info(n)?;
            let v = g.dequantize(t)?;
            (Fmt::F32, 1, v.len(), f32_bytes(&v))
        }
        Job::Q2(n) => {
            let t = g.info(n)?;
            let (n, k) = nk(t);
            (Fmt::Q2Raw, n, k, g.bytes(t).to_vec())
        }
        Job::Bf16Stack { names } => {
            let mut out = Vec::new();
            let (mut n, mut k) = (0, 0);
            for name in names {
                let t = g.info(name)?;
                let (tn, tk) = nk(t);
                ensure!(k == 0 || k == tk, "{name}: width {tk} vs {k}");
                k = tk;
                n += tn;
                out.extend(bf16_of(g, t)?);
            }
            (Fmt::Bf16, n, k, out)
        }
        Job::HcUp(name) => {
            let t = g.info(name)?;
            let bits: Vec<u16> = bf16_of(g, t)?
                .chunks(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            let r = fl::hc_up_repack(&bits);
            let (n, k) = nk(t);
            (Fmt::Bf16, n, k, r.iter().flat_map(|v| v.to_le_bytes()).collect())
        }
        Job::Q4Stack { names, rows, class, .. } => {
            let mut w = Vec::new();
            let mut k = 0;
            for name in names {
                let t = g.info(name)?;
                let (_, tk) = nk(t);
                ensure!(k == 0 || k == tk, "{name}: width {tk} vs {k}");
                k = tk;
                w.extend(g.dequantize(t)?);
            }
            ensure!(w.len() <= rows * k, "{names:?}: more rows than {rows}");
            w.resize(rows * k, 0.0);
            if bf16_classes().iter().any(|c| c == class || c == "all") {
                let b = w.iter().flat_map(|&x| bf16_bits(x).to_le_bytes()).collect();
                (Fmt::Bf16, *rows, k, b)
            } else {
                (Fmt::Q4x, *rows, k, q4x_pack(&w, *rows, k))
            }
        }
    })
}

/// Build (or find) the dense pack; returns the index and the data file path.
pub fn dense(g: &Gguf, dir: &Path, n_layer: usize, is_rec: &[bool], ple_layer: Option<usize>) -> Result<(Vec<Entry>, PathBuf)> {
    let tag = dense_tag();
    let (data, index) = (dir.join(format!("dense.{tag}.bin")), dir.join(format!("dense.{tag}.json")));
    if index.exists() && data.exists() {
        let e: Vec<Entry> = serde_json::from_slice(&std::fs::read(&index)?)?;
        return Ok((e, data));
    }
    std::fs::create_dir_all(dir)?;
    let t0 = std::time::Instant::now();
    let js = jobs(g, n_layer, is_rec, ple_layer)?;
    let tmp = dir.join(format!("dense.{tag}.bin.tmp"));
    let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
    let mut entries = Vec::new();
    let mut off = 0u64;
    // A few jobs at a time in parallel (each is itself parallel over rows), written in order.
    for chunk in js.chunks(8) {
        let built: Vec<Result<Built>> =
            chunk.par_iter().map(|(_, j)| run_job(g, j)).collect();
        for ((name, _), b) in chunk.iter().zip(built) {
            let (fmt, n, k, bytes, segs) = b.with_context(|| format!("packing {name}"))?;
            f.write_all(&bytes)?;
            entries.push(Entry {
                name: name.clone(),
                fmt,
                n,
                k,
                offset: off,
                len: bytes.len() as u64,
                segs,
            });
            off += bytes.len() as u64;
        }
    }
    f.flush()?;
    drop(f);
    std::fs::rename(&tmp, &data)?;
    std::fs::write(&index, serde_json::to_vec_pretty(&entries)?)?;
    eprintln!(
        "flash: packed {} dense tensors ({:.2} GB) in {:.1} s -> {}",
        entries.len(),
        off as f64 / 1e9,
        t0.elapsed().as_secs_f64(),
        data.display()
    );
    Ok((entries, data))
}

// ---- experts ----

/// Drop `len` bytes at `off` of `f` from the page cache.
pub fn drop_cache(f: &std::fs::File, off: u64, len: u64) {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::io::AsRawFd;
        unsafe {
            libc::posix_fadvise(f.as_raw_fd(), off as i64, len as i64, libc::POSIX_FADV_DONTNEED);
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = (f, off, len);
}

/// Read `buf.len()` bytes at `off`.
pub fn pread(f: &std::fs::File, off: u64, buf: &mut [u8]) -> Result<()> {
    use std::os::unix::fs::FileExt;
    f.read_exact_at(buf, off)?;
    Ok(())
}

/// Build `experts.bin` (all `n_layer × 512` blobs in key order) if it isn't there. Reads each
/// layer's three fused tensors with `pread` and drops them from the page cache after.
pub fn experts(g: &Gguf, dir: &Path, n_layer: usize) -> Result<PathBuf> {
    let path = dir.join("experts.bin");
    let want = (n_layer * EXPERTS * ExpertBlob::BYTES) as u64;
    if path.exists() && std::fs::metadata(&path)?.len() == want {
        return Ok(path);
    }
    std::fs::create_dir_all(dir)?;
    let t0 = std::time::Instant::now();
    let tmp = dir.join("experts.bin.tmp");
    let mut out = std::fs::File::create(&tmp)?;
    let shards: Vec<std::fs::File> = g
        .shard_paths()
        .iter()
        .map(std::fs::File::open)
        .collect::<std::io::Result<_>>()?;
    for l in 0..n_layer {
        let mut raw: Vec<Vec<u8>> = Vec::new();
        for s in ["gate", "up", "down"] {
            let t = g.info(&format!("blk.{l}.ffn_{s}_exps.weight"))?;
            if t.ty != GgmlType::Q2_0 {
                bail!("{}: {:?} experts (only Q2_0 is supported)", t.name, t.ty);
            }
            let mut b = vec![0u8; t.nbytes as usize];
            pread(&shards[t.shard], t.offset, &mut b)?;
            drop_cache(&shards[t.shard], t.offset, t.nbytes);
            raw.push(b);
        }
        let per_gu = fl::q2_bytes(FF, HIDDEN);
        let per_d = fl::q2_bytes(HIDDEN, FF);
        let blobs: Vec<Vec<u8>> = (0..EXPERTS)
            .into_par_iter()
            .map(|e| {
                ExpertBlob::from_gguf(
                    &raw[0][e * per_gu..(e + 1) * per_gu],
                    &raw[1][e * per_gu..(e + 1) * per_gu],
                    &raw[2][e * per_d..(e + 1) * per_d],
                )
            })
            .collect();
        for b in &blobs {
            out.write_all(b)?;
        }
        out.flush()?;
        let done = ((l + 1) * EXPERTS * ExpertBlob::BYTES) as u64;
        drop_cache(&out, 0, done);
        if l % 8 == 7 {
            eprintln!(
                "flash: repacked experts of {} layers ({:.0} s)",
                l + 1,
                t0.elapsed().as_secs_f64()
            );
        }
    }
    out.sync_all()?;
    drop_cache(&out, 0, want);
    drop(out);
    std::fs::rename(&tmp, &path)?;
    eprintln!(
        "flash: repacked {} experts ({:.1} GB) in {:.1} s -> {}",
        n_layer * EXPERTS,
        want as f64 / 1e9,
        t0.elapsed().as_secs_f64(),
        path.display()
    );
    Ok(path)
}

/// Read an index entry's bytes.
pub fn read_entry(f: &std::fs::File, e: &Entry) -> Result<Vec<u8>> {
    let mut b = vec![0u8; e.len as usize];
    pread(f, e.offset, &mut b)?;
    Ok(b)
}

pub fn by_name(entries: &[Entry]) -> BTreeMap<&str, &Entry> {
    entries.iter().map(|e| (e.name.as_str(), e)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4x_pack_roundtrip_and_ls_beats_minmax() {
        let (n, k) = (3, 128);
        let w: Vec<f32> = (0..n * k)
            .map(|i| ((i * 7919 % 1013) as f32 / 1013.0 - 0.5).powi(3) * 4.0)
            .collect();
        let b = q4x_pack(&w, n, k);
        assert_eq!(b.len(), n * k / 2 + 4 * n * k / 64);
        let d = q4x_unpack(&b, n, k);
        let e_ls: f64 = w.iter().zip(&d).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
        // min/max RTN for comparison
        let (p, s, bi) = crate::weights::quantize_q4(&w, 64);
        let mm = q4x_unpack(&fl::q4x_repack(&p, &s, &bi, n, k), n, k);
        let e_mm: f64 = w.iter().zip(&mm).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
        assert!(e_ls <= e_mm, "least squares {e_ls} vs min/max {e_mm}");
    }
}
