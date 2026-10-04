//! What happens to Flash-Next's dense weights when the fast engine requantizes them, and to its
//! activations under the int8 contract — simulated on the f32 reference so the cost can be
//! measured as KL before any kernel exists.
//!
//! The fast GPU kernels (`tang_compute::flash`) take a dense weight as one of three formats:
//! bf16 (fp32 activations), native Q2_0, or **Q4X**: tang-Q4 values (`weights::quantize_q4`,
//! MLX affine, group 64, `w = scale · q + bias` with bf16 scale and bias, 4.5 bits a weight)
//! against int8 activations. ISTA's dense tensors come in ten types, so everything that is not
//! already bf16 or Q2_0 has to be requantized at load. A [`DensePolicy`] says to what:
//!
//! - `f32` — no requant: the GGUF's own type, dequantized (the truth).
//! - `q4x` — every such tensor to Q4X with `quantize_q4`'s min/max rounding, exactly as the engine.
//! - `q4x-search` — Q4X, but each group's (scale, bias) chosen by a clipping grid plus least-squares
//!   refits ([`quantize_q4_search`]); same format and bytes, better values.
//! - `q4x-q8hi`, `q4x-search-q8hi` — as above, but tensors whose GGUF type has ≥ 5 bits (Q5_K,
//!   Q6_K, Q5_0, Q8_0) go to Q8_0 (int8 codes, f16 scale per 32: 8.5 bits) instead.
//! - `q8` — everything to Q8_0; or a rule list (see [`DensePolicy`]), where `native` means a
//!   kernel for the GGUF's own type.
//!
//! The token embedding (a row lookup), F32/F16 vectors and the n-gram table are never touched.
//!
//! [`Act::Int8`](super::reference::Act::Int8) is the kernels' activation contract.

use crate::gguf::{f16_to_f32, f32_to_f16, GgmlType, Gguf, TensorInfo};
use anyhow::{bail, Result};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::fmt::Write as _;

/// The formats a dense weight can end up in on the GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Format {
    /// Left as the GGUF has it (bf16, f32, Q2_0, or the truth's exact dequant).
    Keep,
    /// tang-Q4, min/max rounding.
    Q4x,
    /// tang-Q4, searched rounding.
    Q4xSearch,
    /// Q8_0: int8 codes, f16 scale per 32.
    Q8,
}

/// Which dense tensors go to which [`Format`]: an ordered rule list, first match wins, for the
/// tensors [`requantizable`] admits (everything else is always [`Format::Keep`]).
///
/// `--dense-as` takes a preset (`f32`, `q4x`, `q4x-search`, `q4x-q8hi`, `q4x-search-q8hi`,
/// `q8`) or rules `match=format,...` where `match` is a GGUF type name (`Q3_K`), `*`, or a
/// tensor-name substring (`output.weight`, `attn_qkv`), and `format` is `native` (keep the GGUF's
/// own type: a native kernel), `q4x`, `q4xs` (searched Q4X) or `q8`. E.g.
/// `output.weight=q8,Q3_K=native,*=q4xs`.
#[derive(Clone, Debug, PartialEq)]
pub struct DensePolicy {
    pub name: String,
    pub rules: Vec<(String, Format)>,
}

impl DensePolicy {
    pub fn f32() -> Self {
        Self {
            name: "f32".into(),
            rules: vec![],
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        let hi = |f: Format, lo: Format| -> Vec<(String, Format)> {
            let mut v: Vec<(String, Format)> = ["Q5_K", "Q6_K", "Q5_0", "Q5_1", "Q8_0"]
                .iter()
                .map(|t| (t.to_string(), f))
                .collect();
            v.push(("*".into(), lo));
            v
        };
        let rules = match s {
            "f32" => vec![],
            "q4x" => vec![("*".into(), Format::Q4x)],
            "q4x-search" => vec![("*".into(), Format::Q4xSearch)],
            "q8" => vec![("*".into(), Format::Q8)],
            "q4x-q8hi" => hi(Format::Q8, Format::Q4x),
            "q4x-search-q8hi" => hi(Format::Q8, Format::Q4xSearch),
            _ => {
                let mut v = Vec::new();
                for part in s.split(',') {
                    let Some((m, f)) = part.split_once('=') else {
                        bail!("--dense-as {s}: {part:?} is not match=format");
                    };
                    let f = match f {
                        "native" | "keep" => Format::Keep,
                        "q4x" => Format::Q4x,
                        "q4xs" => Format::Q4xSearch,
                        "q8" => Format::Q8,
                        _ => bail!("--dense-as: unknown format {f:?} (native, q4x, q4xs, q8)"),
                    };
                    v.push((m.to_string(), f));
                }
                v
            }
        };
        Ok(Self {
            name: s.into(),
            rules,
        })
    }

    pub fn presets() -> Vec<DensePolicy> {
        [
            "f32",
            "q4x",
            "q4x-search",
            "q4x-q8hi",
            "q4x-search-q8hi",
            "q8",
        ]
        .iter()
        .map(|p| Self::parse(p).expect("preset"))
        .collect()
    }

    /// The format a dense matrix `name` of GGUF type `ty` ends up in.
    pub fn format(&self, name: &str, ty: GgmlType) -> Format {
        if !requantizable(name, ty) {
            return Format::Keep;
        }
        for (m, f) in &self.rules {
            if m == "*"
                || *m == ty.name()
                || (m.contains(|c: char| c.is_ascii_lowercase()) && name.contains(m.as_str()))
            {
                return *f;
            }
        }
        Format::Keep
    }
}

/// A 2-D dense weight the engine would requantize: not bf16/f32/f16, not Q2_0, not an expert
/// tensor, not the embedding or the n-gram table.
pub fn requantizable(name: &str, ty: GgmlType) -> bool {
    !matches!(
        ty,
        GgmlType::Bf16 | GgmlType::F32 | GgmlType::F16 | GgmlType::Q2_0
    ) && !name.contains("_exps")
        && name != "token_embd.weight"
        && !name.starts_with("per_layer_token_embd")
}

/// Replace `w` (rows of `k`) by its round trip through `f`.
pub fn apply(f: Format, w: &mut [f32], k: usize) {
    match f {
        Format::Keep => {}
        Format::Q4x => {
            let (packed, scales, biases) = crate::weights::quantize_q4(w, 64);
            for (i, v) in w.iter_mut().enumerate() {
                let q = (packed[i / 8] >> (4 * (i % 8))) & 0xf;
                let (s, b) = (bf16(scales[i / 64]), bf16(biases[i / 64]));
                *v = s * q as f32 + b;
            }
        }
        Format::Q4xSearch => {
            debug_assert!(k.is_multiple_of(64));
            w.par_chunks_mut(64).for_each(quantize_q4_search);
        }
        Format::Q8 => {
            w.par_chunks_mut(32).for_each(|b| {
                let amax = b.iter().fold(0f32, |m, v| m.max(v.abs()));
                let d = amax / 127.0;
                let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                let dh = f16_to_f32(f32_to_f16(d));
                for v in b.iter_mut() {
                    *v = (*v * id).round() * dh; // ggml's quantize_row_q8_0_ref: roundf
                }
            });
        }
    }
}

fn bf16(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn to_bf16_f32(x: f32) -> f32 {
    let u = x.to_bits();
    f32::from_bits((u + 0x7fff + ((u >> 16) & 1)) & 0xffff_0000)
}

/// Error and codes of one 64-group for a (scale, bias) already rounded to bf16.
fn q4_group(v: &[f32], s: f32, b: f32, q: &mut [u8]) -> f64 {
    let mut err = 0f64;
    for (x, qq) in v.iter().zip(q.iter_mut()) {
        let c = ((x - b) / s).round().clamp(0.0, 15.0);
        *qq = c as u8;
        let e = (x - (s * c + b)) as f64;
        err += e * e;
    }
    err
}

/// Q4X values for one 64-group, in place, with a better (scale, bias) than min/max: try the
/// range clipped by 0..20% at each end (9 × 9 grid), keep the lowest squared error, then refit
/// scale and bias by least squares on the chosen codes twice. Everything is evaluated with the
/// bf16-rounded scale and bias the format stores, so the result is a valid Q4X group.
pub fn quantize_q4_search(v: &mut [f32]) {
    let lo = v.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let r = hi - lo;
    let mut q = vec![0u8; v.len()];
    let mut best = (f64::INFINITY, 0f32, 0f32);
    let pick = |s: f32, b: f32, q: &mut [u8], best: &mut (f64, f32, f32)| {
        let s = to_bf16_f32(s.max(1e-8));
        let b = to_bf16_f32(b);
        let e = q4_group(v, s, b, q);
        if e < best.0 {
            *best = (e, s, b);
        }
    };
    for i in 0..9 {
        for j in 0..9 {
            let l = lo + r * 0.025 * i as f32;
            let h = hi - r * 0.025 * j as f32;
            if h <= l {
                continue;
            }
            pick((h - l) / 15.0, l, &mut q, &mut best);
        }
    }
    // least-squares refits of (s, b) given the codes
    for _ in 0..2 {
        q4_group(v, best.1, best.2, &mut q);
        let n = v.len() as f64;
        let (mut sq, mut sqq, mut sx, mut sqx) = (0f64, 0f64, 0f64, 0f64);
        for (&x, &c) in v.iter().zip(&q) {
            let c = c as f64;
            sq += c;
            sqq += c * c;
            sx += x as f64;
            sqx += c * x as f64;
        }
        let det = n * sqq - sq * sq;
        if det.abs() < 1e-12 {
            break;
        }
        let s = ((n * sqx - sq * sx) / det) as f32;
        let b = ((sqq * sx - sq * sqx) / det) as f32;
        if s > 0.0 {
            pick(s, b, &mut q, &mut best);
        }
    }
    let (_, s, b) = best;
    q4_group(v, s, b, &mut q);
    for (x, &c) in v.iter_mut().zip(&q) {
        *x = s * c as f32 + b;
    }
}

/// Bytes a weight of `n` elements occupies in format `f` (`ty` for [`Format::Keep`]).
pub fn bytes(f: Format, ty: GgmlType, n: usize) -> u64 {
    match f {
        Format::Keep => ty.bytes_for(n).unwrap_or(0) as u64,
        Format::Q4x | Format::Q4xSearch => (n / 2 + 4 * n / 64) as u64,
        Format::Q8 => (n / 32 * 34) as u64,
    }
}

/// Whether a decode step reads this tensor in full (everything except the routed experts, the
/// embedding and the n-gram table, which are read a few rows at a time).
fn read_per_token(t: &TensorInfo) -> bool {
    !t.name.contains("_exps")
        && t.name != "token_embd.weight"
        && !t.name.starts_with("per_layer_token_embd")
}

/// `tang-llm flash-requant <gguf> [--no-errors] [policy...]`: for each source type, the relative
/// squared error ‖W − Q(W)‖² / ‖W‖² of each format, and the dense bytes a decode step reads under
/// the presets and any extra policies.
pub fn report(g: &Gguf, extra: &[DensePolicy], errors: bool) -> Result<String> {
    let mut s = String::new();
    // per source type: (tensors, elements, ||w||^2, err per format)
    #[derive(Default)]
    struct Acc {
        tensors: usize,
        elems: u64,
        bytes: u64,
        norm: f64,
        err: BTreeMap<Format, f64>,
    }
    let mut by_type: BTreeMap<String, Acc> = BTreeMap::new();
    for t in &g.tensors {
        if !errors || t.dims.len() != 2 || !requantizable(&t.name, t.ty) {
            continue;
        }
        let w = g.dequantize(t)?;
        let k = t.dims[0] as usize;
        let a = by_type.entry(t.ty.name()).or_default();
        a.tensors += 1;
        a.elems += w.len() as u64;
        a.bytes += t.nbytes;
        a.norm += w.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>();
        for f in [Format::Q4x, Format::Q4xSearch, Format::Q8] {
            let mut q = w.clone();
            apply(f, &mut q, k);
            let e: f64 = w
                .iter()
                .zip(&q)
                .map(|(&a, &b)| ((a - b) as f64).powi(2))
                .sum();
            *a.err.entry(f).or_default() += e;
        }
    }
    if errors {
        writeln!(
            s,
            "## requant error by source type (relative squared error, sum over tensors)"
        )?;
        writeln!(
            s,
            "type | tensors | params | GGUF bytes | q4x | q4x-search | q8_0"
        )?;
        for (ty, a) in &by_type {
            writeln!(
                s,
                "{ty} | {} | {} | {} | {:.3e} | {:.3e} | {:.3e}",
                a.tensors,
                a.elems,
                a.bytes,
                a.err[&Format::Q4x] / a.norm,
                a.err[&Format::Q4xSearch] / a.norm,
                a.err[&Format::Q8] / a.norm
            )?;
        }
    }
    writeln!(
        s,
        "\n## bytes a decode step reads in full (experts, embedding row, table rows excluded)"
    )?;
    writeln!(s, "policy | requantized | kept as in the GGUF | total")?;
    for p in DensePolicy::presets().iter().chain(extra) {
        let (mut rq, mut kept) = (0u64, 0u64);
        for t in g.tensors.iter().filter(|t| read_per_token(t)) {
            let f = if t.dims.len() == 2 {
                p.format(&t.name, t.ty)
            } else {
                Format::Keep
            };
            let b = bytes(f, t.ty, t.n_elements() as usize);
            if f == Format::Keep {
                kept += b;
            } else {
                rq += b;
            }
        }
        writeln!(s, "{} | {rq} | {kept} | {}", p.name, rq + kept)?;
    }
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_never_worse_than_minmax() {
        let mut seed = 1u64;
        for _ in 0..200 {
            let v: Vec<f32> = (0..64)
                .map(|_| {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let u = (seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                    u * u * u * 0.1 // heavy-ish tails, like real weights
                })
                .collect();
            let mut a = v.clone();
            apply(Format::Q4x, &mut a, 64);
            let mut b = v.clone();
            apply(Format::Q4xSearch, &mut b, 64);
            let ea: f64 = v
                .iter()
                .zip(&a)
                .map(|(x, y)| ((x - y) as f64).powi(2))
                .sum();
            let eb: f64 = v
                .iter()
                .zip(&b)
                .map(|(x, y)| ((x - y) as f64).powi(2))
                .sum();
            assert!(eb <= ea * (1.0 + 1e-6), "{eb} > {ea}");
            // searched values are still a 16-level affine grid
            let mut lv: Vec<f32> = b.clone();
            lv.sort_by(f32::total_cmp);
            lv.dedup();
            assert!(lv.len() <= 16);
        }
    }

    #[test]
    fn q8_of_q8_is_identity() {
        let mut w: Vec<f32> = (0..64).map(|i| (i as f32 - 31.0) * 0.01).collect();
        apply(Format::Q8, &mut w, 64);
        let once = w.clone();
        apply(Format::Q8, &mut w, 64);
        assert_eq!(w, once);
    }

    #[test]
    fn policy_formats() {
        let p = DensePolicy::parse("q4x-q8hi").unwrap();
        assert_eq!(p.format("blk.0.attn_qkv.weight", GgmlType::Q6K), Format::Q8);
        assert_eq!(
            p.format("blk.0.attn_qkv.weight", GgmlType::Iq4Xs),
            Format::Q4x
        );
        assert_eq!(
            p.format("blk.0.attn_qkv.weight", GgmlType::Q2_0),
            Format::Keep
        );
        assert_eq!(
            p.format("blk.0.hc_attn_up.weight", GgmlType::Bf16),
            Format::Keep
        );
        assert_eq!(p.format("token_embd.weight", GgmlType::Q3K), Format::Keep);
        assert_eq!(
            DensePolicy::f32().format("output.weight", GgmlType::Q5K),
            Format::Keep
        );
        let r = DensePolicy::parse("output.weight=q8,Q3_K=native,*=q4xs").unwrap();
        assert_eq!(r.format("output.weight", GgmlType::Q5K), Format::Q8);
        assert_eq!(r.format("blk.3.attn_q.weight", GgmlType::Q3K), Format::Keep);
        assert_eq!(
            r.format("blk.3.attn_q.weight", GgmlType::Iq4Xs),
            Format::Q4xSearch
        );
    }
}
