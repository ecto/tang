//! The MTP draft layer on the GPU (math: `flash::mtp`, the truth track's reference).
//!
//! Cell `i` pairs the main model's final 4-stream residual at position `i` with the token at
//! `i + 1` (rope position `i`) and predicts the token at `i + 2`. After every window the engine
//! runs one teacher-forced cell per kept position (so the MTP K/V cache covers the sequence),
//! whose last cell drafts `d1`; two chain cells (input: the previous cell's output residual and
//! draft) give `d2`, `d3`. All three are in one captured graph per cell count; the host keeps
//! the chain while each draft's probability is ≥ 0.5.
//!
//! Weights: the MTP GGUF's `blk.48` (Q8_0), the main model's embedding (Q3_K, in VRAM for the
//! chain's draft tokens) and head (shared, as unsloth's `shared-` heads do). The 512 routed
//! experts are requantized Q8_0 → Q4_0 at load (1.42 GB instead of 2.67 GB of VRAM;
//! `TANG_FLASH_MTP_Q8=1` keeps Q8_0): a drafter's precision changes acceptance, never output.

use crate::gguf::{dequantize, GgmlType, Gguf, TensorInfo};
use anyhow::{ensure, Context, Result};
use rayon::prelude::*;
use std::path::Path;
use tang_compute::cuda::CudaBuffer as B;
use tang_compute::flash::shape::*;
use tang_compute::{ComputeDevice, CudaComputeDevice};

/// Rows of the MTP attention projection: `[q|gate × 24 | k 512 | v 512]`.
pub const MTP_PROJ: usize = QSA_HEADS * 2 * QSA_D + 2 * QSA_KV * QSA_D;
/// Most draft steps per window (the teacher pass's last cell, then chain cells).
pub const MAX_STEPS: usize = 6;

/// Draft steps per window: `TANG_FLASH_MTP_STEPS` (default 3).
pub fn steps() -> usize {
    std::env::var("TANG_FLASH_MTP_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(3).clamp(1, MAX_STEPS)
}

/// Raw GGUF bytes of a 2-D tensor, its ggml id and row bytes.
pub struct Raw {
    pub bytes: Vec<u8>,
    pub ty: u64,
    pub rows: usize,
    pub k: usize,
    pub rb: usize,
}

pub fn ggml_id(t: GgmlType) -> Result<u64> {
    use GgmlType::*;
    Ok(match t {
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
        o => anyhow::bail!("no native GEMV for {o:?}"),
    })
}

pub fn raw(g: &Gguf, t: &TensorInfo) -> Result<Raw> {
    Ok(Raw {
        bytes: g.bytes(t).to_vec(),
        ty: ggml_id(t.ty)?,
        rows: t.n_rows(),
        k: t.row_len(),
        rb: t.row_bytes()?,
    })
}

/// A tensor dequantized and rounded to bf16 bits.
pub fn bf16(g: &Gguf, t: &TensorInfo) -> Result<Vec<u16>> {
    Ok(g.dequantize(t)?.iter().map(|&x| super::pack::bf16_bits(x)).collect())
}

/// Q8_0 → Q4_0 (ggml's quantize_row_q4_0: d = max / −8 of the block's largest-magnitude value).
pub fn q8_to_q4_0(src: &[u8]) -> Result<Vec<u8>> {
    ensure!(src.len() % 34 == 0, "Q8_0 size");
    let nb = src.len() / 34;
    let mut out = vec![0u8; nb * 18];
    out.par_chunks_mut(18 * 4096).enumerate().try_for_each(|(c, o)| -> Result<()> {
        let mut f = [0f32; 32];
        for (i, dst) in o.chunks_mut(18).enumerate() {
            let b = c * 4096 + i;
            dequantize(GgmlType::Q8_0, &src[b * 34..b * 34 + 34], &mut f)?;
            let mut max = 0f32;
            for &x in &f {
                if x.abs() > max.abs() {
                    max = x;
                }
            }
            let d = max / -8.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            dst[..2].copy_from_slice(&tang_compute::flash::f32_to_f16(d).to_le_bytes());
            for j in 0..16 {
                let q0 = ((f[j] * id + 8.5) as i32).clamp(0, 15) as u8;
                let q1 = ((f[j + 16] * id + 8.5) as i32).clamp(0, 15) as u8;
                dst[2 + j] = q0 | (q1 << 4);
            }
        }
        Ok(())
    })?;
    Ok(out)
}

pub fn upload_padded(dev: &CudaComputeDevice, b: &[u8]) -> B {
    let mut v = b.to_vec();
    v.extend_from_slice(&[0u8; 16]);
    dev.upload_bytes(&v)
}

pub fn open(path: &Path) -> Result<(Gguf, usize)> {
    let g = Gguf::open(path)?;
    let layer = g.meta_u64("qwen4exp.block_count").context("MTP block count")? as usize - 1;
    ensure!(
        g.get(&format!("blk.{layer}.nextn.eh_proj.weight")).is_some(),
        "{}: not an MTP file",
        path.display()
    );
    Ok((g, layer))
}
