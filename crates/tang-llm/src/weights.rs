//! Memory-mapped safetensors checkpoints, single-file or sharded.

use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::path::Path;

pub struct Weights {
    maps: Vec<Mmap>,
    /// tensor name -> index into `maps`
    index: HashMap<String, usize>,
}

impl Weights {
    pub fn open(dir: &Path) -> Result<Self> {
        let mut files: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            bail!("no .safetensors files in {}", dir.display());
        }
        let mut maps = Vec::new();
        let mut index = HashMap::new();
        for (i, f) in files.iter().enumerate() {
            let file = std::fs::File::open(f)?;
            // Safety: checkpoint files aren't modified while we run.
            let map = unsafe { Mmap::map(&file)? };
            let st = SafeTensors::deserialize(&map)
                .with_context(|| format!("reading {}", f.display()))?;
            for name in st.names() {
                index.insert(name.to_string(), i);
            }
            maps.push(map);
        }
        Ok(Self { maps, index })
    }

    pub fn has(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    /// A packed integer tensor (MLX quantized weights).
    pub fn u32(&self, name: &str) -> Result<Vec<u32>> {
        let i = *self
            .index
            .get(name)
            .with_context(|| format!("missing tensor {name}"))?;
        let st = SafeTensors::deserialize(&self.maps[i])?;
        let t = st.tensor(name)?;
        if t.dtype() != Dtype::U32 {
            bail!("{name}: expected U32, got {:?}", t.dtype());
        }
        Ok(t.data()
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect())
    }

    /// A tensor as raw bfloat16 bits (f32/f16 are rounded), for weights kept in bf16 on device.
    pub fn bf16(&self, name: &str) -> Result<Vec<u16>> {
        let i = *self
            .index
            .get(name)
            .with_context(|| format!("missing tensor {name}"))?;
        let st = SafeTensors::deserialize(&self.maps[i])?;
        let t = st.tensor(name)?;
        if t.dtype() == Dtype::BF16 {
            return Ok(t
                .data()
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect());
        }
        Ok(self.f32(name)?.0.into_iter().map(to_bf16).collect())
    }

    /// A tensor as f32 (bf16/f16 are widened), with its shape.
    pub fn f32(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let i = *self
            .index
            .get(name)
            .with_context(|| format!("missing tensor {name}"))?;
        let st = SafeTensors::deserialize(&self.maps[i])?;
        let t = st.tensor(name)?;
        let data = t.data();
        let out = match t.dtype() {
            Dtype::F32 => bytemuck_f32(data),
            Dtype::BF16 => data
                .chunks_exact(2)
                .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
                .collect(),
            Dtype::F16 => data
                .chunks_exact(2)
                .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect(),
            d => bail!("{name}: unsupported dtype {d:?}"),
        };
        Ok((out, t.shape().to_vec()))
    }
}

/// Round-to-nearest 4-bit affine quantization of a row-major matrix with rows of `k`, in
/// MLX's layout: `(packed, scales, biases)` with `w ≈ scale * q + bias` per `group` weights.
pub fn quantize_q4(w: &[f32], group: usize) -> (Vec<u32>, Vec<u16>, Vec<u16>) {
    let groups = w.len() / group;
    let mut packed = vec![0u32; w.len() / 8];
    let (mut scales, mut biases) = (Vec::with_capacity(groups), Vec::with_capacity(groups));
    for g in 0..groups {
        let vals = &w[g * group..(g + 1) * group];
        let lo = vals.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        // Store in bf16 first, then quantize against the stored values.
        let s = from_bf16(to_bf16(((hi - lo) / 15.0).max(1e-8)));
        let b = from_bf16(to_bf16(lo));
        scales.push(to_bf16(s));
        biases.push(to_bf16(b));
        for (j, &v) in vals.iter().enumerate() {
            let q = ((v - b) / s).round().clamp(0.0, 15.0) as u32;
            let i = g * group + j;
            packed[i / 8] |= q << (4 * (i % 8));
        }
    }
    (packed, scales, biases)
}

fn from_bf16(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Round-to-nearest-even f32 -> bf16.
fn to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return ((b >> 16) | 0x40) as u16;
    }
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn bytemuck_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = match exp {
        0 if frac == 0 => sign,
        0 => {
            // Subnormal: renormalize.
            let mut e = 127 - 15 + 1;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((f & 0x3ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | (frac << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    #[test]
    fn q4_round_trip_error_is_within_half_a_step() {
        let w: Vec<f32> = (0..256)
            .map(|i| ((i * 37 % 101) as f32 / 50.0) - 1.0)
            .collect();
        let (p, s, b) = super::quantize_q4(&w, 64);
        for (i, &v) in w.iter().enumerate() {
            let q = (p[i / 8] >> (4 * (i % 8))) & 0xf;
            let (s, b) = (super::from_bf16(s[i / 64]), super::from_bf16(b[i / 64]));
            assert!((s * q as f32 + b - v).abs() <= s * 0.51 + 1e-3, "at {i}");
        }
    }

    #[test]
    fn f16_widening() {
        assert_eq!(super::f16_to_f32(0x3c00), 1.0);
        assert_eq!(super::f16_to_f32(0xc000), -2.0);
        assert_eq!(super::f16_to_f32(0x0001), 2f32.powi(-24));
    }
}
