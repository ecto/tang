//! Memory-mapped safetensors checkpoints, single-file or sharded.

use anyhow::{Context, Result, bail};
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
    fn f16_widening() {
        assert_eq!(super::f16_to_f32(0x3c00), 1.0);
        assert_eq!(super::f16_to_f32(0xc000), -2.0);
        assert_eq!(super::f16_to_f32(0x0001), 2f32.powi(-24));
    }
}
