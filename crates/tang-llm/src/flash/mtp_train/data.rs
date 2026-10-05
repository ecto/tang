use crate::gguf::f16_to_f32;
use anyhow::{ensure, Context, Result};
use memmap2::Mmap;
use std::{
    fs::File,
    path::{Path, PathBuf},
};

pub struct Sequence {
    pub path: PathBuf,
    pub ids: Vec<[u32; 3]>,
    h: Mmap,
    pub width: usize,
}
pub struct Batch {
    pub h: Vec<f32>,
    pub tokens: Vec<u32>,
    pub pos: Vec<usize>,
}
impl Sequence {
    pub fn open(path: &Path, width: usize) -> Result<Self> {
        ensure!(
            path.join("done").exists(),
            "{}: incomplete prompt",
            path.display()
        );
        let bytes = std::fs::read(path.join("ids.u32"))?;
        ensure!(bytes.len() % 12 == 0, "{}: partial ids row", path.display());
        let mut ids: Vec<[u32; 3]> = bytes
            .chunks_exact(12)
            .map(|b| {
                std::array::from_fn(|i| u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap()))
            })
            .collect();
        let f = File::open(path.join("h.f16"))?;
        let h = unsafe { Mmap::map(&f)? };
        ensure!(
            h.len() == ids.len() * width * 2,
            "{}: residual/id length mismatch",
            path.display()
        );
        for r in ids.windows(2) {
            ensure!(
                r[0][0].checked_add(1) == Some(r[1][0]) && r[0][2] == r[1][1],
                "{}: non-contiguous dump",
                path.display()
            );
        }
        // Retain the cell whose next token is EOS, but never include cells after EOS.
        // The EOS token can be a label. It cannot seed another recursive training cell.
        if let Some(i) = ids.iter().position(|r| [248046, 248044].contains(&r[1])) {
            ids.truncate(i);
        } else if let Some(i) = ids.iter().position(|r| [248046, 248044].contains(&r[2])) {
            ids.truncate(i + 1);
        }
        Ok(Self {
            path: path.to_path_buf(),
            ids,
            h,
            width,
        })
    }
    pub fn batch(&self, start: usize, len: usize) -> Result<Batch> {
        ensure!(
            len >= 4 && start + len <= self.ids.len(),
            "batch outside sequence"
        );
        let h = self.h[start * self.width * 2..(start + len) * self.width * 2]
            .chunks_exact(2)
            .map(|b| f16_to_f32(u16::from_le_bytes(b.try_into().unwrap())))
            .collect();
        let mut tokens: Vec<_> = self.ids[start..start + len].iter().map(|r| r[1]).collect();
        tokens.push(self.ids[start + len - 1][2]);
        Ok(Batch {
            h,
            tokens,
            pos: self.ids[start..start + len]
                .iter()
                .map(|r| r[0] as usize)
                .collect(),
        })
    }
}
pub fn discover(root: &Path, width: usize) -> Result<Vec<Sequence>> {
    let mut paths: Vec<_> = std::fs::read_dir(root)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.join("done").exists())
        .collect();
    paths.sort();
    paths
        .into_iter()
        .map(|p| Sequence::open(&p, width).with_context(|| p.display().to_string()))
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::f32_to_f16;
    #[test]
    fn shifting_stop_and_alignment() {
        let root = std::env::temp_dir().join(format!("tang-mtp-data-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let rows = [
            [7u32, 10, 11],
            [8, 11, 12],
            [9, 12, 13],
            [10, 13, 248046],
            [11, 248046, 99],
            [12, 99, 100],
        ];
        std::fs::write(
            root.join("ids.u32"),
            rows.iter()
                .flatten()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        std::fs::write(
            root.join("h.f16"),
            (0..12)
                .flat_map(|v| f32_to_f16(v as f32).to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        assert!(Sequence::open(&root, 2).is_err());
        std::fs::write(root.join("done"), b"").unwrap();
        let seq = Sequence::open(&root, 2).unwrap();
        assert_eq!(seq.ids.len(), 4);
        let b = seq.batch(0, 4).unwrap();
        assert_eq!(b.tokens, [10, 11, 12, 13, 248046]);
        assert_eq!(b.pos, [7, 8, 9, 10]);
        assert_eq!(b.h, (0..8).map(|i| i as f32).collect::<Vec<_>>());
        std::fs::write(root.join("h.f16"), b"bad").unwrap();
        assert!(Sequence::open(&root, 2).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}
