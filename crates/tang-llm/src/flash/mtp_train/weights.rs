use super::tape::{Id, Tape};
use crate::gguf::{f32_to_f16, GgmlType, Gguf};
use anyhow::{bail, ensure, Result};
use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    io::{Seek, SeekFrom, Write},
    path::Path,
    sync::Arc,
};
use tang_compute::ComputeDevice;
use tang_tensor::{Shape, Tensor};
use tang_train::{ModuleAdam, Optimizer, Parameter};

pub struct Weight {
    pub value: Arc<Vec<f32>>,
    pub rows: usize,
    pub cols: usize,
}
pub struct Weights {
    pub expert_q4: bool,
    pub main: Gguf,
    pub mtp: Gguf,
    pub layer: usize,
    pub vocab: Vec<u32>,
    pub dense: BTreeMap<String, Weight>,
    pub params: BTreeMap<String, Parameter<f32>>,
    pub optimizer: ModuleAdam,
    cache: HashMap<(String, usize), Arc<Vec<f32>>>,
    lru: VecDeque<(String, usize)>,
}
impl Weights {
    pub fn open(main: &Path, mtp: &Path, lr: f64, vocab_lo: usize) -> Result<Self> {
        let main = Gguf::open(main)?;
        let mtp = Gguf::open(mtp)?;
        let layer = mtp.meta_u64("qwen4exp.block_count")? as usize - 1;
        let prefix = format!("blk.{layer}.");
        ensure!(
            mtp.get(&format!("{prefix}nextn.eh_proj.weight")).is_some(),
            "not an MTP file"
        );
        let mut dense = BTreeMap::new();
        let mut params = BTreeMap::new();
        for t in &mtp.tensors {
            if !t.name.starts_with(&prefix)
                || t.name.contains("_exps.")
                || t.name.contains(".indexer.")
            {
                continue;
            }
            let values = mtp.dequantize(t)?;
            ensure!(
                matches!(t.ty, GgmlType::F32 | GgmlType::Bf16 | GgmlType::Q8_0),
                "{}: unsupported trainable export type {:?}",
                t.name,
                t.ty
            );
            let name = t.name[prefix.len()..].to_string();
            params.insert(
                name.clone(),
                Parameter::new(Tensor::new(
                    values.clone(),
                    Shape::from_slice(&[values.len()]),
                )),
            );
            dense.insert(
                name,
                Weight {
                    value: Arc::new(values),
                    rows: t.n_rows(),
                    cols: t.row_len(),
                },
            );
        }
        let nv = main.info("output.weight")?.n_rows();
        ensure!(
            vocab_lo == 0 || vocab_lo <= 248044 || vocab_lo >= nv,
            "partial vocabulary overlaps the special-token range"
        );
        let vocab: Vec<u32> = (0..nv)
            .filter(|&i| vocab_lo == 0 || i < vocab_lo || i >= 248044)
            .map(|i| i as u32)
            .collect();
        let head = main.info("output.weight")?;
        let mut v = Vec::with_capacity(vocab.len() * head.row_len());
        if vocab_lo == 0 || vocab_lo >= nv {
            v = main.dequantize(head)?;
        } else {
            v.extend(main.rows(head, 0, vocab_lo)?);
            if nv > 248044 {
                v.extend(main.rows(head, 248044, nv - 248044)?);
            }
        }
        dense.insert(
            "output.weight".into(),
            Weight {
                value: Arc::new(v),
                rows: vocab.len(),
                cols: head.row_len(),
            },
        );
        Ok(Self {
            expert_q4: true,
            main,
            mtp,
            layer,
            vocab,
            dense,
            params,
            optimizer: ModuleAdam::new(lr),
            cache: HashMap::new(),
            lru: VecDeque::new(),
        })
    }
    pub fn leafs<D: ComputeDevice>(&self, tape: &mut Tape<D>) -> BTreeMap<String, Id> {
        self.dense
            .iter()
            .map(|(n, w)| {
                (
                    n.clone(),
                    tape.leaf(w.value.clone(), self.params.contains_key(n)),
                )
            })
            .collect()
    }
    pub fn expert(&mut self, name: &str, e: usize) -> Result<Arc<Vec<f32>>> {
        let key = (name.to_string(), e);
        if let Some(v) = self.cache.get(&key) {
            return Ok(v.clone());
        }
        let ti = self.mtp.info(&format!("blk.{}.{name}", self.layer))?;
        // Match the default deployed Q8_0 -> Q4_0 routed expert weights, not a different drafter.
        ensure!(ti.ty == GgmlType::Q8_0, "MTP experts must be Q8_0");
        let width = ti.dims[0] as usize * ti.dims[1] as usize;
        let bytes = width / 32 * 34;
        let raw = &self.mtp.bytes(ti)[e * bytes..(e + 1) * bytes];
        if !self.expert_q4 {
            return Ok(Arc::new(self.mtp.expert(ti, e)?));
        }
        let mut values = Vec::with_capacity(width);
        for block in raw.chunks_exact(34) {
            let mut f = [0.0; 32];
            crate::gguf::dequantize(GgmlType::Q8_0, block, &mut f)?;
            let max = f
                .iter()
                .copied()
                .fold(0.0f32, |a, b| if b.abs() > a.abs() { b } else { a });
            let d = max / -8.0;
            let id = if d == 0.0 { 0.0 } else { 1.0 / d };
            let rd = crate::gguf::f16_to_f32(f32_to_f16(d));
            values.extend(
                f.iter()
                    .map(|x| (((x * id + 8.5) as i32).clamp(0, 15) - 8) as f32 * rd),
            );
        }
        let value = Arc::new(values);
        self.cache.insert(key.clone(), value.clone());
        self.lru.push_back(key);
        // 192 matrices = 64 expert triplets (~1 GB), in addition to tensors held by this step's tape.
        while self.lru.len() > 192 {
            let old = self.lru.pop_front().unwrap();
            self.cache.remove(&old);
        }
        Ok(value)
    }
    pub fn update(
        &mut self,
        leaves: &BTreeMap<String, Id>,
        mut grad: Vec<Option<Vec<f32>>>,
        clip: f64,
    ) -> Result<f64> {
        let mut sq = 0.0f64;
        for (n, p) in &mut self.params {
            let g = grad[leaves[n]]
                .take()
                .unwrap_or_else(|| vec![0.0; p.data.numel()]);
            ensure!(
                g.iter().all(|v| v.is_finite()),
                "nonfinite gradient for {n}"
            );
            sq += g.iter().map(|&v| (v as f64).powi(2)).sum::<f64>();
            p.grad = Some(Tensor::new(g, p.data.shape().clone()));
        }
        let norm = sq.sqrt();
        let scale = (clip / norm.max(1e-30)).min(1.0) as f32;
        for p in self.params.values_mut() {
            for g in p.grad.as_mut().unwrap().data_mut() {
                *g *= scale;
            }
        }
        self.optimizer
            .step(&mut self.params.values_mut().collect::<Vec<_>>());
        for (n, p) in &mut self.params {
            ensure!(
                p.data.data().iter().all(|v| v.is_finite()),
                "nonfinite parameter {n}"
            );
            self.dense.get_mut(n).unwrap().value = Arc::new(p.data.data().to_vec());
            p.zero_grad();
        }
        Ok(norm)
    }
    pub fn export(&self, out: &Path) -> Result<()> {
        ensure!(
            self.mtp.shard_paths().len() == 1,
            "MTP export requires a single shard"
        );
        ensure!(!out.exists(), "refusing to overwrite {}", out.display());
        let temp = out.with_extension(format!("partial-{}", std::process::id()));
        ensure!(!temp.exists(), "partial checkpoint already exists");
        std::fs::copy(self.mtp.shard_paths()[0], &temp)?;
        let result = (|| -> Result<()> {
            let mut f = std::fs::OpenOptions::new().write(true).open(&temp)?;
            for (n, p) in &self.params {
                let ti = self.mtp.info(&format!("blk.{}.{n}", self.layer))?;
                let bytes = encode(ti.ty, p.data.data())?;
                ensure!(bytes.len() as u64 == ti.nbytes, "export length for {n}");
                f.seek(SeekFrom::Start(ti.offset))?;
                f.write_all(&bytes)?;
            }
            f.sync_all()?;
            let loaded = Gguf::open(&temp)?;
            for (n, p) in &self.params {
                let ti = loaded.info(&format!("blk.{}.{n}", self.layer))?;
                let values = loaded.dequantize(ti)?;
                ensure!(
                    values.len() == p.data.numel() && values.iter().all(|v| v.is_finite()),
                    "export reload for {n}"
                );
            }
            std::fs::rename(&temp, out)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&temp);
        }
        result
    }
}
pub fn encode(ty: GgmlType, v: &[f32]) -> Result<Vec<u8>> {
    ensure!(v.iter().all(|v| v.is_finite()), "nonfinite export");
    Ok(match ty {
        GgmlType::F32 => v.iter().flat_map(|v| v.to_le_bytes()).collect(),
        GgmlType::Bf16 => v
            .iter()
            .flat_map(|&v| crate::flash::pack::bf16_bits(v).to_le_bytes())
            .collect(),
        GgmlType::Q8_0 => {
            ensure!(v.len() % 32 == 0, "Q8 alignment");
            let mut b = Vec::new();
            for x in v.chunks_exact(32) {
                let max = x.iter().map(|v| v.abs()).fold(0.0, f32::max);
                let d = max / 127.0;
                let half = f32_to_f16(d);
                ensure!(
                    crate::gguf::f16_to_f32(half).is_finite(),
                    "Q8 scale overflow"
                );
                b.extend(half.to_le_bytes());
                b.extend(x.iter().map(|&v| {
                    if d == 0.0 {
                        0
                    } else {
                        (v / d).round().clamp(-127.0, 127.0) as i8 as u8
                    }
                }));
            }
            b
        }
        _ => bail!("unsupported export type {ty:?}"),
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn q8_roundtrip() {
        let v: Vec<_> = (0..64).map(|i| (i as f32 - 31.0) / 13.0).collect();
        let b = encode(GgmlType::Q8_0, &v).unwrap();
        let mut q = vec![0.0; 64];
        crate::gguf::dequantize(GgmlType::Q8_0, &b, &mut q).unwrap();
        assert!(v.iter().zip(q).all(|(a, b)| (a - b).abs() < 0.012));
        assert!(encode(GgmlType::Q8_0, &v[..33]).is_err());
    }
}
