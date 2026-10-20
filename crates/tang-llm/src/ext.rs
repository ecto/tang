//! The model side of [`tang_compute::ext`]: what a [`Model`] resolved at load, the whole-step
//! slot, and the parity checks every registered implementation has to pass.
//!
//! See `docs/extensions.md`.

use crate::model::{Model, Segment};
use anyhow::Result;
use std::sync::Arc;
use tang_compute::ext::{self, Entry, Ops, Target};
use tang_compute::ComputeDevice;

pub use tang_compute::ext::{
    register, AllReduce, KvAttentionPaged, Linear, Match, PagedAttention, RmsNorm, Shape,
};

/// A whole forward over a batch, as [`Model::forward_batch`] (a persistent megakernel, a
/// captured CUDA graph). Returns `None` to decline the call (say, a batch with prefill rows),
/// and the built-in path runs it. On `Some`, it must also have done what `forward_batch` does
/// to each segment's cache: written its rows' K/V, and appended its tokens.
pub type StepFn<D> = dyn for<'a> Fn(
        &Model<D>,
        &mut [Segment<'a, <D as ComputeDevice>::Buffer>],
    ) -> Option<Result<Vec<Vec<f32>>>>
    + Send
    + Sync;

pub struct Step<D: ComputeDevice>(pub Arc<StepFn<D>>);

impl<D: ComputeDevice> Clone for Step<D> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

/// What a model resolved: its target, the op slots, and a step override.
pub struct Ext<D: ComputeDevice> {
    pub target: Target,
    pub ops: Ops<D>,
    pub step: Option<Entry<Step<D>>>,
}

impl<D: ComputeDevice> Default for Ext<D> {
    fn default() -> Self {
        Self {
            target: Target::default(),
            ops: Ops::default(),
            step: None,
        }
    }
}

impl<D: ComputeDevice + 'static> Ext<D> {
    pub fn resolve(target: Target) -> Self {
        let e = Self {
            ops: Ops::resolve(&target),
            step: ext::select(&target),
            target,
        };
        let names = e.names();
        if !names.is_empty() {
            eprintln!(
                "tang-llm: extensions for {:?}: {}",
                e.target,
                names.join(", ")
            );
        }
        e
    }

    pub fn names(&self) -> Vec<&'static str> {
        let mut n = self.ops.names();
        n.extend(self.step.as_ref().map(|e| e.name));
        n
    }
}

/// The target a model with config `c` resolves for on `dev`, its matrices stored as `quant`.
pub fn target<D: ComputeDevice>(dev: &D, c: &crate::Config, quant: &str) -> Target {
    Target {
        arch: dev.arch(),
        model: c.model_type.clone(),
        shape: Shape {
            hidden: c.hidden_size,
            layers: c.num_hidden_layers,
            heads: c.num_attention_heads,
            kv_heads: c.kv_heads(),
            head_dim: c.head_dim(),
            ff: c.intermediate_size,
            vocab: c.vocab_size,
        },
        quant: quant.into(),
    }
}

/// Parity checks for registered implementations: the same bar the built-in kernels meet
/// (`*_vs_cpu`, `tests/speculate.rs`), runnable from another crate against its own kernels.
///
/// Each check runs the same work twice on one device, once with the extension resolved and
/// once with the built-in kernels, and compares.
pub mod parity {
    use super::*;
    use crate::model::Cache;

    /// Max |a - b| / max(1, |b|) over two outputs.
    pub fn max_rel_err(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "lengths differ");
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs() / y.abs().max(1.0))
            .fold(0.0, f32::max)
    }

    fn fill(n: usize, seed: &mut u32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (*seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect()
    }

    /// `imp` against [`ComputeDevice::linear`] for each `(m, k, n)` with f32 weights; a case
    /// it declines is skipped. Returns the worst relative error.
    pub fn linear<D: ComputeDevice>(
        dev: &D,
        imp: &Linear<D>,
        cases: &[(usize, usize, usize)],
        tol: f32,
    ) -> Result<f32> {
        let mut seed = 7;
        let mut worst = 0f32;
        for &(m, k, n) in cases {
            let x = dev.upload(&fill(m * k, &mut seed));
            let w = dev.upload(&fill(n * k, &mut seed));
            let want = dev.download(&dev.linear(&x, &w, m, k, n));
            let Some(got) = (imp.0)(dev, &x, &w, m, k, n) else {
                continue;
            };
            let e = max_rel_err(&dev.download(&got), &want);
            anyhow::ensure!(e <= tol, "linear {m}x{k}x{n}: error {e} > {tol}");
            worst = worst.max(e);
        }
        Ok(worst)
    }

    /// `imp` against [`ComputeDevice::rms_norm`] for each `(rows, dim)`.
    pub fn rms_norm<D: ComputeDevice>(
        dev: &D,
        imp: &RmsNorm<D>,
        cases: &[(usize, usize)],
        tol: f32,
    ) -> Result<f32> {
        let mut seed = 11;
        let mut worst = 0f32;
        for &(rows, dim) in cases {
            let x = dev.upload(&fill(rows * dim, &mut seed));
            let w = dev.upload(&fill(dim, &mut seed));
            let want = dev.download(&dev.rms_norm(&x, &w, rows, dim, 1e-6));
            let Some(got) = (imp.0)(dev, &x, &w, rows, dim, 1e-6) else {
                continue;
            };
            let e = max_rel_err(&dev.download(&got), &want);
            anyhow::ensure!(e <= tol, "rms_norm {rows}x{dim}: error {e} > {tol}");
            worst = worst.max(e);
        }
        Ok(worst)
    }

    /// The model end to end: `prompts` prefilled, then `steps` greedy decode steps batched
    /// across them, with the model's extensions and again with the built-in kernels. Logits
    /// must agree within `tol` at every step and the greedy tokens must be identical (the
    /// bar `tests/speculate.rs` holds drafting to). Leaves the model's extensions as they were.
    pub fn model<D: ComputeDevice + 'static>(
        m: &mut Model<D>,
        prompts: &[&[u32]],
        steps: usize,
        tol: f32,
    ) -> Result<f32> {
        let resolved = m.take_ext();
        anyhow::ensure!(
            !resolved.names().is_empty(),
            "nothing resolved for {:?}",
            resolved.target
        );
        let base = decode(m, prompts, steps);
        m.set_ext(resolved);
        let with = decode(m, prompts, steps);
        let (base, with) = (base?, with?);
        let mut worst = 0f32;
        for (i, (b, w)) in base.iter().zip(&with).enumerate() {
            let e = max_rel_err(&w.0, &b.0);
            anyhow::ensure!(e <= tol, "step {i}: logits error {e} > {tol}");
            anyhow::ensure!(w.1 == b.1, "step {i}: tokens {:?} != {:?}", w.1, b.1);
            worst = worst.max(e);
        }
        Ok(worst)
    }

    /// A small random Qwen3-shaped checkpoint in `dir` (64 hidden, 2 layers, 4 query and 2
    /// KV heads of 16, vocab 256), for running [`model`] without downloading one.
    pub fn tiny_checkpoint(dir: &std::path::Path) -> Result<()> {
        use safetensors::tensor::{Dtype, TensorView};
        let (h, ff, v, nh, nkv, hd, layers) = (64, 128, 256, 4, 2, 16, 2);
        std::fs::create_dir_all(dir)?;
        std::fs::write(
            dir.join("config.json"),
            format!(
                r#"{{"model_type":"qwen3","hidden_size":{h},"intermediate_size":{ff},"num_hidden_layers":{layers},"num_attention_heads":{nh},"num_key_value_heads":{nkv},"head_dim":{hd},"vocab_size":{v},"rms_norm_eps":1e-6,"rope_theta":10000,"tie_word_embeddings":true}}"#
            ),
        )?;
        let mut seed = 3;
        let mut tensors: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        let mut add = |name: String, shape: Vec<usize>, ones: bool| {
            let n: usize = shape.iter().product();
            let vals = if ones {
                fill(n, &mut seed).iter().map(|x| 1.0 + 0.2 * x).collect()
            } else {
                fill(n, &mut seed)
            };
            let bytes = vals.iter().flat_map(|x: &f32| x.to_le_bytes()).collect();
            tensors.push((name, shape, bytes));
        };
        add("model.embed_tokens.weight".into(), vec![v, h], false);
        add("model.norm.weight".into(), vec![h], true);
        for l in 0..layers {
            let p = format!("model.layers.{l}");
            add(format!("{p}.input_layernorm.weight"), vec![h], true);
            add(
                format!("{p}.post_attention_layernorm.weight"),
                vec![h],
                true,
            );
            add(
                format!("{p}.self_attn.q_proj.weight"),
                vec![nh * hd, h],
                false,
            );
            add(
                format!("{p}.self_attn.k_proj.weight"),
                vec![nkv * hd, h],
                false,
            );
            add(
                format!("{p}.self_attn.v_proj.weight"),
                vec![nkv * hd, h],
                false,
            );
            add(
                format!("{p}.self_attn.o_proj.weight"),
                vec![h, nh * hd],
                false,
            );
            add(format!("{p}.self_attn.q_norm.weight"), vec![hd], true);
            add(format!("{p}.self_attn.k_norm.weight"), vec![hd], true);
            add(format!("{p}.mlp.gate_proj.weight"), vec![ff, h], false);
            add(format!("{p}.mlp.up_proj.weight"), vec![ff, h], false);
            add(format!("{p}.mlp.down_proj.weight"), vec![h, ff], false);
        }
        let views = tensors
            .iter()
            .map(|(n, s, b)| Ok((n.clone(), TensorView::new(Dtype::F32, s.clone(), b)?)))
            .collect::<Result<Vec<_>>>()?;
        safetensors::serialize_to_file(views, &None, &dir.join("model.safetensors"))?;
        Ok(())
    }

    /// Each step's logits (all sequences, concatenated) and greedy tokens.
    fn decode<D: ComputeDevice>(
        m: &Model<D>,
        prompts: &[&[u32]],
        steps: usize,
    ) -> Result<Vec<(Vec<f32>, Vec<u32>)>> {
        let mut caches: Vec<Cache<D::Buffer>> = prompts.iter().map(|_| m.new_cache()).collect();
        let mut feed: Vec<Vec<u32>> = prompts.iter().map(|p| p.to_vec()).collect();
        let mut out = Vec::new();
        for _ in 0..=steps {
            let mut segs: Vec<Segment<'_, D::Buffer>> = feed
                .iter()
                .zip(caches.iter_mut())
                .map(|(t, cache)| Segment {
                    tokens: t,
                    cache,
                    all: false,
                })
                .collect();
            let logits = m.forward_batch(&mut segs)?;
            drop(segs);
            let toks: Vec<u32> = logits.iter().map(|l| argmax(l)).collect();
            feed = toks.iter().map(|&t| vec![t]).collect();
            out.push((logits.concat(), toks));
        }
        Ok(out)
    }

    fn argmax(l: &[f32]) -> u32 {
        l.iter()
            .enumerate()
            .fold(
                (0, f32::NEG_INFINITY),
                |b, (i, &v)| if v > b.1 { (i, v) } else { b },
            )
            .0 as u32
    }
}
