//! Draft-model speculation: a small model sharing the target's tokenizer (Qwen3-0.6B or -1.7B
//! for Qwen3-32B) guesses the next few tokens greedily, and the target verifies them in one
//! forward. The target still samples every token itself, so outputs don't change; a draft
//! only saves forwards when it guesses right.
//!
//! The drafter keeps its own KV cache of the conversation. Before each draft it drops what no
//! longer matches (rejected guesses, another conversation) and runs what's new.

use crate::model::{Cache, Dtype, Model};
use anyhow::{ensure, Result};
use std::path::Path;
use tang_compute::ComputeDevice;

/// Prefill chunk, like the engine's.
const CHUNK: usize = 512;

pub struct DraftModel<D: ComputeDevice> {
    model: Model<D>,
    cache: Cache<D::Buffer>,
    /// Most tokens a draft holds.
    pub max: usize,
    /// Stop drafting once the drafter gives the whole draft less than this chance of being
    /// right (the product of its tokens' probabilities): a wide verify costs more than a
    /// narrow one, so long shaky drafts lose.
    pub min_prob: f32,
}

impl<D: ComputeDevice> DraftModel<D> {
    /// `vocab`: the target's vocabulary size; the drafter's must not be larger.
    pub fn load(dev: D, dir: &Path, max_ctx: usize, vocab: usize) -> Result<Self> {
        let model = Model::load(dev, dir, max_ctx, Dtype::Bf16)?;
        ensure!(
            model.cfg.vocab_size <= vocab,
            "draft model's vocabulary ({}) is larger than the target's ({vocab})",
            model.cfg.vocab_size
        );
        let cache = model.new_cache();
        let get = |k: &str| std::env::var(format!("TANG_DRAFT_MODEL_{k}")).ok();
        let mut d = Self {
            model,
            cache,
            max: get("MAX").and_then(|v| v.parse().ok()).unwrap_or(8),
            min_prob: get("MIN_PROB").and_then(|v| v.parse().ok()).unwrap_or(0.5),
        };
        // Warm up (kernel compilation, first allocations) here rather than in a request.
        let min_prob = std::mem::replace(&mut d.min_prob, 0.0);
        d.propose(&[0; 16], &[], 4)?;
        d.min_prob = min_prob;
        Ok(d)
    }

    /// Guess up to `room` tokens following `prompt` then `out`.
    pub fn propose(&mut self, prompt: &[u32], out: &[u32], room: usize) -> Result<Vec<u32>> {
        let room = room.min(self.max);
        let n = prompt.len() + out.len();
        if room == 0 || n == 0 || n + room > self.model.max_ctx() {
            return Ok(Vec::new());
        }
        let at = |i: usize| {
            if i < prompt.len() {
                prompt[i]
            } else {
                out[i - prompt.len()]
            }
        };
        // Keep the cached positions that still match, but always run at least the last token
        // (its logits start the draft).
        let keep = (0..self.cache.len.min(n - 1))
            .take_while(|&i| self.cache.tokens[i] == at(i))
            .count();
        self.cache.truncate(keep);
        let new: Vec<u32> = (keep..n).map(at).collect();
        let mut logits = Vec::new();
        for chunk in new.chunks(CHUNK) {
            logits = self.model.forward(chunk, &mut self.cache, false)?;
        }
        let mut draft = Vec::with_capacity(room);
        let mut q = 1.0;
        loop {
            let (tok, p) = top(&logits);
            q *= p;
            if q < self.min_prob {
                break;
            }
            draft.push(tok);
            if draft.len() == room {
                break;
            }
            logits = self.model.forward(&[tok], &mut self.cache, false)?;
        }
        Ok(draft)
    }
}

/// The most likely token and its probability.
fn top(logits: &[f32]) -> (u32, f32) {
    let (mut best, mut max) = (0usize, f32::NEG_INFINITY);
    for (i, &l) in logits.iter().enumerate() {
        if l > max {
            (best, max) = (i, l);
        }
    }
    let sum: f32 = logits.iter().map(|&l| (l - max).exp()).sum();
    (best as u32, 1.0 / sum)
}
