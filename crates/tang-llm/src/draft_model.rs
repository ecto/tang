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
    /// One drafter forward's time relative to one target forward.
    pub step_cost: f32,
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
            step_cost: get("COST").and_then(|v| v.parse().ok()).unwrap_or(0.07),
        };
        // Warm up (kernel compilation, first allocations) here rather than in a request.
        d.propose(&[0; 16], &[], 4, |k| 0.5 + k as f32 * 0.5)?;
        Ok(d)
    }

    /// Guess up to `room` tokens following `prompt` then `out`. `cost(k)`: a target forward
    /// over k tokens relative to one. The draft grows while the expected tokens per unit of
    /// time (verify plus drafting) improve, with the drafter's own probabilities as the odds
    /// that each token survives.
    pub fn propose(
        &mut self,
        prompt: &[u32],
        out: &[u32],
        room: usize,
        cost: impl Fn(usize) -> f32,
    ) -> Result<Vec<u32>> {
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
        // Odds the draft so far survives, expected tokens it yields, and its rate.
        let (mut q, mut gain, mut best) = (1.0f32, 1.0f32, 1.0 / cost(1));
        loop {
            let (tok, p) = top(&logits);
            q *= p;
            let n = draft.len() + 1;
            let rate = (gain + q) / (cost(n + 1) + self.step_cost * n as f32);
            if rate <= best {
                break;
            }
            (gain, best) = (gain + q, rate);
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
