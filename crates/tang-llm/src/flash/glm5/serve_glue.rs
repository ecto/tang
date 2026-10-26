//! `ServeEngine` for GLM-5.3-Flash (W6): adapts W1's [`Glm5Engine`] — whose window takes the
//! ids and returns their logits — to the server's position/verify contract, without touching
//! the engine.
//!
//! Deliberately minimal (GATES.md: WORK before FAST):
//! - Greedy only. `set_sampling` is stored; the GPU head-sampler (a `Sampler`) is not built
//!   for this engine, so `sampler_mut()` is `None` and decode walks argmax.
//! - One sequence (`max_slots() == 1`): the batcher never engages. Concurrent requests queue.
//! - `save_running`/`restore_running` carry only the token prefix: restoring replays it
//!   through the engine (correct, one prefill's worth of time). True snapshots need KDA/DSA
//!   state access on `Glm5Engine` (they are private); until then, prefix reuse across
//!   requests re-prefills. `save_positional`/`restore_positional` are empty (the DSA caches
//!   live inside the engine and rewind with it).
//! - No drafts: `use_mtp()` is false and the server runs this engine with `suffix: false`,
//!   so `verify` always sees a single token and never needs to rewind (a draft rollback
//!   would have to replay).

use super::engine::{argmax, Glm5Engine};
use crate::flash::engine::{Positional, RunningState, WinStats};
use crate::flash::serve::ServeEngine;
use anyhow::{ensure, Result};
use std::path::Path;
use std::time::Instant;

/// GLM-5.3-Flash behind the server's engine trait.
pub struct Glm5Serve {
    e: Glm5Engine,
    /// The last window's logits (`[t][n_vocab]`), for [`ServeEngine::logits`].
    last_logits: Vec<f32>,
    last_window: WinStats,
}

impl Glm5Serve {
    /// Open `path`, move the weights onto the GPU when built with `cuda` and a device is
    /// present, and warm the lazy per-layer weight caches with one throwaway window.
    pub fn load(path: &Path) -> Result<Self> {
        let mut e = Glm5Engine::open(path)?;
        let _ = e.use_gpu()?;
        // Fault in the head and any cached weights once, then drop the throwaway state.
        let _ = e.prefill(&[0], 1);
        e.reset();
        Ok(Glm5Serve {
            e,
            last_logits: Vec::new(),
            last_window: WinStats::default(),
        })
    }
}

impl ServeEngine for Glm5Serve {
    fn tokens(&self) -> &Vec<u32> {
        &self.e.tokens
    }

    fn tokens_mut(&mut self) -> &mut Vec<u32> {
        &mut self.e.tokens
    }

    fn reset(&mut self) {
        self.e.reset();
        self.last_logits.clear();
    }

    fn gguf(&self) -> &crate::gguf::Gguf {
        &self.e.m.g
    }

    fn last(&self) -> WinStats {
        self.last_window
    }

    fn save_running(&mut self) -> Result<RunningState> {
        // Token prefix only (see the module docs): one empty part, no GDN layers.
        RunningState::from_parts(self.e.tokens.len(), 0, vec![Vec::new()])
    }

    fn restore_running(&mut self, s: &RunningState, prefix: &[u32]) -> Result<()> {
        ensure!(prefix.len() == s.pos, "prefix/state length mismatch");
        // Re-run the prefix: the engine's recurrent state is not exposed for snapshotting.
        self.e.reset();
        if !prefix.is_empty() {
            self.e.prefill(prefix, 512)?;
        }
        Ok(())
    }

    fn save_positional(&mut self, from: usize, to: usize) -> Result<Positional> {
        Ok(Positional::from_parts(from, to, Vec::new()))
    }

    fn restore_positional(&mut self, _p: &Positional, _from: usize) -> Result<()> {
        Ok(())
    }

    fn prefill(&mut self, ids: &[u32], chunk: usize) -> Result<u32> {
        // Appends (the trait contract; the caller has not extended `tokens`).
        let mut next = 0u32;
        for c in ids.chunks(chunk.max(1)) {
            let lg = self.run_window(c)?;
            next = argmax(&lg[(c.len() - 1) * self.e.m.hp.n_vocab..]);
        }
        Ok(next)
    }

    fn window(&mut self, pos0: usize, t: usize) -> Result<Vec<u32>> {
        // The caller has already extended `tokens` with the window's ids; the engine's
        // `window` appends what it is given, so hand it exactly that slice.
        ensure!(pos0 + t <= self.e.tokens.len(), "window past the sequence");
        let ids: Vec<u32> = self.e.tokens[pos0..pos0 + t].to_vec();
        let lg = self.run_window(&ids)?;
        Ok((0..t)
            .map(|i| argmax(&lg[i * self.e.m.hp.n_vocab..(i + 1) * self.e.m.hp.n_vocab]))
            .collect())
    }

    fn verify(&mut self, cur: u32, drafts: &[u32]) -> Result<Vec<u32>> {
        // Drafting is off for this engine (no MTP, no suffix drafts), so `drafts` is empty
        // and a verify is one decode step. A non-empty draft list would need a state
        // rewind the engine cannot express yet: run the chain, keep the greedy prefix,
        // replay the kept tokens from scratch (correct, slow, and currently unreachable).
        self.e.tokens.push(cur);
        let mut lg = self.run_window(&[cur])?;
        let nv = self.e.m.hp.n_vocab;
        let mut kept = vec![argmax(&lg[(lg.len() - nv)..])];
        if drafts.is_empty() {
            return Ok(kept);
        }
        for (i, &d) in drafts.iter().enumerate() {
            self.e.tokens.push(d);
            lg = self.run_window(&[d])?;
            let next = argmax(&lg[(lg.len() - nv)..]);
            if next != d || i + 1 == drafts.len() {
                break;
            }
            kept.push(next);
        }
        // Rewind: the engine state now covers the whole attempted chain; rebuild it for the
        // accepted prefix only.
        let keep_to = self.e.tokens.len() - (drafts.len() + 1 - kept.len());
        let prefix: Vec<u32> = self.e.tokens[..keep_to].to_vec();
        self.e.reset();
        if !prefix.is_empty() {
            self.e.prefill(&prefix, 512)?;
        }
        Ok(kept)
    }

    fn logits(&self, t: usize) -> Vec<f32> {
        let nv = self.e.m.hp.n_vocab;
        let start = self.last_logits.len().saturating_sub(t * nv);
        self.last_logits[start..].to_vec()
    }

    fn set_sampling(&mut self, _temperature: f32, _seed: u32) {
        // Greedy only (see the module docs).
    }

    fn sampler_mut(&mut self) -> Option<&mut crate::flash::sampler::Sampler> {
        None
    }

    fn sampler_begin(&mut self, _pos: usize) -> Result<()> {
        Ok(())
    }
}

impl Glm5Serve {
    /// Run `ids` through the engine as one window (the engine appends them itself), cache
    /// the logits, and record the wall time.
    fn run_window(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        let t0 = Instant::now();
        let lg = self.e.window(ids)?;
        let secs = t0.elapsed().as_secs_f64();
        self.last_window = WinStats {
            t: ids.len(),
            wall_ms: secs * 1e3,
            ..WinStats::default()
        };
        crate::telemetry::glm5_window(ids.len(), secs);
        self.last_logits = lg;
        Ok(self.last_logits.clone())
    }
}
