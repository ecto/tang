//! Batched signing and verification on the GPU.
//!
//! Signing runs in rounds: each round tries the next attempts of every
//! message still pending, as one batch. When few remain, it tries several
//! of each one's attempts at once to keep the batch wide, and takes each
//! message's lowest accepted attempt. Accepted messages drop out. A message's round count is its attempt
//! count, which the attempt graph already declassifies, so batching
//! reveals nothing new. Signing is deterministic, so a GPU signature is
//! byte-for-byte the CPU's.
//!
//! Verification is the public verify graph, batched; parsing, hint
//! decoding and the final hash stay on the host.

use crate::sig::{finish_signature, verify_start, Prepared, Sig, SigningKey};
use std::io;
use tang_compile::tensor::modq::gpu::GpuCompiled;

/// The signature's attempt and verify graphs, built for Metal.
pub struct SigMetal {
    attempt: GpuCompiled,
    verify: GpuCompiled,
}

/// What a batch cost.
#[derive(Debug, Default, Clone, Copy)]
pub struct BatchStats {
    /// device time summed over rounds, in nanoseconds
    pub device_ns: u64,
    /// GPU batches run (signing: rounds; verifying: 1)
    pub rounds: usize,
    /// attempts summed over messages, counting to each one's accepted
    /// attempt (speculative attempts past it are not counted)
    pub attempts: usize,
}

impl SigMetal {
    pub fn new(sig: &Sig) -> io::Result<Self> {
        let (g, outs) = sig.attempt_graph_outputs();
        let attempt = g.compile_metal(&outs)?;
        let (g, outs) = sig.verify_graph();
        let verify = g.compile_metal(&outs)?;
        Ok(Self { attempt, verify })
    }

    /// Sign every message under one key. Returns the signatures in order.
    pub fn sign_batch(
        &self,
        sk: &SigningKey,
        msgs: &[&[u8]],
    ) -> io::Result<(Vec<Vec<u8>>, BatchStats)> {
        let preps: Vec<Prepared> = msgs.iter().map(|m| Prepared::new(sk, m)).collect();
        let mut sigs: Vec<Option<Vec<u8>>> = vec![None; msgs.len()];
        let mut pending: Vec<usize> = (0..msgs.len()).collect();
        let mut stats = BatchStats::default();
        // next untried attempt per message
        let mut next = vec![0usize; msgs.len()];
        let width = msgs.len().max(32);
        while !pending.is_empty() {
            // keep the batch about as wide as it started: when few messages
            // remain, try several of each one's next attempts at once
            let k = (width / pending.len()).clamp(1, 64);
            let mut slots = Vec::with_capacity(pending.len() * k);
            for &i in &pending {
                for j in 0..k {
                    assert!(
                        next[i] + j < 1000,
                        "no signature in 1000 attempts (expected ~5)"
                    );
                    slots.push((i, next[i] + j));
                }
            }
            let sets: Vec<Vec<Vec<u32>>> =
                slots.iter().map(|&(i, a)| preps[i].inputs(sk, a)).collect();
            let (outs, ns) = self.attempt.run(&sets)?;
            stats.device_ns += ns;
            stats.rounds += 1;
            // each message's lowest accepted attempt, exactly as the CPU loop
            // would find it; attempts beyond it are discarded
            for (n, &i) in pending.iter().enumerate() {
                for j in 0..k {
                    if let Some(sig) = finish_signature(&outs[n * k + j]) {
                        sigs[i] = Some(sig);
                        stats.attempts += next[i] + j + 1;
                        break;
                    }
                }
                next[i] += k;
            }
            pending.retain(|&i| sigs[i].is_none());
        }
        Ok((
            sigs.into_iter().map(|s| s.expect("signed")).collect(),
            stats,
        ))
    }

    /// Verify `(pk, msg, sig)` triples. Malformed signatures are rejected on
    /// the host and never reach the GPU.
    pub fn verify_batch(
        &self,
        items: &[(&[u8], &[u8], &[u8])],
    ) -> io::Result<(Vec<bool>, BatchStats)> {
        let starts: Vec<_> = items
            .iter()
            .map(|(pk, m, s)| verify_start(pk, m, s))
            .collect();
        let sets: Vec<Vec<Vec<u32>>> = starts
            .iter()
            .flatten()
            .map(|(ins, _)| ins.clone())
            .collect();
        let mut stats = BatchStats::default();
        let ws = if sets.is_empty() {
            Vec::new()
        } else {
            let (ws, ns) = self.verify.run(&sets)?;
            stats.device_ns = ns;
            stats.rounds = 1;
            ws
        };
        let mut ws = ws.into_iter();
        let ok = starts
            .iter()
            .map(|st| match st {
                Some((_, rest)) => rest.finish(&ws.next().expect("one w per started item")),
                None => false,
            })
            .collect();
        Ok((ok, stats))
    }
}
