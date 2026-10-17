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

use crate::sig::{
    finish_signature, key_values, matrix, verify_start_with, Prepared, Sig, SigningKey,
};
use rayon::prelude::*;
use std::io;
use tang_compile::tensor::modq::gpu::GpuCompiled;

/// The signature's attempt and verify graphs, built for Metal.
pub struct SigMetal {
    attempt: GpuCompiled,
    verify: GpuCompiled,
    /// verify with A and t₁ sent once, for batches under one public key
    verify_one_key: GpuCompiled,
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
        // A, s₁, s₂, t₀, K go once per batch: one key signs the whole batch
        let attempt = g.compile_metal_shared(&outs, &sig.attempt_key_inputs())?;
        let (g, outs) = sig.verify_graph();
        let verify = g.compile_metal(&outs)?;
        let verify_one_key = g.compile_metal_shared(&outs, &sig.verify_key_inputs())?;
        Ok(Self {
            attempt,
            verify,
            verify_one_key,
        })
    }

    /// Sign every message under one key. Returns the signatures in order.
    pub fn sign_batch(
        &self,
        sk: &SigningKey,
        msgs: &[&[u8]],
    ) -> io::Result<(Vec<Vec<u8>>, BatchStats)> {
        // A once per key, not once per message
        let a = std::sync::Arc::new(matrix(sk.rho()));
        let shared = key_values(sk, &a);
        let preps: Vec<Prepared> = msgs
            .par_iter()
            .map(|m| Prepared::with_matrix(sk, m, a.clone()))
            .collect();
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
                slots.par_iter().map(|&(i, a)| preps[i].tails(a)).collect();
            let (outs, ns) = self.attempt.run_shared(&shared, &sets)?;
            stats.device_ns += ns;
            stats.rounds += 1;
            // each message's lowest accepted attempt, exactly as the CPU loop
            // would find it; attempts beyond it are discarded
            let found: Vec<Option<(usize, Vec<u8>)>> = (0..pending.len())
                .into_par_iter()
                .map(|n| (0..k).find_map(|j| finish_signature(&outs[n * k + j]).map(|s| (j, s))))
                .collect();
            for (n, f) in found.into_iter().enumerate() {
                let i = pending[n];
                if let Some((j, sig)) = f {
                    sigs[i] = Some(sig);
                    stats.attempts += next[i] + j + 1;
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
        // A once per distinct public key in the batch, sampled in parallel
        let mut rhos: Vec<&[u8]> = items
            .iter()
            .filter(|(pk, _, _)| pk.len() >= 32)
            .map(|(pk, _, _)| &pk[..32])
            .collect();
        rhos.sort();
        rhos.dedup();
        let cache: std::collections::HashMap<&[u8], Vec<u32>> =
            rhos.par_iter().map(|rho| (*rho, matrix(rho))).collect();
        let starts: Vec<_> = items
            .par_iter()
            .map(|(pk, m, s)| verify_start_with(pk, m, s, &mut |rho| cache[rho].clone()))
            .collect();
        let mut stats = BatchStats::default();
        let started: Vec<&Vec<Vec<u32>>> = starts.iter().flatten().map(|(ins, _)| ins).collect();
        // one public key for the whole batch: A and t₁ go once
        let one_key = started.len() > 1
            && started
                .iter()
                .all(|ins| ins[1] == started[0][1] && ins[0] == started[0][0]);
        let ws = if started.is_empty() {
            Vec::new()
        } else if one_key {
            let shared = vec![started[0][0].clone(), started[0][1].clone()];
            let sets: Vec<Vec<Vec<u32>>> = started
                .par_iter()
                .map(|ins| vec![ins[2].clone(), ins[3].clone()])
                .collect();
            let (ws, ns) = self.verify_one_key.run_shared(&shared, &sets)?;
            stats.device_ns = ns;
            stats.rounds = 1;
            ws
        } else {
            let sets: Vec<Vec<Vec<u32>>> = started.par_iter().map(|ins| (*ins).clone()).collect();
            let (ws, ns) = self.verify.run(&sets)?;
            stats.device_ns = ns;
            stats.rounds = 1;
            ws
        };
        // each started item's w, by position among the started ones
        let mut slot = Vec::with_capacity(starts.len());
        let mut k = 0;
        for st in &starts {
            slot.push(st.as_ref().map(|_| {
                k += 1;
                k - 1
            }));
        }
        let ok = starts
            .par_iter()
            .zip(slot.par_iter())
            .map(|(st, w)| match (st, w) {
                (Some((_, rest)), Some(w)) => rest.finish(&ws[*w]),
                _ => false,
            })
            .collect();
        Ok((ok, stats))
    }
}
