//! Token sampling: temperature, top-k, top-p, repetition-free greedy at temperature 0.

#[derive(Debug, Clone)]
pub struct Sampling {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: u64,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 0.6,
            top_k: 20,
            top_p: 0.95,
            seed: 0x5eed,
        }
    }
}

pub struct Sampler {
    cfg: Sampling,
    rng: u64,
}

impl Sampler {
    pub fn new(cfg: Sampling) -> Self {
        let rng = cfg.seed.max(1);
        Self { cfg, rng }
    }

    fn next_f32(&mut self) -> f32 {
        // xorshift64*
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        ((self.rng.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32) / (1u64 << 24) as f32
    }

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        if self.cfg.temperature <= 0.0 {
            return argmax(logits);
        }
        let k = if self.cfg.top_k == 0 {
            logits.len()
        } else {
            self.cfg.top_k.min(logits.len())
        };
        let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
        // Ties go to the lower index, as on the device (`top_k_rows`).
        let by_logit = |a: &u32, b: &u32| {
            logits[*b as usize]
                .total_cmp(&logits[*a as usize])
                .then(a.cmp(b))
        };
        if k < idx.len() {
            idx.select_nth_unstable_by(k - 1, by_logit);
            idx.truncate(k);
        }
        idx.sort_unstable_by(by_logit);
        let vals: Vec<f32> = idx.iter().map(|&i| logits[i as usize]).collect();
        self.pick(&vals, &idx)
    }

    /// How many of a row's largest logits [`sample_top`](Self::sample_top) needs to sample
    /// exactly as [`sample`](Self::sample) would from the whole row, if that's at most `max`.
    pub fn top_needed(&self, max: usize) -> Option<usize> {
        match self.cfg.top_k {
            _ if self.cfg.temperature <= 0.0 => Some(1),
            k if (1..=max).contains(&k) => Some(k),
            _ => None,
        }
    }

    /// [`sample`](Self::sample) from a row's largest logits, `vals` descending with their
    /// columns `idx` (at least [`top_needed`](Self::top_needed) of them).
    pub fn sample_top(&mut self, vals: &[f32], idx: &[u32]) -> u32 {
        if self.cfg.temperature <= 0.0 {
            return idx[0];
        }
        let k = self.cfg.top_k.min(vals.len());
        self.pick(&vals[..k], &idx[..k])
    }

    /// Temperature, nucleus and the draw over candidates sorted by logit, descending.
    fn pick(&mut self, vals: &[f32], idx: &[u32]) -> u32 {
        let max = vals[0];
        let mut probs: Vec<f32> = vals
            .iter()
            .map(|&v| ((v - max) / self.cfg.temperature).exp())
            .collect();
        let sum: f32 = probs.iter().sum();
        probs.iter_mut().for_each(|p| *p /= sum);
        // Nucleus: keep the smallest prefix reaching top_p.
        let mut cum = 0.0;
        let mut keep = probs.len();
        for (i, p) in probs.iter().enumerate() {
            cum += p;
            if cum >= self.cfg.top_p {
                keep = i + 1;
                break;
            }
        }
        let total: f32 = probs[..keep].iter().sum();
        let mut r = self.next_f32() * total;
        for i in 0..keep {
            r -= probs[i];
            if r <= 0.0 {
                return idx[i];
            }
        }
        idx[keep - 1]
    }
}

pub fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_from_the_top_matches_the_whole_row() {
        let logits: Vec<f32> = (0..500).map(|i| ((i * 7919 % 211) as f32) / 37.0).collect();
        for (temperature, top_k) in [(0.0, 20), (0.7, 20), (1.0, 1), (0.6, 64)] {
            let cfg = Sampling {
                temperature,
                top_k,
                top_p: 0.95,
                seed: 3,
            };
            let (mut a, mut b) = (Sampler::new(cfg.clone()), Sampler::new(cfg));
            let k = b.top_needed(64).unwrap();
            let (vals, idx) = tang_compute::top_k_rows_host(&logits, logits.len(), k);
            for _ in 0..50 {
                assert_eq!(a.sample(&logits), b.sample_top(&vals, &idx));
            }
        }
        let full = Sampler::new(Sampling {
            top_k: 0,
            ..Default::default()
        });
        assert_eq!(full.top_needed(64), None);
    }

    #[test]
    fn greedy_and_top_k_one_pick_the_max() {
        let logits = [0.1, 3.0, 2.9, -1.0];
        let mut s = Sampler::new(Sampling {
            temperature: 0.0,
            ..Default::default()
        });
        assert_eq!(s.sample(&logits), 1);
        let mut s = Sampler::new(Sampling {
            temperature: 1.0,
            top_k: 1,
            ..Default::default()
        });
        for _ in 0..20 {
            assert_eq!(s.sample(&logits), 1);
        }
    }

    #[test]
    fn samples_stay_in_the_nucleus() {
        let logits = [5.0, 4.9, -10.0, -10.0];
        let mut s = Sampler::new(Sampling {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.9,
            seed: 7,
        });
        let picks: Vec<u32> = (0..200).map(|_| s.sample(&logits)).collect();
        assert!(picks.iter().all(|&p| p < 2));
        assert!(picks.contains(&0) && picks.contains(&1));
    }
}
