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
        let by_logit = |a: &u32, b: &u32| logits[*b as usize].total_cmp(&logits[*a as usize]);
        if k < idx.len() {
            idx.select_nth_unstable_by(k - 1, by_logit);
            idx.truncate(k);
        }
        idx.sort_unstable_by(by_logit);
        let max = logits[idx[0] as usize];
        let mut probs: Vec<f32> = idx
            .iter()
            .map(|&i| ((logits[i as usize] - max) / self.cfg.temperature).exp())
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
