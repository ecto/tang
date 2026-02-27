use alloc::vec::Vec;
use tang::Scalar;
use tang_tensor::Tensor;

/// Configuration for token sampling.
#[derive(Clone, Debug)]
pub struct SamplingConfig {
    /// Temperature for logit scaling. 0.0 = greedy, 1.0 = standard, >1.0 = more random.
    pub temperature: f64,
    /// Top-k: only consider the k most likely tokens. 0 = disabled.
    pub top_k: usize,
    /// Top-p (nucleus): only consider tokens whose cumulative probability >= p. 1.0 = disabled.
    pub top_p: f64,
    /// Repetition penalty: scale down logits of tokens that have appeared. 1.0 = disabled.
    pub repetition_penalty: f64,
    /// Stop token ID. Generation stops when this is sampled.
    pub stop_token: Option<usize>,
    /// Maximum number of tokens to generate.
    pub max_tokens: usize,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            stop_token: None,
            max_tokens: 256,
        }
    }
}

impl SamplingConfig {
    /// Greedy decoding (always pick the most likely token).
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            ..Default::default()
        }
    }

    /// Typical chat/generation settings.
    pub fn standard() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            ..Default::default()
        }
    }
}

/// Token sampler with configurable strategies.
pub struct Sampler {
    config: SamplingConfig,
    rng_state: u64,
}

impl Sampler {
    pub fn new(config: SamplingConfig) -> Self {
        Self {
            config,
            rng_state: 42,
        }
    }

    pub fn with_seed(config: SamplingConfig, seed: u64) -> Self {
        Self {
            config,
            rng_state: seed,
        }
    }

    /// Simple LCG random number generator returning [0, 1).
    fn rand_f64(&mut self) -> f64 {
        self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.rng_state >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Sample a token index from logits `[vocab_size]`.
    ///
    /// Applies temperature, repetition penalty, top-k, top-p filtering,
    /// then samples from the resulting distribution.
    pub fn sample<S: Scalar>(&mut self, logits: &Tensor<S>, past_tokens: &[usize]) -> usize {
        assert_eq!(logits.ndim(), 1, "logits must be [vocab_size]");
        let vocab_size = logits.shape()[0];

        // Convert to f64 for sampling math
        let mut probs: Vec<f64> = (0..vocab_size).map(|i| logits.get(&[i]).to_f64()).collect();

        // Apply repetition penalty
        if self.config.repetition_penalty != 1.0 {
            for &tok in past_tokens {
                if tok < vocab_size {
                    if probs[tok] > 0.0 {
                        probs[tok] /= self.config.repetition_penalty;
                    } else {
                        probs[tok] *= self.config.repetition_penalty;
                    }
                }
            }
        }

        // Greedy: return argmax
        if self.config.temperature == 0.0 {
            return argmax(&probs);
        }

        // Temperature scaling
        if self.config.temperature != 1.0 {
            let inv_t = 1.0 / self.config.temperature;
            for p in probs.iter_mut() {
                *p *= inv_t;
            }
        }

        // Numerically stable softmax
        let max_val = probs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        for p in probs.iter_mut() {
            *p = (*p - max_val).exp();
        }

        // Top-k filtering
        if self.config.top_k > 0 && self.config.top_k < vocab_size {
            let threshold = kth_largest(&probs, self.config.top_k);
            for p in probs.iter_mut() {
                if *p < threshold {
                    *p = 0.0;
                }
            }
        }

        // Normalize
        let sum: f64 = probs.iter().sum();
        if sum <= 0.0 {
            return argmax(&probs);
        }
        for p in probs.iter_mut() {
            *p /= sum;
        }

        // Top-p (nucleus) filtering
        if self.config.top_p < 1.0 {
            top_p_filter(&mut probs, self.config.top_p);
            // Re-normalize
            let sum: f64 = probs.iter().sum();
            if sum > 0.0 {
                for p in probs.iter_mut() {
                    *p /= sum;
                }
            }
        }

        // Sample from distribution
        self.categorical_sample(&probs)
    }

    /// Sample from a categorical distribution given probabilities.
    fn categorical_sample(&mut self, probs: &[f64]) -> usize {
        let r = self.rand_f64();
        let mut cumsum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return i;
            }
        }
        // Fallback: return last non-zero
        probs.len() - 1
    }
}

/// Find the k-th largest value in a slice (selection without full sort).
fn kth_largest(values: &[f64], k: usize) -> f64 {
    let mut sorted: Vec<f64> = values.to_vec();
    // Partial sort: we just need the k-th element
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Equal));
    sorted[k.min(sorted.len()) - 1]
}

/// Argmax of a slice.
fn argmax(values: &[f64]) -> usize {
    let mut best_idx = 0;
    let mut best_val = f64::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx
}

/// Zero out tokens below the top-p cumulative probability threshold.
fn top_p_filter(probs: &mut [f64], p: f64) {
    // Get sorted indices by probability (descending)
    let mut indices: Vec<usize> = (0..probs.len()).collect();
    indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(core::cmp::Ordering::Equal));

    let mut cumsum = 0.0;
    let mut cutoff = probs.len();
    for (rank, &idx) in indices.iter().enumerate() {
        cumsum += probs[idx];
        if cumsum >= p {
            cutoff = rank + 1;
            break;
        }
    }

    // Zero out everything below the cutoff
    for &idx in &indices[cutoff..] {
        probs[idx] = 0.0;
    }
}

/// Autoregressive generation loop.
///
/// Takes a model forward function, prompt tokens, and sampling config.
/// Returns the generated token sequence (prompt + generated).
///
/// The `forward_fn` should:
/// 1. Accept `(token_ids: &[usize], cache: &mut KVCache<S>)`
/// 2. Return logits `[vocab_size]` for the next token
///
/// This is model-agnostic — the caller provides the forward function
/// that handles embedding, transformer layers, and LM head.
pub fn generate<S, F>(
    prompt: &[usize],
    forward_fn: &mut F,
    config: &SamplingConfig,
    seed: u64,
) -> Vec<usize>
where
    S: Scalar,
    F: FnMut(&[usize]) -> Tensor<S>,
{
    let mut sampler = Sampler::with_seed(config.clone(), seed);
    let mut tokens: Vec<usize> = prompt.to_vec();

    for _ in 0..config.max_tokens {
        let logits = forward_fn(&tokens);
        let next = sampler.sample(&logits, &tokens);

        if let Some(stop) = config.stop_token {
            if next == stop {
                break;
            }
        }

        tokens.push(next);
    }

    tokens
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn greedy_sampling() {
        let config = SamplingConfig::greedy();
        let mut sampler = Sampler::new(config);

        // Logits: token 2 has the highest value
        let logits = Tensor::from_slice(&[1.0_f64, 3.0, 10.0, 2.0]);
        let token = sampler.sample(&logits, &[]);
        assert_eq!(token, 2);
    }

    #[test]
    fn greedy_sampling_repeated() {
        let config = SamplingConfig::greedy();
        let mut sampler = Sampler::new(config);

        let logits = Tensor::from_slice(&[1.0_f64, 5.0, 3.0]);
        // Should always return 1 (greedy)
        for _ in 0..10 {
            assert_eq!(sampler.sample(&logits, &[]), 1);
        }
    }

    #[test]
    fn temperature_zero_is_greedy() {
        let config = SamplingConfig {
            temperature: 0.0,
            ..Default::default()
        };
        let mut sampler = Sampler::new(config);
        let logits = Tensor::from_slice(&[1.0_f64, 2.0, 5.0, 3.0]);
        assert_eq!(sampler.sample(&logits, &[]), 2);
    }

    #[test]
    fn top_k_filters() {
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: 2,
            top_p: 1.0,
            ..Default::default()
        };
        let mut sampler = Sampler::with_seed(config, 12345);

        // With top_k=2, only the top 2 logits should be sampled
        let logits = Tensor::from_slice(&[1.0_f64, 100.0, 0.1, 99.0]);

        // Sample many times: should only get tokens 1 or 3
        let mut seen = [false; 4];
        for _ in 0..100 {
            let tok = sampler.sample(&logits, &[]);
            seen[tok] = true;
        }
        assert!(!seen[0], "token 0 should not be sampled with top_k=2");
        assert!(!seen[2], "token 2 should not be sampled with top_k=2");
        assert!(seen[1], "token 1 should be sampled");
        assert!(seen[3], "token 3 should be sampled");
    }

    #[test]
    fn top_p_filters() {
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.5,
            ..Default::default()
        };
        let mut sampler = Sampler::with_seed(config, 99);

        // One token dominates: softmax([10, 0, 0, 0]) ≈ [0.9999, ...]
        let logits = Tensor::from_slice(&[10.0_f64, 0.0, 0.0, 0.0]);

        // With top_p=0.5, should basically always sample token 0
        for _ in 0..20 {
            assert_eq!(sampler.sample(&logits, &[]), 0);
        }
    }

    #[test]
    fn repetition_penalty() {
        let config = SamplingConfig {
            temperature: 0.0, // greedy
            repetition_penalty: 100.0, // extreme penalty
            ..Default::default()
        };
        let mut sampler = Sampler::new(config);

        // Without penalty, token 0 wins (logit=5.0)
        let logits = Tensor::from_slice(&[5.0_f64, 4.9, 4.8]);
        let past = vec![0]; // token 0 was already generated

        // With huge penalty on token 0, should pick token 1 instead
        let token = sampler.sample(&logits, &past);
        assert_eq!(token, 1);
    }

    #[test]
    fn generate_with_stop_token() {
        let config = SamplingConfig {
            temperature: 0.0,
            stop_token: Some(2),
            max_tokens: 100,
            ..Default::default()
        };

        let mut call_count = 0usize;
        let generated = generate::<f64, _>(
            &[0],
            &mut |_tokens: &[usize]| -> Tensor<f64> {
                call_count += 1;
                if call_count <= 3 {
                    // Return logits favoring token 1
                    Tensor::from_slice(&[0.0, 10.0, 0.0])
                } else {
                    // Return logits favoring stop token 2
                    Tensor::from_slice(&[0.0, 0.0, 10.0])
                }
            },
            &config,
            42,
        );

        // Should be: [0, 1, 1, 1] then stop (token 2 not included)
        assert_eq!(generated, vec![0, 1, 1, 1]);
    }

    #[test]
    fn generate_max_tokens() {
        let config = SamplingConfig {
            temperature: 0.0,
            max_tokens: 5,
            ..Default::default()
        };

        let generated = generate::<f64, _>(
            &[0],
            &mut |_tokens: &[usize]| -> Tensor<f64> {
                Tensor::from_slice(&[0.0, 10.0, 0.0])
            },
            &config,
            42,
        );

        // prompt(1) + generated(5) = 6
        assert_eq!(generated.len(), 6);
        assert_eq!(generated, vec![0, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn sampler_produces_variety_with_temperature() {
        let config = SamplingConfig {
            temperature: 2.0, // high temperature = more random
            top_k: 0,
            top_p: 1.0,
            ..Default::default()
        };
        let mut sampler = Sampler::with_seed(config, 777);

        // Equal-ish logits: should produce variety
        let logits = Tensor::from_slice(&[1.0_f64, 1.0, 1.0, 1.0]);
        let mut seen = [false; 4];
        for _ in 0..100 {
            let tok = sampler.sample(&logits, &[]);
            seen[tok] = true;
        }
        let num_seen = seen.iter().filter(|&&s| s).count();
        assert!(num_seen >= 3, "high temperature should produce variety, saw {} of 4", num_seen);
    }
}
