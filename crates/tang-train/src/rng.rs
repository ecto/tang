/// A simple LCG-based pseudo-random number generator.
///
/// Uses the same multiplier and increment as the existing inline LCGs
/// scattered throughout the crate, but centralised into a reusable struct.
///
/// ```
/// # use tang_train::Rng;
/// let mut rng = Rng::new(42);
/// let u = rng.next_f64(); // uniform in [0, 1)
/// let n = rng.normal();   // standard normal
/// ```
pub struct Rng {
    state: u64,
}

const MULTIPLIER: u64 = 6364136223846793005;
const INCREMENT: u64 = 1442695040888963407;

impl Rng {
    /// Create a new RNG from the given seed.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advance the LCG and return the next raw `u64`.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(MULTIPLIER)
            .wrapping_add(INCREMENT);
        self.state
    }

    /// Uniform random `f64` in [0, 1) with 53 bits of mantissa.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal (mean 0, variance 1) via Box-Muller transform.
    pub fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15); // avoid log(0)
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
    }

    /// Returns `true` with probability `p`.
    pub fn bernoulli(&mut self, p: f64) -> bool {
        self.next_f64() < p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_mean_approx_half() {
        let mut rng = Rng::new(12345);
        let n = 10_000;
        let sum: f64 = (0..n).map(|_| rng.next_f64()).sum();
        let mean = sum / n as f64;
        assert!(
            (mean - 0.5).abs() < 0.02,
            "uniform mean {mean} too far from 0.5"
        );
    }

    #[test]
    fn uniform_in_range() {
        let mut rng = Rng::new(99);
        for _ in 0..10_000 {
            let v = rng.next_f64();
            assert!((0.0..1.0).contains(&v), "value {v} out of [0, 1)");
        }
    }

    #[test]
    fn normal_mean_approx_zero() {
        let mut rng = Rng::new(7777);
        let n = 10_000;
        let sum: f64 = (0..n).map(|_| rng.normal()).sum();
        let mean = sum / n as f64;
        assert!(
            mean.abs() < 0.05,
            "normal mean {mean} too far from 0.0"
        );
    }

    #[test]
    fn normal_variance_approx_one() {
        let mut rng = Rng::new(31415);
        let n = 10_000;
        let samples: alloc::vec::Vec<f64> = (0..n).map(|_| rng.normal()).collect();
        let mean: f64 = samples.iter().sum::<f64>() / n as f64;
        let var: f64 = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        assert!(
            (var - 1.0).abs() < 0.1,
            "normal variance {var} too far from 1.0"
        );
    }

    #[test]
    fn bernoulli_rate() {
        let mut rng = Rng::new(555);
        let n = 10_000;
        let hits = (0..n).filter(|_| rng.bernoulli(0.3)).count();
        let rate = hits as f64 / n as f64;
        assert!(
            (rate - 0.3).abs() < 0.03,
            "bernoulli rate {rate} too far from 0.3"
        );
    }

    #[test]
    fn deterministic() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }
}
