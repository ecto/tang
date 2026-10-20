//! How long a batched step takes, fitted from the steps the engine runs, so the scheduler can
//! size prefill chunks to a time budget (see `docs/batching.md`).
//!
//! A step's seconds are modelled as
//!
//! ```text
//! t = a + f·r + Σ_chunks (g_k + b_k·q + d_k·q·c + e_k·q²)
//! ```
//!
//! for `r` decode rows (one-token and verify-window rows) and prefill chunks of `q` new tokens
//! after `c` already in the cache. `k` is the chunk's regime: up to [`SMALL`] tokens (column
//! GEMM matmuls, split-KV attention), or more (dequantize-and-GEMM matmuls, tiled prefill
//! attention), which on CUDA cost very differently: a 33-token chunk takes twice a 32-token
//! one. `g` is a chunk's matmuls reading the weights again, `b` its per-token work, `d` its
//! attention over the prefix, `e` what grows with its square. Tokens are counted in thousands.
//!
//! The fit is least squares with older steps decayed, ridged toward a prior (a calibration at
//! load, or defaults) so a run of identical steps can't send the terms it doesn't constrain
//! anywhere.

use serde_json::{json, Value};

const N: usize = 10;

/// The largest chunk of the small regime.
pub const SMALL: usize = 32;
/// How fast old steps fade (per observed step).
const DECAY: f64 = 0.995;
/// How much the prior weighs, in steps.
const PRIOR: f64 = 0.05;

/// One step's shape.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Shape {
    pub decode_rows: usize,
    /// Per regime (small, large): chunks, Σq, Σq·c, Σq² (tokens, tokens²).
    pub chunks: [usize; 2],
    pub q: [f64; 2],
    pub qc: [f64; 2],
    pub q2: [f64; 2],
}

impl Shape {
    /// Add a chunk of `q` new tokens after `c` cached ones.
    pub fn chunk(mut self, q: usize, c: usize) -> Self {
        let k = (q > SMALL) as usize;
        let (q, c) = (q as f64, c as f64);
        self.chunks[k] += 1;
        self.q[k] += q;
        self.qc[k] += q * c;
        self.q2[k] += q * q;
        self
    }

    /// Prefill chunks in it.
    pub fn prefills(&self) -> usize {
        self.chunks[0] + self.chunks[1]
    }

    fn features(&self) -> [f64; N] {
        let k = 1e-3;
        [
            1.0,
            self.decode_rows as f64,
            self.chunks[0] as f64,
            self.q[0] * k,
            self.qc[0] * k * k,
            self.chunks[1] as f64,
            self.q[1] * k,
            self.qc[1] * k * k,
            self.q2[1] * k * k,
            self.q2[0] * k * k,
        ]
    }
}

/// The fitted step-time model.
#[derive(Debug, Clone)]
pub struct StepCost {
    prior: [f64; N],
    /// Decayed normal equations: Σ x·xᵀ and Σ x·t.
    xx: [[f64; N]; N],
    xt: [f64; N],
    /// Decayed sample count.
    weight: f64,
    coef: [f64; N],
    pub samples: u64,
}

impl Default for StepCost {
    /// A guess for a mid-sized 4-bit model on a discrete GPU (refined as steps run): 10 ms a
    /// decode step and 0.5 ms a row; a small chunk 30 ms and 1 ms a token, a large one 120 ms
    /// and 1 ms a token, with attention terms.
    fn default() -> Self {
        Self::with_prior([0.010, 0.0005, 0.030, 1.0, 0.1, 0.120, 1.0, 0.5, 0.1, 0.0])
    }
}

impl StepCost {
    /// Start from `prior` (seconds; see the module docs for the terms' units).
    pub fn with_prior(prior: [f64; N]) -> Self {
        Self {
            prior,
            xx: [[0.0; N]; N],
            xt: [0.0; N],
            weight: 0.0,
            coef: prior,
            samples: 0,
        }
    }

    /// The coefficients now: a, f; small chunks' g, b, d; large chunks' g, b, d, e; small e.
    pub fn coef(&self) -> [f64; N] {
        self.coef
    }

    /// Make the current fit the prior (after a calibration) and forget the samples.
    pub fn settle(&mut self) {
        *self = Self::with_prior(self.coef.map(|c| c.max(0.0)));
    }

    /// Predicted seconds for a step of this shape.
    pub fn predict(&self, s: &Shape) -> f64 {
        dot(&self.coef, &s.features())
    }

    /// A step of shape `s` took `secs`.
    pub fn observe(&mut self, s: &Shape, secs: f64) {
        if !secs.is_finite() || secs <= 0.0 {
            return;
        }
        let x = s.features();
        for i in 0..N {
            for j in 0..N {
                self.xx[i][j] = self.xx[i][j] * DECAY + x[i] * x[j];
            }
            self.xt[i] = self.xt[i] * DECAY + x[i] * secs;
        }
        self.weight = self.weight * DECAY + 1.0;
        self.samples += 1;
        self.refit();
    }

    fn refit(&mut self) {
        // (Σxxᵀ + M) θ = Σxt + M θ0, M diagonal: each term's prior weighs as much as PRIOR
        // average steps' worth of that feature (a term no step has exercised stays put).
        let mut a = self.xx;
        let mut y = self.xt;
        for i in 0..N {
            let mu = PRIOR * a[i][i] / self.weight.max(1.0) + 1e-12;
            a[i][i] += mu;
            y[i] += mu * self.prior[i];
        }
        if let Some(c) = solve(a, y) {
            // Every term costs time; a negative one is noise.
            self.coef = c.map(|v| v.max(0.0));
        }
    }

    /// The most new tokens (up to `most`) a chunk after `c` cached ones can take in `secs` of
    /// a step on top of what's already in it (its own fixed cost included): the larger of what
    /// fits in either regime. 0 if no chunk fits.
    pub fn fit_chunk(&self, secs: f64, c: usize, most: usize) -> usize {
        let x = &self.coef;
        let small = solve_q(secs - x[2], x[3], x[4], x[9], c)
            .min(SMALL)
            .min(most);
        let large = if most > SMALL {
            let q = solve_q(secs - x[5], x[6], x[7], x[8], c).min(most);
            if q > SMALL {
                q
            } else {
                0
            }
        } else {
            0
        };
        small.max(large)
    }

    pub fn json(&self) -> Value {
        let x = self.coef.map(|v| v * 1e3);
        json!({
            "model": "secs = a + f*decode_rows + sum over prefill chunks of (g + b*q + d*q*c + e*q^2), \
                      by chunk regime (small: q <= 32); q, c in thousands of tokens; values in ms",
            "a": x[0],
            "f": x[1],
            "small": { "g": x[2], "b": x[3], "d": x[4], "e": x[9] },
            "large": { "g": x[5], "b": x[6], "d": x[7], "e": x[8] },
            "samples": self.samples,
        })
    }
}

/// The largest q with e·q² + (b + d·c)·q ≤ rem (q, c in thousands; returns tokens).
fn solve_q(rem: f64, b: f64, d: f64, e: f64, c: usize) -> usize {
    if rem <= 0.0 {
        return 0;
    }
    let bq = (b + d * c as f64 * 1e-3) * 1e-3;
    let eq = e * 1e-6;
    let q = if eq > 0.0 {
        (-bq + (bq * bq + 4.0 * eq * rem).sqrt()) / (2.0 * eq)
    } else if bq > 0.0 {
        rem / bq
    } else {
        f64::INFINITY
    };
    if q.is_finite() {
        q.max(0.0) as usize
    } else {
        usize::MAX
    }
}

fn dot(a: &[f64; N], b: &[f64; N]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Gaussian elimination with partial pivoting.
fn solve(mut a: [[f64; N]; N], mut y: [f64; N]) -> Option<[f64; N]> {
    for col in 0..N {
        let p = (col..N).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[p][col].abs() < 1e-300 {
            return None;
        }
        a.swap(col, p);
        y.swap(col, p);
        for r in col + 1..N {
            let f = a[r][col] / a[col][col];
            for k in col..N {
                a[r][k] -= f * a[col][k];
            }
            y[r] -= f * y[col];
        }
    }
    let mut x = [0.0; N];
    for r in (0..N).rev() {
        let s: f64 = (r + 1..N).map(|k| a[r][k] * x[k]).sum();
        x[r] = (y[r] - s) / a[r][r];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Roughly Qwen3-8B 4-bit on an RTX 3090.
    const TRUE: [f64; N] = [0.0086, 0.0011, 0.030, 0.6, 0.18, 0.120, 0.75, 1.3, 0.2, 0.0];

    fn shape(r: usize, chunks: &[(usize, usize)]) -> Shape {
        chunks.iter().fold(
            Shape {
                decode_rows: r,
                ..Shape::default()
            },
            |s, &(q, c)| s.chunk(q, c),
        )
    }

    #[test]
    fn fits_the_terms_from_varied_steps() {
        let truth = StepCost::with_prior(TRUE);
        let mut m = StepCost::default();
        for i in 0..600usize {
            let r = i % 9;
            let chunks: Vec<(usize, usize)> = match i % 5 {
                0 => vec![],
                1 => vec![(9 + i % 24, (i * 37) % 16_000)],
                2 => vec![(33 + i % 480, (i * 101) % 8_000)],
                3 => vec![(512, (i * 13) % 4_000), (16, 0)],
                _ => vec![(64 + i % 64, (i * 7) % 2_000)],
            };
            let s = shape(r, &chunks);
            m.observe(&s, truth.predict(&s));
        }
        // Predictions for steps it hasn't seen are within a few percent.
        for (r, chunks) in [
            (4, vec![]),
            (8, vec![(256, 12_000)]),
            (1, vec![(500, 0)]),
            (6, vec![(20, 3_000), (128, 9_000)]),
            (2, vec![(32, 15_000)]),
        ] {
            let s = shape(r, &chunks);
            let (got, want) = (m.predict(&s), truth.predict(&s));
            assert!(
                (got - want).abs() < 0.03 * want,
                "{got} vs {want}: {:?}",
                m.coef()
            );
        }
    }

    #[test]
    fn a_chunk_is_as_big_as_the_budget_allows() {
        let m = StepCost::with_prior(TRUE);
        let cost = |q: usize, c: usize| m.predict(&shape(0, &[(q, c)])) - TRUE[0];
        for (budget, c) in [
            (0.050, 0),
            (0.050, 16_000),
            (0.3, 0),
            (0.3, 8_000),
            (2.0, 4_000),
        ] {
            let q = m.fit_chunk(budget, c, 512);
            assert!(q > 0, "nothing fits {budget} at {c}");
            assert!(cost(q, c) <= budget + 1e-9, "q {q} at c {c}");
            // Nothing bigger fits (in either regime).
            assert!((q + 1..=512).all(|k| cost(k, c) > budget), "q {q} at c {c}");
        }
        // Small budgets get a small-regime chunk, never a large one; longer prefixes, fewer
        // tokens; a budget below a chunk's own cost, none.
        assert!(m.fit_chunk(0.050, 0, 512) <= SMALL);
        assert!(m.fit_chunk(0.3, 8_000, 512) < m.fit_chunk(0.3, 0, 512));
        assert_eq!(m.fit_chunk(0.020, 0, 512), 0);
        assert_eq!(m.fit_chunk(10.0, 0, 100), 100);
    }

    #[test]
    fn the_prior_holds_terms_the_steps_dont_constrain() {
        let mut m = StepCost::with_prior(TRUE);
        // Only plain decode steps of one width: nothing says what a chunk costs.
        for _ in 0..1000 {
            m.observe(&shape(4, &[]), 0.015);
        }
        let c = m.coef();
        assert!((c[0] + 4.0 * c[1] - 0.015).abs() < 1e-4, "{c:?}");
        for i in 2..N {
            assert!((c[i] - TRUE[i]).abs() <= 0.1 * TRUE[i], "{c:?}");
        }
    }
}
