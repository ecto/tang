//! Tensor trains: TT-SVD with error-budget / Gavish–Donoho truncation, and
//! full contraction.
//!
//! A tensor `X(i₁,…,i_d)` is written `G₁(i₁)·G₂(i₂)⋯G_d(i_d)` with core
//! `G_k` of shape `r_{k−1} × n_k × r_k` (`r₀ = r_d = 1`), stored row-major
//! as the `(r_{k−1}·n_k) × r_k` matrix. [`tt_svd`] returns the **canonical
//! left-orthogonal gauge**: every core but the last has orthonormal columns
//! as a `(r_{k−1}n_k) × r_k` matrix, so the whole norm `‖X‖_F` lives in
//! `G_d`, and the TT is unique up to an orthogonal `r_k × r_k` rotation per
//! bond (fixed here by the SVD's singular-vector basis, i.e. the rotation that
//! diagonalises each left-unfolding Gram).

use crate::linalg::{matmul, matmul_tn, svd_left};

/// A tensor train in row-major core storage.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorTrain {
    /// Mode sizes `n_k`.
    pub modes: Vec<usize>,
    /// Bond ranks `r₀..r_d` (`r₀ = r_d = 1`).
    pub ranks: Vec<usize>,
    /// Cores, `G_k` as `(r_{k−1}·n_k) × r_k` row-major.
    pub cores: Vec<Vec<f64>>,
}

impl TensorTrain {
    /// Number of stored parameters.
    pub fn num_params(&self) -> usize {
        self.cores.iter().map(|c| c.len()).sum()
    }

    /// Contract to the full tensor (length `∏ n_k`), left to right.
    pub fn full(&self) -> Vec<f64> {
        contract(&self.modes, &self.ranks, &self.cores)
    }
}

/// Contract cores to the full tensor. Each step is one GEMM:
/// `P (M × r_{k−1}) · G_k (r_{k−1} × n_k r_k)`, and the row-major result
/// *is* `(M n_k) × r_k`.
pub fn contract(modes: &[usize], ranks: &[usize], cores: &[Vec<f64>]) -> Vec<f64> {
    let d = modes.len();
    let mut p = cores[0].clone(); // (1·n₁) × r₁
    let mut rows = modes[0];
    for k in 1..d {
        let r = ranks[k];
        let cols = modes[k] * ranks[k + 1];
        p = matmul(&p, &cores[k], rows, r, cols);
        rows *= modes[k];
    }
    p
}

/// Rank-selection rule for each unfolding in [`tt_svd`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Truncation {
    /// Relative Frobenius budget `ε`: each of the `d−1` SVDs discards at most
    /// `δ = ε‖X‖/√(d−1)` (Oseledets 2011), so `‖X − X̂‖ ≤ ε‖X‖`.
    pub rel_eps: f64,
    /// Additionally cap each rank at the Gavish–Donoho optimal hard
    /// threshold `ω(β)·σ_med` (unknown-noise form).
    pub gavish_donoho: bool,
    /// Hard rank cap.
    pub max_rank: usize,
}

impl Default for Truncation {
    fn default() -> Self {
        Self {
            rel_eps: 1e-12,
            gavish_donoho: false,
            max_rank: usize::MAX,
        }
    }
}

/// Gavish–Donoho (2014) optimal hard threshold coefficient for unknown noise,
/// `ω(β) ≈ 0.56β³ − 0.95β² + 1.82β + 1.43`, `β = min(m,n)/max(m,n)`.
pub fn gd_omega(beta: f64) -> f64 {
    0.56 * beta.powi(3) - 0.95 * beta.powi(2) + 1.82 * beta + 1.43
}

/// Number of singular values above the Gavish–Donoho threshold for an
/// `m × n` matrix with spectrum `s` (descending, length `min(m,n)`).
///
/// The rule needs a bulk of noise singular values to estimate the median
/// from; QTT's leading unfoldings are `2 × 2^{d−1}`, `4 × …` (two or four
/// singular values), where the median is a *signal* value and the cap
/// collapses the rank to 1 (measured: Miranda 64×128×128 at a 60 dB target
/// fell to 12 dB). Unfoldings with `min(m,n) < GD_MIN_DIM` are left uncapped.
pub fn gd_rank(s: &[f64], m: usize, n: usize) -> usize {
    if s.is_empty() {
        return 0;
    }
    if m.min(n) < GD_MIN_DIM {
        return s.len();
    }
    let beta = m.min(n) as f64 / m.max(n) as f64;
    let mut sorted = s.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let len = sorted.len();
    let med = if len % 2 == 1 {
        sorted[len / 2]
    } else {
        0.5 * (sorted[len / 2 - 1] + sorted[len / 2])
    };
    let tau = gd_omega(beta) * med;
    s.iter().filter(|&&x| x > tau).count().max(1)
}

/// Smallest unfolding side on which [`gd_rank`] applies its cap.
pub const GD_MIN_DIM: usize = 32;

/// Smallest rank whose discarded tail energy `Σ_{j≥r} σ_j²` is ≤ `delta2`.
pub fn tail_rank(s: &[f64], delta2: f64) -> usize {
    let mut tail = 0.0;
    let mut r = s.len();
    while r > 1 {
        let t = tail + s[r - 1] * s[r - 1];
        if t > delta2 {
            break;
        }
        tail = t;
        r -= 1;
    }
    r
}

/// TT-SVD (Oseledets 2011) of `x` reshaped to `modes`, left-orthogonal gauge.
pub fn tt_svd(x: &[f64], modes: &[usize], trunc: Truncation) -> TensorTrain {
    let d = modes.len();
    let n: usize = modes.iter().product();
    assert_eq!(x.len(), n, "tt_svd: data length != product of modes");
    let norm2: f64 = x.iter().map(|v| v * v).sum();
    let delta2 = if d > 1 {
        trunc.rel_eps * trunc.rel_eps * norm2 / (d - 1) as f64
    } else {
        0.0
    };
    let mut ranks = vec![1usize];
    let mut cores = Vec::with_capacity(d);
    let mut c = x.to_vec();
    let mut r_prev = 1;
    let mut cols = n;
    for &nk in modes.iter().take(d - 1) {
        let m = r_prev * nk;
        cols /= nk;
        let (_s, u, r) = svd_left(&c, m, cols, |s| {
            let mut r = tail_rank(s, delta2);
            if trunc.gavish_donoho {
                r = r.min(gd_rank(s, m, cols));
            }
            r.min(trunc.max_rank).max(1)
        });
        // Remainder Uᵀ C (r × cols) = Σ Vᵀ.
        c = matmul_tn(&u, &c, m, r, cols);
        cores.push(u);
        ranks.push(r);
        r_prev = r;
    }
    cores.push(c);
    ranks.push(1);
    TensorTrain {
        modes: modes.to_vec(),
        ranks,
        cores,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel_err(a: &[f64], b: &[f64]) -> f64 {
        let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
        let den: f64 = a.iter().map(|x| x * x).sum();
        (num / den).sqrt()
    }

    #[test]
    fn exact_rank_of_exponential_and_sine() {
        // exp(a i) has QTT rank 1; sin(a i) rank 2 (Khoromskij).
        let n = 1 << 12;
        let modes = vec![2; 12];
        let e: Vec<f64> = (0..n).map(|i| (0.001 * i as f64).exp()).collect();
        // ε = 1e-6: the Gram-based SVD resolves σ only to ~1e-7·σ₁ (crate docs).
        let tt = tt_svd(
            &e,
            &modes,
            Truncation {
                rel_eps: 1e-6,
                ..Default::default()
            },
        );
        assert!(tt.ranks.iter().all(|&r| r == 1), "{:?}", tt.ranks);
        let s: Vec<f64> = (0..n).map(|i| (0.01 * i as f64 + 0.3).sin()).collect();
        let tt = tt_svd(
            &s,
            &modes,
            Truncation {
                rel_eps: 1e-6,
                ..Default::default()
            },
        );
        assert!(tt.ranks.iter().all(|&r| r <= 2), "{:?}", tt.ranks);
        assert!(rel_err(&s, &tt.full()) < 1e-6);
    }

    #[test]
    fn error_budget_holds_and_gauge_is_left_orthogonal() {
        let modes = vec![2usize; 14];
        let n = 1 << 14;
        let mut seed = 1u64;
        let x: Vec<f64> = (0..n)
            .map(|i| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let t = i as f64 / n as f64;
                (20.0 * t).sin() * (-3.0 * t).exp() + 0.01 * ((seed >> 33) as f64 / 2e9 - 1.0)
            })
            .collect();
        for eps in [1e-1, 1e-2, 1e-3] {
            let tt = tt_svd(
                &x,
                &modes,
                Truncation {
                    rel_eps: eps,
                    ..Default::default()
                },
            );
            assert!(rel_err(&x, &tt.full()) <= eps * (1.0 + 1e-9));
            for k in 0..modes.len() - 1 {
                let (m, r) = (tt.ranks[k] * modes[k], tt.ranks[k + 1]);
                let g = crate::linalg::gram_cols(&tt.cores[k], m, r);
                for i in 0..r {
                    for j in 0..r {
                        let want = if i == j { 1.0 } else { 0.0 };
                        assert!((g[i * r + j] - want).abs() < 1e-8);
                    }
                }
            }
            let last = tt.cores.last().unwrap();
            let nl: f64 = last.iter().map(|v| v * v).sum();
            let nx: f64 = tt.full().iter().map(|v| v * v).sum();
            assert!((nl - nx).abs() < 1e-8 * nx, "norm lives in the last core");
        }
    }

    #[test]
    fn gavish_donoho_cuts_noise_rank() {
        // rank-1 signal + noise in a 64×64 unfolding: GD keeps ~1.
        let modes = vec![64usize, 64];
        let mut seed = 3u64;
        let x: Vec<f64> = (0..4096)
            .map(|i| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let (a, b) = ((i / 64) as f64, (i % 64) as f64);
                (a / 10.0).sin() * (b / 7.0).cos() * 10.0
                    + 0.1 * ((seed >> 33) as f64 / 2147483648.0 - 0.5)
            })
            .collect();
        let tt = tt_svd(
            &x,
            &modes,
            Truncation {
                rel_eps: 1e-12,
                gavish_donoho: true,
                ..Default::default()
            },
        );
        assert_eq!(tt.ranks[1], 1);
    }
}
