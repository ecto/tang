use tang::Scalar;
use crate::{DVec, DMat};
use alloc::vec::Vec;

/// Singular Value Decomposition: A = U * Σ * V^T
///
/// Golub-Kahan bidiagonalization + implicit QR shifts.
pub struct Svd<S> {
    /// Left singular vectors (m × k), k = min(m, n).
    pub u: DMat<S>,
    /// Singular values (k), sorted descending.
    pub s: DVec<S>,
    /// Right singular vectors (n × k).
    pub vt: DMat<S>,
}

impl<S: Scalar> Svd<S> {
    /// Compute the thin SVD of an m×n matrix.
    ///
    /// Uses the one-sided Jacobi rotation method for robustness.
    /// Not the fastest for large matrices — use the faer bridge for that.
    pub fn new(a: &DMat<S>) -> Self {
        let m = a.nrows();
        let n = a.ncols();

        if m >= n {
            Self::compute_tall(a)
        } else {
            // For wide matrices, compute SVD of A^T then swap U, V
            let at = a.transpose();
            let svd = Self::compute_tall(&at);
            Svd {
                u: svd.vt.transpose(),
                s: svd.s,
                vt: svd.u.transpose(),
            }
        }
    }

    /// SVD for m >= n case via one-sided Jacobi.
    fn compute_tall(a: &DMat<S>) -> Self {
        let m = a.nrows();
        let n = a.ncols();
        assert!(m >= n);

        // Start with A = U_0, V = I
        let mut u = a.clone();
        let mut v = DMat::<S>::identity(n);

        let max_iter = 100 * n * n;
        let tol = S::EPSILON * S::from_i32(10);

        for _ in 0..max_iter {
            let mut converged = true;

            // Sweep all pairs (p, q) with p < q
            for p in 0..n {
                for q in (p + 1)..n {
                    // Compute 2x2 gram matrix entries
                    let mut app = S::ZERO;
                    let mut aqq = S::ZERO;
                    let mut apq = S::ZERO;
                    for i in 0..m {
                        let up = u.get(i, p);
                        let uq = u.get(i, q);
                        app += up * up;
                        aqq += uq * uq;
                        apq += up * uq;
                    }

                    // Skip if already orthogonal
                    if apq.abs() < tol * (app * aqq).sqrt() {
                        continue;
                    }
                    converged = false;

                    // Compute Jacobi rotation angle
                    let tau = (aqq - app) / (S::TWO * apq);
                    let t = if tau >= S::ZERO {
                        (tau + (S::ONE + tau * tau).sqrt()).recip()
                    } else {
                        -((-tau) + (S::ONE + tau * tau).sqrt()).recip()
                    };
                    let c = (S::ONE + t * t).sqrt().recip();
                    let s = t * c;

                    // Apply rotation to U columns p, q
                    for i in 0..m {
                        let up = u.get(i, p);
                        let uq = u.get(i, q);
                        u.set(i, p, c * up - s * uq);
                        u.set(i, q, s * up + c * uq);
                    }

                    // Apply rotation to V columns p, q
                    for i in 0..n {
                        let vp = v.get(i, p);
                        let vq = v.get(i, q);
                        v.set(i, p, c * vp - s * vq);
                        v.set(i, q, s * vp + c * vq);
                    }
                }
            }

            if converged { break; }
        }

        // Extract singular values = norms of U columns
        let mut sigma = Vec::with_capacity(n);
        for j in 0..n {
            let mut norm_sq = S::ZERO;
            for i in 0..m {
                norm_sq += u.get(i, j) * u.get(i, j);
            }
            let norm = norm_sq.sqrt();
            sigma.push(norm);
            if norm > S::EPSILON {
                let inv = norm.recip();
                for i in 0..m {
                    let v = u.get(i, j) * inv;
                    u.set(i, j, v);
                }
            }
        }

        // Sort singular values descending
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| sigma[b].partial_cmp(&sigma[a]).unwrap_or(core::cmp::Ordering::Equal));

        let s = DVec::from_fn(n, |i| sigma[order[i]]);
        let u_sorted = DMat::from_fn(m, n, |i, j| u.get(i, order[j]));
        let vt_sorted = DMat::from_fn(n, n, |i, j| v.get(j, order[i]));

        Svd { u: u_sorted, s, vt: vt_sorted }
    }

    /// Rank (number of significant singular values).
    pub fn rank(&self, tol: S) -> usize {
        self.s.iter().filter(|&&s| s > tol).count()
    }

    /// Pseudoinverse: A⁺ = V Σ⁺ U^T
    pub fn pseudoinverse(&self, tol: S) -> DMat<S> {
        let k = self.s.len();
        let m = self.u.nrows();
        let n = self.vt.ncols();

        // Σ⁺: invert non-tiny singular values
        let s_inv = DVec::from_fn(k, |i| {
            if self.s[i] > tol { self.s[i].recip() } else { S::ZERO }
        });

        // V * diag(s_inv) * U^T
        let ut = self.u.transpose();
        let v = self.vt.transpose();
        let mut result = DMat::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                let mut sum = S::ZERO;
                for l in 0..k {
                    sum += v.get(i, l) * s_inv[l] * ut.get(l, j);
                }
                result.set(i, j, sum);
            }
        }
        result
    }

    /// Reconstruct the original matrix: U * diag(s) * V^T
    pub fn reconstruct(&self) -> DMat<S> {
        let m = self.u.nrows();
        let n = self.vt.ncols();
        let k = self.s.len();
        let mut result = DMat::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                let mut sum = S::ZERO;
                for l in 0..k {
                    sum += self.u.get(i, l) * self.s[l] * self.vt.get(l, j);
                }
                result.set(i, j, sum);
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn svd_identity() {
        let a = DMat::<f64>::identity(3);
        let svd = Svd::new(&a);
        // Singular values should be [1, 1, 1]
        for i in 0..3 {
            assert!((svd.s[i] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn svd_reconstruct() {
        let a = DMat::from_fn(3, 2, |i, j| (i * 2 + j + 1) as f64);
        let svd = Svd::new(&a);
        let recon = svd.reconstruct();
        for i in 0..3 {
            for j in 0..2 {
                assert!((recon.get(i, j) - a.get(i, j)).abs() < 1e-8,
                    "mismatch at ({}, {}): {} vs {}", i, j, recon.get(i, j), a.get(i, j));
            }
        }
    }

    #[test]
    fn svd_rank_deficient() {
        // Rank-1 matrix: [1 2; 2 4; 3 6]
        let a = DMat::from_fn(3, 2, |i, _j| (i + 1) as f64);
        let b = DMat::from_fn(3, 2, |_i, j| (j + 1) as f64);
        let _m = a.mul_mat(&b.transpose()).submatrix(0, 0, 3, 2);
        // Actually, let's use a clearer rank-1 example
        let r1 = DMat::from_fn(3, 2, |i, j| {
            [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]][i][j]
        });
        let svd = Svd::new(&r1);
        assert!(svd.rank(1e-10) == 1, "rank should be 1, got {}", svd.rank(1e-10));
    }

    #[test]
    fn svd_wide_matrix() {
        let a = DMat::from_fn(2, 3, |i, j| (i * 3 + j + 1) as f64);
        let svd = Svd::new(&a);
        let recon = svd.reconstruct();
        for i in 0..2 {
            for j in 0..3 {
                assert!((recon.get(i, j) - a.get(i, j)).abs() < 1e-8,
                    "mismatch at ({}, {}): {} vs {}", i, j, recon.get(i, j), a.get(i, j));
            }
        }
    }

    #[test]
    fn pseudoinverse() {
        let a = DMat::from_fn(3, 2, |i, j| {
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]][i][j]
        });
        let svd = Svd::new(&a);
        let pinv = svd.pseudoinverse(1e-10);
        // A⁺ * A should be I(2×2)
        let prod = pinv.mul_mat(&a);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((prod.get(i, j) - expected).abs() < 1e-8);
            }
        }
    }
}
