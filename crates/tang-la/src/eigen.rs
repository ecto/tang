use crate::{DMat, DVec};
use alloc::vec;
use alloc::vec::Vec;
use tang::Scalar;

/// Eigendecomposition of a symmetric matrix: A = V * diag(λ) * V^T
///
/// Uses Householder tridiagonalization followed by implicit QR iteration
/// with Wilkinson shifts. Falls back to Jacobi for very small matrices.
pub struct SymmetricEigen<S> {
    /// Eigenvalues, sorted ascending.
    pub eigenvalues: DVec<S>,
    /// Eigenvectors as columns.
    pub eigenvectors: DMat<S>,
}

impl<S: Scalar> SymmetricEigen<S> {
    /// Compute eigendecomposition of a symmetric matrix.
    pub fn new(a: &DMat<S>) -> Self {
        assert!(a.is_square(), "SymmetricEigen: matrix must be square");
        let n = a.nrows();

        if n <= 2 {
            return Self::jacobi(a);
        }

        // Phase 1: Householder tridiagonalization — A = Q T Q^T
        let (mut diag, mut offdiag, mut q) = Self::tridiagonalize(a);

        // Phase 2: Implicit QR iteration on the tridiagonal
        Self::trid_qr(&mut diag, &mut offdiag, &mut q, n);

        // Sort eigenvalues ascending and reorder eigenvectors
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            diag[a]
                .partial_cmp(&diag[b])
                .unwrap_or(core::cmp::Ordering::Equal)
        });

        let eigenvalues = DVec::from_fn(n, |i| diag[indices[i]]);
        let eigenvectors = DMat::from_fn(n, n, |i, j| q.get(i, indices[j]));

        Self {
            eigenvalues,
            eigenvectors,
        }
    }

    /// Classical Jacobi eigenvalue algorithm — robust for small matrices.
    pub fn jacobi(a: &DMat<S>) -> Self {
        assert!(a.is_square(), "SymmetricEigen: matrix must be square");
        let n = a.nrows();

        let mut d = a.clone();
        let mut v = DMat::<S>::identity(n);

        let max_iter = 100 * n * n;
        let tol = S::EPSILON * S::from_i32(10);

        for _ in 0..max_iter {
            let mut max_val = S::ZERO;
            let mut p = 0;
            let mut q = 1;
            for i in 0..n {
                for j in (i + 1)..n {
                    let val = d.get(i, j).abs();
                    if val > max_val {
                        max_val = val;
                        p = i;
                        q = j;
                    }
                }
            }

            if max_val < tol {
                break;
            }

            let app = d.get(p, p);
            let aqq = d.get(q, q);
            let apq = d.get(p, q);

            let theta = (aqq - app) / (S::TWO * apq);
            let t = if theta >= S::ZERO {
                (theta + (S::ONE + theta * theta).sqrt()).recip()
            } else {
                -((-theta) + (S::ONE + theta * theta).sqrt()).recip()
            };
            let c = (S::ONE + t * t).sqrt().recip();
            let s = t * c;

            d.set(p, p, app - t * apq);
            d.set(q, q, aqq + t * apq);
            d.set(p, q, S::ZERO);
            d.set(q, p, S::ZERO);

            for i in 0..n {
                if i == p || i == q {
                    continue;
                }
                let dip = d.get(i, p);
                let diq = d.get(i, q);
                d.set(i, p, c * dip - s * diq);
                d.set(p, i, c * dip - s * diq);
                d.set(i, q, s * dip + c * diq);
                d.set(q, i, s * dip + c * diq);
            }

            for i in 0..n {
                let vip = v.get(i, p);
                let viq = v.get(i, q);
                v.set(i, p, c * vip - s * viq);
                v.set(i, q, s * vip + c * viq);
            }
        }

        let mut eigs: Vec<(S, usize)> = (0..n).map(|i| (d.get(i, i), i)).collect();
        eigs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));

        let eigenvalues = DVec::from_fn(n, |i| eigs[i].0);
        let eigenvectors = DMat::from_fn(n, n, |i, j| v.get(i, eigs[j].1));

        Self {
            eigenvalues,
            eigenvectors,
        }
    }

    /// Householder tridiagonalization of symmetric matrix.
    /// Returns (diagonal, off-diagonal, Q) where Q^T A Q = tridiag(diag, offdiag).
    fn tridiagonalize(a: &DMat<S>) -> (Vec<S>, Vec<S>, DMat<S>) {
        let n = a.nrows();
        let mut a = a.clone();
        let mut q = DMat::<S>::identity(n);

        for k in 0..(n - 2) {
            // Compute the Householder vector for column k, rows k+1..n
            let mut x_norm_sq = S::ZERO;
            for i in (k + 1)..n {
                x_norm_sq += a.get(i, k) * a.get(i, k);
            }

            if x_norm_sq < S::EPSILON * S::EPSILON {
                continue;
            }

            let x_norm = x_norm_sq.sqrt();
            let alpha = if a.get(k + 1, k) >= S::ZERO {
                -x_norm
            } else {
                x_norm
            };

            // Householder vector: v = x - alpha * e1
            // v[k+1] = a[k+1,k] - alpha, v[i] = a[i,k] for i > k+1
            let mut v = vec![S::ZERO; n];
            v[k + 1] = a.get(k + 1, k) - alpha;
            for i in (k + 2)..n {
                v[i] = a.get(i, k);
            }

            // tau = 2 / ||v||^2
            // ||v||^2 = (a[k+1,k] - alpha)^2 + sum_{i>k+1} a[i,k]^2
            //         = a[k+1,k]^2 - 2*alpha*a[k+1,k] + alpha^2 + (x_norm_sq - a[k+1,k]^2)
            //         = x_norm_sq - 2*alpha*a[k+1,k] + alpha^2
            //         = 2*alpha^2 - 2*alpha*a[k+1,k]  (since alpha^2 = x_norm_sq)
            //         = 2*alpha*(alpha - a[k+1,k])
            let v_norm_sq = S::TWO * alpha * (alpha - a.get(k + 1, k));
            if v_norm_sq.abs() < S::EPSILON * S::EPSILON {
                continue;
            }
            let tau = S::TWO / v_norm_sq;

            // Two-sided Householder: A <- H A H where H = I - tau * v * v^T
            // Step 1: p = tau * A * v
            let mut p = vec![S::ZERO; n];
            for i in 0..n {
                let mut sum = S::ZERO;
                for j in (k + 1)..n {
                    sum += a.get(i, j) * v[j];
                }
                p[i] = tau * sum;
            }

            // Step 2: beta = (tau/2) * v^T * p
            let mut beta = S::ZERO;
            for i in (k + 1)..n {
                beta += v[i] * p[i];
            }
            beta = beta * tau * S::HALF;

            // Step 3: w = p - beta * v
            let mut w = vec![S::ZERO; n];
            for i in 0..n {
                w[i] = p[i] - beta * v[i];
            }

            // Step 4: A <- A - v * w^T - w * v^T
            for i in 0..n {
                for j in 0..n {
                    let val = a.get(i, j) - v[i] * w[j] - w[i] * v[j];
                    a.set(i, j, val);
                }
            }

            // Accumulate Q: Q <- Q * H = Q - tau * (Q * v) * v^T
            let mut qv = vec![S::ZERO; n];
            for i in 0..n {
                let mut sum = S::ZERO;
                for j in (k + 1)..n {
                    sum += q.get(i, j) * v[j];
                }
                qv[i] = sum;
            }
            for i in 0..n {
                for j in (k + 1)..n {
                    let val = q.get(i, j) - tau * qv[i] * v[j];
                    q.set(i, j, val);
                }
            }
        }

        // Extract diagonal and off-diagonal
        let diag: Vec<S> = (0..n).map(|i| a.get(i, i)).collect();
        let offdiag: Vec<S> = (0..(n - 1)).map(|i| a.get(i + 1, i)).collect();

        (diag, offdiag, q)
    }

    /// Implicit QR iteration with Wilkinson shifts on symmetric tridiagonal matrix.
    /// diag has length n, offdiag has length n-1.
    /// q accumulates eigenvectors.
    fn trid_qr(diag: &mut Vec<S>, offdiag: &mut Vec<S>, q: &mut DMat<S>, n: usize) {
        let max_iter = 30 * n;

        for _ in 0..max_iter {
            // Find the largest unreduced block [lo..=hi]
            // Scan from the bottom to find hi where offdiag[hi-1] is non-negligible
            let mut hi = n - 1;
            while hi > 0 {
                let tol = (diag[hi - 1].abs() + diag[hi].abs()) * S::EPSILON;
                if offdiag[hi - 1].abs() <= tol {
                    hi -= 1;
                } else {
                    break;
                }
            }
            if hi == 0 {
                break; // All converged
            }

            // Find lo: start of the unreduced block ending at hi
            let mut lo = hi - 1;
            while lo > 0 {
                let tol = (diag[lo - 1].abs() + diag[lo].abs()) * S::EPSILON;
                if offdiag[lo - 1].abs() <= tol {
                    break;
                }
                lo -= 1;
            }

            // Wilkinson shift: eigenvalue of trailing 2x2 closer to diag[hi]
            let d_hi = diag[hi];
            let d_hi1 = diag[hi - 1];
            let e_hi1 = offdiag[hi - 1];
            let delta = (d_hi1 - d_hi) * S::HALF;
            let mu = if delta.abs() < S::EPSILON {
                d_hi - e_hi1.abs()
            } else {
                let sign = if delta >= S::ZERO { S::ONE } else { -S::ONE };
                d_hi - e_hi1 * e_hi1 / (delta + sign * (delta * delta + e_hi1 * e_hi1).sqrt())
            };

            // Implicit QR step: chase bulge from lo to hi
            let mut x = diag[lo] - mu;
            let mut z = offdiag[lo];

            for k in lo..hi {
                // Givens rotation to zero out z
                let r = (x * x + z * z).sqrt();
                let c = if r > S::EPSILON { x / r } else { S::ONE };
                let s = if r > S::EPSILON { z / r } else { S::ZERO };

                // Update offdiag[k-1] if applicable
                if k > lo {
                    offdiag[k - 1] = r;
                }

                // Apply Givens rotation to the tridiagonal:
                // Rows/cols k and k+1
                let dk = diag[k];
                let dk1 = diag[k + 1];
                let ek = offdiag[k];

                diag[k] = c * c * dk + S::TWO * c * s * ek + s * s * dk1;
                diag[k + 1] = s * s * dk - S::TWO * c * s * ek + c * c * dk1;
                offdiag[k] = c * s * (dk1 - dk) + (c * c - s * s) * ek;

                // Chase the bulge
                if k + 1 < hi {
                    let ek1 = offdiag[k + 1];
                    z = s * ek1;
                    offdiag[k + 1] = c * ek1;
                    x = offdiag[k];
                }

                // Update eigenvectors: columns k and k+1
                let q_data = q.as_mut_slice();
                let col_k = k * n;
                let col_k1 = (k + 1) * n;
                for i in 0..n {
                    let qk = q_data[col_k + i];
                    let qk1 = q_data[col_k1 + i];
                    q_data[col_k + i] = c * qk + s * qk1;
                    q_data[col_k1 + i] = -s * qk + c * qk1;
                }
            }
        }
    }

    /// Reconstruct: V * diag(λ) * V^T
    pub fn reconstruct(&self) -> DMat<S> {
        let n = self.eigenvalues.len();
        let vt = self.eigenvectors.transpose();
        let mut result = DMat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut sum = S::ZERO;
                for k in 0..n {
                    sum += self.eigenvectors.get(i, k) * self.eigenvalues[k] * vt.get(k, j);
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
    fn diagonal_matrix() {
        let a = DMat::from_fn(3, 3, |i, j| if i == j { (i + 1) as f64 } else { 0.0 });
        let eig = SymmetricEigen::new(&a);
        assert!((eig.eigenvalues[0] - 1.0).abs() < 1e-10);
        assert!((eig.eigenvalues[1] - 2.0).abs() < 1e-10);
        assert!((eig.eigenvalues[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn reconstruct() {
        let a = DMat::from_fn(3, 3, |i, j| {
            [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]][i][j]
        });
        let eig = SymmetricEigen::new(&a);
        let recon = eig.reconstruct();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (recon.get(i, j) - a.get(i, j)).abs() < 1e-8,
                    "mismatch at ({}, {}): {} vs {}",
                    i,
                    j,
                    recon.get(i, j),
                    a.get(i, j)
                );
            }
        }
    }

    #[test]
    fn eigenvectors_orthogonal() {
        let a = DMat::from_fn(3, 3, |i, j| {
            [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]][i][j]
        });
        let eig = SymmetricEigen::new(&a);
        let vtv = eig.eigenvectors.transpose().mul_mat(&eig.eigenvectors);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (vtv.get(i, j) - expected).abs() < 1e-8,
                    "V^T V mismatch at ({}, {}): {}",
                    i,
                    j,
                    vtv.get(i, j)
                );
            }
        }
    }

    #[test]
    fn jacobi_small() {
        let a = DMat::from_fn(2, 2, |i, j| [[3.0, 1.0], [1.0, 2.0]][i][j]);
        let eig = SymmetricEigen::jacobi(&a);
        let expected_0 = (5.0 - 5.0_f64.sqrt()) / 2.0;
        let expected_1 = (5.0 + 5.0_f64.sqrt()) / 2.0;
        assert!((eig.eigenvalues[0] - expected_0).abs() < 1e-10);
        assert!((eig.eigenvalues[1] - expected_1).abs() < 1e-10);
    }

    #[test]
    fn larger_matrix() {
        let n = 5;
        let a = DMat::from_fn(n, n, |i, j| {
            if i == j {
                (i + 1) as f64 * 2.0
            } else {
                1.0 / ((i as f64 - j as f64).abs() + 1.0)
            }
        });
        let a = DMat::from_fn(n, n, |i, j| (a.get(i, j) + a.get(j, i)) * 0.5);

        let eig = SymmetricEigen::new(&a);
        let recon = eig.reconstruct();
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (recon.get(i, j) - a.get(i, j)).abs() < 1e-8,
                    "mismatch at ({}, {}): {} vs {}",
                    i,
                    j,
                    recon.get(i, j),
                    a.get(i, j)
                );
            }
        }

        let vtv = eig.eigenvectors.transpose().mul_mat(&eig.eigenvectors);
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (vtv.get(i, j) - expected).abs() < 1e-8,
                    "V^T V mismatch at ({}, {}): {}",
                    i,
                    j,
                    vtv.get(i, j)
                );
            }
        }
    }

    #[test]
    fn medium_matrix_10x10() {
        let n = 10;
        let a = DMat::from_fn(n, n, |i, j| {
            if i == j {
                (i + 1) as f64 * 3.0
            } else {
                1.0 / ((i as f64 - j as f64).abs() + 0.5)
            }
        });
        let a = DMat::from_fn(n, n, |i, j| (a.get(i, j) + a.get(j, i)) * 0.5);

        let eig = SymmetricEigen::new(&a);
        let recon = eig.reconstruct();
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (recon.get(i, j) - a.get(i, j)).abs() < 1e-6,
                    "mismatch at ({}, {}): {} vs {}",
                    i,
                    j,
                    recon.get(i, j),
                    a.get(i, j)
                );
            }
        }
    }
}
