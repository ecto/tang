use tang::Scalar;
use crate::{DVec, DMat};
use alloc::vec::Vec;

/// Eigendecomposition of a symmetric matrix: A = V * diag(λ) * V^T
///
/// Uses the Jacobi eigenvalue algorithm — robust and simple.
/// For large matrices, use the faer bridge.
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

        let mut d = a.clone(); // Will become diagonal
        let mut v = DMat::<S>::identity(n);

        let max_iter = 100 * n * n;
        let tol = S::EPSILON * S::from_i32(10);

        for _ in 0..max_iter {
            // Find largest off-diagonal element
            let mut max_val = S::ZERO;
            let mut p = 0;
            let mut q = 1;
            for i in 0..n {
                for j in (i + 1)..n {
                    let v = d.get(i, j).abs();
                    if v > max_val {
                        max_val = v;
                        p = i;
                        q = j;
                    }
                }
            }

            if max_val < tol {
                break; // Converged
            }

            // Compute Jacobi rotation to zero out (p, q)
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

            // Apply rotation to D
            let new_pp = app - t * apq;
            let new_qq = aqq + t * apq;
            d.set(p, p, new_pp);
            d.set(q, q, new_qq);
            d.set(p, q, S::ZERO);
            d.set(q, p, S::ZERO);

            for i in 0..n {
                if i == p || i == q { continue; }
                let dip = d.get(i, p);
                let diq = d.get(i, q);
                d.set(i, p, c * dip - s * diq);
                d.set(p, i, c * dip - s * diq);
                d.set(i, q, s * dip + c * diq);
                d.set(q, i, s * dip + c * diq);
            }

            // Apply rotation to V
            for i in 0..n {
                let vip = v.get(i, p);
                let viq = v.get(i, q);
                v.set(i, p, c * vip - s * viq);
                v.set(i, q, s * vip + c * viq);
            }
        }

        // Extract eigenvalues and sort ascending
        let mut eigs: Vec<(S, usize)> = (0..n).map(|i| (d.get(i, i), i)).collect();
        eigs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));

        let eigenvalues = DVec::from_fn(n, |i| eigs[i].0);
        let eigenvectors = DMat::from_fn(n, n, |i, j| v.get(i, eigs[j].1));

        Self { eigenvalues, eigenvectors }
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
        let a = DMat::from_fn(3, 3, |i, j| {
            if i == j { (i + 1) as f64 } else { 0.0 }
        });
        let eig = SymmetricEigen::new(&a);
        // Eigenvalues should be 1, 2, 3 (sorted ascending)
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
                assert!((recon.get(i, j) - a.get(i, j)).abs() < 1e-8,
                    "mismatch at ({}, {}): {} vs {}", i, j, recon.get(i, j), a.get(i, j));
            }
        }
    }

    #[test]
    fn eigenvectors_orthogonal() {
        let a = DMat::from_fn(3, 3, |i, j| {
            [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]][i][j]
        });
        let eig = SymmetricEigen::new(&a);
        // V^T V should be identity
        let vtv = eig.eigenvectors.transpose().mul_mat(&eig.eigenvectors);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((vtv.get(i, j) - expected).abs() < 1e-8,
                    "V^T V mismatch at ({}, {}): {}", i, j, vtv.get(i, j));
            }
        }
    }
}
