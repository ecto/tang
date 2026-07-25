use super::{DMat, DVec};
use crate::Scalar;
use alloc::vec;
use alloc::vec::Vec;

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
        // Deflation floor. The purely relative test `|e| <= eps*(|d_k| + |d_k+1|)`
        // is unusable when the trailing diagonal entries are tiny or zero: for a
        // rank-deficient matrix it degenerates to `|e| == 0`, which never holds in
        // floating point, so those eigenvalues never deflate. The iteration then
        // burns its whole budget on an already-converged tail and returns whatever
        // unconverged values happen to be in `diag`. Zeroing an off-diagonal of
        // size eps*||A|| perturbs the eigenvalues by at most machine precision
        // (Weyl), so that is the correct absolute floor to add. This matches
        // LAPACK dsteqr, which combines the relative test with a safe-minimum term.
        let mut anorm = S::ZERO;
        for i in 0..n {
            let e_prev = if i > 0 { offdiag[i - 1].abs() } else { S::ZERO };
            let e_next = if i + 1 < n { offdiag[i].abs() } else { S::ZERO };
            let row = diag[i].abs() + e_prev + e_next;
            if row > anorm {
                anorm = row;
            }
        }
        let floor = anorm * S::EPSILON;
        let negligible = |e: S, da: S, db: S| e.abs() <= (da.abs() + db.abs()) * S::EPSILON + floor;

        // Budget is per eigenvalue, not for the whole matrix: it resets every time
        // the active block shrinks. A single global cap silently truncates the
        // iteration on larger matrices and yields non-converged eigenvalues.
        let max_iter_per_eig = 30;
        let mut iters = 0;
        let mut prev_hi = n;

        loop {
            // Find the largest unreduced block [lo..=hi]
            // Scan from the bottom to find hi where offdiag[hi-1] is non-negligible
            let mut hi = n - 1;
            while hi > 0 {
                if negligible(offdiag[hi - 1], diag[hi - 1], diag[hi]) {
                    offdiag[hi - 1] = S::ZERO;
                    hi -= 1;
                } else {
                    break;
                }
            }
            if hi == 0 {
                break; // All converged
            }

            if hi < prev_hi {
                prev_hi = hi;
                iters = 0;
            }
            iters += 1;
            debug_assert!(
                iters <= max_iter_per_eig,
                "SymmetricEigen: QR iteration failed to converge"
            );
            if iters > max_iter_per_eig {
                break;
            }

            // Find lo: start of the unreduced block ending at hi
            let mut lo = hi - 1;
            while lo > 0 {
                if negligible(offdiag[lo - 1], diag[lo - 1], diag[lo]) {
                    break;
                }
                lo -= 1;
            }

            // Wilkinson shift: eigenvalue of trailing 2x2 closer to diag[hi]
            let d_hi = diag[hi];
            let d_hi1 = diag[hi - 1];
            let e_hi1 = offdiag[hi - 1];
            let delta = (d_hi1 - d_hi) * S::HALF;
            // Scale-relative test: an absolute `|delta| < EPSILON` would take the
            // fallback branch for every well-separated pair in a block whose own
            // entries are smaller than machine epsilon, wrecking the shift quality.
            let mu = if delta.abs() <= (d_hi.abs() + d_hi1.abs()) * S::EPSILON {
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
                // The guard must be `r == 0`, not `r < EPSILON`. A rotation is
                // well defined for any r > 0 however tiny, and substituting the
                // identity instead leaves the bulge in place — the iteration then
                // stalls forever on blocks whose entries are below machine
                // epsilon in absolute terms (which is routine for a density
                // matrix with near-zero eigenvalues).
                let r = (x * x + z * z).sqrt();
                let (c, s) = if r > S::ZERO {
                    (x / r, z / r)
                } else {
                    (S::ONE, S::ZERO)
                };

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

/// Branchless Jacobi eigendecomposition for tracing through ExprId.
///
/// Unlike `SymmetricEigen::new()`, this variant has no data-dependent branches:
/// - Cyclic sweep order (no max-finding)
/// - Fixed sweep count (no convergence checks)
/// - `S::select()` for sign-dependent terms
/// - Sorting network for eigenvalue ordering
///
/// This makes it safe to trace through symbolic expression graphs.
pub fn branchless_jacobi_eigen<S: Scalar>(mat: &DMat<S>, n_sweeps: usize) -> (DVec<S>, DMat<S>) {
    let n = mat.nrows();
    assert!(
        mat.is_square(),
        "branchless_jacobi_eigen: matrix must be square"
    );

    // Working copy of the matrix (will be diagonalized in place)
    let mut d = mat.clone();
    // Eigenvector accumulator, starts as identity
    let mut v = DMat::<S>::identity(n);

    let eps = S::from_f64(f64::EPSILON);

    for _ in 0..n_sweeps {
        // Cyclic Jacobi: process ALL off-diagonal pairs (p,q) in fixed order
        for p in 0..n {
            for q in (p + 1)..n {
                let a_pq = d.get(p, q);

                // Guard: blend rotation with identity when off-diagonal is tiny
                // active = 1.0 if |a_pq| > eps, else 0.0
                let a_pq_abs = (a_pq * a_pq).sqrt();
                let active = S::select(a_pq_abs - eps, S::ONE, S::ZERO);

                let d_pp = d.get(p, p);
                let d_qq = d.get(q, q);

                // theta = (d[q] - d[p]) / (2 * a[p][q])
                // When a_pq ~ 0, we need to avoid division by zero.
                // Use a_pq + (1 - active) * ONE to make denominator safe.
                let safe_apq = a_pq + (S::ONE - active) * S::ONE;
                let theta = (d_qq - d_pp) / (S::TWO * safe_apq);

                // Branchless t = sign(theta) / (|theta| + sqrt(1 + theta^2))
                let sign_theta = S::select(theta, S::ONE, -S::ONE);
                let abs_theta = (theta * theta).sqrt();
                let t = sign_theta / (abs_theta + (S::ONE + theta * theta).sqrt());

                // Blend: if not active, t = 0 (identity rotation)
                let t = t * active;

                let c = (S::ONE + t * t).sqrt().recip();
                let s = t * c;

                // Update diagonal elements
                let new_pp = d_pp - t * a_pq * active;
                let new_qq = d_qq + t * a_pq * active;
                d.set(p, p, new_pp);
                d.set(q, q, new_qq);
                d.set(p, q, S::ZERO);
                d.set(q, p, S::ZERO);

                // Update off-diagonal rows/columns
                for i in 0..n {
                    if i == p || i == q {
                        continue;
                    }
                    let d_ip = d.get(i, p);
                    let d_iq = d.get(i, q);
                    let new_ip = c * d_ip - s * d_iq;
                    let new_iq = s * d_ip + c * d_iq;
                    d.set(i, p, new_ip);
                    d.set(p, i, new_ip);
                    d.set(i, q, new_iq);
                    d.set(q, i, new_iq);
                }

                // Update eigenvector columns
                for i in 0..n {
                    let v_ip = v.get(i, p);
                    let v_iq = v.get(i, q);
                    v.set(i, p, c * v_ip - s * v_iq);
                    v.set(i, q, s * v_ip + c * v_iq);
                }
            }
        }
    }

    // Extract eigenvalues from diagonal
    let mut eigenvalues: Vec<S> = (0..n).map(|i| d.get(i, i)).collect();
    // Columns of v are the eigenvectors — we need to sort by eigenvalue.

    // Branchless sorting network (bubble sort variant with select)
    // For small n this is fine; produces O(n^2) comparators.
    let half = S::from_f64(0.5);
    for _ in 0..n {
        for j in 0..(n - 1) {
            // Compare eigenvalues[j] and eigenvalues[j+1]
            let a = eigenvalues[j];
            let b = eigenvalues[j + 1];
            // do_swap = 1 if a > b (need to swap to ascending), else 0
            let do_swap = S::select(a - b, S::ONE, S::ZERO);

            // Branchless swap for eigenvalues
            let new_a = S::select(do_swap - half, b, a); // min
            let new_b = S::select(do_swap - half, a, b); // max
            eigenvalues[j] = new_a;
            eigenvalues[j + 1] = new_b;

            // Branchless swap for eigenvector columns
            for i in 0..n {
                let vj = v.get(i, j);
                let vj1 = v.get(i, j + 1);
                let new_vj = S::select(do_swap - half, vj1, vj);
                let new_vj1 = S::select(do_swap - half, vj, vj1);
                v.set(i, j, new_vj);
                v.set(i, j + 1, new_vj1);
            }
        }
    }

    (DVec::from_fn(n, |i| eigenvalues[i]), v)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a symmetric matrix with a deliberately nasty spectrum: several
    /// high-multiplicity clusters spanning many orders of magnitude, plus an
    /// exactly-zero null space. Reduced density matrices of gauge-theory ground
    /// states look exactly like this, and it is the case the QR iteration used to
    /// silently give up on.
    fn clustered_spectrum(n: usize, evs: &[f64]) -> DMat<f64> {
        // Deterministic orthogonal Q from a product of Householder reflections.
        let mut q = DMat::<f64>::identity(n);
        for seed in 1..4usize {
            let v: Vec<f64> = (0..n)
                .map(|i| (((i * 7 + seed * 13) % 23) as f64) - 11.0)
                .collect();
            let vv: f64 = v.iter().map(|x| x * x).sum();
            let h = DMat::from_fn(n, n, |i, j| {
                (if i == j { 1.0 } else { 0.0 }) - 2.0 * v[i] * v[j] / vv
            });
            q = q.mul_mat(&h);
        }
        // A = Q diag(evs) Q^T, symmetrized to be exactly symmetric.
        let mut a = DMat::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += q.get(i, k) * evs[k] * q.get(j, k);
                }
                a.set(i, j, sum);
            }
        }
        DMat::from_fn(n, n, |i, j| (a.get(i, j) + a.get(j, i)) * 0.5)
    }

    fn max_reconstruct_err(a: &DMat<f64>, eig: &SymmetricEigen<f64>) -> f64 {
        let n = a.nrows();
        let recon = eig.reconstruct();
        let mut err: f64 = 0.0;
        for i in 0..n {
            for j in 0..n {
                err = err.max((recon.get(i, j) - a.get(i, j)).abs());
            }
        }
        err
    }

    /// Regression: a rank-deficient matrix (exact zero eigenvalues) used to make
    /// the deflation test `|e| <= eps*(|d_k| + |d_k+1|)` degenerate to `|e| == 0`,
    /// so the null space never deflated and the iteration returned garbage.
    #[test]
    fn rank_one_projector_is_exact() {
        let n = 40;
        let mut psi: Vec<f64> = (0..n)
            .map(|i| (((i * 37 % 101) + 1) as f64).sqrt())
            .collect();
        let norm: f64 = psi.iter().map(|x| x * x).sum::<f64>().sqrt();
        for v in psi.iter_mut() {
            *v /= norm;
        }
        let a = DMat::from_fn(n, n, |i, j| psi[i] * psi[j]);
        let eig = SymmetricEigen::new(&a);

        assert!(max_reconstruct_err(&a, &eig) < 1e-12);
        assert!((eig.eigenvalues[n - 1] - 1.0).abs() < 1e-12);
        // Every other eigenvalue must be zero to machine precision — a rank-1
        // projector has von Neumann entropy identically zero, and any spurious
        // eigenvalue above ~1e-15 shows up as a nonzero entropy.
        for i in 0..(n - 1) {
            assert!(
                eig.eigenvalues[i].abs() < 1e-14,
                "spurious eigenvalue[{i}] = {}",
                eig.eigenvalues[i]
            );
        }
    }

    /// Regression: clustered eigenvalues spanning many decades. The bulge-chase
    /// Givens guard `r > EPSILON` used to substitute the identity rotation for
    /// small-but-nonzero r, stalling the iteration permanently.
    #[test]
    fn clustered_multiscale_spectrum_reconstructs() {
        let mut evs = Vec::new();
        for &(val, mult) in &[
            (0.0, 6usize),
            (1.0e-6, 6),
            (1.0e-4, 6),
            (5.0e-3, 6),
            (9.0e-2, 6),
            (4.0e-1, 1),
        ] {
            for _ in 0..mult {
                evs.push(val);
            }
        }
        let n = evs.len();
        let a = clustered_spectrum(n, &evs);
        let eig = SymmetricEigen::new(&a);

        let err = max_reconstruct_err(&a, &eig);
        assert!(err < 1e-12, "reconstruction error {err:e}");

        let mut sorted = evs.clone();
        sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
        for i in 0..n {
            assert!(
                (eig.eigenvalues[i] - sorted[i]).abs() < 1e-12,
                "eigenvalue[{i}] = {} expected {}",
                eig.eigenvalues[i],
                sorted[i]
            );
        }
    }

    /// The decomposition must not depend on last-bit perturbations of the input:
    /// a non-converged QR iteration is chaotic, so a 1-ulp change to the matrix
    /// moved eigenvalues by ~1e-2.
    #[test]
    fn stable_under_ulp_perturbation() {
        let evs: Vec<f64> = (0..30)
            .map(|i| {
                if i < 8 {
                    0.0
                } else {
                    1e-6 * (1 << (i % 10)) as f64
                }
            })
            .collect();
        let a = clustered_spectrum(evs.len(), &evs);
        let n = a.nrows();
        let perturbed = DMat::from_fn(n, n, |i, j| {
            let v = a.get(i, j);
            if i == j {
                f64::from_bits(v.to_bits() ^ 1)
            } else {
                v
            }
        });

        let base = SymmetricEigen::new(&a);
        let pert = SymmetricEigen::new(&perturbed);
        for i in 0..n {
            let d = (base.eigenvalues[i] - pert.eigenvalues[i]).abs();
            assert!(
                d < 1e-12,
                "eigenvalue[{i}] moved by {d:e} under a 1-ulp change"
            );
        }
    }

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

    #[test]
    fn branchless_jacobi_4x4() {
        let a = DMat::from_fn(4, 4, |i, j| {
            [
                [4.0, 1.0, 0.5, 0.0],
                [1.0, 3.0, 1.0, 0.5],
                [0.5, 1.0, 2.0, 1.0],
                [0.0, 0.5, 1.0, 1.0],
            ][i][j]
        });

        let (evals, evecs) = branchless_jacobi_eigen(&a, 30);
        let eig_ref = SymmetricEigen::new(&a);

        // Check eigenvalues match reference
        for i in 0..4 {
            assert!(
                (evals[i] - eig_ref.eigenvalues[i]).abs() < 1e-10,
                "eigenvalue {} mismatch: branchless={}, ref={}",
                i,
                evals[i],
                eig_ref.eigenvalues[i]
            );
        }

        // Check eigenvectors are orthogonal
        let vtv = evecs.transpose().mul_mat(&evecs);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (vtv.get(i, j) - expected).abs() < 1e-10,
                    "V^T V mismatch at ({}, {}): {}",
                    i,
                    j,
                    vtv.get(i, j)
                );
            }
        }
    }

    #[test]
    fn branchless_jacobi_8x8() {
        let n = 8;
        let a = DMat::from_fn(n, n, |i, j| {
            if i == j {
                (i + 1) as f64 * 2.0
            } else {
                1.0 / ((i as f64 - j as f64).abs() + 1.0)
            }
        });
        let a = DMat::from_fn(n, n, |i, j| (a.get(i, j) + a.get(j, i)) * 0.5);

        let (evals, evecs) = branchless_jacobi_eigen(&a, 30);
        let eig_ref = SymmetricEigen::new(&a);

        for i in 0..n {
            assert!(
                (evals[i] - eig_ref.eigenvalues[i]).abs() < 1e-10,
                "eigenvalue {} mismatch: branchless={}, ref={}",
                i,
                evals[i],
                eig_ref.eigenvalues[i]
            );
        }

        let vtv = evecs.transpose().mul_mat(&evecs);
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (vtv.get(i, j) - expected).abs() < 1e-10,
                    "V^T V mismatch at ({}, {}): {}",
                    i,
                    j,
                    vtv.get(i, j)
                );
            }
        }
    }

    #[test]
    fn branchless_jacobi_16x16() {
        let n = 16;
        let a = DMat::from_fn(n, n, |i, j| {
            if i == j {
                (i + 1) as f64 * 3.0
            } else {
                1.0 / ((i as f64 - j as f64).abs() + 0.5)
            }
        });
        let a = DMat::from_fn(n, n, |i, j| (a.get(i, j) + a.get(j, i)) * 0.5);

        let (evals, evecs) = branchless_jacobi_eigen(&a, 30);
        let eig_ref = SymmetricEigen::new(&a);

        for i in 0..n {
            assert!(
                (evals[i] - eig_ref.eigenvalues[i]).abs() < 1e-8,
                "eigenvalue {} mismatch: branchless={}, ref={}",
                i,
                evals[i],
                eig_ref.eigenvalues[i]
            );
        }

        let vtv = evecs.transpose().mul_mat(&evecs);
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
}
