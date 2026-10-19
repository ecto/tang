//! Small dense helpers on row-major `f64` buffers.
//!
//! Heavy products go through `tang_la::gemm_*` (NEON microkernels on
//! aarch64); symmetric eigendecompositions through `tang_la::SymmetricEigen`
//! (Householder tridiagonalisation + implicit QR). Truncated SVDs of the
//! `m × n` TT unfoldings are computed from the Gram matrix of the *smaller*
//! side (`min(m,n) ≤ 2·r_max`), which is what makes TT-SVD of a 2^24-entry
//! tensor cheap: one-sided Jacobi (`tang_la::Svd`) on a 2 × 2^23 unfolding
//! would be far slower. The Gram route squares the condition number; in f64
//! that leaves singular values down to ~1e-7·σ₁ resolved, well below any
//! truncation tolerance the codec uses (see the crate docs).

use tang_la::{gemm_nn, gemm_nt, gemm_tn, DMat, SymmetricEigen};

/// `A·Aᵀ` for row-major `a` (m × n).
pub fn gram_rows(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut g = vec![0.0; m * m];
    gemm_nt(m, m, n, a, n, a, n, &mut g, m);
    symmetrize(&mut g, m);
    g
}

/// `Aᵀ·A` for row-major `a` (m × n).
pub fn gram_cols(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut g = vec![0.0; n * n];
    gemm_tn(n, n, m, a, n, a, n, &mut g, n);
    symmetrize(&mut g, n);
    g
}

fn symmetrize(g: &mut [f64], n: usize) {
    for i in 0..n {
        for j in i + 1..n {
            let v = 0.5 * (g[i * n + j] + g[j * n + i]);
            g[i * n + j] = v;
            g[j * n + i] = v;
        }
    }
}

/// `C = A·B` (A: m×k, B: k×n), row-major.
pub fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    gemm_nn(m, n, k, a, k, b, n, &mut c, n);
    c
}

/// `C = Aᵀ·B` (A: k×m, B: k×n), row-major.
pub fn matmul_tn(a: &[f64], b: &[f64], k: usize, m: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    gemm_tn(m, n, k, a, m, b, n, &mut c, n);
    c
}

/// Eigenpairs of a symmetric matrix, **descending**: (values, vectors as
/// row-major n×n with eigenvector j in column j).
pub fn sym_eig_desc(g: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    // Symmetric, so the column-major DMat sees the same matrix.
    let e = SymmetricEigen::new(&DMat::from_raw(n, n, g.to_vec()));
    let mut vals = Vec::with_capacity(n);
    let mut vecs = vec![0.0; n * n];
    for (jj, j) in (0..n).rev().enumerate() {
        vals.push(e.eigenvalues[j]);
        for i in 0..n {
            vecs[i * n + jj] = e.eigenvectors.get(i, j);
        }
    }
    (vals, vecs)
}

/// Lower Cholesky factor of SPD `a` (n×n), or `None` if not positive definite.
pub fn cholesky(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if s <= 0.0 || !s.is_finite() {
                    return None;
                }
                l[i * n + i] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    Some(l)
}

/// Solve `L·X = B` in place (L lower n×n, B n×m row-major).
pub fn solve_lower(l: &[f64], b: &mut [f64], n: usize, m: usize) {
    for i in 0..n {
        for k in 0..i {
            let lik = l[i * n + k];
            if lik != 0.0 {
                let (head, tail) = b.split_at_mut(i * m);
                let src = &head[k * m..k * m + m];
                for (t, s) in tail[..m].iter_mut().zip(src) {
                    *t -= lik * s;
                }
            }
        }
        let d = l[i * n + i];
        for t in &mut b[i * m..i * m + m] {
            *t /= d;
        }
    }
}

/// Solve `Lᵀ·X = B` in place (L lower n×n, B n×m row-major).
pub fn solve_lower_t(l: &[f64], b: &mut [f64], n: usize, m: usize) {
    for i in (0..n).rev() {
        for k in i + 1..n {
            let lki = l[k * n + i];
            if lki != 0.0 {
                let (head, tail) = b.split_at_mut(k * m);
                for (t, s) in head[i * m..i * m + m].iter_mut().zip(&tail[..m]) {
                    *t -= lki * s;
                }
            }
        }
        let d = l[i * n + i];
        for t in &mut b[i * m..i * m + m] {
            *t /= d;
        }
    }
}

/// Transpose a row-major m×n matrix.
pub fn transpose(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut t = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            t[j * m + i] = a[i * n + j];
        }
    }
    t
}

/// Truncated SVD of a row-major m×n matrix via the Gram matrix of the
/// smaller side. Returns all singular values (descending, length min(m,n))
/// and a closure-free pair: `u` (m × k, row-major) for the leading `k`
/// columns requested by `pick` (which sees the singular values).
pub fn svd_left(
    a: &[f64],
    m: usize,
    n: usize,
    pick: impl FnOnce(&[f64]) -> usize,
) -> (Vec<f64>, Vec<f64>, usize) {
    if m <= n {
        let g = gram_rows(a, m, n);
        let (lam, v) = sym_eig_desc(&g, m);
        let s: Vec<f64> = lam.iter().map(|&l| l.max(0.0).sqrt()).collect();
        let k = pick(&s).min(m);
        let mut u = vec![0.0; m * k];
        for i in 0..m {
            u[i * k..i * k + k].copy_from_slice(&v[i * m..i * m + k]);
        }
        (s, u, k)
    } else {
        let g = gram_cols(a, m, n);
        let (lam, v) = sym_eig_desc(&g, n);
        let s: Vec<f64> = lam.iter().map(|&l| l.max(0.0).sqrt()).collect();
        let smax = s.first().copied().unwrap_or(0.0);
        let k = pick(&s).min(n).min(
            s.iter()
                .take_while(|&&x| x > 1e-13 * smax && x > 0.0)
                .count(),
        );
        // U = A V Σ⁻¹, then re-orthonormalise (modified Gram–Schmidt, twice).
        let mut vk = vec![0.0; n * k];
        for i in 0..n {
            vk[i * k..i * k + k].copy_from_slice(&v[i * n..i * n + k]);
        }
        let mut u = matmul(a, &vk, m, n, k);
        for j in 0..k {
            for i in 0..m {
                u[i * k + j] /= s[j];
            }
        }
        orthonormalize_cols(&mut u, m, k);
        (s, u, k)
    }
}

/// Modified Gram–Schmidt (two passes) on the columns of row-major m×k `u`.
pub fn orthonormalize_cols(u: &mut [f64], m: usize, k: usize) {
    for _pass in 0..2 {
        for j in 0..k {
            for p in 0..j {
                let mut dot = 0.0;
                for i in 0..m {
                    dot += u[i * k + j] * u[i * k + p];
                }
                for i in 0..m {
                    u[i * k + j] -= dot * u[i * k + p];
                }
            }
            let nrm = (0..m)
                .map(|i| u[i * k + j] * u[i * k + j])
                .sum::<f64>()
                .sqrt();
            if nrm > 0.0 {
                for i in 0..m {
                    u[i * k + j] /= nrm;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn svd_left_reconstructs_both_orientations() {
        for (m, n) in [(4usize, 9usize), (9, 4)] {
            let a: Vec<f64> = (0..m * n).map(|i| ((i * 37 % 11) as f64).cos()).collect();
            let (s, u, k) = svd_left(&a, m, n, |s| s.len());
            assert_eq!(k, m.min(n));
            // U Uᵀ A == A when k = rank-full.
            let uta = matmul_tn(&u, &a, m, k, n);
            let back = matmul(&u, &uta, m, k, n);
            for (x, y) in a.iter().zip(&back) {
                assert!((x - y).abs() < 1e-9);
            }
            let fro: f64 = a.iter().map(|x| x * x).sum();
            let ss: f64 = s.iter().map(|x| x * x).sum();
            assert!((fro - ss).abs() < 1e-9 * fro);
        }
    }

    #[test]
    fn cholesky_solves() {
        let a = [4.0, 2.0, 0.6, 2.0, 5.0, 1.0, 0.6, 1.0, 3.0];
        let l = cholesky(&a, 3).unwrap();
        let mut b = vec![1.0, 2.0, 3.0];
        solve_lower(&l, &mut b, 3, 1);
        solve_lower_t(&l, &mut b, 3, 1);
        let ax: Vec<f64> = (0..3)
            .map(|i| (0..3).map(|j| a[i * 3 + j] * b[j]).sum())
            .collect();
        for (x, y) in ax.iter().zip([1.0, 2.0, 3.0]) {
            assert!((x - y).abs() < 1e-12);
        }
        assert!(cholesky(&[1.0, 2.0, 2.0, 1.0], 2).is_none());
    }
}
