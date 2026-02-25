use tang::Scalar;
use crate::{DVec, DMat};
use alloc::vec::Vec;

/// QR decomposition via Householder reflections: A = Q * R
pub struct Qr<S> {
    /// Householder vectors stored below diagonal + R above.
    qr: DMat<S>,
    /// Diagonal of R.
    r_diag: Vec<S>,
}

impl<S: Scalar> Qr<S> {
    /// Compute QR decomposition.
    pub fn new(a: &DMat<S>) -> Self {
        let m = a.nrows();
        let n = a.ncols();
        let mut qr = a.clone();
        let mut r_diag = Vec::with_capacity(n);

        for k in 0..n.min(m) {
            // Compute norm of column k below diagonal
            let mut norm_sq = S::ZERO;
            for i in k..m {
                norm_sq += qr.get(i, k) * qr.get(i, k);
            }
            let mut norm = norm_sq.sqrt();

            if norm > S::EPSILON {
                // Ensure correct sign
                if qr.get(k, k) > S::ZERO {
                    norm = -norm;
                }

                // Divide column by norm, adjust diagonal
                for i in k..m {
                    let v = qr.get(i, k) / (-norm);
                    qr.set(i, k, v);
                }
                let v = qr.get(k, k) + S::ONE;
                qr.set(k, k, v);

                // Apply transformation to remaining columns
                for j in (k + 1)..n {
                    let mut s = S::ZERO;
                    for i in k..m {
                        s += qr.get(i, k) * qr.get(i, j);
                    }
                    s = -s / qr.get(k, k);
                    for i in k..m {
                        let v = qr.get(i, j) + s * qr.get(i, k);
                        qr.set(i, j, v);
                    }
                }
            }

            r_diag.push(norm);
        }

        Self { qr, r_diag }
    }

    /// Get R (upper triangular).
    pub fn r(&self) -> DMat<S> {
        let m = self.qr.nrows();
        let n = self.qr.ncols();
        let k = m.min(n);
        DMat::from_fn(k, n, |i, j| {
            if i == j {
                self.r_diag[i]
            } else if j > i {
                self.qr.get(i, j)
            } else {
                S::ZERO
            }
        })
    }

    /// Get Q (orthogonal m×m or thin m×min(m,n)).
    pub fn q(&self) -> DMat<S> {
        let m = self.qr.nrows();
        let n = self.qr.ncols();
        let k = m.min(n);
        let mut q = DMat::identity(m);

        for j in (0..k).rev() {
            if self.qr.get(j, j) == S::ZERO { continue; }
            for col in j..m {
                let mut s = S::ZERO;
                for i in j..m {
                    s += self.qr.get(i, j) * q.get(i, col);
                }
                s = -s / self.qr.get(j, j);
                for i in j..m {
                    let v = q.get(i, col) + s * self.qr.get(i, j);
                    q.set(i, col, v);
                }
            }
        }

        // Return thin Q (m × k)
        DMat::from_fn(m, k, |i, j| q.get(i, j))
    }

    /// Solve least-squares: min ||Ax - b||.
    pub fn solve(&self, b: &DVec<S>) -> DVec<S> {
        let m = self.qr.nrows();
        let n = self.qr.ncols();
        assert_eq!(b.len(), m);

        // Apply Q^T to b
        let mut x = DVec::from_slice(b.as_slice());
        for k in 0..n.min(m) {
            if self.qr.get(k, k) == S::ZERO { continue; }
            let mut s = S::ZERO;
            for i in k..m {
                s += self.qr.get(i, k) * x[i];
            }
            s = -s / self.qr.get(k, k);
            for i in k..m {
                x[i] = x[i] + s * self.qr.get(i, k);
            }
        }

        // Back-substitute R
        let mut result = DVec::zeros(n);
        for i in (0..n).rev() {
            let mut sum = x[i];
            for j in (i + 1)..n {
                sum = sum - self.qr.get(i, j) * result[j];
            }
            if self.r_diag[i].abs() < S::EPSILON {
                result[i] = S::ZERO;
            } else {
                result[i] = sum / self.r_diag[i];
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qr_identity() {
        let a = DMat::<f64>::identity(3);
        let qr = Qr::new(&a);
        let q = qr.q();
        let r = qr.r();
        // Q should be identity (up to sign)
        for i in 0..3 {
            assert!(q.get(i, i).abs() > 0.99, "Q diagonal should be ±1");
        }
        // R should be identity (up to sign)
        for i in 0..3 {
            assert!(r.get(i, i).abs() > 0.99, "R diagonal should be ±1");
        }
    }

    #[test]
    fn solve_overdetermined() {
        // A = [1; 1], b = [1; 3] -> least squares: x = 2
        let a = DMat::from_fn(2, 1, |_, _| 1.0_f64);
        let b = DVec::from_slice(&[1.0, 3.0]);
        let qr = Qr::new(&a);
        let x = qr.solve(&b);
        assert!((x[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn qr_reconstruct() {
        let a = DMat::from_fn(3, 2, |i, j| (i * 2 + j + 1) as f64);
        let qr = Qr::new(&a);
        let q = qr.q();
        let r = qr.r();
        let recon = q.mul_mat(&r);
        for i in 0..3 {
            for j in 0..2 {
                assert!((recon.get(i, j) - a.get(i, j)).abs() < 1e-10,
                    "mismatch at ({}, {}): {} vs {}", i, j, recon.get(i, j), a.get(i, j));
            }
        }
    }
}
