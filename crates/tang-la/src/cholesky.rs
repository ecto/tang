use tang::Scalar;
use crate::{DVec, DMat};

/// Cholesky decomposition: A = L * L^T for symmetric positive definite matrices.
pub struct Cholesky<S> {
    l: DMat<S>,
}

impl<S: Scalar> Cholesky<S> {
    /// Compute Cholesky decomposition. Returns None if not SPD.
    pub fn new(a: &DMat<S>) -> Option<Self> {
        assert!(a.is_square(), "Cholesky: matrix must be square");
        let n = a.nrows();
        let mut l = DMat::zeros(n, n);

        for j in 0..n {
            let mut sum = S::ZERO;
            for k in 0..j {
                sum += l.get(j, k) * l.get(j, k);
            }
            let diag = a.get(j, j) - sum;
            if diag <= S::ZERO {
                return None; // Not positive definite
            }
            let ljj = diag.sqrt();
            l.set(j, j, ljj);
            let ljj_inv = ljj.recip();

            for i in (j + 1)..n {
                let mut sum = S::ZERO;
                for k in 0..j {
                    sum += l.get(i, k) * l.get(j, k);
                }
                l.set(i, j, (a.get(i, j) - sum) * ljj_inv);
            }
        }

        Some(Self { l })
    }

    /// The lower-triangular factor L.
    pub fn l(&self) -> &DMat<S> { &self.l }

    /// Solve Ax = b via L L^T x = b.
    pub fn solve(&self, b: &DVec<S>) -> DVec<S> {
        let n = self.l.nrows();
        assert_eq!(b.len(), n);

        // Forward: L y = b
        let mut y = DVec::zeros(n);
        for i in 0..n {
            let mut sum = b[i];
            for j in 0..i {
                sum = sum - self.l.get(i, j) * y[j];
            }
            y[i] = sum * self.l.get(i, i).recip();
        }

        // Back: L^T x = y
        let mut x = DVec::zeros(n);
        for i in (0..n).rev() {
            let mut sum = y[i];
            for j in (i + 1)..n {
                sum = sum - self.l.get(j, i) * x[j];
            }
            x[i] = sum * self.l.get(i, i).recip();
        }

        x
    }

    /// Determinant = product of diagonal^2.
    pub fn det(&self) -> S {
        let n = self.l.nrows();
        let mut d = S::ONE;
        for i in 0..n {
            let l = self.l.get(i, i);
            d = d * l * l;
        }
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_spd() {
        // A = [4 2; 2 3] is SPD
        let a = DMat::from_fn(2, 2, |i, j| {
            [[4.0, 2.0], [2.0, 3.0]][i][j]
        });
        let b = DVec::from_slice(&[1.0, 2.0]);
        let chol = Cholesky::new(&a).unwrap();
        let x = chol.solve(&b);
        // Verify Ax = b
        let ax = a.mul_vec(&x);
        assert!((ax[0] - b[0]).abs() < 1e-10);
        assert!((ax[1] - b[1]).abs() < 1e-10);
    }

    #[test]
    fn not_spd() {
        let a = DMat::from_fn(2, 2, |i, j| {
            [[1.0, 2.0], [2.0, 1.0]][i][j]
        });
        assert!(Cholesky::new(&a).is_none());
    }

    #[test]
    fn determinant() {
        let a = DMat::from_fn(2, 2, |i, j| {
            [[4.0, 2.0], [2.0, 3.0]][i][j]
        });
        let chol = Cholesky::new(&a).unwrap();
        assert!((chol.det() - 8.0).abs() < 1e-10);
    }
}
