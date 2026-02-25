use tang_la::{DVec, DMat, Lu};

/// Levenberg-Marquardt optimizer for nonlinear least-squares.
///
/// Minimizes sum of squares: min_x ||r(x)||^2
/// where r: R^n -> R^m is the residual function.
pub struct LevenbergMarquardt {
    pub max_iter: usize,
    pub tol: f64,
    pub lambda_init: f64,
}

impl LevenbergMarquardt {
    pub fn new() -> Self {
        Self { max_iter: 100, tol: 1e-10, lambda_init: 1e-3 }
    }

    /// Minimize ||r(x)||^2 given residual and Jacobian functions.
    pub fn minimize<R, J>(
        &self,
        x0: &DVec<f64>,
        residual_fn: R,
        jacobian_fn: J,
    ) -> DVec<f64>
    where
        R: Fn(&DVec<f64>) -> DVec<f64>,
        J: Fn(&DVec<f64>) -> DMat<f64>,
    {
        let mut x = x0.clone();
        let mut lambda = self.lambda_init;
        let n = x.len();

        for _ in 0..self.max_iter {
            let r = residual_fn(&x);
            let j = jacobian_fn(&x);
            let jt = j.transpose();

            // Normal equations: (J^T J + λI) δ = -J^T r
            let jtj = jt.mul_mat(&j);
            let jtr = jt.mul_vec(&r);

            // Add damping
            let mut a = jtj.clone();
            for i in 0..n {
                let v = a.get(i, i) + lambda;
                a.set(i, i, v);
            }

            let neg_jtr = DVec::from_fn(n, |i| -jtr[i]);
            if let Some(lu) = Lu::new(&a) {
                let delta = lu.solve(&neg_jtr);

                // Try step
                let mut x_new = x.clone();
                for i in 0..n {
                    x_new[i] = x[i] + delta[i];
                }

                let r_new = residual_fn(&x_new);
                if r_new.norm_sq() < r.norm_sq() {
                    x = x_new;
                    lambda *= 0.1; // Decrease damping
                } else {
                    lambda *= 10.0; // Increase damping
                }
            } else {
                lambda *= 10.0;
            }

            let g = jtr;
            if g.norm() < self.tol {
                break;
            }
        }
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lm_simple_least_squares() {
        // Minimize r(x) = [x - 3] -> x* = 3
        let x0 = DVec::from_slice(&[0.0]);
        let opt = LevenbergMarquardt::new();
        let result = opt.minimize(
            &x0,
            |x| DVec::from_slice(&[x[0] - 3.0]),
            |_| DMat::from_fn(1, 1, |_, _| 1.0),
        );
        assert!((result[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn lm_nonlinear() {
        // Minimize r(x) = [x^2 - 4] -> x* = ±2
        let x0 = DVec::from_slice(&[1.0]);
        let opt = LevenbergMarquardt::new();
        let result = opt.minimize(
            &x0,
            |x| DVec::from_slice(&[x[0] * x[0] - 4.0]),
            |x| DMat::from_fn(1, 1, |_, _| 2.0 * x[0]),
        );
        assert!((result[0] - 2.0).abs() < 1e-6, "expected 2.0, got {}", result[0]);
    }
}
