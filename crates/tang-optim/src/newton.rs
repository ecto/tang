use tang_la::{DVec, DMat, Lu};

/// Newton's method with line search.
pub struct Newton {
    pub max_iter: usize,
    pub tol: f64,
}

impl Newton {
    pub fn new() -> Self {
        Self { max_iter: 100, tol: 1e-10 }
    }

    /// Minimize f(x) given gradient and Hessian functions.
    /// Returns the minimizer.
    pub fn minimize<G, H>(
        &self,
        x0: &DVec<f64>,
        grad_fn: G,
        hessian_fn: H,
    ) -> DVec<f64>
    where
        G: Fn(&DVec<f64>) -> DVec<f64>,
        H: Fn(&DVec<f64>) -> DMat<f64>,
    {
        let mut x = x0.clone();
        for _ in 0..self.max_iter {
            let g = grad_fn(&x);
            if g.norm() < self.tol {
                break;
            }
            let h = hessian_fn(&x);
            if let Some(lu) = Lu::new(&h) {
                let dir = lu.solve(&g);
                // x -= dir (Newton step)
                for i in 0..x.len() {
                    x[i] = x[i] - dir[i];
                }
            } else {
                // Fallback to gradient descent if Hessian is singular
                for i in 0..x.len() {
                    x[i] = x[i] - 0.01 * g[i];
                }
            }
        }
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newton_quadratic() {
        // Minimize f(x, y) = x^2 + y^2
        // grad = [2x, 2y], hessian = [[2, 0], [0, 2]]
        let x0 = DVec::from_slice(&[5.0, 3.0]);
        let opt = Newton::new();
        let result = opt.minimize(
            &x0,
            |x| DVec::from_fn(2, |i| 2.0 * x[i]),
            |_| DMat::from_fn(2, 2, |i, j| if i == j { 2.0 } else { 0.0 }),
        );
        assert!(result[0].abs() < 1e-10);
        assert!(result[1].abs() < 1e-10);
    }
}
