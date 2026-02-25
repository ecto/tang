use tang_la::DVec;
use alloc::collections::VecDeque;

/// Limited-memory BFGS optimizer.
pub struct Lbfgs {
    pub lr: f64,
    pub m: usize,  // history size
    s_hist: VecDeque<DVec<f64>>,
    y_hist: VecDeque<DVec<f64>>,
    prev_x: Option<DVec<f64>>,
    prev_g: Option<DVec<f64>>,
}

impl Lbfgs {
    pub fn new(lr: f64, m: usize) -> Self {
        Self {
            lr, m,
            s_hist: VecDeque::new(),
            y_hist: VecDeque::new(),
            prev_x: None,
            prev_g: None,
        }
    }

    /// Compute L-BFGS direction from gradient history.
    fn direction(&self, grad: &DVec<f64>) -> DVec<f64> {
        let k = self.s_hist.len();
        if k == 0 {
            return grad * (-self.lr);
        }

        let mut q = grad.clone();
        let mut alphas = alloc::vec![0.0; k];

        // First loop (newest to oldest)
        for i in (0..k).rev() {
            let rho = 1.0 / self.y_hist[i].dot(&self.s_hist[i]);
            alphas[i] = rho * self.s_hist[i].dot(&q);
            q.axpy(-alphas[i], &self.y_hist[i]);
        }

        // Scale by H0 = (s^T y / y^T y) * I
        let last = k - 1;
        let gamma = self.s_hist[last].dot(&self.y_hist[last])
            / self.y_hist[last].dot(&self.y_hist[last]);
        q *= gamma;

        // Second loop (oldest to newest)
        for i in 0..k {
            let rho = 1.0 / self.y_hist[i].dot(&self.s_hist[i]);
            let beta = rho * self.y_hist[i].dot(&q);
            q.axpy(alphas[i] - beta, &self.s_hist[i]);
        }

        &q * (-1.0)
    }

    /// Update parameters given current gradient.
    pub fn step(&mut self, params: &mut DVec<f64>, grad: &DVec<f64>) {
        // Update history
        if let (Some(prev_x), Some(prev_g)) = (&self.prev_x, &self.prev_g) {
            let s = &*params - prev_x;
            let y = &*grad - prev_g;
            let sy = s.dot(&y);
            if sy > 1e-10 {
                if self.s_hist.len() >= self.m {
                    self.s_hist.pop_front();
                    self.y_hist.pop_front();
                }
                self.s_hist.push_back(s);
                self.y_hist.push_back(y);
            }
        }

        self.prev_x = Some(params.clone());
        self.prev_g = Some(grad.clone());

        let dir = self.direction(grad);
        for i in 0..params.len() {
            params[i] = params[i] + self.lr * dir[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lbfgs_converges() {
        // Minimize f(x, y) = x^2 + y^2
        let mut params = DVec::from_slice(&[5.0, 3.0]);
        let mut opt = Lbfgs::new(1.0, 5);
        for _ in 0..100 {
            let g = DVec::from_fn(2, |i| 2.0 * params[i]);
            opt.step(&mut params, &g);
        }
        assert!(params[0].abs() < 0.01, "x should converge, got {}", params[0]);
        assert!(params[1].abs() < 0.01, "y should converge, got {}", params[1]);
    }
}
