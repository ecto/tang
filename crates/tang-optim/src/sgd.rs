use tang_la::DVec;

/// Stochastic Gradient Descent with optional momentum.
pub struct Sgd {
    pub lr: f64,
    pub momentum: f64,
    velocity: Option<DVec<f64>>,
}

impl Sgd {
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            momentum: 0.0,
            velocity: None,
        }
    }

    pub fn with_momentum(lr: f64, momentum: f64) -> Self {
        Self {
            lr,
            momentum,
            velocity: None,
        }
    }

    /// Update parameters: params -= lr * grad (with momentum).
    pub fn step(&mut self, params: &mut DVec<f64>, grad: &DVec<f64>) {
        let n = params.len();
        if self.momentum > 0.0 {
            let v = self.velocity.get_or_insert_with(|| DVec::zeros(n));
            for i in 0..n {
                v[i] = self.momentum * v[i] + grad[i];
                params[i] = params[i] - self.lr * v[i];
            }
        } else {
            for i in 0..n {
                params[i] = params[i] - self.lr * grad[i];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sgd_basic() {
        let mut params = DVec::from_slice(&[5.0, 3.0]);
        let grad = DVec::from_slice(&[1.0, -1.0]);
        let mut opt = Sgd::new(0.1);
        opt.step(&mut params, &grad);
        assert!((params[0] - 4.9).abs() < 1e-10);
        assert!((params[1] - 3.1).abs() < 1e-10);
    }
}
