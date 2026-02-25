use tang_la::DVec;

/// Adam optimizer.
pub struct Adam {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    m: Option<DVec<f64>>,
    v: Option<DVec<f64>>,
    t: usize,
}

impl Adam {
    pub fn new(lr: f64) -> Self {
        Self {
            lr, beta1: 0.9, beta2: 0.999, epsilon: 1e-8,
            m: None, v: None, t: 0,
        }
    }

    pub fn with_betas(lr: f64, beta1: f64, beta2: f64) -> Self {
        Self {
            lr, beta1, beta2, epsilon: 1e-8,
            m: None, v: None, t: 0,
        }
    }

    pub fn step(&mut self, params: &mut DVec<f64>, grad: &DVec<f64>) {
        let n = params.len();
        self.t += 1;
        let m = self.m.get_or_insert_with(|| DVec::zeros(n));
        let v = self.v.get_or_insert_with(|| DVec::zeros(n));

        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);

        for i in 0..n {
            m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * grad[i];
            v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * grad[i] * grad[i];
            let m_hat = m[i] / bc1;
            let v_hat = v[i] / bc2;
            params[i] = params[i] - self.lr * m_hat / (v_hat.sqrt() + self.epsilon);
        }
    }
}

/// AdamW optimizer (Adam with decoupled weight decay).
pub struct AdamW {
    inner: Adam,
    pub weight_decay: f64,
}

impl AdamW {
    pub fn new(lr: f64, weight_decay: f64) -> Self {
        Self { inner: Adam::new(lr), weight_decay }
    }

    pub fn step(&mut self, params: &mut DVec<f64>, grad: &DVec<f64>) {
        // Weight decay: params *= (1 - lr * wd)
        let factor = 1.0 - self.inner.lr * self.weight_decay;
        for i in 0..params.len() {
            params[i] = params[i] * factor;
        }
        self.inner.step(params, grad);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adam_converges() {
        // Minimize f(x) = x^2, grad = 2x
        let mut params = DVec::from_slice(&[5.0]);
        let mut opt = Adam::new(0.1);
        for _ in 0..1000 {
            let g = DVec::from_slice(&[2.0 * params[0]]);
            opt.step(&mut params, &g);
        }
        assert!(params[0].abs() < 0.01, "should converge near 0, got {}", params[0]);
    }
}
