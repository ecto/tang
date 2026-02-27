use crate::Parameter;
use alloc::vec::Vec;

/// Trait for optimizers that update module parameters directly.
pub trait Optimizer {
    fn step(&mut self, params: &mut [&mut Parameter<f64>]);
    /// Update the learning rate (used by schedulers).
    fn set_lr(&mut self, lr: f64);
}

/// Adam optimizer for Module parameters.
pub struct ModuleAdam {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    m: Vec<Vec<f64>>,
    v: Vec<Vec<f64>>,
    t: usize,
}

impl ModuleAdam {
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        }
    }

    pub fn with_betas(lr: f64, beta1: f64, beta2: f64) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            epsilon: 1e-8,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        }
    }
}

impl Optimizer for ModuleAdam {
    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    fn step(&mut self, params: &mut [&mut Parameter<f64>]) {
        self.t += 1;

        // Initialize state vectors if needed
        if self.m.is_empty() {
            for p in params.iter() {
                let n = p.data.numel();
                self.m.push(alloc::vec![0.0; n]);
                self.v.push(alloc::vec![0.0; n]);
            }
        }

        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);

        for (i, p) in params.iter_mut().enumerate() {
            if let Some(grad) = &p.grad {
                let data = p.data.data_mut();
                let grad_data = grad.data();

                for j in 0..data.len() {
                    let g = grad_data[j];
                    self.m[i][j] = self.beta1 * self.m[i][j] + (1.0 - self.beta1) * g;
                    self.v[i][j] = self.beta2 * self.v[i][j] + (1.0 - self.beta2) * g * g;
                    let m_hat = self.m[i][j] / bc1;
                    let v_hat = self.v[i][j] / bc2;
                    data[j] -= self.lr * m_hat / (v_hat.sqrt() + self.epsilon);
                }
            }
        }
    }
}

/// SGD optimizer for Module parameters.
pub struct ModuleSgd {
    pub lr: f64,
    pub momentum: f64,
    velocity: Vec<Vec<f64>>,
}

impl ModuleSgd {
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            momentum: 0.0,
            velocity: Vec::new(),
        }
    }

    pub fn with_momentum(lr: f64, momentum: f64) -> Self {
        Self {
            lr,
            momentum,
            velocity: Vec::new(),
        }
    }
}

impl Optimizer for ModuleSgd {
    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    fn step(&mut self, params: &mut [&mut Parameter<f64>]) {
        // Initialize velocity if using momentum
        if self.momentum > 0.0 && self.velocity.is_empty() {
            for p in params.iter() {
                self.velocity.push(alloc::vec![0.0; p.data.numel()]);
            }
        }

        for (i, p) in params.iter_mut().enumerate() {
            if let Some(grad) = &p.grad {
                let data = p.data.data_mut();
                let grad_data = grad.data();

                if self.momentum > 0.0 {
                    let vel = &mut self.velocity[i];
                    for j in 0..data.len() {
                        vel[j] = self.momentum * vel[j] + grad_data[j];
                        data[j] -= self.lr * vel[j];
                    }
                } else {
                    for j in 0..data.len() {
                        data[j] -= self.lr * grad_data[j];
                    }
                }
            }
        }
    }
}
