use crate::Parameter;
use alloc::vec::Vec;
use tang::Scalar;

/// Trait for optimizers that update module parameters directly.
///
/// The optimizer accepts parameters of any scalar type `S`, but maintains
/// internal state (momentum, variance) in f64 for numerical stability.
/// This follows mixed-precision convention: optimizer state in high precision,
/// model weights in whatever precision the forward pass uses.
pub trait Optimizer<S: Scalar> {
    fn step(&mut self, params: &mut [&mut Parameter<S>]);
    /// Update the learning rate (used by schedulers).
    fn set_lr(&mut self, lr: f64);
}

/// Adam optimizer for Module parameters with optional decoupled weight decay (AdamW).
pub struct ModuleAdam {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
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
            weight_decay: 0.0,
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
            weight_decay: 0.0,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        }
    }

    /// Create an Adam optimizer with decoupled weight decay (AdamW).
    pub fn with_weight_decay(lr: f64, weight_decay: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        }
    }

    /// Return references to the internal state vectors (m, v) and step count.
    ///
    /// Each element of `m` and `v` is the momentum/variance buffer for one parameter.
    pub fn state_vecs(&self) -> (&[Vec<f64>], &[Vec<f64>], usize) {
        (&self.m, &self.v, self.t)
    }

    /// Load previously saved optimizer state.
    ///
    /// `m` and `v` must have the same structure as originally produced by training
    /// (one vector per parameter, same lengths).
    pub fn load_state_vecs(&mut self, m: Vec<Vec<f64>>, v: Vec<Vec<f64>>, t: usize) {
        self.m = m;
        self.v = v;
        self.t = t;
    }
}

impl<S: Scalar> Optimizer<S> for ModuleAdam {
    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    fn step(&mut self, params: &mut [&mut Parameter<S>]) {
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
                    // Decoupled weight decay (AdamW): applied before Adam update
                    if self.weight_decay > 0.0 {
                        let w = data[j].to_f64();
                        data[j] = S::from_f64(w * (1.0 - self.lr * self.weight_decay));
                    }

                    let g = grad_data[j].to_f64();
                    self.m[i][j] = self.beta1 * self.m[i][j] + (1.0 - self.beta1) * g;
                    self.v[i][j] = self.beta2 * self.v[i][j] + (1.0 - self.beta2) * g * g;
                    let m_hat = self.m[i][j] / bc1;
                    let v_hat = self.v[i][j] / bc2;
                    let update = self.lr * m_hat / (v_hat.sqrt() + self.epsilon);
                    data[j] = S::from_f64(data[j].to_f64() - update);
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

impl<S: Scalar> Optimizer<S> for ModuleSgd {
    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    fn step(&mut self, params: &mut [&mut Parameter<S>]) {
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
                        vel[j] = self.momentum * vel[j] + grad_data[j].to_f64();
                        data[j] = S::from_f64(data[j].to_f64() - self.lr * vel[j]);
                    }
                } else {
                    for j in 0..data.len() {
                        data[j] = S::from_f64(data[j].to_f64() - self.lr * grad_data[j].to_f64());
                    }
                }
            }
        }
    }
}
