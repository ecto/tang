use core::f64::consts::PI;

/// Learning rate scheduler: returns the LR for a given epoch (0-indexed).
pub trait Scheduler {
    /// Return the learning rate for the given epoch (0-indexed).
    fn lr(&self, epoch: usize) -> f64;
}

/// Constant learning rate (no decay).
pub struct ConstantLr {
    pub lr: f64,
}

impl Scheduler for ConstantLr {
    fn lr(&self, _epoch: usize) -> f64 {
        self.lr
    }
}

/// Step decay: multiply by `gamma` every `step_size` epochs.
///
/// `lr(epoch) = initial_lr * gamma^(epoch / step_size)`
pub struct StepLr {
    pub initial_lr: f64,
    pub step_size: usize,
    pub gamma: f64,
}

impl Scheduler for StepLr {
    fn lr(&self, epoch: usize) -> f64 {
        let exponent = (epoch / self.step_size) as i32;
        self.initial_lr * self.gamma.powi(exponent)
    }
}

/// Cosine annealing from `initial_lr` down to `min_lr` over `total_epochs`.
///
/// `lr(epoch) = min_lr + (initial_lr - min_lr) * 0.5 * (1 + cos(pi * epoch / total_epochs))`
pub struct CosineAnnealingLr {
    pub initial_lr: f64,
    pub min_lr: f64,
    pub total_epochs: usize,
}

impl Scheduler for CosineAnnealingLr {
    fn lr(&self, epoch: usize) -> f64 {
        let t = epoch as f64 / self.total_epochs as f64;
        self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (1.0 + (PI * t).cos())
    }
}

/// Linear warmup for `warmup_epochs`, then cosine decay for the remaining epochs.
pub struct WarmupCosine {
    pub initial_lr: f64,
    pub min_lr: f64,
    pub warmup_epochs: usize,
    pub total_epochs: usize,
}

impl Scheduler for WarmupCosine {
    fn lr(&self, epoch: usize) -> f64 {
        if epoch < self.warmup_epochs {
            // Linear warmup: 0 -> initial_lr over warmup_epochs
            self.initial_lr * (epoch as f64 / self.warmup_epochs as f64)
        } else {
            // Cosine decay over the remaining epochs
            let decay_epochs = self.total_epochs - self.warmup_epochs;
            let t = (epoch - self.warmup_epochs) as f64 / decay_epochs as f64;
            self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (1.0 + (PI * t).cos())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_lr_returns_fixed() {
        let s = ConstantLr { lr: 0.01 };
        assert!((s.lr(0) - 0.01).abs() < 1e-12);
        assert!((s.lr(100) - 0.01).abs() < 1e-12);
    }

    #[test]
    fn step_lr_halves_every_10() {
        let s = StepLr {
            initial_lr: 1.0,
            step_size: 10,
            gamma: 0.5,
        };
        assert!((s.lr(0) - 1.0).abs() < 1e-12);
        assert!((s.lr(9) - 1.0).abs() < 1e-12);
        assert!((s.lr(10) - 0.5).abs() < 1e-12);
        assert!((s.lr(19) - 0.5).abs() < 1e-12);
        assert!((s.lr(20) - 0.25).abs() < 1e-12);
        assert!((s.lr(30) - 0.125).abs() < 1e-12);
    }

    #[test]
    fn cosine_annealing_starts_high_ends_at_min() {
        let s = CosineAnnealingLr {
            initial_lr: 1.0,
            min_lr: 0.0,
            total_epochs: 100,
        };
        // epoch 0: cos(0) = 1 -> lr = 0 + 1.0 * 0.5 * (1 + 1) = 1.0
        assert!((s.lr(0) - 1.0).abs() < 1e-12);
        // epoch 50: cos(pi/2) = 0 -> lr = 0.5
        assert!((s.lr(50) - 0.5).abs() < 1e-12);
        // epoch 100: cos(pi) = -1 -> lr = 0.0
        assert!((s.lr(100) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn cosine_annealing_with_nonzero_min() {
        let s = CosineAnnealingLr {
            initial_lr: 1.0,
            min_lr: 0.1,
            total_epochs: 100,
        };
        assert!((s.lr(0) - 1.0).abs() < 1e-12);
        assert!((s.lr(100) - 0.1).abs() < 1e-12);
    }

    #[test]
    fn warmup_cosine_increases_then_decreases() {
        let s = WarmupCosine {
            initial_lr: 1.0,
            min_lr: 0.0,
            warmup_epochs: 10,
            total_epochs: 110,
        };

        // epoch 0: warmup phase -> 0.0
        assert!((s.lr(0) - 0.0).abs() < 1e-12);
        // epoch 5: halfway through warmup -> 0.5
        assert!((s.lr(5) - 0.5).abs() < 1e-12);
        // epoch 10: end of warmup, start of cosine -> cos(0) = 1 -> 1.0
        assert!((s.lr(10) - 1.0).abs() < 1e-12);
        // epoch 60: midpoint of cosine decay (50/100) -> 0.5
        assert!((s.lr(60) - 0.5).abs() < 1e-12);
        // epoch 110: end of cosine decay -> 0.0
        assert!((s.lr(110) - 0.0).abs() < 1e-12);

        // LR should increase during warmup
        assert!(s.lr(3) < s.lr(7));
        // LR should decrease during cosine phase
        assert!(s.lr(20) > s.lr(80));
    }
}
