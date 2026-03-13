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

/// Warmup-Stable-Decay (WSD) learning rate schedule.
///
/// Three phases:
/// 1. **Warmup** (steps 0..warmup_steps): linear ramp from 0 to `initial_lr`.
/// 2. **Stable** (warmup_steps..stable_end): constant `initial_lr`.
/// 3. **Decay** (stable_end..total_steps): linear decay from `initial_lr` to `min_lr`.
///
/// The stable phase ends at `warmup_steps + stable_fraction * (total_steps - warmup_steps)`.
/// A typical split is 2% warmup, 78% stable, 20% decay (stable_fraction = 0.8).
///
/// Compared to cosine decay (which starts decaying immediately after warmup), WSD
/// keeps the LR at peak for longer, then does a sharp decay at the end to find a
/// better loss basin.
pub struct WarmupStableDecay {
    /// Peak learning rate (reached at end of warmup, held during stable phase).
    pub initial_lr: f64,
    /// Minimum learning rate (reached at end of decay phase).
    pub min_lr: f64,
    /// Number of warmup steps (linear ramp from 0).
    pub warmup_steps: usize,
    /// Fraction of post-warmup steps spent at peak LR. Default: 0.8.
    /// The remaining `(1 - stable_fraction)` fraction is the decay phase.
    pub stable_fraction: f64,
    /// Total number of training steps.
    pub total_steps: usize,
}

impl Scheduler for WarmupStableDecay {
    fn lr(&self, epoch: usize) -> f64 {
        if epoch < self.warmup_steps {
            // Phase 1: linear warmup 0 -> initial_lr
            self.initial_lr * (epoch as f64 / self.warmup_steps as f64)
        } else {
            let post_warmup = self.total_steps - self.warmup_steps;
            let stable_steps = (post_warmup as f64 * self.stable_fraction) as usize;
            let stable_end = self.warmup_steps + stable_steps;

            if epoch < stable_end {
                // Phase 2: constant at initial_lr
                self.initial_lr
            } else {
                // Phase 3: linear decay initial_lr -> min_lr
                let decay_steps = self.total_steps - stable_end;
                if decay_steps == 0 {
                    // No decay phase: stay at peak
                    return self.initial_lr;
                }
                let t = (epoch - stable_end) as f64 / decay_steps as f64;
                self.initial_lr + (self.min_lr - self.initial_lr) * t
            }
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
    fn wsd_warmup_phase() {
        let s = WarmupStableDecay {
            initial_lr: 1.0,
            min_lr: 0.0,
            warmup_steps: 100,
            stable_fraction: 0.8,
            total_steps: 1000,
        };
        // step 0: start of warmup -> 0.0
        assert!((s.lr(0) - 0.0).abs() < 1e-12);
        // step 50: halfway warmup -> 0.5
        assert!((s.lr(50) - 0.5).abs() < 1e-12);
        // step 100: end of warmup -> 1.0 (enters stable)
        assert!((s.lr(100) - 1.0).abs() < 1e-12);
        // LR increases during warmup
        assert!(s.lr(20) < s.lr(80));
    }

    #[test]
    fn wsd_stable_phase() {
        let s = WarmupStableDecay {
            initial_lr: 1.0,
            min_lr: 0.0,
            warmup_steps: 100,
            stable_fraction: 0.8,
            total_steps: 1000,
        };
        // stable_end = 100 + 0.8 * 900 = 820
        // Anywhere in stable phase should be initial_lr
        assert!((s.lr(100) - 1.0).abs() < 1e-12);
        assert!((s.lr(200) - 1.0).abs() < 1e-12);
        assert!((s.lr(500) - 1.0).abs() < 1e-12);
        assert!((s.lr(819) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn wsd_decay_phase() {
        let s = WarmupStableDecay {
            initial_lr: 1.0,
            min_lr: 0.0,
            warmup_steps: 100,
            stable_fraction: 0.8,
            total_steps: 1000,
        };
        // stable_end = 820, decay_steps = 180
        // step 820: start of decay -> 1.0
        assert!((s.lr(820) - 1.0).abs() < 1e-12);
        // step 910: midpoint of decay -> 0.5
        assert!((s.lr(910) - 0.5).abs() < 1e-12);
        // step 1000: end of decay -> 0.0
        assert!((s.lr(1000) - 0.0).abs() < 1e-12);
        // LR decreases during decay
        assert!(s.lr(850) > s.lr(950));
    }

    #[test]
    fn wsd_with_nonzero_min() {
        let s = WarmupStableDecay {
            initial_lr: 1.0,
            min_lr: 0.1,
            warmup_steps: 100,
            stable_fraction: 0.8,
            total_steps: 1000,
        };
        // stable_end = 820, decay_steps = 180
        // At end of decay, LR should reach min_lr
        assert!((s.lr(1000) - 0.1).abs() < 1e-12);
        // Midpoint of decay: 1.0 + (0.1 - 1.0) * 0.5 = 0.55
        assert!((s.lr(910) - 0.55).abs() < 1e-12);
        // Stable phase still at 1.0
        assert!((s.lr(500) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn wsd_typical_split() {
        // 2% warmup, 78% stable, 20% decay (typical WSD split)
        let total = 100_000;
        let warmup = 2000; // 2%
        // stable_fraction chosen so stable = 78% of total:
        // stable_steps = stable_fraction * (total - warmup) = 0.7959... * 98000 = 78000
        // Use 0.7959 to get ~78000 stable steps
        let s = WarmupStableDecay {
            initial_lr: 3e-4,
            min_lr: 3e-5,
            warmup_steps: warmup,
            stable_fraction: 0.8,
            total_steps: total,
        };

        // Warmup: linear ramp
        assert!(s.lr(0) < 1e-15);
        assert!((s.lr(1000) - 1.5e-4).abs() < 1e-10);
        assert!((s.lr(2000) - 3e-4).abs() < 1e-10);

        // Stable: constant at peak
        assert!((s.lr(10_000) - 3e-4).abs() < 1e-10);
        assert!((s.lr(50_000) - 3e-4).abs() < 1e-10);

        // Decay: drops to min
        assert!((s.lr(total) - 3e-5).abs() < 1e-10);

        // After warmup, never goes below min
        for step in (warmup..=total).step_by(1000) {
            assert!(s.lr(step) >= 3e-5 - 1e-15, "lr at step {} was {}", step, s.lr(step));
        }
    }

    #[test]
    fn wsd_zero_decay() {
        // stable_fraction = 1.0 means no decay phase at all
        let s = WarmupStableDecay {
            initial_lr: 1.0,
            min_lr: 0.0,
            warmup_steps: 10,
            stable_fraction: 1.0,
            total_steps: 100,
        };
        // After warmup, LR stays at initial_lr
        assert!((s.lr(50) - 1.0).abs() < 1e-12);
        assert!((s.lr(100) - 1.0).abs() < 1e-12);
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
