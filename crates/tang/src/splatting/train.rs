//! Training utilities for 3D Gaussian Splatting.
//!
//! Provides per-parameter-group Adam optimizer and loss functions.

/// Per-parameter-group Adam optimizer for f32 parameters.
///
/// Each parameter group (position, scale, rotation, opacity, SH) gets
/// its own learning rate while sharing Adam state.
pub struct AdamParam {
    pub lr: f32,
    m: Vec<f32>,
    v: Vec<f32>,
    t: u32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    /// Maximum gradient magnitude (0 = no clipping).
    pub max_grad: f32,
}

impl AdamParam {
    pub fn new(n: usize, lr: f32) -> Self {
        Self {
            lr,
            m: vec![0.0; n],
            v: vec![0.0; n],
            t: 0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-15,
            max_grad: 0.0,
        }
    }

    /// Create with gradient clipping.
    pub fn with_clip(n: usize, lr: f32, max_grad: f32) -> Self {
        let mut adam = Self::new(n, lr);
        adam.max_grad = max_grad;
        adam
    }

    /// Apply one Adam step in-place.
    pub fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);

        for i in 0..params.len() {
            let mut g = grads[i];
            if g.is_nan() || g.is_infinite() {
                continue;
            }
            if self.max_grad > 0.0 {
                g = g.clamp(-self.max_grad, self.max_grad);
            }
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            params[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }

    /// Grow optimizer state to accommodate new parameters (from densification).
    pub fn grow(&mut self, new_len: usize) {
        self.m.resize(new_len, 0.0);
        self.v.resize(new_len, 0.0);
    }
}

/// Training configuration for 3DGS scene optimization.
pub struct TrainConfig {
    /// Position learning rate.
    pub lr_position: f32,
    /// Scale learning rate.
    pub lr_scale: f32,
    /// Rotation learning rate.
    pub lr_rotation: f32,
    /// Opacity learning rate.
    pub lr_opacity: f32,
    /// SH coefficient learning rate.
    pub lr_sh: f32,
    /// Weight of L1 loss (vs D-SSIM).
    pub lambda_l1: f32,
    /// Total training iterations.
    pub iterations: u32,
    /// Start densifying at this iteration.
    pub densify_from: u32,
    /// Stop densifying at this iteration.
    pub densify_until: u32,
    /// Densify every N iterations.
    pub densify_interval: u32,
    /// Reset opacity to this logit value during densification.
    pub opacity_reset_logit: f32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            lr_position: 1.6e-4,
            lr_scale: 5e-3,
            lr_rotation: 1e-3,
            lr_opacity: 5e-2,
            lr_sh: 2.5e-3,
            lambda_l1: 0.8,
            iterations: 7000,
            densify_from: 500,
            densify_until: 5000,
            densify_interval: 100,
            opacity_reset_logit: -2.0, // sigmoid(-2) ≈ 0.12
        }
    }
}

/// Compute L1 loss and gradient w.r.t. rendered image.
///
/// loss = mean(|rendered - target|)
/// dL/d(rendered) = sign(rendered - target) / N
pub fn l1_loss_grad(rendered: &[f32], target: &[f32]) -> (f32, Vec<f32>) {
    let n = rendered.len() as f32;
    let mut loss = 0.0f32;
    let mut grad = Vec::with_capacity(rendered.len());

    for (r, t) in rendered.iter().zip(target.iter()) {
        let diff = r - t;
        loss += diff.abs();
        grad.push(diff.signum() / n);
    }

    (loss / n, grad)
}

/// Compute a simple box-filter SSIM approximation and gradient.
///
/// Uses 11x11 patches. Returns (1 - SSIM) / 2 as loss (so 0 = perfect).
/// This is D-SSIM from the 3DGS paper.
pub fn dssim_loss_grad(
    rendered: &[f32],
    target: &[f32],
    width: u32,
    height: u32,
) -> (f32, Vec<f32>) {
    let c1: f32 = 0.01 * 0.01; // (k1*L)^2 with L=1
    let c2: f32 = 0.03 * 0.03;
    let w = width as usize;
    let h = height as usize;
    let patch = 5i32; // half-window (11x11 total)

    let mut grad = vec![0.0f32; rendered.len()];
    let mut total_ssim = 0.0f32;
    let mut count = 0u32;

    // For each pixel, compute local SSIM over patch
    for py in 0..h {
        for px in 0..w {
            let y0 = (py as i32 - patch).max(0) as usize;
            let y1 = ((py as i32 + patch + 1) as usize).min(h);
            let x0 = (px as i32 - patch).max(0) as usize;
            let x1 = ((px as i32 + patch + 1) as usize).min(w);
            let area = ((y1 - y0) * (x1 - x0)) as f32;

            for c in 0..3 {
                let mut mu_r = 0.0f32;
                let mut mu_t = 0.0f32;
                let mut sig_rr = 0.0f32;
                let mut sig_tt = 0.0f32;
                let mut sig_rt = 0.0f32;

                for y in y0..y1 {
                    for x in x0..x1 {
                        let idx = (y * w + x) * 3 + c;
                        mu_r += rendered[idx];
                        mu_t += target[idx];
                    }
                }
                mu_r /= area;
                mu_t /= area;

                for y in y0..y1 {
                    for x in x0..x1 {
                        let idx = (y * w + x) * 3 + c;
                        let dr = rendered[idx] - mu_r;
                        let dt = target[idx] - mu_t;
                        sig_rr += dr * dr;
                        sig_tt += dt * dt;
                        sig_rt += dr * dt;
                    }
                }
                sig_rr /= area;
                sig_tt /= area;
                sig_rt /= area;

                let num = (2.0 * mu_r * mu_t + c1) * (2.0 * sig_rt + c2);
                let den = (mu_r * mu_r + mu_t * mu_t + c1) * (sig_rr + sig_tt + c2);
                let ssim = num / den;
                total_ssim += ssim;
                count += 1;

                // Gradient of -SSIM w.r.t. each rendered pixel in the patch
                let idx_center = (py * w + px) * 3 + c;
                let d_ssim_d_mu_r = {
                    let d_num = 2.0 * mu_t * (2.0 * sig_rt + c2);
                    let d_den = 2.0 * mu_r * (sig_rr + sig_tt + c2);
                    (d_num * den - num * d_den) / (den * den)
                };
                // For center pixel's contribution to mu_r
                grad[idx_center] += -d_ssim_d_mu_r / area;
            }
        }
    }

    let dssim = (1.0 - total_ssim / count as f32) / 2.0;
    // Normalize gradient
    let scale = 1.0 / (count as f32 / 3.0);
    for g in grad.iter_mut() {
        *g *= scale;
    }

    (dssim, grad)
}

/// Combined loss: lambda * L1 + (1 - lambda) * D-SSIM.
pub fn combined_loss_grad(
    rendered: &[f32],
    target: &[f32],
    width: u32,
    height: u32,
    lambda_l1: f32,
) -> (f32, Vec<f32>) {
    let (l1, g_l1) = l1_loss_grad(rendered, target);
    let (dssim, g_dssim) = dssim_loss_grad(rendered, target, width, height);

    let loss = lambda_l1 * l1 + (1.0 - lambda_l1) * dssim;
    let grad: Vec<f32> = g_l1
        .iter()
        .zip(g_dssim.iter())
        .map(|(a, b)| lambda_l1 * a + (1.0 - lambda_l1) * b)
        .collect();

    (loss, grad)
}
