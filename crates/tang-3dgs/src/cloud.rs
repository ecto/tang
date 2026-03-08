//! Gaussian cloud representation.
//!
//! A cloud is a collection of 3D gaussians, each with position, covariance
//! (parameterized as scale + rotation quaternion), opacity, and spherical
//! harmonics color coefficients.

use bytemuck::{Pod, Zeroable};

/// A single 3D gaussian (SoA layout stored in GaussianCloud).
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct GaussianParams {
    /// 3D position (mean).
    pub position: [f32; 3],
    pub _pad0: f32,
    /// Scale (sx, sy, sz) — exponentiated during rendering for positivity.
    pub scale: [f32; 3],
    pub _pad1: f32,
    /// Rotation quaternion (w, x, y, z) — normalized during rendering.
    pub rotation: [f32; 4],
    /// Opacity (logit) — sigmoid-activated during rendering.
    pub opacity: f32,
    pub _pad2: [f32; 3],
}

/// A cloud of 3D gaussians.
///
/// Stores all parameters in Structure-of-Arrays layout for GPU efficiency.
/// SH coefficients stored separately due to variable size.
#[derive(Debug, Clone)]
pub struct GaussianCloud {
    /// Number of gaussians.
    pub count: usize,
    /// Positions [N, 3].
    pub positions: Vec<[f32; 3]>,
    /// Log-scales [N, 3] (exponentiated for actual scale).
    pub scales: Vec<[f32; 3]>,
    /// Rotation quaternions [N, 4] (w, x, y, z).
    pub rotations: Vec<[f32; 4]>,
    /// Opacity logits [N] (sigmoid for actual opacity).
    pub opacities: Vec<f32>,
    /// Spherical harmonics coefficients [N, C] where C = 3 * (sh_degree+1)^2.
    /// Stored flat: [g0_c0, g0_c1, ..., g1_c0, g1_c1, ...].
    pub sh_coeffs: Vec<f32>,
    /// SH degree (0-3).
    pub sh_degree: u32,
}

impl GaussianCloud {
    /// Number of SH coefficients per gaussian per color channel.
    pub fn sh_coeffs_per_channel(&self) -> usize {
        let d = self.sh_degree as usize + 1;
        d * d
    }

    /// Total SH coefficients per gaussian (3 channels).
    pub fn sh_coeffs_per_gaussian(&self) -> usize {
        self.sh_coeffs_per_channel() * 3
    }

    /// Create a cloud with random gaussians for testing.
    pub fn random(count: usize, sh_degree: u32) -> Self {
        use std::f32::consts::PI;
        let mut positions = Vec::with_capacity(count);
        let mut scales = Vec::with_capacity(count);
        let mut rotations = Vec::with_capacity(count);
        let mut opacities = Vec::with_capacity(count);

        let sh_per_g = 3 * ((sh_degree + 1) * (sh_degree + 1)) as usize;
        let mut sh_coeffs = Vec::with_capacity(count * sh_per_g);

        for i in 0..count {
            let t = i as f32 / count as f32;
            let angle = t * 2.0 * PI;
            positions.push([angle.cos() * 2.0, angle.sin() * 2.0, 0.0]);
            scales.push([-3.0, -3.0, -3.0]); // log-scale, exp(-3) ≈ 0.05
            rotations.push([1.0, 0.0, 0.0, 0.0]); // identity quaternion
            opacities.push(2.0); // sigmoid(2) ≈ 0.88

            // DC component only (first 3 SH coeffs = RGB color)
            let r = (t * 3.0).sin().abs();
            let g = (t * 5.0 + 1.0).sin().abs();
            let b = (t * 7.0 + 2.0).sin().abs();
            sh_coeffs.push(r);
            sh_coeffs.push(g);
            sh_coeffs.push(b);
            for _ in 3..sh_per_g {
                sh_coeffs.push(0.0);
            }
        }

        Self {
            count,
            positions,
            scales,
            rotations,
            opacities,
            sh_coeffs,
            sh_degree,
        }
    }
}
