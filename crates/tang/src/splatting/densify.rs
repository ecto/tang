//! Adaptive density control for gaussian clouds during training.
//!
//! Three operations, applied periodically during optimization:
//!
//! - **Clone**: Duplicate small gaussians with high position gradients
//!   (under-reconstructed regions need more gaussians)
//! - **Split**: Replace large gaussians with two smaller ones
//!   (over-reconstructed regions need finer detail)
//! - **Prune**: Remove gaussians with near-zero opacity
//!   (dead weight after optimization)
//!
//! Schedule: densify every 100 iterations from iter 1000 to 7000.

use super::cloud::GaussianCloud;

/// Accumulated gradient statistics for densification decisions.
pub struct DensifyStats {
    /// Accumulated position gradient magnitudes [N].
    pub grad_accum: Vec<f32>,
    /// Number of times each gaussian was visible [N].
    pub vis_count: Vec<u32>,
}

impl DensifyStats {
    pub fn new(n: usize) -> Self {
        Self {
            grad_accum: vec![0.0; n],
            vis_count: vec![0; n],
        }
    }

    /// Accumulate gradients from one training step.
    pub fn accumulate(&mut self, position_grads: &[[f32; 3]], visible: &[bool]) {
        for (i, (grad, &vis)) in position_grads.iter().zip(visible.iter()).enumerate() {
            if vis {
                let mag = (grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2]).sqrt();
                self.grad_accum[i] += mag;
                self.vis_count[i] += 1;
            }
        }
    }

    /// Average gradient per gaussian.
    pub fn avg_gradients(&self) -> Vec<f32> {
        self.grad_accum
            .iter()
            .zip(self.vis_count.iter())
            .map(|(&g, &c)| if c > 0 { g / c as f32 } else { 0.0 })
            .collect()
    }

    /// Reset accumulators.
    pub fn reset(&mut self) {
        self.grad_accum.fill(0.0);
        self.vis_count.fill(0);
    }
}

/// Densification thresholds.
pub struct DensifyConfig {
    /// Gradient magnitude threshold for cloning/splitting.
    pub grad_threshold: f32,
    /// Scale threshold: gaussians larger than this get split (not cloned).
    pub scale_threshold: f32,
    /// Opacity threshold: gaussians below this get pruned.
    pub opacity_threshold: f32,
}

impl Default for DensifyConfig {
    fn default() -> Self {
        Self {
            grad_threshold: 0.001,
            scale_threshold: 0.01, // in world units
            opacity_threshold: 0.005,
        }
    }
}

/// Perform one densification step on the cloud.
///
/// Returns (new_cloud, index_mapping) where index_mapping maps old → new indices.
pub fn densify(
    cloud: &GaussianCloud,
    stats: &DensifyStats,
    config: &DensifyConfig,
) -> GaussianCloud {
    let avg_grads = stats.avg_gradients();
    let sh_per_g = cloud.sh_coeffs_per_gaussian();

    let mut new_positions = Vec::new();
    let mut new_scales = Vec::new();
    let mut new_rotations = Vec::new();
    let mut new_opacities = Vec::new();
    let mut new_sh = Vec::new();

    for i in 0..cloud.count {
        let opacity = 1.0 / (1.0 + (-cloud.opacities[i]).exp()); // sigmoid
        let scale_mag = cloud.scales[i].iter().map(|s| s.exp()).sum::<f32>() / 3.0;

        // Prune
        if opacity < config.opacity_threshold {
            continue;
        }

        let mut keep = |idx: usize| {
            new_positions.push(cloud.positions[idx]);
            new_scales.push(cloud.scales[idx]);
            new_rotations.push(cloud.rotations[idx]);
            new_opacities.push(cloud.opacities[idx]);
            let sh_start = idx * sh_per_g;
            new_sh.extend_from_slice(&cloud.sh_coeffs[sh_start..sh_start + sh_per_g]);
        };

        if avg_grads[i] > config.grad_threshold {
            if scale_mag > config.scale_threshold {
                // Split: replace with two smaller gaussians
                let new_scale = [
                    cloud.scales[i][0] - 1.6_f32.ln(),
                    cloud.scales[i][1] - 1.6_f32.ln(),
                    cloud.scales[i][2] - 1.6_f32.ln(),
                ];
                // First child at original position
                new_positions.push(cloud.positions[i]);
                new_scales.push(new_scale);
                new_rotations.push(cloud.rotations[i]);
                new_opacities.push(cloud.opacities[i]);
                let sh_start = i * sh_per_g;
                new_sh.extend_from_slice(&cloud.sh_coeffs[sh_start..sh_start + sh_per_g]);
                // Second child offset slightly
                let mut pos2 = cloud.positions[i];
                pos2[0] += scale_mag * 0.5;
                new_positions.push(pos2);
                new_scales.push(new_scale);
                new_rotations.push(cloud.rotations[i]);
                new_opacities.push(cloud.opacities[i]);
                new_sh.extend_from_slice(&cloud.sh_coeffs[sh_start..sh_start + sh_per_g]);
            } else {
                // Clone: duplicate at same position
                keep(i);
                keep(i);
            }
        } else {
            // Keep as-is
            keep(i);
        }
    }

    GaussianCloud {
        count: new_positions.len(),
        positions: new_positions,
        scales: new_scales,
        rotations: new_rotations,
        opacities: new_opacities,
        sh_coeffs: new_sh,
        sh_degree: cloud.sh_degree,
    }
}
