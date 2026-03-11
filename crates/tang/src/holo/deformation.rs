//! Deformation MLPs for audio-driven gaussian animation.
//!
//! Given conditioned spatial features (from cross-attention), predicts
//! per-gaussian deformations: position offset, rotation delta, scale delta.

use super::canonical::Mlp;

/// Deformation predictor: conditioned features → gaussian deformations.
///
/// Output layout per gaussian (10 values):
/// - [0..3]: position delta (dx, dy, dz)
/// - [3..6]: scale delta (dsx, dsy, dsz) — added to canonical log-scale
/// - [6..10]: rotation delta quaternion (dw, dx, dy, dz) — composed with canonical
pub struct DeformationPredictor {
    pub mlp: Mlp,
}

/// Deformation output dimension.
pub const DEFORM_OUTPUT_DIM: usize = 10;

impl DeformationPredictor {
    /// Create a deformation predictor.
    pub fn new(feature_dim: usize) -> Self {
        Self {
            mlp: Mlp::new(&[feature_dim, 64, 32, DEFORM_OUTPUT_DIM]),
        }
    }

    /// Predict deformations for a set of gaussians.
    ///
    /// `conditioned_features`: [N, feature_dim] from cross-attention.
    /// Returns deformations to apply to canonical gaussians.
    pub fn predict(&self, features: &[f32], feature_dim: usize) -> Deformations {
        let n = features.len() / feature_dim;
        let mut pos_deltas = Vec::with_capacity(n);
        let mut scale_deltas = Vec::with_capacity(n);
        let mut rot_deltas = Vec::with_capacity(n);

        for i in 0..n {
            let feat = &features[i * feature_dim..(i + 1) * feature_dim];
            let out = self.mlp.forward(feat);

            // Small deformations (scaled down)
            pos_deltas.push([out[0] * 0.05, out[1] * 0.05, out[2] * 0.05]);
            scale_deltas.push([out[3] * 0.1, out[4] * 0.1, out[5] * 0.1]);

            // Rotation delta quaternion (small rotation near identity)
            let dw = 1.0 + out[6] * 0.01;
            let dx = out[7] * 0.01;
            let dy = out[8] * 0.01;
            let dz = out[9] * 0.01;
            let norm = (dw * dw + dx * dx + dy * dy + dz * dz).sqrt().max(1e-8);
            rot_deltas.push([dw / norm, dx / norm, dy / norm, dz / norm]);
        }

        Deformations {
            position_deltas: pos_deltas,
            scale_deltas,
            rotation_deltas: rot_deltas,
        }
    }

    pub fn num_params(&self) -> usize {
        self.mlp.num_params()
    }

    pub fn params_flat(&self) -> Vec<f32> {
        self.mlp.params_flat()
    }

    pub fn set_params_flat(&mut self, params: &[f32]) {
        self.mlp.set_params_flat(params);
    }
}

/// Predicted gaussian deformations.
pub struct Deformations {
    pub position_deltas: Vec<[f32; 3]>,
    pub scale_deltas: Vec<[f32; 3]>,
    pub rotation_deltas: Vec<[f32; 4]>,
}

impl Deformations {
    /// Apply deformations to canonical gaussians, producing deformed cloud.
    pub fn apply(&self, canonical: &crate::splatting::GaussianCloud) -> crate::splatting::GaussianCloud {
        let n = canonical.count;
        let mut positions = Vec::with_capacity(n);
        let mut scales = Vec::with_capacity(n);
        let mut rotations = Vec::with_capacity(n);

        for i in 0..n {
            // Position: canonical + delta
            positions.push([
                canonical.positions[i][0] + self.position_deltas[i][0],
                canonical.positions[i][1] + self.position_deltas[i][1],
                canonical.positions[i][2] + self.position_deltas[i][2],
            ]);

            // Scale: canonical + delta (in log-space)
            scales.push([
                canonical.scales[i][0] + self.scale_deltas[i][0],
                canonical.scales[i][1] + self.scale_deltas[i][1],
                canonical.scales[i][2] + self.scale_deltas[i][2],
            ]);

            // Rotation: quaternion multiplication (delta * canonical)
            rotations.push(quat_mul(self.rotation_deltas[i], canonical.rotations[i]));
        }

        crate::splatting::GaussianCloud {
            count: n,
            positions,
            scales,
            rotations,
            opacities: canonical.opacities.clone(),
            sh_coeffs: canonical.sh_coeffs.clone(),
            sh_degree: canonical.sh_degree,
        }
    }
}

/// Quaternion multiplication: q1 * q2.
fn quat_mul(q1: [f32; 4], q2: [f32; 4]) -> [f32; 4] {
    let [w1, x1, y1, z1] = q1;
    let [w2, x2, y2, z2] = q2;
    [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quat_mul_identity() {
        let id = [1.0, 0.0, 0.0, 0.0];
        let q = [0.5, 0.5, 0.5, 0.5];
        let result = quat_mul(id, q);
        for i in 0..4 {
            assert!((result[i] - q[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_deformation_predict() {
        let pred = DeformationPredictor::new(32);
        let features = vec![0.1f32; 5 * 32];
        let deforms = pred.predict(&features, 32);
        assert_eq!(deforms.position_deltas.len(), 5);
    }
}
