//! Canonical gaussian predictor.
//!
//! Given triplane features for a set of anchor points, predicts canonical
//! (rest-pose) gaussian parameters: position offset, scale, rotation, opacity, SH.
//!
//! Architecture: feature → MLP → gaussian params

/// Simple MLP with ReLU activations.
pub struct Mlp {
    /// Layer weights: [layers][out_dim * in_dim], row-major.
    pub weights: Vec<Vec<f32>>,
    /// Layer biases: [layers][out_dim].
    pub biases: Vec<Vec<f32>>,
    /// Layer dimensions: [(in, out), ...].
    pub dims: Vec<(usize, usize)>,
}

impl Mlp {
    /// Create an MLP with given layer sizes (e.g., [32, 64, 64, 14]).
    pub fn new(layer_sizes: &[usize]) -> Self {
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        let mut dims = Vec::new();

        let mut seed = 42u64;
        let mut rand = || -> f32 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let bits = ((seed >> 33) as u32) as f32 / u32::MAX as f32;
            (bits - 0.5) * 2.0
        };

        for i in 0..layer_sizes.len() - 1 {
            let in_dim = layer_sizes[i];
            let out_dim = layer_sizes[i + 1];
            dims.push((in_dim, out_dim));

            // Kaiming init
            let scale = (2.0 / in_dim as f32).sqrt();
            let w: Vec<f32> = (0..out_dim * in_dim).map(|_| rand() * scale).collect();
            let b: Vec<f32> = vec![0.0; out_dim];
            weights.push(w);
            biases.push(b);
        }

        Self {
            weights,
            biases,
            dims,
        }
    }

    /// Forward pass for a single input.
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut x = input.to_vec();
        for (i, ((in_dim, out_dim), (w, b))) in
            self.dims.iter().zip(self.weights.iter().zip(self.biases.iter())).enumerate()
        {
            let mut y = vec![0.0f32; *out_dim];
            for j in 0..*out_dim {
                let mut sum = b[j];
                for k in 0..*in_dim {
                    sum += w[j * in_dim + k] * x[k];
                }
                // ReLU for all layers except last
                if i < self.dims.len() - 1 {
                    sum = sum.max(0.0);
                }
                y[j] = sum;
            }
            x = y;
        }
        x
    }

    /// Total number of parameters.
    pub fn num_params(&self) -> usize {
        self.weights.iter().map(|w| w.len()).sum::<usize>()
            + self.biases.iter().map(|b| b.len()).sum::<usize>()
    }

    /// Flatten all parameters for optimizer.
    pub fn params_flat(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.num_params());
        for (w, b) in self.weights.iter().zip(self.biases.iter()) {
            out.extend_from_slice(w);
            out.extend_from_slice(b);
        }
        out
    }

    /// Set parameters from flat slice.
    pub fn set_params_flat(&mut self, params: &[f32]) {
        let mut offset = 0;
        for (w, b) in self.weights.iter_mut().zip(self.biases.iter_mut()) {
            let wlen = w.len();
            w.copy_from_slice(&params[offset..offset + wlen]);
            offset += wlen;
            let blen = b.len();
            b.copy_from_slice(&params[offset..offset + blen]);
            offset += blen;
        }
    }
}

/// Canonical gaussian predictor: triplane features → gaussian parameters.
///
/// Output layout per gaussian (14 values):
/// - [0..3]: position offset (dx, dy, dz)
/// - [3..6]: log-scale (sx, sy, sz)
/// - [6..10]: rotation quaternion (w, x, y, z)
/// - [10]: opacity logit
/// - [11..14]: SH DC color (r, g, b)
pub struct CanonicalPredictor {
    pub mlp: Mlp,
}

/// Output dimension: 3 pos + 3 scale + 4 rot + 1 opacity + 3 sh = 14
pub const CANONICAL_OUTPUT_DIM: usize = 14;

impl CanonicalPredictor {
    /// Create a canonical predictor with given feature dimension.
    pub fn new(feature_dim: usize) -> Self {
        let mut mlp = Mlp::new(&[feature_dim, 64, 64, CANONICAL_OUTPUT_DIM]);

        // Set sensible biases on last layer so random weights produce visible gaussians:
        // [0..3] pos offset: 0 (anchors are already positioned)
        // [3..6] log-scale: -3.0 (exp(-3)≈0.05, small gaussians)
        // [6..10] rotation quat: [1,0,0,0] (identity — avoids NaN from normalize([0,0,0,0]))
        // [10] opacity logit: 2.0 (sigmoid(2)≈0.88, visible)
        // [11..14] SH color: 0.0 (neutral gray after SH decode)
        let last = mlp.biases.last_mut().unwrap();
        last[3] = -3.0;
        last[4] = -3.0;
        last[5] = -3.0;
        last[6] = 1.0; // qw
        last[10] = 2.0; // opacity

        Self { mlp }
    }

    /// Predict canonical gaussian parameters from triplane features.
    ///
    /// Returns (position_offsets, scales, rotations, opacities, sh_coeffs).
    pub fn predict(
        &self,
        features: &[f32],
        feature_dim: usize,
        anchor_positions: &[[f32; 3]],
    ) -> PredictedGaussians {
        let n = anchor_positions.len();
        let mut positions = Vec::with_capacity(n);
        let mut scales = Vec::with_capacity(n);
        let mut rotations = Vec::with_capacity(n);
        let mut opacities = Vec::with_capacity(n);
        let mut sh_coeffs = Vec::with_capacity(n * 3);

        for i in 0..n {
            let feat = &features[i * feature_dim..(i + 1) * feature_dim];
            let out = self.mlp.forward(feat);

            // Position = anchor + offset
            positions.push([
                anchor_positions[i][0] + out[0] * 0.1,
                anchor_positions[i][1] + out[1] * 0.1,
                anchor_positions[i][2] + out[2] * 0.1,
            ]);

            scales.push([out[3], out[4], out[5]]);

            // Normalize quaternion
            let qw = out[6];
            let qx = out[7];
            let qy = out[8];
            let qz = out[9];
            let qnorm = (qw * qw + qx * qx + qy * qy + qz * qz).sqrt().max(1e-8);
            rotations.push([qw / qnorm, qx / qnorm, qy / qnorm, qz / qnorm]);

            opacities.push(out[10]);

            sh_coeffs.push(out[11]);
            sh_coeffs.push(out[12]);
            sh_coeffs.push(out[13]);
        }

        PredictedGaussians {
            positions,
            scales,
            rotations,
            opacities,
            sh_coeffs,
        }
    }
}

/// Predicted canonical gaussian parameters.
pub struct PredictedGaussians {
    pub positions: Vec<[f32; 3]>,
    pub scales: Vec<[f32; 3]>,
    pub rotations: Vec<[f32; 4]>,
    pub opacities: Vec<f32>,
    pub sh_coeffs: Vec<f32>,
}

impl PredictedGaussians {
    /// Convert to a GaussianCloud for rendering.
    pub fn to_cloud(&self) -> tang_3dgs::GaussianCloud {
        tang_3dgs::GaussianCloud {
            count: self.positions.len(),
            positions: self.positions.clone(),
            scales: self.scales.clone(),
            rotations: self.rotations.clone(),
            opacities: self.opacities.clone(),
            sh_coeffs: self.sh_coeffs.clone(),
            sh_degree: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlp_forward() {
        let mlp = Mlp::new(&[4, 8, 3]);
        let input = vec![1.0, 0.5, -0.5, 0.0];
        let output = mlp.forward(&input);
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn test_canonical_predict() {
        let pred = CanonicalPredictor::new(32);
        let features = vec![0.1f32; 32 * 3]; // 3 points, 32-dim features
        let anchors = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let result = pred.predict(&features, 32, &anchors);
        assert_eq!(result.positions.len(), 3);
        assert_eq!(result.sh_coeffs.len(), 9);
    }
}
