//! Spatial-audio cross-attention.
//!
//! Modulates triplane features using audio latent codes via cross-attention.
//! Audio embedding attends to spatial features to produce per-point
//! deformation conditions.

/// Cross-attention layer: spatial features × audio embedding → conditioned features.
///
/// Simplified single-head attention:
/// Q = spatial features (from triplane)
/// K, V = audio embedding (broadcast to all spatial positions)
/// Output = softmax(Q·K^T / √d) · V + residual
pub struct CrossAttention {
    /// Query projection weights [feature_dim, head_dim].
    pub w_q: Vec<f32>,
    /// Key projection weights [audio_dim, head_dim].
    pub w_k: Vec<f32>,
    /// Value projection weights [audio_dim, feature_dim].
    pub w_v: Vec<f32>,
    /// Feature dimension (spatial).
    pub feature_dim: usize,
    /// Audio latent dimension.
    pub audio_dim: usize,
    /// Attention head dimension.
    pub head_dim: usize,
}

impl CrossAttention {
    pub fn new(feature_dim: usize, audio_dim: usize, head_dim: usize) -> Self {
        let scale = 0.02;
        let mut seed = 77u64;
        let mut rand = || -> f32 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as u32) as f32 / u32::MAX as f32 * 2.0 * scale - scale
        };

        Self {
            w_q: (0..feature_dim * head_dim).map(|_| rand()).collect(),
            w_k: (0..audio_dim * head_dim).map(|_| rand()).collect(),
            w_v: (0..audio_dim * feature_dim).map(|_| rand()).collect(),
            feature_dim,
            audio_dim,
            head_dim,
        }
    }

    /// Apply cross-attention to condition spatial features on audio.
    ///
    /// `spatial_features`: [N, feature_dim] — per-point triplane features.
    /// `audio_latent`: [audio_dim] — audio embedding.
    /// Returns: [N, feature_dim] — conditioned features.
    pub fn forward(&self, spatial_features: &[f32], audio_latent: &[f32]) -> Vec<f32> {
        let n = spatial_features.len() / self.feature_dim;
        let mut output = vec![0.0f32; n * self.feature_dim];

        // Compute K and V from audio (shared across all spatial positions)
        let k = self.matmul_vec(&self.w_k, audio_latent, self.audio_dim, self.head_dim);
        let v = self.matmul_vec(&self.w_v, audio_latent, self.audio_dim, self.feature_dim);

        let scale = 1.0 / (self.head_dim as f32).sqrt();

        for i in 0..n {
            let feat = &spatial_features[i * self.feature_dim..(i + 1) * self.feature_dim];

            // Q from spatial feature
            let q = self.matmul_vec(&self.w_q, feat, self.feature_dim, self.head_dim);

            // Attention score (scalar, since single key)
            let score: f32 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum::<f32>() * scale;
            let attn = sigmoid(score); // use sigmoid instead of softmax (single key)

            // Output = attn * V + residual
            for j in 0..self.feature_dim {
                output[i * self.feature_dim + j] = feat[j] + attn * v[j];
            }
        }

        output
    }

    fn matmul_vec(&self, w: &[f32], x: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; out_dim];
        for j in 0..out_dim {
            for k in 0..in_dim {
                out[j] += w[j * in_dim + k] * x[k];
            }
        }
        out
    }

    pub fn num_params(&self) -> usize {
        self.w_q.len() + self.w_k.len() + self.w_v.len()
    }

    pub fn params_flat(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.num_params());
        out.extend_from_slice(&self.w_q);
        out.extend_from_slice(&self.w_k);
        out.extend_from_slice(&self.w_v);
        out
    }

    pub fn set_params_flat(&mut self, params: &[f32]) {
        let mut offset = 0;
        let qlen = self.feature_dim * self.head_dim;
        self.w_q.copy_from_slice(&params[offset..offset + qlen]);
        offset += qlen;
        let klen = self.audio_dim * self.head_dim;
        self.w_k.copy_from_slice(&params[offset..offset + klen]);
        offset += klen;
        let vlen = self.audio_dim * self.feature_dim;
        self.w_v.copy_from_slice(&params[offset..offset + vlen]);
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_attention() {
        let ca = CrossAttention::new(32, 64, 16);
        let spatial = vec![0.1f32; 5 * 32]; // 5 points, 32-dim
        let audio = vec![0.1f32; 64];
        let out = ca.forward(&spatial, &audio);
        assert_eq!(out.len(), 5 * 32);
    }
}
