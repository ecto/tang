//! The complete HoloModel — ties together triplane, canonical predictor,
//! audio encoder, cross-attention, and deformation predictor.

use super::audio_encoder::AudioEncoder;
use super::canonical::CanonicalPredictor;
use super::cross_attention::CrossAttention;
use super::deformation::DeformationPredictor;
use super::triplane::Triplane;
use crate::splatting::GaussianCloud;

/// Configuration for the hologram model.
pub struct HoloConfig {
    /// Triplane resolution.
    pub triplane_res: u32,
    /// Triplane feature dimension.
    pub feature_dim: usize,
    /// Number of anchor points for canonical gaussians.
    pub num_anchors: usize,
    /// Audio mel dimension.
    pub mel_dim: usize,
    /// Audio window size (frames).
    pub audio_window: usize,
    /// Audio latent dimension.
    pub audio_latent_dim: usize,
    /// Cross-attention head dimension.
    pub attn_head_dim: usize,
}

impl Default for HoloConfig {
    fn default() -> Self {
        Self {
            triplane_res: 64,
            feature_dim: 32,
            num_anchors: 2000,
            mel_dim: 80,
            audio_window: 16,
            audio_latent_dim: 64,
            attn_head_dim: 16,
        }
    }
}

/// The complete hologram model.
pub struct HoloModel {
    pub config: HoloConfig,
    /// Spatial feature planes.
    pub triplane: Triplane,
    /// Canonical gaussian predictor.
    pub canonical: CanonicalPredictor,
    /// Audio feature encoder.
    pub audio_encoder: AudioEncoder,
    /// Cross-attention for audio-spatial fusion.
    pub cross_attention: CrossAttention,
    /// Deformation predictor.
    pub deformation: DeformationPredictor,
    /// Anchor points in normalized [-1,1] space.
    pub anchors: Vec<[f32; 3]>,
}

impl HoloModel {
    /// Create a new hologram model with given configuration.
    pub fn new(config: HoloConfig) -> Self {
        let triplane = Triplane::new(config.triplane_res, config.feature_dim);
        let canonical = CanonicalPredictor::new(config.feature_dim);
        let audio_encoder =
            AudioEncoder::new(config.mel_dim, config.audio_window, config.audio_latent_dim);
        let cross_attention =
            CrossAttention::new(config.feature_dim, config.audio_latent_dim, config.attn_head_dim);
        let deformation = DeformationPredictor::new(config.feature_dim);

        // Initialize anchors on a uniform grid in [-1, 1]
        let n = config.num_anchors;
        let side = (n as f32).cbrt().ceil() as usize;
        let mut anchors = Vec::with_capacity(n);
        for iz in 0..side {
            for iy in 0..side {
                for ix in 0..side {
                    if anchors.len() >= n {
                        break;
                    }
                    let x = (ix as f32 / (side - 1).max(1) as f32) * 2.0 - 1.0;
                    let y = (iy as f32 / (side - 1).max(1) as f32) * 2.0 - 1.0;
                    let z = (iz as f32 / (side - 1).max(1) as f32) * 2.0 - 1.0;
                    anchors.push([x, y, z]);
                }
            }
        }
        anchors.truncate(n);

        Self {
            config,
            triplane,
            canonical,
            audio_encoder,
            cross_attention,
            deformation,
            anchors,
        }
    }

    /// Forward pass: produce a deformed gaussian cloud for a given audio window.
    ///
    /// Stage 1 (canonical only): pass `None` for audio.
    /// Stage 2 (with audio): pass audio features.
    pub fn forward(&self, audio_window: Option<&[f32]>) -> GaussianCloud {
        // Query triplane at anchor points
        let features = self.triplane.query_batch(&self.anchors);

        // Predict canonical gaussians
        let canonical = self.canonical.predict(
            &features,
            self.config.feature_dim,
            &self.anchors,
        );

        if let Some(audio) = audio_window {
            // Encode audio
            let audio_latent = self.audio_encoder.encode(audio);

            // Cross-attention: condition spatial features on audio
            let conditioned = self.cross_attention.forward(&features, &audio_latent);

            // Predict deformations
            let deformations = self.deformation.predict(&conditioned, self.config.feature_dim);

            // Apply deformations to canonical cloud
            let canonical_cloud = canonical.to_cloud();
            deformations.apply(&canonical_cloud)
        } else {
            canonical.to_cloud()
        }
    }

    /// Total number of learnable parameters.
    pub fn num_params(&self) -> usize {
        self.triplane.num_params()
            + self.canonical.mlp.num_params()
            + self.audio_encoder.num_params()
            + self.cross_attention.num_params()
            + self.deformation.num_params()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_holo_model_forward_canonical() {
        let config = HoloConfig {
            num_anchors: 100,
            triplane_res: 8,
            ..Default::default()
        };
        let model = HoloModel::new(config);
        let cloud = model.forward(None);
        assert_eq!(cloud.count, 100);
        assert_eq!(cloud.sh_coeffs.len(), 300); // 100 * 3 DC
    }

    #[test]
    fn test_holo_model_forward_with_audio() {
        let config = HoloConfig {
            num_anchors: 50,
            triplane_res: 8,
            ..Default::default()
        };
        let model = HoloModel::new(config);
        let audio = vec![0.0f32; 80 * 16]; // 16 frames * 80 mel
        let cloud = model.forward(Some(&audio));
        assert_eq!(cloud.count, 50);
    }
}
