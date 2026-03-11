//! Audio encoder — converts mel spectrogram features to a latent code.
//!
//! Takes a window of audio features (e.g., 16 frames × 80 mel bins)
//! and produces a compact audio embedding used to drive deformations.

use super::canonical::Mlp;

/// Audio encoder: mel features → audio latent code.
pub struct AudioEncoder {
    /// Temporal convolution + MLP.
    pub mlp: Mlp,
    /// Input window size in frames.
    pub window_size: usize,
    /// Mel feature dimension.
    pub mel_dim: usize,
    /// Output latent dimension.
    pub latent_dim: usize,
}

impl AudioEncoder {
    /// Create an audio encoder.
    ///
    /// `mel_dim`: dimension of per-frame mel features (80).
    /// `window_size`: number of frames in the audio window (16).
    /// `latent_dim`: output embedding dimension (64).
    pub fn new(mel_dim: usize, window_size: usize, latent_dim: usize) -> Self {
        let input_dim = mel_dim * window_size;
        Self {
            mlp: Mlp::new(&[input_dim, 256, 128, latent_dim]),
            window_size,
            mel_dim,
            latent_dim,
        }
    }

    /// Encode a window of audio features.
    ///
    /// `window`: [window_size * mel_dim] flattened audio features.
    /// Returns: [latent_dim] audio embedding.
    pub fn encode(&self, window: &[f32]) -> Vec<f32> {
        assert_eq!(window.len(), self.window_size * self.mel_dim);
        self.mlp.forward(window)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_encoder() {
        let enc = AudioEncoder::new(80, 16, 64);
        let window = vec![0.0f32; 80 * 16];
        let latent = enc.encode(&window);
        assert_eq!(latent.len(), 64);
    }
}
