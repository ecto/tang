//! Audio feature loading.
//!
//! Reads the binary format produced by `scripts/extract_audio.py`.
//! Per-frame 80-dim log mel spectrogram features aligned to video FPS.

use std::path::Path;

/// Default mel spectrogram dimension.
pub const MEL_DIM: usize = 80;

/// Audio features for a video sequence.
pub struct AudioFeatures {
    /// Number of frames.
    pub num_frames: usize,
    /// Feature dimension per frame.
    pub feature_dim: usize,
    /// Audio sample rate used.
    pub sample_rate: u32,
    /// Features [num_frames, feature_dim], row-major.
    pub features: Vec<f32>,
}

impl AudioFeatures {
    /// Load from binary file produced by extract_audio.py.
    pub fn load(path: &Path) -> Result<Self, std::io::Error> {
        let data = std::fs::read(path)?;
        if data.len() < 12 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "file too small",
            ));
        }

        let num_frames = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let feature_dim = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
        let sample_rate = u32::from_le_bytes(data[8..12].try_into().unwrap());

        let expected = 12 + num_frames * feature_dim * 4;
        if data.len() < expected {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("expected {} bytes, got {}", expected, data.len()),
            ));
        }

        let features: Vec<f32> = data[12..]
            .chunks(4)
            .take(num_frames * feature_dim)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        Ok(Self {
            num_frames,
            feature_dim,
            sample_rate,
            features,
        })
    }

    /// Get features for a specific frame.
    pub fn frame(&self, idx: usize) -> &[f32] {
        let start = idx * self.feature_dim;
        &self.features[start..start + self.feature_dim]
    }

    /// Get a window of audio features centered on a frame.
    /// Returns [window_size, feature_dim] features, zero-padded at boundaries.
    pub fn window(&self, center_frame: usize, window_size: usize) -> Vec<f32> {
        let half = window_size / 2;
        let mut out = vec![0.0f32; window_size * self.feature_dim];

        for i in 0..window_size {
            let frame_idx = center_frame as i64 + i as i64 - half as i64;
            if frame_idx >= 0 && (frame_idx as usize) < self.num_frames {
                let src = self.frame(frame_idx as usize);
                let dst_start = i * self.feature_dim;
                out[dst_start..dst_start + self.feature_dim].copy_from_slice(src);
            }
        }

        out
    }
}
