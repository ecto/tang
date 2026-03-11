//! Face segmentation mask loading.
//!
//! Reads the binary format produced by `scripts/segment_face.py`.

use std::path::Path;

/// Segmentation class labels.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegClass {
    Background = 0,
    FaceSkin = 1,
    Hair = 2,
    Torso = 3,
}

impl From<u8> for SegClass {
    fn from(v: u8) -> Self {
        match v {
            0 => SegClass::Background,
            1 => SegClass::FaceSkin,
            2 => SegClass::Hair,
            3 => SegClass::Torso,
            _ => SegClass::Background,
        }
    }
}

/// Per-frame segmentation mask.
pub struct FrameMask {
    pub width: u32,
    pub height: u32,
    /// Class label per pixel [H*W], row-major.
    pub labels: Vec<u8>,
}

impl FrameMask {
    /// Get the foreground alpha mask (1.0 for face/hair, 0.0 for background).
    pub fn foreground_alpha(&self) -> Vec<f32> {
        self.labels
            .iter()
            .map(|&l| if l > 0 { 1.0 } else { 0.0 })
            .collect()
    }

    /// Get a face-only mask (1.0 for face skin, 0.0 elsewhere).
    pub fn face_only_alpha(&self) -> Vec<f32> {
        self.labels
            .iter()
            .map(|&l| if l == SegClass::FaceSkin as u8 { 1.0 } else { 0.0 })
            .collect()
    }
}

/// Segmentation masks for all frames.
pub struct MaskSequence {
    pub width: u32,
    pub height: u32,
    pub frames: Vec<FrameMask>,
}

impl MaskSequence {
    /// Load from binary file produced by segment_face.py.
    pub fn load(path: &Path) -> Result<Self, std::io::Error> {
        let data = std::fs::read(path)?;
        if data.len() < 12 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "file too small",
            ));
        }

        let num_frames = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let width = u32::from_le_bytes(data[4..8].try_into().unwrap());
        let height = u32::from_le_bytes(data[8..12].try_into().unwrap());

        let pixels_per_frame = (width * height) as usize;
        let expected = 12 + num_frames * pixels_per_frame;
        if data.len() < expected {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("expected {} bytes, got {}", expected, data.len()),
            ));
        }

        let mut frames = Vec::with_capacity(num_frames);
        let mut offset = 12;

        for _ in 0..num_frames {
            let labels = data[offset..offset + pixels_per_frame].to_vec();
            offset += pixels_per_frame;
            frames.push(FrameMask {
                width,
                height,
                labels,
            });
        }

        Ok(Self {
            width,
            height,
            frames,
        })
    }
}
