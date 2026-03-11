//! Face tracking data (pose + expression) loading.
//!
//! Reads the binary format produced by `scripts/track_face.py`.

use std::path::Path;

/// Number of FLAME expression coefficients.
pub const NUM_EXPRESSION: usize = 50;
/// Number of FLAME shape coefficients.
pub const NUM_SHAPE: usize = 100;

/// Per-frame tracking data.
pub struct FrameTracking {
    /// Whether tracking succeeded for this frame.
    pub valid: bool,
    /// Head rotation as axis-angle [3].
    pub rotation: [f32; 3],
    /// Head translation [3].
    pub translation: [f32; 3],
    /// Expression coefficients [50].
    pub expression: Vec<f32>,
}

/// Complete tracking sequence.
pub struct TrackingSequence {
    /// Shared shape coefficients [100].
    pub shape: Vec<f32>,
    /// Per-frame tracking.
    pub frames: Vec<FrameTracking>,
}

impl TrackingSequence {
    /// Load from binary file produced by track_face.py.
    pub fn load(path: &Path) -> Result<Self, std::io::Error> {
        let data = std::fs::read(path)?;
        if data.len() < 8 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "file too small",
            ));
        }

        let num_frames = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let num_expr = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;

        let mut offset = 8;

        // Read shape coefficients
        let shape: Vec<f32> = (0..NUM_SHAPE)
            .map(|i| {
                let o = offset + i * 4;
                f32::from_le_bytes(data[o..o + 4].try_into().unwrap())
            })
            .collect();
        offset += NUM_SHAPE * 4;

        let mut frames = Vec::with_capacity(num_frames);

        for _ in 0..num_frames {
            let valid = data[offset] != 0;
            offset += 1;

            let read_f32 = |o: usize| f32::from_le_bytes(data[o..o + 4].try_into().unwrap());

            let rotation = [
                read_f32(offset),
                read_f32(offset + 4),
                read_f32(offset + 8),
            ];
            offset += 12;

            let translation = [
                read_f32(offset),
                read_f32(offset + 4),
                read_f32(offset + 8),
            ];
            offset += 12;

            let expression: Vec<f32> = (0..num_expr).map(|i| read_f32(offset + i * 4)).collect();
            offset += num_expr * 4;

            frames.push(FrameTracking {
                valid,
                rotation,
                translation,
                expression,
            });
        }

        Ok(Self { shape, frames })
    }

    /// Number of valid tracking frames.
    pub fn valid_count(&self) -> usize {
        self.frames.iter().filter(|f| f.valid).count()
    }
}
