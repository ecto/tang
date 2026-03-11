//! Face landmark data loading and utilities.
//!
//! Reads the binary format produced by `scripts/detect_landmarks.py`.

use std::path::Path;

/// 478 MediaPipe Face Mesh landmarks per frame.
pub const NUM_LANDMARKS: usize = 478;

/// Per-frame landmark data.
pub struct FrameLandmarks {
    /// Whether a face was detected in this frame.
    pub detected: bool,
    /// Normalized (x, y, z) landmarks. Zero if not detected.
    pub points: Vec<[f32; 3]>,
}

/// Landmark data for all frames.
pub struct LandmarkSequence {
    pub frames: Vec<FrameLandmarks>,
}

impl LandmarkSequence {
    /// Load from binary file produced by detect_landmarks.py.
    pub fn load(path: &Path) -> Result<Self, std::io::Error> {
        let data = std::fs::read(path)?;
        if data.len() < 8 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "file too small",
            ));
        }

        let num_frames = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let num_landmarks = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;

        let bytes_per_frame = 1 + num_landmarks * 3 * 4; // detected + landmarks
        let expected = 8 + num_frames * bytes_per_frame;
        if data.len() < expected {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("expected {} bytes, got {}", expected, data.len()),
            ));
        }

        let mut frames = Vec::with_capacity(num_frames);
        let mut offset = 8;

        for _ in 0..num_frames {
            let detected = data[offset] != 0;
            offset += 1;

            let mut points = Vec::with_capacity(num_landmarks);
            for _ in 0..num_landmarks {
                let x = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
                let y = f32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap());
                let z = f32::from_le_bytes(data[offset + 8..offset + 12].try_into().unwrap());
                points.push([x, y, z]);
                offset += 12;
            }

            frames.push(FrameLandmarks { detected, points });
        }

        Ok(Self { frames })
    }

    /// Number of frames with detected faces.
    pub fn detected_count(&self) -> usize {
        self.frames.iter().filter(|f| f.detected).count()
    }

    /// Get the face bounding box for a frame (min_x, min_y, max_x, max_y) in normalized coords.
    pub fn face_bbox(&self, frame_idx: usize) -> Option<[f32; 4]> {
        let frame = &self.frames[frame_idx];
        if !frame.detected {
            return None;
        }

        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;

        // Use face oval landmarks (indices 0-16 in MediaPipe face mesh)
        for pt in &frame.points[..NUM_LANDMARKS.min(frame.points.len())] {
            min_x = min_x.min(pt[0]);
            min_y = min_y.min(pt[1]);
            max_x = max_x.max(pt[0]);
            max_y = max_y.max(pt[1]);
        }

        Some([min_x, min_y, max_x, max_y])
    }
}
