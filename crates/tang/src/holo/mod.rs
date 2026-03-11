//! tang-holo — Talking head hologram pipeline.
//!
//! Preprocesses face video into training data, trains a deformable gaussian
//! splatting model, and provides runtime rendering for audio-driven animation.
//!
//! # Pipeline Overview
//!
//! 1. **Preprocess**: Extract frames, detect landmarks, track face, segment background
//! 2. **Train**: Fit canonical gaussians + deformation MLPs to the video
//! 3. **Runtime**: Drive the hologram with audio features from TTS

// Preprocessing
pub mod audio;
pub mod background;
pub mod frames;
pub mod landmarks;
pub mod preprocess;
pub mod segmentation;
pub mod tracking;

// Model architecture
pub mod audio_encoder;
pub mod canonical;
pub mod cross_attention;
pub mod deformation;
pub mod holo_format;
pub mod holo_model;
pub mod triplane;
