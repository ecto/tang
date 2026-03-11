//! Full preprocessing pipeline — orchestrates all steps from video to training data.
//!
//! Runs: frame extraction → landmark detection → segmentation → tracking → audio → background removal

use super::audio::AudioFeatures;
use super::frames::{self, ExtractedFrames, FrameExtractConfig};
use super::landmarks::LandmarkSequence;
use super::segmentation::MaskSequence;
use super::tracking::TrackingSequence;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Preprocessed training data for a single video.
pub struct PreprocessedData {
    /// Extracted frames info.
    pub frames: ExtractedFrames,
    /// Face landmarks per frame.
    pub landmarks: LandmarkSequence,
    /// Segmentation masks per frame.
    pub masks: MaskSequence,
    /// Face tracking (pose + expression) per frame.
    pub tracking: TrackingSequence,
    /// Audio features per frame.
    pub audio: AudioFeatures,
    /// Output directory containing all preprocessed data.
    pub output_dir: PathBuf,
}

/// Configuration for the preprocessing pipeline.
pub struct PreprocessConfig {
    /// Target FPS for frame extraction.
    pub fps: f32,
    /// Target frame width (None = original).
    pub width: Option<u32>,
    /// Target frame height (None = original).
    pub height: Option<u32>,
    /// Background color for masked images.
    pub bg_color: [f32; 3],
    /// Python executable to use.
    pub python: String,
}

impl Default for PreprocessConfig {
    fn default() -> Self {
        Self {
            fps: 25.0,
            width: Some(512),
            height: Some(512),
            bg_color: [0.0, 0.0, 0.0],
            python: "python3".to_string(),
        }
    }
}

/// Run the full preprocessing pipeline.
pub fn preprocess(
    video_path: &Path,
    output_dir: &Path,
    config: &PreprocessConfig,
) -> Result<PreprocessedData, PreprocessError> {
    let scripts_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts");

    std::fs::create_dir_all(output_dir)?;

    // Step 1: Extract frames
    println!("[1/6] Extracting frames...");
    let frames_dir = output_dir.join("frames");
    let frame_config = FrameExtractConfig {
        fps: config.fps,
        width: config.width,
        height: config.height,
    };
    let extracted = frames::extract_frames(video_path, &frames_dir, &frame_config)
        .map_err(|e| PreprocessError::Step("frame extraction", e.to_string()))?;
    println!("       {} frames extracted ({}x{})", extracted.count, extracted.dimensions.0, extracted.dimensions.1);

    // Step 2: Detect landmarks
    println!("[2/6] Detecting face landmarks...");
    let landmarks_path = output_dir.join("landmarks.bin");
    run_python(
        &config.python,
        &scripts_dir.join("detect_landmarks.py"),
        &[frames_dir.to_str().unwrap(), landmarks_path.to_str().unwrap()],
    )?;
    let landmarks = LandmarkSequence::load(&landmarks_path)?;
    println!("       {}/{} frames with faces", landmarks.detected_count(), extracted.count);

    // Step 3: Segment faces
    println!("[3/6] Segmenting faces...");
    let masks_dir = output_dir.join("masks");
    run_python(
        &config.python,
        &scripts_dir.join("segment_face.py"),
        &[frames_dir.to_str().unwrap(), masks_dir.to_str().unwrap()],
    )?;
    let segments_path = masks_dir.join("segments.bin");
    let masks = MaskSequence::load(&segments_path)?;
    println!("       {} mask frames loaded", masks.frames.len());

    // Step 4: Track face pose + expression
    println!("[4/6] Tracking face pose and expression...");
    let tracking_path = output_dir.join("tracking.bin");
    run_python(
        &config.python,
        &scripts_dir.join("track_face.py"),
        &[landmarks_path.to_str().unwrap(), tracking_path.to_str().unwrap()],
    )?;
    let tracking = TrackingSequence::load(&tracking_path)?;
    println!("       {}/{} frames tracked", tracking.valid_count(), extracted.count);

    // Step 5: Extract audio features
    println!("[5/6] Extracting audio features...");
    let audio_path = output_dir.join("audio.bin");
    run_python(
        &config.python,
        &scripts_dir.join("extract_audio.py"),
        &[
            video_path.to_str().unwrap(),
            &config.fps.to_string(),
            audio_path.to_str().unwrap(),
        ],
    )?;
    let audio = AudioFeatures::load(&audio_path)?;
    println!("       {} audio frames ({}D features)", audio.num_frames, audio.feature_dim);

    // Step 6: Background removal
    println!("[6/6] Removing backgrounds...");
    let face_dir = output_dir.join("faces");
    let count = super::background::extract_foreground_frames(
        &frames_dir,
        &masks_dir,
        &face_dir,
        config.bg_color,
    )?;
    println!("       {} foreground face images", count);

    println!("\nPreprocessing complete! Output in {}", output_dir.display());

    Ok(PreprocessedData {
        frames: extracted,
        landmarks,
        masks,
        tracking,
        audio,
        output_dir: output_dir.to_path_buf(),
    })
}

fn run_python(python: &str, script: &Path, args: &[&str]) -> Result<(), PreprocessError> {
    let output = Command::new(python)
        .arg(script)
        .args(args)
        .output()
        .map_err(|e| {
            PreprocessError::Step(
                "python",
                format!("failed to run {}: {}", script.display(), e),
            )
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(PreprocessError::Step(
            "python script",
            format!(
                "{} failed:\nstdout: {}\nstderr: {}",
                script.display(),
                stdout,
                stderr
            ),
        ));
    }

    // Print stdout from the script
    let stdout = String::from_utf8_lossy(&output.stdout);
    if !stdout.is_empty() {
        for line in stdout.lines() {
            println!("       {}", line);
        }
    }

    Ok(())
}

#[derive(Debug)]
pub enum PreprocessError {
    Io(std::io::Error),
    Step(&'static str, String),
}

impl From<std::io::Error> for PreprocessError {
    fn from(e: std::io::Error) -> Self {
        PreprocessError::Io(e)
    }
}

impl std::fmt::Display for PreprocessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PreprocessError::Io(e) => write!(f, "IO error: {}", e),
            PreprocessError::Step(step, msg) => write!(f, "{} error: {}", step, msg),
        }
    }
}

impl std::error::Error for PreprocessError {}
