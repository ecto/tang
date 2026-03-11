//! Video frame extraction via ffmpeg subprocess.
//!
//! Extracts frames at a target FPS from a video file, saving them as
//! numbered PNG files in an output directory.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Configuration for frame extraction.
pub struct FrameExtractConfig {
    /// Target frames per second (default: 25).
    pub fps: f32,
    /// Output image width (None = original).
    pub width: Option<u32>,
    /// Output image height (None = original).
    pub height: Option<u32>,
}

impl Default for FrameExtractConfig {
    fn default() -> Self {
        Self {
            fps: 25.0,
            width: None,
            height: None,
        }
    }
}

/// Result of frame extraction.
pub struct ExtractedFrames {
    /// Directory containing extracted frames.
    pub dir: PathBuf,
    /// Number of frames extracted.
    pub count: usize,
    /// Frame file paths in order.
    pub paths: Vec<PathBuf>,
    /// Actual FPS used.
    pub fps: f32,
    /// Frame dimensions (width, height).
    pub dimensions: (u32, u32),
}

/// Extract frames from a video file using ffmpeg.
///
/// Frames are saved as `frame_000000.png`, `frame_000001.png`, etc.
/// in the specified output directory.
pub fn extract_frames(
    video_path: &Path,
    output_dir: &Path,
    config: &FrameExtractConfig,
) -> Result<ExtractedFrames, FrameError> {
    // Verify video exists
    if !video_path.exists() {
        return Err(FrameError::VideoNotFound(video_path.to_path_buf()));
    }

    // Create output directory
    std::fs::create_dir_all(output_dir)?;

    // Build ffmpeg command
    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-y") // overwrite
        .arg("-i")
        .arg(video_path)
        .arg("-vf");

    // Build filter string
    let mut filters = vec![format!("fps={}", config.fps)];
    if let (Some(w), Some(h)) = (config.width, config.height) {
        filters.push(format!("scale={}:{}", w, h));
    } else if let Some(w) = config.width {
        filters.push(format!("scale={}:-2", w));
    } else if let Some(h) = config.height {
        filters.push(format!("scale=-2:{}", h));
    }
    cmd.arg(filters.join(","));

    let pattern = output_dir.join("frame_%06d.png");
    cmd.arg(pattern.to_str().unwrap());

    // Run ffmpeg
    let output = cmd.output().map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            FrameError::FfmpegNotFound
        } else {
            FrameError::Io(e)
        }
    })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(FrameError::FfmpegFailed(stderr.to_string()));
    }

    // Collect extracted frame paths
    let mut paths: Vec<PathBuf> = std::fs::read_dir(output_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("frame_") && n.ends_with(".png"))
                .unwrap_or(false)
        })
        .collect();
    paths.sort();

    let count = paths.len();
    if count == 0 {
        return Err(FrameError::NoFrames);
    }

    // Read dimensions from first frame
    let first = image::open(&paths[0]).map_err(|e| FrameError::ImageError(e.to_string()))?;
    let dimensions = (first.width(), first.height());

    Ok(ExtractedFrames {
        dir: output_dir.to_path_buf(),
        count,
        paths,
        fps: config.fps,
        dimensions,
    })
}

/// Probe video metadata using ffprobe.
pub struct VideoInfo {
    pub width: u32,
    pub height: u32,
    pub fps: f32,
    pub duration_secs: f32,
    pub frame_count: usize,
}

/// Get video information using ffprobe.
pub fn probe_video(path: &Path) -> Result<VideoInfo, FrameError> {
    let output = Command::new("ffprobe")
        .args([
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-show_format",
        ])
        .arg(path)
        .output()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                FrameError::FfmpegNotFound
            } else {
                FrameError::Io(e)
            }
        })?;

    if !output.status.success() {
        return Err(FrameError::FfmpegFailed("ffprobe failed".into()));
    }

    let json: String = String::from_utf8_lossy(&output.stdout).to_string();
    parse_ffprobe_json(&json)
}

fn parse_ffprobe_json(json: &str) -> Result<VideoInfo, FrameError> {
    // Minimal JSON parsing without serde_json dependency.
    // We extract: width, height, r_frame_rate, duration, nb_frames from the video stream.
    let find_val = |key: &str| -> Option<String> {
        let pattern = format!("\"{}\"", key);
        let pos = json.find(&pattern)?;
        let after = &json[pos + pattern.len()..];
        // Skip : and whitespace, find value
        let colon = after.find(':')?;
        let rest = after[colon + 1..].trim_start();
        if rest.starts_with('"') {
            let end = rest[1..].find('"')?;
            Some(rest[1..1 + end].to_string())
        } else {
            let end = rest.find(|c: char| c == ',' || c == '}' || c == '\n')?;
            Some(rest[..end].trim().to_string())
        }
    };

    let width = find_val("width")
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0);
    let height = find_val("height")
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0);

    let fps = find_val("r_frame_rate")
        .and_then(|s| {
            let parts: Vec<&str> = s.split('/').collect();
            if parts.len() == 2 {
                let num: f32 = parts[0].parse().ok()?;
                let den: f32 = parts[1].parse().ok()?;
                Some(num / den)
            } else {
                s.parse().ok()
            }
        })
        .unwrap_or(25.0);

    let duration_secs = find_val("duration")
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.0);

    let frame_count = find_val("nb_frames")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or((duration_secs * fps) as usize);

    if width == 0 || height == 0 {
        return Err(FrameError::FfmpegFailed("could not determine video dimensions".into()));
    }

    Ok(VideoInfo {
        width,
        height,
        fps,
        duration_secs,
        frame_count,
    })
}

#[derive(Debug)]
pub enum FrameError {
    VideoNotFound(PathBuf),
    FfmpegNotFound,
    FfmpegFailed(String),
    NoFrames,
    ImageError(String),
    Io(std::io::Error),
}

impl From<std::io::Error> for FrameError {
    fn from(e: std::io::Error) -> Self {
        FrameError::Io(e)
    }
}

impl std::fmt::Display for FrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FrameError::VideoNotFound(p) => write!(f, "video not found: {}", p.display()),
            FrameError::FfmpegNotFound => write!(f, "ffmpeg not found in PATH"),
            FrameError::FfmpegFailed(s) => write!(f, "ffmpeg error: {}", s),
            FrameError::NoFrames => write!(f, "no frames extracted"),
            FrameError::ImageError(s) => write!(f, "image error: {}", s),
            FrameError::Io(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for FrameError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_ffprobe_json() {
        let json = r#"{
            "streams": [{
                "codec_type": "video",
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "30000/1001",
                "nb_frames": "900",
                "duration": "30.03"
            }]
        }"#;
        let info = parse_ffprobe_json(json).unwrap();
        assert_eq!(info.width, 1920);
        assert_eq!(info.height, 1080);
        assert!((info.fps - 29.97).abs() < 0.1);
        assert_eq!(info.frame_count, 900);
    }
}
