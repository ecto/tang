//! Background removal using segmentation masks.
//!
//! Applies foreground masks to extract clean face images with
//! transparent or solid-color backgrounds.

use std::path::Path;

/// Apply a foreground mask to an RGB image, producing an RGBA image.
///
/// `image`: RGB float [H*W*3], range [0,1]
/// `mask`: alpha values [H*W], range [0,1]
/// Returns: RGBA float [H*W*4]
pub fn apply_mask(image: &[f32], mask: &[f32], bg_color: [f32; 3]) -> Vec<f32> {
    let n_pixels = mask.len();
    let mut out = Vec::with_capacity(n_pixels * 4);

    for i in 0..n_pixels {
        let alpha = mask[i];
        let r = image[i * 3] * alpha + bg_color[0] * (1.0 - alpha);
        let g = image[i * 3 + 1] * alpha + bg_color[1] * (1.0 - alpha);
        let b = image[i * 3 + 2] * alpha + bg_color[2] * (1.0 - alpha);
        out.push(r);
        out.push(g);
        out.push(b);
        out.push(alpha);
    }

    out
}

/// Extract masked face images from a frame directory + mask directory.
///
/// Reads frame_NNNNNN.png and mask_NNNNNN.png pairs, outputs
/// clean face images with background removed.
pub fn extract_foreground_frames(
    frames_dir: &Path,
    masks_dir: &Path,
    output_dir: &Path,
    bg_color: [f32; 3],
) -> Result<usize, std::io::Error> {
    std::fs::create_dir_all(output_dir)?;

    let mut frame_paths: Vec<_> = std::fs::read_dir(frames_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("frame_") && n.ends_with(".png"))
                .unwrap_or(false)
        })
        .collect();
    frame_paths.sort();

    let mut count = 0;

    for frame_path in &frame_paths {
        let name = frame_path.file_stem().unwrap().to_str().unwrap();
        let idx_str = &name["frame_".len()..];
        let mask_path = masks_dir.join(format!("mask_{}.png", idx_str));

        if !mask_path.exists() {
            continue;
        }

        let frame_img = image::open(frame_path)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?
            .to_rgb8();
        let mask_img = image::open(&mask_path)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?
            .to_luma8();

        let (w, h) = (frame_img.width(), frame_img.height());

        // Convert to float
        let rgb: Vec<f32> = frame_img.pixels().flat_map(|p| p.0.map(|v| v as f32 / 255.0)).collect();
        let mask: Vec<f32> = mask_img.pixels().map(|p| if p.0[0] > 0 { 1.0 } else { 0.0 }).collect();

        let rgba = apply_mask(&rgb, &mask, bg_color);

        // Save as RGBA PNG
        let mut out_data = vec![0u8; (w * h * 4) as usize];
        for i in 0..(w * h) as usize {
            out_data[i * 4] = (rgba[i * 4].clamp(0.0, 1.0) * 255.0) as u8;
            out_data[i * 4 + 1] = (rgba[i * 4 + 1].clamp(0.0, 1.0) * 255.0) as u8;
            out_data[i * 4 + 2] = (rgba[i * 4 + 2].clamp(0.0, 1.0) * 255.0) as u8;
            out_data[i * 4 + 3] = (rgba[i * 4 + 3].clamp(0.0, 1.0) * 255.0) as u8;
        }

        let out_img = image::RgbaImage::from_raw(w, h, out_data)
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::Other, "bad image dims"))?;
        let out_path = output_dir.join(format!("face_{}.png", idx_str));
        out_img
            .save(&out_path)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        count += 1;
    }

    Ok(count)
}
