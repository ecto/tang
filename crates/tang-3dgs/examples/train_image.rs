//! Train gaussians to reproduce an input image.
//!
//! ```bash
//! cargo run --example train_image -p tang-3dgs --release -- input.png [--iters 5000] [--num 5000]
//! # outputs: fitted_*.png showing optimization progress
//! ```

use tang_3dgs::train::{l1_loss_grad, AdamParam};
use tang_3dgs::{Camera, GaussianCloud, Intrinsics, RasterConfig, Rasterizer};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <input.png> [--iters N] [--num N]", args[0]);
        std::process::exit(1);
    }

    let input_path = &args[1];
    let mut iters = 5000u32;
    let mut num_gaussians = 10000usize;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--iters" => { iters = args[i + 1].parse().unwrap(); i += 2; }
            "--num" => { num_gaussians = args[i + 1].parse().unwrap(); i += 2; }
            _ => i += 1,
        }
    }

    // Load target image
    let img = image::open(input_path).expect("failed to open image").to_rgb8();
    let (w, h) = (img.width(), img.height());
    println!("loaded {}x{} image: {}", w, h, input_path);

    let gt_image: Vec<f32> = img
        .pixels()
        .flat_map(|p| p.0.map(|v| v as f32 / 255.0))
        .collect();

    // Estimate background color from image corners
    let bg = estimate_bg_color(&gt_image, w as usize, h as usize);
    println!("estimated bg color: [{:.2}, {:.2}, {:.2}]", bg[0], bg[1], bg[2]);

    let config = RasterConfig {
        width: w,
        height: h,
        bg_color: bg,
        ..Default::default()
    };
    let rasterizer = Rasterizer::new(config);

    // Camera setup: orthographic-like projection.
    // Place camera at z=depth looking at origin. Gaussians at z≈0.
    // fx chosen so world coords [-x_range, x_range] map to full image width.
    let depth = 5.0f32;
    let x_range = 2.0f32;
    let y_range = x_range * h as f32 / w as f32;
    let fx = depth * w as f32 / (2.0 * x_range);
    let fy = depth * h as f32 / (2.0 * y_range);

    let camera = Camera::look_at(
        [0.0, 0.0, depth],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        Intrinsics {
            fx,
            fy,
            cx: w as f32 / 2.0,
            cy: h as f32 / 2.0,
        },
        w,
        h,
        0.01,
        100.0,
    );

    let mut cloud = init_cloud(num_gaussians, w, h, x_range, y_range, depth, fx, &gt_image);

    println!(
        "training {} gaussians for {} iterations (camera: depth={}, fx={:.0})...",
        num_gaussians, iters, depth, fx
    );

    // Optimizers with gradient clipping to prevent NaN
    let n = cloud.count;
    let mut opt_pos = AdamParam::with_clip(n * 3, 5e-4, 10.0);
    let mut opt_scale = AdamParam::with_clip(n * 3, 1e-2, 10.0);
    let mut opt_rot = AdamParam::with_clip(n * 4, 1e-3, 10.0);
    let mut opt_opacity = AdamParam::with_clip(n, 5e-2, 10.0);
    let mut opt_sh = AdamParam::with_clip(cloud.sh_coeffs.len(), 1e-2, 10.0);

    // Debug: check initial rendering
    {
        let output = rasterizer.forward(&cloud, &camera);
        let npix = (w * h) as usize;
        let non_bg: usize = (0..npix)
            .filter(|&i| {
                let r = output.image[i * 3];
                let g = output.image[i * 3 + 1];
                let b = output.image[i * 3 + 2];
                (r - bg[0]).abs() > 0.01 || (g - bg[1]).abs() > 0.01 || (b - bg[2]).abs() > 0.01
            })
            .count();
        let max_val = output.image.iter().cloned().fold(0.0f32, f32::max);
        let min_val = output.image.iter().cloned().fold(f32::MAX, f32::min);
        let radii_nonzero = output.ctx.radii.iter().filter(|&&r| r > 0).count();
        let max_radius = output.ctx.radii.iter().cloned().max().unwrap_or(0);
        let total_contrib: u64 = output.ctx.n_contrib.iter().map(|&c| c as u64).sum();
        let max_contrib = output.ctx.n_contrib.iter().cloned().max().unwrap_or(0);
        let pixels_with_contrib = output.ctx.n_contrib.iter().filter(|&&c| c > 0).count();
        println!(
            "debug: non_bg_pixels={}/{} val_range=[{:.3},{:.3}] radii_nonzero={} max_radius={}",
            non_bg, npix, min_val, max_val, radii_nonzero, max_radius
        );
        println!(
            "debug: pixels_with_contrib={} total_contrib={} max_contrib={}",
            pixels_with_contrib, total_contrib, max_contrib
        );
        // Check tile ranges
        let nonempty_tiles = output.ctx.tile_ranges.iter().filter(|r| r[0] != r[1]).count();
        let total_tile_entries: u64 = output.ctx.tile_ranges.iter()
            .map(|r| (r[1] - r[0]) as u64).sum();
        println!(
            "debug: nonempty_tiles={}/{} total_tile_entries={}",
            nonempty_tiles, output.ctx.tile_ranges.len(), total_tile_entries
        );

        // Print a few gaussian projections
        for idx in [0, 100, 5000, 9999usize] {
            let p = cloud.positions[idx];
            // Project: camera space = world - cam_pos, then screen = fx*X/Z + cx
            let cz = depth - p[2]; // cam-space z (depth from camera)
            let sx = fx * p[0] / cz + w as f32 / 2.0;
            let sy = fy * (-p[1]) / cz + h as f32 / 2.0; // flip y
            println!(
                "  gauss[{}]: world=({:.3},{:.3},{:.3}) → screen=({:.1},{:.1}) radius={}",
                idx, p[0], p[1], p[2], sx, sy, output.ctx.radii[idx]
            );
        }
    }

    for iter in 0..iters {
        let output = rasterizer.forward(&cloud, &camera);
        let (loss, dl_dimage) = l1_loss_grad(&output.image, &gt_image);

        if loss.is_nan() {
            println!("  NaN loss at iter {}, stopping", iter);
            break;
        }

        let grads = rasterizer.backward(&cloud, &camera, &output.ctx, &dl_dimage);

        let pos_flat = bytemuck::cast_slice_mut::<[f32; 3], f32>(&mut cloud.positions);
        let grad_pos_flat = bytemuck::cast_slice::<[f32; 3], f32>(&grads.positions);
        opt_pos.step(pos_flat, grad_pos_flat);

        let scale_flat = bytemuck::cast_slice_mut::<[f32; 3], f32>(&mut cloud.scales);
        let grad_scale_flat = bytemuck::cast_slice::<[f32; 3], f32>(&grads.scales);
        opt_scale.step(scale_flat, grad_scale_flat);

        let rot_flat = bytemuck::cast_slice_mut::<[f32; 4], f32>(&mut cloud.rotations);
        let grad_rot_flat = bytemuck::cast_slice::<[f32; 4], f32>(&grads.rotations);
        opt_rot.step(rot_flat, grad_rot_flat);

        opt_opacity.step(&mut cloud.opacities, &grads.opacities);
        opt_sh.step(&mut cloud.sh_coeffs, &grads.sh_coeffs);

        // Renormalize quaternions to unit length
        for q in &mut cloud.rotations {
            let len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
            if len > 1e-8 {
                for v in q.iter_mut() {
                    *v /= len;
                }
            }
        }

        // Clamp to prevent numerical blow-up
        for p in &mut cloud.positions {
            p[0] = p[0].clamp(-x_range * 1.5, x_range * 1.5);
            p[1] = p[1].clamp(-y_range * 1.5, y_range * 1.5);
            p[2] = p[2].clamp(-1.0, 1.0);
        }
        for s in &mut cloud.scales {
            for v in s.iter_mut() {
                *v = v.clamp(-7.0, 1.0); // exp(-7)≈0.001, exp(1)≈2.7
            }
        }
        for o in &mut cloud.opacities {
            *o = o.clamp(-5.0, 5.0);
        }

        if iter % 200 == 0 || iter == iters - 1 {
            let visible = output.ctx.radii.iter().filter(|&&r| r > 0).count();
            println!(
                "  iter {:>5}: loss={:.6}  visible={}/{}",
                iter, loss, visible, n
            );
        }

        if iter % 500 == 0 || iter == iters - 1 {
            save_png(&output.image, w, h, &format!("fitted_{:04}.png", iter));
        }
    }

    let output = rasterizer.forward(&cloud, &camera);
    save_png(&output.image, w, h, "fitted_final.png");
    save_png(&gt_image, w, h, "fitted_target.png");

    // Save trained cloud for loading by other tools
    let cloud_path = std::path::Path::new("trained_cloud.bin");
    cloud.save(cloud_path).expect("failed to save cloud");
    println!("\ndone! saved fitted_final.png, fitted_target.png, trained_cloud.bin");
}

/// Estimate background color from image corners.
fn estimate_bg_color(image: &[f32], w: usize, h: usize) -> [f32; 3] {
    let mut r = 0.0f32;
    let mut g = 0.0f32;
    let mut b = 0.0f32;
    let mut count = 0.0f32;

    let margin = 8.min(w / 4).min(h / 4);
    for y in 0..h {
        for x in 0..w {
            if (x < margin || x >= w - margin) && (y < margin || y >= h - margin) {
                let idx = (y * w + x) * 3;
                r += image[idx];
                g += image[idx + 1];
                b += image[idx + 2];
                count += 1.0;
            }
        }
    }

    if count > 0.0 {
        [r / count, g / count, b / count]
    } else {
        [0.0, 0.0, 0.0]
    }
}

/// Initialize gaussians on a grid, colored by the target image.
fn init_cloud(
    n: usize,
    w: u32,
    h: u32,
    x_range: f32,
    y_range: f32,
    _depth: f32,
    _fx: f32,
    gt_image: &[f32],
) -> GaussianCloud {
    let mut positions = Vec::with_capacity(n);
    let mut scales = Vec::with_capacity(n);
    let mut rotations = Vec::with_capacity(n);
    let mut opacities = Vec::with_capacity(n);
    let mut sh_coeffs = Vec::with_capacity(n * 3);

    let side = (n as f32).sqrt().ceil() as usize;

    // Each gaussian should fill its grid cell.
    // Grid spacing in pixels = image_width / side ≈ 2.5px for 10K gaussians.
    // Gaussian sigma ≈ spacing/2 so FWHM matches the grid cell.
    let spacing_px = w as f32 / side as f32;
    let sigma_px = spacing_px * 0.7; // slight overlap with neighbors
    let world_sigma = sigma_px * _depth / _fx;
    let log_scale = world_sigma.ln();

    for i in 0..n {
        let ix = i % side;
        let iy = i / side;

        let u = (ix as f32 + 0.5) / side as f32;
        let v = (iy as f32 + 0.5) / side as f32;

        let x = (u * 2.0 - 1.0) * x_range;
        let y = -((v * 2.0 - 1.0) * y_range);
        // Assign z by row index for stable depth ordering within tiles
        let z = (iy as f32 / side as f32 - 0.5) * 0.1;
        positions.push([x, y, z]);

        scales.push([log_scale, log_scale, log_scale - 5.0]); // very flat in z
        rotations.push([1.0, 0.0, 0.0, 0.0]);
        opacities.push(3.0); // sigmoid(3) ≈ 0.95, each gaussian fills its cell

        // Sample color from target image
        let px = (u * w as f32) as usize;
        let py = (v * h as f32) as usize;
        let px = px.min(w as usize - 1);
        let py = py.min(h as usize - 1);
        let idx = (py * w as usize + px) * 3;

        // SH convention: color = sh * C0 + 0.5, so sh = (color - 0.5) / C0
        let sh_c0 = 0.28209479;
        sh_coeffs.push((gt_image[idx] - 0.5) / sh_c0);
        sh_coeffs.push((gt_image[idx + 1] - 0.5) / sh_c0);
        sh_coeffs.push((gt_image[idx + 2] - 0.5) / sh_c0);
    }

    GaussianCloud {
        count: n,
        positions,
        scales,
        rotations,
        opacities,
        sh_coeffs,
        sh_degree: 0,
    }
}

fn save_png(image: &[f32], w: u32, h: u32, path: &str) {
    let mut data = vec![0u8; (w * h * 3) as usize];
    for i in 0..(w * h) as usize {
        data[i * 3] = (image[i * 3].clamp(0.0, 1.0) * 255.0) as u8;
        data[i * 3 + 1] = (image[i * 3 + 1].clamp(0.0, 1.0) * 255.0) as u8;
        data[i * 3 + 2] = (image[i * 3 + 2].clamp(0.0, 1.0) * 255.0) as u8;
    }
    let img = image::RgbImage::from_raw(w, h, data).unwrap();
    img.save(path).unwrap();
}
