//! Train random gaussians to reproduce a reference scene.
//!
//! Renders a "ground truth" image from a known gaussian cloud, then
//! optimizes a random cloud to match it using differentiable rasterization.
//!
//! ```bash
//! cargo run --example train_scene -p tang-3dgs --release
//! # outputs: train_step_*.png showing optimization progress
//! ```

use tang_3dgs::densify::{densify, DensifyConfig, DensifyStats};
use tang_3dgs::train::{l1_loss_grad, AdamParam, TrainConfig};
use tang_3dgs::{Camera, GaussianCloud, Intrinsics, RasterConfig, Rasterizer};

const W: u32 = 256;
const H: u32 = 256;

fn main() {
    let camera = Camera::look_at(
        [0.0, 0.0, 5.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        Intrinsics {
            fx: 250.0,
            fy: 250.0,
            cx: W as f32 / 2.0,
            cy: H as f32 / 2.0,
        },
        W,
        H,
        0.01,
        100.0,
    );

    let config = RasterConfig {
        width: W,
        height: H,
        bg_color: [0.0, 0.0, 0.0],
        ..Default::default()
    };

    let rasterizer = Rasterizer::new(config.clone());

    // Generate ground truth
    println!("generating ground truth from 200 gaussians...");
    let gt_cloud = GaussianCloud::random(200, 0);
    let gt_output = rasterizer.forward(&gt_cloud, &camera);
    let gt_image = gt_output.image.clone();
    save_png(&gt_image, W, H, "train_gt.png");

    // Initialize trainable cloud with same count, different positions
    let n_init = 500;
    let mut cloud = GaussianCloud::random(n_init, 0);
    // Scatter randomly in a box around the scene
    for i in 0..cloud.count {
        let t = i as f32 / cloud.count as f32;
        let angle = t * std::f32::consts::PI * 2.0 + 0.7;
        let radius = 1.0 + (i as f32 * 0.618).fract() * 2.0;
        cloud.positions[i] = [angle.cos() * radius, angle.sin() * radius, (t - 0.5) * 2.0];
        cloud.opacities[i] = 1.0; // sigmoid(1) ≈ 0.73
        cloud.scales[i] = [-3.5, -3.5, -3.5]; // small
        // Random SH colors
        cloud.sh_coeffs[i * 3] = (t * 7.0).sin() * 0.5;
        cloud.sh_coeffs[i * 3 + 1] = (t * 11.0).sin() * 0.5;
        cloud.sh_coeffs[i * 3 + 2] = (t * 13.0).sin() * 0.5;
    }

    let train_config = TrainConfig {
        lr_position: 5e-3,
        lr_scale: 5e-3,
        lr_rotation: 1e-3,
        lr_opacity: 5e-2,
        lr_sh: 5e-3,
        iterations: 2000,
        densify_from: 500,
        densify_until: 1500,
        densify_interval: 100,
        ..Default::default()
    };

    // Per-group optimizers
    let n = cloud.count;
    let mut opt_pos = AdamParam::new(n * 3, train_config.lr_position);
    let mut opt_scale = AdamParam::new(n * 3, train_config.lr_scale);
    let mut opt_rot = AdamParam::new(n * 4, train_config.lr_rotation);
    let mut opt_opacity = AdamParam::new(n, train_config.lr_opacity);
    let mut opt_sh = AdamParam::new(cloud.sh_coeffs.len(), train_config.lr_sh);

    let mut densify_stats = DensifyStats::new(n);
    let densify_config = DensifyConfig {
        grad_threshold: 0.0005,
        scale_threshold: 0.05,
        opacity_threshold: 0.01, // only prune very transparent
        ..Default::default()
    };

    println!(
        "training {} iterations, starting with {} gaussians...",
        train_config.iterations, n
    );

    for iter in 0..train_config.iterations {
        // Forward
        let output = rasterizer.forward(&cloud, &camera);

        // L1 loss (fast, no SSIM for this demo)
        let (loss, dl_dimage) = l1_loss_grad(&output.image, &gt_image);

        // Backward
        let grads = rasterizer.backward(&cloud, &camera, &output.ctx, &dl_dimage);

        // Accumulate densification stats
        let visible: Vec<bool> = output.ctx.radii.iter().map(|&r| r > 0).collect();
        densify_stats.accumulate(&grads.positions, &visible);

        // Adam steps
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

        // Densification
        if iter >= train_config.densify_from
            && iter < train_config.densify_until
            && iter % train_config.densify_interval == 0
        {
            let old_count = cloud.count;
            cloud = densify(&cloud, &densify_stats, &densify_config);
            let new_count = cloud.count;

            if new_count == 0 {
                println!("  WARNING: all gaussians pruned at iter {}!", iter);
                break;
            }

            // Grow optimizer states
            opt_pos.grow(new_count * 3);
            opt_scale.grow(new_count * 3);
            opt_rot.grow(new_count * 4);
            opt_opacity.grow(new_count);
            opt_sh.grow(cloud.sh_coeffs.len());

            // Reset densify stats
            densify_stats = DensifyStats::new(new_count);

            if new_count != old_count {
                println!("  iter {}: densified {} -> {} gaussians", iter, old_count, new_count);
            }
        }

        // Logging
        if iter % 100 == 0 || iter == train_config.iterations - 1 {
            println!(
                "  iter {:>4}: loss={:.6}  gaussians={}",
                iter, loss, cloud.count
            );
        }

        // Save progress images
        if iter % 500 == 0 || iter == train_config.iterations - 1 {
            save_png(&output.image, W, H, &format!("train_step_{:04}.png", iter));
        }
    }

    println!("\ndone! final cloud has {} gaussians", cloud.count);
    println!("saved train_gt.png and train_step_*.png");
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
