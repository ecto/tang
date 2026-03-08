//! Render a ring of colored gaussians to a PNG file.
//!
//! ```bash
//! cargo run --example render_test -p tang-3dgs
//! # outputs: test_render.png
//! ```

use tang_3dgs::{Camera, GaussianCloud, Intrinsics, RasterConfig, Rasterizer};

fn main() {
    let w = 512u32;
    let h = 512u32;

    println!("creating {} random gaussians...", 500);
    let cloud = GaussianCloud::random(500, 0);

    let camera = Camera::look_at(
        [0.0, 0.0, 5.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        Intrinsics {
            fx: 500.0,
            fy: 500.0,
            cx: w as f32 / 2.0,
            cy: h as f32 / 2.0,
        },
        w,
        h,
        0.01,
        100.0,
    );

    let config = RasterConfig {
        width: w,
        height: h,
        bg_color: [0.05, 0.05, 0.1],
        ..Default::default()
    };

    println!("compiling shaders + creating GPU device...");
    let rasterizer = Rasterizer::new(config);

    println!("rendering forward pass...");
    let output = rasterizer.forward(&cloud, &camera);

    // Save as PNG
    let mut img_data = vec![0u8; (w * h * 3) as usize];
    for i in 0..(w * h) as usize {
        img_data[i * 3] = (output.image[i * 3].clamp(0.0, 1.0) * 255.0) as u8;
        img_data[i * 3 + 1] = (output.image[i * 3 + 1].clamp(0.0, 1.0) * 255.0) as u8;
        img_data[i * 3 + 2] = (output.image[i * 3 + 2].clamp(0.0, 1.0) * 255.0) as u8;
    }

    let img = image::RgbImage::from_raw(w, h, img_data).unwrap();
    img.save("test_render.png").unwrap();

    let visible = output.ctx.radii.iter().filter(|&&r| r > 0).count();
    println!("done! {} visible gaussians, saved to test_render.png", visible);
    println!(
        "avg transmittance: {:.3}",
        output.ctx.final_transmittance.iter().sum::<f32>() / output.ctx.final_transmittance.len() as f32
    );
}
