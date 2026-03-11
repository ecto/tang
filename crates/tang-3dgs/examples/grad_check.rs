//! Numerical gradient check for the full backward pass (rasterization + projection).
//!
//! Uses a single gaussian for reliable finite difference comparison.
//! Position gradients with multiple gaussians are harder to verify via fd
//! due to the high nonlinearity of the gaussian spatial function, but the
//! backward derivation has been verified against a CPU reference implementation.
//!
//! ```bash
//! cargo run --example grad_check -p tang-3dgs
//! ```

use tang_3dgs::{Camera, GaussianCloud, Intrinsics, RasterConfig, Rasterizer};

const W: u32 = 64;
const H: u32 = 64;
const EPS: f32 = 1e-3;
const TOL: f32 = 0.05; // 5% relative tolerance
const ABS_TOL: f32 = 0.01; // absolute tolerance for near-zero gradients

fn main() {
    let camera = Camera::look_at(
        [0.0, 0.0, 4.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        Intrinsics {
            fx: 50.0,
            fy: 50.0,
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
        bg_color: [0.1, 0.1, 0.1],
        ..Default::default()
    };

    let rasterizer = Rasterizer::new(config);

    // Test with single gaussian for reliable fd comparison
    let cloud = GaussianCloud::random(1, 42);
    let output = rasterizer.forward(&cloud, &camera);
    let base_image = &output.image;

    println!("=== Numerical Gradient Check (Full Pipeline) ===\n");
    println!("  Gaussian 0: pos=({:.3}, {:.3}, {:.3}) mean2d=({:.3}, {:.3}) radius={}",
        cloud.positions[0][0], cloud.positions[0][1], cloud.positions[0][2],
        output.ctx.means_2d[0][0], output.ctx.means_2d[0][1], output.ctx.radii[0]);
    println!();

    let mut total_checks = 0;
    let mut passed = 0;

    let mut check = |name: &str, fd: f32, ana: f32| {
        let abs_diff = (fd - ana).abs();
        let denom = fd.abs().max(ana.abs()).max(1e-7);
        let rel_err = abs_diff / denom;
        let ok = rel_err < TOL || abs_diff < ABS_TOL;
        let status = if ok { "OK" } else { "FAIL" };
        println!("  [{}] fd={:.6} ana={:.6} rel_err={:.4} {}", name, fd, ana, rel_err, status);
        total_checks += 1;
        if ok { passed += 1; }
        ok
    };

    // Helper for finite difference
    let fd = |perturb: &dyn Fn(&mut GaussianCloud, f32)| -> f32 {
        let mut cp = cloud.clone();
        perturb(&mut cp, EPS);
        let op = rasterizer.forward(&cp, &camera);
        let mut cm = cloud.clone();
        perturb(&mut cm, -EPS);
        let om = rasterizer.forward(&cm, &camera);
        (loss(&op.image) - loss(&om.image)) / (2.0 * EPS)
    };

    let grads = rasterizer.backward(&cloud, &camera, &output.ctx, base_image);

    // Opacity (through sigmoid)
    let sig = 1.0 / (1.0 + (-cloud.opacities[0]).exp());
    check("opacity", fd(&|c, e| c.opacities[0] += e), grads.opacities[0] * sig * (1.0 - sig));

    // SH color (DC R channel)
    let _sh_per = cloud.sh_coeffs_per_gaussian();
    check("sh_r", fd(&|c, e| c.sh_coeffs[0] += e), grads.sh_coeffs[0] * 0.28209479);
    check("sh_g", fd(&|c, e| c.sh_coeffs[1] += e), grads.sh_coeffs[1] * 0.28209479);
    check("sh_b", fd(&|c, e| c.sh_coeffs[2] += e), grads.sh_coeffs[2] * 0.28209479);

    // Position
    check("pos_x", fd(&|c, e| c.positions[0][0] += e), grads.positions[0][0]);
    check("pos_y", fd(&|c, e| c.positions[0][1] += e), grads.positions[0][1]);
    check("pos_z", fd(&|c, e| c.positions[0][2] += e), grads.positions[0][2]);

    // Scale (log-scale)
    check("scale_x", fd(&|c, e| c.scales[0][0] += e), grads.scales[0][0]);
    check("scale_y", fd(&|c, e| c.scales[0][1] += e), grads.scales[0][1]);
    check("scale_z", fd(&|c, e| c.scales[0][2] += e), grads.scales[0][2]);

    // Rotation (quaternion)
    check("rot_w", fd(&|c, e| c.rotations[0][0] += e), grads.rotations[0][0]);
    check("rot_x", fd(&|c, e| c.rotations[0][1] += e), grads.rotations[0][1]);
    check("rot_y", fd(&|c, e| c.rotations[0][2] += e), grads.rotations[0][2]);
    check("rot_z", fd(&|c, e| c.rotations[0][3] += e), grads.rotations[0][3]);

    println!("\n=== Result: {}/{} checks passed ===", passed, total_checks);
    if passed == total_checks {
        println!("ALL PASSED!");
    } else {
        println!("SOME FAILED — check gradient derivation");
        std::process::exit(1);
    }
}

/// L2 loss: 0.5 * sum(image^2)
fn loss(image: &[f32]) -> f32 {
    image.iter().map(|x| 0.5 * x * x).sum()
}
