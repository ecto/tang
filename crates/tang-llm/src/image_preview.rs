//! A bounded affine latent-to-RGB projection, fitted from the installed VAE.
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::io::Cursor;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Projection {
    pub factors: Vec<[f32; 3]>,
    pub bias: [f32; 3],
}
impl Projection {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (1..=64).contains(&self.factors.len()),
            "invalid preview channels"
        );
        ensure!(
            self.factors
                .iter()
                .flatten()
                .chain(&self.bias)
                .all(|x| x.is_finite() && x.abs() <= 100.),
            "invalid preview coefficients"
        );
        Ok(())
    }
    /// Input is channel-first diffusion latents; output is a low-resolution PNG.
    pub fn png(&self, latent: &[f32], h: usize, w: usize) -> Result<Vec<u8>> {
        self.validate()?;
        ensure!(
            (1..=128).contains(&h) && (1..=128).contains(&w),
            "invalid preview dimensions"
        );
        let pixels = h * w;
        ensure!(
            latent.len() == self.factors.len() * pixels && latent.iter().all(|x| x.is_finite()),
            "invalid preview latent"
        );
        let mut rgb = vec![0u8; pixels * 3];
        for i in 0..pixels {
            for c in 0..3 {
                let value = self.bias[c]
                    + self
                        .factors
                        .iter()
                        .enumerate()
                        .map(|(k, f)| latent[k * pixels + i] * f[c])
                        .sum::<f32>();
                rgb[i * 3 + c] = ((value * 0.5 + 0.5).clamp(0., 1.) * 255.).round() as u8;
            }
        }
        let image = image::RgbImage::from_raw(w as u32, h as u32, rgb).unwrap();
        let mut png = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(image).write_to(&mut png, image::ImageFormat::Png)?;
        ensure!(
            png.get_ref().len() <= 64 * 1024,
            "preview exceeds PNG bound"
        );
        Ok(png.into_inner())
    }
}

/// Sufficient statistics for a tiny f64 ridge regression (channels plus intercept).
pub struct Fit {
    channels: usize,
    rows: usize,
    xx: Vec<Vec<f64>>,
    xy: Vec<[f64; 3]>,
}
impl Fit {
    pub fn new(channels: usize) -> Result<Self> {
        ensure!((1..=64).contains(&channels), "invalid preview channels");
        Ok(Self {
            channels,
            rows: 0,
            xx: vec![vec![0.; channels + 1]; channels + 1],
            xy: vec![[0.; 3]; channels + 1],
        })
    }
    pub fn observe(&mut self, latent: &[f32], rgb: [f32; 3]) -> Result<()> {
        ensure!(
            latent.len() == self.channels && latent.iter().chain(&rgb).all(|x| x.is_finite()),
            "invalid calibration sample"
        );
        let x: Vec<_> = latent.iter().map(|x| *x as f64).chain([1.]).collect();
        for i in 0..x.len() {
            for j in 0..x.len() {
                self.xx[i][j] += x[i] * x[j];
            }
            for c in 0..3 {
                self.xy[i][c] += x[i] * rgb[c] as f64;
            }
        }
        self.rows += 1;
        Ok(())
    }
    pub fn finish(mut self, ridge: f64) -> Result<Projection> {
        ensure!(
            self.rows > self.channels && ridge.is_finite() && ridge > 0.,
            "invalid calibration fit"
        );
        let dim = self.channels + 1;
        for i in 0..self.channels {
            self.xx[i][i] += ridge * self.rows as f64;
        }
        // Pivoted elimination also handles rank-deficient channel observations.
        for k in 0..dim {
            let pivot = (k..dim)
                .max_by(|&i, &j| self.xx[i][k].abs().total_cmp(&self.xx[j][k].abs()))
                .unwrap();
            self.xx.swap(k, pivot);
            self.xy.swap(k, pivot);
            let denom = self.xx[k][k];
            ensure!(
                denom.is_finite() && denom.abs() > 1e-12,
                "singular calibration fit"
            );
            for j in k..dim {
                self.xx[k][j] /= denom;
            }
            for c in 0..3 {
                self.xy[k][c] /= denom;
            }
            for i in 0..dim {
                if i == k {
                    continue;
                }
                let factor = self.xx[i][k];
                for j in k..dim {
                    self.xx[i][j] -= factor * self.xx[k][j];
                }
                for c in 0..3 {
                    self.xy[i][c] -= factor * self.xy[k][c];
                }
            }
        }
        let bias = self.xy.pop().unwrap().map(|x| x as f32);
        let result = Projection {
            factors: self.xy.into_iter().map(|x| x.map(|v| v as f32)).collect(),
            bias,
        };
        result.validate()?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn calibration_rejects_mismatched_vae_and_unvalidated_projection() {
        let cfg = crate::vae::Config {
            latent_channels: 16,
            out_channels: 3,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            norm_num_groups: 32,
            act_fn: "silu".into(),
            up_block_types: vec!["UpDecoderBlock2D".into(); 4],
            use_post_quant_conv: false,
            mid_block_add_attention: true,
            scaling_factor: 0.3611,
            shift_factor: 0.1159,
        };
        let mut record = Calibration {
            version: 1,
            vae_hash: "sha256:expected".into(),
            scaling_factor: cfg.scaling_factor,
            shift_factor: cfg.shift_factor,
            projection: Projection {
                factors: vec![[0.1; 3]; 16],
                bias: [0.; 3],
            },
            training_seeds: vec![11],
            held_out_seeds: vec![71],
            amplitudes: vec![1.],
            held_out_mse: 0.01,
            constant_baseline_mse: 0.1,
        };
        assert!(record.validate(&cfg, "sha256:expected").is_ok());
        assert!(record.validate(&cfg, "sha256:other").is_err());
        record.scaling_factor = 1.;
        assert!(record.validate(&cfg, "sha256:expected").is_err());
        record.scaling_factor = cfg.scaling_factor;
        record.held_out_mse = record.constant_baseline_mse;
        assert!(record.validate(&cfg, "sha256:expected").is_err());
    }

    #[test]
    fn fit_predicts_unseen_samples_and_png_clamps_signed_rgb() {
        let mut fit = Fit::new(2).unwrap();
        for x in -3..=3 {
            for y in -3..=3 {
                let (x, y) = (x as f32, y as f32);
                fit.observe(
                    &[x, y],
                    [
                        0.2 * x - 0.1 * y + 0.3,
                        -0.4 * x + 0.3 * y - 0.2,
                        0.1 * x + 0.5 * y,
                    ],
                )
                .unwrap();
            }
        }
        let p = fit.finish(1e-8).unwrap();
        for (c, expected) in [0.5, -0.65, -0.175].iter().enumerate() {
            let actual = p.bias[c] + p.factors[0][c] * 0.75 + p.factors[1][c] * -0.5;
            assert!((actual - expected).abs() < 1e-6);
        }
        let p = Projection {
            factors: vec![[1., -1., 0.]],
            bias: [0.; 3],
        };
        let bytes = p.png(&[-2., 2.], 1, 2).unwrap();
        let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
        assert_eq!(decoded.get_pixel(0, 0).0, [0, 255, 128]);
        assert_eq!(decoded.get_pixel(1, 0).0, [255, 0, 128]);
        assert!(p.png(&[f32::NAN], 1, 1).is_err());
        assert!(p.png(&[], 129, 1).is_err());
        assert!(Fit::new(0).is_err());
        assert!(Fit::new(2).unwrap().finish(1e-3).is_err());
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Calibration {
    pub version: u32,
    pub vae_hash: String,
    pub scaling_factor: f32,
    pub shift_factor: f32,
    pub projection: Projection,
    pub training_seeds: Vec<u64>,
    pub held_out_seeds: Vec<u64>,
    pub amplitudes: Vec<f32>,
    pub held_out_mse: f64,
    pub constant_baseline_mse: f64,
}

pub fn vae_hash(root: &std::path::Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut files = vec![root.join("config.json")];
    for entry in std::fs::read_dir(root)? {
        let path = entry?.path();
        if path.extension().is_some_and(|s| s == "safetensors") {
            files.push(path);
        }
    }
    ensure!(files.len() > 1, "VAE checkpoint missing");
    files.sort();
    let mut hash = Sha256::new();
    hash.update(b"tang-vae-preview-v1\0");
    let mut buffer = vec![0u8; 1024 * 1024];
    for path in files {
        let name = path.file_name().unwrap().to_string_lossy();
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        let mut file = std::fs::File::open(path)?;
        let mut file_hash = Sha256::new();
        loop {
            let n = file.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            file_hash.update(&buffer[..n]);
        }
        hash.update(file_hash.finalize());
    }
    Ok(format!("sha256:{:x}", hash.finalize()))
}

impl Calibration {
    pub fn validate(&self, vae: &crate::vae::Config, fingerprint: &str) -> Result<()> {
        self.projection.validate()?;
        ensure!(
            self.version == 1 && self.vae_hash == fingerprint,
            "preview VAE fingerprint mismatch"
        );
        ensure!(
            self.projection.factors.len() == vae.latent_channels
                && self.scaling_factor == vae.scaling_factor
                && self.shift_factor == vae.shift_factor,
            "preview VAE normalization mismatch"
        );
        ensure!(
            self.held_out_mse.is_finite()
                && self.held_out_mse >= 0.
                && self.constant_baseline_mse.is_finite()
                && self.held_out_mse < self.constant_baseline_mse,
            "preview calibration does not improve held-out error"
        );
        ensure!(
            !self.training_seeds.is_empty()
                && !self.held_out_seeds.is_empty()
                && self
                    .held_out_seeds
                    .iter()
                    .all(|s| !self.training_seeds.contains(s)),
            "invalid calibration seed split"
        );
        ensure!(
            !self.amplitudes.is_empty() && self.amplitudes.iter().all(|v| v.is_finite() && *v > 0.),
            "invalid calibration amplitudes"
        );
        Ok(())
    }
}

/// Fit only against a VAE, with reproducible independent train/held-out samples.
pub fn calibrate<D: tang_compute::ComputeDevice>(
    dev: D,
    root: &std::path::Path,
) -> Result<Calibration> {
    let fingerprint = vae_hash(root)?;
    let vae = crate::vae::Vae::load(&dev, root)?;
    ensure!(
        vae.cfg.scaling_factor.is_finite()
            && vae.cfg.scaling_factor > 0.
            && vae.cfg.shift_factor.is_finite(),
        "invalid preview normalization"
    );
    ensure!(
        vae.cfg.latent_channels == 16 && vae.cfg.out_channels == 3,
        "preview calibration requires a 16-channel RGB VAE"
    );
    let (h, w) = (32usize, 32usize);
    let train = vec![11, 23, 37, 53];
    let held_out = vec![71, 89];
    let amplitudes = vec![0.25, 0.5, 1., 2.];
    let mut fit = Fit::new(16)?;
    let mut mean = [0f64; 3];
    let mut rows = 0;
    let samples = |seed, amplitude: f32| -> Result<(Vec<f32>, Vec<[f32; 3]>)> {
        let mut rng = crate::image_pipeline::Normal::new(seed);
        let latent: Vec<_> = (0..16 * h * w).map(|_| rng.next() * amplitude).collect();
        let input: Vec<_> = latent
            .iter()
            .map(|x| x / vae.cfg.scaling_factor + vae.cfg.shift_factor)
            .collect();
        let output = vae.decode(&dev, &input, h, w)?;
        ensure!(
            output.len() == 3 * h * w * 64 && output.iter().all(|x| x.is_finite()),
            "invalid calibration decode"
        );
        let mut rgb = vec![[0.; 3]; h * w];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let mut sum = 0.;
                    for dy in 0..8 {
                        for dx in 0..8 {
                            sum += output[c * h * w * 64 + (y * 8 + dy) * w * 8 + x * 8 + dx]
                                .clamp(-1., 1.);
                        }
                    }
                    rgb[y * w + x][c] = sum / 64.;
                }
            }
        }
        Ok((latent, rgb))
    };
    for &seed in &train {
        for &amplitude in &amplitudes {
            eprintln!("preview calibration: train seed {seed}, amplitude {amplitude}");
            let (latent, rgb) = samples(seed, amplitude)?;
            for i in 0..h * w {
                let x: Vec<_> = (0..16).map(|c| latent[c * h * w + i]).collect();
                fit.observe(&x, rgb[i])?;
                for c in 0..3 {
                    mean[c] += rgb[i][c] as f64;
                }
                rows += 1;
            }
        }
    }
    let projection = fit.finish(1e-4)?;
    for c in 0..3 {
        mean[c] /= rows as f64;
    }
    let (mut error, mut baseline, mut count) = (0f64, 0f64, 0usize);
    for &seed in &held_out {
        for &amplitude in &amplitudes {
            eprintln!("preview calibration: held-out seed {seed}, amplitude {amplitude}");
            let (latent, rgb) = samples(seed, amplitude)?;
            for i in 0..h * w {
                for c in 0..3 {
                    let predicted = projection.bias[c] as f64
                        + (0..16)
                            .map(|k| latent[k * h * w + i] as f64 * projection.factors[k][c] as f64)
                            .sum::<f64>();
                    error += (predicted.clamp(-1., 1.) - rgb[i][c] as f64).powi(2);
                    baseline += (mean[c] - rgb[i][c] as f64).powi(2);
                    count += 1;
                }
            }
        }
    }
    ensure!(
        vae_hash(root)? == fingerprint,
        "VAE changed during preview calibration"
    );
    let record = Calibration {
        version: 1,
        vae_hash: fingerprint,
        scaling_factor: vae.cfg.scaling_factor,
        shift_factor: vae.cfg.shift_factor,
        projection,
        training_seeds: train,
        held_out_seeds: held_out,
        amplitudes,
        held_out_mse: error / count as f64,
        constant_baseline_mse: baseline / count as f64,
    };
    record.validate(&vae.cfg, &record.vae_hash)?;
    Ok(record)
}
