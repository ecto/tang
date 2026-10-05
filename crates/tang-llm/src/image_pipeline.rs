//! Resident Qwen text encoder + Z-Image DiT + VAE. No implicit model downloads.
use crate::{
    chat::Template,
    dit::DiT,
    model::{Dtype, Model},
    scheduler::Euler,
    vae::Vae,
};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    io::{Cursor, Read},
    path::Path,
};
use tang_compute::ComputeDevice;
use tokenizers::Tokenizer;
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub model: String,
    pub prompt: String,
    #[serde(default = "default_size")]
    pub size: String,
    #[serde(default = "default_count")]
    pub n: usize,
    pub seed: Option<u64>,
    #[serde(default = "default_steps")]
    pub steps: usize,
    #[serde(default)]
    pub stream: bool,
    pub response_format: Option<String>,
}
fn default_size() -> String {
    "512x512".into()
}
fn default_count() -> usize {
    1
}
fn default_steps() -> usize {
    8
}
impl Request {
    pub fn dimensions(&self) -> Result<(usize, usize)> {
        ensure!(self.model == "z-image-turbo", "unknown image model");
        ensure!(
            !self.prompt.trim().is_empty() && self.prompt.len() <= 32000,
            "invalid prompt"
        );
        ensure!(
            (1..=4).contains(&self.n) && (1..=100).contains(&self.steps),
            "invalid count or steps"
        );
        ensure!(
            self.response_format
                .as_deref()
                .is_none_or(|s| s == "b64_json"),
            "only b64_json response format supported"
        );
        let (w, h) = self
            .size
            .split_once('x')
            .context("size must be WIDTHxHEIGHT")?;
        let (w, h) = (w.parse::<usize>()?, h.parse::<usize>()?);
        ensure!(
            (64..=1024).contains(&w) && (64..=1024).contains(&h) && w % 16 == 0 && h % 16 == 0,
            "dimensions must be multiples of 16 between 64 and 1024"
        );
        Ok((w, h))
    }
}
#[derive(Debug, Serialize)]
pub struct Generated {
    #[serde(skip_serializing)]
    pub png: Vec<u8>,
    pub seed: u64,
    pub steps: usize,
    pub width: usize,
    pub height: usize,
}
pub struct Pipeline<D: ComputeDevice> {
    pub encoder: Model<D>,
    dit: DiT<D::Buffer>,
    vae: Vae<D::Buffer>,
    tokenizer: Tokenizer,
    template: Template,
    shift: f32,
}
impl<D: ComputeDevice> Pipeline<D> {
    pub fn load(dev: D, root: &Path) -> Result<Self> {
        check_memory(root, &dev)?;
        let encoder = Model::load(dev, &root.join("text_encoder"), 512, Dtype::Bf16)?;
        ensure!(
            encoder.cfg.model_type == "qwen3" && encoder.cfg.hidden_size == 2560,
            "Z-Image requires Qwen3-4B conditioning"
        );
        let tokenizer = Tokenizer::from_file(root.join("tokenizer/tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        let template = Template::load(&root.join("tokenizer"))?;
        let dit = DiT::load(&encoder.dev, &root.join("transformer"), Dtype::Bf16)?;
        let vae = Vae::load(&encoder.dev, &root.join("vae"))?;
        ensure!(
            dit.cfg.cap_feat_dim == encoder.cfg.hidden_size
                && dit.cfg.in_channels == vae.cfg.latent_channels,
            "incompatible text/DiT/VAE dimensions"
        );
        let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(
            root.join("scheduler/scheduler_config.json"),
        )?)?;
        ensure!(
            cfg["use_dynamic_shifting"] == false,
            "only static Euler shifting supported"
        );
        let shift = cfg["shift"].as_f64().context("scheduler shift missing")? as f32;
        Ok(Self {
            encoder,
            dit,
            vae,
            tokenizer,
            template,
            shift,
        })
    }
    pub fn generate(
        &self,
        request: &Request,
        on: &mut dyn FnMut(usize, usize) -> bool,
    ) -> Result<Vec<Generated>> {
        let (width, height) = request.dimensions()?;
        let dev = &self.encoder.dev;
        let prompt = self.template.render(
            &json!([{"role":"user","content":request.prompt}]),
            None,
            Some(true),
        )?;
        let encoded = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("encoding prompt: {e}"))?;
        let tokens = &encoded.get_ids()[..encoded.len().min(512)];
        let caption = self.encoder.encode_penultimate(tokens)?;
        let base_seed = match request.seed {
            Some(s) => s,
            None => {
                let mut bytes = [0; 8];
                std::fs::File::open("/dev/urandom")?.read_exact(&mut bytes)?;
                u64::from_le_bytes(bytes)
            }
        };
        let schedule = Euler::new(request.steps, self.shift)?;
        let (h, w) = (height / 8, width / 8);
        let mut images = Vec::new();
        for index in 0..request.n {
            let seed = base_seed.wrapping_add(index as u64);
            let mut random = Normal::new(seed);
            let mut latent: Vec<_> = (0..self.dit.cfg.in_channels * h * w)
                .map(|_| random.next())
                .collect();
            for step in 0..request.steps {
                ensure!(
                    on(index * request.steps + step, request.n * request.steps),
                    "generation cancelled"
                );
                let prediction = self.dit.forward(
                    dev,
                    &latent,
                    h,
                    w,
                    &caption,
                    tokens.len(),
                    schedule.normalized_time(step)?,
                )?;
                schedule.step(step, &prediction, &mut latent)?;
            }
            ensure!(
                on((index + 1) * request.steps, request.n * request.steps),
                "generation cancelled"
            );
            let scaled: Vec<_> = latent
                .iter()
                .map(|x| x / self.vae.cfg.scaling_factor + self.vae.cfg.shift_factor)
                .collect();
            let decoded = self.vae.decode(dev, &scaled, h, w)?;
            ensure!(
                self.vae.cfg.out_channels == 3 && decoded.len() == width * height * 3,
                "unexpected decoded image dimensions"
            );
            ensure!(
                decoded.iter().all(|x| x.is_finite()),
                "nonfinite decoder output"
            );
            let decoded = &decoded;
            let pixels: Vec<u8> = (0..width * height)
                .flat_map(|i| {
                    (0..3)
                        .map(move |c| {
                            (decoded[c * width * height + i] * 0.5 + 0.5).clamp(0., 1.) * 255.
                        })
                        .map(|x| x.round() as u8)
                })
                .collect();
            let image = image::RgbImage::from_raw(width as u32, height as u32, pixels)
                .context("PNG shape")?;
            let mut png = Cursor::new(Vec::new());
            image.write_to(&mut png, image::ImageFormat::Png)?;
            images.push(Generated {
                png: png.into_inner(),
                seed,
                steps: request.steps,
                width,
                height,
            });
        }
        Ok(images)
    }
}
fn check_memory<D: ComputeDevice>(root: &Path, dev: &D) -> Result<()> {
    // Matrices are resident in BF16; VAE is explicitly upcast. Retain scratch/headroom.
    let mut stored = 0u64;
    for sub in ["transformer", "text_encoder", "vae"] {
        for entry in std::fs::read_dir(root.join(sub))? {
            let p = entry?.path();
            if p.extension().is_some_and(|s| s == "safetensors") {
                stored += p.metadata()?.len();
            }
        }
    }
    let required = stored * dev.bf16_storage_bytes() as u64 / 2 + 4 * 1024 * 1024 * 1024;
    let mut available = dev.free_memory_bytes() as u64;
    #[cfg(target_os = "macos")]
    if available == 0 {
        let total = std::process::Command::new("/usr/sbin/sysctl")
            .args(["-n", "hw.memsize"])
            .output()?;
        let total = String::from_utf8(total.stdout)?.trim().parse::<u64>()?;
        let output = std::process::Command::new("/usr/bin/memory_pressure")
            .arg("-Q")
            .output()?;
        let text = String::from_utf8(output.stdout)?;
        let percent = text
            .lines()
            .find_map(|l| l.strip_prefix("System-wide memory free percentage: "))
            .and_then(|s| s.strip_suffix('%'))
            .context("free-memory estimate unavailable")?
            .parse::<u64>()?;
        available = total * percent / 100;
    }
    ensure!(available>=required,"not enough free device memory: need {required} bytes including scratch, have {available}; no resident models were evicted");
    Ok(())
}
/// Seeded SplitMix64 + Box-Muller; explicitly independent of PyTorch's RNG stream.
struct Normal {
    state: u64,
    spare: Option<f32>,
}
impl Normal {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare: None,
        }
    }
    fn uniform(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn next(&mut self) -> f32 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        let radius = (-2. * self.uniform().ln()).sqrt();
        let angle = std::f64::consts::TAU * self.uniform();
        self.spare = Some((radius * angle.sin()) as f32);
        (radius * angle.cos()) as f32
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gaussian_is_seeded_and_normalized() {
        let mut a = Normal::new(42);
        let mut b = Normal::new(42);
        let x: Vec<_> = (0..10000).map(|_| a.next()).collect();
        assert_eq!(x, (0..10000).map(|_| b.next()).collect::<Vec<_>>());
        let mean = x.iter().sum::<f32>() / x.len() as f32;
        let variance = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / x.len() as f32;
        assert!(mean.abs() < 0.04 && (variance - 1.).abs() < 0.05);
    }
}
