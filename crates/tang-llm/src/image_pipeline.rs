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
use sha2::{Digest, Sha256};
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
    #[serde(default)]
    pub preview: bool,
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
            (64..=1024).contains(&w)
                && (64..=1024).contains(&h)
                && w.is_multiple_of(16)
                && h.is_multiple_of(16),
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
    pub model_hash: String,
}
pub struct Progress {
    pub step: usize,
    pub of: usize,
    pub image_index: usize,
    pub preview_png: Option<Vec<u8>>,
}

pub struct Pipeline<D: ComputeDevice> {
    pub encoder: Model<D>,
    dit: DiT<D::Buffer>,
    vae: Vae<D::Buffer>,
    tokenizer: Tokenizer,
    template: Template,
    shift: f32,
    model_hash: String,
    preview: Result<Option<crate::image_preview::Projection>, String>,
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
        let model_hash = fingerprint(root)?;
        let preview = (|| -> Result<Option<crate::image_preview::Projection>> {
            let path = root.join("preview.json");
            if !path.exists() {
                return Ok(None);
            }
            ensure!(
                path.metadata()?.len() <= 16 * 1024,
                "preview calibration exceeds limit"
            );
            let calibration: crate::image_preview::Calibration =
                serde_json::from_slice(&std::fs::read(path)?)?;
            calibration.validate(
                &vae.cfg,
                &crate::image_preview::vae_hash(&root.join("vae"))?,
            )?;
            Ok(Some(calibration.projection))
        })()
        .map_err(|e| format!("{e:#}"));
        Ok(Self {
            encoder,
            dit,
            vae,
            tokenizer,
            template,
            shift,
            model_hash,
            preview,
        })
    }
    pub fn generate(
        &self,
        request: &Request,
        on: &mut dyn FnMut(Progress) -> bool,
    ) -> Result<Vec<Generated>> {
        let (width, height) = request.dimensions()?;
        let preview = if request.preview {
            Some(
                self.preview
                    .as_ref()
                    .map_err(|e| anyhow::anyhow!("preview calibration: {e}"))?
                    .as_ref()
                    .context(
                        "preview calibration unavailable; run calibrate-image-preview explicitly",
                    )?,
            )
        } else {
            None
        };
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
                    on(Progress {
                        step: index * request.steps + step,
                        of: request.n * request.steps,
                        image_index: index,
                        preview_png: if step > 0 {
                            preview.map(|p| p.png(&latent, h, w)).transpose()?
                        } else {
                            None
                        }
                    }),
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
                ensure!(
                    latent.iter().all(|x| x.is_finite()),
                    "nonfinite latent at step {}",
                    step + 1
                );
            }
            ensure!(
                on(Progress {
                    step: (index + 1) * request.steps,
                    of: request.n * request.steps,
                    image_index: index,
                    preview_png: preview.map(|p| p.png(&latent, h, w)).transpose()?
                }),
                "generation cancelled"
            );
            let scaled: Vec<_> = latent
                .iter()
                .map(|x| x / self.vae.cfg.scaling_factor + self.vae.cfg.shift_factor)
                .collect();
            let decoded = if std::env::var("TANG_IMAGE_TRACE").as_deref() == Ok("1") {
                eprintln!(
                    "image latent: max abs {}",
                    scaled.iter().map(|x| x.abs()).fold(0f32, f32::max)
                );
                self.vae
                    .decode_trace(dev, &scaled, h, w, &mut |name, buffer| {
                        let values = dev.download(buffer);
                        eprintln!(
                            "image VAE {name}: {} nonfinite, max abs {}",
                            values.iter().filter(|x| !x.is_finite()).count(),
                            values.iter().map(|x| x.abs()).fold(0f32, f32::max)
                        );
                    })?
            } else {
                self.vae.decode(dev, &scaled, h, w)?
            };
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
                model_hash: self.model_hash.clone(),
            });
        }
        Ok(images)
    }
}
/// Hash actual checkpoint bytes once per resident load, not a claimed download revision.
/// Paths and per-file digests include weights, architecture, tokenizer and scheduler.
fn fingerprint(root: &Path) -> Result<String> {
    let mut paths = Vec::new();
    for sub in [
        "transformer",
        "text_encoder",
        "vae",
        "tokenizer",
        "scheduler",
    ] {
        for entry in std::fs::read_dir(root.join(sub))? {
            let path = entry?.path();
            if path.extension().is_some_and(|ext| {
                ["safetensors", "json", "txt", "model", "jinja"]
                    .iter()
                    .any(|s| ext == *s)
            }) {
                paths.push(path);
            }
        }
    }
    paths.sort();
    let mut digest = Sha256::new();
    digest.update(b"tang-z-image-checkpoint-v1\0");
    let mut buffer = vec![0u8; 1024 * 1024];
    for path in paths {
        let name = path
            .strip_prefix(root)?
            .to_string_lossy()
            .replace('\\', "/");
        let mut file = std::fs::File::open(&path)?;
        let mut file_digest = Sha256::new();
        loop {
            let count = file.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            file_digest.update(&buffer[..count]);
        }
        digest.update((name.len() as u64).to_le_bytes());
        digest.update(name.as_bytes());
        digest.update(file_digest.finalize());
    }
    Ok(format!("sha256:{:x}", digest.finalize()))
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
    // Follow the node's free/inactive-page policy. CUDA's VRAM probe is primary;
    // unified-memory devices must also fit currently available host RAM.
    if available == 0 {
        available = crate::node::host_memory()
            .map(|(_, free)| free)
            .context("free-memory estimate unavailable")?;
    } else if cfg!(target_os = "macos") {
        available = available.min(
            crate::node::host_memory()
                .map(|(_, free)| free)
                .context("free-memory estimate unavailable")?,
        );
    }
    ensure!(available>=required,"not enough free device memory: need {required} bytes including scratch, have {available}; no resident models were evicted");
    Ok(())
}
/// Seeded SplitMix64 + Box-Muller; explicitly independent of PyTorch's RNG stream.
pub(crate) struct Normal {
    state: u64,
    spare: Option<f32>,
}
impl Normal {
    pub(crate) fn new(seed: u64) -> Self {
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
    pub(crate) fn next(&mut self) -> f32 {
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
    fn checkpoint_hash_tracks_weight_and_configuration_bytes() {
        let root = std::env::temp_dir().join(format!(
            "tang-image-hash-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        for sub in [
            "transformer",
            "text_encoder",
            "vae",
            "tokenizer",
            "scheduler",
        ] {
            std::fs::create_dir_all(root.join(sub)).unwrap();
        }
        let weight = root.join("transformer/model.safetensors");
        std::fs::write(&weight, b"first weights").unwrap();
        let first = fingerprint(&root).unwrap();
        assert_eq!(first, fingerprint(&root).unwrap());
        std::fs::write(&weight, b"other weights").unwrap();
        let second = fingerprint(&root).unwrap();
        assert_ne!(first, second);
        std::fs::write(root.join("scheduler/config.json"), b"{\"shift\":3}").unwrap();
        assert_ne!(second, fingerprint(&root).unwrap());
        std::fs::remove_dir_all(root).unwrap();
    }
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
