//! .holo file format for serialized hologram models.
//!
//! Binary format:
//! ```text
//! [4 bytes] magic: "HOLO"
//! [4 bytes] version: u32 (1)
//! [4 bytes] config_len: u32
//! [config_len bytes] config: JSON-encoded HoloConfig
//! [4 bytes] num_anchors: u32
//! [num_anchors * 12 bytes] anchors: [f32; 3] each
//! [4 bytes] triplane_params_len: u32
//! [triplane_params_len * 4 bytes] triplane params
//! [4 bytes] canonical_params_len: u32
//! [canonical_params_len * 4 bytes] canonical MLP params
//! [4 bytes] audio_encoder_params_len: u32
//! [audio_encoder_params_len * 4 bytes] audio encoder params
//! [4 bytes] cross_attn_params_len: u32
//! [cross_attn_params_len * 4 bytes] cross-attention params
//! [4 bytes] deformation_params_len: u32
//! [deformation_params_len * 4 bytes] deformation params
//! ```

use super::holo_model::{HoloConfig, HoloModel};
use std::io::Write;
use std::path::Path;

const MAGIC: &[u8; 4] = b"HOLO";
const VERSION: u32 = 1;

/// Save a HoloModel to a .holo file.
pub fn save_holo(model: &HoloModel, path: &Path) -> Result<(), std::io::Error> {
    let mut f = std::fs::File::create(path)?;

    f.write_all(MAGIC)?;
    f.write_all(&VERSION.to_le_bytes())?;

    // Config as JSON
    let config_json = config_to_json(&model.config);
    let config_bytes = config_json.as_bytes();
    f.write_all(&(config_bytes.len() as u32).to_le_bytes())?;
    f.write_all(config_bytes)?;

    // Anchors
    f.write_all(&(model.anchors.len() as u32).to_le_bytes())?;
    for anchor in &model.anchors {
        for &v in anchor {
            f.write_all(&v.to_le_bytes())?;
        }
    }

    // Triplane params
    write_params(&mut f, &model.triplane.params_flat())?;

    // Canonical MLP params
    write_params(&mut f, &model.canonical.mlp.params_flat())?;

    // Audio encoder params
    write_params(&mut f, &model.audio_encoder.params_flat())?;

    // Cross-attention params
    write_params(&mut f, &model.cross_attention.params_flat())?;

    // Deformation params
    write_params(&mut f, &model.deformation.params_flat())?;

    Ok(())
}

/// Load a HoloModel from a .holo file.
pub fn load_holo(path: &Path) -> Result<HoloModel, HoloFormatError> {
    let data = std::fs::read(path)?;
    let mut offset = 0;

    // Magic
    if &data[offset..offset + 4] != MAGIC {
        return Err(HoloFormatError::BadMagic);
    }
    offset += 4;

    // Version
    let version = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
    if version != VERSION {
        return Err(HoloFormatError::UnsupportedVersion(version));
    }
    offset += 4;

    // Config
    let config_len = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
    offset += 4;
    let config_json = std::str::from_utf8(&data[offset..offset + config_len])
        .map_err(|_| HoloFormatError::BadConfig)?;
    let config = json_to_config(config_json).ok_or(HoloFormatError::BadConfig)?;
    offset += config_len;

    // Anchors
    let num_anchors = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
    offset += 4;
    let mut anchors = Vec::with_capacity(num_anchors);
    for _ in 0..num_anchors {
        let x = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
        let y = f32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap());
        let z = f32::from_le_bytes(data[offset + 8..offset + 12].try_into().unwrap());
        anchors.push([x, y, z]);
        offset += 12;
    }

    let mut model = HoloModel::new(config);
    model.anchors = anchors;

    // Triplane params
    let tp = read_params(&data, &mut offset)?;
    model.triplane.set_params_flat(&tp);

    // Canonical MLP params
    let cp = read_params(&data, &mut offset)?;
    model.canonical.mlp.set_params_flat(&cp);

    // Audio encoder params
    let ae = read_params(&data, &mut offset)?;
    model.audio_encoder.set_params_flat(&ae);

    // Cross-attention params
    let ca = read_params(&data, &mut offset)?;
    model.cross_attention.set_params_flat(&ca);

    // Deformation params
    let dp = read_params(&data, &mut offset)?;
    model.deformation.set_params_flat(&dp);

    Ok(model)
}

fn write_params(f: &mut std::fs::File, params: &[f32]) -> Result<(), std::io::Error> {
    f.write_all(&(params.len() as u32).to_le_bytes())?;
    for &v in params {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn read_params(data: &[u8], offset: &mut usize) -> Result<Vec<f32>, HoloFormatError> {
    let len = u32::from_le_bytes(data[*offset..*offset + 4].try_into().unwrap()) as usize;
    *offset += 4;
    let params: Vec<f32> = (0..len)
        .map(|i| {
            let o = *offset + i * 4;
            f32::from_le_bytes(data[o..o + 4].try_into().unwrap())
        })
        .collect();
    *offset += len * 4;
    Ok(params)
}

fn config_to_json(config: &HoloConfig) -> String {
    format!(
        r#"{{"triplane_res":{},"feature_dim":{},"num_anchors":{},"mel_dim":{},"audio_window":{},"audio_latent_dim":{},"attn_head_dim":{}}}"#,
        config.triplane_res,
        config.feature_dim,
        config.num_anchors,
        config.mel_dim,
        config.audio_window,
        config.audio_latent_dim,
        config.attn_head_dim,
    )
}

fn json_to_config(json: &str) -> Option<HoloConfig> {
    let get_u32 = |key: &str| -> Option<u32> {
        let pattern = format!("\"{}\":", key);
        let pos = json.find(&pattern)?;
        let rest = &json[pos + pattern.len()..];
        let end = rest.find(|c: char| c == ',' || c == '}')?;
        rest[..end].trim().parse().ok()
    };

    Some(HoloConfig {
        triplane_res: get_u32("triplane_res")?,
        feature_dim: get_u32("feature_dim")? as usize,
        num_anchors: get_u32("num_anchors")? as usize,
        mel_dim: get_u32("mel_dim")? as usize,
        audio_window: get_u32("audio_window")? as usize,
        audio_latent_dim: get_u32("audio_latent_dim")? as usize,
        attn_head_dim: get_u32("attn_head_dim")? as usize,
    })
}

#[derive(Debug)]
pub enum HoloFormatError {
    Io(std::io::Error),
    BadMagic,
    UnsupportedVersion(u32),
    BadConfig,
}

impl From<std::io::Error> for HoloFormatError {
    fn from(e: std::io::Error) -> Self {
        HoloFormatError::Io(e)
    }
}

impl std::fmt::Display for HoloFormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HoloFormatError::Io(e) => write!(f, "IO error: {}", e),
            HoloFormatError::BadMagic => write!(f, "not a .holo file"),
            HoloFormatError::UnsupportedVersion(v) => write!(f, "unsupported version: {}", v),
            HoloFormatError::BadConfig => write!(f, "bad config in .holo file"),
        }
    }
}

impl std::error::Error for HoloFormatError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_save_load_roundtrip() {
        let config = HoloConfig {
            num_anchors: 50,
            triplane_res: 4,
            feature_dim: 8,
            ..Default::default()
        };
        let model = HoloModel::new(config);

        let tmp = std::env::temp_dir().join("test_model.holo");
        save_holo(&model, &tmp).unwrap();
        let loaded = load_holo(&tmp).unwrap();

        assert_eq!(loaded.anchors.len(), model.anchors.len());
        assert_eq!(loaded.config.triplane_res, 4);
        assert_eq!(loaded.config.feature_dim, 8);
        assert_eq!(loaded.config.num_anchors, 50);

        // Verify params match
        assert_eq!(
            loaded.triplane.params_flat(),
            model.triplane.params_flat()
        );

        std::fs::remove_file(&tmp).ok();
    }
}
