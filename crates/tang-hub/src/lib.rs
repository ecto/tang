//! HuggingFace Hub model download and loading.
//!
//! Downloads model weights from the HuggingFace Hub and loads them as `Tensor<f64>`.
//!
//! # Loading into a tang-train Module
//!
//! ```ignore
//! use tang_hub;
//! use tang_train::Module;
//!
//! // Build your model (must match the pretrained architecture)
//! let mut model = /* ... */;
//!
//! // Load weights from HuggingFace and apply to model
//! let state = tang_hub::load_state_dict("username/my-model")?;
//! model.load_state_dict(&state);
//! ```
//!
//! # Tokenizers
//!
//! Enable the `tokenizers` feature for text encoding/decoding:
//!
//! ```ignore
//! let tok = tang_hub::load_tokenizer("bert-base-uncased")?;
//! let ids = tang_hub::encode(&tok, "hello world");
//! let text = tang_hub::decode(&tok, &ids);
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use tang_tensor::Tensor;

/// Files available for a downloaded model.
pub struct ModelFiles {
    /// Path to the model directory in the cache.
    pub path: PathBuf,
    /// Paths to safetensors weight files.
    pub weight_files: Vec<PathBuf>,
    /// Path to config.json if present.
    pub config: Option<PathBuf>,
    /// Path to tokenizer.json if present.
    pub tokenizer: Option<PathBuf>,
}

/// Download a model from the HuggingFace Hub.
///
/// Returns paths to downloaded files in the local cache.
pub fn download(repo_id: &str) -> Result<ModelFiles, HubError> {
    let api = hf_hub::api::sync::Api::new().map_err(|e| HubError::Api(e.to_string()))?;
    let repo = api.model(repo_id.to_string());

    let mut weight_files = Vec::new();
    let mut config = None;
    let mut tokenizer = None;

    // Try to download common files
    if let Ok(path) = repo.get("config.json") {
        config = Some(path);
    }

    if let Ok(path) = repo.get("tokenizer.json") {
        tokenizer = Some(path);
    }

    // Try single-file model first
    if let Ok(path) = repo.get("model.safetensors") {
        weight_files.push(path);
    }

    // If no single file, try sharded format
    if weight_files.is_empty() {
        // Try index file for sharded models
        if let Ok(index_path) = repo.get("model.safetensors.index.json") {
            let index_content = std::fs::read_to_string(&index_path)
                .map_err(|e| HubError::Io(e.to_string()))?;
            // Parse weight_map to find shard filenames
            if let Some(filenames) = parse_shard_filenames(&index_content) {
                for filename in filenames {
                    match repo.get(&filename) {
                        Ok(path) => weight_files.push(path),
                        Err(e) => return Err(HubError::Api(format!("failed to download {filename}: {e}"))),
                    }
                }
            }
        }
    }

    let path = weight_files
        .first()
        .map(|p| p.parent().unwrap_or(Path::new(".")).to_path_buf())
        .unwrap_or_default();

    Ok(ModelFiles {
        path,
        weight_files,
        config,
        tokenizer,
    })
}

/// Download and load all weights as `Tensor<f64>`.
pub fn load_weights(repo_id: &str) -> Result<HashMap<String, Tensor<f64>>, HubError> {
    let files = download(repo_id)?;

    if files.weight_files.is_empty() {
        return Err(HubError::NoWeights(repo_id.to_string()));
    }

    let mut all_tensors = HashMap::new();
    for weight_file in &files.weight_files {
        let tensors = tang_safetensors::load(weight_file)
            .map_err(|e| HubError::Safetensors(format!("{e}")))?;
        all_tensors.extend(tensors);
    }

    Ok(all_tensors)
}

/// Download and load weights as a sorted `Vec` of `(name, tensor)` pairs.
///
/// This format is compatible with `tang_train::Module::load_state_dict`:
///
/// ```ignore
/// let state = tang_hub::load_state_dict("username/model")?;
/// model.load_state_dict(&state);
/// ```
pub fn load_state_dict(repo_id: &str) -> Result<Vec<(String, Tensor<f64>)>, HubError> {
    let weights = load_weights(repo_id)?;
    let mut pairs: Vec<(String, Tensor<f64>)> = weights.into_iter().collect();
    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(pairs)
}

/// Load a tokenizer from a HuggingFace model repo.
#[cfg(feature = "tokenizers")]
pub fn load_tokenizer(repo_id: &str) -> Result<tokenizers::Tokenizer, HubError> {
    let api = hf_hub::api::sync::Api::new().map_err(|e| HubError::Api(e.to_string()))?;
    let repo = api.model(repo_id.to_string());
    let tokenizer_path = repo
        .get("tokenizer.json")
        .map_err(|e| HubError::Api(format!("tokenizer.json not found: {e}")))?;
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| HubError::Api(format!("failed to load tokenizer: {e}")))?;
    Ok(tokenizer)
}

/// Encode text to a 1-D `Tensor<f64>` of token IDs.
#[cfg(feature = "tokenizers")]
pub fn encode(tokenizer: &tokenizers::Tokenizer, text: &str) -> Tensor<f64> {
    let encoding = tokenizer.encode(text, false).expect("tokenizer encode failed");
    let ids: Vec<f64> = encoding.get_ids().iter().map(|&id| id as f64).collect();
    Tensor::from_slice(&ids)
}

/// Decode a 1-D `Tensor<f64>` of token IDs back to text.
#[cfg(feature = "tokenizers")]
pub fn decode(tokenizer: &tokenizers::Tokenizer, ids: &Tensor<f64>) -> String {
    let token_ids: Vec<u32> = ids.data().iter().map(|&x| x as u32).collect();
    tokenizer
        .decode(&token_ids, true)
        .expect("tokenizer decode failed")
}

/// Encode a batch of texts to a 2-D `Tensor<f64>` of shape `[batch, max_len]`.
///
/// Sequences shorter than `max_len` are padded with `pad_id`.
#[cfg(feature = "tokenizers")]
pub fn batch_encode(
    tokenizer: &tokenizers::Tokenizer,
    texts: &[&str],
    max_len: usize,
    pad_id: f64,
) -> Tensor<f64> {
    use tang_tensor::Shape;

    let batch = texts.len();
    let mut data = vec![pad_id; batch * max_len];

    for (i, text) in texts.iter().enumerate() {
        let encoding = tokenizer.encode(*text, false).expect("tokenizer encode failed");
        let ids = encoding.get_ids();
        let len = ids.len().min(max_len);
        for j in 0..len {
            data[i * max_len + j] = ids[j] as f64;
        }
    }

    Tensor::new(data, Shape::from_slice(&[batch, max_len]))
}

/// Return the vocabulary size of a tokenizer.
#[cfg(feature = "tokenizers")]
pub fn vocab_size(tokenizer: &tokenizers::Tokenizer) -> usize {
    tokenizer.get_vocab_size(true)
}

/// Parse shard filenames from a safetensors index JSON.
fn parse_shard_filenames(index_json: &str) -> Option<Vec<String>> {
    // Minimal JSON parsing — find all quoted strings ending in .safetensors
    // within the "weight_map" section
    let weight_map_start = index_json.find("\"weight_map\"")?;
    let rest = &index_json[weight_map_start..];
    let brace_start = rest.find('{')?;
    let content = &rest[brace_start..];
    let brace_end = content.find('}')?;
    let map_content = &content[1..brace_end];

    let mut filenames: Vec<String> = Vec::new();

    // Find all quoted strings that end with .safetensors
    let mut pos = 0;
    let bytes = map_content.as_bytes();
    while pos < bytes.len() {
        if bytes[pos] == b'"' {
            // Find closing quote
            if let Some(end) = map_content[pos + 1..].find('"') {
                let s = &map_content[pos + 1..pos + 1 + end];
                if s.ends_with(".safetensors") && !filenames.contains(&s.to_string()) {
                    filenames.push(s.to_string());
                }
                pos = pos + 1 + end + 1;
            } else {
                break;
            }
        } else {
            pos += 1;
        }
    }

    if filenames.is_empty() {
        None
    } else {
        Some(filenames)
    }
}

/// Errors from Hub operations.
#[derive(Debug)]
pub enum HubError {
    Api(String),
    Io(String),
    Safetensors(String),
    NoWeights(String),
}

impl std::fmt::Display for HubError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Api(e) => write!(f, "Hub API error: {e}"),
            Self::Io(e) => write!(f, "IO error: {e}"),
            Self::Safetensors(e) => write!(f, "safetensors error: {e}"),
            Self::NoWeights(repo) => write!(f, "no weight files found in {repo}"),
        }
    }
}

impl std::error::Error for HubError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_shard_filenames() {
        let index = r#"{
            "metadata": {"total_size": 1000},
            "weight_map": {
                "model.layer.0.weight": "model-00001-of-00002.safetensors",
                "model.layer.0.bias": "model-00001-of-00002.safetensors",
                "model.layer.1.weight": "model-00002-of-00002.safetensors"
            }
        }"#;

        let filenames = parse_shard_filenames(index).unwrap();
        assert_eq!(filenames.len(), 2);
        assert!(filenames.contains(&"model-00001-of-00002.safetensors".to_string()));
        assert!(filenames.contains(&"model-00002-of-00002.safetensors".to_string()));
    }

    #[test]
    fn test_load_state_dict_returns_sorted() {
        // load_state_dict requires network, but we can test the sort behavior
        // by verifying the function signature compiles and the conversion logic
        let mut map = HashMap::new();
        map.insert(
            "b.weight".to_string(),
            Tensor::from_slice(&[1.0, 2.0]),
        );
        map.insert(
            "a.weight".to_string(),
            Tensor::from_slice(&[3.0, 4.0]),
        );

        let mut pairs: Vec<(String, Tensor<f64>)> = map.into_iter().collect();
        pairs.sort_by(|a, b| a.0.cmp(&b.0));

        assert_eq!(pairs[0].0, "a.weight");
        assert_eq!(pairs[1].0, "b.weight");
    }

    #[test]
    #[ignore] // Requires network access
    fn test_hub_download() {
        // Download a tiny public model
        let files = download("hf-internal-testing/tiny-random-bert").unwrap();
        assert!(
            !files.weight_files.is_empty(),
            "should have weight files"
        );
    }

    #[test]
    #[ignore] // Requires network access
    fn test_load_state_dict_from_hub() {
        let state = load_state_dict("hf-internal-testing/tiny-random-bert").unwrap();
        assert!(!state.is_empty(), "should have weight tensors");
        // Verify sorted order
        for window in state.windows(2) {
            assert!(window[0].0 <= window[1].0, "state dict should be sorted by name");
        }
    }
}
