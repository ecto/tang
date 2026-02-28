//! CPU safetensors load/save for tang tensors.
//!
//! Supports F32, F64, F16, BF16 dtypes from the HuggingFace safetensors format.

use std::collections::HashMap;
use std::path::Path;

use tang_tensor::{Shape, Tensor};

/// Load tensors from a safetensors file as `Tensor<f64>`.
pub fn load(path: &Path) -> Result<HashMap<String, Tensor<f64>>, SafetensorsError> {
    let data = std::fs::read(path).map_err(SafetensorsError::Io)?;
    let tensors =
        safetensors::SafeTensors::deserialize(&data).map_err(SafetensorsError::Parse)?;

    let mut result = HashMap::new();

    for (name, view) in tensors.tensors() {
        let shape: Vec<usize> = view.shape().to_vec();
        let dtype = view.dtype();

        let f64_data: Vec<f64> = match dtype {
            safetensors::Dtype::F64 => {
                let bytes = view.data();
                bytes
                    .chunks_exact(8)
                    .map(|chunk| {
                        f64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ])
                    })
                    .collect()
            }
            safetensors::Dtype::F32 => {
                let bytes = view.data();
                bytes
                    .chunks_exact(4)
                    .map(|chunk| {
                        f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f64
                    })
                    .collect()
            }
            safetensors::Dtype::F16 => {
                let bytes = view.data();
                bytes
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        f16_to_f32(bits) as f64
                    })
                    .collect()
            }
            safetensors::Dtype::BF16 => {
                let bytes = view.data();
                bytes
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        bf16_to_f32(bits) as f64
                    })
                    .collect()
            }
            other => {
                return Err(SafetensorsError::UnsupportedDtype(format!("{other:?}")));
            }
        };

        let tensor = Tensor::new(f64_data, Shape::new(shape));
        result.insert(name, tensor);
    }

    Ok(result)
}

/// Save tensors to a safetensors file (as F64).
pub fn save(
    tensors: &HashMap<String, Tensor<f64>>,
    path: &Path,
) -> Result<(), SafetensorsError> {
    // Collect owned data so we can borrow it for TensorView
    let tensor_data: Vec<(String, Vec<u8>, Vec<usize>)> = tensors
        .iter()
        .map(|(name, tensor)| {
            let bytes: Vec<u8> = tensor
                .data()
                .iter()
                .flat_map(|&v| v.to_le_bytes())
                .collect();
            let shape = tensor.shape().dims().to_vec();
            (name.clone(), bytes, shape)
        })
        .collect();

    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensor_data
        .iter()
        .map(|(name, bytes, shape)| {
            let view = safetensors::tensor::TensorView::new(
                safetensors::Dtype::F64,
                shape.clone(),
                bytes,
            )
            .unwrap();
            (name.clone(), view)
        })
        .collect();

    let serialized = safetensors::tensor::serialize(
        views
            .iter()
            .map(|(name, view)| (name.as_str(), view.clone())),
        &None,
    )
    .map_err(SafetensorsError::Serialize)?;

    std::fs::write(path, serialized).map_err(SafetensorsError::Io)?;

    Ok(())
}

/// Save tensors to a safetensors file (as F32).
pub fn save_f32(
    tensors: &HashMap<String, Tensor<f32>>,
    path: &Path,
) -> Result<(), SafetensorsError> {
    let tensor_data: Vec<(String, Vec<u8>, Vec<usize>)> = tensors
        .iter()
        .map(|(name, tensor)| {
            let bytes: Vec<u8> = tensor
                .data()
                .iter()
                .flat_map(|&v| v.to_le_bytes())
                .collect();
            let shape = tensor.shape().dims().to_vec();
            (name.clone(), bytes, shape)
        })
        .collect();

    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensor_data
        .iter()
        .map(|(name, bytes, shape)| {
            let view = safetensors::tensor::TensorView::new(
                safetensors::Dtype::F32,
                shape.clone(),
                bytes,
            )
            .unwrap();
            (name.clone(), view)
        })
        .collect();

    let serialized = safetensors::tensor::serialize(
        views
            .iter()
            .map(|(name, view)| (name.as_str(), view.clone())),
        &None,
    )
    .map_err(SafetensorsError::Serialize)?;

    std::fs::write(path, serialized).map_err(SafetensorsError::Io)?;

    Ok(())
}

/// Load tensors as `Tensor<f32>` (useful for GPU interop).
pub fn load_f32(path: &Path) -> Result<HashMap<String, Tensor<f32>>, SafetensorsError> {
    let data = std::fs::read(path).map_err(SafetensorsError::Io)?;
    let tensors =
        safetensors::SafeTensors::deserialize(&data).map_err(SafetensorsError::Parse)?;

    let mut result = HashMap::new();

    for (name, view) in tensors.tensors() {
        let shape: Vec<usize> = view.shape().to_vec();
        let dtype = view.dtype();

        let f32_data: Vec<f32> = match dtype {
            safetensors::Dtype::F32 => {
                let bytes = view.data();
                bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect()
            }
            safetensors::Dtype::F64 => {
                let bytes = view.data();
                bytes
                    .chunks_exact(8)
                    .map(|chunk| {
                        f64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ]) as f32
                    })
                    .collect()
            }
            safetensors::Dtype::F16 => {
                let bytes = view.data();
                bytes
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        f16_to_f32(bits)
                    })
                    .collect()
            }
            safetensors::Dtype::BF16 => {
                let bytes = view.data();
                bytes
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        bf16_to_f32(bits)
                    })
                    .collect()
            }
            other => {
                return Err(SafetensorsError::UnsupportedDtype(format!("{other:?}")));
            }
        };

        let tensor = Tensor::new(f32_data, Shape::new(shape));
        result.insert(name, tensor);
    }

    Ok(result)
}

/// Errors from safetensors operations.
#[derive(Debug)]
pub enum SafetensorsError {
    Io(std::io::Error),
    Parse(safetensors::SafeTensorError),
    Serialize(safetensors::SafeTensorError),
    UnsupportedDtype(String),
}

impl std::fmt::Display for SafetensorsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {e}"),
            Self::Parse(e) => write!(f, "parse error: {e}"),
            Self::Serialize(e) => write!(f, "serialize error: {e}"),
            Self::UnsupportedDtype(d) => write!(f, "unsupported dtype: {d}"),
        }
    }
}

impl std::error::Error for SafetensorsError {}

/// Convert f16 bits to f32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal
        let mut m = mant;
        let mut e = 0i32;
        while m & 0x400 == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        let f32_exp = (127 - 15 + 1 + e) as u32;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13));
    }

    if exp == 31 {
        if mant == 0 {
            return f32::from_bits((sign << 31) | (0xFF << 23));
        }
        return f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13));
    }

    let f32_exp = exp + 127 - 15;
    f32::from_bits((sign << 31) | (f32_exp << 23) | (mant << 13))
}

/// Convert bf16 bits to f32.
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safetensors_cpu_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.safetensors");

        let mut tensors = HashMap::new();
        tensors.insert(
            "weight".to_string(),
            Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::from_slice(&[2, 3])),
        );
        tensors.insert(
            "bias".to_string(),
            Tensor::from_slice(&[0.1, 0.2, 0.3]),
        );

        save(&tensors, &path).unwrap();
        let loaded = load(&path).unwrap();

        assert_eq!(loaded.len(), 2);

        let w = loaded.get("weight").unwrap();
        assert_eq!(w.shape().dims(), &[2, 3]);
        assert!((w.get(&[0, 0]) - 1.0).abs() < 1e-10);
        assert!((w.get(&[1, 2]) - 6.0).abs() < 1e-10);

        let b = loaded.get("bias").unwrap();
        assert_eq!(b.shape().dims(), &[3]);
        assert!((b.get(&[1]) - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_load_into_module() {
        use tang_train::{Linear, Module};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");

        // Create a model, save its weights via state_dict
        let mut model = Linear::<f64>::new(3, 2, 42);
        let state = model.state_dict();
        let tensors: HashMap<String, Tensor<f64>> = state.into_iter().collect();
        save(&tensors, &path).unwrap();

        // Load into a fresh model via load_state_dict
        let loaded = load(&path).unwrap();
        let state: Vec<(String, Tensor<f64>)> = loaded.into_iter().collect();
        let mut model2 = Linear::<f64>::new(3, 2, 0); // different seed
        model2.load_state_dict(&state);

        // Verify they produce the same output
        let input = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        let out1 = model.forward(&input);
        let out2 = model2.forward(&input);
        for (&a, &b) in out1.data().iter().zip(out2.data().iter()) {
            assert!((a - b).abs() < 1e-10_f64);
        }
    }

    #[test]
    fn test_sequential_roundtrip() {
        use tang_train::{Linear, Module, Sequential, Tanh};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("seq.safetensors");

        // Build and save a Sequential model
        let mut model = Sequential::<f64>::new(vec![
            Box::new(Linear::new(4, 8, 42)),
            Box::new(Tanh::new()),
            Box::new(Linear::new(8, 2, 137)),
        ]);

        // Verify naming convention
        let names: Vec<String> = model.named_parameters().iter().map(|(n, _)| n.clone()).collect();
        assert_eq!(names, vec!["0.weight", "0.bias", "2.weight", "2.bias"]);

        // Save
        let state = model.state_dict();
        let tensors: HashMap<String, Tensor<f64>> = state.into_iter().collect();
        save(&tensors, &path).unwrap();

        // Load into a fresh model with different seeds
        let loaded = load(&path).unwrap();
        let state: Vec<(String, Tensor<f64>)> = loaded.into_iter().collect();
        let mut model2 = Sequential::<f64>::new(vec![
            Box::new(Linear::new(4, 8, 0)),
            Box::new(Tanh::new()),
            Box::new(Linear::new(8, 2, 0)),
        ]);
        model2.load_state_dict(&state);

        // Both models should produce identical output
        let input = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let out1 = model.forward(&input);
        let out2 = model2.forward(&input);
        for (&a, &b) in out1.data().iter().zip(out2.data().iter()) {
            assert!((a - b).abs() < 1e-10_f64, "mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_f16_conversion() {
        // 1.0 in f16 = 0x3C00
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 1e-6);
        // 0.0 in f16
        assert_eq!(f16_to_f32(0x0000), 0.0);
        // -1.0 in f16 = 0xBC00
        assert!((f16_to_f32(0xBC00) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bf16_conversion() {
        // 1.0 in bf16 = 0x3F80
        assert!((bf16_to_f32(0x3F80) - 1.0).abs() < 1e-6);
        // 0.0
        assert_eq!(bf16_to_f32(0x0000), 0.0);
    }
}
