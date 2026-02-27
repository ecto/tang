//! HuggingFace safetensors weight load/save.

use std::collections::HashMap;
use std::path::Path;

use crate::device::GpuDevice;
use crate::tensor::GpuTensor;

/// Load tensors from a safetensors file.
pub fn load_safetensors(
    device: &GpuDevice,
    path: &Path,
) -> Result<HashMap<String, GpuTensor>, SafetensorsError> {
    let data = std::fs::read(path).map_err(SafetensorsError::Io)?;
    let tensors = safetensors::SafeTensors::deserialize(&data).map_err(SafetensorsError::Parse)?;

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

        let tensor = GpuTensor::from_slice(device, &f32_data, &shape);
        result.insert(name, tensor);
    }

    Ok(result)
}

/// Save tensors to a safetensors file.
pub fn save_safetensors(
    tensors: &HashMap<String, GpuTensor>,
    device: &GpuDevice,
    path: &Path,
) -> Result<(), SafetensorsError> {
    let mut tensor_data: Vec<(String, Vec<f32>, Vec<usize>)> = Vec::new();

    for (name, tensor) in tensors {
        let data = tensor.buffer.to_vec_sync(device);
        let shape = tensor.shape().to_vec();
        tensor_data.push((name.clone(), data, shape));
    }

    // Build the tensor map for serialization
    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensor_data
        .iter()
        .map(|(name, data, shape)| {
            let bytes: &[u8] = bytemuck::cast_slice(data);
            let view =
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
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
