//! Weight quantization: Q8 and Q4 formats with block-wise scaling.
//!
//! Provides quantize/dequantize utilities and quantized matrix-vector multiply
//! for inference. Block-based quantization (like GGML/GGUF) for good accuracy.

use alloc::vec::Vec;
use tang::Scalar;
use tang_tensor::{Shape, Tensor};

/// Block size for quantization (matches GGML Q4_0/Q8_0 format).
const BLOCK_SIZE: usize = 32;

/// Q8 quantized block: 32 int8 values + 1 f32 scale.
#[derive(Clone, Debug)]
pub struct Q8Block {
    /// Scale factor for this block
    pub scale: f32,
    /// Quantized values (-128..127)
    pub values: [i8; BLOCK_SIZE],
}

/// Q4 quantized block: 32 values packed into 16 bytes + 1 f32 scale.
/// Each byte holds two 4-bit values (-8..7).
#[derive(Clone, Debug)]
pub struct Q4Block {
    /// Scale factor for this block
    pub scale: f32,
    /// Packed nibbles: low 4 bits = even index, high 4 bits = odd index
    pub values: [u8; BLOCK_SIZE / 2],
}

/// Quantized tensor in Q8 format.
#[derive(Clone, Debug)]
pub struct QuantizedQ8 {
    pub blocks: Vec<Q8Block>,
    pub shape: Vec<usize>,
}

/// Quantized tensor in Q4 format.
#[derive(Clone, Debug)]
pub struct QuantizedQ4 {
    pub blocks: Vec<Q4Block>,
    pub shape: Vec<usize>,
}

/// Quantize an f32 slice to Q8 blocks.
pub fn quantize_q8(data: &[f32]) -> Vec<Q8Block> {
    let n_blocks = (data.len() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let mut blocks = Vec::with_capacity(n_blocks);

    for b in 0..n_blocks {
        let start = b * BLOCK_SIZE;
        let end = (start + BLOCK_SIZE).min(data.len());
        let chunk = &data[start..end];

        // Find absolute max for scale
        let amax = chunk.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = if amax == 0.0 { 1.0 } else { amax / 127.0 };
        let inv_scale = 1.0 / scale;

        let mut values = [0i8; BLOCK_SIZE];
        for (i, &v) in chunk.iter().enumerate() {
            values[i] = (v * inv_scale).round().clamp(-128.0, 127.0) as i8;
        }

        blocks.push(Q8Block { scale, values });
    }

    blocks
}

/// Dequantize Q8 blocks back to f32.
pub fn dequantize_q8(blocks: &[Q8Block]) -> Vec<f32> {
    let mut out = Vec::with_capacity(blocks.len() * BLOCK_SIZE);
    for block in blocks {
        for &v in &block.values {
            out.push(v as f32 * block.scale);
        }
    }
    out
}

/// Quantize an f32 slice to Q4 blocks (4-bit quantization).
pub fn quantize_q4(data: &[f32]) -> Vec<Q4Block> {
    let n_blocks = (data.len() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let mut blocks = Vec::with_capacity(n_blocks);

    for b in 0..n_blocks {
        let start = b * BLOCK_SIZE;
        let end = (start + BLOCK_SIZE).min(data.len());
        let chunk = &data[start..end];

        let amax = chunk.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = if amax == 0.0 { 1.0 } else { amax / 7.0 };
        let inv_scale = 1.0 / scale;

        let mut values = [0u8; BLOCK_SIZE / 2];
        for i in 0..BLOCK_SIZE / 2 {
            let idx = i * 2;
            let v0 = if idx < chunk.len() {
                (chunk[idx] * inv_scale).round().clamp(-8.0, 7.0) as i8
            } else {
                0
            };
            let v1 = if idx + 1 < chunk.len() {
                (chunk[idx + 1] * inv_scale).round().clamp(-8.0, 7.0) as i8
            } else {
                0
            };
            // Pack: low nibble = v0 (offset by 8), high nibble = v1 (offset by 8)
            values[i] = ((v0 + 8) as u8) | (((v1 + 8) as u8) << 4);
        }

        blocks.push(Q4Block { scale, values });
    }

    blocks
}

/// Dequantize Q4 blocks back to f32.
pub fn dequantize_q4(blocks: &[Q4Block]) -> Vec<f32> {
    let mut out = Vec::with_capacity(blocks.len() * BLOCK_SIZE);
    for block in blocks {
        for &byte in &block.values {
            let v0 = (byte & 0x0F) as i8 - 8;
            let v1 = ((byte >> 4) & 0x0F) as i8 - 8;
            out.push(v0 as f32 * block.scale);
            out.push(v1 as f32 * block.scale);
        }
    }
    out
}

/// Quantize a tensor to Q8 format.
pub fn quantize_tensor_q8<S: Scalar>(tensor: &Tensor<S>) -> QuantizedQ8 {
    let f32_data: Vec<f32> = (0..tensor.numel())
        .map(|i| {
            // Flatten index
            let mut idx = alloc::vec![0usize; tensor.ndim()];
            let mut rem = i;
            for d in (0..tensor.ndim()).rev() {
                idx[d] = rem % tensor.shape()[d];
                rem /= tensor.shape()[d];
            }
            tensor.get(&idx).to_f64() as f32
        })
        .collect();

    QuantizedQ8 {
        blocks: quantize_q8(&f32_data),
        shape: tensor.shape().dims().to_vec(),
    }
}

/// Quantize a tensor to Q4 format.
pub fn quantize_tensor_q4<S: Scalar>(tensor: &Tensor<S>) -> QuantizedQ4 {
    let f32_data: Vec<f32> = (0..tensor.numel())
        .map(|i| {
            let mut idx = alloc::vec![0usize; tensor.ndim()];
            let mut rem = i;
            for d in (0..tensor.ndim()).rev() {
                idx[d] = rem % tensor.shape()[d];
                rem /= tensor.shape()[d];
            }
            tensor.get(&idx).to_f64() as f32
        })
        .collect();

    QuantizedQ4 {
        blocks: quantize_q4(&f32_data),
        shape: tensor.shape().dims().to_vec(),
    }
}

/// Dequantize Q8 back to a tensor.
pub fn dequantize_tensor_q8(q: &QuantizedQ8) -> Tensor<f64> {
    let data: Vec<f64> = dequantize_q8(&q.blocks).into_iter().map(|x| x as f64).collect();
    let numel: usize = q.shape.iter().product();
    Tensor::new(data[..numel].to_vec(), Shape::from_slice(&q.shape))
}

/// Dequantize Q4 back to a tensor.
pub fn dequantize_tensor_q4(q: &QuantizedQ4) -> Tensor<f64> {
    let data: Vec<f64> = dequantize_q4(&q.blocks).into_iter().map(|x| x as f64).collect();
    let numel: usize = q.shape.iter().product();
    Tensor::new(data[..numel].to_vec(), Shape::from_slice(&q.shape))
}

/// Quantize a 2D weight matrix to Q8 format, row-by-row.
///
/// Each row is independently quantized into its own block(s), ensuring
/// alignment for `q8_matvec`. Shape must be `[out_dim, in_dim]`.
pub fn quantize_matrix_q8(data: &[f32], rows: usize, cols: usize) -> QuantizedQ8 {
    assert_eq!(data.len(), rows * cols);
    let mut blocks = Vec::new();
    for r in 0..rows {
        let row = &data[r * cols..(r + 1) * cols];
        let row_blocks = quantize_q8(row);
        blocks.extend(row_blocks);
    }
    QuantizedQ8 {
        blocks,
        shape: alloc::vec![rows, cols],
    }
}

/// Quantize a 2D weight matrix to Q4 format, row-by-row.
pub fn quantize_matrix_q4(data: &[f32], rows: usize, cols: usize) -> QuantizedQ4 {
    assert_eq!(data.len(), rows * cols);
    let mut blocks = Vec::new();
    for r in 0..rows {
        let row = &data[r * cols..(r + 1) * cols];
        let row_blocks = quantize_q4(row);
        blocks.extend(row_blocks);
    }
    QuantizedQ4 {
        blocks,
        shape: alloc::vec![rows, cols],
    }
}

/// Quantized matrix-vector multiply: y = Q8_weight @ x
///
/// Weight must be quantized row-by-row (use `quantize_matrix_q8`).
/// Shape is [out, in], x is [in]. Returns [out].
/// Dequantizes on-the-fly per block for cache efficiency.
pub fn q8_matvec(weight: &QuantizedQ8, x: &[f64]) -> Vec<f64> {
    assert_eq!(weight.shape.len(), 2);
    let out_dim = weight.shape[0];
    let in_dim = weight.shape[1];
    assert_eq!(x.len(), in_dim);

    let blocks_per_row = (in_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let mut y = alloc::vec![0.0f64; out_dim];

    for row in 0..out_dim {
        let mut sum = 0.0f64;
        for b in 0..blocks_per_row {
            let block = &weight.blocks[row * blocks_per_row + b];
            let x_start = b * BLOCK_SIZE;
            let scale = block.scale as f64;
            for i in 0..BLOCK_SIZE {
                let xi = x_start + i;
                if xi < in_dim {
                    sum += (block.values[i] as f64) * scale * x[xi];
                }
            }
        }
        y[row] = sum;
    }

    y
}

/// Quantized matrix-vector multiply: y = Q4_weight @ x
pub fn q4_matvec(weight: &QuantizedQ4, x: &[f64]) -> Vec<f64> {
    assert_eq!(weight.shape.len(), 2);
    let out_dim = weight.shape[0];
    let in_dim = weight.shape[1];
    assert_eq!(x.len(), in_dim);

    let blocks_per_row = (in_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let mut y = alloc::vec![0.0f64; out_dim];

    for row in 0..out_dim {
        let mut sum = 0.0f64;
        for b in 0..blocks_per_row {
            let block = &weight.blocks[row * blocks_per_row + b];
            let x_start = b * BLOCK_SIZE;
            let scale = block.scale as f64;
            for i in 0..BLOCK_SIZE / 2 {
                let byte = block.values[i];
                let v0 = ((byte & 0x0F) as i8 - 8) as f64;
                let v1 = (((byte >> 4) & 0x0F) as i8 - 8) as f64;
                let xi0 = x_start + i * 2;
                let xi1 = xi0 + 1;
                if xi0 < in_dim {
                    sum += v0 * scale * x[xi0];
                }
                if xi1 < in_dim {
                    sum += v1 * scale * x[xi1];
                }
            }
        }
        y[row] = sum;
    }

    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn q8_roundtrip() {
        let data = vec![0.5, -1.0, 2.0, -3.0, 0.0, 1.5, -0.7, 0.3];
        let blocks = quantize_q8(&data);
        let restored = dequantize_q8(&blocks);

        // Q8 should be very close (within ~1% of max)
        for (orig, deq) in data.iter().zip(restored.iter()) {
            assert!(
                (orig - deq).abs() < 0.05,
                "Q8 roundtrip: orig={orig}, deq={deq}"
            );
        }
    }

    #[test]
    fn q4_roundtrip() {
        let data = vec![0.5, -1.0, 2.0, -3.0, 0.0, 1.5, -0.7, 0.3];
        let blocks = quantize_q4(&data);
        let restored = dequantize_q4(&blocks);

        // Q4 has more error (~15% of max) but should be in right ballpark
        for (orig, deq) in data.iter().zip(restored.iter()) {
            assert!(
                (orig - deq).abs() < 1.0,
                "Q4 roundtrip: orig={orig}, deq={deq}"
            );
        }
    }

    #[test]
    fn q8_matvec_basic() {
        // 2x3 weight matrix, quantized row-by-row
        let weight_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let q = quantize_matrix_q8(&weight_data, 2, 3);

        let x = vec![1.0, 0.5, -1.0];
        let y = q8_matvec(&q, &x);

        // Expected: [1*1 + 2*0.5 + 3*(-1), 4*1 + 5*0.5 + 6*(-1)] = [-1.0, 0.5]
        assert_eq!(y.len(), 2);
        assert!((y[0] - (-1.0)).abs() < 0.15, "y[0]={}", y[0]);
        assert!((y[1] - 0.5).abs() < 0.15, "y[1]={}", y[1]);
    }

    #[test]
    fn q4_matvec_basic() {
        let weight_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let q = quantize_matrix_q4(&weight_data, 2, 3);

        let x = vec![1.0, 0.5, -1.0];
        let y = q4_matvec(&q, &x);

        assert_eq!(y.len(), 2);
        // Q4 has more error but should be directionally correct
        assert!((y[0] - (-1.0)).abs() < 1.5, "y[0]={}", y[0]);
        assert!((y[1] - 0.5).abs() < 1.5, "y[1]={}", y[1]);
    }

    #[test]
    fn tensor_quantize_roundtrip() {
        let t = Tensor::from_slice(&[1.0_f64, -2.0, 3.0, -4.0, 5.0, -6.0]);
        let t2d = t.reshape(Shape::from_slice(&[2, 3]));

        let q8 = quantize_tensor_q8(&t2d);
        let restored = dequantize_tensor_q8(&q8);

        assert_eq!(restored.shape().dims(), &[2, 3]);
        for i in 0..2 {
            for j in 0..3 {
                let orig = t2d.get(&[i, j]);
                let deq = restored.get(&[i, j]);
                assert!(
                    (orig - deq).abs() < 0.1,
                    "Q8 tensor roundtrip [{i},{j}]: orig={orig}, deq={deq}"
                );
            }
        }
    }
}
