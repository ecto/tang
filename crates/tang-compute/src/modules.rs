//! Neural network modules generic over ComputeDevice.
//!
//! Provides Linear, RMSNorm, Embedding, KVCache, and InterleavedRoPE modules
//! that work with any ComputeDevice backend (CPU, Metal, CUDA).

use crate::device::{ComputeBuffer, ComputeDevice};
use crate::ops::bias_add;
use crate::tensor::ComputeTensor;

// ---------------------------------------------------------------------------
// Linear
// ---------------------------------------------------------------------------

/// Linear layer: y = x @ W^T + b.
pub struct Linear<B: ComputeBuffer> {
    /// Weight matrix [out_features, in_features] (row-major).
    pub weight: ComputeTensor<B>,
    /// Bias vector [out_features].
    pub bias: ComputeTensor<B>,
    pub in_features: usize,
    pub out_features: usize,
}

impl<B: ComputeBuffer> Linear<B> {
    /// Create from weight and bias data.
    pub fn new<D: ComputeDevice<Buffer = B>>(
        dev: &D,
        weight: &[f32],
        bias: &[f32],
        in_f: usize,
        out_f: usize,
    ) -> Self {
        assert_eq!(weight.len(), out_f * in_f);
        assert_eq!(bias.len(), out_f);
        Self {
            weight: ComputeTensor::from_data(dev, weight, &[out_f, in_f]),
            bias: ComputeTensor::from_data(dev, bias, &[out_f]),
            in_features: in_f,
            out_features: out_f,
        }
    }

    /// Create zero-initialized.
    pub fn zeros<D: ComputeDevice<Buffer = B>>(dev: &D, in_f: usize, out_f: usize) -> Self {
        Self {
            weight: ComputeTensor::zeros(dev, &[out_f, in_f]),
            bias: ComputeTensor::zeros(dev, &[out_f]),
            in_features: in_f,
            out_features: out_f,
        }
    }

    /// Forward pass for 2D input [batch, in_features] → [batch, out_features].
    ///
    /// Computes: input @ weight^T + bias.
    pub fn forward_2d<D: ComputeDevice<Buffer = B>>(
        &self,
        dev: &D,
        input: &ComputeTensor<B>,
    ) -> ComputeTensor<B> {
        let batch = input.numel() / self.in_features;
        assert_eq!(input.numel(), batch * self.in_features);

        // Transpose weight on device: [out_f, in_f] → [in_f, out_f]
        let wt_buf = dev.transpose_2d(&self.weight.buffer, self.out_features, self.in_features);

        // matmul: [batch, in_f] @ [in_f, out_f] = [batch, out_f]
        let out_buf = dev.matmul(&input.buffer, &wt_buf, batch, self.in_features, self.out_features);
        let out = ComputeTensor::from_buffer(out_buf, vec![batch, self.out_features]);

        // Add bias
        bias_add(dev, &out, &self.bias)
    }
}

/// Cached activations from Linear forward_2d_train.
pub struct LinearCache<B: ComputeBuffer> {
    /// Input [batch, in_features] (on device).
    pub input: B,
    pub batch: usize,
}

impl<B: ComputeBuffer> Linear<B> {
    /// Forward pass for training — same as forward_2d but caches input for backward.
    pub fn forward_2d_train<D: ComputeDevice<Buffer = B>>(
        &self,
        dev: &D,
        input: &ComputeTensor<B>,
    ) -> (ComputeTensor<B>, LinearCache<B>) {
        let batch = input.numel() / self.in_features;
        assert_eq!(input.numel(), batch * self.in_features);

        // Transpose weight on device: [out_f, in_f] → [in_f, out_f]
        let wt_buf = dev.transpose_2d(&self.weight.buffer, self.out_features, self.in_features);

        // matmul: [batch, in_f] @ [in_f, out_f] = [batch, out_f]
        let out_buf = dev.matmul(&input.buffer, &wt_buf, batch, self.in_features, self.out_features);
        let out = ComputeTensor::from_buffer(out_buf, vec![batch, self.out_features]);

        // Cache input for backward (copy on device, no CPU round-trip)
        let cached_input = dev.copy_buffer(&input.buffer);

        let out = bias_add(dev, &out, &self.bias);
        let cache = LinearCache { input: cached_input, batch };
        (out, cache)
    }

    /// Backward pass: grad_output [batch, out_f] → (grad_input, grad_weight, grad_bias).
    ///
    /// grad_weight and grad_bias are downloaded to CPU.
    pub fn backward_2d<D: ComputeDevice<Buffer = B>>(
        &self,
        dev: &D,
        grad_output: &ComputeTensor<B>,
        cache: &LinearCache<B>,
    ) -> (ComputeTensor<B>, Vec<f32>, Vec<f32>) {
        let batch = cache.batch;

        // grad_input = grad_output @ W  (W is [out_f, in_f])
        // grad_output: [batch, out_f], W: [out_f, in_f] → [batch, in_f]
        let gi_buf = dev.matmul(&grad_output.buffer, &self.weight.buffer, batch, self.out_features, self.in_features);
        let grad_input = ComputeTensor::from_buffer(gi_buf, vec![batch, self.in_features]);

        // grad_weight = grad_output^T @ input
        // grad_output^T: [out_f, batch], input: [batch, in_f] → [out_f, in_f]
        let go_t = dev.transpose_2d(&grad_output.buffer, batch, self.out_features);
        let gw_buf = dev.matmul(&go_t, &cache.input, self.out_features, batch, self.in_features);
        let grad_weight = dev.download(&gw_buf);

        // grad_bias = sum(grad_output, axis=0) → [out_f]
        let grad_bias = dev.download(
            &dev.reduce_sum(&grad_output.buffer, &[batch, self.out_features], 0),
        );

        (grad_input, grad_weight, grad_bias)
    }
}

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------

/// RMS normalization: x * weight / sqrt(mean(x^2) + eps).
pub struct RMSNorm<B: ComputeBuffer> {
    /// Learnable scale [dim].
    pub weight: ComputeTensor<B>,
    pub eps: f32,
    pub dim: usize,
}

impl<B: ComputeBuffer> RMSNorm<B> {
    /// Create with unit weights.
    pub fn new<D: ComputeDevice<Buffer = B>>(dev: &D, dim: usize, eps: f32) -> Self {
        Self {
            weight: ComputeTensor::from_data(dev, &vec![1.0f32; dim], &[dim]),
            eps,
            dim,
        }
    }

    /// Forward pass: [n_groups, dim] → [n_groups, dim].
    pub fn forward<D: ComputeDevice<Buffer = B>>(
        &self,
        dev: &D,
        input: &ComputeTensor<B>,
    ) -> ComputeTensor<B> {
        let n_groups = input.numel() / self.dim;
        let buf = dev.rms_norm(&input.buffer, &self.weight.buffer, n_groups, self.dim, self.eps);
        ComputeTensor::from_buffer(buf, input.shape().to_vec())
    }
}

/// Cached activations from RMSNorm forward_train.
pub struct RMSNormCache<B: ComputeBuffer> {
    /// Input [n_groups, dim] (on device).
    pub input: B,
}

impl<B: ComputeBuffer> RMSNorm<B> {
    /// Forward pass for training — caches input for backward.
    pub fn forward_train<D: ComputeDevice<Buffer = B>>(
        &self,
        dev: &D,
        input: &ComputeTensor<B>,
    ) -> (ComputeTensor<B>, RMSNormCache<B>) {
        let n_groups = input.numel() / self.dim;
        let buf = dev.rms_norm(&input.buffer, &self.weight.buffer, n_groups, self.dim, self.eps);
        let out = ComputeTensor::from_buffer(buf, input.shape().to_vec());

        let cache = RMSNormCache { input: dev.copy_buffer(&input.buffer) };
        (out, cache)
    }

    /// Backward pass: grad_output [n_groups, dim] → (grad_input, grad_weight).
    ///
    /// grad_weight is downloaded to CPU.
    pub fn backward<D: ComputeDevice<Buffer = B>>(
        &self,
        dev: &D,
        grad_output: &ComputeTensor<B>,
        cache: &RMSNormCache<B>,
    ) -> (ComputeTensor<B>, Vec<f32>) {
        let n_groups = grad_output.numel() / self.dim;
        let (gi_buf, gw_buf) = dev.rms_norm_backward(
            &cache.input, &self.weight.buffer, &grad_output.buffer,
            n_groups, self.dim, self.eps,
        );
        let grad_weight = dev.download(&gw_buf);
        let grad_input = ComputeTensor::from_buffer(gi_buf, grad_output.shape().to_vec());
        (grad_input, grad_weight)
    }
}

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

/// Token embedding lookup table.
pub struct Embedding<B: ComputeBuffer> {
    /// Weight matrix [vocab_size, dim].
    pub weight: ComputeTensor<B>,
    pub vocab_size: usize,
    pub dim: usize,
}

impl<B: ComputeBuffer> Embedding<B> {
    /// Create from weight data.
    pub fn new<D: ComputeDevice<Buffer = B>>(
        dev: &D,
        data: &[f32],
        vocab_size: usize,
        dim: usize,
    ) -> Self {
        assert_eq!(data.len(), vocab_size * dim);
        Self {
            weight: ComputeTensor::from_data(dev, data, &[vocab_size, dim]),
            vocab_size,
            dim,
        }
    }

    /// Create zero-initialized.
    pub fn zeros<D: ComputeDevice<Buffer = B>>(dev: &D, vocab_size: usize, dim: usize) -> Self {
        Self {
            weight: ComputeTensor::zeros(dev, &[vocab_size, dim]),
            vocab_size,
            dim,
        }
    }

    /// Lookup: token IDs → [seq_len, dim].
    pub fn forward<D: ComputeDevice<Buffer = B>>(
        &self,
        dev: &D,
        ids: &B,
        seq_len: usize,
    ) -> ComputeTensor<B> {
        let buf = dev.embedding(&self.weight.buffer, ids, seq_len, self.dim);
        ComputeTensor::from_buffer(buf, vec![seq_len, self.dim])
    }
}

/// Cached activations from Embedding forward_train.
pub struct EmbeddingCache<B: ComputeBuffer> {
    /// Token IDs (on device).
    pub ids: B,
    pub seq_len: usize,
}

impl<B: ComputeBuffer> Embedding<B> {
    /// Forward pass for training — caches IDs for backward.
    pub fn forward_train<D: ComputeDevice<Buffer = B>>(
        &self,
        dev: &D,
        ids: &B,
        seq_len: usize,
    ) -> (ComputeTensor<B>, EmbeddingCache<B>) {
        let buf = dev.embedding(&self.weight.buffer, ids, seq_len, self.dim);
        let out = ComputeTensor::from_buffer(buf, vec![seq_len, self.dim]);

        let cache = EmbeddingCache { ids: dev.copy_buffer(ids), seq_len };
        (out, cache)
    }

    /// Backward pass: grad_output [seq_len, dim] → grad_weight [vocab_size, dim].
    ///
    /// grad_weight is downloaded to CPU.
    pub fn backward<D: ComputeDevice<Buffer = B>>(
        &self,
        dev: &D,
        grad_output: &ComputeTensor<B>,
        cache: &EmbeddingCache<B>,
    ) -> Vec<f32> {
        let gw_buf = dev.embedding_backward(
            &grad_output.buffer, &cache.ids,
            self.vocab_size, cache.seq_len, self.dim,
        );
        dev.download(&gw_buf)
    }
}

// ---------------------------------------------------------------------------
// KVCache (CPU-side storage)
// ---------------------------------------------------------------------------

/// KV cache with CPU-side storage.
///
/// Simple and correct: stores keys/values as flat CPU vectors,
/// uploads to device on demand for attention computation.
pub struct KVCache {
    keys: Vec<f32>,
    values: Vec<f32>,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub len: usize,
}

impl KVCache {
    /// Create a new empty KV cache.
    pub fn new(n_kv_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        let entry_size = n_kv_heads * head_dim;
        Self {
            keys: Vec::with_capacity(max_seq_len * entry_size),
            values: Vec::with_capacity(max_seq_len * entry_size),
            n_kv_heads,
            head_dim,
            max_seq_len,
            len: 0,
        }
    }

    /// Append new key/value entries (one or more positions).
    ///
    /// `new_k` and `new_v` are flat: [new_len, n_kv_heads * head_dim].
    pub fn append(&mut self, new_k: &[f32], new_v: &[f32]) {
        let entry_size = self.n_kv_heads * self.head_dim;
        let new_len = new_k.len() / entry_size;
        assert_eq!(new_k.len(), new_len * entry_size);
        assert_eq!(new_v.len(), new_len * entry_size);
        assert!(self.len + new_len <= self.max_seq_len, "KV cache overflow");
        self.keys.extend_from_slice(new_k);
        self.values.extend_from_slice(new_v);
        self.len += new_len;
    }

    /// Upload current keys to device as [len, n_kv_heads * head_dim].
    pub fn get_keys_buffer<D: ComputeDevice>(&self, dev: &D) -> D::Buffer {
        dev.upload(&self.keys[..self.len * self.n_kv_heads * self.head_dim])
    }

    /// Upload current values to device as [len, n_kv_heads * head_dim].
    pub fn get_values_buffer<D: ComputeDevice>(&self, dev: &D) -> D::Buffer {
        dev.upload(&self.values[..self.len * self.n_kv_heads * self.head_dim])
    }

    /// Reset cache for new generation.
    pub fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
        self.len = 0;
    }
}

// ---------------------------------------------------------------------------
// InterleavedRoPE (CPU-side tables)
// ---------------------------------------------------------------------------

/// Interleaved Rotary Position Embedding.
///
/// Precomputes cos/sin tables on CPU, applies RoPE via CPU roundtrip.
/// Rotates interleaved pairs (2i, 2i+1) to match CPU convention.
pub struct InterleavedRoPE {
    cos_table: Vec<f32>,
    sin_table: Vec<f32>,
    pub head_dim: usize,
    pub max_seq_len: usize,
}

impl InterleavedRoPE {
    /// Create with precomputed tables.
    ///
    /// `base` is the RoPE frequency base (e.g. 500000.0).
    pub fn new(head_dim: usize, max_seq_len: usize, base: f64) -> Self {
        let half_dim = head_dim / 2;
        let mut cos_table = vec![0.0f32; max_seq_len * half_dim];
        let mut sin_table = vec![0.0f32; max_seq_len * half_dim];

        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let freq = 1.0 / base.powf(i as f64 / half_dim as f64);
                let angle = pos as f64 * freq;
                cos_table[pos * half_dim + i] = angle.cos() as f32;
                sin_table[pos * half_dim + i] = angle.sin() as f32;
            }
        }

        Self { cos_table, sin_table, head_dim, max_seq_len }
    }

    /// Apply RoPE to input tensor.
    ///
    /// input: [seq_len, n_heads, head_dim] → output: same shape.
    /// `start_pos` is the absolute position offset (for KV cache continuation).
    pub fn forward<D: ComputeDevice>(
        &self,
        dev: &D,
        input: &ComputeTensor<D::Buffer>,
        start_pos: usize,
    ) -> ComputeTensor<D::Buffer> {
        let shape = input.shape();
        assert_eq!(shape.len(), 3, "InterleavedRoPE expects 3D [seq, heads, dim]");
        let seq_len = shape[0];
        let n_heads = shape[1];
        let head_dim = shape[2];
        assert_eq!(head_dim, self.head_dim);
        let half_dim = head_dim / 2;

        // Download, apply RoPE on CPU, re-upload
        let data = dev.download(&input.buffer);
        let mut out = vec![0.0f32; data.len()];

        for s in 0..seq_len {
            let pos = start_pos + s;
            assert!(pos < self.max_seq_len, "RoPE position {pos} >= max {}", self.max_seq_len);
            for h in 0..n_heads {
                let base_idx = (s * n_heads + h) * head_dim;
                for i in 0..half_dim {
                    let cos = self.cos_table[pos * half_dim + i];
                    let sin = self.sin_table[pos * half_dim + i];
                    let x0 = data[base_idx + 2 * i];
                    let x1 = data[base_idx + 2 * i + 1];
                    out[base_idx + 2 * i] = x0 * cos - x1 * sin;
                    out[base_idx + 2 * i + 1] = x0 * sin + x1 * cos;
                }
            }
        }

        ComputeTensor::from_data(dev, &out, shape)
    }
}

impl InterleavedRoPE {
    /// Backward pass for RoPE: reverse rotation.
    ///
    /// RoPE forward: (x0*cos - x1*sin, x0*sin + x1*cos)
    /// RoPE backward: (g0*cos + g1*sin, -g0*sin + g1*cos)
    pub fn backward<D: ComputeDevice>(
        &self,
        dev: &D,
        grad_output: &ComputeTensor<D::Buffer>,
        start_pos: usize,
    ) -> ComputeTensor<D::Buffer> {
        let shape = grad_output.shape();
        assert_eq!(shape.len(), 3, "InterleavedRoPE backward expects 3D");
        let seq_len = shape[0];
        let n_heads = shape[1];
        let head_dim = shape[2];
        assert_eq!(head_dim, self.head_dim);
        let half_dim = head_dim / 2;

        let data = dev.download(&grad_output.buffer);
        let mut out = vec![0.0f32; data.len()];

        for s in 0..seq_len {
            let pos = start_pos + s;
            for h in 0..n_heads {
                let base_idx = (s * n_heads + h) * head_dim;
                for i in 0..half_dim {
                    let cos = self.cos_table[pos * half_dim + i];
                    let sin = self.sin_table[pos * half_dim + i];
                    let g0 = data[base_idx + 2 * i];
                    let g1 = data[base_idx + 2 * i + 1];
                    // Transpose of rotation matrix
                    out[base_idx + 2 * i] = g0 * cos + g1 * sin;
                    out[base_idx + 2 * i + 1] = -g0 * sin + g1 * cos;
                }
            }
        }

        ComputeTensor::from_data(dev, &out, shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuDevice;

    #[test]
    fn linear_forward() {
        let dev = CpuDevice::new();
        // W = [[1,0],[0,1],[1,1]], b = [0.1, 0.2, 0.3]
        // in=2, out=3
        let lin = Linear::new(&dev, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[0.1, 0.2, 0.3], 2, 3);
        let x = ComputeTensor::from_data(&dev, &[2.0, 3.0], &[1, 2]);
        let y = lin.forward_2d(&dev, &x);
        let v = y.to_vec();
        // [2,3] @ [[1,0,1],[0,1,1]] + [0.1,0.2,0.3] = [2.1, 3.2, 5.3]
        assert!((v[0] - 2.1).abs() < 1e-4);
        assert!((v[1] - 3.2).abs() < 1e-4);
        assert!((v[2] - 5.3).abs() < 1e-4);
    }

    #[test]
    fn linear_batched() {
        let dev = CpuDevice::new();
        // Identity 2x2, zero bias
        let lin = Linear::new(&dev, &[1.0, 0.0, 0.0, 1.0], &[0.0, 0.0], 2, 2);
        let x = ComputeTensor::from_data(&dev, &[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let y = lin.forward_2d(&dev, &x);
        let v = y.to_vec();
        assert!((v[0] - 1.0).abs() < 1e-5);
        assert!((v[1] - 2.0).abs() < 1e-5);
        assert!((v[2] - 3.0).abs() < 1e-5);
        assert!((v[3] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn rms_norm_forward() {
        let dev = CpuDevice::new();
        let norm = RMSNorm::new(&dev, 3, 1e-5);
        let x = ComputeTensor::from_data(&dev, &[1.0, 2.0, 3.0], &[1, 3]);
        let y = norm.forward(&dev, &x);
        let v = y.to_vec();
        // rms = sqrt((1+4+9)/3) = sqrt(14/3) ≈ 2.1602
        // normalized: [1/2.1602, 2/2.1602, 3/2.1602] ≈ [0.4629, 0.9258, 1.3887]
        assert!((v[0] - 0.4629).abs() < 1e-3);
        assert!((v[1] - 0.9258).abs() < 1e-3);
        assert!((v[2] - 1.3887).abs() < 1e-3);
    }

    #[test]
    fn embedding_lookup() {
        let dev = CpuDevice::new();
        // vocab=3, dim=2, weights: [[0.1,0.2], [0.3,0.4], [0.5,0.6]]
        let emb = Embedding::new(&dev, &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 3, 2);
        let ids = dev.upload_u32(&[2, 0]);
        let out = emb.forward(&dev, &ids, 2);
        let v = out.to_vec();
        // token 2 → [0.5, 0.6], token 0 → [0.1, 0.2]
        assert!((v[0] - 0.5).abs() < 1e-5);
        assert!((v[1] - 0.6).abs() < 1e-5);
        assert!((v[2] - 0.1).abs() < 1e-5);
        assert!((v[3] - 0.2).abs() < 1e-5);
    }

    #[test]
    fn kv_cache_append_and_retrieve() {
        let dev = CpuDevice::new();
        let mut kv = KVCache::new(2, 4, 16); // 2 heads, dim=4, max=16
        assert_eq!(kv.len, 0);

        // Append 1 position: [1, 2*4] = [1, 8]
        let k1 = vec![1.0; 8];
        let v1 = vec![2.0; 8];
        kv.append(&k1, &v1);
        assert_eq!(kv.len, 1);

        let keys = kv.get_keys_buffer::<CpuDevice>(&dev);
        assert_eq!(keys.to_vec(), vec![1.0; 8]);

        // Append another position
        let k2 = vec![3.0; 8];
        let v2 = vec![4.0; 8];
        kv.append(&k2, &v2);
        assert_eq!(kv.len, 2);

        let vals = kv.get_values_buffer::<CpuDevice>(&dev);
        let v = vals.to_vec();
        assert_eq!(&v[..8], &[2.0; 8]);
        assert_eq!(&v[8..], &[4.0; 8]);

        kv.clear();
        assert_eq!(kv.len, 0);
    }

    #[test]
    fn rope_identity_at_pos_zero() {
        let dev = CpuDevice::new();
        let rope = InterleavedRoPE::new(4, 32, 10000.0);
        // At position 0, cos=1, sin=0 for all frequencies → identity
        let input = ComputeTensor::from_data(&dev, &[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let out = rope.forward(&dev, &input, 0);
        let v = out.to_vec();
        assert!((v[0] - 1.0).abs() < 1e-5);
        assert!((v[1] - 2.0).abs() < 1e-5);
        assert!((v[2] - 3.0).abs() < 1e-5);
        assert!((v[3] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn linear_backward_grad_shape() {
        let dev = CpuDevice::new();
        let lin = Linear::new(&dev, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[0.1, 0.2, 0.3], 2, 3);
        let x = ComputeTensor::from_data(&dev, &[2.0, 3.0, 1.0, 4.0], &[2, 2]);
        let (out, cache) = lin.forward_2d_train(&dev, &x);
        assert_eq!(out.shape(), &[2, 3]);

        let grad_out = ComputeTensor::from_data(&dev, &[1.0; 6], &[2, 3]);
        let (gi, gw, gb) = lin.backward_2d(&dev, &grad_out, &cache);
        assert_eq!(gi.shape(), &[2, 2]);
        assert_eq!(gw.len(), 6); // [3, 2]
        assert_eq!(gb.len(), 3);
    }

    #[test]
    fn rms_norm_backward_shapes() {
        let dev = CpuDevice::new();
        let norm = RMSNorm::new(&dev, 3, 1e-5);
        let x = ComputeTensor::from_data(&dev, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let (out, cache) = norm.forward_train(&dev, &x);
        assert_eq!(out.shape(), &[2, 3]);

        let grad_out = ComputeTensor::from_data(&dev, &[1.0; 6], &[2, 3]);
        let (gi, gw) = norm.backward(&dev, &grad_out, &cache);
        assert_eq!(gi.shape(), &[2, 3]);
        assert_eq!(gw.len(), 3);
    }

    #[test]
    fn embedding_backward_shapes() {
        let dev = CpuDevice::new();
        let emb = Embedding::new(&dev, &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 3, 2);
        let ids = dev.upload_u32(&[2, 0]);
        let (out, cache) = emb.forward_train(&dev, &ids, 2);
        assert_eq!(out.shape(), &[2, 2]);

        let grad_out = ComputeTensor::from_data(&dev, &[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let gw = emb.backward(&dev, &grad_out, &cache);
        assert_eq!(gw.len(), 6); // vocab=3, dim=2
    }

    #[test]
    fn rope_backward_roundtrip() {
        let dev = CpuDevice::new();
        let rope = InterleavedRoPE::new(4, 32, 10000.0);
        let input = ComputeTensor::from_data(&dev, &[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let fwd = rope.forward(&dev, &input, 3);
        let bwd = rope.backward(&dev, &fwd, 3);
        // backward(forward(x)) should ≈ x (RoPE is orthogonal)
        let v = bwd.to_vec();
        assert!((v[0] - 1.0).abs() < 1e-4);
        assert!((v[1] - 2.0).abs() < 1e-4);
        assert!((v[2] - 3.0).abs() < 1e-4);
        assert!((v[3] - 4.0).abs() < 1e-4);
    }

    #[test]
    fn rope_nonzero_pos_rotates() {
        let dev = CpuDevice::new();
        let rope = InterleavedRoPE::new(4, 32, 10000.0);
        let input = ComputeTensor::from_data(&dev, &[1.0, 0.0, 1.0, 0.0], &[1, 1, 4]);
        let out = rope.forward(&dev, &input, 5);
        let v = out.to_vec();
        // At position 5, rotation should change the values
        assert!((v[0] - 1.0).abs() > 1e-3 || (v[1]).abs() > 1e-3);
    }
}
