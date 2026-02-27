use alloc::vec::Vec;
use tang::Scalar;
use tang_tensor::{Shape, Tensor};

/// Per-layer key/value cache for autoregressive inference.
///
/// Stores K and V tensors for each layer, growing as new tokens are generated.
/// This avoids recomputing attention over the entire sequence at each step,
/// reducing generation from O(n²) to O(n) per token.
///
/// Layout: K/V are stored as `[seq_len, kv_dim]` where `kv_dim = num_kv_heads * head_dim`.
pub struct KVCache<S: Scalar> {
    /// Key cache per layer: `[seq_len, kv_dim]` (grows along seq_len)
    keys: Vec<Vec<S>>,
    /// Value cache per layer: `[seq_len, kv_dim]` (grows along seq_len)
    values: Vec<Vec<S>>,
    /// Current sequence length (same across all layers)
    seq_len: usize,
    /// Number of layers
    num_layers: usize,
    /// KV dimension per token (num_kv_heads * head_dim)
    kv_dim: usize,
    /// Maximum sequence length (pre-allocated capacity)
    max_seq_len: usize,
}

impl<S: Scalar> KVCache<S> {
    /// Create a new empty KV cache.
    ///
    /// - `num_layers`: number of transformer layers
    /// - `max_seq_len`: maximum sequence length (for capacity pre-allocation)
    /// - `num_kv_heads`: number of key/value heads
    /// - `head_dim`: dimension per head
    pub fn new(num_layers: usize, max_seq_len: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let kv_dim = num_kv_heads * head_dim;
        let cap = max_seq_len * kv_dim;
        let mut keys = Vec::with_capacity(num_layers);
        let mut values = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            keys.push(Vec::with_capacity(cap));
            values.push(Vec::with_capacity(cap));
        }
        Self {
            keys,
            values,
            seq_len: 0,
            num_layers,
            kv_dim,
            max_seq_len,
        }
    }

    /// Current cached sequence length.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Number of layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// KV dimension (num_kv_heads * head_dim).
    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    /// Append new key/value for a single token (or multiple tokens) at a given layer.
    ///
    /// - `layer`: layer index
    /// - `new_k`: `[num_new_tokens, kv_dim]`
    /// - `new_v`: `[num_new_tokens, kv_dim]`
    ///
    /// On the first call (prefill), `num_new_tokens` can be the full prompt length.
    /// On subsequent calls (generation), `num_new_tokens` is typically 1.
    pub fn append(&mut self, layer: usize, new_k: &Tensor<S>, new_v: &Tensor<S>) {
        assert!(layer < self.num_layers, "layer index out of bounds");
        assert_eq!(new_k.ndim(), 2);
        assert_eq!(new_v.ndim(), 2);
        let num_new = new_k.shape()[0];
        assert_eq!(new_k.shape()[1], self.kv_dim);
        assert_eq!(new_v.shape()[0], num_new);
        assert_eq!(new_v.shape()[1], self.kv_dim);
        assert!(
            self.seq_len + num_new <= self.max_seq_len,
            "KV cache overflow: {} + {} > {}",
            self.seq_len,
            num_new,
            self.max_seq_len
        );

        self.keys[layer].extend_from_slice(new_k.data());
        self.values[layer].extend_from_slice(new_v.data());

        // Update seq_len only after the last layer appends
        // (caller must append to all layers for the same tokens)
        if layer == self.num_layers - 1 {
            self.seq_len += num_new;
        }
    }

    /// Get the full cached key tensor for a layer: `[seq_len, kv_dim]`.
    pub fn get_keys(&self, layer: usize) -> Tensor<S> {
        assert!(layer < self.num_layers);
        let len = self.keys[layer].len() / self.kv_dim;
        Tensor::new(
            self.keys[layer].clone(),
            Shape::from_slice(&[len, self.kv_dim]),
        )
    }

    /// Get the full cached value tensor for a layer: `[seq_len, kv_dim]`.
    pub fn get_values(&self, layer: usize) -> Tensor<S> {
        assert!(layer < self.num_layers);
        let len = self.values[layer].len() / self.kv_dim;
        Tensor::new(
            self.values[layer].clone(),
            Shape::from_slice(&[len, self.kv_dim]),
        )
    }

    /// Get both K and V for a layer.
    pub fn get(&self, layer: usize) -> (Tensor<S>, Tensor<S>) {
        (self.get_keys(layer), self.get_values(layer))
    }

    /// Clear the cache (reset to empty).
    pub fn clear(&mut self) {
        for layer in 0..self.num_layers {
            self.keys[layer].clear();
            self.values[layer].clear();
        }
        self.seq_len = 0;
    }

    /// Trim the cache to keep only the last `keep` tokens.
    /// Useful for sliding window attention or context length management.
    pub fn trim_to(&mut self, keep: usize) {
        if self.seq_len <= keep {
            return;
        }
        let drop_tokens = self.seq_len - keep;
        let drop_elems = drop_tokens * self.kv_dim;
        for layer in 0..self.num_layers {
            self.keys[layer].drain(..drop_elems);
            self.values[layer].drain(..drop_elems);
        }
        self.seq_len = keep;
    }

    /// Get raw key data slice for a layer (avoids clone for read-only access).
    pub fn keys_data(&self, layer: usize) -> &[S] {
        &self.keys[layer]
    }

    /// Get raw value data slice for a layer (avoids clone for read-only access).
    pub fn values_data(&self, layer: usize) -> &[S] {
        &self.values[layer]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn kv_cache_basic() {
        let mut cache = KVCache::<f64>::new(2, 16, 4, 8); // 2 layers, max 16 tokens, 4 heads, dim 8
        assert_eq!(cache.seq_len(), 0);
        assert_eq!(cache.kv_dim(), 32); // 4 * 8

        // Append 3 tokens to layer 0
        let k = Tensor::new(vec![1.0; 3 * 32], Shape::from_slice(&[3, 32]));
        let v = Tensor::new(vec![2.0; 3 * 32], Shape::from_slice(&[3, 32]));
        cache.append(0, &k, &v);
        // seq_len only updates after last layer
        assert_eq!(cache.seq_len(), 0);

        // Append 3 tokens to layer 1 (last layer)
        cache.append(1, &k, &v);
        assert_eq!(cache.seq_len(), 3);

        // Retrieve
        let (k_out, v_out) = cache.get(0);
        assert_eq!(k_out.shape().dims(), &[3, 32]);
        assert_eq!(v_out.shape().dims(), &[3, 32]);
        assert_eq!(k_out.get(&[0, 0]), 1.0);
        assert_eq!(v_out.get(&[0, 0]), 2.0);
    }

    #[test]
    fn kv_cache_incremental() {
        let mut cache = KVCache::<f64>::new(1, 16, 2, 4);
        let kv_dim = 8;

        // Prefill: 3 tokens
        let k3 = Tensor::new(vec![1.0; 3 * kv_dim], Shape::from_slice(&[3, kv_dim]));
        let v3 = Tensor::new(vec![1.0; 3 * kv_dim], Shape::from_slice(&[3, kv_dim]));
        cache.append(0, &k3, &v3);
        assert_eq!(cache.seq_len(), 3);

        // Generate: 1 token at a time
        let k1 = Tensor::new(vec![2.0; kv_dim], Shape::from_slice(&[1, kv_dim]));
        let v1 = Tensor::new(vec![2.0; kv_dim], Shape::from_slice(&[1, kv_dim]));
        cache.append(0, &k1, &v1);
        assert_eq!(cache.seq_len(), 4);

        let (k_full, _) = cache.get(0);
        assert_eq!(k_full.shape().dims(), &[4, 8]);
        // First 3 tokens have value 1.0, last has 2.0
        assert_eq!(k_full.get(&[2, 0]), 1.0);
        assert_eq!(k_full.get(&[3, 0]), 2.0);
    }

    #[test]
    fn kv_cache_clear() {
        let mut cache = KVCache::<f64>::new(1, 8, 1, 4);
        let k = Tensor::new(vec![1.0; 4], Shape::from_slice(&[1, 4]));
        let v = Tensor::new(vec![1.0; 4], Shape::from_slice(&[1, 4]));
        cache.append(0, &k, &v);
        assert_eq!(cache.seq_len(), 1);

        cache.clear();
        assert_eq!(cache.seq_len(), 0);
        assert!(cache.keys_data(0).is_empty());
    }

    #[test]
    fn kv_cache_trim() {
        let mut cache = KVCache::<f64>::new(1, 16, 1, 4);
        let kv_dim = 4;

        // Add 5 tokens with values 0..4
        for i in 0..5 {
            let k = Tensor::new(vec![i as f64; kv_dim], Shape::from_slice(&[1, kv_dim]));
            let v = Tensor::new(vec![i as f64; kv_dim], Shape::from_slice(&[1, kv_dim]));
            cache.append(0, &k, &v);
        }
        assert_eq!(cache.seq_len(), 5);

        // Keep last 3
        cache.trim_to(3);
        assert_eq!(cache.seq_len(), 3);

        let (k, _) = cache.get(0);
        assert_eq!(k.shape().dims(), &[3, 4]);
        // Should have tokens 2, 3, 4
        assert_eq!(k.get(&[0, 0]), 2.0);
        assert_eq!(k.get(&[1, 0]), 3.0);
        assert_eq!(k.get(&[2, 0]), 4.0);
    }
}
