//! Erasure-coded model distribution (Coded Mesh).
//!
//! Splits model weights into `k` blocks, encodes via a Cauchy generator matrix
//! so any `k`-of-`n` nodes reconstruct the full model. Each node stores one
//! coded shard — `model_size / k` floats.
//!
//! # Encoding
//!
//! ```text
//! Node 0: G[0,:] · W     ─┐
//! Node 1: G[1,:] · W      │  any k decode
//! Node 2: G[2,:] · W      ├──────────────► full W
//!   ...                    │
//! Node n: G[n,:] · W     ─┘
//! ```
//!
//! # Inference
//!
//! Linear layers: each of k nodes computes `shard @ x` (coded computation).
//! Non-linearities: decode → apply activation → re-encode.
//!
//! # Learning
//!
//! Every inference includes a backward pass. Block gradients are encoded the
//! same way as weights: `δshard_i = G[i,:] · δW`. Coding structure preserved.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Shard
// ---------------------------------------------------------------------------

/// A coded weight block stored on a single node.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Shard {
    /// Coded weight data: `G[i,:] · W` — a linear combination of the k blocks.
    pub data: Vec<f32>,
    /// This node's row of the generator matrix (k floats).
    pub code_row: Vec<f32>,
    /// How many gradient steps have been applied.
    pub version: u64,
}

impl Shard {
    /// Apply a coded gradient update: `data -= lr * coded_grad`.
    pub fn apply_update(&mut self, coded_grad: &[f32], lr: f32) {
        assert_eq!(self.data.len(), coded_grad.len());
        for (w, &g) in self.data.iter_mut().zip(coded_grad.iter()) {
            *w -= lr * g;
        }
        self.version += 1;
    }

    /// Coded linear forward: `shard @ x` where shard is (block_size × d_in)
    /// stored row-major, and x is (d_in,). Returns (block_size,) partial output.
    ///
    /// The full output is reconstructed by decoding k such partial outputs.
    pub fn forward_linear(&self, x: &[f32], d_in: usize) -> Vec<f32> {
        assert!(d_in > 0);
        let block_size = self.data.len() / d_in;
        assert_eq!(self.data.len(), block_size * d_in);
        let mut out = vec![0.0f32; block_size];
        for row in 0..block_size {
            let mut sum = 0.0f32;
            let offset = row * d_in;
            for col in 0..d_in {
                sum += self.data[offset + col] * x[col];
            }
            out[row] = sum;
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Generator (Cauchy matrix)
// ---------------------------------------------------------------------------

/// Cauchy generator matrix (n × k). Any k×k submatrix is invertible.
///
/// `G[i,j] = 1 / (x[i] - y[j])` where x and y are chosen so all
/// differences are nonzero (x[i] = i, y[j] = n + j).
#[derive(Clone, Debug)]
pub struct Generator {
    /// Number of total nodes.
    pub n: usize,
    /// Number of blocks (minimum nodes to reconstruct).
    pub k: usize,
    /// Row-major n×k matrix entries.
    data: Vec<f32>,
}

impl Generator {
    /// Create a Cauchy generator matrix for n nodes, k blocks.
    ///
    /// Uses `G[i,j] = 1 / (x[i] - y[j])` with x[i] = i+1, y[j] = -(j+1).
    /// This ensures x[i] != y[j] for all i,j, guaranteeing every k×k
    /// submatrix is invertible (Cauchy matrix property).
    pub fn cauchy(n: usize, k: usize) -> Self {
        assert!(k > 0 && n >= k, "need n >= k > 0");
        let mut data = Vec::with_capacity(n * k);
        for i in 0..n {
            let xi = (i + 1) as f32;
            for j in 0..k {
                let yj = -((j + 1) as f32);
                data.push(1.0 / (xi - yj));
            }
        }
        Self { n, k, data }
    }

    /// Get element G[i, j].
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f32 {
        self.data[i * self.k + j]
    }

    /// Get row i as a slice (k elements).
    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.k..(i + 1) * self.k]
    }

    /// Encode k blocks into n shards.
    ///
    /// `blocks[j]` is block j (all same length). Returns n shards, one per node.
    pub fn encode(&self, blocks: &[&[f32]]) -> Vec<Shard> {
        assert_eq!(blocks.len(), self.k);
        let block_len = blocks[0].len();
        for b in blocks {
            assert_eq!(b.len(), block_len);
        }

        let mut shards = Vec::with_capacity(self.n);
        for i in 0..self.n {
            let mut data = vec![0.0f32; block_len];
            for j in 0..self.k {
                let g = self.get(i, j);
                for (d, &b) in data.iter_mut().zip(blocks[j].iter()) {
                    *d += g * b;
                }
            }
            shards.push(Shard {
                data,
                code_row: self.row(i).to_vec(),
                version: 0,
            });
        }
        shards
    }

    /// Decode original k blocks from any k shards.
    ///
    /// `shards` is a slice of (node_index, &Shard) pairs. Must have exactly k entries.
    /// Returns k blocks in order.
    pub fn decode(&self, shards: &[(usize, &Shard)]) -> Vec<Vec<f32>> {
        assert_eq!(shards.len(), self.k);
        let block_len = shards[0].1.data.len();

        // Build k×k submatrix from the selected rows of G
        let mut sub = vec![0.0f32; self.k * self.k];
        for (row, &(node_idx, _)) in shards.iter().enumerate() {
            for col in 0..self.k {
                sub[row * self.k + col] = self.get(node_idx, col);
            }
        }

        // Invert the k×k matrix
        let inv = invert_matrix(&sub, self.k);

        // Multiply inv × coded_data to recover original blocks
        let mut blocks = vec![vec![0.0f32; block_len]; self.k];
        for j in 0..self.k {
            for (row, &(_, shard)) in shards.iter().enumerate() {
                let coeff = inv[j * self.k + row];
                for (d, &s) in blocks[j].iter_mut().zip(shard.data.iter()) {
                    *d += coeff * s;
                }
            }
        }
        blocks
    }

    /// Encode a gradient update for a specific node.
    ///
    /// Given k block gradients `δW_1..δW_k`, produces the coded update
    /// for node `node_idx`: `G[node_idx, :] · δW`.
    pub fn encode_update(&self, node_idx: usize, block_grads: &[&[f32]]) -> Vec<f32> {
        assert_eq!(block_grads.len(), self.k);
        let block_len = block_grads[0].len();
        let mut coded = vec![0.0f32; block_len];
        for j in 0..self.k {
            let g = self.get(node_idx, j);
            for (c, &bg) in coded.iter_mut().zip(block_grads[j].iter()) {
                *c += g * bg;
            }
        }
        coded
    }
}

/// Invert an n×n matrix (row-major) via Gauss-Jordan elimination.
/// Panics if singular.
fn invert_matrix(m: &[f32], n: usize) -> Vec<f32> {
    assert_eq!(m.len(), n * n);
    // Augmented matrix [M | I]
    let mut aug = vec![0.0f32; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            aug[i * 2 * n + j] = m[i * n + j];
        }
        aug[i * 2 * n + n + i] = 1.0;
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = aug[col * 2 * n + col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let v = aug[row * 2 * n + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        assert!(max_val > 1e-10, "singular matrix in invert_matrix");

        // Swap rows
        if max_row != col {
            for j in 0..2 * n {
                aug.swap(col * 2 * n + j, max_row * 2 * n + j);
            }
        }

        // Scale pivot row
        let pivot = aug[col * 2 * n + col];
        for j in 0..2 * n {
            aug[col * 2 * n + j] /= pivot;
        }

        // Eliminate column
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row * 2 * n + col];
            for j in 0..2 * n {
                aug[row * 2 * n + j] -= factor * aug[col * 2 * n + j];
            }
        }
    }

    // Extract inverse
    let mut inv = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * 2 * n + n + j];
        }
    }
    inv
}

// ---------------------------------------------------------------------------
// Gradient policy
// ---------------------------------------------------------------------------

/// Gradient hygiene: clip, decay, compress.
#[derive(Clone, Debug)]
pub struct GradientPolicy {
    /// Reject updates beyond N sigma of running norm (default: 3.0).
    pub clip_sigma: f32,
    /// Initial learning rate (default: 1e-4).
    pub lr_base: f32,
    /// Per-version decay factor (default: 0.99999).
    pub lr_decay: f32,
    /// Sparse gossip compression ratio (default: 0.001 = top 0.1%).
    pub top_k_ratio: f32,
    /// Running gradient norm mean (EMA).
    running_mean: f32,
    /// Running gradient norm variance (EMA).
    running_var: f32,
}

impl Default for GradientPolicy {
    fn default() -> Self {
        Self {
            clip_sigma: 3.0,
            lr_base: 1e-4,
            lr_decay: 0.99999,
            top_k_ratio: 0.001,
            running_mean: 0.0,
            running_var: 1.0,
        }
    }
}

impl GradientPolicy {
    /// Effective learning rate at the given version.
    pub fn lr_at(&self, version: u64) -> f32 {
        self.lr_base * self.lr_decay.powi(version as i32)
    }

    /// Returns true if the gradient norm is within the acceptable range.
    /// Updates running statistics (EMA with α=0.01).
    pub fn clip_check(&mut self, grad: &[f32]) -> bool {
        let norm: f32 = grad.iter().map(|x| x * x).sum::<f32>().sqrt();
        let threshold = self.running_mean + self.clip_sigma * self.running_var.sqrt();

        // Update EMA (before reject check, so stats track even rejected grads)
        let alpha = 0.01f32;
        self.running_mean = (1.0 - alpha) * self.running_mean + alpha * norm;
        let diff = norm - self.running_mean;
        self.running_var = (1.0 - alpha) * self.running_var + alpha * diff * diff;

        // For the first few updates, accept everything (cold start)
        if self.running_mean < 1e-10 {
            return true;
        }

        norm <= threshold
    }
}

// ---------------------------------------------------------------------------
// Compressed gradient (for gossip)
// ---------------------------------------------------------------------------

/// Sparse int8-compressed gradient for bandwidth-efficient gossip.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompressedGrad {
    /// Indices of the top-k gradient entries.
    pub indices: Vec<u32>,
    /// Quantized values (int8 mapped to [-scale, +scale]).
    pub values: Vec<i8>,
    /// Scale factor: max absolute value in the selected entries.
    pub scale: f32,
    /// Total length of the original gradient vector.
    pub original_len: u32,
}

impl CompressedGrad {
    /// Compress a gradient vector: keep top-k entries, quantize to int8.
    pub fn compress(grad: &[f32], top_k_ratio: f32) -> Self {
        let k = ((grad.len() as f32 * top_k_ratio).ceil() as usize).max(1);

        // Find top-k by absolute value
        let mut indexed: Vec<(usize, f32)> = grad.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        indexed.truncate(k);

        let scale = indexed
            .iter()
            .map(|(_, v)| v.abs())
            .fold(0.0f32, f32::max);

        let (indices, values): (Vec<u32>, Vec<i8>) = if scale < 1e-30 {
            indexed
                .iter()
                .map(|&(i, _)| (i as u32, 0i8))
                .unzip()
        } else {
            indexed
                .iter()
                .map(|&(i, v)| {
                    let q = (v / scale * 127.0).round().clamp(-127.0, 127.0) as i8;
                    (i as u32, q)
                })
                .unzip()
        };

        Self {
            indices,
            values,
            scale,
            original_len: grad.len() as u32,
        }
    }

    /// Decompress back to a dense gradient vector.
    pub fn decompress(&self) -> Vec<f32> {
        let mut grad = vec![0.0f32; self.original_len as usize];
        for (&idx, &val) in self.indices.iter().zip(self.values.iter()) {
            grad[idx as usize] = val as f32 / 127.0 * self.scale;
        }
        grad
    }
}

// ---------------------------------------------------------------------------
// CodedModel — per-node model state
// ---------------------------------------------------------------------------

/// Manages a single node's participation in the coded mesh.
pub struct CodedModel {
    /// This node's coded shard.
    pub shard: Shard,
    /// The generator matrix (shared by all nodes).
    pub generator: Generator,
    /// Gradient hygiene policy.
    pub policy: GradientPolicy,
    /// This node's index in [0, n).
    pub node_index: usize,
}

impl CodedModel {
    /// Create a coded model from full (uncoded) weights.
    ///
    /// Splits weights into k equal blocks, encodes, and returns the model
    /// holding only this node's shard.
    pub fn from_weights(
        weights: &[f32],
        generator: &Generator,
        node_index: usize,
    ) -> Self {
        let k = generator.k;
        let block_len = (weights.len() + k - 1) / k;
        // Pad to block_len * k
        let mut padded = weights.to_vec();
        padded.resize(block_len * k, 0.0);

        let blocks: Vec<&[f32]> = (0..k)
            .map(|j| &padded[j * block_len..(j + 1) * block_len])
            .collect();

        let shards = generator.encode(&blocks);

        Self {
            shard: shards[node_index].clone(),
            generator: generator.clone(),
            policy: GradientPolicy::default(),
            node_index,
        }
    }

    /// Apply a coded gradient update with clip + decay.
    ///
    /// Returns false if the update was rejected (clipped).
    pub fn apply_coded_update(&mut self, coded_grad: &[f32]) -> bool {
        if !self.policy.clip_check(coded_grad) {
            return false;
        }
        let lr = self.policy.lr_at(self.shard.version);
        self.shard.apply_update(coded_grad, lr);
        true
    }

    /// Compress a gradient for gossip transmission.
    pub fn compress_for_gossip(&self, grad: &[f32]) -> CompressedGrad {
        CompressedGrad::compress(grad, self.policy.top_k_ratio)
    }
}

// ---------------------------------------------------------------------------
// Decode helpers (for CodedInferenceServer)
// ---------------------------------------------------------------------------

/// Decode k coded output vectors back to k original blocks.
///
/// `coded_outputs[i] = (node_index, output_vec)` — partial outputs from k nodes.
/// `generator` — the shared generator matrix.
///
/// Returns the reconstructed full output (all k blocks concatenated).
pub fn decode_outputs(
    generator: &Generator,
    coded_outputs: &[(usize, &[f32])],
) -> Vec<f32> {
    let k = generator.k;
    assert_eq!(coded_outputs.len(), k);
    let block_len = coded_outputs[0].1.len();

    // Build k×k submatrix
    let mut sub = vec![0.0f32; k * k];
    for (row, &(node_idx, _)) in coded_outputs.iter().enumerate() {
        for col in 0..k {
            sub[row * k + col] = generator.get(node_idx, col);
        }
    }
    let inv = invert_matrix(&sub, k);

    // Decode: block_j = sum_i inv[j,i] * coded_output_i
    let mut full = vec![0.0f32; k * block_len];
    for j in 0..k {
        for (i, &(_, output)) in coded_outputs.iter().enumerate() {
            let coeff = inv[j * k + i];
            for (idx, &val) in output.iter().enumerate() {
                full[j * block_len + idx] += coeff * val;
            }
        }
    }
    full
}

/// Re-encode a full vector back into k coded vectors.
///
/// Splits `full` into k blocks and returns n coded vectors (one per node).
pub fn encode_outputs(generator: &Generator, full: &[f32]) -> Vec<Vec<f32>> {
    let k = generator.k;
    let block_len = (full.len() + k - 1) / k;
    let mut padded = full.to_vec();
    padded.resize(block_len * k, 0.0);

    let mut coded = Vec::with_capacity(generator.n);
    for i in 0..generator.n {
        let mut out = vec![0.0f32; block_len];
        for j in 0..k {
            let g = generator.get(i, j);
            for (o, &b) in out.iter_mut().zip(padded[j * block_len..].iter()) {
                *o += g * b;
            }
        }
        coded.push(out);
    }
    coded
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Cauchy correctness ---

    #[test]
    fn cauchy_submatrix_invertible() {
        // For n=5, k=3: every 3×3 submatrix of G must be invertible
        let g = Generator::cauchy(5, 3);
        let n = g.n;
        let k = g.k;

        // Test all C(5,3)=10 subsets
        let mut count = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                for l in (j + 1)..n {
                    let mut sub = vec![0.0f32; k * k];
                    for (row, &node) in [i, j, l].iter().enumerate() {
                        for col in 0..k {
                            sub[row * k + col] = g.get(node, col);
                        }
                    }
                    // Should not panic
                    let inv = invert_matrix(&sub, k);
                    // Verify M * M^-1 ≈ I (f32 precision ~1e-3 for 3×3 Cauchy)
                    for r in 0..k {
                        for c in 0..k {
                            let mut dot = 0.0f32;
                            for x in 0..k {
                                dot += sub[r * k + x] * inv[x * k + c];
                            }
                            let expected = if r == c { 1.0 } else { 0.0 };
                            assert!(
                                (dot - expected).abs() < 1e-3,
                                "M*M^-1 [{r},{c}] = {dot}, expected {expected}"
                            );
                        }
                    }
                    count += 1;
                }
            }
        }
        assert_eq!(count, 10);
    }

    #[test]
    fn cauchy_4_2_all_submatrices() {
        let g = Generator::cauchy(4, 2);
        for i in 0..4 {
            for j in (i + 1)..4 {
                let mut sub = vec![0.0f32; 4];
                sub[0] = g.get(i, 0);
                sub[1] = g.get(i, 1);
                sub[2] = g.get(j, 0);
                sub[3] = g.get(j, 1);
                let _inv = invert_matrix(&sub, 2); // panics if singular
            }
        }
    }

    // --- Encode/decode roundtrip ---

    #[test]
    fn encode_decode_roundtrip() {
        let g = Generator::cauchy(5, 3);
        let b0: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let b1: Vec<f32> = (10..20).map(|i| i as f32).collect();
        let b2: Vec<f32> = (20..30).map(|i| i as f32).collect();

        let shards = g.encode(&[&b0, &b1, &b2]);
        assert_eq!(shards.len(), 5);

        // Decode from nodes 0,2,4
        let decoded = g.decode(&[
            (0, &shards[0]),
            (2, &shards[2]),
            (4, &shards[4]),
        ]);
        assert_eq!(decoded.len(), 3);
        for (got, expected) in decoded[0].iter().zip(b0.iter()) {
            assert!((got - expected).abs() < 1e-3, "b0: {got} vs {expected}");
        }
        for (got, expected) in decoded[1].iter().zip(b1.iter()) {
            assert!((got - expected).abs() < 1e-3, "b1: {got} vs {expected}");
        }
        for (got, expected) in decoded[2].iter().zip(b2.iter()) {
            assert!((got - expected).abs() < 1e-3, "b2: {got} vs {expected}");
        }
    }

    #[test]
    fn encode_decode_any_k_subset() {
        let g = Generator::cauchy(5, 3);
        let blocks: Vec<Vec<f32>> = (0..3)
            .map(|i| (0..8).map(|j| (i * 8 + j) as f32 * 0.1).collect())
            .collect();
        let block_refs: Vec<&[f32]> = blocks.iter().map(|b| b.as_slice()).collect();
        let shards = g.encode(&block_refs);

        // Test all C(5,3)=10 decode subsets
        for i in 0..5 {
            for j in (i + 1)..5 {
                for l in (j + 1)..5 {
                    let decoded = g.decode(&[
                        (i, &shards[i]),
                        (j, &shards[j]),
                        (l, &shards[l]),
                    ]);
                    for (bi, block) in blocks.iter().enumerate() {
                        for (idx, (&got, &exp)) in
                            decoded[bi].iter().zip(block.iter()).enumerate()
                        {
                            assert!(
                                (got - exp).abs() < 1e-3,
                                "subset [{i},{j},{l}] block {bi}[{idx}]: {got} vs {exp}"
                            );
                        }
                    }
                }
            }
        }
    }

    // --- Coded update preserves coding structure ---

    #[test]
    fn coded_update_matches_uncoded_sgd() {
        let g = Generator::cauchy(4, 2);
        let weights: Vec<f32> = (0..20).map(|i| i as f32 * 0.1).collect();
        let block_len = 10;
        let b0 = &weights[..block_len];
        let b1 = &weights[block_len..];

        let mut shards = g.encode(&[b0, b1]);
        let lr = 0.01f32;

        // Uncoded SGD step
        let grad0: Vec<f32> = (0..block_len).map(|i| i as f32 * 0.01).collect();
        let grad1: Vec<f32> = (0..block_len).map(|i| -(i as f32) * 0.01).collect();
        let expected_b0: Vec<f32> = b0.iter().zip(&grad0).map(|(&w, &g)| w - lr * g).collect();
        let expected_b1: Vec<f32> = b1.iter().zip(&grad1).map(|(&w, &g)| w - lr * g).collect();

        // Apply coded update to each shard
        for (i, shard) in shards.iter_mut().enumerate() {
            let coded_grad = g.encode_update(i, &[&grad0, &grad1]);
            shard.apply_update(&coded_grad, lr);
        }

        // Decode and verify
        let decoded = g.decode(&[
            (0, &shards[0]),
            (1, &shards[1]),
        ]);
        for (got, exp) in decoded[0].iter().zip(expected_b0.iter()) {
            assert!((got - exp).abs() < 1e-4, "b0: {got} vs {exp}");
        }
        for (got, exp) in decoded[1].iter().zip(expected_b1.iter()) {
            assert!((got - exp).abs() < 1e-4, "b1: {got} vs {exp}");
        }
    }

    // --- Coded inference (forward_linear) ---

    #[test]
    fn coded_forward_matches_uncoded() {
        // 2×3 weight matrix W, input x of size 3
        let w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32]; // (2 rows, 3 cols)
        let x = [1.0, 0.5, -1.0f32];
        // Expected: W @ x = [1*1 + 2*0.5 + 3*(-1), 4*1 + 5*0.5 + 6*(-1)] = [-1.0, 0.5]
        let expected = [-1.0f32, 0.5];

        let g = Generator::cauchy(4, 2);
        // Split W into 2 blocks: block0 = first row (3 floats), block1 = second row (3 floats)
        let b0 = &w[0..3];
        let b1 = &w[3..6];
        let shards = g.encode(&[b0, b1]);

        // Each shard does forward_linear (shard @ x), producing 1 coded output per shard
        // d_in = 3, block_size = 3/3 = ... wait, shard.data.len() = 3, d_in = 3 → block_size = 1
        let coded_outputs: Vec<(usize, Vec<f32>)> = (0..2)
            .map(|i| (i, shards[i].forward_linear(&x, 3)))
            .collect();

        // Decode to get full output
        let output_refs: Vec<(usize, &[f32])> = coded_outputs
            .iter()
            .map(|(i, v)| (*i, v.as_slice()))
            .collect();
        let full = decode_outputs(&g, &output_refs);

        assert!(
            (full[0] - expected[0]).abs() < 1e-4,
            "output[0]: {} vs {}",
            full[0],
            expected[0]
        );
        assert!(
            (full[1] - expected[1]).abs() < 1e-4,
            "output[1]: {} vs {}",
            full[1],
            expected[1]
        );
    }

    // --- Fault tolerance ---

    #[test]
    fn survive_n_minus_k_failures() {
        let g = Generator::cauchy(5, 3);
        let blocks: Vec<Vec<f32>> = (0..3)
            .map(|_| (0..16).map(|j| j as f32 * 0.3).collect())
            .collect();
        let block_refs: Vec<&[f32]> = blocks.iter().map(|b| b.as_slice()).collect();
        let shards = g.encode(&block_refs);

        // Kill nodes 1 and 3 (2 failures out of 5, leaving 3 = k)
        let survivors = [(0, &shards[0]), (2, &shards[2]), (4, &shards[4])];
        let decoded = g.decode(&survivors);
        for (bi, block) in blocks.iter().enumerate() {
            for (idx, (&got, &exp)) in decoded[bi].iter().zip(block.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-3,
                    "block {bi}[{idx}]: {got} vs {exp}"
                );
            }
        }
    }

    // --- Gradient policy ---

    #[test]
    fn gradient_policy_lr_decay() {
        let policy = GradientPolicy::default();
        let lr0 = policy.lr_at(0);
        assert!((lr0 - 1e-4).abs() < 1e-10);

        let lr_70k = policy.lr_at(70_000);
        // 0.99999^70000 ≈ e^(-0.7) ≈ 0.4966
        let expected = 1e-4 * 0.99999f32.powi(70_000);
        assert!((lr_70k - expected).abs() < 1e-10);
        // Should be roughly half
        assert!(lr_70k < lr0 * 0.6);
        assert!(lr_70k > lr0 * 0.4);
    }

    #[test]
    fn gradient_policy_clip_rejects_outliers() {
        let mut policy = GradientPolicy {
            clip_sigma: 3.0,
            running_mean: 1.0,
            running_var: 0.01, // std = 0.1
            ..GradientPolicy::default()
        };
        // Threshold = 1.0 + 3.0 * 0.1 = 1.3
        // Normal gradient with norm ~1.0
        let normal = vec![0.5, 0.5, 0.5, 0.5]; // norm = 1.0
        assert!(policy.clip_check(&normal));

        // Reset stats for clean test
        policy.running_mean = 1.0;
        policy.running_var = 0.01;

        // Outlier gradient with norm ~10.0
        let outlier = vec![5.0, 5.0, 5.0, 5.0]; // norm = 10.0
        assert!(!policy.clip_check(&outlier));
    }

    // --- Compression ---

    #[test]
    fn compress_decompress_roundtrip() {
        let grad: Vec<f32> = (0..1000).map(|i| {
            if i < 5 { (i + 1) as f32 * 10.0 } else { i as f32 * 0.001 }
        }).collect();

        let compressed = CompressedGrad::compress(&grad, 0.01); // top 1%
        assert_eq!(compressed.indices.len(), 10); // 1000 * 0.01 = 10

        let decompressed = compressed.decompress();
        assert_eq!(decompressed.len(), 1000);

        // The top entries should be approximately preserved
        for i in 0..5 {
            let original = (i + 1) as f32 * 10.0;
            let recovered = decompressed[i];
            let rel_error = (recovered - original).abs() / original;
            assert!(
                rel_error < 0.02,
                "entry {i}: {recovered} vs {original} (rel_error={rel_error})"
            );
        }
    }

    #[test]
    fn compress_preserves_top_k_count() {
        let grad: Vec<f32> = (0..500).map(|i| i as f32).collect();
        let compressed = CompressedGrad::compress(&grad, 0.001);
        // 500 * 0.001 = 0.5, ceil = 1, max(1) = 1
        assert_eq!(compressed.indices.len(), 1);
        // Should be index 499 (largest value)
        assert_eq!(compressed.indices[0], 499);
    }

    // --- CodedModel ---

    #[test]
    fn coded_model_from_weights_roundtrip() {
        let g = Generator::cauchy(4, 2);
        let weights: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();

        // Create models for all 4 nodes
        let models: Vec<CodedModel> = (0..4)
            .map(|i| CodedModel::from_weights(&weights, &g, i))
            .collect();

        // Any 2 should reconstruct
        let decoded = g.decode(&[
            (1, &models[1].shard),
            (3, &models[3].shard),
        ]);
        let mut reconstructed = decoded[0].clone();
        reconstructed.extend_from_slice(&decoded[1]);
        reconstructed.truncate(weights.len());

        for (i, (&got, &exp)) in reconstructed.iter().zip(weights.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-3, "weight[{i}]: {got} vs {exp}");
        }
    }

    #[test]
    fn coded_model_apply_update_respects_clip() {
        let g = Generator::cauchy(3, 2);
        let weights = vec![1.0f32; 20];
        let mut model = CodedModel::from_weights(&weights, &g, 0);

        // Set tight clip threshold
        model.policy.running_mean = 0.1;
        model.policy.running_var = 0.001;
        model.policy.clip_sigma = 2.0;

        // Huge gradient should be rejected
        let huge_grad = vec![1000.0f32; 10];
        let accepted = model.apply_coded_update(&huge_grad);
        assert!(!accepted);
        assert_eq!(model.shard.version, 0); // not incremented
    }

    // --- Encode/decode outputs ---

    #[test]
    fn encode_decode_outputs_roundtrip() {
        let g = Generator::cauchy(4, 2);
        let full = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32];
        let coded = encode_outputs(&g, &full);
        assert_eq!(coded.len(), 4); // n=4

        // Decode from any 2
        let output_refs = vec![(0, coded[0].as_slice()), (2, coded[2].as_slice())];
        let reconstructed = decode_outputs(&g, &output_refs);
        for (i, (&got, &exp)) in reconstructed.iter().zip(full.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-3, "output[{i}]: {got} vs {exp}");
        }
    }
}
