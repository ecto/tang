//! ComputeDevice and ComputeBuffer traits.

use tang_expr::codegen::Dialect;
use tang_expr::node::ExprId;

/// GPU/CPU buffer holding f32 data.
pub trait ComputeBuffer: Send {
    /// Number of f32 elements.
    fn len(&self) -> usize;
    /// Whether the buffer is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Download contents to CPU.
    fn to_vec(&self) -> Vec<f32>;
}

/// Compute device abstraction over CPU, Metal, and CUDA backends.
pub trait ComputeDevice: Send {
    /// The buffer type for this device.
    type Buffer: ComputeBuffer;

    /// Which shader dialect this device uses.
    fn dialect(&self) -> Dialect;

    // -- Buffer lifecycle --

    /// Upload f32 data from CPU to device.
    fn upload(&self, data: &[f32]) -> Self::Buffer;

    /// Upload u32 data (e.g. token IDs) to device.
    fn upload_u32(&self, data: &[u32]) -> Self::Buffer;

    /// Allocate uninitialized buffer of `len` f32 elements.
    fn alloc(&self, len: usize) -> Self::Buffer;

    /// Download buffer contents to CPU.
    fn download(&self, buf: &Self::Buffer) -> Vec<f32>;

    // -- Auto-generated elementwise (via tang-expr) --

    /// Fused elementwise operation: trace closure → compile kernel → dispatch.
    ///
    /// The closure receives one `ExprId` per input buffer and returns the output expression.
    /// All operations are fused into a single kernel dispatch.
    fn elementwise(
        &self,
        inputs: &[&Self::Buffer],
        numel: usize,
        f: &dyn Fn(&[ExprId]) -> ExprId,
    ) -> Self::Buffer;

    // -- Hand-optimized operations --

    /// Matrix multiply: C[m,n] = A[m,k] * B[k,n], row-major.
    fn matmul(
        &self,
        a: &Self::Buffer,
        b: &Self::Buffer,
        m: usize,
        k: usize,
        n: usize,
    ) -> Self::Buffer;

    /// Row-wise softmax: each of `n_rows` rows of length `row_len`.
    fn softmax(
        &self,
        data: &Self::Buffer,
        n_rows: usize,
        row_len: usize,
    ) -> Self::Buffer;

    /// RMS normalization: x * weight / sqrt(mean(x^2) + eps).
    fn rms_norm(
        &self,
        data: &Self::Buffer,
        weight: &Self::Buffer,
        n_groups: usize,
        dim: usize,
        eps: f32,
    ) -> Self::Buffer;

    /// Embedding lookup: weight[ids[i]] for each token.
    fn embedding(
        &self,
        weight: &Self::Buffer,
        ids: &Self::Buffer,
        seq_len: usize,
        dim: usize,
    ) -> Self::Buffer;

    /// Reduce sum along an axis.
    fn reduce_sum(
        &self,
        data: &Self::Buffer,
        shape: &[usize],
        axis: usize,
    ) -> Self::Buffer;

    /// Causal self-attention with GQA: Q,K,V → output.
    /// Q: [seq_len, n_heads * head_dim], K,V: [seq_len, n_kv_heads * head_dim].
    /// Output: [seq_len, n_heads * head_dim].
    fn causal_attention(
        &self,
        q: &Self::Buffer,
        k: &Self::Buffer,
        v: &Self::Buffer,
        seq_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Self::Buffer;

    /// KV-cached attention for incremental decoding and batched prefill.
    ///
    /// - `q`: `[q_len, n_heads * head_dim]`
    /// - `k_cache`, `v_cache`: `[cache_start + q_len, n_kv_heads * head_dim]`
    /// - `cache_start`: number of positions already in cache before this batch
    /// - `q_len`: number of new query positions (1 for decode, N for prefill)
    ///
    /// Causal mask: query `i` attends to positions `0..cache_start + i + 1`.
    /// Returns `[q_len, n_heads * head_dim]`.
    fn kv_attention(
        &self,
        q: &Self::Buffer,
        k_cache: &Self::Buffer,
        v_cache: &Self::Buffer,
        cache_start: usize,
        q_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Self::Buffer;

    /// Transpose a 2D matrix on device: [rows, cols] → [cols, rows].
    fn transpose_2d(
        &self,
        buf: &Self::Buffer,
        rows: usize,
        cols: usize,
    ) -> Self::Buffer;

    /// Backward pass for row-wise softmax.
    ///
    /// Given softmax output `sm` and upstream gradient `grad_output`,
    /// computes `grad_input[i,j] = sm[i,j] * (grad[i,j] - dot(sm[i,:], grad[i,:]))`.
    fn softmax_backward(
        &self,
        softmax_out: &Self::Buffer,
        grad_output: &Self::Buffer,
        n_rows: usize,
        row_len: usize,
    ) -> Self::Buffer;

    /// Backward pass for RMS normalization.
    ///
    /// Returns `(grad_input, grad_weight)`.
    fn rms_norm_backward(
        &self,
        input: &Self::Buffer,
        weight: &Self::Buffer,
        grad_output: &Self::Buffer,
        n_groups: usize,
        dim: usize,
        eps: f32,
    ) -> (Self::Buffer, Self::Buffer);

    /// Backward pass for embedding lookup (scatter-add).
    ///
    /// `grad_weight[ids[i]] += grad_output[i]` for each position.
    /// Returns gradient w.r.t. weight: `[vocab_size, dim]`.
    fn embedding_backward(
        &self,
        grad_output: &Self::Buffer,
        ids: &Self::Buffer,
        vocab_size: usize,
        seq_len: usize,
        dim: usize,
    ) -> Self::Buffer;

    /// Backward pass for causal self-attention with GQA.
    ///
    /// Recomputes attention scores from Q,K,V, then computes gradients.
    /// Q, grad_output: `[seq_len, n_heads * head_dim]`
    /// K, V: `[seq_len, n_kv_heads * head_dim]`
    /// Returns `(grad_Q, grad_K, grad_V)` with same shapes as inputs.
    fn causal_attention_backward(
        &self,
        grad_output: &Self::Buffer,
        q: &Self::Buffer,
        k: &Self::Buffer,
        v: &Self::Buffer,
        seq_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> (Self::Buffer, Self::Buffer, Self::Buffer);

    /// Fused cross-entropy forward + backward.
    ///
    /// Computes per-row log-softmax → CE loss, and gradient = (softmax - one_hot) / count.
    /// Positions where `target == pad_id` are excluded from loss and get zero gradient.
    /// Returns `(loss, grad_logits)`.
    fn cross_entropy_forward_backward(
        &self,
        logits: &Self::Buffer,
        targets: &Self::Buffer,
        n_positions: usize,
        vocab_size: usize,
        pad_id: u32,
    ) -> (f32, Self::Buffer);

    /// Wait for all pending operations to complete.
    fn sync(&self);

    /// Copy a buffer on device without CPU round-trip (GPU backends use blit/copy).
    fn copy_buffer(&self, src: &Self::Buffer) -> Self::Buffer {
        let data = self.download(src);
        self.upload(&data)
    }

    /// Broadcast bias addition on device: out[i] = matrix[i] + bias[i % dim].
    ///
    /// `numel` is total elements in matrix, `dim` is the bias length.
    fn bias_add(&self, matrix: &Self::Buffer, bias: &Self::Buffer, numel: usize, dim: usize) -> Self::Buffer {
        let mat_data = self.download(matrix);
        let bias_data = self.download(bias);
        let mut out = mat_data;
        for i in 0..numel {
            out[i] += bias_data[i % dim];
        }
        self.upload(&out)
    }
}
