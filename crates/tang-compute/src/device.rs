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

    /// Causal self-attention: Q,K,V → output.
    /// Q,K,V shapes: [seq_len, n_heads * head_dim].
    fn causal_attention(
        &self,
        q: &Self::Buffer,
        k: &Self::Buffer,
        v: &Self::Buffer,
        seq_len: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Self::Buffer;

    /// KV-cached attention for incremental decoding.
    /// q: [1, n_heads * head_dim], k_cache/v_cache: [cache_len, n_kv_heads * head_dim].
    fn kv_attention(
        &self,
        q: &Self::Buffer,
        k_cache: &Self::Buffer,
        v_cache: &Self::Buffer,
        cache_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Self::Buffer;

    /// Wait for all pending operations to complete.
    fn sync(&self);
}
