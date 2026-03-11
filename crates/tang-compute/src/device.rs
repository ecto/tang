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

    /// Upload data as f32 regardless of device precision mode.
    /// Used for gradient accumulators and optimizer state that need f32 precision.
    fn upload_f32(&self, data: &[f32]) -> Self::Buffer {
        self.upload(data)
    }

    /// Allocate f32 zeros regardless of device precision mode.
    fn alloc_f32(&self, len: usize) -> Self::Buffer {
        self.alloc(len)
    }

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

    /// Matrix multiply with accumulation: C[m,n] += A[m,k] * B[k,n].
    ///
    /// Unlike `matmul`, this adds to `c` instead of overwriting it.
    fn matmul_accumulate(
        &self,
        a: &Self::Buffer,
        b: &Self::Buffer,
        c: &mut Self::Buffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        let tmp = self.matmul(a, b, m, k, n);
        self.add_assign(c, &tmp);
    }

    /// Reduce sum along an axis, accumulating into dst: dst += reduce_sum(data, shape, axis).
    fn reduce_sum_accumulate(
        &self,
        data: &Self::Buffer,
        shape: &[usize],
        axis: usize,
        dst: &mut Self::Buffer,
    ) {
        let tmp = self.reduce_sum(data, shape, axis);
        self.add_assign(dst, &tmp);
    }

    /// Backward pass for RMS normalization, accumulating grad_weight into an existing buffer.
    ///
    /// Returns grad_input. grad_weight is accumulated (+=) into `grad_weight_acc`.
    fn rms_norm_backward_accumulate(
        &self,
        input: &Self::Buffer,
        weight: &Self::Buffer,
        grad_output: &Self::Buffer,
        n_groups: usize,
        dim: usize,
        eps: f32,
        grad_weight_acc: &mut Self::Buffer,
    ) -> Self::Buffer {
        let (gi, gw) = self.rms_norm_backward(input, weight, grad_output, n_groups, dim, eps);
        self.add_assign(grad_weight_acc, &gw);
        gi
    }

    /// Extract columns [col_start, col_start + col_count) from a [batch, total_cols] matrix.
    /// Returns a contiguous [batch, col_count] buffer.
    fn extract_columns(
        &self,
        buf: &Self::Buffer,
        batch: usize,
        total_cols: usize,
        col_start: usize,
        col_count: usize,
    ) -> Self::Buffer {
        let data = buf.to_vec();
        let mut out = Vec::with_capacity(batch * col_count);
        for row in 0..batch {
            let row_start = row * total_cols + col_start;
            out.extend_from_slice(&data[row_start..row_start + col_count]);
        }
        self.upload(&out)
    }

    /// Fused residual add + RMS normalization.
    /// Computes `rms_norm(input + residual, weight, eps)`.
    /// Returns `(normed_output, pre_norm_sum)`.
    fn rms_norm_residual(
        &self,
        input: &Self::Buffer,
        residual: &Self::Buffer,
        weight: &Self::Buffer,
        n_groups: usize,
        dim: usize,
        eps: f32,
    ) -> (Self::Buffer, Self::Buffer) {
        // Default: compute on CPU
        let input_data = input.to_vec();
        let residual_data = residual.to_vec();
        let weight_data = weight.to_vec();
        let mut sum_out = vec![0.0f32; n_groups * dim];
        let mut output = vec![0.0f32; n_groups * dim];
        for g in 0..n_groups {
            let base = g * dim;
            let mut sq_sum = 0.0f32;
            for i in 0..dim {
                let v = input_data[base + i] + residual_data[base + i];
                sum_out[base + i] = v;
                sq_sum += v * v;
            }
            let inv_rms = 1.0 / (sq_sum / dim as f32 + eps).sqrt();
            for i in 0..dim {
                output[base + i] = sum_out[base + i] * inv_rms * weight_data[i];
            }
        }
        (self.upload(&output), self.upload(&sum_out))
    }

    /// In-place element-wise addition: dst[i] += src[i].
    fn add_assign(&self, dst: &mut Self::Buffer, src: &Self::Buffer);

    /// Zero out all elements in a buffer.
    fn zero_buffer(&self, buf: &mut Self::Buffer);

    /// AdamW optimizer step on a single parameter tensor (in-place on device).
    ///
    /// Updates `param`, `m` (first moment), and `v` (second moment) in-place.
    /// Implements decoupled weight decay: param -= lr * wd * param before the Adam update.
    fn adamw_step(
        &self,
        param: &mut Self::Buffer,
        grad: &Self::Buffer,
        m: &mut Self::Buffer,
        v: &mut Self::Buffer,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step_t: usize,
    );
}
