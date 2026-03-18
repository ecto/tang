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

    /// Total device memory in bytes (VRAM). Returns 0 if unknown.
    fn total_memory_bytes(&self) -> usize { 0 }

    /// Free device memory in bytes. Returns 0 if unknown.
    fn free_memory_bytes(&self) -> usize { 0 }

    /// Release cached buffers in the device memory pool. No-op on devices
    /// without pooling. Call between long-running phases to prevent
    /// fragmentation-induced OOM.
    fn pool_clear(&self) {}

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

    /// Allocate buffer without zeroing. Callers must fully initialize before reading.
    /// Used for output buffers (matmul results, kernel outputs) that will be immediately overwritten.
    fn alloc_uninit(&self, len: usize) -> Self::Buffer {
        self.alloc(len)
    }

    /// Like `alloc_uninit` but always f32.
    fn alloc_uninit_f32(&self, len: usize) -> Self::Buffer {
        self.alloc_f32(len)
    }

    /// Download buffer contents to CPU.
    fn download(&self, buf: &Self::Buffer) -> Vec<f32>;

    /// Upload f32 data into an existing buffer (graph-capturable, no new allocation).
    fn upload_into_f32(&self, buf: &mut Self::Buffer, data: &[f32]) {
        *buf = self.upload_f32(data);
    }

    /// Upload u32 data into an existing buffer (graph-capturable, no new allocation).
    fn upload_into_u32(&self, buf: &mut Self::Buffer, data: &[u32]) {
        *buf = self.upload_u32(data);
    }

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

    /// Like `causal_attention_backward`, but uses a cached forward output O
    /// to skip the forward recompute. Default impl ignores O and recomputes.
    fn causal_attention_backward_with_output(
        &self,
        grad_output: &Self::Buffer,
        q: &Self::Buffer,
        k: &Self::Buffer,
        v: &Self::Buffer,
        _output: &Self::Buffer,
        seq_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> (Self::Buffer, Self::Buffer, Self::Buffer) {
        self.causal_attention_backward(grad_output, q, k, v, seq_len, n_heads, n_kv_heads, head_dim)
    }

    /// Batched causal attention forward. Inputs are `[batch_size * seq_len, dim]`.
    /// Default impl loops over batch dimension.
    fn batched_causal_attention(
        &self,
        q: &Self::Buffer,
        k: &Self::Buffer,
        v: &Self::Buffer,
        seq_len: usize,
        batch_size: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Self::Buffer {
        if batch_size == 1 {
            return self.causal_attention(q, k, v, seq_len, n_heads, n_kv_heads, head_dim);
        }
        let total_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let total_rows = seq_len * batch_size;
        let mut out = self.alloc(total_rows * total_dim);
        for b in 0..batch_size {
            let q_b = self.slice_buffer(q, b * seq_len * total_dim, seq_len * total_dim);
            let k_b = self.slice_buffer(k, b * seq_len * kv_dim, seq_len * kv_dim);
            let v_b = self.slice_buffer(v, b * seq_len * kv_dim, seq_len * kv_dim);
            let o_b = self.causal_attention(&q_b, &k_b, &v_b, seq_len, n_heads, n_kv_heads, head_dim);
            self.write_into(&mut out, b * seq_len * total_dim, &o_b);
        }
        out
    }

    /// Batched causal attention backward with cached forward output.
    /// All inputs/outputs are `[batch_size * seq_len, dim]`.
    /// Default impl loops over batch dimension.
    fn batched_causal_attention_backward(
        &self,
        grad_output: &Self::Buffer,
        q: &Self::Buffer,
        k: &Self::Buffer,
        v: &Self::Buffer,
        output: &Self::Buffer,
        seq_len: usize,
        batch_size: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> (Self::Buffer, Self::Buffer, Self::Buffer) {
        if batch_size == 1 {
            return self.causal_attention_backward_with_output(
                grad_output, q, k, v, output, seq_len, n_heads, n_kv_heads, head_dim,
            );
        }
        let total_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let total_rows = seq_len * batch_size;
        let mut gq = self.alloc_f32(total_rows * total_dim);
        let mut gk = self.alloc_f32(total_rows * kv_dim);
        let mut gv = self.alloc_f32(total_rows * kv_dim);
        for b in 0..batch_size {
            let go_b = self.slice_buffer(grad_output, b * seq_len * total_dim, seq_len * total_dim);
            let q_b = self.slice_buffer(q, b * seq_len * total_dim, seq_len * total_dim);
            let k_b = self.slice_buffer(k, b * seq_len * kv_dim, seq_len * kv_dim);
            let v_b = self.slice_buffer(v, b * seq_len * kv_dim, seq_len * kv_dim);
            let o_b = self.slice_buffer(output, b * seq_len * total_dim, seq_len * total_dim);
            let (gq_b, gk_b, gv_b) = self.causal_attention_backward_with_output(
                &go_b, &q_b, &k_b, &v_b, &o_b, seq_len, n_heads, n_kv_heads, head_dim,
            );
            self.write_into(&mut gq, b * seq_len * total_dim, &gq_b);
            self.write_into(&mut gk, b * seq_len * kv_dim, &gk_b);
            self.write_into(&mut gv, b * seq_len * kv_dim, &gv_b);
        }
        (gq, gk, gv)
    }

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

    /// Like cross_entropy_forward_backward but with pre-counted non-pad positions.
    /// Avoids GPU→CPU sync to count targets.
    fn cross_entropy_forward_backward_counted(
        &self,
        logits: &Self::Buffer,
        targets: &Self::Buffer,
        n_positions: usize,
        vocab_size: usize,
        pad_id: u32,
        non_pad_count: u32,
    ) -> (f32, Self::Buffer) {
        // Default: ignore count, fall back to standard impl
        let _ = non_pad_count;
        self.cross_entropy_forward_backward(logits, targets, n_positions, vocab_size, pad_id)
    }

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

    /// Matrix multiply with transposed B: C[m,n] = A[m,k] @ B_stored[n,k]^T.
    ///
    /// `b` is stored as [n,k] row-major (logically transposed to [k,n]).
    fn matmul_b_transposed(
        &self,
        a: &Self::Buffer,       // [m, k] row-major
        b: &Self::Buffer,       // [n, k] row-major (transposed logically)
        m: usize,
        k: usize,
        n: usize,
    ) -> Self::Buffer {
        let b_t = self.transpose_2d(b, n, k);
        self.matmul(a, &b_t, m, k, n)
    }

    /// Matrix multiply with transposed A: C[m,n] = A[k,m]^T @ B[k,n].
    ///
    /// `a` is stored as [k,m] row-major. Avoids materializing the transpose.
    fn matmul_a_transposed(
        &self,
        a: &Self::Buffer,  // [k, m] row-major (will be logically transposed)
        b: &Self::Buffer,   // [k, n] row-major
        m: usize,
        k: usize,
        n: usize,
    ) -> Self::Buffer {
        let a_t = self.transpose_2d(a, k, m);
        self.matmul(&a_t, b, m, k, n)
    }

    /// Matrix multiply with transposed A and accumulation: C[m,n] += A[k,m]^T @ B[k,n].
    ///
    /// `a` is stored as [k,m] row-major. Avoids materializing the transpose.
    fn matmul_accumulate_a_transposed(
        &self,
        a: &Self::Buffer,  // [k, m] row-major (will be logically transposed)
        b: &Self::Buffer,   // [k, n] row-major
        c: &mut Self::Buffer, // [m, n] row-major, accumulated
        m: usize,
        k: usize,
        n: usize,
    ) {
        let a_t = self.transpose_2d(a, k, m);
        self.matmul_accumulate(&a_t, b, c, m, k, n);
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

    /// Backward pass for RMS normalization, fused with residual gradient addition.
    /// Returns grad_input + residual_grad. grad_weight is accumulated into `grad_weight_acc`.
    /// Eliminates a separate `add_tensors` kernel launch.
    fn rms_norm_backward_residual_accumulate(
        &self,
        input: &Self::Buffer,
        weight: &Self::Buffer,
        grad_output: &Self::Buffer,
        residual_grad: &Self::Buffer,
        n_groups: usize,
        dim: usize,
        eps: f32,
        grad_weight_acc: &mut Self::Buffer,
    ) -> Self::Buffer {
        // Default: unfused path
        let gi = self.rms_norm_backward_accumulate(
            input, weight, grad_output, n_groups, dim, eps, grad_weight_acc,
        );
        self.add_tensors_buf(&gi, residual_grad, n_groups * dim)
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

    /// Write `src[batch, col_count]` into columns [col_start, col_start+col_count) of `dst[batch, total_cols]`.
    /// Inverse of `extract_columns`.
    fn concat_columns(
        &self,
        dst: &mut Self::Buffer,
        src: &Self::Buffer,
        batch: usize,
        total_cols: usize,
        col_start: usize,
        col_count: usize,
    ) {
        let dst_data = dst.to_vec();
        let src_data = src.to_vec();
        let mut out = dst_data;
        for row in 0..batch {
            let dst_start = row * total_cols + col_start;
            let src_start = row * col_count;
            out[dst_start..dst_start + col_count]
                .copy_from_slice(&src_data[src_start..src_start + col_count]);
        }
        *dst = self.upload(&out);
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

    /// Apply interleaved RoPE forward: rotates pairs (2i, 2i+1) by position-dependent angles.
    ///
    /// Input: `[seq_len, n_heads, head_dim]` flat buffer.
    /// cos/sin tables: `[max_seq_len, half_dim]` precomputed on CPU.
    /// Returns buffer of same shape.
    fn rope_forward(
        &self,
        input: &Self::Buffer,
        cos_table: &[f32],
        sin_table: &[f32],
        seq_len: usize,
        n_heads: usize,
        head_dim: usize,
        start_pos: usize,
    ) -> Self::Buffer {
        let data = self.download(input);
        let half_dim = head_dim / 2;
        let mut out = vec![0.0f32; data.len()];
        for s in 0..seq_len {
            let pos = start_pos + s;
            for h in 0..n_heads {
                let base = (s * n_heads + h) * head_dim;
                for i in 0..half_dim {
                    let cos = cos_table[pos * half_dim + i];
                    let sin = sin_table[pos * half_dim + i];
                    let x0 = data[base + 2 * i];
                    let x1 = data[base + 2 * i + 1];
                    out[base + 2 * i] = x0 * cos - x1 * sin;
                    out[base + 2 * i + 1] = x0 * sin + x1 * cos;
                }
            }
        }
        self.upload(&out)
    }

    /// Apply interleaved RoPE backward: reverse rotation (transpose of rotation matrix).
    fn rope_backward(
        &self,
        grad_output: &Self::Buffer,
        cos_table: &[f32],
        sin_table: &[f32],
        seq_len: usize,
        n_heads: usize,
        head_dim: usize,
        start_pos: usize,
    ) -> Self::Buffer {
        let data = self.download(grad_output);
        let half_dim = head_dim / 2;
        let mut out = vec![0.0f32; data.len()];
        for s in 0..seq_len {
            let pos = start_pos + s;
            for h in 0..n_heads {
                let base = (s * n_heads + h) * head_dim;
                for i in 0..half_dim {
                    let cos = cos_table[pos * half_dim + i];
                    let sin = sin_table[pos * half_dim + i];
                    let g0 = data[base + 2 * i];
                    let g1 = data[base + 2 * i + 1];
                    out[base + 2 * i] = g0 * cos + g1 * sin;
                    out[base + 2 * i + 1] = -g0 * sin + g1 * cos;
                }
            }
        }
        self.upload(&out)
    }

    /// Batched RoPE backward: input is [batch*seq_len, n_heads, head_dim].
    /// Positions wrap every `seq_len` elements (each batch starts at `start_pos`).
    fn rope_backward_batched(
        &self,
        grad_output: &Self::Buffer,
        cos_table: &[f32],
        sin_table: &[f32],
        total_rows: usize,
        seq_len: usize,
        n_heads: usize,
        head_dim: usize,
        start_pos: usize,
    ) -> Self::Buffer {
        let data = self.download(grad_output);
        let half_dim = head_dim / 2;
        let mut out = vec![0.0f32; data.len()];
        for s in 0..total_rows {
            let pos = start_pos + (s % seq_len);
            for h in 0..n_heads {
                let base = (s * n_heads + h) * head_dim;
                for i in 0..half_dim {
                    let cos = cos_table[pos * half_dim + i];
                    let sin = sin_table[pos * half_dim + i];
                    let g0 = data[base + 2 * i];
                    let g1 = data[base + 2 * i + 1];
                    out[base + 2 * i] = g0 * cos + g1 * sin;
                    out[base + 2 * i + 1] = -g0 * sin + g1 * cos;
                }
            }
        }
        self.upload(&out)
    }

    /// RoPE forward with pre-uploaded cos/sin buffers on device.
    /// Avoids re-uploading tables every call.
    fn rope_forward_cached(
        &self,
        input: &Self::Buffer,
        cos_buf: &Self::Buffer,
        sin_buf: &Self::Buffer,
        seq_len: usize,
        n_heads: usize,
        head_dim: usize,
        start_pos: usize,
    ) -> Self::Buffer {
        // Default: download tables and delegate
        let cos_table = self.download(cos_buf);
        let sin_table = self.download(sin_buf);
        self.rope_forward(
            input, &cos_table, &sin_table,
            seq_len, n_heads, head_dim, start_pos,
        )
    }

    /// Batched RoPE backward with pre-uploaded cos/sin buffers on device.
    /// Avoids re-uploading tables every call.
    fn rope_backward_batched_cached(
        &self,
        grad_output: &Self::Buffer,
        cos_buf: &Self::Buffer,
        sin_buf: &Self::Buffer,
        total_rows: usize,
        seq_len: usize,
        n_heads: usize,
        head_dim: usize,
        start_pos: usize,
    ) -> Self::Buffer {
        // Default: download tables and delegate
        let cos_table = self.download(cos_buf);
        let sin_table = self.download(sin_buf);
        self.rope_backward_batched(
            grad_output, &cos_table, &sin_table,
            total_rows, seq_len, n_heads, head_dim, start_pos,
        )
    }

    /// Element-wise addition: out[i] = a[i] + b[i].
    /// Default: delegates to elementwise(). Override for fused bf16 kernel.
    fn add_tensors_buf(&self, a: &Self::Buffer, b: &Self::Buffer, numel: usize) -> Self::Buffer {
        self.elementwise(&[a, b], numel, &|ids| ids[0] + ids[1])
    }

    /// SwiGLU activation: out[i] = silu(gate[i]) * up[i].
    /// Default: delegates to elementwise(). Override for fused bf16 kernel.
    fn swiglu_fused_buf(&self, gate: &Self::Buffer, up: &Self::Buffer, numel: usize) -> Self::Buffer {
        use tang::Scalar;
        self.elementwise(&[gate, up], numel, &|ids| {
            let one = ExprId::from_f64(1.0);
            let neg_gate = -ids[0];
            let exp_neg = Scalar::exp(neg_gate);
            let sigmoid = one / (one + exp_neg);
            ids[0] * sigmoid * ids[1]
        })
    }

    /// SwiGLU backward: returns (grad_gate, grad_up).
    /// Default: delegates to elementwise(). Override for fused bf16 kernel.
    fn swiglu_backward_buf(
        &self,
        grad: &Self::Buffer,
        gate: &Self::Buffer,
        up: &Self::Buffer,
        numel: usize,
    ) -> (Self::Buffer, Self::Buffer) {
        use tang::Scalar;
        let grad_up = self.elementwise(&[grad, gate], numel, &|ids| {
            let one = ExprId::from_f64(1.0);
            let neg_gate = -ids[1];
            let exp_neg = Scalar::exp(neg_gate);
            let sigmoid = one / (one + exp_neg);
            ids[0] * ids[1] * sigmoid
        });
        let grad_gate = self.elementwise(&[grad, gate, up], numel, &|ids| {
            let one = ExprId::from_f64(1.0);
            let neg_gate = -ids[1];
            let exp_neg = Scalar::exp(neg_gate);
            let sigmoid = one / (one + exp_neg);
            let dsilu = sigmoid * (one + ids[1] * (one - sigmoid));
            ids[0] * ids[2] * dsilu
        });
        (grad_gate, grad_up)
    }

    /// In-place element-wise addition: dst[i] += src[i].
    fn add_assign(&self, dst: &mut Self::Buffer, src: &Self::Buffer);

    /// Zero out all elements in a buffer.
    fn zero_buffer(&self, buf: &mut Self::Buffer);

    /// Accumulate sum-of-squares of `src` into `acc` (single f32 buffer, atomicAdd).
    /// `acc` must be a 1-element buffer, zero-initialized before the first call.
    fn reduce_sum_sq_accumulate(&self, src: &Self::Buffer, acc: &mut Self::Buffer) {
        let data = self.download(src);
        let sq: f32 = data.iter().map(|&v| v * v).sum();
        let mut a = self.download(acc);
        a[0] += sq;
        *acc = self.upload(&a);
    }

    /// Compute sum-of-squares across multiple buffers in a single fused operation.
    /// Returns a 1-element buffer containing the total sum of squares (read later to avoid sync).
    /// Default: calls reduce_sum_sq_accumulate per buffer. Override for GPU-fused version.
    fn fused_sum_sq(&self, bufs: &[&Self::Buffer]) -> Self::Buffer {
        let mut acc = self.upload_f32(&[0.0f32]);
        for buf in bufs {
            self.reduce_sum_sq_accumulate(buf, &mut acc);
        }
        acc
    }

    /// Compute global L2 norm across multiple buffers, clip if above max_norm.
    /// Returns the pre-clip norm. Fused for efficiency (2 kernel launches instead of N).
    fn clip_grad_norm(&self, bufs: &mut [&mut Self::Buffer], max_norm: f32) -> f32 {
        let mut total_sq: f64 = 0.0;
        for buf in bufs.iter() {
            let data = self.download(*buf);
            for &v in &data {
                total_sq += (v as f64) * (v as f64);
            }
        }
        let norm = total_sq.sqrt() as f32;
        if norm > max_norm {
            let scale = max_norm / norm;
            for buf in bufs.iter_mut() {
                self.scale_buffer(*buf, scale);
            }
        }
        norm
    }

    /// Add norm-relative Gaussian noise in-place.
    ///
    /// For each row: `data[row, col] += epsilon * ||row||_2 * N(0,1)`.
    /// Uses counter-based PRNG seeded by `seed` for reproducibility.
    /// `rows` × `cols` must equal the buffer length.
    fn add_norm_relative_noise(
        &self,
        _buf: &mut Self::Buffer,
        _epsilon: f32,
        _seed: u64,
        _rows: usize,
        _cols: usize,
    ) {
        panic!("add_norm_relative_noise not implemented for this device");
    }

    /// In-place scale: buf[i] *= scale.
    fn scale_buffer(&self, buf: &mut Self::Buffer, scale: f32) {
        let mut data = self.download(buf);
        for v in data.iter_mut() {
            *v *= scale;
        }
        *buf = self.upload(&data);
    }

    /// Extract a contiguous sub-range from a buffer (offset and len in elements).
    fn slice_buffer(&self, buf: &Self::Buffer, offset: usize, len: usize) -> Self::Buffer {
        let data = self.download(buf);
        self.upload(&data[offset..offset + len])
    }

    /// Write `src` into `dst` starting at element `offset`.
    fn write_into(&self, dst: &mut Self::Buffer, offset: usize, src: &Self::Buffer) {
        let mut d = self.download(dst);
        let s = self.download(src);
        d[offset..offset + s.len()].copy_from_slice(&s);
        *dst = self.upload(&d);
    }

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
