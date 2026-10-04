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
    fn total_memory_bytes(&self) -> usize {
        0
    }

    /// Free device memory in bytes. Returns 0 if unknown.
    fn free_memory_bytes(&self) -> usize {
        0
    }

    /// Release cached buffers in the device memory pool. No-op on devices
    /// without pooling. Call between long-running phases to prevent
    /// fragmentation-induced OOM.
    fn pool_clear(&self) {}

    /// Peak FLOPS for FP32 compute. Used for MFU calculation.
    /// Returns None if unknown.
    fn peak_flops_f32(&self) -> Option<f64> {
        None
    }

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
    fn softmax(&self, data: &Self::Buffer, n_rows: usize, row_len: usize) -> Self::Buffer;

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
    fn reduce_sum(&self, data: &Self::Buffer, shape: &[usize], axis: usize) -> Self::Buffer;

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
    fn transpose_2d(&self, buf: &Self::Buffer, rows: usize, cols: usize) -> Self::Buffer;

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
            let o_b =
                self.causal_attention(&q_b, &k_b, &v_b, seq_len, n_heads, n_kv_heads, head_dim);
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
                grad_output,
                q,
                k,
                v,
                output,
                seq_len,
                n_heads,
                n_kv_heads,
                head_dim,
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

    /// Submit queued work without waiting, so the device can start on it while the caller
    /// keeps encoding. Default: no-op.
    fn flush(&self) {}

    /// Copy a buffer on device without CPU round-trip (GPU backends use blit/copy).
    fn copy_buffer(&self, src: &Self::Buffer) -> Self::Buffer {
        let data = self.download(src);
        self.upload(&data)
    }

    /// Broadcast bias addition on device: out[i] = matrix[i] + bias[i % dim].
    ///
    /// `numel` is total elements in matrix, `dim` is the bias length.
    fn bias_add(
        &self,
        matrix: &Self::Buffer,
        bias: &Self::Buffer,
        numel: usize,
        dim: usize,
    ) -> Self::Buffer {
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
        a: &Self::Buffer, // [m, k] row-major
        b: &Self::Buffer, // [n, k] row-major (transposed logically)
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
        a: &Self::Buffer, // [k, m] row-major (will be logically transposed)
        b: &Self::Buffer, // [k, n] row-major
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
        a: &Self::Buffer,     // [k, m] row-major (will be logically transposed)
        b: &Self::Buffer,     // [k, n] row-major
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
            input,
            weight,
            grad_output,
            n_groups,
            dim,
            eps,
            grad_weight_acc,
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
            input, &cos_table, &sin_table, seq_len, n_heads, head_dim, start_pos,
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
            grad_output,
            &cos_table,
            &sin_table,
            total_rows,
            seq_len,
            n_heads,
            head_dim,
            start_pos,
        )
    }

    /// Element-wise addition: out[i] = a[i] + b[i].
    /// Default: delegates to elementwise(). Override for fused bf16 kernel.
    fn add_tensors_buf(&self, a: &Self::Buffer, b: &Self::Buffer, numel: usize) -> Self::Buffer {
        self.elementwise(&[a, b], numel, &|ids| ids[0] + ids[1])
    }

    /// SwiGLU activation: out[i] = silu(gate[i]) * up[i].
    /// Default: delegates to elementwise(). Override for fused bf16 kernel.
    fn swiglu_fused_buf(
        &self,
        gate: &Self::Buffer,
        up: &Self::Buffer,
        numel: usize,
    ) -> Self::Buffer {
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

    /// Upload read-only weights given as raw bfloat16 bits. Backends that support it keep them
    /// in bf16 (half the memory and bandwidth) for [`linear`](Self::linear) and
    /// [`embedding`](Self::embedding); the default widens to f32.
    fn upload_bf16(&self, bits: &[u16]) -> Self::Buffer {
        let wide: Vec<f32> = bits
            .iter()
            .map(|&b| f32::from_bits((b as u32) << 16))
            .collect();
        self.upload(&wide)
    }

    /// Upload 4-bit affine-quantized weights in MLX's layout: `packed` holds 8 weights per
    /// u32 (low nibble first) for a row-major `[n, k]` matrix; `scales` and `biases` are bf16
    /// bits, one per `group` consecutive weights of a row (`w = scale * q + bias`). Backends
    /// that support it keep this format for [`linear`](Self::linear) and
    /// [`embedding`](Self::embedding); the default dequantizes to f32.
    fn upload_q4(
        &self,
        packed: &[u32],
        scales: &[u16],
        biases: &[u16],
        group: usize,
    ) -> Self::Buffer {
        let bf = |b: u16| f32::from_bits((b as u32) << 16);
        let n = packed.len() * 8;
        let w: Vec<f32> = (0..n)
            .map(|i| {
                let q = (packed[i / 8] >> (4 * (i % 8))) & 0xf;
                bf(scales[i / group]) * q as f32 + bf(biases[i / group])
            })
            .collect();
        self.upload(&w)
    }

    /// Decoder attention prologue, fused. `qkv` is `[seq, (nh + 2*nkv) * hd]` (a fused Q/K/V
    /// projection). Applies the optional per-head RMS norms to q and k, half-split RoPE at
    /// positions `pos..`, writes k and v into the caches at `pos`, and returns q `[seq, nh*hd]`.
    #[allow(clippy::too_many_arguments)]
    fn attention_prep(
        &self,
        qkv: &Self::Buffer,
        q_norm: Option<&Self::Buffer>,
        k_norm: Option<&Self::Buffer>,
        cos: &Self::Buffer,
        sin: &Self::Buffer,
        k_cache: &mut Self::Buffer,
        v_cache: &mut Self::Buffer,
        seq: usize,
        (nh, nkv, hd): (usize, usize, usize),
        pos: usize,
        eps: f32,
    ) -> Self::Buffer {
        attention_prep_default(
            self,
            qkv,
            q_norm,
            k_norm,
            cos,
            sin,
            k_cache,
            v_cache,
            seq,
            (nh, nkv, hd),
            pos,
            eps,
        )
    }

    /// `kv_attention` where each query sees only the last `window` keys (0: all of them), as in
    /// Gemma's local layers. The portable fallback runs on the host.
    #[allow(clippy::too_many_arguments)]
    fn kv_attention_window(
        &self,
        q: &Self::Buffer,
        k_cache: &Self::Buffer,
        v_cache: &Self::Buffer,
        cache_start: usize,
        q_len: usize,
        (nh, nkv, hd): (usize, usize, usize),
        window: usize,
        causal: bool,
    ) -> Self::Buffer {
        if causal && (window == 0 || cache_start + q_len <= window) {
            return self.kv_attention(q, k_cache, v_cache, cache_start, q_len, nh, nkv, hd);
        }
        let (q, k, v) = (
            self.download(q),
            self.download(k_cache),
            self.download(v_cache),
        );
        let (kvd, scale) = (nkv * hd, 1.0 / (hd as f32).sqrt());
        let mut out = vec![0.0f32; q_len * nh * hd];
        for qi in 0..q_len {
            let qpos = cache_start + qi + 1;
            let attend = if causal { qpos } else { cache_start + q_len };
            let lo = if window > 0 {
                qpos.saturating_sub(window)
            } else {
                0
            };
            for h in 0..nh {
                let kh = h / (nh / nkv);
                let qv = &q[(qi * nh + h) * hd..][..hd];
                let scores: Vec<f32> = (lo..attend)
                    .map(|j| {
                        (0..hd)
                            .map(|d| qv[d] * k[j * kvd + kh * hd + d])
                            .sum::<f32>()
                            * scale
                    })
                    .collect();
                let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
                let z: f32 = e.iter().sum();
                let o = &mut out[(qi * nh + h) * hd..][..hd];
                for (t, j) in (lo..attend).enumerate() {
                    for d in 0..hd {
                        o[d] += e[t] / z * v[j * kvd + kh * hd + d];
                    }
                }
            }
        }
        self.upload(&out)
    }

    /// Bidirectional multi-head attention over `n` positions (vision encoders): q, k, v and the
    /// result are `[n, nh * hd]`.
    fn attention_full(
        &self,
        q: &Self::Buffer,
        k: &Self::Buffer,
        v: &Self::Buffer,
        n: usize,
        nh: usize,
        hd: usize,
    ) -> Self::Buffer {
        self.kv_attention_window(q, k, v, 0, n, (nh, nh, hd), 0, false)
    }

    /// LayerNorm over rows of `dim`, with weight and bias.
    fn layer_norm(
        &self,
        x: &Self::Buffer,
        w: &Self::Buffer,
        b: &Self::Buffer,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Self::Buffer {
        let (x, w, b) = (self.download(x), self.download(w), self.download(b));
        let mut y = Vec::with_capacity(rows * dim);
        for r in x.chunks(dim).take(rows) {
            let mean = r.iter().sum::<f32>() / dim as f32;
            let var = r.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
            let inv = 1.0 / (var + eps).sqrt();
            y.extend(
                r.iter()
                    .enumerate()
                    .map(|(i, v)| (v - mean) * inv * w[i] + b[i]),
            );
        }
        self.upload(&y)
    }

    /// GELU, tanh approximation, elementwise over `n` values.
    fn gelu_tanh(&self, x: &Self::Buffer, n: usize) -> Self::Buffer {
        let x = self.download(x);
        let y: Vec<f32> = x[..n]
            .iter()
            .map(|&g| 0.5 * g * (1.0 + (0.797_884_6 * (g + 0.044715 * g * g * g)).tanh()))
            .collect();
        self.upload(&y)
    }

    /// GeGLU over a fused gate/up projection (Gemma): `gelu_tanh(gate) * up`, `[rows, ff]`.
    fn geglu_split(&self, gu: &Self::Buffer, rows: usize, ff: usize) -> Self::Buffer {
        let d = self.download(gu);
        let mut y = Vec::with_capacity(rows * ff);
        for r in 0..rows {
            for i in 0..ff {
                let g = d[r * 2 * ff + i];
                let t = (0.797_884_6 * (g + 0.044715 * g * g * g)).tanh();
                y.push(0.5 * g * (1.0 + t) * d[r * 2 * ff + ff + i]);
            }
        }
        self.upload(&y)
    }

    /// SwiGLU over a fused gate/up projection: `gu` is `[rows, 2*ff]` (gate then up per row);
    /// returns `silu(gate) * up`, `[rows, ff]`.
    fn swiglu_split(&self, gu: &Self::Buffer, rows: usize, ff: usize) -> Self::Buffer {
        let d = self.download(gu);
        let (mut g, mut u) = (Vec::with_capacity(rows * ff), Vec::with_capacity(rows * ff));
        for r in 0..rows {
            g.extend_from_slice(&d[r * 2 * ff..r * 2 * ff + ff]);
            u.extend_from_slice(&d[r * 2 * ff + ff..(r + 1) * 2 * ff]);
        }
        self.swiglu_fused_buf(&self.upload(&g), &self.upload(&u), rows * ff)
    }

    /// Linear layer: `y[m, n] = x[m, k] @ w[n, k]^T`, with `w` stored the way checkpoints store
    /// it (`[out, in]` row-major). Backends specialize small `m` (decode) as a GEMV.
    fn linear(
        &self,
        x: &Self::Buffer,
        w: &Self::Buffer,
        m: usize,
        k: usize,
        n: usize,
    ) -> Self::Buffer {
        self.matmul_b_transposed(x, w, m, k, n)
    }

    /// Half-split ("NeoX") RoPE, as used by Llama/Qwen checkpoints: rotates pairs
    /// `(i, i + head_dim/2)`. Input `[seq_len, n_heads, head_dim]`; tables `[max_pos, head_dim/2]`
    /// on device. Returns a new buffer.
    fn rope_half_cached(
        &self,
        input: &Self::Buffer,
        cos_buf: &Self::Buffer,
        sin_buf: &Self::Buffer,
        seq_len: usize,
        n_heads: usize,
        head_dim: usize,
        start_pos: usize,
    ) -> Self::Buffer {
        let data = self.download(input);
        let (cos, sin) = (self.download(cos_buf), self.download(sin_buf));
        let half = head_dim / 2;
        let mut out = data.clone();
        for s in 0..seq_len {
            let pos = start_pos + s;
            for h in 0..n_heads {
                let base = (s * n_heads + h) * head_dim;
                for i in 0..half {
                    let (c, sn) = (cos[pos * half + i], sin[pos * half + i]);
                    let (x0, x1) = (data[base + i], data[base + i + half]);
                    out[base + i] = x0 * c - x1 * sn;
                    out[base + i + half] = x1 * c + x0 * sn;
                }
            }
        }
        self.upload(&out)
    }

    // -- Flash-Next decode ops (contracts in `crate::flash`; reference math in `cpu::flash`).
    // The defaults run the reference on the host through download/upload: right on any device,
    // fast on none. Outputs are caller-allocated so a backend can capture the window in a graph.

    /// Upload raw bytes, packed little-endian 4 per element ([`crate::flash`], "Words").
    fn upload_bytes(&self, bytes: &[u8]) -> Self::Buffer {
        self.upload_u32(&crate::flash::bytes_to_words(bytes))
    }

    /// A zeroed buffer of `len` values stored as bf16 where the backend can (the QSA K/V cache).
    /// The default stores f32 (values written to it are rounded to bf16 by the ops anyway).
    fn alloc_bf16(&self, len: usize) -> Self::Buffer {
        self.alloc_f32(len)
    }

    /// The address of `buf`'s first byte as this device's kernels see it (a host address on the
    /// CPU). For expert residency tables and plans.
    fn buffer_addr(&self, _buf: &Self::Buffer) -> u64 {
        panic!("buffer_addr not implemented for this device")
    }

    /// Upload a GGUF Q2_0 matrix `[n, k]` (raw blocks), repacked by [`crate::flash::q2_repack`].
    /// Only the Q2 ops read it.
    fn upload_q2(&self, raw: &[u8], n: usize, k: usize) -> Self::Buffer {
        self.upload_bytes(&crate::flash::q2_repack(raw, n, k))
    }

    /// `out = linear(x, w)` into an existing `[m, n]` buffer (no allocation on backends that
    /// override it, for `m <= 8`).
    fn linear_into(
        &self,
        x: &Self::Buffer,
        w: &Self::Buffer,
        out: &mut Self::Buffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        *out = self.linear(x, w, m, k, n);
    }

    /// Quantize `m` rows of `k` f32 activations to int8 ([`crate::flash::QAct`]).
    fn quantize_act_into(&self, x: &Self::Buffer, xq: &mut Self::Buffer, m: usize, k: usize) {
        let w = crate::cpu::flash::quantize_act(&self.download(x)[..m * k], m, k);
        *xq = self.upload_u32(&w);
    }

    /// `out[m, n] = W · x̂` for a Q2_0 weight `[n, k]` ([`upload_q2`](Self::upload_q2)) and
    /// quantized activations (`m <= 8` rows). Per chunk `fma(d_w · d_x, Σ code·q − Σ q, acc)`;
    /// the order chunks are summed in is the backend's.
    fn q2_linear_into(
        &self,
        xq: &Self::Buffer,
        w: &Self::Buffer,
        out: &mut Self::Buffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        let (xq, w) = (self.download(xq), self.download(w));
        let wb: Vec<u8> = w.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
        let y = crate::cpu::flash::q2_linear(&crate::flash::u32s(&xq), &wb, m, k, n);
        *out = self.upload_f32(&y);
    }

    /// Upload tang-Q4 weights (`upload_q4`'s arrays, group 64) repacked as Q4X
    /// ([`crate::flash::q4x_repack`]) for [`q4x_linear_into`](Self::q4x_linear_into), the
    /// int8-activation path. Only that op reads it.
    fn upload_q4x(
        &self,
        packed: &[u32],
        scales: &[u16],
        biases: &[u16],
        n: usize,
        k: usize,
    ) -> Self::Buffer {
        self.upload_bytes(&crate::flash::q4x_repack(packed, scales, biases, n, k))
    }

    /// `out[m, n] = W · x̂` for a Q4X weight `[n, k]` and int8 activations (`m <= 8`): per
    /// chunk `fma(d_x, fma(scale, Σ q·x̂, bias · Σ x̂), acc)`; chunk summation order is the
    /// backend's. Same weights as tang-Q4 `linear`, but int8 activations (as llama.cpp's MMVQ).
    fn q4x_linear_into(
        &self,
        xq: &Self::Buffer,
        w: &Self::Buffer,
        out: &mut Self::Buffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        let (xq, w) = (self.download(xq), self.download(w));
        let wb: Vec<u8> = w.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
        let y = crate::cpu::flash::q4x_linear(&crate::flash::u32s(&xq), &wb, m, k, n);
        *out = self.upload_f32(&y);
    }

    /// Upload a GGUF Q8_0 matrix `[n, k]` (blocks of fp16 `d` + 32 int8) repacked as Q8X
    /// ([`crate::flash::q8x_repack`]) for [`q8x_linear_into`](Self::q8x_linear_into).
    fn upload_q8x(&self, raw: &[u8], n: usize, k: usize) -> Self::Buffer {
        self.upload_bytes(&crate::flash::q8x_repack(raw, n, k))
    }

    /// `out[m, n] = W · x̂` for a Q8X weight `[n, k]` and int8 activations (`m <= 8`): per half
    /// chunk `fma(d_w · d_x, Σ w·q, acc)`; the order the halves are summed in is the backend's.
    fn q8x_linear_into(
        &self,
        xq: &Self::Buffer,
        w: &Self::Buffer,
        out: &mut Self::Buffer,
        m: usize,
        k: usize,
        n: usize,
    ) {
        let (xq, w) = (self.download(xq), self.download(w));
        let wb: Vec<u8> = w.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
        let y = crate::cpu::flash::q8x_linear(&crate::flash::u32s(&xq), &wb, m, k, n);
        *out = self.upload_f32(&y);
    }

    /// Hyper-connection read for a window of `t` tokens: `r` is `[t][HC][HIDDEN]`; writes
    /// `x [t][HIDDEN]`, `xq` (when given) the same as int8 activations
    /// (`QAct { m: t, k: HIDDEN }`), and, when `w.inject` is set, `inj [t][HC]`. A `pending`
    /// write is applied to `r` first ([`crate::flash::HcPending`]: bitwise the separate ops).
    /// `scratch`: [`crate::flash::hc_scratch_words`].
    #[allow(clippy::too_many_arguments)]
    fn hc_read_into(
        &self,
        r: &mut Self::Buffer,
        pending: Option<crate::flash::HcPending<'_, Self::Buffer>>,
        w: &crate::flash::HcWeights<'_, Self::Buffer>,
        x: &mut Self::Buffer,
        xq: Option<&mut Self::Buffer>,
        inj: Option<&mut Self::Buffer>,
        _scratch: &mut Self::Buffer,
        t: usize,
        eps: f32,
    ) {
        use crate::flash::HcPending;
        let mut rv = self.download(r);
        let pend = pending.map(|p| match p {
            HcPending::Write { y, inj } => (self.download(y), self.download(inj)),
            HcPending::Moe {
                parts,
                w,
                logits,
                stride,
                sg,
                inj,
            } => {
                let y = crate::cpu::flash::moe_combine(
                    &self.download(parts),
                    &self.download(w),
                    &self.download(logits),
                    stride,
                    sg,
                    t,
                );
                (y, self.download(inj))
            }
        });
        let (xv, iv) = crate::cpu::flash::hc_read(
            &mut rv,
            pend.as_ref().map(|(y, i)| (&y[..], &i[..])),
            &self.download(w.norm),
            &self.download(w.down),
            &self.download(w.up),
            w.inject.map(|b| self.download(b)).as_deref(),
            t,
            eps,
        );
        *r = self.upload_f32(&rv);
        if let Some(xq) = xq {
            *xq = self.upload_u32(&crate::cpu::flash::quantize_act(
                &xv,
                t,
                crate::flash::shape::HIDDEN,
            ));
        }
        *x = self.upload_f32(&xv);
        if let Some(inj) = inj {
            *inj = self.upload_f32(&iv);
        }
    }

    /// Hyper-connection write: `r[t][c] += y[t] · 2σ(inj[t][c] / HC)`.
    fn hc_write(&self, r: &mut Self::Buffer, y: &Self::Buffer, inj: &Self::Buffer, t: usize) {
        let mut rv = self.download(r);
        crate::cpu::flash::hc_write(&mut rv, &self.download(y), &self.download(inj), t);
        *r = self.upload_f32(&rv);
    }

    /// GDN conv for a window: reads the `qkv` columns of `proj` (`[t][stride]`) and the history
    /// `hist` ([`crate::flash::GDN_HIST`], not written), writes `h [t][GDN_CONV]`.
    #[allow(clippy::too_many_arguments)]
    fn gdn_conv_into(
        &self,
        proj: &Self::Buffer,
        stride: usize,
        hist: &Self::Buffer,
        conv: &Self::Buffer,
        h: &mut Self::Buffer,
        t: usize,
        eps: f32,
    ) {
        let y = crate::cpu::flash::gdn_conv(
            &self.download(proj),
            stride,
            &self.download(hist),
            &self.download(conv),
            t,
            eps,
        );
        *h = self.upload_f32(&y);
    }

    /// Commit the conv history: the last 3 rows of `[hist | qkv_0 .. qkv_{n−1}]`,
    /// `n = min(win[N_KEEP], t)`.
    fn gdn_conv_commit(
        &self,
        hist: &mut Self::Buffer,
        proj: &Self::Buffer,
        stride: usize,
        win: &Self::Buffer,
        t: usize,
    ) {
        let n = (self.download(win)[crate::flash::Win::N_KEEP].to_bits() as usize).min(t);
        let mut hv = self.download(hist);
        crate::cpu::flash::gdn_conv_commit(&mut hv, &self.download(proj), stride, n);
        *hist = self.upload_f32(&hv);
    }

    /// The GDN recurrence and gated output norm for a window ([`crate::flash::GDN_STATE`] for
    /// the math): `h` from [`gdn_conv_into`](Self::gdn_conv_into), `z`, `a`, `b` from the
    /// stacked projection `proj`; writes `y [t][GDN_V]` for the tokens it runs, and `yq` (when
    /// given) the same rows as int8 activations (`QAct { m: t, k: GDN_V }`).
    #[allow(clippy::too_many_arguments)]
    fn gdn_step(
        &self,
        state: &mut Self::Buffer,
        h: &Self::Buffer,
        proj: &Self::Buffer,
        stride: usize,
        p: &crate::flash::GdnParams<'_, Self::Buffer>,
        y: &mut Self::Buffer,
        yq: Option<&mut Self::Buffer>,
        t: usize,
        mode: crate::flash::GdnMode<'_, Self::Buffer>,
        eps: f32,
    ) {
        let mut sv = self.download(state);
        let (n_run, write) = match mode {
            crate::flash::GdnMode::ReadOnly => (t, false),
            crate::flash::GdnMode::Commit { win } => (
                (self.download(win)[crate::flash::Win::N_KEEP].to_bits() as usize).min(t),
                true,
            ),
        };
        let out = crate::cpu::flash::gdn_step(
            &mut sv,
            &self.download(h),
            &self.download(proj),
            stride,
            (
                &self.download(p.dt_bias),
                &self.download(p.ssm_a),
                &self.download(p.norm),
            ),
            t,
            n_run,
            write,
            eps,
        );
        if write {
            *state = self.upload_f32(&sv);
        }
        let mut yv = self.download(y);
        let n = n_run * crate::flash::shape::GDN_V;
        yv[..n].copy_from_slice(&out[..n]);
        if let Some(yq) = yq {
            let mut q = crate::flash::u32s(&self.download(yq));
            let l = crate::flash::QAct {
                m: t,
                k: crate::flash::shape::GDN_V,
            };
            q.resize(l.words(), 0);
            crate::cpu::flash::quantize_act_rows(&out, l, 0, n_run, &mut q);
            *yq = self.upload_u32(&q);
        }
        *y = self.upload_f32(&yv);
    }

    /// [`gdn_conv_into`](Self::gdn_conv_into) and [`gdn_step`](Self::gdn_step) in one step,
    /// bitwise the two: the conv runs from `proj`'s qkv columns, the history `hist` (not
    /// written) and `p.conv`. Run before [`gdn_conv_commit`](Self::gdn_conv_commit) in a commit,
    /// as the conv needs the history the window started from.
    #[allow(clippy::too_many_arguments)]
    fn gdn_conv_step(
        &self,
        state: &mut Self::Buffer,
        proj: &Self::Buffer,
        stride: usize,
        hist: &Self::Buffer,
        p: &crate::flash::GdnParams<'_, Self::Buffer>,
        y: &mut Self::Buffer,
        yq: Option<&mut Self::Buffer>,
        t: usize,
        mode: crate::flash::GdnMode<'_, Self::Buffer>,
        eps: f32,
    ) {
        let mut h = self.alloc_f32(t * crate::flash::shape::GDN_CONV);
        self.gdn_conv_into(proj, stride, hist, p.conv, &mut h, t, eps);
        self.gdn_step(state, &h, proj, stride, p, y, yq, t, mode, eps);
    }

    /// MoE router for a window: `logits [t][stride]` (experts in columns `0..n_expert`); writes
    /// `ids [t][TOPK]` (u32): the top `TOPK` by (logit desc, index asc); and `w [t][TOPK]`:
    /// `e_k / Σ_top e_j`, `e_k = exp(l_k − l_max)` in f64 summed in rank order, which is the
    /// full softmax renormalised over the top k with the 2^-14 clamp (unreachable for 10 of 512).
    fn router_topk_into(
        &self,
        logits: &Self::Buffer,
        stride: usize,
        n_expert: usize,
        ids: &mut Self::Buffer,
        w: &mut Self::Buffer,
        t: usize,
    ) {
        let (i, wv) = crate::cpu::flash::router_topk(&self.download(logits), stride, t, n_expert);
        *ids = self.upload_u32(&i);
        *w = self.upload_f32(&wv);
    }

    /// Build a [`crate::flash::MoePlan`] on the device from router ids and a residency table
    /// (`EXPERTS` addresses as word pairs, 0 = not resident); `shared` is the shared expert's
    /// blob address, or 0 for none.
    fn moe_plan_into(
        &self,
        ids: &Self::Buffer,
        table: &Self::Buffer,
        shared: u64,
        plan: &mut Self::Buffer,
        t: usize,
    ) {
        let tw = crate::flash::u32s(&self.download(table));
        let table: Vec<u64> = tw
            .chunks(2)
            .map(|c| c[0] as u64 | (c[1] as u64) << 32)
            .collect();
        let p = crate::cpu::flash::moe_plan(
            &crate::flash::u32s(&self.download(ids)),
            &table,
            shared,
            t,
        );
        *plan = self.upload_u32(&p);
    }

    /// Router and plan in one step: [`router_topk_into`](Self::router_topk_into) into `ids` and
    /// `w`, then [`moe_plan_into`](Self::moe_plan_into) from those ids, or from `forced` (a
    /// recorded routing `[t][TOPK]`, for replays and benchmarks) when given.
    #[allow(clippy::too_many_arguments)]
    fn moe_route_into(
        &self,
        logits: &Self::Buffer,
        stride: usize,
        n_expert: usize,
        forced: Option<&Self::Buffer>,
        table: &Self::Buffer,
        shared: u64,
        ids: &mut Self::Buffer,
        w: &mut Self::Buffer,
        plan: &mut Self::Buffer,
        t: usize,
    ) {
        self.router_topk_into(logits, stride, n_expert, ids, w, t);
        self.moe_plan_into(forced.unwrap_or(ids), table, shared, plan, t);
    }

    /// Evaluate the planned experts into `parts` ([`crate::flash::MoePlan::PARTS_ROWS`] rows of
    /// `HIDDEN`); `xq` holds the window's activations quantized as `QAct { m: t, k: HIDDEN }`.
    /// Each weight row of a group is read once for all of its entries. Rows the plan does not
    /// name are not written.
    ///
    /// # Safety
    /// Every group address in the plan must point at a live [`crate::flash::ExpertBlob`] that
    /// this device can read (VRAM, or device-mapped host memory; a host address on the CPU).
    unsafe fn moe_grouped_into(
        &self,
        xq: &Self::Buffer,
        plan: &Self::Buffer,
        _scratch: &mut Self::Buffer,
        parts: &mut Self::Buffer,
        t: usize,
    ) {
        let mut pv = self.download(parts);
        // SAFETY: forwarded from the caller.
        unsafe {
            crate::cpu::flash::moe_grouped(
                &crate::flash::u32s(&self.download(xq)),
                &crate::flash::u32s(&self.download(plan)),
                &mut pv,
                t,
            );
        }
        *parts = self.upload_f32(&pv);
    }

    /// MoE combine: `y[t] = Σ_i w[t][i] · parts[t·TOPK + i] + σ(logits[t][sg]) · shared row`.
    #[allow(clippy::too_many_arguments)]
    fn moe_combine_into(
        &self,
        parts: &Self::Buffer,
        w: &Self::Buffer,
        logits: &Self::Buffer,
        stride: usize,
        sg: Option<usize>,
        y: &mut Self::Buffer,
        t: usize,
    ) {
        let v = crate::cpu::flash::moe_combine(
            &self.download(parts),
            &self.download(w),
            &self.download(logits),
            stride,
            sg,
            t,
        );
        *y = self.upload_f32(&v);
    }

    /// QSA prologue for a window: from the stacked projection `proj [t][stride]`, writes the
    /// normed, rotated queries `q` ([`crate::flash::qsa_q_words`]), appends K/V to the cache,
    /// raw indexer keys to the ring, and pools every block the window completes.
    #[allow(clippy::too_many_arguments)]
    fn qsa_prep(
        &self,
        proj: &Self::Buffer,
        stride: usize,
        win: &Self::Buffer,
        norms: &crate::flash::QsaNorms<'_, Self::Buffer>,
        rope: (&Self::Buffer, &Self::Buffer),
        q: &mut Self::Buffer,
        cache: crate::flash::QsaCache<'_, Self::Buffer>,
        t: usize,
        eps: f32,
    ) {
        let pos0 = self.download(win)[crate::flash::Win::POS0].to_bits() as usize;
        let (mut qv, mut kc, mut vc, mut ring, mut pooled) = (
            self.download(q),
            self.download(cache.k),
            self.download(cache.v),
            self.download(cache.ring),
            self.download(cache.pooled),
        );
        crate::cpu::flash::qsa_prep(
            &self.download(proj),
            stride,
            pos0,
            (
                &self.download(norms.q),
                &self.download(norms.k),
                &self.download(norms.iq),
                &self.download(norms.ik),
            ),
            (&self.download(rope.0), &self.download(rope.1)),
            t,
            eps,
            &mut qv,
            (&mut kc, &mut vc, &mut ring, &mut pooled),
        );
        *q = self.upload_f32(&qv);
        *cache.k = self.upload_f32(&kc);
        *cache.v = self.upload_f32(&vc);
        *cache.ring = self.upload_f32(&ring);
        *cache.pooled = self.upload_f32(&pooled);
    }

    /// QSA selection for a window: scores `pooled` blocks against the indexer queries in `q`
    /// into `scores [t][max_blocks]`, then writes `ids [t][QSA_WIDTH]`
    /// ([`crate::flash::qsa_score_blocks`] for the contract). `max_blocks` = max context / 4.
    #[allow(clippy::too_many_arguments)]
    fn qsa_select_into(
        &self,
        pooled: &Self::Buffer,
        q: &Self::Buffer,
        win: &Self::Buffer,
        scores: &mut Self::Buffer,
        ids: &mut Self::Buffer,
        max_blocks: usize,
        t: usize,
    ) {
        use crate::flash::shape::*;
        let pos0 = self.download(win)[crate::flash::Win::POS0].to_bits() as usize;
        let qv = self.download(q);
        let mut sv = self.download(scores);
        crate::cpu::flash::qsa_scores(
            &self.download(pooled),
            &qv[t * QSA_HEADS * QSA_D..],
            pos0,
            t,
            max_blocks,
            &mut sv,
        );
        let iv = crate::cpu::flash::qsa_select(&sv, pos0, t, max_blocks);
        *scores = self.upload_f32(&sv);
        *ids = self.upload_u32(&iv);
    }

    /// [`qsa_select_into`](Self::qsa_select_into), and also the window's union of selected
    /// blocks into `union` ([`crate::flash::qsa_union_words`]; zero it once at allocation) for
    /// [`qsa_attend_union_into`](Self::qsa_attend_union_into). The default leaves `union`
    /// alone (the default attention reads `ids`).
    #[allow(clippy::too_many_arguments)]
    fn qsa_select_union_into(
        &self,
        pooled: &Self::Buffer,
        q: &Self::Buffer,
        win: &Self::Buffer,
        scores: &mut Self::Buffer,
        ids: &mut Self::Buffer,
        _union: &mut Self::Buffer,
        max_blocks: usize,
        t: usize,
    ) {
        self.qsa_select_into(pooled, q, win, scores, ids, max_blocks, t);
    }

    /// [`qsa_attend_into`](Self::qsa_attend_into) computed over the window's union of
    /// selections: every token still attends to exactly its own selected cells (cells it did
    /// not select score −∞), but each key and value row is read once per window instead of
    /// once per token. Same result up to fp32 summation order. The default runs
    /// `qsa_attend_into` on `ids`. On mew (3090, 4K context) the CUDA version is slower than
    /// per-token `qsa_attend_into` (T = 4: 2.15 vs 0.78 ms for 12 layers): concurrent per-token
    /// blocks already share the rows through L2, and the union kernel walks the tokens serially.
    #[allow(clippy::too_many_arguments)]
    fn qsa_attend_union_into(
        &self,
        q: &Self::Buffer,
        k_cache: &Self::Buffer,
        v_cache: &Self::Buffer,
        ids: &Self::Buffer,
        _union: &Self::Buffer,
        _max_blocks: usize,
        proj: &Self::Buffer,
        stride: usize,
        win: &Self::Buffer,
        scratch: &mut Self::Buffer,
        out: &mut Self::Buffer,
        outq: Option<&mut Self::Buffer>,
        t: usize,
    ) {
        self.qsa_attend_into(
            q, k_cache, v_cache, ids, proj, stride, win, scratch, out, outq, t,
        );
    }

    /// QSA attention over the selected cells with the sigmoid output gate (read from `proj`):
    /// writes `out [t][QSA_OUT]` and `outq` (when given) as int8 activations. `scratch`: [`crate::flash::qsa_attend_scratch_words`].
    #[allow(clippy::too_many_arguments)]
    fn qsa_attend_into(
        &self,
        q: &Self::Buffer,
        k_cache: &Self::Buffer,
        v_cache: &Self::Buffer,
        ids: &Self::Buffer,
        proj: &Self::Buffer,
        stride: usize,
        win: &Self::Buffer,
        _scratch: &mut Self::Buffer,
        out: &mut Self::Buffer,
        outq: Option<&mut Self::Buffer>,
        t: usize,
    ) {
        let pos0 = self.download(win)[crate::flash::Win::POS0].to_bits() as usize;
        let o = crate::cpu::flash::qsa_attend(
            &self.download(q),
            &self.download(k_cache),
            &self.download(v_cache),
            &crate::flash::u32s(&self.download(ids)),
            &self.download(proj),
            stride,
            pos0,
            t,
        );
        if let Some(q) = outq {
            *q = self.upload_u32(&crate::cpu::flash::quantize_act(
                &o,
                t,
                crate::flash::shape::QSA_OUT,
            ));
        }
        *out = self.upload_f32(&o);
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

/// Composition of primitives behind [`ComputeDevice::attention_prep`] (backends that fuse it
/// fall back to this for shapes their kernel doesn't cover).
#[allow(clippy::too_many_arguments)]
pub fn attention_prep_default<D: ComputeDevice + ?Sized>(
    dev: &D,
    qkv: &D::Buffer,
    q_norm: Option<&D::Buffer>,
    k_norm: Option<&D::Buffer>,
    cos: &D::Buffer,
    sin: &D::Buffer,
    k_cache: &mut D::Buffer,
    v_cache: &mut D::Buffer,
    seq: usize,
    (nh, nkv, hd): (usize, usize, usize),
    pos: usize,
    eps: f32,
) -> D::Buffer {
    let (qd, kvd) = (nh * hd, nkv * hd);
    let row = qd + 2 * kvd;
    let all = dev.download(qkv);
    let part = |off: usize, w: usize| -> Vec<f32> {
        (0..seq)
            .flat_map(|s| all[s * row + off..s * row + off + w].to_vec())
            .collect()
    };
    let (mut q, mut k) = (dev.upload(&part(0, qd)), dev.upload(&part(qd, kvd)));
    let v = dev.upload(&part(qd + kvd, kvd));
    if let Some(n) = q_norm {
        q = dev.rms_norm(&q, n, seq * nh, hd, eps);
    }
    if let Some(n) = k_norm {
        k = dev.rms_norm(&k, n, seq * nkv, hd, eps);
    }
    let q = dev.rope_half_cached(&q, cos, sin, seq, nh, hd, pos);
    let k = dev.rope_half_cached(&k, cos, sin, seq, nkv, hd, pos);
    dev.write_into(k_cache, pos * kvd, &k);
    dev.write_into(v_cache, pos * kvd, &v);
    q
}
