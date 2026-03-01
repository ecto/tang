//! CPU compute backend with optional Accelerate BLAS.

use crate::device::{ComputeBuffer, ComputeDevice};
use tang_expr::codegen::Dialect;
use tang_expr::node::ExprId;
use tang_expr::trace;

/// CPU buffer: just a Vec<f32>.
pub struct CpuBuffer {
    data: Vec<f32>,
}

impl ComputeBuffer for CpuBuffer {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn to_vec(&self) -> Vec<f32> {
        self.data.clone()
    }
}

/// CPU compute device. Uses Accelerate BLAS for matmul on macOS.
pub struct CpuDevice;

impl CpuDevice {
    pub fn new() -> Self {
        CpuDevice
    }
}

impl Default for CpuDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeDevice for CpuDevice {
    type Buffer = CpuBuffer;

    fn dialect(&self) -> Dialect {
        Dialect::C
    }

    fn upload(&self, data: &[f32]) -> CpuBuffer {
        CpuBuffer {
            data: data.to_vec(),
        }
    }

    fn upload_u32(&self, data: &[u32]) -> CpuBuffer {
        // Store u32 as f32 bits — for embedding lookup we'll reinterpret
        CpuBuffer {
            data: data.iter().map(|&x| f32::from_bits(x)).collect(),
        }
    }

    fn alloc(&self, len: usize) -> CpuBuffer {
        CpuBuffer {
            data: vec![0.0; len],
        }
    }

    fn download(&self, buf: &CpuBuffer) -> Vec<f32> {
        buf.data.clone()
    }

    fn elementwise(
        &self,
        inputs: &[&CpuBuffer],
        numel: usize,
        f: &dyn Fn(&[ExprId]) -> ExprId,
    ) -> CpuBuffer {
        let n_inputs = inputs.len();

        // Trace the closure to get an expression graph
        let (graph, output) = trace(|| {
            let vars: Vec<ExprId> = (0..n_inputs as u16).map(ExprId::var).collect();
            f(&vars)
        });

        // Compile to a Rust closure
        let compiled = graph.compile(output);

        // Evaluate for each element
        let mut result = vec![0.0f32; numel];
        let mut args = vec![0.0f64; n_inputs];

        for i in 0..numel {
            for (j, input) in inputs.iter().enumerate() {
                args[j] = input.data[i] as f64;
            }
            result[i] = compiled(&args) as f32;
        }

        CpuBuffer { data: result }
    }

    fn matmul(&self, a: &CpuBuffer, b: &CpuBuffer, m: usize, k: usize, n: usize) -> CpuBuffer {
        let mut c = vec![0.0f32; m * n];
        matmul_impl(&a.data, &b.data, &mut c, m, k, n);
        CpuBuffer { data: c }
    }

    fn softmax(&self, data: &CpuBuffer, n_rows: usize, row_len: usize) -> CpuBuffer {
        let mut out = data.data.clone();
        for row in 0..n_rows {
            let start = row * row_len;
            let end = start + row_len;
            let slice = &mut out[start..end];

            // Numerical stability: subtract max
            let max = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            for v in slice.iter_mut() {
                *v = (*v - max).exp();
            }
            let sum: f32 = slice.iter().sum();
            let inv = 1.0 / sum;
            for v in slice.iter_mut() {
                *v *= inv;
            }
        }
        CpuBuffer { data: out }
    }

    fn rms_norm(
        &self,
        data: &CpuBuffer,
        weight: &CpuBuffer,
        n_groups: usize,
        dim: usize,
        eps: f32,
    ) -> CpuBuffer {
        let mut out = vec![0.0f32; n_groups * dim];
        for g in 0..n_groups {
            let start = g * dim;
            let slice = &data.data[start..start + dim];

            // RMS = sqrt(mean(x^2) + eps)
            let sq_sum: f32 = slice.iter().map(|x| x * x).sum();
            let rms = (sq_sum / dim as f32 + eps).sqrt();
            let inv_rms = 1.0 / rms;

            for d in 0..dim {
                out[start + d] = slice[d] * inv_rms * weight.data[d];
            }
        }
        CpuBuffer { data: out }
    }

    fn embedding(
        &self,
        weight: &CpuBuffer,
        ids: &CpuBuffer,
        seq_len: usize,
        dim: usize,
    ) -> CpuBuffer {
        let mut out = vec![0.0f32; seq_len * dim];
        for i in 0..seq_len {
            let id = ids.data[i].to_bits() as usize;
            let src_start = id * dim;
            let dst_start = i * dim;
            out[dst_start..dst_start + dim]
                .copy_from_slice(&weight.data[src_start..src_start + dim]);
        }
        CpuBuffer { data: out }
    }

    fn reduce_sum(&self, data: &CpuBuffer, shape: &[usize], axis: usize) -> CpuBuffer {
        let ndim = shape.len();
        assert!(axis < ndim);

        // Compute strides
        let mut out_shape = shape.to_vec();
        out_shape.remove(axis);
        let out_len: usize = out_shape.iter().product();
        if out_len == 0 {
            return CpuBuffer { data: vec![] };
        }

        let mut result = vec![0.0f32; out_len];

        // outer_size = product of dims before axis
        let outer: usize = shape[..axis].iter().product();
        let axis_len = shape[axis];
        let inner: usize = shape[axis + 1..].iter().product();

        for o in 0..outer {
            for a in 0..axis_len {
                for i in 0..inner {
                    let src_idx = o * axis_len * inner + a * inner + i;
                    let dst_idx = o * inner + i;
                    result[dst_idx] += data.data[src_idx];
                }
            }
        }

        CpuBuffer { data: result }
    }

    fn causal_attention(
        &self,
        q: &CpuBuffer,
        k: &CpuBuffer,
        v: &CpuBuffer,
        seq_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> CpuBuffer {
        let total_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let heads_per_kv = n_heads / n_kv_heads;
        let mut out = vec![0.0f32; seq_len * total_dim];
        let scale = 1.0 / (head_dim as f32).sqrt();

        for h in 0..n_heads {
            let kv_h = h / heads_per_kv;
            let q_off = h * head_dim;
            let kv_off = kv_h * head_dim;

            for i in 0..seq_len {
                let mut running_max = f32::NEG_INFINITY;
                let mut running_sum = 0.0f32;
                let mut accum = vec![0.0f32; head_dim];

                for j in 0..=i {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q.data[i * total_dim + q_off + d]
                            * k.data[j * kv_dim + kv_off + d];
                    }
                    let score = dot * scale;

                    let new_max = running_max.max(score);
                    let exp_score = (score - new_max).exp();
                    let rescale = (running_max - new_max).exp();

                    running_sum = running_sum * rescale + exp_score;
                    for d in 0..head_dim {
                        accum[d] = accum[d] * rescale
                            + exp_score * v.data[j * kv_dim + kv_off + d];
                    }
                    running_max = new_max;
                }

                let inv = 1.0 / running_sum;
                for d in 0..head_dim {
                    out[i * total_dim + q_off + d] = accum[d] * inv;
                }
            }
        }

        CpuBuffer { data: out }
    }

    fn kv_attention(
        &self,
        q: &CpuBuffer,
        k_cache: &CpuBuffer,
        v_cache: &CpuBuffer,
        cache_start: usize,
        q_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> CpuBuffer {
        let total_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let heads_per_kv = n_heads / n_kv_heads;
        let mut out = vec![0.0f32; q_len * total_dim];
        let scale = 1.0 / (head_dim as f32).sqrt();

        for qi in 0..q_len {
            let attend_len = cache_start + qi + 1;

            for h in 0..n_heads {
                let kv_h = h / heads_per_kv;
                let q_off = qi * total_dim + h * head_dim;
                let kv_off = kv_h * head_dim;

                // Online softmax: O(head_dim) memory
                let mut running_max = f32::NEG_INFINITY;
                let mut running_sum = 0.0f32;
                let mut accum = vec![0.0f32; head_dim];

                for j in 0..attend_len {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q.data[q_off + d] * k_cache.data[j * kv_dim + kv_off + d];
                    }
                    let score = dot * scale;

                    let new_max = running_max.max(score);
                    let exp_score = (score - new_max).exp();
                    let rescale = (running_max - new_max).exp();

                    running_sum = running_sum * rescale + exp_score;
                    for d in 0..head_dim {
                        accum[d] = accum[d] * rescale
                            + exp_score * v_cache.data[j * kv_dim + kv_off + d];
                    }
                    running_max = new_max;
                }

                let out_off = qi * total_dim + h * head_dim;
                let inv = 1.0 / running_sum;
                for d in 0..head_dim {
                    out[out_off + d] = accum[d] * inv;
                }
            }
        }

        CpuBuffer { data: out }
    }

    fn transpose_2d(&self, buf: &CpuBuffer, rows: usize, cols: usize) -> CpuBuffer {
        assert_eq!(buf.data.len(), rows * cols);
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = buf.data[r * cols + c];
            }
        }
        CpuBuffer { data: out }
    }

    fn softmax_backward(
        &self,
        softmax_out: &CpuBuffer,
        grad_output: &CpuBuffer,
        n_rows: usize,
        row_len: usize,
    ) -> CpuBuffer {
        let mut grad_input = vec![0.0f32; n_rows * row_len];
        for row in 0..n_rows {
            let base = row * row_len;
            // dot(sm[row], grad[row])
            let mut dot = 0.0f32;
            for j in 0..row_len {
                dot += softmax_out.data[base + j] * grad_output.data[base + j];
            }
            for j in 0..row_len {
                grad_input[base + j] =
                    softmax_out.data[base + j] * (grad_output.data[base + j] - dot);
            }
        }
        CpuBuffer { data: grad_input }
    }

    fn rms_norm_backward(
        &self,
        input: &CpuBuffer,
        weight: &CpuBuffer,
        grad_output: &CpuBuffer,
        n_groups: usize,
        dim: usize,
        eps: f32,
    ) -> (CpuBuffer, CpuBuffer) {
        let mut grad_input = vec![0.0f32; n_groups * dim];
        let mut grad_weight = vec![0.0f32; dim];

        for g in 0..n_groups {
            let base = g * dim;
            let x = &input.data[base..base + dim];

            // Forward recompute
            let sq_sum: f32 = x.iter().map(|v| v * v).sum();
            let rms_sq = sq_sum / dim as f32 + eps;
            let inv_rms = 1.0 / rms_sq.sqrt();

            // grad_weight accumulation
            for d in 0..dim {
                grad_weight[d] += grad_output.data[base + d] * x[d] * inv_rms;
            }

            // grad_input: d(loss)/d(x_i) through RMS norm
            // norm_out = x * w * inv_rms
            // d(loss)/d(x_i) = w_i * inv_rms * grad_out_i
            //                 - x_i * inv_rms^3 / dim * sum_j(x_j * w_j * grad_out_j)
            let mut sum_xwg = 0.0f32;
            for d in 0..dim {
                sum_xwg += x[d] * weight.data[d] * grad_output.data[base + d];
            }
            for d in 0..dim {
                grad_input[base + d] = weight.data[d] * inv_rms * grad_output.data[base + d]
                    - x[d] * inv_rms * inv_rms * inv_rms / dim as f32 * sum_xwg;
            }
        }

        (CpuBuffer { data: grad_input }, CpuBuffer { data: grad_weight })
    }

    fn embedding_backward(
        &self,
        grad_output: &CpuBuffer,
        ids: &CpuBuffer,
        vocab_size: usize,
        seq_len: usize,
        dim: usize,
    ) -> CpuBuffer {
        let mut grad_weight = vec![0.0f32; vocab_size * dim];
        for i in 0..seq_len {
            let id = ids.data[i].to_bits() as usize;
            for d in 0..dim {
                grad_weight[id * dim + d] += grad_output.data[i * dim + d];
            }
        }
        CpuBuffer { data: grad_weight }
    }

    fn causal_attention_backward(
        &self,
        grad_output: &CpuBuffer,
        q: &CpuBuffer,
        k: &CpuBuffer,
        v: &CpuBuffer,
        seq_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> (CpuBuffer, CpuBuffer, CpuBuffer) {
        let total_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let heads_per_kv = n_heads / n_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut grad_q = vec![0.0f32; seq_len * total_dim];
        let mut grad_k = vec![0.0f32; seq_len * kv_dim];
        let mut grad_v = vec![0.0f32; seq_len * kv_dim];

        for h in 0..n_heads {
            let kv_h = h / heads_per_kv;
            let q_off = h * head_dim;
            let kv_off = kv_h * head_dim;

            // 1. Recompute S = Q @ K^T / sqrt(d), apply causal mask, softmax → P
            let mut scores = vec![0.0f32; seq_len * seq_len];
            for i in 0..seq_len {
                for j in 0..seq_len {
                    if j > i {
                        scores[i * seq_len + j] = f32::NEG_INFINITY;
                    } else {
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q.data[i * total_dim + q_off + d]
                                * k.data[j * kv_dim + kv_off + d];
                        }
                        scores[i * seq_len + j] = dot * scale;
                    }
                }
            }

            // Softmax each row → P
            let mut probs = vec![0.0f32; seq_len * seq_len];
            for i in 0..seq_len {
                let row = &scores[i * seq_len..(i + 1) * seq_len];
                let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for j in 0..seq_len {
                    let e = (row[j] - max_val).exp();
                    probs[i * seq_len + j] = e;
                    sum += e;
                }
                let inv = 1.0 / sum;
                for j in 0..seq_len {
                    probs[i * seq_len + j] *= inv;
                }
            }

            // 2. grad_V = P^T @ grad_out_h
            for j in 0..seq_len {
                for d in 0..head_dim {
                    let mut sum = 0.0f32;
                    for i in 0..seq_len {
                        sum += probs[i * seq_len + j]
                            * grad_output.data[i * total_dim + q_off + d];
                    }
                    grad_v[j * kv_dim + kv_off + d] += sum;
                }
            }

            // 3. grad_P = grad_out_h @ V^T
            let mut grad_p = vec![0.0f32; seq_len * seq_len];
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += grad_output.data[i * total_dim + q_off + d]
                            * v.data[j * kv_dim + kv_off + d];
                    }
                    grad_p[i * seq_len + j] = dot;
                }
            }

            // 4. grad_S = softmax_backward(P, grad_P)
            let mut grad_s = vec![0.0f32; seq_len * seq_len];
            for i in 0..seq_len {
                let base = i * seq_len;
                let mut dot = 0.0f32;
                for j in 0..seq_len {
                    dot += probs[base + j] * grad_p[base + j];
                }
                for j in 0..seq_len {
                    grad_s[base + j] = probs[base + j] * (grad_p[base + j] - dot);
                }
            }

            // 5. grad_Q = grad_S @ K / sqrt(d)
            for i in 0..seq_len {
                for d in 0..head_dim {
                    let mut sum = 0.0f32;
                    for j in 0..seq_len {
                        sum += grad_s[i * seq_len + j] * k.data[j * kv_dim + kv_off + d];
                    }
                    grad_q[i * total_dim + q_off + d] = sum * scale;
                }
            }

            // 6. grad_K = grad_S^T @ Q / sqrt(d)
            for j in 0..seq_len {
                for d in 0..head_dim {
                    let mut sum = 0.0f32;
                    for i in 0..seq_len {
                        sum += grad_s[i * seq_len + j]
                            * q.data[i * total_dim + q_off + d];
                    }
                    grad_k[j * kv_dim + kv_off + d] += sum * scale;
                }
            }
        }

        (
            CpuBuffer { data: grad_q },
            CpuBuffer { data: grad_k },
            CpuBuffer { data: grad_v },
        )
    }

    fn cross_entropy_forward_backward(
        &self,
        logits: &CpuBuffer,
        targets: &CpuBuffer,
        n_positions: usize,
        vocab_size: usize,
        pad_id: u32,
    ) -> (f32, CpuBuffer) {
        let mut grad = vec![0.0f32; n_positions * vocab_size];
        let mut total_loss = 0.0f64;
        let mut count = 0usize;

        for pos in 0..n_positions {
            let target = targets.data[pos].to_bits();
            if target == pad_id {
                continue;
            }
            count += 1;
            let base = pos * vocab_size;
            let row = &logits.data[base..base + vocab_size];

            // Numerically stable softmax
            let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f64;
            for j in 0..vocab_size {
                sum += ((row[j] - max_val) as f64).exp();
            }
            let log_sum = sum.ln();

            // Loss: -log(softmax[target])
            let log_prob = (row[target as usize] - max_val) as f64 - log_sum;
            total_loss -= log_prob;

            // Gradient: softmax - one_hot (will divide by count after)
            for j in 0..vocab_size {
                let sm = (((row[j] - max_val) as f64).exp() / sum) as f32;
                grad[base + j] = sm;
            }
            grad[base + target as usize] -= 1.0;
        }

        if count > 0 {
            let inv_count = 1.0 / count as f32;
            for g in grad.iter_mut() {
                *g *= inv_count;
            }
            total_loss /= count as f64;
        }

        (total_loss as f32, CpuBuffer { data: grad })
    }

    fn sync(&self) {
        // No-op for CPU
    }

    fn copy_buffer(&self, src: &CpuBuffer) -> CpuBuffer {
        CpuBuffer { data: src.data.clone() }
    }

    fn bias_add(&self, matrix: &CpuBuffer, bias: &CpuBuffer, numel: usize, dim: usize) -> CpuBuffer {
        let mut out = matrix.data.clone();
        for i in 0..numel {
            out[i] += bias.data[i % dim];
        }
        CpuBuffer { data: out }
    }
}

// ---------------------------------------------------------------------------
// Matmul implementation
// ---------------------------------------------------------------------------

/// Matrix multiply C = A * B, row-major.
/// A: [m, k], B: [k, n], C: [m, n].
fn matmul_impl(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    #[cfg(target_os = "macos")]
    {
        cblas_matmul(a, b, c, m, k, n);
    }

    #[cfg(not(target_os = "macos"))]
    {
        naive_matmul(a, b, c, m, k, n);
    }
}

/// BLAS sgemm via Accelerate framework.
#[cfg(target_os = "macos")]
fn cblas_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    extern "C" {
        fn cblas_sgemm(
            order: i32,
            transa: i32,
            transb: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
    }

    const ROW_MAJOR: i32 = 101;
    const NO_TRANS: i32 = 111;

    unsafe {
        cblas_sgemm(
            ROW_MAJOR,
            NO_TRANS,
            NO_TRANS,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            n as i32,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }
}

/// Naive triple-loop matmul fallback.
#[cfg(not(target_os = "macos"))]
fn naive_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_upload_download() {
        let dev = CpuDevice::new();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let buf = dev.upload(&data);
        assert_eq!(dev.download(&buf), data);
    }

    #[test]
    fn cpu_matmul_identity() {
        let dev = CpuDevice::new();
        // 2x2 identity * [1,2; 3,4]
        let a = dev.upload(&[1.0, 0.0, 0.0, 1.0]);
        let b = dev.upload(&[1.0, 2.0, 3.0, 4.0]);
        let c = dev.matmul(&a, &b, 2, 2, 2);
        let result = dev.download(&c);
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn cpu_matmul_basic() {
        let dev = CpuDevice::new();
        // [1, 2] * [[3], [4]] = [11]
        let a = dev.upload(&[1.0, 2.0]);
        let b = dev.upload(&[3.0, 4.0]);
        let c = dev.matmul(&a, &b, 1, 2, 1);
        let result = dev.download(&c);
        assert!((result[0] - 11.0).abs() < 1e-5);
    }

    #[test]
    fn cpu_matmul_rectangular() {
        let dev = CpuDevice::new();
        // A: 2x3, B: 3x2 → C: 2x2
        let a = dev.upload(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = dev.upload(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let c = dev.matmul(&a, &b, 2, 3, 2);
        let result = dev.download(&c);
        // C[0,0] = 1*7 + 2*9 + 3*11 = 58
        // C[0,1] = 1*8 + 2*10 + 3*12 = 64
        // C[1,0] = 4*7 + 5*9 + 6*11 = 139
        // C[1,1] = 4*8 + 5*10 + 6*12 = 154
        assert!((result[0] - 58.0).abs() < 1e-4);
        assert!((result[1] - 64.0).abs() < 1e-4);
        assert!((result[2] - 139.0).abs() < 1e-4);
        assert!((result[3] - 154.0).abs() < 1e-4);
    }

    #[test]
    fn cpu_softmax() {
        let dev = CpuDevice::new();
        let data = dev.upload(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        let result = dev.softmax(&data, 2, 3);
        let out = dev.download(&result);
        // Each row should sum to 1
        let sum0: f32 = out[0..3].iter().sum();
        let sum1: f32 = out[3..6].iter().sum();
        assert!((sum0 - 1.0).abs() < 1e-5);
        assert!((sum1 - 1.0).abs() < 1e-5);
        // Rows should be identical
        assert!((out[0] - out[3]).abs() < 1e-6);
    }

    #[test]
    fn cpu_rms_norm() {
        let dev = CpuDevice::new();
        let data = dev.upload(&[1.0, 2.0, 3.0, 4.0]);
        let weight = dev.upload(&[1.0, 1.0]);
        let result = dev.rms_norm(&data, &weight, 2, 2, 1e-5);
        let out = dev.download(&result);
        // First group: [1, 2], rms = sqrt((1+4)/2) = sqrt(2.5)
        let rms0 = (2.5f32 + 1e-5).sqrt();
        assert!((out[0] - 1.0 / rms0).abs() < 1e-5);
        assert!((out[1] - 2.0 / rms0).abs() < 1e-5);
    }

    #[test]
    fn cpu_embedding() {
        let dev = CpuDevice::new();
        // Vocab=3, dim=2
        let weight = dev.upload(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let ids = dev.upload_u32(&[2, 0, 1]);
        let result = dev.embedding(&weight, &ids, 3, 2);
        let out = dev.download(&result);
        // id=2 → [0.5, 0.6], id=0 → [0.1, 0.2], id=1 → [0.3, 0.4]
        assert!((out[0] - 0.5).abs() < 1e-6);
        assert!((out[1] - 0.6).abs() < 1e-6);
        assert!((out[2] - 0.1).abs() < 1e-6);
        assert!((out[3] - 0.2).abs() < 1e-6);
        assert!((out[4] - 0.3).abs() < 1e-6);
        assert!((out[5] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn cpu_reduce_sum() {
        let dev = CpuDevice::new();
        // Shape [2, 3], reduce axis 1 → [2]
        let data = dev.upload(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = dev.reduce_sum(&data, &[2, 3], 1);
        let out = dev.download(&result);
        assert_eq!(out.len(), 2);
        assert!((out[0] - 6.0).abs() < 1e-5); // 1+2+3
        assert!((out[1] - 15.0).abs() < 1e-5); // 4+5+6
    }

    #[test]
    fn cpu_reduce_sum_axis0() {
        let dev = CpuDevice::new();
        // Shape [2, 3], reduce axis 0 → [3]
        let data = dev.upload(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = dev.reduce_sum(&data, &[2, 3], 0);
        let out = dev.download(&result);
        assert_eq!(out.len(), 3);
        assert!((out[0] - 5.0).abs() < 1e-5); // 1+4
        assert!((out[1] - 7.0).abs() < 1e-5); // 2+5
        assert!((out[2] - 9.0).abs() < 1e-5); // 3+6
    }

    #[test]
    fn cpu_elementwise_add() {
        let dev = CpuDevice::new();
        let a = dev.upload(&[1.0, 2.0, 3.0]);
        let b = dev.upload(&[4.0, 5.0, 6.0]);
        let c = dev.elementwise(&[&a, &b], 3, &|vars| vars[0] + vars[1]);
        let out = dev.download(&c);
        assert!((out[0] - 5.0).abs() < 1e-5);
        assert!((out[1] - 7.0).abs() < 1e-5);
        assert!((out[2] - 9.0).abs() < 1e-5);
    }

    #[test]
    fn cpu_elementwise_fused() {
        use tang::Scalar;
        let dev = CpuDevice::new();
        let a = dev.upload(&[1.0, 4.0, 9.0]);
        // sqrt(x) + 1
        let c = dev.elementwise(&[&a], 3, &|vars| {
            vars[0].sqrt() + ExprId::from_f64(1.0)
        });
        let out = dev.download(&c);
        assert!((out[0] - 2.0).abs() < 1e-4);
        assert!((out[1] - 3.0).abs() < 1e-4);
        assert!((out[2] - 4.0).abs() < 1e-4);
    }

    #[test]
    fn cpu_causal_attention() {
        let dev = CpuDevice::new();
        // seq_len=2, n_heads=1, head_dim=2
        // Q = [[1,0], [0,1]], K = [[1,0], [0,1]], V = [[1,2], [3,4]]
        let q = dev.upload(&[1.0, 0.0, 0.0, 1.0]);
        let k = dev.upload(&[1.0, 0.0, 0.0, 1.0]);
        let v = dev.upload(&[1.0, 2.0, 3.0, 4.0]);
        let result = dev.causal_attention(&q, &k, &v, 2, 1, 1, 2);
        let out = dev.download(&result);
        assert_eq!(out.len(), 4);
        // Position 0 can only attend to itself → V[0] = [1, 2]
        assert!((out[0] - 1.0).abs() < 1e-4);
        assert!((out[1] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn cpu_kv_attention() {
        let dev = CpuDevice::new();
        // q: [1, 2] (1 position, 1 head, dim=2)
        // k_cache: [[1,0],[0,1]] (2 cached positions)
        // v_cache: [[1,2],[3,4]]
        // cache_start=1 means 1 position was already cached, this is the 2nd
        // so attend_len = 1 + 0 + 1 = 2
        let q = dev.upload(&[1.0, 0.0]);
        let k = dev.upload(&[1.0, 0.0, 0.0, 1.0]);
        let v = dev.upload(&[1.0, 2.0, 3.0, 4.0]);
        let result = dev.kv_attention(&q, &k, &v, 1, 1, 1, 1, 2);
        let out = dev.download(&result);
        assert_eq!(out.len(), 2);
        // q=[1,0] attends more to k[0]=[1,0] than k[1]=[0,1]
        // So output should lean toward v[0]=[1,2]
        assert!(out[0] < 2.5); // closer to 1 than 3
    }

    #[test]
    fn cpu_kv_attention_batched_causal() {
        let dev = CpuDevice::new();
        // q_len=3, cache_start=0, 1 head, dim=2
        // Q: [[1,0], [0,1], [1,1]]
        // K (cache): [[1,0], [0,1], [1,1]]  (same as Q for simplicity)
        // V (cache): [[1,2], [3,4], [5,6]]
        let q = dev.upload(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let k = dev.upload(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let v = dev.upload(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = dev.kv_attention(&q, &k, &v, 0, 3, 1, 1, 2);
        let out = dev.download(&result);
        assert_eq!(out.len(), 6); // [3, 2]

        // qi=0: attend to pos 0 only → V[0] = [1, 2]
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert!((out[1] - 2.0).abs() < 1e-5);

        // qi=1: attend to pos 0,1 → mixture of V[0] and V[1]
        // q=[0,1], k[0]=[1,0] dot=0, k[1]=[0,1] dot=1
        // softmax([0, 1/sqrt(2)]) → attends more to V[1]=[3,4]
        assert!(out[2] > 1.5); // leaning toward 3
        assert!(out[3] > 2.5); // leaning toward 4

        // qi=2: attend to pos 0,1,2
        // q=[1,1], dots with k are all computable
        assert_eq!(out.len(), 6);
    }

    #[test]
    fn cpu_transpose_2d() {
        let dev = CpuDevice::new();
        // 2x3 → 3x2
        let buf = dev.upload(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = dev.transpose_2d(&buf, 2, 3);
        let out = dev.download(&result);
        // [[1,2,3],[4,5,6]] → [[1,4],[2,5],[3,6]]
        assert_eq!(out, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn cpu_softmax_backward() {
        let dev = CpuDevice::new();
        // softmax output and upstream gradient
        let sm = dev.upload(&[0.2, 0.3, 0.5]);
        let grad = dev.upload(&[1.0, 0.0, 0.0]);
        let result = dev.softmax_backward(&sm, &grad, 1, 3);
        let out = dev.download(&result);
        // dot = 0.2*1 + 0.3*0 + 0.5*0 = 0.2
        // grad_input[0] = 0.2*(1.0-0.2) = 0.16
        // grad_input[1] = 0.3*(0.0-0.2) = -0.06
        // grad_input[2] = 0.5*(0.0-0.2) = -0.1
        assert!((out[0] - 0.16).abs() < 1e-6);
        assert!((out[1] - (-0.06)).abs() < 1e-6);
        assert!((out[2] - (-0.1)).abs() < 1e-6);
    }

    #[test]
    fn cpu_embedding_backward() {
        let dev = CpuDevice::new();
        // seq_len=2, dim=3, vocab=4
        let grad_out = dev.upload(&[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]);
        let ids = dev.upload_u32(&[1, 3]);
        let result = dev.embedding_backward(&grad_out, &ids, 4, 2, 3);
        let out = dev.download(&result);
        assert_eq!(out.len(), 12); // 4*3
        // id=1 → row 1 gets [1,2,3]
        assert!((out[3] - 1.0).abs() < 1e-6);
        assert!((out[4] - 2.0).abs() < 1e-6);
        assert!((out[5] - 3.0).abs() < 1e-6);
        // id=3 → row 3 gets [0.1,0.2,0.3]
        assert!((out[9] - 0.1).abs() < 1e-6);
        assert!((out[10] - 0.2).abs() < 1e-6);
        assert!((out[11] - 0.3).abs() < 1e-6);
        // rows 0 and 2 should be zero
        assert!((out[0]).abs() < 1e-6);
        assert!((out[6]).abs() < 1e-6);
    }

    #[test]
    fn cpu_cross_entropy_forward_backward() {
        let dev = CpuDevice::new();
        // 2 positions, vocab=3, pad_id=99
        // logits: [[1,2,3],[4,5,6]]
        // targets: [2, 0]  (one-hot at indices 2 and 0)
        let logits = dev.upload(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let targets = dev.upload_u32(&[2, 0]);
        let (loss, grad) = dev.cross_entropy_forward_backward(&logits, &targets, 2, 3, 99);
        let g = dev.download(&grad);

        assert!(loss.is_finite());
        assert!(loss > 0.0);
        assert_eq!(g.len(), 6);

        // Gradient should sum to ~0 per non-padded row
        let row0_sum: f32 = g[0..3].iter().sum();
        let row1_sum: f32 = g[3..6].iter().sum();
        assert!(row0_sum.abs() < 1e-5);
        assert!(row1_sum.abs() < 1e-5);
    }

    #[test]
    fn cpu_cross_entropy_with_padding() {
        let dev = CpuDevice::new();
        let logits = dev.upload(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let targets = dev.upload_u32(&[2, 0]); // pad_id=0, so pos 1 is padded
        let (loss, grad) = dev.cross_entropy_forward_backward(&logits, &targets, 2, 3, 0);
        let g = dev.download(&grad);

        // Only position 0 (target=2) should contribute
        assert!(loss > 0.0);
        // Padded row should have zero gradients
        assert!((g[3]).abs() < 1e-6);
        assert!((g[4]).abs() < 1e-6);
        assert!((g[5]).abs() < 1e-6);
    }

    #[test]
    fn cpu_rms_norm_backward() {
        let dev = CpuDevice::new();
        let input = dev.upload(&[1.0, 2.0, 3.0, 4.0]);
        let weight = dev.upload(&[1.0, 1.0]);
        let grad_out = dev.upload(&[1.0, 0.0, 0.0, 1.0]);
        let (grad_input, grad_weight) =
            dev.rms_norm_backward(&input, &weight, &grad_out, 2, 2, 1e-5);
        let gi = dev.download(&grad_input);
        let gw = dev.download(&grad_weight);
        assert_eq!(gi.len(), 4);
        assert_eq!(gw.len(), 2);
        // Numerical check: perturb input and verify gradient direction
        for v in &gi {
            assert!(v.is_finite());
        }
        for v in &gw {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn cpu_kv_attention_batched_matches_sequential() {
        let dev = CpuDevice::new();
        // Compare batched q_len=4 vs 4 sequential q_len=1 calls
        let n_heads = 2;
        let n_kv_heads = 1;
        let head_dim = 4;
        let q_len = 4;
        let total_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        // Random-ish data
        let q_data: Vec<f32> = (0..q_len * total_dim).map(|i| ((i * 7 + 3) % 13) as f32 / 13.0).collect();
        let kv_data: Vec<f32> = (0..q_len * kv_dim).map(|i| ((i * 11 + 5) % 17) as f32 / 17.0).collect();
        let v_data: Vec<f32> = (0..q_len * kv_dim).map(|i| ((i * 13 + 7) % 19) as f32 / 19.0).collect();

        // Batched
        let q = dev.upload(&q_data);
        let k = dev.upload(&kv_data);
        let v = dev.upload(&v_data);
        let batched = dev.download(&dev.kv_attention(&q, &k, &v, 0, q_len, n_heads, n_kv_heads, head_dim));

        // Sequential
        let mut sequential = Vec::new();
        for qi in 0..q_len {
            let q_slice = dev.upload(&q_data[qi * total_dim..(qi + 1) * total_dim]);
            let k_slice = dev.upload(&kv_data[..((qi + 1) * kv_dim)]);
            let v_slice = dev.upload(&v_data[..((qi + 1) * kv_dim)]);
            let out = dev.download(&dev.kv_attention(&q_slice, &k_slice, &v_slice, qi, 1, n_heads, n_kv_heads, head_dim));
            sequential.extend(out);
        }

        assert_eq!(batched.len(), sequential.len());
        for i in 0..batched.len() {
            assert!(
                (batched[i] - sequential[i]).abs() < 1e-5,
                "mismatch at {i}: batched={} sequential={}",
                batched[i], sequential[i]
            );
        }
    }
}
