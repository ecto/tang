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
        head_dim: usize,
    ) -> CpuBuffer {
        let total_dim = n_heads * head_dim;
        let mut out = vec![0.0f32; seq_len * total_dim];
        let scale = 1.0 / (head_dim as f32).sqrt();

        for h in 0..n_heads {
            let h_off = h * head_dim;

            for i in 0..seq_len {
                // Compute attention scores for position i
                let mut scores = vec![0.0f32; i + 1];
                for j in 0..=i {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q.data[i * total_dim + h_off + d]
                            * k.data[j * total_dim + h_off + d];
                    }
                    scores[j] = dot * scale;
                }

                // Softmax
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                let inv = 1.0 / sum;
                for s in scores.iter_mut() {
                    *s *= inv;
                }

                // Weighted sum of values
                for d in 0..head_dim {
                    let mut val = 0.0f32;
                    for j in 0..=i {
                        val += scores[j] * v.data[j * total_dim + h_off + d];
                    }
                    out[i * total_dim + h_off + d] = val;
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

                // Compute scores against attended positions
                let mut scores = vec![0.0f32; attend_len];
                for j in 0..attend_len {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q.data[q_off + d] * k_cache.data[j * kv_dim + kv_off + d];
                    }
                    scores[j] = dot * scale;
                }

                // Softmax
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                let inv = 1.0 / sum;
                for s in scores.iter_mut() {
                    *s *= inv;
                }

                // Weighted sum
                let out_off = qi * total_dim + h * head_dim;
                for d in 0..head_dim {
                    let mut val = 0.0f32;
                    for j in 0..attend_len {
                        val += scores[j] * v_cache.data[j * kv_dim + kv_off + d];
                    }
                    out[out_off + d] = val;
                }
            }
        }

        CpuBuffer { data: out }
    }

    fn sync(&self) {
        // No-op for CPU
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
        let result = dev.causal_attention(&q, &k, &v, 2, 1, 2);
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
