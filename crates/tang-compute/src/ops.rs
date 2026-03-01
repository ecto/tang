//! Free-standing tensor operations generic over ComputeDevice.

use crate::device::ComputeDevice;
use crate::tensor::ComputeTensor;
use tang::Scalar;
use tang_expr::node::ExprId;

/// Element-wise addition of two tensors.
pub fn add_tensors<D: ComputeDevice>(
    dev: &D,
    a: &ComputeTensor<D::Buffer>,
    b: &ComputeTensor<D::Buffer>,
) -> ComputeTensor<D::Buffer> {
    assert_eq!(a.numel(), b.numel(), "add_tensors: numel mismatch");
    let buf = dev.elementwise(
        &[&a.buffer, &b.buffer],
        a.numel(),
        &|ids: &[ExprId]| ids[0] + ids[1],
    );
    ComputeTensor::from_buffer(buf, a.shape().to_vec())
}

/// Broadcast bias addition: out[i] = matrix[i] + bias[i % dim].
///
/// matrix: [n_rows, dim], bias: [dim] → out: [n_rows, dim].
pub fn bias_add<D: ComputeDevice>(
    dev: &D,
    matrix: &ComputeTensor<D::Buffer>,
    bias: &ComputeTensor<D::Buffer>,
) -> ComputeTensor<D::Buffer> {
    let numel = matrix.numel();
    let dim = bias.numel();
    assert_eq!(numel % dim, 0, "bias_add: matrix numel not divisible by bias dim");
    let buf = dev.bias_add(&matrix.buffer, &bias.buffer, numel, dim);
    ComputeTensor::from_buffer(buf, matrix.shape().to_vec())
}

/// SwiGLU activation: silu(gate) * up, where silu(x) = x / (1 + exp(-x)).
pub fn swiglu_fused<D: ComputeDevice>(
    dev: &D,
    gate: &ComputeTensor<D::Buffer>,
    up: &ComputeTensor<D::Buffer>,
) -> ComputeTensor<D::Buffer> {
    assert_eq!(gate.numel(), up.numel(), "swiglu_fused: numel mismatch");
    let buf = dev.elementwise(
        &[&gate.buffer, &up.buffer],
        gate.numel(),
        &|ids: &[ExprId]| {
            // silu(gate) = gate * sigmoid(gate) = gate / (1 + exp(-gate))
            let one = ExprId::from_f64(1.0);
            let neg_gate = -ids[0];
            let exp_neg = Scalar::exp(neg_gate);
            let denom = one + exp_neg;
            let sigmoid = one / denom;
            let silu = ids[0] * sigmoid;
            silu * ids[1]
        },
    );
    ComputeTensor::from_buffer(buf, gate.shape().to_vec())
}

/// SwiGLU backward: given grad_output, gate, and up from the forward pass,
/// computes grad_gate and grad_up.
///
/// Forward: out = silu(gate) * up, where silu(x) = x * sigmoid(x).
/// Backward:
///   grad_up = grad * silu(gate)
///   grad_gate = grad * up * dsilu(gate), where dsilu(x) = sigmoid(x) * (1 + x*(1-sigmoid(x)))
pub fn swiglu_backward<D: ComputeDevice>(
    dev: &D,
    grad_output: &ComputeTensor<D::Buffer>,
    gate: &ComputeTensor<D::Buffer>,
    up: &ComputeTensor<D::Buffer>,
) -> (ComputeTensor<D::Buffer>, ComputeTensor<D::Buffer>) {
    let numel = grad_output.numel();

    // grad_up = grad * silu(gate)
    let grad_up_buf = dev.elementwise(
        &[&grad_output.buffer, &gate.buffer],
        numel,
        &|ids: &[ExprId]| {
            let one = ExprId::from_f64(1.0);
            let neg_gate = -ids[1];
            let exp_neg = Scalar::exp(neg_gate);
            let sigmoid = one / (one + exp_neg);
            let silu = ids[1] * sigmoid;
            ids[0] * silu
        },
    );

    // grad_gate = grad * up * dsilu(gate)
    // dsilu(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    //          = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    let grad_gate_buf = dev.elementwise(
        &[&grad_output.buffer, &gate.buffer, &up.buffer],
        numel,
        &|ids: &[ExprId]| {
            let one = ExprId::from_f64(1.0);
            let neg_gate = -ids[1];
            let exp_neg = Scalar::exp(neg_gate);
            let sigmoid = one / (one + exp_neg);
            let dsilu = sigmoid * (one + ids[1] * (one - sigmoid));
            ids[0] * ids[2] * dsilu
        },
    );

    (
        ComputeTensor::from_buffer(grad_gate_buf, grad_output.shape().to_vec()),
        ComputeTensor::from_buffer(grad_up_buf, grad_output.shape().to_vec()),
    )
}

/// Causal self-attention backward.
///
/// Recomputes attention scores from Q,K,V, then computes gradients.
/// Q, grad_output: `[seq_len, n_heads * head_dim]`
/// K, V: `[seq_len, n_kv_heads * head_dim]`
/// For GQA: n_kv_heads < n_heads, with n_heads/n_kv_heads heads per KV group.
///
/// Returns (grad_Q, grad_K, grad_V) with same shapes as inputs.
pub fn causal_attention_backward<D: ComputeDevice>(
    dev: &D,
    grad_output: &ComputeTensor<D::Buffer>,
    q: &ComputeTensor<D::Buffer>,
    k: &ComputeTensor<D::Buffer>,
    v: &ComputeTensor<D::Buffer>,
    seq_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> (ComputeTensor<D::Buffer>, ComputeTensor<D::Buffer>, ComputeTensor<D::Buffer>) {
    let total_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;

    let (gq, gk, gv) = dev.causal_attention_backward(
        &grad_output.buffer,
        &q.buffer,
        &k.buffer,
        &v.buffer,
        seq_len,
        n_heads,
        n_kv_heads,
        head_dim,
    );

    (
        ComputeTensor::from_buffer(gq, vec![seq_len, total_dim]),
        ComputeTensor::from_buffer(gk, vec![seq_len, kv_dim]),
        ComputeTensor::from_buffer(gv, vec![seq_len, kv_dim]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuDevice;

    #[test]
    fn test_add_tensors() {
        let dev = CpuDevice::new();
        let a = ComputeTensor::from_data(&dev, &[1.0, 2.0, 3.0], &[3]);
        let b = ComputeTensor::from_data(&dev, &[4.0, 5.0, 6.0], &[3]);
        let c = add_tensors(&dev, &a, &b);
        let out = c.to_vec();
        assert!((out[0] - 5.0).abs() < 1e-5);
        assert!((out[1] - 7.0).abs() < 1e-5);
        assert!((out[2] - 9.0).abs() < 1e-5);
    }

    #[test]
    fn test_bias_add() {
        let dev = CpuDevice::new();
        // 2x3 matrix + 3-element bias
        let mat = ComputeTensor::from_data(&dev, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let bias = ComputeTensor::from_data(&dev, &[0.1, 0.2, 0.3], &[3]);
        let out = bias_add(&dev, &mat, &bias);
        let v = out.to_vec();
        assert!((v[0] - 1.1).abs() < 1e-5);
        assert!((v[1] - 2.2).abs() < 1e-5);
        assert!((v[2] - 3.3).abs() < 1e-5);
        assert!((v[3] - 4.1).abs() < 1e-5);
        assert!((v[4] - 5.2).abs() < 1e-5);
        assert!((v[5] - 6.3).abs() < 1e-5);
    }

    #[test]
    fn test_swiglu_backward() {
        let dev = CpuDevice::new();
        let gate = ComputeTensor::from_data(&dev, &[0.0, 1.0, -1.0], &[3]);
        let up = ComputeTensor::from_data(&dev, &[1.0, 1.0, 1.0], &[3]);
        let grad = ComputeTensor::from_data(&dev, &[1.0, 1.0, 1.0], &[3]);
        let (grad_gate, grad_up) = swiglu_backward(&dev, &grad, &gate, &up);
        let gg = grad_gate.to_vec();
        let gu = grad_up.to_vec();
        // grad_up[i] = grad[i] * silu(gate[i])
        // At gate=0: silu(0)=0, so grad_up[0] should be 0
        assert!(gu[0].abs() < 1e-5);
        // At gate=1: silu(1)≈0.731, so grad_up[1]≈0.731
        assert!((gu[1] - 0.7311).abs() < 1e-3);
        // grad_gate should be finite
        for v in &gg {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_causal_attention_backward() {
        let dev = CpuDevice::new();
        // seq_len=2, n_heads=1, n_kv_heads=1, head_dim=2
        let q = ComputeTensor::from_data(&dev, &[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let k = ComputeTensor::from_data(&dev, &[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let v = ComputeTensor::from_data(&dev, &[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let grad_out = ComputeTensor::from_data(&dev, &[1.0, 0.0, 0.0, 1.0], &[2, 2]);

        let (gq, gk, gv) = causal_attention_backward(
            &dev, &grad_out, &q, &k, &v, 2, 1, 1, 2,
        );
        assert_eq!(gq.shape(), &[2, 2]);
        assert_eq!(gk.shape(), &[2, 2]);
        assert_eq!(gv.shape(), &[2, 2]);
        // All gradients should be finite
        for v in gq.to_vec().iter().chain(gk.to_vec().iter()).chain(gv.to_vec().iter()) {
            assert!(v.is_finite(), "gradient should be finite");
        }
    }

    #[test]
    fn test_swiglu_fused() {
        let dev = CpuDevice::new();
        let gate = ComputeTensor::from_data(&dev, &[0.0, 1.0, -1.0], &[3]);
        let up = ComputeTensor::from_data(&dev, &[1.0, 1.0, 1.0], &[3]);
        let out = swiglu_fused(&dev, &gate, &up);
        let v = out.to_vec();
        // silu(0) * 1 = 0
        assert!(v[0].abs() < 1e-5);
        // silu(1) * 1 = 1 * sigmoid(1) ≈ 0.7311
        assert!((v[1] - 0.7311).abs() < 1e-3);
        // silu(-1) * 1 = -1 * sigmoid(-1) ≈ -0.2689
        assert!((v[2] - (-0.2689)).abs() < 1e-3);
    }
}
