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
    // CPU roundtrip for simplicity (correct first, optimize later)
    let mat_data = dev.download(&matrix.buffer);
    let bias_data = dev.download(&bias.buffer);
    let mut out = mat_data;
    for i in 0..numel {
        out[i] += bias_data[i % dim];
    }
    ComputeTensor::from_data(dev, &out, matrix.shape())
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
