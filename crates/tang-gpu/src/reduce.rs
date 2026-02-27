//! Reduction operations on GPU (sum, max, mean).

use crate::device::GpuDevice;
use crate::kernel::KernelCache;
use crate::tensor::GpuTensor;

/// Reduce-sum a tensor along an axis.
pub fn reduce_sum(
    device: &GpuDevice,
    cache: &mut KernelCache,
    input: &GpuTensor,
    axis: usize,
) -> GpuTensor {
    reduce_op(device, cache, input, axis, ReduceOp::Sum)
}

/// Reduce-max a tensor along an axis.
pub fn reduce_max(
    device: &GpuDevice,
    cache: &mut KernelCache,
    input: &GpuTensor,
    axis: usize,
) -> GpuTensor {
    reduce_op(device, cache, input, axis, ReduceOp::Max)
}

/// Reduce-mean a tensor along an axis.
pub fn reduce_mean(
    device: &GpuDevice,
    cache: &mut KernelCache,
    input: &GpuTensor,
    axis: usize,
) -> GpuTensor {
    reduce_op(device, cache, input, axis, ReduceOp::Mean)
}

enum ReduceOp {
    Sum,
    Max,
    Mean,
}

fn reduce_op(
    device: &GpuDevice,
    cache: &mut KernelCache,
    input: &GpuTensor,
    axis: usize,
    op: ReduceOp,
) -> GpuTensor {
    let shape = input.shape();
    assert!(axis < shape.len(), "reduce: axis out of bounds");

    // Flush pending GPU commands before CPU readback
    cache.flush(device);

    let axis_size = shape[axis];
    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape.remove(axis);
    if out_shape.is_empty() {
        out_shape.push(1);
    }
    let out_numel: usize = out_shape.iter().product();

    // Compute strides of input
    let ndim = shape.len();
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    let axis_stride = strides[axis];

    // For simplicity, do the reduction on CPU using downloaded data.
    // This is fine for moderate sizes — GPU reduction with atomics
    // would be the next optimization.
    let data = input.buffer.to_vec_sync(device);

    let mut out_data = vec![0.0f32; out_numel];

    // For each output element, reduce across the axis
    let out_strides = {
        let mut s = vec![1usize; out_shape.len()];
        for i in (0..out_shape.len().saturating_sub(1)).rev() {
            s[i] = s[i + 1] * out_shape[i + 1];
        }
        s
    };

    for (out_idx, out_val) in out_data.iter_mut().enumerate() {
        // Convert flat out_idx to multi-index in output shape
        let mut remaining = out_idx;
        let mut in_base = 0usize;
        for (d, &stride) in out_strides.iter().enumerate() {
            let coord = remaining / stride;
            remaining %= stride;
            // Map output dim back to input dim
            let in_dim = if d < axis { d } else { d + 1 };
            in_base += coord * strides[in_dim];
        }

        let init = match op {
            ReduceOp::Sum | ReduceOp::Mean => 0.0f32,
            ReduceOp::Max => f32::NEG_INFINITY,
        };

        let mut acc = init;
        for k in 0..axis_size {
            let val = data[in_base + k * axis_stride];
            match op {
                ReduceOp::Sum | ReduceOp::Mean => acc += val,
                ReduceOp::Max => {
                    if val > acc {
                        acc = val;
                    }
                }
            }
        }

        if matches!(op, ReduceOp::Mean) {
            acc /= axis_size as f32;
        }

        *out_val = acc;
    }

    GpuTensor::from_slice(device, &out_data, &out_shape)
}
