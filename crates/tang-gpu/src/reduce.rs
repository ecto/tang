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

/// Sum all elements to a single scalar [1] tensor, entirely on GPU.
pub fn reduce_sum_all(
    device: &GpuDevice,
    cache: &mut KernelCache,
    input: &GpuTensor,
) -> GpuTensor {
    let mut t = reduce_sum(device, cache, input, 0);
    while t.numel() > 1 {
        t = reduce_sum(device, cache, &t, 0);
    }
    t
}

#[derive(Clone, Copy)]
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

    let axis_size = shape[axis];
    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape.remove(axis);
    if out_shape.is_empty() {
        out_shape.push(1);
    }
    let out_numel: usize = out_shape.iter().product();

    // inner_size = product of dimensions after the axis
    let inner_size: usize = shape[axis + 1..].iter().product();

    let out = GpuTensor::uninit(device, &out_shape);

    // op_code: 0 = sum, 1 = max, 2 = mean
    let op_code = match op {
        ReduceOp::Sum => 0u32,
        ReduceOp::Max => 1,
        ReduceOp::Mean => 2,
    };

    let wgsl = r#"// Reduce along an axis: one thread per output element

struct Params {
    count: u32,
    axis_size: u32,
    inner_size: u32,
    op_code: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) { return; }

    let outer_idx = idx / params.inner_size;
    let inner_idx = idx % params.inner_size;
    let base = outer_idx * (params.axis_size * params.inner_size) + inner_idx;
    let stride = params.inner_size;

    var acc: f32;
    if (params.op_code == 1u) {
        acc = -3.402823e+38; // f32 MIN
    } else {
        acc = 0.0;
    }

    for (var k: u32 = 0u; k < params.axis_size; k = k + 1u) {
        let val = input[base + k * stride];
        if (params.op_code == 1u) {
            acc = max(acc, val);
        } else {
            acc = acc + val;
        }
    }

    if (params.op_code == 2u) {
        acc = acc / f32(params.axis_size);
    }

    output[idx] = acc;
}
"#;

    cache.dispatch_with_params(
        device,
        wgsl,
        &input.buffer,
        &out.buffer,
        &[out_numel as u32, axis_size as u32, inner_size as u32, op_code],
    );

    out
}
