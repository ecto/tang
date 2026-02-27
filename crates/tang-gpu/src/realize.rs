//! Realize: trace through tang-expr, generate fused WGSL, dispatch.
//!
//! This is the core magic — user writes element-wise ops as `ExprId`
//! arithmetic, we trace it, optimize, and run on GPU in one fused kernel.

use tang_expr::{trace, ExprId};

use crate::device::GpuDevice;
use crate::kernel::KernelCache;
use crate::tensor::GpuTensor;

/// Apply an element-wise operation by tracing it through tang-expr,
/// generating a fused WGSL kernel, and dispatching on GPU.
///
/// Each element position gets one thread. The closure receives one
/// `ExprId` per input tensor and returns a single output `ExprId`.
/// All operations inside the closure build an expression graph that
/// gets compiled to a single GPU kernel.
///
/// # Example
///
/// ```ignore
/// let c = map_elementwise(&device, &mut cache, &[&a, &b], |args| {
///     let sum = args[0] + args[1];
///     sum * sum  // fused (a+b)^2 in one kernel
/// });
/// ```
pub fn map_elementwise(
    device: &GpuDevice,
    cache: &mut KernelCache,
    inputs: &[&GpuTensor],
    f: impl FnOnce(&[ExprId]) -> ExprId,
) -> GpuTensor {
    let n_inputs = inputs.len();

    // All inputs must have the same numel (broadcasting not yet supported)
    let numel = inputs[0].numel();
    for (i, t) in inputs.iter().enumerate().skip(1) {
        assert_eq!(
            t.numel(),
            numel,
            "input {i} has {} elements, expected {numel}",
            t.numel()
        );
    }

    // 1. Trace the user's closure to build an expression graph
    let (mut graph, output) = trace(|| {
        let vars: Vec<ExprId> = (0..n_inputs as u16).map(ExprId::var).collect();
        f(&vars)
    });

    // 2. Simplify the expression graph
    let output = graph.simplify(output);

    // 3. Generate WGSL
    let kernel = graph.to_wgsl(&[output], n_inputs);

    // 4. Interleave input buffers (GPU-side for 1-2 inputs)
    let interleaved = interleave_inputs(device, cache, inputs, numel, n_inputs);

    // 5. Allocate output buffer
    let out_tensor = GpuTensor::uninit(device, inputs[0].shape());

    // 6. Dispatch
    cache.dispatch(
        device,
        &kernel.source,
        &interleaved,
        &out_tensor.buffer,
        numel as u32,
    );

    out_tensor
}

/// Apply an element-wise operation with multiple outputs.
pub fn map_elementwise_multi(
    device: &GpuDevice,
    cache: &mut KernelCache,
    inputs: &[&GpuTensor],
    n_outputs: usize,
    f: impl FnOnce(&[ExprId]) -> Vec<ExprId>,
) -> Vec<GpuTensor> {
    let n_inputs = inputs.len();
    let numel = inputs[0].numel();

    let (mut graph, outputs) = trace(|| {
        let vars: Vec<ExprId> = (0..n_inputs as u16).map(ExprId::var).collect();
        f(&vars)
    });

    assert_eq!(outputs.len(), n_outputs);

    let outputs: Vec<ExprId> = outputs.into_iter().map(|o| graph.simplify(o)).collect();
    let kernel = graph.to_wgsl(&outputs, n_inputs);

    let interleaved = interleave_inputs(device, cache, inputs, numel, n_inputs);

    // Output buffer holds n_outputs values per work item
    let out_buf = crate::buffer::GpuBuffer::uninit(device, numel * n_outputs);

    cache.dispatch(
        device,
        &kernel.source,
        &interleaved,
        &out_buf,
        numel as u32,
    );

    // Read back and split
    let all_data = out_buf.to_vec_sync(device);
    let shape = inputs[0].shape();

    (0..n_outputs)
        .map(|k| {
            let data: Vec<f32> = (0..numel).map(|i| all_data[i * n_outputs + k]).collect();
            GpuTensor::from_slice(device, &data, shape)
        })
        .collect()
}

/// Interleave input tensors so each work item sees its values contiguously.
///
/// Input layout: for work item `i`, values are at offsets
/// `[i * n_inputs + 0, i * n_inputs + 1, ..., i * n_inputs + n_inputs-1]`.
fn interleave_inputs(
    device: &GpuDevice,
    cache: &mut KernelCache,
    inputs: &[&GpuTensor],
    numel: usize,
    n_inputs: usize,
) -> crate::buffer::GpuBuffer {
    if n_inputs == 1 {
        // Single input: GPU-side copy (no CPU round-trip).
        return inputs[0].buffer.clone_gpu_batched(device, cache);
    }

    if n_inputs == 2 {
        // 2 inputs: GPU interleave kernel via dispatch_rr_w
        let out = crate::buffer::GpuBuffer::uninit(device, numel * 2);

        let wgsl = r#"// Interleave 2 inputs: output[2*i+0] = a[i], output[2*i+1] = b[i]

struct Params {
    count: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) { return; }
    output[idx * 2u + 0u] = a[idx];
    output[idx * 2u + 1u] = b[idx];
}
"#;

        cache.dispatch_rr_w(
            device,
            wgsl,
            &inputs[0].buffer,
            &inputs[1].buffer,
            &out,
            &[numel as u32, 0, 0, 0],
        );

        return out;
    }

    // N > 2 inputs: CPU fallback
    let input_data: Vec<Vec<f32>> = inputs
        .iter()
        .map(|t| t.buffer.to_vec_sync(device))
        .collect();

    let mut interleaved = Vec::with_capacity(numel * n_inputs);
    for i in 0..numel {
        for input in &input_data {
            interleaved.push(input[i]);
        }
    }

    crate::buffer::GpuBuffer::from_slice(device, &interleaved)
}
