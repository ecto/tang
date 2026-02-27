//! Fused forward-backward kernels.
//!
//! Traces a scalar-valued loss function, differentiates with respect to
//! all inputs, and compiles a single WGSL kernel that computes both the
//! loss and all gradients in one GPU dispatch.

use tang_expr::{trace, ExprId};

use crate::buffer::GpuBuffer;
use crate::device::GpuDevice;
use crate::kernel::KernelCache;

/// A compiled fused forward-backward kernel.
///
/// One dispatch computes `[loss, grad_0, grad_1, ..., grad_{n-1}]`.
pub struct FusedKernel {
    wgsl: String,
    n_inputs: usize,
    n_outputs: usize, // 1 (loss) + n_params (gradients)
}

impl FusedKernel {
    /// Run the fused kernel on GPU.
    ///
    /// Returns (loss, gradients) where gradients has one f32 per input.
    pub fn run(
        &self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        inputs: &[f32],
    ) -> (f32, Vec<f32>) {
        assert_eq!(
            inputs.len(),
            self.n_inputs,
            "expected {} inputs, got {}",
            self.n_inputs,
            inputs.len()
        );

        let in_buf = GpuBuffer::from_slice(device, inputs);
        let out_buf = GpuBuffer::uninit(device, self.n_outputs);

        // Dispatch with count=1 (scalar loss, not batched)
        cache.dispatch(device, &self.wgsl, &in_buf, &out_buf, 1);

        let result = out_buf.to_vec_sync(device);
        let loss = result[0];
        let grads = result[1..].to_vec();
        (loss, grads)
    }

    /// Number of input variables.
    pub fn n_inputs(&self) -> usize {
        self.n_inputs
    }

    /// The generated WGSL source (for debugging).
    pub fn wgsl(&self) -> &str {
        &self.wgsl
    }
}

/// Trace a scalar-valued function, differentiate it with respect to all
/// inputs, and compile a single fused forward-backward WGSL kernel.
///
/// The closure receives `n_inputs` variable ExprIds (var(0)..var(n-1))
/// and must return a single scalar loss ExprId.
///
/// # Example
///
/// ```ignore
/// let kernel = fused_forward_backward(3, |vars| {
///     // Quadratic loss: sum(x_i^2)
///     let mut loss = vars[0] * vars[0];
///     loss = loss + vars[1] * vars[1];
///     loss = loss + vars[2] * vars[2];
///     loss
/// });
/// let (loss, grads) = kernel.run(&device, &mut cache, &[1.0, 2.0, 3.0]);
/// // loss = 14.0, grads = [2.0, 4.0, 6.0]
/// ```
pub fn fused_forward_backward(n_inputs: usize, f: impl FnOnce(&[ExprId]) -> ExprId) -> FusedKernel {
    // 1. Trace the forward pass
    let (mut graph, loss) = trace(|| {
        let vars: Vec<ExprId> = (0..n_inputs as u16).map(ExprId::var).collect();
        f(&vars)
    });

    // 2. Differentiate loss w.r.t. each input variable
    let mut all_outputs = vec![loss];
    for i in 0..n_inputs as u16 {
        let grad = graph.diff(loss, i);
        let grad = graph.simplify(grad);
        all_outputs.push(grad);
    }

    // 3. Simplify the loss too
    all_outputs[0] = graph.simplify(loss);

    // 4. Generate one fused WGSL kernel
    let kernel = graph.to_wgsl(&all_outputs, n_inputs);

    FusedKernel {
        wgsl: kernel.source,
        n_inputs,
        n_outputs: all_outputs.len(),
    }
}

/// Convenience: run a fused forward-backward pass on CPU data.
///
/// Traces, compiles, dispatches, and returns (loss, gradients).
pub fn forward_backward_gpu(
    device: &GpuDevice,
    cache: &mut KernelCache,
    n_inputs: usize,
    inputs: &[f32],
    f: impl FnOnce(&[ExprId]) -> ExprId,
) -> (f32, Vec<f32>) {
    let kernel = fused_forward_backward(n_inputs, f);
    kernel.run(device, cache, inputs)
}
