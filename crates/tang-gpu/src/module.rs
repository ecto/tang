//! Neural network modules and optimizers.

use crate::device::GpuDevice;
use crate::kernel::KernelCache;
use crate::matmul::matmul;
use crate::nn::bias_add;
use crate::reduce::reduce_sum;
use crate::tensor::GpuTensor;

/// A neural network module with trainable parameters.
pub trait GpuModule {
    /// Forward pass.
    fn forward(&self, device: &GpuDevice, cache: &mut KernelCache, input: &GpuTensor) -> GpuTensor;

    /// Collect all parameter tensors (immutable).
    fn parameters(&self) -> Vec<&GpuTensor>;

    /// Collect all parameter tensors (mutable).
    fn parameters_mut(&mut self) -> Vec<&mut GpuTensor>;
}

/// A trainable neural network module with forward and backward passes.
pub trait GpuTrainModule {
    /// Forward pass that caches activations needed for backward.
    fn forward_train(
        &mut self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        input: &GpuTensor,
    ) -> GpuTensor;

    /// Backward pass: given gradient of loss w.r.t. output, compute and store
    /// parameter gradients, and return gradient w.r.t. input.
    fn backward(
        &mut self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        grad_output: &GpuTensor,
    ) -> GpuTensor;

    /// Collect all parameter tensors (immutable).
    fn parameters(&self) -> Vec<&GpuTensor>;

    /// Collect all parameter tensors (mutable).
    fn parameters_mut(&mut self) -> Vec<&mut GpuTensor>;

    /// Collect gradients (one per parameter, None if not yet computed).
    fn gradients(&self) -> Vec<Option<&GpuTensor>>;

    /// Zero all stored gradients.
    fn zero_grad(&mut self);
}

/// A linear (fully connected) layer: y = x @ W^T + b.
pub struct GpuLinear {
    /// Weight matrix [out_features, in_features].
    pub weight: GpuTensor,
    /// Bias vector [out_features].
    pub bias: GpuTensor,
    pub in_features: usize,
    pub out_features: usize,
    // Training state
    cached_input: Option<GpuTensor>,
    weight_grad: Option<GpuTensor>,
    bias_grad: Option<GpuTensor>,
}

impl GpuLinear {
    /// Create a linear layer with given weights and bias.
    pub fn new(
        device: &GpuDevice,
        weight: &[f32],
        bias: &[f32],
        in_f: usize,
        out_f: usize,
    ) -> Self {
        assert_eq!(weight.len(), out_f * in_f);
        assert_eq!(bias.len(), out_f);
        Self {
            weight: GpuTensor::from_slice(device, weight, &[out_f, in_f]),
            bias: GpuTensor::from_slice(device, bias, &[out_f]),
            in_features: in_f,
            out_features: out_f,
            cached_input: None,
            weight_grad: None,
            bias_grad: None,
        }
    }

    /// Create a linear layer with zeros.
    pub fn zeros(device: &GpuDevice, in_f: usize, out_f: usize) -> Self {
        let weight = vec![0.0f32; out_f * in_f];
        let bias = vec![0.0f32; out_f];
        Self::new(device, &weight, &bias, in_f, out_f)
    }

    /// Create a linear layer with Kaiming uniform initialization.
    pub fn kaiming(device: &GpuDevice, in_f: usize, out_f: usize, seed: u64) -> Self {
        // Simple LCG PRNG for deterministic init
        let mut state = seed;
        let bound = (1.0 / in_f as f32).sqrt();
        let mut rand_f32 = || -> f32 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = (state >> 33) as f32 / (1u64 << 31) as f32; // [0, 1)
            (u * 2.0 - 1.0) * bound
        };
        let weight: Vec<f32> = (0..out_f * in_f).map(|_| rand_f32()).collect();
        let bias: Vec<f32> = (0..out_f).map(|_| rand_f32()).collect();
        Self::new(device, &weight, &bias, in_f, out_f)
    }
}

impl GpuModule for GpuLinear {
    fn forward(
        &self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        input: &GpuTensor,
    ) -> GpuTensor {
        // Simple matmul + bias for single vectors
        // input: [in_features], weight: [out_features, in_features]
        // output: [out_features] = weight @ input + bias
        cache.flush(device);
        let in_data = input.buffer.to_vec_sync(device);
        let w_data = self.weight.buffer.to_vec_sync(device);
        let b_data = self.bias.buffer.to_vec_sync(device);

        let mut out = b_data;
        for i in 0..self.out_features {
            for j in 0..self.in_features {
                out[i] += w_data[i * self.in_features + j] * in_data[j];
            }
        }

        GpuTensor::from_slice(device, &out, &[self.out_features])
    }

    fn parameters(&self) -> Vec<&GpuTensor> {
        vec![&self.weight, &self.bias]
    }

    fn parameters_mut(&mut self) -> Vec<&mut GpuTensor> {
        vec![&mut self.weight, &mut self.bias]
    }
}

impl GpuTrainModule for GpuLinear {
    fn forward_train(
        &mut self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        input: &GpuTensor,
    ) -> GpuTensor {
        // Cache input for backward pass
        self.cached_input = Some(input.clone_gpu_batched(device, cache));

        // Support batched [batch, in_features] or single [in_features]
        let is_batched = input.ndim() == 2;
        if is_batched {
            assert_eq!(input.shape()[1], self.in_features);
            // output = input @ W^T + bias
            let wt = self.weight.transpose_gpu(device, cache);
            let out = matmul(device, cache, input, &wt);
            bias_add(device, cache, &out, &self.bias)
        } else {
            // Single vector path (same as GpuModule::forward)
            self.forward(device, cache, input)
        }
    }

    fn backward(
        &mut self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        grad_output: &GpuTensor,
    ) -> GpuTensor {
        let input = self.cached_input.as_ref().expect("must call forward_train before backward");
        let is_batched = input.ndim() == 2;

        if is_batched {
            // grad_output: [batch, out_f]
            // grad_weight = grad_output^T @ input -> [out_f, in_f]
            let grad_output_t = grad_output.transpose_gpu(device, cache);
            let gw = matmul(device, cache, &grad_output_t, input);
            self.weight_grad = Some(gw);

            // grad_bias = sum(grad_output, axis=0) -> [out_f]
            let gb = reduce_sum(device, cache, grad_output, 0);
            self.bias_grad = Some(gb);

            // grad_input = grad_output @ weight -> [batch, in_f]
            let grad_input = matmul(device, cache, grad_output, &self.weight);
            grad_input
        } else {
            // Single vector: grad_output [out_f], input [in_f]
            // grad_weight[i,j] = grad_output[i] * input[j]
            cache.flush(device);
            let go_data = grad_output.buffer.to_vec_sync(device);
            let in_data = input.buffer.to_vec_sync(device);
            let mut gw = vec![0.0f32; self.out_features * self.in_features];
            for i in 0..self.out_features {
                for j in 0..self.in_features {
                    gw[i * self.in_features + j] = go_data[i] * in_data[j];
                }
            }
            self.weight_grad = Some(GpuTensor::from_slice(
                device, &gw, &[self.out_features, self.in_features],
            ));
            self.bias_grad = Some(GpuTensor::from_slice(
                device, &go_data, &[self.out_features],
            ));
            // grad_input = W^T @ grad_output
            let w_data = self.weight.buffer.to_vec_sync(device);
            let mut gi = vec![0.0f32; self.in_features];
            for j in 0..self.in_features {
                for i in 0..self.out_features {
                    gi[j] += w_data[i * self.in_features + j] * go_data[i];
                }
            }
            GpuTensor::from_slice(device, &gi, &[self.in_features])
        }
    }

    fn parameters(&self) -> Vec<&GpuTensor> {
        vec![&self.weight, &self.bias]
    }

    fn parameters_mut(&mut self) -> Vec<&mut GpuTensor> {
        vec![&mut self.weight, &mut self.bias]
    }

    fn gradients(&self) -> Vec<Option<&GpuTensor>> {
        vec![self.weight_grad.as_ref(), self.bias_grad.as_ref()]
    }

    fn zero_grad(&mut self) {
        self.weight_grad = None;
        self.bias_grad = None;
        self.cached_input = None;
    }
}

/// Adam optimizer with GPU-accelerated parameter updates.
pub struct GpuAdam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    /// First moment estimates on GPU, one per parameter tensor.
    m: Vec<GpuTensor>,
    /// Second moment estimates on GPU, one per parameter tensor.
    v: Vec<GpuTensor>,
    /// Step counter.
    t: usize,
}

impl GpuAdam {
    /// Create a new Adam optimizer.
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        }
    }

    /// Perform one optimizer step, updating parameters in-place on GPU.
    ///
    /// `params` and `grads` must have matching shapes.
    pub fn step(
        &mut self,
        device: &GpuDevice,
        cache: &mut KernelCache,
        params: &mut [&mut GpuTensor],
        grads: &[GpuTensor],
    ) {
        assert_eq!(params.len(), grads.len());
        self.t += 1;

        // Initialize moments on first call (zeros on GPU)
        if self.m.is_empty() {
            for p in params.iter() {
                let zeros = vec![0.0f32; p.numel()];
                self.m.push(GpuTensor::from_slice(device, &zeros, p.shape()));
                self.v.push(GpuTensor::from_slice(device, &zeros, p.shape()));
            }
        }

        let t = self.t as f32;
        let bc1 = 1.0 - self.beta1.powf(t);
        let bc2 = 1.0 - self.beta2.powf(t);

        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            let numel = param.numel() as u32;

            // Pack Adam hyperparams into uniform: [count, lr, beta1, beta2, eps, bc1, bc2, 0]
            // We need 8 f32s = 32 bytes. Use two uniform dispatches or pack as f32.
            // Actually, we can pack them as f32 bits into u32 slots since we're using uniform.
            // Better: use a custom dispatch with f32 params.
            cache.dispatch_adam(
                device,
                &param.buffer,
                &grad.buffer,
                &self.m[i].buffer,
                &self.v[i].buffer,
                numel,
                self.lr,
                self.beta1,
                self.beta2,
                self.eps,
                bc1,
                bc2,
            );
        }
    }
}
