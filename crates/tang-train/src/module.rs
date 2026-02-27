use crate::Parameter;
use alloc::string::String;
use alloc::vec::Vec;
use tang::Scalar;
use tang_tensor::Tensor;

/// Trait for neural network modules.
///
/// `forward` takes `&mut self` so layers can cache activations needed for backward.
/// For inference-only use, the caching is harmless. For physics/PINN cases where
/// you want `&self`, use tang-ad or tang-expr directly instead of Module.
pub trait Module<S: Scalar> {
    /// Forward pass: input -> output. Caches activations for backward.
    fn forward(&mut self, input: &Tensor<S>) -> Tensor<S>;

    /// Backward pass: gradient of loss w.r.t. output -> gradient w.r.t. input.
    /// Accumulates gradients into parameters.
    fn backward(&mut self, grad_output: &Tensor<S>) -> Tensor<S>;

    /// Collect all trainable parameters.
    fn parameters(&self) -> Vec<&Parameter<S>>;

    /// Collect all trainable parameters mutably.
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<S>>;

    /// Named parameters — returns `(name, parameter)` pairs.
    ///
    /// Linear layers return `"weight"` and `"bias"`. Sequential prefixes
    /// with the layer index: `"0.weight"`, `"0.bias"`, `"2.weight"`, etc.
    fn named_parameters(&self) -> Vec<(String, &Parameter<S>)> {
        self.parameters()
            .into_iter()
            .enumerate()
            .map(|(i, p)| (alloc::format!("{}", i), p))
            .collect()
    }

    /// Named parameters (mutable). Same naming convention as [`named_parameters`].
    fn named_parameters_mut(&mut self) -> Vec<(String, &mut Parameter<S>)> {
        self.parameters_mut()
            .into_iter()
            .enumerate()
            .map(|(i, p)| (alloc::format!("{}", i), p))
            .collect()
    }

    /// Export all parameter tensors as named pairs (a "state dict").
    ///
    /// ```ignore
    /// let state = model.state_dict();
    /// // state: [("0.weight", Tensor), ("0.bias", Tensor), ...]
    /// ```
    fn state_dict(&self) -> Vec<(String, Tensor<S>)> {
        self.named_parameters()
            .into_iter()
            .map(|(name, param)| (name, param.data.clone()))
            .collect()
    }

    /// Load parameter tensors by name. Silently skips names not in the module.
    fn load_state_dict(&mut self, state: &[(String, Tensor<S>)]) {
        for (name, param) in self.named_parameters_mut() {
            if let Some((_, tensor)) = state.iter().find(|(n, _)| n == &name) {
                param.data = tensor.clone();
            }
        }
    }

    /// Switch between training and eval mode.
    ///
    /// Layers like [`Dropout`] change behavior depending on mode.
    /// [`Sequential`] propagates to all children. The default impl
    /// is a no-op, which is correct for stateless layers like Linear/ReLU/Tanh.
    fn set_training(&mut self, _training: bool) {}

    /// Zero all gradients.
    fn zero_grad(&mut self) {
        for p in self.parameters_mut() {
            p.zero_grad();
        }
    }
}
