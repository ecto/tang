use alloc::vec::Vec;
use tang_tensor::Tensor;
use crate::Parameter;

/// Trait for neural network modules.
pub trait Module {
    /// Forward pass: input -> output.
    fn forward(&self, input: &Tensor<f64>) -> Tensor<f64>;

    /// Collect all trainable parameters.
    fn parameters(&self) -> Vec<&Parameter>;

    /// Collect all trainable parameters mutably.
    fn parameters_mut(&mut self) -> Vec<&mut Parameter>;

    /// Zero all gradients.
    fn zero_grad(&mut self) {
        for p in self.parameters_mut() {
            p.zero_grad();
        }
    }
}
