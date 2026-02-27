use crate::Rng;
use alloc::vec::Vec;
use tang::Scalar;
use tang_tensor::{Shape, Tensor};

/// A trainable parameter — a tensor with an associated gradient buffer.
pub struct Parameter<S: Scalar> {
    pub data: Tensor<S>,
    pub grad: Option<Tensor<S>>,
}

impl<S: Scalar> Parameter<S> {
    pub fn new(data: Tensor<S>) -> Self {
        Self { data, grad: None }
    }

    /// Create parameter with random initialization (Xavier/Glorot uniform).
    /// Uses `Rng` seeded by the given value.
    pub fn randn(shape: Shape, seed: u64) -> Self {
        let n = shape.numel();
        let mut rng = Rng::new(seed);
        let mut data = Vec::with_capacity(n);

        // Xavier scale based on fan_in + fan_out
        let dims = shape.dims();
        let (fan_in, fan_out) = if dims.len() >= 2 {
            (dims[dims.len() - 1] as f64, dims[dims.len() - 2] as f64)
        } else {
            (dims[0] as f64, dims[0] as f64)
        };
        let scale = (2.0 / (fan_in + fan_out)).sqrt();

        for _ in 0..n {
            data.push(S::from_f64(rng.normal() * scale));
        }

        Self::new(Tensor::new(data, shape))
    }

    /// Accumulate gradient — creates grad buffer if needed, otherwise adds.
    pub fn accumulate_grad(&mut self, grad: &Tensor<S>) {
        match &self.grad {
            Some(existing) => {
                self.grad = Some(existing.add(grad));
            }
            None => {
                self.grad = Some(grad.clone());
            }
        }
    }

    pub fn zero_grad(&mut self) {
        self.grad = None;
    }

    pub fn shape(&self) -> &Shape {
        self.data.shape()
    }
}
