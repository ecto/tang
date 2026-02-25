use alloc::vec::Vec;
use tang_tensor::{Tensor, Shape};

/// A trainable parameter — a tensor with an associated gradient buffer.
pub struct Parameter {
    pub data: Tensor<f64>,
    pub grad: Option<Tensor<f64>>,
}

impl Parameter {
    pub fn new(data: Tensor<f64>) -> Self {
        Self { data, grad: None }
    }

    /// Create parameter with random initialization (Xavier/Glorot uniform).
    /// Uses a simple LCG PRNG seeded by the given value.
    pub fn randn(shape: Shape, seed: u64) -> Self {
        let n = shape.numel();
        let mut state = seed;
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
            // Simple LCG-based normal approximation (Box-Muller)
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u1 = (state >> 11) as f64 / (1u64 << 53) as f64;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u2 = (state >> 11) as f64 / (1u64 << 53) as f64;

            let u1 = u1.max(1e-15); // avoid log(0)
            let normal = (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos();
            data.push(normal * scale);
        }

        Self::new(Tensor::new(data, shape))
    }

    pub fn zero_grad(&mut self) {
        self.grad = None;
    }

    pub fn shape(&self) -> &Shape {
        self.data.shape()
    }
}
