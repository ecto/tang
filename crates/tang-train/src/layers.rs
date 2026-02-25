use alloc::boxed::Box;
use alloc::vec::Vec;
use tang_tensor::{Tensor, Shape};
use crate::{Parameter, Module};

/// Fully-connected (dense) linear layer: y = xW^T + b
pub struct Linear {
    pub weight: Parameter,  // [out_features, in_features]
    pub bias: Parameter,    // [out_features]
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, seed: u64) -> Self {
        Self {
            weight: Parameter::randn(Shape::from_slice(&[out_features, in_features]), seed),
            bias: Parameter::new(Tensor::zeros(Shape::from_slice(&[out_features]))),
        }
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor<f64>) -> Tensor<f64> {
        // input: [batch, in_features] or [in_features]
        // weight: [out_features, in_features]
        // output: [batch, out_features] or [out_features]
        let wt = self.weight.data.transpose(); // [in_features, out_features]

        if input.ndim() == 1 {
            // Single sample: [in_features] -> [out_features]
            let input_2d = input.reshape(Shape::from_slice(&[1, input.numel()]));
            let out = input_2d.matmul(&wt); // [1, out_features]
            let out_1d = out.reshape(Shape::from_slice(&[self.bias.data.numel()]));
            out_1d.add(&self.bias.data)
        } else {
            // Batch: [batch, in_features] -> [batch, out_features]
            let out = input.matmul(&wt);
            // Broadcast bias [out_features] over batch
            out.add(&self.bias.data)
        }
    }

    fn parameters(&self) -> Vec<&Parameter> {
        alloc::vec![&self.weight, &self.bias]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        alloc::vec![&mut self.weight, &mut self.bias]
    }
}

/// ReLU activation layer.
pub struct ReLU;

impl Module for ReLU {
    fn forward(&self, input: &Tensor<f64>) -> Tensor<f64> {
        input.relu()
    }

    fn parameters(&self) -> Vec<&Parameter> {
        Vec::new()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        Vec::new()
    }
}

/// Sequential container — chains modules in order.
pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Self { layers }
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Tensor<f64>) -> Tensor<f64> {
        let mut x = input.clone();
        for layer in &self.layers {
            x = (*layer).forward(&x);
        }
        x
    }

    fn parameters(&self) -> Vec<&Parameter> {
        self.layers.iter().flat_map(|l: &Box<dyn Module>| l.parameters()).collect()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        self.layers.iter_mut().flat_map(|l: &mut Box<dyn Module>| l.parameters_mut()).collect()
    }
}
