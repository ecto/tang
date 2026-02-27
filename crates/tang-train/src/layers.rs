use crate::{Module, Parameter, Rng};
use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use tang::Scalar;
use tang_tensor::{Shape, Tensor};

/// Fully-connected (dense) linear layer: y = xW^T + b
pub struct Linear<S: Scalar> {
    pub weight: Parameter<S>, // [out_features, in_features]
    pub bias: Parameter<S>,   // [out_features]
    cached_input: Option<Tensor<S>>,
}

impl<S: Scalar> Linear<S> {
    pub fn new(in_features: usize, out_features: usize, seed: u64) -> Self {
        Self {
            weight: Parameter::randn(Shape::from_slice(&[out_features, in_features]), seed),
            bias: Parameter::new(Tensor::zeros(Shape::from_slice(&[out_features]))),
            cached_input: None,
        }
    }
}

impl<S: Scalar> Module<S> for Linear<S> {
    fn forward(&mut self, input: &Tensor<S>) -> Tensor<S> {
        self.cached_input = Some(input.clone());
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
            out.add(&self.bias.data)
        }
    }

    fn backward(&mut self, grad_output: &Tensor<S>) -> Tensor<S> {
        let input = self
            .cached_input
            .as_ref()
            .expect("must call forward before backward");

        // Ensure 2D for matmul
        let (input_2d, grad_2d) = if input.ndim() == 1 {
            (
                input.reshape(Shape::from_slice(&[1, input.numel()])),
                grad_output.reshape(Shape::from_slice(&[1, grad_output.numel()])),
            )
        } else {
            (input.clone(), grad_output.clone())
        };

        // grad_w = grad_output^T @ input — [out, batch] @ [batch, in] = [out, in]
        let grad_w = grad_2d.transpose().matmul(&input_2d);
        self.weight.accumulate_grad(&grad_w);

        // grad_b = sum(grad_output, axis=0) — [out]
        let grad_b = if grad_2d.shape()[0] == 1 {
            grad_2d.reshape(Shape::from_slice(&[grad_2d.shape()[1]]))
        } else {
            grad_2d.sum_axis(0)
        };
        self.bias.accumulate_grad(&grad_b);

        // grad_input = grad_output @ weight — [batch, out] @ [out, in] = [batch, in]
        let grad_input = grad_2d.matmul(&self.weight.data);

        if input.ndim() == 1 {
            grad_input.reshape(Shape::from_slice(&[input.numel()]))
        } else {
            grad_input
        }
    }

    fn parameters(&self) -> Vec<&Parameter<S>> {
        alloc::vec![&self.weight, &self.bias]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<S>> {
        alloc::vec![&mut self.weight, &mut self.bias]
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<S>)> {
        alloc::vec![
            (String::from("weight"), &self.weight),
            (String::from("bias"), &self.bias),
        ]
    }

    fn named_parameters_mut(&mut self) -> Vec<(String, &mut Parameter<S>)> {
        alloc::vec![
            (String::from("weight"), &mut self.weight),
            (String::from("bias"), &mut self.bias),
        ]
    }
}

/// ReLU activation layer.
pub struct ReLU<S: Scalar> {
    cached_input: Option<Tensor<S>>,
}

impl<S: Scalar> ReLU<S> {
    pub fn new() -> Self {
        Self { cached_input: None }
    }
}

impl<S: Scalar> Default for ReLU<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Scalar> Module<S> for ReLU<S> {
    fn forward(&mut self, input: &Tensor<S>) -> Tensor<S> {
        self.cached_input = Some(input.clone());
        input.relu()
    }

    fn backward(&mut self, grad_output: &Tensor<S>) -> Tensor<S> {
        let input = self
            .cached_input
            .as_ref()
            .expect("must call forward before backward");
        let zero = S::from_f64(0.0);
        let one = S::from_f64(1.0);
        let mask = input.map(|v| if v > zero { one } else { zero });
        grad_output.mul(&mask)
    }

    fn parameters(&self) -> Vec<&Parameter<S>> {
        Vec::new()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<S>> {
        Vec::new()
    }
}

/// Tanh activation layer.
pub struct Tanh<S: Scalar> {
    cached_output: Option<Tensor<S>>,
}

impl<S: Scalar> Tanh<S> {
    pub fn new() -> Self {
        Self {
            cached_output: None,
        }
    }
}

impl<S: Scalar> Default for Tanh<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Scalar> Module<S> for Tanh<S> {
    fn forward(&mut self, input: &Tensor<S>) -> Tensor<S> {
        let output = input.tanh();
        self.cached_output = Some(output.clone());
        output
    }

    fn backward(&mut self, grad_output: &Tensor<S>) -> Tensor<S> {
        let output = self
            .cached_output
            .as_ref()
            .expect("must call forward before backward");
        // d/dx tanh(x) = 1 - tanh(x)^2
        let one = S::from_f64(1.0);
        let dtanh = output.map(|t| one - t * t);
        grad_output.mul(&dtanh)
    }

    fn parameters(&self) -> Vec<&Parameter<S>> {
        Vec::new()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<S>> {
        Vec::new()
    }
}

/// Embedding lookup table — maps integer indices to dense vectors.
///
/// Input: `[batch, seq_len]` of integer indices (stored as `S`, converted via `to_f64`)
/// Output: `[batch, seq_len * embed_dim]` — embeddings concatenated flat for Linear compatibility.
///
/// Replaces manual one-hot encoding with a learnable dense representation.
pub struct Embedding<S: Scalar> {
    pub weight: Parameter<S>, // [num_embeddings, embed_dim]
    embed_dim: usize,
    cached_input_shape: Option<(usize, usize)>, // (batch, seq_len)
    cached_indices: Option<Vec<usize>>,          // flat indices for backward
}

impl<S: Scalar> Embedding<S> {
    pub fn new(num_embeddings: usize, embed_dim: usize, seed: u64) -> Self {
        Self {
            weight: Parameter::randn(Shape::from_slice(&[num_embeddings, embed_dim]), seed),
            embed_dim,
            cached_input_shape: None,
            cached_indices: None,
        }
    }
}

impl<S: Scalar> Module<S> for Embedding<S> {
    fn forward(&mut self, input: &Tensor<S>) -> Tensor<S> {
        assert_eq!(input.ndim(), 2, "Embedding input must be [batch, seq_len]");
        let batch = input.shape()[0];
        let seq_len = input.shape()[1];

        // Collect indices and build output
        let mut indices = Vec::with_capacity(batch * seq_len);
        let mut out_data = Vec::with_capacity(batch * seq_len * self.embed_dim);

        for b in 0..batch {
            for s in 0..seq_len {
                let idx = input.get(&[b, s]).to_f64() as usize;
                indices.push(idx);
                for e in 0..self.embed_dim {
                    out_data.push(self.weight.data.get(&[idx, e]));
                }
            }
        }

        self.cached_input_shape = Some((batch, seq_len));
        self.cached_indices = Some(indices);

        Tensor::new(out_data, Shape::from_slice(&[batch, seq_len * self.embed_dim]))
    }

    fn backward(&mut self, grad_output: &Tensor<S>) -> Tensor<S> {
        let (batch, seq_len) = self.cached_input_shape.expect("must call forward before backward");
        let indices = self.cached_indices.as_ref().expect("must call forward before backward");

        // Scatter-add gradients back to weight matrix
        // grad_output: [batch, seq_len * embed_dim]
        let num_emb = self.weight.data.shape()[0];
        let mut grad_w_data = alloc::vec![S::ZERO; num_emb * self.embed_dim];

        for b in 0..batch {
            for s in 0..seq_len {
                let idx = indices[b * seq_len + s];
                for e in 0..self.embed_dim {
                    grad_w_data[idx * self.embed_dim + e] +=
                        grad_output.get(&[b, s * self.embed_dim + e]);
                }
            }
        }

        let grad_w = Tensor::new(grad_w_data, self.weight.data.shape().clone());
        self.weight.accumulate_grad(&grad_w);

        // No meaningful gradient for integer indices
        Tensor::zeros(Shape::from_slice(&[batch, seq_len]))
    }

    fn parameters(&self) -> Vec<&Parameter<S>> {
        alloc::vec![&self.weight]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<S>> {
        alloc::vec![&mut self.weight]
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<S>)> {
        alloc::vec![(String::from("weight"), &self.weight)]
    }

    fn named_parameters_mut(&mut self) -> Vec<(String, &mut Parameter<S>)> {
        alloc::vec![(String::from("weight"), &mut self.weight)]
    }
}

/// Dropout layer — randomly zeros elements during training.
///
/// During training, each element is zeroed with probability `p` and the remaining
/// elements are scaled by `1/(1-p)` (inverted dropout). During eval, acts as identity.
///
/// ```ignore
/// let mut dropout = Dropout::<f64>::new(0.5, 42);
/// dropout.training = false; // disable for inference
/// ```
pub struct Dropout<S: Scalar> {
    pub p: f64,
    pub training: bool,
    rng: Rng,
    cached_mask: Option<Tensor<S>>,
}

impl<S: Scalar> Dropout<S> {
    pub fn new(p: f64, seed: u64) -> Self {
        assert!((0.0..1.0).contains(&p), "dropout probability must be in [0, 1)");
        Self {
            p,
            training: true,
            rng: Rng::new(seed),
            cached_mask: None,
        }
    }
}

impl<S: Scalar> Module<S> for Dropout<S> {
    fn forward(&mut self, input: &Tensor<S>) -> Tensor<S> {
        if !self.training || self.p == 0.0 {
            self.cached_mask = None;
            return input.clone();
        }

        let scale = S::from_f64(1.0 / (1.0 - self.p));
        let zero = S::ZERO;

        let mask_data: Vec<S> = (0..input.numel())
            .map(|_| {
                if !self.rng.bernoulli(self.p) {
                    scale
                } else {
                    zero
                }
            })
            .collect();

        let mask = Tensor::new(mask_data, input.shape().clone());
        let output = input.mul(&mask);
        self.cached_mask = Some(mask);
        output
    }

    fn backward(&mut self, grad_output: &Tensor<S>) -> Tensor<S> {
        match &self.cached_mask {
            Some(mask) => grad_output.mul(mask),
            None => grad_output.clone(),
        }
    }

    fn parameters(&self) -> Vec<&Parameter<S>> {
        Vec::new()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<S>> {
        Vec::new()
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

/// Sequential container — chains modules in order.
pub struct Sequential<S: Scalar> {
    layers: Vec<Box<dyn Module<S>>>,
}

impl<S: Scalar> Sequential<S> {
    pub fn new(layers: Vec<Box<dyn Module<S>>>) -> Self {
        Self { layers }
    }
}

impl<S: Scalar> Module<S> for Sequential<S> {
    fn forward(&mut self, input: &Tensor<S>) -> Tensor<S> {
        let mut x = input.clone();
        for layer in &mut self.layers {
            x = layer.forward(&x);
        }
        x
    }

    fn backward(&mut self, grad_output: &Tensor<S>) -> Tensor<S> {
        let mut grad = grad_output.clone();
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad);
        }
        grad
    }

    fn parameters(&self) -> Vec<&Parameter<S>> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<S>> {
        self.layers
            .iter_mut()
            .flat_map(|l| l.parameters_mut())
            .collect()
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<S>)> {
        self.layers
            .iter()
            .enumerate()
            .flat_map(|(i, layer)| {
                layer
                    .named_parameters()
                    .into_iter()
                    .map(move |(name, param)| (alloc::format!("{}.{}", i, name), param))
            })
            .collect()
    }

    fn named_parameters_mut(&mut self) -> Vec<(String, &mut Parameter<S>)> {
        self.layers
            .iter_mut()
            .enumerate()
            .flat_map(|(i, layer)| {
                layer
                    .named_parameters_mut()
                    .into_iter()
                    .map(move |(name, param)| (alloc::format!("{}.{}", i, name), param))
            })
            .collect()
    }

    fn set_training(&mut self, training: bool) {
        for layer in &mut self.layers {
            layer.set_training(training);
        }
    }
}
