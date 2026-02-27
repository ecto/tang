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

/// 1D convolution layer (cross-correlation, no padding, stride=1).
///
/// Input: `[batch, in_channels, length]`
/// Output: `[batch, out_channels, length - kernel_size + 1]`
pub struct Conv1d<S: Scalar> {
    pub weight: Parameter<S>, // [out_channels, in_channels, kernel_size]
    pub bias: Parameter<S>,   // [out_channels]
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cached_input: Option<Tensor<S>>,
}

impl<S: Scalar> Conv1d<S> {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, seed: u64) -> Self {
        Self {
            weight: Parameter::randn(
                Shape::from_slice(&[out_channels, in_channels, kernel_size]),
                seed,
            ),
            bias: Parameter::new(Tensor::zeros(Shape::from_slice(&[out_channels]))),
            in_channels,
            out_channels,
            kernel_size,
            cached_input: None,
        }
    }
}

impl<S: Scalar> Module<S> for Conv1d<S> {
    fn forward(&mut self, input: &Tensor<S>) -> Tensor<S> {
        assert_eq!(input.ndim(), 3, "Conv1d input must be [batch, in_channels, length]");
        let batch = input.shape()[0];
        let ic = input.shape()[1];
        assert_eq!(ic, self.in_channels);
        let length = input.shape()[2];
        assert!(length >= self.kernel_size, "input length must be >= kernel_size");
        let out_len = length - self.kernel_size + 1;

        self.cached_input = Some(input.clone());

        Tensor::from_fn(
            Shape::from_slice(&[batch, self.out_channels, out_len]),
            |idx| {
                let (b, oc, pos) = (idx[0], idx[1], idx[2]);
                let mut sum = self.bias.data.get(&[oc]);
                for c in 0..self.in_channels {
                    for k in 0..self.kernel_size {
                        sum += self.weight.data.get(&[oc, c, k]) * input.get(&[b, c, pos + k]);
                    }
                }
                sum
            },
        )
    }

    fn backward(&mut self, grad_output: &Tensor<S>) -> Tensor<S> {
        let input = self
            .cached_input
            .as_ref()
            .expect("must call forward before backward");
        let batch = input.shape()[0];
        let length = input.shape()[2];
        let out_len = length - self.kernel_size + 1;

        // grad_weight[oc][ic][k] = sum over b,pos of grad_output[b][oc][pos] * input[b][ic][pos+k]
        let grad_w = Tensor::from_fn(self.weight.data.shape().clone(), |idx| {
            let (oc, ic, k) = (idx[0], idx[1], idx[2]);
            let mut sum = S::ZERO;
            for b in 0..batch {
                for pos in 0..out_len {
                    sum += grad_output.get(&[b, oc, pos]) * input.get(&[b, ic, pos + k]);
                }
            }
            sum
        });
        self.weight.accumulate_grad(&grad_w);

        // grad_bias[oc] = sum over b,pos of grad_output[b][oc][pos]
        let grad_b = Tensor::from_fn(self.bias.data.shape().clone(), |idx| {
            let oc = idx[0];
            let mut sum = S::ZERO;
            for b in 0..batch {
                for pos in 0..out_len {
                    sum += grad_output.get(&[b, oc, pos]);
                }
            }
            sum
        });
        self.bias.accumulate_grad(&grad_b);

        // grad_input[b][ic][i] = sum over oc,k of weight[oc][ic][k] * grad_output[b][oc][i-k]
        // where 0 <= i-k < out_len
        Tensor::from_fn(input.shape().clone(), |idx| {
            let (b, ic, i) = (idx[0], idx[1], idx[2]);
            let mut sum = S::ZERO;
            for oc in 0..self.out_channels {
                for k in 0..self.kernel_size {
                    if i >= k && (i - k) < out_len {
                        sum += self.weight.data.get(&[oc, ic, k])
                            * grad_output.get(&[b, oc, i - k]);
                    }
                }
            }
            sum
        })
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

/// 2D convolution layer (cross-correlation, no padding, stride=1).
///
/// Input: `[batch, in_channels, height, width]`
/// Output: `[batch, out_channels, height - kh + 1, width - kw + 1]`
pub struct Conv2d<S: Scalar> {
    pub weight: Parameter<S>, // [out_channels, in_channels, kh, kw]
    pub bias: Parameter<S>,   // [out_channels]
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cached_input: Option<Tensor<S>>,
}

impl<S: Scalar> Conv2d<S> {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, seed: u64) -> Self {
        Self {
            weight: Parameter::randn(
                Shape::from_slice(&[out_channels, in_channels, kernel_size, kernel_size]),
                seed,
            ),
            bias: Parameter::new(Tensor::zeros(Shape::from_slice(&[out_channels]))),
            in_channels,
            out_channels,
            kernel_size,
            cached_input: None,
        }
    }
}

impl<S: Scalar> Module<S> for Conv2d<S> {
    fn forward(&mut self, input: &Tensor<S>) -> Tensor<S> {
        assert_eq!(input.ndim(), 4, "Conv2d input must be [batch, in_channels, height, width]");
        let batch = input.shape()[0];
        let ic = input.shape()[1];
        assert_eq!(ic, self.in_channels);
        let height = input.shape()[2];
        let width = input.shape()[3];
        let kh = self.kernel_size;
        let kw = self.kernel_size;
        assert!(height >= kh && width >= kw, "spatial dims must be >= kernel_size");
        let out_h = height - kh + 1;
        let out_w = width - kw + 1;

        self.cached_input = Some(input.clone());

        Tensor::from_fn(
            Shape::from_slice(&[batch, self.out_channels, out_h, out_w]),
            |idx| {
                let (b, oc, oh, ow) = (idx[0], idx[1], idx[2], idx[3]);
                let mut sum = self.bias.data.get(&[oc]);
                for c in 0..self.in_channels {
                    for ki in 0..kh {
                        for kj in 0..kw {
                            sum += self.weight.data.get(&[oc, c, ki, kj])
                                * input.get(&[b, c, oh + ki, ow + kj]);
                        }
                    }
                }
                sum
            },
        )
    }

    fn backward(&mut self, grad_output: &Tensor<S>) -> Tensor<S> {
        let input = self
            .cached_input
            .as_ref()
            .expect("must call forward before backward");
        let batch = input.shape()[0];
        let height = input.shape()[2];
        let width = input.shape()[3];
        let kh = self.kernel_size;
        let kw = self.kernel_size;
        let out_h = height - kh + 1;
        let out_w = width - kw + 1;

        // grad_weight[oc][ic][ki][kj] = sum over b,oh,ow of grad_output[b][oc][oh][ow] * input[b][ic][oh+ki][ow+kj]
        let grad_w = Tensor::from_fn(self.weight.data.shape().clone(), |idx| {
            let (oc, ic, ki, kj) = (idx[0], idx[1], idx[2], idx[3]);
            let mut sum = S::ZERO;
            for b in 0..batch {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        sum += grad_output.get(&[b, oc, oh, ow])
                            * input.get(&[b, ic, oh + ki, ow + kj]);
                    }
                }
            }
            sum
        });
        self.weight.accumulate_grad(&grad_w);

        // grad_bias[oc] = sum over b,oh,ow of grad_output[b][oc][oh][ow]
        let grad_b = Tensor::from_fn(self.bias.data.shape().clone(), |idx| {
            let oc = idx[0];
            let mut sum = S::ZERO;
            for b in 0..batch {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        sum += grad_output.get(&[b, oc, oh, ow]);
                    }
                }
            }
            sum
        });
        self.bias.accumulate_grad(&grad_b);

        // grad_input[b][ic][i][j] = sum over oc,ki,kj of weight[oc][ic][ki][kj] * grad_output[b][oc][i-ki][j-kj]
        // where 0 <= i-ki < out_h and 0 <= j-kj < out_w
        Tensor::from_fn(input.shape().clone(), |idx| {
            let (b, ic, i, j) = (idx[0], idx[1], idx[2], idx[3]);
            let mut sum = S::ZERO;
            for oc in 0..self.out_channels {
                for ki in 0..kh {
                    for kj in 0..kw {
                        if i >= ki && (i - ki) < out_h && j >= kj && (j - kj) < out_w {
                            sum += self.weight.data.get(&[oc, ic, ki, kj])
                                * grad_output.get(&[b, oc, i - ki, j - kj]);
                        }
                    }
                }
            }
            sum
        })
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
