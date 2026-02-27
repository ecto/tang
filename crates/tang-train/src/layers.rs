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

/// Layer normalization over the last dimension.
///
/// Input: `[batch, features]`
/// Output: `[batch, features]`
///
/// Normalizes each row to zero mean and unit variance, then applies
/// a learnable affine transform (gamma * x + beta).
pub struct LayerNorm<S: Scalar> {
    pub gamma: Parameter<S>, // [features]
    pub beta: Parameter<S>,  // [features]
    eps: f64,
    features: usize,
    cached_input: Option<Tensor<S>>,
    cached_norm: Option<Tensor<S>>, // (input - mean) / std
}

impl<S: Scalar> LayerNorm<S> {
    pub fn new(features: usize) -> Self {
        Self {
            gamma: Parameter::new(Tensor::ones(Shape::from_slice(&[features]))),
            beta: Parameter::new(Tensor::zeros(Shape::from_slice(&[features]))),
            eps: 1e-5,
            features,
            cached_input: None,
            cached_norm: None,
        }
    }
}

impl<S: Scalar> Module<S> for LayerNorm<S> {
    fn forward(&mut self, input: &Tensor<S>) -> Tensor<S> {
        assert_eq!(input.ndim(), 2, "LayerNorm input must be [batch, features]");
        let batch = input.shape()[0];
        let features = input.shape()[1];
        assert_eq!(features, self.features);

        self.cached_input = Some(input.clone());

        let eps = S::from_f64(self.eps);
        let n = S::from_f64(features as f64);

        // Compute per-row mean and variance, then normalize
        let mut norm_data = alloc::vec![S::ZERO; batch * features];
        let mut out_data = alloc::vec![S::ZERO; batch * features];

        for b in 0..batch {
            // mean
            let mut mean = S::ZERO;
            for f in 0..features {
                mean += input.get(&[b, f]);
            }
            mean = mean / n;

            // variance
            let mut var = S::ZERO;
            for f in 0..features {
                let diff = input.get(&[b, f]) - mean;
                var += diff * diff;
            }
            var = var / n;

            let inv_std = (var + eps).sqrt();

            for f in 0..features {
                let x_norm = (input.get(&[b, f]) - mean) / inv_std;
                norm_data[b * features + f] = x_norm;
                out_data[b * features + f] =
                    self.gamma.data.get(&[f]) * x_norm + self.beta.data.get(&[f]);
            }
        }

        let norm = Tensor::new(norm_data, input.shape().clone());
        self.cached_norm = Some(norm);
        Tensor::new(out_data, input.shape().clone())
    }

    fn backward(&mut self, grad_output: &Tensor<S>) -> Tensor<S> {
        let input = self
            .cached_input
            .as_ref()
            .expect("must call forward before backward");
        let cached_norm = self
            .cached_norm
            .as_ref()
            .expect("must call forward before backward");

        let batch = input.shape()[0];
        let features = input.shape()[1];
        let eps = S::from_f64(self.eps);
        let n = S::from_f64(features as f64);

        // grad_gamma[f] = sum over batch of (grad_output[b,f] * cached_norm[b,f])
        let grad_gamma = Tensor::from_fn(Shape::from_slice(&[features]), |idx| {
            let f = idx[0];
            let mut sum = S::ZERO;
            for b in 0..batch {
                sum += grad_output.get(&[b, f]) * cached_norm.get(&[b, f]);
            }
            sum
        });
        self.gamma.accumulate_grad(&grad_gamma);

        // grad_beta[f] = sum over batch of grad_output[b,f]
        let grad_beta = Tensor::from_fn(Shape::from_slice(&[features]), |idx| {
            let f = idx[0];
            let mut sum = S::ZERO;
            for b in 0..batch {
                sum += grad_output.get(&[b, f]);
            }
            sum
        });
        self.beta.accumulate_grad(&grad_beta);

        // grad_input: standard layer norm backward
        // dy_hat[b,f] = grad_output[b,f] * gamma[f]
        // grad_input[b,f] = (1/std) * (dy_hat - mean(dy_hat) - x_norm * mean(dy_hat * x_norm))
        //   where means are over the feature dimension
        Tensor::from_fn(input.shape().clone(), |idx| {
            let b = idx[0];
            let f = idx[1];

            // Recompute std for this row
            let mut mean = S::ZERO;
            for j in 0..features {
                mean += input.get(&[b, j]);
            }
            mean = mean / n;

            let mut var = S::ZERO;
            for j in 0..features {
                let diff = input.get(&[b, j]) - mean;
                var += diff * diff;
            }
            var = var / n;
            let inv_std = S::from_f64(1.0) / (var + eps).sqrt();

            // dy_hat = grad_output * gamma (scaled gradient)
            // Compute mean(dy_hat) and mean(dy_hat * x_norm) over features
            let mut mean_dy = S::ZERO;
            let mut mean_dy_xn = S::ZERO;
            for j in 0..features {
                let dy_hat = grad_output.get(&[b, j]) * self.gamma.data.get(&[j]);
                mean_dy += dy_hat;
                mean_dy_xn += dy_hat * cached_norm.get(&[b, j]);
            }
            mean_dy = mean_dy / n;
            mean_dy_xn = mean_dy_xn / n;

            let dy_hat = grad_output.get(&[b, f]) * self.gamma.data.get(&[f]);
            (dy_hat - mean_dy - cached_norm.get(&[b, f]) * mean_dy_xn) * inv_std
        })
    }

    fn parameters(&self) -> Vec<&Parameter<S>> {
        alloc::vec![&self.gamma, &self.beta]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<S>> {
        alloc::vec![&mut self.gamma, &mut self.beta]
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<S>)> {
        alloc::vec![
            (String::from("gamma"), &self.gamma),
            (String::from("beta"), &self.beta),
        ]
    }

    fn named_parameters_mut(&mut self) -> Vec<(String, &mut Parameter<S>)> {
        alloc::vec![
            (String::from("gamma"), &mut self.gamma),
            (String::from("beta"), &mut self.beta),
        ]
    }
}

/// Multi-head scaled dot-product attention.
///
/// Input: `[seq_len, d_model]` (single sequence)
/// Output: `[seq_len, d_model]`
///
/// Computes Q, K, V projections, splits into heads, applies scaled dot-product
/// attention per head, concatenates, and projects output.
pub struct MultiHeadAttention<S: Scalar> {
    wq: Linear<S>,
    wk: Linear<S>,
    wv: Linear<S>,
    wo: Linear<S>,
    num_heads: usize,
    head_dim: usize,
    d_model: usize,
    cached_q: Option<Tensor<S>>,
    cached_k: Option<Tensor<S>>,
    cached_v: Option<Tensor<S>>,
    cached_attn: Option<Tensor<S>>, // softmax weights [num_heads, seq_len, seq_len]
    cached_input: Option<Tensor<S>>,
}

impl<S: Scalar> MultiHeadAttention<S> {
    pub fn new(d_model: usize, num_heads: usize, seed: u64) -> Self {
        assert!(
            d_model % num_heads == 0,
            "d_model must be divisible by num_heads"
        );
        let head_dim = d_model / num_heads;
        Self {
            wq: Linear::new(d_model, d_model, seed),
            wk: Linear::new(d_model, d_model, seed.wrapping_add(1)),
            wv: Linear::new(d_model, d_model, seed.wrapping_add(2)),
            wo: Linear::new(d_model, d_model, seed.wrapping_add(3)),
            num_heads,
            head_dim,
            d_model,
            cached_q: None,
            cached_k: None,
            cached_v: None,
            cached_attn: None,
            cached_input: None,
        }
    }

    /// Softmax per row of a [rows, cols] tensor.
    fn softmax_2d(input: &Tensor<S>) -> Tensor<S> {
        let rows = input.shape()[0];
        let cols = input.shape()[1];
        let mut data = alloc::vec![S::ZERO; rows * cols];
        for r in 0..rows {
            // Numerical stability: subtract row max
            let mut max_val = input.get(&[r, 0]);
            for c in 1..cols {
                let v = input.get(&[r, c]);
                if v > max_val {
                    max_val = v;
                }
            }
            let mut sum = S::ZERO;
            for c in 0..cols {
                let e = (input.get(&[r, c]) - max_val).exp();
                data[r * cols + c] = e;
                sum += e;
            }
            for c in 0..cols {
                data[r * cols + c] = data[r * cols + c] / sum;
            }
        }
        Tensor::new(data, input.shape().clone())
    }
}

impl<S: Scalar> Module<S> for MultiHeadAttention<S> {
    fn forward(&mut self, input: &Tensor<S>) -> Tensor<S> {
        assert_eq!(input.ndim(), 2, "MultiHeadAttention input must be [seq_len, d_model]");
        let seq_len = input.shape()[0];
        assert_eq!(input.shape()[1], self.d_model);

        self.cached_input = Some(input.clone());

        // Project Q, K, V: [seq_len, d_model] -> [seq_len, d_model]
        let q_full = self.wq.forward(input);
        let k_full = self.wk.forward(input);
        let v_full = self.wv.forward(input);

        self.cached_q = Some(q_full.clone());
        self.cached_k = Some(k_full.clone());
        self.cached_v = Some(v_full.clone());

        let scale = S::from_f64(1.0 / (self.head_dim as f64).sqrt());

        // For each head, extract [seq_len, head_dim], compute attention, store results
        // attn_weights: [num_heads, seq_len, seq_len]
        let mut attn_data = alloc::vec![S::ZERO; self.num_heads * seq_len * seq_len];
        // concat output: [seq_len, d_model]
        let mut out_data = alloc::vec![S::ZERO; seq_len * self.d_model];

        for h in 0..self.num_heads {
            let offset = h * self.head_dim;

            // Extract Q_h, K_h: [seq_len, head_dim]
            let q_h = Tensor::from_fn(
                Shape::from_slice(&[seq_len, self.head_dim]),
                |idx| q_full.get(&[idx[0], offset + idx[1]]),
            );
            let k_h = Tensor::from_fn(
                Shape::from_slice(&[seq_len, self.head_dim]),
                |idx| k_full.get(&[idx[0], offset + idx[1]]),
            );
            let v_h = Tensor::from_fn(
                Shape::from_slice(&[seq_len, self.head_dim]),
                |idx| v_full.get(&[idx[0], offset + idx[1]]),
            );

            // scores = Q_h @ K_h^T / sqrt(head_dim) -> [seq_len, seq_len]
            let scores = q_h.matmul(&k_h.transpose()).scale(scale);

            // softmax per row
            let attn_weights = Self::softmax_2d(&scores);

            // Store attention weights for backward
            for i in 0..seq_len {
                for j in 0..seq_len {
                    attn_data[h * seq_len * seq_len + i * seq_len + j] =
                        attn_weights.get(&[i, j]);
                }
            }

            // attn_out = attn_weights @ V_h -> [seq_len, head_dim]
            let attn_out = attn_weights.matmul(&v_h);

            // Write into concat output at the right offset
            for i in 0..seq_len {
                for d in 0..self.head_dim {
                    out_data[i * self.d_model + offset + d] = attn_out.get(&[i, d]);
                }
            }
        }

        self.cached_attn = Some(Tensor::new(
            attn_data,
            Shape::from_slice(&[self.num_heads, seq_len, seq_len]),
        ));

        // concat: [seq_len, d_model] -> output projection
        let concat = Tensor::new(out_data, Shape::from_slice(&[seq_len, self.d_model]));
        self.wo.forward(&concat)
    }

    fn backward(&mut self, grad_output: &Tensor<S>) -> Tensor<S> {
        let q_full = self.cached_q.as_ref().expect("must call forward before backward");
        let k_full = self.cached_k.as_ref().expect("must call forward before backward");
        let v_full = self.cached_v.as_ref().expect("must call forward before backward");
        let attn = self.cached_attn.as_ref().expect("must call forward before backward");

        let seq_len = q_full.shape()[0];
        let scale = S::from_f64(1.0 / (self.head_dim as f64).sqrt());

        // Backward through Wo: grad_concat = wo.backward(grad_output)
        let grad_concat = self.wo.backward(grad_output);

        // Accumulate gradients for Q, K, V projections
        let mut grad_q_full = alloc::vec![S::ZERO; seq_len * self.d_model];
        let mut grad_k_full = alloc::vec![S::ZERO; seq_len * self.d_model];
        let mut grad_v_full = alloc::vec![S::ZERO; seq_len * self.d_model];

        for h in 0..self.num_heads {
            let offset = h * self.head_dim;

            // Extract cached head tensors
            let v_h = Tensor::from_fn(
                Shape::from_slice(&[seq_len, self.head_dim]),
                |idx| v_full.get(&[idx[0], offset + idx[1]]),
            );
            let q_h = Tensor::from_fn(
                Shape::from_slice(&[seq_len, self.head_dim]),
                |idx| q_full.get(&[idx[0], offset + idx[1]]),
            );
            let k_h = Tensor::from_fn(
                Shape::from_slice(&[seq_len, self.head_dim]),
                |idx| k_full.get(&[idx[0], offset + idx[1]]),
            );

            // Extract attention weights for this head: [seq_len, seq_len]
            let attn_h = Tensor::from_fn(
                Shape::from_slice(&[seq_len, seq_len]),
                |idx| attn.get(&[h, idx[0], idx[1]]),
            );

            // grad_concat_h: [seq_len, head_dim]
            let grad_concat_h = Tensor::from_fn(
                Shape::from_slice(&[seq_len, self.head_dim]),
                |idx| grad_concat.get(&[idx[0], offset + idx[1]]),
            );

            // Backward through attn_out = attn_h @ V_h
            // grad_attn_h = grad_concat_h @ V_h^T  -> [seq_len, seq_len]
            let grad_attn_h = grad_concat_h.matmul(&v_h.transpose());
            // grad_v_h = attn_h^T @ grad_concat_h   -> [seq_len, head_dim]
            let grad_v_h = attn_h.transpose().matmul(&grad_concat_h);

            // Backward through softmax: grad_scores
            // For softmax: dL/ds_i = sum_j (dL/da_j * a_j * (delta_ij - a_i))
            //            = a_i * (dL/da_i - sum_j(dL/da_j * a_j))
            let grad_scores = Tensor::from_fn(
                Shape::from_slice(&[seq_len, seq_len]),
                |idx| {
                    let (i, j) = (idx[0], idx[1]);
                    let a_ij = attn_h.get(&[i, j]);
                    // dot = sum_k (grad_attn_h[i,k] * attn_h[i,k])
                    let mut dot = S::ZERO;
                    for k in 0..seq_len {
                        dot += grad_attn_h.get(&[i, k]) * attn_h.get(&[i, k]);
                    }
                    a_ij * (grad_attn_h.get(&[i, j]) - dot)
                },
            );

            // Backward through scores = Q_h @ K_h^T * scale
            let grad_scores_scaled = grad_scores.scale(scale);

            // grad_q_h = grad_scores_scaled @ K_h  -> [seq_len, head_dim]
            let grad_q_h = grad_scores_scaled.matmul(&k_h);
            // grad_k_h = grad_scores_scaled^T @ Q_h -> [seq_len, head_dim]
            let grad_k_h = grad_scores_scaled.transpose().matmul(&q_h);

            // Scatter back to full gradients
            for i in 0..seq_len {
                for d in 0..self.head_dim {
                    grad_q_full[i * self.d_model + offset + d] += grad_q_h.get(&[i, d]);
                    grad_k_full[i * self.d_model + offset + d] += grad_k_h.get(&[i, d]);
                    grad_v_full[i * self.d_model + offset + d] += grad_v_h.get(&[i, d]);
                }
            }
        }

        let grad_q = Tensor::new(grad_q_full, Shape::from_slice(&[seq_len, self.d_model]));
        let grad_k = Tensor::new(grad_k_full, Shape::from_slice(&[seq_len, self.d_model]));
        let grad_v = Tensor::new(grad_v_full, Shape::from_slice(&[seq_len, self.d_model]));

        // Backward through Wq, Wk, Wv projections
        let grad_input_q = self.wq.backward(&grad_q);
        let grad_input_k = self.wk.backward(&grad_k);
        let grad_input_v = self.wv.backward(&grad_v);

        // All three projections share the same input, so gradients add
        grad_input_q.add(&grad_input_k).add(&grad_input_v)
    }

    fn parameters(&self) -> Vec<&Parameter<S>> {
        let mut params = self.wq.parameters();
        params.extend(self.wk.parameters());
        params.extend(self.wv.parameters());
        params.extend(self.wo.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<S>> {
        let mut params = self.wq.parameters_mut();
        params.extend(self.wk.parameters_mut());
        params.extend(self.wv.parameters_mut());
        params.extend(self.wo.parameters_mut());
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<S>)> {
        let mut params = Vec::new();
        for (name, p) in self.wq.named_parameters() {
            params.push((alloc::format!("wq.{}", name), p));
        }
        for (name, p) in self.wk.named_parameters() {
            params.push((alloc::format!("wk.{}", name), p));
        }
        for (name, p) in self.wv.named_parameters() {
            params.push((alloc::format!("wv.{}", name), p));
        }
        for (name, p) in self.wo.named_parameters() {
            params.push((alloc::format!("wo.{}", name), p));
        }
        params
    }

    fn named_parameters_mut(&mut self) -> Vec<(String, &mut Parameter<S>)> {
        let mut params = Vec::new();
        for (name, p) in self.wq.named_parameters_mut() {
            params.push((alloc::format!("wq.{}", name), p));
        }
        for (name, p) in self.wk.named_parameters_mut() {
            params.push((alloc::format!("wk.{}", name), p));
        }
        for (name, p) in self.wv.named_parameters_mut() {
            params.push((alloc::format!("wv.{}", name), p));
        }
        for (name, p) in self.wo.named_parameters_mut() {
            params.push((alloc::format!("wo.{}", name), p));
        }
        params
    }
}

/// Pre-norm transformer block.
///
/// Input: `[seq_len, d_model]`
/// Output: `[seq_len, d_model]`
///
/// Architecture (pre-norm):
/// ```text
/// x -> LayerNorm -> MultiHeadAttention -> + residual -> LayerNorm -> FFN -> + residual
/// ```
pub struct TransformerBlock<S: Scalar> {
    attn: MultiHeadAttention<S>,
    ln1: LayerNorm<S>,
    ln2: LayerNorm<S>,
    ff1: Linear<S>,
    ff2: Linear<S>,
    relu: ReLU<S>,
    cached_residual1: Option<Tensor<S>>,
    cached_residual2: Option<Tensor<S>>,
}

impl<S: Scalar> TransformerBlock<S> {
    pub fn new(d_model: usize, num_heads: usize, ff_dim: usize, seed: u64) -> Self {
        Self {
            attn: MultiHeadAttention::new(d_model, num_heads, seed),
            ln1: LayerNorm::new(d_model),
            ln2: LayerNorm::new(d_model),
            ff1: Linear::new(d_model, ff_dim, seed.wrapping_add(10)),
            ff2: Linear::new(ff_dim, d_model, seed.wrapping_add(11)),
            relu: ReLU::new(),
            cached_residual1: None,
            cached_residual2: None,
        }
    }
}

impl<S: Scalar> Module<S> for TransformerBlock<S> {
    fn forward(&mut self, input: &Tensor<S>) -> Tensor<S> {
        // Pre-norm: LN before attention/FFN
        let residual1 = input.clone();

        let x = self.ln1.forward(input);
        let x = self.attn.forward(&x);
        let x = x.add(&residual1);

        self.cached_residual1 = Some(residual1);

        let residual2 = x.clone();

        let x = self.ln2.forward(&x);
        let x = self.ff1.forward(&x);
        let x = self.relu.forward(&x);
        let x = self.ff2.forward(&x);
        let x = x.add(&residual2);

        self.cached_residual2 = Some(residual2);

        x
    }

    fn backward(&mut self, grad_output: &Tensor<S>) -> Tensor<S> {
        // Backward through second residual: x = ff_out + residual2
        // grad flows to both branches
        let grad_ff = grad_output.clone();
        let grad_res2 = grad_output.clone();

        // Backward through FFN: ff2 -> relu -> ff1 -> ln2
        let grad = self.ff2.backward(&grad_ff);
        let grad = self.relu.backward(&grad);
        let grad = self.ff1.backward(&grad);
        let grad = self.ln2.backward(&grad);

        // Add residual2 gradient
        let grad = grad.add(&grad_res2);

        // Backward through first residual: x = attn_out + residual1
        let grad_attn = grad.clone();
        let grad_res1 = grad.clone();

        // Backward through attention -> ln1
        let grad = self.attn.backward(&grad_attn);
        let grad = self.ln1.backward(&grad);

        // Add residual1 gradient
        grad.add(&grad_res1)
    }

    fn parameters(&self) -> Vec<&Parameter<S>> {
        let mut params = self.attn.parameters();
        params.extend(self.ln1.parameters());
        params.extend(self.ln2.parameters());
        params.extend(self.ff1.parameters());
        params.extend(self.ff2.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<S>> {
        let mut params = self.attn.parameters_mut();
        params.extend(self.ln1.parameters_mut());
        params.extend(self.ln2.parameters_mut());
        params.extend(self.ff1.parameters_mut());
        params.extend(self.ff2.parameters_mut());
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<S>)> {
        let mut params = Vec::new();
        for (name, p) in self.attn.named_parameters() {
            params.push((alloc::format!("attn.{}", name), p));
        }
        for (name, p) in self.ln1.named_parameters() {
            params.push((alloc::format!("ln1.{}", name), p));
        }
        for (name, p) in self.ln2.named_parameters() {
            params.push((alloc::format!("ln2.{}", name), p));
        }
        for (name, p) in self.ff1.named_parameters() {
            params.push((alloc::format!("ff1.{}", name), p));
        }
        for (name, p) in self.ff2.named_parameters() {
            params.push((alloc::format!("ff2.{}", name), p));
        }
        params
    }

    fn named_parameters_mut(&mut self) -> Vec<(String, &mut Parameter<S>)> {
        let mut params = Vec::new();
        for (name, p) in self.attn.named_parameters_mut() {
            params.push((alloc::format!("attn.{}", name), p));
        }
        for (name, p) in self.ln1.named_parameters_mut() {
            params.push((alloc::format!("ln1.{}", name), p));
        }
        for (name, p) in self.ln2.named_parameters_mut() {
            params.push((alloc::format!("ln2.{}", name), p));
        }
        for (name, p) in self.ff1.named_parameters_mut() {
            params.push((alloc::format!("ff1.{}", name), p));
        }
        for (name, p) in self.ff2.named_parameters_mut() {
            params.push((alloc::format!("ff2.{}", name), p));
        }
        params
    }
}

// ---------------------------------------------------------------------------
// SiLU (Swish) activation module
// ---------------------------------------------------------------------------

/// SiLU activation (x * sigmoid(x)), used in LLaMA/Mistral FFN.
pub struct SiLU<S: Scalar> {
    cached_input: Option<Tensor<S>>,
}

impl<S: Scalar> SiLU<S> {
    pub fn new() -> Self {
        Self { cached_input: None }
    }
}

impl<S: Scalar> Default for SiLU<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Scalar> Module<S> for SiLU<S> {
    fn forward(&mut self, input: &Tensor<S>) -> Tensor<S> {
        self.cached_input = Some(input.clone());
        input.silu()
    }

    fn backward(&mut self, grad_output: &Tensor<S>) -> Tensor<S> {
        let input = self
            .cached_input
            .as_ref()
            .expect("must call forward before backward");
        // d/dx silu(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        //              = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        let one = S::from_f64(1.0);
        let sig = input.sigmoid();
        let dsilu = Tensor::from_fn(input.shape().clone(), |idx| {
            let s = sig.get(idx);
            let x = input.get(idx);
            s * (one + x * (one - s))
        });
        grad_output.mul(&dsilu)
    }

    fn parameters(&self) -> Vec<&Parameter<S>> {
        Vec::new()
    }
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<S>> {
        Vec::new()
    }
}

// ---------------------------------------------------------------------------
// GELU activation module (exact and approximate)
// ---------------------------------------------------------------------------

/// GELU activation with exact or tanh-approximate mode.
pub struct GELU<S: Scalar> {
    approximate: bool,
    cached_input: Option<Tensor<S>>,
    _marker: core::marker::PhantomData<S>,
}

impl<S: Scalar> GELU<S> {
    /// Create GELU with exact computation.
    pub fn new() -> Self {
        Self {
            approximate: false,
            cached_input: None,
            _marker: core::marker::PhantomData,
        }
    }

    /// Create GELU with tanh approximation (used by many pretrained models).
    pub fn approximate() -> Self {
        Self {
            approximate: true,
            cached_input: None,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<S: Scalar> Default for GELU<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Scalar> Module<S> for GELU<S> {
    fn forward(&mut self, input: &Tensor<S>) -> Tensor<S> {
        self.cached_input = Some(input.clone());
        input.gelu()
    }

    fn backward(&mut self, grad_output: &Tensor<S>) -> Tensor<S> {
        let input = self
            .cached_input
            .as_ref()
            .expect("must call forward before backward");
        let half = S::from_f64(0.5);
        let one = S::from_f64(1.0);
        let coeff = S::from_f64(0.7978845608028654); // sqrt(2/pi)
        let k = S::from_f64(0.044715);

        if self.approximate {
            // d/dx gelu_approx(x) = 0.5 * (1 + t) + 0.5 * x * (1 - t^2) * coeff * (1 + 3*k*x^2)
            // where t = tanh(coeff * (x + k * x^3))
            let three_k = S::from_f64(3.0 * 0.044715);
            let dgelu = Tensor::from_fn(input.shape().clone(), |idx| {
                let x = input.get(idx);
                let inner = coeff * (x + k * x * x * x);
                let t = inner.tanh();
                half * (one + t) + half * x * (one - t * t) * coeff * (one + three_k * x * x)
            });
            grad_output.mul(&dgelu)
        } else {
            // Same formula — the forward uses the tanh approximation in both cases
            let three_k = S::from_f64(3.0 * 0.044715);
            let dgelu = Tensor::from_fn(input.shape().clone(), |idx| {
                let x = input.get(idx);
                let inner = coeff * (x + k * x * x * x);
                let t = inner.tanh();
                half * (one + t) + half * x * (one - t * t) * coeff * (one + three_k * x * x)
            });
            grad_output.mul(&dgelu)
        }
    }

    fn parameters(&self) -> Vec<&Parameter<S>> {
        Vec::new()
    }
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<S>> {
        Vec::new()
    }
}

// ---------------------------------------------------------------------------
// RMSNorm — used by LLaMA, Mistral, Gemma
// ---------------------------------------------------------------------------

/// Root Mean Square Layer Normalization.
///
/// `y = x / sqrt(mean(x^2) + eps) * weight`
///
/// Simpler and faster than LayerNorm — no mean subtraction or bias.
pub struct RMSNorm<S: Scalar> {
    pub weight: Parameter<S>, // [features]
    eps: f64,
    features: usize,
    cached_input: Option<Tensor<S>>,
    cached_rms: Option<Tensor<S>>, // [batch] — rms values per row
}

impl<S: Scalar> RMSNorm<S> {
    pub fn new(features: usize) -> Self {
        Self {
            weight: Parameter::new(Tensor::ones(Shape::from_slice(&[features]))),
            eps: 1e-6,
            features,
            cached_input: None,
            cached_rms: None,
        }
    }
}

impl<S: Scalar> Module<S> for RMSNorm<S> {
    fn forward(&mut self, input: &Tensor<S>) -> Tensor<S> {
        assert_eq!(input.ndim(), 2, "RMSNorm input must be [batch, features]");
        let batch = input.shape()[0];
        let features = input.shape()[1];
        assert_eq!(features, self.features);

        self.cached_input = Some(input.clone());

        let eps = S::from_f64(self.eps);
        let n = S::from_f64(features as f64);

        let mut rms_data = alloc::vec![S::ZERO; batch];
        let mut out_data = alloc::vec![S::ZERO; batch * features];

        for b in 0..batch {
            // mean(x^2)
            let mut sq_sum = S::ZERO;
            for f in 0..features {
                let x = input.get(&[b, f]);
                sq_sum += x * x;
            }
            let rms = (sq_sum / n + eps).sqrt();
            rms_data[b] = rms;

            for f in 0..features {
                let x_norm = input.get(&[b, f]) / rms;
                out_data[b * features + f] = self.weight.data.get(&[f]) * x_norm;
            }
        }

        self.cached_rms = Some(Tensor::new(rms_data, Shape::from_slice(&[batch])));
        Tensor::new(out_data, input.shape().clone())
    }

    fn backward(&mut self, grad_output: &Tensor<S>) -> Tensor<S> {
        let input = self
            .cached_input
            .as_ref()
            .expect("must call forward before backward");
        let rms = self
            .cached_rms
            .as_ref()
            .expect("must call forward before backward");

        let batch = input.shape()[0];
        let features = input.shape()[1];
        let n = S::from_f64(features as f64);

        // Gradient w.r.t. weight: sum over batch of grad_output * (x / rms)
        let mut grad_weight = alloc::vec![S::ZERO; features];
        let mut grad_input_data = alloc::vec![S::ZERO; batch * features];

        for b in 0..batch {
            let r = rms.get(&[b]);
            let inv_r = S::from_f64(1.0) / r;

            // Accumulate weight gradient
            for f in 0..features {
                let x_norm = input.get(&[b, f]) * inv_r;
                grad_weight[f] += grad_output.get(&[b, f]) * x_norm;
            }

            // grad_input: need to backprop through x / rms(x) * weight
            // d/dx_i (x_i / rms * w_i) considering rms depends on all x
            // = w_i / rms - w_i * x_i * (x_i / (n * rms^3))... simplified:
            // = (w * grad_out) / rms - x * dot(w * grad_out, x) / (n * rms^3)
            let mut dot = S::ZERO;
            for f in 0..features {
                dot += self.weight.data.get(&[f]) * grad_output.get(&[b, f]) * input.get(&[b, f]);
            }

            for f in 0..features {
                let w_grad = self.weight.data.get(&[f]) * grad_output.get(&[b, f]);
                grad_input_data[b * features + f] =
                    (w_grad - input.get(&[b, f]) * dot / (n * r * r)) * inv_r;
            }
        }

        // Accumulate weight gradient
        let grad_w = Tensor::new(grad_weight, Shape::from_slice(&[features]));
        self.weight.accumulate_grad(&grad_w);

        Tensor::new(grad_input_data, input.shape().clone())
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

// ---------------------------------------------------------------------------
// Rotary Position Embedding (RoPE)
// ---------------------------------------------------------------------------

/// Rotary Position Embedding — applies position-dependent rotation to Q and K.
///
/// Used by LLaMA, Mistral, GPT-NeoX. Encodes relative position information
/// by rotating pairs of dimensions by position-dependent angles.
pub struct RotaryEmbedding<S: Scalar> {
    dim: usize,
    max_seq_len: usize,
    base: f64,
    cos_cache: Tensor<S>, // [max_seq_len, dim/2]
    sin_cache: Tensor<S>, // [max_seq_len, dim/2]
}

impl<S: Scalar> RotaryEmbedding<S> {
    pub fn new(dim: usize, max_seq_len: usize) -> Self {
        Self::with_base(dim, max_seq_len, 10000.0)
    }

    pub fn with_base(dim: usize, max_seq_len: usize, base: f64) -> Self {
        assert!(dim % 2 == 0, "RoPE dim must be even");
        let half = dim / 2;

        // Precompute cos/sin tables: theta_i = 1 / base^(2i/dim)
        let cos_cache = Tensor::from_fn(
            Shape::from_slice(&[max_seq_len, half]),
            |idx| {
                let pos = idx[0] as f64;
                let i = idx[1] as f64;
                let theta = pos / base.powf(2.0 * i / dim as f64);
                S::from_f64(theta.cos())
            },
        );
        let sin_cache = Tensor::from_fn(
            Shape::from_slice(&[max_seq_len, half]),
            |idx| {
                let pos = idx[0] as f64;
                let i = idx[1] as f64;
                let theta = pos / base.powf(2.0 * i / dim as f64);
                S::from_f64(theta.sin())
            },
        );

        Self {
            dim,
            max_seq_len,
            base,
            cos_cache,
            sin_cache,
        }
    }

    /// Apply RoPE to a tensor of shape [seq_len, dim].
    /// Rotates pairs (x[2i], x[2i+1]) by position-dependent angle.
    pub fn apply(&self, x: &Tensor<S>, offset: usize) -> Tensor<S> {
        assert_eq!(x.ndim(), 2);
        let seq_len = x.shape()[0];
        assert_eq!(x.shape()[1], self.dim);
        assert!(
            offset + seq_len <= self.max_seq_len,
            "sequence exceeds max_seq_len"
        );
        let half = self.dim / 2;

        Tensor::from_fn(x.shape().clone(), |idx| {
            let pos = idx[0];
            let d = idx[1];
            let pair = d / 2;
            let cos_val = self.cos_cache.get(&[offset + pos, pair]);
            let sin_val = self.sin_cache.get(&[offset + pos, pair]);

            if d % 2 == 0 {
                // x_even * cos - x_odd * sin
                x.get(&[pos, d]) * cos_val - x.get(&[pos, d + 1]) * sin_val
            } else {
                // x_even * sin + x_odd * cos
                x.get(&[pos, d - 1]) * sin_val + x.get(&[pos, d]) * cos_val
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Grouped-Query Attention (GQA)
// ---------------------------------------------------------------------------

/// Grouped-Query Attention — generalizes MHA, MQA, and standard attention.
///
/// - `num_kv_heads == num_heads`: standard multi-head attention
/// - `num_kv_heads == 1`: multi-query attention (MQA)
/// - `1 < num_kv_heads < num_heads`: grouped-query attention (GQA)
///
/// Used by LLaMA 2 70B, Mistral, etc. for efficient inference.
pub struct GroupedQueryAttention<S: Scalar> {
    wq: Linear<S>,
    wk: Linear<S>,
    wv: Linear<S>,
    wo: Linear<S>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    d_model: usize,
    cached_q: Option<Tensor<S>>,
    cached_k: Option<Tensor<S>>,
    cached_v: Option<Tensor<S>>,
    cached_attn: Option<Tensor<S>>,
    cached_input: Option<Tensor<S>>,
}

impl<S: Scalar> GroupedQueryAttention<S> {
    pub fn new(
        d_model: usize,
        num_heads: usize,
        num_kv_heads: usize,
        seed: u64,
    ) -> Self {
        assert!(
            num_heads % num_kv_heads == 0,
            "num_heads must be divisible by num_kv_heads"
        );
        let head_dim = d_model / num_heads;
        let kv_dim = num_kv_heads * head_dim;

        Self {
            wq: Linear::new(d_model, d_model, seed),
            wk: Linear::new(d_model, kv_dim, seed.wrapping_add(1)),
            wv: Linear::new(d_model, kv_dim, seed.wrapping_add(2)),
            wo: Linear::new(d_model, d_model, seed.wrapping_add(3)),
            num_heads,
            num_kv_heads,
            head_dim,
            d_model,
            cached_q: None,
            cached_k: None,
            cached_v: None,
            cached_attn: None,
            cached_input: None,
        }
    }

    /// Softmax per row of a [rows, cols] tensor.
    fn softmax_2d(input: &Tensor<S>) -> Tensor<S> {
        let rows = input.shape()[0];
        let cols = input.shape()[1];
        let mut data = alloc::vec![S::ZERO; rows * cols];
        for r in 0..rows {
            let mut max_val = input.get(&[r, 0]);
            for c in 1..cols {
                let v = input.get(&[r, c]);
                if v > max_val {
                    max_val = v;
                }
            }
            let mut sum = S::ZERO;
            for c in 0..cols {
                let e = (input.get(&[r, c]) - max_val).exp();
                data[r * cols + c] = e;
                sum += e;
            }
            for c in 0..cols {
                data[r * cols + c] = data[r * cols + c] / sum;
            }
        }
        Tensor::new(data, input.shape().clone())
    }
}

impl<S: Scalar> Module<S> for GroupedQueryAttention<S> {
    fn forward(&mut self, input: &Tensor<S>) -> Tensor<S> {
        assert_eq!(input.ndim(), 2, "GQA input must be [seq_len, d_model]");
        let seq_len = input.shape()[0];
        assert_eq!(input.shape()[1], self.d_model);

        self.cached_input = Some(input.clone());

        let q_full = self.wq.forward(input);  // [seq_len, d_model]
        let k_full = self.wk.forward(input);  // [seq_len, kv_dim]
        let v_full = self.wv.forward(input);  // [seq_len, kv_dim]

        self.cached_q = Some(q_full.clone());
        self.cached_k = Some(k_full.clone());
        self.cached_v = Some(v_full.clone());

        let scale = S::from_f64(1.0 / (self.head_dim as f64).sqrt());
        let heads_per_kv = self.num_heads / self.num_kv_heads;

        let mut attn_data = alloc::vec![S::ZERO; self.num_heads * seq_len * seq_len];
        let mut out_data = alloc::vec![S::ZERO; seq_len * self.d_model];

        for h in 0..self.num_heads {
            let q_offset = h * self.head_dim;
            let kv_group = h / heads_per_kv;
            let kv_offset = kv_group * self.head_dim;

            let q_h = Tensor::from_fn(
                Shape::from_slice(&[seq_len, self.head_dim]),
                |idx| q_full.get(&[idx[0], q_offset + idx[1]]),
            );
            let k_h = Tensor::from_fn(
                Shape::from_slice(&[seq_len, self.head_dim]),
                |idx| k_full.get(&[idx[0], kv_offset + idx[1]]),
            );
            let v_h = Tensor::from_fn(
                Shape::from_slice(&[seq_len, self.head_dim]),
                |idx| v_full.get(&[idx[0], kv_offset + idx[1]]),
            );

            let scores = q_h.matmul(&k_h.transpose()).scale(scale);
            let attn_weights = Self::softmax_2d(&scores);

            for i in 0..seq_len {
                for j in 0..seq_len {
                    attn_data[h * seq_len * seq_len + i * seq_len + j] =
                        attn_weights.get(&[i, j]);
                }
            }

            let attn_out = attn_weights.matmul(&v_h);
            for i in 0..seq_len {
                for d in 0..self.head_dim {
                    out_data[i * self.d_model + q_offset + d] = attn_out.get(&[i, d]);
                }
            }
        }

        self.cached_attn = Some(Tensor::new(
            attn_data,
            Shape::from_slice(&[self.num_heads, seq_len, seq_len]),
        ));

        let concat = Tensor::new(out_data, Shape::from_slice(&[seq_len, self.d_model]));
        self.wo.forward(&concat)
    }

    fn backward(&mut self, grad_output: &Tensor<S>) -> Tensor<S> {
        let q_full = self.cached_q.as_ref().expect("must call forward before backward");
        let k_full = self.cached_k.as_ref().expect("must call forward before backward");
        let v_full = self.cached_v.as_ref().expect("must call forward before backward");
        let attn = self.cached_attn.as_ref().expect("must call forward before backward");

        let seq_len = q_full.shape()[0];
        let kv_dim = self.num_kv_heads * self.head_dim;
        let scale = S::from_f64(1.0 / (self.head_dim as f64).sqrt());
        let heads_per_kv = self.num_heads / self.num_kv_heads;

        let grad_concat = self.wo.backward(grad_output);

        let mut grad_q_full = alloc::vec![S::ZERO; seq_len * self.d_model];
        let mut grad_k_full = alloc::vec![S::ZERO; seq_len * kv_dim];
        let mut grad_v_full = alloc::vec![S::ZERO; seq_len * kv_dim];

        for h in 0..self.num_heads {
            let q_offset = h * self.head_dim;
            let kv_group = h / heads_per_kv;
            let kv_offset = kv_group * self.head_dim;

            let v_h = Tensor::from_fn(
                Shape::from_slice(&[seq_len, self.head_dim]),
                |idx| v_full.get(&[idx[0], kv_offset + idx[1]]),
            );
            let q_h = Tensor::from_fn(
                Shape::from_slice(&[seq_len, self.head_dim]),
                |idx| q_full.get(&[idx[0], q_offset + idx[1]]),
            );
            let k_h = Tensor::from_fn(
                Shape::from_slice(&[seq_len, self.head_dim]),
                |idx| k_full.get(&[idx[0], kv_offset + idx[1]]),
            );

            let attn_h = Tensor::from_fn(
                Shape::from_slice(&[seq_len, seq_len]),
                |idx| attn.get(&[h, idx[0], idx[1]]),
            );

            let grad_concat_h = Tensor::from_fn(
                Shape::from_slice(&[seq_len, self.head_dim]),
                |idx| grad_concat.get(&[idx[0], q_offset + idx[1]]),
            );

            let grad_attn_h = grad_concat_h.matmul(&v_h.transpose());
            let grad_v_h = attn_h.transpose().matmul(&grad_concat_h);

            let grad_scores = Tensor::from_fn(
                Shape::from_slice(&[seq_len, seq_len]),
                |idx| {
                    let (i, j) = (idx[0], idx[1]);
                    let a_ij = attn_h.get(&[i, j]);
                    let mut dot = S::ZERO;
                    for k in 0..seq_len {
                        dot += grad_attn_h.get(&[i, k]) * attn_h.get(&[i, k]);
                    }
                    a_ij * (grad_attn_h.get(&[i, j]) - dot)
                },
            );

            let grad_scores_scaled = grad_scores.scale(scale);
            let grad_q_h = grad_scores_scaled.matmul(&k_h);
            let grad_k_h = grad_scores_scaled.transpose().matmul(&q_h);

            for i in 0..seq_len {
                for d in 0..self.head_dim {
                    grad_q_full[i * self.d_model + q_offset + d] += grad_q_h.get(&[i, d]);
                    // KV grads accumulate across all Q heads sharing this KV group
                    grad_k_full[i * kv_dim + kv_offset + d] += grad_k_h.get(&[i, d]);
                    grad_v_full[i * kv_dim + kv_offset + d] += grad_v_h.get(&[i, d]);
                }
            }
        }

        let grad_q = Tensor::new(grad_q_full, Shape::from_slice(&[seq_len, self.d_model]));
        let grad_k = Tensor::new(grad_k_full, Shape::from_slice(&[seq_len, kv_dim]));
        let grad_v = Tensor::new(grad_v_full, Shape::from_slice(&[seq_len, kv_dim]));

        let grad_input_q = self.wq.backward(&grad_q);
        let grad_input_k = self.wk.backward(&grad_k);
        let grad_input_v = self.wv.backward(&grad_v);

        grad_input_q.add(&grad_input_k).add(&grad_input_v)
    }

    fn parameters(&self) -> Vec<&Parameter<S>> {
        let mut params = self.wq.parameters();
        params.extend(self.wk.parameters());
        params.extend(self.wv.parameters());
        params.extend(self.wo.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<S>> {
        let mut params = self.wq.parameters_mut();
        params.extend(self.wk.parameters_mut());
        params.extend(self.wv.parameters_mut());
        params.extend(self.wo.parameters_mut());
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<S>)> {
        let mut params = Vec::new();
        for (name, p) in self.wq.named_parameters() {
            params.push((alloc::format!("wq.{}", name), p));
        }
        for (name, p) in self.wk.named_parameters() {
            params.push((alloc::format!("wk.{}", name), p));
        }
        for (name, p) in self.wv.named_parameters() {
            params.push((alloc::format!("wv.{}", name), p));
        }
        for (name, p) in self.wo.named_parameters() {
            params.push((alloc::format!("wo.{}", name), p));
        }
        params
    }

    fn named_parameters_mut(&mut self) -> Vec<(String, &mut Parameter<S>)> {
        let mut params = Vec::new();
        for (name, p) in self.wq.named_parameters_mut() {
            params.push((alloc::format!("wq.{}", name), p));
        }
        for (name, p) in self.wk.named_parameters_mut() {
            params.push((alloc::format!("wk.{}", name), p));
        }
        for (name, p) in self.wv.named_parameters_mut() {
            params.push((alloc::format!("wv.{}", name), p));
        }
        for (name, p) in self.wo.named_parameters_mut() {
            params.push((alloc::format!("wo.{}", name), p));
        }
        params
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
