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

/// 2D convolution layer with padding, stride, and dilation support.
///
/// Input: `[batch, in_channels, height, width]`
/// Output: `[batch, out_channels, out_h, out_w]`
///
/// where `out_h = (height + 2*padding - dilation*(kh-1) - 1) / stride + 1`
pub struct Conv2d<S: Scalar> {
    pub weight: Parameter<S>, // [out_channels, in_channels, kh, kw]
    pub bias: Parameter<S>,   // [out_channels]
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    cached_input: Option<Tensor<S>>,
}

impl<S: Scalar> Conv2d<S> {
    /// Create Conv2d with default stride=1, padding=0, dilation=1.
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, seed: u64) -> Self {
        Self::with_options(in_channels, out_channels, kernel_size, 1, 0, 1, seed)
    }

    /// Create Conv2d with full options.
    pub fn with_options(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        seed: u64,
    ) -> Self {
        assert!(stride > 0, "stride must be > 0");
        assert!(dilation > 0, "dilation must be > 0");
        Self {
            weight: Parameter::randn(
                Shape::from_slice(&[out_channels, in_channels, kernel_size, kernel_size]),
                seed,
            ),
            bias: Parameter::new(Tensor::zeros(Shape::from_slice(&[out_channels]))),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            cached_input: None,
        }
    }

    /// Compute output spatial dimension.
    fn out_size(&self, input_size: usize) -> usize {
        let effective_k = self.dilation * (self.kernel_size - 1) + 1;
        (input_size + 2 * self.padding - effective_k) / self.stride + 1
    }

    /// Get input value with padding (returns zero for out-of-bounds).
    fn padded_get(input: &Tensor<S>, b: usize, c: usize, i: isize, j: isize, h: usize, w: usize) -> S {
        if i < 0 || j < 0 || (i as usize) >= h || (j as usize) >= w {
            S::ZERO
        } else {
            input.get(&[b, c, i as usize, j as usize])
        }
    }
}

impl<S: Scalar> Module<S> for Conv2d<S> {
    fn forward(&mut self, input: &Tensor<S>) -> Tensor<S> {
        assert_eq!(input.ndim(), 4, "Conv2d input must be [batch, in_channels, height, width]");
        let batch = input.shape()[0];
        assert_eq!(input.shape()[1], self.in_channels);
        let height = input.shape()[2];
        let width = input.shape()[3];
        let out_h = self.out_size(height);
        let out_w = self.out_size(width);

        self.cached_input = Some(input.clone());
        let pad = self.padding as isize;
        let stride = self.stride;
        let dilation = self.dilation;
        let ksize = self.kernel_size;

        Tensor::from_fn(
            Shape::from_slice(&[batch, self.out_channels, out_h, out_w]),
            |idx| {
                let (b, oc, oh, ow) = (idx[0], idx[1], idx[2], idx[3]);
                let mut sum = self.bias.data.get(&[oc]);
                for c in 0..self.in_channels {
                    for ki in 0..ksize {
                        for kj in 0..ksize {
                            let ih = (oh * stride) as isize - pad + (ki * dilation) as isize;
                            let iw = (ow * stride) as isize - pad + (kj * dilation) as isize;
                            sum += self.weight.data.get(&[oc, c, ki, kj])
                                * Self::padded_get(input, b, c, ih, iw, height, width);
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
        let out_h = self.out_size(height);
        let out_w = self.out_size(width);
        let pad = self.padding as isize;
        let stride = self.stride;
        let dilation = self.dilation;
        let ksize = self.kernel_size;

        // grad_weight
        let grad_w = Tensor::from_fn(self.weight.data.shape().clone(), |idx| {
            let (oc, ic, ki, kj) = (idx[0], idx[1], idx[2], idx[3]);
            let mut sum = S::ZERO;
            for b in 0..batch {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let ih = (oh * stride) as isize - pad + (ki * dilation) as isize;
                        let iw = (ow * stride) as isize - pad + (kj * dilation) as isize;
                        sum += grad_output.get(&[b, oc, oh, ow])
                            * Self::padded_get(input, b, ic, ih, iw, height, width);
                    }
                }
            }
            sum
        });
        self.weight.accumulate_grad(&grad_w);

        // grad_bias
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

        // grad_input[b][ic][i][j] — for each input pixel, sum contributions from all output pixels
        Tensor::from_fn(input.shape().clone(), |idx| {
            let (b, ic, i, j) = (idx[0], idx[1], idx[2], idx[3]);
            let mut sum = S::ZERO;
            for oc in 0..self.out_channels {
                for ki in 0..ksize {
                    for kj in 0..ksize {
                        // i = oh * stride - pad + ki * dilation
                        // oh = (i + pad - ki * dilation) / stride
                        let num_h = i as isize + pad - (ki * dilation) as isize;
                        let num_w = j as isize + pad - (kj * dilation) as isize;
                        if num_h >= 0
                            && num_h % stride as isize == 0
                            && num_w >= 0
                            && num_w % stride as isize == 0
                        {
                            let oh = num_h as usize / stride;
                            let ow = num_w as usize / stride;
                            if oh < out_h && ow < out_w {
                                sum += self.weight.data.get(&[oc, ic, ki, kj])
                                    * grad_output.get(&[b, oc, oh, ow]);
                            }
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
// MaxPool2d
// ---------------------------------------------------------------------------

/// 2D max pooling.
///
/// Input: `[batch, channels, height, width]`
/// Output: `[batch, channels, out_h, out_w]`
pub struct MaxPool2d<S: Scalar> {
    kernel_size: usize,
    stride: usize,
    padding: usize,
    cached_max_indices: Option<Vec<(usize, usize)>>, // (ih, iw) of max for each output
    _marker: core::marker::PhantomData<S>,
}

impl<S: Scalar> MaxPool2d<S> {
    pub fn new(kernel_size: usize) -> Self {
        Self::with_options(kernel_size, kernel_size, 0)
    }

    pub fn with_options(kernel_size: usize, stride: usize, padding: usize) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            cached_max_indices: None,
            _marker: core::marker::PhantomData,
        }
    }

    fn out_size(&self, input_size: usize) -> usize {
        (input_size + 2 * self.padding - self.kernel_size) / self.stride + 1
    }
}

impl<S: Scalar> Module<S> for MaxPool2d<S> {
    fn forward(&mut self, input: &Tensor<S>) -> Tensor<S> {
        assert_eq!(input.ndim(), 4);
        let batch = input.shape()[0];
        let channels = input.shape()[1];
        let height = input.shape()[2];
        let width = input.shape()[3];
        let out_h = self.out_size(height);
        let out_w = self.out_size(width);
        let pad = self.padding as isize;

        let total = batch * channels * out_h * out_w;
        let mut out_data = Vec::with_capacity(total);
        let mut indices = Vec::with_capacity(total);

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut max_val = S::from_f64(f64::NEG_INFINITY);
                        let mut max_ih = 0usize;
                        let mut max_iw = 0usize;
                        for ki in 0..self.kernel_size {
                            for kj in 0..self.kernel_size {
                                let ih = (oh * self.stride) as isize - pad + ki as isize;
                                let iw = (ow * self.stride) as isize - pad + kj as isize;
                                if ih >= 0 && (ih as usize) < height && iw >= 0 && (iw as usize) < width {
                                    let v = input.get(&[b, c, ih as usize, iw as usize]);
                                    if v > max_val {
                                        max_val = v;
                                        max_ih = ih as usize;
                                        max_iw = iw as usize;
                                    }
                                }
                            }
                        }
                        out_data.push(max_val);
                        indices.push((max_ih, max_iw));
                    }
                }
            }
        }

        self.cached_max_indices = Some(indices);
        Tensor::new(out_data, Shape::from_slice(&[batch, channels, out_h, out_w]))
    }

    fn backward(&mut self, grad_output: &Tensor<S>) -> Tensor<S> {
        let indices = self.cached_max_indices.as_ref().expect("must call forward before backward");
        let batch = grad_output.shape()[0];
        let channels = grad_output.shape()[1];
        let out_h = grad_output.shape()[2];
        let out_w = grad_output.shape()[3];

        // Reconstruct input spatial dims from output dims
        // input_size = (out_size - 1) * stride - 2 * padding + kernel_size
        let height = (out_h - 1) * self.stride + self.kernel_size - 2 * self.padding;
        let width = (out_w - 1) * self.stride + self.kernel_size - 2 * self.padding;

        let mut grad_input = alloc::vec![S::ZERO; batch * channels * height * width];

        let mut idx = 0;
        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let (ih, iw) = indices[idx];
                        grad_input[b * channels * height * width + c * height * width + ih * width + iw]
                            += grad_output.get(&[b, c, oh, ow]);
                        idx += 1;
                    }
                }
            }
        }

        Tensor::new(grad_input, Shape::from_slice(&[batch, channels, height, width]))
    }

    fn parameters(&self) -> Vec<&Parameter<S>> { Vec::new() }
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<S>> { Vec::new() }
}

// ---------------------------------------------------------------------------
// AvgPool2d
// ---------------------------------------------------------------------------

/// 2D average pooling.
///
/// Input: `[batch, channels, height, width]`
/// Output: `[batch, channels, out_h, out_w]`
pub struct AvgPool2d<S: Scalar> {
    kernel_size: usize,
    stride: usize,
    padding: usize,
    _marker: core::marker::PhantomData<S>,
}

impl<S: Scalar> AvgPool2d<S> {
    pub fn new(kernel_size: usize) -> Self {
        Self::with_options(kernel_size, kernel_size, 0)
    }

    pub fn with_options(kernel_size: usize, stride: usize, padding: usize) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            _marker: core::marker::PhantomData,
        }
    }

    fn out_size(&self, input_size: usize) -> usize {
        (input_size + 2 * self.padding - self.kernel_size) / self.stride + 1
    }
}

impl<S: Scalar> Module<S> for AvgPool2d<S> {
    fn forward(&mut self, input: &Tensor<S>) -> Tensor<S> {
        assert_eq!(input.ndim(), 4);
        let batch = input.shape()[0];
        let channels = input.shape()[1];
        let height = input.shape()[2];
        let width = input.shape()[3];
        let out_h = self.out_size(height);
        let out_w = self.out_size(width);
        let pad = self.padding as isize;
        let k2 = S::from_f64((self.kernel_size * self.kernel_size) as f64);

        Tensor::from_fn(
            Shape::from_slice(&[batch, channels, out_h, out_w]),
            |idx| {
                let (b, c, oh, ow) = (idx[0], idx[1], idx[2], idx[3]);
                let mut sum = S::ZERO;
                for ki in 0..self.kernel_size {
                    for kj in 0..self.kernel_size {
                        let ih = (oh * self.stride) as isize - pad + ki as isize;
                        let iw = (ow * self.stride) as isize - pad + kj as isize;
                        if ih >= 0 && (ih as usize) < height && iw >= 0 && (iw as usize) < width {
                            sum += input.get(&[b, c, ih as usize, iw as usize]);
                        }
                    }
                }
                sum / k2
            },
        )
    }

    fn backward(&mut self, grad_output: &Tensor<S>) -> Tensor<S> {
        let batch = grad_output.shape()[0];
        let channels = grad_output.shape()[1];
        let out_h = grad_output.shape()[2];
        let out_w = grad_output.shape()[3];
        let height = (out_h - 1) * self.stride + self.kernel_size - 2 * self.padding;
        let width = (out_w - 1) * self.stride + self.kernel_size - 2 * self.padding;
        let k2 = S::from_f64((self.kernel_size * self.kernel_size) as f64);
        let pad = self.padding as isize;

        Tensor::from_fn(
            Shape::from_slice(&[batch, channels, height, width]),
            |idx| {
                let (b, c, i, j) = (idx[0], idx[1], idx[2], idx[3]);
                let mut sum = S::ZERO;
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        // Check if (i,j) falls in this pooling window
                        let start_h = (oh * self.stride) as isize - pad;
                        let start_w = (ow * self.stride) as isize - pad;
                        let end_h = start_h + self.kernel_size as isize;
                        let end_w = start_w + self.kernel_size as isize;
                        if (i as isize) >= start_h
                            && (i as isize) < end_h
                            && (j as isize) >= start_w
                            && (j as isize) < end_w
                        {
                            sum += grad_output.get(&[b, c, oh, ow]) / k2;
                        }
                    }
                }
                sum
            },
        )
    }

    fn parameters(&self) -> Vec<&Parameter<S>> { Vec::new() }
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<S>> { Vec::new() }
}

// ---------------------------------------------------------------------------
// AdaptiveAvgPool2d
// ---------------------------------------------------------------------------

/// Adaptive average pooling — outputs a fixed spatial size regardless of input.
///
/// Input: `[batch, channels, H, W]`
/// Output: `[batch, channels, out_h, out_w]`
pub struct AdaptiveAvgPool2d<S: Scalar> {
    output_size: (usize, usize),
    cached_input_shape: Option<(usize, usize, usize, usize)>,
    _marker: core::marker::PhantomData<S>,
}

impl<S: Scalar> AdaptiveAvgPool2d<S> {
    pub fn new(output_h: usize, output_w: usize) -> Self {
        Self {
            output_size: (output_h, output_w),
            cached_input_shape: None,
            _marker: core::marker::PhantomData,
        }
    }

    /// Compute the start index for adaptive pooling bin.
    fn start_index(out_idx: usize, out_size: usize, in_size: usize) -> usize {
        (out_idx * in_size) / out_size
    }

    /// Compute the end index for adaptive pooling bin.
    fn end_index(out_idx: usize, out_size: usize, in_size: usize) -> usize {
        ((out_idx + 1) * in_size + out_size - 1) / out_size
    }
}

impl<S: Scalar> Module<S> for AdaptiveAvgPool2d<S> {
    fn forward(&mut self, input: &Tensor<S>) -> Tensor<S> {
        assert_eq!(input.ndim(), 4);
        let batch = input.shape()[0];
        let channels = input.shape()[1];
        let in_h = input.shape()[2];
        let in_w = input.shape()[3];
        let (out_h, out_w) = self.output_size;

        self.cached_input_shape = Some((batch, channels, in_h, in_w));

        Tensor::from_fn(
            Shape::from_slice(&[batch, channels, out_h, out_w]),
            |idx| {
                let (b, c, oh, ow) = (idx[0], idx[1], idx[2], idx[3]);
                let h_start = Self::start_index(oh, out_h, in_h);
                let h_end = Self::end_index(oh, out_h, in_h);
                let w_start = Self::start_index(ow, out_w, in_w);
                let w_end = Self::end_index(ow, out_w, in_w);
                let count = (h_end - h_start) * (w_end - w_start);
                let mut sum = S::ZERO;
                for ih in h_start..h_end {
                    for iw in w_start..w_end {
                        sum += input.get(&[b, c, ih, iw]);
                    }
                }
                sum / S::from_f64(count as f64)
            },
        )
    }

    fn backward(&mut self, grad_output: &Tensor<S>) -> Tensor<S> {
        let (batch, channels, in_h, in_w) =
            self.cached_input_shape.expect("must call forward before backward");
        let (out_h, out_w) = self.output_size;

        Tensor::from_fn(
            Shape::from_slice(&[batch, channels, in_h, in_w]),
            |idx| {
                let (b, c, i, j) = (idx[0], idx[1], idx[2], idx[3]);
                let mut sum = S::ZERO;
                for oh in 0..out_h {
                    let h_start = Self::start_index(oh, out_h, in_h);
                    let h_end = Self::end_index(oh, out_h, in_h);
                    if i < h_start || i >= h_end {
                        continue;
                    }
                    for ow in 0..out_w {
                        let w_start = Self::start_index(ow, out_w, in_w);
                        let w_end = Self::end_index(ow, out_w, in_w);
                        if j < w_start || j >= w_end {
                            continue;
                        }
                        let count = (h_end - h_start) * (w_end - w_start);
                        sum += grad_output.get(&[b, c, oh, ow]) / S::from_f64(count as f64);
                    }
                }
                sum
            },
        )
    }

    fn parameters(&self) -> Vec<&Parameter<S>> { Vec::new() }
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<S>> { Vec::new() }
}

// ---------------------------------------------------------------------------
// BatchNorm2d
// ---------------------------------------------------------------------------

/// Batch normalization for 2D inputs (CNNs).
///
/// Input: `[batch, channels, height, width]`
///
/// During training: normalizes using batch statistics, updates running stats.
/// During eval: normalizes using running statistics.
pub struct BatchNorm2d<S: Scalar> {
    pub gamma: Parameter<S>,  // [channels]
    pub beta: Parameter<S>,   // [channels]
    running_mean: Tensor<S>,  // [channels]
    running_var: Tensor<S>,   // [channels]
    momentum: f64,
    eps: f64,
    num_channels: usize,
    training: bool,
    cached_input: Option<Tensor<S>>,
    cached_mean: Option<Tensor<S>>,   // [channels]
    cached_var: Option<Tensor<S>>,    // [channels]
}

impl<S: Scalar> BatchNorm2d<S> {
    pub fn new(num_channels: usize) -> Self {
        Self {
            gamma: Parameter::new(Tensor::ones(Shape::from_slice(&[num_channels]))),
            beta: Parameter::new(Tensor::zeros(Shape::from_slice(&[num_channels]))),
            running_mean: Tensor::zeros(Shape::from_slice(&[num_channels])),
            running_var: Tensor::ones(Shape::from_slice(&[num_channels])),
            momentum: 0.1,
            eps: 1e-5,
            num_channels,
            training: true,
            cached_input: None,
            cached_mean: None,
            cached_var: None,
        }
    }
}

impl<S: Scalar> Module<S> for BatchNorm2d<S> {
    fn forward(&mut self, input: &Tensor<S>) -> Tensor<S> {
        assert_eq!(input.ndim(), 4, "BatchNorm2d input must be [batch, channels, H, W]");
        let batch = input.shape()[0];
        let channels = input.shape()[1];
        assert_eq!(channels, self.num_channels);
        let height = input.shape()[2];
        let width = input.shape()[3];
        let spatial = height * width;
        let n = S::from_f64((batch * spatial) as f64);
        let eps = S::from_f64(self.eps);

        if self.training {
            self.cached_input = Some(input.clone());

            // Compute per-channel mean and variance
            let mut mean_data = alloc::vec![S::ZERO; channels];
            let mut var_data = alloc::vec![S::ZERO; channels];

            for c in 0..channels {
                let mut sum = S::ZERO;
                for b in 0..batch {
                    for h in 0..height {
                        for w in 0..width {
                            sum += input.get(&[b, c, h, w]);
                        }
                    }
                }
                mean_data[c] = sum / n;

                let mut var_sum = S::ZERO;
                for b in 0..batch {
                    for h in 0..height {
                        for w in 0..width {
                            let diff = input.get(&[b, c, h, w]) - mean_data[c];
                            var_sum += diff * diff;
                        }
                    }
                }
                var_data[c] = var_sum / n;
            }

            let mean = Tensor::new(mean_data, Shape::from_slice(&[channels]));
            let var = Tensor::new(var_data, Shape::from_slice(&[channels]));

            // Update running stats
            let mom = S::from_f64(self.momentum);
            let one_minus = S::from_f64(1.0 - self.momentum);
            for c in 0..channels {
                self.running_mean.data_mut()[c] =
                    one_minus * self.running_mean.get(&[c]) + mom * mean.get(&[c]);
                self.running_var.data_mut()[c] =
                    one_minus * self.running_var.get(&[c]) + mom * var.get(&[c]);
            }

            self.cached_mean = Some(mean.clone());
            self.cached_var = Some(var.clone());

            // Normalize and scale
            Tensor::from_fn(input.shape().clone(), |idx| {
                let c = idx[1];
                let x_norm = (input.get(idx) - mean.get(&[c])) / (var.get(&[c]) + eps).sqrt();
                self.gamma.data.get(&[c]) * x_norm + self.beta.data.get(&[c])
            })
        } else {
            // Eval mode: use running stats
            Tensor::from_fn(input.shape().clone(), |idx| {
                let c = idx[1];
                let x_norm = (input.get(idx) - self.running_mean.get(&[c]))
                    / (self.running_var.get(&[c]) + eps).sqrt();
                self.gamma.data.get(&[c]) * x_norm + self.beta.data.get(&[c])
            })
        }
    }

    fn backward(&mut self, grad_output: &Tensor<S>) -> Tensor<S> {
        let input = self.cached_input.as_ref().expect("must call forward before backward");
        let mean = self.cached_mean.as_ref().expect("must call forward before backward");
        let var = self.cached_var.as_ref().expect("must call forward before backward");

        let batch = input.shape()[0];
        let channels = input.shape()[1];
        let height = input.shape()[2];
        let width = input.shape()[3];
        let spatial = height * width;
        let n = S::from_f64((batch * spatial) as f64);
        let eps = S::from_f64(self.eps);

        // Gradient w.r.t. gamma and beta
        let mut grad_gamma = alloc::vec![S::ZERO; channels];
        let mut grad_beta = alloc::vec![S::ZERO; channels];

        for c in 0..channels {
            let inv_std = S::from_f64(1.0) / (var.get(&[c]) + eps).sqrt();
            for b in 0..batch {
                for h in 0..height {
                    for w in 0..width {
                        let x_norm = (input.get(&[b, c, h, w]) - mean.get(&[c])) * inv_std;
                        grad_gamma[c] += grad_output.get(&[b, c, h, w]) * x_norm;
                        grad_beta[c] += grad_output.get(&[b, c, h, w]);
                    }
                }
            }
        }

        self.gamma.accumulate_grad(&Tensor::new(grad_gamma, Shape::from_slice(&[channels])));
        self.beta.accumulate_grad(&Tensor::new(grad_beta, Shape::from_slice(&[channels])));

        // Gradient w.r.t. input
        Tensor::from_fn(input.shape().clone(), |idx| {
            let c = idx[1];
            let inv_std = S::from_f64(1.0) / (var.get(&[c]) + eps).sqrt();
            let x_hat = (input.get(idx) - mean.get(&[c])) * inv_std;
            let g = self.gamma.data.get(&[c]);

            // Sum of grad_output * gamma and grad_output * gamma * x_hat for this channel
            let mut sum_dy = S::ZERO;
            let mut sum_dy_xhat = S::ZERO;
            for b in 0..batch {
                for h in 0..height {
                    for w in 0..width {
                        let dy = grad_output.get(&[b, c, h, w]);
                        let xh = (input.get(&[b, c, h, w]) - mean.get(&[c])) * inv_std;
                        sum_dy += dy;
                        sum_dy_xhat += dy * xh;
                    }
                }
            }

            g * inv_std * (grad_output.get(idx) - sum_dy / n - x_hat * sum_dy_xhat / n)
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
            (String::from("weight"), &self.gamma),
            (String::from("bias"), &self.beta),
        ]
    }

    fn named_parameters_mut(&mut self) -> Vec<(String, &mut Parameter<S>)> {
        alloc::vec![
            (String::from("weight"), &mut self.gamma),
            (String::from("bias"), &mut self.beta),
        ]
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
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
    #[allow(dead_code)]
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
