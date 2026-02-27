use crate::Shape;
use alloc::vec::Vec;
use tang::Scalar;
use tang_la::{DMat, DVec};

/// N-dimensional tensor with CPU storage.
#[derive(Debug, Clone)]
pub struct Tensor<S: Scalar> {
    data: Vec<S>,
    shape: Shape,
    strides: Vec<usize>,
}

impl<S: Scalar> Tensor<S> {
    /// Create a tensor from flat data and shape.
    pub fn new(data: Vec<S>, shape: Shape) -> Self {
        let strides = shape.contiguous_strides();
        debug_assert_eq!(data.len(), shape.numel());
        Self {
            data,
            shape,
            strides,
        }
    }

    /// Create a tensor filled with zeros.
    pub fn zeros(shape: Shape) -> Self {
        let n = shape.numel();
        Self::new(alloc::vec![S::from_f64(0.0); n], shape)
    }

    /// Create a tensor filled with ones.
    pub fn ones(shape: Shape) -> Self {
        let n = shape.numel();
        Self::new(alloc::vec![S::from_f64(1.0); n], shape)
    }

    /// Create a tensor from a closure.
    pub fn from_fn(shape: Shape, f: impl Fn(&[usize]) -> S) -> Self {
        let n = shape.numel();
        let strides = shape.contiguous_strides();
        let ndim = shape.ndim();
        let mut data = Vec::with_capacity(n);
        let mut idx = alloc::vec![0usize; ndim];

        for _ in 0..n {
            data.push(f(&idx));
            // Increment multi-index
            for d in (0..ndim).rev() {
                idx[d] += 1;
                if idx[d] < shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        Self {
            data,
            shape,
            strides,
        }
    }

    /// Scalar tensor.
    pub fn scalar(val: S) -> Self {
        Self::new(alloc::vec![val], Shape::scalar())
    }

    /// 1-D tensor from slice.
    pub fn from_slice(s: &[S]) -> Self {
        Self::new(s.to_vec(), Shape::from_slice(&[s.len()]))
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }
    pub fn data(&self) -> &[S] {
        &self.data
    }
    pub fn data_mut(&mut self) -> &mut [S] {
        &mut self.data
    }

    /// Whether strides match contiguous layout.
    pub fn is_contiguous(&self) -> bool {
        self.strides == self.shape.contiguous_strides()
    }

    /// Flat index from multi-index.
    fn flat_index(&self, idx: &[usize]) -> usize {
        debug_assert_eq!(idx.len(), self.ndim());
        idx.iter()
            .zip(self.strides.iter())
            .map(|(i, s)| i * s)
            .sum()
    }

    /// Get element by multi-index.
    pub fn get(&self, idx: &[usize]) -> S {
        self.data[self.flat_index(idx)]
    }

    /// Set element by multi-index.
    pub fn set(&mut self, idx: &[usize], val: S) {
        let fi = self.flat_index(idx);
        self.data[fi] = val;
    }

    /// Reshape to new shape (must have same numel). Returns a new contiguous tensor.
    pub fn reshape(&self, new_shape: Shape) -> Self {
        assert_eq!(
            self.shape.numel(),
            new_shape.numel(),
            "reshape: incompatible sizes"
        );
        let data = if self.is_contiguous() {
            self.data.clone()
        } else {
            self.to_contiguous_data()
        };
        Self::new(data, new_shape)
    }

    /// Transpose last two dimensions (for 2+ dim tensors).
    pub fn transpose(&self) -> Self {
        let nd = self.ndim();
        assert!(nd >= 2, "transpose requires at least 2 dimensions");
        let mut new_dims: Vec<usize> = self.shape.dims().to_vec();
        new_dims.swap(nd - 2, nd - 1);
        let new_shape = Shape::new(new_dims);

        Self::from_fn(new_shape, |idx| {
            let mut src_idx: Vec<usize> = idx.to_vec();
            src_idx.swap(nd - 2, nd - 1);
            self.get(&src_idx)
        })
    }

    /// Collect data in contiguous order.
    fn to_contiguous_data(&self) -> Vec<S> {
        let n = self.numel();
        let ndim = self.ndim();
        let mut result = Vec::with_capacity(n);
        let mut idx = alloc::vec![0usize; ndim];
        for _ in 0..n {
            result.push(self.get(&idx));
            for d in (0..ndim).rev() {
                idx[d] += 1;
                if idx[d] < self.shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        result
    }

    // --- Element-wise operations ---

    /// Apply element-wise unary operation.
    pub fn map(&self, f: impl Fn(S) -> S) -> Self {
        let data: Vec<S> = self.data.iter().map(|&v| f(v)).collect();
        Self::new(data, self.shape.clone())
    }

    /// Element-wise binary operation with broadcasting.
    pub fn zip_with(&self, other: &Self, f: impl Fn(S, S) -> S) -> Self {
        if self.shape == other.shape {
            // Fast path: same shape
            let data: Vec<S> = self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| f(a, b))
                .collect();
            return Self::new(data, self.shape.clone());
        }

        let out_shape = Shape::broadcast(&self.shape, &other.shape)
            .expect("zip_with: incompatible shapes for broadcasting");

        Self::from_fn(out_shape, |idx| {
            let a = self.broadcast_get(idx);
            let b = other.broadcast_get(idx);
            f(a, b)
        })
    }

    /// Get element with broadcasting (index may be larger than shape).
    fn broadcast_get(&self, idx: &[usize]) -> S {
        let nd = self.ndim();
        let offset = idx.len() - nd;
        let mut fi = 0;
        for d in 0..nd {
            let i = idx[d + offset];
            let dim = self.shape[d];
            let actual_i = if dim == 1 { 0 } else { i };
            fi += actual_i * self.strides[d];
        }
        self.data[fi]
    }

    // --- Arithmetic ---

    pub fn add(&self, other: &Self) -> Self {
        self.zip_with(other, |a, b| a + b)
    }

    pub fn sub(&self, other: &Self) -> Self {
        self.zip_with(other, |a, b| a - b)
    }

    pub fn mul(&self, other: &Self) -> Self {
        self.zip_with(other, |a, b| a * b)
    }

    pub fn div(&self, other: &Self) -> Self {
        self.zip_with(other, |a, b| a / b)
    }

    pub fn neg(&self) -> Self {
        self.map(|v| S::from_f64(0.0) - v)
    }

    pub fn scale(&self, s: S) -> Self {
        self.map(|v| v * s)
    }

    // --- Transcendentals ---

    pub fn exp(&self) -> Self {
        self.map(|v| v.exp())
    }
    pub fn ln(&self) -> Self {
        self.map(|v| v.ln())
    }
    pub fn tanh(&self) -> Self {
        self.map(|v| v.tanh())
    }
    pub fn abs(&self) -> Self {
        self.map(|v| v.abs())
    }
    pub fn sqrt(&self) -> Self {
        self.map(|v| v.sqrt())
    }

    /// ReLU activation.
    pub fn relu(&self) -> Self {
        let zero = S::from_f64(0.0);
        self.map(|v| if v > zero { v } else { zero })
    }

    // --- Reductions ---

    /// Sum all elements.
    pub fn sum(&self) -> S {
        self.data
            .iter()
            .copied()
            .fold(S::from_f64(0.0), |a, b| a + b)
    }

    /// Mean of all elements.
    pub fn mean(&self) -> S {
        self.sum() / S::from_f64(self.numel() as f64)
    }

    /// Max element.
    pub fn max(&self) -> S {
        let mut m = self.data[0];
        for &v in &self.data[1..] {
            if v > m {
                m = v;
            }
        }
        m
    }

    /// Sum along an axis, reducing that dimension.
    pub fn sum_axis(&self, axis: usize) -> Self {
        assert!(axis < self.ndim());
        let mut new_dims: Vec<usize> = self.shape.dims().to_vec();
        let axis_size = new_dims.remove(axis);
        if new_dims.is_empty() {
            return Self::scalar(self.sum());
        }
        let new_shape = Shape::new(new_dims);

        Self::from_fn(new_shape, |idx| {
            let mut sum = S::from_f64(0.0);
            let mut full_idx: Vec<usize> = Vec::with_capacity(self.ndim());
            for d in 0..self.ndim() {
                if d == axis {
                    full_idx.push(0); // placeholder
                } else if d < axis {
                    full_idx.push(idx[d]);
                } else {
                    full_idx.push(idx[d - 1]);
                }
            }
            for k in 0..axis_size {
                full_idx[axis] = k;
                sum = sum + self.get(&full_idx);
            }
            sum
        })
    }

    /// Mean along an axis.
    pub fn mean_axis(&self, axis: usize) -> Self {
        let n = S::from_f64(self.shape[axis] as f64);
        self.sum_axis(axis).scale(S::from_f64(1.0) / n)
    }

    // --- Matrix operations ---

    /// Matrix multiply for 2-D tensors. Uses tang-la under the hood.
    pub fn matmul(&self, other: &Self) -> Self {
        assert_eq!(self.ndim(), 2);
        assert_eq!(other.ndim(), 2);
        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];
        assert_eq!(other.shape[0], k);

        let a = DMat::from_fn(m, k, |i, j| self.get(&[i, j]));
        let b = DMat::from_fn(k, n, |i, j| other.get(&[i, j]));
        let c = a.mul_mat(&b);

        let out_shape = Shape::from_slice(&[m, n]);
        Self::from_fn(out_shape, |idx| c.get(idx[0], idx[1]))
    }

    // --- Stacking ---

    /// Stack tensors along a new leading dimension.
    /// All tensors must have the same shape. Result has shape [N, ...original_shape].
    pub fn stack(tensors: &[&Tensor<S>]) -> Self {
        assert!(!tensors.is_empty(), "stack: need at least one tensor");
        let inner_shape = tensors[0].shape();
        for t in &tensors[1..] {
            assert_eq!(
                t.shape(),
                inner_shape,
                "stack: all tensors must have same shape"
            );
        }

        let n = tensors.len();
        let mut new_dims = Vec::with_capacity(inner_shape.ndim() + 1);
        new_dims.push(n);
        new_dims.extend_from_slice(inner_shape.dims());
        let new_shape = Shape::new(new_dims);

        let mut data = Vec::with_capacity(new_shape.numel());
        for t in tensors {
            data.extend_from_slice(t.data());
        }

        Self::new(data, new_shape)
    }

    // --- Conversions ---

    /// Convert 1-D tensor to DVec.
    pub fn to_dvec(&self) -> DVec<S> {
        assert_eq!(self.ndim(), 1);
        DVec::from_slice(&self.data)
    }

    /// Convert DVec to 1-D tensor.
    pub fn from_dvec(v: &DVec<S>) -> Self {
        let n = v.len();
        let data: Vec<S> = (0..n).map(|i| v[i]).collect();
        Self::new(data, Shape::from_slice(&[n]))
    }

    /// Convert 2-D tensor to DMat.
    pub fn to_dmat(&self) -> DMat<S> {
        assert_eq!(self.ndim(), 2);
        let (m, n) = (self.shape[0], self.shape[1]);
        DMat::from_fn(m, n, |i, j| self.get(&[i, j]))
    }

    /// Convert DMat to 2-D tensor.
    pub fn from_dmat(m: &DMat<S>) -> Self {
        let (rows, cols) = (m.nrows(), m.ncols());
        let shape = Shape::from_slice(&[rows, cols]);
        Self::from_fn(shape, |idx| m.get(idx[0], idx[1]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_basics() {
        let t = Tensor::<f64>::zeros(Shape::from_slice(&[2, 3]));
        assert_eq!(t.numel(), 6);
        assert_eq!(t.ndim(), 2);
        assert_eq!(t.get(&[0, 0]), 0.0);
    }

    #[test]
    fn tensor_from_fn() {
        let t = Tensor::<f64>::from_fn(Shape::from_slice(&[2, 3]), |idx| {
            (idx[0] * 3 + idx[1]) as f64
        });
        assert_eq!(t.get(&[0, 0]), 0.0);
        assert_eq!(t.get(&[0, 2]), 2.0);
        assert_eq!(t.get(&[1, 1]), 4.0);
    }

    #[test]
    fn tensor_arithmetic() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        let b = Tensor::from_slice(&[4.0, 5.0, 6.0]);
        let c = a.add(&b);
        assert_eq!(c.data(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn tensor_broadcast() {
        // [2, 3] + [3] -> [2, 3]
        let a = Tensor::<f64>::from_fn(Shape::from_slice(&[2, 3]), |idx| {
            (idx[0] * 3 + idx[1]) as f64
        });
        let b = Tensor::from_slice(&[10.0, 20.0, 30.0]);
        let c = a.add(&b);
        assert_eq!(c.get(&[0, 0]), 10.0);
        assert_eq!(c.get(&[0, 2]), 32.0);
        assert_eq!(c.get(&[1, 0]), 13.0);
    }

    #[test]
    fn tensor_matmul() {
        // [[1, 2], [3, 4]] @ [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
        let a = Tensor::new(alloc::vec![1.0, 2.0, 3.0, 4.0], Shape::from_slice(&[2, 2]));
        let b = Tensor::new(alloc::vec![5.0, 6.0, 7.0, 8.0], Shape::from_slice(&[2, 2]));
        let c = a.matmul(&b);
        assert_eq!(c.get(&[0, 0]), 19.0);
        assert_eq!(c.get(&[0, 1]), 22.0);
        assert_eq!(c.get(&[1, 0]), 43.0);
        assert_eq!(c.get(&[1, 1]), 50.0);
    }

    #[test]
    fn tensor_reductions() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(t.sum(), 10.0);
        assert_eq!(t.mean(), 2.5);
        assert_eq!(t.max(), 4.0);
    }

    #[test]
    fn tensor_sum_axis() {
        let t = Tensor::<f64>::from_fn(Shape::from_slice(&[2, 3]), |idx| {
            (idx[0] * 3 + idx[1] + 1) as f64
        });
        // [[1, 2, 3], [4, 5, 6]]
        let s0 = t.sum_axis(0); // [5, 7, 9]
        assert_eq!(s0.shape().dims(), &[3]);
        assert_eq!(s0.get(&[0]), 5.0);
        assert_eq!(s0.get(&[1]), 7.0);
        assert_eq!(s0.get(&[2]), 9.0);

        let s1 = t.sum_axis(1); // [6, 15]
        assert_eq!(s1.shape().dims(), &[2]);
        assert_eq!(s1.get(&[0]), 6.0);
        assert_eq!(s1.get(&[1]), 15.0);
    }

    #[test]
    fn tensor_reshape() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let r = t.reshape(Shape::from_slice(&[2, 3]));
        assert_eq!(r.get(&[0, 0]), 1.0);
        assert_eq!(r.get(&[1, 2]), 6.0);
    }

    #[test]
    fn tensor_transpose() {
        let t = Tensor::<f64>::from_fn(Shape::from_slice(&[2, 3]), |idx| {
            (idx[0] * 3 + idx[1]) as f64
        });
        let tt = t.transpose();
        assert_eq!(tt.shape().dims(), &[3, 2]);
        assert_eq!(tt.get(&[0, 0]), 0.0);
        assert_eq!(tt.get(&[2, 1]), 5.0);
    }

    #[test]
    fn tensor_activations() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 1.0, 2.0]);
        let r = t.relu();
        assert_eq!(r.data(), &[0.0, 0.0, 1.0, 2.0]);

        let e = Tensor::from_slice(&[0.0_f64]).exp();
        assert!((e.get(&[0]) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn tensor_dvec_roundtrip() {
        let v = DVec::from_slice(&[1.0, 2.0, 3.0]);
        let t = Tensor::from_dvec(&v);
        let v2 = t.to_dvec();
        assert!((v2[0] - 1.0).abs() < 1e-15);
        assert!((v2[2] - 3.0).abs() < 1e-15);
    }
}
