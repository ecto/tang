use alloc::vec::Vec;
use tang::Scalar;
use tang_tensor::{Shape, Tensor};

/// A dataset of input-target pairs.
pub trait Dataset<S: Scalar> {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn get(&self, index: usize) -> (Tensor<S>, Tensor<S>);
}

/// Simple dataset backed by two tensors.
/// Inputs: [N, ...], Targets: [N, ...] or [N]
pub struct TensorDataset<S: Scalar> {
    inputs: Tensor<S>,
    targets: Tensor<S>,
}

impl<S: Scalar> TensorDataset<S> {
    pub fn new(inputs: Tensor<S>, targets: Tensor<S>) -> Self {
        assert_eq!(
            inputs.shape()[0],
            targets.shape()[0],
            "input and target batch sizes must match"
        );
        Self { inputs, targets }
    }
}

impl<S: Scalar> Dataset<S> for TensorDataset<S> {
    fn len(&self) -> usize {
        self.inputs.shape()[0]
    }

    fn get(&self, index: usize) -> (Tensor<S>, Tensor<S>) {
        let input = slice_first_dim(&self.inputs, index);
        let target = slice_first_dim(&self.targets, index);
        (input, target)
    }
}

/// Extract a single element along the first dimension.
fn slice_first_dim<S: Scalar>(tensor: &Tensor<S>, index: usize) -> Tensor<S> {
    let shape = tensor.shape();
    if shape.ndim() == 1 {
        // [N] -> scalar tensor
        Tensor::scalar(tensor.get(&[index]))
    } else {
        // [N, d1, d2, ...] -> [d1, d2, ...]
        let inner_dims: Vec<usize> = shape.dims()[1..].to_vec();
        let inner_shape = Shape::new(inner_dims);
        Tensor::from_fn(inner_shape, |idx| {
            let mut full_idx = Vec::with_capacity(shape.ndim());
            full_idx.push(index);
            full_idx.extend_from_slice(idx);
            tensor.get(&full_idx)
        })
    }
}

/// Iterates over a dataset in batches.
pub struct DataLoader<'a, S: Scalar, D: Dataset<S>> {
    dataset: &'a D,
    batch_size: usize,
    indices: Vec<usize>,
    position: usize,
    _phantom: core::marker::PhantomData<S>,
}

impl<'a, S: Scalar, D: Dataset<S>> DataLoader<'a, S, D> {
    pub fn new(dataset: &'a D, batch_size: usize) -> Self {
        let n = dataset.len();
        Self {
            dataset,
            batch_size,
            indices: (0..n).collect(),
            position: 0,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Shuffle the dataset indices using LCG PRNG.
    pub fn shuffle(&mut self, seed: u64) {
        let mut state = seed;
        let n = self.indices.len();
        for i in (1..n).rev() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = (state >> 33) as usize % (i + 1);
            self.indices.swap(i, j);
        }
    }

    /// Reset the iterator to the beginning.
    pub fn reset(&mut self) {
        self.position = 0;
    }
}

impl<'a, S: Scalar, D: Dataset<S>> Iterator for DataLoader<'a, S, D> {
    type Item = (Tensor<S>, Tensor<S>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.indices.len() {
            return None;
        }
        let end = (self.position + self.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.position..end];
        self.position = end;

        let samples: Vec<(Tensor<S>, Tensor<S>)> =
            batch_indices.iter().map(|&i| self.dataset.get(i)).collect();

        let input_refs: Vec<&Tensor<S>> = samples.iter().map(|(i, _)| i).collect();
        let target_refs: Vec<&Tensor<S>> = samples.iter().map(|(_, t)| t).collect();

        let inputs = Tensor::stack(&input_refs);
        let targets = Tensor::stack(&target_refs);

        Some((inputs, targets))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn tensor_dataset_basics() {
        let inputs = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::from_slice(&[3, 2]),
        );
        let targets = Tensor::from_slice(&[0.0, 1.0, 0.0]);

        let ds = TensorDataset::new(inputs, targets);
        assert_eq!(ds.len(), 3);

        let (x, y) = ds.get(1);
        assert_eq!(x.data(), &[3.0, 4.0]);
        assert_eq!(y.data(), &[1.0]); // scalar
    }

    #[test]
    fn dataloader_batches() {
        let inputs = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            Shape::from_slice(&[4, 2]),
        );
        let targets = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0]);
        let ds = TensorDataset::new(inputs, targets);

        let mut loader = DataLoader::new(&ds, 2);
        let batches: Vec<_> = loader.by_ref().collect();
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].0.shape().dims(), &[2, 2]); // [batch=2, features=2]

        // Reset and iterate again
        loader.reset();
        let batches2: Vec<_> = loader.collect();
        assert_eq!(batches2.len(), 2);
    }
}
