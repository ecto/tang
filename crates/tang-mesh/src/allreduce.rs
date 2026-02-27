//! AllReduce: gradient aggregation across workers.
//!
//! Star-topology implementation first (correct and simple):
//! coordinator gathers all gradients, averages, broadcasts.
//! Ring allreduce is a future optimization behind the same interface.

use serde::{Deserialize, Serialize};

/// Reduction operation for allreduce.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReduceOp {
    /// Element-wise sum.
    Sum,
    /// Element-wise mean.
    Mean,
}

/// AllReduce: aggregates gradient vectors from multiple workers.
pub struct AllReduce {
    op: ReduceOp,
}

impl AllReduce {
    /// Create a new AllReduce with the given operation.
    pub fn new(op: ReduceOp) -> Self {
        Self { op }
    }

    /// Create a Mean allreduce (most common for gradient averaging).
    pub fn mean() -> Self {
        Self::new(ReduceOp::Mean)
    }

    /// Create a Sum allreduce.
    pub fn sum() -> Self {
        Self::new(ReduceOp::Sum)
    }

    /// Reduce multiple gradient vectors into one.
    ///
    /// All input vectors must have the same length.
    /// Star topology: all gradients collected at coordinator, reduced here.
    pub fn reduce(&self, gradients: &[Vec<f32>]) -> Vec<f32> {
        if gradients.is_empty() {
            return Vec::new();
        }

        let len = gradients[0].len();
        debug_assert!(gradients.iter().all(|g| g.len() == len));

        let n = gradients.len() as f32;
        let mut result = vec![0.0f32; len];

        for grad in gradients {
            for (r, g) in result.iter_mut().zip(grad.iter()) {
                *r += g;
            }
        }

        match self.op {
            ReduceOp::Sum => {}
            ReduceOp::Mean => {
                for r in &mut result {
                    *r /= n;
                }
            }
        }

        result
    }

    /// Reduce multiple named parameter gradient sets.
    ///
    /// Each worker contributes a `Vec<(name, gradient)>`. Returns averaged gradients.
    pub fn reduce_named(
        &self,
        worker_grads: &[Vec<(String, Vec<f32>)>],
    ) -> Vec<(String, Vec<f32>)> {
        if worker_grads.is_empty() {
            return Vec::new();
        }

        let n_params = worker_grads[0].len();
        let mut result = Vec::with_capacity(n_params);

        for param_idx in 0..n_params {
            let name = worker_grads[0][param_idx].0.clone();
            let grads: Vec<Vec<f32>> = worker_grads
                .iter()
                .map(|wg| wg[param_idx].1.clone())
                .collect();
            let reduced = self.reduce(&grads);
            result.push((name, reduced));
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_reduce() {
        let ar = AllReduce::mean();
        let grads = vec![
            vec![1.0, 2.0, 3.0],
            vec![3.0, 4.0, 5.0],
        ];
        let result = ar.reduce(&grads);
        assert_eq!(result, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn sum_reduce() {
        let ar = AllReduce::sum();
        let grads = vec![
            vec![1.0, 2.0, 3.0],
            vec![3.0, 4.0, 5.0],
        ];
        let result = ar.reduce(&grads);
        assert_eq!(result, vec![4.0, 6.0, 8.0]);
    }

    #[test]
    fn empty_reduce() {
        let ar = AllReduce::mean();
        let result = ar.reduce(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn single_worker_mean() {
        let ar = AllReduce::mean();
        let grads = vec![vec![1.0, 2.0, 3.0]];
        let result = ar.reduce(&grads);
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn named_reduce() {
        let ar = AllReduce::mean();
        let worker_grads = vec![
            vec![
                ("weight".into(), vec![1.0, 2.0]),
                ("bias".into(), vec![0.1]),
            ],
            vec![
                ("weight".into(), vec![3.0, 4.0]),
                ("bias".into(), vec![0.3]),
            ],
        ];
        let result = ar.reduce_named(&worker_grads);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, "weight");
        assert_eq!(result[0].1, vec![2.0, 3.0]);
        assert_eq!(result[1].0, "bias");
        assert_eq!(result[1].1, vec![0.2]);
    }
}
