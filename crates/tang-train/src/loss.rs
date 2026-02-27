use tang::Scalar;
use tang_tensor::Tensor;

#[cfg(test)]
use tang_tensor::Shape;

/// Mean Squared Error loss: (1/n) * sum((pred - target)^2)
pub fn mse_loss<S: Scalar>(pred: &Tensor<S>, target: &Tensor<S>) -> S {
    let diff = pred.sub(target);
    let sq = diff.mul(&diff);
    sq.mean()
}

/// Gradient of MSE loss w.r.t. predictions: (2/n) * (pred - target)
pub fn mse_loss_grad<S: Scalar>(pred: &Tensor<S>, target: &Tensor<S>) -> Tensor<S> {
    let n = S::from_f64(pred.numel() as f64);
    let two = S::from_f64(2.0);
    pred.sub(target).scale(two / n)
}

/// Huber loss (smooth L1): quadratic for small errors, linear for large.
pub fn huber_loss<S: Scalar>(pred: &Tensor<S>, target: &Tensor<S>, delta: S) -> S {
    let diff = pred.sub(target);
    let n = S::from_f64(diff.numel() as f64);
    let half = S::from_f64(0.5);
    let mut total = S::from_f64(0.0);
    for &d in diff.data() {
        let a = d.abs();
        if a <= delta {
            total += half * d * d;
        } else {
            total += delta * (a - half * delta);
        }
    }
    total / n
}

/// Softmax: exp(x_i) / sum(exp(x_j)) for a 1-D tensor.
pub fn softmax<S: Scalar>(x: &Tensor<S>) -> Tensor<S> {
    let max_val = x.max();
    let shifted = x.map(|v| (v - max_val).exp());
    let sum = shifted.sum();
    shifted.scale(S::from_f64(1.0) / sum)
}

/// Cross-entropy loss for classification.
/// `logits`: [batch, num_classes] raw scores
/// `targets`: [batch] integer class indices (stored as S, converted via to_f64)
pub fn cross_entropy_loss<S: Scalar>(logits: &Tensor<S>, targets: &Tensor<S>) -> S {
    assert_eq!(logits.ndim(), 2);
    assert_eq!(targets.ndim(), 1);
    let batch = logits.shape()[0];
    let num_classes = logits.shape()[1];
    let mut total = S::from_f64(0.0);

    for b in 0..batch {
        // Compute log-softmax for numerical stability
        let mut max_val = logits.get(&[b, 0]);
        for c in 1..num_classes {
            let v = logits.get(&[b, c]);
            if v > max_val {
                max_val = v;
            }
        }
        let mut log_sum_exp = S::from_f64(0.0);
        for c in 0..num_classes {
            log_sum_exp += (logits.get(&[b, c]) - max_val).exp();
        }
        let log_sum_exp = max_val + log_sum_exp.ln();

        let target_class = targets.get(&[b]).to_f64() as usize;
        let log_prob = logits.get(&[b, target_class]) - log_sum_exp;
        total -= log_prob;
    }
    total / S::from_f64(batch as f64)
}

/// Gradient of cross-entropy loss w.r.t. logits.
/// Returns [batch, num_classes] tensor: (softmax(logits) - one_hot(targets)) / batch
pub fn cross_entropy_loss_grad<S: Scalar>(logits: &Tensor<S>, targets: &Tensor<S>) -> Tensor<S> {
    assert_eq!(logits.ndim(), 2);
    assert_eq!(targets.ndim(), 1);
    let batch = logits.shape()[0];
    let num_classes = logits.shape()[1];
    let batch_s = S::from_f64(batch as f64);

    Tensor::from_fn(logits.shape().clone(), |idx| {
        let b = idx[0];
        let c = idx[1];
        // Compute softmax for row b
        let mut max_val = logits.get(&[b, 0]);
        for k in 1..num_classes {
            let v = logits.get(&[b, k]);
            if v > max_val {
                max_val = v;
            }
        }
        let mut sum_exp = S::from_f64(0.0);
        for k in 0..num_classes {
            sum_exp += (logits.get(&[b, k]) - max_val).exp();
        }
        let softmax_c = (logits.get(&[b, c]) - max_val).exp() / sum_exp;
        let target_class = targets.get(&[b]).to_f64() as usize;
        let one_hot = if c == target_class {
            S::from_f64(1.0)
        } else {
            S::from_f64(0.0)
        };
        (softmax_c - one_hot) / batch_s
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mse_zero() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        assert!(mse_loss(&a, &a) < 1e-15);
    }

    #[test]
    fn mse_nonzero() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        let b = Tensor::from_slice(&[2.0, 3.0, 4.0]);
        assert!((mse_loss(&a, &b) - 1.0).abs() < 1e-15); // (1+1+1)/3 = 1
    }

    #[test]
    fn mse_grad_check() {
        let pred = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        let target = Tensor::from_slice(&[1.5, 2.5, 3.5]);
        let grad = mse_loss_grad(&pred, &target);
        // grad = 2/3 * (pred - target) = 2/3 * [-0.5, -0.5, -0.5]
        let expected = [-1.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0];
        for (g, e) in grad.data().iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-10);
        }
    }

    #[test]
    fn softmax_sums_to_one() {
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        let s = softmax(&x);
        assert!((s.sum() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cross_entropy_correct_class() {
        // When prediction is very confident in the correct class, loss should be low
        let logits = Tensor::new(
            alloc::vec![10.0, 0.0, 0.0, 0.0, 10.0, 0.0],
            Shape::from_slice(&[2, 3]),
        );
        let targets = Tensor::from_slice(&[0.0, 1.0]);
        let loss = cross_entropy_loss(&logits, &targets);
        assert!(loss < 0.01, "loss should be near zero, got {}", loss);
    }

    #[test]
    fn cross_entropy_grad_sums_to_zero() {
        // For each sample, softmax sums to 1 and one_hot sums to 1, so grad row sums to 0
        let logits = Tensor::new(
            alloc::vec![1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
            Shape::from_slice(&[2, 3]),
        );
        let targets = Tensor::from_slice(&[0.0, 2.0]);
        let grad = cross_entropy_loss_grad(&logits, &targets);
        // Each row should sum to approximately 0
        for b in 0..2 {
            let row_sum: f64 = (0..3).map(|c| grad.get(&[b, c])).sum();
            assert!(
                row_sum.abs() < 1e-10,
                "row {} sum should be ~0, got {}",
                b,
                row_sum
            );
        }
    }

    #[test]
    fn huber_small_error() {
        let a = Tensor::from_slice(&[1.0]);
        let b = Tensor::from_slice(&[1.1]);
        let loss = huber_loss(&a, &b, 1.0);
        // |0.1| < 1.0, so quadratic: 0.5 * 0.01 = 0.005
        assert!((loss - 0.005).abs() < 1e-10);
    }
}
