use tang_tensor::Tensor;

/// Mean Squared Error loss: (1/n) * sum((pred - target)^2)
pub fn mse_loss(pred: &Tensor<f64>, target: &Tensor<f64>) -> f64 {
    let diff = pred.sub(target);
    let sq = diff.mul(&diff);
    sq.mean()
}

/// Huber loss (smooth L1): quadratic for small errors, linear for large.
pub fn huber_loss(pred: &Tensor<f64>, target: &Tensor<f64>, delta: f64) -> f64 {
    let diff = pred.sub(target);
    let n = diff.numel() as f64;
    let mut total = 0.0;
    for &d in diff.data() {
        let a = d.abs();
        if a <= delta {
            total += 0.5 * d * d;
        } else {
            total += delta * (a - 0.5 * delta);
        }
    }
    total / n
}

/// Softmax: exp(x_i) / sum(exp(x_j)) for a 1-D tensor.
pub fn softmax(x: &Tensor<f64>) -> Tensor<f64> {
    let max_val = x.max();
    let shifted = x.map(|v| (v - max_val).exp());
    let sum = shifted.sum();
    shifted.scale(1.0 / sum)
}

/// Cross-entropy loss for classification.
/// `logits`: [batch, num_classes] raw scores
/// `targets`: [batch] integer class indices (stored as f64)
pub fn cross_entropy_loss(logits: &Tensor<f64>, targets: &Tensor<f64>) -> f64 {
    assert_eq!(logits.ndim(), 2);
    assert_eq!(targets.ndim(), 1);
    let batch = logits.shape()[0];
    let num_classes = logits.shape()[1];
    let mut total = 0.0;

    for b in 0..batch {
        // Compute log-softmax for numerical stability
        let mut max_val = logits.get(&[b, 0]);
        for c in 1..num_classes {
            let v = logits.get(&[b, c]);
            if v > max_val { max_val = v; }
        }
        let mut log_sum_exp = 0.0f64;
        for c in 0..num_classes {
            log_sum_exp += (logits.get(&[b, c]) - max_val).exp();
        }
        let log_sum_exp = max_val + log_sum_exp.ln();

        let target_class = targets.get(&[b]) as usize;
        let log_prob = logits.get(&[b, target_class]) - log_sum_exp;
        total -= log_prob;
    }
    total / batch as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use tang_tensor::Shape;

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
    fn huber_small_error() {
        let a = Tensor::from_slice(&[1.0]);
        let b = Tensor::from_slice(&[1.1]);
        let loss = huber_loss(&a, &b, 1.0);
        // |0.1| < 1.0, so quadratic: 0.5 * 0.01 = 0.005
        assert!((loss - 0.005).abs() < 1e-10);
    }
}
