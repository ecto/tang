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

/// Sequence cross-entropy loss with padding mask.
///
/// `logits`: `[seq_len, vocab_size]` raw scores per position.
/// `targets`: `[seq_len]` integer token IDs (stored as S).
/// `padding_id`: token ID to ignore (PAD token).
///
/// Returns the mean cross-entropy over non-padding positions.
pub fn sequence_cross_entropy<S: Scalar>(
    logits: &Tensor<S>,
    targets: &Tensor<S>,
    padding_id: u32,
) -> S {
    assert_eq!(logits.ndim(), 2, "logits must be [seq_len, vocab_size]");
    assert_eq!(targets.ndim(), 1, "targets must be [seq_len]");
    let seq_len = logits.shape()[0];
    let vocab_size = logits.shape()[1];
    assert_eq!(targets.shape()[0], seq_len);

    let pad = padding_id as f64;
    let mut total = S::ZERO;
    let mut count = 0usize;

    for t in 0..seq_len {
        let target_id = targets.get(&[t]).to_f64();
        if (target_id - pad).abs() < 0.5 {
            continue; // skip padding
        }

        // log-softmax for numerical stability
        let mut max_val = logits.get(&[t, 0]);
        for c in 1..vocab_size {
            let v = logits.get(&[t, c]);
            if v > max_val {
                max_val = v;
            }
        }
        let mut log_sum_exp = S::ZERO;
        for c in 0..vocab_size {
            log_sum_exp += (logits.get(&[t, c]) - max_val).exp();
        }
        let log_sum_exp = max_val + log_sum_exp.ln();

        let target_class = target_id as usize;
        let log_prob = logits.get(&[t, target_class]) - log_sum_exp;
        total -= log_prob;
        count += 1;
    }

    if count == 0 {
        return S::ZERO;
    }
    total / S::from_f64(count as f64)
}

/// Gradient of sequence cross-entropy loss w.r.t. logits, with padding mask.
///
/// Returns `[seq_len, vocab_size]` tensor. Gradient is zero at padding positions.
pub fn sequence_cross_entropy_grad<S: Scalar>(
    logits: &Tensor<S>,
    targets: &Tensor<S>,
    padding_id: u32,
) -> Tensor<S> {
    assert_eq!(logits.ndim(), 2);
    assert_eq!(targets.ndim(), 1);
    let seq_len = logits.shape()[0];
    let vocab_size = logits.shape()[1];

    let pad = padding_id as f64;

    // Count non-padding positions for normalization
    let mut count = 0usize;
    for t in 0..seq_len {
        let target_id = targets.get(&[t]).to_f64();
        if (target_id - pad).abs() >= 0.5 {
            count += 1;
        }
    }

    if count == 0 {
        return Tensor::zeros(logits.shape().clone());
    }
    let count_s = S::from_f64(count as f64);

    Tensor::from_fn(logits.shape().clone(), |idx| {
        let t = idx[0];
        let c = idx[1];

        let target_id = targets.get(&[t]).to_f64();
        if (target_id - pad).abs() < 0.5 {
            return S::ZERO; // padding position
        }

        // softmax for this position
        let mut max_val = logits.get(&[t, 0]);
        for k in 1..vocab_size {
            let v = logits.get(&[t, k]);
            if v > max_val {
                max_val = v;
            }
        }
        let mut sum_exp = S::ZERO;
        for k in 0..vocab_size {
            sum_exp += (logits.get(&[t, k]) - max_val).exp();
        }
        let softmax_c = (logits.get(&[t, c]) - max_val).exp() / sum_exp;

        let target_class = target_id as usize;
        let one_hot = if c == target_class { S::ONE } else { S::ZERO };
        (softmax_c - one_hot) / count_s
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

    // -----------------------------------------------------------------------
    // Sequence cross-entropy tests
    // -----------------------------------------------------------------------

    #[test]
    fn seq_ce_confident_prediction() {
        // 3 positions, vocab_size=4, no padding
        let logits = Tensor::new(
            alloc::vec![
                10.0, 0.0, 0.0, 0.0,   // pos 0: confident in class 0
                0.0, 10.0, 0.0, 0.0,   // pos 1: confident in class 1
                0.0, 0.0, 10.0, 0.0,   // pos 2: confident in class 2
            ],
            Shape::from_slice(&[3, 4]),
        );
        let targets = Tensor::from_slice(&[0.0, 1.0, 2.0]);
        let loss = sequence_cross_entropy(&logits, &targets, 99); // no padding
        assert!(loss < 0.01, "expected low loss, got {}", loss);
    }

    #[test]
    fn seq_ce_skips_padding() {
        let logits = Tensor::new(
            alloc::vec![
                1.0, 2.0, 3.0,   // pos 0
                1.0, 2.0, 3.0,   // pos 1 (will be padding)
            ],
            Shape::from_slice(&[2, 3]),
        );
        let targets = Tensor::from_slice(&[2.0, 99.0]); // pos 1 is PAD=99

        let loss_with_pad = sequence_cross_entropy(&logits, &targets, 99);

        // Should equal single-position CE for pos 0 targeting class 2
        let single_logits = Tensor::new(
            alloc::vec![1.0, 2.0, 3.0],
            Shape::from_slice(&[1, 3]),
        );
        let single_targets = Tensor::from_slice(&[2.0]);
        let loss_single = cross_entropy_loss(&single_logits, &single_targets);

        assert!(
            (loss_with_pad - loss_single).abs() < 1e-10,
            "padding mask mismatch: {} vs {}",
            loss_with_pad,
            loss_single
        );
    }

    #[test]
    fn seq_ce_all_padding_returns_zero() {
        let logits = Tensor::new(
            alloc::vec![1.0, 2.0, 1.0, 2.0],
            Shape::from_slice(&[2, 2]),
        );
        let targets = Tensor::from_slice(&[0.0, 0.0]);
        let loss = sequence_cross_entropy(&logits, &targets, 0);
        assert!((loss - 0.0).abs() < 1e-15);
    }

    #[test]
    fn seq_ce_grad_zero_at_padding() {
        let logits = Tensor::new(
            alloc::vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::from_slice(&[2, 3]),
        );
        let targets = Tensor::from_slice(&[1.0, 0.0]); // pos 1 is PAD=0
        let grad = sequence_cross_entropy_grad(&logits, &targets, 0);

        // Row 1 (padding) should have all-zero gradient
        for c in 0..3 {
            assert!(
                grad.get(&[1, c]).abs() < 1e-15,
                "padding row grad[1][{}] = {} (expected 0)",
                c,
                grad.get(&[1, c])
            );
        }

        // Row 0 should have non-zero gradient and sum to ~0
        let row_sum: f64 = (0..3).map(|c| grad.get(&[0, c])).sum();
        assert!(
            row_sum.abs() < 1e-10,
            "non-padding row sum = {} (expected ~0)",
            row_sum
        );
    }

    #[test]
    fn seq_ce_grad_numerical() {
        let logits = Tensor::new(
            alloc::vec![1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 0.5, 0.5, 0.5],
            Shape::from_slice(&[3, 3]),
        );
        let targets = Tensor::from_slice(&[2.0, 0.0, 1.0]);
        let padding_id = 99u32; // no padding
        let eps = 1e-5;

        let analytic = sequence_cross_entropy_grad(&logits, &targets, padding_id);

        for t in 0..3 {
            for c in 0..3 {
                let mut plus = logits.clone();
                let mut minus = logits.clone();
                plus.set(&[t, c], plus.get(&[t, c]) + eps);
                minus.set(&[t, c], minus.get(&[t, c]) - eps);

                let loss_plus = sequence_cross_entropy(&plus, &targets, padding_id);
                let loss_minus = sequence_cross_entropy(&minus, &targets, padding_id);
                let numerical = (loss_plus - loss_minus) / (2.0 * eps);

                assert!(
                    (numerical - analytic.get(&[t, c])).abs() < 1e-5,
                    "grad mismatch at [{},{}]: numerical={}, analytic={}",
                    t, c, numerical, analytic.get(&[t, c])
                );
            }
        }
    }
}
