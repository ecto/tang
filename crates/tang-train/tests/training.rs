use tang_tensor::{Shape, Tensor};
use tang_train::{
    mse_loss, mse_loss_grad, Conv1d, Conv2d, Dropout, Embedding, Linear, Module, ModuleAdam,
    ModuleSgd, Optimizer, ReLU, Sequential, Tanh,
};

#[test]
fn test_linear_backward_gradient_check() {
    // Compare analytical backward against finite differences
    let mut layer = Linear::<f64>::new(3, 2, 42);
    let input = Tensor::from_slice(&[1.0, 2.0, 3.0]);
    let eps = 1e-5;

    // Forward + backward
    let _output = layer.forward(&input);
    let grad_output = Tensor::from_slice(&[1.0, 1.0]); // dL/dy = [1, 1]
    layer.backward(&grad_output);

    // Check weight gradients via finite differences
    let weight_grad = layer.weight.grad.clone().unwrap();
    for i in 0..2 {
        for j in 0..3 {
            let orig = layer.weight.data.get(&[i, j]);

            // f(w + eps)
            layer.weight.data.set(&[i, j], orig + eps);
            layer.weight.grad = None;
            let out_plus = layer.forward(&input);
            let loss_plus: f64 = out_plus.data().iter().sum();

            // f(w - eps)
            layer.weight.data.set(&[i, j], orig - eps);
            let out_minus = layer.forward(&input);
            let loss_minus: f64 = out_minus.data().iter().sum();

            // Restore
            layer.weight.data.set(&[i, j], orig);

            let numerical = (loss_plus - loss_minus) / (2.0 * eps);
            let analytical = weight_grad.get(&[i, j]);
            assert!(
                (numerical - analytical).abs() < 1e-4,
                "weight grad mismatch at [{}, {}]: numerical={}, analytical={}",
                i,
                j,
                numerical,
                analytical
            );
        }
    }

    // Check bias gradients
    let bias_grad = layer.bias.grad.clone().unwrap();
    for i in 0..2 {
        let orig = layer.bias.data.get(&[i]);

        layer.bias.data.set(&[i], orig + eps);
        let out_plus = layer.forward(&input);
        let loss_plus: f64 = out_plus.data().iter().sum();

        layer.bias.data.set(&[i], orig - eps);
        let out_minus = layer.forward(&input);
        let loss_minus: f64 = out_minus.data().iter().sum();

        layer.bias.data.set(&[i], orig);

        let numerical = (loss_plus - loss_minus) / (2.0 * eps);
        let analytical = bias_grad.get(&[i]);
        assert!(
            (numerical - analytical).abs() < 1e-4,
            "bias grad mismatch at [{}]: numerical={}, analytical={}",
            i,
            numerical,
            analytical
        );
    }
}

#[test]
fn test_relu_backward() {
    let mut relu = ReLU::<f64>::new();
    let input = Tensor::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let _output = relu.forward(&input);

    let grad_output = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0, 1.0]);
    let grad_input = relu.backward(&grad_output);

    // ReLU passes gradient through for positive inputs, zeros for negative
    assert_eq!(grad_input.data(), &[0.0, 0.0, 0.0, 1.0, 1.0]);
}

#[test]
fn test_tanh_backward() {
    let mut tanh_layer = Tanh::<f64>::new();
    let input = Tensor::from_slice(&[0.0, 1.0, -1.0]);
    let _output = tanh_layer.forward(&input);

    let grad_output = Tensor::from_slice(&[1.0, 1.0, 1.0]);
    let grad_input = tanh_layer.backward(&grad_output);

    // d/dx tanh(x) = 1 - tanh(x)^2
    let expected: Vec<f64> = [0.0, 1.0, -1.0]
        .iter()
        .map(|&x: &f64| {
            let t = x.tanh();
            1.0 - t * t
        })
        .collect();
    for (g, e) in grad_input.data().iter().zip(expected.iter()) {
        assert!(
            (g - e).abs() < 1e-10,
            "tanh grad mismatch: got {}, expected {}",
            g,
            e
        );
    }
}

#[test]
fn test_sequential_forward_backward() {
    // Small network: Linear(2, 4) -> ReLU -> Linear(4, 1)
    let mut model = Sequential::<f64>::new(vec![
        Box::new(Linear::new(2, 4, 42)),
        Box::new(ReLU::new()),
        Box::new(Linear::new(4, 1, 43)),
    ]);

    let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], Shape::from_slice(&[2, 2]));
    let target = Tensor::new(vec![1.0, 0.0], Shape::from_slice(&[2, 1]));

    let output = model.forward(&input);
    assert_eq!(output.shape().dims(), &[2, 1]);

    let grad = mse_loss_grad(&output, &target);
    let _grad_input = model.backward(&grad);

    // Verify all parameters have gradients
    for p in model.parameters() {
        assert!(p.grad.is_some(), "parameter should have gradient after backward");
    }
}

#[test]
fn test_training_converges_xor() {
    // XOR problem: 4 samples, 2 inputs, 1 output
    // Using tanh activation (better for XOR than ReLU)
    let inputs = Tensor::new(
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        Shape::from_slice(&[4, 2]),
    );
    let targets = Tensor::new(vec![0.0, 1.0, 1.0, 0.0], Shape::from_slice(&[4, 1]));

    let mut model = Sequential::<f64>::new(vec![
        Box::new(Linear::new(2, 8, 123)),
        Box::new(Tanh::new()),
        Box::new(Linear::new(8, 1, 456)),
    ]);
    let mut optimizer = ModuleAdam::new(0.01);

    let mut initial_loss = 0.0;
    let mut final_loss = 0.0;

    for epoch in 0..500 {
        model.zero_grad();

        let pred = model.forward(&inputs);
        let loss = mse_loss(&pred, &targets);
        let grad = mse_loss_grad(&pred, &targets);
        model.backward(&grad);

        let mut params = model.parameters_mut();
        optimizer.step(&mut params);

        if epoch == 0 {
            initial_loss = loss;
        }
        if epoch == 499 {
            final_loss = loss;
        }
    }

    assert!(
        initial_loss > 0.1,
        "initial loss should be significant, got {}",
        initial_loss
    );
    assert!(
        final_loss < 0.05,
        "final loss should be < 0.05 after 500 epochs, got {}",
        final_loss
    );
}

#[test]
fn test_module_sgd_converges() {
    // Simple regression: y = 2x + 1
    let inputs = Tensor::new(
        vec![0.0, 1.0, 2.0, 3.0],
        Shape::from_slice(&[4, 1]),
    );
    let targets = Tensor::new(vec![1.0, 3.0, 5.0, 7.0], Shape::from_slice(&[4, 1]));

    let mut model = Linear::<f64>::new(1, 1, 99);
    let mut optimizer = ModuleSgd::new(0.01);

    for _ in 0..1000 {
        model.zero_grad();
        let pred = model.forward(&inputs);
        let _loss = mse_loss(&pred, &targets);
        let grad = mse_loss_grad(&pred, &targets);
        model.backward(&grad);
        let mut params = model.parameters_mut();
        optimizer.step(&mut params);
    }

    let final_pred = model.forward(&inputs);
    let final_loss = mse_loss(&final_pred, &targets);
    assert!(
        final_loss < 0.01,
        "SGD should converge on simple regression, loss={}",
        final_loss
    );
}

#[test]
fn test_module_generic_dual() {
    use tang::Dual;

    // Verify Linear<Dual<f64>> compiles and works for forward pass
    let mut layer = Linear::<Dual<f64>>::new(2, 3, 42);
    let input = Tensor::from_slice(&[
        Dual::var(1.0_f64),
        Dual::constant(2.0_f64),
    ]);
    let output = layer.forward(&input);
    assert_eq!(output.numel(), 3);

    // Verify output carries dual parts (derivatives flow through)
    let has_nonzero_dual = output.data().iter().any(|d| d.dual.abs() > 1e-15);
    assert!(
        has_nonzero_dual,
        "Dual forward pass should propagate derivatives"
    );
}

#[test]
fn test_batch_vs_single_backward() {
    // Verify single-sample and batch backward give consistent results
    let mut layer = Linear::<f64>::new(3, 2, 42);

    // Single sample
    let input_1d = Tensor::from_slice(&[1.0, 2.0, 3.0]);
    let _out = layer.forward(&input_1d);
    let grad_out = Tensor::from_slice(&[1.0, 0.5]);
    layer.zero_grad();
    let _out = layer.forward(&input_1d);
    let _grad_in_single = layer.backward(&grad_out);
    let w_grad_single = layer.weight.grad.clone().unwrap();

    // Same as batch of 1
    layer.zero_grad();
    let input_2d = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from_slice(&[1, 3]));
    let _out = layer.forward(&input_2d);
    let grad_out_2d = Tensor::new(vec![1.0, 0.5], Shape::from_slice(&[1, 2]));
    let _grad_in_batch = layer.backward(&grad_out_2d);

    let w_grad_batch = layer.weight.grad.clone().unwrap();

    // Weight gradients should match
    for i in 0..w_grad_single.numel() {
        assert!(
            (w_grad_single.data()[i] - w_grad_batch.data()[i]).abs() < 1e-10,
            "weight grad mismatch at flat index {}",
            i
        );
    }
}

#[test]
fn test_embedding_forward() {
    let mut emb = Embedding::<f64>::new(5, 3, 42); // 5 tokens, 3-dim embeddings

    // Input: [2, 3] — batch=2, seq_len=3
    let input = Tensor::new(
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 0.0],
        Shape::from_slice(&[2, 3]),
    );
    let out = emb.forward(&input);

    // Output should be [2, 9] — 3 embeddings of dim 3, flattened
    assert_eq!(out.shape().dims(), &[2, 9]);

    // First element should be embedding[0] concatenated with embedding[1] and embedding[2]
    for e in 0..3 {
        assert!(
            (out.get(&[0, e]) - emb.weight.data.get(&[0, e])).abs() < 1e-10,
            "embedding lookup mismatch"
        );
    }
}

#[test]
fn test_embedding_backward() {
    let mut emb = Embedding::<f64>::new(4, 2, 42);

    // Single sample, seq_len=2
    let input = Tensor::new(vec![1.0, 3.0], Shape::from_slice(&[1, 2]));
    let _out = emb.forward(&input);

    // Gradient for output [1, 4]
    let grad = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], Shape::from_slice(&[1, 4]));
    emb.backward(&grad);

    let w_grad = emb.weight.grad.as_ref().unwrap();
    // Index 1 should get [1.0, 2.0], index 3 should get [3.0, 4.0]
    assert!((w_grad.get(&[1, 0]) - 1.0).abs() < 1e-10);
    assert!((w_grad.get(&[1, 1]) - 2.0).abs() < 1e-10);
    assert!((w_grad.get(&[3, 0]) - 3.0).abs() < 1e-10);
    assert!((w_grad.get(&[3, 1]) - 4.0).abs() < 1e-10);
    // Unused indices should have zero gradient
    assert!((w_grad.get(&[0, 0])).abs() < 1e-10);
    assert!((w_grad.get(&[2, 0])).abs() < 1e-10);
}

#[test]
fn test_dropout_training() {
    let mut dropout = Dropout::<f64>::new(0.5, 42);
    let input = Tensor::new(vec![1.0; 1000], Shape::from_slice(&[1, 1000]));
    let output = dropout.forward(&input);

    // Some elements should be zero, others scaled by 2.0
    let zeros = output.data().iter().filter(|&&v| v == 0.0).count();
    let scaled = output.data().iter().filter(|&&v| (v - 2.0).abs() < 1e-10).count();
    assert!(zeros > 300 && zeros < 700, "~50% should be zeroed, got {}", zeros);
    assert_eq!(zeros + scaled, 1000, "all elements should be 0 or 2.0");
}

#[test]
fn test_dropout_eval() {
    let mut dropout = Dropout::<f64>::new(0.5, 42);
    dropout.training = false;
    let input = Tensor::from_slice(&[1.0, 2.0, 3.0]);
    let output = dropout.forward(&input);

    // In eval mode, output should equal input
    for (&a, &b) in input.data().iter().zip(output.data().iter()) {
        assert!((a - b).abs() < 1e-10);
    }
}

#[test]
fn test_named_parameters_linear() {
    let layer = Linear::<f64>::new(3, 2, 42);
    let names: Vec<String> = layer.named_parameters().iter().map(|(n, _)| n.clone()).collect();
    assert_eq!(names, vec!["weight", "bias"]);
}

#[test]
fn test_named_parameters_sequential() {
    let model = Sequential::<f64>::new(vec![
        Box::new(Linear::new(4, 8, 42)),
        Box::new(Tanh::new()),
        Box::new(Linear::new(8, 2, 137)),
    ]);
    let names: Vec<String> = model.named_parameters().iter().map(|(n, _)| n.clone()).collect();
    assert_eq!(names, vec!["0.weight", "0.bias", "2.weight", "2.bias"]);
}

#[test]
fn test_set_training_propagates_through_sequential() {
    let mut model = Sequential::<f64>::new(vec![
        Box::new(Linear::new(4, 8, 42)),
        Box::new(Dropout::new(0.5, 99)),
        Box::new(Linear::new(8, 2, 137)),
        Box::new(Dropout::new(0.3, 77)),
    ]);

    // Default: training=true — dropout should zero some elements
    let input = Tensor::new(vec![1.0; 4], Shape::from_slice(&[1, 4]));
    let _out_train = model.forward(&input);

    // Switch to eval
    model.set_training(false);
    let out_eval1 = model.forward(&input);
    let out_eval2 = model.forward(&input);

    // Eval outputs must be deterministic (no dropout randomness)
    for (&a, &b) in out_eval1.data().iter().zip(out_eval2.data().iter()) {
        assert!(
            (a - b).abs() < 1e-10,
            "eval mode should be deterministic: {} vs {}",
            a,
            b,
        );
    }

    // Switch back to training — output should differ from eval (with high probability)
    model.set_training(true);
    let out_train2 = model.forward(&input);
    let differs = out_train2
        .data()
        .iter()
        .zip(out_eval1.data().iter())
        .any(|(&a, &b)| (a - b).abs() > 1e-10);
    assert!(
        differs,
        "training mode output should differ from eval mode (dropout active)",
    );
}

#[test]
fn test_dropout_identity_when_not_training() {
    let mut dropout = Dropout::<f64>::new(0.5, 42);
    dropout.set_training(false);

    let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Shape::from_slice(&[1, 5]));
    let output = dropout.forward(&input);

    // In eval mode, output == input exactly
    for (&a, &b) in input.data().iter().zip(output.data().iter()) {
        assert!(
            (a - b).abs() < 1e-10,
            "dropout eval should be identity: {} vs {}",
            a,
            b,
        );
    }
}

#[test]
fn test_state_dict_roundtrip() {
    let mut model = Sequential::<f64>::new(vec![
        Box::new(Linear::new(3, 5, 42)),
        Box::new(ReLU::new()),
        Box::new(Linear::new(5, 2, 99)),
    ]);

    // Save state dict
    let state = model.state_dict();
    assert_eq!(state.len(), 4); // 2 layers x (weight + bias)

    // Load into fresh model with different seeds
    let mut model2 = Sequential::<f64>::new(vec![
        Box::new(Linear::new(3, 5, 0)),
        Box::new(ReLU::new()),
        Box::new(Linear::new(5, 2, 0)),
    ]);
    model2.load_state_dict(&state);

    // Verify identical output
    let input = Tensor::from_slice(&[1.0, 2.0, 3.0]);
    let out1 = model.forward(&input);
    let out2 = model2.forward(&input);
    for (&a, &b) in out1.data().iter().zip(out2.data().iter()) {
        assert!((a - b).abs() < 1e-10, "state_dict roundtrip mismatch: {} vs {}", a, b);
    }
}

#[test]
fn test_conv1d_forward_shape() {
    let mut conv = Conv1d::<f64>::new(3, 4, 3, 42); // in=3, out=4, kernel=3
    let input = Tensor::new(vec![0.0; 2 * 3 * 10], Shape::from_slice(&[2, 3, 10]));
    let output = conv.forward(&input);
    assert_eq!(output.shape().dims(), &[2, 4, 8]); // out_len = 10 - 3 + 1 = 8
}

#[test]
fn test_conv2d_forward_shape() {
    let mut conv = Conv2d::<f64>::new(1, 4, 3, 42); // in=1, out=4, 3x3 kernel
    let input = Tensor::new(vec![0.0; 1 * 1 * 5 * 5], Shape::from_slice(&[1, 1, 5, 5]));
    let output = conv.forward(&input);
    assert_eq!(output.shape().dims(), &[1, 4, 3, 3]); // 5-3+1 = 3
}

#[test]
fn test_conv1d_backward_gradient_check() {
    // Small conv: 1 in_channel, 1 out_channel, kernel_size=2, input length=3
    let mut conv = Conv1d::<f64>::new(1, 1, 2, 42);
    let input = Tensor::new(vec![1.0, 2.0, 3.0], Shape::from_slice(&[1, 1, 3]));
    let eps = 1e-5;

    // Forward + backward with grad_output = all ones
    let _output = conv.forward(&input);
    // out_len = 3 - 2 + 1 = 2
    let grad_output = Tensor::new(vec![1.0, 1.0], Shape::from_slice(&[1, 1, 2]));
    conv.backward(&grad_output);

    // Check weight gradients numerically
    let weight_grad = conv.weight.grad.clone().unwrap();
    for oc in 0..1 {
        for ic in 0..1 {
            for k in 0..2 {
                let orig = conv.weight.data.get(&[oc, ic, k]);

                conv.weight.data.set(&[oc, ic, k], orig + eps);
                conv.weight.grad = None;
                let out_plus = conv.forward(&input);
                let loss_plus: f64 = out_plus.data().iter().sum();

                conv.weight.data.set(&[oc, ic, k], orig - eps);
                let out_minus = conv.forward(&input);
                let loss_minus: f64 = out_minus.data().iter().sum();

                conv.weight.data.set(&[oc, ic, k], orig);

                let numerical = (loss_plus - loss_minus) / (2.0 * eps);
                let analytical = weight_grad.get(&[oc, ic, k]);
                assert!(
                    (numerical - analytical).abs() < 1e-4,
                    "conv1d weight grad mismatch at [{},{},{}]: numerical={}, analytical={}",
                    oc, ic, k, numerical, analytical
                );
            }
        }
    }

    // Check bias gradients numerically
    let bias_grad = conv.bias.grad.clone().unwrap();
    for oc in 0..1 {
        let orig = conv.bias.data.get(&[oc]);

        conv.bias.data.set(&[oc], orig + eps);
        let out_plus = conv.forward(&input);
        let loss_plus: f64 = out_plus.data().iter().sum();

        conv.bias.data.set(&[oc], orig - eps);
        let out_minus = conv.forward(&input);
        let loss_minus: f64 = out_minus.data().iter().sum();

        conv.bias.data.set(&[oc], orig);

        let numerical = (loss_plus - loss_minus) / (2.0 * eps);
        let analytical = bias_grad.get(&[oc]);
        assert!(
            (numerical - analytical).abs() < 1e-4,
            "conv1d bias grad mismatch at [{}]: numerical={}, analytical={}",
            oc, numerical, analytical
        );
    }

    // Check input gradients numerically
    let _output = conv.forward(&input);
    conv.zero_grad();
    let grad_input = conv.backward(&grad_output);

    let mut input_mut = input.clone();
    for ic in 0..1 {
        for i in 0..3 {
            let orig = input_mut.get(&[0, ic, i]);

            input_mut.set(&[0, ic, i], orig + eps);
            let out_plus = conv.forward(&input_mut);
            let loss_plus: f64 = out_plus.data().iter().sum();

            input_mut.set(&[0, ic, i], orig - eps);
            let out_minus = conv.forward(&input_mut);
            let loss_minus: f64 = out_minus.data().iter().sum();

            input_mut.set(&[0, ic, i], orig);

            let numerical = (loss_plus - loss_minus) / (2.0 * eps);
            let analytical = grad_input.get(&[0, ic, i]);
            assert!(
                (numerical - analytical).abs() < 1e-4,
                "conv1d input grad mismatch at [0,{},{}]: numerical={}, analytical={}",
                ic, i, numerical, analytical
            );
        }
    }
}

#[test]
fn test_conv1d_named_parameters() {
    let conv = Conv1d::<f64>::new(2, 3, 3, 42);
    let names: Vec<String> = conv.named_parameters().iter().map(|(n, _)| n.clone()).collect();
    assert_eq!(names, vec!["weight", "bias"]);
}

#[test]
fn test_conv2d_named_parameters() {
    let conv = Conv2d::<f64>::new(1, 4, 3, 42);
    let names: Vec<String> = conv.named_parameters().iter().map(|(n, _)| n.clone()).collect();
    assert_eq!(names, vec!["weight", "bias"]);
}
