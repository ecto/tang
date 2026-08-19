#[test]
fn ctc_works_in_f32() {
    use tang_tensor::{Shape, Tensor};
    use tang_train::{ctc_loss, ctc_loss_grad, greedy_decode};
    let data: Vec<f32> = vec![0.5, 0.1, 0.2, 0.3, 0.7, 0.4, 0.2, 0.1, 0.8];
    let logits = Tensor::new(data, Shape::from_slice(&[3, 3]));
    let loss = ctc_loss(&logits, &[1, 2], 0).unwrap();
    assert!(loss.is_finite() && loss > 0.0, "f32 loss = {loss}");
    let grad = ctc_loss_grad(&logits, &[1, 2], 0).unwrap();
    assert!(grad.data().iter().all(|g| g.is_finite()));
    let _ = greedy_decode(&logits, 0);
}
