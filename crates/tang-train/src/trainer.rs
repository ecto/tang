use crate::data::{DataLoader, Dataset};
use crate::scheduler::Scheduler;
use crate::{Module, Optimizer};
use alloc::boxed::Box;
use alloc::vec::Vec;
use tang_tensor::Tensor;

/// Loss function type: takes (prediction, target) and returns (scalar loss, gradient tensor).
pub type LossFn = fn(&Tensor<f64>, &Tensor<f64>) -> (f64, Tensor<f64>);

/// Training loop abstraction.
///
/// ```ignore
/// Trainer::new(&mut model, ModuleAdam::new(0.001))
///     .loss_fn(|pred, target| (mse_loss(pred, target), mse_loss_grad(pred, target)))
///     .epochs(100)
///     .fit(&mut data_loader);
/// ```
pub struct Trainer<'a, M, O>
where
    M: Module<f64>,
    O: Optimizer,
{
    model: &'a mut M,
    optimizer: O,
    loss_fn: LossFn,
    num_epochs: usize,
    scheduler: Option<Box<dyn Scheduler>>,
    accumulation_steps: usize,
}

fn default_loss(pred: &Tensor<f64>, target: &Tensor<f64>) -> (f64, Tensor<f64>) {
    (
        crate::mse_loss(pred, target),
        crate::mse_loss_grad(pred, target),
    )
}

impl<'a, M, O> Trainer<'a, M, O>
where
    M: Module<f64>,
    O: Optimizer,
{
    pub fn new(model: &'a mut M, optimizer: O) -> Self {
        Self {
            model,
            optimizer,
            loss_fn: default_loss,
            num_epochs: 100,
            scheduler: None,
            accumulation_steps: 1,
        }
    }

    pub fn loss_fn(mut self, f: LossFn) -> Self {
        self.loss_fn = f;
        self
    }

    pub fn epochs(mut self, n: usize) -> Self {
        self.num_epochs = n;
        self
    }

    /// Set gradient accumulation steps.
    ///
    /// When set to `n > 1`, gradients are accumulated over `n` mini-batches
    /// before calling `optimizer.step()`, effectively multiplying the batch
    /// size by `n` without increasing memory usage.
    pub fn accumulation_steps(mut self, n: usize) -> Self {
        assert!(n >= 1, "accumulation_steps must be >= 1");
        self.accumulation_steps = n;
        self
    }

    /// Set a learning rate scheduler. The scheduler's `lr(epoch)` is called
    /// at the start of each epoch to update the optimizer's learning rate.
    pub fn scheduler(mut self, s: impl Scheduler + 'static) -> Self {
        self.scheduler = Some(Box::new(s));
        self
    }

    /// Run the training loop. Returns per-epoch average loss.
    pub fn fit<D: Dataset<f64>>(
        &mut self,
        loader: &mut DataLoader<'_, f64, D>,
    ) -> Vec<f64> {
        let mut losses = Vec::with_capacity(self.num_epochs);

        for epoch in 0..self.num_epochs {
            if let Some(ref sched) = self.scheduler {
                self.optimizer.set_lr(sched.lr(epoch));
            }
            loader.reset();
            let mut epoch_loss = 0.0;
            let mut batch_count = 0;

            let mut micro_step = 0usize;
            self.model.zero_grad();

            for (inputs, targets) in loader.by_ref() {
                let pred = self.model.forward(&inputs);
                let (loss, grad) = (self.loss_fn)(&pred, &targets);
                self.model.backward(&grad);

                epoch_loss += loss;
                batch_count += 1;
                micro_step += 1;

                if micro_step >= self.accumulation_steps {
                    let mut params = self.model.parameters_mut();
                    self.optimizer.step(&mut params);
                    self.model.zero_grad();
                    micro_step = 0;
                }
            }

            // Flush any remaining accumulated gradients
            if micro_step > 0 {
                let mut params = self.model.parameters_mut();
                self.optimizer.step(&mut params);
                self.model.zero_grad();
            }

            if batch_count > 0 {
                losses.push(epoch_loss / batch_count as f64);
            }
        }

        losses
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::TensorDataset;
    use crate::{Linear, ModuleAdam, Sequential, Tanh, mse_loss, mse_loss_grad};
    use alloc::{boxed::Box, vec};
    use tang_tensor::Shape;

    #[test]
    fn trainer_basic() {
        let inputs = Tensor::new(
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            Shape::from_slice(&[4, 2]),
        );
        let targets = Tensor::new(vec![0.0, 1.0, 1.0, 0.0], Shape::from_slice(&[4, 1]));
        let ds = TensorDataset::new(inputs, targets);
        let mut loader = DataLoader::new(&ds, 4);

        let mut model = Sequential::<f64>::new(vec![
            Box::new(Linear::new(2, 8, 123)),
            Box::new(Tanh::new()),
            Box::new(Linear::new(8, 1, 456)),
        ]);

        let losses = Trainer::new(&mut model, ModuleAdam::new(0.01))
            .loss_fn(|p, t| (mse_loss(p, t), mse_loss_grad(p, t)))
            .epochs(200)
            .fit(&mut loader);

        assert_eq!(losses.len(), 200);
        assert!(
            losses.last().unwrap() < &0.1,
            "trainer should reduce loss, final={}",
            losses.last().unwrap()
        );
    }
}
