//! Computational graph, parameter management, training loop.
//!
//! `tang-train` provides the building blocks for training neural networks on
//! the tang stack: the [`Module`] trait, composable [`layers`](Linear, Tanh, ReLU,
//! Sequential), [`loss`](cross_entropy_loss, mse_loss) functions with hand-written
//! gradients, a [`DataLoader`] that batches and shuffles, and a [`Trainer`] that
//! wires it all together.
//!
//! # Quick start
//!
//! ```ignore
//! use tang_train::*;
//! use tang_tensor::{Tensor, Shape};
//!
//! // 1. Build a model
//! let mut model = Sequential::<f64>::new(vec![
//!     Box::new(Linear::new(2, 8, 42)),
//!     Box::new(Tanh::new()),
//!     Box::new(Linear::new(8, 1, 137)),
//! ]);
//!
//! // 2. Prepare data
//! let ds = TensorDataset::new(inputs, targets);
//! let mut loader = DataLoader::new(&ds, 32);
//!
//! // 3. Train
//! let losses = Trainer::new(&mut model, ModuleAdam::new(0.001))
//!     .loss_fn(|p, t| (cross_entropy_loss(p, t), cross_entropy_loss_grad(p, t)))
//!     .epochs(100)
//!     .fit(&mut loader);
//! ```
//!
//! # Examples
//!
//! **The Quantum Poet** — a character-level text generator trained on physics
//! haikus. Demonstrates the full pipeline end-to-end:
//!
//! ```sh
//! cargo run --example quantum_poet -p tang-train
//! ```

#![no_std]

extern crate alloc;

pub mod data;
mod layers;
mod loss;
mod module;
mod optimizer;
mod parameter;
pub mod pinn;
mod rng;
mod trainer;

pub use data::{DataLoader, Dataset, TensorDataset};
pub use layers::{Dropout, Embedding, Linear, ReLU, Sequential, Tanh};
pub use loss::{
    cross_entropy_loss, cross_entropy_loss_grad, huber_loss, mse_loss, mse_loss_grad, softmax,
};
pub use module::Module;
pub use optimizer::{ModuleAdam, ModuleSgd, Optimizer};
pub use parameter::Parameter;
pub use rng::Rng;
pub use trainer::{LossFn, Trainer};
