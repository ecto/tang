# tang-train

Neural networks on the tang stack. Layers, loss functions, optimizers, schedulers, and a training loop — from embedding to transformer, all `no_std`.

## Define a model

```rust
use tang_train::*;
use tang_tensor::Tensor;

let model = Sequential::new(vec![
    Box::new(Linear::new(784, 128)),
    Box::new(ReLU),
    Box::new(Dropout::new(0.1)),
    Box::new(Linear::new(128, 10)),
]);
```

## Train it

```rust
let mut trainer = Trainer::new(model)
    .optimizer(ModuleAdam::new(0.001))
    .scheduler(WarmupCosine::new(0.001, 10, 100))
    .loss(cross_entropy_loss_grad)
    .build();

for epoch in 0..100 {
    let loader = DataLoader::new(&dataset, 32, true);
    for (input, target) in loader {
        trainer.step(&input, &target);
    }
    trainer.end_epoch();
}
```

## Layers

| Layer | Parameters | Notes |
|-------|-----------|-------|
| `Linear` | W, b | Xavier/Glorot init |
| `ReLU` | — | |
| `Tanh` | — | |
| `Dropout` | — | Inverted; identity in eval mode |
| `Embedding` | lookup table | Integer indices → dense vectors |
| `Conv1d` | kernel, bias | stride=1, no padding |
| `Conv2d` | kernel, bias | stride=1, no padding |
| `LayerNorm` | γ, β | Learnable affine |
| `MultiHeadAttention` | Q, K, V, Out | Scaled dot-product |
| `TransformerBlock` | attention + FFN | Pre-norm with residuals |
| `Sequential` | — | Chains layers |

## Loss functions

```rust
// with hand-written gradients for efficiency
let (loss, grad) = mse_loss_grad(&prediction, &target);
let (loss, grad) = cross_entropy_loss_grad(&logits, &target);
let loss = huber_loss(&prediction, &target, 1.0);
```

## Save / load

```rust
// checkpoint
let state = model.state_dict();

// restore
model.load_state_dict(&state);
```

Works with `tang-safetensors` for HuggingFace-compatible persistence.

## Physics-informed neural networks

The `pinn` module provides AD-powered utilities for PDE constraints:

```rust
use tang_train::pinn;

// automatic derivatives through your network
let du_dx = pinn::grad(|x| network.forward_scalar(x), &point);
let laplacian = pinn::laplacian(|x| network.forward_scalar(x), &point);

// PDE residual at collocation points
let residual = pinn::pde_residual(
    |x| network.forward_scalar(x),
    &collocation_points,
    |u, du, d2u, x| d2u - source_term(x), // Poisson equation
);
```

## Design

- **`#![no_std]`** with `alloc`
- `Module` trait: `forward`, `backward`, `parameters`, `state_dict`, `set_training`
- Hand-written backward passes (no runtime tape overhead)
- Activation caching during forward for backward reuse
- LCG-based deterministic PRNG for reproducible shuffling and dropout

## License

[MIT](../../LICENSE)
