# tang-gpu

GPU compute via wgpu. Write element-wise ops in Rust, get fused WGSL kernels with automatic backward passes. Train neural networks on Metal, Vulkan, or DX12.

## Fused kernels

Write math in Rust using `tang-expr`'s symbolic DSL. The expression gets traced, differentiated, simplified, and compiled to a single GPU kernel — no hand-written shaders:

```rust
use tang_gpu::{GpuDevice, fused_forward_backward, KernelCache};
use tang_expr::ExprId;

let device = GpuDevice::new_sync()?;
let mut cache = KernelCache::new(&device);

// define: f(a, b) = (a + b)²
let kernel = fused_forward_backward(2, |inputs| {
    let sum = inputs[0] + inputs[1];
    sum * sum
});

// one dispatch: forward + backward in a single kernel
```

## Training pipeline

Full neural network training without leaving Rust:

```rust
use tang_gpu::*;

let device = GpuDevice::new_sync()?;
let mut cache = KernelCache::new(&device);

// build model
let l1 = GpuLinear::new(&device, 2, 8);
let l2 = GpuLinear::new(&device, 8, 1);
let model = GpuSequential::new(vec![
    Box::new(l1),
    Box::new(GpuReLULayer::new()),
    Box::new(l2),
]);

// train
let mut optimizer = GpuAdam::new(&device, model.parameters(), 0.01);
let loader = GpuDataLoader::new(&device, &inputs, &targets, 4);

for epoch in 0..100 {
    for (x, y) in loader.iter() {
        let out = model.forward(&x, &mut cache);
        let (loss, grad) = gpu_mse_loss(&device, &mut cache, &out, &y);
        model.backward(&grad, &mut cache);
        optimizer.step(&mut cache);
        model.zero_grad();
    }
}
```

## Architecture

```
tang-expr (symbolic graph)
    │
    ▼
trace → diff → simplify → to_wgsl
    │                         │
    ▼                         ▼
  CPU eval              WGSL kernel
                            │
                            ▼
                     wgpu dispatch
                     ┌─────────────┐
                     │ Metal       │
                     │ Vulkan      │
                     │ DX12        │
                     └─────────────┘
```

## What's included

| Component | Purpose |
|-----------|---------|
| `GpuDevice` | wgpu device + queue wrapper |
| `GpuBuffer` / `GpuTensor` | GPU storage with shape metadata |
| `KernelCache` | JIT compilation cache, command batching |
| `GpuLinear` | Fully connected layer with cached activations |
| `GpuReLULayer` | ReLU with gradient support |
| `GpuSequential` | Layer composition |
| `GpuAdam` | Adam optimizer running entirely on GPU |
| `GpuLayerNorm` | Layer normalization |
| `GpuAttention` | Multi-head attention |
| `GpuTransformerBlock` | Attention + FFN with residuals |
| `GpuDataLoader` | Batched data loading |
| `GpuTrainer` | Training loop orchestrator |
| `gpu_mse_loss` | Loss + gradient on GPU (no readback) |
| `load_safetensors` / `save_safetensors` | HuggingFace weight format |

## Key design choices

- **Fused kernels** — element-wise chains compile to one dispatch, not N
- **Command batching** — `begin_batch()` / `flush()` to amortize submit overhead
- **Tiled matmul** — 16×16 tiles with workgroup shared memory
- **Training stays on GPU** — loss + gradient computed without CPU readback until epoch end
- **Interleaved inputs** — multi-input kernels pack data for coalesced access

## License

[MIT](../../LICENSE)
