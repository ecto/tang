# tang-safetensors

Load and save [safetensors](https://huggingface.co/docs/safetensors) — the HuggingFace weight format — as tang tensors.

## Usage

```rust
use tang_safetensors::{load, save, load_f32};
use std::path::Path;

// load pretrained weights as f64
let weights = load(Path::new("model.safetensors"))?;
for (name, tensor) in &weights {
    println!("{}: {:?}", name, tensor.shape());
}

// save your model
save(&weights, Path::new("checkpoint.safetensors"))?;

// load as f32 (for GPU pipelines)
let weights_f32 = load_f32(Path::new("model.safetensors"))?;
```

## With tang-train

```rust
use tang_safetensors::{load, save};
use tang_train::{Sequential, Linear};

let model = Sequential::new(vec![
    Box::new(Linear::new(768, 256)),
    Box::new(Linear::new(256, 10)),
]);

// save
save(&model.state_dict(), Path::new("model.safetensors"))?;

// load
let state = load(Path::new("model.safetensors"))?;
model.load_state_dict(&state.into_iter().collect());
```

## Supported dtypes

Reads F64, F32, F16, and BF16 from safetensors files. F16/BF16 are converted on load with correct IEEE 754 handling (subnormals, infinities, NaN).

Writes are always F64.

## API

| Function | Returns | Purpose |
|----------|---------|---------|
| `load(path)` | `HashMap<String, Tensor<f64>>` | Load as f64 |
| `load_f32(path)` | `HashMap<String, Tensor<f32>>` | Load as f32 |
| `save(tensors, path)` | `()` | Save f64 tensors |

Errors are reported as `SafetensorsError` with variants for IO, parsing, serialization, and unsupported dtypes.

## License

[MIT](../../LICENSE)
