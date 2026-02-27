# tang-hub

Download pretrained models from HuggingFace Hub and load them as tang tensors. One function call from repo ID to weights in memory.

## Quickstart

```rust
use tang_hub::{load_weights, load_state_dict, download};

// load all weights as a HashMap
let weights = load_weights("bert-base-uncased")?;

// or as sorted pairs (compatible with Module::load_state_dict)
let state = load_state_dict("bert-base-uncased")?;
model.load_state_dict(&state);

// just download files (no weight parsing)
let files = download("bert-base-uncased")?;
println!("config: {:?}", files.config);
println!("weights: {:?}", files.weight_files);
```

## Tokenizers

Enable the `tokenizers` feature for text encoding:

```rust
use tang_hub::{load_tokenizer, encode, decode, batch_encode};

let tok = load_tokenizer("bert-base-uncased")?;

let ids = encode(&tok, "hello world");          // Tensor<f64> of token IDs
let text = decode(&tok, &ids);                  // "hello world"
let batch = batch_encode(&tok, &texts, 128, 0); // [batch, seq_len] with padding
```

## What gets downloaded

Files are cached locally via `hf-hub`. Both single-file and sharded models are supported:

```
model.safetensors              ← single file
model.safetensors.index.json   ← shard index
model-00001-of-00003.safetensors
model-00002-of-00003.safetensors
model-00003-of-00003.safetensors
config.json                    ← optional
tokenizer.json                 ← optional
```

## API

| Function | Returns |
|----------|---------|
| `download(repo_id)` | `ModelFiles` (paths to cached files) |
| `load_weights(repo_id)` | `HashMap<String, Tensor<f64>>` |
| `load_state_dict(repo_id)` | `Vec<(String, Tensor<f64>)>` (sorted) |
| `load_tokenizer(repo_id)` | `tokenizers::Tokenizer` |
| `encode(tok, text)` | `Tensor<f64>` (1D) |
| `decode(tok, ids)` | `String` |
| `batch_encode(tok, texts, max_len, pad_id)` | `Tensor<f64>` (2D) |
| `vocab_size(tok)` | `usize` |

## Features

| Feature | Default | Purpose |
|---------|---------|---------|
| `tokenizers` | no | Text encoding/decoding via HuggingFace tokenizers |

## License

[MIT](../../LICENSE)
