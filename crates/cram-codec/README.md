# cram-codec

Error-bounded lossy compression of scientific `f32`/`f64` arrays, in pure
Rust. Every byte of the stream counts toward the reported rate, encoding is
deterministic, and the error bound you ask for is the one you get.

Codecs (all decode through `cram_codec::decode_any`):

- **QTT** (`codec`) — bit-accounted quantized tensor train: TT-SVD,
  error-feedback quantization, rate–distortion rank choice, static rANS.
- **Wavelet** (`wavcodec`) — CDF 9/7 integer lifting with a context-modelled
  coefficient coder; the strongest codec on most smooth fields.
- **Tucker** (`tucker`) and **interpolation** (`interp`) codecs.
- **tc-gpu / TCG3** (`gpuwav`, `gpuwav3`) — brick-parallel wavelet streams
  laid out for GPU decoding, optionally with a learned coefficient model.

Part of [tang](https://github.com/ecto/tang).

## Usage

```rust
use cram_codec::{decode, encode_f32, Config, Decoded, ErrorBound};

// a 64×64 field
let x: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.01).sin()).collect();

// pointwise |x − x̂| ≤ 1e-3
let (bytes, stats) = encode_f32(&x, &[64, 64], &Config::new(ErrorBound::MaxAbs(1e-3)));
println!("{} bytes, {:.3} bits/value", stats.bytes, stats.bits_per_value);

let Decoded::F32 { data, .. } = decode(&bytes).unwrap() else { unreachable!() };
assert!(x.iter().zip(&data).all(|(a, b)| (a - b).abs() <= 1e-3));
```

Error bounds: `ErrorBound::MaxAbs(e)` (pointwise), `ErrorBound::Rmse(r)`
(absolute RMSE), `ErrorBound::Psnr(db)` (range-normalised PSNR).

Examples (`cargo run --release -p cram-codec --example <name>`):
`roundtrip` (the snippet above, all bounds), `tc` / `qtt` (encode and decode
raw little-endian `.f32` files from the command line).

## Features

- `default = []` — CPU encoders and decoders; dependencies are `tang-la`,
  `tang-optim`, `tang-expr` and `faer` only.
- `gpu` — `gpudec::GpuDecoder`, a wgpu (WebGPU / Vulkan / Metal / DX12)
  decoder for tc-gpu and TCG3 streams. Its output is bit-identical to the
  CPU decoder: the kernels are integer only. `GpuDecoder::new` blocks on
  native; `GpuDecoder::new_async` is the constructor on wasm32.

## WebAssembly

The decoders build for `wasm32-unknown-unknown`. The browser front end
(wasm-bindgen, WebGPU with a CPU fallback) is the `cram-codec-wasm` crate in
the [tang repository](https://github.com/ecto/tang/tree/main/crates/cram-codec-wasm);
it is not published to crates.io.

TCG3 streams can carry their own decoder as a checked `tang-expr` integer
program; see [`SHIPPED_DECODER.md`](SHIPPED_DECODER.md).

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option. Contributions are accepted under the same terms with a DCO sign-off (see CONTRIBUTING.md).
