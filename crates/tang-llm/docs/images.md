# Local image generation

`serve-images <pipeline-directory> --device metal --port 8913` exposes an
OpenAI-compatible `POST /v1/images/generations`. The directory contains diffusers
`text_encoder`, `tokenizer`, `transformer`, `vae` and `scheduler` subdirectories for
Z-Image-Turbo. Nothing downloads implicitly. The default listener is loopback;
`--host` and `--api-key-file` configure remote access using the existing CLI conventions.

The worker loads Qwen3 conditioning, the DiT and VAE on the first generation and keeps
them resident. It refuses a load that would exceed free memory plus scratch, using
tang's node memory probe, without evicting other models. `GET /v1/models` reports the
image model and residency. This dedicated image worker does not yet share a coding
model or a resident vision judge; node integration is a subsequent milestone.

Requests: `{model:"z-image-turbo",prompt,size?,n?,seed?,steps?,response_format?}`.
Sizes are multiples of 16 from 64 to 1024 per axis, count 1–4, steps 1–100 (default 8),
and output is `data[].b64_json` PNG plus actual seed, steps and dimensions. Seeds use
SplitMix64/Box-Muller noise, not PyTorch's RNG sequence. The same seed identifies tang's
stream only; cross-backend reproducibility requires matching noise and arithmetic.

`x-frog-progress: sse` (or `stream:true`) returns `progress` events `{step,of}` and a
terminal `result` event with the image response. Disconnecting cancels at the next
step boundary. Latent previews and cancellation inside VAE decode remain unfinished.

The DiT follows the basic single-image path: RMSNorm/SwiGLU blocks, scale/tanh-gate
modulation, interleaved axis RoPE, image/caption padding and refiners, and joint
attention. Weights are BF16 on GPU; norms remain F32. The VAE upcasts to F32 and uses
bounded tiled im2col/GEMM convolutions. Packing, affine broadcasts and upsampling
are intentionally unfused. Broad quality and performance evaluation remains unfinished.

## Trained-weight smoke test

The pinned converted checkpoint passed real Frog tool approval, all eight streamed steps,
PNG delivery, workspace asset saving and actual generation metadata on a 48 GiB Apple
Silicon Mac. A frog photograph at seed 42 produced a recognizable subject at 320×192
and 512×512; the native Frog transcript and lightbox were inspected. Repeating 320×192
with the resident pipeline produced a byte-identical PNG. First request: 52.3 seconds
including model loading; resident repeat: 11.1 seconds; 512×512: 52.2 seconds. These are
individual eight-step end-to-end measurements, not general benchmark or quality claims.
The chat/vision delivery assertions used a local fixture, not a production vision judge.

## Reference validation

Install torch, diffusers, transformers and safetensors in an isolated Python environment.
The scripts below create deterministic random checkpoints and fp32 boundary tensors;
they download no production model weights. DiT tests include 128-wide heads with the
production RoPE axis widths and 32x32 latents (256x256 pixel resolution). VAE tests
include 256x256 decoded output. These establish operator parity, not trained quality.

```sh
python crates/tang-llm/scripts/z_image_reference.py /tmp/z-image-ref --head128
cargo run -p tang-llm --example z_image_parity -- /tmp/z-image-ref
cargo run -p tang-llm --features metal --example z_image_parity -- /tmp/z-image-ref --metal
python crates/tang-llm/scripts/vae_reference.py /tmp/vae-ref
cargo run -p tang-llm --example z_image_parity -- /tmp/vae-ref
cargo run -p tang-llm --features metal --example z_image_parity -- /tmp/vae-ref --metal
```

Validated with torch 2.14.1 / diffusers 0.40.0: DiT boundary errors below 1.1e-5;
VAE boundary errors below 2e-5 on CPU and Metal. Trained-weight inference also passed
the smoke test above; full trained reference parity remains a separate validation task.

The reference exporters also accept `--checkpoint <local-directory>` and use symlinks
instead of copying weights. DiT export checks available RAM before loading its 24.6 GB
fp32 reference. `--small-only` limits the exported case; the runner's `--case case-8.json`
selects one case. Optional `--relative-tolerance` adds a relative term to the existing
absolute bound. The runner reports every failure, relative L2 error and final output.

Full converted trained DiT at 8×8 latents: final Metal output max error 4.07e-5, but
nine late boundaries exceed `atol=1e-3, rtol=1e-6` (worst max error 0.00647). This strict
boundary check remains failing; it is not covered by the random-checkpoint pass claim.
Accurate CPU RMS statistics reduced the worst CPU boundary error from 0.0427 to 0.0093;
forward/backward consistency and 97 affected regressions pass. Trained VAE at 4×4 latents
passes all 20 Metal boundaries under the existing 1e-3 absolute bound, final max error
2.8e-6. CPU final output is similarly close, but five hidden boundaries exceed its
stricter 1e-4 bound. Larger trained DiT reference cases are being investigated separately.

For constrained disk, `scripts/download_z_image.py <new-directory>` pins a public
Hugging Face revision, streams F32 matrices into BF16, preserves vectors, verifies
original LFS hashes, records converted hashes and retains 4 GiB disk headroom. It
writes about 20.5 GB instead of storing the roughly 32.8 GB original checkpoint.
Completed files can be resumed after size and SHA256 verification against the conversion
manifest; interrupted partial files are refused. The converted files are a distinct artifact; do not insert them into the official
repository's cache as if their hashes matched the originals.
