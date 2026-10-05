# Local image generation

`serve-images <pipeline-directory> --device metal --port 8913` exposes an
OpenAI-compatible `POST /v1/images/generations`. The directory contains diffusers
`text_encoder`, `tokenizer`, `transformer`, `vae` and `scheduler` subdirectories for
Z-Image-Turbo. Nothing downloads implicitly. The default listener is loopback;
`--host` and `--api-key-file` configure remote access using the existing CLI conventions.

The worker loads Qwen3 conditioning, the DiT and VAE on the first generation and keeps
them resident. It refuses a load that would exceed free memory plus scratch, using
tang's node memory probe, without evicting other models. `GET /v1/models` reports the
image model and residency. `serve <chat-model> --image-pipeline <pipeline-directory>`
mounts the same image worker alongside chat/vision, under the same API key. Both models
remain resident, while a shared resource lock serializes heavy inference and model loads.
The image queue accepts eight waiting jobs and returns HTTP 429 when full. `/node`
reports image residency alongside the chat model. Image and chat queues remain separate;
cross-queue priority scheduling and explicit image unloading are later work.

Requests: `{model:"z-image-turbo",prompt,size?,n?,seed?,steps?,response_format?}`.
Sizes are multiples of 16 from 64 to 1024 per axis, count 1–4, steps 1–100 (default 8),
and output is `data[].b64_json` PNG plus actual seed, steps and dimensions. The top-level
`model_hash` fingerprints actual checkpoint weights, architecture, tokenizer and scheduler
bytes once per resident load; it is independent of the requested model name or download
revision. Apple Silicon uses hardware-assisted SHA256. Seeds use
SplitMix64/Box-Muller noise, not PyTorch's RNG sequence. The same seed identifies tang's
stream only; cross-backend reproducibility requires matching noise and arithmetic.

`x-frog-progress: sse` (or `stream:true`) returns `progress` events `{step,of}` and a
terminal `result` event with the image response. Disconnecting cancels at the next
step boundary. Latent previews and cancellation inside VAE decode remain unfinished.

The DiT follows the basic single-image path: RMSNorm/SwiGLU blocks, scale/tanh-gate
modulation, interleaved axis RoPE, image/caption padding and refiners, and joint
attention. Weights are BF16 on GPU; norms remain F32. The VAE upcasts to F32 and uses
bounded tiled im2col/GEMM convolutions. Metal fuses DiT RMS normalization with channel
scale or gated residual, keeping modulation on device. Bounded command batches limit
temporary retention. `TANG_IMAGE_UNFUSED=1` preserves the earlier path for comparisons.
Packing, final-layer modulation and VAE upsampling/convolution remain unfused.
Broad quality and performance evaluation remains unfinished.

## Trained-weight smoke test

The pinned converted checkpoint passed real Frog tool approval, all eight streamed steps,
PNG delivery, workspace asset saving and actual generation metadata on a 48 GiB Apple
Silicon Mac. A frog photograph at seed 42 produced a recognizable subject at 320×192
and 512×512; the native Frog transcript and lightbox were inspected. Repeating 320×192
with the resident pipeline produced a byte-identical PNG. First request: 52.3 seconds
including model loading; resident repeat: 11.1 seconds; 512×512: 52.2 seconds. These are
individual eight-step end-to-end measurements, not general benchmark or quality claims.
These image tests use fixture chat decisions. An additional real Frog Screenshot/Compare
test used WebKit and a resident local Gemma 3 4B vision judge in the combined service;
the matched stop persisted across Frog restart. The judge returned structured differences
but also invented missing letters, so this establishes integration rather than judge quality.
The generator and judge remained resident together without evicting existing workloads.
Additional 128×128 validation exposed a Metal softmax shared-scratch race in VAE attention.
The barrier fix passes real Frog generation with diagnostics disabled and all 80 Metal
compute regressions. `TANG_IMAGE_TRACE=1` optionally logs synchronized VAE finite-value
counts for diagnostics; it is off by default. Nonfinite sampler latents fail at their step.

Three resident 320×192 runs with fused modulation took 8.679, 8.739 and 8.746 seconds;
the earlier path took 13.786, 12.913 and 13.242 seconds in the same session. Fused
repeats produced byte-identical PNGs. Relative to the earlier implementation, changed
pixels differed by at most one channel value, mean absolute channel error below 0.0011.
One resident 512×512 request took 37.8 seconds. Reproducibility is scoped to a backend
implementation; checkpoint hashes do not identify floating-point kernel changes.

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
stricter 1e-4 bound. At 32×32 trained latents, twelve Metal boundaries still exceed the
same strict bound; final output max error is 1.10e-4 (relative L2 5.93e-6).

An independent fp32 audit of PyTorch MATH versus CPU flash attention on the same trained
8×8 case also exceeds the old pointwise bound at ten late boundaries, worst max error
0.0078125 and final-output max error 3.16e-5 (relative L2 5.90e-6). This establishes that
the strict pointwise failure alone does not isolate a porting error. Fused Metal retains
nine late failures under that bound, final-output max error 5.45e-5 (relative L2 6.02e-6);
the deterministic random-head128 cases still pass all boundaries below 8.3e-6. The old
strict checks remain visible; they have not been silently relaxed.

`scripts/download_gemma_judge.py <new-directory>` explicitly downloads a pinned public
Gemma 3 4B MLX checkpoint, verifies hashes and preserves the same disk reserve. Language
weights are Q4 and vision weights BF16. No judge weights download during inference.

For constrained disk, `scripts/download_z_image.py <new-directory>` pins a public
Hugging Face revision, streams F32 matrices into BF16, preserves vectors, verifies
original LFS hashes, records converted hashes and retains 4 GiB disk headroom. It
writes about 20.5 GB instead of storing the roughly 32.8 GB original checkpoint.
Completed files can be resumed after size and SHA256 verification against the conversion
manifest; interrupted partial files are refused. The converted files are a distinct artifact; do not insert them into the official
repository's cache as if their hashes matched the originals.
