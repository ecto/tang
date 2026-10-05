# Calibrated image step previews

The image roadmap calls for cheap latent-to-RGB previews during denoising. Use an affine projection calibrated against the installed VAE: sixteen latent channels and an intercept predict the average decoded RGB of each 8×8 pixel cell. The projection is an approximate color/silhouette preview; the full VAE remains authoritative for final images. Full VAE decoding at every step has substantial cost; a learned tiny decoder introduces another checkpoint. Calibration avoids borrowing projection coefficients from an unrelated implementation.

The explicit calibration command loads only the VAE, decode deterministic training and held-out latents, fit a small ridge regression in f64, and save versioned coefficients with the VAE fingerprint, normalization and validation metrics. No inference-time download or fitting occurs. A missing calibration file leaves existing progress available. Invalid or mismatched calibration is an explicit error when previews are requested.

Streaming progress includes an optional bounded PNG preview, step/total and image index. The preview uses diffusion latent coordinates (before VAE normalization); calibration incorporates the VAE scale/shift. At most 128×128 RGB pixels are encoded. Non-streaming callers incur no preview work. Final image bytes, seeds, scheduling and checkpoint identity must remain unchanged. Cancellation is checked at step boundaries and before preview emission.

Frog will validate preview bytes/dimensions, publish a typed tool-image delta and show only the latest preview while generation runs. Deltas remain replayable, but intermediate images are not appended to model vision context. Final completion replaces the temporary preview with final image attachments. Numeric progress remains compatible with older backends.

Validation includes held-out projection error against downsampled VAE output, a real generated image inspected beside its final preview, streamed frame ordering/bounds/cancellation, byte-identical final generation with previews on/off, durable native replay and a timing check that projection/PNG encoding adds negligible work relative to a denoising step. This design does not claim the optional under-250ms distilled generation milestone.

Run calibration once and install its output at the pipeline root:

```sh
tang-llm calibrate-image-preview /path/to/pipeline/vae /path/to/new-preview.json --device metal
cp /path/to/new-preview.json /path/to/pipeline/preview.json
```

Restart the image server after installation. `/v1/models` advertises `latent_previews` when this file is present; the resident pipeline validates its fingerprint and normalization before using it. Request `"preview": true` with SSE streaming. Progress events carry `preview_b64` (PNG), `preview_kind: "approximate_latent"`, `step`, `of` and zero-based `image_index`. The initial event has no preview; subsequent events describe the updated latent. Missing calibration and non-streaming preview requests return HTTP 400 before queueing. Ordinary generation remains available when an installed calibration is invalid.

A trained Z-Image VAE calibration used training seeds 11, 23, 37 and 53, held-out seeds 71 and 89, and amplitudes 0.25, 0.5, 1 and 2. Held-out cell RGB MSE was 0.0132503, versus 0.0717956 for the constant training-color baseline. This is a calibration measurement, not a semantic image-quality score.

Real 256×256, seed-42, eight-step generation emitted eight 32×32 PNG previews in step order. The final preview retained the frog silhouette and broad colors when inspected against the final PNG. Preview-enabled and preview-disabled final PNGs were byte-identical, including the preceding native SiLU optimization. A resident pair took 5.742 s with previews and 5.882 s without; one pair establishes no measurable slowdown in that run, not a general performance bound. The initial cold request took 63.757 s, including weight loading.
