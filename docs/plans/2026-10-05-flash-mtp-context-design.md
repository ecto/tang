# Matched MTP context experiment

The completed 200-prompt corpus has 902,644 residual rows, with aligned ids and contiguous
positions/tokens. Freeze a private copy and hash every residual/id/prompt file. Preserve the
pNNN % 10 == 0 split (20 held-out prompts, including the previous seven).

Compare 50-update pilots from the original GGUF, seed 42, seq 16, burn 4, lr 1e-5, beta .8,
clip 1, full vocabulary; use identical prompt/window draws and checkpoint at steps 25/50.
The control retains short-window attention. The treatment supplies all earlier teacher K/V,
including prompt positions and earlier generated positions. Prefix K/V is recomputed with
current weights and stays differentiable; recursive residual/K/V also stays attached.
Only K/V-producing sublayers run for prefix rows, avoiding routed experts and full logits.

Reconstruct missing prompt residuals in a separate sidecar directory with one loaded target
engine. Verify prompt positions/tokens and the overlapping last-prompt residual against the
frozen original dump. Never modify original data, models, or deployed server configuration.
Full-prefix cost/memory is measured in a smoke run before starting the two matched pilots.
If it cannot fit, report that limitation before substituting bounded context.

Use fixed early/middle/late windows on each held-out prompt for diagnostics, with the same
panel in both arms. Evaluate original and quantized exports; held-out CE/teacher hits are
not speculative acceptance. Select using repeated four-cell serving benchmarks (code/chat,
thinking on/off) plus exactness, reporting per-depth acceptance and tokens/window. Keep the
original drafter unless repeated serving measurements show a gain without exactness loss.

The unchanged f32 training activation contract still differs from int8 inference. This
experiment isolates context and broader data; it does not claim to eliminate quantization
mismatch or to fine-tune routed experts.
