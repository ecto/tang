# Offline recursive MTP training

Build on mew:

```sh
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_ROOT=/usr/local/cuda LD_LIBRARY_PATH=/usr/local/cuda/lib64
CARGO_TARGET_DIR=target4 cargo build --release -p tang-llm --features cuda,mtp-train
```

Check the differentiable cell against the existing pure-Rust oracle first:

```sh
flock ~/tang-gpu.lock target4/release/tang-llm flash-mtp-train "$MAIN" "$MTP" \
  --data ~/mtp-train --out ~/flash-codex/oracle --check-forward
```

`$MAIN` is the target model's first GGUF shard, `$MTP` the original MTP GGUF. The oracle mode
uses full-vocabulary, original Q8 experts, compares all three recursive residuals and argmaxes,
and creates no output directory. Training uses frozen Q4 expert weights matching default serving.

Start with a small smoke run in a new directory:

```sh
flock ~/tang-gpu.lock target4/release/tang-llm flash-mtp-train "$MAIN" "$MTP" \
  --data ~/mtp-train --out ~/flash-codex/smoke --steps 2 --seq 8 --burn 2 --eval-every 2
```

A longer experiment can use `--steps 100 --seq 32 --burn 8 --lr 1e-5 --beta .8 --clip 1`.
Every `--eval-every` steps, and at the end, `step-NNNNNN/` contains `mtp.gguf` and
`state.bin` / `state.json`. Continue into a **new** output directory with
`--resume /path/to/old/step-NNNNNN`. Supply the same original `$MAIN` and `$MTP`.
`--steps` is the number of additional updates, not the final global step number.

The manifest records the completed prompts available at launch. `pNNN` where NNN % 10 == 0
are held out. Each held-out prompt uses a fixed middle window; metrics are diagnostic three-depth
teacher-token accuracy, **not** measured speculative acceptance. Windows retain absolute RoPE
positions but omit earlier K/V context. The burn-in prefix reduces, rather than eliminates, that
approximation. Use the real engine to select a checkpoint.

Dense block tensors, including HC, router and shared expert, train. Routed experts and the main
embedding/head stay frozen. There is no routed-expert optimizer state. Default loss covers the
full vocabulary; `--vocab 106000` mirrors the drafter's restricted head but rejects any excluded
training target. Keep `TANG_FLASH_MTP_VOCAB=0` when comparing full-vocabulary diagnostics in the
engine; also test its default serving vocabulary separately.

Exports preserve metadata, offsets, tensor names and types, and untouched payload bytes. The
original is copied and same-sized payloads patched; values are requantized to their original
Q8/BF16/F32 type. Inference remains opt-in via `--mtp /path/to/step-NNNNNN/mtp.gguf`.
Run `flash-spec-test` plus all four `flash-bench` cells with that file. Report acceptance by
position and cumulative rates with their denominators, tokens/window, and decode speed. A lower
held-out CE alone is insufficient evidence to replace the original drafter.

For the same exactness and four-cell measurement sequence used by the handoff:

```sh
scripts/flash-mtp-evaluate.sh target4/release/tang-llm "$MAIN" \
  /path/to/step-NNNNNN/mtp.gguf /path/to/new-evaluation-directory
```

Every subprocess takes the GPU lock. `done` plus `exit-code` records completion (zero means
all subprocesses succeeded and speculative exactness printed PASS). The unfiltered logs retain
per-position acceptance, all engine timings, configuration and generated tokens. The script
inherits optional engine environment settings; keep those identical between checkpoints.
