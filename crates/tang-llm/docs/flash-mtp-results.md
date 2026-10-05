# MTP takeover measurements (2026-10-05)

## Merged-head baseline

The source at PR head `7e214e07365770d72bd7515d0b53af7fec4cd06d` was built on mew.
All 52 original library tests passed. `flash-tcheck` found zero differing logits among
248,320 entries, with every probed layer bitwise equal for t1 vs t8 and t8-verify.
`flash-spec-test` passed greedy and sampled output equality including forced-wrong drafts.
Merged target parity against `~/flash-truth/hc/q8_code.jsonl` covered 318 positions: top-1
agreement 315/318, mean KL 0.0017, median KL 0.00004 (p99 0.0346, max 0.0640).
The quantized engine is not bitwise identical to the f32 reference; speculative exactness
means identical output to the same target engine with drafts disabled.

| prompt | thinking | measured tok/s | tokens/window | handoff best tok/s |
|---|---|---:|---:|---:|
| code | off | 235.0 | 4.88 | 232.4 |
| chat | off | 199.6 | 3.07 | 198.8 |
| code | on | 171.5 | 2.76 | 173.0 |
| chat | on | 163.3 | 2.57 | 162.5 |

These are single runs with the handoff configurations and 512-token request limits (507/508
tokens in the printed decode statistics), not new estimates of peak throughput.
Logs: `mew:~/flash-codex/{tcheck,spec,baseline-*-*}.log`. The historical table above is retained
for comparison. GPU runs took the shared lock; the data collector was left running.

## Trainer validation

Build with `cuda,mtp-train`; see [flash-mtp-train.md](flash-mtp-train.md). The added trainer's
nine tests pass (61 library tests total, one pre-existing ignored test). The complete cargo test
command now compiles after repairing the stale Request initializer in `tests/speculate.rs`.
The dense-model GPU integration tests skip without `TANG_LLM_TEST_MODEL`; the Flash CLI
exactness checks above were run on the actual Flash model.

The differentiable forward was compared with the existing pure-Rust reference using original
Q8 experts and the main embedding/head. All checked top tokens match. Relative RMS error in
recursive output residuals was 0.000013, 0.000015, 0.000015 at depths 1–3 (max absolute errors
0.000098, 0.000136, 0.000093). Log: `~/flash-codex/oracle.log`.

The two-update smoke run used 57 training / 7 held-out completed prompts, eight-row windows,
two-row burn-in, full vocabulary, lr 1e-5, beta .8 and global gradient clip 1. Held-out CE fell
from [1.8665, 3.2222, 4.4704] to [1.7529, 3.1040, 4.2851]. Depth-one teacher-token accuracy rose
27/35 → 29/35. Updates took 5.2–5.9 seconds. Its exported GGUF loaded in the engine and
passed greedy/sampled speculative exactness (`~/flash-codex/smoke-spec.log`).

## 50-update pilot

The launch snapshot contains p000–p065: 40 code-writing, 20 debugging/CS, and the first six
math prompts. The later planning and general-chat prompts were not yet collected.
The pilot uses 59 training / 7 held-out prompts, sixteen-row windows, four-row burn-in, full
vocabulary, lr 1e-5, beta .8 and global gradient clip 1. The active dense cell has 88,929,792
trainable parameters. Routed experts (frozen Q4), main embedding/head, and unused indexer
weights stay frozen. Original model files and serving configuration are unchanged.

| step | depth-1 CE | depth-2 CE | depth-3 CE | teacher-token hits (d1/d2/d3) |
|---:|---:|---:|---:|---|
| 0 | 1.4146 | 2.9846 | 3.1639 | 62/77, 41/70, 34/63 |
| 25 | 0.6087 | 1.5106 | 1.8167 | 67/77, 48/70, 39/63 |
| 50 | 0.5342 | 1.3898 | 1.6060 | 67/77, 50/70, 43/63 |

These are fixed middle windows from seven held-out prompts. Teacher-token recursive accuracy
is a diagnostic, not speculative acceptance. Truncated windows omit earlier K/V context;
quantized exported weights and actual greedy recursive drafts must be measured in the engine.

Artifacts: `mew:~/flash-codex/pilot50/step-NNNNNN/{mtp.gguf,state.bin,state.json}`;
manifest and full training metrics are in the parent directory. New collector additions do not
change the pilot's recorded prompt snapshot. The original MTP remains the serving default.

Checkpoint resume was tested on symlinks to the exact 66-prompt pilot snapshot. Restored step-25
validation JSON is identical. The next sampled window (p048, offset 5222), per-depth losses,
gradient norm and teacher-token counts are identical to the uninterrupted step 26. The resumed
checkpoint exported successfully (`~/flash-codex/resume25/step-000026/`).

## Exported checkpoint diagnostics

Evaluation-only runs (`--steps 0`) use the same frozen 66-prompt snapshot, sixteen-row windows,
four-row burn-in, and full vocabulary. No updates or checkpoint writes occur.

| step | f32 CE (d1/d2/d3) | reloaded GGUF CE (d1/d2/d3) | reloaded hits (d1/d2/d3) |
|---:|---|---|---|
| 25 | 0.6087 / 1.5106 / 1.8167 | 0.8654 / 1.9110 / 2.1872 | 66/77, 43/70, 38/63 |
| 50 | 0.5342 / 1.3898 / 1.6060 | 0.6127 / 1.6814 / 1.8396 | 69/77, 47/70, 39/63 |

Quantization erodes part of the short-window loss gain. These reloaded diagnostics still use
f32 activations; they do not emulate every inference int8 activation rounding operation.

## Deployed-engine results and selection

Both step-25 and step-50 exported GGUFs pass greedy and sampled speculative output exactness,
including forced-wrong drafts. The evaluation scripts completed with exit code zero.
Configurations match the original table: hybrid five-step drafts for code/off, MTP three-step
for the other cells; 4K context, 512-token limits, greedy. These are single-run measurements.

| prompt | thinking | original tok/s | step 25 tok/s | step 50 tok/s | tokens/window (original / 25 / 50) |
|---|---|---:|---:|---:|---|
| code | off | 235.0 | 231.4 | 228.3 | 4.88 / 4.76 / 4.85 |
| chat | off | 199.6 | 201.9 | 192.4 | 3.07 / 3.11 / 2.96 |
| code | on | 171.5 | 172.2 | 171.1 | 2.76 / 2.82 / 2.82 |
| chat | on | 163.3 | 162.1 | 158.3 | 2.57 / 2.55 / 2.49 |

Thinking-on accepted/tried draft-position counts (these denominators depend on window widths;
read together with tokens/window, not as independent unbiased accuracy estimates):

| prompt | checkpoint | d1 | d2 | d3 |
|---|---|---|---|---|
| code | original | 145/184 | 100/163 | 79/157 |
| code | step 25 | 146/180 | 103/173 | 79/163 |
| code | step 50 | 146/180 | 107/178 | 75/172 |
| chat | original | 153/198 | 96/197 | 61/195 |
| chat | step 25 | 149/199 | 95/198 | 65/198 |
| chat | step 50 | 146/204 | 95/203 | 63/201 |

**Do not promote either pilot checkpoint.** The original drafter remains the serving default.
The short-window held-out CE improvement is real, and much of it survives export, but neither
checkpoint establishes an overall deployment speed improvement. Step 50 also reduces chat/off
and chat/on tokens/window, so its slower chat measurements are not only a timing fluctuation.

Next controlled experiments should address training/inference mismatch: retain realistic MTP
prefix K/V context, train against the actual quantized activation/weight path (straight-through
quantization or another explicitly tested surrogate), and include the planning/general-chat
part of the corpus when collection reaches it. The current measurements do not isolate which
of these factors dominates. Repeat the fixed four-cell table and exactness checks before
selecting another checkpoint. The other handoff priorities (served wide prefill, tensor-core
dense kernels, host expert reads) remain open.

Raw logs: `mew:~/flash-codex/eval25/` and `eval50/`; each contains `spec.log`, four benchmark
logs, `exit-code`, and `done`. Reload diagnostics: `reload25/` and `reload50/`. All GPU runs
used `~/tang-gpu.lock`; the original collection script, model files, and server settings remain
untouched. The conflicts with main in `cuda.rs` and `server.rs` were subsequently resolved by integrating
main's node API/priority worker with the Flash backend and retaining the shared BF16 pool.
The integrated head passes 78 library tests (one pre-existing ignored) and both CPU node API
tests; CUDA model-dependent integration tests compile but skip without `TANG_LLM_TEST_MODEL`.
Flash keeps worker-local prompt rendering and a fixed model bundle: model load/unload is
rejected, its background prefill does not yield, and dense KV block reporting is unavailable.
