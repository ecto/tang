# GDN chunked prefill (`TANG_CODER_GDN=chunked`)

Status: implemented, CPU reference unit-tested, CUDA kernel **not yet compiled or run** (written
on a Mac with no nvcc). Default path unchanged.

## What runs today

Every GDN layer's mixer in a window is `gdn_in` GEMM -> `fl_gdn_conv_step_seg` -> `fl_gdn_conv_commit_seg`
-> `gdn_out` GEMM (`crates/tang-llm/src/flash/coder.rs`, `enqueue_mixer`). `fl_gdn_conv_step_seg`
(`crates/tang-compute/src/kernels/flash_cuda.rs`, `gdn_step_body<true>`) is one 512-thread block per
v head (48 blocks), the 128x128 state in registers, and runs the delta rule **token by token**:
per token a decay, a `S^T k` reduction, the rank-1 update, a `S^T q` reduction and the gated RMS
norm, with ~6 block barriers. A 256-row prefill window is 256 sequential steps per head, on 48 of
the H200's 132 SMs.

## What `chunked` runs

`fl_gdn_chunk_step_seg`: a drop-in for `fl_gdn_conv_step_seg` (same arguments, grid, state layout,
conv fusion, commit/verify semantics). Picked in `gdn_step_seg` when `TANG_CODER_GDN=chunked` and
the window has at least 32 rows, i.e. wide prefill windows; decode and MTP verify windows (<= 8
rows) keep the per-token kernel either way, so narrow prefill + decode are untouched.

Per chunk of `C = GDN_CHUNK = 16` tokens, from the state `S0` at the chunk start, with
`lg_i = sum_{s<=i} ln g_s` (in-chunk cumulative log decay, `ln g = softplus(a + dt) * A <= 0`) and
`b_i = sigmoid(beta)`:

```
d_i = b_i (v_i - e^{lg_i} S0^T k_i) - b_i sum_{s<i} e^{lg_i - lg_s} (k_i.k_s) d_s   # UT / WY form
o_i = e^{lg_i} S0^T q_i + sum_{s<=i} e^{lg_i - lg_s} (q_i.k_s) d_s
S'  = e^{lg_last} S0 + sum_s e^{lg_last - lg_s} k_s d_s^T
```

i.e. `D = (I + A)^{-1} b (V - Gamma K S0)` with `A` strictly lower triangular. In the kernel: the
`K K^T` / `Q K^T` chunk Gram matrices are 2*C*C parallel dots; `K S0` and `Q S0` are one register-
state GEMM pass; the triangular solve is a C-step forward substitution per value column; the
outputs are a small `[C][C] x [C][128]` product plus the old gated RMS norm; the state pass is one
rank-C update. Gates are in log space so all intra-chunk ratios are `exp(<= 0)`.

The final state (and conv history, via the unchanged `fl_gdn_conv_commit_seg`) handed to decode is
the same quantity the per-token kernel writes, equal up to float reassociation (not bitwise).

CPU reference: `crates/tang-llm/src/flash/gdn_chunked.rs` (`recurrent` = today's math, `chunked` =
the above). `cargo test -p tang-llm --lib gdn_chunked` checks, on random L2-normed q/k, random
gates and betas: outputs and final state match within 2e-4 for d = 16/128, chunk 4/16/32/64,
lengths not a multiple of the chunk, weak and strong decay; and decoding token-by-token from the
chunked final state matches decoding from the recurrent one.

## Expected speedup (honest)

This first kernel is the correctness-first version: still one block per head per window, CUDA
cores, C = 16 (fits 48 KB static shared memory next to q/k/v/d tiles). Its win over today is the
per-token serial chain: the state is touched once per chunk instead of per token, and the per-token
`S^T k` reduction folds into a batched pass. Guess: 1.3-2x on the `gdn_conv_step` stage, which is
only worth it if that stage is a meaningful share of prefill; measure with `TANG_CODER_PROF=1`.

The fla-scale win (5-20x on the GDN part, per the survey) needs the next steps, which this layout
is set up for:

1. Split the kernel: an intra-chunk kernel over (chunk, head), fully parallel across chunks (W/U
   from the UT transform, `K K^T`, `Q K^T`), then a thin sequential state-passing kernel, then a
   parallel output kernel. Grid goes from 48 blocks to 48 * T/C.
2. C = 64 with tensor-core MMA (bf16/tf32) and dynamic shared memory (opt-in > 48 KB).
3. Prefill windows wider than 256 rows (`TANG_CODER_WIDE`) so the parallel parts have work.

## Credit

The chunkwise gated delta rule, the UT/WY representation and the log-space gating follow
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention) (`fla.ops.gated_delta_rule.chunk_gated_delta_rule`,
MIT licence, Songlin Yang, Yu Zhang et al.), and the papers: Yang, Kautz, Hatamizadeh, *Gated
Delta Networks* (arXiv 2412.06464); Yang et al., *Parallelizing Linear Transformers with the Delta
Rule over Sequence Length* (arXiv 2406.06484). No fla code is copied; the kernel is written for
tang's state layout.

## H200 bench + gate

Build:

```
cargo build --release -p tang-llm --no-default-features --features cuda
```

Gate and timing in one: `flash-wide-check` prefills narrow (8-row windows: per-token GDN kernel
regardless of the flag) and wide (256-row windows: chunked under the flag), then checks greedy-64
tokens equal and teacher-forced max KL(narrow || wide) <= 1e-3. It prints wide prefill times.

```
M=<path to the Qwen3-Coder-Next first GGUF shard>
# current (baseline)
TANG_WIDE_CTXS=8000,30000,54000 \
  ./target/release/tang-llm flash-wide-check "$M" -n 64 --prompt-file <prompt.txt>
# chunked
TANG_CODER_GDN=chunked TANG_WIDE_CTXS=8000,30000,54000 \
  ./target/release/tang-llm flash-wide-check "$M" -n 64 --prompt-file <prompt.txt>
```

Pass bar: every length prints `wide-check N: PASS (greedy equal, KL <= 1e-3)` (it will not be
bitwise) and the command exits 0. Compare the `wide ... warm` times against 7.05 / 36.8 / 87.2 s
For the per-stage split, rerun one length with `TANG_CODER_PROF=1 ... --eager` (profiling needs
eager windows and slows them) and compare the `gdn_conv_step` row. `flash-prefill-bench` with the same env gives the
serving-path number. If KL fails, first suspect the kernel (it has not run yet), not the math:
the CPU reference agrees with the recurrence.
