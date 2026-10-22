# GDN chunked prefill v2 (`TANG_CODER_GDN=chunked2`)

Status: implemented; CPU reference of the 3-phase decomposition unit-tested; `cargo check -p tang-llm
--features cuda --tests` passes. The CUDA kernels are NVRTC source compiled at engine load, so they
have **not been compiled or run** yet (written on a Mac with no nvcc). Default path and `chunked`
(v1) unchanged.

## Why v2

H200, cold wide prefill, 256-row windows:

| config | 8k | 30k | 54k |
| --- | --- | --- | --- |
| baseline | 6.95 | 36.9 | 87.5 s |
| fa2 attention | 5.91 | 22.3 | 40.7 s |
| chunked GDN v1 | 7.37 | 38.5 | 90.3 s |

v1 (`fl_gdn_chunk_step_seg`) is still one block per v head (48 blocks on 132 SMs), CUDA cores, chunk
16, serial over chunks inside the block; it raised the conv/step stage from 5.5 % to 14.8 % of GPU
time. v2 is flash-linear-attention's structure (`chunk_gated_delta_rule`: `chunk_fwd_h` /
`recompute_w_u` / `chunk_fwd_o`, MIT): most of the work parallel over (head x chunk), only a thin
state pass sequential, tensor cores throughout.

## Design

Per window (one sequence segment, `t >= 64` rows; anything else falls back to the existing kernels),
chunk `C = 64`, per v head, state `S` `[128 key][128 value]`, in-chunk cumulative log decay `lg`,
`b = sigmoid(beta)`:

| phase | kernel | grid / block / smem | work |
| --- | --- | --- | --- |
| A | `fl_gdn_c2_prep` | (48, t/64) / 256 / 150 KB | conv1d + SiLU + q/k L2 norm (same math as `gdn_step_body`), `lg`, `b`; `KK^T`, `QK^T` on mma (4x8 tiles each); `Akk = strict-lower(b_i e^{lg_i-lg_s} k_i.k_s)`, `Aqk = lower(e^{lg_i-lg_s} q_i.k_s)`; `T = (I + Akk)^-1` by f32 forward substitution (64 rows, 4 threads a column); `W = T diag(b e^lg) K`, `U = T diag(b) V` on mma |
| B | `fl_gdn_c2_state` | (48, 4) / 128 / 102 KB | sequential over chunks; each block owns a 128 x 32 value slice of `S` in shared memory (value columns are independent). Per chunk: store `S_c`; `D = U - W S_c` (mma); `S <- e^{lg_last} S + (e^{lg_last-lg} K)^T D` (mma, f32 accumulator initialized from the decayed state) |
| C | `fl_gdn_c2_out` | (48, t/64) / 256 / 149 KB | `O = diag(e^lg) Q S_c + Aqk D` (mma, both into one accumulator), then the gated RMS norm and `Y` exactly as the per-token kernel |

MMA: `mma.sync.m16n8k16.row.col.f32.bf16.bf16.f32`; operands are f32 in shared memory rounded to
bf16 when the fragments are built (`gc2_mma`, generic strides so transposes are free); all
accumulation, the state, the gates and the triangular solve are f32. This is fla's precision
(bf16 dot operands, f32 state). Padded rows of a short last chunk have `q = k = v = 0`, `b = 0`,
`lg` = the last real one, so they contribute nothing.

Scratch (`C2` in `coder.rs`, allocated in `enable_wide` only under `chunked2`), f32 per (v head,
chunk): q, k, W, U, D `[64][128]`, `Aqk [64][64]`, `lg [64]`, `S_c [128][128]` = 246 KB. Shared by
all GDN layers of a window.

Unchanged semantics: conv inputs from `[hist | P]` (history read-only), the conv history commit is
still `fl_gdn_conv_commit_seg`, the final state is written to `Stab[slot]` in the same
`[key][head][value]` layout only when `commit`, outputs only for the kept rows. Not bitwise the
per-token kernel (reassociation plus bf16 operands); decode continues from the handed-off state.

Profile stages: `gdn_c2_prep`, `gdn_c2_state`, `gdn_c2_out` (v1 and default stay `gdn_conv_step`).

Shared memory over 48 KB is opted into via `set_smem` (cap 200 KB); 150 KB a block needs sm_90 (on
99 KB parts the launch fails; `chunked2` is H200-only for now).

## CPU reference

`crates/tang-llm/src/flash/gdn_chunked.rs`: `chunk_prep` (phase A for one chunk: `lg`, `Aqk`, the
explicit `T = (I + Akk)^-1`, `W`, `U`) and `three_phase` (A for all chunks, then B sequential, then
C). Tests (`cargo test -p tang-llm --lib gdn_chunked`): every existing `check` case also runs
`three_phase` against `recurrent` (outputs and final state within 2e-4, d 16/128, chunks 4..64, short
tails, weak/strong decay); new `three_phase_matches_recurrent_d128_c64` (d 128, chunk 64, 213 and
150 tokens) and `decode_continues_from_three_phase_state`. The reference is f32; it does not model
the kernels' bf16 operand rounding.

## Expected speedup (honest)

With fa2 on, the per-token step is 5.5 % of GPU time (~1.2 s of 22.3 s at 30k, ~2.2 s of 40.7 s at
54k). The bulk of GDN time is the `gdn_in` / `gdn_out` GEMMs, which v2 does not touch. Rough v2 cost
at 30k (36 GDN layers, 48 heads, ~470 chunks): ~9 TFLOP of mma (well under 0.1 s even at a modest
fraction of peak), ~0.5 GB/layer-window-set of scratch traffic (~0.15-0.2 s total), and the
sequential phase B at ~4 dependent chunk steps a 256-row window (latency bound, ~0.1-0.15 s). So the
stage should drop from ~1.2 s to ~0.3-0.4 s:

| | 8k | 30k | 54k |
| --- | --- | --- | --- |
| fa2 (measured) | 5.91 | 22.3 | 40.7 s |
| fa2 + chunked2, guess | ~5.7 | ~21.4 | ~38.9 s |

i.e. ~4-5 % end to end; the win that matters is that the GDN step stops being a cost when attention
and the GEMMs get faster. If `gdn_c2_state` dominates the three, the next step is fla's
`chunk_fwd_h` trick of splitting the key dim too, or larger windows.

## Wider windows (`TANG_CODER_WIDE`)

`enable_wide` already accepts 16..=4096 rows. At 256 rows phase A/C get 4 chunks x 48 heads = 192
blocks (~1.5 per SM); at 1024, 768 blocks; at 2048, 1536. Phase B's sequential length grows with
the window but its count of windows shrinks, so its total is unchanged; the per-window launch and
sync overhead and the GEMMs' efficiency improve.

Memory (estimates, Coder-Next shapes: hidden 2048, 48 v heads, top-10 of 512 experts, ff 512):
the window scratch (`Scratch` + shared projection + int8 staging) is ~0.3 MB a row on device plus
~8 KB a row pinned host; `chunked2` adds ~184 KB a row.

| WIDE | window scratch | + chunked2 | total |
| --- | --- | --- | --- |
| 256 | ~80 MB | ~47 MB | ~127 MB |
| 1024 | ~320 MB | ~189 MB | ~0.5 GB |
| 2048 | ~640 MB | ~378 MB | ~1 GB |

Small against an H200's 141 GB, but the attention partials and MoE grouped kernels have only been
run at 256 rows; each new width also compiles its own module (~1 min).

## H200 bench

FA2 attention lives on `claude/fa2-default`; bench a branch that has both (e.g. merge
`claude/gdn-chunked-v2` into it). Build:

```
cargo build --release -p tang-llm --no-default-features --features cuda
```

Gate + timing (each line prints `wide-check N: PASS ...` and the `wide ... warm` times):

```
M=<Qwen3-Coder-Next first GGUF shard>; PR=<prompt.txt>
for W in 256 1024; do
  for G in default chunked2; do
    echo "== WIDE $W GDN $G"
    TANG_CODER_ATTN=fa2 TANG_CODER_GDN=$G TANG_CODER_WIDE=$W TANG_WIDE_CTXS=8000,30000,54000 \
      ./target/release/tang-llm flash-wide-check "$M" -n 64 --prompt-file "$PR"
  done
done
```

Stage split (eager windows, slower; compare `gdn_conv_step` against `gdn_c2_prep` + `gdn_c2_state`
+ `gdn_c2_out`):

```
for G in default chunked2; do
  TANG_CODER_PROF=1 TANG_CODER_ATTN=fa2 TANG_CODER_GDN=$G TANG_CODER_WIDE=256 TANG_WIDE_CTXS=30000 \
    ./target/release/tang-llm flash-wide-check "$M" -n 64 --prompt-file "$PR" --eager
done
```

Pass bar: greedy-64 equal and KL <= 1e-3 at every length. If KL fails, first suspect the kernels
(never run), then bf16 operand rounding (switch `gc2_mma` to tf32 m16n8k8 to check), not the math:
the CPU three-phase reference agrees with the recurrence.

## Credit

Algorithm and three-phase layout from
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention) (MIT, Songlin Yang,
Yu Zhang et al.); Yang, Kautz, Hatamizadeh, *Gated Delta Networks* (arXiv 2412.06464); Yang et
al., *Parallelizing Linear Transformers with the Delta Rule over Sequence Length* (arXiv
2406.06484). No fla code copied.
